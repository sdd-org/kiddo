//! The core `KdTree` struct lives here, around which the whole crate is based.
//!
//! Alongside it are some ancillary structs:
//! * `ArchivedKdTree` is the type that a `KdTree` serialized with `rkyv` deserializes into when
//!   using `rkyv`'s full zero-copy deserialization mode. It implements `KdTreeAccessor` so that
//!   all queries that can be performed on a `KdTree` can also be performed on an `ArchivedKdTree`.
//! * `KdTreeIter` is the type returned by `KdTree::iter()`.
//! * `KdTreeResolver` is created by the `rkyv::Archive` derive macro. You can almost certainly
//!   ignore it.
//! * [`QueryBuilder`] is returned by `KdTree::query()` - **this is where you'll find documentation
//!   on how to build queries and all the query methods and their semantics.**
//! * `WithinUnsortedIter` is the lazy iterator returned by
//!   `QueryBuilder::within().unsorted().iter()` if you have not also called
//!   `QueryBuilder.with_points()`. This is a more memory-efficient iterator, avoiding materializing
//!   the full result set up-front, maintaining the traversal state, and lazily emitting results as
//!   soon as they are found.
//!
mod builder;
mod construction;
pub(crate) mod item_summary;
mod iter;
pub(crate) mod orchestrator;
pub(crate) mod query;
pub(crate) mod query_context;
pub(crate) mod query_stack;
pub(crate) mod query_stack_simd;
mod stem_leaf_resolution;

use std::marker::PhantomData;

use aligned_vec::{AVec, CACHELINE_ALIGN};
use nonmax::NonMaxUsize;

#[doc(hidden)]
pub use crate::traits::kd_tree::{KdTreeAccessor, StemLeafResolution};
pub use builder::KdTreeBuilder;
pub use construction::DefaultConstruction;
#[cfg(feature = "multi-threaded")]
#[doc(hidden)]
pub use construction::ParallelConstruction;
#[doc(hidden)]
pub use construction::SerialConstruction;
#[cfg(feature = "multi-threaded")]
pub use construction::DEFAULT_PARALLEL_CONSTRUCTION_THRESHOLD;
pub use iter::{KdTreeIter, WithinUnsortedIter};
#[doc(hidden)]
pub use orchestrator::KdTreeQueryOps;
pub use query::QueryBuilder;
#[doc(hidden)]
pub use query::{Exclude, Include, Projection};
pub use query_stack::QueryScratch;
#[doc(hidden)]
pub use stem_leaf_resolution::OwnedStemLeafResolution;

use crate::traits::leaf_strategy::{BucketLimitType, ConstructibleLeafStrategy, Mutability};
use crate::{Axis, Content, LeafStrategy, StemStrategy};

/// Encoded item-leaf mode for ordinary, unsorted leaves.
pub const ITEM_LEAF_MODE_UNSORTED: u8 = 0;

/// Encoded item-leaf mode for sorted leaves without stem-padding summaries.
pub const ITEM_LEAF_MODE_SORTED: u8 = u8::MAX;

/// The interpreted item-leaf mode carried by [`KdTree`]'s final const generic.
///
/// Rust does not currently support enums as const-generic parameter types, so
/// the tree stores this choice in its type as a `u8` and exposes this enum for
/// readable inspection. Codes `1..=254` represent sorted leaves whose stem
/// padding carries encoded subtree-minimum summaries; the code is the number
/// of bits by which minimum items are shifted right. Builders for a concrete
/// encoding may accept a narrower range appropriate to its item type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ItemLeafMode {
    /// Leaves retain their construction order.
    Unsorted,
    /// Leaves are sorted by ascending item without subtree-minimum summaries.
    SortedWithoutEncodedMin,
    /// Leaves are sorted and stem padding encodes subtree-minimum summaries.
    SortedWithEncodedMin {
        /// Power-of-two bucket shift used by the encoded summary.
        shift: u8,
    },
}

impl ItemLeafMode {
    /// Interprets the `u8` item-leaf mode used as a const generic.
    ///
    /// Note that codes `32..=253` are not producible by any builder (a summary
    /// shift must lie in `1..=31`) and query-side consumers treat them as
    /// disabled; this function nevertheless reports them as
    /// [`ItemLeafMode::SortedWithEncodedMin`] so that it remains total. Callers
    /// must not treat the reported `shift` as a valid encoding.
    #[inline]
    pub const fn from_code(code: u8) -> Self {
        match code {
            ITEM_LEAF_MODE_UNSORTED => Self::Unsorted,
            ITEM_LEAF_MODE_SORTED => Self::SortedWithoutEncodedMin,
            shift => Self::SortedWithEncodedMin { shift },
        }
    }

    /// Returns whether the mode guarantees ascending item order within leaves.
    #[inline]
    pub const fn has_sorted_leaves(self) -> bool {
        matches!(
            self,
            Self::SortedWithoutEncodedMin | Self::SortedWithEncodedMin { .. }
        )
    }

    /// Returns the encoded-minimum bucket shift, when enabled.
    #[inline]
    pub const fn encoded_min_shift(self) -> Option<u8> {
        match self {
            Self::SortedWithEncodedMin { shift } => Some(shift),
            _ => None,
        }
    }
}

/// Errors returned by kd-tree construction and mutation when caller-controlled
/// input or configuration cannot be accommodated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstructionError {
    /// Auto-generated item indices from `new_from_slice` do not fit in `T`.
    AutoGeneratedItemIndexOverflow {
        /// Number of source points passed to the constructor.
        item_count: usize,
        /// Human-readable type name of the generated item type.
        item_type: &'static str,
    },
    /// A full mutable bucket could not be split without violating bucket semantics.
    UnsplittableBucket {
        /// Split dimension selected when the failure occurred.
        split_dim: usize,
    },
}

impl std::fmt::Display for ConstructionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AutoGeneratedItemIndexOverflow {
                item_count,
                item_type,
            } => write!(
                f,
                "cannot auto-generate {item_count} item indices for item type {item_type}"
            ),
            Self::UnsplittableBucket { split_dim } => {
                write!(
                    f,
                    "cannot split leaf on dimension {split_dim} because all points have the same value on that dimension"
                )
            }
        }
    }
}

impl std::error::Error for ConstructionError {}

/// Errors returned by in-place owned-tree mutation operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationError {
    /// No entry exactly matched the requested point and item.
    EntryNotFound,
}

impl std::fmt::Display for MutationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EntryNotFound => write!(f, "entry not found"),
        }
    }
}

impl std::error::Error for MutationError {}

/// Errors returned when converting one `KdTree` variant into another.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KdTreeConversionError {
    /// Converting an item failed.
    ItemConversion {
        /// Index of the failing logical entry in iterator order.
        point_index: usize,
        /// Debug-formatted source conversion error.
        source: String,
    },
    /// Converting one coordinate in a point failed.
    AxisConversion {
        /// Index of the failing logical entry in iterator order.
        point_index: usize,
        /// Coordinate dimension that failed to convert.
        dim: usize,
        /// Debug-formatted source conversion error.
        source: String,
    },
    /// Rebuilding the destination tree failed.
    Construction(ConstructionError),
}

impl std::fmt::Display for KdTreeConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ItemConversion {
                point_index,
                source,
            } => write!(
                f,
                "failed to convert item at point_index {point_index}: {source}"
            ),
            Self::AxisConversion {
                point_index,
                dim,
                source,
            } => write!(
                f,
                "failed to convert axis value at point_index {point_index}, dim {dim}: {source}"
            ),
            Self::Construction(err) => write!(f, "failed to rebuild converted kd-tree: {err}"),
        }
    }
}

impl std::error::Error for KdTreeConversionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Construction(err) => Some(err),
            _ => None,
        }
    }
}

impl From<ConstructionError> for KdTreeConversionError {
    fn from(value: ConstructionError) -> Self {
        Self::Construction(value)
    }
}

/// Compile-time rejection of stem/leaf strategy pairings that cannot work.
///
/// A block-at-once stem strategy descends a whole layout block per step, so it
/// cannot observe a terminal stem that sits partway through a block. Immutable
/// trees resolve leaves arithmetically and are unaffected: every leaf sits at the
/// same depth. A mutable tree resolves through a map keyed on the terminal stem
/// index, and a block step can stride straight past one and land on an index the
/// map has no entry for.
///
/// Instantiating this for an incompatible pairing fails to compile, which is why
/// the check hangs off an associated constant rather than a runtime branch.
pub(crate) struct StemLeafCompatibility<A, T, SS, LS, const K: usize, const B: usize>(
    PhantomData<(A, T, SS, LS)>,
);

impl<A, T, SS, LS, const K: usize, const B: usize> StemLeafCompatibility<A, T, SS, LS, K, B>
where
    A: Axis<Coord = A>,
    T: Content,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
{
    pub(crate) const ASSERT_COMPATIBLE: () = assert!(
        !(SS::TRAVERSES_BLOCK_AT_ONCE && <LS::Mutability as Mutability>::IS_MUTABLE),
        "block-at-once stem strategies require an immutable leaf strategy: a block \
         step cannot observe a terminal stem partway through a block, which the \
         mapped leaf resolution of a mutable tree relies on"
    );
}

#[inline(always)]
fn resolve_arithmetic_terminal_stem_idx(
    stem_idx: usize,
    arithmetic_leaf_idx: usize,
    stems_depth: usize,
    leaf_count: usize,
) -> usize {
    if arithmetic_leaf_idx >= leaf_count {
        panic!(
            "arithmetic leaf resolution out of bounds: stem_idx={} arithmetic_leaf_idx={} leaf_count={} stems_depth={}",
            stem_idx, arithmetic_leaf_idx, leaf_count, stems_depth
        );
    }

    arithmetic_leaf_idx
}

#[inline(always)]
fn resolve_mapped_terminal_stem_idx(
    stem_idx: usize,
    min_stem_leaf_idx: usize,
    map_len: usize,
    mut get_map_entry: impl FnMut(usize) -> Option<usize>,
) -> usize {
    if stem_idx >= min_stem_leaf_idx {
        let map_idx = stem_idx - min_stem_leaf_idx;
        get_map_entry(map_idx).unwrap_or_else(|| {
            panic!(
                "mapped leaf resolution miss: stem_idx={} map_idx={} leaf_idx_map_len={}",
                stem_idx, map_idx, map_len
            )
        })
    } else {
        panic!(
            "mapped leaf resolution miss: stem_idx={} below min_stem_leaf_idx={}",
            stem_idx, min_stem_leaf_idx
        )
    }
}

/// A k-d tree for efficient spatial queries.
///
/// # Type Parameters
/// * `A`: [`Axis`] - coordinate type (e.g., `f32`, `f64`, or fixed-point types)
/// * `T`: [`Content`] - item type stored at each point. `u32` is the default choice and most common.
/// * `SS`: [`StemStrategy`] - determines what ordering scheme is used for stem nodes and what
///   approaches are used for prefetch, traversal, backtracking, and leaf node resolution.
/// * `LS`: [`LeafStrategy`] - determines how leaf nodes are stored.
/// * `K`: [`usize`] - Dimensionality (number of dimensions)
/// * `B`: [`usize`] - Bucket size (maximum items per leaf node - 32 is the recommended default)
/// * `ITEM_LEAF_MODE`: encoded [`ItemLeafMode`]; defaults to [`ITEM_LEAF_MODE_UNSORTED`]
#[cfg_attr(
    feature = "rkyv_08",
    derive(rkyv_08::Archive, rkyv_08::Serialize, rkyv_08::Deserialize)
)]
#[cfg_attr(feature = "rkyv_08", rkyv(crate = rkyv_08))]
#[cfg_attr(feature = "rkyv_08", rkyv(attr(allow(missing_docs))))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, PartialEq)]
pub struct KdTree<
    A,              // Axis
    T,              // Content,
    SS,             // StemStrategy
    LS,             // LeafStrategy
    const K: usize, // dimensionality
    const B: usize, // bucket size
    const ITEM_LEAF_MODE: u8 = ITEM_LEAF_MODE_UNSORTED,
> {
    #[cfg_attr(
        feature = "rkyv_08",
        rkyv(with = crate::rkyv::adapters::AsAlignedCachelineABox)
    )]
    stems: AVec<A>,
    leaves: LS,
    pub(crate) stem_leaf_resolution: OwnedStemLeafResolution,

    size: usize,
    max_stem_level: i32,
    pub(crate) max_leaf_len: usize,
    pub(crate) _phantom: std::marker::PhantomData<(SS, T)>,
}

impl<A, T, SS, LS, const K: usize, const B: usize, const ITEM_LEAF_MODE: u8> std::fmt::Debug
    for KdTree<A, T, SS, LS, K, B, ITEM_LEAF_MODE>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KdTree")
            .field("axis", &std::any::type_name::<A>())
            .field("item", &std::any::type_name::<T>())
            .field("stem_strategy", &std::any::type_name::<SS>())
            .field("leaf_strategy", &std::any::type_name::<LS>())
            .field("dimensions", &K)
            .field("bucket_size", &B)
            .field("item_leaf_mode", &ItemLeafMode::from_code(ITEM_LEAF_MODE))
            .field("size", &self.size)
            .field("stem_count", &self.stems.len())
            .field("max_stem_level", &self.max_stem_level)
            .field("max_leaf_len", &self.max_leaf_len)
            .finish_non_exhaustive()
    }
}

impl<A, T, SS, LS, const K: usize, const B: usize, const ITEM_LEAF_MODE: u8>
    KdTreeAccessor<A, T, SS, LS, K, B> for KdTree<A, T, SS, LS, K, B, ITEM_LEAF_MODE>
where
    A: Axis<Coord = A>,
    T: Content,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
{
    #[inline(always)]
    fn stems(&self) -> &[A] {
        self.stems.as_slice()
    }

    #[inline(always)]
    fn leaves(&self) -> &LS {
        &self.leaves
    }

    #[inline(always)]
    fn stem_leaf_resolution(&self) -> &impl StemLeafResolution {
        &self.stem_leaf_resolution
    }

    #[inline(always)]
    fn size(&self) -> usize {
        self.size
    }

    #[inline(always)]
    fn max_stem_level(&self) -> i32 {
        self.max_stem_level
    }

    #[inline(always)]
    fn max_leaf_len(&self) -> usize {
        self.max_leaf_len
    }

    #[inline(always)]
    fn item_leaf_mode_code(&self) -> u8 {
        ITEM_LEAF_MODE
    }
}

#[cfg(feature = "rkyv_08")]
impl<A, T, SS, LS, const K: usize, const B: usize, const ITEM_LEAF_MODE: u8>
    ArchivedKdTree<A, T, SS, LS, K, B, ITEM_LEAF_MODE>
where
    A: rkyv_08::Archive + Axis<Coord = A>,
    T: Content,
    SS: StemStrategy,
    LS: rkyv_08::Archive,
    rkyv_08::Archived<LS>: LeafStrategy<A, T, SS, K, B>,
{
    #[inline]
    pub(crate) fn archived_stems(&self) -> &[rkyv_08::Archived<A>] {
        self.stems.get().as_slice()
    }

    #[inline]
    /// Returns `true` if the archived tree contains no points.
    pub fn is_empty(&self) -> bool {
        self.size.to_native() as usize == 0
    }

    #[inline]
    /// Returns the number of points in the archived tree.
    pub fn size(&self) -> usize {
        self.size.to_native() as usize
    }

    #[inline]
    /// Returns the maximum stem level in the archived tree.
    pub fn max_stem_level(&self) -> i32 {
        self.max_stem_level.to_native()
    }

    #[inline]
    /// Returns the configured maximum leaf size heuristic used to size hot-path scratch buffers.
    pub fn max_leaf_len(&self) -> usize {
        self.max_leaf_len.to_native() as usize
    }

    /// Returns whether each leaf is ordered by ascending item value and can use
    /// the item-sorted `best_n_within` kernel.
    #[inline]
    pub fn item_sorted_leaves(&self) -> bool {
        self.item_leaf_mode().has_sorted_leaves()
    }

    /// Returns the item ordering and subtree-summary mode encoded in this tree's type.
    #[inline]
    pub const fn item_leaf_mode(&self) -> ItemLeafMode {
        ItemLeafMode::from_code(ITEM_LEAF_MODE)
    }

    #[inline]
    /// Returns the number of leaf nodes in the archived tree.
    pub fn leaf_count(&self) -> usize {
        self.leaves.leaf_count()
    }

    #[inline]
    /// Returns an iterator over all item/point pairs in the archived tree.
    pub fn iter(&self) -> KdTreeIter<'_, Self, A, T, SS, rkyv_08::Archived<LS>, K, B> {
        KdTreeIter::new(self)
    }
}

#[cfg(feature = "rkyv_08")]
impl<A, T, SS, LS, const K: usize, const B: usize, const ITEM_LEAF_MODE: u8>
    KdTreeAccessor<A, T, SS, rkyv_08::Archived<LS>, K, B>
    for ArchivedKdTree<A, T, SS, LS, K, B, ITEM_LEAF_MODE>
where
    A: rkyv_08::Archive + Axis<Coord = A>,
    T: Content,
    SS: StemStrategy,
    LS: rkyv_08::Archive,
    rkyv_08::Archived<LS>: LeafStrategy<A, T, SS, K, B>,
{
    #[inline(always)]
    fn stems(&self) -> &[A] {
        crate::rkyv::utils::transform_slice(self.archived_stems())
    }

    #[inline(always)]
    fn leaves(&self) -> &rkyv_08::Archived<LS> {
        &self.leaves
    }

    #[inline(always)]
    fn stem_leaf_resolution(&self) -> &impl StemLeafResolution {
        &self.stem_leaf_resolution
    }

    #[inline(always)]
    fn size(&self) -> usize {
        self.size.to_native() as usize
    }

    #[inline(always)]
    fn max_stem_level(&self) -> i32 {
        self.max_stem_level.to_native()
    }

    #[inline(always)]
    fn max_leaf_len(&self) -> usize {
        self.max_leaf_len.to_native() as usize
    }

    #[inline(always)]
    fn item_leaf_mode_code(&self) -> u8 {
        ITEM_LEAF_MODE
    }
}

impl<A, T, SS, LS, const K: usize, const B: usize> Default for KdTree<A, T, SS, LS, K, B>
where
    A: Axis<Coord = A>,
    T: Content,
    LS: ConstructibleLeafStrategy<A, T, SS, K, B>,
    SS: StemStrategy,
{
    fn default() -> Self {
        // Rejects a block-at-once stem strategy paired with a mutable leaf strategy.
        // Mutable trees are built from `default()` and grown with `add`, so this is
        // the earliest point the pairing can be caught.
        let () = StemLeafCompatibility::<A, T, SS, LS, K, B>::ASSERT_COMPATIBLE;

        // For mutable trees, initialize with sentinel stem at root
        let (stems, max_stem_level, stem_leaf_resolution) = if LS::Mutability::is_mutable() {
            // Get the root index for this stem strategy
            let root_idx = SS::new_no_ptr().stem_idx();

            // Create stems array with sentinel value at root
            let mut stems = AVec::new(CACHELINE_ALIGN);
            stems.resize(root_idx + 1, A::max_value());

            // Start in Mapped state - map root directly to the single initial leaf
            let mut leaf_idx_map = vec![None; root_idx + 1];
            leaf_idx_map[root_idx] = NonMaxUsize::new(0);

            let stem_leaf_resolution = crate::kd_tree::OwnedStemLeafResolution::Mapped {
                min_stem_leaf_idx: 0,
                leaf_idx_map,
            };

            (stems, 0, stem_leaf_resolution)
        } else {
            // Immutable trees start empty
            let stems = AVec::new(CACHELINE_ALIGN);
            let stem_leaf_resolution = OwnedStemLeafResolution::Arithmetic {
                stems_depth: 0,
                leaf_count: 0,
            };

            (stems, -1, stem_leaf_resolution)
        };

        let tree = Self {
            stems,
            leaves: LS::new_with_empty_leaf(),
            stem_leaf_resolution,
            size: 0,
            max_stem_level,
            max_leaf_len: Self::initial_max_leaf_len(),
            _phantom: std::marker::PhantomData,
        };
        tree.maybe_enable_huge_pages();
        tree
    }
}

impl<A, T, SS, LS, const K: usize, const B: usize, const ITEM_LEAF_MODE: u8>
    KdTree<A, T, SS, LS, K, B, ITEM_LEAF_MODE>
where
    A: Axis<Coord = A>,
    T: Content,
    LS: LeafStrategy<A, T, SS, K, B>,
    SS: StemStrategy,
{
    #[inline(always)]
    pub(crate) fn initial_max_leaf_len() -> usize {
        match LS::BUCKET_LIMIT_TYPE {
            // TODO: replace this heuristic with the actual observed maximum leaf size during
            // construction / deserialization if we keep this field.
            BucketLimitType::Hard => B,
            BucketLimitType::Soft => B * 2,
        }
    }

    #[inline]
    pub(crate) fn maybe_enable_huge_pages(&self) {
        crate::huge_pages::maybe_collapse_slice_huge_pages(self.stems.as_ptr(), self.stems.len());
        self.leaves.maybe_enable_huge_pages();
    }

    /// Returns `true` if the tree contains no points.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Returns the number of points in the tree.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the maximum stem level in the tree.
    #[inline]
    pub fn max_stem_level(&self) -> i32 {
        self.max_stem_level
    }

    /// Returns the number of leaf nodes in the tree.
    #[inline]
    pub fn leaf_count(&self) -> usize {
        self.leaves.leaf_count()
    }

    #[inline]
    /// Returns the configured maximum leaf size heuristic used to size hot-path scratch buffers.
    pub fn max_leaf_len(&self) -> usize {
        self.max_leaf_len
    }

    /// Returns whether each leaf is ordered by ascending item value and can use
    /// the item-sorted `best_n_within` kernel.
    #[inline]
    pub fn item_sorted_leaves(&self) -> bool {
        self.item_leaf_mode().has_sorted_leaves()
    }

    /// Returns the item ordering and subtree-summary mode encoded in this tree's type.
    #[inline]
    pub const fn item_leaf_mode(&self) -> ItemLeafMode {
        ItemLeafMode::from_code(ITEM_LEAF_MODE)
    }

    /// Returns an iterator over all item/point pairs in the tree.
    #[inline]
    pub fn iter(&self) -> KdTreeIter<'_, Self, A, T, SS, LS, K, B> {
        KdTreeIter::new(self)
    }

    /// Converts this tree into another `KdTree` variant by rebuilding from its
    /// logical entries.
    #[inline]
    pub fn try_convert<A2, T2, SS2, LS2, const B2: usize>(
        self,
    ) -> Result<KdTree<A2, T2, SS2, LS2, K, B2>, KdTreeConversionError>
    where
        A2: Axis<Coord = A2> + TryFrom<A> + Send + Sync,
        <A2 as TryFrom<A>>::Error: std::fmt::Debug,
        T2: Content + TryFrom<T>,
        <T2 as TryFrom<T>>::Error: std::fmt::Debug,
        SS2: StemStrategy,
        LS2: ConstructibleLeafStrategy<A2, T2, SS2, K, B2>,
    {
        KdTree::<A2, T2, SS2, LS2, K, B2>::try_from(&self)
    }

    /// Find which leaf contains a specific item.
    /// Returns `Some((leaf_idx, position_in_leaf))` if found, `None` if not found.
    pub fn find_leaf_for_item(&self, target_item: T) -> Option<(usize, usize)>
    where
        T: PartialEq,
    {
        for leaf_idx in 0..self.leaves.leaf_count() {
            let leaf_view = self.leaves.leaf_view(leaf_idx);
            let (_points, items) = leaf_view.into_parts();

            for (pos_in_leaf, item) in items.iter().enumerate() {
                if *item == target_item {
                    return Some((leaf_idx, pos_in_leaf));
                }
            }
        }
        None
    }
}

impl<A, T, SS, LS, const K: usize, const B: usize> FromIterator<(usize, [A; K])>
    for KdTree<A, T, SS, LS, K, B>
where
    A: Axis<Coord = A>,
    T: Content,
    LS: ConstructibleLeafStrategy<A, T, SS, K, B> + Default,
    SS: StemStrategy,
{
    fn from_iter<I: IntoIterator<Item = (usize, [A; K])>>(_iter: I) -> Self {
        // TODO: Proper impl
        Self::default()
    }
}

impl<
        'a,
        A1,
        T1,
        SS1,
        LS1,
        A2,
        T2,
        SS2,
        LS2,
        const K: usize,
        const B1: usize,
        const B2: usize,
        const ITEM_LEAF_MODE: u8,
    > TryFrom<&'a KdTree<A1, T1, SS1, LS1, K, B1, ITEM_LEAF_MODE>>
    for KdTree<A2, T2, SS2, LS2, K, B2>
where
    A1: Axis<Coord = A1>,
    T1: Content,
    SS1: StemStrategy,
    LS1: LeafStrategy<A1, T1, SS1, K, B1>,
    A2: Axis<Coord = A2> + TryFrom<A1> + Send + Sync,
    <A2 as TryFrom<A1>>::Error: std::fmt::Debug,
    T2: Content + TryFrom<T1>,
    <T2 as TryFrom<T1>>::Error: std::fmt::Debug,
    SS2: StemStrategy,
    LS2: ConstructibleLeafStrategy<A2, T2, SS2, K, B2>,
{
    type Error = KdTreeConversionError;

    fn try_from(
        source: &'a KdTree<A1, T1, SS1, LS1, K, B1, ITEM_LEAF_MODE>,
    ) -> Result<Self, Self::Error> {
        let mut entries = Vec::with_capacity(source.size());

        for (point_index, (item, point)) in source.iter().enumerate() {
            let converted_item =
                T2::try_from(item).map_err(|err| KdTreeConversionError::ItemConversion {
                    point_index,
                    source: format!("{err:?}"),
                })?;

            let mut converted_point = [A2::zero(); K];
            for dim in 0..K {
                converted_point[dim] = A2::try_from(point[dim]).map_err(|err| {
                    KdTreeConversionError::AxisConversion {
                        point_index,
                        dim,
                        source: format!("{err:?}"),
                    }
                })?;
            }

            entries.push((converted_item, converted_point));
        }

        Self::new_from_entries(&entries).map_err(Into::into)
    }
}

// Display implementation for debugging
impl<A, T, SS, LS, const K: usize, const B: usize, const ITEM_LEAF_MODE: u8> std::fmt::Display
    for KdTree<A, T, SS, LS, K, B, ITEM_LEAF_MODE>
where
    A: Axis<Coord = A> + std::fmt::Display,
    T: Content + std::fmt::Display,
    LS: LeafStrategy<A, T, SS, K, B>,
    SS: StemStrategy,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "KdTree {{")?;
        writeln!(f, "  Summary:")?;
        writeln!(f, "    size: {}", self.size)?;
        writeln!(f, "    max_stem_level: {}", self.max_stem_level)?;
        writeln!(f, "    item_leaf_mode: {:?}", self.item_leaf_mode())?;
        writeln!(f, "    stem len: {}", self.stems.len())?;
        writeln!(f, "    leaf count: {}", self.leaves.leaf_count())?;
        writeln!(f)?;

        // Display stems array
        writeln!(f, "  Stems (len={}):", self.stems.len())?;
        writeln!(f, "    [")?;
        for (i, stem) in self.stems.iter().enumerate() {
            if i % 8 == 0 {
                write!(f, "     ")?;
            }
            write!(f, "{:8.3}", stem)?;
            if i < self.stems.len() - 1 {
                write!(f, ",")?;
            }
            if (i + 1) % 8 == 0 || i == self.stems.len() - 1 {
                writeln!(f)?;
            } else {
                write!(f, "\t")?;
            }
        }
        writeln!(f, "    ]")?;
        writeln!(f)?;

        // Display stem_leaf_resolution
        writeln!(f, "  OwnedStemLeafResolution:")?;
        match &self.stem_leaf_resolution {
            OwnedStemLeafResolution::Arithmetic {
                stems_depth,
                leaf_count,
            } => {
                writeln!(f, "    Arithmetic {{")?;
                writeln!(f, "      stems_depth: {}", stems_depth)?;
                writeln!(f, "      leaf_count: {}", leaf_count)?;
                writeln!(f, "    }}")?;
            }
            OwnedStemLeafResolution::Pristine {
                stems_depth,
                leaf_count,
            } => {
                writeln!(f, "    Pristine {{")?;
                writeln!(f, "      stems_depth: {}", stems_depth)?;
                writeln!(f, "      leaf_count: {}", leaf_count)?;
                writeln!(f, "    }}")?;
            }
            OwnedStemLeafResolution::Mapped {
                min_stem_leaf_idx,
                leaf_idx_map,
            } => {
                writeln!(f, "    Mapped {{")?;
                writeln!(f, "      min_stem_leaf_idx: {}", min_stem_leaf_idx)?;
                writeln!(f, "      leaf_idx_map (len={}): [", leaf_idx_map.len())?;
                for (i, entry) in leaf_idx_map.iter().enumerate() {
                    match entry {
                        Some(idx) => writeln!(f, "        {}: Some({})", i, idx)?,
                        None => writeln!(f, "        {}: None", i)?,
                    }
                }
                writeln!(f, "      ]")?;
                writeln!(f, "    }}")?;
            }
        }
        writeln!(f)?;

        // Display leaves
        writeln!(f, "  Leaves (count={}):", self.leaves.leaf_count())?;
        for leaf_idx in 0..self.leaves.leaf_count() {
            let leaf_view = self.leaves.leaf_view(leaf_idx);
            let (points, items) = leaf_view.into_parts();

            write!(f, "    Leaf {} (count={}): [", leaf_idx, items.len())?;
            for i in 0..items.len() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "(")?;
                for dim in 0..K {
                    if dim > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:.3}", points[dim][i])?;
                }
                write!(f, "): {}", items[i])?;
            }
            writeln!(f, "]")?;
        }

        writeln!(f, "}}")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::leaf_strategy::dummy::DummyLeafStrategy;
    use crate::leaf_strategy::{FlatVec, VecOfArenas, VecOfArrays};
    use crate::stem_strategy::Donnelly;
    #[cfg(all(feature = "rkyv_08", feature = "simd", target_arch = "x86_64"))]
    use crate::stem_strategy::DonnellySimdFull;
    use crate::Eytzinger;
    use crate::SquaredEuclidean;
    #[cfg(feature = "rkyv_08")]
    use std::num::NonZeroUsize;

    #[cfg(feature = "rkyv_08")]
    #[derive(rkyv_08::Archive, rkyv_08::Serialize)]
    #[rkyv(crate = rkyv_08)]
    struct LegacyV6Tree {
        #[rkyv(with = crate::rkyv::adapters::AsAlignedCachelineABox)]
        stems: AVec<f64>,
        leaves: VecOfArenas<f64, u32, 2, 8>,
        stem_leaf_resolution: OwnedStemLeafResolution,
        size: usize,
        max_stem_level: i32,
        max_leaf_len: usize,
        _phantom: std::marker::PhantomData<(Eytzinger, u32)>,
    }

    fn sort_entries_u32<A: Copy, const K: usize>(
        mut entries: Vec<(u32, [A; K])>,
    ) -> Vec<(u32, [A; K])> {
        entries.sort_by_key(|(item, _)| *item);
        entries
    }

    fn sort_entries_u16<A: Copy, const K: usize>(
        mut entries: Vec<(u16, [A; K])>,
    ) -> Vec<(u16, [A; K])> {
        entries.sort_by_key(|(item, _)| *item);
        entries
    }

    #[test]
    fn test_default() {
        let kd_tree: KdTree<f32, u32, Eytzinger, DummyLeafStrategy, 3, 16> = Default::default();

        assert_eq!(kd_tree.size, 0);
        assert!(kd_tree.is_empty());
    }

    #[test]
    fn debug_output_summarizes_tree_and_leaf_storage_without_dumping_entries() {
        const COORDINATE: f64 = 12_345.25;
        const ITEM: u32 = 987_654;
        let entries = [(ITEM, [COORDINATE, -6_789.5])];

        type FlatTree = KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, 2, 4>, 2, 4>;
        type ArraysTree = KdTree<f64, u32, Eytzinger, VecOfArrays<f64, u32, 2, 4>, 2, 4>;
        type ArenasTree = KdTree<f64, u32, Eytzinger, VecOfArenas<f64, u32, 2, 4>, 2, 4>;

        let flat = FlatTree::new_from_entries(&entries).unwrap();
        let arrays = ArraysTree::new_from_entries(&entries).unwrap();
        let arenas = ArenasTree::new_from_entries(&entries).unwrap();

        let tree_debug = format!("{flat:?}");
        assert!(tree_debug.contains("KdTree"));
        assert!(tree_debug.contains("size: 1"));
        assert!(tree_debug.contains("dimensions: 2"));

        for leaf_debug in [
            format!("{:?}", flat.leaves),
            format!("{:?}", arrays.leaves),
            format!("{:?}", arenas.leaves),
        ] {
            assert!(leaf_debug.contains("size: 1"), "{leaf_debug}");
            assert!(leaf_debug.contains("leaf_count: 1"), "{leaf_debug}");
            assert!(!leaf_debug.contains(&COORDINATE.to_string()));
            assert!(!leaf_debug.contains(&ITEM.to_string()));
        }

        assert!(!tree_debug.contains(&COORDINATE.to_string()));
        assert!(!tree_debug.contains(&ITEM.to_string()));
    }

    #[test]
    fn test_from_iterator_empty() {
        let points = vec![[0.0f64; 3]];

        let kd_tree: KdTree<f64, u32, Eytzinger, DummyLeafStrategy, 3, 16> =
            points.into_iter().enumerate().collect();

        assert_eq!(kd_tree.size, 0);
    }

    #[test]
    fn scalar_donnelly_supports_an_incomplete_final_block() {
        // Create a tree whose natural height is not a block boundary.
        // With 100 items and bucket size 32, we get 4 leaves
        // 4 leaves -> depth = log2(4) = 2 levels (levels 0, 1)
        // max_stem_level = 1 (0-indexed)
        // stems_depth = max_stem_level + 1 = 2
        // Donnelly<4> still uses four-level block addressing, but scalar
        // traversal can stop after the two real levels.

        const TREE_SIZE: usize = 100;
        let content_to_add: Vec<[f32; 4]> = (0..TREE_SIZE)
            .map(|i| {
                let x = (i as f32) / (TREE_SIZE as f32);
                [x, x * 2.0, x * 3.0, x * 4.0]
            })
            .collect();

        let tree: KdTree<f32, u32, Donnelly<4>, FlatVec<f32, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&content_to_add).unwrap();

        assert_eq!(tree.size(), TREE_SIZE);

        let stems_depth = tree.max_stem_level() + 1;
        assert_eq!(stems_depth, 2, "scalar Donnelly must retain natural depth");
        assert_eq!(tree.max_stem_level(), 1);

        // Verify leaf resolution across the incomplete block.
        let query_point = [0.5f32, 1.0f32, 1.5f32, 2.0f32];
        let leaf_idx = tree.get_leaf_idx(&query_point);
        assert!(
            leaf_idx < tree.leaf_count(),
            "Leaf index should be valid. leaf_idx={}, leaf_count={}",
            leaf_idx,
            tree.leaf_count()
        );
    }

    #[test]
    fn item_leaf_mode_codes_have_stable_meanings() {
        assert_eq!(ItemLeafMode::from_code(0), ItemLeafMode::Unsorted);
        assert_eq!(
            ItemLeafMode::from_code(2),
            ItemLeafMode::SortedWithEncodedMin { shift: 2 }
        );
        assert_eq!(
            ItemLeafMode::from_code(254),
            ItemLeafMode::SortedWithEncodedMin { shift: 254 }
        );
        assert_eq!(
            ItemLeafMode::from_code(255),
            ItemLeafMode::SortedWithoutEncodedMin
        );

        assert!(!ItemLeafMode::Unsorted.has_sorted_leaves());
        assert!(ItemLeafMode::SortedWithoutEncodedMin.has_sorted_leaves());
        assert!(ItemLeafMode::from_code(2).has_sorted_leaves());
        assert_eq!(ItemLeafMode::from_code(2).encoded_min_shift(), Some(2));
    }

    #[cfg(feature = "rkyv_08")]
    #[test]
    fn rkyv_preserves_item_sorted_leaves_and_best_n_within_dispatch() {
        type Tree = KdTree<f64, u32, Eytzinger, VecOfArenas<f64, u32, 2, 8>, 2, 8>;
        type ArchivedTree = ArchivedKdTree<
            f64,
            u32,
            Eytzinger,
            VecOfArenas<f64, u32, 2, 8>,
            2,
            8,
            ITEM_LEAF_MODE_SORTED,
        >;

        let entries = (0..97u32)
            .map(|idx| {
                (
                    (idx * 37) % 97,
                    [((idx * 19) % 101) as f64, ((idx * 43) % 103) as f64],
                )
            })
            .collect::<Vec<_>>();
        let tree = Tree::builder()
            .with_item_sorted_leaves()
            .build_from_entries(&entries)
            .unwrap();
        let query = [50.0, 50.0];
        let max_qty = NonZeroUsize::new(7).unwrap();
        let expected = tree
            .query(&query)
            .best_n_within::<SquaredEuclidean<f64>>(20_000.0, max_qty)
            .execute()
            .into_sorted_vec();

        let bytes = rkyv_08::api::high::to_bytes_in::<_, rkyv_08::rancor::Error>(
            &tree,
            rkyv_08::util::AlignedVec::<128>::new(),
        )
        .unwrap();
        let archived =
            rkyv_08::access::<ArchivedTree, rkyv_08::rancor::Error>(bytes.as_slice()).unwrap();

        assert!(archived.item_sorted_leaves());
        assert_eq!(
            archived.item_leaf_mode(),
            ItemLeafMode::SortedWithoutEncodedMin
        );
        assert_eq!(
            archived
                .query(&query)
                .best_n_within::<SquaredEuclidean<f64>>(20_000.0, max_qty)
                .execute()
                .into_sorted_vec(),
            expected
        );
    }

    #[cfg(feature = "rkyv_08")]
    #[test]
    fn rkyv_preserves_embedded_min_item_summary_mode_and_queries() {
        type Tree = KdTree<f64, u32, Donnelly<3>, VecOfArenas<f64, u32, 3, 8>, 3, 8>;
        type ArchivedTree =
            ArchivedKdTree<f64, u32, Donnelly<3>, VecOfArenas<f64, u32, 3, 8>, 3, 8, 8>;

        let entries = (0..97u32)
            .map(|idx| {
                (
                    (1u32 << (16 + (idx % 8))) | idx,
                    [
                        ((idx * 19) % 101) as f64,
                        ((idx * 43) % 103) as f64,
                        ((idx * 61) % 107) as f64,
                    ],
                )
            })
            .collect::<Vec<_>>();
        let tree = Tree::builder()
            .with_embedded_min_item_shifted_summary::<8>()
            .build_from_entries(&entries)
            .unwrap();
        let query = [50.0, 50.0, 50.0];
        let max_qty = NonZeroUsize::new(7).unwrap();
        let expected = tree
            .query(&query)
            .best_n_within::<SquaredEuclidean<f64>>(20_000.0, max_qty)
            .execute()
            .into_sorted_vec();

        let bytes = rkyv_08::api::high::to_bytes_in::<_, rkyv_08::rancor::Error>(
            &tree,
            rkyv_08::util::AlignedVec::<128>::new(),
        )
        .unwrap();
        let archived =
            rkyv_08::access::<ArchivedTree, rkyv_08::rancor::Error>(bytes.as_slice()).unwrap();

        assert_eq!(
            archived.item_leaf_mode(),
            ItemLeafMode::SortedWithEncodedMin { shift: 8 }
        );
        assert_eq!(
            archived
                .query(&query)
                .best_n_within::<SquaredEuclidean<f64>>(20_000.0, max_qty)
                .execute()
                .into_sorted_vec(),
            expected
        );
    }

    #[cfg(feature = "rkyv_08")]
    #[test]
    fn rkyv_unsorted_tree_accepts_the_v6_archive_layout() {
        type Tree = KdTree<f64, u32, Eytzinger, VecOfArenas<f64, u32, 2, 8>, 2, 8>;
        type ArchivedTree = ArchivedKdTree<f64, u32, Eytzinger, VecOfArenas<f64, u32, 2, 8>, 2, 8>;

        let entries = (0..97u32)
            .map(|idx| (idx, [((idx * 19) % 101) as f64, ((idx * 43) % 103) as f64]))
            .collect::<Vec<_>>();
        let tree = Tree::new_from_entries(&entries).unwrap();
        let expected = tree.iter().collect::<Vec<_>>();
        let legacy = LegacyV6Tree {
            stems: tree.stems,
            leaves: tree.leaves,
            stem_leaf_resolution: tree.stem_leaf_resolution,
            size: tree.size,
            max_stem_level: tree.max_stem_level,
            max_leaf_len: tree.max_leaf_len,
            _phantom: std::marker::PhantomData,
        };

        let bytes = rkyv_08::api::high::to_bytes_in::<_, rkyv_08::rancor::Error>(
            &legacy,
            rkyv_08::util::AlignedVec::<128>::new(),
        )
        .unwrap();
        let archived =
            rkyv_08::access::<ArchivedTree, rkyv_08::rancor::Error>(bytes.as_slice()).unwrap();

        assert!(!archived.item_sorted_leaves());
        assert_eq!(archived.item_leaf_mode(), ItemLeafMode::Unsorted);
        assert_eq!(archived.size(), entries.len());
        assert_eq!(archived.iter().collect::<Vec<_>>(), expected);
    }

    #[cfg(feature = "rkyv_08")]
    #[test]
    fn rkyv_archived_donnelly_stems_stay_cacheline_aligned() {
        type Tree = KdTree<f64, u32, Donnelly<3>, VecOfArenas<f64, u32, 3, 32>, 3, 32>;

        let points: Vec<[f64; 3]> = (0..4096)
            .map(|i| {
                let x = i as f64 / 4096.0;
                [x, x * 2.0, x * 3.0]
            })
            .collect();

        let tree = Tree::new_from_slice(&points).unwrap();

        let bytes = rkyv_08::api::high::to_bytes_in::<_, rkyv_08::rancor::Error>(
            &tree,
            rkyv_08::util::AlignedVec::<128>::new(),
        )
        .unwrap();

        let archived = rkyv_08::access::<
            ArchivedKdTree<f64, u32, Donnelly<3>, VecOfArenas<f64, u32, 3, 32>, 3, 32>,
            rkyv_08::rancor::Error,
        >(bytes.as_slice())
        .unwrap();

        assert_eq!(bytes.as_ptr() as usize % 128, 0);
        assert_eq!(archived.archived_stems().as_ptr() as usize % 128, 0);
        assert_eq!(archived.size(), tree.size());
        assert_eq!(archived.leaf_count(), tree.leaf_count());
        assert_eq!(archived.max_stem_level(), tree.max_stem_level());
    }

    #[cfg(feature = "rkyv_08")]
    #[test]
    fn rkyv_roundtrip_preserves_alignment_and_query_results() {
        type Tree = KdTree<f64, u32, Donnelly<3>, VecOfArenas<f64, u32, 3, 32>, 3, 32>;
        type ArchivedTree =
            ArchivedKdTree<f64, u32, Donnelly<3>, VecOfArenas<f64, u32, 3, 32>, 3, 32>;

        let points: Vec<[f64; 3]> = (0..2048)
            .map(|i| {
                let x = i as f64 / 2048.0;
                [x, (i % 127) as f64 / 127.0, (i % 63) as f64 / 63.0]
            })
            .collect();

        let tree = Tree::new_from_slice(&points).unwrap();
        let query = [0.123, 0.456, 0.789];
        let expected = tree
            .query(&query)
            .nearest_one::<SquaredEuclidean<f64>>()
            .execute();

        let bytes = rkyv_08::api::high::to_bytes_in::<_, rkyv_08::rancor::Error>(
            &tree,
            rkyv_08::util::AlignedVec::<128>::new(),
        )
        .unwrap();

        let archived =
            rkyv_08::access::<ArchivedTree, rkyv_08::rancor::Error>(bytes.as_slice()).unwrap();
        let roundtrip =
            rkyv_08::api::high::from_bytes::<Tree, rkyv_08::rancor::Error>(bytes.as_slice())
                .unwrap();

        assert_eq!(archived.size(), tree.size());
        assert_eq!(
            roundtrip.stems.as_ptr() as usize % aligned_vec::CACHELINE_ALIGN,
            0
        );
        assert_eq!(
            roundtrip.leaves.leaf_bytes_ptr() as usize % aligned_vec::CACHELINE_ALIGN,
            0
        );
        assert_eq!(
            roundtrip
                .query(&query)
                .nearest_one::<SquaredEuclidean<f64>>()
                .execute(),
            expected
        );
    }

    #[cfg(all(feature = "rkyv_08", feature = "simd", target_arch = "x86_64"))]
    #[test]
    fn rkyv_archived_donnelly_block4_vec_of_arenas_within_matches_owned() {
        type Tree = KdTree<f32, u32, DonnellySimdFull<4>, VecOfArenas<f32, u32, 4, 32>, 4, 32>;
        type ArchivedTree =
            ArchivedKdTree<f32, u32, DonnellySimdFull<4>, VecOfArenas<f32, u32, 4, 32>, 4, 32>;

        let points: Vec<[f32; 4]> = (0..4096)
            .map(|i| {
                let x = i as f32 / 4096.0;
                [
                    x,
                    ((i * 3) % 257) as f32 / 257.0,
                    ((i * 5) % 263) as f32 / 263.0,
                    ((i * 7) % 269) as f32 / 269.0,
                ]
            })
            .collect();

        let tree = Tree::new_from_slice(&points).unwrap();
        let query = [0.33, 0.27, 0.41, 0.59];
        let max_dist = 0.55f32;
        let expected = tree
            .query(&query)
            .within::<crate::Manhattan<f32>>(max_dist)
            .execute();

        let bytes = rkyv_08::api::high::to_bytes_in::<_, rkyv_08::rancor::Error>(
            &tree,
            rkyv_08::util::AlignedVec::<128>::new(),
        )
        .unwrap();

        let archived =
            rkyv_08::access::<ArchivedTree, rkyv_08::rancor::Error>(bytes.as_slice()).unwrap();
        let actual = archived
            .query(&query)
            .within::<crate::Manhattan<f32>>(max_dist)
            .execute();

        assert_eq!(actual, expected);
    }

    #[cfg(feature = "rkyv_08")]
    #[test]
    fn rkyv_archived_vec_of_arenas_supports_queries() {
        type Tree = KdTree<f64, u32, Eytzinger, VecOfArenas<f64, u32, 3, 32>, 3, 32>;
        type ArchivedTree =
            ArchivedKdTree<f64, u32, Eytzinger, VecOfArenas<f64, u32, 3, 32>, 3, 32>;

        let points: Vec<[f64; 3]> = (0..4096)
            .map(|i| {
                [
                    (i % 257) as f64 / 257.0,
                    (i % 131) as f64 / 131.0,
                    (i % 67) as f64 / 67.0,
                ]
            })
            .collect();

        let tree = Tree::new_from_slice(&points).unwrap();
        let query = [0.321, 0.456, 0.789];
        let max_qty = NonZeroUsize::new(8).unwrap();
        let max_dist = 0.025;

        let bytes = rkyv_08::api::high::to_bytes_in::<_, rkyv_08::rancor::Error>(
            &tree,
            rkyv_08::util::AlignedVec::<128>::new(),
        )
        .unwrap();
        let archived =
            rkyv_08::access::<ArchivedTree, rkyv_08::rancor::Error>(bytes.as_slice()).unwrap();

        assert_eq!(
            archived
                .query(&query)
                .nearest_one::<SquaredEuclidean<f64>>()
                .approx()
                .execute(),
            tree.query(&query)
                .nearest_one::<SquaredEuclidean<f64>>()
                .approx()
                .execute()
        );
        assert_eq!(
            archived
                .query(&query)
                .nearest_one::<SquaredEuclidean<f64>>()
                .execute(),
            tree.query(&query)
                .nearest_one::<SquaredEuclidean<f64>>()
                .execute()
        );
        assert_eq!(
            archived
                .query(&query)
                .nearest_n::<SquaredEuclidean<f64>>(max_qty)
                .execute(),
            tree.query(&query)
                .nearest_n::<SquaredEuclidean<f64>>(max_qty)
                .execute()
        );
        assert_eq!(
            archived
                .query(&query)
                .nearest_n::<SquaredEuclidean<f64>>(max_qty)
                .within(max_dist)
                .exclusive_boundaries()
                .execute(),
            tree.query(&query)
                .nearest_n::<SquaredEuclidean<f64>>(max_qty)
                .within(max_dist)
                .exclusive_boundaries()
                .execute()
        );
        assert_eq!(
            archived
                .query(&query)
                .within::<SquaredEuclidean<f64>>(max_dist)
                .exclusive_boundaries()
                .execute(),
            tree.query(&query)
                .within::<SquaredEuclidean<f64>>(max_dist)
                .exclusive_boundaries()
                .execute()
        );
        assert_eq!(
            archived
                .query(&query)
                .within::<SquaredEuclidean<f64>>(max_dist)
                .execute(),
            tree.query(&query)
                .within::<SquaredEuclidean<f64>>(max_dist)
                .execute()
        );
        assert_eq!(
            archived
                .query(&query)
                .within::<SquaredEuclidean<f64>>(max_dist)
                .unsorted()
                .execute()
                .len(),
            tree.query(&query)
                .within::<SquaredEuclidean<f64>>(max_dist)
                .unsorted()
                .execute()
                .len()
        );
        assert_eq!(
            archived
                .query(&query)
                .within::<SquaredEuclidean<f64>>(max_dist)
                .exclusive_boundaries()
                .unsorted()
                .execute()
                .len(),
            tree.query(&query)
                .within::<SquaredEuclidean<f64>>(max_dist)
                .exclusive_boundaries()
                .unsorted()
                .execute()
                .len()
        );
        assert_eq!(
            archived
                .query(&query)
                .within::<SquaredEuclidean<f64>>(max_dist)
                .unsorted()
                .iter()
                .count(),
            tree.query(&query)
                .within::<SquaredEuclidean<f64>>(max_dist)
                .unsorted()
                .iter()
                .count()
        );
        assert_eq!(
            archived
                .query(&query)
                .within::<SquaredEuclidean<f64>>(max_dist)
                .exclusive_boundaries()
                .unsorted()
                .iter()
                .count(),
            tree.query(&query)
                .within::<SquaredEuclidean<f64>>(max_dist)
                .exclusive_boundaries()
                .unsorted()
                .iter()
                .count()
        );
        let archived_iter: Vec<_> = archived.iter().collect();
        let tree_iter: Vec<_> = tree.iter().collect();
        assert_eq!(archived_iter, tree_iter);
        assert_eq!(
            archived
                .query(&query)
                .best_n_within::<SquaredEuclidean<f64>>(max_dist, max_qty)
                .exclusive_boundaries()
                .execute()
                .into_sorted_vec(),
            tree.query(&query)
                .best_n_within::<SquaredEuclidean<f64>>(max_dist, max_qty)
                .exclusive_boundaries()
                .execute()
                .into_sorted_vec()
        );
        assert_eq!(
            archived
                .query(&query)
                .best_n_within::<SquaredEuclidean<f64>>(max_dist, max_qty)
                .execute()
                .into_sorted_vec(),
            tree.query(&query)
                .best_n_within::<SquaredEuclidean<f64>>(max_dist, max_qty)
                .execute()
                .into_sorted_vec()
        );
    }

    #[test]
    fn can_create_points_only_tree() {
        // points-only tree can be created by specifying T / Item parameter as ()
        type Tree = KdTree<f64, (), Eytzinger, VecOfArrays<f64, (), 3, 256>, 3, 256>;

        let points = vec![[0.0f64; 3]];

        let kd_tree = Tree::new_from_slice_no_items(&points).unwrap();

        assert_eq!(kd_tree.size, 1);
    }

    #[test]
    fn can_add_to_and_remove_from_points_only_tree() {
        // points-only tree can be created by specifying T / Item parameter as ()
        type Tree = KdTree<f64, (), Eytzinger, VecOfArrays<f64, (), 3, 256>, 3, 256>;

        let points = vec![[0.0f64; 3]];

        let mut kd_tree = Tree::new_from_slice_no_items(&points).unwrap();

        assert_eq!(kd_tree.size, 1);

        kd_tree.add(&[1.0f64; 3], ()).unwrap();
        assert_eq!(kd_tree.size, 2);

        kd_tree.remove(&[1.0f64; 3], ());
        assert_eq!(kd_tree.size, 1);

        kd_tree.remove(&[0.0f64; 3], ());
        assert_eq!(kd_tree.size, 0);
    }

    #[test]
    fn new_from_entries_preserves_explicit_items() {
        type Tree = KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, 2, 4>, 2, 4>;

        let entries = vec![
            (42u32, [0.0f32, 0.0f32]),
            (7u32, [5.0f32, 5.0f32]),
            (99u32, [10.0f32, 10.0f32]),
        ];

        let tree = Tree::new_from_entries(&entries).unwrap();

        assert_eq!(tree.size(), entries.len());
        assert_eq!(
            sort_entries_u32(tree.iter().collect()),
            sort_entries_u32(entries)
        );

        let nearest = tree
            .query(&[5.1f32, 4.9f32])
            .nearest_one::<SquaredEuclidean<f32>>()
            .execute();
        assert_eq!(nearest.item, 7);
    }

    #[test]
    fn new_from_source_accepts_custom_source_structs() {
        type Tree = KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, 2, 4>, 2, 4>;

        #[derive(Clone, Copy)]
        struct SourcePoint {
            id: u32,
            x: f64,
            y: f64,
        }

        let source = [
            SourcePoint {
                id: 11u32,
                x: 0.0f64,
                y: 0.0f64,
            },
            SourcePoint {
                id: 22u32,
                x: 3.0f64,
                y: 3.0f64,
            },
            SourcePoint {
                id: 33u32,
                x: 9.0f64,
                y: 1.0f64,
            },
        ];

        let tree = Tree::new_from_source(
            &source,
            |point, dim| match dim {
                0 => point.x,
                1 => point.y,
                _ => unreachable!(),
            },
            |_src_idx, point| point.id,
        )
        .unwrap();

        assert_eq!(tree.size(), source.len());
        assert_eq!(
            sort_entries_u32(tree.iter().collect()),
            sort_entries_u32(vec![
                (11u32, [0.0f64, 0.0f64]),
                (22u32, [3.0f64, 3.0f64]),
                (33u32, [9.0f64, 1.0f64]),
            ])
        );
    }

    #[test]
    fn new_from_source_can_use_indices_for_items() {
        type Tree = KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, 2, 4>, 2, 4>;

        let source = [[0.0f64, 0.0f64], [3.0f64, 3.0f64], [9.0f64, 1.0f64]];

        let tree = Tree::new_from_source(
            &source,
            |point, dim| point[dim],
            |src_idx, _| src_idx as u32 + 100,
        )
        .unwrap();

        assert_eq!(
            sort_entries_u32(tree.iter().collect()),
            sort_entries_u32(vec![
                (100u32, [0.0f64, 0.0f64]),
                (101u32, [3.0f64, 3.0f64]),
                (102u32, [9.0f64, 1.0f64]),
            ])
        );
    }

    #[test]
    fn try_from_kdtree_converts_across_variants() {
        type SourceTree = KdTree<f32, u16, Eytzinger, VecOfArrays<f32, u16, 2, 4>, 2, 4>;
        type DestTree = KdTree<f64, u32, Donnelly<2>, FlatVec<f64, u32, 2, 8>, 2, 8>;

        let entries = vec![
            (10u16, [1.0f32, 2.0f32]),
            (20u16, [8.0f32, 3.0f32]),
            (30u16, [2.0f32, 9.0f32]),
            (40u16, [6.0f32, 7.0f32]),
            (50u16, [4.0f32, 4.0f32]),
        ];

        let source = SourceTree::new_from_entries(&entries).unwrap();
        let nearest_source = source
            .query(&[4.0f32, 4.0f32])
            .nearest_one::<SquaredEuclidean<f32>>()
            .execute();

        let converted: DestTree = source.try_convert().unwrap();

        assert_eq!(converted.size(), entries.len());
        assert_eq!(
            sort_entries_u32(converted.iter().collect()),
            sort_entries_u32(vec![
                (10u32, [1.0f64, 2.0f64]),
                (20u32, [8.0f64, 3.0f64]),
                (30u32, [2.0f64, 9.0f64]),
                (40u32, [6.0f64, 7.0f64]),
                (50u32, [4.0f64, 4.0f64]),
            ])
        );

        let nearest_converted = converted
            .query(&[4.0f64, 4.0f64])
            .nearest_one::<SquaredEuclidean<f64>>()
            .execute();
        assert_eq!(nearest_converted.item, nearest_source.item as u32);
    }

    #[test]
    fn try_from_kdtree_reports_item_conversion_failure() {
        type SourceTree = KdTree<u16, u16, Eytzinger, FlatVec<u16, u16, 2, 4>, 2, 4>;
        type DestTree = KdTree<u16, u8, Eytzinger, FlatVec<u16, u8, 2, 4>, 2, 4>;

        let source = SourceTree::new_from_entries(&[(300u16, [1u16, 2u16])]).unwrap();
        let err = match DestTree::try_from(&source) {
            Ok(_) => panic!("expected item conversion to fail"),
            Err(err) => err,
        };

        assert!(matches!(
            err,
            KdTreeConversionError::ItemConversion { point_index: 0, .. }
        ));
    }

    #[test]
    fn try_from_kdtree_reports_axis_conversion_failure() {
        type SourceTree = KdTree<u16, u16, Eytzinger, FlatVec<u16, u16, 2, 4>, 2, 4>;
        type DestTree = KdTree<u8, u16, Eytzinger, FlatVec<u8, u16, 2, 4>, 2, 4>;

        let source = SourceTree::new_from_entries(&[(7u16, [300u16, 2u16])]).unwrap();
        let err = match DestTree::try_from(&source) {
            Ok(_) => panic!("expected axis conversion to fail"),
            Err(err) => err,
        };

        assert!(matches!(
            err,
            KdTreeConversionError::AxisConversion {
                point_index: 0,
                dim: 0,
                ..
            }
        ));
    }

    #[test]
    fn try_from_kdtree_converts_mutable_to_immutable() {
        type SourceTree = KdTree<u16, u16, Eytzinger, VecOfArrays<u16, u16, 2, 4>, 2, 4>;
        type DestTree = KdTree<u16, u16, Eytzinger, FlatVec<u16, u16, 2, 4>, 2, 4>;

        let entries = vec![
            (1u16, [1u16, 1u16]),
            (2u16, [9u16, 9u16]),
            (3u16, [4u16, 5u16]),
        ];
        let source = SourceTree::new_from_entries(&entries).unwrap();
        let converted: DestTree = source.try_convert().unwrap();

        assert_eq!(
            sort_entries_u16(converted.iter().collect()),
            sort_entries_u16(entries)
        );
    }
}

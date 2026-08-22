use nonmax::NonMaxUsize;
use std::fmt::Debug;

use crate::leaf_view::LeafArena;
use crate::{Axis, Content, StemStrategy};

#[doc(inline)]
pub use crate::leaf_view::LeafView;

mod sealed {
    pub trait Sealed {}
}

/// Marker trait indicating whether a leaf strategy supports mutation.
///
/// This trait is used to enable type-level distinction between mutable and
/// immutable leaf strategies, allowing for optimized monomorphization.
pub trait Mutability: sealed::Sealed + 'static {
    /// Whether this is a mutable strategy, usable where a `const` is required.
    ///
    /// [`Self::is_mutable`] cannot be called from a constant, and the stem/leaf
    /// compatibility check in [`crate::kd_tree`] has to run at compile time.
    const IS_MUTABLE: bool;

    /// Returns true if this is a mutable strategy
    fn is_mutable() -> bool;

    /// Creates the appropriate OwnedStemLeafResolution for this mutability type
    fn initial_stem_leaf_resolution<AX, SS, const K: usize>(
        stems_depth: usize,
        leaf_count: usize,
    ) -> crate::kd_tree::OwnedStemLeafResolution
    where
        AX: Axis<Coord = AX>,
        SS: StemStrategy;
}

fn build_mapped_stem_leaf_resolution<AX, SS, const K: usize>(
    stems_depth: usize,
    leaf_count: usize,
) -> crate::kd_tree::OwnedStemLeafResolution
where
    AX: Axis<Coord = AX>,
    SS: StemStrategy,
{
    if leaf_count == 0 {
        return crate::kd_tree::OwnedStemLeafResolution::Mapped {
            min_stem_leaf_idx: 0,
            leaf_idx_map: Vec::new(),
        };
    }

    let min_stem_leaf_idx = 0;

    // Determine highest stem index that can resolve to a leaf at this depth.
    let mut stem_strategy = SS::new_no_ptr();
    for bit_idx in (0..stems_depth).rev() {
        let is_right = (leaf_count - 1) & (1 << bit_idx) != 0;
        stem_strategy.traverse::<AX, K>(is_right);
    }

    let mut leaf_idx_map: Vec<Option<NonMaxUsize>> = vec![None; stem_strategy.stem_idx() + 1];

    // Map each leaf index to the traversal endpoint that would resolve it.
    for leaf_idx in 0..leaf_count {
        let mut stem_strategy = SS::new_no_ptr();
        for bit_idx in (0..stems_depth).rev() {
            let is_right = leaf_idx & (1 << bit_idx) != 0;
            stem_strategy.traverse::<AX, K>(is_right);
        }
        if let Some(existing_leaf_idx) = leaf_idx_map[stem_strategy.stem_idx()] {
            panic!(
                "Duplicate terminal stem index in initial mapped leaf_idx_map construction: stem_idx={} existing_leaf_idx={} new_leaf_idx={}",
                stem_strategy.stem_idx(),
                existing_leaf_idx.get(),
                leaf_idx
            );
        }
        leaf_idx_map[stem_strategy.stem_idx()] =
            Some(NonMaxUsize::new(leaf_idx).expect("leaf_idx overflow"));
    }

    crate::kd_tree::OwnedStemLeafResolution::Mapped {
        min_stem_leaf_idx,
        leaf_idx_map,
    }
}

/// Marker type for immutable leaf strategies.
///
/// Immutable strategies never mutate the tree structure after construction,
/// allowing for simpler and faster traversal logic.
#[derive(Debug, Clone, Copy)]
pub struct Immutable;
impl sealed::Sealed for Immutable {}
impl Mutability for Immutable {
    const IS_MUTABLE: bool = false;

    fn is_mutable() -> bool {
        false
    }

    fn initial_stem_leaf_resolution<AX, SS, const K: usize>(
        stems_depth: usize,
        leaf_count: usize,
    ) -> crate::kd_tree::OwnedStemLeafResolution
    where
        AX: Axis<Coord = AX>,
        SS: StemStrategy,
    {
        crate::kd_tree::OwnedStemLeafResolution::Arithmetic {
            stems_depth,
            leaf_count,
        }
    }
}

/// Marker type for mutable leaf strategies.
///
/// Mutable strategies support adding/removing points after construction,
/// requiring more complex traversal logic to handle non-uniform tree depths.
#[derive(Debug, Clone, Copy)]
pub struct Mutable;
impl sealed::Sealed for Mutable {}
impl Mutability for Mutable {
    const IS_MUTABLE: bool = true;

    fn is_mutable() -> bool {
        true
    }

    fn initial_stem_leaf_resolution<AX, SS, const K: usize>(
        stems_depth: usize,
        leaf_count: usize,
    ) -> crate::kd_tree::OwnedStemLeafResolution
    where
        AX: Axis<Coord = AX>,
        SS: StemStrategy,
    {
        // Start in Mapped state with min_stem_leaf_idx = 0 for simplicity.
        // TODO: Optimize later with Pristine state and dynamic min_stem_leaf_idx
        build_mapped_stem_leaf_resolution::<AX, SS, K>(stems_depth, leaf_count)
    }
}

/// Specifies whether a LeafStrategy's bucket size is a hard or soft limit
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BucketLimitType {
    /// Bucket size is completely fixed
    Hard,

    /// Bucket size is a target and can be larger than specified size if reqd
    Soft,
}

/// The leaf access projection supported by a leaf strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeafProjection {
    /// Strategy exposes leaf data through [`LeafView`].
    LeafView,
    /// Strategy exposes leaf data through an arena-backed internal layout.
    ///
    /// This is used by the [`VecOfArenas`](`crate::leaf_strategy::VecOfArenas`) leaf strategy
    /// and can't be used by third-party leaf strategies.
    LeafArena,
}

/// Query/access strategy for how leaf storage is laid out.
///
/// To see which leaf strategies are available, see the [`leaf_strategies`](`crate::leaf_strategy`) module.
///
/// The generic parameters are the same as those of [`KdTree`](`crate::kd_tree::KdTree`) and must match the tree with which
/// they're being specified for. It is a bit verbose and clunky to have to repeat these parameters
/// that are specified on the `KdTree` itself as well as on a given leaf strategy. Unfortunately,
/// this has been the least worst option after having investigated implementations that would
/// eliminate this duplication. Each provided leaf strategy also exposes a type alias that avoids
/// this if you find that preferable.
///
/// Third-party leaf strategies should provide an implementation for [`LeafStrategy::leaf_view`] and
/// ignore [`LeafStrategy::leaf_arena`], which is used by
/// Kiddo's bundled [`VecOfArenas`](`crate::leaf_strategy::VecOfArenas`) leaf strategy.
pub trait LeafStrategy<A, T, SS, const K: usize, const B: usize>: Debug
where
    A: Axis<Coord = A>,
    T: Content,
    SS: StemStrategy,
{
    /// Coordinate scalar type.
    type Num;

    /// Marker type indicating whether this strategy supports mutation.
    #[allow(private_bounds)]
    type Mutability: Mutability;

    /// Whether bucket size is a hard or soft limit
    const BUCKET_LIMIT_TYPE: BucketLimitType;

    /// The leaf projection exposed by this strategy.
    const LEAF_PROJECTION: LeafProjection;

    // ---- Introspection / minimal accessors ----

    /// Total number of stored items.
    fn size(&self) -> usize;

    /// Number of leaves maintained by the strategy (buckets/extents).
    fn leaf_count(&self) -> usize;

    /// Number of items in a given leaf.
    fn leaf_len(&self, leaf_idx: usize) -> usize;

    /// Returns a view into the specified leaf's data.
    fn leaf_view(&self, leaf_idx: usize) -> LeafView<'_, A, T, K, B>;

    /// Returns arena-backed access for the specified leaf.
    ///
    /// This is an internal optimisation path for [`VecOfArenas`](`crate::leaf_strategy::VecOfArenas`)
    /// and cannot be re-implemented by third-party leaf strategies.
    #[inline(always)]
    fn leaf_arena(&self, _leaf_idx: usize) -> LeafArena<'_, A, T, K> {
        unimplemented!("leaf_arena is unsupported for this leaf strategy")
    }

    /// Returns the point/item pair at `pos_in_leaf`.
    #[inline(always)]
    fn leaf_point_item(&self, leaf_idx: usize, pos_in_leaf: usize) -> ([A; K], T)
    where
        A: Copy,
        T: Copy,
    {
        match Self::LEAF_PROJECTION {
            LeafProjection::LeafView => self.leaf_view(leaf_idx).point_item(pos_in_leaf),
            LeafProjection::LeafArena => self.leaf_arena(leaf_idx).point_item(pos_in_leaf),
        }
    }

    /// Best-effort hook for enabling transparent huge pages on large contiguous buffers.
    ///
    /// Strategies with one or more long-lived contiguous allocations can override this to
    /// call into crate-internal huge-page hints after construction.
    #[inline]
    fn maybe_enable_huge_pages(&self) {}

    /// Replaces the first exact `(point, old_item)` match in the specified leaf.
    ///
    /// Returns `true` if a replacement happened, `false` otherwise.
    #[inline]
    fn replace_item_in_leaf(
        &mut self,
        _leaf_idx: usize,
        _point: &[A; K],
        _old_item: T,
        _new_item: T,
    ) -> bool
    where
        T: PartialEq,
    {
        false
    }
}

/// Leaf strategies that can be constructed by `KdTree` builders.
///
/// Archived leaf strategies only need to implement [`LeafStrategy`], not this trait.
pub trait ConstructibleLeafStrategy<AX, T, SS, const K: usize, const B: usize>:
    LeafStrategy<AX, T, SS, K, B>
where
    AX: Axis<Coord = AX>,
    T: Content,
    SS: StemStrategy,
{
    /// Create a builder with an intended capacity (in points).
    fn new_with_capacity(capacity: usize) -> Self;

    /// Create a new LeafStrategy with a single, empty leaf.
    fn new_with_empty_leaf() -> Self
    where
        Self: Sized,
    {
        Self::new_with_capacity(0)
    }

    /// Appends a new leaf to the storage.
    fn append_leaf(&mut self, leaf_points: &[&[AX]; K], leaf_items: &[T]);
}

/// Trait for leaf strategies that support mutation (adding/removing points).
pub trait MutableLeafStrategy<AX, T, SS, const K: usize, const B: usize>:
    ConstructibleLeafStrategy<AX, T, SS, K, B>
where
    AX: Axis<Coord = AX>,
    T: Content,
    SS: StemStrategy,
{
    /// Add an item to a leaf.
    ///
    /// * The leaf is expected to exist.
    /// * The leaf must not be full.
    /// * The caller must ensure that the right leaf is being added to.
    fn add_to_leaf(&mut self, leaf_idx: usize, point: &[AX; K], item: T);

    /// Remove an item (or items) from a leaf
    ///
    /// * The leaf is expected to exist.
    /// * The whole leaf is searched for any entries where both the point and the
    ///   value match. Any entries matching are removed.
    fn remove_from_leaf(&mut self, leaf_idx: usize, point: &[AX; K], item: T);

    /// Returns true if the specified leaf is full.
    fn is_leaf_full(&self, leaf_idx: usize) -> bool;

    /// Splits a full leaf, returning the pivot value and the index of
    /// the new leaf that the leaf was split into.
    fn split_leaf(
        &mut self,
        leaf_idx: usize,
        split_dim: usize,
    ) -> Result<(AX, usize), crate::kd_tree::ConstructionError>;
}

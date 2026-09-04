use crate::kd_tree::{ConstructionError, ITEM_LEAF_MODE_SORTED, ITEM_LEAF_MODE_UNSORTED};
use crate::stem_strategy::donnelly::DonnellyBlock3SummaryLayout;
use crate::traits::leaf_strategy::{ConstructibleLeafStrategy, Immutable, LeafStrategy};
use crate::{Axis, Content, KdTree, StemStrategy};

use super::construction::{
    populate_f64_donnelly3_min_item_summaries, sort_leaf_scratch_by_item,
    validate_auto_generated_items, DefaultConstruction, LeafSorter, SerialConstruction,
    StemSummaryPopulator,
};
#[cfg(feature = "multi-threaded")]
use super::construction::{ParallelConstruction, DEFAULT_PARALLEL_CONSTRUCTION_THRESHOLD};

/// Configures how a [`KdTree`] is constructed.
///
/// With the default `multi-threaded` feature the builder starts from parallel
/// construction at or above `DEFAULT_PARALLEL_CONSTRUCTION_THRESHOLD` points,
/// using the current Rayon thread pool. Callers can use
/// `rayon::ThreadPool::install` to control its thread count, or
/// [`KdTreeBuilder::with_serial_construction`] to opt out.
///
/// Without that feature the parallel policy does not exist: the builder starts
/// from [`SerialConstruction`], and `with_parallel_construction` and its
/// threshold variant are not compiled. See [`DefaultConstruction`].
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "multi-threaded")] {
/// use kiddo::leaf_strategy::FlatVec;
/// use kiddo::{Eytzinger, KdTree};
///
/// type Tree = KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, 3, 32>, 3, 32>;
///
/// let points = vec![[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]];
/// let tree = Tree::builder()
///     .with_parallel_construction_threshold(1_000)
///     .build_from_slice(&points)
///     .unwrap();
///
/// assert_eq!(tree.size(), 2);
/// # }
/// ```
#[must_use = "a construction builder does nothing until a build method is called"]
pub struct KdTreeBuilder<
    A,
    T,
    SS,
    LS,
    const K: usize,
    const B: usize,
    P = DefaultConstruction,
    const ITEM_LEAF_MODE: u8 = ITEM_LEAF_MODE_UNSORTED,
> {
    // Only the parallel build path reads the policy; the serial one is a ZST
    // marker, so without `multi-threaded` this field is never loaded.
    #[cfg_attr(not(feature = "multi-threaded"), allow(dead_code))]
    pub(in crate::kd_tree) policy: P,
    pub(in crate::kd_tree) leaf_sorter: Option<LeafSorter<A, T, K>>,
    pub(in crate::kd_tree) stem_summary_populator: Option<StemSummaryPopulator<A, LS>>,
    pub(in crate::kd_tree) _phantom: std::marker::PhantomData<(A, T, SS, LS)>,
}

impl<A, T, SS, LS, const K: usize, const B: usize, P, const ITEM_LEAF_MODE: u8>
    KdTreeBuilder<A, T, SS, LS, K, B, P, ITEM_LEAF_MODE>
{
    /// Forces serial construction.
    pub fn with_serial_construction(
        self,
    ) -> KdTreeBuilder<A, T, SS, LS, K, B, SerialConstruction, ITEM_LEAF_MODE> {
        KdTreeBuilder {
            policy: SerialConstruction,
            leaf_sorter: self.leaf_sorter,
            stem_summary_populator: self.stem_summary_populator,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Sorts every leaf by ascending item value after spatial construction.
    ///
    /// This enables an early-exit `best_n_within` leaf kernel. The item type
    /// must implement [`Ord`] so the stored ordering is total and the early
    /// exit remains valid for every item in a leaf.
    ///
    /// This mode is available only for immutable leaf strategies.
    ///
    /// ```compile_fail
    /// use kiddo::leaf_strategy::VecOfArrays;
    /// use kiddo::{Eytzinger, KdTree};
    ///
    /// type MutableTree =
    ///     KdTree<f64, u32, Eytzinger, VecOfArrays<f64, u32, 2, 32>, 2, 32>;
    ///
    /// let _ = MutableTree::builder().with_item_sorted_leaves();
    /// ```
    pub fn with_item_sorted_leaves(
        mut self,
    ) -> KdTreeBuilder<A, T, SS, LS, K, B, P, ITEM_LEAF_MODE_SORTED>
    where
        A: Axis<Coord = A>,
        T: Content + Ord,
        SS: StemStrategy,
        LS: LeafStrategy<A, T, SS, K, B, Mutability = Immutable>,
    {
        self.leaf_sorter = Some(sort_leaf_scratch_by_item::<A, T, K>);
        self.stem_summary_populator = None;
        KdTreeBuilder {
            policy: self.policy,
            leaf_sorter: self.leaf_sorter,
            stem_summary_populator: self.stem_summary_populator,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Forces parallel construction for soft-bucket leaf strategies.
    ///
    /// Hard-bucket strategies currently retain their serial construction
    /// algorithm.
    ///
    /// Requires the `multi-threaded` feature.
    #[cfg(feature = "multi-threaded")]
    pub fn with_parallel_construction(
        self,
    ) -> KdTreeBuilder<A, T, SS, LS, K, B, ParallelConstruction, ITEM_LEAF_MODE> {
        KdTreeBuilder {
            policy: ParallelConstruction::with_threshold(1),
            leaf_sorter: self.leaf_sorter,
            stem_summary_populator: self.stem_summary_populator,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Uses parallel construction at or above `item_count`, and serial
    /// construction below it.
    ///
    /// The threshold also controls recursive Rayon join granularity.
    /// A threshold of zero is treated as forced parallel construction.
    ///
    /// Requires the `multi-threaded` feature.
    #[cfg(feature = "multi-threaded")]
    pub fn with_parallel_construction_threshold(
        self,
        item_count: usize,
    ) -> KdTreeBuilder<A, T, SS, LS, K, B, ParallelConstruction, ITEM_LEAF_MODE> {
        KdTreeBuilder {
            policy: ParallelConstruction::with_threshold(item_count),
            leaf_sorter: self.leaf_sorter,
            stem_summary_populator: self.stem_summary_populator,
            _phantom: std::marker::PhantomData,
        }
    }
}

fn enable_embedded_min_item_summary<
    SS,
    LS,
    const K: usize,
    const B: usize,
    P,
    const ITEM_LEAF_MODE: u8,
    const SHIFT: u8,
>(
    mut builder: KdTreeBuilder<f64, u32, SS, LS, K, B, P, ITEM_LEAF_MODE>,
) -> KdTreeBuilder<f64, u32, SS, LS, K, B, P, SHIFT>
where
    SS: DonnellyBlock3SummaryLayout,
    LS: LeafStrategy<f64, u32, SS, K, B, Mutability = Immutable>,
{
    const {
        assert!(
            SHIFT > ITEM_LEAF_MODE_UNSORTED && SHIFT < u32::BITS as u8,
            "an embedded minimum-item summary shift must be in 1..=31"
        );
    }

    builder.leaf_sorter = Some(sort_leaf_scratch_by_item::<f64, u32, K>);
    builder.stem_summary_populator =
        Some(populate_f64_donnelly3_min_item_summaries::<SS, LS, K, B, SHIFT>);
    KdTreeBuilder {
        policy: builder.policy,
        leaf_sorter: builder.leaf_sorter,
        stem_summary_populator: builder.stem_summary_populator,
        _phantom: std::marker::PhantomData,
    }
}

impl<SS, LS, const K: usize, const B: usize, P, const ITEM_LEAF_MODE: u8>
    KdTreeBuilder<f64, u32, SS, LS, K, B, P, ITEM_LEAF_MODE>
where
    SS: DonnellyBlock3SummaryLayout,
    LS: LeafStrategy<f64, u32, SS, K, B, Mutability = Immutable>,
{
    /// Sorts every leaf by item and embeds subtree-minimum summaries.
    ///
    /// `SHIFT` is stored in the tree's item-leaf mode and must be in `1..=31`;
    /// `0` denotes unsorted leaves and `255` denotes sorted leaves
    /// without embedded summaries. The value is a const generic because the
    /// item-leaf mode is part of the resulting [`KdTree`] type.
    ///
    /// Each Donnelly block's 64-bit padding slot stores eight child codes in
    /// child-index order, from the least-significant byte upward. Each non-empty
    /// code is the child subtree's minimum item shifted right by `SHIFT`,
    /// saturated at `253`. Code `254` marks an empty subtree. This gives a
    /// bucket width of `2^SHIFT` while collapsing all minima at or above
    /// `253 * 2^SHIFT` into one conservative high bucket.
    ///
    /// This mode is currently available only for `u32` items with immutable
    /// leaves and the padding-compatible public `f64` Donnelly block-height-3
    /// stem strategies.
    ///
    /// ```rust
    /// use kiddo::leaf_strategy::FlatVec;
    /// use kiddo::{Donnelly, ItemLeafMode, KdTree};
    ///
    /// type Tree = KdTree<f64, u32, Donnelly<3>, FlatVec<f64, u32, 2, 8>, 2, 8>;
    ///
    /// let tree = Tree::builder()
    ///     .with_embedded_min_item_shifted_summary::<8>()
    ///     .build_from_slice(&[[0.0, 1.0], [2.0, 3.0]])
    ///     .unwrap();
    ///
    /// assert_eq!(
    ///     tree.item_leaf_mode(),
    ///     ItemLeafMode::SortedWithEncodedMin {
    ///         shift: 8,
    ///     }
    /// );
    /// ```
    ///
    /// ```compile_fail
    /// use kiddo::leaf_strategy::VecOfArrays;
    /// use kiddo::{Donnelly, KdTree};
    ///
    /// type MutableTree =
    ///     KdTree<f64, u32, Donnelly<3>, VecOfArrays<f64, u32, 2, 8>, 2, 8>;
    ///
    /// let _ = MutableTree::builder()
    ///     .with_embedded_min_item_shifted_summary::<8>();
    /// ```
    ///
    /// ```compile_fail
    /// use kiddo::leaf_strategy::FlatVec;
    /// use kiddo::{Donnelly, KdTree};
    ///
    /// type Tree = KdTree<f64, u32, Donnelly<3>, FlatVec<f64, u32, 2, 8>, 2, 8>;
    ///
    /// let _ = Tree::builder()
    ///     .with_embedded_min_item_shifted_summary::<0>();
    /// ```
    ///
    /// ```compile_fail
    /// use kiddo::leaf_strategy::FlatVec;
    /// use kiddo::{Donnelly, KdTree};
    ///
    /// type UnsupportedTree =
    ///     KdTree<f64, u32, Donnelly<4>, FlatVec<f64, u32, 2, 8>, 2, 8>;
    ///
    /// let _ = UnsupportedTree::builder()
    ///     .with_embedded_min_item_shifted_summary::<8>();
    /// ```
    pub fn with_embedded_min_item_shifted_summary<const SHIFT: u8>(
        self,
    ) -> KdTreeBuilder<f64, u32, SS, LS, K, B, P, SHIFT> {
        enable_embedded_min_item_summary(self)
    }
}

fn populate_stem_summaries_if_configured<
    A,
    T,
    SS,
    LS,
    const K: usize,
    const B: usize,
    const ITEM_LEAF_MODE: u8,
>(
    mut tree: KdTree<A, T, SS, LS, K, B, ITEM_LEAF_MODE>,
    populator: Option<StemSummaryPopulator<A, LS>>,
) -> KdTree<A, T, SS, LS, K, B, ITEM_LEAF_MODE>
where
    A: Axis<Coord = A>,
    T: Content,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
{
    if let Some(populator) = populator {
        populator(
            &mut tree.stems,
            &tree.leaves,
            &tree.stem_leaf_resolution,
            tree.max_stem_level,
        );
        tree.maybe_enable_huge_pages();
    }
    tree
}

impl<A, T, SS, LS, const K: usize, const B: usize> Default
    for KdTreeBuilder<A, T, SS, LS, K, B, SerialConstruction>
{
    fn default() -> Self {
        Self {
            policy: SerialConstruction,
            leaf_sorter: None,
            stem_summary_populator: None,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<A, T, SS, LS, const K: usize, const B: usize, const ITEM_LEAF_MODE: u8>
    KdTreeBuilder<A, T, SS, LS, K, B, SerialConstruction, ITEM_LEAF_MODE>
where
    A: Axis<Coord = A>,
    T: Content,
    SS: StemStrategy,
    LS: ConstructibleLeafStrategy<A, T, SS, K, B>,
{
    /// Builds a tree from points, using source indices as items.
    pub fn build_from_slice(
        self,
        source: &[[A; K]],
    ) -> Result<KdTree<A, T, SS, LS, K, B, ITEM_LEAF_MODE>, ConstructionError>
    where
        T: TryFrom<usize>,
    {
        validate_auto_generated_items::<T>(source.len())?;
        let tree = KdTree::new_from_source_with(
            source,
            |point: &[A; K], dim| point[dim],
            |src_idx: usize, _point: &[A; K]| {
                T::try_from(src_idx).map_err(|_| {
                    ConstructionError::AutoGeneratedItemIndexOverflow {
                        item_count: source.len(),
                        item_type: core::any::type_name::<T>(),
                    }
                })
            },
            self.leaf_sorter,
        )?;
        Ok(populate_stem_summaries_if_configured(
            tree,
            self.stem_summary_populator,
        ))
    }

    /// Builds a tree from a generic source and coordinate/item accessors.
    pub fn build_from_source<X, FA, FI>(
        self,
        source: &[X],
        axis_at: FA,
        item_at: FI,
    ) -> Result<KdTree<A, T, SS, LS, K, B, ITEM_LEAF_MODE>, ConstructionError>
    where
        FA: Fn(&X, usize) -> A,
        FI: Fn(usize, &X) -> T,
    {
        let tree = KdTree::new_from_source_with(
            source,
            axis_at,
            |src_idx, src| Ok(item_at(src_idx, src)),
            self.leaf_sorter,
        )?;
        Ok(populate_stem_summaries_if_configured(
            tree,
            self.stem_summary_populator,
        ))
    }

    /// Builds a tree from explicit item/point pairs.
    pub fn build_from_entries(
        self,
        source: &[(T, [A; K])],
    ) -> Result<KdTree<A, T, SS, LS, K, B, ITEM_LEAF_MODE>, ConstructionError> {
        self.build_from_source(
            source,
            |entry: &(T, [A; K]), dim| entry.1[dim],
            |_src_idx, entry: &(T, [A; K])| entry.0,
        )
    }
}

impl<A, SS, LS, const K: usize, const B: usize, const ITEM_LEAF_MODE: u8>
    KdTreeBuilder<A, (), SS, LS, K, B, SerialConstruction, ITEM_LEAF_MODE>
where
    A: Axis<Coord = A>,
    SS: StemStrategy,
    LS: ConstructibleLeafStrategy<A, (), SS, K, B>,
{
    /// Builds a tree with no stored items.
    pub fn build_from_slice_no_items(
        self,
        source: &[[A; K]],
    ) -> Result<KdTree<A, (), SS, LS, K, B, ITEM_LEAF_MODE>, ConstructionError> {
        self.build_from_source(
            source,
            |point: &[A; K], dim| point[dim],
            |_src_idx, _point| (),
        )
    }
}

#[cfg(feature = "multi-threaded")]
impl<A, T, SS, LS, const K: usize, const B: usize> Default
    for KdTreeBuilder<A, T, SS, LS, K, B, ParallelConstruction>
{
    fn default() -> Self {
        Self {
            policy: ParallelConstruction::with_threshold(DEFAULT_PARALLEL_CONSTRUCTION_THRESHOLD),
            leaf_sorter: None,
            stem_summary_populator: None,
            _phantom: std::marker::PhantomData,
        }
    }
}

#[cfg(feature = "multi-threaded")]
impl<A, T, SS, LS, const K: usize, const B: usize, const ITEM_LEAF_MODE: u8>
    KdTreeBuilder<A, T, SS, LS, K, B, ParallelConstruction, ITEM_LEAF_MODE>
where
    A: Axis<Coord = A> + Send + Sync,
    T: Content,
    SS: StemStrategy,
    LS: ConstructibleLeafStrategy<A, T, SS, K, B>,
{
    /// Builds a tree from points, using source indices as items.
    pub fn build_from_slice(
        self,
        source: &[[A; K]],
    ) -> Result<KdTree<A, T, SS, LS, K, B, ITEM_LEAF_MODE>, ConstructionError>
    where
        T: TryFrom<usize>,
    {
        validate_auto_generated_items::<T>(source.len())?;
        let tree = KdTree::new_from_source_with_parallel_policy(
            source,
            |point: &[A; K], dim| point[dim],
            |src_idx: usize, _point: &[A; K]| {
                T::try_from(src_idx).map_err(|_| {
                    ConstructionError::AutoGeneratedItemIndexOverflow {
                        item_count: source.len(),
                        item_type: core::any::type_name::<T>(),
                    }
                })
            },
            self.policy,
            self.leaf_sorter,
        )?;
        Ok(populate_stem_summaries_if_configured(
            tree,
            self.stem_summary_populator,
        ))
    }

    /// Builds a tree from a generic source and coordinate/item accessors.
    pub fn build_from_source<X, FA, FI>(
        self,
        source: &[X],
        axis_at: FA,
        item_at: FI,
    ) -> Result<KdTree<A, T, SS, LS, K, B, ITEM_LEAF_MODE>, ConstructionError>
    where
        X: Sync,
        FA: Fn(&X, usize) -> A + Sync,
        FI: Fn(usize, &X) -> T,
    {
        let tree = KdTree::new_from_source_with_parallel_policy(
            source,
            axis_at,
            |src_idx, src| Ok(item_at(src_idx, src)),
            self.policy,
            self.leaf_sorter,
        )?;
        Ok(populate_stem_summaries_if_configured(
            tree,
            self.stem_summary_populator,
        ))
    }

    /// Builds a tree from explicit item/point pairs.
    pub fn build_from_entries(
        self,
        source: &[(T, [A; K])],
    ) -> Result<KdTree<A, T, SS, LS, K, B, ITEM_LEAF_MODE>, ConstructionError>
    where
        T: Sync,
    {
        self.build_from_source(
            source,
            |entry: &(T, [A; K]), dim| entry.1[dim],
            |_src_idx, entry: &(T, [A; K])| entry.0,
        )
    }
}

#[cfg(feature = "multi-threaded")]
impl<A, SS, LS, const K: usize, const B: usize, const ITEM_LEAF_MODE: u8>
    KdTreeBuilder<A, (), SS, LS, K, B, ParallelConstruction, ITEM_LEAF_MODE>
where
    A: Axis<Coord = A> + Send + Sync,
    SS: StemStrategy,
    LS: ConstructibleLeafStrategy<A, (), SS, K, B>,
{
    /// Builds a tree with no stored items.
    pub fn build_from_slice_no_items(
        self,
        source: &[[A; K]],
    ) -> Result<KdTree<A, (), SS, LS, K, B, ITEM_LEAF_MODE>, ConstructionError> {
        self.build_from_source(
            source,
            |point: &[A; K], dim| point[dim],
            |_src_idx, _point| (),
        )
    }
}

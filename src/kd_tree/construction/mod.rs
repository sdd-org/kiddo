use aligned_vec::{avec, AVec, ConstAlign, CACHELINE_ALIGN};
use nonmax::NonMaxUsize;
use std::ptr::NonNull;

use crate::kd_tree::{ConstructionError, KdTreeQueryOps, MutationError, OwnedStemLeafResolution};
use crate::traits::leaf_strategy::{
    BucketLimitType, ConstructibleLeafStrategy, LeafStrategy, Mutability, MutableLeafStrategy,
};
use crate::{Axis, Content, KdTree, StemStrategy};

use super::builder::KdTreeBuilder;

mod parallel;
mod serial;
mod shared;

pub use parallel::ParallelConstruction;
pub use serial::SerialConstruction;
pub(in crate::kd_tree) use shared::validate_auto_generated_items;
use shared::{
    construction_index_fits_u32, ConstructionIndex, ConstructionLeafScratch, SoftConstructionMode,
};

/// Point-count threshold used by the default adaptive construction policy.
pub const DEFAULT_PARALLEL_CONSTRUCTION_THRESHOLD: usize = 262_144;

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: Axis<Coord = A>,
    T: Content,
    SS: StemStrategy,
    LS: MutableLeafStrategy<A, T, SS, K, B>,
{
    /// Adds a point and associated item to the tree.
    ///
    /// If the target leaf is full, it will be split before insertion.
    pub fn add(&mut self, point: &[A; K], item: T) -> Result<(), ConstructionError> {
        // Find the target leaf
        let (stem_strat, parent_stem_idx, is_right_child) = self.find_leaf_with_context(point);
        let leaf_idx = match &self.stem_leaf_resolution {
            OwnedStemLeafResolution::Mapped { leaf_idx_map, .. } => {
                leaf_idx_map[stem_strat.stem_idx()].unwrap().get()
            }
            _ => stem_strat.leaf_idx(),
        };

        if !self.leaves.is_leaf_full(leaf_idx) {
            self.leaves.add_to_leaf(leaf_idx, point, item);
            self.size += 1;
            return Ok(());
        }

        // println!("Leaf {leaf_idx} is full, splitting. {self}");

        // Leaf is full, need to split
        let (pivot_val, split_dim, new_leaf_idx) =
            self.split_leaf(leaf_idx, stem_strat, parent_stem_idx, is_right_child)?;

        // determine which leaf we belong in after the split
        let leaf_idx = if point[split_dim] >= pivot_val {
            new_leaf_idx
        } else {
            leaf_idx
        };

        self.leaves.add_to_leaf(leaf_idx, point, item);
        self.size += 1;
        Ok(())
    }

    /// Find the leaf for a query point, along with context needed for splitting.
    /// Returns: (stem_strategy, parent_stem_idx, is_right_child)
    fn find_leaf_with_context(&self, query: &[A; K]) -> (SS, Option<NonMaxUsize>, bool) {
        let stems_ptr = NonNull::new(self.stems.as_ptr() as *mut u8).unwrap();
        let mut stem_strat: SS = SS::new(stems_ptr);
        let mut parent_stem_idx: Option<NonMaxUsize> = None;
        let mut is_right_child = false;

        while stem_strat.level() <= self.max_stem_level {
            let stem_idx = stem_strat.stem_idx();

            // Check if this stem points directly to a leaf (only for Mapped)
            if let Some(_leaf_idx) = self.resolve_terminal_stem(stem_idx) {
                return (stem_strat, parent_stem_idx, is_right_child);
            }

            parent_stem_idx = Some(NonMaxUsize::new(stem_idx).unwrap());
            let pivot = unsafe { self.stems.get_unchecked(stem_idx) };
            is_right_child = unsafe { *query.get_unchecked(stem_strat.dim::<K>()) } >= *pivot;
            stem_strat.traverse::<A, K>(is_right_child);
        }

        (stem_strat, parent_stem_idx, is_right_child)
    }

    /// Split a full leaf, moving some points in the existing leaf to a new one.
    /// Updates the stem tree to contain the new pivot value, pointing to the existing and
    /// split-off leaf.
    ///
    /// Returns the dimension along which the split occurred and the value of the pivot, as well
    /// as the new leaf index.
    fn split_leaf(
        &mut self,
        leaf_idx: usize,
        stem_strategy: SS,
        _parent_stem_idx: Option<NonMaxUsize>,
        _is_right_child: bool,
    ) -> Result<(A, usize, usize), ConstructionError> {
        let old_leaf_idx = leaf_idx; // stem_strategy.leaf_idx();
        let split_dim = stem_strategy.dim::<K>();

        // Split the leaf
        let (pivot_val, new_leaf_idx) = self.leaves.split_leaf(old_leaf_idx, split_dim)?;

        // Get the indices of the children of the stem at which the split occurs
        let (left_child_idx, right_child_idx) = stem_strategy.child_indices::<A>();
        let stem_idx = stem_strategy.stem_idx();

        // Ensure the stem array is large enough
        if self.stems.len() < stem_idx + 1 {
            self.stems.resize(stem_idx + 1, A::max_value());
            crate::huge_pages::maybe_advise_slice_huge_pages(self.stems.as_ptr(), self.stems.len());
        }

        self.stems[stem_idx] = pivot_val;

        // Update the leaf_idx_map to point children to the two leaves
        if let OwnedStemLeafResolution::Mapped { leaf_idx_map, .. } = &mut self.stem_leaf_resolution
        {
            // Ensure the map is large enough
            if leaf_idx_map.len() < right_child_idx + 1 {
                leaf_idx_map.resize(right_child_idx + 1, None);
            }

            // Map left child to old leaf
            leaf_idx_map[left_child_idx] = leaf_idx_map[stem_idx];
            // Clear the root's mapping (it's now an interior node, not a leaf)
            leaf_idx_map[stem_idx] = None;
            // Map right child to new leaf
            leaf_idx_map[right_child_idx] = NonMaxUsize::new(new_leaf_idx);
        }

        // Track actual deepest interior stem level reached by splits.
        // Splitting a leaf at level L converts that terminal stem into an interior pivot.
        self.max_stem_level = self.max_stem_level.max(stem_strategy.level());

        Ok((pivot_val, split_dim, new_leaf_idx))
    }

    /*    /// Transition from Pristine to Mapped state on first split
    #[allow(unused)]
    fn taint_if_pristine(
        &mut self,
        new_stem_idx: usize,
        _left_leaf_idx: usize,
        _right_leaf_idx: usize,
        parent_stem_idx: Option<usize>,
    ) {
        match &self.stem_leaf_resolution {
            OwnedStemLeafResolution::Pristine {
                stems_depth,
                leaf_count,
            } => {
                // Transition to Mapped
                let min_stem_leaf_idx = 1 << *stems_depth;
                let mut leaf_idx_map = vec![None; self.stems.len()];

                // Map all existing leaves using arithmetic
                for i in 0..*leaf_count {
                    let stem_idx = min_stem_leaf_idx + i;
                    if stem_idx < leaf_idx_map.len() {
                        leaf_idx_map[stem_idx - min_stem_leaf_idx] = NonMaxUsize::new(i);
                    }
                }

                // Update mapping for the new stem and leaves
                if let Some(parent_idx) = parent_stem_idx {
                    // Clear parent's mapping (it now has children)
                    if parent_idx >= min_stem_leaf_idx {
                        leaf_idx_map[parent_idx - min_stem_leaf_idx] = None;
                    }
                }

                // New stem points to the two leaves
                if new_stem_idx >= min_stem_leaf_idx {
                    let idx = new_stem_idx - min_stem_leaf_idx;
                    if idx >= leaf_idx_map.len() {
                        leaf_idx_map.resize(idx + 1, None);
                    }
                }

                self.stem_leaf_resolution = OwnedStemLeafResolution::Mapped {
                    min_stem_leaf_idx,
                    leaf_idx_map,
                };
            }
            OwnedStemLeafResolution::Mapped { .. } => {
                // Already mapped, just update the mapping
                // TODO: implement mapping updates
            }
            _ => {
                // Arithmetic/Immutable - should not be calling this
                panic!("Cannot split leaves in immutable tree");
            }
        }
    }*/

    /// Removes a point and associated item from the tree.
    ///
    /// Note: This does not rebalance the tree.
    pub fn remove(&mut self, point: &[A; K], item: T) {
        let leaf_idx = self.get_leaf_idx(point);
        let old_leaf_len = self.leaves.leaf_len(leaf_idx);

        self.leaves.remove_from_leaf(leaf_idx, point, item);
        let new_leaf_len = self.leaves.leaf_len(leaf_idx);
        self.size -= old_leaf_len - new_leaf_len;

        // TODO: attempt to prune leaf if now empty
    }
}

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: Axis<Coord = A>,
    T: Content + PartialEq,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
{
    /// Replaces the first exact `(point, old_item)` match with `new_item`.
    ///
    /// Returns [`MutationError::EntryNotFound`] if the target leaf contains no
    /// entry whose point and item both match exactly.
    pub fn replace_item(
        &mut self,
        point: &[A; K],
        old_item: T,
        new_item: T,
    ) -> Result<(), MutationError> {
        let leaf_idx = self.get_leaf_idx(point);

        self.leaves
            .replace_item_in_leaf(leaf_idx, point, old_item, new_item)
            .then_some(())
            .ok_or(MutationError::EntryNotFound)
    }
}

// Shared construction implementation (works for both Immutable and Mutable)
impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: Axis<Coord = A>,
    T: Content,
    SS: StemStrategy,
    LS: ConstructibleLeafStrategy<A, T, SS, K, B>,
{
    /// Returns a builder for configuring tree construction.
    pub fn builder() -> KdTreeBuilder<A, T, SS, LS, K, B> {
        KdTreeBuilder::default()
    }

    /// Creates a `KdTree` from a slice of points.
    ///
    /// Items are auto-generated from the point index in the input slice using
    /// `T::try_from(index)`. This is the most convenient constructor when the
    /// caller only has points and is happy for item values to mirror their
    /// original position in the source slice.
    ///
    /// Returns [`ConstructionError::AutoGeneratedItemIndexOverflow`] if the
    /// source length cannot be represented by `T`.
    ///
    /// Construction is serial below
    /// [`DEFAULT_PARALLEL_CONSTRUCTION_THRESHOLD`] points and parallel at or
    /// above it. Use [`KdTree::builder`] to select a different threshold or
    /// force either construction mode.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    /// use kiddo::leaf_strategy::FlatVec;
    /// use kiddo::Eytzinger;
    ///
    /// let points = vec![
    ///     [1.0f64, 2.0f64, 3.0f64],
    ///     [4.0f64, 5.0f64, 6.0f64],
    /// ];
    ///
    /// let tree: KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, 3, 32>, 3, 32> =
    ///     KdTree::new_from_slice(&points).unwrap();
    ///
    /// assert_eq!(tree.size(), 2);
    /// assert_eq!(
    ///     tree.iter().collect::<Vec<_>>(),
    ///     vec![(0u32, [1.0, 2.0, 3.0]), (1u32, [4.0, 5.0, 6.0])]
    /// );
    /// ```
    #[cfg_attr(not(feature = "no_inline"), inline)]
    // TODO: Add checked, Result-returning ingress APIs (`new_from_slice`,
    // `new_from_slice_no_items`, and mutable `add`) that reject coordinates
    // equal to `A::max_value()`, and rename the current behavior to
    // `*_unchecked`.
    pub fn new_from_slice(source: &[[A; K]]) -> Result<Self, ConstructionError>
    where
        A: Send + Sync,
        T: TryFrom<usize>,
    {
        Self::builder().build_from_slice(source)
    }

    /// Creates a `KdTree` from a slice of points using parallel construction.
    ///
    /// This compatibility convenience constructor uses the same thresholded
    /// policy as [`KdTree::new_from_slice`]. Use
    /// [`KdTreeBuilder::with_parallel_construction`] to force the parallel
    /// algorithm below the default threshold.
    ///
    /// This method runs on the current Rayon thread pool. Use
    /// [`rayon::ThreadPool::install`] when a caller needs to select a specific
    /// thread count.
    pub fn new_from_slice_parallel(source: &[[A; K]]) -> Result<Self, ConstructionError>
    where
        A: Send + Sync,
        T: TryFrom<usize>,
    {
        Self::builder()
            .with_parallel_construction_threshold(DEFAULT_PARALLEL_CONSTRUCTION_THRESHOLD)
            .build_from_slice(source)
    }

    /// Creates a `KdTree` from a generic slice source plus axis/item accessors.
    ///
    /// This is the most general bulk-construction ingress API. Callers provide
    /// one callback to read the coordinate value for a source item and
    /// dimension, and another callback to produce the stored item value.
    ///
    /// Use this when your source data is not already shaped as `&[[A; K]]` or
    /// `&[(T, [A; K])]`, for example when points and IDs live in fields on a
    /// custom struct.
    ///
    /// The default adaptive construction policy requires the source and
    /// coordinate accessor to be thread-safe. For a non-[`Sync`] source, use
    /// [`KdTree::builder`], select
    /// [`KdTreeBuilder::with_serial_construction`], and call
    /// [`KdTreeBuilder::build_from_source`].
    ///
    /// The `axis_at` accessor is on the hot path during construction. It is
    /// called exactly `n * k` times to materialize leaf storage for a tree
    /// with `n` source items and dimensionality `k`, plus additional calls
    /// during recursive pivot selection and partitioning. In practice, total
    /// accessor usage is roughly `n * k + O(n log(n))`, so expensive accessors
    /// can noticeably slow down construction. When your data is already
    /// available as a `&[[A; K]]`, prefer [`KdTree::new_from_slice`] to avoid
    /// that extra accessor overhead.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    /// use kiddo::leaf_strategy::FlatVec;
    /// use kiddo::Eytzinger;
    ///
    /// #[derive(Clone, Copy)]
    /// struct Point3D {
    ///     id: u32,
    ///     x: f32,
    ///     y: f32,
    ///     z: f32,
    ///     w: f32,
    /// }
    ///
    /// let points = [
    ///     Point3D { id: 10, x: 1.0, y: 2.0, z: 3.0, w: 0.5 },
    ///     Point3D { id: 20, x: 4.0, y: 5.0, z: 6.0, w: 0.7 },
    /// ];
    ///
    /// let tree: KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, 3, 32>, 3, 32> =
    ///     KdTree::new_from_source(
    ///         &points,
    ///         |point, dim| match dim {
    ///             0 => point.x,
    ///             1 => point.y,
    ///             2 => point.z,
    ///             _ => unreachable!(),
    ///         },
    ///         |_idx, point| point.id,
    ///     )
    ///     .unwrap();
    ///
    /// assert_eq!(
    ///     tree.iter().collect::<Vec<_>>(),
    ///     vec![(10u32, [1.0, 2.0, 3.0]), (20u32, [4.0, 5.0, 6.0])]
    /// );
    /// ```
    #[cfg_attr(not(feature = "no_inline"), inline)]
    pub fn new_from_source<X, FA, FI>(
        source: &[X],
        axis_at: FA,
        item_at: FI,
    ) -> Result<Self, ConstructionError>
    where
        A: Send + Sync,
        X: Sync,
        FA: Fn(&X, usize) -> A + Sync,
        FI: Fn(usize, &X) -> T,
    {
        Self::builder().build_from_source(source, axis_at, item_at)
    }

    /// Creates a `KdTree` from a generic source using parallel construction.
    ///
    /// This compatibility convenience constructor uses the same thresholded
    /// policy as [`KdTree::new_from_source`]. Coordinate access must be
    /// thread-safe because partitioning may invoke `axis_at` concurrently.
    /// Item construction remains sequential.
    pub fn new_from_source_parallel<X, FA, FI>(
        source: &[X],
        axis_at: FA,
        item_at: FI,
    ) -> Result<Self, ConstructionError>
    where
        A: Send + Sync,
        X: Sync,
        FA: Fn(&X, usize) -> A + Sync,
        FI: Fn(usize, &X) -> T,
    {
        Self::builder()
            .with_parallel_construction_threshold(DEFAULT_PARALLEL_CONSTRUCTION_THRESHOLD)
            .build_from_source(source, axis_at, item_at)
    }

    /// Creates a `KdTree` from explicit item/point pairs.
    ///
    /// This is the preferred ingress when callers already have items rather
    /// than wanting `new_from_slice` to auto-generate them from source indices.
    ///
    /// Unlike [`KdTree::new_from_slice`], item values are taken directly from
    /// the input rather than derived from position in the slice.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    /// use kiddo::leaf_strategy::FlatVec;
    /// use kiddo::Eytzinger;
    ///
    /// let entries = vec![
    ///     (42u32, [0.0f32, 1.0f32]),
    ///     (7u32, [2.0f32, 3.0f32]),
    /// ];
    ///
    /// let tree: KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, 2, 32>, 2, 32> =
    ///     KdTree::new_from_entries(&entries).unwrap();
    ///
    /// assert_eq!(tree.size(), 2);
    /// assert_eq!(tree.iter().collect::<Vec<_>>(), entries);
    /// ```
    #[cfg_attr(not(feature = "no_inline"), inline)]
    pub fn new_from_entries(source: &[(T, [A; K])]) -> Result<Self, ConstructionError>
    where
        A: Send + Sync,
        T: Sync,
    {
        Self::builder().build_from_entries(source)
    }

    /// Creates a `KdTree` from item/point pairs using the default thresholded
    /// parallel construction policy.
    pub fn new_from_entries_parallel(source: &[(T, [A; K])]) -> Result<Self, ConstructionError>
    where
        A: Send + Sync,
        T: Sync,
    {
        Self::builder()
            .with_parallel_construction_threshold(DEFAULT_PARALLEL_CONSTRUCTION_THRESHOLD)
            .build_from_entries(source)
    }

    /// Inner constructor shared by all variants. The accessors are invoked
    /// wherever we would normally pull coordinates or items from the source.
    pub(in crate::kd_tree) fn new_from_source_with<X, FA, FI>(
        source: &[X],
        axis_at: FA,
        item_at: FI,
    ) -> Result<Self, ConstructionError>
    where
        FA: Fn(&X, usize) -> A,
        FI: FnMut(usize, &X) -> Result<T, ConstructionError>,
    {
        let item_count = source.len();
        let leaf_node_count = item_count.div_ceil(B);

        if leaf_node_count < 2 {
            return Self::new_from_source_no_stems_with(source, &axis_at, item_at);
        }

        if construction_index_fits_u32(item_count) {
            Self::new_from_source_with_index::<SerialConstruction, u32, _, _, _>(
                source,
                axis_at,
                item_at,
                &SerialConstruction,
            )
        } else {
            Self::new_from_source_with_index::<SerialConstruction, usize, _, _, _>(
                source,
                axis_at,
                item_at,
                &SerialConstruction,
            )
        }
    }

    pub(in crate::kd_tree) fn new_from_source_with_parallel_policy<X, FA, FI>(
        source: &[X],
        axis_at: FA,
        item_at: FI,
        policy: ParallelConstruction,
    ) -> Result<Self, ConstructionError>
    where
        A: Send + Sync,
        X: Sync,
        FA: Fn(&X, usize) -> A + Sync,
        FI: FnMut(usize, &X) -> Result<T, ConstructionError>,
    {
        let item_count = source.len();
        let leaf_node_count = item_count.div_ceil(B);

        if leaf_node_count < 2 {
            return Self::new_from_source_no_stems_with(source, &axis_at, item_at);
        }

        if construction_index_fits_u32(item_count) {
            Self::new_from_source_with_index::<ParallelConstruction, u32, _, _, _>(
                source, axis_at, item_at, &policy,
            )
        } else {
            Self::new_from_source_with_index::<ParallelConstruction, usize, _, _, _>(
                source, axis_at, item_at, &policy,
            )
        }
    }

    fn new_from_source_with_index<M, I, X, FA, FI>(
        source: &[X],
        axis_at: FA,
        mut item_at: FI,
        mode: &M,
    ) -> Result<Self, ConstructionError>
    where
        I: ConstructionIndex,
        FA: Fn(&X, usize) -> A,
        FI: FnMut(usize, &X) -> Result<T, ConstructionError>,
        M: SoftConstructionMode<A, T, SS, LS, I, X, FA, FI, K, B>,
    {
        let item_count = source.len();
        let leaf_node_count = item_count.div_ceil(B);
        let mut stems_depth: usize = leaf_node_count.next_power_of_two().ilog2() as usize;

        // Pad stem tree height to the next block boundary for block-based strategies
        let padding_level_count = if !stems_depth.is_multiple_of(SS::block_size()) {
            let padding_level_count = SS::block_size() - (stems_depth % SS::block_size());
            stems_depth += padding_level_count;
            padding_level_count
        } else {
            0
        };

        // Padding levels will be placed at the root of the tree. Pre-traverse any padding levels
        // so that stem_strat is set to the location where the true root will be
        let mut stem_strat = SS::new_no_ptr();
        for _ in 0..padding_level_count {
            stem_strat.traverse::<A, K>(false);
        }
        let root_stem_strat = stem_strat.clone();

        let soft_leaf_budget = if LS::BUCKET_LIMIT_TYPE == BucketLimitType::Soft {
            1usize << stems_depth
        } else {
            leaf_node_count
        };

        // Traverse to the right-most represented leaf to determine the max used stem index
        let rightmost_leaf_idx = soft_leaf_budget - 1;
        let rightmost_leaf_bit_range = if LS::BUCKET_LIMIT_TYPE == BucketLimitType::Soft {
            0..stems_depth
        } else {
            1..stems_depth
        };
        for bit_idx in rightmost_leaf_bit_range.rev() {
            let is_right = rightmost_leaf_idx & (1 << bit_idx) != 0;
            stem_strat.traverse::<A, K>(is_right);
        }
        let stem_node_count = stem_strat.stem_idx() + 1;

        // rounded up to the nearest multiple of 8 if not a multiple of 8 already
        let stem_node_count_padded = stem_node_count.div_ceil(8) * 8;
        let mut stems = avec![A::max_value(); stem_node_count_padded];

        let mut leaves = LS::new_with_capacity(item_count);
        let mut terminal_stem_indices = Vec::with_capacity(leaf_node_count);
        let mut actual_max_stem_level: i32 = -1;
        let mut max_leaf_len = 0usize;
        let mut leaf_scratch = ConstructionLeafScratch::<A, T, K>::with_capacity(B);
        let mut sort_index = Vec::from_iter((0..item_count).map(I::from_usize));

        match LS::BUCKET_LIMIT_TYPE {
            BucketLimitType::Hard => Self::populate_recursive_hard(
                &mut stems,
                source,
                &axis_at,
                &mut sort_index,
                root_stem_strat,
                stems_depth as i32 - 1,
                leaf_node_count * B,
                &mut leaves,
                &mut terminal_stem_indices,
                &mut actual_max_stem_level,
                &mut max_leaf_len,
                &mut leaf_scratch,
                &mut item_at,
            )?,
            BucketLimitType::Soft => mode.populate(
                &mut stems,
                source,
                &axis_at,
                &mut sort_index,
                root_stem_strat,
                stems_depth as i32 - 1,
                soft_leaf_budget,
                &mut leaves,
                &mut actual_max_stem_level,
                &mut max_leaf_len,
                &mut leaf_scratch,
                &mut item_at,
            )?,
        }

        let initial_max_stem_level = stems_depth as i32 - 1;
        let requires_mapped_resolution = LS::Mutability::is_mutable()
            || LS::BUCKET_LIMIT_TYPE == BucketLimitType::Hard
                && (actual_max_stem_level > initial_max_stem_level
                    || padding_level_count != 0
                    || !Self::terminal_stem_indices_match_arithmetic_layout(
                        &terminal_stem_indices,
                        actual_max_stem_level,
                    ));

        let stem_leaf_resolution = if requires_mapped_resolution {
            Self::mapped_stem_leaf_resolution_from_terminals(&terminal_stem_indices)
        } else {
            LS::Mutability::initial_stem_leaf_resolution::<A, SS, K>(
                stems_depth,
                leaves.leaf_count(),
            )
        };

        crate::leaf_view::assert_leaf_scratch_capacity(max_leaf_len);

        let tree = Self {
            stems,
            leaves,
            stem_leaf_resolution,
            size: item_count,
            max_stem_level: actual_max_stem_level,
            max_leaf_len,
            _phantom: Default::default(),
        };
        tree.maybe_enable_huge_pages();
        Ok(tree)
    }
    fn new_from_source_no_stems_with<X, FA, FI>(
        source: &[X],
        axis_at: &FA,
        mut item_at: FI,
    ) -> Result<Self, ConstructionError>
    where
        FA: Fn(&X, usize) -> A,
        FI: FnMut(usize, &X) -> Result<T, ConstructionError>,
    {
        let item_count = source.len();

        if item_count == 0 {
            return Ok(Self::default());
        }

        let mut leaf_points: [Vec<A>; K] =
            array_init::array_init(|_| Vec::with_capacity(item_count));
        let mut leaf_items: Vec<T> = Vec::with_capacity(item_count);

        for idx in 0..item_count {
            for dim in 0..K {
                leaf_points[dim].push(axis_at(&source[idx], dim));
            }
            leaf_items.push(item_at(idx, &source[idx])?);
        }

        let leaf_points_refs: [&[A]; K] = array_init::array_init(|dim| leaf_points[dim].as_slice());

        let mut leaves = LS::new_with_capacity(item_count);
        leaves.append_leaf(&leaf_points_refs, leaf_items.as_slice());

        let stem_leaf_resolution =
            LS::Mutability::initial_stem_leaf_resolution::<A, SS, K>(0, leaves.leaf_count());

        let max_leaf_len = item_count;
        crate::leaf_view::assert_leaf_scratch_capacity(max_leaf_len);

        let tree = Self {
            stems: avec![A::max_value(); 0],
            leaves,
            stem_leaf_resolution,
            size: item_count,
            max_stem_level: -1,
            max_leaf_len,
            _phantom: Default::default(),
        };
        tree.maybe_enable_huge_pages();
        Ok(tree)
    }
}

impl<A, SS, LS, const K: usize, const B: usize> KdTree<A, (), SS, LS, K, B>
where
    A: Axis<Coord = A>,
    SS: StemStrategy,
    LS: ConstructibleLeafStrategy<A, (), SS, K, B>,
{
    /// Creates a `KdTree` with no stored item values (`T = ()`).
    ///
    /// Leaf item slices will have the correct length but contain only `()`.
    /// LLVM can generally optimize the `Vec<()>` storage away.
    ///
    /// This is useful when the points themselves are the only data you need to
    /// store and query.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    /// use kiddo::leaf_strategy::FlatVec;
    /// use kiddo::Eytzinger;
    ///
    /// let points = vec![[1.0f64, 2.0f64], [3.0f64, 4.0f64]];
    ///
    /// let tree: KdTree<f64, (), Eytzinger, FlatVec<f64, (), 2, 32>, 2, 32> =
    ///     KdTree::new_from_slice_no_items(&points).unwrap();
    ///
    /// assert_eq!(tree.size(), 2);
    /// assert_eq!(
    ///     tree.iter().collect::<Vec<_>>(),
    ///     vec![((), [1.0, 2.0]), ((), [3.0, 4.0])]
    /// );
    /// ```
    #[cfg_attr(not(feature = "no_inline"), inline)]
    pub fn new_from_slice_no_items(source: &[[A; K]]) -> Result<Self, ConstructionError>
    where
        A: Send + Sync,
    {
        Self::builder().build_from_slice_no_items(source)
    }

    /// Creates a `KdTree` with no stored items using the default thresholded
    /// parallel construction policy.
    pub fn new_from_slice_no_items_parallel(source: &[[A; K]]) -> Result<Self, ConstructionError>
    where
        A: Send + Sync,
    {
        Self::builder()
            .with_parallel_construction_threshold(DEFAULT_PARALLEL_CONSTRUCTION_THRESHOLD)
            .build_from_slice_no_items(source)
    }
}

// Shared utility methods for construction (available to both Immutable and Mutable)
impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: Axis<Coord = A>,
    T: Content,
    SS: StemStrategy,
    LS: ConstructibleLeafStrategy<A, T, SS, K, B>,
{
    fn write_leaf_from_sort_index<I, X, FA, FI>(
        source: &[X],
        axis_at: &FA,
        sort_index: &[I],
        leaves: &mut LS,
        max_leaf_len: &mut usize,
        leaf_scratch: &mut ConstructionLeafScratch<A, T, K>,
        item_at: &mut FI,
    ) -> Result<(), ConstructionError>
    where
        I: ConstructionIndex,
        FA: Fn(&X, usize) -> A,
        FI: FnMut(usize, &X) -> Result<T, ConstructionError>,
    {
        let leaf_len = sort_index.len();
        *max_leaf_len = (*max_leaf_len).max(leaf_len);

        leaf_scratch.clear_and_reserve(leaf_len);

        for &src_idx in sort_index {
            let src_idx = src_idx.as_usize();
            for d in 0..K {
                leaf_scratch.points[d].push(axis_at(&source[src_idx], d));
            }
            leaf_scratch.items.push(item_at(src_idx, &source[src_idx])?);
        }

        let leaf_points_refs: [&[A]; K] =
            array_init::array_init(|d| leaf_scratch.points[d].as_slice());
        leaves.append_leaf(&leaf_points_refs, leaf_scratch.items.as_slice());

        Ok(())
    }

    #[inline(always)]
    fn soft_left_leaf_budget(leaf_budget: usize) -> usize {
        debug_assert!(leaf_budget > 1);
        1usize << ((usize::BITS - 1 - (leaf_budget - 1).leading_zeros()) as usize)
    }

    #[inline(always)]
    fn soft_ideal_pivot(chunk_length: usize, left_leaf_budget: usize, leaf_budget: usize) -> usize {
        debug_assert!(leaf_budget > 0);
        debug_assert!(left_leaf_budget < leaf_budget);

        if chunk_length == 0 {
            return 0;
        }

        chunk_length
            .saturating_mul(left_leaf_budget)
            .div_ceil(leaf_budget)
            .clamp(1, chunk_length)
    }

    /// Hard-bucket recursive construction helper.
    #[allow(clippy::too_many_arguments)]
    fn populate_recursive_hard<I, X, FA, FI>(
        stems: &mut AVec<A, ConstAlign<{ CACHELINE_ALIGN }>>,
        source: &[X],
        axis_at: &FA,
        sort_index: &mut [I],
        mut stem_ordering: SS,
        max_stem_level: i32,
        capacity: usize,
        leaves: &mut LS,
        terminal_stem_indices: &mut Vec<usize>,
        actual_max_stem_level: &mut i32,
        max_leaf_len: &mut usize,
        leaf_scratch: &mut ConstructionLeafScratch<A, T, K>,
        item_at: &mut FI,
    ) -> Result<(), ConstructionError>
    where
        I: ConstructionIndex,
        FA: Fn(&X, usize) -> A,
        FI: FnMut(usize, &X) -> Result<T, ConstructionError>,
    {
        let chunk_length = sort_index.len();
        let dim = stem_ordering.construction_dim::<K>();

        debug_assert!(
            chunk_length > 0,
            "recursed an empty chunk (stem_idx={}, level={}, chunk_length={}, capacity={})",
            stem_ordering.stem_idx(),
            stem_ordering.level(),
            chunk_length,
            capacity,
        );

        if chunk_length <= B {
            Self::write_leaf_from_sort_index(
                source,
                axis_at,
                sort_index,
                leaves,
                max_leaf_len,
                leaf_scratch,
                item_at,
            )?;
            terminal_stem_indices.push(stem_ordering.stem_idx());
            return Ok(());
        }

        let levels_below = max_stem_level - stem_ordering.level();
        let clamped_levels_below = levels_below.max(0) as u32;
        let left_capacity = (2usize.pow(clamped_levels_below) * B).min(capacity);
        let right_capacity = capacity.saturating_sub(left_capacity);

        debug_assert!(
            left_capacity > 0,
            "left_capacity is zero - should never happen (stem_idx={}, level={}, chunk_length={}, capacity={}, right_capacity={})",
            stem_ordering.stem_idx(),
            stem_ordering.level(),
            chunk_length,
            capacity,
            right_capacity
        );

        let stem_index = stem_ordering.stem_idx();
        *actual_max_stem_level = (*actual_max_stem_level).max(stem_ordering.level());

        if stem_index >= stems.len() {
            tracing::warn!(
                %stem_index,
                existing_stem_vec_len = %stems.len(),
                "encountered a stem index beyond the end of the stem vec. Growing the vec to fit"
            );

            stems.resize(stem_index + 1, A::max_value());
        }

        let mut pivot = Self::calc_pivot(
            chunk_length,
            stem_index,
            right_capacity,
            LS::BUCKET_LIMIT_TYPE,
        );

        debug_assert!(
            pivot > 0,
            "construction produced initial pivot=0 (empty-left split candidate): \
            stem_index = {}, level={}, chunk_length = {}, capacity = {}, \
            left_capacity = {}, right_capacity = {}, dim = {}",
            stem_index,
            stem_ordering.level(),
            chunk_length,
            capacity,
            left_capacity,
            right_capacity,
            dim,
        );

        // only bother with this logic if we are putting at least one item in the right-hand child
        if pivot < chunk_length {
            pivot = Self::update_pivot(source, axis_at, sort_index, dim, pivot)?;

            debug_assert!(
                pivot > 0,
                "construction produced updated pivot=0 (empty-left split candidate): \
                stem_index = {}, level={}, chunk_length = {}, capacity = {}, \
                left_capacity = {}, right_capacity = {}, dim = {}",
                stem_index,
                stem_ordering.level(),
                chunk_length,
                capacity,
                left_capacity,
                right_capacity,
                dim,
            );

            // if we end up with a pivot of 0, something has gone wrong,
            // unless we only had a slice of len 1 anyway
            // debug_assert!(
            //     pivot > 0 || chunk_length == 1,
            // );

            // if LS::BUCKET_LIMIT_TYPE == BucketLimitType::Hard {
            //     debug_assert!(
            //         right_capacity >= chunk_length.saturating_sub(pivot),
            //         "right_capacity ({right_capacity}) should be greater than chunk_length - pivot ({chunk_length} - {pivot})"
            //     );
            // }

            if pivot < chunk_length {
                debug_assert!(
                    A::Coord::is_max_value(stems[stem_index]),
                    "Wrote to stem #{stem_index:?} for a second time",
                );

                stems[stem_index] = axis_at(&source[sort_index[pivot].as_usize()], dim);
            }
        }

        let right_stem_ordering = stem_ordering.branch::<A, K>();
        let (lower_sort_index, upper_sort_index) = sort_index.split_at_mut(pivot);

        Self::populate_recursive_hard(
            stems,
            source,
            axis_at,
            lower_sort_index,
            stem_ordering,
            max_stem_level,
            left_capacity,
            leaves,
            terminal_stem_indices,
            actual_max_stem_level,
            max_leaf_len,
            leaf_scratch,
            item_at,
        )?;

        if !upper_sort_index.is_empty() {
            Self::populate_recursive_hard(
                stems,
                source,
                axis_at,
                upper_sort_index,
                right_stem_ordering,
                max_stem_level,
                right_capacity,
                leaves,
                terminal_stem_indices,
                actual_max_stem_level,
                max_leaf_len,
                leaf_scratch,
                item_at,
            )?;
        }

        Ok(())
    }

    // TODO: remove this entirely in favor of just taking ownership of terminal_stem_indices
    //       once confident that the debug_asserts never fire
    fn mapped_stem_leaf_resolution_from_terminals(
        terminal_stem_indices: &[usize],
    ) -> OwnedStemLeafResolution {
        if terminal_stem_indices.is_empty() {
            return OwnedStemLeafResolution::Mapped {
                min_stem_leaf_idx: 0,
                leaf_idx_map: Vec::new(),
            };
        }

        // TODO: this should not be needed. Just use terminal_stem_indices.len()
        let max_terminal_stem_idx = terminal_stem_indices.iter().copied().max().unwrap_or(0);
        // debug_assert!(
        //     max_terminal_stem_idx == terminal_stem_indices.len() - 1,
        //     "Leaf array should be contiguous. Construction invariant failed"
        // );
        if max_terminal_stem_idx > terminal_stem_indices.len() - 1 {
            tracing::warn!("Leaf array should be contiguous. Construction invariant failed");
        };
        let mut leaf_idx_map: Vec<Option<NonMaxUsize>> = vec![None; max_terminal_stem_idx + 1];

        for (leaf_idx, &terminal_stem_idx) in terminal_stem_indices.iter().enumerate() {
            debug_assert!(
                leaf_idx_map[terminal_stem_idx].is_none(),
                "Duplicate terminal stem index in mapped leaf_idx_map construction: stem_idx={} existing_leaf_idx={} new_leaf_idx={}",
                terminal_stem_idx,
                leaf_idx_map[terminal_stem_idx].unwrap(),
                leaf_idx
            );

            leaf_idx_map[terminal_stem_idx] = NonMaxUsize::new(leaf_idx);
        }

        OwnedStemLeafResolution::Mapped {
            min_stem_leaf_idx: 0,
            leaf_idx_map,
        }
    }

    fn terminal_stem_indices_match_arithmetic_layout(
        terminal_stem_indices: &[usize],
        actual_max_stem_level: i32,
    ) -> bool {
        let depth = (actual_max_stem_level + 1) as usize;

        for (leaf_idx, &terminal_stem_idx) in terminal_stem_indices.iter().enumerate() {
            let mut stem_ordering = SS::new_no_ptr();
            for bit_idx in (0..depth).rev() {
                let is_right = leaf_idx & (1 << bit_idx) != 0;
                stem_ordering.traverse::<A, K>(is_right);
            }

            if stem_ordering.stem_idx() != terminal_stem_idx {
                return false;
            }
        }

        true
    }

    fn calc_pivot(
        chunk_length: usize,
        _stem_index: usize,
        right_capacity: usize,
        bucket_limit_type: BucketLimitType,
    ) -> usize {
        let mut result = chunk_length
            .saturating_sub(right_capacity)
            .next_multiple_of(B)
            .min(chunk_length);

        // Treat this as the ideal split target: put at least B items on the left
        // whenever the chunk can support it.
        if chunk_length > 0 {
            let min_ideal_left = B.min(chunk_length);
            if result < min_ideal_left {
                result = min_ideal_left;
            }
        }

        // debug_assert!(
        //     result > 0 || chunk_length >= right_capacity,
        //     "Unexpectedly generated an initial pivot to split a slice at position 0 during construction (chunk length: {chunk_length}, right_capacity: {right_capacity})"
        // );

        let result = if bucket_limit_type == BucketLimitType::Hard
            && result >= chunk_length
            && result > B
        {
            let adjusted_result = chunk_length
                .saturating_sub(right_capacity.max(B))
                .next_multiple_of(B)
                .min(chunk_length);

            tracing::debug!(
                orig_pivot = %result,
                adjusted_pivot = %adjusted_result,
                %chunk_length,
                %right_capacity,
                "initial pivot calc would result in infinite recursion due to everything going in left but left being > B. Splitting finer"
            );

            adjusted_result
        } else {
            result
        };

        debug_assert!(
            result < chunk_length + 1,
            "Unexpectedly generated an initial pivot to split a slice at or beyond its end during construction (chunk length: {chunk_length}, right_capacity: {right_capacity})"
        );

        result
    }

    #[cfg_attr(not(feature = "no_inline"), inline)]
    fn update_pivot<I, X, FA>(
        source: &[X],
        axis_at: &FA,
        sort_index: &mut [I],
        dim: usize,
        init_pivot: usize,
    ) -> Result<usize, ConstructionError>
    where
        I: ConstructionIndex,
        FA: Fn(&X, usize) -> A,
    {
        // TODO: this block might be faster by using a quickselect with a fat partition?
        //       we could then run that quickselect and subtract (fat partition length - 1)
        //       from the pivot, avoiding the need for the while loop.

        let mut pivot = init_pivot;

        // ensure the item whose index = pivot is in its correctly sorted position, and any
        // items that are equal to it are adjacent, according to our assumptions about the
        // behaviour of `select_nth_unstable_by` (See examples/check_select_nth_unstable.rs)
        sort_index.select_nth_unstable_by(pivot, |&ia, &ib| {
            A::cmp(
                (*axis_at)(&source[ia.as_usize()], dim),
                (*axis_at)(&source[ib.as_usize()], dim),
            )
        });

        // if the pivot straddles two values that are equal, keep nudging it left until they aren't
        while pivot > 0
            && (*axis_at)(&source[sort_index[pivot].as_usize()], dim)
                == (*axis_at)(&source[sort_index[pivot - 1].as_usize()], dim)
        {
            pivot -= 1;
        }

        // if we nudged it all the way to the left, reset and try nudging it rightwards from the
        // initial pivot point instead. This requires that the entire slice is sorted, rather than
        // just the left-hand side
        if pivot == 0 {
            pivot = init_pivot;

            sort_index.sort_unstable_by(|&ia, &ib| {
                A::cmp(
                    (*axis_at)(&source[ia.as_usize()], dim),
                    (*axis_at)(&source[ib.as_usize()], dim),
                )
            });

            while pivot + 1 < sort_index.len()
                && (*axis_at)(&source[sort_index[pivot].as_usize()], dim)
                    == (*axis_at)(&source[sort_index[pivot + 1].as_usize()], dim)
            {
                pivot += 1;
            }

            if pivot + 1 >= sort_index.len() {
                // if we end up here at the end of the slice, then the source slice is unsplittable
                // in this dimension due to all entries having the same value on the given dimension
                tracing::debug!(
                    slice_len = %sort_index.len(),
                    %dim,
                    "Slice unsplittable along dimension"
                );

                if LS::BUCKET_LIMIT_TYPE == BucketLimitType::Hard {
                    return Err(ConstructionError::UnsplittableBucket { split_dim: dim });
                }

                pivot = sort_index.len();
            } else {
                // Avoid empty-left splits: place the boundary after the run.
                pivot += 1;
                tracing::trace!(
                    slice_len = %sort_index.len(),
                    %dim,
                    %init_pivot,
                    shift = %(pivot - init_pivot),
                    %pivot,
                    "pivot shifted right"
                );
            }
        } else if pivot != init_pivot {
            tracing::trace!(
                slice_len = %sort_index.len(),
                %dim,
                %init_pivot,
                shift = %(init_pivot - pivot),
                %pivot,
                "pivot shifted left"
            );
        }

        Ok(pivot)
    }
}

#[cfg(test)]
mod tests;

use std::num::NonZeroUsize;

use crate::dist::DistanceMetric;
use crate::kd_tree::query_context::QueryContext;
use crate::kd_tree::query_stack::StackTrait;
use crate::kd_tree::KdTreeQueryOps;
use crate::leaf_view::TlsLeafScratch;
use crate::leaf_view_chunked::nearest_n_within::{
    nearest_n_within_with_query_wide, nearest_n_within_with_query_wide_arena,
};
use crate::results::result_buffer::FixedResultCollection;
#[cfg(not(feature = "small_n_result_collectors"))]
use crate::results::result_collection::ThresholdVecResultCollection;
use crate::results::result_collection::{BinaryHeapResultCollection, ResultCollection};
#[cfg(feature = "small_n_result_collectors")]
use crate::results::result_collection::{
    SmallSortedVecResultCollection, SMALL_RESULT_COLLECTION_MAX_QTY,
};
use crate::stem_strategy::donnelly::simd_full::{
    BacktrackBlock3, BacktrackBlock4, SimdSelectBestChildBlock3,
};
use crate::traits::leaf_strategy::LeafProjection;
use crate::{Axis, Content, KdTree, LeafStrategy, QueryResultItem, StemStrategy};

#[cfg(not(feature = "small_n_result_collectors"))]
const MAX_SORTED_VEC_RESULT_SIZE: usize = 192;
#[cfg(not(feature = "small_n_result_collectors"))]
const MAX_UNSORTED_VEC_RESULT_SIZE: usize = 24;

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    LS: LeafStrategy<A, T, SS, K, B>,
    SS: StemStrategy,
{
    #[inline(always)]
    fn process_leaf_nearest_n_within<D, R, const EXCLUSIVE: bool>(
        &self,
        leaf_idx: usize,
        query_wide: &[D::Output; K],
        max_dist: D::Output,
        results: &mut R,
    ) where
        D: DistanceMetric<A>,
        D::Output: Axis<Coord = D::Output> + TlsLeafScratch + 'static,
        R: ResultCollection<D::Output, QueryResultItem<(), T, D::Output>>,
    {
        #[cfg(feature = "result_collection_stats")]
        let was_full = results.is_full();

        #[cfg(feature = "result_collection_stats")]
        if was_full {
            crate::results::result_collection_stats::record_leaf_visit_after_full();
        } else {
            crate::results::result_collection_stats::record_leaf_visit_before_full();
        }

        match LS::LEAF_PROJECTION {
            LeafProjection::LeafArena => {
                let arena = self.leaves.leaf_arena(leaf_idx);
                nearest_n_within_with_query_wide_arena::<A, T, D, R, EXCLUSIVE, K>(
                    &arena, query_wide, max_dist, results,
                );
            }
            LeafProjection::LeafView => {
                let leaf = self.leaves.leaf_view(leaf_idx);
                nearest_n_within_with_query_wide::<A, T, D, R, EXCLUSIVE, K, B>(
                    &leaf, query_wide, max_dist, results,
                );
            }
        }

        #[cfg(feature = "result_collection_stats")]
        {
            if !was_full && results.is_full() {
                crate::results::result_collection_stats::record_collection_full_transition();
            }
            crate::results::result_collection_stats::clear_leaf_phase();
        }
    }

    /// Finds up to N nearest points within a given distance.
    ///
    /// Returns up to `max_qty` points that are within `max_dist` of the query point.
    /// If `sorted` is true, results are returned in order of increasing distance.
    pub(crate) fn nearest_n_within<D>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
        max_qty: NonZeroUsize,
        sorted: bool,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    {
        self.nearest_n_within_impl::<D, false>(query, max_dist, max_qty, sorted)
    }

    pub(crate) fn nearest_n_within_with_scratch<D>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
        max_qty: NonZeroUsize,
        sorted: bool,
        stack: &mut SS::Stack<D::Output>,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS>,
    {
        self.nearest_n_within_impl_with_scratch::<D, false>(query, max_dist, max_qty, sorted, stack)
    }

    pub(crate) fn nearest_n_within_impl<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
        max_qty: NonZeroUsize,
        sorted: bool,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    {
        let max_qty: usize = max_qty.get();

        if max_qty == usize::MAX {
            self.nearest_n_within_inner::<D, Vec<QueryResultItem<(), T, D::Output>>, EXCLUSIVE>(
                query, max_dist, max_qty, sorted,
            )
        } else {
            #[cfg(feature = "small_n_result_collectors")]
            if max_qty <= SMALL_RESULT_COLLECTION_MAX_QTY {
                return self.nearest_n_within_inner::<
                    D,
                    SmallSortedVecResultCollection<QueryResultItem<(), T, D::Output>>,
                    EXCLUSIVE,
                >(query, max_dist, max_qty, sorted);
            }

            #[cfg(not(feature = "small_n_result_collectors"))]
            if max_qty
                <= if sorted {
                    MAX_SORTED_VEC_RESULT_SIZE
                } else {
                    MAX_UNSORTED_VEC_RESULT_SIZE
                }
            {
                return self.nearest_n_within_inner::<
                    D,
                    ThresholdVecResultCollection<QueryResultItem<(), T, D::Output>>,
                    EXCLUSIVE,
                >(query, max_dist, max_qty, sorted);
            }

            self.nearest_n_within_inner::<D, BinaryHeapResultCollection<QueryResultItem<(), T, D::Output>>, EXCLUSIVE>(
                query, max_dist, max_qty, sorted,
            )
        }
    }

    pub(crate) fn nearest_n_within_impl_with_scratch<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
        max_qty: NonZeroUsize,
        sorted: bool,
        stack: &mut SS::Stack<D::Output>,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS>,
    {
        let max_qty: usize = max_qty.get();

        if max_qty == usize::MAX {
            self.nearest_n_within_inner_with_scratch::<
                D,
                Vec<QueryResultItem<(), T, D::Output>>,
                EXCLUSIVE,
            >(query, max_dist, max_qty, sorted, stack)
        } else {
            #[cfg(feature = "small_n_result_collectors")]
            if max_qty <= SMALL_RESULT_COLLECTION_MAX_QTY {
                return self.nearest_n_within_inner_with_scratch::<
                    D,
                    SmallSortedVecResultCollection<QueryResultItem<(), T, D::Output>>,
                    EXCLUSIVE,
                >(query, max_dist, max_qty, sorted, stack);
            }

            #[cfg(not(feature = "small_n_result_collectors"))]
            if max_qty
                <= if sorted {
                    MAX_SORTED_VEC_RESULT_SIZE
                } else {
                    MAX_UNSORTED_VEC_RESULT_SIZE
                }
            {
                return self.nearest_n_within_inner_with_scratch::<
                    D,
                    ThresholdVecResultCollection<QueryResultItem<(), T, D::Output>>,
                    EXCLUSIVE,
                >(query, max_dist, max_qty, sorted, stack);
            }

            self.nearest_n_within_inner_with_scratch::<
                D,
                BinaryHeapResultCollection<QueryResultItem<(), T, D::Output>>,
                EXCLUSIVE,
            >(query, max_dist, max_qty, sorted, stack)
        }
    }

    fn nearest_n_within_inner<D, R, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
        max_qty: usize,
        sorted: bool,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        R: ResultCollection<D::Output, QueryResultItem<(), T, D::Output>>,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    {
        let mut req_ctx = NearestNWithinReqCtx::<A, T, D::Output, R, EXCLUSIVE, K> {
            query,
            max_dist,
            results: R::with_max_qty(max_qty),
            _phantom: std::marker::PhantomData,
        };

        self.backtracking_query::<_, _, D>(&mut req_ctx, |leaf_idx, query_wide, req_ctx| {
            let leaf_max_dist = req_ctx.max_dist();
            self.process_leaf_nearest_n_within::<D, R, EXCLUSIVE>(
                leaf_idx,
                query_wide,
                leaf_max_dist,
                &mut req_ctx.results,
            );
        });

        if sorted {
            req_ctx.results.into_sorted_vec()
        } else {
            req_ctx.results.into_vec()
        }
    }

    fn nearest_n_within_inner_with_scratch<D, R, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
        max_qty: usize,
        sorted: bool,
        stack: &mut SS::Stack<D::Output>,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        R: ResultCollection<D::Output, QueryResultItem<(), T, D::Output>>,
        SS::Stack<D::Output>: StackTrait<D::Output, SS>,
    {
        let mut req_ctx = NearestNWithinReqCtx::<A, T, D::Output, R, EXCLUSIVE, K> {
            query,
            max_dist,
            results: R::with_max_qty(max_qty),
            _phantom: std::marker::PhantomData,
        };

        self.backtracking_query_with_scratch::<_, _, D>(
            &mut req_ctx,
            stack,
            |leaf_idx, query_wide, req_ctx| {
                let leaf_max_dist = req_ctx.max_dist();
                self.process_leaf_nearest_n_within::<D, R, EXCLUSIVE>(
                    leaf_idx,
                    query_wide,
                    leaf_max_dist,
                    &mut req_ctx.results,
                );
            },
        );

        if sorted {
            req_ctx.results.into_sorted_vec()
        } else {
            req_ctx.results.into_vec()
        }
    }
}

#[allow(missing_docs)]
#[cfg(feature = "cargo_asm")]
pub mod cargo_asm {
    use crate::dist::SquaredEuclidean;
    use crate::kd_tree::KdTree;
    use crate::leaf_strategy::VecOfArenas;
    use crate::stem_strategy::DonnellyUnrolled;
    use crate::stem_strategy::Eytzinger;
    use std::num::NonZeroUsize;

    const K: usize = 3;
    const BUCKET_SIZE: usize = 32;
    const MAX_DIST: f64 = 0.0025;
    const MAX_QTY: usize = 16;

    type ArenaLeaves = VecOfArenas<f64, u32, K, BUCKET_SIZE>;
    type EytzingerPfFarKdT = KdTree<f64, u32, Eytzinger, ArenaLeaves, K, BUCKET_SIZE>;
    type DonnellyUnrolledKdT = KdTree<f64, u32, DonnellyUnrolled<3>, ArenaLeaves, K, BUCKET_SIZE>;

    /// Hook for cargo-asm to render the sorted nearest_n_within focus path.
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_sorted_nearest_n_within_donnelly_pf_focus_cargo_asm_hook(
        tree: &DonnellyUnrolledKdT,
        query: [f64; 3],
    ) -> (usize, u64, u64) {
        let results = tree.nearest_n_within::<SquaredEuclidean<f64>>(
            &query,
            MAX_DIST,
            NonZeroUsize::new(MAX_QTY).unwrap(),
            true,
        );

        let mut checksum_item = 0u64;
        let mut checksum_dist_bits = 0u64;
        for result in results.iter() {
            checksum_item = checksum_item.wrapping_add(result.item as u64);
            checksum_dist_bits = checksum_dist_bits.wrapping_add(result.distance.to_bits());
        }

        (results.len(), checksum_item, checksum_dist_bits)
    }

    /// Hook for cargo-asm to render the sorted nearest_n_within focus path for scalar Eytzinger PF-far arena leaves.
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_sorted_nearest_n_within_eytzinger_pf_far_focus_cargo_asm_hook(
        tree: &EytzingerPfFarKdT,
        query: [f64; 3],
    ) -> (usize, u64, u64) {
        let results = tree.nearest_n_within::<SquaredEuclidean<f64>>(
            &query,
            MAX_DIST,
            NonZeroUsize::new(MAX_QTY).unwrap(),
            true,
        );

        let mut checksum_item = 0u64;
        let mut checksum_dist_bits = 0u64;
        for result in results.iter() {
            checksum_item = checksum_item.wrapping_add(result.item as u64);
            checksum_dist_bits = checksum_dist_bits.wrapping_add(result.distance.to_bits());
        }

        (results.len(), checksum_item, checksum_dist_bits)
    }
}

#[allow(unused)]
struct NearestNWithinReqCtx<'a, A, T, O, R, const EXCLUSIVE: bool, const K: usize>
where
    O: Axis<Coord = O>,
{
    query: &'a [A; K],
    max_dist: O,
    results: R,
    _phantom: std::marker::PhantomData<T>,
}

impl<A, T, O, R, const EXCLUSIVE: bool, const K: usize> QueryContext<A, O, K>
    for NearestNWithinReqCtx<'_, A, T, O, R, EXCLUSIVE, K>
where
    O: Axis<Coord = O>,
    R: ResultCollection<O, QueryResultItem<(), T, O>>,
{
    fn query(&self) -> &[A; K] {
        self.query
    }

    fn max_dist(&self) -> O {
        let results_cap = self.results.threshold_distance().unwrap_or(O::max_value());
        if results_cap < self.max_dist {
            results_cap
        } else {
            self.max_dist
        }
    }

    #[inline]
    fn initial_bound_is_unbounded(&self) -> bool {
        self.max_dist == O::max_value()
    }

    #[inline]
    fn prune_on_equal_max_dist(&self) -> bool {
        EXCLUSIVE
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;
    use std::num::{NonZero, NonZeroUsize};

    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    use test_log::test;

    use crate::dist::SquaredEuclidean;
    use crate::kd_tree::KdTree;
    use crate::leaf_strategy::{FlatVec, VecOfArenas, VecOfArrays};
    #[cfg(feature = "result_collection_stats")]
    use crate::results::result_collection_stats::{reset, snapshot};
    #[cfg(all(feature = "result_collection_stats", feature = "simd"))]
    use crate::stem_strategy::DonnellySimdFull;
    use crate::Axis;
    use crate::Eytzinger;

    const RNG_SEED: u64 = 42;

    #[test]
    fn nearest_n_within_exclusive_boundaries_exclude_exact_threshold_matches() {
        let points = vec![[0.0f64, 0.0], [1.0, 0.0], [2.0, 0.0], [0.5, 0.0]];
        let tree: KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, 2, 32>, 2, 32> =
            KdTree::new_from_slice(&points).unwrap();
        let query = [0.0, 0.0];
        let max_qty = NonZeroUsize::new(8).unwrap();

        let inclusive_sorted: Vec<_> = tree
            .query(&query)
            .nearest_n::<SquaredEuclidean<f64>>(max_qty)
            .within(1.0)
            .execute()
            .into_iter()
            .map(|n| n.item)
            .collect();
        let exclusive_sorted: Vec<_> = tree
            .query(&query)
            .nearest_n::<SquaredEuclidean<f64>>(max_qty)
            .within(1.0)
            .exclusive_boundaries()
            .execute()
            .into_iter()
            .map(|n| n.item)
            .collect();
        let mut exclusive_unsorted: Vec<_> = tree
            .query(&query)
            .nearest_n::<SquaredEuclidean<f64>>(max_qty)
            .within(1.0)
            .exclusive_boundaries()
            .unsorted()
            .execute()
            .into_iter()
            .map(|n| n.item)
            .collect();

        exclusive_unsorted.sort_unstable();

        assert_eq!(inclusive_sorted, vec![0, 3, 1]);
        assert_eq!(exclusive_sorted, vec![0, 3]);
        assert_eq!(exclusive_unsorted, vec![0, 3]);
    }

    #[test]
    fn nearest_n_within_sorted_flat_vec_f32() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        let mut points: Vec<[f32; 3]> = vec![];
        for _ in 0..65_536 {
            let x = rng.random_range(0.0..1.0);
            let y = rng.random_range(0.0..1.0);
            let z = rng.random_range(0.0..1.0);
            points.push([x, y, z]);
        }

        let tree: KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points).unwrap();

        assert!(!tree.is_empty());
        assert_eq!(tree.size(), 65_536);
        assert_eq!(tree.leaf_count(), 2048);
        assert_eq!(tree.max_stem_level(), 10);

        let query_point = [0.5, 0.5, 0.5];
        let radius = 0.1;
        let max_qty = NonZeroUsize::new(10).unwrap();

        let results = tree
            .query(&query_point)
            .nearest_n::<SquaredEuclidean<f32>>(max_qty)
            .within(radius)
            .execute();
        assert_eq!(results.len(), 10);

        // Exercise the sorted BinaryHeapResultCollection fallback.
        #[cfg(not(feature = "small_n_result_collectors"))]
        let heap_threshold = super::MAX_SORTED_VEC_RESULT_SIZE;
        #[cfg(feature = "small_n_result_collectors")]
        let heap_threshold = super::SMALL_RESULT_COLLECTION_MAX_QTY;
        let max_qty_large = NonZeroUsize::new(heap_threshold + 1).unwrap();
        let results_large = tree
            .query(&query_point)
            .nearest_n::<SquaredEuclidean<f32>>(max_qty_large)
            .within(radius)
            .execute();
        assert_eq!(results_large.len(), max_qty_large.get());
    }

    #[test]
    fn nearest_n_within_sorted_flat_vec_f32_no_items() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        let mut points: Vec<[f32; 3]> = vec![];
        for _ in 0..65_536 {
            let x = rng.random_range(0.0..1.0);
            let y = rng.random_range(0.0..1.0);
            let z = rng.random_range(0.0..1.0);
            points.push([x, y, z]);
        }

        let tree: KdTree<f32, (), Eytzinger, FlatVec<f32, (), 3, 32>, 3, 32> =
            KdTree::new_from_slice_no_items(&points).unwrap();

        assert!(!tree.is_empty());
        assert_eq!(tree.size(), 65_536);
        assert_eq!(tree.leaf_count(), 2048);
        assert_eq!(tree.max_stem_level(), 10);

        let query_point = [0.5, 0.5, 0.5];
        let radius = 0.1;
        let max_qty = NonZeroUsize::new(10).unwrap();

        let results = tree
            .query(&query_point)
            .nearest_n::<SquaredEuclidean<f32>>(max_qty)
            .within(radius)
            .execute();
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn nearest_n_within_vec_of_arenas_matches_flat_vec_f32() {
        let points: Vec<[f32; 3]> = (0..40)
            .map(|idx| {
                [
                    idx as f32 / 40.0,
                    ((idx * 7) % 40) as f32 / 40.0,
                    ((idx * 13) % 40) as f32 / 40.0,
                ]
            })
            .collect();
        let query = [0.42f32, 0.53, 0.61];
        let max_qty = NonZeroUsize::new(5).unwrap();
        let max_dist = 0.2;

        let flat_tree: KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points).unwrap();
        let arena_tree: KdTree<f32, u32, Eytzinger, VecOfArenas<f32, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points).unwrap();

        let flat_result = flat_tree
            .query(&query)
            .nearest_n::<SquaredEuclidean<f32>>(max_qty)
            .within(max_dist)
            .execute();
        let arena_result = arena_tree
            .query(&query)
            .nearest_n::<SquaredEuclidean<f32>>(max_qty)
            .within(max_dist)
            .execute();

        assert_eq!(arena_result, flat_result);
    }

    #[test]
    fn nearest_n_within_unbounded_vec_of_arenas_matches_flat_vec_f32() {
        let points: Vec<[f32; 3]> = (0..40)
            .map(|idx| {
                [
                    ((idx * 3) % 40) as f32 / 40.0,
                    ((idx * 11) % 40) as f32 / 40.0,
                    ((idx * 17) % 40) as f32 / 40.0,
                ]
            })
            .collect();
        let query = [0.35f32, 0.45, 0.55];
        let max_dist = 0.5;

        let flat_tree: KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points).unwrap();
        let arena_tree: KdTree<f32, u32, Eytzinger, VecOfArenas<f32, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points).unwrap();

        let flat_result = flat_tree.nearest_n_within::<SquaredEuclidean<f32>>(
            &query,
            max_dist,
            NonZeroUsize::MAX,
            true,
        );
        let arena_result = arena_tree.nearest_n_within::<SquaredEuclidean<f32>>(
            &query,
            max_dist,
            NonZeroUsize::MAX,
            true,
        );

        assert_eq!(arena_result, flat_result);
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    #[test]
    fn nearest_n_within_vec_of_arenas_matches_flat_vec_f64_simd() {
        let points: Vec<[f64; 3]> = (0..40)
            .map(|idx| {
                [
                    idx as f64 / 40.0,
                    ((idx * 7) % 40) as f64 / 40.0,
                    ((idx * 13) % 40) as f64 / 40.0,
                ]
            })
            .collect();
        let query = [0.42f64, 0.53, 0.61];
        let max_qty = NonZeroUsize::new(5).unwrap();
        let max_dist = 0.2;

        let flat_tree: KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points).unwrap();
        let arena_tree: KdTree<f64, u32, Eytzinger, VecOfArenas<f64, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points).unwrap();

        let flat_result = flat_tree
            .query(&query)
            .nearest_n::<SquaredEuclidean<f64>>(max_qty)
            .within(max_dist)
            .execute();
        let arena_result = arena_tree
            .query(&query)
            .nearest_n::<SquaredEuclidean<f64>>(max_qty)
            .within(max_dist)
            .execute();

        assert_eq!(arena_result, flat_result);
    }

    #[test]
    fn v6_n_items_within_f32_eytzinger_large_scale() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(RNG_SEED);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        const RADIUS: f32 = 0.2;

        let max_qty: NonZero<usize> = NonZero::new(3).unwrap();

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

        let tree: KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&content_to_add).unwrap();
        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[f32; 4]> = (0..NUM_QUERIES)
            .map(|_| rng.random::<[f32; 4]>()) // Use the seeded rng
            .collect();

        for query_point in query_points.iter() {
            let expected = linear_search(&content_to_add, query_point, RADIUS)
                .into_iter()
                .take(max_qty.into())
                .collect::<Vec<_>>();

            let mut result: Vec<_> = tree
                .nearest_n_within::<SquaredEuclidean<f32>>(query_point, RADIUS, max_qty, true)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            stabilize_sort(&mut result);

            assert_distance_item_pairs_close_f32(&result, &expected);
        }
    }

    #[test]
    fn v6_nearest_n_within_f32_eytzinger_large_vec_of_arrays() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(RNG_SEED);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        const RADIUS: f32 = 0.2;

        let max_qty: NonZero<usize> = NonZero::new(3).unwrap();

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

        let tree: KdTree<f32, u32, Eytzinger, VecOfArrays<f32, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&content_to_add).unwrap();
        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[f32; 4]> = (0..NUM_QUERIES)
            .map(|_| rng.random::<[f32; 4]>()) // Use the seeded rng
            .collect();

        for query_point in query_points.iter() {
            let expected = linear_search(&content_to_add, query_point, RADIUS)
                .into_iter()
                .take(max_qty.into())
                .collect::<Vec<_>>();

            let mut result: Vec<_> = tree
                .nearest_n_within::<SquaredEuclidean<f32>>(query_point, RADIUS, max_qty, true)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            stabilize_sort(&mut result);

            assert_distance_item_pairs_close_f32(&result, &expected);
        }
    }

    #[test]
    fn v6_nearest_n_within_f32_eytzinger_large_vec_of_arrays_mutated_f32() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(RNG_SEED);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        const RADIUS: f32 = 0.2;

        let max_qty: NonZero<usize> = NonZero::new(3).unwrap();

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

        let mut tree: KdTree<f32, u32, Eytzinger, VecOfArrays<f32, u32, 4, 32>, 4, 32> =
            KdTree::default();

        for (idx, point) in content_to_add.iter().enumerate() {
            tree.add(point, idx as u32).unwrap();
        }

        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[f32; 4]> = (0..NUM_QUERIES)
            .map(|_| rng.random::<[f32; 4]>()) // Use the seeded rng
            .collect();

        for query_point in query_points.iter() {
            let expected = linear_search(&content_to_add, query_point, RADIUS)
                .into_iter()
                .take(max_qty.into())
                .collect::<Vec<_>>();

            let mut result: Vec<_> = tree
                .nearest_n_within::<SquaredEuclidean<f32>>(query_point, RADIUS, max_qty, true)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            stabilize_sort(&mut result);

            assert_distance_item_pairs_close_f32(&result, &expected);
        }
    }

    fn assert_distance_item_pairs_close_f32<T>(actual: &[(f32, T)], expected: &[(f32, T)])
    where
        T: std::fmt::Debug + PartialEq,
    {
        assert_eq!(actual.len(), expected.len());

        for ((actual_dist, actual_item), (expected_dist, expected_item)) in
            actual.iter().zip(expected.iter())
        {
            assert_eq!(actual_item, expected_item);
            assert!(
                ulps_diff_f32(*actual_dist, *expected_dist) <= 2,
                "distance mismatch: actual={actual_dist:?} expected={expected_dist:?}"
            );
        }
    }

    fn ulps_diff_f32(a: f32, b: f32) -> u32 {
        canonical_u32(a).abs_diff(canonical_u32(b))
    }

    fn canonical_u32(value: f32) -> u32 {
        let bits = value.to_bits();
        if (bits >> 31) != 0 {
            !bits
        } else {
            bits | (1 << 31)
        }
    }

    fn linear_search<A, const K: usize>(
        content: &[[A; K]],
        query_point: &[A; K],
        radius: A,
    ) -> Vec<(A, u32)>
    where
        A: Axis<Coord = A> + 'static,
        SquaredEuclidean<A>: crate::dist::DistanceMetricCore<A, Output = A>,
    {
        let mut matching_items = vec![];

        for (idx, p) in content.iter().enumerate() {
            let dist = squared_euclidean_dist(query_point, p);
            if dist <= radius {
                matching_items.push((dist, idx as u32));
            }
        }

        stabilize_sort(&mut matching_items);

        matching_items
    }

    fn squared_euclidean_dist<A, const K: usize>(a: &[A; K], b: &[A; K]) -> A
    where
        A: Axis<Coord = A>,
        SquaredEuclidean<A>: crate::dist::DistanceMetricCore<A, Output = A>,
    {
        let aw = (*a).map(|coord| {
            <SquaredEuclidean<A> as crate::dist::DistanceMetricCore<A>>::widen_coord(coord)
        });
        let bw = (*b).map(|coord| {
            <SquaredEuclidean<A> as crate::dist::DistanceMetricCore<A>>::widen_coord(coord)
        });

        <SquaredEuclidean<A> as crate::dist::DistanceMetricCore<A>>::dist::<K>(&aw, &bw)
    }

    fn stabilize_sort<A>(matching_items: &mut [(A, u32)])
    where
        A: Axis<Coord = A>,
    {
        matching_items.sort_unstable_by(|a, b| {
            let dist_cmp = a.0.partial_cmp(&b.0).unwrap();
            if dist_cmp == Ordering::Equal {
                a.1.cmp(&b.1)
            } else {
                dist_cmp
            }
        });
    }

    #[cfg(feature = "result_collection_stats")]
    fn clustered_points_2d() -> Vec<[f64; 2]> {
        const CLUSTERS: [[f64; 2]; 4] = [[0.10, 0.10], [0.90, 0.10], [0.10, 0.90], [0.90, 0.90]];
        let mut points = Vec::new();

        for center in CLUSTERS {
            for idx in 0..32 {
                let dx = (idx % 4) as f64 * 0.0015;
                let dy = (idx / 4) as f64 * 0.0015;
                points.push([center[0] + dx, center[1] + dy]);
            }
        }

        points
    }

    #[cfg(feature = "result_collection_stats")]
    #[test]
    fn nearest_n_within_scalar_prunes_clustered_tree() {
        const K: usize = 2;
        const B: usize = 4;

        let points = clustered_points_2d();
        let query = [0.1045f64, 0.1045];
        let max_qty = NonZeroUsize::new(4).unwrap();
        let max_dist = 0.0004;

        let tree: KdTree<f64, u32, Eytzinger, VecOfArenas<f64, u32, K, B>, K, B> =
            KdTree::new_from_slice(&points).unwrap();

        reset();
        let result = tree
            .query(&query)
            .nearest_n::<SquaredEuclidean<f64>>(max_qty)
            .within(max_dist)
            .execute();
        let stats = snapshot();

        let expected = linear_search(&points, &query, max_dist)
            .into_iter()
            .take(max_qty.get())
            .collect::<Vec<_>>();

        let mut actual = result
            .into_iter()
            .map(|n| (n.distance, n.item))
            .collect::<Vec<_>>();
        let mut expected = expected;
        stabilize_sort(&mut actual);
        stabilize_sort(&mut expected);

        assert_distance_item_pairs_close_f64(&actual, &expected);
        assert!(
            stats.leaf_visits < tree.leaf_count() as u64,
            "scalar path regressed to full leaf scan: leaf_visits={} leaf_count={}",
            stats.leaf_visits,
            tree.leaf_count()
        );
    }

    #[cfg(all(feature = "result_collection_stats", feature = "simd"))]
    #[test]
    fn nearest_n_within_simd_prunes_clustered_tree() {
        const K: usize = 2;
        const B: usize = 4;

        let points = clustered_points_2d();
        let query = [0.1045f64, 0.1045];
        let max_qty = NonZeroUsize::new(4).unwrap();
        let max_dist = 0.0004;

        let tree: KdTree<f64, u32, DonnellySimdFull<3>, VecOfArenas<f64, u32, K, B>, K, B> =
            KdTree::new_from_slice(&points).unwrap();

        reset();
        let result = tree
            .query(&query)
            .nearest_n::<SquaredEuclidean<f64>>(max_qty)
            .within(max_dist)
            .execute();
        let stats = snapshot();

        let expected = linear_search(&points, &query, max_dist)
            .into_iter()
            .take(max_qty.get())
            .collect::<Vec<_>>();

        let mut actual = result
            .into_iter()
            .map(|n| (n.distance, n.item))
            .collect::<Vec<_>>();
        let mut expected = expected;
        stabilize_sort(&mut actual);
        stabilize_sort(&mut expected);

        assert_distance_item_pairs_close_f64(&actual, &expected);
        assert!(
            stats.leaf_visits < tree.leaf_count() as u64,
            "simd path regressed to full leaf scan: leaf_visits={} leaf_count={}",
            stats.leaf_visits,
            tree.leaf_count()
        );
    }

    #[cfg(feature = "result_collection_stats")]
    fn assert_distance_item_pairs_close_f64<T>(actual: &[(f64, T)], expected: &[(f64, T)])
    where
        T: std::fmt::Debug + PartialEq,
    {
        assert_eq!(actual.len(), expected.len());

        for ((actual_dist, actual_item), (expected_dist, expected_item)) in
            actual.iter().zip(expected.iter())
        {
            assert_eq!(actual_item, expected_item);
            assert!(
                ulps_diff_f64(*actual_dist, *expected_dist) <= 2,
                "distance mismatch: actual={actual_dist:?} expected={expected_dist:?}"
            );
        }
    }

    #[cfg(feature = "result_collection_stats")]
    fn ulps_diff_f64(a: f64, b: f64) -> u64 {
        canonical_u64(a).abs_diff(canonical_u64(b))
    }

    #[cfg(feature = "result_collection_stats")]
    fn canonical_u64(value: f64) -> u64 {
        let bits = value.to_bits();
        if (bits >> 63) != 0 {
            !bits
        } else {
            bits | (1 << 63)
        }
    }
}

use std::collections::BinaryHeap;
use std::num::NonZero;

use crate::dist::DistanceMetric;
use crate::kd_tree::query_context::QueryContext;
use crate::kd_tree::query_stack::StackTrait;
use crate::kd_tree::KdTreeQueryOps;
use crate::leaf_view::TlsLeafScratch;
use crate::leaf_view_chunked::best_n_within::{
    best_n_within_with_query_wide, best_n_within_with_query_wide_arena,
};
use crate::results::result_collection::{
    BestNeighbourResultCollection, BinaryHeapResultCollection,
};
#[cfg(feature = "small_n_result_collectors")]
use crate::results::result_collection::{
    SmallBinaryHeapResultCollection, SMALL_RESULT_COLLECTION_MAX_QTY,
};
use crate::stem_strategy::donnelly::simd_full::{
    BacktrackBlock3, BacktrackBlock4, SimdSelectBestChildBlock3,
};
use crate::traits::leaf_strategy::LeafProjection;
use crate::{Axis, BestQueryResultItem, Content, KdTree, LeafStrategy, StemStrategy};

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    LS: LeafStrategy<A, T, SS, K, B>,
    SS: StemStrategy,
{
    #[inline(always)]
    fn process_leaf_best_n_within<D, R, const EXCLUSIVE: bool>(
        &self,
        leaf_idx: usize,
        query_wide: &[D::Output; K],
        max_dist: D::Output,
        results: &mut R,
    ) where
        D: DistanceMetric<A>,
        D::Output: Axis<Coord = D::Output> + TlsLeafScratch + 'static,
        R: BestNeighbourResultCollection<D::Output, T>,
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
                let threshold_item = results.threshold_item();
                best_n_within_with_query_wide_arena::<A, T, D, R, EXCLUSIVE, K>(
                    &arena,
                    query_wide,
                    max_dist,
                    threshold_item,
                    results,
                );
            }
            LeafProjection::LeafView => {
                let leaf = self.leaves.leaf_view(leaf_idx);
                let threshold_item = results.threshold_item();
                best_n_within_with_query_wide::<A, T, D, R, EXCLUSIVE, K, B>(
                    &leaf,
                    query_wide,
                    max_dist,
                    threshold_item,
                    results,
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

    pub(crate) fn best_n_within_impl<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
        max_qty: NonZero<usize>,
    ) -> BinaryHeap<BestQueryResultItem<(), T, D::Output>>
    where
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output, K>: StackTrait<D::Output, SS, K> + 'static,
    {
        let max_qty = max_qty.into();

        #[cfg(feature = "small_n_result_collectors")]
        if max_qty <= SMALL_RESULT_COLLECTION_MAX_QTY {
            return self
                .best_n_within_inner::<D, SmallBinaryHeapResultCollection<
                    BestQueryResultItem<(), T, D::Output>,
                >, EXCLUSIVE>(query, max_dist, max_qty)
                .into_inner();
        }

        self.best_n_within_inner::<
            D,
            BinaryHeapResultCollection<BestQueryResultItem<(), T, D::Output>>,
            EXCLUSIVE,
        >(
            query, max_dist, max_qty,
        )
        .into_inner()
    }

    pub(crate) fn best_n_within_impl_with_scratch<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
        max_qty: NonZero<usize>,
        stack: &mut SS::Stack<D::Output, K>,
    ) -> BinaryHeap<BestQueryResultItem<(), T, D::Output>>
    where
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output, K>: StackTrait<D::Output, SS, K>,
    {
        let max_qty = max_qty.into();

        #[cfg(feature = "small_n_result_collectors")]
        if max_qty <= SMALL_RESULT_COLLECTION_MAX_QTY {
            return self
                .best_n_within_inner_with_scratch::<D, SmallBinaryHeapResultCollection<
                    BestQueryResultItem<(), T, D::Output>,
                >, EXCLUSIVE>(query, max_dist, max_qty, stack)
                .into_inner();
        }

        self.best_n_within_inner_with_scratch::<
            D,
            BinaryHeapResultCollection<BestQueryResultItem<(), T, D::Output>>,
            EXCLUSIVE,
        >(
            query, max_dist, max_qty, stack,
        )
        .into_inner()
    }

    fn best_n_within_inner<D, R, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
        max_qty: usize,
    ) -> R
    where
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        R: BestNeighbourResultCollection<D::Output, T>,
        SS::Stack<D::Output, K>: StackTrait<D::Output, SS, K> + 'static,
    {
        let mut req_ctx = BestNWithinReqCtx::<A, D::Output, R, EXCLUSIVE, K> {
            query,
            max_dist,
            results: R::with_max_qty(max_qty),
        };

        self.backtracking_query::<_, _, D>(&mut req_ctx, |leaf_idx, query_wide, req_ctx| {
            self.process_leaf_best_n_within::<D, _, EXCLUSIVE>(
                leaf_idx,
                query_wide,
                max_dist,
                &mut req_ctx.results,
            );
        });

        req_ctx.results
    }

    fn best_n_within_inner_with_scratch<D, R, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
        max_qty: usize,
        stack: &mut SS::Stack<D::Output, K>,
    ) -> R
    where
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        R: BestNeighbourResultCollection<D::Output, T>,
        SS::Stack<D::Output, K>: StackTrait<D::Output, SS, K>,
    {
        let mut req_ctx = BestNWithinReqCtx::<A, D::Output, R, EXCLUSIVE, K> {
            query,
            max_dist,
            results: R::with_max_qty(max_qty),
        };

        self.backtracking_query_with_scratch::<_, _, D>(
            &mut req_ctx,
            stack,
            |leaf_idx, query_wide, req_ctx| {
                self.process_leaf_best_n_within::<D, _, EXCLUSIVE>(
                    leaf_idx,
                    query_wide,
                    max_dist,
                    &mut req_ctx.results,
                );
            },
        );

        req_ctx.results
    }
}

#[allow(missing_docs)]
#[cfg(feature = "cargo_asm")]
pub mod cargo_asm {
    use crate::dist::SquaredEuclidean;
    use crate::kd_tree::KdTree;
    use crate::leaf_strategy::VecOfArenas;
    use crate::stem_strategy::DonnellyUnrolled;
    use std::num::NonZeroUsize;

    const K: usize = 3;
    const BUCKET_SIZE: usize = 32;
    const MAX_DIST: f64 = 0.0025;
    const MAX_QTY: usize = 16;

    type ArenaLeaves = VecOfArenas<f64, u32, K, BUCKET_SIZE>;
    type DonnellyUnrolledKdT = KdTree<f64, u32, DonnellyUnrolled<3>, ArenaLeaves, K, BUCKET_SIZE>;

    /// Hook for cargo-asm to render the best_n_within focus path.
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_best_n_within_donnelly_pf_focus_cargo_asm_hook(
        tree: &DonnellyUnrolledKdT,
        query: [f64; 3],
    ) -> (usize, u64, u64) {
        let results = tree
            .query(&query)
            .best_n_within::<SquaredEuclidean<f64>>(MAX_DIST, NonZeroUsize::new(MAX_QTY).unwrap())
            .execute();

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
#[derive(Debug)]
struct BestNWithinReqCtx<'a, A, O, R, const EXCLUSIVE: bool, const K: usize>
where
    O: Axis<Coord = O>,
{
    query: &'a [A; K],
    max_dist: O,
    results: R,
}

impl<'a, A, O, R, const EXCLUSIVE: bool, const K: usize> QueryContext<A, O, K>
    for BestNWithinReqCtx<'a, A, O, R, EXCLUSIVE, K>
where
    O: Axis<Coord = O>,
{
    fn query(&self) -> &[A; K] {
        self.query
    }
    fn max_dist(&self) -> O {
        self.max_dist
    }

    #[inline]
    fn prune_on_equal_max_dist(&self) -> bool {
        EXCLUSIVE
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Debug;
    use std::num::{NonZero, NonZeroUsize};

    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    use crate::dist::SquaredEuclidean;
    use crate::kd_tree::KdTree;
    use crate::leaf_strategy::{FlatVec, VecOfArenas, VecOfArrays};
    use crate::{BestQueryResultItem, Eytzinger};

    const RNG_SEED: u64 = 42;

    #[test]
    fn best_n_within_exclusive_boundaries_exclude_exact_threshold_matches() {
        let points = vec![[0.0f64, 0.0], [1.0, 0.0], [2.0, 0.0], [0.5, 0.0]];
        let tree: KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, 2, 32>, 2, 32> =
            KdTree::new_from_slice(&points).unwrap();
        let query = [0.0, 0.0];
        let max_qty = NonZero::new(8usize).unwrap();

        let inclusive: Vec<_> = tree
            .query(&query)
            .best_n_within::<SquaredEuclidean<f64>>(1.0, max_qty)
            .execute()
            .into_sorted_vec()
            .into_iter()
            .map(|n| n.item)
            .collect();
        let exclusive: Vec<_> = tree
            .query(&query)
            .best_n_within::<SquaredEuclidean<f64>>(1.0, max_qty)
            .exclusive_boundaries()
            .execute()
            .into_sorted_vec()
            .into_iter()
            .map(|n| n.item)
            .collect();

        assert_eq!(inclusive, vec![0, 1, 3]);
        assert_eq!(exclusive, vec![0, 3]);
    }

    #[test]
    fn best_n_within_flat_vec_f32() {
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

        // perform a best_n_within query
        let query_point = [0.5, 0.5, 0.5];
        let radius = 0.1f32;
        let max_qty = NonZeroUsize::new(10).unwrap();
        let results = tree
            .query(&query_point)
            .best_n_within::<SquaredEuclidean<f32>>(radius, max_qty)
            .execute();
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn v6_query_best_n_within_large_f64_flat_vec() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        let max_qty = NonZero::new(2).unwrap();

        let content_to_add: Vec<_> = (0..TREE_SIZE).map(|_| rng.random::<[f64; 2]>()).collect();

        let tree: KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, 2, 32>, 2, 32> =
            KdTree::new_from_slice(&content_to_add).unwrap();

        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<_> = (0..NUM_QUERIES)
            .map(|_| rng.random::<_>()) // Use the seeded rng
            .collect();

        for query_point in query_points {
            let radius = 100000f64;
            let expected = linear_search(&content_to_add, &query_point, radius, max_qty.into());

            let result: Vec<_> = tree
                .query(&query_point)
                .best_n_within::<SquaredEuclidean<f64>>(radius, max_qty)
                .execute()
                .into_iter()
                .collect();

            assert_best_neighbours_close_f64(&result, &expected);
        }
    }

    #[test]
    fn v6_query_best_n_within_large_f64_vec_of_arrays() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        let max_qty = NonZero::new(2).unwrap();

        let content_to_add: Vec<_> = (0..TREE_SIZE).map(|_| rng.random::<[f64; 2]>()).collect();

        let tree: KdTree<f64, u32, Eytzinger, VecOfArrays<f64, u32, 2, 32>, 2, 32> =
            KdTree::new_from_slice(&content_to_add).unwrap();

        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<_> = (0..NUM_QUERIES)
            .map(|_| rng.random::<_>()) // Use the seeded rng
            .collect();

        for query_point in query_points {
            let radius = 100000f64;
            let expected = linear_search(&content_to_add, &query_point, radius, max_qty.into());

            let result: Vec<_> = tree
                .query(&query_point)
                .best_n_within::<SquaredEuclidean<f64>>(radius, max_qty)
                .execute()
                .into_iter()
                .collect();

            assert_best_neighbours_close_f64(&result, &expected);
        }
    }

    #[test]
    fn v6_query_best_n_within_large_vec_of_arrays_mutated_f64() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        let max_qty = NonZero::new(2).unwrap();

        let content_to_add: Vec<_> = (0..TREE_SIZE).map(|_| rng.random::<[f64; 2]>()).collect();

        let mut tree: KdTree<f64, u32, Eytzinger, VecOfArrays<f64, u32, 2, 32>, 2, 32> =
            KdTree::default();

        for (idx, point) in content_to_add.iter().enumerate() {
            tree.add(point, idx as u32).unwrap();
        }

        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<_> = (0..NUM_QUERIES)
            .map(|_| rng.random::<_>()) // Use the seeded rng
            .collect();

        for query_point in query_points {
            let radius = 100000f64;
            let expected = linear_search(&content_to_add, &query_point, radius, max_qty.into());

            let result: Vec<_> = tree
                .query(&query_point)
                .best_n_within::<SquaredEuclidean<f64>>(radius, max_qty)
                .execute()
                .into_iter()
                .collect();

            assert_best_neighbours_close_f64(&result, &expected);
        }
    }

    #[test]
    fn v6_query_best_n_within_vec_of_arenas_boundary_parity_f64() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);
        let query = [0.35, 0.65];
        let radius = 10.0f64;
        let max_qty = NonZero::new(5).unwrap();

        for len in [1usize, 2, 4, 8, 32, 33, 47] {
            let points: Vec<[f64; 2]> = (0..len).map(|_| rng.random::<[f64; 2]>()).collect();

            let flat_tree: KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, 2, 32>, 2, 32> =
                KdTree::new_from_slice(&points).unwrap();
            let arena_tree: KdTree<f64, u32, Eytzinger, VecOfArenas<f64, u32, 2, 32>, 2, 32> =
                KdTree::new_from_slice(&points).unwrap();

            let mut flat_results: Vec<_> = flat_tree
                .query(&query)
                .best_n_within::<SquaredEuclidean<f64>>(radius, max_qty)
                .execute()
                .into_sorted_vec();
            let mut arena_results: Vec<_> = arena_tree
                .query(&query)
                .best_n_within::<SquaredEuclidean<f64>>(radius, max_qty)
                .execute()
                .into_sorted_vec();

            flat_results.sort();
            arena_results.sort();

            assert_eq!(arena_results, flat_results, "len={len}");
        }
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    #[test]
    fn best_n_within_vec_of_arenas_matches_flat_vec_f64_simd() {
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
        let max_qty = NonZero::new(5).unwrap();
        let max_dist = 0.2;

        let flat_tree: KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points).unwrap();
        let arena_tree: KdTree<f64, u32, Eytzinger, VecOfArenas<f64, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points).unwrap();

        let flat_result = flat_tree
            .query(&query)
            .best_n_within::<SquaredEuclidean<f64>>(max_dist, max_qty)
            .execute()
            .into_sorted_vec();
        let arena_result = arena_tree
            .query(&query)
            .best_n_within::<SquaredEuclidean<f64>>(max_dist, max_qty)
            .execute()
            .into_sorted_vec();

        assert_eq!(arena_result, flat_result);
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    #[test]
    fn best_n_within_vec_of_arenas_matches_flat_vec_f32_simd() {
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
        let max_qty = NonZero::new(5).unwrap();
        let max_dist = 0.2f32;

        let flat_tree: KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points).unwrap();
        let arena_tree: KdTree<f32, u32, Eytzinger, VecOfArenas<f32, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points).unwrap();

        let flat_result = flat_tree
            .query(&query)
            .best_n_within::<SquaredEuclidean<f32>>(max_dist, max_qty)
            .execute()
            .into_sorted_vec();
        let arena_result = arena_tree
            .query(&query)
            .best_n_within::<SquaredEuclidean<f32>>(max_dist, max_qty)
            .execute()
            .into_sorted_vec();

        assert_eq!(arena_result, flat_result);
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    #[test]
    fn best_n_within_vec_of_arenas_matches_flat_vec_f64_manhattan_simd() {
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
        let max_qty = NonZero::new(5).unwrap();
        let max_dist = 0.4f64;

        let flat_tree: KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points).unwrap();
        let arena_tree: KdTree<f64, u32, Eytzinger, VecOfArenas<f64, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points).unwrap();

        let flat_result = flat_tree
            .query(&query)
            .best_n_within::<crate::dist::Manhattan<f64>>(max_dist, max_qty)
            .execute()
            .into_sorted_vec();
        let arena_result = arena_tree
            .query(&query)
            .best_n_within::<crate::dist::Manhattan<f64>>(max_dist, max_qty)
            .execute()
            .into_sorted_vec();

        assert_eq!(arena_result, flat_result);
    }

    fn assert_best_neighbours_close_f64<T>(
        actual: &[BestQueryResultItem<(), T, f64>],
        expected: &[BestQueryResultItem<(), T, f64>],
    ) where
        T: Debug + PartialEq,
    {
        assert_eq!(actual.len(), expected.len());

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_eq!(actual.item, expected.item);
            assert!(
                ulps_diff_f64(actual.distance, expected.distance) <= 2,
                "distance mismatch: actual={:?} expected={:?}",
                actual.distance,
                expected.distance
            );
        }
    }

    fn ulps_diff_f64(a: f64, b: f64) -> u64 {
        canonical_u64(a).abs_diff(canonical_u64(b))
    }

    fn canonical_u64(value: f64) -> u64 {
        let bits = value.to_bits();
        if (bits >> 63) != 0 {
            !bits
        } else {
            bits | (1 << 63)
        }
    }

    fn linear_search(
        content: &[[f64; 2]],
        query: &[f64; 2],
        radius: f64,
        max_qty: usize,
    ) -> Vec<BestQueryResultItem<(), u32, f64>> {
        let mut best_items = Vec::with_capacity(max_qty);

        for (item, p) in content.iter().enumerate() {
            let distance = squared_euclidean_dist(query, p);
            if distance <= radius {
                if best_items.len() < max_qty {
                    best_items.push(BestQueryResultItem {
                        point: (),
                        distance,
                        item: item as u32,
                    });
                } else if (item as u32) < best_items.last().unwrap().item {
                    best_items.pop().unwrap();
                    best_items.push(BestQueryResultItem {
                        point: (),
                        distance,
                        item: item as u32,
                    });
                }
            }
            best_items.sort_unstable();
        }
        best_items.reverse();

        best_items
    }

    fn squared_euclidean_dist<const K: usize>(a: &[f64; K], b: &[f64; K]) -> f64 {
        let aw = (*a).map(|coord| {
            <crate::dist::SquaredEuclidean<f64> as crate::dist::DistanceMetricCore<f64>>::widen_coord(
                coord,
            )
        });
        let bw = (*b).map(|coord| {
            <crate::dist::SquaredEuclidean<f64> as crate::dist::DistanceMetricCore<f64>>::widen_coord(
                coord,
            )
        });

        <crate::dist::SquaredEuclidean<f64> as crate::dist::DistanceMetricCore<f64>>::dist::<K>(
            &aw, &bw,
        )
    }
}

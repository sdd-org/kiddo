use std::num::NonZero;

use crate::dist::DistanceMetric;
use crate::kd_tree::query_stack::StackTrait;
use crate::leaf_view::TlsLeafScratch;
use crate::stem_strategy::donnelly::simd_full::{
    BacktrackBlock3, BacktrackBlock4, SimdSelectBestChildBlock3,
};
use crate::{Axis, Content, KdTree, LeafStrategy, QueryResultItem, StemStrategy};

impl<A, T, SS, LS, const K: usize, const B: usize, const ITEM_LEAF_MODE: u8>
    KdTree<A, T, SS, LS, K, B, ITEM_LEAF_MODE>
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    LS: LeafStrategy<A, T, SS, K, B>,
    SS: StemStrategy,
{
    /// Finds the N nearest points to the query point.
    ///
    /// If `sorted` is true, results are returned in order of increasing distance.
    pub(crate) fn nearest_n<D>(
        &self,
        query: &[A; K],
        max_qty: NonZero<usize>,
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
        SS::Stack<D::Output, K>: StackTrait<D::Output, SS, K> + 'static,
    {
        self.nearest_n_within::<D>(query, D::Output::max_value(), max_qty, sorted)
    }

    pub(crate) fn nearest_n_with_scratch<D>(
        &self,
        query: &[A; K],
        max_qty: NonZero<usize>,
        sorted: bool,
        stack: &mut SS::Stack<D::Output, K>,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
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
        self.nearest_n_within_with_scratch::<D>(
            query,
            D::Output::max_value(),
            max_qty,
            sorted,
            stack,
        )
    }
}

#[cfg(test)]
mod tests {
    use az::{Az, Cast};
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    use std::array;
    use std::fmt::Debug;
    use std::num::NonZero;

    use crate::dist::SquaredEuclidean;
    use crate::kd_tree::KdTree;
    use crate::leaf_strategy::{FlatVec, VecOfArenas, VecOfArrays};
    use crate::Axis;
    use crate::Eytzinger;

    const RNG_SEED: u64 = 42;
    const TILE_BOUNDARY_CASES: [usize; 7] = [1, 2, 4, 8, 32, 33, 47];

    #[test]
    fn nearest_n_vec_of_arenas_matches_flat_vec_across_tile_boundaries() {
        let query = [0.37f32, 0.49, 0.58];
        let max_qty = NonZero::new(5).unwrap();

        for &len in &TILE_BOUNDARY_CASES {
            let points: Vec<[f32; 3]> = (0..len)
                .map(|idx| {
                    [
                        ((idx * 3) % 97) as f32 / 97.0,
                        ((idx * 11 + 1) % 97) as f32 / 97.0,
                        ((idx * 19 + 2) % 97) as f32 / 97.0,
                    ]
                })
                .collect();

            let flat_tree: KdTree<f32, usize, Eytzinger, FlatVec<f32, usize, 3, 32>, 3, 32> =
                KdTree::new_from_slice(&points).unwrap();
            let arena_tree: KdTree<f32, usize, Eytzinger, VecOfArenas<f32, usize, 3, 32>, 3, 32> =
                KdTree::new_from_slice(&points).unwrap();

            let mut flat: Vec<(f32, usize)> = flat_tree
                .nearest_n::<SquaredEuclidean<f32>>(&query, max_qty, true)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();
            let mut arena: Vec<(f32, usize)> = arena_tree
                .nearest_n::<SquaredEuclidean<f32>>(&query, max_qty, true)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            sort_by_distance_then_item_f32(&mut flat);
            sort_by_distance_then_item_f32(&mut arena);

            assert_eq!(arena, flat, "len={len}");
        }
    }

    #[test]
    fn v6_query_nearest_n_large_f64_flat_vec() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;

        let max_qty = NonZero::new(10).unwrap();

        let content_to_add: Vec<[f64; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f64; 4]>()).collect();

        let tree: KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&content_to_add).unwrap();

        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[f64; 4]> =
            (0..NUM_QUERIES).map(|_| rng.random::<[f64; 4]>()).collect();

        for query_point in query_points {
            let expected = linear_search(&content_to_add, max_qty.into(), &query_point);

            let result: Vec<_> = tree
                .nearest_n::<SquaredEuclidean<f64>>(&query_point, max_qty, true)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            assert_distance_item_pairs_close_f64(&result, &expected);
        }
    }

    #[test]
    fn v6_query_nearest_n_large_f64_vec_of_arrays() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;

        let max_qty = NonZero::new(10).unwrap();

        let content_to_add: Vec<[f64; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f64; 4]>()).collect();

        let tree: KdTree<f64, u32, Eytzinger, VecOfArrays<f64, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&content_to_add).unwrap();

        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[f64; 4]> =
            (0..NUM_QUERIES).map(|_| rng.random::<[f64; 4]>()).collect();

        for query_point in query_points {
            let expected = linear_search(&content_to_add, max_qty.into(), &query_point);

            let result: Vec<_> = tree
                .nearest_n::<SquaredEuclidean<f64>>(&query_point, max_qty, true)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            assert_distance_item_pairs_close_f64(&result, &expected);
        }
    }

    #[test]
    fn v6_query_nearest_n_large_f64_vec_of_arrays_mutated_f64() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;

        let max_qty = NonZero::new(10).unwrap();

        let content_to_add: Vec<[f64; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f64; 4]>()).collect();

        let mut tree: KdTree<f64, u32, Eytzinger, VecOfArrays<f64, u32, 4, 32>, 4, 32> =
            KdTree::default();

        for (idx, point) in content_to_add.iter().enumerate() {
            tree.add(point, idx as u32).unwrap();
        }

        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[f64; 4]> =
            (0..NUM_QUERIES).map(|_| rng.random::<[f64; 4]>()).collect();

        for query_point in query_points {
            let expected = linear_search(&content_to_add, max_qty.into(), &query_point);

            let result: Vec<_> = tree
                .nearest_n::<SquaredEuclidean<f64>>(&query_point, max_qty, true)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            assert_distance_item_pairs_close_f64(&result, &expected);
        }
    }

    fn assert_distance_item_pairs_close_f64<T>(actual: &[(f64, T)], expected: &[(f64, T)])
    where
        T: Debug + PartialEq,
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

    fn linear_search<A, R, const K: usize>(
        content: &[[A; K]],
        qty: usize,
        query_point: &[A; K],
    ) -> Vec<(A, R)>
    where
        A: Axis<Coord = A>,
        usize: Cast<R>,
        SquaredEuclidean<A>: crate::dist::DistanceMetricCore<A, Output = A>,
    {
        let mut results: Vec<(A, R)> = vec![];

        for (idx, p) in content.iter().enumerate() {
            let dist = squared_euclidean_dist(query_point, p);
            if results.len() < qty {
                results.push((dist, idx.az::<R>()));
                results.sort_by(|(a_dist, _), (b_dist, _)| a_dist.partial_cmp(b_dist).unwrap());
            } else if dist < results[qty - 1].0 {
                results[qty - 1] = (dist, idx.az::<R>());
                results.sort_by(|(a_dist, _), (b_dist, _)| a_dist.partial_cmp(b_dist).unwrap());
            }
        }

        results
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

    fn random_point_f32<const K: usize>(rng: &mut StdRng) -> [f32; K] {
        array::from_fn(|_| rng.random_range(-1.0f32..1.0f32))
    }

    fn random_point_f64<const K: usize>(rng: &mut StdRng) -> [f64; K] {
        array::from_fn(|_| rng.random_range(-1.0f64..1.0f64))
    }

    fn sort_by_distance_then_item_f32(items: &mut [(f32, usize)]) {
        items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then_with(|| a.1.cmp(&b.1)));
    }

    fn sort_by_distance_then_item_f64(items: &mut [(f64, usize)]) {
        items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then_with(|| a.1.cmp(&b.1)));
    }

    #[test]
    fn v6_query_nearest_n_unsorted_respects_max_qty_mutable_f32_regression() {
        const K: usize = 2;
        const B: usize = 32;
        const POINT_COUNT: usize = 1024;
        const CONTENT_SEED: u64 = 1_260_253_197;
        const QUERY_SEED: u64 = 13_787_848_794_416_797_126;

        let mut rng_content = StdRng::seed_from_u64(CONTENT_SEED);
        let points: Vec<[f32; K]> = (0..POINT_COUNT)
            .map(|_| random_point_f32::<K>(&mut rng_content))
            .collect();

        let mut tree: KdTree<f32, usize, Eytzinger, VecOfArrays<f32, usize, K, B>, K, B> =
            KdTree::default();
        for (idx, point) in points.iter().enumerate() {
            tree.add(point, idx).unwrap();
        }

        let mut rng_query = StdRng::seed_from_u64(QUERY_SEED);
        let query = random_point_f32::<K>(&mut rng_query);
        let max_qty = NonZero::new(20).unwrap();

        let mut unsorted: Vec<(f32, usize)> = tree
            .nearest_n::<SquaredEuclidean<f32>>(&query, max_qty, false)
            .into_iter()
            .map(|n| (n.distance, n.item))
            .collect();

        assert_eq!(unsorted.len(), max_qty.get());

        let mut sorted: Vec<(f32, usize)> = tree
            .nearest_n::<SquaredEuclidean<f32>>(&query, max_qty, true)
            .into_iter()
            .map(|n| (n.distance, n.item))
            .collect();

        sort_by_distance_then_item_f32(&mut unsorted);
        sort_by_distance_then_item_f32(&mut sorted);

        assert_eq!(unsorted, sorted);
    }

    #[test]
    fn v6_query_nearest_n_unsorted_respects_max_qty_mutable_f64_regression() {
        const K: usize = 2;
        const B: usize = 32;
        const POINT_COUNT: usize = 1024;
        const CONTENT_SEED: u64 = 1_260_253_197;
        const QUERY_SEED: u64 = 13_787_848_794_416_797_126;

        let mut rng_content = StdRng::seed_from_u64(CONTENT_SEED);
        let points: Vec<[f64; K]> = (0..POINT_COUNT)
            .map(|_| random_point_f64::<K>(&mut rng_content))
            .collect();

        let mut tree: KdTree<f64, usize, Eytzinger, VecOfArrays<f64, usize, K, B>, K, B> =
            KdTree::default();
        for (idx, point) in points.iter().enumerate() {
            tree.add(point, idx).unwrap();
        }

        let mut rng_query = StdRng::seed_from_u64(QUERY_SEED);
        let query = random_point_f64::<K>(&mut rng_query);
        let max_qty = NonZero::new(15).unwrap();

        let mut unsorted: Vec<(f64, usize)> = tree
            .nearest_n::<SquaredEuclidean<f64>>(&query, max_qty, false)
            .into_iter()
            .map(|n| (n.distance, n.item))
            .collect();

        assert_eq!(unsorted.len(), max_qty.get());

        let mut sorted: Vec<(f64, usize)> = tree
            .nearest_n::<SquaredEuclidean<f64>>(&query, max_qty, true)
            .into_iter()
            .map(|n| (n.distance, n.item))
            .collect();

        sort_by_distance_then_item_f64(&mut unsorted);
        sort_by_distance_then_item_f64(&mut sorted);

        assert_eq!(unsorted, sorted);
    }
}

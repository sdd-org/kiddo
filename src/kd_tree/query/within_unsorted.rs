use crate::dist::DistanceMetric;
use crate::kd_tree::query_context::QueryContext;
use crate::kd_tree::query_stack::StackTrait;
use crate::kd_tree::KdTreeQueryOps;
use crate::leaf_view::TlsLeafScratch;
use crate::leaf_view_chunked::nearest_n_within::{
    nearest_n_within_with_query_wide, nearest_n_within_with_query_wide_arena,
};
use crate::results::result_collection::{FromLeafCandidate, VisitorResultCollection};
use crate::stem_strategy::donnelly::simd_full::{
    BacktrackBlock3, BacktrackBlock4, SimdSelectBestChildBlock3,
};
use crate::traits::leaf_strategy::LeafProjection;
use crate::{Axis, Content, KdTree, LeafStrategy, QueryResultItem, StemStrategy};

impl<A, T, SS, LS, const K: usize, const B: usize, const ITEM_LEAF_MODE: u8>
    KdTree<A, T, SS, LS, K, B, ITEM_LEAF_MODE>
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    LS: LeafStrategy<A, T, SS, K, B>,
    SS: StemStrategy,
{
    #[inline(always)]
    fn process_leaf_within_unsorted_visit<D, E, F, const EXCLUSIVE: bool>(
        &self,
        leaf_idx: usize,
        query_wide: &[D::Output; K],
        max_dist: D::Output,
        visitor: &mut F,
    ) where
        D: DistanceMetric<A>,
        D::Output: Axis<Coord = D::Output> + TlsLeafScratch + 'static,
        E: FromLeafCandidate<A, T, D::Output, K>,
        F: FnMut(E),
    {
        let mut results = VisitorResultCollection::new(visitor);

        match LS::LEAF_PROJECTION {
            LeafProjection::LeafArena => {
                let arena = self.leaves.leaf_arena(leaf_idx);
                nearest_n_within_with_query_wide_arena::<A, T, D, E, _, EXCLUSIVE, K>(
                    &arena,
                    query_wide,
                    max_dist,
                    &mut results,
                );
            }
            LeafProjection::LeafView => {
                let leaf = self.leaves.leaf_view(leaf_idx);
                nearest_n_within_with_query_wide::<A, T, D, E, _, EXCLUSIVE, K, B>(
                    &leaf,
                    query_wide,
                    max_dist,
                    &mut results,
                );
            }
        }
    }

    #[inline]
    pub(crate) fn within_unsorted_visit_impl<D, F, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
        mut visitor: F,
    ) where
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output, K>: StackTrait<D::Output, SS, K> + 'static,
        F: FnMut(QueryResultItem<(), T, D::Output>),
    {
        let mut req_ctx = WithinUnsortedVisitReqCtx::<A, D::Output, EXCLUSIVE, K> {
            query,
            max_dist,
            _phantom: std::marker::PhantomData,
        };

        self.backtracking_query::<_, _, D>(&mut req_ctx, |leaf_idx, query_wide, req_ctx| {
            self.process_leaf_within_unsorted_visit::<
                D,
                QueryResultItem<(), T, D::Output>,
                F,
                EXCLUSIVE,
            >(
                leaf_idx,
                query_wide,
                req_ctx.max_dist(),
                &mut visitor,
            );
        });
    }

    #[inline]
    pub(crate) fn within_unsorted_visit_with_points_impl<D, F, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
        mut visitor: F,
    ) where
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output, K>: StackTrait<D::Output, SS, K> + 'static,
        F: FnMut(QueryResultItem<[A; K], T, D::Output>),
    {
        let mut req_ctx = WithinUnsortedVisitReqCtx::<A, D::Output, EXCLUSIVE, K> {
            query,
            max_dist,
            _phantom: std::marker::PhantomData,
        };

        self.backtracking_query::<_, _, D>(&mut req_ctx, |leaf_idx, query_wide, req_ctx| {
            self.process_leaf_within_unsorted_visit::<
                D,
                QueryResultItem<[A; K], T, D::Output>,
                F,
                EXCLUSIVE,
            >(
                leaf_idx,
                query_wide,
                req_ctx.max_dist(),
                &mut visitor,
            );
        });
    }

    #[inline]
    pub(crate) fn within_unsorted_visit_impl_with_scratch<D, F, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
        mut visitor: F,
        stack: &mut SS::Stack<D::Output, K>,
    ) where
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output, K>: StackTrait<D::Output, SS, K>,
        F: FnMut(QueryResultItem<(), T, D::Output>),
    {
        let mut req_ctx = WithinUnsortedVisitReqCtx::<A, D::Output, EXCLUSIVE, K> {
            query,
            max_dist,
            _phantom: std::marker::PhantomData,
        };

        self.backtracking_query_with_scratch::<_, _, D>(
            &mut req_ctx,
            stack,
            |leaf_idx, query_wide, req_ctx| {
                self.process_leaf_within_unsorted_visit::<
                    D,
                    QueryResultItem<(), T, D::Output>,
                    F,
                    EXCLUSIVE,
                >(
                    leaf_idx,
                    query_wide,
                    req_ctx.max_dist(),
                    &mut visitor,
                );
            },
        );
    }

    /// Returns a streaming iterator over all points within a given distance, unsorted.
    ///
    /// This avoids materializing the full result set returned by
    /// within + unsorted queries. The iterator keeps traversal state and
    /// per-leaf matches inline in the common case, spilling to heap allocation only if the
    /// tree depth or a single leaf's match count exceeds the inline capacities.
    ///
    /// For the absolute lowest overhead, use [`within_unsorted_visit`](Self::within_unsorted_visit).
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn within_unsorted_iter<D>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
    ) -> crate::kd_tree::WithinUnsortedIter<'_, Self, A, T, SS, LS, D, false, K, B>
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
        crate::kd_tree::WithinUnsortedIter::new(self, query, max_dist)
    }
}

#[allow(missing_docs)]
#[cfg(feature = "cargo_asm")]
pub mod cargo_asm {
    use crate::dist::SquaredEuclidean;
    use crate::kd_tree::KdTree;
    use crate::leaf_strategy::VecOfArenas;
    use crate::Eytzinger;

    const K: usize = 3;
    const BUCKET_SIZE: usize = 32;
    const MAX_DIST: f64 = 0.0025;

    type ArenaLeaves = VecOfArenas<f64, u32, K, BUCKET_SIZE>;
    type EytzingerPfFarKdT = KdTree<f64, u32, Eytzinger, ArenaLeaves, K, BUCKET_SIZE>;

    /// Hook for cargo-asm to render the exact within_unsorted path for scalar Eytzinger PF-far arena leaves.
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_within_unsorted_eytzinger_pf_far_vec_of_arenas_cargo_asm_hook(
        tree: &EytzingerPfFarKdT,
        query: [f64; 3],
    ) -> (usize, u64, u64) {
        let results = tree
            .query(&query)
            .within::<SquaredEuclidean<f64>>(MAX_DIST)
            .unsorted()
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

struct WithinUnsortedVisitReqCtx<'a, A, O, const EXCLUSIVE: bool, const K: usize>
where
    O: Axis<Coord = O>,
{
    query: &'a [A; K],
    max_dist: O,
    _phantom: std::marker::PhantomData<A>,
}

impl<A, O, const EXCLUSIVE: bool, const K: usize> QueryContext<A, O, K>
    for WithinUnsortedVisitReqCtx<'_, A, O, EXCLUSIVE, K>
where
    O: Axis<Coord = O>,
{
    #[inline(always)]
    fn query(&self) -> &[A; K] {
        self.query
    }

    #[inline(always)]
    fn max_dist(&self) -> O {
        self.max_dist
    }

    #[inline(always)]
    fn prune_on_equal_max_dist(&self) -> bool {
        EXCLUSIVE
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    use std::cmp::Ordering;

    use crate::dist::SquaredEuclidean;
    use crate::kd_tree::KdTree;
    use crate::leaf_strategy::{FlatVec, VecOfArenas, VecOfArrays};
    use crate::Axis;
    use crate::Eytzinger;

    const RNG_SEED: u64 = 42;
    const TILE_BOUNDARY_CASES: [usize; 7] = [1, 2, 4, 8, 32, 33, 47];

    #[test]
    fn within_unsorted_exclusive_boundaries_match_visit_and_iter() {
        let points = vec![[0.0f64, 0.0], [1.0, 0.0], [2.0, 0.0], [0.5, 0.0]];
        let tree: KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, 2, 32>, 2, 32> =
            KdTree::new_from_slice(&points).unwrap();
        let query = [0.0, 0.0];

        let mut inclusive_materialized: Vec<_> = tree
            .query(&query)
            .within::<SquaredEuclidean<f64>>(1.0)
            .unsorted()
            .execute()
            .into_iter()
            .map(|n| n.item)
            .collect();
        let mut exclusive_materialized: Vec<_> = tree
            .query(&query)
            .within::<SquaredEuclidean<f64>>(1.0)
            .exclusive_boundaries()
            .unsorted()
            .execute()
            .into_iter()
            .map(|n| n.item)
            .collect();
        let mut exclusive_visit = Vec::new();
        tree.query(&query)
            .within::<SquaredEuclidean<f64>>(1.0)
            .exclusive_boundaries()
            .unsorted()
            .visit(|n| exclusive_visit.push(n.item));
        let mut exclusive_iter: Vec<_> = tree
            .query(&query)
            .within::<SquaredEuclidean<f64>>(1.0)
            .exclusive_boundaries()
            .unsorted()
            .iter()
            .map(|n| n.item)
            .collect();

        inclusive_materialized.sort_unstable();
        exclusive_materialized.sort_unstable();
        exclusive_visit.sort_unstable();
        exclusive_iter.sort_unstable();

        assert_eq!(inclusive_materialized, vec![0, 1, 3]);
        assert_eq!(exclusive_materialized, vec![0, 3]);
        assert_eq!(exclusive_visit, exclusive_materialized);
        assert_eq!(exclusive_iter, exclusive_materialized);
    }

    #[test]
    fn within_unsorted_vec_of_arenas_matches_flat_vec_across_tile_boundaries() {
        let query = [0.29f32, 0.41, 0.53];
        let radius = 0.2;

        for &len in &TILE_BOUNDARY_CASES {
            let points: Vec<[f32; 3]> = (0..len)
                .map(|idx| {
                    [
                        ((idx * 7) % 97) as f32 / 97.0,
                        ((idx * 17 + 1) % 97) as f32 / 97.0,
                        ((idx * 29 + 2) % 97) as f32 / 97.0,
                    ]
                })
                .collect();

            let flat_tree: KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, 3, 32>, 3, 32> =
                KdTree::new_from_slice(&points).unwrap();
            let arena_tree: KdTree<f32, u32, Eytzinger, VecOfArenas<f32, u32, 3, 32>, 3, 32> =
                KdTree::new_from_slice(&points).unwrap();

            let mut flat: Vec<(f32, u32)> = flat_tree
                .query(&query)
                .within::<SquaredEuclidean<f32>>(radius)
                .unsorted()
                .execute()
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();
            let mut arena: Vec<(f32, u32)> = arena_tree
                .query(&query)
                .within::<SquaredEuclidean<f32>>(radius)
                .unsorted()
                .execute()
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            stabilize_sort(&mut flat);
            stabilize_sort(&mut arena);

            assert_eq!(arena, flat, "len={len}");

            let mut flat_with_points = flat_tree
                .query(&query)
                .within::<SquaredEuclidean<f32>>(radius)
                .unsorted()
                .with_points()
                .execute();
            let mut arena_with_points = arena_tree
                .query(&query)
                .within::<SquaredEuclidean<f32>>(radius)
                .unsorted()
                .with_points()
                .execute();
            flat_with_points.sort_unstable_by_key(|result| result.item);
            arena_with_points.sort_unstable_by_key(|result| result.item);

            assert_eq!(arena_with_points, flat_with_points, "len={len}");
            assert!(arena_with_points
                .iter()
                .all(|result| result.point == points[result.item as usize]));
        }
    }

    #[test]
    fn within_unsorted_visit_and_iter_match_materialized_results() {
        let points: Vec<[f64; 3]> = (0..257)
            .map(|idx| {
                [
                    ((idx * 7) % 101) as f64 / 101.0,
                    ((idx * 17 + 3) % 101) as f64 / 101.0,
                    ((idx * 31 + 5) % 101) as f64 / 101.0,
                ]
            })
            .collect();
        type Tree = KdTree<f64, u32, Eytzinger, VecOfArenas<f64, u32, 3, 32>, 3, 32>;
        let tree: Tree = KdTree::new_from_slice(&points).unwrap();
        let query = [0.37, 0.41, 0.43];
        let radius = 0.12;

        let mut materialized: Vec<(u32, u64)> = tree
            .query(&query)
            .within::<SquaredEuclidean<f64>>(radius)
            .unsorted()
            .execute()
            .into_iter()
            .map(|result| (result.item, result.distance.to_bits()))
            .collect();
        let mut visited = Vec::new();
        tree.query(&query)
            .within::<SquaredEuclidean<f64>>(radius)
            .unsorted()
            .visit(|result| {
                visited.push((result.item, result.distance.to_bits()));
            });
        let mut iterated: Vec<(u32, u64)> = tree
            .query(&query)
            .within::<SquaredEuclidean<f64>>(radius)
            .unsorted()
            .iter()
            .map(|result| (result.item, result.distance.to_bits()))
            .collect();

        materialized.sort_unstable();
        visited.sort_unstable();
        iterated.sort_unstable();

        assert_eq!(visited, materialized);
        assert_eq!(iterated, materialized);
    }

    #[test]
    fn iter_visits_every_point_item_pair() {
        let points: Vec<[f64; 3]> = (0..129)
            .map(|idx| [idx as f64, (idx * 2) as f64, (idx * 3) as f64])
            .collect();
        type Tree = KdTree<f64, u32, Eytzinger, VecOfArenas<f64, u32, 3, 32>, 3, 32>;
        let tree: Tree = KdTree::new_from_slice(&points).unwrap();

        let visited: Vec<(u32, [f64; 3])> = tree.iter().collect();

        assert_eq!(visited.len(), points.len());
        for (item, point) in visited {
            assert_eq!(point, points[item as usize]);
        }
    }

    #[test]
    fn v6_query_within_unsorted_large_f32_flat_vec() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        const RADIUS: f32 = 0.2;

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

        let tree: KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&content_to_add).unwrap();

        let query_points: Vec<[f32; 4]> =
            (0..NUM_QUERIES).map(|_| rng.random::<[f32; 4]>()).collect();

        for query_point in query_points {
            let expected = linear_search(&content_to_add, &query_point, RADIUS);

            let mut result: Vec<_> = tree
                .query(&query_point)
                .within::<SquaredEuclidean<f32>>(RADIUS)
                .unsorted()
                .execute()
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            stabilize_sort(&mut result);
            assert_distance_item_pairs_close_f32_by_item(&result, &expected, RADIUS);
        }
    }

    #[test]
    fn v6_query_within_unsorted_f32_flat_vec_no_items() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        const TREE_SIZE: usize = 1_000;
        const NUM_QUERIES: usize = 1;
        const RADIUS: f32 = 0.2;

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

        let tree: KdTree<f32, (), Eytzinger, FlatVec<f32, (), 4, 32>, 4, 32> =
            KdTree::new_from_slice_no_items(&content_to_add).unwrap();

        let query_points: Vec<[f32; 4]> =
            (0..NUM_QUERIES).map(|_| rng.random::<[f32; 4]>()).collect();

        for query_point in query_points {
            let expected: Vec<_> = linear_search(&content_to_add, &query_point, RADIUS)
                .into_iter()
                .map(|(distance, _)| (distance, ()))
                .collect();

            let mut result: Vec<_> = tree
                .query(&query_point)
                .within::<SquaredEuclidean<f32>>(RADIUS)
                .unsorted()
                .execute()
                .into_iter()
                .map(|n| (n.distance, 1))
                .collect();

            stabilize_sort(&mut result);

            let result: Vec<_> = result
                .into_iter()
                .map(|(distance, _)| (distance, ()))
                .collect();

            assert_distance_item_pairs_close_f32(&result, &expected);
        }
    }

    #[test]
    fn v6_query_within_unsorted_large_f32_vec_of_arrays() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        const RADIUS: f32 = 0.2;

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

        let tree: KdTree<f32, u32, Eytzinger, VecOfArrays<f32, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&content_to_add).unwrap();

        let query_points: Vec<[f32; 4]> =
            (0..NUM_QUERIES).map(|_| rng.random::<[f32; 4]>()).collect();

        for query_point in query_points {
            let expected = linear_search(&content_to_add, &query_point, RADIUS);

            let mut result: Vec<_> = tree
                .query(&query_point)
                .within::<SquaredEuclidean<f32>>(RADIUS)
                .unsorted()
                .execute()
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            stabilize_sort(&mut result);
            assert_distance_item_pairs_close_f32_by_item(&result, &expected, RADIUS);
        }
    }

    #[test]
    fn v6_query_within_unsorted_large_f32_vec_of_arrays_mutated() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        const RADIUS: f32 = 0.2;

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

        let mut tree: KdTree<f32, u32, Eytzinger, VecOfArrays<f32, u32, 4, 32>, 4, 32> =
            KdTree::default();

        for (idx, point) in content_to_add.iter().enumerate() {
            tree.add(point, idx as u32).unwrap();
        }

        let query_points: Vec<[f32; 4]> =
            (0..NUM_QUERIES).map(|_| rng.random::<[f32; 4]>()).collect();

        for query_point in query_points {
            let expected = linear_search(&content_to_add, &query_point, RADIUS);

            let mut result: Vec<_> = tree
                .query(&query_point)
                .within::<SquaredEuclidean<f32>>(RADIUS)
                .unsorted()
                .execute()
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            stabilize_sort(&mut result);
            assert_distance_item_pairs_close_f32_by_item(&result, &expected, RADIUS);
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
                ulps_diff_f32(*actual_dist, *expected_dist) <= 3,
                "distance mismatch: actual={actual_dist:?} expected={expected_dist:?}"
            );
        }
    }

    fn assert_distance_item_pairs_close_f32_by_item(
        actual: &[(f32, u32)],
        expected: &[(f32, u32)],
        radius: f32,
    ) {
        use std::collections::BTreeMap;

        let actual_by_item: BTreeMap<u32, f32> =
            actual.iter().map(|(dist, item)| (*item, *dist)).collect();
        let expected_by_item: BTreeMap<u32, f32> =
            expected.iter().map(|(dist, item)| (*item, *dist)).collect();

        for (item, actual_dist) in &actual_by_item {
            if let Some(expected_dist) = expected_by_item.get(item) {
                assert!(
                    ulps_diff_f32(*actual_dist, *expected_dist) <= 3,
                    "distance mismatch for item {item}: actual={actual_dist:?} expected={expected_dist:?}"
                );
            } else {
                assert!(
                    ulps_diff_f32(*actual_dist, radius) <= 3,
                    "unexpected item {item} with distance {actual_dist:?} exceeded 3 ULP radius tolerance from {radius:?}"
                );
            }
        }

        for (item, expected_dist) in &expected_by_item {
            if !actual_by_item.contains_key(item) {
                assert!(
                    ulps_diff_f32(*expected_dist, radius) <= 3,
                    "missing item {item} with distance {expected_dist:?} exceeded 3 ULP radius tolerance from {radius:?}"
                );
            }
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
        A: Axis<Coord = A> + 'static,
        SquaredEuclidean<A>: crate::dist::DistanceMetricCore<A, Output = A>,
    {
        let aw = (*a).map(|coord| {
            <crate::dist::SquaredEuclidean<A> as crate::dist::DistanceMetricCore<A>>::widen_coord(
                coord,
            )
        });
        let bw = (*b).map(|coord| {
            <crate::dist::SquaredEuclidean<A> as crate::dist::DistanceMetricCore<A>>::widen_coord(
                coord,
            )
        });

        <crate::dist::SquaredEuclidean<A> as crate::dist::DistanceMetricCore<A>>::dist::<K>(
            &aw, &bw,
        )
    }

    fn stabilize_sort<A>(matching_items: &mut [(A, u32)])
    where
        A: Axis<Coord = A> + 'static,
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
}

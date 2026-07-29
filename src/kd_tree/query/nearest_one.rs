use crate::dist::DistanceMetric;
use crate::kd_tree::query_context::QueryContext;
use crate::kd_tree::query_stack::StackTrait;
use crate::kd_tree::KdTreeQueryOps;
use crate::leaf_view::TlsLeafScratch;
use crate::leaf_view_chunked::nearest_one::nearest_one_with_query_wide;
use crate::stem_strategy::donnelly::simd_full::{
    BacktrackBlock3, BacktrackBlock4, SimdSelectBestChildBlock3,
};
use crate::traits::leaf_strategy::LeafProjection;
use crate::{Axis, Content, KdTree, LeafStrategy, StemStrategy};

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: Axis<Coord = A> + 'static,
    T: Content,
    LS: LeafStrategy<A, T, SS, K, B>,
    SS: StemStrategy,
{
    #[inline(always)]
    fn process_leaf_nearest_one<D>(
        &self,
        leaf_idx: usize,
        query_wide: &[D::Output; K],
        best_dist: &mut D::Output,
        best_item: &mut T,
    ) where
        D: DistanceMetric<A>,
        D::Output: Axis<Coord = D::Output> + 'static,
    {
        #[cfg(feature = "test_utils")]
        crate::test_utils::exact_query_stats::record_leaf_visit();
        #[cfg(feature = "test_utils")]
        crate::test_utils::exact_query_trace::push(
            crate::test_utils::exact_query_trace::ExactQueryTraceEvent::LeafVisit { leaf_idx },
        );

        match LS::LEAF_PROJECTION {
            LeafProjection::LeafArena => {
                let arena = self.leaves.leaf_arena(leaf_idx);
                crate::leaf_view_chunked::nearest_one::nearest_one_with_query_wide_arena::<
                    A,
                    T,
                    D,
                    K,
                >(&arena, query_wide, best_dist, best_item);
            }
            LeafProjection::LeafView => {
                let leaf = self.leaves.leaf_view(leaf_idx);
                nearest_one_with_query_wide::<A, T, D, K, B>(
                    &leaf, query_wide, best_dist, best_item,
                );
            }
        }
    }

    /// Finds the nearest point to the query point.
    ///
    /// Returns a tuple of (distance, item) for the nearest neighbor.
    #[inline(always)]
    pub(crate) fn nearest_one<D>(&self, query: &[A; K]) -> (D::Output, T)
    where
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + Default + 'static,
        SS: 'static,
    {
        if self.stem_leaf_resolution.uses_arithmetic() {
            return self.nearest_one_arithmetic::<D>(query);
        }

        self.nearest_one_mapped::<D>(query)
    }

    #[inline(always)]
    fn nearest_one_mapped<D>(&self, query: &[A; K]) -> (D::Output, T)
    where
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + Default + 'static,
    {
        let mut req_ctx = NearestOneReqCtx {
            query,
            best_dist: D::Output::max_value(),
            best_item: T::default(),
        };

        self.backtracking_query::<_, _, D>(&mut req_ctx, |leaf_idx, query_wide, query_ctx| {
            self.process_leaf_nearest_one::<D>(
                leaf_idx,
                query_wide,
                &mut query_ctx.best_dist,
                &mut query_ctx.best_item,
            );
        });

        (req_ctx.best_dist, req_ctx.best_item)
    }

    #[inline(always)]
    fn nearest_one_arithmetic<D>(&self, query: &[A; K]) -> (D::Output, T)
    where
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + Default + 'static,
        SS: 'static,
    {
        if !SS::SUPPORTS_ARITHMETIC_LEAF_RESOLUTION {
            return self.nearest_one_mapped::<D>(query);
        }

        let mut req_ctx = NearestOneReqCtx {
            query,
            best_dist: D::Output::max_value(),
            best_item: T::default(),
        };

        self.arithmetic_query::<_, _, D>(&mut req_ctx, |leaf_idx, query_wide, query_ctx| {
            self.process_leaf_nearest_one::<D>(
                leaf_idx,
                query_wide,
                &mut query_ctx.best_dist,
                &mut query_ctx.best_item,
            );
        });

        (req_ctx.best_dist, req_ctx.best_item)
    }

    #[inline(always)]
    fn nearest_one_arithmetic_with_scratch<D>(
        &self,
        query: &[A; K],
        stack: &mut SS::Stack<D::Output>,
    ) -> (D::Output, T)
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
        let mut req_ctx = NearestOneReqCtx {
            query,
            best_dist: D::Output::max_value(),
            best_item: T::default(),
        };

        self.arithmetic_query_with_scratch::<_, _, D>(
            &mut req_ctx,
            stack,
            |leaf_idx, query_wide, query_ctx| {
                self.process_leaf_nearest_one::<D>(
                    leaf_idx,
                    query_wide,
                    &mut query_ctx.best_dist,
                    &mut query_ctx.best_item,
                );
            },
        );

        (req_ctx.best_dist, req_ctx.best_item)
    }

    #[cfg_attr(not(feature = "cargo_asm"), allow(dead_code))]
    #[inline(always)]
    pub(crate) fn nearest_one_with_scratch<D>(
        &self,
        query: &[A; K],
        stack: &mut SS::Stack<D::Output>,
    ) -> (D::Output, T)
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
        if self.stem_leaf_resolution.uses_arithmetic() {
            return self.nearest_one_arithmetic_with_scratch::<D>(query, stack);
        }

        self.nearest_one_mapped_with_scratch::<D>(query, stack)
    }

    #[inline(always)]
    fn nearest_one_mapped_with_scratch<D>(
        &self,
        query: &[A; K],
        stack: &mut SS::Stack<D::Output>,
    ) -> (D::Output, T)
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
        let mut req_ctx = NearestOneReqCtx {
            query,
            best_dist: D::Output::max_value(),
            best_item: T::default(),
        };

        self.backtracking_query_with_scratch::<_, _, D>(
            &mut req_ctx,
            stack,
            |leaf_idx, query_wide, query_ctx| {
                self.process_leaf_nearest_one::<D>(
                    leaf_idx,
                    query_wide,
                    &mut query_ctx.best_dist,
                    &mut query_ctx.best_item,
                );
            },
        );

        (req_ctx.best_dist, req_ctx.best_item)
    }
}

#[allow(missing_docs)]
#[cfg(feature = "cargo_asm")]
pub mod cargo_asm {
    use crate::dist::SquaredEuclidean;
    use crate::kd_tree::query_stack::QueryStack;
    use crate::kd_tree::KdTree;
    use crate::leaf_strategy::{FlatVec, VecOfArenas};
    use crate::stem_strategy::{Donnelly, DonnellySimdFull};
    use crate::Eytzinger;

    const K: usize = 3;
    const BUCKET_SIZE: usize = 32;

    type KdT = KdTree<f64, usize, Eytzinger, FlatVec<f64, usize, K, BUCKET_SIZE>, K, BUCKET_SIZE>;
    type ArenaLeaves = VecOfArenas<f64, usize, K, BUCKET_SIZE>;
    type EytzingerPfFarArenaKdT = KdTree<f64, usize, Eytzinger, ArenaLeaves, K, BUCKET_SIZE>;
    type DonnellyKdT = KdTree<f64, usize, Donnelly<3>, ArenaLeaves, K, BUCKET_SIZE>;
    type DonnellySimdKdT = KdTree<f64, usize, DonnellySimdFull<3>, ArenaLeaves, K, BUCKET_SIZE>;

    /// Hook for cargo-asm to render the v6 nearest-one call path.
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_nearest_one_eytzinger_cargo_asm_hook(
        tree: &KdT,
        query: [f64; 3],
        stack: &mut QueryStack<f64, Eytzinger>,
    ) -> (f64, usize) {
        tree.nearest_one_with_scratch::<SquaredEuclidean<f64>>(&query, stack)
    }

    /// Hook for cargo-asm to render the arithmetic Eytzinger core directly.
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_nearest_one_eytzinger_arithmetic_core_cargo_asm_hook(
        tree: &KdT,
        query: [f64; 3],
        stack: &mut QueryStack<f64, Eytzinger>,
    ) -> (f64, usize) {
        tree.nearest_one_arithmetic_with_scratch::<SquaredEuclidean<f64>>(&query, stack)
    }

    /// Hook for cargo-asm to render the exact nearest-one path for scalar Eytzinger PF-far arena leaves.
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_nearest_one_eytzinger_pf_far_vec_of_arenas_cargo_asm_hook(
        tree: &EytzingerPfFarArenaKdT,
        query: [f64; 3],
        stack: &mut QueryStack<f64, Eytzinger>,
    ) -> (f64, usize) {
        tree.nearest_one_with_scratch::<SquaredEuclidean<f64>>(&query, stack)
    }

    /// Hook for cargo-asm to render the exact nearest-one path for scalar Donnelly.
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_nearest_one_donnelly_vec_of_arenas_cargo_asm_hook(
        tree: &DonnellyKdT,
        query: [f64; 3],
        stack: &mut QueryStack<f64, Donnelly<3>>,
    ) -> (f64, usize) {
        tree.nearest_one_with_scratch::<SquaredEuclidean<f64>>(&query, stack)
    }

    /// Hook for cargo-asm to render the exact nearest-one path for Block3 Donnelly SIMD.
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_nearest_one_donnelly_blocksimd_vec_of_arenas_cargo_asm_hook(
        tree: &DonnellySimdKdT,
        query: [f64; 3],
        stack: &mut <DonnellySimdFull<3> as crate::StemStrategy>::Stack<f64>,
    ) -> (f64, usize) {
        tree.nearest_one_with_scratch::<SquaredEuclidean<f64>>(&query, stack)
    }
}

pub(crate) struct NearestOneReqCtx<'a, A, T, O, const K: usize>
where
    O: Axis<Coord = O>,
{
    query: &'a [A; K],
    best_dist: O,
    best_item: T,
}

impl<A, T, O, const K: usize> QueryContext<A, O, K> for NearestOneReqCtx<'_, A, T, O, K>
where
    O: Axis<Coord = O>,
{
    fn query(&self) -> &[A; K] {
        self.query
    }

    fn max_dist(&self) -> O {
        self.best_dist
    }

    #[inline]
    fn initial_bound_is_unbounded(&self) -> bool {
        true
    }

    // TOOO: investigate into whether this can be removed
    #[inline]
    fn prune_on_equal_max_dist(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use assert_float_eq::assert_float_relative_eq;
    use rand::{RngExt, SeedableRng};
    use test_log::test;

    use crate::dist::{Chebyshev, DistanceMetricCore, Minkowski, SquaredEuclidean};
    use crate::kd_tree::query_stack::QueryStack;
    use crate::kd_tree::KdTree;
    use crate::leaf_strategy::{FlatVec, VecOfArenas, VecOfArrays};
    use crate::stem_strategy::Donnelly;
    use crate::Axis;
    use crate::{Eytzinger, QueryResultItem};

    const REL_EPS_F32: f32 = 1.0e-6;
    const REL_EPS_F64: f64 = 1.0e-12;

    fn assert_nearest_f32(
        actual: QueryResultItem<(), u32, f32>,
        expected: &QueryResultItem<(), usize, f32>,
    ) {
        assert_float_relative_eq!(actual.distance, expected.distance, REL_EPS_F32);
        assert_eq!(actual.item as usize, expected.item);
    }

    #[allow(dead_code)]
    fn assert_nearest_f64(
        actual: QueryResultItem<(), u32, f64>,
        expected: &QueryResultItem<(), usize, f64>,
    ) {
        assert_float_relative_eq!(actual.distance, expected.distance, REL_EPS_F64);
        assert_eq!(actual.item as usize, expected.item);
    }

    #[test]
    fn nearest_one_vec_of_arenas_small_f64() {
        let points = vec![
            [0.0f64, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [0.5, 0.5, 0.6],
        ];

        let tree: KdTree<f64, u32, Eytzinger, VecOfArenas<f64, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points).unwrap();

        let result = tree
            .query(&[0.45, 0.55, 0.65])
            .nearest_one::<SquaredEuclidean<f64>>()
            .execute();

        assert_float_relative_eq!(result.distance, 0.0075, REL_EPS_F64);
        assert_eq!(result.item, 3);
    }

    #[test]
    fn nearest_one_vec_of_arenas_small_f64_no_items() {
        let points = vec![
            [0.0f64, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [0.5, 0.5, 0.6],
        ];

        let tree: KdTree<f64, (), Eytzinger, VecOfArenas<f64, (), 3, 32>, 3, 32> =
            KdTree::new_from_slice_no_items(&points).unwrap();

        let result = tree
            .query(&[0.45, 0.55, 0.65])
            .nearest_one::<SquaredEuclidean<f64>>()
            .execute();

        assert_float_relative_eq!(result.distance, 0.0075, REL_EPS_F64);
        assert_eq!(result.item, ());
    }

    #[test]
    fn nearest_one_handles_degenerate_leaf_larger_than_tls_scratch() {
        let points = vec![[0.25f32, -0.5]; 1_025];
        let tree: KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, 2, 16>, 2, 16> =
            KdTree::new_from_slice(&points).unwrap();

        let exact = tree
            .query(&points[0])
            .nearest_one::<SquaredEuclidean<f32>>()
            .execute();
        let approximate = tree
            .query(&points[0])
            .nearest_one::<SquaredEuclidean<f32>>()
            .approx()
            .execute();
        let within = tree
            .query(&points[0])
            .within::<SquaredEuclidean<f32>>(0.0)
            .execute();

        assert_eq!(exact.distance, 0.0);
        assert_eq!(approximate.distance, 0.0);
        assert_eq!(within.len(), points.len());
    }

    #[test]
    fn nearest_one_vec_of_arenas_matches_flat_vec_f32() {
        let points = vec![
            [0.1f32, 0.2, 0.3],
            [0.9, 0.8, 0.7],
            [0.41, 0.52, 0.63],
            [0.4, 0.5, 0.6],
            [0.7, 0.1, 0.2],
        ];
        let query = [0.39f32, 0.51, 0.61];

        let flat_tree: KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points).unwrap();
        let arena_tree: KdTree<f32, u32, Eytzinger, VecOfArenas<f32, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points).unwrap();

        let flat_result = flat_tree
            .query(&query)
            .nearest_one::<SquaredEuclidean<f32>>()
            .execute();
        let arena_result = arena_tree
            .query(&query)
            .nearest_one::<SquaredEuclidean<f32>>()
            .execute();

        assert_float_relative_eq!(arena_result.distance, flat_result.distance, REL_EPS_F32);
        assert_eq!(arena_result.item, flat_result.item);
    }

    #[test]
    fn nearest_one_arithmetic_with_scratch_matches_expected_result() {
        let points = vec![
            [0.0f64, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.6],
            [2.0, 2.0, 2.0],
        ];

        let tree: KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points).unwrap();
        let mut stack = QueryStack::<f64, Eytzinger>::default();

        let result = tree.nearest_one_arithmetic_with_scratch::<SquaredEuclidean<f64>>(
            &[0.45, 0.55, 0.65],
            &mut stack,
        );

        assert_float_relative_eq!(result.0, 0.0075, REL_EPS_F64);
        assert_eq!(result.1, 2);
        assert!(stack.pop().is_none());
    }

    #[test]
    fn nearest_one_with_scratch_uses_arithmetic_route_when_available() {
        let points = vec![
            [0.0f64, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.6],
            [2.0, 2.0, 2.0],
        ];

        let tree: KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points).unwrap();
        let mut stack = QueryStack::<f64, Eytzinger>::default();

        let result =
            tree.nearest_one_with_scratch::<SquaredEuclidean<f64>>(&[0.45, 0.55, 0.65], &mut stack);

        assert_float_relative_eq!(result.0, 0.0075, REL_EPS_F64);
        assert_eq!(result.1, 2);
        assert!(stack.pop().is_none());
    }

    #[test]
    fn nearest_one_mapped_with_scratch_matches_expected_result() {
        let points = [
            [0.0f32, 0.0],
            [1.0, 1.0],
            [0.4, 0.45],
            [2.0, 2.0],
            [0.6, 0.7],
        ];

        let mut tree: KdTree<f32, u32, Eytzinger, VecOfArrays<f32, u32, 2, 2>, 2, 2> =
            KdTree::default();
        for (idx, point) in points.iter().enumerate() {
            tree.add(point, idx as u32).unwrap();
        }

        let mut stack = QueryStack::<f32, Eytzinger>::default();
        let result =
            tree.nearest_one_mapped_with_scratch::<SquaredEuclidean<f32>>(&[0.5, 0.5], &mut stack);

        assert_float_relative_eq!(result.0, 0.0125, REL_EPS_F32);
        assert_eq!(result.1, 2);
        assert!(stack.pop().is_none());
    }

    #[test]
    fn nearest_one_with_scratch_uses_mapped_route_when_arithmetic_is_unavailable() {
        let points = [
            [0.0f32, 0.0],
            [1.0, 1.0],
            [0.4, 0.45],
            [2.0, 2.0],
            [0.6, 0.7],
        ];

        let mut tree: KdTree<f32, u32, Eytzinger, VecOfArrays<f32, u32, 2, 2>, 2, 2> =
            KdTree::default();
        for (idx, point) in points.iter().enumerate() {
            tree.add(point, idx as u32).unwrap();
        }

        let mut stack = QueryStack::<f32, Eytzinger>::default();
        let result =
            tree.nearest_one_with_scratch::<SquaredEuclidean<f32>>(&[0.5, 0.5], &mut stack);

        assert_float_relative_eq!(result.0, 0.0125, REL_EPS_F32);
        assert_eq!(result.1, 2);
        assert!(stack.pop().is_none());
    }

    #[test]
    fn v6_query_nearest_one_small_f64_flat_vec_eytzinger() {
        let content_to_add: [[f64; 4]; 16] = [
            [0.9f64, 0.0f64, 0.9f64, 0.0f64],
            [0.4f64, 0.5f64, 0.4f64, 0.51f64],
            [0.12f64, 0.3f64, 0.12f64, 0.3f64],
            [0.7f64, 0.2f64, 0.7f64, 0.22f64],
            [0.13f64, 0.4f64, 0.13f64, 0.4f64],
            [0.6f64, 0.3f64, 0.6f64, 0.33f64],
            [0.2f64, 0.7f64, 0.2f64, 0.7f64],
            [0.14f64, 0.5f64, 0.14f64, 0.5f64],
            [0.3f64, 0.6f64, 0.3f64, 0.6f64],
            [0.10f64, 0.1f64, 0.10f64, 0.1f64],
            [0.16f64, 0.7f64, 0.16f64, 0.7f64],
            [0.1f64, 0.8f64, 0.1f64, 0.8f64],
            [0.15f64, 0.6f64, 0.15f64, 0.6f64],
            [0.5f64, 0.4f64, 0.5f64, 0.44f64],
            [0.8f64, 0.1f64, 0.8f64, 0.15f64],
            [0.11f64, 0.2f64, 0.11f64, 0.2f64],
        ];

        let tree: KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&content_to_add).unwrap();

        assert!(!tree.is_empty());
        assert_eq!(tree.size(), 16);

        let query_point = [0.78f64, 0.55f64, 0.78f64, 0.55f64];

        let expected = (0.17570000000000008, 5);

        let results = tree
            .query(&query_point)
            .nearest_one::<SquaredEuclidean<f64>>()
            .execute();
        assert_float_relative_eq!(results.distance, expected.0, REL_EPS_F64);
        assert_eq!(results.item, expected.1);
    }

    #[test]
    fn v6_query_nearest_one_large_f32_flatvec_eytzinger() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 1000;

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

        let tree: KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&content_to_add).unwrap();

        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[f32; 4]> = (0..NUM_QUERIES)
            .map(|_| rng.random::<[f32; 4]>()) // Use the seeded rng
            .collect();

        for query_point in query_points.iter() {
            let expected = linear_search(&content_to_add, query_point);
            let result = tree
                .query(query_point)
                .nearest_one::<SquaredEuclidean<f32>>()
                .execute();

            assert_nearest_f32(result, &expected);
        }
    }

    #[test]
    fn v6_query_nearest_one_large_f32_flatvec_donnelly() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 1000;

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

        let tree: KdTree<f32, u32, Donnelly<4>, FlatVec<f32, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&content_to_add).unwrap();

        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[f32; 4]> = (0..NUM_QUERIES)
            .map(|_| rng.random::<[f32; 4]>()) // Use the seeded rng
            .collect();

        for query_point in query_points.iter() {
            let expected = linear_search(&content_to_add, query_point);
            let result = tree
                .query(query_point)
                .nearest_one::<SquaredEuclidean<f32>>()
                .execute();

            assert_nearest_f32(result, &expected);
        }
    }

    #[test]
    fn v6_query_nearest_one_large_f32_vec_of_arrays_eytzinger() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 1000;

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

        let tree: KdTree<f32, u32, Eytzinger, VecOfArrays<f32, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&content_to_add).unwrap();

        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[f32; 4]> = (0..NUM_QUERIES)
            .map(|_| rng.random::<[f32; 4]>()) // Use the seeded rng
            .collect();

        // println!("Tree: {}", &tree);

        for query_point in query_points.iter() {
            let expected = linear_search(&content_to_add, query_point);
            let result = tree
                .query(query_point)
                .nearest_one::<SquaredEuclidean<f32>>()
                .execute();

            assert_nearest_f32(result, &expected);
        }
    }

    #[test]
    fn v6_query_nearest_one_large_f32_vec_of_arrays_mutated_eytzinger() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 1000;

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

        for (i, query_point) in query_points.iter().enumerate() {
            let expected = linear_search(&content_to_add, query_point);
            let result = tree
                .query(query_point)
                .nearest_one::<SquaredEuclidean<f32>>()
                .execute();

            assert_float_relative_eq!(result.distance, expected.distance, REL_EPS_F32);
            assert_eq!(
                result.item as usize, expected.item,
                "Incorrect item, query index: {i}"
            );
        }
    }

    #[test]
    fn v6_query_nearest_one_large_f32_vec_of_arrays_mutated_donnelly() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 1000;

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

        let mut tree: KdTree<f32, u32, Donnelly<4>, VecOfArrays<f32, u32, 4, 32>, 4, 32> =
            KdTree::default();

        for (idx, point) in content_to_add.iter().enumerate() {
            tree.add(point, idx as u32).unwrap();
        }

        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[f32; 4]> = (0..NUM_QUERIES)
            .map(|_| rng.random::<[f32; 4]>()) // Use the seeded rng
            .collect();

        for query_point in query_points.iter() {
            let expected = linear_search(&content_to_add, query_point);
            let result = tree
                .query(query_point)
                .nearest_one::<SquaredEuclidean<f32>>()
                .execute();

            assert_nearest_f32(result, &expected);
        }
    }

    #[test]
    fn v6_query_nearest_one_large_f32_vec_of_arrays_donnelly() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 1000;

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

        let tree: KdTree<f32, u32, Donnelly<4>, VecOfArrays<f32, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&content_to_add).unwrap();

        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[f32; 4]> = (0..NUM_QUERIES)
            .map(|_| rng.random::<[f32; 4]>()) // Use the seeded rng
            .collect();

        for query_point in query_points.iter() {
            let expected = linear_search(&content_to_add, query_point);
            let result = tree
                .query(query_point)
                .nearest_one::<SquaredEuclidean<f32>>()
                .execute();

            assert_nearest_f32(result, &expected);
        }
    }

    #[test]
    fn v6_query_nearest_one_large_f64_flatvec_eytzinger_chebyshev() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(31);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 1000;

        let content_to_add: Vec<[f64; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f64; 4]>()).collect();

        let tree: KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&content_to_add).unwrap();

        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[f64; 4]> =
            (0..NUM_QUERIES).map(|_| rng.random::<[f64; 4]>()).collect();

        for query_point in query_points.iter() {
            let expected =
                linear_search_with_metric::<f64, Chebyshev<f64>, 4>(&content_to_add, query_point);
            let result = tree
                .query(query_point)
                .nearest_one::<Chebyshev<f64>>()
                .execute();

            assert_nearest_f64(result, &expected);
        }
    }

    #[test]
    fn v6_query_nearest_one_large_f64_flatvec_eytzinger_minkowski_3() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(37);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 1000;

        let content_to_add: Vec<[f64; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f64; 4]>()).collect();

        let tree: KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&content_to_add).unwrap();

        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[f64; 4]> =
            (0..NUM_QUERIES).map(|_| rng.random::<[f64; 4]>()).collect();

        for query_point in query_points.iter() {
            let expected = linear_search_with_metric::<f64, Minkowski<3, f64>, 4>(
                &content_to_add,
                query_point,
            );
            let result = tree
                .query(query_point)
                .nearest_one::<Minkowski<3, f64>>()
                .execute();

            assert_nearest_f64(result, &expected);
        }
    }

    fn linear_search<A, const K: usize>(
        content: &[[A; K]],
        query_point: &[A; K],
    ) -> QueryResultItem<(), usize, A>
    where
        A: Axis<Coord = A>,
        SquaredEuclidean<A>: crate::dist::DistanceMetricCore<A, Output = A>,
    {
        let mut best_dist: A = A::max_value();
        let mut best_item: usize = usize::MAX;

        for (idx, p) in content.iter().enumerate() {
            let dist = squared_euclidean_dist(query_point, p);
            if dist < best_dist {
                best_item = idx;
                best_dist = dist;
            }
        }

        QueryResultItem {
            point: (),
            distance: best_dist,
            item: best_item,
        }
    }

    fn squared_euclidean_dist<A, const K: usize>(a: &[A; K], b: &[A; K]) -> A
    where
        A: Axis<Coord = A>,
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

    fn linear_search_with_metric<A, D, const K: usize>(
        content: &[[A; K]],
        query_point: &[A; K],
    ) -> QueryResultItem<(), usize, A>
    where
        A: Axis<Coord = A>,
        D: DistanceMetricCore<A, Output = A>,
    {
        let mut best_dist: A = A::max_value();
        let mut best_item: usize = usize::MAX;

        for (idx, p) in content.iter().enumerate() {
            let dist = metric_dist::<A, D, K>(query_point, p);
            if dist < best_dist {
                best_item = idx;
                best_dist = dist;
            }
        }

        QueryResultItem {
            point: (),
            distance: best_dist,
            item: best_item,
        }
    }

    fn metric_dist<A, D, const K: usize>(a: &[A; K], b: &[A; K]) -> A
    where
        A: Axis<Coord = A>,
        D: DistanceMetricCore<A, Output = A>,
    {
        let aw = (*a).map(D::widen_coord);
        let bw = (*b).map(D::widen_coord);

        D::dist::<K>(&aw, &bw)
    }

    #[test]
    #[cfg(feature = "simd")]
    #[cfg(target_arch = "x86_64")]
    fn v6_query_nearest_one_donnelly_marker_simd_f64() {
        use crate::stem_strategy::{DonnellySimdFull, DonnellyUnrolled};

        // Test DonnellySimdFull with f64 data using exact nearest_one query
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        // Use 8192 points which with bucket size 32 gives 256 leaves
        // 256 leaves = 2^8, so tree depth = 8
        // With Block3, depth 8 is not divisible by 3, so tree will be padded to depth 9
        let points: Vec<[f64; 3]> = (0..2_048) // 8_192)
            .map(|_| {
                [
                    rng.random::<f64>(),
                    rng.random::<f64>(),
                    rng.random::<f64>(),
                ]
            })
            .collect();

        let tree: KdTree<f64, u32, DonnellySimdFull<3>, FlatVec<f64, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points).unwrap();

        let tree_non_simd: KdTree<f64, u32, DonnellyUnrolled<3>, FlatVec<f64, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points).unwrap();

        assert!(!tree.is_empty());
        assert_eq!(tree.size(), 2_048);
        assert_eq!(tree.leaf_count(), 64);

        // Verify max_stem_level is padded to multiple of block size (3)
        // 256 leaves = depth 8, padded to 9
        assert_eq!((tree.max_stem_level() + 1) % 3, 0);
        assert_eq!(tree.max_stem_level(), 5);

        // println!("NON-SIMD: {}", tree_non_simd);
        // println!("SIMD: {}", tree);

        // Test multiple query points to ensure backtracking queries work correctly
        let query_points: Vec<[f64; 3]> = (0..50)
            .map(|_| {
                [
                    rng.random::<f64>(),
                    rng.random::<f64>(),
                    rng.random::<f64>(),
                ]
            })
            .collect();

        for query_point in query_points.iter() {
            // tracing::debug!("Query point: #{i} ({query_point:?})");

            let expected = linear_search(&points, query_point);
            // println!("\n========== QUERY #{i} ==========");
            // println!("Query point: {:?}", query_point);
            // println!("Expected: item={}, dist²={}", expected.item, expected.distance);

            let _result = tree_non_simd
                .query(query_point)
                .nearest_one::<SquaredEuclidean<f64>>()
                .execute();
            // println!("NON-SIMD: item={}, dist²={}", result.1, result.0);

            let result = tree
                .query(query_point)
                .nearest_one::<SquaredEuclidean<f64>>()
                .execute();
            // println!("SIMD: item={}, dist²={}", result.1, result.0);

            assert_nearest_f64(result, &expected);
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    #[cfg(target_arch = "x86_64")]
    fn v6_query_nearest_one_donnelly_marker_simd_block4_f32() {
        use crate::stem_strategy::DonnellySimdFull;

        // Test DonnellySimdFull with f32 data using exact nearest_one query
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        // Use smaller dataset for faster test (16384 points = 512 leaves = 2^9, depth = 9)
        // Block4 doesn't divide evenly into 9, will be padded to 12
        let points: Vec<[f32; 4]> = (0..16_384)
            .map(|_| {
                [
                    rng.random::<f32>(),
                    rng.random::<f32>(),
                    rng.random::<f32>(),
                    rng.random::<f32>(),
                ]
            })
            .collect();

        let tree: KdTree<f32, u32, DonnellySimdFull<4>, FlatVec<f32, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&points).unwrap();

        assert!(!tree.is_empty());
        assert_eq!(tree.size(), 16_384);
        assert_eq!(tree.leaf_count(), 512);

        // Verify max_stem_level is padded to multiple of block size (4)
        assert_eq!((tree.max_stem_level() + 1) % 4, 0);

        // Test multiple query points to ensure backtracking queries work correctly
        let query_points: Vec<[f32; 4]> = (0..50)
            .map(|_| {
                [
                    rng.random::<f32>(),
                    rng.random::<f32>(),
                    rng.random::<f32>(),
                    rng.random::<f32>(),
                ]
            })
            .collect();

        for query_point in query_points.iter() {
            let expected = linear_search(&points, query_point);
            let result = tree
                .query(query_point)
                .nearest_one::<SquaredEuclidean<f32>>()
                .execute();

            assert_float_relative_eq!(result.distance, expected.distance, REL_EPS_F32);
            assert_eq!(
                result.item as usize, expected.item,
                "Item mismatch for query {:?}",
                query_point
            );
        }
    }
}

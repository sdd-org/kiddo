use std::collections::BinaryHeap;
use std::num::NonZero;

use crate::dist::DistanceMetric;
use crate::kd_tree::query_context::QueryContext;
use crate::kd_tree::query_stack::StackTrait;
use crate::kd_tree::{KdTreeQueryOps, ITEM_LEAF_MODE_SORTED, ITEM_LEAF_MODE_UNSORTED};
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

impl<A, T, SS, LS, const K: usize, const B: usize, const ITEM_LEAF_MODE: u8>
    KdTree<A, T, SS, LS, K, B, ITEM_LEAF_MODE>
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    LS: LeafStrategy<A, T, SS, K, B>,
    SS: StemStrategy,
{
    #[inline(always)]
    fn process_leaf_best_n_within<D, R, const EXCLUSIVE: bool, const LEAF_MODE: u8>(
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
        #[cfg(any(feature = "exact_query_stats", feature = "test_utils"))]
        {
            crate::results::exact_query_stats::record_leaf_visit();
            crate::results::exact_query_stats::record_leaf_items_available(
                self.leaves.leaf_len(leaf_idx),
            );
        }

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
                best_n_within_with_query_wide_arena::<A, T, D, R, EXCLUSIVE, LEAF_MODE, K>(
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
                best_n_within_with_query_wide::<A, T, D, R, EXCLUSIVE, LEAF_MODE, K, B>(
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
        self.best_n_within_inner_ordered::<D, R, EXCLUSIVE, ITEM_LEAF_MODE>(
            query, max_dist, max_qty,
        )
    }

    #[inline(always)]
    fn best_n_within_inner_ordered<D, R, const EXCLUSIVE: bool, const LEAF_MODE: u8>(
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
        let mut req_ctx = BestNWithinReqCtx::<A, D::Output, R, EXCLUSIVE, K, LEAF_MODE> {
            query,
            max_dist,
            results: R::with_max_qty(max_qty),
            item_summary_filter: crate::kd_tree::item_summary::ItemSummaryFilter::from_threshold::<
                T,
                LEAF_MODE,
            >(None),
        };

        self.backtracking_query::<_, _, D>(&mut req_ctx, |leaf_idx, query_wide, req_ctx| {
            self.process_leaf_best_n_within::<D, _, EXCLUSIVE, LEAF_MODE>(
                leaf_idx,
                query_wide,
                max_dist,
                &mut req_ctx.results,
            );
            req_ctx.item_summary_filter =
                crate::kd_tree::item_summary::ItemSummaryFilter::from_threshold::<T, LEAF_MODE>(
                    req_ctx.results.threshold_item(),
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
        self.best_n_within_inner_with_scratch_ordered::<D, R, EXCLUSIVE, ITEM_LEAF_MODE>(
            query, max_dist, max_qty, stack,
        )
    }

    #[inline(always)]
    fn best_n_within_inner_with_scratch_ordered<D, R, const EXCLUSIVE: bool, const LEAF_MODE: u8>(
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
        let mut req_ctx = BestNWithinReqCtx::<A, D::Output, R, EXCLUSIVE, K, LEAF_MODE> {
            query,
            max_dist,
            results: R::with_max_qty(max_qty),
            item_summary_filter: crate::kd_tree::item_summary::ItemSummaryFilter::from_threshold::<
                T,
                LEAF_MODE,
            >(None),
        };

        self.backtracking_query_with_scratch::<_, _, D>(
            &mut req_ctx,
            stack,
            |leaf_idx, query_wide, req_ctx| {
                self.process_leaf_best_n_within::<D, _, EXCLUSIVE, LEAF_MODE>(
                    leaf_idx,
                    query_wide,
                    max_dist,
                    &mut req_ctx.results,
                );
                req_ctx.item_summary_filter =
                    crate::kd_tree::item_summary::ItemSummaryFilter::from_threshold::<T, LEAF_MODE>(
                        req_ctx.results.threshold_item(),
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
struct BestNWithinReqCtx<
    'a,
    A,
    O,
    R,
    const EXCLUSIVE: bool,
    const K: usize,
    const ITEM_LEAF_MODE: u8,
> where
    O: Axis<Coord = O>,
{
    query: &'a [A; K],
    max_dist: O,
    results: R,
    item_summary_filter: crate::kd_tree::item_summary::ItemSummaryFilter,
}

impl<'a, A, O, R, const EXCLUSIVE: bool, const K: usize, const ITEM_LEAF_MODE: u8>
    QueryContext<A, O, K> for BestNWithinReqCtx<'a, A, O, R, EXCLUSIVE, K, ITEM_LEAF_MODE>
where
    O: Axis<Coord = O>,
{
    const USES_EMBEDDED_ITEM_SUMMARY: bool =
        ITEM_LEAF_MODE != ITEM_LEAF_MODE_UNSORTED && ITEM_LEAF_MODE != ITEM_LEAF_MODE_SORTED;

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

    #[inline(always)]
    fn embedded_item_summary_filter(&self) -> crate::kd_tree::item_summary::ItemSummaryFilter {
        self.item_summary_filter
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
    use crate::{
        BestQueryResultItem, Donnelly, DonnellyCyclicSimdDescent, DonnellyCyclicSimdFull,
        DonnellyNoPf, DonnellySimdDescent, DonnellySimdFull, DonnellyUnrolled,
        DonnellyUnrolledBlockDim, Eytzinger,
    };

    const RNG_SEED: u64 = 42;

    #[test]
    fn embedded_summary_builder_and_queries_cover_all_public_donnelly3_variants() {
        let entries = (0..257u32)
            .map(|idx| {
                (
                    (idx * 73) % 257,
                    [
                        ((idx * 19) % 263) as f64,
                        ((idx * 43) % 269) as f64,
                        ((idx * 61) % 271) as f64,
                    ],
                )
            })
            .collect::<Vec<_>>();
        let query = [130.0, 130.0, 130.0];
        let max_qty = NonZeroUsize::new(7).unwrap();

        macro_rules! assert_strategy {
            ($strategy:ty) => {{
                type Tree<SS> = KdTree<f64, u32, SS, FlatVec<f64, u32, 3, 8>, 3, 8>;
                let tree = Tree::<$strategy>::builder()
                    .with_embedded_min_item_shifted_summary::<8>()
                    .with_serial_construction()
                    .build_from_entries(&entries)
                    .unwrap();
                let sorted = Tree::<$strategy>::builder()
                    .with_item_sorted_leaves()
                    .with_serial_construction()
                    .build_from_entries(&entries)
                    .unwrap();
                let results = tree
                    .query(&query)
                    .best_n_within::<SquaredEuclidean<f64>>(f64::INFINITY, max_qty)
                    .execute()
                    .into_sorted_vec();
                assert_eq!(
                    results.iter().map(|result| result.item).collect::<Vec<_>>(),
                    (0..7).collect::<Vec<_>>(),
                    "strategy {}",
                    std::any::type_name::<$strategy>()
                );

                for (query, radius, qty) in [
                    ([0.0, 0.0, 0.0], 2_000.0, 1),
                    ([130.0, 130.0, 130.0], 8_000.0, 7),
                    ([262.0, 268.0, 270.0], 50_000.0, 31),
                ] {
                    let qty = NonZeroUsize::new(qty).unwrap();
                    let expected = sorted
                        .query(&query)
                        .best_n_within::<SquaredEuclidean<f64>>(radius, qty)
                        .execute()
                        .into_sorted_vec();
                    let actual = tree
                        .query(&query)
                        .best_n_within::<SquaredEuclidean<f64>>(radius, qty)
                        .execute()
                        .into_sorted_vec();
                    assert_eq!(
                        actual,
                        expected,
                        "strategy {} query={query:?} radius={radius}",
                        std::any::type_name::<$strategy>(),
                    );
                }
            }};
        }

        assert_strategy!(Donnelly<3>);
        assert_strategy!(DonnellyNoPf<3>);
        assert_strategy!(DonnellyUnrolled<3>);
        assert_strategy!(DonnellyUnrolledBlockDim<3>);
        assert_strategy!(DonnellySimdDescent<3>);
        assert_strategy!(DonnellySimdFull<3>);
        assert_strategy!(DonnellyCyclicSimdDescent<3>);
        assert_strategy!(DonnellyCyclicSimdFull<3>);
    }

    #[cfg(feature = "test_utils")]
    #[test]
    fn embedded_summaries_prune_item_dead_subtrees_before_leaf_visits() {
        type Tree = KdTree<f64, u32, Donnelly<3>, FlatVec<f64, u32, 3, 1>, 3, 1>;
        let entries = (0..64u32)
            .map(|item| (item, [item as f64, item as f64, item as f64]))
            .collect::<Vec<_>>();
        let query = [0.0, 0.0, 0.0];
        let max_qty = NonZeroUsize::new(1).unwrap();

        let sorted = Tree::builder()
            .with_item_sorted_leaves()
            .with_serial_construction()
            .build_from_entries(&entries)
            .unwrap();
        crate::results::exact_query_stats::reset();
        let expected = sorted
            .query(&query)
            .best_n_within::<SquaredEuclidean<f64>>(f64::INFINITY, max_qty)
            .execute()
            .into_sorted_vec();
        let sorted_stats = crate::results::exact_query_stats::snapshot();

        let summarized = Tree::builder()
            .with_embedded_min_item_shifted_summary::<1>()
            .with_serial_construction()
            .build_from_entries(&entries)
            .unwrap();
        crate::results::exact_query_stats::reset();
        let actual = summarized
            .query(&query)
            .best_n_within::<SquaredEuclidean<f64>>(f64::INFINITY, max_qty)
            .execute()
            .into_sorted_vec();
        let summary_stats = crate::results::exact_query_stats::snapshot();

        assert_eq!(actual, expected);
        assert!(summary_stats.item_summary_subtrees_pruned > 0);
        assert!(summary_stats.item_summary_lanes_rejected > 0);
        assert!(summary_stats.leaf_visits < sorted_stats.leaf_visits);
    }

    #[cfg(feature = "test_utils")]
    #[test]
    fn embedded_summaries_are_not_checked_until_results_are_full() {
        type Tree = KdTree<f64, u32, Donnelly<3>, FlatVec<f64, u32, 3, 1>, 3, 1>;
        let entries = (0..64u32)
            .map(|item| (item, [item as f64, item as f64, item as f64]))
            .collect::<Vec<_>>();
        let tree = Tree::builder()
            .with_embedded_min_item_shifted_summary::<1>()
            .with_serial_construction()
            .build_from_entries(&entries)
            .unwrap();

        crate::results::exact_query_stats::reset();
        let results = tree
            .query(&[0.0, 0.0, 0.0])
            .best_n_within::<SquaredEuclidean<f64>>(0.0, NonZeroUsize::new(7).unwrap())
            .execute();
        let stats = crate::results::exact_query_stats::snapshot();

        assert_eq!(results.len(), 1);
        assert_eq!(stats.item_summary_blocks_checked, 0);
        assert_eq!(stats.item_summary_lanes_rejected, 0);
        assert_eq!(stats.item_summary_subtrees_pruned, 0);
    }

    #[test]
    fn best_n_within_item_sorted_leaves_matches_ordinary_kernels() {
        type FlatTree = KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, 2, 8>, 2, 8>;
        type ArenaTree = KdTree<f64, u32, Eytzinger, VecOfArenas<f64, u32, 2, 8>, 2, 8>;

        let entries = (0..97u32)
            .map(|idx| {
                (
                    (idx * 37) % 97,
                    [((idx * 19) % 101) as f64, ((idx * 43) % 103) as f64],
                )
            })
            .collect::<Vec<_>>();
        let query = [50.0, 50.0];
        let radius = 20_000.0;
        let max_qty = NonZeroUsize::new(7).unwrap();

        macro_rules! assert_sorted_matches {
            ($tree:ty) => {{
                let ordinary = <$tree>::builder()
                    .with_serial_construction()
                    .build_from_entries(&entries)
                    .unwrap();
                let item_sorted = <$tree>::builder()
                    .with_serial_construction()
                    .with_item_sorted_leaves()
                    .build_from_entries(&entries)
                    .unwrap();

                let ordinary_results = ordinary
                    .query(&query)
                    .best_n_within::<SquaredEuclidean<f64>>(radius, max_qty)
                    .execute()
                    .into_sorted_vec();
                let sorted_results = item_sorted
                    .query(&query)
                    .best_n_within::<SquaredEuclidean<f64>>(radius, max_qty)
                    .execute()
                    .into_sorted_vec();

                assert_eq!(sorted_results, ordinary_results);
                assert_eq!(
                    sorted_results
                        .iter()
                        .map(|result| result.item)
                        .collect::<Vec<_>>(),
                    vec![0, 1, 2, 3, 4, 5, 6]
                );
            }};
        }

        assert_sorted_matches!(FlatTree);
        assert_sorted_matches!(ArenaTree);
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    #[test]
    fn best_n_within_item_sorted_avx512_matches_ordinary_for_all_tail_widths() {
        macro_rules! assert_avx512_sorted_matches {
            ($axis:ty, $leaf:ident) => {{
                type Tree = KdTree<$axis, u32, Eytzinger, $leaf<$axis, u32, 3, 32>, 3, 32>;

                let assert_results = |actual: &[BestQueryResultItem<(), u32, $axis>],
                                      expected: &[BestQueryResultItem<(), u32, $axis>],
                                      context: &str| {
                    assert_eq!(actual.len(), expected.len(), "{context}");
                    for (actual, expected) in actual.iter().zip(expected) {
                        assert_eq!(actual.item, expected.item, "{context}");
                        let tolerance = (16.0 as $axis)
                            * <$axis>::EPSILON
                            * expected.distance.abs().max(1.0 as $axis);
                        assert!(
                            (actual.distance - expected.distance).abs() <= tolerance,
                            "{context}: actual={actual:?} expected={expected:?}"
                        );
                    }
                };

                for len in [
                    1usize, 2, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 40, 47, 64, 65,
                ] {
                    let entries = (0..len)
                        .map(|idx| {
                            (
                                (len - idx - 1) as u32,
                                [
                                    ((idx * 19) % 101) as $axis / 101.0,
                                    ((idx * 43) % 103) as $axis / 103.0,
                                    ((idx * 61) % 107) as $axis / 107.0,
                                ],
                            )
                        })
                        .collect::<Vec<_>>();
                    let ordinary = Tree::builder()
                        .with_serial_construction()
                        .build_from_entries(&entries)
                        .unwrap();
                    let item_sorted = Tree::builder()
                        .with_serial_construction()
                        .with_item_sorted_leaves()
                        .build_from_entries(&entries)
                        .unwrap();
                    let query = [0.42 as $axis, 0.53 as $axis, 0.61 as $axis];
                    let radius = 0.25 as $axis;
                    let max_qty = NonZeroUsize::new(5).unwrap();

                    let expected = ordinary
                        .query(&query)
                        .best_n_within::<SquaredEuclidean<$axis>>(radius, max_qty)
                        .execute()
                        .into_sorted_vec();
                    let actual = item_sorted
                        .query(&query)
                        .best_n_within::<SquaredEuclidean<$axis>>(radius, max_qty)
                        .execute()
                        .into_sorted_vec();
                    assert_results(&actual, &expected, &format!("inclusive len={len}"));

                    let expected = ordinary
                        .query(&query)
                        .best_n_within::<SquaredEuclidean<$axis>>(radius, max_qty)
                        .exclusive_boundaries()
                        .execute()
                        .into_sorted_vec();
                    let actual = item_sorted
                        .query(&query)
                        .best_n_within::<SquaredEuclidean<$axis>>(radius, max_qty)
                        .exclusive_boundaries()
                        .execute()
                        .into_sorted_vec();
                    assert_results(&actual, &expected, &format!("exclusive len={len}"));
                }
            }};
        }

        assert_avx512_sorted_matches!(f64, FlatVec);
        assert_avx512_sorted_matches!(f64, VecOfArenas);
        assert_avx512_sorted_matches!(f32, FlatVec);
        assert_avx512_sorted_matches!(f32, VecOfArenas);
    }

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

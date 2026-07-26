use std::collections::BinaryHeap;
use std::num::NonZeroUsize;

use crate::dist::DistanceMetric;
use crate::kd_tree::query_context::QueryContext;
use crate::kd_tree::query_stack::StackTrait;
use crate::kd_tree::KdTreeQueryOps;
use crate::kd_tree::{ArchivedKdTree, KdTreeAccessor};
use crate::leaf_view::TlsLeafScratch;
use crate::leaf_view_chunked::best_n_within::best_n_within_with_query_wide_arena;
use crate::leaf_view_chunked::nearest_n_within::nearest_n_within_with_query_wide_arena;
use crate::leaf_view_chunked::nearest_one::{
    nearest_one_with_query_wide, nearest_one_with_query_wide_arena,
};
use crate::results::result_collection::{
    BestNeighbourResultCollection, BinaryHeapResultCollection, FromLeafCandidate, ResultCollection,
    VisitorResultCollection,
};
use crate::stem_strategy::donnelly::simd_full::{
    BacktrackBlock3, BacktrackBlock4, SimdSelectBestChildBlock3,
};
use crate::traits::leaf_strategy::LeafProjection;
use crate::{Axis, BestQueryResultItem, Content, LeafStrategy, QueryResultItem, StemStrategy};

impl<A, T, SS, LS, const K: usize, const B: usize> ArchivedKdTree<A, T, SS, LS, K, B>
where
    A: rkyv_08::Archive + Axis<Coord = A> + 'static,
    T: Content + PartialOrd + PartialEq,
    SS: StemStrategy,
    LS: rkyv_08::Archive,
    rkyv_08::Archived<LS>: LeafStrategy<A, T, SS, K, B>,
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
        match <rkyv_08::Archived<LS> as LeafStrategy<A, T, SS, K, B>>::LEAF_PROJECTION {
            LeafProjection::LeafArena => {
                let arena = self.leaves().leaf_arena(leaf_idx);
                nearest_one_with_query_wide_arena::<A, T, D, K>(
                    &arena, query_wide, best_dist, best_item,
                );
            }
            LeafProjection::LeafView => {
                let leaf = self.leaves().leaf_view(leaf_idx);
                nearest_one_with_query_wide::<A, T, D, K, B>(
                    &leaf, query_wide, best_dist, best_item,
                );
            }
        }
    }

    /// Finds an approximate nearest point to the query point.
    #[inline(always)]
    pub(crate) fn approx_nearest_one<D>(&self, query: &[A; K]) -> (D::Output, T)
    where
        D: DistanceMetric<A, Output = A>,
    {
        let req_ctx = ArchivedApproxNearestOneReqCtx::<A, D::Output, K> {
            query,
            _phantom: std::marker::PhantomData,
        };

        let mut best_dist = A::max_value();
        let mut best_item = T::default();

        self.straight_query(req_ctx, |leaf_idx| {
            self.process_leaf_nearest_one::<D>(leaf_idx, query, &mut best_dist, &mut best_item);
        });

        (best_dist, best_item)
    }

    /// Finds the nearest point to the query point.
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
        let mut req_ctx = ArchivedNearestOneReqCtx {
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

    /// Finds the nearest point to the query point using caller-provided stack scratch.
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
        let mut req_ctx = ArchivedNearestOneReqCtx {
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

impl<A, T, SS, LS, const K: usize, const B: usize> ArchivedKdTree<A, T, SS, LS, K, B>
where
    A: rkyv_08::Archive + Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: rkyv_08::Archive,
    rkyv_08::Archived<LS>: LeafStrategy<A, T, SS, K, B>,
{
    #[inline(always)]
    fn process_leaf_nearest_n_within<D, E, R, const EXCLUSIVE: bool>(
        &self,
        leaf_idx: usize,
        query_wide: &[D::Output; K],
        max_dist: D::Output,
        results: &mut R,
    ) where
        D: DistanceMetric<A>,
        D::Output: Axis<Coord = D::Output> + TlsLeafScratch + 'static,
        E: FromLeafCandidate<A, T, D::Output, K>,
        R: ResultCollection<D::Output, E>,
    {
        match <rkyv_08::Archived<LS> as LeafStrategy<A, T, SS, K, B>>::LEAF_PROJECTION {
            LeafProjection::LeafArena => {
                let arena = self.leaves().leaf_arena(leaf_idx);
                nearest_n_within_with_query_wide_arena::<A, T, D, E, R, EXCLUSIVE, K>(
                    &arena, query_wide, max_dist, results,
                );
            }
            LeafProjection::LeafView => {
                let leaf = self.leaves().leaf_view(leaf_idx);
                crate::leaf_view_chunked::nearest_n_within::nearest_n_within_with_query_wide::<
                    A,
                    T,
                    D,
                    E,
                    R,
                    EXCLUSIVE,
                    K,
                    B,
                >(&leaf, query_wide, max_dist, results);
            }
        }
    }

    /// Finds up to N nearest points within a given distance.
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
        let max_qty = max_qty.get();
        if max_qty == usize::MAX {
            self.nearest_n_within_inner::<D, Vec<QueryResultItem<(), T, D::Output>>, EXCLUSIVE>(
                query, max_dist, max_qty, sorted, None,
            )
        } else {
            self.nearest_n_within_inner::<
                D,
                BinaryHeapResultCollection<QueryResultItem<(), T, D::Output>>,
                EXCLUSIVE,
            >(query, max_dist, max_qty, sorted, None)
        }
    }

    fn nearest_n_within_inner<D, R, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
        max_qty: usize,
        sorted: bool,
        result_capacity: Option<NonZeroUsize>,
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
        let mut req_ctx = ArchivedNearestNWithinReqCtx::<A, T, D::Output, R, EXCLUSIVE, K> {
            query,
            max_dist,
            results: R::with_max_qty_and_capacity(max_qty, result_capacity),
            _phantom: std::marker::PhantomData,
        };

        self.backtracking_query::<_, _, D>(&mut req_ctx, |leaf_idx, query_wide, req_ctx| {
            let leaf_max_dist = req_ctx.max_dist();
            self.process_leaf_nearest_n_within::<
                D,
                QueryResultItem<(), T, D::Output>,
                R,
                EXCLUSIVE,
            >(
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

    /// Finds the N nearest points to the query point.
    pub(crate) fn nearest_n<D>(
        &self,
        query: &[A; K],
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
        self.nearest_n_within::<D>(query, D::Output::max_value(), max_qty, sorted)
    }

    pub(crate) fn nearest_n_with_scratch<D>(
        &self,
        query: &[A; K],
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
        self.nearest_n_within_with_scratch::<D>(
            query,
            D::Output::max_value(),
            max_qty,
            sorted,
            stack,
        )
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

    pub(crate) fn within_impl<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
        result_capacity: Option<NonZeroUsize>,
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
        self.nearest_n_within_inner::<D, Vec<QueryResultItem<(), T, D::Output>>, EXCLUSIVE>(
            query,
            max_dist,
            usize::MAX,
            true,
            result_capacity,
        )
    }

    pub(crate) fn within_impl_with_scratch<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
        result_capacity: Option<NonZeroUsize>,
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
        self.nearest_n_within_inner_with_scratch::<
            D,
            Vec<QueryResultItem<(), T, D::Output>>,
            EXCLUSIVE,
        >(
            query,
            max_dist,
            usize::MAX,
            true,
            result_capacity,
            stack,
        )
    }

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
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
        F: FnMut(QueryResultItem<(), T, D::Output>),
    {
        let mut req_ctx = ArchivedWithinUnsortedVisitReqCtx::<A, D::Output, EXCLUSIVE, K> {
            query,
            max_dist,
            _phantom: std::marker::PhantomData,
        };

        self.backtracking_query::<_, _, D>(&mut req_ctx, |leaf_idx, query_wide, req_ctx| {
            let mut results = VisitorResultCollection::new(&mut visitor);
            self.process_leaf_nearest_n_within::<
                D,
                QueryResultItem<(), T, D::Output>,
                _,
                EXCLUSIVE,
            >(
                leaf_idx,
                query_wide,
                req_ctx.max_dist(),
                &mut results,
            );
        });
    }

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
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
        F: FnMut(QueryResultItem<[A; K], T, D::Output>),
    {
        let mut req_ctx = ArchivedWithinUnsortedVisitReqCtx::<A, D::Output, EXCLUSIVE, K> {
            query,
            max_dist,
            _phantom: std::marker::PhantomData,
        };

        self.backtracking_query::<_, _, D>(&mut req_ctx, |leaf_idx, query_wide, req_ctx| {
            let mut results = VisitorResultCollection::new(&mut visitor);
            self.process_leaf_nearest_n_within::<
                D,
                QueryResultItem<[A; K], T, D::Output>,
                _,
                EXCLUSIVE,
            >(
                leaf_idx,
                query_wide,
                req_ctx.max_dist(),
                &mut results,
            );
        });
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
        let max_qty = max_qty.get();
        if max_qty == usize::MAX {
            self.nearest_n_within_inner_with_scratch::<
                D,
                Vec<QueryResultItem<(), T, D::Output>>,
                EXCLUSIVE,
            >(query, max_dist, max_qty, sorted, None, stack)
        } else {
            self.nearest_n_within_inner_with_scratch::<
                D,
                BinaryHeapResultCollection<QueryResultItem<(), T, D::Output>>,
                EXCLUSIVE,
            >(query, max_dist, max_qty, sorted, None, stack)
        }
    }

    fn nearest_n_within_inner_with_scratch<D, R, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
        max_qty: usize,
        sorted: bool,
        result_capacity: Option<NonZeroUsize>,
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
        let mut req_ctx = ArchivedNearestNWithinReqCtx::<A, T, D::Output, R, EXCLUSIVE, K> {
            query,
            max_dist,
            results: R::with_max_qty_and_capacity(max_qty, result_capacity),
            _phantom: std::marker::PhantomData,
        };

        self.backtracking_query_with_scratch::<_, _, D>(
            &mut req_ctx,
            stack,
            |leaf_idx, query_wide, req_ctx| {
                let leaf_max_dist = req_ctx.max_dist();
                self.process_leaf_nearest_n_within::<
                    D,
                    QueryResultItem<(), T, D::Output>,
                    R,
                    EXCLUSIVE,
                >(
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

    pub(crate) fn within_unsorted_visit_impl_with_scratch<D, F, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
        mut visitor: F,
        stack: &mut SS::Stack<D::Output>,
    ) where
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS>,
        F: FnMut(QueryResultItem<(), T, D::Output>),
    {
        let mut req_ctx = ArchivedWithinUnsortedVisitReqCtx::<A, D::Output, EXCLUSIVE, K> {
            query,
            max_dist,
            _phantom: std::marker::PhantomData,
        };

        self.backtracking_query_with_scratch::<_, _, D>(
            &mut req_ctx,
            stack,
            |leaf_idx, query_wide, req_ctx| {
                let mut results = VisitorResultCollection::new(&mut visitor);
                self.process_leaf_nearest_n_within::<
                    D,
                    QueryResultItem<(), T, D::Output>,
                    _,
                    EXCLUSIVE,
                >(
                    leaf_idx,
                    query_wide,
                    req_ctx.max_dist(),
                    &mut results,
                );
            },
        );
    }

    /// Returns a streaming iterator over all points within a given distance, unsorted.
    ///
    /// This avoids materializing the full result set returned by [`within_unsorted`](Self::within_unsorted).
    /// For the absolute lowest overhead, use [`within_unsorted_visit`](Self::within_unsorted_visit).
    #[allow(dead_code)]
    pub(crate) fn within_unsorted_iter<D>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
    ) -> crate::kd_tree::WithinUnsortedIter<'_, Self, A, T, SS, rkyv_08::Archived<LS>, D, false, K, B>
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
        crate::kd_tree::WithinUnsortedIter::new(self, query, max_dist)
    }
}

impl<A, T, SS, LS, const K: usize, const B: usize> ArchivedKdTree<A, T, SS, LS, K, B>
where
    A: rkyv_08::Archive + Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: rkyv_08::Archive,
    rkyv_08::Archived<LS>: LeafStrategy<A, T, SS, K, B>,
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
        let threshold_item = results.threshold_item();
        match <rkyv_08::Archived<LS> as LeafStrategy<A, T, SS, K, B>>::LEAF_PROJECTION {
            LeafProjection::LeafArena => {
                let arena = self.leaves().leaf_arena(leaf_idx);
                best_n_within_with_query_wide_arena::<A, T, D, R, EXCLUSIVE, K>(
                    &arena,
                    query_wide,
                    max_dist,
                    threshold_item,
                    results,
                );
            }
            LeafProjection::LeafView => {
                let leaf = self.leaves().leaf_view(leaf_idx);
                crate::leaf_view_chunked::best_n_within::best_n_within_with_query_wide::<
                    A,
                    T,
                    D,
                    R,
                    EXCLUSIVE,
                    K,
                    B,
                >(&leaf, query_wide, max_dist, threshold_item, results);
            }
        }
    }

    pub(crate) fn best_n_within_impl<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
        max_qty: NonZeroUsize,
    ) -> BinaryHeap<BestQueryResultItem<(), T, D::Output>>
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
        let mut req_ctx = ArchivedBestNWithinReqCtx::<A, D::Output, _, EXCLUSIVE, K> {
            query,
            max_dist,
            results:
                BinaryHeapResultCollection::<BestQueryResultItem<(), T, D::Output>>::with_max_qty(
                    max_qty.get(),
                ),
        };

        self.backtracking_query::<_, _, D>(&mut req_ctx, |leaf_idx, query_wide, req_ctx| {
            self.process_leaf_best_n_within::<D, _, EXCLUSIVE>(
                leaf_idx,
                query_wide,
                max_dist,
                &mut req_ctx.results,
            );
        });

        req_ctx.results.into_inner()
    }

    pub(crate) fn best_n_within_impl_with_scratch<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
        max_qty: NonZeroUsize,
        stack: &mut SS::Stack<D::Output>,
    ) -> BinaryHeap<BestQueryResultItem<(), T, D::Output>>
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
        let mut req_ctx = ArchivedBestNWithinReqCtx::<A, D::Output, _, EXCLUSIVE, K> {
            query,
            max_dist,
            results:
                BinaryHeapResultCollection::<BestQueryResultItem<(), T, D::Output>>::with_max_qty(
                    max_qty.get(),
                ),
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

        req_ctx.results.into_inner()
    }
}

struct ArchivedApproxNearestOneReqCtx<'a, A, O, const K: usize> {
    query: &'a [A; K],
    _phantom: std::marker::PhantomData<O>,
}

impl<A, O, const K: usize> QueryContext<A, O, K> for ArchivedApproxNearestOneReqCtx<'_, A, O, K> {
    fn query(&self) -> &[A; K] {
        self.query
    }

    fn max_dist(&self) -> O {
        panic!("approx_nearest_one should not be called with max_dist")
    }
}

struct ArchivedNearestOneReqCtx<'a, A, T, O, const K: usize>
where
    O: Axis<Coord = O>,
{
    query: &'a [A; K],
    best_dist: O,
    best_item: T,
}

impl<A, T, O, const K: usize> QueryContext<A, O, K> for ArchivedNearestOneReqCtx<'_, A, T, O, K>
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
    fn prune_on_equal_max_dist(&self) -> bool {
        true
    }
}

struct ArchivedNearestNWithinReqCtx<'a, A, T, O, R, const EXCLUSIVE: bool, const K: usize>
where
    O: Axis<Coord = O>,
{
    query: &'a [A; K],
    max_dist: O,
    results: R,
    _phantom: std::marker::PhantomData<T>,
}

struct ArchivedWithinUnsortedVisitReqCtx<'a, A, O, const EXCLUSIVE: bool, const K: usize>
where
    O: Axis<Coord = O>,
{
    query: &'a [A; K],
    max_dist: O,
    _phantom: std::marker::PhantomData<A>,
}

impl<A, O, const EXCLUSIVE: bool, const K: usize> QueryContext<A, O, K>
    for ArchivedWithinUnsortedVisitReqCtx<'_, A, O, EXCLUSIVE, K>
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

impl<A, T, O, R, const EXCLUSIVE: bool, const K: usize> QueryContext<A, O, K>
    for ArchivedNearestNWithinReqCtx<'_, A, T, O, R, EXCLUSIVE, K>
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
    fn prune_on_equal_max_dist(&self) -> bool {
        EXCLUSIVE
    }
}

struct ArchivedBestNWithinReqCtx<'a, A, O, R, const EXCLUSIVE: bool, const K: usize>
where
    O: Axis<Coord = O>,
{
    query: &'a [A; K],
    max_dist: O,
    results: R,
}

impl<A, O, R, const EXCLUSIVE: bool, const K: usize> QueryContext<A, O, K>
    for ArchivedBestNWithinReqCtx<'_, A, O, R, EXCLUSIVE, K>
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

#![allow(private_bounds)]

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::marker::PhantomData;
use std::num::{NonZero, NonZeroUsize};

use crate::dist::{DistanceMetric, DistanceMetricCore, QueryMetric};
use crate::kd_tree::query_stack::StackTrait;
#[cfg(feature = "rkyv_08")]
use crate::kd_tree::ArchivedKdTree;
use crate::kd_tree::{
    KdTree, KdTreeAccessor, KdTreeIter, KdTreeQueryOps, QueryScratch, WithinUnsortedIter,
};
use crate::leaf_view::TlsLeafScratch;
use crate::results::result_collection::DEFAULT_UNSORTED_RESULT_CAPACITY;
use crate::stem_strategy::donnelly::simd_full::{
    BacktrackBlock3, BacktrackBlock4, SimdSelectBestChildBlock3,
};
use crate::{Axis, BestQueryResultItem, Content, LeafStrategy, QueryResultItem, StemStrategy};

use super::periodic::{periodic_nearest_one_result, periodic_nearest_results};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BoundaryMode {
    Inclusive,
    Exclusive,
}

pub(crate) trait PeriodicAxis: Axis<Coord = Self> {
    fn periodic_box_is_valid(box_size: &[Self]) -> bool {
        box_size
            .iter()
            .all(|axis_len| Self::cmp(*axis_len, Self::zero()) == Ordering::Greater)
    }
}

impl PeriodicAxis for f32 {}
impl PeriodicAxis for f64 {}
#[cfg(feature = "f16")]
impl PeriodicAxis for half::f16 {}

#[doc(hidden)]
pub struct Include;
#[doc(hidden)]
pub struct Exclude;

#[doc(hidden)]
pub struct Projection<P, I, D>(PhantomData<(P, I, D)>);

#[doc(hidden)]
pub trait ProjectionField<T> {
    type Output;

    fn project(value: T) -> Self::Output;
}

trait PointProjectionField<A: Axis<Coord = A>, const K: usize>: ProjectionField<[A; K]> {
    fn absent_point() -> Self::Output;
}

impl<T> ProjectionField<T> for Include {
    type Output = T;

    #[inline(always)]
    fn project(value: T) -> Self::Output {
        value
    }
}

impl<T> ProjectionField<T> for Exclude {
    type Output = ();

    #[inline(always)]
    fn project(_value: T) -> Self::Output {}
}

impl<A: Axis<Coord = A>, const K: usize> PointProjectionField<A, K> for Include {
    #[inline(always)]
    fn absent_point() -> Self::Output {
        [A::zero(); K]
    }
}

impl<A: Axis<Coord = A>, const K: usize> PointProjectionField<A, K> for Exclude {
    #[inline(always)]
    fn absent_point() -> Self::Output {}
}

pub(crate) trait ProjectionSpec<A, T, O, const K: usize> {
    const WANTS_POINTS: bool;
    type NearestOut;
    type BestOut;

    fn nearest_from_parts(point: [A; K], item: T, distance: O) -> Self::NearestOut;
    fn best_from_parts(point: [A; K], item: T, distance: O) -> Self::BestOut;
}

impl<A, T, O, P, I, D, const K: usize> ProjectionSpec<A, T, O, K> for Projection<P, I, D>
where
    P: ProjectionField<[A; K]>,
    I: ProjectionField<T>,
    D: ProjectionField<O>,
{
    const WANTS_POINTS: bool = std::mem::size_of::<P::Output>() != 0;
    type NearestOut = QueryResultItem<P::Output, I::Output, D::Output>;
    type BestOut = BestQueryResultItem<P::Output, I::Output, D::Output>;

    #[inline(always)]
    fn nearest_from_parts(point: [A; K], item: T, distance: O) -> Self::NearestOut {
        QueryResultItem {
            point: P::project(point),
            item: I::project(item),
            distance: D::project(distance),
        }
    }

    #[inline(always)]
    fn best_from_parts(point: [A; K], item: T, distance: O) -> Self::BestOut {
        BestQueryResultItem {
            point: P::project(point),
            item: I::project(item),
            distance: D::project(distance),
        }
    }
}

#[inline(always)]
fn project_nearest_without_point<A, T, O, P, I, D, const K: usize>(
    result: QueryResultItem<(), T, O>,
) -> QueryResultItem<P::Output, I::Output, D::Output>
where
    A: Axis<Coord = A>,
    P: PointProjectionField<A, K>,
    I: ProjectionField<T>,
    D: ProjectionField<O>,
{
    QueryResultItem {
        point: P::absent_point(),
        item: I::project(result.item),
        distance: D::project(result.distance),
    }
}

#[inline(always)]
fn project_nearest_without_point_from_parts<A, T, O, P, I, D, const K: usize>(
    item: T,
    distance: O,
) -> QueryResultItem<P::Output, I::Output, D::Output>
where
    A: Axis<Coord = A>,
    P: PointProjectionField<A, K>,
    I: ProjectionField<T>,
    D: ProjectionField<O>,
{
    QueryResultItem {
        point: P::absent_point(),
        item: I::project(item),
        distance: D::project(distance),
    }
}

#[inline(always)]
fn project_best_without_point<A, T, O, P, I, D, const K: usize>(
    result: BestQueryResultItem<(), T, O>,
) -> BestQueryResultItem<P::Output, I::Output, D::Output>
where
    A: Axis<Coord = A>,
    P: PointProjectionField<A, K>,
    I: ProjectionField<T>,
    D: ProjectionField<O>,
{
    BestQueryResultItem {
        point: P::absent_point(),
        item: I::project(result.item),
        distance: D::project(result.distance),
    }
}

#[derive(Copy, Clone)]
pub(crate) struct FullNearest<A, T, O, const K: usize> {
    pub(crate) point: [A; K],
    pub(crate) item: T,
    pub(crate) distance: O,
}

#[derive(Copy, Clone)]
struct FullBest<A, T, O, const K: usize> {
    point: [A; K],
    item: T,
    distance: O,
}

impl<A, T, O: PartialOrd, const K: usize> PartialEq for FullNearest<A, T, O, K> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl<A, T, O: PartialOrd, const K: usize> Eq for FullNearest<A, T, O, K> {}

#[allow(clippy::non_canonical_partial_ord_impl)]
impl<A, T, O: PartialOrd, const K: usize> PartialOrd for FullNearest<A, T, O, K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl<A, T, O: PartialOrd, const K: usize> Ord for FullNearest<A, T, O, K> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

impl<A, T: Content + PartialOrd, O: PartialEq, const K: usize> PartialEq for FullBest<A, T, O, K> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.item == other.item
    }
}

impl<A, T: Content + PartialOrd, O: PartialEq, const K: usize> Eq for FullBest<A, T, O, K> {}

#[allow(clippy::non_canonical_partial_ord_impl)]
impl<A, T: Content + PartialOrd, O: PartialOrd, const K: usize> PartialOrd
    for FullBest<A, T, O, K>
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.item.partial_cmp(&other.item) {
            Some(Ordering::Equal) => self.distance.partial_cmp(&other.distance),
            ordering => ordering,
        }
    }
}

impl<A, T: Content + PartialOrd, O: PartialOrd, const K: usize> Ord for FullBest<A, T, O, K> {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.item.partial_cmp(&other.item) {
            Some(Ordering::Equal) => self
                .distance
                .partial_cmp(&other.distance)
                .unwrap_or(Ordering::Equal),
            Some(ordering) => ordering,
            None => Ordering::Equal,
        }
    }
}

pub(crate) trait QueryBuilderTreeOps<A, T, SS, LS, const K: usize, const B: usize>:
    KdTreeAccessor<A, T, SS, LS, K, B> + Sized
where
    A: Axis<Coord = A> + 'static,
    T: Content,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
{
    fn qb_nearest_one<D>(&self, query: &[A; K]) -> (D::Output, T)
    where
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + Default + 'static,
        SS: 'static;

    fn qb_approx_nearest_one<D>(&self, query: &[A; K]) -> (D::Output, T)
    where
        T: Copy + Default + PartialOrd + PartialEq,
        D: DistanceMetric<A, Output = A>;

    fn qb_nearest_n<D>(
        &self,
        query: &[A; K],
        max_qty: NonZeroUsize,
        sorted: bool,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static;

    fn qb_nearest_n_within<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        max_qty: NonZeroUsize,
        sorted: bool,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static;

    fn qb_within<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        result_capacity: Option<NonZeroUsize>,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static;

    fn qb_within_unsorted_visit<D, F, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        visitor: F,
    ) where
        T: PartialOrd,
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
        F: FnMut(QueryResultItem<(), T, D::Output>);

    fn qb_within_unsorted_visit_with_points<D, F, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        visitor: F,
    ) where
        T: PartialOrd,
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
        F: FnMut(QueryResultItem<[A; K], T, D::Output>);

    #[inline]
    fn qb_within_unsorted_projected<D, R, F, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        result_capacity: Option<NonZeroUsize>,
        mut project: F,
    ) -> Vec<R>
    where
        T: PartialOrd,
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
        F: FnMut(QueryResultItem<(), T, D::Output>) -> R,
    {
        let mut results = Vec::with_capacity(
            result_capacity.map_or(DEFAULT_UNSORTED_RESULT_CAPACITY, NonZeroUsize::get),
        );
        self.qb_within_unsorted_visit::<D, _, EXCLUSIVE>(query, radius, |result| {
            results.push(project(result));
        });
        results
    }

    #[inline]
    fn qb_within_unsorted_projected_with_points<D, R, F, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        result_capacity: Option<NonZeroUsize>,
        mut project: F,
    ) -> Vec<R>
    where
        T: PartialOrd,
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
        F: FnMut([A; K], T, D::Output) -> R,
    {
        let mut results = Vec::with_capacity(
            result_capacity.map_or(DEFAULT_UNSORTED_RESULT_CAPACITY, NonZeroUsize::get),
        );
        self.qb_within_unsorted_visit_with_points::<D, _, EXCLUSIVE>(query, radius, |result| {
            results.push(project(result.point, result.item, result.distance))
        });
        results
    }

    fn qb_best_n_within<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        max_qty: NonZero<usize>,
    ) -> BinaryHeap<BestQueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static;
}

pub(crate) trait QueryBuilderScratchTreeOps<A, T, SS, LS, const K: usize, const B: usize>:
    QueryBuilderTreeOps<A, T, SS, LS, K, B>
where
    A: Axis<Coord = A> + 'static,
    T: Content,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
{
    fn qb_nearest_one_with_scratch<D>(
        &self,
        query: &[A; K],
        stack: &mut SS::Stack<D::Output>,
    ) -> (D::Output, T)
    where
        T: PartialOrd,
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS>;

    fn qb_nearest_n_with_scratch<D>(
        &self,
        query: &[A; K],
        max_qty: NonZeroUsize,
        sorted: bool,
        stack: &mut SS::Stack<D::Output>,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS>;

    fn qb_nearest_n_within_with_scratch<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        max_qty: NonZeroUsize,
        sorted: bool,
        stack: &mut SS::Stack<D::Output>,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS>;

    fn qb_within_with_scratch<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        result_capacity: Option<NonZeroUsize>,
        stack: &mut SS::Stack<D::Output>,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS>;

    fn qb_within_unsorted_visit_with_scratch<D, F, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        visitor: F,
        stack: &mut SS::Stack<D::Output>,
    ) where
        T: PartialOrd,
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS>,
        F: FnMut(QueryResultItem<(), T, D::Output>);

    #[inline]
    fn qb_within_unsorted_projected_with_scratch<D, R, F, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        result_capacity: Option<NonZeroUsize>,
        mut project: F,
        stack: &mut SS::Stack<D::Output>,
    ) -> Vec<R>
    where
        T: PartialOrd,
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS>,
        F: FnMut(QueryResultItem<(), T, D::Output>) -> R,
    {
        let mut results = Vec::with_capacity(
            result_capacity.map_or(DEFAULT_UNSORTED_RESULT_CAPACITY, NonZeroUsize::get),
        );
        self.qb_within_unsorted_visit_with_scratch::<D, _, EXCLUSIVE>(
            query,
            radius,
            |result| results.push(project(result)),
            stack,
        );
        results
    }

    fn qb_best_n_within_with_scratch<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        max_qty: NonZero<usize>,
        stack: &mut SS::Stack<D::Output>,
    ) -> BinaryHeap<BestQueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS>;
}

impl<A, T, SS, LS, const K: usize, const B: usize> QueryBuilderTreeOps<A, T, SS, LS, K, B>
    for KdTree<A, T, SS, LS, K, B>
where
    A: Axis<Coord = A> + 'static,
    T: Content,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
{
    #[inline]
    fn qb_nearest_one<D>(&self, query: &[A; K]) -> (D::Output, T)
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
        self.nearest_one::<D>(query)
    }

    #[inline]
    fn qb_approx_nearest_one<D>(&self, query: &[A; K]) -> (D::Output, T)
    where
        T: Copy + Default + PartialOrd + PartialEq,
        D: DistanceMetric<A, Output = A>,
    {
        self.approx_nearest_one::<D>(query)
    }

    #[inline]
    fn qb_nearest_n<D>(
        &self,
        query: &[A; K],
        max_qty: NonZeroUsize,
        sorted: bool,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    {
        self.nearest_n::<D>(query, max_qty, sorted)
    }

    #[inline]
    fn qb_nearest_n_within<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        max_qty: NonZeroUsize,
        sorted: bool,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    {
        self.nearest_n_within_impl::<D, EXCLUSIVE>(query, radius, max_qty, sorted)
    }

    #[inline]
    fn qb_within<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        result_capacity: Option<NonZeroUsize>,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    {
        self.within_impl::<D, EXCLUSIVE>(query, radius, result_capacity)
    }

    #[inline]
    fn qb_within_unsorted_visit<D, F, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        visitor: F,
    ) where
        T: PartialOrd,
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
        self.within_unsorted_visit_impl::<D, F, EXCLUSIVE>(query, radius, visitor)
    }

    #[inline]
    fn qb_within_unsorted_visit_with_points<D, F, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        visitor: F,
    ) where
        T: PartialOrd,
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
        self.within_unsorted_visit_with_points_impl::<D, F, EXCLUSIVE>(query, radius, visitor)
    }

    #[inline]
    fn qb_best_n_within<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        max_qty: NonZero<usize>,
    ) -> BinaryHeap<BestQueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    {
        self.best_n_within_impl::<D, EXCLUSIVE>(query, radius, max_qty)
    }
}

impl<A, T, SS, LS, const K: usize, const B: usize> QueryBuilderScratchTreeOps<A, T, SS, LS, K, B>
    for KdTree<A, T, SS, LS, K, B>
where
    A: Axis<Coord = A> + 'static,
    T: Content,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
{
    #[inline]
    fn qb_nearest_one_with_scratch<D>(
        &self,
        query: &[A; K],
        stack: &mut SS::Stack<D::Output>,
    ) -> (D::Output, T)
    where
        T: PartialOrd,
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS>,
    {
        self.nearest_one_with_scratch::<D>(query, stack)
    }

    #[inline]
    fn qb_nearest_n_with_scratch<D>(
        &self,
        query: &[A; K],
        max_qty: NonZeroUsize,
        sorted: bool,
        stack: &mut SS::Stack<D::Output>,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS>,
    {
        self.nearest_n_with_scratch::<D>(query, max_qty, sorted, stack)
    }

    #[inline]
    fn qb_nearest_n_within_with_scratch<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        max_qty: NonZeroUsize,
        sorted: bool,
        stack: &mut SS::Stack<D::Output>,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS>,
    {
        self.nearest_n_within_impl_with_scratch::<D, EXCLUSIVE>(
            query, radius, max_qty, sorted, stack,
        )
    }

    #[inline]
    fn qb_within_with_scratch<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        result_capacity: Option<NonZeroUsize>,
        stack: &mut SS::Stack<D::Output>,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS>,
    {
        self.within_impl_with_scratch::<D, EXCLUSIVE>(query, radius, result_capacity, stack)
    }

    #[inline]
    fn qb_within_unsorted_visit_with_scratch<D, F, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        visitor: F,
        stack: &mut SS::Stack<D::Output>,
    ) where
        T: PartialOrd,
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
        self.within_unsorted_visit_impl_with_scratch::<D, F, EXCLUSIVE>(
            query, radius, visitor, stack,
        )
    }

    #[inline]
    fn qb_best_n_within_with_scratch<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        max_qty: NonZero<usize>,
        stack: &mut SS::Stack<D::Output>,
    ) -> BinaryHeap<BestQueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS>,
    {
        self.best_n_within_impl_with_scratch::<D, EXCLUSIVE>(query, radius, max_qty, stack)
    }
}

#[cfg(feature = "rkyv_08")]
impl<A, T, SS, LS, const K: usize, const B: usize>
    QueryBuilderTreeOps<A, T, SS, rkyv_08::Archived<LS>, K, B>
    for ArchivedKdTree<A, T, SS, LS, K, B>
where
    A: rkyv_08::Archive + Axis<Coord = A> + 'static,
    T: Content + PartialOrd + PartialEq,
    SS: StemStrategy,
    LS: rkyv_08::Archive,
    rkyv_08::Archived<LS>: LeafStrategy<A, T, SS, K, B>,
{
    #[inline]
    fn qb_nearest_one<D>(&self, query: &[A; K]) -> (D::Output, T)
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
        self.nearest_one::<D>(query)
    }

    #[inline]
    fn qb_approx_nearest_one<D>(&self, query: &[A; K]) -> (D::Output, T)
    where
        T: Copy + Default + PartialOrd + PartialEq,
        D: DistanceMetric<A, Output = A>,
    {
        self.approx_nearest_one::<D>(query)
    }

    #[inline]
    fn qb_nearest_n<D>(
        &self,
        query: &[A; K],
        max_qty: NonZeroUsize,
        sorted: bool,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    {
        self.nearest_n::<D>(query, max_qty, sorted)
    }

    #[inline]
    fn qb_nearest_n_within<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        max_qty: NonZeroUsize,
        sorted: bool,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    {
        self.nearest_n_within_impl::<D, EXCLUSIVE>(query, radius, max_qty, sorted)
    }

    #[inline]
    fn qb_within<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        result_capacity: Option<NonZeroUsize>,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    {
        self.within_impl::<D, EXCLUSIVE>(query, radius, result_capacity)
    }

    #[inline]
    fn qb_within_unsorted_visit<D, F, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        visitor: F,
    ) where
        T: PartialOrd,
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
        self.within_unsorted_visit_impl::<D, F, EXCLUSIVE>(query, radius, visitor)
    }

    #[inline]
    fn qb_within_unsorted_visit_with_points<D, F, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        visitor: F,
    ) where
        T: PartialOrd,
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
        self.within_unsorted_visit_with_points_impl::<D, F, EXCLUSIVE>(query, radius, visitor)
    }

    #[inline]
    fn qb_best_n_within<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        max_qty: NonZero<usize>,
    ) -> BinaryHeap<BestQueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    {
        self.best_n_within_impl::<D, EXCLUSIVE>(query, radius, max_qty)
    }
}

#[cfg(feature = "rkyv_08")]
impl<A, T, SS, LS, const K: usize, const B: usize>
    QueryBuilderScratchTreeOps<A, T, SS, rkyv_08::Archived<LS>, K, B>
    for ArchivedKdTree<A, T, SS, LS, K, B>
where
    A: rkyv_08::Archive + Axis<Coord = A> + 'static,
    T: Content + PartialOrd + PartialEq,
    SS: StemStrategy,
    LS: rkyv_08::Archive,
    rkyv_08::Archived<LS>: LeafStrategy<A, T, SS, K, B>,
{
    #[inline]
    fn qb_nearest_one_with_scratch<D>(
        &self,
        query: &[A; K],
        stack: &mut SS::Stack<D::Output>,
    ) -> (D::Output, T)
    where
        T: PartialOrd,
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS>,
    {
        self.nearest_one_with_scratch::<D>(query, stack)
    }

    #[inline]
    fn qb_nearest_n_with_scratch<D>(
        &self,
        query: &[A; K],
        max_qty: NonZeroUsize,
        sorted: bool,
        stack: &mut SS::Stack<D::Output>,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS>,
    {
        self.nearest_n_with_scratch::<D>(query, max_qty, sorted, stack)
    }

    #[inline]
    fn qb_nearest_n_within_with_scratch<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        max_qty: NonZeroUsize,
        sorted: bool,
        stack: &mut SS::Stack<D::Output>,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS>,
    {
        self.nearest_n_within_impl_with_scratch::<D, EXCLUSIVE>(
            query, radius, max_qty, sorted, stack,
        )
    }

    #[inline]
    fn qb_within_with_scratch<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        result_capacity: Option<NonZeroUsize>,
        stack: &mut SS::Stack<D::Output>,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS>,
    {
        self.within_impl_with_scratch::<D, EXCLUSIVE>(query, radius, result_capacity, stack)
    }

    #[inline]
    fn qb_within_unsorted_visit_with_scratch<D, F, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        visitor: F,
        stack: &mut SS::Stack<D::Output>,
    ) where
        T: PartialOrd,
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
        self.within_unsorted_visit_impl_with_scratch::<D, F, EXCLUSIVE>(
            query, radius, visitor, stack,
        )
    }

    #[inline]
    fn qb_best_n_within_with_scratch<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        max_qty: NonZero<usize>,
        stack: &mut SS::Stack<D::Output>,
    ) -> BinaryHeap<BestQueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS>,
    {
        self.best_n_within_impl_with_scratch::<D, EXCLUSIVE>(query, radius, max_qty, stack)
    }
}

#[doc(hidden)]
pub struct NoMetric;

pub struct SelectedMetric<D>(PhantomData<D>);
#[doc(hidden)]
pub struct RootState;
#[doc(hidden)]
pub struct CartesianSpace;
#[doc(hidden)]
pub struct PeriodicSpace;
#[doc(hidden)]
pub struct NearestOneState;
#[doc(hidden)]
pub struct ApproxNearestOneState;
#[doc(hidden)]
pub struct NearestNState;
#[doc(hidden)]
pub struct NearestNWithinState;
#[doc(hidden)]
pub struct WithinState;
#[doc(hidden)]
pub struct BestNWithinState;

trait ResultFamily {}
impl ResultFamily for NearestOneState {}
impl ResultFamily for ApproxNearestOneState {}
impl ResultFamily for NearestNState {}
impl ResultFamily for NearestNWithinState {}
impl ResultFamily for WithinState {}
impl ResultFamily for BestNWithinState {}

trait ConfigurableScratchQueryFamily {}
impl ConfigurableScratchQueryFamily for NearestOneState {}
impl ConfigurableScratchQueryFamily for NearestNState {}
impl ConfigurableScratchQueryFamily for NearestNWithinState {}
impl ConfigurableScratchQueryFamily for WithinState {}
impl ConfigurableScratchQueryFamily for BestNWithinState {}

trait BuilderMetric<A, const K: usize> {
    type Output: Copy;
}

impl<A, const K: usize> BuilderMetric<A, K> for NoMetric {
    type Output = ();
}

impl<A: Copy, D, const K: usize> BuilderMetric<A, K> for SelectedMetric<D>
where
    D: DistanceMetricCore<A>,
{
    type Output = D::Output;
}

/// A fluent query builder created by [`KdTree::query`](crate::kd_tree::KdTree::query).
///
/// A query is assembled in stages: choose a query type, apply optional query
/// modifiers, choose which fields should appear in each result, and then
/// finish with a terminal method.
///
/// # Query Type
///
/// The first choice is the search family and distance metric:
///
/// - [`nearest_one`](Self::nearest_one)`::<D>()` finds the single nearest
///   neighbour.
/// - [`nearest_n`](Self::nearest_n)`::<D>(n)` finds the `n` nearest
///   neighbours.
/// - [`nearest_n`](Self::nearest_n)`::<D>(n).`[`within`](Self::within)`(radius)`
///   finds up to `n` neighbours, but only among entries inside `radius`.
/// - [`within`](Self::within)`::<D>(radius)` returns every entry inside
///   `radius`.
/// - [`best_n_within`](Self::best_n_within)`::<D>(radius, n)` returns up to
///   `n` entries inside `radius`, ranked by item ordering rather than by
///   distance.
///
/// The metric `D` is selected when the query type is chosen, so each fluent
/// chain starts by deciding both what kind of query it is and which distance
/// metric defines closeness.
///
/// # Query Options
///
/// ## `approx`
///
/// [`approx`](Self::approx) is available only after
/// [`nearest_one`](Self::nearest_one)`::<D>()`. It switches the exact
/// nearest-neighbour search to a descent-only approximate nearest-neighbour (ANN)
/// search. This is much faster, but the result may not be the truly nearest
/// neighbour.
///
/// ## `exclusive_boundaries`
///
/// Radius-bounded queries use inclusive `<= radius` semantics by default.
/// [`exclusive_boundaries`](Self::exclusive_boundaries) changes those queries
/// to strict `< radius` semantics. This applies to
/// [`within`](Self::within)`::<D>(radius)`,
/// [`nearest_n`](Self::nearest_n)`::<D>(n).`[`within`](Self::within)`(radius)`,
/// and [`best_n_within`](Self::best_n_within)`::<D>(radius, n)`.
///
/// ## `periodic_boundary_conditions`
///
/// [`periodic_boundary_condition`](Self::periodic_boundary_condition)`(box_size)`
/// must be chosen before the query type. It changes the space from Cartesian
/// coordinates to periodic wrapping, using `box_size` as the period along each
/// axis. Every entry in `box_size` must be strictly positive.
///
/// Periodic queries support [`nearest_one`](Self::nearest_one),
/// [`nearest_n`](Self::nearest_n),
/// [`nearest_n`](Self::nearest_n)`(...).`[`within`](Self::within)`(...)`, and
/// [`within`](Self::within). [`best_n_within`](Self::best_n_within) is
/// Cartesian-only.
///
/// # Scratch Strategy
///
/// Queries that involve backtracking need to store temporary state. The scratch
/// strategy determines how this state is managed. Cartesian (ie not PBC),
/// non-approximate queries that do not include point coordinates can
/// choose how their traversal stack scratch is provided. Both options
/// avoid a heap allocation on the query path (unless your tree is _extremely_
/// deep - typically deeper than you could ever find in a balanced tree and only
/// something that you might find in a _very_ unbalanced tree).
///
/// - By default, queries store scratch in the terminal call's stack frame.
///   This keeps scratch local to each query without a thread-local lookup.
/// - `with_scratch(&mut scratch)` uses caller-owned reusable scratch created by
///   `KdTree::create_scratch()` or the archived-tree method of the same name.
///   This is particularly suited to batches of compatible queries but can can
///   make sense anywhere you'll have the opportunity to re-use the same scratch
///   for more than one query.
///
/// These scratch configuration methods are experimental during the 6.0.0 alpha;
/// not every option is guaranteed to remain in the final 6.0.0 API.
///
/// Scratch strategy selection is not available for `with_points()` or `iter()`.
/// Those paths currently maintain traversal state in a more complex state
/// machine rather than the stack-scratch query orchestrator.
///
/// # Result Content
///
/// Result projection is independent from the query type. By default, builder
/// results include the stored item and the computed distance, but not the
/// point coordinates.
///
/// - [`with_points`](Self::with_points) includes point coordinates.
/// - [`without_points`](Self::without_points) removes point coordinates.
/// - [`with_items`](Self::with_items) includes stored items.
/// - [`without_items`](Self::without_items) removes stored items.
/// - [`with_distances`](Self::with_distances) includes distances.
/// - [`without_distances`](Self::without_distances) removes distances.
///
/// These toggles rewrite the eventual result item type by replacing omitted
/// fields with `()`. For example, removing items produces
/// [`QueryResultItem<_, (), _>`](crate::QueryResultItem).
///
/// [`with_points`](Self::with_points) is available only for Cartesian queries.
/// Periodic queries do not expose a point projection toggle.
///
/// # Result Type
///
/// [`execute`](Self::execute) eagerly runs the query and returns the
/// query-specific container:
///
/// - [`nearest_one`](Self::nearest_one) and
///   [`nearest_one`](Self::nearest_one)`(...).`[`approx`](Self::approx)`()`
///   return one [`QueryResultItem`](crate::QueryResultItem).
/// - [`nearest_n`](Self::nearest_n),
///   [`nearest_n`](Self::nearest_n)`(...).`[`within`](Self::within)`(...)`, and
///   [`within`](Self::within)`(...)` return `Vec<QueryResultItem<...>>`.
/// - [`best_n_within`](Self::best_n_within)`(...)` returns
///   `BinaryHeap<BestQueryResultItem<...>>`.
///
/// [`iter`](Self::iter) and [`visit`](Self::visit) are streaming terminal
/// forms for Cartesian [`within`](Self::within)`(...)` queries after
/// [`unsorted`](Self::unsorted) has been selected:
///
/// - [`iter`](Self::iter) returns a lazy iterator instead of materializing a
///   `Vec`.
/// - [`visit`](Self::visit) calls a visitor closure for each matching result.
///
/// [`unsorted`](Self::unsorted) is also available for the other multi-result
/// query families when traversal order is acceptable.
///
/// ## Result allocation
///
/// Eager [`within`](Self::within)`::<D>(radius)` queries materialize every
/// matching result in a `Vec`. If the approximate number of matches is known,
/// [`with_result_capacity`](Self::with_result_capacity) can reserve that space
/// up front and avoid growth reallocations. The hint does not limit the number
/// of results returned; omit it when a reasonable estimate is unavailable.
///
/// The concrete generic parameters on this type encode the current builder
/// state and are considered an implementation detail of the fluent API.
pub struct QueryBuilder<
    'a,
    Tree,
    A,
    T,
    SS,
    LS,
    D,
    Family,
    Space,
    Pj,
    const SORTED: bool,
    const EXCLUSIVE: bool,
    const K: usize,
    const B: usize,
> where
    D: BuilderMetric<A, K>,
{
    tree: &'a Tree,
    query: &'a [A; K],
    box_size: Option<&'a [A; K]>,
    max_qty: Option<NonZeroUsize>,
    result_capacity: Option<NonZeroUsize>,
    radius: <D as BuilderMetric<A, K>>::Output,
    _phantom: PhantomData<(T, SS, LS, Family, Space, Pj)>,
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        Space,
        Pj,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > Copy for QueryBuilder<'a, Tree, A, T, SS, LS, D, Family, Space, Pj, SORTED, EXCLUSIVE, K, B>
where
    D: BuilderMetric<A, K>,
{
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        Space,
        Pj,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > Clone for QueryBuilder<'a, Tree, A, T, SS, LS, D, Family, Space, Pj, SORTED, EXCLUSIVE, K, B>
where
    D: BuilderMetric<A, K>,
{
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

type RootQueryBuilder<'a, Tree, A, T, SS, LS, const K: usize, const B: usize> = QueryBuilder<
    'a,
    Tree,
    A,
    T,
    SS,
    LS,
    NoMetric,
    RootState,
    CartesianSpace,
    Projection<Exclude, Include, Include>,
    true,
    false,
    K,
    B,
>;

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        Space,
        Pj,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > QueryBuilder<'a, Tree, A, T, SS, LS, D, Family, Space, Pj, SORTED, EXCLUSIVE, K, B>
where
    D: BuilderMetric<A, K>,
{
    #[inline(always)]
    fn rebind<D2, Family2, Space2, Pj2, const SORTED2: bool, const EXCLUSIVE2: bool>(
        self,
        box_size: Option<&'a [A; K]>,
        max_qty: Option<NonZeroUsize>,
        radius: <D2 as BuilderMetric<A, K>>::Output,
    ) -> QueryBuilder<'a, Tree, A, T, SS, LS, D2, Family2, Space2, Pj2, SORTED2, EXCLUSIVE2, K, B>
    where
        D2: BuilderMetric<A, K>,
    {
        QueryBuilder {
            tree: self.tree,
            query: self.query,
            box_size,
            max_qty,
            result_capacity: self.result_capacity,
            radius,
            _phantom: PhantomData,
        }
    }
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Space,
        Pj,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > QueryBuilder<'a, Tree, A, T, SS, LS, D, WithinState, Space, Pj, SORTED, EXCLUSIVE, K, B>
where
    D: BuilderMetric<A, K>,
{
    /// Reserves space for the expected number of materialized radius-query results.
    ///
    /// A good estimate avoids growth reallocations without limiting the number
    /// of results returned. Overestimating can reserve substantially more memory
    /// than the query needs, so omit the hint when no reasonable estimate is
    /// available.
    ///
    /// This option applies to both sorted and unsorted [`within`](Self::within)
    /// queries, including periodic and archived-tree queries. Streaming
    /// [`iter`](Self::iter) and [`visit`](Self::visit) queries do not materialize
    /// a result `Vec` and therefore do not use a capacity hint.
    ///
    /// # Panics
    ///
    /// Panics if `result_capacity` is zero.
    #[inline]
    pub fn with_result_capacity(mut self, result_capacity: usize) -> Self {
        self.result_capacity = Some(
            NonZeroUsize::new(result_capacity).expect("result capacity must be greater than zero"),
        );
        self
    }
}

#[doc(hidden)]
pub trait ExecuteQueryBuilder {
    #[doc(hidden)]
    type Output;

    #[doc(hidden)]
    fn execute(self) -> Self::Output;
}

#[doc(hidden)]
pub trait ExecuteQueryBuilderWithScratch<SS, O>
where
    SS: StemStrategy,
{
    #[doc(hidden)]
    type Output;

    #[doc(hidden)]
    fn execute_with_scratch(self, scratch: &mut QueryScratch<SS, O>) -> Self::Output;
}

/// Query builder adapter that uses caller-owned reusable query scratch.
pub struct ScratchQueryBuilder<'s, B, SS, O>
where
    SS: StemStrategy,
{
    builder: B,
    scratch: &'s mut QueryScratch<SS, O>,
}

impl<B, SS, O> ScratchQueryBuilder<'_, B, SS, O>
where
    SS: StemStrategy,
    B: ExecuteQueryBuilderWithScratch<SS, O>,
{
    /// Executes the configured query with caller-owned reusable scratch storage.
    #[inline]
    pub fn execute(self) -> <B as ExecuteQueryBuilderWithScratch<SS, O>>::Output {
        self.builder.execute_with_scratch(self.scratch)
    }
}

impl<
        's,
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        I,
        Dp,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    >
    ScratchQueryBuilder<
        's,
        QueryBuilder<
            'a,
            Tree,
            A,
            T,
            SS,
            LS,
            SelectedMetric<D>,
            WithinState,
            CartesianSpace,
            Projection<Exclude, I, Dp>,
            false,
            EXCLUSIVE,
            K,
            B,
        >,
        SS,
        D::Output,
    >
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B> + 'a,
    Tree: QueryBuilderScratchTreeOps<A, T, SS, LS, K, B> + 'a,
    D: DistanceMetric<A> + 'a,
    D::Output: crate::stem_strategy::SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static,
    SS::Stack<D::Output>: StackTrait<D::Output, SS>,
    I: ProjectionField<T> + 'a,
    Dp: ProjectionField<D::Output> + 'a,
{
    /// Visits matching entries using caller-owned reusable scratch storage.
    #[inline]
    pub fn visit<F>(self, mut visitor: F)
    where
        F: FnMut(QueryResultItem<(), I::Output, Dp::Output>),
    {
        let builder = self.builder;
        with_cleared_scratch(self.scratch, |stack| {
            builder
                .tree
                .qb_within_unsorted_visit_with_scratch::<D, _, EXCLUSIVE>(
                    builder.query,
                    builder.radius,
                    move |result| {
                        visitor(project_nearest_without_point::<
                            A,
                            T,
                            D::Output,
                            Exclude,
                            I,
                            Dp,
                            K,
                        >(result))
                    },
                    stack,
                );
        });
    }
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        Space,
        Pj,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > QueryBuilder<'a, Tree, A, T, SS, LS, D, Family, Space, Pj, SORTED, EXCLUSIVE, K, B>
where
    D: BuilderMetric<A, K>,
    Self: ExecuteQueryBuilder,
{
    /// Executes the configured query and returns the query-specific result type.
    #[inline]
    pub fn execute(self) -> <Self as ExecuteQueryBuilder>::Output {
        <Self as ExecuteQueryBuilder>::execute(self)
    }
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        I,
        Dp,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    >
    QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        Family,
        CartesianSpace,
        Projection<Exclude, I, Dp>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderScratchTreeOps<A, T, SS, LS, K, B>,
    D: QueryMetric<A>,
    D::Output: crate::stem_strategy::SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static,
    SS::Stack<D::Output>: StackTrait<D::Output, SS>,
    I: ProjectionField<T>,
    Dp: ProjectionField<D::Output>,
    Family: ConfigurableScratchQueryFamily,
{
    /// Uses caller-owned reusable scratch instead of per-call local scratch.
    ///
    /// Scratch must have been created for the same stem strategy and distance
    /// output type, for example with [`KdTree::create_scratch`] or the
    /// archived-tree method of the same name.
    ///
    /// This alpha API is experimental; see the
    /// [`QueryBuilder` scratch strategy section](QueryBuilder#scratch-strategy)
    /// for the available strategies and current limitations.
    #[inline]
    pub fn with_scratch<'s>(
        self,
        scratch: &'s mut QueryScratch<SS, D::Output>,
    ) -> ScratchQueryBuilder<'s, Self, SS, D::Output> {
        ScratchQueryBuilder {
            builder: self,
            scratch,
        }
    }
}

#[doc(hidden)]
pub trait WithoutPointsQueryBuilder {
    #[doc(hidden)]
    type Output;

    #[doc(hidden)]
    fn without_points(self) -> Self::Output;
}

#[doc(hidden)]
pub trait WithItemsQueryBuilder {
    #[doc(hidden)]
    type Output;

    #[doc(hidden)]
    fn with_items(self) -> Self::Output;
}

#[doc(hidden)]
pub trait WithoutItemsQueryBuilder {
    #[doc(hidden)]
    type Output;

    #[doc(hidden)]
    fn without_items(self) -> Self::Output;
}

#[doc(hidden)]
pub trait WithDistancesQueryBuilder {
    #[doc(hidden)]
    type Output;

    #[doc(hidden)]
    fn with_distances(self) -> Self::Output;
}

#[doc(hidden)]
pub trait WithoutDistancesQueryBuilder {
    #[doc(hidden)]
    type Output;

    #[doc(hidden)]
    fn without_distances(self) -> Self::Output;
}

#[doc(hidden)]
pub trait WithPointsQueryBuilder {
    #[doc(hidden)]
    type Output;

    #[doc(hidden)]
    fn with_points(self) -> Self::Output;
}

#[doc(hidden)]
pub trait PeriodicBoundaryConditionQueryBuilder<'a, A, const K: usize> {
    #[doc(hidden)]
    type Output;

    #[doc(hidden)]
    fn periodic_boundary_condition(self, box_size: &'a [A; K]) -> Self::Output;
}

#[doc(hidden)]
pub trait NearestOneQueryBuilder<A: Copy, const K: usize, D>
where
    D: DistanceMetric<A>,
{
    #[doc(hidden)]
    type Output;

    #[doc(hidden)]
    fn nearest_one(self) -> Self::Output;
}

#[doc(hidden)]
pub trait NearestNQueryBuilder<A: Copy, const K: usize, D>
where
    D: DistanceMetric<A>,
{
    #[doc(hidden)]
    type Output;

    #[doc(hidden)]
    fn nearest_n(self, max_qty: NonZeroUsize) -> Self::Output;
}

#[doc(hidden)]
pub trait WithinQueryBuilder<A: Copy, const K: usize, D>
where
    D: DistanceMetric<A>,
{
    #[doc(hidden)]
    type Output;

    #[doc(hidden)]
    fn within(self, radius: D::Output) -> Self::Output;
}

#[doc(hidden)]
pub trait BestNWithinQueryBuilder<A: Copy, const K: usize, D>
where
    D: DistanceMetric<A>,
{
    #[doc(hidden)]
    type Output;

    #[doc(hidden)]
    fn best_n_within(self, radius: D::Output, max_qty: NonZero<usize>) -> Self::Output;
}

#[doc(hidden)]
pub trait ApproxQueryBuilder {
    #[doc(hidden)]
    type Output;

    #[doc(hidden)]
    fn approx(self) -> Self::Output;
}

#[doc(hidden)]
pub trait UnsortedQueryBuilder {
    #[doc(hidden)]
    type Output;

    #[doc(hidden)]
    fn unsorted(self) -> Self::Output;
}

#[doc(hidden)]
pub trait ExclusiveBoundariesQueryBuilder {
    #[doc(hidden)]
    type Output;

    #[doc(hidden)]
    fn exclusive_boundaries(self) -> Self::Output;
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        Space,
        Pj,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > QueryBuilder<'a, Tree, A, T, SS, LS, D, Family, Space, Pj, SORTED, EXCLUSIVE, K, B>
where
    D: BuilderMetric<A, K>,
{
    /// Excludes stored point coordinates from the eventual query results.
    #[inline]
    pub fn without_points(self) -> <Self as WithoutPointsQueryBuilder>::Output
    where
        Self: WithoutPointsQueryBuilder,
    {
        <Self as WithoutPointsQueryBuilder>::without_points(self)
    }

    /// Includes stored items in the eventual query results.
    #[inline]
    pub fn with_items(self) -> <Self as WithItemsQueryBuilder>::Output
    where
        Self: WithItemsQueryBuilder,
    {
        <Self as WithItemsQueryBuilder>::with_items(self)
    }

    /// Excludes stored items from the eventual query results.
    #[inline]
    pub fn without_items(self) -> <Self as WithoutItemsQueryBuilder>::Output
    where
        Self: WithoutItemsQueryBuilder,
    {
        <Self as WithoutItemsQueryBuilder>::without_items(self)
    }

    /// Includes distances in the eventual query results.
    #[inline]
    pub fn with_distances(self) -> <Self as WithDistancesQueryBuilder>::Output
    where
        Self: WithDistancesQueryBuilder,
    {
        <Self as WithDistancesQueryBuilder>::with_distances(self)
    }

    /// Excludes distances from the eventual query results.
    #[inline]
    pub fn without_distances(self) -> <Self as WithoutDistancesQueryBuilder>::Output
    where
        Self: WithoutDistancesQueryBuilder,
    {
        <Self as WithoutDistancesQueryBuilder>::without_distances(self)
    }

    /// Includes stored point coordinates in the eventual query results.
    ///
    /// This is available only for non-periodic queries.
    ///
    /// [`with_scratch`](Self::with_scratch) is not currently available after
    /// `with_points()`. See the
    /// [`QueryBuilder` scratch strategy section](QueryBuilder#scratch-strategy)
    /// for more detail.
    #[inline]
    pub fn with_points(self) -> <Self as WithPointsQueryBuilder>::Output
    where
        Self: WithPointsQueryBuilder,
    {
        <Self as WithPointsQueryBuilder>::with_points(self)
    }

    /// Interprets subsequent queries using periodic boundary conditions.
    ///
    /// `box_size` gives the period along each axis and every value must be
    /// strictly positive.
    #[inline]
    pub fn periodic_boundary_condition(
        self,
        box_size: &'a [A; K],
    ) -> <Self as PeriodicBoundaryConditionQueryBuilder<'a, A, K>>::Output
    where
        Self: PeriodicBoundaryConditionQueryBuilder<'a, A, K>,
    {
        <Self as PeriodicBoundaryConditionQueryBuilder<'a, A, K>>::periodic_boundary_condition(
            self, box_size,
        )
    }

    /// Selects an exact nearest-neighbour query.
    #[inline]
    pub fn nearest_one<Dq>(self) -> <Self as NearestOneQueryBuilder<A, K, Dq>>::Output
    where
        A: Copy,
        Dq: DistanceMetric<A>,
        Self: NearestOneQueryBuilder<A, K, Dq>,
    {
        <Self as NearestOneQueryBuilder<A, K, Dq>>::nearest_one(self)
    }

    /// Selects a k-nearest-neighbours query.
    #[inline]
    pub fn nearest_n<Dq>(
        self,
        max_qty: NonZeroUsize,
    ) -> <Self as NearestNQueryBuilder<A, K, Dq>>::Output
    where
        A: Copy,
        Dq: DistanceMetric<A>,
        Self: NearestNQueryBuilder<A, K, Dq>,
    {
        <Self as NearestNQueryBuilder<A, K, Dq>>::nearest_n(self, max_qty)
    }

    /// Selects a radius query or adds a radius bound to a k-nearest-neighbours query.
    #[inline]
    pub fn within<Dq>(self, radius: Dq::Output) -> <Self as WithinQueryBuilder<A, K, Dq>>::Output
    where
        A: Copy,
        Dq: DistanceMetric<A>,
        Self: WithinQueryBuilder<A, K, Dq>,
    {
        <Self as WithinQueryBuilder<A, K, Dq>>::within(self, radius)
    }

    /// Selects a radius query that keeps the best `max_qty` entries by item ordering.
    #[inline]
    pub fn best_n_within<Dq>(
        self,
        radius: Dq::Output,
        max_qty: NonZero<usize>,
    ) -> <Self as BestNWithinQueryBuilder<A, K, Dq>>::Output
    where
        A: Copy,
        Dq: DistanceMetric<A>,
        Self: BestNWithinQueryBuilder<A, K, Dq>,
    {
        <Self as BestNWithinQueryBuilder<A, K, Dq>>::best_n_within(self, radius, max_qty)
    }

    /// Switches exact nearest-neighbour search to approximate descent-only mode.
    #[inline]
    pub fn approx(self) -> <Self as ApproxQueryBuilder>::Output
    where
        Self: ApproxQueryBuilder,
    {
        <Self as ApproxQueryBuilder>::approx(self)
    }

    /// Returns results in traversal order instead of sorted-by-distance order.
    #[inline]
    pub fn unsorted(self) -> <Self as UnsortedQueryBuilder>::Output
    where
        Self: UnsortedQueryBuilder,
    {
        <Self as UnsortedQueryBuilder>::unsorted(self)
    }

    /// Uses strict `< radius` semantics instead of inclusive `<= radius`.
    #[inline]
    pub fn exclusive_boundaries(self) -> <Self as ExclusiveBoundariesQueryBuilder>::Output
    where
        Self: ExclusiveBoundariesQueryBuilder,
    {
        <Self as ExclusiveBoundariesQueryBuilder>::exclusive_boundaries(self)
    }
}

enum WithinUnsortedBuilderIter<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize>
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: KdTreeAccessor<A, T, SS, LS, K, B>,
    D: QueryMetric<A>,
    D::Output: crate::stem_strategy::SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static,
    SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
{
    Inclusive(WithinUnsortedIter<'a, Tree, A, T, SS, LS, D, false, K, B>),
    Exclusive(WithinUnsortedIter<'a, Tree, A, T, SS, LS, D, true, K, B>),
}

impl<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize> Iterator
    for WithinUnsortedBuilderIter<'a, Tree, A, T, SS, LS, D, K, B>
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: KdTreeAccessor<A, T, SS, LS, K, B>,
    D: DistanceMetric<A>,
    D::Output: crate::stem_strategy::SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static,
    SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
{
    type Item = QueryResultItem<(), T, D::Output>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Inclusive(iter) => iter.next(),
            Self::Exclusive(iter) => iter.next(),
        }
    }
}

#[inline(always)]
pub(crate) fn boundary_accepts<O: Axis<Coord = O>>(
    boundary: BoundaryMode,
    distance: O,
    radius: O,
) -> bool {
    match boundary {
        BoundaryMode::Inclusive => O::cmp(distance, radius) != Ordering::Greater,
        BoundaryMode::Exclusive => O::cmp(distance, radius) == Ordering::Less,
    }
}

#[inline(always)]
fn with_cleared_scratch<SS, O, R>(
    scratch: &mut QueryScratch<SS, O>,
    f: impl FnOnce(&mut SS::Stack<O>) -> R,
) -> R
where
    SS: StemStrategy,
{
    scratch.clear();
    f(scratch.stack_mut())
}

#[inline(always)]
fn map_full_nearest<A, T, O, Pj, const K: usize>(
    candidate: FullNearest<A, T, O, K>,
) -> Pj::NearestOut
where
    Pj: ProjectionSpec<A, T, O, K>,
{
    Pj::nearest_from_parts(candidate.point, candidate.item, candidate.distance)
}

#[inline(always)]
fn map_full_best<A, T, O, Pj, const K: usize>(candidate: FullBest<A, T, O, K>) -> Pj::BestOut
where
    Pj: ProjectionSpec<A, T, O, K>,
{
    Pj::best_from_parts(candidate.point, candidate.item, candidate.distance)
}

fn scan_projected_nearest_one<Tree, A, T, SS, LS, D, Pj, const K: usize, const B: usize>(
    tree: &Tree,
    query: &[A; K],
) -> Pj::NearestOut
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: KdTreeAccessor<A, T, SS, LS, K, B>,
    D: DistanceMetric<A>,
    D::Output: Axis<Coord = D::Output>,
    Pj: ProjectionSpec<A, T, D::Output, K>,
{
    let query_wide = query.map(D::widen_coord);
    let mut best: Option<FullNearest<A, T, D::Output, K>> = None;

    for (item, point) in KdTreeIter::<Tree, A, T, SS, LS, K, B>::new(tree) {
        let point_wide = point.map(D::widen_coord);
        let distance = D::dist(&point_wide, &query_wide);
        let candidate = FullNearest {
            point,
            item,
            distance,
        };
        if best.as_ref().is_none_or(|current| candidate < *current) {
            best = Some(candidate);
        }
    }

    let best = best.unwrap_or(FullNearest {
        point: [A::zero(); K],
        item: T::default(),
        distance: D::Output::max_value(),
    });

    map_full_nearest::<A, T, D::Output, Pj, K>(best)
}

fn scan_projected_nearest_results<Tree, A, T, SS, LS, D, Pj, const K: usize, const B: usize>(
    tree: &Tree,
    query: &[A; K],
    radius: Option<D::Output>,
    max_qty: Option<usize>,
    sorted: bool,
    result_capacity: Option<NonZeroUsize>,
    boundary: BoundaryMode,
) -> Vec<Pj::NearestOut>
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: KdTreeAccessor<A, T, SS, LS, K, B>,
    D: DistanceMetric<A>,
    D::Output: Axis<Coord = D::Output>,
    Pj: ProjectionSpec<A, T, D::Output, K>,
{
    let query_wide = query.map(D::widen_coord);
    let mut results =
        result_capacity.map_or_else(Vec::new, |capacity| Vec::with_capacity(capacity.get()));

    for (item, point) in KdTreeIter::<Tree, A, T, SS, LS, K, B>::new(tree) {
        let point_wide = point.map(D::widen_coord);
        let distance = D::dist(&point_wide, &query_wide);
        if radius.is_none_or(|max_dist| boundary_accepts(boundary, distance, max_dist)) {
            results.push(FullNearest {
                point,
                item,
                distance,
            });
        }
    }

    if sorted || max_qty.is_some() {
        results.sort_unstable();
        if let Some(max_qty) = max_qty {
            results.truncate(max_qty);
        }
    }

    results
        .into_iter()
        .map(map_full_nearest::<A, T, D::Output, Pj, K>)
        .collect()
}

fn scan_projected_best_results<Tree, A, T, SS, LS, D, Pj, const K: usize, const B: usize>(
    tree: &Tree,
    query: &[A; K],
    radius: D::Output,
    max_qty: usize,
    boundary: BoundaryMode,
) -> BinaryHeap<Pj::BestOut>
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: KdTreeAccessor<A, T, SS, LS, K, B>,
    D: DistanceMetric<A>,
    D::Output: Axis<Coord = D::Output>,
    Pj: ProjectionSpec<A, T, D::Output, K>,
    Pj::BestOut: Ord,
{
    let query_wide = query.map(D::widen_coord);
    let mut heap = BinaryHeap::<FullBest<A, T, D::Output, K>>::with_capacity(max_qty);

    for (item, point) in KdTreeIter::<Tree, A, T, SS, LS, K, B>::new(tree) {
        let point_wide = point.map(D::widen_coord);
        let distance = D::dist(&point_wide, &query_wide);
        if !boundary_accepts(boundary, distance, radius) {
            continue;
        }
        let candidate = FullBest {
            point,
            item,
            distance,
        };
        if heap.len() < max_qty {
            heap.push(candidate);
        } else if candidate < *heap.peek().unwrap() {
            *heap.peek_mut().unwrap() = candidate;
        }
    }

    heap.into_iter()
        .map(map_full_best::<A, T, D::Output, Pj, K>)
        .collect()
}

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: Axis<Coord = A>,
    T: Content,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
{
    /// Starts a fluent query against this tree.
    ///
    /// See [`QueryBuilder`] for the full query-building guide, including query
    /// families, option toggles, result projection, and terminal methods.
    #[inline]
    pub fn query<'a>(
        &'a self,
        query: &'a [A; K],
    ) -> RootQueryBuilder<'a, Self, A, T, SS, LS, K, B> {
        QueryBuilder {
            tree: self,
            query,
            box_size: None,
            max_qty: None,
            result_capacity: None,
            radius: (),
            _phantom: PhantomData,
        }
    }

    /// Creates reusable scratch storage for scratch-aware query builder paths.
    ///
    /// Pass the returned value to [`QueryBuilder::with_scratch`] to avoid the
    /// default thread-local stack lookup and reuse any spill allocation across
    /// repeated compatible queries.
    ///
    /// Scratch strategy selection is experimental during the 6.0.0 alpha; see
    /// the [`QueryBuilder` scratch strategy section](QueryBuilder#scratch-strategy).
    ///
    /// # Example
    ///
    /// ```
    /// use kiddo::dist::SquaredEuclidean;
    /// use kiddo::leaf_strategy::FlatVec;
    /// use kiddo::{Eytzinger, KdTree};
    /// use std::num::NonZeroUsize;
    ///
    /// type Tree = KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, 2, 4>, 2, 4>;
    ///
    /// let entries = [
    ///     (10_u32, [0.0, 0.0]),
    ///     (11_u32, [1.0, 1.0]),
    ///     (12_u32, [0.2, 0.1]),
    /// ];
    /// let tree = Tree::new_from_entries(&entries).unwrap();
    /// let mut scratch = tree.create_scratch::<SquaredEuclidean<f64>>();
    /// let max_qty = NonZeroUsize::new(2).unwrap();
    /// let queries = [[0.1, 0.1], [0.9, 0.9], [0.25, 0.15]];
    /// let mut nearest_items = Vec::new();
    ///
    /// for query in queries {
    ///     let results = tree
    ///         .query(&query)
    ///         .nearest_n::<SquaredEuclidean<f64>>(max_qty)
    ///         .with_scratch(&mut scratch)
    ///         .execute();
    ///
    ///     nearest_items.push(results[0].item);
    /// }
    ///
    /// assert_eq!(nearest_items, vec![12, 11, 12]);
    /// ```
    #[inline]
    pub fn create_scratch<D>(&self) -> QueryScratch<SS, D::Output>
    where
        D: DistanceMetric<A>,
    {
        QueryScratch::new()
    }
}

#[cfg(feature = "rkyv_08")]
impl<A, T, SS, LS, const K: usize, const B: usize> ArchivedKdTree<A, T, SS, LS, K, B>
where
    A: rkyv_08::Archive + Axis<Coord = A>,
    T: Content,
    SS: StemStrategy,
    LS: rkyv_08::Archive,
    rkyv_08::Archived<LS>: LeafStrategy<A, T, SS, K, B>,
{
    /// Starts a fluent query against this archived tree.
    ///
    /// See [`QueryBuilder`] for the full query-building guide, including query
    /// families, option toggles, result projection, and terminal methods.
    #[inline]
    pub fn query<'a>(
        &'a self,
        query: &'a [A; K],
    ) -> RootQueryBuilder<'a, Self, A, T, SS, rkyv_08::Archived<LS>, K, B> {
        QueryBuilder {
            tree: self,
            query,
            box_size: None,
            max_qty: None,
            result_capacity: None,
            radius: (),
            _phantom: PhantomData,
        }
    }

    /// Creates reusable scratch storage for scratch-aware query builder paths.
    ///
    /// Pass the returned value to [`QueryBuilder::with_scratch`] to avoid the
    /// default thread-local stack lookup and reuse any spill allocation across
    /// repeated compatible queries.
    ///
    /// Scratch strategy selection is experimental during the 6.0.0 alpha; see
    /// the [`QueryBuilder` scratch strategy section](QueryBuilder#scratch-strategy).
    #[inline]
    pub fn create_scratch<D>(&self) -> QueryScratch<SS, D::Output>
    where
        D: DistanceMetric<A>,
    {
        QueryScratch::new()
    }
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        Space,
        P,
        I,
        Dp,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    >
    QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        Space,
        Projection<P, I, Dp>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
where
    D: BuilderMetric<A, K>,
    Family: ResultFamily,
{
    /// Excludes stored point coordinates from the eventual query results.
    #[allow(clippy::type_complexity)]
    #[inline]
    fn without_points_impl(
        self,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        Space,
        Projection<Exclude, I, Dp>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    > {
        self.rebind::<D, Family, Space, Projection<Exclude, I, Dp>, SORTED, EXCLUSIVE>(
            self.box_size,
            self.max_qty,
            self.radius,
        )
    }

    /// Includes stored items in the eventual query results.
    #[allow(clippy::type_complexity)]
    #[inline]
    fn with_items_impl(
        self,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        Space,
        Projection<P, Include, Dp>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    > {
        self.rebind::<D, Family, Space, Projection<P, Include, Dp>, SORTED, EXCLUSIVE>(
            self.box_size,
            self.max_qty,
            self.radius,
        )
    }

    /// Excludes stored items from the eventual query results.
    #[allow(clippy::type_complexity)]
    #[inline]
    fn without_items_impl(
        self,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        Space,
        Projection<P, Exclude, Dp>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    > {
        self.rebind::<D, Family, Space, Projection<P, Exclude, Dp>, SORTED, EXCLUSIVE>(
            self.box_size,
            self.max_qty,
            self.radius,
        )
    }

    /// Includes distances in the eventual query results.
    #[allow(clippy::type_complexity)]
    #[inline]
    fn with_distances_impl(
        self,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        Space,
        Projection<P, I, Include>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    > {
        self.rebind::<D, Family, Space, Projection<P, I, Include>, SORTED, EXCLUSIVE>(
            self.box_size,
            self.max_qty,
            self.radius,
        )
    }

    /// Excludes distances from the eventual query results.
    #[allow(clippy::type_complexity)]
    #[inline]
    fn without_distances_impl(
        self,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        Space,
        Projection<P, I, Exclude>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    > {
        self.rebind::<D, Family, Space, Projection<P, I, Exclude>, SORTED, EXCLUSIVE>(
            self.box_size,
            self.max_qty,
            self.radius,
        )
    }
}

macro_rules! impl_projection_toggle_trait {
    ($trait:ident, $method:ident, $helper:ident, $out_p:ident, $out_i:ident, $out_d:ident) => {
        impl<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                D,
                Family,
                Space,
                P,
                I,
                Dp,
                const SORTED: bool,
                const EXCLUSIVE: bool,
                const K: usize,
                const B: usize,
            > $trait
            for QueryBuilder<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                D,
                Family,
                Space,
                Projection<P, I, Dp>,
                SORTED,
                EXCLUSIVE,
                K,
                B,
            >
        where
            D: BuilderMetric<A, K>,
            Family: ResultFamily,
        {
            type Output = QueryBuilder<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                D,
                Family,
                Space,
                Projection<$out_p, $out_i, $out_d>,
                SORTED,
                EXCLUSIVE,
                K,
                B,
            >;

            #[inline]
            fn $method(self) -> Self::Output {
                self.$helper()
            }
        }
    };
}

impl_projection_toggle_trait!(
    WithoutPointsQueryBuilder,
    without_points,
    without_points_impl,
    Exclude,
    I,
    Dp
);
impl_projection_toggle_trait!(
    WithItemsQueryBuilder,
    with_items,
    with_items_impl,
    P,
    Include,
    Dp
);
impl_projection_toggle_trait!(
    WithoutItemsQueryBuilder,
    without_items,
    without_items_impl,
    P,
    Exclude,
    Dp
);
impl_projection_toggle_trait!(
    WithDistancesQueryBuilder,
    with_distances,
    with_distances_impl,
    P,
    I,
    Include
);
impl_projection_toggle_trait!(
    WithoutDistancesQueryBuilder,
    without_distances,
    without_distances_impl,
    P,
    I,
    Exclude
);

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        P,
        I,
        Dp,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    >
    QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        CartesianSpace,
        Projection<P, I, Dp>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
where
    D: BuilderMetric<A, K>,
    Family: ResultFamily,
{
    /// Includes stored point coordinates in the eventual query results.
    ///
    /// This is available only for non-periodic queries.
    #[allow(clippy::type_complexity)]
    #[inline]
    fn with_points_impl(
        self,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        CartesianSpace,
        Projection<Include, I, Dp>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    > {
        self.rebind::<D, Family, CartesianSpace, Projection<Include, I, Dp>, SORTED, EXCLUSIVE>(
            self.box_size,
            self.max_qty,
            self.radius,
        )
    }
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        P,
        I,
        Dp,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > WithPointsQueryBuilder
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        CartesianSpace,
        Projection<P, I, Dp>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
where
    D: BuilderMetric<A, K>,
    Family: ResultFamily,
{
    type Output = QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        CartesianSpace,
        Projection<Include, I, Dp>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >;

    #[inline]
    fn with_points(self) -> Self::Output {
        self.with_points_impl()
    }
}

impl<
        'a,
        Tree,
        A: Copy,
        T,
        SS,
        LS,
        Pj,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    >
    QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        NoMetric,
        RootState,
        CartesianSpace,
        Pj,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
{
    /// Interprets subsequent queries using periodic boundary conditions.
    ///
    /// `box_size` gives the period along each axis and every value must be
    /// strictly positive.
    #[allow(clippy::type_complexity)]
    #[inline]
    fn periodic_boundary_condition_impl(
        self,
        box_size: &'a [A; K],
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        NoMetric,
        RootState,
        PeriodicSpace,
        Projection<Exclude, Include, Include>,
        true,
        false,
        K,
        B,
    >
    where
        A: PeriodicAxis,
    {
        self.rebind::<
            NoMetric,
            RootState,
            PeriodicSpace,
            Projection<Exclude, Include, Include>,
            true,
            false,
        >(Some(box_size), None, ())
    }

    /// Selects an exact nearest-neighbour query.
    #[allow(clippy::type_complexity)]
    #[inline]
    fn nearest_one_impl<D>(
        self,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestOneState,
        CartesianSpace,
        Projection<Exclude, Include, Include>,
        true,
        false,
        K,
        B,
    >
    where
        D: DistanceMetric<A>,
    {
        self.rebind::<
            SelectedMetric<D>,
            NearestOneState,
            CartesianSpace,
            Projection<Exclude, Include, Include>,
            true,
            false,
        >(None, None, D::Output::max_value())
    }

    /// Selects a k-nearest-neighbours query.
    #[allow(clippy::type_complexity)]
    #[inline]
    fn nearest_n_impl<D>(
        self,
        max_qty: NonZeroUsize,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestNState,
        CartesianSpace,
        Projection<Exclude, Include, Include>,
        true,
        false,
        K,
        B,
    >
    where
        D: DistanceMetric<A>,
    {
        self.rebind::<
            SelectedMetric<D>,
            NearestNState,
            CartesianSpace,
            Projection<Exclude, Include, Include>,
            true,
            false,
        >(None, Some(max_qty), D::Output::max_value())
    }

    /// Selects a radius query returning all entries within `radius`.
    #[allow(clippy::type_complexity)]
    #[inline]
    fn within_impl<D>(
        self,
        radius: D::Output,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        WithinState,
        CartesianSpace,
        Projection<Exclude, Include, Include>,
        true,
        false,
        K,
        B,
    >
    where
        D: DistanceMetric<A>,
    {
        self.rebind::<
            SelectedMetric<D>,
            WithinState,
            CartesianSpace,
            Projection<Exclude, Include, Include>,
            true,
            false,
        >(None, None, radius)
    }

    /// Selects a radius query that keeps the best `max_qty` entries by item ordering.
    #[allow(clippy::type_complexity)]
    #[inline]
    fn best_n_within_impl<D>(
        self,
        radius: D::Output,
        max_qty: NonZero<usize>,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        BestNWithinState,
        CartesianSpace,
        Projection<Exclude, Include, Include>,
        true,
        false,
        K,
        B,
    >
    where
        T: Content + PartialOrd + 'static,
        D: DistanceMetric<A>,
    {
        self.rebind::<
            SelectedMetric<D>,
            BestNWithinState,
            CartesianSpace,
            Projection<Exclude, Include, Include>,
            true,
            false,
        >(None, Some(max_qty), radius)
    }
}

macro_rules! impl_root_metric_selector_noarg {
    ($space:ty, [$($a_bounds:tt)+], $trait:ident, $method:ident, $helper:ident, $family:ty) => {
        impl<
                'a,
                Tree,
                A: $($a_bounds)+,
                T,
                SS,
                LS,
                Pj,
                const SORTED: bool,
                const EXCLUSIVE: bool,
                const K: usize,
                const B: usize,
                Dq,
            > $trait<A, K, Dq>
            for QueryBuilder<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                NoMetric,
                RootState,
                $space,
                Pj,
                SORTED,
                EXCLUSIVE,
                K,
                B,
            >
        where
            Dq: DistanceMetric<A>,
        {
            type Output = QueryBuilder<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                SelectedMetric<Dq>,
                $family,
                $space,
                Projection<Exclude, Include, Include>,
                true,
                false,
                K,
                B,
            >;

            #[inline]
            fn $method(self) -> Self::Output {
                self.$helper::<Dq>()
            }
        }
    };
}

macro_rules! impl_root_metric_selector_qty {
    ($space:ty, [$($a_bounds:tt)+], $trait:ident, $method:ident, $helper:ident, $family:ty) => {
        impl<
                'a,
                Tree,
                A: $($a_bounds)+,
                T,
                SS,
                LS,
                Pj,
                const SORTED: bool,
                const EXCLUSIVE: bool,
                const K: usize,
                const B: usize,
                Dq,
            > $trait<A, K, Dq>
            for QueryBuilder<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                NoMetric,
                RootState,
                $space,
                Pj,
                SORTED,
                EXCLUSIVE,
                K,
                B,
            >
        where
            Dq: DistanceMetric<A>,
        {
            type Output = QueryBuilder<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                SelectedMetric<Dq>,
                $family,
                $space,
                Projection<Exclude, Include, Include>,
                true,
                false,
                K,
                B,
            >;

            #[inline]
            fn $method(self, max_qty: NonZeroUsize) -> Self::Output {
                self.$helper::<Dq>(max_qty)
            }
        }
    };
}

macro_rules! impl_root_metric_selector_radius {
    ($space:ty, [$($a_bounds:tt)+], $trait:ident, $method:ident, $helper:ident, $family:ty) => {
        impl<
                'a,
                Tree,
                A: $($a_bounds)+,
                T,
                SS,
                LS,
                Pj,
                const SORTED: bool,
                const EXCLUSIVE: bool,
                const K: usize,
                const B: usize,
                Dq,
            > $trait<A, K, Dq>
            for QueryBuilder<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                NoMetric,
                RootState,
                $space,
                Pj,
                SORTED,
                EXCLUSIVE,
                K,
                B,
            >
        where
            Dq: DistanceMetric<A>,
        {
            type Output = QueryBuilder<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                SelectedMetric<Dq>,
                $family,
                $space,
                Projection<Exclude, Include, Include>,
                true,
                false,
                K,
                B,
            >;

            #[inline]
            fn $method(self, radius: Dq::Output) -> Self::Output {
                self.$helper::<Dq>(radius)
            }
        }
    };
}

macro_rules! impl_root_best_n_within_query_builder {
    () => {
        impl<
                'a,
                Tree,
                A: Copy,
                T,
                SS,
                LS,
                Pj,
                const SORTED: bool,
                const EXCLUSIVE: bool,
                const K: usize,
                const B: usize,
                Dq,
            > BestNWithinQueryBuilder<A, K, Dq>
            for QueryBuilder<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                NoMetric,
                RootState,
                CartesianSpace,
                Pj,
                SORTED,
                EXCLUSIVE,
                K,
                B,
            >
        where
            T: Content + PartialOrd + 'static,
            Dq: DistanceMetric<A>,
        {
            type Output = QueryBuilder<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                SelectedMetric<Dq>,
                BestNWithinState,
                CartesianSpace,
                Projection<Exclude, Include, Include>,
                true,
                false,
                K,
                B,
            >;

            #[inline]
            fn best_n_within(self, radius: Dq::Output, max_qty: NonZero<usize>) -> Self::Output {
                self.best_n_within_impl::<Dq>(radius, max_qty)
            }
        }
    };
}

impl<
        'a,
        Tree,
        A: Copy + PeriodicAxis,
        T,
        SS,
        LS,
        Pj,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > PeriodicBoundaryConditionQueryBuilder<'a, A, K>
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        NoMetric,
        RootState,
        CartesianSpace,
        Pj,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
{
    type Output = QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        NoMetric,
        RootState,
        PeriodicSpace,
        Projection<Exclude, Include, Include>,
        true,
        false,
        K,
        B,
    >;

    #[inline]
    fn periodic_boundary_condition(self, box_size: &'a [A; K]) -> Self::Output {
        self.periodic_boundary_condition_impl(box_size)
    }
}
impl_root_metric_selector_noarg!(
    CartesianSpace,
    [Copy],
    NearestOneQueryBuilder,
    nearest_one,
    nearest_one_impl,
    NearestOneState
);
impl_root_metric_selector_qty!(
    CartesianSpace,
    [Copy],
    NearestNQueryBuilder,
    nearest_n,
    nearest_n_impl,
    NearestNState
);
impl_root_metric_selector_radius!(
    CartesianSpace,
    [Copy],
    WithinQueryBuilder,
    within,
    within_impl,
    WithinState
);
impl_root_best_n_within_query_builder!();

impl<
        'a,
        Tree,
        A: Copy + PeriodicAxis,
        T,
        SS,
        LS,
        Pj,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    >
    QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        NoMetric,
        RootState,
        PeriodicSpace,
        Pj,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
{
    /// Selects an exact nearest-neighbour query under periodic boundary conditions.
    #[allow(clippy::type_complexity)]
    #[inline]
    fn nearest_one_impl<D>(
        self,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestOneState,
        PeriodicSpace,
        Projection<Exclude, Include, Include>,
        true,
        false,
        K,
        B,
    >
    where
        D: DistanceMetric<A>,
    {
        self.rebind::<
            SelectedMetric<D>,
            NearestOneState,
            PeriodicSpace,
            Projection<Exclude, Include, Include>,
            true,
            false,
        >(self.box_size, None, D::Output::max_value())
    }

    /// Selects a periodic k-nearest-neighbours query.
    #[allow(clippy::type_complexity)]
    #[inline]
    fn nearest_n_impl<D>(
        self,
        max_qty: NonZeroUsize,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestNState,
        PeriodicSpace,
        Projection<Exclude, Include, Include>,
        true,
        false,
        K,
        B,
    >
    where
        D: DistanceMetric<A>,
    {
        self.rebind::<
            SelectedMetric<D>,
            NearestNState,
            PeriodicSpace,
            Projection<Exclude, Include, Include>,
            true,
            false,
        >(self.box_size, Some(max_qty), D::Output::max_value())
    }

    /// Selects a periodic radius query returning all entries within `radius`.
    #[allow(clippy::type_complexity)]
    #[inline]
    fn within_impl<D>(
        self,
        radius: D::Output,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        WithinState,
        PeriodicSpace,
        Projection<Exclude, Include, Include>,
        true,
        false,
        K,
        B,
    >
    where
        D: DistanceMetric<A>,
    {
        self.rebind::<
            SelectedMetric<D>,
            WithinState,
            PeriodicSpace,
            Projection<Exclude, Include, Include>,
            true,
            false,
        >(self.box_size, None, radius)
    }
}

impl_root_metric_selector_noarg!(
    PeriodicSpace,
    [Copy + PeriodicAxis],
    NearestOneQueryBuilder,
    nearest_one,
    nearest_one_impl,
    NearestOneState
);
impl_root_metric_selector_qty!(
    PeriodicSpace,
    [Copy + PeriodicAxis],
    NearestNQueryBuilder,
    nearest_n,
    nearest_n_impl,
    NearestNState
);
impl_root_metric_selector_radius!(
    PeriodicSpace,
    [Copy + PeriodicAxis],
    WithinQueryBuilder,
    within,
    within_impl,
    WithinState
);

impl<
        'a,
        Tree,
        A: Copy,
        T,
        SS,
        LS,
        D,
        Space,
        Pj,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    >
    QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestOneState,
        Space,
        Pj,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
where
    D: DistanceMetric<A>,
{
    /// Switches exact nearest-neighbour search to approximate descent-only mode.
    #[inline]
    fn approx_impl(
        self,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        ApproxNearestOneState,
        CartesianSpace,
        Pj,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    > {
        self.rebind::<SelectedMetric<D>, ApproxNearestOneState, CartesianSpace, Pj, SORTED, EXCLUSIVE>(
            None,
            self.max_qty,
            self.radius,
        )
    }
}

impl<
        'a,
        Tree,
        A: Copy,
        T,
        SS,
        LS,
        D,
        Space,
        Pj,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > ApproxQueryBuilder
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestOneState,
        Space,
        Pj,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
where
    D: DistanceMetric<A>,
{
    type Output = QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        ApproxNearestOneState,
        CartesianSpace,
        Pj,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >;

    #[inline]
    fn approx(self) -> Self::Output {
        self.approx_impl()
    }
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        I,
        Dp,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > ExecuteQueryBuilderWithScratch<SS, D::Output>
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestOneState,
        CartesianSpace,
        Projection<Exclude, I, Dp>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderScratchTreeOps<A, T, SS, LS, K, B>,
    D: QueryMetric<A>,
    SS::Stack<D::Output>: StackTrait<D::Output, SS>,
    I: ProjectionField<T>,
    Dp: ProjectionField<D::Output>,
{
    type Output = QueryResultItem<(), I::Output, Dp::Output>;

    #[inline]
    fn execute_with_scratch(self, scratch: &mut QueryScratch<SS, D::Output>) -> Self::Output {
        let (distance, item) = with_cleared_scratch(scratch, |stack| {
            self.tree
                .qb_nearest_one_with_scratch::<D>(self.query, stack)
        });
        project_nearest_without_point_from_parts::<A, T, D::Output, Exclude, I, Dp, K>(
            item, distance,
        )
    }
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        I,
        Dp,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > ExecuteQueryBuilderWithScratch<SS, D::Output>
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestNState,
        CartesianSpace,
        Projection<Exclude, I, Dp>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderScratchTreeOps<A, T, SS, LS, K, B>,
    D: QueryMetric<A>,
    SS::Stack<D::Output>: StackTrait<D::Output, SS>,
    I: ProjectionField<T>,
    Dp: ProjectionField<D::Output>,
{
    type Output = Vec<QueryResultItem<(), I::Output, Dp::Output>>;

    #[inline]
    fn execute_with_scratch(self, scratch: &mut QueryScratch<SS, D::Output>) -> Self::Output {
        with_cleared_scratch(scratch, |stack| {
            self.tree
                .qb_nearest_n_with_scratch::<D>(
                    self.query,
                    self.max_qty.expect("max_qty missing"),
                    SORTED,
                    stack,
                )
                .into_iter()
                .map(project_nearest_without_point::<A, T, D::Output, Exclude, I, Dp, K>)
                .collect()
        })
    }
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        I,
        Dp,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > ExecuteQueryBuilderWithScratch<SS, D::Output>
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestNWithinState,
        CartesianSpace,
        Projection<Exclude, I, Dp>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderScratchTreeOps<A, T, SS, LS, K, B>,
    D: QueryMetric<A>,
    SS::Stack<D::Output>: StackTrait<D::Output, SS>,
    I: ProjectionField<T>,
    Dp: ProjectionField<D::Output>,
{
    type Output = Vec<QueryResultItem<(), I::Output, Dp::Output>>;

    #[inline]
    fn execute_with_scratch(self, scratch: &mut QueryScratch<SS, D::Output>) -> Self::Output {
        with_cleared_scratch(scratch, |stack| {
            self.tree
                .qb_nearest_n_within_with_scratch::<D, EXCLUSIVE>(
                    self.query,
                    self.radius,
                    self.max_qty.expect("max_qty missing"),
                    SORTED,
                    stack,
                )
                .into_iter()
                .map(project_nearest_without_point::<A, T, D::Output, Exclude, I, Dp, K>)
                .collect()
        })
    }
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        I,
        Dp,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > ExecuteQueryBuilderWithScratch<SS, D::Output>
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        WithinState,
        CartesianSpace,
        Projection<Exclude, I, Dp>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderScratchTreeOps<A, T, SS, LS, K, B>,
    D: QueryMetric<A>,
    SS::Stack<D::Output>: StackTrait<D::Output, SS>,
    I: ProjectionField<T>,
    Dp: ProjectionField<D::Output>,
{
    type Output = Vec<QueryResultItem<(), I::Output, Dp::Output>>;

    #[inline]
    fn execute_with_scratch(self, scratch: &mut QueryScratch<SS, D::Output>) -> Self::Output {
        with_cleared_scratch(scratch, |stack| {
            if SORTED {
                self.tree
                    .qb_within_with_scratch::<D, EXCLUSIVE>(
                        self.query,
                        self.radius,
                        self.result_capacity,
                        stack,
                    )
                    .into_iter()
                    .map(project_nearest_without_point::<A, T, D::Output, Exclude, I, Dp, K>)
                    .collect()
            } else {
                self.tree
                    .qb_within_unsorted_projected_with_scratch::<D, _, _, EXCLUSIVE>(
                        self.query,
                        self.radius,
                        self.result_capacity,
                        project_nearest_without_point::<A, T, D::Output, Exclude, I, Dp, K>,
                        stack,
                    )
            }
        })
    }
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        I,
        Dp,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > ExecuteQueryBuilderWithScratch<SS, D::Output>
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        BestNWithinState,
        CartesianSpace,
        Projection<Exclude, I, Dp>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderScratchTreeOps<A, T, SS, LS, K, B>,
    D: DistanceMetric<A>,
    D::Output: crate::stem_strategy::SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static,
    SS::Stack<D::Output>: StackTrait<D::Output, SS>,
    I: ProjectionField<T>,
    Dp: ProjectionField<D::Output>,
    BestQueryResultItem<(), I::Output, Dp::Output>: Ord,
{
    type Output = BinaryHeap<BestQueryResultItem<(), I::Output, Dp::Output>>;

    #[inline]
    fn execute_with_scratch(self, scratch: &mut QueryScratch<SS, D::Output>) -> Self::Output {
        with_cleared_scratch(scratch, |stack| {
            self.tree
                .qb_best_n_within_with_scratch::<D, EXCLUSIVE>(
                    self.query,
                    self.radius,
                    self.max_qty.expect("max_qty missing"),
                    stack,
                )
                .into_iter()
                .map(project_best_without_point::<A, T, D::Output, Exclude, I, Dp, K>)
                .collect()
        })
    }
}

impl<
        'a,
        Tree,
        A: Copy,
        T,
        SS,
        LS,
        D,
        Space,
        Pj,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    >
    QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestNState,
        Space,
        Pj,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
where
    D: DistanceMetric<A>,
{
    /// Adds a radius bound to a k-nearest-neighbours query.
    #[inline]
    fn within_impl(
        self,
        radius: D::Output,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestNWithinState,
        Space,
        Pj,
        SORTED,
        false,
        K,
        B,
    > {
        self.rebind::<SelectedMetric<D>, NearestNWithinState, Space, Pj, SORTED, false>(
            self.box_size,
            self.max_qty,
            radius,
        )
    }
}

impl<
        'a,
        Tree,
        A: Copy,
        T,
        SS,
        LS,
        D,
        Space,
        Pj,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > WithinQueryBuilder<A, K, D>
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestNState,
        Space,
        Pj,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
where
    D: DistanceMetric<A>,
{
    type Output = QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestNWithinState,
        Space,
        Pj,
        SORTED,
        false,
        K,
        B,
    >;

    #[inline]
    fn within(self, radius: D::Output) -> Self::Output {
        self.within_impl(radius)
    }
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Space,
        Pj,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    >
    QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestNState,
        Space,
        Pj,
        true,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: Copy,
    D: DistanceMetricCore<A>,
{
    /// Returns results in traversal order instead of sorted-by-distance order.
    #[inline]
    fn unsorted_impl(
        self,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestNState,
        Space,
        Pj,
        false,
        EXCLUSIVE,
        K,
        B,
    > {
        self.rebind::<SelectedMetric<D>, NearestNState, Space, Pj, false, EXCLUSIVE>(
            self.box_size,
            self.max_qty,
            self.radius,
        )
    }
}

macro_rules! impl_unsorted_query_builder {
    ($family:ty) => {
        impl<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                D,
                Space,
                Pj,
                const EXCLUSIVE: bool,
                const K: usize,
                const B: usize,
            > UnsortedQueryBuilder
            for QueryBuilder<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                SelectedMetric<D>,
                $family,
                Space,
                Pj,
                true,
                EXCLUSIVE,
                K,
                B,
            >
        where
            A: Copy,
            D: DistanceMetricCore<A>,
        {
            type Output = QueryBuilder<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                SelectedMetric<D>,
                $family,
                Space,
                Pj,
                false,
                EXCLUSIVE,
                K,
                B,
            >;

            #[inline]
            fn unsorted(self) -> Self::Output {
                self.unsorted_impl()
            }
        }
    };
}

impl_unsorted_query_builder!(NearestNState);

macro_rules! impl_exclusive_boundaries {
    ($family:ty, $space:ty) => {
        impl<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                D,
                Pj,
                const SORTED: bool,
                const EXCLUSIVE: bool,
                const K: usize,
                const B: usize,
            > ExclusiveBoundariesQueryBuilder
            for QueryBuilder<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                D,
                $family,
                $space,
                Pj,
                SORTED,
                EXCLUSIVE,
                K,
                B,
            >
        where
            D: BuilderMetric<A, K>,
        {
            type Output =
                QueryBuilder<'a, Tree, A, T, SS, LS, D, $family, $space, Pj, SORTED, true, K, B>;

            #[inline]
            fn exclusive_boundaries(self) -> Self::Output {
                self.rebind::<D, $family, $space, Pj, SORTED, true>(
                    self.box_size,
                    self.max_qty,
                    self.radius,
                )
            }
        }
    };
}

impl_exclusive_boundaries!(NearestNWithinState, CartesianSpace);
impl_exclusive_boundaries!(NearestNWithinState, PeriodicSpace);
impl_exclusive_boundaries!(WithinState, CartesianSpace);
impl_exclusive_boundaries!(WithinState, PeriodicSpace);
impl_exclusive_boundaries!(BestNWithinState, CartesianSpace);

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Space,
        Pj,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    >
    QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestNWithinState,
        Space,
        Pj,
        true,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: Copy,
    D: DistanceMetricCore<A>,
{
    /// Returns results in traversal order instead of sorted-by-distance order.
    #[inline]
    fn unsorted_impl(
        self,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestNWithinState,
        Space,
        Pj,
        false,
        EXCLUSIVE,
        K,
        B,
    > {
        self.rebind::<SelectedMetric<D>, NearestNWithinState, Space, Pj, false, EXCLUSIVE>(
            self.box_size,
            self.max_qty,
            self.radius,
        )
    }
}

impl_unsorted_query_builder!(NearestNWithinState);

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Space,
        Pj,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    >
    QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        WithinState,
        Space,
        Pj,
        true,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: Copy,
    D: DistanceMetricCore<A>,
{
    /// Returns results in traversal order instead of sorted-by-distance order.
    #[inline]
    fn unsorted_impl(
        self,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        WithinState,
        Space,
        Pj,
        false,
        EXCLUSIVE,
        K,
        B,
    > {
        self.rebind::<SelectedMetric<D>, WithinState, Space, Pj, false, EXCLUSIVE>(
            self.box_size,
            self.max_qty,
            self.radius,
        )
    }
}

impl_unsorted_query_builder!(WithinState);

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        P,
        I,
        Dp,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > ExecuteQueryBuilder
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestOneState,
        CartesianSpace,
        Projection<P, I, Dp>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy + 'static,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B>,
    D: QueryMetric<A>,
    SS::Stack<D::Output>: StackTrait<D::Output, SS> + Default + 'static,
    P: PointProjectionField<A, K>,
    I: ProjectionField<T>,
    Dp: ProjectionField<D::Output>,
{
    type Output = QueryResultItem<P::Output, I::Output, Dp::Output>;

    #[inline]
    fn execute(self) -> QueryResultItem<P::Output, I::Output, Dp::Output> {
        if <Projection<P, I, Dp> as ProjectionSpec<A, T, D::Output, K>>::WANTS_POINTS {
            scan_projected_nearest_one::<Tree, A, T, SS, LS, D, Projection<P, I, Dp>, K, B>(
                self.tree, self.query,
            )
        } else {
            let (distance, item) = self.tree.qb_nearest_one::<D>(self.query);
            project_nearest_without_point_from_parts::<A, T, D::Output, P, I, Dp, K>(item, distance)
        }
    }
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        P,
        I,
        Dp,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > ExecuteQueryBuilder
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        ApproxNearestOneState,
        CartesianSpace,
        Projection<P, I, Dp>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd + PartialEq,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B> + KdTreeAccessor<A, T, SS, LS, K, B>,
    D: DistanceMetric<A, Output = A>,
    P: PointProjectionField<A, K>,
    I: ProjectionField<T>,
    Dp: ProjectionField<A>,
{
    type Output = QueryResultItem<P::Output, I::Output, Dp::Output>;

    #[inline]
    fn execute(self) -> QueryResultItem<P::Output, I::Output, Dp::Output> {
        if <Projection<P, I, Dp> as ProjectionSpec<A, T, A, K>>::WANTS_POINTS {
            scan_projected_nearest_one::<Tree, A, T, SS, LS, D, Projection<P, I, Dp>, K, B>(
                self.tree, self.query,
            )
        } else {
            let (distance, item) = self.tree.qb_approx_nearest_one::<D>(self.query);
            project_nearest_without_point_from_parts::<A, T, A, P, I, Dp, K>(item, distance)
        }
    }
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        P,
        I,
        Dp,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > ExecuteQueryBuilder
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestOneState,
        PeriodicSpace,
        Projection<P, I, Dp>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: PeriodicAxis + 'static,
    T: Content,
    SS: StemStrategy + 'static,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B>,
    D: DistanceMetric<A>,
    D::Output: crate::stem_strategy::SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static,
    SS::Stack<D::Output>: StackTrait<D::Output, SS> + Default + 'static,
    P: PointProjectionField<A, K>,
    I: ProjectionField<T>,
    Dp: ProjectionField<D::Output>,
{
    type Output = QueryResultItem<P::Output, I::Output, Dp::Output>;

    #[inline]
    fn execute(self) -> QueryResultItem<P::Output, I::Output, Dp::Output> {
        let result = periodic_nearest_one_result::<Tree, A, T, SS, LS, D, K, B>(
            self.tree,
            self.query,
            self.box_size.expect("periodic builder missing box size"),
        );
        project_nearest_without_point::<A, T, D::Output, P, I, Dp, K>(result)
    }
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        P,
        I,
        Dp,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > ExecuteQueryBuilder
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestNState,
        CartesianSpace,
        Projection<P, I, Dp>,
        true,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B> + KdTreeAccessor<A, T, SS, LS, K, B>,
    D: DistanceMetric<A>,
    D::Output: crate::stem_strategy::SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static,
    SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    P: PointProjectionField<A, K>,
    I: ProjectionField<T>,
    Dp: ProjectionField<D::Output>,
{
    type Output = Vec<QueryResultItem<P::Output, I::Output, Dp::Output>>;

    #[inline]
    fn execute(self) -> Vec<QueryResultItem<P::Output, I::Output, Dp::Output>> {
        if <Projection<P, I, Dp> as ProjectionSpec<A, T, D::Output, K>>::WANTS_POINTS {
            scan_projected_nearest_results::<Tree, A, T, SS, LS, D, Projection<P, I, Dp>, K, B>(
                self.tree,
                self.query,
                None,
                self.max_qty.map(NonZeroUsize::get),
                true,
                None,
                if EXCLUSIVE {
                    BoundaryMode::Exclusive
                } else {
                    BoundaryMode::Inclusive
                },
            )
        } else {
            self.tree
                .qb_nearest_n::<D>(self.query, self.max_qty.expect("max_qty missing"), true)
                .into_iter()
                .map(project_nearest_without_point::<A, T, D::Output, P, I, Dp, K>)
                .collect()
        }
    }
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        P,
        I,
        Dp,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > ExecuteQueryBuilder
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestNState,
        CartesianSpace,
        Projection<P, I, Dp>,
        false,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B> + KdTreeAccessor<A, T, SS, LS, K, B>,
    D: DistanceMetric<A>,
    D::Output: crate::stem_strategy::SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static,
    SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    P: PointProjectionField<A, K>,
    I: ProjectionField<T>,
    Dp: ProjectionField<D::Output>,
{
    type Output = Vec<QueryResultItem<P::Output, I::Output, Dp::Output>>;

    #[inline]
    fn execute(self) -> Vec<QueryResultItem<P::Output, I::Output, Dp::Output>> {
        if <Projection<P, I, Dp> as ProjectionSpec<A, T, D::Output, K>>::WANTS_POINTS {
            scan_projected_nearest_results::<Tree, A, T, SS, LS, D, Projection<P, I, Dp>, K, B>(
                self.tree,
                self.query,
                None,
                self.max_qty.map(NonZeroUsize::get),
                false,
                None,
                if EXCLUSIVE {
                    BoundaryMode::Exclusive
                } else {
                    BoundaryMode::Inclusive
                },
            )
        } else {
            self.tree
                .qb_nearest_n::<D>(self.query, self.max_qty.expect("max_qty missing"), false)
                .into_iter()
                .map(project_nearest_without_point::<A, T, D::Output, P, I, Dp, K>)
                .collect()
        }
    }
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        P,
        I,
        Dp,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > ExecuteQueryBuilder
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestNWithinState,
        CartesianSpace,
        Projection<P, I, Dp>,
        true,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B> + KdTreeAccessor<A, T, SS, LS, K, B>,
    D: DistanceMetric<A>,
    D::Output: crate::stem_strategy::SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static,
    SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    P: PointProjectionField<A, K>,
    I: ProjectionField<T>,
    Dp: ProjectionField<D::Output>,
{
    type Output = Vec<QueryResultItem<P::Output, I::Output, Dp::Output>>;

    #[inline]
    fn execute(self) -> Vec<QueryResultItem<P::Output, I::Output, Dp::Output>> {
        if <Projection<P, I, Dp> as ProjectionSpec<A, T, D::Output, K>>::WANTS_POINTS {
            scan_projected_nearest_results::<Tree, A, T, SS, LS, D, Projection<P, I, Dp>, K, B>(
                self.tree,
                self.query,
                Some(self.radius),
                self.max_qty.map(NonZeroUsize::get),
                true,
                None,
                if EXCLUSIVE {
                    BoundaryMode::Exclusive
                } else {
                    BoundaryMode::Inclusive
                },
            )
        } else {
            let results = if EXCLUSIVE {
                self.tree.qb_nearest_n_within::<D, true>(
                    self.query,
                    self.radius,
                    self.max_qty.expect("max_qty missing"),
                    true,
                )
            } else {
                self.tree.qb_nearest_n_within::<D, false>(
                    self.query,
                    self.radius,
                    self.max_qty.expect("max_qty missing"),
                    true,
                )
            };
            results
                .into_iter()
                .map(project_nearest_without_point::<A, T, D::Output, P, I, Dp, K>)
                .collect()
        }
    }
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        P,
        I,
        Dp,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > ExecuteQueryBuilder
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestNWithinState,
        CartesianSpace,
        Projection<P, I, Dp>,
        false,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B> + KdTreeAccessor<A, T, SS, LS, K, B>,
    D: DistanceMetric<A>,
    D::Output: crate::stem_strategy::SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static,
    SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    P: PointProjectionField<A, K>,
    I: ProjectionField<T>,
    Dp: ProjectionField<D::Output>,
{
    type Output = Vec<QueryResultItem<P::Output, I::Output, Dp::Output>>;

    #[inline]
    fn execute(self) -> Vec<QueryResultItem<P::Output, I::Output, Dp::Output>> {
        if <Projection<P, I, Dp> as ProjectionSpec<A, T, D::Output, K>>::WANTS_POINTS {
            scan_projected_nearest_results::<Tree, A, T, SS, LS, D, Projection<P, I, Dp>, K, B>(
                self.tree,
                self.query,
                Some(self.radius),
                self.max_qty.map(NonZeroUsize::get),
                false,
                None,
                if EXCLUSIVE {
                    BoundaryMode::Exclusive
                } else {
                    BoundaryMode::Inclusive
                },
            )
        } else {
            let results = if EXCLUSIVE {
                self.tree.qb_nearest_n_within::<D, true>(
                    self.query,
                    self.radius,
                    self.max_qty.expect("max_qty missing"),
                    false,
                )
            } else {
                self.tree.qb_nearest_n_within::<D, false>(
                    self.query,
                    self.radius,
                    self.max_qty.expect("max_qty missing"),
                    false,
                )
            };
            results
                .into_iter()
                .map(project_nearest_without_point::<A, T, D::Output, P, I, Dp, K>)
                .collect()
        }
    }
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        P,
        I,
        Dp,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > ExecuteQueryBuilder
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        WithinState,
        CartesianSpace,
        Projection<P, I, Dp>,
        true,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B> + KdTreeAccessor<A, T, SS, LS, K, B>,
    D: DistanceMetric<A>,
    D::Output: crate::stem_strategy::SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static,
    SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    P: PointProjectionField<A, K>,
    I: ProjectionField<T>,
    Dp: ProjectionField<D::Output>,
{
    type Output = Vec<QueryResultItem<P::Output, I::Output, Dp::Output>>;

    #[inline]
    fn execute(self) -> Vec<QueryResultItem<P::Output, I::Output, Dp::Output>> {
        if <Projection<P, I, Dp> as ProjectionSpec<A, T, D::Output, K>>::WANTS_POINTS {
            scan_projected_nearest_results::<Tree, A, T, SS, LS, D, Projection<P, I, Dp>, K, B>(
                self.tree,
                self.query,
                Some(self.radius),
                None,
                true,
                self.result_capacity,
                if EXCLUSIVE {
                    BoundaryMode::Exclusive
                } else {
                    BoundaryMode::Inclusive
                },
            )
        } else {
            let results = if EXCLUSIVE {
                self.tree
                    .qb_within::<D, true>(self.query, self.radius, self.result_capacity)
            } else {
                self.tree
                    .qb_within::<D, false>(self.query, self.radius, self.result_capacity)
            };
            results
                .into_iter()
                .map(project_nearest_without_point::<A, T, D::Output, P, I, Dp, K>)
                .collect()
        }
    }
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        P,
        I,
        Dp,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > ExecuteQueryBuilder
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        WithinState,
        CartesianSpace,
        Projection<P, I, Dp>,
        false,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B> + KdTreeAccessor<A, T, SS, LS, K, B>,
    D: DistanceMetric<A>,
    D::Output: crate::stem_strategy::SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static,
    SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    P: PointProjectionField<A, K>,
    I: ProjectionField<T>,
    Dp: ProjectionField<D::Output>,
{
    type Output = Vec<QueryResultItem<P::Output, I::Output, Dp::Output>>;

    #[inline]
    fn execute(self) -> Vec<QueryResultItem<P::Output, I::Output, Dp::Output>> {
        if <Projection<P, I, Dp> as ProjectionSpec<A, T, D::Output, K>>::WANTS_POINTS {
            self.tree
                .qb_within_unsorted_projected_with_points::<D, _, _, EXCLUSIVE>(
                    self.query,
                    self.radius,
                    self.result_capacity,
                    Projection::<P, I, Dp>::nearest_from_parts,
                )
        } else {
            self.tree
                .qb_within_unsorted_projected::<D, _, _, EXCLUSIVE>(
                    self.query,
                    self.radius,
                    self.result_capacity,
                    project_nearest_without_point::<A, T, D::Output, P, I, Dp, K>,
                )
        }
    }
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        P,
        I,
        Dp,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    >
    QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        WithinState,
        CartesianSpace,
        Projection<P, I, Dp>,
        false,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B> + 'a,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B> + KdTreeAccessor<A, T, SS, LS, K, B> + 'a,
    D: DistanceMetric<A> + 'a,
    D::Output: crate::stem_strategy::SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static,
    SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    P: PointProjectionField<A, K> + 'a,
    I: ProjectionField<T> + 'a,
    Dp: ProjectionField<D::Output> + 'a,
{
    /// Visits each matching entry without materializing a result vector.
    #[inline]
    pub fn visit<F>(self, mut visitor: F)
    where
        F: FnMut(QueryResultItem<P::Output, I::Output, Dp::Output>),
    {
        if <Projection<P, I, Dp> as ProjectionSpec<A, T, D::Output, K>>::WANTS_POINTS {
            self.tree
                .qb_within_unsorted_visit_with_points::<D, _, EXCLUSIVE>(
                    self.query,
                    self.radius,
                    |result| {
                        visitor(Projection::<P, I, Dp>::nearest_from_parts(
                            result.point,
                            result.item,
                            result.distance,
                        ));
                    },
                );
        } else if EXCLUSIVE {
            self.tree.qb_within_unsorted_visit::<D, _, true>(
                self.query,
                self.radius,
                move |result| {
                    visitor(project_nearest_without_point::<A, T, D::Output, P, I, Dp, K>(result))
                },
            );
        } else {
            self.tree.qb_within_unsorted_visit::<D, _, false>(
                self.query,
                self.radius,
                move |result| {
                    visitor(project_nearest_without_point::<A, T, D::Output, P, I, Dp, K>(result))
                },
            );
        }
    }

    /// Returns an iterator over matching entries without first materializing a result vector.
    ///
    /// Scratch strategy selectors are not applicable to this iterator path. The
    /// iterator owns a traversal state machine rather than borrowing one of the
    /// query builder scratch strategies. See the
    /// [`QueryBuilder` scratch strategy section](QueryBuilder#scratch-strategy)
    /// for more detail.
    #[allow(clippy::type_complexity)]
    #[inline]
    pub fn iter(
        self,
    ) -> Box<dyn Iterator<Item = QueryResultItem<P::Output, I::Output, Dp::Output>> + 'a> {
        if <Projection<P, I, Dp> as ProjectionSpec<A, T, D::Output, K>>::WANTS_POINTS {
            let query_wide = self.query.map(D::widen_coord);
            let radius = self.radius;
            let boundary = if EXCLUSIVE {
                BoundaryMode::Exclusive
            } else {
                BoundaryMode::Inclusive
            };
            Box::new(
                KdTreeIter::<Tree, A, T, SS, LS, K, B>::new(self.tree).filter_map(
                    move |(item, point)| {
                        let point_wide = point.map(D::widen_coord);
                        let distance = D::dist(&point_wide, &query_wide);
                        boundary_accepts(boundary, distance, radius).then(|| {
                            Projection::<P, I, Dp>::nearest_from_parts(point, item, distance)
                        })
                    },
                ),
            )
        } else {
            let iter = if EXCLUSIVE {
                WithinUnsortedBuilderIter::<Tree, A, T, SS, LS, D, K, B>::Exclusive(
                    WithinUnsortedIter::<_, _, _, _, _, _, true, K, B>::new(
                        self.tree,
                        self.query,
                        self.radius,
                    ),
                )
            } else {
                WithinUnsortedBuilderIter::<Tree, A, T, SS, LS, D, K, B>::Inclusive(
                    WithinUnsortedIter::<_, _, _, _, _, _, false, K, B>::new(
                        self.tree,
                        self.query,
                        self.radius,
                    ),
                )
            };
            Box::new(iter.map(project_nearest_without_point::<A, T, D::Output, P, I, Dp, K>))
        }
    }
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        P,
        I,
        Dp,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > ExecuteQueryBuilder
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestNState,
        PeriodicSpace,
        Projection<P, I, Dp>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: PeriodicAxis + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B> + KdTreeQueryOps<A, T, SS, LS, K, B>,
    D: DistanceMetric<A>,
    D::Output: Axis<Coord = D::Output>,
    P: PointProjectionField<A, K>,
    I: ProjectionField<T>,
    Dp: ProjectionField<D::Output>,
{
    type Output = Vec<QueryResultItem<P::Output, I::Output, Dp::Output>>;

    #[inline]
    fn execute(self) -> Vec<QueryResultItem<P::Output, I::Output, Dp::Output>> {
        periodic_nearest_results::<Tree, A, T, SS, LS, D, false, K, B>(
            self.tree,
            self.query,
            self.box_size.expect("periodic builder missing box size"),
            None,
            self.max_qty,
            true,
            self.result_capacity,
        )
        .into_iter()
        .map(project_nearest_without_point::<A, T, D::Output, P, I, Dp, K>)
        .collect()
    }
}

macro_rules! impl_periodic_radius_vec_execute {
    ($family:ty) => {
        impl<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                D,
                P,
                I,
                Dp,
                const SORTED: bool,
                const EXCLUSIVE: bool,
                const K: usize,
                const B: usize,
            > ExecuteQueryBuilder
            for QueryBuilder<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                SelectedMetric<D>,
                $family,
                PeriodicSpace,
                Projection<P, I, Dp>,
                SORTED,
                EXCLUSIVE,
                K,
                B,
            >
        where
            A: PeriodicAxis + 'static,
            T: Content + PartialOrd,
            SS: StemStrategy,
            LS: LeafStrategy<A, T, SS, K, B>,
            Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B> + KdTreeQueryOps<A, T, SS, LS, K, B>,
            D: DistanceMetric<A>,
            D::Output: Axis<Coord = D::Output>,
            P: PointProjectionField<A, K>,
            I: ProjectionField<T>,
            Dp: ProjectionField<D::Output>,
        {
            type Output = Vec<QueryResultItem<P::Output, I::Output, Dp::Output>>;

            #[inline]
            fn execute(self) -> Vec<QueryResultItem<P::Output, I::Output, Dp::Output>> {
                periodic_nearest_results::<Tree, A, T, SS, LS, D, EXCLUSIVE, K, B>(
                    self.tree,
                    self.query,
                    self.box_size.expect("periodic builder missing box size"),
                    Some(self.radius),
                    self.max_qty,
                    SORTED,
                    self.result_capacity,
                )
                .into_iter()
                .map(project_nearest_without_point::<A, T, D::Output, P, I, Dp, K>)
                .collect()
            }
        }
    };
}

impl_periodic_radius_vec_execute!(NearestNWithinState);
impl_periodic_radius_vec_execute!(WithinState);

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        P,
        I,
        Dp,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > ExecuteQueryBuilder
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        BestNWithinState,
        CartesianSpace,
        Projection<P, I, Dp>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B> + KdTreeAccessor<A, T, SS, LS, K, B>,
    D: DistanceMetric<A>,
    D::Output: crate::stem_strategy::SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static,
    SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    P: PointProjectionField<A, K>,
    I: ProjectionField<T>,
    Dp: ProjectionField<D::Output>,
    BestQueryResultItem<P::Output, I::Output, Dp::Output>: Ord,
{
    type Output = BinaryHeap<BestQueryResultItem<P::Output, I::Output, Dp::Output>>;

    #[inline]
    fn execute(self) -> BinaryHeap<BestQueryResultItem<P::Output, I::Output, Dp::Output>> {
        if <Projection<P, I, Dp> as ProjectionSpec<A, T, D::Output, K>>::WANTS_POINTS {
            scan_projected_best_results::<Tree, A, T, SS, LS, D, Projection<P, I, Dp>, K, B>(
                self.tree,
                self.query,
                self.radius,
                self.max_qty.expect("max_qty missing").get(),
                if EXCLUSIVE {
                    BoundaryMode::Exclusive
                } else {
                    BoundaryMode::Inclusive
                },
            )
        } else {
            let results = if EXCLUSIVE {
                self.tree.qb_best_n_within::<D, true>(
                    self.query,
                    self.radius,
                    self.max_qty.expect("max_qty missing"),
                )
            } else {
                self.tree.qb_best_n_within::<D, false>(
                    self.query,
                    self.radius,
                    self.max_qty.expect("max_qty missing"),
                )
            };
            results
                .into_iter()
                .map(project_best_without_point::<A, T, D::Output, P, I, Dp, K>)
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::dist::{QueryMetric, SquaredEuclidean};
    #[cfg(feature = "rkyv_08")]
    use crate::kd_tree::ArchivedKdTree;
    use crate::kd_tree::KdTree;
    #[cfg(feature = "rkyv_08")]
    use crate::leaf_strategy::VecOfArenas;
    use crate::leaf_strategy::{FlatVec, VecOfArrays};
    use crate::stem_strategy::Eytzinger;
    use crate::Axis;
    use std::num::NonZeroUsize;

    type WrapTree = KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, 2, 32>, 2, 32>;

    #[test]
    fn periodic_nearest_one_wraps_and_supports_non_point_projection() {
        let points = [(1u32, [0.95, 0.5]), (2u32, [0.40, 0.5])];
        let tree = WrapTree::new_from_entries(&points).unwrap();
        let query = [0.05, 0.5];
        let box_size = [1.0, 1.0];

        let result = tree
            .query(&query)
            .periodic_boundary_condition(&box_size)
            .nearest_one::<SquaredEuclidean<f64>>()
            .execute();
        assert_eq!(result.item, 1);
        assert!((result.distance - 0.01).abs() < f64::EPSILON);

        let projected = tree
            .query(&query)
            .periodic_boundary_condition(&box_size)
            .nearest_one::<SquaredEuclidean<f64>>()
            .without_items()
            .execute();
        assert_eq!(projected.item, ());
        assert_eq!(projected.point, ());
        assert!((projected.distance - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn periodic_radius_queries_use_per_entry_semantics_and_respect_boundary_mode() {
        let points = [
            (1u32, [0.95, 0.5]),
            (2u32, [0.85, 0.5]),
            (3u32, [0.40, 0.5]),
        ];
        let tree = WrapTree::new_from_entries(&points).unwrap();
        let query = [0.05, 0.5];
        let box_size = [1.0, 1.0];
        let boundary_radius = tree
            .query(&query)
            .periodic_boundary_condition(&box_size)
            .nearest_one::<SquaredEuclidean<f64>>()
            .execute()
            .distance;

        let inclusive = tree
            .query(&query)
            .periodic_boundary_condition(&box_size)
            .within::<SquaredEuclidean<f64>>(boundary_radius)
            .unsorted()
            .execute();
        assert_eq!(inclusive.len(), 1);
        assert_eq!(inclusive[0].item, 1);

        let exclusive = tree
            .query(&query)
            .periodic_boundary_condition(&box_size)
            .within::<SquaredEuclidean<f64>>(boundary_radius)
            .exclusive_boundaries()
            .unsorted()
            .execute();
        assert!(exclusive.is_empty());

        let nearest_within = tree
            .query(&query)
            .periodic_boundary_condition(&box_size)
            .nearest_n::<SquaredEuclidean<f64>>(NonZeroUsize::new(2).unwrap())
            .within(0.05)
            .unsorted()
            .execute();
        let mut items: Vec<_> = nearest_within
            .into_iter()
            .map(|result| result.item)
            .collect();
        items.sort_unstable();
        assert_eq!(items, vec![1, 2]);
    }

    #[test]
    fn periodic_plural_queries_do_not_dedup_duplicate_items() {
        let points = [
            (7u32, [0.95, 0.5]),
            (7u32, [0.96, 0.5]),
            (9u32, [0.85, 0.5]),
        ];
        let tree = WrapTree::new_from_entries(&points).unwrap();
        let query = [0.05, 0.5];
        let box_size = [1.0, 1.0];

        let mut results: Vec<_> = tree
            .query(&query)
            .periodic_boundary_condition(&box_size)
            .within::<SquaredEuclidean<f64>>(0.02)
            .unsorted()
            .execute()
            .into_iter()
            .map(|result| result.item)
            .collect();
        results.sort_unstable();

        assert_eq!(results, vec![7, 7]);
    }

    #[test]
    fn reused_scratch_matches_local_scratch_for_cartesian_queries() {
        let entries = [
            (10u32, [0.0, 0.0]),
            (11u32, [0.2, 0.1]),
            (12u32, [0.4, 0.4]),
            (13u32, [0.8, 0.2]),
            (14u32, [0.9, 0.9]),
            (15u32, [0.3, 0.7]),
        ];
        let tree = WrapTree::new_from_entries(&entries).unwrap();
        let query = [0.25, 0.2];
        let max_qty = NonZeroUsize::new(3).unwrap();
        let radius = 0.25;
        let mut scratch = tree.create_scratch::<SquaredEuclidean<f64>>();

        assert_eq!(
            tree.query(&query)
                .nearest_one::<SquaredEuclidean<f64>>()
                .execute(),
            tree.query(&query)
                .nearest_one::<SquaredEuclidean<f64>>()
                .with_scratch(&mut scratch)
                .execute()
        );

        assert_eq!(
            tree.query(&query)
                .nearest_n::<SquaredEuclidean<f64>>(max_qty)
                .unsorted()
                .execute(),
            tree.query(&query)
                .nearest_n::<SquaredEuclidean<f64>>(max_qty)
                .unsorted()
                .with_scratch(&mut scratch)
                .execute()
        );

        assert_eq!(
            tree.query(&query)
                .within::<SquaredEuclidean<f64>>(radius)
                .unsorted()
                .execute(),
            tree.query(&query)
                .within::<SquaredEuclidean<f64>>(radius)
                .unsorted()
                .with_scratch(&mut scratch)
                .execute()
        );

        let local_best = tree
            .query(&query)
            .best_n_within::<SquaredEuclidean<f64>>(radius, max_qty)
            .execute()
            .into_sorted_vec();
        let reused_best = tree
            .query(&query)
            .best_n_within::<SquaredEuclidean<f64>>(radius, max_qty)
            .with_scratch(&mut scratch)
            .execute()
            .into_sorted_vec();
        assert_eq!(local_best, reused_best);

        let mut local_visit = Vec::new();
        tree.query(&query)
            .within::<SquaredEuclidean<f64>>(radius)
            .unsorted()
            .visit(|result| local_visit.push(result));
        let mut scratch_visit = Vec::new();
        tree.query(&query)
            .within::<SquaredEuclidean<f64>>(radius)
            .unsorted()
            .with_scratch(&mut scratch)
            .visit(|result| scratch_visit.push(result));
        assert_eq!(local_visit, scratch_visit);
    }

    #[test]
    fn unsorted_within_projects_results_and_honors_capacity_hint() {
        let entries = [
            (10u32, [0.0, 0.0]),
            (11u32, [0.2, 0.1]),
            (12u32, [0.4, 0.4]),
            (13u32, [0.8, 0.2]),
            (14u32, [0.9, 0.9]),
            (15u32, [0.3, 0.7]),
        ];
        let tree = WrapTree::new_from_entries(&entries).unwrap();
        let query = [0.25, 0.2];
        let radius = 0.25;
        let requested_capacity = 128;

        let full = tree
            .query(&query)
            .within::<SquaredEuclidean<f64>>(radius)
            .unsorted()
            .execute();
        let item_only = tree
            .query(&query)
            .within::<SquaredEuclidean<f64>>(radius)
            .unsorted()
            .without_distances()
            .with_result_capacity(requested_capacity)
            .execute();

        assert_eq!(
            item_only,
            full.iter()
                .map(|result| crate::QueryResultItem {
                    point: (),
                    item: result.item,
                    distance: (),
                })
                .collect::<Vec<_>>()
        );
        assert!(item_only.capacity() >= requested_capacity);

        let distance_only = tree
            .query(&query)
            .within::<SquaredEuclidean<f64>>(radius)
            .unsorted()
            .without_items()
            .with_result_capacity(requested_capacity)
            .execute();
        assert_eq!(
            distance_only,
            full.iter()
                .map(|result| crate::QueryResultItem {
                    point: (),
                    item: (),
                    distance: result.distance,
                })
                .collect::<Vec<_>>()
        );
        assert!(distance_only.capacity() >= requested_capacity);

        let with_points = tree
            .query(&query)
            .within::<SquaredEuclidean<f64>>(radius)
            .unsorted()
            .with_points()
            .with_result_capacity(requested_capacity)
            .execute();
        assert_eq!(
            with_points
                .iter()
                .map(|result| crate::QueryResultItem {
                    point: (),
                    item: result.item,
                    distance: result.distance,
                })
                .collect::<Vec<_>>(),
            full
        );
        assert!(with_points.capacity() >= requested_capacity);
        for result in &with_points {
            assert_eq!(
                result.point,
                entries
                    .iter()
                    .find_map(|&(item, point)| (item == result.item).then_some(point))
                    .unwrap()
            );
        }

        let item_and_point = tree
            .query(&query)
            .within::<SquaredEuclidean<f64>>(radius)
            .unsorted()
            .with_points()
            .without_distances()
            .execute();
        assert_eq!(
            item_and_point,
            with_points
                .iter()
                .map(|result| crate::QueryResultItem {
                    point: result.point,
                    item: result.item,
                    distance: (),
                })
                .collect::<Vec<_>>()
        );

        let mut visited_with_points = Vec::new();
        tree.query(&query)
            .within::<SquaredEuclidean<f64>>(radius)
            .unsorted()
            .with_points()
            .visit(|result| visited_with_points.push(result));
        assert_eq!(visited_with_points, with_points);

        let exclusive_full = tree
            .query(&query)
            .within::<SquaredEuclidean<f64>>(radius)
            .exclusive_boundaries()
            .unsorted()
            .execute();
        let exclusive_item_only = tree
            .query(&query)
            .within::<SquaredEuclidean<f64>>(radius)
            .exclusive_boundaries()
            .unsorted()
            .without_distances()
            .execute();
        assert_eq!(
            exclusive_item_only,
            exclusive_full
                .iter()
                .map(|result| crate::QueryResultItem {
                    point: (),
                    item: result.item,
                    distance: (),
                })
                .collect::<Vec<_>>()
        );
        let exclusive_with_points = tree
            .query(&query)
            .within::<SquaredEuclidean<f64>>(radius)
            .exclusive_boundaries()
            .unsorted()
            .with_points()
            .execute();
        assert_eq!(
            exclusive_with_points
                .iter()
                .map(|result| crate::QueryResultItem {
                    point: (),
                    item: result.item,
                    distance: result.distance,
                })
                .collect::<Vec<_>>(),
            exclusive_full
        );

        let mut scratch = tree.create_scratch::<SquaredEuclidean<f64>>();
        let item_only_with_scratch = tree
            .query(&query)
            .within::<SquaredEuclidean<f64>>(radius)
            .unsorted()
            .without_distances()
            .with_result_capacity(requested_capacity)
            .with_scratch(&mut scratch)
            .execute();
        assert_eq!(item_only_with_scratch, item_only);
        assert!(item_only_with_scratch.capacity() >= requested_capacity);
    }

    #[test]
    fn scratch_can_be_reused_across_mutable_mapped_tree_queries() {
        type MutableTree = KdTree<f32, u32, Eytzinger, VecOfArrays<f32, u32, 2, 4>, 2, 4>;

        let mut tree = MutableTree::default();
        for idx in 0..32 {
            let point = [
                ((idx * 7) % 23) as f32 / 23.0,
                ((idx * 11 + 3) % 29) as f32 / 29.0,
            ];
            tree.add(&point, idx as u32).unwrap();
        }

        let mut scratch = tree.create_scratch::<SquaredEuclidean<f32>>();
        for query in [[0.1, 0.2], [0.45, 0.55], [0.8, 0.25]] {
            let max_qty = NonZeroUsize::new(5).unwrap();
            let local = tree
                .query(&query)
                .nearest_n::<SquaredEuclidean<f32>>(max_qty)
                .within(0.4)
                .execute();
            let reused = tree
                .query(&query)
                .nearest_n::<SquaredEuclidean<f32>>(max_qty)
                .within(0.4)
                .with_scratch(&mut scratch)
                .execute();
            assert_eq!(local, reused);
        }
    }

    #[test]
    fn scratch_queries_handle_empty_trees() {
        let entries: [(u32, [f64; 2]); 0] = [];
        let tree = WrapTree::new_from_entries(&entries).unwrap();
        let query = [0.25, 0.2];
        let mut scratch = tree.create_scratch::<SquaredEuclidean<f64>>();

        assert_eq!(
            tree.query(&query)
                .nearest_one::<SquaredEuclidean<f64>>()
                .execute(),
            tree.query(&query)
                .nearest_one::<SquaredEuclidean<f64>>()
                .with_scratch(&mut scratch)
                .execute()
        );
        assert_eq!(
            tree.query(&query)
                .within::<SquaredEuclidean<f64>>(0.25)
                .execute(),
            Vec::new()
        );
    }

    #[cfg(feature = "rkyv_08")]
    #[test]
    fn archived_reused_scratch_matches_local_scratch_for_cartesian_queries() {
        type Tree = KdTree<f64, u32, Eytzinger, VecOfArenas<f64, u32, 2, 32>, 2, 32>;
        type ArchivedTree =
            ArchivedKdTree<f64, u32, Eytzinger, VecOfArenas<f64, u32, 2, 32>, 2, 32>;

        let entries = vec![
            (10u32, [0.0, 0.0]),
            (11u32, [0.2, 0.1]),
            (12u32, [0.4, 0.4]),
            (13u32, [0.8, 0.2]),
            (14u32, [0.9, 0.9]),
            (15u32, [0.3, 0.7]),
        ];
        let tree = Tree::new_from_entries(&entries).unwrap();
        let query = [0.25, 0.2];
        let max_qty = NonZeroUsize::new(3).unwrap();
        let radius = 0.25;

        let bytes = rkyv_08::api::high::to_bytes_in::<_, rkyv_08::rancor::Error>(
            &tree,
            rkyv_08::util::AlignedVec::<128>::new(),
        )
        .unwrap();
        let archived =
            rkyv_08::access::<ArchivedTree, rkyv_08::rancor::Error>(bytes.as_slice()).unwrap();
        let mut scratch = archived.create_scratch::<SquaredEuclidean<f64>>();

        assert_eq!(
            archived
                .query(&query)
                .nearest_n::<SquaredEuclidean<f64>>(max_qty)
                .execute(),
            archived
                .query(&query)
                .nearest_n::<SquaredEuclidean<f64>>(max_qty)
                .with_scratch(&mut scratch)
                .execute()
        );
        assert_eq!(
            archived
                .query(&query)
                .within::<SquaredEuclidean<f64>>(radius)
                .unsorted()
                .execute(),
            archived
                .query(&query)
                .within::<SquaredEuclidean<f64>>(radius)
                .unsorted()
                .with_scratch(&mut scratch)
                .execute()
        );

        let projected = archived
            .query(&query)
            .within::<SquaredEuclidean<f64>>(radius)
            .unsorted()
            .without_distances()
            .execute();
        let owned_projected = tree
            .query(&query)
            .within::<SquaredEuclidean<f64>>(radius)
            .unsorted()
            .without_distances()
            .execute();
        assert_eq!(projected, owned_projected);
        assert!(projected.capacity() >= super::DEFAULT_UNSORTED_RESULT_CAPACITY);
        assert!(owned_projected.capacity() >= super::DEFAULT_UNSORTED_RESULT_CAPACITY);

        let archived_with_points = archived
            .query(&query)
            .within::<SquaredEuclidean<f64>>(radius)
            .unsorted()
            .with_points()
            .execute();
        let owned_with_points = tree
            .query(&query)
            .within::<SquaredEuclidean<f64>>(radius)
            .unsorted()
            .with_points()
            .execute();
        assert_eq!(archived_with_points, owned_with_points);

        let archived_item_and_point = archived
            .query(&query)
            .within::<SquaredEuclidean<f64>>(radius)
            .unsorted()
            .with_points()
            .without_distances()
            .execute();
        assert_eq!(
            archived_item_and_point,
            archived_with_points
                .iter()
                .map(|result| crate::QueryResultItem {
                    point: result.point,
                    item: result.item,
                    distance: (),
                })
                .collect::<Vec<_>>()
        );

        let mut archived_visited_with_points = Vec::new();
        archived
            .query(&query)
            .within::<SquaredEuclidean<f64>>(radius)
            .unsorted()
            .with_points()
            .visit(|result| archived_visited_with_points.push(result));
        assert_eq!(archived_visited_with_points, archived_with_points);

        let requested_capacity = 128;
        let projected_with_scratch = archived
            .query(&query)
            .within::<SquaredEuclidean<f64>>(radius)
            .unsorted()
            .without_distances()
            .with_result_capacity(requested_capacity)
            .with_scratch(&mut scratch)
            .execute();
        assert_eq!(projected_with_scratch, projected);
        assert!(projected_with_scratch.capacity() >= requested_capacity);

        let local_best = archived
            .query(&query)
            .best_n_within::<SquaredEuclidean<f64>>(radius, max_qty)
            .execute()
            .into_sorted_vec();
        let reused_best = archived
            .query(&query)
            .best_n_within::<SquaredEuclidean<f64>>(radius, max_qty)
            .with_scratch(&mut scratch)
            .execute()
            .into_sorted_vec();
        assert_eq!(local_best, reused_best);

        let mut local_visit = Vec::new();
        archived
            .query(&query)
            .within::<SquaredEuclidean<f64>>(radius)
            .unsorted()
            .visit(|result| local_visit.push(result));
        let mut scratch_visit = Vec::new();
        archived
            .query(&query)
            .within::<SquaredEuclidean<f64>>(radius)
            .unsorted()
            .with_scratch(&mut scratch)
            .visit(|result| scratch_visit.push(result));
        assert_eq!(local_visit, scratch_visit);
    }

    #[cfg(feature = "rkyv_08")]
    #[test]
    fn archived_periodic_queries_match_owned() {
        type Tree = KdTree<f64, u32, Eytzinger, VecOfArenas<f64, u32, 2, 32>, 2, 32>;
        type ArchivedTree =
            ArchivedKdTree<f64, u32, Eytzinger, VecOfArenas<f64, u32, 2, 32>, 2, 32>;

        let points = vec![
            (1u32, [0.95, 0.5]),
            (2u32, [0.85, 0.5]),
            (3u32, [0.40, 0.5]),
        ];
        let tree = Tree::new_from_entries(&points).unwrap();
        let query = [0.05, 0.5];
        let box_size = [1.0, 1.0];

        let bytes = rkyv_08::api::high::to_bytes_in::<_, rkyv_08::rancor::Error>(
            &tree,
            rkyv_08::util::AlignedVec::<128>::new(),
        )
        .unwrap();
        let archived =
            rkyv_08::access::<ArchivedTree, rkyv_08::rancor::Error>(bytes.as_slice()).unwrap();

        assert_eq!(
            archived
                .query(&query)
                .periodic_boundary_condition(&box_size)
                .nearest_one::<SquaredEuclidean<f64>>()
                .execute(),
            tree.query(&query)
                .periodic_boundary_condition(&box_size)
                .nearest_one::<SquaredEuclidean<f64>>()
                .execute()
        );
        assert_eq!(
            archived
                .query(&query)
                .periodic_boundary_condition(&box_size)
                .nearest_n::<SquaredEuclidean<f64>>(NonZeroUsize::new(2).unwrap())
                .within(0.05)
                .execute(),
            tree.query(&query)
                .periodic_boundary_condition(&box_size)
                .nearest_n::<SquaredEuclidean<f64>>(NonZeroUsize::new(2).unwrap())
                .within(0.05)
                .execute()
        );
    }

    #[test]
    fn generic_query_metric_helper_can_execute_nearest_n() {
        fn query<A, M, const K: usize>(
            tree: &KdTree<A, u32, Eytzinger, VecOfArrays<A, u32, K, 32>, K, 32>,
            point: &[A; K],
            n: NonZeroUsize,
        ) -> Vec<u32>
        where
            A: Axis<Coord = A> + Copy + 'static,
            M: QueryMetric<A>,
        {
            tree.query(point)
                .nearest_n::<M>(n)
                .execute()
                .into_iter()
                .map(|result| result.item)
                .collect()
        }

        let points = [
            (10u32, [0.0f64, 0.0]),
            (20u32, [1.0, 1.0]),
            (30u32, [2.0, 2.0]),
        ];
        let tree =
            KdTree::<f64, u32, Eytzinger, VecOfArrays<f64, u32, 2, 32>, 2, 32>::new_from_entries(
                &points,
            )
            .unwrap();

        let results = query::<f64, SquaredEuclidean<f64>, 2>(
            &tree,
            &[0.1, 0.1],
            NonZeroUsize::new(2).unwrap(),
        );

        assert_eq!(results, vec![10, 20]);
    }
}

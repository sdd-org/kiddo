use std::cmp::Ordering;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::ptr::NonNull;

use crate::dist::DistanceMetric;
use crate::kd_tree::query_context::QueryContext;
use crate::kd_tree::{KdTreeAccessor, KdTreeQueryOps, StemLeafResolution};
use crate::leaf_view::LeafView;
#[cfg(not(feature = "small_n_result_collectors"))]
use crate::results::result_collection::ThresholdVecResultCollection;
use crate::results::result_collection::{BinaryHeapResultCollection, ResultCollection};
#[cfg(feature = "small_n_result_collectors")]
use crate::results::result_collection::{
    SmallSortedVecResultCollection, SMALL_RESULT_COLLECTION_MAX_QTY,
};
use crate::stem_strategy::donnelly::simd_full::{
    interval_distance_1d, BacktrackBlock3, BacktrackBlock4, SimdSelectBestChildBlock3,
};
use crate::traits::leaf_strategy::LeafProjection;
use crate::{Axis, Content, LeafStrategy, QueryResultItem, StemStrategy};

use super::builder::{boundary_accepts, BoundaryMode, PeriodicAxis, QueryBuilderTreeOps};

#[cfg(not(feature = "small_n_result_collectors"))]
const MAX_VEC_RESULT_SIZE: usize = 20;

#[inline(always)]
fn periodic_image_axis_offset<A: PeriodicAxis>(wrapped_coord: A, axis_len: A) -> A {
    if A::cmp(wrapped_coord, A::zero()) == Ordering::Less {
        A::saturating_dist(wrapped_coord, A::zero())
    } else if A::cmp(wrapped_coord, axis_len) == Ordering::Greater {
        A::saturating_dist(wrapped_coord, axis_len)
    } else {
        A::zero()
    }
}

#[derive(Clone, Copy)]
struct PeriodicImageCandidate<A, O, const K: usize> {
    wrapped_query: [A; K],
    lower_bound: O,
}

fn periodic_image_candidates<D, A, const K: usize>(
    query: &[A; K],
    box_size: &[A; K],
    threshold: Option<(BoundaryMode, D::Output)>,
    include_home: bool,
) -> Vec<PeriodicImageCandidate<A, D::Output, K>>
where
    A: PeriodicAxis + 'static,
    D: DistanceMetric<A>,
    D::Output: Axis<Coord = D::Output>,
{
    fn recurse<D, A, const K: usize>(
        query: &[A; K],
        box_size: &[A; K],
        axis: usize,
        wrapped_query: &mut [A; K],
        has_non_zero_shift: bool,
        lower_bound: D::Output,
        threshold: Option<(BoundaryMode, D::Output)>,
        include_home: bool,
        out: &mut Vec<PeriodicImageCandidate<A, D::Output, K>>,
    ) where
        A: PeriodicAxis + 'static,
        D: DistanceMetric<A>,
        D::Output: Axis<Coord = D::Output>,
    {
        if axis == K {
            if include_home || has_non_zero_shift {
                out.push(PeriodicImageCandidate {
                    wrapped_query: *wrapped_query,
                    lower_bound,
                });
            }
            return;
        }

        let original = query[axis];
        let axis_len = box_size[axis];

        for shift in [-1_i8, 0, 1] {
            let wrapped_coord = match shift {
                -1 => original - axis_len,
                0 => original,
                1 => {
                    let mut plus = original;
                    plus += axis_len;
                    plus
                }
                _ => unreachable!(),
            };

            wrapped_query[axis] = wrapped_coord;

            let offset = periodic_image_axis_offset(wrapped_coord, axis_len);
            let mut next_lower_bound = lower_bound;
            D::combine_component(
                &mut next_lower_bound,
                D::dist1(D::widen_coord(offset), D::Output::zero()),
            );

            if threshold.is_some_and(|(boundary, limit)| {
                !boundary_accepts(boundary, next_lower_bound, limit)
            }) {
                continue;
            }

            recurse::<D, A, K>(
                query,
                box_size,
                axis + 1,
                wrapped_query,
                has_non_zero_shift || shift != 0,
                next_lower_bound,
                threshold,
                include_home,
                out,
            );
        }

        wrapped_query[axis] = original;
    }

    let mut candidates = Vec::new();
    let mut wrapped_query = *query;
    recurse::<D, A, K>(
        query,
        box_size,
        0,
        &mut wrapped_query,
        false,
        D::Output::zero(),
        threshold,
        include_home,
        &mut candidates,
    );
    candidates
}

#[inline(always)]
fn sort_periodic_image_candidates<A, O, const K: usize>(
    candidates: &mut [PeriodicImageCandidate<A, O, K>],
) where
    O: Axis<Coord = O>,
{
    candidates.sort_unstable_by(|lhs, rhs| {
        lhs.lower_bound
            .partial_cmp(&rhs.lower_bound)
            .unwrap_or(Ordering::Equal)
    });
}

pub(crate) fn periodic_nearest_one_result<Tree, A, T, SS, LS, D, const K: usize, const B: usize>(
    tree: &Tree,
    query: &[A; K],
    box_size: &[A; K],
) -> QueryResultItem<(), T, D::Output>
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
        + crate::leaf_view::TlsLeafScratch
        + 'static,
    SS::Stack<D::Output, K>:
        crate::kd_tree::query_stack::StackTrait<D::Output, SS, K> + Default + 'static,
{
    assert!(
        A::periodic_box_is_valid(box_size),
        "periodic box sizes must be strictly positive"
    );

    let (home_distance, home_item) = tree.qb_nearest_one::<D>(query);
    let mut best_result = QueryResultItem {
        point: (),
        item: home_item,
        distance: home_distance,
    };

    let mut candidates = periodic_image_candidates::<D, A, K>(
        query,
        box_size,
        Some((BoundaryMode::Exclusive, best_result.distance)),
        false,
    );
    sort_periodic_image_candidates(&mut candidates);

    for candidate in candidates {
        if !boundary_accepts(
            BoundaryMode::Exclusive,
            candidate.lower_bound,
            best_result.distance,
        ) {
            continue;
        }

        let (distance, item) = tree.qb_nearest_one::<D>(&candidate.wrapped_query);
        if D::Output::cmp(distance, best_result.distance) == Ordering::Less {
            best_result.distance = distance;
            best_result.item = item;
        }
    }

    best_result
}

#[derive(Clone)]
struct PeriodicTraversalFrame<O, S, const K: usize> {
    stem_state: S,
    lower: [O; K],
    upper: [O; K],
    off: [O; K],
    rd: O,
}

struct PeriodicNearestResultsReqCtx<'a, A, T, O, R, const EXCLUSIVE: bool, const K: usize>
where
    O: Axis<Coord = O>,
{
    query: &'a [A; K],
    max_dist: O,
    results: R,
    _phantom: PhantomData<T>,
}

impl<A, T, O, R, const EXCLUSIVE: bool, const K: usize> QueryContext<A, O, K>
    for PeriodicNearestResultsReqCtx<'_, A, T, O, R, EXCLUSIVE, K>
where
    O: Axis<Coord = O>,
    R: ResultCollection<O, QueryResultItem<(), T, O>>,
{
    #[inline(always)]
    fn query(&self) -> &[A; K] {
        self.query
    }

    #[inline(always)]
    fn max_dist(&self) -> O {
        let results_cap = self.results.threshold_distance().unwrap_or(O::max_value());
        if results_cap < self.max_dist {
            results_cap
        } else {
            self.max_dist
        }
    }

    #[inline(always)]
    fn prune_on_equal_max_dist(&self) -> bool {
        EXCLUSIVE
    }
}

#[inline(always)]
fn periodic_should_prune<O>(rd: O, max_dist: O, prune_on_equal: bool) -> bool
where
    O: Axis<Coord = O>,
{
    let rd_vs_max = O::cmp(rd, max_dist);
    rd_vs_max == Ordering::Greater || (prune_on_equal && rd_vs_max == Ordering::Equal)
}

#[inline(always)]
fn periodic_min_image_axis_delta<O>(query: O, coord: O, axis_len: O) -> O
where
    O: Axis<Coord = O>,
{
    let delta = O::saturating_dist(query, coord);
    let wrapped = axis_len - delta;
    if O::cmp(wrapped, delta) == Ordering::Less {
        wrapped
    } else {
        delta
    }
}

#[inline(always)]
fn periodic_interval_distance_1d<O>(query: O, lower: O, upper: O, axis_len: O) -> O
where
    O: Axis<Coord = O>,
{
    let mut best = interval_distance_1d(query, lower, upper);

    let minus = interval_distance_1d(query, lower - axis_len, upper - axis_len);
    if O::cmp(minus, best) == Ordering::Less {
        best = minus;
    }

    let mut lower_plus = lower;
    lower_plus += axis_len;
    let mut upper_plus = upper;
    upper_plus += axis_len;
    let plus = interval_distance_1d(query, lower_plus, upper_plus);
    if O::cmp(plus, best) == Ordering::Less {
        best = plus;
    }

    best
}

#[inline(always)]
fn process_periodic_leaf_view<A, T, D, R, const EXCLUSIVE: bool, const K: usize, const B: usize>(
    leaf: &LeafView<'_, A, T, K, B>,
    query_wide: &[D::Output; K],
    box_size_wide: &[D::Output; K],
    max_dist: D::Output,
    results: &mut R,
) where
    A: Axis<Coord = A> + 'static,
    T: Content,
    D: DistanceMetric<A>,
    D::Output: Axis<Coord = D::Output>,
    R: ResultCollection<D::Output, QueryResultItem<(), T, D::Output>>,
{
    let points = leaf.points();
    let items = leaf.items();

    for idx in 0..leaf.len() {
        let mut candidate_dist = D::Output::zero();
        for dim in 0..K {
            let coord = D::widen_coord(unsafe { *points.get_unchecked(dim).get_unchecked(idx) });
            let delta = periodic_min_image_axis_delta(
                unsafe { *query_wide.get_unchecked(dim) },
                coord,
                unsafe { *box_size_wide.get_unchecked(dim) },
            );
            D::combine_component(&mut candidate_dist, D::dist1(delta, D::Output::zero()));
        }

        if boundary_accepts(
            if EXCLUSIVE {
                BoundaryMode::Exclusive
            } else {
                BoundaryMode::Inclusive
            },
            candidate_dist,
            max_dist,
        ) {
            results.add(QueryResultItem {
                point: (),
                item: unsafe { *items.get_unchecked(idx) },
                distance: candidate_dist,
            });
        }
    }
}

#[inline(always)]
fn process_periodic_leaf_arena<A, T, D, R, const EXCLUSIVE: bool, const K: usize>(
    arena: crate::leaf_view::LeafArena<'_, A, T, K>,
    query_wide: &[D::Output; K],
    box_size_wide: &[D::Output; K],
    max_dist: D::Output,
    results: &mut R,
) where
    A: Axis<Coord = A> + 'static,
    T: Content,
    D: DistanceMetric<A>,
    D::Output: Axis<Coord = D::Output>,
    R: ResultCollection<D::Output, QueryResultItem<(), T, D::Output>>,
{
    arena.for_each_tiled_chunk(|tile| {
        for idx in 0..tile.len() {
            let mut candidate_dist = D::Output::zero();
            for dim in 0..K {
                let coord = D::widen_coord(unsafe { tile.point_unaligned(dim, idx) });
                let delta = periodic_min_image_axis_delta(
                    unsafe { *query_wide.get_unchecked(dim) },
                    coord,
                    unsafe { *box_size_wide.get_unchecked(dim) },
                );
                D::combine_component(&mut candidate_dist, D::dist1(delta, D::Output::zero()));
            }

            if boundary_accepts(
                if EXCLUSIVE {
                    BoundaryMode::Exclusive
                } else {
                    BoundaryMode::Inclusive
                },
                candidate_dist,
                max_dist,
            ) {
                results.add(QueryResultItem {
                    point: (),
                    item: unsafe { tile.item_unaligned(idx) },
                    distance: candidate_dist,
                });
            }
        }
    });
}

#[inline(always)]
fn process_periodic_leaf<
    Tree,
    A,
    T,
    SS,
    LS,
    D,
    R,
    const EXCLUSIVE: bool,
    const K: usize,
    const B: usize,
>(
    tree: &Tree,
    leaf_idx: usize,
    query_wide: &[D::Output; K],
    box_size_wide: &[D::Output; K],
    max_dist: D::Output,
    results: &mut R,
) where
    A: Axis<Coord = A> + 'static,
    T: Content,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: KdTreeAccessor<A, T, SS, LS, K, B>,
    D: DistanceMetric<A>,
    D::Output: Axis<Coord = D::Output>,
    R: ResultCollection<D::Output, QueryResultItem<(), T, D::Output>>,
{
    match LS::LEAF_PROJECTION {
        LeafProjection::LeafView => {
            let leaf = tree.leaves().leaf_view(leaf_idx);
            process_periodic_leaf_view::<A, T, D, R, EXCLUSIVE, K, B>(
                &leaf,
                query_wide,
                box_size_wide,
                max_dist,
                results,
            );
        }
        LeafProjection::LeafArena => {
            let arena = tree.leaves().leaf_arena(leaf_idx);
            process_periodic_leaf_arena::<A, T, D, R, EXCLUSIVE, K>(
                arena,
                query_wide,
                box_size_wide,
                max_dist,
                results,
            );
        }
    }
}

fn periodic_backtracking_query<Tree, A, T, SS, LS, QC, O, D, F, const K: usize, const B: usize>(
    tree: &Tree,
    query_ctx: &mut QC,
    box_size: &[A; K],
    mut process_leaf: F,
) where
    Tree: KdTreeAccessor<A, T, SS, LS, K, B> + KdTreeQueryOps<A, T, SS, LS, K, B>,
    A: PeriodicAxis + 'static,
    T: Content,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    QC: QueryContext<A, O, K>,
    O: Axis<Coord = O>,
    D: DistanceMetric<A, Output = O>,
    F: FnMut(usize, &[O; K], &[O; K], &mut QC),
{
    if tree.size() == 0 {
        return;
    }

    let stems_ptr = NonNull::new(tree.stems().as_ptr() as *mut u8).unwrap();
    let mut stem_strat: SS = SS::new(stems_ptr);

    let query = *query_ctx.query();
    let mut query_wide = [O::zero(); K];
    let mut box_size_wide = [O::zero(); K];
    for dim in 0..K {
        query_wide[dim] = D::widen_coord(query[dim]);
        box_size_wide[dim] = D::widen_coord(box_size[dim]);
    }

    let mut lower = [O::zero(); K];
    let mut upper = box_size_wide;
    let mut off = [O::zero(); K];
    let mut rd = O::zero();

    let mut stack: Vec<PeriodicTraversalFrame<O, SS::DeferredState, K>> =
        Vec::with_capacity((tree.max_stem_level().max(0) as usize) + 1);

    loop {
        loop {
            if let Some(leaf_idx) = tree.resolve_terminal_stem(stem_strat.stem_idx()) {
                process_leaf(leaf_idx, &query_wide, &box_size_wide, query_ctx);
                break;
            }

            if stem_strat.level() > tree.max_stem_level() {
                let leaf_idx = tree
                    .stem_leaf_resolution()
                    .resolve_terminal_stem_idx(stem_strat.stem_idx(), stem_strat.leaf_idx());
                process_leaf(leaf_idx, &query_wide, &box_size_wide, query_ctx);
                break;
            }

            let dim = stem_strat.dim::<K>();
            let stem_idx = stem_strat.stem_idx();
            let pivot = if stem_idx < tree.stems().len() {
                unsafe { *tree.stems().get_unchecked(stem_idx) }
            } else {
                A::max_value()
            };

            if pivot >= A::max_value() {
                stem_strat.traverse::<A, K>(false);
                continue;
            }

            let old_lower = lower[dim];
            let old_upper = upper[dim];
            let pivot_wide = D::widen_coord(pivot);
            let query_val = query[dim];
            let query_wide_val = query_wide[dim];
            let is_right_child = query_val >= pivot;
            let far_ctx = stem_strat.branch_relative::<A, K>(is_right_child);

            let (near_lower, near_upper, far_lower, far_upper) = if is_right_child {
                (
                    O::max(old_lower, pivot_wide),
                    old_upper,
                    old_lower,
                    if O::cmp(old_upper, pivot_wide) == Ordering::Less {
                        old_upper
                    } else {
                        pivot_wide
                    },
                )
            } else {
                (
                    old_lower,
                    if O::cmp(old_upper, pivot_wide) == Ordering::Less {
                        old_upper
                    } else {
                        pivot_wide
                    },
                    O::max(old_lower, pivot_wide),
                    old_upper,
                )
            };

            let near_off = periodic_interval_distance_1d(
                query_wide_val,
                near_lower,
                near_upper,
                box_size_wide[dim],
            );
            let far_off = periodic_interval_distance_1d(
                query_wide_val,
                far_lower,
                far_upper,
                box_size_wide[dim],
            );
            let near_rd = D::rect_dist_after_update(rd, &off, dim, near_off);
            let far_rd = D::rect_dist_after_update(rd, &off, dim, far_off);

            let threshold = query_ctx.max_dist();
            let prune_on_equal = query_ctx.prune_on_equal_max_dist();
            let near_pruned = periodic_should_prune(near_rd, threshold, prune_on_equal);
            let far_pruned = periodic_should_prune(far_rd, threshold, prune_on_equal);

            if !far_pruned {
                let mut far_lower_state = lower;
                far_lower_state[dim] = far_lower;
                let mut far_upper_state = upper;
                far_upper_state[dim] = far_upper;
                let mut far_off_state = off;
                far_off_state[dim] = far_off;
                stack.push(PeriodicTraversalFrame {
                    stem_state: far_ctx.deferred_state(),
                    lower: far_lower_state,
                    upper: far_upper_state,
                    off: far_off_state,
                    rd: far_rd,
                });
            }

            if near_pruned {
                if let Some(frame) = stack.pop() {
                    stem_strat.rehydrate_deferred_state(frame.stem_state);
                    lower = frame.lower;
                    upper = frame.upper;
                    off = frame.off;
                    rd = frame.rd;
                    continue;
                }
                return;
            }

            lower[dim] = near_lower;
            upper[dim] = near_upper;
            off[dim] = near_off;
            rd = near_rd;
        }

        if let Some(frame) = stack.pop() {
            stem_strat.rehydrate_deferred_state(frame.stem_state);
            lower = frame.lower;
            upper = frame.upper;
            off = frame.off;
            rd = frame.rd;
            continue;
        }

        break;
    }
}

fn periodic_nearest_results_inner<
    Tree,
    A,
    T,
    SS,
    LS,
    D,
    R,
    const EXCLUSIVE: bool,
    const K: usize,
    const B: usize,
>(
    tree: &Tree,
    query: &[A; K],
    box_size: &[A; K],
    max_dist: D::Output,
    max_qty: usize,
    sorted: bool,
    result_capacity: Option<NonZeroUsize>,
) -> Vec<QueryResultItem<(), T, D::Output>>
where
    Tree: KdTreeAccessor<A, T, SS, LS, K, B> + KdTreeQueryOps<A, T, SS, LS, K, B>,
    A: PeriodicAxis + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    D: DistanceMetric<A>,
    D::Output: Axis<Coord = D::Output>,
    R: ResultCollection<D::Output, QueryResultItem<(), T, D::Output>>,
{
    let mut req_ctx = PeriodicNearestResultsReqCtx::<A, T, D::Output, R, EXCLUSIVE, K> {
        query,
        max_dist,
        results: R::with_max_qty_and_capacity(max_qty, result_capacity),
        _phantom: PhantomData,
    };

    periodic_backtracking_query::<Tree, A, T, SS, LS, _, _, D, _, K, B>(
        tree,
        &mut req_ctx,
        box_size,
        |leaf_idx, query_wide, box_size_wide, req_ctx| {
            let leaf_max_dist = req_ctx.max_dist();
            process_periodic_leaf::<Tree, A, T, SS, LS, D, R, EXCLUSIVE, K, B>(
                tree,
                leaf_idx,
                query_wide,
                box_size_wide,
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

pub(crate) fn periodic_nearest_results<
    Tree,
    A,
    T,
    SS,
    LS,
    D,
    const EXCLUSIVE: bool,
    const K: usize,
    const B: usize,
>(
    tree: &Tree,
    query: &[A; K],
    box_size: &[A; K],
    radius: Option<D::Output>,
    max_qty: Option<NonZeroUsize>,
    sorted: bool,
    result_capacity: Option<NonZeroUsize>,
) -> Vec<QueryResultItem<(), T, D::Output>>
where
    Tree: KdTreeAccessor<A, T, SS, LS, K, B> + KdTreeQueryOps<A, T, SS, LS, K, B>,
    A: PeriodicAxis + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    D: DistanceMetric<A>,
    D::Output: Axis<Coord = D::Output>,
{
    assert!(
        A::periodic_box_is_valid(box_size),
        "periodic box sizes must be strictly positive"
    );
    assert!(
        radius.is_some() || max_qty.is_some(),
        "periodic plural queries require radius and/or max_qty"
    );

    let max_dist = radius.unwrap_or_else(D::Output::max_value);
    let max_qty = max_qty.map_or(usize::MAX, NonZeroUsize::get);

    if max_qty == usize::MAX {
        return periodic_nearest_results_inner::<
            Tree,
            A,
            T,
            SS,
            LS,
            D,
            Vec<QueryResultItem<(), T, D::Output>>,
            EXCLUSIVE,
            K,
            B,
        >(
            tree,
            query,
            box_size,
            max_dist,
            max_qty,
            sorted,
            result_capacity,
        );
    }

    if sorted {
        #[cfg(feature = "small_n_result_collectors")]
        if max_qty <= SMALL_RESULT_COLLECTION_MAX_QTY {
            return periodic_nearest_results_inner::<
                Tree,
                A,
                T,
                SS,
                LS,
                D,
                SmallSortedVecResultCollection<QueryResultItem<(), T, D::Output>>,
                EXCLUSIVE,
                K,
                B,
            >(tree, query, box_size, max_dist, max_qty, sorted, None);
        }

        #[cfg(not(feature = "small_n_result_collectors"))]
        if max_qty <= MAX_VEC_RESULT_SIZE {
            return periodic_nearest_results_inner::<
                Tree,
                A,
                T,
                SS,
                LS,
                D,
                ThresholdVecResultCollection<QueryResultItem<(), T, D::Output>>,
                EXCLUSIVE,
                K,
                B,
            >(tree, query, box_size, max_dist, max_qty, sorted, None);
        }
    }

    periodic_nearest_results_inner::<
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        BinaryHeapResultCollection<QueryResultItem<(), T, D::Output>>,
        EXCLUSIVE,
        K,
        B,
    >(tree, query, box_size, max_dist, max_qty, sorted, None)
}

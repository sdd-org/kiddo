mod simd;

use std::ptr::NonNull;

use crate::dist::DistanceMetric;
use crate::kd_tree::query_context::QueryContext;
use crate::kd_tree::query_stack::{
    ScalarContinuationFar, ScalarContinuationFarStack, ScalarContinuationRestore,
    ScalarContinuationRestoreStack, ScalarStackContext, StackTrait,
};
use crate::kd_tree::{KdTreeAccessor, StemLeafResolution};
use crate::stem_strategy::{
    donnelly::simd_full::{BacktrackBlock3, BacktrackBlock4},
    SimdPrune, SimdSelectBestChildBlock3,
};
use crate::traits::stem_strategy::PreparedBlockQuery;
use crate::{Axis, Content, LeafStrategy, StemStrategy};

#[derive(Clone, Copy, Debug)]
struct Block3PendingSelection<O> {
    child_idx: u8,
    remaining_mask: u8,
    child_rd: O,
    child_off: O,
}

#[inline(always)]
fn select_block3_pending_child<O>(
    rd_values: &[O; 8],
    new_off_values: &[O; 8],
    candidate_mask: u8,
) -> Option<Block3PendingSelection<O>>
where
    O: Axis<Coord = O> + SimdSelectBestChildBlock3,
{
    if candidate_mask == 0 {
        return None;
    }

    let child_idx =
        O::simd_select_best_child_block3(rd_values, candidate_mask).unwrap_or_else(|| unsafe {
            debug_assert!(false, "candidate_mask != 0");
            core::hint::unreachable_unchecked()
        });

    Some(Block3PendingSelection {
        child_idx,
        remaining_mask: candidate_mask & !(1u8 << child_idx),
        child_rd: unsafe { *rd_values.get_unchecked(child_idx as usize) },
        child_off: unsafe { *new_off_values.get_unchecked(child_idx as usize) },
    })
}

#[inline(always)]
fn rebuild_interval_offs<O, const K: usize>(
    query_wide: &[O; K],
    lower: &[O; K],
    upper: &[O; K],
) -> [O; K]
where
    O: Axis<Coord = O>,
{
    let mut off = [O::zero(); K];
    for dim in 0..K {
        off[dim] = crate::stem_strategy::donnelly::simd_full::interval_distance_1d(
            query_wide[dim],
            lower[dim],
            upper[dim],
        );
    }
    off
}

#[cfg(feature = "test_utils")]
#[inline]
fn force_mapped_simd_block_step() -> bool {
    matches!(
        std::env::var("KIDDO_FORCE_MAPPED_SIMD_BLOCK_STEP")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

#[cfg(not(feature = "test_utils"))]
#[inline]
fn force_mapped_simd_block_step() -> bool {
    false
}

#[cfg(feature = "cargo_asm")]
pub mod cargo_asm {
    use super::select_block3_pending_child;

    /// Hook for cargo-asm to render the exact Block3 pending prune/select kernel directly.
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn donnelly_block3_pending_select_f64_cargo_asm_hook(
        rd_values: &[f64; 8],
        new_off_values: &[f64; 8],
        candidate_mask: u8,
    ) -> Option<(u8, u8, f64, f64)> {
        select_block3_pending_child(rd_values, new_off_values, candidate_mask).map(|selection| {
            (
                selection.child_idx,
                selection.remaining_mask,
                selection.child_rd,
                selection.child_off,
            )
        })
    }
}

impl<Tree, A, T, SS, LS, const K: usize, const B: usize> KdTreeQueryOps<A, T, SS, LS, K, B> for Tree
where
    Tree: KdTreeAccessor<A, T, SS, LS, K, B>,
    A: Axis<Coord = A>,
    T: Content,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
{
}

#[allow(missing_docs, private_bounds)]
pub trait KdTreeQueryOps<A, T, SS, LS, const K: usize, const B: usize>:
    KdTreeAccessor<A, T, SS, LS, K, B> + Sized
where
    A: Axis<Coord = A>,
    T: Content,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
{
    #[inline]
    fn get_leaf_idx(&self, query: &[A; K]) -> usize {
        if self.stem_leaf_resolution().uses_arithmetic() {
            self.get_leaf_idx_unmapped(query)
        } else {
            self.get_leaf_idx_mapped(query)
        }
    }

    /// Non-backtracking query
    #[inline]
    fn straight_query<QC, O>(&self, query_ctx: QC, mut process_leaf: impl FnMut(usize))
    where
        QC: QueryContext<A, O, K>,
    {
        let leaf_idx = self.get_leaf_idx(query_ctx.query());

        tracing::trace!(%leaf_idx, "processing leaf");
        process_leaf(leaf_idx);
    }

    /// Get the leaf index for a query (unmapped leaves)
    #[inline]
    fn get_leaf_idx_unmapped(&self, query: &[A; K]) -> usize {
        SS::get_leaf_idx(self.stems(), query, self.max_stem_level())
    }

    /// Get the leaf index for a query (mapped leaves)
    #[inline(always)]
    fn get_leaf_idx_mapped(&self, query: &[A; K]) -> usize {
        let stems_ptr = NonNull::new(self.stems().as_ptr() as *mut u8).unwrap();
        let mut stem_strat: SS = SS::new(stems_ptr);

        while stem_strat.level() <= self.max_stem_level() {
            if let Some(leaf_idx) = self.resolve_terminal_stem(stem_strat.stem_idx()) {
                return leaf_idx;
            }

            let pivot = unsafe { self.stems().get_unchecked(stem_strat.stem_idx()) };
            let is_right_child: bool =
                *unsafe { query.get_unchecked(stem_strat.dim::<K>()) } >= *pivot;
            stem_strat.traverse::<A, K>(is_right_child);
        }

        self.stem_leaf_resolution()
            .resolve_terminal_stem_idx(stem_strat.stem_idx(), stem_strat.leaf_idx())
    }

    // TODO: don't like this structure
    /// Check if a stem points directly to a leaf
    #[inline(always)]
    fn resolve_terminal_stem(&self, stem_idx: usize) -> Option<usize> {
        if self.stem_leaf_resolution().is_terminal_stem_idx(stem_idx) {
            Some(
                self.stem_leaf_resolution()
                    .resolve_terminal_stem_idx(stem_idx, 0),
            )
        } else {
            None
        }
    }

    // #[allow(unused)]
    // #[inline(always)]
    // fn subtree_may_contain_leaf(
    //     &self,
    //     stem_idx: usize,
    //     level: i32,
    //     leaf_idx_prefix: usize,
    // ) -> bool {
    //     let _ = (level, leaf_idx_prefix);
    //     stem_idx < self.stems().len() || self.stem_leaf_resolution().is_terminal_stem_idx(stem_idx)
    // }

    /// Backtracking query
    #[inline(always)]
    fn backtracking_query<QC, O, D>(
        &self,
        query_ctx: &mut QC,
        process_leaf: impl FnMut(usize, &[O; K], &mut QC),
    ) where
        QC: QueryContext<A, O, K>,
        O: Axis<Coord = O>
            + SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4,
        D: DistanceMetric<A, Output = O>,
        SS::Stack<O, K>: StackTrait<O, SS, K> + Default,
    {
        if self.stem_leaf_resolution().uses_arithmetic() && SS::SUPPORTS_ARITHMETIC_LEAF_RESOLUTION
        {
            self.arithmetic_query::<QC, O, D>(query_ctx, process_leaf);
            return;
        }

        let mut stack = SS::Stack::<O, K>::default();
        self.backtracking_query_with_scratch::<QC, O, D>(query_ctx, &mut stack, process_leaf);
    }

    /// Backtracking query with explicit stack
    #[inline(always)]
    fn backtracking_query_with_scratch<QC, O, D>(
        &self,
        query_ctx: &mut QC,
        stack: &mut SS::Stack<O, K>,
        process_leaf: impl FnMut(usize, &[O; K], &mut QC),
    ) where
        QC: QueryContext<A, O, K>,
        O: Axis<Coord = O>
            + SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4,
        D: DistanceMetric<A, Output = O>,
        SS::Stack<O, K>: StackTrait<O, SS, K>,
    {
        if self.stem_leaf_resolution().uses_arithmetic() && SS::SUPPORTS_ARITHMETIC_LEAF_RESOLUTION
        {
            self.arithmetic_query_with_scratch::<QC, O, D>(query_ctx, stack, process_leaf);
            return;
        }

        SS::backtracking_query_with_scratch::<Self, A, T, O, D, QC, LS, K, B>(
            self,
            query_ctx,
            stack,
            process_leaf,
        );
    }

    /// Implementation of backtracking query with scalar stack.
    /// Called by default StemStrategy::backtracking_query_with_scratch implementation.
    /// SIMD strategies override the trait method with custom implementations.
    #[inline(always)]
    fn backtracking_query_with_scratch_impl<QC, O, D>(
        &self,
        query_ctx: &mut QC,
        stack: &mut SS::Stack<O, K>,
        mut process_leaf: impl FnMut(usize, &[O; K], &mut QC),
    ) where
        QC: QueryContext<A, O, K>,
        O: Axis<Coord = O> + BacktrackBlock3 + BacktrackBlock4,
        D: DistanceMetric<A, Output = O>,
        SS::Stack<O, K>: StackTrait<O, SS, K>,
        SS::StackContext<O, K>: ScalarStackContext<O, SS::DeferredState>,
    {
        let stems_ptr = NonNull::new(self.stems().as_ptr() as *mut u8).unwrap();
        let mut stem_strat: SS = SS::new(stems_ptr);

        let query: [A; K] = *query_ctx.query();
        let mut query_wide: [O; K] = [O::zero(); K];
        for dim in 0..K {
            query_wide[dim] = D::widen_coord(query[dim]);
        }

        let mut off = [O::zero(); K];
        stack.push(SS::StackContext::<O, K>::from_parts(
            stem_strat.deferred_state(),
            O::zero(),
            O::zero(),
        ));
        #[cfg(feature = "result_collection_stats")]
        crate::results::result_collection_stats::record_query_stack_push();

        while let Some(stack_ctx) = stack.pop() {
            #[cfg(feature = "result_collection_stats")]
            crate::results::result_collection_stats::record_query_stack_pop();
            #[cfg(any(feature = "exact_query_stats", feature = "test_utils"))]
            crate::results::exact_query_stats::record_scalar_stack_pop();

            let (stem_state, restore_dim, old_off, rd) =
                SS::StackContext::<O, K>::into_parts_with_restore_dim(stack_ctx);
            stem_strat.rehydrate_deferred_state(stem_state);
            let mut dim = stem_strat.dim::<K>();
            let restore_dim = restore_dim.unwrap_or(dim);
            tracing::trace!(%dim, %old_off, %rd, ?off, "Popped stack context");

            let max_dist = query_ctx.max_dist();
            let rd_vs_max = O::cmp(rd, max_dist);

            // TOOO: investigate into whether prune_on_equal_max_dist can be removed
            let should_prune = rd_vs_max == std::cmp::Ordering::Greater
                || (query_ctx.prune_on_equal_max_dist() && rd_vs_max == std::cmp::Ordering::Equal);
            if should_prune {
                #[cfg(feature = "result_collection_stats")]
                crate::results::result_collection_stats::record_query_prune();
                tracing::trace!(%rd, %max_dist, "SCALAR Prune check: PRUNE");
                continue;
            }
            tracing::trace!(%rd, %max_dist, "SCALAR Prune check: VISIT");

            tracing::trace!(
                "Restoring off[{}]. was {}, now {}",
                restore_dim,
                unsafe { *off.get_unchecked(restore_dim) },
                old_off
            );
            unsafe { *off.get_unchecked_mut(restore_dim) = old_off };

            let best_dist = query_ctx.max_dist();
            if let Some(leaf_idx) = self.traverse_to_leaf::<O, D>(
                &query,
                &query_wide,
                &mut stem_strat,
                &mut off,
                &mut dim,
                rd,
                best_dist,
                stack,
            ) {
                tracing::trace!(%leaf_idx, "processing leaf");
                process_leaf(leaf_idx, &query_wide, query_ctx);
            }
        }
    }

    /// Arithmetic-resolution backtracking query.
    #[inline(always)]
    fn arithmetic_query<QC, O, D>(
        &self,
        query_ctx: &mut QC,
        process_leaf: impl FnMut(usize, &[O; K], &mut QC),
    ) where
        QC: QueryContext<A, O, K>,
        O: Axis<Coord = O>
            + SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4,
        D: DistanceMetric<A, Output = O>,
        SS::Stack<O, K>: StackTrait<O, SS, K> + Default,
    {
        let mut stack = SS::Stack::<O, K>::default();
        self.arithmetic_query_with_scratch::<QC, O, D>(query_ctx, &mut stack, process_leaf);
    }

    /// Arithmetic-resolution backtracking query with explicit stack.
    #[inline(always)]
    fn arithmetic_query_with_scratch<QC, O, D>(
        &self,
        query_ctx: &mut QC,
        stack: &mut SS::Stack<O, K>,
        process_leaf: impl FnMut(usize, &[O; K], &mut QC),
    ) where
        QC: QueryContext<A, O, K>,
        O: Axis<Coord = O>
            + SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4,
        D: DistanceMetric<A, Output = O>,
        SS::Stack<O, K>: StackTrait<O, SS, K>,
        SS::StackContext<O, K>: ScalarStackContext<O, SS::DeferredState>,
    {
        SS::arithmetic_query_with_scratch::<Self, A, T, O, D, QC, LS, K, B>(
            self,
            query_ctx,
            stack,
            process_leaf,
        );
    }

    /// Implementation of arithmetic-resolution query with scalar stack.
    /// Called by default StemStrategy::arithmetic_query_with_scratch implementation.
    #[inline(always)]
    fn arithmetic_query_with_scratch_impl<QC, O, D>(
        &self,
        query_ctx: &mut QC,
        stack: &mut SS::Stack<O, K>,
        mut process_leaf: impl FnMut(usize, &[O; K], &mut QC),
    ) where
        QC: QueryContext<A, O, K>,
        O: Axis<Coord = O> + BacktrackBlock3 + BacktrackBlock4,
        D: DistanceMetric<A, Output = O>,
        SS::Stack<O, K>: StackTrait<O, SS, K>,
    {
        if self.size() == 0 {
            return;
        }

        let stems_ptr = NonNull::new(self.stems().as_ptr() as *mut u8).unwrap();
        let mut stem_strat: SS = SS::new(stems_ptr);

        let query: [A; K] = *query_ctx.query();
        let prepared_block_query = SS::prepare_block_query(&query);
        let mut query_wide: [O; K] = [O::zero(); K];
        for dim in 0..K {
            query_wide[dim] = D::widen_coord(query[dim]);
        }

        let mut off = [O::zero(); K];
        stack.clear();
        let max_query_depth = self.max_stem_level() + 1;
        debug_assert!(
            max_query_depth <= 64,
            "scalar continuation stack capacity exceeded: depth={} capacity=64",
            max_query_depth
        );
        let mut restore_stack = ScalarContinuationRestoreStack::<O>::default();
        let mut far_stack = ScalarContinuationFarStack::<O, SS::DeferredState>::default();

        let mut rd = O::zero();
        let leaf_idx = if query_ctx.initial_bound_is_unbounded() {
            self.arithmetic_descend_to_leaf::<QC, O, D, false, false>(
                &query,
                &prepared_block_query,
                &query_wide,
                query_ctx,
                &mut stem_strat,
                &off,
                rd,
                &mut restore_stack,
                &mut far_stack,
            )
        } else {
            self.arithmetic_descend_to_leaf::<QC, O, D, true, false>(
                &query,
                &prepared_block_query,
                &query_wide,
                query_ctx,
                &mut stem_strat,
                &off,
                rd,
                &mut restore_stack,
                &mut far_stack,
            )
        };
        process_leaf(leaf_idx, &query_wide, query_ctx);

        let mut dim = stem_strat.dim::<K>();
        'query: loop {
            while let Some(frame) = restore_stack.pop() {
                #[cfg(feature = "exact_query_stats")]
                crate::results::exact_query_stats::record_continuation_frame_pop();
                #[cfg(feature = "result_collection_stats")]
                crate::results::result_collection_stats::record_query_scalar_continuation_frame_pop(
                );

                let restore_dim = if dim == 0 { K - 1 } else { dim - 1 };
                let old_off = frame.old_off;
                unsafe { *off.get_unchecked_mut(restore_dim) = old_off };
                dim = restore_dim;

                if !frame.has_far {
                    continue;
                }

                let far = far_stack
                    .pop()
                    .expect("scalar continuation far stack underflow");

                #[cfg(feature = "exact_query_stats")]
                crate::results::exact_query_stats::record_far_recheck();

                #[cfg(feature = "result_collection_stats")]
                crate::results::result_collection_stats::record_query_scalar_continuation_far_recheck();

                let rd_vs_max = O::cmp(far.rd, query_ctx.max_dist());
                let should_prune = rd_vs_max == std::cmp::Ordering::Greater
                    || (query_ctx.prune_on_equal_max_dist()
                        && rd_vs_max == std::cmp::Ordering::Equal);
                if should_prune {
                    #[cfg(feature = "exact_query_stats")]
                    crate::results::exact_query_stats::record_far_reject_after_near();
                    #[cfg(feature = "result_collection_stats")]
                    {
                        crate::results::result_collection_stats::record_query_prune();
                        crate::results::result_collection_stats::record_query_scalar_continuation_far_reject_after_near();
                    }
                    continue;
                }

                restore_stack
                    .push_unchecked_inline(ScalarContinuationRestore::restore_only(old_off));
                #[cfg(feature = "exact_query_stats")]
                crate::results::exact_query_stats::record_continuation_frame_push();
                #[cfg(feature = "result_collection_stats")]
                crate::results::result_collection_stats::record_query_scalar_continuation_frame_push();

                unsafe { *off.get_unchecked_mut(restore_dim) = far.far_off };
                stem_strat.rehydrate_deferred_state(far.stem_state);
                rd = far.rd;

                #[cfg(feature = "exact_query_stats")]
                crate::results::exact_query_stats::record_far_enter();

                #[cfg(feature = "result_collection_stats")]
                crate::results::result_collection_stats::record_query_scalar_continuation_far_enter(
                );

                let leaf_idx = self.arithmetic_descend_to_leaf::<QC, O, D, true, true>(
                    &query,
                    &prepared_block_query,
                    &query_wide,
                    query_ctx,
                    &mut stem_strat,
                    &off,
                    rd,
                    &mut restore_stack,
                    &mut far_stack,
                );
                process_leaf(leaf_idx, &query_wide, query_ctx);
                dim = stem_strat.dim::<K>();

                continue 'query;
            }

            break;
        }
    }

    #[allow(private_interfaces)]
    #[inline(always)]
    fn arithmetic_descend_to_leaf<
        QC,
        O,
        D,
        const CHECK_BOUND: bool,
        const RESYNC_BACKTRACKED: bool,
    >(
        &self,
        query: &[A; K],
        prepared_block_query: &PreparedBlockQuery<A, K>,
        query_wide: &[O; K],
        query_ctx: &QC,
        stem_strat: &mut SS,
        off: &[O; K],
        rd: O,
        restore_stack: &mut ScalarContinuationRestoreStack<O>,
        far_stack: &mut ScalarContinuationFarStack<O, SS::DeferredState>,
    ) -> usize
    where
        QC: QueryContext<A, O, K>,
        O: Axis<Coord = O> + BacktrackBlock3 + BacktrackBlock4,
        D: DistanceMetric<A, Output = O>,
    {
        macro_rules! descend_one_with_direction {
            ($dim:expr, $pivot:expr, $is_right_child:expr, $branch_method:ident, $traverse_method:ident) => {{
                #[cfg(feature = "result_collection_stats")]
                crate::results::result_collection_stats::record_query_scalar_traverse_step();

                #[cfg(feature = "result_collection_stats")]
                {
                    let rd_from_off = D::rect_dist_from_off(off);
                    crate::results::result_collection_stats::record_query_scalar_rd_off_check(
                        O::cmp(rd_from_off, rd),
                    );
                }

                let dim = $dim;
                let pivot = $pivot;
                #[cfg(feature = "exact_query_stats")]
                crate::results::exact_query_stats::record_stem_step(pivot < A::max_value());
                if pivot < A::max_value() {
                    let is_right_child = $is_right_child;
                    let far_ctx = stem_strat.$branch_method::<A, K>(is_right_child);

                    let pivot_wide = D::widen_coord(pivot);
                    let query_elem_wide = unsafe { *query_wide.get_unchecked(dim) };
                    let new_off = O::saturating_dist(query_elem_wide, pivot_wide);
                    let old_off = unsafe { *off.get_unchecked(dim) };
                    let rd_far = D::rect_dist_after_update(rd, off, dim, new_off);
                    let far_is_candidate = if CHECK_BOUND {
                        O::cmp(rd_far, query_ctx.max_dist()) != std::cmp::Ordering::Greater
                    } else {
                        true
                    };

                    if far_is_candidate {
                        #[cfg(feature = "exact_query_stats")]
                        crate::results::exact_query_stats::record_initial_far_candidate();
                        #[cfg(feature = "result_collection_stats")]
                        crate::results::result_collection_stats::record_query_scalar_far_child_candidate();
                        restore_stack
                            .push_unchecked_inline(ScalarContinuationRestore::with_far(old_off));
                        far_stack.push_unchecked_inline(ScalarContinuationFar {
                            stem_state: far_ctx.deferred_state(),
                            far_off: new_off,
                            rd: rd_far,
                        });
                    } else {
                        #[cfg(feature = "exact_query_stats")]
                        crate::results::exact_query_stats::record_initial_far_reject();
                        #[cfg(feature = "result_collection_stats")]
                        crate::results::result_collection_stats::record_query_scalar_far_child_reject();
                        restore_stack
                            .push_unchecked_inline(ScalarContinuationRestore::restore_only(old_off));
                    }
                    #[cfg(feature = "result_collection_stats")]
                    crate::results::result_collection_stats::record_query_scalar_continuation_frame_push();
                } else {
                    let old_off = unsafe { *off.get_unchecked(dim) };
                    restore_stack
                        .push_unchecked_inline(ScalarContinuationRestore::restore_only(old_off));
                    #[cfg(feature = "result_collection_stats")]
                    crate::results::result_collection_stats::record_query_scalar_continuation_frame_push();
                    stem_strat.$traverse_method::<A, K>(false);
                }
                #[cfg(feature = "exact_query_stats")]
                crate::results::exact_query_stats::record_continuation_frame_push();
            }};
        }

        macro_rules! descend_one {
            ($branch_method:ident, $traverse_method:ident) => {{
                let dim = stem_strat.dim::<K>();
                let pivot = unsafe { *self.stems().get_unchecked(stem_strat.stem_idx()) };
                let query_elem = unsafe { *query.get_unchecked(dim) };
                descend_one_with_direction!(
                    dim,
                    pivot,
                    query_elem >= pivot,
                    $branch_method,
                    $traverse_method
                );
            }};
        }

        macro_rules! descend_selected {
            ($child_idx:ident, $bit:expr, $branch_method:ident, $traverse_method:ident) => {{
                let dim = stem_strat.dim::<K>();
                let pivot = unsafe { *self.stems().get_unchecked(stem_strat.stem_idx()) };
                descend_one_with_direction!(
                    dim,
                    pivot,
                    $child_idx & $bit != 0,
                    $branch_method,
                    $traverse_method
                );
            }};
        }

        // Only a deferred far child can resume part-way through a Donnelly
        // block. `RESYNC_BACKTRACKED` is false for initial descent, so
        // monomorphisation removes this entire test and loop from that hot path.
        // Once this loop reaches a block root, the unrolled body below proceeds
        // without any further phase checks.
        if RESYNC_BACKTRACKED && (SS::USES_UNROLLED_SCALAR_TRAVERSAL || SS::USES_SIMD_BLOCK_DESCENT)
        {
            while stem_strat.level() <= self.max_stem_level()
                && !(stem_strat.level() as usize).is_multiple_of(SS::BLOCK_SIZE)
            {
                descend_one!(branch_relative, traverse);
            }
        }

        // SIMD descent chooses the terminal child of a complete minor triangle
        // in one comparison. Replay only the selected root-to-terminal path so
        // exact search still creates precisely the ordinary scalar far-child
        // continuations, in the ordinary near-first order.
        if SS::USES_SIMD_BLOCK_DESCENT
            && (!RESYNC_BACKTRACKED || SS::SIMD_BLOCK_DESCENT_ON_BACKTRACK)
        {
            if SS::BLOCK_SIZE == 4 {
                while stem_strat.level() + 3 <= self.max_stem_level() {
                    let dim = stem_strat.dim::<K>();
                    let child_idx = stem_strat.select_prepared_block_child(
                        self.stems(),
                        query,
                        prepared_block_query,
                        dim,
                    );

                    descend_selected!(child_idx, 0b1000, branch_relative_head, traverse_head);
                    descend_selected!(child_idx, 0b0100, branch_relative_head, traverse_head);
                    descend_selected!(child_idx, 0b0010, branch_relative_head, traverse_head);
                    descend_selected!(child_idx, 0b0001, branch_relative_tail, traverse_tail);
                }
            } else if SS::BLOCK_SIZE == 3 {
                while stem_strat.level() + 2 <= self.max_stem_level() {
                    let dim = stem_strat.dim::<K>();
                    let child_idx = stem_strat.select_prepared_block_child(
                        self.stems(),
                        query,
                        prepared_block_query,
                        dim,
                    );

                    descend_selected!(child_idx, 0b100, branch_relative_head, traverse_head);
                    descend_selected!(child_idx, 0b010, branch_relative_head, traverse_head);
                    descend_selected!(child_idx, 0b001, branch_relative_tail, traverse_tail);
                }
            }
        }

        if SS::USES_UNROLLED_SCALAR_TRAVERSAL {
            if SS::BLOCK_SIZE == 7 {
                while stem_strat.level() + 6 <= self.max_stem_level() {
                    descend_one!(branch_relative_head, traverse_head);
                    descend_one!(branch_relative_head, traverse_head);
                    descend_one!(branch_relative_head, traverse_head);
                    descend_one!(branch_relative_head, traverse_head);
                    descend_one!(branch_relative_head, traverse_head);
                    descend_one!(branch_relative_head, traverse_head);
                    descend_one!(branch_relative_tail, traverse_tail);
                }
            } else if SS::BLOCK_SIZE == 6 {
                while stem_strat.level() + 5 <= self.max_stem_level() {
                    descend_one!(branch_relative_head, traverse_head);
                    descend_one!(branch_relative_head, traverse_head);
                    descend_one!(branch_relative_head, traverse_head);
                    descend_one!(branch_relative_head, traverse_head);
                    descend_one!(branch_relative_head, traverse_head);
                    descend_one!(branch_relative_tail, traverse_tail);
                }
            } else if SS::BLOCK_SIZE == 5 {
                while stem_strat.level() + 4 <= self.max_stem_level() {
                    descend_one!(branch_relative_head, traverse_head);
                    descend_one!(branch_relative_head, traverse_head);
                    descend_one!(branch_relative_head, traverse_head);
                    descend_one!(branch_relative_head, traverse_head);
                    descend_one!(branch_relative_tail, traverse_tail);
                }
            } else if SS::BLOCK_SIZE == 4 {
                while stem_strat.level() + 3 <= self.max_stem_level() {
                    descend_one!(branch_relative_head, traverse_head);
                    descend_one!(branch_relative_head, traverse_head);
                    descend_one!(branch_relative_head, traverse_head);
                    descend_one!(branch_relative_tail, traverse_tail);
                }
            } else if SS::BLOCK_SIZE == 3 {
                while stem_strat.level() + 2 <= self.max_stem_level() {
                    descend_one!(branch_relative_head, traverse_head);
                    descend_one!(branch_relative_head, traverse_head);
                    descend_one!(branch_relative_tail, traverse_tail);
                }
            }
        }

        while stem_strat.level() <= self.max_stem_level() {
            descend_one!(branch_relative, traverse);
        }

        let leaf_idx = stem_strat.leaf_idx();
        debug_assert!(
            leaf_idx < self.leaf_count(),
            "arithmetic query resolved invalid leaf_idx={} leaf_count={}",
            leaf_idx,
            self.leaf_count()
        );
        leaf_idx
    }

    /// traverse to leaf
    #[inline(always)]
    fn traverse_to_leaf<O, D>(
        &self,
        query: &[A; K],
        query_wide: &[O; K],
        stem_strat: &mut SS,
        off: &mut [O; K],
        dim: &mut usize,
        rd: O,
        best_dist: O,
        stack: &mut SS::Stack<O, K>,
    ) -> Option<usize>
    where
        O: Axis<Coord = O> + BacktrackBlock3 + BacktrackBlock4,
        D: DistanceMetric<A, Output = O>,
        SS::Stack<O, K>: StackTrait<O, SS, K>,
        SS::StackContext<O, K>: ScalarStackContext<O, SS::DeferredState>,
    {
        loop {
            // Check if current stem points directly to a leaf
            // For Immutable trees, this should optimise away since resolve_terminal_stem_idx
            // will always return None
            if let Some(leaf_idx) = self.resolve_terminal_stem(stem_strat.stem_idx()) {
                return Some(leaf_idx);
            }

            // Delegate to stem strategy for traversal step
            // Default impl does level-by-level, block-based strategies do block-at-once
            let should_continue = stem_strat.backtracking_traverse_step::<A, O, D, K>(
                self.stems(),
                query,
                query_wide,
                off,
                dim,
                rd,
                self.max_stem_level(),
                best_dist,
                stack,
            );

            if !should_continue {
                break;
            }
        }

        Some(
            self.stem_leaf_resolution()
                .resolve_terminal_stem_idx(stem_strat.stem_idx(), stem_strat.leaf_idx()),
        )
    }

    /// Implementation of backtracking query with SIMD stack.
    /// Called by DonnellySimdFull's backtracking_query_with_scratch override.
    #[inline(always)]
    fn backtracking_query_with_block3_simd_stack_impl<QC, O, D>(
        &self,
        query_ctx: &mut QC,
        stack: &mut SS::Stack<O, K>,
        mut process_leaf: impl FnMut(usize, &[O; K], &mut QC),
    ) where
        QC: QueryContext<A, O, K>,
        O: Axis<Coord = O>
            + SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4,
        D: DistanceMetric<A, Output = O>,
        SS: StemStrategy + crate::stem_strategy::donnelly::simd_full::DeferredBlockTraversal,
        SS::StackContext<O, K>: crate::kd_tree::query_stack_simd::Block3ExactStackContext<O, SS, K>
            + crate::kd_tree::query_stack_simd::SimdIntervalStackContext<O, SS, K>,
    {
        use crate::kd_tree::query_stack_simd::{
            Block3ExactStackContext, Block3ExactStackContextState,
        };

        let stems_ptr = NonNull::new(self.stems().as_ptr() as *mut u8).unwrap();
        let stem_strat: SS = SS::new(stems_ptr);

        let query: [A; K] = *query_ctx.query();
        let mut query_wide: [O; K] = [O::zero(); K];
        for dim in 0..K {
            query_wide[dim] = D::widen_coord(query[dim]);
        }

        let mut off = [O::zero(); K];
        // Both are restored from the popped entry's snapshot before any read; the root
        // entry carries the unbounded slab, so these initial values are never observed.
        let mut lower: [O; K];
        let mut upper: [O; K];

        stack.push(<SS::StackContext<O, K> as Block3ExactStackContext<
            O,
            SS,
            K,
        >>::new_single(stem_strat));
        #[cfg(feature = "result_collection_stats")]
        crate::results::result_collection_stats::record_query_stack_push();

        while let Some(ctx) = stack.pop() {
            #[cfg(feature = "result_collection_stats")]
            crate::results::result_collection_stats::record_query_stack_pop();
            match <SS::StackContext<O, K> as Block3ExactStackContext<O, SS, K>>::into_block3_exact_state(ctx) {
                Block3ExactStackContextState::Single {
                    stem_strat: mut ss,
                    dim: dim_val,
                    lower_snapshot,
                    upper_snapshot,
                    old_off,
                    rd,
                } => {
                #[cfg(any(feature = "exact_query_stats", feature = "test_utils"))]
                crate::results::exact_query_stats::record_simd_single_pop();

                    let restore_dim = dim_val;
                    let mut dim = ss.dim::<K>();
                    tracing::trace!(
                        %restore_dim,
                        resumed_dim = %dim,
                        %old_off,
                        %rd,
                        ?off,
                        "Popped single context"
                    );

                    let max_dist = query_ctx.max_dist();
                    let rd_vs_max = O::cmp(rd, max_dist);
                    let should_prune = rd_vs_max == std::cmp::Ordering::Greater
                        || (query_ctx.prune_on_equal_max_dist()
                            && rd_vs_max == std::cmp::Ordering::Equal);
                    if should_prune {
                        #[cfg(feature = "result_collection_stats")]
                        crate::results::result_collection_stats::record_query_prune();
                        tracing::trace!(%rd, %max_dist, "Prune check: PRUNE");
                        continue;
                    }

                    lower = lower_snapshot;
                    upper = upper_snapshot;
                    off = rebuild_interval_offs(&query_wide, &lower, &upper);
                    unsafe {
                        *off.get_unchecked_mut(restore_dim) = old_off;
                    }

                    let best_dist = query_ctx.max_dist();
                    if let Some(leaf_idx) = self.traverse_to_leaf_simd::<O, D>(
                        &query,
                        &query_wide,
                        &mut ss,
                        &mut lower,
                        &mut upper,
                        &mut off,
                        &mut dim,
                        rd,
                        best_dist,
                        stack,
                    ) {
                        tracing::trace!(%leaf_idx, "processing leaf");
                        process_leaf(leaf_idx, &query_wide, query_ctx);
                    }
                }
                Block3ExactStackContextState::Block3Pending {
                    base,
                    pending_mask,
                    rd,
                    lower: restored_lower,
                    upper: restored_upper,
                } => {
                    #[cfg(any(feature = "exact_query_stats", feature = "test_utils"))]
                    crate::results::exact_query_stats::record_block3_pending_pop(pending_mask);

                    let dim_val = base.dim::<K>();
                    lower = restored_lower;
                    upper = restored_upper;
                    off = rebuild_interval_offs(&query_wide, &lower, &upper);
                    let old_off = off[dim_val];
                    let lower_bound = lower[dim_val];
                    let upper_bound = upper[dim_val];

                    tracing::trace!(%dim_val, %old_off, %rd, %pending_mask, "Popped Block3 pending context");

                    #[cfg(feature = "test_utils")]
                    let trace_enabled =
                        crate::test_utils::exact_query_trace::enabled()
                            && std::any::type_name::<O>() == "f64";

                    #[cfg(not(feature = "test_utils"))]
                    let trace_enabled = false;

                    let best_dist = query_ctx.max_dist();
                    let query_wide_val = query_wide[dim_val];
                    let (
                        candidate_mask,
                        selection,
                        trace_pending_arrays,
                    ) = if trace_enabled {
                        let mut rd_values = [O::zero(); 8];
                        let mut new_off_values = [O::zero(); 8];
                        let mut lower_bounds = [O::zero(); 8];
                        let mut upper_bounds = [O::zero(); 8];
                        let candidate_mask = base
                            .fill_block3_pending_values::<A, O, D, K>(
                                self.stems(),
                                query_wide_val,
                                lower_bound,
                                upper_bound,
                                old_off,
                                rd,
                                best_dist,
                                &mut new_off_values,
                                &mut rd_values,
                                &mut lower_bounds,
                                &mut upper_bounds,
                            )
                            & pending_mask;
                        let selection = select_block3_pending_child(
                            &rd_values,
                            &new_off_values,
                            candidate_mask,
                        )
                        .map(|selection| {
                            (
                                selection.child_idx,
                                selection.remaining_mask,
                                selection.child_rd,
                                selection.child_off,
                                lower_bounds[selection.child_idx as usize],
                                upper_bounds[selection.child_idx as usize],
                            )
                        });
                        (
                            candidate_mask,
                            selection,
                            Some((rd_values, new_off_values, lower_bounds, upper_bounds)),
                        )
                    } else {
                        let candidate_mask = base
                            .backtrack_block3_pending_mask::<A, O, D, K>(
                                self.stems(),
                                query_wide_val,
                                lower_bound,
                                upper_bound,
                                old_off,
                                rd,
                                best_dist,
                            )
                            & pending_mask;

                        let selection = if candidate_mask == 0 {
                            None
                        } else if candidate_mask.is_power_of_two() {
                            let child_idx = candidate_mask.trailing_zeros() as u8;
                            let (child_rd, child_off, child_lower_bound, child_upper_bound) = base
                                .selected_block3_pending_child_state::<A, O, D>(
                                    self.stems(),
                                    child_idx,
                                    query_wide_val,
                                    lower_bound,
                                    upper_bound,
                                    old_off,
                                    rd,
                                );
                            Some((
                                child_idx,
                                candidate_mask & !(1u8 << child_idx),
                                child_rd,
                                child_off,
                                child_lower_bound,
                                child_upper_bound,
                            ))
                        } else {
                            let mut rd_values = [O::zero(); 8];
                            let mut new_off_values = [O::zero(); 8];
                            let mut lower_bounds = [O::zero(); 8];
                            let mut upper_bounds = [O::zero(); 8];
                            let candidate_mask = base
                                .fill_block3_pending_values::<A, O, D, K>(
                                    self.stems(),
                                    query_wide_val,
                                    lower_bound,
                                    upper_bound,
                                    old_off,
                                    rd,
                                    best_dist,
                                    &mut new_off_values,
                                    &mut rd_values,
                                    &mut lower_bounds,
                                    &mut upper_bounds,
                                )
                                & candidate_mask;
                            let selection = select_block3_pending_child(
                                &rd_values,
                                &new_off_values,
                                candidate_mask,
                            )
                            .map(|selection| {
                                let child_idx = selection.child_idx as usize;
                                (
                                    selection.child_idx,
                                    selection.remaining_mask,
                                    selection.child_rd,
                                    selection.child_off,
                                    unsafe { *lower_bounds.get_unchecked(child_idx) },
                                    unsafe { *upper_bounds.get_unchecked(child_idx) },
                                )
                            });
                            selection
                        };
                        (candidate_mask, selection, None)
                    };

                    #[cfg(not(feature = "test_utils"))]
                    let _ = (candidate_mask, &trace_pending_arrays);

                    #[cfg(any(feature = "exact_query_stats", feature = "test_utils"))]
                    crate::results::exact_query_stats::record_block3_candidate_mask(
                        candidate_mask,
                    );

                    let Some((
                        selected_child_idx,
                        remaining_mask,
                        selected_child_rd,
                        selected_child_off,
                        selected_lower_bound,
                        selected_upper_bound,
                    )) = selection
                    else {
                        #[cfg(feature = "result_collection_stats")]
                        crate::results::result_collection_stats::record_query_prune();
                        tracing::trace!("All Block3 children pruned");
                        continue;
                    };

                    #[cfg(feature = "test_utils")]
                    {
                        if let Some((rd_values, new_off_values, lower_bounds, upper_bounds)) =
                            trace_pending_arrays
                        {
                            let old_off_f = unsafe { *(&old_off as *const O as *const f64) };
                            let rd_f = unsafe { *(&rd as *const O as *const f64) };
                            let lower_bound_f =
                                unsafe { *(&lower_bound as *const O as *const f64) };
                            let upper_bound_f =
                                unsafe { *(&upper_bound as *const O as *const f64) };
                            let child_off_f =
                                unsafe { *(&selected_child_off as *const O as *const f64) };
                            let child_rd_f =
                                unsafe { *(&selected_child_rd as *const O as *const f64) };
                            let new_off_values_f = unsafe {
                                *(&new_off_values as *const [O; 8] as *const [f64; 8])
                            };
                            let rd_values_f =
                                unsafe { *(&rd_values as *const [O; 8] as *const [f64; 8]) };
                            let lower_bounds_f = unsafe {
                                *(&lower_bounds as *const [O; 8] as *const [f64; 8])
                            };
                            let upper_bounds_f = unsafe {
                                *(&upper_bounds as *const [O; 8] as *const [f64; 8])
                            };
                            crate::test_utils::exact_query_trace::push(
                                crate::test_utils::exact_query_trace::ExactQueryTraceEvent::Block3PendingSelection {
                                    stem_idx: base.stem_idx(),
                                    level: base.level(),
                                    dim: dim_val,
                                    pending_mask,
                                    candidate_mask,
                                    selected_child_idx,
                                    child_off: child_off_f,
                                    child_rd: child_rd_f,
                                    parent_lower_bound: lower_bound_f,
                                    parent_upper_bound: upper_bound_f,
                                    old_off: old_off_f,
                                    rd: rd_f,
                                    new_off_values: new_off_values_f,
                                    rd_values: rd_values_f,
                                    lower_bounds: lower_bounds_f,
                                    upper_bounds: upper_bounds_f,
                                },
                            );
                        }
                    }

                    if remaining_mask != 0 {
                        stack.push(
                            <SS::StackContext<O, K> as Block3ExactStackContext<O, SS, K>>::new_block3_pending_from_state(
                                base,
                                remaining_mask,
                                rd,
                                &lower,
                                &upper,
                            ),
                        );
                        #[cfg(feature = "result_collection_stats")]
                        crate::results::result_collection_stats::record_query_stack_push();
                    }

                    let mut ss = base.block_child::<K>(selected_child_idx);
                    let mut dim = ss.dim::<K>();

                    unsafe {
                        *off.get_unchecked_mut(dim_val) = selected_child_off;
                        *lower.get_unchecked_mut(dim_val) = selected_lower_bound;
                        *upper.get_unchecked_mut(dim_val) = selected_upper_bound;
                    }

                    if let Some(leaf_idx) = self.traverse_to_leaf_simd::<O, D>(
                        &query,
                        &query_wide,
                        &mut ss,
                        &mut lower,
                        &mut upper,
                        &mut off,
                        &mut dim,
                        selected_child_rd,
                        best_dist,
                        stack,
                    ) {
                        tracing::trace!(%leaf_idx, "processing leaf");
                        process_leaf(leaf_idx, &query_wide, query_ctx);
                    }
                }
            }
        }
    }

    /// Implementation of backtracking query with SIMD stack.
    /// Called by DonnellySimdFull's backtracking_query_with_scratch override.
    #[inline(always)]
    fn backtracking_query_with_simd_stack_impl<QC, O, D>(
        &self,
        query_ctx: &mut QC,
        stack: &mut SS::Stack<O, K>,
        mut process_leaf: impl FnMut(usize, &[O; K], &mut QC),
    ) where
        QC: QueryContext<A, O, K>,
        O: Axis<Coord = O>
            + SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4,
        D: DistanceMetric<A, Output = O>,
        SS: StemStrategy<
                StackContext<O, K> = crate::kd_tree::query_stack_simd::SimdQueryStackContext<
                    O,
                    SS,
                    K,
                >,
            > + crate::stem_strategy::donnelly::simd_full::DeferredBlockTraversal,
    {
        use crate::kd_tree::query_stack_simd::SimdQueryStackContext;

        let stems_ptr = NonNull::new(self.stems().as_ptr() as *mut u8).unwrap();
        let stem_strat: SS = SS::new(stems_ptr);

        let query: [A; K] = *query_ctx.query();
        let mut query_wide: [O; K] = [O::zero(); K];
        for dim in 0..K {
            query_wide[dim] = D::widen_coord(query[dim]);
        }

        let mut off = [O::zero(); K];
        // Both are restored from the popped entry's snapshot before any read; the root
        // entry carries the unbounded slab, so these initial values are never observed.
        let mut lower: [O; K];
        let mut upper: [O; K];

        stack.push(SimdQueryStackContext::new_single(stem_strat));
        #[cfg(feature = "result_collection_stats")]
        crate::results::result_collection_stats::record_query_stack_push();

        // Backtracking loop
        while let Some(ctx) = stack.pop() {
            #[cfg(feature = "result_collection_stats")]
            crate::results::result_collection_stats::record_query_stack_pop();
            match ctx {
                SimdQueryStackContext::Single {
                    stem_strat: mut ss,
                    dim: dim_val,
                    lower_snapshot,
                    upper_snapshot,
                    old_off,
                    rd,
                } => {
                    // Single entry - standard scalar processing
                    let restore_dim = dim_val;
                    let mut dim = ss.dim::<K>();
                    tracing::trace!(
                        %restore_dim,
                        resumed_dim = %dim,
                        %old_off,
                        %rd,
                        ?off,
                        "Popped single context"
                    );

                    let max_dist = query_ctx.max_dist();
                    let rd_vs_max = O::cmp(rd, max_dist);
                    // TOOO: investigate into whether prune_on_equal_max_dist can be removed
                    let should_prune = rd_vs_max == std::cmp::Ordering::Greater
                        || (query_ctx.prune_on_equal_max_dist()
                            && rd_vs_max == std::cmp::Ordering::Equal);
                    if should_prune {
                        #[cfg(feature = "result_collection_stats")]
                        crate::results::result_collection_stats::record_query_prune();
                        tracing::trace!(%rd, %max_dist, "Prune check: PRUNE");
                        continue;
                    }
                    tracing::trace!(%rd, %max_dist, "SIMD Prune check: VISIT");

                    tracing::trace!("Restoring interval state for dim {}", restore_dim);
                    lower = lower_snapshot;
                    upper = upper_snapshot;
                    off = rebuild_interval_offs(&query_wide, &lower, &upper);
                    unsafe {
                        *off.get_unchecked_mut(restore_dim) = old_off;
                    }

                    let best_dist = query_ctx.max_dist();
                    if let Some(leaf_idx) = self.traverse_to_leaf_simd::<O, D>(
                        &query,
                        &query_wide,
                        &mut ss,
                        &mut lower,
                        &mut upper,
                        &mut off,
                        &mut dim,
                        rd,
                        best_dist,
                        stack,
                    ) {
                        tracing::trace!(%leaf_idx, "processing leaf");
                        process_leaf(leaf_idx, &query_wide, query_ctx);
                    }
                }
                SimdQueryStackContext::Block3Pending {
                    base,
                    rd_values,
                    new_off_values,
                    lower_bounds,
                    upper_bounds,
                    pending_mask,
                    dim: dim_val,
                    old_off,
                    lower_snapshot,
                    upper_snapshot,
                } => {
                    tracing::trace!(
                        %dim_val,
                        %old_off,
                        ?rd_values,
                        %pending_mask,
                        "Popped Block3 pending context"
                    );
                    unsafe {
                        *off.get_unchecked_mut(dim_val) = old_off;
                    }
                    lower = lower_snapshot;
                    upper = upper_snapshot;
                    off = rebuild_interval_offs(&query_wide, &lower, &upper);

                    let Some(selection) = select_block3_pending_child(
                        &rd_values,
                        &new_off_values,
                        simd::simd_prune_block(&rd_values, query_ctx.max_dist(), pending_mask),
                    ) else {
                        #[cfg(feature = "result_collection_stats")]
                        crate::results::result_collection_stats::record_query_prune();
                        tracing::trace!("All Block3 children pruned");
                        continue;
                    };

                    if selection.remaining_mask != 0 {
                        stack.push(SimdQueryStackContext::new_block3_pending(
                            base,
                            rd_values,
                            new_off_values,
                            lower_bounds,
                            upper_bounds,
                            selection.remaining_mask,
                            dim_val,
                            old_off,
                            lower,
                            upper,
                        ));
                        #[cfg(feature = "result_collection_stats")]
                        crate::results::result_collection_stats::record_query_stack_push();
                    }

                    let mut ss = base.block_child::<K>(selection.child_idx);
                    let mut dim = ss.dim::<K>();

                    unsafe {
                        *off.get_unchecked_mut(dim_val) = selection.child_off;
                        *lower.get_unchecked_mut(dim_val) =
                            lower_bounds[selection.child_idx as usize];
                        *upper.get_unchecked_mut(dim_val) =
                            upper_bounds[selection.child_idx as usize];
                    }

                    let best_dist = query_ctx.max_dist();
                    if let Some(leaf_idx) = self.traverse_to_leaf_simd::<O, D>(
                        &query,
                        &query_wide,
                        &mut ss,
                        &mut lower,
                        &mut upper,
                        &mut off,
                        &mut dim,
                        selection.child_rd,
                        best_dist,
                        stack,
                    ) {
                        tracing::trace!(%leaf_idx, "processing leaf");
                        process_leaf(leaf_idx, &query_wide, query_ctx);
                    }
                }
                SimdQueryStackContext::Block {
                    siblings,
                    rd_values,
                    new_off_values,
                    lower_bounds,
                    upper_bounds,
                    sibling_mask,
                    dim: dim_val,
                    old_off,
                    lower_snapshot,
                    upper_snapshot,
                } => {
                    tracing::trace!(%dim_val, %old_off, ?rd_values, %sibling_mask, "Popped block context");
                    // Restore the parent split-dimension offset captured for this block context.
                    unsafe {
                        *off.get_unchecked_mut(dim_val) = old_off;
                    }
                    lower = lower_snapshot;
                    upper = upper_snapshot;
                    off = rebuild_interval_offs(&query_wide, &lower, &upper);

                    // SIMD pruning: check which siblings pass the backtrack test
                    let max_dist = query_ctx.max_dist();
                    let surviving_mask =
                        simd::simd_prune_block::<O>(&rd_values, max_dist, sibling_mask);

                    if surviving_mask == 0 {
                        #[cfg(feature = "result_collection_stats")]
                        crate::results::result_collection_stats::record_query_prune();
                        tracing::trace!("All siblings pruned");
                        continue;
                    }

                    tracing::trace!(
                        surviving_mask = format!("{:08b}", surviving_mask),
                        "Some siblings survive"
                    );

                    // Save the current off state before processing siblings
                    let saved_off = off;
                    let saved_lower = lower;
                    let saved_upper = upper;

                    // Process each surviving sibling
                    for sibling_idx in 0..8 {
                        if surviving_mask & (1 << sibling_idx) != 0 {
                            let mut ss = siblings[sibling_idx];
                            let rd = rd_values[sibling_idx];
                            let new_off = new_off_values[sibling_idx];
                            let mut dim = ss.dim::<K>();

                            // Restore off array to saved state, then update the split dimension
                            // Use the per-sibling new_off value (e.g., interval distance)
                            off = saved_off;
                            lower = saved_lower;
                            upper = saved_upper;
                            tracing::trace!(
                                "Restoring off[{}]. was {}, now {} (interval dist for sibling {}). Parent dim was {}, sibling dim is {}",
                                dim_val,
                                unsafe { *off.get_unchecked(dim_val) },
                                new_off,
                                sibling_idx,
                                dim_val,
                                dim
                            );
                            // The interval distance was computed for the parent block's split dim.
                            unsafe {
                                *off.get_unchecked_mut(dim_val) = new_off;
                                *lower.get_unchecked_mut(dim_val) = lower_bounds[sibling_idx];
                                *upper.get_unchecked_mut(dim_val) = upper_bounds[sibling_idx];
                            }

                            let best_dist = query_ctx.max_dist();
                            if let Some(leaf_idx) = self.traverse_to_leaf_simd::<O, D>(
                                &query,
                                &query_wide,
                                &mut ss,
                                &mut lower,
                                &mut upper,
                                &mut off,
                                &mut dim,
                                rd,
                                best_dist,
                                stack,
                            ) {
                                tracing::trace!(%leaf_idx, "processing leaf");
                                process_leaf(leaf_idx, &query_wide, query_ctx);
                            }
                        }
                    }
                }
                SimdQueryStackContext::DeferredBlock {
                    base,
                    child_base,
                    rd_values,
                    new_off_values,
                    lower_bounds,
                    upper_bounds,
                    sibling_mask,
                    dim: dim_val,
                    old_off,
                    lower_snapshot,
                    upper_snapshot,
                } => {
                    tracing::trace!(
                        %dim_val,
                        %old_off,
                        ?rd_values,
                        %sibling_mask,
                        %child_base,
                        "Popped deferred block context"
                    );
                    unsafe {
                        *off.get_unchecked_mut(dim_val) = old_off;
                    }
                    lower = lower_snapshot;
                    upper = upper_snapshot;
                    off = rebuild_interval_offs(&query_wide, &lower, &upper);

                    let max_dist = query_ctx.max_dist();
                    let surviving_mask =
                        simd::simd_prune_block::<O>(&rd_values, max_dist, sibling_mask);

                    if surviving_mask == 0 {
                        #[cfg(feature = "result_collection_stats")]
                        crate::results::result_collection_stats::record_query_prune();
                        tracing::trace!("All deferred block siblings pruned");
                        continue;
                    }

                    let saved_off = off;
                    let saved_lower = lower;
                    let saved_upper = upper;

                    for sibling_idx in 0..8 {
                        if surviving_mask & (1 << sibling_idx) != 0 {
                            let mut ss = base.block_child::<K>(child_base + sibling_idx as u8);
                            let rd = rd_values[sibling_idx];
                            let new_off = new_off_values[sibling_idx];
                            let mut dim = ss.dim::<K>();

                            off = saved_off;
                            lower = saved_lower;
                            upper = saved_upper;
                            unsafe {
                                *off.get_unchecked_mut(dim_val) = new_off;
                                *lower.get_unchecked_mut(dim_val) = lower_bounds[sibling_idx];
                                *upper.get_unchecked_mut(dim_val) = upper_bounds[sibling_idx];
                            }

                            let best_dist = query_ctx.max_dist();
                            if let Some(leaf_idx) = self.traverse_to_leaf_simd::<O, D>(
                                &query,
                                &query_wide,
                                &mut ss,
                                &mut lower,
                                &mut upper,
                                &mut off,
                                &mut dim,
                                rd,
                                best_dist,
                                stack,
                            ) {
                                tracing::trace!(%leaf_idx, "processing leaf");
                                process_leaf(leaf_idx, &query_wide, query_ctx);
                            }
                        }
                    }
                }
            }
        }
    }

    /// traverse to leaf with SIMD stack
    #[inline(always)]
    fn traverse_to_leaf_simd<O, D>(
        &self,
        query: &[A; K],
        query_wide: &[O; K],
        stem_strat: &mut SS,
        lower: &mut [O; K],
        upper: &mut [O; K],
        off: &mut [O; K],
        dim: &mut usize,
        rd: O,
        best_dist: O,
        stack: &mut SS::Stack<O, K>,
    ) -> Option<usize>
    where
        O: Axis<Coord = O> + SimdSelectBestChildBlock3 + BacktrackBlock3 + BacktrackBlock4,
        D: DistanceMetric<A, Output = O>,
        SS: StemStrategy + crate::stem_strategy::donnelly::simd_full::DeferredBlockTraversal,
        SS::StackContext<O, K>:
            crate::kd_tree::query_stack_simd::SimdIntervalStackContext<O, SS, K>,
    {
        let use_scalar_step = !self.stem_leaf_resolution().uses_arithmetic()
            && SS::BLOCK_SIZE != 3
            && !force_mapped_simd_block_step();

        loop {
            let stem_idx = stem_strat.stem_idx();
            // Check if current stem points directly to a leaf
            if let Some(leaf_idx) = self.resolve_terminal_stem(stem_idx) {
                return Some(leaf_idx);
            }
            let stem_oob = stem_idx >= self.stems().len();

            let should_continue = if use_scalar_step {
                if stem_strat.level() > self.max_stem_level() {
                    false
                } else {
                    let dim_val = *dim;
                    let pivot = if stem_oob {
                        A::max_value()
                    } else {
                        unsafe { *self.stems().get_unchecked(stem_idx) }
                    };

                    if pivot < A::max_value() {
                        let query_elem = unsafe { *query.get_unchecked(dim_val) };
                        let is_right_child = query_elem >= pivot;
                        let far_ctx = stem_strat.branch_relative::<A, K>(is_right_child);

                        let pivot_wide: O = D::widen_coord(pivot);
                        let query_elem_wide = unsafe { *query_wide.get_unchecked(dim_val) };
                        let new_off = O::saturating_dist(query_elem_wide, pivot_wide);

                        let rd_far = D::rect_dist_after_update(rd, off, dim_val, new_off);

                        let old_lower = unsafe { *lower.get_unchecked(dim_val) };
                        let old_upper = unsafe { *upper.get_unchecked(dim_val) };
                        let near_lower;
                        let near_upper;
                        let far_lower;
                        let far_upper;

                        if is_right_child {
                            near_lower = O::max(old_lower, pivot_wide);
                            near_upper = old_upper;
                            far_lower = old_lower;
                            far_upper = if O::cmp(old_upper, pivot_wide) == std::cmp::Ordering::Less
                            {
                                old_upper
                            } else {
                                pivot_wide
                            };
                        } else {
                            near_lower = old_lower;
                            near_upper =
                                if O::cmp(old_upper, pivot_wide) == std::cmp::Ordering::Less {
                                    old_upper
                                } else {
                                    pivot_wide
                                };
                            far_lower = O::max(old_lower, pivot_wide);
                            far_upper = old_upper;
                        }

                        let near_off =
                            crate::stem_strategy::donnelly::simd_full::interval_distance_1d(
                                query_elem_wide,
                                near_lower,
                                near_upper,
                            );

                        if O::cmp(rd_far, best_dist) != std::cmp::Ordering::Greater {
                            // Snapshot every axis' slab, overriding this one with the far
                            // child's: the other axes cannot be rebuilt on pop.
                            let mut far_lower_snapshot = *lower;
                            let mut far_upper_snapshot = *upper;
                            far_lower_snapshot[dim_val] = far_lower;
                            far_upper_snapshot[dim_val] = far_upper;
                            stack.push(<SS::StackContext<O, K> as crate::kd_tree::query_stack_simd::SimdIntervalStackContext<O, SS, K>>::new_single_with_bounds(
                                far_ctx,
                                dim_val,
                                far_lower_snapshot,
                                far_upper_snapshot,
                                new_off,
                                rd_far,
                            ));
                        }

                        unsafe {
                            *lower.get_unchecked_mut(dim_val) = near_lower;
                            *upper.get_unchecked_mut(dim_val) = near_upper;
                            *off.get_unchecked_mut(dim_val) = near_off;
                        }
                    } else {
                        // +Inf pivots can still encode structural branches in left-aligned trees.
                        // Scalar traversal treats these as structural padding and just descends
                        // left without enqueuing a synthetic sibling branch.
                        stem_strat.traverse::<A, K>(false);
                    }

                    *dim = stem_strat.dim::<K>();
                    true
                }
            } else {
                stem_strat.backtracking_traverse_step_with_bounds::<A, O, D, K>(
                    self.stems(),
                    query,
                    query_wide,
                    lower,
                    upper,
                    off,
                    dim,
                    rd,
                    self.max_stem_level(),
                    best_dist,
                    stack,
                )
            };

            tracing::trace!(
                stem_idx = %stem_strat.stem_idx(),
                level = %stem_strat.level(),
                dim = %stem_strat.dim::<K>(),
                "Descended one block"
            );

            if !should_continue {
                break;
            }
        }

        Some(
            self.stem_leaf_resolution()
                .resolve_terminal_stem_idx(stem_strat.stem_idx(), stem_strat.leaf_idx()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::SquaredEuclidean;
    use crate::kd_tree::query_context::QueryContext;
    use crate::kd_tree::query_stack_simd::SimdQueryStackContext;
    use crate::kd_tree::stem_leaf_resolution::OwnedStemLeafResolution;
    use crate::leaf_strategy::DummyLeafStrategy;
    use crate::stem_strategy::donnelly::simd_full::DeferredBlockTraversal;
    use crate::stem_strategy::DonnellySimdFull;
    use nonmax::NonMaxUsize;

    #[test]
    fn test_select_block3_pending_child() {
        let rd_values = [1.0, 2.0, 0.5, 4.0, 5.0, 6.0, 7.0, 8.0];
        let new_off_values = [0.1, 0.2, 0.05, 0.4, 0.5, 0.6, 0.7, 0.8];
        let candidate_mask = 0b111; // 0, 1, 2

        let selection =
            select_block3_pending_child::<f64>(&rd_values, &new_off_values, candidate_mask)
                .unwrap();
        assert_eq!(selection.child_idx, 2);
        assert_eq!(selection.child_rd, 0.5);
        assert_eq!(selection.remaining_mask, 0b011);
    }

    #[test]
    fn test_rebuild_interval_offs() {
        let query = [5.0, 5.0];
        let lower = [0.0, 6.0];
        let upper = [4.0, 10.0];
        let off = rebuild_interval_offs::<f64, 2>(&query, &lower, &upper);

        assert_eq!(off[0], 1.0);
        assert_eq!(off[1], 1.0);
    }

    type TestStemStrategy = DonnellySimdFull<4>;
    type TestLeafStrategy = DummyLeafStrategy;

    struct TestQueryContext {
        query: [f64; 3],
        max_dist: f64,
    }

    impl QueryContext<f64, f64, 3> for TestQueryContext {
        fn query(&self) -> &[f64; 3] {
            &self.query
        }

        fn max_dist(&self) -> f64 {
            self.max_dist
        }
    }

    struct TestTree {
        stems: Vec<f64>,
        resolution: OwnedStemLeafResolution,
    }

    struct Block3TestTree {
        stems: Vec<f64>,
        resolution: OwnedStemLeafResolution,
    }

    impl KdTreeAccessor<f64, u32, TestStemStrategy, TestLeafStrategy, 3, 16> for TestTree {
        fn stems(&self) -> &[f64] {
            &self.stems
        }

        fn leaves(&self) -> &TestLeafStrategy {
            static DUMMY: TestLeafStrategy = DummyLeafStrategy {};
            &DUMMY
        }

        fn stem_leaf_resolution(&self) -> &impl StemLeafResolution {
            &self.resolution
        }

        fn size(&self) -> usize {
            0
        }

        fn max_stem_level(&self) -> i32 {
            0
        }

        fn max_leaf_len(&self) -> usize {
            0
        }
    }

    impl KdTreeAccessor<f64, u32, DonnellySimdFull<3>, TestLeafStrategy, 3, 16> for Block3TestTree {
        fn stems(&self) -> &[f64] {
            &self.stems
        }

        fn leaves(&self) -> &TestLeafStrategy {
            static DUMMY: TestLeafStrategy = DummyLeafStrategy {};
            &DUMMY
        }

        fn stem_leaf_resolution(&self) -> &impl StemLeafResolution {
            &self.resolution
        }

        fn size(&self) -> usize {
            0
        }

        fn max_stem_level(&self) -> i32 {
            0
        }

        fn max_leaf_len(&self) -> usize {
            0
        }
    }

    fn build_root_strat() -> TestStemStrategy {
        TestStemStrategy::new(NonNull::dangling())
    }

    fn build_test_tree_with_terminal_stems(terminal_stems: &[(usize, usize)]) -> TestTree {
        let max_stem_idx = terminal_stems
            .iter()
            .map(|(stem_idx, _)| *stem_idx)
            .max()
            .unwrap_or(0);
        let mut leaf_idx_map = vec![None; max_stem_idx + 1];
        for &(stem_idx, leaf_idx) in terminal_stems {
            leaf_idx_map[stem_idx] = Some(NonMaxUsize::new(leaf_idx).expect("leaf_idx overflow"));
        }

        TestTree {
            stems: Vec::new(),
            resolution: OwnedStemLeafResolution::Mapped {
                min_stem_leaf_idx: 0,
                leaf_idx_map,
            },
        }
    }

    #[test]
    fn test_backtracking_query_with_simd_stack_exercises_block3_pending_arm() {
        let root = build_root_strat();
        let child0 = root.block_child::<3>(0);
        let child1 = root.block_child::<3>(1);
        let child2 = root.block_child::<3>(2);
        let tree = build_test_tree_with_terminal_stems(&[
            (root.stem_idx(), 100),
            (child0.stem_idx(), 200),
            (child1.stem_idx(), 201),
            (child2.stem_idx(), 202),
        ]);

        let mut stack = <TestStemStrategy as StemStrategy>::Stack::<f64, 3>::default();
        let rd_values = [3.0, 1.0, 2.0, 99.0, 99.0, 99.0, 99.0, 99.0];
        let new_off_values = [0.3, 0.1, 0.2, 9.9, 9.9, 9.9, 9.9, 9.9];
        let lower_bounds = [-3.0, -2.0, -1.0, -9.0, -9.0, -9.0, -9.0, -9.0];
        let upper_bounds = [3.0, 2.0, 1.0, 9.0, 9.0, 9.0, 9.0, 9.0];

        stack.push(SimdQueryStackContext::new_block3_pending(
            root,
            rd_values,
            new_off_values,
            lower_bounds,
            upper_bounds,
            0b0000_0111,
            0,
            0.0,
            [f64::NEG_INFINITY; 3],
            [f64::INFINITY; 3],
        ));

        let mut query_ctx = TestQueryContext {
            query: [0.0, 0.0, 0.0],
            max_dist: 10.0,
        };
        let mut visited = Vec::new();

        tree.backtracking_query_with_simd_stack_impl::<_, f64, SquaredEuclidean<f64>>(
            &mut query_ctx,
            &mut stack,
            |leaf_idx, _, _| visited.push(leaf_idx),
        );

        assert_eq!(visited, vec![100, 201, 202, 200]);
    }

    #[test]
    fn test_backtracking_query_with_simd_stack_exercises_block_arm() {
        let root = build_root_strat();
        let siblings = std::array::from_fn(|idx| root.block_child::<3>(idx as u8));
        let tree = build_test_tree_with_terminal_stems(&[
            (root.stem_idx(), 100),
            (siblings[0].stem_idx(), 300),
            (siblings[2].stem_idx(), 302),
        ]);

        let mut stack = <TestStemStrategy as StemStrategy>::Stack::<f64, 3>::default();
        stack.push(SimdQueryStackContext::new_block(
            siblings,
            [0.1, 99.0, 0.2, 99.0, 99.0, 99.0, 99.0, 99.0],
            [0.01, 9.9, 0.02, 9.9, 9.9, 9.9, 9.9, 9.9],
            [-1.0, -9.0, -2.0, -9.0, -9.0, -9.0, -9.0, -9.0],
            [1.0, 9.0, 2.0, 9.0, 9.0, 9.0, 9.0, 9.0],
            0b0000_0101,
            0,
            0.0,
            [f64::NEG_INFINITY; 3],
            [f64::INFINITY; 3],
        ));

        let mut query_ctx = TestQueryContext {
            query: [0.0, 0.0, 0.0],
            max_dist: 1.0,
        };
        let mut visited = Vec::new();

        tree.backtracking_query_with_simd_stack_impl::<_, f64, SquaredEuclidean<f64>>(
            &mut query_ctx,
            &mut stack,
            |leaf_idx, _, _| visited.push(leaf_idx),
        );

        assert_eq!(visited, vec![100, 300, 302]);
    }

    #[test]
    fn test_backtracking_query_with_simd_stack_exercises_deferred_block_arm() {
        let root = build_root_strat();
        let child0 = root.block_child::<3>(0);
        let child2 = root.block_child::<3>(2);
        let tree = build_test_tree_with_terminal_stems(&[
            (root.stem_idx(), 100),
            (child0.stem_idx(), 400),
            (child2.stem_idx(), 402),
        ]);

        let mut stack = <TestStemStrategy as StemStrategy>::Stack::<f64, 3>::default();
        stack.push(SimdQueryStackContext::new_deferred_block(
            root,
            0,
            [0.1, 99.0, 0.2, 99.0, 99.0, 99.0, 99.0, 99.0],
            [0.01, 9.9, 0.02, 9.9, 9.9, 9.9, 9.9, 9.9],
            [-1.0, -9.0, -2.0, -9.0, -9.0, -9.0, -9.0, -9.0],
            [1.0, 9.0, 2.0, 9.0, 9.0, 9.0, 9.0, 9.0],
            0b0000_0101,
            0,
            0.0,
            [f64::NEG_INFINITY; 3],
            [f64::INFINITY; 3],
        ));

        let mut query_ctx = TestQueryContext {
            query: [0.0, 0.0, 0.0],
            max_dist: 1.0,
        };
        let mut visited = Vec::new();

        tree.backtracking_query_with_simd_stack_impl::<_, f64, SquaredEuclidean<f64>>(
            &mut query_ctx,
            &mut stack,
            |leaf_idx, _, _| visited.push(leaf_idx),
        );

        assert_eq!(visited, vec![100, 400, 402]);
    }

    #[cfg(feature = "test_utils")]
    #[test]
    fn test_backtracking_query_with_block3_simd_stack_emits_pending_selection_trace() {
        use crate::kd_tree::query_stack_simd::Block3ExactStackContext;
        type Block3StemStrategy = DonnellySimdFull<3>;
        type Block3Tree = Block3TestTree;

        let stems = vec![0.2f64, 0.4, 0.6, 0.1, 0.3, 0.5, 0.7, f64::INFINITY];
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        let base = Block3StemStrategy::new(stems_ptr);
        let child0 = base.block_child::<3>(0);
        let child1 = base.block_child::<3>(1);
        let child2 = base.block_child::<3>(2);

        let tree = Block3Tree {
            stems,
            resolution: OwnedStemLeafResolution::Mapped {
                min_stem_leaf_idx: 0,
                leaf_idx_map: {
                    let mut map = vec![None; child2.stem_idx() + 1];
                    map[base.stem_idx()] = Some(NonMaxUsize::new(100).unwrap());
                    map[child0.stem_idx()] = Some(NonMaxUsize::new(200).unwrap());
                    map[child1.stem_idx()] = Some(NonMaxUsize::new(201).unwrap());
                    map[child2.stem_idx()] = Some(NonMaxUsize::new(202).unwrap());
                    map
                },
            },
        };

        let mut stack = <Block3StemStrategy as StemStrategy>::Stack::<f64, 3>::default();
        stack.push(
            <<Block3StemStrategy as StemStrategy>::StackContext<f64, 3> as Block3ExactStackContext<
                f64,
                Block3StemStrategy,
                3,
            >>::new_block3_pending_from_state(
                base,
                0b0000_0111,
                0.0,
                &[f64::NEG_INFINITY; 3],
                &[f64::INFINITY; 3],
            ),
        );

        let mut query_ctx = TestQueryContext {
            query: [0.45, 0.0, 0.0],
            max_dist: 0.2,
        };
        let mut visited = Vec::new();

        crate::test_utils::exact_query_trace::set_enabled(true);
        crate::test_utils::exact_query_stats::reset();

        tree.backtracking_query_with_block3_simd_stack_impl::<_, f64, SquaredEuclidean<f64>>(
            &mut query_ctx,
            &mut stack,
            |leaf_idx, _, _| visited.push(leaf_idx),
        );

        let events = crate::test_utils::exact_query_trace::snapshot();
        let stats = crate::test_utils::exact_query_stats::snapshot();
        crate::test_utils::exact_query_trace::set_enabled(false);

        assert!(visited.contains(&100));
        assert!(visited
            .iter()
            .any(|&leaf_idx| matches!(leaf_idx, 200..=202)));
        assert_eq!(stats.simd_single_pops, 1);
        assert!(stats.block3_pending_pops >= 1);
        assert!(stats.block3_candidate_mask_nonzero >= 1);

        let pending_event = events.iter().find_map(|event| {
            match event {
            crate::test_utils::exact_query_trace::ExactQueryTraceEvent::Block3PendingSelection {
                stem_idx,
                level,
                dim,
                pending_mask,
                candidate_mask,
                selected_child_idx,
                parent_lower_bound,
                parent_upper_bound,
                old_off,
                rd,
                ..
            } => Some((
                *stem_idx,
                *level,
                *dim,
                *pending_mask,
                *candidate_mask,
                *selected_child_idx,
                *parent_lower_bound,
                *parent_upper_bound,
                *old_off,
                *rd,
            )),
            _ => None,
        }
        });

        let (
            stem_idx,
            level,
            dim,
            pending_mask,
            candidate_mask,
            selected_child_idx,
            parent_lower_bound,
            parent_upper_bound,
            old_off,
            rd,
        ) = pending_event.expect("expected Block3PendingSelection trace event");

        assert_eq!(stem_idx, base.stem_idx());
        assert_eq!(level, base.level());
        assert_eq!(dim, 0);
        assert_eq!(pending_mask, 0b0000_0111);
        assert_ne!(candidate_mask, 0);
        assert!(candidate_mask & (1u8 << selected_child_idx) != 0);
        assert_eq!(parent_lower_bound, f64::NEG_INFINITY);
        assert_eq!(parent_upper_bound, f64::INFINITY);
        assert_eq!(old_off, 0.0);
        assert_eq!(rd, 0.0);
    }
}

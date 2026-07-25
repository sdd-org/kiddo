use aligned_vec::AVec;
use std::ptr::NonNull;

use crate::kd_tree::query_stack::{ScalarStackContext, StackTrait};
use crate::stem_strategy::donnelly::simd_full::{BacktrackBlock3, BacktrackBlock4};
use crate::{Axis, Content};

/// Trait that needs to be implemented by any potential stem ordering
/// algorithm used by a KdTree.
///
/// To see which stem strategies are available, see the [`stem_strategies`](`crate::stem_strategy`) module.
pub trait StemStrategy: Clone + Sync + Send + 'static {
    /// The stem index of the root node of the tree
    const ROOT_IDX: usize = 0;

    /// The block size of this strategy
    ///
    /// The default is 1, which means that the strategy is not block-based.
    const BLOCK_SIZE: usize = 1;

    /// Number of non-pivot terminal levels reserved at the bottom of the final block.
    ///
    /// Experimental stem layouts can use this to make construction align the pivot
    /// depth so terminal metadata occupies the last level of a block. Ordinary stem
    /// strategies leave this at zero.
    #[doc(hidden)]
    const TERMINAL_METADATA_LEVELS: usize = 0;

    /// Compact state persisted on scalar backtracking stacks.
    ///
    /// Scalar strategies can use this to store only the state needed to resume a deferred branch.
    /// SIMD / block strategies may ignore this and continue to use custom stack types.
    type DeferredState: Sized;

    /// Query stack context type for backtracking queries.
    ///
    /// Non-block strategies use simple scalar stack context (QueryStackContext).
    /// Block-based SIMD strategies use SimdQueryStackContext.
    type StackContext<A>: Sized
        + crate::kd_tree::query_stack::ScalarStackContext<A, Self::DeferredState>
    where
        Self: Sized;

    /// Query stack type for backtracking queries.
    ///
    /// Non-block strategies use simple scalar stack (QueryStack).
    /// Block-based SIMD strategies use SimdQueryStack.
    type Stack<A>: Default + crate::kd_tree::query_stack::StackTrait<A, Self>
    where
        Self: Sized;

    /// Create a new instance of this strategy at the root.
    fn new(stems_ptr: NonNull<u8>) -> Self;

    /// Create a new instance of this strategy at the root, with a dangling pointer.
    ///
    /// Useful for generating traversal indices without performing prefetches
    #[inline(always)]
    fn new_no_ptr() -> Self {
        Self::new(NonNull::dangling())
    }

    /// Returns the block size of this strategy
    #[inline(always)]
    fn block_size() -> usize {
        Self::BLOCK_SIZE
    }

    /// Get the current stem index this strategy points to.
    fn stem_idx(&self) -> usize;

    /// Snapshot the minimal scalar deferred state needed to resume traversal later.
    fn deferred_state(&self) -> Self::DeferredState;

    /// Restore this strategy from deferred scalar traversal state.
    ///
    /// Implementations may assume `self` already holds a valid `stems_ptr`.
    fn rehydrate_deferred_state(&mut self, state: Self::DeferredState);

    /// Get the current leaf index this strategy points to.
    fn leaf_idx(&self) -> usize;

    /// Get the current dimension (query time)
    fn dim<const K: usize>(&self) -> usize;

    /// Get the current dimension (construction time)
    fn construction_dim<const K: usize>(&self) -> usize {
        self.dim::<K>()
    }

    /// Get the current level
    fn level(&self) -> i32;

    /// Advance `self` down to a child in-place.
    fn traverse<A: Axis<Coord = A>, const K: usize>(&mut self, is_right: bool);

    /// Advance `self` down to a child in-place. Specialized for use as one
    /// of the non-final stages when loop-unrolling to the level of a minor tri height
    #[inline(always)]
    fn traverse_head<A: Axis<Coord = A>, const K: usize>(&mut self, is_right: bool) {
        self.traverse::<A, K>(is_right);
    }

    /// Advance `self` down to a child in-place. Specialized for use as the
    /// last stage when loop-unrolled to the level of a minor tri height
    #[inline(always)]
    fn traverse_tail<A: Axis<Coord = A>, const K: usize>(&mut self, is_right: bool) {
        self.traverse::<A, K>(is_right);
    }

    /// Advance `self` down to one child, returning the other.
    /// - `self` mutates into the left child
    /// - return value is the right child
    fn branch<A: Axis<Coord = A>, const K: usize>(&mut self) -> Self;

    /// Advance `self` to the "closer" child, returning the "further" one.
    #[inline(always)]
    fn branch_relative<A: Axis<Coord = A>, const K: usize>(&mut self, is_right: bool) -> Self {
        if is_right {
            let mut right = self.branch::<A, K>();
            std::mem::swap(self, &mut right);
            right
        } else {
            self.branch::<A, K>()
        }
    }

    /// Split `self` into two independent child strategies (left, right).
    fn split<A: Axis<Coord = A>, const K: usize>(mut self) -> (Self, Self)
    where
        Self: Sized,
    {
        let right = self.branch::<A, K>();
        (self, right)
    }

    /// Split `self` into (closer, further) given a direction.
    fn split_relative<A: Axis<Coord = A>, const K: usize>(self, is_right: bool) -> (Self, Self)
    where
        Self: Sized,
    {
        let (l, r) = self.split::<A, K>();
        if is_right {
            (r, l)
        } else {
            (l, r)
        }
    }

    /// Get the stem indices where the left and right children would be located.
    /// Returns (left_child_stem_idx, right_child_stem_idx).
    fn child_indices<A: Axis<Coord = A>>(&self) -> (usize, usize);

    /// Calculate the stem node count for a given leaf node count.
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn get_stem_node_count_from_leaf_node_count(_leaf_node_count: usize) -> usize {
        unimplemented!()
    }

    /// Factor by which to pad the stem node allocation.
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn stem_node_padding_factor() -> usize {
        1
    }

    /// Trim unneeded stem nodes.
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn trim_unneeded_stems<A: Axis<Coord = A>, const K: usize>(
        _stems: &mut AVec<A>,
        _max_stem_level: usize,
    ) {
        // Default: no-op
    }

    /// Emit cache-simulation events while advancing one level in the stem tree.
    ///
    /// Implementations should mirror `traverse` behavior but also report memory
    /// accesses via `event_tx` for the cache simulator.
    #[cfg(feature = "simulator")]
    fn simulate_traverse<A: Axis<Coord = A>, const K: usize>(
        &mut self,
        _is_right: bool,
        _event_tx: &std::sync::mpsc::Sender<crate::test_utils::cache_simulator::Event>,
    ) {
        unimplemented!();
    }

    /// Get leaf index for a query point. Default uses simple while loop.
    /// Block-based strategies override with unrolled loops.
    fn get_leaf_idx<A: Axis<Coord = A>, const K: usize>(
        stems: &[A],
        query: &[A; K],
        max_stem_level: i32,
    ) -> usize
    where
        Self: Sized,
    {
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        let mut stem_strat = Self::new(stems_ptr);

        while stem_strat.level() <= max_stem_level {
            let pivot = unsafe { stems.get_unchecked(stem_strat.stem_idx()) };
            let is_right = unsafe { *query.get_unchecked(stem_strat.dim::<K>()) } >= *pivot;
            stem_strat.traverse::<A, K>(is_right);
        }

        stem_strat.leaf_idx()
    }

    /// Single step of backtracking traversal.
    ///
    /// Default implementation handles level-by-level traversal with one pivot per step.
    /// Block-based strategies override to handle multiple levels at once with SIMD.
    ///
    /// Returns true if traversal should continue (more levels to go), false if at leaf.
    #[inline(always)]
    fn backtracking_traverse_step<A, O, D, const K: usize>(
        &mut self,
        stems: &[A],
        query: &[A; K],
        query_wide: &[O; K],
        off: &mut [O; K],
        dim: &mut usize,
        rd: O,
        max_stem_level: i32,
        best_dist: O,
        stack: &mut Self::Stack<O>,
    ) -> bool
    where
        Self: Sized,
        A: Axis<Coord = A>,
        O: Axis<Coord = O> + BacktrackBlock3 + BacktrackBlock4,
        D: crate::dist::DistanceMetric<A, Output = O>,
        Self::Stack<O>: StackTrait<O, Self>,
    {
        // Default implementation for scalar strategies
        // SIMD strategies override this entire method
        if self.level() > max_stem_level {
            return false;
        }

        #[cfg(feature = "result_collection_stats")]
        crate::results::result_collection_stats::record_query_scalar_traverse_step();

        #[cfg(feature = "result_collection_stats")]
        {
            let rd_from_off = D::rect_dist_from_off(off);
            crate::results::result_collection_stats::record_query_scalar_rd_off_check(O::cmp(
                rd_from_off,
                rd,
            ));
        }

        let pivot = *unsafe { stems.get_unchecked(self.stem_idx()) };

        if pivot < A::max_value() {
            let query_elem = *unsafe { query.get_unchecked(*dim) };
            let is_right_child = query_elem >= pivot;

            let old_stem_idx = self.stem_idx();
            let far_ctx = self.branch_relative::<A, K>(is_right_child);

            tracing::trace!(
                %pivot,
                dim = %self.dim::<K>(),
                %query_elem,
                %is_right_child,
                %old_stem_idx,
                new_stem_idx = %self.stem_idx(),
                level = self.level(),
                "Traverse down one level"
            );

            let pivot_wide: O = D::widen_coord(pivot);
            let query_elem_wide = *unsafe { query_wide.get_unchecked(*dim) };

            let new_off = O::saturating_dist(query_elem_wide, pivot_wide);
            #[cfg(feature = "test_utils")]
            let old_off = *unsafe { off.get_unchecked(*dim) };
            let rd_far = D::rect_dist_after_update(rd, off, *dim, new_off);

            #[cfg(feature = "test_utils")]
            {
                if crate::test_utils::exact_query_trace::enabled()
                    && std::any::type_name::<A>() == "f64"
                    && std::any::type_name::<O>() == "f64"
                {
                    let pivot_f = unsafe { *(&pivot as *const A as *const f64) };
                    let query_elem_f = unsafe { *(&query_elem as *const A as *const f64) };
                    let old_off_f = unsafe { *(&old_off as *const O as *const f64) };
                    let new_off_f = unsafe { *(&new_off as *const O as *const f64) };
                    let rd_f = unsafe { *(&rd as *const O as *const f64) };
                    let rd_far_f = unsafe { *(&rd_far as *const O as *const f64) };
                    crate::test_utils::exact_query_trace::push(
                        crate::test_utils::exact_query_trace::ExactQueryTraceEvent::ScalarStep {
                            stem_idx: old_stem_idx,
                            level: self.level() - 1,
                            dim: *dim,
                            pivot: pivot_f,
                            query_elem: query_elem_f,
                            is_right_child,
                            old_off: old_off_f,
                            new_off: new_off_f,
                            rd: rd_f,
                            rd_far: rd_far_f,
                            near_stem_idx: self.stem_idx(),
                            far_stem_idx: far_ctx.stem_idx(),
                        },
                    );
                }
            }

            // Only push if the sibling is worth exploring
            if O::cmp(rd_far, best_dist) != std::cmp::Ordering::Greater {
                stack.push(Self::StackContext::<O>::from_parts_with_restore_dim(
                    far_ctx.deferred_state(),
                    *dim,
                    new_off,
                    rd_far,
                ));
                #[cfg(feature = "result_collection_stats")]
                crate::results::result_collection_stats::record_query_scalar_far_child_push();
            } else {
                #[cfg(feature = "result_collection_stats")]
                crate::results::result_collection_stats::record_query_scalar_far_child_reject();
            }
        } else {
            self.traverse::<A, K>(false);
        }

        *dim = self.dim::<K>();
        true
    }

    /// Single step of backtracking traversal for interval-aware SIMD exact paths.
    ///
    /// Default implementation ignores the interval bounds and delegates to
    /// `backtracking_traverse_step`.
    #[inline(always)]
    fn backtracking_traverse_step_with_bounds<A, O, D, const K: usize>(
        &mut self,
        stems: &[A],
        query: &[A; K],
        query_wide: &[O; K],
        lower: &mut [O; K],
        upper: &mut [O; K],
        off: &mut [O; K],
        dim: &mut usize,
        rd: O,
        max_stem_level: i32,
        best_dist: O,
        stack: &mut Self::Stack<O>,
    ) -> bool
    where
        Self: Sized,
        A: Axis<Coord = A>,
        O: Axis<Coord = O>
            + crate::stem_strategy::SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4,
        D: crate::dist::DistanceMetric<A, Output = O>,
        Self::Stack<O>: StackTrait<O, Self>,
    {
        let _ = lower;
        let _ = upper;
        self.backtracking_traverse_step::<A, O, D, K>(
            stems,
            query,
            query_wide,
            off,
            dim,
            rd,
            max_stem_level,
            best_dist,
            stack,
        )
    }

    /// Execute backtracking query with explicit stack.
    ///
    /// Default implementation delegates to KdTree's backtracking_query_with_scratch_impl.
    /// Block-based SIMD strategies can override to use SIMD stack with block pruning.
    ///
    /// This method exists to allow strategies to customize backtracking behavior at compile time.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn backtracking_query_with_scratch<Tree, A, T, O, D, QC, LS, const K2: usize, const B: usize>(
        tree: &Tree,
        query_ctx: &mut QC,
        stack: &mut Self::Stack<O>,
        process_leaf: impl FnMut(usize, &[O; K2], &mut QC),
    ) where
        Self: Sized,
        Tree: crate::kd_tree::KdTreeAccessor<A, T, Self, LS, K2, B>
            + crate::kd_tree::KdTreeQueryOps<A, T, Self, LS, K2, B>,
        A: Axis<Coord = A>,
        T: Content,
        O: Axis<Coord = O>
            + crate::stem_strategy::SimdPrune
            + crate::stem_strategy::SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4,
        D: crate::dist::DistanceMetric<A, Output = O>,
        QC: crate::kd_tree::query_context::QueryContext<A, O, K2>,
        LS: crate::LeafStrategy<A, T, Self, K2, B>,
        Self::Stack<O>: StackTrait<O, Self>,
    {
        tree.backtracking_query_with_scratch_impl::<QC, O, D>(query_ctx, stack, process_leaf);
    }

    /// Execute arithmetic-resolution backtracking query with explicit stack.
    ///
    /// Default implementation delegates to KdTree's scalar arithmetic implementation.
    /// Strategies may override this to provide a more specialized arithmetic walk.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn arithmetic_query_with_scratch<Tree, A, T, O, D, QC, LS, const K2: usize, const B: usize>(
        tree: &Tree,
        query_ctx: &mut QC,
        stack: &mut Self::Stack<O>,
        process_leaf: impl FnMut(usize, &[O; K2], &mut QC),
    ) where
        Self: Sized,
        Tree: crate::kd_tree::KdTreeAccessor<A, T, Self, LS, K2, B>
            + crate::kd_tree::KdTreeQueryOps<A, T, Self, LS, K2, B>,
        A: Axis<Coord = A>,
        T: Content,
        O: Axis<Coord = O> + BacktrackBlock3 + BacktrackBlock4,
        D: crate::dist::DistanceMetric<A, Output = O>,
        QC: crate::kd_tree::query_context::QueryContext<A, O, K2>,
        LS: crate::LeafStrategy<A, T, Self, K2, B>,
        Self::Stack<O>: StackTrait<O, Self>,
    {
        tree.arithmetic_query_with_scratch_impl::<QC, O, D>(query_ctx, stack, process_leaf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::SquaredEuclidean;
    use crate::kd_tree::query_stack::{QueryStack, QueryStackContext};

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct TestStemState {
        stem_idx: usize,
        leaf_idx: usize,
        dim: usize,
        level: i32,
    }

    #[derive(Clone, Debug)]
    struct TestStemStrategy {
        state: TestStemState,
        _stems_ptr: NonNull<u8>,
    }

    unsafe impl Send for TestStemStrategy {}
    unsafe impl Sync for TestStemStrategy {}

    impl StemStrategy for TestStemStrategy {
        type DeferredState = TestStemState;
        type StackContext<A>
            = QueryStackContext<A, Self::DeferredState>
        where
            Self: Sized;
        type Stack<A>
            = QueryStack<A, Self>
        where
            Self: Sized;

        fn new(stems_ptr: NonNull<u8>) -> Self {
            Self {
                state: TestStemState {
                    stem_idx: 0,
                    leaf_idx: 0,
                    dim: 0,
                    level: 0,
                },
                _stems_ptr: stems_ptr,
            }
        }

        fn stem_idx(&self) -> usize {
            self.state.stem_idx
        }

        fn deferred_state(&self) -> Self::DeferredState {
            self.state
        }

        fn rehydrate_deferred_state(&mut self, state: Self::DeferredState) {
            self.state = state;
        }

        fn leaf_idx(&self) -> usize {
            self.state.leaf_idx
        }

        fn dim<const K: usize>(&self) -> usize {
            self.state.dim
        }

        fn level(&self) -> i32 {
            self.state.level
        }

        fn traverse<A: Axis<Coord = A>, const K: usize>(&mut self, is_right: bool) {
            self.state.stem_idx = self.state.stem_idx * 2 + 1 + usize::from(is_right);
            self.state.leaf_idx = (self.state.leaf_idx << 1) | usize::from(is_right);
            self.state.dim = (self.state.dim + 1) % 2;
            self.state.level += 1;
        }

        fn branch<A: Axis<Coord = A>, const K: usize>(&mut self) -> Self {
            let mut right = self.clone();
            self.state.stem_idx = self.state.stem_idx * 2 + 1;
            self.state.leaf_idx <<= 1;
            self.state.dim = (self.state.dim + 1) % 2;
            self.state.level += 1;

            right.state.stem_idx = right.state.stem_idx * 2 + 2;
            right.state.leaf_idx = (right.state.leaf_idx << 1) | 1;
            right.state.dim = (right.state.dim + 1) % 2;
            right.state.level += 1;
            right
        }

        fn child_indices<A: Axis<Coord = A>>(&self) -> (usize, usize) {
            (self.state.stem_idx * 2 + 1, self.state.stem_idx * 2 + 2)
        }
    }

    #[test]
    fn default_traverse_head_and_split_relative_follow_direction() {
        let mut stems = [0.0f32; 8];
        let stems_ptr = NonNull::new(stems.as_mut_ptr() as *mut u8).unwrap();

        let mut head = TestStemStrategy::new(stems_ptr);
        head.traverse_head::<f32, 2>(true);
        assert_eq!(head.stem_idx(), 2);
        assert_eq!(head.leaf_idx(), 1);
        assert_eq!(head.dim::<2>(), 1);
        assert_eq!(head.level(), 1);

        let left_first = TestStemStrategy::new(stems_ptr).split_relative::<f32, 2>(false);
        assert_eq!(left_first.0.stem_idx(), 1);
        assert_eq!(left_first.1.stem_idx(), 2);
        assert_eq!(left_first.0.leaf_idx(), 0);
        assert_eq!(left_first.1.leaf_idx(), 1);

        let right_first = TestStemStrategy::new(stems_ptr).split_relative::<f32, 2>(true);
        assert_eq!(right_first.0.stem_idx(), 2);
        assert_eq!(right_first.1.stem_idx(), 1);
        assert_eq!(right_first.0.leaf_idx(), 1);
        assert_eq!(right_first.1.leaf_idx(), 0);
    }

    #[test]
    fn default_backtracking_traverse_step_with_bounds_delegates_to_scalar_step() {
        let mut stems = [5.0f32, 0.0, 0.0, 0.0];
        let mut strat = TestStemStrategy::new(NonNull::new(stems.as_mut_ptr() as *mut u8).unwrap());
        let query = [7.0f32, 1.0f32];
        let query_wide = [7.0f32, 1.0f32];
        let mut lower = [f32::NEG_INFINITY; 2];
        let mut upper = [f32::INFINITY; 2];
        let mut off = [0.0f32; 2];
        let mut dim = 0usize;
        let mut stack = QueryStack::<f32, TestStemStrategy>::default();

        let should_continue = strat
            .backtracking_traverse_step_with_bounds::<f32, f32, SquaredEuclidean<f32>, 2>(
                &stems,
                &query,
                &query_wide,
                &mut lower,
                &mut upper,
                &mut off,
                &mut dim,
                0.0,
                3,
                100.0,
                &mut stack,
            );

        assert!(should_continue);
        assert_eq!(strat.stem_idx(), 2);
        assert_eq!(strat.leaf_idx(), 1);
        assert_eq!(strat.level(), 1);
        assert_eq!(strat.dim::<2>(), 1);
        assert_eq!(dim, 1);
        assert_eq!(off, [0.0, 0.0]);
        assert_eq!(lower, [f32::NEG_INFINITY, f32::NEG_INFINITY]);
        assert_eq!(upper, [f32::INFINITY, f32::INFINITY]);

        let ctx = stack.pop().expect("far child should be pushed");
        let (stem_state, restore_dim, old_off, rd) = ctx.into_parts_with_restore_dim();
        assert_eq!(stem_state.stem_idx, 1);
        assert_eq!(stem_state.leaf_idx, 0);
        assert_eq!(stem_state.level, 1);
        assert_eq!(stem_state.dim, 1);
        assert_eq!(restore_dim, Some(0));
        assert_eq!(old_off, 2.0);
        assert_eq!(rd, 4.0);
    }

    #[cfg(feature = "test_utils")]
    #[test]
    fn default_backtracking_traverse_step_emits_exact_query_trace_event() {
        let mut stems = [5.0f64, 0.0, 0.0, 0.0];
        let mut strat = TestStemStrategy::new(NonNull::new(stems.as_mut_ptr() as *mut u8).unwrap());
        let query = [7.0f64, 1.0f64];
        let query_wide = [7.0f64, 1.0f64];
        let mut off = [0.0f64; 2];
        let mut dim = 0usize;
        let mut stack = QueryStack::<f64, TestStemStrategy>::default();

        crate::test_utils::exact_query_trace::set_enabled(true);

        let should_continue = strat
            .backtracking_traverse_step::<f64, f64, SquaredEuclidean<f64>, 2>(
                &stems,
                &query,
                &query_wide,
                &mut off,
                &mut dim,
                0.0,
                3,
                100.0,
                &mut stack,
            );

        let events = crate::test_utils::exact_query_trace::snapshot();
        crate::test_utils::exact_query_trace::set_enabled(false);

        assert!(should_continue);
        assert_eq!(events.len(), 1);

        match &events[0] {
            crate::test_utils::exact_query_trace::ExactQueryTraceEvent::ScalarStep {
                stem_idx,
                level,
                dim,
                pivot,
                query_elem,
                is_right_child,
                old_off,
                new_off,
                rd,
                rd_far,
                near_stem_idx,
                far_stem_idx,
            } => {
                assert_eq!(*stem_idx, 0);
                assert_eq!(*level, 0);
                assert_eq!(*dim, 0);
                assert_eq!(*pivot, 5.0);
                assert_eq!(*query_elem, 7.0);
                assert!(*is_right_child);
                assert_eq!(*old_off, 0.0);
                assert_eq!(*new_off, 2.0);
                assert_eq!(*rd, 0.0);
                assert_eq!(*rd_far, 4.0);
                assert_eq!(*near_stem_idx, 2);
                assert_eq!(*far_stem_idx, 1);
            }
            other => panic!("expected ScalarStep trace event, got {other:?}"),
        }
    }
}

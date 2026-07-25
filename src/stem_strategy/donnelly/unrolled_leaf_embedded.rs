//! Benchmark-only Donnelly layout reserving the final block level for leaf descriptors.

use std::ptr::NonNull;

use crate::stem_strategy::donnelly::core::DonnellyCore;
use crate::{Axis, StemStrategy};

/// Block-height-three Donnelly traversal whose final block level is leaf metadata.
///
/// This exists only to test the embedded-leaf-descriptor layout. Its ordinary
/// `get_leaf_idx` implementation stops before the metadata level and continues to
/// resolve leaves through the leaf strategy's extent table.
#[doc(hidden)]
#[derive(Copy, Clone, Debug)]
pub struct DonnellyUnrolledLeafEmbedded3 {
    core: DonnellyCore<3>,
}

impl StemStrategy for DonnellyUnrolledLeafEmbedded3 {
    const ROOT_IDX: usize = 0;
    const BLOCK_SIZE: usize = 3;
    const TERMINAL_METADATA_LEVELS: usize = 1;

    type DeferredState = Self;
    type StackContext<A> = crate::kd_tree::query_stack::QueryStackContext<A, Self::DeferredState>;
    type Stack<A> = crate::kd_tree::query_stack::QueryStack<A, Self>;

    #[inline(always)]
    fn new(stems_ptr: NonNull<u8>) -> Self {
        Self {
            core: DonnellyCore::new(stems_ptr),
        }
    }

    #[inline(always)]
    fn stem_idx(&self) -> usize {
        self.core.stem_idx()
    }

    #[inline(always)]
    fn deferred_state(&self) -> Self::DeferredState {
        *self
    }

    #[inline(always)]
    fn rehydrate_deferred_state(&mut self, state: Self::DeferredState) {
        *self = state;
    }

    #[inline(always)]
    fn leaf_idx(&self) -> usize {
        self.core.leaf_idx()
    }

    #[inline(always)]
    fn dim<const K: usize>(&self) -> usize {
        self.core.dim::<K>()
    }

    #[inline(always)]
    fn level(&self) -> i32 {
        self.core.level()
    }

    #[inline(always)]
    fn traverse<A: Axis<Coord = A>, const K: usize>(&mut self, is_right: bool) {
        self.core.traverse::<A, K>(is_right);
    }

    #[inline(always)]
    fn traverse_head<A: Axis<Coord = A>, const K: usize>(&mut self, is_right: bool) {
        self.core.traverse_head::<A, K>(is_right);
    }

    #[inline(always)]
    fn traverse_tail<A: Axis<Coord = A>, const K: usize>(&mut self, is_right: bool) {
        self.core.traverse_tail_with_block_size::<A, K>(is_right, 3);
    }

    #[inline(always)]
    fn branch<A: Axis<Coord = A>, const K: usize>(&mut self) -> Self {
        Self {
            core: self.core.branch::<A, K>(),
        }
    }

    #[inline(always)]
    fn child_indices<A: Axis<Coord = A>>(&self) -> (usize, usize) {
        self.core.child_indices::<A>()
    }

    fn get_leaf_idx<A: Axis<Coord = A>, const K: usize>(
        stems: &[A],
        query: &[A; K],
        max_stem_level: i32,
    ) -> usize {
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        let mut strat = Self::new(stems_ptr);
        let pivot_depth = max_stem_level + 1;

        debug_assert_eq!((pivot_depth + 1) % 3, 0);

        while strat.level() + 2 < pivot_depth {
            let pivot = unsafe { *stems.get_unchecked(strat.stem_idx()) };
            let is_right = unsafe { *query.get_unchecked(strat.dim::<K>()) } >= pivot;
            strat.traverse_head::<A, K>(is_right);

            let pivot = unsafe { *stems.get_unchecked(strat.stem_idx()) };
            let is_right = unsafe { *query.get_unchecked(strat.dim::<K>()) } >= pivot;
            strat.traverse_head::<A, K>(is_right);

            let pivot = unsafe { *stems.get_unchecked(strat.stem_idx()) };
            let is_right = unsafe { *query.get_unchecked(strat.dim::<K>()) } >= pivot;
            strat.traverse_tail::<A, K>(is_right);
        }

        while strat.level() < pivot_depth {
            let pivot = unsafe { *stems.get_unchecked(strat.stem_idx()) };
            let is_right = unsafe { *query.get_unchecked(strat.dim::<K>()) } >= pivot;
            strat.traverse_head::<A, K>(is_right);
        }

        strat.leaf_idx()
    }
}

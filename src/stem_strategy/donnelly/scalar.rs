//! Donnelly stem strategy with software prefetch.
//!
//! This wrapper currently delegates traversal and prefetch behavior to
//! [`DonnellyCore`], which is already block-height-based and derives byte
//! offsets from `A::VALUE_WIDTH_BYTES`.

use std::ptr::NonNull;

use aligned_vec::AVec;

use crate::stem_strategy::donnelly::core::{DonnellyCore, DonnellyCoreDeferred};
use crate::{Axis, StemStrategy};

/// Donnelly strategy with software prefetch, parameterized by block height.
#[derive(Copy, Clone, Debug)]
pub struct Donnelly<const BH: usize> {
    core: DonnellyCore<BH>,
}

impl<const BH: usize> StemStrategy for Donnelly<BH> {
    const ROOT_IDX: usize = 0;
    const BLOCK_SIZE: usize = BH;
    const REQUIRES_BLOCK_ALIGNED_STEM_HEIGHT: bool = false;
    const SUPPORTS_ARITHMETIC_LEAF_RESOLUTION: bool = true;

    type DeferredState = DonnellyCoreDeferred;
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
        self.core.deferred_state()
    }

    #[inline(always)]
    fn rehydrate_deferred_state(&mut self, state: Self::DeferredState) {
        self.core.rehydrate_deferred_state(state);
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
        self.core
            .traverse_tail_with_block_size::<A, K>(is_right, BH as u32);
    }

    #[inline(always)]
    fn branch<A: Axis<Coord = A>, const K: usize>(&mut self) -> Self {
        Self {
            core: self.core.branch::<A, K>(),
        }
    }

    #[inline(always)]
    fn branch_relative<A: Axis<Coord = A>, const K: usize>(&mut self, is_right: bool) -> Self {
        Self {
            core: self.core.branch_relative::<A, K>(is_right),
        }
    }

    #[inline(always)]
    fn child_indices<A: Axis<Coord = A>>(&self) -> (usize, usize) {
        self.core.child_indices::<A>()
    }

    fn get_stem_node_count_from_leaf_node_count(leaf_node_count: usize) -> usize {
        if leaf_node_count < 2 {
            0
        } else {
            leaf_node_count.next_power_of_two() - 1
        }
    }

    fn stem_node_padding_factor() -> usize {
        50
    }

    fn trim_unneeded_stems<A: Axis<Coord = A>, const K: usize>(
        stems: &mut AVec<A>,
        max_stem_level: usize,
    ) {
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        if stems.is_empty() {
            return;
        }

        let mut so = Self::new(stems_ptr);
        loop {
            let val = stems[so.stem_idx()];
            let is_right_child = !A::is_max_value(val);
            so.traverse::<A, K>(is_right_child);
            if so.level() as usize == max_stem_level {
                break;
            }
        }

        #[cfg(debug_assertions)]
        {
            for i in (so.stem_idx() + 1)..stems.len() {
                debug_assert!(A::is_max_value(stems[i]), "stems[{i}] = {}", stems[i]);
            }
        }

        stems.truncate(so.stem_idx() + 1);
    }
}

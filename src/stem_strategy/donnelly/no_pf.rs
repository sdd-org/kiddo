//! Donnelly stem strategy backed by [`DonnellyCore`].

use std::ptr::NonNull;

use aligned_vec::AVec;

use crate::stem_strategy::donnelly::core::{DonnellyCore, DonnellyCoreDeferred};
use crate::{Axis, StemStrategy};

/// Scalar Donnelly stem strategy without software prefetch.
#[derive(Copy, Clone, Debug)]
pub struct DonnellyNoPf<const BH: usize> {
    core: DonnellyCore<BH>,
}

impl<const BH: usize> StemStrategy for DonnellyNoPf<BH> {
    const ROOT_IDX: usize = 0;
    const BLOCK_SIZE: usize = BH;
    const REQUIRES_BLOCK_ALIGNED_STEM_HEIGHT: bool = false;
    const SUPPORTS_ARITHMETIC_LEAF_RESOLUTION: bool = true;

    type DeferredState = DonnellyCoreDeferred;
    type StackContext<A, const K: usize> =
        crate::kd_tree::query_stack::QueryStackContext<A, Self::DeferredState>;
    type Stack<A, const K: usize> = crate::kd_tree::query_stack::QueryStack<A, Self, K>;

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

    #[inline(always)]
    fn get_leaf_idx<A: Axis<Coord = A>, const K: usize>(
        stems: &[A],
        query: &[A; K],
        max_stem_level: i32,
    ) -> usize {
        if BH == 3 {
            return crate::stem_strategy::donnelly::core::get_leaf_idx_block3(
                stems,
                query,
                max_stem_level,
            );
        }
        if BH == 4 {
            return crate::stem_strategy::donnelly::core::get_leaf_idx_block4(
                stems,
                query,
                max_stem_level,
            );
        }

        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        let mut strat = Self::new(stems_ptr);
        while strat.level() <= max_stem_level {
            let pivot = unsafe { *stems.get_unchecked(strat.stem_idx()) };
            let is_right = unsafe { *query.get_unchecked(strat.dim::<K>()) } >= pivot;
            strat.traverse::<A, K>(is_right);
        }
        strat.leaf_idx()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn donnelly_block3_round_trips_deferred_state() {
        let stems = [0.0f64; 256];
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();

        let mut strat = DonnellyNoPf::<3>::new(stems_ptr);
        let saved = strat.deferred_state();

        strat.traverse::<f64, 3>(true);
        strat.traverse::<f64, 3>(false);
        assert_ne!(strat.stem_idx(), 0);

        strat.rehydrate_deferred_state(saved);
        assert_eq!(strat.stem_idx(), 0);
        assert_eq!(strat.leaf_idx(), 0);
        assert_eq!(strat.level(), 0);
        assert_eq!(strat.dim::<3>(), 0);
    }

    #[test]
    fn donnelly_block3_matches_core_traversal() {
        let stems = [0.0f64; 256];
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();

        let mut wrapper = DonnellyNoPf::<3>::new(stems_ptr);
        let mut core = DonnellyCore::<3>::new(stems_ptr);

        wrapper.traverse_head::<f64, 3>(true);
        core.traverse_head::<f64, 3>(true);
        wrapper.traverse_tail::<f64, 3>(false);
        core.traverse_tail_with_block_size::<f64, 3>(false, 3);

        assert_eq!(wrapper.stem_idx(), core.stem_idx());
        assert_eq!(wrapper.leaf_idx(), core.leaf_idx());
        assert_eq!(wrapper.level(), core.level());
        assert_eq!(wrapper.dim::<3>(), core.dim::<3>());
        assert_eq!(wrapper.child_indices::<f64>(), core.child_indices::<f64>());
    }
}

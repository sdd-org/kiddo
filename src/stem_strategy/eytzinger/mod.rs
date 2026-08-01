//! Eytzinger stem ordering strategies.
//!
//! All Eytzinger variants use the same breadth-first stem layout. The only
//! difference between them is the software-prefetch policy applied while
//! descending the tree.
//!
//! [`EytzingerFlexPf`] is the underlying configurable implementation. It has
//! two compile-time prefetch slots:
//!
//! - `PF1` controls the immediate child lookahead.
//! - `PF2` controls a deeper lookahead further down the same descent path.
//!
//! Each slot accepts the numeric value of a
//! [`crate::stem_strategies::eytzinger::PrefetchAction`] variant:
//! `0` for [`crate::stem_strategies::eytzinger::PrefetchAction::T0`],
//! `1` for [`crate::stem_strategies::eytzinger::PrefetchAction::T1`], and `-1`
//! for [`crate::stem_strategies::eytzinger::PrefetchAction::None`].
//!
//! The two aliases cover the normal public choices:
//!
//! - [`Eytzinger`] is the default configuration and maps to
//!   `EytzingerFlexPf<0, 1>`.
//! - [`EytzingerNoPf`] disables software prefetch entirely and maps to
//!   `EytzingerFlexPf<-1, -1>`.
//!
//! Most users should pick one of those aliases. [`EytzingerFlexPf`] is public
//! so custom prefetch combinations remain available for experimentation.

use crate::stem_strategy::prefetch::{prefetch_t0, prefetch_t1};
use crate::{Axis, StemStrategy};
use std::ptr::NonNull;

const MAX_MUTABLE_STEM_SPARSITY_FACTOR: usize = 16;

/// Prefetch policy for an Eytzinger traversal slot.
#[derive(Clone, Debug, PartialOrd, Ord, Eq, PartialEq)]
pub enum PrefetchAction {
    /// Issue a temporal-L1 prefetch.
    T0 = 0,
    /// Issue a temporal-L2-style prefetch.
    T1 = 1,
    /// Do not prefetch for this slot.
    None = -1,
}

/// Default Eytzinger stem strategy with the preferred prefetch configuration.
pub type Eytzinger = EytzingerFlexPf<0, 1>;

/// Eytzinger stem strategy with no software prefetching.
pub type EytzingerNoPf = EytzingerFlexPf<-1, -1>;

/// Eytzinger stem strategy with configurable software-prefetch policy.
///
/// `PF1` and `PF2` describe two lookahead slots along the active descent path.
/// They use the numeric encoding from [`PrefetchAction`]:
///
/// - `0` = [`PrefetchAction::T0`]
/// - `1` = [`PrefetchAction::T1`]
/// - `-1` = [`PrefetchAction::None`]
///
/// The default alias [`Eytzinger`] chooses `PF1 = 0` and `PF2 = 1`, while
/// [`EytzingerNoPf`] disables both slots.
#[derive(Clone, Debug)]
pub struct EytzingerFlexPf<const PF1: isize = 0, const PF2: isize = 1> {
    stem_idx: u32,
    dim: usize,
    level: i32,

    stems_ptr: NonNull<u8>,
}

/// Compact deferred traversal state for [`EytzingerFlexPf`].
#[doc(hidden)]
pub struct EytzingerFlexDeferred {
    stem_idx: u32,
    level: u16,
    dim: u16,
}

unsafe impl<const PF1: isize, const PF2: isize> Send for EytzingerFlexPf<PF1, PF2> {}
unsafe impl<const PF1: isize, const PF2: isize> Sync for EytzingerFlexPf<PF1, PF2> {}

impl<const PF1: isize, const PF2: isize> StemStrategy for EytzingerFlexPf<PF1, PF2> {
    const ROOT_IDX: usize = 1;

    type DeferredState = EytzingerFlexDeferred;
    type StackContext<A> = crate::kd_tree::query_stack::QueryStackContext<A, Self::DeferredState>;
    type Stack<A> = crate::kd_tree::query_stack::QueryStack<A, Self>;

    fn new(stems_ptr: NonNull<u8>) -> Self {
        Self {
            stem_idx: Self::ROOT_IDX as u32,
            dim: 0,
            level: 0,
            stems_ptr,
        }
    }

    #[inline(always)]
    fn stem_idx(&self) -> usize {
        self.stem_idx as usize
    }
    fn deferred_state(&self) -> Self::DeferredState {
        EytzingerFlexDeferred {
            stem_idx: self.stem_idx,
            level: self.level as u16,
            dim: self.dim as u16,
        }
    }
    fn rehydrate_deferred_state(&mut self, state: Self::DeferredState) {
        self.stem_idx = state.stem_idx;
        self.level = state.level as i32;
        self.dim = state.dim as usize;
    }

    #[inline(always)]
    fn leaf_idx(&self) -> usize {
        let mask = 1u32.wrapping_shl(self.level as u32);
        (self.stem_idx & !mask) as usize
    }

    #[inline(always)]
    fn dim<const K: usize>(&self) -> usize {
        self.dim
    }

    #[inline(always)]
    fn level(&self) -> i32 {
        self.level
    }

    #[inline]
    fn traverse<A: Axis, const K: usize>(&mut self, is_right_child: bool) {
        self.stem_idx = Self::step_pure::<A>(self.stem_idx, is_right_child, self.stems_ptr);

        self.level = self.level.wrapping_add(1);

        let wrap_dim_mask = 0usize.wrapping_sub((self.dim == (K - 1)) as usize);
        self.dim = self.dim.wrapping_add(1) & !wrap_dim_mask;
    }

    #[cfg(feature = "simulator")]
    fn simulate_traverse<A, const K: usize>(
        &mut self,
        is_right: bool,
        event_tx: &std::sync::mpsc::Sender<crate::test_utils::cache_simulator::Event>,
    ) where
        A: Axis<Coord = A>,
    {
        self.traverse::<A, K>(is_right);

        // MCA analysis shows that Eytzinger step_pure is just one LEA instr with est 3.5IPC and est
        // RThroughput of 0.5. Adding the estimate for the level and dim updating gets us to 1.5 to 2 cycles
        let _ = event_tx.send(crate::test_utils::cache_simulator::Event::Working(2));
    }

    fn branch<A: Axis<Coord = A>, const K: usize>(&mut self) -> Self {
        self.stem_idx = self.stem_idx.wrapping_shl(1);
        let right = self.stem_idx | 1;

        self.level = self.level.wrapping_add(1);

        let wrap_dim_mask = 0usize.wrapping_sub((self.dim == (K - 1)) as usize);
        self.dim = self.dim.wrapping_add(1) & !wrap_dim_mask;

        Self {
            stem_idx: right,
            ..*self
        }
    }

    #[inline(always)]
    fn branch_relative<A: Axis<Coord = A>, const K: usize>(&mut self, is_right: bool) -> Self {
        let child_base = self.stem_idx.wrapping_shl(1);
        let direction = is_right as u32;
        let near_idx = child_base | direction;
        let far_idx = child_base | (direction ^ 1);

        self.level = self.level.wrapping_add(1);

        let wrap_dim_mask = 0usize.wrapping_sub((self.dim == (K - 1)) as usize);
        self.dim = self.dim.wrapping_add(1) & !wrap_dim_mask;

        self.stem_idx = near_idx;

        Self::prefetch_descendants::<A>(self.stem_idx, self.stems_ptr);

        Self {
            stem_idx: far_idx,
            dim: self.dim,
            level: self.level,
            stems_ptr: self.stems_ptr,
        }
    }

    fn child_indices<A: Axis<Coord = A>>(&self) -> (usize, usize) {
        let left = (self.stem_idx << 1) as usize;
        let right = left | 1;
        (left, right)
    }

    fn mutable_split_requires_rebuild(&self, leaf_count_after_split: usize) -> bool {
        let compact_terminal_limit = leaf_count_after_split
            .checked_next_power_of_two()
            .unwrap_or(usize::MAX)
            .saturating_mul(2);
        let sparse_terminal_limit =
            compact_terminal_limit.saturating_mul(MAX_MUTABLE_STEM_SPARSITY_FACTOR);
        let right_child_idx = (self.stem_idx as usize).saturating_mul(2).saturating_add(1);

        right_child_idx >= sparse_terminal_limit || right_child_idx > u32::MAX as usize
    }

    fn get_stem_node_count_from_leaf_node_count(leaf_node_count: usize) -> usize {
        if leaf_node_count < 2 {
            0
        } else {
            leaf_node_count.next_power_of_two()
        }
    }
    fn stem_node_padding_factor() -> usize {
        1
    }
}

impl<const PF1: isize, const PF2: isize> EytzingerFlexPf<PF1, PF2> {
    #[inline(always)]
    fn prefetch_descendants<A: Axis>(stem_idx: u32, stems_ptr: NonNull<u8>) {
        match PF1 {
            0 => unsafe {
                let nxt_ptr = stems_ptr
                    .as_ptr()
                    .add((stem_idx.wrapping_shl(1) as usize) * A::VALUE_WIDTH_BYTES);
                prefetch_t0(nxt_ptr);
            },
            1 => unsafe {
                let nxt_ptr = stems_ptr
                    .as_ptr()
                    .add((stem_idx.wrapping_shl(1) as usize) * A::VALUE_WIDTH_BYTES);
                prefetch_t1(nxt_ptr);
            },
            _ => {}
        };

        match PF2 {
            0 => unsafe {
                let far_ptr = stems_ptr
                    .as_ptr()
                    .add((stem_idx.wrapping_shl(4) as usize) * A::VALUE_WIDTH_BYTES);
                prefetch_t0(far_ptr);
            },
            1 => unsafe {
                let far_ptr = stems_ptr
                    .as_ptr()
                    .add((stem_idx.wrapping_shl(4) as usize) * A::VALUE_WIDTH_BYTES);
                prefetch_t1(far_ptr);
            },
            _ => {}
        };
    }

    #[allow(missing_docs)]
    #[inline(always)]
    pub fn step_pure<A: Axis>(stem_idx: u32, is_right_child: bool, stems_ptr: NonNull<u8>) -> u32 {
        let result = stem_idx.wrapping_shl(1) | is_right_child as u32;

        Self::prefetch_descendants::<A>(result, stems_ptr);

        result
    }
}

/// Exposed pure function for use with cargo-asm.
#[doc(hidden)]
#[inline(never)]
pub fn calc_child_idx(curr_idx: u32, is_right_child: bool, stems_ptr: NonNull<u8>) -> u32 {
    EytzingerFlexPf::<0, 1>::step_pure::<f64>(curr_idx, is_right_child, stems_ptr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn branch_relative_selects_near_child_and_returns_far_child() {
        let stems = vec![0.0f64; 1024];
        let stems_ptr = NonNull::new(stems.as_ptr().cast_mut().cast()).unwrap();

        for is_right in [false, true] {
            let mut state = Eytzinger::new(stems_ptr);
            let far = state.branch_relative::<f64, 3>(is_right);

            assert_eq!(state.stem_idx(), if is_right { 3 } else { 2 });
            assert_eq!(far.stem_idx(), if is_right { 2 } else { 3 });
            assert_eq!(state.level(), 1);
            assert_eq!(far.level(), 1);
            assert_eq!(state.dim::<3>(), 1);
            assert_eq!(far.dim::<3>(), 1);
        }
    }
}

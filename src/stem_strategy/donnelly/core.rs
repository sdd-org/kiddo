use crate::stem_strategy::prefetch::prefetch_t1;
use crate::{Axis, StemStrategy};
use std::ptr::NonNull;

/// Donnelly Strategy - Core
///
/// Inner implementation that holds state and core logic.
/// - BH: Block height, in rows
#[derive(Copy, Clone, Debug)]
pub(crate) struct DonnellyCore<const BH: usize> {
    stem_idx: u32,
    dim: usize,
    level: i32,
    minor_level: u32,
    leaf_idx: u32,
    stems_ptr: NonNull<u8>,
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct DonnellyCoreDeferred {
    /// `leaf_idx` in bits 0..32, `stem_idx` in bits 32..64.
    indices: u64,
    /// `dim` in bits 0..16, `level` in bits 16..24, `minor_level` in bits 24..32.
    meta: u32,
}

// SAFETY: NonNull<u8> is not Send or Sync, preventing DonnellyCore from being automatically
// Send & Sync. But, we can safely manually declare DonnellyCore as Send and Sync here
// because we are only using it with prefetch instructions, which do not deref the pointer
// and are guaranteed to succeed even with an invalid pointer
unsafe impl<const BH: usize> Send for DonnellyCore<BH> {}
unsafe impl<const BH: usize> Sync for DonnellyCore<BH> {}

impl<const BH: usize> StemStrategy for DonnellyCore<BH> {
    const ROOT_IDX: usize = 0;

    type DeferredState = DonnellyCoreDeferred;
    type StackContext<A> = crate::kd_tree::query_stack::QueryStackContext<A, Self::DeferredState>;
    type Stack<A> = crate::kd_tree::query_stack::QueryStack<A, Self>;

    fn new(stems_ptr: NonNull<u8>) -> Self {
        assert!(
            BH <= u8::MAX as usize,
            "Donnelly block height must fit in packed deferred state"
        );

        Self {
            stem_idx: Self::ROOT_IDX as u32,
            dim: 0,
            level: 0,
            minor_level: 0,
            leaf_idx: 0,
            stems_ptr,
        }
    }

    #[inline(always)]
    fn stem_idx(&self) -> usize {
        self.stem_idx as usize
    }
    fn deferred_state(&self) -> Self::DeferredState {
        debug_assert!(self.dim <= u16::MAX as usize);
        debug_assert!((0..u32::BITS as i32).contains(&self.level));
        debug_assert!(self.minor_level <= u8::MAX as u32);

        DonnellyCoreDeferred {
            indices: u64::from(self.leaf_idx) | (u64::from(self.stem_idx) << 32),
            meta: self.dim as u32 | ((self.level as u32) << 16) | (self.minor_level << 24),
        }
    }
    fn rehydrate_deferred_state(&mut self, state: Self::DeferredState) {
        self.leaf_idx = state.indices as u32;
        self.stem_idx = (state.indices >> 32) as u32;
        self.dim = (state.meta & u16::MAX as u32) as usize;
        self.level = ((state.meta >> 16) & u8::MAX as u32) as i32;
        self.minor_level = state.meta >> 24;
    }

    #[inline(always)]
    fn leaf_idx(&self) -> usize {
        self.leaf_idx as usize
    }

    #[inline(always)]
    fn dim<const K: usize>(&self) -> usize {
        assert!(
            K <= u16::MAX as usize + 1,
            "Donnelly arithmetic queries require dimensions to fit in u16"
        );
        self.dim
    }

    #[inline(always)]
    fn level(&self) -> i32 {
        self.level
    }

    #[inline(always)]
    fn traverse<A: Axis<Coord = A>, const K: usize>(&mut self, is_right: bool) {
        let (idx, lvl) = Self::step_pure(self.stem_idx, self.minor_level, is_right, self.stems_ptr);
        self.stem_idx = idx;
        self.minor_level = lvl;

        self.level = self.level.wrapping_add(1);

        if K == BH {
            self.dim = lvl as usize;
        } else {
            let wrap_dim_mask = 0usize.wrapping_sub((self.dim == (K - 1)) as usize);
            self.dim = self.dim.wrapping_add(1) & !wrap_dim_mask;
        }

        self.leaf_idx = self.leaf_idx.wrapping_shl(1) | is_right as u32;
    }

    /// Used when running loop-unrolled
    ///
    /// PRECONDITIONS: assumes that
    /// * we stay within a minor triangle;
    /// * we don't hit the bottom level of the tree as a whole
    #[inline(always)]
    fn traverse_head<A: crate::Axis<Coord = A>, const K: usize>(&mut self, is_right: bool) {
        let (idx, lvl) =
            Self::step_pure_head(self.stem_idx, self.minor_level, is_right, self.stems_ptr);
        self.stem_idx = idx;
        self.minor_level = lvl;

        self.level = self.level.wrapping_add(1);

        if K == BH {
            self.dim = lvl as usize;
        } else {
            let wrap_dim_mask = 0usize.wrapping_sub((self.dim == (K - 1)) as usize);
            self.dim = self.dim.wrapping_add(1) & !wrap_dim_mask;
        }

        self.leaf_idx = self.leaf_idx.wrapping_shl(1) | is_right as u32;
    }

    #[inline(always)]
    fn branch<A: Axis, const K: usize>(&mut self) -> Self {
        let (left, right) = Self::both_children_pure(self.stem_idx, self.minor_level);

        // mutate self into left
        self.stem_idx = left;
        self.minor_level = (self.minor_level + 1)
            & !(0u32.wrapping_sub((self.minor_level + 1u32 == BH as u32) as u32));

        self.level = self.level.wrapping_add(1);

        if K == BH {
            self.dim = self.minor_level as usize;
        } else {
            let wrap_dim_mask = 0usize.wrapping_sub((self.dim == (K - 1)) as usize);
            self.dim = self.dim.wrapping_add(1) & !wrap_dim_mask;
        }

        self.leaf_idx = self.leaf_idx.wrapping_shl(1);

        // return right child as a new strategy
        Self {
            stem_idx: right,
            leaf_idx: self.leaf_idx | 1,
            ..*self
        }
    }

    #[inline(always)]
    fn branch_relative<A: Axis<Coord = A>, const K: usize>(&mut self, is_right: bool) -> Self {
        let (left_idx, right_idx, minor_level) =
            Self::both_children_predictable(self.stem_idx, self.minor_level);

        self.level = self.level.wrapping_add(1);
        self.minor_level = minor_level;
        if K == BH {
            self.dim = minor_level as usize;
        } else {
            let wrap_dim_mask = 0usize.wrapping_sub((self.dim == (K - 1)) as usize);
            self.dim = self.dim.wrapping_add(1) & !wrap_dim_mask;
        }

        let (near_idx, far_idx) = if is_right {
            (right_idx, left_idx)
        } else {
            (left_idx, right_idx)
        };
        let left_leaf_idx = self.leaf_idx.wrapping_shl(1);
        let right_leaf_idx = left_leaf_idx | 1;
        let (near_leaf_idx, far_leaf_idx) = if is_right {
            (right_leaf_idx, left_leaf_idx)
        } else {
            (left_leaf_idx, right_leaf_idx)
        };

        self.stem_idx = near_idx;
        self.leaf_idx = near_leaf_idx;

        Self {
            stem_idx: far_idx,
            leaf_idx: far_leaf_idx,
            ..*self
        }
    }

    #[inline(always)]
    fn child_indices<A: Axis>(&self) -> (usize, usize) {
        let res = DonnellyCore::<BH>::both_children_pure(self.stem_idx, self.minor_level);
        (res.0 as usize, res.1 as usize)
    }
}

impl<const BH: usize> DonnellyCore<BH> {
    #[inline(always)]
    pub(crate) fn minor_level(&self) -> u32 {
        self.minor_level
    }

    /// Traverse an entire block at once
    ///
    /// - `child_idx`: index of the child block to traverse to the root of
    /// - `block_size`: block height in levels
    ///
    /// We use the same dimension for the whole block, incrementing it for the next block
    ///
    /// PRECONDITIONS:
    /// - Tree height is padded to block boundary
    /// - Traversals must be exclusively block mode or per-level, not mixed
    #[allow(unused)] // used when simd feature is on
    #[inline(always)]
    pub(crate) fn traverse_block<const K: usize>(&mut self, child_idx: u8, block_size: u32) {
        // debug_assert_eq!(
        //     block_size,
        //     Self::BLOCK_SIZE as u32,
        //     "Block size ({block_size}) must match BLOCK_SIZE constant ({})",
        //     Self::BLOCK_SIZE
        // );
        debug_assert!(child_idx < (1u8 << block_size));
        debug_assert_eq!(self.minor_level, 0);
        debug_assert_eq!(self.stem_idx & Self::line_mask(), 0);

        // TODO: this used to call step_pure_block. Either remove step_pure_block or factor
        //       this code back into it
        let major_base = self.stem_idx & Self::line_mask_inv();

        let major_offset = Self::items_per_line()
            .wrapping_sub(block_size.wrapping_shl(1))
            .wrapping_sub(1);

        self.stem_idx = major_base
            .wrapping_add(major_offset)
            .wrapping_add(child_idx as u32)
            .wrapping_shl(BH as u32);

        self.minor_level = 0;
        self.level = self.level.wrapping_add(block_size as i32);

        let wrap_dim_mask = 0usize.wrapping_sub((self.dim == (K - 1)) as usize);
        self.dim = (self.dim + 1) & !wrap_dim_mask;

        self.leaf_idx = self.leaf_idx.wrapping_shl(block_size) | child_idx as u32;
    }

    /// Used when running loop-unrolled
    ///
    /// PRECONDITIONS: assumes that
    /// * we are on the bottom level of a minor triangle
    #[inline(always)]
    pub(crate) fn traverse_tail_with_block_size<A: Axis, const K: usize>(
        &mut self,
        is_right: bool,
        block_size: u32,
    ) {
        let (idx, lvl) = Self::step_pure_tail::<A, K>(
            block_size,
            self.stem_idx,
            self.minor_level,
            is_right,
            self.stems_ptr,
        );
        self.stem_idx = idx;
        self.minor_level = lvl;

        self.level = self.level.wrapping_add(1);

        if K == BH {
            self.dim = lvl as usize;
        } else {
            let wrap_dim_mask = 0usize.wrapping_sub((self.dim == (K - 1)) as usize);
            self.dim = self.dim.wrapping_add(1) & !wrap_dim_mask;
        }

        self.leaf_idx = self.leaf_idx.wrapping_shl(1) | is_right as u32;
    }

    #[inline(always)]
    const fn items_per_line() -> u32 {
        1u32 << BH
        // CL / (A::VALUE_WIDTH_BYTES as u32)
    }
    // #[inline(always)]
    // const fn log2_items_per_line() -> u32 {
    //     Self::items_per_line().ilog2()
    // }
    #[inline(always)]
    const fn line_mask() -> u32 {
        Self::items_per_line() - 1
    }
    #[inline(always)]
    const fn line_mask_inv() -> u32 {
        !Self::line_mask()
    }

    #[inline(always)]
    fn step_pure(
        curr_idx: u32,
        mut minor_level: u32,
        is_right_child: bool,
        _stems_ptr: NonNull<u8>,
    ) -> (u32, u32) {
        let is_right_child = u32::from(is_right_child);

        // index into current minor triangle / cache line
        let min_idx = curr_idx & Self::line_mask();

        // column in current minor triangle
        let min_col_idx = min_idx.wrapping_sub(minor_level).wrapping_sub(1);

        let base_no_right = (curr_idx & Self::line_mask_inv()).wrapping_add(1);
        let next_prefetch_base = base_no_right
            .wrapping_add(min_col_idx.wrapping_shl(1))
            .wrapping_shl(BH as u32);

        let base_with_side: u32 = base_no_right.wrapping_add(is_right_child);
        let same_base = base_with_side.wrapping_add(min_idx.wrapping_shl(1));

        let next_result_base =
            next_prefetch_base.wrapping_add(is_right_child.wrapping_shl(BH as u32));

        let inc_major_level = (minor_level.wrapping_add(1) == BH as u32) as u32;
        let inc_major_level_mask = 0u32.wrapping_sub(inc_major_level);

        let result =
            (next_result_base & inc_major_level_mask) | (same_base & !inc_major_level_mask);

        minor_level = minor_level.wrapping_add(1);
        minor_level &= !inc_major_level_mask;

        (result, minor_level)
    }

    #[inline(always)]
    fn step_pure_head(
        curr_idx: u32,
        mut minor_level: u32,
        is_right_child: bool,
        _stems_ptr: NonNull<u8>,
    ) -> (u32, u32) {
        let is_right_child = u32::from(is_right_child);

        // index into current minor triangle / cache line
        let minor_idx = curr_idx & Self::line_mask();

        let base_no_right = (curr_idx & Self::line_mask_inv()).wrapping_add(1);

        let base_with_side: u32 = base_no_right.wrapping_add(is_right_child);
        let result = base_with_side.wrapping_add(minor_idx.wrapping_shl(1));

        minor_level = minor_level.wrapping_add(1);
        // println!("is_right_child: {is_right_child}, min_idx: {minor_idx}, base_no_right: {base_no_right}, base_with_side: {base_with_side}, next stem_idx: {result}");

        (result, minor_level)
    }

    #[inline(always)]
    fn step_pure_tail<A: Axis, const K: usize>(
        block_size: u32,
        curr_idx: u32,
        mut minor_level: u32,
        is_right_child: bool,
        stems_ptr: NonNull<u8>,
    ) -> (u32, u32) {
        let is_right_child = u32::from(is_right_child);

        // index into current minor triangle / cache line
        let min_idx = curr_idx & Self::line_mask();

        // row in current minor triangle
        let min_row_idx = min_idx.wrapping_sub(minor_level).wrapping_sub(1);

        let base_no_right = (curr_idx & Self::line_mask_inv()).wrapping_add(1);
        let next_prefetch_base = base_no_right
            .wrapping_add(min_row_idx.wrapping_shl(1))
            .wrapping_shl(BH as u32);

        let result = next_prefetch_base.wrapping_add(is_right_child.wrapping_shl(BH as u32));

        // Prefetch result? Not much point, it's likely gonna be requested within 1 cycle
        // unsafe {
        //     let nxt_ptr = stems_ptr
        //         .as_ptr()
        //         .add((result * VB) as usize);
        //     prefetch_t0(nxt_ptr);
        // }

        // Prefetch deeper-level 8 base ptrs to L2
        let next_base_no_right = (result & Self::line_mask_inv()).wrapping_add(7);
        let next_next_prefetch_base = next_base_no_right.wrapping_shl(BH as u32);

        Self::prefetch_next_base::<A>(
            stems_ptr,
            next_next_prefetch_base,
            2u32.pow(block_size) as usize,
        );

        // println!("is_right_child: {is_right_child}, min_idx: {min_idx}, min_row_idx: {min_row_idx}, base_no_right: {base_no_right}, next_prefetch_base: {next_prefetch_base}, next stem_idx: {result}");
        // println!("next_next_prefetch_base: {next_next_prefetch_base} -> {}", next_next_prefetch_base + 128);

        minor_level = 0;

        (result, minor_level)
    }

    #[allow(unused)]
    #[inline(always)]
    fn step_pure_block(curr_idx: u32, child_idx: u8) -> u32 {
        curr_idx
            .wrapping_add(1)
            .wrapping_shl(BH as u32)
            .wrapping_add((child_idx as u32).wrapping_shl(BH as u32))
    }

    #[inline(always)]
    fn prefetch_next_base<A: Axis>(
        stems_ptr: NonNull<u8>,
        next_base: u32,
        cache_line_count: usize,
    ) {
        #[cfg(target_arch = "x86_64")]
        const BYTES_PER_LINE: usize = 64;

        // TODO: auto-detect M2+ where this is 128 bytes
        #[cfg(target_arch = "aarch64")]
        const BYTES_PER_LINE: usize = 64; // 64 for most ARM, 128 for Apple M-series

        let base_ptr = unsafe {
            stems_ptr
                .as_ptr()
                .add((next_base as usize) * A::VALUE_WIDTH_BYTES)
        };

        for i in 0..cache_line_count {
            let ptr = unsafe { base_ptr.add(i * BYTES_PER_LINE) };
            unsafe { prefetch_t1(ptr) };
        }
    }

    /// Two-children step in one pass (left=false, right=true).
    /// Advances minor_level once; does NOT change curr_idx (so caller can choose a child later).
    #[inline(always)]
    pub(crate) fn both_children_pure(curr_idx: u32, minor_level: u32) -> (u32, u32) {
        // precompute pieces identical to step_pure
        let line_mask = Self::line_mask();
        let line_mask_inv = Self::line_mask_inv();

        let min_idx = curr_idx & line_mask;
        let min_row_idx = min_idx.wrapping_sub(minor_level).wrapping_sub(1);

        let inc_major = (minor_level.wrapping_add(1) == BH as u32) as u32;
        let inc_mask = 0u32.wrapping_sub(inc_major);

        let base_no_right = (curr_idx & line_mask_inv).wrapping_add(1);

        // same-block left/right
        let same_left = base_no_right.wrapping_add(min_idx.wrapping_shl(1));
        let same_right = same_left.wrapping_add(1);

        // next-block left/right (note: add right after shift by L)
        let next_pre = base_no_right.wrapping_add(min_row_idx.wrapping_shl(1));
        let next_left = next_pre.wrapping_shl(BH as u32);
        let next_right = next_left.wrapping_add(1u32.wrapping_shl(BH as u32));

        // masked select between same/next for both children
        let left = (same_left & !inc_mask) | (next_left & inc_mask);
        let right = (same_right & !inc_mask) | (next_right & inc_mask);

        (left, right)
    }

    /// Compute both children with a predictable block-boundary branch.
    ///
    /// Exact traversal visits block phases in a fixed cycle, so this avoids
    /// evaluating both the same-line and next-line recurrences at every level.
    #[inline(always)]
    pub(crate) fn both_children_predictable(curr_idx: u32, minor_level: u32) -> (u32, u32, u32) {
        let line_base = curr_idx & Self::line_mask_inv();
        let local_idx = curr_idx & Self::line_mask();
        let next_minor_level = minor_level + 1;

        if next_minor_level == BH as u32 {
            let path_prefix = local_idx.wrapping_sub(minor_level).wrapping_sub(1);
            let left_idx = line_base
                .wrapping_add(1)
                .wrapping_add(path_prefix.wrapping_shl(1))
                .wrapping_shl(BH as u32);
            (
                left_idx,
                left_idx.wrapping_add(1u32.wrapping_shl(BH as u32)),
                0,
            )
        } else {
            let left_idx = line_base
                .wrapping_add(1)
                .wrapping_add(local_idx.wrapping_shl(1));
            (left_idx, left_idx.wrapping_add(1), next_minor_level)
        }
    }
}

/// Exposed pure function for use with cargo-asm
#[inline(never)]
pub fn calc_child_idx_hook(
    curr_idx: u32,
    minor_index: u32,
    is_right_child: bool,
    stems_ptr: NonNull<u8>,
) -> (u32, u32) {
    DonnellyCore::<3>::step_pure(curr_idx, minor_index, is_right_child, stems_ptr)
}

/// Exposed pure function for use with cargo-asm
#[inline(never)]
pub fn both_children_pure_hook(curr_idx: u32, minor_index: u32) -> (u32, u32) {
    DonnellyCore::<3>::both_children_pure(curr_idx, minor_index)
}

/// Exposed pure function for use with cargo-asm
#[inline(never)]
pub fn test_traverse_hook(is_right_child: bool, stems: *mut u8) -> usize {
    let stems_ptr = NonNull::new(stems).unwrap();

    let mut stem_strat = DonnellyCore::<3>::new(stems_ptr);

    stem_strat.traverse::<f64, 3>(is_right_child);
    stem_strat.traverse::<f64, 3>(!is_right_child);
    stem_strat.traverse::<f64, 3>(is_right_child);

    stem_strat.stem_idx()
}

#[cfg(test)]
mod tests {
    use super::*;
    use aligned_vec::avec;
    use rstest::rstest;

    fn assert_branch_relative_matches_default<const BH: usize, const K: usize>() {
        for seed in 0usize..32 {
            let mut state = DonnellyCore::<BH>::new_no_ptr();
            for level in 0usize..20 {
                for is_right in [false, true] {
                    let mut expected_near = state;
                    let mut expected_far = expected_near.branch::<f64, K>();
                    if is_right {
                        std::mem::swap(&mut expected_near, &mut expected_far);
                    }

                    let mut actual_near = state;
                    let actual_far = actual_near.branch_relative::<f64, K>(is_right);

                    assert_eq!(actual_near.stem_idx, expected_near.stem_idx);
                    assert_eq!(actual_near.leaf_idx, expected_near.leaf_idx);
                    assert_eq!(actual_near.dim, expected_near.dim);
                    assert_eq!(actual_near.level, expected_near.level);
                    assert_eq!(actual_near.minor_level, expected_near.minor_level);
                    assert_eq!(actual_far.stem_idx, expected_far.stem_idx);
                    assert_eq!(actual_far.leaf_idx, expected_far.leaf_idx);
                    assert_eq!(actual_far.dim, expected_far.dim);
                    assert_eq!(actual_far.level, expected_far.level);
                    assert_eq!(actual_far.minor_level, expected_far.minor_level);
                }

                let is_right = (seed.wrapping_mul(17) + level.wrapping_mul(29)) & 4 != 0;
                state.traverse::<f64, K>(is_right);
            }
        }
    }

    #[test]
    fn branch_relative_matches_default_across_block_boundaries() {
        assert_branch_relative_matches_default::<3, 3>();
        assert_branch_relative_matches_default::<3, 5>();
        assert_branch_relative_matches_default::<4, 3>();
        assert_branch_relative_matches_default::<4, 4>();
    }

    fn assert_compact_deferred_state_round_trips<const BH: usize>() {
        assert_eq!(std::mem::size_of::<DonnellyCoreDeferred>(), 16);
        assert_eq!(std::mem::size_of::<DonnellyCore<BH>>(), 32);

        for seed in 0usize..32 {
            let mut state = DonnellyCore::<BH>::new_no_ptr();
            for level in 0usize..20 {
                let expected = state;
                let saved = state.deferred_state();

                let mut restored = DonnellyCore::<BH>::new_no_ptr();
                restored.rehydrate_deferred_state(saved);
                assert_eq!(restored.stem_idx, expected.stem_idx);
                assert_eq!(restored.leaf_idx, expected.leaf_idx);
                assert_eq!(restored.dim, expected.dim);
                assert_eq!(restored.level, expected.level);
                assert_eq!(restored.minor_level, expected.minor_level);

                let is_right = (seed.wrapping_mul(17) + level.wrapping_mul(29)) & 4 != 0;
                state.traverse::<f64, 3>(is_right);
            }
        }
    }

    #[test]
    fn compact_deferred_state_round_trips_for_supported_block_heights() {
        assert_compact_deferred_state_round_trips::<3>();
        assert_compact_deferred_state_round_trips::<4>();
    }

    #[rstest]
    #[case(vec![], 0)]
    #[case(vec![false], 1)] // 1 Maj idx: 1
    #[case(vec![true], 2)] // 2
    #[case(vec![false, false], 3)] // 3
    #[case(vec![false, true], 4)] // 4
    #[case(vec![true, false], 5)] // 5
    #[case(vec![true, true], 6)] // 6
    #[case(vec![false, false, false], 8)] // 7
    #[case(vec![false, false, true], 16)] // 8
    #[case(vec![false, true, false], 24)] // 9
    #[case(vec![false, true, true], 32)] // 10
    #[case(vec![true, false, false], 40)] // 11
    #[case(vec![true, false, true], 48)] // 12
    #[case(vec![true, true, false], 56)] // 13
    #[case(vec![true, true, true], 64)] // 14
    #[case(vec![false, false, false, false], 9)] // 15 Maj idx: 2
    #[case(vec![false, false, false, true], 10)] // 16
    #[case(vec![false, false, false, false, false], 11)] // 17
    #[case(vec![false, false, false, false, true], 12)] // 18
    #[case(vec![false, false, false, true, false], 13)] // 19
    #[case(vec![false, false, false, true, true], 14)] // 20
    #[case(vec![false, false, false, false, false, false], 72)] // 21
    #[case(vec![false, false, false, false, false, true], 80)] // 22
    #[case(vec![false, false, false, false, true, false], 88)] // 23
    #[case(vec![false, false, false, false, true, true], 96)] // 24
    #[case(vec![false, false, false, true, false, false], 104)] // 25
    #[case(vec![false, false, false, true, false, true], 112)] // 26
    #[case(vec![false, false, false, true, true, false], 120)] // 27
    #[case(vec![false, false, false, true, true, true], 128)] // 28
    #[case(vec![false, false, true, false], 17)] // 29  Maj index: 3
    #[case(vec![false, false, true, true], 18)] // 30
    #[case(vec![false, false, true, false, false], 19)] // 31
    #[case(vec![false, false, true, false, true], 20)] // 32
    #[case(vec![false, false, true, true, false], 21)] // 33
    #[case(vec![false, false, true, true, true], 22)] // 34
    #[case(vec![false, false, true, false, false, false], 136)] // 35
    #[case(vec![false, false, true, false, false, true], 144)] // 36
    #[case(vec![false, false, true, false, true, false], 152)] // 37
    #[case(vec![false, false, true, false, true, true], 160)] // 38
    #[case(vec![false, false, true, true, false, false], 168)] // 39
    #[case(vec![false, false, true, true, false, true], 176)] // 40
    #[case(vec![false, false, true, true, true, false], 184)] // 41
    #[case(vec![false, false, true, true, true, true], 192)] // 42
    fn donnelly_core_get_child_idx_produces_correct_values(
        #[case] input: Vec<bool>,
        #[case] expected: usize,
    ) {
        let stems = avec![f64::INFINITY; 9];
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();

        let mut stem_strat = DonnellyCore::<3>::new(stems_ptr);
        let mut result = 0;
        input.iter().for_each(|selection| {
            stem_strat.traverse::<f64, 3>(*selection);
            result = stem_strat.stem_idx();
        });

        assert_eq!(result, expected);
    }

    #[rstest]
    #[case(vec![], (1, 2))]
    #[case(vec![false], (3, 4))] // 1 Maj idx: 1
    #[case(vec![true], (5, 6))] // 2
    #[case(vec![false, false], (8, 16))] // 3
    #[case(vec![false, true], (24, 32))] // 4
    #[case(vec![true, false], (40, 48))] // 5
    #[case(vec![true, true], (56, 64))] // 6
    #[case(vec![false, false, false], (9, 10))] // 7
    #[case(vec![false, false, true], (17, 18))] // 8
    #[case(vec![false, true, false], (25, 26))] // 9
    #[case(vec![false, true, true], (33, 34))] // 10
    #[case(vec![true, false, false], (41, 42))] // 11
    #[case(vec![true, false, true], (49, 50))] // 12
    #[case(vec![true, true, false], (57, 58))] // 13
    #[case(vec![true, true, true], (65, 66))] // 14
    #[case(vec![false, false, false, false], (11, 12))] // 15 Maj idx: 2
    #[case(vec![false, false, false, true], (13, 14))] // 16
    #[case(vec![false, false, false, false, false], (72, 80))] // 17
    #[case(vec![false, false, false, false, true], (88, 96))] // 18
    #[case(vec![false, false, false, true, false], (104, 112))] // 19
    #[case(vec![false, false, false, true, true], (120, 128))] // 20
    #[case(vec![false, false, false, false, false, false], (73, 74))] // 21
    #[case(vec![false, false, false, false, false, true], (81, 82))] // 22
    #[case(vec![false, false, false, false, true, false], (89, 90))] // 23
    #[case(vec![false, false, false, false, true, true], (97, 98))] // 24
    #[case(vec![false, false, false, true, false, false], (105, 106))] // 25
    #[case(vec![false, false, false, true, false, true], (113, 114))] // 26
    #[case(vec![false, false, false, true, true, false], (121, 122))] // 27
    #[case(vec![false, false, false, true, true, true], (129, 130))] // 28
    #[case(vec![false, false, true, false], (19, 20))] // 29  Maj index: 3
    #[case(vec![false, false, true, true], (21, 22))] // 30
    #[case(vec![false, false, true, false, false], (136, 144))] // 31
    #[case(vec![false, false, true, false, true], (152, 160))] // 32
    #[case(vec![false, false, true, true, false], (168, 176))] // 33
    #[case(vec![false, false, true, true, true], (184, 192))] // 34
    #[case(vec![false, false, true, false, false, false], (137, 138))] // 35
    #[case(vec![false, false, true, false, false, true], (145, 146))] // 36
    #[case(vec![false, false, true, false, true, false], (153, 154))] // 37
    #[case(vec![false, false, true, false, true, true], (161, 162))] // 38
    #[case(vec![false, false, true, true, false, false], (169, 170))] // 39
    #[case(vec![false, false, true, true, false, true], (177, 178))] // 40
    #[case(vec![false, false, true, true, true, false], (185, 186))] // 41
    #[case(vec![false, false, true, true, true, true], (193, 194))] // 42
    fn donnelly_core_get_both_child_idxs_produces_correct_values(
        #[case] input: Vec<bool>,
        #[case] expected: (usize, usize),
    ) {
        let stems = avec![f64::INFINITY; 9];
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();

        let mut stem_strat = DonnellyCore::<3>::new(stems_ptr);
        // let mut stem_strat = Donnelly::<3, 64, 4, 4>::new();

        // let last = input.last().unwrap();
        input.iter().for_each(|selection| {
            stem_strat.branch_relative::<f64, 3>(*selection);
        });

        let results = stem_strat.split::<f64, 3>();
        let result = (results.0.stem_idx(), results.1.stem_idx());

        assert_eq!(result, expected);
    }

    #[rstest]
    #[case(vec![], 0)]
    #[case(vec![false], 1)] // 1 Maj idx: 1
    #[case(vec![true], 2)] // 2
    #[case(vec![false, false], 3)] // 3
    #[case(vec![false, true], 4)] // 4
    #[case(vec![true, false], 5)] // 5
    #[case(vec![true, true], 6)] // 6
    #[case(vec![false, false, false], 8)] // 7
    #[case(vec![false, false, true], 16)] // 8
    #[case(vec![false, true, false], 24)] // 9
    #[case(vec![false, true, true], 32)] // 10
    #[case(vec![true, false, false], 40)] // 11
    #[case(vec![true, false, true], 48)] // 12
    #[case(vec![true, true, false], 56)] // 13
    #[case(vec![true, true, true], 64)] // 14
    #[case(vec![false, false, false, false], 9)] // 15 Maj idx: 2
    #[case(vec![false, false, false, true], 10)] // 16
    #[case(vec![false, false, false, false, false], 11)] // 17
    #[case(vec![false, false, false, false, true], 12)] // 18
    #[case(vec![false, false, false, true, false], 13)] // 19
    #[case(vec![false, false, false, true, true], 14)] // 20
    #[case(vec![false, false, false, false, false, false], 72)] // 21
    #[case(vec![false, false, false, false, false, true], 80)] // 22
    #[case(vec![false, false, false, false, true, false], 88)] // 23
    #[case(vec![false, false, false, false, true, true], 96)] // 24
    #[case(vec![false, false, false, true, false, false], 104)] // 25
    #[case(vec![false, false, false, true, false, true], 112)] // 26
    #[case(vec![false, false, false, true, true, false], 120)] // 27
    #[case(vec![false, false, false, true, true, true], 128)] // 28
    #[case(vec![false, false, true, false], 17)] // 29  Maj index: 3
    #[case(vec![false, false, true, true], 18)] // 30
    #[case(vec![false, false, true, false, false], 19)] // 31
    #[case(vec![false, false, true, false, true], 20)] // 32
    #[case(vec![false, false, true, true, false], 21)] // 33
    #[case(vec![false, false, true, true, true], 22)] // 34
    #[case(vec![false, false, true, false, false, false], 136)] // 35
    #[case(vec![false, false, true, false, false, true], 144)] // 36
    #[case(vec![false, false, true, false, true, false], 152)] // 37
    #[case(vec![false, false, true, false, true, true], 160)] // 38
    #[case(vec![false, false, true, true, false, false], 168)] // 39
    #[case(vec![false, false, true, true, false, true], 176)] // 40
    #[case(vec![false, false, true, true, true, false], 184)] // 41
    #[case(vec![false, false, true, true, true, true], 192)] // 42
    fn donnelly_core_get_child_idx_unrolled_produces_correct_values(
        #[case] input: Vec<bool>,
        #[case] expected: usize,
    ) {
        let stems = avec![f64::INFINITY; 9];
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();

        let mut stem_strat = DonnellyCore::<3>::new(stems_ptr);
        let mut result = 0;
        let mut minor_tri_idx = 0;
        input.iter().for_each(|selection| {
            if minor_tri_idx == 2 {
                stem_strat.traverse_tail::<f64, 3>(*selection);
                minor_tri_idx = 0;
            } else {
                minor_tri_idx += 1;
                stem_strat.traverse_head::<f64, 3>(*selection);
            }

            result = stem_strat.stem_idx();
        });

        assert_eq!(result, expected);
    }

    #[rstest]
    #[case(vec![], 0)]
    #[case(vec![0], 8)] // 1
    #[case(vec![1], 16)] // 2
    #[case(vec![2], 24)] // 3
    #[case(vec![3], 32)] // 4
    #[case(vec![4], 40)] // 5
    #[case(vec![5], 48)] // 6
    #[case(vec![6], 56)] // 7
    #[case(vec![7], 64)] // 8
    #[case(vec![0, 0], 72)] // 9
    #[case(vec![0, 1], 80)] // 10
    #[case(vec![0, 2], 88)] // 11
    #[case(vec![0, 3], 96)] // 12
    #[case(vec![0, 4], 104)] // 13
    #[case(vec![0, 5], 112)] // 14
    #[case(vec![0, 6], 120)] // 15
    #[case(vec![0, 7], 128)] // 16
    #[case(vec![1, 0], 136)] // 17
    #[case(vec![1, 1], 144)] // 18
    #[case(vec![1, 2], 152)] // 19
    #[case(vec![1, 3], 160)] // 20
    #[case(vec![1, 4], 168)] // 21
    #[case(vec![1, 5], 176)] // 22
    #[case(vec![1, 6], 184)] // 23
    #[case(vec![1, 7], 192)] // 24
    fn donnelly_core_traverse_block_produces_correct_values(
        #[case] input: Vec<u8>,
        #[case] expected: usize,
    ) {
        let stems = avec![f64::INFINITY; 9];
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();

        let mut stem_strat = DonnellyCore::<3>::new(stems_ptr);
        let mut result = 0;
        input.iter().for_each(|selection| {
            stem_strat.traverse_block::<3>(*selection, 3);
            result = stem_strat.stem_idx();
        });

        assert_eq!(result, expected);
    }

    #[test]
    fn regression_block4_traverse_block_matches_repeated_traverse_f32() {
        let stems = avec![f32::INFINITY; 2_048];
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();

        let start_paths: [&[bool]; 4] = [
            &[],
            &[false, false, false, false],
            &[false, false, false, true],
            &[true, true, true, true],
        ];

        for start_path in start_paths {
            let mut base = DonnellyCore::<4>::new(stems_ptr);
            for &is_right in start_path {
                base.traverse::<f32, 2>(is_right);
            }
            assert_eq!(base.level() % 4, 0, "base must be at block boundary");

            for child_idx in 0u8..16 {
                let mut block = base;
                block.traverse_block::<2>(child_idx, 4);

                let mut repeated = base;
                for shift in (0..4).rev() {
                    repeated.traverse::<f32, 2>(((child_idx >> shift) & 1) != 0);
                }

                assert_eq!(
                    block.stem_idx(),
                    repeated.stem_idx(),
                    "stem_idx mismatch for start_path={:?}, child_idx={}",
                    start_path,
                    child_idx
                );
                assert_eq!(
                    block.leaf_idx(),
                    repeated.leaf_idx(),
                    "leaf_idx mismatch for start_path={:?}, child_idx={}",
                    start_path,
                    child_idx
                );
                assert_eq!(
                    block.level(),
                    repeated.level(),
                    "level mismatch for start_path={:?}, child_idx={}",
                    start_path,
                    child_idx
                );
            }
        }

        // Also verify deeper block boundaries (level 8) reached via two full block traversals.
        for block_a in 0u8..16 {
            for block_b in 0u8..16 {
                let mut base = DonnellyCore::<4>::new(stems_ptr);
                base.traverse_block::<2>(block_a, 4);
                base.traverse_block::<2>(block_b, 4);
                assert_eq!(base.level() % 4, 0, "base must be at block boundary");

                for child_idx in 0u8..16 {
                    let mut block = base;
                    block.traverse_block::<2>(child_idx, 4);

                    let mut repeated = base;
                    for shift in (0..4).rev() {
                        repeated.traverse::<f32, 2>(((child_idx >> shift) & 1) != 0);
                    }

                    assert_eq!(
                        block.stem_idx(),
                        repeated.stem_idx(),
                        "deep stem_idx mismatch for block_a={}, block_b={}, child_idx={}",
                        block_a,
                        block_b,
                        child_idx
                    );
                    assert_eq!(
                        block.leaf_idx(),
                        repeated.leaf_idx(),
                        "deep leaf_idx mismatch for block_a={}, block_b={}, child_idx={}",
                        block_a,
                        block_b,
                        child_idx
                    );
                }
            }
        }
    }

    #[test]
    fn regression_branch_relative_matches_traverse_children_block4_f32() {
        use crate::StemStrategy;

        let stems = avec![f32::INFINITY; 4_096];
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();

        // Exercise a broad set of states including block boundaries and tails.
        for path_len in 0..=12 {
            let combinations = 1usize << path_len.min(10);
            for bits in 0..combinations {
                let mut base = DonnellyCore::<4>::new(stems_ptr);
                for step in 0..path_len {
                    let is_right = if step < 10 {
                        (bits >> step) & 1 == 1
                    } else {
                        // Deterministic extension once the bit-combination cap is reached.
                        step % 2 == 1
                    };
                    base.traverse::<f32, 2>(is_right);
                }

                for &is_right in &[false, true] {
                    let mut branched = base;
                    let far = branched.branch_relative::<f32, 2>(is_right);
                    let near = branched;

                    let mut near_ref = base;
                    near_ref.traverse::<f32, 2>(is_right);

                    let mut far_ref = base;
                    far_ref.traverse::<f32, 2>(!is_right);

                    assert_eq!(near.stem_idx(), near_ref.stem_idx());
                    assert_eq!(near.level(), near_ref.level());
                    assert_eq!(near.leaf_idx(), near_ref.leaf_idx());

                    assert_eq!(far.stem_idx(), far_ref.stem_idx());
                    assert_eq!(far.level(), far_ref.level());
                    assert_eq!(far.leaf_idx(), far_ref.leaf_idx());
                }
            }
        }
    }
}

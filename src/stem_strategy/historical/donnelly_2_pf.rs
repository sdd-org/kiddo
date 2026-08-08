//! V2 Donnelly Stem Strategy with Prefetch

use std::ptr::NonNull;

use aligned_vec::AVec;

use crate::stem_strategy::prefetch::{prefetch_t0, prefetch_t1};
use crate::{Axis, StemStrategy};

/// Donnelly Strategy
///
/// A modification of the van Emde Boas layout, improved
/// for better cache sympathy.
/// - L:     levels per block
/// - CL:    cache line width in bytes (64 or 128)
/// - VB:    value width in bytes (4 or 8)
#[derive(Copy, Clone, Debug)]
pub struct DonnellyPf<const L: u32, const CL: u32, const VB: u32, const K: usize> {
    stem_idx: u32,
    dim: usize,
    level: i32,
    minor_level: u32,
    leaf_idx: usize,

    stems_ptr: NonNull<u8>,
}

// FIXME: this is a hack to make the compiler happy. remove after testing
unsafe impl<const L: u32, const CL: u32, const VB: u32, const K: usize> Send
    for DonnellyPf<L, CL, VB, K>
{
}
unsafe impl<const L: u32, const CL: u32, const VB: u32, const K: usize> Sync
    for DonnellyPf<L, CL, VB, K>
{
}

impl<const L: u32, const CL: u32, const VB: u32, const K: usize> StemStrategy
    for DonnellyPf<L, CL, VB, K>
{
    const ROOT_IDX: usize = 0;

    type DeferredState = Self;
    type StackContext<A, const K: usize> = crate::kd_tree::query_stack::QueryStackContext<A, Self::DeferredState>;
    type Stack<A, const K: usize> = crate::kd_tree::query_stack::QueryStack<A, Self, K>;

    fn new(stems_ptr: NonNull<u8>) -> Self {
        debug_assert!(L >= 2 && L <= 8);
        debug_assert!(CL > VB); // item wider than cache line would break layout

        Self {
            stem_idx: Self::ROOT_IDX as u32,
            dim: 0,
            level: 0,
            minor_level: 0,
            leaf_idx: 0,
            stems_ptr,
        }
    }

    fn stem_idx(&self) -> usize {
        self.stem_idx as usize
    }
    fn deferred_state(&self) -> Self::DeferredState {
        *self
    }
    fn rehydrate_deferred_state(&mut self, state: Self::DeferredState) {
        *self = state;
    }

    fn leaf_idx(&self) -> usize {
        self.leaf_idx
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn level(&self) -> i32 {
        self.level
    }

    fn traverse<A: Axis<Coord = A>, const K2: usize>(&mut self, is_right: bool) {
        let (idx, lvl) = Self::step_pure(self.stem_idx, self.minor_level, is_right, self.stems_ptr);
        self.stem_idx = idx;
        self.minor_level = lvl;

        self.level = self.level.wrapping_add(1);

        let wrap_dim_mask = 0usize.wrapping_sub((self.dim == (K - 1)) as usize);
        self.dim = self.dim.wrapping_add(1) & !wrap_dim_mask;

        self.leaf_idx = self.leaf_idx.wrapping_shl(1) | is_right as usize;
    }

    /// When running loop-unrolled, traverse_head operates under the assumption that
    /// we stay within a minor triangle and don't hit the bottom level of the tree as a whole
    fn traverse_head<A: Axis<Coord = A>, const K2: usize>(&mut self, is_right: bool) {
        let (idx, lvl) =
            Self::step_pure_head(self.stem_idx, self.minor_level, is_right, self.stems_ptr);
        self.stem_idx = idx;
        self.minor_level = lvl;

        // self.level = self.level.wrapping_add(1);

        let wrap_dim_mask = 0usize.wrapping_sub((self.dim == (K - 1)) as usize);
        self.dim = self.dim.wrapping_add(1) & !wrap_dim_mask;

        self.leaf_idx = self.leaf_idx.wrapping_shl(1) | is_right as usize;
    }

    /// When running loop-unrolled, traverse_head operates under the assumption that
    /// we are on the bottom level of a minor triangle
    fn traverse_tail<A: Axis<Coord = A>, const K2: usize>(&mut self, is_right: bool) {
        let (idx, lvl) =
            Self::step_pure_tail(self.stem_idx, self.minor_level, is_right, self.stems_ptr);
        self.stem_idx = idx;
        self.minor_level = lvl;

        self.level = self.level.wrapping_add(1);

        let wrap_dim_mask = 0usize.wrapping_sub((self.dim == (K - 1)) as usize);
        self.dim = self.dim.wrapping_add(1) & !wrap_dim_mask;

        self.leaf_idx = self.leaf_idx.wrapping_shl(1) | is_right as usize;
    }

    #[cfg(feature = "simulator")]
    fn simulate_traverse<A: Axis<Coord = A>, const K2: usize>(
        &mut self,
        is_right: bool,
        event_tx: &std::sync::mpsc::Sender<crate::test_utils::cache_simulator::Event>,
    ) {
        use crate::test_utils::cache_simulator::Event;

        // Execute the real traversal logic
        self.traverse::<A, K2>(is_right);

        // Emit synthetic "work" delay representing ~5 cycles of integer ops

        // ~5 ops estimate is slightly pessimistic rounding of 4.5 cycles coming from
        // MCA analysis of step_pure giving ~3.6 cycles, plus 1 cycle estimate for
        // level / dim / leaf_level update

        let _ = event_tx.send(Event::Working(5));
    }

    fn branch<const K2: usize>(&mut self) -> Self {
        let (left, right) = Self::both_children_pure(self.stem_idx, self.minor_level);

        // mutate self into left
        self.stem_idx = left;
        self.minor_level = (self.minor_level + 1)
            & !(0u32.wrapping_sub((self.minor_level + 1 == Self::log2_items_per_line()) as u32));

        self.level = self.level.wrapping_add(1);

        let wrap_dim_mask = 0usize.wrapping_sub((self.dim == (K - 1)) as usize);
        self.dim = self.dim.wrapping_add(1) & !wrap_dim_mask;

        self.leaf_idx = self.leaf_idx.wrapping_shl(1);

        // return right child as a new strategy
        Self {
            stem_idx: right,
            leaf_idx: self.leaf_idx | 1,
            ..*self
        }
    }

    fn branch_relative<const K2: usize>(&mut self, is_right: bool) -> Self {
        // precompute both children (left,right) at current (stem_idx, minor_level)
        let (left, right) = Self::both_children_pure(self.stem_idx, self.minor_level);

        // masks
        let m_r32 = mask32(is_right);
        let nm_r32 = !m_r32;
        let m_r64 = maskusize(is_right);
        let nm_r64 = !m_r64;

        // Which child is "near" vs "far" depends on the sign of (query[dim] - val):
        // is_right==false -> near=left,  far=right
        // is_right==true  -> near=right, far=left
        let near_stem = (left & nm_r32) | (right & m_r32);
        let far_stem = (right & nm_r32) | (left & m_r32);

        // minor_level update (same rule for both children)
        let ml1 = self.minor_level.wrapping_add(1);
        let at_boundary = ml1 == Self::log2_items_per_line();
        let m_b32 = mask32(at_boundary);
        let near_minor = ml1 & !m_b32; // zero when crossing to next block
        let far_minor = near_minor; // same progression for far side

        // level increments (both advance one)
        let next_level = self.level.wrapping_add(1);

        // dim update (branchless wrap)
        let wrap_dim_mask = 0usize.wrapping_sub((self.dim == (K - 1)) as usize);
        let next_dim = self.dim.wrapping_add(1) & !wrap_dim_mask;

        // leaf index: near takes bit 0, far takes bit 1
        let li2 = self.leaf_idx << 1;
        let near_leaf = (li2 & nm_r64) | ((li2 | 1) & m_r64); // if right -> OR 1
        let far_leaf = (li2 & m_r64) | ((li2 | 1) & nm_r64); // opposite bit

        // mutate self -> NEAR child
        self.stem_idx = near_stem;
        self.minor_level = near_minor;
        self.level = next_level;
        self.dim = next_dim;
        self.leaf_idx = near_leaf;

        // return FAR child as a new strategy (copy of "self" with swapped fields)
        Self {
            stem_idx: far_stem,
            minor_level: far_minor,
            level: next_level,
            dim: next_dim,
            leaf_idx: far_leaf,
            stems_ptr: self.stems_ptr,
        }
    }

    fn get_stem_node_count_from_leaf_node_count(leaf_node_count: usize) -> usize {
        if leaf_node_count < 2 {
            0
        } else {
            leaf_node_count.next_power_of_two() - 1
        }
    }
    // TODO: It would be nice to be able to determine the exact required length up-front.
    //  Instead, we just trim the stems afterwards by traversing right-child non-inf nodes
    //  till we hit max level to get the max used stem
    fn stem_node_padding_factor() -> usize {
        50
    }
    fn trim_unneeded_stems<A: Axis<Coord = A>>(stems: &mut AVec<A>, max_stem_level: usize) {
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        if !stems.is_empty() {
            let mut so = Self::new(stems_ptr);
            loop {
                let val = &stems[so.stem_idx()];
                let is_right_child = A::is_max_value(*val);
                so.traverse::<A, K>(is_right_child);
                if so.level() as usize == max_stem_level {
                    break;
                }
            }

            stems.truncate(so.stem_idx() + 1);
        }
    }

    fn child_indices(&self) -> (usize, usize) {
        unimplemented!("child_indices not yet implemented for DonnellyPf")
    }
}

#[inline(always)]
fn mask32(b: bool) -> u32 {
    0u32.wrapping_sub(b as u32)
} // false->0x00000000, true->0xFFFFFFFF

#[inline(always)]
fn maskusize(b: bool) -> usize {
    0usize.wrapping_sub(b as usize)
}

impl<const L: u32, const CL: u32, const VB: u32, const K: usize> DonnellyPf<L, CL, VB, K> {
    // ---- layout helpers ----
    #[inline(always)]
    const fn items_per_line() -> u32 {
        CL / VB
    }
    #[inline(always)]
    const fn log2_items_per_line() -> u32 {
        Self::items_per_line().ilog2()
    }
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
        debug_assert!(L >= 2 && L <= 8);
        let is_right_child = u32::from(is_right_child);

        // index into current minor triangle / cache line
        let min_idx = curr_idx & Self::line_mask();

        // row in current minor triangle
        let min_row_idx = min_idx.wrapping_sub(minor_level).wrapping_sub(1);

        let base_no_right = (curr_idx & Self::line_mask_inv()).wrapping_add(1);
        let next_prefetch_base = base_no_right
            .wrapping_add(min_row_idx.wrapping_shl(1))
            .wrapping_shl(Self::log2_items_per_line());

        // if minor_level == 0 {
        //     Self::prefetch_next_base(stems_ptr, next_prefetch_base);
        // }

        let base_with_side: u32 = base_no_right.wrapping_add(is_right_child);
        let same_base = base_with_side.wrapping_add(min_idx.wrapping_shl(1));

        // unsafe {
        //     let nxt_ptr = stems_ptr.as_ptr().add((same_base as usize) * VB as usize);
        //     prefetch_t0(nxt_ptr);
        // }

        let next_result_base = next_prefetch_base
            .wrapping_add(is_right_child.wrapping_shl(Self::log2_items_per_line()));

        // boolean flag for cmov indicating if we're transitioning to the next minor triangle
        let inc_major_level = (minor_level.wrapping_add(1) == Self::log2_items_per_line()) as u32;
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
        debug_assert!(L >= 2 && L <= 8);
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
    fn step_pure_tail(
        curr_idx: u32,
        mut minor_level: u32,
        is_right_child: bool,
        stems_ptr: NonNull<u8>,
    ) -> (u32, u32) {
        debug_assert!(L >= 2 && L <= 8);
        let is_right_child = u32::from(is_right_child);

        // index into current minor triangle / cache line
        let min_idx = curr_idx & Self::line_mask();

        // row in current minor triangle
        let min_row_idx = min_idx.wrapping_sub(minor_level).wrapping_sub(1);

        let base_no_right = (curr_idx & Self::line_mask_inv()).wrapping_add(1);
        let next_prefetch_base = base_no_right
            .wrapping_add(min_row_idx.wrapping_shl(1))
            .wrapping_shl(Self::log2_items_per_line());

        let result = next_prefetch_base
            .wrapping_add(is_right_child.wrapping_shl(Self::log2_items_per_line()));

        // Prefetch result? Not much point, it's likely gonna be requested within 1 cycle
        // unsafe {
        //     let nxt_ptr = stems_ptr
        //         .as_ptr()
        //         .add((result * VB) as usize);
        //     prefetch_t0(nxt_ptr);
        // }

        // Prefetch deeper-level 8 base ptrs to L2
        let next_base_no_right = (result & Self::line_mask_inv()).wrapping_add(7);
        let next_next_prefetch_base = next_base_no_right.wrapping_shl(Self::log2_items_per_line());

        Self::prefetch_next_base(stems_ptr, next_next_prefetch_base, 2u32.pow(L) as usize);

        // println!("is_right_child: {is_right_child}, min_idx: {min_idx}, min_row_idx: {min_row_idx}, base_no_right: {base_no_right}, next_prefetch_base: {next_prefetch_base}, next stem_idx: {result}");
        // println!("next_next_prefetch_base: {next_next_prefetch_base} -> {}", next_next_prefetch_base + 128);

        minor_level = 0;

        (result, minor_level)
    }

    #[allow(dead_code)]
    #[inline(always)]
    fn prefetch_next_base(stems_ptr: NonNull<u8>, next_base: u32, cache_line_count: usize) {
        #[cfg(target_arch = "x86_64")]
        const BYTES_PER_LINE: usize = 64;

        #[cfg(target_arch = "aarch64")]
        const BYTES_PER_LINE: usize = 64; // 64 for most ARM, 128 for Apple M-series

        let base_ptr = unsafe { stems_ptr.as_ptr().add((next_base as usize) * VB as usize) };

        for i in 0..cache_line_count {
            let ptr = unsafe { base_ptr.add(i * BYTES_PER_LINE) };
            unsafe { prefetch_t1(ptr) };
        }
    }

    /// Two-children step in one pass (left=false, right=true).
    /// Advances minor_level once; does NOT change curr_idx (so caller can choose a child later).
    #[inline(always)]
    fn both_children_pure(curr_idx: u32, minor_level: u32) -> (u32, u32) {
        // precompute pieces identical to step_pure
        let line_mask = Self::line_mask();
        let line_mask_inv = Self::line_mask_inv();
        let l2_items = Self::log2_items_per_line();

        let min_idx = curr_idx & line_mask;
        let min_row_idx = min_idx.wrapping_sub(minor_level).wrapping_sub(1);

        let inc_major = (minor_level.wrapping_add(1) == l2_items) as u32;
        let inc_mask = 0u32.wrapping_sub(inc_major);

        let base_no_right = (curr_idx & line_mask_inv).wrapping_add(1);

        // same-block left/right
        let same_left = base_no_right.wrapping_add(min_idx.wrapping_shl(1));
        let same_right = same_left.wrapping_add(1);

        // next-block left/right (note: add right after shift by L)
        let next_pre = base_no_right.wrapping_add(min_row_idx.wrapping_shl(1));
        let next_left = next_pre.wrapping_shl(l2_items);
        let next_right = next_left.wrapping_add(1u32.wrapping_shl(l2_items));

        // masked select between same/next for both children
        let left = (same_left & !inc_mask) | (next_left & inc_mask);
        let right = (same_right & !inc_mask) | (next_right & inc_mask);

        (left, right)
    }

    #[allow(dead_code)]
    #[inline(always)]
    fn prefetch_next_minor_tri(&self, stems_ptr: *const f32) {
        // Only act on the first level of each minor triangle
        if self.minor_level == 0 {
            // Each minor tri in Donnelly-3 advances after 3 levels, i.e. 8 lines (512B)
            let curr_line = line_base_f32(self.stem_idx);
            let next_line = curr_line + 16 * 8; // 8 lines * 16 f32/line

            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            unsafe {
                prefetch_8_lines_f32(stems_ptr, next_line);
            }
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
    DonnellyPf::<3, 64, 8, 3>::step_pure(curr_idx, minor_index, is_right_child, stems_ptr)
}

/// Exposed pure function for use with cargo-asm
#[inline(never)]
pub fn both_children_pure_hook(curr_idx: u32, minor_index: u32) -> (u32, u32) {
    DonnellyPf::<3, 64, 8, 3>::both_children_pure(curr_idx, minor_index)
}

/// Exposed pure function for use with cargo-asm
#[inline(never)]
pub fn test_traverse_hook(is_right_child: bool, stems: *mut u8) -> usize {
    let stems_ptr = NonNull::new(stems).unwrap();

    let mut stem_strat = DonnellyPf::<3, 64, 8, 3>::new(stems_ptr);

    stem_strat.traverse::<f64, 3>(is_right_child);
    stem_strat.traverse::<f64, 3>(!is_right_child);
    stem_strat.traverse::<f64, 3>(is_right_child);

    stem_strat.stem_idx()
}

// helper: line base (16 f32 per 64B line)
#[inline(always)]
fn line_base_f32(idx: u32) -> u32 {
    idx & !15
}

// prefetch an 8-line run starting at base_line (in f32 indices)
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn prefetch_8_lines_f32(stems_ptr: *const f32, base_line: u32) {
    let p0 = stems_ptr.add(base_line as usize) as *const u8;

    prefetch_t0(p0);

    // Optionally prefetch additional lines:
    // let p1 = stems_ptr.add((base_line + 16) as usize) as *const u8;
    // prefetch_t0(p1);
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn prefetch_8_lines_f32(stems_ptr: *const f32, base_line: u32) {
    let p0 = stems_ptr.add(base_line as usize) as *const u8;

    // Usually one is enough; try uncommenting p1 if needed.
    prefetch_t0(p0);

    // let p1 = stems_ptr.add((base_line + 16) as usize) as *const u8;
    // prefetch_t0(p1);

    // If A isn’t enough, at L1 also prefetch the tail:
    // let plast = stems_ptr.add((base_line + 16*7) as usize) as *const u8;
    // prefetch_t1(plast);
}

#[cfg(test)]
mod tests {
    use super::*;
    use aligned_vec::avec;
    use rstest::rstest;

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
    fn donnelly_v2_get_child_idx_produces_correct_values(
        #[case] input: Vec<bool>,
        #[case] expected: usize,
    ) {
        let stems = avec![f64::INFINITY; 9];
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();

        let mut stem_strat = DonnellyPf::<3, 64, 8, 3>::new(stems_ptr);
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
    fn donnelly_v2_get_both_child_idxs_produces_correct_values(
        #[case] input: Vec<bool>,
        #[case] expected: (usize, usize),
    ) {
        let stems = avec![f64::INFINITY; 9];
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();

        let mut stem_strat = DonnellyPf::<3, 64, 8, 3>::new(stems_ptr);
        // let mut stem_strat = Donnelly::<3, 64, 4, 4>::new();

        // let last = input.last().unwrap();
        input.iter().for_each(|selection| {
            stem_strat.branch_relative::<3>(*selection);
        });

        let results = stem_strat.split::<3>();
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
    fn donnelly_v2_get_child_idx_unrolled_produces_correct_values(
        #[case] input: Vec<bool>,
        #[case] expected: usize,
    ) {
        let stems = avec![f64::INFINITY; 9];
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();

        let mut stem_strat = DonnellyPf::<3, 64, 8, 3>::new(stems_ptr);
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
}

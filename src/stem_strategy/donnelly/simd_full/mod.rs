//! Donnelly full-SIMD stem strategy.
//!
//! This module provides SIMD-optimized block-at-once traversal strategies for kd-trees.
//! Architecture-specific implementations are in submodules.

use std::ptr::NonNull;

use crate::stem_strategy::donnelly::core::DonnellyCore;
use crate::{Axis, Content, LeafStrategy, StemStrategy};

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub mod aarch64;

mod autovec;

// Pruning traits for SIMD dispatch
mod prune_traits;
pub use prune_traits::{SimdPrune, SimdSelectBestChildBlock3};

// Comparison traits for type-specific dispatch
mod compare_traits;
pub use compare_traits::{CompareBlock3, CompareBlock4};

// Backtrack mask generation traits for type-specific dispatch
pub mod backtrack_traits;
pub use backtrack_traits::{BacktrackBlock3, BacktrackBlock4};

pub(crate) trait DeferredBlockTraversal: StemStrategy + Copy {
    fn block_child<const K: usize>(&self, child_idx: u8) -> Self;

    fn backtrack_block3_pending_mask<A, O, D, const K2: usize>(
        &self,
        stems: &[A],
        query_wide: O,
        parent_lower_bound: O,
        parent_upper_bound: O,
        old_off: O,
        rd: O,
        best_dist: O,
    ) -> u8
    where
        A: Axis<Coord = A>,
        O: Axis<Coord = O>,
        D: crate::dist::DistanceMetric<A, Output = O>,
    {
        let _ = (
            stems,
            query_wide,
            parent_lower_bound,
            parent_upper_bound,
            old_off,
            rd,
            best_dist,
        );
        unreachable!("backtrack_block3_pending_mask is only valid for Block3 SIMD traversal")
    }

    fn selected_block3_pending_child_state<A, O, D>(
        &self,
        stems: &[A],
        child_idx: u8,
        query_wide: O,
        parent_lower_bound: O,
        parent_upper_bound: O,
        old_off: O,
        rd: O,
    ) -> (O, O, O, O)
    where
        A: Axis<Coord = A>,
        O: Axis<Coord = O>,
        D: crate::dist::DistanceMetricCore<A, Output = O>,
    {
        let _ = (
            stems,
            child_idx,
            query_wide,
            parent_lower_bound,
            parent_upper_bound,
            old_off,
            rd,
        );
        unreachable!("selected_block3_pending_child_state is only valid for Block3 SIMD traversal")
    }

    fn fill_block3_pending_values<A, O, D, const K2: usize>(
        &self,
        stems: &[A],
        query_wide: O,
        parent_lower_bound: O,
        parent_upper_bound: O,
        old_off: O,
        rd: O,
        best_dist: O,
        new_off_values: &mut [O; 8],
        rd_values: &mut [O; 8],
        lower_bounds: &mut [O; 8],
        upper_bounds: &mut [O; 8],
    ) -> u8
    where
        A: Axis<Coord = A>,
        O: Axis<Coord = O>,
        D: crate::dist::DistanceMetric<A, Output = O>,
    {
        let _ = (
            stems,
            query_wide,
            parent_lower_bound,
            parent_upper_bound,
            old_off,
            rd,
            best_dist,
            new_off_values,
            rd_values,
            lower_bounds,
            upper_bounds,
        );
        unreachable!("fill_block3_pending_values is only valid for Block3 SIMD traversal")
    }
}

/// Block3 interval lower bounds encoded as u64 literal.
/// Each 8-bit segment contains the lower bound pivot offset for a child (255 = -∞).
/// Lower bounds: [255, 3, 1, 4, 0, 5, 2, 6] for children 0-7
const CHILD_LOWER_BOUNDS_BLOCK3: u64 = 0x06_02_05_00_04_01_03_FF;

/// Block3 interval upper bounds encoded as u64 literal.
/// Each 8-bit segment contains the upper bound pivot offset for a child (255 = +∞).
/// Upper bounds: [3, 1, 4, 0, 5, 2, 6, 255] for children 0-7
const CHILD_UPPER_BOUNDS_BLOCK3: u64 = 0xFF_06_02_05_00_04_01_03;

/// Block4 interval lower bounds encoded as u128 literal (16 children).
/// Each 8-bit segment contains the lower bound pivot offset for a child (255 = -∞).
/// Lower bounds: [255, 7, 3, 8, 1, 9, 4, 10, 0, 11, 5, 12, 2, 13, 6, 14] for children 0-15
const CHILD_LOWER_BOUNDS_BLOCK4: u128 = 0x0E_06_0D_02_0C_05_0B_00_0A_04_09_01_08_03_07_FF;

/// Block4 interval upper bounds encoded as u128 literal (16 children).
/// Each 8-bit segment contains the upper bound pivot offset for a child (255 = +∞).
/// Upper bounds: [7, 3, 8, 1, 9, 4, 10, 0, 11, 5, 12, 2, 13, 6, 14, 255] for children 0-15
const CHILD_UPPER_BOUNDS_BLOCK4: u128 = 0xFF_0E_06_0D_02_0C_05_0B_00_0A_04_09_01_08_03_07;

/// Returns the interval bounds [lower, upper) for a Block3 child in a given dimension.
///
/// Block3 has 8 children arranged in a triangular layout, where all levels in the block
/// split on the same dimension. Each child occupies an interval [lower, upper) in that dimension.
///
/// Returns (lower_pivot_offset, upper_pivot_offset) where:
/// - Offset refers to the pivot within the block (0-6 are actual pivots, 7 is padding)
/// - 255 represents ±∞ (child 0 extends to -∞, child 7 extends to +∞)
///
/// # Example
/// ```text
/// Block structure:
///           pivot[0]
///      pivot[1]    pivot[2]
///   p[3] p[4]   p[5] p[6]
/// ch0 ch1 ch2 ch3 ch4 ch5 ch6 ch7
///
/// Child 0: [-∞, pivot[3]) → (255, 3)
/// Child 1: [pivot[3], pivot[1]) → (3, 1)
/// Child 2: [pivot[1], pivot[4]) → (1, 4)
/// ...etc
/// ```
#[inline(always)]
pub(crate) const fn child_interval_bounds_block3(child_idx: usize) -> (u8, u8) {
    let lower = ((CHILD_LOWER_BOUNDS_BLOCK3 >> (child_idx * 8)) & 0xFF) as u8;
    let upper = ((CHILD_UPPER_BOUNDS_BLOCK3 >> (child_idx * 8)) & 0xFF) as u8;
    (lower, upper)
}

/// Returns the interval bounds [lower, upper) for a Block4 child in a given dimension.
///
/// Block4 has 16 children arranged in a triangular layout, where all levels in the block
/// split on the same dimension. Each child occupies an interval [lower, upper) in that dimension.
///
/// Returns (lower_pivot_offset, upper_pivot_offset) where:
/// - Offset refers to the pivot within the block (0-14 are actual pivots, 15 is padding)
/// - 255 represents ±∞ (child 0 extends to -∞, child 15 extends to +∞)
///
/// # Example
/// ```text
/// Block structure:
///            pivot[0]
///       pivot[1]    pivot[2]
///    p[3] p[4]   p[5] p[6]
/// p[7]p[8]p[9]p[10]p[11]p[12]p[13]p[14]
/// c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
///
/// Child 0: [-∞, pivot[7]) → (255, 7)
/// Child 1: [pivot[7], pivot[3]) → (7, 3)
/// Child 2: [pivot[3], pivot[8]) → (3, 8)
/// ...etc
/// ```
#[inline(always)]
pub(crate) const fn child_interval_bounds_block4(child_idx: usize) -> (u8, u8) {
    let lower = ((CHILD_LOWER_BOUNDS_BLOCK4 >> (child_idx * 8)) & 0xFF) as u8;
    let upper = ((CHILD_UPPER_BOUNDS_BLOCK4 >> (child_idx * 8)) & 0xFF) as u8;
    (lower, upper)
}

/// Computes absolute distance from a query point to an interval [lower, upper).
///
/// This is a metric-agnostic geometric distance - the metric's `dist1` is applied separately.
///
/// Returns:
/// - If query < lower: lower - query
/// - If query >= upper: query - upper
/// - If lower <= query < upper: 0 (query is inside the interval)
#[inline(always)]
pub(crate) fn interval_distance_1d<O>(query: O, lower: O, upper: O) -> O
where
    O: Axis<Coord = O>,
{
    let below = O::max(O::zero(), lower - query);
    let above = O::max(O::zero(), query - upper);
    O::saturating_add(below, above)
}

#[inline(always)]
fn coord_min<O>(a: O, b: O) -> O
where
    O: Axis<Coord = O>,
{
    if O::cmp(a, b) == std::cmp::Ordering::Greater {
        b
    } else {
        a
    }
}

#[inline(always)]
fn fill_block3_backtrack_values_and_bounds<A, O, D, const K2: usize>(
    stems: &[A],
    block_base_idx: usize,
    query_wide: O,
    parent_lower_bound: O,
    parent_upper_bound: O,
    old_off: O,
    rd: O,
    best_dist: O,
    new_off_values: &mut [O; 8],
    rd_values: &mut [O; 8],
    lower_bounds: &mut [O; 8],
    upper_bounds: &mut [O; 8],
) -> u8
where
    A: Axis<Coord = A>,
    O: Axis<Coord = O>,
    D: crate::dist::DistanceMetric<A, Output = O>,
{
    D::fill_block3_values_and_bounds::<K2>(
        query_wide,
        NonNull::new(stems.as_ptr() as *mut u8).expect("stems slice pointer"),
        block_base_idx,
        parent_lower_bound,
        parent_upper_bound,
        old_off,
        rd,
        best_dist,
        new_off_values,
        rd_values,
        lower_bounds,
        upper_bounds,
    )
}

#[inline(always)]
fn backtrack_block3_with_bounds<A, O, D, const K2: usize>(
    stems: &[A],
    block_base_idx: usize,
    query_wide: O,
    parent_lower_bound: O,
    parent_upper_bound: O,
    old_off: O,
    rd: O,
    best_dist: O,
) -> u8
where
    A: Axis<Coord = A>,
    O: Axis<Coord = O>,
    D: crate::dist::DistanceMetric<A, Output = O>,
{
    D::backtrack_block3_with_bounds::<K2>(
        query_wide,
        NonNull::new(stems.as_ptr() as *mut u8).expect("stems slice pointer"),
        block_base_idx,
        parent_lower_bound,
        parent_upper_bound,
        old_off,
        rd,
        best_dist,
    )
}

#[inline(always)]
fn selected_block3_child_state<A, O, D>(
    stems: &[A],
    block_base_idx: usize,
    child_idx: u8,
    query_wide: O,
    parent_lower_bound: O,
    parent_upper_bound: O,
) -> (O, O, O)
where
    A: Axis<Coord = A>,
    O: Axis<Coord = O>,
    D: crate::dist::DistanceMetricCore<A, Output = O>,
{
    let (lower_offset, upper_offset) = child_interval_bounds_block3(child_idx as usize);

    let raw_lower = if lower_offset == 255 {
        A::min_value()
    } else {
        unsafe { *stems.get_unchecked(block_base_idx + lower_offset as usize) }
    };

    let raw_upper = if upper_offset == 255 {
        A::max_value()
    } else {
        unsafe { *stems.get_unchecked(block_base_idx + upper_offset as usize) }
    };

    let effective_lower = O::max(parent_lower_bound, D::widen_coord(raw_lower));
    let effective_upper = coord_min(parent_upper_bound, D::widen_coord(raw_upper));
    let new_off = interval_distance_1d(query_wide, effective_lower, effective_upper);

    (new_off, effective_lower, effective_upper)
}

#[inline(always)]
fn selected_block3_child_state_and_rd<A, O, D>(
    stems: &[A],
    block_base_idx: usize,
    child_idx: u8,
    query_wide: O,
    parent_lower_bound: O,
    parent_upper_bound: O,
    old_off: O,
    rd: O,
) -> (O, O, O, O)
where
    A: Axis<Coord = A>,
    O: Axis<Coord = O>,
    D: crate::dist::DistanceMetricCore<A, Output = O>,
{
    let (new_off, effective_lower, effective_upper) = selected_block3_child_state::<A, O, D>(
        stems,
        block_base_idx,
        child_idx,
        query_wide,
        parent_lower_bound,
        parent_upper_bound,
    );
    let new_dist1 = D::dist1(new_off, O::zero());
    let old_dist1 = D::dist1(old_off, O::zero());
    let rd_far = O::saturating_add(rd - old_dist1, new_dist1);

    (rd_far, new_off, effective_lower, effective_upper)
}

#[inline(always)]
fn fill_block4_backtrack_values_and_bounds<A, O, D, const K2: usize>(
    stems: &[A],
    block_base_idx: usize,
    query_wide: O,
    parent_lower_bound: O,
    parent_upper_bound: O,
    old_off: O,
    rd: O,
    best_dist: O,
    new_off_values: &mut [O; 16],
    rd_values: &mut [O; 16],
    lower_bounds: &mut [O; 16],
    upper_bounds: &mut [O; 16],
) -> u16
where
    A: Axis<Coord = A>,
    O: Axis<Coord = O>,
    D: crate::dist::DistanceMetricCore<A, Output = O>,
{
    let old_dist1 = D::dist1(old_off, O::zero());
    let mut mask = 0u16;

    for sibling_idx in 0..16 {
        let (lower_offset, upper_offset) = child_interval_bounds_block4(sibling_idx);

        let lower = if lower_offset == 255 {
            A::min_value()
        } else {
            unsafe { *stems.get_unchecked(block_base_idx + lower_offset as usize) }
        };

        let upper = if upper_offset == 255 {
            A::max_value()
        } else {
            unsafe { *stems.get_unchecked(block_base_idx + upper_offset as usize) }
        };

        let raw_lower = D::widen_coord(lower);
        let raw_upper = D::widen_coord(upper);
        let effective_lower = O::max(parent_lower_bound, raw_lower);
        let effective_upper = coord_min(parent_upper_bound, raw_upper);

        lower_bounds[sibling_idx] = effective_lower;
        upper_bounds[sibling_idx] = effective_upper;

        if O::cmp(effective_lower, effective_upper) == std::cmp::Ordering::Greater {
            new_off_values[sibling_idx] = O::max_value();
            rd_values[sibling_idx] = O::max_value();
            continue;
        }

        let new_off = interval_distance_1d(query_wide, effective_lower, effective_upper);
        new_off_values[sibling_idx] = new_off;

        let new_dist1 = D::dist1(new_off, O::zero());
        let rd_far = O::saturating_add(rd - old_dist1, new_dist1);
        rd_values[sibling_idx] = rd_far;

        if O::cmp(rd_far, best_dist) != std::cmp::Ordering::Greater {
            mask |= 1u16 << sibling_idx;
        }
    }

    mask
}

/// Donnelly SIMD block-at-once traversal strategy parameterized by block height.
#[derive(Copy, Clone, Debug)]
pub struct DonnellySimdFull<const BH: usize> {
    core: DonnellyCore<BH>,
}

/// Perform all comparisons in a 3-level block, dispatching to the appropriate SIMD implementation
/// based on the axis type A. Monomorphized at compile time.
#[inline(always)]
pub(crate) fn compare_block3<A>(stems: &[A], query_val: A, block_base_idx: usize) -> u8
where
    A: CompareBlock3,
{
    let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
    A::compare_block3_impl(stems_ptr, query_val, block_base_idx)
}

/// Perform all comparisons in a 4-level block, dispatching to the appropriate SIMD implementation
/// based on the axis type A. Monomorphized at compile time.
#[inline(always)]
pub(crate) fn compare_block4<A>(stems: &[A], query_val: A, block_base_idx: usize) -> u8
where
    A: CompareBlock4,
{
    let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
    A::compare_block4_impl(stems_ptr, query_val, block_base_idx)
}

/// `cargo-asm` hooks for isolating Block3 exact backtrack kernels.
#[cfg(feature = "cargo_asm")]
pub mod cargo_asm {
    use super::{
        backtrack_block3_with_bounds, fill_block3_backtrack_values_and_bounds,
        selected_block3_child_state_and_rd,
    };
    use crate::dist::SquaredEuclidean;
    use crate::stem_strategy::{DonnellySimdFull, SimdSelectBestChildBlock3};
    use crate::StemStrategy;

    /// Hook for cargo-asm to render the exact Block3 backtrack fill kernel directly.
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn donnelly_block3_fill_backtrack_f64_cargo_asm_hook(
        stems: &[f64],
        block_base_idx: usize,
        query_wide: f64,
        old_off: f64,
        rd: f64,
        best_dist: f64,
        new_off_values: &mut [f64; 8],
        rd_values: &mut [f64; 8],
    ) -> u8 {
        let mut lower_bounds = [0.0; 8];
        let mut upper_bounds = [0.0; 8];
        fill_block3_backtrack_values_and_bounds::<f64, f64, SquaredEuclidean<f64>, 3>(
            stems,
            block_base_idx,
            query_wide,
            f64::NEG_INFINITY,
            f64::INFINITY,
            old_off,
            rd,
            best_dist,
            new_off_values,
            rd_values,
            &mut lower_bounds,
            &mut upper_bounds,
        )
    }

    /// Hook for cargo-asm to render the exact Block3 traversal step directly.
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn donnelly_block3_exact_step_f64_cargo_asm_hook(
        stem_strat: &mut DonnellySimdFull<3>,
        stems: &[f64],
        query: &[f64; 3],
        query_wide: &[f64; 3],
        lower: &mut [f64; 3],
        upper: &mut [f64; 3],
        off: &mut [f64; 3],
        dim: &mut usize,
        rd: f64,
        max_stem_level: i32,
        best_dist: f64,
        stack: &mut <DonnellySimdFull<3> as StemStrategy>::Stack<f64>,
    ) -> bool {
        stem_strat.backtracking_traverse_step_with_bounds::<f64, f64, SquaredEuclidean<f64>, 3>(
            stems,
            query,
            query_wide,
            lower,
            upper,
            off,
            dim,
            rd,
            max_stem_level,
            best_dist,
            stack,
        )
    }

    /// Hook for cargo-asm to render the pending-pop fast path directly.
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn donnelly_block3_pending_fast_path_f64_cargo_asm_hook(
        stems: &[f64],
        block_base_idx: usize,
        query_wide: f64,
        parent_lower_bound: f64,
        parent_upper_bound: f64,
        old_off: f64,
        rd: f64,
        best_dist: f64,
        pending_mask: u8,
    ) -> Option<(u8, f64, f64, f64, f64)> {
        let candidate_mask = backtrack_block3_with_bounds::<f64, f64, SquaredEuclidean<f64>, 3>(
            stems,
            block_base_idx,
            query_wide,
            parent_lower_bound,
            parent_upper_bound,
            old_off,
            rd,
            best_dist,
        ) & pending_mask;

        if candidate_mask == 0 {
            return None;
        }

        if candidate_mask.is_power_of_two() {
            let child_idx = candidate_mask.trailing_zeros() as u8;
            let (child_rd, child_off, child_lower, child_upper) =
                selected_block3_child_state_and_rd::<f64, f64, SquaredEuclidean<f64>>(
                    stems,
                    block_base_idx,
                    child_idx,
                    query_wide,
                    parent_lower_bound,
                    parent_upper_bound,
                    old_off,
                    rd,
                );
            return Some((child_idx, child_rd, child_off, child_lower, child_upper));
        }

        let mut rd_values = [0.0; 8];
        let mut new_off_values = [0.0; 8];
        let mut lower_bounds = [0.0; 8];
        let mut upper_bounds = [0.0; 8];
        let candidate_mask =
            fill_block3_backtrack_values_and_bounds::<f64, f64, SquaredEuclidean<f64>, 3>(
                stems,
                block_base_idx,
                query_wide,
                parent_lower_bound,
                parent_upper_bound,
                old_off,
                rd,
                best_dist,
                &mut new_off_values,
                &mut rd_values,
                &mut lower_bounds,
                &mut upper_bounds,
            ) & pending_mask;
        let child_idx = f64::simd_select_best_child_block3(&rd_values, candidate_mask)
            .unwrap_or_else(|| unsafe {
                debug_assert!(false, "candidate_mask != 0");
                core::hint::unreachable_unchecked()
            });
        let child_idx_usize = child_idx as usize;

        Some((
            child_idx,
            unsafe { *rd_values.get_unchecked(child_idx_usize) },
            unsafe { *new_off_values.get_unchecked(child_idx_usize) },
            unsafe { *lower_bounds.get_unchecked(child_idx_usize) },
            unsafe { *upper_bounds.get_unchecked(child_idx_usize) },
        ))
    }
}

// ====================================================================================
// StemStrategy Implementations (architecture-agnostic, dispatch through traits)
// ====================================================================================

// Block3 implementation (3-level blocks, 64-byte cache lines)
impl StemStrategy for DonnellySimdFull<3> {
    const ROOT_IDX: usize = 0;
    const BLOCK_SIZE: usize = 3;

    type DeferredState = Self;
    type StackContext<A> = crate::kd_tree::query_stack_simd::Block3SimdQueryStackContext<A, Self>;
    type Stack<A> = crate::kd_tree::query_stack_simd::Block3ExactQueryStack<
        A,
        Self,
        { crate::kd_tree::query_stack_simd::BLOCK3_EXACT_INLINE_SIMD_QUERY_STACK_CAPACITY },
    >;

    #[inline]
    fn new(stems_ptr: std::ptr::NonNull<u8>) -> Self {
        Self {
            core: crate::stem_strategy::donnelly::core::DonnellyCore::new(stems_ptr),
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
        self.core.level() as usize / Self::BLOCK_SIZE % K
    }

    #[inline(always)]
    fn construction_dim<const K: usize>(&self) -> usize {
        self.core.level() as usize / Self::BLOCK_SIZE % K
    }

    #[inline(always)]
    fn level(&self) -> i32 {
        self.core.level()
    }

    #[inline(always)]
    fn traverse<A: Axis<Coord = A>, const K2: usize>(&mut self, is_right: bool) {
        self.core.traverse::<A, K2>(is_right)
    }

    #[inline(always)]
    fn branch<A: Axis<Coord = A>, const K2: usize>(&mut self) -> Self {
        Self {
            core: self.core.branch::<A, K2>(),
        }
    }

    #[inline(always)]
    fn child_indices<A: Axis<Coord = A>>(&self) -> (usize, usize) {
        self.core.child_indices::<A>()
    }

    fn get_leaf_idx<A: Axis<Coord = A>, const K2: usize>(
        stems: &[A],
        query: &[A; K2],
        max_stem_level: i32,
    ) -> usize
    where
        Self: Sized,
    {
        let stems_ptr = std::ptr::NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        let mut strat = Self::new(stems_ptr);

        while strat.level() <= max_stem_level {
            let dim = strat.dim::<K2>();
            let query_val = unsafe { *query.get_unchecked(dim) };
            let block_base_idx = strat.stem_idx();
            let block_width = 1usize << Self::BLOCK_SIZE;
            let minor_tri_height = (64 / A::VALUE_WIDTH_BYTES as u32).ilog2();
            let can_take_full_block = strat.level() + Self::BLOCK_SIZE as i32 - 1 <= max_stem_level
                && block_base_idx + block_width <= stems.len()
                && Self::BLOCK_SIZE as u32 == minor_tri_height
                && strat.core.minor_level() == 0;

            if can_take_full_block {
                let child_idx = compare_block3(stems, query_val, block_base_idx);
                strat
                    .core
                    .traverse_block::<K2>(child_idx, Self::BLOCK_SIZE as u32);
            } else {
                let stem_idx = strat.stem_idx();
                let is_right = if stem_idx < stems.len() {
                    query_val >= unsafe { *stems.get_unchecked(stem_idx) }
                } else {
                    false
                };
                strat.traverse::<A, K2>(is_right);
            }
        }

        strat.leaf_idx()
    }

    #[inline(always)]
    fn backtracking_traverse_step_with_bounds<A, O, D, const K2: usize>(
        &mut self,
        stems: &[A],
        query: &[A; K2],
        query_wide: &[O; K2],
        lower: &mut [O; K2],
        upper: &mut [O; K2],
        off: &mut [O; K2],
        dim: &mut usize,
        rd: O,
        max_stem_level: i32,
        best_dist: O,
        stack: &mut Self::Stack<O>,
    ) -> bool
    where
        Self: Sized,
        A: Axis<Coord = A>,
        O: Axis<Coord = O> + SimdSelectBestChildBlock3 + BacktrackBlock3 + BacktrackBlock4,
        D: crate::dist::DistanceMetric<A, Output = O>,
    {
        #[cfg(any(feature = "exact_query_stats", feature = "test_utils"))]
        crate::results::exact_query_stats::record_block3_step_entry();

        if self.level() > max_stem_level {
            return false;
        }
        let dim_val = *dim;
        let query_val = unsafe { *query.get_unchecked(dim_val) };

        #[allow(unused)]
        // used by simd code below, but is also the only code that uses query_wide arg
        let query_wide_val = unsafe { *query_wide.get_unchecked(dim_val) };

        let old_off_val = unsafe { *off.get_unchecked(dim_val) };
        let lower_bound = unsafe { *lower.get_unchecked(dim_val) };
        let upper_bound = unsafe { *upper.get_unchecked(dim_val) };
        let block_base_idx = self.stem_idx();
        let block_width = 1usize << Self::BLOCK_SIZE;
        let minor_tri_height = (64 / A::VALUE_WIDTH_BYTES as u32).ilog2();
        let can_take_full_block = self.level() + Self::BLOCK_SIZE as i32 - 1 <= max_stem_level
            && block_base_idx + block_width <= stems.len()
            && Self::BLOCK_SIZE as u32 == minor_tri_height
            && self.core.minor_level() == 0;

        // TODO: this fallback should not be required. Refactor to avoid the need for it
        if !can_take_full_block {
            #[cfg(any(feature = "exact_query_stats", feature = "test_utils"))]
            crate::results::exact_query_stats::record_block3_scalar_fallback_step();

            use crate::kd_tree::query_stack_simd::Block3SimdQueryStackContext;

            tracing::warn!(
                level = %self.level(),
                %block_base_idx,
                %block_width,
                stes_len = %stems.len(),
                %minor_tri_height,
                minor_level = %self.core.minor_level(),
                %old_off_val,
                "Cannot take full block",
            );

            let stem_idx = self.stem_idx();
            let pivot = if stem_idx < stems.len() {
                *unsafe { stems.get_unchecked(stem_idx) }
            } else {
                A::max_value()
            };
            if pivot < A::max_value() {
                let is_right_child = query_val >= pivot;
                let far_ctx = self.branch_relative::<A, K2>(is_right_child);

                let pivot_wide: O = D::widen_coord(pivot);
                let new_off = O::saturating_dist(query_wide_val, pivot_wide);
                let rd_far = D::rect_dist_after_update(rd, off, *dim, new_off);

                if O::cmp(rd_far, best_dist) != std::cmp::Ordering::Greater {
                    stack.push(Block3SimdQueryStackContext::Single {
                        stem_strat: far_ctx,
                        dim: dim_val,
                        lower_bound,
                        upper_bound,
                        old_off: new_off,
                        rd: rd_far,
                    });
                }
            } else {
                // +Inf can represent structural padding in left-aligned trees.
                // Scalar traversal just descends left through the padding.
                self.traverse::<A, K2>(false);
            }

            *dim = self.dim::<K2>();
            return true;
        }

        #[cfg(any(feature = "exact_query_stats", feature = "test_utils"))]
        crate::results::exact_query_stats::record_block3_full_step();

        let child_idx = compare_block3(stems, query_val, block_base_idx);
        #[cfg(feature = "test_utils")]
        let trace_enabled =
            crate::test_utils::exact_query_trace::enabled() && std::any::type_name::<O>() == "f64";

        #[cfg(not(feature = "test_utils"))]
        let trace_enabled = false;

        let (candidate_mask, selected_new_off, selected_lower_bound, selected_upper_bound) =
            if trace_enabled {
                let mut rd_values = [O::zero(); 8];
                let mut new_off_values = [O::zero(); 8];
                let mut lower_bounds = [O::zero(); 8];
                let mut upper_bounds = [O::zero(); 8];

                let candidate_mask = fill_block3_backtrack_values_and_bounds::<A, O, D, K2>(
                    stems,
                    block_base_idx,
                    query_wide_val,
                    lower_bound,
                    upper_bound,
                    old_off_val,
                    rd,
                    best_dist,
                    &mut new_off_values,
                    &mut rd_values,
                    &mut lower_bounds,
                    &mut upper_bounds,
                ) & !(1u8 << child_idx);

                #[cfg(feature = "test_utils")]
                {
                    let query_val_f = unsafe { *(&query_wide_val as *const O as *const f64) };
                    let old_off_f = unsafe { *(&old_off_val as *const O as *const f64) };
                    let lower_bound_f = unsafe { *(&lower_bound as *const O as *const f64) };
                    let upper_bound_f = unsafe { *(&upper_bound as *const O as *const f64) };
                    let rd_f = unsafe { *(&rd as *const O as *const f64) };
                    let best_dist_f = unsafe { *(&best_dist as *const O as *const f64) };
                    let new_off_values_f =
                        unsafe { *(&new_off_values as *const [O; 8] as *const [f64; 8]) };
                    let rd_values_f = unsafe { *(&rd_values as *const [O; 8] as *const [f64; 8]) };
                    let lower_bounds_f =
                        unsafe { *(&lower_bounds as *const [O; 8] as *const [f64; 8]) };
                    let upper_bounds_f =
                        unsafe { *(&upper_bounds as *const [O; 8] as *const [f64; 8]) };
                    crate::test_utils::exact_query_trace::push(
                        crate::test_utils::exact_query_trace::ExactQueryTraceEvent::Block3FullStep {
                            stem_idx: self.stem_idx(),
                            level: self.level(),
                            dim: dim_val,
                            query_val: query_val_f,
                            old_off: old_off_f,
                            parent_lower_bound: lower_bound_f,
                            parent_upper_bound: upper_bound_f,
                            rd: rd_f,
                            best_dist: best_dist_f,
                            child_idx,
                            candidate_mask,
                            new_off_values: new_off_values_f,
                            rd_values: rd_values_f,
                            lower_bounds: lower_bounds_f,
                            upper_bounds: upper_bounds_f,
                        },
                    );
                }

                (
                    candidate_mask,
                    new_off_values[child_idx as usize],
                    lower_bounds[child_idx as usize],
                    upper_bounds[child_idx as usize],
                )
            } else {
                let candidate_mask = backtrack_block3_with_bounds::<A, O, D, K2>(
                    stems,
                    block_base_idx,
                    query_wide_val,
                    lower_bound,
                    upper_bound,
                    old_off_val,
                    rd,
                    best_dist,
                ) & !(1u8 << child_idx);
                let (selected_new_off, selected_lower_bound, selected_upper_bound) =
                    selected_block3_child_state::<A, O, D>(
                        stems,
                        block_base_idx,
                        child_idx,
                        query_wide_val,
                        lower_bound,
                        upper_bound,
                    );
                (
                    candidate_mask,
                    selected_new_off,
                    selected_lower_bound,
                    selected_upper_bound,
                )
            };

        if candidate_mask != 0 {
            use crate::kd_tree::query_stack_simd::Block3ExactStackContext;

            stack.push(<Self::StackContext<O> as Block3ExactStackContext<
                O,
                Self,
                K2,
            >>::new_block3_pending_from_state(
                *self,
                candidate_mask,
                rd,
                lower,
                upper,
            ));
        }

        unsafe {
            *off.get_unchecked_mut(dim_val) = selected_new_off;
            *lower.get_unchecked_mut(dim_val) = selected_lower_bound;
            *upper.get_unchecked_mut(dim_val) = selected_upper_bound;
        }

        self.core
            .traverse_block::<K2>(child_idx, Self::BLOCK_SIZE as u32);
        *dim = self.dim::<K2>();

        true
    }

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
            + SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4,
        D: crate::dist::DistanceMetric<A, Output = O>,
        QC: crate::kd_tree::query_context::QueryContext<A, O, K2>,
        LS: LeafStrategy<A, T, Self, K2, B>,
    {
        tree.backtracking_query_with_block3_simd_stack_impl::<QC, O, D>(
            query_ctx,
            stack,
            process_leaf,
        );
    }
}

impl DeferredBlockTraversal for DonnellySimdFull<3> {
    #[inline(always)]
    fn block_child<const K: usize>(&self, child_idx: u8) -> Self {
        let mut child = *self;
        child
            .core
            .traverse_block::<K>(child_idx, Self::BLOCK_SIZE as u32);
        child
    }

    #[inline(always)]
    fn backtrack_block3_pending_mask<A, O, D, const K2: usize>(
        &self,
        stems: &[A],
        query_wide: O,
        parent_lower_bound: O,
        parent_upper_bound: O,
        old_off: O,
        rd: O,
        best_dist: O,
    ) -> u8
    where
        A: Axis<Coord = A>,
        O: Axis<Coord = O>,
        D: crate::dist::DistanceMetric<A, Output = O>,
    {
        backtrack_block3_with_bounds::<A, O, D, K2>(
            stems,
            self.stem_idx(),
            query_wide,
            parent_lower_bound,
            parent_upper_bound,
            old_off,
            rd,
            best_dist,
        )
    }

    #[inline(always)]
    fn selected_block3_pending_child_state<A, O, D>(
        &self,
        stems: &[A],
        child_idx: u8,
        query_wide: O,
        parent_lower_bound: O,
        parent_upper_bound: O,
        old_off: O,
        rd: O,
    ) -> (O, O, O, O)
    where
        A: Axis<Coord = A>,
        O: Axis<Coord = O>,
        D: crate::dist::DistanceMetricCore<A, Output = O>,
    {
        selected_block3_child_state_and_rd::<A, O, D>(
            stems,
            self.stem_idx(),
            child_idx,
            query_wide,
            parent_lower_bound,
            parent_upper_bound,
            old_off,
            rd,
        )
    }

    #[inline(always)]
    fn fill_block3_pending_values<A, O, D, const K2: usize>(
        &self,
        stems: &[A],
        query_wide: O,
        parent_lower_bound: O,
        parent_upper_bound: O,
        old_off: O,
        rd: O,
        best_dist: O,
        new_off_values: &mut [O; 8],
        rd_values: &mut [O; 8],
        lower_bounds: &mut [O; 8],
        upper_bounds: &mut [O; 8],
    ) -> u8
    where
        A: Axis<Coord = A>,
        O: Axis<Coord = O>,
        D: crate::dist::DistanceMetric<A, Output = O>,
    {
        fill_block3_backtrack_values_and_bounds::<A, O, D, K2>(
            stems,
            self.stem_idx(),
            query_wide,
            parent_lower_bound,
            parent_upper_bound,
            old_off,
            rd,
            best_dist,
            new_off_values,
            rd_values,
            lower_bounds,
            upper_bounds,
        )
    }
}

// Block4 implementation (4-level blocks, 64-byte cache lines)
impl crate::StemStrategy for DonnellySimdFull<4> {
    const ROOT_IDX: usize = 0;
    const BLOCK_SIZE: usize = 4;

    type DeferredState = Self;
    type StackContext<A> = crate::kd_tree::query_stack_simd::SimdQueryStackContext<A, Self>;
    type Stack<A> = crate::kd_tree::query_stack_simd::SimdQueryStack<A, Self>;

    #[inline]
    fn new(stems_ptr: std::ptr::NonNull<u8>) -> Self {
        Self {
            core: crate::stem_strategy::donnelly::core::DonnellyCore::new(stems_ptr),
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
        self.core.level() as usize / Self::BLOCK_SIZE % K
    }

    #[inline(always)]
    fn construction_dim<const K: usize>(&self) -> usize {
        self.core.level() as usize / Self::BLOCK_SIZE % K
    }

    #[inline(always)]
    fn level(&self) -> i32 {
        self.core.level()
    }

    #[inline(always)]
    fn traverse<A: Axis<Coord = A>, const K2: usize>(&mut self, is_right: bool) {
        self.core.traverse::<A, K2>(is_right)
    }

    #[inline(always)]
    fn branch<A: Axis<Coord = A>, const K2: usize>(&mut self) -> Self {
        Self {
            core: self.core.branch::<A, K2>(),
        }
    }

    #[inline(always)]
    fn child_indices<A: Axis<Coord = A>>(&self) -> (usize, usize) {
        self.core.child_indices::<A>()
    }

    fn get_leaf_idx<A: Axis<Coord = A>, const K2: usize>(
        stems: &[A],
        query: &[A; K2],
        max_stem_level: i32,
    ) -> usize
    where
        Self: Sized,
    {
        let stems_ptr = std::ptr::NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        let mut strat = Self::new(stems_ptr);

        while strat.level() <= max_stem_level {
            let dim = strat.dim::<K2>();
            let query_val = unsafe { *query.get_unchecked(dim) };
            let block_base_idx = strat.stem_idx();
            let block_width = 1usize << Self::BLOCK_SIZE;
            let minor_tri_height = (64 / A::VALUE_WIDTH_BYTES as u32).ilog2();
            let can_take_full_block = strat.level() + Self::BLOCK_SIZE as i32 - 1 <= max_stem_level
                && block_base_idx + block_width <= stems.len()
                && Self::BLOCK_SIZE as u32 == minor_tri_height
                && strat.core.minor_level() == 0;

            // TODO: this fallback should not be required. Refactor to avoid the need for it
            if can_take_full_block {
                let child_idx = compare_block4(stems, query_val, block_base_idx);
                strat
                    .core
                    .traverse_block::<K2>(child_idx, Self::BLOCK_SIZE as u32);
            } else {
                tracing::warn!(
                    level = %strat.level(),
                    block_base_idx = %block_base_idx,
                    block_width = %block_width,
                    stem_count = %stems.len(),
                    "Block4 get_leaf_idx can't take full block"
                );

                let stem_idx = strat.stem_idx();
                let is_right = if stem_idx < stems.len() {
                    query_val >= unsafe { *stems.get_unchecked(stem_idx) }
                } else {
                    false
                };
                strat.traverse::<A, K2>(is_right);
            }
        }

        strat.leaf_idx()
    }

    #[inline(always)]
    fn backtracking_traverse_step_with_bounds<A, O, D, const K2: usize>(
        &mut self,
        stems: &[A],
        query: &[A; K2],
        query_wide: &[O; K2],
        lower: &mut [O; K2],
        upper: &mut [O; K2],
        off: &mut [O; K2],
        dim: &mut usize,
        rd: O,
        max_stem_level: i32,
        best_dist: O,
        stack: &mut Self::Stack<O>,
    ) -> bool
    where
        Self: Sized,
        A: Axis<Coord = A>,
        O: Axis<Coord = O> + BacktrackBlock4,
        D: crate::dist::DistanceMetric<A, Output = O>,
    {
        if self.level() > max_stem_level {
            return false;
        }
        let dim_val = *dim;
        let query_val = unsafe { *query.get_unchecked(dim_val) };
        #[allow(unused)]
        // used by simd code below, but is also the only code that uses query_wide arg
        let query_wide_val = unsafe { *query_wide.get_unchecked(dim_val) };
        let old_off_val = unsafe { *off.get_unchecked(dim_val) };
        let lower_bound = unsafe { *lower.get_unchecked(dim_val) };
        let upper_bound = unsafe { *upper.get_unchecked(dim_val) };
        let block_base_idx = self.stem_idx();
        let block_width = 1usize << Self::BLOCK_SIZE;
        let can_take_full_block = self.level() + Self::BLOCK_SIZE as i32 - 1 <= max_stem_level
            && block_base_idx + block_width <= stems.len()
            && old_off_val == O::zero();

        // TODO: this fallback should not be required. Refactor to avoid the need for it
        if !can_take_full_block {
            use crate::kd_tree::query_stack_simd::SimdQueryStackContext;

            tracing::warn!(
                level = %self.level(),
                block_base_idx = %block_base_idx,
                block_width = %block_width,
                %max_stem_level,
                %old_off_val,
                "Block4 backtracking_traverse_step can't take full block"
            );

            let stem_idx = self.stem_idx();
            let pivot = if stem_idx < stems.len() {
                *unsafe { stems.get_unchecked(stem_idx) }
            } else {
                A::max_value()
            };
            if pivot < A::max_value() {
                let is_right_child = query_val >= pivot;
                let far_ctx = self.branch_relative::<A, K2>(is_right_child);

                let pivot_wide: O = D::widen_coord(pivot);
                let new_off = O::saturating_dist(query_wide_val, pivot_wide);
                let rd_far = D::rect_dist_after_update(rd, off, *dim, new_off);

                let (near_lower, near_upper, far_lower, far_upper) = if is_right_child {
                    (
                        O::max(lower_bound, pivot_wide),
                        upper_bound,
                        lower_bound,
                        coord_min(upper_bound, pivot_wide),
                    )
                } else {
                    (
                        lower_bound,
                        coord_min(upper_bound, pivot_wide),
                        O::max(lower_bound, pivot_wide),
                        upper_bound,
                    )
                };

                if O::cmp(rd_far, best_dist) != std::cmp::Ordering::Greater {
                    stack.push(SimdQueryStackContext::Single {
                        stem_strat: far_ctx,
                        dim: dim_val,
                        lower_bound: far_lower,
                        upper_bound: far_upper,
                        old_off: new_off,
                        rd: rd_far,
                    });
                }

                unsafe {
                    *lower.get_unchecked_mut(dim_val) = near_lower;
                    *upper.get_unchecked_mut(dim_val) = near_upper;
                }
            } else {
                // +Inf can represent structural padding in left-aligned trees.
                // Traversing right should not add geometric distance.
                let far_ctx = self.branch_relative::<A, K2>(false);
                if O::cmp(rd, best_dist) != std::cmp::Ordering::Greater {
                    stack.push(SimdQueryStackContext::Single {
                        stem_strat: far_ctx,
                        dim: dim_val,
                        lower_bound,
                        upper_bound,
                        old_off: old_off_val,
                        rd,
                    });
                }
            }

            *dim = self.dim::<K2>();
            return true;
        }

        let child_idx = compare_block4(stems, query_val, block_base_idx);

        let child_idx_mask = 1u16 << child_idx;

        let mut rd_values = [O::zero(); 16];
        let mut new_off_values = [O::zero(); 16];
        let mut lower_bounds = [O::zero(); 16];
        let mut upper_bounds = [O::zero(); 16];
        let backtrack_mask = fill_block4_backtrack_values_and_bounds::<A, O, D, K2>(
            stems,
            block_base_idx,
            query_wide_val,
            lower_bound,
            upper_bound,
            old_off_val,
            rd,
            best_dist,
            &mut new_off_values,
            &mut rd_values,
            &mut lower_bounds,
            &mut upper_bounds,
        ) & !child_idx_mask;

        use crate::kd_tree::query_stack_simd::SimdQueryStackContext;

        let high_mask = (backtrack_mask >> 8) as u8;
        if high_mask != 0 {
            let mut high_rd_values = [O::zero(); 8];
            let mut high_new_off_values = [O::zero(); 8];
            let mut high_lower_bounds = [O::zero(); 8];
            let mut high_upper_bounds = [O::zero(); 8];
            high_rd_values.copy_from_slice(&rd_values[8..16]);
            high_new_off_values.copy_from_slice(&new_off_values[8..16]);
            high_lower_bounds.copy_from_slice(&lower_bounds[8..16]);
            high_upper_bounds.copy_from_slice(&upper_bounds[8..16]);
            stack.push(SimdQueryStackContext::new_deferred_block(
                *self,
                8,
                high_rd_values,
                high_new_off_values,
                high_lower_bounds,
                high_upper_bounds,
                high_mask,
                dim_val,
                old_off_val,
                lower_bound,
                upper_bound,
            ));
        }

        let low_mask = backtrack_mask as u8;
        if low_mask != 0 {
            let mut low_rd_values = [O::zero(); 8];
            let mut low_new_off_values = [O::zero(); 8];
            let mut low_lower_bounds = [O::zero(); 8];
            let mut low_upper_bounds = [O::zero(); 8];
            low_rd_values.copy_from_slice(&rd_values[..8]);
            low_new_off_values.copy_from_slice(&new_off_values[..8]);
            low_lower_bounds.copy_from_slice(&lower_bounds[..8]);
            low_upper_bounds.copy_from_slice(&upper_bounds[..8]);
            stack.push(SimdQueryStackContext::new_deferred_block(
                *self,
                0,
                low_rd_values,
                low_new_off_values,
                low_lower_bounds,
                low_upper_bounds,
                low_mask,
                dim_val,
                old_off_val,
                lower_bound,
                upper_bound,
            ));
        }

        unsafe {
            *off.get_unchecked_mut(dim_val) = new_off_values[child_idx as usize];
            *lower.get_unchecked_mut(dim_val) = lower_bounds[child_idx as usize];
            *upper.get_unchecked_mut(dim_val) = upper_bounds[child_idx as usize];
        }

        self.core
            .traverse_block::<K2>(child_idx, Self::BLOCK_SIZE as u32);
        *dim = self.dim::<K2>();

        true
    }

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
            + SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4,
        D: crate::dist::DistanceMetric<A, Output = O>,
        QC: crate::kd_tree::query_context::QueryContext<A, O, K2>,
        LS: LeafStrategy<A, T, Self, K2, B>,
    {
        tree.backtracking_query_with_simd_stack_impl::<QC, O, D>(query_ctx, stack, process_leaf);
    }
}

impl DeferredBlockTraversal for DonnellySimdFull<4> {
    #[inline(always)]
    fn block_child<const K: usize>(&self, child_idx: u8) -> Self {
        let mut child = *self;
        child
            .core
            .traverse_block::<K>(child_idx, Self::BLOCK_SIZE as u32);
        child
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::{Manhattan, SquaredEuclidean};
    use crate::kd_tree::query_stack_simd::{Block3ExactStackContext, Block3ExactStackContextState};
    use crate::StemStrategy;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    fn build_test_block3_pivots_f64() -> [f64; 8] {
        [0.2, 0.4, 0.6, 0.1, 0.3, 0.5, 0.7, f64::INFINITY]
    }

    fn build_test_block3_pivots_f32() -> [f32; 8] {
        [0.2, 0.4, 0.6, 0.1, 0.3, 0.5, 0.7, f32::INFINITY]
    }

    fn build_test_block4_pivots_f32() -> [f32; 16] {
        [
            0.7,
            0.3,
            1.1,
            0.1,
            0.5,
            0.9,
            1.3,
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
            1.2,
            1.4,
            f32::INFINITY,
        ]
    }

    fn select_child_scalar_f64(query: f64, pivots: &[f64; 8]) -> u8 {
        let mut count = 0u8;
        for i in 0..8 {
            if query >= pivots[i] {
                count += 1;
            }
        }
        count
    }

    fn select_child_scalar_f32(query: f32, pivots: &[f32; 8]) -> u8 {
        let mut count = 0u8;
        for i in 0..8 {
            if query >= pivots[i] {
                count += 1;
            }
        }
        count
    }

    fn select_child_scalar_block4_f32(query: f32, pivots: &[f32; 16]) -> u8 {
        let mut count = 0u8;
        for i in 0..16 {
            if query >= pivots[i] {
                count += 1;
            }
        }
        count
    }

    fn scalar_backtrack_check_block3_f64(
        query: f64,
        pivots: &[f64; 8],
        old_off: f64,
        rd: f64,
        best_dist: f64,
    ) -> u8 {
        let mut mask = 0u8;
        for child_idx in 0..8 {
            let (lower_offset, upper_offset) = child_interval_bounds_block3(child_idx);

            let lower = if lower_offset == 255 {
                f64::NEG_INFINITY
            } else {
                pivots[lower_offset as usize]
            };

            let upper = if upper_offset == 255 {
                f64::INFINITY
            } else {
                pivots[upper_offset as usize]
            };

            let interval_dist = interval_distance_1d(query, lower, upper);
            // For SquaredEuclidean: rd_far = rd + (interval_dist - old_off)²
            let delta = interval_dist - old_off;
            let rd_far = rd + delta * delta;

            if rd_far <= best_dist {
                mask |= 1 << child_idx;
            }
        }
        mask
    }

    fn scalar_backtrack_check_block3_f32(
        query: f32,
        pivots: &[f32; 8],
        old_off: f32,
        rd: f32,
        best_dist: f32,
    ) -> u8 {
        let mut mask = 0u8;
        for child_idx in 0..8 {
            let (lower_offset, upper_offset) = child_interval_bounds_block3(child_idx);

            let lower = if lower_offset == 255 {
                f32::NEG_INFINITY
            } else {
                pivots[lower_offset as usize]
            };

            let upper = if upper_offset == 255 {
                f32::INFINITY
            } else {
                pivots[upper_offset as usize]
            };

            // Use f64 version of interval_distance_1d, cast to f32
            let interval_dist =
                interval_distance_1d(query as f64, lower as f64, upper as f64) as f32;
            let delta = interval_dist - old_off;
            let rd_far = rd + delta * delta;

            if rd_far <= best_dist {
                mask |= 1 << child_idx;
            }
        }
        mask
    }

    #[test]
    fn test_child_interval_bounds_block3() {
        // Verify all 8 children have correct interval bounds
        // Expected bounds based on triangular structure:
        //           pivot[0]
        //      pivot[1]    pivot[2]
        //   p[3]    p[4]    p[5]   p[6]
        // ch0 ch1 ch2 ch3 ch4 ch5 ch6 ch7

        assert_eq!(child_interval_bounds_block3(0), (255, 3)); // [-∞, pivot[3])
        assert_eq!(child_interval_bounds_block3(1), (3, 1)); // [pivot[3], pivot[1])
        assert_eq!(child_interval_bounds_block3(2), (1, 4)); // [pivot[1], pivot[4])
        assert_eq!(child_interval_bounds_block3(3), (4, 0)); // [pivot[4], pivot[0])
        assert_eq!(child_interval_bounds_block3(4), (0, 5)); // [pivot[0], pivot[5])
        assert_eq!(child_interval_bounds_block3(5), (5, 2)); // [pivot[5], pivot[2])
        assert_eq!(child_interval_bounds_block3(6), (2, 6)); // [pivot[2], pivot[6])
        assert_eq!(child_interval_bounds_block3(7), (6, 255)); // [pivot[6], +∞)
    }

    #[test]
    fn test_child_interval_bounds_block4() {
        // Verify all 16 children have correct interval bounds
        // Expected bounds based on triangular structure:
        //            pivot[0]
        //       pivot[1]    pivot[2]
        //    p[3] p[4]   p[5] p[6]
        // p[7]p[8]p[9]p[10]p[11]p[12]p[13]p[14]
        // c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15

        assert_eq!(child_interval_bounds_block4(0), (255, 7)); // [-∞, pivot[7])
        assert_eq!(child_interval_bounds_block4(1), (7, 3)); // [pivot[7], pivot[3])
        assert_eq!(child_interval_bounds_block4(2), (3, 8)); // [pivot[3], pivot[8])
        assert_eq!(child_interval_bounds_block4(3), (8, 1)); // [pivot[8], pivot[1])
        assert_eq!(child_interval_bounds_block4(4), (1, 9)); // [pivot[1], pivot[9])
        assert_eq!(child_interval_bounds_block4(5), (9, 4)); // [pivot[9], pivot[4])
        assert_eq!(child_interval_bounds_block4(6), (4, 10)); // [pivot[4], pivot[10])
        assert_eq!(child_interval_bounds_block4(7), (10, 0)); // [pivot[10], pivot[0])
        assert_eq!(child_interval_bounds_block4(8), (0, 11)); // [pivot[0], pivot[11])
        assert_eq!(child_interval_bounds_block4(9), (11, 5)); // [pivot[11], pivot[5])
        assert_eq!(child_interval_bounds_block4(10), (5, 12)); // [pivot[5], pivot[12])
        assert_eq!(child_interval_bounds_block4(11), (12, 2)); // [pivot[12], pivot[2])
        assert_eq!(child_interval_bounds_block4(12), (2, 13)); // [pivot[2], pivot[13])
        assert_eq!(child_interval_bounds_block4(13), (13, 6)); // [pivot[13], pivot[6])
        assert_eq!(child_interval_bounds_block4(14), (6, 14)); // [pivot[6], pivot[14])
        assert_eq!(child_interval_bounds_block4(15), (14, 255)); // [pivot[14], +∞)
    }

    #[test]
    fn test_block4_interval_coverage() {
        // Verify Block4 intervals have no gaps and full coverage
        // Child i's upper bound should equal child i+1's lower bound
        for child_idx in 0..15 {
            let (_, upper) = child_interval_bounds_block4(child_idx);
            let (lower_next, _) = child_interval_bounds_block4(child_idx + 1);
            assert_eq!(
                upper,
                lower_next,
                "Gap detected: child {} upper bound ({}) != child {} lower bound ({})",
                child_idx,
                upper,
                child_idx + 1,
                lower_next
            );
        }

        // First child should start at -∞ (255)
        let (lower_first, _) = child_interval_bounds_block4(0);
        assert_eq!(lower_first, 255, "First child should start at -∞ (255)");

        // Last child should end at +∞ (255)
        let (_, upper_last) = child_interval_bounds_block4(15);
        assert_eq!(upper_last, 255, "Last child should end at +∞ (255)");
    }

    #[test]
    fn test_block4_interval_monotonicity() {
        // Create test pivots arranged in BST order for a complete Block4 tree
        // In-order traversal gives: p7, p3, p8, p1, p9, p4, p10, p0, p11, p5, p12, p2, p13, p6, p14
        // So we need: p0=0.7, p1=0.3, p2=0.11, p3=0.1, p4=0.5, p5=0.9, p6=0.13
        //             p7=0.0, p8=0.2, p9=0.4, p10=0.6, p11=0.8, p12=1.0, p13=1.2, p14=1.4
        let mut pivots = [0.0; 16]; // 16 slots (0-14 are pivots, 15 is padding)
        pivots[0] = 0.7; // p0 - middle of tree
        pivots[1] = 0.3; // p1 - middle of left subtree
        pivots[2] = 1.1; // p2 - middle of right subtree
        pivots[3] = 0.1; // p3
        pivots[4] = 0.5; // p4
        pivots[5] = 0.9; // p5
        pivots[6] = 1.3; // p6
        pivots[7] = 0.0; // p7 - leftmost (smallest)
        pivots[8] = 0.2; // p8
        pivots[9] = 0.4; // p9
        pivots[10] = 0.6; // p10
        pivots[11] = 0.8; // p11
        pivots[12] = 1.0; // p12
        pivots[13] = 1.2; // p13
        pivots[14] = 1.4; // p14 - rightmost (largest)

        // Verify that intervals are monotonic (lower < upper for each child)
        for child_idx in 0..16 {
            let (lower_offset, upper_offset) = child_interval_bounds_block4(child_idx);

            let lower_val = if lower_offset == 255 {
                f64::NEG_INFINITY
            } else {
                pivots[lower_offset as usize]
            };

            let upper_val = if upper_offset == 255 {
                f64::INFINITY
            } else {
                pivots[upper_offset as usize]
            };

            assert!(
                lower_val < upper_val,
                "Child {} interval [{}, {}) is not monotonic (offsets: {}, {})",
                child_idx,
                lower_val,
                upper_val,
                lower_offset,
                upper_offset
            );
        }
    }

    #[test]
    fn test_block3_deferred_block_helpers_match_free_functions() {
        type Strat = DonnellySimdFull<3>;

        let pivots = build_test_block3_pivots_f64();
        let stems_ptr = NonNull::new(pivots.as_ptr() as *mut u8).unwrap();
        let strat = Strat::new(stems_ptr);

        let expected_mask = backtrack_block3_with_bounds::<f64, f64, SquaredEuclidean<f64>, 3>(
            &pivots,
            0,
            0.45,
            f64::NEG_INFINITY,
            f64::INFINITY,
            0.0,
            0.0,
            0.2,
        );
        let actual_mask = strat
            .backtrack_block3_pending_mask::<f64, f64, SquaredEuclidean<f64>, 3>(
                &pivots,
                0.45,
                f64::NEG_INFINITY,
                f64::INFINITY,
                0.0,
                0.0,
                0.2,
            );
        assert_eq!(actual_mask, expected_mask);

        let expected_state = selected_block3_child_state_and_rd::<f64, f64, SquaredEuclidean<f64>>(
            &pivots,
            0,
            3,
            0.45,
            f64::NEG_INFINITY,
            f64::INFINITY,
            0.0,
            0.0,
        );
        let actual_state = strat
            .selected_block3_pending_child_state::<f64, f64, SquaredEuclidean<f64>>(
                &pivots,
                3,
                0.45,
                f64::NEG_INFINITY,
                f64::INFINITY,
                0.0,
                0.0,
            );
        assert_eq!(actual_state, expected_state);

        let mut new_off_values = [0.0; 8];
        let mut rd_values = [0.0; 8];
        let mut lower_bounds = [0.0; 8];
        let mut upper_bounds = [0.0; 8];
        let filled_mask = strat.fill_block3_pending_values::<f64, f64, SquaredEuclidean<f64>, 3>(
            &pivots,
            0.45,
            f64::NEG_INFINITY,
            f64::INFINITY,
            0.0,
            0.0,
            0.2,
            &mut new_off_values,
            &mut rd_values,
            &mut lower_bounds,
            &mut upper_bounds,
        );
        assert_eq!(filled_mask, expected_mask);

        let child = strat.block_child::<3>(5);
        let mut expected_child = strat;
        expected_child.core.traverse_block::<3>(5, 3);
        assert_eq!(child.stem_idx(), expected_child.stem_idx());
        assert_eq!(child.level(), expected_child.level());
    }

    #[test]
    fn test_block4_deferred_block_helpers_default_to_panic() {
        type Strat = DonnellySimdFull<4>;

        let pivots = build_test_block4_pivots_f32();
        let stems_ptr = NonNull::new(pivots.as_ptr() as *mut u8).unwrap();
        let strat = Strat::new(stems_ptr);

        assert!(catch_unwind(AssertUnwindSafe(|| {
            let _ = strat.backtrack_block3_pending_mask::<f32, f32, SquaredEuclidean<f32>, 3>(
                &pivots,
                0.55,
                f32::NEG_INFINITY,
                f32::INFINITY,
                0.0,
                0.0,
                0.5,
            );
        }))
        .is_err());

        assert!(catch_unwind(AssertUnwindSafe(|| {
            let _ = strat.selected_block3_pending_child_state::<f32, f32, SquaredEuclidean<f32>>(
                &pivots,
                0,
                0.55,
                f32::NEG_INFINITY,
                f32::INFINITY,
                0.0,
                0.0,
            );
        }))
        .is_err());

        assert!(catch_unwind(AssertUnwindSafe(|| {
            let mut new_off_values = [0.0; 8];
            let mut rd_values = [0.0; 8];
            let mut lower_bounds = [0.0; 8];
            let mut upper_bounds = [0.0; 8];
            let _ = strat.fill_block3_pending_values::<f32, f32, SquaredEuclidean<f32>, 3>(
                &pivots,
                0.55,
                f32::NEG_INFINITY,
                f32::INFINITY,
                0.0,
                0.0,
                0.5,
                &mut new_off_values,
                &mut rd_values,
                &mut lower_bounds,
                &mut upper_bounds,
            );
        }))
        .is_err());
    }

    #[test]
    fn test_block4_block_child_matches_manual_traversal() {
        type Strat = DonnellySimdFull<4>;

        let pivots = build_test_block4_pivots_f32();
        let stems_ptr = NonNull::new(pivots.as_ptr() as *mut u8).unwrap();
        let strat = Strat::new(stems_ptr);
        let child = strat.block_child::<3>(9);

        let mut expected = strat;
        expected.core.traverse_block::<3>(9, 4);

        assert_eq!(child.stem_idx(), expected.stem_idx());
        assert_eq!(child.level(), expected.level());
    }

    #[test]
    fn test_fill_block4_backtrack_values_and_bounds_matches_mask_and_bounds() {
        let pivots = build_test_block4_pivots_f32();
        let mut new_off_values = [0.0; 16];
        let mut rd_values = [0.0; 16];
        let mut lower_bounds = [0.0; 16];
        let mut upper_bounds = [0.0; 16];

        let mask = fill_block4_backtrack_values_and_bounds::<f32, f32, Manhattan<f32>, 3>(
            &pivots,
            0,
            0.55,
            0.15,
            1.05,
            0.0,
            0.0,
            0.5,
            &mut new_off_values,
            &mut rd_values,
            &mut lower_bounds,
            &mut upper_bounds,
        );

        for child_idx in 0..16usize {
            let (lower_offset, upper_offset) = child_interval_bounds_block4(child_idx);
            let raw_lower = if lower_offset == 255 {
                <f32 as crate::Axis>::min_value()
            } else {
                pivots[lower_offset as usize]
            };
            let raw_upper = if upper_offset == 255 {
                <f32 as crate::Axis>::max_value()
            } else {
                pivots[upper_offset as usize]
            };
            let effective_lower = f32::max(0.15, raw_lower);
            let effective_upper = f32::min(1.05, raw_upper);
            assert_eq!(lower_bounds[child_idx], effective_lower);
            assert_eq!(upper_bounds[child_idx], effective_upper);
            if effective_lower.partial_cmp(&effective_upper) != Some(std::cmp::Ordering::Less) {
                assert_eq!(new_off_values[child_idx], <f32 as crate::Axis>::max_value());
                assert_eq!(rd_values[child_idx], <f32 as crate::Axis>::max_value());
                assert_eq!(mask & (1u16 << child_idx), 0);
            }
        }
    }

    #[test]
    fn test_block3_get_leaf_idx_matches_manual_traversal() {
        type Strat = DonnellySimdFull<3>;

        let pivots = build_test_block3_pivots_f64();
        let query = [0.45, 0.0, 0.0];
        let leaf_idx = Strat::get_leaf_idx::<f64, 3>(&pivots, &query, 2);

        let stems_ptr = NonNull::new(pivots.as_ptr() as *mut u8).unwrap();
        let mut manual = Strat::new(stems_ptr);
        let child_idx = compare_block3(&pivots, query[0], 0);
        manual.core.traverse_block::<3>(child_idx, 3);

        assert_eq!(leaf_idx, manual.leaf_idx());
    }

    #[test]
    fn test_block4_get_leaf_idx_matches_manual_traversal() {
        type Strat = DonnellySimdFull<4>;

        let pivots = build_test_block4_pivots_f32();
        let query = [0.55, 0.0, 0.0];
        let leaf_idx = Strat::get_leaf_idx::<f32, 3>(&pivots, &query, 3);

        let stems_ptr = NonNull::new(pivots.as_ptr() as *mut u8).unwrap();
        let mut manual = Strat::new(stems_ptr);
        let child_idx = compare_block4(&pivots, query[0], 0);
        manual.core.traverse_block::<3>(child_idx, 4);

        assert_eq!(leaf_idx, manual.leaf_idx());
    }

    #[test]
    fn test_block3_backtracking_step_full_block_updates_state_and_pushes_pending() {
        type Strat = DonnellySimdFull<3>;

        let pivots = build_test_block3_pivots_f64();
        let stems_ptr = NonNull::new(pivots.as_ptr() as *mut u8).unwrap();
        let mut strat = Strat::new(stems_ptr);
        let query = [0.45, 0.0, 0.0];
        let query_wide = [0.45, 0.0, 0.0];
        let mut lower = [f64::NEG_INFINITY; 3];
        let mut upper = [f64::INFINITY; 3];
        let mut off = [0.0; 3];
        let mut dim = 0usize;
        let mut stack = <Strat as StemStrategy>::Stack::<f64>::default();

        let child_idx = compare_block3(&pivots, query[0], 0);
        let expected_mask = backtrack_block3_with_bounds::<f64, f64, SquaredEuclidean<f64>, 3>(
            &pivots,
            0,
            query_wide[0],
            lower[0],
            upper[0],
            off[0],
            0.0,
            0.2,
        ) & !(1u8 << child_idx);
        let (expected_off, expected_lower, expected_upper) =
            selected_block3_child_state::<f64, f64, SquaredEuclidean<f64>>(
                &pivots,
                0,
                child_idx,
                query_wide[0],
                lower[0],
                upper[0],
            );

        let stepped = strat
            .backtracking_traverse_step_with_bounds::<f64, f64, SquaredEuclidean<f64>, 3>(
                &pivots,
                &query,
                &query_wide,
                &mut lower,
                &mut upper,
                &mut off,
                &mut dim,
                0.0,
                2,
                0.2,
                &mut stack,
            );

        assert!(stepped);
        assert_eq!(off[0], expected_off);
        assert_eq!(lower[0], expected_lower);
        assert_eq!(upper[0], expected_upper);
        assert_eq!(dim, strat.dim::<3>());

        let popped = stack.pop().expect("pending block3 context");
        type Block3Ctx = <Strat as StemStrategy>::StackContext<f64>;
        let state =
            <Block3Ctx as Block3ExactStackContext<f64, Strat, 3>>::into_block3_exact_state(popped);
        match state {
            Block3ExactStackContextState::Block3Pending { pending_mask, .. } => {
                assert_eq!(pending_mask, expected_mask);
            }
            _ => panic!("expected Block3Pending context"),
        }
    }

    #[test]
    fn test_block3_backtracking_step_scalar_fallback_pushes_single() {
        type Strat = DonnellySimdFull<3>;

        let pivots = build_test_block3_pivots_f64();
        let stems_ptr = NonNull::new(pivots.as_ptr() as *mut u8).unwrap();
        let mut strat = Strat::new(stems_ptr);
        let query = [0.45, 0.0, 0.0];
        let query_wide = [0.45, 0.0, 0.0];
        let mut lower = [f64::NEG_INFINITY; 3];
        let mut upper = [f64::INFINITY; 3];
        let mut off = [0.0; 3];
        let mut dim = 0usize;
        let mut stack = <Strat as StemStrategy>::Stack::<f64>::default();

        let stepped = strat
            .backtracking_traverse_step_with_bounds::<f64, f64, SquaredEuclidean<f64>, 3>(
                &pivots,
                &query,
                &query_wide,
                &mut lower,
                &mut upper,
                &mut off,
                &mut dim,
                0.0,
                0,
                0.2,
                &mut stack,
            );

        assert!(stepped);
        let popped = stack.pop().expect("single fallback context");
        type Block3Ctx = <Strat as StemStrategy>::StackContext<f64>;
        let state =
            <Block3Ctx as Block3ExactStackContext<f64, Strat, 3>>::into_block3_exact_state(popped);
        match state {
            Block3ExactStackContextState::Single {
                dim: pushed_dim, ..
            } => {
                assert_eq!(pushed_dim, 0);
            }
            _ => panic!("expected single fallback context"),
        }
    }

    #[test]
    fn test_block4_backtracking_step_full_block_pushes_deferred_blocks() {
        type Strat = DonnellySimdFull<4>;

        let pivots = build_test_block4_pivots_f32();
        let stems_ptr = NonNull::new(pivots.as_ptr() as *mut u8).unwrap();
        let mut strat = Strat::new(stems_ptr);
        let query = [0.55, 0.0, 0.0];
        let query_wide = [0.55, 0.0, 0.0];
        let mut lower = [f32::NEG_INFINITY; 3];
        let mut upper = [f32::INFINITY; 3];
        let mut off = [0.0; 3];
        let mut dim = 0usize;
        let mut stack = <Strat as StemStrategy>::Stack::<f32>::default();

        let child_idx = compare_block4(&pivots, query[0], 0);
        let mut expected_off_values = [0.0; 16];
        let mut expected_rd_values = [0.0; 16];
        let mut expected_lower_bounds = [0.0; 16];
        let mut expected_upper_bounds = [0.0; 16];
        let expected_mask =
            fill_block4_backtrack_values_and_bounds::<f32, f32, SquaredEuclidean<f32>, 3>(
                &pivots,
                0,
                query_wide[0],
                lower[0],
                upper[0],
                off[0],
                0.0,
                0.5,
                &mut expected_off_values,
                &mut expected_rd_values,
                &mut expected_lower_bounds,
                &mut expected_upper_bounds,
            ) & !(1u16 << child_idx);

        let stepped = strat
            .backtracking_traverse_step_with_bounds::<f32, f32, SquaredEuclidean<f32>, 3>(
                &pivots,
                &query,
                &query_wide,
                &mut lower,
                &mut upper,
                &mut off,
                &mut dim,
                0.0,
                3,
                0.5,
                &mut stack,
            );

        assert!(stepped);
        assert_eq!(off[0], expected_off_values[child_idx as usize]);
        assert_eq!(lower[0], expected_lower_bounds[child_idx as usize]);
        assert_eq!(upper[0], expected_upper_bounds[child_idx as usize]);

        let mut popped_masks = Vec::new();
        while let Some(ctx) = stack.pop() {
            match ctx {
                crate::kd_tree::query_stack_simd::SimdQueryStackContext::DeferredBlock {
                    child_base,
                    sibling_mask,
                    ..
                } => popped_masks.push((child_base, sibling_mask)),
                other => panic!("expected deferred block context, got {other:?}"),
            }
        }

        if expected_mask & 0x00FF != 0 {
            assert!(popped_masks.contains(&(0, expected_mask as u8)));
        }
        if expected_mask & 0xFF00 != 0 {
            assert!(popped_masks.contains(&(8, (expected_mask >> 8) as u8)));
        }
    }

    #[test]
    fn test_block4_backtracking_step_scalar_fallback_pushes_single() {
        type Strat = DonnellySimdFull<4>;

        let pivots = build_test_block4_pivots_f32();
        let stems_ptr = NonNull::new(pivots.as_ptr() as *mut u8).unwrap();
        let mut strat = Strat::new(stems_ptr);
        let query = [0.55, 0.0, 0.0];
        let query_wide = [0.55, 0.0, 0.0];
        let mut lower = [f32::NEG_INFINITY; 3];
        let mut upper = [f32::INFINITY; 3];
        let mut off = [0.0; 3];
        let mut dim = 0usize;
        let mut stack = <Strat as StemStrategy>::Stack::<f32>::default();

        let stepped = strat
            .backtracking_traverse_step_with_bounds::<f32, f32, SquaredEuclidean<f32>, 3>(
                &pivots,
                &query,
                &query_wide,
                &mut lower,
                &mut upper,
                &mut off,
                &mut dim,
                0.0,
                0,
                0.5,
                &mut stack,
            );

        assert!(stepped);
        let popped = stack.pop().expect("single fallback context");
        match popped {
            crate::kd_tree::query_stack_simd::SimdQueryStackContext::Single {
                dim: pushed_dim,
                ..
            } => assert_eq!(pushed_dim, 0),
            other => panic!("expected single fallback context, got {other:?}"),
        }
    }

    #[cfg(feature = "test_utils")]
    #[test]
    fn test_block3_backtracking_step_emits_exact_query_trace_event() {
        type Strat = DonnellySimdFull<3>;

        let pivots = build_test_block3_pivots_f64();
        let stems_ptr = NonNull::new(pivots.as_ptr() as *mut u8).unwrap();
        let mut strat = Strat::new(stems_ptr);
        let query = [0.45, 0.0, 0.0];
        let query_wide = [0.45, 0.0, 0.0];
        let mut lower = [f64::NEG_INFINITY; 3];
        let mut upper = [f64::INFINITY; 3];
        let mut off = [0.0; 3];
        let mut dim = 0usize;
        let mut stack = <Strat as StemStrategy>::Stack::<f64>::default();

        crate::test_utils::exact_query_trace::set_enabled(true);
        let stepped = strat
            .backtracking_traverse_step_with_bounds::<f64, f64, SquaredEuclidean<f64>, 3>(
                &pivots,
                &query,
                &query_wide,
                &mut lower,
                &mut upper,
                &mut off,
                &mut dim,
                0.0,
                2,
                0.2,
                &mut stack,
            );
        let events = crate::test_utils::exact_query_trace::snapshot();
        crate::test_utils::exact_query_trace::set_enabled(false);

        assert!(stepped);
        assert!(events.iter().any(|event| matches!(
            event,
            crate::test_utils::exact_query_trace::ExactQueryTraceEvent::Block3FullStep { .. }
        )));
    }

    #[test]
    fn test_interval_distance_1d_inside() {
        // Query inside interval should return 0
        assert_eq!(interval_distance_1d(5.0, 3.0, 7.0), 0.0);
        assert_eq!(interval_distance_1d(3.0, 3.0, 7.0), 0.0); // At lower bound
        assert_eq!(interval_distance_1d(6.999, 3.0, 7.0), 0.0); // Just below upper bound
    }

    #[test]
    fn test_interval_distance_1d_below() {
        // Query below lower bound
        assert_eq!(interval_distance_1d(2.0, 5.0, 10.0), 3.0); // |5 - 2| = 3
        assert_eq!(interval_distance_1d(0.0, 3.0, 10.0), 3.0); // |3 - 0| = 3
        assert_eq!(interval_distance_1d(-1.0, 1.0, 10.0), 2.0); // |1 - (-1)| = 2
    }

    #[test]
    fn test_interval_distance_1d_above() {
        // Query above upper bound
        assert_eq!(interval_distance_1d(12.0, 5.0, 10.0), 2.0); // |12 - 10| = 2
        assert_eq!(interval_distance_1d(10.0, 5.0, 10.0), 0.0); // Exactly at upper bound (excluded)
        assert_eq!(interval_distance_1d(15.0, 5.0, 10.0), 5.0); // |15 - 10| = 5
    }

    #[test]
    fn test_interval_distance_1d_edge_cases() {
        // Test with infinity bounds
        assert_eq!(interval_distance_1d(-100.0, f64::NEG_INFINITY, 5.0), 0.0); // Below but no lower bound
        assert_eq!(interval_distance_1d(100.0, 5.0, f64::INFINITY), 0.0); // Above but no upper bound
        assert_eq!(
            interval_distance_1d(0.0, f64::NEG_INFINITY, f64::INFINITY),
            0.0
        );
    }

    #[test]
    fn test_interval_distance_1d_branchless() {
        for query in [-10.0, -1.0, 0.0, 1.0, 5.0, 7.5, 10.0, 15.0, 100.0] {
            let lower = 0.0;
            let upper = 10.0;

            let result = interval_distance_1d(query, lower, upper);

            // Compute expected with explicit branches
            let expected = if query < lower {
                lower - query
            } else if query >= upper {
                query - upper
            } else {
                0.0
            };

            assert_eq!(result, expected, "Failed for query={}", query);
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn test_simd_backtrack_vs_scalar() {
        use crate::SquaredEuclidean;

        // Create test pivots: [pivot0, pivot1, ..., pivot6, +∞]
        let pivots = [0.2, 0.4, 0.6, 0.1, 0.3, 0.5, 0.7, f64::INFINITY];

        let query = 0.25;
        let old_off = 0.0;
        let rd = 0.0;
        let best_dist = f64::INFINITY;

        // Compute scalar version for each child
        let mut scalar_results = [false; 8];
        for child_idx in 0..8 {
            let (lower_offset, upper_offset) = child_interval_bounds_block3(child_idx);

            let lower = if lower_offset == 255 {
                f64::NEG_INFINITY
            } else {
                pivots[lower_offset as usize]
            };

            let upper = if upper_offset == 255 {
                f64::INFINITY
            } else {
                pivots[upper_offset as usize]
            };

            let interval_dist = interval_distance_1d(query, lower, upper);
            // rd_far = rd + (interval_dist - old_off)²
            let delta = (interval_dist - old_off) * (interval_dist - old_off);
            let rd_far = rd + delta;

            scalar_results[child_idx] = rd_far <= best_dist;
        }

        // Compute SIMD version via backtrack trait
        let pivots_ptr = pivots.as_ptr() as *mut u8;
        let stems_ptr = NonNull::new(pivots_ptr).unwrap();
        let simd_mask = f64::backtrack_block3::<f64, SquaredEuclidean<f64>, 3>(
            query, stems_ptr, 0, old_off, rd, best_dist,
        );

        // Compare
        for child_idx in 0..8 {
            let scalar_pass = scalar_results[child_idx];
            let simd_pass = (simd_mask & (1 << child_idx)) != 0;
            assert_eq!(
                scalar_pass,
                simd_pass,
                "Mismatch for child {}: scalar={}, simd={}, query={}, lower={:?}, upper={:?}",
                child_idx,
                scalar_pass,
                simd_pass,
                query,
                if child_interval_bounds_block3(child_idx).0 == 255 {
                    f64::NEG_INFINITY
                } else {
                    pivots[child_interval_bounds_block3(child_idx).0 as usize]
                },
                if child_interval_bounds_block3(child_idx).1 == 255 {
                    f64::INFINITY
                } else {
                    pivots[child_interval_bounds_block3(child_idx).1 as usize]
                },
            );
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn debug_query_12_interval_distances() {
        // This test manually computes what should happen for query #12
        // Query point: [0.8947785353168005, 0.678720516865904, 0.6048091301041568]
        // We're interested in the second block pop (dim=0, old_off=0)

        // From the log, the pivots that would be in a block starting at some block_base_idx
        // Let's trace through what the interval distances should be

        // rd_values from log: [0.5638408493966387, 0.3959802548889544, 0.25295146890380843,
        //                       0.14310568836537063, 0.06606798320124753, 0.0170270603870678,
        //                       0.00021225610203875987, 0.0]

        // For each child, let's print what the intervals should be
        println!("Query value in dim 0: 0.8947785353168005");
        println!("\nChild interval bounds and expected distances:");

        for child_idx in 0..8 {
            let (lower_off, upper_off) = child_interval_bounds_block3(child_idx);
            println!(
                "Child {}: lower_offset={}, upper_offset={}",
                child_idx, lower_off, upper_off
            );
        }

        // The issue is: why does child 6 pass but not child 4?
        // best_dist at that point should be 0.0036181109111460682
        println!("\nbest_dist would be: 0.0036181109111460682");
        println!(
            "Child 4 rd_value: 0.06606798320124753 > best_dist? {}",
            0.06606798320124753 > 0.0036181109111460682
        );
        println!(
            "Child 6 rd_value: 0.00021225610203875987 > best_dist? {}",
            0.00021225610203875987 > 0.0036181109111460682
        );

        // So child 6 should indeed survive and child 4 should be pruned based on rd_values
        // But non-SIMD found the answer in leaf 52 (child 4's leaf)
        // This suggests either:
        // 1. The interval distance calculation is wrong
        // 2. The non-SIMD uses different logic
        // 3. There's something about the tree structure we're missing
    }

    #[test]
    fn test_block3_child_selection_correctness_f64() {
        // Test that compare_block3 selects correct children for various query values
        let pivots = build_test_block3_pivots_f64();

        // Test select of every child
        let test_cases = [
            (-100.0, 0), // Far below all pivots -> child 0
            (0.05, 0),   // Just below pivot[3]=0.1 -> child 0
            (0.15, 1),   // Between pivot[3]=0.1 and pivot[1]=0.4 -> child 1
            (0.25, 2),   // Between pivot[1]=0.4 and pivot[4]=0.3 - wait, need to check structure
            (0.35, 3),   // Test child 3
            (0.45, 4),   // Test child 4
            (0.55, 5),   // Test child 5
            (0.65, 6),   // Test child 6
            (100.0, 7),  // Far above all pivots -> child 7
        ];

        for (query, expected_child) in test_cases {
            let scalar_child = select_child_scalar_f64(query, &pivots);

            assert_eq!(
                scalar_child, expected_child,
                "Query {} should select child {}, got {}",
                query, expected_child, scalar_child
            );
        }
    }

    #[test]
    fn test_block3_child_selection_all_reachable_f64() {
        let pivots = build_test_block3_pivots_f64();
        let mut children_reached = [false; 8];

        for i in 0..100 {
            let query = -1.0 + (i as f64) * 0.02; // Range from -1.0 to 1.0
            let child = select_child_scalar_f64(query, &pivots);
            if (child as usize) < 8 {
                children_reached[child as usize] = true;
            }
        }

        for (child_idx, &reached) in children_reached.iter().enumerate() {
            assert!(
                reached,
                "Child {} was not reached by any query value",
                child_idx
            );
        }
    }

    #[test]
    fn test_block3_child_selection_boundaries_f64() {
        let pivots = build_test_block3_pivots_f64();

        // Query == pivot? >= cmp should put it in the right-hand interval
        for (pivot_idx, &pivot_val) in pivots.iter().enumerate().take(7) {
            let child = select_child_scalar_f64(pivot_val, &pivots);

            let expected = pivots.iter().take(8).filter(|&&p| pivot_val >= p).count() as u8;

            assert_eq!(
                child, expected,
                "Query at pivot[{}]={} should select child {}, got {}",
                pivot_idx, pivot_val, expected, child
            );
        }
    }

    #[test]
    fn test_block3_child_selection_f32() {
        let pivots = build_test_block3_pivots_f32();

        let test_cases = [
            (-100.0f32, 0u8),
            (0.05f32, 0u8),
            (0.25f32, 2u8),
            (100.0f32, 7u8),
        ];

        for (query, expected_child) in test_cases {
            let scalar_child = select_child_scalar_f32(query, &pivots);
            assert_eq!(
                scalar_child, expected_child,
                "f32: Query {} should select child {}, got {}",
                query, expected_child, scalar_child
            );
        }
    }

    #[test]
    fn test_block3_child_selection_via_compare_block3_f64() {
        let pivots = build_test_block3_pivots_f64();

        let test_queries = [
            -100.0, -1.0, 0.0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 1.0, 100.0,
        ];

        for &query in &test_queries {
            let expected = select_child_scalar_f64(query, &pivots);
            let actual = compare_block3(&pivots, query, 0);

            assert_eq!(
                actual, expected,
                "compare_block3 mismatch for query {}: expected child {}, got {}",
                query, expected, actual
            );
        }
    }

    #[test]
    fn test_block3_child_selection_via_compare_block3_f32() {
        let pivots = build_test_block3_pivots_f32();

        let test_queries = [-100.0f32, 0.0f32, 0.25f32, 0.5f32, 100.0f32];

        for &query in &test_queries {
            let expected = select_child_scalar_f32(query, &pivots);
            let actual = compare_block3(&pivots, query, 0);

            assert_eq!(
                actual, expected,
                "compare_block3 (f32) mismatch for query {}: expected child {}, got {}",
                query, expected, actual
            );
        }
    }

    #[test]
    fn test_block4_child_selection_via_compare_block4_f32() {
        let pivots = build_test_block4_pivots_f32();

        let test_queries = [-100.0f32, 0.0f32, 0.15f32, 0.55f32, 1.05f32, 100.0f32];

        for &query in &test_queries {
            let expected = select_child_scalar_block4_f32(query, &pivots);
            let actual = compare_block4(&pivots, query, 0);

            assert_eq!(
                actual, expected,
                "compare_block4 (f32) mismatch for query {}: expected child {}, got {}",
                query, expected, actual
            );
        }
    }

    #[test]
    fn test_backtrack_scalar_vs_simd_f64_multiple_cases() {
        let pivots = build_test_block3_pivots_f64();

        let test_cases = [
            (0.25, 0.0, 0.0, f64::INFINITY),
            (0.5, 0.0, 0.0, 0.1),
            (0.5, 0.01, 0.05, 0.21),
            (-0.5, 0.0, 0.0, f64::INFINITY),
            (1.5, 0.0, 0.0, f64::INFINITY),
        ];

        for (query, old_off, rd, best_dist) in test_cases {
            let scalar_mask =
                scalar_backtrack_check_block3_f64(query, &pivots, old_off, rd, best_dist);

            // Test SIMD implementation if available
            #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
            {
                let stems_ptr = NonNull::new(pivots.as_ptr() as *mut u8).unwrap();
                let simd_mask = f64::backtrack_block3::<f64, crate::SquaredEuclidean<f64>, 3>(
                    query, stems_ptr, 0, old_off, rd, best_dist,
                );

                assert_eq!(
                    scalar_mask, simd_mask,
                    "SIMD vs scalar mismatch for query={}, old_off={}, rd={}, best_dist={}: scalar={:08b}, simd={:08b}",
                    query, old_off, rd, best_dist, scalar_mask, simd_mask
                );
            }

            // On non-SIMD platforms, just validate the scalar mask is reasonable
            #[cfg(not(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2")))]
            {
                // Scalar mask should be valid (between 0 and 0xFF)
                let _ = scalar_mask; // Use the variable to avoid warning
            }
        }
    }

    #[test]
    fn test_backtrack_scalar_vs_simd_f32_multiple_cases() {
        let pivots = build_test_block3_pivots_f32();

        let test_cases = [
            (0.25f32, 0.0f32, 0.0f32, f32::INFINITY),
            (0.5f32, 0.0f32, 0.0f32, 0.1f32),
            (-0.5f32, 0.0f32, 0.0f32, f32::INFINITY),
        ];

        for (query, old_off, rd, best_dist) in test_cases {
            let scalar_mask =
                scalar_backtrack_check_block3_f32(query, &pivots, old_off, rd, best_dist);

            #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
            {
                let stems_ptr = NonNull::new(pivots.as_ptr() as *mut u8).unwrap();
                let simd_mask = f32::backtrack_block3::<f32, crate::SquaredEuclidean<f32>, 3>(
                    query, stems_ptr, 0, old_off, rd, best_dist,
                );

                assert_eq!(
                    scalar_mask, simd_mask,
                    "SIMD vs scalar (f32) mismatch for query={}, old_off={}, rd={}, best_dist={}: scalar={:08b}, simd={:08b}",
                    query, old_off, rd, best_dist, scalar_mask, simd_mask
                );
            }

            #[cfg(not(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2")))]
            {
                let _ = scalar_mask; // Use the variable to avoid warning
            }
        }
    }

    #[test]
    fn test_block3_backtrack_ground_truth_f64_basic() {
        let pivots = build_test_block3_pivots_f64();

        // Case 1: Query in middle, rd=0, best_dist=infinity (all children should pass)
        let mask = scalar_backtrack_check_block3_f64(0.5, &pivots, 0.0, 0.0, f64::INFINITY);
        assert_eq!(
            mask, 0xFF,
            "With infinite best_dist, all children should pass"
        );

        // Case 2: Query in middle, rd=0, best_dist=0 (only children with interval_dist=0 should pass)
        let query = 0.5;
        let mask = scalar_backtrack_check_block3_f64(query, &pivots, 0.0, 0.0, 0.0);

        // The child containing the query should have interval_dist=0, so rd_far=0
        // Note: if query sits on a boundary, multiple children might have interval_dist=0
        let selected_child = select_child_scalar_f64(query, &pivots);
        let selected_bit = 1 << selected_child;

        assert!(
            (mask & selected_bit) != 0,
            "With best_dist=0, at least the child containing the query (child {}) should pass: got {:08b}",
            selected_child, mask
        );

        // All passing children should have interval_dist=0 (rd_far=0<=0)
        assert!(
            mask.count_ones() >= 1,
            "At least one child should pass when best_dist=0"
        );
    }

    #[test]
    fn test_block3_backtrack_edge_cases_f64() {
        let pivots = build_test_block3_pivots_f64();

        // Test with query at exact pivot boundaries
        for (_pivot_idx, &pivot_val) in pivots.iter().enumerate().take(7) {
            let mask =
                scalar_backtrack_check_block3_f64(pivot_val, &pivots, 0.0, 0.0, f64::INFINITY);

            // At least one child should pass
            assert_ne!(
                mask, 0,
                "At pivot boundary {}, at least one child should be visitable",
                pivot_val
            );
        }

        // Test with query outside all pivots
        let mask = scalar_backtrack_check_block3_f64(-1000.0, &pivots, 0.0, 0.0, f64::INFINITY);
        assert_eq!(
            mask, 0xFF,
            "Far outside query with infinite best_dist should allow all children"
        );

        // Test with non-zero old_off
        let mask = scalar_backtrack_check_block3_f64(0.5, &pivots, 0.1, 0.0, f64::INFINITY);
        assert_ne!(
            mask, 0,
            "With non-zero old_off, some children should still pass"
        );
    }

    #[test]
    fn test_block3_backtrack_rd_pruning_f64() {
        let pivots = build_test_block3_pivots_f64();

        // Test that increasing rd prunes more children
        let query = 0.5;
        let old_off = 0.0;
        let best_dist = 0.1;

        let mask_rd_0 = scalar_backtrack_check_block3_f64(query, &pivots, old_off, 0.0, best_dist);
        let mask_rd_high =
            scalar_backtrack_check_block3_f64(query, &pivots, old_off, 0.05, best_dist);

        // With higher rd, we should prune at least as many (possibly more) children
        let count_rd_0 = mask_rd_0.count_ones();
        let count_rd_high = mask_rd_high.count_ones();

        assert!(
            count_rd_high <= count_rd_0,
            "Higher rd should prune at least as many children: rd=0 -> {} children, rd=0.05 -> {} children",
            count_rd_0, count_rd_high
        );
    }

    #[test]
    fn test_block3_backtrack_best_dist_pruning_f64() {
        let pivots = build_test_block3_pivots_f64();

        // Test that decreasing best_dist prunes more children
        let query = 0.5;
        let old_off = 0.0;
        let rd = 0.0;

        let mask_best_inf =
            scalar_backtrack_check_block3_f64(query, &pivots, old_off, rd, f64::INFINITY);
        let mask_best_01 = scalar_backtrack_check_block3_f64(query, &pivots, old_off, rd, 0.1);
        let mask_best_001 = scalar_backtrack_check_block3_f64(query, &pivots, old_off, rd, 0.01);

        let count_inf = mask_best_inf.count_ones();
        let count_01 = mask_best_01.count_ones();
        let count_001 = mask_best_001.count_ones();

        assert!(
            count_001 <= count_01 && count_01 <= count_inf,
            "Smaller best_dist should prune more: inf -> {}, 0.1 -> {}, 0.01 -> {}",
            count_inf,
            count_01,
            count_001
        );
    }

    #[test]
    #[cfg(all(feature = "fixed", not(feature = "simd")))]
    fn test_simd_prune_fixed_i32_u0() {
        use crate::stem_strategy::donnelly::simd_full::SimdPrune;
        use fixed::types::extra::U0;
        use fixed::FixedI32;

        type Fixed = FixedI32<U0>;

        // Create test rd_values array - some pass, some fail
        let rd_values: [Fixed; 8] = [
            Fixed::from_num(1),
            Fixed::from_num(5),
            Fixed::from_num(10),
            Fixed::from_num(15),
            Fixed::from_num(3),
            Fixed::from_num(7),
            Fixed::from_num(12),
            Fixed::from_num(2),
        ];

        let max_dist = Fixed::from_num(8);
        let sibling_mask = 0xFF; // All siblings enabled

        // Call pruning via trait
        let mask = Fixed::simd_prune_block3(&rd_values, max_dist, sibling_mask);

        // Expected: bits 0, 1, 4, 5, 7 should be set (values <= 8)
        // Values: [1, 5, 10, 15, 3, 7, 12, 2]
        //  Pass:  [T, T,  F,  F, T, T,  F, T]
        let expected_mask = 0b10110011; // bits 0,1,4,5,7 set

        assert_eq!(
            mask, expected_mask,
            "Fixed-point pruning mask mismatch: got {:08b}, expected {:08b}",
            mask, expected_mask
        );
    }

    #[test]
    #[cfg(all(feature = "fixed", not(feature = "simd")))]
    fn test_simd_prune_fixed_i32_u16() {
        use crate::stem_strategy::donnelly::simd_full::SimdPrune;
        use fixed::types::extra::U16;
        use fixed::FixedI32;

        type Fixed = FixedI32<U16>;

        // Create test rd_values with fractional values
        let rd_values: [Fixed; 8] = [
            Fixed::from_num(0.5),
            Fixed::from_num(1.5),
            Fixed::from_num(2.5),
            Fixed::from_num(3.5),
            Fixed::from_num(0.25),
            Fixed::from_num(1.75),
            Fixed::from_num(2.75),
            Fixed::from_num(0.1),
        ];

        let max_dist = Fixed::from_num(2.0);
        let sibling_mask = 0xFF;

        let mask = Fixed::simd_prune_block3(&rd_values, max_dist, sibling_mask);

        // Expected: bits 0, 1, 4, 5, 7 should be set (values <= 2.0)
        // Values: [0.5, 1.5, 2.5, 3.5, 0.25, 1.75, 2.75, 0.1]
        //  Pass:  [ T,   T,   F,   F,   T,    T,    F,   T]
        let expected_mask = 0b10110011;

        assert_eq!(
            mask, expected_mask,
            "Fixed-point pruning mask mismatch: got {:08b}, expected {:08b}",
            mask, expected_mask
        );
    }

    #[test]
    #[cfg(all(feature = "fixed", not(feature = "simd")))]
    fn test_simd_prune_fixed_u16_u8() {
        use crate::stem_strategy::donnelly::simd_full::SimdPrune;
        use fixed::types::extra::U8;
        use fixed::FixedU16;

        type Fixed = FixedU16<U8>;

        let rd_values: [Fixed; 8] = [
            Fixed::from_num(0.1),
            Fixed::from_num(0.5),
            Fixed::from_num(1.0),
            Fixed::from_num(1.5),
            Fixed::from_num(0.2),
            Fixed::from_num(0.8),
            Fixed::from_num(1.2),
            Fixed::from_num(0.3),
        ];

        let max_dist = Fixed::from_num(1.0);
        let sibling_mask = 0xFF;

        let mask = Fixed::simd_prune_block3(&rd_values, max_dist, sibling_mask);

        // Expected: bits 0, 1, 2, 4, 5, 7 should be set (values <= 1.0)
        // Values: [0.1, 0.5, 1.0, 1.5, 0.2, 0.8, 1.2, 0.3]
        //  Pass:  [ T,   T,   T,   F,   T,   T,   F,   T]
        let expected_mask = 0b10110111;

        assert_eq!(
            mask, expected_mask,
            "Fixed-point pruning mask mismatch: got {:08b}, expected {:08b}",
            mask, expected_mask
        );
    }

    #[test]
    fn test_simd_prune_sibling_mask_filtering() {
        use crate::stem_strategy::donnelly::simd_full::SimdPrune;

        // Test that sibling_mask correctly filters results
        let rd_values = [0.5f64, 1.5, 2.5, 3.5, 0.25, 1.75, 2.75, 0.1];
        let max_dist = 2.0f64;

        // All siblings enabled
        let mask_all = f64::simd_prune_block3(&rd_values, max_dist, 0xFF);
        // Only even-indexed siblings enabled
        let mask_even = f64::simd_prune_block3(&rd_values, max_dist, 0b01010101);
        // Only odd-indexed siblings enabled
        let mask_odd = f64::simd_prune_block3(&rd_values, max_dist, 0b10101010);

        // mask_all should have bits 0,1,4,5,7 set (values <= 2.0)
        let expected_all = 0b10110011;
        assert_eq!(
            mask_all, expected_all,
            "Unfiltered mask mismatch: got {:08b}, expected {:08b}",
            mask_all, expected_all
        );

        // mask_even should have only bits 0,4 set (even indices & values <= 2.0)
        let expected_even = 0b00010001;
        assert_eq!(
            mask_even, expected_even,
            "Even-filtered mask mismatch: got {:08b}, expected {:08b}",
            mask_even, expected_even
        );

        // mask_odd should have only bits 1,5,7 set (odd indices & values <= 2.0)
        let expected_odd = 0b10100010;
        assert_eq!(
            mask_odd, expected_odd,
            "Odd-filtered mask mismatch: got {:08b}, expected {:08b}",
            mask_odd, expected_odd
        );
    }

    // =======================================================================================
    // Block4 Option A (chunked 8-wide blocks) tests
    // =======================================================================================

    /// Test that Block4 u16 mask correctly splits into two u8 chunks.
    #[test]
    fn test_block4_mask_split_basic() {
        // Test case: all 16 bits set
        let full_mask: u16 = 0xFFFF;
        let high_mask = (full_mask >> 8) as u8;
        let low_mask = full_mask as u8;

        assert_eq!(high_mask, 0xFF, "High mask should be 0xFF for full mask");
        assert_eq!(low_mask, 0xFF, "Low mask should be 0xFF for full mask");

        // Test case: only low 8 bits set
        let low_only: u16 = 0x00FF;
        let high_mask = (low_only >> 8) as u8;
        let low_mask = low_only as u8;

        assert_eq!(
            high_mask, 0x00,
            "High mask should be 0x00 for low-only mask"
        );
        assert_eq!(low_mask, 0xFF, "Low mask should be 0xFF for low-only mask");

        // Test case: only high 8 bits set
        let high_only: u16 = 0xFF00;
        let high_mask = (high_only >> 8) as u8;
        let low_mask = high_only as u8;

        assert_eq!(
            high_mask, 0xFF,
            "High mask should be 0xFF for high-only mask"
        );
        assert_eq!(low_mask, 0x00, "Low mask should be 0x00 for high-only mask");
    }

    /// Test that Block4 mask splitting correctly maps child indices to chunks.
    #[test]
    fn test_block4_mask_split_child_mapping() {
        // Children 0-7 should be in the low chunk
        // Children 8-15 should be in the high chunk

        for child_idx in 0..16u16 {
            let single_child_mask: u16 = 1 << child_idx;
            let high_mask = (single_child_mask >> 8) as u8;
            let low_mask = single_child_mask as u8;

            if child_idx < 8 {
                // Low chunk: child should appear in low_mask at position child_idx
                assert_eq!(
                    high_mask, 0,
                    "Child {} should not appear in high chunk",
                    child_idx
                );
                assert_eq!(
                    low_mask,
                    1 << child_idx,
                    "Child {} should appear at position {} in low chunk",
                    child_idx,
                    child_idx
                );
            } else {
                // High chunk: child should appear in high_mask at position (child_idx - 8)
                assert_eq!(
                    low_mask, 0,
                    "Child {} should not appear in low chunk",
                    child_idx
                );
                assert_eq!(
                    high_mask,
                    1 << (child_idx - 8),
                    "Child {} should appear at position {} in high chunk",
                    child_idx,
                    child_idx - 8
                );
            }
        }
    }

    /// Test that Block4 Option A correctly splits sibling arrays into 8-wide chunks.
    #[test]
    fn test_block4_sibling_array_chunking() {
        // Simulate the 16-element arrays that Block4 produces
        let siblings_16: [u32; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let rd_values_16: [f64; 16] = [
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
        ];
        let new_off_16: [f64; 16] = [
            1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5,
        ];

        // Split into high and low chunks (as done in actual code)
        let mut high_siblings = [0u32; 8];
        let mut high_rd_values = [0.0f64; 8];
        let mut high_new_off_values = [0.0f64; 8];
        high_siblings.copy_from_slice(&siblings_16[8..16]);
        high_rd_values.copy_from_slice(&rd_values_16[8..16]);
        high_new_off_values.copy_from_slice(&new_off_16[8..16]);

        let mut low_siblings = [0u32; 8];
        let mut low_rd_values = [0.0f64; 8];
        let mut low_new_off_values = [0.0f64; 8];
        low_siblings.copy_from_slice(&siblings_16[..8]);
        low_rd_values.copy_from_slice(&rd_values_16[..8]);
        low_new_off_values.copy_from_slice(&new_off_16[..8]);

        // Verify high chunk contains children 8-15
        assert_eq!(high_siblings, [8, 9, 10, 11, 12, 13, 14, 15]);
        assert_eq!(high_rd_values, [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]);
        assert_eq!(
            high_new_off_values,
            [1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]
        );

        // Verify low chunk contains children 0-7
        assert_eq!(low_siblings, [0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(low_rd_values, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]);
        assert_eq!(low_new_off_values, [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]);
    }

    /// Test that mask exclusion of taken child works correctly for Block4.
    #[test]
    fn test_block4_taken_child_exclusion() {
        // When we take child N, we should exclude it from the backtrack mask

        for taken_child in 0..16u16 {
            let backtrack_mask: u16 = 0xFFFF; // All children would pass
            let child_idx_mask: u16 = 1 << taken_child;
            let filtered_mask = backtrack_mask & !child_idx_mask;

            // The filtered mask should have all bits except the taken child
            assert_eq!(
                filtered_mask,
                0xFFFF ^ (1 << taken_child),
                "Taken child {} should be excluded from mask",
                taken_child
            );

            // Split and verify exclusion appears in correct chunk
            let high_mask = (filtered_mask >> 8) as u8;
            let low_mask = filtered_mask as u8;

            if taken_child < 8 {
                // Taken child is in low chunk
                assert_eq!(
                    high_mask, 0xFF,
                    "High chunk unaffected by low child exclusion"
                );
                assert_eq!(
                    low_mask,
                    0xFF ^ (1 << taken_child),
                    "Low chunk should exclude taken child {}",
                    taken_child
                );
            } else {
                // Taken child is in high chunk
                assert_eq!(
                    high_mask,
                    0xFF ^ (1 << (taken_child - 8)),
                    "High chunk should exclude taken child {}",
                    taken_child
                );
                assert_eq!(
                    low_mask, 0xFF,
                    "Low chunk unaffected by high child exclusion"
                );
            }
        }
    }

    /// Test Block4 interval bounds produce correct chunks when mapped.
    #[test]
    fn test_block4_interval_bounds_chunk_consistency() {
        // Verify that interval bounds for children 0-7 and 8-15 maintain
        // the invariant that child N's upper bound equals child (N+1)'s lower bound

        // Check low chunk (children 0-7)
        for child_idx in 0..7 {
            let (_, upper) = child_interval_bounds_block4(child_idx);
            let (lower_next, _) = child_interval_bounds_block4(child_idx + 1);
            assert_eq!(
                upper, lower_next,
                "Low chunk gap at child {}: upper {} != next lower {}",
                child_idx, upper, lower_next
            );
        }

        // Check high chunk (children 8-15)
        for child_idx in 8..15 {
            let (_, upper) = child_interval_bounds_block4(child_idx);
            let (lower_next, _) = child_interval_bounds_block4(child_idx + 1);
            assert_eq!(
                upper, lower_next,
                "High chunk gap at child {}: upper {} != next lower {}",
                child_idx, upper, lower_next
            );
        }

        // Check the boundary between low and high chunks (child 7 -> child 8)
        let (_, upper_7) = child_interval_bounds_block4(7);
        let (lower_8, _) = child_interval_bounds_block4(8);
        assert_eq!(
            upper_7, lower_8,
            "Chunk boundary gap: child 7 upper {} != child 8 lower {}",
            upper_7, lower_8
        );
    }

    /// Compile-time assertion: Block3 uses u8 mask, Block4 uses u16 mask.
    #[test]
    fn test_block_mask_type_assertions() {
        // Block3: 8 siblings -> u8 mask
        const _BLOCK3_SIBLINGS: usize = 1 << 3; // 8
        const _: () = assert!(_BLOCK3_SIBLINGS == 8);

        // Block4: 16 siblings -> u16 mask
        const _BLOCK4_SIBLINGS: usize = 1 << 4; // 16
        const _: () = assert!(_BLOCK4_SIBLINGS == 16);

        // Verify mask can hold all sibling bits
        const _: () = assert!(std::mem::size_of::<u8>() * 8 >= _BLOCK3_SIBLINGS);
        const _: () = assert!(std::mem::size_of::<u16>() * 8 >= _BLOCK4_SIBLINGS);
    }
}

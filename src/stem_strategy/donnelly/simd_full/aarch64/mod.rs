//! ARM aarch64 SIMD implementations (NEON) for Donnelly block marker strategy.

use std::ptr::NonNull;

/// Compare query value against 7 pivots in a Block3 (f64, NEON)
///
/// Returns the child index (0-7) based on how many pivots the query exceeds.
///
/// # Safety
/// - `stems_ptr` must point to at least 8 contiguous f64 values starting at
///   `cache_line_base * 8`.
/// - The caller must ensure the pointer is valid for reads and properly aligned
///   for `f64` loads on the target.
#[inline(always)]
pub unsafe fn compare_block3_f64_neon(
    stems_ptr: NonNull<u8>,
    cache_line_base: usize,
    query_val: f64,
) -> u8 {
    use core::arch::aarch64::*;

    let ptr = stems_ptr.as_ptr().add(cache_line_base * 8) as *const f64;
    let query_vec = vdupq_n_f64(query_val);

    let pivots_0 = vld1q_f64(ptr);
    let pivots_1 = vld1q_f64(ptr.add(2));
    let pivots_2 = vld1q_f64(ptr.add(4));
    let pivots_3 = vld1q_f64(ptr.add(6));

    let cmp_0 = vcgeq_f64(query_vec, pivots_0);
    let cmp_1 = vcgeq_f64(query_vec, pivots_1);
    let cmp_2 = vcgeq_f64(query_vec, pivots_2);
    let cmp_3 = vcgeq_f64(query_vec, pivots_3);

    let ones_0 = vshrq_n_u64(cmp_0, 63);
    let ones_1 = vshrq_n_u64(cmp_1, 63);
    let ones_2 = vshrq_n_u64(cmp_2, 63);
    let ones_3 = vshrq_n_u64(cmp_3, 63);

    let count_0 = vaddvq_u64(ones_0);
    let count_1 = vaddvq_u64(ones_1);
    let count_2 = vaddvq_u64(ones_2);
    // Lane 7 is padding in Block3 and may contain an embedded item summary.
    let count_3 = vgetq_lane_u64(ones_3, 0);

    (count_0 + count_1 + count_2 + count_3) as u8
}

/// Compare query value against 7 pivots in a Block3 (f32, NEON)
///
/// Returns the child index (0-7) based on how many pivots the query exceeds.
///
/// # Safety
/// - `stems_ptr` must point to at least 8 contiguous f32 values starting at
///   `cache_line_base * 4`.
/// - The caller must ensure the pointer is valid for reads and properly aligned
///   for `f32` loads on the target.
#[inline(always)]
pub unsafe fn compare_block3_f32_neon(
    stems_ptr: NonNull<u8>,
    cache_line_base: usize,
    query_val: f32,
) -> u8 {
    use core::arch::aarch64::*;

    let ptr = stems_ptr.as_ptr().add(cache_line_base * 4) as *const f32;
    let query_vec = vdupq_n_f32(query_val);

    let pivots_0 = vld1q_f32(ptr);
    let pivots_1 = vld1q_f32(ptr.add(4));

    let cmp_0 = vcgeq_f32(query_vec, pivots_0);
    let cmp_1 = vcgeq_f32(query_vec, pivots_1);

    let ones_0 = vshrq_n_u32(cmp_0, 31);
    let ones_1 = vshrq_n_u32(cmp_1, 31);

    let count_0 = vaddvq_u32(ones_0);
    // Lane 7 is padding in Block3 and may contain an embedded item summary.
    let count_1 = vaddvq_u32(ones_1) - vgetq_lane_u32(ones_1, 3);

    (count_0 + count_1) as u8
}

/// Compare query value against 15 pivots in a Block4 (f32, NEON)
///
/// Returns the child index (0-15) based on how many pivots the query exceeds.
///
/// # Safety
/// - `stems_ptr` must point to at least 16 contiguous f32 values starting at
///   `cache_line_base * 4`.
/// - The caller must ensure the pointer is valid for reads and properly aligned
///   for `f32` loads on the target.
#[inline(always)]
pub unsafe fn compare_block4_f32_neon(
    stems_ptr: NonNull<u8>,
    cache_line_base: usize,
    query_val: f32,
) -> u8 {
    use core::arch::aarch64::*;

    let ptr = stems_ptr.as_ptr().add(cache_line_base * 4) as *const f32;
    let query_vec = vdupq_n_f32(query_val);

    let pivots_0 = vld1q_f32(ptr);
    let pivots_1 = vld1q_f32(ptr.add(4));
    let pivots_2 = vld1q_f32(ptr.add(8));
    let pivots_3 = vld1q_f32(ptr.add(12));

    let cmp_0 = vcgeq_f32(query_vec, pivots_0);
    let cmp_1 = vcgeq_f32(query_vec, pivots_1);
    let cmp_2 = vcgeq_f32(query_vec, pivots_2);
    let cmp_3 = vcgeq_f32(query_vec, pivots_3);

    let ones_0 = vshrq_n_u32(cmp_0, 31);
    let ones_1 = vshrq_n_u32(cmp_1, 31);
    let ones_2 = vshrq_n_u32(cmp_2, 31);
    let ones_3 = vshrq_n_u32(cmp_3, 31);

    let count_0 = vaddvq_u32(ones_0);
    let count_1 = vaddvq_u32(ones_1);
    let count_2 = vaddvq_u32(ones_2);
    let count_3 = vaddvq_u32(ones_3);

    (count_0 + count_1 + count_2 + count_3) as u8
}

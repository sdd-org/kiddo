//! Traits for SIMD-accelerated pruning in query orchestration.
//!
//! This module defines the `SimdPrune` trait which provides type-specific
//! implementations for comparing distance values during backtracking traversal.

use crate::Axis;

mod sealed {
    pub trait Sealed {}
}

/// Trait for SIMD-accelerated pruning operations during query backtracking.
///
/// This trait is sealed and only implemented for types that have explicit
/// SIMD or autovec implementations.
///
/// The trait provides block-level pruning for Block3 (8 children).
/// TODO: block-level pruning for Block4 & 5
pub trait SimdPrune: Axis<Coord = Self> + sealed::Sealed {
    /// Compare 8 rd_values against max_dist and return a bitmask.
    ///
    /// Returns a u8 bitmask where bit i is set if rd_values\[i\] <= max_dist.
    /// The result is ANDed with sibling_mask to exclude siblings that are
    /// already pruned by other criteria.
    ///
    /// # Arguments
    /// * `rd_values` - Array of 8 distance values to compare
    /// * `max_dist` - Maximum distance threshold
    /// * `sibling_mask` - Pre-computed mask of valid siblings
    fn simd_prune_block3(rd_values: &[Self; 8], max_dist: Self, sibling_mask: u8) -> u8;

    /// Compare 16 Block4 child distances against `max_dist`.
    #[inline(always)]
    fn simd_prune_block4(rd_values: &[Self; 16], max_dist: Self, sibling_mask: u16) -> u16 {
        let mut mask = 0u16;
        let mut lane = 0usize;
        while lane < 16 {
            if rd_values[lane] <= max_dist {
                mask |= 1u16 << lane;
            }
            lane += 1;
        }
        mask & sibling_mask
    }

    /// Select the live Block4 child with the smallest rectangle distance.
    #[inline(always)]
    fn simd_select_best_child_block4(rd_values: &[Self; 16], candidate_mask: u16) -> Option<u8> {
        if candidate_mask == 0 {
            return None;
        }

        let mut remaining = candidate_mask;
        let first = remaining.trailing_zeros() as usize;
        let mut best_idx = first;
        let mut best_rd = rd_values[first];
        remaining &= remaining - 1;

        while remaining != 0 {
            let idx = remaining.trailing_zeros() as usize;
            if Self::cmp(rd_values[idx], best_rd) == std::cmp::Ordering::Less {
                best_idx = idx;
                best_rd = rd_values[idx];
            }
            remaining &= remaining - 1;
        }

        Some(best_idx as u8)
    }
}

/// Select the child with the smallest lower-bound distance among the live Block3 lanes.
///
/// `candidate_mask` marks the currently viable children. Returns `None` if no children remain.
pub trait SimdSelectBestChildBlock3: Axis<Coord = Self> + sealed::Sealed {
    /// Returns the lowest-index live child whose lower-bound distance is minimal.
    fn simd_select_best_child_block3(rd_values: &[Self; 8], candidate_mask: u8) -> Option<u8>;
}

#[inline(always)]
fn scalar_select_best_child_block3<O>(rd_values: &[O; 8], candidate_mask: u8) -> Option<u8>
where
    O: Axis<Coord = O>,
{
    if candidate_mask == 0 {
        return None;
    }

    let mut remaining = candidate_mask;
    let first = remaining.trailing_zeros() as usize;
    let mut best_idx = first;
    let mut best_rd = rd_values[first];
    remaining &= remaining - 1;

    while remaining != 0 {
        let idx = remaining.trailing_zeros() as usize;
        if O::cmp(rd_values[idx], best_rd) == std::cmp::Ordering::Less {
            best_idx = idx;
            best_rd = rd_values[idx];
        }
        remaining &= remaining - 1;
    }

    Some(best_idx as u8)
}

/// Macro to generate the autovec fallback implementation.
/// This is the same for all types - a simple loop with comparisons.
///
/// # Parameters
/// - `$width`: Block width (8 for Block3, 16 for Block4, 32 for Block5)
/// - `$mask_ty`: Mask type (u8 for Block3, u16 for Block4, u32 for Block5)
/// - `$rd_values`: Array of distance values
/// - `$max_dist`: Maximum distance threshold
/// - `$sibling_mask`: Pre-computed mask of valid siblings
#[allow(unused_macros)]
macro_rules! autovec_fallback {
    ($width:expr, $mask_ty:ty, $rd_values:expr, $max_dist:expr, $sibling_mask:expr) => {{
        let mut mask: $mask_ty = 0;
        for i in 0..$width {
            if $rd_values[i] <= $max_dist {
                mask |= 1 << i;
            }
        }
        mask & $sibling_mask
    }};
}

impl sealed::Sealed for f64 {}
impl SimdPrune for f64 {
    #[inline(always)]
    fn simd_prune_block3(rd_values: &[f64; 8], max_dist: f64, sibling_mask: u8) -> u8 {
        #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
        {
            unsafe {
                use std::arch::x86_64::*;
                let max_dist_vec = _mm256_set1_pd(max_dist);
                let rd_low = _mm256_loadu_pd(rd_values.as_ptr());
                let rd_high = _mm256_loadu_pd(rd_values.as_ptr().add(4));

                let cmp_low = _mm256_cmp_pd(rd_low, max_dist_vec, _CMP_LE_OQ);
                let cmp_high = _mm256_cmp_pd(rd_high, max_dist_vec, _CMP_LE_OQ);

                let mask_low = _mm256_movemask_pd(cmp_low) as u8;
                let mask_high = _mm256_movemask_pd(cmp_high) as u8;

                let mask = mask_low | (mask_high << 4);
                mask & sibling_mask
            }
        }

        #[cfg(all(
            feature = "simd",
            target_arch = "aarch64",
            not(all(target_arch = "x86_64", target_feature = "avx2"))
        ))]
        {
            unsafe {
                use core::arch::aarch64::*;
                let max_vec = vdupq_n_f64(max_dist);
                let rd_0 = vld1q_f64(rd_values.as_ptr());
                let rd_1 = vld1q_f64(rd_values.as_ptr().add(2));
                let rd_2 = vld1q_f64(rd_values.as_ptr().add(4));
                let rd_3 = vld1q_f64(rd_values.as_ptr().add(6));

                let cmp_0 = vcleq_f64(rd_0, max_vec);
                let cmp_1 = vcleq_f64(rd_1, max_vec);
                let cmp_2 = vcleq_f64(rd_2, max_vec);
                let cmp_3 = vcleq_f64(rd_3, max_vec);

                let weights_0 = [1u64, 2u64];
                let weights_1 = [4u64, 8u64];
                let weights_2 = [16u64, 32u64];
                let weights_3 = [64u64, 128u64];

                let mask_0 = vaddvq_u64(vandq_u64(cmp_0, vld1q_u64(weights_0.as_ptr())));
                let mask_1 = vaddvq_u64(vandq_u64(cmp_1, vld1q_u64(weights_1.as_ptr())));
                let mask_2 = vaddvq_u64(vandq_u64(cmp_2, vld1q_u64(weights_2.as_ptr())));
                let mask_3 = vaddvq_u64(vandq_u64(cmp_3, vld1q_u64(weights_3.as_ptr())));

                let mask = (mask_0 | mask_1 | mask_2 | mask_3) as u8;
                mask & sibling_mask
            }
        }

        #[cfg(not(any(
            all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"),
            all(feature = "simd", target_arch = "aarch64")
        )))]
        {
            autovec_fallback!(8, u8, rd_values, max_dist, sibling_mask)
        }
    }

    #[inline(always)]
    fn simd_prune_block4(rd_values: &[f64; 16], max_dist: f64, sibling_mask: u16) -> u16 {
        #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            use std::arch::x86_64::*;

            let max_dist = _mm512_set1_pd(max_dist);
            let low = _mm512_loadu_pd(rd_values.as_ptr());
            let high = _mm512_loadu_pd(rd_values.as_ptr().add(8));
            let low_mask = _mm512_cmp_pd_mask(low, max_dist, _CMP_LE_OQ) as u16;
            let high_mask = _mm512_cmp_pd_mask(high, max_dist, _CMP_LE_OQ) as u16;
            (low_mask | (high_mask << 8)) & sibling_mask
        }

        #[cfg(not(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f")))]
        {
            let mut mask = 0u16;
            for (lane, &rd) in rd_values.iter().enumerate() {
                if rd <= max_dist {
                    mask |= 1u16 << lane;
                }
            }
            mask & sibling_mask
        }
    }
}

impl SimdSelectBestChildBlock3 for f64 {
    #[inline(always)]
    fn simd_select_best_child_block3(rd_values: &[f64; 8], candidate_mask: u8) -> Option<u8> {
        #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
        {
            unsafe {
                use std::arch::x86_64::*;

                if candidate_mask == 0 {
                    return None;
                }

                let rd_vec = _mm512_loadu_pd(rd_values.as_ptr());
                let masked =
                    _mm512_mask_mov_pd(_mm512_set1_pd(f64::INFINITY), candidate_mask, rd_vec);
                let min_val = _mm512_reduce_min_pd(masked);
                let eq_mask = _mm512_cmp_pd_mask(masked, _mm512_set1_pd(min_val), _CMP_EQ_OQ)
                    & candidate_mask;

                Some(eq_mask.trailing_zeros() as u8)
            }
        }

        #[cfg(not(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f")))]
        {
            scalar_select_best_child_block3(rd_values, candidate_mask)
        }
    }
}

impl sealed::Sealed for f32 {}
impl SimdPrune for f32 {
    #[inline(always)]
    fn simd_prune_block3(rd_values: &[f32; 8], max_dist: f32, sibling_mask: u8) -> u8 {
        #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
        {
            unsafe {
                use std::arch::x86_64::*;
                let max_dist_vec = _mm256_set1_ps(max_dist);
                let rd_vec = _mm256_loadu_ps(rd_values.as_ptr());

                let cmp = _mm256_cmp_ps(rd_vec, max_dist_vec, _CMP_LE_OQ);
                let mask = _mm256_movemask_ps(cmp) as u8;

                mask & sibling_mask
            }
        }

        #[cfg(all(
            feature = "simd",
            target_arch = "aarch64",
            not(all(target_arch = "x86_64", target_feature = "avx2"))
        ))]
        {
            unsafe {
                use core::arch::aarch64::*;
                let max_vec = vdupq_n_f32(max_dist);
                let rd_0 = vld1q_f32(rd_values.as_ptr());
                let rd_1 = vld1q_f32(rd_values.as_ptr().add(4));

                let cmp_0 = vcleq_f32(rd_0, max_vec);
                let cmp_1 = vcleq_f32(rd_1, max_vec);

                let weights_0 = [1u32, 2u32, 4u32, 8u32];
                let weights_1 = [16u32, 32u32, 64u32, 128u32];

                let mask_0 = vaddvq_u32(vandq_u32(cmp_0, vld1q_u32(weights_0.as_ptr())));
                let mask_1 = vaddvq_u32(vandq_u32(cmp_1, vld1q_u32(weights_1.as_ptr())));

                let mask = (mask_0 | mask_1) as u8;
                mask & sibling_mask
            }
        }

        #[cfg(not(any(
            all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"),
            all(feature = "simd", target_arch = "aarch64")
        )))]
        {
            autovec_fallback!(8, u8, rd_values, max_dist, sibling_mask)
        }
    }

    #[inline(always)]
    fn simd_prune_block4(rd_values: &[f32; 16], max_dist: f32, sibling_mask: u16) -> u16 {
        #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            use std::arch::x86_64::*;

            let rd = _mm512_loadu_ps(rd_values.as_ptr());
            let max_dist = _mm512_set1_ps(max_dist);
            _mm512_cmp_ps_mask(rd, max_dist, _CMP_LE_OQ) & sibling_mask
        }

        #[cfg(not(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f")))]
        {
            let mut mask = 0u16;
            for (lane, &rd) in rd_values.iter().enumerate() {
                if rd <= max_dist {
                    mask |= 1u16 << lane;
                }
            }
            mask & sibling_mask
        }
    }

    #[inline(always)]
    fn simd_select_best_child_block4(rd_values: &[f32; 16], candidate_mask: u16) -> Option<u8> {
        #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            use std::arch::x86_64::*;

            if candidate_mask == 0 {
                return None;
            }

            let rd = _mm512_loadu_ps(rd_values.as_ptr());
            let masked = _mm512_mask_mov_ps(_mm512_set1_ps(f32::INFINITY), candidate_mask, rd);
            let min = _mm512_reduce_min_ps(masked);
            let minima = _mm512_cmp_ps_mask(masked, _mm512_set1_ps(min), _CMP_EQ_OQ);
            Some((minima & candidate_mask).trailing_zeros() as u8)
        }

        #[cfg(not(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f")))]
        {
            if candidate_mask == 0 {
                return None;
            }

            let mut remaining = candidate_mask;
            let mut best_idx = remaining.trailing_zeros() as usize;
            let mut best_rd = rd_values[best_idx];
            remaining &= remaining - 1;
            while remaining != 0 {
                let idx = remaining.trailing_zeros() as usize;
                if rd_values[idx] < best_rd {
                    best_idx = idx;
                    best_rd = rd_values[idx];
                }
                remaining &= remaining - 1;
            }
            Some(best_idx as u8)
        }
    }
}

impl SimdSelectBestChildBlock3 for f32 {
    #[inline(always)]
    fn simd_select_best_child_block3(rd_values: &[f32; 8], candidate_mask: u8) -> Option<u8> {
        scalar_select_best_child_block3(rd_values, candidate_mask)
    }
}

#[cfg(feature = "fixed")]
mod fixed_impls {
    use super::*;

    /// Macro to implement SimdPrune for FixedI32 with specific fractional bit count.
    ///
    /// Note: While all FixedI32 variants share the same 32-bit integer representation,
    /// multiplication operations (if used in distance calculations) may differ based on
    /// fractional bits. The comparison operation itself (<=) is the same for all variants.
    #[allow(unused_macros)]
    macro_rules! impl_simd_prune_fixed_i32 {
        ($frac:ty) => {
            impl sealed::Sealed for fixed::FixedI32<$frac> {}

            impl SimdPrune for fixed::FixedI32<$frac> {
                #[inline(always)]
                fn simd_prune_block3(
                    rd_values: &[fixed::FixedI32<$frac>; 8],
                    max_dist: fixed::FixedI32<$frac>,
                    sibling_mask: u8,
                ) -> u8 {
                    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
                    {
                        // TODO: i32 SIMD implementation for FixedI32<$frac>
                        // Use _mm256_cmpgt_epi32 for signed comparison
                        // Note: This macro is parameterized by $frac in case multiply ops
                        // need fractional-bit-specific handling
                        let _ = (rd_values, max_dist, sibling_mask);
                        todo!(
                            "SIMD implementation for FixedI32<{}> not yet implemented",
                            stringify!($frac)
                        )
                    }

                    #[cfg(all(
                        feature = "simd",
                        target_arch = "aarch64",
                        not(all(target_arch = "x86_64", target_feature = "avx2"))
                    ))]
                    {
                        // TODO: i32 NEON SIMD implementation for FixedI32<$frac>
                        let _ = (rd_values, max_dist, sibling_mask);
                        todo!(
                            "SIMD implementation for FixedI32<{}> not yet implemented",
                            stringify!($frac)
                        )
                    }

                    #[cfg(not(any(
                        all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"),
                        all(feature = "simd", target_arch = "aarch64")
                    )))]
                    {
                        autovec_fallback!(8, u8, rd_values, max_dist, sibling_mask)
                    }
                }
            }

            impl SimdSelectBestChildBlock3 for fixed::FixedI32<$frac> {
                #[inline(always)]
                fn simd_select_best_child_block3(
                    rd_values: &[fixed::FixedI32<$frac>; 8],
                    candidate_mask: u8,
                ) -> Option<u8> {
                    scalar_select_best_child_block3(rd_values, candidate_mask)
                }
            }
        };
    }

    /// Macro to implement SimdPrune for FixedU32 with specific fractional bit count.
    ///
    /// Note: Parameterized by fractional bits in case multiply operations differ.
    #[allow(unused_macros)]
    macro_rules! impl_simd_prune_fixed_u32 {
        ($frac:ty) => {
            impl sealed::Sealed for fixed::FixedU32<$frac> {}

            impl SimdPrune for fixed::FixedU32<$frac> {
                #[inline(always)]
                fn simd_prune_block3(
                    rd_values: &[fixed::FixedU32<$frac>; 8],
                    max_dist: fixed::FixedU32<$frac>,
                    sibling_mask: u8,
                ) -> u8 {
                    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
                    {
                        // TODO: u32 SIMD implementation for FixedU32<$frac>
                        let _ = (rd_values, max_dist, sibling_mask);
                        todo!(
                            "SIMD implementation for FixedU32<{}> not yet implemented",
                            stringify!($frac)
                        )
                    }

                    #[cfg(all(
                        feature = "simd",
                        target_arch = "aarch64",
                        not(all(target_arch = "x86_64", target_feature = "avx2"))
                    ))]
                    {
                        // TODO: u32 NEON SIMD implementation for FixedU32<$frac>
                        let _ = (rd_values, max_dist, sibling_mask);
                        todo!(
                            "SIMD implementation for FixedU32<{}> not yet implemented",
                            stringify!($frac)
                        )
                    }

                    #[cfg(not(any(
                        all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"),
                        all(feature = "simd", target_arch = "aarch64")
                    )))]
                    {
                        autovec_fallback!(8, u8, rd_values, max_dist, sibling_mask)
                    }
                }
            }

            impl SimdSelectBestChildBlock3 for fixed::FixedU32<$frac> {
                #[inline(always)]
                fn simd_select_best_child_block3(
                    rd_values: &[fixed::FixedU32<$frac>; 8],
                    candidate_mask: u8,
                ) -> Option<u8> {
                    scalar_select_best_child_block3(rd_values, candidate_mask)
                }
            }
        };
    }

    /// Macro to implement SimdPrune for FixedI16 with specific fractional bit count.
    ///
    /// Note: Parameterized by fractional bits in case multiply operations differ.
    #[allow(unused_macros)]
    macro_rules! impl_simd_prune_fixed_i16 {
        ($frac:ty) => {
            impl sealed::Sealed for fixed::FixedI16<$frac> {}

            impl SimdPrune for fixed::FixedI16<$frac> {
                #[inline(always)]
                fn simd_prune_block3(
                    rd_values: &[fixed::FixedI16<$frac>; 8],
                    max_dist: fixed::FixedI16<$frac>,
                    sibling_mask: u8,
                ) -> u8 {
                    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
                    {
                        // TODO: i16 SIMD implementation for FixedI16<$frac>
                        let _ = (rd_values, max_dist, sibling_mask);
                        todo!(
                            "SIMD implementation for FixedI16<{}> not yet implemented",
                            stringify!($frac)
                        )
                    }

                    #[cfg(all(
                        feature = "simd",
                        target_arch = "aarch64",
                        not(all(target_arch = "x86_64", target_feature = "avx2"))
                    ))]
                    {
                        // TODO: i16 NEON SIMD implementation for FixedI16<$frac>
                        let _ = (rd_values, max_dist, sibling_mask);
                        todo!(
                            "SIMD implementation for FixedI16<{}> not yet implemented",
                            stringify!($frac)
                        )
                    }

                    #[cfg(not(any(
                        all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"),
                        all(feature = "simd", target_arch = "aarch64")
                    )))]
                    {
                        autovec_fallback!(8, u8, rd_values, max_dist, sibling_mask)
                    }
                }
            }

            impl SimdSelectBestChildBlock3 for fixed::FixedI16<$frac> {
                #[inline(always)]
                fn simd_select_best_child_block3(
                    rd_values: &[fixed::FixedI16<$frac>; 8],
                    candidate_mask: u8,
                ) -> Option<u8> {
                    scalar_select_best_child_block3(rd_values, candidate_mask)
                }
            }
        };
    }

    /// Macro to implement SimdPrune for FixedU16 with specific fractional bit count.
    ///
    /// Note: Parameterized by fractional bits in case multiply operations differ.
    #[allow(unused_macros)]
    macro_rules! impl_simd_prune_fixed_u16 {
        ($frac:ty) => {
            impl sealed::Sealed for fixed::FixedU16<$frac> {}

            impl SimdPrune for fixed::FixedU16<$frac> {
                #[inline(always)]
                fn simd_prune_block3(
                    rd_values: &[fixed::FixedU16<$frac>; 8],
                    max_dist: fixed::FixedU16<$frac>,
                    sibling_mask: u8,
                ) -> u8 {
                    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
                    {
                        // TODO: u16 SIMD implementation for FixedU16<$frac>
                        let _ = (rd_values, max_dist, sibling_mask);
                        todo!(
                            "SIMD implementation for FixedU16<{}> not yet implemented",
                            stringify!($frac)
                        )
                    }

                    #[cfg(all(
                        feature = "simd",
                        target_arch = "aarch64",
                        not(all(target_arch = "x86_64", target_feature = "avx2"))
                    ))]
                    {
                        // TODO: u16 NEON SIMD implementation for FixedU16<$frac>
                        let _ = (rd_values, max_dist, sibling_mask);
                        todo!(
                            "SIMD implementation for FixedU16<{}> not yet implemented",
                            stringify!($frac)
                        )
                    }

                    #[cfg(not(any(
                        all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"),
                        all(feature = "simd", target_arch = "aarch64")
                    )))]
                    {
                        autovec_fallback!(8, u8, rd_values, max_dist, sibling_mask)
                    }
                }
            }

            impl SimdSelectBestChildBlock3 for fixed::FixedU16<$frac> {
                #[inline(always)]
                fn simd_select_best_child_block3(
                    rd_values: &[fixed::FixedU16<$frac>; 8],
                    candidate_mask: u8,
                ) -> Option<u8> {
                    scalar_select_best_child_block3(rd_values, candidate_mask)
                }
            }
        };
    }

    // Generate implementations for the fixed-point types used in utils
    use fixed::types::extra::{U0, U16, U8};

    impl_simd_prune_fixed_i32!(U0);
    impl_simd_prune_fixed_i32!(U16);
    impl_simd_prune_fixed_u16!(U8);
}

#[cfg(feature = "f16")]
mod f16_impl {
    use super::*;
    use half::f16;

    impl sealed::Sealed for f16 {}

    impl SimdPrune for f16 {
        #[inline(always)]
        fn simd_prune_block3(rd_values: &[f16; 8], max_dist: f16, sibling_mask: u8) -> u8 {
            #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
            {
                // TODO: f16 SIMD implementation (possibly widen to f32)
                let _ = (rd_values, max_dist, sibling_mask);
                todo!("SIMD implementation for f16 not yet implemented")
            }

            #[cfg(all(
                feature = "simd",
                target_arch = "aarch64",
                not(all(target_arch = "x86_64", target_feature = "avx2"))
            ))]
            {
                // TODO: f16 SIMD implementation (possibly widen to f32)
                let _ = (rd_values, max_dist, sibling_mask);
                todo!("SIMD implementation for f16 not yet implemented")
            }

            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                autovec_fallback!(8, u8, rd_values, max_dist, sibling_mask)
            }
        }
    }

    impl SimdSelectBestChildBlock3 for f16 {
        #[inline(always)]
        fn simd_select_best_child_block3(rd_values: &[f16; 8], candidate_mask: u8) -> Option<u8> {
            scalar_select_best_child_block3(rd_values, candidate_mask)
        }
    }
}

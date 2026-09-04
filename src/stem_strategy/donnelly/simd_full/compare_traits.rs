//! Type-specific block comparison traits for Donnelly SIMD strategies.
//!
//! Replaces size_of-based dispatch with explicit trait implementations per type.

use std::ptr::NonNull;

/// Trait for comparing a query value against 7 pivots in a Block3 (3-level block).
///
/// Block3 produces 8 possible children (2^3).
///
/// Default implementation panics - types must provide actual implementations to support Block3.
pub trait CompareBlock3: Copy {
    /// Compare query value against block pivots, returning child index (0-7).
    ///
    /// # Arguments
    /// * `stems_ptr` - Pointer to start of stems array (cast to u8 for offset calc)
    /// * `query_val` - Query value in this dimension
    /// * `block_base_idx` - Cache-line base index for this block
    ///
    /// # Returns
    /// Child index (0-7) indicating which of 8 children the query falls into.
    ///
    /// # Panics
    /// Default implementation panics. Override this method to support Block3 traversal.
    fn compare_block3_impl(
        _stems_ptr: NonNull<u8>,
        _query_val: Self,
        _block_base_idx: usize,
    ) -> u8 {
        unimplemented!(
            "Type {} does not support Block3 comparison. Use a supported type (f32, f64, or fixed-point types) \
             or implement CompareBlock3 trait.",
            std::any::type_name::<Self>()
        )
    }
}

/// Trait for comparing a query value against 15 pivots in a Block4 (4-level block).
///
/// Block4 produces 16 possible children (2^4).
///
/// Default implementation panics - types must provide actual implementations to support Block4.
pub trait CompareBlock4: Copy {
    /// Compare query value against block pivots, returning child index (0-15).
    ///
    /// # Arguments
    /// * `stems_ptr` - Pointer to start of stems array (cast to u8 for offset calc)
    /// * `query_val` - Query value in this dimension
    /// * `block_base_idx` - Cache-line base index for this block
    ///
    /// # Returns
    /// Child index (0-15) indicating which of 16 children the query falls into.
    ///
    /// # Panics
    /// Default implementation panics. Override this method to support Block4 traversal.
    fn compare_block4_impl(
        _stems_ptr: NonNull<u8>,
        _query_val: Self,
        _block_base_idx: usize,
    ) -> u8 {
        unimplemented!(
            "Type {} does not support Block4 comparison. Use a supported type (f32, or f64 on 128-byte cache line systems) \
             or implement CompareBlock4 trait.",
            std::any::type_name::<Self>()
        )
    }
}

/// Autovectorization-friendly comparison macro for block traversal.
///
/// Compares query value against `$pivot_count` real pivots (ignores padding).
/// Returns count of pivots <= query_val, which maps to child index.
///
/// # Parameters
/// * `$pivot_count` - Number of actual pivots (7 for Block3, 15 for Block4)
/// * `$ty` - Type of values being compared
/// * `$stems_ptr` - `NonNull<u8>` pointer to stems
/// * `$block_base_idx` - Cache line base index
/// * `$query_val` - Query value to compare
macro_rules! autovec_compare_block {
    ($pivot_count:expr, $ty:ty, $stems_ptr:expr, $block_base_idx:expr, $query_val:expr) => {{
        unsafe {
            let ptr = $stems_ptr
                .as_ptr()
                .add($block_base_idx * std::mem::size_of::<$ty>())
                as *const $ty;
            let mut count = 0u8;
            for i in 0..$pivot_count {
                if $query_val >= *ptr.add(i) {
                    count += 1;
                }
            }
            count
        }
    }};
}

// ====================================================================================
// f64 implementations
// ====================================================================================

impl CompareBlock3 for f64 {
    #[inline(always)]
    fn compare_block3_impl(stems_ptr: NonNull<u8>, query_val: Self, block_base_idx: usize) -> u8 {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            #[cfg(target_feature = "avx512f")]
            {
                unsafe {
                    use std::arch::x86_64::*;

                    let ptr = stems_ptr.as_ptr().add(block_base_idx * 8) as *const f64;
                    let pivots = _mm512_loadu_pd(ptr);
                    let query_vec = _mm512_set1_pd(query_val);
                    // Lane 7 is the Block3 padding slot and may carry an embedded
                    // item-summary bit pattern on summary-mode f64 trees. This kernel
                    // is not parameterised by the tree's item-leaf mode, so excluding
                    // lane 7 here is a negligible but unavoidable cost even for
                    // unsorted trees, where the padding is ±INFINITY and could never
                    // be selected anyway.
                    let mask = _mm512_cmp_pd_mask(query_vec, pivots, _CMP_GE_OQ) & 0x7f;
                    _popcnt32(mask as i32) as u8
                }
            }

            #[cfg(not(target_feature = "avx512f"))]
            {
                unsafe {
                    use std::arch::x86_64::*;

                    let ptr = stems_ptr.as_ptr().add(block_base_idx * 8) as *const f64;
                    let pivots_low = _mm256_loadu_pd(ptr);
                    let pivots_high = _mm256_loadu_pd(ptr.add(4));
                    let query_vec = _mm256_set1_pd(query_val);

                    let cmp_low = _mm256_cmp_pd(query_vec, pivots_low, _CMP_GE_OQ);
                    let cmp_high = _mm256_cmp_pd(query_vec, pivots_high, _CMP_GE_OQ);

                    let mask_low = _mm256_movemask_pd(cmp_low) as u32;
                    let mask_high = _mm256_movemask_pd(cmp_high) as u32;
                    // Excludes lane 7 (Block3 padding); see the note above.
                    let mask = (mask_low | (mask_high << 4)) & 0x7f;

                    _popcnt32(mask as i32) as u8
                }
            }
        }

        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        {
            unsafe {
                crate::stem_strategy::donnelly::simd_full::aarch64::compare_block3_f64_neon(
                    stems_ptr,
                    block_base_idx,
                    query_val,
                )
            }
        }

        #[cfg(not(any(
            all(feature = "simd", target_arch = "x86_64"),
            all(feature = "simd", target_arch = "aarch64")
        )))]
        {
            unsafe {
                crate::stem_strategy::donnelly::simd_full::autovec::compare_block3_f64_autovec(
                    stems_ptr,
                    block_base_idx,
                    query_val,
                )
            }
        }
    }
}

// f64 Block4: Only compile on 128-byte cache line systems
#[cfg(cache_line_128)]
impl CompareBlock4 for f64 {
    #[inline(always)]
    fn compare_block4_impl(stems_ptr: NonNull<u8>, query_val: Self, block_base_idx: usize) -> u8 {
        // For now, only autovec (no SIMD intrinsics for f64 Block4 yet)
        autovec_compare_block!(15, f64, stems_ptr, block_base_idx, query_val)
    }
}

// f64 Block4 on 64-byte cache lines: use default unimplemented!() from trait
#[cfg(not(cache_line_128))]
impl CompareBlock4 for f64 {}

// ====================================================================================
// f32 implementations
// ====================================================================================

impl CompareBlock3 for f32 {
    #[inline(always)]
    fn compare_block3_impl(stems_ptr: NonNull<u8>, query_val: Self, block_base_idx: usize) -> u8 {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            #[cfg(target_feature = "avx512f")]
            {
                unsafe {
                    use std::arch::x86_64::*;

                    let ptr = stems_ptr.as_ptr().add(block_base_idx * 4) as *const f32;
                    let pivots = _mm256_loadu_ps(ptr);
                    let query_vec = _mm256_set1_ps(query_val);
                    // Lane 7 is the Block3 padding slot. It never carries summaries
                    // for f32 trees, but this kernel is not parameterised by the
                    // tree's item-leaf mode, so excluding it is a negligible but
                    // unavoidable cost; for unsorted trees the padding is ±INFINITY
                    // and could never be selected anyway.
                    let mask = _mm256_cmp_ps_mask(query_vec, pivots, _CMP_GE_OQ) & 0x7f;
                    _popcnt32(mask as i32) as u8
                }
            }

            #[cfg(not(target_feature = "avx512f"))]
            {
                unsafe {
                    use std::arch::x86_64::*;

                    let ptr = stems_ptr.as_ptr().add(block_base_idx * 4) as *const f32;
                    let pivots = _mm256_loadu_ps(ptr);
                    let query_vec = _mm256_set1_ps(query_val);

                    let cmp = _mm256_cmp_ps(query_vec, pivots, _CMP_GE_OQ);
                    // Excludes lane 7 (Block3 padding); see the note above.
                    let mask = (_mm256_movemask_ps(cmp) as u32) & 0x7f;

                    _popcnt32(mask as i32) as u8
                }
            }
        }

        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        {
            unsafe {
                crate::stem_strategy::donnelly::simd_full::aarch64::compare_block3_f32_neon(
                    stems_ptr,
                    block_base_idx,
                    query_val,
                )
            }
        }

        #[cfg(not(any(
            all(feature = "simd", target_arch = "x86_64"),
            all(feature = "simd", target_arch = "aarch64")
        )))]
        {
            unsafe {
                crate::stem_strategy::donnelly::simd_full::autovec::compare_block3_f32_autovec(
                    stems_ptr,
                    block_base_idx,
                    query_val,
                )
            }
        }
    }
}

impl CompareBlock4 for f32 {
    #[inline(always)]
    fn compare_block4_impl(stems_ptr: NonNull<u8>, query_val: Self, block_base_idx: usize) -> u8 {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            #[cfg(target_feature = "avx512f")]
            {
                unsafe {
                    use std::arch::x86_64::*;

                    let ptr = stems_ptr.as_ptr().add(block_base_idx * 4) as *const f32;
                    let pivots = _mm512_loadu_ps(ptr);
                    let query_vec = _mm512_set1_ps(query_val);

                    let mask = _mm512_cmp_ps_mask(query_vec, pivots, _CMP_GE_OQ);
                    _popcnt32(mask as i32) as u8
                }
            }

            #[cfg(not(target_feature = "avx512f"))]
            {
                unsafe {
                    use std::arch::x86_64::*;

                    let ptr = stems_ptr.as_ptr().add(block_base_idx * 4) as *const f32;
                    let pivots_low = _mm256_loadu_ps(ptr);
                    let pivots_high = _mm256_loadu_ps(ptr.add(8));
                    let query_vec = _mm256_set1_ps(query_val);

                    let cmp_low = _mm256_cmp_ps(query_vec, pivots_low, _CMP_GE_OQ);
                    let cmp_high = _mm256_cmp_ps(query_vec, pivots_high, _CMP_GE_OQ);

                    let mask_low = _mm256_movemask_ps(cmp_low) as u32;
                    let mask_high = _mm256_movemask_ps(cmp_high) as u32;
                    let mask = mask_low | (mask_high << 8);

                    _popcnt32(mask as i32) as u8
                }
            }
        }

        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        {
            unsafe {
                crate::stem_strategy::donnelly::simd_full::aarch64::compare_block4_f32_neon(
                    stems_ptr,
                    block_base_idx,
                    query_val,
                )
            }
        }

        #[cfg(not(any(
            all(feature = "simd", target_arch = "x86_64"),
            all(feature = "simd", target_arch = "aarch64")
        )))]
        {
            unsafe {
                crate::stem_strategy::donnelly::simd_full::autovec::compare_block4_f32_autovec(
                    stems_ptr,
                    block_base_idx,
                    query_val,
                )
            }
        }
    }
}

// ====================================================================================
// Fixed-point implementations
// ====================================================================================

#[cfg(feature = "fixed")]
mod fixed_impls {
    use super::*;
    use fixed::{types::extra, FixedI32, FixedU16};

    type U0 = extra::U0;
    type U16 = extra::U16;
    type U8 = extra::U8;

    /// Macro to implement CompareBlock traits for fixed-point types.
    ///
    /// Currently uses autovec for all architectures. Future work could add
    /// integer SIMD intrinsics (e.g., _mm256_cmpgt_epi32 for signed comparisons).
    ///
    /// Note: The $frac parameter is preserved even though comparison logic is the same
    /// for all fractional bit variants, because future SIMD implementations may need
    /// type-specific handling for other operations.
    macro_rules! impl_compare_fixed {
        ($fixed_ty:ty, $frac:ty) => {
            impl CompareBlock3 for $fixed_ty {
                #[inline(always)]
                fn compare_block3_impl(
                    stems_ptr: NonNull<u8>,
                    query_val: Self,
                    block_base_idx: usize,
                ) -> u8 {
                    autovec_compare_block!(7, $fixed_ty, stems_ptr, block_base_idx, query_val)
                }
            }

            impl CompareBlock4 for $fixed_ty {
                #[inline(always)]
                fn compare_block4_impl(
                    stems_ptr: NonNull<u8>,
                    query_val: Self,
                    block_base_idx: usize,
                ) -> u8 {
                    autovec_compare_block!(15, $fixed_ty, stems_ptr, block_base_idx, query_val)
                }
            }
        };
    }

    // Implement for all fixed-point types that have AxisUnified impls
    impl_compare_fixed!(FixedI32<U16>, U16);
    impl_compare_fixed!(FixedI32<U0>, U0);
    impl_compare_fixed!(FixedU16<U8>, U8);
}

// ====================================================================================
// f16 implementations
// ====================================================================================

#[cfg(feature = "f16")]
mod f16_impls {
    use super::*;
    use half::f16;

    impl CompareBlock3 for f16 {
        #[inline(always)]
        fn compare_block3_impl(
            stems_ptr: NonNull<u8>,
            query_val: Self,
            block_base_idx: usize,
        ) -> u8 {
            // Autovec for now; future optimization could widen to f32 for SIMD
            autovec_compare_block!(7, f16, stems_ptr, block_base_idx, query_val)
        }
    }

    impl CompareBlock4 for f16 {
        #[inline(always)]
        fn compare_block4_impl(
            stems_ptr: NonNull<u8>,
            query_val: Self,
            block_base_idx: usize,
        ) -> u8 {
            autovec_compare_block!(15, f16, stems_ptr, block_base_idx, query_val)
        }
    }
}

/// Macro to implement CompareBlock traits for uint types.
///
/// Currently uses autovec for all architectures. Future work could add
/// integer SIMD intrinsics (e.g., _mm256_cmpgt_epi32 for comparisons).
macro_rules! impl_compare_uint {
    ($uint_ty:ty) => {
        impl CompareBlock3 for $uint_ty {
            #[inline(always)]
            fn compare_block3_impl(
                stems_ptr: NonNull<u8>,
                query_val: Self,
                block_base_idx: usize,
            ) -> u8 {
                autovec_compare_block!(7, $uint_ty, stems_ptr, block_base_idx, query_val)
            }
        }

        impl CompareBlock4 for $uint_ty {
            #[inline(always)]
            fn compare_block4_impl(
                stems_ptr: NonNull<u8>,
                query_val: Self,
                block_base_idx: usize,
            ) -> u8 {
                autovec_compare_block!(15, $uint_ty, stems_ptr, block_base_idx, query_val)
            }
        }
    };
}

impl_compare_uint!(u8);
impl_compare_uint!(u16);
impl_compare_uint!(u32);

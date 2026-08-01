#![allow(clippy::missing_safety_doc)]

use core::arch::x86_64::{__m128, __m128d, __m256, __m256d, __m512, __m512d};

/// AVX512 f64 leaf-kernel operations contract.
///
/// Mirrors the shape used by the hand-tuned `ultimate_*` AVX512 kernel.
pub trait Avx512F64LeafOps {
    /// calculate distance on 8 f64's at once (dimension 0, ie set initial accumulator value)
    unsafe fn dist_k0_f64x8(delta: __m512d) -> __m512d;

    /// calculate distance on 8 f64's at once (dimensions 1+, ie add to accumulator)
    unsafe fn dist_kn_f64x8(acc: __m512d, delta: __m512d) -> __m512d;

    /// Calculate the distance to three-dimensional rectangles from their
    /// non-negative per-axis offsets.
    ///
    /// Metrics may override this composition to expose more instruction-level
    /// parallelism than the accumulator-shaped leaf operations permit.
    #[inline(always)]
    unsafe fn rect_dist_f64x8_3(off0: __m512d, off1: __m512d, off2: __m512d) -> __m512d {
        let acc = Self::dist_k0_f64x8(off0);
        let acc = Self::dist_kn_f64x8(acc, off1);
        Self::dist_kn_f64x8(acc, off2)
    }

    /// calculate distance on 4 f64's at once (dimension 0, ie set initial accumulator value)
    unsafe fn dist_k0_f64x4(delta: __m256d) -> __m256d;

    /// calculate distance on 4 f64's at once (dimensions 1+, ie add to accumulator)
    unsafe fn dist_kn_f64x4(acc: __m256d, delta: __m256d) -> __m256d;

    /// calculate distance on 2 f64's at once (dimension 0, ie set initial accumulator value)
    unsafe fn dist_k0_f64x2(delta: __m128d) -> __m128d;

    /// calculate distance on 2 f64's at once (dimensions 1+, ie add to accumulator)
    unsafe fn dist_kn_f64x2(acc: __m128d, delta: __m128d) -> __m128d;

    /// calculate distance on scalar f64 (dimension 0, ie set initial accumulator value)
    fn dist_k0_f64x1(delta: f64) -> f64;

    /// calculate distance on scalar f64 (dimensions 1+, ie add to accumulator)
    fn dist_kn_f64x1(acc: f64, delta: f64) -> f64;
}

/// Placeholder implementation for metrics without AVX512 f64 specializations.
pub struct UnsupportedAvx512F64LeafOps;

impl Avx512F64LeafOps for UnsupportedAvx512F64LeafOps {
    /// calculate distance on 8 f64's at once (dimension 0, ie set initial accumulator value)
    #[inline(always)]
    unsafe fn dist_k0_f64x8(_delta: __m512d) -> __m512d {
        panic!("AVX512 f64 leaf ops are not implemented for this metric")
    }

    /// calculate distance on 8 f64's at once (dimensions 1+, ie add to accumulator)
    #[inline(always)]
    unsafe fn dist_kn_f64x8(_acc: __m512d, _delta: __m512d) -> __m512d {
        panic!("AVX512 f64 leaf ops are not implemented for this metric")
    }

    /// calculate distance on 4 f64's at once (dimension 0, ie set initial accumulator value)
    #[inline(always)]
    unsafe fn dist_k0_f64x4(_delta: __m256d) -> __m256d {
        panic!("AVX512 f64 leaf ops are not implemented for this metric")
    }

    /// calculate distance on 4 f64's at once (dimensions 1+, ie add to accumulator)
    #[inline(always)]
    unsafe fn dist_kn_f64x4(_acc: __m256d, _delta: __m256d) -> __m256d {
        panic!("AVX512 f64 leaf ops are not implemented for this metric")
    }

    /// calculate distance on 2 f64's at once (dimension 0, ie set initial accumulator value)
    #[inline(always)]
    unsafe fn dist_k0_f64x2(_delta: __m128d) -> __m128d {
        panic!("AVX512 f64 leaf ops are not implemented for this metric")
    }

    /// calculate distance on 2 f64's at once (dimensions 1+, ie add to accumulator)
    #[inline(always)]
    unsafe fn dist_kn_f64x2(_acc: __m128d, _delta: __m128d) -> __m128d {
        panic!("AVX512 f64 leaf ops are not implemented for this metric")
    }

    /// calculate distance on scalar f64 (dimension 0, ie set initial accumulator value)
    #[inline(always)]
    fn dist_k0_f64x1(_delta: f64) -> f64 {
        panic!("AVX512 f64 leaf ops are not implemented for this metric")
    }

    /// calculate distance on scalar f64 (dimensions 1+, ie add to accumulator)
    #[inline(always)]
    fn dist_kn_f64x1(_acc: f64, _delta: f64) -> f64 {
        panic!("AVX512 f64 leaf ops are not implemented for this metric")
    }
}

/// AVX512 f32 leaf-kernel operations contract.
pub trait Avx512F32LeafOps {
    /// Calculate distance on 16 f32 lanes for the first dimension.
    unsafe fn dist_k0_f32x16(delta: __m512) -> __m512;

    /// Accumulate distance on 16 f32 lanes for subsequent dimensions.
    unsafe fn dist_kn_f32x16(acc: __m512, delta: __m512) -> __m512;

    /// Calculate the distance to four-dimensional rectangles from their
    /// non-negative per-axis offsets.
    ///
    /// Metrics may override this composition to expose more instruction-level
    /// parallelism than the accumulator-shaped leaf operations permit.
    #[inline(always)]
    unsafe fn rect_dist_f32x16_4(off0: __m512, off1: __m512, off2: __m512, off3: __m512) -> __m512 {
        let acc = Self::dist_k0_f32x16(off0);
        let acc = Self::dist_kn_f32x16(acc, off1);
        let acc = Self::dist_kn_f32x16(acc, off2);
        Self::dist_kn_f32x16(acc, off3)
    }

    /// Calculate distance on 8 f32 lanes for the first dimension.
    unsafe fn dist_k0_f32x8(delta: __m256) -> __m256;

    /// Accumulate distance on 8 f32 lanes for subsequent dimensions.
    unsafe fn dist_kn_f32x8(acc: __m256, delta: __m256) -> __m256;

    /// Calculate distance on 4 f32 lanes for the first dimension.
    unsafe fn dist_k0_f32x4(delta: __m128) -> __m128;

    /// Accumulate distance on 4 f32 lanes for subsequent dimensions.
    unsafe fn dist_kn_f32x4(acc: __m128, delta: __m128) -> __m128;

    /// Calculate scalar f32 distance for the first dimension.
    fn dist_k0_f32x1(delta: f32) -> f32;

    /// Accumulate scalar f32 distance for subsequent dimensions.
    fn dist_kn_f32x1(acc: f32, delta: f32) -> f32;
}

/// Placeholder implementation for metrics without AVX512 f32 specializations.
pub struct UnsupportedAvx512F32LeafOps;

impl Avx512F32LeafOps for UnsupportedAvx512F32LeafOps {
    #[inline(always)]
    unsafe fn dist_k0_f32x16(_delta: __m512) -> __m512 {
        panic!("AVX512 f32 leaf ops are not implemented for this metric")
    }

    #[inline(always)]
    unsafe fn dist_kn_f32x16(_acc: __m512, _delta: __m512) -> __m512 {
        panic!("AVX512 f32 leaf ops are not implemented for this metric")
    }

    #[inline(always)]
    unsafe fn dist_k0_f32x8(_delta: __m256) -> __m256 {
        panic!("AVX512 f32 leaf ops are not implemented for this metric")
    }

    #[inline(always)]
    unsafe fn dist_kn_f32x8(_acc: __m256, _delta: __m256) -> __m256 {
        panic!("AVX512 f32 leaf ops are not implemented for this metric")
    }

    #[inline(always)]
    unsafe fn dist_k0_f32x4(_delta: __m128) -> __m128 {
        panic!("AVX512 f32 leaf ops are not implemented for this metric")
    }

    #[inline(always)]
    unsafe fn dist_kn_f32x4(_acc: __m128, _delta: __m128) -> __m128 {
        panic!("AVX512 f32 leaf ops are not implemented for this metric")
    }

    #[inline(always)]
    fn dist_k0_f32x1(_delta: f32) -> f32 {
        panic!("AVX512 f32 leaf ops are not implemented for this metric")
    }

    #[inline(always)]
    fn dist_kn_f32x1(_acc: f32, _delta: f32) -> f32 {
        panic!("AVX512 f32 leaf ops are not implemented for this metric")
    }
}

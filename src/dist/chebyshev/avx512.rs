use std::arch::x86_64::*;

use crate::dist::common::{avx2::*, avx512::*};
use crate::dist::distance_metric_avx512::{Avx512F32LeafOps, Avx512F64LeafOps};

pub struct ChebyshevAvx512F64LeafOps;

impl Avx512F64LeafOps for ChebyshevAvx512F64LeafOps {
    #[inline(always)]
    unsafe fn dist_k0_f64x8(delta: __m512d) -> __m512d {
        custom_mm512_abs_pd(delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f64x8(acc: __m512d, delta: __m512d) -> __m512d {
        _mm512_max_pd(acc, custom_mm512_abs_pd(delta))
    }

    #[inline(always)]
    unsafe fn rect_dist_f64x8_3(off0: __m512d, off1: __m512d, off2: __m512d) -> __m512d {
        _mm512_max_pd(_mm512_max_pd(off0, off1), off2)
    }

    #[inline(always)]
    unsafe fn dist_k0_f64x4(delta: __m256d) -> __m256d {
        custom_mm256_abs_pd(delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f64x4(acc: __m256d, delta: __m256d) -> __m256d {
        _mm256_max_pd(acc, custom_mm256_abs_pd(delta))
    }

    #[inline(always)]
    unsafe fn dist_k0_f64x2(delta: __m128d) -> __m128d {
        custom_mm_abs_pd(delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f64x2(acc: __m128d, delta: __m128d) -> __m128d {
        _mm_max_pd(acc, custom_mm_abs_pd(delta))
    }

    #[inline(always)]
    fn dist_k0_f64x1(delta: f64) -> f64 {
        delta.abs()
    }

    #[inline(always)]
    fn dist_kn_f64x1(acc: f64, delta: f64) -> f64 {
        acc.max(delta.abs())
    }
}

pub struct ChebyshevAvx512F32LeafOps;

impl Avx512F32LeafOps for ChebyshevAvx512F32LeafOps {
    #[inline(always)]
    unsafe fn dist_k0_f32x16(delta: __m512) -> __m512 {
        custom_mm512_abs_ps(delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f32x16(acc: __m512, delta: __m512) -> __m512 {
        _mm512_max_ps(acc, custom_mm512_abs_ps(delta))
    }

    #[inline(always)]
    unsafe fn rect_dist_f32x16_4(off0: __m512, off1: __m512, off2: __m512, off3: __m512) -> __m512 {
        _mm512_max_ps(_mm512_max_ps(off0, off1), _mm512_max_ps(off2, off3))
    }

    #[inline(always)]
    unsafe fn dist_k0_f32x8(delta: __m256) -> __m256 {
        custom_mm256_abs_ps(delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f32x8(acc: __m256, delta: __m256) -> __m256 {
        _mm256_max_ps(acc, custom_mm256_abs_ps(delta))
    }

    #[inline(always)]
    unsafe fn dist_k0_f32x4(delta: __m128) -> __m128 {
        custom_mm_abs_ps(delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f32x4(acc: __m128, delta: __m128) -> __m128 {
        _mm_max_ps(acc, custom_mm_abs_ps(delta))
    }

    #[inline(always)]
    fn dist_k0_f32x1(delta: f32) -> f32 {
        delta.abs()
    }

    #[inline(always)]
    fn dist_kn_f32x1(acc: f32, delta: f32) -> f32 {
        acc.max(delta.abs())
    }
}

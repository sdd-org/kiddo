use std::arch::x86_64::*;

use crate::dist::distance_metric_avx512::{Avx512F32LeafOps, Avx512F64LeafOps};

pub struct SquaredEuclideanAvx512F64LeafOps;

impl Avx512F64LeafOps for SquaredEuclideanAvx512F64LeafOps {
    #[inline(always)]
    unsafe fn dist_k0_f64x8(delta: __m512d) -> __m512d {
        _mm512_mul_pd(delta, delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f64x8(acc: __m512d, delta: __m512d) -> __m512d {
        _mm512_fmadd_pd(delta, delta, acc)
    }

    #[inline(always)]
    unsafe fn rect_dist_f64x8_3(off0: __m512d, off1: __m512d, off2: __m512d) -> __m512d {
        let dist0 = _mm512_mul_pd(off0, off0);
        let dist1 = _mm512_mul_pd(off1, off1);
        let dist2 = _mm512_mul_pd(off2, off2);
        _mm512_add_pd(_mm512_add_pd(dist0, dist1), dist2)
    }

    #[inline(always)]
    unsafe fn dist_k0_f64x4(delta: __m256d) -> __m256d {
        _mm256_mul_pd(delta, delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f64x4(acc: __m256d, delta: __m256d) -> __m256d {
        _mm256_fmadd_pd(delta, delta, acc)
    }

    #[inline(always)]
    unsafe fn dist_k0_f64x2(delta: __m128d) -> __m128d {
        _mm_mul_pd(delta, delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f64x2(acc: __m128d, delta: __m128d) -> __m128d {
        _mm_fmadd_pd(delta, delta, acc)
    }

    #[inline(always)]
    fn dist_k0_f64x1(delta: f64) -> f64 {
        delta * delta
    }

    #[inline(always)]
    fn dist_kn_f64x1(acc: f64, delta: f64) -> f64 {
        acc + delta * delta
    }
}

pub struct SquaredEuclideanAvx512F32LeafOps;

impl Avx512F32LeafOps for SquaredEuclideanAvx512F32LeafOps {
    #[inline(always)]
    unsafe fn dist_k0_f32x16(delta: __m512) -> __m512 {
        _mm512_mul_ps(delta, delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f32x16(acc: __m512, delta: __m512) -> __m512 {
        _mm512_fmadd_ps(delta, delta, acc)
    }

    #[inline(always)]
    unsafe fn rect_dist_f32x16_4(off0: __m512, off1: __m512, off2: __m512, off3: __m512) -> __m512 {
        let dist01 = _mm512_add_ps(_mm512_mul_ps(off0, off0), _mm512_mul_ps(off1, off1));
        let dist23 = _mm512_add_ps(_mm512_mul_ps(off2, off2), _mm512_mul_ps(off3, off3));
        _mm512_add_ps(dist01, dist23)
    }

    #[inline(always)]
    unsafe fn dist_k0_f32x8(delta: __m256) -> __m256 {
        _mm256_mul_ps(delta, delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f32x8(acc: __m256, delta: __m256) -> __m256 {
        _mm256_fmadd_ps(delta, delta, acc)
    }

    #[inline(always)]
    unsafe fn dist_k0_f32x4(delta: __m128) -> __m128 {
        _mm_mul_ps(delta, delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f32x4(acc: __m128, delta: __m128) -> __m128 {
        _mm_fmadd_ps(delta, delta, acc)
    }

    #[inline(always)]
    fn dist_k0_f32x1(delta: f32) -> f32 {
        delta * delta
    }

    #[inline(always)]
    fn dist_kn_f32x1(acc: f32, delta: f32) -> f32 {
        acc + delta * delta
    }
}

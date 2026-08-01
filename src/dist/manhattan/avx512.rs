use std::arch::x86_64::*;

use crate::dist::distance_metric_avx512::{Avx512F32LeafOps, Avx512F64LeafOps};
#[inline(always)]
unsafe fn abs_pd_512(x: __m512d) -> __m512d {
    let sign = _mm512_set1_pd(-0.0);
    _mm512_andnot_pd(sign, x)
}

#[inline(always)]
unsafe fn abs_pd_256(x: __m256d) -> __m256d {
    let sign = _mm256_set1_pd(-0.0);
    _mm256_andnot_pd(sign, x)
}

#[inline(always)]
unsafe fn abs_pd_128(x: __m128d) -> __m128d {
    let sign = _mm_set1_pd(-0.0);
    _mm_andnot_pd(sign, x)
}

#[inline(always)]
unsafe fn abs_ps_512(x: __m512) -> __m512 {
    let sign = _mm512_set1_ps(-0.0);
    _mm512_andnot_ps(sign, x)
}

#[inline(always)]
unsafe fn abs_ps_256(x: __m256) -> __m256 {
    let sign = _mm256_set1_ps(-0.0);
    _mm256_andnot_ps(sign, x)
}

#[inline(always)]
unsafe fn abs_ps_128(x: __m128) -> __m128 {
    let sign = _mm_set1_ps(-0.0);
    _mm_andnot_ps(sign, x)
}

pub struct ManhattanAvx512F64LeafOps;

impl Avx512F64LeafOps for ManhattanAvx512F64LeafOps {
    #[inline(always)]
    unsafe fn dist_k0_f64x8(delta: __m512d) -> __m512d {
        abs_pd_512(delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f64x8(acc: __m512d, delta: __m512d) -> __m512d {
        _mm512_add_pd(acc, abs_pd_512(delta))
    }

    #[inline(always)]
    unsafe fn rect_dist_f64x8_3(off0: __m512d, off1: __m512d, off2: __m512d) -> __m512d {
        _mm512_add_pd(_mm512_add_pd(off0, off1), off2)
    }

    #[inline(always)]
    unsafe fn dist_k0_f64x4(delta: __m256d) -> __m256d {
        abs_pd_256(delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f64x4(acc: __m256d, delta: __m256d) -> __m256d {
        _mm256_add_pd(acc, abs_pd_256(delta))
    }

    #[inline(always)]
    unsafe fn dist_k0_f64x2(delta: __m128d) -> __m128d {
        abs_pd_128(delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f64x2(acc: __m128d, delta: __m128d) -> __m128d {
        _mm_add_pd(acc, abs_pd_128(delta))
    }

    #[inline(always)]
    fn dist_k0_f64x1(delta: f64) -> f64 {
        delta.abs()
    }

    #[inline(always)]
    fn dist_kn_f64x1(acc: f64, delta: f64) -> f64 {
        acc + delta.abs()
    }
}

pub struct ManhattanAvx512F32LeafOps;

impl Avx512F32LeafOps for ManhattanAvx512F32LeafOps {
    #[inline(always)]
    unsafe fn dist_k0_f32x16(delta: __m512) -> __m512 {
        abs_ps_512(delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f32x16(acc: __m512, delta: __m512) -> __m512 {
        _mm512_add_ps(acc, abs_ps_512(delta))
    }

    #[inline(always)]
    unsafe fn rect_dist_f32x16_4(off0: __m512, off1: __m512, off2: __m512, off3: __m512) -> __m512 {
        _mm512_add_ps(_mm512_add_ps(off0, off1), _mm512_add_ps(off2, off3))
    }

    #[inline(always)]
    unsafe fn dist_k0_f32x8(delta: __m256) -> __m256 {
        abs_ps_256(delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f32x8(acc: __m256, delta: __m256) -> __m256 {
        _mm256_add_ps(acc, abs_ps_256(delta))
    }

    #[inline(always)]
    unsafe fn dist_k0_f32x4(delta: __m128) -> __m128 {
        abs_ps_128(delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f32x4(acc: __m128, delta: __m128) -> __m128 {
        _mm_add_ps(acc, abs_ps_128(delta))
    }

    #[inline(always)]
    fn dist_k0_f32x1(delta: f32) -> f32 {
        delta.abs()
    }

    #[inline(always)]
    fn dist_kn_f32x1(acc: f32, delta: f32) -> f32 {
        acc + delta.abs()
    }
}

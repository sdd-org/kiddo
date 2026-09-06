//! Squared-Euclidean tile kernels, preserving scalar axis accumulation order.
use std::arch::x86_64::*;

#[inline(always)]
pub(super) unsafe fn mask_f32<const K: usize, const EX: bool>(
    query: &[f32; K],
    columns: &[*const f32; K],
    len: usize,
    radius: f32,
    distances: &mut [f32; 32],
) -> u32 {
    debug_assert!(len <= 32);
    let threshold = _mm256_set1_ps(radius);
    let mut mask = 0u32;
    let mut j = 0;
    while j + 8 <= len {
        let mut acc = _mm256_setzero_ps();
        for d in 0..K {
            let delta = _mm256_sub_ps(_mm256_set1_ps(query[d]), _mm256_loadu_ps(columns[d].add(j)));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(delta, delta));
        }
        _mm256_storeu_ps(distances.as_mut_ptr().add(j), acc);
        let bits = if EX {
            _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_LT_OQ>(acc, threshold))
        } else {
            _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_LE_OQ>(acc, threshold))
        } as u32;
        mask |= bits << j;
        j += 8;
    }
    while j < len {
        let mut distance = 0.0;
        for d in 0..K {
            let delta = query[d] - columns[d].add(j).read_unaligned();
            distance += delta * delta;
        }
        distances[j] = distance;
        if if EX {
            distance < radius
        } else {
            distance <= radius
        } {
            mask |= 1 << j;
        }
        j += 1;
    }
    mask
}

#[inline(always)]
pub(super) unsafe fn mask_f64<const K: usize, const EX: bool>(
    query: &[f64; K],
    columns: &[*const f64; K],
    len: usize,
    radius: f64,
    distances: &mut [f64; 32],
) -> u32 {
    debug_assert!(len <= 32);
    let threshold = _mm256_set1_pd(radius);
    let mut mask = 0u32;
    let mut j = 0;
    while j + 4 <= len {
        let mut acc = _mm256_setzero_pd();
        for d in 0..K {
            let delta = _mm256_sub_pd(_mm256_set1_pd(query[d]), _mm256_loadu_pd(columns[d].add(j)));
            acc = _mm256_add_pd(acc, _mm256_mul_pd(delta, delta));
        }
        _mm256_storeu_pd(distances.as_mut_ptr().add(j), acc);
        let bits = if EX {
            _mm256_movemask_pd(_mm256_cmp_pd::<_CMP_LT_OQ>(acc, threshold))
        } else {
            _mm256_movemask_pd(_mm256_cmp_pd::<_CMP_LE_OQ>(acc, threshold))
        } as u32;
        mask |= bits << j;
        j += 4;
    }
    while j < len {
        let mut distance = 0.0;
        for d in 0..K {
            let delta = query[d] - columns[d].add(j).read_unaligned();
            distance += delta * delta;
        }
        distances[j] = distance;
        if if EX {
            distance < radius
        } else {
            distance <= radius
        } {
            mask |= 1 << j;
        }
        j += 1;
    }
    mask
}

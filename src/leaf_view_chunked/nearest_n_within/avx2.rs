#![allow(clippy::missing_safety_doc)]

use std::arch::x86_64::*;

use array_init::array_init;

use crate::dist::distance_metric_avx2::{Avx2F32LeafOps, Avx2F64LeafOps};
use crate::leaf_view::LeafView;
use crate::Content;

#[inline(always)]
unsafe fn emit_results_avx2_f64<T, F, const EXCLUSIVE: bool>(
    dists: __m256d,
    items: *const T,
    base: usize,
    max_dist: f64,
    emit: &mut F,
) where
    T: Content,
    F: FnMut(usize, f64, T),
{
    let mask = if EXCLUSIVE {
        _mm256_movemask_pd(_mm256_cmp_pd(dists, _mm256_set1_pd(max_dist), _CMP_LT_OQ)) as u32
    } else {
        _mm256_movemask_pd(_mm256_cmp_pd(dists, _mm256_set1_pd(max_dist), _CMP_LE_OQ)) as u32
    };
    if mask == 0 {
        return;
    }

    let mut dist_values = [0.0f64; 4];
    _mm256_storeu_pd(dist_values.as_mut_ptr(), dists);

    let mut remaining = mask;
    while remaining != 0 {
        let lane = remaining.trailing_zeros() as usize;
        emit(
            base + lane,
            *dist_values.get_unchecked(lane),
            std::ptr::read_unaligned(items.add(base + lane)),
        );
        remaining &= remaining - 1;
    }
}

#[inline(always)]
unsafe fn emit_results_sse_f64<T, F, const EXCLUSIVE: bool>(
    dists: __m128d,
    items: *const T,
    base: usize,
    max_dist: f64,
    emit: &mut F,
) where
    T: Content,
    F: FnMut(usize, f64, T),
{
    let mask = _mm_movemask_pd(if EXCLUSIVE {
        _mm_cmplt_pd(dists, _mm_set1_pd(max_dist))
    } else {
        _mm_cmple_pd(dists, _mm_set1_pd(max_dist))
    }) as u32;
    if mask == 0 {
        return;
    }

    let mut dist_values = [0.0f64; 2];
    _mm_storeu_pd(dist_values.as_mut_ptr(), dists);

    let mut remaining = mask;
    while remaining != 0 {
        let lane = remaining.trailing_zeros() as usize;
        emit(
            base + lane,
            *dist_values.get_unchecked(lane),
            std::ptr::read_unaligned(items.add(base + lane)),
        );
        remaining &= remaining - 1;
    }
}

#[inline(always)]
unsafe fn line_dists_avx2_f64<L, const K: usize>(
    points: &[*const f64; K],
    query: &[__m256d; K],
    base: usize,
) -> __m256d
where
    L: Avx2F64LeafOps,
{
    let a0 = _mm256_loadu_pd(points[0].add(base));
    let d0 = _mm256_sub_pd(a0, query[0]);
    let mut acc = L::dist_k0_f64x4(d0);

    for dim in 1..K {
        let a = _mm256_loadu_pd(points[dim].add(base));
        let d = _mm256_sub_pd(a, query[dim]);
        acc = L::dist_kn_f64x4(acc, d);
    }

    acc
}

#[inline(always)]
unsafe fn line_dists_sse_f64<L, const K: usize>(
    points: &[*const f64; K],
    query: &[__m128d; K],
    base: usize,
) -> __m128d
where
    L: Avx2F64LeafOps,
{
    let a0 = _mm_loadu_pd(points[0].add(base));
    let d0 = _mm_sub_pd(a0, query[0]);
    let mut acc = L::dist_k0_f64x2(d0);

    for dim in 1..K {
        let a = _mm_loadu_pd(points[dim].add(base));
        let d = _mm_sub_pd(a, query[dim]);
        acc = L::dist_kn_f64x2(acc, d);
    }

    acc
}

#[inline(always)]
unsafe fn dist_scalar_f64<L, const K: usize>(
    points: &[*const f64; K],
    query: &[f64; K],
    idx: usize,
) -> f64
where
    L: Avx2F64LeafOps,
{
    let mut dist = L::dist_k0_f64x1(*points[0].add(idx) - query[0]);
    for dim in 1..K {
        dist = L::dist_kn_f64x1(dist, *points[dim].add(idx) - query[dim]);
    }
    dist
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn nearest_n_within_avx2_unchecked_f64<
    L,
    T,
    F,
    const EXCLUSIVE: bool,
    const K: usize,
    const B: usize,
>(
    leaf: &LeafView<'_, f64, T, K, B>,
    query: &[f64; K],
    max_dist: f64,
    emit: &mut F,
) where
    L: Avx2F64LeafOps,
    T: Content,
    F: FnMut(usize, f64, T),
{
    let points = leaf.points();
    let point_ptrs = array_init(|dim| points[dim].as_ptr());
    nearest_n_within_avx2_raw_f64::<L, T, F, EXCLUSIVE, K>(
        point_ptrs,
        leaf.items().as_ptr(),
        leaf.items().len(),
        query,
        max_dist,
        emit,
    );
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn nearest_n_within_avx2_arena_unchecked_f64<
    L,
    T,
    F,
    const EXCLUSIVE: bool,
    const K: usize,
>(
    tile_base: *const u8,
    len: usize,
    query: &[f64; K],
    max_dist: f64,
    emit: &mut F,
) where
    L: Avx2F64LeafOps,
    T: Content,
    F: FnMut(usize, f64, T),
{
    let point_base = tile_base as *const f64;
    let point_ptrs = array_init(|dim| point_base.add(dim * len));
    let items = tile_base.add(K * len * std::mem::size_of::<f64>()) as *const T;

    nearest_n_within_avx2_raw_f64::<L, T, F, EXCLUSIVE, K>(
        point_ptrs, items, len, query, max_dist, emit,
    );
}

#[target_feature(enable = "avx2")]
unsafe fn nearest_n_within_avx2_raw_f64<L, T, F, const EXCLUSIVE: bool, const K: usize>(
    points: [*const f64; K],
    items: *const T,
    len: usize,
    query: &[f64; K],
    max_dist: f64,
    emit: &mut F,
) where
    L: Avx2F64LeafOps,
    T: Content,
    F: FnMut(usize, f64, T),
{
    if len == 0 {
        return;
    }

    let query_avx2 = array_init(|dim| _mm256_set1_pd(query[dim]));
    let query_sse = array_init(|dim| _mm_set1_pd(query[dim]));
    let mut base = 0usize;

    while base + 4 <= len {
        let d0 = line_dists_avx2_f64::<L, K>(&points, &query_avx2, base);
        emit_results_avx2_f64::<_, _, EXCLUSIVE>(d0, items, base, max_dist, emit);
        base += 4;
    }

    if base + 2 <= len {
        let d0 = line_dists_sse_f64::<L, K>(&points, &query_sse, base);
        emit_results_sse_f64::<_, _, EXCLUSIVE>(d0, items, base, max_dist, emit);
        base += 2;
    }

    for idx in base..len {
        let dist = dist_scalar_f64::<L, K>(&points, query, idx);
        let is_within_dist = if EXCLUSIVE {
            dist < max_dist
        } else {
            dist <= max_dist
        };

        if is_within_dist {
            emit(idx, dist, std::ptr::read_unaligned(items.add(idx)));
        }
    }
}

#[inline(always)]
unsafe fn emit_results_avx2_f32<T, F, const EXCLUSIVE: bool>(
    dists: __m256,
    items: *const T,
    base: usize,
    max_dist: f32,
    emit: &mut F,
) where
    T: Content,
    F: FnMut(usize, f32, T),
{
    let mask = if EXCLUSIVE {
        _mm256_movemask_ps(_mm256_cmp_ps(dists, _mm256_set1_ps(max_dist), _CMP_LT_OQ)) as u32
    } else {
        _mm256_movemask_ps(_mm256_cmp_ps(dists, _mm256_set1_ps(max_dist), _CMP_LE_OQ)) as u32
    };
    if mask == 0 {
        return;
    }

    let mut dist_values = [0.0f32; 8];
    _mm256_storeu_ps(dist_values.as_mut_ptr(), dists);

    let mut remaining = mask;
    while remaining != 0 {
        let lane = remaining.trailing_zeros() as usize;
        emit(
            base + lane,
            *dist_values.get_unchecked(lane),
            std::ptr::read_unaligned(items.add(base + lane)),
        );
        remaining &= remaining - 1;
    }
}

#[inline(always)]
unsafe fn emit_results_sse_f32<T, F, const EXCLUSIVE: bool>(
    dists: __m128,
    items: *const T,
    base: usize,
    max_dist: f32,
    emit: &mut F,
) where
    T: Content,
    F: FnMut(usize, f32, T),
{
    let mask = if EXCLUSIVE {
        _mm_movemask_ps(_mm_cmp_ps(dists, _mm_set1_ps(max_dist), _CMP_LT_OQ)) as u32
    } else {
        _mm_movemask_ps(_mm_cmp_ps(dists, _mm_set1_ps(max_dist), _CMP_LE_OQ)) as u32
    };
    if mask == 0 {
        return;
    }

    let mut dist_values = [0.0f32; 4];
    _mm_storeu_ps(dist_values.as_mut_ptr(), dists);

    let mut remaining = mask;
    while remaining != 0 {
        let lane = remaining.trailing_zeros() as usize;
        emit(
            base + lane,
            *dist_values.get_unchecked(lane),
            std::ptr::read_unaligned(items.add(base + lane)),
        );
        remaining &= remaining - 1;
    }
}

#[inline(always)]
unsafe fn line_dists_avx2_f32<L, const K: usize>(
    points: &[*const f32; K],
    query: &[__m256; K],
    base: usize,
) -> __m256
where
    L: Avx2F32LeafOps,
{
    let a0 = _mm256_loadu_ps(points[0].add(base));
    let d0 = _mm256_sub_ps(a0, query[0]);
    let mut acc = L::dist_k0_f32x8(d0);

    for dim in 1..K {
        let a = _mm256_loadu_ps(points[dim].add(base));
        let d = _mm256_sub_ps(a, query[dim]);
        acc = L::dist_kn_f32x8(acc, d);
    }

    acc
}

#[inline(always)]
unsafe fn line_dists_sse_f32<L, const K: usize>(
    points: &[*const f32; K],
    query: &[__m128; K],
    base: usize,
) -> __m128
where
    L: Avx2F32LeafOps,
{
    let a0 = _mm_loadu_ps(points[0].add(base));
    let d0 = _mm_sub_ps(a0, query[0]);
    let mut acc = L::dist_k0_f32x4(d0);

    for dim in 1..K {
        let a = _mm_loadu_ps(points[dim].add(base));
        let d = _mm_sub_ps(a, query[dim]);
        acc = L::dist_kn_f32x4(acc, d);
    }

    acc
}

#[inline(always)]
unsafe fn dist_scalar_f32<L, const K: usize>(
    points: &[*const f32; K],
    query: &[f32; K],
    idx: usize,
) -> f32
where
    L: Avx2F32LeafOps,
{
    let mut dist = L::dist_k0_f32x1(*points[0].add(idx) - query[0]);
    for dim in 1..K {
        dist = L::dist_kn_f32x1(dist, *points[dim].add(idx) - query[dim]);
    }
    dist
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn nearest_n_within_avx2_unchecked_f32<
    L,
    T,
    F,
    const EXCLUSIVE: bool,
    const K: usize,
    const B: usize,
>(
    leaf: &LeafView<'_, f32, T, K, B>,
    query: &[f32; K],
    max_dist: f32,
    emit: &mut F,
) where
    L: Avx2F32LeafOps,
    T: Content,
    F: FnMut(usize, f32, T),
{
    let points = leaf.points();
    let point_ptrs = array_init(|dim| points[dim].as_ptr());
    nearest_n_within_avx2_raw_f32::<L, T, F, EXCLUSIVE, K>(
        point_ptrs,
        leaf.items().as_ptr(),
        leaf.items().len(),
        query,
        max_dist,
        emit,
    );
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn nearest_n_within_avx2_arena_unchecked_f32<
    L,
    T,
    F,
    const EXCLUSIVE: bool,
    const K: usize,
>(
    tile_base: *const u8,
    len: usize,
    query: &[f32; K],
    max_dist: f32,
    emit: &mut F,
) where
    L: Avx2F32LeafOps,
    T: Content,
    F: FnMut(usize, f32, T),
{
    let point_base = tile_base as *const f32;
    let point_ptrs = array_init(|dim| point_base.add(dim * len));
    let items = tile_base.add(K * len * std::mem::size_of::<f32>()) as *const T;

    nearest_n_within_avx2_raw_f32::<L, T, F, EXCLUSIVE, K>(
        point_ptrs, items, len, query, max_dist, emit,
    );
}

#[target_feature(enable = "avx2")]
unsafe fn nearest_n_within_avx2_raw_f32<L, T, F, const EXCLUSIVE: bool, const K: usize>(
    points: [*const f32; K],
    items: *const T,
    len: usize,
    query: &[f32; K],
    max_dist: f32,
    emit: &mut F,
) where
    L: Avx2F32LeafOps,
    T: Content,
    F: FnMut(usize, f32, T),
{
    if len == 0 {
        return;
    }

    let query_avx2 = array_init(|dim| _mm256_set1_ps(query[dim]));
    let query_sse = array_init(|dim| _mm_set1_ps(query[dim]));
    let mut base = 0usize;

    while base + 8 <= len {
        let d0 = line_dists_avx2_f32::<L, K>(&points, &query_avx2, base);
        emit_results_avx2_f32::<_, _, EXCLUSIVE>(d0, items, base, max_dist, emit);
        base += 8;
    }

    if base + 4 <= len {
        let d0 = line_dists_sse_f32::<L, K>(&points, &query_sse, base);
        emit_results_sse_f32::<_, _, EXCLUSIVE>(d0, items, base, max_dist, emit);
        base += 4;
    }

    for idx in base..len {
        let dist = dist_scalar_f32::<L, K>(&points, query, idx);
        let is_within_dist = if EXCLUSIVE {
            dist < max_dist
        } else {
            dist <= max_dist
        };

        if is_within_dist {
            emit(idx, dist, std::ptr::read_unaligned(items.add(idx)));
        }
    }
}

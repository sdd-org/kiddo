#![allow(clippy::missing_safety_doc)]

use std::arch::x86_64::*;

use array_init::array_init;

use crate::dist::distance_metric_avx512::{Avx512F32LeafOps, Avx512F64LeafOps};
use crate::leaf_view::LeafView;
use crate::Content;

const CHUNK_SIZE: usize = 32;
const LINE_SIZE: usize = 8;
const AVX2_LINE_SIZE: usize = 4;
const CHUNK_SIZE_F32: usize = 64;
const LINE_SIZE_F32: usize = 16;
const AVX2_LINE_SIZE_F32: usize = 8;
const SSE_LINE_SIZE_F32: usize = 4;

#[inline(always)]
unsafe fn emit_results_avx512<T, F, const EXCLUSIVE: bool>(
    dists: __m512d,
    items: *const T,
    base: usize,
    max_dist: f64,
    emit: &mut F,
) where
    T: Content,
    F: FnMut(usize, f64, T),
{
    let mask = if EXCLUSIVE {
        _mm512_cmp_pd_mask(dists, _mm512_set1_pd(max_dist), _CMP_LT_OQ) as u32
    } else {
        _mm512_cmp_pd_mask(dists, _mm512_set1_pd(max_dist), _CMP_LE_OQ) as u32
    };
    if mask == 0 {
        return;
    }

    let mut dist_values = [0.0f64; LINE_SIZE];
    _mm512_storeu_pd(dist_values.as_mut_ptr(), dists);

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
unsafe fn emit_results_avx2<T, F, const EXCLUSIVE: bool>(
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

    let mut dist_values = [0.0f64; AVX2_LINE_SIZE];
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
unsafe fn emit_results_avx128<T, F, const EXCLUSIVE: bool>(
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
unsafe fn line_dists_avx512<L, const K: usize>(
    points: &[*const f64; K],
    query: &[__m512d; K],
    base: usize,
) -> __m512d
where
    L: Avx512F64LeafOps,
{
    let a0 = _mm512_loadu_pd(points[0].add(base));
    let d0 = _mm512_sub_pd(a0, query[0]);
    let mut acc = L::dist_k0_f64x8(d0);

    for dim in 1..K {
        let a = _mm512_loadu_pd(points[dim].add(base));
        let d = _mm512_sub_pd(a, query[dim]);
        acc = L::dist_kn_f64x8(acc, d);
    }

    acc
}

#[inline(always)]
unsafe fn line_dists_avx2<L, const K: usize>(
    points: &[*const f64; K],
    query: &[__m256d; K],
    base: usize,
) -> __m256d
where
    L: Avx512F64LeafOps,
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
unsafe fn line_dists_avx128<L, const K: usize>(
    points: &[*const f64; K],
    query: &[__m128d; K],
    base: usize,
) -> __m128d
where
    L: Avx512F64LeafOps,
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
unsafe fn dist_scalar<L, const K: usize>(
    points: &[*const f64; K],
    query: &[f64; K],
    idx: usize,
) -> f64
where
    L: Avx512F64LeafOps,
{
    let mut dist = L::dist_k0_f64x1(*points[0].add(idx) - query[0]);
    for dim in 1..K {
        dist = L::dist_kn_f64x1(dist, *points[dim].add(idx) - query[dim]);
    }
    dist
}

#[inline(always)]
unsafe fn item_sorted_line_can_start<T: Content + PartialOrd>(
    items: *const T,
    base: usize,
    threshold_item: Option<T>,
) -> bool {
    let item = std::ptr::read_unaligned(items.add(base));
    let can_start = !threshold_item.is_some_and(|worst_item| item >= worst_item);
    if !can_start {
        #[cfg(any(feature = "exact_query_stats", feature = "test_utils"))]
        crate::results::exact_query_stats::record_item_sorted_threshold_stop();
        #[cfg(feature = "result_collection_stats")]
        crate::results::result_collection_stats::record_best_item_threshold_reject();
    }
    can_start
}

#[inline(always)]
unsafe fn emit_item_sorted_results_f64<T, F>(
    mut remaining: u32,
    dist_values: &[f64],
    items: *const T,
    base: usize,
    threshold_item: &mut Option<T>,
    emit: &mut F,
) -> bool
where
    T: Content + PartialOrd,
    F: FnMut(usize, f64, T) -> Option<T>,
{
    while remaining != 0 {
        let lane = remaining.trailing_zeros() as usize;
        let item = std::ptr::read_unaligned(items.add(base + lane));
        if threshold_item.is_some_and(|worst_item| item >= worst_item) {
            #[cfg(any(feature = "exact_query_stats", feature = "test_utils"))]
            crate::results::exact_query_stats::record_item_sorted_threshold_stop();
            #[cfg(feature = "result_collection_stats")]
            crate::results::result_collection_stats::record_best_item_threshold_reject();
            return false;
        }

        *threshold_item = emit(base + lane, *dist_values.get_unchecked(lane), item);
        remaining &= remaining - 1;
    }
    true
}

#[inline(always)]
unsafe fn emit_item_sorted_results_avx512<T, F, const EXCLUSIVE: bool>(
    dists: __m512d,
    items: *const T,
    base: usize,
    max_dist: f64,
    threshold_item: &mut Option<T>,
    emit: &mut F,
) -> bool
where
    T: Content + PartialOrd,
    F: FnMut(usize, f64, T) -> Option<T>,
{
    let mask = if EXCLUSIVE {
        _mm512_cmp_pd_mask(dists, _mm512_set1_pd(max_dist), _CMP_LT_OQ)
    } else {
        _mm512_cmp_pd_mask(dists, _mm512_set1_pd(max_dist), _CMP_LE_OQ)
    };
    if mask == 0 {
        return true;
    }

    let mut dist_values = [0.0f64; LINE_SIZE];
    _mm512_storeu_pd(dist_values.as_mut_ptr(), dists);
    emit_item_sorted_results_f64(mask as u32, &dist_values, items, base, threshold_item, emit)
}

#[inline(always)]
unsafe fn emit_item_sorted_results_avx2<T, F, const EXCLUSIVE: bool>(
    dists: __m256d,
    items: *const T,
    base: usize,
    max_dist: f64,
    threshold_item: &mut Option<T>,
    emit: &mut F,
) -> bool
where
    T: Content + PartialOrd,
    F: FnMut(usize, f64, T) -> Option<T>,
{
    let mask = if EXCLUSIVE {
        _mm256_movemask_pd(_mm256_cmp_pd(dists, _mm256_set1_pd(max_dist), _CMP_LT_OQ)) as u8
    } else {
        _mm256_movemask_pd(_mm256_cmp_pd(dists, _mm256_set1_pd(max_dist), _CMP_LE_OQ)) as u8
    };
    if mask == 0 {
        return true;
    }

    let mut dist_values = [0.0f64; AVX2_LINE_SIZE];
    _mm256_storeu_pd(dist_values.as_mut_ptr(), dists);
    emit_item_sorted_results_f64(mask as u32, &dist_values, items, base, threshold_item, emit)
}

#[inline(always)]
unsafe fn emit_item_sorted_results_avx128<T, F, const EXCLUSIVE: bool>(
    dists: __m128d,
    items: *const T,
    base: usize,
    max_dist: f64,
    threshold_item: &mut Option<T>,
    emit: &mut F,
) -> bool
where
    T: Content + PartialOrd,
    F: FnMut(usize, f64, T) -> Option<T>,
{
    let mask = _mm_movemask_pd(if EXCLUSIVE {
        _mm_cmplt_pd(dists, _mm_set1_pd(max_dist))
    } else {
        _mm_cmple_pd(dists, _mm_set1_pd(max_dist))
    }) as u8;
    if mask == 0 {
        return true;
    }

    let mut dist_values = [0.0f64; 2];
    _mm_storeu_pd(dist_values.as_mut_ptr(), dists);
    emit_item_sorted_results_f64(mask as u32, &dist_values, items, base, threshold_item, emit)
}

#[target_feature(enable = "avx512f,avx512vl,fma")]
unsafe fn best_n_within_item_sorted_avx512_raw<L, T, F, const EXCLUSIVE: bool, const K: usize>(
    points: [*const f64; K],
    items: *const T,
    len: usize,
    query: &[f64; K],
    max_dist: f64,
    mut threshold_item: Option<T>,
    emit: &mut F,
) -> (Option<T>, bool)
where
    L: Avx512F64LeafOps,
    T: Content + PartialOrd,
    F: FnMut(usize, f64, T) -> Option<T>,
{
    if len == 0 {
        return (threshold_item, false);
    }

    let query_512 = array_init(|dim| _mm512_set1_pd(query[dim]));
    let query_256 = array_init(|dim| _mm256_set1_pd(query[dim]));
    let query_128 = array_init(|dim| _mm_set1_pd(query[dim]));
    let mut base = 0usize;

    while base + LINE_SIZE <= len {
        if !item_sorted_line_can_start(items, base, threshold_item) {
            return (threshold_item, true);
        }
        #[cfg(any(feature = "exact_query_stats", feature = "test_utils"))]
        crate::results::exact_query_stats::record_leaf_items_distance_evaluated(LINE_SIZE);
        let dists = line_dists_avx512::<L, K>(&points, &query_512, base);
        if !emit_item_sorted_results_avx512::<_, _, EXCLUSIVE>(
            dists,
            items,
            base,
            max_dist,
            &mut threshold_item,
            emit,
        ) {
            return (threshold_item, true);
        }
        base += LINE_SIZE;
    }

    if base + AVX2_LINE_SIZE <= len {
        if !item_sorted_line_can_start(items, base, threshold_item) {
            return (threshold_item, true);
        }
        #[cfg(any(feature = "exact_query_stats", feature = "test_utils"))]
        crate::results::exact_query_stats::record_leaf_items_distance_evaluated(AVX2_LINE_SIZE);
        let dists = line_dists_avx2::<L, K>(&points, &query_256, base);
        if !emit_item_sorted_results_avx2::<_, _, EXCLUSIVE>(
            dists,
            items,
            base,
            max_dist,
            &mut threshold_item,
            emit,
        ) {
            return (threshold_item, true);
        }
        base += AVX2_LINE_SIZE;
    }

    if base + 2 <= len {
        if !item_sorted_line_can_start(items, base, threshold_item) {
            return (threshold_item, true);
        }
        #[cfg(any(feature = "exact_query_stats", feature = "test_utils"))]
        crate::results::exact_query_stats::record_leaf_items_distance_evaluated(2);
        let dists = line_dists_avx128::<L, K>(&points, &query_128, base);
        if !emit_item_sorted_results_avx128::<_, _, EXCLUSIVE>(
            dists,
            items,
            base,
            max_dist,
            &mut threshold_item,
            emit,
        ) {
            return (threshold_item, true);
        }
        base += 2;
    }

    while base < len {
        let item = std::ptr::read_unaligned(items.add(base));
        if threshold_item.is_some_and(|worst_item| item >= worst_item) {
            #[cfg(any(feature = "exact_query_stats", feature = "test_utils"))]
            crate::results::exact_query_stats::record_item_sorted_threshold_stop();
            #[cfg(feature = "result_collection_stats")]
            crate::results::result_collection_stats::record_best_item_threshold_reject();
            return (threshold_item, true);
        }
        #[cfg(any(feature = "exact_query_stats", feature = "test_utils"))]
        crate::results::exact_query_stats::record_leaf_items_distance_evaluated(1);
        let dist = dist_scalar::<L, K>(&points, query, base);
        let is_within_dist = if EXCLUSIVE {
            dist < max_dist
        } else {
            dist <= max_dist
        };
        if is_within_dist {
            threshold_item = emit(base, dist, item);
        }
        base += 1;
    }

    (threshold_item, false)
}

#[target_feature(enable = "avx512f,avx512vl,fma")]
unsafe fn nearest_n_within_avx512_raw<L, T, F, const EXCLUSIVE: bool, const K: usize>(
    points: [*const f64; K],
    items: *const T,
    len: usize,
    query: &[f64; K],
    max_dist: f64,
    emit: &mut F,
) where
    L: Avx512F64LeafOps,
    T: Content,
    F: FnMut(usize, f64, T),
{
    if len == 0 {
        return;
    }

    let query_512 = array_init(|dim| _mm512_set1_pd(query[dim]));
    let query_256 = array_init(|dim| _mm256_set1_pd(query[dim]));
    let query_128 = array_init(|dim| _mm_set1_pd(query[dim]));

    let full_chunks_len = len & !(CHUNK_SIZE - 1);
    let mut base = 0usize;

    while base != full_chunks_len {
        let d0 = line_dists_avx512::<L, K>(&points, &query_512, base);
        let d1 = line_dists_avx512::<L, K>(&points, &query_512, base + LINE_SIZE);
        let d2 = line_dists_avx512::<L, K>(&points, &query_512, base + 2 * LINE_SIZE);
        let d3 = line_dists_avx512::<L, K>(&points, &query_512, base + 3 * LINE_SIZE);

        emit_results_avx512::<_, _, EXCLUSIVE>(d0, items, base, max_dist, emit);
        emit_results_avx512::<_, _, EXCLUSIVE>(d1, items, base + LINE_SIZE, max_dist, emit);
        emit_results_avx512::<_, _, EXCLUSIVE>(d2, items, base + 2 * LINE_SIZE, max_dist, emit);
        emit_results_avx512::<_, _, EXCLUSIVE>(d3, items, base + 3 * LINE_SIZE, max_dist, emit);

        base += CHUNK_SIZE;
    }

    let full_lines_len = full_chunks_len + ((len - full_chunks_len) & !(LINE_SIZE - 1));
    while base != full_lines_len {
        let d0 = line_dists_avx512::<L, K>(&points, &query_512, base);
        emit_results_avx512::<_, _, EXCLUSIVE>(d0, items, base, max_dist, emit);
        base += LINE_SIZE;
    }

    if base + AVX2_LINE_SIZE <= len {
        let d0 = line_dists_avx2::<L, K>(&points, &query_256, base);
        emit_results_avx2::<_, _, EXCLUSIVE>(d0, items, base, max_dist, emit);
        base += AVX2_LINE_SIZE;
    }

    if base + 2 <= len {
        let d0 = line_dists_avx128::<L, K>(&points, &query_128, base);
        emit_results_avx128::<_, _, EXCLUSIVE>(d0, items, base, max_dist, emit);
        base += 2;
    }

    for idx in base..len {
        let dist = dist_scalar::<L, K>(&points, query, idx);
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

#[target_feature(enable = "avx512f,avx512vl,fma")]
pub(crate) unsafe fn nearest_n_within_avx512_unchecked<
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
    L: Avx512F64LeafOps,
    T: Content,
    F: FnMut(usize, f64, T),
{
    let points = leaf.points();
    let point_ptrs = array_init(|dim| points[dim].as_ptr());
    nearest_n_within_avx512_raw::<L, T, F, EXCLUSIVE, K>(
        point_ptrs,
        leaf.items().as_ptr(),
        leaf.items().len(),
        query,
        max_dist,
        emit,
    );
}

#[target_feature(enable = "avx512f,avx512vl,fma")]
pub(crate) unsafe fn nearest_n_within_avx512_arena_unchecked<
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
    L: Avx512F64LeafOps,
    T: Content,
    F: FnMut(usize, f64, T),
{
    let point_base = tile_base as *const f64;
    let point_ptrs = array_init(|dim| point_base.add(dim * len));
    let items = tile_base.add(K * len * std::mem::size_of::<f64>()) as *const T;

    nearest_n_within_avx512_raw::<L, T, F, EXCLUSIVE, K>(
        point_ptrs, items, len, query, max_dist, emit,
    );
}

#[target_feature(enable = "avx512f,avx512vl,fma")]
pub(crate) unsafe fn best_n_within_item_sorted_avx512_unchecked<
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
    threshold_item: Option<T>,
    emit: &mut F,
) -> (Option<T>, bool)
where
    L: Avx512F64LeafOps,
    T: Content + PartialOrd,
    F: FnMut(usize, f64, T) -> Option<T>,
{
    let points = leaf.points();
    let point_ptrs = array_init(|dim| points[dim].as_ptr());
    best_n_within_item_sorted_avx512_raw::<L, T, F, EXCLUSIVE, K>(
        point_ptrs,
        leaf.items().as_ptr(),
        leaf.items().len(),
        query,
        max_dist,
        threshold_item,
        emit,
    )
}

#[target_feature(enable = "avx512f,avx512vl,fma")]
pub(crate) unsafe fn best_n_within_item_sorted_avx512_arena_unchecked<
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
    threshold_item: Option<T>,
    emit: &mut F,
) -> (Option<T>, bool)
where
    L: Avx512F64LeafOps,
    T: Content + PartialOrd,
    F: FnMut(usize, f64, T) -> Option<T>,
{
    let point_base = tile_base as *const f64;
    let point_ptrs = array_init(|dim| point_base.add(dim * len));
    let items = tile_base.add(K * len * std::mem::size_of::<f64>()) as *const T;
    best_n_within_item_sorted_avx512_raw::<L, T, F, EXCLUSIVE, K>(
        point_ptrs,
        items,
        len,
        query,
        max_dist,
        threshold_item,
        emit,
    )
}

#[inline(always)]
unsafe fn emit_results_avx512_f32<T, F, const EXCLUSIVE: bool>(
    dists: __m512,
    items: *const T,
    base: usize,
    max_dist: f32,
    emit: &mut F,
) where
    T: Content,
    F: FnMut(usize, f32, T),
{
    let mask = if EXCLUSIVE {
        _mm512_cmp_ps_mask(dists, _mm512_set1_ps(max_dist), _CMP_LT_OQ) as u32
    } else {
        _mm512_cmp_ps_mask(dists, _mm512_set1_ps(max_dist), _CMP_LE_OQ) as u32
    };
    if mask == 0 {
        return;
    }

    let mut dist_values = [0.0f32; LINE_SIZE_F32];
    _mm512_storeu_ps(dist_values.as_mut_ptr(), dists);

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

    let mut dist_values = [0.0f32; AVX2_LINE_SIZE_F32];
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
unsafe fn emit_results_avx128_f32<T, F, const EXCLUSIVE: bool>(
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

    let mut dist_values = [0.0f32; SSE_LINE_SIZE_F32];
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
unsafe fn line_dists_avx512_f32<L, const K: usize>(
    points: &[*const f32; K],
    query: &[__m512; K],
    base: usize,
) -> __m512
where
    L: Avx512F32LeafOps,
{
    let a0 = _mm512_loadu_ps(points[0].add(base));
    let d0 = _mm512_sub_ps(a0, query[0]);
    let mut acc = L::dist_k0_f32x16(d0);

    for dim in 1..K {
        let a = _mm512_loadu_ps(points[dim].add(base));
        let d = _mm512_sub_ps(a, query[dim]);
        acc = L::dist_kn_f32x16(acc, d);
    }

    acc
}

#[inline(always)]
unsafe fn line_dists_avx2_f32<L, const K: usize>(
    points: &[*const f32; K],
    query: &[__m256; K],
    base: usize,
) -> __m256
where
    L: Avx512F32LeafOps,
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
unsafe fn line_dists_avx128_f32<L, const K: usize>(
    points: &[*const f32; K],
    query: &[__m128; K],
    base: usize,
) -> __m128
where
    L: Avx512F32LeafOps,
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
    L: Avx512F32LeafOps,
{
    let mut dist = L::dist_k0_f32x1(*points[0].add(idx) - query[0]);
    for dim in 1..K {
        dist = L::dist_kn_f32x1(dist, *points[dim].add(idx) - query[dim]);
    }
    dist
}

#[inline(always)]
unsafe fn emit_item_sorted_results_f32<T, F>(
    mut remaining: u32,
    dist_values: &[f32],
    items: *const T,
    base: usize,
    threshold_item: &mut Option<T>,
    emit: &mut F,
) -> bool
where
    T: Content + PartialOrd,
    F: FnMut(usize, f32, T) -> Option<T>,
{
    while remaining != 0 {
        let lane = remaining.trailing_zeros() as usize;
        let item = std::ptr::read_unaligned(items.add(base + lane));
        if threshold_item.is_some_and(|worst_item| item >= worst_item) {
            #[cfg(feature = "result_collection_stats")]
            crate::results::result_collection_stats::record_best_item_threshold_reject();
            return false;
        }

        *threshold_item = emit(base + lane, *dist_values.get_unchecked(lane), item);
        remaining &= remaining - 1;
    }
    true
}

#[inline(always)]
unsafe fn emit_item_sorted_results_avx512_f32<T, F, const EXCLUSIVE: bool>(
    dists: __m512,
    items: *const T,
    base: usize,
    max_dist: f32,
    threshold_item: &mut Option<T>,
    emit: &mut F,
) -> bool
where
    T: Content + PartialOrd,
    F: FnMut(usize, f32, T) -> Option<T>,
{
    let mask = if EXCLUSIVE {
        _mm512_cmp_ps_mask(dists, _mm512_set1_ps(max_dist), _CMP_LT_OQ)
    } else {
        _mm512_cmp_ps_mask(dists, _mm512_set1_ps(max_dist), _CMP_LE_OQ)
    };
    if mask == 0 {
        return true;
    }

    let mut dist_values = [0.0f32; LINE_SIZE_F32];
    _mm512_storeu_ps(dist_values.as_mut_ptr(), dists);
    emit_item_sorted_results_f32(mask as u32, &dist_values, items, base, threshold_item, emit)
}

#[inline(always)]
unsafe fn emit_item_sorted_results_avx2_f32<T, F, const EXCLUSIVE: bool>(
    dists: __m256,
    items: *const T,
    base: usize,
    max_dist: f32,
    threshold_item: &mut Option<T>,
    emit: &mut F,
) -> bool
where
    T: Content + PartialOrd,
    F: FnMut(usize, f32, T) -> Option<T>,
{
    let mask = if EXCLUSIVE {
        _mm256_movemask_ps(_mm256_cmp_ps(dists, _mm256_set1_ps(max_dist), _CMP_LT_OQ)) as u8
    } else {
        _mm256_movemask_ps(_mm256_cmp_ps(dists, _mm256_set1_ps(max_dist), _CMP_LE_OQ)) as u8
    };
    if mask == 0 {
        return true;
    }

    let mut dist_values = [0.0f32; AVX2_LINE_SIZE_F32];
    _mm256_storeu_ps(dist_values.as_mut_ptr(), dists);
    emit_item_sorted_results_f32(mask as u32, &dist_values, items, base, threshold_item, emit)
}

#[inline(always)]
unsafe fn emit_item_sorted_results_avx128_f32<T, F, const EXCLUSIVE: bool>(
    dists: __m128,
    items: *const T,
    base: usize,
    max_dist: f32,
    threshold_item: &mut Option<T>,
    emit: &mut F,
) -> bool
where
    T: Content + PartialOrd,
    F: FnMut(usize, f32, T) -> Option<T>,
{
    let mask = if EXCLUSIVE {
        _mm_movemask_ps(_mm_cmp_ps(dists, _mm_set1_ps(max_dist), _CMP_LT_OQ)) as u8
    } else {
        _mm_movemask_ps(_mm_cmp_ps(dists, _mm_set1_ps(max_dist), _CMP_LE_OQ)) as u8
    };
    if mask == 0 {
        return true;
    }

    let mut dist_values = [0.0f32; SSE_LINE_SIZE_F32];
    _mm_storeu_ps(dist_values.as_mut_ptr(), dists);
    emit_item_sorted_results_f32(mask as u32, &dist_values, items, base, threshold_item, emit)
}

#[target_feature(enable = "avx512f,avx512vl,fma")]
unsafe fn best_n_within_item_sorted_avx512_raw_f32<L, T, F, const EXCLUSIVE: bool, const K: usize>(
    points: [*const f32; K],
    items: *const T,
    len: usize,
    query: &[f32; K],
    max_dist: f32,
    mut threshold_item: Option<T>,
    emit: &mut F,
) -> (Option<T>, bool)
where
    L: Avx512F32LeafOps,
    T: Content + PartialOrd,
    F: FnMut(usize, f32, T) -> Option<T>,
{
    if len == 0 {
        return (threshold_item, false);
    }

    let query_512 = array_init(|dim| _mm512_set1_ps(query[dim]));
    let query_256 = array_init(|dim| _mm256_set1_ps(query[dim]));
    let query_128 = array_init(|dim| _mm_set1_ps(query[dim]));
    let mut base = 0usize;

    while base + LINE_SIZE_F32 <= len {
        if !item_sorted_line_can_start(items, base, threshold_item) {
            return (threshold_item, true);
        }
        let dists = line_dists_avx512_f32::<L, K>(&points, &query_512, base);
        if !emit_item_sorted_results_avx512_f32::<_, _, EXCLUSIVE>(
            dists,
            items,
            base,
            max_dist,
            &mut threshold_item,
            emit,
        ) {
            return (threshold_item, true);
        }
        base += LINE_SIZE_F32;
    }

    if base + AVX2_LINE_SIZE_F32 <= len {
        if !item_sorted_line_can_start(items, base, threshold_item) {
            return (threshold_item, true);
        }
        let dists = line_dists_avx2_f32::<L, K>(&points, &query_256, base);
        if !emit_item_sorted_results_avx2_f32::<_, _, EXCLUSIVE>(
            dists,
            items,
            base,
            max_dist,
            &mut threshold_item,
            emit,
        ) {
            return (threshold_item, true);
        }
        base += AVX2_LINE_SIZE_F32;
    }

    if base + SSE_LINE_SIZE_F32 <= len {
        if !item_sorted_line_can_start(items, base, threshold_item) {
            return (threshold_item, true);
        }
        let dists = line_dists_avx128_f32::<L, K>(&points, &query_128, base);
        if !emit_item_sorted_results_avx128_f32::<_, _, EXCLUSIVE>(
            dists,
            items,
            base,
            max_dist,
            &mut threshold_item,
            emit,
        ) {
            return (threshold_item, true);
        }
        base += SSE_LINE_SIZE_F32;
    }

    while base < len {
        let item = std::ptr::read_unaligned(items.add(base));
        if threshold_item.is_some_and(|worst_item| item >= worst_item) {
            #[cfg(feature = "result_collection_stats")]
            crate::results::result_collection_stats::record_best_item_threshold_reject();
            return (threshold_item, true);
        }
        let dist = dist_scalar_f32::<L, K>(&points, query, base);
        let is_within_dist = if EXCLUSIVE {
            dist < max_dist
        } else {
            dist <= max_dist
        };
        if is_within_dist {
            threshold_item = emit(base, dist, item);
        }
        base += 1;
    }

    (threshold_item, false)
}

#[target_feature(enable = "avx512f,avx512vl,fma")]
unsafe fn nearest_n_within_avx512_raw_f32<L, T, F, const EXCLUSIVE: bool, const K: usize>(
    points: [*const f32; K],
    items: *const T,
    len: usize,
    query: &[f32; K],
    max_dist: f32,
    emit: &mut F,
) where
    L: Avx512F32LeafOps,
    T: Content,
    F: FnMut(usize, f32, T),
{
    if len == 0 {
        return;
    }

    let query_512 = array_init(|dim| _mm512_set1_ps(query[dim]));
    let query_256 = array_init(|dim| _mm256_set1_ps(query[dim]));
    let query_128 = array_init(|dim| _mm_set1_ps(query[dim]));
    let mut base = 0usize;

    let full_chunks_len = len & !(CHUNK_SIZE_F32 - 1);
    while base != full_chunks_len {
        let d0 = line_dists_avx512_f32::<L, K>(&points, &query_512, base);
        let d1 = line_dists_avx512_f32::<L, K>(&points, &query_512, base + LINE_SIZE_F32);
        let d2 = line_dists_avx512_f32::<L, K>(&points, &query_512, base + 2 * LINE_SIZE_F32);
        let d3 = line_dists_avx512_f32::<L, K>(&points, &query_512, base + 3 * LINE_SIZE_F32);

        emit_results_avx512_f32::<_, _, EXCLUSIVE>(d0, items, base, max_dist, emit);
        emit_results_avx512_f32::<_, _, EXCLUSIVE>(d1, items, base + LINE_SIZE_F32, max_dist, emit);
        emit_results_avx512_f32::<_, _, EXCLUSIVE>(
            d2,
            items,
            base + 2 * LINE_SIZE_F32,
            max_dist,
            emit,
        );
        emit_results_avx512_f32::<_, _, EXCLUSIVE>(
            d3,
            items,
            base + 3 * LINE_SIZE_F32,
            max_dist,
            emit,
        );

        base += CHUNK_SIZE_F32;
    }

    let full_lines_len = full_chunks_len + ((len - full_chunks_len) & !(LINE_SIZE_F32 - 1));
    while base != full_lines_len {
        let d0 = line_dists_avx512_f32::<L, K>(&points, &query_512, base);
        emit_results_avx512_f32::<_, _, EXCLUSIVE>(d0, items, base, max_dist, emit);
        base += LINE_SIZE_F32;
    }

    if base + AVX2_LINE_SIZE_F32 <= len {
        let d0 = line_dists_avx2_f32::<L, K>(&points, &query_256, base);
        emit_results_avx2_f32::<_, _, EXCLUSIVE>(d0, items, base, max_dist, emit);
        base += AVX2_LINE_SIZE_F32;
    }

    if base + SSE_LINE_SIZE_F32 <= len {
        let d0 = line_dists_avx128_f32::<L, K>(&points, &query_128, base);
        emit_results_avx128_f32::<_, _, EXCLUSIVE>(d0, items, base, max_dist, emit);
        base += SSE_LINE_SIZE_F32;
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

#[target_feature(enable = "avx512f,avx512vl,fma")]
pub(crate) unsafe fn nearest_n_within_avx512_unchecked_f32<
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
    L: Avx512F32LeafOps,
    T: Content,
    F: FnMut(usize, f32, T),
{
    let points = leaf.points();
    let point_ptrs = array_init(|dim| points[dim].as_ptr());
    nearest_n_within_avx512_raw_f32::<L, T, F, EXCLUSIVE, K>(
        point_ptrs,
        leaf.items().as_ptr(),
        leaf.items().len(),
        query,
        max_dist,
        emit,
    );
}

#[target_feature(enable = "avx512f,avx512vl,fma")]
pub(crate) unsafe fn nearest_n_within_avx512_arena_unchecked_f32<
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
    L: Avx512F32LeafOps,
    T: Content,
    F: FnMut(usize, f32, T),
{
    let point_base = tile_base as *const f32;
    let point_ptrs = array_init(|dim| point_base.add(dim * len));
    let items = tile_base.add(K * len * std::mem::size_of::<f32>()) as *const T;

    nearest_n_within_avx512_raw_f32::<L, T, F, EXCLUSIVE, K>(
        point_ptrs, items, len, query, max_dist, emit,
    );
}

#[target_feature(enable = "avx512f,avx512vl,fma")]
pub(crate) unsafe fn best_n_within_item_sorted_avx512_unchecked_f32<
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
    threshold_item: Option<T>,
    emit: &mut F,
) -> (Option<T>, bool)
where
    L: Avx512F32LeafOps,
    T: Content + PartialOrd,
    F: FnMut(usize, f32, T) -> Option<T>,
{
    let points = leaf.points();
    let point_ptrs = array_init(|dim| points[dim].as_ptr());
    best_n_within_item_sorted_avx512_raw_f32::<L, T, F, EXCLUSIVE, K>(
        point_ptrs,
        leaf.items().as_ptr(),
        leaf.items().len(),
        query,
        max_dist,
        threshold_item,
        emit,
    )
}

#[target_feature(enable = "avx512f,avx512vl,fma")]
pub(crate) unsafe fn best_n_within_item_sorted_avx512_arena_unchecked_f32<
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
    threshold_item: Option<T>,
    emit: &mut F,
) -> (Option<T>, bool)
where
    L: Avx512F32LeafOps,
    T: Content + PartialOrd,
    F: FnMut(usize, f32, T) -> Option<T>,
{
    let point_base = tile_base as *const f32;
    let point_ptrs = array_init(|dim| point_base.add(dim * len));
    let items = tile_base.add(K * len * std::mem::size_of::<f32>()) as *const T;
    best_n_within_item_sorted_avx512_raw_f32::<L, T, F, EXCLUSIVE, K>(
        point_ptrs,
        items,
        len,
        query,
        max_dist,
        threshold_item,
        emit,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::{DistanceMetricAvx512, SquaredEuclidean};

    #[test]
    fn item_sorted_avx512_stops_inside_first_f64_line_at_item_threshold() {
        type Ops = <SquaredEuclidean<f64> as DistanceMetricAvx512<f64>>::Avx512F64Ops;
        let points = [(0..16).map(|idx| idx as f64).collect::<Vec<_>>()];
        let items = (0..16u32).collect::<Vec<_>>();
        let leaf = LeafView::<f64, u32, 1, 32>::new([&points[0]], &items);
        let mut emitted = Vec::new();
        let mut emit = |_, _distance, item| {
            emitted.push(item);
            Some(4)
        };

        let (_, stopped) = unsafe {
            best_n_within_item_sorted_avx512_unchecked::<Ops, _, _, false, 1, 32>(
                &leaf,
                &[0.0],
                f64::INFINITY,
                Some(4),
                &mut emit,
            )
        };

        assert!(stopped);
        assert_eq!(emitted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn item_sorted_avx512_stops_inside_first_f32_line_at_item_threshold() {
        type Ops = <SquaredEuclidean<f32> as DistanceMetricAvx512<f32>>::Avx512F32Ops;
        let points = [(0..32).map(|idx| idx as f32).collect::<Vec<_>>()];
        let items = (0..32u32).collect::<Vec<_>>();
        let leaf = LeafView::<f32, u32, 1, 32>::new([&points[0]], &items);
        let mut emitted = Vec::new();
        let mut emit = |_, _distance, item| {
            emitted.push(item);
            Some(7)
        };

        let (_, stopped) = unsafe {
            best_n_within_item_sorted_avx512_unchecked_f32::<Ops, _, _, false, 1, 32>(
                &leaf,
                &[0.0],
                f32::INFINITY,
                Some(7),
                &mut emit,
            )
        };

        assert!(stopped);
        assert_eq!(emitted, vec![0, 1, 2, 3, 4, 5, 6]);
    }
}

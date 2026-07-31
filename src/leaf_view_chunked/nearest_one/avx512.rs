#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use std::arch::x86_64::*;

use array_init::array_init;

use crate::dist::distance_metric_avx512::{Avx512F32LeafOps, Avx512F64LeafOps};
use crate::leaf_view::LeafView;
use crate::{Axis, Content};

const CHUNK_SIZE: usize = 32;
const LINE_SIZE: usize = 8;
const AVX2_LINE_SIZE: usize = 4;

#[repr(C)]
#[derive(Copy, Clone)]
struct BestResult<T: Content> {
    dist: f64,
    item: T,
}

#[cfg(feature = "leaf_nta_prefetch")]
#[inline(always)]
unsafe fn prefetch_nta(ptr: *const u8) {
    _mm_prefetch::<{ _MM_HINT_NTA }>(ptr as *const i8);
}

#[inline(always)]
unsafe fn update_best_chunk_avx512_raw<T: Content>(
    d0: __m512d,
    d1: __m512d,
    d2: __m512d,
    d3: __m512d,
    items: *const T,
    base: usize,
    best_dist: f64,
    best_item: T,
) -> (f64, T) {
    let bb = _mm512_set1_pd(best_dist);
    let m0 = _mm512_cmp_pd_mask(d0, bb, _CMP_LT_OQ);
    let m1 = _mm512_cmp_pd_mask(d1, bb, _CMP_LT_OQ);
    let m2 = _mm512_cmp_pd_mask(d2, bb, _CMP_LT_OQ);
    let m3 = _mm512_cmp_pd_mask(d3, bb, _CMP_LT_OQ);

    if (m0 | m1 | m2 | m3) == 0 {
        return (best_dist, best_item);
    }

    let min01 = _mm512_min_pd(d0, d1);
    let min23 = _mm512_min_pd(d2, d3);
    let min0123 = _mm512_min_pd(min01, min23);

    let hi256 = _mm512_extractf64x4_pd(min0123, 1);
    let lo256 = _mm512_castpd512_pd256(min0123);
    let min256 = _mm256_min_pd(lo256, hi256);

    let hi128 = _mm256_extractf128_pd(min256, 1);
    let lo128 = _mm256_castpd256_pd128(min256);
    let min128 = _mm_min_pd(lo128, hi128);

    let hi64 = _mm_unpackhi_pd(min128, min128);
    let min_scalar = _mm_min_sd(min128, hi64);
    let chunk_min = _mm_cvtsd_f64(min_scalar);

    let min_bcast = _mm512_set1_pd(chunk_min);
    let eq0 = _mm512_cmp_pd_mask(d0, min_bcast, _CMP_EQ_OQ);
    let eq1 = _mm512_cmp_pd_mask(d1, min_bcast, _CMP_EQ_OQ);
    let eq2 = _mm512_cmp_pd_mask(d2, min_bcast, _CMP_EQ_OQ);
    let eq3 = _mm512_cmp_pd_mask(d3, min_bcast, _CMP_EQ_OQ);

    let combined = (eq0 as u32) | ((eq1 as u32) << 8) | ((eq2 as u32) << 16) | ((eq3 as u32) << 24);
    core::hint::assert_unchecked(combined != 0);
    let idx = combined.trailing_zeros() as usize;

    (chunk_min, std::ptr::read_unaligned(items.add(base + idx)))
}

#[inline(always)]
unsafe fn update_best_line_avx512_raw<T: Content>(
    d0: __m512d,
    items: *const T,
    base: usize,
    best_dist: f64,
    best_item: T,
) -> (f64, T) {
    let bb = _mm512_set1_pd(best_dist);
    let m0 = _mm512_cmp_pd_mask(d0, bb, _CMP_LT_OQ);

    if m0 == 0 {
        return (best_dist, best_item);
    }

    let hi256 = _mm512_extractf64x4_pd(d0, 1);
    let lo256 = _mm512_castpd512_pd256(d0);
    let min256 = _mm256_min_pd(lo256, hi256);

    let hi128 = _mm256_extractf128_pd(min256, 1);
    let lo128 = _mm256_castpd256_pd128(min256);
    let min128 = _mm_min_pd(lo128, hi128);

    let hi64 = _mm_unpackhi_pd(min128, min128);
    let min_scalar = _mm_min_sd(min128, hi64);
    let chunk_min = _mm_cvtsd_f64(min_scalar);

    let min_bcast = _mm512_set1_pd(chunk_min);
    let eq0 = _mm512_cmp_pd_mask(d0, min_bcast, _CMP_EQ_OQ);

    core::hint::assert_unchecked(eq0 != 0);
    let idx = eq0.trailing_zeros() as usize;

    (chunk_min, std::ptr::read_unaligned(items.add(base + idx)))
}

#[inline(always)]
unsafe fn update_best_line_avx512_masked_raw<T: Content>(
    d0: __m512d,
    valid_mask: __mmask8,
    items: *const T,
    base: usize,
    best_dist: f64,
    best_item: T,
) -> (f64, T) {
    let inf = _mm512_set1_pd(f64::INFINITY);
    let masked = _mm512_mask_blend_pd(valid_mask, inf, d0);
    update_best_line_avx512_raw(masked, items, base, best_dist, best_item)
}

#[inline(always)]
unsafe fn update_best_line_avx2_raw<T: Content>(
    d0: __m256d,
    items: *const T,
    base: usize,
    best_dist: f64,
    best_item: T,
) -> (f64, T) {
    let bb = _mm256_set1_pd(best_dist);
    let m0 = _mm256_movemask_pd(_mm256_cmp_pd(d0, bb, _CMP_LT_OQ)) as u32;

    if m0 == 0 {
        return (best_dist, best_item);
    }

    let hi128 = _mm256_extractf128_pd(d0, 1);
    let lo128 = _mm256_castpd256_pd128(d0);
    let min128 = _mm_min_pd(lo128, hi128);

    let hi64 = _mm_unpackhi_pd(min128, min128);
    let min_scalar = _mm_min_sd(min128, hi64);
    let chunk_min = _mm_cvtsd_f64(min_scalar);

    let bcast = _mm256_set1_pd(chunk_min);
    let eq0 = _mm256_movemask_pd(_mm256_cmp_pd(d0, bcast, _CMP_EQ_OQ)) as u32;
    core::hint::assert_unchecked(eq0 != 0);
    let idx = eq0.trailing_zeros() as usize;

    (chunk_min, std::ptr::read_unaligned(items.add(base + idx)))
}

macro_rules! impl_leaf_kernel_k {
    ($extern_name:ident, $k:expr, [$(($dim:literal, $p:ident, $qs:ident, $qv:ident)),*]) => {
        #[target_feature(enable = "avx512f,avx512vl,fma")]
        unsafe fn $extern_name<AX, L, T>(
            points: *const *const f64,
            items: *const T,
            len: usize,
            query: *const f64,
            mut best_dist: f64,
            mut best_item: T,
        ) -> BestResult<T>
        where
            AX: Axis<Coord = AX>,
            L: Avx512F64LeafOps,
            T: Content,
        {
            let p0 = *points.add(0);
            let q0s = *query.add(0);
            $(
                let $p = *points.add($dim);
                let $qs = *query.add($dim);
            )*

            let qv0 = _mm512_set1_pd(q0s);
            $(
                let $qv = _mm512_set1_pd($qs);
            )*

            let full_chunks_len = len & !(CHUNK_SIZE - 1);
            let mut base = 0usize;
            while base != full_chunks_len {
                macro_rules! chunk {
                    ($off:expr) => {{
                        let a0 = _mm512_loadu_pd(p0.add(base + $off * 8));
                        let d0 = _mm512_sub_pd(a0, qv0);
                        #[allow(unused_mut)]
                        let mut acc = L::dist_k0_f64x8(d0);
                        $(
                            let a = _mm512_loadu_pd($p.add(base + $off * 8));
                            let d = _mm512_sub_pd(a, $qv);
                            acc = L::dist_kn_f64x8(acc, d);
                        )*
                        acc
                    }};
                }

                let d0 = chunk!(0);
                let d1 = chunk!(1);
                let d2 = chunk!(2);
                let d3 = chunk!(3);

                (best_dist, best_item) =
                    update_best_chunk_avx512_raw(d0, d1, d2, d3, items, base, best_dist, best_item);

                base += CHUNK_SIZE;
            }

            let full_lines_len = full_chunks_len + ((len - full_chunks_len) & !(LINE_SIZE - 1));
            while base != full_lines_len {
                let a0 = _mm512_loadu_pd(p0.add(base));
                let d0 = _mm512_sub_pd(a0, qv0);
                #[allow(unused_mut)]
                let mut acc = L::dist_k0_f64x8(d0);

                $(
                    let a = _mm512_loadu_pd($p.add(base));
                    let d = _mm512_sub_pd(a, $qv);
                    acc = L::dist_kn_f64x8(acc, d);
                )*

                (best_dist, best_item) =
                    update_best_line_avx512_raw(acc, items, base, best_dist, best_item);

                base += LINE_SIZE;
            }

            if base + AVX2_LINE_SIZE <= len {
                let qy0 = _mm512_castpd512_pd256(qv0);
                let a0 = _mm256_loadu_pd(p0.add(base));
                let d0 = _mm256_sub_pd(a0, qy0);
                #[allow(unused_mut)]
                let mut acc = L::dist_k0_f64x4(d0);

                $(
                    let qy = _mm512_castpd512_pd256($qv);
                    let a = _mm256_loadu_pd($p.add(base));
                    let d = _mm256_sub_pd(a, qy);
                    acc = L::dist_kn_f64x4(acc, d);
                )*

                (best_dist, best_item) =
                    update_best_line_avx2_raw(acc, items, base, best_dist, best_item);

                base += AVX2_LINE_SIZE;
            }

            for idx in base..len {
                #[allow(unused_mut)]
                let mut d = L::dist_k0_f64x1(*p0.add(idx) - q0s);
                $(
                    d = L::dist_kn_f64x1(d, *$p.add(idx) - $qs);
                )*
                if d < best_dist {
                    best_dist = d;
                    best_item = std::ptr::read_unaligned(items.add(idx));
                }
            }

            BestResult { dist: best_dist, item: best_item }
        }
    };
}

impl_leaf_kernel_k!(leaf_nearest_one_chunked_nozero_f64_k1, 1, []);
impl_leaf_kernel_k!(
    leaf_nearest_one_chunked_nozero_f64_k2,
    2,
    [(1, p1, q1s, qv1)]
);
impl_leaf_kernel_k!(
    leaf_nearest_one_chunked_nozero_f64_k3,
    3,
    [(1, p1, q1s, qv1), (2, p2, q2s, qv2)]
);
impl_leaf_kernel_k!(
    leaf_nearest_one_chunked_nozero_f64_k4,
    4,
    [(1, p1, q1s, qv1), (2, p2, q2s, qv2), (3, p3, q3s, qv3)]
);
impl_leaf_kernel_k!(
    leaf_nearest_one_chunked_nozero_f64_k5,
    5,
    [
        (1, p1, q1s, qv1),
        (2, p2, q2s, qv2),
        (3, p3, q3s, qv3),
        (4, p4, q4s, qv4)
    ]
);
impl_leaf_kernel_k!(
    leaf_nearest_one_chunked_nozero_f64_k6,
    6,
    [
        (1, p1, q1s, qv1),
        (2, p2, q2s, qv2),
        (3, p3, q3s, qv3),
        (4, p4, q4s, qv4),
        (5, p5, q5s, qv5)
    ]
);
impl_leaf_kernel_k!(
    leaf_nearest_one_chunked_nozero_f64_k7,
    7,
    [
        (1, p1, q1s, qv1),
        (2, p2, q2s, qv2),
        (3, p3, q3s, qv3),
        (4, p4, q4s, qv4),
        (5, p5, q5s, qv5),
        (6, p6, q6s, qv6)
    ]
);
impl_leaf_kernel_k!(
    leaf_nearest_one_chunked_nozero_f64_k8,
    8,
    [
        (1, p1, q1s, qv1),
        (2, p2, q2s, qv2),
        (3, p3, q3s, qv3),
        (4, p4, q4s, qv4),
        (5, p5, q5s, qv5),
        (6, p6, q6s, qv6),
        (7, p7, q7s, qv7)
    ]
);

macro_rules! impl_leaf_arena_kernel_k {
    ($extern_name:ident, $k:expr, [$(($dim:literal, $p:ident, $qs:ident, $qv:ident, $qy:ident, $qx:ident)),*]) => {
        #[target_feature(enable = "avx512f,avx512vl,fma")]
        pub(crate) unsafe fn $extern_name<AX, L, T>(
            tile_base: *const u8,
            len: usize,
            query: *const f64,
            best_dist: &mut f64,
            best_item: &mut T,
        )
        where
            AX: Axis<Coord = AX>,
            L: Avx512F64LeafOps,
            T: Content,
        {
            if len == 0 {
                return;
            }

            let p0 = tile_base as *const f64;
            let items = tile_base.add($k * len * std::mem::size_of::<f64>()) as *const T;
            let q0s = *query.add(0);
            $(
                let $p = p0.add($dim * len);
                let $qs = *query.add($dim);
            )*

            let qv0 = _mm512_set1_pd(q0s);
            $(
                let $qv = _mm512_set1_pd($qs);
            )*

            let mut best_dist_val = *best_dist;
            let mut best_item_val = *best_item;
            let mut base = 0usize;

            if base + CHUNK_SIZE <= len {
                #[cfg(feature = "leaf_nta_prefetch")]
                {
                    // perf shows that the "base + 0" prefetches are
                    // too late to be useful but still seem to have some
                    // benefit.
                    // TODO: revisit this once we can ZC-deserealize a tree
                    //       via rkyv so that we get less noise in the perf output

                    // prefetch_nta(p0.add(base) as *const u8);
                    prefetch_nta(p0.add(base + 8) as *const u8);
                    prefetch_nta(p0.add(base + 16) as *const u8);
                    prefetch_nta(p0.add(base + 24) as *const u8);
                    $(
                        // prefetch_nta($p.add(base) as *const u8);
                        prefetch_nta($p.add(base + 8) as *const u8);
                        prefetch_nta($p.add(base + 16) as *const u8);
                        prefetch_nta($p.add(base + 24) as *const u8);
                    )*
                }

                let x0 = _mm512_loadu_pd(p0.add(base));
                let x1 = _mm512_loadu_pd(p0.add(base + 8));
                let x2 = _mm512_loadu_pd(p0.add(base + 16));
                let x3 = _mm512_loadu_pd(p0.add(base + 24));

                let dx0 = _mm512_sub_pd(x0, qv0);
                let dx1 = _mm512_sub_pd(x1, qv0);
                let dx2 = _mm512_sub_pd(x2, qv0);
                let dx3 = _mm512_sub_pd(x3, qv0);

                #[allow(unused_mut)]
                let mut d0 = L::dist_k0_f64x8(dx0);
                #[allow(unused_mut)]
                let mut d1 = L::dist_k0_f64x8(dx1);
                #[allow(unused_mut)]
                let mut d2 = L::dist_k0_f64x8(dx2);
                #[allow(unused_mut)]
                let mut d3 = L::dist_k0_f64x8(dx3);

                $(
                    let y0 = _mm512_loadu_pd($p.add(base));
                    let y1 = _mm512_loadu_pd($p.add(base + 8));
                    let y2 = _mm512_loadu_pd($p.add(base + 16));
                    let y3 = _mm512_loadu_pd($p.add(base + 24));

                    let dy0 = _mm512_sub_pd(y0, $qv);
                    let dy1 = _mm512_sub_pd(y1, $qv);
                    let dy2 = _mm512_sub_pd(y2, $qv);
                    let dy3 = _mm512_sub_pd(y3, $qv);

                    d0 = L::dist_kn_f64x8(d0, dy0);
                    d1 = L::dist_kn_f64x8(d1, dy1);
                    d2 = L::dist_kn_f64x8(d2, dy2);
                    d3 = L::dist_kn_f64x8(d3, dy3);
                )*

                (best_dist_val, best_item_val) = update_best_chunk_avx512_raw(
                    d0,
                    d1,
                    d2,
                    d3,
                    items,
                    base,
                    best_dist_val,
                    best_item_val,
                );

                base += CHUNK_SIZE;
            }

            if base < len {
                let remaining = len - base;
                let tail_mask = ((1u16 << remaining) - 1) as __mmask8;

                #[cfg(feature = "leaf_nta_prefetch")]
                {
                    prefetch_nta(p0.add(base) as *const u8);
                    $(
                        prefetch_nta($p.add(base) as *const u8);
                    )*
                }

                let a0 = _mm512_maskz_loadu_pd(tail_mask, p0.add(base));
                let d0 = _mm512_sub_pd(a0, qv0);
                #[allow(unused_mut)]
                let mut acc = L::dist_k0_f64x8(d0);

                $(
                    let a = _mm512_maskz_loadu_pd(tail_mask, $p.add(base));
                    let d = _mm512_sub_pd(a, $qv);
                    acc = L::dist_kn_f64x8(acc, d);
                )*

                (best_dist_val, best_item_val) = update_best_line_avx512_masked_raw(
                    acc,
                    tail_mask,
                    items,
                    base,
                    best_dist_val,
                    best_item_val,
                );
            }

            *best_dist = best_dist_val;
            *best_item = best_item_val;
        }
    };
}

impl_leaf_arena_kernel_k!(leaf_nearest_one_arena_nozero_f64_k1, 1, []);
impl_leaf_arena_kernel_k!(
    leaf_nearest_one_arena_nozero_f64_k2,
    2,
    [(1, p1, q1s, qv1, qy1, qx1)]
);
impl_leaf_arena_kernel_k!(
    leaf_nearest_one_arena_nozero_f64_k3,
    3,
    [(1, p1, q1s, qv1, qy1, qx1), (2, p2, q2s, qv2, qy2, qx2)]
);
impl_leaf_arena_kernel_k!(
    leaf_nearest_one_arena_nozero_f64_k4,
    4,
    [
        (1, p1, q1s, qv1, qy1, qx1),
        (2, p2, q2s, qv2, qy2, qx2),
        (3, p3, q3s, qv3, qy3, qx3)
    ]
);

#[inline(always)]
unsafe fn scalar_fallback_dynamic<AX, L, T>(
    points: *const *const f64,
    items: *const T,
    len: usize,
    k: usize,
    query: *const f64,
    mut best_dist: f64,
    mut best_item: T,
) -> BestResult<T>
where
    AX: Axis<Coord = AX>,
    L: Avx512F64LeafOps,
    T: Content,
{
    let points = std::slice::from_raw_parts(points, k);
    let items = std::slice::from_raw_parts(items, len);
    let query = std::slice::from_raw_parts(query, k);

    for idx in 0..len {
        let mut d =
            L::dist_k0_f64x1(*(*points.get_unchecked(0)).add(idx) - *query.get_unchecked(0));
        for dim in 1..k {
            d = L::dist_kn_f64x1(
                d,
                *(*points.get_unchecked(dim)).add(idx) - *query.get_unchecked(dim),
            );
        }
        if d < best_dist {
            best_dist = d;
            best_item = *items.get_unchecked(idx);
        }
    }

    BestResult {
        dist: best_dist,
        item: best_item,
    }
}

#[inline(always)]
unsafe fn scalar_fallback_arena_dynamic<AX, L, T>(
    tile_base: *const u8,
    len: usize,
    k: usize,
    query: *const f64,
    best_dist: &mut f64,
    best_item: &mut T,
) where
    AX: Axis<Coord = AX>,
    L: Avx512F64LeafOps,
    T: Content,
{
    let points = tile_base as *const f64;
    let items = tile_base.add(k * len * std::mem::size_of::<f64>()) as *const T;
    let mut best_dist_val = *best_dist;
    let mut best_item_val = *best_item;

    for idx in 0..len {
        let mut d = L::dist_k0_f64x1(*points.add(idx) - *query.add(0));
        for dim in 1..k {
            let axis = points.add(dim * len);
            d = L::dist_kn_f64x1(d, *axis.add(idx) - *query.add(dim));
        }
        if d < best_dist_val {
            best_dist_val = d;
            best_item_val = std::ptr::read_unaligned(items.add(idx));
        }
    }

    *best_dist = best_dist_val;
    *best_item = best_item_val;
}

#[inline(always)]
unsafe fn leaf_nearest_one_chunked_nozero_f64_selector<AX, L, T>(
    k: usize,
    points: *const *const f64,
    items: *const T,
    len: usize,
    query: *const f64,
    best_dist_in: f64,
    best_item_in: T,
) -> BestResult<T>
where
    AX: Axis<Coord = AX>,
    L: Avx512F64LeafOps,
    T: Content,
{
    match k {
        1 => leaf_nearest_one_chunked_nozero_f64_k1::<AX, L, T>(
            points,
            items,
            len,
            query,
            best_dist_in,
            best_item_in,
        ),
        2 => leaf_nearest_one_chunked_nozero_f64_k2::<AX, L, T>(
            points,
            items,
            len,
            query,
            best_dist_in,
            best_item_in,
        ),
        3 => leaf_nearest_one_chunked_nozero_f64_k3::<AX, L, T>(
            points,
            items,
            len,
            query,
            best_dist_in,
            best_item_in,
        ),
        4 => leaf_nearest_one_chunked_nozero_f64_k4::<AX, L, T>(
            points,
            items,
            len,
            query,
            best_dist_in,
            best_item_in,
        ),
        5 => leaf_nearest_one_chunked_nozero_f64_k5::<AX, L, T>(
            points,
            items,
            len,
            query,
            best_dist_in,
            best_item_in,
        ),
        6 => leaf_nearest_one_chunked_nozero_f64_k6::<AX, L, T>(
            points,
            items,
            len,
            query,
            best_dist_in,
            best_item_in,
        ),
        7 => leaf_nearest_one_chunked_nozero_f64_k7::<AX, L, T>(
            points,
            items,
            len,
            query,
            best_dist_in,
            best_item_in,
        ),
        8 => leaf_nearest_one_chunked_nozero_f64_k8::<AX, L, T>(
            points,
            items,
            len,
            query,
            best_dist_in,
            best_item_in,
        ),
        _ => scalar_fallback_dynamic::<AX, L, T>(
            points,
            items,
            len,
            k,
            query,
            best_dist_in,
            best_item_in,
        ),
    }
}

#[inline(always)]
unsafe fn leaf_nearest_one_arena_nozero_f64_selector<AX, L, T>(
    k: usize,
    tile_base: *const u8,
    len: usize,
    query: *const f64,
    best_dist: &mut f64,
    best_item: &mut T,
) where
    AX: Axis<Coord = AX>,
    L: Avx512F64LeafOps,
    T: Content,
{
    match k {
        1 => leaf_nearest_one_arena_nozero_f64_k1::<AX, L, T>(
            tile_base, len, query, best_dist, best_item,
        ),
        2 => leaf_nearest_one_arena_nozero_f64_k2::<AX, L, T>(
            tile_base, len, query, best_dist, best_item,
        ),
        3 => leaf_nearest_one_arena_nozero_f64_k3::<AX, L, T>(
            tile_base, len, query, best_dist, best_item,
        ),
        4 => leaf_nearest_one_arena_nozero_f64_k4::<AX, L, T>(
            tile_base, len, query, best_dist, best_item,
        ),
        _ => scalar_fallback_arena_dynamic::<AX, L, T>(
            tile_base, len, k, query, best_dist, best_item,
        ),
    }
}

#[inline(always)]
pub(crate) unsafe fn nearest_one_avx512_raw_unchecked<AX, L, T, const K: usize>(
    points: [*const AX; K],
    items: *const T,
    len: usize,
    query_wide: &[AX; K],
    best_dist: &mut AX,
    best_item: &mut T,
) where
    AX: Axis<Coord = AX>,
    L: Avx512F64LeafOps,
    T: Content,
{
    if len == 0 {
        return;
    }

    let points_ptrs: [*const f64; K] = std::array::from_fn(|dim| points[dim] as *const f64);
    let query_ptr = query_wide.as_ptr() as *const f64;
    let best_dist_ptr = best_dist as *mut AX as *mut f64;

    let result = leaf_nearest_one_chunked_nozero_f64_selector::<AX, L, T>(
        K,
        points_ptrs.as_ptr(),
        items,
        len,
        query_ptr,
        *best_dist_ptr,
        *best_item,
    );

    *best_dist_ptr = result.dist;
    *best_item = result.item;
}

#[inline(always)]
pub(crate) unsafe fn nearest_one_avx512_arena_unchecked<AX, L, T, const K: usize>(
    tile_base: *const u8,
    len: usize,
    query: *const f64,
    best_dist: &mut f64,
    best_item: &mut T,
) where
    AX: Axis<Coord = AX>,
    L: Avx512F64LeafOps,
    T: Content,
{
    leaf_nearest_one_arena_nozero_f64_selector::<AX, L, T>(
        K, tile_base, len, query, best_dist, best_item,
    );
}

#[inline(always)]
pub(crate) unsafe fn nearest_one_avx512_unchecked<AX, L, T, const K: usize, const B: usize>(
    leaf: &LeafView<'_, AX, T, K, B>,
    query_wide: &[AX; K],
    best_dist: &mut AX,
    best_item: &mut T,
) where
    AX: Axis<Coord = AX>,
    L: Avx512F64LeafOps,
    T: Content,
{
    let points = leaf.points();
    let point_ptrs = std::array::from_fn(|dim| points[dim].as_ptr());

    nearest_one_avx512_raw_unchecked::<AX, L, T, K>(
        point_ptrs,
        leaf.items().as_ptr(),
        leaf.items().len(),
        query_wide,
        best_dist,
        best_item,
    );
}

#[cfg(all(feature = "cargo_asm", feature = "simd", target_arch = "x86_64"))]
#[allow(dead_code)]
#[doc(hidden)]
#[inline(never)]
#[unsafe(no_mangle)]
pub fn v6_nearest_one_arena_leaf_cargo_asm_hook(
    tile_base: *const u8,
    len: usize,
    query: &[f64; 3],
    best_dist: &mut f64,
    best_item: &mut usize,
) {
    unsafe {
        leaf_nearest_one_arena_nozero_f64_k3::<
            f64,
            <crate::dist::SquaredEuclidean<f64> as crate::dist::DistanceMetricAvx512<f64>>::Avx512F64Ops,
            usize,
        >(tile_base, len, query.as_ptr(), best_dist, best_item);
    }
}

const CHUNK_SIZE_F32: usize = 64;
const LINE_SIZE_F32: usize = 16;
const AVX2_LINE_SIZE_F32: usize = 8;
const SSE_LINE_SIZE_F32: usize = 4;

#[inline(always)]
unsafe fn update_best_chunk_avx512_raw_f32<T: Content>(
    d0: __m512,
    d1: __m512,
    d2: __m512,
    d3: __m512,
    items: *const T,
    base: usize,
    best_dist: &mut f32,
    best_item: &mut T,
) {
    let current_best = *best_dist;
    let bb = _mm512_set1_ps(current_best);
    let m0 = _mm512_cmp_ps_mask(d0, bb, _CMP_LT_OQ);
    let m1 = _mm512_cmp_ps_mask(d1, bb, _CMP_LT_OQ);
    let m2 = _mm512_cmp_ps_mask(d2, bb, _CMP_LT_OQ);
    let m3 = _mm512_cmp_ps_mask(d3, bb, _CMP_LT_OQ);

    if (m0 | m1 | m2 | m3) == 0 {
        return;
    }

    // Reduce only lanes that can improve the incumbent.  Besides avoiding
    // unnecessary work on non-candidates, this gives NaNs the same ordered
    // comparison semantics as the scalar fallback: they cannot win.
    let inf = _mm512_set1_ps(f32::INFINITY);
    let candidate0 = _mm512_mask_blend_ps(m0, inf, d0);
    let candidate1 = _mm512_mask_blend_ps(m1, inf, d1);
    let candidate2 = _mm512_mask_blend_ps(m2, inf, d2);
    let candidate3 = _mm512_mask_blend_ps(m3, inf, d3);
    let min01 = _mm512_min_ps(candidate0, candidate1);
    let min23 = _mm512_min_ps(candidate2, candidate3);
    let min0123 = _mm512_min_ps(min01, min23);
    let chunk_min = reduce_min_avx512_f32(min0123);

    let min_bcast = _mm512_set1_ps(chunk_min);
    let eq0 = _mm512_cmp_ps_mask(d0, min_bcast, _CMP_EQ_OQ);
    let eq1 = _mm512_cmp_ps_mask(d1, min_bcast, _CMP_EQ_OQ);
    let eq2 = _mm512_cmp_ps_mask(d2, min_bcast, _CMP_EQ_OQ);
    let eq3 = _mm512_cmp_ps_mask(d3, min_bcast, _CMP_EQ_OQ);
    let combined =
        (eq0 as u64) | ((eq1 as u64) << 16) | ((eq2 as u64) << 32) | ((eq3 as u64) << 48);
    core::hint::assert_unchecked(combined != 0);
    let idx = combined.trailing_zeros() as usize;

    *best_dist = chunk_min;
    *best_item = std::ptr::read_unaligned(items.add(base + idx));
}

#[inline(always)]
unsafe fn update_best_line_avx512_raw_f32<T: Content>(
    d0: __m512,
    items: *const T,
    base: usize,
    best_dist: &mut f32,
    best_item: &mut T,
) {
    let current_best = *best_dist;
    let bb = _mm512_set1_ps(current_best);
    let m0 = _mm512_cmp_ps_mask(d0, bb, _CMP_LT_OQ);

    if m0 == 0 {
        return;
    }

    let inf = _mm512_set1_ps(f32::INFINITY);
    let chunk_min = reduce_min_avx512_f32(_mm512_mask_blend_ps(m0, inf, d0));
    let min_bcast = _mm512_set1_ps(chunk_min);
    let eq0 = _mm512_cmp_ps_mask(d0, min_bcast, _CMP_EQ_OQ);
    core::hint::assert_unchecked(eq0 != 0);
    let idx = eq0.trailing_zeros() as usize;

    *best_dist = chunk_min;
    *best_item = std::ptr::read_unaligned(items.add(base + idx));
}

#[inline(always)]
unsafe fn update_best_line_avx2_raw_f32<T: Content>(
    d0: __m256,
    items: *const T,
    base: usize,
    best_dist: &mut f32,
    best_item: &mut T,
) {
    let current_best = *best_dist;
    let bb = _mm256_set1_ps(current_best);
    let cmp = _mm256_cmp_ps(d0, bb, _CMP_LT_OQ);
    let m0 = _mm256_movemask_ps(cmp) as u32;

    if m0 == 0 {
        return;
    }

    let inf = _mm256_set1_ps(f32::INFINITY);
    let chunk_min = reduce_min_avx2_f32(_mm256_blendv_ps(inf, d0, cmp));
    let min_bcast = _mm256_set1_ps(chunk_min);
    let eq0 = _mm256_movemask_ps(_mm256_cmp_ps(d0, min_bcast, _CMP_EQ_OQ)) as u32;
    core::hint::assert_unchecked(eq0 != 0);
    let idx = eq0.trailing_zeros() as usize;

    *best_dist = chunk_min;
    *best_item = std::ptr::read_unaligned(items.add(base + idx));
}

#[inline(always)]
unsafe fn update_best_line_avx128_raw_f32<T: Content>(
    d0: __m128,
    items: *const T,
    base: usize,
    best_dist: &mut f32,
    best_item: &mut T,
) {
    let current_best = *best_dist;
    let bb = _mm_set1_ps(current_best);
    let cmp = _mm_cmplt_ps(d0, bb);
    let m0 = _mm_movemask_ps(cmp) as u32;

    if m0 == 0 {
        return;
    }

    let inf = _mm_set1_ps(f32::INFINITY);
    let chunk_min = reduce_min_avx128_f32(_mm_blendv_ps(inf, d0, cmp));
    let min_bcast = _mm_set1_ps(chunk_min);
    let eq0 = _mm_movemask_ps(_mm_cmpeq_ps(d0, min_bcast)) as u32;
    core::hint::assert_unchecked(eq0 != 0);
    let idx = eq0.trailing_zeros() as usize;

    *best_dist = chunk_min;
    *best_item = std::ptr::read_unaligned(items.add(base + idx));
}

#[inline(always)]
unsafe fn reduce_min_avx512_f32(values: __m512) -> f32 {
    let hi256 = _mm512_extractf32x8_ps(values, 1);
    let lo256 = _mm512_castps512_ps256(values);
    reduce_min_avx2_f32(_mm256_min_ps(lo256, hi256))
}

#[inline(always)]
unsafe fn reduce_min_avx2_f32(values: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(values, 1);
    let lo128 = _mm256_castps256_ps128(values);
    reduce_min_avx128_f32(_mm_min_ps(lo128, hi128))
}

#[inline(always)]
unsafe fn reduce_min_avx128_f32(values: __m128) -> f32 {
    let hi64 = _mm_movehl_ps(values, values);
    let min2 = _mm_min_ps(values, hi64);
    let other = _mm_shuffle_ps(min2, min2, 0b01_01_01_01);
    _mm_cvtss_f32(_mm_min_ss(min2, other))
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

#[target_feature(enable = "avx512f,avx512vl,fma")]
unsafe fn nearest_one_avx512_raw_unchecked_f32<L, T, const K: usize>(
    points: [*const f32; K],
    items: *const T,
    len: usize,
    query_wide: &[f32; K],
    best_dist: &mut f32,
    best_item: &mut T,
) where
    L: Avx512F32LeafOps,
    T: Content,
{
    if len == 0 {
        return;
    }

    let query_512 = array_init(|dim| _mm512_set1_ps(query_wide[dim]));
    let query_256 = array_init(|dim| _mm256_set1_ps(query_wide[dim]));
    let query_128 = array_init(|dim| _mm_set1_ps(query_wide[dim]));
    let mut base = 0usize;

    let full_chunks_len = len & !(CHUNK_SIZE_F32 - 1);
    while base != full_chunks_len {
        let d0 = line_dists_avx512_f32::<L, K>(&points, &query_512, base);
        let d1 = line_dists_avx512_f32::<L, K>(&points, &query_512, base + LINE_SIZE_F32);
        let d2 = line_dists_avx512_f32::<L, K>(&points, &query_512, base + 2 * LINE_SIZE_F32);
        let d3 = line_dists_avx512_f32::<L, K>(&points, &query_512, base + 3 * LINE_SIZE_F32);
        update_best_chunk_avx512_raw_f32(d0, d1, d2, d3, items, base, best_dist, best_item);
        base += CHUNK_SIZE_F32;
    }

    let full_lines_len = full_chunks_len + ((len - full_chunks_len) & !(LINE_SIZE_F32 - 1));
    while base != full_lines_len {
        let d0 = line_dists_avx512_f32::<L, K>(&points, &query_512, base);
        update_best_line_avx512_raw_f32(d0, items, base, best_dist, best_item);
        base += LINE_SIZE_F32;
    }

    if base + AVX2_LINE_SIZE_F32 <= len {
        let d0 = line_dists_avx2_f32::<L, K>(&points, &query_256, base);
        update_best_line_avx2_raw_f32(d0, items, base, best_dist, best_item);
        base += AVX2_LINE_SIZE_F32;
    }

    if base + SSE_LINE_SIZE_F32 <= len {
        let d0 = line_dists_avx128_f32::<L, K>(&points, &query_128, base);
        update_best_line_avx128_raw_f32(d0, items, base, best_dist, best_item);
        base += SSE_LINE_SIZE_F32;
    }

    while base < len {
        let dist = dist_scalar_f32::<L, K>(&points, query_wide, base);
        if dist < *best_dist {
            *best_dist = dist;
            *best_item = std::ptr::read_unaligned(items.add(base));
        }
        base += 1;
    }
}

#[inline(always)]
pub(crate) unsafe fn nearest_one_avx512_arena_unchecked_f32<L, T, const K: usize>(
    tile_base: *const u8,
    len: usize,
    query: *const f32,
    best_dist: &mut f32,
    best_item: &mut T,
) where
    L: Avx512F32LeafOps,
    T: Content,
{
    let point_base = tile_base as *const f32;
    let point_ptrs = std::array::from_fn(|dim| point_base.add(dim * len));
    let query = &*(query as *const [f32; K]);
    let items = tile_base.add(K * len * std::mem::size_of::<f32>()) as *const T;

    nearest_one_avx512_raw_unchecked_f32::<L, T, K>(
        point_ptrs, items, len, query, best_dist, best_item,
    );
}

#[inline(always)]
pub(crate) unsafe fn nearest_one_avx512_unchecked_f32<L, T, const K: usize, const B: usize>(
    leaf: &LeafView<'_, f32, T, K, B>,
    query_wide: &[f32; K],
    best_dist: &mut f32,
    best_item: &mut T,
) where
    L: Avx512F32LeafOps,
    T: Content,
{
    let points = leaf.points();
    let point_ptrs = std::array::from_fn(|dim| points[dim].as_ptr());

    nearest_one_avx512_raw_unchecked_f32::<L, T, K>(
        point_ptrs,
        leaf.items().as_ptr(),
        leaf.items().len(),
        query_wide,
        best_dist,
        best_item,
    );
}

#[cfg(all(feature = "cargo_asm", feature = "simd", target_arch = "x86_64"))]
#[allow(dead_code)]
#[doc(hidden)]
#[inline(never)]
#[unsafe(no_mangle)]
pub fn v6_nearest_one_f32_leaf_cargo_asm_hook(
    points: [*const f32; 3],
    items: *const usize,
    len: usize,
    query: &[f32; 3],
    best_dist: &mut f32,
    best_item: &mut usize,
) {
    unsafe {
        nearest_one_avx512_raw_unchecked_f32::<
            <crate::dist::SquaredEuclidean<f32> as crate::dist::DistanceMetricAvx512<f32>>::Avx512F32Ops,
            usize,
            3,
        >(points, items, len, query, best_dist, best_item);
    }
}

#[cfg(test)]
mod tests {
    use super::nearest_one_avx512_raw_unchecked_f32;
    use crate::dist::{DistanceMetricAvx512, SquaredEuclidean};

    type SquaredF32Ops = <SquaredEuclidean<f32> as DistanceMetricAvx512<f32>>::Avx512F32Ops;

    fn assert_matches_scalar(values: Vec<f32>, expected_idx: usize) {
        let items: Vec<usize> = (0..values.len()).map(|idx| 10_000 + idx).collect();
        let query = [0.0f32];
        let mut best_dist = f32::INFINITY;
        let mut best_item = usize::MAX;

        unsafe {
            nearest_one_avx512_raw_unchecked_f32::<SquaredF32Ops, usize, 1>(
                [values.as_ptr()],
                items.as_ptr(),
                values.len(),
                &query,
                &mut best_dist,
                &mut best_item,
            );
        }

        let (scalar_idx, scalar_dist) = values
            .iter()
            .enumerate()
            .filter_map(|(idx, &value)| {
                let dist = value * value;
                (dist.is_finite()).then_some((idx, dist))
            })
            .min_by(|(left_idx, left_dist), (right_idx, right_dist)| {
                left_dist
                    .partial_cmp(right_dist)
                    .unwrap()
                    .then_with(|| left_idx.cmp(right_idx))
            })
            .unwrap();

        assert_eq!(scalar_idx, expected_idx);
        assert_eq!(best_item, items[scalar_idx]);
        assert_eq!(best_dist, scalar_dist);
    }

    #[test]
    fn f32_avx512_leaf_minimum_matches_scalar_across_vector_boundaries() {
        for len in [1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 63, 64, 65, 127] {
            let expected_idx = len / 2;
            let mut values: Vec<f32> = (0..len).map(|idx| idx as f32 + 2.0).collect();
            values[expected_idx] = 0.25;
            assert_matches_scalar(values, expected_idx);
        }
    }

    #[test]
    fn f32_avx512_leaf_minimum_ignores_nans_and_preserves_first_tie() {
        let mut values: Vec<f32> = (0..65).map(|idx| idx as f32 + 2.0).collect();
        values[0] = f32::NAN;
        values[17] = -0.5;
        values[48] = 0.5;
        assert_matches_scalar(values, 17);
    }

    #[test]
    fn f32_avx512_leaf_minimum_keeps_an_equal_incumbent() {
        let values: Vec<f32> = (0..64).map(|idx| idx as f32 + 1.0).collect();
        let items: Vec<usize> = (0..values.len()).collect();
        let query = [0.0f32];
        let mut best_dist = 1.0;
        let mut best_item = 123_456;

        unsafe {
            nearest_one_avx512_raw_unchecked_f32::<SquaredF32Ops, usize, 1>(
                [values.as_ptr()],
                items.as_ptr(),
                values.len(),
                &query,
                &mut best_dist,
                &mut best_item,
            );
        }

        assert_eq!(best_dist, 1.0);
        assert_eq!(best_item, 123_456);
    }
}

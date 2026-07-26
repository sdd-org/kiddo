use std::arch::aarch64::*;

use array_init::array_init;

use crate::dist::distance_metric_neon::{NeonF32LeafOps, NeonF64LeafOps};
use crate::leaf_view::LeafView;
use crate::Content;

#[inline(always)]
unsafe fn emit_results_neon_f64<T, F, const EXCLUSIVE: bool>(
    dists: float64x2_t,
    items: *const T,
    base: usize,
    max_dist: f64,
    emit: &mut F,
) where
    T: Content,
    F: FnMut(usize, f64, T),
{
    let mask = if EXCLUSIVE {
        vcltq_f64(dists, vdupq_n_f64(max_dist))
    } else {
        vcleq_f64(dists, vdupq_n_f64(max_dist))
    };
    let lane0 = vgetq_lane_u64(mask, 0);
    let lane1 = vgetq_lane_u64(mask, 1);
    if lane0 == 0 && lane1 == 0 {
        return;
    }

    let mut dist_values = [0.0f64; 2];
    vst1q_f64(dist_values.as_mut_ptr(), dists);

    if lane0 != 0 {
        emit(
            base,
            dist_values[0],
            std::ptr::read_unaligned(items.add(base)),
        );
    }
    if lane1 != 0 {
        emit(
            base + 1,
            dist_values[1],
            std::ptr::read_unaligned(items.add(base + 1)),
        );
    }
}

#[inline(always)]
unsafe fn line_dists_neon_f64<L, const K: usize>(
    points: &[*const f64; K],
    query: &[float64x2_t; K],
    base: usize,
) -> float64x2_t
where
    L: NeonF64LeafOps,
{
    let a0 = vld1q_f64(points[0].add(base));
    let d0 = vsubq_f64(a0, query[0]);
    let mut acc = L::dist_k0_f64x2(d0);

    for dim in 1..K {
        let a = vld1q_f64(points[dim].add(base));
        let d = vsubq_f64(a, query[dim]);
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
    L: NeonF64LeafOps,
{
    let mut dist = L::dist_k0_f64x1(*points[0].add(idx) - query[0]);
    for dim in 1..K {
        dist = L::dist_kn_f64x1(dist, *points[dim].add(idx) - query[dim]);
    }
    dist
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn nearest_n_within_neon_unchecked_f64<
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
    L: NeonF64LeafOps,
    T: Content,
    F: FnMut(usize, f64, T),
{
    let points = leaf.points();
    let point_ptrs = array_init(|dim| points[dim].as_ptr());
    nearest_n_within_neon_raw_f64::<L, T, F, EXCLUSIVE, K>(
        point_ptrs,
        leaf.items().as_ptr(),
        leaf.items().len(),
        query,
        max_dist,
        emit,
    );
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn nearest_n_within_neon_arena_unchecked_f64<
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
    L: NeonF64LeafOps,
    T: Content,
    F: FnMut(usize, f64, T),
{
    let point_base = tile_base as *const f64;
    let point_ptrs = array_init(|dim| point_base.add(dim * len));
    let items = tile_base.add(K * len * std::mem::size_of::<f64>()) as *const T;

    nearest_n_within_neon_raw_f64::<L, T, F, EXCLUSIVE, K>(
        point_ptrs, items, len, query, max_dist, emit,
    );
}

#[target_feature(enable = "neon")]
unsafe fn nearest_n_within_neon_raw_f64<L, T, F, const EXCLUSIVE: bool, const K: usize>(
    points: [*const f64; K],
    items: *const T,
    len: usize,
    query: &[f64; K],
    max_dist: f64,
    emit: &mut F,
) where
    L: NeonF64LeafOps,
    T: Content,
    F: FnMut(usize, f64, T),
{
    if len == 0 {
        return;
    }

    let query_neon = array_init(|dim| vdupq_n_f64(query[dim]));
    let mut base = 0usize;

    while base + 2 <= len {
        let d0 = line_dists_neon_f64::<L, K>(&points, &query_neon, base);
        emit_results_neon_f64::<_, _, EXCLUSIVE>(d0, items, base, max_dist, emit);
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
unsafe fn emit_results_neon_f32<T, F, const EXCLUSIVE: bool>(
    dists: float32x4_t,
    items: *const T,
    base: usize,
    max_dist: f32,
    emit: &mut F,
) where
    T: Content,
    F: FnMut(usize, f32, T),
{
    let mask = if EXCLUSIVE {
        vcltq_f32(dists, vdupq_n_f32(max_dist))
    } else {
        vcleq_f32(dists, vdupq_n_f32(max_dist))
    };
    let lane0 = vgetq_lane_u32(mask, 0);
    let lane1 = vgetq_lane_u32(mask, 1);
    let lane2 = vgetq_lane_u32(mask, 2);
    let lane3 = vgetq_lane_u32(mask, 3);
    if lane0 == 0 && lane1 == 0 && lane2 == 0 && lane3 == 0 {
        return;
    }

    let mut dist_values = [0.0f32; 4];
    vst1q_f32(dist_values.as_mut_ptr(), dists);

    if lane0 != 0 {
        emit(
            base,
            dist_values[0],
            std::ptr::read_unaligned(items.add(base)),
        );
    }
    if lane1 != 0 {
        emit(
            base + 1,
            dist_values[1],
            std::ptr::read_unaligned(items.add(base + 1)),
        );
    }
    if lane2 != 0 {
        emit(
            base + 2,
            dist_values[2],
            std::ptr::read_unaligned(items.add(base + 2)),
        );
    }
    if lane3 != 0 {
        emit(
            base + 3,
            dist_values[3],
            std::ptr::read_unaligned(items.add(base + 3)),
        );
    }
}

#[inline(always)]
unsafe fn line_dists_neon_f32<L, const K: usize>(
    points: &[*const f32; K],
    query: &[float32x4_t; K],
    base: usize,
) -> float32x4_t
where
    L: NeonF32LeafOps,
{
    let a0 = vld1q_f32(points[0].add(base));
    let d0 = vsubq_f32(a0, query[0]);
    let mut acc = L::dist_k0_f32x4(d0);

    for dim in 1..K {
        let a = vld1q_f32(points[dim].add(base));
        let d = vsubq_f32(a, query[dim]);
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
    L: NeonF32LeafOps,
{
    let mut dist = L::dist_k0_f32x1(*points[0].add(idx) - query[0]);
    for dim in 1..K {
        dist = L::dist_kn_f32x1(dist, *points[dim].add(idx) - query[dim]);
    }
    dist
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn nearest_n_within_neon_unchecked_f32<
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
    L: NeonF32LeafOps,
    T: Content,
    F: FnMut(usize, f32, T),
{
    let points = leaf.points();
    let point_ptrs = array_init(|dim| points[dim].as_ptr());
    nearest_n_within_neon_raw_f32::<L, T, F, EXCLUSIVE, K>(
        point_ptrs,
        leaf.items().as_ptr(),
        leaf.items().len(),
        query,
        max_dist,
        emit,
    );
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn nearest_n_within_neon_arena_unchecked_f32<
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
    L: NeonF32LeafOps,
    T: Content,
    F: FnMut(usize, f32, T),
{
    let point_base = tile_base as *const f32;
    let point_ptrs = array_init(|dim| point_base.add(dim * len));
    let items = tile_base.add(K * len * std::mem::size_of::<f32>()) as *const T;

    nearest_n_within_neon_raw_f32::<L, T, F, EXCLUSIVE, K>(
        point_ptrs, items, len, query, max_dist, emit,
    );
}

#[target_feature(enable = "neon")]
unsafe fn nearest_n_within_neon_raw_f32<L, T, F, const EXCLUSIVE: bool, const K: usize>(
    points: [*const f32; K],
    items: *const T,
    len: usize,
    query: &[f32; K],
    max_dist: f32,
    emit: &mut F,
) where
    L: NeonF32LeafOps,
    T: Content,
    F: FnMut(usize, f32, T),
{
    if len == 0 {
        return;
    }

    let query_neon = array_init(|dim| vdupq_n_f32(query[dim]));
    let mut base = 0usize;

    while base + 4 <= len {
        let d0 = line_dists_neon_f32::<L, K>(&points, &query_neon, base);
        emit_results_neon_f32::<_, _, EXCLUSIVE>(d0, items, base, max_dist, emit);
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

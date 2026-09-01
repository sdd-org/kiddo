use core::arch::wasm32::*;

use array_init::array_init;

use crate::{
    dist::distance_metric_wasm_simd::{WasmSimdF32LeafOps, WasmSimdF64LeafOps},
    leaf_view::LeafView,
    Content,
};

/// Load 2 consecutive f64 lanes from an arbitrarily-aligned address.
///
/// Arena coordinate columns can land on an 8-byte boundary, so `v128_load` and
/// its alignment of 16 would be unsound here.
#[inline(always)]
pub(crate) unsafe fn load_f64x2(ptr: *const f64) -> v128 {
    std::ptr::read_unaligned(ptr as *const v128)
}

/// Load 4 consecutive f32 lanes from an arbitrarily-aligned address.
///
/// Unaligned for the same reason as [`load_f64x2`].
#[inline(always)]
unsafe fn load_f32x4(ptr: *const f32) -> v128 {
    std::ptr::read_unaligned(ptr as *const v128)
}

#[inline(always)]
unsafe fn emit_results_wasm_f64<T, F, const EXCLUSIVE: bool>(
    dists: v128,
    items: *const T,
    base: usize,
    max_dist: f64,
    emit: &mut F,
) where
    T: Content,
    F: FnMut(usize, f64, T),
{
    let mask = if EXCLUSIVE {
        f64x2_lt(dists, f64x2_splat(max_dist))
    } else {
        f64x2_le(dists, f64x2_splat(max_dist))
    };
    let lanes = i64x2_bitmask(mask);
    if lanes == 0 {
        return;
    }

    if lanes & 0b01 != 0 {
        emit(
            base,
            f64x2_extract_lane::<0>(dists),
            std::ptr::read_unaligned(items.add(base)),
        );
    }
    if lanes & 0b10 != 0 {
        emit(
            base + 1,
            f64x2_extract_lane::<1>(dists),
            std::ptr::read_unaligned(items.add(base + 1)),
        );
    }
}

#[inline(always)]
pub(crate) unsafe fn line_dists_wasm_f64<L, const K: usize>(
    points: &[*const f64; K],
    query: &[v128; K],
    base: usize,
) -> v128
where
    L: WasmSimdF64LeafOps,
{
    let a0 = load_f64x2(points[0].add(base));
    let d0 = f64x2_sub(a0, query[0]);
    let mut acc = L::dist_k0_f64x2(d0);

    for dim in 1..K {
        let a = load_f64x2(points[dim].add(base));
        let d = f64x2_sub(a, query[dim]);
        acc = L::dist_kn_f64x2(acc, d);
    }

    acc
}

#[inline(always)]
pub(crate) unsafe fn dist_scalar_f64<L, const K: usize>(
    points: &[*const f64; K],
    query: &[f64; K],
    idx: usize,
) -> f64
where
    L: WasmSimdF64LeafOps,
{
    let mut dist = L::dist_k0_f64x1(*points[0].add(idx) - query[0]);
    for dim in 1..K {
        dist = L::dist_kn_f64x1(dist, *points[dim].add(idx) - query[dim]);
    }
    dist
}

pub(crate) unsafe fn nearest_n_within_wasm_unchecked_f64<
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
    L: WasmSimdF64LeafOps,
    T: Content,
    F: FnMut(usize, f64, T),
{
    let points = leaf.points();
    let point_ptrs = array_init(|dim| points[dim].as_ptr());
    nearest_n_within_wasm_raw_f64::<L, T, F, EXCLUSIVE, K>(
        point_ptrs,
        leaf.items().as_ptr(),
        leaf.items().len(),
        query,
        max_dist,
        emit,
    );
}

pub(crate) unsafe fn nearest_n_within_wasm_arena_unchecked_f64<
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
    L: WasmSimdF64LeafOps,
    T: Content,
    F: FnMut(usize, f64, T),
{
    let point_base = tile_base as *const f64;
    let point_ptrs = array_init(|dim| point_base.add(dim * len));
    let items = tile_base.add(K * len * std::mem::size_of::<f64>()) as *const T;

    nearest_n_within_wasm_raw_f64::<L, T, F, EXCLUSIVE, K>(
        point_ptrs, items, len, query, max_dist, emit,
    );
}

unsafe fn nearest_n_within_wasm_raw_f64<L, T, F, const EXCLUSIVE: bool, const K: usize>(
    points: [*const f64; K],
    items: *const T,
    len: usize,
    query: &[f64; K],
    max_dist: f64,
    emit: &mut F,
) where
    L: WasmSimdF64LeafOps,
    T: Content,
    F: FnMut(usize, f64, T),
{
    if len == 0 {
        return;
    }

    let query_wide = array_init(|dim| f64x2_splat(query[dim]));
    let mut base = 0usize;

    while base + 2 <= len {
        let d0 = line_dists_wasm_f64::<L, K>(&points, &query_wide, base);
        emit_results_wasm_f64::<_, _, EXCLUSIVE>(d0, items, base, max_dist, emit);
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
unsafe fn emit_results_wasm_f32<T, F, const EXCLUSIVE: bool>(
    dists: v128,
    items: *const T,
    base: usize,
    max_dist: f32,
    emit: &mut F,
) where
    T: Content,
    F: FnMut(usize, f32, T),
{
    let mask = if EXCLUSIVE {
        f32x4_lt(dists, f32x4_splat(max_dist))
    } else {
        f32x4_le(dists, f32x4_splat(max_dist))
    };
    let lanes = i32x4_bitmask(mask);
    if lanes == 0 {
        return;
    }

    if lanes & 0b0001 != 0 {
        emit(
            base,
            f32x4_extract_lane::<0>(dists),
            std::ptr::read_unaligned(items.add(base)),
        );
    }
    if lanes & 0b0010 != 0 {
        emit(
            base + 1,
            f32x4_extract_lane::<1>(dists),
            std::ptr::read_unaligned(items.add(base + 1)),
        );
    }
    if lanes & 0b0100 != 0 {
        emit(
            base + 2,
            f32x4_extract_lane::<2>(dists),
            std::ptr::read_unaligned(items.add(base + 2)),
        );
    }
    if lanes & 0b1000 != 0 {
        emit(
            base + 3,
            f32x4_extract_lane::<3>(dists),
            std::ptr::read_unaligned(items.add(base + 3)),
        );
    }
}

#[inline(always)]
pub(crate) unsafe fn line_dists_wasm_f32<L, const K: usize>(
    points: &[*const f32; K],
    query: &[v128; K],
    base: usize,
) -> v128
where
    L: WasmSimdF32LeafOps,
{
    let a0 = load_f32x4(points[0].add(base));
    let d0 = f32x4_sub(a0, query[0]);
    let mut acc = L::dist_k0_f32x4(d0);

    for dim in 1..K {
        let a = load_f32x4(points[dim].add(base));
        let d = f32x4_sub(a, query[dim]);
        acc = L::dist_kn_f32x4(acc, d);
    }

    acc
}

#[inline(always)]
pub(crate) unsafe fn dist_scalar_f32<L, const K: usize>(
    points: &[*const f32; K],
    query: &[f32; K],
    idx: usize,
) -> f32
where
    L: WasmSimdF32LeafOps,
{
    let mut dist = L::dist_k0_f32x1(*points[0].add(idx) - query[0]);
    for dim in 1..K {
        dist = L::dist_kn_f32x1(dist, *points[dim].add(idx) - query[dim]);
    }
    dist
}

pub(crate) unsafe fn nearest_n_within_wasm_unchecked_f32<
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
    L: WasmSimdF32LeafOps,
    T: Content,
    F: FnMut(usize, f32, T),
{
    let points = leaf.points();
    let point_ptrs = array_init(|dim| points[dim].as_ptr());
    nearest_n_within_wasm_raw_f32::<L, T, F, EXCLUSIVE, K>(
        point_ptrs,
        leaf.items().as_ptr(),
        leaf.items().len(),
        query,
        max_dist,
        emit,
    );
}

pub(crate) unsafe fn nearest_n_within_wasm_arena_unchecked_f32<
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
    L: WasmSimdF32LeafOps,
    T: Content,
    F: FnMut(usize, f32, T),
{
    let point_base = tile_base as *const f32;
    let point_ptrs = array_init(|dim| point_base.add(dim * len));
    let items = tile_base.add(K * len * std::mem::size_of::<f32>()) as *const T;

    nearest_n_within_wasm_raw_f32::<L, T, F, EXCLUSIVE, K>(
        point_ptrs, items, len, query, max_dist, emit,
    );
}

unsafe fn nearest_n_within_wasm_raw_f32<L, T, F, const EXCLUSIVE: bool, const K: usize>(
    points: [*const f32; K],
    items: *const T,
    len: usize,
    query: &[f32; K],
    max_dist: f32,
    emit: &mut F,
) where
    L: WasmSimdF32LeafOps,
    T: Content,
    F: FnMut(usize, f32, T),
{
    if len == 0 {
        return;
    }

    let query_wide = array_init(|dim| f32x4_splat(query[dim]));
    let mut base = 0usize;

    while base + 4 <= len {
        let d0 = line_dists_wasm_f32::<L, K>(&points, &query_wide, base);
        emit_results_wasm_f32::<_, _, EXCLUSIVE>(d0, items, base, max_dist, emit);
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

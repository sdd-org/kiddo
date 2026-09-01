use core::arch::wasm32::*;

use array_init::array_init;

use super::super::nearest_n_within::wasm_simd::{
    dist_scalar_f32, dist_scalar_f64, line_dists_wasm_f32, line_dists_wasm_f64,
};
use crate::{
    dist::distance_metric_wasm_simd::{WasmSimdF32LeafOps, WasmSimdF64LeafOps},
    leaf_view::LeafView,
    Content,
};

/// Sentinel for "no lane has produced a candidate yet".
const NO_CANDIDATE: i64 = -1;

pub(crate) unsafe fn nearest_one_wasm_unchecked_f64<L, T, const K: usize, const B: usize>(
    leaf: &LeafView<'_, f64, T, K, B>,
    query: &[f64; K],
    best_dist: &mut f64,
    best_item: &mut T,
) where
    L: WasmSimdF64LeafOps,
    T: Content,
{
    let points = leaf.points();
    let point_ptrs = array_init(|dim| points[dim].as_ptr());
    nearest_one_wasm_raw_f64::<L, T, K>(
        point_ptrs,
        leaf.items().as_ptr(),
        leaf.items().len(),
        query,
        best_dist,
        best_item,
    );
}

pub(crate) unsafe fn nearest_one_wasm_arena_unchecked_f64<L, T, const K: usize>(
    tile_base: *const u8,
    len: usize,
    query: &[f64; K],
    best_dist: &mut f64,
    best_item: &mut T,
) where
    L: WasmSimdF64LeafOps,
    T: Content,
{
    let point_base = tile_base as *const f64;
    let point_ptrs = array_init(|dim| point_base.add(dim * len));
    let items = tile_base.add(K * len * std::mem::size_of::<f64>()) as *const T;

    nearest_one_wasm_raw_f64::<L, T, K>(point_ptrs, items, len, query, best_dist, best_item);
}

unsafe fn nearest_one_wasm_raw_f64<L, T, const K: usize>(
    points: [*const f64; K],
    items: *const T,
    len: usize,
    query: &[f64; K],
    best_dist: &mut f64,
    best_item: &mut T,
) where
    L: WasmSimdF64LeafOps,
    T: Content,
{
    if len == 0 {
        return;
    }

    let query_wide: [v128; K] = array_init(|dim| f64x2_splat(query[dim]));

    let mut best_vec = f64x2_splat(*best_dist);
    let mut best_idx = i64x2_splat(NO_CANDIDATE);
    let mut idx_vec = i64x2(0, 1);
    let step = i64x2_splat(2);

    let mut base = 0usize;
    while base + 2 <= len {
        let dists = line_dists_wasm_f64::<L, K>(&points, &query_wide, base);
        let improves = f64x2_lt(dists, best_vec);
        best_vec = v128_bitselect(dists, best_vec, improves);
        best_idx = v128_bitselect(idx_vec, best_idx, improves);
        idx_vec = i64x2_add(idx_vec, step);
        base += 2;
    }

    // Break a distance tie on the smaller index.  The scalar fallback updates on
    // a strict `<`, keeping the first point that achieves the minimum; within a
    // lane `f64x2_lt` does the same, but across lanes only this comparison does.
    let (mut winning_dist, mut winning_idx) = (
        f64x2_extract_lane::<0>(best_vec),
        i64x2_extract_lane::<0>(best_idx),
    );
    let (lane1_dist, lane1_idx) = (
        f64x2_extract_lane::<1>(best_vec),
        i64x2_extract_lane::<1>(best_idx),
    );
    if lane1_dist < winning_dist || (lane1_dist == winning_dist && lane1_idx < winning_idx) {
        winning_dist = lane1_dist;
        winning_idx = lane1_idx;
    }

    for idx in base..len {
        let dist = dist_scalar_f64::<L, K>(&points, query, idx);
        if dist < winning_dist {
            winning_dist = dist;
            winning_idx = idx as i64;
        }
    }

    if winning_idx != NO_CANDIDATE {
        *best_dist = winning_dist;
        *best_item = std::ptr::read_unaligned(items.add(winning_idx as usize));
    }
}

pub(crate) unsafe fn nearest_one_wasm_unchecked_f32<L, T, const K: usize, const B: usize>(
    leaf: &LeafView<'_, f32, T, K, B>,
    query: &[f32; K],
    best_dist: &mut f32,
    best_item: &mut T,
) where
    L: WasmSimdF32LeafOps,
    T: Content,
{
    let points = leaf.points();
    let point_ptrs = array_init(|dim| points[dim].as_ptr());
    nearest_one_wasm_raw_f32::<L, T, K>(
        point_ptrs,
        leaf.items().as_ptr(),
        leaf.items().len(),
        query,
        best_dist,
        best_item,
    );
}

pub(crate) unsafe fn nearest_one_wasm_arena_unchecked_f32<L, T, const K: usize>(
    tile_base: *const u8,
    len: usize,
    query: &[f32; K],
    best_dist: &mut f32,
    best_item: &mut T,
) where
    L: WasmSimdF32LeafOps,
    T: Content,
{
    let point_base = tile_base as *const f32;
    let point_ptrs = array_init(|dim| point_base.add(dim * len));
    let items = tile_base.add(K * len * std::mem::size_of::<f32>()) as *const T;

    nearest_one_wasm_raw_f32::<L, T, K>(point_ptrs, items, len, query, best_dist, best_item);
}

unsafe fn nearest_one_wasm_raw_f32<L, T, const K: usize>(
    points: [*const f32; K],
    items: *const T,
    len: usize,
    query: &[f32; K],
    best_dist: &mut f32,
    best_item: &mut T,
) where
    L: WasmSimdF32LeafOps,
    T: Content,
{
    if len == 0 {
        return;
    }

    let query_wide: [v128; K] = array_init(|dim| f32x4_splat(query[dim]));

    let mut best_vec = f32x4_splat(*best_dist);
    let mut best_idx = i32x4_splat(NO_CANDIDATE as i32);
    let mut idx_vec = i32x4(0, 1, 2, 3);
    let step = i32x4_splat(4);

    let mut base = 0usize;
    while base + 4 <= len {
        let dists = line_dists_wasm_f32::<L, K>(&points, &query_wide, base);
        let improves = f32x4_lt(dists, best_vec);
        best_vec = v128_bitselect(dists, best_vec, improves);
        best_idx = v128_bitselect(idx_vec, best_idx, improves);
        idx_vec = i32x4_add(idx_vec, step);
        base += 4;
    }

    let mut winning_dist = f32x4_extract_lane::<0>(best_vec);
    let mut winning_idx = i32x4_extract_lane::<0>(best_idx);
    let lanes = [
        (
            f32x4_extract_lane::<1>(best_vec),
            i32x4_extract_lane::<1>(best_idx),
        ),
        (
            f32x4_extract_lane::<2>(best_vec),
            i32x4_extract_lane::<2>(best_idx),
        ),
        (
            f32x4_extract_lane::<3>(best_vec),
            i32x4_extract_lane::<3>(best_idx),
        ),
    ];
    for (dist, idx) in lanes {
        if dist < winning_dist || (dist == winning_dist && idx < winning_idx) {
            winning_dist = dist;
            winning_idx = idx;
        }
    }

    for idx in base..len {
        let dist = dist_scalar_f32::<L, K>(&points, query, idx);
        if dist < winning_dist {
            winning_dist = dist;
            winning_idx = idx as i32;
        }
    }

    if winning_idx != NO_CANDIDATE as i32 {
        *best_dist = winning_dist;
        *best_item = std::ptr::read_unaligned(items.add(winning_idx as usize));
    }
}

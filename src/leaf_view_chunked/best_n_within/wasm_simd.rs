use crate::{
    dist::distance_metric_wasm_simd::{WasmSimdF32LeafOps, WasmSimdF64LeafOps},
    leaf_view::LeafView,
    Content,
};

pub(crate) unsafe fn best_n_within_wasm_unchecked_f64<
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
    T: Content + PartialOrd,
    F: FnMut(f64, T),
{
    let mut emit_positioned = |_, distance, item| emit(distance, item);
    crate::leaf_view_chunked::nearest_n_within::wasm_simd::nearest_n_within_wasm_unchecked_f64::<
        L,
        T,
        _,
        EXCLUSIVE,
        K,
        B,
    >(leaf, query, max_dist, &mut emit_positioned);
}

pub(crate) unsafe fn best_n_within_wasm_arena_unchecked_f64<
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
    T: Content + PartialOrd,
    F: FnMut(f64, T),
{
    let mut emit_positioned = |_, distance, item| emit(distance, item);
    crate::leaf_view_chunked::nearest_n_within::wasm_simd::nearest_n_within_wasm_arena_unchecked_f64::<
        L,
        T,
        _,
        EXCLUSIVE,
        K,
    >(tile_base, len, query, max_dist, &mut emit_positioned);
}

pub(crate) unsafe fn best_n_within_wasm_unchecked_f32<
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
    T: Content + PartialOrd,
    F: FnMut(f32, T),
{
    let mut emit_positioned = |_, distance, item| emit(distance, item);
    crate::leaf_view_chunked::nearest_n_within::wasm_simd::nearest_n_within_wasm_unchecked_f32::<
        L,
        T,
        _,
        EXCLUSIVE,
        K,
        B,
    >(leaf, query, max_dist, &mut emit_positioned);
}

pub(crate) unsafe fn best_n_within_wasm_arena_unchecked_f32<
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
    T: Content + PartialOrd,
    F: FnMut(f32, T),
{
    let mut emit_positioned = |_, distance, item| emit(distance, item);
    crate::leaf_view_chunked::nearest_n_within::wasm_simd::nearest_n_within_wasm_arena_unchecked_f32::<
        L,
        T,
        _,
        EXCLUSIVE,
        K,
    >(tile_base, len, query, max_dist, &mut emit_positioned);
}

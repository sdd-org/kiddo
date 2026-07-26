use crate::dist::distance_metric_neon::{NeonF32LeafOps, NeonF64LeafOps};
use crate::leaf_view::LeafView;
use crate::Content;

#[target_feature(enable = "neon")]
pub(crate) unsafe fn best_n_within_neon_unchecked_f64<
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
    T: Content + PartialOrd,
    F: FnMut(f64, T),
{
    let mut emit_positioned = |_, distance, item| emit(distance, item);
    crate::leaf_view_chunked::nearest_n_within::neon::nearest_n_within_neon_unchecked_f64::<
        L,
        T,
        _,
        EXCLUSIVE,
        K,
        B,
    >(leaf, query, max_dist, &mut emit_positioned);
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn best_n_within_neon_arena_unchecked_f64<
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
    T: Content + PartialOrd,
    F: FnMut(f64, T),
{
    let mut emit_positioned = |_, distance, item| emit(distance, item);
    crate::leaf_view_chunked::nearest_n_within::neon::nearest_n_within_neon_arena_unchecked_f64::<
        L,
        T,
        _,
        EXCLUSIVE,
        K,
    >(tile_base, len, query, max_dist, &mut emit_positioned);
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn best_n_within_neon_unchecked_f32<
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
    T: Content + PartialOrd,
    F: FnMut(f32, T),
{
    let mut emit_positioned = |_, distance, item| emit(distance, item);
    crate::leaf_view_chunked::nearest_n_within::neon::nearest_n_within_neon_unchecked_f32::<
        L,
        T,
        _,
        EXCLUSIVE,
        K,
        B,
    >(leaf, query, max_dist, &mut emit_positioned);
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn best_n_within_neon_arena_unchecked_f32<
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
    T: Content + PartialOrd,
    F: FnMut(f32, T),
{
    let mut emit_positioned = |_, distance, item| emit(distance, item);
    crate::leaf_view_chunked::nearest_n_within::neon::nearest_n_within_neon_arena_unchecked_f32::<
        L,
        T,
        _,
        EXCLUSIVE,
        K,
    >(tile_base, len, query, max_dist, &mut emit_positioned);
}

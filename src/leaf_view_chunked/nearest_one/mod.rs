mod fallback;

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
pub(crate) mod avx512;

#[cfg(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) mod wasm_simd;

use crate::dist::DistanceMetric;
use crate::leaf_view::{LeafArena, LeafView};
use crate::{Axis, Content};
// #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
// pub(crate) use avx512::nearest_one_avx512_unchecked;

pub(crate) use fallback::{
    nearest_one_with_query_wide_arena_fallback, nearest_one_with_query_wide_fallback,
};

#[inline(always)]
pub(crate) fn nearest_one_with_query_wide_arena<AX, T, D, const K: usize>(
    arena: &LeafArena<'_, AX, T, K>,
    query_wide: &[D::Output; K],
    best_dist: &mut D::Output,
    best_item: &mut T,
) where
    AX: Axis<Coord = AX> + 'static,
    T: Content,
    D: DistanceMetric<AX>,
    D::Output: Axis<Coord = D::Output> + 'static,
{
    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    if unsafe {
        try_nearest_one_arena_avx512::<AX, T, D, K>(arena, query_wide, best_dist, best_item)
    } {
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128"))]
    if unsafe {
        try_nearest_one_arena_wasm_simd::<AX, T, D, K>(arena, query_wide, best_dist, best_item)
    } {
        return;
    }

    nearest_one_with_query_wide_arena_fallback::<AX, T, D, K>(
        arena, query_wide, best_dist, best_item,
    );
}

#[inline(always)]
pub(crate) fn nearest_one_with_query_wide<AX, T, D, const K: usize, const B: usize>(
    leaf: &LeafView<'_, AX, T, K, B>,
    query_wide: &[D::Output; K],
    best_dist: &mut D::Output,
    best_item: &mut T,
) where
    AX: Axis<Coord = AX> + 'static,
    T: Content,
    D: DistanceMetric<AX>,
    D::Output: Axis<Coord = D::Output> + 'static,
{
    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    if unsafe { try_nearest_one_avx512::<AX, T, D, K, B>(leaf, query_wide, best_dist, best_item) } {
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128"))]
    if unsafe {
        try_nearest_one_wasm_simd::<AX, T, D, K, B>(leaf, query_wide, best_dist, best_item)
    } {
        return;
    }

    nearest_one_with_query_wide_fallback::<AX, T, D, K, B>(leaf, query_wide, best_dist, best_item);
}

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn try_nearest_one_avx512<AX, T, D, const K: usize, const B: usize>(
    leaf: &LeafView<'_, AX, T, K, B>,
    query_wide: &[D::Output; K],
    best_dist: &mut D::Output,
    best_item: &mut T,
) -> bool
where
    AX: Axis<Coord = AX> + 'static,
    T: Content,
    D: DistanceMetric<AX>,
    D::Output: Axis<Coord = D::Output> + 'static,
{
    D::try_nearest_one_leaf_avx512(leaf, query_wide, best_dist, best_item)
}

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn try_nearest_one_arena_avx512<AX, T, D, const K: usize>(
    arena: &LeafArena<'_, AX, T, K>,
    query_wide: &[D::Output; K],
    best_dist: &mut D::Output,
    best_item: &mut T,
) -> bool
where
    AX: Axis<Coord = AX> + 'static,
    T: Content,
    D: DistanceMetric<AX>,
    D::Output: Axis<Coord = D::Output> + 'static,
{
    D::try_nearest_one_arena_avx512(arena, query_wide, best_dist, best_item)
}

#[cfg(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
unsafe fn try_nearest_one_wasm_simd<AX, T, D, const K: usize, const B: usize>(
    leaf: &LeafView<'_, AX, T, K, B>,
    query_wide: &[D::Output; K],
    best_dist: &mut D::Output,
    best_item: &mut T,
) -> bool
where
    AX: Axis<Coord = AX> + 'static,
    T: Content,
    D: DistanceMetric<AX>,
    D::Output: Axis<Coord = D::Output> + 'static,
{
    D::try_nearest_one_leaf_wasm_simd(leaf, query_wide, best_dist, best_item)
}

#[cfg(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
unsafe fn try_nearest_one_arena_wasm_simd<AX, T, D, const K: usize>(
    arena: &LeafArena<'_, AX, T, K>,
    query_wide: &[D::Output; K],
    best_dist: &mut D::Output,
    best_item: &mut T,
) -> bool
where
    AX: Axis<Coord = AX> + 'static,
    T: Content,
    D: DistanceMetric<AX>,
    D::Output: Axis<Coord = D::Output> + 'static,
{
    D::try_nearest_one_arena_wasm_simd(arena, query_wide, best_dist, best_item)
}

mod fallback;

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
pub(crate) mod avx2;

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
pub(crate) mod avx512;

#[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
pub(crate) mod neon;

#[cfg(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) mod wasm_simd;

use crate::dist::DistanceMetric;
use crate::leaf_view::{LeafArena, LeafView, TlsLeafScratch};
use crate::results::result_collection::{FromLeafCandidate, ResultCollection};
use crate::{Axis, Content};

pub(crate) use fallback::{
    nearest_n_within_with_query_wide_arena_fallback, nearest_n_within_with_query_wide_fallback,
};

#[inline(always)]
pub(crate) fn nearest_n_within_with_query_wide_arena<
    AX,
    T,
    D,
    E,
    R,
    const EXCLUSIVE: bool,
    const K: usize,
>(
    arena: &LeafArena<'_, AX, T, K>,
    query_wide: &[D::Output; K],
    dist: D::Output,
    results: &mut R,
) where
    AX: Axis<Coord = AX> + 'static,
    T: Content,
    D: DistanceMetric<AX>,
    D::Output: Axis<Coord = D::Output> + 'static,
    E: FromLeafCandidate<AX, T, D::Output, K>,
    R: ResultCollection<D::Output, E>,
{
    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    if unsafe {
        try_nearest_n_within_arena_avx512::<AX, T, D, E, R, EXCLUSIVE, K>(
            arena, query_wide, dist, results,
        )
    } {
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    if unsafe {
        try_nearest_n_within_arena_avx2::<AX, T, D, E, R, EXCLUSIVE, K>(
            arena, query_wide, dist, results,
        )
    } {
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
    if unsafe {
        try_nearest_n_within_arena_neon::<AX, T, D, E, R, EXCLUSIVE, K>(
            arena, query_wide, dist, results,
        )
    } {
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128"))]
    if unsafe {
        try_nearest_n_within_arena_wasm_simd::<AX, T, D, E, R, EXCLUSIVE, K>(
            arena, query_wide, dist, results,
        )
    } {
        return;
    }

    nearest_n_within_with_query_wide_arena_fallback::<AX, T, D, E, R, EXCLUSIVE, K>(
        arena, query_wide, dist, results,
    );
}

#[inline(always)]
pub(crate) fn nearest_n_within_with_query_wide<
    AX,
    T,
    D,
    E,
    R,
    const EXCLUSIVE: bool,
    const K: usize,
    const B: usize,
>(
    leaf: &LeafView<'_, AX, T, K, B>,
    query_wide: &[D::Output; K],
    dist: D::Output,
    results: &mut R,
) where
    AX: Axis<Coord = AX> + 'static,
    T: Content + PartialOrd,
    D: DistanceMetric<AX>,
    D::Output: Axis<Coord = D::Output> + TlsLeafScratch + 'static,
    E: FromLeafCandidate<AX, T, D::Output, K>,
    R: ResultCollection<D::Output, E>,
{
    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    if unsafe {
        try_nearest_n_within_avx512::<AX, T, D, E, R, EXCLUSIVE, K, B>(
            leaf, query_wide, dist, results,
        )
    } {
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    if unsafe {
        try_nearest_n_within_avx2::<AX, T, D, E, R, EXCLUSIVE, K, B>(
            leaf, query_wide, dist, results,
        )
    } {
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
    if unsafe {
        try_nearest_n_within_neon::<AX, T, D, E, R, EXCLUSIVE, K, B>(
            leaf, query_wide, dist, results,
        )
    } {
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128"))]
    if unsafe {
        try_nearest_n_within_wasm_simd::<AX, T, D, E, R, EXCLUSIVE, K, B>(
            leaf, query_wide, dist, results,
        )
    } {
        return;
    }

    nearest_n_within_with_query_wide_fallback::<AX, T, D, E, R, EXCLUSIVE, K, B>(
        leaf, query_wide, dist, results,
    );
}

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn try_nearest_n_within_avx512<
    AX,
    T,
    D,
    E,
    R,
    const EXCLUSIVE: bool,
    const K: usize,
    const B: usize,
>(
    leaf: &LeafView<'_, AX, T, K, B>,
    query_wide: &[D::Output; K],
    dist: D::Output,
    results: &mut R,
) -> bool
where
    AX: Axis<Coord = AX> + 'static,
    T: Content,
    D: DistanceMetric<AX>,
    D::Output: Axis<Coord = D::Output> + 'static,
    E: FromLeafCandidate<AX, T, D::Output, K>,
    R: ResultCollection<D::Output, E>,
{
    D::try_nearest_n_within_leaf_avx512::<T, E, R, EXCLUSIVE, K, B>(leaf, query_wide, dist, results)
}

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn try_nearest_n_within_arena_avx512<AX, T, D, E, R, const EXCLUSIVE: bool, const K: usize>(
    arena: &LeafArena<'_, AX, T, K>,
    query_wide: &[D::Output; K],
    dist: D::Output,
    results: &mut R,
) -> bool
where
    AX: Axis<Coord = AX> + 'static,
    T: Content,
    D: DistanceMetric<AX>,
    D::Output: Axis<Coord = D::Output> + 'static,
    E: FromLeafCandidate<AX, T, D::Output, K>,
    R: ResultCollection<D::Output, E>,
{
    D::try_nearest_n_within_arena_avx512::<T, E, R, EXCLUSIVE, K>(arena, query_wide, dist, results)
}

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn try_nearest_n_within_avx2<
    AX,
    T,
    D,
    E,
    R,
    const EXCLUSIVE: bool,
    const K: usize,
    const B: usize,
>(
    leaf: &LeafView<'_, AX, T, K, B>,
    query_wide: &[D::Output; K],
    dist: D::Output,
    results: &mut R,
) -> bool
where
    AX: Axis<Coord = AX> + 'static,
    T: Content,
    D: DistanceMetric<AX>,
    D::Output: Axis<Coord = D::Output> + 'static,
    E: FromLeafCandidate<AX, T, D::Output, K>,
    R: ResultCollection<D::Output, E>,
{
    D::try_nearest_n_within_leaf_avx2::<T, E, R, EXCLUSIVE, K, B>(leaf, query_wide, dist, results)
}

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn try_nearest_n_within_arena_avx2<AX, T, D, E, R, const EXCLUSIVE: bool, const K: usize>(
    arena: &LeafArena<'_, AX, T, K>,
    query_wide: &[D::Output; K],
    dist: D::Output,
    results: &mut R,
) -> bool
where
    AX: Axis<Coord = AX> + 'static,
    T: Content,
    D: DistanceMetric<AX>,
    D::Output: Axis<Coord = D::Output> + 'static,
    E: FromLeafCandidate<AX, T, D::Output, K>,
    R: ResultCollection<D::Output, E>,
{
    D::try_nearest_n_within_arena_avx2::<T, E, R, EXCLUSIVE, K>(arena, query_wide, dist, results)
}

#[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
unsafe fn try_nearest_n_within_neon<
    AX,
    T,
    D,
    E,
    R,
    const EXCLUSIVE: bool,
    const K: usize,
    const B: usize,
>(
    leaf: &LeafView<'_, AX, T, K, B>,
    query_wide: &[D::Output; K],
    dist: D::Output,
    results: &mut R,
) -> bool
where
    AX: Axis<Coord = AX> + 'static,
    T: Content,
    D: DistanceMetric<AX>,
    D::Output: Axis<Coord = D::Output> + 'static,
    E: FromLeafCandidate<AX, T, D::Output, K>,
    R: ResultCollection<D::Output, E>,
{
    D::try_nearest_n_within_leaf_neon::<T, E, R, EXCLUSIVE, K, B>(leaf, query_wide, dist, results)
}

#[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
unsafe fn try_nearest_n_within_arena_neon<AX, T, D, E, R, const EXCLUSIVE: bool, const K: usize>(
    arena: &LeafArena<'_, AX, T, K>,
    query_wide: &[D::Output; K],
    dist: D::Output,
    results: &mut R,
) -> bool
where
    AX: Axis<Coord = AX> + 'static,
    T: Content,
    D: DistanceMetric<AX>,
    D::Output: Axis<Coord = D::Output> + 'static,
    E: FromLeafCandidate<AX, T, D::Output, K>,
    R: ResultCollection<D::Output, E>,
{
    D::try_nearest_n_within_arena_neon::<T, E, R, EXCLUSIVE, K>(arena, query_wide, dist, results)
}

#[cfg(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
unsafe fn try_nearest_n_within_wasm_simd<
    AX,
    T,
    D,
    E,
    R,
    const EXCLUSIVE: bool,
    const K: usize,
    const B: usize,
>(
    leaf: &LeafView<'_, AX, T, K, B>,
    query_wide: &[D::Output; K],
    dist: D::Output,
    results: &mut R,
) -> bool
where
    AX: Axis<Coord = AX> + 'static,
    T: Content,
    D: DistanceMetric<AX>,
    D::Output: Axis<Coord = D::Output> + 'static,
    E: FromLeafCandidate<AX, T, D::Output, K>,
    R: ResultCollection<D::Output, E>,
{
    D::try_nearest_n_within_leaf_wasm_simd::<T, E, R, EXCLUSIVE, K, B>(
        leaf, query_wide, dist, results,
    )
}

#[cfg(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
unsafe fn try_nearest_n_within_arena_wasm_simd<
    AX,
    T,
    D,
    E,
    R,
    const EXCLUSIVE: bool,
    const K: usize,
>(
    arena: &LeafArena<'_, AX, T, K>,
    query_wide: &[D::Output; K],
    dist: D::Output,
    results: &mut R,
) -> bool
where
    AX: Axis<Coord = AX> + 'static,
    T: Content,
    D: DistanceMetric<AX>,
    D::Output: Axis<Coord = D::Output> + 'static,
    E: FromLeafCandidate<AX, T, D::Output, K>,
    R: ResultCollection<D::Output, E>,
{
    D::try_nearest_n_within_arena_wasm_simd::<T, E, R, EXCLUSIVE, K>(
        arena, query_wide, dist, results,
    )
}

#![allow(clippy::missing_safety_doc)]
#![allow(private_bounds)]

/// AVX2 distance metric trait
#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
#[doc(hidden)]
pub mod distance_metric_avx2;

/// AVX512 distance metric trait
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
#[doc(hidden)]
pub mod distance_metric_avx512;

/// Core distance metric trait
#[doc(hidden)]
pub mod distance_metric_core;

/// NEON distance metric trait
#[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
#[doc(hidden)]
pub mod distance_metric_neon;

/// Chebyshev distance metric
#[doc(hidden)]
pub mod chebyshev;

/// Minkowski distance metric
#[doc(hidden)]
pub mod minkowski;

/// Manhattan distance metric
#[doc(hidden)]
pub mod manhattan;

/// Squared Euclidean distance metric
#[doc(hidden)]
pub mod squared_euclidean;

use std::ptr::NonNull;

/// Shared distance metric functions
pub(crate) mod common;

pub(crate) use distance_metric_core::DistanceMetricScalar as DistanceMetricCore;
#[doc(hidden)]
pub use distance_metric_core::{DistanceMetricScalar, WideningCastFrom};
#[cfg(any(
    all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"),
    all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"),
    all(feature = "simd", target_arch = "aarch64", target_feature = "neon")
))]
use std::any::TypeId;

use crate::leaf_view::TlsLeafScratch;
use crate::stem_strategy::donnelly::simd_full::{BacktrackBlock3, BacktrackBlock4};
use crate::stem_strategy::{SimdPrune, SimdSelectBestChildBlock3};
use crate::Axis;
#[cfg(any(
    all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"),
    all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"),
    all(feature = "simd", target_arch = "aarch64", target_feature = "neon")
))]
use crate::{
    leaf_view::{LeafArena, LeafView},
    results::result_collection::{
        BestNeighbourResultCollection, FromLeafCandidate, ResultCollection,
    },
    BestQueryResultItem,
};

#[doc(inline)]
pub use chebyshev::Chebyshev;
#[doc(inline)]
pub use manhattan::Manhattan;
#[doc(inline)]
pub use minkowski::Minkowski;
#[doc(inline)]
pub use squared_euclidean::SquaredEuclidean;

/// Widened distance/output types that Kiddo's executable query paths support.
///
/// This is mainly useful as part of [`QueryMetric`]. Most library users should
/// not need to name this trait directly.
pub trait QueryDistance:
    SimdPrune + SimdSelectBestChildBlock3 + BacktrackBlock3 + BacktrackBlock4 + TlsLeafScratch + 'static
{
}

impl<T> QueryDistance for T where
    T: SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static
{
}

mod query_metric_sealed {
    use super::{DistanceMetric, QueryDistance};

    pub trait Sealed<A: Copy> {}

    impl<T, A: Copy> Sealed<A> for T where T: DistanceMetric<A, Output: QueryDistance> {}
}

/// Public bound for distance metrics that can be used with query-builder APIs
/// that finish with `.execute()` or `.execute_with_scratch(...)`.
///
/// Implementing [`DistanceMetric`] is still enough to define a custom metric.
/// `QueryMetric` is the stronger bound required when a generic helper wants to
/// execute a query built with that metric.
///
/// In practice, this means the metric's widened [`DistanceMetricScalar::Output`] type
/// must be one of Kiddo's supported query distance/output types. Metrics using
/// the usual widened output types such as `f32` or `f64` continue to work.
/// Metrics with a brand new output type will not satisfy this bound unless that
/// output type also supports Kiddo's internal query-execution operations.
///
/// # Example
///
/// ```ignore
/// use kiddo::KdTree;
/// use kiddo::dist::{QueryMetric, SquaredEuclidean};
/// use kiddo::Axis;
/// use std::num::NonZeroUsize;
///
/// fn query<A, M, const K: usize>(
///     tree: &KdTree<A, u32, _, _, K, 32>,
///     point: &[A; K],
///     n: NonZeroUsize,
/// ) -> Vec<u32>
/// where
///     A: Axis<Coord = A> + Copy + 'static,
///     M: QueryMetric<A>,
/// {
///     tree.query(point)
///         .nearest_n::<M>(n)
///         .execute()
///         .into_iter()
///         .map(|result| result.item)
///         .collect()
/// }
/// ```
pub trait QueryMetric<A: Copy>:
    DistanceMetric<A, Output: QueryDistance> + query_metric_sealed::Sealed<A>
{
}

impl<T, A: Copy> QueryMetric<A> for T where T: DistanceMetric<A, Output: QueryDistance> {}

#[cfg(any(
    all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"),
    all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"),
    all(feature = "simd", target_arch = "aarch64", target_feature = "neon")
))]
macro_rules! with_nearest_result_emitter {
    ($results:expr, $entry_ty:ty, $distance_ty:ty, $item_ty:ty, $emit:ident, $body:block) => {{
        let mut $emit = |candidate_point: [$distance_ty; K],
                         candidate_dist: $distance_ty,
                         item: $item_ty| {
            #[cfg(feature = "result_collection_stats")]
            crate::results::result_collection_stats::record_candidate_emitted();

            let point = std::mem::transmute_copy::<[$distance_ty; K], [A; K]>(&candidate_point);
            let distance = std::mem::transmute_copy::<$distance_ty, Self::Output>(&candidate_dist);
            let candidate =
                <$entry_ty as FromLeafCandidate<A, $item_ty, Self::Output, K>>::from_leaf_candidate(
                    point, item, distance,
                );

            $results.add(candidate);
        };

        $body
    }};
}

#[cfg(any(
    all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"),
    all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"),
    all(feature = "simd", target_arch = "aarch64", target_feature = "neon")
))]
#[inline(always)]
unsafe fn point_from_leaf_arena_tile<A: Copy, const K: usize>(
    tile_base: *const u8,
    tile_len: usize,
    position: usize,
) -> [A; K] {
    let point_base = tile_base as *const A;
    std::array::from_fn(|dim| std::ptr::read_unaligned(point_base.add(dim * tile_len + position)))
}

#[cfg(any(
    all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"),
    all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"),
    all(feature = "simd", target_arch = "aarch64", target_feature = "neon")
))]
macro_rules! with_best_result_emitter {
    ($results:expr, $threshold_item:expr, $distance_ty:ty, $item_ty:ty, $emit:ident, $body:block) => {{
        let threshold_item = $threshold_item;
        let mut $emit = |candidate_dist: $distance_ty, item: $item_ty| {
            if threshold_item.is_some_and(|worst_item| item >= worst_item) {
                #[cfg(feature = "result_collection_stats")]
                crate::results::result_collection_stats::record_best_item_threshold_reject();
                return;
            }

            #[cfg(feature = "result_collection_stats")]
            crate::results::result_collection_stats::record_candidate_emitted();

            let candidate = BestQueryResultItem {
                point: (),
                distance: std::mem::transmute_copy::<$distance_ty, Self::Output>(&candidate_dist),
                item,
            };

            $results.add(candidate);
        };

        $body
    }};
}

/// AVX512 extension hooks.
///
/// Default behavior is "not specialized". Concrete metrics can override hook
/// methods in arch-specific code without changing public query bounds.
#[doc(hidden)]
pub trait DistanceMetricAvx512<A: Copy>: DistanceMetricScalar<A> {
    /// Type that provides implementations of the AVX512 leaf ops
    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    type Avx512F64Ops: distance_metric_avx512::Avx512F64LeafOps + 'static;

    /// Type that provides implementations of the AVX512 f32 leaf ops.
    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    type Avx512F32Ops: distance_metric_avx512::Avx512F32LeafOps + 'static;

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    #[inline(always)]
    /// Try an AVX512-specialized `nearest_one` leaf kernel.
    unsafe fn try_nearest_one_leaf_avx512<T, const K: usize, const B: usize>(
        leaf: &LeafView<'_, A, T, K, B>,
        query_wide: &[Self::Output; K],
        best_dist: &mut Self::Output,
        best_item: &mut T,
    ) -> bool
    where
        A: Axis<Coord = A> + 'static,
        T: crate::Content,
        Self::Output: Axis<Coord = Self::Output> + 'static,
    {
        if TypeId::of::<A>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Avx512F64Ops>()
                != TypeId::of::<distance_metric_avx512::UnsupportedAvx512F64LeafOps>()
        {
            let leaf =
                &*(leaf as *const LeafView<'_, A, T, K, B> as *const LeafView<'_, f64, T, K, B>);
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f64; K]);
            let best_dist = &mut *(best_dist as *mut Self::Output as *mut f64);

            crate::leaf_view_chunked::nearest_one::avx512::nearest_one_avx512_unchecked::<
                f64,
                Self::Avx512F64Ops,
                T,
                K,
                B,
            >(leaf, query_wide, best_dist, best_item);

            return true;
        }

        if TypeId::of::<A>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Avx512F32Ops>()
                != TypeId::of::<distance_metric_avx512::UnsupportedAvx512F32LeafOps>()
        {
            let leaf =
                &*(leaf as *const LeafView<'_, A, T, K, B> as *const LeafView<'_, f32, T, K, B>);
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f32; K]);
            let best_dist = &mut *(best_dist as *mut Self::Output as *mut f32);

            crate::leaf_view_chunked::nearest_one::avx512::nearest_one_avx512_unchecked_f32::<
                Self::Avx512F32Ops,
                T,
                K,
                B,
            >(leaf, query_wide, best_dist, best_item);

            return true;
        }

        false
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    #[inline(always)]
    /// Try an AVX512-specialized `nearest_one` arena kernel.
    unsafe fn try_nearest_one_arena_avx512<T, const K: usize>(
        arena: &LeafArena<'_, A, T, K>,
        query_wide: &[Self::Output; K],
        best_dist: &mut Self::Output,
        best_item: &mut T,
    ) -> bool
    where
        A: Axis<Coord = A> + 'static,
        T: crate::Content,
        Self::Output: Axis<Coord = Self::Output> + 'static,
    {
        if TypeId::of::<A>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Avx512F64Ops>()
                != TypeId::of::<distance_metric_avx512::UnsupportedAvx512F64LeafOps>()
        {
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f64; K]);
            let best_dist = &mut *(best_dist as *mut Self::Output as *mut f64);
            let query_ptr = query_wide.as_ptr();
            let mut tile_base = arena.as_ptr();
            let mut remaining = arena.len();

            while remaining != 0 {
                let tile_len = crate::leaf_view::leaf_arena_tile_len(remaining);
                crate::leaf_view_chunked::nearest_one::avx512::nearest_one_avx512_arena_unchecked::<
                    f64,
                    Self::Avx512F64Ops,
                    T,
                    K,
                >(tile_base, tile_len, query_ptr, best_dist, best_item);
                let tile_bytes =
                    K * tile_len * std::mem::size_of::<f64>() + tile_len * std::mem::size_of::<T>();
                tile_base = tile_base.add(tile_bytes);
                remaining -= tile_len;
            }

            return true;
        }

        if TypeId::of::<A>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Avx512F32Ops>()
                != TypeId::of::<distance_metric_avx512::UnsupportedAvx512F32LeafOps>()
        {
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f32; K]);
            let best_dist = &mut *(best_dist as *mut Self::Output as *mut f32);
            let query_ptr = query_wide.as_ptr();
            let mut tile_base = arena.as_ptr();
            let mut remaining = arena.len();

            while remaining != 0 {
                let tile_len = crate::leaf_view::leaf_arena_tile_len(remaining);
                crate::leaf_view_chunked::nearest_one::avx512::nearest_one_avx512_arena_unchecked_f32::<
                    Self::Avx512F32Ops,
                    T,
                    K,
                >(tile_base, tile_len, query_ptr, best_dist, best_item);
                let tile_bytes =
                    K * tile_len * std::mem::size_of::<f32>() + tile_len * std::mem::size_of::<T>();
                tile_base = tile_base.add(tile_bytes);
                remaining -= tile_len;
            }

            return true;
        }

        false
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    #[inline(always)]
    /// Try an AVX512-specialized `nearest_n_within` leaf kernel.
    unsafe fn try_nearest_n_within_leaf_avx512<
        T,
        E,
        R,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    >(
        leaf: &LeafView<'_, A, T, K, B>,
        query_wide: &[Self::Output; K],
        max_dist: Self::Output,
        results: &mut R,
    ) -> bool
    where
        A: Axis<Coord = A> + 'static,
        T: crate::Content,
        Self::Output: Axis<Coord = Self::Output> + 'static,
        E: FromLeafCandidate<A, T, Self::Output, K>,
        R: ResultCollection<Self::Output, E>,
    {
        if TypeId::of::<A>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Avx512F64Ops>()
                != TypeId::of::<distance_metric_avx512::UnsupportedAvx512F64LeafOps>()
        {
            let leaf =
                &*(leaf as *const LeafView<'_, A, T, K, B> as *const LeafView<'_, f64, T, K, B>);
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f64; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f64);
            with_nearest_result_emitter!(results, E, f64, T, emit, {
                let mut emit_positioned =
                    |position, distance, item| emit(leaf.point(position), distance, item);
                crate::leaf_view_chunked::nearest_n_within::avx512::nearest_n_within_avx512_unchecked::<
                    Self::Avx512F64Ops,
                    T,
                    _,
                    EXCLUSIVE,
                    K,
                    B,
                >(leaf, query_wide, max_dist, &mut emit_positioned);
            });

            return true;
        }

        if TypeId::of::<A>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Avx512F32Ops>()
                != TypeId::of::<distance_metric_avx512::UnsupportedAvx512F32LeafOps>()
        {
            let leaf =
                &*(leaf as *const LeafView<'_, A, T, K, B> as *const LeafView<'_, f32, T, K, B>);
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f32; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f32);
            with_nearest_result_emitter!(results, E, f32, T, emit, {
                let mut emit_positioned =
                    |position, distance, item| emit(leaf.point(position), distance, item);
                crate::leaf_view_chunked::nearest_n_within::avx512::nearest_n_within_avx512_unchecked_f32::<
                    Self::Avx512F32Ops,
                    T,
                    _,
                    EXCLUSIVE,
                    K,
                    B,
                >(leaf, query_wide, max_dist, &mut emit_positioned);
            });

            return true;
        }

        false
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    #[inline(always)]
    /// Try an AVX512-specialized `nearest_n_within` arena kernel.
    unsafe fn try_nearest_n_within_arena_avx512<T, E, R, const EXCLUSIVE: bool, const K: usize>(
        arena: &LeafArena<'_, A, T, K>,
        query_wide: &[Self::Output; K],
        max_dist: Self::Output,
        results: &mut R,
    ) -> bool
    where
        A: Axis<Coord = A> + 'static,
        T: crate::Content,
        Self::Output: Axis<Coord = Self::Output> + 'static,
        E: FromLeafCandidate<A, T, Self::Output, K>,
        R: ResultCollection<Self::Output, E>,
    {
        if TypeId::of::<A>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Avx512F64Ops>()
                != TypeId::of::<distance_metric_avx512::UnsupportedAvx512F64LeafOps>()
        {
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f64; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f64);
            with_nearest_result_emitter!(results, E, f64, T, emit, {
                let mut tile_base = arena.as_ptr();
                let mut remaining = arena.len();

                while remaining != 0 {
                    let tile_len = crate::leaf_view::leaf_arena_tile_len(remaining);
                    let mut emit_tile = |position, distance, item| {
                        emit(
                            point_from_leaf_arena_tile::<f64, K>(tile_base, tile_len, position),
                            distance,
                            item,
                        )
                    };
                    crate::leaf_view_chunked::nearest_n_within::avx512::nearest_n_within_avx512_arena_unchecked::<
                        Self::Avx512F64Ops,
                        T,
                        _,
                        EXCLUSIVE,
                        K,
                    >(tile_base, tile_len, query_wide, max_dist, &mut emit_tile);
                    let tile_bytes = K * tile_len * std::mem::size_of::<f64>()
                        + tile_len * std::mem::size_of::<T>();
                    tile_base = tile_base.add(tile_bytes);
                    remaining -= tile_len;
                }
            });

            return true;
        }

        if TypeId::of::<A>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Avx512F32Ops>()
                != TypeId::of::<distance_metric_avx512::UnsupportedAvx512F32LeafOps>()
        {
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f32; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f32);
            with_nearest_result_emitter!(results, E, f32, T, emit, {
                let mut tile_base = arena.as_ptr();
                let mut remaining = arena.len();

                while remaining != 0 {
                    let tile_len = crate::leaf_view::leaf_arena_tile_len(remaining);
                    let mut emit_tile = |position, distance, item| {
                        emit(
                            point_from_leaf_arena_tile::<f32, K>(tile_base, tile_len, position),
                            distance,
                            item,
                        )
                    };
                    crate::leaf_view_chunked::nearest_n_within::avx512::nearest_n_within_avx512_arena_unchecked_f32::<
                        Self::Avx512F32Ops,
                        T,
                        _,
                        EXCLUSIVE,
                        K,
                    >(tile_base, tile_len, query_wide, max_dist, &mut emit_tile);
                    let tile_bytes = K * tile_len * std::mem::size_of::<f32>()
                        + tile_len * std::mem::size_of::<T>();
                    tile_base = tile_base.add(tile_bytes);
                    remaining -= tile_len;
                }
            });

            return true;
        }

        false
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    #[inline(always)]
    /// Try an AVX512-specialized `best_n_within` leaf kernel.
    unsafe fn try_best_n_within_leaf_avx512<
        T,
        R,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    >(
        leaf: &LeafView<'_, A, T, K, B>,
        query_wide: &[Self::Output; K],
        max_dist: Self::Output,
        threshold_item: Option<T>,
        results: &mut R,
    ) -> bool
    where
        A: Axis<Coord = A> + 'static,
        T: crate::Content + PartialOrd,
        Self::Output: Axis<Coord = Self::Output> + 'static,
        R: BestNeighbourResultCollection<Self::Output, T>,
    {
        if TypeId::of::<A>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Avx512F64Ops>()
                != TypeId::of::<distance_metric_avx512::UnsupportedAvx512F64LeafOps>()
        {
            let leaf =
                &*(leaf as *const LeafView<'_, A, T, K, B> as *const LeafView<'_, f64, T, K, B>);
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f64; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f64);
            with_best_result_emitter!(results, threshold_item, f64, T, emit, {
                crate::leaf_view_chunked::best_n_within::avx512::best_n_within_avx512_unchecked::<
                    Self::Avx512F64Ops,
                    T,
                    _,
                    EXCLUSIVE,
                    K,
                    B,
                >(leaf, query_wide, max_dist, &mut emit);
            });

            return true;
        }

        if TypeId::of::<A>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Avx512F32Ops>()
                != TypeId::of::<distance_metric_avx512::UnsupportedAvx512F32LeafOps>()
        {
            let leaf =
                &*(leaf as *const LeafView<'_, A, T, K, B> as *const LeafView<'_, f32, T, K, B>);
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f32; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f32);
            with_best_result_emitter!(results, threshold_item, f32, T, emit, {
                crate::leaf_view_chunked::best_n_within::avx512::best_n_within_avx512_unchecked_f32::<
                    Self::Avx512F32Ops,
                    T,
                    _,
                    EXCLUSIVE,
                    K,
                    B,
                >(leaf, query_wide, max_dist, &mut emit);
            });

            return true;
        }

        false
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    #[inline(always)]
    /// Try an AVX512-specialized `best_n_within` arena kernel.
    unsafe fn try_best_n_within_arena_avx512<T, R, const EXCLUSIVE: bool, const K: usize>(
        arena: &LeafArena<'_, A, T, K>,
        query_wide: &[Self::Output; K],
        max_dist: Self::Output,
        threshold_item: Option<T>,
        results: &mut R,
    ) -> bool
    where
        A: Axis<Coord = A> + 'static,
        T: crate::Content + PartialOrd,
        Self::Output: Axis<Coord = Self::Output> + 'static,
        R: BestNeighbourResultCollection<Self::Output, T>,
    {
        if TypeId::of::<A>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Avx512F64Ops>()
                != TypeId::of::<distance_metric_avx512::UnsupportedAvx512F64LeafOps>()
        {
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f64; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f64);
            with_best_result_emitter!(results, threshold_item, f64, T, emit, {
                let mut tile_base = arena.as_ptr();
                let mut remaining = arena.len();

                while remaining != 0 {
                    let tile_len = crate::leaf_view::leaf_arena_tile_len(remaining);
                    crate::leaf_view_chunked::best_n_within::avx512::best_n_within_avx512_arena_unchecked::<
                        Self::Avx512F64Ops,
                        T,
                        _,
                        EXCLUSIVE,
                        K,
                    >(tile_base, tile_len, query_wide, max_dist, &mut emit);
                    let tile_bytes = K * tile_len * std::mem::size_of::<f64>()
                        + tile_len * std::mem::size_of::<T>();
                    tile_base = tile_base.add(tile_bytes);
                    remaining -= tile_len;
                }
            });

            return true;
        }

        if TypeId::of::<A>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Avx512F32Ops>()
                != TypeId::of::<distance_metric_avx512::UnsupportedAvx512F32LeafOps>()
        {
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f32; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f32);
            with_best_result_emitter!(results, threshold_item, f32, T, emit, {
                let mut tile_base = arena.as_ptr();
                let mut remaining = arena.len();

                while remaining != 0 {
                    let tile_len = crate::leaf_view::leaf_arena_tile_len(remaining);
                    crate::leaf_view_chunked::best_n_within::avx512::best_n_within_avx512_arena_unchecked_f32::<
                        Self::Avx512F32Ops,
                        T,
                        _,
                        EXCLUSIVE,
                        K,
                    >(tile_base, tile_len, query_wide, max_dist, &mut emit);
                    let tile_bytes = K * tile_len * std::mem::size_of::<f32>()
                        + tile_len * std::mem::size_of::<T>();
                    tile_base = tile_base.add(tile_bytes);
                    remaining -= tile_len;
                }
            });

            return true;
        }

        false
    }
}

/// AVX2 extension hooks.
#[doc(hidden)]
pub trait DistanceMetricAvx2<A: Copy>: DistanceMetricScalar<A> {
    /// Whether a specialized AVX2 path is provided by this metric impl.
    const HAS_AVX2_SPECIALIZATION: bool = false;

    /// Type that provides implementations of the AVX2 f64 leaf ops.
    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    type Avx2F64Ops: distance_metric_avx2::Avx2F64LeafOps + 'static;

    /// Type that provides implementations of the AVX2 f32 leaf ops.
    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    type Avx2F32Ops: distance_metric_avx2::Avx2F32LeafOps + 'static;

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    #[inline(always)]
    /// Try an AVX2-specialized `nearest_n_within` leaf kernel.
    unsafe fn try_nearest_n_within_leaf_avx2<
        T,
        E,
        R,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    >(
        leaf: &LeafView<'_, A, T, K, B>,
        query_wide: &[Self::Output; K],
        max_dist: Self::Output,
        results: &mut R,
    ) -> bool
    where
        A: Axis<Coord = A> + 'static,
        T: crate::Content,
        Self::Output: Axis<Coord = Self::Output> + 'static,
        E: FromLeafCandidate<A, T, Self::Output, K>,
        R: ResultCollection<Self::Output, E>,
    {
        if TypeId::of::<A>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Avx2F64Ops>()
                != TypeId::of::<distance_metric_avx2::UnsupportedAvx2F64LeafOps>()
        {
            let leaf =
                &*(leaf as *const LeafView<'_, A, T, K, B> as *const LeafView<'_, f64, T, K, B>);
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f64; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f64);
            with_nearest_result_emitter!(results, E, f64, T, emit, {
                let mut emit_positioned =
                    |position, distance, item| emit(leaf.point(position), distance, item);
                crate::leaf_view_chunked::nearest_n_within::avx2::nearest_n_within_avx2_unchecked_f64::<
                    Self::Avx2F64Ops,
                    T,
                    _,
                    EXCLUSIVE,
                    K,
                    B,
                >(leaf, query_wide, max_dist, &mut emit_positioned);
            });

            return true;
        }

        if TypeId::of::<A>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Avx2F32Ops>()
                != TypeId::of::<distance_metric_avx2::UnsupportedAvx2F32LeafOps>()
        {
            let leaf =
                &*(leaf as *const LeafView<'_, A, T, K, B> as *const LeafView<'_, f32, T, K, B>);
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f32; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f32);
            with_nearest_result_emitter!(results, E, f32, T, emit, {
                let mut emit_positioned =
                    |position, distance, item| emit(leaf.point(position), distance, item);
                crate::leaf_view_chunked::nearest_n_within::avx2::nearest_n_within_avx2_unchecked_f32::<
                    Self::Avx2F32Ops,
                    T,
                    _,
                    EXCLUSIVE,
                    K,
                    B,
                >(leaf, query_wide, max_dist, &mut emit_positioned);
            });

            return true;
        }

        false
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    #[inline(always)]
    /// Try an AVX2-specialized `nearest_n_within` arena kernel.
    unsafe fn try_nearest_n_within_arena_avx2<T, E, R, const EXCLUSIVE: bool, const K: usize>(
        arena: &LeafArena<'_, A, T, K>,
        query_wide: &[Self::Output; K],
        max_dist: Self::Output,
        results: &mut R,
    ) -> bool
    where
        A: Axis<Coord = A> + 'static,
        T: crate::Content,
        Self::Output: Axis<Coord = Self::Output> + 'static,
        E: FromLeafCandidate<A, T, Self::Output, K>,
        R: ResultCollection<Self::Output, E>,
    {
        if TypeId::of::<A>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Avx2F64Ops>()
                != TypeId::of::<distance_metric_avx2::UnsupportedAvx2F64LeafOps>()
        {
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f64; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f64);
            with_nearest_result_emitter!(results, E, f64, T, emit, {
                let mut tile_base = arena.as_ptr();
                let mut remaining = arena.len();

                while remaining != 0 {
                    let tile_len = crate::leaf_view::leaf_arena_tile_len(remaining);
                    let mut emit_tile = |position, distance, item| {
                        emit(
                            point_from_leaf_arena_tile::<f64, K>(tile_base, tile_len, position),
                            distance,
                            item,
                        )
                    };
                    crate::leaf_view_chunked::nearest_n_within::avx2::nearest_n_within_avx2_arena_unchecked_f64::<
                        Self::Avx2F64Ops,
                        T,
                        _,
                        EXCLUSIVE,
                        K,
                    >(tile_base, tile_len, query_wide, max_dist, &mut emit_tile);
                    let tile_bytes = K * tile_len * std::mem::size_of::<f64>()
                        + tile_len * std::mem::size_of::<T>();
                    tile_base = tile_base.add(tile_bytes);
                    remaining -= tile_len;
                }
            });

            return true;
        }

        if TypeId::of::<A>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Avx2F32Ops>()
                != TypeId::of::<distance_metric_avx2::UnsupportedAvx2F32LeafOps>()
        {
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f32; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f32);
            with_nearest_result_emitter!(results, E, f32, T, emit, {
                let mut tile_base = arena.as_ptr();
                let mut remaining = arena.len();

                while remaining != 0 {
                    let tile_len = crate::leaf_view::leaf_arena_tile_len(remaining);
                    let mut emit_tile = |position, distance, item| {
                        emit(
                            point_from_leaf_arena_tile::<f32, K>(tile_base, tile_len, position),
                            distance,
                            item,
                        )
                    };
                    crate::leaf_view_chunked::nearest_n_within::avx2::nearest_n_within_avx2_arena_unchecked_f32::<
                        Self::Avx2F32Ops,
                        T,
                        _,
                        EXCLUSIVE,
                        K,
                    >(tile_base, tile_len, query_wide, max_dist, &mut emit_tile);
                    let tile_bytes = K * tile_len * std::mem::size_of::<f32>()
                        + tile_len * std::mem::size_of::<T>();
                    tile_base = tile_base.add(tile_bytes);
                    remaining -= tile_len;
                }
            });

            return true;
        }

        false
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    #[inline(always)]
    /// Try an AVX2-specialized `best_n_within` leaf kernel.
    unsafe fn try_best_n_within_leaf_avx2<
        T,
        R,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    >(
        leaf: &LeafView<'_, A, T, K, B>,
        query_wide: &[Self::Output; K],
        max_dist: Self::Output,
        threshold_item: Option<T>,
        results: &mut R,
    ) -> bool
    where
        A: Axis<Coord = A> + 'static,
        T: crate::Content + PartialOrd,
        Self::Output: Axis<Coord = Self::Output> + 'static,
        R: BestNeighbourResultCollection<Self::Output, T>,
    {
        if TypeId::of::<A>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Avx2F64Ops>()
                != TypeId::of::<distance_metric_avx2::UnsupportedAvx2F64LeafOps>()
        {
            let leaf =
                &*(leaf as *const LeafView<'_, A, T, K, B> as *const LeafView<'_, f64, T, K, B>);
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f64; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f64);
            with_best_result_emitter!(results, threshold_item, f64, T, emit, {
                crate::leaf_view_chunked::best_n_within::avx2::best_n_within_avx2_unchecked_f64::<
                    Self::Avx2F64Ops,
                    T,
                    _,
                    EXCLUSIVE,
                    K,
                    B,
                >(leaf, query_wide, max_dist, &mut emit);
            });

            return true;
        }

        if TypeId::of::<A>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Avx2F32Ops>()
                != TypeId::of::<distance_metric_avx2::UnsupportedAvx2F32LeafOps>()
        {
            let leaf =
                &*(leaf as *const LeafView<'_, A, T, K, B> as *const LeafView<'_, f32, T, K, B>);
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f32; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f32);
            with_best_result_emitter!(results, threshold_item, f32, T, emit, {
                crate::leaf_view_chunked::best_n_within::avx2::best_n_within_avx2_unchecked_f32::<
                    Self::Avx2F32Ops,
                    T,
                    _,
                    EXCLUSIVE,
                    K,
                    B,
                >(leaf, query_wide, max_dist, &mut emit);
            });

            return true;
        }

        false
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    #[inline(always)]
    /// Try an AVX2-specialized `best_n_within` arena kernel.
    unsafe fn try_best_n_within_arena_avx2<T, R, const EXCLUSIVE: bool, const K: usize>(
        arena: &LeafArena<'_, A, T, K>,
        query_wide: &[Self::Output; K],
        max_dist: Self::Output,
        threshold_item: Option<T>,
        results: &mut R,
    ) -> bool
    where
        A: Axis<Coord = A> + 'static,
        T: crate::Content + PartialOrd,
        Self::Output: Axis<Coord = Self::Output> + 'static,
        R: BestNeighbourResultCollection<Self::Output, T>,
    {
        if TypeId::of::<A>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Avx2F64Ops>()
                != TypeId::of::<distance_metric_avx2::UnsupportedAvx2F64LeafOps>()
        {
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f64; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f64);
            with_best_result_emitter!(results, threshold_item, f64, T, emit, {
                let mut tile_base = arena.as_ptr();
                let mut remaining = arena.len();

                while remaining != 0 {
                    let tile_len = crate::leaf_view::leaf_arena_tile_len(remaining);
                    crate::leaf_view_chunked::best_n_within::avx2::best_n_within_avx2_arena_unchecked_f64::<
                        Self::Avx2F64Ops,
                        T,
                        _,
                        EXCLUSIVE,
                        K,
                    >(tile_base, tile_len, query_wide, max_dist, &mut emit);
                    let tile_bytes = K * tile_len * std::mem::size_of::<f64>()
                        + tile_len * std::mem::size_of::<T>();
                    tile_base = tile_base.add(tile_bytes);
                    remaining -= tile_len;
                }
            });

            return true;
        }

        if TypeId::of::<A>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Avx2F32Ops>()
                != TypeId::of::<distance_metric_avx2::UnsupportedAvx2F32LeafOps>()
        {
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f32; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f32);
            with_best_result_emitter!(results, threshold_item, f32, T, emit, {
                let mut tile_base = arena.as_ptr();
                let mut remaining = arena.len();

                while remaining != 0 {
                    let tile_len = crate::leaf_view::leaf_arena_tile_len(remaining);
                    crate::leaf_view_chunked::best_n_within::avx2::best_n_within_avx2_arena_unchecked_f32::<
                        Self::Avx2F32Ops,
                        T,
                        _,
                        EXCLUSIVE,
                        K,
                    >(tile_base, tile_len, query_wide, max_dist, &mut emit);
                    let tile_bytes = K * tile_len * std::mem::size_of::<f32>()
                        + tile_len * std::mem::size_of::<T>();
                    tile_base = tile_base.add(tile_bytes);
                    remaining -= tile_len;
                }
            });

            return true;
        }

        false
    }
}

/// NEON extension hooks.
#[doc(hidden)]
pub trait DistanceMetricNeon<A: Copy>: DistanceMetricScalar<A> {
    /// Whether a specialized NEON path is provided by this metric impl.
    const HAS_NEON_SPECIALIZATION: bool = false;

    /// Type that provides implementations of the NEON f64 leaf ops.
    #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
    type NeonF64Ops: distance_metric_neon::NeonF64LeafOps + 'static;

    /// Type that provides implementations of the NEON f32 leaf ops.
    #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
    type NeonF32Ops: distance_metric_neon::NeonF32LeafOps + 'static;

    #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
    #[inline(always)]
    /// Try a NEON-specialized `nearest_n_within` leaf kernel.
    unsafe fn try_nearest_n_within_leaf_neon<
        T,
        E,
        R,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    >(
        leaf: &LeafView<'_, A, T, K, B>,
        query_wide: &[Self::Output; K],
        max_dist: Self::Output,
        results: &mut R,
    ) -> bool
    where
        A: Axis<Coord = A> + 'static,
        T: crate::Content,
        Self::Output: Axis<Coord = Self::Output> + 'static,
        E: FromLeafCandidate<A, T, Self::Output, K>,
        R: ResultCollection<Self::Output, E>,
    {
        if TypeId::of::<A>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f64>()
            && TypeId::of::<Self::NeonF64Ops>()
                != TypeId::of::<distance_metric_neon::UnsupportedNeonF64LeafOps>()
        {
            let leaf =
                &*(leaf as *const LeafView<'_, A, T, K, B> as *const LeafView<'_, f64, T, K, B>);
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f64; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f64);
            with_nearest_result_emitter!(results, E, f64, T, emit, {
                let mut emit_positioned =
                    |position, distance, item| emit(leaf.point(position), distance, item);
                crate::leaf_view_chunked::nearest_n_within::neon::nearest_n_within_neon_unchecked_f64::<
                    Self::NeonF64Ops,
                    T,
                    _,
                    EXCLUSIVE,
                    K,
                    B,
                >(leaf, query_wide, max_dist, &mut emit_positioned);
            });

            return true;
        }

        if TypeId::of::<A>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f32>()
            && TypeId::of::<Self::NeonF32Ops>()
                != TypeId::of::<distance_metric_neon::UnsupportedNeonF32LeafOps>()
        {
            let leaf =
                &*(leaf as *const LeafView<'_, A, T, K, B> as *const LeafView<'_, f32, T, K, B>);
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f32; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f32);
            with_nearest_result_emitter!(results, E, f32, T, emit, {
                let mut emit_positioned =
                    |position, distance, item| emit(leaf.point(position), distance, item);
                crate::leaf_view_chunked::nearest_n_within::neon::nearest_n_within_neon_unchecked_f32::<
                    Self::NeonF32Ops,
                    T,
                    _,
                    EXCLUSIVE,
                    K,
                    B,
                >(leaf, query_wide, max_dist, &mut emit_positioned);
            });

            return true;
        }

        false
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
    #[inline(always)]
    /// Try a NEON-specialized `nearest_n_within` arena kernel.
    unsafe fn try_nearest_n_within_arena_neon<T, E, R, const EXCLUSIVE: bool, const K: usize>(
        arena: &LeafArena<'_, A, T, K>,
        query_wide: &[Self::Output; K],
        max_dist: Self::Output,
        results: &mut R,
    ) -> bool
    where
        A: Axis<Coord = A> + 'static,
        T: crate::Content,
        Self::Output: Axis<Coord = Self::Output> + 'static,
        E: FromLeafCandidate<A, T, Self::Output, K>,
        R: ResultCollection<Self::Output, E>,
    {
        if TypeId::of::<A>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f64>()
            && TypeId::of::<Self::NeonF64Ops>()
                != TypeId::of::<distance_metric_neon::UnsupportedNeonF64LeafOps>()
        {
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f64; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f64);
            with_nearest_result_emitter!(results, E, f64, T, emit, {
                let mut tile_base = arena.as_ptr();
                let mut remaining = arena.len();

                while remaining != 0 {
                    let tile_len = crate::leaf_view::leaf_arena_tile_len(remaining);
                    let mut emit_tile = |position, distance, item| {
                        emit(
                            point_from_leaf_arena_tile::<f64, K>(tile_base, tile_len, position),
                            distance,
                            item,
                        )
                    };
                    crate::leaf_view_chunked::nearest_n_within::neon::nearest_n_within_neon_arena_unchecked_f64::<
                        Self::NeonF64Ops,
                        T,
                        _,
                        EXCLUSIVE,
                        K,
                    >(tile_base, tile_len, query_wide, max_dist, &mut emit_tile);
                    let tile_bytes = K * tile_len * std::mem::size_of::<f64>()
                        + tile_len * std::mem::size_of::<T>();
                    tile_base = tile_base.add(tile_bytes);
                    remaining -= tile_len;
                }
            });

            return true;
        }

        if TypeId::of::<A>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f32>()
            && TypeId::of::<Self::NeonF32Ops>()
                != TypeId::of::<distance_metric_neon::UnsupportedNeonF32LeafOps>()
        {
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f32; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f32);
            with_nearest_result_emitter!(results, E, f32, T, emit, {
                let mut tile_base = arena.as_ptr();
                let mut remaining = arena.len();

                while remaining != 0 {
                    let tile_len = crate::leaf_view::leaf_arena_tile_len(remaining);
                    let mut emit_tile = |position, distance, item| {
                        emit(
                            point_from_leaf_arena_tile::<f32, K>(tile_base, tile_len, position),
                            distance,
                            item,
                        )
                    };
                    crate::leaf_view_chunked::nearest_n_within::neon::nearest_n_within_neon_arena_unchecked_f32::<
                        Self::NeonF32Ops,
                        T,
                        _,
                        EXCLUSIVE,
                        K,
                    >(tile_base, tile_len, query_wide, max_dist, &mut emit_tile);
                    let tile_bytes = K * tile_len * std::mem::size_of::<f32>()
                        + tile_len * std::mem::size_of::<T>();
                    tile_base = tile_base.add(tile_bytes);
                    remaining -= tile_len;
                }
            });

            return true;
        }

        false
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
    #[inline(always)]
    /// Try a NEON-specialized `best_n_within` leaf kernel.
    unsafe fn try_best_n_within_leaf_neon<
        T,
        R,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    >(
        leaf: &LeafView<'_, A, T, K, B>,
        query_wide: &[Self::Output; K],
        max_dist: Self::Output,
        threshold_item: Option<T>,
        results: &mut R,
    ) -> bool
    where
        A: Axis<Coord = A> + 'static,
        T: crate::Content + PartialOrd,
        Self::Output: Axis<Coord = Self::Output> + 'static,
        R: BestNeighbourResultCollection<Self::Output, T>,
    {
        if TypeId::of::<A>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f64>()
            && TypeId::of::<Self::NeonF64Ops>()
                != TypeId::of::<distance_metric_neon::UnsupportedNeonF64LeafOps>()
        {
            let leaf =
                &*(leaf as *const LeafView<'_, A, T, K, B> as *const LeafView<'_, f64, T, K, B>);
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f64; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f64);
            with_best_result_emitter!(results, threshold_item, f64, T, emit, {
                crate::leaf_view_chunked::best_n_within::neon::best_n_within_neon_unchecked_f64::<
                    Self::NeonF64Ops,
                    T,
                    _,
                    EXCLUSIVE,
                    K,
                    B,
                >(leaf, query_wide, max_dist, &mut emit);
            });

            return true;
        }

        if TypeId::of::<A>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f32>()
            && TypeId::of::<Self::NeonF32Ops>()
                != TypeId::of::<distance_metric_neon::UnsupportedNeonF32LeafOps>()
        {
            let leaf =
                &*(leaf as *const LeafView<'_, A, T, K, B> as *const LeafView<'_, f32, T, K, B>);
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f32; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f32);
            with_best_result_emitter!(results, threshold_item, f32, T, emit, {
                crate::leaf_view_chunked::best_n_within::neon::best_n_within_neon_unchecked_f32::<
                    Self::NeonF32Ops,
                    T,
                    _,
                    EXCLUSIVE,
                    K,
                    B,
                >(leaf, query_wide, max_dist, &mut emit);
            });

            return true;
        }

        false
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
    #[inline(always)]
    /// Try a NEON-specialized `best_n_within` arena kernel.
    unsafe fn try_best_n_within_arena_neon<T, R, const EXCLUSIVE: bool, const K: usize>(
        arena: &LeafArena<'_, A, T, K>,
        query_wide: &[Self::Output; K],
        max_dist: Self::Output,
        threshold_item: Option<T>,
        results: &mut R,
    ) -> bool
    where
        A: Axis<Coord = A> + 'static,
        T: crate::Content + PartialOrd,
        Self::Output: Axis<Coord = Self::Output> + 'static,
        R: BestNeighbourResultCollection<Self::Output, T>,
    {
        if TypeId::of::<A>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f64>()
            && TypeId::of::<Self::NeonF64Ops>()
                != TypeId::of::<distance_metric_neon::UnsupportedNeonF64LeafOps>()
        {
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f64; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f64);
            with_best_result_emitter!(results, threshold_item, f64, T, emit, {
                let mut tile_base = arena.as_ptr();
                let mut remaining = arena.len();

                while remaining != 0 {
                    let tile_len = crate::leaf_view::leaf_arena_tile_len(remaining);
                    crate::leaf_view_chunked::best_n_within::neon::best_n_within_neon_arena_unchecked_f64::<
                        Self::NeonF64Ops,
                        T,
                        _,
                        EXCLUSIVE,
                        K,
                    >(tile_base, tile_len, query_wide, max_dist, &mut emit);
                    let tile_bytes = K * tile_len * std::mem::size_of::<f64>()
                        + tile_len * std::mem::size_of::<T>();
                    tile_base = tile_base.add(tile_bytes);
                    remaining -= tile_len;
                }
            });

            return true;
        }

        if TypeId::of::<A>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f32>()
            && TypeId::of::<Self::NeonF32Ops>()
                != TypeId::of::<distance_metric_neon::UnsupportedNeonF32LeafOps>()
        {
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f32; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f32);
            with_best_result_emitter!(results, threshold_item, f32, T, emit, {
                let mut tile_base = arena.as_ptr();
                let mut remaining = arena.len();

                while remaining != 0 {
                    let tile_len = crate::leaf_view::leaf_arena_tile_len(remaining);
                    crate::leaf_view_chunked::best_n_within::neon::best_n_within_neon_arena_unchecked_f32::<
                        Self::NeonF32Ops,
                        T,
                        _,
                        EXCLUSIVE,
                        K,
                    >(tile_base, tile_len, query_wide, max_dist, &mut emit);
                    let tile_bytes = K * tile_len * std::mem::size_of::<f32>()
                        + tile_len * std::mem::size_of::<T>();
                    tile_base = tile_base.add(tile_bytes);
                    remaining -= tile_len;
                }
            });

            return true;
        }

        false
    }
}

/// Trait representing a distance metric that can be used in a `KdTree` query.
#[doc(hidden)]
pub trait DistanceMetric<A: Copy>:
    DistanceMetricScalar<A> + DistanceMetricAvx512<A> + DistanceMetricAvx2<A> + DistanceMetricNeon<A>
{
    /// Autovec/scalar Block3 backtrack mask generation.
    #[inline(always)]
    fn backtrack_block3<const K: usize>(
        query_wide: Self::Output,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: Self::Output,
        rd: Self::Output,
        best_dist: Self::Output,
    ) -> u8
    where
        A: Axis<Coord = A>,
        Self: Sized,
        Self::Output: Axis<Coord = Self::Output>
            + std::ops::Add<Output = Self::Output>
            + std::ops::Sub<Output = Self::Output>,
    {
        crate::stem_strategy::donnelly::simd_full::backtrack_traits::autovec_backtrack_block3::<
            Self::Output,
            A,
            Self,
            K,
        >(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }

    /// Block3 bounds-aware backtrack mask generation.
    #[inline(always)]
    fn backtrack_block3_with_bounds<const K: usize>(
        query_wide: Self::Output,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        parent_lower_bound: Self::Output,
        parent_upper_bound: Self::Output,
        old_off: Self::Output,
        rd: Self::Output,
        best_dist: Self::Output,
    ) -> u8
    where
        A: Axis<Coord = A>,
        Self: Sized,
        Self::Output: Axis<Coord = Self::Output>,
    {
        crate::stem_strategy::donnelly::simd_full::backtrack_traits::autovec_backtrack_block3_with_bounds::<
            A,
            Self::Output,
            Self,
            K,
        >(
            query_wide,
            stems_ptr,
            block_base_idx,
            parent_lower_bound,
            parent_upper_bound,
            old_off,
            rd,
            best_dist,
        )
    }

    /// Block3 bounds-aware backtrack state fill.
    #[inline(always)]
    fn fill_block3_values_and_bounds<const K: usize>(
        query_wide: Self::Output,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        parent_lower_bound: Self::Output,
        parent_upper_bound: Self::Output,
        old_off: Self::Output,
        rd: Self::Output,
        best_dist: Self::Output,
        new_off_values: &mut [Self::Output; 8],
        rd_values: &mut [Self::Output; 8],
        lower_bounds: &mut [Self::Output; 8],
        upper_bounds: &mut [Self::Output; 8],
    ) -> u8
    where
        A: Axis<Coord = A>,
        Self: Sized,
        Self::Output: Axis<Coord = Self::Output>,
    {
        crate::stem_strategy::donnelly::simd_full::backtrack_traits::autovec_fill_block3_values_and_bounds::<
            A,
            Self::Output,
            Self,
            K,
        >(
            query_wide,
            stems_ptr,
            block_base_idx,
            parent_lower_bound,
            parent_upper_bound,
            old_off,
            rd,
            best_dist,
            new_off_values,
            rd_values,
            lower_bounds,
            upper_bounds,
        )
    }

    /// Autovec/scalar Block4 backtrack mask generation.
    #[inline(always)]
    fn backtrack_block4<const K: usize>(
        query_wide: Self::Output,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: Self::Output,
        rd: Self::Output,
        best_dist: Self::Output,
    ) -> u16
    where
        A: Axis<Coord = A>,
        Self: Sized,
        Self::Output: Axis<Coord = Self::Output>
            + std::ops::Add<Output = Self::Output>
            + std::ops::Sub<Output = Self::Output>,
    {
        crate::stem_strategy::donnelly::simd_full::backtrack_traits::autovec_backtrack_block4::<
            Self::Output,
            A,
            Self,
            K,
        >(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }
}

impl<T, A: Copy> DistanceMetric<A> for T where
    T: DistanceMetricScalar<A>
        + DistanceMetricAvx512<A>
        + DistanceMetricAvx2<A>
        + DistanceMetricNeon<A>
{
}

#[cfg(test)]
mod tests {
    use super::{
        Chebyshev, DistanceMetric, DistanceMetricScalar, Manhattan, Minkowski, QueryMetric,
        SquaredEuclidean,
    };

    #[test]
    fn v3_squared_euclidean_f64_works() {
        type M = SquaredEuclidean<f64>;
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 2.0, -1.0];

        let aw = a.map(M::widen_coord);
        let bw = b.map(M::widen_coord);
        let d = <M as DistanceMetricScalar<f64>>::dist::<3>(&aw, &bw);
        assert_eq!(d, 25.0);
    }

    #[test]
    fn v3_manhattan_f64_works() {
        type M = Manhattan<f64>;
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 2.0, -1.0];

        let aw = a.map(M::widen_coord);
        let bw = b.map(M::widen_coord);
        let d = <M as DistanceMetricScalar<f64>>::dist::<3>(&aw, &bw);
        assert_eq!(d, 7.0);
    }

    #[test]
    fn v3_manhattan_rect_dist_update_uses_offsets_directly() {
        type M = Manhattan<f64>;
        let off = [0.0, 2.0, 5.0];

        assert_eq!(
            <M as DistanceMetricScalar<f64>>::rect_dist_after_update(7.0, &off, 1, 4.0),
            9.0
        );
    }

    #[test]
    fn v3_chebyshev_f64_works() {
        type M = Chebyshev<f64>;
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 2.5, -1.0];

        let aw = a.map(M::widen_coord);
        let bw = b.map(M::widen_coord);
        let d = <M as DistanceMetricScalar<f64>>::dist::<3>(&aw, &bw);
        assert_eq!(d, 4.0);
    }

    #[test]
    fn v3_chebyshev_rect_dist_update_rescans_offsets_directly() {
        type M = Chebyshev<f64>;
        let off = [0.0, 2.0, 5.0];

        assert_eq!(
            <M as DistanceMetricScalar<f64>>::rect_dist_after_update(5.0, &off, 2, 1.0),
            2.0
        );
        assert_eq!(
            <M as DistanceMetricScalar<f64>>::rect_dist_after_update(5.0, &off, 0, 6.0),
            6.0
        );
    }

    #[test]
    fn v3_minkowski_3_f64_works() {
        type M = Minkowski<3, f64>;
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 2.5, -1.0];

        let aw = a.map(M::widen_coord);
        let bw = b.map(M::widen_coord);
        let d = <M as DistanceMetricScalar<f64>>::dist::<3>(&aw, &bw);
        assert_eq!(d, 91.125);
    }

    #[test]
    fn v3_unified_bound_is_satisfied() {
        fn assert_unified<M: DistanceMetric<f64>>() {}
        assert_unified::<SquaredEuclidean<f64>>();
        assert_unified::<Chebyshev<f64>>();
        assert_unified::<Minkowski<3, f64>>();
    }

    #[test]
    fn v3_query_metric_bound_is_satisfied() {
        fn assert_query_metric<M: QueryMetric<f64>>() {}
        assert_query_metric::<SquaredEuclidean<f64>>();
        assert_query_metric::<Chebyshev<f64>>();
        assert_query_metric::<Minkowski<3, f64>>();
    }
}

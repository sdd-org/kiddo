use std::ops::Add;

use crate::Axis;

use crate::dist::{
    DistanceMetricAvx2, DistanceMetricAvx512, DistanceMetricNeon, DistanceMetricScalar,
};

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
mod avx2;

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
mod avx512;

#[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
mod neon;

/// Manhattan / L1 distance metric, parameterized by output type `R`.
pub struct Manhattan<R>(core::marker::PhantomData<R>);

impl<A, R> DistanceMetricScalar<A> for Manhattan<R>
where
    A: Copy,
    R: Axis<Coord = R> + From<A> + Add<Output = R>,
{
    type Output = R;

    #[inline(always)]
    fn widen_coord(a: A) -> R {
        <R as From<A>>::from(a)
    }

    #[inline(always)]
    fn dist1(a: R, b: R) -> R {
        R::saturating_dist(a, b)
    }

    #[inline(always)]
    fn rect_dist_after_update<const K: usize>(
        rd: Self::Output,
        off: &[Self::Output; K],
        dim: usize,
        new_off: Self::Output,
    ) -> Self::Output {
        Self::Output::saturating_add(rd - off[dim], new_off)
    }
}

impl<A, R> DistanceMetricAvx512<A> for Manhattan<R>
where
    A: Copy,
    R: Axis<Coord = R> + From<A> + Add<Output = R>,
{
    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    type Avx512F64Ops = avx512::ManhattanAvx512F64LeafOps;

    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    type Avx512F32Ops = avx512::ManhattanAvx512F32LeafOps;
}

impl<A, R> DistanceMetricAvx2<A> for Manhattan<R>
where
    A: Copy,
    R: Axis<Coord = R> + From<A> + Add<Output = R>,
{
    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    type Avx2F64Ops = avx2::ManhattanAvx2F64LeafOps;

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    type Avx2F32Ops = avx2::ManhattanAvx2F32LeafOps;
}

impl<A, R> DistanceMetricNeon<A> for Manhattan<R>
where
    A: Copy,
    R: Axis<Coord = R> + From<A> + Add<Output = R>,
{
    #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
    type NeonF64Ops = neon::ManhattanNeonF64LeafOps;

    #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
    type NeonF32Ops = neon::ManhattanNeonF32LeafOps;
}

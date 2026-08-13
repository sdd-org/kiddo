use num_traits::Float;

use crate::Axis;

use crate::dist::{
    DistanceMetricAvx2, DistanceMetricAvx512, DistanceMetricNeon, DistanceMetricScalar,
    WideningCastFrom,
};

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
mod avx2;

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
mod avx512;

#[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
mod neon;

/// Generalized integer-power Minkowski distance metric.
///
/// This implementation returns the sum of powered absolute differences,
/// `sum(|dx|^P)`, rather than taking the final `1/P` root. That preserves
/// nearest-neighbour ordering while avoiding an unnecessary root in query hot
/// paths, just like [`crate::SquaredEuclidean`] does for `P = 2`.
#[doc(alias = "taxicab")]
#[doc(alias = "l1")]
#[doc(alias = "l2")]
#[doc(alias = "euclidean")]
pub struct Minkowski<const P: u32, R>(core::marker::PhantomData<R>);

impl<const P: u32, R> Minkowski<P, R> {
    const CHECK_P: () = {
        if P == 1 {
            panic!("Use `kiddo::Manhattan` instead of Minkowski<1>.");
        }
        if P == 2 {
            panic!("Use `kiddo::SquaredEuclidean` instead of Minkowski<2>.");
        }
    };
}

impl<A, R, const P: u32> DistanceMetricScalar<A> for Minkowski<P, R>
where
    A: Copy,
    R: Axis<Coord = R> + WideningCastFrom<A> + Float,
{
    type Output = R;

    #[inline(always)]
    fn widen_coord(a: A) -> R {
        R::widening_cast_from(a)
    }

    #[inline(always)]
    fn dist1(a: R, b: R) -> R {
        #[allow(clippy::let_unit_value)]
        let _ = Self::CHECK_P;
        (a - b).abs().powi(P as i32)
    }
}

impl<A, R, const P: u32> DistanceMetricAvx512<A> for Minkowski<P, R>
where
    A: Copy,
    R: Axis<Coord = R> + WideningCastFrom<A> + Float,
{
    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    type Avx512F64Ops = avx512::MinkowskiAvx512F64LeafOps<P>;

    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    type Avx512F32Ops = avx512::MinkowskiAvx512F32LeafOps<P>;
}

impl<A, R, const P: u32> DistanceMetricAvx2<A> for Minkowski<P, R>
where
    A: Copy,
    R: Axis<Coord = R> + WideningCastFrom<A> + Float,
{
    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    type Avx2F64Ops = avx2::MinkowskiAvx2F64LeafOps<P>;

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    type Avx2F32Ops = avx2::MinkowskiAvx2F32LeafOps<P>;
}

impl<A, R, const P: u32> DistanceMetricNeon<A> for Minkowski<P, R>
where
    A: Copy,
    R: Axis<Coord = R> + WideningCastFrom<A> + Float,
{
    #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
    type NeonF64Ops = neon::MinkowskiNeonF64LeafOps<P>;

    #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
    type NeonF32Ops = neon::MinkowskiNeonF32LeafOps<P>;
}

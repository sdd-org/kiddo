use std::ops::{Add, Mul};

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

/// Squared Euclidean distance metric, parameterized by output type `R`.
pub struct SquaredEuclidean<R>(core::marker::PhantomData<R>);

impl<A, R> DistanceMetricScalar<A> for SquaredEuclidean<R>
where
    A: Copy,
    R: Axis<Coord = R> + WideningCastFrom<A> + Mul<Output = R> + Add<Output = R>,
{
    type Output = R;

    #[inline(always)]
    fn widen_coord(a: A) -> R {
        R::widening_cast_from(a)
    }

    #[inline(always)]
    fn dist1(a: R, b: R) -> R {
        let d = if R::IS_SIGNED || a >= b { a - b } else { b - a };
        d * d
    }
}

impl<A, R> DistanceMetricAvx512<A> for SquaredEuclidean<R>
where
    A: Copy,
    R: Axis<Coord = R> + WideningCastFrom<A> + Mul<Output = R> + Add<Output = R>,
{
    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    type Avx512F64Ops = avx512::SquaredEuclideanAvx512F64LeafOps;

    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    type Avx512F32Ops = avx512::SquaredEuclideanAvx512F32LeafOps;
}

impl<A, R> DistanceMetricAvx2<A> for SquaredEuclidean<R>
where
    A: Copy,
    R: Axis<Coord = R> + WideningCastFrom<A> + Mul<Output = R> + Add<Output = R>,
{
    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    type Avx2F64Ops = avx2::SquaredEuclideanAvx2F64LeafOps;

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    type Avx2F32Ops = avx2::SquaredEuclideanAvx2F32LeafOps;
}

impl<A, R> DistanceMetricNeon<A> for SquaredEuclidean<R>
where
    A: Copy,
    R: Axis<Coord = R> + WideningCastFrom<A> + Mul<Output = R> + Add<Output = R>,
{
    #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
    type NeonF64Ops = neon::SquaredEuclideanNeonF64LeafOps;

    #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
    type NeonF32Ops = neon::SquaredEuclideanNeonF32LeafOps;
}

#[cfg(feature = "cargo_asm")]
pub mod cargo_asm {
    use super::SquaredEuclidean;
    use crate::dist::DistanceMetricScalar;

    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_squared_euclidean_dist1_f64_cargo_asm_hook(a: f64, b: f64) -> f64 {
        <SquaredEuclidean<f64> as DistanceMetricScalar<f64>>::dist1(a, b)
    }

    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_squared_euclidean_dist1_u16_cargo_asm_hook(a: u16, b: u16) -> u16 {
        <SquaredEuclidean<u16> as DistanceMetricScalar<u16>>::dist1(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::SquaredEuclidean;
    use crate::dist::DistanceMetricScalar;

    #[test]
    fn signed_dist1_squares_direct_delta_in_both_directions() {
        type Metric = SquaredEuclidean<f64>;

        assert_eq!(<Metric as DistanceMetricScalar<f64>>::dist1(5.0, 2.0), 9.0);
        assert_eq!(<Metric as DistanceMetricScalar<f64>>::dist1(2.0, 5.0), 9.0);
    }

    #[test]
    fn unsigned_dist1_uses_ordered_subtraction() {
        type Metric = SquaredEuclidean<u16>;

        assert_eq!(<Metric as DistanceMetricScalar<u16>>::dist1(5, 2), 9);
        assert_eq!(<Metric as DistanceMetricScalar<u16>>::dist1(2, 5), 9);
    }

    #[cfg(feature = "f16")]
    #[test]
    fn f16_coordinates_and_output_remain_supported() {
        use half::f16;

        type Metric = SquaredEuclidean<f16>;
        let a = [f16::from_f32(1.0), f16::from_f32(2.0)];
        let b = [f16::from_f32(1.5), f16::from_f32(2.5)];

        assert_eq!(
            <Metric as DistanceMetricScalar<f16>>::dist_raw(&a, &b),
            f16::from_f32(0.5)
        );
    }

    #[cfg(feature = "fixed")]
    #[test]
    fn signed_fixed_dist1_squares_direct_delta() {
        use fixed::{types::extra::U16, FixedI32};

        type Output = FixedI32<U16>;
        type Metric = SquaredEuclidean<Output>;
        let a = Output::from_num(-1.5);
        let b = Output::from_num(0.5);

        assert_eq!(
            <Metric as DistanceMetricScalar<Output>>::dist1(a, b),
            Output::from_num(4)
        );
        assert_eq!(
            <Metric as DistanceMetricScalar<Output>>::dist1(b, a),
            Output::from_num(4)
        );
    }
}

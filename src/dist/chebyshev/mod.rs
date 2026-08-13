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

/// Chebyshev / L-infinity distance metric, parameterized by output type `R`.
pub struct Chebyshev<R>(core::marker::PhantomData<R>);

impl<A, R> DistanceMetricScalar<A> for Chebyshev<R>
where
    A: Copy,
    R: Axis<Coord = R> + WideningCastFrom<A>,
{
    type Output = R;

    #[inline(always)]
    fn widen_coord(a: A) -> R {
        R::widening_cast_from(a)
    }

    #[inline(always)]
    fn dist1(a: R, b: R) -> R {
        R::saturating_dist(a, b)
    }

    #[inline(always)]
    fn combine_component(acc: &mut Self::Output, component: Self::Output) {
        if component > *acc {
            *acc = component;
        }
    }

    #[inline(always)]
    fn rect_dist_after_update<const K: usize>(
        _rd: Self::Output,
        off: &[Self::Output; K],
        dim: usize,
        new_off: Self::Output,
    ) -> Self::Output {
        let mut acc = Self::Output::zero();

        for axis in 0..K {
            let off_val = if axis == dim { new_off } else { off[axis] };
            <Self as DistanceMetricScalar<A>>::combine_component(&mut acc, off_val);
        }

        acc
    }

    #[inline(always)]
    fn rect_dist_after_axis_update(rd: R, old_off: R, new_off: R) -> R {
        // `rd` is the largest offset across all axes. When this axis was not the largest,
        // the other axes still account for `rd` and the update is exact. When it was (or
        // tied for) the largest, the other axes' maximum is not recoverable from `rd`
        // alone, so fall back to this axis alone: an under-estimate that keeps the sibling
        // in the search rather than pruning it incorrectly.
        if R::cmp(old_off, rd) == core::cmp::Ordering::Less {
            R::max(rd, new_off)
        } else {
            new_off
        }
    }
}

impl<A, R> DistanceMetricAvx512<A> for Chebyshev<R>
where
    A: Copy,
    R: Axis<Coord = R> + WideningCastFrom<A>,
{
    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    type Avx512F64Ops = avx512::ChebyshevAvx512F64LeafOps;

    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    type Avx512F32Ops = avx512::ChebyshevAvx512F32LeafOps;
}

impl<A, R> DistanceMetricAvx2<A> for Chebyshev<R>
where
    A: Copy,
    R: Axis<Coord = R> + WideningCastFrom<A>,
{
    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    type Avx2F64Ops = avx2::ChebyshevAvx2F64LeafOps;

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    type Avx2F32Ops = avx2::ChebyshevAvx2F32LeafOps;
}

impl<A, R> DistanceMetricNeon<A> for Chebyshev<R>
where
    A: Copy,
    R: Axis<Coord = R> + WideningCastFrom<A>,
{
    #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
    type NeonF64Ops = neon::ChebyshevNeonF64LeafOps;

    #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
    type NeonF32Ops = neon::ChebyshevNeonF32LeafOps;
}

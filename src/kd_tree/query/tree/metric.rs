use crate::dist::{Chebyshev, DistanceMetricScalar, Manhattan, Minkowski, SquaredEuclidean};
use crate::Axis;

mod sealed {
    pub trait Float {}
    impl Float for f32 {}
    impl Float for f64 {}
    pub trait Metric<A> {}
}

/// Floating-point types supported by tree joins.
#[doc(hidden)]
pub trait JoinFloat: sealed::Float + Axis<Coord = Self> + Send + Sync {
    const ZERO: Self;
    const INFINITY: Self;
    const NEG_INFINITY: Self;
    fn is_nan(self) -> bool;
    fn is_finite(self) -> bool;
}
macro_rules! float {
    ($t:ty) => {
        impl JoinFloat for $t {
            const ZERO: Self = 0.0;
            const INFINITY: Self = Self::INFINITY;
            const NEG_INFINITY: Self = Self::NEG_INFINITY;
            fn is_nan(self) -> bool {
                self.is_nan()
            }
            fn is_finite(self) -> bool {
                self.is_finite()
            }
        }
    };
}
float!(f32);
float!(f64);

/// Sealed metric capability for exact immutable tree joins.
/// Supports the built-in float metrics, with identity or f32-to-f64 accumulation.
#[doc(hidden)]
pub trait JoinMetric<A: JoinFloat>:
    sealed::Metric<A> + DistanceMetricScalar<A, Output: JoinFloat>
{
    const VALID: () = ();
    const SQUARED_EUCLIDEAN: bool = false;
}

macro_rules! metrics {
    ($a:ty, $o:ty) => {
        impl sealed::Metric<$a> for SquaredEuclidean<$o> {}
        impl JoinMetric<$a> for SquaredEuclidean<$o> {
            const SQUARED_EUCLIDEAN: bool = true;
        }
        impl sealed::Metric<$a> for Manhattan<$o> {}
        impl JoinMetric<$a> for Manhattan<$o> {}
        impl sealed::Metric<$a> for Chebyshev<$o> {}
        impl JoinMetric<$a> for Chebyshev<$o> {}
        impl<const P: u32> sealed::Metric<$a> for Minkowski<P, $o> {}
        impl<const P: u32> JoinMetric<$a> for Minkowski<P, $o> {
            const VALID: () = assert!(
                P >= 3 && P <= i32::MAX as u32,
                "tree joins require Minkowski power in 3..=i32::MAX"
            );
        }
    };
}
metrics!(f32, f32);
metrics!(f32, f64);
metrics!(f64, f64);

#[derive(Clone, Copy)]
pub(super) struct Bounds<O, const K: usize> {
    pub min: [[O; K]; 2],
    pub max: [[O; K]; 2],
}

impl<O: JoinFloat, const K: usize> Bounds<O, K> {
    pub fn new() -> Self {
        Self {
            min: [[O::NEG_INFINITY; K]; 2],
            max: [[O::INFINITY; K]; 2],
        }
    }

    pub fn lower<A: JoinFloat, D: JoinMetric<A, Output = O>>(&self) -> O {
        let mut acc = O::ZERO;
        for d in 0..K {
            let c = if self.max[0][d] < self.min[1][d] {
                D::dist1(self.max[0][d], self.min[1][d])
            } else if self.max[1][d] < self.min[0][d] {
                D::dist1(self.max[1][d], self.min[0][d])
            } else {
                O::ZERO
            };
            D::combine_component(&mut acc, c);
        }
        acc
    }

    pub fn upper<A: JoinFloat, D: JoinMetric<A, Output = O>>(&self) -> O {
        let mut acc = O::ZERO;
        for d in 0..K {
            if !(self.min[0][d].is_finite()
                && self.min[1][d].is_finite()
                && self.max[0][d].is_finite()
                && self.max[1][d].is_finite())
            {
                return O::INFINITY;
            }
            let a = D::dist1(self.min[0][d], self.max[1][d]);
            let b = D::dist1(self.max[0][d], self.min[1][d]);
            D::combine_component(&mut acc, if a > b { a } else { b });
        }
        acc
    }
}

#[inline(always)]
pub(super) fn accepts<O: JoinFloat, const EX: bool>(distance: O, radius: O) -> bool {
    if EX {
        distance < radius
    } else {
        distance <= radius
    }
}

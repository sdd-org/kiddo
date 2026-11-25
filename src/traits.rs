//! Definitions and implementations for some traits that are common between the [`float`](crate::float), [`immutable`](crate::immutable) and [`fixed`](crate::fixed)  modules
use az::Cast;
use divrem::DivCeil;
use num_traits::{One, PrimInt, Unsigned, Zero};
use std::fmt::Debug;

/// Content trait.
///
/// Must be implemented by any type that you want to use to represent the content
/// stored in a KdTree. Generally this will be `usize`, `u32`, or for trees with less
/// than 65535 points, you could use a `u16`. All these types implement `Content` with no
/// extra changes. Start off with a `usize` as that's easiest
/// since you won't need to cast to / from usize when using query results to index into
/// a Vec, and try switching tqo a smaller type and benchmarking to see if you get better
/// performance.
pub trait Content:
    Zero + One + PartialEq + Default + Clone + Copy + Ord + Debug + std::ops::SubAssign + Sync + Send
{
}
impl<
        T: Zero
            + One
            + PartialEq
            + Default
            + Clone
            + Copy
            + Ord
            + Debug
            + std::ops::SubAssign
            + Sync
            + Send,
    > Content for T
{
}

/// Implemented on u16 and u32 so that they can be used internally to index the
/// `Vec`s of Stem and Leaf nodes.
///
/// Allows `u32` or `u16` to be used as the 5th generic parameter of `float::KdTree`
/// and `fixed::KdTree`. If you will be storing fewer than `BUCKET_SIZE` * ~32k items
/// in the tree, selecting `u16` will slightly reduce the size of the Stem Nodes,
/// ensuring that more of them can be kept in the CPU cache, which may improve
/// performance (this may be offset on some architectures if it results in a
/// misalignment penalty).
pub trait Index: PrimInt + Unsigned + Zero + Cast<usize> + Sync {
    #[doc(hidden)]
    type T: Cast<usize>;
    #[doc(hidden)]
    fn max() -> Self;
    #[doc(hidden)]
    fn min() -> Self;
    #[doc(hidden)]
    fn leaf_offset() -> Self;
    #[doc(hidden)]
    fn ilog2(self) -> Self;
    #[doc(hidden)]
    fn div_ceil(self, b: Self::T) -> Self;
    #[doc(hidden)]
    fn capacity_with_bucket_size(bucket_size: usize) -> usize;
}

impl Index for u32 {
    type T = u32;
    fn max() -> u32 {
        u32::MAX
    }
    fn min() -> u32 {
        0u32
    }
    fn leaf_offset() -> u32 {
        u32::MAX.overflowing_shr(1).0
    }
    fn ilog2(self) -> u32 {
        u32::ilog2(self)
    }
    fn div_ceil(self, b: u32) -> u32 {
        DivCeil::div_ceil(self, b)
    }
    fn capacity_with_bucket_size(bucket_size: usize) -> usize {
        ((u32::MAX - u32::MAX.overflowing_shr(1).0) as usize).saturating_mul(bucket_size)
    }
}

impl Index for u16 {
    type T = u16;
    fn max() -> u16 {
        u16::MAX
    }
    fn min() -> u16 {
        0u16
    }
    fn leaf_offset() -> u16 {
        u16::MAX.overflowing_shr(1).0
    }
    fn ilog2(self) -> u16 {
        u16::ilog2(self) as u16
    }
    fn div_ceil(self, b: u16) -> u16 {
        DivCeil::div_ceil(self, b)
    }
    fn capacity_with_bucket_size(bucket_size: usize) -> usize {
        ((u16::MAX - u16::MAX.overflowing_shr(1).0) as usize).saturating_mul(bucket_size)
    }
}

pub(crate) fn is_stem_index<IDX: Index<T = IDX>>(x: IDX) -> bool {
    x < <IDX as Index>::leaf_offset()
}

/// Defines how distances are measured and compared for k-d tree queries.
///
/// Implement this trait to use custom distance metrics with [`kiddo:KdTree`](crate::KdTree).
///
/// # Distance Metrics in k-d Trees
///
/// **Distance aggregation**: How to combine per-dimension distances into a total distance
///  - Sum-based: `dist(p,q) = Σ |p[i] - q[i]|` (Manhattan, SquaredEuclidean)
///  - Max-based: `dist(p,q) = max_i |p[i] - q[i]|` (Chebyshev/L∞)
///
/// # Required Methods
///
/// - [`dist()`](Self::dist): Compute total distance between two points
/// - [`dist1()`](Self::dist1): Compute per-dimension distance component
/// - [`accumulate()`](Self::accumulate): Aggregate distance components (add or max)
///
pub trait DistanceMetric<A, const K: usize> {
    /// Returns the distance between two K-d points, as measured by this metric.
    fn dist(a: &[A; K], b: &[A; K]) -> A;

    /// Returns the distance between two points along a single dimension.
    ///
    /// Used internally by NN query implementations to extend the minimum
    /// acceptable distance for a node when recursing back up the tree.
    fn dist1(a: A, b: A) -> A;

    /// Aggregates a distance contribution into a running total.
    ///
    /// This defines how per-dimension distances combine into a total distance.
    /// Choose based on your distance metric:
    ///
    /// - **Sum-based (L1, L2)**: Use `rd + delta` or `rd.saturating_add(delta)` for fixed-point types
    /// - **Max-based (L∞/Chebyshev)**: Use `rd.max(delta)`
    ///
    /// The implementation should match the mathematical definition of your metric:
    /// - Manhattan: `dist(p,q) = Σ |p[i] - q[i]|` -> accumulate by adding
    /// - SquaredEuclidean: `dist(p,q) = Σ (p[i] - q[i])²` -> accumulate by adding
    /// - Chebyshev: `dist(p,q) = max_i |p[i] - q[i]|` -> accumulate by taking max
    /// - Generalised Minkowski (L_p): `dist(p,q) = (Σ |p[i] - q[i]|^p)^(1/p)`.
    ///   For k-d tree pruning, use the sum of powers: accumulate by adding.
    ///   Only the limit p → ∞ (Chebyshev) uses `max`.
    ///
    /// The default implementation uses regular addition (`rd + delta`), which works for
    /// both integer and floating-point types. For fixed-point types where overflow is a
    /// concern, override this with `rd.saturating_add(delta)`.
    fn accumulate(rd: A, delta: A) -> A
    where
        A: std::ops::Add<Output = A>,
    {
        rd + delta
    }
}

#[cfg(test)]
mod tests {

    use super::DistanceMetric;
    use crate::traits::Index;
    use rstest::rstest;

    #[test]
    fn test_u16() {
        assert_eq!(<u16 as Index>::max(), u16::MAX);
        assert_eq!(<u16 as Index>::min(), 0u16);
        assert_eq!(<u16 as Index>::leaf_offset(), 32_767u16);
        assert_eq!(256u16.ilog2(), 8u32);
        assert_eq!(u16::capacity_with_bucket_size(32), 1_048_576);
    }

    #[test]
    fn test_u32() {
        assert_eq!(<u32 as Index>::max(), u32::MAX);
        assert_eq!(<u32 as Index>::min(), 0u32);
        assert_eq!(<u32 as Index>::leaf_offset(), 2_147_483_647);
        assert_eq!(256u32.ilog2(), 8u32);

        #[cfg(target_pointer_width = "64")]
        assert_eq!(u32::capacity_with_bucket_size(32), 68_719_476_736);

        #[cfg(target_pointer_width = "32")]
        assert_eq!(u32::capacity_with_bucket_size(32), u32::MAX);
    }
    #[test]
    fn test_u32_simulate_32bit_target_pointer() {
        // TODO: replace this with wasm-bindgen-tests at some point
        let bucket_size: u32 = 32;
        let capacity_with_bucket_size =
            (u32::MAX - u32::MAX.overflowing_shr(1).0).saturating_mul(bucket_size);
        assert_eq!(capacity_with_bucket_size, u32::MAX);
    }

    struct TestMetricU32;
    struct TestMetricI64;

    impl<const K: usize> DistanceMetric<u32, K> for TestMetricU32 {
        fn dist(_a: &[u32; K], _b: &[u32; K]) -> u32 {
            0
        }
        fn dist1(a: u32, _b: u32) -> u32 {
            a
        }
    }

    impl<const K: usize> DistanceMetric<i64, K> for TestMetricI64 {
        fn dist(_a: &[i64; K], _b: &[i64; K]) -> i64 {
            0
        }
        fn dist1(a: i64, _b: i64) -> i64 {
            a
        }
    }

    #[rstest]
    #[case(5u32, 3u32, 8u32)]
    #[case(10u32, 20u32, 30u32)]
    fn test_default_accumulate_u32(#[case] rd: u32, #[case] delta: u32, #[case] expected: u32) {
        assert_eq!(
            <TestMetricU32 as DistanceMetric<u32, 1>>::accumulate(rd, delta),
            expected
        );
    }

    #[rstest]
    #[case(10i64, 20i64, 30i64)]
    #[case(100i64, 200i64, 300i64)]
    fn test_default_accumulate_i64(#[case] rd: i64, #[case] delta: i64, #[case] expected: i64) {
        assert_eq!(
            <TestMetricI64 as DistanceMetric<i64, 1>>::accumulate(rd, delta),
            expected
        );
    }
}

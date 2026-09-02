use std::num::NonZeroUsize;

use kiddo::dist::{
    DistanceMetric, DistanceMetricAvx2, DistanceMetricAvx512, DistanceMetricNeon,
    DistanceMetricScalar, DistanceMetricWasmSimd, QueryMetric,
};
use kiddo::leaf_strategies::VecOfArrays;
use kiddo::{Axis, Eytzinger, KdTree};

struct AbsDistance;

impl DistanceMetricScalar<f64> for AbsDistance {
    type Output = f64;

    fn widen_coord(a: f64) -> Self::Output {
        a
    }

    fn dist1(a: Self::Output, b: Self::Output) -> Self::Output {
        (a - b).abs()
    }
}

impl DistanceMetricAvx512<f64> for AbsDistance {
    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    type Avx512F64Ops = kiddo::traits::dist::UnsupportedAvx512F64LeafOps;

    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    type Avx512F32Ops = kiddo::traits::dist::UnsupportedAvx512F32LeafOps;
}

impl DistanceMetricAvx2<f64> for AbsDistance {
    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    type Avx2F64Ops = kiddo::traits::dist::UnsupportedAvx2F64LeafOps;

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    type Avx2F32Ops = kiddo::traits::dist::UnsupportedAvx2F32LeafOps;
}

impl DistanceMetricNeon<f64> for AbsDistance {
    #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
    type NeonF64Ops = kiddo::traits::dist::UnsupportedNeonF64LeafOps;

    #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
    type NeonF32Ops = kiddo::traits::dist::UnsupportedNeonF32LeafOps;
}

impl DistanceMetricWasmSimd<f64> for AbsDistance {
    #[cfg(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128"))]
    type WasmSimdF64Ops = kiddo::traits::dist::UnsupportedWasmSimdF64LeafOps;

    #[cfg(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128"))]
    type WasmSimdF32Ops = kiddo::traits::dist::UnsupportedWasmSimdF32LeafOps;
}

type TestTree = KdTree<f64, u32, Eytzinger, VecOfArrays<f64, u32, 2, 32>, 2, 32>;

fn assert_distance_metric<M: DistanceMetric<f64>>() {}

fn assert_query_metric<M: QueryMetric<f64>>() {}

fn query_items<A, M, const K: usize>(
    tree: &KdTree<A, u32, Eytzinger, VecOfArrays<A, u32, K, 32>, K, 32>,
    point: &[A; K],
    n: NonZeroUsize,
) -> Vec<u32>
where
    A: Axis<Coord = A> + Copy + 'static,
    M: QueryMetric<A>,
{
    tree.query(point)
        .nearest_n::<M>(n)
        .execute()
        .into_iter()
        .map(|result| result.item)
        .collect()
}

#[test]
fn external_metric_satisfies_query_metric_and_generic_execute_helper() {
    assert_distance_metric::<AbsDistance>();
    assert_query_metric::<AbsDistance>();

    let points = [
        (10u32, [0.0f64, 0.0]),
        (20u32, [1.0, 1.0]),
        (30u32, [2.0, 2.0]),
    ];
    let tree: TestTree = KdTree::new_from_entries(&points).unwrap();

    let items =
        query_items::<f64, AbsDistance, 2>(&tree, &[0.1, 0.1], NonZeroUsize::new(2).unwrap());

    assert_eq!(items, vec![10, 20]);
}

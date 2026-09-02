use kiddo::dist::{
    DistanceMetric, DistanceMetricAvx2, DistanceMetricAvx512, DistanceMetricNeon,
    DistanceMetricScalar, DistanceMetricWasmSimd,
};
use kiddo::leaf_strategies::VecOfArenas;
use kiddo::stem_strategies::{DonnellySimdFull, Eytzinger};
use kiddo::KdTree;

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

type GeneralTree = KdTree<f64, u32, Eytzinger, VecOfArenas<f64, u32, 2, 32>, 2, 32>;
type BlockTree = KdTree<f64, u32, DonnellySimdFull<3>, VecOfArenas<f64, u32, 2, 32>, 2, 32>;

fn assert_distance_metric<M: DistanceMetric<f64>>() {}

#[test]
fn external_metric_works_for_general_and_block_strategies() {
    assert_distance_metric::<AbsDistance>();

    let points = [[0.0, 0.0], [1.0, 1.0], [2.0, 0.5], [4.0, 4.0]];
    let query = [1.9, 0.6];

    let general_tree: GeneralTree = KdTree::new_from_slice(&points).unwrap();
    let block_tree: BlockTree = KdTree::new_from_slice(&points).unwrap();

    let general_nearest = general_tree
        .query(&query)
        .nearest_one::<AbsDistance>()
        .execute();
    let block_nearest = block_tree
        .query(&query)
        .nearest_one::<AbsDistance>()
        .execute();

    assert_eq!(general_nearest.item, 2);
    assert_eq!(block_nearest.item, 2);
    assert!((general_nearest.distance - 0.2).abs() < 1e-12);
    assert!((block_nearest.distance - 0.2).abs() < 1e-12);
}

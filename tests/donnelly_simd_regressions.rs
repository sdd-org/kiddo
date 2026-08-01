#[cfg(feature = "simd")]
use assert_float_eq::assert_float_relative_eq;
#[cfg(feature = "simd")]
use kiddo::dist::DistanceMetricScalar;
#[cfg(feature = "simd")]
use kiddo::kd_tree::KdTree;
#[cfg(feature = "simd")]
use kiddo::leaf_strategy::FlatVec;
#[cfg(feature = "simd")]
use kiddo::stem_strategy::{
    Donnelly, DonnellyCyclicSimdDescent, DonnellyCyclicSimdFull, DonnellySimdFull,
};
#[cfg(feature = "simd")]
use kiddo::BestQueryResultItem;
#[cfg(feature = "simd")]
use kiddo::{Chebyshev, Manhattan, Minkowski, SquaredEuclidean};
#[cfg(feature = "simd")]
use rand::rngs::StdRng;
#[cfg(feature = "simd")]
use rand::{RngExt, SeedableRng};
#[cfg(feature = "simd")]
use std::cmp::Ordering;
#[cfg(feature = "simd")]
use std::num::NonZeroUsize;

#[cfg(feature = "simd")]
const CONTENT_SEED: u64 = 1_260_253_197;
#[cfg(feature = "simd")]
const QUERY_SEED: u64 = 13_787_848_794_416_797_126;
#[cfg(feature = "simd")]
const POINT_COUNT: usize = 1_022;
#[cfg(feature = "simd")]
const REL_EPS_F32: f32 = 1.0e-6;
#[cfg(feature = "simd")]
const REL_EPS_F64: f64 = 1.0e-12;

#[test]
#[cfg(feature = "simd")]
fn cyclic_simd_handles_f32_k3_with_root_padding() {
    const K: usize = 3;
    const B: usize = 8;
    let points: Vec<[f32; K]> = (0..33)
        .map(|i| {
            let x = i as f32 / 37.0;
            [x, (x * 1.7).fract(), (x * 2.9).fract()]
        })
        .collect();
    let tree: KdTree<f32, usize, DonnellyCyclicSimdDescent<4>, FlatVec<f32, usize, K, B>, K, B> =
        KdTree::new_from_slice(&points).unwrap();

    assert_eq!(tree.max_stem_level(), 3);
    for (expected_item, point) in points.iter().enumerate() {
        let result = tree
            .query(point)
            .nearest_one::<SquaredEuclidean<f32>>()
            .execute();
        assert_eq!(result.distance, 0.0);
        assert_eq!(result.item, expected_item);
    }

    let full_tree: KdTree<f32, usize, DonnellyCyclicSimdFull<4>, FlatVec<f32, usize, K, B>, K, B> =
        KdTree::new_from_slice(&points).unwrap();
    for (expected_item, point) in points.iter().enumerate() {
        let result = full_tree
            .query(point)
            .nearest_one::<SquaredEuclidean<f32>>()
            .execute();
        assert_eq!(result.distance, 0.0);
        assert_eq!(result.item, expected_item);
    }
}

#[test]
#[cfg(feature = "simd")]
fn cyclic_simd_handles_f64_k4_with_two_root_padding_levels() {
    const K: usize = 4;
    const B: usize = 8;
    let points: Vec<[f64; K]> = (0..65)
        .map(|i| {
            let x = i as f64 / 71.0;
            [x, (x * 1.7).fract(), (x * 2.9).fract(), (x * 4.3).fract()]
        })
        .collect();
    let tree: KdTree<f64, usize, DonnellyCyclicSimdDescent<3>, FlatVec<f64, usize, K, B>, K, B> =
        KdTree::new_from_slice(&points).unwrap();

    assert_eq!(tree.max_stem_level(), 5);
    for (expected_item, point) in points.iter().enumerate() {
        let result = tree
            .query(point)
            .nearest_one::<SquaredEuclidean<f64>>()
            .execute();
        assert_eq!(result.distance, 0.0);
        assert_eq!(result.item, expected_item);
    }

    let full_tree: KdTree<f64, usize, DonnellyCyclicSimdFull<3>, FlatVec<f64, usize, K, B>, K, B> =
        KdTree::new_from_slice(&points).unwrap();
    for (expected_item, point) in points.iter().enumerate() {
        let result = full_tree
            .query(point)
            .nearest_one::<SquaredEuclidean<f64>>()
            .execute();
        assert_eq!(result.distance, 0.0);
        assert_eq!(result.item, expected_item);
    }
}

#[test]
#[cfg(feature = "simd")]
fn cyclic_simd_full_native_f64_k3_metrics_match_linear_search() {
    const K: usize = 3;
    const B: usize = 8;
    let mut rng = StdRng::seed_from_u64(CONTENT_SEED);
    let points: Vec<[f64; K]> = (0..777)
        .map(|_| std::array::from_fn(|_| rng.random_range(-1.0..1.0)))
        .collect();
    let tree: KdTree<f64, usize, DonnellyCyclicSimdFull<3>, FlatVec<f64, usize, K, B>, K, B> =
        KdTree::new_from_slice(&points).unwrap();
    assert_eq!((tree.max_stem_level() + 1) % 3, 0);

    macro_rules! assert_metric {
        ($metric:ty) => {{
            let mut query_rng = StdRng::seed_from_u64(QUERY_SEED);
            for _ in 0..256 {
                let query = std::array::from_fn(|_| query_rng.random_range(-1.0..1.0));
                let expected = points
                    .iter()
                    .enumerate()
                    .map(|(item, point)| {
                        (
                            <$metric as DistanceMetricScalar<f64>>::dist_raw(&query, point),
                            item,
                        )
                    })
                    .min_by(|lhs, rhs| lhs.0.partial_cmp(&rhs.0).unwrap())
                    .unwrap();
                let actual = tree.query(&query).nearest_one::<$metric>().execute();
                assert_float_relative_eq!(actual.distance, expected.0, REL_EPS_F64);
                assert_eq!(actual.item, expected.1);
            }
        }};
    }

    assert_metric!(SquaredEuclidean<f64>);
    assert_metric!(Manhattan<f64>);
    assert_metric!(Chebyshev<f64>);
    assert_metric!(Minkowski<3, f64>);
}

#[test]
#[cfg(feature = "simd")]
fn cyclic_simd_full_native_f32_k4_metrics_match_linear_search() {
    const K: usize = 4;
    const B: usize = 8;
    let mut rng = StdRng::seed_from_u64(CONTENT_SEED);
    let points: Vec<[f32; K]> = (0..777)
        .map(|_| std::array::from_fn(|_| rng.random_range(-1.0..1.0)))
        .collect();
    let tree: KdTree<f32, usize, DonnellyCyclicSimdFull<4>, FlatVec<f32, usize, K, B>, K, B> =
        KdTree::new_from_slice(&points).unwrap();
    assert_eq!((tree.max_stem_level() + 1) % 4, 0);

    macro_rules! assert_metric {
        ($metric:ty) => {{
            let mut query_rng = StdRng::seed_from_u64(QUERY_SEED);
            for _ in 0..256 {
                let query = std::array::from_fn(|_| query_rng.random_range(-1.0..1.0));
                let expected = points
                    .iter()
                    .enumerate()
                    .map(|(item, point)| {
                        (
                            <$metric as DistanceMetricScalar<f32>>::dist_raw(&query, point),
                            item,
                        )
                    })
                    .min_by(|lhs, rhs| lhs.0.partial_cmp(&rhs.0).unwrap())
                    .unwrap();
                let actual = tree.query(&query).nearest_one::<$metric>().execute();
                assert_float_relative_eq!(actual.distance, expected.0, REL_EPS_F32);
                assert_eq!(actual.item, expected.1);
            }
        }};
    }

    assert_metric!(SquaredEuclidean<f32>);
    assert_metric!(Manhattan<f32>);
    assert_metric!(Chebyshev<f32>);
    assert_metric!(Minkowski<3, f32>);
}

#[cfg(feature = "simd")]
fn build_points_f32_k2() -> Vec<[f32; 2]> {
    let mut rng = StdRng::seed_from_u64(CONTENT_SEED);
    (0..POINT_COUNT)
        .map(|_| [rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)])
        .collect()
}

#[cfg(feature = "simd")]
fn build_query_f32_k2() -> [f32; 2] {
    let mut rng = StdRng::seed_from_u64(QUERY_SEED);
    [rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)]
}

#[cfg(feature = "simd")]
fn linear_search(points: &[[f32; 2]], query: &[f32; 2]) -> (f32, usize) {
    let mut best_dist = f32::INFINITY;
    let mut best_idx = 0usize;
    for (idx, point) in points.iter().enumerate() {
        let dist =
            <kiddo::SquaredEuclidean<f32> as DistanceMetricScalar<f32>>::dist_raw(query, point);
        if dist < best_dist {
            best_dist = dist;
            best_idx = idx;
        }
    }
    (best_dist, best_idx)
}

#[cfg(feature = "simd")]
fn linear_search_f64(points: &[[f64; 2]], query: &[f64; 2]) -> (f64, usize) {
    let mut best_dist = f64::INFINITY;
    let mut best_idx = 0usize;
    for (idx, point) in points.iter().enumerate() {
        let dist =
            <kiddo::SquaredEuclidean<f64> as DistanceMetricScalar<f64>>::dist_raw(query, point);
        if dist < best_dist {
            best_dist = dist;
            best_idx = idx;
        }
    }
    (best_dist, best_idx)
}

#[cfg(feature = "simd")]
fn linear_best_n_within_f32(
    points: &[[f32; 2]],
    query: &[f32; 2],
    max_dist: f32,
    max_qty: usize,
) -> Vec<BestQueryResultItem<(), usize, f32>> {
    let mut best_items = Vec::with_capacity(max_qty);

    for (item, point) in points.iter().enumerate() {
        let distance =
            <kiddo::SquaredEuclidean<f32> as DistanceMetricScalar<f32>>::dist_raw(query, point);
        if distance <= max_dist {
            if best_items.len() < max_qty {
                best_items.push(BestQueryResultItem {
                    point: (),
                    distance,
                    item,
                });
            } else if item < best_items.last().unwrap().item {
                best_items.pop();
                best_items.push(BestQueryResultItem {
                    point: (),
                    distance,
                    item,
                });
            }
            best_items.sort_unstable();
        }
    }

    best_items
}

#[cfg(feature = "simd")]
fn linear_best_n_within_f64(
    points: &[[f64; 2]],
    query: &[f64; 2],
    max_dist: f64,
    max_qty: usize,
) -> Vec<BestQueryResultItem<(), usize, f64>> {
    let mut best_items = Vec::with_capacity(max_qty);

    for (item, point) in points.iter().enumerate() {
        let distance =
            <kiddo::SquaredEuclidean<f64> as DistanceMetricScalar<f64>>::dist_raw(query, point);
        if distance <= max_dist {
            if best_items.len() < max_qty {
                best_items.push(BestQueryResultItem {
                    point: (),
                    distance,
                    item,
                });
            } else if item < best_items.last().unwrap().item {
                best_items.pop();
                best_items.push(BestQueryResultItem {
                    point: (),
                    distance,
                    item,
                });
            }
            best_items.sort_unstable();
        }
    }

    best_items
}

#[cfg(feature = "simd")]
fn adversarial_grid_points_f32(size: usize) -> Vec<[f32; 2]> {
    (0..size)
        .map(|i| {
            let x = ((i % 7) as f32 - 3.0) / 4.0;
            let y = (((i / 7) % 7) as f32 - 3.0) / 4.0;
            [x, y]
        })
        .collect()
}

#[cfg(feature = "simd")]
fn stabilize_neighbours_f32(items: &mut [(f32, usize)]) {
    items.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });
}

#[cfg(feature = "simd")]
fn linear_within_f32(points: &[[f32; 2]], query: &[f32; 2], max_dist: f32) -> Vec<(f32, usize)> {
    let mut result: Vec<(f32, usize)> = points
        .iter()
        .enumerate()
        .filter_map(|(idx, point)| {
            let distance =
                <kiddo::SquaredEuclidean<f32> as DistanceMetricScalar<f32>>::dist_raw(query, point);
            (distance <= max_dist).then_some((distance, idx))
        })
        .collect();
    stabilize_neighbours_f32(&mut result);
    result
}

#[test]
#[cfg(feature = "simd")]
fn regression_donnelly_simd_block4_f32_nearest_one_matches_scalar_and_linear() {
    let points = build_points_f32_k2();
    let query = build_query_f32_k2();
    let expected = linear_search(&points, &query);

    let tree_scalar: KdTree<f32, usize, Donnelly<4>, FlatVec<f32, usize, 2, 16>, 2, 16> =
        KdTree::new_from_slice(&points).unwrap();
    let tree_simd: KdTree<f32, usize, DonnellySimdFull<4>, FlatVec<f32, usize, 2, 16>, 2, 16> =
        KdTree::new_from_slice(&points).unwrap();

    let scalar_result = tree_scalar
        .query(&query)
        .nearest_one::<SquaredEuclidean<f32>>()
        .execute();
    let simd_result = tree_simd
        .query(&query)
        .nearest_one::<SquaredEuclidean<f32>>()
        .execute();

    assert_float_relative_eq!(scalar_result.distance, expected.0, REL_EPS_F32);
    assert_eq!(scalar_result.item, expected.1);
    assert_float_relative_eq!(simd_result.distance, expected.0, REL_EPS_F32);
    assert_eq!(simd_result.item, expected.1);
}

#[test]
#[cfg(feature = "simd")]
fn regression_donnelly_simd_block4_f32_approx_self_lookup_hits_zero_distance() {
    let points = build_points_f32_k2();

    let tree_scalar: KdTree<f32, usize, Donnelly<4>, FlatVec<f32, usize, 2, 16>, 2, 16> =
        KdTree::new_from_slice(&points).unwrap();
    let tree_simd: KdTree<f32, usize, DonnellySimdFull<4>, FlatVec<f32, usize, 2, 16>, 2, 16> =
        KdTree::new_from_slice(&points).unwrap();

    for point in points.iter() {
        let scalar_result = tree_scalar
            .query(point)
            .nearest_one::<SquaredEuclidean<f32>>()
            .approx()
            .execute();
        let simd_result = tree_simd
            .query(point)
            .nearest_one::<SquaredEuclidean<f32>>()
            .approx()
            .execute();

        assert_eq!(scalar_result.distance, 0.0);
        assert_eq!(simd_result.distance, 0.0);
    }
}

#[test]
#[cfg(feature = "simd")]
fn regression_donnelly_simd_block4_f32_best_n_within_matches_scalar_and_linear() {
    let points = build_points_f32_k2();
    let query = build_query_f32_k2();
    let max_dist = 2.0f32;
    let max_qty = NonZeroUsize::new(16).unwrap();
    let expected = linear_best_n_within_f32(&points, &query, max_dist, max_qty.get());

    let tree_scalar: KdTree<f32, usize, Donnelly<4>, FlatVec<f32, usize, 2, 16>, 2, 16> =
        KdTree::new_from_slice(&points).unwrap();
    let tree_simd: KdTree<f32, usize, DonnellySimdFull<4>, FlatVec<f32, usize, 2, 16>, 2, 16> =
        KdTree::new_from_slice(&points).unwrap();

    let scalar_result = tree_scalar
        .query(&query)
        .best_n_within::<SquaredEuclidean<f32>>(max_dist, max_qty)
        .execute()
        .into_sorted_vec();
    let simd_result = tree_simd
        .query(&query)
        .best_n_within::<SquaredEuclidean<f32>>(max_dist, max_qty)
        .execute()
        .into_sorted_vec();

    assert_eq!(scalar_result.len(), expected.len());
    assert_eq!(simd_result.len(), expected.len());

    for (actual, expected) in scalar_result.iter().zip(expected.iter()) {
        assert_float_relative_eq!(actual.distance, expected.distance, REL_EPS_F32);
        assert_eq!(actual.item, expected.item);
    }

    for (actual, expected) in simd_result.iter().zip(expected.iter()) {
        assert_float_relative_eq!(actual.distance, expected.distance, REL_EPS_F32);
        assert_eq!(actual.item, expected.item);
    }
}

#[test]
#[cfg(feature = "simd")]
fn control_donnelly_simd_block3_f64_nearest_one_matches_scalar_and_linear() {
    let mut rng = StdRng::seed_from_u64(CONTENT_SEED);
    let points: Vec<[f64; 2]> = (0..POINT_COUNT)
        .map(|_| [rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)])
        .collect();

    let mut rng = StdRng::seed_from_u64(QUERY_SEED);
    let query = [rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)];

    let expected = linear_search_f64(&points, &query);

    let tree_scalar: KdTree<f64, usize, Donnelly<3>, FlatVec<f64, usize, 2, 16>, 2, 16> =
        KdTree::new_from_slice(&points).unwrap();
    let tree_simd: KdTree<f64, usize, DonnellySimdFull<3>, FlatVec<f64, usize, 2, 16>, 2, 16> =
        KdTree::new_from_slice(&points).unwrap();

    let scalar_result = tree_scalar
        .query(&query)
        .nearest_one::<SquaredEuclidean<f64>>()
        .execute();
    let simd_result = tree_simd
        .query(&query)
        .nearest_one::<SquaredEuclidean<f64>>()
        .execute();

    assert_float_relative_eq!(scalar_result.distance, expected.0, REL_EPS_F64);
    assert_eq!(scalar_result.item, expected.1);
    assert_float_relative_eq!(simd_result.distance, expected.0, REL_EPS_F64);
    assert_eq!(simd_result.item, expected.1);
}

#[test]
#[cfg(feature = "simd")]
fn control_donnelly_simd_block3_f64_best_n_within_matches_scalar_and_linear() {
    let mut rng = StdRng::seed_from_u64(CONTENT_SEED);
    let points: Vec<[f64; 2]> = (0..POINT_COUNT)
        .map(|_| [rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)])
        .collect();

    let mut rng = StdRng::seed_from_u64(QUERY_SEED);
    let query = [rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)];
    let max_dist = 2.0f64;
    let max_qty = NonZeroUsize::new(16).unwrap();
    let expected = linear_best_n_within_f64(&points, &query, max_dist, max_qty.get());

    let tree_scalar: KdTree<f64, usize, Donnelly<3>, FlatVec<f64, usize, 2, 16>, 2, 16> =
        KdTree::new_from_slice(&points).unwrap();
    let tree_simd: KdTree<f64, usize, DonnellySimdFull<3>, FlatVec<f64, usize, 2, 16>, 2, 16> =
        KdTree::new_from_slice(&points).unwrap();

    let scalar_result = tree_scalar
        .query(&query)
        .best_n_within::<SquaredEuclidean<f64>>(max_dist, max_qty)
        .execute()
        .into_sorted_vec();
    let simd_result = tree_simd
        .query(&query)
        .best_n_within::<SquaredEuclidean<f64>>(max_dist, max_qty)
        .execute()
        .into_sorted_vec();

    assert_eq!(scalar_result.len(), expected.len());
    assert_eq!(simd_result.len(), expected.len());

    for (actual, expected) in scalar_result.iter().zip(expected.iter()) {
        assert_float_relative_eq!(actual.distance, expected.distance, REL_EPS_F64);
        assert_eq!(actual.item, expected.item);
    }

    for (actual, expected) in simd_result.iter().zip(expected.iter()) {
        assert_float_relative_eq!(actual.distance, expected.distance, REL_EPS_F64);
        assert_eq!(actual.item, expected.item);
    }
}

#[test]
#[cfg(feature = "simd")]
fn regression_donnelly_simd_block4_f32_small_grid_within_variants_match_linear() {
    let points = adversarial_grid_points_f32(17);
    let query = [0.125f32, -0.75];
    let max_dist = 0.6f32;
    let expected = linear_within_f32(&points, &query, max_dist);
    let max_qty = NonZeroUsize::new(usize::MAX).unwrap();

    let tree_scalar: KdTree<f32, usize, Donnelly<4>, FlatVec<f32, usize, 2, 16>, 2, 16> =
        KdTree::new_from_slice(&points).unwrap();
    let tree_simd: KdTree<f32, usize, DonnellySimdFull<4>, FlatVec<f32, usize, 2, 16>, 2, 16> =
        KdTree::new_from_slice(&points).unwrap();

    let mut scalar_within_unsorted: Vec<(f32, usize)> = tree_scalar
        .query(&query)
        .within::<SquaredEuclidean<f32>>(max_dist)
        .unsorted()
        .execute()
        .into_iter()
        .map(|n| (n.distance, n.item))
        .collect();
    stabilize_neighbours_f32(&mut scalar_within_unsorted);

    let mut simd_within_unsorted: Vec<(f32, usize)> = tree_simd
        .query(&query)
        .within::<SquaredEuclidean<f32>>(max_dist)
        .unsorted()
        .execute()
        .into_iter()
        .map(|n| (n.distance, n.item))
        .collect();
    stabilize_neighbours_f32(&mut simd_within_unsorted);

    let mut scalar_within_sorted: Vec<(f32, usize)> = tree_scalar
        .query(&query)
        .within::<SquaredEuclidean<f32>>(max_dist)
        .execute()
        .into_iter()
        .map(|n| (n.distance, n.item))
        .collect();
    stabilize_neighbours_f32(&mut scalar_within_sorted);

    let mut simd_within_sorted: Vec<(f32, usize)> = tree_simd
        .query(&query)
        .within::<SquaredEuclidean<f32>>(max_dist)
        .execute()
        .into_iter()
        .map(|n| (n.distance, n.item))
        .collect();
    stabilize_neighbours_f32(&mut simd_within_sorted);

    let mut scalar_nearest_n_unsorted: Vec<(f32, usize)> = tree_scalar
        .query(&query)
        .nearest_n::<SquaredEuclidean<f32>>(max_qty)
        .within(max_dist)
        .unsorted()
        .execute()
        .into_iter()
        .map(|n| (n.distance, n.item))
        .collect();
    stabilize_neighbours_f32(&mut scalar_nearest_n_unsorted);

    let mut simd_nearest_n_unsorted: Vec<(f32, usize)> = tree_simd
        .query(&query)
        .nearest_n::<SquaredEuclidean<f32>>(max_qty)
        .within(max_dist)
        .unsorted()
        .execute()
        .into_iter()
        .map(|n| (n.distance, n.item))
        .collect();
    stabilize_neighbours_f32(&mut simd_nearest_n_unsorted);

    let mut scalar_nearest_n_sorted: Vec<(f32, usize)> = tree_scalar
        .query(&query)
        .nearest_n::<SquaredEuclidean<f32>>(max_qty)
        .within(max_dist)
        .execute()
        .into_iter()
        .map(|n| (n.distance, n.item))
        .collect();
    stabilize_neighbours_f32(&mut scalar_nearest_n_sorted);

    let mut simd_nearest_n_sorted: Vec<(f32, usize)> = tree_simd
        .query(&query)
        .nearest_n::<SquaredEuclidean<f32>>(max_qty)
        .within(max_dist)
        .execute()
        .into_iter()
        .map(|n| (n.distance, n.item))
        .collect();
    stabilize_neighbours_f32(&mut simd_nearest_n_sorted);

    assert_eq!(scalar_within_unsorted, expected);
    assert_eq!(scalar_within_sorted, expected);
    assert_eq!(scalar_nearest_n_unsorted, expected);
    assert_eq!(scalar_nearest_n_sorted, expected);

    let mut mismatches = Vec::new();
    if simd_within_unsorted != expected {
        mismatches.push(format!(
            "within_unsorted got={simd_within_unsorted:?} expected={expected:?}"
        ));
    }
    if simd_within_sorted != expected {
        mismatches.push(format!(
            "within got={simd_within_sorted:?} expected={expected:?}"
        ));
    }
    if simd_nearest_n_unsorted != expected {
        mismatches.push(format!(
            "nearest_n_within unsorted got={simd_nearest_n_unsorted:?} expected={expected:?}"
        ));
    }
    if simd_nearest_n_sorted != expected {
        mismatches.push(format!(
            "nearest_n_within sorted got={simd_nearest_n_sorted:?} expected={expected:?}"
        ));
    }

    assert!(
        mismatches.is_empty(),
        "SIMD mismatches:\n{}",
        mismatches.join("\n")
    );
}

#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![cfg_attr(coverage_nightly, coverage(off))]

//! Query-only comparison against current Rust k-d tree crates.
//!
//! All implementations receive the same seeded, uniformly distributed 3D
//! `f32` and `f64` points and queries. Tree construction happens outside
//! Criterion's timed regions. Each external library's native exact query API is
//! used, including its normal result allocation and sorting behaviour. Kiddo is
//! benchmarked separately by its focused suites.
//!
//! FNNTW does not expose a radius query. Nabo can cap a k-NN query by radius,
//! but cannot return an unbounded set of all points within that radius, so
//! neither library is included in the `within_radius` cases.

use criterion::measurement::WallTime;
use criterion::{
    criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion, Throughput,
};
use fnntw::Tree as FnntwTree;
use kd_tree::KdTree as ExternalKdTree;
use nabo::simple_point::SimplePoint;
use nabo::{KDTree as NaboTree, NotNan, Point as NaboPoint};
use neighbourhood::KdTree as NeighbourhoodTree;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;

const K: usize = 3;
const DEFAULT_QUERY_COUNT: usize = 1_000;
const DEFAULT_MIN_LOG2_POINTS: u32 = 16;
const DEFAULT_MAX_LOG2_POINTS: u32 = 25;
const DEFAULT_RADIUS: f64 = 0.05;
// Match the focused Kiddo suites so their separately collected results are comparable.
const POINT_SEED: u64 = 0x5eed_0000_0000_0001;
const QUERY_SEED: u64 = 0x5eed_0000_0000_0002;
const MAX_QTYS: [usize; 3] = [5, 20, 50];
const FNNTW_LEAF_SIZE: usize = 32;

type PointF32 = [f32; K];
type PointF64 = [f64; K];
type NaboPointF32 = SimplePoint<K>;

#[derive(Clone, Copy, Debug)]
struct NaboPointF64([NotNan<f64>; K]);

impl Default for NaboPointF64 {
    fn default() -> Self {
        Self([NotNan::new(0.0).unwrap(); K])
    }
}

impl NaboPoint<f64> for NaboPointF64 {
    const DIM: u32 = K as u32;

    fn set(&mut self, index: u32, value: NotNan<f64>) {
        self.0[index as usize] = value;
    }

    fn get(&self, index: u32) -> NotNan<f64> {
        self.0[index as usize]
    }
}

fn read_usize_env(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("{var} must be a positive integer"))
        })
        .unwrap_or(default)
}

fn read_u32_env(var: &str, default: u32) -> u32 {
    std::env::var(var)
        .ok()
        .map(|value| {
            value
                .parse::<u32>()
                .unwrap_or_else(|_| panic!("{var} must be a positive integer"))
        })
        .unwrap_or(default)
}

fn read_f64_env(var: &str, default: f64) -> f64 {
    std::env::var(var)
        .ok()
        .map(|value| {
            value
                .parse::<f64>()
                .unwrap_or_else(|_| panic!("{var} must be a finite positive number"))
        })
        .unwrap_or(default)
}

fn build_points_f32(point_count: usize) -> Vec<PointF32> {
    let mut rng = ChaCha8Rng::seed_from_u64(POINT_SEED);
    (0..point_count).map(|_| rng.random()).collect()
}

fn build_queries_f32(query_count: usize) -> Vec<PointF32> {
    let mut rng = ChaCha8Rng::seed_from_u64(QUERY_SEED);
    (0..query_count).map(|_| rng.random()).collect()
}

fn build_points_f64(point_count: usize) -> Vec<PointF64> {
    let mut rng = ChaCha8Rng::seed_from_u64(POINT_SEED);
    (0..point_count).map(|_| rng.random()).collect()
}

fn build_queries_f64(query_count: usize) -> Vec<PointF64> {
    let mut rng = ChaCha8Rng::seed_from_u64(QUERY_SEED);
    (0..query_count).map(|_| rng.random()).collect()
}

fn id(library: &str, query: &str, point_count: usize) -> BenchmarkId {
    BenchmarkId::new(format!("{library}_{query}"), point_count)
}

fn assert_distances_match(label: &str, mut actual: Vec<f32>, expected: &[f32]) {
    actual.sort_unstable_by(f32::total_cmp);
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label} returned the wrong number of neighbours"
    );
    for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
        let tolerance = 1.0e-5 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label} distance mismatch at result {index}: actual={actual} expected={expected}"
        );
    }
}

fn validate_implementations_f32(points: &[PointF32], query: &PointF32, radius: f32) {
    let max_qty = *MAX_QTYS.last().unwrap();
    let max_dist = radius * radius;
    let mut expected: Vec<_> = points
        .iter()
        .map(|point| {
            point
                .iter()
                .zip(query)
                .map(|(point, query)| (point - query) * (point - query))
                .sum::<f32>()
        })
        .collect();
    expected.sort_unstable_by(f32::total_cmp);
    expected.truncate(max_qty);
    let expected_within = points
        .iter()
        .filter(|point| {
            point
                .iter()
                .zip(query)
                .map(|(point, query)| (point - query) * (point - query))
                .sum::<f32>()
                <= max_dist
        })
        .count();

    let fnntw = FnntwTree::new(points, FNNTW_LEAF_SIZE).unwrap();
    let (fnntw_nearest, _) = fnntw.query_nearest_k(query, max_qty).unwrap();
    assert_distances_match("FNNTW", fnntw_nearest, &expected);

    let nabo_points = to_nabo_points_f32(points);
    let nabo_query = NaboPointF32::from_raw(query).unwrap();
    let nabo = NaboTree::new(&nabo_points);
    let nabo_nearest = nabo
        .knn(max_qty as u32, &nabo_query)
        .into_iter()
        .map(|result| result.dist2.into_inner())
        .collect();
    assert_distances_match("nabo", nabo_nearest, &expected);

    let kd_tree: ExternalKdTree<PointF32> = ExternalKdTree::build_by_ordered_float(points.to_vec());
    let kd_tree_nearest = kd_tree
        .nearests(query, max_qty)
        .into_iter()
        .map(|result| result.squared_distance)
        .collect();
    assert_distances_match("kd-tree", kd_tree_nearest, &expected);
    assert_eq!(
        kd_tree.within_radius(query, radius).len(),
        expected_within,
        "kd-tree returned the wrong within-radius count"
    );

    let neighbourhood = NeighbourhoodTree::new(points.to_vec());
    let neighbourhood_nearest = neighbourhood
        .knn_by_index(query, max_qty)
        .into_iter()
        .map(|result| result.0 * result.0)
        .collect();
    assert_distances_match("neighbourhood", neighbourhood_nearest, &expected);
    assert_eq!(
        neighbourhood.neighbourhood_by_index(query, radius).len(),
        expected_within,
        "neighbourhood returned the wrong within-radius count"
    );
}

fn assert_distances_match_f64(label: &str, mut actual: Vec<f64>, expected: &[f64]) {
    actual.sort_unstable_by(f64::total_cmp);
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label} returned the wrong number of neighbours"
    );
    for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
        let tolerance = 1.0e-12 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label} distance mismatch at result {index}: actual={actual} expected={expected}"
        );
    }
}

fn to_nabo_points_f64(points: &[PointF64]) -> Vec<NaboPointF64> {
    points
        .iter()
        .map(|point| NaboPointF64::from_raw(point).unwrap())
        .collect()
}

fn validate_implementations_f64(points: &[PointF64], query: &PointF64, radius: f64) {
    let max_qty = *MAX_QTYS.last().unwrap();
    let max_dist = radius * radius;
    let mut expected: Vec<_> = points
        .iter()
        .map(|point| {
            point
                .iter()
                .zip(query)
                .map(|(point, query)| (point - query) * (point - query))
                .sum::<f64>()
        })
        .collect();
    expected.sort_unstable_by(f64::total_cmp);
    expected.truncate(max_qty);
    let expected_within = points
        .iter()
        .filter(|point| {
            point
                .iter()
                .zip(query)
                .map(|(point, query)| (point - query) * (point - query))
                .sum::<f64>()
                <= max_dist
        })
        .count();

    let fnntw = FnntwTree::new(points, FNNTW_LEAF_SIZE).unwrap();
    let (fnntw_nearest, _) = fnntw.query_nearest_k(query, max_qty).unwrap();
    assert_distances_match_f64("FNNTW", fnntw_nearest, &expected);

    let nabo_points = to_nabo_points_f64(points);
    let nabo_query = NaboPointF64::from_raw(query).unwrap();
    let nabo = NaboTree::new(&nabo_points);
    let nabo_nearest = nabo
        .knn(max_qty as u32, &nabo_query)
        .into_iter()
        .map(|result| result.dist2.into_inner())
        .collect();
    assert_distances_match_f64("nabo", nabo_nearest, &expected);

    let kd_tree: ExternalKdTree<PointF64> = ExternalKdTree::build_by_ordered_float(points.to_vec());
    let kd_tree_nearest = kd_tree
        .nearests(query, max_qty)
        .into_iter()
        .map(|result| result.squared_distance)
        .collect();
    assert_distances_match_f64("kd-tree", kd_tree_nearest, &expected);
    assert_eq!(
        kd_tree.within_radius(query, radius).len(),
        expected_within,
        "kd-tree returned the wrong within-radius count"
    );

    let neighbourhood = NeighbourhoodTree::new(points.to_vec());
    let neighbourhood_nearest = neighbourhood
        .knn_by_index(query, max_qty)
        .into_iter()
        .map(|result| result.0 * result.0)
        .collect();
    assert_distances_match_f64("neighbourhood", neighbourhood_nearest, &expected);
    assert_eq!(
        neighbourhood.neighbourhood_by_index(query, radius).len(),
        expected_within,
        "neighbourhood returned the wrong within-radius count"
    );
}

fn bench_fnntw(
    group: &mut BenchmarkGroup<'_, WallTime>,
    point_count: usize,
    points: &[PointF32],
    queries: &[PointF32],
) {
    let tree = FnntwTree::new(points, FNNTW_LEAF_SIZE).unwrap();

    group.bench_function(id("fnntw", "nearest_one", point_count), |b| {
        b.iter(|| {
            let mut distance = 0.0f32;
            let mut item = 0u64;
            for query in queries {
                let result = tree.query_nearest(black_box(query)).unwrap();
                distance += result.0;
                item = item.wrapping_add(result.1);
            }
            black_box((distance, item))
        });
    });

    for max_qty in MAX_QTYS {
        group.bench_function(
            id("fnntw", &format!("nearest_n_k{max_qty}"), point_count),
            |b| {
                b.iter(|| {
                    let mut len = 0usize;
                    let mut distance = 0.0f32;
                    let mut item = 0u64;
                    for query in queries {
                        let (distances, items) =
                            tree.query_nearest_k(black_box(query), max_qty).unwrap();
                        len = len.wrapping_add(distances.len());
                        distance += distances.into_iter().sum::<f32>();
                        item = items
                            .into_iter()
                            .fold(item, |checksum, value| checksum.wrapping_add(value));
                    }
                    black_box((len, distance, item))
                });
            },
        );
    }
}

fn to_nabo_points_f32(points: &[PointF32]) -> Vec<NaboPointF32> {
    points
        .iter()
        .map(|point| NaboPointF32::from_raw(point).unwrap())
        .collect()
}

fn bench_nabo(
    group: &mut BenchmarkGroup<'_, WallTime>,
    point_count: usize,
    points: &[PointF32],
    queries: &[PointF32],
) {
    let points = to_nabo_points_f32(points);
    let queries = to_nabo_points_f32(queries);
    let tree = NaboTree::new(&points);

    group.bench_function(id("nabo", "nearest_one", point_count), |b| {
        b.iter(|| {
            let mut distance = 0.0f32;
            let mut item = 0u64;
            for query in &queries {
                let result = tree.knn(1, black_box(query));
                distance += result[0].dist2.into_inner();
                item = item.wrapping_add(result[0].index as u64);
            }
            black_box((distance, item))
        });
    });

    for max_qty in MAX_QTYS {
        group.bench_function(
            id("nabo", &format!("nearest_n_k{max_qty}"), point_count),
            |b| {
                b.iter(|| {
                    let mut len = 0usize;
                    let mut distance = 0.0f32;
                    let mut item = 0u64;
                    for query in &queries {
                        let results = tree.knn(max_qty as u32, black_box(query));
                        len = len.wrapping_add(results.len());
                        for result in results {
                            distance += result.dist2.into_inner();
                            item = item.wrapping_add(result.index as u64);
                        }
                    }
                    black_box((len, distance, item))
                });
            },
        );
    }
}

fn bench_kd_tree(
    group: &mut BenchmarkGroup<'_, WallTime>,
    point_count: usize,
    points: &[PointF32],
    queries: &[PointF32],
    radius: f32,
    within_radius_id: &str,
) {
    let tree: ExternalKdTree<PointF32> = ExternalKdTree::build_by_ordered_float(points.to_vec());

    group.bench_function(id("kd_tree", "nearest_one", point_count), |b| {
        b.iter(|| {
            let mut distance = 0.0f32;
            let mut point_sum = 0.0f32;
            for query in queries {
                let result = tree.nearest(black_box(query)).unwrap();
                distance += result.squared_distance;
                point_sum += result.item.iter().sum::<f32>();
            }
            black_box((distance, point_sum))
        });
    });

    for max_qty in MAX_QTYS {
        group.bench_function(
            id("kd_tree", &format!("nearest_n_k{max_qty}"), point_count),
            |b| {
                b.iter(|| {
                    let mut len = 0usize;
                    let mut distance = 0.0f32;
                    let mut point_sum = 0.0f32;
                    for query in queries {
                        let results = tree.nearests(black_box(query), max_qty);
                        len = len.wrapping_add(results.len());
                        for result in results {
                            distance += result.squared_distance;
                            point_sum += result.item.iter().sum::<f32>();
                        }
                    }
                    black_box((len, distance, point_sum))
                });
            },
        );
    }

    group.bench_function(id("kd_tree", within_radius_id, point_count), |b| {
        b.iter(|| {
            let mut len = 0usize;
            let mut point_sum = 0.0f32;
            for query in queries {
                let results = tree.within_radius(black_box(query), radius);
                len = len.wrapping_add(results.len());
                for point in results {
                    point_sum += point.iter().sum::<f32>();
                }
            }
            black_box((len, point_sum))
        });
    });
}

fn bench_neighbourhood(
    group: &mut BenchmarkGroup<'_, WallTime>,
    point_count: usize,
    points: &[PointF32],
    queries: &[PointF32],
    radius: f32,
    within_radius_id: &str,
) {
    let tree = NeighbourhoodTree::new(points.to_vec());

    group.bench_function(id("neighbourhood", "nearest_one", point_count), |b| {
        b.iter(|| {
            let mut distance = 0.0f32;
            let mut item = 0usize;
            for query in queries {
                let result = tree.knn_by_index(black_box(query), 1);
                distance += result[0].0;
                item = item.wrapping_add(result[0].1);
            }
            black_box((distance, item))
        });
    });

    for max_qty in MAX_QTYS {
        group.bench_function(
            id(
                "neighbourhood",
                &format!("nearest_n_k{max_qty}"),
                point_count,
            ),
            |b| {
                b.iter(|| {
                    let mut len = 0usize;
                    let mut distance = 0.0f32;
                    let mut item = 0usize;
                    for query in queries {
                        let results = tree.knn_by_index(black_box(query), max_qty);
                        len = len.wrapping_add(results.len());
                        for result in results {
                            distance += result.0;
                            item = item.wrapping_add(result.1);
                        }
                    }
                    black_box((len, distance, item))
                });
            },
        );
    }

    group.bench_function(id("neighbourhood", within_radius_id, point_count), |b| {
        b.iter(|| {
            let mut len = 0usize;
            let mut item = 0usize;
            for query in queries {
                let results = tree.neighbourhood_by_index(black_box(query), radius);
                len = len.wrapping_add(results.len());
                item = results
                    .into_iter()
                    .fold(item, |checksum, value| checksum.wrapping_add(value));
            }
            black_box((len, item))
        });
    });
}

fn bench_fnntw_f64(
    group: &mut BenchmarkGroup<'_, WallTime>,
    point_count: usize,
    points: &[PointF64],
    queries: &[PointF64],
) {
    let tree = FnntwTree::new(points, FNNTW_LEAF_SIZE).unwrap();

    group.bench_function(id("fnntw", "nearest_one", point_count), |b| {
        b.iter(|| {
            let mut distance = 0.0f64;
            let mut item = 0u64;
            for query in queries {
                let result = tree.query_nearest(black_box(query)).unwrap();
                distance += result.0;
                item = item.wrapping_add(result.1);
            }
            black_box((distance, item))
        });
    });

    for max_qty in MAX_QTYS {
        group.bench_function(
            id("fnntw", &format!("nearest_n_k{max_qty}"), point_count),
            |b| {
                b.iter(|| {
                    let mut len = 0usize;
                    let mut distance = 0.0f64;
                    let mut item = 0u64;
                    for query in queries {
                        let (distances, items) =
                            tree.query_nearest_k(black_box(query), max_qty).unwrap();
                        len = len.wrapping_add(distances.len());
                        distance += distances.into_iter().sum::<f64>();
                        item = items
                            .into_iter()
                            .fold(item, |checksum, value| checksum.wrapping_add(value));
                    }
                    black_box((len, distance, item))
                });
            },
        );
    }
}

fn bench_nabo_f64(
    group: &mut BenchmarkGroup<'_, WallTime>,
    point_count: usize,
    points: &[PointF64],
    queries: &[PointF64],
) {
    let points = to_nabo_points_f64(points);
    let queries = to_nabo_points_f64(queries);
    let tree = NaboTree::new(&points);

    group.bench_function(id("nabo", "nearest_one", point_count), |b| {
        b.iter(|| {
            let mut distance = 0.0f64;
            let mut item = 0u64;
            for query in &queries {
                let result = tree.knn(1, black_box(query));
                distance += result[0].dist2.into_inner();
                item = item.wrapping_add(result[0].index as u64);
            }
            black_box((distance, item))
        });
    });

    for max_qty in MAX_QTYS {
        group.bench_function(
            id("nabo", &format!("nearest_n_k{max_qty}"), point_count),
            |b| {
                b.iter(|| {
                    let mut len = 0usize;
                    let mut distance = 0.0f64;
                    let mut item = 0u64;
                    for query in &queries {
                        let results = tree.knn(max_qty as u32, black_box(query));
                        len = len.wrapping_add(results.len());
                        for result in results {
                            distance += result.dist2.into_inner();
                            item = item.wrapping_add(result.index as u64);
                        }
                    }
                    black_box((len, distance, item))
                });
            },
        );
    }
}

fn bench_kd_tree_f64(
    group: &mut BenchmarkGroup<'_, WallTime>,
    point_count: usize,
    points: &[PointF64],
    queries: &[PointF64],
    radius: f64,
    within_radius_id: &str,
) {
    let tree: ExternalKdTree<PointF64> = ExternalKdTree::build_by_ordered_float(points.to_vec());

    group.bench_function(id("kd_tree", "nearest_one", point_count), |b| {
        b.iter(|| {
            let mut distance = 0.0f64;
            let mut point_sum = 0.0f64;
            for query in queries {
                let result = tree.nearest(black_box(query)).unwrap();
                distance += result.squared_distance;
                point_sum += result.item.iter().sum::<f64>();
            }
            black_box((distance, point_sum))
        });
    });

    for max_qty in MAX_QTYS {
        group.bench_function(
            id("kd_tree", &format!("nearest_n_k{max_qty}"), point_count),
            |b| {
                b.iter(|| {
                    let mut len = 0usize;
                    let mut distance = 0.0f64;
                    let mut point_sum = 0.0f64;
                    for query in queries {
                        let results = tree.nearests(black_box(query), max_qty);
                        len = len.wrapping_add(results.len());
                        for result in results {
                            distance += result.squared_distance;
                            point_sum += result.item.iter().sum::<f64>();
                        }
                    }
                    black_box((len, distance, point_sum))
                });
            },
        );
    }

    group.bench_function(id("kd_tree", within_radius_id, point_count), |b| {
        b.iter(|| {
            let mut len = 0usize;
            let mut point_sum = 0.0f64;
            for query in queries {
                let results = tree.within_radius(black_box(query), radius);
                len = len.wrapping_add(results.len());
                for point in results {
                    point_sum += point.iter().sum::<f64>();
                }
            }
            black_box((len, point_sum))
        });
    });
}

fn bench_neighbourhood_f64(
    group: &mut BenchmarkGroup<'_, WallTime>,
    point_count: usize,
    points: &[PointF64],
    queries: &[PointF64],
    radius: f64,
    within_radius_id: &str,
) {
    let tree = NeighbourhoodTree::new(points.to_vec());

    group.bench_function(id("neighbourhood", "nearest_one", point_count), |b| {
        b.iter(|| {
            let mut distance = 0.0f64;
            let mut item = 0usize;
            for query in queries {
                let result = tree.knn_by_index(black_box(query), 1);
                distance += result[0].0;
                item = item.wrapping_add(result[0].1);
            }
            black_box((distance, item))
        });
    });

    for max_qty in MAX_QTYS {
        group.bench_function(
            id(
                "neighbourhood",
                &format!("nearest_n_k{max_qty}"),
                point_count,
            ),
            |b| {
                b.iter(|| {
                    let mut len = 0usize;
                    let mut distance = 0.0f64;
                    let mut item = 0usize;
                    for query in queries {
                        let results = tree.knn_by_index(black_box(query), max_qty);
                        len = len.wrapping_add(results.len());
                        for result in results {
                            distance += result.0;
                            item = item.wrapping_add(result.1);
                        }
                    }
                    black_box((len, distance, item))
                });
            },
        );
    }

    group.bench_function(id("neighbourhood", within_radius_id, point_count), |b| {
        b.iter(|| {
            let mut len = 0usize;
            let mut item = 0usize;
            for query in queries {
                let results = tree.neighbourhood_by_index(black_box(query), radius);
                len = len.wrapping_add(results.len());
                item = results
                    .into_iter()
                    .fold(item, |checksum, value| checksum.wrapping_add(value));
            }
            black_box((len, item))
        });
    });
}

fn external_kd_trees(c: &mut Criterion) {
    let query_count = read_usize_env("KIDDO_PROFILE_QUERIES", DEFAULT_QUERY_COUNT);
    let min_log2_points = read_u32_env("KIDDO_PROFILE_MIN_LOG2_POINTS", DEFAULT_MIN_LOG2_POINTS);
    let max_log2_points = read_u32_env("KIDDO_PROFILE_MAX_LOG2_POINTS", DEFAULT_MAX_LOG2_POINTS);
    let radius_f64 = read_f64_env("KIDDO_PROFILE_RADIUS", DEFAULT_RADIUS);
    let radius_f32 = radius_f64 as f32;
    let within_radius_id = format!("within_radius_r{radius_f64}");

    assert!(query_count > 0, "KIDDO_PROFILE_QUERIES must be positive");
    assert!(
        min_log2_points <= max_log2_points,
        "KIDDO_PROFILE_MIN_LOG2_POINTS must not exceed KIDDO_PROFILE_MAX_LOG2_POINTS"
    );
    assert!(
        max_log2_points <= 31,
        "KIDDO_PROFILE_MAX_LOG2_POINTS must fit u32 item indices"
    );
    assert!(
        radius_f64.is_finite() && radius_f64 > 0.0 && radius_f32.is_finite() && radius_f32 > 0.0,
        "KIDDO_PROFILE_RADIUS must be a finite positive number"
    );

    eprintln!(
        "benchmarking external k-d trees: dims={K} scalars=f32,f64 tree_sizes=2^{min_log2_points}..2^{max_log2_points} queries={query_count} nearest_n={MAX_QTYS:?} radius={radius_f64} point_seed={POINT_SEED} query_seed={QUERY_SEED}"
    );
    eprintln!(
        "within_radius coverage: kd-tree, neighbourhood (FNNTW has no radius query; nabo has only radius-capped k-NN)"
    );

    let queries = build_queries_f32(query_count);
    let mut group = c.benchmark_group("profile_external_kd_trees/f32");
    group.throughput(Throughput::Elements(query_count as u64));

    for log2_points in min_log2_points..=max_log2_points {
        let point_count = 1usize << log2_points;
        let points = build_points_f32(point_count);
        if log2_points == min_log2_points {
            validate_implementations_f32(&points, &queries[0], radius_f32);
        }

        bench_fnntw(&mut group, point_count, &points, &queries);
        bench_nabo(&mut group, point_count, &points, &queries);
        bench_kd_tree(
            &mut group,
            point_count,
            &points,
            &queries,
            radius_f32,
            &within_radius_id,
        );
        bench_neighbourhood(
            &mut group,
            point_count,
            &points,
            &queries,
            radius_f32,
            &within_radius_id,
        );
    }

    group.finish();

    let queries = build_queries_f64(query_count);
    let mut group = c.benchmark_group("profile_external_kd_trees/f64");
    group.throughput(Throughput::Elements(query_count as u64));

    for log2_points in min_log2_points..=max_log2_points {
        let point_count = 1usize << log2_points;
        let points = build_points_f64(point_count);
        if log2_points == min_log2_points {
            validate_implementations_f64(&points, &queries[0], radius_f64);
        }

        bench_fnntw_f64(&mut group, point_count, &points, &queries);
        bench_nabo_f64(&mut group, point_count, &points, &queries);
        bench_kd_tree_f64(
            &mut group,
            point_count,
            &points,
            &queries,
            radius_f64,
            &within_radius_id,
        );
        bench_neighbourhood_f64(
            &mut group,
            point_count,
            &points,
            &queries,
            radius_f64,
            &within_radius_id,
        );
    }

    group.finish();
}

criterion_group!(benches, external_kd_trees);
criterion_main!(benches);

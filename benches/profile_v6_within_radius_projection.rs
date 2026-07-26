#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![cfg_attr(coverage_nightly, coverage(off))]

//! Focused rerun of the 3D within-radius cases in the external comparison chart.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kiddo::dist::SquaredEuclidean;
use kiddo::kd_tree::KdTree;
use kiddo::leaf_strategy::FlatVec;
use kiddo::stem_strategy::Eytzinger;
use neighbourhood::KdTree as NeighbourhoodTree;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;

const K: usize = 3;
const B: usize = 32;
const DEFAULT_QUERY_COUNT: usize = 1_000;
const DEFAULT_MIN_LOG2_POINTS: u32 = 16;
const DEFAULT_MAX_LOG2_POINTS: u32 = 27;
const DEFAULT_RADIUS: f64 = 0.05;
const POINT_SEED: u64 = 0x5eed_0000_0000_0001;
const QUERY_SEED: u64 = 0x5eed_0000_0000_0002;

type F64Tree = KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, K, B>, K, B>;
type F32Tree = KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, K, B>, K, B>;

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
                .unwrap_or_else(|_| panic!("{var} must be a non-negative integer"))
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

fn build_points_f64(point_count: usize) -> Vec<[f64; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(POINT_SEED);
    (0..point_count).map(|_| rng.random()).collect()
}

fn build_queries_f64(query_count: usize) -> Vec<[f64; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(QUERY_SEED);
    (0..query_count).map(|_| rng.random()).collect()
}

fn build_points_f32(point_count: usize) -> Vec<[f32; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(POINT_SEED);
    (0..point_count).map(|_| rng.random()).collect()
}

fn build_queries_f32(query_count: usize) -> Vec<[f32; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(QUERY_SEED);
    (0..query_count).map(|_| rng.random()).collect()
}

fn run_kiddo_item_distance_f64(
    tree: &F64Tree,
    queries: &[[f64; K]],
    radius_squared: f64,
) -> (usize, u64, f64) {
    let mut len = 0usize;
    let mut item = 0u64;
    let mut distance = 0.0f64;
    for query in queries {
        let results = tree
            .query(black_box(query))
            .within::<SquaredEuclidean<f64>>(radius_squared)
            .unsorted()
            .execute();
        len = len.wrapping_add(results.len());
        for result in results {
            item = item.wrapping_add(result.item as u64);
            distance += result.distance;
        }
    }
    (len, item, distance)
}

fn run_kiddo_item_only_f64(
    tree: &F64Tree,
    queries: &[[f64; K]],
    radius_squared: f64,
) -> (usize, u64) {
    let mut len = 0usize;
    let mut item = 0u64;
    for query in queries {
        let results = tree
            .query(black_box(query))
            .within::<SquaredEuclidean<f64>>(radius_squared)
            .unsorted()
            .without_distances()
            .execute();
        len = len.wrapping_add(results.len());
        for result in results {
            item = item.wrapping_add(result.item as u64);
        }
    }
    (len, item)
}

fn run_kiddo_point_item_distance_f64(
    tree: &F64Tree,
    queries: &[[f64; K]],
    radius_squared: f64,
) -> (usize, u64, f64, f64) {
    let mut len = 0usize;
    let mut item = 0u64;
    let mut distance = 0.0f64;
    let mut point = 0.0f64;
    for query in queries {
        let results = tree
            .query(black_box(query))
            .within::<SquaredEuclidean<f64>>(radius_squared)
            .unsorted()
            .with_points()
            .execute();
        len = len.wrapping_add(results.len());
        for result in results {
            item = item.wrapping_add(result.item as u64);
            distance += result.distance;
            point += result.point.into_iter().sum::<f64>();
        }
    }
    (len, item, distance, point)
}

fn run_kiddo_point_item_f64(
    tree: &F64Tree,
    queries: &[[f64; K]],
    radius_squared: f64,
) -> (usize, u64, f64) {
    let mut len = 0usize;
    let mut item = 0u64;
    let mut point = 0.0f64;
    for query in queries {
        let results = tree
            .query(black_box(query))
            .within::<SquaredEuclidean<f64>>(radius_squared)
            .unsorted()
            .with_points()
            .without_distances()
            .execute();
        len = len.wrapping_add(results.len());
        for result in results {
            item = item.wrapping_add(result.item as u64);
            point += result.point.into_iter().sum::<f64>();
        }
    }
    (len, item, point)
}

fn run_neighbourhood_f64(
    tree: &NeighbourhoodTree<f64, K>,
    queries: &[[f64; K]],
    radius: f64,
) -> (usize, usize) {
    let mut len = 0usize;
    let mut item = 0usize;
    for query in queries {
        let results = tree.neighbourhood_by_index(black_box(query), radius);
        len = len.wrapping_add(results.len());
        item = results
            .into_iter()
            .fold(item, |checksum, value| checksum.wrapping_add(value));
    }
    (len, item)
}

fn run_kiddo_item_distance_f32(
    tree: &F32Tree,
    queries: &[[f32; K]],
    radius_squared: f32,
) -> (usize, u64, f32) {
    let mut len = 0usize;
    let mut item = 0u64;
    let mut distance = 0.0f32;
    for query in queries {
        let results = tree
            .query(black_box(query))
            .within::<SquaredEuclidean<f32>>(radius_squared)
            .unsorted()
            .execute();
        len = len.wrapping_add(results.len());
        for result in results {
            item = item.wrapping_add(result.item as u64);
            distance += result.distance;
        }
    }
    (len, item, distance)
}

fn run_kiddo_item_only_f32(
    tree: &F32Tree,
    queries: &[[f32; K]],
    radius_squared: f32,
) -> (usize, u64) {
    let mut len = 0usize;
    let mut item = 0u64;
    for query in queries {
        let results = tree
            .query(black_box(query))
            .within::<SquaredEuclidean<f32>>(radius_squared)
            .unsorted()
            .without_distances()
            .execute();
        len = len.wrapping_add(results.len());
        for result in results {
            item = item.wrapping_add(result.item as u64);
        }
    }
    (len, item)
}

fn run_kiddo_point_item_distance_f32(
    tree: &F32Tree,
    queries: &[[f32; K]],
    radius_squared: f32,
) -> (usize, u64, f32, f32) {
    let mut len = 0usize;
    let mut item = 0u64;
    let mut distance = 0.0f32;
    let mut point = 0.0f32;
    for query in queries {
        let results = tree
            .query(black_box(query))
            .within::<SquaredEuclidean<f32>>(radius_squared)
            .unsorted()
            .with_points()
            .execute();
        len = len.wrapping_add(results.len());
        for result in results {
            item = item.wrapping_add(result.item as u64);
            distance += result.distance;
            point += result.point.into_iter().sum::<f32>();
        }
    }
    (len, item, distance, point)
}

fn run_kiddo_point_item_f32(
    tree: &F32Tree,
    queries: &[[f32; K]],
    radius_squared: f32,
) -> (usize, u64, f32) {
    let mut len = 0usize;
    let mut item = 0u64;
    let mut point = 0.0f32;
    for query in queries {
        let results = tree
            .query(black_box(query))
            .within::<SquaredEuclidean<f32>>(radius_squared)
            .unsorted()
            .with_points()
            .without_distances()
            .execute();
        len = len.wrapping_add(results.len());
        for result in results {
            item = item.wrapping_add(result.item as u64);
            point += result.point.into_iter().sum::<f32>();
        }
    }
    (len, item, point)
}

fn run_neighbourhood_f32(
    tree: &NeighbourhoodTree<f32, K>,
    queries: &[[f32; K]],
    radius: f32,
) -> (usize, usize) {
    let mut len = 0usize;
    let mut item = 0usize;
    for query in queries {
        let results = tree.neighbourhood_by_index(black_box(query), radius);
        len = len.wrapping_add(results.len());
        item = results
            .into_iter()
            .fold(item, |checksum, value| checksum.wrapping_add(value));
    }
    (len, item)
}

fn within_radius_projection(c: &mut Criterion) {
    let query_count = read_usize_env("KIDDO_PROFILE_QUERIES", DEFAULT_QUERY_COUNT);
    let min_log2 = read_u32_env("KIDDO_PROFILE_MIN_LOG2_POINTS", DEFAULT_MIN_LOG2_POINTS);
    let max_log2 = read_u32_env("KIDDO_PROFILE_MAX_LOG2_POINTS", DEFAULT_MAX_LOG2_POINTS);
    let radius_f64 = read_f64_env("KIDDO_PROFILE_RADIUS", DEFAULT_RADIUS);
    let radius_f32 = radius_f64 as f32;

    assert!(query_count > 0, "KIDDO_PROFILE_QUERIES must be positive");
    assert!(
        min_log2 <= max_log2,
        "KIDDO_PROFILE_MIN_LOG2_POINTS must not exceed KIDDO_PROFILE_MAX_LOG2_POINTS"
    );
    assert!(
        max_log2 <= 31,
        "KIDDO_PROFILE_MAX_LOG2_POINTS must fit u32 item indices"
    );
    assert!(
        radius_f64.is_finite() && radius_f64 > 0.0 && radius_f32.is_finite() && radius_f32 > 0.0,
        "KIDDO_PROFILE_RADIUS must be a finite positive number"
    );

    eprintln!(
        "benchmarking focused 3D within-radius projections: scalars=f32,f64 tree_sizes=2^{min_log2}..2^{max_log2} queries={query_count} radius={radius_f64} point_seed={POINT_SEED} query_seed={QUERY_SEED}"
    );

    let queries = build_queries_f32(query_count);
    let mut group = c.benchmark_group("profile_v6_within_radius_projection/f32");
    group.throughput(Throughput::Elements(query_count as u64));
    for log2_points in min_log2..=max_log2 {
        let point_count = 1usize << log2_points;
        let points = build_points_f32(point_count);
        let tree: F32Tree = KdTree::new_from_slice(&points).unwrap();
        let radius_squared = radius_f32 * radius_f32;

        group.bench_function(BenchmarkId::new("kiddo_item_distance", point_count), |b| {
            b.iter(|| black_box(run_kiddo_item_distance_f32(&tree, &queries, radius_squared)));
        });
        group.bench_function(BenchmarkId::new("kiddo_item_only", point_count), |b| {
            b.iter(|| black_box(run_kiddo_item_only_f32(&tree, &queries, radius_squared)));
        });
        group.bench_function(
            BenchmarkId::new("kiddo_point_item_distance", point_count),
            |b| {
                b.iter(|| {
                    black_box(run_kiddo_point_item_distance_f32(
                        &tree,
                        &queries,
                        radius_squared,
                    ))
                });
            },
        );
        group.bench_function(BenchmarkId::new("kiddo_point_item", point_count), |b| {
            b.iter(|| black_box(run_kiddo_point_item_f32(&tree, &queries, radius_squared)));
        });
        drop(tree);

        let tree = NeighbourhoodTree::new(points);
        group.bench_function(
            BenchmarkId::new("neighbourhood_item_only", point_count),
            |b| {
                b.iter(|| black_box(run_neighbourhood_f32(&tree, &queries, radius_f32)));
            },
        );
    }
    group.finish();

    let queries = build_queries_f64(query_count);
    let mut group = c.benchmark_group("profile_v6_within_radius_projection/f64");
    group.throughput(Throughput::Elements(query_count as u64));
    for log2_points in min_log2..=max_log2 {
        let point_count = 1usize << log2_points;
        let points = build_points_f64(point_count);
        let tree: F64Tree = KdTree::new_from_slice(&points).unwrap();
        let radius_squared = radius_f64 * radius_f64;

        group.bench_function(BenchmarkId::new("kiddo_item_distance", point_count), |b| {
            b.iter(|| black_box(run_kiddo_item_distance_f64(&tree, &queries, radius_squared)));
        });
        group.bench_function(BenchmarkId::new("kiddo_item_only", point_count), |b| {
            b.iter(|| black_box(run_kiddo_item_only_f64(&tree, &queries, radius_squared)));
        });
        group.bench_function(
            BenchmarkId::new("kiddo_point_item_distance", point_count),
            |b| {
                b.iter(|| {
                    black_box(run_kiddo_point_item_distance_f64(
                        &tree,
                        &queries,
                        radius_squared,
                    ))
                });
            },
        );
        group.bench_function(BenchmarkId::new("kiddo_point_item", point_count), |b| {
            b.iter(|| black_box(run_kiddo_point_item_f64(&tree, &queries, radius_squared)));
        });
        drop(tree);

        let tree = NeighbourhoodTree::new(points);
        group.bench_function(
            BenchmarkId::new("neighbourhood_item_only", point_count),
            |b| {
                b.iter(|| black_box(run_neighbourhood_f64(&tree, &queries, radius_f64)));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, within_radius_projection);
criterion_main!(benches);

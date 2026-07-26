#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![cfg_attr(coverage_nightly, coverage(off))]

//! Focused 3D within-radius benchmark for point-bearing result projections.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kiddo::dist::SquaredEuclidean;
use kiddo::kd_tree::KdTree;
use kiddo::leaf_strategy::VecOfArenas;
use kiddo::stem_strategy::Eytzinger;
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

type F64Tree = KdTree<f64, u32, Eytzinger, VecOfArenas<f64, u32, K, B>, K, B>;
type F32Tree = KdTree<f32, u32, Eytzinger, VecOfArenas<f32, u32, K, B>, K, B>;

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

fn run_points_only_f64(tree: &F64Tree, queries: &[[f64; K]], radius_squared: f64) -> usize {
    let mut len = 0usize;
    for query in queries {
        let results = tree
            .query(black_box(query))
            .within::<SquaredEuclidean<f64>>(radius_squared)
            .unsorted()
            .with_points()
            .without_items()
            .without_distances()
            .execute();
        len = len.wrapping_add(results.len());
        black_box(&results);
    }
    len
}

fn run_points_distances_f64(tree: &F64Tree, queries: &[[f64; K]], radius_squared: f64) -> usize {
    let mut len = 0usize;
    for query in queries {
        let results = tree
            .query(black_box(query))
            .within::<SquaredEuclidean<f64>>(radius_squared)
            .unsorted()
            .with_points()
            .without_items()
            .execute();
        len = len.wrapping_add(results.len());
        black_box(&results);
    }
    len
}

fn run_points_distances_items_f64(
    tree: &F64Tree,
    queries: &[[f64; K]],
    radius_squared: f64,
) -> usize {
    let mut len = 0usize;
    for query in queries {
        let results = tree
            .query(black_box(query))
            .within::<SquaredEuclidean<f64>>(radius_squared)
            .unsorted()
            .with_points()
            .execute();
        len = len.wrapping_add(results.len());
        black_box(&results);
    }
    len
}

fn run_points_only_f32(tree: &F32Tree, queries: &[[f32; K]], radius_squared: f32) -> usize {
    let mut len = 0usize;
    for query in queries {
        let results = tree
            .query(black_box(query))
            .within::<SquaredEuclidean<f32>>(radius_squared)
            .unsorted()
            .with_points()
            .without_items()
            .without_distances()
            .execute();
        len = len.wrapping_add(results.len());
        black_box(&results);
    }
    len
}

fn run_points_distances_f32(tree: &F32Tree, queries: &[[f32; K]], radius_squared: f32) -> usize {
    let mut len = 0usize;
    for query in queries {
        let results = tree
            .query(black_box(query))
            .within::<SquaredEuclidean<f32>>(radius_squared)
            .unsorted()
            .with_points()
            .without_items()
            .execute();
        len = len.wrapping_add(results.len());
        black_box(&results);
    }
    len
}

fn run_points_distances_items_f32(
    tree: &F32Tree,
    queries: &[[f32; K]],
    radius_squared: f32,
) -> usize {
    let mut len = 0usize;
    for query in queries {
        let results = tree
            .query(black_box(query))
            .within::<SquaredEuclidean<f32>>(radius_squared)
            .unsorted()
            .with_points()
            .execute();
        len = len.wrapping_add(results.len());
        black_box(&results);
    }
    len
}

fn within_radius_point_projection(c: &mut Criterion) {
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
        "benchmarking focused 3D within-radius point projections: scalars=f32,f64 tree_sizes=2^{min_log2}..2^{max_log2} queries={query_count} radius={radius_f64} point_seed={POINT_SEED} query_seed={QUERY_SEED}"
    );

    let queries = build_queries_f32(query_count);
    let mut group = c.benchmark_group("profile_v6_within_radius_point_projection/f32");
    group.throughput(Throughput::Elements(query_count as u64));
    for log2_points in min_log2..=max_log2 {
        let point_count = 1usize << log2_points;
        let points = build_points_f32(point_count);
        let tree: F32Tree = KdTree::new_from_slice(&points).unwrap();
        let radius_squared = radius_f32 * radius_f32;

        group.bench_function(BenchmarkId::new("points_only", point_count), |b| {
            b.iter(|| black_box(run_points_only_f32(&tree, &queries, radius_squared)));
        });
        group.bench_function(BenchmarkId::new("points_distances", point_count), |b| {
            b.iter(|| black_box(run_points_distances_f32(&tree, &queries, radius_squared)));
        });
        group.bench_function(
            BenchmarkId::new("points_distances_items", point_count),
            |b| {
                b.iter(|| {
                    black_box(run_points_distances_items_f32(
                        &tree,
                        &queries,
                        radius_squared,
                    ))
                });
            },
        );
    }
    group.finish();

    let queries = build_queries_f64(query_count);
    let mut group = c.benchmark_group("profile_v6_within_radius_point_projection/f64");
    group.throughput(Throughput::Elements(query_count as u64));
    for log2_points in min_log2..=max_log2 {
        let point_count = 1usize << log2_points;
        let points = build_points_f64(point_count);
        let tree: F64Tree = KdTree::new_from_slice(&points).unwrap();
        let radius_squared = radius_f64 * radius_f64;

        group.bench_function(BenchmarkId::new("points_only", point_count), |b| {
            b.iter(|| black_box(run_points_only_f64(&tree, &queries, radius_squared)));
        });
        group.bench_function(BenchmarkId::new("points_distances", point_count), |b| {
            b.iter(|| black_box(run_points_distances_f64(&tree, &queries, radius_squared)));
        });
        group.bench_function(
            BenchmarkId::new("points_distances_items", point_count),
            |b| {
                b.iter(|| {
                    black_box(run_points_distances_items_f64(
                        &tree,
                        &queries,
                        radius_squared,
                    ))
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, within_radius_point_projection);
criterion_main!(benches);

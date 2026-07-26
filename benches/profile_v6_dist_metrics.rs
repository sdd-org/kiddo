#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![cfg_attr(coverage_nightly, coverage(off))]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kiddo::dist::{DistanceMetricScalar, QueryMetric};
use kiddo::kd_tree::KdTree;
use kiddo::leaf_strategy::FlatVec;
use kiddo::stem_strategy::Eytzinger;
use kiddo::{Chebyshev, Manhattan, Minkowski, SquaredEuclidean};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;

mod profile_tree_sizes;

const K: usize = 3;
const B: usize = 32;
const DEFAULT_QUERY_COUNT: usize = 1_000;
const POINT_SEED: u64 = 0x5eed_0000_0000_0001;
const QUERY_SEED: u64 = 0x5eed_0000_0000_0002;
const TREE_SIZES: [usize; 6] = [1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 25];

type F64Tree = KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, K, B>, K, B>;

fn read_usize_env(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn build_points(point_count: usize) -> Vec<[f64; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(POINT_SEED);
    (0..point_count).map(|_| rng.random::<[f64; K]>()).collect()
}

fn build_queries(query_count: usize) -> Vec<[f64; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(QUERY_SEED);
    (0..query_count).map(|_| rng.random::<[f64; K]>()).collect()
}

fn run_queries<M>(tree: &F64Tree, queries: &[[f64; K]]) -> (f64, u64)
where
    M: QueryMetric<f64> + DistanceMetricScalar<f64, Output = f64>,
{
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for query in queries {
        let result = tree.query(black_box(query)).nearest_one::<M>().execute();
        checksum_dist += result.distance;
        checksum_item = checksum_item.wrapping_add(result.item as u64);
    }

    (checksum_dist, checksum_item)
}

fn dist_metrics(c: &mut Criterion) {
    let query_count = read_usize_env("KIDDO_PROFILE_QUERIES", DEFAULT_QUERY_COUNT);
    let tree_sizes = profile_tree_sizes::from_env(&TREE_SIZES);

    eprintln!(
        "benchmarking v6 distance metrics: dims={} tree_sizes=2^{}..2^{} queries={} point_seed={} query_seed={}",
        K, tree_sizes[0].ilog2(), tree_sizes.last().unwrap().ilog2(), query_count, POINT_SEED, QUERY_SEED
    );

    let queries = build_queries(query_count);
    let mut group = c.benchmark_group("profile_v6_dist_metrics/f64");
    group.throughput(Throughput::Elements(query_count as u64));

    for point_count in tree_sizes {
        let points = build_points(point_count);
        let tree: F64Tree = KdTree::new_from_slice(&points).unwrap();

        group.bench_function(BenchmarkId::new("squared_euclidean", point_count), |b| {
            b.iter(|| black_box(run_queries::<SquaredEuclidean<f64>>(&tree, &queries)))
        });
        group.bench_function(BenchmarkId::new("manhattan", point_count), |b| {
            b.iter(|| black_box(run_queries::<Manhattan<f64>>(&tree, &queries)))
        });
        group.bench_function(BenchmarkId::new("chebyshev", point_count), |b| {
            b.iter(|| black_box(run_queries::<Chebyshev<f64>>(&tree, &queries)))
        });
        group.bench_function(BenchmarkId::new("minkowski_p3", point_count), |b| {
            b.iter(|| black_box(run_queries::<Minkowski<3, f64>>(&tree, &queries)))
        });
    }

    group.finish();
}

criterion_group!(benches, dist_metrics);
criterion_main!(benches);

#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![cfg_attr(coverage_nightly, coverage(off))]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kiddo::kd_tree::KdTree;
use kiddo::leaf_strategy::{FlatVec, VecOfArenas, VecOfArrays};
use kiddo::stem_strategy::Eytzinger;
use kiddo::SquaredEuclidean;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;

mod profile_tree_sizes;

const K: usize = 3;
const B: usize = 32;
const DEFAULT_QUERY_COUNT: usize = 1_000;
const POINT_SEED: u64 = 0x5eed_0000_0000_0201;
const QUERY_SEED: u64 = 0x5eed_0000_0000_0202;
const TREE_SIZES: [usize; 6] = [1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 25];

type FlatTree = KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, K, B>, K, B>;
type ArenaTree = KdTree<f64, u32, Eytzinger, VecOfArenas<f64, u32, K, B>, K, B>;
type VecOfArraysTree = KdTree<f64, u32, Eytzinger, VecOfArrays<f64, u32, K, B>, K, B>;

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

fn run_queries_flat(tree: &FlatTree, queries: &[[f64; K]]) -> (f64, u64) {
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for query in queries {
        let result = tree
            .query(black_box(query))
            .nearest_one::<SquaredEuclidean<f64>>()
            .execute();
        checksum_dist += result.distance;
        checksum_item = checksum_item.wrapping_add(result.item as u64);
    }

    (checksum_dist, checksum_item)
}

fn run_queries_arena(tree: &ArenaTree, queries: &[[f64; K]]) -> (f64, u64) {
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for query in queries {
        let result = tree
            .query(black_box(query))
            .nearest_one::<SquaredEuclidean<f64>>()
            .execute();
        checksum_dist += result.distance;
        checksum_item = checksum_item.wrapping_add(result.item as u64);
    }

    (checksum_dist, checksum_item)
}

fn run_queries_vec_of_arrays(tree: &VecOfArraysTree, queries: &[[f64; K]]) -> (f64, u64) {
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for query in queries {
        let result = tree
            .query(black_box(query))
            .nearest_one::<SquaredEuclidean<f64>>()
            .execute();
        checksum_dist += result.distance;
        checksum_item = checksum_item.wrapping_add(result.item as u64);
    }

    (checksum_dist, checksum_item)
}

fn leaf_strategies(c: &mut Criterion) {
    let query_count = read_usize_env("KIDDO_PROFILE_QUERIES", DEFAULT_QUERY_COUNT);
    let tree_sizes = profile_tree_sizes::from_env(&TREE_SIZES);

    eprintln!(
        "benchmarking v6 leaf strategies: dims={} tree_sizes=2^{}..2^{} queries={} point_seed={} query_seed={}",
        K, tree_sizes[0].ilog2(), tree_sizes.last().unwrap().ilog2(), query_count, POINT_SEED, QUERY_SEED
    );

    let queries = build_queries(query_count);
    let mut group = c.benchmark_group("profile_v6_leaf_strategies/f64");
    group.throughput(Throughput::Elements(query_count as u64));

    for point_count in tree_sizes {
        let points = build_points(point_count);

        let flat_tree: FlatTree = KdTree::new_from_slice(&points).unwrap();
        group.bench_function(BenchmarkId::new("flat", point_count), |b| {
            b.iter(|| black_box(run_queries_flat(&flat_tree, &queries)))
        });

        let arena_tree: ArenaTree = KdTree::new_from_slice(&points).unwrap();
        group.bench_function(BenchmarkId::new("arena", point_count), |b| {
            b.iter(|| black_box(run_queries_arena(&arena_tree, &queries)))
        });

        let vec_of_arrays_tree: VecOfArraysTree = KdTree::new_from_slice(&points).unwrap();
        group.bench_function(BenchmarkId::new("vec_of_arrays", point_count), |b| {
            b.iter(|| black_box(run_queries_vec_of_arrays(&vec_of_arrays_tree, &queries)))
        });
    }

    group.finish();
}

criterion_group!(benches, leaf_strategies);
criterion_main!(benches);

#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![cfg_attr(coverage_nightly, coverage(off))]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kiddo::dist::SquaredEuclidean;
use kiddo::kd_tree::KdTree;
use kiddo::leaf_strategy::FlatVec;
use kiddo::stem_strategy::Eytzinger;
use kiddo::ITEM_LEAF_MODE_SORTED;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;
use std::num::NonZeroUsize;

mod profile_tree_sizes;

const K: usize = 3;
const B: usize = 32;
const DEFAULT_QUERY_COUNT: usize = 1_000;
const POINT_SEED: u64 = 0x5eed_0000_0000_0001;
const QUERY_SEED: u64 = 0x5eed_0000_0000_0002;
const MAX_DIST: f64 = 0.0025;
const MAX_QTYS: [usize; 3] = [5, 20, 50];
const TREE_SIZES: [usize; 10] = [
    1 << 16,
    1 << 17,
    1 << 18,
    1 << 19,
    1 << 20,
    1 << 21,
    1 << 22,
    1 << 23,
    1 << 24,
    1 << 25,
];

type F64Tree = KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, K, B>, K, B>;
type F32Tree = KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, K, B>, K, B>;
type ItemSortedF64Tree =
    KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, K, B>, K, B, ITEM_LEAF_MODE_SORTED>;
type ItemSortedF32Tree =
    KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, K, B>, K, B, ITEM_LEAF_MODE_SORTED>;

fn read_usize_env(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn build_points_f64(point_count: usize) -> Vec<[f64; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(POINT_SEED);
    (0..point_count).map(|_| rng.random::<[f64; K]>()).collect()
}

fn build_queries_f64(query_count: usize) -> Vec<[f64; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(QUERY_SEED);
    (0..query_count).map(|_| rng.random::<[f64; K]>()).collect()
}

fn build_points_f32(point_count: usize) -> Vec<[f32; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(POINT_SEED);
    (0..point_count).map(|_| rng.random::<[f32; K]>()).collect()
}

fn build_queries_f32(query_count: usize) -> Vec<[f32; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(QUERY_SEED);
    (0..query_count).map(|_| rng.random::<[f32; K]>()).collect()
}

fn run_within_f64(tree: &F64Tree, queries: &[[f64; K]]) -> (usize, f64, u64) {
    let mut checksum_len = 0usize;
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for query in queries {
        let results = tree
            .query(black_box(query))
            .within::<SquaredEuclidean<f64>>(MAX_DIST)
            .execute();
        checksum_len = checksum_len.wrapping_add(results.len());
        for result in results {
            checksum_dist += result.distance;
            checksum_item = checksum_item.wrapping_add(result.item as u64);
        }
    }

    (checksum_len, checksum_dist, checksum_item)
}

fn run_within_f32(tree: &F32Tree, queries: &[[f32; K]]) -> (usize, f32, u64) {
    let mut checksum_len = 0usize;
    let mut checksum_dist = 0.0f32;
    let mut checksum_item = 0u64;

    for query in queries {
        let results = tree
            .query(black_box(query))
            .within::<SquaredEuclidean<f32>>(MAX_DIST as f32)
            .execute();
        checksum_len = checksum_len.wrapping_add(results.len());
        for result in results {
            checksum_dist += result.distance;
            checksum_item = checksum_item.wrapping_add(result.item as u64);
        }
    }

    (checksum_len, checksum_dist, checksum_item)
}

fn run_within_unsorted_f64(tree: &F64Tree, queries: &[[f64; K]]) -> (usize, f64, u64) {
    let mut checksum_len = 0usize;
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for query in queries {
        let results = tree
            .query(black_box(query))
            .within::<SquaredEuclidean<f64>>(MAX_DIST)
            .unsorted()
            .execute();
        checksum_len = checksum_len.wrapping_add(results.len());
        for result in results {
            checksum_dist += result.distance;
            checksum_item = checksum_item.wrapping_add(result.item as u64);
        }
    }

    (checksum_len, checksum_dist, checksum_item)
}

fn run_within_unsorted_f32(tree: &F32Tree, queries: &[[f32; K]]) -> (usize, f32, u64) {
    let mut checksum_len = 0usize;
    let mut checksum_dist = 0.0f32;
    let mut checksum_item = 0u64;

    for query in queries {
        let results = tree
            .query(black_box(query))
            .within::<SquaredEuclidean<f32>>(MAX_DIST as f32)
            .unsorted()
            .execute();
        checksum_len = checksum_len.wrapping_add(results.len());
        for result in results {
            checksum_dist += result.distance;
            checksum_item = checksum_item.wrapping_add(result.item as u64);
        }
    }

    (checksum_len, checksum_dist, checksum_item)
}

fn run_nearest_n_within_f64(
    tree: &F64Tree,
    queries: &[[f64; K]],
    max_qty: NonZeroUsize,
) -> (usize, f64, u64) {
    let mut checksum_len = 0usize;
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for query in queries {
        let results = tree
            .query(black_box(query))
            .nearest_n::<SquaredEuclidean<f64>>(max_qty)
            .within(MAX_DIST)
            .execute();
        checksum_len = checksum_len.wrapping_add(results.len());
        for result in results {
            checksum_dist += result.distance;
            checksum_item = checksum_item.wrapping_add(result.item as u64);
        }
    }

    (checksum_len, checksum_dist, checksum_item)
}

fn run_nearest_n_within_f32(
    tree: &F32Tree,
    queries: &[[f32; K]],
    max_qty: NonZeroUsize,
) -> (usize, f32, u64) {
    let mut checksum_len = 0usize;
    let mut checksum_dist = 0.0f32;
    let mut checksum_item = 0u64;

    for query in queries {
        let results = tree
            .query(black_box(query))
            .nearest_n::<SquaredEuclidean<f32>>(max_qty)
            .within(MAX_DIST as f32)
            .execute();
        checksum_len = checksum_len.wrapping_add(results.len());
        for result in results {
            checksum_dist += result.distance;
            checksum_item = checksum_item.wrapping_add(result.item as u64);
        }
    }

    (checksum_len, checksum_dist, checksum_item)
}

fn run_nearest_n_within_unsorted_f64(
    tree: &F64Tree,
    queries: &[[f64; K]],
    max_qty: NonZeroUsize,
) -> (usize, f64, u64) {
    let mut checksum_len = 0usize;
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for query in queries {
        let results = tree
            .query(black_box(query))
            .nearest_n::<SquaredEuclidean<f64>>(max_qty)
            .within(MAX_DIST)
            .unsorted()
            .execute();
        checksum_len = checksum_len.wrapping_add(results.len());
        for result in results {
            checksum_dist += result.distance;
            checksum_item = checksum_item.wrapping_add(result.item as u64);
        }
    }

    (checksum_len, checksum_dist, checksum_item)
}

fn run_nearest_n_within_unsorted_f32(
    tree: &F32Tree,
    queries: &[[f32; K]],
    max_qty: NonZeroUsize,
) -> (usize, f32, u64) {
    let mut checksum_len = 0usize;
    let mut checksum_dist = 0.0f32;
    let mut checksum_item = 0u64;

    for query in queries {
        let results = tree
            .query(black_box(query))
            .nearest_n::<SquaredEuclidean<f32>>(max_qty)
            .within(MAX_DIST as f32)
            .unsorted()
            .execute();
        checksum_len = checksum_len.wrapping_add(results.len());
        for result in results {
            checksum_dist += result.distance;
            checksum_item = checksum_item.wrapping_add(result.item as u64);
        }
    }

    (checksum_len, checksum_dist, checksum_item)
}

fn run_best_n_within_f64(
    tree: &ItemSortedF64Tree,
    queries: &[[f64; K]],
    max_qty: NonZeroUsize,
) -> (usize, f64, u64) {
    let mut checksum_len = 0usize;
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for query in queries {
        let results = tree
            .query(black_box(query))
            .best_n_within::<SquaredEuclidean<f64>>(MAX_DIST, max_qty)
            .execute();
        checksum_len = checksum_len.wrapping_add(results.len());
        for result in results.into_vec() {
            checksum_dist += result.distance;
            checksum_item = checksum_item.wrapping_add(result.item as u64);
        }
    }

    (checksum_len, checksum_dist, checksum_item)
}

fn run_best_n_within_f32(
    tree: &ItemSortedF32Tree,
    queries: &[[f32; K]],
    max_qty: NonZeroUsize,
) -> (usize, f32, u64) {
    let mut checksum_len = 0usize;
    let mut checksum_dist = 0.0f32;
    let mut checksum_item = 0u64;

    for query in queries {
        let results = tree
            .query(black_box(query))
            .best_n_within::<SquaredEuclidean<f32>>(MAX_DIST as f32, max_qty)
            .execute();
        checksum_len = checksum_len.wrapping_add(results.len());
        for result in results.into_vec() {
            checksum_dist += result.distance;
            checksum_item = checksum_item.wrapping_add(result.item as u64);
        }
    }

    (checksum_len, checksum_dist, checksum_item)
}

fn query_family(c: &mut Criterion) {
    let query_count = read_usize_env("KIDDO_PROFILE_QUERIES", DEFAULT_QUERY_COUNT);
    let tree_sizes = profile_tree_sizes::from_env(&TREE_SIZES);

    eprintln!(
        "benchmarking v6 query family: dims={} tree_sizes=2^{}..2^{} queries={} max_dist={} ks={:?} point_seed={} query_seed={}",
        K, tree_sizes[0].ilog2(), tree_sizes.last().unwrap().ilog2(), query_count, MAX_DIST, MAX_QTYS, POINT_SEED, QUERY_SEED
    );

    let f64_queries = build_queries_f64(query_count);
    let mut f64_group = c.benchmark_group("profile_v6_query_family_eytzinger/f64");
    f64_group.throughput(Throughput::Elements(query_count as u64));
    for point_count in tree_sizes.iter().copied() {
        let points = build_points_f64(point_count);
        let tree: F64Tree = KdTree::new_from_slice(&points).unwrap();
        f64_group.bench_function(BenchmarkId::new("within", point_count), |b| {
            b.iter(|| black_box(run_within_f64(&tree, &f64_queries)));
        });
        f64_group.bench_function(BenchmarkId::new("within_unsorted", point_count), |b| {
            b.iter(|| black_box(run_within_unsorted_f64(&tree, &f64_queries)));
        });
        for max_qty in MAX_QTYS {
            let max_qty = NonZeroUsize::new(max_qty).unwrap();
            f64_group.bench_function(
                BenchmarkId::new(format!("nearest_n_within_k{}", max_qty), point_count),
                |b| b.iter(|| black_box(run_nearest_n_within_f64(&tree, &f64_queries, max_qty))),
            );
            f64_group.bench_function(
                BenchmarkId::new(
                    format!("nearest_n_within_unsorted_k{}", max_qty),
                    point_count,
                ),
                |b| {
                    b.iter(|| {
                        black_box(run_nearest_n_within_unsorted_f64(
                            &tree,
                            &f64_queries,
                            max_qty,
                        ))
                    })
                },
            );
        }
        drop(tree);

        let item_sorted_tree: ItemSortedF64Tree = F64Tree::builder()
            .with_item_sorted_leaves()
            .build_from_slice(&points)
            .unwrap();
        for max_qty in MAX_QTYS {
            let max_qty = NonZeroUsize::new(max_qty).unwrap();
            f64_group.bench_function(
                BenchmarkId::new(format!("best_n_within_k{}", max_qty), point_count),
                |b| {
                    b.iter(|| {
                        black_box(run_best_n_within_f64(
                            &item_sorted_tree,
                            &f64_queries,
                            max_qty,
                        ))
                    })
                },
            );
        }
    }
    f64_group.finish();

    let f32_queries = build_queries_f32(query_count);
    let mut f32_group = c.benchmark_group("profile_v6_query_family_eytzinger/f32");
    f32_group.throughput(Throughput::Elements(query_count as u64));
    for point_count in tree_sizes {
        let points = build_points_f32(point_count);
        let tree: F32Tree = KdTree::new_from_slice(&points).unwrap();
        f32_group.bench_function(BenchmarkId::new("within", point_count), |b| {
            b.iter(|| black_box(run_within_f32(&tree, &f32_queries)));
        });
        f32_group.bench_function(BenchmarkId::new("within_unsorted", point_count), |b| {
            b.iter(|| black_box(run_within_unsorted_f32(&tree, &f32_queries)));
        });
        for max_qty in MAX_QTYS {
            let max_qty = NonZeroUsize::new(max_qty).unwrap();
            f32_group.bench_function(
                BenchmarkId::new(format!("nearest_n_within_k{}", max_qty), point_count),
                |b| b.iter(|| black_box(run_nearest_n_within_f32(&tree, &f32_queries, max_qty))),
            );
            f32_group.bench_function(
                BenchmarkId::new(
                    format!("nearest_n_within_unsorted_k{}", max_qty),
                    point_count,
                ),
                |b| {
                    b.iter(|| {
                        black_box(run_nearest_n_within_unsorted_f32(
                            &tree,
                            &f32_queries,
                            max_qty,
                        ))
                    })
                },
            );
        }
        drop(tree);

        let item_sorted_tree: ItemSortedF32Tree = F32Tree::builder()
            .with_item_sorted_leaves()
            .build_from_slice(&points)
            .unwrap();
        for max_qty in MAX_QTYS {
            let max_qty = NonZeroUsize::new(max_qty).unwrap();
            f32_group.bench_function(
                BenchmarkId::new(format!("best_n_within_k{}", max_qty), point_count),
                |b| {
                    b.iter(|| {
                        black_box(run_best_n_within_f32(
                            &item_sorted_tree,
                            &f32_queries,
                            max_qty,
                        ))
                    })
                },
            );
        }
    }
    f32_group.finish();
}

criterion_group!(benches, query_family);
criterion_main!(benches);

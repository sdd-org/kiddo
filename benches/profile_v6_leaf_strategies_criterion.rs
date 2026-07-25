#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![cfg_attr(coverage_nightly, coverage(off))]

use criterion::{
    criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion, Throughput,
};
use kiddo::kd_tree::embedded_leaf_descriptor_experiment::EmbeddedLeafDescriptorF64Tree;
use kiddo::kd_tree::KdTree;
use kiddo::leaf_strategy::{FlatVec, VecOfArenas, VecOfArrays};
use kiddo::stem_strategy::{DonnellySimdDescent, DonnellyUnrolled, Eytzinger};
use kiddo::{LeafStrategy, SquaredEuclidean, StemStrategy};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;

const K: usize = 3;
const B: usize = 32;
const DEFAULT_QUERY_COUNT: usize = 1_000;
const DEFAULT_MAX_LOG2_POINTS: usize = 26;
const POINT_SEED: u64 = 0x5eed_0000_0000_0201;
const QUERY_SEED: u64 = 0x5eed_0000_0000_0202;
const TREE_SIZES: [usize; 6] = [1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 25];
const APPROX_TREE_SIZES: [usize; 10] = [
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

type FlatTree = KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, K, B>, K, B>;
type ArenaTree = KdTree<f64, u32, Eytzinger, VecOfArenas<f64, u32, K, B>, K, B>;
type VecOfArraysTree = KdTree<f64, u32, Eytzinger, VecOfArrays<f64, u32, K, B>, K, B>;
type ArenaLeaves = VecOfArenas<f64, u32, K, B>;
type ApproxBaselineTree<SS> = KdTree<f64, u32, SS, ArenaLeaves, K, B>;

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

fn run_approx_embedded(
    tree: &EmbeddedLeafDescriptorF64Tree<K, B>,
    queries: &[[f64; K]],
) -> (f64, u64) {
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for query in queries {
        let (distance, item) =
            tree.approx_nearest_one_embedded::<SquaredEuclidean<f64>>(black_box(query));
        checksum_dist += distance;
        checksum_item = checksum_item.wrapping_add(u64::from(item));
    }

    (checksum_dist, checksum_item)
}

fn run_approx_extent_control(
    tree: &EmbeddedLeafDescriptorF64Tree<K, B>,
    queries: &[[f64; K]],
) -> (f64, u64) {
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for query in queries {
        let (distance, item) =
            tree.approx_nearest_one_via_extents::<SquaredEuclidean<f64>>(black_box(query));
        checksum_dist += distance;
        checksum_item = checksum_item.wrapping_add(u64::from(item));
    }

    (checksum_dist, checksum_item)
}

fn run_approx_baseline<SS>(tree: &ApproxBaselineTree<SS>, queries: &[[f64; K]]) -> (f64, u64)
where
    SS: StemStrategy,
    ArenaLeaves: LeafStrategy<f64, u32, SS, K, B>,
{
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for query in queries {
        let result = tree
            .query(black_box(query))
            .nearest_one::<SquaredEuclidean<f64>>()
            .approx()
            .execute();
        checksum_dist += result.distance;
        checksum_item = checksum_item.wrapping_add(u64::from(result.item));
    }

    (checksum_dist, checksum_item)
}

fn bench_approx_baseline<SS>(
    group: &mut BenchmarkGroup<'_, criterion::measurement::WallTime>,
    label: &str,
    point_count: usize,
    points: &[[f64; K]],
    queries: &[[f64; K]],
) where
    SS: StemStrategy,
    ArenaLeaves: LeafStrategy<f64, u32, SS, K, B>,
{
    let tree: ApproxBaselineTree<SS> = KdTree::new_from_slice(points).unwrap();
    group.bench_function(BenchmarkId::new(label, point_count), |b| {
        b.iter(|| black_box(run_approx_baseline(&tree, queries)))
    });
}

fn leaf_strategies(c: &mut Criterion) {
    let query_count = read_usize_env("KIDDO_PROFILE_QUERIES", DEFAULT_QUERY_COUNT);
    let max_log2_points = read_usize_env("KIDDO_PROFILE_MAX_LOG2_POINTS", DEFAULT_MAX_LOG2_POINTS);

    eprintln!(
        "benchmarking v6 leaf strategies: dims={} tree_sizes=2^16,2^18,... max_log2_points={} queries={} point_seed={} query_seed={}",
        K, max_log2_points, query_count, POINT_SEED, QUERY_SEED
    );

    let queries = build_queries(query_count);
    let mut group = c.benchmark_group("profile_v6_leaf_strategies/f64");
    group.throughput(Throughput::Elements(query_count as u64));

    for point_count in TREE_SIZES
        .into_iter()
        .filter(|point_count| point_count.ilog2() as usize <= max_log2_points)
    {
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

    eprintln!(
        "benchmarking v6 approx_nearest_one f64 embedded leaf descriptors: dims={} tree_sizes=2^16..=2^25 queries={}",
        K, query_count
    );

    let mut approx_group = c.benchmark_group("profile_v6_leaf_strategies/approx_nearest_one_f64");
    approx_group.throughput(Throughput::Elements(query_count as u64));

    for point_count in APPROX_TREE_SIZES {
        let points = build_points(point_count);
        let embedded_tree = EmbeddedLeafDescriptorF64Tree::<K, B>::new_from_slice(&points).unwrap();

        assert_eq!(
            run_approx_embedded(&embedded_tree, &queries),
            run_approx_extent_control(&embedded_tree, &queries)
        );

        approx_group.bench_function(BenchmarkId::new("embedded_descriptor", point_count), |b| {
            b.iter(|| black_box(run_approx_embedded(&embedded_tree, &queries)))
        });
        approx_group.bench_function(
            BenchmarkId::new("embedded_extent_control", point_count),
            |b| b.iter(|| black_box(run_approx_extent_control(&embedded_tree, &queries))),
        );
        drop(embedded_tree);

        bench_approx_baseline::<DonnellyUnrolled<3>>(
            &mut approx_group,
            "baseline_donnelly_unrolled",
            point_count,
            &points,
            &queries,
        );
        bench_approx_baseline::<DonnellySimdDescent<3>>(
            &mut approx_group,
            "baseline_donnelly_simd_descent",
            point_count,
            &points,
            &queries,
        );
        bench_approx_baseline::<Eytzinger>(
            &mut approx_group,
            "baseline_eytzinger",
            point_count,
            &points,
            &queries,
        );
    }

    approx_group.finish();
}

criterion_group!(benches, leaf_strategies);
criterion_main!(benches);

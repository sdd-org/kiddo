use criterion::{
    criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion, Throughput,
};
use kiddo::dist::SquaredEuclidean;
use kiddo::kd_tree::embedded_leaf_descriptor_experiment::EmbeddedLeafDescriptorF64Tree;
use kiddo::kd_tree::KdTree;
use kiddo::leaf_strategy::VecOfArenas;
use kiddo::stem_strategy::{Donnelly, DonnellySimdDescent, DonnellyUnrolled, Eytzinger};
use kiddo::{LeafStrategy, StemStrategy};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;

const K: usize = 3;
const B: usize = 32;
const DEFAULT_QUERY_COUNT: usize = 1_000;
const DEFAULT_MIN_LOG2_POINTS: u32 = 16;
const DEFAULT_MAX_LOG2_POINTS: u32 = 26;
const POINT_SEED: u64 = 0x5eed_0000_0000_ed01;
const QUERY_SEED: u64 = 0x5eed_0000_0000_ed02;

type ArenaLeaves = VecOfArenas<f64, u32, K, B>;
type BaselineTree<SS> = KdTree<f64, u32, SS, ArenaLeaves, K, B>;

fn read_usize_env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn read_u32_env(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn build_points(point_count: usize) -> Vec<[f64; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(POINT_SEED);
    (0..point_count).map(|_| rng.random()).collect()
}

fn build_queries(query_count: usize) -> Vec<[f64; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(QUERY_SEED);
    (0..query_count).map(|_| rng.random()).collect()
}

fn run_embedded(tree: &EmbeddedLeafDescriptorF64Tree<K, B>, queries: &[[f64; K]]) -> (f64, u64) {
    let mut distance_checksum = 0.0;
    let mut item_checksum = 0u64;
    for query in queries {
        let (distance, item) =
            tree.approx_nearest_one_embedded::<SquaredEuclidean<f64>>(black_box(query));
        distance_checksum += distance;
        item_checksum = item_checksum.wrapping_add(u64::from(item));
    }
    (distance_checksum, item_checksum)
}

fn run_extent_control(
    tree: &EmbeddedLeafDescriptorF64Tree<K, B>,
    queries: &[[f64; K]],
) -> (f64, u64) {
    let mut distance_checksum = 0.0;
    let mut item_checksum = 0u64;
    for query in queries {
        let (distance, item) =
            tree.approx_nearest_one_via_extents::<SquaredEuclidean<f64>>(black_box(query));
        distance_checksum += distance;
        item_checksum = item_checksum.wrapping_add(u64::from(item));
    }
    (distance_checksum, item_checksum)
}

fn run_baseline<SS>(tree: &BaselineTree<SS>, queries: &[[f64; K]]) -> (f64, u64)
where
    SS: StemStrategy,
    ArenaLeaves: LeafStrategy<f64, u32, SS, K, B>,
{
    let mut distance_checksum = 0.0;
    let mut item_checksum = 0u64;
    for query in queries {
        let result = tree
            .query(black_box(query))
            .nearest_one::<SquaredEuclidean<f64>>()
            .approx()
            .execute();
        distance_checksum += result.distance;
        item_checksum = item_checksum.wrapping_add(u64::from(result.item));
    }
    (distance_checksum, item_checksum)
}

fn bench_baseline<SS>(
    group: &mut BenchmarkGroup<'_, criterion::measurement::WallTime>,
    label: &str,
    point_count: usize,
    points: &[[f64; K]],
    queries: &[[f64; K]],
) where
    SS: StemStrategy,
    ArenaLeaves: LeafStrategy<f64, u32, SS, K, B>,
{
    let tree: BaselineTree<SS> = KdTree::new_from_slice(points).unwrap();
    group.bench_function(BenchmarkId::new(label, point_count), |bencher| {
        bencher.iter(|| black_box(run_baseline(&tree, queries)))
    });
}

fn embedded_leaf_descriptor(c: &mut Criterion) {
    let query_count = read_usize_env("KIDDO_BENCH_QUERIES", DEFAULT_QUERY_COUNT);
    let min_log2 = read_u32_env("KIDDO_BENCH_MIN_LOG2_POINTS", DEFAULT_MIN_LOG2_POINTS);
    let max_log2 = read_u32_env("KIDDO_BENCH_MAX_LOG2_POINTS", DEFAULT_MAX_LOG2_POINTS);
    assert!(min_log2 <= max_log2);

    let queries = build_queries(query_count);
    let mut group = c.benchmark_group("profile_v6_embedded_leaf_descriptor/f64_k3_b32");
    group.throughput(Throughput::Elements(query_count as u64));

    for log2_points in min_log2..=max_log2 {
        let point_count = 1usize << log2_points;
        let points = build_points(point_count);
        let tree = EmbeddedLeafDescriptorF64Tree::<K, B>::new_from_slice(&points).unwrap();

        assert_eq!(
            run_embedded(&tree, &queries),
            run_extent_control(&tree, &queries)
        );
        eprintln!(
            "embedded descriptor tree: 2^{log2_points} points, {} leaves, {} stem words",
            tree.leaf_count(),
            tree.stem_word_count()
        );

        group.bench_function(
            BenchmarkId::new("embedded_descriptor", point_count),
            |bencher| bencher.iter(|| black_box(run_embedded(&tree, &queries))),
        );
        group.bench_function(
            BenchmarkId::new("embedded_extent_control", point_count),
            |bencher| bencher.iter(|| black_box(run_extent_control(&tree, &queries))),
        );
        drop(tree);

        bench_baseline::<DonnellyUnrolled<3>>(
            &mut group,
            "baseline_donnelly_unrolled",
            point_count,
            &points,
            &queries,
        );
        bench_baseline::<Donnelly<3>>(
            &mut group,
            "baseline_donnelly",
            point_count,
            &points,
            &queries,
        );
        bench_baseline::<DonnellySimdDescent<3>>(
            &mut group,
            "baseline_donnelly_simd_descent",
            point_count,
            &points,
            &queries,
        );
        bench_baseline::<Eytzinger>(
            &mut group,
            "baseline_eytzinger",
            point_count,
            &points,
            &queries,
        );
    }

    group.finish();
}

criterion_group!(benches, embedded_leaf_descriptor);
criterion_main!(benches);

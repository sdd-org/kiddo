#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![cfg_attr(coverage_nightly, coverage(off))]

use criterion::measurement::WallTime;
use criterion::{
    criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion, Throughput,
};
use kiddo::kd_tree::KdTree;
use kiddo::leaf_strategy::FlatVec;
use kiddo::StemStrategy;
use kiddo::{
    Donnelly, DonnellyNoPf, DonnellySimdDescent, DonnellySimdFull, DonnellyUnrolled,
    DonnellyUnrolledBlockDim, Eytzinger, EytzingerFlexPf, EytzingerNoPf, SquaredEuclidean,
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;

mod profile_tree_sizes;

const K: usize = 3;
const B: usize = 32;
const DEFAULT_QUERY_COUNT: usize = 1_000;
const POINT_SEED: u64 = 0x5eed_0000_0000_0301;
const QUERY_SEED: u64 = 0x5eed_0000_0000_0302;
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

type F64Tree<SS> = KdTree<f64, u32, SS, FlatVec<f64, u32, K, B>, K, B>;
type F32Tree<SS> = KdTree<f32, u32, SS, FlatVec<f32, u32, K, B>, K, B>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StemBenchMode {
    Scalar,
    Avx2,
    Avx512,
    Neon,
}

fn read_usize_env(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn read_mode_env() -> StemBenchMode {
    match std::env::var("KIDDO_STEM_BENCH_MODE")
        .ok()
        .as_deref()
        .unwrap_or("scalar")
    {
        "avx2" => StemBenchMode::Avx2,
        "avx512" => StemBenchMode::Avx512,
        "neon" => StemBenchMode::Neon,
        _ => StemBenchMode::Scalar,
    }
}

fn strategy_enabled(label: &str) -> bool {
    let Ok(filter) = std::env::var("KIDDO_STEM_STRATEGIES") else {
        return true;
    };
    filter.split(',').any(|candidate| candidate.trim() == label)
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

fn run_queries_f64<SS: StemStrategy>(tree: &F64Tree<SS>, queries: &[[f64; K]]) -> (f64, u64) {
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

fn run_queries_f32<SS: StemStrategy>(tree: &F32Tree<SS>, queries: &[[f32; K]]) -> (f32, u64) {
    let mut checksum_dist = 0.0f32;
    let mut checksum_item = 0u64;

    for query in queries {
        let result = tree
            .query(black_box(query))
            .nearest_one::<SquaredEuclidean<f32>>()
            .execute();
        checksum_dist += result.distance;
        checksum_item = checksum_item.wrapping_add(result.item as u64);
    }

    (checksum_dist, checksum_item)
}

fn bench_f64_strategy<SS: StemStrategy>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    label: &str,
    point_count: usize,
    points: &[[f64; K]],
    queries: &[[f64; K]],
) {
    if !strategy_enabled(label) {
        return;
    }

    let tree: F64Tree<SS> = KdTree::new_from_slice(points).unwrap();
    group.bench_function(BenchmarkId::new(label, point_count), |b| {
        b.iter(|| black_box(run_queries_f64(&tree, queries)))
    });
}

fn bench_f32_strategy<SS: StemStrategy>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    label: &str,
    point_count: usize,
    points: &[[f32; K]],
    queries: &[[f32; K]],
) {
    if !strategy_enabled(label) {
        return;
    }

    let tree: F32Tree<SS> = KdTree::new_from_slice(points).unwrap();
    group.bench_function(BenchmarkId::new(label, point_count), |b| {
        b.iter(|| black_box(run_queries_f32(&tree, queries)))
    });
}

fn bench_scalar_f64(
    group: &mut BenchmarkGroup<'_, WallTime>,
    point_count: usize,
    points: &[[f64; K]],
    queries: &[[f64; K]],
) {
    bench_f64_strategy::<Eytzinger>(group, "eytzinger", point_count, points, queries);
    bench_f64_strategy::<EytzingerNoPf>(group, "eytzinger_no_pf", point_count, points, queries);
    bench_f64_strategy::<EytzingerFlexPf<-1, 0>>(
        group,
        "eytzinger_flex_pf_none_t0",
        point_count,
        points,
        queries,
    );
    bench_f64_strategy::<EytzingerFlexPf<-1, 1>>(
        group,
        "eytzinger_flex_pf_none_t1",
        point_count,
        points,
        queries,
    );
    bench_f64_strategy::<EytzingerFlexPf<0, -1>>(
        group,
        "eytzinger_flex_pf_t0_none",
        point_count,
        points,
        queries,
    );
    bench_f64_strategy::<EytzingerFlexPf<0, 0>>(
        group,
        "eytzinger_flex_pf_t0_t0",
        point_count,
        points,
        queries,
    );
    bench_f64_strategy::<EytzingerFlexPf<1, -1>>(
        group,
        "eytzinger_flex_pf_t1_none",
        point_count,
        points,
        queries,
    );
    bench_f64_strategy::<EytzingerFlexPf<1, 0>>(
        group,
        "eytzinger_flex_pf_t1_t0",
        point_count,
        points,
        queries,
    );
    bench_f64_strategy::<EytzingerFlexPf<1, 1>>(
        group,
        "eytzinger_flex_pf_t1_t1",
        point_count,
        points,
        queries,
    );
    bench_f64_strategy::<DonnellyNoPf<3>>(group, "donnelly_no_pf", point_count, points, queries);
    bench_f64_strategy::<Donnelly<3>>(group, "donnelly", point_count, points, queries);
    bench_f64_strategy::<DonnellyUnrolled<3>>(
        group,
        "donnelly_unrolled",
        point_count,
        points,
        queries,
    );
    bench_f64_strategy::<DonnellyUnrolledBlockDim<3>>(
        group,
        "donnelly_unrolled_block_dim",
        point_count,
        points,
        queries,
    );
}

fn bench_scalar_f32(
    group: &mut BenchmarkGroup<'_, WallTime>,
    point_count: usize,
    points: &[[f32; K]],
    queries: &[[f32; K]],
) {
    bench_f32_strategy::<Eytzinger>(group, "eytzinger", point_count, points, queries);
    bench_f32_strategy::<EytzingerNoPf>(group, "eytzinger_no_pf", point_count, points, queries);
    bench_f32_strategy::<EytzingerFlexPf<-1, 0>>(
        group,
        "eytzinger_flex_pf_none_t0",
        point_count,
        points,
        queries,
    );
    bench_f32_strategy::<EytzingerFlexPf<-1, 1>>(
        group,
        "eytzinger_flex_pf_none_t1",
        point_count,
        points,
        queries,
    );
    bench_f32_strategy::<EytzingerFlexPf<0, -1>>(
        group,
        "eytzinger_flex_pf_t0_none",
        point_count,
        points,
        queries,
    );
    bench_f32_strategy::<EytzingerFlexPf<0, 0>>(
        group,
        "eytzinger_flex_pf_t0_t0",
        point_count,
        points,
        queries,
    );
    bench_f32_strategy::<EytzingerFlexPf<1, -1>>(
        group,
        "eytzinger_flex_pf_t1_none",
        point_count,
        points,
        queries,
    );
    bench_f32_strategy::<EytzingerFlexPf<1, 0>>(
        group,
        "eytzinger_flex_pf_t1_t0",
        point_count,
        points,
        queries,
    );
    bench_f32_strategy::<EytzingerFlexPf<1, 1>>(
        group,
        "eytzinger_flex_pf_t1_t1",
        point_count,
        points,
        queries,
    );
    bench_f32_strategy::<DonnellyNoPf<4>>(group, "donnelly_no_pf", point_count, points, queries);
    bench_f32_strategy::<Donnelly<4>>(group, "donnelly", point_count, points, queries);
    bench_f32_strategy::<DonnellyUnrolled<4>>(
        group,
        "donnelly_unrolled",
        point_count,
        points,
        queries,
    );
    bench_f32_strategy::<DonnellyUnrolledBlockDim<4>>(
        group,
        "donnelly_unrolled_block_dim",
        point_count,
        points,
        queries,
    );
}

fn bench_simd_f64(
    group: &mut BenchmarkGroup<'_, WallTime>,
    point_count: usize,
    points: &[[f64; K]],
    queries: &[[f64; K]],
) {
    bench_f64_strategy::<DonnellySimdDescent<3>>(
        group,
        "donnelly_simd_descent",
        point_count,
        points,
        queries,
    );
    bench_f64_strategy::<DonnellySimdFull<3>>(
        group,
        "donnelly_simd_full",
        point_count,
        points,
        queries,
    );
}

fn bench_simd_f32(
    group: &mut BenchmarkGroup<'_, WallTime>,
    point_count: usize,
    points: &[[f32; K]],
    queries: &[[f32; K]],
) {
    bench_f32_strategy::<DonnellySimdDescent<3>>(
        group,
        "donnelly_simd_descent",
        point_count,
        points,
        queries,
    );
    bench_f32_strategy::<DonnellySimdFull<4>>(
        group,
        "donnelly_simd_full",
        point_count,
        points,
        queries,
    );
}

fn stem_strategies(c: &mut Criterion) {
    let query_count = read_usize_env("KIDDO_PROFILE_QUERIES", DEFAULT_QUERY_COUNT);
    let mode = read_mode_env();
    let tree_sizes = profile_tree_sizes::from_env(&TREE_SIZES);

    eprintln!(
        "benchmarking v6 stem strategies: dims={} tree_sizes=2^{}..2^{} queries={} mode={:?} point_seed={} query_seed={}",
        K,
        tree_sizes[0].ilog2(),
        tree_sizes.last().unwrap().ilog2(),
        query_count,
        mode,
        POINT_SEED,
        QUERY_SEED
    );

    let f64_queries = build_queries_f64(query_count);
    let mut f64_group = c.benchmark_group("profile_v6_stem_strategies/f64");
    f64_group.throughput(Throughput::Elements(query_count as u64));
    for point_count in tree_sizes.iter().copied() {
        let points = build_points_f64(point_count);
        match mode {
            StemBenchMode::Scalar => {
                bench_scalar_f64(&mut f64_group, point_count, &points, &f64_queries)
            }
            StemBenchMode::Avx2 | StemBenchMode::Avx512 | StemBenchMode::Neon => {
                bench_simd_f64(&mut f64_group, point_count, &points, &f64_queries)
            }
        }
    }
    f64_group.finish();

    let f32_queries = build_queries_f32(query_count);
    let mut f32_group = c.benchmark_group("profile_v6_stem_strategies/f32");
    f32_group.throughput(Throughput::Elements(query_count as u64));
    for point_count in tree_sizes {
        let points = build_points_f32(point_count);
        match mode {
            StemBenchMode::Scalar => {
                bench_scalar_f32(&mut f32_group, point_count, &points, &f32_queries)
            }
            StemBenchMode::Avx2 | StemBenchMode::Avx512 | StemBenchMode::Neon => {
                bench_simd_f32(&mut f32_group, point_count, &points, &f32_queries)
            }
        }
    }
    f32_group.finish();
}

criterion_group!(benches, stem_strategies);
criterion_main!(benches);

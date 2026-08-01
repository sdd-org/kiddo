#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![cfg_attr(coverage_nightly, coverage(off))]

use criterion::measurement::WallTime;
use criterion::{
    criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion, Throughput,
};
use kiddo::kd_tree::KdTree;
use kiddo::leaf_strategy::FlatVec;
use kiddo::stem_strategies::donnelly::DonnellySimdInitialDescent;
use kiddo::{
    Donnelly, DonnellyCyclicSimdDescent, DonnellySimdDescent, DonnellySimdFull, DonnellyUnrolled,
    DonnellyUnrolledBlockDim, EytzingerFlexPf, SquaredEuclidean, StemStrategy,
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;

const F64_K: usize = 3;
const F32_K: usize = 4;
const B: usize = 32;
const DEFAULT_POINT_LOG2: usize = 27;
const POINT_SEED: u64 = 0x5eed_0000_0000_0301;
const QUERY_SEED: u64 = 0x5eed_0000_0000_0302;
const SPLITMIX_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;
const DEFAULT_POOL_SIZES: [usize; 7] = [256, 512, 1_000, 2_048, 4_096, 8_192, 16_384];
const SUPPORTED_STRATEGIES: [&str; 8] = [
    "eytzinger",
    "donnelly",
    "donnelly_unrolled",
    "donnelly_unrolled_block_dim",
    "donnelly_simd_descent",
    "donnelly_cyclic_simd_descent",
    "donnelly_simd_initial_descent",
    "donnelly_simd_full",
];

type F64Tree<SS> = KdTree<f64, u32, SS, FlatVec<f64, u32, F64_K, B>, F64_K, B>;
type F32Tree<SS> = KdTree<f32, u32, SS, FlatVec<f32, u32, F32_K, B>, F32_K, B>;
type EytzingerComparator = EytzingerFlexPf<0, -1>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AxisSelection {
    Both,
    F64,
    F32,
}

fn read_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn read_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn read_axis() -> AxisSelection {
    match std::env::var("KIDDO_QUERY_POOL_AXIS")
        .unwrap_or_else(|_| "both".to_owned())
        .as_str()
    {
        "f64" => AxisSelection::F64,
        "f32" => AxisSelection::F32,
        _ => AxisSelection::Both,
    }
}

fn read_pool_sizes() -> Vec<usize> {
    let Ok(value) = std::env::var("KIDDO_QUERY_POOL_SIZES") else {
        return DEFAULT_POOL_SIZES.to_vec();
    };
    let mut sizes: Vec<usize> = value
        .split(',')
        .map(|entry| {
            entry
                .trim()
                .parse()
                .expect("KIDDO_QUERY_POOL_SIZES entries must be positive integers")
        })
        .collect();
    assert!(!sizes.is_empty());
    assert!(sizes.iter().all(|&size| size > 0));
    sizes.sort_unstable();
    sizes.dedup();
    sizes
}

fn strategy_enabled(name: &str) -> bool {
    std::env::var("KIDDO_QUERY_POOL_STRATEGIES")
        .unwrap_or_else(|_| "eytzinger,donnelly".to_owned())
        .split(',')
        .any(|entry| entry.trim() == name)
}

fn validate_strategy_selection() {
    let selection = std::env::var("KIDDO_QUERY_POOL_STRATEGIES")
        .unwrap_or_else(|_| "eytzinger,donnelly".to_owned());
    let requested: Vec<_> = selection
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .collect();
    assert!(!requested.is_empty(), "no query-pool strategies selected");
    for strategy in requested {
        assert!(
            SUPPORTED_STRATEGIES.contains(&strategy),
            "unsupported KIDDO_QUERY_POOL_STRATEGIES entry: {strategy}"
        );
    }
}

#[inline(always)]
fn splitmix64(counter: u64) -> u64 {
    let mut value = counter.wrapping_add(SPLITMIX_GAMMA);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[inline(always)]
fn query_word<const K: usize>(seed: u64, query_index: usize, dimension: usize) -> u64 {
    let counter = (query_index * K + dimension) as u64;
    splitmix64(seed.wrapping_add(counter.wrapping_mul(SPLITMIX_GAMMA)))
}

#[inline(always)]
fn query_f64(seed: u64, query_index: usize) -> [f64; F64_K] {
    const SCALE: f64 = 1.0 / ((1u64 << 53) as f64);
    [
        (query_word::<F64_K>(seed, query_index, 0) >> 11) as f64 * SCALE,
        (query_word::<F64_K>(seed, query_index, 1) >> 11) as f64 * SCALE,
        (query_word::<F64_K>(seed, query_index, 2) >> 11) as f64 * SCALE,
    ]
}

#[inline(always)]
fn query_f32(seed: u64, query_index: usize) -> [f32; F32_K] {
    const SCALE: f32 = 1.0 / ((1u32 << 24) as f32);
    [
        (query_word::<F32_K>(seed, query_index, 0) >> 40) as f32 * SCALE,
        (query_word::<F32_K>(seed, query_index, 1) >> 40) as f32 * SCALE,
        (query_word::<F32_K>(seed, query_index, 2) >> 40) as f32 * SCALE,
        (query_word::<F32_K>(seed, query_index, 3) >> 40) as f32 * SCALE,
    ]
}

fn build_points_f64(point_count: usize, seed: u64) -> Vec<[f64; F64_K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..point_count)
        .map(|_| rng.random::<[f64; F64_K]>())
        .collect()
}

fn build_points_f32(point_count: usize, seed: u64) -> Vec<[f32; F32_K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..point_count)
        .map(|_| rng.random::<[f32; F32_K]>())
        .collect()
}

fn run_stored_f64<SS: StemStrategy>(tree: &F64Tree<SS>, queries: &[[f64; F64_K]]) -> (f64, u64) {
    let mut distance = 0.0;
    let mut item = 0u64;
    for query in queries {
        let result = tree
            .query(black_box(query))
            .nearest_one::<SquaredEuclidean<f64>>()
            .execute();
        distance += result.distance;
        item = item.wrapping_add(result.item as u64);
    }
    (distance, item)
}

fn run_stored_f32<SS: StemStrategy>(tree: &F32Tree<SS>, queries: &[[f32; F32_K]]) -> (f32, u64) {
    let mut distance = 0.0;
    let mut item = 0u64;
    for query in queries {
        let result = tree
            .query(black_box(query))
            .nearest_one::<SquaredEuclidean<f32>>()
            .execute();
        distance += result.distance;
        item = item.wrapping_add(result.item as u64);
    }
    (distance, item)
}

fn run_generated_f64<SS: StemStrategy>(
    tree: &F64Tree<SS>,
    query_count: usize,
    seed: u64,
) -> (f64, u64) {
    let mut distance = 0.0;
    let mut item = 0u64;
    for index in 0..query_count {
        let query = query_f64(seed, index);
        let result = tree
            .query(black_box(&query))
            .nearest_one::<SquaredEuclidean<f64>>()
            .execute();
        distance += result.distance;
        item = item.wrapping_add(result.item as u64);
    }
    (distance, item)
}

fn run_generated_f32<SS: StemStrategy>(
    tree: &F32Tree<SS>,
    query_count: usize,
    seed: u64,
) -> (f32, u64) {
    let mut distance = 0.0;
    let mut item = 0u64;
    for index in 0..query_count {
        let query = query_f32(seed, index);
        let result = tree
            .query(black_box(&query))
            .nearest_one::<SquaredEuclidean<f32>>()
            .execute();
        distance += result.distance;
        item = item.wrapping_add(result.item as u64);
    }
    (distance, item)
}

fn run_generation_control_f64(query_count: usize, seed: u64) -> u64 {
    let mut checksum = 0u64;
    for index in 0..query_count {
        let query = black_box(query_f64(seed, index));
        checksum = checksum.wrapping_add(query[0].to_bits()).rotate_left(7)
            ^ query[1].to_bits()
            ^ query[2].to_bits();
    }
    checksum
}

fn run_generation_control_f32(query_count: usize, seed: u64) -> u64 {
    let mut checksum = 0u64;
    for index in 0..query_count {
        let query = black_box(query_f32(seed, index));
        checksum = checksum
            .wrapping_add(query[0].to_bits() as u64)
            .rotate_left(7)
            ^ query[1].to_bits() as u64
            ^ query[2].to_bits() as u64
            ^ query[3].to_bits() as u64;
    }
    checksum
}

fn bench_f64_strategy<SS: StemStrategy>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    label: &str,
    points: &[[f64; F64_K]],
    stored_queries: &[[f64; F64_K]],
    pool_sizes: &[usize],
    query_seed: u64,
) {
    let tree: F64Tree<SS> = KdTree::new_from_slice(points).unwrap();
    for &pool_size in pool_sizes {
        let queries = &stored_queries[..pool_size];
        group.throughput(Throughput::Elements(pool_size as u64));
        group.bench_function(
            BenchmarkId::new(format!("stored_{label}"), pool_size),
            |b| b.iter(|| black_box(run_stored_f64(&tree, queries))),
        );
    }
    for &pool_size in pool_sizes {
        group.throughput(Throughput::Elements(pool_size as u64));
        group.bench_function(
            BenchmarkId::new(format!("generated_{label}"), pool_size),
            |b| b.iter(|| black_box(run_generated_f64(&tree, pool_size, query_seed))),
        );
    }
}

fn bench_f32_strategy<SS: StemStrategy>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    label: &str,
    points: &[[f32; F32_K]],
    stored_queries: &[[f32; F32_K]],
    pool_sizes: &[usize],
    query_seed: u64,
) {
    let tree: F32Tree<SS> = KdTree::new_from_slice(points).unwrap();
    for &pool_size in pool_sizes {
        let queries = &stored_queries[..pool_size];
        group.throughput(Throughput::Elements(pool_size as u64));
        group.bench_function(
            BenchmarkId::new(format!("stored_{label}"), pool_size),
            |b| b.iter(|| black_box(run_stored_f32(&tree, queries))),
        );
    }
    for &pool_size in pool_sizes {
        group.throughput(Throughput::Elements(pool_size as u64));
        group.bench_function(
            BenchmarkId::new(format!("generated_{label}"), pool_size),
            |b| b.iter(|| black_box(run_generated_f32(&tree, pool_size, query_seed))),
        );
    }
}

fn query_pool(c: &mut Criterion) {
    validate_strategy_selection();
    let point_log2 = read_usize("KIDDO_QUERY_POOL_POINT_LOG2", DEFAULT_POINT_LOG2);
    assert!(point_log2 < usize::BITS as usize);
    let point_count = 1usize << point_log2;
    let point_seed = read_u64("KIDDO_PROFILE_POINT_SEED", POINT_SEED);
    let query_seed = read_u64("KIDDO_PROFILE_QUERY_SEED", QUERY_SEED);
    let pool_sizes = read_pool_sizes();
    let max_pool_size = *pool_sizes.last().unwrap();
    let axis = read_axis();

    eprintln!(
        "benchmarking exact query-pool crossover: f64_dimensions={F64_K} \
         f32_dimensions={F32_K} points=2^{point_log2} \
         pools={pool_sizes:?} axis={axis:?} point_seed={point_seed} query_seed={query_seed}"
    );

    if axis != AxisSelection::F32 {
        let points = build_points_f64(point_count, point_seed);
        let stored_queries: Vec<_> = (0..max_pool_size)
            .map(|index| query_f64(query_seed, index))
            .collect();
        assert!(stored_queries
            .iter()
            .enumerate()
            .all(|(index, query)| *query == query_f64(query_seed, index)));

        let mut group = c.benchmark_group(format!("profile_v6_query_pool/f64/{point_count}"));
        for &pool_size in &pool_sizes {
            group.throughput(Throughput::Elements(pool_size as u64));
            group.bench_function(BenchmarkId::new("generated_control", pool_size), |b| {
                b.iter(|| black_box(run_generation_control_f64(pool_size, query_seed)))
            });
        }
        if strategy_enabled("eytzinger") {
            bench_f64_strategy::<EytzingerComparator>(
                &mut group,
                "eytzinger",
                &points,
                &stored_queries,
                &pool_sizes,
                query_seed,
            );
        }
        if strategy_enabled("donnelly") {
            bench_f64_strategy::<Donnelly<3>>(
                &mut group,
                "donnelly",
                &points,
                &stored_queries,
                &pool_sizes,
                query_seed,
            );
        }
        if strategy_enabled("donnelly_unrolled") {
            bench_f64_strategy::<DonnellyUnrolled<3>>(
                &mut group,
                "donnelly_unrolled",
                &points,
                &stored_queries,
                &pool_sizes,
                query_seed,
            );
        }
        if strategy_enabled("donnelly_unrolled_block_dim") {
            bench_f64_strategy::<DonnellyUnrolledBlockDim<3>>(
                &mut group,
                "donnelly_unrolled_block_dim",
                &points,
                &stored_queries,
                &pool_sizes,
                query_seed,
            );
        }
        if strategy_enabled("donnelly_simd_descent") {
            bench_f64_strategy::<DonnellySimdDescent<3>>(
                &mut group,
                "donnelly_simd_descent",
                &points,
                &stored_queries,
                &pool_sizes,
                query_seed,
            );
        }
        if strategy_enabled("donnelly_cyclic_simd_descent") {
            bench_f64_strategy::<DonnellyCyclicSimdDescent<3>>(
                &mut group,
                "donnelly_cyclic_simd_descent",
                &points,
                &stored_queries,
                &pool_sizes,
                query_seed,
            );
        }
        if strategy_enabled("donnelly_simd_initial_descent") {
            bench_f64_strategy::<DonnellySimdInitialDescent<3>>(
                &mut group,
                "donnelly_simd_initial_descent",
                &points,
                &stored_queries,
                &pool_sizes,
                query_seed,
            );
        }
        if strategy_enabled("donnelly_simd_full") {
            bench_f64_strategy::<DonnellySimdFull<3>>(
                &mut group,
                "donnelly_simd_full",
                &points,
                &stored_queries,
                &pool_sizes,
                query_seed,
            );
        }
        group.finish();
    }

    if axis != AxisSelection::F64 {
        let points = build_points_f32(point_count, point_seed);
        let stored_queries: Vec<_> = (0..max_pool_size)
            .map(|index| query_f32(query_seed, index))
            .collect();
        assert!(stored_queries
            .iter()
            .enumerate()
            .all(|(index, query)| *query == query_f32(query_seed, index)));

        let mut group = c.benchmark_group(format!("profile_v6_query_pool/f32/{point_count}"));
        for &pool_size in &pool_sizes {
            group.throughput(Throughput::Elements(pool_size as u64));
            group.bench_function(BenchmarkId::new("generated_control", pool_size), |b| {
                b.iter(|| black_box(run_generation_control_f32(pool_size, query_seed)))
            });
        }
        if strategy_enabled("donnelly") {
            bench_f32_strategy::<Donnelly<4>>(
                &mut group,
                "donnelly",
                &points,
                &stored_queries,
                &pool_sizes,
                query_seed,
            );
        }
        if strategy_enabled("donnelly_unrolled") {
            bench_f32_strategy::<DonnellyUnrolled<4>>(
                &mut group,
                "donnelly_unrolled",
                &points,
                &stored_queries,
                &pool_sizes,
                query_seed,
            );
        }
        if strategy_enabled("donnelly_unrolled_block_dim") {
            bench_f32_strategy::<DonnellyUnrolledBlockDim<4>>(
                &mut group,
                "donnelly_unrolled_block_dim",
                &points,
                &stored_queries,
                &pool_sizes,
                query_seed,
            );
        }
        if strategy_enabled("donnelly_simd_descent") {
            bench_f32_strategy::<DonnellySimdDescent<4>>(
                &mut group,
                "donnelly_simd_descent",
                &points,
                &stored_queries,
                &pool_sizes,
                query_seed,
            );
        }
        if strategy_enabled("donnelly_cyclic_simd_descent") {
            bench_f32_strategy::<DonnellyCyclicSimdDescent<4>>(
                &mut group,
                "donnelly_cyclic_simd_descent",
                &points,
                &stored_queries,
                &pool_sizes,
                query_seed,
            );
        }
        if strategy_enabled("donnelly_simd_initial_descent") {
            bench_f32_strategy::<DonnellySimdInitialDescent<4>>(
                &mut group,
                "donnelly_simd_initial_descent",
                &points,
                &stored_queries,
                &pool_sizes,
                query_seed,
            );
        }
        if strategy_enabled("donnelly_simd_full") {
            bench_f32_strategy::<DonnellySimdFull<4>>(
                &mut group,
                "donnelly_simd_full",
                &points,
                &stored_queries,
                &pool_sizes,
                query_seed,
            );
        }
        if strategy_enabled("eytzinger") {
            bench_f32_strategy::<EytzingerComparator>(
                &mut group,
                "eytzinger",
                &points,
                &stored_queries,
                &pool_sizes,
                query_seed,
            );
        }
        group.finish();
    }
}

criterion_group!(benches, query_pool);
criterion_main!(benches);

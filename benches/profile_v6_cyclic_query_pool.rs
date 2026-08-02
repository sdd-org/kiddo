#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![cfg_attr(coverage_nightly, coverage(off))]

//! Exact-nearest-one query-pool benchmark for cyclic Donnelly strategies.
//!
//! This deliberately excludes the abandoned block-dimension strategies. It can
//! instantiate the native and awkward phase combinations: f64/K3, f64/K4,
//! f32/K4, and f32/K3.

use criterion::measurement::WallTime;
use criterion::{
    criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion, Throughput,
};
use kiddo::kd_tree::KdTree;
use kiddo::leaf_strategy::FlatVec;
use kiddo::{
    DonnellyCyclicSimdDescent, DonnellyCyclicSimdFull, DonnellyUnrolled, EytzingerFlexPf,
    SquaredEuclidean, StemStrategy,
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;

const B: usize = 32;
const DEFAULT_POINT_LOG2: usize = 23;
const DEFAULT_POOL_SIZES: [usize; 7] = [256, 512, 1_000, 2_048, 4_096, 8_192, 16_384];
const POINT_SEED: u64 = 0x5eed_0000_0000_0401;
const QUERY_SEED: u64 = 0x5eed_0000_0000_0402;
const SPLITMIX_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;
const SUPPORTED_STRATEGIES: [&str; 4] = [
    "eytzinger",
    "donnelly_unrolled",
    "donnelly_cyclic_simd_descent",
    "donnelly_cyclic_simd_full",
];

type EytzingerComparator = EytzingerFlexPf<0, -1>;

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

fn read_pool_sizes() -> Vec<usize> {
    let Ok(value) = std::env::var("KIDDO_CYCLIC_POOL_SIZES") else {
        return DEFAULT_POOL_SIZES.to_vec();
    };
    let mut sizes: Vec<usize> = value
        .split(',')
        .map(|entry| {
            entry
                .trim()
                .parse()
                .expect("KIDDO_CYCLIC_POOL_SIZES entries must be positive integers")
        })
        .collect();
    assert!(!sizes.is_empty());
    assert!(sizes.iter().all(|&size| size > 0));
    sizes.sort_unstable();
    sizes.dedup();
    sizes
}

fn strategies() -> Vec<String> {
    let value = std::env::var("KIDDO_CYCLIC_STRATEGIES").unwrap_or_else(|_| {
        "eytzinger,donnelly_unrolled,donnelly_cyclic_simd_descent,donnelly_cyclic_simd_full"
            .to_owned()
    });
    let requested: Vec<String> = value
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(str::to_owned)
        .collect();
    assert!(!requested.is_empty(), "no cyclic strategies selected");
    for strategy in &requested {
        assert!(
            SUPPORTED_STRATEGIES.contains(&strategy.as_str()),
            "unsupported KIDDO_CYCLIC_STRATEGIES entry: {strategy}"
        );
    }
    requested
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
fn query_f64<const K: usize>(seed: u64, query_index: usize) -> [f64; K] {
    const SCALE: f64 = 1.0 / ((1u64 << 53) as f64);
    std::array::from_fn(|dimension| {
        (query_word::<K>(seed, query_index, dimension) >> 11) as f64 * SCALE
    })
}

#[inline(always)]
fn query_f32<const K: usize>(seed: u64, query_index: usize) -> [f32; K] {
    const SCALE: f32 = 1.0 / ((1u32 << 24) as f32);
    std::array::from_fn(|dimension| {
        (query_word::<K>(seed, query_index, dimension) >> 40) as f32 * SCALE
    })
}

fn run_stored_f64<SS: StemStrategy, const K: usize>(
    tree: &KdTree<f64, u32, SS, FlatVec<f64, u32, K, B>, K, B>,
    queries: &[[f64; K]],
) -> (f64, u64) {
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

fn run_stored_f32<SS: StemStrategy, const K: usize>(
    tree: &KdTree<f32, u32, SS, FlatVec<f32, u32, K, B>, K, B>,
    queries: &[[f32; K]],
) -> (f32, u64) {
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

fn run_generated_f64<SS: StemStrategy, const K: usize>(
    tree: &KdTree<f64, u32, SS, FlatVec<f64, u32, K, B>, K, B>,
    count: usize,
    seed: u64,
) -> (f64, u64) {
    let mut distance = 0.0;
    let mut item = 0u64;
    for index in 0..count {
        let query = query_f64::<K>(seed, index);
        let result = tree
            .query(black_box(&query))
            .nearest_one::<SquaredEuclidean<f64>>()
            .execute();
        distance += result.distance;
        item = item.wrapping_add(result.item as u64);
    }
    (distance, item)
}

fn run_generated_f32<SS: StemStrategy, const K: usize>(
    tree: &KdTree<f32, u32, SS, FlatVec<f32, u32, K, B>, K, B>,
    count: usize,
    seed: u64,
) -> (f32, u64) {
    let mut distance = 0.0;
    let mut item = 0u64;
    for index in 0..count {
        let query = query_f32::<K>(seed, index);
        let result = tree
            .query(black_box(&query))
            .nearest_one::<SquaredEuclidean<f32>>()
            .execute();
        distance += result.distance;
        item = item.wrapping_add(result.item as u64);
    }
    (distance, item)
}

fn bench_f64_strategy<SS: StemStrategy, const K: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    label: &str,
    points: &[[f64; K]],
    queries: &[[f64; K]],
    pool_sizes: &[usize],
    seed: u64,
) {
    let tree: KdTree<f64, u32, SS, FlatVec<f64, u32, K, B>, K, B> =
        KdTree::new_from_slice(points).unwrap();
    for &pool_size in pool_sizes {
        group.throughput(Throughput::Elements(pool_size as u64));
        group.bench_function(
            BenchmarkId::new(format!("stored_{label}"), pool_size),
            |b| b.iter(|| black_box(run_stored_f64(&tree, &queries[..pool_size]))),
        );
        group.bench_function(
            BenchmarkId::new(format!("generated_{label}"), pool_size),
            |b| b.iter(|| black_box(run_generated_f64(&tree, pool_size, seed))),
        );
    }
}

fn bench_f32_strategy<SS: StemStrategy, const K: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    label: &str,
    points: &[[f32; K]],
    queries: &[[f32; K]],
    pool_sizes: &[usize],
    seed: u64,
) {
    let tree: KdTree<f32, u32, SS, FlatVec<f32, u32, K, B>, K, B> =
        KdTree::new_from_slice(points).unwrap();
    for &pool_size in pool_sizes {
        group.throughput(Throughput::Elements(pool_size as u64));
        group.bench_function(
            BenchmarkId::new(format!("stored_{label}"), pool_size),
            |b| b.iter(|| black_box(run_stored_f32(&tree, &queries[..pool_size]))),
        );
        group.bench_function(
            BenchmarkId::new(format!("generated_{label}"), pool_size),
            |b| b.iter(|| black_box(run_generated_f32(&tree, pool_size, seed))),
        );
    }
}

fn bench_f64<const K: usize>(
    c: &mut Criterion,
    point_count: usize,
    pool_sizes: &[usize],
    enabled: &[String],
    point_seed: u64,
    query_seed: u64,
) {
    let mut rng = ChaCha8Rng::seed_from_u64(point_seed);
    let points: Vec<[f64; K]> = (0..point_count)
        .map(|_| std::array::from_fn(|_| rng.random::<f64>()))
        .collect();
    let queries: Vec<_> = (0..*pool_sizes.last().unwrap())
        .map(|index| query_f64::<K>(query_seed, index))
        .collect();
    let mut group = c.benchmark_group(format!(
        "profile_v6_cyclic_query_pool/f64_k{K}/{point_count}"
    ));

    for strategy in enabled {
        match strategy.as_str() {
            "eytzinger" => bench_f64_strategy::<EytzingerComparator, K>(
                &mut group, strategy, &points, &queries, pool_sizes, query_seed,
            ),
            "donnelly_unrolled" => bench_f64_strategy::<DonnellyUnrolled<3>, K>(
                &mut group, strategy, &points, &queries, pool_sizes, query_seed,
            ),
            "donnelly_cyclic_simd_descent" => {
                bench_f64_strategy::<DonnellyCyclicSimdDescent<3>, K>(
                    &mut group, strategy, &points, &queries, pool_sizes, query_seed,
                )
            }
            "donnelly_cyclic_simd_full" => bench_f64_strategy::<DonnellyCyclicSimdFull<3>, K>(
                &mut group, strategy, &points, &queries, pool_sizes, query_seed,
            ),
            _ => unreachable!(),
        }
    }
    group.finish();
}

fn bench_f32<const K: usize>(
    c: &mut Criterion,
    point_count: usize,
    pool_sizes: &[usize],
    enabled: &[String],
    point_seed: u64,
    query_seed: u64,
) {
    let mut rng = ChaCha8Rng::seed_from_u64(point_seed);
    let points: Vec<[f32; K]> = (0..point_count)
        .map(|_| std::array::from_fn(|_| rng.random::<f32>()))
        .collect();
    let queries: Vec<_> = (0..*pool_sizes.last().unwrap())
        .map(|index| query_f32::<K>(query_seed, index))
        .collect();
    let mut group = c.benchmark_group(format!(
        "profile_v6_cyclic_query_pool/f32_k{K}/{point_count}"
    ));

    for strategy in enabled {
        match strategy.as_str() {
            "eytzinger" => bench_f32_strategy::<EytzingerComparator, K>(
                &mut group, strategy, &points, &queries, pool_sizes, query_seed,
            ),
            "donnelly_unrolled" => bench_f32_strategy::<DonnellyUnrolled<4>, K>(
                &mut group, strategy, &points, &queries, pool_sizes, query_seed,
            ),
            "donnelly_cyclic_simd_descent" => {
                bench_f32_strategy::<DonnellyCyclicSimdDescent<4>, K>(
                    &mut group, strategy, &points, &queries, pool_sizes, query_seed,
                )
            }
            "donnelly_cyclic_simd_full" => bench_f32_strategy::<DonnellyCyclicSimdFull<4>, K>(
                &mut group, strategy, &points, &queries, pool_sizes, query_seed,
            ),
            _ => unreachable!(),
        }
    }
    group.finish();
}

fn query_pool(c: &mut Criterion) {
    let axis = std::env::var("KIDDO_CYCLIC_AXIS").unwrap_or_else(|_| "f64".to_owned());
    let dimensions = read_usize("KIDDO_CYCLIC_DIMENSIONS", if axis == "f32" { 4 } else { 3 });
    let point_log2 = read_usize("KIDDO_CYCLIC_POINT_LOG2", DEFAULT_POINT_LOG2);
    assert!(point_log2 < usize::BITS as usize);
    let point_count = 1usize << point_log2;
    let pool_sizes = read_pool_sizes();
    let enabled = strategies();
    let point_seed = read_u64("KIDDO_PROFILE_POINT_SEED", POINT_SEED);
    let query_seed = read_u64("KIDDO_PROFILE_QUERY_SEED", QUERY_SEED);

    eprintln!(
        "cyclic exact-NN: axis={axis} dimensions={dimensions} points=2^{point_log2} \
         pools={pool_sizes:?} strategies={enabled:?}"
    );

    match (axis.as_str(), dimensions) {
        ("f64", 3) => bench_f64::<3>(
            c,
            point_count,
            &pool_sizes,
            &enabled,
            point_seed,
            query_seed,
        ),
        ("f64", 4) => bench_f64::<4>(
            c,
            point_count,
            &pool_sizes,
            &enabled,
            point_seed,
            query_seed,
        ),
        ("f32", 3) => bench_f32::<3>(
            c,
            point_count,
            &pool_sizes,
            &enabled,
            point_seed,
            query_seed,
        ),
        ("f32", 4) => bench_f32::<4>(
            c,
            point_count,
            &pool_sizes,
            &enabled,
            point_seed,
            query_seed,
        ),
        _ => panic!("supported combinations are f64/f32 with 3 or 4 dimensions"),
    }
}

criterion_group!(benches, query_pool);
criterion_main!(benches);

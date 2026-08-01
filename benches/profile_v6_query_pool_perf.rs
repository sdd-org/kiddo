#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![cfg_attr(coverage_nightly, coverage(off))]

//! Fixed-work exact-NN query-pool runner for `perf stat` and structural counts.
//!
//! Tree construction and warmup happen before an optional SIGSTOP.  The helper
//! script attaches perf to the stopped process and resumes it, so construction
//! is excluded from the counters without relying on a guessed delay.

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
use std::io::Write;
use std::time::Instant;

const F64_K: usize = 3;
const F32_K: usize = 4;
const B: usize = 32;
const DEFAULT_POINT_LOG2: usize = 27;
const DEFAULT_POOL_SIZE: usize = 2_048;
const DEFAULT_TOTAL_QUERIES: usize = 4_000_000;
const POINT_SEED: u64 = 0x5eed_0000_0000_0301;
const QUERY_SEED: u64 = 0x5eed_0000_0000_0302;
const SPLITMIX_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

type F64Tree<SS> = KdTree<f64, u32, SS, FlatVec<f64, u32, F64_K, B>, F64_K, B>;
type F32Tree<SS> = KdTree<f32, u32, SS, FlatVec<f32, u32, F32_K, B>, F32_K, B>;
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

fn run_f64<SS: StemStrategy>(
    tree: &F64Tree<SS>,
    queries: &[[f64; F64_K]],
    repeats: usize,
) -> (f64, u64) {
    let mut distance = 0.0;
    let mut item = 0u64;
    for _ in 0..repeats {
        for query in queries {
            let result = tree
                .query(black_box(query))
                .nearest_one::<SquaredEuclidean<f64>>()
                .execute();
            distance += result.distance;
            item = item.wrapping_add(result.item as u64);
        }
    }
    black_box((distance, item))
}

fn run_f32<SS: StemStrategy>(
    tree: &F32Tree<SS>,
    queries: &[[f32; F32_K]],
    repeats: usize,
) -> (f32, u64) {
    let mut distance = 0.0;
    let mut item = 0u64;
    for _ in 0..repeats {
        for query in queries {
            let result = tree
                .query(black_box(query))
                .nearest_one::<SquaredEuclidean<f32>>()
                .execute();
            distance += result.distance;
            item = item.wrapping_add(result.item as u64);
        }
    }
    black_box((distance, item))
}

#[cfg(target_family = "unix")]
fn stop_for_profiler_if_requested() {
    if std::env::var("KIDDO_PERF_STOP").as_deref() != Ok("1") {
        return;
    }
    println!("READY pid={}", std::process::id());
    std::io::stdout().flush().unwrap();
    unsafe extern "C" {
        fn raise(signal: i32) -> i32;
    }
    const SIGSTOP: i32 = 19;
    assert_eq!(unsafe { raise(SIGSTOP) }, 0);
}

#[cfg(not(target_family = "unix"))]
fn stop_for_profiler_if_requested() {}

#[cfg(feature = "exact_query_stats")]
fn reset_stats() {
    kiddo::results::exact_query_stats::reset();
}

#[cfg(not(feature = "exact_query_stats"))]
fn reset_stats() {}

#[cfg(feature = "exact_query_stats")]
fn print_stats() {
    let stats = kiddo::results::exact_query_stats::snapshot();
    println!(
        "STATS queries={} leaf_visits={} stem_steps={} real_pivot_steps={} padding_steps={} \
         initial_far_candidates={} initial_far_rejects={} continuation_frames_pushed={} \
         continuation_frames_popped={} far_rechecks={} far_enters={} \
         far_rejects_after_near={} scalar_stack_pops={}",
        stats.queries,
        stats.leaf_visits,
        stats.stem_steps,
        stats.real_pivot_steps,
        stats.padding_steps,
        stats.initial_far_candidates,
        stats.initial_far_rejects,
        stats.continuation_frames_pushed,
        stats.continuation_frames_popped,
        stats.far_rechecks,
        stats.far_enters,
        stats.far_rejects_after_near,
        stats.scalar_stack_pops,
    );
    let queries = stats.queries.max(1) as f64;
    println!(
        "STATS_PER_QUERY leaf_visits={:.6} stem_steps={:.6} real_pivot_steps={:.6} \
         padding_steps={:.6} initial_far_candidates={:.6} initial_far_rejects={:.6} \
         continuation_frames_pushed={:.6} continuation_frames_popped={:.6} \
         far_rechecks={:.6} far_enters={:.6} far_rejects_after_near={:.6}",
        stats.leaf_visits as f64 / queries,
        stats.stem_steps as f64 / queries,
        stats.real_pivot_steps as f64 / queries,
        stats.padding_steps as f64 / queries,
        stats.initial_far_candidates as f64 / queries,
        stats.initial_far_rejects as f64 / queries,
        stats.continuation_frames_pushed as f64 / queries,
        stats.continuation_frames_popped as f64 / queries,
        stats.far_rechecks as f64 / queries,
        stats.far_enters as f64 / queries,
        stats.far_rejects_after_near as f64 / queries,
    );
}

#[cfg(not(feature = "exact_query_stats"))]
fn print_stats() {}

fn execute_f64<SS: StemStrategy>(
    points: &[[f64; F64_K]],
    queries: &[[f64; F64_K]],
    warmup_repeats: usize,
    repeats: usize,
) -> (u128, f64, u64) {
    let tree: F64Tree<SS> = KdTree::new_from_slice(points).unwrap();
    black_box(run_f64(&tree, queries, warmup_repeats));
    stop_for_profiler_if_requested();
    reset_stats();
    let start = Instant::now();
    let (distance, item) = run_f64(&tree, queries, repeats);
    (start.elapsed().as_nanos(), distance, item)
}

fn execute_f32<SS: StemStrategy>(
    points: &[[f32; F32_K]],
    queries: &[[f32; F32_K]],
    warmup_repeats: usize,
    repeats: usize,
) -> (u128, f32, u64) {
    let tree: F32Tree<SS> = KdTree::new_from_slice(points).unwrap();
    black_box(run_f32(&tree, queries, warmup_repeats));
    stop_for_profiler_if_requested();
    reset_stats();
    let start = Instant::now();
    let (distance, item) = run_f32(&tree, queries, repeats);
    (start.elapsed().as_nanos(), distance, item)
}

fn main() {
    let axis = std::env::var("KIDDO_PERF_AXIS").unwrap_or_else(|_| "f64".to_owned());
    let strategy = std::env::var("KIDDO_PERF_STRATEGY").unwrap_or_else(|_| "eytzinger".to_owned());
    let point_log2 = read_usize("KIDDO_PERF_POINT_LOG2", DEFAULT_POINT_LOG2);
    let point_count = 1usize << point_log2;
    let pool_size = read_usize("KIDDO_PERF_POOL_SIZE", DEFAULT_POOL_SIZE);
    let total_query_target = read_usize("KIDDO_PERF_TOTAL_QUERIES", DEFAULT_TOTAL_QUERIES);
    let warmup_repeats = read_usize("KIDDO_PERF_WARMUP_REPEATS", 2);
    let point_seed = read_u64("KIDDO_PROFILE_POINT_SEED", POINT_SEED);
    let query_seed = read_u64("KIDDO_PROFILE_QUERY_SEED", QUERY_SEED);
    let repeats = total_query_target.div_ceil(pool_size).max(1);
    let total_queries = repeats * pool_size;

    eprintln!(
        "building exact query-pool perf case: axis={axis} strategy={strategy} \
         points=2^{point_log2} pool={pool_size} repeats={repeats} total_queries={total_queries}"
    );

    let (elapsed_ns, distance, item) = match axis.as_str() {
        "f64" => {
            let points = build_points_f64(point_count, point_seed);
            let queries: Vec<_> = (0..pool_size).map(|i| query_f64(query_seed, i)).collect();
            let (elapsed, distance, item) = match strategy.as_str() {
                "eytzinger" => {
                    execute_f64::<EytzingerComparator>(&points, &queries, warmup_repeats, repeats)
                }
                "donnelly" => {
                    execute_f64::<Donnelly<3>>(&points, &queries, warmup_repeats, repeats)
                }
                "donnelly_unrolled" => {
                    execute_f64::<DonnellyUnrolled<3>>(&points, &queries, warmup_repeats, repeats)
                }
                "donnelly_unrolled_block_dim" => execute_f64::<DonnellyUnrolledBlockDim<3>>(
                    &points,
                    &queries,
                    warmup_repeats,
                    repeats,
                ),
                "donnelly_simd_descent" => execute_f64::<DonnellySimdDescent<3>>(
                    &points,
                    &queries,
                    warmup_repeats,
                    repeats,
                ),
                "donnelly_cyclic_simd_descent" => execute_f64::<DonnellyCyclicSimdDescent<3>>(
                    &points,
                    &queries,
                    warmup_repeats,
                    repeats,
                ),
                "donnelly_simd_initial_descent" => execute_f64::<DonnellySimdInitialDescent<3>>(
                    &points,
                    &queries,
                    warmup_repeats,
                    repeats,
                ),
                "donnelly_simd_full" => {
                    execute_f64::<DonnellySimdFull<3>>(&points, &queries, warmup_repeats, repeats)
                }
                _ => panic!("unsupported KIDDO_PERF_STRATEGY"),
            };
            (elapsed, distance, item)
        }
        "f32" => {
            let points = build_points_f32(point_count, point_seed);
            let queries: Vec<_> = (0..pool_size).map(|i| query_f32(query_seed, i)).collect();
            let (elapsed, distance, item) = match strategy.as_str() {
                "eytzinger" => {
                    execute_f32::<EytzingerComparator>(&points, &queries, warmup_repeats, repeats)
                }
                "donnelly" => {
                    execute_f32::<Donnelly<4>>(&points, &queries, warmup_repeats, repeats)
                }
                "donnelly_unrolled" => {
                    execute_f32::<DonnellyUnrolled<4>>(&points, &queries, warmup_repeats, repeats)
                }
                "donnelly_unrolled_block_dim" => execute_f32::<DonnellyUnrolledBlockDim<4>>(
                    &points,
                    &queries,
                    warmup_repeats,
                    repeats,
                ),
                "donnelly_simd_descent" => execute_f32::<DonnellySimdDescent<4>>(
                    &points,
                    &queries,
                    warmup_repeats,
                    repeats,
                ),
                "donnelly_cyclic_simd_descent" => execute_f32::<DonnellyCyclicSimdDescent<4>>(
                    &points,
                    &queries,
                    warmup_repeats,
                    repeats,
                ),
                "donnelly_simd_initial_descent" => execute_f32::<DonnellySimdInitialDescent<4>>(
                    &points,
                    &queries,
                    warmup_repeats,
                    repeats,
                ),
                "donnelly_simd_full" => {
                    execute_f32::<DonnellySimdFull<4>>(&points, &queries, warmup_repeats, repeats)
                }
                _ => panic!("unsupported KIDDO_PERF_STRATEGY"),
            };
            (elapsed, distance as f64, item)
        }
        _ => panic!("KIDDO_PERF_AXIS must be f64 or f32"),
    };

    println!(
        "RESULT axis={axis} strategy={strategy} point_log2={point_log2} pool={pool_size} \
         queries={total_queries} elapsed_ns={elapsed_ns} ns_per_query={:.6} \
         checksum_distance={distance:.17e} checksum_item={item}",
        elapsed_ns as f64 / total_queries as f64,
    );
    print_stats();
}

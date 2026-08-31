#![allow(clippy::too_many_arguments)]

//! Long-running v6 correctness and approximate-quality fuzz-style tests.
//!
//! Run the complete non-SIMD and SIMD matrix locally with
//! `just fuzz-kd-tree-v6`. The generated sweeps are ignored so ordinary test
//! runs stay fast; the compact adversarial checks run whenever this test target
//! is selected. The just task and manual CI workflow use `--include-ignored`
//! deliberately.

use std::any::TypeId;
use std::array;
use std::collections::{BinaryHeap, HashSet};
use std::env;
use std::fs::OpenOptions;
use std::io::Write;
use std::num::NonZeroUsize;
use std::sync::{Mutex, OnceLock};

use kiddo::kd_tree::KdTree;
use kiddo::leaf_strategy::{FlatVec, VecOfArenas, VecOfArrays};
use kiddo::stem_strategy::{Donnelly, Eytzinger};
use kiddo::traits::leaf_strategy::ConstructibleLeafStrategy;
use kiddo::{dist::DistanceMetricScalar, Chebyshev, Manhattan, SquaredEuclidean};
use kiddo::{Axis, LeafStrategy, QueryResultItem, StemStrategy};

#[cfg(feature = "simd")]
use kiddo::stem_strategy::DonnellySimdFull;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

const DEFAULT_CASES: usize = 8;
const DEFAULT_MIN_POW: u32 = 10;
const DEFAULT_MAX_POW: u32 = 20;
const DEFAULT_PERTURB_MIN: i32 = -5;
const DEFAULT_PERTURB_MAX: i32 = 5;
const DEFAULT_MAX_NEAREST_N: usize = 32;
const DEFAULT_SEED: u64 = 0x4b1d_f00d;
const DEFAULT_QUERY_COUNT: usize = 100;
#[cfg(feature = "simd")]
const SIMD_FAST_CASES: usize = 1;
#[cfg(feature = "simd")]
const SIMD_FAST_QUERY_COUNT: usize = 10;
const PROGRESS_EVERY: usize = 100;
const PREVIEW_LEN: usize = 8;
const REPORT_PATH: &str = "kd_tree_fuzz_v6_report.txt";
const SEED_MIX_CASE: u64 = 0x9e37_79b9_7f4a_7c15;
const SEED_MIX_QUERY: u64 = 0xbf58_476d_1ce4_e5b9;

static REPORT_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
static FUZZ_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

#[derive(Clone, Copy)]
struct FuzzConfig {
    seed: u64,
    cases: usize,
    query_count: usize,
    min_pow: u32,
    max_pow: u32,
    perturb_min: i32,
    perturb_max: i32,
    max_nearest_n: usize,
}

#[derive(Clone, Copy)]
struct ReproMeta {
    kind: &'static str,
    leaf: &'static str,
    scalar: &'static str,
    strategy: &'static str,
    b: usize,
    k: usize,
}

impl FuzzConfig {
    fn from_env() -> Self {
        Self {
            seed: read_env_u64("KIDDO_FUZZ_SEED", DEFAULT_SEED),
            cases: read_env_usize("KIDDO_FUZZ_CASES", DEFAULT_CASES),
            query_count: read_env_usize("KIDDO_FUZZ_QUERY_COUNT", DEFAULT_QUERY_COUNT).max(1),
            min_pow: read_env_u32("KIDDO_FUZZ_MIN_POW", DEFAULT_MIN_POW),
            max_pow: read_env_u32("KIDDO_FUZZ_MAX_POW", DEFAULT_MAX_POW),
            perturb_min: read_env_i32("KIDDO_FUZZ_PERTURB_MIN", DEFAULT_PERTURB_MIN),
            perturb_max: read_env_i32("KIDDO_FUZZ_PERTURB_MAX", DEFAULT_PERTURB_MAX),
            max_nearest_n: read_env_usize("KIDDO_FUZZ_MAX_NEAREST_N", DEFAULT_MAX_NEAREST_N),
        }
    }

    fn case_seed(self, case_idx: usize) -> u64 {
        self.seed ^ (case_idx as u64).wrapping_mul(SEED_MIX_CASE)
    }

    #[cfg(feature = "simd")]
    fn for_simd(self) -> Self {
        if read_env_bool("KIDDO_FUZZ_V6_SIMD_FAST", false) {
            Self {
                cases: self.cases.clamp(1, SIMD_FAST_CASES),
                query_count: self.query_count.clamp(1, SIMD_FAST_QUERY_COUNT),
                ..self
            }
        } else {
            self
        }
    }
}

fn read_env_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn read_env_u32(key: &str, default: u32) -> u32 {
    env::var(key)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(default)
}

fn read_env_i32(key: &str, default: i32) -> i32 {
    env::var(key)
        .ok()
        .and_then(|value| value.parse::<i32>().ok())
        .unwrap_or(default)
}

fn read_env_u64(key: &str, default: u64) -> u64 {
    env::var(key)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default)
}

fn read_env_bool(key: &str, default: bool) -> bool {
    env::var(key)
        .ok()
        .map(|value| match value.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => default,
        })
        .unwrap_or(default)
}

fn fuzz_pool() -> &'static rayon::ThreadPool {
    FUZZ_POOL.get_or_init(|| {
        let available = std::thread::available_parallelism()
            .map(NonZeroUsize::get)
            .unwrap_or(1);
        let threads = read_env_usize("KIDDO_FUZZ_THREADS", available.min(4))
            .max(1)
            .min(available);

        println!("v6 fuzz query worker threads: {threads}");
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|idx| format!("kiddo-fuzz-{idx}"))
            .build()
            .expect("failed to build v6 fuzz worker pool")
    })
}

fn should_run_non_simd_paths() -> bool {
    read_env_bool("KIDDO_FUZZ_V6_RUN_NON_SIMD", true)
}

#[cfg(feature = "simd")]
fn should_run_simd_paths() -> bool {
    read_env_bool("KIDDO_FUZZ_V6_RUN_SIMD", true)
}

#[cfg(feature = "simd")]
fn should_run_simd_f32_paths() -> bool {
    should_run_simd_paths() && read_env_bool("KIDDO_FUZZ_V6_RUN_SIMD_F32", true)
}

fn log_failure(message: &str) {
    let lock = REPORT_LOCK.get_or_init(|| Mutex::new(()));
    let _guard = lock.lock().ok();

    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(REPORT_PATH)
    {
        let _ = writeln!(file, "{message}");
    }
}

fn log_and_panic(message: String) -> ! {
    log_failure(&message);
    panic!("{message}");
}

fn build_repro_id(
    meta: ReproMeta,
    point_count: usize,
    content_seed: u64,
    query_seed: u64,
) -> String {
    format!(
        "failure-kind_{}-leaf_{}-ty_{}-strategy_{}-b_{}-k_{}-size_{}-content_seed_{}-query_seed_{}",
        meta.kind,
        meta.leaf,
        meta.scalar,
        meta.strategy,
        meta.b,
        meta.k,
        point_count,
        content_seed,
        query_seed
    )
}

fn format_preview<A: std::fmt::Debug>(items: &[(A, usize)], limit: usize) -> String {
    let len = items.len();
    if len <= limit {
        format!("{items:?}")
    } else {
        format!("{:?} ... (len={len})", &items[..limit])
    }
}

fn query_seed(content_seed: u64, query_idx: usize) -> u64 {
    content_seed.wrapping_add(SEED_MIX_QUERY) ^ (query_idx as u64).wrapping_mul(SEED_MIX_CASE)
}

fn random_point_f32<const K: usize>(rng: &mut StdRng) -> [f32; K] {
    array::from_fn(|_| rng.random_range(-1.0f32..1.0f32))
}

fn random_point_f64<const K: usize>(rng: &mut StdRng) -> [f64; K] {
    array::from_fn(|_| rng.random_range(-1.0f64..1.0f64))
}

fn random_point_count(cfg: FuzzConfig, rng: &mut StdRng) -> usize {
    let min_pow = cfg.min_pow.min(cfg.max_pow);
    let max_pow = cfg.min_pow.max(cfg.max_pow);
    let exp = rng.random_range(min_pow..=max_pow);
    let base = 1usize << exp;
    let perturb_min = cfg.perturb_min.min(cfg.perturb_max);
    let perturb_max = cfg.perturb_min.max(cfg.perturb_max);
    let perturb = rng.random_range(perturb_min..=perturb_max);

    if perturb < 0 {
        base.saturating_sub((-perturb) as usize).max(1)
    } else {
        base + perturb as usize
    }
}

fn random_radius_f32<const K: usize>(rng: &mut StdRng) -> f32 {
    rng.random_range(0.0f32..(0.5f32 * K as f32))
}

fn random_radius_f64<const K: usize>(rng: &mut StdRng) -> f64 {
    rng.random_range(0.0f64..(0.5f64 * K as f64))
}

fn sort_by_distance_then_index<A: Axis<Coord = A> + PartialOrd>(items: &mut [(A, usize)]) {
    items.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .expect("NaN distance in sort")
            .then_with(|| a.1.cmp(&b.1))
    });
}

fn distances_match_for_fuzz<A: Copy + PartialEq + 'static>(expected: A, got: A) -> bool {
    if expected == got {
        return true;
    }

    if TypeId::of::<A>() == TypeId::of::<f32>() {
        let expected = unsafe { *(&expected as *const A as *const f32) };
        let got = unsafe { *(&got as *const A as *const f32) };
        return expected.is_finite()
            && got.is_finite()
            && expected.to_bits().abs_diff(got.to_bits()) <= 4;
    }

    if TypeId::of::<A>() == TypeId::of::<f64>() {
        let expected = unsafe { *(&expected as *const A as *const f64) };
        let got = unsafe { *(&got as *const A as *const f64) };
        return expected.is_finite()
            && got.is_finite()
            && expected.to_bits().abs_diff(got.to_bits()) <= 4;
    }

    false
}

fn compare_nearest_n_sorted<
    A: Axis<Coord = A> + PartialOrd + PartialEq + std::fmt::Debug + 'static,
>(
    expected: &[(A, usize)],
    got: &[(A, usize)],
) -> Result<(), String> {
    if expected.len() != got.len() {
        return Err(format!(
            "len mismatch expected={} got={}",
            expected.len(),
            got.len()
        ));
    }
    if expected.is_empty() {
        return Ok(());
    }

    // The tree kernels may legitimately compute distances a few ULP
    // differently from the brute-force reference (wider intermediates, SIMD
    // reduction order), so every pair of corresponding entries must agree to
    // within the fuzz tolerance. Anything further apart is a real difference,
    // not float noise.
    for (idx, ((exp_dist, _), (got_dist, _))) in expected.iter().zip(got).enumerate() {
        if !distances_match_for_fuzz(*exp_dist, *got_dist) {
            return Err(format!(
                "distance mismatch at index {idx} expected={exp_dist:?} got={got_dist:?}"
            ));
        }
    }

    // Within a run of mutually tolerable distances the tree may return the
    // tied items in any order, and at the max-qty boundary it may pick any
    // subset of the tie, so tie runs are compared as item sets. A run only
    // ends where BOTH sides agree the neighbouring distances genuinely differ:
    // if either side considers them within tolerance, they are a tie. The
    // final run (containing the tail) is exempt from the item-set check
    // because which members of the tail tie make the cut is
    // implementation-defined.
    let mut start = 0usize;
    while start < expected.len() {
        let mut end = start + 1;
        while end < expected.len()
            && (distances_match_for_fuzz(expected[end - 1].0, expected[end].0)
                || distances_match_for_fuzz(got[end - 1].0, got[end].0))
        {
            end += 1;
        }

        if end < expected.len() {
            let mut exp_items: Vec<usize> =
                expected[start..end].iter().map(|(_, item)| *item).collect();
            let mut got_items: Vec<usize> = got[start..end].iter().map(|(_, item)| *item).collect();
            exp_items.sort_unstable();
            got_items.sort_unstable();
            if exp_items != got_items {
                return Err(format!(
                    "tie mismatch at dist {:?} expected_items={exp_items:?} got_items={got_items:?}",
                    expected[start].0
                ));
            }
        }

        start = end;
    }

    Ok(())
}

fn compare_item_sorted_results<
    A: Axis<Coord = A> + PartialEq + std::fmt::Debug + std::fmt::Display + 'static,
>(
    expected: &[(A, usize)],
    got: &[(A, usize)],
) -> Result<(), String> {
    if expected.len() != got.len() {
        return Err(format!(
            "len mismatch expected={} got={} {}",
            expected.len(),
            got.len(),
            format_item_delta(expected, got)
        ));
    }

    for (idx, (expected_entry, got_entry)) in expected.iter().zip(got.iter()).enumerate() {
        if expected_entry.1 != got_entry.1 {
            return Err(format!(
                "item mismatch at idx={idx} expected_item={} got_item={}",
                expected_entry.1, got_entry.1
            ));
        }
        if !distances_match_for_fuzz(expected_entry.0, got_entry.0) {
            return Err(format!(
                "distance mismatch at idx={idx} expected={} got={}",
                expected_entry.0, got_entry.0
            ));
        }
    }

    Ok(())
}

fn within_boundary_matches_for_fuzz<A: Copy + PartialEq + 'static>(dist: A, radius: A) -> bool {
    distances_match_for_fuzz(dist, radius)
}

fn compare_within_results<
    A: Axis<Coord = A> + PartialEq + std::fmt::Debug + std::fmt::Display + 'static,
>(
    expected: &mut [(A, usize)],
    got: &mut [(A, usize)],
    radius: A,
) -> Result<(), String> {
    sort_by_item_idx(expected);
    sort_by_item_idx(got);

    let mut i = 0usize;
    let mut j = 0usize;

    while i < expected.len() || j < got.len() {
        match (expected.get(i), got.get(j)) {
            (Some((expected_dist, expected_item)), Some((got_dist, got_item))) => {
                match expected_item.cmp(got_item) {
                    std::cmp::Ordering::Equal => {
                        if !distances_match_for_fuzz(*expected_dist, *got_dist) {
                            return Err(format!(
                                "distance mismatch at item={} expected={} got={}",
                                expected_item, expected_dist, got_dist
                            ));
                        }
                        i += 1;
                        j += 1;
                    }
                    std::cmp::Ordering::Less => {
                        if !within_boundary_matches_for_fuzz(*expected_dist, radius) {
                            return Err(format!(
                                "len mismatch expected={} got={} {}",
                                expected.len(),
                                got.len(),
                                format_item_delta(expected, got)
                            ));
                        }
                        i += 1;
                    }
                    std::cmp::Ordering::Greater => {
                        if !within_boundary_matches_for_fuzz(*got_dist, radius) {
                            return Err(format!(
                                "len mismatch expected={} got={} {}",
                                expected.len(),
                                got.len(),
                                format_item_delta(expected, got)
                            ));
                        }
                        j += 1;
                    }
                }
            }
            (Some((expected_dist, _)), None) => {
                if !within_boundary_matches_for_fuzz(*expected_dist, radius) {
                    return Err(format!(
                        "len mismatch expected={} got={} {}",
                        expected.len(),
                        got.len(),
                        format_item_delta(expected, got)
                    ));
                }
                i += 1;
            }
            (None, Some((got_dist, _))) => {
                if !within_boundary_matches_for_fuzz(*got_dist, radius) {
                    return Err(format!(
                        "len mismatch expected={} got={} {}",
                        expected.len(),
                        got.len(),
                        format_item_delta(expected, got)
                    ));
                }
                j += 1;
            }
            (None, None) => break,
        }
    }

    Ok(())
}

fn format_item_delta<A>(expected: &[(A, usize)], got: &[(A, usize)]) -> String {
    const SAMPLE: usize = 8;

    let got_items: HashSet<usize> = got.iter().map(|entry| entry.1).collect();
    let expected_items: HashSet<usize> = expected.iter().map(|entry| entry.1).collect();

    let mut missing: Vec<usize> = expected_items.difference(&got_items).copied().collect();
    let mut extra: Vec<usize> = got_items.difference(&expected_items).copied().collect();
    missing.sort_unstable();
    extra.sort_unstable();

    format!(
        "missing_item_count={} missing_item_sample={:?} extra_item_count={} extra_item_sample={:?}",
        missing.len(),
        &missing[..missing.len().min(SAMPLE)],
        extra.len(),
        &extra[..extra.len().min(SAMPLE)]
    )
}

fn format_missing_leaf_sample<
    A: Axis<Coord = A>,
    SS: StemStrategy,
    LS: LeafStrategy<A, usize, SS, K, B>,
    const K: usize,
    const B: usize,
>(
    tree: &KdTree<A, usize, SS, LS, K, B>,
    expected: &[(A, usize)],
    got: &[(A, usize)],
) -> String {
    const SAMPLE: usize = 8;

    let got_items: HashSet<usize> = got.iter().map(|entry| entry.1).collect();
    let mut missing: Vec<usize> = expected
        .iter()
        .map(|entry| entry.1)
        .filter(|item| !got_items.contains(item))
        .collect();
    missing.sort_unstable();
    missing.dedup();

    let sample: Vec<_> = missing
        .into_iter()
        .take(SAMPLE)
        .map(|item| (item, tree.find_leaf_for_item(item)))
        .collect();

    format!("missing_leaf_sample={sample:?}")
}

macro_rules! assert_distance_match_for_fuzz {
    ($expected:expr, $got:expr, $($arg:tt)+) => {{
        let expected_value = $expected;
        let got_value = $got;
        assert!(
            distances_match_for_fuzz(expected_value, got_value),
            $($arg)+
        );
    }};
}

fn format_context<A: std::fmt::Display>(
    label: &str,
    case_idx: usize,
    query_idx: usize,
    _content_seed: u64,
    _query_seed: u64,
    point_count: usize,
    max_qty: usize,
    radius_sq: A,
    radius_man: A,
) -> String {
    format!(
        "label={label} case={} query={} points={} max_qty={} radius_sq={} radius_man={}",
        case_idx + 1,
        query_idx + 1,
        point_count,
        max_qty,
        radius_sq,
        radius_man
    )
}

fn log_mismatch<A: std::fmt::Display>(
    meta: ReproMeta,
    label: &str,
    case_idx: usize,
    query_idx: usize,
    content_seed: u64,
    query_seed: u64,
    point_count: usize,
    max_qty: usize,
    radius_sq: A,
    radius_man: A,
    detail: String,
) -> ! {
    let repro = build_repro_id(meta, point_count, content_seed, query_seed);
    let context = format_context(
        label,
        case_idx,
        query_idx,
        content_seed,
        query_seed,
        point_count,
        max_qty,
        radius_sq,
        radius_man,
    );
    log_and_panic(format!("repro={repro} {context} {detail}"));
}

struct MetricState<A: Axis<Coord = A> + PartialOrd> {
    best_dist: A,
    best_items: Vec<usize>,
    heap: BinaryHeap<QueryResultItem<(), usize, A>>,
    within: Vec<(A, usize)>,
    max_qty: usize,
    radius: A,
}

impl<A: Axis<Coord = A> + PartialOrd> MetricState<A> {
    fn new(max_qty: usize, radius: A) -> Self {
        Self {
            best_dist: A::max_value(),
            best_items: Vec::new(),
            heap: BinaryHeap::with_capacity(max_qty + 1),
            within: Vec::new(),
            max_qty,
            radius,
        }
    }

    fn update(&mut self, dist: A, idx: usize) {
        if dist < self.best_dist {
            self.best_dist = dist;
            self.best_items.clear();
            self.best_items.push(idx);
        } else if dist == self.best_dist {
            self.best_items.push(idx);
        }

        if dist <= self.radius {
            self.within.push((dist, idx));
        }

        if self.max_qty == 0 {
            return;
        }

        if self.heap.len() < self.max_qty {
            self.heap.push(QueryResultItem {
                point: (),
                distance: dist,
                item: idx,
            });
        } else if let Some(top) = self.heap.peek() {
            if dist < top.distance {
                self.heap.pop();
                self.heap.push(QueryResultItem {
                    point: (),
                    distance: dist,
                    item: idx,
                });
            }
        }
    }

    fn take_nearest_n_sorted(&mut self) -> Vec<(A, usize)> {
        let heap = std::mem::take(&mut self.heap);
        heap.into_sorted_vec()
            .into_iter()
            .map(|n| (n.distance, n.item))
            .collect()
    }

    fn take_within_sorted(&mut self) -> Vec<(A, usize)> {
        let mut items = std::mem::take(&mut self.within);
        sort_by_distance_then_index(&mut items);
        items
    }
}

fn brute_states_f32<const K: usize>(
    points: &[[f32; K]],
    query: &[f32; K],
    max_qty: usize,
    radius_sq: f32,
    radius_manhattan: f32,
) -> (MetricState<f32>, MetricState<f32>) {
    let mut sq = MetricState::new(max_qty, radius_sq);
    let mut man = MetricState::new(max_qty, radius_manhattan);

    for (idx, point) in points.iter().enumerate() {
        let dist_sq = <SquaredEuclidean<f32> as DistanceMetricScalar<f32>>::dist_raw(query, point);
        let dist_man = <Manhattan<f32> as DistanceMetricScalar<f32>>::dist_raw(query, point);
        sq.update(dist_sq, idx);
        man.update(dist_man, idx);
    }

    (sq, man)
}

fn brute_states_f64<const K: usize>(
    points: &[[f64; K]],
    query: &[f64; K],
    max_qty: usize,
    radius_sq: f64,
    radius_manhattan: f64,
) -> (MetricState<f64>, MetricState<f64>) {
    let mut sq = MetricState::new(max_qty, radius_sq);
    let mut man = MetricState::new(max_qty, radius_manhattan);

    for (idx, point) in points.iter().enumerate() {
        let dist_sq = <SquaredEuclidean<f64> as DistanceMetricScalar<f64>>::dist_raw(query, point);
        let dist_man = <Manhattan<f64> as DistanceMetricScalar<f64>>::dist_raw(query, point);
        sq.update(dist_sq, idx);
        man.update(dist_man, idx);
    }

    (sq, man)
}

fn assert_nearest_one<A: Axis<Coord = A> + PartialEq + std::fmt::Display + 'static>(
    meta: ReproMeta,
    label: &str,
    metric: &str,
    case_idx: usize,
    query_idx: usize,
    content_seed: u64,
    query_seed: u64,
    point_count: usize,
    max_qty: usize,
    radius_sq: A,
    radius_man: A,
    result: QueryResultItem<(), usize, A>,
    expected: &MetricState<A>,
) {
    if !distances_match_for_fuzz(result.distance, expected.best_dist) {
        log_mismatch(
            meta,
            label,
            case_idx,
            query_idx,
            content_seed,
            query_seed,
            point_count,
            max_qty,
            radius_sq,
            radius_man,
            format!(
                "metric={metric} op=nearest_one mismatch=distance expected={} got={}",
                expected.best_dist, result.distance
            ),
        );
    }

    if !expected.best_items.contains(&result.item) {
        log_mismatch(
            meta,
            label,
            case_idx,
            query_idx,
            content_seed,
            query_seed,
            point_count,
            max_qty,
            radius_sq,
            radius_man,
            format!(
                "metric={metric} op=nearest_one mismatch=item expected_one_of={:?} got={}",
                expected.best_items, result.item
            ),
        );
    }
}

fn assert_approx_nearest_one_f32<D, const K: usize>(
    meta: ReproMeta,
    label: &str,
    metric: &str,
    case_idx: usize,
    query_idx: usize,
    content_seed: u64,
    query_seed: u64,
    point_count: usize,
    max_qty: usize,
    radius_sq: f32,
    radius_man: f32,
    query: &[f32; K],
    points: &[[f32; K]],
    result: QueryResultItem<(), usize, f32>,
) where
    D: DistanceMetricScalar<f32, Output = f32>,
{
    if result.item >= points.len() {
        log_mismatch(
            meta,
            label,
            case_idx,
            query_idx,
            content_seed,
            query_seed,
            point_count,
            max_qty,
            radius_sq,
            radius_man,
            format!(
                "metric={metric} op=approx_nearest_one mismatch=item out_of_bounds={} len={}",
                result.item,
                points.len()
            ),
        );
    }

    let expected_dist = D::dist_raw(query, &points[result.item]);
    if !distances_match_for_fuzz(result.distance, expected_dist) {
        log_mismatch(
            meta,
            label,
            case_idx,
            query_idx,
            content_seed,
            query_seed,
            point_count,
            max_qty,
            radius_sq,
            radius_man,
            format!(
                "metric={metric} op=approx_nearest_one mismatch=distance expected={} got={} item={}",
                expected_dist, result.distance, result.item
            ),
        );
    }
}

fn assert_approx_nearest_one_f64<D, const K: usize>(
    meta: ReproMeta,
    label: &str,
    metric: &str,
    case_idx: usize,
    query_idx: usize,
    content_seed: u64,
    query_seed: u64,
    point_count: usize,
    max_qty: usize,
    radius_sq: f64,
    radius_man: f64,
    query: &[f64; K],
    points: &[[f64; K]],
    result: QueryResultItem<(), usize, f64>,
) where
    D: DistanceMetricScalar<f64, Output = f64>,
{
    if result.item >= points.len() {
        log_mismatch(
            meta,
            label,
            case_idx,
            query_idx,
            content_seed,
            query_seed,
            point_count,
            max_qty,
            radius_sq,
            radius_man,
            format!(
                "metric={metric} op=approx_nearest_one mismatch=item out_of_bounds={} len={}",
                result.item,
                points.len()
            ),
        );
    }

    let expected_dist = D::dist_raw(query, &points[result.item]);
    if !distances_match_for_fuzz(result.distance, expected_dist) {
        log_mismatch(
            meta,
            label,
            case_idx,
            query_idx,
            content_seed,
            query_seed,
            point_count,
            max_qty,
            radius_sq,
            radius_man,
            format!(
                "metric={metric} op=approx_nearest_one mismatch=distance expected={} got={} item={}",
                expected_dist, result.distance, result.item
            ),
        );
    }
}

fn query_nearest_n_sorted<A, T, SO, LS, D, const K: usize, const B: usize>(
    tree: &KdTree<A, T, SO, LS, K, B>,
    query: &[A; K],
    max_qty: NonZeroUsize,
) -> Vec<(D::Output, T)>
where
    A: Axis<Coord = A> + 'static,
    T: kiddo::Content + PartialOrd,
    SO: StemStrategy,
    LS: LeafStrategy<A, T, SO, K, B>,
    D: kiddo::dist::DistanceMetric<A>,
    D::Output: kiddo::stem_strategy::SimdPrune
        + kiddo::stem_strategy::donnelly::simd_full::SimdSelectBestChildBlock3
        + kiddo::stem_strategy::donnelly::simd_full::BacktrackBlock3
        + kiddo::stem_strategy::donnelly::simd_full::BacktrackBlock4
        + kiddo::leaf_view::TlsLeafScratch
        + 'static,
{
    tree.query(query)
        .nearest_n::<D>(max_qty)
        .execute()
        .into_iter()
        .map(|n| (n.distance, n.item))
        .collect()
}

fn query_nearest_n_unsorted<A, T, SO, LS, D, const K: usize, const B: usize>(
    tree: &KdTree<A, T, SO, LS, K, B>,
    query: &[A; K],
    max_qty: NonZeroUsize,
) -> Vec<(D::Output, T)>
where
    A: Axis<Coord = A> + 'static,
    T: kiddo::Content + PartialOrd,
    SO: StemStrategy,
    LS: LeafStrategy<A, T, SO, K, B>,
    D: kiddo::dist::DistanceMetric<A>,
    D::Output: kiddo::stem_strategy::SimdPrune
        + kiddo::stem_strategy::donnelly::simd_full::SimdSelectBestChildBlock3
        + kiddo::stem_strategy::donnelly::simd_full::BacktrackBlock3
        + kiddo::stem_strategy::donnelly::simd_full::BacktrackBlock4
        + kiddo::leaf_view::TlsLeafScratch
        + 'static,
{
    tree.query(query)
        .nearest_n::<D>(max_qty)
        .unsorted()
        .execute()
        .into_iter()
        .map(|n| (n.distance, n.item))
        .collect()
}

fn query_within_unsorted<A, T, SO, LS, D, const K: usize, const B: usize>(
    tree: &KdTree<A, T, SO, LS, K, B>,
    query: &[A; K],
    radius: D::Output,
) -> Vec<(D::Output, T)>
where
    A: Axis<Coord = A> + 'static,
    T: kiddo::Content + PartialOrd,
    SO: StemStrategy,
    LS: LeafStrategy<A, T, SO, K, B>,
    D: kiddo::dist::DistanceMetric<A>,
    D::Output: kiddo::stem_strategy::SimdPrune
        + kiddo::stem_strategy::donnelly::simd_full::SimdSelectBestChildBlock3
        + kiddo::stem_strategy::donnelly::simd_full::BacktrackBlock3
        + kiddo::stem_strategy::donnelly::simd_full::BacktrackBlock4
        + kiddo::leaf_view::TlsLeafScratch
        + 'static,
{
    tree.query(query)
        .within::<D>(radius)
        .unsorted()
        .execute()
        .into_iter()
        .map(|n| (n.distance, n.item))
        .collect()
}

fn query_within_sorted<A, T, SO, LS, D, const K: usize, const B: usize>(
    tree: &KdTree<A, T, SO, LS, K, B>,
    query: &[A; K],
    radius: D::Output,
) -> Vec<(D::Output, T)>
where
    A: Axis<Coord = A> + 'static,
    T: kiddo::Content + PartialOrd,
    SO: StemStrategy,
    LS: LeafStrategy<A, T, SO, K, B>,
    D: kiddo::dist::DistanceMetric<A>,
    D::Output: kiddo::stem_strategy::SimdPrune
        + kiddo::stem_strategy::donnelly::simd_full::SimdSelectBestChildBlock3
        + kiddo::stem_strategy::donnelly::simd_full::BacktrackBlock3
        + kiddo::stem_strategy::donnelly::simd_full::BacktrackBlock4
        + kiddo::leaf_view::TlsLeafScratch
        + 'static,
{
    tree.query(query)
        .within::<D>(radius)
        .execute()
        .into_iter()
        .map(|n| (n.distance, n.item))
        .collect()
}

fn assert_nearest_n_unsorted_contains_top_k<
    A: Axis<Coord = A> + PartialOrd + std::fmt::Display + 'static,
>(
    meta: ReproMeta,
    label: &str,
    metric: &str,
    case_idx: usize,
    query_idx: usize,
    content_seed: u64,
    query_seed: u64,
    point_count: usize,
    max_qty: usize,
    radius_sq: A,
    radius_man: A,
    expected_top_k: &[(A, usize)],
    got_unsorted: &mut [(A, usize)],
) {
    sort_by_distance_then_index(got_unsorted);

    if got_unsorted.len() != expected_top_k.len() {
        log_mismatch(
            meta,
            label,
            case_idx,
            query_idx,
            content_seed,
            query_seed,
            point_count,
            max_qty,
            radius_sq,
            radius_man,
            format!(
                "metric={metric} op=nearest_n_unsorted len mismatch expected={} got={}",
                expected_top_k.len(),
                got_unsorted.len()
            ),
        );
    }

    if let Err(reason) = compare_nearest_n_sorted(expected_top_k, got_unsorted) {
        log_mismatch(
            meta,
            label,
            case_idx,
            query_idx,
            content_seed,
            query_seed,
            point_count,
            max_qty,
            radius_sq,
            radius_man,
            format!(
                "metric={metric} op=nearest_n_unsorted {reason} expected={} got={}",
                format_preview(expected_top_k, PREVIEW_LEN),
                format_preview(got_unsorted, PREVIEW_LEN)
            ),
        );
    }
}

fn sort_by_item_idx<A>(items: &mut [(A, usize)]) {
    items.sort_unstable_by_key(|(_, item)| *item);
}

fn expected_best_n_within<A: Copy>(within_items: &[(A, usize)], max_qty: usize) -> Vec<(A, usize)> {
    let mut expected = within_items.to_vec();
    sort_by_item_idx(&mut expected);
    expected.truncate(max_qty);
    expected
}

fn print_case_start(label: &str, case_idx: usize, cases: usize, point_count: usize, seed: u64) {
    println!(
        "{label}: case {}/{} with {} points (content_seed={seed})",
        case_idx + 1,
        cases,
        point_count
    );
}

fn print_query_progress(
    label: &str,
    case_idx: usize,
    query_idx: usize,
    query_count: usize,
    seed: u64,
) {
    if query_idx.is_multiple_of(PROGRESS_EVERY) {
        println!(
            "{label}: case {} query {}/{} (query_seed={seed})",
            case_idx + 1,
            query_idx + 1,
            query_count
        );
    }
}

fn run_mutable_case_f32<const K: usize, const B: usize, SO>(
    cfg: FuzzConfig,
    label: &str,
    meta: ReproMeta,
) where
    SO: StemStrategy + 'static,
    <SO as StemStrategy>::Stack<f32, 2>: 'static,
{
    for case_idx in 0..cfg.cases {
        let content_seed = cfg.case_seed(case_idx);
        let mut rng_content = StdRng::seed_from_u64(content_seed);
        let point_count = random_point_count(cfg, &mut rng_content);
        let points: Vec<[f32; K]> = (0..point_count)
            .map(|_| random_point_f32::<K>(&mut rng_content))
            .collect();

        print_case_start(label, case_idx, cfg.cases, point_count, content_seed);

        let mut tree: KdTree<f32, usize, SO, VecOfArrays<f32, usize, K, B>, K, B> =
            KdTree::default();

        for (idx, point) in points.iter().enumerate() {
            tree.add(point, idx)
                .expect("random finite point set must be splittable");
        }

        let max_nearest_n = cfg.max_nearest_n.max(1).min(point_count);

        fuzz_pool().install(|| {
            (0..cfg.query_count).into_par_iter().for_each(|query_idx| {
            let query_seed = query_seed(content_seed, query_idx);
            let mut rng_query = StdRng::seed_from_u64(query_seed);
            let query = random_point_f32::<K>(&mut rng_query);
            let max_qty = rng_query.random_range(1..=max_nearest_n);
            let radius_sq = random_radius_f32::<K>(&mut rng_query);
            let radius_man = random_radius_f32::<K>(&mut rng_query);

            let (mut sq_state, mut man_state) =
                brute_states_f32(&points, &query, max_qty, radius_sq, radius_man);

            let result_sq = tree
                .query(&query)
                .nearest_one::<SquaredEuclidean<f32>>()
                .execute();
            assert_nearest_one(
                meta,
                label,
                "SquaredEuclidean",
                case_idx,
                query_idx,
                content_seed,
                query_seed,
                point_count,
                max_qty,
                radius_sq,
                radius_man,
                result_sq,
                &sq_state,
            );

            let result_man = tree.query(&query).nearest_one::<Manhattan<f32>>().execute();
            assert_nearest_one(
                meta,
                label,
                "Manhattan",
                case_idx,
                query_idx,
                content_seed,
                query_seed,
                point_count,
                max_qty,
                radius_sq,
                radius_man,
                result_man,
                &man_state,
            );

            let approx_sq = tree
                .query(&query)
                .nearest_one::<SquaredEuclidean<f32>>()
                .approx()
                .execute();
            assert_approx_nearest_one_f32::<SquaredEuclidean<f32>, K>(
                meta,
                label,
                "SquaredEuclidean",
                case_idx,
                query_idx,
                content_seed,
                query_seed,
                point_count,
                max_qty,
                radius_sq,
                radius_man,
                &query,
                &points,
                approx_sq,
            );

            let approx_man = tree
                .query(&query)
                .nearest_one::<Manhattan<f32>>()
                .approx()
                .execute();
            assert_approx_nearest_one_f32::<Manhattan<f32>, K>(
                meta,
                label,
                "Manhattan",
                case_idx,
                query_idx,
                content_seed,
                query_seed,
                point_count,
                max_qty,
                radius_sq,
                radius_man,
                &query,
                &points,
                approx_man,
            );

            let mut expected_n_sq = sq_state.take_nearest_n_sorted();
            sort_by_distance_then_index(&mut expected_n_sq);
            let max_qty_nz = NonZeroUsize::new(max_qty).expect("max_qty must be non-zero");
            let mut result_n_sq: Vec<(f32, usize)> =
                query_nearest_n_sorted::<_, _, _, _, SquaredEuclidean<f32>, K, B>(
                    &tree, &query, max_qty_nz,
                );
            sort_by_distance_then_index(&mut result_n_sq);
            if let Err(reason) = compare_nearest_n_sorted(&expected_n_sq, &result_n_sq) {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=SquaredEuclidean op=nearest_n {reason} expected={} got={}",
                        format_preview(&expected_n_sq, PREVIEW_LEN),
                        format_preview(&result_n_sq, PREVIEW_LEN)
                    ),
                );
            }

            let max_qty_nz = NonZeroUsize::new(max_qty).expect("max_qty must be non-zero");
            let mut result_n_sq_unsorted: Vec<(f32, usize)> =
                query_nearest_n_unsorted::<_, _, _, _, SquaredEuclidean<f32>, K, B>(
                    &tree, &query, max_qty_nz,
                );
            assert_nearest_n_unsorted_contains_top_k(
                meta,
                label,
                "SquaredEuclidean",
                case_idx,
                query_idx,
                content_seed,
                query_seed,
                point_count,
                max_qty,
                radius_sq,
                radius_man,
                &expected_n_sq,
                &mut result_n_sq_unsorted,
            );

            let mut expected_n_man = man_state.take_nearest_n_sorted();
            sort_by_distance_then_index(&mut expected_n_man);
            let max_qty_nz = NonZeroUsize::new(max_qty).expect("max_qty must be non-zero");
            let mut result_n_man: Vec<(f32, usize)> =
                query_nearest_n_sorted::<_, _, _, _, Manhattan<f32>, K, B>(
                    &tree, &query, max_qty_nz,
                );
            sort_by_distance_then_index(&mut result_n_man);
            if let Err(reason) = compare_nearest_n_sorted(&expected_n_man, &result_n_man) {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=Manhattan op=nearest_n {reason} expected={} got={}",
                        format_preview(&expected_n_man, PREVIEW_LEN),
                        format_preview(&result_n_man, PREVIEW_LEN)
                    ),
                );
            }

            let max_qty_nz = NonZeroUsize::new(max_qty).expect("max_qty must be non-zero");
            let mut result_n_man_unsorted: Vec<(f32, usize)> =
                query_nearest_n_unsorted::<_, _, _, _, Manhattan<f32>, K, B>(
                    &tree, &query, max_qty_nz,
                );
            assert_nearest_n_unsorted_contains_top_k(
                meta,
                label,
                "Manhattan",
                case_idx,
                query_idx,
                content_seed,
                query_seed,
                point_count,
                max_qty,
                radius_sq,
                radius_man,
                &expected_n_man,
                &mut result_n_man_unsorted,
            );

            let mut expected_within_sq = sq_state.take_within_sorted();
            let mut result_within_sq: Vec<(f32, usize)> =
                query_within_unsorted::<_, _, _, _, SquaredEuclidean<f32>, K, B>(
                    &tree, &query, radius_sq,
                );
            if let Err(reason) =
                compare_within_results(&mut expected_within_sq, &mut result_within_sq, radius_sq)
            {
                let missing_leaf_sample =
                    format_missing_leaf_sample(&tree, &expected_within_sq, &result_within_sq);
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=SquaredEuclidean op=within_unsorted {reason} {missing_leaf_sample} expected={} got={}",
                        format_preview(&expected_within_sq, PREVIEW_LEN),
                        format_preview(&result_within_sq, PREVIEW_LEN)
                    ),
                );
            }

            let mut result_within_sq_sorted: Vec<(f32, usize)> =
                query_within_sorted::<_, _, _, _, SquaredEuclidean<f32>, K, B>(
                    &tree, &query, radius_sq,
                );
            if let Err(reason) = compare_within_results(
                &mut expected_within_sq,
                &mut result_within_sq_sorted,
                radius_sq,
            ) {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=SquaredEuclidean op=within {reason} expected={} got={}",
                        format_preview(&expected_within_sq, PREVIEW_LEN),
                        format_preview(&result_within_sq_sorted, PREVIEW_LEN)
                    ),
                );
            }

            let expected_best_within_sq = expected_best_n_within(&expected_within_sq, max_qty);
            let max_qty_nz = NonZeroUsize::new(max_qty).expect("max_qty must be non-zero");
            let mut result_best_within_sq: Vec<(f32, usize)> = tree
                .query(&query)
                .best_n_within::<SquaredEuclidean<f32>>(radius_sq, max_qty_nz)
                .execute()
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();
            sort_by_item_idx(&mut result_best_within_sq);
            if let Err(reason) =
                compare_item_sorted_results(&expected_best_within_sq, &result_best_within_sq)
            {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=SquaredEuclidean op=best_n_within {reason} expected={} got={}",
                        format_preview(&expected_best_within_sq, PREVIEW_LEN),
                        format_preview(&result_best_within_sq, PREVIEW_LEN)
                    ),
                );
            }

            let mut expected_within_man = man_state.take_within_sorted();
            let mut result_within_man: Vec<(f32, usize)> =
                query_within_unsorted::<_, _, _, _, Manhattan<f32>, K, B>(
                    &tree, &query, radius_man,
                );
            if let Err(reason) =
                compare_within_results(&mut expected_within_man, &mut result_within_man, radius_man)
            {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=Manhattan op=within_unsorted {reason} expected={} got={}",
                        format_preview(&expected_within_man, PREVIEW_LEN),
                        format_preview(&result_within_man, PREVIEW_LEN)
                    ),
                );
            }

            let mut result_within_man_sorted: Vec<(f32, usize)> =
                query_within_sorted::<_, _, _, _, Manhattan<f32>, K, B>(&tree, &query, radius_man);
            if let Err(reason) = compare_within_results(
                &mut expected_within_man,
                &mut result_within_man_sorted,
                radius_man,
            ) {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=Manhattan op=within {reason} expected={} got={}",
                        format_preview(&expected_within_man, PREVIEW_LEN),
                        format_preview(&result_within_man_sorted, PREVIEW_LEN)
                    ),
                );
            }

            let expected_best_within_man = expected_best_n_within(&expected_within_man, max_qty);
            let max_qty_nz = NonZeroUsize::new(max_qty).expect("max_qty must be non-zero");
            let mut result_best_within_man: Vec<(f32, usize)> = tree
                .query(&query)
                .best_n_within::<Manhattan<f32>>(radius_man, max_qty_nz)
                .execute()
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();
            sort_by_item_idx(&mut result_best_within_man);
            if let Err(reason) =
                compare_item_sorted_results(&expected_best_within_man, &result_best_within_man)
            {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=Manhattan op=best_n_within {reason} expected={} got={}",
                        format_preview(&expected_best_within_man, PREVIEW_LEN),
                        format_preview(&result_best_within_man, PREVIEW_LEN)
                    ),
                );
            }

                print_query_progress(label, case_idx, query_idx, cfg.query_count, query_seed);
            });
        });
    }
}

fn run_mutable_case_f64<const K: usize, const B: usize, SO>(
    cfg: FuzzConfig,
    label: &str,
    meta: ReproMeta,
) where
    SO: StemStrategy + 'static,
    <SO as StemStrategy>::Stack<f64, 2>: 'static,
{
    for case_idx in 0..cfg.cases {
        let content_seed = cfg.case_seed(case_idx);
        let mut rng_content = StdRng::seed_from_u64(content_seed);
        let point_count = random_point_count(cfg, &mut rng_content);
        let points: Vec<[f64; K]> = (0..point_count)
            .map(|_| random_point_f64::<K>(&mut rng_content))
            .collect();

        print_case_start(label, case_idx, cfg.cases, point_count, content_seed);

        let mut tree: KdTree<f64, usize, SO, VecOfArrays<f64, usize, K, B>, K, B> =
            KdTree::default();

        for (idx, point) in points.iter().enumerate() {
            tree.add(point, idx)
                .expect("random finite point set must be splittable");
        }

        let max_nearest_n = cfg.max_nearest_n.max(1).min(point_count);

        fuzz_pool().install(|| {
            (0..cfg.query_count).into_par_iter().for_each(|query_idx| {
            let query_seed = query_seed(content_seed, query_idx);
            let mut rng_query = StdRng::seed_from_u64(query_seed);
            let query = random_point_f64::<K>(&mut rng_query);
            let max_qty = rng_query.random_range(1..=max_nearest_n);
            let radius_sq = random_radius_f64::<K>(&mut rng_query);
            let radius_man = random_radius_f64::<K>(&mut rng_query);

            let (mut sq_state, mut man_state) =
                brute_states_f64(&points, &query, max_qty, radius_sq, radius_man);

            let result_sq = tree
                .query(&query)
                .nearest_one::<SquaredEuclidean<f64>>()
                .execute();
            assert_nearest_one(
                meta,
                label,
                "SquaredEuclidean",
                case_idx,
                query_idx,
                content_seed,
                query_seed,
                point_count,
                max_qty,
                radius_sq,
                radius_man,
                result_sq,
                &sq_state,
            );

            let result_man = tree.query(&query).nearest_one::<Manhattan<f64>>().execute();
            assert_nearest_one(
                meta,
                label,
                "Manhattan",
                case_idx,
                query_idx,
                content_seed,
                query_seed,
                point_count,
                max_qty,
                radius_sq,
                radius_man,
                result_man,
                &man_state,
            );

            let approx_sq = tree
                .query(&query)
                .nearest_one::<SquaredEuclidean<f64>>()
                .approx()
                .execute();
            assert_approx_nearest_one_f64::<SquaredEuclidean<f64>, K>(
                meta,
                label,
                "SquaredEuclidean",
                case_idx,
                query_idx,
                content_seed,
                query_seed,
                point_count,
                max_qty,
                radius_sq,
                radius_man,
                &query,
                &points,
                approx_sq,
            );

            let approx_man = tree
                .query(&query)
                .nearest_one::<Manhattan<f64>>()
                .approx()
                .execute();
            assert_approx_nearest_one_f64::<Manhattan<f64>, K>(
                meta,
                label,
                "Manhattan",
                case_idx,
                query_idx,
                content_seed,
                query_seed,
                point_count,
                max_qty,
                radius_sq,
                radius_man,
                &query,
                &points,
                approx_man,
            );

            let mut expected_n_sq = sq_state.take_nearest_n_sorted();
            sort_by_distance_then_index(&mut expected_n_sq);
            let max_qty_nz = NonZeroUsize::new(max_qty).expect("max_qty must be non-zero");
            let mut result_n_sq: Vec<(f64, usize)> =
                query_nearest_n_sorted::<_, _, _, _, SquaredEuclidean<f64>, K, B>(
                    &tree, &query, max_qty_nz,
                );
            sort_by_distance_then_index(&mut result_n_sq);
            if let Err(reason) = compare_nearest_n_sorted(&expected_n_sq, &result_n_sq) {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=SquaredEuclidean op=nearest_n {reason} expected={} got={}",
                        format_preview(&expected_n_sq, PREVIEW_LEN),
                        format_preview(&result_n_sq, PREVIEW_LEN)
                    ),
                );
            }

            let max_qty_nz = NonZeroUsize::new(max_qty).expect("max_qty must be non-zero");
            let mut result_n_sq_unsorted: Vec<(f64, usize)> =
                query_nearest_n_unsorted::<_, _, _, _, SquaredEuclidean<f64>, K, B>(
                    &tree, &query, max_qty_nz,
                );
            assert_nearest_n_unsorted_contains_top_k(
                meta,
                label,
                "SquaredEuclidean",
                case_idx,
                query_idx,
                content_seed,
                query_seed,
                point_count,
                max_qty,
                radius_sq,
                radius_man,
                &expected_n_sq,
                &mut result_n_sq_unsorted,
            );

            let mut expected_n_man = man_state.take_nearest_n_sorted();
            sort_by_distance_then_index(&mut expected_n_man);
            let max_qty_nz = NonZeroUsize::new(max_qty).expect("max_qty must be non-zero");
            let mut result_n_man: Vec<(f64, usize)> =
                query_nearest_n_sorted::<_, _, _, _, Manhattan<f64>, K, B>(
                    &tree, &query, max_qty_nz,
                );
            sort_by_distance_then_index(&mut result_n_man);
            if let Err(reason) = compare_nearest_n_sorted(&expected_n_man, &result_n_man) {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=Manhattan op=nearest_n {reason} expected={} got={}",
                        format_preview(&expected_n_man, PREVIEW_LEN),
                        format_preview(&result_n_man, PREVIEW_LEN)
                    ),
                );
            }

            let max_qty_nz = NonZeroUsize::new(max_qty).expect("max_qty must be non-zero");
            let mut result_n_man_unsorted: Vec<(f64, usize)> =
                query_nearest_n_unsorted::<_, _, _, _, Manhattan<f64>, K, B>(
                    &tree, &query, max_qty_nz,
                );
            assert_nearest_n_unsorted_contains_top_k(
                meta,
                label,
                "Manhattan",
                case_idx,
                query_idx,
                content_seed,
                query_seed,
                point_count,
                max_qty,
                radius_sq,
                radius_man,
                &expected_n_man,
                &mut result_n_man_unsorted,
            );

            let mut expected_within_sq = sq_state.take_within_sorted();
            let mut result_within_sq: Vec<(f64, usize)> =
                query_within_unsorted::<_, _, _, _, SquaredEuclidean<f64>, K, B>(
                    &tree, &query, radius_sq,
                );
            if let Err(reason) =
                compare_within_results(&mut expected_within_sq, &mut result_within_sq, radius_sq)
            {
                let missing_leaf_sample =
                    format_missing_leaf_sample(&tree, &expected_within_sq, &result_within_sq);
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=SquaredEuclidean op=within_unsorted {reason} {missing_leaf_sample} expected={} got={}",
                        format_preview(&expected_within_sq, PREVIEW_LEN),
                        format_preview(&result_within_sq, PREVIEW_LEN)
                    ),
                );
            }

            let mut result_within_sq_sorted: Vec<(f64, usize)> =
                query_within_sorted::<_, _, _, _, SquaredEuclidean<f64>, K, B>(
                    &tree, &query, radius_sq,
                );
            if let Err(reason) = compare_within_results(
                &mut expected_within_sq,
                &mut result_within_sq_sorted,
                radius_sq,
            ) {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=SquaredEuclidean op=within {reason} expected={} got={}",
                        format_preview(&expected_within_sq, PREVIEW_LEN),
                        format_preview(&result_within_sq_sorted, PREVIEW_LEN)
                    ),
                );
            }

            let expected_best_within_sq = expected_best_n_within(&expected_within_sq, max_qty);
            let max_qty_nz = NonZeroUsize::new(max_qty).expect("max_qty must be non-zero");
            let mut result_best_within_sq: Vec<(f64, usize)> = tree
                .query(&query)
                .best_n_within::<SquaredEuclidean<f64>>(radius_sq, max_qty_nz)
                .execute()
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();
            sort_by_item_idx(&mut result_best_within_sq);
            if let Err(reason) =
                compare_item_sorted_results(&expected_best_within_sq, &result_best_within_sq)
            {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=SquaredEuclidean op=best_n_within {reason} expected={} got={}",
                        format_preview(&expected_best_within_sq, PREVIEW_LEN),
                        format_preview(&result_best_within_sq, PREVIEW_LEN)
                    ),
                );
            }

            let mut expected_within_man = man_state.take_within_sorted();
            let mut result_within_man: Vec<(f64, usize)> =
                query_within_unsorted::<_, _, _, _, Manhattan<f64>, K, B>(
                    &tree, &query, radius_man,
                );
            if let Err(reason) =
                compare_within_results(&mut expected_within_man, &mut result_within_man, radius_man)
            {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=Manhattan op=within_unsorted {reason} expected={} got={}",
                        format_preview(&expected_within_man, PREVIEW_LEN),
                        format_preview(&result_within_man, PREVIEW_LEN)
                    ),
                );
            }

            let mut result_within_man_sorted: Vec<(f64, usize)> =
                query_within_sorted::<_, _, _, _, Manhattan<f64>, K, B>(&tree, &query, radius_man);
            if let Err(reason) = compare_within_results(
                &mut expected_within_man,
                &mut result_within_man_sorted,
                radius_man,
            ) {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=Manhattan op=within {reason} expected={} got={}",
                        format_preview(&expected_within_man, PREVIEW_LEN),
                        format_preview(&result_within_man_sorted, PREVIEW_LEN)
                    ),
                );
            }

            let expected_best_within_man = expected_best_n_within(&expected_within_man, max_qty);
            let max_qty_nz = NonZeroUsize::new(max_qty).expect("max_qty must be non-zero");
            let mut result_best_within_man: Vec<(f64, usize)> = tree
                .query(&query)
                .best_n_within::<Manhattan<f64>>(radius_man, max_qty_nz)
                .execute()
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();
            sort_by_item_idx(&mut result_best_within_man);
            if let Err(reason) =
                compare_item_sorted_results(&expected_best_within_man, &result_best_within_man)
            {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=Manhattan op=best_n_within {reason} expected={} got={}",
                        format_preview(&expected_best_within_man, PREVIEW_LEN),
                        format_preview(&result_best_within_man, PREVIEW_LEN)
                    ),
                );
            }

                print_query_progress(label, case_idx, query_idx, cfg.query_count, query_seed);
            });
        });
    }
}

fn run_immutable_case_f32_with_leaf<const K: usize, const B: usize, SO, LS>(
    cfg: FuzzConfig,
    label: &str,
    meta: ReproMeta,
) where
    SO: StemStrategy + 'static,
    <SO as StemStrategy>::Stack<f32, 2>: 'static,
    LS: LeafStrategy<f32, usize, SO, K, B> + ConstructibleLeafStrategy<f32, usize, SO, K, B> + Sync,
{
    for case_idx in 0..cfg.cases {
        let content_seed = cfg.case_seed(case_idx);
        let mut rng_content = StdRng::seed_from_u64(content_seed);
        let point_count = random_point_count(cfg, &mut rng_content);
        let points: Vec<[f32; K]> = (0..point_count)
            .map(|_| random_point_f32::<K>(&mut rng_content))
            .collect();

        print_case_start(label, case_idx, cfg.cases, point_count, content_seed);

        let tree: KdTree<f32, usize, SO, LS, K, B> = KdTree::new_from_slice(&points).unwrap();

        let max_nearest_n = cfg.max_nearest_n.max(1).min(point_count);

        fuzz_pool().install(|| {
            (0..cfg.query_count).into_par_iter().for_each(|query_idx| {
            let query_seed = query_seed(content_seed, query_idx);
            let mut rng_query = StdRng::seed_from_u64(query_seed);
            let query = random_point_f32::<K>(&mut rng_query);
            let max_qty = rng_query.random_range(1..=max_nearest_n);
            let radius_sq = random_radius_f32::<K>(&mut rng_query);
            let radius_man = random_radius_f32::<K>(&mut rng_query);

            let (mut sq_state, mut man_state) =
                brute_states_f32(&points, &query, max_qty, radius_sq, radius_man);

            let result_sq = tree
                .query(&query)
                .nearest_one::<SquaredEuclidean<f32>>()
                .execute();
            assert_nearest_one(
                meta,
                label,
                "SquaredEuclidean",
                case_idx,
                query_idx,
                content_seed,
                query_seed,
                point_count,
                max_qty,
                radius_sq,
                radius_man,
                result_sq,
                &sq_state,
            );

            let result_man = tree.query(&query).nearest_one::<Manhattan<f32>>().execute();
            assert_nearest_one(
                meta,
                label,
                "Manhattan",
                case_idx,
                query_idx,
                content_seed,
                query_seed,
                point_count,
                max_qty,
                radius_sq,
                radius_man,
                result_man,
                &man_state,
            );

            let approx_sq = tree
                .query(&query)
                .nearest_one::<SquaredEuclidean<f32>>()
                .approx()
                .execute();
            assert_approx_nearest_one_f32::<SquaredEuclidean<f32>, K>(
                meta,
                label,
                "SquaredEuclidean",
                case_idx,
                query_idx,
                content_seed,
                query_seed,
                point_count,
                max_qty,
                radius_sq,
                radius_man,
                &query,
                &points,
                approx_sq,
            );

            let approx_man = tree
                .query(&query)
                .nearest_one::<Manhattan<f32>>()
                .approx()
                .execute();
            assert_approx_nearest_one_f32::<Manhattan<f32>, K>(
                meta,
                label,
                "Manhattan",
                case_idx,
                query_idx,
                content_seed,
                query_seed,
                point_count,
                max_qty,
                radius_sq,
                radius_man,
                &query,
                &points,
                approx_man,
            );

            let mut expected_n_sq = sq_state.take_nearest_n_sorted();
            sort_by_distance_then_index(&mut expected_n_sq);
            let max_qty_nz = NonZeroUsize::new(max_qty).expect("max_qty must be non-zero");
            let mut result_n_sq: Vec<(f32, usize)> =
                query_nearest_n_sorted::<_, _, _, _, SquaredEuclidean<f32>, K, B>(
                    &tree, &query, max_qty_nz,
                );
            sort_by_distance_then_index(&mut result_n_sq);
            if let Err(reason) = compare_nearest_n_sorted(&expected_n_sq, &result_n_sq) {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=SquaredEuclidean op=nearest_n {reason} expected={} got={}",
                        format_preview(&expected_n_sq, PREVIEW_LEN),
                        format_preview(&result_n_sq, PREVIEW_LEN)
                    ),
                );
            }

            let max_qty_nz = NonZeroUsize::new(max_qty).expect("max_qty must be non-zero");
            let mut result_n_sq_unsorted: Vec<(f32, usize)> =
                query_nearest_n_unsorted::<_, _, _, _, SquaredEuclidean<f32>, K, B>(
                    &tree, &query, max_qty_nz,
                );
            assert_nearest_n_unsorted_contains_top_k(
                meta,
                label,
                "SquaredEuclidean",
                case_idx,
                query_idx,
                content_seed,
                query_seed,
                point_count,
                max_qty,
                radius_sq,
                radius_man,
                &expected_n_sq,
                &mut result_n_sq_unsorted,
            );

            let mut expected_n_man = man_state.take_nearest_n_sorted();
            sort_by_distance_then_index(&mut expected_n_man);
            let max_qty_nz = NonZeroUsize::new(max_qty).expect("max_qty must be non-zero");
            let mut result_n_man: Vec<(f32, usize)> =
                query_nearest_n_sorted::<_, _, _, _, Manhattan<f32>, K, B>(
                    &tree, &query, max_qty_nz,
                );
            sort_by_distance_then_index(&mut result_n_man);
            if let Err(reason) = compare_nearest_n_sorted(&expected_n_man, &result_n_man) {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=Manhattan op=nearest_n {reason} expected={} got={}",
                        format_preview(&expected_n_man, PREVIEW_LEN),
                        format_preview(&result_n_man, PREVIEW_LEN)
                    ),
                );
            }

            let max_qty_nz = NonZeroUsize::new(max_qty).expect("max_qty must be non-zero");
            let mut result_n_man_unsorted: Vec<(f32, usize)> =
                query_nearest_n_unsorted::<_, _, _, _, Manhattan<f32>, K, B>(
                    &tree, &query, max_qty_nz,
                );
            assert_nearest_n_unsorted_contains_top_k(
                meta,
                label,
                "Manhattan",
                case_idx,
                query_idx,
                content_seed,
                query_seed,
                point_count,
                max_qty,
                radius_sq,
                radius_man,
                &expected_n_man,
                &mut result_n_man_unsorted,
            );

            let mut expected_within_sq = sq_state.take_within_sorted();
            let mut result_within_sq: Vec<(f32, usize)> =
                query_within_unsorted::<_, _, _, _, SquaredEuclidean<f32>, K, B>(
                    &tree, &query, radius_sq,
                );
            if let Err(reason) =
                compare_within_results(&mut expected_within_sq, &mut result_within_sq, radius_sq)
            {
                let missing_leaf_sample =
                    format_missing_leaf_sample(&tree, &expected_within_sq, &result_within_sq);
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=SquaredEuclidean op=within_unsorted {reason} {missing_leaf_sample} expected={} got={}",
                        format_preview(&expected_within_sq, PREVIEW_LEN),
                        format_preview(&result_within_sq, PREVIEW_LEN)
                    ),
                );
            }

            let mut result_within_sq_sorted: Vec<(f32, usize)> =
                query_within_sorted::<_, _, _, _, SquaredEuclidean<f32>, K, B>(
                    &tree, &query, radius_sq,
                );
            if let Err(reason) = compare_within_results(
                &mut expected_within_sq,
                &mut result_within_sq_sorted,
                radius_sq,
            ) {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=SquaredEuclidean op=within {reason} expected={} got={}",
                        format_preview(&expected_within_sq, PREVIEW_LEN),
                        format_preview(&result_within_sq_sorted, PREVIEW_LEN)
                    ),
                );
            }

            let expected_best_within_sq = expected_best_n_within(&expected_within_sq, max_qty);
            let max_qty_nz = NonZeroUsize::new(max_qty).expect("max_qty must be non-zero");
            let mut result_best_within_sq: Vec<(f32, usize)> = tree
                .query(&query)
                .best_n_within::<SquaredEuclidean<f32>>(radius_sq, max_qty_nz)
                .execute()
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();
            sort_by_item_idx(&mut result_best_within_sq);
            if let Err(reason) =
                compare_item_sorted_results(&expected_best_within_sq, &result_best_within_sq)
            {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=SquaredEuclidean op=best_n_within {reason} expected={} got={}",
                        format_preview(&expected_best_within_sq, PREVIEW_LEN),
                        format_preview(&result_best_within_sq, PREVIEW_LEN)
                    ),
                );
            }

            let mut expected_within_man = man_state.take_within_sorted();
            let mut result_within_man: Vec<(f32, usize)> =
                query_within_unsorted::<_, _, _, _, Manhattan<f32>, K, B>(
                    &tree, &query, radius_man,
                );
            if let Err(reason) =
                compare_within_results(&mut expected_within_man, &mut result_within_man, radius_man)
            {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=Manhattan op=within_unsorted {reason} expected={} got={}",
                        format_preview(&expected_within_man, PREVIEW_LEN),
                        format_preview(&result_within_man, PREVIEW_LEN)
                    ),
                );
            }

            let mut result_within_man_sorted: Vec<(f32, usize)> =
                query_within_sorted::<_, _, _, _, Manhattan<f32>, K, B>(&tree, &query, radius_man);
            if let Err(reason) = compare_within_results(
                &mut expected_within_man,
                &mut result_within_man_sorted,
                radius_man,
            ) {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=Manhattan op=within {reason} expected={} got={}",
                        format_preview(&expected_within_man, PREVIEW_LEN),
                        format_preview(&result_within_man_sorted, PREVIEW_LEN)
                    ),
                );
            }

            let expected_best_within_man = expected_best_n_within(&expected_within_man, max_qty);
            let max_qty_nz = NonZeroUsize::new(max_qty).expect("max_qty must be non-zero");
            let mut result_best_within_man: Vec<(f32, usize)> = tree
                .query(&query)
                .best_n_within::<Manhattan<f32>>(radius_man, max_qty_nz)
                .execute()
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();
            sort_by_item_idx(&mut result_best_within_man);
            if let Err(reason) =
                compare_item_sorted_results(&expected_best_within_man, &result_best_within_man)
            {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=Manhattan op=best_n_within {reason} expected={} got={}",
                        format_preview(&expected_best_within_man, PREVIEW_LEN),
                        format_preview(&result_best_within_man, PREVIEW_LEN)
                    ),
                );
            }

                print_query_progress(label, case_idx, query_idx, cfg.query_count, query_seed);
            });
        });
    }
}

fn run_immutable_case_f32<const K: usize, const B: usize, SO>(
    cfg: FuzzConfig,
    label: &str,
    meta: ReproMeta,
) where
    SO: StemStrategy + 'static,
    <SO as StemStrategy>::Stack<f32, 2>: 'static,
{
    run_immutable_case_f32_with_leaf::<K, B, SO, FlatVec<f32, usize, K, B>>(cfg, label, meta);
}

fn run_immutable_case_f32_arenas<const K: usize, const B: usize, SO>(
    cfg: FuzzConfig,
    label: &str,
    meta: ReproMeta,
) where
    SO: StemStrategy + 'static,
    <SO as StemStrategy>::Stack<f32, 2>: 'static,
{
    run_immutable_case_f32_with_leaf::<K, B, SO, VecOfArenas<f32, usize, K, B>>(cfg, label, meta);
}

fn run_immutable_case_f64_with_leaf<const K: usize, const B: usize, SO, LS>(
    cfg: FuzzConfig,
    label: &str,
    meta: ReproMeta,
) where
    SO: StemStrategy + 'static,
    <SO as StemStrategy>::Stack<f64, 2>: 'static,
    LS: LeafStrategy<f64, usize, SO, K, B> + ConstructibleLeafStrategy<f64, usize, SO, K, B> + Sync,
{
    for case_idx in 0..cfg.cases {
        let content_seed = cfg.case_seed(case_idx);
        let mut rng_content = StdRng::seed_from_u64(content_seed);
        let point_count = random_point_count(cfg, &mut rng_content);
        let points: Vec<[f64; K]> = (0..point_count)
            .map(|_| random_point_f64::<K>(&mut rng_content))
            .collect();

        print_case_start(label, case_idx, cfg.cases, point_count, content_seed);

        let tree: KdTree<f64, usize, SO, LS, K, B> = KdTree::new_from_slice(&points).unwrap();

        let max_nearest_n = cfg.max_nearest_n.max(1).min(point_count);

        fuzz_pool().install(|| {
            (0..cfg.query_count).into_par_iter().for_each(|query_idx| {
            let query_seed = query_seed(content_seed, query_idx);
            let mut rng_query = StdRng::seed_from_u64(query_seed);
            let query = random_point_f64::<K>(&mut rng_query);
            let max_qty = rng_query.random_range(1..=max_nearest_n);
            let radius_sq = random_radius_f64::<K>(&mut rng_query);
            let radius_man = random_radius_f64::<K>(&mut rng_query);

            let (mut sq_state, mut man_state) =
                brute_states_f64(&points, &query, max_qty, radius_sq, radius_man);

            let result_sq = tree
                .query(&query)
                .nearest_one::<SquaredEuclidean<f64>>()
                .execute();
            assert_nearest_one(
                meta,
                label,
                "SquaredEuclidean",
                case_idx,
                query_idx,
                content_seed,
                query_seed,
                point_count,
                max_qty,
                radius_sq,
                radius_man,
                result_sq,
                &sq_state,
            );

            let result_man = tree.query(&query).nearest_one::<Manhattan<f64>>().execute();
            assert_nearest_one(
                meta,
                label,
                "Manhattan",
                case_idx,
                query_idx,
                content_seed,
                query_seed,
                point_count,
                max_qty,
                radius_sq,
                radius_man,
                result_man,
                &man_state,
            );

            let approx_sq = tree
                .query(&query)
                .nearest_one::<SquaredEuclidean<f64>>()
                .approx()
                .execute();
            assert_approx_nearest_one_f64::<SquaredEuclidean<f64>, K>(
                meta,
                label,
                "SquaredEuclidean",
                case_idx,
                query_idx,
                content_seed,
                query_seed,
                point_count,
                max_qty,
                radius_sq,
                radius_man,
                &query,
                &points,
                approx_sq,
            );

            let approx_man = tree
                .query(&query)
                .nearest_one::<Manhattan<f64>>()
                .approx()
                .execute();
            assert_approx_nearest_one_f64::<Manhattan<f64>, K>(
                meta,
                label,
                "Manhattan",
                case_idx,
                query_idx,
                content_seed,
                query_seed,
                point_count,
                max_qty,
                radius_sq,
                radius_man,
                &query,
                &points,
                approx_man,
            );

            let mut expected_n_sq = sq_state.take_nearest_n_sorted();
            sort_by_distance_then_index(&mut expected_n_sq);
            let max_qty_nz = NonZeroUsize::new(max_qty).expect("max_qty must be non-zero");
            let mut result_n_sq: Vec<(f64, usize)> =
                query_nearest_n_sorted::<_, _, _, _, SquaredEuclidean<f64>, K, B>(
                    &tree, &query, max_qty_nz,
                );
            sort_by_distance_then_index(&mut result_n_sq);
            if let Err(reason) = compare_nearest_n_sorted(&expected_n_sq, &result_n_sq) {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=SquaredEuclidean op=nearest_n {reason} expected={} got={}",
                        format_preview(&expected_n_sq, PREVIEW_LEN),
                        format_preview(&result_n_sq, PREVIEW_LEN)
                    ),
                );
            }

            let max_qty_nz = NonZeroUsize::new(max_qty).expect("max_qty must be non-zero");
            let mut result_n_sq_unsorted: Vec<(f64, usize)> =
                query_nearest_n_unsorted::<_, _, _, _, SquaredEuclidean<f64>, K, B>(
                    &tree, &query, max_qty_nz,
                );
            assert_nearest_n_unsorted_contains_top_k(
                meta,
                label,
                "SquaredEuclidean",
                case_idx,
                query_idx,
                content_seed,
                query_seed,
                point_count,
                max_qty,
                radius_sq,
                radius_man,
                &expected_n_sq,
                &mut result_n_sq_unsorted,
            );

            let mut expected_n_man = man_state.take_nearest_n_sorted();
            sort_by_distance_then_index(&mut expected_n_man);
            let max_qty_nz = NonZeroUsize::new(max_qty).expect("max_qty must be non-zero");
            let mut result_n_man: Vec<(f64, usize)> =
                query_nearest_n_sorted::<_, _, _, _, Manhattan<f64>, K, B>(
                    &tree, &query, max_qty_nz,
                );
            sort_by_distance_then_index(&mut result_n_man);
            if let Err(reason) = compare_nearest_n_sorted(&expected_n_man, &result_n_man) {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=Manhattan op=nearest_n {reason} expected={} got={}",
                        format_preview(&expected_n_man, PREVIEW_LEN),
                        format_preview(&result_n_man, PREVIEW_LEN)
                    ),
                );
            }

            let max_qty_nz = NonZeroUsize::new(max_qty).expect("max_qty must be non-zero");
            let mut result_n_man_unsorted: Vec<(f64, usize)> =
                query_nearest_n_unsorted::<_, _, _, _, Manhattan<f64>, K, B>(
                    &tree, &query, max_qty_nz,
                );
            assert_nearest_n_unsorted_contains_top_k(
                meta,
                label,
                "Manhattan",
                case_idx,
                query_idx,
                content_seed,
                query_seed,
                point_count,
                max_qty,
                radius_sq,
                radius_man,
                &expected_n_man,
                &mut result_n_man_unsorted,
            );

            let mut expected_within_sq = sq_state.take_within_sorted();
            let mut result_within_sq: Vec<(f64, usize)> =
                query_within_unsorted::<_, _, _, _, SquaredEuclidean<f64>, K, B>(
                    &tree, &query, radius_sq,
                );
            if let Err(reason) =
                compare_within_results(&mut expected_within_sq, &mut result_within_sq, radius_sq)
            {
                let missing_leaf_sample =
                    format_missing_leaf_sample(&tree, &expected_within_sq, &result_within_sq);
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=SquaredEuclidean op=within_unsorted {reason} {missing_leaf_sample} expected={} got={}",
                        format_preview(&expected_within_sq, PREVIEW_LEN),
                        format_preview(&result_within_sq, PREVIEW_LEN)
                    ),
                );
            }

            let mut result_within_sq_sorted: Vec<(f64, usize)> =
                query_within_sorted::<_, _, _, _, SquaredEuclidean<f64>, K, B>(
                    &tree, &query, radius_sq,
                );
            if let Err(reason) = compare_within_results(
                &mut expected_within_sq,
                &mut result_within_sq_sorted,
                radius_sq,
            ) {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=SquaredEuclidean op=within {reason} expected={} got={}",
                        format_preview(&expected_within_sq, PREVIEW_LEN),
                        format_preview(&result_within_sq_sorted, PREVIEW_LEN)
                    ),
                );
            }

            let expected_best_within_sq = expected_best_n_within(&expected_within_sq, max_qty);
            let max_qty_nz = NonZeroUsize::new(max_qty).expect("max_qty must be non-zero");
            let mut result_best_within_sq: Vec<(f64, usize)> = tree
                .query(&query)
                .best_n_within::<SquaredEuclidean<f64>>(radius_sq, max_qty_nz)
                .execute()
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();
            sort_by_item_idx(&mut result_best_within_sq);
            if let Err(reason) =
                compare_item_sorted_results(&expected_best_within_sq, &result_best_within_sq)
            {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=SquaredEuclidean op=best_n_within {reason} expected={} got={}",
                        format_preview(&expected_best_within_sq, PREVIEW_LEN),
                        format_preview(&result_best_within_sq, PREVIEW_LEN)
                    ),
                );
            }

            let mut expected_within_man = man_state.take_within_sorted();
            let mut result_within_man: Vec<(f64, usize)> =
                query_within_unsorted::<_, _, _, _, Manhattan<f64>, K, B>(
                    &tree, &query, radius_man,
                );
            if let Err(reason) =
                compare_within_results(&mut expected_within_man, &mut result_within_man, radius_man)
            {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=Manhattan op=within_unsorted {reason} expected={} got={}",
                        format_preview(&expected_within_man, PREVIEW_LEN),
                        format_preview(&result_within_man, PREVIEW_LEN)
                    ),
                );
            }

            let mut result_within_man_sorted: Vec<(f64, usize)> =
                query_within_sorted::<_, _, _, _, Manhattan<f64>, K, B>(&tree, &query, radius_man);
            if let Err(reason) = compare_within_results(
                &mut expected_within_man,
                &mut result_within_man_sorted,
                radius_man,
            ) {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=Manhattan op=within {reason} expected={} got={}",
                        format_preview(&expected_within_man, PREVIEW_LEN),
                        format_preview(&result_within_man_sorted, PREVIEW_LEN)
                    ),
                );
            }

            let expected_best_within_man = expected_best_n_within(&expected_within_man, max_qty);
            let max_qty_nz = NonZeroUsize::new(max_qty).expect("max_qty must be non-zero");
            let mut result_best_within_man: Vec<(f64, usize)> = tree
                .query(&query)
                .best_n_within::<Manhattan<f64>>(radius_man, max_qty_nz)
                .execute()
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();
            sort_by_item_idx(&mut result_best_within_man);
            if let Err(reason) =
                compare_item_sorted_results(&expected_best_within_man, &result_best_within_man)
            {
                log_mismatch(
                    meta,
                    label,
                    case_idx,
                    query_idx,
                    content_seed,
                    query_seed,
                    point_count,
                    max_qty,
                    radius_sq,
                    radius_man,
                    format!(
                        "metric=Manhattan op=best_n_within {reason} expected={} got={}",
                        format_preview(&expected_best_within_man, PREVIEW_LEN),
                        format_preview(&result_best_within_man, PREVIEW_LEN)
                    ),
                );
            }

                print_query_progress(label, case_idx, query_idx, cfg.query_count, query_seed);
            });
        });
    }
}

fn run_immutable_case_f64<const K: usize, const B: usize, SO>(
    cfg: FuzzConfig,
    label: &str,
    meta: ReproMeta,
) where
    SO: StemStrategy + 'static,
    <SO as StemStrategy>::Stack<f64, 2>: 'static,
{
    run_immutable_case_f64_with_leaf::<K, B, SO, FlatVec<f64, usize, K, B>>(cfg, label, meta);
}

fn run_immutable_case_f64_arenas<const K: usize, const B: usize, SO>(
    cfg: FuzzConfig,
    label: &str,
    meta: ReproMeta,
) where
    SO: StemStrategy + 'static,
    <SO as StemStrategy>::Stack<f64, 2>: 'static,
{
    run_immutable_case_f64_with_leaf::<K, B, SO, VecOfArenas<f64, usize, K, B>>(cfg, label, meta);
}

#[allow(type_alias_bounds)]
type DonnellyF32<const K: usize> = Donnelly<4>;
#[allow(type_alias_bounds)]
type DonnellyF64<const K: usize> = Donnelly<3>;

#[cfg(feature = "simd")]
#[allow(type_alias_bounds)]
type DonnellySimdBlock4F32<const K: usize> = DonnellySimdFull<4>;
#[cfg(feature = "simd")]
#[allow(type_alias_bounds)]
type DonnellySimdBlock3F64<const K: usize> = DonnellySimdFull<3>;

#[cfg(feature = "simd")]
macro_rules! run_simd_matrix_f32 {
    ($runner:ident, $cfg:expr, $meta:expr, $prefix:literal, $strategy:ident, $block:literal) => {
        $runner::<2, 32, $strategy<2>>($cfg, concat!($prefix, " ", $block, " K=2 B=32"), $meta);
        $runner::<2, 64, $strategy<2>>($cfg, concat!($prefix, " ", $block, " K=2 B=64"), $meta);
        $runner::<2, 128, $strategy<2>>($cfg, concat!($prefix, " ", $block, " K=2 B=128"), $meta);
        $runner::<3, 32, $strategy<3>>($cfg, concat!($prefix, " ", $block, " K=3 B=32"), $meta);
        $runner::<3, 64, $strategy<3>>($cfg, concat!($prefix, " ", $block, " K=3 B=64"), $meta);
        $runner::<3, 128, $strategy<3>>($cfg, concat!($prefix, " ", $block, " K=3 B=128"), $meta);
        $runner::<4, 32, $strategy<4>>($cfg, concat!($prefix, " ", $block, " K=4 B=32"), $meta);
        $runner::<4, 64, $strategy<4>>($cfg, concat!($prefix, " ", $block, " K=4 B=64"), $meta);
        $runner::<4, 128, $strategy<4>>($cfg, concat!($prefix, " ", $block, " K=4 B=128"), $meta);
    };
}

#[cfg(feature = "simd")]
macro_rules! run_simd_matrix_f64 {
    ($runner:ident, $cfg:expr, $meta:expr, $prefix:literal, $strategy:ident, $block:literal) => {
        $runner::<2, 32, $strategy<2>>($cfg, concat!($prefix, " ", $block, " K=2 B=32"), $meta);
        $runner::<2, 64, $strategy<2>>($cfg, concat!($prefix, " ", $block, " K=2 B=64"), $meta);
        $runner::<2, 128, $strategy<2>>($cfg, concat!($prefix, " ", $block, " K=2 B=128"), $meta);
        $runner::<3, 32, $strategy<3>>($cfg, concat!($prefix, " ", $block, " K=3 B=32"), $meta);
        $runner::<3, 64, $strategy<3>>($cfg, concat!($prefix, " ", $block, " K=3 B=64"), $meta);
        $runner::<3, 128, $strategy<3>>($cfg, concat!($prefix, " ", $block, " K=3 B=128"), $meta);
        $runner::<4, 32, $strategy<4>>($cfg, concat!($prefix, " ", $block, " K=4 B=32"), $meta);
        $runner::<4, 64, $strategy<4>>($cfg, concat!($prefix, " ", $block, " K=4 B=64"), $meta);
        $runner::<4, 128, $strategy<4>>($cfg, concat!($prefix, " ", $block, " K=4 B=128"), $meta);
    };
}

macro_rules! run_immutable_matrix_f32 {
    ($runner:ident, $cfg:expr, $meta:expr, $prefix:literal, Eytzinger) => {
        $runner::<2, 16, Eytzinger>($cfg, concat!($prefix, " K=2 B=16"), $meta);
        $runner::<2, 32, Eytzinger>($cfg, concat!($prefix, " K=2 B=32"), $meta);
        $runner::<2, 64, Eytzinger>($cfg, concat!($prefix, " K=2 B=64"), $meta);
        $runner::<3, 16, Eytzinger>($cfg, concat!($prefix, " K=3 B=16"), $meta);
        $runner::<3, 32, Eytzinger>($cfg, concat!($prefix, " K=3 B=32"), $meta);
        $runner::<3, 64, Eytzinger>($cfg, concat!($prefix, " K=3 B=64"), $meta);
        $runner::<4, 16, Eytzinger>($cfg, concat!($prefix, " K=4 B=16"), $meta);
        $runner::<4, 32, Eytzinger>($cfg, concat!($prefix, " K=4 B=32"), $meta);
        $runner::<4, 64, Eytzinger>($cfg, concat!($prefix, " K=4 B=64"), $meta);
    };
    ($runner:ident, $cfg:expr, $meta:expr, $prefix:literal, $strategy:ident) => {
        $runner::<2, 16, $strategy<2>>($cfg, concat!($prefix, " K=2 B=16"), $meta);
        $runner::<2, 32, $strategy<2>>($cfg, concat!($prefix, " K=2 B=32"), $meta);
        $runner::<2, 64, $strategy<2>>($cfg, concat!($prefix, " K=2 B=64"), $meta);
        $runner::<3, 16, $strategy<3>>($cfg, concat!($prefix, " K=3 B=16"), $meta);
        $runner::<3, 32, $strategy<3>>($cfg, concat!($prefix, " K=3 B=32"), $meta);
        $runner::<3, 64, $strategy<3>>($cfg, concat!($prefix, " K=3 B=64"), $meta);
        $runner::<4, 16, $strategy<4>>($cfg, concat!($prefix, " K=4 B=16"), $meta);
        $runner::<4, 32, $strategy<4>>($cfg, concat!($prefix, " K=4 B=32"), $meta);
        $runner::<4, 64, $strategy<4>>($cfg, concat!($prefix, " K=4 B=64"), $meta);
    };
}

macro_rules! run_immutable_matrix_f64 {
    ($runner:ident, $cfg:expr, $meta:expr, $prefix:literal, Eytzinger) => {
        $runner::<2, 16, Eytzinger>($cfg, concat!($prefix, " K=2 B=16"), $meta);
        $runner::<2, 32, Eytzinger>($cfg, concat!($prefix, " K=2 B=32"), $meta);
        $runner::<2, 64, Eytzinger>($cfg, concat!($prefix, " K=2 B=64"), $meta);
        $runner::<3, 16, Eytzinger>($cfg, concat!($prefix, " K=3 B=16"), $meta);
        $runner::<3, 32, Eytzinger>($cfg, concat!($prefix, " K=3 B=32"), $meta);
        $runner::<3, 64, Eytzinger>($cfg, concat!($prefix, " K=3 B=64"), $meta);
        $runner::<4, 16, Eytzinger>($cfg, concat!($prefix, " K=4 B=16"), $meta);
        $runner::<4, 32, Eytzinger>($cfg, concat!($prefix, " K=4 B=32"), $meta);
        $runner::<4, 64, Eytzinger>($cfg, concat!($prefix, " K=4 B=64"), $meta);
    };
    ($runner:ident, $cfg:expr, $meta:expr, $prefix:literal, $strategy:ident) => {
        $runner::<2, 16, $strategy<2>>($cfg, concat!($prefix, " K=2 B=16"), $meta);
        $runner::<2, 32, $strategy<2>>($cfg, concat!($prefix, " K=2 B=32"), $meta);
        $runner::<2, 64, $strategy<2>>($cfg, concat!($prefix, " K=2 B=64"), $meta);
        $runner::<3, 16, $strategy<3>>($cfg, concat!($prefix, " K=3 B=16"), $meta);
        $runner::<3, 32, $strategy<3>>($cfg, concat!($prefix, " K=3 B=32"), $meta);
        $runner::<3, 64, $strategy<3>>($cfg, concat!($prefix, " K=3 B=64"), $meta);
        $runner::<4, 16, $strategy<4>>($cfg, concat!($prefix, " K=4 B=16"), $meta);
        $runner::<4, 32, $strategy<4>>($cfg, concat!($prefix, " K=4 B=32"), $meta);
        $runner::<4, 64, $strategy<4>>($cfg, concat!($prefix, " K=4 B=64"), $meta);
    };
}

#[test]
#[ignore = "long-running fuzz-style correctness checks"]
fn fuzz_v6_mutable_f32() {
    let cfg = FuzzConfig::from_env();
    let run_non_simd_paths = should_run_non_simd_paths();
    let meta = ReproMeta {
        kind: "v6_mutable",
        leaf: "vec_of_arrays",
        scalar: "f32",
        strategy: "eytzinger",
        b: 16,
        k: 2,
    };

    if run_non_simd_paths {
        run_mutable_case_f32::<2, 32, Eytzinger>(cfg, "v6 mutable f32 Eytzinger K=2 B=32", meta);
        run_mutable_case_f32::<2, 64, Eytzinger>(cfg, "v6 mutable f32 Eytzinger K=2 B=64", meta);
        run_mutable_case_f32::<2, 128, Eytzinger>(cfg, "v6 mutable f32 Eytzinger K=2 B=128", meta);
        run_mutable_case_f32::<3, 32, Eytzinger>(cfg, "v6 mutable f32 Eytzinger K=3 B=32", meta);
        run_mutable_case_f32::<3, 64, Eytzinger>(cfg, "v6 mutable f32 Eytzinger K=3 B=64", meta);
        run_mutable_case_f32::<3, 128, Eytzinger>(cfg, "v6 mutable f32 Eytzinger K=3 B=128", meta);
        run_mutable_case_f32::<4, 32, Eytzinger>(cfg, "v6 mutable f32 Eytzinger K=4 B=32", meta);
        run_mutable_case_f32::<4, 64, Eytzinger>(cfg, "v6 mutable f32 Eytzinger K=4 B=64", meta);
        run_mutable_case_f32::<4, 128, Eytzinger>(cfg, "v6 mutable f32 Eytzinger K=4 B=128", meta);

        let meta = ReproMeta {
            strategy: "donnelly",
            ..meta
        };

        run_mutable_case_f32::<2, 16, DonnellyF32<2>>(
            cfg,
            "v6 mutable f32 Donnelly K=2 B=16",
            meta,
        );
        run_mutable_case_f32::<2, 32, DonnellyF32<2>>(
            cfg,
            "v6 mutable f32 Donnelly K=2 B=32",
            meta,
        );
        run_mutable_case_f32::<2, 64, DonnellyF32<2>>(
            cfg,
            "v6 mutable f32 Donnelly K=2 B=64",
            meta,
        );
        run_mutable_case_f32::<3, 16, DonnellyF32<3>>(
            cfg,
            "v6 mutable f32 Donnelly K=3 B=16",
            meta,
        );
        run_mutable_case_f32::<3, 32, DonnellyF32<3>>(
            cfg,
            "v6 mutable f32 Donnelly K=3 B=32",
            meta,
        );
        run_mutable_case_f32::<3, 64, DonnellyF32<3>>(
            cfg,
            "v6 mutable f32 Donnelly K=3 B=64",
            meta,
        );
        run_mutable_case_f32::<4, 16, DonnellyF32<4>>(
            cfg,
            "v6 mutable f32 Donnelly K=4 B=16",
            meta,
        );
        run_mutable_case_f32::<4, 32, DonnellyF32<4>>(
            cfg,
            "v6 mutable f32 Donnelly K=4 B=32",
            meta,
        );
        run_mutable_case_f32::<4, 64, DonnellyF32<4>>(
            cfg,
            "v6 mutable f32 Donnelly K=4 B=64",
            meta,
        );
    }

    // No SIMD block-at-once matrix here: see the note in `fuzz_v6_mutable_f64`.
    // Those strategies pair only with immutable leaf strategies, and the immutable
    // sweeps cover Block4.
}

#[test]
#[ignore = "long-running fuzz-style correctness checks"]
fn fuzz_v6_mutable_f64() {
    let cfg = FuzzConfig::from_env();
    let run_non_simd_paths = should_run_non_simd_paths();
    let meta = ReproMeta {
        kind: "v6_mutable",
        leaf: "vec_of_arrays",
        scalar: "f64",
        strategy: "eytzinger",
        b: 16,
        k: 2,
    };

    if run_non_simd_paths {
        run_mutable_case_f64::<2, 32, Eytzinger>(cfg, "v6 mutable f64 Eytzinger K=2 B=32", meta);
        run_mutable_case_f64::<2, 64, Eytzinger>(cfg, "v6 mutable f64 Eytzinger K=2 B=64", meta);
        run_mutable_case_f64::<2, 128, Eytzinger>(cfg, "v6 mutable f64 Eytzinger K=2 B=128", meta);
        run_mutable_case_f64::<3, 32, Eytzinger>(cfg, "v6 mutable f64 Eytzinger K=3 B=32", meta);
        run_mutable_case_f64::<3, 64, Eytzinger>(cfg, "v6 mutable f64 Eytzinger K=3 B=64", meta);
        run_mutable_case_f64::<3, 128, Eytzinger>(cfg, "v6 mutable f64 Eytzinger K=3 B=128", meta);
        run_mutable_case_f64::<4, 32, Eytzinger>(cfg, "v6 mutable f64 Eytzinger K=4 B=32", meta);
        run_mutable_case_f64::<4, 64, Eytzinger>(cfg, "v6 mutable f64 Eytzinger K=4 B=64", meta);
        run_mutable_case_f64::<4, 128, Eytzinger>(cfg, "v6 mutable f64 Eytzinger K=4 B=128", meta);

        let meta = ReproMeta {
            strategy: "donnelly",
            ..meta
        };

        run_mutable_case_f64::<2, 16, DonnellyF64<2>>(
            cfg,
            "v6 mutable f64 Donnelly K=2 B=16",
            meta,
        );
        run_mutable_case_f64::<2, 32, DonnellyF64<2>>(
            cfg,
            "v6 mutable f64 Donnelly K=2 B=32",
            meta,
        );
        run_mutable_case_f64::<2, 64, DonnellyF64<2>>(
            cfg,
            "v6 mutable f64 Donnelly K=2 B=64",
            meta,
        );
        run_mutable_case_f64::<3, 16, DonnellyF64<3>>(
            cfg,
            "v6 mutable f64 Donnelly K=3 B=16",
            meta,
        );
        run_mutable_case_f64::<3, 32, DonnellyF64<3>>(
            cfg,
            "v6 mutable f64 Donnelly K=3 B=32",
            meta,
        );
        run_mutable_case_f64::<3, 64, DonnellyF64<3>>(
            cfg,
            "v6 mutable f64 Donnelly K=3 B=64",
            meta,
        );
        run_mutable_case_f64::<4, 16, DonnellyF64<4>>(
            cfg,
            "v6 mutable f64 Donnelly K=4 B=16",
            meta,
        );
        run_mutable_case_f64::<4, 32, DonnellyF64<4>>(
            cfg,
            "v6 mutable f64 Donnelly K=4 B=32",
            meta,
        );
        run_mutable_case_f64::<4, 64, DonnellyF64<4>>(
            cfg,
            "v6 mutable f64 Donnelly K=4 B=64",
            meta,
        );
    }

    // No SIMD block-at-once matrix here: those strategies descend a whole layout
    // block per step and cannot observe a terminal stem partway through one, so
    // they only pair with immutable leaf strategies. `StemLeafCompatibility`
    // rejects the combination at compile time, and the immutable sweeps cover
    // Block3 and Block4.
}

#[test]
#[ignore = "long-running fuzz-style correctness checks"]
fn fuzz_v6_immutable_f32() {
    let cfg = FuzzConfig::from_env();
    let run_non_simd_paths = should_run_non_simd_paths();
    #[cfg(feature = "simd")]
    let simd_cfg = cfg.for_simd();
    let meta = ReproMeta {
        kind: "v6_immutable",
        leaf: "flat_vec",
        scalar: "f32",
        strategy: "eytzinger",
        b: 16,
        k: 2,
    };

    if run_non_simd_paths {
        run_immutable_matrix_f32!(
            run_immutable_case_f32,
            cfg,
            meta,
            "v6 immutable f32 Eytzinger",
            Eytzinger
        );

        let donnelly_meta = ReproMeta {
            strategy: "donnelly",
            ..meta
        };
        run_immutable_matrix_f32!(
            run_immutable_case_f32,
            cfg,
            donnelly_meta,
            "v6 immutable f32 Donnelly",
            DonnellyF32
        );

        let arena_meta = ReproMeta {
            leaf: "vec_of_arenas",
            strategy: "eytzinger",
            ..meta
        };
        run_immutable_matrix_f32!(
            run_immutable_case_f32_arenas,
            cfg,
            arena_meta,
            "v6 immutable f32 Eytzinger VecOfArenas",
            Eytzinger
        );

        let arena_donnelly_meta = ReproMeta {
            strategy: "donnelly",
            ..arena_meta
        };
        run_immutable_matrix_f32!(
            run_immutable_case_f32_arenas,
            cfg,
            arena_donnelly_meta,
            "v6 immutable f32 Donnelly VecOfArenas",
            DonnellyF32
        );
    }

    #[cfg(feature = "simd")]
    {
        if should_run_simd_f32_paths() {
            let block4_meta = ReproMeta {
                strategy: "donnelly_simd_block4",
                ..meta
            };
            run_simd_matrix_f32!(
                run_immutable_case_f32,
                simd_cfg,
                block4_meta,
                "v6 immutable f32 DonnellySimd",
                DonnellySimdBlock4F32,
                "Block4"
            );

            let block4_arena_meta = ReproMeta {
                leaf: "vec_of_arenas",
                strategy: "donnelly_simd_block4",
                ..meta
            };
            run_simd_matrix_f32!(
                run_immutable_case_f32_arenas,
                simd_cfg,
                block4_arena_meta,
                "v6 immutable f32 DonnellySimd VecOfArenas",
                DonnellySimdBlock4F32,
                "Block4"
            );
        }
    }
}

#[test]
#[ignore = "long-running fuzz-style correctness checks"]
fn fuzz_v6_immutable_f64() {
    let cfg = FuzzConfig::from_env();
    let run_non_simd_paths = should_run_non_simd_paths();
    #[cfg(feature = "simd")]
    let run_simd_paths = should_run_simd_paths();
    #[cfg(feature = "simd")]
    let simd_cfg = cfg.for_simd();
    let meta = ReproMeta {
        kind: "v6_immutable",
        leaf: "flat_vec",
        scalar: "f64",
        strategy: "eytzinger",
        b: 16,
        k: 2,
    };

    if run_non_simd_paths {
        run_immutable_matrix_f64!(
            run_immutable_case_f64,
            cfg,
            meta,
            "v6 immutable f64 Eytzinger",
            Eytzinger
        );

        let donnelly_meta = ReproMeta {
            strategy: "donnelly",
            ..meta
        };
        run_immutable_matrix_f64!(
            run_immutable_case_f64,
            cfg,
            donnelly_meta,
            "v6 immutable f64 Donnelly",
            DonnellyF64
        );

        let arena_meta = ReproMeta {
            leaf: "vec_of_arenas",
            strategy: "eytzinger",
            ..meta
        };
        run_immutable_matrix_f64!(
            run_immutable_case_f64_arenas,
            cfg,
            arena_meta,
            "v6 immutable f64 Eytzinger VecOfArenas",
            Eytzinger
        );

        let arena_donnelly_meta = ReproMeta {
            strategy: "donnelly",
            ..arena_meta
        };
        run_immutable_matrix_f64!(
            run_immutable_case_f64_arenas,
            cfg,
            arena_donnelly_meta,
            "v6 immutable f64 Donnelly VecOfArenas",
            DonnellyF64
        );
    }

    #[cfg(feature = "simd")]
    {
        if run_simd_paths {
            let block3_meta = ReproMeta {
                strategy: "donnelly_simd_block3",
                ..meta
            };
            run_simd_matrix_f64!(
                run_immutable_case_f64,
                simd_cfg,
                block3_meta,
                "v6 immutable f64 DonnellySimd",
                DonnellySimdBlock3F64,
                "Block3"
            );

            let block3_arena_meta = ReproMeta {
                leaf: "vec_of_arenas",
                strategy: "donnelly_simd_block3",
                ..meta
            };
            run_simd_matrix_f64!(
                run_immutable_case_f64_arenas,
                simd_cfg,
                block3_arena_meta,
                "v6 immutable f64 DonnellySimd VecOfArenas",
                DonnellySimdBlock3F64,
                "Block3"
            );
        }
    }
}

const ADVERSARIAL_B: usize = 16;
const ADVERSARIAL_MAX_QTY: usize = 5;
const ADVERSARIAL_RADIUS_SQ_F32: f32 = 0.625;
const ADVERSARIAL_RADIUS_MAN_F32: f32 = 0.75;
#[cfg(feature = "simd")]
const ADVERSARIAL_RADIUS_SQ_F64: f64 = 0.625;
#[cfg(feature = "simd")]
const ADVERSARIAL_RADIUS_MAN_F64: f64 = 0.75;

const ADVERSARIAL_SIZES: [usize; 18] = [
    0,
    1,
    2,
    ADVERSARIAL_B - 1,
    ADVERSARIAL_B,
    ADVERSARIAL_B + 1,
    2 * ADVERSARIAL_B - 1,
    2 * ADVERSARIAL_B,
    2 * ADVERSARIAL_B + 1,
    4 * ADVERSARIAL_B - 1,
    4 * ADVERSARIAL_B,
    4 * ADVERSARIAL_B + 1,
    8 * ADVERSARIAL_B - 1,
    8 * ADVERSARIAL_B,
    8 * ADVERSARIAL_B + 1,
    64 * ADVERSARIAL_B - 1,
    64 * ADVERSARIAL_B,
    64 * ADVERSARIAL_B + 1,
];

#[derive(Clone, Copy)]
enum AdversarialPattern {
    Grid,
    Quantized,
    AxisDegenerate,
    AllSame,
    MonotonicDiagonal,
    TwoClusters,
    AlternatingExtremes,
    NearCoincident,
    MixedScale,
}

impl AdversarialPattern {
    fn name(self) -> &'static str {
        match self {
            AdversarialPattern::Grid => "grid",
            AdversarialPattern::Quantized => "quantized",
            AdversarialPattern::AxisDegenerate => "axis_degenerate",
            AdversarialPattern::AllSame => "all_same",
            AdversarialPattern::MonotonicDiagonal => "monotonic_diagonal",
            AdversarialPattern::TwoClusters => "two_clusters",
            AdversarialPattern::AlternatingExtremes => "alternating_extremes",
            AdversarialPattern::NearCoincident => "near_coincident",
            AdversarialPattern::MixedScale => "mixed_scale",
        }
    }

    fn id(self) -> u64 {
        match self {
            AdversarialPattern::Grid => 1,
            AdversarialPattern::Quantized => 2,
            AdversarialPattern::AxisDegenerate => 3,
            AdversarialPattern::AllSame => 4,
            AdversarialPattern::MonotonicDiagonal => 5,
            AdversarialPattern::TwoClusters => 6,
            AdversarialPattern::AlternatingExtremes => 7,
            AdversarialPattern::NearCoincident => 8,
            AdversarialPattern::MixedScale => 9,
        }
    }
}

const ADVERSARIAL_PATTERNS: [AdversarialPattern; 9] = [
    AdversarialPattern::Grid,
    AdversarialPattern::Quantized,
    AdversarialPattern::AxisDegenerate,
    AdversarialPattern::AllSame,
    AdversarialPattern::MonotonicDiagonal,
    AdversarialPattern::TwoClusters,
    AdversarialPattern::AlternatingExtremes,
    AdversarialPattern::NearCoincident,
    AdversarialPattern::MixedScale,
];

type EntryF32 = ([f32; 2], usize);
#[cfg(feature = "simd")]
type EntryF64 = ([f64; 2], usize);
type ApproxTreeF32Builder =
    fn(
        &[[f32; 2]],
    )
        -> KdTree<f32, usize, Eytzinger, FlatVec<f32, usize, 2, ADVERSARIAL_B>, 2, ADVERSARIAL_B>;

fn adversarial_queries_f32() -> [[f32; 2]; 12] {
    [
        [0.0, 0.0],
        [-0.0, 0.0],
        [f32::EPSILON, -f32::EPSILON],
        [f32::from_bits(1), -f32::from_bits(1)],
        [0.125, -0.75],
        [-0.5, 0.5],
        [0.75, -0.25],
        [1.0, 1.0],
        [-1.0, -1.0],
        [1.0 + f32::EPSILON, -1.0 - f32::EPSILON],
        [2.0, -2.0],
        [1.0e15, -1.0e15],
    ]
}

#[cfg(feature = "simd")]
fn adversarial_queries_f64() -> [[f64; 2]; 12] {
    [
        [0.0, 0.0],
        [-0.0, 0.0],
        [f64::EPSILON, -f64::EPSILON],
        [f64::from_bits(1), -f64::from_bits(1)],
        [0.125, -0.75],
        [-0.5, 0.5],
        [0.75, -0.25],
        [1.0, 1.0],
        [-1.0, -1.0],
        [1.0 + f64::EPSILON, -1.0 - f64::EPSILON],
        [2.0, -2.0],
        [1.0e150, -1.0e150],
    ]
}

fn adversarial_points_f32(size: usize, pattern: AdversarialPattern, seed: u64) -> Vec<[f32; 2]> {
    match pattern {
        AdversarialPattern::Grid => (0..size)
            .map(|i| {
                let x = ((i % 7) as f32 - 3.0) / 4.0;
                let y = (((i / 7) % 7) as f32 - 3.0) / 4.0;
                [x, y]
            })
            .collect(),
        AdversarialPattern::Quantized => {
            let mut rng = StdRng::seed_from_u64(seed ^ (pattern.id() << 20));
            (0..size)
                .map(|_| {
                    let x = (rng.random_range(-1.0f32..1.0f32) * 8.0).round() / 8.0;
                    let y = (rng.random_range(-1.0f32..1.0f32) * 8.0).round() / 8.0;
                    [x, y]
                })
                .collect()
        }
        AdversarialPattern::AxisDegenerate => (0..size)
            .map(|i| {
                let y = (((i * 13) % 29) as f32 - 14.0) / 14.0;
                [0.0, y]
            })
            .collect(),
        AdversarialPattern::AllSame => vec![[0.125, -0.75]; size],
        AdversarialPattern::MonotonicDiagonal => (0..size)
            .map(|i| {
                let t = -1.0 + 2.0 * i as f32 / size.saturating_sub(1).max(1) as f32;
                [t, t]
            })
            .collect(),
        AdversarialPattern::TwoClusters => (0..size)
            .map(|i| {
                let center = if i.is_multiple_of(2) { -0.75 } else { 0.75 };
                let jitter = (i % 7) as f32 * f32::EPSILON;
                [center + jitter, center - jitter]
            })
            .collect(),
        AdversarialPattern::AlternatingExtremes => {
            const EXTREMES: [[f32; 2]; 6] = [
                [-1.0, -1.0],
                [1.0, 1.0],
                [-1.0, 1.0],
                [1.0, -1.0],
                [0.0, -0.0],
                [-0.0, 0.0],
            ];
            (0..size).map(|i| EXTREMES[i % EXTREMES.len()]).collect()
        }
        AdversarialPattern::NearCoincident => (0..size)
            .map(|i| {
                let ulps = (i % 8) as u32;
                [
                    f32::from_bits(0.125f32.to_bits() + ulps),
                    f32::from_bits((-0.75f32).to_bits() - ulps),
                ]
            })
            .collect(),
        AdversarialPattern::MixedScale => {
            const VALUES: [f32; 8] = [
                -1.0e15,
                -1.0,
                -f32::MIN_POSITIVE,
                -f32::from_bits(1),
                f32::from_bits(1),
                f32::MIN_POSITIVE,
                1.0,
                1.0e15,
            ];
            (0..size)
                .map(|i| [VALUES[i % VALUES.len()], VALUES[(i * 5 + 3) % VALUES.len()]])
                .collect()
        }
    }
}

#[cfg(feature = "simd")]
fn adversarial_points_f64(size: usize, pattern: AdversarialPattern, seed: u64) -> Vec<[f64; 2]> {
    match pattern {
        AdversarialPattern::Grid => (0..size)
            .map(|i| {
                let x = ((i % 7) as f64 - 3.0) / 4.0;
                let y = (((i / 7) % 7) as f64 - 3.0) / 4.0;
                [x, y]
            })
            .collect(),
        AdversarialPattern::Quantized => {
            let mut rng = StdRng::seed_from_u64(seed ^ (pattern.id() << 20));
            (0..size)
                .map(|_| {
                    let x = (rng.random_range(-1.0f64..1.0f64) * 8.0).round() / 8.0;
                    let y = (rng.random_range(-1.0f64..1.0f64) * 8.0).round() / 8.0;
                    [x, y]
                })
                .collect()
        }
        AdversarialPattern::AxisDegenerate => (0..size)
            .map(|i| {
                let y = (((i * 13) % 29) as f64 - 14.0) / 14.0;
                [0.0, y]
            })
            .collect(),
        AdversarialPattern::AllSame => vec![[0.125, -0.75]; size],
        AdversarialPattern::MonotonicDiagonal => (0..size)
            .map(|i| {
                let t = -1.0 + 2.0 * i as f64 / size.saturating_sub(1).max(1) as f64;
                [t, t]
            })
            .collect(),
        AdversarialPattern::TwoClusters => (0..size)
            .map(|i| {
                let center = if i.is_multiple_of(2) { -0.75 } else { 0.75 };
                let jitter = (i % 7) as f64 * f64::EPSILON;
                [center + jitter, center - jitter]
            })
            .collect(),
        AdversarialPattern::AlternatingExtremes => {
            const EXTREMES: [[f64; 2]; 6] = [
                [-1.0, -1.0],
                [1.0, 1.0],
                [-1.0, 1.0],
                [1.0, -1.0],
                [0.0, -0.0],
                [-0.0, 0.0],
            ];
            (0..size).map(|i| EXTREMES[i % EXTREMES.len()]).collect()
        }
        AdversarialPattern::NearCoincident => (0..size)
            .map(|i| {
                let ulps = (i % 8) as u64;
                [
                    f64::from_bits(0.125f64.to_bits() + ulps),
                    f64::from_bits((-0.75f64).to_bits() - ulps),
                ]
            })
            .collect(),
        AdversarialPattern::MixedScale => {
            const VALUES: [f64; 8] = [
                -1.0e150,
                -1.0,
                -f64::MIN_POSITIVE,
                -f64::from_bits(1),
                f64::from_bits(1),
                f64::MIN_POSITIVE,
                1.0,
                1.0e150,
            ];
            (0..size)
                .map(|i| [VALUES[i % VALUES.len()], VALUES[(i * 5 + 3) % VALUES.len()]])
                .collect()
        }
    }
}

fn adversarial_mutation_point_f32(size: usize, query_idx: usize) -> [f32; 2] {
    let x = (((size + query_idx * 3) % 17) as f32 - 8.0) / 8.0;
    let y = (((size * 5 + query_idx * 7) % 19) as f32 - 9.0) / 9.0;
    [x, y]
}

fn brute_ranked_entries_f32<D>(entries: &[EntryF32], query: &[f32; 2]) -> Vec<(f32, usize)>
where
    D: DistanceMetricScalar<f32, Output = f32>,
{
    let mut ranked: Vec<(f32, usize)> = entries
        .iter()
        .map(|(point, item)| (D::dist_raw(query, point), *item))
        .collect();
    sort_by_distance_then_index(&mut ranked);
    ranked
}

#[cfg(feature = "simd")]
fn brute_ranked_entries_f64<D>(entries: &[EntryF64], query: &[f64; 2]) -> Vec<(f64, usize)>
where
    D: DistanceMetricScalar<f64, Output = f64>,
{
    let mut ranked: Vec<(f64, usize)> = entries
        .iter()
        .map(|(point, item)| (D::dist_raw(query, point), *item))
        .collect();
    sort_by_distance_then_index(&mut ranked);
    ranked
}

fn find_point_by_item_f32(entries: &[EntryF32], item: usize) -> [f32; 2] {
    entries
        .iter()
        .find(|(_, entry_item)| *entry_item == item)
        .map(|(point, _)| *point)
        .unwrap_or_else(|| panic!("item {item} not found in adversarial entry set"))
}

fn validate_tree_contents_f32<SO, LS, const B: usize>(
    tree: &KdTree<f32, usize, SO, LS, 2, B>,
    entries: &[EntryF32],
    context: &str,
) where
    SO: StemStrategy,
    LS: LeafStrategy<f32, usize, SO, 2, B>,
{
    let mut expected: Vec<(usize, [u32; 2])> = entries
        .iter()
        .map(|(point, item)| (*item, point.map(f32::to_bits)))
        .collect();
    let mut got: Vec<(usize, [u32; 2])> = tree
        .iter()
        .map(|(item, point)| (item, point.map(f32::to_bits)))
        .collect();
    expected.sort_unstable();
    got.sort_unstable();
    if got != expected {
        let differences: Vec<(usize, [f32; 2], usize, [f32; 2])> = expected
            .iter()
            .zip(got.iter())
            .filter(|(expected, got)| expected != got)
            .take(PREVIEW_LEN)
            .map(|((expected_item, expected_point), (got_item, got_point))| {
                (
                    *expected_item,
                    expected_point.map(f32::from_bits),
                    *got_item,
                    got_point.map(f32::from_bits),
                )
            })
            .collect();
        panic!(
            "{context} tree contents mismatch expected_len={} got_len={} first_differences={differences:?}",
            expected.len(),
            got.len()
        );
    }
}

#[cfg(feature = "simd")]
fn find_point_by_item_f64(entries: &[EntryF64], item: usize) -> [f64; 2] {
    entries
        .iter()
        .find(|(_, entry_item)| *entry_item == item)
        .map(|(point, _)| *point)
        .unwrap_or_else(|| panic!("item {item} not found in adversarial entry set"))
}

#[cfg(feature = "simd")]
fn validate_tree_contents_f64<SO, LS, const B: usize>(
    tree: &KdTree<f64, usize, SO, LS, 2, B>,
    entries: &[EntryF64],
    context: &str,
) where
    SO: StemStrategy,
    LS: LeafStrategy<f64, usize, SO, 2, B>,
{
    let mut expected: Vec<(usize, [u64; 2])> = entries
        .iter()
        .map(|(point, item)| (*item, point.map(f64::to_bits)))
        .collect();
    let mut got: Vec<(usize, [u64; 2])> = tree
        .iter()
        .map(|(item, point)| (item, point.map(f64::to_bits)))
        .collect();
    expected.sort_unstable();
    got.sort_unstable();
    if got != expected {
        let differences: Vec<(usize, [f64; 2], usize, [f64; 2])> = expected
            .iter()
            .zip(got.iter())
            .filter(|(expected, got)| expected != got)
            .take(PREVIEW_LEN)
            .map(|((expected_item, expected_point), (got_item, got_point))| {
                (
                    *expected_item,
                    expected_point.map(f64::from_bits),
                    *got_item,
                    got_point.map(f64::from_bits),
                )
            })
            .collect();
        panic!(
            "{context} tree contents mismatch expected_len={} got_len={} first_differences={differences:?}",
            expected.len(),
            got.len()
        );
    }
}

fn validate_result_projections_f32<SO, LS, const B: usize>(
    tree: &KdTree<f32, usize, SO, LS, 2, B>,
    entries: &[EntryF32],
    query: &[f32; 2],
    context: &str,
) where
    SO: StemStrategy + 'static,
    <SO as StemStrategy>::Stack<f32, 2>: 'static,
    LS: LeafStrategy<f32, usize, SO, 2, B>,
{
    let full = tree
        .query(query)
        .within::<SquaredEuclidean<f32>>(ADVERSARIAL_RADIUS_SQ_F32)
        .unsorted()
        .with_points()
        .execute();
    let item_only = tree
        .query(query)
        .within::<SquaredEuclidean<f32>>(ADVERSARIAL_RADIUS_SQ_F32)
        .unsorted()
        .without_distances()
        .execute();
    let distance_only = tree
        .query(query)
        .within::<SquaredEuclidean<f32>>(ADVERSARIAL_RADIUS_SQ_F32)
        .unsorted()
        .without_items()
        .execute();
    let points_only = tree
        .query(query)
        .within::<SquaredEuclidean<f32>>(ADVERSARIAL_RADIUS_SQ_F32)
        .unsorted()
        .with_points()
        .without_items()
        .without_distances()
        .execute();

    assert_eq!(
        full.len(),
        item_only.len(),
        "{context} item projection length"
    );
    assert_eq!(
        full.len(),
        distance_only.len(),
        "{context} distance projection length"
    );
    assert_eq!(
        full.len(),
        points_only.len(),
        "{context} point projection length"
    );

    for result in &full {
        let expected_point = find_point_by_item_f32(entries, result.item);
        assert_eq!(
            result.point, expected_point,
            "{context} full point projection item={}",
            result.item
        );
    }

    let mut full_items: Vec<usize> = full.iter().map(|result| result.item).collect();
    let mut projected_items: Vec<usize> = item_only.iter().map(|result| result.item).collect();
    full_items.sort_unstable();
    projected_items.sort_unstable();
    assert_eq!(
        projected_items, full_items,
        "{context} item-only projection"
    );

    let mut full_distances: Vec<f32> = full.iter().map(|result| result.distance).collect();
    let mut projected_distances: Vec<f32> =
        distance_only.iter().map(|result| result.distance).collect();
    full_distances.sort_by(|a, b| a.partial_cmp(b).expect("NaN full projected distance"));
    projected_distances.sort_by(|a, b| {
        a.partial_cmp(b)
            .expect("NaN distance-only projected distance")
    });
    for (idx, (expected, got)) in full_distances
        .iter()
        .zip(projected_distances.iter())
        .enumerate()
    {
        assert_distance_match_for_fuzz!(
            *expected,
            *got,
            "{context} distance-only projection mismatch index={idx} expected={} got={}",
            expected,
            got
        );
    }

    let mut full_points: Vec<[u32; 2]> = full
        .iter()
        .map(|result| result.point.map(f32::to_bits))
        .collect();
    let mut projected_points: Vec<[u32; 2]> = points_only
        .iter()
        .map(|result| result.point.map(f32::to_bits))
        .collect();
    full_points.sort_unstable();
    projected_points.sort_unstable();
    assert_eq!(
        projected_points, full_points,
        "{context} points-only projection"
    );

    let mut expected_zero: Vec<(f32, usize)> =
        brute_ranked_entries_f32::<SquaredEuclidean<f32>>(entries, query)
            .into_iter()
            .filter(|(distance, _)| *distance <= 0.0)
            .collect();
    let mut inclusive_zero: Vec<(f32, usize)> = tree
        .query(query)
        .within::<SquaredEuclidean<f32>>(0.0)
        .unsorted()
        .execute()
        .into_iter()
        .map(|result| (result.distance, result.item))
        .collect();
    sort_by_item_idx(&mut expected_zero);
    sort_by_item_idx(&mut inclusive_zero);
    if let Err(reason) = compare_item_sorted_results(&expected_zero, &inclusive_zero) {
        panic!("{context} zero-radius inclusive-boundary mismatch: {reason}");
    }

    let exclusive_zero = tree
        .query(query)
        .within::<SquaredEuclidean<f32>>(0.0)
        .exclusive_boundaries()
        .unsorted()
        .execute();
    assert!(
        exclusive_zero.is_empty(),
        "{context} zero-radius exclusive query returned {} results",
        exclusive_zero.len()
    );

    let mut expected_all: Vec<usize> = entries.iter().map(|(_, item)| *item).collect();
    let mut got_all: Vec<usize> = tree
        .query(query)
        .within::<SquaredEuclidean<f32>>(f32::MAX)
        .unsorted()
        .without_distances()
        .execute()
        .into_iter()
        .map(|result| result.item)
        .collect();
    expected_all.sort_unstable();
    got_all.sort_unstable();
    assert_eq!(got_all, expected_all, "{context} maximum-radius query");

    let mut expected_exclusive: Vec<(f32, usize)> =
        brute_ranked_entries_f32::<SquaredEuclidean<f32>>(entries, query)
            .into_iter()
            .filter(|(distance, _)| *distance < ADVERSARIAL_RADIUS_SQ_F32)
            .collect();
    let mut exclusive: Vec<(f32, usize)> = tree
        .query(query)
        .within::<SquaredEuclidean<f32>>(ADVERSARIAL_RADIUS_SQ_F32)
        .exclusive_boundaries()
        .unsorted()
        .execute()
        .into_iter()
        .map(|result| (result.distance, result.item))
        .collect();
    sort_by_item_idx(&mut expected_exclusive);
    sort_by_item_idx(&mut exclusive);
    if let Err(reason) = compare_item_sorted_results(&expected_exclusive, &exclusive) {
        panic!("{context} exclusive-boundary projection mismatch: {reason}");
    }
}

#[cfg(feature = "simd")]
fn validate_result_projections_f64<SO, LS, const B: usize>(
    tree: &KdTree<f64, usize, SO, LS, 2, B>,
    entries: &[EntryF64],
    query: &[f64; 2],
    context: &str,
) where
    SO: StemStrategy + 'static,
    <SO as StemStrategy>::Stack<f64, 2>: 'static,
    LS: LeafStrategy<f64, usize, SO, 2, B>,
{
    let full = tree
        .query(query)
        .within::<SquaredEuclidean<f64>>(ADVERSARIAL_RADIUS_SQ_F64)
        .unsorted()
        .with_points()
        .execute();
    let item_only = tree
        .query(query)
        .within::<SquaredEuclidean<f64>>(ADVERSARIAL_RADIUS_SQ_F64)
        .unsorted()
        .without_distances()
        .execute();
    let distance_only = tree
        .query(query)
        .within::<SquaredEuclidean<f64>>(ADVERSARIAL_RADIUS_SQ_F64)
        .unsorted()
        .without_items()
        .execute();
    let points_only = tree
        .query(query)
        .within::<SquaredEuclidean<f64>>(ADVERSARIAL_RADIUS_SQ_F64)
        .unsorted()
        .with_points()
        .without_items()
        .without_distances()
        .execute();

    assert_eq!(
        full.len(),
        item_only.len(),
        "{context} item projection length"
    );
    assert_eq!(
        full.len(),
        distance_only.len(),
        "{context} distance projection length"
    );
    assert_eq!(
        full.len(),
        points_only.len(),
        "{context} point projection length"
    );

    for result in &full {
        let expected_point = find_point_by_item_f64(entries, result.item);
        assert_eq!(
            result.point, expected_point,
            "{context} full point projection item={}",
            result.item
        );
    }

    let mut full_items: Vec<usize> = full.iter().map(|result| result.item).collect();
    let mut projected_items: Vec<usize> = item_only.iter().map(|result| result.item).collect();
    full_items.sort_unstable();
    projected_items.sort_unstable();
    assert_eq!(
        projected_items, full_items,
        "{context} item-only projection"
    );

    let mut full_distances: Vec<f64> = full.iter().map(|result| result.distance).collect();
    let mut projected_distances: Vec<f64> =
        distance_only.iter().map(|result| result.distance).collect();
    full_distances.sort_by(|a, b| a.partial_cmp(b).expect("NaN full projected distance"));
    projected_distances.sort_by(|a, b| {
        a.partial_cmp(b)
            .expect("NaN distance-only projected distance")
    });
    for (idx, (expected, got)) in full_distances
        .iter()
        .zip(projected_distances.iter())
        .enumerate()
    {
        assert_distance_match_for_fuzz!(
            *expected,
            *got,
            "{context} distance-only projection mismatch index={idx} expected={} got={}",
            expected,
            got
        );
    }

    let mut full_points: Vec<[u64; 2]> = full
        .iter()
        .map(|result| result.point.map(f64::to_bits))
        .collect();
    let mut projected_points: Vec<[u64; 2]> = points_only
        .iter()
        .map(|result| result.point.map(f64::to_bits))
        .collect();
    full_points.sort_unstable();
    projected_points.sort_unstable();
    assert_eq!(
        projected_points, full_points,
        "{context} points-only projection"
    );

    let mut expected_zero: Vec<(f64, usize)> =
        brute_ranked_entries_f64::<SquaredEuclidean<f64>>(entries, query)
            .into_iter()
            .filter(|(distance, _)| *distance <= 0.0)
            .collect();
    let mut inclusive_zero: Vec<(f64, usize)> = tree
        .query(query)
        .within::<SquaredEuclidean<f64>>(0.0)
        .unsorted()
        .execute()
        .into_iter()
        .map(|result| (result.distance, result.item))
        .collect();
    sort_by_item_idx(&mut expected_zero);
    sort_by_item_idx(&mut inclusive_zero);
    if let Err(reason) = compare_item_sorted_results(&expected_zero, &inclusive_zero) {
        panic!("{context} zero-radius inclusive-boundary mismatch: {reason}");
    }

    let exclusive_zero = tree
        .query(query)
        .within::<SquaredEuclidean<f64>>(0.0)
        .exclusive_boundaries()
        .unsorted()
        .execute();
    assert!(
        exclusive_zero.is_empty(),
        "{context} zero-radius exclusive query returned {} results",
        exclusive_zero.len()
    );

    let mut expected_all: Vec<usize> = entries.iter().map(|(_, item)| *item).collect();
    let mut got_all: Vec<usize> = tree
        .query(query)
        .within::<SquaredEuclidean<f64>>(f64::MAX)
        .unsorted()
        .without_distances()
        .execute()
        .into_iter()
        .map(|result| result.item)
        .collect();
    expected_all.sort_unstable();
    got_all.sort_unstable();
    assert_eq!(got_all, expected_all, "{context} maximum-radius query");

    let mut expected_exclusive: Vec<(f64, usize)> =
        brute_ranked_entries_f64::<SquaredEuclidean<f64>>(entries, query)
            .into_iter()
            .filter(|(distance, _)| *distance < ADVERSARIAL_RADIUS_SQ_F64)
            .collect();
    let mut exclusive: Vec<(f64, usize)> = tree
        .query(query)
        .within::<SquaredEuclidean<f64>>(ADVERSARIAL_RADIUS_SQ_F64)
        .exclusive_boundaries()
        .unsorted()
        .execute()
        .into_iter()
        .map(|result| (result.distance, result.item))
        .collect();
    sort_by_item_idx(&mut expected_exclusive);
    sort_by_item_idx(&mut exclusive);
    if let Err(reason) = compare_item_sorted_results(&expected_exclusive, &exclusive) {
        panic!("{context} exclusive-boundary projection mismatch: {reason}");
    }
}

fn validate_adversarial_tree_f32<SO, LS, const B: usize>(
    tree: &KdTree<f32, usize, SO, LS, 2, B>,
    entries: &[EntryF32],
    query: &[f32; 2],
    context: &str,
) where
    SO: StemStrategy + 'static,
    <SO as StemStrategy>::Stack<f32, 2>: 'static,
    LS: LeafStrategy<f32, usize, SO, 2, B>,
{
    let max_qty = ADVERSARIAL_MAX_QTY;
    let max_qty_nz = NonZeroUsize::new(max_qty).expect("adversarial max_qty must be non-zero");
    validate_tree_contents_f32(tree, entries, context);
    validate_result_projections_f32(tree, entries, query, context);

    let expected_sq = brute_ranked_entries_f32::<SquaredEuclidean<f32>>(entries, query);
    let expected_sq_n: Vec<(f32, usize)> = expected_sq.iter().take(max_qty).copied().collect();

    let mut got_sq_sorted: Vec<(f32, usize)> =
        query_nearest_n_sorted::<_, _, _, _, SquaredEuclidean<f32>, 2, B>(tree, query, max_qty_nz);
    sort_by_distance_then_index(&mut got_sq_sorted);
    if let Err(reason) = compare_nearest_n_sorted(&expected_sq_n, &got_sq_sorted) {
        panic!(
            "{context} metric=SquaredEuclidean op=nearest_n sorted {reason} expected={} got={}",
            format_preview(&expected_sq_n, PREVIEW_LEN),
            format_preview(&got_sq_sorted, PREVIEW_LEN)
        );
    }

    let mut got_sq_unsorted: Vec<(f32, usize)> =
        query_nearest_n_unsorted::<_, _, _, _, SquaredEuclidean<f32>, 2, B>(
            tree, query, max_qty_nz,
        );
    sort_by_distance_then_index(&mut got_sq_unsorted);
    if let Err(reason) = compare_nearest_n_sorted(&expected_sq_n, &got_sq_unsorted) {
        panic!(
            "{context} metric=SquaredEuclidean op=nearest_n unsorted {reason} expected={} got={}",
            format_preview(&expected_sq_n, PREVIEW_LEN),
            format_preview(&got_sq_unsorted, PREVIEW_LEN)
        );
    }

    let mut expected_within_sq: Vec<(f32, usize)> = expected_sq
        .iter()
        .copied()
        .filter(|(dist, _)| *dist <= ADVERSARIAL_RADIUS_SQ_F32)
        .collect();
    let mut got_within_sq: Vec<(f32, usize)> =
        query_within_unsorted::<_, _, _, _, SquaredEuclidean<f32>, 2, B>(
            tree,
            query,
            ADVERSARIAL_RADIUS_SQ_F32,
        );
    if let Err(reason) = compare_within_results(
        &mut expected_within_sq,
        &mut got_within_sq,
        ADVERSARIAL_RADIUS_SQ_F32,
    ) {
        panic!(
            "{context} metric=SquaredEuclidean op=within_unsorted {reason} expected={} got={}",
            format_preview(&expected_within_sq, PREVIEW_LEN),
            format_preview(&got_within_sq, PREVIEW_LEN)
        );
    }

    let mut got_within_sq_sorted: Vec<(f32, usize)> =
        query_within_sorted::<_, _, _, _, SquaredEuclidean<f32>, 2, B>(
            tree,
            query,
            ADVERSARIAL_RADIUS_SQ_F32,
        );
    if let Err(reason) = compare_within_results(
        &mut expected_within_sq,
        &mut got_within_sq_sorted,
        ADVERSARIAL_RADIUS_SQ_F32,
    ) {
        panic!(
            "{context} metric=SquaredEuclidean op=within {reason} expected={} got={}",
            format_preview(&expected_within_sq, PREVIEW_LEN),
            format_preview(&got_within_sq_sorted, PREVIEW_LEN)
        );
    }

    let expected_best_sq = expected_best_n_within(&expected_within_sq, max_qty);
    let mut got_best_sq: Vec<(f32, usize)> = tree
        .query(query)
        .best_n_within::<SquaredEuclidean<f32>>(ADVERSARIAL_RADIUS_SQ_F32, max_qty_nz)
        .execute()
        .into_iter()
        .map(|n| (n.distance, n.item))
        .collect();
    sort_by_item_idx(&mut got_best_sq);
    if let Err(reason) = compare_item_sorted_results(&expected_best_sq, &got_best_sq) {
        panic!(
            "{context} metric=SquaredEuclidean op=best_n_within {reason} expected={} got={}",
            format_preview(&expected_best_sq, PREVIEW_LEN),
            format_preview(&got_best_sq, PREVIEW_LEN)
        );
    }

    let expected_man = brute_ranked_entries_f32::<Manhattan<f32>>(entries, query);
    let expected_man_n: Vec<(f32, usize)> = expected_man.iter().take(max_qty).copied().collect();

    let mut got_man_sorted: Vec<(f32, usize)> =
        query_nearest_n_sorted::<_, _, _, _, Manhattan<f32>, 2, B>(tree, query, max_qty_nz);
    sort_by_distance_then_index(&mut got_man_sorted);
    if let Err(reason) = compare_nearest_n_sorted(&expected_man_n, &got_man_sorted) {
        panic!(
            "{context} metric=Manhattan op=nearest_n sorted {reason} expected={} got={}",
            format_preview(&expected_man_n, PREVIEW_LEN),
            format_preview(&got_man_sorted, PREVIEW_LEN)
        );
    }

    let mut got_man_unsorted: Vec<(f32, usize)> =
        query_nearest_n_unsorted::<_, _, _, _, Manhattan<f32>, 2, B>(tree, query, max_qty_nz);
    sort_by_distance_then_index(&mut got_man_unsorted);
    if let Err(reason) = compare_nearest_n_sorted(&expected_man_n, &got_man_unsorted) {
        panic!(
            "{context} metric=Manhattan op=nearest_n unsorted {reason} expected={} got={}",
            format_preview(&expected_man_n, PREVIEW_LEN),
            format_preview(&got_man_unsorted, PREVIEW_LEN)
        );
    }

    let mut expected_within_man: Vec<(f32, usize)> = expected_man
        .iter()
        .copied()
        .filter(|(dist, _)| *dist <= ADVERSARIAL_RADIUS_MAN_F32)
        .collect();
    let mut got_within_man: Vec<(f32, usize)> =
        query_within_unsorted::<_, _, _, _, Manhattan<f32>, 2, B>(
            tree,
            query,
            ADVERSARIAL_RADIUS_MAN_F32,
        );
    if let Err(reason) = compare_within_results(
        &mut expected_within_man,
        &mut got_within_man,
        ADVERSARIAL_RADIUS_MAN_F32,
    ) {
        panic!(
            "{context} metric=Manhattan op=within_unsorted {reason} expected={} got={}",
            format_preview(&expected_within_man, PREVIEW_LEN),
            format_preview(&got_within_man, PREVIEW_LEN)
        );
    }

    let mut got_within_man_sorted: Vec<(f32, usize)> =
        query_within_sorted::<_, _, _, _, Manhattan<f32>, 2, B>(
            tree,
            query,
            ADVERSARIAL_RADIUS_MAN_F32,
        );
    if let Err(reason) = compare_within_results(
        &mut expected_within_man,
        &mut got_within_man_sorted,
        ADVERSARIAL_RADIUS_MAN_F32,
    ) {
        panic!(
            "{context} metric=Manhattan op=within {reason} expected={} got={}",
            format_preview(&expected_within_man, PREVIEW_LEN),
            format_preview(&got_within_man_sorted, PREVIEW_LEN)
        );
    }

    let expected_best_man = expected_best_n_within(&expected_within_man, max_qty);
    let mut got_best_man: Vec<(f32, usize)> = tree
        .query(query)
        .best_n_within::<Manhattan<f32>>(ADVERSARIAL_RADIUS_MAN_F32, max_qty_nz)
        .execute()
        .into_iter()
        .map(|n| (n.distance, n.item))
        .collect();
    sort_by_item_idx(&mut got_best_man);
    if let Err(reason) = compare_item_sorted_results(&expected_best_man, &got_best_man) {
        panic!(
            "{context} metric=Manhattan op=best_n_within {reason} expected={} got={}",
            format_preview(&expected_best_man, PREVIEW_LEN),
            format_preview(&got_best_man, PREVIEW_LEN)
        );
    }

    if let Some((best_sq_dist, _)) = expected_sq.first().copied() {
        let expected_sq_items: Vec<usize> = expected_sq
            .iter()
            .take_while(|(dist, _)| *dist == best_sq_dist)
            .map(|(_, item)| *item)
            .collect();
        let got_sq = tree
            .query(query)
            .nearest_one::<SquaredEuclidean<f32>>()
            .execute();
        assert_distance_match_for_fuzz!(
            best_sq_dist,
            got_sq.distance,
            "{context} metric=SquaredEuclidean op=nearest_one distance mismatch expected={} got={}",
            best_sq_dist,
            got_sq.distance
        );
        assert!(
            expected_sq_items.contains(&got_sq.item),
            "{context} metric=SquaredEuclidean op=nearest_one item mismatch expected_one_of={expected_sq_items:?} got={}",
            got_sq.item
        );

        let approx_sq = tree
            .query(query)
            .nearest_one::<SquaredEuclidean<f32>>()
            .approx()
            .execute();
        let approx_sq_point = find_point_by_item_f32(entries, approx_sq.item);
        let approx_sq_dist =
            <SquaredEuclidean<f32> as DistanceMetricScalar<f32>>::dist_raw(query, &approx_sq_point);
        assert_distance_match_for_fuzz!(
            approx_sq_dist,
            approx_sq.distance,
            "{context} metric=SquaredEuclidean op=approx_nearest_one distance mismatch expected={} got={} item={}",
            approx_sq_dist,
            approx_sq.distance,
            approx_sq.item
        );
    }

    if let Some((best_man_dist, _)) = expected_man.first().copied() {
        let expected_man_items: Vec<usize> = expected_man
            .iter()
            .take_while(|(dist, _)| *dist == best_man_dist)
            .map(|(_, item)| *item)
            .collect();
        let got_man = tree.query(query).nearest_one::<Manhattan<f32>>().execute();
        assert_distance_match_for_fuzz!(
            best_man_dist,
            got_man.distance,
            "{context} metric=Manhattan op=nearest_one distance mismatch expected={} got={}",
            best_man_dist,
            got_man.distance
        );
        assert!(
            expected_man_items.contains(&got_man.item),
            "{context} metric=Manhattan op=nearest_one item mismatch expected_one_of={expected_man_items:?} got={}",
            got_man.item
        );

        let approx_man = tree
            .query(query)
            .nearest_one::<Manhattan<f32>>()
            .approx()
            .execute();
        let approx_man_point = find_point_by_item_f32(entries, approx_man.item);
        let approx_man_dist =
            <Manhattan<f32> as DistanceMetricScalar<f32>>::dist_raw(query, &approx_man_point);
        assert_distance_match_for_fuzz!(
            approx_man_dist,
            approx_man.distance,
            "{context} metric=Manhattan op=approx_nearest_one distance mismatch expected={} got={} item={}",
            approx_man_dist,
            approx_man.distance,
            approx_man.item
        );
    }

    let expected_cheb = brute_ranked_entries_f32::<Chebyshev<f32>>(entries, query);
    let expected_cheb_n: Vec<(f32, usize)> = expected_cheb.iter().take(max_qty).copied().collect();
    let mut got_cheb_n =
        query_nearest_n_sorted::<_, _, _, _, Chebyshev<f32>, 2, B>(tree, query, max_qty_nz);
    sort_by_distance_then_index(&mut got_cheb_n);
    if let Err(reason) = compare_nearest_n_sorted(&expected_cheb_n, &got_cheb_n) {
        panic!(
            "{context} metric=Chebyshev op=nearest_n {reason} expected={} got={}",
            format_preview(&expected_cheb_n, PREVIEW_LEN),
            format_preview(&got_cheb_n, PREVIEW_LEN)
        );
    }
    if let Some((best_cheb_dist, _)) = expected_cheb.first().copied() {
        let expected_cheb_items: Vec<usize> = expected_cheb
            .iter()
            .take_while(|(dist, _)| *dist == best_cheb_dist)
            .map(|(_, item)| *item)
            .collect();
        let got_cheb = tree.query(query).nearest_one::<Chebyshev<f32>>().execute();
        assert_distance_match_for_fuzz!(
            best_cheb_dist,
            got_cheb.distance,
            "{context} metric=Chebyshev op=nearest_one distance mismatch expected={} got={}",
            best_cheb_dist,
            got_cheb.distance
        );
        assert!(
            expected_cheb_items.contains(&got_cheb.item),
            "{context} metric=Chebyshev op=nearest_one item mismatch expected_one_of={expected_cheb_items:?} got={}",
            got_cheb.item
        );
    }
}

#[cfg(feature = "simd")]
fn validate_adversarial_tree_f64<SO, LS, const B: usize>(
    tree: &KdTree<f64, usize, SO, LS, 2, B>,
    entries: &[EntryF64],
    query: &[f64; 2],
    context: &str,
) where
    SO: StemStrategy + 'static,
    <SO as StemStrategy>::Stack<f64, 2>: 'static,
    LS: LeafStrategy<f64, usize, SO, 2, B>,
{
    let max_qty = ADVERSARIAL_MAX_QTY;
    let max_qty_nz = NonZeroUsize::new(max_qty).expect("adversarial max_qty must be non-zero");
    validate_tree_contents_f64(tree, entries, context);
    validate_result_projections_f64(tree, entries, query, context);

    let expected_sq = brute_ranked_entries_f64::<SquaredEuclidean<f64>>(entries, query);
    let expected_sq_n: Vec<(f64, usize)> = expected_sq.iter().take(max_qty).copied().collect();

    let mut got_sq_sorted: Vec<(f64, usize)> =
        query_nearest_n_sorted::<_, _, _, _, SquaredEuclidean<f64>, 2, B>(tree, query, max_qty_nz);
    sort_by_distance_then_index(&mut got_sq_sorted);
    if let Err(reason) = compare_nearest_n_sorted(&expected_sq_n, &got_sq_sorted) {
        panic!(
            "{context} metric=SquaredEuclidean op=nearest_n sorted {reason} expected={} got={}",
            format_preview(&expected_sq_n, PREVIEW_LEN),
            format_preview(&got_sq_sorted, PREVIEW_LEN)
        );
    }

    let mut got_sq_unsorted: Vec<(f64, usize)> =
        query_nearest_n_unsorted::<_, _, _, _, SquaredEuclidean<f64>, 2, B>(
            tree, query, max_qty_nz,
        );
    sort_by_distance_then_index(&mut got_sq_unsorted);
    if let Err(reason) = compare_nearest_n_sorted(&expected_sq_n, &got_sq_unsorted) {
        panic!(
            "{context} metric=SquaredEuclidean op=nearest_n unsorted {reason} expected={} got={}",
            format_preview(&expected_sq_n, PREVIEW_LEN),
            format_preview(&got_sq_unsorted, PREVIEW_LEN)
        );
    }

    let mut expected_within_sq: Vec<(f64, usize)> = expected_sq
        .iter()
        .copied()
        .filter(|(dist, _)| *dist <= ADVERSARIAL_RADIUS_SQ_F64)
        .collect();
    let mut got_within_sq: Vec<(f64, usize)> =
        query_within_unsorted::<_, _, _, _, SquaredEuclidean<f64>, 2, B>(
            tree,
            query,
            ADVERSARIAL_RADIUS_SQ_F64,
        );
    if let Err(reason) = compare_within_results(
        &mut expected_within_sq,
        &mut got_within_sq,
        ADVERSARIAL_RADIUS_SQ_F64,
    ) {
        panic!(
            "{context} metric=SquaredEuclidean op=within_unsorted {reason} expected={} got={}",
            format_preview(&expected_within_sq, PREVIEW_LEN),
            format_preview(&got_within_sq, PREVIEW_LEN)
        );
    }

    let mut got_within_sq_sorted: Vec<(f64, usize)> =
        query_within_sorted::<_, _, _, _, SquaredEuclidean<f64>, 2, B>(
            tree,
            query,
            ADVERSARIAL_RADIUS_SQ_F64,
        );
    if let Err(reason) = compare_within_results(
        &mut expected_within_sq,
        &mut got_within_sq_sorted,
        ADVERSARIAL_RADIUS_SQ_F64,
    ) {
        panic!(
            "{context} metric=SquaredEuclidean op=within {reason} expected={} got={}",
            format_preview(&expected_within_sq, PREVIEW_LEN),
            format_preview(&got_within_sq_sorted, PREVIEW_LEN)
        );
    }

    let expected_best_sq = expected_best_n_within(&expected_within_sq, max_qty);
    let mut got_best_sq: Vec<(f64, usize)> = tree
        .query(query)
        .best_n_within::<SquaredEuclidean<f64>>(ADVERSARIAL_RADIUS_SQ_F64, max_qty_nz)
        .execute()
        .into_iter()
        .map(|n| (n.distance, n.item))
        .collect();
    sort_by_item_idx(&mut got_best_sq);
    if let Err(reason) = compare_item_sorted_results(&expected_best_sq, &got_best_sq) {
        panic!(
            "{context} metric=SquaredEuclidean op=best_n_within {reason} expected={} got={}",
            format_preview(&expected_best_sq, PREVIEW_LEN),
            format_preview(&got_best_sq, PREVIEW_LEN)
        );
    }

    let expected_man = brute_ranked_entries_f64::<Manhattan<f64>>(entries, query);
    let expected_man_n: Vec<(f64, usize)> = expected_man.iter().take(max_qty).copied().collect();

    let mut got_man_sorted: Vec<(f64, usize)> =
        query_nearest_n_sorted::<_, _, _, _, Manhattan<f64>, 2, B>(tree, query, max_qty_nz);
    sort_by_distance_then_index(&mut got_man_sorted);
    if let Err(reason) = compare_nearest_n_sorted(&expected_man_n, &got_man_sorted) {
        panic!(
            "{context} metric=Manhattan op=nearest_n sorted {reason} expected={} got={}",
            format_preview(&expected_man_n, PREVIEW_LEN),
            format_preview(&got_man_sorted, PREVIEW_LEN)
        );
    }

    let mut got_man_unsorted: Vec<(f64, usize)> =
        query_nearest_n_unsorted::<_, _, _, _, Manhattan<f64>, 2, B>(tree, query, max_qty_nz);
    sort_by_distance_then_index(&mut got_man_unsorted);
    if let Err(reason) = compare_nearest_n_sorted(&expected_man_n, &got_man_unsorted) {
        panic!(
            "{context} metric=Manhattan op=nearest_n unsorted {reason} expected={} got={}",
            format_preview(&expected_man_n, PREVIEW_LEN),
            format_preview(&got_man_unsorted, PREVIEW_LEN)
        );
    }

    let mut expected_within_man: Vec<(f64, usize)> = expected_man
        .iter()
        .copied()
        .filter(|(dist, _)| *dist <= ADVERSARIAL_RADIUS_MAN_F64)
        .collect();
    let mut got_within_man: Vec<(f64, usize)> =
        query_within_unsorted::<_, _, _, _, Manhattan<f64>, 2, B>(
            tree,
            query,
            ADVERSARIAL_RADIUS_MAN_F64,
        );
    if let Err(reason) = compare_within_results(
        &mut expected_within_man,
        &mut got_within_man,
        ADVERSARIAL_RADIUS_MAN_F64,
    ) {
        panic!(
            "{context} metric=Manhattan op=within_unsorted {reason} expected={} got={}",
            format_preview(&expected_within_man, PREVIEW_LEN),
            format_preview(&got_within_man, PREVIEW_LEN)
        );
    }

    let mut got_within_man_sorted: Vec<(f64, usize)> =
        query_within_sorted::<_, _, _, _, Manhattan<f64>, 2, B>(
            tree,
            query,
            ADVERSARIAL_RADIUS_MAN_F64,
        );
    if let Err(reason) = compare_within_results(
        &mut expected_within_man,
        &mut got_within_man_sorted,
        ADVERSARIAL_RADIUS_MAN_F64,
    ) {
        panic!(
            "{context} metric=Manhattan op=within {reason} expected={} got={}",
            format_preview(&expected_within_man, PREVIEW_LEN),
            format_preview(&got_within_man_sorted, PREVIEW_LEN)
        );
    }

    let expected_best_man = expected_best_n_within(&expected_within_man, max_qty);
    let mut got_best_man: Vec<(f64, usize)> = tree
        .query(query)
        .best_n_within::<Manhattan<f64>>(ADVERSARIAL_RADIUS_MAN_F64, max_qty_nz)
        .execute()
        .into_iter()
        .map(|n| (n.distance, n.item))
        .collect();
    sort_by_item_idx(&mut got_best_man);
    if let Err(reason) = compare_item_sorted_results(&expected_best_man, &got_best_man) {
        panic!(
            "{context} metric=Manhattan op=best_n_within {reason} expected={} got={}",
            format_preview(&expected_best_man, PREVIEW_LEN),
            format_preview(&got_best_man, PREVIEW_LEN)
        );
    }

    if let Some((best_sq_dist, _)) = expected_sq.first().copied() {
        let expected_sq_items: Vec<usize> = expected_sq
            .iter()
            .take_while(|(dist, _)| *dist == best_sq_dist)
            .map(|(_, item)| *item)
            .collect();
        let got_sq = tree
            .query(query)
            .nearest_one::<SquaredEuclidean<f64>>()
            .execute();
        assert_distance_match_for_fuzz!(
            best_sq_dist,
            got_sq.distance,
            "{context} metric=SquaredEuclidean op=nearest_one distance mismatch expected={} got={}",
            best_sq_dist,
            got_sq.distance
        );
        assert!(
            expected_sq_items.contains(&got_sq.item),
            "{context} metric=SquaredEuclidean op=nearest_one item mismatch expected_one_of={expected_sq_items:?} got={}",
            got_sq.item
        );

        let approx_sq = tree
            .query(query)
            .nearest_one::<SquaredEuclidean<f64>>()
            .approx()
            .execute();
        let approx_sq_point = find_point_by_item_f64(entries, approx_sq.item);
        let approx_sq_dist =
            <SquaredEuclidean<f64> as DistanceMetricScalar<f64>>::dist_raw(query, &approx_sq_point);
        assert_distance_match_for_fuzz!(
            approx_sq_dist,
            approx_sq.distance,
            "{context} metric=SquaredEuclidean op=approx_nearest_one distance mismatch expected={} got={} item={}",
            approx_sq_dist,
            approx_sq.distance,
            approx_sq.item
        );
    }

    if let Some((best_man_dist, _)) = expected_man.first().copied() {
        let expected_man_items: Vec<usize> = expected_man
            .iter()
            .take_while(|(dist, _)| *dist == best_man_dist)
            .map(|(_, item)| *item)
            .collect();
        let got_man = tree.query(query).nearest_one::<Manhattan<f64>>().execute();
        assert_distance_match_for_fuzz!(
            best_man_dist,
            got_man.distance,
            "{context} metric=Manhattan op=nearest_one distance mismatch expected={} got={}",
            best_man_dist,
            got_man.distance
        );
        assert!(
            expected_man_items.contains(&got_man.item),
            "{context} metric=Manhattan op=nearest_one item mismatch expected_one_of={expected_man_items:?} got={}",
            got_man.item
        );

        let approx_man = tree
            .query(query)
            .nearest_one::<Manhattan<f64>>()
            .approx()
            .execute();
        let approx_man_point = find_point_by_item_f64(entries, approx_man.item);
        let approx_man_dist =
            <Manhattan<f64> as DistanceMetricScalar<f64>>::dist_raw(query, &approx_man_point);
        assert_distance_match_for_fuzz!(
            approx_man_dist,
            approx_man.distance,
            "{context} metric=Manhattan op=approx_nearest_one distance mismatch expected={} got={} item={}",
            approx_man_dist,
            approx_man.distance,
            approx_man.item
        );
    }

    let expected_cheb = brute_ranked_entries_f64::<Chebyshev<f64>>(entries, query);
    let expected_cheb_n: Vec<(f64, usize)> = expected_cheb.iter().take(max_qty).copied().collect();
    let mut got_cheb_n =
        query_nearest_n_sorted::<_, _, _, _, Chebyshev<f64>, 2, B>(tree, query, max_qty_nz);
    sort_by_distance_then_index(&mut got_cheb_n);
    if let Err(reason) = compare_nearest_n_sorted(&expected_cheb_n, &got_cheb_n) {
        panic!(
            "{context} metric=Chebyshev op=nearest_n {reason} expected={} got={}",
            format_preview(&expected_cheb_n, PREVIEW_LEN),
            format_preview(&got_cheb_n, PREVIEW_LEN)
        );
    }
    if let Some((best_cheb_dist, _)) = expected_cheb.first().copied() {
        let expected_cheb_items: Vec<usize> = expected_cheb
            .iter()
            .take_while(|(dist, _)| *dist == best_cheb_dist)
            .map(|(_, item)| *item)
            .collect();
        let got_cheb = tree.query(query).nearest_one::<Chebyshev<f64>>().execute();
        assert_distance_match_for_fuzz!(
            best_cheb_dist,
            got_cheb.distance,
            "{context} metric=Chebyshev op=nearest_one distance mismatch expected={} got={}",
            best_cheb_dist,
            got_cheb.distance
        );
        assert!(
            expected_cheb_items.contains(&got_cheb.item),
            "{context} metric=Chebyshev op=nearest_one item mismatch expected_one_of={expected_cheb_items:?} got={}",
            got_cheb.item
        );
    }
}

fn run_adversarial_immutable_f32_with_leaf<SO, LS>(label: &str)
where
    SO: StemStrategy + 'static,
    <SO as StemStrategy>::Stack<f32, 2>: 'static,
    LS: LeafStrategy<f32, usize, SO, 2, ADVERSARIAL_B>
        + ConstructibleLeafStrategy<f32, usize, SO, 2, ADVERSARIAL_B>,
{
    let queries = adversarial_queries_f32();

    for &size in &ADVERSARIAL_SIZES {
        for &pattern in &ADVERSARIAL_PATTERNS {
            if size == 0 {
                // FlatVec immutable currently does not implement `new_with_empty_leaf`.
                continue;
            }
            let seed =
                DEFAULT_SEED ^ ((size as u64) << 12) ^ (pattern.id() << 24) ^ 0x9e37_79b9_7f4a_7c15;
            let points = adversarial_points_f32(size, pattern, seed);
            let entries: Vec<EntryF32> = points
                .iter()
                .copied()
                .enumerate()
                .map(|(item, point)| (point, item))
                .collect();

            let tree: KdTree<f32, usize, SO, LS, 2, ADVERSARIAL_B> =
                KdTree::new_from_slice(&points).unwrap();

            for (query_idx, query) in queries.iter().enumerate() {
                let context = format!(
                    "{label} mode=immutable pattern={} size={} query={}",
                    pattern.name(),
                    size,
                    query_idx + 1
                );
                validate_adversarial_tree_f32(&tree, &entries, query, &context);
            }
        }
    }
}

fn run_adversarial_immutable_f32<SO>(label: &str)
where
    SO: StemStrategy + 'static,
    <SO as StemStrategy>::Stack<f32, 2>: 'static,
{
    run_adversarial_immutable_f32_with_leaf::<SO, FlatVec<f32, usize, 2, ADVERSARIAL_B>>(label);
}

fn run_adversarial_immutable_f32_arenas<SO>(label: &str)
where
    SO: StemStrategy + 'static,
    <SO as StemStrategy>::Stack<f32, 2>: 'static,
{
    run_adversarial_immutable_f32_with_leaf::<SO, VecOfArenas<f32, usize, 2, ADVERSARIAL_B>>(label);
}

fn run_adversarial_mutable_f32<SO>(label: &str)
where
    SO: StemStrategy + 'static,
    <SO as StemStrategy>::Stack<f32, 2>: 'static,
{
    let queries = adversarial_queries_f32();

    for &size in &ADVERSARIAL_SIZES {
        for &pattern in &ADVERSARIAL_PATTERNS {
            let effective_size = match pattern {
                // VecOfArrays is hard-capped; adversarial degenerate sets above B are expected to panic.
                AdversarialPattern::AxisDegenerate | AdversarialPattern::AllSame
                    if size > ADVERSARIAL_B =>
                {
                    ADVERSARIAL_B
                }
                _ => size,
            };
            let seed =
                DEFAULT_SEED ^ ((size as u64) << 16) ^ (pattern.id() << 28) ^ 0xbf58_476d_1ce4_e5b9;
            let points = adversarial_points_f32(effective_size, pattern, seed);

            let mut tree: KdTree<
                f32,
                usize,
                SO,
                VecOfArrays<f32, usize, 2, ADVERSARIAL_B>,
                2,
                ADVERSARIAL_B,
            > = KdTree::default();

            let mut entries: Vec<EntryF32> = Vec::with_capacity(points.len() + queries.len());
            for (item, point) in points.iter().copied().enumerate() {
                if tree.add(&point, item).is_ok() {
                    entries.push((point, item));
                }
            }

            let mut next_item = points.len();
            for (query_idx, query) in queries.iter().enumerate() {
                let context = format!(
                    "{label} mode=mutable pattern={} requested_size={} effective_size={} query={} before_mutation_size={}",
                    pattern.name(),
                    size,
                    effective_size,
                    query_idx + 1,
                    entries.len()
                );
                validate_adversarial_tree_f32(&tree, &entries, query, &context);

                let hard_degenerate = matches!(
                    pattern,
                    AdversarialPattern::AxisDegenerate | AdversarialPattern::AllSame
                );

                if hard_degenerate && entries.len() >= ADVERSARIAL_B && !entries.is_empty() {
                    let remove_idx = (size + query_idx * 5 + 1) % entries.len();
                    let (remove_point, remove_item) = entries.swap_remove(remove_idx);
                    tree.remove(&remove_point, remove_item);
                    validate_tree_contents_f32(
                        &tree,
                        &entries,
                        &format!("{context} after_pre_add_remove"),
                    );
                }

                let add_point = adversarial_mutation_point_f32(size, query_idx);
                if tree.add(&add_point, next_item).is_ok() {
                    entries.push((add_point, next_item));
                }
                validate_tree_contents_f32(&tree, &entries, &format!("{context} after_add"));
                next_item += 1;

                if !hard_degenerate && !entries.is_empty() {
                    let remove_idx = (size + query_idx * 5 + 1) % entries.len();
                    let (remove_point, remove_item) = entries.swap_remove(remove_idx);
                    tree.remove(&remove_point, remove_item);
                    validate_tree_contents_f32(
                        &tree,
                        &entries,
                        &format!("{context} after_post_add_remove"),
                    );
                }
            }
        }
    }
}

#[cfg(feature = "simd")]
fn run_adversarial_immutable_f64_with_leaf<SO, LS>(label: &str)
where
    SO: StemStrategy + 'static,
    <SO as StemStrategy>::Stack<f64, 2>: 'static,
    LS: LeafStrategy<f64, usize, SO, 2, ADVERSARIAL_B>
        + ConstructibleLeafStrategy<f64, usize, SO, 2, ADVERSARIAL_B>,
{
    let queries = adversarial_queries_f64();

    for &size in &ADVERSARIAL_SIZES {
        for &pattern in &ADVERSARIAL_PATTERNS {
            if size == 0 {
                // FlatVec immutable currently does not implement `new_with_empty_leaf`.
                continue;
            }
            let seed =
                DEFAULT_SEED ^ ((size as u64) << 10) ^ (pattern.id() << 18) ^ 0x94d0_49bb_1331_11eb;
            let points = adversarial_points_f64(size, pattern, seed);
            let entries: Vec<EntryF64> = points
                .iter()
                .copied()
                .enumerate()
                .map(|(item, point)| (point, item))
                .collect();

            let tree: KdTree<f64, usize, SO, LS, 2, ADVERSARIAL_B> =
                KdTree::new_from_slice(&points).unwrap();

            for (query_idx, query) in queries.iter().enumerate() {
                let context = format!(
                    "{label} mode=immutable pattern={} size={} query={}",
                    pattern.name(),
                    size,
                    query_idx + 1
                );
                validate_adversarial_tree_f64(&tree, &entries, query, &context);
            }
        }
    }
}

#[cfg(feature = "simd")]
fn run_adversarial_immutable_f64<SO>(label: &str)
where
    SO: StemStrategy + 'static,
    <SO as StemStrategy>::Stack<f64, 2>: 'static,
{
    run_adversarial_immutable_f64_with_leaf::<SO, FlatVec<f64, usize, 2, ADVERSARIAL_B>>(label);
}

#[cfg(feature = "simd")]
fn run_adversarial_immutable_f64_arenas<SO>(label: &str)
where
    SO: StemStrategy + 'static,
    <SO as StemStrategy>::Stack<f64, 2>: 'static,
{
    run_adversarial_immutable_f64_with_leaf::<SO, VecOfArenas<f64, usize, 2, ADVERSARIAL_B>>(label);
}

#[test]
fn fuzz_v6_adversarial_fast_non_simd() {
    if !should_run_non_simd_paths() {
        return;
    }

    run_adversarial_mutable_f32::<Eytzinger>("v6 adversarial non-simd f32 Eytzinger");
    run_adversarial_immutable_f32::<Eytzinger>("v6 adversarial non-simd f32 Eytzinger");
    run_adversarial_immutable_f32_arenas::<Eytzinger>(
        "v6 adversarial non-simd f32 Eytzinger VecOfArenas",
    );
    run_adversarial_mutable_f32::<DonnellyF32<2>>("v6 adversarial non-simd f32 Donnelly");
    run_adversarial_immutable_f32::<DonnellyF32<2>>("v6 adversarial non-simd f32 Donnelly");
    run_adversarial_immutable_f32_arenas::<DonnellyF32<2>>(
        "v6 adversarial non-simd f32 Donnelly VecOfArenas",
    );
}

#[test]
#[cfg(feature = "simd")]
fn fuzz_v6_adversarial_fast_simd() {
    if !should_run_simd_paths() {
        return;
    }

    if should_run_simd_f32_paths() {
        run_adversarial_immutable_f32::<DonnellySimdBlock4F32<2>>("v6 adversarial simd f32 Block4");
        run_adversarial_immutable_f32_arenas::<DonnellySimdBlock4F32<2>>(
            "v6 adversarial simd f32 Block4 VecOfArenas",
        );
    }
    run_adversarial_immutable_f64::<DonnellySimdBlock3F64<2>>("v6 adversarial simd f64 Block3");
    run_adversarial_immutable_f64_arenas::<DonnellySimdBlock3F64<2>>(
        "v6 adversarial simd f64 Block3 VecOfArenas",
    );
}

fn assert_approx_invariants_f32<SO, LS, const B: usize>(
    tree: &KdTree<f32, usize, SO, LS, 2, B>,
    entries: &[EntryF32],
    query: &[f32; 2],
    context: &str,
) where
    SO: StemStrategy + 'static,
    <SO as StemStrategy>::Stack<f32, 2>: 'static,
    LS: LeafStrategy<f32, usize, SO, 2, B>,
{
    if entries.is_empty() {
        return;
    }

    let expected_sq = brute_ranked_entries_f32::<SquaredEuclidean<f32>>(entries, query);
    let best_sq_dist = expected_sq[0].0;
    let best_sq_items: Vec<usize> = expected_sq
        .iter()
        .take_while(|(dist, _)| *dist == best_sq_dist)
        .map(|(_, item)| *item)
        .collect();

    let approx_sq = tree
        .query(query)
        .nearest_one::<SquaredEuclidean<f32>>()
        .approx()
        .execute();
    let approx_sq_point = find_point_by_item_f32(entries, approx_sq.item);
    let approx_sq_dist =
        <SquaredEuclidean<f32> as DistanceMetricScalar<f32>>::dist_raw(query, &approx_sq_point);
    assert_distance_match_for_fuzz!(
        approx_sq_dist,
        approx_sq.distance,
        "{context} metric=SquaredEuclidean op=approx_nearest_one distance mismatch expected={} got={} item={}",
        approx_sq_dist,
        approx_sq.distance,
        approx_sq.item
    );

    let exact_sq = tree
        .query(query)
        .nearest_one::<SquaredEuclidean<f32>>()
        .execute();
    assert_distance_match_for_fuzz!(
        best_sq_dist,
        exact_sq.distance,
        "{context} metric=SquaredEuclidean op=nearest_one distance mismatch expected={} got={}",
        best_sq_dist,
        exact_sq.distance
    );
    assert!(
        best_sq_items.contains(&exact_sq.item),
        "{context} metric=SquaredEuclidean op=nearest_one item mismatch expected_one_of={best_sq_items:?} got={}",
        exact_sq.item
    );

    if tree.leaf_count() == 1 {
        assert_distance_match_for_fuzz!(
            exact_sq.distance,
            approx_sq.distance,
            "{context} metric=SquaredEuclidean op=approx_single_leaf distance mismatch expected={} got={}",
            exact_sq.distance,
            approx_sq.distance
        );
        assert!(
            best_sq_items.contains(&approx_sq.item),
            "{context} metric=SquaredEuclidean op=approx_single_leaf item mismatch expected_one_of={best_sq_items:?} got={}",
            approx_sq.item
        );
    }

    let expected_man = brute_ranked_entries_f32::<Manhattan<f32>>(entries, query);
    let best_man_dist = expected_man[0].0;
    let best_man_items: Vec<usize> = expected_man
        .iter()
        .take_while(|(dist, _)| *dist == best_man_dist)
        .map(|(_, item)| *item)
        .collect();

    let approx_man = tree
        .query(query)
        .nearest_one::<Manhattan<f32>>()
        .approx()
        .execute();
    let approx_man_point = find_point_by_item_f32(entries, approx_man.item);
    let approx_man_dist =
        <Manhattan<f32> as DistanceMetricScalar<f32>>::dist_raw(query, &approx_man_point);
    assert_distance_match_for_fuzz!(
        approx_man_dist,
        approx_man.distance,
        "{context} metric=Manhattan op=approx_nearest_one distance mismatch expected={} got={} item={}",
        approx_man_dist,
        approx_man.distance,
        approx_man.item
    );

    let exact_man = tree.query(query).nearest_one::<Manhattan<f32>>().execute();
    assert_distance_match_for_fuzz!(
        best_man_dist,
        exact_man.distance,
        "{context} metric=Manhattan op=nearest_one distance mismatch expected={} got={}",
        best_man_dist,
        exact_man.distance
    );
    assert!(
        best_man_items.contains(&exact_man.item),
        "{context} metric=Manhattan op=nearest_one item mismatch expected_one_of={best_man_items:?} got={}",
        exact_man.item
    );

    if tree.leaf_count() == 1 {
        assert_distance_match_for_fuzz!(
            exact_man.distance,
            approx_man.distance,
            "{context} metric=Manhattan op=approx_single_leaf distance mismatch expected={} got={}",
            exact_man.distance,
            approx_man.distance
        );
        assert!(
            best_man_items.contains(&approx_man.item),
            "{context} metric=Manhattan op=approx_single_leaf item mismatch expected_one_of={best_man_items:?} got={}",
            approx_man.item
        );
    }
}

#[cfg(feature = "simd")]
fn assert_approx_invariants_f64<SO, LS, const B: usize>(
    tree: &KdTree<f64, usize, SO, LS, 2, B>,
    entries: &[EntryF64],
    query: &[f64; 2],
    context: &str,
) where
    SO: StemStrategy + 'static,
    <SO as StemStrategy>::Stack<f64, 2>: 'static,
    LS: LeafStrategy<f64, usize, SO, 2, B>,
{
    if entries.is_empty() {
        return;
    }

    let expected_sq = brute_ranked_entries_f64::<SquaredEuclidean<f64>>(entries, query);
    let best_sq_dist = expected_sq[0].0;
    let best_sq_items: Vec<usize> = expected_sq
        .iter()
        .take_while(|(dist, _)| *dist == best_sq_dist)
        .map(|(_, item)| *item)
        .collect();

    let approx_sq = tree
        .query(query)
        .nearest_one::<SquaredEuclidean<f64>>()
        .approx()
        .execute();
    let approx_sq_point = find_point_by_item_f64(entries, approx_sq.item);
    let approx_sq_dist =
        <SquaredEuclidean<f64> as DistanceMetricScalar<f64>>::dist_raw(query, &approx_sq_point);
    assert_distance_match_for_fuzz!(
        approx_sq_dist,
        approx_sq.distance,
        "{context} metric=SquaredEuclidean op=approx_nearest_one distance mismatch expected={} got={} item={}",
        approx_sq_dist,
        approx_sq.distance,
        approx_sq.item
    );

    let exact_sq = tree
        .query(query)
        .nearest_one::<SquaredEuclidean<f64>>()
        .execute();
    assert_distance_match_for_fuzz!(
        best_sq_dist,
        exact_sq.distance,
        "{context} metric=SquaredEuclidean op=nearest_one distance mismatch expected={} got={}",
        best_sq_dist,
        exact_sq.distance
    );
    assert!(
        best_sq_items.contains(&exact_sq.item),
        "{context} metric=SquaredEuclidean op=nearest_one item mismatch expected_one_of={best_sq_items:?} got={}",
        exact_sq.item
    );

    if tree.leaf_count() == 1 {
        assert_distance_match_for_fuzz!(
            exact_sq.distance,
            approx_sq.distance,
            "{context} metric=SquaredEuclidean op=approx_single_leaf distance mismatch expected={} got={}",
            exact_sq.distance,
            approx_sq.distance
        );
        assert!(
            best_sq_items.contains(&approx_sq.item),
            "{context} metric=SquaredEuclidean op=approx_single_leaf item mismatch expected_one_of={best_sq_items:?} got={}",
            approx_sq.item
        );
    }

    let expected_man = brute_ranked_entries_f64::<Manhattan<f64>>(entries, query);
    let best_man_dist = expected_man[0].0;
    let best_man_items: Vec<usize> = expected_man
        .iter()
        .take_while(|(dist, _)| *dist == best_man_dist)
        .map(|(_, item)| *item)
        .collect();

    let approx_man = tree
        .query(query)
        .nearest_one::<Manhattan<f64>>()
        .approx()
        .execute();
    let approx_man_point = find_point_by_item_f64(entries, approx_man.item);
    let approx_man_dist =
        <Manhattan<f64> as DistanceMetricScalar<f64>>::dist_raw(query, &approx_man_point);
    assert_distance_match_for_fuzz!(
        approx_man_dist,
        approx_man.distance,
        "{context} metric=Manhattan op=approx_nearest_one distance mismatch expected={} got={} item={}",
        approx_man_dist,
        approx_man.distance,
        approx_man.item
    );

    let exact_man = tree.query(query).nearest_one::<Manhattan<f64>>().execute();
    assert_distance_match_for_fuzz!(
        best_man_dist,
        exact_man.distance,
        "{context} metric=Manhattan op=nearest_one distance mismatch expected={} got={}",
        best_man_dist,
        exact_man.distance
    );
    assert!(
        best_man_items.contains(&exact_man.item),
        "{context} metric=Manhattan op=nearest_one item mismatch expected_one_of={best_man_items:?} got={}",
        exact_man.item
    );

    if tree.leaf_count() == 1 {
        assert_distance_match_for_fuzz!(
            exact_man.distance,
            approx_man.distance,
            "{context} metric=Manhattan op=approx_single_leaf distance mismatch expected={} got={}",
            exact_man.distance,
            approx_man.distance
        );
        assert!(
            best_man_items.contains(&approx_man.item),
            "{context} metric=Manhattan op=approx_single_leaf item mismatch expected_one_of={best_man_items:?} got={}",
            approx_man.item
        );
    }
}

fn run_approx_hard_immutable_f32<SO>(label: &str)
where
    SO: StemStrategy + 'static,
    <SO as StemStrategy>::Stack<f32, 2>: 'static,
{
    let queries = adversarial_queries_f32();
    for &size in &ADVERSARIAL_SIZES {
        if size == 0 {
            continue;
        }
        for &pattern in &ADVERSARIAL_PATTERNS {
            let seed = DEFAULT_SEED ^ ((size as u64) << 8) ^ (pattern.id() << 29) ^ 0x1234_5678;
            let points = adversarial_points_f32(size, pattern, seed);
            let entries: Vec<EntryF32> = points
                .iter()
                .copied()
                .enumerate()
                .map(|(item, point)| (point, item))
                .collect();
            let tree: KdTree<
                f32,
                usize,
                SO,
                FlatVec<f32, usize, 2, ADVERSARIAL_B>,
                2,
                ADVERSARIAL_B,
            > = KdTree::new_from_slice(&points).unwrap();

            for (query_idx, query) in queries.iter().enumerate() {
                let context = format!(
                    "{label} mode=immutable pattern={} size={} query={}",
                    pattern.name(),
                    size,
                    query_idx + 1
                );
                assert_approx_invariants_f32(&tree, &entries, query, &context);
            }
        }
    }
}

fn run_approx_hard_mutable_f32<SO>(label: &str)
where
    SO: StemStrategy + 'static,
    <SO as StemStrategy>::Stack<f32, 2>: 'static,
{
    let queries = adversarial_queries_f32();
    for &size in &ADVERSARIAL_SIZES {
        for &pattern in &ADVERSARIAL_PATTERNS {
            let effective_size = match pattern {
                AdversarialPattern::AxisDegenerate | AdversarialPattern::AllSame
                    if size > ADVERSARIAL_B =>
                {
                    ADVERSARIAL_B
                }
                _ => size,
            };
            let seed = DEFAULT_SEED ^ ((size as u64) << 20) ^ (pattern.id() << 26) ^ 0x89ab_cdef;
            let points = adversarial_points_f32(effective_size, pattern, seed);

            let mut tree: KdTree<
                f32,
                usize,
                SO,
                VecOfArrays<f32, usize, 2, ADVERSARIAL_B>,
                2,
                ADVERSARIAL_B,
            > = KdTree::default();

            let mut entries: Vec<EntryF32> = Vec::with_capacity(points.len() + queries.len());
            for (item, point) in points.iter().copied().enumerate() {
                if tree.add(&point, item).is_ok() {
                    entries.push((point, item));
                }
            }

            let mut next_item = points.len();
            for (query_idx, query) in queries.iter().enumerate() {
                let context = format!(
                    "{label} mode=mutable pattern={} requested_size={} effective_size={} query={} size={}",
                    pattern.name(),
                    size,
                    effective_size,
                    query_idx + 1,
                    entries.len()
                );
                assert_approx_invariants_f32(&tree, &entries, query, &context);

                let hard_degenerate = matches!(
                    pattern,
                    AdversarialPattern::AxisDegenerate | AdversarialPattern::AllSame
                );
                if hard_degenerate && entries.len() >= ADVERSARIAL_B && !entries.is_empty() {
                    let remove_idx = (size + query_idx * 5 + 3) % entries.len();
                    let (remove_point, remove_item) = entries.swap_remove(remove_idx);
                    tree.remove(&remove_point, remove_item);
                    validate_tree_contents_f32(
                        &tree,
                        &entries,
                        &format!("{context} after_pre_add_remove"),
                    );
                }

                let add_point = adversarial_mutation_point_f32(size + 31, query_idx);
                if tree.add(&add_point, next_item).is_ok() {
                    entries.push((add_point, next_item));
                }
                validate_tree_contents_f32(&tree, &entries, &format!("{context} after_add"));
                next_item += 1;

                if !hard_degenerate && !entries.is_empty() {
                    let remove_idx = (size + query_idx * 7 + 1) % entries.len();
                    let (remove_point, remove_item) = entries.swap_remove(remove_idx);
                    tree.remove(&remove_point, remove_item);
                    validate_tree_contents_f32(
                        &tree,
                        &entries,
                        &format!("{context} after_post_add_remove"),
                    );
                }
            }
        }
    }
}

#[cfg(feature = "simd")]
fn run_approx_hard_immutable_f64<SO>(label: &str)
where
    SO: StemStrategy + 'static,
    <SO as StemStrategy>::Stack<f64, 2>: 'static,
{
    let queries = adversarial_queries_f64();
    for &size in &ADVERSARIAL_SIZES {
        if size == 0 {
            continue;
        }
        for &pattern in &ADVERSARIAL_PATTERNS {
            let seed = DEFAULT_SEED ^ ((size as u64) << 5) ^ (pattern.id() << 23) ^ 0x0fed_cba9;
            let points = adversarial_points_f64(size, pattern, seed);
            let entries: Vec<EntryF64> = points
                .iter()
                .copied()
                .enumerate()
                .map(|(item, point)| (point, item))
                .collect();
            let tree: KdTree<
                f64,
                usize,
                SO,
                FlatVec<f64, usize, 2, ADVERSARIAL_B>,
                2,
                ADVERSARIAL_B,
            > = KdTree::new_from_slice(&points).unwrap();

            for (query_idx, query) in queries.iter().enumerate() {
                let context = format!(
                    "{label} mode=immutable pattern={} size={} query={}",
                    pattern.name(),
                    size,
                    query_idx + 1
                );
                assert_approx_invariants_f64(&tree, &entries, query, &context);
            }
        }
    }
}

#[derive(Default)]
struct ApproxRatioStats {
    sq_ratios: Vec<f64>,
    man_ratios: Vec<f64>,
}

impl ApproxRatioStats {
    fn push_sq(&mut self, value: f64) {
        self.sq_ratios.push(value);
    }

    fn push_man(&mut self, value: f64) {
        self.man_ratios.push(value);
    }
}

fn quantile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return 1.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("NaN ratio"));
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn record_approx_quality_case_f32<SO, const B: usize>(
    label: &str,
    tree: &KdTree<f32, usize, SO, FlatVec<f32, usize, 2, B>, 2, B>,
    entries: &[EntryF32],
    queries: &[[f32; 2]],
    stats: &mut ApproxRatioStats,
) where
    SO: StemStrategy,
{
    if entries.is_empty() {
        return;
    }

    for query in queries {
        let exact_sq =
            brute_ranked_entries_f32::<SquaredEuclidean<f32>>(entries, query)[0].0 as f64;
        let exact_man = brute_ranked_entries_f32::<Manhattan<f32>>(entries, query)[0].0 as f64;
        let approx_sq = tree
            .query(query)
            .nearest_one::<SquaredEuclidean<f32>>()
            .approx()
            .execute()
            .distance as f64;
        let approx_man = tree
            .query(query)
            .nearest_one::<Manhattan<f32>>()
            .approx()
            .execute()
            .distance as f64;

        let sq_ratio = if exact_sq == 0.0 {
            if approx_sq == 0.0 {
                1.0
            } else {
                f64::INFINITY
            }
        } else {
            approx_sq / exact_sq
        };
        let man_ratio = if exact_man == 0.0 {
            if approx_man == 0.0 {
                1.0
            } else {
                f64::INFINITY
            }
        } else {
            approx_man / exact_man
        };
        stats.push_sq(sq_ratio);
        stats.push_man(man_ratio);
    }

    println!(
        "{label}: approx quality samples={} sq_p95={:.4} sq_p99={:.4} sq_worst={:.4} man_p95={:.4} man_p99={:.4} man_worst={:.4}",
        stats.sq_ratios.len(),
        quantile(&stats.sq_ratios, 0.95),
        quantile(&stats.sq_ratios, 0.99),
        quantile(&stats.sq_ratios, 1.0),
        quantile(&stats.man_ratios, 0.95),
        quantile(&stats.man_ratios, 0.99),
        quantile(&stats.man_ratios, 1.0)
    );
}

#[cfg(feature = "simd")]
fn record_approx_quality_case_f64<SO, const B: usize>(
    label: &str,
    tree: &KdTree<f64, usize, SO, FlatVec<f64, usize, 2, B>, 2, B>,
    entries: &[EntryF64],
    queries: &[[f64; 2]],
    stats: &mut ApproxRatioStats,
) where
    SO: StemStrategy,
{
    if entries.is_empty() {
        return;
    }

    for query in queries {
        let exact_sq = brute_ranked_entries_f64::<SquaredEuclidean<f64>>(entries, query)[0].0;
        let exact_man = brute_ranked_entries_f64::<Manhattan<f64>>(entries, query)[0].0;
        let approx_sq = tree
            .query(query)
            .nearest_one::<SquaredEuclidean<f64>>()
            .approx()
            .execute()
            .distance;
        let approx_man = tree
            .query(query)
            .nearest_one::<Manhattan<f64>>()
            .approx()
            .execute()
            .distance;

        let sq_ratio = if exact_sq == 0.0 {
            if approx_sq == 0.0 {
                1.0
            } else {
                f64::INFINITY
            }
        } else {
            approx_sq / exact_sq
        };
        let man_ratio = if exact_man == 0.0 {
            if approx_man == 0.0 {
                1.0
            } else {
                f64::INFINITY
            }
        } else {
            approx_man / exact_man
        };
        stats.push_sq(sq_ratio);
        stats.push_man(man_ratio);
    }

    println!(
        "{label}: approx quality samples={} sq_p95={:.4} sq_p99={:.4} sq_worst={:.4} man_p95={:.4} man_p99={:.4} man_worst={:.4}",
        stats.sq_ratios.len(),
        quantile(&stats.sq_ratios, 0.95),
        quantile(&stats.sq_ratios, 0.99),
        quantile(&stats.sq_ratios, 1.0),
        quantile(&stats.man_ratios, 0.95),
        quantile(&stats.man_ratios, 0.99),
        quantile(&stats.man_ratios, 1.0)
    );
}

#[test]
fn fuzz_v6_approx_nearest_one_fast_hard() {
    if should_run_non_simd_paths() {
        run_approx_hard_immutable_f32::<Eytzinger>("v6 approx hard non-simd f32 Eytzinger");
        run_approx_hard_mutable_f32::<Eytzinger>("v6 approx hard non-simd f32 Eytzinger");
        run_approx_hard_immutable_f32::<DonnellyF32<2>>("v6 approx hard non-simd f32 Donnelly");
        run_approx_hard_mutable_f32::<DonnellyF32<2>>("v6 approx hard non-simd f32 Donnelly");
    }

    #[cfg(feature = "simd")]
    {
        if should_run_simd_paths() {
            if should_run_simd_f32_paths() {
                run_approx_hard_immutable_f32::<DonnellySimdBlock4F32<2>>(
                    "v6 approx hard simd f32 Block4",
                );
            }
            run_approx_hard_immutable_f64::<DonnellySimdBlock3F64<2>>(
                "v6 approx hard simd f64 Block3",
            );
        }
    }
}

#[test]
#[ignore = "long-running approx quality sweep; enable with --ignored"]
fn fuzz_v6_approx_nearest_one_quality() {
    let cases = read_env_usize("KIDDO_FUZZ_APPROX_CASES", 4).max(1);
    let query_count = read_env_usize("KIDDO_FUZZ_APPROX_QUERY_COUNT", 128).max(1);
    let enforce = read_env_bool("KIDDO_FUZZ_APPROX_ENFORCE", false);
    let max_p99 = env::var("KIDDO_FUZZ_APPROX_MAX_P99")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(8.0);
    let max_worst = env::var("KIDDO_FUZZ_APPROX_MAX_WORST")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(32.0);

    let mut rng = StdRng::seed_from_u64(DEFAULT_SEED ^ 0x55aa_33cc);

    let mut run_f32 = |label: &str, build_tree: ApproxTreeF32Builder| {
        let mut stats = ApproxRatioStats::default();
        for case_idx in 0..cases {
            let size = 1usize << rng.random_range(8..=12);
            let points: Vec<[f32; 2]> =
                (0..size).map(|_| random_point_f32::<2>(&mut rng)).collect();
            let entries: Vec<EntryF32> = points
                .iter()
                .copied()
                .enumerate()
                .map(|(item, point)| (point, item))
                .collect();
            let queries: Vec<[f32; 2]> = (0..query_count)
                .map(|_| random_point_f32::<2>(&mut rng))
                .collect();
            let tree = build_tree(&points);
            record_approx_quality_case_f32(
                &format!("{label} case={}", case_idx + 1),
                &tree,
                &entries,
                &queries,
                &mut stats,
            );
        }

        let p99_sq = quantile(&stats.sq_ratios, 0.99);
        let worst_sq = quantile(&stats.sq_ratios, 1.0);
        let p99_man = quantile(&stats.man_ratios, 0.99);
        let worst_man = quantile(&stats.man_ratios, 1.0);
        if enforce {
            assert!(
                p99_sq <= max_p99 && p99_man <= max_p99 && worst_sq <= max_worst && worst_man <= max_worst,
                "{label}: approx quality threshold failed sq_p99={p99_sq:.4} sq_worst={worst_sq:.4} man_p99={p99_man:.4} man_worst={worst_man:.4} max_p99={max_p99:.4} max_worst={max_worst:.4}"
            );
        }
    };

    if should_run_non_simd_paths() {
        run_f32("v6 approx quality non-simd f32 Eytzinger", |points| {
            KdTree::<
                f32,
                usize,
                Eytzinger,
                FlatVec<f32, usize, 2, ADVERSARIAL_B>,
                2,
                ADVERSARIAL_B,
            >::new_from_slice(points)
            .unwrap()
        });
    }

    #[cfg(feature = "simd")]
    {
        if should_run_simd_paths() {
            let mut stats_f64_simd = ApproxRatioStats::default();
            let run_f32_simd = should_run_simd_f32_paths();
            let mut stats_f32_simd = run_f32_simd.then(ApproxRatioStats::default);

            for case_idx in 0..cases {
                let size = 1usize << rng.random_range(8..=12);
                if run_f32_simd {
                    let points_f32: Vec<[f32; 2]> =
                        (0..size).map(|_| random_point_f32::<2>(&mut rng)).collect();
                    let entries_f32: Vec<EntryF32> = points_f32
                        .iter()
                        .copied()
                        .enumerate()
                        .map(|(item, point)| (point, item))
                        .collect();
                    let queries_f32: Vec<[f32; 2]> = (0..query_count)
                        .map(|_| random_point_f32::<2>(&mut rng))
                        .collect();
                    let tree_f32: KdTree<
                        f32,
                        usize,
                        DonnellySimdBlock4F32<2>,
                        FlatVec<f32, usize, 2, ADVERSARIAL_B>,
                        2,
                        ADVERSARIAL_B,
                    > = KdTree::new_from_slice(&points_f32).unwrap();
                    record_approx_quality_case_f32(
                        &format!("v6 approx quality simd f32 Block4 case={}", case_idx + 1),
                        &tree_f32,
                        &entries_f32,
                        &queries_f32,
                        stats_f32_simd
                            .as_mut()
                            .expect("f32 SIMD stats must exist when f32 SIMD is enabled"),
                    );
                }

                let points_f64: Vec<[f64; 2]> =
                    (0..size).map(|_| random_point_f64::<2>(&mut rng)).collect();
                let entries_f64: Vec<EntryF64> = points_f64
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(item, point)| (point, item))
                    .collect();
                let queries_f64: Vec<[f64; 2]> = (0..query_count)
                    .map(|_| random_point_f64::<2>(&mut rng))
                    .collect();
                let tree_f64: KdTree<
                    f64,
                    usize,
                    DonnellySimdBlock3F64<2>,
                    FlatVec<f64, usize, 2, ADVERSARIAL_B>,
                    2,
                    ADVERSARIAL_B,
                > = KdTree::new_from_slice(&points_f64).unwrap();
                record_approx_quality_case_f64(
                    &format!("v6 approx quality simd f64 Block3 case={}", case_idx + 1),
                    &tree_f64,
                    &entries_f64,
                    &queries_f64,
                    &mut stats_f64_simd,
                );
            }

            let p99_sq_f64 = quantile(&stats_f64_simd.sq_ratios, 0.99);
            let worst_sq_f64 = quantile(&stats_f64_simd.sq_ratios, 1.0);
            let p99_man_f64 = quantile(&stats_f64_simd.man_ratios, 0.99);
            let worst_man_f64 = quantile(&stats_f64_simd.man_ratios, 1.0);

            if enforce {
                if let Some(stats_f32_simd) = stats_f32_simd.as_ref() {
                    let p99_sq_f32 = quantile(&stats_f32_simd.sq_ratios, 0.99);
                    let worst_sq_f32 = quantile(&stats_f32_simd.sq_ratios, 1.0);
                    let p99_man_f32 = quantile(&stats_f32_simd.man_ratios, 0.99);
                    let worst_man_f32 = quantile(&stats_f32_simd.man_ratios, 1.0);
                    assert!(
                        p99_sq_f32 <= max_p99
                            && p99_man_f32 <= max_p99
                            && worst_sq_f32 <= max_worst
                            && worst_man_f32 <= max_worst,
                        "v6 approx quality simd f32 thresholds failed sq_p99={p99_sq_f32:.4} sq_worst={worst_sq_f32:.4} man_p99={p99_man_f32:.4} man_worst={worst_man_f32:.4} max_p99={max_p99:.4} max_worst={max_worst:.4}"
                    );
                }
                assert!(
                    p99_sq_f64 <= max_p99
                        && p99_man_f64 <= max_p99
                        && worst_sq_f64 <= max_worst
                        && worst_man_f64 <= max_worst,
                    "v6 approx quality simd f64 thresholds failed sq_p99={p99_sq_f64:.4} sq_worst={worst_sq_f64:.4} man_p99={p99_man_f64:.4} man_worst={worst_man_f64:.4} max_p99={max_p99:.4} max_worst={max_worst:.4}"
                );
            }
        }
    }
}

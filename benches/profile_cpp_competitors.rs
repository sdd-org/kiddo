#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![cfg_attr(coverage_nightly, coverage(off))]

//! Query-only comparison against non-Rust k-d tree / nearest-neighbour
//! libraries, mirroring `profile_external_kd_trees.rs`'s methodology for the
//! Rust competitors: the same seeded, uniformly distributed 3D `f32`/`f64`
//! points and queries, tree construction outside Criterion's timed regions,
//! and each library's own exact-query API (eps=0 wherever a library exposes
//! an error-bound knob).
//!
//! - **nanoflann**: full coverage (nearest_one, nearest_n, within_radius,
//!   f32 + f64). ANN, libkdtree++, and FASTANN were tried and dropped: ANN
//!   and libkdtree++ were both substantially slower than nanoflann with no
//!   offsetting advantage, and FASTANN's "exact" mode turned out to be an
//!   O(N) brute-force linear scan rather than a tree, making it a poor
//!   comparison point.
//! - **ALGLIB** (free C++ edition, GPL 2+, vendored source-only for local
//!   benchmarking): full coverage but f64-only, since ALGLIB's `real` type
//!   is hard-coded to `double` throughout, same limitation as ANN had.
//! - **Pkd-tree** (ucrparlay/Pkd-tree, SIGMOD'25, ParlayLib fork-join
//!   parallelism): nearest_one/nearest_n only, f32 + f64. No within_radius
//!   -- its range query is another raw box-composition API of similar
//!   complexity to k_nearest, not worth the added risk here. Benchmarked in
//!   two genuinely different, non-comparable ways: a single-threaded
//!   sequential per-query mode (`pkdtree_*` in the main
//!   `profile_cpp_competitors` groups, comparable to every other
//!   competitor's numbers) and a batch-parallel mode that submits every
//!   query at once via `parlay::parallel_for` across all cores (in the
//!   separate `profile_pkdtree_batch` groups) -- the latter is Pkd-tree's
//!   actual design intent, but is a batch-throughput metric, not a
//!   per-query latency one, so it gets its own chart rather than a line on
//!   the shared one.
//!
//! `profile_pkdtree_batch` also carries kiddo's own native
//! `kiddo::batch::Executor::serial()`/`parallel()` numbers (via
//! `KdTree::query_batch`, no FFI needed) alongside Pkd-tree's batch mode, so
//! the two genuinely parallel-capable structures land on the same
//! batch-throughput chart.
//!
//! `kiddo_vs_pkdtree_single` (`profile_kiddo_vs_pkdtree` groups) is a
//! dedicated two-library, single-threaded per-query comparison, kept
//! separate from `profile_cpp_competitors` so it can be run at larger tree
//! sizes without nanoflann/ALGLIB also having to build and hold a tree that
//! size in memory at the same time. It reads its own
//! `KIDDO_LARGE_MIN/MAX_LOG2_POINTS` for that reason. 2^24 f64 peaks at
//! around 2 GB resident; sizes beyond that were what exhausted memory
//! previously, so raise the range deliberately rather than by default.
//!
//! Vendored C++ sources are fetched at build time behind the
//! `cpp_competitors` feature; see `build_cpp_competitors.rs`.
//!
//! # Selecting work
//!
//! Every `criterion_group!` function body runs on every invocation:
//! Criterion's `--` filter only decides which `bench_function` results get
//! recorded, not which surrounding Rust setup code executes. Selecting with
//! that filter alone therefore still generates every point cloud and builds
//! every tree, which is what made full-range runs exhaust memory. Selection
//! is done up front here instead, before anything is allocated:
//!
//! - `KIDDO_CPP_SUITES`: `cpp_competitors`, `pkdtree_batch`,
//!   `kiddo_vs_pkdtree`, `pkdtree_probe`. The probe is a temporary
//!   diagnostic and is opt-in; the other three are the default.
//! - `KIDDO_CPP_LIBRARIES`: `nanoflann`, `alglib`, `pkdtree`, `kiddo`. A
//!   library name covers all of its modes, so `pkdtree` selects both the
//!   sequential and the batch-parallel Pkd-tree benchmarks.
//! - `KIDDO_CPP_SCALARS`: `f32`, `f64`.
//!
//! Each is a comma-separated list, unset means "all of the defaults", and an
//! unrecognised name is an error rather than a silently empty run. A point
//! cloud is generated, and a tree built, only once something still selected
//! actually needs it.

use criterion::measurement::WallTime;
use criterion::{
    criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion, Throughput,
};
use kiddo::batch::Executor;
use kiddo::kd_tree::KdTree;
use kiddo::leaf_strategy::FlatVec;
use kiddo::stem_strategy::{DonnellyCyclicSimdDescent, DonnellyUnrolled, Eytzinger};
use kiddo::SquaredEuclidean;
use kiddo::StemStrategy;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::ffi::c_void;
use std::hint::black_box;
use std::num::NonZeroUsize;

const K: usize = 3;
const DEFAULT_QUERY_COUNT: usize = 1_000;
const DEFAULT_MIN_LOG2_POINTS: u32 = 16;
const DEFAULT_MAX_LOG2_POINTS: u32 = 24;
const DEFAULT_RADIUS: f64 = 0.05;
// Match the focused Kiddo suites and profile_external_kd_trees so all
// separately collected results are comparable.
const POINT_SEED: u64 = 0x5eed_0000_0000_0001;
const QUERY_SEED: u64 = 0x5eed_0000_0000_0002;
const MAX_QTYS: [usize; 3] = [5, 20, 50];
// Pkd-tree has no dedicated single-nearest entry point, so its nearest_one
// series is its k-nearest batch call with k=1. kiddo's is the real
// nearest_one, whose flat one-result-per-query collection is the whole reason
// it exists as a separate query.
const NEAREST_ONE_K: usize = 1;
// Generous cap on how many within_radius matches are copied out through FFI;
// the true match count is always returned even if it exceeds this.
const RADIUS_CAP: usize = 4096;

const SUITE_CPP_COMPETITORS: &str = "cpp_competitors";
const SUITE_PKDTREE_BATCH: &str = "pkdtree_batch";
const SUITE_KIDDO_VS_PKDTREE: &str = "kiddo_vs_pkdtree";
const SUITE_PKDTREE_PROBE: &str = "pkdtree_probe";
const ALL_SUITES: [&str; 4] = [
    SUITE_CPP_COMPETITORS,
    SUITE_PKDTREE_BATCH,
    SUITE_KIDDO_VS_PKDTREE,
    SUITE_PKDTREE_PROBE,
];
// The probe is a temporary diagnostic, so it is opt-in rather than part of an
// unqualified run.
const DEFAULT_SUITES: [&str; 3] = [
    SUITE_CPP_COMPETITORS,
    SUITE_PKDTREE_BATCH,
    SUITE_KIDDO_VS_PKDTREE,
];
const NANOFLANN: &str = "nanoflann";
const ALGLIB: &str = "alglib";
const PKDTREE: &str = "pkdtree";
const KIDDO: &str = "kiddo";
const SKDTREE: &str = "skdtree";
const ALL_LIBRARIES: [&str; 5] = [NANOFLANN, ALGLIB, PKDTREE, KIDDO, SKDTREE];
const F32: &str = "f32";
const F64: &str = "f64";
const ALL_SCALARS: [&str; 2] = [F32, F64];

// Pkd-tree's build_recursive has a box-containment assertion that fails for
// f32 from 2^22 points upward: 2^21 builds, 2^22 and 2^23 abort, and f64 is
// unaffected at least through 2^24. The assertion aborts the process rather
// than returning an error, which would take every later benchmark in the same
// run down with it, so oversized f32 Pkd-tree cells are skipped up front.
const PKDTREE_MAX_LOG2_POINTS_F32: u32 = 21;

/// Asserts that the experimental Donnelly cyclic SIMD descent answers exactly
/// as the Eytzinger baseline does on this tree, so a throughput win can never
/// be a wrong-answer win. Checked once per run, at the smallest size.
fn validate_kiddo_strategies_f64(points: &[PointF64], queries: &[PointF64]) {
    let eytzinger: KiddoF64 = KdTree::new_from_slice(points).unwrap();
    let donnelly: KiddoF64<DonnellyCyclicSimdDescent<3>> = KdTree::new_from_slice(points).unwrap();
    let max_qty = NonZeroUsize::new(*MAX_QTYS.last().unwrap()).unwrap();

    for (index, query) in queries.iter().enumerate() {
        let expected = eytzinger
            .query(query)
            .nearest_one::<SquaredEuclidean<f64>>()
            .execute();
        let actual = donnelly
            .query(query)
            .nearest_one::<SquaredEuclidean<f64>>()
            .execute();
        assert_eq!(
            (actual.item, actual.distance),
            (expected.item, expected.distance),
            "Donnelly cyclic SIMD nearest_one disagrees with Eytzinger at query {index}"
        );

        let expected = eytzinger
            .query(query)
            .nearest_n::<SquaredEuclidean<f64>>(max_qty)
            .execute();
        let actual = donnelly
            .query(query)
            .nearest_n::<SquaredEuclidean<f64>>(max_qty)
            .execute();
        let expected: Vec<_> = expected.iter().map(|r| (r.item, r.distance)).collect();
        let actual: Vec<_> = actual.iter().map(|r| (r.item, r.distance)).collect();
        assert_eq!(
            actual, expected,
            "Donnelly cyclic SIMD nearest_n disagrees with Eytzinger at query {index}"
        );
    }
}

/// The executors charted side by side in the batch-throughput suite.
fn kiddo_batch_executors() -> [(&'static str, Executor); 3] {
    [
        ("serial", Executor::serial()),
        ("parallel", Executor::parallel()),
        // What a caller who has read the tuning guidance would actually
        // configure, and therefore the number kiddo should be judged on.
        // `parallel` stays in as the untuned control.
        (
            "tuned",
            Executor::parallel()
                .with_default_static_chunking()
                .with_serial_fallback(),
        ),
    ]
}

fn pkdtree_supports_f32(point_count: usize) -> bool {
    point_count <= 1usize << PKDTREE_MAX_LOG2_POINTS_F32
}

/// Whether Pkd-tree should be benchmarked for this f32 cell, warning once per
/// skipped size so an absent series is never mistaken for a missing run.
fn pkdtree_f32_selected(libraries: &Selection, point_count: usize) -> bool {
    if !libraries.contains(PKDTREE) {
        return false;
    }
    if !pkdtree_supports_f32(point_count) {
        eprintln!(
            "skipping Pkd-tree f32 at {point_count} points: upstream build assertion fails above 2^{PKDTREE_MAX_LOG2_POINTS_F32}"
        );
        return false;
    }
    true
}

type PointF32 = [f32; K];
type PointF64 = [f64; K];
const B: usize = 32;
type KiddoF32<S = Eytzinger> = KdTree<f32, u32, S, FlatVec<f32, u32, K, B>, K, B>;
type KiddoF64<S = Eytzinger> = KdTree<f64, u32, S, FlatVec<f64, u32, K, B>, K, B>;
// DonnellyCyclicSimdDescent's AVX-512 path asserts BH == K, and is implemented
// only for f64/3D/BH3 and f32/4D/BH4. These benchmarks are 3D for every
// library, so f64 gets the SIMD descent and f32 (K=3) cannot have it at all.

mod ffi {
    use std::ffi::c_void;

    #[allow(clippy::duplicated_attributes)]
    #[link(name = "nanoflann_shim", kind = "static")]
    #[link(name = "alglib_shim", kind = "static")]
    #[link(name = "pkdtree_shim", kind = "static")]
    #[link(name = "skdtree_shim", kind = "static")]
    #[link(name = "stdc++")]
    #[link(name = "pthread")]
    extern "C" {
        pub fn nanoflann_build_f32(points: *const f32, n: u64, dim: u32) -> *mut c_void;
        pub fn nanoflann_build_f64(points: *const f64, n: u64, dim: u32) -> *mut c_void;
        pub fn nanoflann_free_f32(handle: *mut c_void);
        pub fn nanoflann_free_f64(handle: *mut c_void);
        pub fn nanoflann_nearest_one_f32(
            handle: *mut c_void,
            q: *const f32,
            out_idx: *mut u64,
            out_dist2: *mut f32,
        );
        pub fn nanoflann_nearest_one_f64(
            handle: *mut c_void,
            q: *const f64,
            out_idx: *mut u64,
            out_dist2: *mut f64,
        );
        pub fn nanoflann_nearest_n_f32(
            handle: *mut c_void,
            q: *const f32,
            k: u64,
            out_idx: *mut u64,
            out_dist2: *mut f32,
        ) -> u64;
        pub fn nanoflann_nearest_n_f64(
            handle: *mut c_void,
            q: *const f64,
            k: u64,
            out_idx: *mut u64,
            out_dist2: *mut f64,
        ) -> u64;
        pub fn nanoflann_within_radius_f32(
            handle: *mut c_void,
            q: *const f32,
            radius2: f32,
            out_idx: *mut u64,
            out_dist2: *mut f32,
            cap: u64,
        ) -> u64;
        pub fn nanoflann_within_radius_f64(
            handle: *mut c_void,
            q: *const f64,
            radius2: f64,
            out_idx: *mut u64,
            out_dist2: *mut f64,
            cap: u64,
        ) -> u64;

        pub fn alglib_build_f64(points: *const f64, n: u64, dim: u32) -> *mut c_void;
        pub fn alglib_free_f64(handle: *mut c_void);
        pub fn alglib_nearest_one_f64(
            handle: *mut c_void,
            q: *const f64,
            out_idx: *mut u64,
            out_dist2: *mut f64,
        );
        pub fn alglib_nearest_n_f64(
            handle: *mut c_void,
            q: *const f64,
            k: u64,
            out_idx: *mut u64,
            out_dist2: *mut f64,
        ) -> u64;
        pub fn alglib_within_radius_f64(
            handle: *mut c_void,
            q: *const f64,
            radius2: f64,
            out_idx: *mut u64,
            out_dist2: *mut f64,
            cap: u64,
        ) -> u64;

        pub fn skdtree_build_f64(points: *const f64, n: u64, dim: u32) -> *mut c_void;
        pub fn skdtree_free_f64(handle: *mut c_void);
        pub fn skdtree_nearest_n_f64(
            handle: *mut c_void,
            q: *const f64,
            k: u64,
            out_dist2: *mut f64,
        ) -> u64;
        pub fn pkdtree_build_f32(points: *const f32, n: u64, dim: u32) -> *mut c_void;
        pub fn pkdtree_build_f64(points: *const f64, n: u64, dim: u32) -> *mut c_void;
        pub fn pkdtree_free_f32(handle: *mut c_void);
        pub fn pkdtree_free_f64(handle: *mut c_void);
        pub fn pkdtree_single_query_f32(
            handle: *mut c_void,
            q: *const f32,
            k: u64,
            out_idx: *mut u64,
            out_dist2: *mut f32,
        ) -> u64;
        pub fn pkdtree_single_query_f64(
            handle: *mut c_void,
            q: *const f64,
            k: u64,
            out_idx: *mut u64,
            out_dist2: *mut f64,
        ) -> u64;
        pub fn pkdtree_batch_query_f32(
            handle: *mut c_void,
            queries_flat: *const f32,
            num_queries: u64,
            k: u64,
            out_idx: *mut u64,
            out_dist2: *mut f32,
        );
        pub fn pkdtree_batch_query_f64(
            handle: *mut c_void,
            queries_flat: *const f64,
            num_queries: u64,
            k: u64,
            out_idx: *mut u64,
            out_dist2: *mut f64,
        );
    }
}

macro_rules! raii_handle {
    ($name:ident, $free:path) => {
        struct $name(*mut c_void);
        impl Drop for $name {
            fn drop(&mut self) {
                unsafe { $free(self.0) }
            }
        }
    };
}

raii_handle!(NanoflannF32, ffi::nanoflann_free_f32);
raii_handle!(NanoflannF64, ffi::nanoflann_free_f64);
raii_handle!(AlglibF64, ffi::alglib_free_f64);
raii_handle!(PkdtreeF32, ffi::pkdtree_free_f32);
raii_handle!(PkdtreeF64, ffi::pkdtree_free_f64);
raii_handle!(SkdtreeF64, ffi::skdtree_free_f64);

fn read_usize_env(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("{var} must be a positive integer"))
        })
        .unwrap_or(default)
}

fn read_u32_env(var: &str, default: u32) -> u32 {
    std::env::var(var)
        .ok()
        .map(|value| {
            value
                .parse::<u32>()
                .unwrap_or_else(|_| panic!("{var} must be a positive integer"))
        })
        .unwrap_or(default)
}

fn read_f64_env(var: &str, default: f64) -> f64 {
    std::env::var(var)
        .ok()
        .map(|value| {
            value
                .parse::<f64>()
                .unwrap_or_else(|_| panic!("{var} must be a finite positive number"))
        })
        .unwrap_or(default)
}

/// A comma-separated up-front selection read from an environment variable.
///
/// Consulted before any point cloud is generated or any tree is built, so
/// that deselected work costs nothing instead of being built, benchmarked and
/// then dropped from the exported results.
struct Selection {
    names: Vec<String>,
}

impl Selection {
    fn from_env(var: &str, valid: &[&str], default: &[&str]) -> Self {
        let names: Vec<String> = std::env::var(var)
            .unwrap_or_default()
            .split(',')
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(str::to_owned)
            .collect();

        if names.is_empty() {
            return Self {
                names: default.iter().map(|name| (*name).to_owned()).collect(),
            };
        }

        for name in &names {
            assert!(
                valid.contains(&name.as_str()),
                "{var} contains unknown entry {name:?}; valid entries are {valid:?}"
            );
        }

        Self { names }
    }

    fn contains(&self, name: &str) -> bool {
        self.names.iter().any(|selected| selected == name)
    }

    /// True when at least one of `names` is selected, i.e. when the enclosing
    /// point cloud is still worth generating.
    fn any(&self, names: &[&str]) -> bool {
        names.iter().any(|name| self.contains(name))
    }

    fn list(&self) -> String {
        self.names.join(",")
    }
}

fn suite_selection() -> Selection {
    Selection::from_env("KIDDO_CPP_SUITES", &ALL_SUITES, &DEFAULT_SUITES)
}

fn library_selection() -> Selection {
    Selection::from_env("KIDDO_CPP_LIBRARIES", &ALL_LIBRARIES, &ALL_LIBRARIES)
}

fn scalar_selection() -> Selection {
    Selection::from_env("KIDDO_CPP_SCALARS", &ALL_SCALARS, &ALL_SCALARS)
}

fn build_points_f32(point_count: usize) -> Vec<PointF32> {
    let mut rng = ChaCha8Rng::seed_from_u64(POINT_SEED);
    (0..point_count).map(|_| rng.random()).collect()
}

fn build_queries_f32(query_count: usize) -> Vec<PointF32> {
    let mut rng = ChaCha8Rng::seed_from_u64(QUERY_SEED);
    (0..query_count).map(|_| rng.random()).collect()
}

fn build_points_f64(point_count: usize) -> Vec<PointF64> {
    let mut rng = ChaCha8Rng::seed_from_u64(POINT_SEED);
    (0..point_count).map(|_| rng.random()).collect()
}

fn build_queries_f64(query_count: usize) -> Vec<PointF64> {
    let mut rng = ChaCha8Rng::seed_from_u64(QUERY_SEED);
    (0..query_count).map(|_| rng.random()).collect()
}

fn id(library: &str, query: &str, point_count: usize) -> BenchmarkId {
    BenchmarkId::new(format!("{library}_{query}"), point_count)
}

fn expected_nearest_f32(points: &[PointF32], query: &PointF32, max_qty: usize) -> Vec<f32> {
    let mut expected: Vec<_> = points
        .iter()
        .map(|point| {
            point
                .iter()
                .zip(query)
                .map(|(p, q)| (p - q) * (p - q))
                .sum::<f32>()
        })
        .collect();
    expected.sort_unstable_by(f32::total_cmp);
    expected.truncate(max_qty);
    expected
}

fn expected_nearest_f64(points: &[PointF64], query: &PointF64, max_qty: usize) -> Vec<f64> {
    let mut expected: Vec<_> = points
        .iter()
        .map(|point| {
            point
                .iter()
                .zip(query)
                .map(|(p, q)| (p - q) * (p - q))
                .sum::<f64>()
        })
        .collect();
    expected.sort_unstable_by(f64::total_cmp);
    expected.truncate(max_qty);
    expected
}

fn expected_within_count_f32(points: &[PointF32], query: &PointF32, max_dist2: f32) -> usize {
    points
        .iter()
        .filter(|point| {
            point
                .iter()
                .zip(query)
                .map(|(p, q)| (p - q) * (p - q))
                .sum::<f32>()
                <= max_dist2
        })
        .count()
}

fn expected_within_count_f64(points: &[PointF64], query: &PointF64, max_dist2: f64) -> usize {
    points
        .iter()
        .filter(|point| {
            point
                .iter()
                .zip(query)
                .map(|(p, q)| (p - q) * (p - q))
                .sum::<f64>()
                <= max_dist2
        })
        .count()
}

fn assert_distances_match_f32(label: &str, mut actual: Vec<f32>, expected: &[f32]) {
    actual.sort_unstable_by(f32::total_cmp);
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label} returned the wrong number of neighbours"
    );
    for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
        let tolerance = 1.0e-5 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label} distance mismatch at result {index}: actual={actual} expected={expected}"
        );
    }
}

fn assert_distances_match_f64(label: &str, mut actual: Vec<f64>, expected: &[f64]) {
    actual.sort_unstable_by(f64::total_cmp);
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label} returned the wrong number of neighbours"
    );
    for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
        let tolerance = 1.0e-12 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label} distance mismatch at result {index}: actual={actual} expected={expected}"
        );
    }
}

const VALIDATED_F32: [&str; 2] = [NANOFLANN, PKDTREE];

fn validate_implementations_f32(
    points: &[PointF32],
    query: &PointF32,
    radius: f32,
    libraries: &Selection,
) {
    if !libraries.any(&VALIDATED_F32) {
        return;
    }

    let max_qty = *MAX_QTYS.last().unwrap();
    let radius2 = radius * radius;
    let expected = expected_nearest_f32(points, query, max_qty);
    let mut idx = vec![0u64; max_qty];
    let mut dist2 = vec![0f32; max_qty];

    unsafe {
        if libraries.contains(NANOFLANN) {
            let expected_within = expected_within_count_f32(points, query, radius2);
            let handle = NanoflannF32(ffi::nanoflann_build_f32(
                points.as_ptr().cast(),
                points.len() as u64,
                K as u32,
            ));
            let n = ffi::nanoflann_nearest_n_f32(
                handle.0,
                query.as_ptr(),
                max_qty as u64,
                idx.as_mut_ptr(),
                dist2.as_mut_ptr(),
            );
            assert_distances_match_f32("nanoflann", dist2[..n as usize].to_vec(), &expected);
            let mut ridx = vec![0u64; RADIUS_CAP];
            let mut rdist2 = vec![0f32; RADIUS_CAP];
            let total = ffi::nanoflann_within_radius_f32(
                handle.0,
                query.as_ptr(),
                radius2,
                ridx.as_mut_ptr(),
                rdist2.as_mut_ptr(),
                RADIUS_CAP as u64,
            );
            assert_eq!(
                total as usize, expected_within,
                "nanoflann returned the wrong within-radius count"
            );
        }

        if libraries.contains(PKDTREE) && pkdtree_supports_f32(points.len()) {
            let handle = PkdtreeF32(ffi::pkdtree_build_f32(
                points.as_ptr().cast(),
                points.len() as u64,
                K as u32,
            ));
            let n = ffi::pkdtree_single_query_f32(
                handle.0,
                query.as_ptr(),
                max_qty as u64,
                idx.as_mut_ptr(),
                dist2.as_mut_ptr(),
            );
            assert_distances_match_f32("Pkd-tree", dist2[..n as usize].to_vec(), &expected);
        }
    }
}

const VALIDATED_F64: [&str; 4] = [NANOFLANN, ALGLIB, PKDTREE, SKDTREE];

fn validate_implementations_f64(
    points: &[PointF64],
    query: &PointF64,
    radius: f64,
    libraries: &Selection,
) {
    if !libraries.any(&VALIDATED_F64) {
        return;
    }

    let max_qty = *MAX_QTYS.last().unwrap();
    let radius2 = radius * radius;
    let expected = expected_nearest_f64(points, query, max_qty);
    let mut idx = vec![0u64; max_qty];
    let mut dist2 = vec![0f64; max_qty];
    let mut ridx = vec![0u64; RADIUS_CAP];
    let mut rdist2 = vec![0f64; RADIUS_CAP];
    let expected_within = if libraries.any(&[NANOFLANN, ALGLIB]) {
        expected_within_count_f64(points, query, radius2)
    } else {
        0
    };

    unsafe {
        if libraries.contains(NANOFLANN) {
            let handle = NanoflannF64(ffi::nanoflann_build_f64(
                points.as_ptr().cast(),
                points.len() as u64,
                K as u32,
            ));
            let n = ffi::nanoflann_nearest_n_f64(
                handle.0,
                query.as_ptr(),
                max_qty as u64,
                idx.as_mut_ptr(),
                dist2.as_mut_ptr(),
            );
            assert_distances_match_f64("nanoflann", dist2[..n as usize].to_vec(), &expected);
            let total = ffi::nanoflann_within_radius_f64(
                handle.0,
                query.as_ptr(),
                radius2,
                ridx.as_mut_ptr(),
                rdist2.as_mut_ptr(),
                RADIUS_CAP as u64,
            );
            assert_eq!(
                total as usize, expected_within,
                "nanoflann returned the wrong within-radius count"
            );
        }

        if libraries.contains(ALGLIB) {
            let handle = AlglibF64(ffi::alglib_build_f64(
                points.as_ptr().cast(),
                points.len() as u64,
                K as u32,
            ));
            let n = ffi::alglib_nearest_n_f64(
                handle.0,
                query.as_ptr(),
                max_qty as u64,
                idx.as_mut_ptr(),
                dist2.as_mut_ptr(),
            );
            assert_distances_match_f64("ALGLIB", dist2[..n as usize].to_vec(), &expected);
            let total = ffi::alglib_within_radius_f64(
                handle.0,
                query.as_ptr(),
                radius2,
                ridx.as_mut_ptr(),
                rdist2.as_mut_ptr(),
                RADIUS_CAP as u64,
            );
            assert_eq!(
                total as usize, expected_within,
                "ALGLIB returned the wrong within-radius count"
            );
        }

        if libraries.contains(SKDTREE) {
            // Returns null if the globals are already occupied or a coordinate
            // is outside the quantisable range; both are silent-wrong-answer
            // hazards, so treat null as a hard failure rather than a skip.
            let raw = ffi::skdtree_build_f64(points.as_ptr().cast(), points.len() as u64, K as u32);
            assert!(
                !raw.is_null(),
                "skd-tree refused to build; see skdtree_shim.cpp"
            );
            let handle = SkdtreeF64(raw);
            let n = ffi::skdtree_nearest_n_f64(
                handle.0,
                query.as_ptr(),
                max_qty as u64,
                dist2.as_mut_ptr(),
            );
            assert_distances_match_f64("skd-tree", dist2[..n as usize].to_vec(), &expected);
        }

        if libraries.contains(PKDTREE) {
            let handle = PkdtreeF64(ffi::pkdtree_build_f64(
                points.as_ptr().cast(),
                points.len() as u64,
                K as u32,
            ));
            let n = ffi::pkdtree_single_query_f64(
                handle.0,
                query.as_ptr(),
                max_qty as u64,
                idx.as_mut_ptr(),
                dist2.as_mut_ptr(),
            );
            assert_distances_match_f64("Pkd-tree", dist2[..n as usize].to_vec(), &expected);
        }
    }
}

fn bench_nanoflann_f32(
    group: &mut BenchmarkGroup<'_, WallTime>,
    point_count: usize,
    points: &[PointF32],
    queries: &[PointF32],
    radius2: f32,
    within_radius_id: &str,
) {
    let handle = unsafe {
        NanoflannF32(ffi::nanoflann_build_f32(
            points.as_ptr().cast(),
            points.len() as u64,
            K as u32,
        ))
    };

    group.bench_function(id("nanoflann", "nearest_one", point_count), |b| {
        b.iter(|| {
            let mut distance = 0.0f32;
            let mut item = 0u64;
            for query in queries {
                let mut idx = 0u64;
                let mut dist2 = 0.0f32;
                unsafe {
                    ffi::nanoflann_nearest_one_f32(
                        handle.0,
                        black_box(query.as_ptr()),
                        &mut idx,
                        &mut dist2,
                    );
                }
                distance += dist2;
                item = item.wrapping_add(idx);
            }
            black_box((distance, item))
        });
    });

    for max_qty in MAX_QTYS {
        let mut idx_buf = vec![0u64; max_qty];
        let mut dist2_buf = vec![0f32; max_qty];
        group.bench_function(
            id("nanoflann", &format!("nearest_n_k{max_qty}"), point_count),
            |b| {
                b.iter(|| {
                    let mut len = 0u64;
                    let mut distance = 0.0f32;
                    let mut item = 0u64;
                    for query in queries {
                        let n = unsafe {
                            ffi::nanoflann_nearest_n_f32(
                                handle.0,
                                black_box(query.as_ptr()),
                                max_qty as u64,
                                idx_buf.as_mut_ptr(),
                                dist2_buf.as_mut_ptr(),
                            )
                        };
                        len = len.wrapping_add(n);
                        distance += dist2_buf[..n as usize].iter().sum::<f32>();
                        item = idx_buf[..n as usize]
                            .iter()
                            .fold(item, |acc, v| acc.wrapping_add(*v));
                    }
                    black_box((len, distance, item))
                });
            },
        );
    }

    let mut idx_buf = vec![0u64; RADIUS_CAP];
    let mut dist2_buf = vec![0f32; RADIUS_CAP];
    group.bench_function(id("nanoflann", within_radius_id, point_count), |b| {
        b.iter(|| {
            let mut len = 0u64;
            let mut distance = 0.0f32;
            for query in queries {
                let total = unsafe {
                    ffi::nanoflann_within_radius_f32(
                        handle.0,
                        black_box(query.as_ptr()),
                        radius2,
                        idx_buf.as_mut_ptr(),
                        dist2_buf.as_mut_ptr(),
                        RADIUS_CAP as u64,
                    )
                };
                let copied = (total as usize).min(RADIUS_CAP);
                len = len.wrapping_add(total);
                distance += dist2_buf[..copied].iter().sum::<f32>();
            }
            black_box((len, distance))
        });
    });
}

fn bench_nanoflann_f64(
    group: &mut BenchmarkGroup<'_, WallTime>,
    point_count: usize,
    points: &[PointF64],
    queries: &[PointF64],
    radius2: f64,
    within_radius_id: &str,
) {
    let handle = unsafe {
        NanoflannF64(ffi::nanoflann_build_f64(
            points.as_ptr().cast(),
            points.len() as u64,
            K as u32,
        ))
    };

    group.bench_function(id("nanoflann", "nearest_one", point_count), |b| {
        b.iter(|| {
            let mut distance = 0.0f64;
            let mut item = 0u64;
            for query in queries {
                let mut idx = 0u64;
                let mut dist2 = 0.0f64;
                unsafe {
                    ffi::nanoflann_nearest_one_f64(
                        handle.0,
                        black_box(query.as_ptr()),
                        &mut idx,
                        &mut dist2,
                    );
                }
                distance += dist2;
                item = item.wrapping_add(idx);
            }
            black_box((distance, item))
        });
    });

    for max_qty in MAX_QTYS {
        let mut idx_buf = vec![0u64; max_qty];
        let mut dist2_buf = vec![0f64; max_qty];
        group.bench_function(
            id("nanoflann", &format!("nearest_n_k{max_qty}"), point_count),
            |b| {
                b.iter(|| {
                    let mut len = 0u64;
                    let mut distance = 0.0f64;
                    let mut item = 0u64;
                    for query in queries {
                        let n = unsafe {
                            ffi::nanoflann_nearest_n_f64(
                                handle.0,
                                black_box(query.as_ptr()),
                                max_qty as u64,
                                idx_buf.as_mut_ptr(),
                                dist2_buf.as_mut_ptr(),
                            )
                        };
                        len = len.wrapping_add(n);
                        distance += dist2_buf[..n as usize].iter().sum::<f64>();
                        item = idx_buf[..n as usize]
                            .iter()
                            .fold(item, |acc, v| acc.wrapping_add(*v));
                    }
                    black_box((len, distance, item))
                });
            },
        );
    }

    let mut idx_buf = vec![0u64; RADIUS_CAP];
    let mut dist2_buf = vec![0f64; RADIUS_CAP];
    group.bench_function(id("nanoflann", within_radius_id, point_count), |b| {
        b.iter(|| {
            let mut len = 0u64;
            let mut distance = 0.0f64;
            for query in queries {
                let total = unsafe {
                    ffi::nanoflann_within_radius_f64(
                        handle.0,
                        black_box(query.as_ptr()),
                        radius2,
                        idx_buf.as_mut_ptr(),
                        dist2_buf.as_mut_ptr(),
                        RADIUS_CAP as u64,
                    )
                };
                let copied = (total as usize).min(RADIUS_CAP);
                len = len.wrapping_add(total);
                distance += dist2_buf[..copied].iter().sum::<f64>();
            }
            black_box((len, distance))
        });
    });
}

fn bench_alglib_f64(
    group: &mut BenchmarkGroup<'_, WallTime>,
    point_count: usize,
    points: &[PointF64],
    queries: &[PointF64],
    radius2: f64,
    within_radius_id: &str,
) {
    let handle = unsafe {
        AlglibF64(ffi::alglib_build_f64(
            points.as_ptr().cast(),
            points.len() as u64,
            K as u32,
        ))
    };

    group.bench_function(id("alglib", "nearest_one", point_count), |b| {
        b.iter(|| {
            let mut distance = 0.0f64;
            let mut item = 0u64;
            for query in queries {
                let mut idx = 0u64;
                let mut dist2 = 0.0f64;
                unsafe {
                    ffi::alglib_nearest_one_f64(
                        handle.0,
                        black_box(query.as_ptr()),
                        &mut idx,
                        &mut dist2,
                    );
                }
                distance += dist2;
                item = item.wrapping_add(idx);
            }
            black_box((distance, item))
        });
    });

    for max_qty in MAX_QTYS {
        let mut idx_buf = vec![0u64; max_qty];
        let mut dist2_buf = vec![0f64; max_qty];
        group.bench_function(
            id("alglib", &format!("nearest_n_k{max_qty}"), point_count),
            |b| {
                b.iter(|| {
                    let mut len = 0u64;
                    let mut distance = 0.0f64;
                    let mut item = 0u64;
                    for query in queries {
                        let n = unsafe {
                            ffi::alglib_nearest_n_f64(
                                handle.0,
                                black_box(query.as_ptr()),
                                max_qty as u64,
                                idx_buf.as_mut_ptr(),
                                dist2_buf.as_mut_ptr(),
                            )
                        };
                        len = len.wrapping_add(n);
                        distance += dist2_buf[..n as usize].iter().sum::<f64>();
                        item = idx_buf[..n as usize]
                            .iter()
                            .fold(item, |acc, v| acc.wrapping_add(*v));
                    }
                    black_box((len, distance, item))
                });
            },
        );
    }

    let mut idx_buf = vec![0u64; RADIUS_CAP];
    let mut dist2_buf = vec![0f64; RADIUS_CAP];
    group.bench_function(id("alglib", within_radius_id, point_count), |b| {
        b.iter(|| {
            let mut len = 0u64;
            let mut distance = 0.0f64;
            for query in queries {
                let total = unsafe {
                    ffi::alglib_within_radius_f64(
                        handle.0,
                        black_box(query.as_ptr()),
                        radius2,
                        idx_buf.as_mut_ptr(),
                        dist2_buf.as_mut_ptr(),
                        RADIUS_CAP as u64,
                    )
                };
                let copied = (total as usize).min(RADIUS_CAP);
                len = len.wrapping_add(total);
                distance += dist2_buf[..copied].iter().sum::<f64>();
            }
            black_box((len, distance))
        });
    });
}

fn bench_pkdtree_f32(
    group: &mut BenchmarkGroup<'_, WallTime>,
    point_count: usize,
    points: &[PointF32],
    queries: &[PointF32],
) {
    let handle = unsafe {
        PkdtreeF32(ffi::pkdtree_build_f32(
            points.as_ptr().cast(),
            points.len() as u64,
            K as u32,
        ))
    };

    group.bench_function(id("pkdtree", "nearest_one", point_count), |b| {
        let mut idx_buf = [0u64; 1];
        let mut dist2_buf = [0f32; 1];
        b.iter(|| {
            let mut distance = 0.0f32;
            let mut item = 0u64;
            for query in queries {
                unsafe {
                    ffi::pkdtree_single_query_f32(
                        handle.0,
                        black_box(query.as_ptr()),
                        1,
                        idx_buf.as_mut_ptr(),
                        dist2_buf.as_mut_ptr(),
                    );
                }
                distance += dist2_buf[0];
                item = item.wrapping_add(idx_buf[0]);
            }
            black_box((distance, item))
        });
    });

    for max_qty in MAX_QTYS {
        let mut idx_buf = vec![0u64; max_qty];
        let mut dist2_buf = vec![0f32; max_qty];
        group.bench_function(
            id("pkdtree", &format!("nearest_n_k{max_qty}"), point_count),
            |b| {
                b.iter(|| {
                    let mut len = 0u64;
                    let mut distance = 0.0f32;
                    let mut item = 0u64;
                    for query in queries {
                        let n = unsafe {
                            ffi::pkdtree_single_query_f32(
                                handle.0,
                                black_box(query.as_ptr()),
                                max_qty as u64,
                                idx_buf.as_mut_ptr(),
                                dist2_buf.as_mut_ptr(),
                            )
                        };
                        len = len.wrapping_add(n);
                        distance += dist2_buf[..n as usize].iter().sum::<f32>();
                        item = idx_buf[..n as usize]
                            .iter()
                            .fold(item, |acc, v| acc.wrapping_add(*v));
                    }
                    black_box((len, distance, item))
                });
            },
        );
    }
}

/// skd-tree is f64-only and has no batch API, so it appears here in the
/// per-query suite and not on the batch-throughput chart. It also has no
/// dedicated single-nearest entry point, so nearest_one is k=1.
fn bench_skdtree_f64(
    group: &mut BenchmarkGroup<'_, WallTime>,
    point_count: usize,
    points: &[PointF64],
    queries: &[PointF64],
) {
    let raw =
        unsafe { ffi::skdtree_build_f64(points.as_ptr().cast(), points.len() as u64, K as u32) };
    assert!(
        !raw.is_null(),
        "skd-tree refused to build at {point_count} points; see skdtree_shim.cpp"
    );
    let handle = SkdtreeF64(raw);

    for (query_id, max_qty) in std::iter::once(("nearest_one".to_owned(), NEAREST_ONE_K))
        .chain(MAX_QTYS.map(|max_qty| (format!("nearest_n_k{max_qty}"), max_qty)))
    {
        let mut dist2_buf = vec![0f64; max_qty];
        group.bench_function(id("skdtree", &query_id, point_count), |b| {
            b.iter(|| {
                let mut len = 0u64;
                let mut distance = 0.0f64;
                for query in queries {
                    let n = unsafe {
                        ffi::skdtree_nearest_n_f64(
                            handle.0,
                            black_box(query.as_ptr()),
                            max_qty as u64,
                            dist2_buf.as_mut_ptr(),
                        )
                    };
                    len = len.wrapping_add(n);
                    distance += dist2_buf[..n as usize].iter().sum::<f64>();
                }
                black_box((len, distance))
            });
        });
    }
}

fn bench_pkdtree_f64(
    group: &mut BenchmarkGroup<'_, WallTime>,
    point_count: usize,
    points: &[PointF64],
    queries: &[PointF64],
) {
    let handle = unsafe {
        PkdtreeF64(ffi::pkdtree_build_f64(
            points.as_ptr().cast(),
            points.len() as u64,
            K as u32,
        ))
    };

    group.bench_function(id("pkdtree", "nearest_one", point_count), |b| {
        let mut idx_buf = [0u64; 1];
        let mut dist2_buf = [0f64; 1];
        b.iter(|| {
            let mut distance = 0.0f64;
            let mut item = 0u64;
            for query in queries {
                unsafe {
                    ffi::pkdtree_single_query_f64(
                        handle.0,
                        black_box(query.as_ptr()),
                        1,
                        idx_buf.as_mut_ptr(),
                        dist2_buf.as_mut_ptr(),
                    );
                }
                distance += dist2_buf[0];
                item = item.wrapping_add(idx_buf[0]);
            }
            black_box((distance, item))
        });
    });

    for max_qty in MAX_QTYS {
        let mut idx_buf = vec![0u64; max_qty];
        let mut dist2_buf = vec![0f64; max_qty];
        group.bench_function(
            id("pkdtree", &format!("nearest_n_k{max_qty}"), point_count),
            |b| {
                b.iter(|| {
                    let mut len = 0u64;
                    let mut distance = 0.0f64;
                    let mut item = 0u64;
                    for query in queries {
                        let n = unsafe {
                            ffi::pkdtree_single_query_f64(
                                handle.0,
                                black_box(query.as_ptr()),
                                max_qty as u64,
                                idx_buf.as_mut_ptr(),
                                dist2_buf.as_mut_ptr(),
                            )
                        };
                        len = len.wrapping_add(n);
                        distance += dist2_buf[..n as usize].iter().sum::<f64>();
                        item = idx_buf[..n as usize]
                            .iter()
                            .fold(item, |acc, v| acc.wrapping_add(*v));
                    }
                    black_box((len, distance, item))
                });
            },
        );
    }
}

fn bench_kiddo_single_f32(
    group: &mut BenchmarkGroup<'_, WallTime>,
    point_count: usize,
    points: &[PointF32],
    queries: &[PointF32],
) {
    let tree: KiddoF32 = KdTree::new_from_slice(points).unwrap();

    group.bench_function(id("kiddo", "nearest_one", point_count), |b| {
        b.iter(|| {
            let mut distance = 0.0f32;
            let mut item = 0u64;
            for query in queries {
                let result = tree
                    .query(black_box(query))
                    .nearest_one::<SquaredEuclidean<f32>>()
                    .execute();
                distance += result.distance;
                item = item.wrapping_add(result.item as u64);
            }
            black_box((distance, item))
        });
    });

    for max_qty in MAX_QTYS {
        let k = NonZeroUsize::new(max_qty).unwrap();
        group.bench_function(
            id("kiddo", &format!("nearest_n_k{max_qty}"), point_count),
            |b| {
                b.iter(|| {
                    let mut len = 0u64;
                    let mut distance = 0.0f32;
                    let mut item = 0u64;
                    for query in queries {
                        let results = tree
                            .query(black_box(query))
                            .nearest_n::<SquaredEuclidean<f32>>(k)
                            .execute();
                        len = len.wrapping_add(results.len() as u64);
                        for result in results.iter() {
                            distance += result.distance;
                            item = item.wrapping_add(result.item as u64);
                        }
                    }
                    black_box((len, distance, item))
                });
            },
        );
    }
}

fn bench_kiddo_single_f64(
    group: &mut BenchmarkGroup<'_, WallTime>,
    point_count: usize,
    points: &[PointF64],
    queries: &[PointF64],
) {
    let tree: KiddoF64 = KdTree::new_from_slice(points).unwrap();

    group.bench_function(id("kiddo", "nearest_one", point_count), |b| {
        b.iter(|| {
            let mut distance = 0.0f64;
            let mut item = 0u64;
            for query in queries {
                let result = tree
                    .query(black_box(query))
                    .nearest_one::<SquaredEuclidean<f64>>()
                    .execute();
                distance += result.distance;
                item = item.wrapping_add(result.item as u64);
            }
            black_box((distance, item))
        });
    });

    for max_qty in MAX_QTYS {
        let k = NonZeroUsize::new(max_qty).unwrap();
        group.bench_function(
            id("kiddo", &format!("nearest_n_k{max_qty}"), point_count),
            |b| {
                b.iter(|| {
                    let mut len = 0u64;
                    let mut distance = 0.0f64;
                    let mut item = 0u64;
                    for query in queries {
                        let results = tree
                            .query(black_box(query))
                            .nearest_n::<SquaredEuclidean<f64>>(k)
                            .execute();
                        len = len.wrapping_add(results.len() as u64);
                        for result in results.iter() {
                            distance += result.distance;
                            item = item.wrapping_add(result.item as u64);
                        }
                    }
                    black_box((len, distance, item))
                });
            },
        );
    }
}

fn kiddo_vs_pkdtree_single(c: &mut Criterion) {
    let suites = suite_selection();
    if !suites.contains(SUITE_KIDDO_VS_PKDTREE) {
        return;
    }
    let libraries = library_selection();
    let scalars = scalar_selection();
    if !libraries.any(&[KIDDO, PKDTREE]) {
        return;
    }

    // Deliberately distinct env vars from KIDDO_PROFILE_MIN/MAX_LOG2_POINTS,
    // so that one invocation can sweep the competitor suite over its usual
    // range while the large-tree suites use a different, coarser one.
    let query_count = read_usize_env("KIDDO_PROFILE_QUERIES", DEFAULT_QUERY_COUNT);
    let min_log2_points = read_u32_env("KIDDO_LARGE_MIN_LOG2_POINTS", DEFAULT_MIN_LOG2_POINTS);
    let max_log2_points = read_u32_env("KIDDO_LARGE_MAX_LOG2_POINTS", DEFAULT_MAX_LOG2_POINTS);

    eprintln!(
        "benchmarking kiddo vs Pkd-tree only, single-threaded per-query: scalars={} libraries={} tree_sizes=2^{min_log2_points}..2^{max_log2_points} queries={query_count}",
        scalars.list(),
        libraries.list()
    );

    if scalars.contains(F32) {
        let queries_f32 = build_queries_f32(query_count);
        let mut group = c.benchmark_group("profile_kiddo_vs_pkdtree/f32");
        group.throughput(Throughput::Elements(query_count as u64));
        for log2_points in min_log2_points..=max_log2_points {
            let point_count = 1usize << log2_points;
            let points = build_points_f32(point_count);
            if libraries.contains(KIDDO) {
                bench_kiddo_single_f32(&mut group, point_count, &points, &queries_f32);
            }
            if pkdtree_f32_selected(&libraries, point_count) {
                bench_pkdtree_f32(&mut group, point_count, &points, &queries_f32);
            }
        }
        group.finish();
    }

    if scalars.contains(F64) {
        let queries_f64 = build_queries_f64(query_count);
        let mut group = c.benchmark_group("profile_kiddo_vs_pkdtree/f64");
        group.throughput(Throughput::Elements(query_count as u64));
        for log2_points in min_log2_points..=max_log2_points {
            let point_count = 1usize << log2_points;
            let points = build_points_f64(point_count);
            if libraries.contains(KIDDO) {
                bench_kiddo_single_f64(&mut group, point_count, &points, &queries_f64);
            }
            if libraries.contains(PKDTREE) {
                bench_pkdtree_f64(&mut group, point_count, &points, &queries_f64);
            }
            if libraries.contains(SKDTREE) {
                bench_skdtree_f64(&mut group, point_count, &points, &queries_f64);
            }
        }
        group.finish();
    }
}

fn bench_pkdtree_batch_f32(
    group: &mut BenchmarkGroup<'_, WallTime>,
    point_count: usize,
    points: &[PointF32],
    queries: &[PointF32],
) {
    let handle = unsafe {
        PkdtreeF32(ffi::pkdtree_build_f32(
            points.as_ptr().cast(),
            points.len() as u64,
            K as u32,
        ))
    };
    let flat_queries: Vec<f32> = queries.iter().flatten().copied().collect();
    let num_queries = queries.len() as u64;

    for (query_id, max_qty) in std::iter::once(("nearest_one".to_owned(), NEAREST_ONE_K))
        .chain(MAX_QTYS.map(|max_qty| (format!("nearest_n_k{max_qty}"), max_qty)))
    {
        let mut idx_buf = vec![0u64; queries.len() * max_qty];
        let mut dist2_buf = vec![0f32; queries.len() * max_qty];
        group.bench_function(id("pkdtree_batch", &query_id, point_count), |b| {
            b.iter(|| {
                unsafe {
                    ffi::pkdtree_batch_query_f32(
                        handle.0,
                        black_box(flat_queries.as_ptr()),
                        num_queries,
                        max_qty as u64,
                        idx_buf.as_mut_ptr(),
                        dist2_buf.as_mut_ptr(),
                    );
                }
                black_box((dist2_buf.iter().sum::<f32>(), idx_buf.iter().sum::<u64>()))
            });
        });
    }
}

fn bench_pkdtree_batch_f64(
    group: &mut BenchmarkGroup<'_, WallTime>,
    point_count: usize,
    points: &[PointF64],
    queries: &[PointF64],
) {
    let handle = unsafe {
        PkdtreeF64(ffi::pkdtree_build_f64(
            points.as_ptr().cast(),
            points.len() as u64,
            K as u32,
        ))
    };
    let flat_queries: Vec<f64> = queries.iter().flatten().copied().collect();
    let num_queries = queries.len() as u64;

    for (query_id, max_qty) in std::iter::once(("nearest_one".to_owned(), NEAREST_ONE_K))
        .chain(MAX_QTYS.map(|max_qty| (format!("nearest_n_k{max_qty}"), max_qty)))
    {
        let mut idx_buf = vec![0u64; queries.len() * max_qty];
        let mut dist2_buf = vec![0f64; queries.len() * max_qty];
        group.bench_function(id("pkdtree_batch", &query_id, point_count), |b| {
            b.iter(|| {
                unsafe {
                    ffi::pkdtree_batch_query_f64(
                        handle.0,
                        black_box(flat_queries.as_ptr()),
                        num_queries,
                        max_qty as u64,
                        idx_buf.as_mut_ptr(),
                        dist2_buf.as_mut_ptr(),
                    );
                }
                black_box((dist2_buf.iter().sum::<f64>(), idx_buf.iter().sum::<u64>()))
            });
        });
    }
}

fn bench_kiddo_batch_f32<S: StemStrategy>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    label: &str,
    point_count: usize,
    points: &[PointF32],
    queries: &[PointF32],
) {
    let tree: KiddoF32<S> = KdTree::new_from_slice(points).unwrap();

    // nearest_one, not nearest_n(1): it collects one result per query into a
    // flat Vec instead of a Vec per query, which is the point of having a
    // separate entry point at all.
    for (executor_name, executor) in kiddo_batch_executors() {
        group.bench_function(
            id(
                &format!("{label}_{executor_name}"),
                "nearest_one",
                point_count,
            ),
            |b| {
                b.iter(|| {
                    let batch = tree
                        .query_batch(black_box(queries))
                        .with_executor(&executor)
                        .nearest_one::<SquaredEuclidean<f32>>()
                        .execute();
                    black_box(batch.len())
                });
            },
        );
    }

    for max_qty in MAX_QTYS {
        let k = NonZeroUsize::new(max_qty).unwrap();
        for (executor_name, executor) in kiddo_batch_executors() {
            group.bench_function(
                id(
                    &format!("{label}_{executor_name}"),
                    &format!("nearest_n_k{max_qty}"),
                    point_count,
                ),
                |b| {
                    b.iter(|| {
                        let batch = tree
                            .query_batch(black_box(queries))
                            .with_executor(&executor)
                            .nearest_n::<SquaredEuclidean<f32>>(k)
                            .execute();
                        black_box(batch.total_len())
                    });
                },
            );
        }
    }
}

fn bench_kiddo_batch_f64<S: StemStrategy>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    label: &str,
    point_count: usize,
    points: &[PointF64],
    queries: &[PointF64],
) {
    let tree: KiddoF64<S> = KdTree::new_from_slice(points).unwrap();

    for (executor_name, executor) in kiddo_batch_executors() {
        group.bench_function(
            id(
                &format!("{label}_{executor_name}"),
                "nearest_one",
                point_count,
            ),
            |b| {
                b.iter(|| {
                    let batch = tree
                        .query_batch(black_box(queries))
                        .with_executor(&executor)
                        .nearest_one::<SquaredEuclidean<f64>>()
                        .execute();
                    black_box(batch.len())
                });
            },
        );
    }

    for max_qty in MAX_QTYS {
        let k = NonZeroUsize::new(max_qty).unwrap();
        for (executor_name, executor) in kiddo_batch_executors() {
            group.bench_function(
                id(
                    &format!("{label}_{executor_name}"),
                    &format!("nearest_n_k{max_qty}"),
                    point_count,
                ),
                |b| {
                    b.iter(|| {
                        let batch = tree
                            .query_batch(black_box(queries))
                            .with_executor(&executor)
                            .nearest_n::<SquaredEuclidean<f64>>(k)
                            .execute();
                        black_box(batch.total_len())
                    });
                },
            );
        }
    }
}

fn pkdtree_batch(c: &mut Criterion) {
    let suites = suite_selection();
    if !suites.contains(SUITE_PKDTREE_BATCH) {
        return;
    }
    let libraries = library_selection();
    let scalars = scalar_selection();
    if !libraries.any(&[KIDDO, PKDTREE]) {
        return;
    }

    // See kiddo_vs_pkdtree_single's comment: deliberately distinct env vars
    // from cpp_competitors' KIDDO_PROFILE_MIN/MAX_LOG2_POINTS.
    let query_count = read_usize_env("KIDDO_PROFILE_QUERIES", DEFAULT_QUERY_COUNT);
    let min_log2_points = read_u32_env("KIDDO_LARGE_MIN_LOG2_POINTS", DEFAULT_MIN_LOG2_POINTS);
    let max_log2_points = read_u32_env("KIDDO_LARGE_MAX_LOG2_POINTS", DEFAULT_MAX_LOG2_POINTS);

    eprintln!(
        "benchmarking batch-parallel throughput (kiddo serial+parallel executors, Pkd-tree parallel_for): scalars={} libraries={} tree_sizes=2^{min_log2_points}..2^{max_log2_points} queries={query_count} (all submitted at once)",
        scalars.list(),
        libraries.list()
    );

    if scalars.contains(F32) {
        let queries_f32 = build_queries_f32(query_count);
        let mut group = c.benchmark_group("profile_pkdtree_batch/f32");
        group.throughput(Throughput::Elements(query_count as u64));
        for log2_points in min_log2_points..=max_log2_points {
            let point_count = 1usize << log2_points;
            let points = build_points_f32(point_count);
            if libraries.contains(KIDDO) {
                bench_kiddo_batch_f32::<Eytzinger>(
                    &mut group,
                    "kiddo_batch",
                    point_count,
                    &points,
                    &queries_f32,
                );
                // f32/3D cannot use DonnellyCyclicSimdDescent, whose AVX-512
                // path requires BH == K == 4. DonnellyUnrolled has no such
                // constraint, so f32 still gets the Donnelly memory layout,
                // just with scalar rather than block-at-once descent.
                bench_kiddo_batch_f32::<DonnellyUnrolled<4>>(
                    &mut group,
                    "kiddo_donnelly_batch",
                    point_count,
                    &points,
                    &queries_f32,
                );
            }
            if pkdtree_f32_selected(&libraries, point_count) {
                bench_pkdtree_batch_f32(&mut group, point_count, &points, &queries_f32);
            }
        }
        group.finish();
    }

    if scalars.contains(F64) {
        let queries_f64 = build_queries_f64(query_count);
        let mut group = c.benchmark_group("profile_pkdtree_batch/f64");
        group.throughput(Throughput::Elements(query_count as u64));
        for log2_points in min_log2_points..=max_log2_points {
            let point_count = 1usize << log2_points;
            let points = build_points_f64(point_count);
            if libraries.contains(KIDDO) {
                if log2_points == min_log2_points {
                    validate_kiddo_strategies_f64(
                        &points,
                        &queries_f64[..queries_f64.len().min(64)],
                    );
                }
                bench_kiddo_batch_f64::<Eytzinger>(
                    &mut group,
                    "kiddo_batch",
                    point_count,
                    &points,
                    &queries_f64,
                );
                // Each strategy builds and drops its own tree in turn, so only
                // one is resident at a time.
                bench_kiddo_batch_f64::<DonnellyCyclicSimdDescent<3>>(
                    &mut group,
                    "kiddo_donnelly_batch",
                    point_count,
                    &points,
                    &queries_f64,
                );
            }
            if libraries.contains(PKDTREE) {
                bench_pkdtree_batch_f64(&mut group, point_count, &points, &queries_f64);
            }
        }
        group.finish();
    }
}

fn cpp_competitors(c: &mut Criterion) {
    let suites = suite_selection();
    if !suites.contains(SUITE_CPP_COMPETITORS) {
        return;
    }
    let libraries = library_selection();
    let scalars = scalar_selection();

    let query_count = read_usize_env("KIDDO_PROFILE_QUERIES", DEFAULT_QUERY_COUNT);
    let min_log2_points = read_u32_env("KIDDO_PROFILE_MIN_LOG2_POINTS", DEFAULT_MIN_LOG2_POINTS);
    let max_log2_points = read_u32_env("KIDDO_PROFILE_MAX_LOG2_POINTS", DEFAULT_MAX_LOG2_POINTS);
    let radius_f64 = read_f64_env("KIDDO_PROFILE_RADIUS", DEFAULT_RADIUS);
    let radius_f32 = radius_f64 as f32;
    let within_radius_id = format!("within_radius_r{radius_f64}");

    assert!(query_count > 0, "KIDDO_PROFILE_QUERIES must be positive");
    assert!(
        min_log2_points <= max_log2_points,
        "KIDDO_PROFILE_MIN_LOG2_POINTS must not exceed KIDDO_PROFILE_MAX_LOG2_POINTS"
    );
    assert!(
        max_log2_points <= 31,
        "KIDDO_PROFILE_MAX_LOG2_POINTS must fit u32 item indices"
    );
    assert!(
        radius_f64.is_finite() && radius_f64 > 0.0 && radius_f32.is_finite() && radius_f32 > 0.0,
        "KIDDO_PROFILE_RADIUS must be a finite positive number"
    );

    eprintln!(
        "benchmarking C++ k-d trees: dims={K} scalars={} libraries={} tree_sizes=2^{min_log2_points}..2^{max_log2_points} queries={query_count} nearest_n={MAX_QTYS:?} radius={radius_f64} point_seed={POINT_SEED} query_seed={QUERY_SEED}",
        scalars.list(),
        libraries.list()
    );

    // ALGLIB's `real` is hard-coded to `double`, so it has no f32 half.
    if scalars.contains(F32) && libraries.any(&[NANOFLANN, PKDTREE]) {
        let queries_f32 = build_queries_f32(query_count);
        let mut group = c.benchmark_group("profile_cpp_competitors/f32");
        group.throughput(Throughput::Elements(query_count as u64));

        for log2_points in min_log2_points..=max_log2_points {
            let point_count = 1usize << log2_points;
            let points = build_points_f32(point_count);
            if log2_points == min_log2_points {
                validate_implementations_f32(&points, &queries_f32[0], radius_f32, &libraries);
            }

            if libraries.contains(NANOFLANN) {
                bench_nanoflann_f32(
                    &mut group,
                    point_count,
                    &points,
                    &queries_f32,
                    radius_f32 * radius_f32,
                    &within_radius_id,
                );
            }
            if pkdtree_f32_selected(&libraries, point_count) {
                bench_pkdtree_f32(&mut group, point_count, &points, &queries_f32);
            }
        }

        group.finish();
    }

    if scalars.contains(F64) && libraries.any(&VALIDATED_F64) {
        let queries_f64 = build_queries_f64(query_count);
        let mut group = c.benchmark_group("profile_cpp_competitors/f64");
        group.throughput(Throughput::Elements(query_count as u64));

        for log2_points in min_log2_points..=max_log2_points {
            let point_count = 1usize << log2_points;
            let points = build_points_f64(point_count);
            if log2_points == min_log2_points {
                validate_implementations_f64(&points, &queries_f64[0], radius_f64, &libraries);
            }

            if libraries.contains(NANOFLANN) {
                bench_nanoflann_f64(
                    &mut group,
                    point_count,
                    &points,
                    &queries_f64,
                    radius_f64 * radius_f64,
                    &within_radius_id,
                );
            }
            if libraries.contains(ALGLIB) {
                bench_alglib_f64(
                    &mut group,
                    point_count,
                    &points,
                    &queries_f64,
                    radius_f64 * radius_f64,
                    &within_radius_id,
                );
            }
            if libraries.contains(PKDTREE) {
                bench_pkdtree_f64(&mut group, point_count, &points, &queries_f64);
            }
            if libraries.contains(SKDTREE) {
                bench_skdtree_f64(&mut group, point_count, &points, &queries_f64);
            }
        }

        group.finish();
    }
}

// Temporary diagnostic: is Pkd-tree's build_recursive assertion failure
// (observed with f32 at 2^22+ points) f32-precision-specific, or does f64
// hit it too? f64-only, registered before anything that touches f32, so it
// can't be preempted by an f32 abort. Opt-in via KIDDO_CPP_SUITES, since it
// otherwise builds a large tree on every unrelated run. Remove once answered.
fn pkdtree_f64_probe(c: &mut Criterion) {
    if !suite_selection().contains(SUITE_PKDTREE_PROBE) {
        return;
    }
    let min_log2_points = read_u32_env("KIDDO_LARGE_MIN_LOG2_POINTS", 22);
    let max_log2_points = read_u32_env("KIDDO_LARGE_MAX_LOG2_POINTS", 22);
    for log2n in min_log2_points..=max_log2_points {
        let n = 1usize << log2n;
        eprintln!("pkdtree_f64_probe: building at n={n} (2^{log2n})...");
        let points = build_points_f64(n);
        let handle = unsafe {
            PkdtreeF64(ffi::pkdtree_build_f64(
                points.as_ptr().cast(),
                points.len() as u64,
                K as u32,
            ))
        };
        let query = build_queries_f64(1);
        let mut idx = [0u64; 1];
        let mut dist2 = [0f64; 1];
        unsafe {
            ffi::pkdtree_single_query_f64(
                handle.0,
                query[0].as_ptr(),
                1,
                idx.as_mut_ptr(),
                dist2.as_mut_ptr(),
            );
        }
        eprintln!(
            "pkdtree_f64_probe: OK at n={n}, idx={} dist2={}",
            idx[0], dist2[0]
        );
    }
    let _ = c;
}

criterion_group!(
    benches,
    pkdtree_f64_probe,
    cpp_competitors,
    pkdtree_batch,
    kiddo_vs_pkdtree_single
);
criterion_main!(benches);

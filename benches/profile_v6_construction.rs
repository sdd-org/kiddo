#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![cfg_attr(coverage_nightly, coverage(off))]

use criterion::measurement::WallTime;
use criterion::{
    criterion_group, criterion_main, BatchSize, BenchmarkGroup, BenchmarkId, Criterion,
    SamplingMode, Throughput,
};
use kiddo::kd_tree::KdTree;
use kiddo::leaf_strategy::VecOfArenas;
use kiddo::{DonnellySimdDescent, Eytzinger};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;
use std::time::Duration;

const K: usize = 3;
const B: usize = 32;
const POINT_SEED: u64 = 0x5eed_0000_0000_0401;
const TREE_SIZES: [usize; 11] = [
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
    1 << 26,
];

type F32EytzingerTree = KdTree<f32, u32, Eytzinger, VecOfArenas<f32, u32, K, B>, K, B>;
type F64EytzingerTree = KdTree<f64, u32, Eytzinger, VecOfArenas<f64, u32, K, B>, K, B>;
type F64DonnellySimdDescentTree =
    KdTree<f64, u32, DonnellySimdDescent<3>, VecOfArenas<f64, u32, K, B>, K, B>;

fn build_points_f32(point_count: usize) -> Vec<[f32; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(POINT_SEED);
    (0..point_count).map(|_| rng.random::<[f32; K]>()).collect()
}

fn build_points_f64(point_count: usize) -> Vec<[f64; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(POINT_SEED);
    (0..point_count).map(|_| rng.random::<[f64; K]>()).collect()
}

fn configure_group<'a>(c: &'a mut Criterion, group_id: &str) -> BenchmarkGroup<'a, WallTime> {
    let mut group = c.benchmark_group(group_id);
    group.throughput(Throughput::Elements(1));
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group
}

fn bench_construction<A, Tree>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    strategy: &str,
    point_count: usize,
    points: &[[A; K]],
    build: impl Fn(&[[A; K]]) -> Tree,
) {
    group.bench_function(BenchmarkId::new(strategy, point_count), |b| {
        b.iter_batched(
            || (),
            |()| black_box(build(black_box(points))),
            BatchSize::LargeInput,
        )
    });
}

fn construction(c: &mut Criterion) {
    eprintln!(
        "benchmarking v6 construction: dims={} leaf=VecOfArenas tree_sizes=2^16..=2^26 \
         rayon_threads={} point_seed={}",
        K,
        std::env::var("RAYON_NUM_THREADS")
            .as_deref()
            .unwrap_or("unset"),
        POINT_SEED,
    );

    {
        let mut group = configure_group(c, "profile_v6_construction/f32");
        for point_count in TREE_SIZES {
            let points = build_points_f32(point_count);
            bench_construction(&mut group, "eytzinger", point_count, &points, |points| {
                F32EytzingerTree::new_from_slice(points).expect("f32 Eytzinger construction failed")
            });
        }
        group.finish();
    }

    {
        let mut group = configure_group(c, "profile_v6_construction/f64");
        for point_count in TREE_SIZES {
            let points = build_points_f64(point_count);
            bench_construction(&mut group, "eytzinger", point_count, &points, |points| {
                F64EytzingerTree::new_from_slice(points).expect("f64 Eytzinger construction failed")
            });
            bench_construction(
                &mut group,
                "donnelly_simd_descent",
                point_count,
                &points,
                |points| {
                    F64DonnellySimdDescentTree::new_from_slice(points)
                        .expect("f64 DonnellySimdDescent<3> construction failed")
                },
            );
        }
        group.finish();
    }
}

criterion_group!(benches, construction);
criterion_main!(benches);

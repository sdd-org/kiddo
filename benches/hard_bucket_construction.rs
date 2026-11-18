use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kiddo::kd_tree::KdTree;
use kiddo::leaf_strategy::VecOfArrays;
use kiddo::stem_strategy::Eytzinger;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::time::Duration;

const K: usize = 3;
const B: usize = 32;
const SIZES: [usize; 4] = [1 << 14, 1 << 16, 1 << 18, 1 << 20];
const F64_SEED: u64 = 0x5eed_0000_0000_0601;
const U32_SEED: u64 = 0x5eed_0000_0000_0602;

type F64Tree = KdTree<f64, u32, Eytzinger, VecOfArrays<f64, u32, K, B>, K, B>;
type U32Tree = KdTree<u32, u32, Eytzinger, VecOfArrays<u32, u32, K, B>, K, B>;

fn uniform_f64(point_count: usize) -> Vec<[f64; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(F64_SEED);
    (0..point_count).map(|_| rng.random::<[f64; K]>()).collect()
}

fn quantised_u32(point_count: usize) -> Vec<[u32; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(U32_SEED);
    // Keep the expected number of points per coordinate value around eight, so
    // the hard bucket stays representable while still containing many ties.
    let grid = (point_count / 8).next_power_of_two().max(1) as u32;
    (0..point_count)
        .map(|_| std::array::from_fn(|_| rng.random_range(0..grid)))
        .collect()
}

fn hard_bucket_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("hard_bucket_construction");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    for point_count in SIZES {
        let points = uniform_f64(point_count);
        assert!(
            F64Tree::new_from_slice(&points).is_ok(),
            "uniform f64 benchmark input must be representable"
        );
        group.throughput(Throughput::Elements(point_count as u64));
        group.bench_with_input(
            BenchmarkId::new("uniform_f64", point_count),
            &points,
            |b, points| {
                b.iter(|| black_box(F64Tree::new_from_slice(black_box(points.as_slice())).unwrap()))
            },
        );
    }

    for point_count in SIZES {
        let points = quantised_u32(point_count);
        assert!(
            U32Tree::new_from_slice(&points).is_ok(),
            "quantised u32 benchmark input must be representable"
        );
        group.throughput(Throughput::Elements(point_count as u64));
        group.bench_with_input(
            BenchmarkId::new("quantised_u32", point_count),
            &points,
            |b, points| {
                b.iter(|| black_box(U32Tree::new_from_slice(black_box(points.as_slice())).unwrap()))
            },
        );
    }

    group.finish();
}

criterion_group!(benches, hard_bucket_construction);
criterion_main!(benches);

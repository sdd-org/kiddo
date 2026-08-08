use codspeed_criterion_compat::{
    black_box, criterion_group, criterion_main, Criterion, Throughput,
};
use kiddo::dist::SquaredEuclidean;
use kiddo::kd_tree::KdTree;
use kiddo::leaf_strategy::VecOfArrays;
use kiddo::stem_strategy::Eytzinger;
use kiddo::QueryScratch;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::num::NonZeroUsize;

const K: usize = 3;
const B: usize = 32;
const POINT_COUNT: usize = 1_000_000;
const QUERY_COUNT: usize = 10_000;
const MAX_QTY: usize = 16;
const POINT_SEED: u64 = 0x5eed_5c12_a7e9_1001;
const QUERY_SEED: u64 = 0x5eed_5c12_a7e9_1002;

type Tree = KdTree<f64, u32, Eytzinger, VecOfArrays<f64, u32, K, B>, K, B>;

fn build_points() -> Vec<[f64; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(POINT_SEED);
    (0..POINT_COUNT).map(|_| rng.random::<[f64; K]>()).collect()
}

fn build_queries() -> Vec<[f64; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(QUERY_SEED);
    (0..QUERY_COUNT).map(|_| rng.random::<[f64; K]>()).collect()
}

fn run_local_default(
    tree: &Tree,
    queries: &[[f64; K]],
    max_qty: NonZeroUsize,
) -> (usize, u64, f64) {
    let mut checksum_len = 0usize;
    let mut checksum_item = 0u64;
    let mut checksum_dist = 0.0f64;

    for query in queries {
        let results = tree
            .query(black_box(query))
            .nearest_n::<SquaredEuclidean<f64>>(max_qty)
            .execute();
        checksum_len = checksum_len.wrapping_add(results.len());
        for result in results {
            checksum_item = checksum_item.wrapping_add(result.item as u64);
            checksum_dist += result.distance;
        }
    }

    (checksum_len, checksum_item, checksum_dist)
}

fn run_reused_scratch(
    tree: &Tree,
    queries: &[[f64; K]],
    max_qty: NonZeroUsize,
    scratch: &mut QueryScratch<Eytzinger, f64, K>,
) -> (usize, u64, f64) {
    let mut checksum_len = 0usize;
    let mut checksum_item = 0u64;
    let mut checksum_dist = 0.0f64;

    for query in queries {
        let results = tree
            .query(black_box(query))
            .nearest_n::<SquaredEuclidean<f64>>(max_qty)
            .with_scratch(scratch)
            .execute();
        checksum_len = checksum_len.wrapping_add(results.len());
        for result in results {
            checksum_item = checksum_item.wrapping_add(result.item as u64);
            checksum_dist += result.distance;
        }
    }

    (checksum_len, checksum_item, checksum_dist)
}

fn scratch_strategies(c: &mut Criterion) {
    let points = build_points();
    let queries = build_queries();
    let tree = Tree::new_from_slice(&points).unwrap();
    let max_qty = NonZeroUsize::new(MAX_QTY).unwrap();

    let mut group = c.benchmark_group("v6 scratch strategies");
    group.throughput(Throughput::Elements(QUERY_COUNT as u64));

    group.bench_function("nearest_n local default", |b| {
        b.iter(|| black_box(run_local_default(&tree, &queries, max_qty)));
    });

    let mut scratch = tree.create_scratch::<SquaredEuclidean<f64>>();
    group.bench_function("nearest_n reused scratch", |b| {
        b.iter(|| black_box(run_reused_scratch(&tree, &queries, max_qty, &mut scratch)));
    });

    group.finish();
}

criterion_group!(benches, scratch_strategies);
criterion_main!(benches);

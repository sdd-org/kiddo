//! Where does kiddo's parallel batch mode lose throughput at small batch
//! sizes? Isolates two candidate costs against the same tree and executor:
//!
//! - `nearest_one` produces one `QueryResultItem` per query and collects into
//!   a flat `Vec`, while `nearest_n` produces a `Vec` per query and collects
//!   into `Vec<Vec<_>>`. Comparing `nearest_one` with `nearest_n(k=1)` prices
//!   the per-query heap allocation at equal query work.
//! - static chunking and the serial fallback change how, and whether, the
//!   batch is dispatched to the pool at all.
//!
//! Run: cargo run --release --example batch_overhead_probe

use kiddo::batch::Executor;
use kiddo::dist::SquaredEuclidean;
use kiddo::kd_tree::KdTree;
use kiddo::leaf_strategy::FlatVec;
use kiddo::stem_strategy::Eytzinger;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;
use std::num::NonZeroUsize;
use std::time::Instant;

const K: usize = 3;
const B: usize = 32;
const LOG2_POINTS: u32 = 21;
const QUERY_COUNTS: [usize; 8] = [500, 1_000, 4_000, 16_000, 50_000, 100_000, 200_000, 400_000];
const REPEATS: usize = 40;

type Tree = KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, K, B>, K, B>;

fn points(count: usize, seed: u64) -> Vec<[f32; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..count).map(|_| rng.random()).collect()
}

/// Best-of-`REPEATS` throughput in queries/sec, after a warm-up pass that also
/// leaves the rayon pool awake.
fn throughput(queries: usize, mut run: impl FnMut() -> usize) -> f64 {
    for _ in 0..3 {
        black_box(run());
    }
    let mut best = f64::INFINITY;
    for _ in 0..REPEATS {
        let started = Instant::now();
        black_box(run());
        let elapsed = started.elapsed().as_secs_f64();
        best = best.min(elapsed);
    }
    queries as f64 / best
}

fn main() {
    let point_count = 1usize << LOG2_POINTS;
    let tree: Tree = KdTree::new_from_slice(&points(point_count, 1)).unwrap();
    let all_queries = points(*QUERY_COUNTS.last().unwrap(), 2);
    let parallel = Executor::parallel();
    let k20 = NonZeroUsize::new(20).unwrap();

    println!("tree=2^{LOG2_POINTS} f32 {K}D, throughput in Melem/s, best of {REPEATS}\n");
    println!(
        "{:>8}  {:>12}  {:>12}  {:>7}  {:>12}  {:>12}  {:>12}",
        "queries",
        "one adaptive",
        "one chunked",
        "chunk%",
        "one fallback",
        "k20 adaptive",
        "k20 chunked"
    );

    for count in QUERY_COUNTS {
        let queries = &all_queries[..count];

        let chunked = Executor::parallel().with_default_static_chunking();
        let fallback = Executor::parallel()
            .with_default_static_chunking()
            .with_serial_fallback();

        let one_adaptive = throughput(count, || {
            tree.query_batch(queries)
                .with_executor(&parallel)
                .nearest_one::<SquaredEuclidean<f32>>()
                .execute()
                .len()
        });
        let one_chunked = throughput(count, || {
            tree.query_batch(queries)
                .with_executor(&chunked)
                .nearest_one::<SquaredEuclidean<f32>>()
                .execute()
                .len()
        });
        let one_fallback = throughput(count, || {
            tree.query_batch(queries)
                .with_executor(&fallback)
                .nearest_one::<SquaredEuclidean<f32>>()
                .execute()
                .len()
        });

        let k20_adaptive = throughput(count, || {
            tree.query_batch(queries)
                .with_executor(&parallel)
                .nearest_n::<SquaredEuclidean<f32>>(k20)
                .execute()
                .total_len()
        });
        let k20_chunked = throughput(count, || {
            tree.query_batch(queries)
                .with_executor(&chunked)
                .nearest_n::<SquaredEuclidean<f32>>(k20)
                .execute()
                .total_len()
        });

        println!(
            "{:>8}  {:>12.2}  {:>12.2}  {:>+6.0}%  {:>12.2}  {:>12.2}  {:>12.2}",
            count,
            one_adaptive / 1.0e6,
            one_chunked / 1.0e6,
            100.0 * (one_chunked - one_adaptive) / one_adaptive,
            one_fallback / 1.0e6,
            k20_adaptive / 1.0e6,
            k20_chunked / 1.0e6,
        );
    }
}

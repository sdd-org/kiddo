#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![cfg_attr(coverage_nightly, coverage(off))]

//! Reproduction of neighbourhood's published 3D within-range benchmark.

use kiddo::dist::SquaredEuclidean;
use kiddo::kd_tree::KdTree;
use kiddo::leaf_strategy::FlatVec;
use kiddo::stem_strategy::Eytzinger;
use neighbourhood::KdTree as NeighbourhoodTree;
use rand_09::distr::{Distribution, Uniform};
use rand_09::rngs::SmallRng;
use rand_09::SeedableRng;
use std::hint::black_box;
use std::time::{Duration, Instant};

const K: usize = 3;
const B: usize = 32;
const DEFAULT_POINT_COUNT: usize = 100_000_000;
const DEFAULT_QUERY_COUNT: usize = 20_000;
const DEFAULT_RUNS: usize = 1;
const DEFAULT_EPSILONS: [f64; 4] = [0.02, 0.05, 0.1, 0.2];
const POINT_SEED: u64 = 0;
const QUERY_SEED: u64 = u64::MAX;

type KiddoTree = KdTree<f64, u64, Eytzinger, FlatVec<f64, u64, K, B>, K, B>;

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

fn read_epsilons() -> Vec<f64> {
    std::env::var("KIDDO_NEIGHBOURHOOD_EPSILONS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(|epsilon| {
                    epsilon.trim().parse::<f64>().unwrap_or_else(|_| {
                        panic!(
                            "KIDDO_NEIGHBOURHOOD_EPSILONS must be comma-separated positive numbers"
                        )
                    })
                })
                .collect()
        })
        .unwrap_or_else(|| DEFAULT_EPSILONS.to_vec())
}

fn make_points(num_points: usize, seed: u64) -> Vec<[f64; K]> {
    let mut points = Vec::with_capacity(num_points);
    let mut rng = SmallRng::seed_from_u64(seed);
    let distribution = Uniform::new_inclusive(-10.0, 10.0).unwrap();
    for _ in 0..num_points {
        let mut point = [0.0; K];
        for coordinate in &mut point {
            *coordinate = distribution.sample(&mut rng);
        }
        points.push(point);
    }
    points
}

fn time_queries<R>(
    queries: &[[f64; K]],
    runs: usize,
    mut query: impl FnMut(&[f64; K]) -> Vec<R>,
) -> (Duration, usize) {
    let mut best = Duration::MAX;
    let mut expected_matches = None;

    for _ in 0..runs {
        let started = Instant::now();
        let mut matches = 0usize;
        for point in queries {
            let results = query(black_box(point));
            matches = matches.wrapping_add(results.len());
            black_box(results);
        }
        let elapsed = started.elapsed();
        if let Some(expected) = expected_matches {
            assert_eq!(matches, expected, "query result count changed between runs");
        } else {
            expected_matches = Some(matches);
        }
        best = best.min(elapsed);
    }

    (best, expected_matches.unwrap())
}

fn print_result(
    library: &str,
    point_count: usize,
    query_count: usize,
    epsilon: f64,
    elapsed: Duration,
    matches: usize,
    build_elapsed: Duration,
) {
    let ns_per_query = elapsed.as_secs_f64() * 1.0e9 / query_count as f64;
    println!(
        "{library}\t{point_count}\t{query_count}\t{epsilon}\t{ns_per_query:.3}\t{matches}\t{:.6}",
        build_elapsed.as_secs_f64()
    );
}

fn main() {
    let point_count = read_usize_env("KIDDO_NEIGHBOURHOOD_POINTS", DEFAULT_POINT_COUNT);
    let query_count = read_usize_env("KIDDO_NEIGHBOURHOOD_QUERIES", DEFAULT_QUERY_COUNT);
    let runs = read_usize_env("KIDDO_NEIGHBOURHOOD_RUNS", DEFAULT_RUNS);
    let epsilons = read_epsilons();

    assert!(
        point_count > 0,
        "KIDDO_NEIGHBOURHOOD_POINTS must be positive"
    );
    assert!(
        query_count > 0,
        "KIDDO_NEIGHBOURHOOD_QUERIES must be positive"
    );
    assert!(runs > 0, "KIDDO_NEIGHBOURHOOD_RUNS must be positive");
    assert!(
        epsilons
            .iter()
            .all(|epsilon| epsilon.is_finite() && *epsilon > 0.0),
        "KIDDO_NEIGHBOURHOOD_EPSILONS must contain finite positive numbers"
    );

    eprintln!(
        "reproducing neighbourhood 0.2.0 benchmark: dims={K} points={point_count} queries={query_count} epsilons={epsilons:?} runs={runs} cube=[-10,10] point_seed={POINT_SEED} query_seed={QUERY_SEED}"
    );
    println!("library\tpoints\tqueries\tepsilon\tns_per_query\ttotal_matches\tbuild_seconds");

    let queries = make_points(query_count, QUERY_SEED);

    let points = make_points(point_count, POINT_SEED);
    let tree_points = points.clone();
    let started = Instant::now();
    let tree = NeighbourhoodTree::new(tree_points);
    let build_elapsed = started.elapsed();
    for &epsilon in &epsilons {
        let (elapsed, matches) =
            time_queries(&queries, runs, |query| tree.neighbourhood(query, epsilon));
        print_result(
            "neighbourhood_0.2.0",
            point_count,
            query_count,
            epsilon,
            elapsed,
            matches,
            build_elapsed,
        );
    }
    drop(tree);
    drop(points);

    let points = make_points(point_count, POINT_SEED);
    let tree_points = points.clone();
    let started = Instant::now();
    let tree: KiddoTree = KdTree::new_from_slice(&tree_points).unwrap();
    let build_elapsed = started.elapsed();
    for &epsilon in &epsilons {
        let radius_squared = epsilon * epsilon;
        let (elapsed, matches) = time_queries(&queries, runs, |query| {
            tree.query(query)
                .within::<SquaredEuclidean<f64>>(radius_squared)
                .unsorted()
                .without_distances()
                .execute()
        });
        print_result(
            "kiddo_v6_item_only",
            point_count,
            query_count,
            epsilon,
            elapsed,
            matches,
            build_elapsed,
        );
    }
    drop(tree);
    drop(tree_points);
    drop(points);
}

use kiddo::batch::Executor;
use kiddo::dist::{Chebyshev, Manhattan, Minkowski, SquaredEuclidean};
use kiddo::leaf_strategy::{FlatVec, VecOfArenas};
use kiddo::stem_strategy::{
    Donnelly, DonnellyCyclicSimdFull, DonnellySimdFull, DonnellyUnrolled, Eytzinger,
};
use kiddo::{KdTree, StemStrategy};

fn entries<const K: usize>(n: usize, seed: u64) -> Vec<(u32, [f64; K])> {
    let mut state = seed;
    (0..n)
        .map(|i| {
            (
                i as u32,
                std::array::from_fn(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state >> 11) as f64 / (1u64 << 53) as f64
                }),
            )
        })
        .collect()
}

fn check<SL: StemStrategy, SR: StemStrategy, const K: usize, const B: usize>(n: usize, m: usize) {
    let a = entries::<K>(n, 17);
    let b = entries::<K>(m, 123);
    let left = KdTree::<f64, u32, SL, FlatVec<f64, u32, K, B>, K, B>::new_from_entries(&a).unwrap();
    let right = KdTree::<f64, u64, SR, VecOfArenas<f64, u64, K, 7>, K, 7>::new_from_entries(
        &b.iter()
            .map(|&(i, p)| (u64::from(i), p))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    for radius in [0.0, 0.005, 0.1, 0.6, 10.0, f64::INFINITY] {
        let mut expected = Vec::new();
        for &(i, p) in &a {
            for &(j, q) in &b {
                let distance: f64 = p.iter().zip(q).map(|(x, y)| (x - y) * (x - y)).sum();
                if distance <= radius {
                    expected.push((i, u64::from(j), distance));
                }
            }
        }
        let serial = Executor::serial();
        let mut actual: Vec<_> = left
            .query_tree(&right)
            .within::<SquaredEuclidean<f64>>(radius)
            .with_executor(&serial)
            .execute()
            .into_iter()
            .map(|p| (p.left_item, p.right_item, p.distance))
            .collect();
        actual.sort_by_key(|p| (p.0, p.1));
        assert_eq!(actual, expected, "n={n} m={m} K={K} B={B} radius={radius}");
        assert_eq!(
            left.query_tree(&right)
                .within::<SquaredEuclidean<f64>>(radius)
                .with_executor(&serial)
                .count(),
            expected.len()
        );
        assert_eq!(
            left.query_tree(&right)
                .within::<SquaredEuclidean<f64>>(radius)
                .without_distances()
                .execute()
                .len(),
            expected.len()
        );
        let parallel = left
            .query_tree(&right)
            .within::<SquaredEuclidean<f64>>(radius)
            .execute();
        let mut parallel: Vec<_> = parallel
            .into_iter()
            .map(|p| (p.left_item, p.right_item, p.distance))
            .collect();
        parallel.sort_by_key(|p| (p.0, p.1));
        assert_eq!(parallel, expected);
    }
}

#[test]
fn dual_tree_nearest_one_matches_brute_force() {
    let a = entries::<3>(137, 17);
    let b = entries::<3>(113, 123);
    let left =
        KdTree::<f64, u32, Donnelly<3>, FlatVec<f64, u32, 3, 32>, 3, 32>::new_from_entries(&a)
            .unwrap();
    let right = KdTree::<f64, u64, Eytzinger, VecOfArenas<f64, u64, 3, 7>, 3, 7>::new_from_entries(
        &b.iter()
            .map(|&(item, point)| (u64::from(item), point))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let mut actual: Vec<_> = left
        .query_tree(&right)
        .nearest_one::<SquaredEuclidean<f64>>()
        .execute()
        .into_iter()
        .map(|p| (p.left_item, p.right_item, p.distance))
        .collect();
    actual.sort_by_key(|entry| entry.0);
    let expected: Vec<_> = a
        .iter()
        .map(|&(left_item, point)| {
            let (right_item, distance) = b
                .iter()
                .map(|&(right_item, target)| {
                    (
                        u64::from(right_item),
                        point
                            .iter()
                            .zip(target)
                            .map(|(a, b)| (a - b) * (a - b))
                            .sum::<f64>(),
                    )
                })
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            (left_item, right_item, distance)
        })
        .collect();
    assert_eq!(actual, expected);
}

#[test]
fn mixed_layouts_and_shapes() {
    for (n, m) in [
        (0, 0),
        (0, 3),
        (1, 1),
        (3, 2),
        (9, 17),
        (31, 47),
        (65, 37),
        (137, 93),
    ] {
        check::<Eytzinger, Eytzinger, 1, 4>(n, m);
        check::<Eytzinger, Donnelly<3>, 2, 8>(n, m);
        check::<Donnelly<3>, Eytzinger, 3, 16>(n, m);
        check::<DonnellyUnrolled<4>, Donnelly<3>, 7, 4>(n, m);
        check::<DonnellySimdFull<3>, DonnellyCyclicSimdFull<4>, 3, 4>(n, m);
    }
}

#[test]
fn remaining_layout_variants() {
    use kiddo::stem_strategy::{
        DonnellyCyclicSimdDescent, DonnellyNoPf, DonnellySimdDescent, DonnellyUnrolledBlockDim,
        EytzingerNoPf,
    };
    check::<EytzingerNoPf, DonnellyNoPf<4>, 3, 4>(63, 93);
    check::<DonnellyUnrolledBlockDim<3>, DonnellySimdDescent<4>, 2, 8>(129, 37);
    check::<DonnellyCyclicSimdDescent<4>, DonnellySimdFull<3>, 7, 4>(91, 65);
}

#[test]
fn duplicates_and_built_in_metrics() {
    type Tree = KdTree<f32, u32, Eytzinger, VecOfArenas<f32, u32, 2, 4>, 2, 4>;
    let left = Tree::new_from_entries(&vec![(7, [0.0, 0.0]); 97]).unwrap();
    let right = Tree::new_from_entries(&vec![(7, [0.0, -0.0]); 133]).unwrap();
    assert_eq!(
        left.query_tree(&right)
            .within::<SquaredEuclidean<f32>>(0.0)
            .count(),
        97 * 133
    );
    assert_eq!(
        left.query_tree(&right)
            .within::<SquaredEuclidean<f64>>(0.0)
            .execute()
            .len(),
        97 * 133
    );
    assert_eq!(
        left.query_tree(&right)
            .within::<Manhattan<f32>>(0.0)
            .count(),
        97 * 133
    );
    assert_eq!(
        left.query_tree(&right)
            .within::<Chebyshev<f32>>(0.0)
            .count(),
        97 * 133
    );
    assert_eq!(
        left.query_tree(&right)
            .within::<Minkowski<3, f64>>(0.0)
            .count(),
        97 * 133
    );
    assert_eq!(
        left.query_tree(&right)
            .within::<SquaredEuclidean<f32>>(0.0)
            .exclusive_boundaries()
            .count(),
        0
    );
}

#[test]
fn boundaries_and_overflow() {
    type Tree = KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, 2, 4>, 2, 4>;
    let left = Tree::new_from_slice(&[[0.0, 0.0], [f64::MAX, 0.0]]).unwrap();
    let right = Tree::new_from_slice(&[[1.0, 0.0], [-f64::MAX, 0.0]]).unwrap();
    assert_eq!(
        left.query_tree(&right)
            .within::<SquaredEuclidean<f64>>(1.0)
            .count(),
        1
    );
    assert_eq!(
        left.query_tree(&right)
            .within::<SquaredEuclidean<f64>>(1.0)
            .exclusive_boundaries()
            .count(),
        0
    );
    assert_eq!(
        left.query_tree(&right)
            .within::<SquaredEuclidean<f64>>(1.0f64.next_down())
            .count(),
        0
    );
    assert_eq!(
        left.query_tree(&right)
            .within::<SquaredEuclidean<f64>>(1.0f64.next_up())
            .count(),
        1
    );
    assert_eq!(
        left.query_tree(&right)
            .within::<SquaredEuclidean<f64>>(f64::INFINITY)
            .count(),
        4
    );
    assert_eq!(
        left.query_tree(&right)
            .within::<SquaredEuclidean<f64>>(f64::INFINITY)
            .exclusive_boundaries()
            .count(),
        1
    );
}

#[test]
fn rejects_same_object_and_invalid_thresholds() {
    type Tree = KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, 2, 4>, 2, 4>;
    for data in [vec![], vec![[0.0, 0.0]]] {
        let left = Tree::new_from_slice(&data).unwrap();
        let right = Tree::new_from_slice(&data).unwrap();
        assert!(std::panic::catch_unwind(|| left.query_tree(&left)).is_err());
        assert_eq!(
            left.query_tree(&right)
                .within::<SquaredEuclidean<f64>>(0.0)
                .count(),
            data.len()
        );
        for radius in [-1.0, f64::NAN, f64::NEG_INFINITY] {
            assert!(std::panic::catch_unwind(|| left
                .query_tree(&right)
                .within::<SquaredEuclidean<f64>>(radius))
            .is_err());
        }
    }
}

#[test]
fn nonzero_metric_and_float32_boundaries() {
    use kiddo::dist::DistanceMetricScalar;
    type Tree = KdTree<f32, u32, Donnelly<3>, VecOfArenas<f32, u32, 3, 7>, 3, 7>;
    let a: Vec<_> = entries::<3>(93, 77)
        .into_iter()
        .map(|(i, p)| (i, p.map(|x| x as f32)))
        .collect();
    let b: Vec<_> = entries::<3>(71, 19)
        .into_iter()
        .map(|(i, p)| (i, p.map(|x| x as f32)))
        .collect();
    let left = Tree::new_from_entries(&a).unwrap();
    let right = Tree::new_from_entries(&b).unwrap();
    macro_rules! check_metric {
        ($metric:ty) => {{
            for index in 0..8 {
                let edge =
                    <$metric as DistanceMetricScalar<f32>>::dist_raw(&a[index].1, &b[index].1);
                for radius in [edge.next_down(), edge, edge.next_up()] {
                    let mut inclusive = Vec::new();
                    let mut exclusive = Vec::new();
                    for &(i, p) in &a {
                        for &(j, q) in &b {
                            let d = <$metric as DistanceMetricScalar<f32>>::dist_raw(&p, &q);
                            if d <= radius {
                                inclusive.push((i, j, d));
                            }
                            if d < radius {
                                exclusive.push((i, j, d));
                            }
                        }
                    }
                    let mut actual: Vec<_> = left
                        .query_tree(&right)
                        .within::<$metric>(radius)
                        .execute()
                        .into_iter()
                        .map(|x| (x.left_item, x.right_item, x.distance))
                        .collect();
                    actual.sort_by_key(|p| (p.0, p.1));
                    assert_eq!(actual, inclusive);
                    let mut strict: Vec<_> = left
                        .query_tree(&right)
                        .within::<$metric>(radius)
                        .exclusive_boundaries()
                        .execute()
                        .into_iter()
                        .map(|x| (x.left_item, x.right_item, x.distance))
                        .collect();
                    strict.sort_by_key(|p| (p.0, p.1));
                    assert_eq!(strict, exclusive);
                    assert_eq!(
                        left.query_tree(&right).within::<$metric>(radius).count(),
                        inclusive.len()
                    );
                    assert_eq!(
                        left.query_tree(&right)
                            .within::<$metric>(radius)
                            .exclusive_boundaries()
                            .count(),
                        exclusive.len()
                    );
                }
            }
        }};
    }
    check_metric!(SquaredEuclidean<f32>);
    check_metric!(SquaredEuclidean<f64>);
    check_metric!(Manhattan<f32>);
    check_metric!(Chebyshev<f32>);
    check_metric!(Minkowski<3,f32>);
    check_metric!(Minkowski<4,f64>);
}

#[test]
fn large_workload_rounding_regressions() {
    // Reduced from the 2^24-point differential diagnostic. Historical SIMD
    // radius queries round these distances down onto the boundary; scalar
    // metric arithmetic puts them one ULP above it. The join follows scalar.
    type Tree = KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, 3, 32>, 3, 32>;
    for (p, q) in [
        (
            [0.08790941, 0.8717605, 0.5734007],
            [0.09077622, 0.8729093, 0.5681331],
        ),
        (
            [0.5666268, 0.083839536, 0.5078225],
            [0.56211233, 0.087421425, 0.50580376],
        ),
        (
            [0.8286047, 0.6387919, 0.60467845],
            [0.828009, 0.64475083, 0.60587096],
        ),
        (
            [0.31691444, 0.8285374, 0.07836475],
            [0.31279126, 0.8324673, 0.08056492],
        ),
    ] {
        let left = Tree::new_from_slice(&[p; 17]).unwrap();
        let right = Tree::new_from_slice(&[q; 32]).unwrap();
        let radius = f32::from_bits(941384473);
        assert_eq!(
            left.query_tree(&right)
                .within::<SquaredEuclidean<f32>>(radius)
                .count(),
            0
        );
        assert!(left
            .query_tree(&right)
            .within::<SquaredEuclidean<f32>>(radius)
            .execute()
            .is_empty());
        assert_eq!(
            left.query_tree(&right)
                .within::<SquaredEuclidean<f32>>(radius.next_up())
                .count(),
            17 * 32
        );
    }
}

#[cfg(feature = "multi-threaded")]
#[test]
fn owned_pools_visitors_and_panic_recovery() {
    use kiddo::batch::ThreadPoolBuilder;
    use std::sync::{Arc, Mutex};
    type Tree = KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, 3, 32>, 3, 32>;
    let left = Tree::new_from_entries(&entries::<3>(531, 17)).unwrap();
    let right = Tree::new_from_entries(&entries::<3>(731, 78)).unwrap();
    let serial = Executor::serial();
    let mut expected: Vec<_> = left
        .query_tree(&right)
        .within::<SquaredEuclidean<f64>>(0.08)
        .with_executor(&serial)
        .execute()
        .into_iter()
        .map(|x| (x.left_item, x.right_item))
        .collect();
    expected.sort_unstable();
    for threads in [1, 2, 4, 8] {
        let pool = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|i| format!("join-owned-{i}"))
                .build()
                .unwrap(),
        );
        let executor = Executor::parallel_in_pool(pool).with_default_static_chunking();
        let output = Mutex::new(Vec::new());
        left.query_tree(&right)
            .within::<SquaredEuclidean<f64>>(0.08)
            .without_distances()
            .with_executor(&executor)
            .for_each(|pair| {
                assert!(std::thread::current()
                    .name()
                    .unwrap()
                    .starts_with("join-owned-"));
                output
                    .lock()
                    .unwrap()
                    .push((pair.left_item, pair.right_item));
            });
        let mut output = output.into_inner().unwrap();
        output.sort_unstable();
        assert_eq!(output, expected);
        assert_eq!(
            left.query_tree(&right)
                .within::<SquaredEuclidean<f64>>(0.08)
                .with_executor(&executor)
                .count(),
            expected.len()
        );
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            left.query_tree(&right)
                .within::<SquaredEuclidean<f64>>(0.08)
                .with_executor(&executor)
                .for_each(|_| panic!("visitor failure"));
        }))
        .is_err());
        assert_eq!(
            left.query_tree(&right)
                .within::<SquaredEuclidean<f64>>(0.08)
                .with_executor(&executor)
                .count(),
            expected.len()
        );
    }
}

/// Expensive differential diagnostic; emits minimal coordinate/threshold
/// fixtures when the historical radius kernel and scalar metric disagree.
#[test]
#[ignore]
fn large_radius_boundary_diagnostic() {
    use kiddo::dist::DistanceMetricScalar;
    use std::sync::atomic::{AtomicU32, Ordering};
    type Tree = KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, 3, 32>, 3, 32>;
    let n = 1 << 24;
    let make = |mut state: u64| {
        (0..n)
            .map(|_| {
                std::array::from_fn(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    ((state >> 11) as f64 / (1u64 << 53) as f64) as f32
                })
            })
            .collect::<Vec<[f32; 3]>>()
    };
    let a = make(17);
    let b = make(78);
    let left = Tree::new_from_slice(&a).unwrap();
    let right = Tree::new_from_slice(&b).unwrap();
    let radius = (16.0 / (n as f64 * (4.0 * std::f64::consts::PI / 3.0))).powf(1.0 / 3.0) as f32;
    let r2 = radius * radius;
    let counts: Vec<_> = (0..n).map(|_| AtomicU32::new(0)).collect();
    left.query_tree(&right)
        .within::<SquaredEuclidean<f32>>(r2)
        .for_each(|p| {
            counts[p.left_item as usize].fetch_add(1, Ordering::Relaxed);
        });
    for (i, p) in a.iter().enumerate() {
        let mut old = Vec::new();
        right
            .query(p)
            .within::<SquaredEuclidean<f32>>(r2)
            .unsorted()
            .visit(|q| old.push(q));
        let new = counts[i].load(Ordering::Relaxed) as usize;
        if old.len() != new {
            let mut exact = Vec::new();
            for (j, q) in b.iter().enumerate() {
                let d = <SquaredEuclidean<f32> as DistanceMetricScalar<f32>>::dist_raw(p, q);
                if d <= r2 {
                    exact.push(j as u32);
                }
            }
            eprintln!(
                "row={i} old={} new={new} oracle={} r2={r2:?} r2_bits={} p={p:?}",
                old.len(),
                exact.len(),
                r2.to_bits()
            );
            for result in old {
                if !exact.contains(&result.item) {
                    let q = b[result.item as usize];
                    let d = <SquaredEuclidean<f32> as DistanceMetricScalar<f32>>::dist_raw(p, &q);
                    eprintln!(
                        "old-only: q={q:?} old_dist={:?} scalar={d:?}",
                        result.distance
                    );
                }
            }
            assert_eq!(new, exact.len());
        }
    }
}

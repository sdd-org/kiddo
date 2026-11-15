//! Metric-independent differential checks for generated metric-pruning sweeps.
use super::*;
use kiddo::dist::{DistanceMetric, QueryDistance};

pub(super) fn validate_metric<A, SO, LS, D, const K: usize, const B: usize>(
    tree: &KdTree<A, usize, SO, LS, K, B>,
    points: &[[A; K]],
    query: &[A; K],
    max_qty: usize,
    radius: D::Output,
) -> Result<(), String>
where
    A: Axis<Coord = A> + 'static,
    SO: StemStrategy,
    LS: LeafStrategy<A, usize, SO, K, B>,
    D: DistanceMetric<A>,
    D::Output: QueryDistance,
{
    let metric = std::any::type_name::<D>();
    let mut oracle = MetricState::new(max_qty, radius);
    for (item, point) in points.iter().enumerate() {
        oracle.update(D::dist_raw(query, point), item);
    }
    let expected = oracle.take_nearest_n_sorted();
    let within = oracle.take_within_sorted();
    let n = NonZeroUsize::new(max_qty).unwrap();

    // In particular, do not allow the relaxed kth-boundary tie comparison to
    // conceal nonexistent/duplicate items or incorrect reported distances.
    let validate_items = |got: &[(D::Output, usize)]| -> Result<(), String> {
        let mut seen = HashSet::new();
        for &(distance, item) in got {
            let point = points
                .get(item)
                .ok_or_else(|| format!("invalid item {item}"))?;
            if !seen.insert(item) {
                return Err(format!("duplicate item {item}"));
            }
            let actual = D::dist_raw(query, point);
            if !distances_match_for_fuzz(distance, actual) {
                return Err(format!("item {item}: reported {distance}, actual {actual}"));
            }
        }
        Ok(())
    };
    let check_nearest = |expected: &[(D::Output, usize)],
                         mut got: Vec<(D::Output, usize)>,
                         op: &str| {
        validate_items(&got)
            .and_then(|()| {
                if op != "nearest_n_unsorted" && got.windows(2).any(|pair| pair[0].0 > pair[1].0) {
                    return Err("results are not sorted by distance".to_owned());
                }
                sort_by_distance_then_index(&mut got);
                compare_nearest_n_sorted(expected, &got)
            })
            .map_err(|error| format!("metric={metric} op={op} {error}"))
    };
    check_nearest(
        &expected,
        query_nearest_n_sorted::<_, _, _, _, D, K, B>(tree, query, n),
        "nearest_n",
    )?;
    check_nearest(
        &expected,
        query_nearest_n_unsorted::<_, _, _, _, D, K, B>(tree, query, n),
        "nearest_n_unsorted",
    )?;
    let bounded = tree
        .query(query)
        .nearest_n::<D>(n)
        .within(radius)
        .execute()
        .into_iter()
        .map(|r| (r.distance, r.item))
        .collect();
    check_nearest(
        &within[..within.len().min(max_qty)],
        bounded,
        "nearest_n_within",
    )?;

    if !points.is_empty() {
        let got = tree.query(query).nearest_one::<D>().execute();
        validate_items(&[(got.distance, got.item)])?;
        if !distances_match_for_fuzz(got.distance, oracle.best_dist) {
            return Err(format!(
                "metric={metric} op=nearest_one expected={} got={}",
                oracle.best_dist, got.distance
            ));
        }
    }
    for (op, mut got) in [
        (
            "within",
            query_within_sorted::<_, _, _, _, D, K, B>(tree, query, radius),
        ),
        (
            "within_unsorted",
            query_within_unsorted::<_, _, _, _, D, K, B>(tree, query, radius),
        ),
        (
            "within_iter",
            tree.query(query)
                .within::<D>(radius)
                .unsorted()
                .iter()
                .map(|r| (r.distance, r.item))
                .collect(),
        ),
    ] {
        validate_items(&got)?;
        compare_within_results(&mut within.clone(), &mut got, radius)
            .map_err(|error| format!("metric={metric} op={op} {error}"))?;
    }
    let expected_best = expected_best_n_within(&within, max_qty);
    let mut best = tree
        .query(query)
        .best_n_within::<D>(radius, n)
        .execute()
        .into_iter()
        .map(|r| (r.distance, r.item))
        .collect::<Vec<_>>();
    validate_items(&best)?;
    sort_by_item_idx(&mut best);
    compare_item_sorted_results(&expected_best, &best)
        .map_err(|error| format!("metric={metric} op=best_n_within {error}"))
}

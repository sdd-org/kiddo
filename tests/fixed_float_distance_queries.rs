#![cfg(feature = "fixed")]

use fixed::{types::extra::U16, FixedI32};
use kiddo::{Chebyshev, ImmutableKdTree, Manhattan, Minkowski, SquaredEuclidean};

type Fixed = FixedI32<U16>;

fn fixed(value: f64) -> Fixed {
    Fixed::from_num(value)
}

fn tree_and_query() -> (ImmutableKdTree<Fixed, 2>, [Fixed; 2]) {
    let points = [
        [fixed(-4.0), fixed(3.0)],
        [fixed(1.5), fixed(2.25)],
        [fixed(8.0), fixed(-2.0)],
    ];

    (
        ImmutableKdTree::new_from_slice(&points).unwrap(),
        [fixed(1.75), fixed(2.0)],
    )
}

#[test]
fn fixed_tree_supports_f32_distances() {
    let (tree, query) = tree_and_query();
    let squared_euclidean = tree
        .query(&query)
        .nearest_one::<SquaredEuclidean<f32>>()
        .execute();
    let manhattan = tree.query(&query).nearest_one::<Manhattan<f32>>().execute();
    let chebyshev = tree.query(&query).nearest_one::<Chebyshev<f32>>().execute();
    let minkowski = tree
        .query(&query)
        .nearest_one::<Minkowski<3, f32>>()
        .execute();

    assert_eq!(squared_euclidean.item, 1);
    assert_eq!(squared_euclidean.distance, 0.125);
    assert_eq!(manhattan.item, 1);
    assert_eq!(chebyshev.item, 1);
    assert_eq!(minkowski.item, 1);
}

#[test]
fn fixed_tree_supports_f64_distances() {
    let (tree, query) = tree_and_query();
    let squared_euclidean = tree
        .query(&query)
        .nearest_one::<SquaredEuclidean<f64>>()
        .execute();
    let manhattan = tree.query(&query).nearest_one::<Manhattan<f64>>().execute();
    let chebyshev = tree.query(&query).nearest_one::<Chebyshev<f64>>().execute();
    let minkowski = tree
        .query(&query)
        .nearest_one::<Minkowski<3, f64>>()
        .execute();

    assert_eq!(squared_euclidean.item, 1);
    assert_eq!(squared_euclidean.distance, 0.125);
    assert_eq!(manhattan.item, 1);
    assert_eq!(chebyshev.item, 1);
    assert_eq!(minkowski.item, 1);
}

use kiddo::{Chebyshev, ImmutableKdTree, NearestNeighbour};

#[test]
fn immutable_nearest_one_uses_chebyshev_accumulation() {
    let points = [[0.0f64, 0.0], [10.0, 10.0]];
    let tree: ImmutableKdTree<f64, 2> = ImmutableKdTree::new_from_slice(&points);

    assert_eq!(
        tree.nearest_one::<Chebyshev>(&[9.0, 9.0]),
        NearestNeighbour {
            distance: 1.0,
            item: 1,
        }
    );
}

#[test]
fn immutable_approx_nearest_one_uses_chebyshev_accumulation() {
    let points = [[0.0f64, 0.0], [10.0, 10.0]];
    let tree: ImmutableKdTree<f64, 2> = ImmutableKdTree::new_from_slice(&points);

    assert_eq!(
        tree.approx_nearest_one::<Chebyshev>(&[9.0, 9.0]),
        NearestNeighbour {
            distance: 1.0,
            item: 1,
        }
    );
}

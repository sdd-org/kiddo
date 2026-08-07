use kiddo::{ImmutableKdTree, SquaredEuclidean};

fn main() {
    let points = [[0.0_f64, 0.0], [1.0, 1.0]];
    let tree = ImmutableKdTree::new_from_slice(&points).unwrap();

    let nearest = tree
        .query(&[0.0, 0.0])
        .nearest_one::<SquaredEuclidean<f64>>()
        .execute();

    assert_eq!(nearest.item, 0);
    assert_eq!(nearest.distance, 0.0);
}

use kiddo::{MutableKdTree, SquaredEuclidean};

#[test]
fn exact_point_remains_reachable_after_mutable_split() {
    let coords = [
        3.0f32, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 2.0, 3.0, 0.0, 3.0,
        0.0, 0.0, 0.0, 2.0, 0.0, 3.0, 2.0, 0.0, 3.0, 3.0, 2.0, 0.0, 0.0, 3.0, 0.0,
    ];

    let mut tree: MutableKdTree<f32, 1> = MutableKdTree::default();
    for (item, coord) in coords.iter().enumerate() {
        tree.add(&[*coord], item as u32).unwrap();
    }

    let before = tree
        .query(&[1.0])
        .within::<SquaredEuclidean<f32>>(0.0)
        .execute();
    assert!(before.iter().any(|result| result.item == 1));

    // Adding one more point splits the full bucket and exposes the split boundary.
    tree.add(&[0.5], coords.len() as u32).unwrap();

    let after = tree
        .query(&[1.0])
        .within::<SquaredEuclidean<f32>>(0.0)
        .execute();
    assert!(
        after.iter().any(|result| result.item == 1),
        "point 1 became unreachable after the mutable split"
    );
}

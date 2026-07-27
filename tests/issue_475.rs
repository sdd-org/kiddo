use kiddo::{KdTree, SquaredEuclidean};

#[test]
fn add_accepts_more_than_one_bucket_of_points_sharing_an_axis_coordinate() {
    let mut tree: KdTree<f64, 2> = KdTree::new();

    for item in 0..33u64 {
        tree.add(&[5.0, item as f64], item);
    }

    assert_eq!(tree.size(), 33);

    for item in 0..33u64 {
        let nearest = tree.nearest_one::<SquaredEuclidean>(&[5.0, item as f64]);
        assert_eq!(nearest.distance, 0.0);
        assert_eq!(nearest.item, item);
    }

    for item in 0..33u64 {
        assert_eq!(tree.remove(&[5.0, item as f64], item), 1);
    }
    assert_eq!(tree.size(), 0);
}

#[test]
fn add_can_advance_past_constant_axes_in_either_direction() {
    type SmallTree = kiddo::float::kdtree::KdTree<f64, u64, 2, 4, u32>;

    let mut above: SmallTree = SmallTree::new();
    for item in 0..4 {
        above.add(&[5.0, 5.0], item);
    }
    above.add(&[5.0, 6.0], 4);

    let mut below: SmallTree = SmallTree::new();
    for item in 0..4 {
        below.add(&[5.0, 5.0], item);
    }
    below.add(&[5.0, 4.0], 4);

    assert_eq!(above.nearest_one::<SquaredEuclidean>(&[5.0, 6.0]).item, 4);
    assert_eq!(below.nearest_one::<SquaredEuclidean>(&[5.0, 4.0]).item, 4);
}

#[test]
#[should_panic(
    expected = "Cannot insert another item at [5.0, 5.0]: this leaf already contains 4 items at \
                exactly the same point."
)]
fn add_rejects_only_an_unsplittable_bucket_of_identical_points() {
    type SmallTree = kiddo::float::kdtree::KdTree<f64, u64, 2, 4, u32>;

    let mut tree: SmallTree = SmallTree::new();
    for item in 0..5 {
        tree.add(&[5.0, 5.0], item);
    }
}

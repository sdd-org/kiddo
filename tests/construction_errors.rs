use kiddo::kd_tree::ConstructionError;
use kiddo::leaf_strategies::VecOfArrays;
use kiddo::stem_strategy::Eytzinger;
use kiddo::KdTree;

#[test]
fn unsplittable_bucket_error_explains_fixed_capacity() {
    type Tree = KdTree<f32, u32, Eytzinger, VecOfArrays<f32, u32, 3, 2>, 3, 2>;

    let points = [[1.0f32, 1.0, 1.0]; 3];
    let error = match Tree::new_from_slice(&points) {
        Ok(_) => panic!("expected an unsplittable hard bucket"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        ConstructionError::UnsplittableBucket { split_dim: 0 }
    ));
    assert_eq!(
        error.to_string(),
        "cannot split a fixed capacity leaf bucket on dimension 0 because every point in the bucket has the same value there"
    );
}

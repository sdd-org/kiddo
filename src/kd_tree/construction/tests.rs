use super::*;
use crate::dist::SquaredEuclidean;
use crate::leaf_strategy::FlatVec;
use crate::leaf_strategy::VecOfArenas;
use crate::leaf_strategy::VecOfArrays;
use crate::Eytzinger;

#[test]
fn construction_index_selection_is_adaptive() {
    assert!(construction_index_fits_u32(0));
    assert!(construction_index_fits_u32(u32::MAX as usize));

    #[cfg(target_pointer_width = "64")]
    assert!(!construction_index_fits_u32(u32::MAX as usize + 1));
}

#[test]
fn update_pivot_shifts_right_when_left_scan_hits_zero() {
    type TestTree = KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, 2, 32>, 2, 32>;

    let source = [
        [1.0f32, 10.0],
        [1.0, 20.0],
        [1.0, 30.0],
        [2.0, 40.0],
        [3.0, 50.0],
    ];
    let mut sort_index = [0u32, 1, 2, 3, 4];

    let pivot = TestTree::update_pivot(
        &source,
        &|point: &[f32; 2], dim| point[dim],
        &mut sort_index,
        0,
        1,
    )
    .unwrap();

    assert_eq!(pivot, 3);
    assert_eq!(sort_index, [0, 1, 2, 3, 4]);
    assert_eq!(source[sort_index[pivot - 1].as_usize()][0], 1.0);
    assert_eq!(source[sort_index[pivot].as_usize()][0], 2.0);
}

#[test]
fn replace_item_updates_flat_vec_tree_without_changing_size() {
    type TestTree = KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, 2, 32>, 2, 32>;

    let entries = [
        (10u32, [1.0f32, 10.0]),
        (11u32, [2.0, 20.0]),
        (12u32, [1.0, 10.0]),
    ];
    let mut tree = TestTree::new_from_entries(&entries).unwrap();

    assert_eq!(tree.size(), 3);
    tree.replace_item(&[1.0, 10.0], 10, 99).unwrap();
    assert_eq!(tree.size(), 3);

    let iterated = tree.iter().collect::<Vec<_>>();
    assert_eq!(iterated[0], (99, [1.0, 10.0]));
    assert_eq!(iterated[1], (11, [2.0, 20.0]));
    assert_eq!(iterated[2], (12, [1.0, 10.0]));
}

#[test]
fn replace_item_returns_entry_not_found_when_exact_match_is_missing() {
    type TestTree = KdTree<f32, u32, Eytzinger, VecOfArrays<f32, u32, 2, 32>, 2, 32>;

    let entries = [(10u32, [1.0f32, 10.0]), (11u32, [2.0, 20.0])];
    let mut tree = TestTree::new_from_entries(&entries).unwrap();

    assert_eq!(
        tree.replace_item(&[1.0, 10.0], 99, 100),
        Err(MutationError::EntryNotFound)
    );
    assert_eq!(
        tree.replace_item(&[9.0, 90.0], 10, 100),
        Err(MutationError::EntryNotFound)
    );
}

#[test]
fn replace_item_updates_vec_of_arenas_tree() {
    type TestTree = KdTree<f64, u32, Eytzinger, VecOfArenas<f64, u32, 2, 32>, 2, 32>;

    let entries = [
        (20u32, [1.0f64, 10.0]),
        (21u32, [2.0, 20.0]),
        (22u32, [3.0, 30.0]),
    ];
    let mut tree = TestTree::new_from_entries(&entries).unwrap();

    tree.replace_item(&[2.0, 20.0], 21, 77).unwrap();

    let iterated = tree.iter().collect::<Vec<_>>();
    assert_eq!(
        iterated,
        vec![(20, [1.0, 10.0]), (77, [2.0, 20.0]), (22, [3.0, 30.0])]
    );
}

#[test]
fn irregular_immutable_soft_layout_preserves_arithmetic_resolution() {
    type TestTree = KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, 2, 2>, 2, 2>;

    let points = vec![
        [3.0, 0.0],
        [1.0, 0.6],
        [1.0, 1.4],
        [3.0, 3.3],
        [3.0, 3.8],
        [0.0, 1.8],
        [3.0, 1.5],
        [3.0, 2.7],
        [1.0, 3.3],
    ];
    let query = [2.9142656, 5.220647];

    let tree = TestTree::new_from_slice(&points).unwrap();
    assert!(tree.stem_leaf_resolution.uses_arithmetic());
    assert_eq!(tree.leaf_count(), 8);
    assert_eq!(tree.max_leaf_len(), 3);
    assert_eq!(
        (0..tree.leaf_count())
            .map(|leaf_idx| {
                <FlatVec<f32, u32, 2, 2> as LeafStrategy<f32, u32, Eytzinger, 2, 2>>::leaf_len(
                    &tree.leaves,
                    leaf_idx,
                )
            })
            .collect::<Vec<_>>(),
        vec![2, 0, 1, 1, 3, 0, 2, 0]
    );

    let result = tree
        .query(&query)
        .nearest_one::<SquaredEuclidean<f32>>()
        .execute();
    assert_eq!(result.item, 4);
    assert!((result.distance - 2.025588).abs() < 1.0e-6);
}

#[test]
fn irregular_hard_terminal_layout_is_detected_and_mapped() {
    type TestTree = KdTree<f32, u32, Eytzinger, VecOfArrays<f32, u32, 2, 2>, 2, 2>;

    let terminal_stem_indices = vec![8usize, 10, 3];

    assert!(!TestTree::terminal_stem_indices_match_arithmetic_layout(
        &terminal_stem_indices,
        2,
    ));

    let stem_leaf_resolution =
        TestTree::mapped_stem_leaf_resolution_from_terminals(&terminal_stem_indices);
    assert!(!stem_leaf_resolution.uses_arithmetic());
    assert_eq!(stem_leaf_resolution.resolve_terminal_stem_idx(8, 0), 0);
    assert_eq!(stem_leaf_resolution.resolve_terminal_stem_idx(10, 0), 1);
    assert_eq!(stem_leaf_resolution.resolve_terminal_stem_idx(3, 0), 2);
}

#[test]
fn unsplittable_immutable_hard_bucket_returns_error() {
    type TestTree = KdTree<f32, u32, Eytzinger, VecOfArrays<f32, u32, 2, 2>, 2, 2>;

    let points = vec![[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]];

    assert!(matches!(
        TestTree::new_from_slice(&points),
        Err(ConstructionError::UnsplittableBucket { split_dim: 0 })
    ));
}

#[test]
fn parallel_construction_threshold_is_inclusive() {
    type TestTree = KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, 2, 32>, 2, 32>;

    let policy = ParallelConstruction::with_threshold(1_024);
    let default = TestTree::builder();
    let forced = TestTree::builder().with_parallel_construction();
    let zero_threshold = TestTree::builder().with_parallel_construction_threshold(0);

    assert!(!policy.should_parallelize(1_023));
    assert!(policy.should_parallelize(1_024));
    assert!(policy.should_parallelize(1_025));
    assert_eq!(
        default.policy.threshold(),
        DEFAULT_PARALLEL_CONSTRUCTION_THRESHOLD
    );
    assert_eq!(forced.policy.threshold(), 1);
    assert_eq!(zero_threshold.policy.threshold(), 1);
    assert_eq!(DEFAULT_PARALLEL_CONSTRUCTION_THRESHOLD, 262_144);
}

#[test]
fn parallel_soft_construction_matches_sequential_construction() {
    type FlatTree = KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, 2, 32>, 2, 32>;
    type ArenaTree = KdTree<f32, u32, Eytzinger, VecOfArenas<f32, u32, 2, 32>, 2, 32>;

    let points = (0..4_096)
        .map(|idx| {
            [
                ((idx * 17) % 997) as f32,
                ((idx * 53 + idx / 7) % 991) as f32,
            ]
        })
        .collect::<Vec<_>>();

    macro_rules! assert_parallel_matches {
        ($tree:ty) => {{
            let sequential = <$tree>::builder()
                .with_serial_construction()
                .build_from_slice(&points)
                .unwrap();
            let parallel = <$tree>::builder()
                .with_parallel_construction()
                .build_from_slice(&points)
                .unwrap();
            let threshold_parallel = <$tree>::builder()
                .with_parallel_construction_threshold(points.len())
                .build_from_slice(&points)
                .unwrap();

            assert_eq!(sequential.stems.as_slice(), parallel.stems.as_slice());
            assert_eq!(
                sequential.stems.as_slice(),
                threshold_parallel.stems.as_slice()
            );
            assert_eq!(sequential.size(), parallel.size());
            assert_eq!(sequential.leaf_count(), parallel.leaf_count());
            assert_eq!(sequential.max_leaf_len(), parallel.max_leaf_len());
            assert_eq!(
                sequential.iter().collect::<Vec<_>>(),
                parallel.iter().collect::<Vec<_>>()
            );

            for query in [[0.0, 0.0], [500.0, 500.0], [996.0, 990.0]] {
                assert_eq!(
                    sequential
                        .query(&query)
                        .nearest_one::<SquaredEuclidean<f32>>()
                        .execute(),
                    parallel
                        .query(&query)
                        .nearest_one::<SquaredEuclidean<f32>>()
                        .execute()
                );
            }
        }};
    }

    assert_parallel_matches!(FlatTree);
    assert_parallel_matches!(ArenaTree);
}

#[test]
fn parallel_constructor_preserves_hard_bucket_construction() {
    type TestTree = KdTree<f32, u32, Eytzinger, VecOfArrays<f32, u32, 2, 32>, 2, 32>;

    let points = (0..4_096)
        .map(|idx| [idx as f32, ((idx * 31) % 4_099) as f32])
        .collect::<Vec<_>>();
    let sequential = TestTree::new_from_slice(&points).unwrap();
    let parallel = TestTree::builder()
        .with_parallel_construction()
        .build_from_slice(&points)
        .unwrap();

    assert_eq!(sequential.stems.as_slice(), parallel.stems.as_slice());
    assert_eq!(
        sequential.iter().collect::<Vec<_>>(),
        parallel.iter().collect::<Vec<_>>()
    );
}

#[test]
fn builder_supports_parallel_entries_sources_and_no_items() {
    type ItemTree = KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, 2, 32>, 2, 32>;
    type NoItemsTree = KdTree<f32, (), Eytzinger, FlatVec<f32, (), 2, 32>, 2, 32>;

    let entries = (0..2_048)
        .map(|idx| (idx as u32 + 10, [idx as f32, (idx % 127) as f32]))
        .collect::<Vec<_>>();

    let from_entries = ItemTree::builder()
        .with_parallel_construction()
        .build_from_entries(&entries)
        .unwrap();
    let from_source = ItemTree::builder()
        .with_parallel_construction()
        .build_from_source(
            &entries,
            |entry, dim| entry.1[dim],
            |_src_idx, entry| entry.0,
        )
        .unwrap();
    assert_eq!(
        from_entries.iter().collect::<Vec<_>>(),
        from_source.iter().collect::<Vec<_>>()
    );

    let points = entries.iter().map(|entry| entry.1).collect::<Vec<_>>();
    let no_items = NoItemsTree::builder()
        .with_parallel_construction()
        .build_from_slice_no_items(&points)
        .unwrap();
    assert_eq!(no_items.size(), points.len());
    assert_eq!(no_items.iter().count(), points.len());
}

#[test]
fn serial_builder_supports_non_sync_sources() {
    use std::cell::Cell;

    struct Point {
        coords: [Cell<f32>; 2],
    }

    type TestTree = KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, 2, 32>, 2, 32>;

    let points = [
        Point {
            coords: [Cell::new(1.0), Cell::new(2.0)],
        },
        Point {
            coords: [Cell::new(3.0), Cell::new(4.0)],
        },
    ];
    let tree = TestTree::builder()
        .with_serial_construction()
        .build_from_source(
            &points,
            |point, dim| point.coords[dim].get(),
            |idx, _point| idx as u32,
        )
        .unwrap();

    assert_eq!(tree.size(), points.len());
}

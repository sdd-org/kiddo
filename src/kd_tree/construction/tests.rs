use super::*;
use std::num::NonZeroUsize;

use crate::dist::SquaredEuclidean;
use crate::leaf_strategy::FlatVec;
use crate::leaf_strategy::VecOfArenas;
use crate::leaf_strategy::VecOfArrays;
use crate::Donnelly;
use crate::Eytzinger;
use crate::ItemLeafMode;

use super::item_summary::encode_min_item;
use crate::kd_tree::item_summary::EMPTY_SUBTREE_CODE;

fn assert_item_sorted_leaves<A, SS, LS, const K: usize, const B: usize, const ITEM_LEAF_MODE: u8>(
    tree: &KdTree<A, u32, SS, LS, K, B, ITEM_LEAF_MODE>,
) where
    A: Axis<Coord = A>,
    SS: StemStrategy,
    LS: LeafStrategy<A, u32, SS, K, B>,
{
    assert!(tree.item_sorted_leaves());
    for leaf_idx in 0..tree.leaf_count() {
        let items = (0..tree.leaves.leaf_len(leaf_idx))
            .map(|position| tree.leaves.leaf_point_item(leaf_idx, position).1)
            .collect::<Vec<_>>();
        assert!(
            items.windows(2).all(|pair| pair[0] <= pair[1]),
            "leaf {leaf_idx} was not item-sorted: {items:?}"
        );
    }
}

#[test]
fn embedded_min_item_summary_builder_supports_f64_donnelly_3() {
    type F64Tree = KdTree<f64, u32, Donnelly<3>, FlatVec<f64, u32, 3, 8>, 3, 8>;

    let f64_entries = (0..97u32)
        .map(|item| {
            let scrambled_item = (item * 37) % 97;
            (
                scrambled_item,
                [
                    ((item * 19) % 101) as f64,
                    ((item * 43) % 103) as f64,
                    ((item * 61) % 107) as f64,
                ],
            )
        })
        .collect::<Vec<_>>();

    let f64_tree = F64Tree::builder()
        .with_embedded_min_item_shifted_summary::<1>()
        .with_serial_construction()
        .build_from_entries(&f64_entries)
        .unwrap();
    let sorted_without_summary = F64Tree::builder()
        .with_embedded_min_item_shifted_summary::<1>()
        .with_item_sorted_leaves()
        .with_serial_construction()
        .build_from_entries(&f64_entries)
        .unwrap();

    assert_item_sorted_leaves(&f64_tree);
    assert_eq!(
        f64_tree.item_leaf_mode(),
        ItemLeafMode::SortedWithEncodedMin { shift: 1 }
    );
    assert_eq!(
        sorted_without_summary.item_leaf_mode(),
        ItemLeafMode::SortedWithoutEncodedMin
    );
    assert!(sorted_without_summary.stems[7].is_infinite());
}

fn embedded_summary_codes<LS, const K: usize, const B: usize, const ITEM_LEAF_MODE: u8>(
    tree: &KdTree<f64, u32, Donnelly<3>, LS, K, B, ITEM_LEAF_MODE>,
    block_base: usize,
) -> [u8; 8]
where
    LS: LeafStrategy<f64, u32, Donnelly<3>, K, B>,
{
    let packed = tree.stems[block_base + 7].to_bits();
    array_init::array_init(|child_idx| ((packed >> (child_idx * 8)) & 0xff) as u8)
}

fn leaf_min_item<LS, const K: usize, const B: usize, const ITEM_LEAF_MODE: u8>(
    tree: &KdTree<f64, u32, Donnelly<3>, LS, K, B, ITEM_LEAF_MODE>,
    leaf_idx: usize,
) -> Option<u32>
where
    LS: LeafStrategy<f64, u32, Donnelly<3>, K, B>,
{
    (tree.leaves.leaf_len(leaf_idx) != 0).then(|| tree.leaves.leaf_point_item(leaf_idx, 0).1)
}

fn encoded_optional_min<const SHIFT: u8>(items: impl Iterator<Item = Option<u32>>) -> u8 {
    items
        .flatten()
        .min()
        .map_or(EMPTY_SUBTREE_CODE, encode_min_item::<SHIFT>)
}

#[test]
fn embedded_min_item_summaries_propagate_across_complete_blocks() {
    const SHIFT: u8 = 8;
    type Tree = KdTree<f64, u32, Donnelly<3>, FlatVec<f64, u32, 3, 1>, 3, 1>;

    let entries = (0..64u32)
        .map(|idx| {
            let exponent = 8 + ((idx * 7) % 20);
            let item = (1u32 << exponent) | idx;
            (
                item,
                [
                    ((idx * 19) % 67) as f64,
                    ((idx * 31) % 71) as f64,
                    ((idx * 43) % 73) as f64,
                ],
            )
        })
        .collect::<Vec<_>>();
    let tree = Tree::builder()
        .with_embedded_min_item_shifted_summary::<SHIFT>()
        .with_serial_construction()
        .build_from_entries(&entries)
        .unwrap();
    let sorted_tree = Tree::builder()
        .with_item_sorted_leaves()
        .with_serial_construction()
        .build_from_entries(&entries)
        .unwrap();

    assert!(tree.stems[7].is_finite());
    let root_codes = embedded_summary_codes(&tree, 0);
    for (child_idx, &actual) in root_codes.iter().enumerate() {
        let first_leaf = child_idx * 8;
        let expected = encoded_optional_min::<SHIFT>(
            (first_leaf..first_leaf + 8).map(|leaf_idx| leaf_min_item(&tree, leaf_idx)),
        );
        assert_eq!(actual, expected, "root child {child_idx}");
    }

    for root_child_idx in 0..8 {
        let block_base = (root_child_idx + 1) * 8;
        assert!(tree.stems[block_base + 7].is_finite());
        let child_codes = embedded_summary_codes(&tree, block_base);
        for (child_idx, &actual) in child_codes.iter().enumerate() {
            let leaf_idx = root_child_idx * 8 + child_idx;
            let expected =
                encoded_optional_min::<SHIFT>(std::iter::once(leaf_min_item(&tree, leaf_idx)));
            assert_eq!(actual, expected, "block {root_child_idx} child {child_idx}");
        }
    }

    for query in [[0.0, 0.0, 0.0], [31.0, 29.0, 23.0], [66.0, 70.0, 72.0]] {
        let max_qty = NonZeroUsize::new(7).unwrap();
        let expected = sorted_tree
            .query(&query)
            .best_n_within::<SquaredEuclidean<f64>>(20_000.0, max_qty)
            .execute()
            .into_sorted_vec();
        let actual = tree
            .query(&query)
            .best_n_within::<SquaredEuclidean<f64>>(20_000.0, max_qty)
            .execute()
            .into_sorted_vec();
        assert_eq!(actual, expected);
    }
}

#[test]
fn embedded_min_item_summaries_fill_partial_block_child_ranges() {
    const SHIFT: u8 = 8;
    type Tree = KdTree<f64, u32, Donnelly<3>, FlatVec<f64, u32, 3, 1>, 3, 1>;

    let entries = (0..9u32)
        .map(|idx| {
            (
                (1u32 << (8 + idx)) | idx,
                [
                    idx as f64,
                    ((idx * 7) % 17) as f64,
                    ((idx * 11) % 19) as f64,
                ],
            )
        })
        .collect::<Vec<_>>();
    let tree = Tree::builder()
        .with_serial_construction()
        .with_embedded_min_item_shifted_summary::<SHIFT>()
        .build_from_entries(&entries)
        .unwrap();

    for root_child_idx in 0..8 {
        let block_base = (root_child_idx + 1) * 8;
        assert!(tree.stems[block_base + 7].is_finite());
        let child_codes = embedded_summary_codes(&tree, block_base);
        let left_code = encoded_optional_min::<SHIFT>(std::iter::once(leaf_min_item(
            &tree,
            root_child_idx * 2,
        )));
        let right_code = encoded_optional_min::<SHIFT>(std::iter::once(leaf_min_item(
            &tree,
            root_child_idx * 2 + 1,
        )));
        assert_eq!(
            child_codes,
            [
                left_code, left_code, left_code, left_code, right_code, right_code, right_code,
                right_code,
            ]
        );
    }
    assert!(
        (0..tree.leaf_count()).any(|leaf_idx| leaf_min_item(&tree, leaf_idx).is_none()),
        "fixture must exercise empty leaves"
    );
    assert!(
        (0..8).any(|root_child_idx| {
            embedded_summary_codes(&tree, (root_child_idx + 1) * 8).contains(&EMPTY_SUBTREE_CODE)
        }),
        "empty leaves must use the reserved empty-subtree code"
    );
}

#[cfg(feature = "multi-threaded")]
#[test]
fn embedded_min_item_summaries_match_between_serial_and_parallel_construction() {
    type Tree = KdTree<f64, u32, Donnelly<3>, FlatVec<f64, u32, 3, 8>, 3, 8>;

    let entries = (0..4_096u32)
        .map(|idx| {
            (
                (1u32 << (8 + (idx % 20))) | idx,
                [
                    ((idx * 19) % 4_099) as f64,
                    ((idx * 31) % 4_111) as f64,
                    ((idx * 43) % 4_129) as f64,
                ],
            )
        })
        .collect::<Vec<_>>();
    let serial = Tree::builder()
        .with_embedded_min_item_shifted_summary::<8>()
        .with_serial_construction()
        .build_from_entries(&entries)
        .unwrap();
    let parallel = Tree::builder()
        .with_embedded_min_item_shifted_summary::<8>()
        .with_parallel_construction()
        .build_from_entries(&entries)
        .unwrap();

    assert_eq!(
        serial
            .stems
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        parallel
            .stems
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        serial.iter().collect::<Vec<_>>(),
        parallel.iter().collect::<Vec<_>>()
    );
}

#[test]
fn item_sorted_leaf_builder_sorts_immutable_leaf_layouts() {
    type FlatTree = KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, 2, 8>, 2, 8>;
    type ArenaTree = KdTree<f32, u32, Eytzinger, VecOfArenas<f32, u32, 2, 8>, 2, 8>;

    let entries = (0..97u32)
        .map(|item| {
            let scrambled_item = (item * 37) % 97;
            (
                scrambled_item,
                [((item * 19) % 101) as f32, ((item * 43) % 103) as f32],
            )
        })
        .collect::<Vec<_>>();

    let ordinary = FlatTree::builder()
        .with_serial_construction()
        .build_from_entries(&entries)
        .unwrap();
    assert!(!ordinary.item_sorted_leaves());
    assert_eq!(ordinary.item_leaf_mode(), ItemLeafMode::Unsorted);

    let flat = FlatTree::builder()
        .with_item_sorted_leaves()
        .with_serial_construction()
        .build_from_entries(&entries)
        .unwrap();
    let arena = ArenaTree::builder()
        .with_serial_construction()
        .with_item_sorted_leaves()
        .build_from_entries(&entries)
        .unwrap();
    assert_item_sorted_leaves(&flat);
    assert_item_sorted_leaves(&arena);
    assert_eq!(flat.item_leaf_mode(), ItemLeafMode::SortedWithoutEncodedMin);
    assert_eq!(
        arena.item_leaf_mode(),
        ItemLeafMode::SortedWithoutEncodedMin
    );

    let mut expected = entries.clone();
    expected.sort_unstable_by_key(|entry| entry.0);
    for mut actual in [
        flat.iter().collect::<Vec<_>>(),
        arena.iter().collect::<Vec<_>>(),
    ] {
        actual.sort_unstable_by_key(|entry| entry.0);
        assert_eq!(actual, expected);
    }
}

#[test]
fn empty_mutable_bulk_construction_remains_addable() {
    type Tree = KdTree<f32, u32, Eytzinger, VecOfArrays<f32, u32, 2, 8>, 2, 8>;
    let entries: [(u32, [f32; 2]); 0] = [];

    let mut tree = Tree::builder().build_from_entries(&entries).unwrap();
    tree.add(&[1.0, 2.0], 7).unwrap();

    assert_eq!(tree.size(), 1);
    assert_eq!(tree.iter().collect::<Vec<_>>(), vec![(7, [1.0, 2.0])]);
}

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

#[cfg(feature = "multi-threaded")]
#[test]
fn mutable_split_preserves_point_item_associations_when_pivot_scan_retries() {
    type TestTree = KdTree<f32, u32, Eytzinger, VecOfArrays<f32, u32, 2, 16>, 2, 16>;

    const EXTREMES: [[f32; 2]; 6] = [
        [-1.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [0.0, -0.0],
        [-0.0, 0.0],
    ];

    let mut tree = TestTree::default();
    let mut expected = Vec::new();
    for item in 0..33u32 {
        let point = EXTREMES[item as usize % EXTREMES.len()];
        tree.add(&point, item).unwrap();
        expected.push((item, point.map(f32::to_bits)));
    }

    let final_point = [1.0, 4.0 / 9.0];
    tree.add(&final_point, 33).unwrap();
    expected.push((33, final_point.map(f32::to_bits)));

    let mut got = tree
        .iter()
        .map(|(item, point)| (item, point.map(f32::to_bits)))
        .collect::<Vec<_>>();
    got.sort_unstable();

    assert_eq!(got, expected);
}

#[test]
fn rejected_mutable_split_preserves_existing_point_item_associations() {
    type TestTree = KdTree<f32, u32, Eytzinger, VecOfArrays<f32, u32, 2, 4>, 2, 4>;

    let mut tree = TestTree::default();
    let mut expected = Vec::new();
    for item in 0..4u32 {
        let point = [2.0, 200.0 + item as f32];
        tree.add(&point, item).unwrap();
        expected.push((item, point));
    }

    assert_eq!(
        tree.add(&[2.0, 204.0], 4),
        Err(ConstructionError::UnsplittableBucket { split_dim: 0 })
    );
    assert_eq!(tree.size(), expected.len());

    let mut got = tree.iter().collect::<Vec<_>>();
    got.sort_unstable_by_key(|(item, _)| *item);
    assert_eq!(got, expected);
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

#[cfg(feature = "multi-threaded")]
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
            let sequential_item_sorted = <$tree>::builder()
                .with_item_sorted_leaves()
                .with_serial_construction()
                .build_from_slice(&points)
                .unwrap();
            let parallel_item_sorted = <$tree>::builder()
                .with_parallel_construction()
                .with_item_sorted_leaves()
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
            assert!(sequential_item_sorted.item_sorted_leaves());
            assert!(parallel_item_sorted.item_sorted_leaves());
            assert_eq!(
                sequential_item_sorted.iter().collect::<Vec<_>>(),
                parallel_item_sorted.iter().collect::<Vec<_>>()
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

#[cfg(feature = "multi-threaded")]
#[test]
fn parallel_soft_construction_handles_scalar_donnelly_partial_block() {
    type TestTree = KdTree<f64, u32, Donnelly<3>, FlatVec<f64, u32, 2, 4>, 2, 4>;

    // Sixteen leaves require four construction levels. This deliberately ends
    // one level into Donnelly<3>'s second block.
    let points = (0..64)
        .map(|idx| [((idx * 17) % 67) as f64, ((idx * 29 + idx / 5) % 71) as f64])
        .collect::<Vec<_>>();

    let sequential = TestTree::builder()
        .with_serial_construction()
        .build_from_slice(&points)
        .unwrap();
    let parallel = TestTree::builder()
        .with_parallel_construction()
        .build_from_slice(&points)
        .unwrap();

    assert_eq!(sequential.stems.as_slice(), parallel.stems.as_slice());
    assert_eq!(sequential.leaf_count(), 16);
    assert_eq!(sequential.leaf_count(), parallel.leaf_count());
    assert_eq!(sequential.max_stem_level(), 3);
    assert_eq!(sequential.max_stem_level(), parallel.max_stem_level());
    assert_eq!(sequential.max_leaf_len(), parallel.max_leaf_len());
    assert_eq!(
        sequential.iter().collect::<Vec<_>>(),
        parallel.iter().collect::<Vec<_>>()
    );

    for query in [[0.0, 0.0], [31.0, 37.0], [66.0, 70.0]] {
        assert_eq!(
            sequential
                .query(&query)
                .nearest_one::<SquaredEuclidean<f64>>()
                .execute(),
            parallel
                .query(&query)
                .nearest_one::<SquaredEuclidean<f64>>()
                .execute()
        );
    }
}

#[cfg(feature = "multi-threaded")]
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

#[cfg(feature = "multi-threaded")]
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

use crate::dist::DistanceMetric;
use crate::kd_tree::query_context::QueryContext;
use crate::kd_tree::KdTreeQueryOps;
use crate::leaf_view_chunked::nearest_one::nearest_one_with_query_wide;
use crate::traits::leaf_strategy::LeafProjection;
use crate::{Axis, Content, KdTree, LeafStrategy, StemStrategy};

impl<A, T, SS, LS, const K: usize, const B: usize, const ITEM_LEAF_MODE: u8>
    KdTree<A, T, SS, LS, K, B, ITEM_LEAF_MODE>
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd + PartialEq,
    LS: LeafStrategy<A, T, SS, K, B>,
    SS: StemStrategy,
{
    #[inline(always)]
    fn process_leaf_approx_nearest_one<D>(
        &self,
        leaf_idx: usize,
        query_wide: &[D::Output; K],
        best_dist: &mut D::Output,
        best_item: &mut T,
    ) where
        D: DistanceMetric<A, Output = A>,
    {
        match LS::LEAF_PROJECTION {
            LeafProjection::LeafArena => {
                let arena = self.leaves.leaf_arena(leaf_idx);
                crate::leaf_view_chunked::nearest_one::nearest_one_with_query_wide_arena::<
                    A,
                    T,
                    D,
                    K,
                >(&arena, query_wide, best_dist, best_item);
            }
            LeafProjection::LeafView => {
                let leaf = self.leaves.leaf_view(leaf_idx);
                nearest_one_with_query_wide::<A, T, D, K, B>(
                    &leaf, query_wide, best_dist, best_item,
                );
            }
        }
    }

    /// Finds an approximate nearest point to the query point.
    ///
    /// This is faster than `nearest_one` but may not return the true nearest neighbour.
    /// It searches only the leaf that the query point falls into.
    #[inline(always)]
    pub(crate) fn approx_nearest_one<D>(&self, query: &[A; K]) -> (D::Output, T)
    where
        D: DistanceMetric<A, Output = A>,
    {
        let req_ctx = ApproxNearestOneReqCtx::<A, D::Output, K> {
            query,
            _phantom: std::marker::PhantomData,
        };

        let mut best_dist = A::max_value();
        let mut best_item = T::default();

        self.straight_query(req_ctx, |leaf_idx| {
            self.process_leaf_approx_nearest_one::<D>(
                leaf_idx,
                query,
                &mut best_dist,
                &mut best_item,
            );
        });

        (best_dist, best_item)
    }
}

struct ApproxNearestOneReqCtx<'a, A, O, const K: usize> {
    query: &'a [A; K],
    _phantom: std::marker::PhantomData<O>,
}

impl<A, O, const K: usize> QueryContext<A, O, K> for ApproxNearestOneReqCtx<'_, A, O, K> {
    fn query(&self) -> &[A; K] {
        self.query
    }

    fn max_dist(&self) -> O {
        panic!("approx_nearest_one should not be called with max_dist")
    }
}

#[cfg(feature = "cargo_asm")]
pub mod cargo_asm {
    use crate::dist::SquaredEuclidean;
    use crate::kd_tree::KdTree;
    use crate::leaf_strategy::{FlatVec, VecOfArenas, VecOfArrays};
    use crate::Eytzinger;

    const K: usize = 3;
    const BUCKET_SIZE: usize = 64;

    type FlatVecKdT =
        KdTree<f64, usize, Eytzinger, FlatVec<f64, usize, K, BUCKET_SIZE>, K, BUCKET_SIZE>;
    type VecOfArraysKdT =
        KdTree<f64, usize, Eytzinger, VecOfArrays<f64, usize, K, BUCKET_SIZE>, K, BUCKET_SIZE>;
    type VecOfArenasKdT =
        KdTree<f64, usize, Eytzinger, VecOfArenas<f64, usize, K, BUCKET_SIZE>, K, BUCKET_SIZE>;

    /// Hook for cargo-asm to render the v6 approx-nearest-one call path.
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_approx_nearest_one_eytzinger_cargo_asm_hook(
        tree: &FlatVecKdT,
        query: [f64; 3],
    ) -> (f64, usize) {
        let result = tree
            .query(&query)
            .nearest_one::<SquaredEuclidean<f64>>()
            .approx()
            .execute();
        (result.distance, result.item)
    }

    /// Hook for cargo-asm to render the v6 approx-nearest-one call path with VecOfArrays leaves.
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_approx_nearest_one_eytzinger_vec_of_arrays_cargo_asm_hook(
        tree: &VecOfArraysKdT,
        query: [f64; 3],
    ) -> (f64, usize) {
        let result = tree
            .query(&query)
            .nearest_one::<SquaredEuclidean<f64>>()
            .approx()
            .execute();
        (result.distance, result.item)
    }

    /// Hook for cargo-asm to render the v6 approx-nearest-one call path with VecOfArenas leaves.
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_approx_nearest_one_eytzinger_vec_of_arenas_cargo_asm_hook(
        tree: &VecOfArenasKdT,
        query: [f64; 3],
    ) -> (f64, usize) {
        let result = tree
            .query(&query)
            .nearest_one::<SquaredEuclidean<f64>>()
            .approx()
            .execute();
        (result.distance, result.item)
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    use crate::kd_tree::KdTree;
    use crate::leaf_strategy::{FlatVec, VecOfArenas, VecOfArrays};
    use crate::stem_strategy::{Donnelly, DonnellyUnrolled};

    use crate::dist::SquaredEuclidean;
    use crate::Eytzinger;

    const RNG_SEED: u64 = 42;

    #[test]
    fn v6_approx_nearest_one_flat_vec_f32() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        let mut points: Vec<[f32; 3]> = vec![];
        for _ in 0..65_536 {
            let x = rng.random_range(0.0..1.0);
            let y = rng.random_range(0.0..1.0);
            let z = rng.random_range(0.0..1.0);
            points.push([x, y, z]);
        }

        let tree: KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points).unwrap();

        assert!(!tree.is_empty());
        assert_eq!(tree.size(), 65_536);
        assert_eq!(tree.leaf_count(), 2048);
        assert_eq!(tree.max_stem_level(), 10);

        let query_point = [0.5, 0.5, 0.5];

        let results = tree
            .query(&query_point)
            .nearest_one::<SquaredEuclidean<f32>>()
            .approx()
            .execute();

        assert_eq!(results.distance, 0.0014114721);
        assert_eq!(results.item, 19074);
    }

    #[test]
    fn v6_approx_nearest_one_flat_vec_f32_no_items() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        let mut points: Vec<[f32; 3]> = vec![];
        for _ in 0..65_536 {
            let x = rng.random_range(0.0..1.0);
            let y = rng.random_range(0.0..1.0);
            let z = rng.random_range(0.0..1.0);
            points.push([x, y, z]);
        }

        let tree: KdTree<f32, (), Eytzinger, FlatVec<f32, (), 3, 32>, 3, 32> =
            KdTree::new_from_slice_no_items(&points).unwrap();

        assert!(!tree.is_empty());
        assert_eq!(tree.size(), 65_536);
        assert_eq!(tree.leaf_count(), 2048);
        assert_eq!(tree.max_stem_level(), 10);

        let query_point = [0.5, 0.5, 0.5];

        let results = tree
            .query(&query_point)
            .nearest_one::<SquaredEuclidean<f32>>()
            .approx()
            .execute();

        assert_eq!(results.distance, 0.0014114721);
        assert_eq!(results.item, ());
    }

    #[test]
    fn v6_approx_nearest_one_vec_of_arrays_f32() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        let mut points: Vec<[f32; 3]> = vec![];
        for _ in 0..65_536 {
            let x = rng.random_range(0.0..1.0);
            let y = rng.random_range(0.0..1.0);
            let z = rng.random_range(0.0..1.0);
            points.push([x, y, z]);
        }

        let tree: KdTree<f32, u32, Eytzinger, VecOfArrays<f32, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points).unwrap();

        assert!(!tree.is_empty());
        assert_eq!(tree.size(), 65_536);
        assert_eq!(tree.leaf_count(), 2048);
        assert_eq!(tree.max_stem_level(), 10);

        let query_point = [0.5, 0.5, 0.5];

        let results = tree
            .query(&query_point)
            .nearest_one::<SquaredEuclidean<f32>>()
            .approx()
            .execute();

        assert_eq!(results.distance, 0.0014114721);
        assert_eq!(results.item, 19074);
    }

    #[test]
    fn v6_approx_nearest_one_vec_of_arenas_matches_flat_vec_f32() {
        let points = vec![
            [0.1f32, 0.2, 0.3],
            [0.9, 0.8, 0.7],
            [0.41, 0.52, 0.63],
            [0.4, 0.5, 0.6],
            [0.7, 0.1, 0.2],
        ];
        let query = [0.39f32, 0.51, 0.61];

        let flat_tree: KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points).unwrap();
        let arena_tree: KdTree<f32, u32, Eytzinger, VecOfArenas<f32, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points).unwrap();

        let flat_result = flat_tree
            .query(&query)
            .nearest_one::<SquaredEuclidean<f32>>()
            .approx()
            .execute();
        let arena_result = arena_tree
            .query(&query)
            .nearest_one::<SquaredEuclidean<f32>>()
            .approx()
            .execute();

        assert_eq!(arena_result, flat_result);
    }

    #[test]
    fn v6_approx_nearest_one_vec_of_arrays_mutated_one_split_f32() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        // Add 33 points to trigger exactly one split (bucket size is 32)
        let mut points: Vec<[f32; 3]> = vec![];
        for _ in 0..33 {
            let x = rng.random_range(0.0..1.0);
            let y = rng.random_range(0.0..1.0);
            let z = rng.random_range(0.0..1.0);
            points.push([x, y, z]);
        }

        let mut tree: KdTree<f32, u32, Eytzinger, VecOfArrays<f32, u32, 3, 32>, 3, 32> =
            KdTree::default();

        for (idx, point) in points.iter().enumerate() {
            tree.add(point, idx as u32).unwrap();
        }

        // Print tree state for debugging
        println!("\nTree state after adding 33 points:");
        println!("{}", tree);

        assert!(!tree.is_empty());
        assert_eq!(tree.size(), 33);
        assert_eq!(tree.leaf_count(), 2); // Should have split into 2 leaves

        // Check that the root stem value was updated from max_value to a reasonable pivot
        let root_stem_value = tree.stems[1]; // Eytzinger root is at index 1
        assert!(
            root_stem_value < f32::INFINITY,
            "Root stem should be updated from max_value"
        );
        assert!(
            (0.0..=1.0).contains(&root_stem_value),
            "Root stem should be a reasonable value in [0, 1]"
        );

        // Just verify the tree queries successfully - we'll verify correctness later
        let query_point = [0.5, 0.5, 0.5];
        let _results = tree
            .query(&query_point)
            .nearest_one::<SquaredEuclidean<f32>>()
            .approx()
            .execute();
    }

    #[test]
    fn v6_approx_nearest_one_vec_of_arrays_mutated_two_splits_f32() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        // Add 65 points to trigger exactly two splits (bucket size is 32)
        // First split at 33 points: 1 leaf -> 2 leaves
        // Second split at 65 points: 2 leaves -> 3 leaves
        let mut points: Vec<[f32; 3]> = vec![];
        for _ in 0..65 {
            let x = rng.random_range(0.0..1.0);
            let y = rng.random_range(0.0..1.0);
            let z = rng.random_range(0.0..1.0);
            points.push([x, y, z]);
        }

        let mut tree: KdTree<f32, u32, Eytzinger, VecOfArrays<f32, u32, 3, 32>, 3, 32> =
            KdTree::default();

        for (idx, point) in points.iter().enumerate() {
            // println!("Tree at item {}: {}", idx, tree);
            tree.add(point, idx as u32).unwrap();
        }

        // Print tree state for debugging
        // println!("\nTree state after adding 65 points:");
        println!("{}", tree);

        assert!(!tree.is_empty());
        assert_eq!(tree.size(), 65);
        assert_eq!(tree.leaf_count(), 3); // Should have split into 3 leaves

        // Check that the root stem value was updated from max_value to a reasonable pivot
        let root_stem_value = tree.stems[1]; // Eytzinger root is at index 1
        assert!(
            root_stem_value < f32::INFINITY,
            "Root stem should be updated from max_value"
        );
        assert!(
            (0.0..=1.0).contains(&root_stem_value),
            "Root stem should be a reasonable value in [0, 1]"
        );

        // Check that at least one child stem was also initialized
        let left_child_stem = tree.stems[2]; // Eytzinger left child at index 2
        let right_child_stem = tree.stems[3]; // Eytzinger right child at index 3

        // At least one child should have been split (not max_value any more)
        let has_initialized_child =
            (left_child_stem < f32::INFINITY) || (right_child_stem < f32::INFINITY);
        assert!(
            has_initialized_child,
            "At least one child stem should be initialized after second split"
        );

        // Just verify the tree queries successfully - we'll verify correctness later
        let query_point = [0.5, 0.5, 0.5];
        let _results = tree
            .query(&query_point)
            .nearest_one::<SquaredEuclidean<f32>>()
            .approx()
            .execute();
    }

    #[test]
    fn v6_approx_nearest_one_vec_of_arrays_mutated_f32() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        let mut points: Vec<[f32; 3]> = vec![];
        for _ in 0..65_536 {
            let x = rng.random_range(0.0..1.0);
            let y = rng.random_range(0.0..1.0);
            let z = rng.random_range(0.0..1.0);
            points.push([x, y, z]);
        }

        let mut tree: KdTree<f32, u32, Eytzinger, VecOfArrays<f32, u32, 3, 32>, 3, 32> =
            KdTree::default();

        for (idx, point) in points.iter().enumerate() {
            // println!("Tree at item {}: {}", idx, tree);
            tree.add(point, idx as u32).unwrap();
        }

        assert!(!tree.is_empty());
        assert_eq!(tree.size(), 65_536);
        // assert_eq!(tree.leaf_count(), 2048);
        // assert_eq!(tree.max_stem_level(), 10);

        let query_point = [0.5, 0.5, 0.5];

        let results = tree
            .query(&query_point)
            .nearest_one::<SquaredEuclidean<f32>>()
            .approx()
            .execute();

        assert_eq!(results.distance, 0.0003201659);
        assert_eq!(results.item, 21996);
    }

    #[test]
    fn v6_approx_nearest_one_flat_vec_f32_donnelly_marker() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        let mut points: Vec<[f32; 4]> = vec![];
        for _ in 0..131_072 {
            let x = rng.random_range(0.0..1.0);
            let y = rng.random_range(0.0..1.0);
            let z = rng.random_range(0.0..1.0);
            let w = rng.random_range(0.0..1.0);
            points.push([x, y, z, w]);
        }

        // Use DonnellyUnrolled with Block4 (4 levels per block, matching 16 f32s per 64-byte cache line)
        let tree: KdTree<f32, u32, DonnellyUnrolled<4>, FlatVec<f32, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&points).unwrap();

        assert!(!tree.is_empty());
        assert_eq!(tree.size(), 131_072);

        let query_point = [0.5, 0.5, 0.5, 0.5];

        let results = tree
            .query(&query_point)
            .nearest_one::<SquaredEuclidean<f32>>()
            .approx()
            .execute();

        assert!(
            results.distance < 0.1,
            "Distance should be reasonably small"
        );
        assert!(results.item < 131_072, "Item index should be valid");
    }

    #[test]
    fn v6_approx_nearest_one_donnelly_marker_matches_eytzinger() {
        // Verify that DonnellyUnrolled produces the same results as Eytzinger
        // for the same input data
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        let points: Vec<[f32; 4]> = (0..8_192)
            .map(|_| {
                [
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                ]
            })
            .collect();

        let tree_eytz: KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&points).unwrap();

        let tree_donnelly: KdTree<f32, u32, DonnellyUnrolled<4>, FlatVec<f32, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&points).unwrap();

        // Test multiple query points
        let query_points: Vec<[f32; 4]> = (0..100)
            .map(|_| {
                [
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                ]
            })
            .collect();

        for (idx, query_point) in query_points.iter().enumerate() {
            let result_eytz = tree_eytz
                .query(query_point)
                .nearest_one::<SquaredEuclidean<f32>>()
                .approx()
                .execute();
            let result_donnelly = tree_donnelly
                .query(query_point)
                .nearest_one::<SquaredEuclidean<f32>>()
                .approx()
                .execute();

            // The results should be identical since both use the same construction
            // and the approximate query just returns the best match in the target leaf
            assert_eq!(
                result_eytz, result_donnelly,
                "DonnellyUnrolled should produce same results as Eytzinger for query #{} ({:?})",
                idx, query_point
            );
        }
    }

    #[test]
    fn v6_approx_nearest_one_donnelly_matches_eytzinger() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        let points: Vec<[f32; 4]> = (0..8_192)
            .map(|_| {
                [
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                ]
            })
            .collect();

        let tree_eytz: KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&points).unwrap();

        let tree_donnelly: KdTree<f32, u32, Donnelly<4>, FlatVec<f32, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&points).unwrap();

        // Test multiple query points
        let query_points: Vec<[f32; 4]> = (0..100)
            .map(|_| {
                [
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                ]
            })
            .collect();

        for (idx, query_point) in query_points.iter().enumerate() {
            let result_eytz = tree_eytz
                .query(query_point)
                .nearest_one::<SquaredEuclidean<f32>>()
                .approx()
                .execute();
            let result_donnelly = tree_donnelly
                .query(query_point)
                .nearest_one::<SquaredEuclidean<f32>>()
                .approx()
                .execute();

            assert_eq!(
                result_eytz, result_donnelly,
                "Donnelly should produce same results as Eytzinger for query #{} ({:?})",
                idx, query_point
            );
        }
    }

    #[test]
    fn v6_approx_nearest_one_donnelly_marker_matches_donnelly() {
        // Verify that DonnellyUnrolled produces the same results as Donnelly
        // for the same input data
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        let points: Vec<[f32; 4]> = (0..8_192)
            .map(|_| {
                [
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                ]
            })
            .collect();

        let tree_donnelly: KdTree<f32, u32, Donnelly<4>, FlatVec<f32, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&points).unwrap();

        let tree_donnelly_marker: KdTree<
            f32,
            u32,
            DonnellyUnrolled<4>,
            FlatVec<f32, u32, 4, 32>,
            4,
            32,
        > = KdTree::new_from_slice(&points).unwrap();

        // Test multiple query points
        let query_points: Vec<[f32; 4]> = (0..100)
            .map(|_| {
                [
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                ]
            })
            .collect();

        for (idx, query_point) in query_points.iter().enumerate() {
            let result_eytz = tree_donnelly
                .query(query_point)
                .nearest_one::<SquaredEuclidean<f32>>()
                .approx()
                .execute();
            let result_donnelly = tree_donnelly_marker
                .query(query_point)
                .nearest_one::<SquaredEuclidean<f32>>()
                .approx()
                .execute();

            // The results should be identical since both use the same construction
            // and the approximate query just returns the best match in the target leaf
            assert_eq!(
                result_eytz, result_donnelly,
                "DonnellyUnrolled should produce same results as Donnelly for query #{} ({:?})",
                idx, query_point
            );
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    #[cfg(target_arch = "x86_64")]
    fn v6_approx_nearest_one_donnelly_marker_simd_f64() {
        use crate::stem_strategy::DonnellySimdFull;

        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        let points: Vec<[f64; 3]> = (0..2_097_152)
            .map(|_| {
                [
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                ]
            })
            .collect();

        let tree: KdTree<f64, u32, DonnellySimdFull<3>, FlatVec<f64, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points).unwrap();

        assert!(!tree.is_empty());
        assert_eq!(tree.size(), 2_097_152);
        assert_eq!(tree.leaf_count(), 65_536);

        assert_eq!((tree.max_stem_level() + 1) % 3, 0);
        assert_eq!(tree.max_stem_level(), 17);

        // Test multiple query points to ensure queries work correctly
        let query_points: Vec<[f64; 3]> = (0..100)
            .map(|_| {
                [
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                ]
            })
            .collect();

        for query_point in query_points.iter() {
            let result = tree
                .query(query_point)
                .nearest_one::<SquaredEuclidean<f64>>()
                .approx()
                .execute();

            // Verify result is valid
            assert!(result.distance >= 0.0, "Distance should be non-negative");
            assert!(result.item < 2_097_152, "Item index should be valid");

            // Verify the returned item is actually close to the query
            // (approximate query returns best in target leaf, so should be reasonably close)
            assert!(
                result.distance < 1.0,
                "Distance should be reasonable for unit cube"
            );
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    #[cfg(target_arch = "x86_64")]
    fn v6_approx_nearest_one_donnelly_marker_simd_f32() {
        use crate::stem_strategy::DonnellySimdFull;

        // Test DonnellySimdFull with f32 data
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        let points: Vec<[f32; 4]> = (0..2_097_152)
            .map(|_| {
                [
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                ]
            })
            .collect();

        // 2_097_152 points with bucket size 32 = 65_536 leaves = 2^16, depth = 16
        // Block4 divides evenly into 16
        let tree: KdTree<f32, u32, DonnellySimdFull<4>, FlatVec<f32, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&points).unwrap();

        assert!(!tree.is_empty());
        assert_eq!(tree.size(), 2_097_152);
        assert_eq!(tree.leaf_count(), 65_536);
        // Verify max_stem_level is padded to multiple of block size
        assert_eq!((tree.max_stem_level() + 1) % 4, 0);

        let query_point = [0.5, 0.5, 0.5, 0.5];
        let results = tree
            .query(&query_point)
            .nearest_one::<SquaredEuclidean<f32>>()
            .approx()
            .execute();

        assert!(
            results.distance < 0.2,
            "Distance ({}) should be reasonably small",
            results.distance
        );
        assert!(results.item < 2_097_152, "Item index should be valid");
    }
}

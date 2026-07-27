use std::num::NonZeroUsize;

use crate::dist::DistanceMetric;
use crate::kd_tree::query_stack::StackTrait;
use crate::leaf_view::TlsLeafScratch;
use crate::stem_strategy::donnelly::simd_full::{
    BacktrackBlock3, BacktrackBlock4, SimdSelectBestChildBlock3,
};
use crate::{Axis, Content, KdTree, LeafStrategy, QueryResultItem, StemStrategy};

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    LS: LeafStrategy<A, T, SS, K, B>,
    SS: StemStrategy,
{
    pub(crate) fn within_impl<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
        result_capacity: Option<NonZeroUsize>,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    {
        self.nearest_n_within_inner::<D, Vec<QueryResultItem<(), T, D::Output>>, EXCLUSIVE>(
            query,
            max_dist,
            usize::MAX,
            true,
            result_capacity,
        )
    }

    pub(crate) fn within_impl_with_scratch<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
        result_capacity: Option<NonZeroUsize>,
        stack: &mut SS::Stack<D::Output>,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        D: DistanceMetric<A>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS>,
    {
        self.nearest_n_within_inner_with_scratch::<
            D,
            Vec<QueryResultItem<(), T, D::Output>>,
            EXCLUSIVE,
        >(
            query,
            max_dist,
            usize::MAX,
            true,
            result_capacity,
            stack,
        )
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    use std::cmp::Ordering;

    use crate::dist::Manhattan;
    use crate::dist::SquaredEuclidean;
    use crate::kd_tree::{KdTree, KdTreeQueryOps};
    use crate::leaf_strategy::{FlatVec, VecOfArenas, VecOfArrays};
    use crate::Axis;
    use crate::Donnelly;
    use crate::Eytzinger;

    const RNG_SEED: u64 = 42;
    const TILE_BOUNDARY_CASES: [usize; 7] = [1, 2, 4, 8, 32, 33, 47];

    #[test]
    fn result_capacity_hint_is_preserved_for_materialized_results() {
        let points: Vec<_> = (0..128).map(|idx| [idx as f64 / 128.0, 0.0]).collect();
        let tree: KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, 2, 32>, 2, 32> =
            KdTree::new_from_slice(&points).unwrap();
        let query = [0.0, 0.0];

        for results in [
            tree.query(&query)
                .within::<SquaredEuclidean<f64>>(1.0)
                .with_result_capacity(256)
                .execute(),
            tree.query(&query)
                .within::<SquaredEuclidean<f64>>(1.0)
                .unsorted()
                .with_result_capacity(256)
                .execute(),
        ] {
            assert_eq!(results.len(), points.len());
            assert!(results.capacity() >= 256);
        }
    }

    #[test]
    fn within_exclusive_boundaries_excludes_exact_threshold_matches() {
        let points = vec![[0.0f64, 0.0], [1.0, 0.0], [2.0, 0.0], [0.5, 0.0]];
        let tree: KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, 2, 32>, 2, 32> =
            KdTree::new_from_slice(&points).unwrap();
        let query = [0.0, 0.0];

        let inclusive: Vec<_> = tree
            .query(&query)
            .within::<SquaredEuclidean<f64>>(1.0)
            .execute()
            .into_iter()
            .map(|n| n.item)
            .collect();
        let exclusive: Vec<_> = tree
            .query(&query)
            .within::<SquaredEuclidean<f64>>(1.0)
            .exclusive_boundaries()
            .execute()
            .into_iter()
            .map(|n| n.item)
            .collect();

        assert_eq!(inclusive, vec![0, 3, 1]);
        assert_eq!(exclusive, vec![0, 3]);
    }

    #[test]
    fn zero_radius_inclusive_finds_point_after_monotonic_eytzinger_growth() {
        type TestTree = KdTree<f32, u32, Eytzinger, VecOfArrays<f32, u32, 2, 16>, 2, 16>;

        let mut tree = TestTree::default();
        for item in 0..1_023u32 {
            let t = -1.0 + 2.0 * item as f32 / 1_022.0;
            tree.add(&[t, t], item).unwrap();
        }

        let stored_leaf = tree.find_leaf_for_item(511).unwrap().0;
        let routed_leaf = tree.get_leaf_idx(&[0.0, 0.0]);
        assert_eq!(routed_leaf, stored_leaf);

        let results = tree
            .query(&[0.0, 0.0])
            .within::<SquaredEuclidean<f32>>(0.0)
            .execute();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].item, 511);
        assert_eq!(results[0].distance, 0.0);
    }

    #[test]
    fn zero_radius_inclusive_finds_point_after_monotonic_donnelly_growth() {
        type TestTree = KdTree<f32, u32, Donnelly<4>, VecOfArrays<f32, u32, 2, 16>, 2, 16>;

        let mut tree = TestTree::default();
        for item in 0..1_023u32 {
            let t = -1.0 + 2.0 * item as f32 / 1_022.0;
            tree.add(&[t, t], item).unwrap();
        }

        let stored_leaf = tree.find_leaf_for_item(511).unwrap().0;
        let routed_leaf = tree.get_leaf_idx(&[0.0, 0.0]);
        assert_eq!(routed_leaf, stored_leaf);

        let results = tree
            .query(&[0.0, 0.0])
            .within::<SquaredEuclidean<f32>>(0.0)
            .execute();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].item, 511);
        assert_eq!(results[0].distance, 0.0);
    }

    #[test]
    fn within_vec_of_arenas_matches_flat_vec_across_tile_boundaries() {
        let query = [0.31f32, 0.47, 0.59];
        let radius = 0.35;

        for &len in &TILE_BOUNDARY_CASES {
            let points: Vec<[f32; 3]> = (0..len)
                .map(|idx| {
                    [
                        ((idx * 5) % 97) as f32 / 97.0,
                        ((idx * 13 + 1) % 97) as f32 / 97.0,
                        ((idx * 23 + 2) % 97) as f32 / 97.0,
                    ]
                })
                .collect();

            let flat_tree: KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, 3, 32>, 3, 32> =
                KdTree::new_from_slice(&points).unwrap();
            let arena_tree: KdTree<f32, u32, Eytzinger, VecOfArenas<f32, u32, 3, 32>, 3, 32> =
                KdTree::new_from_slice(&points).unwrap();

            let mut flat: Vec<(f32, u32)> = flat_tree
                .query(&query)
                .within::<Manhattan<f32>>(radius)
                .execute()
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();
            let mut arena: Vec<(f32, u32)> = arena_tree
                .query(&query)
                .within::<Manhattan<f32>>(radius)
                .execute()
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            stabilize_sort(&mut flat);
            stabilize_sort(&mut arena);

            assert_eq!(arena, flat, "len={len}");
        }
    }

    #[test]
    fn can_query_items_within_radius_large_scale() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        const RADIUS: f32 = 0.2;

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

        let tree: KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&content_to_add).unwrap();

        let query_points: Vec<[f32; 4]> =
            (0..NUM_QUERIES).map(|_| rng.random::<[f32; 4]>()).collect();

        for query_point in query_points {
            let expected = linear_search(&content_to_add, &query_point, RADIUS);

            let mut result: Vec<_> = tree
                .query(&query_point)
                .within::<Manhattan<f32>>(RADIUS)
                .execute()
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            stabilize_sort(&mut result);

            assert_eq!(result, expected);
        }
    }

    #[test]
    fn v6_query_within_large_vec_of_arrays_f32() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        const RADIUS: f32 = 0.2;

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

        let tree: KdTree<f32, u32, Eytzinger, VecOfArrays<f32, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&content_to_add).unwrap();

        let query_points: Vec<[f32; 4]> =
            (0..NUM_QUERIES).map(|_| rng.random::<[f32; 4]>()).collect();

        for query_point in query_points {
            let expected = linear_search(&content_to_add, &query_point, RADIUS);

            let mut result: Vec<_> = tree
                .query(&query_point)
                .within::<Manhattan<f32>>(RADIUS)
                .execute()
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            stabilize_sort(&mut result);

            assert_eq!(result, expected);
        }
    }

    #[test]
    fn v6_query_within_vec_of_arrays_f32_no_items() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        const TREE_SIZE: usize = 1_000;
        const NUM_QUERIES: usize = 1;
        const RADIUS: f32 = 0.2;

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

        let tree: KdTree<f32, (), Eytzinger, VecOfArrays<f32, (), 4, 32>, 4, 32> =
            KdTree::new_from_slice_no_items(&content_to_add).unwrap();

        let query_points: Vec<[f32; 4]> =
            (0..NUM_QUERIES).map(|_| rng.random::<[f32; 4]>()).collect();

        for query_point in query_points {
            let expected: Vec<_> = linear_search(&content_to_add, &query_point, RADIUS)
                .into_iter()
                .map(|(distance, _)| (distance, ()))
                .collect();

            let mut result: Vec<_> = tree
                .query(&query_point)
                .within::<Manhattan<f32>>(RADIUS)
                .execute()
                .into_iter()
                .map(|n| (n.distance, 1))
                .collect();

            stabilize_sort(&mut result);

            let result: Vec<_> = result
                .into_iter()
                .map(|(distance, _)| (distance, ()))
                .collect();

            assert_eq!(result, expected);
        }
    }

    #[test]
    fn v6_query_within_large_vec_of_arrays_mutated_f32() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        const RADIUS: f32 = 0.2;

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

        let mut tree: KdTree<f32, u32, Eytzinger, VecOfArrays<f32, u32, 4, 32>, 4, 32> =
            KdTree::default();

        for (idx, point) in content_to_add.iter().enumerate() {
            tree.add(point, idx as u32).unwrap();
        }

        let query_points: Vec<[f32; 4]> =
            (0..NUM_QUERIES).map(|_| rng.random::<[f32; 4]>()).collect();

        for query_point in query_points {
            let expected = linear_search(&content_to_add, &query_point, RADIUS);

            let mut result: Vec<_> = tree
                .query(&query_point)
                .within::<Manhattan<f32>>(RADIUS)
                .execute()
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            stabilize_sort(&mut result);

            assert_eq!(result, expected);
        }
    }

    #[test]
    fn v6_within_queries_fix_issue_258() {
        // see https://github.com/sdd/kiddo/issues/258

        let points = issue_258_test_data();

        type Tree = KdTree<f64, usize, Eytzinger, VecOfArenas<f64, usize, 2, 32>, 2, 32>;

        // let tree = kdtree::ImmutableKdTree::<f64, usize, 2, 32>::new_from_slice(&points);
        let tree: Tree = Tree::new_from_slice(&points).unwrap();

        let dist = 6.0;
        let dist_sq = dist * dist;
        let test_index = 4;

        // let result = tree.within::<SquaredEuclidean<f64>>(&points[test_index], dist_sq);
        let result = tree
            .query(&points[test_index])
            .within::<SquaredEuclidean<f64>>(dist_sq)
            .execute();

        let mut expected = Vec::new();
        for (i, p) in points.iter().enumerate() {
            let d_sq = issue_258_distance_squared(&points[test_index], p);
            if d_sq <= dist_sq {
                expected.push(i);
            }
        }

        println!("Expected: {:?}", expected);
        let mut found = result.iter().map(|r| r.item).collect::<Vec<_>>();
        found.sort();

        println!("Found: {:?}", found);

        assert_eq!(expected, found);

        for r in &result {
            let neighbor = &points[r.item];
            let checked_squared = issue_258_distance_squared(&points[test_index], neighbor);
            assert_eq!(
                r.distance, checked_squared,
                "Point i={}, dist^2 kiddo ({}) !=  manual ({})",
                r.item, r.distance, checked_squared
            );

            println!(
                "Point i={}, dist^2 kiddo ({}) == manual ({})",
                r.item, r.distance, checked_squared
            );
        }
    }

    fn issue_258_distance_squared(a: &[f64; 2], b: &[f64; 2]) -> f64 {
        (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
    }

    // see https://github.com/sdd/kiddo/issues/258
    fn issue_258_test_data() -> Vec<[f64; 2]> {
        vec![
            [3317.0756414929883, 811.7122408967787],
            [3318.251752812044, 810.4325124656744],
            [3319.4278641310993, 809.1527840345702],
            [3320.603975450155, 807.8730556034661],
            [3321.7800867692104, 806.5933271723619],
            [3322.956198088266, 805.3135987412577],
            [3324.1323094073214, 804.0338703101535],
            [3325.1233671193845, 802.5234025770475],
            [3326.1144248314476, 801.0129348439415],
            [3325.995136117253, 799.9751975016522],
            [3325.8758474030583, 798.9374601593631],
            [3326.3636855948803, 797.2830881719838],
            [3326.851523786702, 795.6287161846043],
            [3327.339361978524, 793.9743441972249],
            [3328.3787056579517, 793.9846642699185],
            [3329.4180493373797, 793.9949843426123],
            [3330.046449352118, 795.7532903690851],
            [3330.6748493668565, 797.511596395558],
            [3331.303249381595, 799.2699024220307],
            [3331.9316493963333, 801.0282084485035],
            [3333.407854588155, 801.0867536751596],
            [3334.8840597799767, 801.1452989018157],
            [3336.605294699901, 801.2866065912483],
            [3338.3265296198256, 801.4279142806807],
            [3339.4855974129955, 799.8317465248002],
            [3340.644665206165, 798.2355787689198],
            [3341.803732999335, 796.6394110130393],
            [3342.962800792505, 795.0432432571588],
            [3344.3555759433593, 795.3023618471735],
            [3345.748351094214, 795.5614804371883],
            [3347.2328465494534, 795.8232509905426],
            [3348.08670862355, 795.9153914507592],
            [3349.559617942115, 795.9708806897642],
            [3350.6376181983205, 795.9827334003453],
            [3351.715618454526, 795.9945861109263],
            [3352.94028646958, 795.9971006088998],
            [3354.1649544846337, 795.9996151068733],
            [3355.4140794822115, 795.9998075661966],
            [3356.663204479789, 796.00000002552],
            [3357.913204479789, 796.00000001276],
            [3359.163204479789, 796.0],
            [3360.413204479789, 796.0],
            [3361.663204479789, 796.0],
            [3362.913204479789, 796.0],
            [3364.163204479789, 796.0],
            [3365.413204479789, 796.0],
            [3366.663204479789, 796.0],
            [3367.913204479789, 796.0],
            [3369.163204479789, 796.0],
            [3370.413204479789, 796.0],
            [3371.663204479789, 796.0],
            [3372.913204479789, 795.4473296001761],
            [3374.163204479789, 794.8946592003522],
            [3375.413204479789, 793.4432227738059],
            [3376.663204479789, 791.9917863472598],
            [3377.913204479789, 790.5397362845654],
            [3379.163204479789, 789.0876862218711],
            [3380.413204479789, 788.3528340151082],
            [3381.663204479789, 787.6179818083453],
            [3382.913204479789, 787.7815929334186],
            [3384.163204479789, 787.9452040584916],
            [3385.413204479789, 788.1091338254666],
            [3386.663204479789, 788.2730635924416],
            [3387.913204479789, 789.2799232655191],
            [3389.163204479789, 790.2867829385964],
            [3390.413204479789, 791.4385968429092],
            [3391.663204479789, 792.590410747222],
            [3392.909947993963, 793.0832482743491],
            [3394.156691508137, 793.5760858014762],
            [3395.351695376479, 793.218240953876],
            [3396.5466992448205, 792.8603961062759],
            [3397.5341084798883, 792.3360608910593],
            [3398.5215177149557, 791.8117256758427],
            [3399.864388200669, 791.948360011792],
            [3400.362416082534, 793.5833457391662],
            [3400.8604439643987, 795.2183314665405],
            [3401.358471846263, 796.8533171939147],
            [3401.856499728128, 798.4883029212889],
            [3402.5676357016137, 797.3490994260089],
            [3403.278771675099, 796.2098959307291],
            [3403.8182218519833, 795.104044878583],
            [3404.357672028868, 793.9981938264369],
            [3404.1893224114624, 792.2313615329177],
            [3404.0209727940573, 790.4645292393985],
            [3403.852623176652, 788.6976969458792],
            [3403.684273559247, 786.93086465236],
            [3403.5159239418413, 785.1640323588408],
            [3404.614894836291, 784.8565517117287],
            [3405.713865730741, 784.5490710646166],
            [3407.460747589397, 785.3224266341614],
            [3409.2076294480535, 786.0957822037062],
            [3410.9545113067097, 786.869137773251],
            [3412.701393165366, 787.6424933427959],
            [3414.448275024022, 788.4158489123406],
            [3416.195156882678, 789.1892044818854],
            [3417.9420387413343, 789.9625600514303],
            [3419.6889205999905, 790.7359156209751],
            [3421.435802458647, 791.5092711905199],
            [3423.182684317303, 792.2826267600648],
            [3424.9295661759593, 793.0559823296096],
            [3426.6764480346155, 793.8293378991544],
            [3428.423329893272, 794.6026934686993],
            [3430.170211751928, 795.3760490382441],
            [3431.917093610584, 796.1494046077889],
            [3433.66397546924, 796.9227601773337],
            [3435.4108573278963, 797.6961157468785],
            [3437.1577391865526, 798.4694713164233],
            [3438.904621045209, 799.2428268859682],
            [3440.651502903865, 800.016182455513],
        ]
    }

    fn linear_search<A, const K: usize>(
        content: &[[A; K]],
        query_point: &[A; K],
        radius: A,
    ) -> Vec<(A, u32)>
    where
        A: Axis<Coord = A> + 'static,
        Manhattan<A>: crate::dist::DistanceMetricCore<A, Output = A>,
    {
        let mut matching_items = vec![];

        for (idx, p) in content.iter().enumerate() {
            let dist = manhattan_dist(query_point, p);
            if dist <= radius {
                matching_items.push((dist, idx as u32));
            }
        }

        stabilize_sort(&mut matching_items);

        matching_items
    }

    fn manhattan_dist<A, const K: usize>(a: &[A; K], b: &[A; K]) -> A
    where
        A: Axis<Coord = A>,
        Manhattan<A>: crate::dist::DistanceMetricCore<A, Output = A>,
    {
        let aw = (*a)
            .map(|coord| <Manhattan<A> as crate::dist::DistanceMetricCore<A>>::widen_coord(coord));
        let bw = (*b)
            .map(|coord| <Manhattan<A> as crate::dist::DistanceMetricCore<A>>::widen_coord(coord));

        <Manhattan<A> as crate::dist::DistanceMetricCore<A>>::dist::<K>(&aw, &bw)
    }

    fn stabilize_sort<A>(matching_items: &mut [(A, u32)])
    where
        A: Axis<Coord = A>,
    {
        matching_items.sort_unstable_by(|a, b| {
            let dist_cmp = a.0.partial_cmp(&b.0).unwrap();
            if dist_cmp == Ordering::Equal {
                a.1.cmp(&b.1)
            } else {
                dist_cmp
            }
        });
    }
}

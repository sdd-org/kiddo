use std::num::NonZeroUsize;

use kiddo::dist::SquaredEuclidean;
use kiddo::leaf_view::LeafView;
use kiddo::traits::leaf_strategy::{
    BucketLimitType, ConstructibleLeafStrategy, Immutable, LeafProjection,
};
use kiddo::{Axis, Content, Eytzinger, KdTree, LeafStrategy, StemStrategy};

struct ExternalFlatLeaf<A, T, const K: usize, const B: usize> {
    leaf_points: [Vec<A>; K],
    leaf_items: Vec<T>,
    leaf_extents: Vec<(u32, u32)>,
    size: usize,
}

impl<A, T, const K: usize, const B: usize> std::fmt::Debug for ExternalFlatLeaf<A, T, K, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExternalFlatLeaf")
            .field("dimensions", &K)
            .field("bucket_size", &B)
            .field("size", &self.size)
            .field("leaf_count", &self.leaf_extents.len())
            .finish()
    }
}

impl<A, T, SS, const K: usize, const B: usize> LeafStrategy<A, T, SS, K, B>
    for ExternalFlatLeaf<A, T, K, B>
where
    A: Axis<Coord = A>,
    T: Content,
    SS: StemStrategy,
{
    type Num = A;
    type Mutability = Immutable;

    const BUCKET_LIMIT_TYPE: BucketLimitType = BucketLimitType::Soft;
    const LEAF_PROJECTION: LeafProjection = LeafProjection::LeafView;

    fn size(&self) -> usize {
        self.size
    }

    fn leaf_count(&self) -> usize {
        self.leaf_extents.len()
    }

    fn leaf_len(&self, leaf_idx: usize) -> usize {
        let (start, end) = self.leaf_extents[leaf_idx];
        (end - start) as usize
    }

    fn leaf_view(&self, leaf_idx: usize) -> LeafView<'_, A, T, K, B> {
        let (start, end) = self.leaf_extents[leaf_idx];
        let start = start as usize;
        let end = end as usize;

        let points = array_init::array_init(|dim| &self.leaf_points[dim][start..end]);
        let items = &self.leaf_items[start..end];

        LeafView::new(points, items)
    }
}

impl<A, T, SS, const K: usize, const B: usize> ConstructibleLeafStrategy<A, T, SS, K, B>
    for ExternalFlatLeaf<A, T, K, B>
where
    A: Axis<Coord = A>,
    T: Content,
    SS: StemStrategy,
{
    fn new_with_capacity(capacity: usize) -> Self {
        Self {
            leaf_points: array_init::array_init(|_| Vec::with_capacity(capacity)),
            leaf_items: Vec::with_capacity(capacity),
            leaf_extents: Vec::new(),
            size: 0,
        }
    }

    fn append_leaf(&mut self, leaf_points: &[&[A]; K], leaf_items: &[T]) {
        let start = self.leaf_items.len() as u32;
        let len = leaf_items.len();
        let end = start + len as u32;

        for (dim, item) in leaf_points.iter().enumerate() {
            self.leaf_points[dim].extend_from_slice(item);
        }
        self.leaf_items.extend_from_slice(leaf_items);
        self.leaf_extents.push((start, end));
        self.size += len;
    }
}

type TestTree = KdTree<f64, u32, Eytzinger, ExternalFlatLeaf<f64, u32, 2, 32>, 2, 32>;

fn query_items<const K: usize>(
    tree: &KdTree<f64, u32, Eytzinger, ExternalFlatLeaf<f64, u32, K, 32>, K, 32>,
    point: &[f64; K],
    n: NonZeroUsize,
) -> Vec<u32> {
    tree.query(point)
        .nearest_n::<SquaredEuclidean<f64>>(n)
        .execute()
        .into_iter()
        .map(|result| result.item)
        .collect()
}

#[test]
fn external_leaf_strategy_supports_build_and_execute_query() {
    let points = [
        (10u32, [0.0f64, 0.0]),
        (20u32, [1.0, 1.0]),
        (30u32, [2.0, 2.0]),
    ];
    let tree: TestTree = KdTree::new_from_entries(&points).unwrap();

    assert_eq!(tree.size(), 3);
    assert_eq!(tree.leaf_count(), 1);

    let items = query_items::<2>(&tree, &[0.1, 0.1], NonZeroUsize::new(2).unwrap());

    assert_eq!(items, vec![10, 20]);
}

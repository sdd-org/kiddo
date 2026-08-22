use crate::leaf_view::LeafView;
use crate::traits::leaf_strategy::{
    BucketLimitType, ConstructibleLeafStrategy, Immutable, LeafProjection,
};
use crate::{Axis, Content, KdTree, LeafStrategy, StemStrategy};

/// A leaf storage strategy using flat vectors for coordinates.
///
/// Stores coordinates as `K` separate vectors (one per dimension) and items in a
/// separate vector. Leaves are not stored as standalone structs; instead,
/// `leaf_extents` records the `(start, end)` range for each logical leaf within
/// the shared per-dimension arrays.
///
/// Memory layout:
///
/// ```text
/// leaf_extents = [(0, 2), (2, 5), (5, 6)]
///
/// leaf_points[0] = [ x00 x01 | x10 x11 x12 | x20 ]
/// leaf_points[1] = [ y00 y01 | y10 y11 y12 | y20 ]
/// leaf_points[2] = [ z00 z01 | z10 z11 z12 | z20 ]
/// leaf_items    = [ i00 i01 | i10 i11 i12 | i20 ]
///                  \ leaf 0 / \  leaf 1   / \l2/
/// ```
///
/// This is effectively a global column-store layout with per-leaf slice ranges.
/// It is friendly to autovectorisation and explicit SIMD because each axis is
/// already contiguous, but processing one leaf still means touching multiple
/// independent streams: one vector per axis plus the item vector.
#[cfg_attr(
    feature = "rkyv_08",
    derive(rkyv_08::Archive, rkyv_08::Serialize, rkyv_08::Deserialize)
)]
#[cfg_attr(feature = "rkyv_08", rkyv(crate = rkyv_08))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "A: serde::Serialize, T: serde::Serialize",
        deserialize = "A: serde::Deserialize<'de> + Copy + Default, T: serde::Deserialize<'de>"
    ))
)]
pub struct FlatVec<A, T, const K: usize, const B: usize> {
    #[cfg_attr(feature = "serde", serde(with = "crate::custom_serde::array_of_vecs"))]
    leaf_points: [Vec<A>; K],
    leaf_items: Vec<T>,
    leaf_extents: Vec<(u32, u32)>,
    size: usize,
}

impl<A, T, const K: usize, const B: usize> std::fmt::Debug for FlatVec<A, T, K, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlatVec")
            .field("dimensions", &K)
            .field("bucket_size", &B)
            .field("size", &self.leaf_items.len())
            .field("leaf_count", &self.leaf_extents.len())
            .finish()
    }
}

#[cfg(feature = "rkyv_08")]
impl<A, T, const K: usize, const B: usize> std::fmt::Debug for ArchivedFlatVec<A, T, K, B>
where
    A: rkyv_08::Archive,
    T: rkyv_08::Archive,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArchivedFlatVec")
            .field("dimensions", &K)
            .field("bucket_size", &B)
            .field("size", &self.leaf_items.len())
            .field("leaf_count", &self.leaf_extents.len())
            .finish()
    }
}

/// Convenience type alias for a [`KdTree`] with a [`FlatVec`] leaf strategy, to avoid having to
/// specify the KdTree's generic parameters twice.
pub type FlatVecKdTree<A, T, SS, const K: usize, const B: usize> =
    KdTree<A, T, SS, FlatVec<A, u32, K, 32>, K, 32>;

impl<AX, T, SS, const K: usize, const B: usize> LeafStrategy<AX, T, SS, K, B>
    for FlatVec<AX, T, K, B>
where
    AX: Axis<Coord = AX>,
    T: Content,
    SS: StemStrategy,
{
    type Num = AX;
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
        let (start, end) = unsafe { *self.leaf_extents.get_unchecked(leaf_idx) };
        (end - start) as usize
    }

    fn leaf_view(&self, leaf_idx: usize) -> LeafView<'_, AX, T, K, B> {
        let (start, end) = unsafe { *self.leaf_extents.get_unchecked(leaf_idx) };
        let start = start as usize;
        let end = end as usize;

        let leaf_points_view = array_init::array_init(|i| unsafe {
            self.leaf_points.get_unchecked(i).get_unchecked(start..end)
        });

        let leaf_items_view = unsafe { self.leaf_items.get_unchecked(start..end) };

        LeafView::new(leaf_points_view, leaf_items_view)
    }

    fn replace_item_in_leaf(
        &mut self,
        leaf_idx: usize,
        point: &[AX; K],
        old_item: T,
        new_item: T,
    ) -> bool
    where
        T: PartialEq,
    {
        let (start, end) = unsafe { *self.leaf_extents.get_unchecked(leaf_idx) };
        let start = start as usize;
        let end = end as usize;

        for idx in start..end {
            let point_matches = (0..K).all(|dim| self.leaf_points[dim][idx] == point[dim]);
            if point_matches && self.leaf_items[idx] == old_item {
                self.leaf_items[idx] = new_item;
                return true;
            }
        }

        false
    }
}

#[cfg(feature = "rkyv_08")]
impl<AX, T, SS, const K: usize, const B: usize> LeafStrategy<AX, T, SS, K, B>
    for ArchivedFlatVec<AX, T, K, B>
where
    AX: rkyv_08::Archive + Axis<Coord = AX>,
    T: rkyv_08::Archive + Content,
    SS: StemStrategy,
{
    type Num = AX;
    type Mutability = Immutable;

    const BUCKET_LIMIT_TYPE: BucketLimitType = BucketLimitType::Soft;
    const LEAF_PROJECTION: LeafProjection = LeafProjection::LeafView;

    fn size(&self) -> usize {
        self.size.to_native() as usize
    }

    fn leaf_count(&self) -> usize {
        self.leaf_extents.len()
    }

    fn leaf_len(&self, leaf_idx: usize) -> usize {
        let extent = unsafe { self.leaf_extents.get_unchecked(leaf_idx) };
        let start = extent.0.to_native() as usize;
        let end = extent.1.to_native() as usize;
        end - start
    }

    fn leaf_view(&self, leaf_idx: usize) -> LeafView<'_, AX, T, K, B> {
        let extent = unsafe { self.leaf_extents.get_unchecked(leaf_idx) };
        let start = extent.0.to_native() as usize;
        let end = extent.1.to_native() as usize;

        let leaf_points_view = array_init::array_init(|i| {
            crate::rkyv::utils::transform_slice::<AX, _>(
                unsafe { self.leaf_points.get_unchecked(i) }
                    .as_slice()
                    .get(start..end)
                    .unwrap(),
            )
        });

        let leaf_items_view = crate::rkyv::utils::transform_slice::<T, _>(
            self.leaf_items.as_slice().get(start..end).unwrap(),
        );

        LeafView::new(leaf_points_view, leaf_items_view)
    }
}

impl<AX, T, SS, const K: usize, const B: usize> ConstructibleLeafStrategy<AX, T, SS, K, B>
    for FlatVec<AX, T, K, B>
where
    AX: Axis<Coord = AX>,
    T: Content,
    SS: StemStrategy,
{
    fn new_with_capacity(capacity: usize) -> Self {
        Self {
            leaf_points: array_init::array_init(|_| Vec::with_capacity(capacity)),
            leaf_items: Vec::with_capacity(capacity),
            leaf_extents: Vec::with_capacity(capacity),
            size: 0,
        }
    }

    fn append_leaf(&mut self, leaf_points: &[&[AX]; K], leaf_items: &[T]) {
        let chunk_length = leaf_items.len();

        debug_assert!(leaf_points[0].len() == chunk_length);
        for d in leaf_points.iter() {
            debug_assert!(d.len() == chunk_length);
        }

        self.leaf_extents.push((
            self.leaf_items.len() as u32,
            (self.leaf_items.len() + chunk_length) as u32,
        ));

        for dim in 0..K {
            self.leaf_points[dim].extend_from_slice(&leaf_points[dim][..chunk_length]);
        }
        self.leaf_items
            .extend_from_slice(&leaf_items[..chunk_length]);
    }
}

#[cfg(test)]
mod test {
    use fixed::{types::extra::U8, FixedU16};
    use rand::RngExt;
    use std::num::NonZeroUsize;

    use crate::dist::SquaredEuclidean;
    use crate::kd_tree::KdTreeAccessor;
    use crate::leaf_strategy::flat_vec::FlatVec;
    use crate::LeafStrategy;
    use crate::{kd_tree, Eytzinger};

    #[test]
    fn default_constructs_flat_vec_kd_tree() {
        let tree: kd_tree::KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, 3, 32>, 3, 32> =
            kd_tree::KdTree::default();

        assert!(tree.is_empty());
        assert_eq!(tree.size(), 0);
    }

    #[test]
    fn create_single_leaf_flat_vec_float_kd_tree() {
        let points: Vec<[f32; 3]> = vec![[1.0f32, 2.0f32, 3.0f32]];
        let tree: kd_tree::KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, 3, 32>, 3, 32> =
            kd_tree::KdTree::new_from_slice(&points).unwrap();

        assert_eq!(tree.size(), 1);

        let leaf_view =
            <FlatVec<f32, u32, 3, 32> as LeafStrategy<f32, u32, Eytzinger, 3, 32>>::leaf_view(
                tree.leaves(),
                0,
            );

        let (leaf_points, leaf_items) = leaf_view.into_parts();
        assert_eq!(leaf_points[0][0], points[0][0]);
        assert_eq!(leaf_points[1][0], points[0][1]);
        assert_eq!(leaf_points[2][0], points[0][2]);
        assert_eq!(leaf_items, vec![0]);
    }

    #[test]
    fn create_single_leaf_flat_vec_float_no_items_kd_tree() {
        let points: Vec<[f32; 3]> = vec![[1.0f32, 2.0f32, 3.0f32]];
        let tree: kd_tree::KdTree<f32, (), Eytzinger, FlatVec<f32, (), 3, 32>, 3, 32> =
            kd_tree::KdTree::new_from_slice_no_items(&points).unwrap();

        assert_eq!(tree.size(), 1);

        let leaf_view =
            <FlatVec<f32, (), 3, 32> as LeafStrategy<f32, (), Eytzinger, 3, 32>>::leaf_view(
                tree.leaves(),
                0,
            );

        let (leaf_points, leaf_items) = leaf_view.into_parts();
        assert_eq!(leaf_points[0][0], points[0][0]);
        assert_eq!(leaf_points[1][0], points[0][1]);
        assert_eq!(leaf_points[2][0], points[0][2]);
        assert_eq!(leaf_items, vec![()]);
    }

    #[test]
    fn create_single_leaf_flat_vec_fixed_point_kd_tree() {
        let points: Vec<[FixedU16<U8>; 3]> = vec![[1.into(), 2.into(), 3.into()]];
        let tree: kd_tree::KdTree<
            FixedU16<U8>,
            u32,
            Eytzinger,
            FlatVec<FixedU16<U8>, u32, 3, 32>,
            3,
            32,
        > = kd_tree::KdTree::new_from_slice(&points).unwrap();

        assert_eq!(tree.size(), 1);

        let leaf_view = <FlatVec<FixedU16<U8>, u32, 3, 32> as LeafStrategy<
            FixedU16<U8>,
            u32,
            Eytzinger,
            3,
            32,
        >>::leaf_view(tree.leaves(), 0);

        let (leaf_points, leaf_items) = leaf_view.into_parts();
        assert_eq!(leaf_points[0][0], points[0][0]);
        assert_eq!(leaf_points[1][0], points[0][1]);
        assert_eq!(leaf_points[2][0], points[0][2]);
        assert_eq!(leaf_items, vec![0]);
    }

    #[test]
    fn create_multiple_leaf_flat_vec_float_kd_tree() {
        // create 2^16 random 3d points in the unit cube
        let mut rng = rand::rng();
        let mut points: Vec<[f32; 3]> = vec![];
        for _ in 0..65_536 {
            let x = rng.random_range(0.0..1.0);
            let y = rng.random_range(0.0..1.0);
            let z = rng.random_range(0.0..1.0);
            points.push([x, y, z]);
        }

        let tree: kd_tree::KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, 3, 32>, 3, 32> =
            kd_tree::KdTree::new_from_slice(&points).unwrap();

        assert!(!tree.is_empty());
        assert_eq!(tree.size(), 65_536);
        assert_eq!(tree.leaf_count(), 2048);
        assert_eq!(tree.max_stem_level(), 10);

        // perform a best_n_within query
        let query_point = [0.5, 0.5, 0.5];
        let radius = 0.1;
        let max_qty = NonZeroUsize::new(10).unwrap();
        let results = tree
            .query(&query_point)
            .best_n_within::<SquaredEuclidean<f32>>(radius, max_qty)
            .execute();
        assert_eq!(results.len(), 10);
    }
}

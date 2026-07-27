use crate::kd_tree::ConstructionError;
use crate::leaf_view::LeafView;
use crate::mirror_select_nth_unstable_by::mirror_select_nth_unstable_by;
use crate::traits::leaf_strategy::{
    BucketLimitType, ConstructibleLeafStrategy, LeafProjection, Mutable, MutableLeafStrategy,
};
use crate::{Axis, Content, LeafStrategy, StemStrategy};

/// A leaf storage strategy using vectors of fixed-size arrays.
///
/// Each leaf is a standalone `LeafNode` containing `K` fixed-capacity coordinate
/// arrays, one fixed-capacity item array, and a logical `size`. This is the
/// mutable, hard-bucket layout.
///
/// Memory layout:
///
/// ```text
/// leaves: Vec<LeafNode>
///
/// leaves[0]
///   content_points[0] = [ x00 x01 x02  ..  .. ]
///   content_points[1] = [ y00 y01 y02  ..  .. ]
///   content_points[2] = [ z00 z01 z02  ..  .. ]
///   content_items    = [ i00 i01 i02  ..  .. ]
///   size = 3
///
/// leaves[1]
///   content_points[0] = [ x10 x11 .. .. .. ]
///   content_points[1] = [ y10 y11 .. .. .. ]
///   content_points[2] = [ z10 z11 .. .. .. ]
///   content_items    = [ i10 i11 .. .. .. ]
///   size = 2
/// ```
///
/// The unused tail of each leaf stays allocated so inserts/removals can happen
/// in place up to bucket capacity `B`.
#[cfg_attr(
    feature = "rkyv_08",
    derive(rkyv_08::Archive, rkyv_08::Serialize, rkyv_08::Deserialize)
)]
#[cfg_attr(feature = "rkyv_08", rkyv(crate = rkyv_08))]
#[cfg_attr(feature = "rkyv_08", rkyv(attr(allow(missing_docs))))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "A: serde::Serialize, T: serde::Serialize",
        deserialize = "A: serde::Deserialize<'de> + Copy + Default, T: serde::Deserialize<'de> + Copy + Default"
    ))
)]
pub struct VecOfArrays<A, T, const K: usize, const B: usize> {
    leaves: Vec<LeafNode<A, T, K, B>>,
    size: usize,
}

/// A single leaf node storing up to `B` points.
///
/// Layout within one leaf:
///
/// ```text
/// content_points: [[A; B]; K]
///
/// dim 0 -> [ x0 x1 x2 ... ]
/// dim 1 -> [ y0 y1 y2 ... ]
/// dim 2 -> [ z0 z1 z2 ... ]
///
/// content_items: [ i0 i1 i2 ... ]
/// size: number of live entries from the front
/// ```
#[cfg_attr(
    feature = "rkyv_08",
    derive(rkyv_08::Archive, rkyv_08::Serialize, rkyv_08::Deserialize)
)]
#[cfg_attr(feature = "rkyv_08", rkyv(crate = rkyv_08))]
#[cfg_attr(feature = "rkyv_08", rkyv(attr(allow(missing_docs))))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "A: serde::Serialize, T: serde::Serialize",
        deserialize = "A: serde::Deserialize<'de> + Copy + Default, T: serde::Deserialize<'de> + Copy + Default"
    ))
)]
#[allow(missing_docs)]
#[derive(Clone, Debug, PartialEq)]
pub struct LeafNode<A, T, const K: usize, const B: usize> {
    /// Point coordinates organized by dimension
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::custom_serde::array_of_arrays")
    )]
    pub content_points: [[A; B]; K],
    /// Items associated with each point
    #[cfg_attr(feature = "serde", serde(with = "crate::custom_serde::array"))]
    pub content_items: [T; B],
    /// Number of points currently stored
    pub size: usize,
}

impl<AX, T, SS, const K: usize, const B: usize> LeafStrategy<AX, T, SS, K, B>
    for VecOfArrays<AX, T, K, B>
where
    AX: Axis<Coord = AX>,
    T: Content,
    SS: StemStrategy,
{
    type Num = AX;
    type Mutability = Mutable;

    const BUCKET_LIMIT_TYPE: BucketLimitType = BucketLimitType::Hard;
    const LEAF_PROJECTION: LeafProjection = LeafProjection::LeafView;

    fn size(&self) -> usize {
        self.size
    }

    fn leaf_count(&self) -> usize {
        self.leaves.len()
    }

    fn leaf_len(&self, leaf_idx: usize) -> usize {
        self.leaves[leaf_idx].size
    }

    fn leaf_view(&self, leaf_idx: usize) -> LeafView<'_, AX, T, K, B> {
        let leaf = &self.leaves[leaf_idx];

        let points: [&[AX]; K] =
            array_init::array_init(|i| &leaf.content_points[i].as_slice()[..leaf.size]);
        let leaf_items_view = &leaf.content_items[..leaf.size];

        LeafView::new(points, leaf_items_view)
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
        let leaf = &mut self.leaves[leaf_idx];

        for item_idx in 0..leaf.size {
            let point_matches = (0..K).all(|dim| leaf.content_points[dim][item_idx] == point[dim]);
            if point_matches && leaf.content_items[item_idx] == old_item {
                leaf.content_items[item_idx] = new_item;
                return true;
            }
        }

        false
    }
}

#[cfg(feature = "rkyv_08")]
impl<AX, T, SS, const K: usize, const B: usize> LeafStrategy<AX, T, SS, K, B>
    for ArchivedVecOfArrays<AX, T, K, B>
where
    AX: rkyv_08::Archive + Axis<Coord = AX>,
    T: rkyv_08::Archive + Content,
    SS: StemStrategy,
{
    type Num = AX;
    type Mutability = Mutable;

    const BUCKET_LIMIT_TYPE: BucketLimitType = BucketLimitType::Hard;
    const LEAF_PROJECTION: LeafProjection = LeafProjection::LeafView;

    fn size(&self) -> usize {
        self.size.to_native() as usize
    }

    fn leaf_count(&self) -> usize {
        self.leaves.len()
    }

    fn leaf_len(&self, leaf_idx: usize) -> usize {
        self.leaves[leaf_idx].size.to_native() as usize
    }

    fn leaf_view(&self, leaf_idx: usize) -> LeafView<'_, AX, T, K, B> {
        let leaf = &self.leaves[leaf_idx];
        let leaf_len = leaf.size.to_native() as usize;

        let points: [&[AX]; K] = array_init::array_init(|i| {
            crate::rkyv::utils::transform_slice::<AX, _>(
                leaf.content_points[i].as_slice().get(..leaf_len).unwrap(),
            )
        });
        let leaf_items_view = crate::rkyv::utils::transform_slice::<T, _>(
            leaf.content_items.as_slice().get(..leaf_len).unwrap(),
        );

        LeafView::new(points, leaf_items_view)
    }
}

impl<AX, T, const K: usize, const B: usize> VecOfArrays<AX, T, K, B>
where
    AX: Axis<Coord = AX>,
    T: Content,
{
    // Enforce bucket size of at least 2. B=1 can trigger UB in split fn
    // https://github.com/sdd/kiddo/issues/295
    const _VALID_B: () = {
        assert!(
            B > 1,
            "Bucket size must be at least 2 with VecOfArrays leaf strategy"
        );
    };

    fn new(leaves: Vec<LeafNode<AX, T, K, B>>, size: usize) -> Self {
        #[allow(clippy::let_unit_value)]
        let _ = Self::_VALID_B;

        Self { leaves, size }
    }
}

impl<AX, T, SS, const K: usize, const B: usize> ConstructibleLeafStrategy<AX, T, SS, K, B>
    for VecOfArrays<AX, T, K, B>
where
    AX: Axis<Coord = AX>,
    T: Content,
    SS: StemStrategy,
{
    fn new_with_capacity(capacity: usize) -> Self {
        Self::new(Vec::with_capacity(capacity / B + 1), 0)
    }

    fn new_with_empty_leaf() -> Self {
        let mut result = Self::new(Vec::new(), 0);

        let leaf = LeafNode {
            content_points: [[AX::zero(); B]; K],
            content_items: [T::default(); B],
            size: 0,
        };

        result.leaves.push(leaf);

        result
    }

    fn append_leaf(&mut self, leaf_points: &[&[AX]; K], leaf_items: &[T]) {
        let leaf_len = leaf_items.len();
        debug_assert!(leaf_len <= B);

        // Sanity: all dims should have the same length
        debug_assert!(leaf_points.iter().all(|p| p.len() == leaf_len));

        // Initialize fixed-size storage with defaults
        let mut content_points: [[AX; B]; K] = array_init::array_init(|_| [AX::zero(); B]);
        let mut content_items: [T; B] = [T::default(); B];

        // Copy the actual data into the first `leaf_len` slots
        for i in 0..leaf_len {
            for dim in 0..K {
                content_points[dim][i] = leaf_points[dim][i];
            }
            content_items[i] = leaf_items[i];
        }

        self.leaves.push(LeafNode {
            content_points,
            content_items,
            size: leaf_len,
        });
        self.size += leaf_len;
    }
}

impl<AX, T, const K: usize, const B: usize> VecOfArrays<AX, T, K, B>
where
    AX: Axis,
    T: Copy + PartialEq,
{
    fn should_remove(
        leaf: &LeafNode<AX, T, K, B>,
        point: &[AX; K],
        item: T,
        leaf_idx: usize,
    ) -> bool {
        for dim in 0..K {
            if leaf.content_points[dim][leaf_idx] != point[dim] {
                return false;
            }
        }

        if leaf.content_items[leaf_idx] != item {
            return false;
        }

        true
    }
}

impl<AX, T, const K: usize, const B: usize> VecOfArrays<AX, T, K, B>
where
    AX: Axis<Coord = AX>,
    T: Content + PartialEq,
{
    fn copy_split_data_to_new_leaf(
        &mut self,
        old_leaf: &LeafNode<AX, T, K, B>,
        pivot_idx: usize,
    ) -> usize {
        debug_assert!(pivot_idx < B, "pivot_idx out of bounds");

        // Create a new empty leaf
        let mut new_leaf = LeafNode {
            content_points: [[AX::zero(); B]; K],
            content_items: [T::default(); B],
            size: 0,
        };

        // Copy the right half to the new leaf
        let num_items = B - pivot_idx;
        unsafe {
            // Copy each dimension's data
            for dim in 0..K {
                new_leaf
                    .content_points
                    .get_unchecked_mut(dim)
                    .get_unchecked_mut(..num_items)
                    .copy_from_slice(
                        old_leaf
                            .content_points
                            .get_unchecked(dim)
                            .get_unchecked(pivot_idx..B),
                    );
            }
            // Copy items
            new_leaf
                .content_items
                .get_unchecked_mut(..num_items)
                .copy_from_slice(old_leaf.content_items.get_unchecked(pivot_idx..B));
        }

        // set new leaf size
        new_leaf.size = num_items;

        // Add the new leaf
        self.leaves.push(new_leaf);

        // return the new leaf index
        self.leaves.len() - 1
    }
}

impl<AX, T, SS, const K: usize, const B: usize> MutableLeafStrategy<AX, T, SS, K, B>
    for VecOfArrays<AX, T, K, B>
where
    AX: Axis<Coord = AX>,
    T: Content + PartialEq,
    SS: StemStrategy,
{
    fn add_to_leaf(&mut self, leaf_idx: usize, point: &[AX; K], item: T) {
        debug_assert!(leaf_idx < self.leaves.len(), "leaf_idx out of bounds");
        let leaf = unsafe { self.leaves.get_unchecked_mut(leaf_idx) };

        let idx = leaf.size;
        debug_assert!(leaf.size < B, "leaf is full (max capacity reached)");

        for dim in 0..K {
            leaf.content_points[dim][idx] = unsafe { *point.get_unchecked(dim) };
        }
        leaf.content_items[idx] = item;

        leaf.size += 1;
        self.size += 1;
    }

    fn is_leaf_full(&self, leaf_idx: usize) -> bool {
        debug_assert!(leaf_idx < self.leaves.len(), "leaf_idx out of bounds");

        let leaf = unsafe { self.leaves.get_unchecked(leaf_idx) };

        leaf.size >= B
    }

    fn remove_from_leaf(&mut self, leaf_idx: usize, point: &[AX; K], item: T) {
        debug_assert!(leaf_idx < self.leaves.len(), "leaf_idx out of bounds");
        let leaf = unsafe { self.leaves.get_unchecked_mut(leaf_idx) };

        let mut new_idx = 0;
        for curr_idx in 0..leaf.size {
            // skip items that need removal
            if Self::should_remove(leaf, point, item, curr_idx) {
                continue;
            }

            // if no items have yet needed removal, no action yet needs to be taken
            if new_idx == curr_idx {
                new_idx += 1;
                continue;
            }

            // curr item needs to be kept but needs copying down to lower position in leaf
            for dim in 0..K {
                leaf.content_points[dim][new_idx] = leaf.content_points[dim][curr_idx];
            }
            leaf.content_items[new_idx] = leaf.content_items[curr_idx];
            new_idx += 1;
        }
        self.size -= leaf.size - new_idx;
        leaf.size = new_idx;
    }

    /// Splits this leaf by:
    /// * sorting it along split_dim
    /// * finding the midpoint
    /// * creating a new empty leaf
    /// * moving items from the midpoint onwards to the new leaf
    ///
    /// Returns: A tuple of (split_value, new_leaf_idx)
    fn split_leaf(
        &mut self,
        leaf_idx: usize,
        split_dim: usize,
    ) -> Result<(AX, usize), ConstructionError> {
        debug_assert!(leaf_idx < self.leaves.len(), "leaf_idx out of bounds");
        unsafe {
            let orig = self.leaves.get_unchecked_mut(leaf_idx);

            let mut pivot_idx: usize = orig.size / 2;

            // Create a mirror array to track the permutation
            let mut mirror: [usize; B] = array_init::array_init(|i| i);

            // Sort the split dimension and track the permutation in mirror
            mirror_select_nth_unstable_by(
                &mut orig.content_points[split_dim][..orig.size],
                &mut mirror[..orig.size],
                pivot_idx,
                |a, b| a.partial_cmp(b).expect("Leaf node sort failed."),
            );

            let mut split_val = *orig
                .content_points
                .get_unchecked(split_dim)
                .get_unchecked(pivot_idx);
            let mut unsplittable = false;

            // At this point we hava a candidate position at which to split the leaf, that
            // being the midpoint. This may not be a valid location, however. Take this 1D case
            // as an example. We've sorted our leaf and it looks like this:
            //
            // [1, 2, 2, 4, 5]
            //
            // If we choose the actual midpoint and move it, and everything to the right of it,
            // to a new leaf, and we choose 2 as the new stem value, what about the 2 that was to the
            // left of it? It is now in the wrong bucket. So, we need to adjust our split point by
            // moving it one place to the left.
            //
            // What about in this situation though?
            //
            // [2, 2, 2, 4, 5]
            //
            // If we follow the same algorithm, we end up with a split that has zero items in the
            // left bucket and 5 on the right. This is no good! If we're splitting so that we can add
            // a 4, for example, once the split completes, we still can't add it!
            //
            // So, if we've tried to adjust the split point leftwards, and got all the way to the
            // start of the bucket, we reset back to the midpoint and start moving rightwards.
            // In the above case this ends up with us splitting into [2, 2, 2] and [4, 5], with
            // the pivot value being 4.
            //
            // But what if we had a bucket that looked like this?
            //
            // [2, 2, 2, 2, 2]
            //
            // Well then, to put it bluntly, we're fucked. This VecOfArrays leaf strategy has
            // fixed-size buckets by design, to avoid the extra layer of indirection that would result
            // from using a Vec for the bucket. There is no way to store another 2 in the tree - you need
            // to either increase the bucket size or switch to a different, more permissive leaf strategy.

            // if the pivot point choice results in some items whose position on the split
            // dimension is the same as the split value being on the wrong side of the split:
            if *orig
                .content_points
                .get_unchecked(split_dim)
                .get_unchecked(pivot_idx - 1)
                == split_val
            {
                let orig_pivot_idx = pivot_idx;

                // Ensure that if pivot index would result in items that share the same co-ordinate
                // on the splitting dimension would end up on different sides of the split, that we
                // move the pivot to prevent this. We first try moving down, since that's the only
                // part of the bucket that was sorted by mirror_select_nth_unstable_by
                while pivot_idx > 0
                    && *orig
                        .content_points
                        .get_unchecked(split_dim)
                        .get_unchecked(pivot_idx - 1)
                        == split_val
                {
                    pivot_idx -= 1;
                }

                // If the attempt to move the pivot point above would have resulted in the pivot
                // point moving to the start of the bucket, search forwards from the original
                // pivot point instead. We need to first ensure the upper half of the bucket
                // is sorted
                if pivot_idx == 0 {
                    // Re-sort to ensure the entire array is sorted
                    mirror_select_nth_unstable_by(
                        &mut orig.content_points[split_dim][..orig.size],
                        &mut mirror[..orig.size],
                        orig.size - 1,
                        |a, b| a.partial_cmp(b).expect("Leaf node sort failed."),
                    );

                    pivot_idx = orig_pivot_idx;
                    while *orig
                        .content_points
                        .get_unchecked(split_dim)
                        .get_unchecked(pivot_idx)
                        == split_val
                    {
                        pivot_idx += 1;

                        if pivot_idx == B {
                            unsplittable = true;
                            break;
                        }
                    }
                }

                if !unsplittable {
                    split_val = *orig
                        .content_points
                        .get_unchecked(split_dim)
                        .get_unchecked(pivot_idx);
                }
            }

            // `mirror` tracks the cumulative permutation back to the original leaf,
            // including the optional second selection pass above. Apply it exactly
            // once so every coordinate and item remains associated.
            for dim in 0..K {
                if dim != split_dim {
                    let temp = orig.content_points[dim];
                    for i in 0..orig.size {
                        orig.content_points[dim][i] = temp[mirror[i]];
                    }
                }
            }
            let temp_items = orig.content_items;
            for i in 0..orig.size {
                orig.content_items[i] = temp_items[mirror[i]];
            }

            if unsplittable {
                return Err(ConstructionError::UnsplittableBucket { split_dim });
            }

            // At this point, we have a valid index at which we can split the bucket.
            // We need to copy the old leaf data before we can borrow self again
            let old_leaf_copy = orig.clone();
            orig.size = pivot_idx;

            let new_leaf_idx = self.copy_split_data_to_new_leaf(&old_leaf_copy, pivot_idx);

            Ok((split_val, new_leaf_idx))
        }
    }
}

#[cfg(test)]
mod test {
    use fixed::{types::extra::U8, FixedU16};
    use rand::RngExt;

    use crate::dist::SquaredEuclidean;
    use crate::kd_tree::KdTreeAccessor;
    #[cfg(feature = "rkyv_08")]
    use crate::leaf_strategy::vec_of_arrays::ArchivedVecOfArrays;
    use crate::leaf_strategy::vec_of_arrays::VecOfArrays;
    use crate::traits::leaf_strategy::{ConstructibleLeafStrategy, MutableLeafStrategy};
    use crate::LeafStrategy;
    use crate::{kd_tree, Eytzinger};

    #[test]
    fn default_constructs_vec_of_arrays_kd_tree() {
        let tree: kd_tree::KdTree<f32, u32, Eytzinger, VecOfArrays<f32, u32, 3, 32>, 3, 32> =
            kd_tree::KdTree::default();

        assert!(tree.is_empty());
        assert_eq!(tree.size(), 0);
    }

    #[test]
    fn create_single_leaf_vec_of_arrays_float_kd_tree() {
        let points: Vec<[f32; 3]> = vec![[1.0f32, 2.0f32, 3.0f32]];
        let tree: kd_tree::KdTree<f32, u32, Eytzinger, VecOfArrays<f32, u32, 3, 32>, 3, 32> =
            kd_tree::KdTree::new_from_slice(&points).unwrap();

        assert_eq!(tree.size(), 1);

        let leaf_view =
            <VecOfArrays<f32, u32, 3, 32> as LeafStrategy<f32, u32, Eytzinger, 3, 32>>::leaf_view(
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
    fn create_single_leaf_vec_of_arrays_fixed_point_kd_tree() {
        let points: Vec<[FixedU16<U8>; 3]> = vec![[1.into(), 2.into(), 3.into()]];
        let tree: kd_tree::KdTree<
            FixedU16<U8>,
            u32,
            Eytzinger,
            VecOfArrays<FixedU16<U8>, u32, 3, 32>,
            3,
            32,
        > = kd_tree::KdTree::new_from_slice(&points).unwrap();

        assert_eq!(tree.size(), 1);

        let leaf_view = <VecOfArrays<FixedU16<U8>, u32, 3, 32> as LeafStrategy<
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
    fn create_single_leaf_vec_of_arrays_float_no_items_kd_tree() {
        let points: Vec<[f32; 3]> = vec![[1.0f32, 2.0f32, 3.0f32]];

        let tree: kd_tree::KdTree<f32, (), Eytzinger, VecOfArrays<f32, (), 3, 32>, 3, 32> =
            kd_tree::KdTree::new_from_slice_no_items(&points).unwrap();

        assert_eq!(tree.size(), 1);

        let leaf_view =
            <VecOfArrays<f32, (), 3, 32> as LeafStrategy<f32, (), Eytzinger, 3, 32>>::leaf_view(
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
    fn create_multiple_leaf_vec_of_arrays_float_kd_tree() {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);

        // create 2^16 random 3d points in the unit cube
        let mut points: Vec<[f32; 3]> = vec![];
        for _ in 0..65_536 {
            let x = rng.random_range(0.0..1.0);
            let y = rng.random_range(0.0..1.0);
            let z = rng.random_range(0.0..1.0);
            points.push([x, y, z]);
        }

        let tree: kd_tree::KdTree<f32, u32, Eytzinger, VecOfArrays<f32, u32, 3, 32>, 3, 32> =
            kd_tree::KdTree::new_from_slice(&points).unwrap();

        assert!(!tree.is_empty());
        assert_eq!(tree.size(), 65_536);
        // Hard-bucket construction may need deeper splits to keep all leaves <= B.
        // That can increase leaf count above the nominal ceil(size / B) value.
        assert!(tree.leaf_count() >= 2048);
        assert!(tree.max_stem_level() >= 10);

        // perform a nearest_one query
        let query_point = [0.5, 0.5, 0.5];

        let _nearest = tree
            .query(&query_point)
            .nearest_one::<SquaredEuclidean<f32>>()
            .execute();
    }

    #[test]
    fn vec_of_arrays_size_and_leaf_len_track_contents() {
        let mut leaves = <VecOfArrays<f32, u32, 2, 4> as ConstructibleLeafStrategy<
            f32,
            u32,
            Eytzinger,
            2,
            4,
        >>::new_with_empty_leaf();

        assert_eq!(
            <VecOfArrays<f32, u32, 2, 4> as LeafStrategy<f32, u32, Eytzinger, 2, 4>>::size(&leaves),
            0
        );
        assert_eq!(
            <VecOfArrays<f32, u32, 2, 4> as LeafStrategy<f32, u32, Eytzinger, 2, 4>>::leaf_len(
                &leaves, 0
            ),
            0
        );

        <VecOfArrays<f32, u32, 2, 4> as MutableLeafStrategy<f32, u32, Eytzinger, 2, 4>>::add_to_leaf(&mut leaves, 0, &[1.0, 10.0], 7);
        <VecOfArrays<f32, u32, 2, 4> as MutableLeafStrategy<f32, u32, Eytzinger, 2, 4>>::add_to_leaf(&mut leaves, 0, &[2.0, 20.0], 8);

        let extra_points = [&[3.0f32, 4.0][..], &[30.0f32, 40.0][..]];
        let extra_items = [9u32, 10];
        <VecOfArrays<f32, u32, 2, 4> as ConstructibleLeafStrategy<
            f32,
            u32,
            Eytzinger,
            2,
            4,
        >>::append_leaf(&mut leaves, &extra_points, &extra_items);

        assert_eq!(
            <VecOfArrays<f32, u32, 2, 4> as LeafStrategy<f32, u32, Eytzinger, 2, 4>>::size(&leaves),
            4
        );
        assert_eq!(
            <VecOfArrays<f32, u32, 2, 4> as LeafStrategy<f32, u32, Eytzinger, 2, 4>>::leaf_len(
                &leaves, 0
            ),
            2
        );
        assert_eq!(
            <VecOfArrays<f32, u32, 2, 4> as LeafStrategy<f32, u32, Eytzinger, 2, 4>>::leaf_len(
                &leaves, 1
            ),
            2
        );
    }

    #[cfg(feature = "rkyv_08")]
    #[test]
    fn archived_vec_of_arrays_accessors_round_trip() {
        type Leaves = VecOfArrays<f32, u32, 2, 4>;
        type ArchivedLeaves = ArchivedVecOfArrays<f32, u32, 2, 4>;

        let mut leaves =
            <Leaves as ConstructibleLeafStrategy<f32, u32, Eytzinger, 2, 4>>::new_with_empty_leaf();
        <Leaves as MutableLeafStrategy<f32, u32, Eytzinger, 2, 4>>::add_to_leaf(
            &mut leaves,
            0,
            &[1.0, 10.0],
            7,
        );
        <Leaves as MutableLeafStrategy<f32, u32, Eytzinger, 2, 4>>::add_to_leaf(
            &mut leaves,
            0,
            &[2.0, 20.0],
            8,
        );
        let extra_points = [&[3.0f32][..], &[30.0f32][..]];
        let extra_items = [9u32];
        <Leaves as ConstructibleLeafStrategy<f32, u32, Eytzinger, 2, 4>>::append_leaf(
            &mut leaves,
            &extra_points,
            &extra_items,
        );

        let bytes = rkyv_08::api::high::to_bytes_in::<_, rkyv_08::rancor::Error>(
            &leaves,
            rkyv_08::util::AlignedVec::<16>::new(),
        )
        .unwrap();
        let archived =
            rkyv_08::access::<ArchivedLeaves, rkyv_08::rancor::Error>(bytes.as_slice()).unwrap();

        assert_eq!(
            <ArchivedLeaves as LeafStrategy<f32, u32, Eytzinger, 2, 4>>::size(archived),
            3
        );
        assert_eq!(
            <ArchivedLeaves as LeafStrategy<f32, u32, Eytzinger, 2, 4>>::leaf_count(archived),
            2
        );
        assert_eq!(
            <ArchivedLeaves as LeafStrategy<f32, u32, Eytzinger, 2, 4>>::leaf_len(archived, 0),
            2
        );
        assert_eq!(
            <ArchivedLeaves as LeafStrategy<f32, u32, Eytzinger, 2, 4>>::leaf_len(archived, 1),
            1
        );

        let view =
            <ArchivedLeaves as LeafStrategy<f32, u32, Eytzinger, 2, 4>>::leaf_view(archived, 0);
        let (points, items) = view.into_parts();
        assert_eq!(points[0], &[1.0, 2.0]);
        assert_eq!(points[1], &[10.0, 20.0]);
        assert_eq!(items, &[7, 8]);
    }

    #[test]
    fn vec_of_arrays_should_remove_requires_exact_point_and_item_match() {
        let leaf = super::LeafNode {
            content_points: [[1.0f32, 2.0, 0.0, 0.0], [10.0f32, 20.0, 0.0, 0.0]],
            content_items: [5u32, 6, 0, 0],
            size: 2,
        };

        assert!(VecOfArrays::<f32, u32, 2, 4>::should_remove(
            &leaf,
            &[1.0, 10.0],
            5,
            0
        ));
        assert!(!VecOfArrays::<f32, u32, 2, 4>::should_remove(
            &leaf,
            &[1.0, 10.0],
            99,
            0
        ));
        assert!(!VecOfArrays::<f32, u32, 2, 4>::should_remove(
            &leaf,
            &[1.0, 99.0],
            5,
            0
        ));
    }

    #[test]
    fn vec_of_arrays_remove_from_leaf_removes_only_matching_entries_and_compacts() {
        let mut leaves = <VecOfArrays<f32, u32, 2, 4> as ConstructibleLeafStrategy<
            f32,
            u32,
            Eytzinger,
            2,
            4,
        >>::new_with_empty_leaf();
        <VecOfArrays<f32, u32, 2, 4> as MutableLeafStrategy<f32, u32, Eytzinger, 2, 4>>::add_to_leaf(&mut leaves, 0, &[1.0, 10.0], 5);
        <VecOfArrays<f32, u32, 2, 4> as MutableLeafStrategy<f32, u32, Eytzinger, 2, 4>>::add_to_leaf(&mut leaves, 0, &[2.0, 20.0], 6);
        <VecOfArrays<f32, u32, 2, 4> as MutableLeafStrategy<f32, u32, Eytzinger, 2, 4>>::add_to_leaf(&mut leaves, 0, &[1.0, 10.0], 7);

        <VecOfArrays<f32, u32, 2, 4> as MutableLeafStrategy<f32, u32, Eytzinger, 2, 4>>::remove_from_leaf(&mut leaves, 0, &[1.0, 10.0], 5);

        assert_eq!(leaves.size, 2);
        assert_eq!(leaves.leaves[0].size, 2);
        assert_eq!(leaves.leaves[0].content_points[0][0], 2.0);
        assert_eq!(leaves.leaves[0].content_points[1][0], 20.0);
        assert_eq!(leaves.leaves[0].content_items[0], 6);
        assert_eq!(leaves.leaves[0].content_points[0][1], 1.0);
        assert_eq!(leaves.leaves[0].content_points[1][1], 10.0);
        assert_eq!(leaves.leaves[0].content_items[1], 7);
    }

    #[test]
    fn vec_of_arrays_replace_item_in_leaf_replaces_only_first_exact_match() {
        let mut leaves = <VecOfArrays<f32, u32, 2, 4> as ConstructibleLeafStrategy<
            f32,
            u32,
            Eytzinger,
            2,
            4,
        >>::new_with_empty_leaf();
        <VecOfArrays<f32, u32, 2, 4> as MutableLeafStrategy<f32, u32, Eytzinger, 2, 4>>::add_to_leaf(&mut leaves, 0, &[1.0, 10.0], 5);
        <VecOfArrays<f32, u32, 2, 4> as MutableLeafStrategy<f32, u32, Eytzinger, 2, 4>>::add_to_leaf(&mut leaves, 0, &[1.0, 10.0], 5);
        <VecOfArrays<f32, u32, 2, 4> as MutableLeafStrategy<f32, u32, Eytzinger, 2, 4>>::add_to_leaf(&mut leaves, 0, &[2.0, 20.0], 6);

        assert!(<VecOfArrays<f32, u32, 2, 4> as LeafStrategy<
            f32,
            u32,
            Eytzinger,
            2,
            4,
        >>::replace_item_in_leaf(
            &mut leaves, 0, &[1.0, 10.0], 5, 9
        ));

        assert_eq!(leaves.leaves[0].content_items[0], 9);
        assert_eq!(leaves.leaves[0].content_items[1], 5);
        assert_eq!(leaves.leaves[0].content_items[2], 6);
        assert_eq!(leaves.size, 3);
        assert_eq!(leaves.leaves[0].size, 3);
    }

    #[test]
    fn vec_of_arrays_split_leaf_handles_pivot_idx_zero_by_scanning_right() {
        let mut leaves = <VecOfArrays<f32, u32, 2, 4> as ConstructibleLeafStrategy<
            f32,
            u32,
            Eytzinger,
            2,
            4,
        >>::new_with_empty_leaf();
        <VecOfArrays<f32, u32, 2, 4> as MutableLeafStrategy<f32, u32, Eytzinger, 2, 4>>::add_to_leaf(&mut leaves, 0, &[2.0, 200.0], 10);
        <VecOfArrays<f32, u32, 2, 4> as MutableLeafStrategy<f32, u32, Eytzinger, 2, 4>>::add_to_leaf(&mut leaves, 0, &[2.0, 201.0], 11);
        <VecOfArrays<f32, u32, 2, 4> as MutableLeafStrategy<f32, u32, Eytzinger, 2, 4>>::add_to_leaf(&mut leaves, 0, &[2.0, 202.0], 12);
        <VecOfArrays<f32, u32, 2, 4> as MutableLeafStrategy<f32, u32, Eytzinger, 2, 4>>::add_to_leaf(&mut leaves, 0, &[4.0, 400.0], 13);

        let (split_val, new_leaf_idx) = <VecOfArrays<f32, u32, 2, 4> as MutableLeafStrategy<
            f32,
            u32,
            Eytzinger,
            2,
            4,
        >>::split_leaf(&mut leaves, 0, 0)
        .unwrap();

        assert_eq!(split_val, 4.0);
        assert_eq!(new_leaf_idx, 1);
        assert_eq!(leaves.leaves[0].size, 3);
        assert_eq!(leaves.leaves[1].size, 1);

        let left_view =
            <VecOfArrays<f32, u32, 2, 4> as LeafStrategy<f32, u32, Eytzinger, 2, 4>>::leaf_view(
                &leaves, 0,
            );
        let right_view =
            <VecOfArrays<f32, u32, 2, 4> as LeafStrategy<f32, u32, Eytzinger, 2, 4>>::leaf_view(
                &leaves, 1,
            );
        let (left_points, left_items) = left_view.into_parts();
        let (right_points, right_items) = right_view.into_parts();

        assert_eq!(left_points[0], &[2.0, 2.0, 2.0]);
        let mut left_items_sorted = left_items.to_vec();
        left_items_sorted.sort_unstable();
        assert_eq!(left_items_sorted, vec![10, 11, 12]);
        assert_eq!(right_points[0], &[4.0]);
        assert_eq!(right_points[1], &[400.0]);
        assert_eq!(right_items, &[13]);

        let mut item_points = left_items
            .iter()
            .enumerate()
            .map(|(idx, item)| (*item, [left_points[0][idx], left_points[1][idx]]))
            .chain(
                right_items
                    .iter()
                    .enumerate()
                    .map(|(idx, item)| (*item, [right_points[0][idx], right_points[1][idx]])),
            )
            .collect::<Vec<_>>();
        item_points.sort_unstable_by_key(|(item, _)| *item);
        assert_eq!(
            item_points,
            vec![
                (10, [2.0, 200.0]),
                (11, [2.0, 201.0]),
                (12, [2.0, 202.0]),
                (13, [4.0, 400.0]),
            ]
        );
    }
}

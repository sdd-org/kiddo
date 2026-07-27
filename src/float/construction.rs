use crate::float::kdtree::{Axis, KdTree, LeafNode, StemNode};
use crate::mirror_select_nth_unstable_by::mirror_select_nth_unstable_by;
use crate::traits::{is_stem_index, Content, Index};
use az::{Az, Cast};
use std::ops::Rem;

impl<A: Axis, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>>
    KdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
{
    /// Adds an item to the tree.
    ///
    /// The first argument specifies co-ordinates of the point where the item is located.
    /// The second argument is an integer identifier / index for the item being stored.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    ///
    /// let mut tree: KdTree<f64, 3> = KdTree::new();
    ///
    /// tree.add(&[1.0, 2.0, 5.0], 100);
    ///
    /// assert_eq!(tree.size(), 1);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if more than `B` items are added at exactly the same point. Mutable v5 trees use
    /// fixed-size leaf buckets, and identical points cannot be separated by a spatial split.
    #[inline]
    pub fn add(&mut self, query: &[A; K], item: T) {
        unsafe {
            let mut stem_idx = self.root_index;
            let mut split_dim = 0;
            let mut stem_node;
            let mut parent_idx = <IDX as Index>::max();
            let mut is_left_child: bool = false;

            while is_stem_index(stem_idx) {
                parent_idx = stem_idx;
                stem_node = self.stems.get_unchecked_mut(stem_idx.az::<usize>());

                stem_idx = if *query.get_unchecked(split_dim) < stem_node.split_val {
                    is_left_child = true;
                    stem_node.left
                } else {
                    is_left_child = false;
                    stem_node.right
                };

                split_dim = (split_dim + 1).rem(K);
            }

            let mut leaf_idx = stem_idx - IDX::leaf_offset();
            let mut leaf_node = self.leaves.get_unchecked_mut(leaf_idx.az::<usize>());

            while leaf_node.size == B.az::<IDX>() {
                stem_idx = self.split(leaf_idx, split_dim, parent_idx, is_left_child, query);
                let node = self.stems.get_unchecked_mut(stem_idx.az::<usize>());

                let child_idx = if *query.get_unchecked(split_dim) < node.split_val {
                    is_left_child = true;
                    node.left
                } else {
                    is_left_child = false;
                    node.right
                };

                parent_idx = stem_idx;
                split_dim = (split_dim + 1).rem(K);
                leaf_idx = child_idx - IDX::leaf_offset();
                leaf_node = self.leaves.get_unchecked_mut(leaf_idx.az::<usize>());
            }

            *leaf_node
                .content_points
                .get_unchecked_mut(leaf_node.size.az::<usize>()) = *query;
            *leaf_node
                .content_items
                .get_unchecked_mut(leaf_node.size.az::<usize>()) = item;

            leaf_node.size = leaf_node.size + IDX::one();
        }
        self.size = self.size + T::one();
    }

    /// Removes an item from the tree.
    ///
    /// The first argument specifies co-ordinates of the point where the item is located.
    /// The second argument is the integer identifier / index for the stored item.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    ///
    /// let mut tree: KdTree<f64, 3> = KdTree::new();
    ///
    /// tree.add(&[1.0, 2.0, 5.0], 100);
    /// tree.add(&[1.0, 2.0, 5.0], 200);
    /// assert_eq!(tree.size(), 2);
    ///
    /// tree.remove(&[1.0, 2.0, 5.0], 100);
    /// assert_eq!(tree.size(), 1);
    ///
    /// tree.remove(&[1.0, 2.0, 5.0], 200);
    /// assert_eq!(tree.size(), 0);
    /// ```
    #[inline]
    pub fn remove(&mut self, query: &[A; K], item: T) -> usize {
        let mut stem_idx = self.root_index;
        let mut split_dim = 0;
        let mut removed: usize = 0;

        while is_stem_index(stem_idx) {
            let Some(stem_node) = self.stems.get_mut(stem_idx.az::<usize>()) else {
                return removed;
            };

            stem_idx = if query[split_dim] < stem_node.split_val {
                stem_node.left
            } else {
                stem_node.right
            };

            split_dim = (split_dim + 1).rem(K);
        }

        let leaf_idx = stem_idx - IDX::leaf_offset();

        if let Some(leaf_node) = self.leaves.get_mut(leaf_idx.az::<usize>()) {
            let mut p_index = 0;
            while p_index < leaf_node.size.az::<usize>() {
                if &leaf_node.content_points[p_index] == query
                    && leaf_node.content_items[p_index] == item
                {
                    leaf_node.content_points[p_index] =
                        leaf_node.content_points[leaf_node.size.az::<usize>() - 1];
                    leaf_node.content_items[p_index] =
                        leaf_node.content_items[leaf_node.size.az::<usize>() - 1];

                    self.size -= T::one();
                    removed += 1;
                    leaf_node.size = leaf_node.size - IDX::one();
                } else {
                    p_index += 1;
                }
            }
        }

        removed
    }

    unsafe fn split(
        &mut self,
        leaf_idx: IDX,
        split_dim: usize,
        parent_idx: IDX,
        was_parents_left: bool,
        query: &[A; K],
    ) -> IDX {
        let orig = self.leaves.get_unchecked_mut(leaf_idx.az::<usize>());
        let mut pivot_idx = (B / 2).az::<IDX>();

        mirror_select_nth_unstable_by(
            &mut orig.content_points,
            &mut orig.content_items,
            pivot_idx.az::<usize>(),
            |a, b| unsafe {
                a.get_unchecked(split_dim)
                    .partial_cmp(b.get_unchecked(split_dim))
                    .expect("Leaf node sort failed.")
            },
        );

        let mut split_val = *orig
            .content_points
            .get_unchecked(pivot_idx.az::<usize>())
            .get_unchecked(split_dim);

        // if the chosen pivot point would result in some items whose position on the split
        // dimension is the same as the split value being on the wrong side of the split:
        if *orig
            .content_points
            .get_unchecked(pivot_idx.az::<usize>() - 1)
            .get_unchecked(split_dim)
            == split_val
        {
            let orig_pivot_idx = pivot_idx;

            // Ensure that if pivot index would result in items that share the same co-ordinate
            // on the splitting dimension would end up on different sides of the split, that we
            // move the pivot to prevent this. We first try moving down, since that's the only
            // part of the bucket that was sorted by mirror_select_nth_unstable_by
            while pivot_idx > IDX::zero()
                && *orig
                    .content_points
                    .get_unchecked(pivot_idx.az::<usize>() - 1)
                    .get_unchecked(split_dim)
                    == split_val
            {
                pivot_idx = pivot_idx - IDX::one();
            }

            // If the attempt to move the pivot point above would have resulted in the pivot
            // point moving to the start of the bucket, search forwards from the original
            // pivot point instead. We need to first ensure the upper half of the bucket
            // is sorted
            if pivot_idx == IDX::zero() {
                mirror_select_nth_unstable_by(
                    &mut orig.content_points,
                    &mut orig.content_items,
                    B - 1,
                    |a, b| unsafe {
                        a.get_unchecked(split_dim)
                            .partial_cmp(b.get_unchecked(split_dim))
                            .expect("Leaf node sort failed.")
                    },
                );

                pivot_idx = orig_pivot_idx;
                let mut all_equal = false;
                while *orig
                    .content_points
                    .get_unchecked(pivot_idx.az::<usize>())
                    .get_unchecked(split_dim)
                    == split_val
                {
                    pivot_idx = pivot_idx + IDX::one();

                    if pivot_idx.az::<usize>() == B {
                        all_equal = true;
                        break;
                    }
                }

                if all_equal {
                    let query_val = *query.get_unchecked(split_dim);

                    if query_val > split_val {
                        // Keep the full original leaf on the left and create an empty right leaf
                        // for the new point.
                        split_val = query_val;
                    } else {
                        if query_val == split_val
                            && orig.content_points.iter().all(|point| point == query)
                        {
                            panic!(
                                "Cannot insert another item at {query:?}: this leaf already contains \
                                 {B} items at exactly the same point. Kiddo v5's fixed-size leaf \
                                 buckets cannot spatially split points whose coordinates are \
                                 identical on every axis. Increase the bucket size `B` above the \
                                 maximum duplicate count, deduplicate the input, or bulk-build a \
                                 Kiddo v6 `ImmutableKdTree` with soft leaf buckets."
                            );
                        }

                        // Put the full original leaf on the right. If the new point has the same
                        // coordinate, add() will continue to the next split dimension.
                        pivot_idx = IDX::zero();
                    }
                } else {
                    split_val = *orig
                        .content_points
                        .get_unchecked(pivot_idx.az::<usize>())
                        .get_unchecked(split_dim);
                }
            } else {
                split_val = *orig
                    .content_points
                    .get_unchecked(pivot_idx.az::<usize>())
                    .get_unchecked(split_dim);
            }
        }

        let mut right = LeafNode::new();
        orig.size = pivot_idx;
        let dest_slice_end = B - pivot_idx.az::<usize>();

        right
            .content_points
            .get_unchecked_mut(..dest_slice_end)
            .copy_from_slice(
                orig.content_points
                    .get_unchecked((pivot_idx.az::<usize>())..),
            );
        right
            .content_items
            .get_unchecked_mut(..dest_slice_end)
            .copy_from_slice(
                orig.content_items
                    .get_unchecked((pivot_idx.az::<usize>())..),
            );

        right.size = (B.az::<IDX>()) - pivot_idx;

        self.leaves.push(right);

        self.stems.push(StemNode {
            left: leaf_idx + IDX::leaf_offset(),
            right: (self.leaves.len().az::<IDX>()) - IDX::one() + IDX::leaf_offset(),
            split_val,
        });
        let new_stem_index: IDX = (self.stems.len().az::<IDX>()) - IDX::one();

        if parent_idx != <IDX as Index>::max() {
            let parent_node = self.stems.get_unchecked_mut(parent_idx.az::<usize>());
            if was_parents_left {
                parent_node.left = new_stem_index;
            } else {
                parent_node.right = new_stem_index;
            }
        } else {
            self.root_index = new_stem_index;
        }

        new_stem_index
    }
}

impl<A: Axis, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>>
    FromIterator<([A; K], T)> for KdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
{
    #[inline]
    fn from_iter<I: IntoIterator<Item = ([A; K], T)>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut tree = Self::with_capacity(iter.size_hint().0);
        for (point, item) in iter {
            tree.add(&point, item);
        }
        tree
    }
}

impl<A: Axis, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>, const N: usize>
    From<[([A; K], T); N]> for KdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
{
    #[inline]
    fn from(value: [([A; K], T); N]) -> Self {
        value.into_iter().collect()
    }
}

impl<A: Axis, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>> Extend<([A; K], T)>
    for KdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
{
    #[inline]
    fn extend<I: IntoIterator<Item = ([A; K], T)>>(&mut self, iter: I) {
        for (point, item) in iter {
            self.add(&point, item);
        }
    }
}

impl<
        'a,
        't,
        A: Axis + Copy,
        T: Content + Copy,
        const K: usize,
        const B: usize,
        IDX: Index<T = IDX>,
    > Extend<(&'a [A; K], &'t T)> for KdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
{
    #[inline]
    fn extend<I: IntoIterator<Item = (&'a [A; K], &'t T)>>(&mut self, iter: I) {
        for (point, item) in iter.into_iter() {
            self.add(point, *item);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::float::kdtree::KdTree;
    use rand::RngExt;

    type Flt = f32;

    fn n(num: Flt) -> Flt {
        num
    }

    #[test]
    fn can_add_an_item() {
        let mut tree: KdTree<Flt, u32, 4, 32, u32> = KdTree::new();

        let point: [Flt; 4] = [n(0.1f32), n(0.2f32), n(0.3f32), n(0.4f32)];
        let item = 123;

        tree.add(&point, item);

        assert_eq!(tree.size(), 1);
    }

    #[test]
    fn can_add_enough_items_to_cause_a_split() {
        let mut tree: KdTree<Flt, u32, 4, 4, u32> = KdTree::new();

        let content_to_add: [([Flt; 4], u32); 16] = [
            ([n(0.9f32), n(0.0f32), n(0.9f32), n(0.0f32)], 9),
            ([n(0.4f32), n(0.5f32), n(0.4f32), n(0.5f32)], 4),
            ([n(0.12f32), n(0.3f32), n(0.12f32), n(0.3f32)], 12),
            ([n(0.7f32), n(0.2f32), n(0.7f32), n(0.2f32)], 7),
            ([n(0.13f32), n(0.4f32), n(0.13f32), n(0.4f32)], 13),
            ([n(0.6f32), n(0.3f32), n(0.6f32), n(0.3f32)], 6),
            ([n(0.2f32), n(0.7f32), n(0.2f32), n(0.7f32)], 2),
            ([n(0.14f32), n(0.5f32), n(0.14f32), n(0.5f32)], 14),
            ([n(0.3f32), n(0.6f32), n(0.3f32), n(0.6f32)], 3),
            ([n(0.10f32), n(0.1f32), n(0.10f32), n(0.1f32)], 10),
            ([n(0.16f32), n(0.7f32), n(0.16f32), n(0.7f32)], 16),
            ([n(0.1f32), n(0.8f32), n(0.1f32), n(0.8f32)], 1),
            ([n(0.15f32), n(0.6f32), n(0.15f32), n(0.6f32)], 15),
            ([n(0.5f32), n(0.4f32), n(0.5f32), n(0.4f32)], 5),
            ([n(0.8f32), n(0.1f32), n(0.8f32), n(0.1f32)], 8),
            ([n(0.11f32), n(0.2f32), n(0.11f32), n(0.2f32)], 11),
        ];

        for (point, item) in content_to_add {
            tree.add(&point, item);
        }

        assert_eq!(tree.size(), 16);
    }

    #[test]
    fn can_remove_an_item() {
        let mut tree: KdTree<Flt, u32, 4, 4, u32> = KdTree::new();

        let content_to_add: [([Flt; 4], u32); 16] = [
            ([n(0.9f32), n(0.0f32), n(0.9f32), n(0.0f32)], 9),
            ([n(0.4f32), n(0.5f32), n(0.4f32), n(0.5f32)], 4),
            ([n(0.12f32), n(0.3f32), n(0.12f32), n(0.3f32)], 12),
            ([n(0.7f32), n(0.2f32), n(0.7f32), n(0.2f32)], 7),
            ([n(0.13f32), n(0.4f32), n(0.13f32), n(0.4f32)], 13),
            ([n(0.6f32), n(0.3f32), n(0.6f32), n(0.3f32)], 6),
            ([n(0.2f32), n(0.7f32), n(0.2f32), n(0.7f32)], 2),
            ([n(0.14f32), n(0.5f32), n(0.14f32), n(0.5f32)], 14),
            ([n(0.3f32), n(0.6f32), n(0.3f32), n(0.6f32)], 3),
            ([n(0.10f32), n(0.1f32), n(0.10f32), n(0.1f32)], 10),
            ([n(0.16f32), n(0.7f32), n(0.16f32), n(0.7f32)], 16),
            ([n(0.1f32), n(0.8f32), n(0.1f32), n(0.8f32)], 1),
            ([n(0.15f32), n(0.6f32), n(0.15f32), n(0.6f32)], 15),
            ([n(0.5f32), n(0.4f32), n(0.5f32), n(0.4f32)], 5),
            ([n(0.8f32), n(0.1f32), n(0.8f32), n(0.1f32)], 8),
            ([n(0.11f32), n(0.2f32), n(0.11f32), n(0.2f32)], 11),
        ];

        for (point, item) in content_to_add {
            tree.add(&point, item);
        }

        assert_eq!(tree.size(), 16);

        let removed = tree.remove(&[n(0.9f32), n(0.0f32), n(0.9f32), n(0.0f32)], 9);

        assert_eq!(removed, 1);
        assert_eq!(tree.size(), 15);
    }

    #[test]
    fn can_add_shitloads_of_points() {
        let mut tree: KdTree<Flt, u32, 4, 5, u32> = KdTree::new();

        let mut rng = rand::rng();
        for i in 0..1000 {
            let point = [
                n(rng.random_range(0f32..0.99998f32)),
                n(rng.random_range(0f32..0.99998f32)),
                n(rng.random_range(0f32..0.99998f32)),
                n(rng.random_range(0f32..0.99998f32)),
            ];

            tree.add(&point, i);
        }

        assert_eq!(tree.size(), 1000);
    }

    #[test]
    fn can_add_shitloads_of_random_points() {
        fn rand_data_2d() -> ([f64; 2], u32) {
            rand::random()
        }

        let points_to_add: Vec<([f64; 2], u32)> = (0..100_000).map(|_| rand_data_2d()).collect();

        let mut points = vec![];
        let mut kdtree = KdTree::<f64, u32, 2, 32, u32>::with_capacity(200_000);
        for _ in 0..100_000 {
            points.push(rand_data_2d());
        }
        for point in &points {
            kdtree.add(&point.0, point.1);
        }

        points_to_add
            .iter()
            .for_each(|point| kdtree.add(&point.0, point.1));
    }

    #[test]
    fn test_can_handle_remove_edge_cases_from_issues_12_and_28() {
        // See: https://github.com/sdd/kiddo/issues/12
        //      https://github.com/sdd/kiddo/issues/28
        let pts = vec![
            [19.2023, 7.1812], // 0: same x val as point 21
            [7.6427, 22.5779],
            [26.6314, 34.8920],
            [36.7890, 27.2663],
            [28.3226, 8.5047],
            [5.3914, 28.1360],
            [5.0978, 3.6814],
            [0.5114, 11.6552],
            [4.7981, 21.6210],
            [29.0030, 9.6799],
            [35.5580, 1.8891],
            [3.9160, 25.5702],
            [22.2497, 31.6140],
            [30.7110, 36.7514],
            [0.2828, 12.4298],
            [20.0206, 3.0635],
            [30.6153, 2.8582],
            [23.7179, 6.2048],
            [13.0438, 4.2319],
            [4.6433, 30.9660],
            [5.0588, 5.2028],
            [19.2023, 23.7406], // 21: same x val as point 0
            [37.3171, 32.7523],
            [12.6957, 15.7080],
            [15.6001, 14.3995],
            [36.0203, 21.0366],
            [6.3956, 2.7644],
            [3.1719, 8.7039],
            [0.9159, 12.2299],
            [23.8157, 14.0699],
            [27.7757, 7.3597],
            [28.4198, 31.3427],
            [2.3290, 6.2364],
            [10.1126, 7.7009],
        ];

        let mut tree = KdTree::<f64, usize, 2, 32, u32>::new();

        for (i, pt) in pts.iter().enumerate() {
            tree.add(pt, i);
        }

        for (i, pt) in pts.iter().enumerate() {
            assert_eq!(tree.remove(pt, i), 1, "failed to remove point {i}");
        }
    }

    #[test]
    fn test_can_handle_split_that_needs_pivot_to_move_towards_end_of_bucket() {
        let pts = [
            [19.2023, 6.2364],
            [19.2023, 7.1812],
            [19.2023, 7.1812],
            [19.2023, 7.1812],
            [19.2023, 7.1812],
            [19.2023, 7.1812],
            [19.2023, 7.1812],
            [36.7890, 27.2663],
            [28.3226, 8.5047],
            [26.6314, 34.8920],
            [29.0030, 9.6799],
            [35.5580, 1.8891],
        ];

        let mut tree = KdTree::<f64, usize, 2, 10, u32>::new();

        for (i, pt) in pts.iter().enumerate() {
            tree.add(pt, i);
        }

        for (i, pt) in pts.iter().enumerate() {
            assert_eq!(tree.remove(pt, i), 1, "failed to remove point {i}");
        }
    }
}

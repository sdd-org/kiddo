use aligned_vec::{AVec, CACHELINE_ALIGN};

use crate::leaf_view::{LeafArena, LeafView};
use crate::traits::leaf_strategy::{
    BucketLimitType, ConstructibleLeafStrategy, Immutable, LeafProjection,
};
use crate::{Axis, Content, LeafStrategy, StemStrategy};

/// Immutable leaf storage using chunk-tiled arenas encoded into a single byte buffer.
///
/// All leaves are encoded into one contiguous `leaf_bytes` arena. `leaf_extents`
/// stores `(byte_offset, leaf_len)` for each logical leaf. Within a leaf, data is
/// written as descending tile widths (`32, 8, 4, 2, 1`), with each tile laid out
/// column-major by axis followed immediately by that tile's items.
///
/// Memory layout:
///
/// ```text
/// leaf_extents = [(off0, len0), (off1, len1), ...]
///
/// leaf_bytes =
///   [ leaf 0 tile32: x[0..32]  y[0..32]  z[0..32]  items[0..32] ]
///   [ leaf 0 tile8 : x[32..40] y[32..40] z[32..40] items[32..40] ]
///   [ leaf 0 tile4 : x[40..44] y[40..44] z[40..44] items[40..44] ]
///   [ leaf 0 tile2 : x[44..46] y[44..46] z[44..46] items[44..46] ]
///   [ leaf 0 tile1 : x[46]     y[46]     z[46]     item[46]      ]
///   [ leaf 1 tile32: ... ]
///   ...
/// ```
///
///
/// The main advantage over [`FlatVec`](crate::leaf_strategies::FlatVec) is not
/// just that the leaf is self-contained. The data is laid out in the same order
/// that the SIMD/autovec leaf kernel consumes it:
///
/// 1. stream one tile's `x` coordinates
/// 2. then the same tile's `y`, `z`, ...
/// 3. then the matching tile's items
///
/// That means the CPU can usually service the whole leaf with a single forward
/// prefetch stream. By contrast, `FlatVec` pulls from several separate vectors
/// (`K` coordinate arrays plus the item array), which is still vector-friendly
/// but asks more of the hardware prefetcher's limited stream-tracking resources.
///
/// So `VecOfArenas` aims to keep the same good column-major compute shape while
/// also matching the leaf kernel's access order more closely.
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
        deserialize = "A: serde::Deserialize<'de>, T: serde::Deserialize<'de>"
    ))
)]
pub struct VecOfArenas<A, T, const K: usize, const B: usize> {
    leaf_extents: Vec<(usize, usize)>,
    #[cfg_attr(
        feature = "rkyv_08",
        rkyv(with = crate::rkyv::adapters::AsAlignedCachelineABox)
    )]
    leaf_bytes: AVec<u8>,
    size: usize,
    _phantom: std::marker::PhantomData<(A, T)>,
}

#[cfg(feature = "test_utils")]
impl<A: Copy, T: Copy, const K: usize, const B: usize> VecOfArenas<A, T, K, B> {
    #[inline(always)]
    pub(crate) fn leaf_extent_for_embedded_descriptor(&self, leaf_idx: usize) -> (usize, usize) {
        debug_assert!(leaf_idx < self.leaf_extents.len());
        unsafe { *self.leaf_extents.get_unchecked(leaf_idx) }
    }

    #[inline(always)]
    pub(crate) fn leaf_arena_from_embedded_descriptor(
        &self,
        byte_offset: usize,
        leaf_len: usize,
    ) -> LeafArena<'_, A, T, K> {
        #[cfg(debug_assertions)]
        {
            debug_assert!(
                byte_offset + LeafArena::<A, T, K>::encoded_len_bytes(leaf_len)
                    <= self.leaf_bytes.len()
            );
        }
        LeafArena::new(
            unsafe { self.leaf_bytes.as_ptr().add(byte_offset) },
            leaf_len,
        )
    }
}

#[cfg(feature = "rkyv_08")]
impl<A, T, const K: usize, const B: usize> ArchivedVecOfArenas<A, T, K, B>
where
    A: rkyv_08::Archive + Copy,
    T: rkyv_08::Archive + Copy,
{
    /// Returns the number of items in a leaf.
    #[inline]
    pub fn leaf_len(&self, leaf_idx: usize) -> usize {
        self.leaf_extents[leaf_idx].1.to_native() as usize
    }

    /// Returns an arena-backed view over a leaf.
    #[inline]
    pub fn leaf_arena(&self, leaf_idx: usize) -> LeafArena<'_, A, T, K> {
        let extent = &self.leaf_extents[leaf_idx];
        let offset = extent.0.to_native() as usize;
        let len = extent.1.to_native() as usize;
        let bytes = self.leaf_bytes.get().as_slice();
        LeafArena::new(unsafe { bytes.as_ptr().add(offset) }, len)
    }
}

#[cfg(feature = "rkyv_08")]
impl<AX, T, SS, const K: usize, const B: usize> LeafStrategy<AX, T, SS, K, B>
    for ArchivedVecOfArenas<AX, T, K, B>
where
    AX: rkyv_08::Archive + Axis<Coord = AX>,
    T: rkyv_08::Archive + Content,
    SS: StemStrategy,
{
    type Num = AX;
    type Mutability = Immutable;

    const BUCKET_LIMIT_TYPE: BucketLimitType = BucketLimitType::Soft;
    const LEAF_PROJECTION: LeafProjection = LeafProjection::LeafArena;

    fn size(&self) -> usize {
        self.size.to_native() as usize
    }

    fn leaf_count(&self) -> usize {
        self.leaf_extents.len()
    }

    fn leaf_len(&self, leaf_idx: usize) -> usize {
        ArchivedVecOfArenas::leaf_len(self, leaf_idx)
    }

    fn leaf_view(&self, _leaf_idx: usize) -> LeafView<'_, AX, T, K, B> {
        unimplemented!("VecOfArenas currently exposes only arena-backed hot paths")
    }

    fn leaf_arena(&self, leaf_idx: usize) -> LeafArena<'_, AX, T, K> {
        ArchivedVecOfArenas::leaf_arena(self, leaf_idx)
    }
}

#[cfg(test)]
#[inline(always)]
fn extend_bytes_from_slice<U: Copy>(dst: &mut Vec<u8>, src: &[U]) {
    let byte_len = std::mem::size_of_val(src);
    let start = dst.len();
    dst.resize(start + byte_len, 0u8);

    unsafe {
        std::ptr::copy_nonoverlapping(
            src.as_ptr() as *const u8,
            dst.as_mut_ptr().add(start),
            byte_len,
        );
    }
}

#[inline(always)]
fn extend_avec_bytes_from_slice<U: Copy>(dst: &mut AVec<u8>, src: &[U]) {
    let byte_len = std::mem::size_of_val(src);
    let start = dst.len();
    dst.resize(start + byte_len, 0u8);

    unsafe {
        std::ptr::copy_nonoverlapping(
            src.as_ptr() as *const u8,
            dst.as_mut_ptr().add(start),
            byte_len,
        );
    }
}

#[inline(always)]
fn align_up(value: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (value + align - 1) & !(align - 1)
}

impl<AX, T, SS, const K: usize, const B: usize> LeafStrategy<AX, T, SS, K, B>
    for VecOfArenas<AX, T, K, B>
where
    AX: Axis<Coord = AX>,
    T: Content,
    SS: StemStrategy,
{
    type Num = AX;
    type Mutability = Immutable;

    const BUCKET_LIMIT_TYPE: BucketLimitType = BucketLimitType::Soft;
    const LEAF_PROJECTION: LeafProjection = LeafProjection::LeafArena;

    fn size(&self) -> usize {
        self.size
    }

    fn leaf_count(&self) -> usize {
        self.leaf_extents.len()
    }

    fn leaf_len(&self, leaf_idx: usize) -> usize {
        debug_assert!(leaf_idx < self.leaf_extents.len());
        unsafe { self.leaf_extents.get_unchecked(leaf_idx).1 }
    }

    fn leaf_view(&self, _leaf_idx: usize) -> LeafView<'_, AX, T, K, B> {
        unimplemented!("VecOfArenas currently exposes only arena-backed hot paths")
    }

    fn leaf_arena(&self, leaf_idx: usize) -> LeafArena<'_, AX, T, K> {
        debug_assert!(leaf_idx < self.leaf_extents.len());
        let (offset, len) = unsafe { *self.leaf_extents.get_unchecked(leaf_idx) };
        #[cfg(debug_assertions)]
        {
            let byte_len = LeafArena::<AX, T, K>::encoded_len_bytes(len);
            debug_assert!(offset + byte_len <= self.leaf_bytes.len());
        }

        LeafArena::new(unsafe { self.leaf_bytes.as_ptr().add(offset) }, len)
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
        debug_assert!(leaf_idx < self.leaf_extents.len());
        let (offset, len) = unsafe { *self.leaf_extents.get_unchecked(leaf_idx) };
        let mut byte_offset = offset;
        let mut remaining = len;

        while remaining != 0 {
            let tile_len = crate::leaf_view::leaf_arena_tile_len(remaining);
            let tile: LeafArena<'_, AX, T, K> = LeafArena::new(
                unsafe { self.leaf_bytes.as_ptr().add(byte_offset) },
                tile_len,
            );

            for tile_idx in 0..tile_len {
                let stored_point = tile.point_item(tile_idx).0;
                let point_matches = (0..K).all(|dim| stored_point[dim] == point[dim]);
                if point_matches {
                    let item_offset = byte_offset
                        + K * tile_len * std::mem::size_of::<AX>()
                        + tile_idx * std::mem::size_of::<T>();
                    let current_item = unsafe {
                        std::ptr::read_unaligned(
                            self.leaf_bytes.as_ptr().add(item_offset) as *const T
                        )
                    };

                    if current_item == old_item {
                        unsafe {
                            std::ptr::write_unaligned(
                                self.leaf_bytes.as_mut_ptr().add(item_offset) as *mut T,
                                new_item,
                            );
                        }
                        return true;
                    }
                }
            }

            byte_offset +=
                K * tile_len * std::mem::size_of::<AX>() + tile_len * std::mem::size_of::<T>();
            remaining -= tile_len;
        }

        false
    }

    #[inline]
    fn maybe_enable_huge_pages(&self) {
        crate::huge_pages::maybe_collapse_slice_huge_pages(
            self.leaf_extents.as_ptr(),
            self.leaf_extents.len(),
        );
        crate::huge_pages::maybe_collapse_slice_huge_pages(
            self.leaf_bytes.as_ptr(),
            self.leaf_bytes.len(),
        );
    }
}

impl<AX, T, SS, const K: usize, const B: usize> ConstructibleLeafStrategy<AX, T, SS, K, B>
    for VecOfArenas<AX, T, K, B>
where
    AX: Axis<Coord = AX>,
    T: Content,
    SS: StemStrategy,
{
    fn new_with_capacity(capacity: usize) -> Self {
        Self {
            leaf_extents: Vec::with_capacity(capacity / B + 1),
            leaf_bytes: AVec::new(CACHELINE_ALIGN),
            size: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    fn append_leaf(&mut self, leaf_points: &[&[AX]; K], leaf_items: &[T]) {
        let leaf_len = leaf_items.len();

        debug_assert!(leaf_points[0].len() == leaf_len);
        for dim in 1..K {
            debug_assert!(leaf_points[dim].len() == leaf_len);
        }

        let leaf_align = std::mem::align_of::<AX>()
            .max(std::mem::align_of::<T>())
            .max(std::mem::size_of::<u64>());
        let start_offset = align_up(self.leaf_bytes.len(), leaf_align);
        if start_offset != self.leaf_bytes.len() {
            self.leaf_bytes.resize(start_offset, 0u8);
        }
        self.leaf_extents.push((start_offset, leaf_len));

        let mut base = 0usize;
        crate::leaf_view::for_each_leaf_arena_tile_len(leaf_len, |tile_len| {
            for dim in 0..K {
                extend_avec_bytes_from_slice(
                    &mut self.leaf_bytes,
                    &leaf_points[dim][base..base + tile_len],
                );
            }
            extend_avec_bytes_from_slice(&mut self.leaf_bytes, &leaf_items[base..base + tile_len]);

            base += tile_len;
        });

        self.size += leaf_len;
    }
}

#[cfg(test)]
#[cfg(feature = "rkyv_08")]
impl<A, T, const K: usize, const B: usize> VecOfArenas<A, T, K, B> {
    pub(crate) fn leaf_bytes_ptr(&self) -> *const u8 {
        self.leaf_bytes.as_ptr()
    }
}

#[cfg(test)]
mod tests {
    use super::VecOfArenas;
    use crate::kd_tree;
    use crate::leaf_strategy::vec_of_arenas::extend_bytes_from_slice;
    use crate::leaf_view::LeafArena;
    use crate::traits::leaf_strategy::ConstructibleLeafStrategy;
    use crate::{Eytzinger, LeafStrategy};

    #[test]
    fn default_constructs_vec_of_arenas_kd_tree() {
        let tree: kd_tree::KdTree<f64, u32, Eytzinger, VecOfArenas<f64, u32, 3, 32>, 3, 32> =
            kd_tree::KdTree::default();

        assert!(tree.is_empty());
        assert_eq!(tree.size(), 0);
    }

    #[test]
    fn vec_of_arenas_appends_leafs_with_expected_extents() {
        let mut leaves = <VecOfArenas<f64, u32, 3, 32> as ConstructibleLeafStrategy<
            f64,
            u32,
            Eytzinger,
            3,
            32,
        >>::new_with_capacity(64);
        let x = [1.0, 2.0, 3.0];
        let y = [4.0, 5.0, 6.0];
        let z = [7.0, 8.0, 9.0];
        let items = [10u32, 11, 12];

        <VecOfArenas<f64, u32, 3, 32> as ConstructibleLeafStrategy<
            f64,
            u32,
            Eytzinger,
            3,
            32,
        >>::append_leaf(&mut leaves, &[&x, &y, &z], &items);

        assert_eq!(leaves.size, 3);
        assert_eq!(leaves.leaf_extents.len(), 1);
        assert_eq!(leaves.leaf_extents, vec![(0, 3)]);
        assert_eq!(
            leaves.leaf_bytes.len(),
            LeafArena::<f64, u32, 3>::encoded_len_bytes(3)
        );
    }

    #[test]
    fn vec_of_arenas_decodes_tiled_points_and_items() {
        let mut leaves = <VecOfArenas<f64, u32, 3, 32> as ConstructibleLeafStrategy<
            f64,
            u32,
            Eytzinger,
            3,
            32,
        >>::new_with_capacity(64);
        let x: Vec<f64> = (0..40).map(|v| v as f64).collect();
        let y: Vec<f64> = (100..140).map(|v| v as f64).collect();
        let z: Vec<f64> = (200..240).map(|v| v as f64).collect();
        let items: Vec<u32> = (1000..1040).collect();

        <VecOfArenas<f64, u32, 3, 32> as ConstructibleLeafStrategy<
            f64,
            u32,
            Eytzinger,
            3,
            32,
        >>::append_leaf(&mut leaves, &[&x, &y, &z], &items);

        let arena =
            <VecOfArenas<f64, u32, 3, 32> as LeafStrategy<f64, u32, Eytzinger, 3, 32>>::leaf_arena(
                &leaves, 0,
            );

        let mut recovered_x = Vec::new();
        let mut recovered_y = Vec::new();
        let mut recovered_z = Vec::new();
        let mut recovered_items = Vec::new();

        arena.for_each_tiled_chunk(|tile| {
            for idx in 0..tile.len() {
                unsafe {
                    recovered_x.push(tile.point_unaligned(0, idx));
                    recovered_y.push(tile.point_unaligned(1, idx));
                    recovered_z.push(tile.point_unaligned(2, idx));
                    recovered_items.push(tile.item_unaligned(idx));
                }
            }
        });

        assert_eq!(recovered_x, x);
        assert_eq!(recovered_y, y);
        assert_eq!(recovered_z, z);
        assert_eq!(recovered_items, items);
    }

    #[test]
    fn vec_of_arenas_uses_descending_chunk_sizes_per_leaf() {
        let mut leaves = <VecOfArenas<f64, u32, 3, 32> as ConstructibleLeafStrategy<
            f64,
            u32,
            Eytzinger,
            3,
            32,
        >>::new_with_capacity(64);
        let x: Vec<f64> = (0..47).map(|v| v as f64).collect();
        let y: Vec<f64> = (100..147).map(|v| v as f64).collect();
        let z: Vec<f64> = (200..247).map(|v| v as f64).collect();
        let items: Vec<u32> = (1000..1047).collect();

        <VecOfArenas<f64, u32, 3, 32> as ConstructibleLeafStrategy<
            f64,
            u32,
            Eytzinger,
            3,
            32,
        >>::append_leaf(&mut leaves, &[&x, &y, &z], &items);

        let arena =
            <VecOfArenas<f64, u32, 3, 32> as LeafStrategy<f64, u32, Eytzinger, 3, 32>>::leaf_arena(
                &leaves, 0,
            );

        let mut tile_lens = Vec::new();
        arena.for_each_tiled_chunk(|tile| tile_lens.push(tile.len()));

        assert_eq!(tile_lens, vec![32, 8, 4, 2, 1]);
    }

    #[test]
    fn extend_bytes_from_slice_appends_exact_repr_bytes() {
        let mut bytes = Vec::new();
        let values = [1u32, 2, 3];

        extend_bytes_from_slice(&mut bytes, &values);

        assert_eq!(bytes.len(), std::mem::size_of_val(&values));
    }

    #[test]
    fn vec_of_arenas_replace_item_in_leaf_replaces_first_exact_match() {
        let mut leaves = <VecOfArenas<f64, u32, 2, 32> as ConstructibleLeafStrategy<
            f64,
            u32,
            Eytzinger,
            2,
            32,
        >>::new_with_capacity(64);
        let x = [1.0, 1.0, 2.0];
        let y = [10.0, 10.0, 20.0];
        let items = [5u32, 5, 6];

        <VecOfArenas<f64, u32, 2, 32> as ConstructibleLeafStrategy<
            f64,
            u32,
            Eytzinger,
            2,
            32,
        >>::append_leaf(&mut leaves, &[&x, &y], &items);

        assert!(<VecOfArenas<f64, u32, 2, 32> as LeafStrategy<
            f64,
            u32,
            Eytzinger,
            2,
            32,
        >>::replace_item_in_leaf(
            &mut leaves, 0, &[1.0, 10.0], 5, 9
        ));

        let arena =
            <VecOfArenas<f64, u32, 2, 32> as LeafStrategy<f64, u32, Eytzinger, 2, 32>>::leaf_arena(
                &leaves, 0,
            );
        assert_eq!(arena.point_item(0), ([1.0, 10.0], 9));
        assert_eq!(arena.point_item(1), ([1.0, 10.0], 5));
        assert_eq!(arena.point_item(2), ([2.0, 20.0], 6));
        assert_eq!(leaves.size, 3);
    }

    #[test]
    fn vec_of_arenas_replace_item_in_leaf_returns_false_when_point_does_not_match() {
        let mut leaves = <VecOfArenas<f64, u32, 2, 32> as ConstructibleLeafStrategy<
            f64,
            u32,
            Eytzinger,
            2,
            32,
        >>::new_with_capacity(64);
        let x = [1.0, 2.0, 3.0];
        let y = [10.0, 20.0, 30.0];
        let items = [5u32, 6, 7];

        <VecOfArenas<f64, u32, 2, 32> as ConstructibleLeafStrategy<
            f64,
            u32,
            Eytzinger,
            2,
            32,
        >>::append_leaf(&mut leaves, &[&x, &y], &items);

        assert!(!<VecOfArenas<f64, u32, 2, 32> as LeafStrategy<
            f64,
            u32,
            Eytzinger,
            2,
            32,
        >>::replace_item_in_leaf(
            &mut leaves, 0, &[9.0, 90.0], 5, 9
        ));

        let arena =
            <VecOfArenas<f64, u32, 2, 32> as LeafStrategy<f64, u32, Eytzinger, 2, 32>>::leaf_arena(
                &leaves, 0,
            );
        assert_eq!(arena.point_item(0), ([1.0, 10.0], 5));
        assert_eq!(arena.point_item(1), ([2.0, 20.0], 6));
        assert_eq!(arena.point_item(2), ([3.0, 30.0], 7));
    }

    #[test]
    fn vec_of_arenas_replace_item_in_leaf_returns_false_when_item_does_not_match() {
        let mut leaves = <VecOfArenas<f64, u32, 2, 32> as ConstructibleLeafStrategy<
            f64,
            u32,
            Eytzinger,
            2,
            32,
        >>::new_with_capacity(64);
        let x = [1.0, 2.0, 3.0];
        let y = [10.0, 20.0, 30.0];
        let items = [5u32, 6, 7];

        <VecOfArenas<f64, u32, 2, 32> as ConstructibleLeafStrategy<
            f64,
            u32,
            Eytzinger,
            2,
            32,
        >>::append_leaf(&mut leaves, &[&x, &y], &items);

        assert!(!<VecOfArenas<f64, u32, 2, 32> as LeafStrategy<
            f64,
            u32,
            Eytzinger,
            2,
            32,
        >>::replace_item_in_leaf(
            &mut leaves, 0, &[2.0, 20.0], 99, 9
        ));

        let arena =
            <VecOfArenas<f64, u32, 2, 32> as LeafStrategy<f64, u32, Eytzinger, 2, 32>>::leaf_arena(
                &leaves, 0,
            );
        assert_eq!(arena.point_item(0), ([1.0, 10.0], 5));
        assert_eq!(arena.point_item(1), ([2.0, 20.0], 6));
        assert_eq!(arena.point_item(2), ([3.0, 30.0], 7));
    }

    #[test]
    fn vec_of_arenas_replace_item_in_leaf_replaces_match_in_later_tile() {
        let mut leaves = <VecOfArenas<f64, u32, 2, 32> as ConstructibleLeafStrategy<
            f64,
            u32,
            Eytzinger,
            2,
            32,
        >>::new_with_capacity(64);
        let x: Vec<f64> = (0..40).map(|v| v as f64).collect();
        let y: Vec<f64> = (100..140).map(|v| v as f64).collect();
        let items: Vec<u32> = (1000..1040).collect();

        <VecOfArenas<f64, u32, 2, 32> as ConstructibleLeafStrategy<
            f64,
            u32,
            Eytzinger,
            2,
            32,
        >>::append_leaf(&mut leaves, &[&x, &y], &items);

        assert!(<VecOfArenas<f64, u32, 2, 32> as LeafStrategy<
            f64,
            u32,
            Eytzinger,
            2,
            32,
        >>::replace_item_in_leaf(
            &mut leaves, 0, &[35.0, 135.0], 1035, 9999
        ));

        let arena =
            <VecOfArenas<f64, u32, 2, 32> as LeafStrategy<f64, u32, Eytzinger, 2, 32>>::leaf_arena(
                &leaves, 0,
            );
        assert_eq!(arena.point_item(31), ([31.0, 131.0], 1031));
        assert_eq!(arena.point_item(35), ([35.0, 135.0], 9999));
        assert_eq!(arena.point_item(39), ([39.0, 139.0], 1039));
    }
}

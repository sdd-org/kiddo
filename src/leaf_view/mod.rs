use std::any::TypeId;
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;

use crate::dist::DistanceMetric;
use crate::results::result_collection::{
    BestNeighbourResultCollection, FromLeafCandidate, ResultCollection,
};
use crate::{Axis, BestQueryResultItem, Content};

#[cfg(feature = "fixed")]
use fixed::{
    types::extra::{U0, U16, U8},
    FixedI32, FixedU16,
};
use std::marker::PhantomData;

// TODO: chunking
#[allow(unused)]
const CHUNK_SIZE: usize = 32;
pub(crate) const LEAF_SCRATCH_CAPACITY: usize = 1024;
pub(crate) const LEAF_ARENA_TILE_WIDTHS: [usize; 5] = [32, 8, 4, 2, 1];

#[inline(always)]
pub(crate) fn leaf_arena_tile_len(remaining: usize) -> usize {
    for width in LEAF_ARENA_TILE_WIDTHS {
        if remaining >= width {
            return width;
        }
    }

    1
}

#[inline(always)]
pub(crate) fn for_each_leaf_arena_tile_len(len: usize, mut f: impl FnMut(usize)) {
    let mut remaining = len;

    while remaining != 0 {
        let tile_len = leaf_arena_tile_len(remaining);
        f(tile_len);
        remaining -= tile_len;
    }
}

struct LeafScratch<O> {
    acc: [MaybeUninit<O>; LEAF_SCRATCH_CAPACITY],
    coord_wide: [MaybeUninit<O>; LEAF_SCRATCH_CAPACITY],
}

impl<O> LeafScratch<O> {
    const fn new() -> Self {
        Self {
            acc: [const { MaybeUninit::uninit() }; LEAF_SCRATCH_CAPACITY],
            coord_wide: [const { MaybeUninit::uninit() }; LEAF_SCRATCH_CAPACITY],
        }
    }
}

#[inline]
unsafe fn leaf_scratch_slice_mut<O>(
    scratch: &mut [MaybeUninit<O>; LEAF_SCRATCH_CAPACITY],
    len: usize,
) -> &mut [O] {
    debug_assert!(len <= LEAF_SCRATCH_CAPACITY);
    std::slice::from_raw_parts_mut(scratch.as_mut_ptr() as *mut O, len)
}

#[doc(hidden)]
pub trait TlsLeafScratch: Axis<Coord = Self> + 'static {
    fn with_tls_leaf_scratch<R>(len: usize, f: impl FnOnce(&mut [Self], &mut [Self]) -> R) -> R;
}

macro_rules! impl_tls_leaf_scratch {
    ($tls_name:ident, $t:ty) => {
        thread_local! {
            static $tls_name: UnsafeCell<LeafScratch<$t>> =
                const { UnsafeCell::new(LeafScratch::new()) };
        }

        impl TlsLeafScratch for $t {
            #[inline]
            fn with_tls_leaf_scratch<R>(
                len: usize,
                f: impl FnOnce(&mut [Self], &mut [Self]) -> R,
            ) -> R {
                if len > LEAF_SCRATCH_CAPACITY {
                    let mut acc = vec![<$t as Axis>::zero(); len];
                    let mut coord_wide = vec![<$t as Axis>::zero(); len];
                    return f(&mut acc, &mut coord_wide);
                }

                $tls_name.with(|scratch| {
                    let scratch = unsafe { &mut *scratch.get() };
                    let acc = unsafe { leaf_scratch_slice_mut(&mut scratch.acc, len) };
                    let coord_wide =
                        unsafe { leaf_scratch_slice_mut(&mut scratch.coord_wide, len) };
                    f(acc, coord_wide)
                })
            }
        }
    };
}

impl_tls_leaf_scratch!(LEAF_SCRATCH_F32, f32);
impl_tls_leaf_scratch!(LEAF_SCRATCH_F64, f64);
#[cfg(feature = "fixed")]
impl_tls_leaf_scratch!(LEAF_SCRATCH_FIXED_I32_U16, FixedI32<U16>);
#[cfg(feature = "fixed")]
impl_tls_leaf_scratch!(LEAF_SCRATCH_FIXED_I32_U0, FixedI32<U0>);
#[cfg(feature = "fixed")]
impl_tls_leaf_scratch!(LEAF_SCRATCH_FIXED_U16_U8, FixedU16<U8>);

#[cfg(feature = "f16")]
impl_tls_leaf_scratch!(LEAF_SCRATCH_F16, half::f16);

/// A view into a leaf node's data.
///
/// Provides a unified interface for accessing leaf data regardless of the underlying
/// storage strategy.
#[derive(Debug)]
pub struct LeafView<'a, AX, T, const K: usize, const B: usize> {
    points: [&'a [AX]; K],
    items: &'a [T],
}

/// Arena-backed view over one encoded leaf.
#[derive(Clone, Copy, Debug)]
pub struct LeafArena<'a, AX, T, const K: usize> {
    bytes: *const u8,
    len: usize,
    _phantom: PhantomData<(&'a AX, &'a T)>,
}

/// Arena-backed view over one encoded tile within a leaf.
#[derive(Clone, Copy, Debug)]
pub struct LeafArenaTile<'a, AX, T, const K: usize> {
    bytes: *const u8,
    len: usize,
    _phantom: PhantomData<(&'a AX, &'a T)>,
}

impl<'a, AX: Axis<Coord = AX>, T: Content, const K: usize, const B: usize>
    LeafView<'a, AX, T, K, B>
{
    pub fn new(points: [&'a [AX]; K], items: &'a [T]) -> Self {
        Self { points, items }
    }

    #[allow(dead_code)]
    pub(crate) fn into_parts(self) -> ([&'a [AX]; K], &'a [T]) {
        (self.points, self.items)
    }

    #[inline(always)]
    pub(crate) fn points(&self) -> [&'a [AX]; K] {
        self.points
    }

    #[inline(always)]
    pub(crate) fn items(&self) -> &'a [T] {
        self.items
    }

    #[inline(always)]
    pub(crate) fn point(&self, idx: usize) -> [AX; K] {
        debug_assert!(idx < self.items.len());
        array_init::array_init(|dim| unsafe { *self.points.get_unchecked(dim).get_unchecked(idx) })
    }

    #[inline(always)]
    pub(crate) fn point_item(&self, idx: usize) -> ([AX; K], T) {
        debug_assert!(idx < self.items.len());
        let point = self.point(idx);
        let item = unsafe { *self.items.get_unchecked(idx) };
        (point, item)
    }

    #[allow(dead_code)]
    #[cfg_attr(not(feature = "no_inline"), inline)]
    pub(crate) fn len(&self) -> usize {
        unsafe { self.points.get_unchecked(0).len() }
    }

    #[allow(dead_code)]
    #[inline]
    pub(crate) fn nearest_one<D>(
        &self,
        query: &[AX; K],
        best_dist: &mut D::Output,
        best_item: &mut T,
    ) where
        D: DistanceMetric<AX>,
        D::Output: TlsLeafScratch,
        AX: 'static,
    {
        let mut query_wide: [D::Output; K] = [D::Output::zero(); K];
        for dim in 0..K {
            query_wide[dim] = D::widen_coord(query[dim]);
        }

        self.nearest_one_with_query_wide::<D>(&query_wide, best_dist, best_item);
    }

    #[inline(always)]
    pub(crate) fn nearest_one_with_query_wide<D>(
        &self,
        query_wide: &[D::Output; K],
        best_dist: &mut D::Output,
        best_item: &mut T,
    ) where
        D: DistanceMetric<AX>,
        D::Output: TlsLeafScratch,
        AX: 'static,
    {
        if crate::leaf_view_chunked::try_nearest_one_with_query_wide::<AX, T, D, K, B>(
            self, query_wide, best_dist, best_item,
        ) {
            return;
        }

        let n = self.len();

        D::Output::with_tls_leaf_scratch(n, |acc, coord_wide| {
            if n == 0 {
                return;
            }

            let points_dim = unsafe { *self.points.get_unchecked(0) };
            let q = unsafe { *query_wide.get_unchecked(0) };
            let widened_axis =
                if let Some(identity_axis) = try_identity_widen_axis::<AX, D::Output>(points_dim) {
                    identity_axis
                } else {
                    D::widen_axis(points_dim, coord_wide);
                    &*coord_wide
                };

            let mut idx = 0;
            while idx < n {
                unsafe {
                    *acc.get_unchecked_mut(idx) = D::dist1(*widened_axis.get_unchecked(idx), q);
                }
                idx += 1;
            }

            for dim in 1..K {
                let points_dim = unsafe { *self.points.get_unchecked(dim) };
                let q = unsafe { *query_wide.get_unchecked(dim) };
                let widened_axis = if let Some(identity_axis) =
                    try_identity_widen_axis::<AX, D::Output>(points_dim)
                {
                    identity_axis
                } else {
                    D::widen_axis(points_dim, coord_wide);
                    &*coord_wide
                };

                let mut idx = 0;
                while idx < n {
                    unsafe {
                        D::combine_component(
                            acc.get_unchecked_mut(idx),
                            D::dist1(*widened_axis.get_unchecked(idx), q),
                        );
                    }
                    idx += 1;
                }
            }

            let mut leaf_best_idx = 0usize;
            let mut leaf_best_dist = unsafe { *acc.get_unchecked(0) };

            let mut idx = 1usize;
            while idx < n {
                let dist = unsafe { *acc.get_unchecked(idx) };
                if dist < leaf_best_dist {
                    leaf_best_dist = dist;
                    leaf_best_idx = idx;
                }
                idx += 1;
            }

            if leaf_best_dist < *best_dist {
                *best_dist = leaf_best_dist;
                *best_item = unsafe { *self.items.get_unchecked(leaf_best_idx) };
            }
        });
    }

    #[cfg_attr(not(feature = "no_inline"), inline)]
    pub(crate) fn with_dists_for_slice_wide<D, R>(
        &self,
        query_wide: &[D::Output; K],
        f: impl FnOnce(&[D::Output]) -> R,
    ) -> R
    where
        D: DistanceMetric<AX>,
        D::Output: TlsLeafScratch,
        AX: 'static,
    {
        let n = self.len();

        D::Output::with_tls_leaf_scratch(n, |acc, coord_wide| {
            acc.fill(D::Output::zero());

            for dim in 0..K {
                let points_dim = unsafe { *self.points.get_unchecked(dim) };
                let q = unsafe { *query_wide.get_unchecked(dim) };
                let widened_axis = if let Some(identity_axis) =
                    try_identity_widen_axis::<AX, D::Output>(points_dim)
                {
                    identity_axis
                } else {
                    D::widen_axis(points_dim, coord_wide);
                    &*coord_wide
                };

                for (dst, &coord) in acc.iter_mut().zip(widened_axis.iter()) {
                    D::combine_component(dst, D::dist1(coord, q));
                }
            }

            f(acc)
        })
    }

    #[allow(dead_code)]
    #[cfg_attr(not(feature = "no_inline"), inline)]
    pub(crate) fn update_nearest_dist<O>(
        dists: &[O],
        items: &[T],
        best_dist: &mut O,
        best_item: &mut T,
    ) where
        O: Axis<Coord = O>,
    {
        let (leaf_best_item, leaf_best_dist) = dists
            .iter()
            .enumerate()
            .min_by(|(_, &a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        if *leaf_best_dist < *best_dist {
            *best_dist = *leaf_best_dist;
            *best_item = items[leaf_best_item];
        }
    }

    #[cfg_attr(not(feature = "no_inline"), inline)]
    pub(crate) fn update_nearest_dists<O, E, R, const EXCLUSIVE: bool>(
        &self,
        dists: &[O],
        items: &[T],
        dist: O,
        results: &mut R,
    ) where
        O: Axis<Coord = O>,
        E: FromLeafCandidate<AX, T, O, K>,
        R: ResultCollection<O, E>,
    {
        dists
            .iter()
            .zip(items)
            .enumerate()
            .for_each(|(idx, (&d, &i))| {
                let is_within_dist = if EXCLUSIVE { d < dist } else { d <= dist };

                if is_within_dist {
                    #[cfg(feature = "result_collection_stats")]
                    crate::results::result_collection_stats::record_candidate_emitted();
                    results.add(E::from_leaf_candidate(self.point(idx), i, d));
                }
            })
    }
}

impl<'a, AX, T, const K: usize> LeafArena<'a, AX, T, K>
where
    AX: Copy,
    T: Copy,
{
    pub(crate) fn new(bytes: *const u8, len: usize) -> Self {
        Self {
            bytes,
            len,
            _phantom: PhantomData,
        }
    }

    #[inline(always)]
    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[cfg(any(
        all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"),
        all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"),
        all(feature = "simd", target_arch = "aarch64", target_feature = "neon")
    ))]
    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    #[cfg(any(
        all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"),
        all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"),
        all(feature = "simd", target_arch = "aarch64", target_feature = "neon")
    ))]
    #[inline(always)]
    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.bytes
    }

    #[cfg(any(debug_assertions, test))]
    #[inline(always)]
    pub(crate) fn encoded_len_bytes(len: usize) -> usize {
        let mut total = 0usize;

        for_each_leaf_arena_tile_len(len, |tile_len| {
            total += K * tile_len * std::mem::size_of::<AX>();
            total += tile_len * std::mem::size_of::<T>();
        });

        total
    }

    #[inline(always)]
    pub(crate) fn for_each_tiled_chunk(&self, mut f: impl FnMut(LeafArenaTile<'a, AX, T, K>)) {
        let mut byte_offset = 0usize;

        for_each_leaf_arena_tile_len(self.len, |tile_len| {
            let tile_byte_len =
                K * tile_len * std::mem::size_of::<AX>() + tile_len * std::mem::size_of::<T>();

            f(LeafArenaTile {
                bytes: unsafe { self.bytes.add(byte_offset) },
                len: tile_len,
                _phantom: PhantomData,
            });

            byte_offset += tile_byte_len;
        });
    }

    #[inline(always)]
    pub(crate) fn point_item(&self, idx: usize) -> ([AX; K], T) {
        debug_assert!(idx < self.len);
        let mut base = 0usize;
        let mut byte_offset = 0usize;

        for width in LEAF_ARENA_TILE_WIDTHS {
            while self.len - base >= width {
                if idx < base + width {
                    let tile: LeafArenaTile<'_, AX, T, K> = LeafArenaTile {
                        bytes: unsafe { self.bytes.add(byte_offset) },
                        len: width,
                        _phantom: PhantomData,
                    };
                    let tile_idx = idx - base;
                    let point = array_init::array_init(|dim| unsafe {
                        tile.point_unaligned(dim, tile_idx)
                    });
                    let item = unsafe { tile.item_unaligned(tile_idx) };
                    return (point, item);
                }

                base += width;
                byte_offset +=
                    K * width * std::mem::size_of::<AX>() + width * std::mem::size_of::<T>();
            }
        }

        unsafe { std::hint::unreachable_unchecked() }
    }
}

impl<AX, T, const K: usize> LeafArenaTile<'_, AX, T, K>
where
    AX: Copy,
    T: Copy,
{
    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub(crate) fn point(&self, idx: usize) -> [AX; K] {
        array_init::array_init(|dim| unsafe { self.point_unaligned(dim, idx) })
    }

    #[inline(always)]
    pub(crate) unsafe fn point_unaligned(&self, dim: usize, idx: usize) -> AX {
        debug_assert!(dim < K);
        debug_assert!(idx < self.len);

        let point_offset = (dim * self.len + idx) * std::mem::size_of::<AX>();
        std::ptr::read_unaligned(self.bytes.add(point_offset) as *const AX)
    }

    #[inline(always)]
    pub(crate) unsafe fn item_unaligned(&self, idx: usize) -> T {
        debug_assert!(idx < self.len);

        let item_offset = K * self.len * std::mem::size_of::<AX>() + idx * std::mem::size_of::<T>();
        std::ptr::read_unaligned(self.bytes.add(item_offset) as *const T)
    }
}

#[inline(always)]
pub(crate) fn try_identity_widen_axis<AX: 'static, O: 'static>(axis: &[AX]) -> Option<&[O]> {
    let same_type = TypeId::of::<AX>() == TypeId::of::<O>();
    let float_identity =
        TypeId::of::<AX>() == TypeId::of::<f32>() || TypeId::of::<AX>() == TypeId::of::<f64>();

    if same_type && float_identity {
        Some(unsafe { std::slice::from_raw_parts(axis.as_ptr() as *const O, axis.len()) })
    } else {
        None
    }
}

impl<'a, AX: Axis<Coord = AX>, T: Content + PartialOrd, const K: usize, const B: usize>
    LeafView<'a, AX, T, K, B>
{
    #[cfg_attr(not(feature = "no_inline"), inline)]
    pub(crate) fn update_best_dists<O, R, const EXCLUSIVE: bool>(
        dists: &[O],
        items: &[T],
        dist: O,
        threshold_item: Option<T>,
        results: &mut R,
    ) where
        O: Axis<Coord = O>,
        R: BestNeighbourResultCollection<O, T>,
    {
        dists.iter().zip(items).for_each(|(&d, &item)| {
            let is_within_dist = if EXCLUSIVE { d < dist } else { d <= dist };

            if is_within_dist {
                if threshold_item.is_some_and(|worst_item| item >= worst_item) {
                    #[cfg(feature = "result_collection_stats")]
                    crate::results::result_collection_stats::record_best_item_threshold_reject();
                    return;
                }
                #[cfg(feature = "result_collection_stats")]
                crate::results::result_collection_stats::record_candidate_emitted();
                results.add(BestQueryResultItem {
                    point: (),
                    distance: d,
                    item,
                });
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::SquaredEuclidean;
    use crate::results::result_collection::{BestNeighbourResultCollection, ResultCollection};
    use crate::QueryResultItem;

    fn encode_leaf_arena<const K: usize>(points: [&[f32]; K], items: &[u32]) -> Vec<u8> {
        let len = items.len();
        let mut bytes = Vec::with_capacity(LeafArena::<f32, u32, K>::encoded_len_bytes(len));

        let mut base = 0usize;
        for_each_leaf_arena_tile_len(len, |tile_len| {
            for axis in points.iter().take(K) {
                for value in &axis[base..base + tile_len] {
                    bytes.extend_from_slice(&value.to_ne_bytes());
                }
            }
            for item in &items[base..base + tile_len] {
                bytes.extend_from_slice(&item.to_ne_bytes());
            }
            base += tile_len;
        });

        bytes
    }

    #[derive(Default)]
    struct TestNearestResults {
        entries: Vec<QueryResultItem<(), u32, f32>>,
    }

    impl ResultCollection<f32, QueryResultItem<(), u32, f32>> for TestNearestResults {
        fn with_max_qty(_max_qty: usize) -> Self {
            Self::default()
        }

        fn max_qty(&self) -> usize {
            usize::MAX
        }

        fn len(&self) -> usize {
            self.entries.len()
        }

        fn add(&mut self, entry: QueryResultItem<(), u32, f32>) {
            self.entries.push(entry);
        }

        fn threshold_distance(&self) -> Option<f32> {
            None
        }

        fn into_vec(self) -> Vec<QueryResultItem<(), u32, f32>> {
            self.entries
        }

        fn into_sorted_vec(mut self) -> Vec<QueryResultItem<(), u32, f32>> {
            self.entries
                .sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
            self.entries
        }
    }

    #[derive(Default)]
    struct TestBestResults {
        entries: Vec<BestQueryResultItem<(), u32, f32>>,
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    impl ResultCollection<f32, BestQueryResultItem<(), u32, f32>> for TestBestResults {
        fn with_max_qty(_max_qty: usize) -> Self {
            Self::default()
        }

        fn max_qty(&self) -> usize {
            usize::MAX
        }

        fn len(&self) -> usize {
            self.entries.len()
        }

        fn add(&mut self, entry: BestQueryResultItem<(), u32, f32>) {
            self.entries.push(entry);
        }

        fn threshold_distance(&self) -> Option<f32> {
            None
        }

        fn into_vec(self) -> Vec<BestQueryResultItem<(), u32, f32>> {
            self.entries
        }

        fn into_sorted_vec(mut self) -> Vec<BestQueryResultItem<(), u32, f32>> {
            self.entries.sort_by(|a, b| {
                a.item
                    .partial_cmp(&b.item)
                    .unwrap()
                    .then(a.distance.partial_cmp(&b.distance).unwrap())
            });
            self.entries
        }
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    impl BestNeighbourResultCollection<f32, u32> for TestBestResults {
        fn threshold_item(&self) -> Option<u32> {
            None
        }
    }

    #[test]
    fn leaf_arena_tile_len_prefers_largest_supported_tile() {
        assert_eq!(leaf_arena_tile_len(1), 1);
        assert_eq!(leaf_arena_tile_len(2), 2);
        assert_eq!(leaf_arena_tile_len(3), 2);
        assert_eq!(leaf_arena_tile_len(4), 4);
        assert_eq!(leaf_arena_tile_len(7), 4);
        assert_eq!(leaf_arena_tile_len(8), 8);
        assert_eq!(leaf_arena_tile_len(31), 8);
        assert_eq!(leaf_arena_tile_len(32), 32);
        assert_eq!(leaf_arena_tile_len(47), 32);
        assert_eq!(leaf_arena_tile_len(100), 32);
    }

    #[test]
    fn for_each_leaf_arena_tile_len_emits_expected_sequence() {
        let mut widths = Vec::new();
        for_each_leaf_arena_tile_len(47, |tile_len| widths.push(tile_len));
        assert_eq!(widths, vec![32, 8, 4, 2, 1]);
    }

    #[test]
    fn leaf_view_accessors_round_trip_points_and_items() {
        let xs = [1.0f32, 2.0, 3.0];
        let ys = [10.0f32, 20.0, 30.0];
        let items = [7u32, 8, 9];
        let view = LeafView::<f32, u32, 2, 8>::new([&xs, &ys], &items);

        assert_eq!(view.len(), 3);
        assert_eq!(view.points()[0], &xs);
        assert_eq!(view.points()[1], &ys);
        assert_eq!(view.items(), &items);
        assert_eq!(view.point_item(1), ([2.0, 20.0], 8));

        let (parts_points, parts_items) = view.into_parts();
        assert_eq!(parts_points[0], &xs);
        assert_eq!(parts_points[1], &ys);
        assert_eq!(parts_items, &items);
    }

    #[test]
    fn try_identity_widen_axis_only_accepts_float_identity() {
        let f32_axis = [1.0f32, 2.5, 9.0];
        let f64_axis = [1.0f64, 2.5, 9.0];
        let u32_axis = [1u32, 2, 3];

        let widened_f32 = try_identity_widen_axis::<f32, f32>(&f32_axis).unwrap();
        let widened_f64 = try_identity_widen_axis::<f64, f64>(&f64_axis).unwrap();

        assert_eq!(widened_f32, &f32_axis);
        assert_eq!(widened_f64, &f64_axis);
        assert!(try_identity_widen_axis::<u32, u32>(&u32_axis).is_none());
        assert!(try_identity_widen_axis::<f32, f64>(&f32_axis).is_none());
    }

    #[test]
    fn leaf_arena_round_trips_encoded_tiles() {
        let xs: Vec<f32> = (0..47).map(|idx| idx as f32 + 0.25).collect();
        let ys: Vec<f32> = (0..47).map(|idx| 100.0 + idx as f32 + 0.5).collect();
        let items: Vec<u32> = (0..47).map(|idx| idx as u32 + 10).collect();
        let bytes = encode_leaf_arena::<2>([&xs, &ys], &items);

        assert_eq!(
            bytes.len(),
            LeafArena::<f32, u32, 2>::encoded_len_bytes(items.len())
        );

        let arena = LeafArena::<f32, u32, 2>::new(bytes.as_ptr(), items.len());
        assert!(!arena.is_empty());

        #[cfg(any(
            all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"),
            all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"),
            all(feature = "simd", target_arch = "aarch64", target_feature = "neon")
        ))]
        {
            assert_eq!(arena.len(), items.len());
            assert_eq!(arena.as_ptr(), bytes.as_ptr());
        }

        for idx in [0usize, 1, 31, 32, 40, 46] {
            assert_eq!(arena.point_item(idx), ([xs[idx], ys[idx]], items[idx]));
        }

        let mut chunk_lens = Vec::new();
        let mut first_chunk_values = None;
        arena.for_each_tiled_chunk(|tile| {
            chunk_lens.push(tile.len());
            if first_chunk_values.is_none() {
                first_chunk_values = Some(unsafe {
                    (
                        [tile.point_unaligned(0, 0), tile.point_unaligned(1, 0)],
                        tile.item_unaligned(0),
                    )
                });
            }
        });

        assert_eq!(chunk_lens, vec![32, 8, 4, 2, 1]);
        assert_eq!(first_chunk_values, Some(([xs[0], ys[0]], items[0])));
    }

    #[test]
    fn update_nearest_dist_replaces_best_when_smaller_distance_found() {
        let dists = [4.0f32, 1.5, 2.0];
        let items = [11u32, 22, 33];
        let mut best_dist = 10.0f32;
        let mut best_item = 0u32;

        LeafView::<f32, u32, 2, 8>::update_nearest_dist(
            &dists,
            &items,
            &mut best_dist,
            &mut best_item,
        );

        assert_eq!(best_dist, 1.5);
        assert_eq!(best_item, 22);
    }

    #[test]
    fn leaf_view_nearest_one_finds_closest_item() {
        let xs = [0.0f32, 2.0, 5.0];
        let ys = [0.0f32, 2.0, 1.0];
        let items = [10u32, 20, 30];
        let view = LeafView::<f32, u32, 2, 8>::new([&xs, &ys], &items);
        let mut best_dist = f32::INFINITY;
        let mut best_item = 0u32;

        view.nearest_one::<SquaredEuclidean<f32>>(&[1.5, 1.5], &mut best_dist, &mut best_item);

        assert_eq!(best_dist, 0.5);
        assert_eq!(best_item, 20);
    }

    #[test]
    fn leaf_view_nearest_one_with_query_wide_handles_empty_leaf() {
        let xs: [f32; 0] = [];
        let ys: [f32; 0] = [];
        let items: [u32; 0] = [];
        let view = LeafView::<f32, u32, 2, 8>::new([&xs, &ys], &items);
        let mut best_dist = 7.5f32;
        let mut best_item = 99u32;

        view.nearest_one_with_query_wide::<SquaredEuclidean<f32>>(
            &[1.0f32, 2.0f32],
            &mut best_dist,
            &mut best_item,
        );

        assert_eq!(best_dist, 7.5);
        assert_eq!(best_item, 99);
    }

    #[test]
    fn leaf_view_nearest_one_with_query_wide_keeps_existing_better_best() {
        let xs = [0.0f32, 2.0, 5.0];
        let ys = [0.0f32, 2.0, 1.0];
        let items = [10u32, 20, 30];
        let view = LeafView::<f32, u32, 2, 8>::new([&xs, &ys], &items);
        let mut best_dist = 0.25f32;
        let mut best_item = 77u32;

        view.nearest_one_with_query_wide::<SquaredEuclidean<f32>>(
            &[1.5f32, 1.5f32],
            &mut best_dist,
            &mut best_item,
        );

        assert_eq!(best_dist, 0.25);
        assert_eq!(best_item, 77);
    }

    #[test]
    fn leaf_view_with_dists_for_slice_wide_returns_expected_distances() {
        let xs = [0.0f32, 2.0, 5.0];
        let ys = [0.0f32, 2.0, 1.0];
        let items = [10u32, 20, 30];
        let view = LeafView::<f32, u32, 2, 8>::new([&xs, &ys], &items);

        let dists = view
            .with_dists_for_slice_wide::<SquaredEuclidean<f32>, _>(&[1.0, 1.0], |dists| {
                dists.to_vec()
            });

        assert_eq!(dists, vec![2.0, 2.0, 16.0]);
    }

    #[test]
    fn update_nearest_dists_adds_only_items_within_threshold() {
        let dists = [4.0f32, 1.5, 2.0, 5.0];
        let items = [11u32, 22, 33, 44];
        let xs = [0.0f32; 4];
        let ys = [0.0f32; 4];
        let view = LeafView::<f32, u32, 2, 8>::new([&xs, &ys], &items);
        let mut results = TestNearestResults::default();

        view.update_nearest_dists::<_, _, _, false>(&dists, &items, 2.0, &mut results);

        let entries = results.into_sorted_vec();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].distance, 1.5);
        assert_eq!(entries[0].item, 22);
        assert_eq!(entries[1].distance, 2.0);
        assert_eq!(entries[1].item, 33);
    }

    #[test]
    fn update_best_dists_honors_distance_and_threshold_item() {
        let dists = [4.0f32, 1.5, 2.0, 1.0];
        let items = [11u32, 22, 33, 44];
        let mut results = TestBestResults::default();

        LeafView::<f32, u32, 2, 8>::update_best_dists::<_, _, false>(
            &dists,
            &items,
            2.0,
            Some(30),
            &mut results,
        );

        let entries = results.into_sorted_vec();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].distance, 1.5);
        assert_eq!(entries[0].item, 22);
    }

    #[test]
    fn leaf_scratch_slice_mut_returns_writable_prefix() {
        let mut scratch = [const { MaybeUninit::<u32>::uninit() }; LEAF_SCRATCH_CAPACITY];

        let slice = unsafe { leaf_scratch_slice_mut(&mut scratch, 3) };
        slice.copy_from_slice(&[7, 8, 9]);

        assert_eq!(slice, &[7, 8, 9]);
    }

    #[test]
    fn tls_leaf_scratch_with_tls_leaf_scratch_provides_two_mutable_buffers() {
        let result = <f32 as TlsLeafScratch>::with_tls_leaf_scratch(4, |acc, coord_wide| {
            acc.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
            coord_wide.copy_from_slice(&[10.0, 20.0, 30.0, 40.0]);
            (acc.to_vec(), coord_wide.to_vec())
        });

        assert_eq!(result.0, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(result.1, vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn tls_leaf_scratch_uses_heap_fallback_above_fixed_capacity() {
        let len = LEAF_SCRATCH_CAPACITY + 1;
        let result = <f32 as TlsLeafScratch>::with_tls_leaf_scratch(len, |acc, coord_wide| {
            acc[len - 1] = 7.0;
            coord_wide[len - 1] = 11.0;
            (
                acc.len(),
                coord_wide.len(),
                acc[len - 1],
                coord_wide[len - 1],
            )
        });

        assert_eq!(result, (len, len, 7.0, 11.0));
    }
}

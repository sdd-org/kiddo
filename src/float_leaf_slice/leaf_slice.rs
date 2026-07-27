use crate::float::result_collection::ResultCollection;
use az::Cast;
use std::collections::BinaryHeap;
use std::slice::ChunksExact;

const CHUNK_SIZE: usize = 32;

/*#[cfg(all(
    feature = "simd",
    target_feature = "avx2",
    any(target_arch = "x86", target_arch = "x86_64")
))]
use super::f64_avx2::get_best_from_dists_f64_avx2;*/
//use super::{f32_avx2::get_best_from_dists_f32_avx2};

// #[cfg(all(
//     feature = "simd",
//     target_feature = "avx512f",
//     any(target_arch = "x86", target_arch = "x86_64")
// ))]
// use super::f64_avx512::get_best_from_dists_f64_avx512;

use super::fallback::{
    update_best_dists_within_autovec, update_nearest_dist_autovec,
    update_nearest_dists_within_autovec,
};

use crate::traits::DistanceMetric;
use crate::{float::kdtree::Axis, traits::Content, BestNeighbour, NearestNeighbour};

#[doc(hidden)]
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct LeafFixedSlice<'a, A: Axis, T: Content, const K: usize, const C: usize> {
    pub content_points: [&'a [A; C]; K],
    pub content_items: &'a [T; C],
}

impl<A, T, const K: usize, const C: usize> LeafFixedSlice<'_, A, T, K, C>
where
    A: Axis + LeafSliceFloat<T> + LeafSliceFloatChunk<T, K>,
    T: Content,
    usize: Cast<T>,
{
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn nearest_one<D>(&self, query: &[A; K], best_dist: &mut A, best_item: &mut T)
    where
        D: DistanceMetric<A, K>,
    {
        // AVX512: 4 loops of 32 iterations, each 4x unrolled, 5 instructions per pre-unrolled iteration
        let mut acc = [A::zero(); C];
        (0..K).step_by(1).for_each(|dim| {
            let qd = [query[dim]; C];

            (0..C).step_by(1).for_each(|idx| {
                acc[idx] =
                    D::accumulate(acc[idx], D::dist1(self.content_points[dim][idx], qd[idx]));
            });
        });

        A::update_nearest_dist(acc, self.content_items, best_dist, best_item);
    }
}

#[doc(hidden)]
#[derive(Debug)]
pub(crate) struct LeafSlice<'a, A: Axis, T: Content, const K: usize> {
    pub content_points: [&'a [A]; K],
    pub content_items: &'a [T],
}

impl<A: Axis, T: Content, const K: usize> LeafSlice<'_, A, T, K> {
    #[allow(dead_code)]
    #[inline]
    fn len(&self) -> usize {
        self.content_items.len()
    }
}

pub(crate) struct LeafFixedSliceIterator<'a, A: Axis, T: Content, const K: usize, const C: usize> {
    points_iterators: [ChunksExact<'a, A>; K],
    items_iterator: ChunksExact<'a, T>,
}

impl<'a, A: Axis, T: Content, const K: usize, const C: usize> Iterator
    for LeafFixedSliceIterator<'a, A, T, K, C>
{
    type Item = ([&'a [A; C]; K], &'a [T; C]);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.items_iterator.len() == 0 {
            None
        } else {
            let points_chunk: [&[A; C]; K] = array_init::array_init(|i| {
                self.points_iterators[i].next().unwrap().try_into().unwrap()
            });
            let item_chunk: &[T; C] = self.items_iterator.next().unwrap().try_into().unwrap();
            Some((points_chunk, item_chunk))
        }
    }
}

impl<'a, A: Axis, T: Content, const K: usize, const C: usize>
    LeafFixedSliceIterator<'a, A, T, K, C>
{
    #[inline]
    fn remainder(&self) -> ([&'a [A]; K], &'a [T]) {
        (
            array_init::array_init(|i| self.points_iterators[i].remainder()),
            self.items_iterator.remainder(),
        )
    }
}

pub trait LeafSliceFloatChunk<T, const K: usize>
where
    T: Content,
{
    fn dists_for_chunk<D, const C: usize>(chunk: [&[Self; C]; K], query: &[Self; K]) -> [Self; C]
    where
        D: DistanceMetric<Self, K>,
        Self: Sized;
}

pub trait LeafSliceFloat<T>
where
    T: Content,
{
    fn update_nearest_dist<const C: usize>(
        acc: [Self; C],
        items: &[T; C],
        best_dist: &mut Self,
        best_item: &mut T,
    ) where
        Self: Sized;

    fn update_nearest_dists_within<R, const C: usize>(
        acc: [Self; C],
        items: &[T; C],
        radius: Self,
        results: &mut R,
        inclusive: bool,
    ) where
        R: ResultCollection<Self, T>,
        usize: Cast<T>,
        Self: Axis + Sized;

    fn update_best_dists_within<const C: usize>(
        acc: [Self; C],
        items: &[T; C],
        radius: Self,
        max_qty: usize,
        results: &mut BinaryHeap<BestNeighbour<Self, T>>,
        inclusive: bool,
    ) where
        Self: Axis + Sized;
}

impl<A, T, const K: usize> LeafSlice<'_, A, T, K>
where
    A: Axis + LeafSliceFloat<T> + LeafSliceFloatChunk<T, K>,
    T: Content,
    usize: Cast<T>,
{
    #[inline]
    pub(crate) fn new<'a>(
        content_points: [&'a [A]; K],
        content_items: &'a [T],
    ) -> LeafSlice<'a, A, T, K> {
        let size = content_items.len();
        for arr in content_points {
            debug_assert_eq!(arr.len(), size);
        }

        LeafSlice {
            content_items,
            content_points,
        }
    }

    #[inline]
    // `slice::as_chunks` is unavailable at the crate's Rust 1.85 MSRV.
    #[allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]
    fn as_full_chunks<const C: usize>(&self) -> LeafFixedSliceIterator<'_, A, T, K, C> {
        let points_iterators = self.content_points.map(|i| i.chunks_exact(C));
        let items_iterator = self.content_items.chunks_exact(C);

        LeafFixedSliceIterator {
            items_iterator,
            points_iterators,
        }
    }

    #[inline]
    pub(crate) fn nearest_one<D>(&self, query: &[A; K], best_dist: &mut A, best_item: &mut T)
    where
        D: DistanceMetric<A, K>,
    {
        let chunk_iter = self.as_full_chunks::<CHUNK_SIZE>();
        let (remainder_points, remainder_items) = chunk_iter.remainder();
        for chunk in chunk_iter {
            let dists = A::dists_for_chunk::<D, CHUNK_SIZE>(chunk.0, query);
            A::update_nearest_dist(dists, chunk.1, best_dist, best_item);
        }

        #[allow(clippy::needless_range_loop)]
        for idx in 0..remainder_items.len() {
            let mut dist = A::zero();
            (0..K).step_by(1).for_each(|dim| {
                dist = D::accumulate(dist, D::dist1(remainder_points[dim][idx], query[dim]));
            });

            // TODO: make branchless
            let dist_is_better = u8::from(dist < *best_dist);
            // best_dist.cmovnz(&dist, dist_is_better);
            // best_item.cmovnz(&remainder_items[idx], dist_is_better);

            if dist_is_better == 1 {
                *best_dist = dist;
                *best_item = remainder_items[idx];
            }
        }
    }

    #[inline]
    pub(crate) fn nearest_n_within<D, R>(
        &self,
        query: &[A; K],
        radius: A,
        results: &mut R,
        inclusive: bool,
    ) where
        D: DistanceMetric<A, K>,
        R: ResultCollection<A, T>,
    {
        let chunk_iter = self.as_full_chunks::<CHUNK_SIZE>();
        let (remainder_points, remainder_items) = chunk_iter.remainder();
        for chunk in chunk_iter {
            let dists = A::dists_for_chunk::<D, CHUNK_SIZE>(chunk.0, query);

            A::update_nearest_dists_within(dists, chunk.1, radius, results, inclusive);
        }

        #[allow(clippy::needless_range_loop)]
        for idx in 0..remainder_items.len() {
            let mut distance = A::zero();
            (0..K).step_by(1).for_each(|dim| {
                distance =
                    D::accumulate(distance, D::dist1(remainder_points[dim][idx], query[dim]));
            });
            if if inclusive {
                distance <= radius
            } else {
                distance < radius
            } {
                results.add(NearestNeighbour {
                    distance,
                    item: remainder_items[idx],
                });
            }
        }
    }

    #[inline]
    pub(crate) fn best_n_within<D>(
        &self,
        query: &[A; K],
        radius: A,
        max_qty: usize,
        results: &mut BinaryHeap<BestNeighbour<A, T>>,
        inclusive: bool,
    ) where
        D: DistanceMetric<A, K>,
    {
        let chunk_iter = self.as_full_chunks::<CHUNK_SIZE>();
        let (remainder_points, remainder_items) = chunk_iter.remainder();
        for chunk in chunk_iter {
            let dists = A::dists_for_chunk::<D, CHUNK_SIZE>(chunk.0, query);

            A::update_best_dists_within(dists, chunk.1, radius, max_qty, results, inclusive);
        }

        #[allow(clippy::needless_range_loop)]
        for idx in 0..remainder_items.len() {
            let mut distance = A::zero();
            (0..K).step_by(1).for_each(|dim| {
                distance =
                    D::accumulate(distance, D::dist1(remainder_points[dim][idx], query[dim]));
            });

            if if inclusive {
                distance <= radius
            } else {
                distance < radius
            } {
                let item = remainder_items[idx];
                if results.len() < max_qty {
                    results.push(BestNeighbour { distance, item });
                } else {
                    let mut top = results.peek_mut().unwrap();
                    if item < top.item {
                        top.item = item;
                        top.distance = distance;
                    }
                }
            }
        }
    }
}

impl<T: Content> LeafSliceFloat<T> for f64
where
    T: Content,
    usize: Cast<T>,
{
    #[inline]
    fn update_nearest_dist<const C: usize>(
        acc: [f64; C],
        items: &[T; C],
        best_dist: &mut f64,
        best_item: &mut T,
    ) {
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            /*if is_x86_feature_detected!("avx512f") {
                #[cfg(target_feature = "avx512f")]
                unsafe {
                    get_best_from_dists_f64_avx512(&acc, items, best_dist, best_item)
                }
            } else */
            /*if is_x86_feature_detected!("avx2") {
                #[cfg(target_feature = "avx2")]
                unsafe {
                    get_best_from_dists_f64_avx2(&acc, items, best_dist, best_item)
                }
            } else {*/
            update_nearest_dist_autovec(&acc, items, best_dist, best_item)
            // }
        }

        #[cfg(any(
            not(feature = "simd"),
            not(any(target_arch = "x86", target_arch = "x86_64"))
        ))]
        {
            update_nearest_dist_autovec(&acc, items, best_dist, best_item)
        }
    }

    #[inline]
    fn update_nearest_dists_within<R, const C: usize>(
        acc: [f64; C],
        items: &[T; C],
        radius: f64,
        results: &mut R,
        inclusive: bool,
    ) where
        R: ResultCollection<f64, T>,
    {
        update_nearest_dists_within_autovec(&acc, items, radius, results, inclusive)
    }

    #[inline]
    fn update_best_dists_within<const C: usize>(
        acc: [f64; C],
        items: &[T; C],
        radius: f64,
        max_qty: usize,
        results: &mut BinaryHeap<BestNeighbour<f64, T>>,
        inclusive: bool,
    ) {
        update_best_dists_within_autovec(&acc, items, radius, max_qty, results, inclusive)
    }
}

impl<T: Content, const K: usize> LeafSliceFloatChunk<T, K> for f64
where
    T: Content,
    usize: Cast<T>,
{
    #[inline]
    fn dists_for_chunk<D, const C: usize>(chunk: [&[Self; C]; K], query: &[Self; K]) -> [Self; C]
    where
        D: DistanceMetric<Self, K>,
        Self: Sized,
    {
        // AVX512: 4 loops of 32 iterations, each 4x unrolled, 5 instructions per pre-unrolled iteration
        let mut acc = [0f64; C];
        (0..K).step_by(1).for_each(|dim| {
            let qd = [query[dim]; C];
            (0..C).step_by(1).for_each(|idx| {
                acc[idx] = D::accumulate(acc[idx], D::dist1(chunk[dim][idx], qd[idx]));
            });
        });
        acc
    }
}

impl<T: Content> LeafSliceFloat<T> for f32
where
    T: Content,
    usize: Cast<T>,
{
    #[inline]
    fn update_nearest_dist<const C: usize>(
        acc: [f32; C],
        items: &[T; C],
        best_dist: &mut f32,
        best_item: &mut T,
    ) {
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            /* if is_x86_feature_detected!("avx512f") {
                // TODO
                unimplemented!()
            } else */
            /*if is_x86_feature_detected!("avx2") {
                #[cfg(target_feature = "avx2")]
                unsafe {
                    get_best_from_dists_f32_avx2(&acc, items, best_dist, best_item)
                }
            } else {*/
            update_nearest_dist_autovec(&acc, items, best_dist, best_item)
            //}
        }

        #[cfg(any(
            not(feature = "simd"),
            not(any(target_arch = "x86", target_arch = "x86_64"))
        ))]
        {
            update_nearest_dist_autovec(&acc, items, best_dist, best_item)
        }
    }

    #[inline]
    fn update_nearest_dists_within<R, const C: usize>(
        acc: [f32; C],
        items: &[T; C],
        radius: f32,
        results: &mut R,
        inclusive: bool,
    ) where
        R: ResultCollection<f32, T>,
    {
        update_nearest_dists_within_autovec(&acc, items, radius, results, inclusive)
    }

    #[inline]
    fn update_best_dists_within<const C: usize>(
        acc: [f32; C],
        items: &[T; C],
        radius: f32,
        max_qty: usize,
        results: &mut BinaryHeap<BestNeighbour<f32, T>>,
        inclusive: bool,
    ) {
        update_best_dists_within_autovec(&acc, items, radius, max_qty, results, inclusive)
    }
}

impl<T: Content, const K: usize> LeafSliceFloatChunk<T, K> for f32
where
    T: Content,
    usize: Cast<T>,
{
    #[inline]
    fn dists_for_chunk<D, const C: usize>(chunk: [&[Self; C]; K], query: &[Self; K]) -> [Self; C]
    where
        D: DistanceMetric<Self, K>,
        Self: Sized,
    {
        // AVX512: 4 loops of 32 iterations, each 4x unrolled, 5 instructions per pre-unrolled iteration
        let mut acc = [0f32; C];
        (0..K).step_by(1).for_each(|dim| {
            let qd = [query[dim]; C];
            (0..C).step_by(1).for_each(|idx| {
                acc[idx] = D::accumulate(acc[idx], D::dist1(chunk[dim][idx], qd[idx]));
            });
        });
        acc
    }
}

#[cfg(test)]
mod test {
    use crate::float_leaf_slice::leaf_slice::{LeafFixedSlice, LeafSlice, LeafSliceFloat};
    use crate::{BestNeighbour, Chebyshev, NearestNeighbour, SquaredEuclidean};
    use std::collections::BinaryHeap;

    #[test]
    fn leaf_fixed_slice_nearest_one_works() {
        let content_points = [
            [0.0f64, 3.0f64, 5.0f64, 0.0f64],
            [0.0f64, 4.0f64, 12.0f64, 0.0f64],
        ];

        let content_items = [1u32, 2u32, 3u32, 4u32];

        let slice = LeafFixedSlice {
            content_points: [&content_points[0], &content_points[1]],
            content_items: &content_items,
        };

        let mut best_dist = f64::INFINITY;
        let mut best_item = u32::MAX;

        slice.nearest_one::<SquaredEuclidean>(&[0.0f64, 0.0f64], &mut best_dist, &mut best_item);

        assert_eq!(best_dist, 0f64);
        assert_eq!(best_item, 1u32);
    }

    #[test]
    fn leaf_fixed_slice_nearest_one_uses_metric_accumulator() {
        let content_points = [[0.0f64, 10.0], [0.0f64, 10.0]];
        let content_items = [0u32, 1u32];
        let slice = LeafFixedSlice {
            content_points: [&content_points[0], &content_points[1]],
            content_items: &content_items,
        };
        let mut best_dist = f64::INFINITY;
        let mut best_item = u32::MAX;

        slice.nearest_one::<Chebyshev>(&[9.0, 9.0], &mut best_dist, &mut best_item);

        assert_eq!(best_dist, 1.0);
        assert_eq!(best_item, 1);
    }

    #[test]
    fn test_f64_leafslicefloat_update_nearest_dists_within() {
        let dists = [10000f64, 20000f64, 20f64];
        let items = [1u32, 3u32, 5u32];

        let radius = 200f64;

        let mut results: BinaryHeap<NearestNeighbour<f64, u32>> = BinaryHeap::new();
        results.push(NearestNeighbour {
            distance: 10f64,
            item: 100u32,
        });

        f64::update_nearest_dists_within(dists, &items, radius, &mut results, true);

        let results = results.into_vec();

        assert_eq!(
            results,
            vec![
                NearestNeighbour {
                    distance: 20f64,
                    item: 5u32
                },
                NearestNeighbour {
                    distance: 10f64,
                    item: 100u32
                },
            ]
        );
    }

    #[test]
    fn test_f64_update_best_dists_within_autovec_leaves_nearest() {
        let dists = [10000f64, 20000f64, 20f64, 15f64];
        let items = [1u32, 3u32, 5u32, 7u32];

        let radius = 200f64;

        let max_qty = 2usize;

        let mut results = BinaryHeap::new();
        results.push(BestNeighbour {
            distance: 10f64,
            item: 100u32,
        });

        f64::update_best_dists_within(dists, &items, radius, max_qty, &mut results, true);

        let results = results.into_vec();

        assert_eq!(
            results,
            vec![
                BestNeighbour {
                    distance: 15f64,
                    item: 7u32
                },
                BestNeighbour {
                    distance: 20f64,
                    item: 5u32
                },
            ]
        );
    }

    #[test]
    fn test_f32_leafslicefloat_update_nearest_dists_within() {
        let dists = [10000f32, 20000f32, 20f32];
        let items = [1u32, 3u32, 5u32];

        let radius = 200f32;

        let mut results: BinaryHeap<NearestNeighbour<f32, u32>> = BinaryHeap::new();
        results.push(NearestNeighbour {
            distance: 10f32,
            item: 100u32,
        });

        f32::update_nearest_dists_within(dists, &items, radius, &mut results, true);

        let results = results.into_vec();

        assert_eq!(
            results,
            vec![
                NearestNeighbour {
                    distance: 20f32,
                    item: 5u32
                },
                NearestNeighbour {
                    distance: 10f32,
                    item: 100u32
                },
            ]
        );
    }

    #[test]
    fn test_f32_update_best_dists_within_autovec_leaves_nearest() {
        let dists = [10000f32, 20000f32, 20f32, 15f32];
        let items = [1u32, 3u32, 5u32, 7u32];

        let radius = 200f32;

        let max_qty = 2usize;

        let mut results = BinaryHeap::new();
        results.push(BestNeighbour {
            distance: 10f32,
            item: 100u32,
        });

        f32::update_best_dists_within(dists, &items, radius, max_qty, &mut results, true);

        let results = results.into_vec();

        assert_eq!(
            results,
            vec![
                BestNeighbour {
                    distance: 15f32,
                    item: 7u32
                },
                BestNeighbour {
                    distance: 20f32,
                    item: 5u32
                },
            ]
        );
    }

    // Test for remainder path processing with non-chunk-aligned sizes (CHUNK_SIZE=32)
    // Verifies the fix for using remainder_items[idx] instead of self.content_items[idx]
    #[test]
    fn test_remainder_processing_finds_correct_item() {
        // Size 33 = 1 chunk (32) + 1 remainder
        // Item 32 is in the remainder region - if the bug existed, it would return
        // self.content_items[0] instead of remainder_items[0]
        let mut dim0 = Vec::with_capacity(33);
        let mut dim1 = Vec::with_capacity(33);
        let mut items = Vec::with_capacity(33);
        for i in 0..33 {
            dim0.push(i as f64);
            dim1.push(0.0f64);
            items.push(i as u32);
        }

        let slice = LeafSlice {
            content_points: [&dim0[..], &dim1[..]],
            content_items: &items[..],
        };

        let mut results: BinaryHeap<NearestNeighbour<f64, u32>> = BinaryHeap::with_capacity(10);
        slice.nearest_n_within::<SquaredEuclidean, _>(
            &[32.0f64, 0.0f64],
            4.0f64,
            &mut results,
            true,
        );

        let items_found: Vec<_> = results.iter().map(|n| n.item).collect();
        assert!(
            items_found.contains(&32u32),
            "Should find item 32 in remainder region"
        );
    }
}

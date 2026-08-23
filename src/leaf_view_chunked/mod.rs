pub(crate) mod best_n_within;
pub(crate) mod nearest_n_within;
pub(crate) mod nearest_one;

use std::mem::MaybeUninit;

use array_init::array_init;

use crate::dist::DistanceMetric;
use crate::leaf_view::{try_identity_widen_axis, LeafView};
use crate::{Axis, Content};

const CHUNK_SIZE: usize = 32;

/// A view into a leaf node's data.
///
/// Provides a unified interface for accessing leaf data regardless of the underlying
/// storage strategy.
#[allow(unused)]
#[derive(Debug)]
pub struct LeafViewChunked<'a, AX, T, const K: usize> {
    points: [&'a [AX]; K],
    items: &'a [T],
}

#[inline(always)]
pub(crate) fn try_nearest_one_with_query_wide<AX, T, D, const K: usize, const B: usize>(
    leaf: &LeafView<'_, AX, T, K, B>,
    query_wide: &[D::Output; K],
    best_dist: &mut D::Output,
    best_item: &mut T,
) -> bool
where
    AX: Axis<Coord = AX> + 'static,
    T: Content,
    D: DistanceMetric<AX>,
    D::Output: Axis<Coord = D::Output> + 'static,
{
    let points = leaf.points();
    let widened_points = points.map(|axis| try_identity_widen_axis::<AX, D::Output>(axis));

    if widened_points.iter().any(Option::is_none) {
        return false;
    }

    let widened_points = widened_points.map(|axis| axis.unwrap());
    let items = leaf.items();
    let points_chunks = widened_points.map(|axis| axis.as_chunks::<CHUNK_SIZE>());
    let (items_chunks, remainder_items) = items.as_chunks::<CHUNK_SIZE>();

    for (chunk_idx, items_chunk) in items_chunks.iter().enumerate() {
        let points_chunk: [&[D::Output; CHUNK_SIZE]; K] =
            array_init(|dim| &points_chunks[dim].0[chunk_idx]);

        let dists = dists_for_chunk::<AX, D, K, CHUNK_SIZE>(points_chunk, query_wide);
        update_nearest_dist_chunk(dists, items_chunk, best_dist, best_item);
    }

    let remainder_points: [&[D::Output]; K] = array_init(|dim| points_chunks[dim].1);

    // TODO: For FlatVec specifically, we can experiment with overreading the tail into the next
    // contiguous leaf (or padded end-of-vec region) to keep the distance kernel branchless, while
    // masking/scalarizing only the final min-selection step. Keep this first pass leaf-bounded so
    // correctness is easy to validate and benchmark against the current scalar fallback.
    if !remainder_items.is_empty() {
        for idx in 0..remainder_items.len() {
            let mut dist = D::Output::zero();
            for dim in 0..K {
                unsafe {
                    D::combine_component(
                        &mut dist,
                        D::dist1(
                            *remainder_points.get_unchecked(dim).get_unchecked(idx),
                            *query_wide.get_unchecked(dim),
                        ),
                    );
                }
            }

            if dist < *best_dist {
                *best_dist = dist;
                *best_item = unsafe { *remainder_items.get_unchecked(idx) };
            }
        }
    }

    true
}

#[inline(always)]
fn dists_for_chunk<A, D, const K: usize, const C: usize>(
    chunk: [&[D::Output; C]; K],
    query: &[D::Output; K],
) -> [D::Output; C]
where
    A: Copy,
    D: DistanceMetric<A>,
{
    let mut acc = [const { MaybeUninit::<D::Output>::uninit() }; C];
    let acc_ptr = acc.as_mut_ptr() as *mut D::Output;

    let q = unsafe { *query.get_unchecked(0) };
    let axis = unsafe { *chunk.get_unchecked(0) };

    for idx in 0..C {
        unsafe {
            *acc_ptr.add(idx) = D::dist1(*axis.get_unchecked(idx), q);
        }
    }

    for dim in 1..K {
        let q = unsafe { *query.get_unchecked(dim) };
        let axis = unsafe { *chunk.get_unchecked(dim) };

        for idx in 0..C {
            unsafe {
                D::combine_component(
                    &mut *acc_ptr.add(idx),
                    D::dist1(*axis.get_unchecked(idx), q),
                );
            }
        }
    }

    unsafe { std::ptr::read(acc.as_ptr() as *const [D::Output; C]) }
}

/// Scalar nearest-one adapter for the V3 distance-metric trait stack.
///
/// This path is intended to be testable on any architecture and acts as the
/// non-SIMD baseline for the new `leaf_view_chunked` module.
#[allow(unused)]
#[inline(always)]
pub(crate) fn try_nearest_one_with_query_wide_v3<AX, T, D, const K: usize, const B: usize>(
    leaf: &LeafView<'_, AX, T, K, B>,
    query_wide: &[D::Output; K],
    best_dist: &mut D::Output,
    best_item: &mut T,
) -> bool
where
    AX: Axis<Coord = AX> + 'static,
    T: Content,
    D: DistanceMetric<AX>,
    D::Output: Axis<Coord = D::Output> + 'static,
{
    nearest_one::nearest_one_with_query_wide_fallback::<AX, T, D, K, B>(
        leaf, query_wide, best_dist, best_item,
    );
    true
}

#[inline(always)]
fn update_nearest_dist_chunk<O, T, const C: usize>(
    acc: [O; C],
    items: &[T; C],
    best_dist: &mut O,
    best_item: &mut T,
) where
    O: Axis<Coord = O>,
    T: Content,
{
    let mut candidate_idx = 0usize;
    let mut candidate_dist = *best_dist;
    let mut improved = false;

    for idx in 0..C {
        let dist = unsafe { *acc.get_unchecked(idx) };
        if dist < candidate_dist {
            candidate_dist = dist;
            candidate_idx = idx;
            improved = true;
        }
    }

    if improved {
        *best_dist = candidate_dist;
        *best_item = unsafe { *items.get_unchecked(candidate_idx) };
    }
}

#[cfg(test)]
mod tests {
    use super::try_nearest_one_with_query_wide_v3;
    use crate::dist::{Manhattan, SquaredEuclidean};
    use crate::leaf_view::LeafView;

    #[test]
    fn v3_scalar_nearest_one_f64_squared_euclidean() {
        let x = [0.0f64, 2.0, 5.0, -1.0];
        let y = [0.0f64, 1.0, 3.0, 7.0];
        let items = [10u32, 11u32, 12u32, 13u32];
        let leaf = LeafView::<f64, u32, 2, 32>::new([&x, &y], &items);

        let query = [1.8f64, 1.2f64];
        let mut best_dist = f64::INFINITY;
        let mut best_item = 0u32;

        let handled = try_nearest_one_with_query_wide_v3::<f64, u32, SquaredEuclidean<f64>, 2, 32>(
            &leaf,
            &query,
            &mut best_dist,
            &mut best_item,
        );

        assert!(handled);
        assert_eq!(best_item, 11);
        assert!((best_dist - 0.08).abs() < 1e-12);
    }

    #[test]
    fn v3_scalar_nearest_one_f32_manhattan() {
        let x = [0.0f32, 2.0, 5.0, -1.0];
        let y = [0.0f32, 1.0, 3.0, 7.0];
        let items = [20u32, 21u32, 22u32, 23u32];
        let leaf = LeafView::<f32, u32, 2, 32>::new([&x, &y], &items);

        let query = [1.8f32, 1.2f32];
        let mut best_dist = f32::INFINITY;
        let mut best_item = 0u32;

        let handled = try_nearest_one_with_query_wide_v3::<f32, u32, Manhattan<f32>, 2, 32>(
            &leaf,
            &query,
            &mut best_dist,
            &mut best_item,
        );

        assert!(handled);
        assert_eq!(best_item, 21);
        assert!((best_dist - 0.4).abs() < 1e-6);
    }
}

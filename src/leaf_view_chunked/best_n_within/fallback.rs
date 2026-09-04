use crate::dist::DistanceMetric;
use crate::leaf_view::{LeafArena, LeafView, TlsLeafScratch};
use crate::results::result_collection::BestNeighbourResultCollection;
use crate::{Axis, BestQueryResultItem, Content};

#[inline(always)]
pub(crate) fn best_n_within_with_query_wide_fallback<
    AX,
    T,
    D,
    R,
    const EXCLUSIVE: bool,
    const K: usize,
    const B: usize,
>(
    leaf: &LeafView<'_, AX, T, K, B>,
    query_wide: &[D::Output; K],
    dist: D::Output,
    threshold_item: Option<T>,
    results: &mut R,
) where
    AX: Axis<Coord = AX> + 'static,
    T: Content + PartialOrd,
    D: DistanceMetric<AX>,
    D::Output: Axis<Coord = D::Output> + TlsLeafScratch + 'static,
    R: BestNeighbourResultCollection<D::Output, T>,
{
    leaf.with_dists_for_slice_wide::<D, _>(query_wide, |dists| {
        LeafView::<AX, T, K, B>::update_best_dists::<_, _, EXCLUSIVE>(
            dists,
            leaf.items(),
            dist,
            threshold_item,
            results,
        );
    });
}

#[inline(always)]
pub(crate) fn best_n_within_with_query_wide_arena_fallback<
    AX,
    T,
    D,
    R,
    const EXCLUSIVE: bool,
    const K: usize,
>(
    arena: &LeafArena<'_, AX, T, K>,
    query_wide: &[D::Output; K],
    dist: D::Output,
    threshold_item: Option<T>,
    results: &mut R,
) where
    AX: Axis<Coord = AX> + 'static,
    T: Content + PartialOrd,
    D: DistanceMetric<AX>,
    D::Output: Axis<Coord = D::Output> + 'static,
    R: BestNeighbourResultCollection<D::Output, T>,
{
    if arena.is_empty() {
        return;
    }

    arena.for_each_tiled_chunk(|tile| {
        for idx in 0..tile.len() {
            let mut candidate_dist = D::Output::zero();

            for dim in 0..K {
                let coord = unsafe { tile.point_unaligned(dim, idx) };
                D::combine_component(
                    &mut candidate_dist,
                    D::dist1(D::widen_coord(coord), unsafe {
                        *query_wide.get_unchecked(dim)
                    }),
                );
            }

            let is_within_dist = if EXCLUSIVE {
                candidate_dist < dist
            } else {
                candidate_dist <= dist
            };

            if is_within_dist {
                let item = unsafe { tile.item_unaligned(idx) };
                if threshold_item.is_some_and(|worst_item| item >= worst_item) {
                    #[cfg(feature = "result_collection_stats")]
                    crate::results::result_collection_stats::record_best_item_threshold_reject();
                    continue;
                }
                #[cfg(feature = "result_collection_stats")]
                crate::results::result_collection_stats::record_candidate_emitted();

                let candidate = BestQueryResultItem {
                    point: (),
                    distance: candidate_dist,
                    item,
                };

                results.add(candidate);
            }
        }
    });
}

#[inline(always)]
pub(crate) fn best_n_within_with_query_wide_item_sorted_fallback<
    AX,
    T,
    D,
    R,
    const EXCLUSIVE: bool,
    const K: usize,
    const B: usize,
>(
    leaf: &LeafView<'_, AX, T, K, B>,
    query_wide: &[D::Output; K],
    dist: D::Output,
    results: &mut R,
) where
    AX: Axis<Coord = AX> + 'static,
    T: Content + PartialOrd,
    D: DistanceMetric<AX>,
    D::Output: Axis<Coord = D::Output> + 'static,
    R: BestNeighbourResultCollection<D::Output, T>,
{
    let points = leaf.points();
    let items = leaf.items();
    let mut idx = 0usize;

    while idx < items.len() {
        let item = unsafe { *items.get_unchecked(idx) };
        if results
            .threshold_item()
            .is_some_and(|worst_item| item >= worst_item)
        {
            #[cfg(feature = "result_collection_stats")]
            crate::results::result_collection_stats::record_best_item_threshold_reject();
            break;
        }

        let mut candidate_dist = D::Output::zero();
        for dim in 0..K {
            let coord = unsafe { *points.get_unchecked(dim).get_unchecked(idx) };
            D::combine_component(
                &mut candidate_dist,
                D::dist1(D::widen_coord(coord), unsafe {
                    *query_wide.get_unchecked(dim)
                }),
            );
        }

        let is_within_dist = if EXCLUSIVE {
            candidate_dist < dist
        } else {
            candidate_dist <= dist
        };
        if is_within_dist {
            #[cfg(feature = "result_collection_stats")]
            crate::results::result_collection_stats::record_candidate_emitted();
            results.add(BestQueryResultItem {
                point: (),
                distance: candidate_dist,
                item,
            });
        }

        idx += 1;
    }
}

#[inline(always)]
pub(crate) fn best_n_within_with_query_wide_arena_item_sorted_fallback<
    AX,
    T,
    D,
    R,
    const EXCLUSIVE: bool,
    const K: usize,
>(
    arena: &LeafArena<'_, AX, T, K>,
    query_wide: &[D::Output; K],
    dist: D::Output,
    results: &mut R,
) where
    AX: Axis<Coord = AX> + 'static,
    T: Content + PartialOrd,
    D: DistanceMetric<AX>,
    D::Output: Axis<Coord = D::Output> + 'static,
    R: BestNeighbourResultCollection<D::Output, T>,
{
    let mut finished = false;

    arena.for_each_tiled_chunk(|tile| {
        if finished {
            return;
        }

        let mut idx = 0usize;
        while idx < tile.len() {
            let item = unsafe { tile.item_unaligned(idx) };
            if results
                .threshold_item()
                .is_some_and(|worst_item| item >= worst_item)
            {
                #[cfg(feature = "result_collection_stats")]
                crate::results::result_collection_stats::record_best_item_threshold_reject();
                finished = true;
                break;
            }

            let mut candidate_dist = D::Output::zero();
            for dim in 0..K {
                let coord = unsafe { tile.point_unaligned(dim, idx) };
                D::combine_component(
                    &mut candidate_dist,
                    D::dist1(D::widen_coord(coord), unsafe {
                        *query_wide.get_unchecked(dim)
                    }),
                );
            }

            let is_within_dist = if EXCLUSIVE {
                candidate_dist < dist
            } else {
                candidate_dist <= dist
            };
            if is_within_dist {
                #[cfg(feature = "result_collection_stats")]
                crate::results::result_collection_stats::record_candidate_emitted();
                results.add(BestQueryResultItem {
                    point: (),
                    distance: candidate_dist,
                    item,
                });
            }

            idx += 1;
        }
    });
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::dist::SquaredEuclidean;
    use crate::results::result_collection::ResultCollection;

    struct FixedItemThresholdResults {
        threshold_calls: Cell<usize>,
        added: Vec<BestQueryResultItem<(), u32, f64>>,
    }

    impl ResultCollection<f64, BestQueryResultItem<(), u32, f64>> for FixedItemThresholdResults {
        fn with_max_qty(_max_qty: usize) -> Self {
            Self {
                threshold_calls: Cell::new(0),
                added: Vec::new(),
            }
        }

        fn max_qty(&self) -> usize {
            1
        }

        fn len(&self) -> usize {
            self.added.len()
        }

        fn add(&mut self, entry: BestQueryResultItem<(), u32, f64>) {
            self.added.push(entry);
        }

        fn threshold_distance(&self) -> Option<f64> {
            None
        }

        fn into_vec(self) -> Vec<BestQueryResultItem<(), u32, f64>> {
            self.added
        }

        fn into_sorted_vec(self) -> Vec<BestQueryResultItem<(), u32, f64>> {
            self.added
        }
    }

    impl BestNeighbourResultCollection<f64, u32> for FixedItemThresholdResults {
        fn threshold_item(&self) -> Option<u32> {
            self.threshold_calls.set(self.threshold_calls.get() + 1);
            Some(2)
        }
    }

    fn new_fixed_threshold_results() -> FixedItemThresholdResults {
        FixedItemThresholdResults::with_max_qty(1)
    }

    #[test]
    fn item_sorted_leaf_view_kernel_stops_at_current_worst_item() {
        let points = [[0.0f64, 1.0, 2.0, 3.0]];
        let items = [1u32, 2, 3, 4];
        let leaf = LeafView::<f64, u32, 1, 8>::new([&points[0]], &items);
        let mut results = new_fixed_threshold_results();

        best_n_within_with_query_wide_item_sorted_fallback::<
            f64,
            u32,
            SquaredEuclidean<f64>,
            _,
            false,
            1,
            8,
        >(&leaf, &[0.0], f64::MAX, &mut results);

        assert_eq!(results.threshold_calls.get(), 2);
        assert_eq!(results.added.len(), 1);
        assert_eq!(results.added[0].item, 1);
    }

    #[test]
    fn item_sorted_leaf_arena_kernel_stops_at_current_worst_item() {
        let points = [0.0f64, 1.0, 2.0, 3.0];
        let items = [1u32, 2, 3, 4];
        let mut bytes = Vec::new();
        unsafe {
            bytes.extend_from_slice(std::slice::from_raw_parts(
                points.as_ptr().cast::<u8>(),
                std::mem::size_of_val(&points),
            ));
            bytes.extend_from_slice(std::slice::from_raw_parts(
                items.as_ptr().cast::<u8>(),
                std::mem::size_of_val(&items),
            ));
        }
        let arena = LeafArena::<f64, u32, 1>::new(bytes.as_ptr(), items.len());
        let mut results = new_fixed_threshold_results();

        best_n_within_with_query_wide_arena_item_sorted_fallback::<
            f64,
            u32,
            SquaredEuclidean<f64>,
            _,
            false,
            1,
        >(&arena, &[0.0], f64::MAX, &mut results);

        assert_eq!(results.threshold_calls.get(), 2);
        assert_eq!(results.added.len(), 1);
        assert_eq!(results.added[0].item, 1);
    }
}

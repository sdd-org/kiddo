//! Exact batched nearest-one traversal shared by tree-to-tree consumers.

use super::cursor::{Cursor, JoinTree};
use super::metric::{JoinFloat, JoinMetric};
use super::traversal::Work;
use super::TreeNearestQueryResult;
use crate::{Content, StemStrategy};

#[derive(Clone, Copy)]
struct Candidate<L, R, O> {
    left: L,
    right: R,
    distance: O,
}

fn candidate_upper<L: Content, R: Content, O: JoinFloat, SL>(
    starts: &[usize],
    cursor: &Cursor<SL>,
    candidates: &[Candidate<L, R, O>],
) -> O {
    let start = starts[cursor.range.start];
    let end = starts[cursor.range.end];
    candidates[start..end]
        .iter()
        .fold(O::ZERO, |max, candidate| {
            if candidate.distance > max {
                candidate.distance
            } else {
                max
            }
        })
}

fn update_leaf<L, R, D, const K: usize>(
    left: &L,
    right: &R,
    left_leaf: usize,
    right_leaf: usize,
    starts: &[usize],
    candidates: &mut [Candidate<L::Item, R::Item, D::Output>],
) where
    L: JoinTree<K>,
    R: JoinTree<K, A = L::A>,
    D: JoinMetric<L::A>,
{
    let mut left_offset = 0;
    left.tiles(left_leaf, 0..usize::MAX, |left_tile| {
        right.tiles(right_leaf, 0..usize::MAX, |right_tile| {
            for i in 0..left_tile.len {
                let candidate = &mut candidates[starts[left_leaf] + left_offset + i];
                for j in 0..right_tile.len {
                    let mut distance = D::Output::ZERO;
                    for dim in 0..K {
                        D::combine_component(
                            &mut distance,
                            D::dist1(
                                D::widen_coord(left_tile.coord(dim, i)),
                                D::widen_coord(right_tile.coord(dim, j)),
                            ),
                        );
                    }
                    if distance < candidate.distance {
                        candidate.distance = distance;
                        candidate.right = right_tile.item(j);
                    }
                }
            }
        });
        left_offset += left_tile.len;
    });
}

/// Exact dual-tree all-nearest-neighbours traversal.
pub(super) fn execute<L, R, D, const K: usize>(
    left: &L,
    right: &R,
) -> Vec<TreeNearestQueryResult<L::Item, R::Item, D::Output>>
where
    L: JoinTree<K>,
    R: JoinTree<K, A = L::A>,
    D: JoinMetric<L::A>,
{
    if left.size() == 0 || right.size() == 0 {
        return Vec::new();
    }

    let mut starts = Vec::with_capacity(left.leaf_count() + 1);
    starts.push(0);
    for leaf in 0..left.leaf_count() {
        starts.push(starts[leaf] + left.leaf_len(leaf));
    }
    let mut candidates = Vec::with_capacity(left.size());
    for leaf in 0..left.leaf_count() {
        left.tiles(leaf, 0..usize::MAX, |tile| {
            for i in 0..tile.len {
                candidates.push(Candidate {
                    left: tile.item(i),
                    right: R::Item::default(),
                    distance: D::Output::INFINITY,
                });
            }
        });
    }

    let mut work = vec![super::traversal::root::<L, R, D::Output, K>(left, right)];
    while let Some(Work {
        left: lc,
        right: rc,
        bounds,
        rows: _,
    }) = work.pop()
    {
        if lc.span() == 0 || rc.span() == 0 {
            continue;
        }
        // The box lower bound applies to every left entry in this node pair.
        // If it cannot improve even the worst current left candidate, it cannot
        // improve any candidate in the subtree.
        if bounds.lower::<L::A, D>() >= candidate_upper(&starts, &lc, &candidates) {
            continue;
        }
        if lc.leaf() && rc.leaf() {
            for left_leaf in lc.range.clone() {
                for right_leaf in rc.range.clone() {
                    update_leaf::<L, R, D, K>(
                        left,
                        right,
                        left_leaf,
                        right_leaf,
                        &starts,
                        &mut candidates,
                    );
                }
            }
            continue;
        }

        let side = usize::from(lc.leaf() || (!rc.leaf() && rc.span() > lc.span()));
        let (dim, pivot) = if side == 0 {
            (
                lc.strategy.construction_dim::<K>(),
                D::widen_coord(left.stems()[lc.strategy.stem_idx()]),
            )
        } else {
            (
                rc.strategy.construction_dim::<K>(),
                D::widen_coord(right.stems()[rc.strategy.stem_idx()]),
            )
        };
        let padding =
            (side == 0 && lc.padding_remaining > 0) || (side == 1 && rc.padding_remaining > 0);
        if !pivot.is_finite() && padding {
            work.push(Work {
                left: if side == 0 {
                    lc.descend_padding::<L::A, K>()
                } else {
                    lc
                },
                right: if side == 1 {
                    rc.descend_padding::<L::A, K>()
                } else {
                    rc
                },
                bounds,
                rows: [0..usize::MAX, 0..usize::MAX],
            });
            continue;
        }

        let (near_left, near_right, far_left, far_right) = if side == 0 {
            let (near, far) = lc.split::<L::A, K>();
            (near, rc.clone(), far, rc)
        } else {
            let (near, far) = rc.split::<L::A, K>();
            (lc.clone(), near, lc, far)
        };
        let mut near_bounds = bounds;
        if pivot.is_finite() && pivot < near_bounds.max[side][dim] {
            near_bounds.max[side][dim] = pivot;
        }
        work.push(Work {
            left: near_left,
            right: near_right,
            bounds: near_bounds,
            rows: [0..usize::MAX, 0..usize::MAX],
        });
        // A real infinite pivot routes all entries to the left child.
        if pivot.is_finite() {
            let mut far_bounds = bounds;
            if pivot > far_bounds.min[side][dim] {
                far_bounds.min[side][dim] = pivot;
            }
            work.push(Work {
                left: far_left,
                right: far_right,
                bounds: far_bounds,
                rows: [0..usize::MAX, 0..usize::MAX],
            });
        }
    }

    candidates
        .into_iter()
        .map(|candidate| TreeNearestQueryResult {
            left_item: candidate.left,
            right_item: candidate.right,
            distance: candidate.distance,
        })
        .collect()
}

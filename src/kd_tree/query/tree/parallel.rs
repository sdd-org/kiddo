//! Bounded geometric frontier; all task payloads own their traversal state.
use super::cursor::JoinTree;
use super::metric::{JoinFloat, JoinMetric};
use super::traversal::{rejected, root, Work};
use crate::StemStrategy;

type Task<L, R, D, const K: usize> = Work<
    <L as JoinTree<K>>::Strategy,
    <R as JoinTree<K>>::Strategy,
    <D as crate::dist::DistanceMetricScalar<<L as JoinTree<K>>::A>>::Output,
    K,
>;

pub(super) fn frontier<L, R, D, const K: usize, const EX: bool>(
    left: &L,
    right: &R,
    radius: D::Output,
    budget: usize,
) -> Vec<Task<L, R, D, K>>
where
    L: JoinTree<K>,
    R: JoinTree<K, A = L::A>,
    D: JoinMetric<L::A>,
{
    let mut tasks = vec![root(left, right)];
    while tasks.len() < budget {
        let best = tasks
            .iter()
            .enumerate()
            .filter_map(|(i, w)| {
                let l = if w.left.leaf() {
                    left.leaf_len(w.left.range.start)
                        .min(w.rows[0].end)
                        .saturating_sub(w.rows[0].start)
                } else {
                    w.left
                        .span()
                        .saturating_mul(left.max_leaf_len())
                        .min(left.size())
                };
                let r = if w.right.leaf() {
                    right
                        .leaf_len(w.right.range.start)
                        .min(w.rows[1].end)
                        .saturating_sub(w.rows[1].start)
                } else {
                    w.right
                        .span()
                        .saturating_mul(right.max_leaf_len())
                        .min(right.size())
                };
                let score = l.saturating_mul(r);
                (score > 4096).then_some((i, score))
            })
            .max_by_key(|&(_, score)| score);
        let Some((index, _)) = best else {
            break;
        };
        let mut a = tasks.swap_remove(index);
        if rejected::<L::A, D, K, EX>(&a.bounds, radius) {
            continue;
        }
        let mut b = a.clone();
        if a.left.leaf() && a.right.leaf() {
            a.rows[0].end = a.rows[0].end.min(left.leaf_len(a.left.range.start));
            a.rows[1].end = a.rows[1].end.min(right.leaf_len(a.right.range.start));
            b.rows = a.rows.clone();
            let side = usize::from(a.rows[1].len() > a.rows[0].len());
            let len = a.rows[side].len();
            let mid = a.rows[side].start + ((len / 2).div_ceil(32) * 32).min(len - 1);
            a.rows[side].end = mid;
            b.rows[side].start = mid;
        } else {
            let side =
                usize::from(a.left.leaf() || (!a.right.leaf() && a.right.span() > a.left.span()));
            let (dim, pivot) = if side == 0 {
                let dim = a.left.strategy.construction_dim::<K>();
                let pivot = D::widen_coord(left.stems()[a.left.strategy.stem_idx()]);
                (dim, pivot)
            } else {
                let dim = a.right.strategy.construction_dim::<K>();
                let pivot = D::widen_coord(right.stems()[a.right.strategy.stem_idx()]);
                (dim, pivot)
            };
            let padding = (side == 0 && a.left.padding_remaining > 0)
                || (side == 1 && a.right.padding_remaining > 0);
            if !pivot.is_finite() && padding {
                if side == 0 {
                    a.left = a.left.descend_padding::<L::A, K>();
                } else {
                    a.right = a.right.descend_padding::<L::A, K>();
                }
                if a.left.span() > 0 && a.right.span() > 0 {
                    tasks.push(a);
                }
                continue;
            }
            if side == 0 {
                (a.left, b.left) = a.left.split::<L::A, K>();
            } else {
                (a.right, b.right) = a.right.split::<L::A, K>();
            }
            // A real infinite pivot denotes a degenerate split.  Unlike the
            // synthetic root padding handled above, every point belongs to
            // the left child, so the right child must not be scheduled.
            if !pivot.is_finite() {
                if a.left.span() > 0
                    && a.right.span() > 0
                    && !rejected::<L::A, D, K, EX>(&a.bounds, radius)
                {
                    tasks.push(a);
                }
                continue;
            }
            if pivot.is_finite() && pivot < a.bounds.max[side][dim] {
                a.bounds.max[side][dim] = pivot;
            }
            if pivot.is_finite() && pivot > b.bounds.min[side][dim] {
                b.bounds.min[side][dim] = pivot;
            }
        }
        for task in [a, b] {
            if task.left.span() > 0
                && task.right.span() > 0
                && !rejected::<L::A, D, K, EX>(&task.bounds, radius)
            {
                tasks.push(task);
            }
        }
    }
    #[cfg(feature = "exact_query_stats")]
    super::stats::record(super::stats::Event::Tasks, tasks.len());
    tasks
}

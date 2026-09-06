use super::cursor::{Cursor, JoinTree};
use super::leaf::leaf_pair;
use super::metric::{accepts, Bounds, JoinFloat, JoinMetric};
use crate::{Content, StemStrategy};
use std::ops::Range;

pub(super) const COUNT_OVERFLOW: &str = "tree join pair count exceeds usize::MAX";

pub(super) trait Sink<L: Content, R: Content, O: JoinFloat> {
    const COUNT: bool = false;
    const DISTANCES: bool;
    fn emit(&mut self, left: L, right: R, distance: O);
    fn add_count(&mut self, _count: usize) {
        unreachable!()
    }
}

#[derive(Default)]
pub(super) struct Counter(pub usize);
impl<L: Content, R: Content, O: JoinFloat> Sink<L, R, O> for Counter {
    const COUNT: bool = true;
    const DISTANCES: bool = false;
    fn emit(&mut self, _: L, _: R, _: O) {
        unreachable!()
    }
    fn add_count(&mut self, count: usize) {
        self.0 = self.0.checked_add(count).expect(COUNT_OVERFLOW);
    }
}

#[derive(Clone)]
pub(super) struct Work<SL, SR, O, const K: usize> {
    pub left: Cursor<SL>,
    pub right: Cursor<SR>,
    pub bounds: Bounds<O, K>,
    pub rows: [Range<usize>; 2],
}

pub(super) fn root<L: JoinTree<K>, R: JoinTree<K, A = L::A>, O: JoinFloat, const K: usize>(
    left: &L,
    right: &R,
) -> Work<L::Strategy, R::Strategy, O, K> {
    Work {
        left: Cursor::root(left),
        right: Cursor::root(right),
        bounds: Bounds::new(),
        rows: [0..usize::MAX, 0..usize::MAX],
    }
}

pub(super) fn rejected<A: JoinFloat, D: JoinMetric<A>, const K: usize, const EX: bool>(
    bounds: &Bounds<D::Output, K>,
    radius: D::Output,
) -> bool {
    let lower = bounds.lower::<A, D>();
    // NaN is uncertainty, never permission to prune.
    if EX {
        lower >= radius
    } else {
        lower > radius
    }
}

enum Frame<SL, SR, O> {
    Right {
        left: Cursor<SL>,
        right: Cursor<SR>,
        side: usize,
        dim: usize,
        min: O,
        max: O,
        pivot: O,
    },
    Restore {
        side: usize,
        dim: usize,
        min: O,
        max: O,
    },
}

struct Stack<T, const N: usize> {
    inline: [Option<T>; N],
    len: usize,
    spill: Vec<T>,
}
impl<T, const N: usize> Stack<T, N> {
    fn new() -> Self {
        Self {
            inline: std::array::from_fn(|_| None),
            len: 0,
            spill: Vec::new(),
        }
    }
    fn push(&mut self, value: T) {
        if self.len < N && self.spill.is_empty() {
            self.inline[self.len] = Some(value);
            self.len += 1;
        } else {
            #[cfg(feature = "exact_query_stats")]
            super::stats::record(super::stats::Event::Spill, 1);
            self.spill.push(value);
        }
    }
    fn pop(&mut self) -> Option<T> {
        if let Some(x) = self.spill.pop() {
            return Some(x);
        }
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        self.inline[self.len].take()
    }
}

pub(super) fn walk<L, R, D, S, const K: usize, const EX: bool>(
    left: &L,
    right: &R,
    work: Work<L::Strategy, R::Strategy, D::Output, K>,
    radius: D::Output,
    sink: &mut S,
) where
    L: JoinTree<K>,
    R: JoinTree<K, A = L::A>,
    D: JoinMetric<L::A>,
    S: Sink<L::Item, R::Item, D::Output>,
{
    let Work {
        left: mut lc,
        right: mut rc,
        mut bounds,
        rows,
    } = work;
    let mut stack = Stack::<Frame<L::Strategy, R::Strategy, D::Output>, 64>::new();
    loop {
        #[cfg(feature = "exact_query_stats")]
        super::stats::record(super::stats::Event::Node, 1);
        if lc.span() > 0 && rc.span() > 0 && !rejected::<L::A, D, K, EX>(&bounds, radius) {
            if !S::DISTANCES && accepts::<_, EX>(bounds.upper::<L::A, D>(), radius) {
                bulk::<L, R, D, S, K>(left, right, &lc, &rc, &rows, sink);
            } else if lc.leaf() && rc.leaf() {
                leaf_pair::<L, R, D, S, K, EX>(
                    left,
                    right,
                    lc.range.start,
                    rc.range.start,
                    &rows,
                    radius,
                    sink,
                );
            } else {
                let side = usize::from(lc.leaf() || (!rc.leaf() && rc.span() > lc.span()));
                let (dim, pivot) = if side == 0 {
                    let dim = lc.strategy.construction_dim::<K>();
                    let pivot = D::widen_coord(left.stems()[lc.strategy.stem_idx()]);
                    (dim, pivot)
                } else {
                    let dim = rc.strategy.construction_dim::<K>();
                    let pivot = D::widen_coord(right.stems()[rc.strategy.stem_idx()]);
                    (dim, pivot)
                };
                if !pivot.is_finite()
                    && ((side == 0 && lc.padding_remaining > 0)
                        || (side == 1 && rc.padding_remaining > 0))
                {
                    // Construction puts block-alignment padding above the
                    // true root. It must not halve the logical leaf range.
                    if side == 0 {
                        lc = lc.descend_padding::<L::A, K>();
                    } else {
                        rc = rc.descend_padding::<L::A, K>();
                    }
                    continue;
                }
                let (next_l, next_r, far_l, far_r) = if side == 0 {
                    let (near, far) = lc.split::<L::A, K>();
                    (near, rc.clone(), far, rc)
                } else {
                    let (near, far) = rc.split::<L::A, K>();
                    (lc.clone(), near, lc, far)
                };
                let min = bounds.min[side][dim];
                let max = bounds.max[side][dim];
                if pivot.is_finite() {
                    stack.push(Frame::Right {
                        left: far_l,
                        right: far_r,
                        side,
                        dim,
                        min,
                        max,
                        pivot,
                    });
                    bounds.max[side][dim] = if pivot < max { pivot } else { max };
                }
                lc = next_l;
                rc = next_r;
                continue;
            }
        } else {
            #[cfg(feature = "exact_query_stats")]
            super::stats::record(super::stats::Event::Rejected, 1);
        }
        loop {
            match stack.pop() {
                None => return,
                Some(Frame::Restore {
                    side,
                    dim,
                    min,
                    max,
                }) => {
                    bounds.min[side][dim] = min;
                    bounds.max[side][dim] = max;
                }
                Some(Frame::Right {
                    left,
                    right,
                    side,
                    dim,
                    min,
                    max,
                    pivot,
                }) => {
                    bounds.min[side][dim] = if pivot > min { pivot } else { min };
                    bounds.max[side][dim] = max;
                    stack.push(Frame::Restore {
                        side,
                        dim,
                        min,
                        max,
                    });
                    lc = left;
                    rc = right;
                    break;
                }
            }
        }
    }
}

fn cardinality<Tree: JoinTree<K>, const K: usize>(
    tree: &Tree,
    leaves: Range<usize>,
    rows: &Range<usize>,
) -> usize {
    leaves
        .map(|i| tree.leaf_len(i).min(rows.end).saturating_sub(rows.start))
        .try_fold(0usize, usize::checked_add)
        .expect(COUNT_OVERFLOW)
}

fn bulk<L, R, D, S, const K: usize>(
    left: &L,
    right: &R,
    lc: &Cursor<L::Strategy>,
    rc: &Cursor<R::Strategy>,
    rows: &[Range<usize>; 2],
    sink: &mut S,
) where
    L: JoinTree<K>,
    R: JoinTree<K, A = L::A>,
    D: JoinMetric<L::A>,
    S: Sink<L::Item, R::Item, D::Output>,
{
    #[cfg(feature = "exact_query_stats")]
    {
        super::stats::record(super::stats::Event::Accepted, 1);
        let count = cardinality(left, lc.range.clone(), &rows[0])
            .checked_mul(cardinality(right, rc.range.clone(), &rows[1]))
            .expect(COUNT_OVERFLOW);
        super::stats::record(super::stats::Event::Matches, count);
    }
    if S::COUNT {
        let l = cardinality(left, lc.range.clone(), &rows[0]);
        let r = cardinality(right, rc.range.clone(), &rows[1]);
        sink.add_count(l.checked_mul(r).expect(COUNT_OVERFLOW));
        return;
    }
    for li in lc.range.clone() {
        left.tiles(li, rows[0].clone(), |lt| {
            for ri in rc.range.clone() {
                right.tiles(ri, rows[1].clone(), |rt| {
                    for i in 0..lt.len {
                        for j in 0..rt.len {
                            sink.emit(lt.item(i), rt.item(j), D::Output::ZERO);
                        }
                    }
                });
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stack_spills_and_reuses() {
        let mut stack = Stack::<_, 2>::new();
        for i in 0..8 {
            stack.push(i);
        }
        for i in (0..8).rev() {
            assert_eq!(stack.pop(), Some(i));
        }
        assert_eq!(stack.pop(), None);
        stack.push(42);
        assert_eq!(stack.pop(), Some(42));
    }
    #[test]
    #[should_panic(expected = "tree join pair count exceeds usize::MAX")]
    fn count_overflow() {
        let mut count = Counter(usize::MAX);
        <Counter as Sink<u32, u32, f64>>::add_count(&mut count, 1);
    }
}

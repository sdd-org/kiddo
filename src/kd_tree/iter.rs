use std::mem::MaybeUninit;
use std::ptr::NonNull;

use crate::dist::DistanceMetric;
use crate::leaf_view::TlsLeafScratch;
use crate::leaf_view_chunked::nearest_n_within::{
    nearest_n_within_with_query_wide, nearest_n_within_with_query_wide_arena,
};
use crate::results::result_collection::ResultCollection;
use crate::traits::leaf_strategy::LeafProjection;
use crate::{Axis, Content, LeafStrategy, QueryResultItem, StemStrategy};

use super::{KdTreeAccessor, StemLeafResolution};

const WITHIN_UNSORTED_ITER_INLINE_STACK_CAPACITY: usize = 64;
const WITHIN_UNSORTED_ITER_INLINE_RESULT_CAPACITY: usize = 64;

/// Iterator over all point/item pairs in a kd-tree.
pub struct KdTreeIter<'a, Tree, A, T, SS, LS, const K: usize, const B: usize>
where
    A: Axis<Coord = A>,
    T: Content,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: KdTreeAccessor<A, T, SS, LS, K, B>,
{
    tree: &'a Tree,
    leaf_idx: usize,
    pos_in_leaf: usize,
    yielded: usize,
    _phantom: std::marker::PhantomData<(A, T, SS, LS)>,
}

impl<'a, Tree, A, T, SS, LS, const K: usize, const B: usize>
    KdTreeIter<'a, Tree, A, T, SS, LS, K, B>
where
    A: Axis<Coord = A>,
    T: Content,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: KdTreeAccessor<A, T, SS, LS, K, B>,
{
    #[inline]
    pub(crate) fn new(tree: &'a Tree) -> Self {
        Self {
            tree,
            leaf_idx: 0,
            pos_in_leaf: 0,
            yielded: 0,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<Tree, A, T, SS, LS, const K: usize, const B: usize> Iterator
    for KdTreeIter<'_, Tree, A, T, SS, LS, K, B>
where
    A: Axis<Coord = A>,
    T: Content,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: KdTreeAccessor<A, T, SS, LS, K, B>,
{
    type Item = (T, [A; K]);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.leaf_idx < self.tree.leaf_count() {
            let leaf_len = self.tree.leaves().leaf_len(self.leaf_idx);
            if self.pos_in_leaf < leaf_len {
                let (point, item) = self
                    .tree
                    .leaves()
                    .leaf_point_item(self.leaf_idx, self.pos_in_leaf);
                self.pos_in_leaf += 1;
                self.yielded += 1;
                return Some((item, point));
            }

            self.leaf_idx += 1;
            self.pos_in_leaf = 0;
        }

        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.tree.size().saturating_sub(self.yielded);
        (remaining, Some(remaining))
    }
}

impl<Tree, A, T, SS, LS, const K: usize, const B: usize> ExactSizeIterator
    for KdTreeIter<'_, Tree, A, T, SS, LS, K, B>
where
    A: Axis<Coord = A>,
    T: Content,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: KdTreeAccessor<A, T, SS, LS, K, B>,
{
}

struct TraversalFrame<SS, O, const K: usize> {
    stem_strat: SS,
    off: [O; K],
    rd: O,
}

struct InlineStack<T, const N: usize> {
    inline: [MaybeUninit<T>; N],
    inline_len: usize,
    spill: Vec<T>,
}

impl<T, const N: usize> InlineStack<T, N> {
    #[inline]
    fn new() -> Self {
        Self {
            inline: [const { MaybeUninit::uninit() }; N],
            inline_len: 0,
            spill: Vec::new(),
        }
    }

    #[inline]
    fn push(&mut self, value: T) {
        if self.spill.is_empty() && self.inline_len < N {
            unsafe { self.inline.get_unchecked_mut(self.inline_len) }.write(value);
            self.inline_len += 1;
        } else {
            self.spill.push(value);
        }
    }

    #[inline]
    fn pop(&mut self) -> Option<T> {
        if let Some(value) = self.spill.pop() {
            return Some(value);
        }

        if self.inline_len == 0 {
            None
        } else {
            self.inline_len -= 1;
            Some(unsafe {
                self.inline
                    .get_unchecked(self.inline_len)
                    .assume_init_read()
            })
        }
    }
}

impl<T, const N: usize> Drop for InlineStack<T, N> {
    fn drop(&mut self) {
        while self.inline_len != 0 {
            self.inline_len -= 1;
            unsafe {
                self.inline
                    .get_unchecked_mut(self.inline_len)
                    .assume_init_drop()
            };
        }
    }
}

struct InlineResultBuffer<E, const N: usize> {
    inline: [MaybeUninit<E>; N],
    inline_len: usize,
    spill: Vec<E>,
}

impl<E, const N: usize> InlineResultBuffer<E, N> {
    #[inline]
    fn new() -> Self {
        Self {
            inline: [const { MaybeUninit::uninit() }; N],
            inline_len: 0,
            spill: Vec::new(),
        }
    }

    #[inline]
    fn clear(&mut self) {
        while self.inline_len != 0 {
            self.inline_len -= 1;
            unsafe {
                self.inline
                    .get_unchecked_mut(self.inline_len)
                    .assume_init_drop()
            };
        }
        self.spill.clear();
    }

    #[inline]
    fn len(&self) -> usize {
        self.inline_len + self.spill.len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<E: Copy, const N: usize> InlineResultBuffer<E, N> {
    #[inline]
    fn push(&mut self, value: E) {
        if self.spill.is_empty() && self.inline_len < N {
            unsafe { self.inline.get_unchecked_mut(self.inline_len) }.write(value);
            self.inline_len += 1;
        } else {
            self.spill.push(value);
        }
    }

    #[inline]
    fn get(&self, idx: usize) -> E {
        if idx < self.inline_len {
            unsafe { self.inline.get_unchecked(idx).assume_init_read() }
        } else {
            unsafe { *self.spill.get_unchecked(idx - self.inline_len) }
        }
    }
}

impl<E, const N: usize> Drop for InlineResultBuffer<E, N> {
    fn drop(&mut self) {
        self.clear();
    }
}

impl<O, E, const N: usize> ResultCollection<O, E> for InlineResultBuffer<E, N>
where
    O: Axis<Coord = O>,
    E: Copy + Ord,
{
    #[inline(always)]
    fn with_max_qty(_max_qty: usize) -> Self {
        Self::new()
    }

    #[inline(always)]
    fn max_qty(&self) -> usize {
        usize::MAX
    }

    #[inline(always)]
    fn len(&self) -> usize {
        InlineResultBuffer::len(self)
    }

    #[inline(always)]
    fn add(&mut self, entry: E) {
        self.push(entry);
    }

    #[inline(always)]
    fn threshold_distance(&self) -> Option<O> {
        None
    }

    #[inline]
    fn into_vec(self) -> Vec<E> {
        self.into_vec_unsorted()
    }

    #[inline]
    fn into_sorted_vec(self) -> Vec<E> {
        let mut result = self.into_vec_unsorted();
        result.sort_unstable();
        result
    }
}

impl<E: Copy, const N: usize> InlineResultBuffer<E, N> {
    #[inline]
    fn into_vec_unsorted(mut self) -> Vec<E> {
        let mut result = Vec::with_capacity(self.len());
        for idx in 0..self.inline_len {
            result.push(unsafe { self.inline.get_unchecked(idx).assume_init_read() });
        }
        result.append(&mut self.spill);
        self.inline_len = 0;
        result
    }
}

/// Lazy iterator returned by `within_unsorted_iter`.
///
/// This is the ergonomic streaming API for callers who want to avoid materializing the full
/// result set. It keeps traversal state and per-leaf matches inline in the common case, spilling
/// to heap allocation only if the tree depth or a single leaf's match count exceeds the inline
/// capacities.
pub struct WithinUnsortedIter<
    'a,
    Tree,
    A,
    T,
    SS,
    LS,
    D,
    const EXCLUSIVE: bool,
    const K: usize,
    const B: usize,
> where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    D: DistanceMetric<A>,
    D::Output: Axis<Coord = D::Output> + TlsLeafScratch + 'static,
    Tree: KdTreeAccessor<A, T, SS, LS, K, B>,
{
    tree: &'a Tree,
    query: [A; K],
    query_wide: [D::Output; K],
    max_dist: D::Output,
    stack:
        InlineStack<TraversalFrame<SS, D::Output, K>, WITHIN_UNSORTED_ITER_INLINE_STACK_CAPACITY>,
    leaf_results: InlineResultBuffer<
        QueryResultItem<(), T, D::Output>,
        WITHIN_UNSORTED_ITER_INLINE_RESULT_CAPACITY,
    >,
    leaf_result_pos: usize,
    _phantom: std::marker::PhantomData<(T, LS, D)>,
}

impl<'a, Tree, A, T, SS, LS, D, const EXCLUSIVE: bool, const K: usize, const B: usize>
    WithinUnsortedIter<'a, Tree, A, T, SS, LS, D, EXCLUSIVE, K, B>
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    D: DistanceMetric<A>,
    D::Output: Axis<Coord = D::Output> + TlsLeafScratch + 'static,
    Tree: KdTreeAccessor<A, T, SS, LS, K, B>,
{
    #[inline]
    pub(crate) fn new(tree: &'a Tree, query: &[A; K], max_dist: D::Output) -> Self {
        let mut stack = InlineStack::new();
        if tree.size() != 0 {
            let stems_ptr = NonNull::new(tree.stems().as_ptr() as *mut u8).unwrap();
            stack.push(TraversalFrame {
                stem_strat: SS::new(stems_ptr),
                off: [D::Output::zero(); K],
                rd: D::Output::zero(),
            });
        }

        Self {
            tree,
            query: *query,
            query_wide: query.map(D::widen_coord),
            max_dist,
            stack,
            leaf_results: InlineResultBuffer::new(),
            leaf_result_pos: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    #[inline(always)]
    fn resolve_current_leaf(&self, stem_strat: &SS) -> Option<usize> {
        if self
            .tree
            .stem_leaf_resolution()
            .is_terminal_stem_idx(stem_strat.stem_idx())
        {
            Some(
                self.tree
                    .stem_leaf_resolution()
                    .resolve_terminal_stem_idx(stem_strat.stem_idx(), 0),
            )
        } else if stem_strat.level() > self.tree.max_stem_level() {
            Some(
                self.tree
                    .stem_leaf_resolution()
                    .resolve_terminal_stem_idx(stem_strat.stem_idx(), stem_strat.leaf_idx()),
            )
        } else {
            None
        }
    }

    #[inline(always)]
    fn load_leaf_results(&mut self, leaf_idx: usize) -> bool {
        self.leaf_results.clear();
        self.leaf_result_pos = 0;

        match LS::LEAF_PROJECTION {
            LeafProjection::LeafArena => {
                let arena = self.tree.leaves().leaf_arena(leaf_idx);
                nearest_n_within_with_query_wide_arena::<
                    A,
                    T,
                    D,
                    QueryResultItem<(), T, D::Output>,
                    _,
                    EXCLUSIVE,
                    K,
                >(
                    &arena,
                    &self.query_wide,
                    self.max_dist,
                    &mut self.leaf_results,
                );
            }
            LeafProjection::LeafView => {
                let leaf = self.tree.leaves().leaf_view(leaf_idx);
                nearest_n_within_with_query_wide::<
                    A,
                    T,
                    D,
                    QueryResultItem<(), T, D::Output>,
                    _,
                    EXCLUSIVE,
                    K,
                    B,
                >(
                    &leaf,
                    &self.query_wide,
                    self.max_dist,
                    &mut self.leaf_results,
                );
            }
        }

        !self.leaf_results.is_empty()
    }

    #[inline]
    fn load_next_non_empty_leaf(&mut self) -> bool {
        // TODO: specialize this traversal cursor for Donnelly Block SIMD so iterator traversal
        // follows the same block-at-once pruning path as the callback/materialized queries.
        while let Some(frame) = self.stack.pop() {
            let should_prune = if EXCLUSIVE {
                D::Output::cmp(frame.rd, self.max_dist) != std::cmp::Ordering::Less
            } else {
                D::Output::cmp(frame.rd, self.max_dist) == std::cmp::Ordering::Greater
            };

            if should_prune {
                continue;
            }

            let mut stem_strat = frame.stem_strat;
            let off = frame.off;
            let rd = frame.rd;

            loop {
                if let Some(leaf_idx) = self.resolve_current_leaf(&stem_strat) {
                    if self.load_leaf_results(leaf_idx) {
                        return true;
                    }
                    break;
                }

                let dim = stem_strat.dim::<K>();
                let stem_idx = stem_strat.stem_idx();
                let pivot = if stem_idx < self.tree.stems().len() {
                    unsafe { *self.tree.stems().get_unchecked(stem_idx) }
                } else {
                    A::max_value()
                };

                if pivot < A::max_value() {
                    let query_elem = unsafe { *self.query.get_unchecked(dim) };
                    let is_right_child = query_elem >= pivot;
                    let far_ctx = stem_strat.branch_relative::<A, K>(is_right_child);

                    let pivot_wide = D::widen_coord(pivot);
                    let query_elem_wide = unsafe { *self.query_wide.get_unchecked(dim) };
                    let new_off = D::Output::saturating_dist(query_elem_wide, pivot_wide);
                    let rd_far = D::rect_dist_after_update(rd, &off, dim, new_off);

                    let should_visit_far = if EXCLUSIVE {
                        D::Output::cmp(rd_far, self.max_dist) == std::cmp::Ordering::Less
                    } else {
                        D::Output::cmp(rd_far, self.max_dist) != std::cmp::Ordering::Greater
                    };

                    if should_visit_far {
                        let mut far_off = off;
                        unsafe { *far_off.get_unchecked_mut(dim) = new_off };
                        self.stack.push(TraversalFrame {
                            stem_strat: far_ctx,
                            off: far_off,
                            rd: rd_far,
                        });
                    }
                } else {
                    stem_strat.traverse::<A, K>(false);
                }
            }
        }

        false
    }
}

impl<Tree, A, T, SS, LS, D, const EXCLUSIVE: bool, const K: usize, const B: usize> Iterator
    for WithinUnsortedIter<'_, Tree, A, T, SS, LS, D, EXCLUSIVE, K, B>
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    D: DistanceMetric<A>,
    D::Output: Axis<Coord = D::Output> + TlsLeafScratch + 'static,
    Tree: KdTreeAccessor<A, T, SS, LS, K, B>,
{
    type Item = QueryResultItem<(), T, D::Output>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.leaf_result_pos < self.leaf_results.len() {
                let result = self.leaf_results.get(self.leaf_result_pos);
                self.leaf_result_pos += 1;
                return Some(result);
            }

            if !self.load_next_non_empty_leaf() {
                return None;
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

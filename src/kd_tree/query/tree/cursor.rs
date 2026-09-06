use std::marker::PhantomData;
use std::ops::Range;

use super::metric::JoinFloat;
use crate::kd_tree::{KdTreeAccessor, OwnedStemLeafResolution};
use crate::leaf_strategy::{FlatVec, VecOfArenas};
use crate::traits::leaf_strategy::{Immutable, LeafProjection};
use crate::{Content, KdTree, LeafStrategy, StemStrategy};

mod sealed {
    pub trait Storage {}
    impl<A, T, const K: usize, const B: usize> Storage for crate::leaf_strategy::FlatVec<A, T, K, B> {}
    impl<A, T, const K: usize, const B: usize> Storage
        for crate::leaf_strategy::VecOfArenas<A, T, K, B>
    {
    }
    pub trait Tree {}
    impl<A, T, SS, LS, const K: usize, const B: usize> Tree for crate::KdTree<A, T, SS, LS, K, B> {}
}

#[doc(hidden)]
pub trait JoinStorage: sealed::Storage {}
impl<A, T, const K: usize, const B: usize> JoinStorage for FlatVec<A, T, K, B> {}
impl<A, T, const K: usize, const B: usize> JoinStorage for VecOfArenas<A, T, K, B> {}

/// Borrowed columns of at most 32 points. Pointers may be unaligned.
#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct Tile<'a, A, T, const K: usize> {
    pub(super) columns: [*const A; K],
    items: *const T,
    pub(super) len: usize,
    _borrow: PhantomData<&'a (A, T)>,
}
impl<A: Copy, T: Copy, const K: usize> Tile<'_, A, T, K> {
    #[inline(always)]
    pub(super) fn coord(&self, dim: usize, i: usize) -> A {
        debug_assert!(dim < K && i < self.len);
        unsafe { self.columns[dim].add(i).read_unaligned() }
    }
    #[inline(always)]
    pub(super) fn item(&self, i: usize) -> T {
        debug_assert!(i < self.len);
        unsafe { self.items.add(i).read_unaligned() }
    }
}

/// Internal sealed access to implemented immutable storage.
#[doc(hidden)]
pub trait JoinTree<const K: usize>: sealed::Tree + Sync {
    type A: JoinFloat;
    type Item: Content;
    type Strategy: StemStrategy;
    fn stems(&self) -> &[Self::A];
    fn depth(&self) -> usize;
    fn max_stem_level(&self) -> i32;
    fn size(&self) -> usize;
    fn leaf_count(&self) -> usize;
    fn max_leaf_len(&self) -> usize;
    fn leaf_len(&self, leaf: usize) -> usize;
    fn tiles(
        &self,
        leaf: usize,
        rows: Range<usize>,
        f: impl FnMut(Tile<'_, Self::A, Self::Item, K>),
    );
}

impl<A, T, SS, LS, const K: usize, const B: usize> JoinTree<K> for KdTree<A, T, SS, LS, K, B>
where
    A: JoinFloat,
    T: Content,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B, Mutability = Immutable> + JoinStorage + Sync,
{
    type A = A;
    type Item = T;
    type Strategy = SS;
    fn stems(&self) -> &[A] {
        KdTreeAccessor::stems(self)
    }
    fn depth(&self) -> usize {
        match self.stem_leaf_resolution {
            OwnedStemLeafResolution::Arithmetic { stems_depth, .. } => stems_depth,
            _ => unreachable!("immutable join requires arithmetic leaf resolution"),
        }
    }
    fn size(&self) -> usize {
        self.size()
    }
    fn max_stem_level(&self) -> i32 {
        KdTreeAccessor::max_stem_level(self)
    }
    fn leaf_count(&self) -> usize {
        self.leaf_count()
    }
    fn max_leaf_len(&self) -> usize {
        self.max_leaf_len()
    }
    fn leaf_len(&self, leaf: usize) -> usize {
        self.leaves().leaf_len(leaf)
    }
    fn tiles(&self, leaf: usize, rows: Range<usize>, mut f: impl FnMut(Tile<'_, A, T, K>)) {
        match LS::LEAF_PROJECTION {
            LeafProjection::LeafView => {
                let view = self.leaves().leaf_view(leaf);
                let end = rows.end.min(view.len());
                for start in (rows.start.min(end)..end).step_by(32) {
                    f(Tile {
                        columns: std::array::from_fn(|d| unsafe {
                            view.points()[d].as_ptr().add(start)
                        }),
                        items: unsafe { view.items().as_ptr().add(start) },
                        len: (end - start).min(32),
                        _borrow: PhantomData,
                    });
                }
            }
            LeafProjection::LeafArena => {
                let arena = self.leaves().leaf_arena(leaf);
                let mut base = 0;
                arena.for_each_tiled_chunk(|tile| {
                    let start = rows.start.saturating_sub(base).min(tile.len());
                    let end = rows.end.saturating_sub(base).min(tile.len());
                    if start < end {
                        let (columns, items) = tile.join_columns();
                        f(Tile {
                            columns: columns.map(|p| unsafe { p.add(start) }),
                            items: unsafe { items.add(start) },
                            len: end - start,
                            _borrow: PhantomData,
                        });
                    }
                    base += tile.len();
                });
            }
        }
    }
}

#[derive(Clone)]
pub(super) struct Cursor<S> {
    pub strategy: S,
    pub range: Range<usize>,
    pub remaining: usize,
    /// Synthetic root levels added solely to align a block strategy.
    pub padding_remaining: usize,
}
impl<S: StemStrategy> Cursor<S> {
    pub fn root<const K: usize, Tree: JoinTree<K, Strategy = S>>(tree: &Tree) -> Self {
        Self {
            strategy: S::new_no_ptr(),
            range: 0..tree.leaf_count(),
            remaining: tree.depth(),
            padding_remaining: tree
                .depth()
                .saturating_sub((tree.max_stem_level() + 1).max(0) as usize),
        }
    }
    pub fn leaf(&self) -> bool {
        self.remaining == 0
    }
    pub fn span(&self) -> usize {
        self.range.end - self.range.start
    }
    pub fn split<A: JoinFloat, const K: usize>(&self) -> (Self, Self) {
        assert!(self.remaining > 0);
        let half = 1usize
            .checked_shl((self.remaining - 1) as u32)
            .expect("join tree depth exceeds address width");
        let mid = self.range.start.saturating_add(half).min(self.range.end);
        let mut left = self.clone();
        let right_strategy = left.strategy.branch::<A, K>();
        left.remaining -= 1;
        left.range.end = mid;
        let right = Self {
            strategy: right_strategy,
            range: mid..self.range.end,
            remaining: left.remaining,
            padding_remaining: left.padding_remaining,
        };
        (left, right)
    }

    /// Descends through a structural `+inf` padding pivot. It advances the
    /// physical layout only: that pivot partitions no real leaves.
    pub fn descend_padding<A: JoinFloat, const K: usize>(&self) -> Self {
        assert!(self.remaining > 0);
        assert!(self.padding_remaining > 0);
        let mut next = self.clone();
        next.strategy.traverse::<A, K>(false);
        next.remaining -= 1;
        next.padding_remaining -= 1;
        next
    }
}

use std::mem::MaybeUninit;

use crate::kd_tree::query_stack::{ScalarStackContext, StackTrait};
use crate::{Axis, StemStrategy};

const DEFAULT_INLINE_SIMD_QUERY_STACK_CAPACITY: usize = 50;
/// Native cyclic SIMD-full traversal pushes one frame per Donnelly block.
/// Twelve inline entries cover every tree representable by the current u32
/// Donnelly address space while keeping the hot stack footprint compact.
pub(crate) const CYCLIC_SIMD_FULL_INLINE_QUERY_STACK_CAPACITY: usize = 12;
// Block3 exact traversal pushes at most one pending context per 3-level block.
// Supporting trees up to 30 stem levels therefore requires ceil(30 / 3) = 10 entries.
pub(crate) const BLOCK3_EXACT_INLINE_SIMD_QUERY_STACK_CAPACITY: usize = 10;
pub(crate) const BLOCK3_EXACT_INLINE_DIM_CAPACITY: usize = 16;

#[derive(Debug)]
pub struct SimdQueryStack<
    A,
    SS: StemStrategy,
    const INLINE_CAPACITY: usize = DEFAULT_INLINE_SIMD_QUERY_STACK_CAPACITY,
> {
    stack: [MaybeUninit<SS::StackContext<A>>; INLINE_CAPACITY],
    spill: Vec<SS::StackContext<A>>,
    len: usize,
}

#[derive(Debug)]
pub struct Block3ExactQueryStack<
    A,
    SS: StemStrategy<StackContext<A> = Block3SimdQueryStackContext<A, SS>>,
    const INLINE_CAPACITY: usize = BLOCK3_EXACT_INLINE_SIMD_QUERY_STACK_CAPACITY,
> {
    stack: [MaybeUninit<Block3SimdQueryStackContext<A, SS>>; INLINE_CAPACITY],
    len: usize,
}

pub trait SimdIntervalStackContext<A, SS: StemStrategy>: Sized {
    fn new_single_with_bounds(
        stem_strat: SS,
        dim: usize,
        lower_bound: A,
        upper_bound: A,
        old_off: A,
        rd: A,
    ) -> Self;
}

#[derive(Debug)]
pub enum Block3ExactStackContextState<A, SS, const K: usize> {
    Single {
        stem_strat: SS,
        dim: usize,
        lower_bound: A,
        upper_bound: A,
        old_off: A,
        rd: A,
    },
    Block3Pending {
        base: SS,
        pending_mask: u8,
        rd: A,
        lower: [A; K],
        upper: [A; K],
    },
}

/// Compact stack state for native cyclic SIMD-full traversal.
///
/// Pending block entries retain the parent rectangle's per-axis query offsets.
/// For squared Euclidean distance these four values completely describe the
/// rectangle distance and are enough to derive every child offset in a block.
#[derive(Debug)]
pub enum CyclicSimdFullQueryStackContext<A, S> {
    Scalar {
        stem_state: S,
        restore_dim: Option<usize>,
        old_off: A,
        rd: A,
    },
    Pending {
        base_state: S,
        pending_mask: u16,
        off: [A; 4],
        /// Exact minimum lower-bound distance in `pending_mask`, or zero when
        /// the frame came from cheap near descent and has not been evaluated.
        min_rd: A,
    },
}

impl<A, S> CyclicSimdFullQueryStackContext<A, S> {
    #[inline(always)]
    pub fn pending(base_state: S, pending_mask: u16, off: [A; 4], min_rd: A) -> Self {
        Self::Pending {
            base_state,
            pending_mask,
            off,
            min_rd,
        }
    }
}

pub trait Block3ExactStackContext<A, SS: StemStrategy, const K: usize>: Sized {
    fn new_single(stem_strat: SS) -> Self
    where
        A: Axis<Coord = A>;

    fn new_block3_pending_from_state<const K2: usize>(
        base: SS,
        pending_mask: u8,
        rd: A,
        lower: &[A; K2],
        upper: &[A; K2],
    ) -> Self
    where
        A: Axis<Coord = A>;

    fn into_block3_exact_state(self) -> Block3ExactStackContextState<A, SS, K>;
}

impl<A, SS: StemStrategy, const INLINE_CAPACITY: usize> Default
    for SimdQueryStack<A, SS, INLINE_CAPACITY>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<
        A,
        SS: StemStrategy<StackContext<A> = Block3SimdQueryStackContext<A, SS>>,
        const INLINE_CAPACITY: usize,
    > Default for Block3ExactQueryStack<A, SS, INLINE_CAPACITY>
{
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub enum SimdQueryStackContext<A, SS> {
    Single {
        stem_strat: SS,
        dim: usize,
        lower_bound: A,
        upper_bound: A,
        old_off: A,
        rd: A,
    },
    Block3Pending {
        base: SS,
        rd_values: [A; 8],
        new_off_values: [A; 8],
        lower_bounds: [A; 8],
        upper_bounds: [A; 8],
        pending_mask: u8,
        dim: usize,
        old_off: A,
        lower_bound: A,
        upper_bound: A,
    },
    DeferredBlock {
        base: SS,
        child_base: u8,
        rd_values: [A; 8],
        new_off_values: [A; 8],
        lower_bounds: [A; 8],
        upper_bounds: [A; 8],
        sibling_mask: u8,
        dim: usize,
        old_off: A,
        lower_bound: A,
        upper_bound: A,
    },
    Block {
        siblings: [SS; 8],
        rd_values: [A; 8],
        new_off_values: [A; 8], // Per-sibling offset values (e.g., interval distances)
        lower_bounds: [A; 8],
        upper_bounds: [A; 8],
        sibling_mask: u8,
        dim: usize,
        old_off: A,
        lower_bound: A,
        upper_bound: A,
    },
}

#[derive(Debug)]
pub enum Block3SimdQueryStackContext<A, SS> {
    Single {
        stem_strat: SS,
        dim: usize,
        lower_bound: A,
        upper_bound: A,
        old_off: A,
        rd: A,
    },
    Block3Pending {
        base: SS,
        pending_mask: u8,
        rd: A,
        dim_count: usize,
        lower: [A; BLOCK3_EXACT_INLINE_DIM_CAPACITY],
        upper: [A; BLOCK3_EXACT_INLINE_DIM_CAPACITY],
    },
}

#[inline(always)]
fn copy_state_array_to_inline<A, const SRC: usize>(
    src: &[A; SRC],
) -> [A; BLOCK3_EXACT_INLINE_DIM_CAPACITY]
where
    A: Axis<Coord = A>,
{
    assert!(
        SRC <= BLOCK3_EXACT_INLINE_DIM_CAPACITY,
        "Block3 exact stack context dimension overflow: dims={SRC}, max={BLOCK3_EXACT_INLINE_DIM_CAPACITY}"
    );
    let mut dst = [A::zero(); BLOCK3_EXACT_INLINE_DIM_CAPACITY];
    dst[..SRC].copy_from_slice(src);
    dst
}

#[inline(always)]
fn copy_inline_state_array<A, const DST: usize>(
    src: &[A; BLOCK3_EXACT_INLINE_DIM_CAPACITY],
    dim_count: usize,
) -> [A; DST]
where
    A: Axis<Coord = A>,
{
    assert_eq!(
        dim_count, DST,
        "Block3 exact stack context dimension mismatch: src={dim_count}, dst={DST}"
    );
    let mut dst = [A::zero(); DST];
    dst.copy_from_slice(&src[..DST]);
    dst
}

impl<A, SS: StemStrategy, const INLINE_CAPACITY: usize> StackTrait<A, SS>
    for SimdQueryStack<A, SS, INLINE_CAPACITY>
{
    #[inline]
    fn push(&mut self, item: SS::StackContext<A>) {
        if self.len < INLINE_CAPACITY {
            unsafe { self.stack.get_unchecked_mut(self.len) }.write(item);
        } else {
            self.spill.push(item);
        }
        self.len += 1;

        #[cfg(any(feature = "exact_query_stats", feature = "test_utils"))]
        crate::results::exact_query_stats::record_simd_stack_len(self.len);
    }

    #[inline]
    fn pop(&mut self) -> Option<SS::StackContext<A>> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            if self.len >= INLINE_CAPACITY {
                Some(self.spill.pop().expect("simd query stack spill underflow"))
            } else {
                Some(unsafe { self.stack.get_unchecked(self.len).assume_init_read() })
            }
        }
    }

    #[inline]
    fn clear(&mut self) {
        if self.len > INLINE_CAPACITY {
            self.spill.clear();
            self.len = INLINE_CAPACITY;
        }

        while self.len > 0 {
            self.len -= 1;
            unsafe { self.stack.get_unchecked_mut(self.len).assume_init_drop() };
        }
    }
}

impl<
        A,
        SS: StemStrategy<StackContext<A> = Block3SimdQueryStackContext<A, SS>>,
        const INLINE_CAPACITY: usize,
    > StackTrait<A, SS> for Block3ExactQueryStack<A, SS, INLINE_CAPACITY>
{
    #[inline]
    fn push(&mut self, item: SS::StackContext<A>) {
        debug_assert!(
            self.len < INLINE_CAPACITY,
            "Block3 exact stack overflow: len={}, capacity={INLINE_CAPACITY}. Increase BLOCK3_EXACT_INLINE_SIMD_QUERY_STACK_CAPACITY.",
            self.len
        );
        unsafe { core::hint::assert_unchecked(self.len < INLINE_CAPACITY) };

        unsafe { self.stack.get_unchecked_mut(self.len) }.write(item);
        self.len += 1;

        #[cfg(any(feature = "exact_query_stats", feature = "test_utils"))]
        crate::results::exact_query_stats::record_simd_stack_len(self.len);
    }

    #[inline]
    fn pop(&mut self) -> Option<SS::StackContext<A>> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            Some(unsafe { self.stack.get_unchecked(self.len).assume_init_read() })
        }
    }

    #[inline]
    fn clear(&mut self) {
        while self.len > 0 {
            self.len -= 1;
            unsafe { self.stack.get_unchecked_mut(self.len).assume_init_drop() };
        }
    }
}

impl<A, SS: StemStrategy, const INLINE_CAPACITY: usize> SimdQueryStack<A, SS, INLINE_CAPACITY> {
    #[inline]
    pub fn new() -> Self {
        Self {
            stack: [const { MaybeUninit::uninit() }; INLINE_CAPACITY],
            spill: Vec::new(),
            len: 0,
        }
    }

    #[inline]
    pub fn push(&mut self, item: SS::StackContext<A>) {
        <Self as StackTrait<A, SS>>::push(self, item);
    }

    #[inline]
    pub fn pop(&mut self) -> Option<SS::StackContext<A>> {
        <Self as StackTrait<A, SS>>::pop(self)
    }
}

impl<
        A,
        SS: StemStrategy<StackContext<A> = Block3SimdQueryStackContext<A, SS>>,
        const INLINE_CAPACITY: usize,
    > Block3ExactQueryStack<A, SS, INLINE_CAPACITY>
{
    #[inline]
    pub fn new() -> Self {
        Self {
            stack: [const { MaybeUninit::uninit() }; INLINE_CAPACITY],
            len: 0,
        }
    }

    #[inline]
    pub fn push(&mut self, item: SS::StackContext<A>) {
        <Self as StackTrait<A, SS>>::push(self, item);
    }

    #[inline]
    pub fn pop(&mut self) -> Option<SS::StackContext<A>> {
        <Self as StackTrait<A, SS>>::pop(self)
    }
}

impl<A, SS: StemStrategy, const INLINE_CAPACITY: usize> Drop
    for SimdQueryStack<A, SS, INLINE_CAPACITY>
{
    fn drop(&mut self) {
        self.clear();
    }
}

impl<
        A,
        SS: StemStrategy<StackContext<A> = Block3SimdQueryStackContext<A, SS>>,
        const INLINE_CAPACITY: usize,
    > Drop for Block3ExactQueryStack<A, SS, INLINE_CAPACITY>
{
    fn drop(&mut self) {
        self.clear();
    }
}

impl<A: Axis<Coord = A>, SS: StemStrategy> SimdQueryStackContext<A, SS> {
    pub fn new_single<const K: usize>(stem_strat: SS) -> Self {
        Self::Single {
            dim: stem_strat.dim::<K>(),
            lower_bound: A::min_value(),
            upper_bound: A::max_value(),
            stem_strat,
            old_off: A::zero(),
            rd: A::zero(),
        }
    }

    pub fn new_block(
        siblings: [SS; 8],
        rd_values: [A; 8],
        new_off_values: [A; 8],
        lower_bounds: [A; 8],
        upper_bounds: [A; 8],
        sibling_mask: u8,
        dim: usize,
        old_off: A,
        lower_bound: A,
        upper_bound: A,
    ) -> Self {
        Self::Block {
            siblings,
            rd_values,
            new_off_values,
            lower_bounds,
            upper_bounds,
            sibling_mask,
            dim,
            old_off,
            lower_bound,
            upper_bound,
        }
    }

    pub fn new_deferred_block(
        base: SS,
        child_base: u8,
        rd_values: [A; 8],
        new_off_values: [A; 8],
        lower_bounds: [A; 8],
        upper_bounds: [A; 8],
        sibling_mask: u8,
        dim: usize,
        old_off: A,
        lower_bound: A,
        upper_bound: A,
    ) -> Self {
        Self::DeferredBlock {
            base,
            child_base,
            rd_values,
            new_off_values,
            lower_bounds,
            upper_bounds,
            sibling_mask,
            dim,
            old_off,
            lower_bound,
            upper_bound,
        }
    }

    pub fn new_block3_pending(
        base: SS,
        rd_values: [A; 8],
        new_off_values: [A; 8],
        lower_bounds: [A; 8],
        upper_bounds: [A; 8],
        pending_mask: u8,
        dim: usize,
        old_off: A,
        lower_bound: A,
        upper_bound: A,
    ) -> Self {
        Self::Block3Pending {
            base,
            rd_values,
            new_off_values,
            lower_bounds,
            upper_bounds,
            pending_mask,
            dim,
            old_off,
            lower_bound,
            upper_bound,
        }
    }

    pub fn into_parts(self) -> (SS, A, A) {
        match self {
            Self::Single {
                stem_strat,
                dim: _,
                lower_bound: _,
                upper_bound: _,
                old_off,
                rd,
            } => (stem_strat, old_off, rd),
            Self::Block3Pending { .. } | Self::DeferredBlock { .. } | Self::Block { .. } => {
                panic!("into_parts called on block variant")
            }
        }
    }
}

impl<A, SS> SimdIntervalStackContext<A, SS> for SimdQueryStackContext<A, SS>
where
    SS: StemStrategy,
{
    #[inline(always)]
    fn new_single_with_bounds(
        stem_strat: SS,
        dim: usize,
        lower_bound: A,
        upper_bound: A,
        old_off: A,
        rd: A,
    ) -> Self {
        Self::Single {
            stem_strat,
            dim,
            lower_bound,
            upper_bound,
            old_off,
            rd,
        }
    }
}

impl<A: Axis<Coord = A>, SS: StemStrategy> Block3SimdQueryStackContext<A, SS> {
    pub fn new_single<const K: usize>(stem_strat: SS) -> Self {
        Self::Single {
            dim: stem_strat.dim::<K>(),
            lower_bound: A::min_value(),
            upper_bound: A::max_value(),
            stem_strat,
            old_off: A::zero(),
            rd: A::zero(),
        }
    }

    pub fn new_block3_pending(
        base: SS,
        pending_mask: u8,
        rd: A,
        dim_count: usize,
        lower: &[A; BLOCK3_EXACT_INLINE_DIM_CAPACITY],
        upper: &[A; BLOCK3_EXACT_INLINE_DIM_CAPACITY],
    ) -> Self {
        Self::Block3Pending {
            base,
            pending_mask,
            rd,
            dim_count,
            lower: *lower,
            upper: *upper,
        }
    }

    pub fn into_parts(self) -> (SS, A, A) {
        match self {
            Self::Single {
                stem_strat,
                dim: _,
                lower_bound: _,
                upper_bound: _,
                old_off,
                rd,
            } => (stem_strat, old_off, rd),
            Self::Block3Pending { .. } => panic!("into_parts called on block variant"),
        }
    }
}

impl<A, SS, const K: usize> Block3ExactStackContext<A, SS, K> for Block3SimdQueryStackContext<A, SS>
where
    A: Axis<Coord = A>,
    SS: StemStrategy,
{
    #[inline(always)]
    fn new_single(stem_strat: SS) -> Self
    where
        A: Axis<Coord = A>,
    {
        Block3SimdQueryStackContext::new_single::<K>(stem_strat)
    }

    #[inline(always)]
    fn new_block3_pending_from_state<const K2: usize>(
        base: SS,
        pending_mask: u8,
        rd: A,
        lower: &[A; K2],
        upper: &[A; K2],
    ) -> Self
    where
        A: Axis<Coord = A>,
    {
        Block3SimdQueryStackContext::new_block3_pending(
            base,
            pending_mask,
            rd,
            K2,
            &copy_state_array_to_inline(lower),
            &copy_state_array_to_inline(upper),
        )
    }

    #[inline(always)]
    fn into_block3_exact_state(self) -> Block3ExactStackContextState<A, SS, K> {
        match self {
            Self::Single {
                stem_strat,
                dim,
                lower_bound,
                upper_bound,
                old_off,
                rd,
            } => Block3ExactStackContextState::Single {
                stem_strat,
                dim,
                lower_bound,
                upper_bound,
                old_off,
                rd,
            },
            Self::Block3Pending {
                base,
                pending_mask,
                rd,
                dim_count,
                lower,
                upper,
            } => Block3ExactStackContextState::Block3Pending {
                base,
                pending_mask,
                rd,
                lower: copy_inline_state_array(&lower, dim_count),
                upper: copy_inline_state_array(&upper, dim_count),
            },
        }
    }
}

impl<A, SS> SimdIntervalStackContext<A, SS> for Block3SimdQueryStackContext<A, SS>
where
    SS: StemStrategy,
{
    #[inline(always)]
    fn new_single_with_bounds(
        stem_strat: SS,
        dim: usize,
        lower_bound: A,
        upper_bound: A,
        old_off: A,
        rd: A,
    ) -> Self {
        Self::Single {
            stem_strat,
            dim,
            lower_bound,
            upper_bound,
            old_off,
            rd,
        }
    }
}

impl<A, SS, S> ScalarStackContext<A, S> for SimdQueryStackContext<A, SS> {
    #[inline(always)]
    fn from_parts(_stem_state: S, _old_off: A, _rd: A) -> Self {
        unreachable!("SIMD stack contexts do not support scalar stack packing")
    }

    #[inline(always)]
    fn into_parts(self) -> (S, A, A) {
        unreachable!("SIMD stack contexts do not support scalar stack unpacking")
    }
}

impl<A, SS, S> ScalarStackContext<A, S> for Block3SimdQueryStackContext<A, SS> {
    #[inline(always)]
    fn from_parts(_stem_state: S, _old_off: A, _rd: A) -> Self {
        unreachable!("SIMD stack contexts do not support scalar stack packing")
    }

    #[inline(always)]
    fn into_parts(self) -> (S, A, A) {
        unreachable!("SIMD stack contexts do not support scalar stack unpacking")
    }
}

impl<A, S> ScalarStackContext<A, S> for CyclicSimdFullQueryStackContext<A, S> {
    #[inline(always)]
    fn from_parts(stem_state: S, old_off: A, rd: A) -> Self {
        Self::Scalar {
            stem_state,
            restore_dim: None,
            old_off,
            rd,
        }
    }

    #[inline(always)]
    fn into_parts(self) -> (S, A, A) {
        match self {
            Self::Scalar {
                stem_state,
                old_off,
                rd,
                ..
            } => (stem_state, old_off, rd),
            Self::Pending { .. } => {
                panic!("scalar stack unpack called on cyclic SIMD pending block")
            }
        }
    }

    #[inline(always)]
    fn from_parts_with_restore_dim(stem_state: S, restore_dim: usize, old_off: A, rd: A) -> Self {
        Self::Scalar {
            stem_state,
            restore_dim: Some(restore_dim),
            old_off,
            rd,
        }
    }

    #[inline(always)]
    fn into_parts_with_restore_dim(self) -> (S, Option<usize>, A, A) {
        match self {
            Self::Scalar {
                stem_state,
                restore_dim,
                old_off,
                rd,
            } => (stem_state, restore_dim, old_off, rd),
            Self::Pending { .. } => {
                panic!("scalar stack unpack called on cyclic SIMD pending block")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stem_strategy::Eytzinger;
    use std::ptr::NonNull;

    type TestStem = Eytzinger;

    fn test_stem() -> TestStem {
        TestStem::new(NonNull::dangling())
    }

    #[test]
    fn simd_query_stack_context_new_block_stores_all_fields() {
        let root = test_stem();
        let siblings = std::array::from_fn(|idx| {
            let mut stem = root.clone();
            stem.traverse::<f32, 3>(idx % 2 == 1);
            stem
        });

        let ctx = SimdQueryStackContext::new_block(
            siblings.clone(),
            [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [1.1f32, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
            [-1.0f32, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0],
            [2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            0b1010_0101,
            2,
            9.5,
            -10.0,
            10.0,
        );

        match ctx {
            SimdQueryStackContext::Block {
                siblings: stored_siblings,
                rd_values,
                new_off_values,
                lower_bounds,
                upper_bounds,
                sibling_mask,
                dim,
                old_off,
                lower_bound,
                upper_bound,
            } => {
                assert_eq!(stored_siblings[0].stem_idx(), siblings[0].stem_idx());
                assert_eq!(stored_siblings[7].stem_idx(), siblings[7].stem_idx());
                assert_eq!(rd_values, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
                assert_eq!(new_off_values, [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]);
                assert_eq!(
                    lower_bounds,
                    [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]
                );
                assert_eq!(upper_bounds, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
                assert_eq!(sibling_mask, 0b1010_0101);
                assert_eq!(dim, 2);
                assert_eq!(old_off, 9.5);
                assert_eq!(lower_bound, -10.0);
                assert_eq!(upper_bound, 10.0);
            }
            _ => panic!("expected block context"),
        }
    }

    #[test]
    fn simd_query_stack_context_new_block3_pending_stores_all_fields() {
        let base = test_stem();

        let ctx = SimdQueryStackContext::new_block3_pending(
            base.clone(),
            [0.5f32, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
            [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [-8.0f32, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0],
            [8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            0b0011_0110,
            1,
            4.25,
            -12.0,
            12.0,
        );

        match ctx {
            SimdQueryStackContext::Block3Pending {
                base: stored_base,
                rd_values,
                new_off_values,
                lower_bounds,
                upper_bounds,
                pending_mask,
                dim,
                old_off,
                lower_bound,
                upper_bound,
            } => {
                assert_eq!(stored_base.stem_idx(), base.stem_idx());
                assert_eq!(rd_values, [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]);
                assert_eq!(new_off_values, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
                assert_eq!(
                    lower_bounds,
                    [-8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0]
                );
                assert_eq!(upper_bounds, [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
                assert_eq!(pending_mask, 0b0011_0110);
                assert_eq!(dim, 1);
                assert_eq!(old_off, 4.25);
                assert_eq!(lower_bound, -12.0);
                assert_eq!(upper_bound, 12.0);
            }
            _ => panic!("expected block3 pending context"),
        }
    }

    #[test]
    fn simd_query_stack_context_into_parts_extracts_single_fields() {
        let stem = test_stem();
        let ctx = SimdQueryStackContext::<f32, TestStem>::Single {
            stem_strat: stem.clone(),
            dim: 1,
            lower_bound: -5.0,
            upper_bound: 5.0,
            old_off: 1.25,
            rd: 2.5,
        };

        let (stored_stem, old_off, rd) = ctx.into_parts();

        assert_eq!(stored_stem.stem_idx(), stem.stem_idx());
        assert_eq!(old_off, 1.25);
        assert_eq!(rd, 2.5);
    }

    #[test]
    fn block3_simd_query_stack_context_into_parts_extracts_single_fields() {
        let stem = test_stem();
        let ctx = Block3SimdQueryStackContext::<f32, TestStem>::Single {
            stem_strat: stem.clone(),
            dim: 2,
            lower_bound: -3.0,
            upper_bound: 7.0,
            old_off: 0.75,
            rd: 1.5,
        };

        let (stored_stem, old_off, rd) = ctx.into_parts();

        assert_eq!(stored_stem.stem_idx(), stem.stem_idx());
        assert_eq!(old_off, 0.75);
        assert_eq!(rd, 1.5);
    }

    #[test]
    fn block3_simd_query_stack_context_new_single_with_bounds_sets_single_variant() {
        let stem = test_stem();
        let ctx = <Block3SimdQueryStackContext<f32, TestStem> as SimdIntervalStackContext<
            f32,
            TestStem,
        >>::new_single_with_bounds(stem.clone(), 2, -4.0, 9.0, 1.0, 3.0);

        match ctx {
            Block3SimdQueryStackContext::Single {
                stem_strat,
                dim,
                lower_bound,
                upper_bound,
                old_off,
                rd,
            } => {
                assert_eq!(stem_strat.stem_idx(), stem.stem_idx());
                assert_eq!(dim, 2);
                assert_eq!(lower_bound, -4.0);
                assert_eq!(upper_bound, 9.0);
                assert_eq!(old_off, 1.0);
                assert_eq!(rd, 3.0);
            }
            _ => panic!("expected single block3 simd context"),
        }
    }
}

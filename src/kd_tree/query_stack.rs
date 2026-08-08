use std::mem::MaybeUninit;

use crate::{Axis, StemStrategy};

const INLINE_QUERY_STACK_CAPACITY: usize = 50;
#[allow(dead_code)]
const INLINE_SCALAR_CONTINUATION_STACK_CAPACITY: usize = 64;

pub trait ScalarStackContext<A, S>: Sized {
    fn from_parts(stem_state: S, old_off: A, rd: A) -> Self;
    fn into_parts(self) -> (S, A, A);

    #[inline(always)]
    fn from_parts_with_restore_dim(stem_state: S, _restore_dim: usize, old_off: A, rd: A) -> Self {
        Self::from_parts(stem_state, old_off, rd)
    }

    #[inline(always)]
    fn into_parts_with_restore_dim(self) -> (S, Option<usize>, A, A) {
        let (stem_state, old_off, rd) = self.into_parts();
        (stem_state, None, old_off, rd)
    }
}

/// Trait for query stack types to enable generic backtracking implementations
pub trait StackTrait<A, SS: StemStrategy, const K: usize> {
    fn push(&mut self, item: SS::StackContext<A, K>);
    fn pop(&mut self) -> Option<SS::StackContext<A, K>>;
    fn clear(&mut self);
}

/// Reusable traversal scratch storage for query builder scratch APIs.
///
/// The concrete stack representation is selected by the stem strategy and is
/// intentionally hidden so it can keep changing during the v6 alpha cycle.
#[derive(Debug)]
pub struct QueryScratch<SS: StemStrategy, O, const K: usize> {
    stack: SS::Stack<O, K>,
}

impl<SS: StemStrategy, O, const K: usize> Default for QueryScratch<SS, O, K> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<SS: StemStrategy, O, const K: usize> QueryScratch<SS, O, K> {
    /// Creates empty query traversal scratch storage.
    #[inline]
    pub fn new() -> Self {
        Self {
            stack: SS::Stack::<O, K>::default(),
        }
    }

    #[inline]
    pub(crate) fn clear(&mut self) {
        self.stack.clear();
    }

    #[inline]
    pub(crate) fn stack_mut(&mut self) -> &mut SS::Stack<O, K> {
        &mut self.stack
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct ScalarContinuationFar<A, S> {
    pub stem_state: S,
    pub far_off: A,
    pub rd: A,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct ScalarContinuationRestore<A> {
    pub old_off: A,
    pub has_far: bool,
}

#[allow(dead_code)]
impl<A> ScalarContinuationRestore<A> {
    #[inline(always)]
    pub fn restore_only(old_off: A) -> Self {
        Self {
            old_off,
            has_far: false,
        }
    }

    #[inline(always)]
    pub fn with_far(old_off: A) -> Self {
        Self {
            old_off,
            has_far: true,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct ScalarContinuationRestoreStack<
    A,
    const INLINE_CAPACITY: usize = INLINE_SCALAR_CONTINUATION_STACK_CAPACITY,
> {
    stack: [MaybeUninit<ScalarContinuationRestore<A>>; INLINE_CAPACITY],
    len: usize,
}

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct ScalarContinuationFarStack<
    A,
    S,
    const INLINE_CAPACITY: usize = INLINE_SCALAR_CONTINUATION_STACK_CAPACITY,
> {
    stack: [MaybeUninit<ScalarContinuationFar<A, S>>; INLINE_CAPACITY],
    len: usize,
}

#[derive(Debug)]
pub struct QueryStack<A, SS: StemStrategy, const K: usize> {
    stack: [MaybeUninit<SS::StackContext<A, K>>; INLINE_QUERY_STACK_CAPACITY],
    spill: Vec<SS::StackContext<A, K>>,
    len: usize,
}

impl<A, SS: StemStrategy, const K: usize> Default for QueryStack<A, SS, K> {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(dead_code)]
impl<A, const INLINE_CAPACITY: usize> Default
    for ScalarContinuationRestoreStack<A, INLINE_CAPACITY>
{
    fn default() -> Self {
        Self::new()
    }
}

#[allow(dead_code)]
impl<A, S, const INLINE_CAPACITY: usize> Default
    for ScalarContinuationFarStack<A, S, INLINE_CAPACITY>
{
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct QueryStackContext<A, S> {
    pub stem_state: S,
    pub restore_dim: Option<usize>,
    pub old_off: A,
    pub rd: A,
}

impl<A, SS: StemStrategy, const K: usize> StackTrait<A, SS, K> for QueryStack<A, SS, K> {
    #[inline]
    fn push(&mut self, item: SS::StackContext<A, K>) {
        if self.len < INLINE_QUERY_STACK_CAPACITY {
            unsafe { self.stack.get_unchecked_mut(self.len) }.write(item);
        } else {
            self.spill.push(item);
        }
        self.len += 1;
    }

    #[inline]
    fn pop(&mut self) -> Option<SS::StackContext<A, K>> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            if self.len >= INLINE_QUERY_STACK_CAPACITY {
                Some(self.spill.pop().expect("query stack spill underflow"))
            } else {
                Some(unsafe { self.stack.get_unchecked(self.len).assume_init_read() })
            }
        }
    }

    #[inline]
    fn clear(&mut self) {
        if self.len > INLINE_QUERY_STACK_CAPACITY {
            self.spill.clear();
            self.len = INLINE_QUERY_STACK_CAPACITY;
        }

        while self.len > 0 {
            self.len -= 1;
            unsafe { self.stack.get_unchecked_mut(self.len).assume_init_drop() };
        }
    }
}

impl<A, SS: StemStrategy, const K: usize> QueryStack<A, SS, K> {
    #[inline]
    pub fn new() -> Self {
        Self {
            stack: [const { MaybeUninit::uninit() }; INLINE_QUERY_STACK_CAPACITY],
            spill: Vec::new(),
            len: 0,
        }
    }

    #[inline]
    pub fn push(&mut self, item: SS::StackContext<A, K>) {
        <Self as StackTrait<A, SS, K>>::push(self, item);
    }

    #[inline]
    pub fn pop(&mut self) -> Option<SS::StackContext<A, K>> {
        <Self as StackTrait<A, SS, K>>::pop(self)
    }
}

impl<A, SS: StemStrategy, const K: usize> Drop for QueryStack<A, SS, K> {
    fn drop(&mut self) {
        self.clear();
    }
}

#[allow(dead_code)]
impl<A, const INLINE_CAPACITY: usize> ScalarContinuationRestoreStack<A, INLINE_CAPACITY> {
    #[inline]
    pub fn new() -> Self {
        Self {
            stack: [const { MaybeUninit::uninit() }; INLINE_CAPACITY],
            len: 0,
        }
    }

    #[inline]
    pub fn push_unchecked_inline(&mut self, item: ScalarContinuationRestore<A>) {
        debug_assert!(self.len < INLINE_CAPACITY);
        unsafe { self.stack.get_unchecked_mut(self.len) }.write(item);
        self.len += 1;
    }

    #[inline]
    pub fn pop(&mut self) -> Option<ScalarContinuationRestore<A>> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            Some(unsafe { self.stack.get_unchecked(self.len).assume_init_read() })
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        while self.len > 0 {
            self.len -= 1;
            unsafe { self.stack.get_unchecked_mut(self.len).assume_init_drop() };
        }
    }
}

#[allow(dead_code)]
impl<A, S, const INLINE_CAPACITY: usize> ScalarContinuationFarStack<A, S, INLINE_CAPACITY> {
    #[inline]
    pub fn new() -> Self {
        Self {
            stack: [const { MaybeUninit::uninit() }; INLINE_CAPACITY],
            len: 0,
        }
    }

    #[inline]
    pub fn push_unchecked_inline(&mut self, item: ScalarContinuationFar<A, S>) {
        debug_assert!(self.len < INLINE_CAPACITY);
        unsafe { self.stack.get_unchecked_mut(self.len) }.write(item);
        self.len += 1;
    }

    #[inline]
    pub fn pop(&mut self) -> Option<ScalarContinuationFar<A, S>> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            Some(unsafe { self.stack.get_unchecked(self.len).assume_init_read() })
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        while self.len > 0 {
            self.len -= 1;
            unsafe { self.stack.get_unchecked_mut(self.len).assume_init_drop() };
        }
    }
}

#[allow(dead_code)]
impl<A, const INLINE_CAPACITY: usize> Drop for ScalarContinuationRestoreStack<A, INLINE_CAPACITY> {
    fn drop(&mut self) {
        self.clear();
    }
}

#[allow(dead_code)]
impl<A, S, const INLINE_CAPACITY: usize> Drop
    for ScalarContinuationFarStack<A, S, INLINE_CAPACITY>
{
    fn drop(&mut self) {
        self.clear();
    }
}

impl<A: Axis<Coord = A>, S> QueryStackContext<A, S> {
    pub fn new(stem_state: S) -> Self {
        Self {
            stem_state,
            restore_dim: None,
            old_off: A::zero(),
            rd: A::zero(),
        }
    }
}

impl<A, S> QueryStackContext<A, S> {
    pub fn into_parts(self) -> (S, A, A) {
        (self.stem_state, self.old_off, self.rd)
    }
}

impl<A, S> ScalarStackContext<A, S> for QueryStackContext<A, S> {
    #[inline(always)]
    fn from_parts(stem_state: S, old_off: A, rd: A) -> Self {
        Self {
            stem_state,
            restore_dim: None,
            old_off,
            rd,
        }
    }

    #[inline(always)]
    fn from_parts_with_restore_dim(stem_state: S, restore_dim: usize, old_off: A, rd: A) -> Self {
        Self {
            stem_state,
            restore_dim: Some(restore_dim),
            old_off,
            rd,
        }
    }

    #[inline(always)]
    fn into_parts(self) -> (S, A, A) {
        QueryStackContext::into_parts(self)
    }

    #[inline(always)]
    fn into_parts_with_restore_dim(self) -> (S, Option<usize>, A, A) {
        (self.stem_state, self.restore_dim, self.old_off, self.rd)
    }
}

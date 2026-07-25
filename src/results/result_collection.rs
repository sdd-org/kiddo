use std::collections::BinaryHeap;
use std::num::NonZeroUsize;

#[cfg(any(
    feature = "buffered_result_collection",
    feature = "small_n_result_collectors"
))]
use smallvec::SmallVec;
use sorted_vec::SortedVec;

use crate::{Axis, BestQueryResultItem, Content, QueryResultItem};

pub(crate) const DEFAULT_UNSORTED_RESULT_CAPACITY: usize = 64;

#[cfg(feature = "small_n_result_collectors")]
pub(crate) const SMALL_RESULT_COLLECTION_MAX_QTY: usize = 32;

#[cfg(feature = "buffered_result_collection")]
#[allow(dead_code)]
pub(crate) const BUFFERED_RESULT_COLLECTION_INLINE_CAPACITY: usize = 64;

#[cfg(feature = "buffered_result_collection")]
#[allow(dead_code)]
pub(crate) type ResultBuffer<E> = SmallVec<[E; BUFFERED_RESULT_COLLECTION_INLINE_CAPACITY]>;

pub trait ResultCollection<O: Axis<Coord = O>, E>: Sized {
    fn with_max_qty(max_qty: usize) -> Self;
    #[inline]
    fn with_max_qty_and_capacity(max_qty: usize, result_capacity: Option<NonZeroUsize>) -> Self {
        let _ = result_capacity;
        Self::with_max_qty(max_qty)
    }
    fn max_qty(&self) -> usize;
    fn len(&self) -> usize;
    fn add(&mut self, entry: E);
    fn threshold_distance(&self) -> Option<O>;
    fn into_vec(self) -> Vec<E>;
    fn into_sorted_vec(self) -> Vec<E>;

    #[cfg(feature = "buffered_result_collection")]
    #[inline(always)]
    fn add_all<I>(&mut self, entries: I)
    where
        I: IntoIterator<Item = E>,
    {
        #[cfg(feature = "result_collection_stats")]
        {
            let iter = entries.into_iter();
            crate::results::result_collection_stats::record_collector_add_all_call(
                iter.size_hint().0,
            );
            for entry in iter {
                self.add(entry);
            }
        }

        #[cfg(not(feature = "result_collection_stats"))]
        for entry in entries {
            self.add(entry);
        }
    }

    #[inline(always)]
    fn is_full(&self) -> bool {
        self.len() >= self.max_qty()
    }
}

pub(crate) struct VisitorResultCollection<'a, O, E, F>
where
    O: Axis<Coord = O>,
    F: FnMut(E),
{
    visitor: &'a mut F,
    len: usize,
    _phantom: std::marker::PhantomData<(O, E)>,
}

impl<'a, O, E, F> VisitorResultCollection<'a, O, E, F>
where
    O: Axis<Coord = O>,
    F: FnMut(E),
{
    #[inline(always)]
    pub(crate) fn new(visitor: &'a mut F) -> Self {
        Self {
            visitor,
            len: 0,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<O, E, F> ResultCollection<O, E> for VisitorResultCollection<'_, O, E, F>
where
    O: Axis<Coord = O>,
    F: FnMut(E),
{
    #[inline(always)]
    fn with_max_qty(_max_qty: usize) -> Self {
        panic!("VisitorResultCollection must be constructed with VisitorResultCollection::new")
    }

    #[inline(always)]
    fn max_qty(&self) -> usize {
        usize::MAX
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    fn add(&mut self, entry: E) {
        self.len += 1;
        (self.visitor)(entry);
    }

    #[inline(always)]
    fn threshold_distance(&self) -> Option<O> {
        None
    }

    #[inline(always)]
    fn into_vec(self) -> Vec<E> {
        Vec::new()
    }

    #[inline(always)]
    fn into_sorted_vec(self) -> Vec<E> {
        Vec::new()
    }
}

/// Vec-based result collection for small-k nearest_n queries.
///
/// Inserts candidates via plain `Vec::push` (O(1)) during the fill phase.
/// Once `max_qty` items have been collected, the vec is sorted once.
/// Subsequent insertions that beat the current threshold (last/farthest
/// element) are inserted by scanning backward while shifting larger entries,
/// maintaining sorted order in a single pass. The threshold is always accurate
/// (the farthest of the current top-k), enabling effective branch pruning during
/// tree traversal.
/// A separate backward scan followed by `copy_within` was also benchmarked, but
/// was consistently and materially slower for result sizes through 256.
#[cfg(any(not(feature = "small_n_result_collectors"), feature = "test_utils"))]
#[derive(Debug)]
pub(crate) struct ThresholdVecResultCollection<E> {
    max_qty: usize,
    inner: Vec<E>,
}

#[cfg(any(not(feature = "small_n_result_collectors"), feature = "test_utils"))]
impl<O: Axis<Coord = O>, T: Copy> ResultCollection<O, QueryResultItem<(), T, O>>
    for ThresholdVecResultCollection<QueryResultItem<(), T, O>>
{
    fn with_max_qty(max_qty: usize) -> Self {
        Self {
            max_qty,
            inner: Vec::with_capacity(max_qty),
        }
    }

    fn max_qty(&self) -> usize {
        self.max_qty
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn add(&mut self, entry: QueryResultItem<(), T, O>) {
        if self.inner.len() < self.max_qty {
            self.inner.push(entry);
            if self.inner.len() == self.max_qty {
                self.inner.sort_unstable_by(|a, b| {
                    a.distance
                        .partial_cmp(&b.distance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        } else if entry < self.inner[self.max_qty - 1] {
            let mut idx = self.max_qty - 1;
            while idx > 0 && self.inner[idx - 1] > entry {
                self.inner[idx] = self.inner[idx - 1];
                idx -= 1;
            }
            self.inner[idx] = entry;
        }
    }

    fn threshold_distance(&self) -> Option<O> {
        if self.inner.len() == self.max_qty {
            self.inner.last().map(|n| n.distance)
        } else {
            None
        }
    }

    fn into_vec(self) -> Vec<QueryResultItem<(), T, O>> {
        self.inner
    }

    fn into_sorted_vec(self) -> Vec<QueryResultItem<(), T, O>> {
        let max_qty = self.max_qty();
        let mut vec = self.into_vec();
        if vec.len() < max_qty {
            vec.sort_unstable_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        vec
    }
}

#[doc(hidden)]
pub trait BestNeighbourResultCollection<O: Axis<Coord = O>, T: Content + PartialOrd>:
    ResultCollection<O, BestQueryResultItem<(), T, O>>
{
    fn threshold_item(&self) -> Option<T>;
}

#[derive(Debug)]
pub(crate) struct BinaryHeapResultCollection<E> {
    max_qty: usize,
    inner: BinaryHeap<E>,
}

#[derive(Debug)]
#[cfg_attr(feature = "small_n_result_collectors", allow(dead_code))]
#[allow(dead_code)]
pub(crate) struct SortedVecResultCollection<E: Ord> {
    max_qty: usize,
    inner: SortedVec<E>,
}

#[cfg(feature = "small_n_result_collectors")]
#[derive(Debug)]
pub(crate) struct SmallBinaryHeapResultCollection<E: Ord> {
    max_qty: usize,
    inner: SmallVec<[E; SMALL_RESULT_COLLECTION_MAX_QTY]>,
}

#[cfg(feature = "small_n_result_collectors")]
#[derive(Debug)]
pub(crate) struct SmallSortedVecResultCollection<E: Ord> {
    max_qty: usize,
    inner: SmallVec<[E; SMALL_RESULT_COLLECTION_MAX_QTY]>,
}

impl<E> BinaryHeapResultCollection<E> {
    #[inline(always)]
    pub(crate) fn into_inner(self) -> BinaryHeap<E> {
        self.inner
    }
}

#[cfg(feature = "small_n_result_collectors")]
impl<E: Ord> SmallBinaryHeapResultCollection<E> {
    #[inline(always)]
    pub(crate) fn into_inner(self) -> BinaryHeap<E> {
        BinaryHeap::from(self.inner.into_vec())
    }
}

#[cfg(feature = "small_n_result_collectors")]
#[inline(always)]
fn sift_up_max_heap<E: Ord>(heap: &mut [E], mut idx: usize) {
    while idx > 0 {
        let parent = (idx - 1) / 2;
        if heap[parent] >= heap[idx] {
            break;
        }
        heap.swap(parent, idx);
        idx = parent;
    }
}

#[cfg(feature = "small_n_result_collectors")]
#[inline(always)]
fn sift_down_max_heap<E: Ord>(heap: &mut [E], mut idx: usize) {
    let len = heap.len();
    loop {
        let left = idx * 2 + 1;
        if left >= len {
            break;
        }

        let right = left + 1;
        let mut largest = left;
        if right < len && heap[right] > heap[left] {
            largest = right;
        }

        if heap[idx] >= heap[largest] {
            break;
        }

        heap.swap(idx, largest);
        idx = largest;
    }
}

#[cfg(all(
    feature = "small_n_result_collectors",
    feature = "buffered_result_collection"
))]
#[inline(always)]
fn heapify_max_heap<E: Ord>(heap: &mut [E]) {
    if heap.len() <= 1 {
        return;
    }

    let mut idx = heap.len() / 2;
    while idx > 0 {
        idx -= 1;
        sift_down_max_heap(heap, idx);
    }
}

#[cfg(feature = "small_n_result_collectors")]
#[inline(always)]
fn small_sorted_insert<E: Ord>(
    inner: &mut SmallVec<[E; SMALL_RESULT_COLLECTION_MAX_QTY]>,
    max_qty: usize,
    entry: E,
) {
    if inner.len() < max_qty {
        let insert_at = inner.partition_point(|existing| *existing <= entry);
        #[cfg(feature = "result_collection_stats")]
        crate::results::result_collection_stats::record_sorted_insert(
            insert_at,
            inner.len() - insert_at,
        );
        inner.insert(insert_at, entry);
    } else if entry < *inner.last().unwrap() {
        let insert_at = inner.partition_point(|existing| *existing <= entry);
        #[cfg(feature = "result_collection_stats")]
        crate::results::result_collection_stats::record_sorted_insert(
            insert_at,
            inner.len().saturating_sub(insert_at + 1),
        );
        inner.insert(insert_at, entry);
        inner.pop();
    }
}

#[cfg(feature = "buffered_result_collection")]
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn flush_result_buffer<O, E, R>(results: &mut R, buffer: &mut ResultBuffer<E>)
where
    O: Axis<Coord = O>,
    R: ResultCollection<O, E>,
{
    #[cfg(feature = "result_collection_stats")]
    crate::results::result_collection_stats::record_buffer_flush(buffer.len());

    if !buffer.is_empty() {
        results.add_all(buffer.drain(..));
    }
}

impl<O: Axis<Coord = O>, T> ResultCollection<O, QueryResultItem<(), T, O>>
    for BinaryHeapResultCollection<QueryResultItem<(), T, O>>
{
    fn with_max_qty(max_qty: usize) -> Self {
        Self {
            max_qty,
            inner: BinaryHeap::with_capacity(max_qty),
        }
    }

    fn max_qty(&self) -> usize {
        self.max_qty
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn add(&mut self, entry: QueryResultItem<(), T, O>) {
        #[cfg(feature = "result_collection_stats")]
        crate::results::result_collection_stats::record_collector_add_call();

        if self.inner.len() < self.max_qty {
            #[cfg(feature = "result_collection_stats")]
            crate::results::result_collection_stats::record_heap_add_push();
            self.inner.push(entry);
        } else {
            let mut max_heap_value = self.inner.peek_mut().unwrap();
            if entry < *max_heap_value {
                #[cfg(feature = "result_collection_stats")]
                crate::results::result_collection_stats::record_heap_add_replacement();
                *max_heap_value = entry;
            } else {
                #[cfg(feature = "result_collection_stats")]
                crate::results::result_collection_stats::record_heap_add_discard();
            }
        }
    }

    #[cfg(feature = "buffered_result_collection")]
    #[allow(unreachable_code)] // needed because of early return when result_collection_stats feature is enabled
    fn add_all<I>(&mut self, entries: I)
    where
        I: IntoIterator<Item = QueryResultItem<(), T, O>>,
    {
        #[cfg(feature = "result_collection_stats")]
        {
            let iter = entries.into_iter();
            crate::results::result_collection_stats::record_collector_add_all_call(
                iter.size_hint().0,
            );
            let mut combined = std::mem::take(&mut self.inner).into_vec();
            combined.extend(iter);

            if combined.len() > self.max_qty {
                combined.sort_unstable();
                combined.truncate(self.max_qty);
            }

            self.inner = BinaryHeap::from(combined);
            return;
        }

        let mut combined = std::mem::take(&mut self.inner).into_vec();
        combined.extend(entries);

        if combined.len() > self.max_qty {
            combined.sort_unstable();
            combined.truncate(self.max_qty);
        }

        self.inner = BinaryHeap::from(combined);
    }

    fn threshold_distance(&self) -> Option<O> {
        let is_full = self.is_full();
        let result = if is_full {
            self.inner.peek().map(|n| n.distance)
        } else {
            None
        };
        #[cfg(feature = "result_collection_stats")]
        crate::results::result_collection_stats::record_threshold_distance_call(
            is_full,
            result.is_some(),
        );
        result
    }

    fn into_vec(self) -> Vec<QueryResultItem<(), T, O>> {
        self.inner.into_vec()
    }

    fn into_sorted_vec(self) -> Vec<QueryResultItem<(), T, O>> {
        self.inner.into_sorted_vec()
    }
}

impl<O: Axis<Coord = O>, T: Content + PartialOrd> ResultCollection<O, BestQueryResultItem<(), T, O>>
    for BinaryHeapResultCollection<BestQueryResultItem<(), T, O>>
{
    fn with_max_qty(max_qty: usize) -> Self {
        Self {
            max_qty,
            inner: BinaryHeap::with_capacity(max_qty),
        }
    }

    fn max_qty(&self) -> usize {
        self.max_qty
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn add(&mut self, entry: BestQueryResultItem<(), T, O>) {
        #[cfg(feature = "result_collection_stats")]
        crate::results::result_collection_stats::record_collector_add_call();

        if self.inner.len() < self.max_qty {
            #[cfg(feature = "result_collection_stats")]
            crate::results::result_collection_stats::record_heap_add_push();
            self.inner.push(entry);
        } else {
            let mut max_heap_value = self.inner.peek_mut().unwrap();
            if entry < *max_heap_value {
                #[cfg(feature = "result_collection_stats")]
                crate::results::result_collection_stats::record_heap_add_replacement();
                *max_heap_value = entry;
            } else {
                #[cfg(feature = "result_collection_stats")]
                crate::results::result_collection_stats::record_heap_add_discard();
            }
        }
    }

    #[allow(unreachable_code)] // needed because of early return when result_collection_stats feature is enabled
    #[cfg(feature = "buffered_result_collection")]
    fn add_all<I>(&mut self, entries: I)
    where
        I: IntoIterator<Item = BestQueryResultItem<(), T, O>>,
    {
        #[cfg(feature = "result_collection_stats")]
        {
            let iter = entries.into_iter();
            crate::results::result_collection_stats::record_collector_add_all_call(
                iter.size_hint().0,
            );
            let mut combined = std::mem::take(&mut self.inner).into_vec();
            combined.extend(iter);

            if combined.len() > self.max_qty {
                combined.sort_unstable();
                combined.truncate(self.max_qty);
            }

            self.inner = BinaryHeap::from(combined);
            return;
        }

        let mut combined = std::mem::take(&mut self.inner).into_vec();
        combined.extend(entries);

        if combined.len() > self.max_qty {
            combined.sort_unstable();
            combined.truncate(self.max_qty);
        }

        self.inner = BinaryHeap::from(combined);
    }

    fn threshold_distance(&self) -> Option<O> {
        None
    }

    fn into_vec(self) -> Vec<BestQueryResultItem<(), T, O>> {
        self.inner.into_vec()
    }

    fn into_sorted_vec(self) -> Vec<BestQueryResultItem<(), T, O>> {
        self.inner.into_sorted_vec()
    }
}

impl<O: Axis<Coord = O>, T: Content + PartialOrd> BestNeighbourResultCollection<O, T>
    for BinaryHeapResultCollection<BestQueryResultItem<(), T, O>>
{
    #[inline(always)]
    fn threshold_item(&self) -> Option<T> {
        if self.is_full() {
            self.inner.peek().map(|n| n.item)
        } else {
            None
        }
    }
}

impl<O: Axis<Coord = O>, T> ResultCollection<O, QueryResultItem<(), T, O>>
    for SortedVecResultCollection<QueryResultItem<(), T, O>>
{
    fn with_max_qty(max_qty: usize) -> Self {
        Self {
            max_qty,
            inner: SortedVec::with_capacity(max_qty),
        }
    }

    fn max_qty(&self) -> usize {
        self.max_qty
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn add(&mut self, entry: QueryResultItem<(), T, O>) {
        #[cfg(feature = "result_collection_stats")]
        crate::results::result_collection_stats::record_collector_add_call();

        if self.inner.len() < self.max_qty {
            #[cfg(feature = "result_collection_stats")]
            {
                let insert_at = self.inner.partition_point(|existing| *existing <= entry);
                crate::results::result_collection_stats::record_sorted_insert(
                    insert_at,
                    self.inner.len() - insert_at,
                );
            }
            self.inner.insert(entry);
        } else if entry < *self.inner.last().unwrap() {
            #[cfg(feature = "result_collection_stats")]
            {
                let insert_at = self.inner.partition_point(|existing| *existing <= entry);
                crate::results::result_collection_stats::record_sorted_insert(
                    insert_at,
                    self.inner.len().saturating_sub(insert_at + 1),
                );
            }
            self.inner.pop();
            self.inner.push(entry);
        }
    }

    #[allow(unreachable_code)] // needed because of early return when result_collection_stats feature is enabled
    #[cfg(feature = "buffered_result_collection")]
    fn add_all<I>(&mut self, entries: I)
    where
        I: IntoIterator<Item = QueryResultItem<(), T, O>>,
    {
        #[cfg(feature = "result_collection_stats")]
        {
            let iter = entries.into_iter();
            crate::results::result_collection_stats::record_collector_add_all_call(
                iter.size_hint().0,
            );
            let mut combined = std::mem::take(&mut self.inner).into_vec();
            combined.extend(iter);
            self.inner = SortedVec::from_unsorted(combined);
            if self.inner.len() > self.max_qty {
                self.inner.mutate_vec(|vec| {
                    vec.truncate(self.max_qty);
                });
            }
            return;
        }

        let mut combined = std::mem::take(&mut self.inner).into_vec();
        combined.extend(entries);
        self.inner = SortedVec::from_unsorted(combined);
        if self.inner.len() > self.max_qty {
            self.inner.mutate_vec(|vec| {
                vec.truncate(self.max_qty);
            });
        }
    }

    fn threshold_distance(&self) -> Option<O> {
        let is_full = self.is_full();
        let result = if is_full {
            self.inner.last().map(|n| n.distance)
        } else {
            None
        };
        #[cfg(feature = "result_collection_stats")]
        crate::results::result_collection_stats::record_threshold_distance_call(
            is_full,
            result.is_some(),
        );
        result
    }

    fn into_vec(self) -> Vec<QueryResultItem<(), T, O>> {
        self.inner.into_vec()
    }

    fn into_sorted_vec(self) -> Vec<QueryResultItem<(), T, O>> {
        self.inner.into_vec()
    }
}

#[cfg(feature = "small_n_result_collectors")]
impl<O: Axis<Coord = O>, T> ResultCollection<O, QueryResultItem<(), T, O>>
    for SmallBinaryHeapResultCollection<QueryResultItem<(), T, O>>
{
    fn with_max_qty(max_qty: usize) -> Self {
        debug_assert!(max_qty <= SMALL_RESULT_COLLECTION_MAX_QTY);
        Self {
            max_qty,
            inner: SmallVec::new(),
        }
    }

    fn max_qty(&self) -> usize {
        self.max_qty
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn add(&mut self, entry: QueryResultItem<(), T, O>) {
        #[cfg(feature = "result_collection_stats")]
        crate::results::result_collection_stats::record_collector_add_call();

        if self.inner.len() < self.max_qty {
            #[cfg(feature = "result_collection_stats")]
            crate::results::result_collection_stats::record_heap_add_push();
            self.inner.push(entry);
            let len = self.inner.len();
            sift_up_max_heap(self.inner.as_mut_slice(), len - 1);
        } else if entry < self.inner[0] {
            #[cfg(feature = "result_collection_stats")]
            crate::results::result_collection_stats::record_heap_add_replacement();
            self.inner[0] = entry;
            sift_down_max_heap(self.inner.as_mut_slice(), 0);
        } else {
            #[cfg(feature = "result_collection_stats")]
            crate::results::result_collection_stats::record_heap_add_discard();
        }
    }

    #[allow(unreachable_code)]
    #[cfg(feature = "buffered_result_collection")]
    fn add_all<I>(&mut self, entries: I)
    where
        I: IntoIterator<Item = QueryResultItem<(), T, O>>,
    {
        #[cfg(feature = "result_collection_stats")]
        {
            let iter = entries.into_iter();
            crate::results::result_collection_stats::record_collector_add_all_call(
                iter.size_hint().0,
            );
            let mut combined = std::mem::take(&mut self.inner).into_vec();
            combined.extend(iter);

            if combined.len() > self.max_qty {
                combined.sort_unstable();
                combined.truncate(self.max_qty);
            }

            self.inner.extend(combined);
            heapify_max_heap(self.inner.as_mut_slice());
            return;
        }

        let mut combined = std::mem::take(&mut self.inner).into_vec();
        combined.extend(entries);

        if combined.len() > self.max_qty {
            combined.sort_unstable();
            combined.truncate(self.max_qty);
        }

        self.inner.extend(combined);
        heapify_max_heap(self.inner.as_mut_slice());
    }

    fn threshold_distance(&self) -> Option<O> {
        let is_full = self.is_full();
        let result = if is_full {
            self.inner.first().map(|n| n.distance)
        } else {
            None
        };
        #[cfg(feature = "result_collection_stats")]
        crate::results::result_collection_stats::record_threshold_distance_call(
            is_full,
            result.is_some(),
        );
        result
    }

    fn into_vec(self) -> Vec<QueryResultItem<(), T, O>> {
        self.inner.into_vec()
    }

    fn into_sorted_vec(self) -> Vec<QueryResultItem<(), T, O>> {
        let mut vec = self.inner.into_vec();
        vec.sort();
        vec
    }
}

#[cfg(feature = "small_n_result_collectors")]
impl<O: Axis<Coord = O>, T: Content + PartialOrd> ResultCollection<O, BestQueryResultItem<(), T, O>>
    for SmallBinaryHeapResultCollection<BestQueryResultItem<(), T, O>>
{
    fn with_max_qty(max_qty: usize) -> Self {
        debug_assert!(max_qty <= SMALL_RESULT_COLLECTION_MAX_QTY);
        Self {
            max_qty,
            inner: SmallVec::new(),
        }
    }

    fn max_qty(&self) -> usize {
        self.max_qty
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn add(&mut self, entry: BestQueryResultItem<(), T, O>) {
        #[cfg(feature = "result_collection_stats")]
        crate::results::result_collection_stats::record_collector_add_call();

        if self.inner.len() < self.max_qty {
            #[cfg(feature = "result_collection_stats")]
            crate::results::result_collection_stats::record_heap_add_push();
            self.inner.push(entry);
            let len = self.inner.len();
            sift_up_max_heap(self.inner.as_mut_slice(), len - 1);
        } else if entry < self.inner[0] {
            #[cfg(feature = "result_collection_stats")]
            crate::results::result_collection_stats::record_heap_add_replacement();
            self.inner[0] = entry;
            sift_down_max_heap(self.inner.as_mut_slice(), 0);
        } else {
            #[cfg(feature = "result_collection_stats")]
            crate::results::result_collection_stats::record_heap_add_discard();
        }
    }

    #[allow(unreachable_code)]
    #[cfg(feature = "buffered_result_collection")]
    fn add_all<I>(&mut self, entries: I)
    where
        I: IntoIterator<Item = BestQueryResultItem<(), T, O>>,
    {
        #[cfg(feature = "result_collection_stats")]
        {
            let iter = entries.into_iter();
            crate::results::result_collection_stats::record_collector_add_all_call(
                iter.size_hint().0,
            );
            let mut combined = std::mem::take(&mut self.inner).into_vec();
            combined.extend(iter);

            if combined.len() > self.max_qty {
                combined.sort_unstable();
                combined.truncate(self.max_qty);
            }

            self.inner.extend(combined);
            heapify_max_heap(self.inner.as_mut_slice());
            return;
        }

        let mut combined = std::mem::take(&mut self.inner).into_vec();
        combined.extend(entries);

        if combined.len() > self.max_qty {
            combined.sort_unstable();
            combined.truncate(self.max_qty);
        }

        self.inner.extend(combined);
        heapify_max_heap(self.inner.as_mut_slice());
    }

    fn threshold_distance(&self) -> Option<O> {
        None
    }

    fn into_vec(self) -> Vec<BestQueryResultItem<(), T, O>> {
        self.inner.into_vec()
    }

    fn into_sorted_vec(self) -> Vec<BestQueryResultItem<(), T, O>> {
        let mut vec = self.inner.into_vec();
        vec.sort();
        vec
    }
}

#[cfg(feature = "small_n_result_collectors")]
impl<O: Axis<Coord = O>, T: Content + PartialOrd> BestNeighbourResultCollection<O, T>
    for SmallBinaryHeapResultCollection<BestQueryResultItem<(), T, O>>
{
    #[inline(always)]
    fn threshold_item(&self) -> Option<T> {
        if self.is_full() {
            self.inner.first().map(|n| n.item)
        } else {
            None
        }
    }
}

#[cfg(feature = "small_n_result_collectors")]
impl<O: Axis<Coord = O>, T> ResultCollection<O, QueryResultItem<(), T, O>>
    for SmallSortedVecResultCollection<QueryResultItem<(), T, O>>
{
    fn with_max_qty(max_qty: usize) -> Self {
        debug_assert!(max_qty <= SMALL_RESULT_COLLECTION_MAX_QTY);
        Self {
            max_qty,
            inner: SmallVec::new(),
        }
    }

    fn max_qty(&self) -> usize {
        self.max_qty
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn add(&mut self, entry: QueryResultItem<(), T, O>) {
        #[cfg(feature = "result_collection_stats")]
        crate::results::result_collection_stats::record_collector_add_call();
        small_sorted_insert(&mut self.inner, self.max_qty, entry);
    }

    #[allow(unreachable_code)]
    #[cfg(feature = "buffered_result_collection")]
    fn add_all<I>(&mut self, entries: I)
    where
        I: IntoIterator<Item = QueryResultItem<(), T, O>>,
    {
        #[cfg(feature = "result_collection_stats")]
        {
            let iter = entries.into_iter();
            crate::results::result_collection_stats::record_collector_add_all_call(
                iter.size_hint().0,
            );
            let mut combined = std::mem::take(&mut self.inner).into_vec();
            combined.extend(iter);
            combined.sort_unstable();
            if combined.len() > self.max_qty {
                combined.truncate(self.max_qty);
            }
            self.inner.extend(combined);
            return;
        }

        let mut combined = std::mem::take(&mut self.inner).into_vec();
        combined.extend(entries);
        combined.sort_unstable();
        if combined.len() > self.max_qty {
            combined.truncate(self.max_qty);
        }
        self.inner.extend(combined);
    }

    fn threshold_distance(&self) -> Option<O> {
        let is_full = self.is_full();
        let result = if is_full {
            self.inner.last().map(|n| n.distance)
        } else {
            None
        };
        #[cfg(feature = "result_collection_stats")]
        crate::results::result_collection_stats::record_threshold_distance_call(
            is_full,
            result.is_some(),
        );
        result
    }

    fn into_vec(self) -> Vec<QueryResultItem<(), T, O>> {
        self.inner.into_vec()
    }

    fn into_sorted_vec(self) -> Vec<QueryResultItem<(), T, O>> {
        self.inner.into_vec()
    }
}

#[allow(missing_docs)]
#[cfg(feature = "cargo_asm")]
pub mod cargo_asm {
    use super::*;

    const MAX_QTY: usize = 16;

    const SORTED_NEAREST_INPUTS: [QueryResultItem<(), u32, f64>; MAX_QTY] = [
        QueryResultItem {
            point: (),
            distance: 0.91,
            item: 91,
        },
        QueryResultItem {
            point: (),
            distance: 0.12,
            item: 12,
        },
        QueryResultItem {
            point: (),
            distance: 0.54,
            item: 54,
        },
        QueryResultItem {
            point: (),
            distance: 0.07,
            item: 7,
        },
        QueryResultItem {
            point: (),
            distance: 0.63,
            item: 63,
        },
        QueryResultItem {
            point: (),
            distance: 0.33,
            item: 33,
        },
        QueryResultItem {
            point: (),
            distance: 0.88,
            item: 88,
        },
        QueryResultItem {
            point: (),
            distance: 0.19,
            item: 19,
        },
        QueryResultItem {
            point: (),
            distance: 0.41,
            item: 41,
        },
        QueryResultItem {
            point: (),
            distance: 0.02,
            item: 2,
        },
        QueryResultItem {
            point: (),
            distance: 0.76,
            item: 76,
        },
        QueryResultItem {
            point: (),
            distance: 0.27,
            item: 27,
        },
        QueryResultItem {
            point: (),
            distance: 0.69,
            item: 69,
        },
        QueryResultItem {
            point: (),
            distance: 0.58,
            item: 58,
        },
        QueryResultItem {
            point: (),
            distance: 0.15,
            item: 15,
        },
        QueryResultItem {
            point: (),
            distance: 0.47,
            item: 47,
        },
    ];

    const BEST_INPUTS: [BestQueryResultItem<(), u32, f64>; MAX_QTY] = [
        BestQueryResultItem {
            point: (),
            distance: 0.91,
            item: 91,
        },
        BestQueryResultItem {
            point: (),
            distance: 0.12,
            item: 12,
        },
        BestQueryResultItem {
            point: (),
            distance: 0.54,
            item: 54,
        },
        BestQueryResultItem {
            point: (),
            distance: 0.07,
            item: 7,
        },
        BestQueryResultItem {
            point: (),
            distance: 0.63,
            item: 63,
        },
        BestQueryResultItem {
            point: (),
            distance: 0.33,
            item: 33,
        },
        BestQueryResultItem {
            point: (),
            distance: 0.88,
            item: 88,
        },
        BestQueryResultItem {
            point: (),
            distance: 0.19,
            item: 19,
        },
        BestQueryResultItem {
            point: (),
            distance: 0.41,
            item: 41,
        },
        BestQueryResultItem {
            point: (),
            distance: 0.02,
            item: 2,
        },
        BestQueryResultItem {
            point: (),
            distance: 0.76,
            item: 76,
        },
        BestQueryResultItem {
            point: (),
            distance: 0.27,
            item: 27,
        },
        BestQueryResultItem {
            point: (),
            distance: 0.69,
            item: 69,
        },
        BestQueryResultItem {
            point: (),
            distance: 0.58,
            item: 58,
        },
        BestQueryResultItem {
            point: (),
            distance: 0.15,
            item: 15,
        },
        BestQueryResultItem {
            point: (),
            distance: 0.47,
            item: 47,
        },
    ];

    #[inline(always)]
    fn checksum_nearest(results: &[QueryResultItem<(), u32, f64>]) -> (usize, u64, u64) {
        let mut checksum_item = 0u64;
        let mut checksum_dist = 0u64;

        for entry in results {
            checksum_item = checksum_item.wrapping_add(entry.item as u64);
            checksum_dist = checksum_dist.wrapping_add(entry.distance.to_bits());
        }

        (results.len(), checksum_item, checksum_dist)
    }

    #[inline(always)]
    fn checksum_best(results: &[BestQueryResultItem<(), u32, f64>]) -> (usize, u64, u64) {
        let mut checksum_item = 0u64;
        let mut checksum_dist = 0u64;

        for entry in results {
            checksum_item = checksum_item.wrapping_add(entry.item as u64);
            checksum_dist = checksum_dist.wrapping_add(entry.distance.to_bits());
        }

        (results.len(), checksum_item, checksum_dist)
    }

    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_sorted_vec_result_collection_add_cargo_asm_hook() -> (usize, u64, u64) {
        let mut results =
            SortedVecResultCollection::<QueryResultItem<(), u32, f64>>::with_max_qty(MAX_QTY);
        for entry in SORTED_NEAREST_INPUTS {
            results.add(entry);
        }
        let vec = results.into_sorted_vec();
        checksum_nearest(&vec)
    }

    #[cfg(feature = "buffered_result_collection")]
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_sorted_vec_result_collection_add_all_cargo_asm_hook() -> (usize, u64, u64) {
        let mut results =
            SortedVecResultCollection::<QueryResultItem<(), u32, f64>>::with_max_qty(MAX_QTY);
        results.add_all(SORTED_NEAREST_INPUTS);
        let vec = results.into_sorted_vec();
        checksum_nearest(&vec)
    }

    #[cfg(not(feature = "small_n_result_collectors"))]
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_threshold_vec_result_collection_add_cargo_asm_hook() -> (usize, u64, u64) {
        let mut results =
            ThresholdVecResultCollection::<QueryResultItem<(), u32, f64>>::with_max_qty(MAX_QTY);
        for entry in SORTED_NEAREST_INPUTS {
            results.add(entry);
        }
        let vec = results.into_sorted_vec();
        checksum_nearest(&vec)
    }

    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_binary_heap_result_collection_add_cargo_asm_hook() -> (usize, u64, u64) {
        let mut results =
            BinaryHeapResultCollection::<BestQueryResultItem<(), u32, f64>>::with_max_qty(MAX_QTY);
        for entry in BEST_INPUTS {
            results.add(entry);
        }
        let vec = results.into_sorted_vec();
        checksum_best(&vec)
    }

    #[cfg(feature = "buffered_result_collection")]
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_binary_heap_result_collection_add_all_cargo_asm_hook() -> (usize, u64, u64) {
        let mut results =
            BinaryHeapResultCollection::<BestQueryResultItem<(), u32, f64>>::with_max_qty(MAX_QTY);
        results.add_all(BEST_INPUTS);
        let vec = results.into_sorted_vec();
        checksum_best(&vec)
    }

    #[cfg(feature = "small_n_result_collectors")]
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_small_sorted_vec_result_collection_add_cargo_asm_hook() -> (usize, u64, u64) {
        let mut results =
            SmallSortedVecResultCollection::<QueryResultItem<(), u32, f64>>::with_max_qty(MAX_QTY);
        for entry in SORTED_NEAREST_INPUTS {
            results.add(entry);
        }
        let vec = results.into_sorted_vec();
        checksum_nearest(&vec)
    }

    #[cfg(feature = "small_n_result_collectors")]
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_small_sorted_vec_result_collection_add_all_cargo_asm_hook() -> (usize, u64, u64) {
        let mut results =
            SmallSortedVecResultCollection::<QueryResultItem<(), u32, f64>>::with_max_qty(MAX_QTY);
        results.add_all(SORTED_NEAREST_INPUTS);
        let vec = results.into_sorted_vec();
        checksum_nearest(&vec)
    }

    #[cfg(feature = "small_n_result_collectors")]
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_small_binary_heap_result_collection_add_cargo_asm_hook() -> (usize, u64, u64) {
        let mut results =
            SmallBinaryHeapResultCollection::<BestQueryResultItem<(), u32, f64>>::with_max_qty(
                MAX_QTY,
            );
        for entry in BEST_INPUTS {
            results.add(entry);
        }
        let vec = results.into_sorted_vec();
        checksum_best(&vec)
    }

    #[cfg(feature = "small_n_result_collectors")]
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_small_binary_heap_result_collection_add_all_cargo_asm_hook() -> (usize, u64, u64) {
        let mut results =
            SmallBinaryHeapResultCollection::<BestQueryResultItem<(), u32, f64>>::with_max_qty(
                MAX_QTY,
            );
        results.add_all(BEST_INPUTS);
        let vec = results.into_sorted_vec();
        checksum_best(&vec)
    }
}

impl<O: Axis<Coord = O>, E: Ord> ResultCollection<O, E> for Vec<E> {
    fn with_max_qty(max_qty: usize) -> Self {
        <Self as ResultCollection<O, E>>::with_max_qty_and_capacity(max_qty, None)
    }

    fn with_max_qty_and_capacity(max_qty: usize, result_capacity: Option<NonZeroUsize>) -> Self {
        if max_qty == usize::MAX {
            // Unsorted query path: start with reasonable capacity to avoid
            // the first several realloc waves when many candidates fall within
            // the radius.
            Vec::with_capacity(
                result_capacity.map_or(DEFAULT_UNSORTED_RESULT_CAPACITY, NonZeroUsize::get),
            )
        } else {
            Vec::with_capacity(max_qty)
        }
    }

    fn max_qty(&self) -> usize {
        usize::MAX
    }

    fn len(&self) -> usize {
        Vec::len(self)
    }

    fn add(&mut self, entry: E) {
        self.push(entry)
    }

    #[cfg(feature = "buffered_result_collection")]
    fn add_all<I>(&mut self, entries: I)
    where
        I: IntoIterator<Item = E>,
    {
        self.extend(entries);
    }

    fn threshold_distance(&self) -> Option<O> {
        None
    }

    fn into_vec(self) -> Vec<E> {
        self
    }

    fn into_sorted_vec(mut self) -> Vec<E> {
        self.sort();
        self
    }
}

#[cfg(test)]
#[cfg(all(not(feature = "small_n_result_collectors"), test))]
mod tests {
    use super::*;

    const INPUTS: [QueryResultItem<(), u32, f64>; 5] = [
        QueryResultItem {
            point: (),
            distance: 0.4,
            item: 1,
        },
        QueryResultItem {
            point: (),
            distance: 0.1,
            item: 2,
        },
        QueryResultItem {
            point: (),
            distance: 0.8,
            item: 3,
        },
        QueryResultItem {
            point: (),
            distance: 0.3,
            item: 4,
        },
        QueryResultItem {
            point: (),
            distance: 0.6,
            item: 5,
        },
    ];

    #[test]
    fn threshold_vec_sorted_output() {
        let k = 3;
        let mut results =
            ThresholdVecResultCollection::<QueryResultItem<(), u32, f64>>::with_max_qty(k);
        assert_eq!(results.max_qty(), k);
        assert_eq!(results.len(), 0);
        for entry in INPUTS {
            results.add(entry);
        }
        assert_eq!(results.len(), k);
        let sorted = results.into_sorted_vec();
        assert_eq!(sorted.len(), k);
        for i in 1..sorted.len() {
            assert!(sorted[i - 1].distance <= sorted[i].distance);
        }
    }

    #[test]
    fn threshold_vec_threshold_distance() {
        let mut results =
            ThresholdVecResultCollection::<QueryResultItem<(), u32, f64>>::with_max_qty(3);
        assert_eq!(results.max_qty(), 3);
        assert_eq!(results.len(), 0);
        assert!(results.threshold_distance().is_none());
        results.add(INPUTS[0]);
        assert_eq!(results.len(), 1);
        assert!(results.threshold_distance().is_none());
        results.add(INPUTS[1]);
        assert_eq!(results.len(), 2);
        assert!(results.threshold_distance().is_none());
        results.add(INPUTS[2]); // now full
        assert_eq!(results.len(), 3);
        assert!(results.threshold_distance().is_some());
    }

    #[test]
    fn threshold_vec_select_nth_unstable() {
        let mut results =
            ThresholdVecResultCollection::<QueryResultItem<(), u32, f64>>::with_max_qty(3);
        assert_eq!(results.max_qty(), 3);
        assert_eq!(results.len(), 0);
        for entry in INPUTS.iter().take(3) {
            results.add(*entry); // 0.4, 0.1, 0.8: farthest is 0.8
        }
        assert_eq!(results.len(), 3);
        // Add closer than farthest (0.3 < 0.8): should replace it
        results.add(INPUTS[3]); // 0.3
        assert_eq!(results.len(), 3);
        let sorted = results.into_sorted_vec();
        assert_eq!(sorted.len(), 3);
        assert!(sorted.iter().any(|item| item.distance == 0.3));
        assert!(!sorted.iter().any(|item| item.distance == 0.8));
    }

    fn result_item(item: u32, distance: f64) -> QueryResultItem<(), u32, f64> {
        QueryResultItem {
            point: (),
            item,
            distance,
        }
    }

    #[test]
    fn threshold_vec_linear_insertion_handles_front_middle_tail_and_ties() {
        let mut results =
            ThresholdVecResultCollection::<QueryResultItem<(), u32, f64>>::with_max_qty(4);
        for (item, distance) in [(1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0)] {
            results.add(result_item(item, distance));
        }

        results.add(result_item(5, 3.5));
        assert_eq!(
            results
                .inner
                .iter()
                .map(|entry| (entry.item, entry.distance))
                .collect::<Vec<_>>(),
            [(1, 1.0), (2, 2.0), (3, 3.0), (5, 3.5)]
        );

        results.add(result_item(6, 1.5));
        assert_eq!(
            results
                .inner
                .iter()
                .map(|entry| (entry.item, entry.distance))
                .collect::<Vec<_>>(),
            [(1, 1.0), (6, 1.5), (2, 2.0), (3, 3.0)]
        );

        results.add(result_item(7, 0.5));
        assert_eq!(
            results
                .inner
                .iter()
                .map(|entry| (entry.item, entry.distance))
                .collect::<Vec<_>>(),
            [(7, 0.5), (1, 1.0), (6, 1.5), (2, 2.0)]
        );

        results.add(result_item(8, 1.0));
        assert_eq!(
            results
                .inner
                .iter()
                .map(|entry| (entry.item, entry.distance))
                .collect::<Vec<_>>(),
            [(7, 0.5), (1, 1.0), (8, 1.0), (6, 1.5)]
        );
    }
}

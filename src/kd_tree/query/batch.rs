//! Batch query execution.
//!
//! Batch queries run many query points against one tree in a single call. The
//! API deliberately says as little as possible about *how* that happens, so
//! that the execution strategy can keep improving without breaking callers.
//!
//! # What is guaranteed
//!
//! * Results are addressed by the **index of the query point in the input
//!   slice**. [`BatchResults`] and [`BatchGroups`] are indexed the same way as
//!   `queries`, and [`BatchQueryBuilder::for_each`] passes the index alongside
//!   each result.
//! * Every query point produces exactly one entry.
//! * Results are identical to running the same query individually through
//!   [`KdTree::query`](crate::kd_tree::KdTree::query).
//!
//! # What is explicitly *not* guaranteed
//!
//! These are the degrees of freedom the implementation reserves, and they may
//! change in any release:
//!
//! * The order in which query points are executed.
//! * Which thread executes any given query point, or how many threads are used.
//! * Whether query points are copied, reordered, grouped, or interleaved
//!   internally.
//! * The order in which a [`for_each`](BatchQueryBuilder::for_each) visitor is
//!   invoked, and which thread invokes it.
//! * The in-memory representation of [`BatchResults`] and [`BatchGroups`].
//!
//! Code that depends on any of the above is relying on unspecified behaviour.

use std::collections::BinaryHeap;
use std::num::NonZeroUsize;
use std::ops::{Deref, Index};
#[cfg(feature = "multi-threaded")]
use std::sync::Arc;

#[cfg(feature = "multi-threaded")]
use rayon::prelude::*;

/// Re-exported so that callers of
/// [`Executor::parallel_in_pool`] can name and build a pool without taking
/// their own `rayon` dependency, which would have to be version-matched
/// against the one Kiddo links.
#[cfg(feature = "multi-threaded")]
pub use rayon::{ThreadPool, ThreadPoolBuilder};

/// Static chunks per pool thread used by
/// [`Executor::with_default_static_chunking`].
///
/// Four is enough slack for the pool to even out uneven query costs, while
/// still being far fewer tasks than adaptive splitting creates.
#[cfg(feature = "multi-threaded")]
pub const DEFAULT_STATIC_CHUNK_THREAD_MULTIPLIER: NonZeroUsize = NonZeroUsize::new(4).unwrap();

/// Ceiling on the task-size floor imposed by
/// [`Executor::with_static_chunk_thread_multiplier`].
///
/// A thread-proportional chunk count is constant in the batch size, so on a
/// large batch each chunk is large, and load imbalance between chunks costs
/// more than the dispatch coordination it saves. Measured on a 32-thread pool
/// against a 2^24..2^27 `f64` sweep at 100,000 queries — where the uncapped
/// floor works out at 781 — static chunking was 10-15% *slower* than letting
/// rayon split adaptively, while at 1,000-4,000 queries it was 23-47% faster.
/// Capping the floor keeps the small-batch win and lets large batches fall
/// back to adaptive splitting, which is what work stealing needs.
#[cfg(feature = "multi-threaded")]
pub const MAX_STATIC_CHUNK_QUERIES: usize = 128;

/// Batch size below which [`Executor::with_serial_fallback`] runs on the
/// calling thread.
///
/// Parallel dispatch costs a fixed amount per batch, which a small enough
/// batch never recovers. Measured on a 2^21-point 3D `f32` tree with
/// [`with_default_static_chunking`](Executor::with_default_static_chunking)
/// also enabled, `nearest_one` broke even at roughly 110 query points: at 96
/// points serial reached 8.66 Melem/s against parallel's 7.99, and at 128 that
/// had reversed to 8.77 against 9.69.
///
/// A query count is a crude proxy for the work in a batch, and this default is
/// taken from the cheapest query there is. Costlier queries — a large `k`, or
/// a high-dimensional tree — earn back the dispatch at far smaller batches, so
/// prefer a lower threshold for those, or leave the fallback off entirely.
#[cfg(feature = "multi-threaded")]
pub const DEFAULT_SERIAL_FALLBACK_THRESHOLD: usize = 128;

use super::builder::{
    ApproxQueryBuilder, BestNWithinQueryBuilder, ExclusiveBoundariesQueryBuilder,
    ExecuteQueryBuilder, NearestNQueryBuilder, NearestOneQueryBuilder,
    PeriodicBoundaryConditionQueryBuilder, UnsortedQueryBuilder, WithDistancesQueryBuilder,
    WithItemsQueryBuilder, WithPointsQueryBuilder, WithResultCapacityQueryBuilder,
    WithinQueryBuilder, WithoutDistancesQueryBuilder, WithoutItemsQueryBuilder,
    WithoutPointsQueryBuilder,
};
use crate::dist::DistanceMetric;
use crate::results::query_result_item::QueryResultItem;

/// How a batch query should be executed.
///
/// This is an opaque scheduling policy rather than a thread-pool handle. It
/// carries a coarse choice (serial vs parallel) plus advisory hints; the
/// meaning of the hints, and the machinery used to honour them, may change
/// between releases.
///
/// Multi-threaded execution requires the `multi-threaded` crate feature, which
/// is enabled by default. Without it the parallel constructors and hints are
/// not compiled, and [`new`](Self::new) is equivalent to
/// [`serial`](Self::serial).
///
/// # Small batches
///
/// Splitting a batch across threads costs a fixed amount per call, so a batch
/// small enough is faster to run on the calling thread. Two opt-in hints
/// address that, both no-ops for [`serial`](Self::serial):
///
/// * [`with_default_static_chunking`](Self::with_default_static_chunking)
///   stops the batch being split into tasks smaller than a size derived from
///   the pool, instead of subdividing it as far as the splitter likes.
/// * [`with_serial_fallback`](Self::with_serial_fallback) runs batches below a
///   threshold on the calling thread, skipping the dispatch entirely.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "multi-threaded")] {
/// use kiddo::batch::Executor;
///
/// let executor = Executor::parallel()
///     .with_default_static_chunking()
///     .with_serial_fallback();
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct Executor {
    kind: ExecutorKind,
    #[cfg(feature = "multi-threaded")]
    static_chunk_thread_multiplier: Option<NonZeroUsize>,
    #[cfg(feature = "multi-threaded")]
    serial_fallback_threshold: Option<usize>,
}

#[derive(Clone, Debug)]
enum ExecutorKind {
    Serial,
    #[cfg(feature = "multi-threaded")]
    Parallel,
    /// Parallel, but confined to a pool the caller owns and configures.
    #[cfg(feature = "multi-threaded")]
    ParallelInPool(Arc<ThreadPool>),
}

impl PartialEq for ExecutorKind {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Serial, Self::Serial) => true,
            #[cfg(feature = "multi-threaded")]
            (Self::Parallel, Self::Parallel) => true,
            // Thread pools have no equality of their own, so two executors are
            // the same only when they name the very same pool.
            #[cfg(feature = "multi-threaded")]
            (Self::ParallelInPool(left), Self::ParallelInPool(right)) => Arc::ptr_eq(left, right),
            _ => false,
        }
    }
}

impl Eq for ExecutorKind {}

impl PartialEq for Executor {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        #[cfg(feature = "multi-threaded")]
        {
            self.kind == other.kind
                && self.static_chunk_thread_multiplier == other.static_chunk_thread_multiplier
                && self.serial_fallback_threshold == other.serial_fallback_threshold
        }

        #[cfg(not(feature = "multi-threaded"))]
        {
            self.kind == other.kind
        }
    }
}

impl Eq for Executor {}

impl Default for Executor {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Executor {
    /// Creates the default executor.
    ///
    /// The default is `parallel` with the `multi-threaded`
    /// feature enabled — which it is unless a caller turns default features off
    /// — and [`serial`](Self::serial) without it.
    #[inline]
    pub fn new() -> Self {
        #[cfg(feature = "multi-threaded")]
        {
            Self::parallel()
        }

        #[cfg(not(feature = "multi-threaded"))]
        {
            Self::serial()
        }
    }

    /// Creates an executor that runs every query on the calling thread.
    ///
    /// Query points are still processed in an unspecified order.
    #[inline]
    pub fn serial() -> Self {
        Self {
            kind: ExecutorKind::Serial,
            #[cfg(feature = "multi-threaded")]
            static_chunk_thread_multiplier: None,
            #[cfg(feature = "multi-threaded")]
            serial_fallback_threshold: None,
        }
    }

    /// Creates an executor that may spread queries across multiple threads,
    /// using Rayon's global pool.
    ///
    /// Requires the `multi-threaded` feature.
    #[cfg(feature = "multi-threaded")]
    #[inline]
    pub fn parallel() -> Self {
        Self {
            kind: ExecutorKind::Parallel,
            static_chunk_thread_multiplier: None,
            serial_fallback_threshold: None,
        }
    }

    /// Creates an executor that runs every batch on `pool` rather than Rayon's
    /// global pool.
    ///
    /// Use this to bound how many cores batch queries may take, to keep them
    /// off a pool shared with latency-sensitive work, or to reuse one
    /// configured pool across many batches. The pool is shared, not consumed,
    /// so one pool can back any number of executors.
    ///
    /// Requires the `multi-threaded` feature.
    ///
    /// # Example
    ///
    /// ```
    /// # #[cfg(feature = "multi-threaded")] {
    /// use kiddo::batch::{Executor, ThreadPoolBuilder};
    /// use std::sync::Arc;
    ///
    /// let pool = Arc::new(ThreadPoolBuilder::new().num_threads(4).build().unwrap());
    /// let executor = Executor::parallel_in_pool(Arc::clone(&pool));
    /// # }
    /// ```
    #[cfg(feature = "multi-threaded")]
    #[inline]
    pub fn parallel_in_pool(pool: Arc<ThreadPool>) -> Self {
        Self {
            kind: ExecutorKind::ParallelInPool(pool),
            static_chunk_thread_multiplier: None,
            serial_fallback_threshold: None,
        }
    }

    /// Stops the batch being split below `queries / (threads * multiplier)`,
    /// rather than letting the work-stealing splitter subdivide it as far as
    /// it likes.
    ///
    /// Adaptive splitting reacts to load imbalance, but pays coordination cost
    /// per split. Bounding the split trades some of that reactivity for a
    /// cheaper dispatch, which is what dominates on small batches. The bound
    /// is derived from the pool size rather than given as an absolute, so it
    /// always leaves at least `threads * multiplier` tasks and cannot starve
    /// the pool, and is capped at [`MAX_STATIC_CHUNK_QUERIES`] so that it
    /// stops applying to batches large enough to need work stealing instead.
    ///
    /// This helps small batches and does nothing for large ones. It is not a
    /// general throughput knob.
    ///
    /// This is advisory, and a no-op for [`serial`](Self::serial).
    ///
    /// Requires the `multi-threaded` feature.
    #[cfg(feature = "multi-threaded")]
    #[inline]
    pub fn with_static_chunk_thread_multiplier(mut self, multiplier: NonZeroUsize) -> Self {
        self.static_chunk_thread_multiplier = Some(multiplier);
        self
    }

    /// Enables static chunking at
    /// [`DEFAULT_STATIC_CHUNK_THREAD_MULTIPLIER`] chunks per thread.
    ///
    /// Requires the `multi-threaded` feature.
    #[cfg(feature = "multi-threaded")]
    #[inline]
    pub fn with_default_static_chunking(self) -> Self {
        self.with_static_chunk_thread_multiplier(DEFAULT_STATIC_CHUNK_THREAD_MULTIPLIER)
    }

    /// Runs batches of fewer than `threshold` query points on the calling
    /// thread, skipping parallel dispatch.
    ///
    /// # Choosing a threshold
    ///
    /// Parallel dispatch costs roughly a fixed amount per batch, so the
    /// break-even point is where the batch's *total* work first exceeds it:
    /// `threshold ≈ dispatch_cost / per_query_cost`. Dispatch cost is a
    /// property of the machine and pool; per-query cost is a property of the
    /// query, and is what varies. It rises with:
    ///
    /// * **`k`** — the dominant factor. A `nearest_n(50)` point costs several
    ///   times a `nearest_one` point, so it breaks even at a proportionately
    ///   smaller batch.
    /// * **Tree size**, but only weakly: descent is logarithmic, and past the
    ///   point where the tree stops fitting in cache the per-query cost is
    ///   dominated by misses rather than by comparisons.
    /// * **Dimensionality**, through both wider distance computations and less
    ///   effective pruning.
    /// * **Radius**, for the `within` family, since the cost scales with the
    ///   number of matches rather than with the depth of the descent.
    ///
    /// The practical recipe is to time one batch each way at a size near the
    /// suspected threshold, rather than to derive it. Failing that, scale
    /// [`DEFAULT_SERIAL_FALLBACK_THRESHOLD`] down by the ratio of your query's
    /// cost to a `nearest_one`: for `nearest_n`, dividing by `k` is a
    /// serviceable first guess. Erring low is the safer direction — the
    /// penalty for parallelising slightly too small a batch is a few
    /// microseconds, while the penalty for serialising a large one is the
    /// whole speedup.
    ///
    /// This is advisory, and a no-op for [`serial`](Self::serial).
    ///
    /// Requires the `multi-threaded` feature.
    #[cfg(feature = "multi-threaded")]
    #[inline]
    pub fn with_serial_fallback_threshold(mut self, threshold: usize) -> Self {
        self.serial_fallback_threshold = Some(threshold);
        self
    }

    /// Enables the serial fallback at [`DEFAULT_SERIAL_FALLBACK_THRESHOLD`]
    /// query points.
    ///
    /// Requires the `multi-threaded` feature.
    #[cfg(feature = "multi-threaded")]
    #[inline]
    pub fn with_serial_fallback(self) -> Self {
        self.with_serial_fallback_threshold(DEFAULT_SERIAL_FALLBACK_THRESHOLD)
    }

    /// Whether a batch of `query_count` points should run in parallel, which
    /// requires a parallel executor and a batch above any serial-fallback
    /// threshold.
    #[cfg(feature = "multi-threaded")]
    #[inline]
    fn is_parallel(&self, query_count: usize) -> bool {
        if matches!(self.kind, ExecutorKind::Serial) {
            return false;
        }
        self.serial_fallback_threshold
            .is_none_or(|threshold| query_count >= threshold)
    }

    /// The pool to run on, or `None` for Rayon's global pool.
    #[cfg(feature = "multi-threaded")]
    #[inline]
    fn pool(&self) -> Option<&ThreadPool> {
        match &self.kind {
            ExecutorKind::ParallelInPool(pool) => Some(pool),
            _ => None,
        }
    }

    /// Smallest number of queries a task may be split down to.
    ///
    /// Returns 1 — rayon's own default, i.e. split as finely as the scheduler
    /// likes — unless static chunking asked for a coarser floor. Because the
    /// floor is derived from the pool size, it always leaves at least
    /// `threads * multiplier` tasks available, so it can never starve the pool
    /// the way a caller-supplied absolute minimum could.
    ///
    /// The floor is additionally capped at [`MAX_STATIC_CHUNK_QUERIES`], so it
    /// stops binding on large batches. A fixed number of chunks means each one
    /// grows with the batch, and past a point the imbalance between chunks
    /// costs more than the dispatch it saves.
    #[cfg(feature = "multi-threaded")]
    #[inline]
    fn static_chunk_len(&self, query_count: usize) -> usize {
        let Some(multiplier) = self.static_chunk_thread_multiplier else {
            return 1;
        };
        let threads = self
            .pool()
            .map_or_else(rayon::current_num_threads, ThreadPool::current_num_threads);
        let chunks = threads.max(1).saturating_mul(multiplier.get());
        query_count
            .div_ceil(chunks)
            .clamp(1, MAX_STATIC_CHUNK_QUERIES)
    }
}

/// One result per query point, addressed by query index.
///
/// Returned by [`BatchQueryBuilder::execute`] for query families that produce a
/// single result per query point, such as
/// [`nearest_one`](BatchQueryBuilder::nearest_one).
///
/// The internal representation is not part of the public API; access results
/// through the slice-oriented methods, or take ownership with
/// [`into_vec`](Self::into_vec).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchResults<R> {
    results: Vec<R>,
}

impl<R> BatchResults<R> {
    #[inline]
    pub(crate) fn from_vec(results: Vec<R>) -> Self {
        Self { results }
    }

    /// Returns the number of query points in the batch.
    #[inline]
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Returns `true` if the batch contained no query points.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Returns the result for the query point at `index`, if in range.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&R> {
        self.results.get(index)
    }

    /// Returns all results, in query-point order.
    #[inline]
    pub fn as_slice(&self) -> &[R] {
        &self.results
    }

    /// Iterates over results in query-point order.
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, R> {
        self.results.iter()
    }

    /// Converts into a `Vec` of results, in query-point order.
    #[inline]
    pub fn into_vec(self) -> Vec<R> {
        self.results
    }
}

impl<R> Deref for BatchResults<R> {
    type Target = [R];

    #[inline]
    fn deref(&self) -> &[R] {
        &self.results
    }
}

impl<R> Index<usize> for BatchResults<R> {
    type Output = R;

    #[inline]
    fn index(&self, index: usize) -> &R {
        &self.results[index]
    }
}

impl<R> IntoIterator for BatchResults<R> {
    type Item = R;
    type IntoIter = std::vec::IntoIter<R>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.results.into_iter()
    }
}

impl<'r, R> IntoIterator for &'r BatchResults<R> {
    type Item = &'r R;
    type IntoIter = std::slice::Iter<'r, R>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.results.iter()
    }
}

/// A group of results per query point, addressed by query index.
///
/// Returned by [`BatchQueryBuilder::execute`] for query families that produce
/// zero or more results per query point, such as
/// [`nearest_n`](BatchQueryBuilder::nearest_n),
/// [`within`](BatchQueryBuilder::within) and
/// [`best_n_within`](BatchQueryBuilder::best_n_within).
///
/// Groups are exposed as slices so that the underlying storage can change —
/// for example to a single flat allocation — without breaking callers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchGroups<R> {
    groups: Vec<Vec<R>>,
}

impl<R> BatchGroups<R> {
    #[inline]
    pub(crate) fn from_nested_vec(groups: Vec<Vec<R>>) -> Self {
        Self { groups }
    }

    /// Returns the number of query points in the batch.
    #[inline]
    pub fn len(&self) -> usize {
        self.groups.len()
    }

    /// Returns `true` if the batch contained no query points.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    /// Returns the total number of results across every query point.
    #[inline]
    pub fn total_len(&self) -> usize {
        self.groups.iter().map(Vec::len).sum()
    }

    /// Returns the results for the query point at `index`, if in range.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&[R]> {
        self.groups.get(index).map(Vec::as_slice)
    }

    /// Iterates over per-query-point result groups, in query-point order.
    #[inline]
    pub fn iter(&self) -> GroupIter<'_, R> {
        GroupIter {
            inner: self.groups.iter(),
        }
    }

    /// Converts into one `Vec` per query point, in query-point order.
    #[inline]
    pub fn into_nested_vec(self) -> Vec<Vec<R>> {
        self.groups
    }

    /// Concatenates every group into one `Vec`, in query-point order.
    ///
    /// Group boundaries are lost; use this when only the union of results
    /// matters.
    #[inline]
    pub fn into_flat_vec(self) -> Vec<R> {
        let mut flat = Vec::with_capacity(self.total_len());
        for group in self.groups {
            flat.extend(group);
        }
        flat
    }
}

impl<R> Index<usize> for BatchGroups<R> {
    type Output = [R];

    #[inline]
    fn index(&self, index: usize) -> &[R] {
        &self.groups[index]
    }
}

impl<'g, R> IntoIterator for &'g BatchGroups<R> {
    type Item = &'g [R];
    type IntoIter = GroupIter<'g, R>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator over the per-query-point result groups of a [`BatchGroups`].
#[derive(Clone, Debug)]
pub struct GroupIter<'g, R> {
    inner: std::slice::Iter<'g, Vec<R>>,
}

impl<'g, R> Iterator for GroupIter<'g, R> {
    type Item = &'g [R];

    #[inline]
    fn next(&mut self) -> Option<&'g [R]> {
        self.inner.next().map(Vec::as_slice)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<R> ExactSizeIterator for GroupIter<'_, R> {}

impl<R> DoubleEndedIterator for GroupIter<'_, R> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back().map(Vec::as_slice)
    }
}

/// Maps a single-query result type onto the container used to hold a batch of
/// them.
///
/// Implementation detail of [`BatchQueryBuilder::execute`].
#[doc(hidden)]
pub trait BatchCollect: Sized {
    #[doc(hidden)]
    type Batch;

    #[doc(hidden)]
    fn collect_batch(results: Vec<Self>) -> Self::Batch;
}

impl<P, T, D> BatchCollect for QueryResultItem<P, T, D> {
    type Batch = BatchResults<Self>;

    #[inline]
    fn collect_batch(results: Vec<Self>) -> Self::Batch {
        BatchResults::from_vec(results)
    }
}

impl<R> BatchCollect for Vec<R> {
    type Batch = BatchGroups<R>;

    #[inline]
    fn collect_batch(results: Vec<Self>) -> Self::Batch {
        BatchGroups::from_nested_vec(results)
    }
}

impl<R: Ord> BatchCollect for BinaryHeap<R> {
    type Batch = BatchGroups<R>;

    #[inline]
    fn collect_batch(results: Vec<Self>) -> Self::Batch {
        BatchGroups::from_nested_vec(
            results
                .into_iter()
                .map(BinaryHeap::into_sorted_vec)
                .collect(),
        )
    }
}

/// A configured single-point query that a batch can re-point at each query
/// point in turn.
///
/// Implementation detail of [`BatchQueryBuilder`].
#[doc(hidden)]
pub trait BatchQuerySink<'a, A, const K: usize>: Copy {
    #[doc(hidden)]
    fn set_query_point(&mut self, query: &'a [A; K]);
}

/// A fluent batch query builder, created by
/// [`KdTree::query_batch`](crate::kd_tree::KdTree::query_batch).
///
/// A batch query is configured exactly like a single-point query — the same
/// query families, modifiers and result projections are available, with the
/// same names — and differs only in its terminal methods, which return one
/// result per query point instead of one result overall.
///
/// ```
/// use kiddo::batch::Executor;
/// use kiddo::dist::SquaredEuclidean;
/// use kiddo::leaf_strategy::FlatVec;
/// use kiddo::{Eytzinger, KdTree};
/// use std::num::NonZeroUsize;
///
/// type Tree = KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, 2, 4>, 2, 4>;
///
/// let entries = [(10_u32, [0.0, 0.0]), (11_u32, [1.0, 1.0]), (12_u32, [0.2, 0.1])];
/// let tree = Tree::new_from_entries(&entries).unwrap();
///
/// let queries = [[0.1, 0.1], [0.9, 0.9], [0.25, 0.15]];
///
/// let results = tree
///     .query_batch(&queries)
///     .nearest_one::<SquaredEuclidean<f64>>()
///     .execute();
///
/// assert_eq!(results.len(), 3);
/// assert_eq!(results[0].item, 12);
/// assert_eq!(results[1].item, 11);
/// ```
///
/// See the [module documentation](crate::batch) for exactly which properties of batch
/// execution are guaranteed and which are free to change.
pub struct BatchQueryBuilder<'a, Qb, A, const K: usize> {
    /// The configured single-point query, cloned per query point at execution
    /// time. `None` only when the batch is empty, in which case there is no
    /// query point to anchor the prototype's borrow.
    prototype: Option<Qb>,
    queries: &'a [[A; K]],
    executor: Executor,
}

impl<'a, Qb, A, const K: usize> BatchQueryBuilder<'a, Qb, A, K> {
    #[inline]
    pub(crate) fn new(prototype: Option<Qb>, queries: &'a [[A; K]]) -> Self {
        Self {
            prototype,
            queries,
            executor: Executor::new(),
        }
    }

    /// Applies a single-point builder transition to the batch's prototype.
    #[inline]
    fn map<Qb2>(self, transition: impl FnOnce(Qb) -> Qb2) -> BatchQueryBuilder<'a, Qb2, A, K> {
        BatchQueryBuilder {
            prototype: self.prototype.map(transition),
            queries: self.queries,
            executor: self.executor,
        }
    }

    /// Returns the number of query points in the batch.
    #[inline]
    pub fn len(&self) -> usize {
        self.queries.len()
    }

    /// Returns `true` if the batch contains no query points.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.queries.is_empty()
    }

    /// Sets the execution policy for this batch.
    ///
    /// Defaults to [`Executor::new`].
    #[inline]
    pub fn with_executor(mut self, executor: &Executor) -> Self {
        self.executor = executor.clone();
        self
    }
}

/// Query family selection and result projection.
///
/// Each method mirrors the identically-named method on
/// [`QueryBuilder`](crate::kd_tree::QueryBuilder) and is available under
/// exactly the same conditions.
impl<'a, Qb, A, const K: usize> BatchQueryBuilder<'a, Qb, A, K> {
    /// Excludes stored point coordinates from the eventual query results.
    #[inline]
    pub fn without_points(self) -> BatchQueryBuilder<'a, Qb::Output, A, K>
    where
        Qb: WithoutPointsQueryBuilder,
    {
        self.map(Qb::without_points)
    }

    /// Includes stored point coordinates in the eventual query results.
    #[inline]
    pub fn with_points(self) -> BatchQueryBuilder<'a, Qb::Output, A, K>
    where
        Qb: WithPointsQueryBuilder,
    {
        self.map(Qb::with_points)
    }

    /// Includes stored items in the eventual query results.
    #[inline]
    pub fn with_items(self) -> BatchQueryBuilder<'a, Qb::Output, A, K>
    where
        Qb: WithItemsQueryBuilder,
    {
        self.map(Qb::with_items)
    }

    /// Excludes stored items from the eventual query results.
    #[inline]
    pub fn without_items(self) -> BatchQueryBuilder<'a, Qb::Output, A, K>
    where
        Qb: WithoutItemsQueryBuilder,
    {
        self.map(Qb::without_items)
    }

    /// Includes distances in the eventual query results.
    #[inline]
    pub fn with_distances(self) -> BatchQueryBuilder<'a, Qb::Output, A, K>
    where
        Qb: WithDistancesQueryBuilder,
    {
        self.map(Qb::with_distances)
    }

    /// Excludes distances from the eventual query results.
    #[inline]
    pub fn without_distances(self) -> BatchQueryBuilder<'a, Qb::Output, A, K>
    where
        Qb: WithoutDistancesQueryBuilder,
    {
        self.map(Qb::without_distances)
    }

    /// Interprets every query point in the batch using periodic boundary
    /// conditions.
    #[inline]
    pub fn periodic_boundary_condition(
        self,
        box_size: &'a [A; K],
    ) -> BatchQueryBuilder<'a, Qb::Output, A, K>
    where
        Qb: PeriodicBoundaryConditionQueryBuilder<'a, A, K>,
    {
        self.map(|prototype| prototype.periodic_boundary_condition(box_size))
    }

    /// Selects an exact nearest-neighbour query.
    #[inline]
    pub fn nearest_one<Dq>(self) -> BatchQueryBuilder<'a, Qb::Output, A, K>
    where
        A: Copy,
        Dq: DistanceMetric<A>,
        Qb: NearestOneQueryBuilder<A, K, Dq>,
    {
        self.map(Qb::nearest_one)
    }

    /// Selects a k-nearest-neighbours query.
    #[inline]
    pub fn nearest_n<Dq>(self, max_qty: NonZeroUsize) -> BatchQueryBuilder<'a, Qb::Output, A, K>
    where
        A: Copy,
        Dq: DistanceMetric<A>,
        Qb: NearestNQueryBuilder<A, K, Dq>,
    {
        self.map(|prototype| prototype.nearest_n(max_qty))
    }

    /// Selects a radius query, or adds a radius bound to a k-nearest-neighbours
    /// query.
    ///
    /// The same radius applies to every query point in the batch.
    #[inline]
    pub fn within<Dq>(self, radius: Dq::Output) -> BatchQueryBuilder<'a, Qb::Output, A, K>
    where
        A: Copy,
        Dq: DistanceMetric<A>,
        Qb: WithinQueryBuilder<A, K, Dq>,
    {
        self.map(|prototype| prototype.within(radius))
    }

    /// Selects a radius query that keeps the best `max_qty` entries by item
    /// ordering.
    ///
    /// The single-point query returns a `BinaryHeap`, whose iteration order is
    /// unspecified. [`execute`](Self::execute) instead yields each query
    /// point's group as a slice sorted by item ordering, so that batch results
    /// are reproducible.
    #[inline]
    pub fn best_n_within<Dq>(
        self,
        radius: Dq::Output,
        max_qty: NonZeroUsize,
    ) -> BatchQueryBuilder<'a, Qb::Output, A, K>
    where
        A: Copy,
        Dq: DistanceMetric<A>,
        Qb: BestNWithinQueryBuilder<A, K, Dq>,
    {
        self.map(|prototype| prototype.best_n_within(radius, max_qty))
    }

    /// Switches exact nearest-neighbour search to approximate descent-only mode.
    #[inline]
    pub fn approx(self) -> BatchQueryBuilder<'a, Qb::Output, A, K>
    where
        Qb: ApproxQueryBuilder,
    {
        self.map(Qb::approx)
    }

    /// Returns results in traversal order instead of sorted-by-distance order.
    #[inline]
    pub fn unsorted(self) -> BatchQueryBuilder<'a, Qb::Output, A, K>
    where
        Qb: UnsortedQueryBuilder,
    {
        self.map(Qb::unsorted)
    }

    /// Uses strict `< radius` semantics instead of inclusive `<= radius`.
    #[inline]
    pub fn exclusive_boundaries(self) -> BatchQueryBuilder<'a, Qb::Output, A, K>
    where
        Qb: ExclusiveBoundariesQueryBuilder,
    {
        self.map(Qb::exclusive_boundaries)
    }

    /// Reserves space for the expected number of results *per query point*.
    #[inline]
    pub fn with_result_capacity(self, result_capacity: usize) -> Self
    where
        Qb: WithResultCapacityQueryBuilder,
    {
        self.map(|prototype| prototype.with_result_capacity(result_capacity))
    }
}

/// Terminal methods.
impl<'a, Qb, A, const K: usize> BatchQueryBuilder<'a, Qb, A, K>
where
    A: Sync,
    Qb: ExecuteQueryBuilder + BatchQuerySink<'a, A, K> + Send + Sync,
    Qb::Output: BatchCollect + Send,
{
    /// Executes every query point in the batch and collects the results.
    ///
    /// Results are addressed by the index of their query point in the slice
    /// passed to
    /// [`query_batch`](crate::kd_tree::KdTree::query_batch), regardless of the
    /// order in which they were computed or which thread computed them.
    #[inline]
    pub fn execute(self) -> <Qb::Output as BatchCollect>::Batch {
        let Some(prototype) = self.prototype else {
            return <Qb::Output as BatchCollect>::collect_batch(Vec::new());
        };

        let run = |query: &'a [A; K]| {
            let mut query_builder = prototype;
            query_builder.set_query_point(query);
            query_builder.execute()
        };

        #[cfg(feature = "multi-threaded")]
        if self.executor.is_parallel(self.queries.len()) {
            // Bounding the split length rather than pre-splitting keeps the
            // iterator indexed, so rayon writes results straight into the
            // output instead of collecting through per-task lists.
            let collect = || {
                self.queries
                    .par_iter()
                    .with_min_len(self.executor.static_chunk_len(self.queries.len()))
                    .map(run)
                    .collect::<Vec<_>>()
            };
            let results = match self.executor.pool() {
                Some(pool) => pool.install(collect),
                None => collect(),
            };

            return <Qb::Output as BatchCollect>::collect_batch(results);
        }

        <Qb::Output as BatchCollect>::collect_batch(self.queries.iter().map(run).collect())
    }

    /// Executes every query point in the batch, passing each result to
    /// `visitor` as it is produced.
    ///
    /// `visitor` receives the index of the query point in the slice passed to
    /// [`query_batch`](crate::kd_tree::KdTree::query_batch), along with that
    /// query point's result.
    ///
    /// # Concurrency
    ///
    /// `visitor` may be called from any thread, from several threads at once,
    /// and in any order — which is why it is `Fn + Send + Sync` rather than
    /// `FnMut`. Accumulate through shared state that tolerates this, such as an
    /// atomic, a `Mutex`, or per-index writes into a preallocated buffer.
    ///
    /// # Example
    ///
    /// ```
    /// use kiddo::dist::SquaredEuclidean;
    /// use kiddo::leaf_strategy::FlatVec;
    /// use kiddo::{Eytzinger, KdTree};
    /// use std::sync::atomic::{AtomicUsize, Ordering};
    ///
    /// type Tree = KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, 2, 4>, 2, 4>;
    ///
    /// let entries = [(10_u32, [0.0, 0.0]), (11_u32, [1.0, 1.0])];
    /// let tree = Tree::new_from_entries(&entries).unwrap();
    /// let queries = [[0.1, 0.1], [0.9, 0.9]];
    ///
    /// let matches = AtomicUsize::new(0);
    /// tree.query_batch(&queries)
    ///     .nearest_one::<SquaredEuclidean<f64>>()
    ///     .for_each(|_index, result| {
    ///         if result.item == 10 {
    ///             matches.fetch_add(1, Ordering::Relaxed);
    ///         }
    ///     });
    ///
    /// assert_eq!(matches.load(Ordering::Relaxed), 1);
    /// ```
    #[inline]
    pub fn for_each<F>(self, visitor: F)
    where
        F: Fn(usize, Qb::Output) + Send + Sync,
    {
        let Some(prototype) = self.prototype else {
            return;
        };

        let run = |(index, query): (usize, &'a [A; K])| {
            let mut query_builder = prototype;
            query_builder.set_query_point(query);
            visitor(index, query_builder.execute());
        };

        #[cfg(feature = "multi-threaded")]
        if self.executor.is_parallel(self.queries.len()) {
            let visit = || {
                self.queries
                    .par_iter()
                    .enumerate()
                    .with_min_len(self.executor.static_chunk_len(self.queries.len()))
                    .for_each(run)
            };
            match self.executor.pool() {
                Some(pool) => pool.install(visit),
                None => visit(),
            }

            return;
        }

        self.queries.iter().enumerate().for_each(run);
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    use std::num::NonZeroUsize;
    use std::sync::Mutex;

    use crate::batch::Executor;
    use crate::dist::{Manhattan, SquaredEuclidean};
    use crate::kd_tree::KdTree;
    use crate::leaf_strategy::{FlatVec, VecOfArenas, VecOfArrays};
    use crate::Eytzinger;

    const RNG_SEED: u64 = 42;
    const K: usize = 3;
    const BUCKET: usize = 32;

    type Tree = KdTree<f64, u32, Eytzinger, VecOfArenas<f64, u32, K, BUCKET>, K, BUCKET>;
    type SmallTree = KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, 2, 4>, 2, 4>;

    pub(super) fn tree_and_queries(points: usize, queries: usize) -> (Tree, Vec<[f64; K]>) {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);
        let content: Vec<[f64; K]> = (0..points).map(|_| rng.random()).collect();
        let queries: Vec<[f64; K]> = (0..queries).map(|_| rng.random()).collect();

        (Tree::new_from_slice(&content).unwrap(), queries)
    }

    /// Every executor available in this build. `Executor::parallel` only
    /// exists with the `multi-threaded` feature.
    fn executors() -> Vec<Executor> {
        #[cfg(feature = "multi-threaded")]
        {
            // Every scheduling path must produce identical, correctly indexed
            // results, including the ones that change how the batch is split.
            vec![
                Executor::serial(),
                Executor::parallel(),
                Executor::new(),
                Executor::parallel().with_default_static_chunking(),
                // A multiplier of one is the coarsest legal chunking, and a
                // deliberately odd one exercises a ragged final chunk.
                Executor::parallel()
                    .with_static_chunk_thread_multiplier(NonZeroUsize::new(1).unwrap()),
                Executor::parallel()
                    .with_static_chunk_thread_multiplier(NonZeroUsize::new(7).unwrap()),
                // Thresholds either side of the test batch sizes, so both the
                // fallback and the parallel path are taken.
                Executor::parallel().with_serial_fallback_threshold(usize::MAX),
                Executor::parallel().with_serial_fallback_threshold(0),
                Executor::parallel_in_pool(std::sync::Arc::new(
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(3)
                        .build()
                        .unwrap(),
                ))
                .with_default_static_chunking(),
            ]
        }

        #[cfg(not(feature = "multi-threaded"))]
        {
            vec![Executor::serial(), Executor::new()]
        }
    }

    #[test]
    fn nearest_one_batch_matches_single_point_queries() {
        let (tree, queries) = tree_and_queries(2048, 257);

        for executor in executors() {
            let batch = tree
                .query_batch(&queries)
                .nearest_one::<SquaredEuclidean<f64>>()
                .with_executor(&executor)
                .execute();

            assert_eq!(batch.len(), queries.len());

            for (index, query) in queries.iter().enumerate() {
                let expected = tree
                    .query(query)
                    .nearest_one::<SquaredEuclidean<f64>>()
                    .execute();

                assert_eq!(batch[index].item, expected.item, "index {index}");
                assert_eq!(batch[index].distance, expected.distance, "index {index}");
            }
        }
    }

    #[test]
    fn nearest_n_batch_matches_single_point_queries() {
        let (tree, queries) = tree_and_queries(2048, 129);
        let max_qty = NonZeroUsize::new(7).unwrap();

        for executor in executors() {
            let batch = tree
                .query_batch(&queries)
                .nearest_n::<SquaredEuclidean<f64>>(max_qty)
                .with_executor(&executor)
                .execute();

            assert_eq!(batch.len(), queries.len());
            assert_eq!(batch.total_len(), queries.len() * max_qty.get());

            for (index, query) in queries.iter().enumerate() {
                let expected = tree
                    .query(query)
                    .nearest_n::<SquaredEuclidean<f64>>(max_qty)
                    .execute();

                let actual = batch.get(index).unwrap();
                assert_eq!(actual.len(), expected.len(), "index {index}");

                for (actual, expected) in actual.iter().zip(expected.iter()) {
                    assert_eq!(actual.item, expected.item, "index {index}");
                    assert_eq!(actual.distance, expected.distance, "index {index}");
                }
            }
        }
    }

    #[test]
    fn nearest_n_within_batch_matches_single_point_queries() {
        let (tree, queries) = tree_and_queries(2048, 65);
        let max_qty = NonZeroUsize::new(5).unwrap();
        let radius = 0.05;

        let batch = tree
            .query_batch(&queries)
            .nearest_n::<SquaredEuclidean<f64>>(max_qty)
            .within::<SquaredEuclidean<f64>>(radius)
            .execute();

        for (index, query) in queries.iter().enumerate() {
            let expected = tree
                .query(query)
                .nearest_n::<SquaredEuclidean<f64>>(max_qty)
                .within::<SquaredEuclidean<f64>>(radius)
                .execute();

            let actual = &batch[index];
            assert_eq!(actual.len(), expected.len(), "index {index}");

            for (actual, expected) in actual.iter().zip(expected.iter()) {
                assert_eq!(actual.item, expected.item, "index {index}");
                assert_eq!(actual.distance, expected.distance, "index {index}");
            }
        }
    }

    #[test]
    fn batch_queries_work_against_a_mutable_tree() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);
        let mut tree: KdTree<f64, u32, Eytzinger, VecOfArrays<f64, u32, K, BUCKET>, K, BUCKET> =
            KdTree::default();

        for item in 0..512u32 {
            tree.add(&rng.random::<[f64; K]>(), item).unwrap();
        }

        let queries: Vec<[f64; K]> = (0..64).map(|_| rng.random()).collect();

        let batch = tree
            .query_batch(&queries)
            .nearest_one::<SquaredEuclidean<f64>>()
            .execute();

        for (index, query) in queries.iter().enumerate() {
            let expected = tree
                .query(query)
                .nearest_one::<SquaredEuclidean<f64>>()
                .execute();

            assert_eq!(batch[index].item, expected.item, "index {index}");
        }
    }

    #[test]
    fn within_batch_matches_single_point_queries() {
        let (tree, queries) = tree_and_queries(2048, 65);
        let radius = 0.05;

        let batch = tree
            .query_batch(&queries)
            .within::<SquaredEuclidean<f64>>(radius)
            .execute();

        assert_eq!(batch.len(), queries.len());

        let mut flat_len = 0;
        for (index, query) in queries.iter().enumerate() {
            let expected = tree
                .query(query)
                .within::<SquaredEuclidean<f64>>(radius)
                .execute();

            let actual = &batch[index];
            assert_eq!(actual.len(), expected.len(), "index {index}");
            flat_len += expected.len();

            for (actual, expected) in actual.iter().zip(expected.iter()) {
                assert_eq!(actual.item, expected.item, "index {index}");
            }
        }

        assert_eq!(batch.total_len(), flat_len);
        assert_eq!(batch.into_flat_vec().len(), flat_len);
    }

    #[test]
    fn best_n_within_batch_matches_single_point_queries() {
        let (tree, queries) = tree_and_queries(2048, 33);
        let radius = 0.05;
        let max_qty = NonZeroUsize::new(5).unwrap();

        let batch = tree
            .query_batch(&queries)
            .best_n_within::<SquaredEuclidean<f64>>(radius, max_qty)
            .execute();

        for (index, query) in queries.iter().enumerate() {
            // Batch groups are sorted by item ordering; the single-point query
            // hands back a heap whose iteration order is unspecified.
            let expected = tree
                .query(query)
                .best_n_within::<SquaredEuclidean<f64>>(radius, max_qty)
                .execute()
                .into_sorted_vec();

            let actual = &batch[index];
            assert_eq!(actual.len(), expected.len(), "index {index}");

            for (actual, expected) in actual.iter().zip(expected.iter()) {
                assert_eq!(actual.item, expected.item, "index {index}");
                assert_eq!(actual.distance, expected.distance, "index {index}");
            }
        }
    }

    #[test]
    fn projections_and_modifiers_carry_through_to_batch() {
        let (tree, queries) = tree_and_queries(1024, 17);
        let max_qty = NonZeroUsize::new(3).unwrap();

        let batch = tree
            .query_batch(&queries)
            .nearest_n::<Manhattan<f64>>(max_qty)
            .with_points()
            .without_items()
            .execute();

        for (index, query) in queries.iter().enumerate() {
            let expected = tree
                .query(query)
                .nearest_n::<Manhattan<f64>>(max_qty)
                .with_points()
                .without_items()
                .execute();

            for (actual, expected) in batch[index].iter().zip(expected.iter()) {
                assert_eq!(actual.point, expected.point, "index {index}");
                assert_eq!(actual.distance, expected.distance, "index {index}");
                assert_eq!(actual.item, (), "index {index}");
            }
        }
    }

    #[test]
    fn unsorted_and_exclusive_boundaries_carry_through_to_batch() {
        let (tree, queries) = tree_and_queries(1024, 17);
        let radius = 0.05;

        let batch = tree
            .query_batch(&queries)
            .within::<SquaredEuclidean<f64>>(radius)
            .unsorted()
            .exclusive_boundaries()
            .with_result_capacity(8)
            .execute();

        for (index, query) in queries.iter().enumerate() {
            let expected = tree
                .query(query)
                .within::<SquaredEuclidean<f64>>(radius)
                .unsorted()
                .exclusive_boundaries()
                .with_result_capacity(8)
                .execute();

            assert_eq!(batch[index].len(), expected.len(), "index {index}");
        }
    }

    #[test]
    fn approx_nearest_one_carries_through_to_batch() {
        let (tree, queries) = tree_and_queries(1024, 17);

        let batch = tree
            .query_batch(&queries)
            .nearest_one::<SquaredEuclidean<f64>>()
            .approx()
            .execute();

        for (index, query) in queries.iter().enumerate() {
            let expected = tree
                .query(query)
                .nearest_one::<SquaredEuclidean<f64>>()
                .approx()
                .execute();

            assert_eq!(batch[index].item, expected.item, "index {index}");
        }
    }

    #[test]
    fn periodic_boundary_conditions_carry_through_to_batch() {
        let (tree, queries) = tree_and_queries(1024, 17);
        let box_size = [1.0f64; K];

        let batch = tree
            .query_batch(&queries)
            .periodic_boundary_condition(&box_size)
            .nearest_one::<SquaredEuclidean<f64>>()
            .execute();

        for (index, query) in queries.iter().enumerate() {
            let expected = tree
                .query(query)
                .periodic_boundary_condition(&box_size)
                .nearest_one::<SquaredEuclidean<f64>>()
                .execute();

            assert_eq!(batch[index].item, expected.item, "index {index}");
            assert_eq!(batch[index].distance, expected.distance, "index {index}");
        }
    }

    #[test]
    fn for_each_visits_every_query_exactly_once_with_its_own_index() {
        let (tree, queries) = tree_and_queries(2048, 513);

        for executor in executors() {
            let seen: Mutex<Vec<Option<u32>>> = Mutex::new(vec![None; queries.len()]);

            tree.query_batch(&queries)
                .nearest_one::<SquaredEuclidean<f64>>()
                .with_executor(&executor)
                .for_each(|index, result| {
                    let mut seen = seen.lock().unwrap();
                    assert!(seen[index].is_none(), "index {index} visited twice");
                    seen[index] = Some(result.item);
                });

            let seen = seen.into_inner().unwrap();
            for (index, query) in queries.iter().enumerate() {
                let expected = tree
                    .query(query)
                    .nearest_one::<SquaredEuclidean<f64>>()
                    .execute();
                assert_eq!(seen[index], Some(expected.item), "index {index}");
            }
        }
    }

    #[test]
    fn empty_batch_yields_empty_results() {
        let (tree, _) = tree_and_queries(256, 0);
        let queries: [[f64; K]; 0] = [];

        let nearest_one = tree
            .query_batch(&queries)
            .nearest_one::<SquaredEuclidean<f64>>()
            .execute();
        assert!(nearest_one.is_empty());
        assert_eq!(nearest_one.len(), 0);
        assert!(nearest_one.into_vec().is_empty());

        let nearest_n = tree
            .query_batch(&queries)
            .nearest_n::<SquaredEuclidean<f64>>(NonZeroUsize::new(4).unwrap())
            .execute();
        assert!(nearest_n.is_empty());
        assert_eq!(nearest_n.total_len(), 0);
        assert_eq!(nearest_n.iter().count(), 0);

        // A visitor over an empty batch must never fire.
        tree.query_batch(&queries)
            .nearest_one::<SquaredEuclidean<f64>>()
            .for_each(|_, _| panic!("visitor called for an empty batch"));
    }

    #[test]
    fn single_query_batch_is_supported() {
        let entries = [
            (10_u32, [0.0, 0.0]),
            (11_u32, [1.0, 1.0]),
            (12_u32, [0.2, 0.1]),
        ];
        let tree = SmallTree::new_from_entries(&entries).unwrap();
        let queries = [[0.9, 0.9]];

        let results = tree
            .query_batch(&queries)
            .nearest_one::<SquaredEuclidean<f64>>()
            .execute();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].item, 11);
    }

    #[test]
    fn builder_reports_batch_size_before_execution() {
        let (tree, queries) = tree_and_queries(256, 9);

        let builder = tree.query_batch(&queries);
        assert_eq!(builder.len(), 9);
        assert!(!builder.is_empty());

        let empty: [[f64; K]; 0] = [];
        assert!(tree.query_batch(&empty).is_empty());
    }

    #[cfg(feature = "multi-threaded")]
    #[test]
    fn scheduling_hints_do_not_change_results() {
        let (tree, queries) = tree_and_queries(1024, 200);

        let baseline = tree
            .query_batch(&queries)
            .nearest_one::<SquaredEuclidean<f64>>()
            .with_executor(&Executor::serial())
            .execute();

        for executor in executors() {
            let hinted = tree
                .query_batch(&queries)
                .nearest_one::<SquaredEuclidean<f64>>()
                .with_executor(&executor)
                .execute();

            assert_eq!(
                hinted.as_slice(),
                baseline.as_slice(),
                "executor {executor:?}"
            );
        }
    }

    /// A ragged final chunk is the case most likely to misindex results, so
    /// walk batch sizes that do not divide evenly by the chunk count.
    #[cfg(feature = "multi-threaded")]
    #[test]
    fn static_chunking_indexes_ragged_batches_correctly() {
        for query_count in [1usize, 2, 3, 5, 17, 63, 64, 65, 257] {
            let (tree, queries) = tree_and_queries(512, query_count);

            let baseline = tree
                .query_batch(&queries)
                .nearest_one::<SquaredEuclidean<f64>>()
                .with_executor(&Executor::serial())
                .execute();

            for multiplier in [1usize, 3, 4, 16] {
                let executor = Executor::parallel()
                    .with_static_chunk_thread_multiplier(NonZeroUsize::new(multiplier).unwrap());

                let chunked = tree
                    .query_batch(&queries)
                    .nearest_one::<SquaredEuclidean<f64>>()
                    .with_executor(&executor)
                    .execute();

                assert_eq!(
                    chunked.as_slice(),
                    baseline.as_slice(),
                    "{query_count} queries, multiplier {multiplier}"
                );
            }
        }
    }

    /// `for_each` reports indices itself rather than returning a positional
    /// collection, so it needs its own ragged-batch check.
    #[cfg(feature = "multi-threaded")]
    #[test]
    fn static_chunking_for_each_reports_correct_indices() {
        use std::sync::Mutex;

        for query_count in [1usize, 5, 17, 63, 65, 257] {
            let (tree, queries) = tree_and_queries(512, query_count);

            let expected: Vec<_> = queries
                .iter()
                .map(|query| {
                    tree.query(query)
                        .nearest_one::<SquaredEuclidean<f64>>()
                        .execute()
                })
                .collect();

            let seen = Mutex::new(vec![None; query_count]);
            tree.query_batch(&queries)
                .nearest_one::<SquaredEuclidean<f64>>()
                .with_executor(&Executor::parallel().with_default_static_chunking())
                .for_each(|index, result| {
                    seen.lock().unwrap()[index] = Some(result);
                });

            let seen = seen.into_inner().unwrap();
            for (index, (actual, expected)) in seen.iter().zip(expected.iter()).enumerate() {
                assert_eq!(
                    actual.as_ref(),
                    Some(expected),
                    "{query_count} queries, index {index}"
                );
            }
        }
    }

    #[cfg(feature = "multi-threaded")]
    #[test]
    fn serial_fallback_threshold_selects_the_calling_thread() {
        let (tree, queries) = tree_and_queries(1024, 64);

        // Above the threshold the batch goes parallel, below it stays here;
        // either way the results must be identical.
        for threshold in [0usize, 32, 64, 65, usize::MAX] {
            let executor = Executor::parallel().with_serial_fallback_threshold(threshold);
            assert_eq!(
                executor.is_parallel(queries.len()),
                queries.len() >= threshold
            );

            let results = tree
                .query_batch(&queries)
                .nearest_one::<SquaredEuclidean<f64>>()
                .with_executor(&executor)
                .execute();

            assert_eq!(results.len(), queries.len());
        }
    }

    #[cfg(feature = "multi-threaded")]
    #[test]
    fn caller_supplied_pool_bounds_the_thread_count() {
        use std::collections::HashSet;
        use std::sync::Mutex;

        let (tree, queries) = tree_and_queries(4096, 512);
        let pool = std::sync::Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .unwrap(),
        );

        let threads = Mutex::new(HashSet::new());
        tree.query_batch(&queries)
            .nearest_one::<SquaredEuclidean<f64>>()
            .with_executor(&Executor::parallel_in_pool(std::sync::Arc::clone(&pool)))
            .for_each(|_index, _result| {
                threads
                    .lock()
                    .unwrap()
                    .insert(rayon::current_thread_index());
            });

        // Assert on the indices rather than on how many are distinct: the
        // calling thread is not a pool worker, so if it ran a chunk itself it
        // would contribute a `None` and inflate a count-based check.
        let threads = threads.into_inner().unwrap();
        assert!(
            threads
                .iter()
                .all(|index| index.is_none_or(|index| index < 2)),
            "work escaped the caller-supplied 2-thread pool: {threads:?}"
        );
    }

    #[test]
    fn results_containers_expose_owned_and_borrowed_views() {
        let (tree, queries) = tree_and_queries(512, 5);

        let results = tree
            .query_batch(&queries)
            .nearest_one::<SquaredEuclidean<f64>>()
            .execute();
        let borrowed: Vec<u32> = results.iter().map(|result| result.item).collect();
        let by_ref: Vec<u32> = (&results).into_iter().map(|result| result.item).collect();
        let owned: Vec<u32> = results.into_iter().map(|result| result.item).collect();
        assert_eq!(borrowed, owned);
        assert_eq!(by_ref, owned);

        let groups = tree
            .query_batch(&queries)
            .nearest_n::<SquaredEuclidean<f64>>(NonZeroUsize::new(2).unwrap())
            .execute();
        let group_lens: Vec<usize> = groups.iter().map(<[_]>::len).collect();
        let by_ref_lens: Vec<usize> = (&groups).into_iter().map(<[_]>::len).collect();
        assert_eq!(group_lens, vec![2; queries.len()]);
        assert_eq!(by_ref_lens, group_lens);
        assert_eq!(groups.into_nested_vec().len(), queries.len());
    }
}

#[cfg(all(test, feature = "multi-threaded"))]
mod parallel_tests {
    use std::collections::HashSet;
    use std::sync::Mutex;
    use std::thread::ThreadId;

    use crate::batch::Executor;
    use crate::dist::SquaredEuclidean;

    use super::tests::tree_and_queries;

    /// Not an API guarantee — a smoke test that the parallel executor really
    /// does reach Rayon rather than silently falling back to the serial path.
    #[test]
    fn parallel_executor_uses_the_rayon_pool() {
        if rayon::current_num_threads() < 2 {
            return;
        }

        let (tree, queries) = tree_and_queries(4096, 8192);
        let threads: Mutex<HashSet<ThreadId>> = Mutex::new(HashSet::new());

        tree.query_batch(&queries)
            .nearest_one::<SquaredEuclidean<f64>>()
            .with_executor(&Executor::parallel())
            .for_each(|_, _| {
                threads.lock().unwrap().insert(std::thread::current().id());
            });

        assert!(
            threads.into_inner().unwrap().len() > 1,
            "expected work to reach more than one thread"
        );
    }
}

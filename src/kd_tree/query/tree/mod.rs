//! Exact radius joins between two owned immutable trees.
#![allow(private_bounds)]

#[cfg(all(
    feature = "simd",
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
mod avx2;
#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
mod avx512;
mod cursor;
mod leaf;
mod metric;
mod nearest;
#[cfg(feature = "multi-threaded")]
mod parallel;
#[cfg(feature = "exact_query_stats")]
pub mod stats;
mod traversal;

use crate::batch::Executor;
use crate::traits::leaf_strategy::Immutable;
use crate::{Content, KdTree, LeafStrategy, StemStrategy};
use cursor::{JoinStorage, JoinTree};
use metric::{JoinFloat, JoinMetric};
#[cfg(feature = "multi-threaded")]
use rayon::prelude::*;
use std::marker::PhantomData;
use traversal::{Counter, Sink};

/// The exact nearest entry in the right tree for one entry in the left tree.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct TreeNearestQueryResult<L, R, D> {
    /// Item from the tree on which `query_tree` was called.
    pub left_item: L,
    /// Nearest item from the argument to `query_tree`.
    pub right_item: R,
    /// Distance in the selected metric's output units.
    pub distance: D,
}

/// One matching entry pair. Equality compares both items and the distance.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct PairQueryResult<L, R, D> {
    /// Item from the tree on which `query_tree` was called.
    pub left_item: L,
    /// Item from the argument to `query_tree`.
    pub right_item: R,
    /// Metric distance, or `()` when distances are omitted.
    pub distance: D,
}

#[doc(hidden)]
pub struct NoMetric;
#[doc(hidden)]
pub struct WithDistances;
#[doc(hidden)]
pub struct WithoutDistances;

#[doc(hidden)]
pub trait DistanceProjection<O> {
    type Output: Send;
    const ENABLED: bool;
    fn project(distance: O) -> Self::Output;
}
impl<O: Send> DistanceProjection<O> for WithDistances {
    type Output = O;
    const ENABLED: bool = true;
    fn project(distance: O) -> O {
        distance
    }
}
impl<O> DistanceProjection<O> for WithoutDistances {
    type Output = ();
    const ENABLED: bool = false;
    fn project(_: O) {}
}

/// Builder for an exact, unsorted radius join between distinct immutable trees.
///
/// Stored coordinates must be finite. Item values and coordinates may repeat;
/// every matching entry pair is preserved. Ordering is unspecified. Distances
/// use the metric's output units, just like single-point queries.
///
/// ```
/// use kiddo::{ImmutableKdTree, SquaredEuclidean};
/// let left = ImmutableKdTree::new_from_slice(&[[0.0f64, 0.0], [1.0, 0.0]]).unwrap();
/// let right = ImmutableKdTree::new_from_slice(&[[0.5f64, 0.0]]).unwrap();
/// let query = left.query_tree(&right).within::<SquaredEuclidean<f64>>(0.25);
/// assert_eq!(query.count(), 2usize);
/// let pairs = left.query_tree(&right).within::<SquaredEuclidean<f64>>(0.25)
///     .without_distances().execute();
/// assert_eq!(pairs.len(), 2);
/// ```
///
/// Execution requires selecting a metric:
/// ```compile_fail
/// use kiddo::ImmutableKdTree;
/// let a = ImmutableKdTree::new_from_slice(&[[0.0f64, 0.0]]).unwrap();
/// let b = ImmutableKdTree::new_from_slice(&[[0.0f64, 0.0]]).unwrap();
/// a.query_tree(&b).execute();
/// ```
///
/// Mutable trees are not join operands:
/// ```compile_fail
/// use kiddo::{ImmutableKdTree, MutableKdTree};
/// let a = ImmutableKdTree::new_from_slice(&[[0.0f64, 0.0]]).unwrap();
/// let b = MutableKdTree::new_from_slice(&[[0.0f64, 0.0]]).unwrap();
/// a.query_tree(&b);
/// ```
///
/// Output precision cannot narrow the stored coordinates:
/// ```compile_fail
/// use kiddo::{ImmutableKdTree, SquaredEuclidean};
/// let a = ImmutableKdTree::new_from_slice(&[[0.0f64, 0.0]]).unwrap();
/// let b = ImmutableKdTree::new_from_slice(&[[1.0f64, 0.0]]).unwrap();
/// a.query_tree(&b).within::<SquaredEuclidean<f32>>(1.0);
/// ```
///
/// Integer coordinate trees are not supported:
/// ```compile_fail
/// use kiddo::ImmutableKdTree;
/// let a = ImmutableKdTree::new_from_slice(&[[0u16, 0]]).unwrap();
/// let b = ImmutableKdTree::new_from_slice(&[[1u16, 0]]).unwrap();
/// a.query_tree(&b);
/// ```
///
/// A custom scalar metric cannot opt into box pruning implicitly:
/// ```compile_fail
/// use kiddo::{ImmutableKdTree, dist::DistanceMetricScalar};
/// struct Custom;
/// impl DistanceMetricScalar<f64> for Custom {
///     type Output = f64;
///     fn widen_coord(x: f64) -> f64 { x }
///     fn dist1(a: f64, b: f64) -> f64 { (a-b).abs() }
/// }
/// let a = ImmutableKdTree::new_from_slice(&[[0.0f64, 0.0]]).unwrap();
/// let b = ImmutableKdTree::new_from_slice(&[[1.0f64, 0.0]]).unwrap();
/// a.query_tree(&b).within::<Custom>(1.0);
/// ```
pub struct TreeQueryBuilder<
    'a,
    L,
    R,
    const K: usize,
    D = NoMetric,
    P = WithDistances,
    const EX: bool = false,
    O = (),
> {
    left: &'a L,
    right: &'a R,
    radius: O,
    executor: Executor,
    _state: PhantomData<(D, P)>,
}

/// Builder for an exact dual-tree nearest-one query.
///
/// This evaluates every entry in the left tree against the right tree, but
/// shares node-pair bounds between neighbouring left entries. Results are in
/// unspecified left-tree storage order.
pub struct TreeNearestQueryBuilder<'a, L, R, const K: usize, D = NoMetric> {
    left: &'a L,
    right: &'a R,
    _state: PhantomData<D>,
}

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: JoinFloat,
    T: Content,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B, Mutability = Immutable> + JoinStorage + Sync,
{
    /// Starts a radius join with another owned immutable tree.
    ///
    /// Both trees must have finite coordinates and the same coordinate type and
    /// dimensionality. Item types, leaf stores, layouts and bucket sizes may differ.
    ///
    /// # Panics
    /// Panics if both references identify the same tree object. Independently
    /// allocated trees with identical contents are allowed.
    pub fn query_tree<'a, R: JoinTree<K, A = A>>(
        &'a self,
        other: &'a R,
    ) -> TreeQueryBuilder<'a, Self, R, K> {
        assert!(
            !std::ptr::eq(
                (self as *const Self).cast::<()>(),
                (other as *const R).cast::<()>()
            ),
            "tree joins require two distinct tree objects"
        );
        assert!(K > 0, "tree joins require at least one dimension");
        TreeQueryBuilder {
            left: self,
            right: other,
            radius: (),
            executor: Executor::new(),
            _state: PhantomData,
        }
    }
}

impl<'a, L: JoinTree<K>, R: JoinTree<K, A = L::A>, const K: usize, P, const EX: bool>
    TreeQueryBuilder<'a, L, R, K, NoMetric, P, EX>
{
    /// Selects a metric and threshold in metric-output units.
    ///
    /// # Panics
    /// Negative and NaN thresholds are rejected. Zero and positive infinity are valid.
    pub fn within<D: JoinMetric<L::A>>(
        self,
        radius: D::Output,
    ) -> TreeQueryBuilder<'a, L, R, K, D, P, EX, D::Output> {
        let () = D::VALID;
        assert!(
            !radius.is_nan() && radius >= D::Output::ZERO,
            "tree join threshold must be nonnegative and not NaN"
        );
        TreeQueryBuilder {
            left: self.left,
            right: self.right,
            radius,
            executor: self.executor,
            _state: PhantomData,
        }
    }
}

impl<'a, L: JoinTree<K>, R: JoinTree<K, A = L::A>, const K: usize, P, const EX: bool>
    TreeQueryBuilder<'a, L, R, K, NoMetric, P, EX>
{
    /// Selects an exact nearest target entry for each left-tree entry.
    pub fn nearest_one<D: JoinMetric<L::A>>(self) -> TreeNearestQueryBuilder<'a, L, R, K, D> {
        let () = D::VALID;
        TreeNearestQueryBuilder {
            left: self.left,
            right: self.right,
            _state: PhantomData,
        }
    }
}

impl<L, R, const K: usize, D> TreeNearestQueryBuilder<'_, L, R, K, D>
where
    L: JoinTree<K>,
    R: JoinTree<K, A = L::A>,
    D: JoinMetric<L::A>,
{
    /// Executes the exact dual-tree nearest-one query.
    ///
    /// An empty right tree has no nearest entries, so it returns an empty
    /// vector. Ties are resolved in unspecified tree-traversal order.
    pub fn execute(self) -> Vec<TreeNearestQueryResult<L::Item, R::Item, D::Output>> {
        nearest::execute::<L, R, D, K>(self.left, self.right)
    }
}

impl<'a, L, R, const K: usize, D, P, const EX: bool, O> TreeQueryBuilder<'a, L, R, K, D, P, EX, O> {
    /// Omits returned distances, enabling bulk acceptance of whole subtree pairs.
    pub fn without_distances(self) -> TreeQueryBuilder<'a, L, R, K, D, WithoutDistances, EX, O> {
        TreeQueryBuilder {
            left: self.left,
            right: self.right,
            radius: self.radius,
            executor: self.executor,
            _state: PhantomData,
        }
    }
    /// Includes metric distances (the default).
    pub fn with_distances(self) -> TreeQueryBuilder<'a, L, R, K, D, WithDistances, EX, O> {
        TreeQueryBuilder {
            left: self.left,
            right: self.right,
            radius: self.radius,
            executor: self.executor,
            _state: PhantomData,
        }
    }
    /// Selects strict `distance < threshold` membership.
    pub fn exclusive_boundaries(self) -> TreeQueryBuilder<'a, L, R, K, D, P, true, O> {
        TreeQueryBuilder {
            left: self.left,
            right: self.right,
            radius: self.radius,
            executor: self.executor,
            _state: PhantomData,
        }
    }
    /// Selects serial, global-pool, or caller-owned-pool execution.
    /// The default follows [`Executor::new`].
    ///
    /// Serial fallback uses the left tree's entry count. A static chunk
    /// multiplier is interpreted as the desired number of frontier tasks per
    /// pool thread (eight by default); it does not fix task or thread order.
    pub fn with_executor(mut self, executor: &Executor) -> Self {
        self.executor = executor.clone();
        self
    }
}

struct Visitor<'a, F, P>(&'a F, PhantomData<P>);
impl<
        L: Content,
        R: Content,
        O: JoinFloat,
        P: DistanceProjection<O>,
        F: Fn(PairQueryResult<L, R, P::Output>),
    > Sink<L, R, O> for Visitor<'_, F, P>
{
    const DISTANCES: bool = P::ENABLED;
    fn emit(&mut self, left: L, right: R, distance: O) {
        (self.0)(PairQueryResult {
            left_item: left,
            right_item: right,
            distance: P::project(distance),
        });
    }
}
struct Collector<L, R, O, P: DistanceProjection<O>>(Vec<PairQueryResult<L, R, P::Output>>);
impl<L: Content, R: Content, O: JoinFloat, P: DistanceProjection<O>> Sink<L, R, O>
    for Collector<L, R, O, P>
{
    const DISTANCES: bool = P::ENABLED;
    fn emit(&mut self, left: L, right: R, distance: O) {
        self.0.push(PairQueryResult {
            left_item: left,
            right_item: right,
            distance: P::project(distance),
        });
    }
}

impl<L, R, const K: usize, D, P, const EX: bool> TreeQueryBuilder<'_, L, R, K, D, P, EX, D::Output>
where
    L: JoinTree<K>,
    R: JoinTree<K, A = L::A>,
    D: JoinMetric<L::A>,
    P: DistanceProjection<D::Output>,
{
    /// Collects every matching entry pair in unspecified order.
    pub fn execute(self) -> Vec<PairQueryResult<L::Item, R::Item, P::Output>> {
        if self.left.size() == 0 || self.right.size() == 0 {
            return Vec::new();
        }
        #[cfg(feature = "multi-threaded")]
        if self.executor.is_parallel(self.left.size()) {
            let tasks = parallel::frontier::<_, _, D, K, EX>(
                self.left,
                self.right,
                self.radius,
                self.executor.join_task_budget(),
            );
            let run = || {
                let chunks: Vec<_> = tasks
                    .into_par_iter()
                    .map(|task| {
                        let mut sink = Collector::<_, _, _, P>(Vec::new());
                        traversal::walk::<_, _, D, _, K, EX>(
                            self.left,
                            self.right,
                            task,
                            self.radius,
                            &mut sink,
                        );
                        sink.0
                    })
                    .collect();
                let len = chunks
                    .iter()
                    .map(Vec::len)
                    .try_fold(0usize, usize::checked_add)
                    .expect(traversal::COUNT_OVERFLOW);
                let mut result = Vec::with_capacity(len);
                for chunk in chunks {
                    result.extend(chunk);
                }
                result
            };
            return match self.executor.pool() {
                Some(pool) => pool.install(run),
                None => run(),
            };
        }
        let mut sink = Collector::<_, _, _, P>(Vec::new());
        if self.left.size() != 0 && self.right.size() != 0 {
            traversal::walk::<_, _, D, _, K, EX>(
                self.left,
                self.right,
                traversal::root(self.left, self.right),
                self.radius,
                &mut sink,
            );
        }
        sink.0
    }
    /// Visits each matching pair without collecting results.
    ///
    /// The visitor may be called concurrently and in any order. Panics propagate.
    pub fn for_each<F: Fn(PairQueryResult<L::Item, R::Item, P::Output>) + Send + Sync>(
        self,
        visitor: F,
    ) {
        if self.left.size() == 0 || self.right.size() == 0 {
            return;
        }
        #[cfg(feature = "multi-threaded")]
        if self.executor.is_parallel(self.left.size()) {
            let tasks = parallel::frontier::<_, _, D, K, EX>(
                self.left,
                self.right,
                self.radius,
                self.executor.join_task_budget(),
            );
            let run = || {
                tasks.into_par_iter().for_each(|task| {
                    traversal::walk::<_, _, D, _, K, EX>(
                        self.left,
                        self.right,
                        task,
                        self.radius,
                        &mut Visitor::<_, P>(&visitor, PhantomData),
                    );
                })
            };
            match self.executor.pool() {
                Some(pool) => pool.install(run),
                None => run(),
            };
            return;
        }
        if self.left.size() != 0 && self.right.size() != 0 {
            traversal::walk::<_, _, D, _, K, EX>(
                self.left,
                self.right,
                traversal::root(self.left, self.right),
                self.radius,
                &mut Visitor::<_, P>(&visitor, PhantomData),
            );
        }
    }
    /// Counts matching entry pairs without allocating results or loading items.
    ///
    /// # Panics
    /// Panics if the result exceeds `usize::MAX`, in debug and release builds.
    pub fn count(self) -> usize {
        if self.left.size() == 0 || self.right.size() == 0 {
            return 0;
        }
        if !EX && self.radius == D::Output::INFINITY {
            let count = self
                .left
                .size()
                .checked_mul(self.right.size())
                .expect(traversal::COUNT_OVERFLOW);
            #[cfg(feature = "exact_query_stats")]
            {
                stats::record(stats::Event::Node, 1);
                stats::record(stats::Event::Accepted, 1);
                stats::record(stats::Event::Matches, count);
            }
            return count;
        }
        #[cfg(feature = "multi-threaded")]
        if self.executor.is_parallel(self.left.size()) {
            let tasks = parallel::frontier::<_, _, D, K, EX>(
                self.left,
                self.right,
                self.radius,
                self.executor.join_task_budget(),
            );
            let run = || {
                tasks
                    .into_par_iter()
                    .map(|task| {
                        let mut sink = Counter::default();
                        traversal::walk::<_, _, D, _, K, EX>(
                            self.left,
                            self.right,
                            task,
                            self.radius,
                            &mut sink,
                        );
                        sink.0
                    })
                    .reduce(
                        || 0usize,
                        |a, b| a.checked_add(b).expect(traversal::COUNT_OVERFLOW),
                    )
            };
            return match self.executor.pool() {
                Some(pool) => pool.install(run),
                None => run(),
            };
        }
        let mut sink = Counter::default();
        if self.left.size() != 0 && self.right.size() != 0 {
            traversal::walk::<_, _, D, _, K, EX>(
                self.left,
                self.right,
                traversal::root(self.left, self.right),
                self.radius,
                &mut sink,
            );
        }
        sink.0
    }
}

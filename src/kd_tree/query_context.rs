/// Context for spatial queries, providing the query point and pruning distance.
///
/// This trait is implemented by query-specific context structs that hold
/// the query point and track the maximum distance for branch pruning during
/// backtracking search.
pub trait QueryContext<A, O, const K: usize> {
    /// Whether this query instantiation consumes embedded item summaries.
    /// This associated constant lets monomorphisation remove every summary
    /// check from other query kinds and item-leaf modes.
    #[doc(hidden)]
    const USES_EMBEDDED_ITEM_SUMMARY: bool = false;

    /// Returns the query point coordinates.
    fn query(&self) -> &[A; K];

    /// Returns the current maximum distance for pruning.
    ///
    /// This is used during backtracking to prune branches that cannot contain
    /// better results than already found. For nearest neighbor queries, this
    /// returns the distance to the best point found so far.
    fn max_dist(&self) -> O;

    /// Returns true when the query starts without a meaningful pruning bound.
    ///
    /// Arithmetic traversal can then defer bound checks until the first leaf
    /// establishes one. Radius-limited queries should retain the default.
    #[inline]
    fn initial_bound_is_unbounded(&self) -> bool {
        false
    }

    // TOOO: investigate into whether prune_on_equal_max_dist can be removed
    /// Returns true if branches with `rd == max_dist` should be pruned.
    ///
    /// Nearest-one queries can safely prune equality and gain performance.
    /// Radius-based queries generally need to keep equality (boundary points).
    #[inline]
    fn prune_on_equal_max_dist(&self) -> bool {
        false
    }

    /// Current encoded-summary filter for item-based subtree pruning. Summary
    /// queries keep this disabled until their result collection is full.
    #[doc(hidden)]
    #[inline(always)]
    fn embedded_item_summary_filter(&self) -> crate::kd_tree::item_summary::ItemSummaryFilter {
        crate::kd_tree::item_summary::ItemSummaryFilter::disabled()
    }
}

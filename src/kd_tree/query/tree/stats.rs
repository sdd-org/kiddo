//! Invasive, process-wide tree-join counters, enabled by `exact_query_stats`.
//! Reset only when no joins are running. Do not enable these counters in timing
//! builds: recording synchronizes across workers and changes performance.
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Default, Clone, Copy)]
/// Structural counters for exact dual-tree radius-join development.
pub struct TreeJoinStats {
    /// Node pairs considered by serial task traversals.
    pub node_pairs: usize,
    /// Empty or geometrically excluded node pairs.
    pub rejected_pairs: usize,
    /// Subtree pairs emitted without calculating point distances.
    pub accepted_pairs: usize,
    /// Leaf pairs evaluated by the distance kernel.
    pub leaf_pairs: usize,
    /// Point distances evaluated by leaf kernels.
    pub point_distances: usize,
    /// Matching entry pairs counted or emitted.
    pub matches: usize,
    /// Work units surviving parallel frontier construction.
    pub tasks: usize,
    /// Continuation frames placed in heap spill storage.
    pub stack_spills: usize,
}

static COUNTERS: [AtomicUsize; 8] = [const { AtomicUsize::new(0) }; 8];

pub(super) enum Event {
    Node,
    Rejected,
    Accepted,
    Leaf,
    Distances,
    Matches,
    Tasks,
    Spill,
}
#[inline]
pub(super) fn record(event: Event, count: usize) {
    COUNTERS[event as usize].fetch_add(count, Ordering::Relaxed);
}

/// Clears the process-wide counters. Do not race this with query execution.
pub fn reset() {
    for counter in &COUNTERS {
        counter.store(0, Ordering::Relaxed);
    }
}

/// Reads the counters; concurrent queries may still change them.
pub fn snapshot() -> TreeJoinStats {
    let c = std::array::from_fn::<_, 8, _>(|i| COUNTERS[i].load(Ordering::Relaxed));
    TreeJoinStats {
        node_pairs: c[0],
        rejected_pairs: c[1],
        accepted_pairs: c[2],
        leaf_pairs: c[3],
        point_distances: c[4],
        matches: c[5],
        tasks: c[6],
        stack_spills: c[7],
    }
}

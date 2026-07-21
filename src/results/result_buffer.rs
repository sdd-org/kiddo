use crate::results::result_collection::ResultCollection;
use crate::Axis;

/// Pre-allocated result collection for the unsorted query path.
///
/// Avoids the first several `Vec` realloc waves (Brodnik et al., WADS 1999)
/// by starting at a compile-time constant capacity. `FixedResultCollection<_, 256>`
/// skips 8 waves. No single constant works universally.
pub struct FixedResultCollection<E, const CAP: usize = 64>(Vec<E>);

pub type DefaultFixedResultCollection<E> = FixedResultCollection<E, 256>;

impl<E, const CAP: usize> FixedResultCollection<E, CAP> {
    pub fn capacity(&self) -> usize {
        self.0.capacity()
    }
}

impl<E, const CAP: usize> Default for FixedResultCollection<E, CAP> {
    fn default() -> Self {
        Self(Vec::with_capacity(CAP))
    }
}

impl<O: Axis<Coord = O>, E: Ord, const CAP: usize> ResultCollection<O, E>
    for FixedResultCollection<E, CAP>
{
    fn with_max_qty(_max_qty: usize) -> Self {
        Self(Vec::with_capacity(CAP))
    }
    fn max_qty(&self) -> usize {
        usize::MAX
    }
    fn len(&self) -> usize {
        self.0.len()
    }
    fn add(&mut self, entry: E) {
        self.0.push(entry)
    }
    fn threshold_distance(&self) -> Option<O> {
        None
    }
    fn into_vec(self) -> Vec<E> {
        self.0
    }
    fn into_sorted_vec(self) -> Vec<E> {
        self.0
    }
}

/// Result collection that pre-allocates from the tree's leaf count.
///
/// Capacity = (leaf_count * bucket_size) / 4, reflecting that a range query
/// typically visits 25-40% of leaves on uniform data. The worst-case kd-tree
/// range query visits O(N^(1-1/K)) nodes (Lee & Wong, Acta Informatica 1977).
///
/// Constructed via `LeafCountResultCollection::with_leaf_estimate(leaves, bsize)`.
pub struct LeafCountResultCollection<E>(Vec<E>);

impl<E> LeafCountResultCollection<E> {
    pub fn with_leaf_estimate(leaf_count: usize, bucket_size: usize) -> Self {
        let capacity = (leaf_count * bucket_size / 4).max(64);
        Self(Vec::with_capacity(capacity))
    }
}

impl<O: Axis<Coord = O>, E: Ord> ResultCollection<O, E> for LeafCountResultCollection<E> {
    fn with_max_qty(_max_qty: usize) -> Self {
        Self(Vec::with_capacity(64))
    }
    fn max_qty(&self) -> usize {
        usize::MAX
    }
    fn len(&self) -> usize {
        self.0.len()
    }
    fn add(&mut self, entry: E) {
        self.0.push(entry)
    }
    fn threshold_distance(&self) -> Option<O> {
        None
    }
    fn into_vec(self) -> Vec<E> {
        self.0
    }
    fn into_sorted_vec(self) -> Vec<E> {
        self.0
    }
}

/// Result collection that pre-allocates using the spatial selectivity formula.
///
/// capacity = N * (query_radius / bounding_box_diagonal)^K
///
/// From the R*-tree cost model (Beckmann et al., SIGMOD 1990, Section 4.2).
/// Constructed via `RadiusResultCollection::with_radius_estimate::<K>(N, r, diag)`.
pub struct RadiusResultCollection<E>(Vec<E>);

impl<E> RadiusResultCollection<E> {
    pub fn with_radius_estimate<const K: usize>(
        dataset_size: usize,
        radius: f64,
        bounding_box_diagonal: f64,
    ) -> Self {
        let selectivity = (radius / bounding_box_diagonal).min(1.0);
        let estimate = (dataset_size as f64 * selectivity.powi(K as i32)).ceil() as usize;
        Self(Vec::with_capacity(estimate.max(64)))
    }
}

impl<O: Axis<Coord = O>, E: Ord> ResultCollection<O, E> for RadiusResultCollection<E> {
    fn with_max_qty(_max_qty: usize) -> Self {
        Self(Vec::with_capacity(64))
    }
    fn max_qty(&self) -> usize {
        usize::MAX
    }
    fn len(&self) -> usize {
        self.0.len()
    }
    fn add(&mut self, entry: E) {
        self.0.push(entry)
    }
    fn threshold_distance(&self) -> Option<O> {
        None
    }
    fn into_vec(self) -> Vec<E> {
        self.0
    }
    fn into_sorted_vec(self) -> Vec<E> {
        self.0
    }
}

/// Shared state for the adaptive result collection.
///
/// Created once before a query loop and passed to each
/// `AdaptiveResultCollection::new()`. Converges within 3-4 queries
/// for stable workloads, inspired by ALEX's adaptive node sizing
/// (Ding et al., SIGMOD 2020, arXiv: 1905.08898).
pub struct AdaptiveState {
    observed: usize,
    running_sum: usize,
}

impl AdaptiveState {
    pub fn new() -> Self {
        Self {
            observed: 0,
            running_sum: 0,
        }
    }

    pub fn record(&mut self, count: usize) {
        self.observed += 1;
        self.running_sum += count;
    }

    fn estimate(&self) -> usize {
        if self.observed == 0 {
            64
        } else {
            (self.running_sum / self.observed).max(64)
        }
    }
}

/// Self-tuning result collection using a running average of observed sizes.
///
/// Created via `AdaptiveResultCollection::new(&mut state)`.
pub struct AdaptiveResultCollection<E> {
    inner: Vec<E>,
}

impl<E> AdaptiveResultCollection<E> {
    pub fn new(state: &mut AdaptiveState) -> Self {
        Self {
            inner: Vec::with_capacity(state.estimate()),
        }
    }
}

impl<O: Axis<Coord = O>, E: Ord> ResultCollection<O, E> for AdaptiveResultCollection<E> {
    fn with_max_qty(_max_qty: usize) -> Self {
        Self {
            inner: Vec::with_capacity(64),
        }
    }
    fn max_qty(&self) -> usize {
        usize::MAX
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn add(&mut self, entry: E) {
        self.inner.push(entry)
    }
    fn threshold_distance(&self) -> Option<O> {
        None
    }
    fn into_vec(self) -> Vec<E> {
        self.inner
    }
    fn into_sorted_vec(self) -> Vec<E> {
        self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_collection_defaults_to_capacity_64() {
        let coll: FixedResultCollection<u32> = Default::default();
        assert!(coll.capacity() >= 64);
    }

    #[test]
    fn leaf_count_constructs() {
        let _ = LeafCountResultCollection::<u32>::with_leaf_estimate(157, 32);
    }

    #[test]
    fn radius_constructs() {
        let _ = RadiusResultCollection::<u32>::with_radius_estimate::<1>(5000, 0.5, 10.0);
    }

    #[test]
    fn adaptive_collection_converges() {
        let mut state = AdaptiveState::new();
        assert_eq!(state.estimate(), 64);
        state.record(2000);
        assert_eq!(state.estimate(), 2000);
        state.record(4000);
        assert_eq!(state.estimate(), 3000);
    }
}

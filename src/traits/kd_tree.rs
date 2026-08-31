use crate::{Axis, Content, LeafStrategy, StemStrategy};

/// Query-facing interface for resolving stem indices to leaf indices during traversal.
pub trait StemLeafResolution {
    /// Returns true if this resolution strategy uses arithmetic (fast path).
    fn uses_arithmetic(&self) -> bool;

    /// Resolves a terminal stem index to a leaf index.
    fn resolve_terminal_stem_idx(&self, stem_idx: usize, arithmetic_leaf_idx: usize) -> usize;

    /// Returns true if `stem_idx` has an explicit terminal mapping in mapped mode.
    fn is_terminal_stem_idx(&self, stem_idx: usize) -> bool;
}

/// Read-only query access over owned and archived kd-tree storage.
pub trait KdTreeAccessor<A, T, SS, LS, const K: usize, const B: usize>
where
    A: Axis<Coord = A>,
    T: Content,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
{
    /// Stem/branch pivot values in the stem strategy's native layout.
    fn stems(&self) -> &[A];

    /// Leaf storage.
    fn leaves(&self) -> &LS;

    /// Stem-to-leaf resolution strategy.
    fn stem_leaf_resolution(&self) -> &impl StemLeafResolution;

    /// Total number of items.
    fn size(&self) -> usize;

    /// Deepest stem level that can contain an actual pivot.
    fn max_stem_level(&self) -> i32;

    /// Maximum leaf length used for scratch sizing.
    fn max_leaf_len(&self) -> usize;

    /// Encoded item ordering and subtree-summary mode.
    #[inline]
    fn item_leaf_mode_code(&self) -> u8 {
        crate::kd_tree::ITEM_LEAF_MODE_UNSORTED
    }

    /// Whether leaf entries are ordered by ascending item value.
    #[inline]
    fn item_sorted_leaves(&self) -> bool {
        crate::kd_tree::ItemLeafMode::from_code(self.item_leaf_mode_code()).has_sorted_leaves()
    }

    /// Number of leaves.
    #[inline]
    fn leaf_count(&self) -> usize {
        self.leaves().leaf_count()
    }
}

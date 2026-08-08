//! Donnelly SIMD descent with scalar exact-search continuations.
//!
//! One SIMD comparison selects the root-to-terminal path through each complete
//! block. Exact queries then replay only those `BH` path decisions through the
//! ordinary scalar continuation machinery, preserving the same near-first
//! backtracking and pruning semantics as [`super::DonnellyUnrolledBlockDim`].

use std::ptr::NonNull;

use aligned_vec::AVec;

use crate::kd_tree::query_stack::{QueryStack, QueryStackContext};
use crate::stem_strategy::donnelly::core::{
    leaf_idx_from_block_base, DonnellyCore, DonnellyCoreDeferred,
};
use crate::stem_strategy::donnelly::simd_full::{compare_block3, compare_block4};
use crate::{Axis, StemStrategy};

/// Donnelly block-at-once path selection with scalar pruning/backtracking.
#[derive(Copy, Clone, Debug)]
pub struct DonnellySimdDescent<const BH: usize> {
    core: DonnellyCore<BH>,
}

/// Experimental control that uses SIMD only on the initial root-to-leaf
/// descent. Deferred subtrees use the same scalar block-unrolled descent as
/// [`super::DonnellyUnrolledBlockDim`].
#[doc(hidden)]
#[derive(Copy, Clone, Debug)]
pub struct DonnellySimdInitialDescent<const BH: usize> {
    core: DonnellyCore<BH>,
}

macro_rules! impl_donnelly_simd_descent {
    ($strategy:ident, $bh:literal, $compare:path, $block_base_bias:literal, $simd_on_backtrack:literal) => {
        impl StemStrategy for $strategy<$bh> {
            const ROOT_IDX: usize = 0;
            const BLOCK_SIZE: usize = $bh;
            const SUPPORTS_ARITHMETIC_LEAF_RESOLUTION: bool = true;
            const USES_UNROLLED_SCALAR_TRAVERSAL: bool = true;
            const USES_SIMD_BLOCK_DESCENT: bool = true;
            const SIMD_BLOCK_DESCENT_ON_BACKTRACK: bool = $simd_on_backtrack;

            type DeferredState = DonnellyCoreDeferred;
            type StackContext<A, const K: usize> = QueryStackContext<A, Self::DeferredState>;
            type Stack<A, const K: usize> = QueryStack<A, Self, K>;

            #[inline(always)]
            fn new(stems_ptr: NonNull<u8>) -> Self {
                Self {
                    core: DonnellyCore::new(stems_ptr),
                }
            }

            #[inline(always)]
            fn stem_idx(&self) -> usize {
                self.core.stem_idx()
            }

            #[inline(always)]
            fn deferred_state(&self) -> Self::DeferredState {
                self.core.deferred_state()
            }

            #[inline(always)]
            fn rehydrate_deferred_state(&mut self, state: Self::DeferredState) {
                self.core.rehydrate_deferred_state(state);
            }

            #[inline(always)]
            fn leaf_idx(&self) -> usize {
                self.core.leaf_idx()
            }

            #[inline(always)]
            fn dim<const K: usize>(&self) -> usize {
                self.core.level() as usize / $bh % K
            }

            #[inline(always)]
            fn construction_dim<const K: usize>(&self) -> usize {
                self.core.level() as usize / $bh % K
            }

            #[inline(always)]
            fn select_block_child<A: Axis<Coord = A>, const K: usize>(
                &self,
                stems: &[A],
                query: &[A; K],
                start_dim: usize,
            ) -> u8 {
                $compare(
                    stems,
                    unsafe { *query.get_unchecked(start_dim) },
                    self.stem_idx(),
                )
            }

            #[inline(always)]
            fn level(&self) -> i32 {
                self.core.level()
            }

            #[inline(always)]
            fn traverse<A: Axis<Coord = A>, const K: usize>(&mut self, is_right: bool) {
                self.core.traverse::<A, K>(is_right);
            }

            #[inline(always)]
            fn traverse_head<A: Axis<Coord = A>, const K: usize>(&mut self, is_right: bool) {
                self.core.traverse_head::<A, K>(is_right);
            }

            #[inline(always)]
            fn traverse_tail<A: Axis<Coord = A>, const K: usize>(&mut self, is_right: bool) {
                self.core
                    .traverse_tail_with_block_size::<A, K>(is_right, $bh as u32);
            }

            #[inline(always)]
            fn branch<A: Axis<Coord = A>, const K: usize>(&mut self) -> Self {
                Self {
                    core: self.core.branch::<A, K>(),
                }
            }

            #[inline(always)]
            fn branch_relative<A: Axis<Coord = A>, const K: usize>(
                &mut self,
                is_right: bool,
            ) -> Self {
                Self {
                    core: self.core.branch_relative::<A, K>(is_right),
                }
            }

            #[inline(always)]
            fn branch_relative_head<A: Axis<Coord = A>, const K: usize>(
                &mut self,
                is_right: bool,
            ) -> Self {
                Self {
                    core: self.core.branch_relative_head::<A, K>(is_right),
                }
            }

            #[inline(always)]
            fn branch_relative_tail<A: Axis<Coord = A>, const K: usize>(
                &mut self,
                is_right: bool,
            ) -> Self {
                Self {
                    core: self.core.branch_relative_tail::<A, K>(is_right),
                }
            }

            #[inline(always)]
            fn child_indices<A: Axis<Coord = A>>(&self) -> (usize, usize) {
                self.core.child_indices::<A>()
            }

            fn get_stem_node_count_from_leaf_node_count(leaf_node_count: usize) -> usize {
                if leaf_node_count < 2 {
                    0
                } else {
                    leaf_node_count.next_power_of_two() - 1
                }
            }

            fn stem_node_padding_factor() -> usize {
                50
            }

            fn trim_unneeded_stems<A: Axis<Coord = A>, const K: usize>(
                stems: &mut AVec<A>,
                max_stem_level: usize,
            ) {
                if stems.is_empty() {
                    return;
                }

                let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
                let mut ordering = Self::new(stems_ptr);
                loop {
                    let is_right_child = !A::is_max_value(stems[ordering.stem_idx()]);
                    ordering.traverse::<A, K>(is_right_child);
                    if ordering.level() as usize == max_stem_level {
                        break;
                    }
                }

                #[cfg(debug_assertions)]
                for idx in (ordering.stem_idx() + 1)..stems.len() {
                    debug_assert!(A::is_max_value(stems[idx]), "stems[{idx}] = {}", stems[idx]);
                }

                stems.truncate(ordering.stem_idx() + 1);
            }

            fn get_leaf_idx<A: Axis<Coord = A>, const K: usize>(
                stems: &[A],
                query: &[A; K],
                max_stem_level: i32,
            ) -> usize {
                let total_levels = (max_stem_level + 1).max(0) as usize;
                let cache_line_block_height = (64 / A::VALUE_WIDTH_BYTES as u32).ilog2() as usize;

                if cache_line_block_height == $bh && total_levels.is_multiple_of($bh) {
                    let mut block_base = 0u32;
                    let mut dim = 0usize;

                    for _ in 0..(total_levels / $bh) {
                        debug_assert!(block_base as usize + (1 << $bh) <= stems.len());
                        let query_elem = unsafe { *query.get_unchecked(dim) };
                        let child_idx = $compare(stems, query_elem, block_base as usize);
                        block_base = block_base
                            .wrapping_add($block_base_bias)
                            .wrapping_add(child_idx as u32)
                            .wrapping_shl($bh as u32);

                        dim += 1;
                        if dim == K {
                            dim = 0;
                        }
                    }

                    return leaf_idx_from_block_base::<$bh>(block_base, total_levels);
                }

                let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
                let mut ordering = Self::new(stems_ptr);
                while ordering.level() <= max_stem_level {
                    let dim = ordering.dim::<K>();
                    let query_elem = unsafe { *query.get_unchecked(dim) };
                    let stem_idx = ordering.stem_idx();
                    let is_right = stem_idx < stems.len()
                        && query_elem >= unsafe { *stems.get_unchecked(stem_idx) };
                    ordering.traverse::<A, K>(is_right);
                }
                ordering.leaf_idx()
            }
        }
    };
}

impl_donnelly_simd_descent!(DonnellySimdDescent, 3, compare_block3, 1, true);
impl_donnelly_simd_descent!(DonnellySimdDescent, 4, compare_block4, 7, true);
impl_donnelly_simd_descent!(DonnellySimdInitialDescent, 3, compare_block3, 1, false);
impl_donnelly_simd_descent!(DonnellySimdInitialDescent, 4, compare_block4, 7, false);

#[cfg(test)]
mod tests {
    use super::*;

    fn block3_pivots() -> [f64; 8] {
        [0.4, 0.2, 0.6, 0.1, 0.3, 0.5, 0.7, f64::INFINITY]
    }

    fn block4_pivots() -> [f32; 16] {
        [
            0.8,
            0.4,
            1.2,
            0.2,
            0.6,
            1.0,
            1.4,
            0.1,
            0.3,
            0.5,
            0.7,
            0.9,
            1.1,
            1.3,
            1.5,
            f32::INFINITY,
        ]
    }

    fn replay_child<const BH: usize, A: Axis<Coord = A>>(
        stems: &[A],
        child_idx: u8,
    ) -> DonnellySimdDescent<BH>
    where
        DonnellySimdDescent<BH>: StemStrategy,
    {
        let ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        let mut ordering = DonnellySimdDescent::<BH>::new(ptr);
        for bit in (0..BH).rev() {
            ordering.traverse::<A, 3>(child_idx & (1 << bit) != 0);
        }
        ordering
    }

    #[test]
    fn block3_simd_child_encodes_scalar_path() {
        let stems = block3_pivots();
        for query in [0.05, 0.15, 0.35, 0.45, 0.65, 0.75] {
            let ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
            let ordering = DonnellySimdDescent::<3>::new(ptr);
            let child_idx = ordering.select_block_child(&stems, &[query, 0.0, 0.0], 0);
            let replayed = replay_child::<3, f64>(&stems, child_idx);
            assert_eq!(replayed.leaf_idx(), child_idx as usize);
        }
    }

    #[test]
    fn block4_simd_child_encodes_scalar_path() {
        let stems = block4_pivots();
        for query in [0.05, 0.15, 0.45, 0.85, 1.25, 1.55] {
            let ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
            let ordering = DonnellySimdDescent::<4>::new(ptr);
            let child_idx = ordering.select_block_child(&stems, &[query, 0.0, 0.0], 0);
            let replayed = replay_child::<4, f32>(&stems, child_idx);
            assert_eq!(replayed.leaf_idx(), child_idx as usize);
        }
    }

    #[test]
    fn canonical_variants_use_arithmetic_scalar_continuations() {
        const {
            assert!(DonnellySimdDescent::<3>::SUPPORTS_ARITHMETIC_LEAF_RESOLUTION);
            assert!(DonnellySimdDescent::<4>::SUPPORTS_ARITHMETIC_LEAF_RESOLUTION);
            assert!(DonnellySimdDescent::<3>::USES_SIMD_BLOCK_DESCENT);
            assert!(DonnellySimdDescent::<4>::USES_SIMD_BLOCK_DESCENT);
        }
    }

    #[test]
    fn block_fast_leaf_selection_matches_scalar_replay() {
        let stems = block3_pivots();
        let query = [0.45, 0.0, 0.0];
        let selected = DonnellySimdDescent::<3>::get_leaf_idx(&stems, &query, 2);
        let child_idx = compare_block3(&stems, query[0], 0);
        let replayed = replay_child::<3, f64>(&stems, child_idx);
        assert_eq!(selected, replayed.leaf_idx());
    }
}

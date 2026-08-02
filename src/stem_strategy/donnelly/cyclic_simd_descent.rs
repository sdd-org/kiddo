//! Cyclic-axis SIMD descent for cache-line-sized Donnelly blocks.
//!
//! Unlike [`super::DonnellySimdDescent`], this strategy preserves the ordinary
//! per-level axis cadence. Its SIMD query vector therefore mirrors the heap
//! levels in a block: `X, YY, ZZZZ` for `f64`/3D/block-height 3 and
//! `X, YY, ZZZZ, WWWWWWWW` for `f32`/4D/block-height 4.

use std::ptr::NonNull;

use aligned_vec::AVec;

use crate::kd_tree::query_stack::{QueryStack, QueryStackContext};
use crate::stem_strategy::donnelly::core::{
    leaf_idx_from_block_base, DonnellyCore, DonnellyCoreDeferred,
};
use crate::{Axis, StemStrategy};

/// Type-specific cyclic block comparison used by [`Axis`].
pub(crate) trait CyclicBlockCompare: Copy {
    fn compare_cyclic_block<const BH: usize, const K: usize>(
        stems: &[Self],
        query: &[Self; K],
        block_base_idx: usize,
        start_dim: usize,
    ) -> u8;
}

#[inline(always)]
#[cfg(not(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f")))]
fn scalar_path<A: PartialOrd + Copy, const BH: usize, const K: usize>(
    stems: &[A],
    query: &[A; K],
    block_base_idx: usize,
    start_dim: usize,
) -> u8 {
    let mut heap_idx = 0usize;
    let mut child = 0u8;
    let mut depth = 0usize;
    while depth < BH {
        let dim = (start_dim + depth) % K;
        let is_right =
            unsafe { *query.get_unchecked(dim) >= *stems.get_unchecked(block_base_idx + heap_idx) };
        child = (child << 1) | is_right as u8;
        heap_idx = (heap_idx << 1) + 1 + is_right as usize;
        depth += 1;
    }
    child
}

// TODO: can remove?
#[allow(unused)]
#[inline(always)]
fn block3_child_from_mask(mask: u8) -> u8 {
    let b0 = mask & 1;
    let b1 = (mask >> (1 + b0)) & 1;
    let b2 = (mask >> (3 + (b0 << 1) + b1)) & 1;
    (b0 << 2) | (b1 << 1) | b2
}

// TODO: can remove?
#[allow(unused)]
#[inline(always)]
fn block4_child_from_mask(mask: u16) -> u8 {
    let b0 = (mask & 1) as u8;
    let b1 = ((mask >> (1 + b0)) & 1) as u8;
    let b2 = ((mask >> (3 + (b0 << 1) + b1)) & 1) as u8;
    let b3 = ((mask >> (7 + (b0 << 2) + (b1 << 1) + b2)) & 1) as u8;
    (b0 << 3) | (b1 << 2) | (b2 << 1) | b3
}

impl CyclicBlockCompare for f64 {
    #[inline(always)]
    fn compare_cyclic_block<const BH: usize, const K: usize>(
        stems: &[Self],
        query: &[Self; K],
        block_base_idx: usize,
        start_dim: usize,
    ) -> u8 {
        assert!(BH == 3 && K == 3, "f64 cyclic SIMD requires BH=3 and K=3");
        debug_assert_eq!(start_dim, 0);

        #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            use std::arch::x86_64::*;

            let query_xyz = _mm256_maskz_loadu_pd(0b0111, query.as_ptr());
            let repeated = _mm512_broadcast_f64x4(query_xyz);
            let lane_indices = _mm512_set_epi64(2, 2, 2, 2, 2, 1, 1, 0);
            let query_lanes = _mm512_permutexvar_pd(lane_indices, repeated);
            let pivots = _mm512_loadu_pd(stems.as_ptr().add(block_base_idx));
            let mask = _mm512_cmp_pd_mask(query_lanes, pivots, _CMP_GE_OQ);
            block3_child_from_mask(mask)
        }

        #[cfg(not(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f")))]
        scalar_path::<Self, BH, K>(stems, query, block_base_idx, start_dim)
    }
}

impl CyclicBlockCompare for f32 {
    #[inline(always)]
    fn compare_cyclic_block<const BH: usize, const K: usize>(
        stems: &[Self],
        query: &[Self; K],
        block_base_idx: usize,
        start_dim: usize,
    ) -> u8 {
        assert!(BH == 4 && K == 4, "f32 cyclic SIMD requires BH=4 and K=4");
        debug_assert_eq!(start_dim, 0);

        #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            use std::arch::x86_64::*;

            let query_xyzw = _mm_loadu_ps(query.as_ptr());
            let repeated = _mm512_broadcast_f32x4(query_xyzw);
            let lane_indices = _mm512_set_epi32(3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 0);
            let query_lanes = _mm512_permutexvar_ps(lane_indices, repeated);
            let pivots = _mm512_loadu_ps(stems.as_ptr().add(block_base_idx));
            let mask = _mm512_cmp_ps_mask(query_lanes, pivots, _CMP_GE_OQ);
            block4_child_from_mask(mask)
        }

        #[cfg(not(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f")))]
        scalar_path::<Self, BH, K>(stems, query, block_base_idx, start_dim)
    }
}

/// Donnelly block-at-once descent with the ordinary per-level axis cadence.
///
/// AVX-512 acceleration is currently restricted to `f64`/3D/BH3 and
/// `f32`/4D/BH4. Other type/dimension combinations are not supported by this
/// experimental strategy.
#[derive(Copy, Clone, Debug)]
pub struct DonnellyCyclicSimdDescent<const BH: usize> {
    core: DonnellyCore<BH>,
}

macro_rules! impl_cyclic_simd_descent {
    ($bh:literal, $block_base_bias:literal) => {
        impl StemStrategy for DonnellyCyclicSimdDescent<$bh> {
            const ROOT_IDX: usize = 0;
            const BLOCK_SIZE: usize = $bh;
            const SUPPORTS_ARITHMETIC_LEAF_RESOLUTION: bool = true;
            const USES_UNROLLED_SCALAR_TRAVERSAL: bool = true;
            const USES_SIMD_BLOCK_DESCENT: bool = true;

            type DeferredState = DonnellyCoreDeferred;
            type StackContext<A> = QueryStackContext<A, Self::DeferredState>;
            type Stack<A> = QueryStack<A, Self>;

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
                self.core.dim::<K>()
            }
            #[inline(always)]
            fn construction_dim<const K: usize>(&self) -> usize {
                self.core.dim::<K>()
            }

            #[inline(always)]
            fn select_block_child<A: Axis<Coord = A>, const K: usize>(
                &self,
                stems: &[A],
                query: &[A; K],
                start_dim: usize,
            ) -> u8 {
                A::compare_cyclic_block::<$bh, K>(stems, query, self.stem_idx(), start_dim)
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
                let ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
                let mut ordering = Self::new(ptr);
                loop {
                    let is_right = !A::is_max_value(stems[ordering.stem_idx()]);
                    ordering.traverse::<A, K>(is_right);
                    if ordering.level() as usize == max_stem_level {
                        break;
                    }
                }
                stems.truncate(ordering.stem_idx() + 1);
            }

            fn get_leaf_idx<A: Axis<Coord = A>, const K: usize>(
                stems: &[A],
                query: &[A; K],
                max_stem_level: i32,
            ) -> usize {
                let total_levels = (max_stem_level + 1).max(0) as usize;
                let native_bh = (64 / A::VALUE_WIDTH_BYTES as u32).ilog2() as usize;
                if native_bh == $bh && K == $bh && total_levels.is_multiple_of($bh) {
                    let mut block_base = 0u32;
                    for _ in 0..(total_levels / $bh) {
                        let child =
                            A::compare_cyclic_block::<$bh, K>(stems, query, block_base as usize, 0);
                        block_base = block_base
                            .wrapping_add($block_base_bias)
                            .wrapping_add(child as u32)
                            .wrapping_shl($bh as u32);
                    }
                    return leaf_idx_from_block_base::<$bh>(block_base, total_levels);
                }

                let ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
                let mut ordering = Self::new(ptr);
                while ordering.level() <= max_stem_level {
                    let dim = ordering.dim::<K>();
                    let pivot = unsafe { *stems.get_unchecked(ordering.stem_idx()) };
                    let is_right = unsafe { *query.get_unchecked(dim) } >= pivot;
                    ordering.traverse::<A, K>(is_right);
                }
                ordering.leaf_idx()
            }
        }
    };
}

impl_cyclic_simd_descent!(3, 1);
impl_cyclic_simd_descent!(4, 7);

#[cfg(feature = "cargo_asm")]
mod cargo_asm {
    use crate::Axis;

    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn donnelly_cyclic_block3_f64_k3_cargo_asm_hook(
        stems: &[f64],
        query: &[f64; 3],
        block_base: usize,
    ) -> u8 {
        <f64 as Axis>::compare_cyclic_block::<3, 3>(stems, query, block_base, 0)
    }

    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn donnelly_cyclic_block4_f32_k4_cargo_asm_hook(
        stems: &[f32],
        query: &[f32; 4],
        block_base: usize,
    ) -> u8 {
        <f32 as Axis>::compare_cyclic_block::<4, 4>(stems, query, block_base, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_block3_mask_follows_xyz_path() {
        let stems = [0.5, 0.25, 0.75, 0.1, 0.4, 0.6, 0.9, f64::INFINITY];
        let query = [0.8, 0.2, 0.7];
        assert_eq!(
            <f64 as Axis>::compare_cyclic_block::<3, 3>(&stems, &query, 0, 0),
            0b101
        );
    }

    #[test]
    fn f32_block4_mask_follows_xyzw_path() {
        let stems = [
            0.5,
            0.25,
            0.75,
            0.1,
            0.4,
            0.6,
            0.9,
            0.05,
            0.2,
            0.3,
            0.45,
            0.55,
            0.7,
            0.8,
            0.95,
            f32::INFINITY,
        ];
        let query = [0.8, 0.2, 0.7, 0.75];
        assert_eq!(
            <f32 as Axis>::compare_cyclic_block::<4, 4>(&stems, &query, 0, 0),
            0b1011
        );
    }
}

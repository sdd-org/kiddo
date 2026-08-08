//! Cyclic-axis SIMD descent for cache-line-sized Donnelly blocks.
//!
//! Unlike [`super::DonnellySimdDescent`], this strategy preserves the ordinary
//! per-level axis cadence. Its SIMD query vector therefore mirrors the heap
//! levels in a block. For a block beginning at dimension `d`, the query lanes
//! are `d, (d+1)(d+1), (d+2)(d+2)(d+2)(d+2)` for block height 3, and the
//! analogous 1/2/4/8 pattern for block height 4. Dimensions wrap modulo `K`.

use std::mem::MaybeUninit;
use std::ptr::NonNull;

use aligned_vec::AVec;

use crate::dist::DistanceMetric;
use crate::kd_tree::query_context::QueryContext;
use crate::kd_tree::query_stack::{QueryStack, QueryStackContext};
use crate::kd_tree::{KdTreeAccessor, KdTreeQueryOps};
use crate::stem_strategy::donnelly::core::{
    leaf_idx_from_block_base, DonnellyCore, DonnellyCoreDeferred,
};
use crate::stem_strategy::SimdPrune;
use crate::traits::stem_strategy::PreparedBlockQuery;
use crate::{Axis, Content, LeafStrategy, StemStrategy};

/// Type-specific cyclic block comparison used by [`Axis`].
pub(crate) trait CyclicBlockCompare: Copy {
    fn compare_cyclic_block<const BH: usize, const K: usize>(
        stems: &[Self],
        query: &[Self; K],
        block_base_idx: usize,
        start_dim: usize,
    ) -> u8;

    fn compare_prepared_cyclic_block<const BH: usize>(
        stems: &[Self],
        query_lanes: &[MaybeUninit<Self>; 16],
        block_base_idx: usize,
    ) -> u8;
}

#[inline(always)]
pub(super) fn prepare_query_lanes<A: Axis<Coord = A>, const BH: usize, const K: usize>(
    query: &[A; K],
) -> PreparedBlockQuery<A, K> {
    assert!(K > 0);
    let mut phases = PreparedBlockQuery([[MaybeUninit::uninit(); 16]; K]);
    let mut phase = 0usize;
    loop {
        let dim1 = if phase + 1 == K { 0 } else { phase + 1 };
        let dim2 = if dim1 + 1 == K { 0 } else { dim1 + 1 };
        let q0 = query[phase];
        let q1 = query[dim1];
        let q2 = query[dim2];
        if BH == 3 {
            phases.0[phase] = [
                MaybeUninit::new(q0),
                MaybeUninit::new(q1),
                MaybeUninit::new(q1),
                MaybeUninit::new(q2),
                MaybeUninit::new(q2),
                MaybeUninit::new(q2),
                MaybeUninit::new(q2),
                MaybeUninit::new(q2),
                MaybeUninit::uninit(),
                MaybeUninit::uninit(),
                MaybeUninit::uninit(),
                MaybeUninit::uninit(),
                MaybeUninit::uninit(),
                MaybeUninit::uninit(),
                MaybeUninit::uninit(),
                MaybeUninit::uninit(),
            ];
        } else {
            assert!(BH == 4, "cyclic SIMD preparation requires BH=3 or BH=4");
            let dim3 = if dim2 + 1 == K { 0 } else { dim2 + 1 };
            let q3 = query[dim3];
            phases.0[phase] = [
                MaybeUninit::new(q0),
                MaybeUninit::new(q1),
                MaybeUninit::new(q1),
                MaybeUninit::new(q2),
                MaybeUninit::new(q2),
                MaybeUninit::new(q2),
                MaybeUninit::new(q2),
                MaybeUninit::new(q3),
                MaybeUninit::new(q3),
                MaybeUninit::new(q3),
                MaybeUninit::new(q3),
                MaybeUninit::new(q3),
                MaybeUninit::new(q3),
                MaybeUninit::new(q3),
                MaybeUninit::new(q3),
                MaybeUninit::new(q3),
            ];
        }
        phase = (phase + BH) % K;
        if phase == 0 {
            break;
        }
    }
    phases
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
#[cfg(not(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f")))]
fn scalar_prepared_path<A: PartialOrd + Copy, const BH: usize>(
    stems: &[A],
    query_lanes: &[MaybeUninit<A>; 16],
    block_base_idx: usize,
) -> u8 {
    let mut heap_idx = 0usize;
    let mut child = 0u8;
    let mut depth = 0usize;
    while depth < BH {
        let is_right = unsafe {
            query_lanes.get_unchecked(heap_idx).assume_init()
                >= *stems.get_unchecked(block_base_idx + heap_idx)
        };
        child = (child << 1) | is_right as u8;
        heap_idx = (heap_idx << 1) + 1 + is_right as usize;
        depth += 1;
    }
    child
}

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
fn block3_child_from_mask(mask: u8) -> u8 {
    let b0 = mask & 1;
    let b1 = (mask >> (1 + b0)) & 1;
    let b2 = (mask >> (3 + (b0 << 1) + b1)) & 1;
    (b0 << 2) | (b1 << 1) | b2
}

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
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
        assert!(BH == 3, "f64 cyclic SIMD requires BH=3");
        assert!(K > 0, "cyclic SIMD requires at least one dimension");
        debug_assert!(start_dim < K);

        #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            use std::arch::x86_64::*;

            let dim1 = if start_dim + 1 == K { 0 } else { start_dim + 1 };
            let dim2 = if dim1 + 1 == K { 0 } else { dim1 + 1 };
            let q0 = *query.get_unchecked(start_dim);
            let q1 = *query.get_unchecked(dim1);
            let q2 = *query.get_unchecked(dim2);
            let query_lanes = _mm512_set_pd(q2, q2, q2, q2, q2, q1, q1, q0);
            let pivots = _mm512_loadu_pd(stems.as_ptr().add(block_base_idx));
            let mask = _mm512_cmp_pd_mask(query_lanes, pivots, _CMP_GE_OQ);
            block3_child_from_mask(mask)
        }

        #[cfg(not(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f")))]
        scalar_path::<Self, BH, K>(stems, query, block_base_idx, start_dim)
    }

    #[inline(always)]
    fn compare_prepared_cyclic_block<const BH: usize>(
        stems: &[Self],
        query_lanes: &[MaybeUninit<Self>; 16],
        block_base_idx: usize,
    ) -> u8 {
        assert!(BH == 3, "f64 cyclic SIMD requires BH=3");

        #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            use std::arch::x86_64::*;

            let prepared = _mm512_loadu_pd(query_lanes.as_ptr().cast());
            let pivots = _mm512_loadu_pd(stems.as_ptr().add(block_base_idx));
            return block3_child_from_mask(_mm512_cmp_pd_mask(prepared, pivots, _CMP_GE_OQ));
        }

        #[cfg(not(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f")))]
        scalar_prepared_path::<Self, BH>(stems, query_lanes, block_base_idx)
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
        assert!(BH == 4, "f32 cyclic SIMD requires BH=4");
        assert!(K > 0, "cyclic SIMD requires at least one dimension");
        debug_assert!(start_dim < K);

        #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            use std::arch::x86_64::*;

            let dim1 = if start_dim + 1 == K { 0 } else { start_dim + 1 };
            let dim2 = if dim1 + 1 == K { 0 } else { dim1 + 1 };
            let dim3 = if dim2 + 1 == K { 0 } else { dim2 + 1 };
            let q0 = *query.get_unchecked(start_dim);
            let q1 = *query.get_unchecked(dim1);
            let q2 = *query.get_unchecked(dim2);
            let q3 = *query.get_unchecked(dim3);
            let query_lanes = _mm512_set_ps(
                q3, q3, q3, q3, q3, q3, q3, q3, q3, q2, q2, q2, q2, q1, q1, q0,
            );
            let pivots = _mm512_loadu_ps(stems.as_ptr().add(block_base_idx));
            let mask = _mm512_cmp_ps_mask(query_lanes, pivots, _CMP_GE_OQ);
            block4_child_from_mask(mask)
        }

        #[cfg(not(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f")))]
        scalar_path::<Self, BH, K>(stems, query, block_base_idx, start_dim)
    }

    #[inline(always)]
    fn compare_prepared_cyclic_block<const BH: usize>(
        stems: &[Self],
        query_lanes: &[MaybeUninit<Self>; 16],
        block_base_idx: usize,
    ) -> u8 {
        assert!(BH == 4, "f32 cyclic SIMD requires BH=4");

        #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            use std::arch::x86_64::*;

            let prepared = _mm512_loadu_ps(query_lanes.as_ptr().cast());
            let pivots = _mm512_loadu_ps(stems.as_ptr().add(block_base_idx));
            return block4_child_from_mask(_mm512_cmp_ps_mask(prepared, pivots, _CMP_GE_OQ));
        }

        #[cfg(not(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f")))]
        scalar_prepared_path::<Self, BH>(stems, query_lanes, block_base_idx)
    }
}

/// Donnelly block-at-once descent with the ordinary per-level axis cadence.
///
/// AVX-512 acceleration uses a block-height-3 specialization for `f64` and a
/// block-height-4 specialization for `f32`. The dimension count is independent
/// of the block height; blocks carry their cyclic start phase explicitly.
#[derive(Copy, Clone, Debug)]
pub struct DonnellyCyclicSimdDescent<const BH: usize> {
    core: DonnellyCore<BH>,
}

macro_rules! impl_cyclic_simd_descent {
    (
        $strategy:ident,
        $bh:literal,
        $block_base_bias:literal,
        $stack_context:ty,
        $stack_type:ty,
        $tree:ident,
        $query_ctx:ident,
        $stack:ident,
        $process_leaf:ident,
        $arithmetic:block
    ) => {
        impl StemStrategy for $strategy<$bh> {
            const ROOT_IDX: usize = 0;
            const BLOCK_SIZE: usize = $bh;
            const SUPPORTS_ARITHMETIC_LEAF_RESOLUTION: bool = true;
            const USES_UNROLLED_SCALAR_TRAVERSAL: bool = true;
            const USES_SIMD_BLOCK_DESCENT: bool = true;
            const USES_PREPARED_BLOCK_QUERY: bool = true;

            type DeferredState = DonnellyCoreDeferred;
            type StackContext<A, const K: usize> = $stack_context;
            type Stack<A, const K: usize> = $stack_type;

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
            fn prepare_block_query<A: Axis<Coord = A>, const K: usize>(
                query: &[A; K],
            ) -> PreparedBlockQuery<A, K> {
                prepare_query_lanes::<A, $bh, K>(query)
            }

            #[inline(always)]
            fn select_prepared_block_child<A: Axis<Coord = A>, const K: usize>(
                &self,
                stems: &[A],
                _query: &[A; K],
                prepared: &PreparedBlockQuery<A, K>,
                start_dim: usize,
            ) -> u8 {
                A::compare_prepared_cyclic_block::<$bh>(
                    stems,
                    &prepared[start_dim],
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
                if native_bh == $bh && total_levels.is_multiple_of($bh) {
                    let mut block_base = 0u32;
                    let mut start_dim = 0usize;
                    let prepared = prepare_query_lanes::<A, $bh, K>(query);
                    for _ in 0..(total_levels / $bh) {
                        let child = A::compare_prepared_cyclic_block::<$bh>(
                            stems,
                            &prepared[start_dim],
                            block_base as usize,
                        );
                        block_base = block_base
                            .wrapping_add($block_base_bias)
                            .wrapping_add(child as u32)
                            .wrapping_shl($bh as u32);
                        start_dim = (start_dim + $bh) % K;
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

            #[inline(always)]
            fn arithmetic_query_with_scratch<
                Tree,
                A,
                T,
                O,
                D,
                QC,
                LS,
                const K2: usize,
                const B: usize,
            >(
                $tree: &Tree,
                $query_ctx: &mut QC,
                $stack: &mut Self::Stack<O, K2>,
                $process_leaf: impl FnMut(usize, &[O; K2], &mut QC),
            ) where
                Self: Sized,
                Tree: KdTreeAccessor<A, T, Self, LS, K2, B>
                    + KdTreeQueryOps<A, T, Self, LS, K2, B>,
                A: Axis<Coord = A>,
                T: Content,
                O: Axis<Coord = O>
                    + SimdPrune
                    + crate::stem_strategy::SimdSelectBestChildBlock3
                    + super::simd_full::BacktrackBlock3
                    + super::simd_full::BacktrackBlock4,
                D: DistanceMetric<A, Output = O>,
                QC: QueryContext<A, O, K2>,
                LS: LeafStrategy<A, T, Self, K2, B>,
                Self::Stack<O, K2>: crate::kd_tree::query_stack::StackTrait<O, Self, K2>,
            $arithmetic
        }
    };
}

impl_cyclic_simd_descent!(
    DonnellyCyclicSimdDescent,
    3,
    1,
    QueryStackContext<A, Self::DeferredState>,
    QueryStack<A, Self, K>,
    tree,
    query_ctx,
    stack,
    process_leaf,
    {
        tree.arithmetic_query_with_scratch_impl::<QC, O, D>(query_ctx, stack, process_leaf);
    }
);
impl_cyclic_simd_descent!(
    DonnellyCyclicSimdDescent,
    4,
    7,
    QueryStackContext<A, Self::DeferredState>,
    QueryStack<A, Self, K>,
    tree,
    query_ctx,
    stack,
    process_leaf,
    {
        tree.arithmetic_query_with_scratch_impl::<QC, O, D>(query_ctx, stack, process_leaf);
    }
);
#[cfg(feature = "cargo_asm")]
mod cargo_asm {
    use super::{prepare_query_lanes, MaybeUninit, PreparedBlockQuery};
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

    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn donnelly_cyclic_block3_f64_prepared_cargo_asm_hook(
        stems: &[f64],
        query_lanes: &[MaybeUninit<f64>; 16],
        block_base: usize,
    ) -> u8 {
        <f64 as Axis>::compare_prepared_cyclic_block::<3>(stems, query_lanes, block_base)
    }

    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn donnelly_cyclic_prepare_f64_k4_cargo_asm_hook(
        query: &[f64; 4],
    ) -> PreparedBlockQuery<f64, 4> {
        prepare_query_lanes::<f64, 3, 4>(query)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_reference<A: PartialOrd + Copy, const BH: usize, const K: usize>(
        stems: &[A],
        query: &[A; K],
        start_dim: usize,
    ) -> u8 {
        let mut heap_idx = 0usize;
        let mut child = 0u8;
        for depth in 0..BH {
            let is_right = query[(start_dim + depth) % K] >= stems[heap_idx];
            child = (child << 1) | is_right as u8;
            heap_idx = (heap_idx << 1) + 1 + is_right as usize;
        }
        child
    }

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

    #[test]
    fn f64_block3_honours_every_k4_start_phase() {
        let stems = [0.5, 0.25, 0.75, 0.1, 0.4, 0.6, 0.9, f64::INFINITY];
        let query = [0.8, 0.2, 0.7, 0.35];
        let prepared = prepare_query_lanes::<f64, 3, 4>(&query);
        for start_dim in 0..4 {
            assert_eq!(
                <f64 as Axis>::compare_cyclic_block::<3, 4>(&stems, &query, 0, start_dim,),
                scalar_reference::<_, 3, 4>(&stems, &query, start_dim),
            );
            assert_eq!(
                <f64 as Axis>::compare_prepared_cyclic_block::<3>(&stems, &prepared[start_dim], 0,),
                scalar_reference::<_, 3, 4>(&stems, &query, start_dim),
            );
        }
    }

    #[test]
    fn f32_block4_honours_every_k3_start_phase() {
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
        let query = [0.8, 0.2, 0.7];
        let prepared = prepare_query_lanes::<f32, 4, 3>(&query);
        for start_dim in 0..3 {
            assert_eq!(
                <f32 as Axis>::compare_cyclic_block::<4, 3>(&stems, &query, 0, start_dim,),
                scalar_reference::<_, 4, 3>(&stems, &query, start_dim),
            );
            assert_eq!(
                <f32 as Axis>::compare_prepared_cyclic_block::<4>(&stems, &prepared[start_dim], 0,),
                scalar_reference::<_, 4, 3>(&stems, &query, start_dim),
            );
        }
    }
}

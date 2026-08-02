//! Cyclic-axis SIMD descent with block-level SIMD pruning and backtracking.

use std::any::TypeId;
use std::ptr::NonNull;

use aligned_vec::AVec;

use crate::dist::DistanceMetric;
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
use crate::dist::{
    distance_metric_avx512::{
        Avx512F32LeafOps, Avx512F64LeafOps, UnsupportedAvx512F32LeafOps,
        UnsupportedAvx512F64LeafOps,
    },
    DistanceMetricAvx512,
};
use crate::kd_tree::query_context::QueryContext;
use crate::kd_tree::query_stack::StackTrait;
use crate::kd_tree::query_stack_simd::{
    CyclicSimdFullQueryStackContext, SimdQueryStack, CYCLIC_SIMD_FULL_INLINE_QUERY_STACK_CAPACITY,
};
use crate::kd_tree::{KdTreeAccessor, KdTreeQueryOps};
use crate::stem_strategy::donnelly::core::{
    leaf_idx_from_block_base, DonnellyCore, DonnellyCoreDeferred,
};
use crate::stem_strategy::SimdPrune;
use crate::traits::stem_strategy::PreparedBlockQuery;
use crate::{Axis, Content, LeafStrategy, StemStrategy};

use super::cyclic_simd_descent::prepare_query_lanes;

#[derive(Clone, Copy, Debug)]
struct CyclicChildSelection<O> {
    child_idx: u8,
    remaining_mask: u16,
    child_off: [O; 4],
}

#[inline(always)]
fn selected_cyclic_child_offsets<A, O, D, const K: usize, const BH: usize>(
    stems: &[A],
    block_base_idx: usize,
    query: &[A; K],
    parent_off: &[O; 4],
    child_idx: u8,
) -> Option<([O; 4], O)>
where
    A: Axis<Coord = A>,
    O: Axis<Coord = O>,
    D: DistanceMetric<A, Output = O>,
{
    debug_assert_eq!(K, BH);
    debug_assert!(K <= 4);

    let mut child_off = *parent_off;
    let mut heap_idx = 0usize;
    let mut depth = 0usize;
    while depth < BH {
        let goes_right = ((child_idx >> (BH - 1 - depth)) & 1) != 0;
        let pivot = unsafe { *stems.get_unchecked(block_base_idx + heap_idx) };
        if A::is_max_value(pivot) {
            if goes_right {
                return None;
            }
        } else {
            let query_value = unsafe { *query.get_unchecked(depth) };
            let query_goes_right = query_value >= pivot;
            if goes_right != query_goes_right {
                child_off[depth] =
                    O::saturating_dist(D::widen_coord(query_value), D::widen_coord(pivot));
            }
        }
        heap_idx = (heap_idx << 1) + 1 + goes_right as usize;
        depth += 1;
    }

    let rd = D::rect_dist_from_off(unsafe { &*child_off.as_ptr().cast::<[O; K]>() });
    Some((child_off, rd))
}

#[inline(always)]
fn select_cyclic_child_scalar<A, O, D, const K: usize, const BH: usize>(
    stems: &[A],
    block_base_idx: usize,
    query: &[A; K],
    parent_off: &[O; 4],
    pending_mask: u16,
    max_dist: O,
    prune_equal: bool,
) -> Option<CyclicChildSelection<O>>
where
    A: Axis<Coord = A>,
    O: Axis<Coord = O>,
    D: DistanceMetric<A, Output = O>,
{
    let mut candidate_mask = 0u16;
    let mut rd_values = [O::max_value(); 16];
    let mut off_values = [[O::zero(); 4]; 16];
    let mut remaining = pending_mask;
    while remaining != 0 {
        let child = remaining.trailing_zeros() as usize;
        remaining &= remaining - 1;
        if let Some((child_off, rd)) = selected_cyclic_child_offsets::<A, O, D, K, BH>(
            stems,
            block_base_idx,
            query,
            parent_off,
            child as u8,
        ) {
            let ordering = O::cmp(rd, max_dist);
            let survives = ordering == std::cmp::Ordering::Less
                || (!prune_equal && ordering == std::cmp::Ordering::Equal);
            if survives {
                candidate_mask |= 1u16 << child;
                rd_values[child] = rd;
                off_values[child] = child_off;
            }
        }
    }

    if candidate_mask == 0 {
        return None;
    }
    let mut live = candidate_mask;
    let mut best = live.trailing_zeros() as usize;
    live &= live - 1;
    while live != 0 {
        let child = live.trailing_zeros() as usize;
        live &= live - 1;
        if O::cmp(rd_values[child], rd_values[best]) == std::cmp::Ordering::Less {
            best = child;
        }
    }
    let remaining_mask = candidate_mask & !(1u16 << best);
    Some(CyclicChildSelection {
        child_idx: best as u8,
        remaining_mask,
        child_off: off_values[best],
    })
}

#[cfg(all(
    feature = "simd",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "fma"
))]
#[inline(always)]
unsafe fn cyclic_axis_offsets_f64(
    pivots: std::arch::x86_64::__m512d,
    right_mask: u8,
    query: f64,
    parent_off: f64,
) -> (std::arch::x86_64::__m512d, u8) {
    use std::arch::x86_64::*;

    let query = _mm512_set1_pd(query);
    let query_right = _mm512_cmp_pd_mask(query, pivots, _CMP_GE_OQ);
    let opposite = right_mask ^ query_right;
    let diff = _mm512_andnot_pd(_mm512_set1_pd(-0.0), _mm512_sub_pd(query, pivots));
    let off = _mm512_mask_mov_pd(_mm512_set1_pd(parent_off), opposite, diff);
    let real_pivot = _mm512_cmp_pd_mask(pivots, _mm512_set1_pd(f64::INFINITY), _CMP_LT_OQ);
    (off, (!right_mask) | real_pivot)
}

#[cfg(all(
    feature = "simd",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "fma"
))]
#[inline(always)]
unsafe fn select_cyclic_block3_f64<Ops: Avx512F64LeafOps>(
    stems: *const f64,
    block_base_idx: usize,
    query: *const f64,
    parent_off: *const f64,
    pending_mask: u16,
    max_dist: f64,
    prune_equal: bool,
    child_off: *mut f64,
) -> u32 {
    use std::arch::x86_64::*;

    let block = _mm512_loadu_pd(stems.add(block_base_idx));
    let x = _mm512_permutexvar_pd(_mm512_setzero_si512(), block);
    let y = _mm512_permutexvar_pd(_mm512_set_epi64(2, 2, 2, 2, 1, 1, 1, 1), block);
    let z = _mm512_permutexvar_pd(_mm512_set_epi64(6, 6, 5, 5, 4, 4, 3, 3), block);

    let (off_x, valid_x) = cyclic_axis_offsets_f64(x, 0xf0, *query, *parent_off);
    let (off_y, valid_y) = cyclic_axis_offsets_f64(y, 0xcc, *query.add(1), *parent_off.add(1));
    let (off_z, valid_z) = cyclic_axis_offsets_f64(z, 0xaa, *query.add(2), *parent_off.add(2));
    let rd = Ops::rect_dist_f64x8_3(off_x, off_y, off_z);
    let threshold = _mm512_set1_pd(max_dist);
    let within = if prune_equal {
        _mm512_cmp_pd_mask(rd, threshold, _CMP_LT_OQ)
    } else {
        _mm512_cmp_pd_mask(rd, threshold, _CMP_LE_OQ)
    };
    let candidates = (pending_mask as u8) & valid_x & valid_y & valid_z & within;
    if candidates == 0 {
        return u32::MAX;
    }

    let child = candidates.trailing_zeros() as u8;
    let lane = _mm512_set1_epi64(i64::from(child));
    *child_off = _mm_cvtsd_f64(_mm512_castpd512_pd128(_mm512_permutexvar_pd(lane, off_x)));
    *child_off.add(1) = _mm_cvtsd_f64(_mm512_castpd512_pd128(_mm512_permutexvar_pd(lane, off_y)));
    *child_off.add(2) = _mm_cvtsd_f64(_mm512_castpd512_pd128(_mm512_permutexvar_pd(lane, off_z)));
    *child_off.add(3) = 0.0;

    let remaining = candidates & candidates.wrapping_sub(1);
    u32::from(child) | (u32::from(remaining) << 8)
}

#[cfg(all(
    feature = "simd",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "fma"
))]
#[inline(always)]
unsafe fn cyclic_axis_offsets_f32(
    pivots: std::arch::x86_64::__m512,
    right_mask: u16,
    query: f32,
    parent_off: f32,
) -> (std::arch::x86_64::__m512, u16) {
    use std::arch::x86_64::*;

    let query = _mm512_set1_ps(query);
    let query_right = _mm512_cmp_ps_mask(query, pivots, _CMP_GE_OQ);
    let opposite = right_mask ^ query_right;
    let diff = _mm512_andnot_ps(_mm512_set1_ps(-0.0), _mm512_sub_ps(query, pivots));
    let off = _mm512_mask_mov_ps(_mm512_set1_ps(parent_off), opposite, diff);
    let real_pivot = _mm512_cmp_ps_mask(pivots, _mm512_set1_ps(f32::INFINITY), _CMP_LT_OQ);
    (off, (!right_mask) | real_pivot)
}

#[cfg(all(
    feature = "simd",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "fma"
))]
#[inline(always)]
unsafe fn select_cyclic_block4_f32<Ops: Avx512F32LeafOps>(
    stems: *const f32,
    block_base_idx: usize,
    query: *const f32,
    parent_off: *const f32,
    pending_mask: u16,
    max_dist: f32,
    prune_equal: bool,
    child_off: *mut f32,
) -> u32 {
    use std::arch::x86_64::*;

    let block = _mm512_loadu_ps(stems.add(block_base_idx));
    let x = _mm512_permutexvar_ps(_mm512_setzero_si512(), block);
    let y = _mm512_permutexvar_ps(
        _mm512_set_epi32(2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1),
        block,
    );
    let z = _mm512_permutexvar_ps(
        _mm512_set_epi32(6, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3),
        block,
    );
    let w = _mm512_permutexvar_ps(
        _mm512_set_epi32(14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8, 7, 7),
        block,
    );

    let (off_x, valid_x) = cyclic_axis_offsets_f32(x, 0xff00, *query, *parent_off);
    let (off_y, valid_y) = cyclic_axis_offsets_f32(y, 0xf0f0, *query.add(1), *parent_off.add(1));
    let (off_z, valid_z) = cyclic_axis_offsets_f32(z, 0xcccc, *query.add(2), *parent_off.add(2));
    let (off_w, valid_w) = cyclic_axis_offsets_f32(w, 0xaaaa, *query.add(3), *parent_off.add(3));
    let rd = Ops::rect_dist_f32x16_4(off_x, off_y, off_z, off_w);
    let threshold = _mm512_set1_ps(max_dist);
    let within = if prune_equal {
        _mm512_cmp_ps_mask(rd, threshold, _CMP_LT_OQ)
    } else {
        _mm512_cmp_ps_mask(rd, threshold, _CMP_LE_OQ)
    };
    let candidates = pending_mask & valid_x & valid_y & valid_z & valid_w & within;
    if candidates == 0 {
        return u32::MAX;
    }

    let child = candidates.trailing_zeros() as u8;
    let lane = _mm512_set1_epi32(i32::from(child));
    *child_off = _mm_cvtss_f32(_mm512_castps512_ps128(_mm512_permutexvar_ps(lane, off_x)));
    *child_off.add(1) = _mm_cvtss_f32(_mm512_castps512_ps128(_mm512_permutexvar_ps(lane, off_y)));
    *child_off.add(2) = _mm_cvtss_f32(_mm512_castps512_ps128(_mm512_permutexvar_ps(lane, off_z)));
    *child_off.add(3) = _mm_cvtss_f32(_mm512_castps512_ps128(_mm512_permutexvar_ps(lane, off_w)));

    let remaining = candidates & candidates.wrapping_sub(1);
    u32::from(child) | (u32::from(remaining) << 8)
}

#[inline(always)]
fn select_cyclic_child<A, O, D, const K: usize, const BH: usize>(
    stems: &[A],
    block_base_idx: usize,
    query: &[A; K],
    parent_off: &[O; 4],
    pending_mask: u16,
    max_dist: O,
    prune_equal: bool,
) -> Option<CyclicChildSelection<O>>
where
    A: Axis<Coord = A>,
    O: Axis<Coord = O>,
    D: DistanceMetric<A, Output = O>,
{
    #[cfg(all(
        feature = "simd",
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "fma"
    ))]
    unsafe {
        // SAFETY: TypeId proves both stored coordinates and widened output have
        // the exact primitive representation expected by the selected kernel.
        // The K/BH guards prove the fixed query and bounds lengths it reads.
        if K == 3
            && BH == 3
            && TypeId::of::<A>() == TypeId::of::<f64>()
            && TypeId::of::<O>() == TypeId::of::<f64>()
            && TypeId::of::<<D as DistanceMetricAvx512<A>>::Avx512F64Ops>()
                != TypeId::of::<UnsupportedAvx512F64LeafOps>()
        {
            let mut child_off = [O::zero(); 4];
            let packed = select_cyclic_block3_f64::<<D as DistanceMetricAvx512<A>>::Avx512F64Ops>(
                stems.as_ptr().cast(),
                block_base_idx,
                query.as_ptr().cast(),
                parent_off.as_ptr().cast(),
                pending_mask,
                *(&max_dist as *const O).cast::<f64>(),
                prune_equal,
                child_off.as_mut_ptr().cast(),
            );
            return (packed != u32::MAX).then_some(CyclicChildSelection {
                child_idx: packed as u8,
                remaining_mask: (packed >> 8) as u16,
                child_off,
            });
        }
        if K == 4
            && BH == 4
            && TypeId::of::<A>() == TypeId::of::<f32>()
            && TypeId::of::<O>() == TypeId::of::<f32>()
            && TypeId::of::<<D as DistanceMetricAvx512<A>>::Avx512F32Ops>()
                != TypeId::of::<UnsupportedAvx512F32LeafOps>()
        {
            let mut child_off = [O::zero(); 4];
            let packed = select_cyclic_block4_f32::<<D as DistanceMetricAvx512<A>>::Avx512F32Ops>(
                stems.as_ptr().cast(),
                block_base_idx,
                query.as_ptr().cast(),
                parent_off.as_ptr().cast(),
                pending_mask,
                *(&max_dist as *const O).cast::<f32>(),
                prune_equal,
                child_off.as_mut_ptr().cast(),
            );
            return (packed != u32::MAX).then_some(CyclicChildSelection {
                child_idx: packed as u8,
                remaining_mask: (packed >> 8) as u16,
                child_off,
            });
        }
    }

    select_cyclic_child_scalar::<A, O, D, K, BH>(
        stems,
        block_base_idx,
        query,
        parent_off,
        pending_mask,
        max_dist,
        prune_equal,
    )
}

/// Cyclic-axis SIMD descent plus native block-level pruning and backtracking.
///
/// For `f64`/3D/Block3 and `f32`/4D/Block4, each terminal block child is a
/// multi-axis rectangle. AVX-512 computes all child rectangle distances in one
/// operation; a pending mask retains viable siblings for later backtracking.
/// Ordinary near descent continues to use the cheaper cyclic block comparator.
/// Other type/dimension combinations retain the scalar continuation engine.
#[derive(Copy, Clone, Debug)]
pub struct DonnellyCyclicSimdFull<const BH: usize> {
    core: DonnellyCore<BH>,
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn cyclic_simd_full_arithmetic_query<
    Tree,
    A,
    T,
    O,
    D,
    QC,
    LS,
    const K: usize,
    const B: usize,
    const BH: usize,
>(
    tree: &Tree,
    query_ctx: &mut QC,
    stack: &mut SimdQueryStack<
        O,
        DonnellyCyclicSimdFull<BH>,
        CYCLIC_SIMD_FULL_INLINE_QUERY_STACK_CAPACITY,
    >,
    mut process_leaf: impl FnMut(usize, &[O; K], &mut QC),
) where
    Tree: KdTreeAccessor<A, T, DonnellyCyclicSimdFull<BH>, LS, K, B>
        + KdTreeQueryOps<A, T, DonnellyCyclicSimdFull<BH>, LS, K, B>,
    A: Axis<Coord = A>,
    T: Content,
    O: Axis<Coord = O>
        + SimdPrune
        + crate::stem_strategy::SimdSelectBestChildBlock3
        + super::simd_full::BacktrackBlock3
        + super::simd_full::BacktrackBlock4,
    D: DistanceMetric<A, Output = O>,
    QC: QueryContext<A, O, K>,
    LS: LeafStrategy<A, T, DonnellyCyclicSimdFull<BH>, K, B>,
    DonnellyCyclicSimdFull<BH>: StemStrategy<
        DeferredState = DonnellyCoreDeferred,
        StackContext<O> = CyclicSimdFullQueryStackContext<O, DonnellyCoreDeferred>,
        Stack<O> = SimdQueryStack<
            O,
            DonnellyCyclicSimdFull<BH>,
            CYCLIC_SIMD_FULL_INLINE_QUERY_STACK_CAPACITY,
        >,
    >,
{
    let native_backend = cfg!(all(
        feature = "simd",
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "fma"
    ));
    let native_f64 = BH == 3
        && K == 3
        && TypeId::of::<A>() == TypeId::of::<f64>()
        && TypeId::of::<O>() == TypeId::of::<f64>();
    let native_f32 = BH == 4
        && K == 4
        && TypeId::of::<A>() == TypeId::of::<f32>()
        && TypeId::of::<O>() == TypeId::of::<f32>();
    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    let native_metric = (native_f64
        && TypeId::of::<<D as DistanceMetricAvx512<A>>::Avx512F64Ops>()
            != TypeId::of::<UnsupportedAvx512F64LeafOps>())
        || (native_f32
            && TypeId::of::<<D as DistanceMetricAvx512<A>>::Avx512F32Ops>()
                != TypeId::of::<UnsupportedAvx512F32LeafOps>());
    #[cfg(not(all(feature = "simd", target_feature = "avx512f")))]
    let native_metric = false;
    if !native_backend || !native_metric || !(native_f64 || native_f32) {
        tree.arithmetic_query_with_scratch_impl::<QC, O, D>(query_ctx, stack, process_leaf);
        return;
    }
    if tree.size() == 0 {
        return;
    }

    let query = *query_ctx.query();
    let mut query_wide = [O::zero(); K];
    let mut dim = 0usize;
    while dim < K {
        query_wide[dim] = D::widen_coord(query[dim]);
        dim += 1;
    }

    let stems = tree.stems();
    let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).expect("non-empty stem storage");
    let root = DonnellyCore::<BH>::new(stems_ptr);
    let root_off = [O::zero(); 4];
    let full_mask = ((1u32 << (1usize << BH)) - 1) as u16;

    stack.clear();

    macro_rules! descend_near_to_leaf {
        ($root:expr, $root_off:expr) => {{
            let mut current = $root;
            let current_off = $root_off;
            loop {
                let near_child =
                    A::compare_cyclic_block::<BH, K>(stems, &query, current.stem_idx(), 0);
                let pending_mask = full_mask & !(1u16 << near_child);
                unsafe {
                    stack.push_inline_unchecked(CyclicSimdFullQueryStackContext::pending(
                        current.deferred_state(),
                        pending_mask,
                        current_off,
                    ));
                }
                current.traverse_block::<K>(near_child, BH as u32);
                if current.level() > tree.max_stem_level() {
                    break current.leaf_idx();
                }
            }
        }};
    }

    let first_leaf = descend_near_to_leaf!(root, root_off);
    process_leaf(first_leaf, &query_wide, query_ctx);

    while let Some(frame) = unsafe { stack.pop_inline_unchecked() } {
        let (base_state, pending_mask, parent_off) = unsafe { frame.into_pending_unchecked() };
        let mut base = DonnellyCore::<BH>::new(stems_ptr);
        base.rehydrate_deferred_state(base_state);
        let max_dist = query_ctx.max_dist();
        let selection = select_cyclic_child::<A, O, D, K, BH>(
            stems,
            base.stem_idx(),
            &query,
            &parent_off,
            pending_mask,
            max_dist,
            query_ctx.prune_on_equal_max_dist(),
        );
        let Some(selection) = selection else {
            continue;
        };
        if selection.remaining_mask != 0 {
            unsafe {
                stack.push_inline_unchecked(CyclicSimdFullQueryStackContext::pending(
                    base_state,
                    selection.remaining_mask,
                    parent_off,
                ));
            }
        }

        base.traverse_block::<K>(selection.child_idx, BH as u32);
        if base.level() > tree.max_stem_level() {
            process_leaf(base.leaf_idx(), &query_wide, query_ctx);
        } else {
            let leaf_idx = descend_near_to_leaf!(base, selection.child_off);
            process_leaf(leaf_idx, &query_wide, query_ctx);
        }
    }
}

macro_rules! impl_cyclic_simd_full {
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
            type StackContext<A> = $stack_context;
            type Stack<A> = $stack_type;

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
                $stack: &mut Self::Stack<O>,
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
                Self::Stack<O>: crate::kd_tree::query_stack::StackTrait<O, Self>,
            $arithmetic
        }
    };
}

impl_cyclic_simd_full!(
    DonnellyCyclicSimdFull,
    3,
    1,
    CyclicSimdFullQueryStackContext<A, Self::DeferredState>,
    SimdQueryStack<A, Self, CYCLIC_SIMD_FULL_INLINE_QUERY_STACK_CAPACITY>,
    tree,
    query_ctx,
    stack,
    process_leaf,
    {
        cyclic_simd_full_arithmetic_query::<Tree, A, T, O, D, QC, LS, K2, B, 3>(
            tree,
            query_ctx,
            stack,
            process_leaf,
        );
    }
);
impl_cyclic_simd_full!(
    DonnellyCyclicSimdFull,
    4,
    7,
    CyclicSimdFullQueryStackContext<A, Self::DeferredState>,
    SimdQueryStack<A, Self, CYCLIC_SIMD_FULL_INLINE_QUERY_STACK_CAPACITY>,
    tree,
    query_ctx,
    stack,
    process_leaf,
    {
        cyclic_simd_full_arithmetic_query::<Tree, A, T, O, D, QC, LS, K2, B, 4>(
            tree,
            query_ctx,
            stack,
            process_leaf,
        );
    }
);

#[cfg(feature = "cargo_asm")]
mod cargo_asm {
    use super::select_cyclic_child;
    use crate::SquaredEuclidean;

    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn donnelly_cyclic_full_block3_f64_cargo_asm_hook(
        stems: &[f64],
        query: &[f64; 3],
        block_base: usize,
        off: &[f64; 4],
        max_dist: f64,
        child_off: &mut [f64; 4],
    ) -> u32 {
        let Some(selection) = select_cyclic_child::<f64, f64, SquaredEuclidean<f64>, 3, 3>(
            stems, block_base, query, off, 0xff, max_dist, true,
        ) else {
            return u32::MAX;
        };
        *child_off = selection.child_off;
        u32::from(selection.child_idx) | (u32::from(selection.remaining_mask) << 8)
    }

    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn donnelly_cyclic_full_block4_f32_cargo_asm_hook(
        stems: &[f32],
        query: &[f32; 4],
        block_base: usize,
        off: &[f32; 4],
        max_dist: f32,
        child_off: &mut [f32; 4],
    ) -> u32 {
        let Some(selection) = select_cyclic_child::<f32, f32, SquaredEuclidean<f32>, 4, 4>(
            stems, block_base, query, off, 0xffff, max_dist, true,
        ) else {
            return u32::MAX;
        };
        *child_off = selection.child_off;
        u32::from(selection.child_idx) | (u32::from(selection.remaining_mask) << 8)
    }
}

#[cfg(all(
    test,
    feature = "simd",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "fma"
))]
mod tests {
    use super::*;

    #[test]
    fn native_f64_block3_selection_kernel_matches_scalar_reference() {
        let stems = [
            0.5,
            0.25,
            f64::INFINITY,
            0.1,
            0.4,
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
        ];
        let query = [0.32, 0.61, 0.18];
        let off = [0.11, 0.07, 0.13, 0.0];
        let actual = select_cyclic_child::<f64, f64, crate::SquaredEuclidean<f64>, 3, 3>(
            &stems,
            0,
            &query,
            &off,
            0xff,
            f64::INFINITY,
            true,
        )
        .unwrap();
        let expected = select_cyclic_child_scalar::<f64, f64, crate::SquaredEuclidean<f64>, 3, 3>(
            &stems,
            0,
            &query,
            &off,
            0xff,
            f64::INFINITY,
            true,
        )
        .unwrap();

        let expected_candidates = expected.remaining_mask | (1u16 << expected.child_idx);
        assert_eq!(actual.child_idx, expected_candidates.trailing_zeros() as u8);
        assert_eq!(
            actual.remaining_mask,
            expected_candidates & !(1u16 << actual.child_idx)
        );
        let expected_selected = selected_cyclic_child_offsets::<
            f64,
            f64,
            crate::SquaredEuclidean<f64>,
            3,
            3,
        >(&stems, 0, &query, &off, actual.child_idx)
        .unwrap();
        for dim in 0..4 {
            assert!((actual.child_off[dim] - expected_selected.0[dim]).abs() <= 1.0e-14);
        }
    }

    #[test]
    fn native_f32_block4_selection_kernel_matches_scalar_reference() {
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
        let query = [0.32, 0.61, 0.18, 0.73];
        let off = [0.11, 0.07, 0.13, 0.09];
        let actual = select_cyclic_child::<f32, f32, crate::SquaredEuclidean<f32>, 4, 4>(
            &stems,
            0,
            &query,
            &off,
            0xffff,
            f32::INFINITY,
            true,
        )
        .unwrap();
        let expected = select_cyclic_child_scalar::<f32, f32, crate::SquaredEuclidean<f32>, 4, 4>(
            &stems,
            0,
            &query,
            &off,
            0xffff,
            f32::INFINITY,
            true,
        )
        .unwrap();

        let expected_candidates = expected.remaining_mask | (1u16 << expected.child_idx);
        assert_eq!(actual.child_idx, expected_candidates.trailing_zeros() as u8);
        assert_eq!(
            actual.remaining_mask,
            expected_candidates & !(1u16 << actual.child_idx)
        );
        let expected_selected = selected_cyclic_child_offsets::<
            f32,
            f32,
            crate::SquaredEuclidean<f32>,
            4,
            4,
        >(&stems, 0, &query, &off, actual.child_idx)
        .unwrap();
        for dim in 0..4 {
            assert!((actual.child_off[dim] - expected_selected.0[dim]).abs() <= 1.0e-6);
        }
    }
}

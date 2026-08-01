//! Unrolled Donnelly stem strategy with block-sized dimension scheduling.

use std::ptr::NonNull;

use crate::stem_strategy::donnelly::core::{DonnellyCore, DonnellyCoreDeferred};
use crate::{Axis, StemStrategy};

/// Unrolled Donnelly strategy using block-sized dimension scheduling.
#[derive(Copy, Clone, Debug)]
pub struct DonnellyUnrolledBlockDim<const BH: usize> {
    core: DonnellyCore<BH>,
}

macro_rules! impl_donnelly_unrolled_block_dim_strategy {
    ($size:tt) => {
        impl StemStrategy for DonnellyUnrolledBlockDim<$size> {
            const ROOT_IDX: usize = 0;
            const BLOCK_SIZE: usize = $size;
            const SUPPORTS_ARITHMETIC_LEAF_RESOLUTION: bool = true;
            const USES_UNROLLED_SCALAR_TRAVERSAL: bool = true;

            type DeferredState = DonnellyCoreDeferred;
            type StackContext<A> =
                crate::kd_tree::query_stack::QueryStackContext<A, Self::DeferredState>;
            type Stack<A> = crate::kd_tree::query_stack::QueryStack<A, Self>;

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
                self.core.level() as usize / $size % K
            }

            #[inline(always)]
            fn construction_dim<const K: usize>(&self) -> usize {
                self.core.level() as usize / $size % K
            }

            #[inline(always)]
            fn level(&self) -> i32 {
                self.core.level()
            }

            #[inline(always)]
            fn traverse<A: Axis<Coord = A>, const K: usize>(&mut self, is_right: bool) {
                self.core.traverse::<A, K>(is_right)
            }

            #[inline(always)]
            fn traverse_head<A: Axis<Coord = A>, const K: usize>(&mut self, is_right: bool) {
                self.core.traverse_head::<A, K>(is_right)
            }

            #[inline(always)]
            fn traverse_tail<A: Axis<Coord = A>, const K: usize>(&mut self, is_right: bool) {
                self.core
                    .traverse_tail_with_block_size::<A, K>(is_right, $size as u32)
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

            fn get_stem_node_count_from_leaf_node_count(_leaf_node_count: usize) -> usize {
                unimplemented!()
            }

            fn stem_node_padding_factor() -> usize {
                unimplemented!()
            }

            fn get_leaf_idx<A: Axis<Coord = A>, const K2: usize>(
                stems: &[A],
                query: &[A; K2],
                max_stem_level: i32,
            ) -> usize {
                if $size == 3 {
                    return crate::stem_strategy::donnelly::core::get_leaf_idx_block3_block_dim(
                        stems,
                        query,
                        max_stem_level,
                    );
                }
                if $size == 4 {
                    return crate::stem_strategy::donnelly::core::get_leaf_idx_block4_block_dim(
                        stems,
                        query,
                        max_stem_level,
                    );
                }

                let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
                let mut strat = Self::new(stems_ptr);

                while strat.level() + ($size as i32) <= max_stem_level + 1 {
                    impl_donnelly_unrolled_block_dim_strategy!(@unroll $size, stems, query, strat);
                }

                while strat.level() <= max_stem_level {
                    let pivot = unsafe { stems.get_unchecked(strat.stem_idx()) };
                    let is_right = unsafe { *query.get_unchecked(strat.dim::<K2>()) } >= *pivot;
                    strat.traverse::<A, K2>(is_right);
                }

                strat.leaf_idx()
            }
        }
    };

    (@unroll 3, $stems:expr, $query:expr, $strat:expr) => {
        impl_donnelly_unrolled_block_dim_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_block_dim_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_block_dim_strategy!(@tail $stems, $query, $strat);
    };
    (@unroll 4, $stems:expr, $query:expr, $strat:expr) => {
        impl_donnelly_unrolled_block_dim_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_block_dim_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_block_dim_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_block_dim_strategy!(@tail $stems, $query, $strat);
    };
    (@unroll 5, $stems:expr, $query:expr, $strat:expr) => {
        impl_donnelly_unrolled_block_dim_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_block_dim_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_block_dim_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_block_dim_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_block_dim_strategy!(@tail $stems, $query, $strat);
    };
    (@unroll 6, $stems:expr, $query:expr, $strat:expr) => {
        impl_donnelly_unrolled_block_dim_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_block_dim_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_block_dim_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_block_dim_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_block_dim_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_block_dim_strategy!(@tail $stems, $query, $strat);
    };
    (@unroll 7, $stems:expr, $query:expr, $strat:expr) => {
        impl_donnelly_unrolled_block_dim_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_block_dim_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_block_dim_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_block_dim_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_block_dim_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_block_dim_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_block_dim_strategy!(@tail $stems, $query, $strat);
    };

    (@head $stems:expr, $query:expr, $strat:expr) => {{
        let pivot = unsafe { $stems.get_unchecked($strat.stem_idx()) };
        let is_right = unsafe { *$query.get_unchecked($strat.dim::<K2>()) } >= *pivot;
        $strat.traverse_head::<A, K2>(is_right);
    }};
    (@tail $stems:expr, $query:expr, $strat:expr) => {{
        let pivot = unsafe { $stems.get_unchecked($strat.stem_idx()) };
        let is_right = unsafe { *$query.get_unchecked($strat.dim::<K2>()) } >= *pivot;
        $strat.traverse_tail::<A, K2>(is_right);
    }};
}

impl_donnelly_unrolled_block_dim_strategy!(3);
impl_donnelly_unrolled_block_dim_strategy!(4);
impl_donnelly_unrolled_block_dim_strategy!(5);
impl_donnelly_unrolled_block_dim_strategy!(6);
impl_donnelly_unrolled_block_dim_strategy!(7);

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    type Block3Scalar = DonnellyUnrolledBlockDim<3>;

    #[test]
    fn donnelly_marker_scalar_block3_basics_and_state_round_trip() {
        let mut stems = [f64::INFINITY; 256];
        let stems_ptr = NonNull::new(stems.as_mut_ptr() as *mut u8).unwrap();

        let mut strat = Block3Scalar::new(stems_ptr);
        assert_eq!(strat.stem_idx(), 0);
        assert_eq!(strat.leaf_idx(), 0);
        assert_eq!(strat.dim::<3>(), 0);
        assert_eq!(strat.construction_dim::<3>(), 0);
        assert_eq!(strat.level(), 0);
        assert_eq!(Block3Scalar::block_size(), 3);

        let saved = strat.deferred_state();
        strat.traverse::<f64, 3>(true);
        strat.traverse::<f64, 3>(false);
        assert_ne!(strat.stem_idx(), 0);
        assert_ne!(strat.leaf_idx(), 0);

        strat.rehydrate_deferred_state(saved);
        assert_eq!(strat.stem_idx(), 0);
        assert_eq!(strat.leaf_idx(), 0);
        assert_eq!(strat.dim::<3>(), 0);
        assert_eq!(strat.construction_dim::<3>(), 0);
        assert_eq!(strat.level(), 0);
    }

    #[test]
    fn donnelly_marker_scalar_block3_traversal_branch_and_children_match_core() {
        let mut stems = [0.0f64; 256];
        let stems_ptr = NonNull::new(stems.as_mut_ptr() as *mut u8).unwrap();

        let mut wrapper = Block3Scalar::new(stems_ptr);
        let mut core = DonnellyCore::<3>::new(stems_ptr);

        wrapper.traverse_head::<f64, 3>(true);
        core.traverse_head::<f64, 3>(true);
        assert_eq!(wrapper.stem_idx(), core.stem_idx());
        assert_eq!(wrapper.leaf_idx(), core.leaf_idx());
        assert_eq!(wrapper.level(), core.level());
        assert_eq!(wrapper.dim::<3>(), core.level() as usize / 3 % 3);
        assert_eq!(
            wrapper.construction_dim::<3>(),
            core.level() as usize / 3 % 3
        );

        wrapper.traverse_tail::<f64, 3>(false);
        core.traverse_tail_with_block_size::<f64, 3>(false, 3);
        assert_eq!(wrapper.stem_idx(), core.stem_idx());
        assert_eq!(wrapper.leaf_idx(), core.leaf_idx());
        assert_eq!(wrapper.level(), core.level());
        assert_eq!(wrapper.child_indices::<f64>(), core.child_indices::<f64>());

        let mut branched_wrapper = Block3Scalar::new(stems_ptr);
        let mut branched_core = DonnellyCore::<3>::new(stems_ptr);
        let right_wrapper = branched_wrapper.branch::<f64, 3>();
        let right_core = branched_core.branch::<f64, 3>();

        assert_eq!(branched_wrapper.stem_idx(), branched_core.stem_idx());
        assert_eq!(branched_wrapper.leaf_idx(), branched_core.leaf_idx());
        assert_eq!(right_wrapper.stem_idx(), right_core.stem_idx());
        assert_eq!(right_wrapper.leaf_idx(), right_core.leaf_idx());
        assert_eq!(right_wrapper.level(), right_core.level());
        assert_eq!(
            right_wrapper.dim::<3>(),
            right_core.level() as usize / 3 % 3
        );
    }

    #[test]
    fn donnelly_marker_scalar_block3_get_leaf_idx_uses_unrolled_and_scalar_paths() {
        let stems = [0.0f64; 256];
        let query = [1.0f64, 1.0, 1.0];

        let leaf_idx = Block3Scalar::get_leaf_idx(&stems, &query, 4);

        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        let mut manual = Block3Scalar::new(stems_ptr);
        while manual.level() <= 4 {
            manual.traverse::<f64, 3>(true);
        }

        assert_eq!(leaf_idx, manual.leaf_idx());
        assert_eq!(leaf_idx, 31);
    }

    #[test]
    fn donnelly_marker_scalar_block3_unimplemented_metadata_helpers_panic() {
        assert!(catch_unwind(AssertUnwindSafe(|| {
            let _ = Block3Scalar::get_stem_node_count_from_leaf_node_count(8);
        }))
        .is_err());

        assert!(catch_unwind(AssertUnwindSafe(|| {
            let _ = Block3Scalar::stem_node_padding_factor();
        }))
        .is_err());
    }
}

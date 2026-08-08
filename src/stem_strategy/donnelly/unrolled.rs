//! Unrolled Donnelly stem strategy.

use std::ptr::NonNull;

use crate::stem_strategy::donnelly::core::{DonnellyCore, DonnellyCoreDeferred};
use crate::{Axis, StemStrategy};

/// Unrolled Donnelly strategy with per-level dimension scheduling.
#[derive(Copy, Clone, Debug)]
pub struct DonnellyUnrolled<const BH: usize> {
    core: DonnellyCore<BH>,
}

macro_rules! impl_donnelly_unrolled_strategy {
    ($size:tt) => {
        impl StemStrategy for DonnellyUnrolled<$size> {
            const ROOT_IDX: usize = 0;
            const BLOCK_SIZE: usize = $size;
            const SUPPORTS_ARITHMETIC_LEAF_RESOLUTION: bool = true;
            const USES_UNROLLED_SCALAR_TRAVERSAL: bool = true;

            type DeferredState = DonnellyCoreDeferred;
            type StackContext<A, const K: usize> =
                crate::kd_tree::query_stack::QueryStackContext<A, Self::DeferredState>;
            type Stack<A, const K: usize> = crate::kd_tree::query_stack::QueryStack<A, Self, K>;

            #[inline(always)]
            fn new(stems_ptr: NonNull<u8>) -> Self {
                Self {
                    core: DonnellyCore::new(stems_ptr),
                }
            }

            #[inline(always)]
            fn stem_idx(&self) -> usize { self.core.stem_idx() }

            #[inline(always)]
            fn deferred_state(&self) -> Self::DeferredState { self.core.deferred_state() }

            #[inline(always)]
            fn rehydrate_deferred_state(&mut self, state: Self::DeferredState) {
                self.core.rehydrate_deferred_state(state);
            }

            #[inline(always)]
            fn leaf_idx(&self) -> usize { self.core.leaf_idx() }

            #[inline(always)]
            fn dim<const K: usize>(&self) -> usize { self.core.dim::<K>() }

            #[inline(always)]
            fn level(&self) -> i32 { self.core.level() }

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
                impl_donnelly_unrolled_strategy!(
                    @get_leaf $size,
                    stems,
                    query,
                    max_stem_level
                )
            }
        }
    };

    (@get_leaf 3, $stems:expr, $query:expr, $max_stem_level:expr) => {{
        crate::stem_strategy::donnelly::core::get_leaf_idx_block3::<A, K2>(
            $stems,
            $query,
            $max_stem_level,
        )
    }};

    (@get_leaf 4, $stems:expr, $query:expr, $max_stem_level:expr) => {{
        crate::stem_strategy::donnelly::core::get_leaf_idx_block4::<A, K2>(
            $stems,
            $query,
            $max_stem_level,
        )
    }};

    (@get_leaf $size:tt, $stems:expr, $query:expr, $max_stem_level:expr) => {{
                let stems_ptr = NonNull::new($stems.as_ptr() as *mut u8).unwrap();
                let mut strat = Self::new(stems_ptr);

                while strat.level() + ($size as i32) <= $max_stem_level + 1 {
                    impl_donnelly_unrolled_strategy!(@unroll $size, $stems, $query, strat);
                }

                while strat.level() <= $max_stem_level {
                    let pivot = unsafe { $stems.get_unchecked(strat.stem_idx()) };
                    let is_right = unsafe { *$query.get_unchecked(strat.dim::<K2>()) } >= *pivot;
                    strat.traverse::<A, K2>(is_right);
                }

                strat.leaf_idx()
    }};

    (@unroll 3, $stems:expr, $query:expr, $strat:expr) => {
        impl_donnelly_unrolled_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_strategy!(@tail $stems, $query, $strat);
    };
    (@unroll 4, $stems:expr, $query:expr, $strat:expr) => {
        impl_donnelly_unrolled_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_strategy!(@tail $stems, $query, $strat);
    };
    (@unroll 5, $stems:expr, $query:expr, $strat:expr) => {
        impl_donnelly_unrolled_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_strategy!(@tail $stems, $query, $strat);
    };
    (@unroll 6, $stems:expr, $query:expr, $strat:expr) => {
        impl_donnelly_unrolled_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_strategy!(@tail $stems, $query, $strat);
    };
    (@unroll 7, $stems:expr, $query:expr, $strat:expr) => {
        impl_donnelly_unrolled_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_strategy!(@head $stems, $query, $strat);
        impl_donnelly_unrolled_strategy!(@tail $stems, $query, $strat);
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

impl_donnelly_unrolled_strategy!(3);
impl_donnelly_unrolled_strategy!(4);
impl_donnelly_unrolled_strategy!(5);
impl_donnelly_unrolled_strategy!(6);
impl_donnelly_unrolled_strategy!(7);

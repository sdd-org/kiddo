use std::sync::OnceLock;

use aligned_vec::{AVec, ConstAlign, CACHELINE_ALIGN};

use crate::kd_tree::ConstructionError;
use crate::traits::leaf_strategy::ConstructibleLeafStrategy;
use crate::{Axis, Content, KdTree, StemStrategy};

use super::shared::{ConstructionIndex, ConstructionLeafScratch, SoftConstructionMode};
use super::SerialConstruction;

/// Type-state configuration for parallel construction.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct ParallelConstruction {
    threshold: usize,
}

impl ParallelConstruction {
    #[inline(always)]
    pub(in crate::kd_tree) const fn with_threshold(threshold: usize) -> Self {
        Self {
            threshold: if threshold == 0 { 1 } else { threshold },
        }
    }

    #[cfg(test)]
    pub(in crate::kd_tree) const fn threshold(self) -> usize {
        self.threshold
    }

    #[inline(always)]
    pub(super) const fn should_parallelize(self, item_count: usize) -> bool {
        item_count >= self.threshold
    }
}

impl<A, T, SS, LS, I, X, FA, FI, const K: usize, const B: usize>
    SoftConstructionMode<A, T, SS, LS, I, X, FA, FI, K, B> for ParallelConstruction
where
    A: Axis<Coord = A> + Send + Sync,
    T: Content,
    SS: StemStrategy,
    LS: ConstructibleLeafStrategy<A, T, SS, K, B>,
    I: ConstructionIndex,
    X: Sync,
    FA: Fn(&X, usize) -> A + Sync,
    FI: FnMut(usize, &X) -> Result<T, ConstructionError>,
{
    fn populate(
        &self,
        stems: &mut AVec<A, ConstAlign<{ CACHELINE_ALIGN }>>,
        source: &[X],
        axis_at: &FA,
        sort_index: &mut [I],
        root_stem_ordering: SS,
        max_stem_level: i32,
        leaf_budget: usize,
        leaves: &mut LS,
        actual_max_stem_level: &mut i32,
        max_leaf_len: &mut usize,
        leaf_scratch: &mut ConstructionLeafScratch<A, T, K>,
        item_at: &mut FI,
    ) -> Result<(), ConstructionError> {
        if !self.should_parallelize(sort_index.len()) {
            return <SerialConstruction as SoftConstructionMode<
                A,
                T,
                SS,
                LS,
                I,
                X,
                FA,
                FI,
                K,
                B,
            >>::populate(
                &SerialConstruction,
                stems,
                source,
                axis_at,
                sort_index,
                root_stem_ordering,
                max_stem_level,
                leaf_budget,
                leaves,
                actual_max_stem_level,
                max_leaf_len,
                leaf_scratch,
                item_at,
            );
        }

        KdTree::<A, T, SS, LS, K, B>::populate_parallel_soft(
            stems,
            source,
            axis_at,
            sort_index,
            root_stem_ordering,
            max_stem_level,
            leaf_budget,
            leaves,
            actual_max_stem_level,
            max_leaf_len,
            leaf_scratch,
            item_at,
            self.threshold,
        )
    }
}

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: Axis<Coord = A>,
    T: Content,
    SS: StemStrategy,
    LS: ConstructibleLeafStrategy<A, T, SS, K, B>,
{
    #[allow(clippy::too_many_arguments)]
    fn populate_parallel_soft<I, X, FA, FI>(
        stems: &mut AVec<A, ConstAlign<{ CACHELINE_ALIGN }>>,
        source: &[X],
        axis_at: &FA,
        sort_index: &mut [I],
        root_stem_ordering: SS,
        max_stem_level: i32,
        leaf_budget: usize,
        leaves: &mut LS,
        actual_max_stem_level: &mut i32,
        max_leaf_len: &mut usize,
        leaf_scratch: &mut ConstructionLeafScratch<A, T, K>,
        item_at: &mut FI,
        parallel_threshold: usize,
    ) -> Result<(), ConstructionError>
    where
        A: Send + Sync,
        I: ConstructionIndex,
        X: Sync,
        FA: Fn(&X, usize) -> A + Sync,
        FI: FnMut(usize, &X) -> Result<T, ConstructionError>,
    {
        let mut leaf_ranges = vec![(0usize, 0usize); leaf_budget];
        let parallel_stems = (0..stems.len())
            .map(|_| OnceLock::new())
            .collect::<Vec<_>>();
        *max_leaf_len = Self::partition_recursive_soft_parallel(
            &parallel_stems,
            source,
            axis_at,
            sort_index,
            0,
            root_stem_ordering,
            max_stem_level,
            leaf_budget,
            &mut leaf_ranges,
            parallel_threshold,
        )?;
        for (stem_idx, parallel_stem) in parallel_stems.into_iter().enumerate() {
            if let Some(value) = parallel_stem.into_inner() {
                stems[stem_idx] = value;
            }
        }
        *actual_max_stem_level = (*actual_max_stem_level).max(max_stem_level);

        for (start, len) in leaf_ranges {
            Self::write_leaf_from_sort_index(
                source,
                axis_at,
                &sort_index[start..start + len],
                leaves,
                max_leaf_len,
                leaf_scratch,
                item_at,
            )?;
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn partition_recursive_soft_parallel<I, X, FA>(
        parallel_stems: &[OnceLock<A>],
        source: &[X],
        axis_at: &FA,
        sort_index: &mut [I],
        sort_offset: usize,
        mut stem_ordering: SS,
        max_stem_level: i32,
        leaf_budget: usize,
        leaf_ranges: &mut [(usize, usize)],
        parallel_threshold: usize,
    ) -> Result<usize, ConstructionError>
    where
        A: Send + Sync,
        I: ConstructionIndex,
        X: Sync,
        FA: Fn(&X, usize) -> A + Sync,
    {
        if leaf_budget == 0 {
            return Ok(0);
        }
        if stem_ordering.level() > max_stem_level {
            debug_assert_eq!(leaf_ranges.len(), 1);
            leaf_ranges[0] = (sort_offset, sort_index.len());
            return Ok(sort_index.len());
        }

        let chunk_length = sort_index.len();
        let dim = stem_ordering.construction_dim::<K>();
        let stem_index = stem_ordering.stem_idx();

        let (left_leaf_budget, right_leaf_budget, pivot) = if leaf_budget == 1 {
            (1usize, 0usize, chunk_length)
        } else {
            let left_leaf_budget = Self::soft_left_leaf_budget(leaf_budget);
            let right_leaf_budget = leaf_budget - left_leaf_budget;
            let mut pivot = Self::soft_ideal_pivot(chunk_length, left_leaf_budget, leaf_budget);
            if pivot < chunk_length {
                pivot = Self::update_pivot(source, axis_at, sort_index, dim, pivot)?;
            }
            (left_leaf_budget, right_leaf_budget, pivot)
        };

        if pivot < chunk_length {
            let pivot_value = axis_at(&source[sort_index[pivot].as_usize()], dim);
            assert!(
                parallel_stems[stem_index].set(pivot_value).is_ok(),
                "parallel construction wrote stem {stem_index} more than once"
            );
        }

        let right_stem_ordering = stem_ordering.branch::<A, K>();
        let (lower_sort_index, upper_sort_index) = sort_index.split_at_mut(pivot);
        let (lower_leaf_ranges, upper_leaf_ranges) = leaf_ranges.split_at_mut(left_leaf_budget);

        let recurse_left = || {
            Self::partition_recursive_soft_parallel(
                parallel_stems,
                source,
                axis_at,
                lower_sort_index,
                sort_offset,
                stem_ordering,
                max_stem_level,
                left_leaf_budget,
                lower_leaf_ranges,
                parallel_threshold,
            )
        };
        let recurse_right = || {
            Self::partition_recursive_soft_parallel(
                parallel_stems,
                source,
                axis_at,
                upper_sort_index,
                sort_offset + pivot,
                right_stem_ordering,
                max_stem_level,
                right_leaf_budget,
                upper_leaf_ranges,
                parallel_threshold,
            )
        };

        let (left_max_leaf, right_max_leaf) = if chunk_length >= parallel_threshold
            && left_leaf_budget > 0
            && right_leaf_budget > 0
        {
            rayon::join(recurse_left, recurse_right)
        } else {
            (recurse_left(), recurse_right())
        };

        Ok(left_max_leaf?.max(right_max_leaf?))
    }
}

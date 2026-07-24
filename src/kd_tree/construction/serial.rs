use aligned_vec::{AVec, ConstAlign, CACHELINE_ALIGN};

use crate::kd_tree::ConstructionError;
use crate::traits::leaf_strategy::ConstructibleLeafStrategy;
use crate::{Axis, Content, KdTree, StemStrategy};

use super::shared::{ConstructionIndex, ConstructionLeafScratch, SoftConstructionMode};

/// Type-state marker for serial construction.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SerialConstruction;

impl<A, T, SS, LS, I, X, FA, FI, const K: usize, const B: usize>
    SoftConstructionMode<A, T, SS, LS, I, X, FA, FI, K, B> for SerialConstruction
where
    A: Axis<Coord = A>,
    T: Content,
    SS: StemStrategy,
    LS: ConstructibleLeafStrategy<A, T, SS, K, B>,
    I: ConstructionIndex,
    FA: Fn(&X, usize) -> A,
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
        KdTree::<A, T, SS, LS, K, B>::populate_recursive_soft(
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
    /// Soft-bucket recursive construction helper preserving arithmetic layout.
    #[allow(clippy::too_many_arguments)]
    fn populate_recursive_soft<I, X, FA, FI>(
        stems: &mut AVec<A, ConstAlign<{ CACHELINE_ALIGN }>>,
        source: &[X],
        axis_at: &FA,
        sort_index: &mut [I],
        mut stem_ordering: SS,
        max_stem_level: i32,
        leaf_budget: usize,
        leaves: &mut LS,
        actual_max_stem_level: &mut i32,
        max_leaf_len: &mut usize,
        leaf_scratch: &mut ConstructionLeafScratch<A, T, K>,
        item_at: &mut FI,
    ) -> Result<(), ConstructionError>
    where
        I: ConstructionIndex,
        FA: Fn(&X, usize) -> A,
        FI: FnMut(usize, &X) -> Result<T, ConstructionError>,
    {
        if leaf_budget == 0 {
            return Ok(());
        }

        if stem_ordering.level() > max_stem_level {
            Self::write_leaf_from_sort_index(
                source,
                axis_at,
                sort_index,
                leaves,
                max_leaf_len,
                leaf_scratch,
                item_at,
            )?;
            return Ok(());
        }

        let chunk_length = sort_index.len();
        let dim = stem_ordering.construction_dim::<K>();
        let stem_index = stem_ordering.stem_idx();
        *actual_max_stem_level = (*actual_max_stem_level).max(stem_ordering.level());

        if stem_index >= stems.len() {
            tracing::warn!(
                %stem_index,
                existing_stem_vec_len = %stems.len(),
                "encountered a stem index beyond the end of the stem vec. Growing the vec to fit"
            );
            stems.resize(stem_index + 1, A::max_value());
        }

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
            debug_assert!(
                A::Coord::is_max_value(stems[stem_index]),
                "Wrote to stem #{stem_index:?} for a second time",
            );
            stems[stem_index] = axis_at(&source[sort_index[pivot].as_usize()], dim);
        }

        let right_stem_ordering = stem_ordering.branch::<A, K>();
        let split_idx = pivot.min(chunk_length);
        let (lower_sort_index, upper_sort_index) = sort_index.split_at_mut(split_idx);

        Self::populate_recursive_soft(
            stems,
            source,
            axis_at,
            lower_sort_index,
            stem_ordering,
            max_stem_level,
            left_leaf_budget,
            leaves,
            actual_max_stem_level,
            max_leaf_len,
            leaf_scratch,
            item_at,
        )?;

        Self::populate_recursive_soft(
            stems,
            source,
            axis_at,
            upper_sort_index,
            right_stem_ordering,
            max_stem_level,
            right_leaf_budget,
            leaves,
            actual_max_stem_level,
            max_leaf_len,
            leaf_scratch,
            item_at,
        )?;

        Ok(())
    }
}

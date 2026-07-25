//! Benchmark-only embedded leaf descriptor experiment.

use std::ptr::NonNull;

use aligned_vec::{AVec, CACHELINE_ALIGN};

use crate::dist::DistanceMetric;
use crate::kd_tree::{ConstructionError, KdTree};
use crate::leaf_strategy::VecOfArenas;
use crate::stem_strategy::DonnellyUnrolledLeafEmbedded3;
use crate::{LeafStrategy, StemStrategy};

const DESCRIPTOR_OFFSET_BITS: u32 = 48;
const DESCRIPTOR_OFFSET_MASK: u64 = (1u64 << DESCRIPTOR_OFFSET_BITS) - 1;
const FINAL_PIVOT_LEVELS: i32 = 2;

type ConstructionTree<const K: usize, const B: usize> =
    KdTree<f64, u32, DonnellyUnrolledLeafEmbedded3, VecOfArenas<f64, u32, K, B>, K, B>;

/// Construction failure for the benchmark-only 48-bit-offset/16-bit-length layout.
#[doc(hidden)]
#[derive(Debug)]
pub enum EmbeddedLeafDescriptorError {
    /// The ordinary kd-tree construction failed.
    Construction(ConstructionError),
    /// A byte offset did not fit in the descriptor's low 48 bits.
    ByteOffsetOverflow { leaf_idx: usize, byte_offset: usize },
    /// A leaf length did not fit in the descriptor's high 16 bits.
    LeafLengthOverflow { leaf_idx: usize, leaf_len: usize },
}

impl From<ConstructionError> for EmbeddedLeafDescriptorError {
    fn from(value: ConstructionError) -> Self {
        Self::Construction(value)
    }
}

/// An aligned integer-word Donnelly tree used only by the descriptor benchmark.
///
/// Pivot words contain `f64::to_bits()` values. The last level of every final
/// block contains a packed 48-bit arena byte offset and 16-bit leaf length.
#[doc(hidden)]
pub struct EmbeddedLeafDescriptorF64Tree<const K: usize, const B: usize> {
    stem_words: AVec<u64>,
    leaves: VecOfArenas<f64, u32, K, B>,
    max_stem_level: i32,
    leaf_count: usize,
}

impl<const K: usize, const B: usize> EmbeddedLeafDescriptorF64Tree<K, B> {
    /// Builds the benchmark tree while retaining the ordinary leaf extent table.
    pub fn new_from_slice(source: &[[f64; K]]) -> Result<Self, EmbeddedLeafDescriptorError> {
        let tree: ConstructionTree<K, B> = KdTree::new_from_slice(source)?;
        let KdTree {
            stems,
            leaves,
            max_stem_level,
            ..
        } = tree;

        let leaf_count = <VecOfArenas<f64, u32, K, B> as LeafStrategy<
            f64,
            u32,
            DonnellyUnrolledLeafEmbedded3,
            K,
            B,
        >>::leaf_count(&leaves);

        let mut stem_words = AVec::new(CACHELINE_ALIGN);
        stem_words.resize(stems.len(), 0);
        for (word, pivot) in stem_words.iter_mut().zip(stems.iter()) {
            *word = pivot.to_bits();
        }

        let pivot_depth = (max_stem_level + 1) as usize;
        debug_assert_eq!((pivot_depth + 1) % 3, 0);

        for leaf_idx in 0..leaf_count {
            let (byte_offset, leaf_len) = leaves.leaf_extent_for_embedded_descriptor(leaf_idx);
            let descriptor = pack_descriptor(leaf_idx, byte_offset, leaf_len)?;
            let terminal_stem_idx = terminal_stem_idx::<K>(leaf_idx, pivot_depth);
            debug_assert!(terminal_stem_idx < stem_words.len());
            unsafe {
                *stem_words.get_unchecked_mut(terminal_stem_idx) = descriptor;
            }
        }

        Ok(Self {
            stem_words,
            leaves,
            max_stem_level,
            leaf_count,
        })
    }

    /// Approximate nearest-one using the existing extent indirection.
    #[inline(always)]
    pub fn approx_nearest_one_via_extents<D>(&self, query: &[f64; K]) -> (f64, u32)
    where
        D: DistanceMetric<f64, Output = f64>,
    {
        let (leaf_idx, _) = self.descend(query);
        let arena = <VecOfArenas<f64, u32, K, B> as LeafStrategy<
            f64,
            u32,
            DonnellyUnrolledLeafEmbedded3,
            K,
            B,
        >>::leaf_arena(&self.leaves, leaf_idx);
        process_arena::<D, K>(&arena, query)
    }

    /// Approximate nearest-one using the descriptor selected from the final block.
    #[inline(always)]
    pub fn approx_nearest_one_embedded<D>(&self, query: &[f64; K]) -> (f64, u32)
    where
        D: DistanceMetric<f64, Output = f64>,
    {
        let (_, descriptor_idx) = self.descend(query);
        let descriptor = unsafe { *self.stem_words.get_unchecked(descriptor_idx) };
        let (byte_offset, leaf_len) = unpack_descriptor(descriptor);
        let arena = self
            .leaves
            .leaf_arena_from_embedded_descriptor(byte_offset, leaf_len);
        process_arena::<D, K>(&arena, query)
    }

    /// Number of logical leaves, exposed for benchmark validation.
    #[inline]
    pub fn leaf_count(&self) -> usize {
        self.leaf_count
    }

    /// Number of packed stem/descriptor words, exposed for benchmark diagnostics.
    #[inline]
    pub fn stem_word_count(&self) -> usize {
        self.stem_words.len()
    }

    #[inline(always)]
    fn descend(&self, query: &[f64; K]) -> (usize, usize) {
        let stems_ptr = NonNull::new(self.stem_words.as_ptr() as *mut u8).unwrap();
        let mut strat = DonnellyUnrolledLeafEmbedded3::new(stems_ptr);
        let pivot_depth = self.max_stem_level + 1;

        debug_assert_eq!((pivot_depth + 1) % 3, 0);

        while strat.level() + FINAL_PIVOT_LEVELS < pivot_depth {
            self.traverse_head(&mut strat, query);
            self.traverse_head(&mut strat, query);
            self.traverse_tail(&mut strat, query);
        }

        self.traverse_head(&mut strat, query);
        self.traverse_head(&mut strat, query);

        let leaf_idx = strat.leaf_idx();
        debug_assert!(leaf_idx < self.leaf_count);
        (leaf_idx, strat.stem_idx())
    }

    #[inline(always)]
    fn traverse_head(&self, strat: &mut DonnellyUnrolledLeafEmbedded3, query: &[f64; K]) {
        let pivot = f64::from_bits(unsafe { *self.stem_words.get_unchecked(strat.stem_idx()) });
        let query_value = unsafe { *query.get_unchecked(strat.dim::<K>()) };
        strat.traverse_head::<f64, K>(query_value >= pivot);
    }

    #[inline(always)]
    fn traverse_tail(&self, strat: &mut DonnellyUnrolledLeafEmbedded3, query: &[f64; K]) {
        let pivot = f64::from_bits(unsafe { *self.stem_words.get_unchecked(strat.stem_idx()) });
        let query_value = unsafe { *query.get_unchecked(strat.dim::<K>()) };
        strat.traverse_tail::<f64, K>(query_value >= pivot);
    }
}

#[inline(always)]
fn process_arena<D, const K: usize>(
    arena: &crate::leaf_view::LeafArena<'_, f64, u32, K>,
    query: &[f64; K],
) -> (f64, u32)
where
    D: DistanceMetric<f64, Output = f64>,
{
    let mut best_dist = f64::INFINITY;
    let mut best_item = 0;
    crate::leaf_view_chunked::nearest_one::nearest_one_with_query_wide_arena::<f64, u32, D, K>(
        arena,
        query,
        &mut best_dist,
        &mut best_item,
    );
    (best_dist, best_item)
}

fn pack_descriptor(
    leaf_idx: usize,
    byte_offset: usize,
    leaf_len: usize,
) -> Result<u64, EmbeddedLeafDescriptorError> {
    if byte_offset as u128 > DESCRIPTOR_OFFSET_MASK as u128 {
        return Err(EmbeddedLeafDescriptorError::ByteOffsetOverflow {
            leaf_idx,
            byte_offset,
        });
    }
    let leaf_len = u16::try_from(leaf_len)
        .map_err(|_| EmbeddedLeafDescriptorError::LeafLengthOverflow { leaf_idx, leaf_len })?;
    Ok((u64::from(leaf_len) << DESCRIPTOR_OFFSET_BITS) | byte_offset as u64)
}

#[inline(always)]
fn unpack_descriptor(descriptor: u64) -> (usize, usize) {
    (
        (descriptor & DESCRIPTOR_OFFSET_MASK) as usize,
        (descriptor >> DESCRIPTOR_OFFSET_BITS) as usize,
    )
}

fn terminal_stem_idx<const K: usize>(leaf_idx: usize, pivot_depth: usize) -> usize {
    let mut strat = DonnellyUnrolledLeafEmbedded3::new_no_ptr();
    for bit_idx in (0..pivot_depth).rev() {
        let is_right = leaf_idx & (1usize << bit_idx) != 0;
        strat.traverse::<f64, K>(is_right);
    }
    strat.stem_idx()
}

#[cfg(test)]
mod tests {
    use super::EmbeddedLeafDescriptorF64Tree;
    use crate::dist::SquaredEuclidean;
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn embedded_and_extent_paths_match() {
        const K: usize = 3;
        const B: usize = 32;
        let mut rng = ChaCha8Rng::seed_from_u64(0x5eed_edd0);
        let points: Vec<[f64; K]> = (0..4096).map(|_| rng.random()).collect();
        let queries: Vec<[f64; K]> = (0..256).map(|_| rng.random()).collect();
        let tree = EmbeddedLeafDescriptorF64Tree::<K, B>::new_from_slice(&points).unwrap();

        assert_eq!(tree.leaf_count(), 128);
        assert!(tree.stem_word_count() > tree.leaf_count());
        for query in &queries {
            assert_eq!(
                tree.approx_nearest_one_via_extents::<SquaredEuclidean<f64>>(query),
                tree.approx_nearest_one_embedded::<SquaredEuclidean<f64>>(query)
            );
        }
    }
}

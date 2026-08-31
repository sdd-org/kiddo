use aligned_vec::{AVec, ConstAlign, CACHELINE_ALIGN};

use crate::kd_tree::item_summary::EMPTY_SUBTREE_CODE;
use crate::kd_tree::OwnedStemLeafResolution;
use crate::stem_strategy::donnelly::DonnellyBlock3SummaryLayout;
use crate::traits::leaf_strategy::LeafStrategy;
use crate::{Donnelly, StemStrategy};

const BLOCK_HEIGHT: usize = 3;
const CHILD_COUNT: usize = 1 << BLOCK_HEIGHT;
const BLOCK_MASK: usize = CHILD_COUNT - 1;
const MAX_NONEMPTY_CODE: u32 = crate::kd_tree::item_summary::MAX_NONEMPTY_CODE as u32;

/// Encodes a conservative linear-bucket lower bound for a subtree minimum.
///
/// The code is `item >> SHIFT`, saturated at `253`. Code `254` is reserved for
/// an empty subtree.
#[inline]
pub(super) const fn encode_min_item<const SHIFT: u8>(item: u32) -> u8 {
    let bucket = item >> SHIFT;
    if bucket > MAX_NONEMPTY_CODE {
        MAX_NONEMPTY_CODE as u8
    } else {
        bucket as u8
    }
}

#[inline]
#[cfg(test)]
pub(super) fn decode_min_item_lower_bound<const SHIFT: u8>(code: u8) -> Option<u32> {
    match code {
        EMPTY_SUBTREE_CODE => None,
        _ => Some(((code as u64) << SHIFT).min(u64::from(u32::MAX)) as u32),
    }
}

#[inline]
fn encode_optional_min<const SHIFT: u8>(item: Option<u32>) -> u8 {
    item.map_or(EMPTY_SUBTREE_CODE, encode_min_item::<SHIFT>)
}

#[inline]
fn optional_min(left: Option<u32>, right: Option<u32>) -> Option<u32> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

struct SummaryBuilder<'a, SS, LS, const K: usize, const B: usize, const SHIFT: u8> {
    stems: &'a mut AVec<f64, ConstAlign<{ CACHELINE_ALIGN }>>,
    leaves: &'a LS,
    resolution: &'a OwnedStemLeafResolution,
    max_terminal_level: i32,
    _stem_strategy: std::marker::PhantomData<SS>,
}

impl<'a, SS, LS, const K: usize, const B: usize, const SHIFT: u8>
    SummaryBuilder<'a, SS, LS, K, B, SHIFT>
where
    SS: DonnellyBlock3SummaryLayout,
    LS: LeafStrategy<f64, u32, SS, K, B>,
{
    fn terminal_leaf_idx(&self, stem: &Donnelly<3>) -> Option<usize> {
        match self.resolution {
            OwnedStemLeafResolution::Arithmetic {
                stems_depth,
                leaf_count,
            }
            | OwnedStemLeafResolution::Pristine {
                stems_depth,
                leaf_count,
            } if stem.level() >= *stems_depth as i32 => {
                let leaf_idx = self
                    .resolution
                    .resolve_terminal_stem_idx(stem.stem_idx(), stem.leaf_idx());
                (leaf_idx < *leaf_count).then_some(leaf_idx)
            }
            OwnedStemLeafResolution::Mapped { .. }
                if self.resolution.is_terminal_stem_idx(stem.stem_idx()) =>
            {
                Some(
                    self.resolution
                        .resolve_terminal_stem_idx(stem.stem_idx(), stem.leaf_idx()),
                )
            }
            _ => None,
        }
    }

    fn leaf_min(&self, leaf_idx: usize) -> Option<u32> {
        (self.leaves.leaf_len(leaf_idx) != 0).then(|| self.leaves.leaf_point_item(leaf_idx, 0).1)
    }

    fn summarize_block(&mut self, root: Donnelly<3>) -> Option<u32> {
        let block_base = root.stem_idx() & !BLOCK_MASK;
        debug_assert_eq!(root.stem_idx(), block_base);

        let mut child_codes = [EMPTY_SUBTREE_CODE; CHILD_COUNT];
        let minimum = self.summarize_within_block(root, 0, 0, &mut child_codes);
        let packed = child_codes
            .into_iter()
            .enumerate()
            .fold(0u64, |word, (child_idx, code)| {
                word | (u64::from(code) << (child_idx * u8::BITS as usize))
            });

        let padding_idx = block_base + BLOCK_MASK;
        if padding_idx >= self.stems.len() {
            self.stems.resize(padding_idx + 1, f64::INFINITY);
        }
        self.stems[padding_idx] = f64::from_bits(packed);

        minimum
    }

    fn summarize_within_block(
        &mut self,
        stem: Donnelly<3>,
        depth_in_block: usize,
        child_prefix: usize,
        child_codes: &mut [u8; CHILD_COUNT],
    ) -> Option<u32> {
        if let Some(leaf_idx) = self.terminal_leaf_idx(&stem) {
            let minimum = self.leaf_min(leaf_idx);
            let suffix_bits = BLOCK_HEIGHT - depth_in_block;
            let first_child = child_prefix << suffix_bits;
            let end_child = (child_prefix + 1) << suffix_bits;
            child_codes[first_child..end_child].fill(encode_optional_min::<SHIFT>(minimum));
            return minimum;
        }

        if stem.level() > self.max_terminal_level {
            let suffix_bits = BLOCK_HEIGHT - depth_in_block;
            let first_child = child_prefix << suffix_bits;
            let end_child = (child_prefix + 1) << suffix_bits;
            child_codes[first_child..end_child].fill(EMPTY_SUBTREE_CODE);
            return None;
        }

        if depth_in_block == BLOCK_HEIGHT {
            let minimum = self.summarize_block(stem);
            child_codes[child_prefix] = encode_optional_min::<SHIFT>(minimum);
            return minimum;
        }

        let (left, right) = stem.split::<f64, K>();
        let left_minimum =
            self.summarize_within_block(left, depth_in_block + 1, child_prefix << 1, child_codes);
        let right_minimum = self.summarize_within_block(
            right,
            depth_in_block + 1,
            (child_prefix << 1) | 1,
            child_codes,
        );
        optional_min(left_minimum, right_minimum)
    }
}

pub(in crate::kd_tree) fn populate_f64_donnelly3_min_item_summaries<
    SS,
    LS,
    const K: usize,
    const B: usize,
    const SHIFT: u8,
>(
    stems: &mut AVec<f64, ConstAlign<{ CACHELINE_ALIGN }>>,
    leaves: &LS,
    resolution: &OwnedStemLeafResolution,
    max_stem_level: i32,
) where
    SS: DonnellyBlock3SummaryLayout,
    LS: LeafStrategy<f64, u32, SS, K, B>,
{
    if stems.is_empty() {
        return;
    }

    SummaryBuilder::<SS, LS, K, B, SHIFT> {
        stems,
        leaves,
        resolution,
        max_terminal_level: max_stem_level + 1,
        _stem_strategy: std::marker::PhantomData,
    }
    .summarize_block(Donnelly::<3>::new_no_ptr());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_encoding_is_a_conservative_saturating_lower_bound() {
        assert_eq!(encode_min_item::<8>(0), 0);
        assert_eq!(encode_min_item::<8>(255), 0);
        assert_eq!(encode_min_item::<8>(256), 1);
        assert_eq!(encode_min_item::<8>(511), 1);
        assert_eq!(encode_min_item::<8>(512), 2);
        assert_eq!(encode_min_item::<8>(64_767), 252);
        assert_eq!(encode_min_item::<8>(64_768), 253);
        assert_eq!(encode_min_item::<8>(116_000), 253);
        assert_eq!(encode_min_item::<8>(u32::MAX), 253);

        for item in [0, 1, 255, 256, 511, 512, 64_767, 64_768, 116_000] {
            let code = encode_min_item::<8>(item);
            let lower_bound = decode_min_item_lower_bound::<8>(code).unwrap();
            assert!(lower_bound <= item);
        }
        assert_eq!(decode_min_item_lower_bound::<8>(EMPTY_SUBTREE_CODE), None);
    }
}

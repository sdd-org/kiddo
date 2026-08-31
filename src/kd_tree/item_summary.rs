use std::any::TypeId;

use crate::Axis;

use super::{ITEM_LEAF_MODE_SORTED, ITEM_LEAF_MODE_UNSORTED};

const DONNELLY_BLOCK_HEIGHT: usize = 3;
const DONNELLY_BLOCK_WIDTH: usize = 1 << DONNELLY_BLOCK_HEIGHT;
const DONNELLY_BLOCK_MASK: usize = DONNELLY_BLOCK_WIDTH - 1;

/// Reserved summary code marking an empty subtree. Shared with the
/// construction-side encoder in `kd_tree::construction::item_summary`.
pub(crate) const EMPTY_SUBTREE_CODE: u8 = u8::MAX - 1;
/// Highest non-empty summary code; every higher code is `EMPTY_SUBTREE_CODE`.
pub(crate) const MAX_NONEMPTY_CODE: u8 = EMPTY_SUBTREE_CODE - 1;

/// Compact query-time state for filtering embedded subtree-minimum codes.
/// It remains disabled until `best_n_within` has filled its result collection.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct ItemSummaryFilter {
    max_live_code: u8,
    enabled: bool,
    any_live: bool,
}

impl ItemSummaryFilter {
    #[inline(always)]
    pub(crate) const fn disabled() -> Self {
        Self {
            max_live_code: MAX_NONEMPTY_CODE,
            enabled: false,
            any_live: true,
        }
    }

    #[inline(always)]
    pub(crate) fn from_threshold<T: Copy + 'static, const ITEM_LEAF_MODE: u8>(
        threshold: Option<T>,
    ) -> Self {
        if ITEM_LEAF_MODE == ITEM_LEAF_MODE_UNSORTED
            || ITEM_LEAF_MODE == ITEM_LEAF_MODE_SORTED
            || ITEM_LEAF_MODE >= u32::BITS as u8
            || TypeId::of::<T>() != TypeId::of::<u32>()
        {
            return Self::disabled();
        }

        let Some(threshold) = threshold else {
            return Self::disabled();
        };

        // SAFETY: the TypeId equality above proves that T is u32.
        let threshold = unsafe { *(&threshold as *const T).cast::<u32>() };
        if threshold == 0 {
            return Self {
                max_live_code: 0,
                enabled: true,
                any_live: false,
            };
        }

        let max_live_code =
            ((threshold - 1) >> ITEM_LEAF_MODE).min(u32::from(MAX_NONEMPTY_CODE)) as u8;

        Self {
            max_live_code,
            enabled: true,
            any_live: true,
        }
    }

    #[inline(always)]
    pub(crate) fn live_mask(self, packed_codes: u64) -> u8 {
        if !self.enabled {
            return u8::MAX;
        }
        if !self.any_live {
            return 0;
        }
        packed_code_live_mask(packed_codes, self.max_live_code)
    }
}

#[inline(always)]
fn packed_code_live_mask(packed_codes: u64, max_live_code: u8) -> u8 {
    #[cfg(all(
        feature = "simd",
        target_arch = "x86_64",
        target_feature = "avx512bw",
        target_feature = "avx512vl"
    ))]
    unsafe {
        use std::arch::x86_64::*;

        let codes = _mm_cvtsi64_si128(packed_codes as i64);
        let cutoff = _mm_set1_epi8(max_live_code as i8);
        // The compare covers all 16 vector bytes; the upper eight are zero, so
        // the `as u8` cast keeps exactly the eight summary lanes.
        _mm_cmp_epu8_mask(codes, cutoff, _MM_CMPINT_LE) as u8
    }

    #[cfg(all(
        feature = "simd",
        target_arch = "x86_64",
        target_feature = "avx2",
        not(all(target_feature = "avx512bw", target_feature = "avx512vl"))
    ))]
    unsafe {
        use std::arch::x86_64::*;

        let bytes = _mm_cvtsi64_si128(packed_codes as i64);
        let codes = _mm256_cvtepu8_epi32(bytes);
        let cutoff = _mm256_set1_epi32(i32::from(max_live_code));
        let rejected = _mm256_cmpgt_epi32(codes, cutoff);
        !(_mm256_movemask_ps(_mm256_castsi256_ps(rejected)) as u8)
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    unsafe {
        use std::arch::aarch64::*;

        let codes = vcreate_u8(packed_codes);
        let cutoff = vdup_n_u8(max_live_code);
        let live = vcle_u8(codes, cutoff);
        let weights = vcreate_u8(0x8040_2010_0804_0201);
        vaddv_u8(vand_u8(live, weights))
    }

    #[cfg(not(any(
        all(
            feature = "simd",
            target_arch = "x86_64",
            target_feature = "avx512bw",
            target_feature = "avx512vl"
        ),
        all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"),
        all(feature = "simd", target_arch = "aarch64")
    )))]
    {
        let mut mask = 0u8;
        let mut lane = 0usize;
        while lane < DONNELLY_BLOCK_WIDTH {
            let code = (packed_codes >> (lane * u8::BITS as usize)) as u8;
            if code <= max_live_code {
                mask |= 1u8 << lane;
            }
            lane += 1;
        }
        mask
    }
}

#[allow(missing_docs)]
#[cfg(feature = "cargo_asm")]
pub mod cargo_asm {
    use super::ItemSummaryFilter;

    /// Assembly-inspection hook for the monomorphised u32/shift-8 mask kernel.
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_embedded_min_item_summary_mask_cargo_asm_hook(
        packed_codes: u64,
        threshold: u32,
    ) -> u8 {
        ItemSummaryFilter::from_threshold::<u32, 8>(Some(threshold)).live_mask(packed_codes)
    }
}

#[inline(always)]
pub(crate) fn donnelly3_block_live_mask<A>(
    stems: &[A],
    block_base: usize,
    filter: ItemSummaryFilter,
) -> u8
where
    A: Axis<Coord = A>,
{
    if !filter.enabled {
        return u8::MAX;
    }
    let Some(&padding) = stems.get(block_base + DONNELLY_BLOCK_MASK) else {
        return u8::MAX;
    };
    let Some(word) = A::embedded_item_summary_word(padding) else {
        return u8::MAX;
    };
    let mask = filter.live_mask(word);
    #[cfg(any(feature = "exact_query_stats", feature = "test_utils"))]
    crate::results::exact_query_stats::record_item_summary_block(mask);
    mask
}

#[inline(always)]
pub(crate) fn donnelly3_subtree_is_live<A>(
    stems: &[A],
    stem_idx: usize,
    level: i32,
    filter: ItemSummaryFilter,
) -> bool
where
    A: Axis<Coord = A>,
{
    if !filter.enabled || level < 0 {
        return true;
    }

    let depth = level as usize % DONNELLY_BLOCK_HEIGHT;
    let block_base = stem_idx & !DONNELLY_BLOCK_MASK;
    let local_idx = stem_idx.wrapping_sub(block_base);
    let first_at_depth = (1usize << depth) - 1;
    if local_idx < first_at_depth || local_idx >= first_at_depth + (1usize << depth) {
        return true;
    }

    let prefix = local_idx - first_at_depth;
    let suffix_bits = DONNELLY_BLOCK_HEIGHT - depth;
    let first_lane = prefix << suffix_bits;
    let lane_count = 1usize << suffix_bits;
    let subtree_mask = (((1u16 << lane_count) - 1) << first_lane) as u8;

    let live = donnelly3_block_live_mask(stems, block_base, filter) & subtree_mask != 0;
    #[cfg(any(feature = "exact_query_stats", feature = "test_utils"))]
    if !live {
        crate::results::exact_query_stats::record_item_summary_subtree_prune();
    }
    live
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack(codes: [u8; 8]) -> u64 {
        codes.into_iter().enumerate().fold(0, |word, (lane, code)| {
            word | (u64::from(code) << (lane * 8))
        })
    }

    #[test]
    fn cutoff_tracks_strict_item_improvement_at_bucket_boundaries() {
        let first_bucket_end = ItemSummaryFilter::from_threshold::<u32, 8>(Some(256));
        assert_eq!(
            first_bucket_end.live_mask(pack([0, 1, 2, 3, 253, 254, 0, 1])),
            0b0100_0001
        );

        let second_bucket_start = ItemSummaryFilter::from_threshold::<u32, 8>(Some(257));
        assert_eq!(
            second_bucket_start.live_mask(pack([0, 1, 2, 3, 253, 254, 0, 1])),
            0b1100_0011
        );

        let saturation_start = ItemSummaryFilter::from_threshold::<u32, 8>(Some(64_768));
        assert_eq!(
            saturation_start.live_mask(pack([252, 253, 254, 0, 251, 252, 253, 254])),
            0b0011_1001
        );

        let inside_saturated_bucket = ItemSummaryFilter::from_threshold::<u32, 8>(Some(64_769));
        assert_eq!(
            inside_saturated_bucket.live_mask(pack([252, 253, 254, 0, 251, 252, 253, 254])),
            0b0111_1011
        );
    }

    #[test]
    fn absent_threshold_disables_summary_checks() {
        let filter = ItemSummaryFilter::from_threshold::<u32, 8>(None);
        assert_eq!(filter.live_mask(pack([0, 1, 2, 31, 254, 0, 1, 2])), u8::MAX);
    }

    #[test]
    fn zero_threshold_rejects_every_subtree() {
        let filter = ItemSummaryFilter::from_threshold::<u32, 8>(Some(0));
        assert_eq!(filter.live_mask(u64::MAX), 0);
    }
}

#![allow(missing_docs)]

use std::cell::Cell;

/// Invasive structural accounting for exact-query development runs.
///
/// This is thread-local and deliberately compiled out unless `test_utils` or
/// `exact_query_stats` is enabled.  Do not enable it in timing or hardware-
/// counter runs: every event adds work to the hot query path.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExactQueryStats {
    pub queries: u64,
    pub leaf_visits: u64,
    pub leaf_items_available: u64,
    pub leaf_items_distance_evaluated: u64,
    pub item_sorted_threshold_stops: u64,
    pub stem_steps: u64,
    pub real_pivot_steps: u64,
    pub padding_steps: u64,
    pub initial_far_candidates: u64,
    pub initial_far_rejects: u64,
    pub continuation_frames_pushed: u64,
    pub continuation_frames_popped: u64,
    pub far_rechecks: u64,
    pub far_enters: u64,
    pub far_rejects_after_near: u64,
    pub scalar_stack_pops: u64,
    pub simd_single_pops: u64,
    pub simd_stack_max_len: u64,
    pub block3_pending_pops: u64,
    pub block3_pending_mask_bits: u64,
    pub block3_candidate_mask_bits: u64,
    pub block3_candidate_mask_nonzero: u64,
    pub block3_step_entries: u64,
    pub block3_full_steps: u64,
    pub block3_scalar_fallback_steps: u64,
    pub item_summary_blocks_checked: u64,
    pub item_summary_lanes_rejected: u64,
    pub item_summary_subtrees_pruned: u64,
}

thread_local! {
    static STATS: Cell<ExactQueryStats> = Cell::new(ExactQueryStats::default());
}

#[inline]
pub fn reset() {
    STATS.with(|stats| stats.set(ExactQueryStats::default()));
}

#[inline]
pub fn snapshot() -> ExactQueryStats {
    STATS.with(Cell::get)
}

#[inline]
fn update(f: impl FnOnce(&mut ExactQueryStats)) {
    STATS.with(|stats| {
        let mut value = stats.get();
        f(&mut value);
        stats.set(value);
    });
}

#[inline]
pub fn record_query() {
    update(|stats| stats.queries += 1);
}

#[inline]
pub fn record_leaf_visit() {
    update(|stats| stats.leaf_visits += 1);
}

#[inline]
pub fn record_leaf_items_available(len: usize) {
    update(|stats| stats.leaf_items_available += len as u64);
}

#[inline]
pub fn record_leaf_items_distance_evaluated(len: usize) {
    update(|stats| stats.leaf_items_distance_evaluated += len as u64);
}

#[inline]
pub fn record_item_sorted_threshold_stop() {
    update(|stats| stats.item_sorted_threshold_stops += 1);
}

#[inline]
pub fn record_stem_step(real_pivot: bool) {
    update(|stats| {
        stats.stem_steps += 1;
        if real_pivot {
            stats.real_pivot_steps += 1;
        } else {
            stats.padding_steps += 1;
        }
    });
}

#[inline]
pub fn record_initial_far_candidate() {
    update(|stats| stats.initial_far_candidates += 1);
}

#[inline]
pub fn record_initial_far_reject() {
    update(|stats| stats.initial_far_rejects += 1);
}

#[inline]
pub fn record_continuation_frame_push() {
    update(|stats| stats.continuation_frames_pushed += 1);
}

#[inline]
pub fn record_continuation_frame_pop() {
    update(|stats| stats.continuation_frames_popped += 1);
}

#[inline]
pub fn record_far_recheck() {
    update(|stats| stats.far_rechecks += 1);
}

#[inline]
pub fn record_far_enter() {
    update(|stats| stats.far_enters += 1);
}

#[inline]
pub fn record_far_reject_after_near() {
    update(|stats| stats.far_rejects_after_near += 1);
}

#[inline]
pub fn record_scalar_stack_pop() {
    update(|stats| stats.scalar_stack_pops += 1);
}

#[inline]
pub fn record_simd_single_pop() {
    update(|stats| stats.simd_single_pops += 1);
}

#[inline]
pub fn record_simd_stack_len(len: usize) {
    update(|stats| stats.simd_stack_max_len = stats.simd_stack_max_len.max(len as u64));
}

#[inline]
pub fn record_block3_pending_pop(mask: u8) {
    update(|stats| {
        stats.block3_pending_pops += 1;
        stats.block3_pending_mask_bits += mask.count_ones() as u64;
    });
}

#[inline]
pub fn record_block3_candidate_mask(mask: u8) {
    update(|stats| {
        stats.block3_candidate_mask_bits += mask.count_ones() as u64;
        if mask != 0 {
            stats.block3_candidate_mask_nonzero += 1;
        }
    });
}

#[inline]
pub fn record_block3_full_step() {
    update(|stats| stats.block3_full_steps += 1);
}

#[inline]
pub fn record_block3_scalar_fallback_step() {
    update(|stats| stats.block3_scalar_fallback_steps += 1);
}

#[inline]
pub fn record_block3_step_entry() {
    update(|stats| stats.block3_step_entries += 1);
}

#[inline]
pub fn record_item_summary_block(mask: u8) {
    update(|stats| {
        stats.item_summary_blocks_checked += 1;
        stats.item_summary_lanes_rejected += u64::from((!mask).count_ones());
    });
}

#[inline]
pub fn record_item_summary_subtree_prune() {
    update(|stats| stats.item_summary_subtrees_pruned += 1);
}

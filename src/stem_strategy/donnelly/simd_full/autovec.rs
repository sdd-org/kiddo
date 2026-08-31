//! Autovectorization-friendly block marker comparisons.

use std::ptr::NonNull;

/// Compare query value against 7 pivots in a Block3 (f64, autovec).
#[inline(always)]
#[allow(unused)]
pub unsafe fn compare_block3_f64_autovec(
    stems_ptr: NonNull<u8>,
    cache_line_base: usize,
    query_val: f64,
) -> u8 {
    let ptr = stems_ptr.as_ptr().add(cache_line_base * 8) as *const f64;
    let mut count = 0u8;
    for i in 0..7 {
        if query_val >= *ptr.add(i) {
            count += 1;
        }
    }
    count
}

/// Compare query value against 7 pivots in a Block3 (f32, autovec).
#[inline(always)]
#[allow(unused)]
pub unsafe fn compare_block3_f32_autovec(
    stems_ptr: NonNull<u8>,
    cache_line_base: usize,
    query_val: f32,
) -> u8 {
    let ptr = stems_ptr.as_ptr().add(cache_line_base * 4) as *const f32;
    let mut count = 0u8;
    for i in 0..7 {
        if query_val >= *ptr.add(i) {
            count += 1;
        }
    }
    count
}

/// Compare query value against 15 pivots in a Block4 (f32, autovec).
#[inline(always)]
#[allow(unused)]
pub unsafe fn compare_block4_f32_autovec(
    stems_ptr: NonNull<u8>,
    cache_line_base: usize,
    query_val: f32,
) -> u8 {
    let ptr = stems_ptr.as_ptr().add(cache_line_base * 4) as *const f32;
    let mut count = 0u8;
    for i in 0..16 {
        if query_val >= *ptr.add(i) {
            count += 1;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_block3_f64_autovec_counts_matches_in_selected_block() {
        let stems = [
            -9.0,
            -8.0,
            -7.0,
            -6.0,
            -5.0,
            -4.0,
            -3.0,
            f64::NEG_INFINITY,
            0.2,
            0.4,
            0.6,
            0.1,
            0.3,
            0.5,
            0.7,
            f64::INFINITY,
        ];
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();

        let cases = [(0.05, 0), (0.3, 3), (0.55, 5), (0.7, 7)];

        for (query, expected) in cases {
            let actual = unsafe { compare_block3_f64_autovec(stems_ptr, 8, query) };
            assert_eq!(actual, expected, "query={query}");
        }
    }

    #[test]
    fn compare_block3_f32_autovec_counts_matches_in_selected_block() {
        let stems = [
            -9.0f32,
            -8.0,
            -7.0,
            -6.0,
            -5.0,
            -4.0,
            -3.0,
            f32::NEG_INFINITY,
            0.2,
            0.4,
            0.6,
            0.1,
            0.3,
            0.5,
            0.7,
            f32::INFINITY,
        ];
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();

        let cases = [(0.05f32, 0), (0.3, 3), (0.55, 5), (0.7, 7)];

        for (query, expected) in cases {
            let actual = unsafe { compare_block3_f32_autovec(stems_ptr, 8, query) };
            assert_eq!(actual, expected, "query={query}");
        }
    }

    #[test]
    fn compare_block4_f32_autovec_counts_matches_in_selected_block() {
        let stems = [
            -16.0f32,
            -15.0,
            -14.0,
            -13.0,
            -12.0,
            -11.0,
            -10.0,
            -9.0,
            -8.0,
            -7.0,
            -6.0,
            -5.0,
            -4.0,
            -3.0,
            -2.0,
            f32::INFINITY,
            0.7,
            0.3,
            1.1,
            0.1,
            0.5,
            0.9,
            1.3,
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
            1.2,
            1.4,
            f32::INFINITY,
        ];
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();

        let cases = [(0.05f32, 1), (0.65, 7), (1.0, 11), (1.4, 15)];

        for (query, expected) in cases {
            let actual = unsafe { compare_block4_f32_autovec(stems_ptr, 16, query) };
            assert_eq!(actual, expected, "query={query}");
        }
    }
}

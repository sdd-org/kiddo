//! Regression for #571: scalar/mapped backtracking must restore
//! deferred far-child offsets. The tree is valid (the `within(U+221E)` check
//! below passes), but `nearest_n` previously over-pruned a subtree containing
//! valid results.

use std::collections::HashSet;
use std::num::NonZeroUsize;

use kiddo::{Chebyshev, Eytzinger, KdTree, VecOfArrays};

type Tree<const B: usize> = KdTree<f64, u32, Eytzinger, VecOfArrays<f64, u32, 3, B>, 3, B>;

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

fn entries(n: usize, seed: u64) -> Vec<(u32, [f64; 3])> {
    let mut s = seed | 1;
    let mut seen = HashSet::new();
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let key = [0u64; 3].map(|_| xorshift(&mut s) % 1_000_003);
        if seen.insert(key) {
            let point = key.map(|v| v as f64 / 1_000_003.0);
            out.push((out.len() as u32, point));
        }
    }
    out
}

fn queries(seed: u64, count: usize) -> Vec<[f64; 3]> {
    let mut s = seed | 1;
    (0..count)
        .map(|_| [0u64; 3].map(|_| (xorshift(&mut s) % 1_000_003) as f64 / 1_000_003.0))
        .collect()
}

fn chebyshev(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn mutable_scalar_backtracking_restores_deferred_offsets() {
    let entries = entries(300, 1);
    let tree = Tree::<4>::new_from_entries(&entries).unwrap();

    assert_eq!(tree.size(), entries.len());
    assert_eq!(tree.iter().count(), entries.len());

    let query = queries(0x9e37_79b9 ^ 17 ^ 300, 40)[0];

    let mut expected: Vec<f64> = entries
        .iter()
        .map(|(_, point)| chebyshev(&query, point))
        .collect();
    expected.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut all: Vec<f64> = tree
        .query(&query)
        .within::<Chebyshev<f64>>(f64::INFINITY)
        .execute()
        .into_iter()
        .map(|result| result.distance)
        .collect();
    all.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(all, expected, "the constructed tree should be valid");

    let mut got: Vec<f64> = tree
        .query(&query)
        .nearest_n::<Chebyshev<f64>>(NonZeroUsize::new(33).unwrap())
        .execute()
        .into_iter()
        .map(|result| result.distance)
        .collect();
    got.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert_eq!(got, &expected[..33]);
}

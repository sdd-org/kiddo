#![cfg_attr(
    all(kiddo_nightly, target_arch = "aarch64"),
    feature(stdarch_aarch64_prefetch)
)]
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![warn(rustdoc::missing_crate_level_docs)]
#![deny(rustdoc::invalid_codeblock_attributes)]
#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]
#![warn(rustdoc::private_intra_doc_links)]
#![doc(html_root_url = "https://docs.rs/kiddo/6.0.0-alpha.4")]
#![doc(issue_tracker_base_url = "https://github.com/sdd/kiddo/issues/")]
#![allow(clippy::pointers_in_nomem_asm_block)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

//! # Kiddo
//!
//! A high-performance k-d tree library for exact and approximate nearest-neighbour
//! queries in low-dimensional spaces.
//!
//! Built with an aggressive focus on query performance, including cache-aware
//! layouts and optional SIMD-accelerated code paths. See the companion
//! benchmarking site to compare Kiddo against other k-d tree implementations
//! across a range of workloads.
//!
//! Kiddo v6 provides a single generic [`KdTree`](crate::kd_tree::KdTree) that supports floating-point
//! (`f64`, `f32`, `f16`), selected fixed-point (via the `fixed` crate), and
//! unsigned-integer (`u8`, `u16`, `u32`) types as coordinates, along with both mutable
//! and immutable usage patterns.
//!
//! Kiddo is designed for low-dimensional (< ~10D) search problems, especially 2D, 3D,
//! and 4D workloads. Typical use cases include point-cloud analysis,
//! astronomical catalogue crossmatching, colour quantization and palette
//! lookup, local neighbourhood queries in simulations, and other
//! nearest-neighbour and radius-search tasks. Kiddo has been used for diverse
//! geographical and scientific workloads including geocoding, astronomy,
//! cosmology, computer-aided drug discovery, crystallography, and
//! computational neuroscience.
//!
//! Kiddo supports the following query types:
//!
//! - **Exact Nearest Neighbour**: Useful for tasks like finding the nearest airport to a given
//!   location, or finding the nearest catalogued star to a sky position.
//!   `tree.query(&point).nearest_one().execute()`
//! - **k-nearest-neighbour (k-NN)** search, finding the `k` nearest items to a query point:
//!     - ordered by distance: `tree.query(&point).nearest_n(5).execute()`.
//!       Useful for finding the nearest weather stations or sensors to a location,
//!       or generating candidate correspondences for point-cloud registration.
//!     - `k` items within a max radius: `tree.query(&point).nearest_n(5).within(max_dist).execute()`.
//!       Useful when you want the closest local neighbours inside a meaningful
//!       cutoff, such as the nearest shops within 5 miles, or nearby atoms within
//!       an interaction radius.
//!     - All items within a radius: `tree.query(&point).within(max_dist).execute()`. Finds all
//!       items within a specified radius of a query point, ordered by distance. Useful for radial
//!       catalogue searches in astronomy, or collision and proximity queries where the full
//!       neighbourhood is needed in sorted order.
//!     - Unsorted, e.g. `tree.query(&point).nearest_n(5).within(max_dist).unsorted().execute()`:
//!       This is often faster than the sorted radius-query form when result order does not
//!       matter, such as finding all customers within 5 miles of a store, or
//!       collecting point-cloud neighbourhoods for clustering or normal estimation.
//! - **Approximate nearest-neighbour** (ANN) search: `tree.query(&point).nearest_one().approx().execute()`
//!   Returns a good approximate nearest item. Generally much faster than exact nearest-neighbour
//!   search. Useful for latency-sensitive workloads like interactive point-cloud picking, or
//!   mapping image pixels to a palette colour during colour quantization.
//! - **"best" `n` items**: `tree.query(&point).best_n(5, max_dist).execute()`. Finds the "best" n
//!   items within a specified distance of a query point, for some definition of "best". For
//!   example, "give me the 5 largest settlements within 50km of a given point, ordered by
//!   descending population", or "the 5 brightest stars within a degree of a point on the sky,
//!   ordered brightest first". This only makes semantic sense when your item type has meaningful
//!   ordering; for points-only trees with `T = ()`, the query is allowed but not useful.
//! - **[Periodic Boundary Conditions](https://en.wikipedia.org/wiki/Periodic_boundary_conditions) (PBC)**,
//!   whereby the points in the tree are considered to represent a single subunit that repeats
//!   across space. Useful primarily for simulations, such as within cosmology or molecular dynamics
//!   simulations:
//!   `tree.query(&point).periodic_boundary_condition(box_size).within(max_dist).execute()`
//! - **Exclusive Boundary Queries**, where the query radius filter is an exclusive boundary
//!   (< max_dist), rather than the default inclusive boundary (<= max_dist):
//!   `tree.query(&point).within(max_dist).exclusive_boundaries().execute()`
//!
//! If your points are known up front and the tree will be built once and then
//! queried, start with [`ImmutableKdTree`]. It offers the best query
//! performance and pairs well with `rkyv` for zero-copy loading of prebuilt
//! trees from disk; when used with memory-mapped files, loading can be
//! effectively instant.
//!
//! If you need to add or remove points after construction, start with
//! [`MutableKdTree`]. Mutable trees remain a good fit for many dynamic
//! workloads, but they do not currently perform dynamic rebalancing, so
//! workloads with substantial growth or heavy churn may benefit from periodic
//! rebuilds.
//!
//! [`ImmutableKdTree`] and [`MutableKdTree`] are convenience aliases for
//! [`KdTree`](crate::kd_tree::KdTree) with sensible defaults for these common read-heavy and mutable
//! workloads.
//!
//! Kiddo is not intended as a library for high-dimensional vector search or
//! feature matching over hundreds or thousands of dimensions, where
//! k-d trees are usually the wrong data structure and other approaches are more
//! appropriate. The API does not impose a hard dimensional limit, but Kiddo is
//! primarily intended for low-dimensional workloads.
//!
//! ## Installation
//!
//! Add `kiddo` to `Cargo.toml`
//! ```toml
//! [dependencies]
//! kiddo = "6.0.0-alpha.4"
//! ```
//!
//! ## Usage
//! ```rust
//! use std::num::NonZero;
//!
//! use kiddo::leaf_strategies::VecOfArrays;
//! use kiddo::SquaredEuclidean;
//! use kiddo::QueryResultItem;
//! use kiddo::{Eytzinger, ImmutableKdTree};
//!
//! let entries = vec![
//!     [0f64, 0f64],
//!     [1f64, 1f64],
//!     [2f64, 2f64],
//!     [3f64, 3f64]
//! ];
//!
//! let kdtree = ImmutableKdTree::new_from_slice(&entries).unwrap();
//!
//! // How many items are in tree?
//! assert_eq!(kdtree.size(), 4);
//!
//! // find the nearest item to [0f64, 0f64].
//! let nearest = kdtree
//!     .query(&[0f64, 0f64])
//!     .nearest_one::<SquaredEuclidean<f64>>()
//!     .execute();
//! assert_eq!(nearest.distance, 0f64);
//! assert_eq!(nearest.item, 0);
//!
//! // find the nearest 3 items to [0f64, 0f64], and collect into a `Vec`
//! assert_eq!(
//!     kdtree
//!         .query(&[0f64, 0f64])
//!         .nearest_n::<SquaredEuclidean<f64>>(NonZero::new(3usize).unwrap())
//!         .execute(),
//!     vec![
//!         QueryResultItem { point: (), distance: 0f64, item: 0 },
//!         QueryResultItem { point: (), distance: 2f64, item: 1 },
//!         QueryResultItem { point: (), distance: 8f64, item: 2 }
//!     ]
//! );
//! ```
//!
//! See the [examples documentation](https://github.com/sdd/kiddo/tree/master/examples) for some more in-depth examples.
//!
//! ## Optional Features
//!
//! Kiddo exposes a number of optional crate features:
//!
//! - `fixed` enables support for fixed-point coordinate
//!   types from the [`fixed`](https://docs.rs/fixed/latest/fixed) crate.
//!
//! - `f16` enables support for half-precision floating-point coordinates via
//!   the [`half`](https://docs.rs/half/latest/half) crate.
//!
//! - `serde` enables serialization and deserialization via
//!   [`Serde`](https://docs.rs/serde/latest/serde/).
//!
//! - `rkyv_08` enables zero-copy serialization and deserialization via
//!   [`rkyv`](https://docs.rs/rkyv/latest/rkyv/) 0.8.x. This is particularly
//!   useful for prebuilt immutable trees that you want to load very quickly,
//!   especially in conjunction with memory-mapped files.
//!
//! - `simd` **(NIGHTLY)** enables handwritten SIMD and prefetch intrinsics for
//!   additional performance where available. This requires a nightly Rust
//!   toolchain.
//!
//! - `huge_pages` enables Linux-specific huge-page advice helpers for owned and
//!   archived tree storage.
//!
//! - `leaf_nta_prefetch` enables additional non-temporal leaf prefetch hints in
//!   some query paths. This is an advanced tuning feature and is only useful in
//!   specific workloads.
//!
//! Kiddo also contains a number of additional feature flags used for internal
//! experimentation, benchmarking, simulation, and specialized tuning. Most
//! users will not need them.
//!
//! ## MSRV
//!
//! Kiddo v6's current minimum supported Rust version (MSRV) is **1.89.0**
//! (**1.85.0** for `v5.x.x`).
//!
//! Kiddo will aim to support at least **N-4** stable Rust releases, which is
//! roughly six months of stable compiler history, when used with the default
//! crate features.
//!
//! Kiddo will also endeavour to increase MSRV only when doing so would provide
//! a material improvement for users, rather than simply for the sake of using
//! a newer compiler.
//!
//! Optional features may require a newer toolchain than the default-feature
//! MSRV if their dependency stack requires it. The `simd` feature is
//! nightly-only and is outside the stable MSRV policy.
//!
//! **NOTE**: Support for rkyv 0.7 was removed in Kiddo v6.

#[doc(hidden)]
#[cfg(feature = "serde")]
#[doc(hidden)]
mod custom_serde;

pub mod kd_tree;
#[doc(hidden)]
pub type KdTree<A, T, SS, LS, const K: usize, const B: usize> = kd_tree::KdTree<A, T, SS, LS, K, B>;
pub use kd_tree::QueryScratch;

/// Distance metrics
pub mod dist;
#[doc(hidden)]
pub type Chebyshev<R> = crate::dist::Chebyshev<R>;
#[doc(hidden)]
pub type Manhattan<R> = crate::dist::Manhattan<R>;
#[doc(hidden)]
pub type Minkowski<const P: u32, R> = crate::dist::Minkowski<P, R>;
#[doc(hidden)]
pub type SquaredEuclidean<R> = crate::dist::SquaredEuclidean<R>;

/// Stem ordering strategies for the kd-tree
#[path = "stem_strategy/mod.rs"]
pub mod stem_strategies;
#[doc(hidden)]
pub use stem_strategies as stem_strategy;
#[doc(hidden)]
pub type Donnelly<const BH: usize> = stem_strategies::donnelly::Donnelly<BH>;
#[doc(hidden)]
pub type DonnellyNoPf<const BH: usize> = stem_strategies::donnelly::DonnellyNoPf<BH>;
#[doc(hidden)]
pub type DonnellySimdDescent<const BH: usize> = stem_strategies::donnelly::DonnellySimdDescent<BH>;
#[doc(hidden)]
pub type DonnellySimdFull<const BH: usize> = stem_strategies::donnelly::DonnellySimdFull<BH>;
#[doc(hidden)]
pub type DonnellyUnrolled<const BH: usize> = stem_strategies::donnelly::DonnellyUnrolled<BH>;
#[doc(hidden)]
pub type DonnellyUnrolledBlockDim<const BH: usize> =
    stem_strategies::donnelly::DonnellyUnrolledBlockDim<BH>;
#[doc(hidden)]
pub type Eytzinger = stem_strategies::eytzinger::Eytzinger;
#[doc(hidden)]
pub type EytzingerFlexPf<const PF1: isize = 0, const PF2: isize = 1> =
    stem_strategies::eytzinger::EytzingerFlexPf<PF1, PF2>;
#[doc(hidden)]
pub type EytzingerNoPf = stem_strategies::eytzinger::EytzingerNoPf;

/// Leaf storage strategies for the kd-tree
#[path = "leaf_strategy/mod.rs"]
pub mod leaf_strategies;
#[doc(hidden)]
pub use leaf_strategies as leaf_strategy;
#[doc(hidden)]
pub type FlatVec<A, T, const K: usize, const B: usize> = leaf_strategies::FlatVec<A, T, K, B>;
#[doc(hidden)]
pub type VecOfArenas<A, T, const K: usize, const B: usize> =
    leaf_strategies::VecOfArenas<A, T, K, B>;
#[doc(hidden)]
pub type VecOfArrays<A, T, const K: usize, const B: usize> =
    leaf_strategies::VecOfArrays<A, T, K, B>;

/// Convenience type alias for recommended default params for an immutable KdTree
pub type ImmutableKdTree<AX, const K: usize> =
    KdTree<AX, u32, Eytzinger, VecOfArenas<AX, u32, K, 32>, K, 32>;

/// Convenience type alias for recommended default params for a mutable KdTree
pub type MutableKdTree<AX, const K: usize> =
    KdTree<AX, u32, Eytzinger, VecOfArrays<AX, u32, K, 32>, K, 32>;

pub mod huge_pages;

/// Leaf view abstraction for accessing leaf data
#[doc(hidden)]
pub mod leaf_view;

/// Chunked Leaf view abstraction for accessing leaf data
pub(crate) mod leaf_view_chunked;

mod mirror_select_nth_unstable_by;

/// Structs that are returned as query results
#[doc(hidden)]
pub mod results;
pub use results::{
    best_query_result_item::BestQueryResultItem, query_result_item::QueryResultItem,
};

#[cfg(feature = "rkyv_08")]
mod rkyv;

#[doc(hidden)]
#[cfg(feature = "test_utils")]
pub mod test_utils;

pub mod traits;
#[doc(hidden)]
pub use traits::{
    Axis, ConstructibleLeafStrategy, Content, Immutable, LeafStrategy, Mutability, Mutable,
    MutableLeafStrategy, StemStrategy,
};

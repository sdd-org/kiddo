# Kiddo

A high-performance [k-d tree](https://en.wikipedia.org/wiki/K-d_tree) library for exact and approximate nearest-neighbour queries in low-dimensional spaces.

Built with an aggressive focus on query performance, including cache-aware layouts and optional SIMD-accelerated code paths. See the [companion benchmarking site](https://sdd.github.io/kd-tree-comparison-webapp/) to compare Kiddo against other k-d tree implementations across a range of workloads.

- [Crate](https://crates.io/crates/kiddo)
- [Documentation](https://docs.rs/kiddo)
- [Usage](#usage)
- [Examples](https://github.com/sdd/kiddo/blob/master/examples/Readme.md)
- [What's New In v6](#whats-new-in-v6)
- [MSRV](#msrv)
- [Benchmarks](#benchmarks)
- [Change Log](https://github.com/sdd/kiddo/blob/master/CHANGELOG.md)
- [License](#license)

Kiddo v6 provides a single generic [`KdTree`](https://docs.rs/kiddo/latest/kiddo/struct.KdTree.html) that supports floating-point (`f64`, `f32`, `f16`), selected fixed-point (via the `fixed` crate), and unsigned-integer (`u8`, `u16`, `u32`) types as coordinates, along with both mutable and immutable usage patterns.

Kiddo is designed for low-dimensional search problems, especially 2D, 3D, and 4D workloads. Typical use cases include point-cloud analysis, astronomical catalogue crossmatching, colour quantization and palette lookup, local neighbourhood queries in simulations, and other nearest-neighbour and radius-search tasks. Kiddo has been used for diverse geographical and scientific workloads including geocoding, astronomy, cosmology, computer-aided drug discovery, crystallography, and computational neuroscience.

If your points are known up front and the tree will be built once and then queried, start with [`ImmutableKdTree`](https://docs.rs/kiddo/latest/kiddo/type.ImmutableKdTree.html). It offers the best query performance and pairs well with `rkyv` for zero-copy loading of prebuilt trees from disk; when used with memory-mapped files, loading can be effectively instant.

If you need to add or remove points after construction, start with [`MutableKdTree`](https://docs.rs/kiddo/latest/kiddo/type.MutableKdTree.html). Mutable trees remain a good fit for many dynamic workloads, but they do not currently perform dynamic rebalancing, so workloads with substantial growth or heavy churn may benefit from periodic rebuilds.

[`ImmutableKdTree`](https://docs.rs/kiddo/latest/kiddo/type.ImmutableKdTree.html) and [`MutableKdTree`](https://docs.rs/kiddo/latest/kiddo/type.MutableKdTree.html) are convenience aliases for [`KdTree`](https://docs.rs/kiddo/latest/kiddo/struct.KdTree.html) with sensible defaults for these common read-heavy and mutable workloads.

Kiddo is not intended as a library for high-dimensional vector search or feature matching over hundreds or thousands of dimensions, where plain k-d trees are usually the wrong data structure and other approaches are more appropriate. The API does not impose a hard dimensional limit, but Kiddo is primarily intended for low-dimensional workloads.

Kiddo supports the following query types through its fluent builder API, starting from [`KdTree::query`](https://docs.rs/kiddo/latest/kiddo/struct.KdTree.html#method.query):

- `.nearest_one()` finds the single nearest item to a query point.
  Useful for tasks like finding the nearest airport to a given location, or finding the nearest catalogued star to a sky position.

- `.best_n_within(radius, n)` finds the "best" `n` items within a specified distance of a query point, for some definition of "best".
  For example, "give me the 5 largest settlements within 50km of a given point, ordered by descending population", or "the 5 brightest stars within a degree of a point on the sky, ordered brightest first".

- `.nearest_one().approx()` performs approximate nearest-neighbour (ANN) search, returning a good approximate nearest item, often much faster than exact nearest-neighbour search.
  Useful for latency-sensitive workloads like interactive point-cloud picking, or mapping image pixels to a palette colour during colour quantization.

- `.nearest_n(n)` performs k-nearest-neighbour (k-NN) search, finding the `n` nearest items to a query point ordered by distance.
  Useful for finding the nearest weather stations or sensors to a location, or generating candidate correspondences for point-cloud registration.

- `.nearest_n(n).within(radius)` finds up to `n` nearest items within a specified radius of a query point, ordered by distance.
  Useful when you want the closest local neighbours inside a meaningful cutoff, such as the nearest shops within 5 miles, or nearby atoms within an interaction radius.

- `.within(radius)` finds all items within a specified radius of a query point, ordered by distance.
  Useful for radial catalogue searches in astronomy, or collision and proximity queries where the full neighbourhood is needed in sorted order.

- `.within(radius).unsorted()` finds all items within a specified radius of a query point without sorting the results.
  This is often faster than `.within(radius)` when result order does not matter, such as finding all customers within 5 miles of a store, or collecting point-cloud neighbourhoods for clustering or normal estimation.

The builder API also supports boundary exclusivity, result projection, and periodic boundary conditions where applicable. If you can estimate how many matches an eagerly materialized `.within(radius)` query will return, add `.with_result_capacity(estimated_count)` before `.execute()` to avoid result-vector growth reallocations. This is only an allocation hint and does not limit the number of results returned; omit it when no reasonable estimate is available.

## What's New In v6

Version 6 is a substantial API and architecture refresh. The biggest user-facing changes are:

- A single generic [`KdTree`](https://docs.rs/kiddo/latest/kiddo/struct.KdTree.html) now underpins both mutable and immutable usage patterns.
- Queries now go through a fluent builder API starting from [`KdTree::query`](https://docs.rs/kiddo/latest/kiddo/struct.KdTree.html#method.query).
- Query results now use explicit structs such as [`QueryResultItem`](https://docs.rs/kiddo/latest/kiddo/struct.QueryResultItem.html) and [`BestQueryResultItem`](https://docs.rs/kiddo/latest/kiddo/struct.BestQueryResultItem.html).
- Radius-bounded queries support exclusive boundaries.
- Periodic boundary conditions are supported through the query builder.
- Owned trees now support in-place item replacement via `replace_item`.
- Support for `rkyv` 0.7 has been removed; v6 uses `rkyv_08`.

### Migration from v5

The largest source-level breaking changes when moving from v5 are:

- Direct query methods such as `nearest_one`, `nearest_n`, `nearest_n_within`, `within`, and `within_unsorted` have been replaced by builder chains beginning with `.query(&point)`.
- Query results are no longer returned as tuples or `NearestNeighbour` / `BestNeighbour`; use `.distance`, `.item`, and `.point` on `QueryResultItem` / `BestQueryResultItem`.
- `rkyv` support is now exposed through the `rkyv_08` crate feature.
- The old `kiddo::mutable::*` and `kiddo::immutable::*` module-based public API is gone; use [`KdTree`](https://docs.rs/kiddo/latest/kiddo/struct.KdTree.html), [`ImmutableKdTree`](https://docs.rs/kiddo/latest/kiddo/type.ImmutableKdTree.html), and [`MutableKdTree`](https://docs.rs/kiddo/latest/kiddo/type.MutableKdTree.html) from the crate root.

In practice, code like this in v5:

```rust
let result = kdtree.nearest_one::<SquaredEuclidean<f64>>(&query);
```

becomes this in v6:

```rust
let result = kdtree
    .query(&query)
    .nearest_one::<SquaredEuclidean<f64>>()
    .execute();
```

## Usage

Add `kiddo` to `Cargo.toml`

```toml
[dependencies]
kiddo = "6.0.0-alpha.4"
```

Add points to a k-d tree and query the nearest points with a distance metric:

```rust
use std::num::NonZero;

use kiddo::{ImmutableKdTree, QueryResultItem, SquaredEuclidean};

let entries = vec![
    [0f64, 0f64],
    [1f64, 1f64],
    [2f64, 2f64],
    [3f64, 3f64]
];

let kdtree = ImmutableKdTree::new_from_slice(&entries).unwrap();

// How many items are in tree?
assert_eq!(kdtree.size(), 4);

// Find the nearest item to [0f64, 0f64].
let nearest = kdtree
    .query(&[0f64, 0f64])
    .nearest_one::<SquaredEuclidean<f64>>()
    .execute();

assert_eq!(nearest.distance, 0f64);
assert_eq!(nearest.item, 0);

assert_eq!(
    kdtree
        .query(&[0f64, 0f64])
        .nearest_n::<SquaredEuclidean<f64>>(NonZero::new(3usize).unwrap())
        .execute(),
    vec![
        QueryResultItem { point: (), distance: 0f64, item: 0 },
        QueryResultItem { point: (), distance: 2f64, item: 1 },
        QueryResultItem { point: (), distance: 8f64, item: 2 }
    ]
);
```

See the [examples documentation](https://github.com/sdd/kiddo/tree/master/examples) for some more detailed examples.

The same builder can be used to select unsorted range queries, approximate nearest-neighbour search, exclusive radius boundaries, result projection, and periodic boundary conditions. For example:

```rust
use std::num::NonZero;

use kiddo::{ImmutableKdTree, SquaredEuclidean};

let entries = vec![
    [0.05f64, 0.50],
    [0.95f64, 0.50],
    [0.40f64, 0.40],
];

let kdtree = ImmutableKdTree::new_from_slice(&entries).unwrap();
let box_size = [1.0f64, 1.0];

let wrapped = kdtree
    .query(&[0.99, 0.50])
    .periodic_boundary_condition(&box_size)
    .nearest_one::<SquaredEuclidean<f64>>()
    .execute();

assert_eq!(wrapped.item, 1);

let neighbours = kdtree
    .query(&[0.99, 0.50])
    .periodic_boundary_condition(&box_size)
    .nearest_n::<SquaredEuclidean<f64>>(NonZero::new(2).unwrap())
    .within(0.10)
    .execute();

assert_eq!(neighbours.len(), 2);
```

## Optional Features

Kiddo exposes a number of optional crate features:

- `fixed` enables support for fixed-point coordinate types from the [`fixed`](https://docs.rs/fixed/latest/fixed) crate.

- `f16` enables support for half-precision floating-point coordinates via the [`half`](https://docs.rs/half/latest/half) crate.

- `serde` enables serialization and deserialization via [`Serde`](https://docs.rs/serde/latest/serde/).

- `rkyv_08` enables zero-copy serialization and deserialization via [`rkyv`](https://docs.rs/rkyv/latest/rkyv/) 0.8.x. This is particularly useful for prebuilt immutable trees that you want to load very quickly, especially in conjunction with memory-mapped files.

- `simd` **(NIGHTLY)** enables handwritten SIMD and prefetch intrinsics for additional performance where available. This requires a nightly Rust toolchain.

- `huge_pages` enables Linux-specific huge-page advice helpers for owned and archived tree storage.

- `leaf_nta_prefetch` enables additional non-temporal leaf prefetch hints in some query paths. This is an advanced tuning feature and is only useful in specific workloads.

Kiddo also contains a number of additional feature flags used for internal experimentation, benchmarking, simulation, and specialized tuning. Most users will not need them.

**NOTE**: Support for rkyv 0.7 was removed in Kiddo v6.

## v5.x

Version 5 bundled a complete re-write of [`ImmutableKdTree`](https://docs.rs/kiddo/5.0.0/kiddo/type.ImmutableKdTree.html) alongside some rationalization of feature names and a change of type of the `max_qty` parameter present in some query methods from `usize` to `NonZero<usize>`.

### `ImmutableKdTree` rewrite

Many people had previously unsuccessfully tried to use [`ImmutableKdTree`](https://docs.rs/kiddo/5.0.0/kiddo/type.ImmutableKdTree.html) with data containing many points that had the same value on one or more of their axes, for example point cloud data containing many points on a flat axis-aligned plane.
The v5 rewrite of [`ImmutableKdTree`](https://docs.rs/kiddo/5.0.0/kiddo/type.ImmutableKdTree.html) experienced none of these kinds of problems and could be safely used no matter what your data looked like.
Query performance was in many cases faster than the prior version, but sometimes slightly slower - your mileage may vary but differences in query performance were pretty small.
Construction performance was considerably improved, with up to a 2x speedup, with the improvement becoming more pronounced as the tree size increased.
Memory efficiency was slightly better also.

### Modified van Emde Boas Stem Ordering

The experimental `modified_van_emde_boas` feature allowed an alternative stem node ordering mode to be enabled. This mode was more cache-friendly. Under the standard Eytzinger ordering, a new cache line would be fetched for almost every level traversed within the stem nodes beyond the third level. The Modified van Emde Boas ordering was more cache efficient - meaning that on CPU architectures with a 64-byte cache line (ie almost all of them in servers, desktops and laptops), a cache line needed fetching only once every 3 stem levels for f64, and every 4 levels for f32.
On architectures with a 128-byte cache lines (some Apple M3 and newer at the moment), this is every 4 levels for f64 and every 5 levels for f32.
The downside was that logic to calculate the next stem index was significantly more complex than with the Eytzinger layout, requiring around 10 integer ops (one being a divide) vs just one integer op (a shift) for Eytzinger.
At the time, performance when using `modified_van_emde_boas` was between 1% faster and 5% slower than standard, at least on the machines that I'd tested it on.
I'd have loved to hear how it fared on a machine with a 128-byte cache line width, if anyone cared to try it. I continued to work on the performance of this and perhaps one day it may end up faster than Eytzinger if I can optimise the logic well enough - the initial implementation required 24 operations, so progress had been made.

### Feature name changes

It was pointed out in https://github.com/sdd/kiddo/issues/159 that it was necessary to anable both `rkyv` and `serialize_rkyv` features to use Rkyv serialization. I took the opportunity of the major version bump to rationalize the feature names to make them easier to use.
`serialize_rkyv` has been removed and now only `rkyv` feature is needed to enable Rkyv serialization.
`serialize` has been renamed to `serde` in line with ecosystem conventions.
`half` has been renamed to `f16` for clarity.

### `max_qty` Changed to `NonZero<usize>`

It was noted by [@ezrasingh](https://github.com/sdd/kiddo/issues/168#issuecomment-2335183999) that specifying `max_qty` as zero in version 4.2.1 alongside `sorted = false` resulted in a panic. Since requesting a `max_qty` of zero makes no sense, and to avoid adding a run-time check, the type of `max_qty` has been changed to `NonZero<usize>` to make this a compile-time check instead.

### `ImmutableKdTree` + `rkyv`

The v5 [`ImmutableKdTree`](https://docs.rs/kiddo/5.0.0/kiddo/type.ImmutableKdTree.html) used an Aligned Vec internally for storing stem nodes. It was not possible to zero-copy deserialize
into an Aligned Vec with `rkyv` as there is no guarantee that the stem vec in the underlying buffer respects the alignment.
As such, unfortunately this meant that [`ImmutableKdTree`](https://docs.rs/kiddo/5.0.0/kiddo/type.ImmutableKdTree.html) itself couldn't be fully zero-copy serialized / deserialized, but there
were some related types that were provided that allowed zero-copy deserialization to be performed for all other parts of the tree
except for the stems, which themselves got copied into an aligned array from the buffer.
In practice this was still very fast as the stems were only a very small part of the overall tree.

See `immutable-rkyv-serialize` and `immutable-rkyv-deserialize` in the examples for how to do this.

## v3.x

Version 3.x changed the distance metrics syntax, switching from function pointers to a trait-based
approach that permitted some ergonomics and performance improvements. This was a breaking change though:
whereas prior to v3, you may have had queries that looked like this:

```
use kiddo::distance::squared_euclidean;
let result = kdtree.nearest_one(&[0f64, 0f64], &squared_euclidean);
```

For v3 onwards, you needed to switch to this syntax:

```
use kiddo::SquaredEuclidean;
let result = kdtree.nearest_one::<SquaredEuclidean>(&[0f64, 0f64]);
```

V3 also introduced the [`ImmutableKdTree`](https://docs.rs/kiddo/3.0.0/kiddo/type.ImmutableKdTree.html) variant. It was designed for use cases where all the points that you needed to add
to the tree were known up-front, and no modifications needed to be made after the tree was initially populated.
[`ImmutableKdTree`](https://docs.rs/kiddo/3.0.0/kiddo/type.ImmutableKdTree.html) balanced and optimised the tree at construction time, ensuring much more efficient
memory usage (and a correspondingly smaller size on-disk for serialized trees). Since the interior
nodes of the [`ImmutableKdTree`](https://docs.rs/kiddo/3.0.0/kiddo/type.ImmutableKdTree.html) also took up less space in memory, more of them could fit in the CPU cache, potentially
improving performance in some cases.

## v2.x

Version 2.x was a complete rewrite, providing:

- a new internal architecture for **much-improved performance**;
- Added **integer / fixed point support** via the [`Fixed`](https://docs.rs/fixed/latest/fixed/) library;
- **instant zero-copy deserialization** and serialization via [`Rkyv`](https://docs.rs/rkyv/latest/rkyv/) ([`Serde`](https://docs.rs/serde/latest/serde/) still available).
- See the [changelog](https://github.com/sdd/kiddo/blob/master/CHANGELOG.md) for a detailed run-down on all the changes made in v2.

## Benchmarks

The results of all the below benchmarks are viewable in an interactive webapp over at [https://sdd.github.io/kd-tree-comparison-webapp/](https://sdd.github.io/kd-tree-comparison-webapp/).

The comparative benchmark suite is located in another project, [https://github.com/sdd/kd-tree-comparison](https://github.com/sdd/kd-tree-comparison).

Criterion was used to perform a series of benchmarks. We compare Kiddo v3 to:

- Kiddo v2.x
- [Kiddo v1.x / v0.2.x](https://github.com/sdd/kiddo_v1)
- [FNNTW](https://crates.io/crates/fnntw) v0.2.3
- [nabo-rs](https://crates.io/crates/nabo) v0.2.1
- [pykdtree](https://github.com/storpipfugl/pykdtree) v1.3.4
- [sklearn.neighbours.KDTree](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html) v1.2.2
- [scipy.spatial.KDTree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html) v1.10.1

The following activities are benchmarked (where implemented):

- Construction of a k-d tree from a list of points and indexes
- Querying the nearest one, ten, or one hundred points to a given query point
- Querying all points within a set radius of a given point (both unsorted results, and results sorted by distance)
- Querying the nearest n items within a specified radius (sorted and unsorted)
- Each action is benchmarked against trees that contain 100, 1,000, 10,000, 100,000, 1,000,000 and in some cases 10,000,000 nodes.

The benchmarks are repeated against 2d, 3d and 4d trees, as well as with points that are both of type `f32` and of type `f64`, as well as a 16-bit fixed point use case for Kiddo v2.

The trees are populated with random source data whose points are all on a unit sphere. This use case is representative of common k-d tree usages in geospatial and astronomical contexts.

## MSRV

Kiddo v6's current minimum supported Rust version (MSRV) is **1.89.0** (**1.85.0** for `v5.x.x`).

Kiddo will aim to support at least **N-4** stable Rust releases, which is roughly six months of stable compiler history, when used with the **default crate features**.

Kiddo will also endeavour to increase MSRV only when doing so would provide a material improvement for users, rather than simply for the sake of using a newer compiler.

Optional features may require a newer toolchain than the default-feature MSRV if their dependency stack requires it. The `simd` feature is nightly-only and is outside the stable MSRV policy.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Contributing

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

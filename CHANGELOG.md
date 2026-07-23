# Kiddo Changelog

## [6.0.0-alpha.4] - 2026-07-23

### ✨ Features

- Add result capacity hint for radius queries ([@sdd](https://github.com/sdd))

### 🧪 Testing

- Add result collection threshold profiler ([@sdd](https://github.com/sdd))

- Add v6 release parity benchmark suites ([@sdd](https://github.com/sdd))

### 🤖 CI

- Add stem strategy benchmark variant ([@sdd](https://github.com/sdd))

- Rename basic benchmark variant ([@sdd](https://github.com/sdd))

- Pass benchmark features explicitly ([@sdd](https://github.com/sdd))

- Cap leaf benchmark tree size ([@sdd](https://github.com/sdd))

- Fix leaf benchmark export filter ([@sdd](https://github.com/sdd))

- Cap benchmark trees at 2^25 ([@sdd](https://github.com/sdd))

- Fix benchmark v5-v6 chart matching ([@sdd](https://github.com/sdd))

## [6.0.0-alpha.3] - 2026-07-21

### ✨ Features

- Make local query scratch the default ([@sdd](https://github.com/sdd))

### 🐛 Bug Fixes

- Remove broken dot product dist metric and associated code ([@sdd](https://github.com/sdd))

### ⚡️ Performance

- Alternate path to avoid wasted calcs on descent ([@sdd](https://github.com/sdd))

- Add IS_SIGNED assoc value to Axis ([@sdd](https://github.com/sdd))

- Dist1 on Manhattan and Chebyshev use saturating_dist ([@sdd](https://github.com/sdd))

- Improved offset update for Chebyshev and Manhattah ([@sdd](https://github.com/sdd))

- Use fused linear insertion for threshold vec results ([@sdd](https://github.com/sdd))

- Tune sorted and unsorted threshold vec limits ([@sdd](https://github.com/sdd))

### 🤖 CI

- Harden pre-release string updater workflow against no changes ([@sdd](https://github.com/sdd))

- Publish custom benchmark reports ([@sdd](https://github.com/sdd))

- Add bench chart justfile tasks ([@sdd](https://github.com/sdd))

- Fix benchmark workflow bootstrap ([@sdd](https://github.com/sdd))

- Derive benchmark key without just ([@sdd](https://github.com/sdd))

- Pass benchmark args to just correctly ([@sdd](https://github.com/sdd))

- Pass benchmark recipe arguments positionally ([@sdd](https://github.com/sdd))

- Authenticate initial benchmark pages push ([@sdd](https://github.com/sdd))

- Rank featured chart by relative change ([@sdd](https://github.com/sdd))

- Add distance metric ISA matrix ([@sdd](https://github.com/sdd))

- Fix distance metric ISA benchmark builds ([@sdd](https://github.com/sdd))

- Simplify benchmark workflows ([@sdd](https://github.com/sdd))

- Fix benchmark workflow shellcheck ([@sdd](https://github.com/sdd))

- Suggest benchmarks for performance-sensitive PRs ([@sdd](https://github.com/sdd))

- Use heuristic benchmark suggestions ([@sdd](https://github.com/sdd))

- Allow benchmark suggestion comment updates ([@sdd](https://github.com/sdd))

- Add leaf strategy benchmark variant ([@sdd](https://github.com/sdd))

- Allow org members to trigger benchmark runs ([@sdd](https://github.com/sdd))

- Clarify benchmark run names ([@sdd](https://github.com/sdd))

- Split nightly debug and release tests ([@sdd](https://github.com/sdd))

### 🧹 Chore

- Bump actions/setup-node from 6 to 7 ([@dependabot[bot]](https://github.com/dependabot[bot]), Signed-off-by:dependabot[bot] <support@github.com>)

- Update las requirement in the cargo-dependencies group ([@dependabot[bot]](https://github.com/dependabot[bot]), Signed-off-by:dependabot[bot] <support@github.com>)

## [6.0.0-alpha.2] - 2026-07-17

### ✨ Features

- Configurable scratch location ([@sdd](https://github.com/sdd))

- Introduce QueryMetric public trait ([@sdd](https://github.com/sdd), Fixes:[Issue #390](https://github.com/sdd/kiddo/issues/390))

- Ensure third-party leaf strategies can be used ([@sdd](https://github.com/sdd))

- Add example showing how to embed a kd-tree ([@sdd](https://github.com/sdd))

- Add ThresholdVecResultCollection for small-k `nearest_n` ([@cbueth](https://github.com/cbueth))

### 🐛 Bug Fixes

- Add dafault impl of LeafStrategy::new_with_empty_leaf ([@sdd](https://github.com/sdd))

- Cache `threshold_distance` in `ThresholdVecResultCollection` ([@cbueth](https://github.com/cbueth))

- Apply `ThresholdVecResultCollection` to scratch-based `nearest_n` path ([@cbueth](https://github.com/cbueth))

- Dispatch `into_sorted_vec()` through `into_vec()` in ThresholdVec ([@cbueth](https://github.com/cbueth))

- Extend `ThresholdVec` optimisation to unsorted `nearest_n` path ([@cbueth](https://github.com/cbueth))

### ⚡️ Performance

- Pre-allocate unsorted result Vec with capacity 64 ([@cbueth](https://github.com/cbueth))

- Use `select_nth_unstable` extraction in `ThresholdVecResultCollection` ([@cbueth](https://github.com/cbueth))

- Hybrid sorted-vec result collection for small-k `nearest_n` ([@cbueth](https://github.com/cbueth))

### 🧪 Testing

- Add regression test to ensure KdTree::default works ([@sdd](https://github.com/sdd))

- Add k=21 nearest_n_within case for BinaryHeap coverage, remove padding test ([@cbueth](https://github.com/cbueth))

### 🤖 CI

- Prepare embedded example artifacts in CI ([@sdd](https://github.com/sdd))

- Add bencher for master push and non-fork PRs ([@sdd](https://github.com/sdd))

- Run codspeed in simulation mode only ([@sdd](https://github.com/sdd))

- Install cmake for bencher runner ([@sdd](https://github.com/sdd))

- Add eytzinger nearest_n bencher profiles ([@sdd](https://github.com/sdd))

- Allow bencher runs for collaborators or manually permissioned forks ([@sdd](https://github.com/sdd))

- Switch to standard collab model now that repo is in a personal org ([@sdd](https://github.com/sdd))

- Ensure examples/data is .gitignored ([@sdd](https://github.com/sdd))

- Ensure git-cliff emits well-formatted md ([@sdd](https://github.com/sdd))

- Ensure git-cliff emits well-formatted md ([@sdd](https://github.com/sdd))

- Ensure git-cliff emits well-formatted md ([@sdd](https://github.com/sdd))

- Git-cliff nicer formatting ([@sdd](https://github.com/sdd))

- Git-cliff fix formatting yet again ([@sdd](https://github.com/sdd))

### 🧹 Chore

- Bump LoliGothick/clippy-check ([@dependabot[bot]](https://github.com/dependabot[bot]), Signed-off-by:dependabot[bot] <support@github.com>)

- Migrate config renovate.json ([@renovate[bot]](https://github.com/renovate[bot]))

- Move binaries to be examples or benches to avoid confusion in the crates.io page ([@sdd](https://github.com/sdd))

- Bump actions/github-script from 8 to 9 ([@dependabot[bot]](https://github.com/dependabot[bot]), Signed-off-by:dependabot[bot] <support@github.com>)

- Bump LoliGothick/clippy-check ([@dependabot[bot]](https://github.com/dependabot[bot]), Signed-off-by:dependabot[bot] <support@github.com>)

- Update reqwest dep ([@sdd](https://github.com/sdd))

- Update rust crate zip to v8 ([@renovate[bot]](https://github.com/renovate[bot]))

- Update git-cliff config ([@sdd](https://github.com/sdd))

- Workflow to permit /intro comments on release-plz prs ([@sdd](https://github.com/sdd))

- Ensure release-plz and intro-comment format changelog ([@sdd](https://github.com/sdd))

## [6.0.0-alpha.1] - 2026-06-30

Almost a year in the making and counting, Kiddo v6 is effectively a full rewrite, addressing some
long-standing issues.

- V6 represents a fundamental shift to a unified single `KdTree` struct, replacing the previous
  mutable/immutable and float/fixed splits. The need for separate mutable / immutable trees has
  been removed by introducing the `LeafStrategy` trait, which the `KdTree` has as a generic parameter.
  LeafStrategies can be mutable or immutable.
- `KdTree` is also now generic over the new `StemStrategy` trait too. The combination of these two
  orthogonal traits allow experimentation and selection of alternative stem layouts and traversal
  mechanisms, alongside configurable leaf strategies, for performance experimentation.
- The query API has been refactored from a handful of separate methods that were not particularly
  cohesive into a single unified fluent builder API. The builder approach is much more orthogonal,
  and protects against breaking changes when new options are introduced in the future. It permits
  constraints to be encoded in the type system for impossible option combinations, prevents the need
  for as many methods that differ only by some combination of features; allowing for configurable
  result type projection, query types, and query configurations with defaults that match the v5-era
  separate methods.
- Some of the other new features added include configurable boundary-inclusivity, periodic boundary
  conditions, the `within_unsorted_visit` result mode to avoid materialization of results, `TryFrom`
  for converting between different `KdTree` types, `new_from_source`, and `replace_item`.
- Distance metric coverage has also been expanded and tidied up, including the addition of Chebyshev
  and Minkowski metrics and support for queries whose result type is wider than the
  stored coordinate type, which is especially important for fixed-point trees.
- With regard to the Stem Strategy trait system, alongside the Eytzinger ordering from v5 and prior
  is still available but now supports configurable prefetch behaviour. I've introduced the new
  Donnelly stem ordering via a family of stem strategies, featuring configurable pre-fetch and block size,
  unrolled traversal, SIMD descent, and full SIMD pruning and backtracking variants. The SIMD variant
  contains custom AVX2, AVX512, and NEON kernels.

## [5.3.2] - 2026-06-02

### Ci

- Backport workflow and repo config updates
- Backport build and coverage workflow updates

### 🐛 Bug Fixes

- Add msrv, use alernative to f64::fract

### 💄 Styling

- Formatting and clean up unneeded md file

## Unreleased

I've kept back these changes for now as, whilst extremely welcome, the change to the return type of `size()` would be breaking.
I'm hoping to have v6 available soon, which will include both these changes as well as addressing quite a few other long-running feature requests.

### ✨ Features

- Improve the flexibility of T type, KdTree.size. Add generate nearest_one_point (https://github.com/sdd/kiddo/pull/214, @pjkundert)
- Fix broken test of custom struct as T. Add test for `()` as T (https://github.com/sdd/kiddo/pull/243 @pjkundert)

## [5.3.1] - 2026-04-15

### 🐛 Bug Fixes

- Add a compile-time assertion to prevent bucket size of less than 2 for `KdTree` to avoid UB when trying to split a bucket of size 1.
  Fixes https://github.com/sdd/kiddo/issues/295. (Note that even though bucket sizes of 2 are possible, I would generally not recommend
  using a value of B below 32 anyway, for best performance.)

## [5.3.0] - 2026-03-19

I'm extremely grateful to @cbueth for his fantastic set of contributions to this release. The new distance metrics are a great addition to the library,
come with extensive tests, and he was even able to contribute a big fix and some welcome refactors along the way. Thanks very much, Carlson!

### ✨ Features

- Add Chebyshev (L_∞ norm) distance metric (https://github.com/sdd/kiddo/pull/290, @cbueth)
- Add Generalised Minkowski (L_p norm) distance metric (https://github.com/sdd/kiddo/pull/291, @cbueth)
- Add `*_exclusive` methods for querying with an exclusive (`<`) rather than inclusive (`<=`) boundary check (https://github.com/sdd/kiddo/pull/294, @cbueth)

### 🐛 Bug Fixes

- Ensure that all existing query methods have boundary checks that are inclusive (`<=`), in line with typical k-d tree expectations (@cbueth)
  If you were relying on these checks being exclusive, switch over to using the new `*_exclusive` variants of the query methods.

### Docs

- Correct a mistake in the docs for `fixed::distance::Manhattan` (https://github.com/sdd/kiddo/pull/283, @Luca-spopo)

### Deps

- update cmov dep from 0.4 to 0.5 (@sdd)

## [5.2.4] - 2026-01-01 (Happy New Year! 🎉)

### Deps

- update cmov dep from 0.3 to 0.4 after 0.3 got yanked (see https://github.com/RustCrypto/utils/issues/1304). Thanks @yuby and @jqnatividad

## [5.2.3] - 2025-12-08

### 🐛 Bug Fixes

- Correct slice access in remainder processing and remove unsafe (@MarkusZoppelt)
- Use `try_from()` with error for `leaf_items.len()` (@MarkusZoppelt)

### ♻️ Refactor

- within_unsorted_iter no longer uses a generator (@KvA2KLvAST)
- Remove needless SubAssign trait bound from Content trait

### Deps

- Remove doc-comment dependency and use doc attribute that was added in Rust 1.54 instead (@jqnatividad)
- Use `doc` attribute instead of `doc_comment!` (@jqnatividad)
- Update actions/checkout action to v6
- Update codspeedhq/action action to v4
- Update ad-m/github-push-action action to v1
- Update rust crate rstest to 0.26
- Update rust crate codspeed-criterion-compat to v4

### Ci

- Update CI workflow triggers to include PR and workflow_dispatch
- Permit coverage to run for PRs as well
- Fix release-plz and add commitlint

### 💄 Styling

- Remove unnecessary parentheses
- Fix formatting
- Fix some lint issues

### 🧪 Testing

- Add regression test for remainder slice access bug (@MarkusZoppelt)

## [5.2.2] - 2025-12-08

### 🐛 Bug Fixes

- Correct slice access in remainder processing and remove unsafe (@MarkusZoppelt)
- Use `try_from()` with error for `leaf_items.len()` (@MarkusZoppelt)

### ♻️ Refactor

- within_unsorted_iter no longer uses a generator (@KvA2KLvAST)
- Remove needless SubAssign trait bound from Content trait

### Deps

- Remove doc-comment dependency and use doc attribute that was added in Rust 1.54 instead (@jqnatividad)
- Use `doc` attribute instead of `doc_comment!` (@jqnatividad)
- Update actions/checkout action to v6
- Update codspeedhq/action action to v4
- Update ad-m/github-push-action action to v1
- Update rust crate rstest to 0.26
- Update rust crate codspeed-criterion-compat to v4

### Ci

- Update CI workflow triggers to include PR and workflow_dispatch
- Permit coverage to run for PRs as well
- Fix release-plz and add commitlint

### 💄 Styling

- Remove unnecessary parentheses
- Fix formatting
- Fix some lint issues

### 🧪 Testing

- Add regression test for remainder slice access bug, Signed-off-by:Markus Zoppelt <markus.zoppelt@helsing.ai>

## [5.2.1] - 2025-06-29

### 📝 Documentation

- Update some stale documentation. Remove the global_allocate feature which is no longer used for anything

## [5.2.0] - 2025-06-29

It's been a while since the last release as my focus has been elsewhere, but I'm back in the k-d tree groove now
and looking to bring some new features and performance updates over the next few weeks.

The major addition in this release is support for version 0.8 of Rkyv. Rkyv 0.8 is almost a complete rewrite compared to 0.7,
and so this required quite a lot of changes.

Right now, the pre-existing `rkyv` crate feature still provides support for
Rkyv 0.7 as before, and so the introduction of Rkyv 0.8 is non-breaking. To use Rkyv 0.8, enable the crate feature that is
unsurprisingly named `rkyv_08`. There are some caveats to Rkyv 0.8 support:

- The rkyv 0.8.x Archived structs are prefixed with `ArchivedR8` rather than `Archived` to avoid clashing with the pre-existing
  rkyv 0.7 types.
- You're unlikely to be able to successfully deserialize a tree serialized with Rkyv 0.7 into an Rkyv 0.8 object, and vice-versa.
- Anyone using `rkyv` and `f16` / `half` at the same time may encounter issues updating to Rkyv 0.8. This is because the `half`
  crate only supports Rkyv 0.7 at version 2.4.1 and below, and only supports Rkyv 0.8 at versions
  2.5.0+.
- Supporting both Rkyv 0.7 and 0.8 in the Kiddo crate at the same time is a massive pain and will add significant maintenance burden
  going forward. As such, **this 5.x.x branch will be the only one that supports both Rkyv 0.7 and Rkyv 0.8. An upcoming version 6
  of Kiddo will remove support for Rkyv 0.7 entirely and only support Rkyv 0.8.** At this point the annoying non-standard `ArchivedR8`
  prefix for the rkyv 0.8 types will be dropped in favour of the default naming scheme.
-

### ✨ Features

- Add support for Rkyv v0.8
- Gate `fixed` support behind a crate feature to reduce compile times
- Preallocate binary heap capacity

### 🧪 Testing

- Fixup rkyv doctests

### Ci

- Update github actions to exercise `rkyv_08` feature
- Update renovate bot config so that it ignores deps that are pined to old versions for legacy feature compatibility

### Chore

- address spacing causing clippy lint failure
- Update rust crate rstest to 0.25
- Convert doctest file load `unwrap`s to `expect`s to help track down when files are missing
- Clean up unused deps and move some deps to dev-deps that did not need to be full deps
- Update to latest versions of rand crates
- Update to latest version of criterion
- Update bincode dev dep to v2 for serde example
- Update `itertools` dependency to 0.14
- Update formatting to comply with updated clippy rules of recent versions of Rust
- Update rust crate ordered-float to v5

All the best! Scott (@sdd)

## [5.1.0] - 2025-06-20: **YANKED**

5.1.0 was yanked. It only supported Rkyv 0.7 and 0.8 in a mutually exclusive way, but I found a way to support both simultaneously
a few days after publishing 5.1.0. This came at the expense of being a breaking change to 5.1.0, and to respect semver I'd have needed
to release a 6.0.0 version that would have been immediately superseded by 7.0.0 that removed rkyv 0.7. Since 5.1.0 had only been out
for a couple of days, and since the mutual exclusivity meant that the rkyv 0.8 structs were not even mentioned in the 5.1.0
docs on docs.rs, I chose to yank 5.1.0 instead.

## [5.0.3] - 2024-12-21

### Deps

- Update generator dependency to 0.8.4, Fixes:https://github.com/sdd/kiddo/issues/182

## [5.0.2] - 2024-12-15

### 🐛 Bug Fixes

- Disable broken get_best_from_dists_f64_avx2 until fixed
- only export WithinUnsortedIter on archs where the within_unsorted_iter mod is configured (https://github.com/sdd/kiddo/issues/190)
- fix: consistently use CACHELINE_ALIGN (https://github.com/sdd/kiddo/issues/190)

## [5.0.1] - 2024-12-08

### Performance

- fix a performance regression on the immutable tree.

### Documentation

- ensure that the top-level documentation example shows v5 being imported.

## [5.0.0] - 2024-11-30

Version 5 bundles a complete re-write of [`ImmutableKdTree`](`immutable::float::kdtree::ImmutableKdTree`) alongside some rationalization of feature names and a change of type of the `max_qty` parameter present in some query methods from `usize` to `NonZero<usize>`.

### `ImmutableKdTree` rewrite

**BREAKING CHANGE**: For anyone that has been serializing `ImmutableKdTree` (using either `serde` or `rkyv`), version 5 constitutes a breaking change as serialized trees from prior versions will not be deserializable with v5 and vice-versa.

Quite a few people (https://github.com/sdd/kiddo/issues/172, https://github.com/sdd/kiddo/issues/158, https://github.com/sdd/kiddo/issues/78) have previously unsuccessfully tried to use `ImmutableKdTree` with data containing many points that have the same value on one or more of their axes, for example point cloud data containing many points on a flat axis-aligned plane.
The v5 rewrite of `ImmutableKdTree` experiences none of these kinds of problems and can be safely used no matter what your data looks like.
Query performance is in many cases faster than the prior version, but sometimes slightly slower - your mileage may vary but differences in query performance is pretty small.
Construction performance is considerably improved, with up to a 2x speedup, with the improvement becoming more pronounced as the tree size increases.
Memory efficiency is slightly better also.

Behind-the-scenes, the structure of the `ImmutableKdTree` has changed from using a Vec of fixed-size array-based buckets to using a single array-of-vecs to store all the points, with per-bucket offsets being stored for each leaf. To avoid dynamic allocation at query time, a fixed slice that chunks the bucket is used, permitting autovectorisation to work well and giving the opportunity for manual SIMD to be used on the fixed-length slice. Trailing values beyond the last full slice are processed individually.

### Modified van Emde Boas Stem Ordering

The experimental `modified_van_emde_boas` feature allows an alternative stem node ordering mode to be enabled.
When enabled, the ordering of stem nodes changes from using Eytzinger ordering to a modified van Emde Boas (can I call this a Donnelly ordering? :-p) order.
This is a novel implementation unique to Kiddo v5 that ensures that a cache line only needs to be retrieved at most once every three levels (on most CPUs when using f64), or every four levels (on most CPUs when using f32). This increases by an extra one level on CPU architectures with a 128-byte cache line width (this is quite rare at the moment but can be found on some Apple M3 and newer CPUs).
Previous literature has indicated that a standard van Emde Boas layout provided no advantage, but thanks to an efficient branchless implementation of the stem ordering logic, and a refinement to leave the last slot on each cache line empty, rather than straddling levels across cache lines, cache efficiency is improved to the point where gains can sometimes be seen over the previously-best Eytzinger layout.
Typically, performance varies from between 1% faster an 5% slower than Eytzinger, from what I've seen during testing, with the differences often being statistically insignificant.

### `ImmutableKdTree` + `rkyv`

The v5 `ImmutableKdTree` uses an Aligned Vec internally for storing stem nodes. It is not possible to zero-copy deserialize
into an Aligned Vec with `rkyv` as there is no guarantee that the stem vec in the underlying buffer respects the alignment.
As such, unfortunately this means that `ImmutableKdTree` itself can't be fully zero-copy serialized / deserialized, but there
are some related types that are provided that allow zero-copy deserialization to be performed for all other parts of the tree
except for the stems, which themselves get copied into an aligned array from the buffer.
In practice this is still very fast as the stems are only a very small part of the overall tree.

See `immutable-rkyv-serialize` and `immutable-rkyv-deserialize` in the examples for how to do this.

### Feature name changes

**BREAKING CHANGE**: It was pointed out in https://github.com/sdd/kiddo/issues/159 that it was necessary to enable both `rkyv` and `serialize_rkyv` features to use Rkyv serialization. I took the opportunity of the major version bump to rationalize the feature names to make them easier to use.
`serialize_rkyv` has been removed and now only `rkyv` feature is needed to enable Rkyv serialization.
`serialize` has been renamed to `serde` in line with ecosystem conventions.
`half` has been renamed to `f16` for clarity (but is not needed for `f16` support anyway and is only used to ensure that the `half` crate is only
depended upon within the "half" examples and not as a core dependency)

### `max_qty` Changed to `NonZero<usize>`

**BREAKING CHANGE**: It was noted by [@ezrasingh](https://github.com/sdd/kiddo/issues/168#issuecomment-2335183999) that specifying `max_qty` as `0` in version 4.2.1 alongside `sorted = false` resulted in a panic. Since requesting a `max_qty` of zero makes no sense, and to avoid adding a run-time check for users who have no possibility of specifying a `max_qty` of `0`, the type of `max_qty` has been changed to `NonZero<usize>` to make this a compile-time check instead.

## [4.2.1] - 2024-08-17

### Chore

- Refactor trait bounds to silence new clippy lints
- Silence new lint error
- Silence error regarding doc_cfg no longer working
- Update codspeedhq/action action to v3
- Update rust crate generator to 0.8.1

### 🐛 Bug Fixes

- Nearest_n_within does not limit num of items when not sorted, Issue:https://github.com/sdd/kiddo/issues/168
- Update rust crate itertools to 0.13

## [4.2.0] - 2024-02-18

### ✨ Features

- Add f16 support, example and docs to show usage with half crate

### 🐛 Bug Fixes

- Prevent assertion failure when stem optimisation needs a large shift

## [4.1.1] - 2024-02-17

### 🐛 Bug Fixes

- Prevent overflow in capacity_with_bucket_size on non-64 bit architectures

## [4.1.0] - 2024-02-17

### Chore

- Remove unused import
- Update actions/cache action to v4
- Update codecov/codecov-action action to v4
- Update baptiste0928/cargo-install action to v3

### Ci

- Fix issue with CI lint steps

### Deps

- Relax strictness of some deps to reduce renovate noise

### ♻️ Refactor

- Remove need for gated import

### ✨ Features

- Make tracing an optional dependency gated by tracing feature flag
- Iterate over trees
- Make rand and rayon optional

### 🐛 Bug Fixes

- Stdsimd removed from unstable, fix:fixup simd removal
- Add missing global_allocate feature definition and sort feature defs alphabetically
- Update benches to require test_utils feature. update clippy and test steps to include new test_utils feature
- Claytonwramsey bug, Fixes:[#138](https://github.com/sdd/kiddo/pull/138)

### 🧪 Testing

- Fix iter doctests and remove unused var
- Add hacky workaround to enable tests to run without having to specify --features=test_utils

## [4.0.0] - 2023-12-04

Despite the major version bump, this is unlikely to be a breaking change for any users. The `within_unsorted_iter` method of `ImmutableKdTree` is now only present on x86_64 and Aarch64 targets.
Considering that v3.0.0 would not even compile on these targets when the `immutable` crate feature was activated,
it seems vanishingly unlikely that this breaks anyone.
Additionally, the `immutable` feature has been removed and the `global_allocate` feature added. If you were using `ImmutableKdTree` and your build
breaks because the `immutable` feature does not exist - don't worry, you don't need it anymore.
Simply remove any reference to it ant the `ImmutableKdTree` should be available without it.

### ✨ Features

- `ImmutableKdTree` now works on stable

### 🐛 Bug Fixes

- Update rust crate itertools to 0.12
- Re-enable support for wasm targets

### 📝 Documentation

- Update feature docs in lib.rs

## [3.0.0] - 2023-11-05

I can't believe how long it has taken me to get v3 into shape, but it's finally here! :tada:

The [ImmutableKdTree](https://docs.rs/kiddo/3.0.0/kiddo/immutable/float/kdtree/struct.ImmutableKdTree.html) is finally ready! :tada: Designed for use cases where all the points that you need to add
to the tree are known up-front, and no modifications need to be made after the tree is initially populated.
`ImmutableKdTree` balances and optimises the tree at construction time, ensuring much more efficient
memory usage (and a correspondingly smaller size on-disk for serialized trees). Since the interior
nodes of the `ImmutableKdTree` also take up less space in memory, more of them can fit in the CPU cache, potentially
improving performance in some cases.

The `immutable` crate feature needs to be activated in order to use `ImmutableKdTree`.
More info on `ImmutableKdTree` can be found below in the 3.0.0 beta and RC changelog entries.

Version 3.x changes the distance metrics syntax, switching from function pointers to a trait-based
approach that permitted some ergonomics and performance improvements. This is a breaking change though:
whereas prior to v3, you may have had queries that look like this:

```
use kiddo::distance::squared_euclidean;

let result = kdtree.nearest_one(&[0f64, 0f64], &squared_euclidean);
```

Now in v3, you'll need to switch to this syntax:

```
use kiddo::SquaredEuclidean;

let result = kdtree.nearest_one::<SquaredEuclidean>(&[0f64, 0f64]);
```

## [3.0.0-rc.1] - 2023-10-17

### Features

- the `ImmutableKdTree` is now only usable by enabling the `immutable` crate feature. This ensures that the crate as a whole retains compatible with stable rust, as `ImmutableKdTree` depends on some unstable features at present.

### Refactors

- Leaf nodes for Immutable now store their points in columnar format. Searches across them have been re-written to autovectorise better. This has been tested on Compiler Explorer to demonstrate that AVX512 instructions are generated, ensuring vectorization is as wide as is possible. Handwritten SIMD intrinsics have been used (activated by enabling the `simd` crate feature) to manually vectorise code that the compiler could not autovectorise. **NOTE** `simd` is currently quite unstable and not as well tested as the rest of the library, so use it with caution until it stabilizes in the full `v3.0.0` release!

### Style / Tests

- Increase reliability of `within()` test for `ImmutableKdTree`.
- Remove some commented-out code and some useless comments

## [2.1.2] - 2023-10-10

### Fixes

- fix incompatibility with the `num-traits` feature of the `fixed` crate

## [3.0.0-beta.4] - 2023-08-28

### Style / Tests

- Increase reliability of `within()` test for `ImmutableKdTree`.
- Remove some commented-out code and some useless comments in some tests

## [3.0.0-beta.3] - 2023-08-28

### Features

- Implement query methods on `ArchivedImmutableKdTree`, similarly to `ArchivedKdTree`

## [3.0.0-beta.2] - 2023-08-26

Introducing the [ImmutableKdTree](https://docs.rs/kiddo/3.0.0-beta.2/kiddo/immutable/float/kdtree/struct.ImmutableKdTree.html) for floating point! :tada:

`ImmutableKdTree` is intended for use when the smallest possible on-disk serialized size of a tree is of paramount importance, and / or the fastest possible query speed is required.

Expect improvements in query time of 10-15%, and a reduction in the size of serialized trees by 33% or so on average.

These capabilities come with a few trade-offs:

1. This tree does not provide the capability to modify its contents after it has been constructed. The co-ordinates of the points to be stored must have all been generated in advance.
2. Construction time can be quite a bit slower. Typically, this can be twice as long as the default `kiddo::float::kdtree::KdTree`.
3. The more common that duplicate values are amongst your source points, the slower it will take to construct the tree. If you're using `f64` data that is fairly random-ish, you will probably not encounter any issues. I've successfully created 250 million node `ImmutableTree` instances with random `f64` data with no issues, limited only by RAM during construction. Likewise for `f32` based trees, up to a few million nodes. As per the other Kiddo float-type trees, points being stored in the tree must be floats (`f64` or `f32` are supported currently).

## [3.0.0-beta.1] - 2023-06-18

### Breaking Changes

- feat!: queries return structs instead of tuples. Query methods have been updated so that they all return
  either a `NearestNeighbour`, `Vec<NearestNeighbour>`,
  or `Vec<BestNeighbour>`, for consistency.
- feat!: use a trait instead of a function pointer for distance metrics (See [SquaredEuclidean](https://docs.rs/kiddo/3.0.0-beta.1/kiddo/float/distance/struct.SquaredEuclidean.html) and [Manhattan](https://docs.rs/kiddo/3.0.0-beta.1/kiddo/float/distance/struct.Manhattan.html))
- feat: add [within_unsorted_iter](https://docs.rs/kiddo/3.0.0-beta.1/kiddo/float/kdtree/struct.KdTree.html#method.within_unsorted_iter) query

### Performance

- perf: refactor [within](https://docs.rs/kiddo/3.0.0-beta.1/kiddo/float/kdtree/struct.KdTree.html#method.within) to simply sort the result of [within_unsorted](https://docs.rs/kiddo/3.0.0-beta.1/kiddo/float/kdtree/struct.KdTree.html#method.within_unsorted).
  Previously, `within` was keeping its results in a `BinaryHeap` and calling
  its `into_sorted_vec` method to, well, return a sorted `Vec`.
  Whilst a `BinaryHeap` is great if you are frequently adding and removing
  items, if your use case is to gradually add all your items, and then sort
  them all at once, it's quicker to just put things in a `Vec` and then
  sort the `Vec` at the end.
  Benchmarking shows that this change improves performance by anything from
  5 to 60% in practice.

## [2.1.1] - 2023-06-07

### Refactor

- update Axis trait to include some methods so that the `nearest_one` methods can be identical between `float` and `fixed`.
- float and fixed both use a common macro for best_n_within
- float and fixed both use a common macro for within_unsorted
- float and fixed both use a common macro for within
- float and fixed both use a common macro for nearest_n

### CI

- Update pre-release.yml to remove Clippy check
- Add CI for format/clippy/test/coverage/release/publish

## [2.1.0]

- feat: implement the main query methods plus `size` on `kiddo::float::kdtree::ArchivedKdTree` and improve the Rkyv example.

The previous Rkyv example was not really using Rkyv in the most efficient way (Thanks to @cavemanloverboy for spotting my mistakes!). In order to properly use rkyv's zero-copy deserialization, you need to use `rkyv::archived_root` to transmute a buffer into an `ArchivedKdTree`. For `ArchivedKdTree` to be useful, it actually needs some methods though!

v2.1.0 refactors the query code so that the method bodies of the queries are templated by macros, allowing them to be implemented on `KdTree` and `ArchivedKdTree` without completely duplicating the code.

The updated rkyv example shows the difference that zero-copy usage of rkyv makes vs deserializing, as well as also showing the gains that can be made using mmap compared to standard file access. Combining both together results in absolutely mind-blowing performance when measuring time-from-binary-start-to-first-query-result.

See for yourself by downloading the sample datasets mentioned in the examples readme and running:

```sh
cargo run --example rkyv --features=serialize_rkyv --release
```

On my machine, using the old technique of normal file access and deserialization into `KdTree`, the example code takes 348 milliseconds to load and query. The memmapped code that just transmutes to an `ArchivedKdTree` and then queries it takes 182 **micro** seconds(!) - an improvement by a factor of 1900x!!

I'll follow up this release with equivalent methods for `Fixed`, and some more ergonomic methods for loading and saving.

## [2.0.2]

- fix: properly split buckets.
  Previously, when a bucket had multiple items with the same value in the splitting dimension as the split plane, and these values straddled the pivot point, some items could end up in the wrong bucket after the split.

## [2.0.1]

- refactor: removed the requirement to use unstable features so that Kiddo should now work on Rust stable.

## [2.0.0]

Version 2 is a complete rewrite and re-architecting of Kiddo. The same methods have been provided (except periodic boundary conditions, for now), but large performance improvements have been made across the board, and some improvements have been made to the ergonomics of the library also.
Needless to say, this is a breaking change, but I hope you find the upgrade not too difficult as the improvements are significant.

### Major Changes

- Complete internal re-architecture for massive performance improvements. See below for details.
- **Integer axes as well as float:** There are now two high-level versions of the library: `float`, which (like the previous versions) uses float types for the positions of points, and `fixed`, which uses either integer or fixed point representation for the position of points.
- **integer-only contents**: The contents of the tree is restricted to being integer values only now - usually usize or u32, but you could go down to u16 or even u8 if you want. This simplifies some aspects of the internals. It has always been much more performant to store content objects in a separate Vec or array, and only store an index into this array in Kiddo itself; this change now makes that the only way.
- **Generic bucket size:** The bucket size is now a generic parameter. This allows us to use `Array`s for the bucket contents rather than `Vec`s, preventing the need for a second indirection and allowing all the memory for an entire tree to be allocated up-front in a single allocation at creation time if the required capacity of the tree is known beforehand.
- **Generic bucket index:** There is an underlying struct that additionally has a 5th generic parameter to determine the integer type that is used to index the buckets internally. Choosing a type for this param that is just small enough to permit the maximum number of buckets that your use case will encounter can give a performance boost that is especially noticeable when storing 100k -> 10M items by permitting many more of the index nodes to fit inside the CPU cache.
- **`rkyv` feature:** Previous versions provided the `serde` feature for serialization and deserialization. Due to the large number of memory allocations that were needed for big trees though, this could be quite slow - my primary use case for which Kiddo was created makes use of a ~1Gb 15-million-node tree, which took anywhere between 9 seconds to deserialize for a quick desktop all the way up to around 30s for an AWS Lambda function. Whilst in v2 the `serde` functionality is still there, you can now try an alternative approach using the incredibly quick `rkyv` zero-copy deserialization library. Amazingly, this new feature reduced the deserialization time for the use-case above to 0.6s - and the vast majority of that time is just the overhead of memory-mapping the raw file from SSD into memory.
- **`select_nth_unstable_by` node splits**: kd trees need to split the contents of a bucket between two new buckets once a bucket gets full. Previously, Kiddo would fully sort the contents of the bucket to be split, but we don't need the list to be _fully_ sorted - we only care that the "smaller" half of the items come before the "larger" ones, not the order within those two groups. Rust has a `select_nth_unstable_by` function that can do this, enabling nodes to be split more quickly. This is complicated by the architectural change mentioned below that changes Leaf nodes to use separate Arrays to store points compared to contents. We need to run `select_nth_unstable_by` over the points array but then apply the same actions that were taken to sort that array to the contents array. This required the development of a custom version of `select_nth_unstable_by` that applies the same sort actions made on a "main" array to a "mirror" array.
- **unchecked indexing of nodes**: access into the node Vecs is now unchecked, to gain a performance improvement by eliminating bounds checking.
- **empty / full and NaN checks removed**: Pre-v2 Kiddo had a lot of checks to ensure that it behaved well when items with NaNs inside their positions were attempted to be inserted, or when queries were made against empty trees. If you need these kind of checks, **This is now your responsibility** - check for NaNs and empty trees prior to insertion / query. For anyone that does not need these checks, you will benefit from better performance by not having them performed.

### Architectural changes

- Previously, the node tree structure was pointer-based, i.e. node structs had `Box<>`ed references to `left` and `right` child nodes. Kiddo v2 moves to an index-based approach: a top-level KdTree struct stores nodes in a Vec<>, and each node's `left` and `right` properties are indexes into this `Vec<>` instead. This gives a few advantages:
  - Creating a `KdTree` with a known up-front capacity can pre-allocate all nodes in a single dynamic allocation, rather than requiring one allocation per node. Even in the case where the number of nodes is not known in advance, `Vec<>` grows in chunks rather than an item at a time, resulting again in far fewer dynamic allocations being required.
- Prior to v2, Kiddo used a single Node struct that was implemented as an enum, with one enum version being used for Stem node and one for Leaf nodes. v2 switches to having separate Leaf node and Stem node structs, each being stored in their own Vec<> inside the KdTree struct. This gives several advantages:
  - Stem nodes can be made as small as possible by choosing generic types that minimize the size of each Stem node. This ensures that more of them can fit into the CPU cache.
  - Stem nodes are all next to each other and accessed in ascending order, initially close to sequentially, again helping to improve cache utilisation during the stem traversal phase of a query.
  - Leaf nodes can now use arrays to store their contents rather than Vec<>s, cutting down on multiple indirection. Due to the previous implementation using an Enum for both node types, this would have wasted too much space.
- Leaf nodes used to store a vec of structs, with each struct containing the coordinates and content for each item stored in the tree. With v2, leaf nodes now keeps two separate arrays to store their items - one for the points themselves, and one for the content. This means that the points are contiguous in memory which makes it easier for the compiler to autovectorise iteration across the points during queries, improving query performance.
- Leaf and Stem nodes now no longer store max and min bounds of their contents. The same technique as nabo-rs is used instead, whereby the distance from the most distant current result point to the most recent 3 split planes is used. This massively reduces the size of the stem nodes, so that much more of them fit in the CPU cache. Construction time is also slightly quicker thanks to not needing to calculate bounds.

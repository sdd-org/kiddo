# Tests and fuzzing

Run the library unit tests with:

```sh
cargo test --lib --features test_utils
```

## Fuzz tests

[`kd_tree_fuzz_v6.rs`](kd_tree_fuzz_v6.rs) contains generated and adversarial correctness checks. The generated tests build trees from seeded point sets and compare exact query results with a brute-force oracle across tree layouts and query types. Other tests exercise adversarial inputs and approximate search. Long-running sweeps are marked `#[ignore]`.

The fixed-point metric sweeps limit repeated coordinates so mutable buckets remain splittable. The unsigned Q8 sweep caps the tree size at roughly 8,192 points; a signed Q16 sweep covers larger trees.

For a small scalar run of one generated sweep:

```sh
KIDDO_FUZZ_CASES=1 KIDDO_FUZZ_QUERY_COUNT=4 \
KIDDO_FUZZ_MIN_POW=8 KIDDO_FUZZ_MAX_POW=8 \
KIDDO_FUZZ_PERTURB_MIN=0 KIDDO_FUZZ_PERTURB_MAX=0 \
cargo test --features fuzz --test kd_tree_fuzz_v6 fuzz_v6_mutable_f32 -- --ignored --nocapture
```

`KIDDO_FUZZ_SEED` selects the seed; without it, the tests use a fixed default. Cases, query count, size powers, perturbation and worker threads can be adjusted with the `KIDDO_FUZZ_*` environment variables in the test source. The full scalar and SIMD sweeps can take a long time and require nightly Rust:

```sh
RUSTUP_TOOLCHAIN=nightly just fuzz-kd-tree-v6
```

## Failures and replay

A captured failure key records the seed, generator settings, test and, for generated queries, the selected case and query. Keys do not store the point set. Replaying a key rebuilds the input using the same generator and assertions. Keep generator changes and dependency versions in mind when replaying older keys.

Checked-in keys are kept in [`fuzz_failures.json`](fuzz_failures.json) and replayed in feature PR CI. To replay them locally, in both scalar and SIMD modes:

```sh
just fuzz-replay
```

To replay a JSON array downloaded from a fuzz run instead, set `KIDDO_FUZZ_REPLAY_FILE=/path/to/keys.json` before the same command. Local sweeps write captured keys to `kd_tree_fuzz_failures.jsonl`; convert that report to a JSON array with:

```sh
cargo run --features fuzz --example fuzz-report -- collect \
  kd_tree_fuzz_failures.jsonl /tmp/fuzz-failures.json
```

## CI

The post-merge workflow runs the full scalar and SIMD fuzz suites on `master`; the generated sweeps use a seed derived from the commit. It replays each captured failure before opening a PR that adds reproducible keys to the checked-in array. Push a fix to that PR and keep the keys as regressions. The run artifacts also contain the captured keys and `Cargo.lock`. Failures that do not produce a reproducible key remain failed workflow runs without an automatic PR. Release PRs also run the full suite.

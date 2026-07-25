# Available CI Benchmark Options

This file is the authoritative guide for choosing which benchmark command, if any, should be suggested for a pull request.

The workflow reads the YAML block below.

- Rules are evaluated in order, top to bottom.
- The first matching rule wins.
- Put the most specific rules first.
- Use `command: null` for "no suggestion".

```yaml
rules:
  - command: /benchmark construction
    reason: Changes to the tree builder or construction pipeline can affect tree creation performance.
    match_any:
      - src/kd_tree/builder.rs
      - src/kd_tree/construction.rs
      - src/kd_tree/construction/**

  - command: /benchmark dist
    reason: Changes under `src/dist/` can affect distance metric performance.
    match_any:
      - src/dist/**

  - command: /benchmark stems
    reason: Changes under `src/stem_strategy/` or `src/traits/stem_strategy.rs` can affect stem strategy performance.
    match_any:
      - src/stem_strategy/**
      - src/traits/stem_strategy.rs

  - command: /benchmark leaf
    reason: Changes under `src/kd_tree/leaf_strategies/` can affect leaf strategy performance.
    match_any:
      - src/kd_tree/leaf_strategies/**

  - command: /benchmark extended
    reason: Changes under `src/kd_tree/query/` can affect within/nearest-within query performance and should use the extended query-family benchmark suite.
    match_any:
      - src/kd_tree/query/**

  - command: /benchmark
    reason: Changes under `src/` or to `Cargo.toml` can affect core runtime performance and should use the basic benchmark suite.
    match_any:
      - src/**
      - Cargo.toml

  - command: null
    reason: No benchmark suggestion is needed when none of the higher-priority rules match.
    match_any:
      - "**"
```

## Notes

- `/benchmark dist` is reserved for distance-metric-focused changes and should stay above broader rules.
- `/benchmark construction` is reserved for tree-builder and construction-pipeline changes and should stay above broader rules.
- `/benchmark stems` is reserved for stem-strategy-specific changes and should stay above the broader `/benchmark` rule.
- `/benchmark leaf` is reserved for leaf-strategy-specific changes and should stay above the broader `/benchmark` rule.
- `/benchmark extended` is reserved for query-pipeline changes and should stay above the broader `/benchmark` rule.
- `/benchmark` is the default command for the basic benchmark suite and for general source changes.
- If you add more benchmark commands later, insert the more specific ones above the broader defaults.

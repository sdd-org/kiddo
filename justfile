#!/usr/bin/env just --justfile

default:
  just --list

benchmark_result_key := `date -u +%Y%m%dT%H%M%SZ`
profile_min_log2_points := env_var_or_default("KIDDO_PROFILE_MIN_LOG2_POINTS", "16")
profile_max_log2_points := env_var_or_default("KIDDO_PROFILE_MAX_LOG2_POINTS", "25")
profile_queries := env_var_or_default("KIDDO_PROFILE_QUERIES", "1000")

fmt:
    cargo fmt --all
    cargo sort -wg
    taplo format *.toml
    npx --yes prettier --write ".github/workflows/*.{yml,yaml}" "*.yml" "*.yaml" ".*.yml" ".*.yaml" "**/*.md"

test-donnelly:
    cargo test donnelly

test-fast ARGS='':
    cargo test --profile fast-tests {{ARGS}}

test-fast-simd ARGS='':
    cargo test --profile fast-tests --features simd {{ARGS}}

test-fast-lib FILTER:
    cargo test --profile fast-tests --lib {{FILTER}}

test-fast-v6-nearest-one-large-f32:
    cargo test --profile fast-tests --lib v6_query_nearest_one_large_f32

fuzz-kd-tree-v6:
    rm -f kd_tree_fuzz_v6_report.txt
    RUST_TEST_THREADS=1 KIDDO_FUZZ_V6_RUN_NON_SIMD=1 KIDDO_FUZZ_V6_RUN_SIMD=0 cargo test --profile fast-tests --features fuzz --test kd_tree_fuzz_v6 -- --include-ignored --nocapture
    RUST_TEST_THREADS=1 KIDDO_FUZZ_V6_RUN_NON_SIMD=0 KIDDO_FUZZ_V6_RUN_SIMD=1 cargo test --profile fast-tests --features "fuzz simd" --test kd_tree_fuzz_v6 -- --include-ignored --nocapture

fuzz-kd-tree-v6-non-simd:
    rm -f kd_tree_fuzz_v6_report.txt
    RUST_TEST_THREADS=1 KIDDO_FUZZ_V6_RUN_NON_SIMD=1 KIDDO_FUZZ_V6_RUN_SIMD=0 cargo test --profile fast-tests --features fuzz --test kd_tree_fuzz_v6 -- --include-ignored --nocapture

fuzz-kd-tree-v6-simd:
    rm -f kd_tree_fuzz_v6_report.txt
    RUST_TEST_THREADS=1 KIDDO_FUZZ_V6_RUN_NON_SIMD=0 KIDDO_FUZZ_V6_RUN_SIMD=1 cargo test --profile fast-tests --features "fuzz simd" --test kd_tree_fuzz_v6 -- --include-ignored --nocapture

fuzz-kd-tree-v6-simd-fast:
    rm -f kd_tree_fuzz_v6_report.txt
    RUST_TEST_THREADS=1 KIDDO_FUZZ_V6_RUN_NON_SIMD=0 KIDDO_FUZZ_V6_RUN_SIMD=1 KIDDO_FUZZ_V6_SIMD_FAST=1 cargo test --profile fast-tests --features "fuzz simd" --test kd_tree_fuzz_v6 -- --include-ignored --nocapture

fuzz-case-repro REPRO:
    cargo run --features "fuzz simd" --example fuzz-case-repro -- {{REPRO}}

bench-d-v2:
    cargo bench --bench donnelly_v2

bench-d-v2b:
    cargo bench --bench donnelly_v2_branchless

# Generate x86-64-v4 assembly for donnelly_get_idx_v2
asm-x86-v4:
    RUSTFLAGS="-C target-cpu=znver3 -C opt-level=2" \
    cargo rustc --lib --release -- --emit asm -o target/donnelly_get_idx_v2_x86_64_v4.s
    @echo "Assembly output written to target/donnelly_get_idx_v2_x86_64_v4.s"
    @echo "Search for 'donnelly_get_idx_v2' in the file to find the function"

# Generate Apple M2 assembly for donnelly_get_idx_v2
asm-m4:
    RUSTFLAGS="-C target-cpu=apple-m4 -C opt-level=2" \
    cargo rustc --lib --release --features no_inline -- --emit asm -o target/donnelly_get_idx_v2_apple_m2.s
    @echo "Assembly output written to target/donnelly_get_idx_v2_apple_m2.s"
    @echo "Search for 'donnelly_get_idx_v2' in the file to find the function"


asm-k6-nearest-one-eytz:
    cargo asm --features cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" "kiddo::immutable::float::query::nearest_one::cargo_asm::v6_nearest_one_eytzinger_with_scratch" > v6_nearest_one_eytzinger.asm

asm-k6-nearest-one-eytz-v3:
    cargo asm --features cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" "v6_nearest_one_eytzinger_cargo_asm_hook" > v6_nearest_one_eytzinger_v3.asm

asm-k6-nearest-one-eytz-v3-core:
    cargo asm --features cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" "v6_nearest_one_eytzinger_arithmetic_core_cargo_asm_hook" > v6_nearest_one_eytzinger_v3_core.asm

asm-k6-nearest-one-eytz-v3-avx512:
    cargo asm --features simd,cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "v6_nearest_one_eytzinger_cargo_asm_hook" > v6_nearest_one_eytzinger_v3_avx512.asm

asm-k6-approx-nearest-one-eytz-v3-avx512:
    cargo asm --features simd,cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "v6_approx_nearest_one_eytzinger_cargo_asm_hook" > v6_approx_nearest_one_eytzinger_v3_avx512.asm

asm-k6-approx-nearest-one-eytz-v3-avx512-clean:
    RUSTC_WRAPPER= cargo asm --simplify --features simd,cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "v6_approx_nearest_one_eytzinger_cargo_asm_hook" | python3 scripts/clean_cargo_asm.py > v6_approx_nearest_one_eytzinger_v3_avx512_clean.asm

asm-k6-approx-nearest-one-eytz-voa-v3-avx512:
    cargo asm --features simd,cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "v6_approx_nearest_one_eytzinger_vec_of_arrays_cargo_asm_hook" > v6_approx_nearest_one_eytzinger_vec_of_arrays_v3_avx512.asm

asm-k6-approx-nearest-one-eytz-voa-v3-avx512-clean:
    RUSTC_WRAPPER= cargo asm --simplify --features simd,cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "v6_approx_nearest_one_eytzinger_vec_of_arrays_cargo_asm_hook" | python3 scripts/clean_cargo_asm.py > v6_approx_nearest_one_eytzinger_vec_of_arrays_v3_avx512_clean.asm

asm-k6-approx-nearest-one-eytz-voarena-v3-avx512:
    cargo asm --features simd,cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "v6_approx_nearest_one_eytzinger_vec_of_arenas_cargo_asm_hook" | python3 scripts/clean_cargo_asm.py > v6_approx_nearest_one_eytzinger_vec_of_arenas_v3_avx512.asm

asm-k6-approx-nearest-one-eytz-voarena-v3-avx512-clean:
    RUSTC_WRAPPER= cargo asm --simplify --features simd,cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "v6_approx_nearest_one_eytzinger_vec_of_arenas_cargo_asm_hook" | python3 scripts/clean_cargo_asm.py > v6_approx_nearest_one_eytzinger_vec_of_arenas_v3_avx512_clean.asm

asm-k6-nearest-one-arena-fallback-v3:
    cargo asm --features cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" "v6_nearest_one_with_query_wide_arena_fallback_cargo_asm_hook" > v6_nearest_one_with_query_wide_arena_fallback_v3.asm

asm-k6-nearest-one-arena-fallback-v3-clean:
    RUSTC_WRAPPER= cargo asm --simplify --features cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" "v6_nearest_one_with_query_wide_arena_fallback_cargo_asm_hook" | python3 scripts/clean_cargo_asm.py > v6_nearest_one_with_query_wide_arena_fallback_v3_clean.asm

asm-k6-nearest-one-donnelly-block3-fill-avx512-clean:
    RUSTC_WRAPPER= cargo asm --simplify --features simd,cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "donnelly_block3_fill_backtrack_f64_cargo_asm_hook" | python3 scripts/clean_cargo_asm.py > v6_nearest_one_donnelly_block3_fill_avx512_clean.asm

asm-k6-nearest-one-donnelly-block3-pending-select-avx512-clean:
    RUSTC_WRAPPER= cargo asm --simplify --features simd,cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "donnelly_block3_pending_select_f64_cargo_asm_hook" | python3 scripts/clean_cargo_asm.py > v6_nearest_one_donnelly_block3_pending_select_avx512_clean.asm

asm-k6-nearest-one-donnelly-block3-pending-fast-path-avx512-clean:
    RUSTC_WRAPPER= cargo asm --simplify --features simd,cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "donnelly_block3_pending_fast_path_f64_cargo_asm_hook" | python3 scripts/clean_cargo_asm.py > v6_nearest_one_donnelly_block3_pending_fast_path_avx512_clean.asm

asm-k6-nearest-one-donnelly-block3-exact-step-avx512-clean:
    RUSTC_WRAPPER= cargo asm --simplify --features simd,cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "donnelly_block3_exact_step_f64_cargo_asm_hook" | python3 scripts/clean_cargo_asm.py > v6_nearest_one_donnelly_block3_exact_step_avx512_clean.asm

asm-k6-nearest-one-donnelly-voarena-v3-avx512-clean:
    RUSTC_WRAPPER= cargo asm --simplify --features simd,cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "v6_nearest_one_donnelly_vec_of_arenas_cargo_asm_hook" | python3 scripts/clean_cargo_asm.py > v6_nearest_one_donnelly_vec_of_arenas_v3_avx512_clean.asm

asm-k6-nearest-one-donnelly-blocksimd-voarena-v3-avx512-clean:
    RUSTC_WRAPPER= cargo asm --simplify --features simd,cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "v6_nearest_one_donnelly_blocksimd_vec_of_arenas_cargo_asm_hook" | python3 scripts/clean_cargo_asm.py > v6_nearest_one_donnelly_blocksimd_vec_of_arenas_v3_avx512_clean.asm

benchmark-derive-key REF_NAME PYTHON='python3':
    {{quote(PYTHON)}} scripts/benchmark_site.py derive-key --ref-name {{quote(REF_NAME)}}

benchmark-derive-path-key REF_NAME PYTHON='python3':
    {{quote(PYTHON)}} scripts/benchmark_site.py derive-path-key --ref-name {{quote(REF_NAME)}}

bench-v6-eytzinger-focus RESULT_KEY OUTPUT_DIR='.' FEATURES='simd,test_utils,logging_off' QUERIES=profile_queries:
    #!/usr/bin/env bash
    set -euo pipefail
    result_key={{quote(RESULT_KEY)}}
    output_dir={{quote(OUTPUT_DIR)}}
    if [[ ! "$result_key" =~ ^[A-Za-z0-9][A-Za-z0-9._+:-]*$ ]]; then
        echo "RESULT_KEY must contain only letters, digits, '.', '_', '+', ':', or '-'" >&2
        exit 2
    fi
    mkdir -p "$output_dir"
    RUSTC_WRAPPER= \
        KIDDO_PROFILE_QUERIES={{quote(QUERIES)}} \
        RUSTFLAGS='-C target-cpu=native' \
        cargo criterion \
            --bench profile_v6_nearest_n_eytzinger \
            --features {{quote(FEATURES)}}
    cargo run --quiet --manifest-path tools/criterion-export/Cargo.toml -- \
        target/criterion \
        "$output_dir/bench_result-v6-nearest_n-eytzinger-${result_key}.json" \
        profile_v6_nearest_n_eytzinger
    RUSTC_WRAPPER= \
        KIDDO_PROFILE_QUERIES={{quote(QUERIES)}} \
        RUSTFLAGS='-C target-cpu=native' \
        cargo criterion \
            --bench profile_v6_nearest_one_eytzinger \
            --features {{quote(FEATURES)}}
    cargo run --quiet --manifest-path tools/criterion-export/Cargo.toml -- \
        target/criterion \
        "$output_dir/bench_result-v6-nearest_one-eytzinger-${result_key}.json" \
        profile_v6_nearest_one_eytzinger
    RUSTC_WRAPPER= \
        KIDDO_PROFILE_QUERIES={{quote(QUERIES)}} \
        RUSTFLAGS='-C target-cpu=native' \
        cargo criterion \
            --bench profile_v6_approx_nearest_one_eytzinger \
            --features {{quote(FEATURES)}}
    cargo run --quiet --manifest-path tools/criterion-export/Cargo.toml -- \
        target/criterion \
        "$output_dir/bench_result-v6-approx_nearest_one-eytzinger-${result_key}.json" \
        profile_v6_approx_nearest_one_eytzinger

bench-v6-query-family RESULT_KEY=benchmark_result_key OUTPUT_DIR='.' FEATURES='simd,test_utils,logging_off' QUERIES=profile_queries:
    #!/usr/bin/env bash
    set -euo pipefail
    result_key={{quote(RESULT_KEY)}}
    output_dir={{quote(OUTPUT_DIR)}}
    if [[ ! "$result_key" =~ ^[A-Za-z0-9][A-Za-z0-9._+:-]*$ ]]; then
        echo "RESULT_KEY must contain only letters, digits, '.', '_', '+', ':', or '-'" >&2
        exit 2
    fi
    mkdir -p "$output_dir"
    RUSTC_WRAPPER= \
        KIDDO_PROFILE_QUERIES={{quote(QUERIES)}} \
        RUSTFLAGS='-C target-cpu=native' \
        cargo criterion \
            --bench profile_v6_query_family_eytzinger \
            --features {{quote(FEATURES)}}
    cargo run --quiet --manifest-path tools/criterion-export/Cargo.toml -- \
        target/criterion \
        "$output_dir/bench_result-v6-query-family-eytzinger-${result_key}.json" \
        profile_v6_query_family_eytzinger

bench-v6-dist-metrics RESULT_KEY=benchmark_result_key OUTPUT_DIR='.' QUERIES=profile_queries SCALAR_FEATURES='test_utils,logging_off' SIMD_FEATURES='simd,test_utils,logging_off':
    #!/usr/bin/env bash
    set -euo pipefail
    result_key={{quote(RESULT_KEY)}}
    output_dir={{quote(OUTPUT_DIR)}}
    scalar_features={{quote(SCALAR_FEATURES)}}
    simd_features={{quote(SIMD_FEATURES)}}
    if [[ ! "$result_key" =~ ^[A-Za-z0-9][A-Za-z0-9._+:-]*$ ]]; then
        echo "RESULT_KEY must contain only letters, digits, '.', '_', '+', ':', or '-'" >&2
        exit 2
    fi
    host=$(rustc -vV | sed -n 's/^host: //p')
    if [[ "$host" != x86_64-* ]]; then
        echo "bench-v6-dist-metrics currently requires an x86-64 host; found $host" >&2
        exit 2
    fi
    mkdir -p "$output_dir"

    run_mode() {
        local mode=$1
        local rustflags=$2
        local features=$3

        # An explicit target keeps ISA flags off host build scripts and proc macros.
        RUSTC_WRAPPER= \
            KIDDO_PROFILE_QUERIES={{quote(QUERIES)}} \
            RUSTFLAGS="$rustflags" \
            cargo criterion \
                --target "$host" \
                --bench profile_v6_dist_metrics \
                --features "$features"
        cargo run --quiet --manifest-path tools/criterion-export/Cargo.toml -- \
            target/criterion \
            "$output_dir/bench_result-v6-dist-metrics-${mode}-${result_key}.json" \
            profile_v6_dist_metrics
    }

    run_mode scalar "-C target-cpu=x86-64-v2" "$scalar_features"
    run_mode avx2 "-C target-cpu=x86-64-v2 -C target-feature=+avx2" "$simd_features"
    run_mode avx512 "-C target-cpu=native" "$simd_features"

bench-v6-leaf-strategies RESULT_KEY=benchmark_result_key OUTPUT_DIR='.' FEATURES='simd,test_utils,logging_off' QUERIES=profile_queries:
    #!/usr/bin/env bash
    set -euo pipefail
    result_key={{quote(RESULT_KEY)}}
    output_dir={{quote(OUTPUT_DIR)}}
    if [[ ! "$result_key" =~ ^[A-Za-z0-9][A-Za-z0-9._+:-]*$ ]]; then
        echo "RESULT_KEY must contain only letters, digits, '.', '_', '+', ':', or '-'" >&2
        exit 2
    fi
    mkdir -p "$output_dir"
    RUSTC_WRAPPER= \
        KIDDO_PROFILE_QUERIES={{quote(QUERIES)}} \
        RUSTFLAGS='-C target-cpu=native' \
        cargo criterion \
            --bench profile_v6_leaf_strategies_criterion \
            --features {{quote(FEATURES)}}
    cargo run --quiet --manifest-path tools/criterion-export/Cargo.toml -- \
        target/criterion \
        "$output_dir/bench_result-v6-leaf-strategies-${result_key}.json" \
        profile_v6_leaf_strategies

bench-v6-stem-strategies RESULT_KEY=benchmark_result_key OUTPUT_DIR='.' QUERIES=profile_queries SCALAR_FEATURES='simd,test_utils,logging_off' SIMD_FEATURES='simd,test_utils,logging_off':
    #!/usr/bin/env bash
    set -euo pipefail
    result_key={{quote(RESULT_KEY)}}
    output_dir={{quote(OUTPUT_DIR)}}
    scalar_features={{quote(SCALAR_FEATURES)}}
    simd_features={{quote(SIMD_FEATURES)}}
    if [[ ! "$result_key" =~ ^[A-Za-z0-9][A-Za-z0-9._+:-]*$ ]]; then
        echo "RESULT_KEY must contain only letters, digits, '.', '_', '+', ':', or '-'" >&2
        exit 2
    fi
    host=$(rustc -vV | sed -n 's/^host: //p')
    mkdir -p "$output_dir"

    run_mode() {
        local mode=$1
        local rustflags=$2
        local features=$3

        RUSTC_WRAPPER= \
            KIDDO_STEM_BENCH_MODE="$mode" \
            KIDDO_PROFILE_QUERIES={{quote(QUERIES)}} \
            RUSTFLAGS="$rustflags" \
            cargo criterion \
                --target "$host" \
                --bench profile_v6_stem_strategies \
                --features "$features"
        cargo run --quiet --manifest-path tools/criterion-export/Cargo.toml -- \
            target/criterion \
            "$output_dir/bench_result-v6-stem-strategies-${mode}-${result_key}.json" \
            profile_v6_stem_strategies
    }

    case "$host" in
        x86_64-*)
            run_mode scalar "-C target-cpu=x86-64-v2" "$scalar_features"
            run_mode avx2 "-C target-cpu=x86-64-v2 -C target-feature=+avx2" "$simd_features"
            run_mode avx512 "-C target-cpu=native" "$simd_features"
            ;;
        aarch64-*)
            run_mode scalar "-C target-cpu=native" "$scalar_features"
            run_mode neon "-C target-cpu=native -C target-feature=+neon" "$simd_features"
            ;;
        *)
            echo "bench-v6-stem-strategies does not support host $host" >&2
            exit 2
            ;;
    esac

bench-v6-construction RESULT_KEY=benchmark_result_key OUTPUT_DIR='.' FEATURES='simd,test_utils,logging_off':
    #!/usr/bin/env bash
    set -euo pipefail
    result_key={{quote(RESULT_KEY)}}
    output_dir={{quote(OUTPUT_DIR)}}
    if [[ ! "$result_key" =~ ^[A-Za-z0-9][A-Za-z0-9._+:-]*$ ]]; then
        echo "RESULT_KEY must contain only letters, digits, '.', '_', '+', ':', or '-'" >&2
        exit 2
    fi
    host=$(rustc -vV | sed -n 's/^host: //p')
    if [[ "$host" != x86_64-* ]]; then
        echo "bench-v6-construction requires an x86_64 AVX-512 host; found $host" >&2
        exit 2
    fi
    if ! rustc --print cfg -C target-cpu=native | grep -qx 'target_feature="avx512f"'; then
        echo "bench-v6-construction requires AVX-512F support" >&2
        exit 2
    fi
    mkdir -p "$output_dir"
    RUSTC_WRAPPER= \
        RAYON_NUM_THREADS=6 \
        RUSTFLAGS='-C target-cpu=native' \
        cargo criterion \
            --target "$host" \
            --bench profile_v6_construction \
            --features {{quote(FEATURES)}}
    cargo run --quiet --manifest-path tools/criterion-export/Cargo.toml -- \
        target/criterion \
        "$output_dir/bench_result-v6-construction-avx512-${result_key}.json" \
        profile_v6_construction

bench-v6-all RESULT_KEY=benchmark_result_key OUTPUT_DIR='.':
    just bench-v6-eytzinger-focus {{quote(RESULT_KEY)}} {{quote(OUTPUT_DIR)}}
    just bench-v6-query-family {{quote(RESULT_KEY)}} {{quote(OUTPUT_DIR)}}
    just bench-v6-dist-metrics {{quote(RESULT_KEY)}} {{quote(OUTPUT_DIR)}}
    just bench-v6-leaf-strategies {{quote(RESULT_KEY)}} {{quote(OUTPUT_DIR)}}
    just bench-v6-stem-strategies {{quote(RESULT_KEY)}} {{quote(OUTPUT_DIR)}}

bench-external-kd-trees RESULT_KEY=benchmark_result_key OUTPUT_DIR='.' FEATURES='simd,test_utils,logging_off' QUERIES=profile_queries MIN_LOG2_POINTS=profile_min_log2_points MAX_LOG2_POINTS=profile_max_log2_points RADIUS='0.05':
    #!/usr/bin/env bash
    set -euo pipefail
    result_key={{quote(RESULT_KEY)}}
    output_dir={{quote(OUTPUT_DIR)}}
    if [[ ! "$result_key" =~ ^[A-Za-z0-9][A-Za-z0-9._+:-]*$ ]]; then
        echo "RESULT_KEY must contain only letters, digits, '.', '_', '+', ':', or '-'" >&2
        exit 2
    fi
    mkdir -p "$output_dir"
    RUSTC_WRAPPER= \
        KIDDO_PROFILE_QUERIES={{quote(QUERIES)}} \
        KIDDO_PROFILE_MIN_LOG2_POINTS={{quote(MIN_LOG2_POINTS)}} \
        KIDDO_PROFILE_MAX_LOG2_POINTS={{quote(MAX_LOG2_POINTS)}} \
        KIDDO_PROFILE_RADIUS={{quote(RADIUS)}} \
        RUSTFLAGS='-C target-cpu=native' \
        cargo criterion \
            --bench profile_external_kd_trees \
            --features {{quote(FEATURES)}}
    cargo run --quiet --manifest-path tools/criterion-export/Cargo.toml -- \
        target/criterion \
        "$output_dir/bench_result-external-kd-trees-${result_key}.json" \
        profile_external_kd_trees

bench-v6-within-radius-projection RESULT_KEY=benchmark_result_key OUTPUT_DIR='./focused-results' FEATURES='simd,test_utils,logging_off' QUERIES='1000' MIN_LOG2_POINTS='16' MAX_LOG2_POINTS='27' RADIUS='0.05':
    #!/usr/bin/env bash
    set -euo pipefail
    result_key={{quote(RESULT_KEY)}}
    output_dir={{quote(OUTPUT_DIR)}}
    if [[ ! "$result_key" =~ ^[A-Za-z0-9][A-Za-z0-9._+:-]*$ ]]; then
        echo "RESULT_KEY must contain only letters, digits, '.', '_', '+', ':', or '-'" >&2
        exit 2
    fi
    mkdir -p "$output_dir"
    RUSTC_WRAPPER= \
        KIDDO_PROFILE_QUERIES={{quote(QUERIES)}} \
        KIDDO_PROFILE_MIN_LOG2_POINTS={{quote(MIN_LOG2_POINTS)}} \
        KIDDO_PROFILE_MAX_LOG2_POINTS={{quote(MAX_LOG2_POINTS)}} \
        KIDDO_PROFILE_RADIUS={{quote(RADIUS)}} \
        RUSTFLAGS='-C target-cpu=native' \
        cargo criterion \
            --bench profile_v6_within_radius_projection \
            --features {{quote(FEATURES)}}
    cargo run --quiet --manifest-path tools/criterion-export/Cargo.toml -- \
        target/criterion \
        "$output_dir/bench_result-v6-within-radius-projection-${result_key}.json" \
        profile_v6_within_radius_projection

bench-neighbourhood-published RESULT_KEY=benchmark_result_key OUTPUT_DIR='./focused-results' FEATURES='simd,test_utils,logging_off' POINTS='100000000' QUERIES='20000' RUNS='1' EPSILONS='0.02,0.05,0.1,0.2':
    #!/usr/bin/env bash
    set -euo pipefail
    result_key={{quote(RESULT_KEY)}}
    output_dir={{quote(OUTPUT_DIR)}}
    if [[ ! "$result_key" =~ ^[A-Za-z0-9][A-Za-z0-9._+:-]*$ ]]; then
        echo "RESULT_KEY must contain only letters, digits, '.', '_', '+', ':', or '-'" >&2
        exit 2
    fi
    mkdir -p "$output_dir"
    RUSTC_WRAPPER= \
        KIDDO_NEIGHBOURHOOD_POINTS={{quote(POINTS)}} \
        KIDDO_NEIGHBOURHOOD_QUERIES={{quote(QUERIES)}} \
        KIDDO_NEIGHBOURHOOD_RUNS={{quote(RUNS)}} \
        KIDDO_NEIGHBOURHOOD_EPSILONS={{quote(EPSILONS)}} \
        RUSTFLAGS='-C target-cpu=native' \
        cargo bench \
            --bench profile_neighbourhood_published \
            --features {{quote(FEATURES)}} \
        | tee "$output_dir/bench_result-neighbourhood-published-${result_key}.tsv"

chart-v6-within-radius-projection RESULT_KEY RESULTS_DIR='./focused-results' OUTPUT_DIR='./focused-charts' PYTHON='python3':
    {{quote(PYTHON)}} scripts/chart_within_radius_projection_results.py charts \
        --result {{quote(RESULTS_DIR)}}/bench_result-v6-within-radius-projection-{{quote(RESULT_KEY)}}.json \
        --result-label {{quote(RESULT_KEY)}} \
        --output-dir {{quote(OUTPUT_DIR)}}

html-v6-within-radius-projection RESULT_KEY RESULTS_DIR='./focused-results' OUTPUT_DIR='./focused-charts' HTML_NAME='within-radius-projection-benchmarks.html' PYTHON='python3':
    {{quote(PYTHON)}} scripts/chart_within_radius_projection_results.py all \
        --result {{quote(RESULTS_DIR)}}/bench_result-v6-within-radius-projection-{{quote(RESULT_KEY)}}.json \
        --result-label {{quote(RESULT_KEY)}} \
        --output-dir {{quote(OUTPUT_DIR)}} \
        --html-name {{quote(HTML_NAME)}}

view-v6-within-radius-projection RESULT_KEY RESULTS_DIR='./focused-results' OUTPUT_DIR='./focused-charts' HTML_NAME='within-radius-projection-benchmarks.html' PYTHON='python3':
    just html-v6-within-radius-projection {{quote(RESULT_KEY)}} {{quote(RESULTS_DIR)}} {{quote(OUTPUT_DIR)}} {{quote(HTML_NAME)}} {{quote(PYTHON)}}
    xdg-open {{quote(OUTPUT_DIR)}}/{{quote(HTML_NAME)}}

html-v6-within-radius-item-only RESULT_KEY RESULTS_DIR='./focused-results' OUTPUT_DIR='./focused-charts' HTML_NAME='within-radius-item-only-comparison.html' PYTHON='python3':
    {{quote(PYTHON)}} scripts/chart_within_radius_projection_results.py all \
        --result {{quote(RESULTS_DIR)}}/bench_result-v6-within-radius-projection-{{quote(RESULT_KEY)}}.json \
        --result-label {{quote(RESULT_KEY)}} \
        --output-dir {{quote(OUTPUT_DIR)}} \
        --html-name {{quote(HTML_NAME)}} \
        --item-only

view-v6-within-radius-item-only RESULT_KEY RESULTS_DIR='./focused-results' OUTPUT_DIR='./focused-charts' HTML_NAME='within-radius-item-only-comparison.html' PYTHON='python3':
    just html-v6-within-radius-item-only {{quote(RESULT_KEY)}} {{quote(RESULTS_DIR)}} {{quote(OUTPUT_DIR)}} {{quote(HTML_NAME)}} {{quote(PYTHON)}}
    xdg-open {{quote(OUTPUT_DIR)}}/{{quote(HTML_NAME)}}

chart-external-kd-tree-results EXTERNAL_RESULT_KEY KIDDO_RESULT_KEY=EXTERNAL_RESULT_KEY RESULTS_DIR='.' OUTPUT_DIR='.' PYTHON='python3':
    {{quote(PYTHON)}} scripts/chart_external_kd_tree_results.py charts \
        --external {{quote(RESULTS_DIR)}}/bench_result-external-kd-trees-{{quote(EXTERNAL_RESULT_KEY)}}.json \
        --kiddo-nearest-one {{quote(RESULTS_DIR)}}/bench_result-v6-nearest_one-eytzinger-{{quote(KIDDO_RESULT_KEY)}}.json \
        --kiddo-nearest-n {{quote(RESULTS_DIR)}}/bench_result-v6-nearest_n-eytzinger-{{quote(KIDDO_RESULT_KEY)}}.json \
        --kiddo-query-family {{quote(RESULTS_DIR)}}/bench_result-v6-query-family-eytzinger-{{quote(KIDDO_RESULT_KEY)}}.json \
        --result-label {{quote(EXTERNAL_RESULT_KEY)}} \
        --output-dir {{quote(OUTPUT_DIR)}}

html-external-kd-tree-results EXTERNAL_RESULT_KEY KIDDO_RESULT_KEY=EXTERNAL_RESULT_KEY RESULTS_DIR='.' OUTPUT_DIR='.' HTML_NAME='external-kd-tree-benchmarks.html' PYTHON='python3':
    {{quote(PYTHON)}} scripts/chart_external_kd_tree_results.py all \
        --external {{quote(RESULTS_DIR)}}/bench_result-external-kd-trees-{{quote(EXTERNAL_RESULT_KEY)}}.json \
        --kiddo-nearest-one {{quote(RESULTS_DIR)}}/bench_result-v6-nearest_one-eytzinger-{{quote(KIDDO_RESULT_KEY)}}.json \
        --kiddo-nearest-n {{quote(RESULTS_DIR)}}/bench_result-v6-nearest_n-eytzinger-{{quote(KIDDO_RESULT_KEY)}}.json \
        --kiddo-query-family {{quote(RESULTS_DIR)}}/bench_result-v6-query-family-eytzinger-{{quote(KIDDO_RESULT_KEY)}}.json \
        --result-label {{quote(EXTERNAL_RESULT_KEY)}} \
        --output-dir {{quote(OUTPUT_DIR)}} \
        --html-name {{quote(HTML_NAME)}}

chart-v5-v6-results V6_RESULT_KEY V5_RESULT_KEY RESULTS_DIR='.' OUTPUT_DIR='.' PYTHON='python3':
    {{quote(PYTHON)}} scripts/chart_v5_v6_results.py charts \
        --v6-key {{quote(V6_RESULT_KEY)}} \
        --v5-key {{quote(V5_RESULT_KEY)}} \
        --results-dir {{quote(RESULTS_DIR)}} \
        --output-dir {{quote(OUTPUT_DIR)}}

html-v5-v6-results V6_RESULT_KEY V5_RESULT_KEY RESULTS_DIR='.' OUTPUT_DIR='.' HTML_NAME='v5-v6-benchmarks.html' PYTHON='python3':
    {{quote(PYTHON)}} scripts/chart_v5_v6_results.py all \
        --v6-key {{quote(V6_RESULT_KEY)}} \
        --v5-key {{quote(V5_RESULT_KEY)}} \
        --results-dir {{quote(RESULTS_DIR)}} \
        --output-dir {{quote(OUTPUT_DIR)}} \
        --html-name {{quote(HTML_NAME)}}

view-v5-v6-results V6_RESULT_KEY V5_RESULT_KEY RESULTS_DIR='.' OUTPUT_DIR='.' HTML_NAME='v5-v6-benchmarks.html' PYTHON='python3':
    just html-v5-v6-results {{quote(V6_RESULT_KEY)}} {{quote(V5_RESULT_KEY)}} {{quote(RESULTS_DIR)}} {{quote(OUTPUT_DIR)}} {{quote(HTML_NAME)}} {{quote(PYTHON)}}
    xdg-open {{quote(OUTPUT_DIR)}}/{{quote(HTML_NAME)}}

chart-benchmark-results VARIANT_KEY RESULTS_DIR='.' OUTPUT_DIR='.' PYTHON='python3':
    {{quote(PYTHON)}} scripts/chart_benchmark_results.py {{quote(VARIANT_KEY)}} --results-dir {{quote(RESULTS_DIR)}} --output-dir {{quote(OUTPUT_DIR)}}

chart-default-results VARIANT_KEY RESULTS_DIR='.' OUTPUT_DIR='.' PYTHON='python3':
    {{quote(PYTHON)}} scripts/chart_benchmark_results.py {{quote(VARIANT_KEY)}} --baseline-key v6-baseline --v5-baseline-key v5-baseline --results-dir {{quote(RESULTS_DIR)}} --output-dir {{quote(OUTPUT_DIR)}}

chart-v6-nearest-one-scratch-results RESULT_KEY RESULTS_DIR='.' OUTPUT_DIR='.' PYTHON='python3':
    {{quote(PYTHON)}} scripts/chart_benchmark_results.py {{quote(RESULT_KEY)}} --scratch --html-name latest_v6_nearest_one_scratch.html --results-dir {{quote(RESULTS_DIR)}} --output-dir {{quote(OUTPUT_DIR)}}

benchmark-site-build REF_NAME SHA RESULTS_DIR PAGES_ROOT SITE_URL_BASE='' PYTHON='python3':
    #!/usr/bin/env bash
    set -euo pipefail
    args=(publish-run --ref-name {{quote(REF_NAME)}} --sha {{quote(SHA)}} --results-dir {{quote(RESULTS_DIR)}} --pages-root {{quote(PAGES_ROOT)}})
    if [[ -n {{quote(SITE_URL_BASE)}} ]]; then
        args+=(--site-url-base {{quote(SITE_URL_BASE)}})
    fi
    {{quote(PYTHON)}} scripts/benchmark_site.py "${args[@]}"

benchmark-site-serve SITE_ROOT PORT='8000' PYTHON='python3':
    cd {{quote(SITE_ROOT)}} && {{quote(PYTHON)}} -m http.server {{quote(PORT)}}

benchmark-pages-preview REF_NAME SHA RESULTS_DIR PAGES_ROOT SITE_URL_BASE='' PYTHON='python3':
    just benchmark-site-build "{{REF_NAME}}" "{{SHA}}" "{{RESULTS_DIR}}" "{{PAGES_ROOT}}" "{{SITE_URL_BASE}}" "{{PYTHON}}"

benchmark-pages-apply REF_NAME SHA RESULTS_DIR REPO DATA_BRANCH='gh-pages' SITE_URL_BASE='' PYTHON='python3':
    #!/usr/bin/env bash
    set -euo pipefail
    pages_root=.benchmark-pages
    rm -rf "$pages_root"
    git fetch "{{REPO}}" "{{DATA_BRANCH}}" || true
    if git show-ref --verify --quiet "refs/remotes/origin/{{DATA_BRANCH}}"; then
        git worktree add "$pages_root" "origin/{{DATA_BRANCH}}"
    else
        mkdir -p "$pages_root"
        git -C "$pages_root" init
        git -C "$pages_root" checkout --orphan "{{DATA_BRANCH}}"
    fi
    just benchmark-site-build "{{REF_NAME}}" "{{SHA}}" "{{RESULTS_DIR}}" "$pages_root" "{{SITE_URL_BASE}}" "{{PYTHON}}"
    git -C "$pages_root" add .
    if git -C "$pages_root" diff --cached --quiet; then
        echo "No changes to publish"
        exit 0
    fi
    git -C "$pages_root" commit -m "benchmark: publish {{REF_NAME}} {{SHA}}"
    git -C "$pages_root" push "{{REPO}}" HEAD:"{{DATA_BRANCH}}"

benchmark-pr-comment-preview REF_NAME SHA PR_NUMBER SITE_URL_BASE PAGES_ROOT PYTHON='python3':
    #!/usr/bin/env bash
    set -euo pipefail
    branch_path_key=$({{quote(PYTHON)}} scripts/benchmark_site.py derive-path-key --ref-name {{quote(REF_NAME)}})
    summary_file={{quote(PAGES_ROOT)}}/branches/"$branch_path_key"/latest/run.json
    {{quote(PYTHON)}} scripts/benchmark_site.py render-pr-comment --summary-path "$summary_file" --site-url-base {{quote(SITE_URL_BASE)}}

benchmark-pr-comment-apply REF_NAME SHA PR_NUMBER REPO SITE_URL_BASE PAGES_ROOT PYTHON='python3':
    #!/usr/bin/env bash
    set -euo pipefail
    branch_path_key=$({{quote(PYTHON)}} scripts/benchmark_site.py derive-path-key --ref-name {{quote(REF_NAME)}})
    summary_file={{quote(PAGES_ROOT)}}/branches/"$branch_path_key"/latest/run.json
    {{quote(PYTHON)}} scripts/benchmark_site.py update-pr-comment --repo {{quote(REPO)}} --pr-number {{quote(PR_NUMBER)}} --summary-path "$summary_file" --site-url-base {{quote(SITE_URL_BASE)}}

asm-v6-sorted-nearest-n-within-donnelly-pf-focus-clean FEATURES='simd,cargo_asm,logging_off' SUFFIX='baseline':
    RUSTC_WRAPPER= cargo asm --simplify --features {{FEATURES}} --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "v6_sorted_nearest_n_within_donnelly_pf_focus_cargo_asm_hook" | python3 scripts/clean_cargo_asm.py > v6_sorted_nearest_n_within_donnelly_pf_focus_{{SUFFIX}}_clean.asm

asm-v6-query-hook-clean HOOK FEATURES='simd,cargo_asm,logging_off' SUFFIX='baseline':
    RUSTC_WRAPPER= cargo asm --simplify --features {{FEATURES}} --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "{{HOOK}}" | python3 scripts/clean_cargo_asm.py > {{HOOK}}_{{SUFFIX}}_clean.asm

asm-v6-best-n-within-donnelly-pf-focus-clean FEATURES='simd,cargo_asm,logging_off' SUFFIX='baseline':
    RUSTC_WRAPPER= cargo asm --simplify --features {{FEATURES}} --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "v6_best_n_within_donnelly_pf_focus_cargo_asm_hook" | python3 scripts/clean_cargo_asm.py > v6_best_n_within_donnelly_pf_focus_{{SUFFIX}}_clean.asm

asm-v6-result-collection-hook-clean HOOK FEATURES='simd,cargo_asm,logging_off' SUFFIX='baseline':
    RUSTC_WRAPPER= cargo asm --simplify --features {{FEATURES}} --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "{{HOOK}}" | python3 scripts/clean_cargo_asm.py > {{HOOK}}_{{SUFFIX}}_clean.asm

mca-v6-query-hook ASM_FILE OUT_FILE:
    llvm-mca -march=x86-64 -mcpu=znver5 -x86-asm-syntax=intel -skip-unsupported-instructions=parse-failure --instruction-info --summary-view {{ASM_FILE}} > {{OUT_FILE}} 2>&1

mca-v6-result-collection-focus ASM_FILE OUT_FILE:
    llvm-mca -march=x86-64 -mcpu=znver5 -x86-asm-syntax=intel -skip-unsupported-instructions=parse-failure --instruction-info --summary-view {{ASM_FILE}} > {{OUT_FILE}} 2>&1

asm-v6-nearest-one-epf-var-clean FEATURES='simd,cargo_asm,logging_off' SUFFIX='baseline':
    just asm-v6-query-hook-clean v6_nearest_one_eytzinger_pf_far_vec_of_arenas_cargo_asm_hook "{{FEATURES}}" "{{SUFFIX}}"

asm-v6-within-unsorted-epf-var-clean FEATURES='simd,cargo_asm,logging_off' SUFFIX='baseline':
    just asm-v6-query-hook-clean v6_within_unsorted_eytzinger_pf_far_vec_of_arenas_cargo_asm_hook "{{FEATURES}}" "{{SUFFIX}}"

asm-v6-nnws-epf-var-clean FEATURES='simd,cargo_asm,logging_off' SUFFIX='baseline':
    just asm-v6-query-hook-clean v6_sorted_nearest_n_within_eytzinger_pf_far_focus_cargo_asm_hook "{{FEATURES}}" "{{SUFFIX}}"

profile-v6-stem-exact-stats FEATURES='simd,test_utils,logging_off' POINTS='4194304' QUERIES='10000' REPEATS='1':
    RUSTC_WRAPPER= \
    KIDDO_PROFILE_POINTS={{POINTS}} \
    KIDDO_PROFILE_QUERIES={{QUERIES}} \
    KIDDO_PROFILE_QUERY_BATCH_REPEATS={{REPEATS}} \
    RUSTFLAGS='-C target-cpu=native' \
    cargo bench --bench profile_v6_stem_exact_stats --features {{FEATURES}}

build-v6-profile-archives FEATURES='rkyv_08,simd,test_utils,logging_off' POINTS='16777216' QUERIES='100' PREFIX='./target/kiddo-profile-v6-result-collection':
    RUSTC_WRAPPER= \
    KIDDO_PROFILE_POINTS={{POINTS}} \
    KIDDO_PROFILE_QUERIES={{QUERIES}} \
    KIDDO_PROFILE_ARCHIVE_PREFIX={{PREFIX}} \
    RUSTFLAGS='-C target-cpu=native' \
    cargo run --release --example build_v6_profile_archives --features {{FEATURES}}

profile-v6-result-collection-stats FEATURES='rkyv_08,simd,test_utils,result_collection_stats,logging_off' REPEATS='1' MAX_QTY='16' MAX_DIST='0.0025' PREFIX='./target/kiddo-profile-v6-result-collection':
    RUSTC_WRAPPER= \
    KIDDO_PROFILE_ARCHIVE_PREFIX={{PREFIX}} \
    KIDDO_PROFILE_QUERY_BATCH_REPEATS={{REPEATS}} \
    KIDDO_PROFILE_MAX_QTY={{MAX_QTY}} \
    KIDDO_PROFILE_MAX_DIST={{MAX_DIST}} \
    RUSTFLAGS='-C target-cpu=native' \
    cargo bench --bench profile_v6_result_collection_stats --features {{FEATURES}}

profile-v6-result-collection-thresholds FEATURES='test_utils,logging_off' TREE_SIZES='262144,1048576,4194304' QUERIES='1024' SAMPLES='5' MAX_QTYS='16,20,24,32,48,64,80,96,112,128,144,160,176,192,208,224,256':
    RUSTC_WRAPPER= \
    KIDDO_PROFILE_TREE_SIZES={{quote(TREE_SIZES)}} \
    KIDDO_PROFILE_QUERIES={{quote(QUERIES)}} \
    KIDDO_PROFILE_SAMPLES={{quote(SAMPLES)}} \
    KIDDO_PROFILE_MAX_QTYS={{quote(MAX_QTYS)}} \
    RUSTFLAGS='-C target-cpu=native' \
    cargo bench --bench profile_v6_result_collection_thresholds --features {{FEATURES}}

# Compare default radius-query allocation growth with a caller-provided capacity hint.
profile-v6-result-capacity FEATURES='test_utils,logging_off' TREE_SIZE='1048576' QUERIES='256' SAMPLES='5':
    RUSTC_WRAPPER= KIDDO_PROFILE_TREE_SIZE='{{TREE_SIZE}}' KIDDO_PROFILE_QUERIES='{{QUERIES}}' KIDDO_PROFILE_SAMPLES='{{SAMPLES}}' RUSTFLAGS='-C target-cpu=native' \
    cargo bench --bench profile_v6_result_capacity --features {{FEATURES}}

build-v6-hugepage-archives FEATURES='rkyv_08,test_utils,logging_off' POINTS='33554432' QUERIES='100000' PREFIX='./target/kiddo-hugepage-v6':
    RUSTC_WRAPPER= \
    KIDDO_PROFILE_POINTS={{POINTS}} \
    KIDDO_PROFILE_QUERIES={{QUERIES}} \
    KIDDO_PROFILE_ARCHIVE_PREFIX={{PREFIX}} \
    RUSTFLAGS='-C target-cpu=native' \
    cargo run --release --example build_v6_hugepage_archives --features {{FEATURES}}

build-v6-query-focus-archives FEATURES='rkyv_08,test_utils,logging_off' POINTS='33554432' QUERIES='100000' PREFIX='./target/kiddo-query-focus-v6':
    RUSTC_WRAPPER= \
    KIDDO_PROFILE_POINTS={{POINTS}} \
    KIDDO_PROFILE_QUERIES={{QUERIES}} \
    KIDDO_PROFILE_ARCHIVE_PREFIX={{PREFIX}} \
    RUSTFLAGS='-C target-cpu=native' \
    cargo run --release --example build_v6_query_focus_archives --features {{FEATURES}}

build-profile-v6-archived-hugepages FEATURES='rkyv_08,huge_pages,simd,logging_off':
    RUSTC_WRAPPER= \
    RUSTFLAGS='-C target-cpu=native' \
    cargo build --release --bench profile_v6_archived_huge_pages --features {{FEATURES}}

build-profile-v6-archived-query-focus FEATURES='rkyv_08,huge_pages,simd,logging_off':
    RUSTC_WRAPPER= \
    RUSTFLAGS='-C target-cpu=native' \
    cargo build --release --bench profile_v6_archived_query_focus --features {{FEATURES}}

profile-v6-archived-hugepages FEATURES='rkyv_08,huge_pages,simd,logging_off' MODE='collapse' LOAD='mmap' QUERY='nearest-one' REPEATS='100' PREFIX='./target/kiddo-hugepage-v6' START_DELAY_MS='0':
    RUSTC_WRAPPER= \
    KIDDO_PROFILE_ARCHIVE_PREFIX={{PREFIX}} \
    KIDDO_PROFILE_HUGE_PAGES={{MODE}} \
    KIDDO_PROFILE_LOAD_MODE={{LOAD}} \
    KIDDO_PROFILE_QUERY_KIND={{QUERY}} \
    KIDDO_PROFILE_QUERY_BATCH_REPEATS={{REPEATS}} \
    KIDDO_PROFILE_START_DELAY_MS={{START_DELAY_MS}} \
    RUSTFLAGS='-C target-cpu=native' \
    cargo bench --bench profile_v6_archived_huge_pages --features {{FEATURES}}

profile-v6-archived-query-focus FEATURES='rkyv_08,huge_pages,simd,logging_off' LOAD='mmap' HUGE='off' QUERY='nearest-one' REPEATS='100' MAX_DIST='0.01' MAX_QTY='1000' PREFIX='./target/kiddo-query-focus-v6' START_DELAY_MS='0':
    RUSTC_WRAPPER= \
    KIDDO_PROFILE_ARCHIVE_PREFIX={{PREFIX}} \
    KIDDO_PROFILE_LOAD_MODE={{LOAD}} \
    KIDDO_PROFILE_HUGE_PAGES={{HUGE}} \
    KIDDO_PROFILE_QUERY_KIND={{QUERY}} \
    KIDDO_PROFILE_QUERY_BATCH_REPEATS={{REPEATS}} \
    KIDDO_PROFILE_MAX_DIST={{MAX_DIST}} \
    KIDDO_PROFILE_MAX_QTY={{MAX_QTY}} \
    KIDDO_PROFILE_START_DELAY_MS={{START_DELAY_MS}} \
    RUSTFLAGS='-C target-cpu=native' \
    cargo bench --bench profile_v6_archived_query_focus --features {{FEATURES}}

perf-v6-archived-query-focus-core FEATURES='rkyv_08,huge_pages,simd,logging_off' LOAD='mmap' HUGE='off' QUERY='nearest-one' REPEATS='100' MAX_DIST='0.01' MAX_QTY='1000' PREFIX='./target/kiddo-query-focus-v6' START_DELAY_MS='0' PERF_DELAY_MS='0':
    RUSTC_WRAPPER= \
    RUSTFLAGS='-C target-cpu=native' \
    cargo build --release --bench profile_v6_archived_query_focus --features {{FEATURES}}
    BENCH_BIN=$(find target/release/deps -maxdepth 1 -type f -perm -111 -name 'profile_v6_archived_query_focus-*' | head -n 1) && \
    KIDDO_PROFILE_ARCHIVE_PREFIX={{PREFIX}} \
    KIDDO_PROFILE_LOAD_MODE={{LOAD}} \
    KIDDO_PROFILE_HUGE_PAGES={{HUGE}} \
    KIDDO_PROFILE_QUERY_KIND={{QUERY}} \
    KIDDO_PROFILE_QUERY_BATCH_REPEATS={{REPEATS}} \
    KIDDO_PROFILE_MAX_DIST={{MAX_DIST}} \
    KIDDO_PROFILE_MAX_QTY={{MAX_QTY}} \
    KIDDO_PROFILE_START_DELAY_MS={{START_DELAY_MS}} \
    perf stat -D {{PERF_DELAY_MS}} \
        -e cycles,instructions,branches,branch-misses \
        "$BENCH_BIN"

perf-v6-archived-query-focus-cache FEATURES='rkyv_08,huge_pages,simd,logging_off' LOAD='mmap' HUGE='off' QUERY='nearest-one' REPEATS='100' MAX_DIST='0.01' MAX_QTY='1000' PREFIX='./target/kiddo-query-focus-v6' START_DELAY_MS='0' PERF_DELAY_MS='0':
    RUSTC_WRAPPER= \
    RUSTFLAGS='-C target-cpu=native' \
    cargo build --release --bench profile_v6_archived_query_focus --features {{FEATURES}}
    BENCH_BIN=$(find target/release/deps -maxdepth 1 -type f -perm -111 -name 'profile_v6_archived_query_focus-*' | head -n 1) && \
    KIDDO_PROFILE_ARCHIVE_PREFIX={{PREFIX}} \
    KIDDO_PROFILE_LOAD_MODE={{LOAD}} \
    KIDDO_PROFILE_HUGE_PAGES={{HUGE}} \
    KIDDO_PROFILE_QUERY_KIND={{QUERY}} \
    KIDDO_PROFILE_QUERY_BATCH_REPEATS={{REPEATS}} \
    KIDDO_PROFILE_MAX_DIST={{MAX_DIST}} \
    KIDDO_PROFILE_MAX_QTY={{MAX_QTY}} \
    KIDDO_PROFILE_START_DELAY_MS={{START_DELAY_MS}} \
    perf stat -D {{PERF_DELAY_MS}} \
        -e cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
        "$BENCH_BIN"

perf-v6-archived-query-focus-tlb FEATURES='rkyv_08,huge_pages,simd,logging_off' LOAD='mmap' HUGE='off' QUERY='nearest-one' REPEATS='100' MAX_DIST='0.01' MAX_QTY='1000' PREFIX='./target/kiddo-query-focus-v6' START_DELAY_MS='0' PERF_DELAY_MS='0':
    RUSTC_WRAPPER= \
    RUSTFLAGS='-C target-cpu=native' \
    cargo build --release --bench profile_v6_archived_query_focus --features {{FEATURES}}
    BENCH_BIN=$(find target/release/deps -maxdepth 1 -type f -perm -111 -name 'profile_v6_archived_query_focus-*' | head -n 1) && \
    KIDDO_PROFILE_ARCHIVE_PREFIX={{PREFIX}} \
    KIDDO_PROFILE_LOAD_MODE={{LOAD}} \
    KIDDO_PROFILE_HUGE_PAGES={{HUGE}} \
    KIDDO_PROFILE_QUERY_KIND={{QUERY}} \
    KIDDO_PROFILE_QUERY_BATCH_REPEATS={{REPEATS}} \
    KIDDO_PROFILE_MAX_DIST={{MAX_DIST}} \
    KIDDO_PROFILE_MAX_QTY={{MAX_QTY}} \
    KIDDO_PROFILE_START_DELAY_MS={{START_DELAY_MS}} \
    perf stat -D {{PERF_DELAY_MS}} \
        -e dTLB-loads,dTLB-load-misses,page-faults,minor-faults,major-faults \
        "$BENCH_BIN"

profile-v6-archived-query-focus-samply FEATURES='rkyv_08,huge_pages,simd,logging_off' LOAD='mmap' HUGE='off' QUERY='nearest-one' REPEATS='100' MAX_DIST='0.01' MAX_QTY='1000' PREFIX='./target/kiddo-query-focus-v6':
    RUSTC_WRAPPER= \
    RUSTFLAGS='-C target-cpu=native' \
    cargo build --release --bench profile_v6_archived_query_focus --features {{FEATURES}}
    BENCH_BIN=$(find target/release/deps -maxdepth 1 -type f -perm -111 -name 'profile_v6_archived_query_focus-*' | head -n 1) && \
    KIDDO_PROFILE_ARCHIVE_PREFIX={{PREFIX}} \
    KIDDO_PROFILE_LOAD_MODE={{LOAD}} \
    KIDDO_PROFILE_HUGE_PAGES={{HUGE}} \
    KIDDO_PROFILE_QUERY_KIND={{QUERY}} \
    KIDDO_PROFILE_QUERY_BATCH_REPEATS={{REPEATS}} \
    KIDDO_PROFILE_MAX_DIST={{MAX_DIST}} \
    KIDDO_PROFILE_MAX_QTY={{MAX_QTY}} \
    samply record "$BENCH_BIN"

perf-v6-archived-hugepages FEATURES='rkyv_08,huge_pages,simd,logging_off' MODE='collapse' LOAD='mmap' QUERY='nearest-one' REPEATS='100' PREFIX='./target/kiddo-hugepage-v6' START_DELAY_MS='0' PERF_DELAY_MS='0':
    RUSTC_WRAPPER= \
    RUSTFLAGS='-C target-cpu=native' \
    cargo build --release --bench profile_v6_archived_huge_pages --features {{FEATURES}}
    BENCH_BIN=$(find target/release/deps -maxdepth 1 -type f -perm -111 -name 'profile_v6_archived_huge_pages-*' | head -n 1) && \
    KIDDO_PROFILE_ARCHIVE_PREFIX={{PREFIX}} \
    KIDDO_PROFILE_HUGE_PAGES={{MODE}} \
    KIDDO_PROFILE_LOAD_MODE={{LOAD}} \
    KIDDO_PROFILE_QUERY_KIND={{QUERY}} \
    KIDDO_PROFILE_QUERY_BATCH_REPEATS={{REPEATS}} \
    KIDDO_PROFILE_START_DELAY_MS={{START_DELAY_MS}} \
    perf stat -D {{PERF_DELAY_MS}} \
        -e cycles,instructions,branches,branch-misses,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses,page-faults,minor-faults,major-faults \
        "$BENCH_BIN"

perf-v6-archived-hugepages-pair FEATURES='rkyv_08,huge_pages,simd,logging_off' LOAD='mmap' QUERY='nearest-one' REPEATS='100' PREFIX='./target/kiddo-hugepage-v6' START_DELAY_MS='0' PERF_DELAY_MS='0':
    just perf-v6-archived-hugepages-tlb "{{FEATURES}}" nohuge "{{LOAD}}" "{{QUERY}}" "{{REPEATS}}" "{{PREFIX}}" "{{START_DELAY_MS}}" "{{PERF_DELAY_MS}}"
    just perf-v6-archived-hugepages-tlb "{{FEATURES}}" collapse "{{LOAD}}" "{{QUERY}}" "{{REPEATS}}" "{{PREFIX}}" "{{START_DELAY_MS}}" "{{PERF_DELAY_MS}}"

perf-v6-archived-hugepages-core FEATURES='rkyv_08,huge_pages,simd,logging_off' MODE='collapse' LOAD='mmap' QUERY='nearest-one' REPEATS='100' PREFIX='./target/kiddo-hugepage-v6' START_DELAY_MS='0' PERF_DELAY_MS='0':
    RUSTC_WRAPPER= \
    RUSTFLAGS='-C target-cpu=native' \
    cargo build --release --bench profile_v6_archived_huge_pages --features {{FEATURES}}
    BENCH_BIN=$(find target/release/deps -maxdepth 1 -type f -perm -111 -name 'profile_v6_archived_huge_pages-*' | head -n 1) && \
    KIDDO_PROFILE_ARCHIVE_PREFIX={{PREFIX}} \
    KIDDO_PROFILE_HUGE_PAGES={{MODE}} \
    KIDDO_PROFILE_LOAD_MODE={{LOAD}} \
    KIDDO_PROFILE_QUERY_KIND={{QUERY}} \
    KIDDO_PROFILE_QUERY_BATCH_REPEATS={{REPEATS}} \
    KIDDO_PROFILE_START_DELAY_MS={{START_DELAY_MS}} \
    perf stat -D {{PERF_DELAY_MS}} \
        -e cycles,instructions,branches,branch-misses \
        "$BENCH_BIN"

perf-v6-archived-hugepages-cache FEATURES='rkyv_08,huge_pages,simd,logging_off' MODE='collapse' LOAD='mmap' QUERY='nearest-one' REPEATS='100' PREFIX='./target/kiddo-hugepage-v6' START_DELAY_MS='0' PERF_DELAY_MS='0':
    RUSTC_WRAPPER= \
    RUSTFLAGS='-C target-cpu=native' \
    cargo build --release --bench profile_v6_archived_huge_pages --features {{FEATURES}}
    BENCH_BIN=$(find target/release/deps -maxdepth 1 -type f -perm -111 -name 'profile_v6_archived_huge_pages-*' | head -n 1) && \
    KIDDO_PROFILE_ARCHIVE_PREFIX={{PREFIX}} \
    KIDDO_PROFILE_HUGE_PAGES={{MODE}} \
    KIDDO_PROFILE_LOAD_MODE={{LOAD}} \
    KIDDO_PROFILE_QUERY_KIND={{QUERY}} \
    KIDDO_PROFILE_QUERY_BATCH_REPEATS={{REPEATS}} \
    KIDDO_PROFILE_START_DELAY_MS={{START_DELAY_MS}} \
    perf stat -D {{PERF_DELAY_MS}} \
        -e cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses \
        "$BENCH_BIN"

perf-v6-archived-hugepages-tlb FEATURES='rkyv_08,huge_pages,simd,logging_off' MODE='collapse' LOAD='mmap' QUERY='nearest-one' REPEATS='100' PREFIX='./target/kiddo-hugepage-v6' START_DELAY_MS='0' PERF_DELAY_MS='0':
    RUSTC_WRAPPER= \
    RUSTFLAGS='-C target-cpu=native' \
    cargo build --release --bench profile_v6_archived_huge_pages --features {{FEATURES}}
    BENCH_BIN=$(find target/release/deps -maxdepth 1 -type f -perm -111 -name 'profile_v6_archived_huge_pages-*' | head -n 1) && \
    KIDDO_PROFILE_ARCHIVE_PREFIX={{PREFIX}} \
    KIDDO_PROFILE_HUGE_PAGES={{MODE}} \
    KIDDO_PROFILE_LOAD_MODE={{LOAD}} \
    KIDDO_PROFILE_QUERY_KIND={{QUERY}} \
    KIDDO_PROFILE_QUERY_BATCH_REPEATS={{REPEATS}} \
    KIDDO_PROFILE_START_DELAY_MS={{START_DELAY_MS}} \
    perf stat -D {{PERF_DELAY_MS}} \
        -e dTLB-loads,dTLB-load-misses,page-faults,minor-faults,major-faults \
        "$BENCH_BIN"

perf-v6-archived-hugepages-tlb-pair FEATURES='rkyv_08,huge_pages,simd,logging_off' LOAD='mmap' QUERY='nearest-one' REPEATS='100' PREFIX='./target/kiddo-hugepage-v6' START_DELAY_MS='0' PERF_DELAY_MS='0':
    just perf-v6-archived-hugepages-tlb "{{FEATURES}}" nohuge "{{LOAD}}" "{{QUERY}}" "{{REPEATS}}" "{{PREFIX}}" "{{START_DELAY_MS}}" "{{PERF_DELAY_MS}}"
    just perf-v6-archived-hugepages-tlb "{{FEATURES}}" collapse "{{LOAD}}" "{{QUERY}}" "{{REPEATS}}" "{{PREFIX}}" "{{START_DELAY_MS}}" "{{PERF_DELAY_MS}}"

repro-donnelly-block3-exact-divergence FEATURES='simd,test_utils,logging_off' POINTS='4194304' QUERIES='10000':
    RUSTC_WRAPPER= \
    KIDDO_REPRO_POINTS={{POINTS}} \
    KIDDO_REPRO_QUERIES={{QUERIES}} \
    RUSTFLAGS='-C target-cpu=native' \
    cargo run --release --example repro_donnelly_block3_exact_divergence --features {{FEATURES}}

asm-k6-nearest-one-eytz-v3-core-avx512:
    cargo asm --features simd,cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "v6_nearest_one_eytzinger_arithmetic_core_cargo_asm_hook" > v6_nearest_one_eytzinger_v3_core_avx512.asm

asm-k6-nearest-one-eytz-v3-leaf-avx512:
    cargo asm --features simd,cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "kiddo::leaf_view_chunked::nearest_one::avx512::leaf_nearest_one_chunked_nozero_f64_k3::<f64, kiddo::dist::squared_euclidean::SquaredEuclidean<f64>, usize>" > v6_nearest_one_eytzinger_v3_leaf_avx512.asm

asm-k6-nearest-one-arena-leaf-v3-avx512-clean:
    RUSTC_WRAPPER= cargo asm --simplify --features simd,cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "v6_nearest_one_arena_leaf_cargo_asm_hook" | python3 scripts/clean_cargo_asm.py > v6_nearest_one_arena_leaf_v3_avx512_clean.asm

asm-k6-nearest-one-eytz-v3-leaf-avx512-clean:
    RUSTC_WRAPPER= cargo asm --simplify --features simd,cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "kiddo::leaf_view_chunked::nearest_one::avx512::leaf_nearest_one_chunked_nozero_f64_k3::<f64, kiddo::dist::squared_euclidean::SquaredEuclidean<f64>, usize>" | python3 scripts/clean_cargo_asm.py > v6_nearest_one_eytzinger_v3_leaf_avx512_clean.asm

asm-k6-nearest-one-voarena-v3-leaf-avx512:
    cargo asm --features simd,cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "kiddo::leaf_view_chunked::nearest_one::avx512::leaf_nearest_one_arena_nozero_f64_k3::<f64, kiddo::dist::squared_euclidean::SquaredEuclidean<f64>, usize>" > v6_nearest_one_arena_leaf_k3_avx512.asm

asm-k6-nearest-one-voarena-v3-leaf-avx512-clean:
    RUSTC_WRAPPER= cargo asm --simplify --features simd,cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "kiddo::leaf_view_chunked::nearest_one::avx512::leaf_nearest_one_arena_nozero_f64_k3::<f64, kiddo::dist::squared_euclidean::SquaredEuclidean<f64>, usize>" | python3 scripts/clean_cargo_asm.py > v6_nearest_one_arena_leaf_k3_avx512_clean.asm

objdump-k6-nearest-one-eytz:
    cargo objdump --release --lib --features cargo_asm,logging_off -- --disassemble-symbols="kiddo::immutable::float::query::nearest_one::cargo_asm::v6_nearest_one_eytzinger_with_scratch" --demangle > v6_nearest_one_eytzinger.objdump

objdump-k5-nearest-one:
    cd ../kiddo-v5 && cargo objdump --release --lib -- --disassemble-symbols="kiddo::immutable::float::kdtree::cargo_asm::v5_nearest_one_immutable" --demangle > v5_immutable_nearest_one.objdump

objdump-k5-symbols:
    cd ../kiddo-v5 && cargo objdump --release --lib -- --syms --demangle > v5_symbols.objdump

objdump-k5-nearest-one-recurse SYMBOL:
    cd ../kiddo-v5 && cargo objdump --release --lib -- --disassemble-symbols="{{SYMBOL}}" --demangle > v5_nearest_one_recurse.objdump

objdump-profile-v5-symbols:
    cargo objdump --release --bin profile_v5_nearest_one_eytzinger --features profile_v5 -- --syms --demangle > profile_v5_symbols.objdump

objdump-profile-v5-symbol SYMBOL:
    cargo objdump --release --bin profile_v5_nearest_one_eytzinger --features profile_v5 -- --disassemble-symbols="{{SYMBOL}}" --demangle > profile_v5_symbol.objdump

build-profile-v6-nearest-one-eytz:
    cargo build --release --features test_utils --bench profile_v6_nearest_one_eytzinger

profile-v6-nearest-one-eytz-samply: build-profile-v6-nearest-one-eytz
    BENCH_BIN=$(find target/release/deps -maxdepth 1 -type f -perm -111 -name 'profile_v6_nearest_one_eytzinger-*' | head -n 1) && \
    KIDDO_PROFILE_QUERY_BATCH_REPEATS=2000 samply record "$BENCH_BIN"

build-profile-v6-leaf-strategies FEATURES='simd,test_utils':
    cargo build --release --features {{FEATURES}} --bench profile_v6_leaf_strategies

perf-v6-leaf-strategies QUERY='nearest' STRATEGY='arena' POINTS='4194304' QUERIES='1000' REPEATS='2000' FEATURES='simd,test_utils':
    cargo build --release --features {{FEATURES}} --bench profile_v6_leaf_strategies
    BENCH_BIN=$(find target/release/deps -maxdepth 1 -type f -perm -111 -name 'profile_v6_leaf_strategies-*' | head -n 1) && \
    KIDDO_PROFILE_QUERY_KIND={{QUERY}} \
    KIDDO_PROFILE_STRATEGY={{STRATEGY}} \
    KIDDO_PROFILE_POINTS={{POINTS}} \
    KIDDO_PROFILE_QUERIES={{QUERIES}} \
    KIDDO_PROFILE_QUERY_BATCH_REPEATS={{REPEATS}} \
    perf stat -d -d -d \
        -e cycles,instructions,branches,branch-misses,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses \
        "$BENCH_BIN"

perf-v6-leaf-strategies-branch QUERY='nearest' STRATEGY='arena' POINTS='4194304' QUERIES='1000' REPEATS='2000' FEATURES='simd,test_utils':
    cargo build --release --features {{FEATURES}} --bench profile_v6_leaf_strategies
    BENCH_BIN=$(find target/release/deps -maxdepth 1 -type f -perm -111 -name 'profile_v6_leaf_strategies-*' | head -n 1) && \
    KIDDO_PROFILE_QUERY_KIND={{QUERY}} \
    KIDDO_PROFILE_STRATEGY={{STRATEGY}} \
    KIDDO_PROFILE_POINTS={{POINTS}} \
    KIDDO_PROFILE_QUERIES={{QUERIES}} \
    KIDDO_PROFILE_QUERY_BATCH_REPEATS={{REPEATS}} \
    perf stat \
        -e cycles,instructions,branches,branch-misses \
        "$BENCH_BIN"

perf-v6-leaf-strategies-cache QUERY='nearest' STRATEGY='arena' POINTS='4194304' QUERIES='1000' REPEATS='2000' FEATURES='simd,test_utils':
    cargo build --release --features {{FEATURES}} --bench profile_v6_leaf_strategies
    BENCH_BIN=$(find target/release/deps -maxdepth 1 -type f -perm -111 -name 'profile_v6_leaf_strategies-*' | head -n 1) && \
    KIDDO_PROFILE_QUERY_KIND={{QUERY}} \
    KIDDO_PROFILE_STRATEGY={{STRATEGY}} \
    KIDDO_PROFILE_POINTS={{POINTS}} \
    KIDDO_PROFILE_QUERIES={{QUERIES}} \
    KIDDO_PROFILE_QUERY_BATCH_REPEATS={{REPEATS}} \
    perf stat \
        -e cache-misses,L1-dcache-load-misses,l2_cache_req_stat.ls_rd_blk_c,ls_dmnd_fills_from_sys.local_l2,ls_dmnd_fills_from_sys.local_ccx,ls_dmnd_fills_from_sys.dram_io_near \
        "$BENCH_BIN"

perf-v6-leaf-strategies-other QUERY='nearest' STRATEGY='arena' POINTS='4194304' QUERIES='1000' REPEATS='2000' FEATURES='simd,test_utils':
    cargo build --release --features {{FEATURES}} --bench profile_v6_leaf_strategies
    BENCH_BIN=$(find target/release/deps -maxdepth 1 -type f -perm -111 -name 'profile_v6_leaf_strategies-*' | head -n 1) && \
    KIDDO_PROFILE_QUERY_KIND={{QUERY}} \
    KIDDO_PROFILE_STRATEGY={{STRATEGY}} \
    KIDDO_PROFILE_POINTS={{POINTS}} \
    KIDDO_PROFILE_QUERIES={{QUERIES}} \
    KIDDO_PROFILE_QUERY_BATCH_REPEATS={{REPEATS}} \
    perf stat \
        -e dTLB-loads,dTLB-load-misses,itlb-loads,itlb-load-misses,ls_l1_d_tlb_miss.tlb_reload_4k_l2_hit,ls_l1_d_tlb_miss.tlb_reload_4k_l2_miss \
        "$BENCH_BIN"

perf-v6-leaf-strategies-prefetch QUERY='nearest' STRATEGY='arena' POINTS='4194304' QUERIES='1000' REPEATS='2000' FEATURES='simd,test_utils':
    cargo build --release --features {{FEATURES}} --bench profile_v6_leaf_strategies
    BENCH_BIN=$(find target/release/deps -maxdepth 1 -type f -perm -111 -name 'profile_v6_leaf_strategies-*' | head -n 1) && \
    KIDDO_PROFILE_QUERY_KIND={{QUERY}} \
    KIDDO_PROFILE_STRATEGY={{STRATEGY}} \
    KIDDO_PROFILE_POINTS={{POINTS}} \
    KIDDO_PROFILE_QUERIES={{QUERIES}} \
    KIDDO_PROFILE_QUERY_BATCH_REPEATS={{REPEATS}} \
    perf stat \
        -e ls_pref_instr_disp.prefetch_nta,ls_inef_sw_pref.mab_mch_cnt,ls_sw_pf_dc_fills.local_ccx,l2_pf_hit_l2.l2_hwpf,l2_pf_miss_l2_hit_l3.l2_hwpf,l2_pf_miss_l2_l3.l2_hwpf \
        "$BENCH_BIN"

uprof-v6-leaf-strategies QUERY='nearest' STRATEGY='arena' POINTS='4194304' QUERIES='1000' REPEATS='2000' FEATURES='simd,test_utils' OUT='./uprof-output-v6-leaf-strategies':
    cargo build --release --features {{FEATURES}} --bench profile_v6_leaf_strategies
    BENCH_BIN=$(find target/release/deps -maxdepth 1 -type f -perm -111 -name 'profile_v6_leaf_strategies-*' | head -n 1) && \
    KIDDO_PROFILE_QUERY_KIND={{QUERY}} \
    KIDDO_PROFILE_STRATEGY={{STRATEGY}} \
    KIDDO_PROFILE_POINTS={{POINTS}} \
    KIDDO_PROFILE_QUERIES={{QUERIES}} \
    KIDDO_PROFILE_QUERY_BATCH_REPEATS={{REPEATS}} \
    /opt/AMD/AMDuProf_Linux_x64_5.1.701/bin/AMDuProfCLI collect \
        --config ibs \
        --interval 10000 \
        --format csv \
        -w /home/scotty/projects/kiddo \
        -o {{OUT}} \
        "$BENCH_BIN"

build-profile-v5-nearest-one-eytz:
    cargo build --release --features profile_v5 --bin profile_v5_nearest_one_eytzinger

profile-v5-nearest-one-eytz-samply: build-profile-v5-nearest-one-eytz
    KIDDO_PROFILE_QUERY_BATCH_REPEATS=2000 samply record ./target/release/profile_v5_nearest_one_eytzinger


build:
    RUSTFLAGS="-C target-cpu=znver3 -C opt-level=2" cargo build --release --example immutable-large-ann-donnelly --example immutable-large-ann-eytzinger

cg-donnelly: build
    valgrind --tool=cachegrind --branch-sim=yes --cache-sim=yes \
             --cachegrind-out-file=cachegrind.out.donnelly \
             target/release/examples/immutable-large-ann-donnelly
    cg_annotate cachegrind.out.donnelly > cachegrind.annot.donnelly

cg-eytzinger: build
    valgrind --tool=cachegrind --branch-sim=yes --cache-sim=yes \
             --cachegrind-out-file=cachegrind.out.eytzinger \
             target/release/examples/immutable-large-ann-eytzinger
    cg_annotate cachegrind.out.eytzinger > cachegrind.annot.eytzinger

cg-diff: cg-donnelly cg-eytzinger
    cg_diff cachegrind.out.donnelly cachegrind.out.eytzinger \
      | cg_annotate > cachegrind.diff.txt
    @echo "Diff written to cachegrind.diff.txt"

perf-donnelly:
    perf stat -e cycles,instructions,L1-dcache-load-misses,LLC-load-misses,branch-misses ./target/release/examples/immutable-large-ann-donnelly

perf-eytzinger:
    perf stat -e cycles,instructions,L1-dcache-load-misses,LLC-load-misses,branch-misses ./target/release/examples/immutable-large-ann-eytzinger


uprof-eytzinger:
    /opt/AMD/AMDuProf_Linux_x64_5.1.701/bin/AMDuProfCLI collect \
        --config ibs \
        --interval 10000 \
        --format csv \
        -w /home/scotty/projects/kiddo \
        -o ./uprof-output-eytz \
        target/release/examples/immutable-large-ann-eytzinger-deserialize-and-query

uprof-donnelly:
    /opt/AMD/AMDuProf_Linux_x64_5.1.701/bin/AMDuProfCLI collect \
        --config ibs \
        --interval 10000 \
        --format csv \
        -w /home/scotty/projects/kiddo \
        -o ./uprof-output-eytz \
        target/release/examples/immutable-large-ann-donnelly-deserialize-and-query

# Criterion reproduction and decomposition of the 1k-to-4k exact-NN crossover.
# The stored and stateless generated-by-index modes use identical SplitMix64
# queries, allowing query-array footprint to be separated from the repeated
# tree-path working set.
bench-v6-donnelly-vs-eytzinger-query-pool POINT_LOG2='27' POOL_SIZES='256,512,1000,2048,4096,8192,16384' WARMUP='3' MEASUREMENT='8' SAMPLE_SIZE='30' RESULT_KEY='query-pool' OUTPUT_DIR='./focused-results' AXIS='both':
    #!/usr/bin/env bash
    set -euo pipefail
    result_key={{quote(RESULT_KEY)}}
    output_dir={{quote(OUTPUT_DIR)}}
    if [[ ! "$result_key" =~ ^[A-Za-z0-9][A-Za-z0-9._+:-]*$ ]]; then
        echo "RESULT_KEY must contain only letters, digits, '.', '_', '+', ':', or '-'" >&2
        exit 2
    fi
    RUSTC_WRAPPER= \
    KIDDO_QUERY_POOL_POINT_LOG2={{quote(POINT_LOG2)}} \
    KIDDO_QUERY_POOL_SIZES={{quote(POOL_SIZES)}} \
    KIDDO_QUERY_POOL_AXIS={{quote(AXIS)}} \
    RUSTFLAGS='-C target-cpu=native' \
        cargo bench \
        --bench profile_v6_query_pool \
        --features simd,test_utils,logging_off \
        -- \
        profile_v6_query_pool \
        --warm-up-time {{quote(WARMUP)}} \
        --measurement-time {{quote(MEASUREMENT)}} \
        --sample-size {{quote(SAMPLE_SIZE)}} \
        --noplot

    mkdir -p -- "$output_dir"
    cargo run --quiet --manifest-path tools/criterion-export/Cargo.toml -- \
        target/criterion \
        "$output_dir/bench_result-v6-donnelly-vs-eytzinger-query-pool-${result_key}.json" \
        profile_v6_query_pool

# IRQ-audited variant. Compilation occurs first; the benchmark executable and
# all Criterion records live on tmpfs during the CPU-8 measurement interval.
bench-v6-donnelly-vs-eytzinger-query-pool-clean POINT_LOG2='27' POOL_SIZES='256,512,1000,2048,4096,8192,16384' WARMUP='3' MEASUREMENT='8' SAMPLE_SIZE='30' RESULT_KEY='query-pool' OUTPUT_DIR='./focused-results' AXIS='both':
    #!/usr/bin/env bash
    set -euo pipefail
    point_log2={{quote(POINT_LOG2)}}
    pool_sizes={{quote(POOL_SIZES)}}
    warmup={{quote(WARMUP)}}
    measurement={{quote(MEASUREMENT)}}
    sample_size={{quote(SAMPLE_SIZE)}}
    result_key={{quote(RESULT_KEY)}}
    output_dir={{quote(OUTPUT_DIR)}}
    axis={{quote(AXIS)}}
    if [[ ! "$result_key" =~ ^[A-Za-z0-9][A-Za-z0-9._+:-]*$ ]]; then
        echo "RESULT_KEY must contain only letters, digits, '.', '_', '+', ':', or '-'" >&2
        exit 2
    fi

    benchmark_exe="$(
        RUSTC_WRAPPER= RUSTFLAGS='-C target-cpu=native' \
            cargo bench \
            --bench profile_v6_query_pool \
            --features simd,test_utils,logging_off \
            --no-run \
            --message-format=json |
        jq -r \
            'select(.reason == "compiler-artifact" and .target.name == "profile_v6_query_pool") | .executable // empty' |
        tail -n 1
    )"
    if [[ -z "$benchmark_exe" || ! -x "$benchmark_exe" ]]; then
        echo "Could not resolve the compiled query-pool benchmark executable" >&2
        exit 1
    fi

    run_dir="$(mktemp -d /dev/shm/kiddo-query-pool.XXXXXX)"
    trap 'rm -rf -- "$run_dir/criterion"; rm -f -- "$run_dir/benchmark"; rmdir -- "$run_dir"' EXIT
    cp -- "$benchmark_exe" "$run_dir/benchmark"
    chmod 0755 "$run_dir/benchmark"

    set +e
    bench-profile-run env \
        CRITERION_HOME="$run_dir/criterion" \
        KIDDO_QUERY_POOL_POINT_LOG2="$point_log2" \
        KIDDO_QUERY_POOL_SIZES="$pool_sizes" \
        KIDDO_QUERY_POOL_AXIS="$axis" \
        "$run_dir/benchmark" \
        profile_v6_query_pool \
        --warm-up-time "$warmup" \
        --measurement-time "$measurement" \
        --sample-size "$sample_size" \
        --noplot \
        --bench
    run_status=$?
    set -e

    if [[ ! -d "$run_dir/criterion" ]]; then
        echo "Criterion produced no records under $run_dir/criterion" >&2
        if (( run_status == 0 )); then
            run_status=1
        fi
    else
        mkdir -p -- "$output_dir"
        cargo run --quiet --manifest-path tools/criterion-export/Cargo.toml -- \
            "$run_dir/criterion" \
            "$output_dir/bench_result-v6-donnelly-vs-eytzinger-query-pool-${result_key}.json" \
            profile_v6_query_pool
    fi
    exit "$run_status"

# Screen exact-NN Donnelly traversal variants using the same stored/generated
# query-pool workload as the paper comparison.
bench-v6-donnelly-variants-query-pool-clean POINT_LOG2='27' POOL_SIZES='256,512,1000,2048,4096,8192,16384' WARMUP='3' MEASUREMENT='8' SAMPLE_SIZE='30' RESULT_KEY='donnelly-variants' OUTPUT_DIR='./focused-results' AXIS='both' STRATEGIES='eytzinger,donnelly,donnelly_unrolled,donnelly_unrolled_block_dim,donnelly_simd_descent,donnelly_cyclic_simd_descent,donnelly_simd_initial_descent,donnelly_simd_full':
    KIDDO_QUERY_POOL_STRATEGIES={{quote(STRATEGIES)}} \
        just bench-v6-donnelly-vs-eytzinger-query-pool-clean \
        {{quote(POINT_LOG2)}} {{quote(POOL_SIZES)}} {{quote(WARMUP)}} \
        {{quote(MEASUREMENT)}} {{quote(SAMPLE_SIZE)}} {{quote(RESULT_KEY)}} \
        {{quote(OUTPUT_DIR)}} {{quote(AXIS)}}

chart-v6-donnelly-variants-query-pool RESULT_KEY RESULTS_DIR='./focused-results' OUTPUT_DIR='./focused-charts' PYTHON='python3':
    {{quote(PYTHON)}} scripts/chart_donnelly_variant_results.py charts \
        --result {{quote(RESULTS_DIR)}}/bench_result-v6-donnelly-vs-eytzinger-query-pool-{{quote(RESULT_KEY)}}.json \
        --result-label {{quote(RESULT_KEY)}} \
        --output-dir {{quote(OUTPUT_DIR)}}

html-v6-donnelly-variants-query-pool RESULT_KEY RESULTS_DIR='./focused-results' OUTPUT_DIR='./focused-charts' HTML_NAME='donnelly-variant-screen.html' PYTHON='python3':
    {{quote(PYTHON)}} scripts/chart_donnelly_variant_results.py all \
        --result {{quote(RESULTS_DIR)}}/bench_result-v6-donnelly-vs-eytzinger-query-pool-{{quote(RESULT_KEY)}}.json \
        --result-label {{quote(RESULT_KEY)}} \
        --output-dir {{quote(OUTPUT_DIR)}} \
        --html-name {{quote(HTML_NAME)}}

view-v6-donnelly-variants-query-pool RESULT_KEY RESULTS_DIR='./focused-results' OUTPUT_DIR='./focused-charts' HTML_NAME='donnelly-variant-screen.html' PYTHON='python3':
    just html-v6-donnelly-variants-query-pool {{quote(RESULT_KEY)}} {{quote(RESULTS_DIR)}} {{quote(OUTPUT_DIR)}} {{quote(HTML_NAME)}} {{quote(PYTHON)}}
    xdg-open {{quote(OUTPUT_DIR)}}/{{quote(HTML_NAME)}}

# Run the Donnelly variant screen across several tree heights, retrying an
# interval if the benchmark-core IRQ audit invalidates it, then merge and chart.
bench-v6-donnelly-variants-height-sweep-clean POINT_LOG2S='25,26,27' POOL_SIZES='256,512,1000,2048,4096,8192,16384' WARMUP='3' MEASUREMENT='8' SAMPLE_SIZE='30' RESULT_KEY='donnelly-variants-height-sweep' OUTPUT_DIR='./focused-results' CHART_DIR='./focused-charts' AXIS='both' IRQ_RETRIES='2' STRATEGIES='eytzinger,donnelly,donnelly_unrolled,donnelly_unrolled_block_dim,donnelly_simd_descent,donnelly_cyclic_simd_descent,donnelly_simd_initial_descent,donnelly_simd_full':
    #!/usr/bin/env bash
    set -euo pipefail
    IFS=',' read -ra heights <<< {{quote(POINT_LOG2S)}}
    result_key={{quote(RESULT_KEY)}}
    output_dir={{quote(OUTPUT_DIR)}}
    chart_dir={{quote(CHART_DIR)}}
    irq_retries={{quote(IRQ_RETRIES)}}
    strategies={{quote(STRATEGIES)}}
    mkdir -p -- "$output_dir" "$chart_dir"
    inputs=()
    for height in "${heights[@]}"; do
        [[ "$height" =~ ^[0-9]+$ ]] || { echo "invalid point height: $height" >&2; exit 2; }
        cell_key="${result_key}-2p${height}"
        attempt=0
        while true; do
            set +e
            just bench-v6-donnelly-variants-query-pool-clean \
                "$height" {{quote(POOL_SIZES)}} {{quote(WARMUP)}} \
                {{quote(MEASUREMENT)}} {{quote(SAMPLE_SIZE)}} "$cell_key" \
                "$output_dir" {{quote(AXIS)}} "$strategies"
            status=$?
            set -e
            if (( status != 125 )); then
                (( status == 0 )) || exit "$status"
                break
            fi
            if (( attempt >= irq_retries )); then
                echo "IRQ retry limit reached for 2^$height" >&2
                exit 125
            fi
            attempt=$((attempt + 1))
            echo "Retrying IRQ-invalidated 2^$height interval ($attempt/$irq_retries)"
        done
        inputs+=("$output_dir/bench_result-v6-donnelly-vs-eytzinger-query-pool-${cell_key}.json")
    done
    merged="$output_dir/bench_result-v6-donnelly-vs-eytzinger-query-pool-${result_key}.json"
    jq -s --arg criterion_root "merged:$result_key" '
        {
            schema_version: 1,
            criterion_root: $criterion_root,
            collected_at_unix_ms: ([.[].collected_at_unix_ms] | max),
            filters: ([.[].filters[]] | unique),
            results: ([.[].results[]] | sort_by(.benchmark))
        }
    ' "${inputs[@]}" > "$merged"
    python3 scripts/chart_donnelly_variant_results.py all \
        --result "$merged" --result-label "$result_key" \
        --output-dir "$chart_dir" --html-name "${result_key}.html"
    echo "merged results: $merged"
    echo "charts: $chart_dir/${result_key}.html"

chart-v6-donnelly-vs-eytzinger-query-pool RESULT_KEY RESULTS_DIR='./focused-results' OUTPUT_DIR='./focused-charts' PYTHON='python3':
    {{quote(PYTHON)}} scripts/chart_query_pool_results.py charts \
        --result {{quote(RESULTS_DIR)}}/bench_result-v6-donnelly-vs-eytzinger-query-pool-{{quote(RESULT_KEY)}}.json \
        --result-label {{quote(RESULT_KEY)}} \
        --output-dir {{quote(OUTPUT_DIR)}}

# Publication-style cache-context charts. The third panel is an analytical
# lower bound, not measured cache occupancy or a fitted miss model.
chart-v6-donnelly-vs-eytzinger-query-pool-cache-context RESULT_KEY RESULTS_DIR='./focused-results' OUTPUT_DIR='./focused-charts' PYTHON='python3':
    {{quote(PYTHON)}} scripts/chart_query_pool_cache_context.py \
        --result {{quote(RESULTS_DIR)}}/bench_result-v6-donnelly-vs-eytzinger-query-pool-{{quote(RESULT_KEY)}}.json \
        --output-dir {{quote(OUTPUT_DIR)}}

html-v6-donnelly-vs-eytzinger-query-pool RESULT_KEY RESULTS_DIR='./focused-results' OUTPUT_DIR='./focused-charts' HTML_NAME='donnelly-vs-eytzinger-query-pool.html' PYTHON='python3':
    {{quote(PYTHON)}} scripts/chart_query_pool_results.py all \
        --result {{quote(RESULTS_DIR)}}/bench_result-v6-donnelly-vs-eytzinger-query-pool-{{quote(RESULT_KEY)}}.json \
        --result-label {{quote(RESULT_KEY)}} \
        --output-dir {{quote(OUTPUT_DIR)}} \
        --html-name {{quote(HTML_NAME)}}

view-v6-donnelly-vs-eytzinger-query-pool RESULT_KEY RESULTS_DIR='./focused-results' OUTPUT_DIR='./focused-charts' HTML_NAME='donnelly-vs-eytzinger-query-pool.html' PYTHON='python3':
    just html-v6-donnelly-vs-eytzinger-query-pool {{quote(RESULT_KEY)}} {{quote(RESULTS_DIR)}} {{quote(OUTPUT_DIR)}} {{quote(HTML_NAME)}} {{quote(PYTHON)}}
    xdg-open {{quote(OUTPUT_DIR)}}/{{quote(HTML_NAME)}}

# Fixed-work exact-NN counter run. Construction and warmup finish before the
# benchmark SIGSTOPs; perf attaches to the stopped process and only then resumes
# the measured query loop. Keep exact_query_stats disabled here.
perf-v6-donnelly-vs-eytzinger-query-pool POINT_LOG2='27' POOL_SIZE='2048' TOTAL_QUERIES='4000000' WARMUP_REPEATS='2' AXIS='f64' STRATEGY='eytzinger' EVENT_SET='cache' RESULT_KEY='query-pool-perf' OUTPUT_DIR='./focused-results':
    #!/usr/bin/env bash
    set -euo pipefail
    point_log2={{quote(POINT_LOG2)}}
    pool_size={{quote(POOL_SIZE)}}
    total_queries={{quote(TOTAL_QUERIES)}}
    warmup_repeats={{quote(WARMUP_REPEATS)}}
    axis={{quote(AXIS)}}
    strategy={{quote(STRATEGY)}}
    event_set={{quote(EVENT_SET)}}
    result_key={{quote(RESULT_KEY)}}
    output_dir={{quote(OUTPUT_DIR)}}
    if [[ ! "$result_key" =~ ^[A-Za-z0-9][A-Za-z0-9._+:-]*$ ]]; then
        echo "RESULT_KEY contains unsupported characters" >&2
        exit 2
    fi
    case "$axis" in f32|f64) ;; *) echo "AXIS must be f32 or f64" >&2; exit 2 ;; esac
    case "$strategy" in
        eytzinger|donnelly|donnelly_unrolled|donnelly_unrolled_block_dim|donnelly_simd_descent|donnelly_cyclic_simd_descent|donnelly_simd_initial_descent|donnelly_simd_full) ;;
        *) echo "unsupported STRATEGY: $strategy" >&2; exit 2 ;;
    esac
    events="$(scripts/query_pool_perf_events.sh "$event_set")"

    benchmark_exe="$(
        RUSTC_WRAPPER= RUSTFLAGS='-C target-cpu=native' \
            cargo bench --bench profile_v6_query_pool_perf \
            --features simd,test_utils,logging_off --no-run --message-format=json |
        jq -r 'select(.reason == "compiler-artifact" and .target.name == "profile_v6_query_pool_perf") | .executable // empty' |
        tail -n 1
    )"
    if [[ -z "$benchmark_exe" || ! -x "$benchmark_exe" ]]; then
        echo "Could not resolve profile_v6_query_pool_perf executable" >&2
        exit 1
    fi

    run_dir="$(mktemp -d /dev/shm/kiddo-query-pool-perf-result.XXXXXX)"
    trap 'rm -rf -- "$run_dir"' EXIT
    temporary_result="$run_dir/perf.csv"
    final_stem="${result_key}-${axis}-${strategy}-q${pool_size}-${event_set}"

    bench-profile-run env \
        KIDDO_PERF_POINT_LOG2="$point_log2" \
        KIDDO_PERF_POOL_SIZE="$pool_size" \
        KIDDO_PERF_TOTAL_QUERIES="$total_queries" \
        KIDDO_PERF_WARMUP_REPEATS="$warmup_repeats" \
        KIDDO_PERF_AXIS="$axis" \
        KIDDO_PERF_STRATEGY="$strategy" \
        scripts/run_query_pool_perf.sh "$benchmark_exe" "$temporary_result" "$events"

    mkdir -p -- "$output_dir"
    cp -- "$temporary_result" "$output_dir/${final_stem}.perf.csv"
    cp -- "${temporary_result%.csv}.run.txt" "$output_dir/${final_stem}.run.txt"
    echo "saved $output_dir/${final_stem}.perf.csv"

# A convenient paired sweep around the observed private-cache transition.
perf-v6-donnelly-vs-eytzinger-query-pool-sweep POINT_LOG2='27' POOL_SIZES='1000,2048,4096,8192,16384' TOTAL_QUERIES='4000000' WARMUP_REPEATS='2' EVENT_SET='cache' RESULT_KEY='query-pool-perf' OUTPUT_DIR='./focused-results':
    #!/usr/bin/env bash
    set -euo pipefail
    IFS=',' read -ra pools <<< {{quote(POOL_SIZES)}}
    for axis in f64 f32; do
        for pool in "${pools[@]}"; do
            for strategy in eytzinger donnelly; do
                just perf-v6-donnelly-vs-eytzinger-query-pool \
                    {{quote(POINT_LOG2)}} "$pool" {{quote(TOTAL_QUERIES)}} \
                    {{quote(WARMUP_REPEATS)}} "$axis" "$strategy" \
                    {{quote(EVENT_SET)}} {{quote(RESULT_KEY)}} {{quote(OUTPUT_DIR)}}
            done
        done
    done

# Structural counts use a separate, deliberately invasive build. The elapsed
# time printed by this task is diagnostic only and must not enter paper plots.
stats-v6-donnelly-vs-eytzinger-query-pool POINT_LOG2='27' POOL_SIZE='2048' TOTAL_QUERIES='200000' WARMUP_REPEATS='1' AXIS='f64' STRATEGY='eytzinger' RESULT_KEY='query-pool-stats' OUTPUT_DIR='./focused-results':
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p -- {{quote(OUTPUT_DIR)}}
    output={{quote(OUTPUT_DIR)}}/{{quote(RESULT_KEY)}}-{{quote(AXIS)}}-{{quote(STRATEGY)}}-q{{quote(POOL_SIZE)}}.txt
    RUSTC_WRAPPER= RUSTFLAGS='-C target-cpu=native' \
    KIDDO_PERF_POINT_LOG2={{quote(POINT_LOG2)}} \
    KIDDO_PERF_POOL_SIZE={{quote(POOL_SIZE)}} \
    KIDDO_PERF_TOTAL_QUERIES={{quote(TOTAL_QUERIES)}} \
    KIDDO_PERF_WARMUP_REPEATS={{quote(WARMUP_REPEATS)}} \
    KIDDO_PERF_AXIS={{quote(AXIS)}} \
    KIDDO_PERF_STRATEGY={{quote(STRATEGY)}} \
        cargo bench --bench profile_v6_query_pool_perf \
        --features simd,test_utils,logging_off,exact_query_stats -- 2>&1 | tee "$output"
    echo "saved $output"

stats-v6-donnelly-vs-eytzinger-query-pool-sweep POINT_LOG2='27' POOL_SIZES='1000,2048,4096,8192,16384' TOTAL_QUERIES='200000' WARMUP_REPEATS='1' RESULT_KEY='query-pool-stats' OUTPUT_DIR='./focused-results':
    #!/usr/bin/env bash
    set -euo pipefail
    IFS=',' read -ra pools <<< {{quote(POOL_SIZES)}}
    for axis in f64 f32; do
        for pool in "${pools[@]}"; do
            for strategy in eytzinger donnelly; do
                just stats-v6-donnelly-vs-eytzinger-query-pool \
                    {{quote(POINT_LOG2)}} "$pool" {{quote(TOTAL_QUERIES)}} \
                    {{quote(WARMUP_REPEATS)}} "$axis" "$strategy" \
                    {{quote(RESULT_KEY)}} {{quote(OUTPUT_DIR)}}
            done
        done
    done

#!/usr/bin/env bash
set -euo pipefail

# Controlled exact-NN matrix for phase-aware cyclic SIMD. The four cells pair a
# native dimension/block-height case with an awkward dimension count, and an
# exact block height with a root-padded height, for both scalar widths.

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

POOL_SIZES=${POOL_SIZES:-256,512,1000,2048,4096,8192,16384}
CRITERION_WARMUP=${CRITERION_WARMUP:-3}
CRITERION_MEASUREMENT=${CRITERION_MEASUREMENT:-8}
CRITERION_SAMPLE_SIZE=${CRITERION_SAMPLE_SIZE:-30}
STRATEGIES=${STRATEGIES:-eytzinger,donnelly_cyclic_simd_descent,donnelly_cyclic_simd_full}
IRQ_RETRIES=${IRQ_RETRIES:-2}
PERF_POOL_SIZE=${PERF_POOL_SIZE:-4096}
PERF_TOTAL_QUERIES=${PERF_TOTAL_QUERIES:-4000000}
PERF_WARMUP_REPEATS=${PERF_WARMUP_REPEATS:-2}
PERF_EVENT_SETS=${PERF_EVENT_SETS:-core,cache,tlb}
OUTPUT_BASE=${OUTPUT_BASE:-$REPO_DIR/focused-results}
CHART_BASE=${CHART_BASE:-$REPO_DIR/focused-charts}
SUITE_LABEL=${SUITE_LABEL:-donnelly-cyclic-phase-matrix-$(date -u +%Y%m%dT%H%M%SZ)}

readonly RUN_DIR="$OUTPUT_BASE/$SUITE_LABEL"
readonly CHART_DIR="$CHART_BASE/$SUITE_LABEL"
readonly RESULT_FILE="$RUN_DIR/bench_result-v6-donnelly-cyclic-phase-$SUITE_LABEL.json"
readonly HTML_FILE="$CHART_DIR/$SUITE_LABEL.html"
readonly LOG_FILE="$RUN_DIR/run.log"

for command_name in bench-profile-run bench-profile-status cargo jq just mktemp perf python3 rustc; do
    command -v "$command_name" >/dev/null || {
        echo "Required command is unavailable: $command_name" >&2
        exit 2
    }
done
[[ "$SUITE_LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._+:-]*$ ]] || {
    echo "SUITE_LABEL contains unsupported characters" >&2
    exit 2
}
[[ ! -e "$RUN_DIR" && ! -e "$CHART_DIR" ]] || {
    echo "Refusing to overwrite existing suite $SUITE_LABEL" >&2
    exit 2
}

IFS=',' read -r -a pools <<<"$POOL_SIZES"
IFS=',' read -r -a strategies <<<"$STRATEGIES"
IFS=',' read -r -a perf_event_sets <<<"$PERF_EVENT_SETS"
readonly -a supported=(
    eytzinger donnelly_cyclic_simd_descent donnelly_cyclic_simd_full
)
for strategy in "${strategies[@]}"; do
    [[ " ${supported[*]} " == *" $strategy "* ]] || {
        echo "Unsupported strategy: $strategy" >&2
        exit 2
    }
done
for pool_size in "${pools[@]}"; do
    [[ "$pool_size" =~ ^[1-9][0-9]*$ ]] || {
        echo "POOL_SIZES entries must be positive integers: $pool_size" >&2
        exit 2
    }
done
for value_name in CRITERION_WARMUP CRITERION_MEASUREMENT; do
    value=${!value_name}
    [[ "$value" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || {
        echo "$value_name must be a non-negative number: $value" >&2
        exit 2
    }
done
[[ "$CRITERION_SAMPLE_SIZE" =~ ^[1-9][0-9]*$ ]] || {
    echo "CRITERION_SAMPLE_SIZE must be a positive integer" >&2
    exit 2
}
[[ "$IRQ_RETRIES" =~ ^[0-9]+$ ]] || {
    echo "IRQ_RETRIES must be a non-negative integer" >&2
    exit 2
}
for value_name in PERF_POOL_SIZE PERF_TOTAL_QUERIES PERF_WARMUP_REPEATS; do
    value=${!value_name}
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
        echo "$value_name must be a positive integer" >&2
        exit 2
    }
done
for event_set in "${perf_event_sets[@]}"; do
    case "$event_set" in
        core|cache|tlb) ;;
        *) echo "Unsupported PERF_EVENT_SETS entry: $event_set" >&2; exit 2 ;;
    esac
done
[[ " ${strategies[*]} " == *" eytzinger "* ]] || {
    echo "STRATEGIES must retain the Eytzinger control" >&2
    exit 2
}

native_cfg=$(rustc --print cfg -C target-cpu=native)
[[ "$native_cfg" == *'target_feature="avx512f"'* ]] || {
    echo "This experiment requires AVX-512F" >&2
    exit 2
}

mkdir -p -- "$RUN_DIR" "$CHART_DIR"
capture_profile_status() {
    local destination=$1 phase=$2 status
    set +e
    bench-profile-status >"$destination" 2>&1
    status=$?
    set -e
    if (( status != 0 )); then
        cat "$destination" >&2
        echo >&2
        echo "Benchmark profile validation failed $phase; refusing to continue." >&2
        echo "Full status was saved to $destination" >&2
        return "$status"
    fi
}
capture_profile_status "$RUN_DIR/bench-profile-status-before.txt" "before the suite"
for event_set in "${perf_event_sets[@]}"; do
    "$SCRIPT_DIR/query_pool_perf_events.sh" --check "$event_set"
done

benchmark_exe="$(
    cd -- "$REPO_DIR"
    RUSTC_WRAPPER= RUSTFLAGS='-C target-cpu=native' \
        cargo bench --offline --bench profile_v6_cyclic_query_pool \
        --features simd,test_utils,logging_off --no-run --message-format=json |
        jq -r \
            'select(.reason == "compiler-artifact" and .target.name == "profile_v6_cyclic_query_pool") | .executable // empty' |
        tail -n 1
)"
[[ -n "$benchmark_exe" && -x "$benchmark_exe" ]] || {
    echo "Could not resolve profile_v6_cyclic_query_pool executable" >&2
    exit 1
}

run_irq_retry_logged() {
    local attempt=0 status
    while true; do
        set +e
        (cd -- "$REPO_DIR"; "$@") 2>&1 | tee -a "$LOG_FILE"
        status=${PIPESTATUS[0]}
        set -e
        if (( status != 125 || attempt >= IRQ_RETRIES )); then
            return "$status"
        fi
        attempt=$((attempt + 1))
        echo "Retrying IRQ-invalidated perf interval ($attempt/$IRQ_RETRIES)" |
            tee -a "$LOG_FILE"
    done
}

# B=32 removes five levels. p23 f64 and p25 f32 are exact native block
# boundaries; p22 f64 and p24 f32 each require one cyclic root-padding level.
readonly -a cell_axes=(f64 f64 f32 f32)
readonly -a cell_dimensions=(3 4 4 3)
readonly -a cell_heights=(23 22 25 24)
readonly -a cell_roles=(
    f64-k3-exact
    f64-k4-padded
    f32-k4-exact
    f32-k3-padded
)

run_cell() {
    local axis=$1 dimensions=$2 height=$3 role=$4
    local pool_csv=${5:-$POOL_SIZES}
    local strategy_csv=${6:-$STRATEGIES}
    local warmup=${7:-$CRITERION_WARMUP}
    local measurement=${8:-$CRITERION_MEASUREMENT}
    local sample_size=${9:-$CRITERION_SAMPLE_SIZE}
    local output_dir=${10:-$RUN_DIR}
    local attempt=0 status result_key result_path tmp_dir

    result_key="$SUITE_LABEL-$role-2p$height"
    result_path="$output_dir/bench_result-v6-donnelly-cyclic-phase-$result_key.json"
    while true; do
        tmp_dir="$(mktemp -d /dev/shm/kiddo-cyclic-phase.XXXXXX)"
        cp -- "$benchmark_exe" "$tmp_dir/benchmark"
        chmod 0755 "$tmp_dir/benchmark"

        set +e
        bench-profile-run env \
            CRITERION_HOME="$tmp_dir/criterion" \
            KIDDO_CYCLIC_AXIS="$axis" \
            KIDDO_CYCLIC_DIMENSIONS="$dimensions" \
            KIDDO_CYCLIC_POINT_LOG2="$height" \
            KIDDO_CYCLIC_POOL_SIZES="$pool_csv" \
            KIDDO_CYCLIC_STRATEGIES="$strategy_csv" \
            "$tmp_dir/benchmark" \
            profile_v6_cyclic_query_pool \
            --warm-up-time "$warmup" \
            --measurement-time "$measurement" \
            --sample-size "$sample_size" \
            --noplot --bench 2>&1 | tee -a "$LOG_FILE" >&2
        status=${PIPESTATUS[0]}
        set -e

        if (( status == 0 )); then
            (
                cd -- "$REPO_DIR"
                cargo run --quiet --offline \
                    --manifest-path tools/criterion-export/Cargo.toml -- \
                    "$tmp_dir/criterion" "$result_path" \
                    profile_v6_cyclic_query_pool >&2
            )
            rm -rf -- "$tmp_dir"
            printf '%s\n' "$result_path"
            return 0
        fi

        rm -rf -- "$tmp_dir"
        if (( status != 125 || attempt >= IRQ_RETRIES )); then
            return "$status"
        fi
        attempt=$((attempt + 1))
        echo "Retrying IRQ-invalidated cell $role ($attempt/$IRQ_RETRIES)" |
            tee -a "$LOG_FILE"
    done
}

# Exercise the exact executable, isolation wrapper, exporter, and chart parser
# before committing to the long matrix. This takes seconds and catches CLI or
# Criterion-format drift while the run is still cheap to restart.
readonly PREFLIGHT_DIR="$(mktemp -d /dev/shm/kiddo-cyclic-preflight.XXXXXX)"
cleanup_preflight() {
    rm -rf -- "$PREFLIGHT_DIR"
}
trap cleanup_preflight EXIT INT TERM
preflight_result=$(run_cell f64 4 10 preflight 16 "$STRATEGIES" 0.01 0.02 10 "$PREFLIGHT_DIR")
preflight_expected=$((${#strategies[@]} * 2))
preflight_actual=$(jq '.results | length' "$preflight_result")
if (( preflight_actual != preflight_expected )); then
    echo "Preflight produced $preflight_actual entries; expected $preflight_expected" >&2
    exit 1
fi
python3 "$SCRIPT_DIR/chart_donnelly_variant_results.py" all \
    --result "$preflight_result" --result-label "$SUITE_LABEL-preflight" \
    --output-dir "$PREFLIGHT_DIR/charts" --html-name preflight.html
run_irq_retry_logged just perf-v6-donnelly-vs-eytzinger-query-pool \
    10 16 128 1 f64 eytzinger core "$SUITE_LABEL-preflight-perf" "$PREFLIGHT_DIR"
preflight_perf="$PREFLIGHT_DIR/$SUITE_LABEL-preflight-perf-f64-eytzinger-q16-core.perf.csv"
"$SCRIPT_DIR/query_pool_perf_events.sh" --validate "$preflight_perf"
cleanup_preflight
trap - EXIT INT TERM

inputs=()
for index in "${!cell_axes[@]}"; do
    result_path=$(run_cell \
        "${cell_axes[index]}" \
        "${cell_dimensions[index]}" \
        "${cell_heights[index]}" \
        "${cell_roles[index]}")
    inputs+=("$result_path")
done

jq -s --arg criterion_root "merged:$SUITE_LABEL" '
    {
        schema_version: 1,
        criterion_root: $criterion_root,
        collected_at_unix_ms: ([.[].collected_at_unix_ms] | max),
        filters: ([.[].filters[]] | unique),
        results: ([.[].results[]] | sort_by(.benchmark))
    }
' "${inputs[@]}" >"$RESULT_FILE"

expected_per_cell=$((${#pools[@]} * ${#strategies[@]} * 2))
expected_total=$((expected_per_cell * ${#cell_axes[@]}))
actual_total=$(jq '.results | length' "$RESULT_FILE")
if (( actual_total != expected_total )); then
    echo "Merged result has $actual_total entries; expected $expected_total" >&2
    exit 1
fi

python3 "$SCRIPT_DIR/chart_donnelly_variant_results.py" all \
    --result "$RESULT_FILE" --result-label "$SUITE_LABEL" \
    --output-dir "$CHART_DIR" --html-name "$SUITE_LABEL.html"

# One exact-block-height counter pass per scalar width. Counter classes remain
# separate so perf never multiplexes or scales the publication evidence.
readonly -a perf_axes=(f64 f32)
readonly -a perf_heights=(23 25)
for perf_index in "${!perf_axes[@]}"; do
    axis=${perf_axes[perf_index]}
    height=${perf_heights[perf_index]}
    for strategy in "${strategies[@]}"; do
        for event_set in "${perf_event_sets[@]}"; do
            result_key="$SUITE_LABEL-exact-height"
            run_irq_retry_logged just perf-v6-donnelly-vs-eytzinger-query-pool \
                "$height" "$PERF_POOL_SIZE" "$PERF_TOTAL_QUERIES" \
                "$PERF_WARMUP_REPEATS" "$axis" "$strategy" "$event_set" \
                "$result_key" "$RUN_DIR"
            perf_csv="$RUN_DIR/$result_key-$axis-$strategy-q$PERF_POOL_SIZE-$event_set.perf.csv"
            "$SCRIPT_DIR/query_pool_perf_events.sh" --validate "$perf_csv"
        done
    done
done

capture_profile_status "$RUN_DIR/bench-profile-status-after.txt" "after the suite"
{
    echo "suite_label=$SUITE_LABEL"
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "pool_sizes=$POOL_SIZES"
    echo "strategies=$STRATEGIES"
    echo "perf_axes=f64:2p23,f32:2p25"
    echo "perf_pool_size=$PERF_POOL_SIZE"
    echo "perf_total_queries=$PERF_TOTAL_QUERIES"
    echo "perf_warmup_repeats=$PERF_WARMUP_REPEATS"
    echo "perf_event_sets=$PERF_EVENT_SETS"
    echo "cells=f64-k3-p23,f64-k4-p22,f32-k4-p25,f32-k3-p24"
    echo "result_count=$actual_total"
    echo "result_file=$RESULT_FILE"
    echo "chart_file=$HTML_FILE"
    echo "git_head=$(git -C "$REPO_DIR" rev-parse HEAD)"
    echo "rustc=$(rustc --version)"
} >"$RUN_DIR/manifest.txt"

echo
echo "Cyclic phase matrix complete"
echo "Results: $RESULT_FILE"
echo "Charts:  $HTML_FILE"

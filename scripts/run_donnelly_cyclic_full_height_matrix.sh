#!/usr/bin/env bash
set -euo pipefail

# Publication-oriented exact-nearest-one matrix for the final cyclic Donnelly
# strategies. Every tree size is measured at the two query-pool sizes around
# the previously observed cache crossover. Selected exact-height/worst-padding
# pairs receive the complete query-pool sweep.

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

HEIGHTS=${HEIGHTS:-20,21,22,23,24,25,26,27,28,29}
BASE_POOL_SIZES=${BASE_POOL_SIZES:-1000,4096}
DENSE_POOL_SIZES=${DENSE_POOL_SIZES:-256,512,1000,2048,4096,8192,16384}
F64_DENSE_HEIGHTS=${F64_DENSE_HEIGHTS:-26,27,29}
F32_DENSE_HEIGHTS=${F32_DENSE_HEIGHTS:-25,26,29}
CRITERION_WARMUP=${CRITERION_WARMUP:-3}
CRITERION_MEASUREMENT=${CRITERION_MEASUREMENT:-8}
CRITERION_SAMPLE_SIZE=${CRITERION_SAMPLE_SIZE:-30}
STRATEGIES=${STRATEGIES:-eytzinger,donnelly_unrolled,donnelly_cyclic_simd_descent,donnelly_cyclic_simd_full}
IRQ_RETRIES=${IRQ_RETRIES:-2}
RUN_PERF=${RUN_PERF:-1}
PERF_F64_HEIGHT=${PERF_F64_HEIGHT:-26}
PERF_F32_HEIGHT=${PERF_F32_HEIGHT:-25}
PERF_POOL_SIZE=${PERF_POOL_SIZE:-4096}
PERF_TOTAL_QUERIES=${PERF_TOTAL_QUERIES:-4000000}
PERF_WARMUP_REPEATS=${PERF_WARMUP_REPEATS:-2}
PERF_EVENT_SETS=${PERF_EVENT_SETS:-core,cache,tlb}
OUTPUT_BASE=${OUTPUT_BASE:-$REPO_DIR/focused-results}
CHART_BASE=${CHART_BASE:-$REPO_DIR/focused-charts}
SUITE_LABEL=${SUITE_LABEL:-donnelly-cyclic-full-height-matrix-$(date -u +%Y%m%dT%H%M%SZ)}

readonly RUN_DIR="$OUTPUT_BASE/$SUITE_LABEL"
readonly CHART_DIR="$CHART_BASE/$SUITE_LABEL"
readonly RESULT_FILE="$RUN_DIR/bench_result-v6-donnelly-cyclic-full-$SUITE_LABEL.json"
readonly HTML_FILE="$CHART_DIR/$SUITE_LABEL.html"
readonly LOG_FILE="$RUN_DIR/run.log"

for command_name in bench-profile-run bench-profile-status cargo git jq just mktemp perf python3 rustc; do
    command -v "$command_name" >/dev/null || {
        echo "Required command is unavailable: $command_name" >&2
        exit 2
    }
done
[[ "$SUITE_LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._+:-]*$ ]] || {
    echo "SUITE_LABEL contains unsupported characters: $SUITE_LABEL" >&2
    exit 2
}
[[ ! -e "$RUN_DIR" && ! -e "$CHART_DIR" ]] || {
    echo "Refusing to overwrite existing suite $SUITE_LABEL" >&2
    exit 2
}
case "$RUN_PERF" in
    0|1) ;;
    *) echo "RUN_PERF must be 0 or 1" >&2; exit 2 ;;
esac

IFS=',' read -r -a heights <<<"$HEIGHTS"
IFS=',' read -r -a base_pools <<<"$BASE_POOL_SIZES"
IFS=',' read -r -a dense_pools <<<"$DENSE_POOL_SIZES"
IFS=',' read -r -a f64_dense_heights <<<"$F64_DENSE_HEIGHTS"
IFS=',' read -r -a f32_dense_heights <<<"$F32_DENSE_HEIGHTS"
IFS=',' read -r -a strategies <<<"$STRATEGIES"
IFS=',' read -r -a perf_event_sets <<<"$PERF_EVENT_SETS"

readonly -a supported_strategies=(
    eytzinger donnelly_unrolled donnelly_cyclic_simd_descent
    donnelly_cyclic_simd_full
)

contains() {
    local needle=$1
    shift
    local value
    for value in "$@"; do
        [[ "$value" == "$needle" ]] && return 0
    done
    return 1
}

validate_unique_positive_list() {
    local label=$1
    shift
    local -a values=("$@")
    local i j value
    (( ${#values[@]} > 0 )) || {
        echo "$label must not be empty" >&2
        exit 2
    }
    for value in "${values[@]}"; do
        [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
            echo "$label contains an invalid positive integer: $value" >&2
            exit 2
        }
    done
    for ((i = 0; i < ${#values[@]}; i++)); do
        for ((j = i + 1; j < ${#values[@]}; j++)); do
            [[ "${values[i]}" != "${values[j]}" ]] || {
                echo "$label contains a duplicate: ${values[i]}" >&2
                exit 2
            }
        done
    done
}

validate_unique_positive_list HEIGHTS "${heights[@]}"
validate_unique_positive_list BASE_POOL_SIZES "${base_pools[@]}"
validate_unique_positive_list DENSE_POOL_SIZES "${dense_pools[@]}"
validate_unique_positive_list F64_DENSE_HEIGHTS "${f64_dense_heights[@]}"
validate_unique_positive_list F32_DENSE_HEIGHTS "${f32_dense_heights[@]}"

for height in "${heights[@]}"; do
    (( height >= 5 && height < 30 )) || {
        echo "HEIGHTS entries must be between 5 and 29: $height" >&2
        exit 2
    }
done
for height in "${f64_dense_heights[@]}" "${f32_dense_heights[@]}"; do
    contains "$height" "${heights[@]}" || {
        echo "Dense height $height is absent from HEIGHTS" >&2
        exit 2
    }
done
for pool in "${base_pools[@]}"; do
    contains "$pool" "${dense_pools[@]}" || {
        echo "DENSE_POOL_SIZES must include base pool $pool" >&2
        exit 2
    }
done

(( ${#strategies[@]} > 0 )) || {
    echo "STRATEGIES must not be empty" >&2
    exit 2
}
for strategy in "${strategies[@]}"; do
    contains "$strategy" "${supported_strategies[@]}" || {
        echo "Unsupported strategy: $strategy" >&2
        exit 2
    }
done
for ((i = 0; i < ${#strategies[@]}; i++)); do
    for ((j = i + 1; j < ${#strategies[@]}; j++)); do
        [[ "${strategies[i]}" != "${strategies[j]}" ]] || {
            echo "STRATEGIES contains a duplicate: ${strategies[i]}" >&2
            exit 2
        }
    done
done
contains eytzinger "${strategies[@]}" || {
    echo "STRATEGIES must retain the Eytzinger control" >&2
    exit 2
}

for value_name in CRITERION_WARMUP CRITERION_MEASUREMENT; do
    value=${!value_name}
    [[ "$value" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || {
        echo "$value_name must be a non-negative number: $value" >&2
        exit 2
    }
done
[[ "$CRITERION_SAMPLE_SIZE" =~ ^[1-9][0-9]*$ ]] \
    && (( CRITERION_SAMPLE_SIZE >= 10 )) || {
    echo "CRITERION_SAMPLE_SIZE must be an integer of at least 10" >&2
    exit 2
}
[[ "$IRQ_RETRIES" =~ ^[0-9]+$ ]] || {
    echo "IRQ_RETRIES must be a non-negative integer" >&2
    exit 2
}
for value_name in PERF_F64_HEIGHT PERF_F32_HEIGHT PERF_POOL_SIZE \
    PERF_TOTAL_QUERIES PERF_WARMUP_REPEATS; do
    value=${!value_name}
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
        echo "$value_name must be a positive integer: $value" >&2
        exit 2
    }
done
(( PERF_F64_HEIGHT >= 5 && PERF_F64_HEIGHT < 30 )) || {
    echo "PERF_F64_HEIGHT must be between 5 and 29" >&2
    exit 2
}
(( PERF_F32_HEIGHT >= 5 && PERF_F32_HEIGHT < 30 )) || {
    echo "PERF_F32_HEIGHT must be between 5 and 29" >&2
    exit 2
}
(( (PERF_F64_HEIGHT - 5) % 3 == 0 )) || {
    echo "PERF_F64_HEIGHT must be an exact Block3 height" >&2
    exit 2
}
(( (PERF_F32_HEIGHT - 5) % 4 == 0 )) || {
    echo "PERF_F32_HEIGHT must be an exact Block4 height" >&2
    exit 2
}
for event_set in "${perf_event_sets[@]}"; do
    case "$event_set" in
        core|cache|tlb) ;;
        *) echo "Unsupported PERF_EVENT_SETS entry: $event_set" >&2; exit 2 ;;
    esac
done
for ((i = 0; i < ${#perf_event_sets[@]}; i++)); do
    for ((j = i + 1; j < ${#perf_event_sets[@]}; j++)); do
        [[ "${perf_event_sets[i]}" != "${perf_event_sets[j]}" ]] || {
            echo "PERF_EVENT_SETS contains a duplicate: ${perf_event_sets[i]}" >&2
            exit 2
        }
    done
done

native_cfg=$(rustc --print cfg -C target-cpu=native)
[[ "$native_cfg" == *'target_feature="avx512f"'* ]] || {
    echo "This experiment requires native AVX-512F support" >&2
    exit 2
}

mkdir -p -- "$RUN_DIR" "$CHART_DIR"

planned_dense_cells=$((${#f64_dense_heights[@]} + ${#f32_dense_heights[@]}))
planned_cells=$((2 * ${#heights[@]}))
planned_base_cells=$((planned_cells - planned_dense_cells))
planned_results=$((
    planned_base_cells * ${#base_pools[@]} * ${#strategies[@]} * 2
    + planned_dense_cells * ${#dense_pools[@]} * ${#strategies[@]} * 2
))
{
    echo "Planned timing cells: $planned_cells ($planned_dense_cells dense query sweeps)"
    echo "Planned Criterion functions: $planned_results"
    echo "Per-function budget: ${CRITERION_WARMUP}s warmup + ${CRITERION_MEASUREMENT}s measurement"
    if (( RUN_PERF == 1 )); then
        echo "Planned perf passes: $((2 * ${#strategies[@]} * ${#perf_event_sets[@]}))"
    fi
} | tee -a "$LOG_FILE"

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
if (( RUN_PERF == 1 )); then
    for event_set in "${perf_event_sets[@]}"; do
        "$SCRIPT_DIR/query_pool_perf_events.sh" --check "$event_set"
    done
fi

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

run_cell() {
    local axis=$1 dimensions=$2 height=$3 pool_csv=$4 role=$5
    local warmup=${6:-$CRITERION_WARMUP}
    local measurement=${7:-$CRITERION_MEASUREMENT}
    local sample_size=${8:-$CRITERION_SAMPLE_SIZE}
    local output_dir=${9:-$RUN_DIR}
    local attempt=0 status result_key result_path tmp_dir

    result_key="$SUITE_LABEL-$role-2p$height"
    result_path="$output_dir/bench_result-v6-donnelly-cyclic-full-$result_key.json"
    while true; do
        tmp_dir="$(mktemp -d /dev/shm/kiddo-cyclic-full.XXXXXX)"
        cp -- "$benchmark_exe" "$tmp_dir/benchmark"
        chmod 0755 "$tmp_dir/benchmark"

        set +e
        bench-profile-run env \
            CRITERION_HOME="$tmp_dir/criterion" \
            KIDDO_CYCLIC_AXIS="$axis" \
            KIDDO_CYCLIC_DIMENSIONS="$dimensions" \
            KIDDO_CYCLIC_POINT_LOG2="$height" \
            KIDDO_CYCLIC_POOL_SIZES="$pool_csv" \
            KIDDO_CYCLIC_STRATEGIES="$STRATEGIES" \
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
        echo "Retrying IRQ-invalidated cell $role 2^$height ($attempt/$IRQ_RETRIES)" |
            tee -a "$LOG_FILE" >&2
    done
}

# Exercise the exact executable, isolation wrapper, exporter, and chart parser
# before starting the long matrix.
readonly PREFLIGHT_DIR="$(mktemp -d /dev/shm/kiddo-cyclic-full-preflight.XXXXXX)"
cleanup_preflight() {
    rm -rf -- "$PREFLIGHT_DIR"
}
trap cleanup_preflight EXIT INT TERM
preflight_result=$(run_cell f64 3 10 16 preflight 0.01 0.02 10 "$PREFLIGHT_DIR")
preflight_expected=$((${#strategies[@]} * 2))
preflight_actual=$(jq '.results | length' "$preflight_result")
if (( preflight_actual != preflight_expected )); then
    echo "Preflight produced $preflight_actual entries; expected $preflight_expected" >&2
    exit 1
fi
python3 "$SCRIPT_DIR/chart_donnelly_variant_results.py" all \
    --result "$preflight_result" --result-label "$SUITE_LABEL-preflight" \
    --output-dir "$PREFLIGHT_DIR/charts" --html-name preflight.html
if (( RUN_PERF == 1 )); then
    run_irq_retry_logged just perf-v6-donnelly-vs-eytzinger-query-pool \
        10 16 128 1 f64 eytzinger core "$SUITE_LABEL-preflight-perf" "$PREFLIGHT_DIR"
    preflight_perf="$PREFLIGHT_DIR/$SUITE_LABEL-preflight-perf-f64-eytzinger-q16-core.perf.csv"
    "$SCRIPT_DIR/query_pool_perf_events.sh" --validate "$preflight_perf"
fi
cleanup_preflight
trap - EXIT INT TERM

inputs=()
expected_total=0
for axis in f64 f32; do
    if [[ "$axis" == f64 ]]; then
        dimensions=3
        dense_height_csv=$F64_DENSE_HEIGHTS
        dense_height_list=("${f64_dense_heights[@]}")
    else
        dimensions=4
        dense_height_csv=$F32_DENSE_HEIGHTS
        dense_height_list=("${f32_dense_heights[@]}")
    fi
    for height in "${heights[@]}"; do
        pool_csv=$BASE_POOL_SIZES
        pool_count=${#base_pools[@]}
        if contains "$height" "${dense_height_list[@]}"; then
            pool_csv=$DENSE_POOL_SIZES
            pool_count=${#dense_pools[@]}
        fi
        result_path=$(run_cell "$axis" "$dimensions" "$height" "$pool_csv" "$axis-k$dimensions")
        inputs+=("$result_path")
        expected_total=$((expected_total + pool_count * ${#strategies[@]} * 2))
    done
    echo "$axis dense heights: $dense_height_csv" | tee -a "$LOG_FILE"
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

actual_total=$(jq '.results | length' "$RESULT_FILE")
if (( actual_total != expected_total )); then
    echo "Merged result has $actual_total entries; expected $expected_total" >&2
    exit 1
fi

python3 "$SCRIPT_DIR/chart_donnelly_variant_results.py" all \
    --result "$RESULT_FILE" --result-label "$SUITE_LABEL" \
    --output-dir "$CHART_DIR" --html-name "$SUITE_LABEL.html"

if (( RUN_PERF == 1 )); then
    for axis in f64 f32; do
        if [[ "$axis" == f64 ]]; then
            perf_height=$PERF_F64_HEIGHT
        else
            perf_height=$PERF_F32_HEIGHT
        fi
        for strategy in "${strategies[@]}"; do
            for event_set in "${perf_event_sets[@]}"; do
                result_key="$SUITE_LABEL-exact-height"
                run_irq_retry_logged just perf-v6-donnelly-vs-eytzinger-query-pool \
                    "$perf_height" "$PERF_POOL_SIZE" "$PERF_TOTAL_QUERIES" \
                    "$PERF_WARMUP_REPEATS" "$axis" "$strategy" "$event_set" \
                    "$result_key" "$RUN_DIR"
                perf_csv="$RUN_DIR/$result_key-$axis-$strategy-q$PERF_POOL_SIZE-$event_set.perf.csv"
                "$SCRIPT_DIR/query_pool_perf_events.sh" --validate "$perf_csv"
            done
        done
    done
fi

capture_profile_status "$RUN_DIR/bench-profile-status-after.txt" "after the suite"
{
    echo "suite_label=$SUITE_LABEL"
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "heights=$HEIGHTS"
    echo "base_pool_sizes=$BASE_POOL_SIZES"
    echo "dense_pool_sizes=$DENSE_POOL_SIZES"
    echo "f64_dense_heights=$F64_DENSE_HEIGHTS"
    echo "f32_dense_heights=$F32_DENSE_HEIGHTS"
    echo "strategies=$STRATEGIES"
    echo "cells=f64-k3:2p20-2p29,f32-k4:2p20-2p29"
    echo "exact_heights=f64:2p20,2p23,2p26,2p29;f32:2p21,2p25,2p29"
    echo "run_perf=$RUN_PERF"
    echo "perf_heights=f64:2p$PERF_F64_HEIGHT,f32:2p$PERF_F32_HEIGHT"
    echo "perf_pool_size=$PERF_POOL_SIZE"
    echo "perf_total_queries=$PERF_TOTAL_QUERIES"
    echo "perf_event_sets=$PERF_EVENT_SETS"
    echo "result_count=$actual_total"
    echo "result_file=$RESULT_FILE"
    echo "chart_file=$HTML_FILE"
    echo "git_head=$(git -C "$REPO_DIR" rev-parse HEAD)"
    echo "rustc=$(rustc --version)"
} >"$RUN_DIR/manifest.txt"

echo
echo "Cyclic SIMD-full height/query matrix complete"
echo "Results: $RESULT_FILE"
echo "Charts:  $HTML_FILE"

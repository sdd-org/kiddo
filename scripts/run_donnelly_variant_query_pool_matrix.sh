#!/usr/bin/env bash
set -euo pipefail

# Exact-nearest-one timing matrix and representative fixed-work perf passes for
# three deliberately distinct experiments:
#
# 1. f64/3D/2^23: every strategy at a balanced UBD supercycle boundary;
# 2. f32/4D/2^25: cyclic layouts only, at a useful cache-pressure size; and
# 3. f32/4D/2^21: block-dimension strategies at their balanced supercycle.
#
# Run from the dedicated benchmark boot profile. Each tree-height interval is
# pinned and IRQ-audited by bench-profile-run through the underlying just task.
# Each cell has its own overridable strategy list. Every list must retain the
# Eytzinger baseline because the generated charts plot advantage over it.

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

F64_POINT_LOG2=${F64_POINT_LOG2:-23}
F32_CYCLIC_POINT_LOG2=${F32_CYCLIC_POINT_LOG2:-25}
F32_UBD_POINT_LOG2=${F32_UBD_POINT_LOG2:-21}
POOL_SIZES=${POOL_SIZES:-256,512,1000,2048,4096,8192,16384}
CRITERION_WARMUP=${CRITERION_WARMUP:-3}
CRITERION_MEASUREMENT=${CRITERION_MEASUREMENT:-8}
CRITERION_SAMPLE_SIZE=${CRITERION_SAMPLE_SIZE:-30}
PERF_POOL_SIZE=${PERF_POOL_SIZE:-4096}
PERF_TOTAL_QUERIES=${PERF_TOTAL_QUERIES:-4000000}
PERF_WARMUP_REPEATS=${PERF_WARMUP_REPEATS:-2}
PERF_EVENT_SETS=${PERF_EVENT_SETS:-core,cache,tlb}
F64_STRATEGIES=${F64_STRATEGIES:-eytzinger,donnelly,donnelly_unrolled,donnelly_unrolled_block_dim,donnelly_simd_descent,donnelly_cyclic_simd_descent,donnelly_simd_initial_descent,donnelly_simd_full}
F32_CYCLIC_STRATEGIES=${F32_CYCLIC_STRATEGIES:-eytzinger,donnelly,donnelly_unrolled,donnelly_cyclic_simd_descent}
F32_UBD_STRATEGIES=${F32_UBD_STRATEGIES:-eytzinger,donnelly_unrolled,donnelly_unrolled_block_dim,donnelly_simd_descent,donnelly_simd_full}
IRQ_RETRIES=${IRQ_RETRIES:-2}
OUTPUT_BASE=${OUTPUT_BASE:-$REPO_DIR/focused-results}
CHART_BASE=${CHART_BASE:-$REPO_DIR/focused-charts}
SUITE_LABEL=${SUITE_LABEL:-donnelly-variant-matrix-$(date -u +%Y%m%dT%H%M%SZ)}

if [[ ! "$SUITE_LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._+:-]*$ ]]; then
    echo "SUITE_LABEL contains unsupported characters: $SUITE_LABEL" >&2
    exit 2
fi

readonly RUN_DIR="$OUTPUT_BASE/$SUITE_LABEL"
readonly CHART_DIR="$CHART_BASE/$SUITE_LABEL"
readonly LOG_FILE="$RUN_DIR/run.log"
readonly RESULT_FILE="$RUN_DIR/bench_result-v6-donnelly-vs-eytzinger-query-pool-$SUITE_LABEL.json"
readonly HTML_FILE="$CHART_DIR/$SUITE_LABEL.html"

for command_name in bench-profile-run bench-profile-status cargo git jq just mktemp perf python3 rustc; do
    if ! command -v "$command_name" >/dev/null; then
        echo "Required command is unavailable: $command_name" >&2
        exit 2
    fi
done

if [[ -e "$RUN_DIR" || -e "$CHART_DIR" ]]; then
    echo "Refusing to overwrite an existing suite: $SUITE_LABEL" >&2
    exit 2
fi

IFS=',' read -r -a pools <<<"$POOL_SIZES"
IFS=',' read -r -a perf_event_sets <<<"$PERF_EVENT_SETS"
IFS=',' read -r -a f64_strategies <<<"$F64_STRATEGIES"
IFS=',' read -r -a f32_cyclic_strategies <<<"$F32_CYCLIC_STRATEGIES"
IFS=',' read -r -a f32_ubd_strategies <<<"$F32_UBD_STRATEGIES"
readonly -a supported_strategies=(
    eytzinger donnelly donnelly_unrolled donnelly_unrolled_block_dim
    donnelly_simd_descent donnelly_cyclic_simd_descent
    donnelly_simd_initial_descent donnelly_simd_full
)

validate_strategy_list() {
    local label=$1
    shift
    local -a strategies=("$@")
    local i j strategy
    if (( ${#strategies[@]} == 0 )); then
        echo "$label must contain at least one strategy" >&2
        exit 2
    fi
    for strategy in "${strategies[@]}"; do
        if [[ ! " ${supported_strategies[*]} " =~ " $strategy " ]]; then
            echo "$label contains unsupported strategy: $strategy" >&2
            exit 2
        fi
    done
    for ((i = 0; i < ${#strategies[@]}; i++)); do
        for ((j = i + 1; j < ${#strategies[@]}; j++)); do
            if [[ "${strategies[i]}" == "${strategies[j]}" ]]; then
                echo "$label contains a duplicate: ${strategies[i]}" >&2
                exit 2
            fi
        done
    done
    if [[ ! " ${strategies[*]} " =~ " eytzinger " ]]; then
        echo "$label must include eytzinger because charts use it as the baseline" >&2
        exit 2
    fi
}

validate_strategy_list F64_STRATEGIES "${f64_strategies[@]}"
validate_strategy_list F32_CYCLIC_STRATEGIES "${f32_cyclic_strategies[@]}"
validate_strategy_list F32_UBD_STRATEGIES "${f32_ubd_strategies[@]}"

native_rust_cfg=$(rustc --print cfg -C target-cpu=native)
all_strategy_lists=" $F64_STRATEGIES,$F32_CYCLIC_STRATEGIES,$F32_UBD_STRATEGIES "
if [[ "$all_strategy_lists" == *donnelly_cyclic_simd_descent* ]] \
    && [[ "$native_rust_cfg" != *'target_feature="avx512f"'* ]]; then
    echo "donnelly_cyclic_simd_descent requires native AVX-512F support" >&2
    exit 2
fi

for value_name in F64_POINT_LOG2 F32_CYCLIC_POINT_LOG2 F32_UBD_POINT_LOG2; do
    value=${!value_name}
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$value_name must be a positive integer; got $value" >&2
        exit 2
    fi
done
if (( F64_POINT_LOG2 < 5 || (F64_POINT_LOG2 - 5) % 9 != 0 )); then
    echo "F64_POINT_LOG2=$F64_POINT_LOG2 is not a Donnelly<3>, K=3 supercycle boundary" >&2
    exit 2
fi
if (( F32_CYCLIC_POINT_LOG2 < 5 || (F32_CYCLIC_POINT_LOG2 - 5) % 4 != 0 )); then
    echo "F32_CYCLIC_POINT_LOG2=$F32_CYCLIC_POINT_LOG2 is not a complete Donnelly<4>, K=4 block boundary" >&2
    exit 2
fi
if (( F32_UBD_POINT_LOG2 < 5 || (F32_UBD_POINT_LOG2 - 5) % 16 != 0 )); then
    echo "F32_UBD_POINT_LOG2=$F32_UBD_POINT_LOG2 is not a Donnelly<4>, K=4 UBD supercycle boundary" >&2
    exit 2
fi
for pool in "${pools[@]}"; do
    if [[ ! "$pool" =~ ^[1-9][0-9]*$ ]]; then
        echo "POOL_SIZES must contain comma-separated positive integers; got $pool" >&2
        exit 2
    fi
done
for value_name in CRITERION_WARMUP CRITERION_MEASUREMENT; do
    value=${!value_name}
    if [[ ! "$value" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "$value_name must be numeric; got $value" >&2
        exit 2
    fi
done
if [[ ! "$CRITERION_SAMPLE_SIZE" =~ ^[1-9][0-9]*$ ]] \
    || (( CRITERION_SAMPLE_SIZE < 10 )); then
    echo "CRITERION_SAMPLE_SIZE must be an integer of at least 10" >&2
    exit 2
fi
if [[ ! "$IRQ_RETRIES" =~ ^[0-9]+$ ]]; then
    echo "IRQ_RETRIES must be a non-negative integer" >&2
    exit 2
fi
for value_name in PERF_POOL_SIZE PERF_TOTAL_QUERIES PERF_WARMUP_REPEATS
do
    value=${!value_name}
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$value_name must be a positive integer; got $value" >&2
        exit 2
    fi
done

# B=32 removes five levels. The timing and perf passes deliberately use the
# same point height for each experiment cell.
for event_set in "${perf_event_sets[@]}"; do
    case "$event_set" in
        core|cache|tlb) ;;
        *) echo "PERF_EVENT_SETS contains unsupported set: $event_set" >&2; exit 2 ;;
    esac
done
if (( ${#perf_event_sets[@]} == 0 )); then
    echo "PERF_EVENT_SETS must contain at least one event set" >&2
    exit 2
fi
for ((i = 0; i < ${#perf_event_sets[@]}; i++)); do
    for ((j = i + 1; j < ${#perf_event_sets[@]}; j++)); do
        if [[ "${perf_event_sets[i]}" == "${perf_event_sets[j]}" ]]; then
            echo "PERF_EVENT_SETS contains a duplicate: ${perf_event_sets[i]}" >&2
            exit 2
        fi
    done
done

run_irq_retry_logged() {
    local attempt=1 status max_attempts=$((IRQ_RETRIES + 1))
    while true; do
        set +e
        "$@" 2>&1 | tee -a "$LOG_FILE"
        status=${PIPESTATUS[0]}
        set -e
        if (( status != 125 || attempt >= max_attempts )); then
            return "$status"
        fi
        attempt=$((attempt + 1))
        echo "Retrying IRQ-invalidated benchmark interval (attempt $attempt/$max_attempts)" \
            | tee -a "$LOG_FILE"
    done
}

capture_profile_status() {
    local destination=$1 phase=$2 profile_status
    set +e
    bench-profile-status >"$destination" 2>&1
    profile_status=$?
    set -e
    if (( profile_status != 0 )); then
        cat "$destination" >&2
        echo >&2
        echo "Benchmark profile validation failed $phase; refusing to continue." >&2
        echo "Full status was saved to $destination" >&2
        return "$profile_status"
    fi
}

validate_criterion_cell() {
    local result_file=$1 strategies_csv=$2 expected_pool_count=$3 label=$4
    local -a strategies
    local actual expected function strategy
    IFS=',' read -r -a strategies <<<"$strategies_csv"

    if [[ ! -f "$result_file" ]]; then
        echo "$label result is missing: $result_file" >&2
        return 1
    fi

    expected=$((expected_pool_count * (2 * ${#strategies[@]} + 1)))
    actual=$(jq '.results | length' "$result_file")
    if (( actual != expected )); then
        echo "$label produced $actual Criterion results; expected $expected" >&2
        return 1
    fi

    actual=$(jq \
        '[.results[] | select(.metadata.function_id == "generated_control")] | length' \
        "$result_file")
    if (( actual != expected_pool_count )); then
        echo "$label generated_control has $actual results; expected $expected_pool_count" >&2
        return 1
    fi
    for strategy in "${strategies[@]}"; do
        for function in "stored_$strategy" "generated_$strategy"; do
            actual=$(jq --arg function "$function" \
                '[.results[] | select(.metadata.function_id == $function)] | length' \
                "$result_file")
            if (( actual != expected_pool_count )); then
                echo "$label $function has $actual results; expected $expected_pool_count" >&2
                return 1
            fi
        done
    done
}

# These cells are intentionally asymmetric. In particular, the f32 cyclic
# screen does not carry UBD strategies into a height where their axis schedule
# is unbalanced, and the f32 UBD control does not pretend its small tree is a
# headline cache-pressure result.
readonly -a cell_axes=(f64 f32 f32)
readonly -a cell_heights=(
    "$F64_POINT_LOG2" "$F32_CYCLIC_POINT_LOG2" "$F32_UBD_POINT_LOG2"
)
readonly -a cell_roles=(f64-balanced f32-cyclic f32-ubd-control)
readonly -a cell_strategy_csvs=(
    "$F64_STRATEGIES" "$F32_CYCLIC_STRATEGIES" "$F32_UBD_STRATEGIES"
)

mkdir -p -- "$RUN_DIR" "$CHART_DIR"
capture_profile_status "$RUN_DIR/bench-profile-status-before.txt" "before the suite"
for event_set in "${perf_event_sets[@]}"; do
    "$SCRIPT_DIR/query_pool_perf_events.sh" --check "$event_set"
done
RUSTC_WRAPPER= RUSTFLAGS='-C target-cpu=native' \
    cargo bench --bench profile_v6_query_pool_perf \
    --features simd,test_utils,logging_off --no-run
RUSTC_WRAPPER= RUSTFLAGS='-C target-cpu=native' \
    cargo bench --bench profile_v6_query_pool \
    --features simd,test_utils,logging_off --no-run

# Exercise the complete orchestration path before committing to the long run:
# one tiny Criterion cell for each distinct strategy set, a heterogeneous merge,
# charting, and one real perf attach/SIGSTOP/counter-validation interval.
readonly PREFLIGHT_DIR="$(mktemp -d /dev/shm/kiddo-variant-suite-preflight.XXXXXX)"
cleanup_preflight() {
    rm -rf -- "$PREFLIGHT_DIR"
}
trap cleanup_preflight EXIT INT TERM
preflight_inputs=()
for preflight_index in "${!cell_axes[@]}"; do
    preflight_axis=${cell_axes[preflight_index]}
    preflight_role=${cell_roles[preflight_index]}
    preflight_strategies=${cell_strategy_csvs[preflight_index]}
    # Give the second f32 cell a distinct point count so its benchmark IDs and
    # chart cannot collide with the first during the heterogeneous merge.
    if [[ "$preflight_role" == f32-ubd-control ]]; then
        preflight_height=11
    else
        preflight_height=10
    fi
    preflight_key="$SUITE_LABEL-preflight-$preflight_role"
    run_irq_retry_logged just bench-v6-donnelly-variants-query-pool-clean \
        "$preflight_height" 16 0.01 0.02 10 "$preflight_key" \
        "$PREFLIGHT_DIR" "$preflight_axis" "$preflight_strategies"
    preflight_json="$PREFLIGHT_DIR/bench_result-v6-donnelly-vs-eytzinger-query-pool-$preflight_key.json"
    validate_criterion_cell "$preflight_json" "$preflight_strategies" 1 \
        "Criterion preflight $preflight_role"
    preflight_inputs+=("$preflight_json")
done
readonly PREFLIGHT_JSON="$PREFLIGHT_DIR/preflight-merged.json"
jq -s --arg criterion_root "merged:$SUITE_LABEL:preflight" '
    {
        schema_version: 1,
        criterion_root: $criterion_root,
        collected_at_unix_ms: ([.[].collected_at_unix_ms] | max),
        filters: ([.[].filters[]] | unique),
        results: ([.[].results[]] | sort_by(.benchmark))
    }
' "${preflight_inputs[@]}" >"$PREFLIGHT_JSON"
python3 "$SCRIPT_DIR/chart_donnelly_variant_results.py" all \
    --result "$PREFLIGHT_JSON" --result-label "$SUITE_LABEL-preflight" \
    --output-dir "$PREFLIGHT_DIR/charts" --html-name preflight.html
run_irq_retry_logged just perf-v6-donnelly-vs-eytzinger-query-pool \
    10 16 128 1 f64 eytzinger core "$SUITE_LABEL-preflight-perf" "$PREFLIGHT_DIR"
readonly PREFLIGHT_PERF="$PREFLIGHT_DIR/$SUITE_LABEL-preflight-perf-f64-eytzinger-q16-core.perf.csv"
"$SCRIPT_DIR/query_pool_perf_events.sh" --validate "$PREFLIGHT_PERF"
cleanup_preflight
trap - EXIT INT TERM

{
    echo "suite_label=$SUITE_LABEL"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "experiment_cells=f64-balanced,f32-cyclic,f32-ubd-control"
    echo "f64_point_log2=$F64_POINT_LOG2"
    echo "f32_cyclic_point_log2=$F32_CYCLIC_POINT_LOG2"
    echo "f32_ubd_point_log2=$F32_UBD_POINT_LOG2"
    echo "pool_sizes=$POOL_SIZES"
    echo "axes=f64,f32"
    echo "dimensions=f64:3,f32:4"
    echo "query_modes=stored,generated"
    echo "f64_strategies=$F64_STRATEGIES"
    echo "f32_cyclic_strategies=$F32_CYCLIC_STRATEGIES"
    echo "f32_ubd_strategies=$F32_UBD_STRATEGIES"
    echo "criterion_warmup_seconds=$CRITERION_WARMUP"
    echo "criterion_measurement_seconds=$CRITERION_MEASUREMENT"
    echo "criterion_sample_size=$CRITERION_SAMPLE_SIZE"
    echo "perf_cells=timing_cells"
    echo "perf_pool_size=$PERF_POOL_SIZE"
    echo "perf_total_queries=$PERF_TOTAL_QUERIES"
    echo "perf_warmup_repeats=$PERF_WARMUP_REPEATS"
    echo "perf_event_sets=$PERF_EVENT_SETS"
    echo "perf_strategy_sets=cell-specific"
    echo "irq_retries=$IRQ_RETRIES"
    echo "git_head=$(git -C "$REPO_DIR" rev-parse HEAD)"
    echo "rustc=$(rustc --version)"
    echo "kernel=$(uname -srmo)"
} >"$RUN_DIR/manifest.txt"
git -C "$REPO_DIR" status --short >"$RUN_DIR/git-status.txt"

cd -- "$REPO_DIR"
inputs=()
expected_results=0
for cell_index in "${!cell_axes[@]}"; do
    axis=${cell_axes[cell_index]}
    height=${cell_heights[cell_index]}
    role=${cell_roles[cell_index]}
    strategies_csv=${cell_strategy_csvs[cell_index]}
    cell_key="$SUITE_LABEL-$role-2p$height"
    run_irq_retry_logged just bench-v6-donnelly-variants-query-pool-clean \
        "$height" "$POOL_SIZES" "$CRITERION_WARMUP" \
        "$CRITERION_MEASUREMENT" "$CRITERION_SAMPLE_SIZE" "$cell_key" \
        "$RUN_DIR" "$axis" "$strategies_csv"
    cell_json="$RUN_DIR/bench_result-v6-donnelly-vs-eytzinger-query-pool-$cell_key.json"
    validate_criterion_cell "$cell_json" "$strategies_csv" "${#pools[@]}" \
        "timing cell $role"
    inputs+=("$cell_json")
    IFS=',' read -r -a cell_strategies <<<"$strategies_csv"
    expected_results=$((
        expected_results + ${#pools[@]} * (2 * ${#cell_strategies[@]} + 1)
    ))
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
python3 "$SCRIPT_DIR/chart_donnelly_variant_results.py" all \
    --result "$RESULT_FILE" --result-label "$SUITE_LABEL" \
    --output-dir "$CHART_DIR" --html-name "$SUITE_LABEL.html"

if [[ ! -f "$RESULT_FILE" ]]; then
    echo "Merged Criterion result is missing: $RESULT_FILE" >&2
    exit 1
fi

actual_results=$(jq '.results | length' "$RESULT_FILE")
if (( actual_results != expected_results )); then
    echo "Merged result contains $actual_results entries; expected $expected_results" >&2
    exit 1
fi

echo | tee -a "$LOG_FILE"
echo "Running complete-block-boundary fixed-work perf characterization" | tee -a "$LOG_FILE"
expected_perf_files=0
for cell_index in "${!cell_axes[@]}"; do
    axis=${cell_axes[cell_index]}
    perf_point_log2=${cell_heights[cell_index]}
    role=${cell_roles[cell_index]}
    IFS=',' read -r -a perf_strategies <<<"${cell_strategy_csvs[cell_index]}"
    for event_set in "${perf_event_sets[@]}"; do
        for strategy in "${perf_strategies[@]}"; do
            run_irq_retry_logged just perf-v6-donnelly-vs-eytzinger-query-pool \
                "$perf_point_log2" "$PERF_POOL_SIZE" "$PERF_TOTAL_QUERIES" \
                "$PERF_WARMUP_REPEATS" "$axis" "$strategy" "$event_set" \
                "$SUITE_LABEL-perf-$role-2p$perf_point_log2" "$RUN_DIR"
        done
    done
    expected_perf_files=$((
        expected_perf_files + ${#perf_event_sets[@]} * ${#perf_strategies[@]}
    ))
done

perf_count=$(find "$RUN_DIR" -maxdepth 1 -type f -name '*.perf.csv' | wc -l)
perf_run_count=$(find "$RUN_DIR" -maxdepth 1 -type f -name '*.run.txt' | wc -l)
if (( perf_count != expected_perf_files || perf_run_count != expected_perf_files )); then
    echo "Perf pass produced $perf_count CSVs/$perf_run_count metadata files; expected $expected_perf_files" >&2
    exit 1
fi
while IFS= read -r perf_csv; do
    "$SCRIPT_DIR/query_pool_perf_events.sh" --validate "$perf_csv"
done < <(find "$RUN_DIR" -maxdepth 1 -type f -name '*.perf.csv' -print)

capture_profile_status "$RUN_DIR/bench-profile-status-after.txt" "after the suite"
{
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "result_count=$actual_results"
    echo "perf_interval_count=$perf_count"
    echo "result_file=$RESULT_FILE"
    echo "chart_file=$HTML_FILE"
} >>"$RUN_DIR/manifest.txt"

echo
echo "Donnelly variant matrix complete"
echo "Results: $RESULT_FILE"
echo "Charts:  $HTML_FILE"
echo "Perf:    $RUN_DIR/*.perf.csv"

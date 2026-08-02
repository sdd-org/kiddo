#!/usr/bin/env bash
set -euo pipefail

# No-holds-barred parallel throughput: kiddo against Pkd-tree only, across
# every kiddo stem strategy, with no CPU restrictions at all -- every thread,
# SMT on, boost on. This settles who holds the crown for maximum achievable
# parallel query rate on this machine, not implementation-detail comparisons,
# which is why it is deliberately narrower than the controlled multi-core
# matrix: two contenders, the largest safe tree size, the largest query
# count.
#
# Requires the UNLIMITED bench-profile boot entry
# (https://github.com/sdd/bench-profile): no isolation, so results are NOT
# comparable with the controlled multi-core matrix's numbers -- housekeeping
# and interrupts share these cores by design. Gated on
# `bench-profile-status expect-unrestricted`, which still runs the
# parallel-dispatch canary (gated on physical core count here, since an
# ALU-bound probe gains little from SMT).
#
# Kiddo stem strategies (KIDDO_STRATEGIES, comma-separated):
#   eytzinger                     baseline
#   donnelly_unrolled              Donnelly memory layout, scalar descent
#   donnelly_cyclic_simd_descent   block-at-once AVX-512 descent
#   donnelly_cyclic_simd_full      + AVX-512 rectangle pruning/backtracking
#
# Both runtimes are pinned to the same worker count: without it, ParlayLib's
# hardware_concurrency() and rayon's available_parallelism() could disagree
# on how many threads "every core" means, and the comparison would be
# meaningless. On an unlimited boot both already see the same online set, so
# this matters less than in the multi-core profile, but it costs nothing to
# still pin explicitly for the record.
#
# Tree size defaults to the largest EXACT stem-height size this repository's
# scripts have used at the multi-core matrix's scale: see
# run_single_core_strategy_matrix.sh for the full multiple-of-block-height
# derivation (stem height = log2(points) - 5, leaf capacity 32). f64 (BH=3)
# -> 2^26; f32 (BH=4) -> 2^21, capped there regardless by Pkd-tree's f32
# build-assertion limit.

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=lib_bench_profile_matrix.sh
source "$SCRIPT_DIR/lib_bench_profile_matrix.sh"

readonly PHASE=unlimited
readonly REQUIRED_PROFILE=unlimited
readonly EXPECT_VERB=expect-unrestricted
readonly STRICT_ISOLATION=0

LIBRARIES=${LIBRARIES:-kiddo,pkdtree}
STRATEGIES=${STRATEGIES:-$ALL_STRATEGIES}
BENCH_CPUS=${BENCH_CPUS:-}
F64_HEIGHTS=${F64_HEIGHTS:-26}
F32_HEIGHTS=${F32_HEIGHTS:-21}
QUERY_COUNTS=${QUERY_COUNTS:-100000}
SCALARS=${SCALARS:-f64,f32}
CRITERION_WARMUP=${CRITERION_WARMUP:-3}
CRITERION_MEASUREMENT=${CRITERION_MEASUREMENT:-8}
CRITERION_SAMPLE_SIZE=${CRITERION_SAMPLE_SIZE:-30}
IRQ_RETRIES=${IRQ_RETRIES:-2}
FEATURES=${FEATURES:-cpp_competitors,test_utils,logging_off,simd}
OUTPUT_BASE=${OUTPUT_BASE:-$REPO_DIR/cpp-competitor-results}
CHART_BASE=${CHART_BASE:-$REPO_DIR/cpp-competitor-charts}
SUITE_LABEL=${SUITE_LABEL:-unlimited-strategy-matrix-$(date -u +%Y%m%dT%H%M%SZ)}

readonly RUN_DIR="$OUTPUT_BASE/$SUITE_LABEL"
readonly CHART_DIR="$CHART_BASE/$SUITE_LABEL"
readonly LOG_FILE="$RUN_DIR/run.log"

require_commands cargo jq mktemp python3 rustc taskset
[[ "$SUITE_LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._+:-]*$ ]] || {
    echo "SUITE_LABEL contains unsupported characters: $SUITE_LABEL" >&2
    exit 2
}
[[ ! -e "$RUN_DIR" && ! -e "$CHART_DIR" ]] || {
    echo "Refusing to overwrite existing suite $SUITE_LABEL" >&2
    exit 2
}

booted_profile() {
    local token
    for token in $(</proc/cmdline); do
        case "$token" in
            systemd.setenv=BENCH_PROFILE=*)
                printf '%s\n' "${token#systemd.setenv=BENCH_PROFILE=}"
                return 0
                ;;
        esac
    done
    return 1
}
BOOT_PROFILE=$(booted_profile || true)
readonly BOOT_PROFILE
# Unconditional: an empty BOOT_PROFILE (a normal desktop boot has no
# BENCH_PROFILE marker at all) must refuse exactly like a wrong one. These
# are single-purpose scripts with no override, so there is no legitimate
# case where an unlabeled boot should be allowed through.
if [[ "$BOOT_PROFILE" != "$REQUIRED_PROFILE" ]]; then
    echo "This script needs the '$REQUIRED_PROFILE' boot profile, but this machine is" >&2
    echo "booted into '${BOOT_PROFILE:-a normal, non-benchmark boot}'." >&2
    echo "Reboot into the '... benchmark $REQUIRED_PROFILE' entry." >&2
    exit 2
fi

online_cpu_list() {
    local candidate cpu online
    for candidate in /sys/devices/system/cpu/cpu[0-9]*; do
        cpu=${candidate##*/cpu}
        [[ "$cpu" =~ ^[0-9]+$ ]] || continue
        online=1
        [[ -r "$candidate/online" ]] && online=$(<"$candidate/online")
        (( online == 1 )) && printf '%s\n' "$cpu"
    done | sort -n
}
if [[ -z "$BENCH_CPUS" ]]; then
    BENCH_CPUS=$(online_cpu_list | paste -sd, -)
fi

IFS=',' read -r -a f64_heights <<<"$F64_HEIGHTS"
IFS=',' read -r -a f32_heights <<<"$F32_HEIGHTS"
IFS=',' read -r -a query_counts <<<"$QUERY_COUNTS"
IFS=',' read -r -a scalars <<<"$SCALARS"
IFS=',' read -r -a strategies <<<"$STRATEGIES"

validate_unique_positive_list QUERY_COUNTS "${query_counts[@]}"
(( ${#scalars[@]} > 0 )) || { echo "SCALARS must not be empty" >&2; exit 2; }
for scalar in "${scalars[@]}"; do
    case "$scalar" in
        f32|f64) ;;
        *) echo "SCALARS entries must be f32 or f64: $scalar" >&2; exit 2 ;;
    esac
done
contains f64 "${scalars[@]}" && validate_unique_positive_list F64_HEIGHTS "${f64_heights[@]}"
contains f32 "${scalars[@]}" && validate_unique_positive_list F32_HEIGHTS "${f32_heights[@]}"

readonly -a supported_strategies=(eytzinger donnelly_unrolled donnelly_cyclic_simd_descent donnelly_cyclic_simd_full)
(( ${#strategies[@]} > 0 )) || { echo "STRATEGIES must not be empty" >&2; exit 2; }
for strategy in "${strategies[@]}"; do
    contains "$strategy" "${supported_strategies[@]}" || {
        echo "Unsupported strategy: $strategy (supported: ${supported_strategies[*]})" >&2
        exit 2
    }
done

if contains f32 "${scalars[@]}" && [[ "$LIBRARIES" == *pkdtree* ]]; then
    for height in "${f32_heights[@]}"; do
        (( height <= 21 )) || {
            echo "F32_HEIGHTS entry $height exceeds Pkd-tree's f32 build limit of 2^21" >&2
            exit 2
        }
    done
fi

mapfile -t bench_cpus < <(expand_cpu_list "$BENCH_CPUS") || {
    echo "BENCH_CPUS is not a valid CPU list: $BENCH_CPUS" >&2
    exit 2
}
(( ${#bench_cpus[@]} > 1 )) || {
    echo "This script measures parallel throughput; BENCH_CPUS must name more than one CPU" >&2
    exit 2
}
readonly WORKER_COUNT=${#bench_cpus[@]}
pin_command=(taskset --cpu-list "$BENCH_CPUS")
readonly pin_command

mkdir -p -- "$RUN_DIR" "$CHART_DIR"

build_benchmark
validate_environment "$RUN_DIR/environment-before.txt"

# --- preflight ---------------------------------------------------------------

readonly PREFLIGHT_DIR="$(mktemp -d /dev/shm/kiddo-unlimited-preflight.XXXXXX)"
cleanup_preflight() { rm -rf -- "$PREFLIGHT_DIR"; }
trap cleanup_preflight EXIT INT TERM

log "Preflight: f64 2^14, 256 queries"
preflight_result=$(run_cell profile_pkdtree_batch \
    "KIDDO_CPP_SUITES=pkdtree_batch KIDDO_CPP_LIBRARIES=$LIBRARIES KIDDO_CPP_SCALARS=f64 KIDDO_STRATEGIES=$STRATEGIES" \
    f64 14 256 preflight 0.5 1 10 "$PREFLIGHT_DIR")
(( $(jq '.results | length' "$preflight_result") > 0 )) || {
    echo "Preflight produced no results" >&2
    exit 1
}
python3 "$SCRIPT_DIR/chart_pkdtree_batch_results.py" all \
    --result "$preflight_result" --result-label "$SUITE_LABEL-preflight" \
    --output-dir "$PREFLIGHT_DIR/charts" --html-name preflight.html
cleanup_preflight
trap - EXIT INT TERM

# --- matrix --------------------------------------------------------------

planned=0
for scalar in "${scalars[@]}"; do
    if [[ "$scalar" == f64 ]]; then
        planned=$((planned + ${#f64_heights[@]} * ${#query_counts[@]}))
    else
        planned=$((planned + ${#f32_heights[@]} * ${#query_counts[@]}))
    fi
done
log "Phase: unlimited (boot profile: ${BOOT_PROFILE:-none}, gate: $EXPECT_VERB, cpus: $BENCH_CPUS, $WORKER_COUNT workers)"
log "Libraries: $LIBRARIES  strategies: $STRATEGIES  scalars: $SCALARS"
log "Planned cells: $planned"
log "NOTE: no CPU isolation in this profile -- these numbers are NOT comparable"
log "      with the controlled multi-core matrix's, only with each other."

batch_results=()
completed=0

log "=== pkdtree_batch (unrestricted): kiddo ($STRATEGIES) vs Pkd-tree parallel_for ==="
for scalar in "${scalars[@]}"; do
    if [[ "$scalar" == f64 ]]; then
        heights=("${f64_heights[@]}")
    else
        heights=("${f32_heights[@]}")
    fi
    for height in "${heights[@]}"; do
        for query_count in "${query_counts[@]}"; do
            completed=$((completed + 1))
            log "[$completed/$planned] unlimited $scalar 2^$height, $query_count queries"
            batch_results+=("$(run_cell profile_pkdtree_batch \
                "KIDDO_CPP_SUITES=pkdtree_batch KIDDO_CPP_LIBRARIES=$LIBRARIES KIDDO_CPP_SCALARS=$scalar KIDDO_STRATEGIES=$STRATEGIES" \
                "$scalar" "$height" "$query_count" matrix)")
        done
    done
done

validate_environment "$RUN_DIR/environment-after.txt"

# --- charts --------------------------------------------------------------

if (( ${#batch_results[@]} > 0 )); then
    chart_args=()
    for result in "${batch_results[@]}"; do
        chart_args+=(--result "$result")
    done
    python3 "$SCRIPT_DIR/chart_pkdtree_batch_results.py" all \
        "${chart_args[@]}" \
        --result-label "$SUITE_LABEL" \
        --output-dir "$CHART_DIR" \
        --html-name "$SUITE_LABEL-batch.html"
    log "Batch charts: $CHART_DIR/$SUITE_LABEL-batch.html"
fi

log "Suite complete: ${#batch_results[@]} result files in $RUN_DIR"

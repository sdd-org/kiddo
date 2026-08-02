#!/usr/bin/env bash
set -euo pipefail

# Single-threaded per-query comparison: nanoflann and Pkd-tree against every
# kiddo stem strategy, in one session so the numbers are genuinely comparable
# -- addresses the gap in the earlier competitor matrix, where kiddo's numbers
# came from a different session collected on a different day, and the
# nanoflann/Pkd-tree comparison never included kiddo at all.
#
# Requires the SINGLE-CORE bench-profile boot entry
# (https://github.com/sdd/bench-profile): one isolated core, load balancing
# off. Gated on `bench-profile-status expect-single-core`.
#
# Two Criterion suites, same session, same seeds, same per-scalar dimension
# (see below), so the point clouds and queries are byte-identical across both
# at a given size even though they are separate invocations:
#
#   kiddo_vs_pkdtree   kiddo (every stem strategy below) vs Pkd-tree,
#                      sequential. The comparison that matters most; charted
#                      automatically with chart_kiddo_vs_pkdtree_results.py.
#   cpp_competitors    nanoflann alone. nanoflann is consistently the fastest
#                      C++ competitor at these sizes, so it is the one worth
#                      the extra FFI-build cost; ALGLIB and skd-tree are left
#                      out by default (set LIBRARIES= to add them back).
#                      Charted separately -- see the end of this script's
#                      output for the command.
#
# Kiddo stem strategies (KIDDO_STRATEGIES, comma-separated):
#   eytzinger                     baseline
#   donnelly_unrolled              Donnelly memory layout, scalar descent
#   donnelly_cyclic_simd_descent   block-at-once AVX-512 descent
#   donnelly_cyclic_simd_full      + AVX-512 rectangle pruning/backtracking
#
# The cyclic strategies panic unless the tree's block height matches what
# their AVX-512 code is compiled for: BH=3 for f64, BH=4 for f32. The bench
# file's K_F32/K_F64 constants already pick each scalar's dimension to equal
# its native block height, so this just works -- but see the height note
# below for a caveat this script does still need to handle.
#
# Tree sizes are chosen so the stem height (log2(points) - log2(leaf
# capacity), leaf capacity is 32 = 2^5) is an EXACT multiple of the block
# height, for now: a stem that pads to a taller height than its content
# needs is a different, currently-unmeasured performance regime, not a
# methodology this script has validated. That gives log2(points) = 5 + m*BH:
#   f64 (BH=3): 8, 11, 14, ..., 20, 23, 26, 29
#   f32 (BH=4): 9, 13, 17, ..., 21, 25, 29
# Defaults below use f64 up to 26 rather than 29 -- sequential per-query
# timing at 2^29 points (~13GB resident for f64) is a slow way to spend a
# single core repeatedly -- and f32 stops at 21 regardless, Pkd-tree's f32
# build-assertion limit (25 and 29 are otherwise exact heights, but Pkd-tree
# aborts the process above 2^21 for f32, so they are not usable here while
# Pkd-tree stays in LIBRARIES). Override F64_HEIGHTS/F32_HEIGHTS for a wider
# sweep, but keep every entry on the multiple-of-BH list above.

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=lib_bench_profile_matrix.sh
source "$SCRIPT_DIR/lib_bench_profile_matrix.sh"

readonly PHASE=single
readonly REQUIRED_PROFILE=single-core
readonly EXPECT_VERB=expect-single-core
readonly STRICT_ISOLATION=1

LIBRARIES=${LIBRARIES:-nanoflann,pkdtree,kiddo}
STRATEGIES=${STRATEGIES:-$ALL_STRATEGIES}
F64_HEIGHTS=${F64_HEIGHTS:-20,23,26}
F32_HEIGHTS=${F32_HEIGHTS:-21}
SCALARS=${SCALARS:-f64,f32}
SINGLE_QUERY_COUNT=${SINGLE_QUERY_COUNT:-1000}
RADIUS=${RADIUS:-0.05}
BENCH_CPU=${BENCH_CPU:-8}
CRITERION_WARMUP=${CRITERION_WARMUP:-3}
CRITERION_MEASUREMENT=${CRITERION_MEASUREMENT:-8}
CRITERION_SAMPLE_SIZE=${CRITERION_SAMPLE_SIZE:-30}
IRQ_RETRIES=${IRQ_RETRIES:-2}
FEATURES=${FEATURES:-cpp_competitors,test_utils,logging_off,simd}
OUTPUT_BASE=${OUTPUT_BASE:-$REPO_DIR/cpp-competitor-results}
CHART_BASE=${CHART_BASE:-$REPO_DIR/cpp-competitor-charts}
SUITE_LABEL=${SUITE_LABEL:-single-core-strategy-matrix-$(date -u +%Y%m%dT%H%M%SZ)}

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

# The bench-profile boot entries carry their identity on the kernel cmdline.
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

IFS=',' read -r -a f64_heights <<<"$F64_HEIGHTS"
IFS=',' read -r -a f32_heights <<<"$F32_HEIGHTS"
IFS=',' read -r -a scalars <<<"$SCALARS"
IFS=',' read -r -a strategies <<<"$STRATEGIES"

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

# Pkd-tree's build_recursive asserts above 2^21 points for f32, whatever K is
# (reconfirmed at K_F32=4; see profile_cpp_competitors.rs).
if contains f32 "${scalars[@]}" && [[ "$LIBRARIES" == *pkdtree* ]]; then
    for height in "${f32_heights[@]}"; do
        (( height <= 21 )) || {
            echo "F32_HEIGHTS entry $height exceeds Pkd-tree's f32 build limit of 2^21" >&2
            exit 2
        }
    done
fi

mapfile -t single_bench_cpus < <(expand_cpu_list "$BENCH_CPU") || {
    echo "BENCH_CPU is not a valid CPU: $BENCH_CPU" >&2
    exit 2
}
(( ${#single_bench_cpus[@]} == 1 )) || {
    echo "This script measures sequential latency; BENCH_CPU must name exactly one CPU" >&2
    exit 2
}
readonly WORKER_COUNT=1
single_pin_command=(taskset --cpu-list "$BENCH_CPU")
readonly single_pin_command
# validate_environment (shared lib) reads bench_cpus/pin_command/BENCH_CPUS,
# not the single_* names above -- alias them so it validates the one core
# this script actually uses.
bench_cpus=("${single_bench_cpus[@]}")
readonly BENCH_CPUS="$BENCH_CPU"
pin_command=("${single_pin_command[@]}")

mkdir -p -- "$RUN_DIR" "$CHART_DIR"

build_benchmark
validate_environment "$RUN_DIR/environment-before.txt"

# --- preflight ---------------------------------------------------------------

readonly PREFLIGHT_DIR="$(mktemp -d /dev/shm/kiddo-single-preflight.XXXXXX)"
cleanup_preflight() { rm -rf -- "$PREFLIGHT_DIR"; }
trap cleanup_preflight EXIT INT TERM

log "Preflight: kiddo_vs_pkdtree f64 2^14"
preflight_kvp=$(run_single_cell profile_kiddo_vs_pkdtree \
    "KIDDO_CPP_SUITES=kiddo_vs_pkdtree KIDDO_CPP_LIBRARIES=kiddo,pkdtree KIDDO_CPP_SCALARS=f64 KIDDO_STRATEGIES=$STRATEGIES" \
    f64 14 preflight 0.5 1 10 "$PREFLIGHT_DIR")
(( $(jq '.results | length' "$preflight_kvp") > 0 )) || {
    echo "kiddo_vs_pkdtree preflight produced no results" >&2
    exit 1
}
if [[ "$LIBRARIES" == *nanoflann* ]]; then
    log "Preflight: cpp_competitors f64 2^14"
    preflight_cpp=$(run_single_cell profile_cpp_competitors \
        "KIDDO_CPP_SUITES=cpp_competitors KIDDO_CPP_LIBRARIES=nanoflann KIDDO_CPP_SCALARS=f64 KIDDO_PROFILE_RADIUS=$RADIUS" \
        f64 14 preflight 0.5 1 10 "$PREFLIGHT_DIR")
    (( $(jq '.results | length' "$preflight_cpp")  > 0 )) || {
        echo "cpp_competitors preflight produced no results" >&2
        exit 1
    }
fi
cleanup_preflight
trap - EXIT INT TERM

# --- matrix --------------------------------------------------------------

planned=0
for scalar in "${scalars[@]}"; do
    if [[ "$scalar" == f64 ]]; then
        planned=$((planned + ${#f64_heights[@]}))
    else
        planned=$((planned + ${#f32_heights[@]}))
    fi
done
if [[ "$LIBRARIES" == *nanoflann* ]]; then
    planned=$((planned * 2))
fi
log "Phase: single (boot profile: ${BOOT_PROFILE:-none}, gate: $EXPECT_VERB, cpu: $BENCH_CPU)"
log "Libraries: $LIBRARIES  strategies: $STRATEGIES  scalars: $SCALARS"
log "Planned cells: $planned"

kvp_results=()
cpp_results=()
completed=0

log "=== kiddo_vs_pkdtree: kiddo ($STRATEGIES) vs Pkd-tree, cpu $BENCH_CPU ==="
for scalar in "${scalars[@]}"; do
    if [[ "$scalar" == f64 ]]; then
        heights=("${f64_heights[@]}")
    else
        heights=("${f32_heights[@]}")
    fi
    for height in "${heights[@]}"; do
        completed=$((completed + 1))
        log "[$completed/$planned] kiddo_vs_pkdtree $scalar 2^$height"
        kvp_results+=("$(run_single_cell profile_kiddo_vs_pkdtree \
            "KIDDO_CPP_SUITES=kiddo_vs_pkdtree KIDDO_CPP_LIBRARIES=kiddo,pkdtree KIDDO_CPP_SCALARS=$scalar KIDDO_STRATEGIES=$STRATEGIES" \
            "$scalar" "$height" matrix)")
    done
done

if [[ "$LIBRARIES" == *nanoflann* ]]; then
    log "=== cpp_competitors: nanoflann, cpu $BENCH_CPU ==="
    for scalar in "${scalars[@]}"; do
        if [[ "$scalar" == f64 ]]; then
            heights=("${f64_heights[@]}")
        else
            heights=("${f32_heights[@]}")
        fi
        for height in "${heights[@]}"; do
            completed=$((completed + 1))
            log "[$completed/$planned] cpp_competitors $scalar 2^$height"
            cpp_results+=("$(run_single_cell profile_cpp_competitors \
                "KIDDO_CPP_SUITES=cpp_competitors KIDDO_CPP_LIBRARIES=nanoflann KIDDO_CPP_SCALARS=$scalar KIDDO_PROFILE_MIN_LOG2_POINTS=$height KIDDO_PROFILE_MAX_LOG2_POINTS=$height KIDDO_PROFILE_RADIUS=$RADIUS" \
                "$scalar" "$height" matrix)")
        done
    done
fi

validate_environment "$RUN_DIR/environment-after.txt"

# --- charts --------------------------------------------------------------

if (( ${#kvp_results[@]} > 0 )); then
    chart_args=()
    for result in "${kvp_results[@]}"; do
        chart_args+=(--result "$result")
    done
    python3 "$SCRIPT_DIR/chart_kiddo_vs_pkdtree_results.py" all \
        "${chart_args[@]}" \
        --result-label "$SUITE_LABEL" \
        --output-dir "$CHART_DIR" \
        --html-name "$SUITE_LABEL-kiddo-vs-pkdtree.html"
    log "kiddo vs Pkd-tree charts: $CHART_DIR/$SUITE_LABEL-kiddo-vs-pkdtree.html"
fi

if (( ${#cpp_results[@]} > 0 )); then
    log "nanoflann results (same session and sizes as the kiddo_vs_pkdtree results above,"
    log "so byte-identical point clouds and queries at each size -- but a different"
    log "Criterion group, so no existing chart script plots all three libraries"
    log "together yet. chart_cpp_competitor_results.py's --kiddo-* flags want a"
    log "profile_v6_* export, which is a still-separate session; do not use it here,"
    log "that is the exact gap this script exists to close for kiddo vs Pkd-tree."
    log "For now, compare the raw numbers below against the kiddo_vs_pkdtree chart,"
    log "or merge the JSON exports (matching group_id/value_str) by hand."
    log "  nanoflann result files:"
    for result in "${cpp_results[@]}"; do
        log "    $result"
    done
fi

log "Suite complete: $((${#kvp_results[@]} + ${#cpp_results[@]})) result files in $RUN_DIR"

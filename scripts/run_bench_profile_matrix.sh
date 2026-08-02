#!/usr/bin/env bash
set -euo pipefail

# Competitor matrix for the bench-profile boot profiles.
#
# ONE PHASE PER INVOCATION, because each phase needs a different boot profile
# and the machine can only be in one at a time. The phase is derived from the
# booted profile by default; the run is gated on `bench-profile-status
# expect-<profile>` so a phase can never silently execute in the wrong CPU
# state. That gate exists because it already happened once: a full parallel
# matrix was collected under isolcpus=domain, which disables load balancing, so
# every thread pool ran on a single core and the numbers were meaningless.
#
#   PHASE single      (boot: benchmark single-core)
#     Per-query latency: nanoflann and Pkd-tree's sequential mode, plus kiddo's
#     own serial benches. Inherently single-threaded, so it pins to ONE
#     isolated core -- the maximum-isolation case, matching the methodology of
#     the existing single-core query-pool runs. Both runtimes are forced to one
#     worker: ParlayLib starts its global scheduler on first use whether or not
#     parallel_for is called, so without PARLAY_NUM_THREADS=1 it would put a
#     worker per hardware thread onto that single core.
#
#   PHASE parallel    (boot: benchmark multi-core)
#     Parallel throughput, controlled: kiddo (serial, parallel and tuned
#     executors, Eytzinger and Donnelly stems) against Pkd-tree's parallel_for,
#     across one whole core complex. Each CCD has its own L3 and its own path
#     to the memory controller, so confining the run to one CCD removes
#     cross-complex traffic and keeps housekeeping's cache footprint entirely
#     on the other complex. Boost is off in this profile, so cells measured
#     minutes apart stay comparable. This is the "does A beat B" phase.
#
#   PHASE unlimited   (boot: benchmark unlimited)
#     No holds barred: every thread, SMT on, boost on, no pinning and no
#     isolation. kiddo against Pkd-tree only -- this phase exists to settle who
#     holds the crown for maximum parallel query rate on this machine, not to
#     compare implementation details. Isolation and IRQ gates relax to
#     warnings, because with every core in use they cannot hold by
#     construction. Numbers here are NOT comparable with the parallel phase's.
#
# Every phase pins both runtimes to the same worker count. This is not a
# nicety: ParlayLib falls back to `std::thread::hardware_concurrency()`, which
# ignores the affinity mask, while rayon uses `available_parallelism()`, which
# honours it. Left alone under taskset, Pkd-tree would oversubscribe the
# benchmark cores while kiddo did not, and the comparison would be meaningless.
#
# This deliberately does NOT use bench-profile-run: that wrapper pins to the
# profile's whole benchmark set, which is right for the parallel phase but
# wrong for the single phase's per-cell IRQ accounting and retry logic, which
# is implemented here instead.

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# auto -> derive the phase from the booted profile. Override only to force a
# phase in an unusual environment; the profile gate still applies.
PHASE=${PHASE:-auto}
SINGLE_LIBRARIES=${SINGLE_LIBRARIES:-nanoflann,pkdtree}
SINGLE_F64_HEIGHTS=${SINGLE_F64_HEIGHTS:-16,18,20,22,24}
SINGLE_F32_HEIGHTS=${SINGLE_F32_HEIGHTS:-16,18,20,21}
SINGLE_QUERY_COUNT=${SINGLE_QUERY_COUNT:-1000}
SINGLE_RADIUS=${SINGLE_RADIUS:-0.05}
BENCH_CPUS=${BENCH_CPUS:-}
F64_HEIGHTS=${F64_HEIGHTS:-24,25,26,27}
F32_HEIGHTS=${F32_HEIGHTS:-21}
QUERY_COUNTS=${QUERY_COUNTS:-1000,4096,16384,100000}
SCALARS=${SCALARS:-f64,f32}
BATCH_LIBRARIES=${BATCH_LIBRARIES:-kiddo,pkdtree}
# The unlimited phase is a head-to-head for the maximum-parallel-QPS crown, so
# it is deliberately limited to the two contenders and to the largest cells.
UNLIMITED_LIBRARIES=${UNLIMITED_LIBRARIES:-kiddo,pkdtree}
UNLIMITED_F64_HEIGHTS=${UNLIMITED_F64_HEIGHTS:-24,27}
UNLIMITED_F32_HEIGHTS=${UNLIMITED_F32_HEIGHTS:-21}
UNLIMITED_QUERY_COUNTS=${UNLIMITED_QUERY_COUNTS:-100000}
CRITERION_WARMUP=${CRITERION_WARMUP:-3}
CRITERION_MEASUREMENT=${CRITERION_MEASUREMENT:-8}
CRITERION_SAMPLE_SIZE=${CRITERION_SAMPLE_SIZE:-30}
IRQ_RETRIES=${IRQ_RETRIES:-2}
FEATURES=${FEATURES:-cpp_competitors,test_utils,logging_off,simd}
OUTPUT_BASE=${OUTPUT_BASE:-$REPO_DIR/cpp-competitor-results}
CHART_BASE=${CHART_BASE:-$REPO_DIR/cpp-competitor-charts}
SUITE_LABEL=${SUITE_LABEL:-batch-parallel-matrix-$(date -u +%Y%m%dT%H%M%SZ)}

readonly RUN_DIR="$OUTPUT_BASE/$SUITE_LABEL"
readonly CHART_DIR="$CHART_BASE/$SUITE_LABEL"
readonly LOG_FILE="$RUN_DIR/run.log"

for command_name in cargo jq mktemp python3 rustc taskset; do
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

# --- phase and boot profile -------------------------------------------------

# The bench-profile boot entries carry their identity on the kernel cmdline.
booted_profile() {
    local token
    for token in $(< /proc/cmdline); do
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

if [[ "$PHASE" == auto ]]; then
    case "$BOOT_PROFILE" in
        single-core) PHASE=single ;;
        multi-core)  PHASE=parallel ;;
        unlimited)   PHASE=unlimited ;;
        *)
            cat >&2 <<'EOF'
Not booted into a bench-profile benchmark profile, so no phase can be derived.

Each phase needs a different CPU state, and the machine can only be in one at a
time. Reboot into the entry for the phase you want:

  ... benchmark single-core   -> PHASE=single     nanoflann + Pkd-tree, one core
  ... benchmark multi-core    -> PHASE=parallel   kiddo vs Pkd-tree, one CCD
  ... benchmark unlimited     -> PHASE=unlimited  kiddo vs Pkd-tree, whole machine

Install the profiles from https://github.com/sdd/bench-profile if they are not
in the boot menu. Set PHASE= explicitly only to override this deliberately.
EOF
            exit 2
            ;;
    esac
fi

case "$PHASE" in
    single)   REQUIRED_PROFILE=single-core; EXPECT_VERB=expect-single-core ;;
    parallel) REQUIRED_PROFILE=multi-core;  EXPECT_VERB=expect-multi-core ;;
    unlimited)
        REQUIRED_PROFILE=unlimited
        EXPECT_VERB=expect-unrestricted
        # The crown run is a head-to-head at the largest sizes only, so it
        # replaces the matrix dimensions before they are parsed below.
        BATCH_LIBRARIES="$UNLIMITED_LIBRARIES"
        F64_HEIGHTS="$UNLIMITED_F64_HEIGHTS"
        F32_HEIGHTS="$UNLIMITED_F32_HEIGHTS"
        QUERY_COUNTS="$UNLIMITED_QUERY_COUNTS"
        ;;
    *)
        echo "PHASE must be single, parallel or unlimited: $PHASE" >&2
        exit 2
        ;;
esac
readonly PHASE REQUIRED_PROFILE EXPECT_VERB

if [[ -n "$BOOT_PROFILE" && "$BOOT_PROFILE" != "$REQUIRED_PROFILE" ]]; then
    echo "PHASE=$PHASE needs the '$REQUIRED_PROFILE' boot profile, but this machine" >&2
    echo "is booted into '$BOOT_PROFILE'. Reboot, or accept that the result is not" >&2
    echo "comparable with the rest of the series." >&2
    exit 2
fi

IFS=',' read -r -a f64_heights <<<"$F64_HEIGHTS"
IFS=',' read -r -a f32_heights <<<"$F32_HEIGHTS"
IFS=',' read -r -a query_counts <<<"$QUERY_COUNTS"
IFS=',' read -r -a scalars <<<"$SCALARS"

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

for value_name in CRITERION_WARMUP CRITERION_MEASUREMENT; do
    value=${!value_name}
    [[ "$value" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || {
        echo "$value_name must be a non-negative number: $value" >&2
        exit 2
    }
done
[[ "$CRITERION_SAMPLE_SIZE" =~ ^[1-9][0-9]*$ ]] && (( CRITERION_SAMPLE_SIZE >= 10 )) || {
    echo "CRITERION_SAMPLE_SIZE must be an integer of at least 10" >&2
    exit 2
}
[[ "$IRQ_RETRIES" =~ ^[0-9]+$ ]] || {
    echo "IRQ_RETRIES must be a non-negative integer" >&2
    exit 2
}

# Pkd-tree's f32 build asserts above 2^21 points, aborting the process.
if contains f32 "${scalars[@]}" && [[ "$BATCH_LIBRARIES" == *pkdtree* ]]; then
    for height in "${f32_heights[@]}"; do
        (( height <= 21 )) || {
            echo "F32_HEIGHTS entry $height exceeds Pkd-tree's f32 build limit of 2^21" >&2
            exit 2
        }
    done
fi

IFS=',' read -r -a single_f64_heights <<<"$SINGLE_F64_HEIGHTS"
IFS=',' read -r -a single_f32_heights <<<"$SINGLE_F32_HEIGHTS"
if [[ "$PHASE" == single ]]; then
    contains f64 "${scalars[@]}" && validate_unique_positive_list SINGLE_F64_HEIGHTS "${single_f64_heights[@]}"
    contains f32 "${scalars[@]}" && validate_unique_positive_list SINGLE_F32_HEIGHTS "${single_f32_heights[@]}"
    validate_unique_positive_list SINGLE_QUERY_COUNT "$SINGLE_QUERY_COUNT"
    # Pkd-tree's f32 build asserts above 2^21 in the single-query suite too.
    if [[ "$SINGLE_LIBRARIES" == *pkdtree* ]] && contains f32 "${scalars[@]}"; then
        for height in "${single_f32_heights[@]}"; do
            (( height <= 21 )) || {
                echo "SINGLE_F32_HEIGHTS entry $height exceeds Pkd-tree's f32 build limit of 2^21" >&2
                exit 2
            }
        done
    fi
fi

expand_cpu_list() {
    local spec=$1 segment start end cpu
    local -a segments
    IFS=',' read -r -a segments <<<"$spec"
    for segment in "${segments[@]}"; do
        if [[ "$segment" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            start=${BASH_REMATCH[1]}
            end=${BASH_REMATCH[2]}
            (( start <= end )) || return 1
            for ((cpu = start; cpu <= end; cpu++)); do printf '%s\n' "$cpu"; done
        elif [[ "$segment" =~ ^[0-9]+$ ]]; then
            printf '%s\n' "$segment"
        else
            return 1
        fi
    done
}

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

# --- benchmark CPU set ------------------------------------------------------

case "$PHASE" in
    single)
        # Sequential latency: one isolated core is the maximum-isolation case,
        # and adding cores could not make a sequential query faster.
        BENCH_CPUS=${BENCH_CPUS:-8}
        readonly STRICT_ISOLATION=1
        ;;
    parallel)
        # One isolated core complex: its own L3, its own path to the memory
        # controller, housekeeping confined to the other complex.
        BENCH_CPUS=${BENCH_CPUS:-8-15}
        readonly STRICT_ISOLATION=1
        ;;
    unlimited)
        # Every online CPU, SMT included. Isolation cannot hold by
        # construction, so the environment gates downgrade to warnings and IRQs
        # stop invalidating cells; otherwise every cell would be discarded.
        if [[ -z "$BENCH_CPUS" ]]; then
            BENCH_CPUS=$(online_cpu_list | paste -sd, -)
        fi
        readonly STRICT_ISOLATION=0
        ;;
esac


if [[ -n "$BOOT_PROFILE" && "$BOOT_PROFILE" != "$REQUIRED_PROFILE" ]]; then
    echo "PHASE=$PHASE needs the '$REQUIRED_PROFILE' boot profile, but this machine" >&2
    echo "is booted into '$BOOT_PROFILE'. Reboot, or accept that the result is not" >&2
    echo "comparable with the rest of the series." >&2
    exit 2
fi

mapfile -t bench_cpus < <(expand_cpu_list "$BENCH_CPUS") || {
    echo "BENCH_CPUS is not a valid CPU list: $BENCH_CPUS" >&2
    exit 2
}
if [[ "$PHASE" == single ]]; then
    (( ${#bench_cpus[@]} == 1 )) || {
        echo "The single phase measures sequential latency; BENCH_CPUS must name one CPU" >&2
        exit 2
    }
else
    (( ${#bench_cpus[@]} > 1 )) || {
        echo "PHASE=$PHASE measures parallel throughput; BENCH_CPUS must name more than one CPU" >&2
        exit 2
    }
fi
readonly WORKER_COUNT=${#bench_cpus[@]}

if (( STRICT_ISOLATION == 1 )); then
    pin_command=(taskset --cpu-list "$BENCH_CPUS")
else
    pin_command=(env)
fi
readonly pin_command

mkdir -p -- "$RUN_DIR" "$CHART_DIR"

log() {
    printf '%s\n' "$*" | tee -a "$LOG_FILE"
}

# --- environment validation -------------------------------------------------

validate_environment() {
    local status_file=$1
    local failures=0 cpu isolated online governor siblings sibling
    local -a isolated_cpus

    # In machine mode nothing below can hold: the point is to use every core.
    local verdict=FAIL
    (( STRICT_ISOLATION == 1 )) || verdict=WARN

    {
        echo "phase: $PHASE (boot profile: ${BOOT_PROFILE:-none})"
        echo "benchmark CPUs: $BENCH_CPUS ($WORKER_COUNT workers)"
        echo "kernel cmdline: $(cat /proc/cmdline)"
    } >"$status_file"

    isolated=$(cat /sys/devices/system/cpu/isolated 2>/dev/null || true)
    mapfile -t isolated_cpus < <(expand_cpu_list "${isolated:-}" 2>/dev/null || true)

    for cpu in "${bench_cpus[@]}"; do
        online=1
        if [[ -r "/sys/devices/system/cpu/cpu$cpu/online" ]]; then
            online=$(<"/sys/devices/system/cpu/cpu$cpu/online")
        fi
        if (( online != 1 )); then
            echo "FAIL: cpu$cpu is offline" >>"$status_file"
            failures=$((failures + 1))
            continue
        fi

        if contains "$cpu" "${isolated_cpus[@]}"; then
            echo "PASS: cpu$cpu isolated" >>"$status_file"
        else
            echo "$verdict: cpu$cpu is not in isolcpus (${isolated:-none})" >>"$status_file"
            (( STRICT_ISOLATION == 1 )) && failures=$((failures + 1))
        fi

        governor=$(cat "/sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_governor" 2>/dev/null || echo unknown)
        if [[ "$governor" == performance ]]; then
            echo "PASS: cpu$cpu governor performance" >>"$status_file"
        else
            # bench-profile-status only checks the single BENCHMARK_CPU, so a
            # profile provisioned for single-core work routinely leaves the
            # rest of the isolated set on powersave. Say how to fix it.
            echo "FAIL: cpu$cpu governor is $governor (fix: sudo cpupower -c $BENCH_CPUS frequency-set -g performance)" >>"$status_file"
            failures=$((failures + 1))
        fi

        # An online SMT sibling outside the set would let unrelated work share
        # the core's execution resources, which shows up as throughput noise.
        siblings=$(cat "/sys/devices/system/cpu/cpu$cpu/topology/thread_siblings_list" 2>/dev/null || echo "$cpu")
        while read -r sibling; do
            [[ -n "$sibling" && "$sibling" != "$cpu" ]] || continue
            online=1
            if [[ -r "/sys/devices/system/cpu/cpu$sibling/online" ]]; then
                online=$(<"/sys/devices/system/cpu/cpu$sibling/online")
            fi
            if (( online == 1 )) && ! contains "$sibling" "${bench_cpus[@]}"; then
                echo "$verdict: cpu$cpu has online SMT sibling cpu$sibling outside BENCH_CPUS" >>"$status_file"
                (( STRICT_ISOLATION == 1 )) && failures=$((failures + 1))
            fi
        done < <(expand_cpu_list "$siblings" 2>/dev/null || true)
    done

    # Something has to service interrupts and run kernel threads.
    local housekeeping=0 candidate
    for candidate in /sys/devices/system/cpu/cpu[0-9]*; do
        cpu=${candidate##*/cpu}
        [[ "$cpu" =~ ^[0-9]+$ ]] || continue
        online=1
        if [[ -r "$candidate/online" ]]; then
            online=$(<"$candidate/online")
        fi
        (( online == 1 )) || continue
        contains "$cpu" "${bench_cpus[@]}" || housekeeping=$((housekeeping + 1))
    done
    if (( housekeeping > 0 )); then
        echo "PASS: $housekeeping online housekeeping CPUs outside BENCH_CPUS" >>"$status_file"
    else
        echo "$verdict: no online CPU outside BENCH_CPUS to take IRQs" >>"$status_file"
        (( STRICT_ISOLATION == 1 )) && failures=$((failures + 1))
    fi

    # bench-profile-status is the authority when installed: it asserts this
    # phase's whole desired CPU state, and -- for the parallel phases -- runs
    # the parallel-dispatch canary, the only check that would have caught the
    # isolcpus=domain collapse that silently invalidated the 2026-08-02 matrix.
    # The inline checks above remain the fallback for machines without it.
    if command -v bench-profile-status >/dev/null; then
        echo "--- bench-profile-status $EXPECT_VERB ---" >>"$status_file"
        if bench-profile-status "$EXPECT_VERB" >>"$status_file" 2>&1; then
            echo "PASS: bench-profile-status $EXPECT_VERB" >>"$status_file"
        else
            echo "FAIL: the machine is not in the '$REQUIRED_PROFILE' benchmark state." >>"$status_file"
            echo "      Boot the '... benchmark $REQUIRED_PROFILE' entry; bench-prep.service" >>"$status_file"
            echo "      applies the policy, or run 'sudo bench-prep' by hand." >>"$status_file"
            # The unlimited phase cannot satisfy the isolation gates by design,
            # but a failed canary there is still fatal, so the verb is still run
            # and recorded -- it just does not block.
            (( STRICT_ISOLATION == 1 )) && failures=$((failures + 1))
        fi
    else
        echo "NOTE: bench-profile-status is not installed; relying on the inline checks" >>"$status_file"
        echo "      above, which cannot detect a collapsed thread pool." >>"$status_file"
        echo "      Install from https://github.com/sdd/bench-profile" >>"$status_file"
    fi

    if (( failures > 0 )); then
        cat "$status_file" >&2
        echo "Benchmark environment validation failed with $failures problems." >&2
        return 1
    fi
    cat "$status_file" >>"$LOG_FILE"
}

# --- IRQ accounting ---------------------------------------------------------

# Total hardware interrupts delivered to the benchmark CPUs so far.
interrupt_total() {
    local cpu_csv
    cpu_csv=$(IFS=,; echo "${bench_cpus[*]}")
    awk -v cpus="$cpu_csv" '
        BEGIN {
            split(cpus, wanted, ",")
            for (i in wanted) target["CPU" wanted[i]] = 1
        }
        NR == 1 {
            for (field = 1; field <= NF; field++)
                if ($field in target) column[field + 1] = 1
            next
        }
        $1 ~ /^[0-9]+:$/ {
            for (field in column)
                if (field <= NF && $field ~ /^[0-9]+$/) total += $field
        }
        END { print total + 0 }
    ' /proc/interrupts
}

# --- build ------------------------------------------------------------------

log "Building profile_cpp_competitors (features: $FEATURES)"
benchmark_exe="$(
    cd -- "$REPO_DIR"
    RUSTC_WRAPPER= RUSTFLAGS='-C target-cpu=native' \
        cargo bench --bench profile_cpp_competitors \
        --features "$FEATURES" --no-run --message-format=json |
        jq -r 'select(.reason == "compiler-artifact" and .target.name == "profile_cpp_competitors") | .executable // empty' |
        tail -n 1
)"
[[ -n "$benchmark_exe" && -x "$benchmark_exe" ]] || {
    echo "Could not resolve profile_cpp_competitors executable" >&2
    exit 1
}

opt_level=$(cd -- "$REPO_DIR" && python3 - <<'PY'
import re, pathlib
text = pathlib.Path("Cargo.toml").read_text()
section = re.search(r"\[profile\.bench\](.*?)(?=\n\[|\Z)", text, re.S)
match = re.search(r"opt-level\s*=\s*(\d+)", section.group(1)) if section else None
print(match.group(1) if match else "unset")
PY
)
[[ "$opt_level" == 3 ]] || {
    echo "profile.bench opt-level is $opt_level; hand-written SIMD kernels must be measured at 3" >&2
    exit 2
}

native_cfg=$(rustc --print cfg -C target-cpu=native)
[[ "$native_cfg" == *'target_feature="avx512f"'* ]] || {
    echo "Donnelly cyclic SIMD descent requires native AVX-512F support" >&2
    exit 2
}

validate_environment "$RUN_DIR/environment-before.txt"

# --- cell execution ---------------------------------------------------------

run_cell() {
    local scalar=$1 height=$2 query_count=$3 role=$4
    local warmup=${5:-$CRITERION_WARMUP}
    local measurement=${6:-$CRITERION_MEASUREMENT}
    local sample_size=${7:-$CRITERION_SAMPLE_SIZE}
    local output_dir=${8:-$RUN_DIR}
    local attempt=0 status result_key result_path tmp_dir irq_before irq_after

    result_key="$SUITE_LABEL-$PHASE-batch-$role-$scalar-2p$height-q$query_count"
    result_path="$output_dir/bench_result-pkdtree-batch-$result_key.json"

    while true; do
        tmp_dir="$(mktemp -d /dev/shm/kiddo-batch-parallel.XXXXXX)"
        cp -- "$benchmark_exe" "$tmp_dir/benchmark"
        chmod 0755 "$tmp_dir/benchmark"

        irq_before=$(interrupt_total)
        set +e
        "${pin_command[@]}" env \
            CRITERION_HOME="$tmp_dir/criterion" \
            RAYON_NUM_THREADS="$WORKER_COUNT" \
            PARLAY_NUM_THREADS="$WORKER_COUNT" \
            KIDDO_CPP_SUITES=pkdtree_batch \
            KIDDO_CPP_LIBRARIES="$BATCH_LIBRARIES" \
            KIDDO_CPP_SCALARS="$scalar" \
            KIDDO_PROFILE_QUERIES="$query_count" \
            KIDDO_LARGE_MIN_LOG2_POINTS="$height" \
            KIDDO_LARGE_MAX_LOG2_POINTS="$height" \
            "$tmp_dir/benchmark" \
            profile_pkdtree_batch \
            --warm-up-time "$warmup" \
            --measurement-time "$measurement" \
            --sample-size "$sample_size" \
            --noplot --bench 2>&1 | tee -a "$LOG_FILE" >&2
        status=${PIPESTATUS[0]}
        set -e
        irq_after=$(interrupt_total)

        if (( status == 0 && irq_after > irq_before )); then
            echo "Cell $role $scalar 2^$height q$query_count saw $((irq_after - irq_before)) hardware IRQs on the benchmark CPUs" |
                tee -a "$LOG_FILE" >&2
            # Only a controlled run can promise an IRQ-free interval.
            (( STRICT_ISOLATION == 1 )) && status=125
        fi

        if (( status == 0 )); then
            (
                cd -- "$REPO_DIR"
                cargo run --quiet --manifest-path tools/criterion-export/Cargo.toml -- \
                    "$tmp_dir/criterion" "$result_path" profile_pkdtree_batch >&2
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
        echo "Retrying invalidated cell $role $scalar 2^$height q$query_count ($attempt/$IRQ_RETRIES)" |
            tee -a "$LOG_FILE" >&2
    done
}


# --- single-query cells -----------------------------------------------------

# Per-query latency is sequential, so this pins to the FIRST benchmark CPU
# only. Using the whole set would not make a sequential query faster and would
# expose the measurement to seven more cores' worth of interference.
readonly SINGLE_CPU=${bench_cpus[0]}
if (( STRICT_ISOLATION == 1 )); then
    single_pin_command=(taskset --cpu-list "$SINGLE_CPU")
else
    single_pin_command=(env)
fi
readonly single_pin_command

run_single_cell() {
    local scalar=$1 height=$2 role=$3
    local warmup=${4:-$CRITERION_WARMUP}
    local measurement=${5:-$CRITERION_MEASUREMENT}
    local sample_size=${6:-$CRITERION_SAMPLE_SIZE}
    local output_dir=${7:-$RUN_DIR}
    local attempt=0 status result_key result_path tmp_dir irq_before irq_after

    result_key="$SUITE_LABEL-$PHASE-single-$role-$scalar-2p$height"
    result_path="$output_dir/bench_result-cpp-competitors-$result_key.json"

    while true; do
        tmp_dir="$(mktemp -d /dev/shm/kiddo-single.XXXXXX)"
        cp -- "$benchmark_exe" "$tmp_dir/benchmark"
        chmod 0755 "$tmp_dir/benchmark"

        irq_before=$(interrupt_total)
        set +e
        "${single_pin_command[@]}" env \
            CRITERION_HOME="$tmp_dir/criterion" \
            RAYON_NUM_THREADS=1 \
            PARLAY_NUM_THREADS=1 \
            KIDDO_CPP_SUITES=cpp_competitors \
            KIDDO_CPP_LIBRARIES="$SINGLE_LIBRARIES" \
            KIDDO_CPP_SCALARS="$scalar" \
            KIDDO_PROFILE_QUERIES="$SINGLE_QUERY_COUNT" \
            KIDDO_PROFILE_MIN_LOG2_POINTS="$height" \
            KIDDO_PROFILE_MAX_LOG2_POINTS="$height" \
            KIDDO_PROFILE_RADIUS="$SINGLE_RADIUS" \
            "$tmp_dir/benchmark" \
            profile_cpp_competitors \
            --warm-up-time "$warmup" \
            --measurement-time "$measurement" \
            --sample-size "$sample_size" \
            --noplot --bench 2>&1 | tee -a "$LOG_FILE" >&2
        status=${PIPESTATUS[0]}
        set -e
        irq_after=$(interrupt_total)

        if (( status == 0 && irq_after > irq_before )); then
            echo "Single cell $role $scalar 2^$height saw $((irq_after - irq_before)) hardware IRQs" |
                tee -a "$LOG_FILE" >&2
            (( STRICT_ISOLATION == 1 )) && status=125
        fi

        if (( status == 0 )); then
            (
                cd -- "$REPO_DIR"
                cargo run --quiet --manifest-path tools/criterion-export/Cargo.toml -- \
                    "$tmp_dir/criterion" "$result_path" profile_cpp_competitors >&2
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
        echo "Retrying invalidated single cell $role $scalar 2^$height ($attempt/$IRQ_RETRIES)" |
            tee -a "$LOG_FILE" >&2
    done
}

# --- preflight --------------------------------------------------------------

# Exercise the executable, pinning, IRQ accounting, exporter and chart parser
# on a tiny cell before committing to the matrix.
readonly PREFLIGHT_DIR="$(mktemp -d /dev/shm/kiddo-batch-preflight.XXXXXX)"
cleanup_preflight() { rm -rf -- "$PREFLIGHT_DIR"; }
trap cleanup_preflight EXIT INT TERM

if [[ "$PHASE" != single ]]; then
    log "Preflight ($PHASE): f64 2^16, 256 queries"
    preflight_result=$(run_cell f64 16 256 preflight 0.5 1 10 "$PREFLIGHT_DIR")
    (( $(jq '.results | length' "$preflight_result") > 0 )) || {
        echo "$PHASE preflight produced no results" >&2
        exit 1
    }
    python3 "$SCRIPT_DIR/chart_pkdtree_batch_results.py" all \
        --result "$preflight_result" --result-label "$SUITE_LABEL-preflight" \
        --output-dir "$PREFLIGHT_DIR/charts" --html-name preflight.html
fi
if [[ "$PHASE" == single ]]; then
    log "Preflight (single): f64 2^16"
    preflight_single=$(run_single_cell f64 16 preflight 0.5 1 10 "$PREFLIGHT_DIR")
    (( $(jq '.results | length' "$preflight_single") > 0 )) || {
        echo "Single preflight produced no results" >&2
        exit 1
    }
fi
cleanup_preflight
trap - EXIT INT TERM

# --- matrix -----------------------------------------------------------------

planned=0
if [[ "$PHASE" != single ]]; then
    for scalar in "${scalars[@]}"; do
        if [[ "$scalar" == f64 ]]; then
            planned=$((planned + ${#f64_heights[@]} * ${#query_counts[@]}))
        else
            planned=$((planned + ${#f32_heights[@]} * ${#query_counts[@]}))
        fi
    done
fi
if [[ "$PHASE" == single ]]; then
    for scalar in "${scalars[@]}"; do
        if [[ "$scalar" == f64 ]]; then
            planned=$((planned + ${#single_f64_heights[@]}))
        else
            planned=$((planned + ${#single_f32_heights[@]}))
        fi
    done
fi
log "Phase: $PHASE (boot profile: ${BOOT_PROFILE:-none}, gate: $EXPECT_VERB)"
log "Planned cells: $planned"
log "Per-cell budget: ${CRITERION_WARMUP}s warmup + ${CRITERION_MEASUREMENT}s measurement, ${CRITERION_SAMPLE_SIZE} samples"

batch_results=()
single_results=()
completed=0

if [[ "$PHASE" == single ]]; then
    log "=== single-query phase: $SINGLE_LIBRARIES on cpu $SINGLE_CPU ==="
    for scalar in "${scalars[@]}"; do
        if [[ "$scalar" == f64 ]]; then
            heights=("${single_f64_heights[@]}")
        else
            heights=("${single_f32_heights[@]}")
        fi
        for height in "${heights[@]}"; do
            completed=$((completed + 1))
            log "[$completed/$planned] single $scalar 2^$height"
            single_results+=("$(run_single_cell "$scalar" "$height" matrix)")
        done
    done
fi

if [[ "$PHASE" != single ]]; then
    log "=== $PHASE phase: $BATCH_LIBRARIES on cpus $BENCH_CPUS ($WORKER_COUNT workers) ==="
    for scalar in "${scalars[@]}"; do
        if [[ "$scalar" == f64 ]]; then
            heights=("${f64_heights[@]}")
        else
            heights=("${f32_heights[@]}")
        fi
        for height in "${heights[@]}"; do
            for query_count in "${query_counts[@]}"; do
                completed=$((completed + 1))
                log "[$completed/$planned] $PHASE $scalar 2^$height, $query_count queries"
                batch_results+=("$(run_cell "$scalar" "$height" "$query_count" matrix)")
            done
        done
    done
fi

validate_environment "$RUN_DIR/environment-after.txt"

# --- charts -----------------------------------------------------------------

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

# chart_cpp_competitor_results.py also needs kiddo's own single-query exports,
# which come from the focused v6 suites rather than from here, so the
# single-query phase stops at the result files and is charted separately.
if (( ${#single_results[@]} > 0 )); then
    log "Single-query results (chart with chart_cpp_competitor_results.py, adding the kiddo exports):"
    for result in "${single_results[@]}"; do
        log "  $result"
    done
fi

log "Suite complete: $((${#batch_results[@]} + ${#single_results[@]})) result files in $RUN_DIR"

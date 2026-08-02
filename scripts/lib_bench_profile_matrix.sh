# shellcheck shell=bash
#
# Shared machinery for the three run_*_strategy_matrix.sh scripts (single-core,
# multi-core, unlimited). Sourced, never executed directly.
#
# Split out because the three scripts share almost everything except which
# boot profile they require, which CPUs they use, and which suite(s) they
# select -- duplicating ~500 lines of environment validation, IRQ accounting
# and retry logic three times over would make the one thing that must stay in
# sync (the isolcpus=domain canary in validate_environment) three times as
# likely to drift.
#
# Callers set PHASE, REQUIRED_PROFILE, EXPECT_VERB, BENCH_CPUS, STRICT_ISOLATION
# before sourcing, then call the functions below.

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
    ((${#values[@]} > 0)) || {
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

expand_cpu_list() {
    local spec=$1 segment start end cpu
    local -a segments
    IFS=',' read -r -a segments <<<"$spec"
    for segment in "${segments[@]}"; do
        if [[ "$segment" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            start=${BASH_REMATCH[1]}
            end=${BASH_REMATCH[2]}
            ((start <= end)) || return 1
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
        ((online == 1)) && printf '%s\n' "$cpu"
    done | sort -n
}

# Every kiddo strategy this repository's bench file knows how to build, and
# the tree-native block height for each scalar. DonnellyCyclicSimdDescent and
# DonnellyCyclicSimdFull both panic unless BH matches this exactly (3 for
# f64, 4 for f32) -- see profile_cpp_competitors.rs's K_F32/K_F64 comment.
readonly ALL_STRATEGIES="eytzinger,donnelly_unrolled,donnelly_cyclic_simd_descent,donnelly_cyclic_simd_full"

log() {
    printf '%s\n' "$*" | tee -a "$LOG_FILE"
}

require_commands() {
    local command_name
    for command_name in "$@"; do
        command -v "$command_name" >/dev/null || {
            echo "Required command is unavailable: $command_name" >&2
            exit 2
        }
    done
}

# --- environment validation -------------------------------------------------

validate_environment() {
    local status_file=$1
    local failures=0 cpu isolated online governor siblings sibling
    local -a isolated_cpus

    # In relaxed (unlimited) mode nothing below can hold: the point is to use
    # every core, so isolation/sibling/housekeeping gates downgrade to
    # warnings rather than failing a run that was never meant to isolate.
    local verdict=FAIL
    ((STRICT_ISOLATION == 1)) || verdict=WARN

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
        if ((online != 1)); then
            echo "FAIL: cpu$cpu is offline" >>"$status_file"
            failures=$((failures + 1))
            continue
        fi

        if contains "$cpu" "${isolated_cpus[@]}"; then
            echo "PASS: cpu$cpu isolated" >>"$status_file"
        else
            echo "$verdict: cpu$cpu is not in isolcpus (${isolated:-none})" >>"$status_file"
            ((STRICT_ISOLATION == 1)) && failures=$((failures + 1))
        fi

        governor=$(cat "/sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_governor" 2>/dev/null || echo unknown)
        if [[ "$governor" == performance ]]; then
            echo "PASS: cpu$cpu governor performance" >>"$status_file"
        else
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
            if ((online == 1)) && ! contains "$sibling" "${bench_cpus[@]}"; then
                echo "$verdict: cpu$cpu has online SMT sibling cpu$sibling outside BENCH_CPUS" >>"$status_file"
                ((STRICT_ISOLATION == 1)) && failures=$((failures + 1))
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
        ((online == 1)) || continue
        contains "$cpu" "${bench_cpus[@]}" || housekeeping=$((housekeeping + 1))
    done
    if ((housekeeping > 0)); then
        echo "PASS: $housekeeping online housekeeping CPUs outside BENCH_CPUS" >>"$status_file"
    else
        echo "$verdict: no online CPU outside BENCH_CPUS to take IRQs" >>"$status_file"
        ((STRICT_ISOLATION == 1)) && failures=$((failures + 1))
    fi

    # bench-profile-status is the authority when installed: it asserts this
    # profile's whole desired CPU state, and -- for multi-core/unlimited --
    # runs the parallel-dispatch canary, the only check that would have caught
    # the isolcpus=domain collapse that silently invalidated an earlier
    # version of this matrix. The inline checks above are the fallback for
    # machines without it.
    if command -v bench-profile-status >/dev/null; then
        echo "--- bench-profile-status $EXPECT_VERB ---" >>"$status_file"
        if bench-profile-status "$EXPECT_VERB" >>"$status_file" 2>&1; then
            echo "PASS: bench-profile-status $EXPECT_VERB" >>"$status_file"
        else
            echo "FAIL: the machine is not in the '$REQUIRED_PROFILE' benchmark state." >>"$status_file"
            echo "      Boot the '... benchmark $REQUIRED_PROFILE' entry; bench-prep.service" >>"$status_file"
            echo "      applies the policy, or run 'sudo bench-prep' by hand." >>"$status_file"
            # Always fatal, regardless of STRICT_ISOLATION: that flag relaxes
            # isolation-specific checks that cannot hold by construction in
            # the unlimited profile (siblings/isolcpus/housekeeping), not
            # whether the machine is booted into the right profile at all. A
            # normal desktop boot must never be mistaken for the unlimited
            # benchmark profile just because both leave isolation off.
            failures=$((failures + 1))
        fi
    else
        echo "NOTE: bench-profile-status is not installed; relying on the inline checks" >>"$status_file"
        echo "      above, which cannot detect a collapsed thread pool." >>"$status_file"
        echo "      Install from https://github.com/sdd/bench-profile" >>"$status_file"
    fi

    if ((failures > 0)); then
        cat "$status_file" >&2
        echo "Benchmark environment validation failed with $failures problems." >&2
        return 1
    fi
    cat "$status_file" >>"$LOG_FILE"
}

# --- IRQ accounting ---------------------------------------------------------

# Total hardware interrupts delivered to the given CPUs so far.
interrupt_total() {
    local -a cpus=("$@")
    local cpu_csv
    cpu_csv=$(
        IFS=,
        echo "${cpus[*]}"
    )
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

# --- build & preconditions ---------------------------------------------------

# Builds profile_cpp_competitors and resolves its executable path. Validates
# the two build-time preconditions the cyclic Donnelly strategies need: an
# opt-level-3 bench profile (a hand-written SIMD kernel measured a level below
# release is not representative) and native AVX-512F support (without it every
# cyclic strategy panics on the very first tree it builds).
build_benchmark() {
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

    local opt_level
    opt_level=$(
        cd -- "$REPO_DIR" && python3 - <<'PY'
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

    local native_cfg
    native_cfg=$(rustc --print cfg -C target-cpu=native)
    [[ "$native_cfg" == *'target_feature="avx512f"'* ]] || {
        echo "The cyclic Donnelly strategies require native AVX-512F support" >&2
        exit 2
    }
}

# --- cell execution ----------------------------------------------------------
#
# run_cell and run_single_cell both retry a cell up to IRQ_RETRIES times if a
# hardware IRQ landed on the benchmark CPUs during the measurement window --
# bench-profile-run's 125 convention, reimplemented here rather than reused
# because that wrapper pins to the profile's whole benchmark set, which is
# wrong for run_single_cell's one-CPU pin.

# run_cell <criterion-suite> <extra-env-string> <scalar> <height> <query-count> <role> [warmup] [measurement] [sample-size] [output-dir]
#
# extra-env-string is a space-separated list of NAME=value pairs appended to
# the env invocation, so callers can select KIDDO_CPP_SUITES/LIBRARIES/
# STRATEGIES without this function needing to know about all three.
run_cell() {
    local criterion_group=$1 extra_env=$2 scalar=$3 height=$4 query_count=$5 role=$6
    local warmup=${7:-$CRITERION_WARMUP}
    local measurement=${8:-$CRITERION_MEASUREMENT}
    local sample_size=${9:-$CRITERION_SAMPLE_SIZE}
    local output_dir=${10:-$RUN_DIR}
    local attempt=0 status result_key result_path tmp_dir irq_before irq_after
    local -a extra_env_array
    read -ra extra_env_array <<<"$extra_env"

    result_key="$SUITE_LABEL-$PHASE-$role-$scalar-2p$height-q$query_count"
    local group_slug="${criterion_group#profile_}"
    group_slug="${group_slug//_/-}"
    result_path="$output_dir/bench_result-$group_slug-$result_key.json"

    while true; do
        tmp_dir="$(mktemp -d /dev/shm/kiddo-matrix.XXXXXX)"
        cp -- "$benchmark_exe" "$tmp_dir/benchmark"
        chmod 0755 "$tmp_dir/benchmark"

        irq_before=$(interrupt_total "${bench_cpus[@]}")
        set +e
        "${pin_command[@]}" env \
            CRITERION_HOME="$tmp_dir/criterion" \
            RAYON_NUM_THREADS="$WORKER_COUNT" \
            PARLAY_NUM_THREADS="$WORKER_COUNT" \
            KIDDO_PROFILE_QUERIES="$query_count" \
            KIDDO_LARGE_MIN_LOG2_POINTS="$height" \
            KIDDO_LARGE_MAX_LOG2_POINTS="$height" \
            "${extra_env_array[@]}" \
            "$tmp_dir/benchmark" \
            "$criterion_group" \
            --warm-up-time "$warmup" \
            --measurement-time "$measurement" \
            --sample-size "$sample_size" \
            --noplot --bench 2>&1 | tee -a "$LOG_FILE" >&2
        status=${PIPESTATUS[0]}
        set -e
        irq_after=$(interrupt_total "${bench_cpus[@]}")

        if ((status == 0 && irq_after > irq_before)); then
            echo "Cell $role $scalar 2^$height q$query_count saw $((irq_after - irq_before)) hardware IRQs on the benchmark CPUs" |
                tee -a "$LOG_FILE" >&2
            # Only a controlled run can promise an IRQ-free interval.
            ((STRICT_ISOLATION == 1)) && status=125
        fi

        if ((status == 0)); then
            (
                cd -- "$REPO_DIR"
                cargo run --quiet --manifest-path tools/criterion-export/Cargo.toml -- \
                    "$tmp_dir/criterion" "$result_path" "$criterion_group" >&2
            )
            rm -rf -- "$tmp_dir"
            printf '%s\n' "$result_path"
            return 0
        fi

        rm -rf -- "$tmp_dir"
        if ((status != 125 || attempt >= IRQ_RETRIES)); then
            return "$status"
        fi
        attempt=$((attempt + 1))
        echo "Retrying invalidated cell $role $scalar 2^$height q$query_count ($attempt/$IRQ_RETRIES)" |
            tee -a "$LOG_FILE" >&2
    done
}

# run_single_cell <criterion-suite> <extra-env-string> <scalar> <height> <role> [warmup] [measurement] [sample-size] [output-dir]
#
# Sequential-latency counterpart of run_cell: pins to ONE CPU
# (single_bench_cpus[0]) rather than the whole benchmark set, since a
# sequential query cannot be made faster by adding cores and pinning to more
# than one would only expose the measurement to interference.
run_single_cell() {
    local criterion_group=$1 extra_env=$2 scalar=$3 height=$4 role=$5
    local warmup=${6:-$CRITERION_WARMUP}
    local measurement=${7:-$CRITERION_MEASUREMENT}
    local sample_size=${8:-$CRITERION_SAMPLE_SIZE}
    local output_dir=${9:-$RUN_DIR}
    local attempt=0 status result_key result_path tmp_dir irq_before irq_after
    local -a extra_env_array
    read -ra extra_env_array <<<"$extra_env"

    result_key="$SUITE_LABEL-$PHASE-single-$role-$scalar-2p$height"
    local group_slug="${criterion_group#profile_}"
    group_slug="${group_slug//_/-}"
    result_path="$output_dir/bench_result-$group_slug-$result_key.json"

    while true; do
        tmp_dir="$(mktemp -d /dev/shm/kiddo-single.XXXXXX)"
        cp -- "$benchmark_exe" "$tmp_dir/benchmark"
        chmod 0755 "$tmp_dir/benchmark"

        irq_before=$(interrupt_total "${single_bench_cpus[@]}")
        set +e
        "${single_pin_command[@]}" env \
            CRITERION_HOME="$tmp_dir/criterion" \
            RAYON_NUM_THREADS=1 \
            PARLAY_NUM_THREADS=1 \
            KIDDO_PROFILE_QUERIES="$SINGLE_QUERY_COUNT" \
            KIDDO_LARGE_MIN_LOG2_POINTS="$height" \
            KIDDO_LARGE_MAX_LOG2_POINTS="$height" \
            "${extra_env_array[@]}" \
            "$tmp_dir/benchmark" \
            "$criterion_group" \
            --warm-up-time "$warmup" \
            --measurement-time "$measurement" \
            --sample-size "$sample_size" \
            --noplot --bench 2>&1 | tee -a "$LOG_FILE" >&2
        status=${PIPESTATUS[0]}
        set -e
        irq_after=$(interrupt_total "${single_bench_cpus[@]}")

        if ((status == 0 && irq_after > irq_before)); then
            echo "Single cell $role $scalar 2^$height saw $((irq_after - irq_before)) hardware IRQs" |
                tee -a "$LOG_FILE" >&2
            ((STRICT_ISOLATION == 1)) && status=125
        fi

        if ((status == 0)); then
            (
                cd -- "$REPO_DIR"
                cargo run --quiet --manifest-path tools/criterion-export/Cargo.toml -- \
                    "$tmp_dir/criterion" "$result_path" "$criterion_group" >&2
            )
            rm -rf -- "$tmp_dir"
            printf '%s\n' "$result_path"
            return 0
        fi

        rm -rf -- "$tmp_dir"
        if ((status != 125 || attempt >= IRQ_RETRIES)); then
            return "$status"
        fi
        attempt=$((attempt + 1))
        echo "Retrying invalidated single cell $role $scalar 2^$height ($attempt/$IRQ_RETRIES)" |
            tee -a "$LOG_FILE" >&2
    done
}

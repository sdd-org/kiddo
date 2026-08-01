#!/usr/bin/env bash
set -euo pipefail

if (( $# != 3 )); then
    echo "usage: $0 BENCHMARK_EXECUTABLE PERF_OUTPUT EVENT_LIST" >&2
    exit 2
fi

benchmark=$1
perf_output=$2
events=$3
run_dir=$(mktemp -d /dev/shm/kiddo-query-pool-perf.XXXXXX)
target_pid=
perf_pid=

cleanup() {
    if [[ -n "$target_pid" ]] && kill -0 "$target_pid" 2>/dev/null; then
        kill -CONT "$target_pid" 2>/dev/null || true
        kill -TERM "$target_pid" 2>/dev/null || true
    fi
    if [[ -n "$perf_pid" ]] && kill -0 "$perf_pid" 2>/dev/null; then
        kill -INT "$perf_pid" 2>/dev/null || true
    fi
    rm -rf -- "$run_dir"
}
trap cleanup EXIT INT TERM

cp -- "$benchmark" "$run_dir/benchmark"
chmod 0755 "$run_dir/benchmark"

KIDDO_PERF_STOP=1 "$run_dir/benchmark" >"$run_dir/run.out" 2>"$run_dir/run.err" &
target_pid=$!

state=
for _ in {1..6000}; do
    if [[ ! -r "/proc/$target_pid/status" ]]; then
        break
    fi
    state=$(awk '$1 == "State:" { print $2 }' "/proc/$target_pid/status")
    [[ "$state" == T ]] && break
    sleep 0.01
done

if [[ "$state" != T ]]; then
    cat "$run_dir/run.err" >&2 || true
    cat "$run_dir/run.out" >&2 || true
    echo "benchmark did not reach its pre-measurement SIGSTOP" >&2
    wait "$target_pid" || true
    exit 1
fi

mkdir -p -- "$(dirname -- "$perf_output")"
perf stat --no-big-num -x, -o "$run_dir/perf.csv" -e "$events" -p "$target_pid" &
perf_pid=$!

# Give perf time to open and enable the events while the target remains stopped.
sleep 0.2
if ! kill -0 "$perf_pid" 2>/dev/null; then
    wait "$perf_pid" || true
    cat "$run_dir/perf.csv" >&2 || true
    echo "perf failed before the measurement interval began" >&2
    exit 1
fi

kill -CONT "$target_pid"
set +e
wait "$target_pid"
target_status=$?
set -e
target_pid=

if kill -0 "$perf_pid" 2>/dev/null; then
    kill -INT "$perf_pid"
fi
set +e
wait "$perf_pid"
perf_status=$?
set -e
perf_pid=

cat "$run_dir/run.err" >&2
cat "$run_dir/run.out"
cp -- "$run_dir/perf.csv" "$perf_output"
cp -- "$run_dir/run.out" "${perf_output%.csv}.run.txt"

if (( target_status != 0 )); then
    echo "benchmark exited with status $target_status" >&2
    exit "$target_status"
fi
if (( perf_status != 0 && perf_status != 130 )); then
    echo "perf exited with status $perf_status" >&2
    exit "$perf_status"
fi

"$(dirname -- "${BASH_SOURCE[0]}")/query_pool_perf_events.sh" \
    --validate "$perf_output"

echo "perf counters: $perf_output"
echo "run metadata: ${perf_output%.csv}.run.txt"

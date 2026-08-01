#!/usr/bin/env bash
set -euo pipefail

events_for_set() {
    case "$1" in
        core)
            # Six events, verified to schedule for 100% of the interval on the
            # target Zen 5 PMU. The final event is the native Zen back-end-stall
            # event; generic stalled-cycles-backend is unsupported on this CPU.
            echo 'cycles,instructions,branches,branch-misses,stalled-cycles-frontend,de_no_dispatch_per_slot.backend_stalls'
            ;;
        cache)
            # cache-misses is the same event (0x64/0x09) as
            # l2_cache_req_stat.ic_dc_miss_in_l2 on this Zen PMU, so do not
            # spend a second counter measuring the alias.
            echo 'L1-dcache-loads,L1-dcache-load-misses,l2_request_g1.all_dc,l2_cache_req_stat.ic_dc_miss_in_l2,cache-references'
            ;;
        tlb)
            # Use the unambiguous native events. The generic dTLB-load-misses
            # event reported exactly the same counts as all_l2_miss in the
            # end-to-end smoke, so it adds no independent evidence here.
            echo 'ls_l1_d_tlb_miss.all,ls_l1_d_tlb_miss.all_l2_miss'
            ;;
        *)
            echo "EVENT_SET must be core, cache, or tlb" >&2
            return 2
            ;;
    esac
}

validate_csv() {
    local csv=$1

    if [[ ! -s "$csv" ]]; then
        echo "perf produced no counter output: $csv" >&2
        return 1
    fi
    if grep -Eq '<not (counted|supported)>' "$csv"; then
        echo "perf could not count every requested event:" >&2
        cat "$csv" >&2
        return 1
    fi

    # perf's CSV percentage column is time_running/time_enabled. Anything
    # materially below 100% means events were multiplexed; reject it rather
    # than treating scaled/aliased measurements as publication evidence.
    if ! awk -F, '
        /^[[:space:]]*#/ || /^[[:space:]]*$/ { next }
        {
            rows++
            if ($5 == "" || ($5 + 0) < 99.99) {
                printf "multiplexed or malformed perf row: %s\n", $0 > "/dev/stderr"
                bad = 1
            }
        }
        END { exit (rows == 0 || bad) }
    ' "$csv"; then
        echo "perf counters were not scheduled for the complete interval" >&2
        return 1
    fi
}

case "${1:-}" in
    --check)
        [[ $# == 2 ]] || { echo "usage: $0 --check EVENT_SET" >&2; exit 2; }
        events=$(events_for_set "$2")
        output=$(mktemp /dev/shm/kiddo-perf-event-check.XXXXXX)
        trap 'rm -f -- "$output"' EXIT
        perf stat --no-big-num -x, -o "$output" -e "$events" true
        validate_csv "$output"
        ;;
    --validate)
        [[ $# == 2 ]] || { echo "usage: $0 --validate PERF_CSV" >&2; exit 2; }
        validate_csv "$2"
        ;;
    '')
        echo "usage: $0 EVENT_SET | --check EVENT_SET | --validate PERF_CSV" >&2
        exit 2
        ;;
    *)
        [[ $# == 1 ]] || { echo "usage: $0 EVENT_SET" >&2; exit 2; }
        events_for_set "$1"
        ;;
esac

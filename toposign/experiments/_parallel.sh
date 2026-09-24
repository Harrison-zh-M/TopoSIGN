#!/usr/bin/env bash
# Parallel job helpers sourced by all table scripts.

_parallel_fail=0

_cleanup() {
    echo "[parallel] Caught signal — killing child jobs..." >&2
    jobs -rp | xargs -r kill -TERM 2>/dev/null || true
    exit 1
}
trap '_cleanup' INT TERM

normalize_jobs() {
    local jobs="${1:-1}"
    if ! [[ "$jobs" =~ ^[0-9]+$ ]] || [ "$jobs" -lt 1 ]; then
        jobs=1
    fi
    printf '%s\n' "$jobs"
}

wait_for_slot() {
    local max_jobs
    max_jobs="$(normalize_jobs "$1")"
    while [ "$(jobs -rp | wc -l)" -ge "$max_jobs" ]; do
        if ! wait -n; then
            _parallel_fail=1
        fi
    done
}

run_limited() {
    local max_jobs="$1"
    local label="$2"
    shift 2
    wait_for_slot "$max_jobs"
    echo "$label"
    "$@" &
}

wait_all_jobs() {
    while [ "$(jobs -rp | wc -l)" -gt 0 ]; do
        if ! wait -n; then
            _parallel_fail=1
        fi
    done
    return "$_parallel_fail"
}

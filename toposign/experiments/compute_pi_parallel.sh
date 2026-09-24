#!/usr/bin/env bash
# Compute PI tensors in parallel for a set of datasets.
# Uses GNU parallel when available; otherwise uses Python --workers flag.
#
# Usage: bash toposign/experiments/compute_pi_parallel.sh [TARGET] [OPTIONS]
#   TARGET: all | settingN | rainfall | sp1500  (default: all)
#   --tda_type signed|positive  (default: signed)
#   --force                     Recompute even if tensor already exists
#   --jobs N                    Parallel jobs (default: number of datasets)
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON:-python}"
PI_JOBS="${PI_JOBS:-0}"
TARGET="${1:-all}"
TDA_TYPE="signed"
FORCE_ARGS=()
PIXEL_ARGS=()

if [ "$#" -gt 0 ]; then shift; fi
while [ "$#" -gt 0 ]; do
    case "$1" in
        --tda_type)         TDA_TYPE="$2"; shift ;;
        --positive)         TDA_TYPE="positive" ;;
        --signed)           TDA_TYPE="signed" ;;
        --force)            FORCE_ARGS=(--force) ;;
        --pixel_size|--pixel-size) PIXEL_ARGS=(--pixel_size "$2"); shift ;;
        --jobs)             PI_JOBS="$2"; shift ;;
        --debug|-D)         ;;  # no-op for PI computation
        *)                  ;;
    esac
    shift
done

if ! [[ "$PI_JOBS" =~ ^[0-9]+$ ]]; then PI_JOBS=0; fi

DATASETS=()
if [ "$TARGET" = "all" ]; then
    for setting in setting1 setting2 setting3 setting4; do
        for run_id in 0 1 2 3 4; do
            DATASETS+=("${setting}_run${run_id}")
        done
    done
    DATASETS+=(rainfall sp1500)
elif [[ "$TARGET" =~ ^setting[0-9]+$ ]]; then
    for run_id in 0 1 2 3 4; do
        DATASETS+=("${TARGET}_run${run_id}")
    done
else
    DATASETS=("$TARGET")
fi

[ "$PI_JOBS" -eq 0 ] && PI_JOBS="${#DATASETS[@]}"

PARALLEL_BIN="${PARALLEL_BIN:-parallel}"
if command -v "$PARALLEL_BIN" &>/dev/null; then
    echo "[compute_pi_parallel] GNU parallel: jobs=${PI_JOBS}  tda_type=${TDA_TYPE}"
    "$PARALLEL_BIN" -j"$PI_JOBS" \
        "$PYTHON_BIN" -m toposign.data_generation.compute_pi \
            --dataset {} --tda_type "$TDA_TYPE" \
            "${PIXEL_ARGS[@]}" "${FORCE_ARGS[@]}" \
        ::: "${DATASETS[@]}"
else
    echo "[compute_pi_parallel] Python workers=${PI_JOBS}  tda_type=${TDA_TYPE}"
    "$PYTHON_BIN" -m toposign.data_generation.compute_pi \
        --dataset "$TARGET" --workers "$PI_JOBS" --tda_type "$TDA_TYPE" \
        "${PIXEL_ARGS[@]}" "${FORCE_ARGS[@]}"
fi

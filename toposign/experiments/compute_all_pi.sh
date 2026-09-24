#!/usr/bin/env bash
# Compute signed and positive-edge persistence-image (PI) tensors for all datasets.
# Uses GNU parallel if available, otherwise falls back to Python multiprocessing.
#
# Usage: bash toposign/experiments/compute_all_pi.sh [--force] [--python PYTHON] [--debug]
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON:-python}"
PI_JOBS="${PI_JOBS:-0}"
FORCE_ARGS=()
DEBUG_ARGS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --python)        PYTHON_BIN="$2"; shift ;;
        --force)         FORCE_ARGS=(--force) ;;
        --debug|-D)      DEBUG_ARGS=(--debug) ;;
        --jobs)          PI_JOBS="$2"; shift ;;
        *)               ;;
    esac
    shift
done

echo "[compute_all_pi] Computing signed PI tensors..."
PI_JOBS="$PI_JOBS" PYTHON="$PYTHON_BIN" bash toposign/experiments/compute_pi_parallel.sh all \
    --tda_type signed "${FORCE_ARGS[@]}" "${DEBUG_ARGS[@]+"${DEBUG_ARGS[@]}"}"

echo "[compute_all_pi] Computing positive-edge PI tensors..."
PI_JOBS="$PI_JOBS" PYTHON="$PYTHON_BIN" bash toposign/experiments/compute_pi_parallel.sh all \
    --tda_type positive "${FORCE_ARGS[@]}" "${DEBUG_ARGS[@]+"${DEBUG_ARGS[@]}"}"

echo "[compute_all_pi] Done."

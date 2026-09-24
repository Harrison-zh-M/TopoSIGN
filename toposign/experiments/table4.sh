#!/usr/bin/env bash
# Table 4: Pre-training task ablation (SP+DP, SP+3C, SP+4C, SP+5C).
# The SP-only row is cross-loaded from Table 1 by aggregate_results.py.
# Datasets: SDSBM-1..4 (5 runs each)
set -euo pipefail
cd "$(dirname "$0")/../.."
source toposign/experiments/_parallel.sh

JOBS="${TABLE_JOBS:-5}"
PYTHON_BIN="${PYTHON:-python}"
DEBUG_ARGS=()
EXTRA_ARGS=()
HPARAM_ARGS=(--hparam_search)

while [ "$#" -gt 0 ]; do
    case "$1" in
        --debug|-D)         DEBUG_ARGS=(-D) ;;
        --no-hparam-search) HPARAM_ARGS=() ;;
        --jobs)             JOBS="$2"; shift ;;
        *)                  EXTRA_ARGS+=("$1") ;;
    esac
    shift
done
JOBS="$(normalize_jobs "$JOBS")"

SEEDS=(0 10 20 30 40)

echo "[table4] Parallel jobs: ${JOBS}"

_run_table4() {
    local backbone="$1" label="$2"
    for setting in setting1 setting2 setting3 setting4; do
        for run_id in 0 1 2 3 4; do
            dataset="${setting}_run${run_id}"
            seed="${SEEDS[$run_id]}"
            run_limited "$JOBS" "[table4/${label}] ${dataset}" \
                "$PYTHON_BIN" -m toposign.run_table4 \
                --dataset "$dataset" --seeds "$seed" \
                --backbone "$backbone" \
                --tda_type signed -sd_input_features -weighted_input_features \
                "${DEBUG_ARGS[@]}" "${HPARAM_ARGS[@]}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
        done
    done
}

_run_table4 "TopoSIGN" "msgnn"   # TopoMSGNN_pretrain_SP+*
_run_table4 "SSSNET"   "sssnet"  # TopoSSSNET_pretrain_SP+*

wait_all_jobs
echo "[table4] Done."

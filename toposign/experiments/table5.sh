#!/usr/bin/env bash
# Table 5: Prompt method comparison (GPPT, Gprompt, GPF, All-in-one).
# The GPPT rows are cross-loaded from Table 1 by aggregate_results.py.
# Datasets: SDSBM-1..4 (5 runs each) + Rainfall (5 runs) + SP1500 (5 runs)
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

PROMPT_METHODS=("Gprompt" "GPF" "All-in-one")  # GPPT cross-loaded from Table 1
SEEDS=(0 10 20 30 40)

echo "[table5] Parallel jobs: ${JOBS}"

_run_table5() {
    local backbone="$1" label="$2"
    for setting in setting1 setting2 setting3 setting4; do
        for run_id in 0 1 2 3 4; do
            dataset="${setting}_run${run_id}"
            seed="${SEEDS[$run_id]}"
            for pm in "${PROMPT_METHODS[@]}"; do
                run_limited "$JOBS" "[table5/${label}] ${dataset}  pm=${pm}" \
                    "$PYTHON_BIN" -m toposign.run_table5 \
                    --dataset "$dataset" --prompt_method "$pm" --seeds "$seed" \
                    --backbone "$backbone" \
                    --tda_type signed -sd_input_features -weighted_input_features \
                    "${DEBUG_ARGS[@]}" "${HPARAM_ARGS[@]}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
            done
        done
    done
    for base_dataset in rainfall sp1500; do
        for run_id in 0 1 2 3 4; do
            dataset="${base_dataset}_run${run_id}"
            seed="${SEEDS[$run_id]}"
            for pm in "${PROMPT_METHODS[@]}"; do
                run_limited "$JOBS" "[table5/${label}] ${dataset}  pm=${pm}" \
                    "$PYTHON_BIN" -m toposign.run_table5 \
                    --dataset "$dataset" --prompt_method "$pm" --seeds "$seed" \
                    --backbone "$backbone" \
                    --tda_type signed -sd_input_features -weighted_input_features \
                    "${DEBUG_ARGS[@]}" "${HPARAM_ARGS[@]}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
            done
        done
    done
}

_run_table5 "TopoSIGN" "msgnn"   # TopoMSGNN+Gprompt/GPF/All-in-one
_run_table5 "SSSNET"   "sssnet"  # TopoSSSNET+Gprompt/GPF/All-in-one

wait_all_jobs
echo "[table5] Done."

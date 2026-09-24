#!/usr/bin/env bash
# Table 6: Backbone ablation (SigMaNet, DSGC on SDSBM-1..3).
# TopoMSGNN and TopoSSSNET rows are cross-loaded from Table 1 by aggregate_results.py.
# Datasets: SDSBM-1..3 (5 runs each)
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

BACKBONE_METHODS=("SigMaNet" "DSGC")  # TopoMSGNN/TopoSSSNET cross-loaded from Table 1
SEEDS=(0 10 20 30 40)

echo "[table6] Parallel jobs: ${JOBS}"

for setting in setting1 setting2 setting3; do
    for run_id in 0 1 2 3 4; do
        dataset="${setting}_run${run_id}"
        seed="${SEEDS[$run_id]}"
        for backbone in "${BACKBONE_METHODS[@]}"; do
            run_limited "$JOBS" "[table6] ${dataset}  backbone=${backbone}" \
                "$PYTHON_BIN" -m toposign.run_table6 \
                --dataset "$dataset" --backbone "$backbone" --seeds "$seed" \
                --tda_type signed -sd_input_features -weighted_input_features \
                "${DEBUG_ARGS[@]}" "${HPARAM_ARGS[@]}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
        done
    done
done

wait_all_jobs
echo "[table6] Done."

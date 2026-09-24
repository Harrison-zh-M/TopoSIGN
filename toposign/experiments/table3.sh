#!/usr/bin/env bash
# Table 3: Filtration ablation for MSGNN and SSSNET backbones.
# Variants: SGE_only, Topo_only, SGE+posTopo, TopoSIGN (for each backbone)
# Datasets: SDSBM-1..4 (5 runs each) + Rainfall (5 runs) + SP1500 (5 runs)
set -euo pipefail
cd "$(dirname "$0")/../.."
source toposign/experiments/_parallel.sh

JOBS="${TABLE_JOBS:-5}"
PYTHON_BIN="${PYTHON:-python}"
DEBUG_ARGS=()
EXTRA_ARGS=()
HPARAM_ARGS=(--hparam_search)
VARIANT_MODE="all"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --debug|-D)         DEBUG_ARGS=(-D) ;;
        --no-hparam-search) HPARAM_ARGS=() ;;
        --skip-positive)    VARIANT_MODE="nonpositive" ;;
        --only-positive)    VARIANT_MODE="positive" ;;
        --jobs)             JOBS="$2"; shift ;;
        *)                  EXTRA_ARGS+=("$1") ;;
    esac
    shift
done
JOBS="$(normalize_jobs "$JOBS")"

SEEDS=(0 10 20 30 40)

echo "[table3] Parallel jobs: ${JOBS}"

_run_table3() {
    local backbone="$1" label="$2"
    for setting in setting1 setting2 setting3 setting4; do
        for run_id in 0 1 2 3 4; do
            dataset="${setting}_run${run_id}"
            seed="${SEEDS[$run_id]}"
            run_limited "$JOBS" "[table3/${label}] ${dataset}" \
                "$PYTHON_BIN" -m toposign.run_table3 \
                --dataset "$dataset" --seeds "$seed" \
                --backbone "$backbone" --variant "$VARIANT_MODE" \
                -sd_input_features -weighted_input_features \
                "${DEBUG_ARGS[@]}" "${HPARAM_ARGS[@]}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
        done
    done
    for base_dataset in rainfall sp1500; do
        for run_id in 0 1 2 3 4; do
            dataset="${base_dataset}_run${run_id}"
            seed="${SEEDS[$run_id]}"
            run_limited "$JOBS" "[table3/${label}] ${dataset}" \
                "$PYTHON_BIN" -m toposign.run_table3 \
                --dataset "$dataset" --seeds "$seed" \
                --backbone "$backbone" --variant "$VARIANT_MODE" \
                -sd_input_features -weighted_input_features \
                "${DEBUG_ARGS[@]}" "${HPARAM_ARGS[@]}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
        done
    done
}

_run_table3 "TopoSIGN" "msgnn"   # MSGNN_only, Topo_only, MSGNN+posTopo, TopoMSGNN
_run_table3 "SSSNET"   "sssnet"  # SSSNET_only, SSSNET+posTopo, TopoSSSNET

wait_all_jobs
echo "[table3] Done."

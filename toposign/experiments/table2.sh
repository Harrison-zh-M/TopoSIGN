#!/usr/bin/env bash
# Table 2: Semi-supervised node clustering baselines + +Topo variants.
# Methods: SSSNET, SigMaNet, MSGNN, DSGC, DSGC+Topo, SigMaNet+Topo, MSGNN+Topo, SSSNET+Topo
# Datasets: SDSBM-1..4 (5 runs each) + Rainfall (5 runs) + SP1500 (5 runs)
set -euo pipefail
cd "$(dirname "$0")/../.."
source toposign/experiments/_parallel.sh

JOBS="${TABLE_JOBS:-5}"
PYTHON_BIN="${PYTHON:-python}"
DEBUG_ARGS=()
EXTRA_ARGS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --debug|-D) DEBUG_ARGS=(-D) ;;
        --jobs)     JOBS="$2"; shift ;;
        *)          EXTRA_ARGS+=("$1") ;;
    esac
    shift
done
JOBS="$(normalize_jobs "$JOBS")"

METHODS=("SSSNET" "SigMaNet" "MSGNN" "DSGC" "DSGC+Topo" "SigMaNet+Topo" "MSGNN+Topo" "SSSNET+Topo")
SEEDS=(0 10 20 30 40)

echo "[table2] Parallel jobs: ${JOBS}"

for setting in setting1 setting2 setting3 setting4; do
    for run_id in 0 1 2 3 4; do
        dataset="${setting}_run${run_id}"
        seed="${SEEDS[$run_id]}"
        for method in "${METHODS[@]}"; do
            run_limited "$JOBS" "[table2] ${dataset}  method=${method}" \
                "$PYTHON_BIN" -m toposign.run_table2 \
                --dataset "$dataset" --method "$method" --seeds "$seed" \
                --pixel_size 1.0 -sd_input_features -weighted_input_features \
                "${DEBUG_ARGS[@]}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
        done
    done
done

for base_dataset in rainfall sp1500; do
    for run_id in 0 1 2 3 4; do
        dataset="${base_dataset}_run${run_id}"
        seed="${SEEDS[$run_id]}"
        for method in "${METHODS[@]}"; do
            run_limited "$JOBS" "[table2] ${dataset}  method=${method}" \
                "$PYTHON_BIN" -m toposign.run_table2 \
                --dataset "$dataset" --method "$method" --seeds "$seed" \
                --pixel_size 1.0 -sd_input_features -weighted_input_features \
                "${DEBUG_ARGS[@]}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
        done
    done
done

wait_all_jobs
echo "[table2] Done."

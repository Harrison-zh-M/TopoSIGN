#!/usr/bin/env bash
# Full TopoSIGN reproduction pipeline.
#
# Steps:
#   1. Generate SDSBM synthetic graphs (settings 1–4, 5 runs each)
#   2. Compute signed and positive-edge persistence-image (PI) tensors
#   3. Run all experiment tables (1–6)
#   4. Aggregate results and write LaTeX tables
#
# Usage:
#   bash toposign/experiments/run_all.sh [OPTIONS]
#
# Options:
#   --jobs N            Parallel jobs per table script (default: 5)
#   --skip-data-gen     Skip SDSBM generation (data already exists)
#   --skip-pi           Skip PI computation (tensors already exist)
#   --debug             Smoke-test mode: 2 epochs, 2 seeds
#   --no-hparam-search  Skip hyperparameter search
#
# Environment variables:
#   PYTHON              Python binary (default: python)
#   TABLE_JOBS          Per-table parallel jobs

set -euo pipefail
cd "$(dirname "$0")/../.."
source toposign/experiments/_parallel.sh

PYTHON_BIN="${PYTHON:-python}"
JOBS="${TABLE_JOBS:-5}"
SKIP_DATA_GEN=0
SKIP_PI=0
DEBUG_ARGS=()
HPARAM_ARGS=(--hparam_search)
EXTRA_ARGS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --jobs)             JOBS="$2"; shift ;;
        --skip-data-gen)    SKIP_DATA_GEN=1 ;;
        --skip-pi)          SKIP_PI=1 ;;
        --debug|-D)         DEBUG_ARGS=(--debug) ;;
        --no-hparam-search) HPARAM_ARGS=(--no-hparam-search) ;;
        *)                  EXTRA_ARGS+=("$1") ;;
    esac
    shift
done

JOBS="$(normalize_jobs "$JOBS")"
export TABLE_JOBS="${JOBS}"
export PYTHON="${PYTHON_BIN}"

echo "========================================================"
echo " TopoSIGN: Full Reproduction Pipeline"
echo "  Python:  ${PYTHON_BIN}"
echo "  Jobs:    ${JOBS}"
[ "${#DEBUG_ARGS[@]}" -gt 0 ] && echo "  Mode:    DEBUG (smoke test)"
echo "========================================================"

_tbl() {
    local label="$1"; shift
    echo ""
    echo "[Step 3] ${label}"
    bash "$@" --jobs "${JOBS}" \
        "${DEBUG_ARGS[@]}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
}

# ── Step 1: Generate SDSBM datasets ──────────────────────────────────────────
if [ "$SKIP_DATA_GEN" -eq 0 ]; then
    echo ""
    echo "[Step 1/4] Generating SDSBM synthetic datasets..."
    "$PYTHON_BIN" -m toposign.data_generation.generate_sdsbm
    echo "[Step 1/4] Done."
else
    echo "[Step 1/4] Skipped (--skip-data-gen)."
fi

# ── Step 2: Compute PI tensors ────────────────────────────────────────────────
if [ "$SKIP_PI" -eq 0 ]; then
    echo ""
    echo "[Step 2/4] Computing persistence-image tensors (signed + positive)..."
    bash toposign/experiments/compute_all_pi.sh --jobs "${JOBS}"
    echo "[Step 2/4] Done."
else
    echo "[Step 2/4] Skipped (--skip-pi)."
fi

# ── Step 3: Run experiment tables ─────────────────────────────────────────────
echo ""
echo "[Step 3/4] Running experiment tables..."

_tbl "Table 1 (GFM pre-training + prompt learning)" \
    toposign/experiments/table1.sh "${HPARAM_ARGS[@]}"

_tbl "Table 2 (semi-supervised node clustering baselines)" \
    toposign/experiments/table2.sh

_tbl "Table 3 (filtration ablation)" \
    toposign/experiments/table3.sh "${HPARAM_ARGS[@]}"

_tbl "Table 4 (pre-training task ablation)" \
    toposign/experiments/table4.sh "${HPARAM_ARGS[@]}"

_tbl "Table 5 (prompt method comparison)" \
    toposign/experiments/table5.sh "${HPARAM_ARGS[@]}"

_tbl "Table 6 (backbone ablation)" \
    toposign/experiments/table6.sh "${HPARAM_ARGS[@]}"

echo ""
echo "[Step 3/4] All tables complete."

# ── Step 4: Aggregate and write LaTeX ─────────────────────────────────────────
echo ""
echo "[Step 4/4] Aggregating results and generating LaTeX tables..."
bash toposign/experiments/aggregate_results.sh
echo "[Step 4/4] Done."

echo ""
echo "========================================================"
echo " Pipeline complete.  LaTeX tables: results/latex/"
echo "========================================================"

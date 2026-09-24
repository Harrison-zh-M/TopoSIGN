#!/usr/bin/env bash
# Aggregate all experiment results and write LaTeX tables to results/latex/.
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON:-python}"
EXTRA_ARGS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --results_dir) EXTRA_ARGS+=(--results_dir "$2"); shift ;;
        --latex_dir)   EXTRA_ARGS+=(--latex_dir "$2"); shift ;;
        --latex_only)  EXTRA_ARGS+=(--latex_only) ;;
        --no_latex)    EXTRA_ARGS+=(--no_latex) ;;
        *)             EXTRA_ARGS+=("$1") ;;
    esac
    shift
done

echo "[aggregate] Aggregating results and generating LaTeX tables..."
"$PYTHON_BIN" -m toposign.aggregate_results "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
echo "[aggregate] Done. LaTeX files written to results/latex/"

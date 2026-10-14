#!/usr/bin/env bash
# v3 (Cell Systems) driver for Figure 1. No data stage: plot_data is the frozen copy of the accepted 20261007 tables
# (delivery_R_20261009_v3/figure1/plot_data); prepare_data.py is kept for provenance only and must not be re-run.
set -euo pipefail
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
creed_rscript=${CREED_RSCRIPT:-/public/home/mengxl/.conda/envs/creed-r/bin/Rscript}
[[ -x "$creed_rscript" ]] || creed_rscript=Rscript
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
case "${1:-help}" in
  panels)   shift; "$creed_rscript" "$script_dir/render_panels.R" "$@" ;;     # run.sh panels [A B ...]
  assemble) "$creed_rscript" "$script_dir/assemble.R" ;;
  blank)    "$creed_rscript" "$script_dir/assemble.R" blank ;;
  *) printf 'Usage: bash r/figure1/run.sh {panels [A B ...]|assemble|blank}\nInspect the panels before running assemble.\n' ;;
esac

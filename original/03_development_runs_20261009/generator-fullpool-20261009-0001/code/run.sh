#!/bin/bash
# generator-fullpool-20261009-0001: identity-safe M+RNA pool extraction (no training, no outer scores)
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
RUN=/public/home/mengxl/creed-research-runs/generator-fullpool-20261009-0001
cd "$RUN"
/public/home/mengxl/.conda/envs/CellFM/bin/python -I code/extract_fullpool.py "$RUN"

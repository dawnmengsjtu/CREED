#!/bin/bash
# Launch: integrity check -> new_cell main -> tissue main -> pre-registered analysis. 8 threads, one heavy process at a time.
set -e
RUN=/public/home/mengxl/creed-research-runs/crosslayer-matrix-20261009-0001
PY=/public/home/mengxl/.conda/envs/CellFM/bin/python
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd $RUN
sha256sum -c SHA256SUMS
date -Is
$PY code/crosslayer.py new_cell main $RUN/output/new_cell
$PY code/crosslayer.py tissue main $RUN/output/tissue
$PY code/analyze.py $RUN/output
date -Is
echo RUN_COMPLETE

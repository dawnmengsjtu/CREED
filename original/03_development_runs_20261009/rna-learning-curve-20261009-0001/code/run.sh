#!/bin/bash
# Launch: new_cell -> tissue -> pre-registered analysis. 8 threads, one heavy process at a time.
set -e
RUN=/public/home/mengxl/creed-research-runs/rna-learning-curve-20261009-0001
PY=/public/home/mengxl/.conda/envs/CellFM/bin/python
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd $RUN
sha256sum -c SHA256SUMS
$PY code/rna_lc.py new_cell $RUN/output/new_cell
$PY code/rna_lc.py tissue $RUN/output/tissue
$PY code/analyze.py $RUN/output
echo RUN_COMPLETE

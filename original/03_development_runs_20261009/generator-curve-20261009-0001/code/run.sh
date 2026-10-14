#!/bin/bash
# Launch: integrity check -> new_cell main -> tissue main -> pre-registered analysis -> output hashes. 8 threads, one process at a time.
set -e
RUN=/public/home/mengxl/creed-research-runs/generator-curve-20261009-0001
PY=/public/home/mengxl/.conda/envs/CellFM/bin/python
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd $RUN
sha256sum -c SHA256SUMS
date -Is
$PY code/gencurve.py new_cell main $RUN/output/new_cell
$PY code/gencurve.py tissue main $RUN/output/tissue
$PY code/analyze_curve.py $RUN/output
(cd $RUN && find output -type f | sort | xargs sha256sum > OUTPUT_SHA256SUMS)
date -Is
echo RUN_COMPLETE

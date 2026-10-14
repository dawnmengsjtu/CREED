#!/bin/bash
# Stage A step 2 (CRISPR, PLAN.md section 5): 10 (scenario, fold) processes -> analysis.
set -e
RUN=/public/home/mengxl/creed-research-runs/settle-stageA-20261011-0001
PY=/public/home/mengxl/.conda/envs/CellFM/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd $RUN
sha256sum -c SHA256SUMS_CRISPR
date -Is
mkdir -p output_crispr/new_cell output_crispr/tissue logs_crispr
for s in new_cell tissue; do for f in 0 1 2 3 4; do
  $PY code/stageA_crispr.py $s main output_crispr/$s/fold-$f $f > logs_crispr/$s-fold-$f.log 2>&1 &
done; done
wait
$PY code/analyze_crispr.py output_crispr > logs_crispr/analyze.log 2>&1
date -Is
echo RUN_COMPLETE

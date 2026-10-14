#!/bin/bash
# Stage A main: integrity check -> 10 (scenario, fold) processes in parallel -> pre-registered analysis.
set -e
RUN=/public/home/mengxl/creed-research-runs/settle-stageA-20261011-0001
PY=/public/home/mengxl/.conda/envs/CellFM/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd $RUN
sha256sum -c SHA256SUMS
date -Is
mkdir -p output/new_cell output/tissue logs
for s in new_cell tissue; do for f in 0 1 2 3 4; do
  $PY code/stageA.py $s main output/$s/fold-$f $f > logs/$s-fold-$f.log 2>&1 &
done; done
wait
$PY code/analyze.py output > logs/analyze.log 2>&1
date -Is
echo RUN_COMPLETE

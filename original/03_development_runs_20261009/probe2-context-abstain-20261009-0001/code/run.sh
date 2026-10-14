#!/bin/bash
# Main launch: integrity check -> part A (10 single-thread fold jobs, max 8 concurrent) + part B (2 jobs) -> analysis.
set -e
RUN=/public/home/mengxl/creed-research-runs/probe2-context-abstain-20261009-0001
PY=/public/home/mengxl/.conda/envs/CellFM/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $RUN
sha256sum -c SHA256SUMS
date -Is
mkdir -p output/A/new_cell output/A/tissue output/B logs
{
  for f in 0 1 2 3 4; do echo "A new_cell $f"; done
  for f in 0 1 2 3 4; do echo "A tissue $f"; done
  echo "B new_cell"; echo "B tissue"
} > jobs.txt
job() {
  if [ "$1" = A ]; then $PY code/probe2.py $2 fold $RUN/output/A/$2 $3 > logs/A_$2_fold$3.log 2>&1
  else $PY code/abstain.py $2 $RUN/output/B > logs/B_$2.log 2>&1; fi
  echo "done $*  $(date -Is)"
}
export -f job; export PY RUN
xargs -P 8 -L 1 bash -c 'job $0 $1 $2' < jobs.txt
$PY code/analyze.py $RUN/output > logs/analyze.log 2>&1
cat logs/analyze.log
date -Is
echo RUN_COMPLETE

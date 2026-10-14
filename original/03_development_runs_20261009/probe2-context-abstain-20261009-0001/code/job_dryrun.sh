#!/bin/bash
# Engineering dryrun: fold-0 inner-0 bank only, both scenarios in parallel (1 thread each). No outer score is computed.
set -e
RUN=/public/home/mengxl/creed-research-runs/probe2-context-abstain-20261009-0001
PY=/public/home/mengxl/.conda/envs/CellFM/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $RUN; mkdir -p dryrun logs
$PY code/probe2.py new_cell dryrun $RUN/dryrun > logs/dryrun_new_cell.log 2>&1 &
$PY code/probe2.py tissue dryrun $RUN/dryrun > logs/dryrun_tissue.log 2>&1 &
wait
tail -n 2 logs/dryrun_new_cell.log logs/dryrun_tissue.log

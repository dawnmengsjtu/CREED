#!/bin/bash
# Engineering dryrun: fold-0 inner-0 bank only, both scenarios in parallel (4 threads each). No outer score is computed.
set -e
RUN=/public/home/mengxl/creed-research-runs/rna-panel-20261009-0001
PY=/public/home/mengxl/.conda/envs/CellFM/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd $RUN; mkdir -p dryrun logs
date -Is
$PY code/rna_panel.py new_cell dryrun $RUN/dryrun/new_cell > logs/dryrun_new_cell.log 2>&1 &
$PY code/rna_panel.py tissue dryrun $RUN/dryrun/tissue > logs/dryrun_tissue.log 2>&1 &
wait
date -Is
tail -n 1 logs/dryrun_new_cell.log logs/dryrun_tissue.log

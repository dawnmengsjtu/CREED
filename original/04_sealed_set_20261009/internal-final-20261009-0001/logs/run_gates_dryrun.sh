#!/bin/bash
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /public/home/mengxl/creed-research-runs/internal-final-20261009-0001
PY=/public/home/mengxl/.conda/envs/CellFM/bin/python
$PY code/oneshot.py gates > logs/gates_v2.log 2>&1
echo "GATES_EXIT $?" >> logs/gates_v2.log
$PY code/oneshot.py dryrun > logs/dryrun_attempt1.log 2>&1
echo "DRYRUN_EXIT $?" >> logs/dryrun_attempt1.log

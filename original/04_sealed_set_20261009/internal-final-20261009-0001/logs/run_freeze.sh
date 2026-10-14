#!/bin/bash
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /public/home/mengxl/creed-research-runs/internal-final-20261009-0001
/public/home/mengxl/.conda/envs/CellFM/bin/python code/oneshot.py freeze > logs/freeze.log 2>&1
echo "FREEZE_EXIT $?" >> logs/freeze.log

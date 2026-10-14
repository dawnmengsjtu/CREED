#!/bin/bash
# CRISPR gene-effect endpoint run. usage: bash run.sh RUN_DIR STAGE
set -euo pipefail
RUN="$1"; STAGE="$2"
PY=/public/home/mengxl/.conda/envs/CellFM/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
mkdir -p "$RUN/logs"
if [ "$STAGE" = "dryrun" ]; then
  export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
  $PY "$RUN/code/crispr_endpoint.py" new_cell dryrun "$RUN/dryrun/new_cell" 2>&1 | tee "$RUN/logs/dryrun_new_cell.log"
  $PY "$RUN/code/crispr_endpoint.py" tissue   dryrun "$RUN/dryrun/tissue"   2>&1 | tee "$RUN/logs/dryrun_tissue.log"
  exit 0
fi
# main: two scenarios in parallel, 4 threads each (8 total)
$PY "$RUN/code/crispr_endpoint.py" new_cell main "$RUN/output/new_cell" > "$RUN/logs/main_new_cell.log" 2>&1 &
P1=$!
$PY "$RUN/code/crispr_endpoint.py" tissue   main "$RUN/output/tissue"   > "$RUN/logs/main_tissue.log" 2>&1 &
P2=$!
wait $P1; R1=$?
wait $P2; R2=$?
echo "new_cell exit=$R1 tissue exit=$R2"
[ "$R1" -eq 0 ] && [ "$R2" -eq 0 ] || exit 1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
$PY "$RUN/code/analyze.py" "$RUN/output" 2>&1 | tee "$RUN/logs/analyze.log"

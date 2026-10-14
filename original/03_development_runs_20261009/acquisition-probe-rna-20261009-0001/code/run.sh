#!/bin/bash
# Launcher: integrity check -> dryrun (fold-0 inner banks, both scenarios) -> main new_cell -> main tissue -> analyze. 8 threads, serial.
set -e
RUN=/public/home/mengxl/creed-research-runs/acquisition-probe-rna-20261009-0001
PY=/public/home/mengxl/.conda/envs/CellFM/bin/python
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd $RUN
sha256sum -c SHA256SUMS | grep -v '^#'
date -Is
for sc in new_cell tissue; do $PY code/acquisition_probe_rna.py $sc dryrun $RUN/output/${sc}_dryrun; done
date -Is
$PY code/acquisition_probe_rna.py new_cell main $RUN/output/new_cell_main
date -Is
$PY code/acquisition_probe_rna.py tissue main $RUN/output/tissue_main
date -Is
$PY code/acquisition_probe_rna.py x analyze $RUN/output
date -Is
echo RUN_COMPLETE

#!/bin/bash
# Main launch: integrity check -> repro stage (gates A/B) -> main stage (all arms) -> analysis -> output hashes.
# Two scenarios run in parallel with 4 threads each (8 threads total).
set -e
RUN=/public/home/mengxl/creed-research-runs/rna-panel-20261009-0001
PY=/public/home/mengxl/.conda/envs/CellFM/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd $RUN
sha256sum -c SHA256SUMS
echo "START $(date -Is)"
mkdir -p output/repro logs
$PY code/rna_panel.py new_cell repro $RUN/output/repro/new_cell > logs/repro_new_cell.log 2>&1 &
$PY code/rna_panel.py tissue repro $RUN/output/repro/tissue > logs/repro_tissue.log 2>&1 &
wait
echo "REPRO_DONE $(date -Is)"
tail -n 1 logs/repro_new_cell.log logs/repro_tissue.log
if [ -f output/repro/new_cell/GATE_FAILED.json ] || [ -f output/repro/tissue/GATE_FAILED.json ]; then echo "GATE FAILED - stopping before new arms"; exit 2; fi
$PY - <<'PYEOF'
import json
for sc in ('new_cell', 'tissue'):
    r = json.load(open(f'/public/home/mengxl/creed-research-runs/rna-panel-20261009-0001/output/repro/{sc}/summary.json'))['reproduction']
    print(sc, 'passed', r['passed']); assert r['passed']['gate_A'] and r['passed']['gate_B'], 'reproduction gate failed'
PYEOF
$PY code/rna_panel.py new_cell main $RUN/output/new_cell > logs/main_new_cell.log 2>&1 &
$PY code/rna_panel.py tissue main $RUN/output/tissue > logs/main_tissue.log 2>&1 &
wait
echo "MAIN_DONE $(date -Is)"
tail -n 1 logs/main_new_cell.log logs/main_tissue.log
if [ -f output/new_cell/GATE_FAILED.json ] || [ -f output/tissue/GATE_FAILED.json ]; then echo "GATE FAILED in main"; exit 2; fi
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
$PY code/analyze.py $RUN > logs/analyze.log 2>&1
cat logs/analyze.log
find output -type f | sort | xargs sha256sum > OUTPUT_SHA256SUMS
echo "END $(date -Is)"
echo RUN_COMPLETE

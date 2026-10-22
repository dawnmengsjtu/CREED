#!/bin/bash
# PROSPECTIVE_LOWDIM_V1 preflight (RUN_GUIDE step 3). Checks that every input and saved output the run reads exists,
# that the frozen code is byte-identical to the original runs' copies, and reports cores / memory / disk / Python.
# Opens no numeric value of any data file. Safe to run any number of times before registration.
ROOT=/public/home/mengxl/creed-research-runs
RUN=$ROOT/prospective-lowdim-20261022-0001
PY=/public/home/mengxl/.conda/envs/CellFM/bin/python
miss=0
chk(){ if [ -e "$1" ]; then :; else echo "MISSING $1"; miss=$((miss+1)); fi; }
for s in new_cell tissue; do
  for f in 0 1 2 3 4; do
    for b in inner-0 inner-1 inner-2 outer; do chk $ROOT/extra-supervision-20261008-0001/output/$s/fold-$f/$b/geometry.npz; done
    chk $ROOT/crosslayer-matrix-20261009-0001/output/$s/fold-${f}_predictions.npz
    chk $ROOT/crosslayer-matrix-20261009-0001/output/$s/fold-${f}_inner_losses.npz
    chk $ROOT/settle-stageA-20261011-0001/output/$s/fold-$f/predictions.npz
    chk $ROOT/settle-stageA-20261011-0001/output/$s/fold-$f/fold_result.json
    chk $ROOT/settle-stageA-20261011-0001/output_crispr/$s/fold-$f/fold_result.json
    chk $ROOT/crispr-endpoint-20261009-0001/output/$s/fold-${f}_predictions.npz
    chk $ROOT/protein-panel-20261009-0001/output/${s}_main/fold-${f}_predictions.npz
    chk $ROOT/rna-panel-20261009-0001/output/$s/fold-${f}_predictions.npz
  done
  chk $ROOT/crosslayer-matrix-20261009-0001/output/$s/bank_info.json
  chk $ROOT/crispr-endpoint-20261009-0001/output/$s/fold-0_bank_info.json
  chk $ROOT/protein-panel-20261009-0001/output/${s}_main/bank_info.json
  chk $ROOT/rna-panel-20261009-0001/output/$s/fold-0_selection.npz
done
chk $ROOT/extra-supervision-20261008-0001/resources/m_gdsc_extra19.npz
chk $ROOT/gdsc-metabolite-full-protein-20260908-0001/output/result/source-inputs.npz
for x in rna_log_cpm.npy sample_ids.npy; do chk $ROOT/rna421-20260906-0002/$x; done
for x in effects.npy sample_ids.npy gene_ids.npy; do chk $ROOT/crispr310-prepare-20260908-0001/output/result/$x; done
echo "missing files: $miss"
# frozen code must equal the original runs' server copies byte for byte
cd $RUN 2>/dev/null && {
  sha256sum -c code/FROZEN_SHA256SUMS
  for p in "crosslayer.py settle-stageA-20261011-0001/code" "crispr_endpoint.py settle-stageA-20261011-0001/code" \
           "stageA.py settle-stageA-20261011-0001/code" "stageA_crispr.py settle-stageA-20261011-0001/code" \
           "protein_panel.py protein-panel-20261009-0001/code" "rna_panel.py rna-panel-20261009-0001/code"; do
    set -- $p; cmp -s code/frozen/$1 $ROOT/$2/$1 && echo "IDENTICAL $1 = $2/$1" || echo "DIFFERS  $1 vs $2/$1"
  done
}
echo "cores: $(nproc)"; free -g | head -2; df -h $ROOT | tail -1
$PY -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"
$PY -c "import openpyxl; print('openpyxl', openpyxl.__version__)" 2>/dev/null || echo "openpyxl not importable (the stdlib xlsx reader will be used)"

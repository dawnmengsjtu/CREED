#!/bin/bash
# PROSPECTIVE_LOWDIM_V1: the ONE invocation of the run (PLAN.md section 10). Fixed stage order; every gate is a hard stop.
#   0 integrity (SHA256SUMS, fresh directory)  1 GDSC1 data gate  2 dryrun (no outer bank)  3 main = outer predictions +
#   reproduction gates, no score  4 score (only after every gate of the scored endpoints passed)  5 analysis  6 output hashes
# 4 threads per process, as in the N6 run (settle-stageA-20261011-0001/code/run.sh), so the reproduction gates compare like with like;
# JOBS processes at a time (default 10, the N6 footprint of 40 threads). Start inside tmux/screen: bash code/run.sh > run.log 2>&1
set -euo pipefail
RUN=/public/home/mengxl/creed-research-runs/prospective-lowdim-20261022-0001
export PY=/public/home/mengxl/.conda/envs/CellFM/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 LOWDIM_RUN=$RUN PYTHONDONTWRITEBYTECODE=1
JOBS=${JOBS:-10}
cd $RUN
sha256sum -c SHA256SUMS
for d in data_gdsc1 dryrun output analysis; do test ! -e $d || { echo "STOP: $d already exists (one invocation only)"; exit 1; }; done
source ./INPUTS.env                      # GDSC1_FILE, GDSC1_SHA256, GDSC2_FILE (frozen in SHA256SUMS)
mkdir -p logs
echo "START $(date -Is)"
$PY -c "import sys, numpy, scipy; print(sys.version); print('numpy', numpy.__version__, 'scipy', scipy.__version__); numpy.show_config()" > logs/env.txt 2>&1
nproc >> logs/env.txt; free -g >> logs/env.txt 2>/dev/null || true

# ---- 1. GDSC1 data gate (PLAN 3.3). Failure stops part A only; part B (drug + CRISPR) continues.
EPS="drug crispr gdsc1"
if ! $PY code/prepare_gdsc1.py --gdsc1 "$GDSC1_FILE" --gdsc1-sha "$GDSC1_SHA256" --gdsc2 "$GDSC2_FILE" --out data_gdsc1 > logs/data_gate.log 2>&1; then
  echo "DATA_GATE_FAILED $(date -Is): part A (GDSC1) not run; see data_gdsc1/data_gate.json and logs/data_gate.log"
  EPS="drug crispr"
fi
echo "DATA_GATE_DONE $(date -Is) endpoints: $EPS"

# ---- 2. dryrun: fold-0 inner-0 only, no outer bank, no outer score
mkdir -p dryrun
for ep in $EPS; do for s in new_cell tissue; do echo "$ep $s"; done; done | \
  xargs -P $JOBS -L 1 bash -c '$PY code/lowdim.py $0 $1 dryrun dryrun/$0-$1 > logs/dryrun-$0-$1.log 2>&1' || true
for ep in $EPS; do for s in new_cell tissue; do
  $PY -c "import json,sys; sys.exit(0 if json.load(open('dryrun/$ep-$s/summary.json'))['passed'] else 1)" \
    || { echo "STOP: dryrun $ep $s failed (no outer prediction was made); see logs/dryrun-$ep-$s.log"; exit 2; }
done; done
echo "DRYRUN_PASSED $(date -Is)"

# ---- 3. main: outer predictions + reproduction gates (no score is computed in this stage)
for ep in $EPS; do for s in new_cell tissue; do for f in 0 1 2 3 4; do mkdir -p output/$ep/$s; echo "$ep $s $f"; done; done; done | \
  xargs -P $JOBS -L 1 bash -c '$PY code/lowdim.py $0 $1 main output/$0/$1/fold-$2 $2 > logs/main-$0-$1-fold-$2.log 2>&1' || true
SCORE_EPS=""
for ep in $EPS; do
  ok=1
  for s in new_cell tissue; do for f in 0 1 2 3 4; do
    d=output/$ep/$s/fold-$f
    if [ ! -f $d/gate.json ] || [ -f $d/GATE_FAILED.json ]; then ok=0; echo "GATE_NOT_PASSED $ep $s fold-$f"; fi
  done; done
  [ $ok = 1 ] && SCORE_EPS="$SCORE_EPS $ep"
done
# drug and CRISPR reproduce the frozen runs; if either failed, nothing is scored (PLAN 9.3)
case " $SCORE_EPS " in *" drug "*) ;; *) echo "STOP: drug-response gate failed; no score computed"; exit 2;; esac
case " $SCORE_EPS " in *" crispr "*) ;; *) echo "STOP: CRISPR gate failed; no score computed"; exit 2;; esac
echo "GATES_PASSED $(date -Is) scoring:$SCORE_EPS"

# ---- 4. score
for ep in $SCORE_EPS; do for s in new_cell tissue; do for f in 0 1 2 3 4; do echo "$ep $s $f"; done; done; done | \
  xargs -P $JOBS -L 1 bash -c '$PY code/lowdim.py $0 $1 score output/$0/$1/fold-$2 $2 > logs/score-$0-$1-fold-$2.log 2>&1'
echo "SCORED $(date -Is)"

# ---- 5. pre-registered analysis
$PY code/analyze.py $RUN > logs/analyze.log 2>&1
cat logs/analyze.log

# ---- 6. output hashes
find data_gdsc1 dryrun output analysis -type f 2>/dev/null | sort | xargs sha256sum > OUTPUT_SHA256SUMS
echo "END $(date -Is)"
echo RUN_COMPLETE

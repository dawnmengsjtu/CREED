#!/bin/bash
# PROSPECTIVE_LOWDIM_V1, ADDENDUM 1: resume after the score-stage software error (KeyError 'passed' in score_fold).
# Re-checks every main-stage gate, then runs stages 4-6 of code/run.sh unchanged: score -> analysis -> output hashes.
# Same threads and parallelism as code/run.sh. Every failure prints STOP. Start detached: bash code_addendum1/resume_after_addendum1.sh > run_resume.log 2>&1
set -euo pipefail
RUN=/public/home/mengxl/creed-research-runs/prospective-lowdim-20261022-0001
export PY=/public/home/mengxl/.conda/envs/CellFM/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 LOWDIM_RUN=$RUN PYTHONDONTWRITEBYTECODE=1
JOBS=${JOBS:-10}
cd $RUN
sha256sum -c ADDENDUM1_SHA256SUMS
test ! -e analysis || { echo "STOP: analysis/ already exists"; exit 1; }
test -z "$(find output -name fold_result.json -o -name per_target.npz)" || { echo "STOP: a fold was already scored"; exit 1; }
echo "RESUME_START $(date -Is)"

# ---- re-check the main-stage gates (run.log: GATES_PASSED 2026-10-22T09:33:55+0800 scoring: drug crispr gdsc1)
SCORE_EPS=""
for ep in drug crispr gdsc1; do
  ok=1
  for s in new_cell tissue; do for f in 0 1 2 3 4; do
    d=output/$ep/$s/fold-$f
    if [ -f $d/gate.json ] && [ ! -f $d/GATE_FAILED.json ] && \
       $PY -c "import json,sys; sys.exit(0 if json.load(open('$d/gate.json'))['gate']['passed'] is True else 1)"; then :; else
      ok=0; echo "GATE_NOT_PASSED $ep $s fold-$f"
    fi
  done; done
  [ $ok = 1 ] && SCORE_EPS="$SCORE_EPS $ep"
done
case " $SCORE_EPS " in *" drug "*) ;; *) echo "STOP: drug-response gate failed; no score computed"; exit 2;; esac
case " $SCORE_EPS " in *" crispr "*) ;; *) echo "STOP: CRISPR gate failed; no score computed"; exit 2;; esac
echo "GATES_RECHECKED $(date -Is) scoring:$SCORE_EPS"

# ---- 4. score (run.sh stage 4, unchanged except for the explicit STOP)
for ep in $SCORE_EPS; do for s in new_cell tissue; do for f in 0 1 2 3 4; do echo "$ep $s $f"; done; done; done | \
  xargs -P $JOBS -L 1 bash -c '$PY code/lowdim.py $0 $1 score output/$0/$1/fold-$2 $2 > logs/score-$0-$1-fold-$2.log 2>&1' \
  || { echo "STOP: a score job failed; see logs/score-*.log"; exit 3; }
echo "SCORED $(date -Is)"

# ---- 5. pre-registered analysis
$PY code/analyze.py $RUN > logs/analyze.log 2>&1 || { echo "STOP: analysis failed; see logs/analyze.log"; exit 4; }
cat logs/analyze.log

# ---- 6. output hashes
find data_gdsc1 dryrun output analysis -type f 2>/dev/null | sort | xargs sha256sum > OUTPUT_SHA256SUMS
echo "END $(date -Is)"
echo RUN_COMPLETE

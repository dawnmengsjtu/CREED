"""PROSPECTIVE_LOWDIM_V1, ADDENDUM 1: synthetic end-to-end test of the score and analysis stages.

Runs in the smoke directory, never in the registered run directory, and never computes a score against real labels.
  part A  score_fold on one fold per endpoint (new_cell, fold 0) after the real truth matrix is replaced by random
          numbers with the same NaN pattern; tests the gate check, the metrics call and the reference rows.
  part B  analyze.py on fully synthetic fold_result.json / per_target.npz files for all 30 folds, built from the arm
          lists in gate.json and the column lists in predictions.npz (only the 'cols' and 'evaluated' members are read).
usage: python synthetic_test.py REAL_RUN SYN_DIR
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

REAL, SYN = sys.argv[1], sys.argv[2]
PY = sys.executable
CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code')
EPS, SCENS, FOLDS = ('drug', 'crispr', 'gdsc1'), ('new_cell', 'tissue'), range(5)
REF = ('CREED_M+RNA', 'CREED_M+P', 'TEACHER_P', 'TEACHER_G')
rng = np.random.default_rng(20261022)
assert not os.path.exists(SYN), 'SYN_DIR must not exist'

# ---- part A: score_fold with randomised truth
for ep in EPS:
    src = f'{REAL}/output/{ep}/new_cell/fold-0'
    dst = f'{SYN}/A/output/{ep}/new_cell/fold-0'
    os.makedirs(dst)
    for f in ('gate.json', 'bank_info.json'):
        shutil.copy(f'{src}/{f}', dst)
    Z = np.load(f'{src}/predictions.npz', allow_pickle=False)
    arrays = {k: Z[k] for k in Z.files}
    truth = arrays['truth']
    fake = rng.normal(size=truth.shape)
    fake[~np.isfinite(truth)] = np.nan
    arrays['truth'] = fake
    np.savez_compressed(f'{dst}/predictions.npz', **arrays)
    r = subprocess.run([PY, f'{CODE}/lowdim.py', ep, 'new_cell', 'score', dst, '0'], capture_output=True, text=True)
    ok = r.returncode == 0 and os.path.exists(f'{dst}/fold_result.json') and os.path.exists(f'{dst}/per_target.npz')
    print(f'A score_fold {ep}: {"OK" if ok else "FAILED"}', (r.stderr or '')[-600:] if not ok else '')
    if not ok:
        sys.exit(1)

# ---- part B: analyze.py on synthetic fold results
B = f'{SYN}/B'
os.makedirs(f'{B}/data_gdsc1')
for f in ('data_gate.json', 'name_in_gdsc2.npy'):
    shutil.copy(f'{REAL}/data_gdsc1/{f}', f'{B}/data_gdsc1/{f}')
for ep in EPS:
    for s in SCENS:
        for k in FOLDS:
            src, dst = f'{REAL}/output/{ep}/{s}/fold-{k}', f'{B}/output/{ep}/{s}/fold-{k}'
            os.makedirs(dst)
            g = json.load(open(f'{src}/gate.json'))
            info = json.load(open(f'{src}/bank_info.json'))
            Z = np.load(f'{src}/predictions.npz', allow_pickle=False)
            cols, use = Z['cols'], Z['evaluated']
            arms = list(g['arms']) + [a for a in REF if a not in g['arms']]
            rows = [dict(fold=k, arm=a, score=float(1 + 0.02 * rng.normal()), spearman=float(0.2 + 0.05 * rng.normal()),
                         centered_mse=float(1 + 0.02 * rng.normal())) for a in arms]
            np.savez_compressed(f'{dst}/per_target.npz', cols=cols, evaluated=use, n_query_obs=rng.integers(5, 80, size=len(cols)),
                                **{'mse__' + a: rng.gamma(2., .5, size=len(cols)) for a in arms},
                                **{'rho__' + a: rng.uniform(-.3, .6, size=len(cols)) for a in arms})
            res = dict(endpoint=ep, scenario=s, fold=k, gate=g['gate'], rows=rows, n_fit=info['outer']['n_fit'],
                       n_query=info['outer']['n_query'], n_targets_evaluated=int(use.sum()),
                       diag={b: v['diag'] for b, v in info.items()}, seconds=0.0)
            json.dump(res, open(f'{dst}/fold_result.json', 'w'), indent=1, default=float)
r = subprocess.run([PY, f'{CODE}/analyze.py', B], capture_output=True, text=True)
ok = r.returncode == 0 and all(os.path.exists(f'{B}/analysis/{f}') for f in ('analysis.json', 'RESULT_AUTO.md'))
print(f'B analyze.py: {"OK" if ok else "FAILED"}', (r.stderr or '')[-1500:] if not ok else '')
if ok:
    print('analysis files:', sorted(os.listdir(f'{B}/analysis')))
sys.exit(0 if ok else 1)

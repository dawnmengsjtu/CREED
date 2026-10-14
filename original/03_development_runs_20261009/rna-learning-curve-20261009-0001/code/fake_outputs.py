"""Dev-only: synthetic outputs (random truth/predictions) to test analyze.py without any real outer scoring."""
import csv
import json
import os
import sys
import numpy as np

OUT = sys.argv[1]
EXT = '/public/home/mengxl/creed-research-runs/extra-supervision-20261008-0001/output'
ARMS = ['m_f25', 'm_f50', 'm_f75', 'm_f100', 'rna_c_f25', 'rna_c_f50', 'rna_c_f75', 'rna_c_f100', 'rna_c_extra', 'rna_u_f25',
        'rna_u_f50', 'rna_u_f75', 'rna_u_f100', 'rna_u_extra', 'tissue_f100', 'tissue_extra', 'rnat_c_f100', 'rnat_c_extra',
        'm_extra19', 'm_pseudo', 'm_pseudo_shuffle', 'm_pseudo_mean', 'mP_extra19', 'mP_pseudo', 'null_f100', 'null_extra354',
        'saved__extra__m_only', 'saved__extra__m_only__protein', 'saved__original__virtual_fm__protein']
SIZES = dict(rna=8, rnat=24, m_pseudo=12, mP_pseudo=72)
rng = np.random.default_rng(0)
for s in ('new_cell', 'tissue'):
    os.makedirs(f'{OUT}/{s}')
    rows = []
    for f in range(5):
        G = np.load(f'{EXT}/{s}/fold-{f}/outer/geometry.npz')
        held, fit, panel = G['query'], G['fit'], G['panel']
        truth = rng.random((len(held), len(panel))); truth[rng.random(truth.shape) < .2] = np.nan
        sd = np.full(len(panel), .2); use = np.ones(len(panel), bool)
        preds = {a: truth + rng.normal(0, .2 + .01 * i, truth.shape) for i, a in enumerate(ARMS)}
        sel = {a: rng.integers(0, 8, len(panel)) for a in ARMS if not a.startswith(('null', 'saved'))}
        np.savez_compressed(f'{OUT}/{s}/fold-{f}_predictions.npz', truth=truth, train_sd=sd, evaluated=use, panel=panel,
                            drug_ids=np.array([f'D{p}' for p in panel]), held=held, fit=fit,
                            **{'pred__' + k: np.nan_to_num(v, nan=.5) for k, v in preds.items()}, **{'sel__' + k: v for k, v in sel.items()})
        for a, v in preds.items():
            v = np.nan_to_num(v, nan=.5); obs = np.isfinite(truth)
            mse = np.where(obs, ((v - truth) / sd) ** 2, 0).sum(0) / np.maximum(obs.sum(0), 1)
            rows.append(dict(fold=f, arm=a, score=float(np.sqrt(mse[use].mean())), spearman=0., centered_mse=0.))
    with open(f'{OUT}/{s}/fold_metrics.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(dict(reproduction=dict(passed=True), summary={a: dict(spearman=0., centered_mse=0.) for a in ARMS}),
              open(f'{OUT}/{s}/summary.json', 'w'))

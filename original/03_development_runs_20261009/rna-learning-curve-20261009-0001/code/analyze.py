"""Pre-registered comparisons (PLAN.md section 8) from saved outer predictions of rna_lc.py.

usage: python analyze.py OUTPUT_DIR   (expects OUTPUT_DIR/new_cell and OUTPUT_DIR/tissue)
Writes OUTPUT_DIR/analysis.json and OUTPUT_DIR/comparisons.csv. Reads source-inputs only for tissue labels of test cells.
"""
import csv
import json
import os
import sys

import numpy as np
from scipy import stats

SOURCE = '/public/home/mengxl/creed-research-runs/gdsc-metabolite-full-protein-20260908-0001/output/result/source-inputs.npz'
OUT = sys.argv[1]
NBOOT = 2000
SEED = 20261009
SCEN = [s for s in ('new_cell', 'tissue') if os.path.exists(f'{OUT}/{s}/summary.json')]

# (id, new, base, role)
COMPARISONS = [
    ('G', 'rna_c_extra', 'rna_c_f100', 'primary_gate'),
    ('R', 'rna_c_f100', 'm_f100', 'primary_descriptive'),
    ('G_u', 'rna_u_extra', 'rna_u_f100', 'secondary'),
    ('corr', 'rna_c_f100', 'rna_u_f100', 'secondary'),
    ('corr_extra', 'rna_c_extra', 'rna_u_extra', 'secondary'),
    ('tissue_extra', 'tissue_extra', 'tissue_f100', 'secondary'),
    ('rnat_extra', 'rnat_c_extra', 'rnat_c_f100', 'secondary'),
    ('rna_extra_vs_tissue_extra', 'rna_c_extra', 'tissue_extra', 'secondary'),
    ('rna_extra_vs_null_extra', 'rna_c_extra', 'null_extra354', 'secondary'),
    ('rna_vs_null', 'rna_c_f100', 'null_f100', 'secondary'),
    ('null_extra_vs_null', 'null_extra354', 'null_f100', 'secondary'),
    ('rna_extra_vs_m', 'rna_c_extra', 'm_f100', 'secondary'),
    ('rnat_vs_rna', 'rnat_c_f100', 'rna_c_f100', 'secondary'),
    ('rnat_vs_tissue', 'rnat_c_f100', 'tissue_f100', 'secondary'),
    ('S', 'm_pseudo', 'saved__extra__m_only', 'primary_step2'),
    ('S_vs_shuffle', 'm_pseudo', 'm_pseudo_shuffle', 'attribution'),
    ('S_vs_mean', 'm_pseudo', 'm_pseudo_mean', 'attribution'),
    ('shuffle_vs_base', 'm_pseudo_shuffle', 'saved__extra__m_only', 'secondary'),
    ('mean_vs_base', 'm_pseudo_mean', 'saved__extra__m_only', 'secondary'),
    ('SP', 'mP_pseudo', 'saved__extra__m_only__protein', 'secondary_current_candidate'),
    ('S_vs_fmP', 'm_pseudo', 'saved__original__virtual_fm__protein', 'secondary_tissue_strongest'),
    ('SP_vs_fmP', 'mP_pseudo', 'saved__original__virtual_fm__protein', 'secondary_tissue_strongest'),
]
for p in (25, 50, 75):
    COMPARISONS.append((f'lc_rna_c_f100_vs_f{p}', 'rna_c_f100', f'rna_c_f{p}', 'learning_curve'))
    COMPARISONS.append((f'lc_m_f100_vs_f{p}', 'm_f100', f'm_f{p}', 'learning_curve'))
ROBUST = {'G', 'R', 'S', 'SP'}


def load(scenario):
    folds = []
    for f in range(5):
        z = np.load(f'{OUT}/{scenario}/fold-{f}_predictions.npz', allow_pickle=False)
        folds.append({k: z[k] for k in z.files})
    return folds


def per_drug_err(d, arm):
    t, sd = d['truth'], d['train_sd']
    return ((d['pred__' + arm] - t) / sd) ** 2


def fold_score(d, arm, drug_mask=None):
    e = per_drug_err(d, arm); obs = np.isfinite(d['truth'])
    mse = np.where(obs, e, 0).sum(0) / np.maximum(obs.sum(0), 1)
    use = d['evaluated'].astype(bool) if drug_mask is None else d['evaluated'].astype(bool) & drug_mask
    return float(np.sqrt(mse[use].mean()))


def nb_test(r, ratio):
    r = np.asarray(r); k = len(r); mean = r.mean(); s2 = r.var(ddof=1)
    se = np.sqrt((1 / k + ratio) * s2)
    if se == 0:
        return dict(t=None, p=None, ci=[float(mean), float(mean)])
    t = mean / se; p = 2 * stats.t.sf(abs(t), k - 1); q = stats.t.ppf(.975, k - 1)
    return dict(t=float(t), p=float(p), ci=[float(mean - q * se), float(mean + q * se)])


def main():
    S = np.load(SOURCE, allow_pickle=False); stis = S['tissues']
    data = {s: load(s) for s in SCEN}
    # common panel = intersection of tissue-scenario panels (if available), else all
    if 'tissue' in data:
        common = set.intersection(*[set(d['drug_ids'].tolist()) for d in data['tissue']])
    else:
        common = None
    result = dict(common_panel_size=len(common) if common else None, scenarios={})
    rows = []
    for s, folds in data.items():
        summ = json.load(open(f'{OUT}/{s}/summary.json'))
        arms = [k[6:] for k in folds[0] if k.startswith('pred__')]
        ratio = float(np.mean([len(d['held']) / len(d['fit']) for d in folds]))
        scores = {a: [fold_score(d, a) for d in folds] for a in arms}
        cmask = [np.isin(d['drug_ids'], list(common)) if common else np.ones(len(d['drug_ids']), bool) for d in folds]
        cscores = {a: [fold_score(d, a, cm) for d, cm in zip(folds, cmask)] for a in arms}
        # cross-check against rna_lc fold_metrics
        fm = {}
        with open(f'{OUT}/{s}/fold_metrics.csv') as fh:
            for r in csv.DictReader(fh):
                fm[(int(r['fold']), r['arm'])] = r
        xcheck = max(abs(scores[a][f] - float(fm[(f, a)]['score'])) for a in arms for f in range(5))
        # bootstrap weights (common random numbers across comparisons)
        rng = np.random.default_rng(SEED)
        union = sorted(set().union(*[set(d['drug_ids'].tolist()) for d in folds]))
        uidx = {g: i for i, g in enumerate(union)}
        dw = rng.multinomial(len(union), np.full(len(union), 1 / len(union)), size=NBOOT).astype(float)
        cw = []
        for d in folds:
            t = stis[d['held']]; ut, inv = np.unique(t, return_inverse=True)
            cnt = rng.multinomial(len(ut), np.full(len(ut), 1 / len(ut)), size=NBOOT).astype(float)
            cw.append(cnt[:, inv])

        def boot_scores(arm):
            out = np.zeros((NBOOT, 5))
            for f, d in enumerate(folds):
                e = per_drug_err(d, arm); obs = np.isfinite(e)
                num = cw[f] @ np.where(obs, e, 0); den = cw[f] @ obs.astype(float)
                mse = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
                v = dw[:, [uidx[g] for g in d['drug_ids']]] * d['evaluated'][None] * (den > 0)
                out[:, f] = np.sqrt((v * mse).sum(1) / v.sum(1))
            return out
        bcache = {}
        res = dict(ratio_ntest_ntrain=ratio, metric_crosscheck_max_abs=xcheck, reproduction=summ['reproduction'],
                   arms={a: dict(score=float(np.mean(scores[a])), fold_scores=scores[a],
                                 spearman=summ['summary'][a]['spearman'], centered_mse=summ['summary'][a]['centered_mse'],
                                 common_panel_score=float(np.mean(cscores[a])))
                         for a in arms}, comparisons={})
        for cid, new, base, role in COMPARISONS:
            if new not in scores or base not in scores:
                continue
            n_, b_ = np.array(scores[new]), np.array(scores[base])
            rel = (b_ - n_) / b_
            c = dict(new=new, base=base, role=role, new_mean=float(n_.mean()), base_mean=float(b_.mean()),
                     rel_by_fold=rel.tolist(), rel_mean=float(rel.mean()),
                     rel_of_means=float((b_.mean() - n_.mean()) / b_.mean()), folds_won=int((n_ < b_).sum()),
                     nb=nb_test(rel, ratio))
            cn, cb = np.array(cscores[new]), np.array(cscores[base]); crel = (cb - cn) / cb
            c['common_panel'] = dict(rel_mean=float(crel.mean()), folds_won=int((cn < cb).sum()), rel_by_fold=crel.tolist())
            c['drop_one_fold'] = [float(np.delete(rel, f).mean()) for f in range(5)]
            if cid in ROBUST:
                for a in (new, base):
                    if a not in bcache:
                        bcache[a] = boot_scores(a)
                br = ((bcache[base] - bcache[new]) / bcache[base]).mean(1)
                c['grouped_bootstrap'] = dict(reps=NBOOT, q025=float(np.quantile(br, .025)), q975=float(np.quantile(br, .975)),
                                              frac_positive=float((br > 0).mean()))
            if role == 'primary_gate':
                c['verdict'] = 'gate_pass' if (c['rel_mean'] >= .005 and c['folds_won'] >= 4) else 'gate_fail'
            elif cid in ('S', 'SP', 'S_vs_fmP', 'SP_vs_fmP'):
                c['verdict'] = ('push_line' if (c['rel_mean'] >= .01 and c['folds_won'] >= 4)
                                else 'small' if c['rel_mean'] > 0 else 'no_gain')
            res['comparisons'][cid] = c
            rows.append(dict(scenario=s, id=cid, role=role, new=new, base=base, new_mean=c['new_mean'], base_mean=c['base_mean'],
                             rel_mean_pct=100 * c['rel_mean'], folds_won=c['folds_won'], nb_p=c['nb']['p'],
                             ci_low_pct=100 * c['nb']['ci'][0], ci_high_pct=100 * c['nb']['ci'][1],
                             rel_by_fold_pct=' '.join(f'{100 * x:.3f}' for x in rel), verdict=c.get('verdict', '')))
        if 'S' in res['comparisons']:
            a1, a2 = res['comparisons'].get('S_vs_shuffle'), res['comparisons'].get('S_vs_mean')
            res['attribution'] = bool(a1 and a2 and a1['rel_mean'] > 0 and a2['rel_mean'] > 0 and a1['folds_won'] >= 4 and a2['folds_won'] >= 4)
        # selected configuration summaries (outer selection made on inner banks)
        sel = {}
        for a, size, labels in (('rna_c_extra', 4, ('k',)), ('rna_c_f100', 4, ('k',)), ('rna_u_extra', 4, ('k',)), ('rna_u_f100', 4, ('k',)),
                                ('rnat_c_f100', 8, ('w',)), ('rnat_c_extra', 8, ('w',)), ('m_pseudo', 4, ('w_p',)),
                                ('m_pseudo_shuffle', 4, ('w_p',)), ('m_pseudo_mean', 4, ('w_p',)), ('mP_pseudo', 24, ('w_p',))):
            if 'sel__' + a not in folds[0]:
                continue
            idx = np.concatenate([d['sel__' + a] for d in folds]) // size
            sel[a] = {labels[0] + str(i): int((idx == i).sum()) for i in range(int(idx.max()) + 1)}
        res['selected_block_counts'] = sel
        result['scenarios'][s] = res
    json.dump(result, open(f'{OUT}/analysis.json', 'w'), indent=1)
    with open(f'{OUT}/comparisons.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == '__main__':
    main()

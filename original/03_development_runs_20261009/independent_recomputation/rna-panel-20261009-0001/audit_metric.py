"""Auditor: independent recomputation of sRMSE / Spearman / centered MSE from saved predictions (read-only, prints only)."""
import csv, json, sys
import numpy as np
from scipy.stats import spearmanr, t as tdist

ROOT = '/public/home/mengxl/creed-research-runs'
RUN = ROOT + '/rna-panel-20261009-0001/output'
EXT = ROOT + '/extra-supervision-20261008-0001/output'
MAT = ROOT + '/crosslayer-matrix-20261009-0001/output'
PP = ROOT + '/protein-panel-20261009-0001/output'
KS = (8, 16, 32, 64, 128)


def my_metrics(truth, pred, sd, use):
    """Explicit per-drug loop: mse_j over observed test cells of ((pred-truth)/sd_j)^2; fold score = sqrt(mean over evaluated)."""
    nd = truth.shape[1]
    mse = np.full(nd, np.nan); rho = np.full(nd, np.nan); cmse = np.full(nd, np.nan)
    for j in range(nd):
        o = ~np.isnan(truth[:, j])
        if o.sum() == 0:
            mse[j] = 0.0; cmse[j] = 0.0
            continue
        r = (pred[o, j] - truth[o, j]) / sd[j]
        mse[j] = np.mean(r ** 2)
        cmse[j] = mse[j] - np.mean(r) ** 2
        if use[j] and truth[o, j].std() > 0 and pred[o, j].std() > 1e-15:
            rho[j] = spearmanr(truth[o, j], pred[o, j])[0]
    use = use.astype(bool)
    return float(np.sqrt(np.mean(mse[use]))), float(np.nanmean(rho[use])), float(np.mean(cmse[use]))


def nb(new, base, ratio):
    new, base = np.asarray(new), np.asarray(base)
    d = (base - new) / base * 100
    imp = (base.mean() - new.mean()) / base.mean() * 100
    wins = int((new < base).sum())
    v = d.var(ddof=1)
    se = np.sqrt((0.2 + ratio) * v)
    tv = d.mean() / se
    p = 2 * tdist.sf(abs(tv), 4)
    q = tdist.ppf(0.975, 4)
    lofo = min((np.delete(base, f).mean() - np.delete(new, f).mean()) / np.delete(base, f).mean() * 100 for f in range(5))
    return dict(imp=round(float(imp), 4), wins=wins, t=round(float(tv), 3), p=round(float(p), 4),
                ci=[round(float(d.mean() - q * se), 3), round(float(d.mean() + q * se), 3)], lofo_min=round(float(lofo), 4),
                per_fold=d.round(3).tolist())


# reported fold metrics of the run
rep = {}
for r in csv.DictReader(open(RUN + '/fold_metrics.csv')):
    rep[(r['scenario'], int(r['fold']), r['arm'])] = (float(r['score']), float(r['spearman']), float(r['centered_mse']))
summ = json.load(open(RUN + '/summary.json'))
ana = json.load(open(RUN + '/analysis.json'))

out = {}
for sc in ('new_cell', 'tissue'):
    fs = {}; ratio_parts = []
    maxd = dict(score=0., spearman=0., cmse=0.)
    chk = dict(m_only_vs_ext=0., m_allcand_vs_matrix=0., teacher_vs_matrix=0., truth_sd_use_vs_ext=0., m_only_vs_matrix_M=0.,
               ppanel_m_only=0., ppanel_teacher_identity=None)
    for f in range(5):
        z = np.load(f'{RUN}/{sc}/fold-{f}_predictions.npz', allow_pickle=False)
        truth, sd, use = z['truth'], z['train_sd'], z['evaluated']
        ratio_parts.append(len(z['held']) / len(z['fit']))
        E = np.load(f'{EXT}/{sc}/fold-{f}/predictions.npz', allow_pickle=False)
        chk['truth_sd_use_vs_ext'] = max(chk['truth_sd_use_vs_ext'],
                                         float(np.nanmax(np.abs(np.nan_to_num(E['truth'], nan=-999) - np.nan_to_num(truth, nan=-999)))),
                                         float(np.max(np.abs(E['train_sd'] - sd))), float((E['evaluated'] != use).sum()))
        chk['m_only_vs_ext'] = max(chk['m_only_vs_ext'], float(np.max(np.abs(z['m_only'] - E['original__m_only']))))
        C = np.load(f'{MAT}/{sc}/fold-{f}_predictions.npz', allow_pickle=False)
        assert (C['held'] == z['held']).all() and (C['panel'] == z['panel']).all() and (E['held'] == z['held']).all()
        chk['m_allcand_vs_matrix'] = max(chk['m_allcand_vs_matrix'], float(np.max(np.abs(z['m_allcand'] - C['pred__M+RNA']))))
        chk['teacher_vs_matrix'] = max(chk['teacher_vs_matrix'], float(np.max(np.abs(z['teacher_saved'] - C['pred__M+RNA']))))
        chk['m_only_vs_matrix_M'] = max(chk['m_only_vs_matrix_M'], float(np.max(np.abs(z['m_only'] - C['pred__M']))))
        P = np.load(f'{PP}/{sc}_main/fold-{f}_predictions.npz', allow_pickle=False)
        assert (P['held'] == z['held']).all() and (P['panel'] == z['panel']).all()
        chk['ppanel_m_only'] = max(chk['ppanel_m_only'], float(np.max(np.abs(z['m_only'] - P['m_only']))))
        arms = [k for k in z.files if k not in ('truth', 'train_sd', 'evaluated', 'fit', 'held', 'panel', 'drug_ids')]
        for a in arms:
            s, rho, cm = my_metrics(truth, z[a], sd, use)
            fs.setdefault(a, [None] * 5)[f] = s
            rs, rr, rc = rep[(sc, f, a)]
            maxd['score'] = max(maxd['score'], abs(s - rs)); maxd['cmse'] = max(maxd['cmse'], abs(cm - rc))
            if np.isfinite(rr) or np.isfinite(rho):
                maxd['spearman'] = max(maxd['spearman'], abs(rho - rr))
        # protein-panel saved teacher and m_panel arms: recompute their fold scores too
        for a in ('m_panel_k8', 'm_panel_k16', 'm_panel_k32', 'm_panel_k64', 'm_panel_sel', 'teacher_saved', 'm_allcand', 'm_only'):
            if a in P.files:
                s, _, _ = my_metrics(truth, P[a], sd, use)
                fs.setdefault('P::' + a, [None] * 5)[f] = s
    ratio = float(np.mean(ratio_parts))
    for p in (['random'] if sc == 'tissue' else ['random', 'random_tissue']):
        for k in (16, 32):
            fs[f'{p}_k{k}_mean'] = list(np.mean([fs[f'{p}_k{k}_s{s}'] for s in range(5)], axis=0))
    means = {a: float(np.mean(v)) for a, v in fs.items()}
    # compare with summary.json fold scores
    sdiff = 0.
    for a, v in summ[sc]['fold_scores'].items():
        sdiff = max(sdiff, float(np.max(np.abs(np.array(v) - np.array(fs[a])))))
    # protein fold scores from its fold_metrics.csv
    ppcsv = {}
    for r in csv.DictReader(open(f'{PP}/{sc}_main/fold_metrics.csv')):
        ppcsv.setdefault(r['arm'], [None] * 5)[int(r['fold'])] = float(r['score'])
    ppdiff = max(abs(np.array(ppcsv[a]) - np.array(fs['P::' + a])).max() for a in ('m_panel_k8', 'm_panel_k16', 'm_panel_k32', 'm_panel_k64', 'teacher_saved', 'm_only'))
    res = dict(nb_ratio=ratio, max_abs_diff_vs_fold_metrics=maxd, max_abs_diff_vs_summary_fold_scores=sdiff,
               protein_run_csv_vs_my_recompute=float(ppdiff), checks=chk)
    want = ['null_drug', 'm_only', 'm_allcand', 'teacher_saved', 'm_panel_sel', 'random_k16_mean', 'random_k32_mean', 'm_panel_pg_k32'] + [f'm_panel_k{k}' for k in KS]
    if sc == 'new_cell':
        want += ['tissue_only', 'm_tissue', 'm_panel_tissue_sel', 'm_allcand_tissue', 'random_tissue_k16_mean', 'random_tissue_k32_mean'] + [f'm_panel_tissue_k{k}' for k in KS]
    else:
        want += ['vfm', 'vfm_allcand', 'vfm_panel_sel'] + [f'vfm_panel_k{k}' for k in KS]
    want += ['P::m_only', 'P::teacher_saved', 'P::m_allcand', 'P::m_panel_sel'] + [f'P::m_panel_k{k}' for k in (8, 16, 32, 64)]
    res['means'] = {a: round(means[a], 6) for a in want}
    res['folds'] = {a: [round(x, 4) for x in fs[a]] for a in want}
    cmp = {'P1 m_panel_sel vs m_only': ('m_panel_sel', 'm_only'),
           'P2 k16': ('m_panel_k16', 'random_k16_mean'), 'P2 k32': ('m_panel_k32', 'random_k32_mean'),
           'P2c sel vs random_k32_mean': ('m_panel_sel', 'random_k32_mean'),
           'sec m_allcand vs m_only': ('m_allcand', 'm_only'), 'sec sel vs k64': ('m_panel_sel', 'm_panel_k64'),
           'sec k128 vs k64': ('m_panel_k128', 'm_panel_k64'), 'sec teacher vs sel': ('teacher_saved', 'm_panel_sel'),
           'X k8 G vs P': ('m_panel_k8', 'P::m_panel_k8'), 'X k16 G vs P': ('m_panel_k16', 'P::m_panel_k16'),
           'X k32 G vs P': ('m_panel_k32', 'P::m_panel_k32'), 'X k64 G vs P': ('m_panel_k64', 'P::m_panel_k64'),
           'X teacher G vs P': ('teacher_saved', 'P::teacher_saved'),
           'pg vs G k32': ('m_panel_pg_k32', 'm_panel_k32'), 'pg vs G k64': ('m_panel_pg_k32', 'm_panel_k64'),
           'pg vs P k32': ('m_panel_pg_k32', 'P::m_panel_k32'), 'pg vs P k64': ('m_panel_pg_k32', 'P::m_panel_k64')}
    for k in KS:
        cmp[f'sec m_panel_k{k} vs m_only'] = (f'm_panel_k{k}', 'm_only')
    if sc == 'new_cell':
        cmp.update({'P1b tissue_sel vs m_tissue': ('m_panel_tissue_sel', 'm_tissue'), 'P5 tissue_sel vs tissue_only': ('m_panel_tissue_sel', 'tissue_only'),
                    'P2t k16': ('m_panel_tissue_k16', 'random_tissue_k16_mean'), 'P2t k32': ('m_panel_tissue_k32', 'random_tissue_k32_mean')})
    else:
        cmp.update({'vfm_panel_sel vs vfm': ('vfm_panel_sel', 'vfm'), 'vfm_allcand vs vfm': ('vfm_allcand', 'vfm')})
    res['nb'] = {lab: nb(fs[n], fs[b], ratio) for lab, (n, b) in cmp.items()}
    # recovery
    m, tch = np.array(fs['m_only']), np.array(fs['teacher_saved'])
    rec_arms = [f'm_panel_k{k}' for k in KS] + ['m_panel_sel', 'm_panel_pg_k32', 'm_allcand', 'random_k16_mean', 'random_k32_mean']
    rec_arms += ([f'm_panel_tissue_k{k}' for k in KS] + ['m_panel_tissue_sel'] if sc == 'new_cell' else [f'vfm_panel_k{k}' for k in KS] + ['vfm_panel_sel'])
    res['R_rna'] = {a: dict(mean=round(float((m.mean() - np.mean(fs[a])) / (m.mean() - tch.mean())), 3),
                            per_fold=((m - np.array(fs[a])) / (m - tch)).round(2).tolist()) for a in rec_arms}
    pm, pt = np.array(fs['P::m_only']), np.array(fs['P::teacher_saved'])
    res['R_protein'] = {a: dict(mean=round(float((pm.mean() - np.mean(fs['P::' + a])) / (pm.mean() - pt.mean())), 3),
                                per_fold=((pm - np.array(fs['P::' + a])) / (pm - pt)).round(2).tolist()) for a in ['m_panel_k8', 'm_panel_k16', 'm_panel_k32', 'm_panel_k64', 'm_panel_sel', 'm_allcand']}
    # compare with analysis.json comparisons
    adiff = 0.
    for c in ana[sc]['comparisons'] + ana[sc]['crosslayer_comparisons']:
        n, b = c['new'], c['base']
        n_ = fs[n] if n in fs else None; b_ = fs[b] if b in fs else None
        if n_ is None or b_ is None:
            continue
        mine = nb(n_, b_, ratio)
        adiff = max(adiff, abs(mine['imp'] - c['improvement_pct']), abs(mine['p'] - c['nb_p']) if np.isfinite(c['nb_p']) else 0)
    res['max_abs_diff_vs_analysis_json_imp_and_p'] = float(adiff)
    res['analysis_nb_ratio'] = ana[sc]['nb_ratio_test_train']
    out[sc] = res
print(json.dumps(out, indent=1))

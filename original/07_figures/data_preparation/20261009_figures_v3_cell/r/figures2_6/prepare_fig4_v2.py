#!/usr/bin/env python
"""prepare_fig4_v2.py -- Figure 4 (V2): information controls and training-time information.

Reads ONLY the allowed research-run outputs listed in INPUTS and writes tidy long tables to
<V2>/delivery_R_20261009/figures2_6/figure4/plot_data/ plus FIG4_DATA_SOURCES.md (paths + sha256).

Nothing is re-estimated here: fold sRMSE values are copied from fold_metrics / fold_scores tables,
relative gains are ratios of fold means (the same definition the RESULT tables use), leave-one-fold-out
values are either copied verbatim (pushline run) or computed with that identical definition from the
saved fold scores (matrix run; checked against the pushline definition), and every interval / p-value is
copied verbatim from a run's comparisons.csv.
"""
import ast
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

RUNS = '/public/home/mengxl/creed-research-runs'
MATRIX = f'{RUNS}/crosslayer-matrix-20261009-0001/output'
PUSH = f'{RUNS}/pushline-checks-20261009-0001/output'
TK = f'{RUNS}/tissue-kernel-20261009-0001/output'
V2 = '/public/home/mengxl/workspaces/creed-research/figure_work/20261009_figures2_6_v2'
OUT = f'{V2}/delivery_R_20261009/figures2_6/figure4/plot_data'
os.makedirs(OUT, exist_ok=True)

INPUTS = []


def rd(path):
    INPUTS.append(path)
    return pd.read_csv(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        h.update(fh.read())
    return h.hexdigest()


def rel_gain(base, cand):
    """Relative improvement (%) of cand over base, both 5-fold mean sRMSE (RESULT definition)."""
    return 100.0 * (np.mean(base) - np.mean(cand)) / np.mean(base)


def lofo(base, cand):
    """Leave-one-fold-out relative gains (%), pushline definition: ratio of the remaining 4-fold means."""
    base = np.asarray(base, float)
    cand = np.asarray(cand, float)
    return [rel_gain(base[np.arange(5) != k], cand[np.arange(5) != k]) for k in range(5)]


# ----------------------------------------------------------------------------- matrix run
fm = {s: rd(f'{MATRIX}/{s}/fold_metrics.csv') for s in ('new_cell', 'tissue')}
mc = rd(f'{MATRIX}/comparisons.csv')


def mscore(s, arm):
    t = fm[s][fm[s].arm == arm].sort_values('fold')
    assert len(t) == 5, (s, arm)
    return t.score.to_numpy()


NULL = {'new_cell': float(np.mean(mscore('new_cell', 'null_drug'))),
        'tissue': float(np.mean(mscore('tissue', 'null_drug')))}

# ---- B: comparator minus virtual arm, per fold ------------------------------------------------
VARM = {'vRNA': 'M+vRNA', 'vP': 'M+vP'}
COMPS = {'vRNA': {'shuffle': 'M+vRNA_shuf', 'm_only': 'M', 'tissue_label': 'T', 'm_plus_t': 'M+T',
                  'measured_rna': 'M+RNA', 'measured_p': 'M+P'},
         'vP': {'shuffle': 'M+vP_shuf', 'm_only': 'M', 'tissue_label': 'T', 'm_plus_t': 'M+T',
                'measured_rna': 'M+RNA', 'measured_p': 'M+P'}}
rows = []
for s in fm:
    arms = set(fm[s].arm)
    for vl, cd in COMPS.items():
        v = mscore(s, VARM[vl])
        for ck, carm in cd.items():
            if carm not in arms:
                continue  # T and M+T were not run in the whole-tissue scenario (held-out tissues are unseen)
            c = mscore(s, carm)
            for f in range(5):
                rows.append(dict(scenario=s, virtual_layer=vl, virtual_arm=VARM[vl], comparator=ck,
                                 comparator_arm=carm, fold=f, srmse_comparator=c[f], srmse_virtual=v[f],
                                 delta=c[f] - v[f]))
controls = pd.DataFrame(rows)
controls.to_csv(f'{OUT}/fig4_controls.csv', index=False)

# ---- C: protein sample-graph regularisation (pushline run) -----------------------------------
fs = rd(f'{PUSH}/fold_scores.csv')
pc = rd(f'{PUSH}/comparisons.csv')


def pscore(s, arm):
    t = fs[(fs.scenario == s) & (fs.arm == arm)].sort_values('fold')
    assert len(t) == 5, (s, arm)
    return t.score.to_numpy()


GRAPH = {
    'new_cell': dict(base='extra__m_only', deploy='M only', data='original + 19 extra training cells',
                     arms={'metabolite': 'extra__m_only__metabolite', 'shuffle': 'extra__m_only__shuffle',
                           'mask': 'extra__m_only__mask', 'protein': 'extra__m_only__protein'},
                     pid='P1', pid_mask='P1_mask', pid_shuffle='P1_struct0', pid_metab='P1_struct1',
                     null_arm='null_extra'),
    'tissue': dict(base='original__virtual_fm', deploy='M only (frozen virtual-FM representation)',
                   data='original training cells',
                   arms={'metabolite': 'original__virtual_fm__metabolite', 'shuffle': 'original__virtual_fm__shuffle',
                         'mask': 'original__virtual_fm__mask', 'protein': 'original__virtual_fm__protein'},
                   pid='P3', pid_mask='P3_mask', pid_shuffle='P3_struct0', pid_metab='P3_struct1',
                   null_arm='null_original'),
}
rows, summ = [], []
for s, g in GRAPH.items():
    b = pscore(s, g['base'])
    nul = pscore(s, g['null_arm'])
    for f in range(5):
        rows.append(dict(scenario=s, base_arm=g['base'], arm='no_graph', arm_name=g['base'], fold=f, srmse=b[f],
                         srmse_base=b[f], rel_gain_pct=0.0, null_srmse=nul[f]))
    for a, an in g['arms'].items():
        x = pscore(s, an)
        for f in range(5):
            rows.append(dict(scenario=s, base_arm=g['base'], arm=a, arm_name=an, fold=f, srmse=x[f],
                             srmse_base=b[f], rel_gain_pct=100 * (b[f] - x[f]) / b[f], null_srmse=nul[f]))
        summ.append(dict(scenario=s, arm=a, arm_name=an, base_arm=g['base'], deploy_input=g['deploy'],
                         training_data=g['data'], base_mean=float(b.mean()), arm_mean=float(x.mean()),
                         null_mean=float(nul.mean()), mean_pct=rel_gain(b, x),
                         folds_won=int(np.sum(x < b)), source='fold_scores.csv (ratio of 5-fold means)'))
    r = pc[pc.id == g['pid']].iloc[0]
    for q in summ:
        if q['scenario'] == s and q['arm'] == 'protein':
            assert abs(q['mean_pct'] - r.rel_pct) < 1e-9
            q['mean_pct_verbatim'] = float(r.rel_pct)
            q['folds_won_verbatim'] = int(r.folds_won)
            q['nb_p_verbatim'] = float(r.nb_p)
            q['source'] = f'comparisons.csv id={g["pid"]} (verbatim)'
graph = pd.DataFrame(rows)
graph.to_csv(f'{OUT}/fig4_graph.csv', index=False)
graph_summary = pd.DataFrame(summ)
graph_summary.to_csv(f'{OUT}/fig4_graph_summary.csv', index=False)

# ---- D: leave-one-fold-out, E: intervals ------------------------------------------------------
lofo_rows, int_rows = [], []
for s in ('new_cell', 'tissue'):
    # virtual layers (matrix run): LOFO computed from saved fold scores with the pushline definition
    for eff, borrow in (('virtual_rna', 'RNA'), ('virtual_p', 'P')):
        r = mc[(mc.scenario == s) & (mc.kind == 'recovered') & (mc.deployed == 'M') & (mc.borrowed == borrow)].iloc[0]
        base, cand = mscore(s, r.base), mscore(s, r.new)
        # matrix-run definition: gain = mean of the per-fold relative gains (per_fold column, 4 dp)
        pf = 100.0 * (base - cand) / base
        assert np.allclose(pf, ast.literal_eval(r.per_fold), atol=6e-5), (s, eff, pf, r.per_fold)
        full = float(pf.mean())
        assert abs(full - r.gain) < 1e-9, (s, eff, full, r.gain)
        ci = ast.literal_eval(r.ci)
        for k in range(5):
            lofo_rows.append(dict(scenario=s, effect=eff, candidate=r.new, baseline=r.base, omitted_fold=k,
                                  rel_pct=float(pf[np.arange(5) != k].mean()), full_mean_pct=full,
                                  source='matrix fold_metrics.csv: mean of the remaining four per-fold relative gains (matrix-run definition)'))
        int_rows.append(dict(scenario=s, effect=eff, candidate=r.new, baseline=r.base, mean_pct=float(r.gain),
                             folds_won=int(r.wins), nb_p=float(r.p), nb_lo=ci[0], nb_hi=ci[1],
                             boot_lo=np.nan, boot_hi=np.nan, boot2_lo=np.nan, boot2_hi=np.nan,
                             interval_source='crosslayer-matrix comparisons.csv: Nadeau-Bengio corrected 95% CI only (no grouped bootstrap stored in that run)'))
    # protein graph (pushline run): everything verbatim
    g = GRAPH[s]
    for eff, pid in (('protein_graph', g['pid']), ('protein_vs_mask', g['pid_mask'])):
        r = pc[pc.id == pid].iloc[0]
        lo = [float(x) for x in str(r.lofo_rel_pct).split(';')]
        chk = lofo(pscore(s, r.base), pscore(s, r.cand))
        assert np.allclose(lo, chk, atol=5e-5), (pid, lo, chk)
        for k, v in enumerate(lo):
            lofo_rows.append(dict(scenario=s, effect=eff, candidate=r.cand, baseline=r.base, omitted_fold=k,
                                  rel_pct=v, full_mean_pct=float(r.rel_pct),
                                  source=f'pushline comparisons.csv id={pid} lofo_rel_pct (verbatim)'))
        int_rows.append(dict(scenario=s, effect=eff, candidate=r.cand, baseline=r.base, mean_pct=float(r.rel_pct),
                             folds_won=int(r.folds_won), nb_p=float(r.nb_p), nb_lo=float(r.nb_ci_lo),
                             nb_hi=float(r.nb_ci_hi), boot_lo=float(r.boot_lo), boot_hi=float(r.boot_hi),
                             boot2_lo=float(r.boot2_lo), boot2_hi=float(r.boot2_hi),
                             interval_source=f'pushline comparisons.csv id={pid}: grouped bootstrap 95% (cells by tissue cluster x drugs, B=2000) and Nadeau-Bengio corrected 95% CI (verbatim)'))
pd.DataFrame(lofo_rows).to_csv(f'{OUT}/fig4_lofo.csv', index=False)
pd.DataFrame(int_rows).to_csv(f'{OUT}/fig4_intervals.csv', index=False)

# ---- F: free tissue label (new-cell) ------------------------------------------------------------
tk = rd(f'{TK}/fold_metrics.csv')


def tscore(arm):
    t = tk[tk.arm == arm].sort_values('fold')
    assert len(t) == 5, arm
    return t.score.to_numpy()


rows = []
mnull = mscore('new_cell', 'null_drug')
for arm, lab, dep in (('T', 'T only', 'tissue label'), ('M', 'M only', 'M'), ('M+vRNA', 'M + virtual RNA', 'M'),
                      ('M+T', 'M + T', 'M + tissue label')):
    x = mscore('new_cell', arm)
    for f in range(5):
        rows.append(dict(run='matrix', arm=arm, arm_label=lab, deploy_input=dep, fold=f, srmse=x[f],
                         null_srmse=mnull[f], gain_vs_null_pct=100 * (mnull[f] - x[f]) / mnull[f],
                         mean_srmse=float(x.mean()), mean_gain_vs_null_pct=rel_gain(mnull, x)))
tnull = tscore('null_drug')
for arm, lab, dep in (('null_tissue_drug', 'Drug x tissue mean', 'tissue label'), ('tissue_only', 'T only', 'tissue label'),
                      ('repro_m', 'M only', 'M'), ('m_tissue', 'M + T', 'M + tissue label'),
                      ('m_tissue__protein', 'M + T + P graph', 'M + tissue label')):
    x = tscore(arm)
    for f in range(5):
        rows.append(dict(run='tissue_kernel', arm=arm, arm_label=lab, deploy_input=dep, fold=f, srmse=x[f],
                         null_srmse=tnull[f], gain_vs_null_pct=100 * (tnull[f] - x[f]) / tnull[f],
                         mean_srmse=float(x.mean()), mean_gain_vs_null_pct=rel_gain(tnull, x)))
pd.DataFrame(rows).to_csv(f'{OUT}/fig4_tissue_label.csv', index=False)

# ---- A: numbers for the schematic -------------------------------------------------------------
design = dict(
    cells=421, folds=5,
    drugs={'new_cell': int(fm['new_cell'].drugs.iloc[0]),
           'tissue_min': int(fm['tissue'].drugs.min()), 'tissue_max': int(fm['tissue'].drugs.max())},
    null_srmse={k: round(v, 3) for k, v in NULL.items()},
    deployment_inputs=['M only (225 metabolites)', 'M + tissue label T (new-cell only)'],
    training_time_information=['none (M only)', 'virtual RNA / virtual P (cross-fitted KRR generators)',
                               'paired shuffle of the virtual layer', 'protein sample graph: numeric / mask / shuffled / metabolite'],
    measured_references=['M + measured RNA', 'M + measured P'],
    graph_gain_pct={'new_cell': float(pc[pc.id == 'P1'].rel_pct.iloc[0]), 'tissue': float(pc[pc.id == 'P3'].rel_pct.iloc[0])},
)
with open(f'{OUT}/fig4_design.json', 'w') as fh:
    json.dump(design, fh, indent=1)

# ---- provenance ----------------------------------------------------------------------------------
outs = sorted(p for p in os.listdir(OUT) if p.startswith('fig4_'))
with open(f'{OUT}/FIG4_DATA_SOURCES.md', 'w') as fh:
    fh.write('# Figure 4 (V2) data sources\n\nGenerated by r/figures2_6/prepare_fig4_v2.py. '
             'Inputs are read-only research-run outputs; sha256 of each file at read time.\n\n## Inputs\n\n')
    for p in sorted(set(INPUTS)):
        fh.write(f'- `{p}`  \n  sha256 `{sha256(p)}`\n')
    fh.write('\n## Outputs (plot_data)\n\n')
    for p in outs:
        fh.write(f'- `{OUT}/{p}`  sha256 `{sha256(os.path.join(OUT, p))}`\n')
    fh.write('\n## Definitions\n\n'
             '- `delta` (fig4_controls): fold sRMSE(comparator) - fold sRMSE(virtual arm); positive = virtual arm has lower error.\n'
             '- `rel_gain_pct` / `mean_pct`: 100 x (baseline - candidate) / baseline on fold sRMSE (per fold) or on 5-fold means (summary); identical to the RESULT tables.\n'
             '- leave-one-fold-out: pushline rows verbatim (that run defines gain as the ratio of 4-/5-fold means); matrix rows use that run\'s own definition (mean of the per-fold relative gains; its `gain` and `per_fold` columns are asserted against the saved fold scores) and omit one fold from that mean.\n'
             '- intervals: copied verbatim; grouped bootstrap exists only for the pushline comparisons; matrix comparisons carry Nadeau-Bengio corrected 95% CIs only.\n'
             '- gain_vs_null_pct (fig4_tissue_label): 100 x (null_drug - arm) / null_drug per fold, each run against its own per-drug training-mean null.\n')
print('WROTE', OUT, outs)
print('NULL', NULL, 'graph gains', design['graph_gain_pct'])

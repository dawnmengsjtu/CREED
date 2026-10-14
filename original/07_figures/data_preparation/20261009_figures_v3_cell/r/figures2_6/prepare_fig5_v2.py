#!/usr/bin/env python
"""Figure 5 (V2) data stage: panel law + pre-registered negatives.

Reads ONLY:
  protein-panel-20261009-0001/output/{new_cell,tissue}_main/fold_metrics.csv
  probe2-context-abstain-20261009-0001/output/A/fold_metrics.csv, B/riskcov.csv, B/aurc.csv, summary.json
  acquisition-probe-20261009-0001/output/{new_cell,tissue}_main/fold_metrics.csv, analysis/summary.json
  rna-panel-20261009-0001/output/** (only if output/summary.json exists AND the local RESULT.md flag file is given)
Writes ONLY: <V2>/delivery_R_20261009/figures2_6/figure5/plot_data/*.csv|json + FIG5_DATA_SOURCES.md
"""
import hashlib, json, os, sys, glob
import numpy as np, pandas as pd

RUNS = '/public/home/mengxl/creed-research-runs'
PP = f'{RUNS}/protein-panel-20261009-0001/output'
P2 = f'{RUNS}/probe2-context-abstain-20261009-0001/output'
AQ = f'{RUNS}/acquisition-probe-20261009-0001/output'
RP = f'{RUNS}/rna-panel-20261009-0001/output'
OUT = '/public/home/mengxl/workspaces/creed-research/figure_work/20261009_figures2_6_v2/delivery_R_20261009/figures2_6/figure5/plot_data'
GENE_OK = '--gene-panel-ready' in sys.argv  # set by the caller only when the local RESULT.md exists too
os.makedirs(OUT, exist_ok=True)
SCN = ['new_cell', 'tissue']
SCN_LABEL = {'new_cell': 'New cell', 'tissue': 'Whole tissue'}
KS = [8, 16, 32, 64]
inputs = []

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''):
            h.update(ch)
    return h.hexdigest()

def rd(p, **kw):
    inputs.append(p)
    return pd.read_csv(p, **kw)

def rj(p):
    inputs.append(p)
    return json.load(open(p))

def wr(df, name):
    df.to_csv(os.path.join(OUT, name + '.csv'), index=False)
    print('wrote', name, df.shape)

# ---------------------------------------------------------------- protein panel (B, C, F reference)
pp = {}
for s in SCN:
    d = rd(f'{PP}/{s}_main/fold_metrics.csv')
    pp[s] = d.pivot(index='arm', columns='fold', values='score')
p2a = rd(f'{P2}/A/fold_metrics.csv')
p2 = {s: p2a[p2a.scenario == s].pivot(index='arm', columns='fold', values='score') for s in SCN}

def frac_means(base, arm, teacher):
    """R of 5-fold means (what the RESULT.md tables print)."""
    return float((base.mean() - arm.mean()) / (base.mean() - teacher.mean()))

def frac_folds(base, arm, teacher):
    return ((base - arm) / (base - teacher)).astype(float)

b_curve, b_fold = [], []
for s in SCN:
    P = pp[s]
    base, teach = P.loc['m_only'], P.loc['teacher_saved']
    # supervised protein panels, fixed k (m_panel_k) + all candidates
    for k in KS + ['all']:
        arm = P.loc[f'm_panel_k{k}'] if k != 'all' else P.loc['m_allcand']
        b_curve.append(dict(scenario=s, series='protein_supervised', k=str(k), k_num=(k if k != 'all' else np.nan),
                            R=frac_means(base, arm, teach), R_lo=np.nan, R_hi=np.nan, srmse=arm.mean(), n_seeds=np.nan,
                            baseline_srmse=base.mean(), teacher_srmse=teach.mean(), teacher='M + full proteome (teacher_saved)'))
        for f, r in frac_folds(base, arm, teach).items():
            b_fold.append(dict(scenario=s, series='protein_supervised', k=str(k), k_num=(k if k != 'all' else np.nan), fold=int(f), R=r,
                               srmse=float(arm[f])))
    # random protein panels: probe2 random_global_k{8,16,32,64}_s{0,1,2} (seeds s0-s2 are bit-identical to protein-panel random_k*_s0-2)
    Q = p2[s]
    qb, qt = Q.loc['m_only'], Q.loc['teacher_fullP_saved']
    assert np.allclose(qb.values, base.values) and np.allclose(qt.values, teach.values)
    for k in KS:
        seeds = [a for a in Q.index if a.startswith(f'random_global_k{k}_s')]
        Rs = [frac_means(qb, Q.loc[a], qt) for a in seeds]
        seed_mean_fold = Q.loc[seeds].mean(axis=0)
        b_curve.append(dict(scenario=s, series='protein_random', k=str(k), k_num=k, R=float(np.mean(Rs)), R_lo=float(np.min(Rs)),
                            R_hi=float(np.max(Rs)), srmse=float(seed_mean_fold.mean()), n_seeds=len(seeds),
                            baseline_srmse=qb.mean(), teacher_srmse=qt.mean(), teacher='M + full proteome (teacher_saved)'))
        for f, r in frac_folds(qb, seed_mean_fold, qt).items():
            b_fold.append(dict(scenario=s, series='protein_random', k=str(k), k_num=k, fold=int(f), R=r, srmse=float(seed_mean_fold[f])))

# ---------------------------------------------------------------- gene panels (optional)
gene_status = 'GENE PANEL PENDING'
if GENE_OK and os.path.exists(f'{RP}/summary.json'):
    try:
        for s in SCN:
            cands = [p for p in glob.glob(f'{RP}/**/fold_metrics.csv', recursive=True) if s in p and 'repro' not in p and 'dryrun' not in p]
            if not cands:
                raise FileNotFoundError(f'no main fold_metrics for {s}')
            g = rd(sorted(cands)[0])
            if 'scenario' in g.columns:
                g = g[g.scenario == s]
            G = g.pivot(index='arm', columns='fold', values='score')
            gb = G.loc['m_only']
            gt = G.loc['teacher_saved'] if 'teacher_saved' in G.index else G.loc['m_allcand']
            for k in [8, 16, 32, 64, 128, 'all']:
                name = f'm_panel_k{k}' if k != 'all' else 'm_allcand'
                if name not in G.index:
                    continue
                arm = G.loc[name]
                b_curve.append(dict(scenario=s, series='gene_supervised', k=str(k), k_num=(k if k != 'all' else np.nan), R=frac_means(gb, arm, gt),
                                    R_lo=np.nan, R_hi=np.nan, srmse=arm.mean(), n_seeds=np.nan, baseline_srmse=gb.mean(), teacher_srmse=gt.mean(),
                                    teacher='M + measured RNA (teacher_saved)'))
                for f, r in frac_folds(gb, arm, gt).items():
                    b_fold.append(dict(scenario=s, series='gene_supervised', k=str(k), k_num=(k if k != 'all' else np.nan), fold=int(f), R=r, srmse=float(arm[f])))
            for k in KS + [128]:
                seeds = [a for a in G.index if a.startswith(f'random_k{k}_s')]
                if not seeds:
                    continue
                Rs = [frac_means(gb, G.loc[a], gt) for a in seeds]
                sm = G.loc[seeds].mean(axis=0)
                b_curve.append(dict(scenario=s, series='gene_random', k=str(k), k_num=k, R=float(np.mean(Rs)), R_lo=float(np.min(Rs)), R_hi=float(np.max(Rs)),
                                    srmse=float(sm.mean()), n_seeds=len(seeds), baseline_srmse=gb.mean(), teacher_srmse=gt.mean(), teacher='M + measured RNA (teacher_saved)'))
                for f, r in frac_folds(gb, sm, gt).items():
                    b_fold.append(dict(scenario=s, series='gene_random', k=str(k), k_num=k, fold=int(f), R=r, srmse=float(sm[f])))
        gene_status = 'GENE PANEL INCLUDED'
    except Exception as e:  # noqa
        gene_status = f'GENE PANEL PENDING (read failed: {e})'
print(gene_status)
b_curve = pd.DataFrame(b_curve); b_fold = pd.DataFrame(b_fold)
b_curve['scenario_label'] = b_curve.scenario.map(SCN_LABEL); b_fold['scenario_label'] = b_fold.scenario.map(SCN_LABEL)
wr(b_curve, 'b_recovery_curve'); wr(b_fold, 'b_recovery_fold')

# ---------------------------------------------------------------- C: absolute sRMSE per fold
C_ARMS = [('null_drug', 'Null (per-drug mean)', 'none'), ('m_only', 'M only', 'M only'), ('m_panel_k16', 'M + 16 proteins', 'M + 16 measured proteins'),
          ('m_panel_k64', 'M + 64 proteins', 'M + 64 measured proteins'), ('teacher_saved', 'M + full proteome', 'M + full proteome (teacher)')]
c_fold, c_mean = [], []
for s in SCN:
    P = pp[s]
    for i, (a, lab, dep) in enumerate(C_ARMS):
        v = P.loc[a]
        for f in v.index:
            c_fold.append(dict(scenario=s, scenario_label=SCN_LABEL[s], arm=a, arm_label=lab, deployment_input=dep, order=i, fold=int(f), srmse=float(v[f]),
                               delta_vs_m_only=float(v[f] - P.loc['m_only'][f])))
        c_mean.append(dict(scenario=s, scenario_label=SCN_LABEL[s], arm=a, arm_label=lab, deployment_input=dep, order=i, srmse_mean=float(v.mean()),
                           delta_mean=float((v - P.loc['m_only']).mean()), null_mean=float(P.loc['null_drug'].mean())))
wr(pd.DataFrame(c_fold), 'c_abs_fold'); wr(pd.DataFrame(c_mean), 'c_abs_mean')

# ---------------------------------------------------------------- D: context-adaptive panels (probe2 A)
d_rows = []
for s in SCN:
    Q = p2[s]
    variants = ['tissue_adaptive', 'cluster_adaptive'] if s == 'new_cell' else ['cluster_adaptive']
    for var in variants:
        for k in [16, 32, 64]:
            ad = Q.loc[f'{var}_k{k}']
            refs = {'global_static': Q.loc[f'global_static_k{k}'],
                    'matched_random': Q.loc[[a for a in Q.index if a.startswith(f'matched_random_{var.split("_")[0]}_k{k}_s')]].mean(axis=0)}
            for ref, rv in refs.items():
                for f in ad.index:
                    d_rows.append(dict(scenario=s, scenario_label=SCN_LABEL[s], variant=var, reference=ref, k=k, fold=int(f), srmse_adaptive=float(ad[f]),
                                       srmse_reference=float(rv[f]), delta_abs=float(ad[f] - rv[f]), improvement_pct=float(100 * (rv[f] - ad[f]) / rv[f])))
d = pd.DataFrame(d_rows)
d_sum = d.groupby(['scenario', 'scenario_label', 'variant', 'reference', 'k']).apply(
    lambda g: pd.Series(dict(improvement_pct_of_means=100 * (g.srmse_reference.mean() - g.srmse_adaptive.mean()) / g.srmse_reference.mean(),
                             mean_fold_improvement_pct=g.improvement_pct.mean(), wins=int((g.delta_abs < 0).sum()), n_folds=len(g))), include_groups=False).reset_index()
wr(d, 'd_context_fold'); wr(d_sum, 'd_context_summary')

# ---------------------------------------------------------------- E: risk-coverage (probe2 B)
rc = rd(f'{P2}/B/riskcov.csv'); au = rd(f'{P2}/B/aurc.csv')
e_rows = []
for s in SCN:
    for arm, u1 in [('M+vRNA', 'u1_vRNA'), ('M+vP', 'u1_vP')]:
        sub = rc[(rc.scenario == s) & (rc.arm == arm)]
        full = sub[sub.coverage == 1.0].groupby('fold').score.first()  # identical across proxies
        for proxy_label, proxies in [('u1', [u1]), ('u3', ['u3']), ('random', [f'random_s{i}' for i in range(5)]), ('oracle', ['oracle'])]:
            t = sub[sub.proxy.isin(proxies)].groupby(['coverage', 'fold']).score.mean().reset_index()
            for _, r in t.iterrows():
                e_rows.append(dict(scenario=s, scenario_label=SCN_LABEL[s], arm=arm, arm_label={'M+vRNA': 'M + virtual RNA', 'M+vP': 'M + virtual protein'}[arm],
                                   proxy=proxy_label, coverage=float(r.coverage), fold=int(r.fold), srmse=float(r.score),
                                   full_srmse=float(full[r.fold]), rel_pct=float(100 * (r.score / full[r.fold] - 1))))
e = pd.DataFrame(e_rows)
e_mean = e.groupby(['scenario', 'scenario_label', 'arm', 'arm_label', 'proxy', 'coverage'], as_index=False).agg(srmse_mean=('srmse', 'mean'), rel_pct_mean=('rel_pct', 'mean'))
null_mean = {s: float(pp[s].loc['null_drug'].mean()) for s in SCN}
e_ref = e[e.coverage == 1.0].groupby(['scenario', 'scenario_label', 'arm', 'arm_label'], as_index=False).agg(full_srmse_mean=('srmse', 'mean'))
e_ref['null_srmse_mean'] = e_ref.scenario.map(null_mean)
# AURC per proxy (fold mean) for the caption
au2 = []
for s in SCN:
    for arm, u1 in [('M+vRNA', 'u1_vRNA'), ('M+vP', 'u1_vP')]:
        sub = au[(au.scenario == s) & (au.arm == arm)]
        for proxy_label, proxies in [('u1', [u1]), ('u3', ['u3']), ('random', [f'random_s{i}' for i in range(5)]), ('oracle', ['oracle'])]:
            au2.append(dict(scenario=s, arm=arm, proxy=proxy_label, aurc_mean=float(sub[sub.proxy.isin(proxies)].groupby('fold').aurc.mean().mean())))
wr(e, 'e_riskcov_fold'); wr(e_mean, 'e_riskcov_mean'); wr(e_ref, 'e_riskcov_ref'); wr(pd.DataFrame(au2), 'e_aurc')

# ---------------------------------------------------------------- F: fixed-head acquisition (acquisition probe) vs retrained head
f_curve, f_fold = [], []
for s in SCN:
    a = rd(f'{AQ}/{s}_main/fold_metrics.csv')
    A = a[a['head'] == 'H_meas'].copy(); A['k'] = A['k'].astype(str)
    piv = A.pivot_table(index=['policy', 'k'], columns='fold', values='score')
    k0 = piv.loc[('virtual', '0')]; kall = piv.loc[('measured_all', 'all')]
    assert np.allclose(kall.values, pp[s].loc['m_allcand'].values, atol=1e-10)
    for pol, lab in [('random', 'random'), ('variance', 'variance'), ('static', 'static'), ('adaptive', 'adaptive')]:
        for k in KS:
            if pol == 'random':
                seeds = [(f'random_s{i}', str(k)) for i in range(5)]
                Rs = [frac_means(k0, piv.loc[sd], kall) for sd in seeds]
                v = piv.loc[seeds].mean(axis=0)
                f_curve.append(dict(scenario=s, scenario_label=SCN_LABEL[s], head='fixed', series=f'fixed_{pol}', policy=lab, k=k, R=float(np.mean(Rs)),
                                    R_lo=float(np.min(Rs)), R_hi=float(np.max(Rs)), srmse=float(v.mean()), baseline_srmse=float(k0.mean()), teacher_srmse=float(kall.mean())))
            else:
                v = piv.loc[(pol, str(k))]
                f_curve.append(dict(scenario=s, scenario_label=SCN_LABEL[s], head='fixed', series=f'fixed_{pol}', policy=lab, k=k, R=frac_means(k0, v, kall),
                                    R_lo=np.nan, R_hi=np.nan, srmse=float(v.mean()), baseline_srmse=float(k0.mean()), teacher_srmse=float(kall.mean())))
            for f, r in frac_folds(k0, v, kall).items():
                f_fold.append(dict(scenario=s, scenario_label=SCN_LABEL[s], head='fixed', series=f'fixed_{pol}', policy=lab, k=k, fold=int(f), R=r, srmse=float(v[f])))
    # retrained head: protein-panel m_panel_k (= probe2 global_static_k, bit-identical), baseline M only, teacher m_allcand (same teacher as the probe)
    P = pp[s]; base, teach = P.loc['m_only'], P.loc['m_allcand']
    for k in KS:
        v = P.loc[f'm_panel_k{k}']
        f_curve.append(dict(scenario=s, scenario_label=SCN_LABEL[s], head='retrained', series='retrained_static', policy='static, head retrained', k=k,
                            R=frac_means(base, v, teach), R_lo=np.nan, R_hi=np.nan, srmse=float(v.mean()), baseline_srmse=float(base.mean()), teacher_srmse=float(teach.mean())))
        for f, r in frac_folds(base, v, teach).items():
            f_fold.append(dict(scenario=s, scenario_label=SCN_LABEL[s], head='retrained', series='retrained_static', policy='static, head retrained', k=k, fold=int(f), R=r, srmse=float(v[f])))
aq_sum = rj(f'{AQ}/analysis/summary.json')
wr(pd.DataFrame(f_curve), 'f_acquisition_curve'); wr(pd.DataFrame(f_fold), 'f_acquisition_fold')

# ---------------------------------------------------------------- A: protocol numbers
p2s = rj(f'{P2}/summary.json')
proto = dict(n_cells=421, n_folds=5, k_grid=KS, candidate_proteins='about 1,800 (fit-observed >= 90 %)', full_proteome='about 3,350',
             m_only=dict(new_cell=float(pp['new_cell'].loc['m_only'].mean()), tissue=float(pp['tissue'].loc['m_only'].mean())),
             teacher=dict(new_cell=float(pp['new_cell'].loc['teacher_saved'].mean()), tissue=float(pp['tissue'].loc['teacher_saved'].mean())),
             null=null_mean, gene_status=gene_status, kernel='0.5 K_M + 0.5 K_panel', score='sum over drugs of |Spearman(x_j, AUC)| on training cells')
json.dump(proto, open(os.path.join(OUT, 'a_protocol.json'), 'w'), indent=1)

# ---------------------------------------------------------------- provenance
seen = []
with open(os.path.join(OUT, '..', 'FIG5_DATA_SOURCES.md'), 'w') as fh:
    fh.write('# Figure 5 (V2) data sources\n\nGenerated by prepare_fig5_v2.py. Status: %s\n\n| input | sha256 |\n|---|---|\n' % gene_status)
    for p in inputs:
        if p in seen:
            continue
        seen.append(p); fh.write(f'| {p} | {sha(p)} |\n')
    fh.write('\nOutputs: ' + ', '.join(sorted(os.listdir(OUT))) + '\n')
print('DONE', gene_status)

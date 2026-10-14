#!/usr/bin/env python
"""Figure 7 (V2) data stage: the cross-layer map and the panel law on a second functional endpoint
(CRISPR gene effect) side by side with the GDSC drug-response numbers already used in Figures 2 and 5.

Reads ONLY (all read-only research-run outputs or existing V2 plot_data tables):
  crispr-endpoint-20261009-0001/output/{summary.json, fold_metrics.csv, comparisons.csv}
  protein-panel-20261009-0001/output/{new_cell,tissue}_main/fold_metrics.csv, output/analysis.json
  rna-panel-20261009-0001/output/fold_metrics.csv, output/analysis.json
  <V2>/delivery_R_20261009/figures2_6/figure2/plot_data/{xl_matrix.csv, xl_pair_folds.csv, xl_protocol.json}   (GDSC map)
  <V2>/delivery_R_20261009/figures2_6/figure5/plot_data/{b_recovery_curve.csv, b_recovery_fold.csv, a_protocol.json}  (GDSC panel law)
Writes ONLY: <V2>/delivery_R_20261009/figures2_6/figure7/plot_data/*.csv|json + FIG7_DATA_SOURCES.md

Conventions (same as Figures 2 and 5):
  gain%  = fold-averaged relative sRMSE improvement = mean over the 5 folds of 100*(base_f - new_f)/base_f
  share  = recovered gain / room gain (ratio of the two fold-averaged gains; printed as an integer %)
  R(k)   = (mean sRMSE[M only] - mean sRMSE[M + k panel]) / (mean sRMSE[M only] - mean sRMSE[teacher]),
           computed from 5-fold means; per-fold R uses the same formula within a fold
Ground truth for every printed number: the local RESULT.md / DECISION.md tables of the four runs.
"""
import hashlib, json, os
import numpy as np, pandas as pd

RUNS = '/public/home/mengxl/creed-research-runs'
CR = f'{RUNS}/crispr-endpoint-20261009-0001/output'
PP = f'{RUNS}/protein-panel-20261009-0001/output'
RP = f'{RUNS}/rna-panel-20261009-0001/output'
V2 = '/public/home/mengxl/workspaces/creed-research/figure_work/20261009_figures2_6_v2/delivery_R_20261009/figures2_6'
F2 = f'{V2}/figure2/plot_data'
F5 = f'{V2}/figure5/plot_data'
OUT = f'{V2}/figure7/plot_data'
os.makedirs(OUT, exist_ok=True)

SCN = ['new_cell', 'tissue']
SCN_LABEL = {'new_cell': 'New cell lines', 'tissue': 'Held-out tissue'}
END_LABEL = {'GDSC': 'GDSC drug response (AUC)', 'CRISPR': 'CRISPR gene effect'}
LAYERS = ['M', 'RNA', 'P']
KS = [8, 16, 32, 64]
inputs, checks = [], []


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


def check(name, got, want, tol):
    ok = abs(float(got) - float(want)) <= tol
    checks.append((name, got, want, tol, ok))
    if not ok:
        raise AssertionError(f'CHECK FAILED {name}: got {got} want {want} (tol {tol})')


def gain_fold(base, new):
    """per-fold paired relative gain (%), base/new are fold-indexed Series"""
    return 100 * (base - new) / base


# ====================================================================== CRISPR run
S = rj(f'{CR}/summary.json')
cfm = rd(f'{CR}/fold_metrics.csv', keep_default_na=False, na_values=[''])   # the null arm is literally named 'null'
assert 'null' in set(cfm.arm), 'null arm missing'
ccmp = rd(f'{CR}/comparisons.csv', keep_default_na=False, na_values=[''])   # read for provenance / cross-check of p-values
CF = {s: cfm[cfm.scenario == s].pivot(index='arm', columns='fold', values='score') for s in SCN}

# ====================================================================== GDSC tables (Figures 2 and 5)
xm = rd(f'{F2}/xl_matrix.csv')
xpf = rd(f'{F2}/xl_pair_folds.csv')
xproto = rj(f'{F2}/xl_protocol.json')
b_curve = rd(f'{F5}/b_recovery_curve.csv')
b_fold = rd(f'{F5}/b_recovery_fold.csv')
f5proto = rj(f'{F5}/a_protocol.json')

# ---------------------------------------------------------------- A: the two maps
a_rows = []
# GDSC map from Figure 2's xl_matrix (room / recovered / share; blocked = room <= 0.3 %)
for _, r in xm.iterrows():
    kind = 'M_row' if r.deployed == 'M' else ('no_room' if bool(r.blocked) else 'marginal')
    a_rows.append(dict(endpoint='GDSC', scenario=r.scenario, deployed=r.deployed, borrowed=r.borrowed,
                       room_gain=r.headroom_gain, room_wins=int(r.headroom_wins), room_p=r.headroom_p,
                       recovered_gain=r.recovered_gain, recovered_wins=int(r.recovered_wins), recovered_p=r.recovered_p,
                       share=(r.share if pd.notna(r.share) else np.nan), kind=kind))
# CRISPR map from summary.json cells (gain_pct is the fold-averaged gain; share = recovered/room when room > 0.3 %)
for s in SCN:
    for X in LAYERS:
        for Y in LAYERS:
            if X == Y:
                continue
            c = S[s]['cells'][X][Y]
            hr, rc = c['headroom'], c['recovered']
            share = c['share']
            blocked = hr['gain_pct'] <= 0.3
            kind = 'M_row' if X == 'M' else ('no_room' if hr['gain_pct'] <= 0 else 'marginal')
            # cross-check gain_pct = mean of per-fold gains recomputed from fold_metrics
            base, new, vnew = CF[s].loc[X], CF[s].loc[hr['new']], CF[s].loc[rc['new']]
            check(f'CRISPR {s} {X}->{Y} room = mean fold gain', hr['gain_pct'], gain_fold(base, new).mean(), 1e-6)
            check(f'CRISPR {s} {X}->{Y} recovered = mean fold gain', rc['gain_pct'], gain_fold(base, vnew).mean(), 1e-6)
            if share is not None:
                check(f'CRISPR {s} {X}->{Y} share = recovered/room', share, rc['gain_pct'] / hr['gain_pct'], 1e-9)
            a_rows.append(dict(endpoint='CRISPR', scenario=s, deployed=X, borrowed=Y,
                               room_gain=hr['gain_pct'], room_wins=int(hr['wins']), room_p=hr['nb_p'],
                               recovered_gain=rc['gain_pct'], recovered_wins=int(rc['wins']), recovered_p=rc['nb_p'],
                               share=(share if share is not None else np.nan), kind=kind))
a = pd.DataFrame(a_rows)
a['scenario_label'] = a.scenario.map(SCN_LABEL)
a['endpoint_label'] = a.endpoint.map(END_LABEL)
a['pair'] = a.deployed + '→' + a.borrowed
a['share_pct'] = 100 * a.share
# room ratio CRISPR / GDSC for the same cell (M row only is quoted on the figure)
g = a[a.endpoint == 'GDSC'].set_index(['scenario', 'deployed', 'borrowed']).room_gain
a['room_ratio_crispr_over_gdsc'] = [
    (r.room_gain / g.loc[(r.scenario, r.deployed, r.borrowed)]) if r.endpoint == 'CRISPR' else np.nan for _, r in a.iterrows()]
wr(a, 'a_map')

# printed numbers must match RESULT.md tables
want_cr = {('new_cell', 'RNA'): (1.934, 0.651, 34), ('new_cell', 'P'): (1.225, 0.388, 32),
           ('tissue', 'RNA'): (1.282, 0.389, 30), ('tissue', 'P'): (0.831, 0.252, 30)}
want_gd = {('new_cell', 'RNA'): (3.079, 1.056, 34), ('new_cell', 'P'): (2.401, 0.700, 29),
           ('tissue', 'RNA'): (2.195, 0.716, 33), ('tissue', 'P'): (2.207, 0.729, 33)}
for (s, Y), (room, rec, sh) in want_cr.items():
    r = a[(a.endpoint == 'CRISPR') & (a.scenario == s) & (a.deployed == 'M') & (a.borrowed == Y)].iloc[0]
    check(f'RESULT.md CRISPR {s} M->{Y} room', r.room_gain, room, 5e-4)
    check(f'RESULT.md CRISPR {s} M->{Y} recovered', r.recovered_gain, rec, 5e-4)
    check(f'RESULT.md CRISPR {s} M->{Y} share %', round(r.share_pct), sh, 0)
for (s, Y), (room, rec, sh) in want_gd.items():
    r = a[(a.endpoint == 'GDSC') & (a.scenario == s) & (a.deployed == 'M') & (a.borrowed == Y)].iloc[0]
    check(f'RESULT.md GDSC {s} M->{Y} room', r.room_gain, room, 5e-4)
    check(f'RESULT.md GDSC {s} M->{Y} recovered', r.recovered_gain, rec, 5e-4)
    check(f'RESULT.md GDSC {s} M->{Y} share %', round(r.share_pct), sh, 0)
# "CRISPR room is 40-60 % smaller": 1 - ratio for the four M-row cells
rr = a[(a.endpoint == 'CRISPR') & (a.deployed == 'M')].room_ratio_crispr_over_gdsc
print('CRISPR room smaller than GDSC by', [f'{100 * (1 - x):.0f}%' for x in rr])

# ---------------------------------------------------------------- B: share per fold, M row
b_fold_rows, b_mean_rows = [], []
for _, r in xpf[xpf.deployed == 'M'].iterrows():
    b_fold_rows.append(dict(endpoint='GDSC', scenario=r.scenario, deployed='M', borrowed=r.borrowed, fold=int(r.fold),
                            room_fold=r.headroom_fold, recovered_fold=r.recovered_fold, share_fold=r.share_fold))
for s in SCN:
    for Y in ['RNA', 'P']:
        c = S[s]['cells']['M'][Y]
        base, new, vnew = CF[s].loc['M'], CF[s].loc[c['headroom']['new']], CF[s].loc[c['recovered']['new']]
        hf, rf = gain_fold(base, new), gain_fold(base, vnew)
        for f in hf.index:
            check(f'CRISPR {s} M->{Y} fold {f} room == summary per_fold_pct', hf[f], c['headroom']['per_fold_pct'][int(f)], 6e-5)
            check(f'CRISPR {s} M->{Y} fold {f} recovered == summary per_fold_pct', rf[f], c['recovered']['per_fold_pct'][int(f)], 6e-5)
            b_fold_rows.append(dict(endpoint='CRISPR', scenario=s, deployed='M', borrowed=Y, fold=int(f),
                                    room_fold=float(hf[f]), recovered_fold=float(rf[f]), share_fold=float(rf[f] / hf[f])))
bf = pd.DataFrame(b_fold_rows)
bf['scenario_label'] = bf.scenario.map(SCN_LABEL); bf['endpoint_label'] = bf.endpoint.map(END_LABEL); bf['pair'] = 'M→' + bf.borrowed
bf['share_fold_pct'] = 100 * bf.share_fold
wr(bf, 'b_share_fold')
bm = a[a.deployed == 'M'][['endpoint', 'endpoint_label', 'scenario', 'scenario_label', 'pair', 'borrowed', 'room_gain', 'room_wins', 'room_p',
                            'recovered_gain', 'recovered_wins', 'recovered_p', 'share', 'share_pct']].copy()
bm['share_label'] = bm.share_pct.round().astype(int).astype(str) + '%'
# the GDSC share_fold column is recovered_fold / headroom_fold; verify on one row
r0 = xpf[(xpf.deployed == 'M')].iloc[0]
check('GDSC share_fold = recovered_fold/headroom_fold', r0.share_fold, r0.recovered_fold / r0.headroom_fold, 1e-9)
wr(bm, 'b_share_mean')

# ---------------------------------------------------------------- C: panel law R(k)
c_curve, c_fold = [], []
SER = {'protein_supervised': 'protein', 'gene_supervised': 'gene'}
for _, r in b_curve[b_curve.series.isin(SER) & b_curve.k_num.isin(KS)].iterrows():
    c_curve.append(dict(endpoint='GDSC', scenario=r.scenario, layer=SER[r.series], k=int(r.k_num), R=r.R, srmse=r.srmse,
                        baseline_srmse=r.baseline_srmse, teacher_srmse=r.teacher_srmse, teacher=r.teacher))
for _, r in b_fold[b_fold.series.isin(SER) & b_fold.k_num.isin(KS)].iterrows():
    c_fold.append(dict(endpoint='GDSC', scenario=r.scenario, layer=SER[r.series], k=int(r.k_num), fold=int(r.fold), R=r.R, srmse=r.srmse))
TEACH = {'P': ('M+Pval', 'protein'), 'G': ('M+RNA', 'gene')}
for s in SCN:
    for L, (tarm, lname) in TEACH.items():
        base, teach = CF[s].loc['M'], CF[s].loc[tarm]
        for k in KS:
            e = S[s]['panels'][L][str(k)]
            arm = CF[s].loc[f'm_panel_{L}_k{k}']
            R = (base.mean() - arm.mean()) / (base.mean() - teach.mean())
            check(f'CRISPR {s} {lname} k={k} R recomputed == summary', R, e['recovery']['R'], 1e-9)
            check(f'CRISPR {s} {lname} k={k} teacher arm', 0 if e['recovery']['teacher'] == tarm else 1, 0, 0)
            c_curve.append(dict(endpoint='CRISPR', scenario=s, layer=lname, k=k, R=float(R), srmse=float(arm.mean()),
                                baseline_srmse=float(base.mean()), teacher_srmse=float(teach.mean()), teacher=f'M + measured {tarm.split("+")[1]} (CRISPR run)'))
            Rf = (base - arm) / (base - teach)
            for f in Rf.index:
                check(f'CRISPR {s} {lname} k={k} fold {f} R == summary', Rf[f], e['recovery']['R_per_fold'][int(f)], 6e-5)
                c_fold.append(dict(endpoint='CRISPR', scenario=s, layer=lname, k=k, fold=int(f), R=float(Rf[f]), srmse=float(arm[f])))
cc = pd.DataFrame(c_curve); cf = pd.DataFrame(c_fold)
for d in (cc, cf):
    d['scenario_label'] = d.scenario.map(SCN_LABEL); d['endpoint_label'] = d.endpoint.map(END_LABEL)
    d['layer_label'] = d.layer.map({'protein': 'Protein panel', 'gene': 'Gene panel'})
wr(cc, 'c_panel_curve'); wr(cf, 'c_panel_fold')
# RESULT.md tables (CRISPR §5; GDSC rna_panel §4 / protein_panel P3 as printed in Fig. 5)
want_R = {('CRISPR', 'new_cell', 'protein'): [0.429, 0.477, 0.606, 0.748], ('CRISPR', 'new_cell', 'gene'): [0.336, 0.399, 0.560, 0.713],
          ('CRISPR', 'tissue', 'protein'): [0.489, 0.578, 0.663, 0.890], ('CRISPR', 'tissue', 'gene'): [0.451, 0.484, 0.563, 0.700],
          ('GDSC', 'new_cell', 'protein'): [0.48, 0.56, 0.65, 0.72], ('GDSC', 'new_cell', 'gene'): [0.37, 0.56, 0.72, 0.76],
          ('GDSC', 'tissue', 'protein'): [0.64, 0.57, 0.66, 0.74], ('GDSC', 'tissue', 'gene'): [0.45, 0.64, 0.78, 0.76]}
for (e, s, L), vals in want_R.items():
    for k, v in zip(KS, vals):
        got = cc[(cc.endpoint == e) & (cc.scenario == s) & (cc.layer == L) & (cc.k == k)].R.iloc[0]
        check(f'RESULT.md R(k) {e} {s} {L} k={k}', got, v, 5e-4 if e == 'CRISPR' else 5e-3)

# ---------------------------------------------------------------- D left: supervised minus random panel gain, k = 16, 32
d_fold, d_mean = [], []
pp_an = rj(f'{PP}/analysis.json'); rp_an = rj(f'{RP}/analysis.json')
ppf = {s: rd(f'{PP}/{s}_main/fold_metrics.csv').pivot(index='arm', columns='fold', values='score') for s in SCN}
rpf_all = rd(f'{RP}/fold_metrics.csv')
rpf = {s: rpf_all[rpf_all.scenario == s].pivot(index='arm', columns='fold', values='score') for s in SCN}


def find_cmp(an, s, new, base):
    hits = [c for c in an[s]['comparisons'] if c['new'] == new and c['base'] == base]
    assert len(hits) == 1, (s, new, base, len(hits))
    return hits[0]


for s in SCN:
    for lname, piv, an, pref in [('protein', ppf[s], pp_an, ''), ('gene', rpf[s], rp_an, '')]:
        for k in [16, 32]:
            sup = piv.loc[f'm_panel_k{k}']
            seeds = sorted(a_ for a_ in piv.index if a_.startswith(f'random_k{k}_s'))
            assert len(seeds) == 5, (s, lname, k, seeds)
            rnd = piv.loc[seeds].mean(axis=0)
            gf = gain_fold(rnd, sup)
            cmp_ = find_cmp(an, s, f'm_panel_k{k}', f'random_k{k}_mean')
            for f in gf.index:
                check(f'GDSC {s} {lname} k={k} fold {f} sup-vs-random gain == analysis.json', gf[f], cmp_['per_fold_improvement_pct'][int(f)], 6e-3)
                d_fold.append(dict(endpoint='GDSC', scenario=s, layer=lname, k=k, fold=int(f), gain_pct=float(gf[f]),
                                   srmse_supervised=float(sup[f]), srmse_random_mean=float(rnd[f])))
            rom = 100 * (rnd.mean() - sup.mean()) / rnd.mean()
            check(f'GDSC {s} {lname} k={k} ratio-of-means == analysis.json improvement_pct', rom, cmp_['improvement_pct'], 1e-6)
            d_mean.append(dict(endpoint='GDSC', scenario=s, layer=lname, k=k, gain_pct_fold_mean=float(gf.mean()), gain_pct_ratio_of_means=float(rom),
                               wins=int((gf > 0).sum()), nb_p=float(cmp_['nb_p']), n_seeds=5,
                               result_md_value=float(cmp_['improvement_pct']), result_md_convention='ratio of 5-fold means (protein_panel / rna_panel RESULT.md P2)'))
    for L, lname in [('P', 'protein'), ('G', 'gene')]:
        for k in [16, 32]:
            e = S[s]['panels'][L][f'vs_random_k{k}']
            sup = CF[s].loc[f'm_panel_{L}_k{k}']
            seeds = sorted(a_ for a_ in CF[s].index if a_.startswith(f'random_{L}_k{k}_s'))
            assert len(seeds) == 5, (s, L, k, seeds)
            rnd = CF[s].loc[seeds].mean(axis=0)
            gf = gain_fold(rnd, sup)
            check(f'CRISPR {s} {lname} k={k} fold-mean sup-vs-random == summary gain_pct', gf.mean(), e['gain_pct'], 1e-6)
            for f in gf.index:
                check(f'CRISPR {s} {lname} k={k} fold {f} == summary per_fold_pct', gf[f], e['per_fold_pct'][int(f)], 6e-5)
                d_fold.append(dict(endpoint='CRISPR', scenario=s, layer=lname, k=k, fold=int(f), gain_pct=float(gf[f]),
                                   srmse_supervised=float(sup[f]), srmse_random_mean=float(rnd[f])))
            d_mean.append(dict(endpoint='CRISPR', scenario=s, layer=lname, k=k, gain_pct_fold_mean=float(gf.mean()),
                               gain_pct_ratio_of_means=float(e['gain_pct_ratio_of_means']), wins=int(e['wins']), nb_p=float(e['nb_p']), n_seeds=5,
                               result_md_value=float(e['gain_pct']), result_md_convention='fold-averaged gain (crispr RESULT.md §5)'))
df_ = pd.DataFrame(d_fold); dm = pd.DataFrame(d_mean)
for d in (df_, dm):
    d['scenario_label'] = d.scenario.map(SCN_LABEL); d['endpoint_label'] = d.endpoint.map(END_LABEL)
    d['layer_label'] = d.layer.map({'protein': 'Protein', 'gene': 'Gene'})
    d['group'] = d.layer_label + ' k=' + d.k.astype(str)
wr(df_, 'd_random_fold'); wr(dm, 'd_random_mean')
want_rand = {('GDSC', 'new_cell', 'protein', 16): (0.524, 0.007), ('GDSC', 'tissue', 'protein', 16): (0.424, 0.102),
             ('GDSC', 'new_cell', 'protein', 32): (0.325, 0.079), ('GDSC', 'tissue', 'protein', 32): (0.261, 0.199),
             ('GDSC', 'new_cell', 'gene', 16): (0.456, 0.158), ('GDSC', 'new_cell', 'gene', 32): (0.464, 0.145),
             ('GDSC', 'tissue', 'gene', 16): (0.548, 0.257), ('GDSC', 'tissue', 'gene', 32): (0.635, 0.443),
             ('CRISPR', 'new_cell', 'protein', 16): (0.332, 0.130), ('CRISPR', 'new_cell', 'protein', 32): (0.259, 0.223),
             ('CRISPR', 'new_cell', 'gene', 16): (0.051, 0.763), ('CRISPR', 'new_cell', 'gene', 32): (-0.004, 0.974),
             ('CRISPR', 'tissue', 'protein', 16): (0.168, 0.190), ('CRISPR', 'tissue', 'protein', 32): (0.077, 0.483),
             ('CRISPR', 'tissue', 'gene', 16): (0.135, 0.362), ('CRISPR', 'tissue', 'gene', 32): (0.064, 0.669)}
for (e, s, L, k), (v, p) in want_rand.items():
    r = dm[(dm.endpoint == e) & (dm.scenario == s) & (dm.layer == L) & (dm.k == k)].iloc[0]
    check(f'RESULT.md sup-vs-random {e} {s} {L} k={k} value', r.result_md_value, v, 5e-4)
    check(f'RESULT.md sup-vs-random {e} {s} {L} k={k} p', r.nb_p, p, 5e-4)

# ---------------------------------------------------------------- D right: CRISPR absolute sRMSE per fold
ARMS = [('null', 'Null (per-gene fit mean)', 'none'), ('M', 'M only', 'M only'), ('M+vRNA', 'M + virtual RNA', 'M only'),
        ('M+RNA', 'M + measured RNA (ref.)', 'measured RNA (not deployable)')]
e_fold, e_mean = [], []
for s in SCN:
    for i, (arm, lab, dep) in enumerate(ARMS):
        v = CF[s].loc[arm]
        for f in v.index:
            e_fold.append(dict(scenario=s, scenario_label=SCN_LABEL[s], arm=arm, arm_label=lab, deployment_input=dep, order=i, fold=int(f), srmse=float(v[f])))
        e_mean.append(dict(scenario=s, scenario_label=SCN_LABEL[s], arm=arm, arm_label=lab, deployment_input=dep, order=i, srmse_mean=float(v.mean()),
                           null_mean=float(CF[s].loc['null'].mean()), gain_vs_null_pct=float(gain_fold(CF[s].loc['null'], v).mean()),
                           wins_vs_null=int((v < CF[s].loc['null']).sum())))
    nvm = S[s]['secondary']['null_vs_M']
    check(f'CRISPR {s} M vs null gain == summary', gain_fold(CF[s].loc['null'], CF[s].loc['M']).mean(), nvm['gain_pct'], 1e-6)
    check(f'CRISPR {s} null mean == summary null_score', CF[s].loc['null'].mean(), S[s]['null_score'], 1e-9)
check('RESULT.md CRISPR tissue M vs null +0.008%', S['tissue']['secondary']['null_vs_M']['gain_pct'], 0.008, 5e-4)
check('RESULT.md CRISPR new_cell M vs null +1.026%', S['new_cell']['secondary']['null_vs_M']['gain_pct'], 1.026, 5e-4)
check('RESULT.md CRISPR tissue M vs null wins 3/5', S['tissue']['secondary']['null_vs_M']['wins'], 3, 0)
wr(pd.DataFrame(e_fold), 'd_abs_fold'); wr(pd.DataFrame(e_mean), 'd_abs_mean')

# ---------------------------------------------------------------- protocol numbers for notes / caption
q = {s: {str(f): S[s]['evaluated_per_bank'][str(f)]['outer']['n_query_crispr'] for f in range(5)} for s in SCN}
nq = [v for s in SCN for v in q[s].values()]
proto = dict(
    crispr=dict(run='crispr-endpoint-20261009-0001', cells_with_crispr=310, development_cells=421, cells_without_crispr_label=111,
                query_cells_with_crispr_per_fold=q, query_cells_range=[int(min(nq)), int(max(nq))],
                fit_cells_with_crispr_per_fold={s: {str(f): S[s]['evaluated_per_bank'][str(f)]['outer']['n_fit_crispr'] for f in range(5)} for s in SCN},
                targets_evaluated_per_fold={s: S[s]['evaluated_per_fold'] for s in SCN}, target_rule='fit SD >= 0.2 (pre-registered)',
                null=dict(new_cell=S['new_cell']['null_score'], tissue=S['tissue']['null_score']), null_definition='per-gene fit-set mean',
                M_vs_null_gain_pct={s: S[s]['secondary']['null_vs_M']['gain_pct'] for s in SCN},
                M_vs_null_wins={s: S[s]['secondary']['null_vs_M']['wins'] for s in SCN},
                M_vs_null_p={s: S[s]['secondary']['null_vs_M']['nb_p'] for s in SCN},
                teacher=dict(protein='M+Pval (value kernel of ~3,350 protein candidates)', gene='M+RNA (3,000 genes)'),
                gene_pool=3000, protein_pool='3,342-3,350 (>= 50 % observed); panel candidates 1,742-1,944 (>= 90 % observed)',
                random_seeds=5, criteria={s: S[s]['criteria']['verdict'] for s in SCN}, n_passed={s: S[s]['criteria']['n_passed'] for s in SCN}),
    gdsc=dict(matrix_run='crosslayer-matrix-20261009-0001', protein_panel_run='protein-panel-20261009-0001', rna_panel_run='rna-panel-20261009-0001',
              cells=xproto['cells_total'], drugs_new_cell=xproto['drugs_new_cell'], drugs_tissue=xproto['drugs_tissue'],
              null=xproto['null_mean'], m_only=f5proto['m_only'], teacher_protein=f5proto['teacher'],
              teacher_gene={s: float(b_curve[(b_curve.series == 'gene_supervised') & (b_curve.scenario == s)].teacher_srmse.iloc[0]) for s in SCN},
              random_seeds=5),
    k_grid=KS, scenarios=SCN_LABEL, endpoints=END_LABEL,
    definitions=dict(gain='fold-averaged relative sRMSE improvement: mean over 5 folds of 100*(base_f-new_f)/base_f',
                     room='gain of X + measured Y over X', recovered='gain of X + virtual Y over X',
                     share='recovered / room, defined when room > 0.3 %', R='(mean sRMSE[M] - mean sRMSE[M+k panel]) / (mean sRMSE[M] - mean sRMSE[teacher])'))
json.dump(proto, open(os.path.join(OUT, 'protocol.json'), 'w'), indent=1)
print('query cells with CRISPR per fold:', q, 'range', proto['crispr']['query_cells_range'])

# ---------------------------------------------------------------- provenance
seen = []
with open(os.path.join(OUT, 'FIG7_DATA_SOURCES.md'), 'w') as fh:
    fh.write('# Figure 7 (V2) data sources\n\nGenerated by r/figures2_6/prepare_fig7_v2.py. Read-only inputs (sha256):\n\n| input | sha256 |\n|---|---|\n')
    for p in inputs:
        if p in seen:
            continue
        seen.append(p); fh.write(f'| `{p}` | `{sha(p)}` |\n')
    fh.write('\nGround truth for printed numbers (local): crispr_endpoint_20261009/{RESULT.md, DECISION.md}, crosslayer_matrix_20261009/RESULT.md, '
             'protein_panel_20261009/RESULT.md, rna_panel_20261009/RESULT.md.\n')
    fh.write('\nOutputs: ' + ', '.join(sorted(os.listdir(OUT))) + '\n')
    fh.write(f'\nChecks run: {len(checks)}, all passed. Each check compares a recomputed quantity (from fold_metrics.csv or the Figure 2/5 tables) '
             'with summary.json / analysis.json / the RESULT.md tables.\n\n| check | got | want | tol |\n|---|---|---|---|\n')
    for name, got, want, tol, ok in checks:
        fh.write(f'| {name} | {float(got):.6g} | {float(want):.6g} | {tol:g} |\n')
print('DONE; checks passed:', len(checks))

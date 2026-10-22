"""PROSPECTIVE_LOWDIM_V1 pre-registered analysis (PLAN.md sections 6-8). Numbers and rule verdicts only; no interpretation.

usage: python analyze.py RUN_DIR
reads  RUN_DIR/output/{drug,crispr,gdsc1}/{new_cell,tissue}/fold-{0..4}/{fold_result.json,per_target.npz}
       RUN_DIR/data_gdsc1/{data_gate.json,name_in_gdsc2.npy}; N6 fold results (T512 rows for the figure table only)
writes RUN_DIR/analysis/{analysis.json, RESULT_AUTO.md, figure_2H_prospective.csv, panel_comparisons.csv, cells_A.csv}
Refuses to compute any verdict for an endpoint unless all ten folds of that endpoint exist and passed their gates.
"""
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import t as tdist

RUN = sys.argv[1]
OUT = f'{RUN}/analysis'
N6_RUN = '/public/home/mengxl/creed-research-runs/settle-stageA-20261011-0001'
SCEN = ('new_cell', 'tissue')
YS = ('RNA', 'P')
RS = (4, 8, 16, 32, 64)
KS = (8, 16, 32, 64)
TAGS = ('P', 'G')
NSEED = 5
EP_LABEL = {'drug': 'Drug response (GDSC2 AUC)', 'crispr': 'CRISPR gene effect', 'gdsc1': 'Drug response (GDSC1 AUC)'}
SC_LABEL = {'new_cell': 'New cell lines', 'tissue': 'Held-out tissue'}
# ---- pre-registered thresholds (PLAN.md sections 7-8); changing any of them after registration voids the verdicts
MIN_ROOM = 0.3            # % ; shares are defined only where the room is >= 0.3 %
H086_SHARE = (0.20, 0.45)  # P2
A6_R4_MIN = 0.60          # P3
A1_R4 = (0.15, 0.45)      # P4
GAP_MIN = 0.30            # P5
R_MIN = 0.50              # P6, R(32) and R(64)
DELTA_NI = 0.25           # B1 non-inferiority margin, percentage points of relative sRMSE
ALPHA_B1, ALPHA_B2 = 0.10, 0.10   # B1 one-sided; B2 two-sided (= one-sided 0.05 in the predicted direction)


def rel(base, arm):
    return (np.asarray(base) - np.asarray(arm)) / np.asarray(base) * 100


def nb(d, ratio):
    """Nadeau-Bengio corrected paired t over k = 5 folds (df = 4)."""
    d = np.asarray(d, float); k = len(d)
    se = float(np.sqrt(d.var(ddof=1) * (1 / k + ratio)))
    m = float(d.mean())
    if se <= 0:
        return dict(mean=m, se=se, t=float('nan'), p_two=float('nan'), p_greater=float('nan'), p_less=float('nan'),
                    lo90_one=float('nan'), hi95_one=float('nan'), ci95=[float('nan')] * 2, wins=int((d > 0).sum()), folds=d.tolist())
    t = m / se
    return dict(mean=m, se=se, t=t, p_two=float(2 * tdist.sf(abs(t), k - 1)), p_greater=float(tdist.sf(t, k - 1)),
                p_less=float(tdist.cdf(t, k - 1)), lo90_one=m - float(tdist.ppf(.90, k - 1)) * se,
                hi95_one=m + float(tdist.ppf(.95, k - 1)) * se,
                ci95=[m - float(tdist.ppf(.975, k - 1)) * se, m + float(tdist.ppf(.975, k - 1)) * se], wins=int((d > 0).sum()), folds=d.tolist())


def load(ep, scen):
    R = []
    for f in range(5):
        p = f'{RUN}/output/{ep}/{scen}/fold-{f}/fold_result.json'
        if not os.path.exists(p):
            return None
        r = json.load(open(p))
        if not r['gate']['passed']:
            return None
        R.append(r)
    S = {}
    for r in R:
        for row in r['rows']:
            S.setdefault(row['arm'], []).append(row['score'])
    S = {a: np.array(v) for a, v in S.items() if len(v) == 5}
    ratio = float(np.mean([r['n_query'] for r in R]) / np.mean([r['n_fit'] for r in R]))
    return dict(R=R, S=S, ratio=ratio)


def rescore(ep, scen, arms, mask_fn):
    """Fold scores of `arms` on a target subset (mask_fn(cols) -> bool), from the saved per-target MSE."""
    out = {a: [] for a in arms}; n = []
    for f in range(5):
        Z = np.load(f'{RUN}/output/{ep}/{scen}/fold-{f}/per_target.npz', allow_pickle=False)
        use = Z['evaluated'] & mask_fn(Z['cols'])
        n.append(int(use.sum()))
        for a in arms:
            out[a].append(float(np.sqrt(Z['mse__' + a][use].mean())))
    return {a: np.array(v) for a, v in out.items()}, n


def rnd_mean(S, pre, k):
    return np.mean([S[f'{pre}_k{k}_s{s}'] for s in range(NSEED)], axis=0)


def cell_A(S, y, ratio, ep):
    """Room, shares and controls of one scenario x borrowed-layer cell."""
    M, meas = S['M'], S[f'CREED_M+{y}']
    room_f = rel(M, meas); room = float(room_f.mean())
    ok = room >= MIN_ROOM
    sh = (lambda v: float(rel(M, v).mean() / room)) if ok else (lambda v: float('nan'))
    c = dict(room=room, room_folds_positive=int((room_f > 0).sum()), room_nb=nb(room_f, ratio), shares_defined=bool(ok),
             h086_share=sh(S[f'H086_{y}']), h086_wins_vs_M=int((rel(M, S[f'H086_{y}']) > 0).sum()),
             a6={r: sh(S[f'A6_{y}_r{r}']) for r in RS}, a1={r: sh(S[f'A1_{y}_r{r}']) for r in RS},
             a6_r4_wins_vs_M=int((rel(M, S[f'A6_{y}_r4']) > 0).sum()))
    if f'H086_{y}_shuf' in S:
        c['h086_wins_vs_shuf'] = int((rel(S[f'H086_{y}_shuf'], S[f'H086_{y}']) > 0).sum())
        c['h086_shuf_share'] = sh(S[f'H086_{y}_shuf'])
    if f'T512_{y}' in S:
        c['t512_share'] = sh(S[f'T512_{y}'])
    a6 = S[f'A6_{y}_r4']
    rand = np.mean([S[f'A6rand_{y}_r4_s{s}'] for s in range(NSEED)], axis=0)
    perm = np.mean([S[f'A6perm_{y}_r4_s{s}'] for s in range(NSEED)], axis=0)
    c['controls_r4'] = dict(pca_share=sh(S[f'A6pca_{y}_r4']), random_share=sh(rand), permuted_share=sh(perm),
                            a6_beats_pca_folds=int((a6 < S[f'A6pca_{y}_r4']).sum()), a6_beats_random_folds=int((a6 < rand).sum()),
                            a6_beats_permuted_folds=int((a6 < perm).sum()),
                            a6_vs_pca=nb(rel(S[f'A6pca_{y}_r4'], a6), ratio), a6_vs_random=nb(rel(rand, a6), ratio),
                            a6_vs_permuted=nb(rel(perm, a6), ratio))
    return c


def r_of_k(S, arm, teacher):
    return float((S['M'].mean() - S[arm].mean()) / (S['M'].mean() - S[teacher].mean()))


def predictions_A(cells, rk):
    """P1-P6 per cell (PLAN.md section 7); a prediction is confirmed in 4/4 cells, partial in 2-3/4, not confirmed in 0-1/4."""
    out = {}
    for key, c in cells.items():
        d = c['shares_defined']
        out[key] = dict(
            P1=bool(c['room'] > 0 and c['room_folds_positive'] >= 4),
            P2=bool(d and H086_SHARE[0] <= c['h086_share'] <= H086_SHARE[1] and c['h086_wins_vs_M'] >= 4
                    and c.get('h086_wins_vs_shuf', 5) >= 4),
            P3=bool(d and c['a6'][4] >= A6_R4_MIN and c['a6_r4_wins_vs_M'] >= 4),
            P4=bool(d and A1_R4[0] <= c['a1'][4] <= A1_R4[1]),
            P5=bool(d and c['a6'][4] - c['a1'][4] >= GAP_MIN))
    for key, v in rk.items():                     # key = scenario/panel layer (P protein, G gene)
        out[f'{key}_panel'] = dict(P6=bool(v['SUP'][32] >= R_MIN and v['SUP'][64] >= R_MIN))
    verdict = {}
    for p in ('P1', 'P2', 'P3', 'P4', 'P5', 'P6'):
        n = sum(v[p] for v in out.values() if p in v); m = sum(p in v for v in out.values())
        verdict[p] = dict(cells_holding=n, cells=m, verdict='confirmed' if n == m else 'partial' if n >= 2 else 'not confirmed')
    core = all(verdict[p]['verdict'] == 'confirmed' for p in ('P1', 'P3', 'P5'))
    return out, verdict, dict(low_dimensional_pattern_replicates=bool(core),
                              n_confirmed=sum(v['verdict'] == 'confirmed' for v in verdict.values()), n_predictions=6)


def panel_B(S, ratio, t, ep):
    """Direction-guided panel vs supervised-sum and random panels (PLAN.md section 8)."""
    per_k, dsup, drnd = {}, [], []
    for k in KS:
        dir_ = S[f'DIR_{t}_k{k}']
        a = rel(S[f'SUP_{t}_k{k}'], dir_); b = rel(rnd_mean(S, f'RND_{t}', k), dir_)
        dsup.append(a); drnd.append(b)
        per_k[k] = dict(dir_vs_sup=nb(a, ratio), dir_vs_rnd=nb(b, ratio),
                        sup_vs_rnd=nb(rel(rnd_mean(S, f'RND_{t}', k), S[f'SUP_{t}_k{k}']), ratio),
                        dir_vs_sup512=nb(rel(S[f'SUP512_{t}_k{k}'], dir_), ratio),
                        dir_vs_rnd512=nb(rel(rnd_mean(S, f'RND512_{t}', k), dir_), ratio))
        if f'DIRx_{t}_k{k}' in S:
            per_k[k]['dirx_vs_dir'] = nb(rel(dir_, S[f'DIRx_{t}_k{k}']), ratio)
    b1 = nb(np.mean(dsup, axis=0), ratio); b2 = nb(np.mean(drnd, axis=0), ratio)
    # PASS = non-inferior (one-sided 90% lower bound above -delta); WORSE = not PASS and significantly lower
    # (one-sided 95% upper bound below 0); INCONCLUSIVE otherwise. significantly_lower is reported for every cell.
    b1['outcome'] = 'PASS' if b1['lo90_one'] > -DELTA_NI else 'WORSE' if b1['hi95_one'] < 0 else 'INCONCLUSIVE'
    b1['significantly_lower'] = bool(b1['hi95_one'] < 0)
    b2['outcome'] = 'PASS' if (b2['mean'] > 0 and b2['wins'] >= 4 and b2['p_two'] < ALPHA_B2) else 'FAIL'
    pooled = {}
    for name, (num, den) in dict(sup_vs_rnd=(f'RND_{t}', f'SUP_{t}'), dir_vs_sup512=(f'SUP512_{t}', f'DIR_{t}'),
                                 dir_vs_rnd512=(f'RND512_{t}', f'DIR_{t}'), sup512_vs_rnd512=(f'RND512_{t}', f'SUP512_{t}')).items():
        dd = []
        for k in KS:
            base = rnd_mean(S, num, k) if num.startswith('RND') else S[f'{num}_k{k}']
            dd.append(rel(base, S[f'{den}_k{k}']))
        pooled[name] = nb(np.mean(dd, axis=0), ratio)
    return dict(B1_dir_vs_sup_pooled=b1, B2_dir_vs_rnd_pooled=b2, pooled_secondary=pooled, per_k=per_k)


def main():
    os.makedirs(OUT, exist_ok=True)
    rep = dict(protocol='PROSPECTIVE_LOWDIM_V1', thresholds=dict(MIN_ROOM=MIN_ROOM, H086_SHARE=H086_SHARE, A6_R4_MIN=A6_R4_MIN, A1_R4=A1_R4,
               GAP_MIN=GAP_MIN, R_MIN=R_MIN, DELTA_NI=DELTA_NI, ALPHA_B1=ALPHA_B1, ALPHA_B2=ALPHA_B2), endpoints={})
    fig_rows, cmp_rows, a_rows = [], [], []
    md = ['# PROSPECTIVE_LOWDIM_V1 result (auto-generated by analyze.py; numbers and rule verdicts only)', '']
    gate_path = f'{RUN}/data_gdsc1/data_gate.json'
    rep['gdsc1_data_gate'] = json.load(open(gate_path)) if os.path.exists(gate_path) else 'missing'
    for ep in ('drug', 'crispr', 'gdsc1'):
        L = {s: load(ep, s) for s in SCEN}
        if any(v is None for v in L.values()):
            rep['endpoints'][ep] = 'not analysed: missing fold or failed gate'
            md += [f'## {ep}: not analysed (missing fold or failed gate)', '']
            continue
        E = dict(cells={}, panels={}, R_of_k={}, diag={})
        for scen in SCEN:
            S, ratio = L[scen]['S'], L[scen]['ratio']
            for y in YS:
                E['cells'][f'{scen}/{y}'] = cell_A(S, y, ratio, ep)
            for t in TAGS:
                teacher = f'TEACHER_{t}'
                E['R_of_k'][f'{scen}/{t}'] = {arm: {k: (r_of_k(S, f'{arm}_{t}_k{k}', teacher) if not arm.startswith('RND') else
                                                        float((S['M'].mean() - rnd_mean(S, f'{arm}_{t}', k).mean()) / (S['M'].mean() - S[teacher].mean())))
                                                    for k in KS} for arm in ('SUP', 'DIR', 'RND', 'SUP512', 'RND512')}
                E['panels'][f'{scen}/{t}'] = panel_B(S, ratio, t, ep)
            # descriptive diagnostics (outer banks, fold means)
            dg = {}
            for y in YS:
                ds = [r['diag']['outer'][y] for r in L[scen]['R']]
                dg[y] = {k: (np.mean([d[k][str(4)] if isinstance(d[k], dict) else d[k] for d in ds], axis=0).tolist())
                         for k in ds[0] if k in ('cos2_crispr_vs_drug', 'cos2_gdsc1_vs_gdsc2', 'splithalf_cos2', 'ev_frac_top')}
            E['diag'][scen] = dg
            # figure table (same schema as figure_2H/lowdim_retention.csv)
            n6 = None
            if ep in ('drug', 'crispr'):
                root = 'output' if ep == 'drug' else 'output_crispr'
                n6 = {}
                for f in range(5):
                    for row in json.load(open(f'{N6_RUN}/{root}/{scen}/fold-{f}/fold_result.json'))['rows']:
                        n6.setdefault(row['arm'], []).append(row['score'])
            for y in YS:
                M = S['M']; room = float(rel(M, S[f'CREED_M+{y}']).mean())
                for q, proj, full in (('measured_retention', 'A6', 'T512'), ('virtual_recovery', 'A1', 'H086')):
                    for r in RS + (512,):
                        arm = f'{proj}_{y}_r{r}' if r != 512 else f'{full}_{y}'
                        v = S.get(arm) if arm in S else (np.array(n6[arm]) if n6 and arm in n6 else None)
                        if v is None:
                            continue
                        g = rel(M, v)
                        fig_rows.append(dict(endpoint=ep, endpoint_label=EP_LABEL[ep], scenario=scen, scenario_label=SC_LABEL[scen], layer=y, r=r,
                                             is_full=int(r == 512), quantity=q, arm=arm, share=round(float(g.mean() / room), 6),
                                             share_pct=round(float(g.mean() / room * 100), 3), gain_pct=round(float(g.mean()), 6), room_pct=round(room, 6),
                                             standard_share=round(float(rel(M, S[f'H086_{y}']).mean() / room), 6),
                                             folds_gain_positive=int((g > 0).sum()), n_folds=5,
                                             source=f'prospective-lowdim-20261022-0001/output/{ep}/{scen}/fold-0..4/fold_result.json'
                                             if arm in S else f'settle-stageA-20261011-0001/{"output" if ep == "drug" else "output_crispr"}/{scen}/fold-0..4/fold_result.json'))
        if ep == 'gdsc1':
            cells, verdict, head = predictions_A(E['cells'], {k: v for k, v in E['R_of_k'].items()})
            E['predictions'] = dict(per_cell=cells, verdict=verdict, headline=head)
            # sensitivity: drugs whose names do not occur in the GDSC2 panel (D6)
            dis = ~np.load(f'{RUN}/data_gdsc1/name_in_gdsc2.npy')
            ev = rep['gdsc1_data_gate']['items']['D6_name_disjoint_sensitivity']['evaluable']
            sens = {}
            if ev:
                for scen in SCEN:
                    arms = ['M'] + [a for y in YS for a in (f'CREED_M+{y}', f'H086_{y}', f'H086_{y}_shuf', f'A6_{y}_r4', f'A1_{y}_r4')]
                    Sd, nsub = rescore('gdsc1', scen, arms, lambda c: dis[c])
                    for y in YS:
                        room_f = rel(Sd['M'], Sd[f'CREED_M+{y}']); room = float(room_f.mean())
                        sh = lambda a: float(rel(Sd['M'], Sd[a]).mean() / room) if room >= MIN_ROOM else float('nan')
                        sens[f'{scen}/{y}'] = dict(drugs_per_fold=nsub, room=room, room_folds_positive=int((room_f > 0).sum()),
                                                   h086_share=sh(f'H086_{y}'), a6_r4=sh(f'A6_{y}_r4'), a1_r4=sh(f'A1_{y}_r4'))
            E['sensitivity_name_disjoint'] = sens if ev else 'not evaluable (D6)'
        else:
            E['predictions'] = 'reference endpoint: shares and controls reported, P1-P6 not applied'
        # panel verdicts (primary for drug and CRISPR; GDSC1 descriptive)
        b1 = {k: v['B1_dir_vs_sup_pooled']['outcome'] for k, v in E['panels'].items()}
        b2 = {k: v['B2_dir_vs_rnd_pooled']['outcome'] for k, v in E['panels'].items()}
        E['panel_outcomes'] = dict(B1=b1, B2=b2, B2_direction_positive={k: v['B2_dir_vs_rnd_pooled']['mean'] > 0 for k, v in E['panels'].items()})
        rep['endpoints'][ep] = E
        # tables
        md += [f'## {EP_LABEL[ep]}', '', '| cell | room % (folds +) | standard share | A6 r=4 | A1 r=4 | A6 r=8 | A6 r=16 | PCA-4 | random-4 | permuted-4 | A6 beats pca/random/perm (folds) |',
               '|---|---|---|---|---|---|---|---|---|---|---|']
        for key, c in E['cells'].items():
            k4 = c['controls_r4']
            md.append(f"| {key} | {c['room']:+.3f} ({c['room_folds_positive']}/5) | {c['h086_share']:.3f} | {c['a6'][4]:.3f} | {c['a1'][4]:.3f} | "
                      f"{c['a6'][8]:.3f} | {c['a6'][16]:.3f} | {k4['pca_share']:.3f} | {k4['random_share']:.3f} | {k4['permuted_share']:.3f} | "
                      f"{k4['a6_beats_pca_folds']}/{k4['a6_beats_random_folds']}/{k4['a6_beats_permuted_folds']} |")
            a_rows.append(dict(endpoint=ep, cell=key, room=c['room'], room_folds_positive=c['room_folds_positive'], standard_share=c['h086_share'],
                               **{f'A6_r{r}': c['a6'][r] for r in RS}, **{f'A1_r{r}': c['a1'][r] for r in RS},
                               pca4=k4['pca_share'], random4=k4['random_share'], permuted4=k4['permuted_share']))
        md += ['', '| panel cell | B1 DIR vs SUP pooled (mean, NB se, one-sided 90% lower bound) | B1 | B2 DIR vs random pooled (mean, folds, two-sided p) | B2 | R(32) SUP / DIR / random |',
               '|---|---|---|---|---|---|']
        for key, v in E['panels'].items():
            a, b = v['B1_dir_vs_sup_pooled'], v['B2_dir_vs_rnd_pooled']; R = E['R_of_k'][key]
            md.append(f"| {key} | {a['mean']:+.3f}, {a['se']:.3f}, {a['lo90_one']:+.3f} | {a['outcome']} | {b['mean']:+.3f}, {b['wins']}/5, {b['p_two']:.3f} | "
                      f"{b['outcome']} | {R['SUP'][32]:.3f} / {R['DIR'][32]:.3f} / {R['RND'][32]:.3f} |")
            for k, pk in v['per_k'].items():
                for comp, r_ in pk.items():
                    cmp_rows.append(dict(endpoint=ep, cell=key, k=k, comparison=comp, mean=r_['mean'], se=r_['se'], t=r_['t'], p_two=r_['p_two'],
                                         wins=r_['wins'], ci95_lo=r_['ci95'][0], ci95_hi=r_['ci95'][1]))
            for comp, r_ in [('B1_dir_vs_sup', a), ('B2_dir_vs_rnd', b)] + list(v['pooled_secondary'].items()):
                cmp_rows.append(dict(endpoint=ep, cell=key, k='pooled', comparison=comp, mean=r_['mean'], se=r_['se'], t=r_['t'], p_two=r_['p_two'],
                                     wins=r_['wins'], ci95_lo=r_['ci95'][0], ci95_hi=r_['ci95'][1]))
        if ep == 'gdsc1':
            md += ['', '### GDSC1 predictions', '', '```', json.dumps(E['predictions']['verdict'], indent=1), json.dumps(E['predictions']['headline']), '```']
        md.append('')
    # overall panel verdict over drug + CRISPR (PLAN.md section 8.3)
    if all(isinstance(rep['endpoints'].get(e), dict) for e in ('drug', 'crispr')):
        b1 = [rep['endpoints'][e]['panel_outcomes']['B1'][k] for e in ('drug', 'crispr') for k in rep['endpoints'][e]['panel_outcomes']['B1']]
        b2d = [rep['endpoints']['drug']['panel_outcomes']['B2'][f'{s}/P'] for s in SCEN]
        dirpos = [v for e in ('drug', 'crispr') for v in rep['endpoints'][e]['panel_outcomes']['B2_direction_positive'].values()]
        rep['panel_verdict'] = dict(
            B1=('worse than supervised' if 'WORSE' in b1 else 'not worse than supervised (confirmed)' if b1.count('PASS') >= 4 else 'inconclusive (underpowered)'),
            B1_counts={o: b1.count(o) for o in ('PASS', 'INCONCLUSIVE', 'WORSE')},
            B2=('better than random where predicted (confirmed)' if all(x == 'PASS' for x in b2d) else 'not confirmed'),
            B2_drug_protein=b2d, B2_direction_positive_cells=int(sum(dirpos)), B2_direction_confirmed=bool(sum(dirpos) >= 6))
        md += ['## Panel verdict (drug + CRISPR)', '', '```', json.dumps(rep['panel_verdict'], indent=1), '```']
    # S1 (PLAN 7.3): A6 r=4 beats the random-subspace and the label-permuted mean in >= 4/5 folds; confirmed in >= 10/12 cells
    s1 = {f'{e}/{k}': bool(c['controls_r4']['a6_beats_random_folds'] >= 4 and c['controls_r4']['a6_beats_permuted_folds'] >= 4)
          for e, v in rep['endpoints'].items() if isinstance(v, dict) for k, c in v['cells'].items()}
    rep['S1_controls'] = dict(per_cell=s1, cells_holding=int(sum(s1.values())), cells=len(s1),
                              verdict='confirmed' if len(s1) == 12 and sum(s1.values()) >= 10 else 'not confirmed' if len(s1) == 12 else 'incomplete')
    s2 = {f'{e}/{k}': c['controls_r4']['a6_beats_pca_folds'] for e, v in rep['endpoints'].items() if isinstance(v, dict) for k, c in v['cells'].items()}
    rep['S2_pca_folds_won'] = s2
    md += ['## S1 / S2 (measured-layer controls at r = 4)', '', '```', json.dumps(dict(S1=rep['S1_controls'], S2_folds_A6_beats_PCA=s2), indent=1), '```']
    json.dump(rep, open(f'{OUT}/analysis.json', 'w'), indent=1, default=float)
    open(f'{OUT}/RESULT_AUTO.md', 'w').write('\n'.join(md) + '\n')
    for name, rows in (('figure_2H_prospective.csv', fig_rows), ('panel_comparisons.csv', cmp_rows), ('cells_A.csv', a_rows)):
        if rows:
            with open(f'{OUT}/{name}', 'w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps({e: (v['predictions']['headline'] if isinstance(v, dict) and isinstance(v.get('predictions'), dict) else
                          v if isinstance(v, str) else 'reference') for e, v in rep['endpoints'].items()}, default=float))
    print(json.dumps(rep.get('panel_verdict', 'panel verdict not computed'), default=float))


if __name__ == '__main__':
    main()

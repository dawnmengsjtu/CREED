#!/usr/bin/env python
"""Figure 8 (V2; FIG id 9 in the figures2_6 pipeline) data stage:
one-shot confirmation on the sealed internal-final set (123 cell lines, protocol CREED_INTERNAL_FINAL_ONESHOT_V1).

Reads ONLY (read-only):
  /public/home/mengxl/creed-research-runs/internal-final-20261009-0001/output/{summary.json, RESULT_DRAFT.md, progress.jsonl}
      (never output/arms/*.npz, never any raw file)
  <V2>/delivery_R_20261009/figures2_6/figure7/plot_data/{a_map.csv, c_panel_curve.csv}
      (development-fold values already plotted in Figure 7; used ONLY to cross-check the development constants below)
Writes ONLY: <V2>/delivery_R_20261009/figures2_6/figure9/plot_data/*.csv|json + FIG8_DATA_SOURCES.md

Ground truth for every SEALED number printed on the figure: summary.json (analysis.drug / analysis.crispr / analysis.F,
meta.gates.crispr_123.lines_with_crispr = 95, receipt.written_at). RESULT_DRAFT.md and progress.jsonl are parsed only as
independent cross-checks of the same numbers (the runner wrote them in the same invocation).
Development-fold constants (hollow symbols) are the values of the manuscript's Table R8-1 / Fig. 8 legend
(manuscript_v0/CREED_manuscript_v0.md), whose sources are crosslayer_matrix_20261009/RESULT.md section 3 (new cell lines,
T row), pushline_checks_20261009/RESULT.md section 3 (P1 row), protein_panel_20261009/RESULT.md section 8,
rna_panel_20261009/RESULT.md sections 0/4 and crispr_endpoint_20261009/RESULT.md sections 4/5. Those that also appear in the
Figure 7 tables are asserted against them here.

Conventions: gain% = 100*(sRMSE[M] - sRMSE[arm])/sRMSE[M] on the single sealed set (no folds); intervals = grouped bootstrap
2.5th/97.5th percentiles (B = 2000, seed 20261009; cells within tissue strata x drugs/targets by identifier, same draws for every
arm); share = gain(M+vY)/gain(M+Y); R(k) = gain(M+k panel)/gain(teacher), teacher M+Pval (proteins) / M+RNA (genes).
"""
import hashlib, json, os, re
import pandas as pd

RUN = '/public/home/mengxl/creed-research-runs/internal-final-20261009-0001/output'
V2 = '/public/home/mengxl/workspaces/creed-research/figure_work/20261009_figures2_6_v2/delivery_R_20261009/figures2_6'
F7 = f'{V2}/figure7/plot_data'
OUT = f'{V2}/figure9/plot_data'
os.makedirs(OUT, exist_ok=True)
inputs, checks = [], []


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''):
            h.update(ch)
    return h.hexdigest()


def rj(p):
    inputs.append(p)
    return json.load(open(p))


def rd(p, **kw):
    inputs.append(p)
    return pd.read_csv(p, **kw)


def rt(p):
    inputs.append(p)
    return open(p, encoding='utf-8').read()


def wr(df, name):
    df.to_csv(os.path.join(OUT, name + '.csv'), index=False)
    print('wrote', name, df.shape)


def check(name, got, want, tol):
    ok = abs(float(got) - float(want)) <= tol
    checks.append((name, got, want, tol, ok))
    if not ok:
        raise AssertionError(f'CHECK FAILED {name}: got {got} want {want} (tol {tol})')


S = rj(f'{RUN}/summary.json')
A = S['analysis']
DRAFT = rt(f'{RUN}/RESULT_DRAFT.md')
PROG = [json.loads(l) for l in open(f'{RUN}/progress.jsonl', encoding='utf-8') if l.strip()]
inputs.append(f'{RUN}/progress.jsonl')
AMAP = rd(f'{F7}/a_map.csv')
CURVE = rd(f'{F7}/c_panel_curve.csv')

# ------------------------------------------------------------------ development-fold constants (manuscript Table R8-1 / legend)
DEV_GAIN = {'drug': {'M+vRNA': 1.056, 'M+vP': 0.700, 'M+T': 1.485, 'M+P_graph': 0.433, 'M+vRNA_shuf': -0.205, 'M+vP_shuf': -0.152},
            'crispr': {'M+vRNA': 0.651, 'M+vP': 0.388}}
DEV_SRC = {'drug': {'M+vRNA': 'crosslayer_matrix RESULT.md s3 new cell (Table R8-1 F1)', 'M+vP': 'crosslayer_matrix RESULT.md s3 new cell (Table R8-1 F2)',
                    'M+T': 'crosslayer_matrix RESULT.md s3 T row (Table R8-1 F6)', 'M+P_graph': 'pushline_checks RESULT.md s3 P1 row, ratio of fold means (Table R8-1 F7)',
                    'M+vRNA_shuf': 'crosslayer_matrix RESULT.md s3 new cell, 1/5 folds (Table R8-1 F3)', 'M+vP_shuf': 'crosslayer_matrix RESULT.md s3 new cell, 1/5 folds (Table R8-1 F3)'},
           'crispr': {'M+vRNA': 'crispr_endpoint RESULT.md s4 new cell (Table R8-1 F8)', 'M+vP': 'crispr_endpoint RESULT.md s4 new cell (Table R8-1 F8)'}}
DEV_SHARE = {'drug': {'RNA': 0.34, 'P': 0.29}, 'crispr': {'RNA': 0.34, 'P': 0.32}}
DEV_ROOM = {'drug': {'RNA': 3.079, 'P': 2.401}, 'crispr': {'RNA': 1.934, 'P': 1.225}}
DEV_R = {'drug': {('protein', 16): 0.557, ('protein', 32): 0.648, ('protein', 64): 0.716, ('gene', 16): 0.56, ('gene', 32): 0.717, ('gene', 64): 0.764},
         'crispr': {('protein', 32): 0.606, ('protein', 64): 0.748, ('gene', 32): 0.560, ('gene', 64): 0.713}}
BAND = (0.15, 0.55)
R_CRIT = 0.5

# cross-check the development constants that exist in the Figure 7 tables (GDSC = drug endpoint, new cell lines)
for ep, epl in (('drug', 'GDSC'), ('crispr', 'CRISPR')):
    for lay in ('RNA', 'P'):
        r = AMAP[(AMAP.endpoint == epl) & (AMAP.scenario == 'new_cell') & (AMAP.deployed == 'M') & (AMAP.borrowed == lay)].iloc[0]
        check(f'dev {ep} M+v{lay} gain == figure7 a_map recovered_gain', DEV_GAIN[ep]['M+v' + lay], r.recovered_gain, 5.01e-4)
        check(f'dev {ep} share {lay} == figure7 a_map share', DEV_SHARE[ep][lay], r.share, 5.01e-3)
        check(f'dev {ep} room {lay} == figure7 a_map room_gain', DEV_ROOM[ep][lay], r.room_gain, 5.01e-4)
    for (lay, k), v in DEV_R[ep].items():
        r = CURVE[(CURVE.endpoint == epl) & (CURVE.scenario == 'new_cell') & (CURVE.layer == lay) & (CURVE.k == k)].iloc[0]
        check(f'dev {ep} R({k}) {lay} == figure7 c_panel_curve', v, r.R, 5.01e-3 if v in (0.56,) else 5.01e-4)

# ------------------------------------------------------------------ parse RESULT_DRAFT.md arm tables (independent cross-check)
def parse_draft_tables(txt):
    out = {}
    sec = None
    for line in txt.splitlines():
        if line.startswith('## 药敏'):
            sec = 'drug'
        elif line.startswith('## CRISPR'):
            sec = 'crispr'
        elif line.startswith('## '):
            sec = None
        if sec and line.startswith('| ') and not line.startswith('| 臂') and '---' not in line:
            c = [x.strip() for x in line.strip().strip('|').split('|')]
            if len(c) != 8:
                continue
            arm = c[0]
            d = dict(srmse=float(c[1]))
            m = re.match(r'\[([-\d.]+), ([-\d.]+)\]', c[2])
            d['srmse_lo'], d['srmse_hi'] = float(m.group(1)), float(m.group(2))
            if c[5] != '—':
                d['gain'] = float(c[5])
                m = re.match(r'\[([-\d.]+), ([-\d.]+)\] P\(>0\)=([\d.]+)', c[6])
                d['lo'], d['hi'], d['p'] = float(m.group(1)), float(m.group(2)), float(m.group(3))
                m = re.match(r'(\d+)/(\d+)/(\d+)', c[7])
                d['wins'], d['losses'], d['ties'] = int(m.group(1)), int(m.group(2)), int(m.group(3))
            out.setdefault(sec, {})[arm] = d
    return out


DT = parse_draft_tables(DRAFT)
assert len(DT['drug']) == 39 and len(DT['crispr']) == 35, {k: len(v) for k, v in DT.items()}
PS = {(p['endpoint'], p['arm']): p['score'] for p in PROG}
assert len(PS) == 74

# ------------------------------------------------------------------ arm rows for the two forest plots
GROUPS = [  # key, order, label (header printed on the forest), deployable
    ('tissue', 1, 'Tissue label', True),
    ('virtual', 2, 'Free virtual layers (generated from M; no new measurement)', True),
    ('shuffle', 3, 'Paired-shuffle controls (same generator, pairing broken)', True),
    ('graph', 4, 'Training-time protein graph (M-only deployment input)', True),
    ('panel', 5, 'Measured panels at deployment (k molecules + M)', True),
    ('reference', 6, 'Measured full layer (reference, not deployable)', False)]
GL = {g[0]: g for g in GROUPS}
ARMS = {  # arm key -> (group, row order within group, label)
    'M+T': ('tissue', 1, 'M + tissue label'), 'T': ('tissue', 2, 'Tissue label alone'),
    'M+vRNA': ('virtual', 1, 'M + virtual RNA'), 'M+vP': ('virtual', 2, 'M + virtual protein'), 'M+vRNA+T': ('virtual', 3, 'M + virtual RNA + tissue'),
    'M+vRNA_shuf': ('shuffle', 1, 'M + shuffled virtual RNA'), 'M+vP_shuf': ('shuffle', 2, 'M + shuffled virtual protein'),
    'M+P_graph': ('graph', 1, 'M + protein graph'),
    'M+P16': ('panel', 1, '16 proteins, supervised'), 'M+P32': ('panel', 2, '32 proteins, supervised'), 'M+P64': ('panel', 3, '64 proteins, supervised'),
    'M+G16': ('panel', 4, '16 genes, supervised'), 'M+G32': ('panel', 5, '32 genes, supervised'), 'M+G64': ('panel', 6, '64 genes, supervised'),
    'M+Prand16_mean': ('panel', 7, '16 proteins, random (5 seeds)'), 'M+Prand32_mean': ('panel', 8, '32 proteins, random (5 seeds)'),
    'M+Grand16_mean': ('panel', 9, '16 genes, random (5 seeds)'), 'M+Grand32_mean': ('panel', 10, '32 genes, random (5 seeds)'),
    'M+RNA': ('reference', 1, 'M + measured RNA'), 'M+P': ('reference', 2, 'M + measured protein'), 'M+Pval': ('reference', 3, 'M + protein values (teacher)')}
SERIES = {'M+P16': ('protein', 'supervised'), 'M+P32': ('protein', 'supervised'), 'M+P64': ('protein', 'supervised'),
          'M+G16': ('gene', 'supervised'), 'M+G32': ('gene', 'supervised'), 'M+G64': ('gene', 'supervised'),
          'M+Prand16_mean': ('protein', 'random'), 'M+Prand32_mean': ('protein', 'random'), 'M+Grand16_mean': ('gene', 'random'), 'M+Grand32_mean': ('gene', 'random')}


def fmt_n(n):
    return f'{n:,}'


rows, seeds = [], []
for ep in ('drug', 'crispr'):
    E = A[ep]
    n_t = E['n_targets_evaluated']
    n_cells = E['n_eval_cells'] if ep == 'drug' else S['meta']['gates']['crispr_123']['lines_with_crispr']
    M = E['arms']['M']
    check(f'{ep} M score == progress.jsonl', M['score'], PS[(ep, 'M')], 0)
    check(f'{ep} M score == RESULT_DRAFT', M['score'], DT[ep]['M']['srmse'], 5e-6)
    for arm, (grp, ro, lab) in ARMS.items():
        if arm not in E['arms'] and arm not in E['panel_recovery']:
            continue
        g = GL[grp]
        base = dict(endpoint=ep, endpoint_label='GDSC drug response (AUC)' if ep == 'drug' else 'CRISPR gene effect', n_cells=n_cells, n_targets=n_t,
                    group=grp, group_order=g[1], group_label=g[2], deployable=g[3], row_order=ro, arm=arm, arm_label=lab,
                    layer=SERIES.get(arm, ('', ''))[0], panel_series=SERIES.get(arm, ('', ''))[1],
                    dev_gain_pct=DEV_GAIN[ep].get(arm), dev_source=DEV_SRC[ep].get(arm, ''))
        if arm.endswith('_mean'):
            pr = E['panel_recovery'][arm]
            stem = arm.replace('_mean', '')
            sl = []
            for s in range(pr['seeds']):
                sa = E['arms'][f'{stem}_s{s}']
                sl.append(sa)
                seeds.append(dict(endpoint=ep, group=grp, mean_arm=arm, arm=f'{stem}_s{s}', seed=s, srmse=sa['score'], gain_pct=sa['gain_vs_M_pct'],
                                  lo=sa['gain_vs_M_boot']['lo'], hi=sa['gain_vs_M_boot']['hi'], prob_gt0=sa['gain_vs_M_boot']['prob_gt0'],
                                  wins=sa['sign_test_vs_M']['wins'], losses=sa['sign_test_vs_M']['losses'], ties=sa['sign_test_vs_M']['ties']))
                check(f'{ep} {stem}_s{s} score == progress.jsonl', sa['score'], PS[(ep, f'{stem}_s{s}')], 0)
                dd = DT[ep][f'{stem}_s{s}']
                check(f'{ep} {stem}_s{s} gain == RESULT_DRAFT', sa['gain_vs_M_pct'], dd['gain'], 5.01e-4)
                check(f'{ep} {stem}_s{s} wins == RESULT_DRAFT', sa['sign_test_vs_M']['wins'], dd['wins'], 0)
            mean_score = sum(x['score'] for x in sl) / len(sl)
            check(f'{ep} {arm} score_mean_over_seeds == mean of seed scores', pr['score_mean_over_seeds'], mean_score, 1e-12)
            gain_from_mean = 100 * (M['score'] - pr['score_mean_over_seeds']) / M['score']
            check(f'{ep} {arm} gain_vs_M_pct == 100*(M - mean score)/M', pr['gain_vs_M_pct'], gain_from_mean, 1e-9)
            w = [x['sign_test_vs_M']['wins'] for x in sl]
            rows.append(dict(base, kind='random_mean', srmse=pr['score_mean_over_seeds'], srmse_lo=None, srmse_hi=None,
                             gain_pct=pr['gain_vs_M_pct'], lo=None, hi=None, prob_gt0=None, median=None, wins=None, losses=None, ties=None,
                             wins_min=min(w), wins_max=max(w), seed_gain_min=min(x['gain_vs_M_pct'] for x in sl), seed_gain_max=max(x['gain_vs_M_pct'] for x in sl),
                             wins_label=f'{min(w)}–{max(w)}/{fmt_n(n_t)}', significant=None, worse=None,
                             R=pr['R'], R_lo=pr['R_boot']['lo'], R_hi=pr['R_boot']['hi'], teacher=pr['teacher']))
        else:
            a = E['arms'][arm]
            cmp_key = f'{arm}_vs_M'
            if cmp_key in E['comparisons']:
                c = E['comparisons'][cmp_key]
                gain, boot, st = c['gain_pct'], c['boot'], c['sign_test']
                check(f'{ep} {arm} comparisons.gain == arms.gain_vs_M_pct', gain, a['gain_vs_M_pct'], 1e-12)
                check(f'{ep} {arm} comparisons.lo == arms.gain_vs_M_boot.lo', boot['lo'], a['gain_vs_M_boot']['lo'], 1e-12)
                check(f'{ep} {arm} comparisons.wins == arms.sign_test_vs_M.wins', st['wins'], a['sign_test_vs_M']['wins'], 0)
            else:
                gain, boot, st = a['gain_vs_M_pct'], a['gain_vs_M_boot'], a['sign_test_vs_M']
            check(f'{ep} {arm} gain == 100*(M-arm)/M', gain, 100 * (M['score'] - a['score']) / M['score'], 1e-9)
            check(f'{ep} {arm} score == progress.jsonl', a['score'], PS[(ep, arm)], 0)
            dd = DT[ep][arm]
            check(f'{ep} {arm} score == RESULT_DRAFT', a['score'], dd['srmse'], 5e-6)
            check(f'{ep} {arm} gain == RESULT_DRAFT', gain, dd['gain'], 5.01e-4)
            check(f'{ep} {arm} lo == RESULT_DRAFT', boot['lo'], dd['lo'], 5.01e-3)
            check(f'{ep} {arm} hi == RESULT_DRAFT', boot['hi'], dd['hi'], 5.01e-3)
            check(f'{ep} {arm} P(>0) == RESULT_DRAFT', boot['prob_gt0'], dd['p'], 5.01e-4)
            check(f'{ep} {arm} wins == RESULT_DRAFT', st['wins'], dd['wins'], 0)
            check(f'{ep} {arm} losses == RESULT_DRAFT', st['losses'], dd['losses'], 0)
            check(f'{ep} {arm} ties == RESULT_DRAFT', st['ties'], dd['ties'], 0)
            check(f'{ep} {arm} wins+losses+ties == n_targets', st['wins'] + st['losses'] + st['ties'], n_t, 0)
            wl = f'{fmt_n(st["wins"])}/{fmt_n(n_t)}' + (f' ({st["ties"]} ties)' if st['ties'] else '')
            pr = E['panel_recovery'].get(arm)
            rows.append(dict(base, kind='arm', srmse=a['score'], srmse_lo=a['boot_score']['lo'], srmse_hi=a['boot_score']['hi'],
                             gain_pct=gain, lo=boot['lo'], hi=boot['hi'], prob_gt0=boot['prob_gt0'], median=boot['median'],
                             wins=st['wins'], losses=st['losses'], ties=st['ties'], wins_min=None, wins_max=None, seed_gain_min=None, seed_gain_max=None,
                             wins_label=wl, significant=boot['lo'] > 0, worse=boot['hi'] < 0,
                             R=pr['R'] if pr else None, R_lo=pr['R_boot']['lo'] if pr else None, R_hi=pr['R_boot']['hi'] if pr else None, teacher=pr['teacher'] if pr else ''))
forest = pd.DataFrame(rows)
wr(forest, 'forest')
wr(pd.DataFrame(seeds), 'forest_seeds')
# a few literal spot checks against the manuscript legend numbers (Fig. 8 legend / Table R8-1)
fd = forest.set_index(['endpoint', 'arm'])
for ep, arm, g, lo, hi in [('drug', 'M+vRNA', 0.855, 0.26, 1.41), ('drug', 'M+vP', 0.504, 0.03, 0.95), ('drug', 'M+T', 1.854, 0.99, 2.74),
                           ('drug', 'M+P_graph', 0.436, 0.08, 0.80), ('drug', 'M+vRNA_shuf', 0.064, -0.22, 0.32), ('drug', 'M+vP_shuf', -0.062, -0.39, 0.24),
                           ('drug', 'M+RNA', 4.482, 3.33, 5.83), ('drug', 'M+P', 3.670, 2.73, 4.78), ('drug', 'M+Pval', 3.972, 2.99, 5.17), ('drug', 'T', 1.199, 0.07, 2.29),
                           ('drug', 'M+vRNA+T', 1.966, 1.09, 2.87), ('crispr', 'M+vRNA', 0.703, 0.51, 0.92), ('crispr', 'M+vP', 0.577, 0.41, 0.76),
                           ('crispr', 'M+vRNA_shuf', -0.062, -0.15, 0.03), ('crispr', 'M+vP_shuf', -0.094, -0.19, -0.00), ('crispr', 'M+RNA', 2.456, 2.08, 2.86),
                           ('crispr', 'M+P', 1.626, 1.28, 1.96), ('crispr', 'M+Pval', 1.698, 1.34, 2.05)]:
    check(f'legend {ep} {arm} gain', fd.loc[(ep, arm), 'gain_pct'], g, 5.01e-4)
    check(f'legend {ep} {arm} lo', fd.loc[(ep, arm), 'lo'], lo, 5.01e-3)
    check(f'legend {ep} {arm} hi', fd.loc[(ep, arm), 'hi'], hi, 5.01e-3)
for ep, arm, w in [('drug', 'M+vRNA', 214), ('drug', 'M+vP', 195), ('drug', 'M+T', 201), ('drug', 'M+P_graph', 177), ('crispr', 'M+vRNA', 1723), ('crispr', 'M+vP', 1700)]:
    check(f'legend {ep} {arm} wins', fd.loc[(ep, arm), 'wins'], w, 0)
check('legend drug M+T ties', fd.loc[('drug', 'M+T'), 'ties'], 46, 0)
check('legend drug M+P_graph ties', fd.loc[('drug', 'M+P_graph'), 'ties'], 21, 0)
for ep, arm, g in [('drug', 'M+P16', 1.26), ('drug', 'M+P32', 1.95), ('drug', 'M+P64', 1.94), ('drug', 'M+G16', 2.06), ('drug', 'M+G32', 2.70), ('drug', 'M+G64', 2.95),
                   ('crispr', 'M+P16', 0.91), ('crispr', 'M+P32', 1.13), ('crispr', 'M+P64', 1.34), ('crispr', 'M+G16', 1.23), ('crispr', 'M+G32', 1.54), ('crispr', 'M+G64', 1.72)]:
    check(f'legend {ep} {arm} gain (2 dp)', fd.loc[(ep, arm), 'gain_pct'], g, 5.01e-3)

# ------------------------------------------------------------------ panel b (and c caption): recovered share
sh = []
for ep in ('drug', 'crispr'):
    for lay in ('RNA', 'P'):
        s = A[ep]['share'][lay]
        c_room = A[ep]['comparisons'][f'M+{lay}_vs_M']['gain_pct']
        c_rec = A[ep]['comparisons'][f'M+v{lay}_vs_M']['gain_pct']
        check(f'{ep} share {lay} room == M+{lay} gain', s['room_pct'], c_room, 1e-12)
        check(f'{ep} share {lay} recovered == M+v{lay} gain', s['recovered_pct'], c_rec, 1e-12)
        check(f'{ep} share {lay} == recovered/room', s['share'], s['recovered_pct'] / s['room_pct'], 1e-12)
        m = re.search(rf'- M→{lay}: room ([\d.]+)%，recovered ([\d.]+)%，share ([\d.]+) \[([\d.]+), ([\d.]+)\]', DRAFT.split('## CRISPR')[0 if ep == 'drug' else 1])
        check(f'{ep} share {lay} == RESULT_DRAFT', s['share'], float(m.group(3)), 5.01e-4)
        check(f'{ep} share {lay} lo == RESULT_DRAFT', s['share_boot']['lo'], float(m.group(4)), 5.01e-3)
        check(f'{ep} share {lay} hi == RESULT_DRAFT', s['share_boot']['hi'], float(m.group(5)), 5.01e-3)
        in_band = BAND[0] <= s['share'] <= BAND[1]
        if ep == 'drug':
            check(f'F4 share_in_range {lay} == summary', in_band, A['F']['F4']['share_in_range'][lay], 0)
        sh.append(dict(endpoint=ep, layer=lay, layer_label={'RNA': 'RNA', 'P': 'Protein'}[lay], pair=f'M→{lay}', room_pct=s['room_pct'], recovered_pct=s['recovered_pct'],
                       share=s['share'], lo=s['share_boot']['lo'], hi=s['share_boot']['hi'], prob_gt0=s['share_boot']['prob_gt0'], median=s['share_boot']['median'],
                       dev_share=DEV_SHARE[ep][lay], dev_room_pct=DEV_ROOM[ep][lay], dev_recovered_pct=DEV_GAIN[ep][f'M+v{lay}'],
                       band_lo=BAND[0], band_hi=BAND[1], in_band=in_band, distance_to_band=(0 if in_band else (BAND[0] - s['share'] if s['share'] < BAND[0] else s['share'] - BAND[1]))))
share = pd.DataFrame(sh)
wr(share, 'b_share')
check('legend drug share RNA', share.query('endpoint=="drug" and layer=="RNA"').share.iloc[0], 0.191, 5.01e-4)
check('legend drug share P', share.query('endpoint=="drug" and layer=="P"').share.iloc[0], 0.137, 5.01e-4)
check('legend crispr share RNA', share.query('endpoint=="crispr" and layer=="RNA"').share.iloc[0], 0.286, 5.01e-4)
check('legend crispr share P', share.query('endpoint=="crispr" and layer=="P"').share.iloc[0], 0.355, 5.01e-4)
check('task text drug share P lo 0.010', share.query('endpoint=="drug" and layer=="P"').lo.iloc[0], 0.010, 5.01e-4)
check('task text drug share P hi 0.250', share.query('endpoint=="drug" and layer=="P"').hi.iloc[0], 0.250, 5.01e-4)
check('task text drug share RNA lo 0.061', share.query('endpoint=="drug" and layer=="RNA"').lo.iloc[0], 0.061, 5.01e-4)
check('task text drug share RNA hi 0.304', share.query('endpoint=="drug" and layer=="RNA"').hi.iloc[0], 0.304, 5.01e-4)

# ------------------------------------------------------------------ panel b (right) and c caption: panel recovery R(k)
pr_rows = []
for ep in ('drug', 'crispr'):
    E = A[ep]
    M = E['arms']['M']
    for lay, stem, rstem, teacher in (('protein', 'M+P', 'M+Prand', 'M+Pval'), ('gene', 'M+G', 'M+Grand', 'M+RNA')):
        tg = E['comparisons'][f'{teacher}_vs_M']['gain_pct']
        for k in (16, 32, 64):
            p = E['panel_recovery'][f'{stem}{k}']
            assert p['teacher'] == teacher
            check(f'{ep} {stem}{k} teacher_gain == {teacher} gain', p['teacher_gain_pct'], tg, 1e-12)
            check(f'{ep} {stem}{k} R == gain/teacher gain', p['R'], E['comparisons'][f'{stem}{k}_vs_M']['gain_pct'] / tg, 1e-9)
            m = re.search(rf'- {re.escape(stem)}{k}（教师 {re.escape(teacher)}，教师增益 [\d.]+%）: R = ([\d.]+) \[([\d.]+), ([\d.]+)\]', DRAFT.split('## CRISPR')[0 if ep == 'drug' else 1])
            check(f'{ep} {stem}{k} R == RESULT_DRAFT', p['R'], float(m.group(1)), 5.01e-4)
            check(f'{ep} {stem}{k} R lo == RESULT_DRAFT', p['R_boot']['lo'], float(m.group(2)), 5.01e-3)
            check(f'{ep} {stem}{k} R hi == RESULT_DRAFT', p['R_boot']['hi'], float(m.group(3)), 5.01e-3)
            pr_rows.append(dict(endpoint=ep, layer=lay, layer_label={'protein': 'Protein panel', 'gene': 'Gene panel'}[lay], k=k, series='supervised', arm=f'{stem}{k}',
                                teacher=teacher, teacher_gain_pct=tg, gain_pct=E['comparisons'][f'{stem}{k}_vs_M']['gain_pct'], srmse=E['arms'][f'{stem}{k}']['score'],
                                R=p['R'], lo=p['R_boot']['lo'], hi=p['R_boot']['hi'], prob_gt0=p['R_boot']['prob_gt0'],
                                dev_R=DEV_R[ep].get((lay, k)), criterion_applies=k in (32, 64), criterion=R_CRIT, passes=(p['R'] >= R_CRIT) if k in (32, 64) else None))
            if k in (16, 32):
                q = E['panel_recovery'][f'{rstem}{k}_mean']
                check(f'{ep} {rstem}{k}_mean R == gain_vs_M/teacher gain', q['R'], q['gain_vs_M_pct'] / tg, 1e-9)
                pr_rows.append(dict(endpoint=ep, layer=lay, layer_label={'protein': 'Protein panel', 'gene': 'Gene panel'}[lay], k=k, series='random_mean', arm=f'{rstem}{k}_mean',
                                    teacher=teacher, teacher_gain_pct=tg, gain_pct=q['gain_vs_M_pct'], srmse=q['score_mean_over_seeds'],
                                    R=q['R'], lo=q['R_boot']['lo'], hi=q['R_boot']['hi'], prob_gt0=q['R_boot']['prob_gt0'],
                                    dev_R=None, criterion_applies=False, criterion=R_CRIT, passes=None))
panelR = pd.DataFrame(pr_rows)
wr(panelR, 'b_panel_R')
pq = panelR.set_index(['endpoint', 'layer', 'k', 'series'])
for ep, lay, k, v, lo, hi in [('drug', 'protein', 32, 0.490, 0.283, 0.682), ('drug', 'protein', 64, 0.489, 0.282, 0.653), ('drug', 'gene', 32, 0.603, 0.460, 0.752),
                              ('drug', 'gene', 64, 0.659, 0.536, 0.781), ('drug', 'protein', 16, 0.317, None, None), ('drug', 'gene', 16, 0.459, None, None),
                              ('crispr', 'protein', 32, 0.665, 0.52, 0.82), ('crispr', 'protein', 64, 0.788, 0.66, 0.92), ('crispr', 'gene', 32, 0.627, 0.54, 0.72), ('crispr', 'gene', 64, 0.700, 0.61, 0.78)]:
    check(f'task/legend {ep} {lay} R({k})', pq.loc[(ep, lay, k, 'supervised'), 'R'], v, 5.01e-4)
    if lo is not None:
        check(f'task/legend {ep} {lay} R({k}) lo', pq.loc[(ep, lay, k, 'supervised'), 'lo'], lo, 5.01e-3)
        check(f'task/legend {ep} {lay} R({k}) hi', pq.loc[(ep, lay, k, 'supervised'), 'hi'], hi, 5.01e-3)
for lay, k, v in [('protein', 16, 0.342), ('protein', 32, 0.515), ('gene', 16, 0.375), ('gene', 32, 0.518)]:
    check(f'legend drug random {lay} R({k})', pq.loc[('drug', lay, k, 'random_mean'), 'R'], v, 5.01e-4)
# F5 verdict consistency
f5 = A['F']['F5']
for arm in ('M+P32', 'M+P64', 'M+G32', 'M+G64'):
    check(f'F5 evidence {arm} R == panel_recovery', f5['evidence'][arm]['R'], A['drug']['panel_recovery'][arm]['R'], 0)
f5_pass = all(A['drug']['panel_recovery'][a]['R'] >= R_CRIT for a in ('M+P32', 'M+P64', 'M+G32', 'M+G64'))
check('F5 confirmed == all four R >= 0.5', f5_pass, f5['confirmed'], 0)
f8 = A['F']['F8']
for arm in ('M+P32', 'M+P64', 'M+G32', 'M+G64'):
    check(f'F8 F5_detail {arm} == R >= 0.5', A['crispr']['panel_recovery'][arm]['R'] >= R_CRIT, f8['F5_detail'][arm], 0)

# ------------------------------------------------------------------ panel d: scoreboard
F = A['F']
check('F total 5/7', F['total'] == '5/7 confirmed', True, 0)
check('F1..F7 confirmed count == 5', sum(F[f'F{i}']['confirmed'] for i in range(1, 8)), 5, 0)
for i in range(1, 8):
    assert isinstance(F[f'F{i}']['confirmed'], bool)
assert F['F8']['confirmed'] is True and F['F8']['F1'] and F['F8']['F2'] and F['F8']['F5']
check('F1 rule gain>0 & lo>0', (F['F1']['evidence']['gain_pct'] > 0) and (F['F1']['evidence']['boot']['lo'] > 0), F['F1']['confirmed'], 0)
check('F2 rule gain>0 & lo>0', (F['F2']['evidence']['gain_pct'] > 0) and (F['F2']['evidence']['boot']['lo'] > 0), F['F2']['confirmed'], 0)
check('F3 rule both shuffles contain 0 or <0', all((e['boot']['lo'] <= 0 <= e['boot']['hi']) or e['gain_pct'] < 0 for e in F['F3']['evidence'].values()), F['F3']['confirmed'], 0)
check('F4 rule both shares in band & F1 & F2', all(BAND[0] <= F['F4']['evidence'][l]['share'] <= BAND[1] for l in ('RNA', 'P')) and F['F1']['confirmed'] and F['F2']['confirmed'], F['F4']['confirmed'], 0)
check('F6 rule gain>0 & lo>0', (F['F6']['evidence']['gain_pct'] > 0) and (F['F6']['evidence']['boot']['lo'] > 0), F['F6']['confirmed'], 0)
check('F7 rule gain>0', F['F7']['evidence']['gain_pct'] > 0, F['F7']['confirmed'], 0)
check('crispr n_cells with screen == 95', S['meta']['gates']['crispr_123']['lines_with_crispr'], 95, 0)
check('summary F8 n_cells field (123 = lines decoded, not lines scored)', F['F8']['n_cells'], 123, 0)
check('drug n_eval_cells 123', A['drug']['n_eval_cells'], 123, 0)
check('drug n_targets 295', A['drug']['n_targets_evaluated'], 295, 0)
check('crispr n_targets 2456', A['crispr']['n_targets_evaluated'], 2456, 0)
check('drug n_strata 20', A['drug']['n_strata'], 20, 0)
check('bootstrap B', A['bootstrap']['B'], 2000, 0)
check('bootstrap seed', A['bootstrap']['seed'], 20261009, 0)


def g3(c):
    return f'{c["gain_pct"]:+.3f}% [{c["boot"]["lo"]:.2f}, {c["boot"]["hi"]:.2f}]'


def pw(c):
    return f'P(>0) {c["boot"]["prob_gt0"]:.3f}'


def wl(c, unit):
    st = c['sign_test']
    return f'wins {st["wins"]}, losses {st["losses"]}, ties {st["ties"]}' if st['ties'] else f'{fmt_n(st["wins"])}/{fmt_n(st["wins"] + st["losses"])} {unit} better'


e1, e2, e3, e4, e5, e6, e7, e8 = (F[f'F{i}']['evidence'] for i in range(1, 9))
sb = [
    dict(item='F1', claim='Free virtual RNA beats M', rule='gain > 0 and 95% lower bound > 0', sealed=g3(e1), sealed2=pw(e1) + ' · ' + wl(e1, 'drugs'), sealed3='',
         dev=f'{DEV_GAIN["drug"]["M+vRNA"]:+.3f}%', dev2='5/5 folds', dev3='', confirmed=F['F1']['confirmed']),
    dict(item='F2', claim='Free virtual protein beats M', rule='gain > 0 and 95% lower bound > 0', sealed=g3(e2), sealed2=pw(e2) + ' · ' + wl(e2, 'drugs'), sealed3='',
         dev=f'{DEV_GAIN["drug"]["M+vP"]:+.3f}%', dev2='5/5 folds', dev3='', confirmed=F['F2']['confirmed']),
    dict(item='F3', claim='Shuffle controls do not beat M', rule='interval contains 0, or gain < 0', sealed=f'vRNA {g3(e3["RNA"])}', sealed2=f'vP {g3(e3["P"])}', sealed3='',
         dev=f'vRNA {DEV_GAIN["drug"]["M+vRNA_shuf"]:+.3f}%', dev2=f'vP {DEV_GAIN["drug"]["M+vP_shuf"]:+.3f}%', dev3='1/5 folds', confirmed=F['F3']['confirmed']),
    dict(item='F4', claim='Recovered share in [0.15, 0.55]', rule='for RNA and for protein (given F1, F2)',
         sealed=f'RNA {e4["RNA"]["share"]:.3f} [{e4["RNA"]["share_boot"]["lo"]:.2f}, {e4["RNA"]["share_boot"]["hi"]:.2f}] in band',
         sealed2=f'P {e4["P"]["share"]:.3f} [{e4["P"]["share_boot"]["lo"]:.2f}, {e4["P"]["share_boot"]["hi"]:.2f}]',
         sealed3=f'{BAND[0] - e4["P"]["share"]:.3f} below the band',
         dev=f'RNA {DEV_SHARE["drug"]["RNA"]:.2f}', dev2=f'P {DEV_SHARE["drug"]["P"]:.2f}', dev3='', confirmed=F['F4']['confirmed']),
    dict(item='F5', claim='Panel law: R(32) ≥ 0.5, R(64) ≥ 0.5', rule='protein and gene panels, saved teachers',
         sealed=f'proteins {e5["M+P32"]["R"]:.3f} / {e5["M+P64"]["R"]:.3f} (< 0.5)', sealed2=f'genes {e5["M+G32"]["R"]:.3f} / {e5["M+G64"]["R"]:.3f}', sealed3='',
         dev=f'P {DEV_R["drug"][("protein", 32)]:.3f}/{DEV_R["drug"][("protein", 64)]:.3f}', dev2=f'G {DEV_R["drug"][("gene", 32)]:.3f}/{DEV_R["drug"][("gene", 64)]:.3f}', dev3='', confirmed=F['F5']['confirmed']),
    dict(item='F6', claim='Tissue label has value', rule='gain(M+T vs M) > 0 and lower bound > 0', sealed=g3(e6), sealed2=pw(e6), sealed3=wl(e6, 'drugs'),
         dev=f'{DEV_GAIN["drug"]["M+T"]:+.3f}%', dev2='5/5 folds', dev3='', confirmed=F['F6']['confirmed']),
    dict(item='F7', claim='Training-time protein graph', rule='gain > 0 (direction; interval reported)', sealed=g3(e7), sealed2=pw(e7), sealed3=wl(e7, 'drugs'),
         dev=f'{DEV_GAIN["drug"]["M+P_graph"]:+.3f}%', dev2='5/5 folds', dev3='', confirmed=F['F7']['confirmed']),
    dict(item='F8', claim='F1, F2, F5 on CRISPR gene effect', rule='same rules; 2,456 targets; 95 lines',
         sealed=f'vRNA {g3(e8["vRNA"])}', sealed2=f'vP {g3(e8["vP"])}',
         sealed3=f'R(32)/R(64) P {e8["panels"]["M+P32"]["R"]:.3f}/{e8["panels"]["M+P64"]["R"]:.3f}', sealed4=f'G {e8["panels"]["M+G32"]["R"]:.3f}/{e8["panels"]["M+G64"]["R"]:.3f}',
         dev=f'vRNA {DEV_GAIN["crispr"]["M+vRNA"]:+.3f}%', dev2=f'vP {DEV_GAIN["crispr"]["M+vP"]:+.3f}%',
         dev3=f'P {DEV_R["crispr"][("protein", 32)]:.3f}/{DEV_R["crispr"][("protein", 64)]:.3f}', dev4=f'G {DEV_R["crispr"][("gene", 32)]:.3f}/{DEV_R["crispr"][("gene", 64)]:.3f}', confirmed=F['F8']['confirmed'])]
for r in sb:
    r.setdefault('sealed4', ''); r.setdefault('dev4', '')
for r in sb:
    r['verdict'] = 'confirmed' if r['confirmed'] else 'not confirmed'
    r['mark'] = '✓' if r['confirmed'] else '✗'
sbd = pd.DataFrame(sb)
sbd['row'] = range(1, 9)
wr(sbd, 'd_scoreboard')

# ------------------------------------------------------------------ protocol / provenance numbers
draft_time = re.search(r'生成时间 (\S+)。', DRAFT).group(1)
proto = dict(
    protocol_id=S['receipt']['protocol_id'], label=A['label'], n_train=S['meta']['n_train'], n_eval=S['meta']['n_eval'],
    drug=dict(n_cells=A['drug']['n_eval_cells'], n_targets=A['drug']['n_targets_evaluated'], n_strata=A['drug']['n_strata'], null_score=A['drug']['arms']['null']['score'],
              M_score=A['drug']['arms']['M']['score'], null_vs_M_gain_pct=A['drug']['arms']['null']['gain_vs_M_pct']),
    crispr=dict(n_cells_decoded=A['crispr']['n_eval_cells'], n_cells_with_screen=S['meta']['gates']['crispr_123']['lines_with_crispr'], n_targets=A['crispr']['n_targets_evaluated'],
                n_strata=A['crispr']['n_strata'], M_score=A['crispr']['arms']['M']['score'], null_score=A['crispr']['arms']['null']['score'],
                note='summary.json F8.n_cells says 123 (lines decoded); only 95 carry a DepMap 24Q2 screen and contribute to the score (RECOMPUTE.md correction)'),
    bootstrap=A['bootstrap'],
    timeline=dict(plan_registered='2026-10-09T19:33:56+0800 (internal_final_20261009/PLAN_REGISTERED_AT.txt, local)', receipt_written_at=S['receipt']['written_at'],
                  summary_written=draft_time, retries=0, note='one invocation; output/ created once (os.makedirs exist_ok=False); no arm_failed'),
    recompute='independent recomputation (internal_final_20261009/RECOMPUTE.md, local): 74 arm sRMSE, all gains, intervals, P(>0), sign tests, shares, R(k) and F1-F8 verdicts agree; max |diff| 0',
    scope='sealed set = 123 lines of the same CCLE universe held out by the pre-registered identity split; a one-shot confirmation, NOT an external cohort',
    verdicts={f'F{i}': F[f'F{i}']['confirmed'] for i in range(1, 9)}, total=F['total'],
    band=list(BAND), R_criterion=R_CRIT,
    dev_constants=dict(gain_pct=DEV_GAIN, share=DEV_SHARE, room_pct=DEV_ROOM, R={ep: {f'{l}_{k}': v for (l, k), v in d.items()} for ep, d in DEV_R.items()},
                       source='manuscript_v0 Table R8-1 / Fig. 8 legend; cross-checked against figure7/plot_data a_map.csv and c_panel_curve.csv where present'),
    definitions=dict(gain='100*(sRMSE[M]-sRMSE[arm])/sRMSE[M] on the single sealed set', share='gain(M+vY)/gain(M+Y)', R='gain(M+k panel)/gain(teacher), teacher M+Pval (protein) or M+RNA (gene)',
                     interval='grouped bootstrap 2.5th-97.5th percentiles; cells resampled within tissue strata, drugs/targets by identifier, same draws for all arms',
                     wins='targets on which the arm standardized MSE is lower than M (paired sign test)'))
json.dump(proto, open(os.path.join(OUT, 'protocol.json'), 'w'), indent=1, ensure_ascii=False)
print('timeline:', proto['timeline'])

# ------------------------------------------------------------------ provenance
seen = []
with open(os.path.join(OUT, 'FIG8_DATA_SOURCES.md'), 'w', encoding='utf-8') as fh:
    fh.write('# Figure 8 (V2; FIG id 9) data sources\n\nGenerated by r/figures2_6/prepare_fig8_v2.py. Read-only inputs (sha256):\n\n| input | sha256 |\n|---|---|\n')
    for p in inputs:
        if p in seen:
            continue
        seen.append(p)
        fh.write(f'| `{p}` | `{sha(p)}` |\n')
    fh.write('\nSealed-set ground truth: summary.json (analysis.drug, analysis.crispr, analysis.F, meta.gates.crispr_123, receipt). '
             'RESULT_DRAFT.md and progress.jsonl were parsed only to cross-check the same numbers. No arms/*.npz and no raw file was opened. '
             'Development-fold constants (hollow symbols) are the manuscript Table R8-1 / Fig. 8 legend values; those present in the Figure 7 tables were asserted against them.\n')
    fh.write('\nOutputs: ' + ', '.join(sorted(os.listdir(OUT))) + '\n')
    fh.write(f'\nChecks run: {len(checks)}, all passed.\n\n| check | got | want | tol |\n|---|---|---|---|\n')
    for name, got, want, tol, ok in checks:
        fh.write(f'| {name} | {float(got):.6g} | {float(want):.6g} | {tol:g} |\n')
print('DONE; checks passed:', len(checks))

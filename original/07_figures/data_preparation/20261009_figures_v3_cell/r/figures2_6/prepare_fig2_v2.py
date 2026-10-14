#!/usr/bin/env python
"""Figure 2 (V2) data stage: cross-layer recoverability map.

Reads ONLY the crosslayer-matrix run outputs and writes tidy plot tables into
<V2>/delivery_R_20261009/figures2_6/figure2/plot_data/ plus FIG2_DATA_SOURCES.md.
Every printed number in Figure 2 must agree with crosslayer_matrix_20261009/RESULT.md.
"""
import hashlib, json, os, sys
import numpy as np
import pandas as pd

RUN = '/public/home/mengxl/creed-research-runs/crosslayer-matrix-20261009-0001/output'
V2 = '/public/home/mengxl/workspaces/creed-research/figure_work/20261009_figures2_6_v2'
OUT = os.path.join(V2, 'delivery_R_20261009/figures2_6/figure2/plot_data')
os.makedirs(OUT, exist_ok=True)

SCEN = {'new_cell': 'New cell lines', 'tissue': 'Held-out tissue'}
PAIRS = [('M', 'RNA'), ('M', 'P'), ('RNA', 'M'), ('RNA', 'P'), ('P', 'M'), ('P', 'RNA')]
LABEL_EN = {'有空间': 'room', '小幅空间': 'small room', '无空间': 'no room',
            '有恢复': 'recovered', '无恢复': 'not recovered',
            '部分收回': 'partly recovered', '基本未收回': 'essentially not recovered'}

sources = []
def reg(path):
    h = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    sources.append((path, h))
    return path

# ---------------------------------------------------------------- matrix cells
mats = []
for sc in SCEN:
    m = pd.read_csv(reg(os.path.join(RUN, f'matrix_{sc}.csv')))
    m.insert(0, 'scenario', sc)
    mats.append(m)
mat = pd.concat(mats, ignore_index=True)
for c in ['room_label', 'recovery_label', 'share_label']:
    mat[c] = mat[c].map(lambda s: LABEL_EN.get(s, s) if isinstance(s, str) else 'no room')
mat['blocked'] = mat['share'].isna()           # headroom <= 0.3 % -> share undefined
mat['scenario_label'] = mat['scenario'].map(SCEN)
mat['pair'] = mat['deployed'] + '→' + mat['borrowed']
mat.to_csv(os.path.join(OUT, 'xl_matrix.csv'), index=False)

# ---------------------------------------------------------------- per-fold scores (all arms)
fs = []
for sc in SCEN:
    f = pd.read_csv(reg(os.path.join(RUN, sc, 'fold_metrics.csv')))
    f.insert(0, 'scenario', sc)
    fs.append(f)
fold = pd.concat(fs, ignore_index=True)
fold['scenario_label'] = fold['scenario'].map(SCEN)
# consistency: 5-fold means vs summary.json
for sc in SCEN:
    s = json.load(open(reg(os.path.join(RUN, sc, 'summary.json'))))
    mean = fold[fold.scenario == sc].groupby('arm')['score'].mean()
    for arm, v in s['summary'].items():
        assert abs(mean[arm] - v['score']) < 1e-9, (sc, arm, mean[arm], v['score'])
    assert s['reproduction']['passed']
fold.to_csv(os.path.join(OUT, 'xl_fold_scores.csv'), index=False)
null_mean = fold[fold.arm == 'null_drug'].groupby('scenario')['score'].mean().to_dict()

# ---------------------------------------------------------------- per-fold paired gains per cell
comp = pd.read_csv(reg(os.path.join(RUN, 'comparisons.csv')))
comp['per_fold'] = comp['per_fold'].map(json.loads)
comp['ci'] = comp['ci'].map(json.loads)
comp_long = comp.explode('per_fold').reset_index(drop=True)
comp_long['fold'] = comp_long.groupby(['scenario', 'kind', 'deployed', 'borrowed']).cumcount()
comp_long = comp_long.rename(columns={'per_fold': 'gain_fold', 'gain': 'gain_mean'})
comp_long['gain_fold'] = comp_long['gain_fold'].astype(float)
comp_long['ci_lo'] = comp_long['ci'].map(lambda c: c[0]); comp_long['ci_hi'] = comp_long['ci'].map(lambda c: c[1])
comp_long = comp_long.drop(columns=['ci'])
# verify per-fold gains against fold_metrics (gain = 100*(base-new)/base per fold, rounded to 4 dp in comparisons.csv)
sc_fold = fold.set_index(['scenario', 'arm', 'fold'])['score']
for _, r in comp_long.iterrows():
    g = 100 * (sc_fold[(r.scenario, r.base, r.fold)] - sc_fold[(r.scenario, r.new, r.fold)]) / sc_fold[(r.scenario, r.base, r.fold)]
    assert abs(g - r.gain_fold) < 6e-5, (r.scenario, r.kind, r.new, r.base, r.fold, g, r.gain_fold)
comp_long['scenario_label'] = comp_long['scenario'].map(SCEN)
comp_long.to_csv(os.path.join(OUT, 'xl_comparisons_fold.csv'), index=False)

# generator OOF R2 per fold (mean over the 4 banks of that fold: inner-0..2 + outer) and over all 20 banks
rows = []
for sc in SCEN:
    b = json.load(open(reg(os.path.join(RUN, sc, 'bank_info.json'))))
    for X, Y in PAIRS:
        key = f'{X}->{Y}'
        allb = []
        for f in range(5):
            vals = [b[f'fold-{f}/{k}']['generators'][key]['oof_r2_mean'] for k in ['inner-0', 'inner-1', 'inner-2', 'outer']]
            shuf = [b[f'fold-{f}/{k}']['generators'][key + '_shuf']['oof_r2_mean'] for k in ['inner-0', 'inner-1', 'inner-2', 'outer']]
            allb += vals
            rows.append(dict(scenario=sc, deployed=X, borrowed=Y, fold=f, oof_r2_fold=float(np.mean(vals)),
                             oof_r2_shuf_fold=float(np.mean(shuf)), n_banks=len(vals)))
        m20 = float(np.mean(allb))
        ref = float(mat[(mat.scenario == sc) & (mat.deployed == X) & (mat.borrowed == Y)]['generator_oof_r2'].iloc[0])
        assert abs(m20 - ref) < 1e-9, (sc, key, m20, ref)
r2 = pd.DataFrame(rows)

wide = comp_long.pivot_table(index=['scenario', 'deployed', 'borrowed', 'fold'], columns='kind', values='gain_fold').reset_index()
wide = wide.rename(columns={'headroom': 'headroom_fold', 'recovered': 'recovered_fold', 'shuffle': 'shuffle_fold', 'virt_vs_shuf': 'virt_vs_shuf_fold'})
wide = wide.merge(r2, on=['scenario', 'deployed', 'borrowed', 'fold'])
wide = wide.merge(mat[['scenario', 'deployed', 'borrowed', 'headroom_gain', 'recovered_gain', 'share', 'blocked', 'room_label', 'recovery_label', 'generator_oof_r2']],
                  on=['scenario', 'deployed', 'borrowed'])
# per-fold recovery fraction: only defined where the cell has room (mean headroom > 0.3 %, as pre-registered)
wide['share_fold'] = np.where(wide['blocked'], np.nan, wide['recovered_fold'] / wide['headroom_fold'])
wide['scenario_label'] = wide['scenario'].map(SCEN)
wide['pair'] = wide['deployed'] + '→' + wide['borrowed']
wide.to_csv(os.path.join(OUT, 'xl_pair_folds.csv'), index=False)

# ---------------------------------------------------------------- protocol numbers for the schematic (panel A)
b0 = json.load(open(os.path.join(RUN, 'new_cell', 'bank_info.json')))['fold-0/outer']
proto = dict(
    scenarios=SCEN, folds=5, pairs=[f'{x}→{y}' for x, y in PAIRS],
    n_metabolites=b0['M']['features'], n_rna_genes=b0['RNA']['features'], n_protein_candidates=b0['P']['candidates'],
    virtual_targets=512, generator='cross-fitted kernel ridge (3 inner folds), 4 lambda', banks_per_fold=4,
    null_mean={k: round(v, 6) for k, v in null_mean.items()},
    drugs_new_cell=int(fold[(fold.scenario == 'new_cell')]['drugs'].iloc[0]),
    drugs_tissue=sorted(fold[(fold.scenario == 'tissue')]['drugs'].unique().tolist()),
    cells_total=int(fold[(fold.scenario == 'new_cell') & (fold.arm == 'M')][['n_fit', 'n_held']].sum(axis=1).iloc[0]),
    definitions=dict(room='sRMSE(X) - sRMSE(X+Y measured), % of sRMSE(X), mean of per-fold gains',
                     recovery='sRMSE(X) - sRMSE(X+virtual Y), % of sRMSE(X)',
                     share='recovery / room, defined when room > 0.3 %'))
json.dump(proto, open(os.path.join(OUT, 'xl_protocol.json'), 'w'), indent=1, ensure_ascii=False)

# ---------------------------------------------------------------- sources
with open(os.path.join(OUT, 'FIG2_DATA_SOURCES.md'), 'w') as fh:
    fh.write('# Figure 2 (V2) data sources\n\nGenerated by r/figures2_6/prepare_fig2_v2.py. Read-only inputs (sha256):\n\n')
    for p, h in sources:
        fh.write(f'- `{p}`  \n  sha256 `{h}`\n')
    fh.write('\nGround truth for printed numbers: /Users/xlzhexue/Desktop/proj/creed/crosslayer_matrix_20261009/RESULT.md (tables 2 and 3).\n')
    fh.write('\nOutputs: xl_matrix.csv (12 cells), xl_fold_scores.csv (all arms x 5 folds x 2 scenarios), xl_comparisons_fold.csv (per-fold paired gains), xl_pair_folds.csv (per cell x fold: room, recovered, shuffle, share, generator OOF R2), xl_protocol.json.\n')
    fh.write('\nChecks passed: 5-fold means == summary.json; per-fold gains == 100*(base-new)/base from fold_metrics.csv (<6e-5); 20-bank generator OOF R2 mean == matrix_*.csv generator_oof_r2.\n')
    fh.write(f'\nNull model (per-drug training mean) 5-fold mean sRMSE: {null_mean}\n')
print('OK', OUT)
print(mat[['scenario', 'deployed', 'borrowed', 'headroom_gain', 'recovered_gain', 'share', 'blocked']].round(4).to_string())
print(wide.groupby(['scenario', 'pair'])[['share_fold', 'oof_r2_fold']].agg(['min', 'max']).round(3).to_string())

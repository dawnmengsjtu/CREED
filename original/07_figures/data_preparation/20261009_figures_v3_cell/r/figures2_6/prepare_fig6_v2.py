"""Figure 6 (V2, 2026-10-09): functional effects organized by existing drug annotations.

Arms are rebound from the 2026-09-08 M->virtual-P run to
  * crosslayer-matrix-20261009-0001  : M (direct), M+vRNA, M+vP (deployable: M only), M+RNA (measured RNA; not deployable), null_drug
  * protein-panel-20261009-0001      : m_panel_k32 (deployable: M + 32 supervised proteins)
for BOTH scenarios (new_cell, tissue).  No training; every per-drug score is replayed from frozen out-of-fold predictions
and checked against the runs' own fold_metrics.csv / per_drug_metrics.csv.

Annotation grouping rules are copied verbatim from prepare_gdsc.py (20261007): stable-ID join, exact whole putative-target
strings (comma-separated strings NOT split), literal NA-like strings treated as absent, 'Unannotated' retained, top-7 strings by
full-panel frequency + 'Other annotations'.

Reads only: the two run output dirs above and the frozen raw inputs under <V2>/data/raw (symlinks to allowed runs).
Writes only: <V2>/delivery_R_20261009/figures2_6/figure6/plot_data/*.csv|json and <V2>/.../figure6/FIG6_DATA_SOURCES.md
"""
from pathlib import Path
from collections import Counter
import numpy as np, csv, json, hashlib, warnings, datetime

V2 = Path('/public/home/mengxl/workspaces/creed-research/figure_work/20261009_figures2_6_v2')
RAW = V2 / 'data/raw'
RUNS = Path('/public/home/mengxl/creed-research-runs')
XL = RUNS / 'crosslayer-matrix-20261009-0001/output'
PP = RUNS / 'protein-panel-20261009-0001/output'
OUT = V2 / 'delivery_R_20261009/figures2_6/figure6'
PD = OUT / 'plot_data'
PD.mkdir(parents=True, exist_ok=True)

SOURCES = {}
def reg(p):
    p = Path(p); rp = p.resolve()
    if str(p) not in SOURCES:
        SOURCES[str(p)] = (str(rp), hashlib.sha256(rp.read_bytes()).hexdigest(), rp.stat().st_size)
    return p
def load(p): return np.load(reg(p), allow_pickle=False)
def save(name, rows):
    rows = list(rows); assert rows, name
    with (PD / f'{name}.csv').open('w', newline='') as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader()
        for r in rows:
            w.writerow({k: ('' if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v) for k, v in r.items()})
    print('wrote', name, len(rows), flush=True)

# ---------------------------------------------------------------- annotations (rules identical to prepare_gdsc.py)
src = load(RAW / 'gdsc_metabolite_full_protein/source-inputs.npz')
annotation_path = reg(RAW / 'figure2_6_extra/drug_annotations.json'); annot = json.loads(annotation_path.read_text())
name = lambda did: annot.get(str(did), {}).get('names', [str(did)])[0]
def target(d):
    x = [v for v in annot.get(str(d), {}).get('putative_targets', []) if v and v.strip().upper() not in ('NA', 'N/A', 'NONE', 'UNKNOWN')]
    return ' | '.join(sorted(set(x))) if x else 'Unannotated'
allids = src['drug_ids'].astype(str).tolist(); assert len(allids) == len(set(allids))
categories = Counter(target(d) for d in allids); top = sorted(categories, key=lambda g: (-categories[g], g))[:7]
group = lambda d: target(d) if target(d) in top else 'Other annotations'
groups = top + ['Other annotations']
print('groups:', groups, flush=True)

# ---------------------------------------------------------------- arms
SCEN = {'new_cell': ('new_cell', 'new_cell_main'), 'tissue': ('tissue', 'tissue_main')}
XL_ARMS = {'m_only': 'M', 'm_vrna': 'M+vRNA', 'm_vp': 'M+vP', 'm_rna_measured': 'M+RNA', 'null_drug': 'null_drug'}
PP_ARMS = {'m_panel32': 'm_panel_k32'}
ARMS = ['m_only', 'm_vrna', 'm_vp', 'm_panel32', 'm_rna_measured', 'null_drug']
COMP = ['m_vrna', 'm_vp', 'm_panel32', 'm_rna_measured']   # compared with m_only
DEPLOY = {'m_only': 'M only', 'm_vrna': 'M only', 'm_vp': 'M only', 'm_panel32': 'M + 32 measured proteins',
          'm_rna_measured': 'M + measured RNA (not deployable)', 'null_drug': 'none (training drug mean)'}

def read_fm(p):
    reg(p); rows = list(csv.DictReader(open(p)))
    return {(int(r['fold']), r['arm']): float(r['score']) for r in rows}
def read_pdm(p):
    reg(p); rows = list(csv.DictReader(open(p)))
    return {(int(r['fold']), r['arm'], r['drug_id']): float(r['standardized_mse']) for r in rows if r['evaluated'] == 'True'}

smse = {}        # (scen, fold, arm) -> {drug_id: smse}
scores = {}      # (scen, fold, arm) -> fold sRMSE
common = {}      # scen -> sorted list of drug ids evaluated in all five folds
inventory = []
n_replay = 0; n_drug_checks = 0
for scen, (xd, pdir) in SCEN.items():
    fm_x = read_fm(XL / xd / 'fold_metrics.csv'); fm_p = read_fm(PP / pdir / 'fold_metrics.csv'); pdm = read_pdm(XL / xd / 'per_drug_metrics.csv')
    eval_sets = []
    for f in range(5):
        zx = load(XL / xd / f'fold-{f}_predictions.npz'); zp = load(PP / pdir / f'fold-{f}_predictions.npz')
        for k in ['truth', 'train_sd', 'evaluated', 'panel', 'held', 'fit']:
            assert np.array_equal(zx[k], zp[k], equal_nan=True), (scen, f, k)
        assert np.max(np.abs(zx['pred__M'] - zp['m_only'])) <= 1e-8, (scen, f, 'M vs m_only')
        truth = zx['truth']; sd = zx['train_sd']; ev = zx['evaluated'].astype(bool); dids = zx['drug_ids'].astype(str)
        assert set(dids) <= set(allids) and (sd > 1e-8).all()
        assert np.array_equal(np.array(allids)[zx['panel']], dids)
        eval_sets.append({d for d, e in zip(dids, ev) if e})
        for a in ARMS:
            pred = zx['pred__' + XL_ARMS[a]] if a in XL_ARMS else zp[PP_ARMS[a]]
            assert np.isfinite(pred).all(), (scen, f, a)
            err = ((pred - truth) / sd) ** 2
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning); dm = np.nanmean(err, axis=0)
            score = float(np.sqrt(dm[ev].mean()))
            stored = fm_x[(f, XL_ARMS[a])] if a in XL_ARMS else fm_p[(f, PP_ARMS[a])]
            assert np.isclose(score, stored, atol=1e-10, rtol=1e-10), (scen, f, a, score, stored)
            n_replay += 1
            scores[scen, f, a] = score
            smse[scen, f, a] = {d: float(dm[j]) for j, d in enumerate(dids) if ev[j]}
            if a in XL_ARMS and a != 'null_drug':
                for d, v in smse[scen, f, a].items():
                    assert np.isclose(v, pdm[(f, XL_ARMS[a], d)], atol=1e-9, rtol=1e-9); n_drug_checks += 1
        inventory.append(dict(scenario=scen, fold=f, held_n=int(len(zx['held'])), fit_n=int(len(zx['fit'])), panel_drugs=int(len(dids)), evaluated_drugs=int(ev.sum())))
    common[scen] = sorted(set.intersection(*eval_sets), key=int)
    print(scen, 'common evaluated drugs:', len(common[scen]), flush=True)
print('fold-arm score replays:', n_replay, '; per-drug sMSE checks vs per_drug_metrics.csv:', n_drug_checks, flush=True)

# ---------------------------------------------------------------- tables
save('drug_annotations', [dict(drug_id=d, name=name(d), target_annotation=target(d), group=group(d),
                               new_cell_all_folds=d in common['new_cell'], tissue_all_folds=d in common['tissue']) for d in allids])
PANEL = {'all': 'All drugs', 'new_cell': 'New cell lines, all 5 folds', 'tissue': 'Whole tissue, all 5 folds'}
save('annotation_counts', [dict(group=g, order=i + 1, panel_key=k, panel=PANEL[k], n=sum(group(d) == g for d in dd))
                           for i, g in enumerate(groups) for k, dd in [('all', allids), ('new_cell', common['new_cell']), ('tissue', common['tissue'])]])
save('inventory', inventory)
save('arm_scores', [dict(scenario=s, fold=f, arm=a, deployment_input=DEPLOY[a], srmse=scores[s, f, a]) for s in SCEN for f in range(5) for a in ARMS])

gains = []; flows = []; effects = []; winfrac = []
for s in SCEN:
    for d in common[s]:
        fm = {a: float(np.mean([smse[s, f, a][d] for f in range(5)])) for a in ARMS}
        row = dict(scenario=s, drug_id=d, name=name(d), group=group(d), direct_smse=fm['m_only'])
        for a in COMP:
            g = fm['m_only'] - fm[a]; st = 'Lower error' if fm[a] < fm['m_only'] else 'Higher / equal'
            gains.append(dict(scenario=s, drug_id=d, name=name(d), group=group(d), arm=a, direct_smse=fm['m_only'], arm_smse=fm[a], gain=g, status=st,
                              folds_lower=int(sum(smse[s, f, a][d] < smse[s, f, 'm_only'][d] for f in range(5)))))
            row['gain_' + a] = g; row['status_' + a] = st
        flows.append(row)
    for f in range(5):
        for i, g in enumerate(groups):
            dd = [d for d in common[s] if group(d) == g]
            if not dd: continue
            for a in COMP:
                dl = [smse[s, f, 'm_only'][d] - smse[s, f, a][d] for d in dd]
                effects.append(dict(scenario=s, fold=f, group=g, order=i + 1, arm=a, n=len(dd), delta=float(np.mean(dl)), median_delta=float(np.median(dl))))
                winfrac.append(dict(scenario=s, fold=f, group=g, order=i + 1, arm=a, n=len(dd), n_lower=int(sum(x > 0 for x in dl)), frac=float(np.mean([x > 0 for x in dl]))))
save('drug_arm_gains', gains); save('drug_flows', flows); save('annotation_effects', effects); save('fold_win_fraction', winfrac)

# ---------------------------------------------------------------- protocol numbers for panel A / caption
means = {s: {a: float(np.mean([scores[s, f, a] for f in range(5)])) for a in ARMS} for s in SCEN}
wins = {s: {a: dict(lower=sum(1 for r in gains if r['scenario'] == s and r['arm'] == a and r['status'] == 'Lower error'), total=len(common[s]))
            for a in COMP} for s in SCEN}
fold_wins = {s: {a: int(sum(scores[s, f, a] < scores[s, f, 'm_only'] for f in range(5))) for a in COMP} for s in SCEN}
protocol = dict(full_drugs=len(allids), folds=5, common_drugs={s: len(common[s]) for s in SCEN}, groups=groups,
                arm_labels={'m_only': 'M only', 'm_vrna': 'M + virtual RNA', 'm_vp': 'M + virtual P', 'm_panel32': 'M + 32 proteins',
                            'm_rna_measured': 'M + RNA (measured)', 'null_drug': 'Null (drug mean)'},
                deployment_input=DEPLOY, mean_srmse=means, drugs_lower_than_direct=wins, folds_lower_than_direct=fold_wins,
                group_counts={k: {g: sum(group(d) == g for d in dd) for g in groups} for k, dd in [('all', allids), ('new_cell', common['new_cell']), ('tissue', common['tissue'])]},
                sources={'crosslayer': str(XL), 'protein_panel': str(PP), 'annotations': str(annotation_path)},
                checks=dict(fold_arm_score_replays=n_replay, per_drug_smse_checks=n_drug_checks, protein_panel_m_only_equals_crosslayer_M=True, shared_truth_held_fit_panel=True),
                evaluation='archived development outer folds; candidate_only; no new training; no p-values computed here')
(PD / 'protocol.json').write_text(json.dumps(protocol, indent=1))
print(json.dumps({k: protocol[k] for k in ['common_drugs', 'mean_srmse', 'drugs_lower_than_direct', 'folds_lower_than_direct']}, indent=1), flush=True)

# ---------------------------------------------------------------- data sources receipt
lines = ['# Figure 6 (V2) data sources', '', f'Generated {datetime.datetime.now().isoformat(timespec="seconds")} by `{Path(__file__).resolve()}` (sha256 {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}).', '',
         'Every input file read by the script, with sha256 of the resolved file. Outputs: `plot_data/*.csv`, `plot_data/protocol.json`.', '',
         '| input path | resolved path | bytes | sha256 |', '|---|---|---:|---|']
for p, (rp, h, n) in sorted(SOURCES.items()):
    lines.append(f'| `{p}` | `{rp if rp != p else "="}` | {n} | `{h}` |')
lines += ['', '## Checks', f'- {n_replay} fold x arm sRMSE replays match the runs\' fold_metrics.csv (atol 1e-10).',
          f'- {n_drug_checks} per-drug standardized-MSE values match crosslayer per_drug_metrics.csv (atol 1e-9).',
          '- protein-panel `m_only` equals crosslayer `pred__M` (max |diff| <= 1e-8) and both runs share truth/train_sd/evaluated/panel/held/fit per fold, so per-drug comparisons are paired on identical held-out samples.',
          f'- Common evaluated drugs: new_cell {len(common["new_cell"])}, tissue {len(common["tissue"])} (of {len(allids)} full-panel drugs).',
          '- Annotation grouping rules identical to 20261007 prepare_gdsc.py: ' + ', '.join(groups) + '.']
(OUT / 'FIG6_DATA_SOURCES.md').write_text('\n'.join(lines) + '\n')
print('wrote', OUT / 'FIG6_DATA_SOURCES.md', flush=True)

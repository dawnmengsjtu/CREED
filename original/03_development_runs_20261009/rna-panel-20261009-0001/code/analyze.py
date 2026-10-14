"""Pre-registered comparisons for the RNA-panel run (PLAN.md sections 7, 9, 10) + descriptive cross-layer table vs the protein-panel run.

usage: python analyze.py RUN_DIR   (expects RUN_DIR/output/{new_cell,tissue}/ from rna_panel.py main and RUN_DIR/output/repro/{sc}/)
Reads only this run's outputs, source-inputs.npz (tissues, drug ids), rna421 gene symbols, the protein-panel run's fold_metrics.csv
and the crosslayer-matrix run's fold_metrics.csv. Writes RUN_DIR/output/{analysis.json, fold_metrics.csv, summary.json, result_tables.md}.
"""
import csv
import json
import sys
import numpy as np
from scipy import stats

RUN = sys.argv[1]
OUT = RUN + '/output'
ROOT = '/public/home/mengxl/creed-research-runs'
SOURCE = ROOT + '/gdsc-metabolite-full-protein-20260908-0001/output/result/source-inputs.npz'
RNA421 = ROOT + '/rna421-20260906-0002'
PPANEL = ROOT + '/protein-panel-20261009-0001/output'
MATRIX = ROOT + '/crosslayer-matrix-20261009-0001/output'
KS = (8, 16, 32, 64, 128); KSEL = (8, 16, 32, 64); RKS = (16, 32); SEEDS = (0, 1, 2, 3, 4); WS = (0., .25, 1.); NA = 4
B = 2000
SRC = np.load(SOURCE, allow_pickle=False)
TIS, DRUG = SRC['tissues'], SRC['drug_ids']
try:
    SYM = np.load(RNA421 + '/gene_symbols.npy', allow_pickle=False)
except Exception:
    SYM = None


def read_fold_metrics(path):
    fs = {}
    for r in csv.DictReader(open(path)):
        fs.setdefault(r['arm'], [None] * 5)[int(r['fold'])] = (float(r['score']), float(r['spearman']), float(r['centered_mse']))
    return fs


def sq_err(fold, arm):
    z = fold['z']; t = z['truth']; obs = np.isfinite(t)
    return np.where(obs, ((z[arm] - t) / z['train_sd']) ** 2, 0.), obs


def weighted_scores(E, obs, use, CW, DW):
    num = CW @ E; den = CW @ obs.astype(float)
    ok = (den > 0) & use[None, :] & (DW > 0)
    mse = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    w = np.where(ok, DW, 0.)
    return np.sqrt((w * mse).sum(1) / w.sum(1))


def build_weights(sc, folds, rng):
    ids = sorted({str(x) for fo in folds for x in DRUG[fo['z']['panel']]})
    pos = {d: i for i, d in enumerate(ids)}
    draw = rng.integers(0, len(ids), size=(B, len(ids)))
    dcount = np.zeros((B, len(ids)))
    np.add.at(dcount, (np.repeat(np.arange(B), len(ids)), draw.ravel()), 1)
    out = []
    for fo in folds:
        z = fo['z']; held = z['held']; tis = TIS[held]; nq = len(held)
        DW = dcount[:, [pos[str(x)] for x in DRUG[z['panel']]]]
        CW = np.zeros((B, nq))
        if sc == 'new_cell':
            groups = sorted(set(tis.tolist())); gi = np.array([groups.index(t) for t in tis])
            g = rng.integers(0, len(groups), size=(B, len(groups)))
            gc = np.zeros((B, len(groups))); np.add.at(gc, (np.repeat(np.arange(B), len(groups)), g.ravel()), 1)
            CW = gc[:, gi]
        else:
            for t in sorted(set(tis.tolist())):
                cells = np.flatnonzero(tis == t)
                g = rng.integers(0, len(cells), size=(B, len(cells)))
                np.add.at(CW, (np.repeat(np.arange(B), len(cells)), cells[g].ravel()), 1)
        out.append((CW, DW))
    return out


class Scen:
    def __init__(self, sc):
        self.sc = sc
        d = f'{OUT}/{sc}'
        self.summ = json.load(open(f'{d}/summary.json'))
        self.fs = read_fold_metrics(f'{d}/fold_metrics.csv')
        self.ext = {'P::' + a: v for a, v in read_fold_metrics(f'{PPANEL}/{sc}_main/fold_metrics.csv').items()}
        self.ext.update({'X::' + a: v for a, v in read_fold_metrics(f'{MATRIX}/{sc}/fold_metrics.csv').items()})
        self.folds = []
        for f in range(5):
            z = np.load(f'{d}/fold-{f}_predictions.npz', allow_pickle=False)
            s = np.load(f'{d}/fold-{f}_selection.npz', allow_pickle=False)
            self.folds.append(dict(z={k: z[k] for k in z.files}, sel={k: s[k] for k in s.files if not k.startswith(('loss__', 'count__'))},
                                   info=json.load(open(f'{d}/fold-{f}_bank_info.json'))))
        self.ratio = float(np.mean([len(fo['z']['held']) / len(fo['z']['fit']) for fo in self.folds]))
        self.W = build_weights(sc, self.folds, np.random.default_rng(20261009 + (0 if sc == 'new_cell' else 1)))
        self.common = set.intersection(*[set(DRUG[fo['z']['panel']][fo['z']['evaluated']].tolist()) for fo in self.folds])
        self.groups = {}
        for p in (['random'] if sc == 'tissue' else ['random', 'random_tissue']):
            for k in RKS:
                self.groups[f'{p}_k{k}_mean'] = [f'{p}_k{k}_s{s}' for s in SEEDS]
        self._boot = {}; self._common = {}

    def members(self, arm):
        return self.groups.get(arm, [arm])

    def table(self, a):
        return self.ext[a] if '::' in a else self.fs[a]

    def fold_metric(self, arm, i=0):
        return np.mean([[self.table(a)[f][i] for f in range(5)] for a in self.members(arm)], axis=0)

    def fold_scores(self, arm):
        return self.fold_metric(arm, 0)

    def boot(self, arm):
        if arm not in self._boot:
            per = []
            for a in self.members(arm):
                s = np.zeros((B, 5))
                for f, fo in enumerate(self.folds):
                    E, obs = sq_err(fo, a); CW, DW = self.W[f]
                    s[:, f] = weighted_scores(E, obs, fo['z']['evaluated'], CW, DW)
                per.append(s)
            self._boot[arm] = np.mean(per, axis=0).mean(1)
        return self._boot[arm]

    def common_scores(self, arm):
        if arm not in self._common:
            per = []
            for a in self.members(arm):
                s = []
                for fo in self.folds:
                    E, obs = sq_err(fo, a); z = fo['z']
                    use = z['evaluated'] & np.isin(DRUG[z['panel']], list(self.common))
                    mse = E.sum(0) / np.maximum(obs.sum(0), 1)
                    s.append(np.sqrt(mse[use].mean()))
                per.append(s)
            self._common[arm] = np.mean(per, axis=0)
        return self._common[arm]


def verdict(imp, wins):
    return '过推进线' if imp >= 1 and wins >= 4 else ('小幅' if imp > 0 else '无增益')


def compare(S, new, base, label, robust=True, criterion=True):
    n, b = S.fold_scores(new), S.fold_scores(base)
    imp = (b.mean() - n.mean()) / b.mean() * 100
    d = (b - n) / b * 100; wins = int((n < b).sum())
    v = d.var(ddof=1)
    se = np.sqrt((1 / 5 + S.ratio) * v) if v > 0 else np.nan
    tval = d.mean() / se if np.isfinite(se) and se > 0 else np.nan
    p = float(2 * stats.t.sf(abs(tval), 4)) if np.isfinite(tval) else np.nan
    q = stats.t.ppf(.975, 4)
    out = dict(comparison=label, scenario=S.sc, new=new, base=base, new_mean=float(n.mean()), base_mean=float(b.mean()),
               improvement_pct=float(imp), per_fold_improvement_pct=d.round(4).tolist(), folds_won=f'{wins}/5', nb_t=float(tval), nb_p=p,
               nb_ci95_pct=[float(d.mean() - q * se), float(d.mean() + q * se)] if np.isfinite(se) else [np.nan, np.nan],
               verdict=verdict(imp, wins) if criterion else '描述性', new_fold_scores=n.round(6).tolist(), base_fold_scores=b.round(6).tolist())
    if robust:
        lofo = [float((b[k].mean() - n[k].mean()) / b[k].mean() * 100) for k in (np.arange(5) != f for f in range(5))]
        out['leave_one_fold_out_pct'] = [round(x, 4) for x in lofo]; out['lofo_all_positive'] = bool(min(lofo) > 0)
        bn, bb = S.boot(new), S.boot(base); bi = (bb - bn) / bb * 100
        out['boot_ci95_pct'] = [float(np.percentile(bi, 2.5)), float(np.percentile(bi, 97.5))]; out['boot_prob_positive'] = float((bi > 0).mean())
        if S.sc == 'tissue':
            cn, cb = S.common_scores(new), S.common_scores(base)
            out.update(common_panel_drugs=len(S.common), common_panel_improvement_pct=float((cb.mean() - cn.mean()) / cb.mean() * 100),
                       common_panel_folds_won=f'{int((cn < cb).sum())}/5', common_panel_new_mean=float(cn.mean()), common_panel_base_mean=float(cb.mean()))
    return out


def recovery(S, arm, teacher, base='m_only'):
    m, p, t = S.fold_scores(base), S.fold_scores(arm), S.fold_scores(teacher)
    return dict(arm=arm, base=base, teacher=teacher, arm_mean=float(p.mean()), teacher_mean=float(t.mean()),
                fraction_of_means=float((m.mean() - p.mean()) / (m.mean() - t.mean())), per_fold=((m - p) / (m - t)).round(4).tolist())


def best(S, arms):
    return min(arms, key=lambda a: S.fold_scores(a).mean())


def sel_dist(S, arm, decode):
    cnt = {}
    for fo in S.folds:
        for v in fo['sel'][arm][fo['z']['evaluated']]:
            key = decode(int(v)); cnt[key] = cnt.get(key, 0) + 1
    return dict(sorted(cnt.items()))


def sym(i):
    return str(SYM[i]) if SYM is not None and i < len(SYM) else str(i)


def panel_overlap(S):
    out = {}
    for k in KS:
        sets = [set(S.folds[f]['info'][f'fold-{f}/outer']['top128'][:k]) for f in range(5)]
        jac = [len(sets[i] & sets[j]) / len(sets[i] | sets[j]) for i in range(5) for j in range(i + 1, 5)]
        out[f'k{k}_mean_pairwise_jaccard_outer'] = float(np.mean(jac))
        out[f'k{k}_in_all_5_outer'] = [sym(i) for i in sorted(set.intersection(*sets))]
    freq = {}
    for f in range(5):
        for i in S.folds[f]['info'][f'fold-{f}/outer']['top128'][:16]:
            freq[sym(i)] = freq.get(sym(i), 0) + 1
    out['outer_top16_frequency'] = dict(sorted(freq.items(), key=lambda x: -x[1]))
    out['outer_top16_fold0'] = [sym(i) for i in S.folds[0]['info']['fold-0/outer']['top128'][:16]]
    out['protein_top32_fold0_count'] = len(S.folds[0]['info']['fold-0/outer'].get('protein', {}).get('top32', []))
    return out


def fmt(x, nd=3):
    return '—' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{nd}f}'


def main():
    res = {}; md = []
    merged_rows, merged_summary = [], {}
    for sc in ('new_cell', 'tissue'):
        S = Scen(sc)
        R = dict(reproduction=S.summ['reproduction'], repro_stage=json.load(open(f'{OUT}/repro/{sc}/summary.json'))['reproduction'],
                 nb_ratio_test_train=S.ratio, common_panel_drugs=len(S.common), seconds=S.summ['seconds'])
        for r in csv.DictReader(open(f'{OUT}/{sc}/fold_metrics.csv')):
            merged_rows.append(dict(scenario=sc, **r))
        merged_summary[sc] = dict(summary=S.summ['summary'], fold_scores=S.summ['fold_scores'], reproduction=S.summ['reproduction'])
        arms = sorted(S.fs) + sorted(S.groups)
        R['arms'] = {a: dict(score=float(S.fold_scores(a).mean()), spearman=float(np.nanmean(S.fold_metric(a, 1))),
                             centered_mse=float(S.fold_metric(a, 2).mean()), fold_scores=S.fold_scores(a).round(6).tolist()) for a in arms}
        R['protein_run_arms'] = {a: dict(score=float(S.fold_scores(a).mean()), fold_scores=S.fold_scores(a).round(6).tolist())
                                 for a in sorted(S.ext) if a.startswith('P::') and any(t in a for t in ('m_panel_k', 'teacher_saved', 'm_allcand', 'm_only', 'm_tissue', 'tissue_only', 'vfm', 'm_panel_sel', 'm_panel_tissue_sel'))}
        C = []
        C.append(compare(S, 'm_panel_sel', 'm_only', 'P1 面板价值：m_panel_sel vs m_only'))
        if sc == 'new_cell':
            C.append(compare(S, 'm_panel_tissue_sel', 'm_tissue', 'P1b 面板价值（含组织）：m_panel_tissue_sel vs m_tissue'))
        p2 = []
        for k in RKS:
            c = compare(S, f'm_panel_k{k}', f'random_k{k}_mean', f'P2 监督 vs 随机 k={k}'); C.append(c); p2.append(c)
        R['P2_label'] = '监督挑选优于随机' if all(int(c['folds_won'][0]) >= 4 and c['improvement_pct'] > 0 for c in p2) else '未优于随机'
        rb = best(S, [f'random_k{k}_mean' for k in RKS])
        C.append(compare(S, 'm_panel_sel', rb, f'P2c（描述）m_panel_sel vs 最强随机均值（{rb}）'))
        if sc == 'new_cell':
            for k in RKS:
                C.append(compare(S, f'm_panel_tissue_k{k}', f'random_tissue_k{k}_mean', f'P2t（描述）含组织 监督 vs 随机 k={k}'))
            C.append(compare(S, 'm_panel_tissue_sel', 'tissue_only', 'P5 m_panel_tissue_sel vs tissue_only'))
            C.append(compare(S, 'm_panel_tissue_sel', 'm_only', '附 m_panel_tissue_sel vs m_only', robust=False))
            C.append(compare(S, 'm_allcand_tissue', 'm_tissue', '次要 m_allcand_tissue vs m_tissue', robust=False))
        # secondary descriptive
        C.append(compare(S, 'm_panel_k128', 'm_panel_k64', '次要 k=128 vs k=64', robust=False, criterion=False))
        C.append(compare(S, 'm_panel_sel', 'm_panel_k64', '次要 m_panel_sel vs m_panel_k64', robust=False, criterion=False))
        C.append(compare(S, 'm_allcand', 'm_panel_sel', '次要 全候选 3000 基因 vs 小面板', robust=False, criterion=False))
        C.append(compare(S, 'teacher_saved', 'm_panel_sel', '次要 教师（矩阵 M+RNA）vs 小面板', robust=False, criterion=False))
        C.append(compare(S, 'm_allcand', 'm_only', '次要 教师复现 m_allcand vs m_only（= 矩阵 headroom）', robust=False, criterion=False))
        for k in KS:
            C.append(compare(S, f'm_panel_k{k}', 'm_only', f'次要 m_panel_k{k} vs m_only', robust=False, criterion=False))
        if sc == 'tissue':
            C.append(compare(S, 'vfm_panel_sel', 'vfm', '次要 vfm_panel_sel vs vfm', robust=False, criterion=False))
            C.append(compare(S, 'vfm_panel_k64', 'vfm', '次要 vfm_panel_k64 vs vfm', robust=False, criterion=False))
            C.append(compare(S, 'vfm_allcand', 'vfm', '次要 vfm_allcand vs vfm', robust=False, criterion=False))
        # cross-layer (descriptive): same folds, protein-panel run's saved fold scores
        X = []
        for k in KSEL:
            X.append(compare(S, f'm_panel_k{k}', f'P::m_panel_k{k}', f'跨层 M+{k}基因 vs M+{k}蛋白', robust=False, criterion=False))
        X.append(compare(S, 'teacher_saved', 'P::teacher_saved', '跨层 教师 M+RNA3000 vs 教师 M+全P', robust=False, criterion=False))
        X.append(compare(S, 'm_allcand', 'P::m_allcand', '跨层 M+3000基因 vs M+全候选蛋白(~1800)', robust=False, criterion=False))
        for b_ in ('m_panel_k32', 'm_panel_k64', 'P::m_panel_k32', 'P::m_panel_k64', 'm_only'):
            X.append(compare(S, 'm_panel_pg_k32', b_, f'双层 m_panel_pg_k32 vs {b_}', robust=False, criterion=False))
        if sc == 'new_cell':
            for k in KSEL:
                X.append(compare(S, f'm_panel_tissue_k{k}', f'P::m_panel_tissue_k{k}', f'跨层（含组织）M+T+{k}基因 vs M+T+{k}蛋白', robust=False, criterion=False))
        R['comparisons'] = C; R['crosslayer_comparisons'] = X
        # recovery
        rec = []
        panel_arms = [f'm_panel_k{k}' for k in KS] + ['m_panel_sel', 'm_panel_pg_k32', 'm_allcand']
        if sc == 'new_cell':
            panel_arms += [f'm_panel_tissue_k{k}' for k in KS] + ['m_panel_tissue_sel', 'm_tissue', 'm_allcand_tissue']
        else:
            panel_arms += [f'vfm_panel_k{k}' for k in KS] + ['vfm_panel_sel', 'vfm', 'vfm_allcand']
        for a in panel_arms:
            for t in ('teacher_saved', 'm_allcand'):
                rec.append(recovery(S, a, t))
        R['recovery'] = rec
        R['recovery_protein_run'] = [recovery(S, f'P::{a}', 'P::teacher_saved', base='P::m_only') for a in
                                     [f'm_panel_k{k}' for k in KSEL] + ['m_panel_sel', 'm_allcand'] +
                                     ([f'm_panel_tissue_k{k}' for k in KSEL] + ['m_panel_tissue_sel'] if sc == 'new_cell' else [f'vfm_panel_k{k}' for k in KSEL] + ['vfm_panel_sel'])]
        R['selection'] = {'m_panel_sel_k': sel_dist(S, 'm_panel_sel', lambda v: f'k{KSEL[v // NA]}'),
                          'm_panel_sel_alpha': sel_dist(S, 'm_panel_sel', lambda v: f'alpha{(0.1, 1, 10, 100)[v % NA]}')}
        if sc == 'new_cell':
            R['selection'].update({'m_panel_tissue_sel_k': sel_dist(S, 'm_panel_tissue_sel', lambda v: f'k{KSEL[v // (3 * NA)]}'),
                                   'm_panel_tissue_sel_w': sel_dist(S, 'm_panel_tissue_sel', lambda v: f'w{WS[(v // NA) % 3]}'),
                                   'm_tissue_w': sel_dist(S, 'm_tissue', lambda v: f'w{WS[v // NA]}')})
        else:
            R['selection']['vfm_panel_sel_k'] = sel_dist(S, 'vfm_panel_sel', lambda v: f'k{KSEL[v // NA]}')
        R['panels'] = panel_overlap(S)
        res[sc] = R
        print(json.dumps(dict(event='scenario_done', scenario=sc)), flush=True)
    json.dump(res, open(f'{OUT}/analysis.json', 'w'), indent=1, ensure_ascii=False)
    with open(f'{OUT}/fold_metrics.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(merged_rows[0])); w.writeheader(); w.writerows(merged_rows)
    json.dump(merged_summary, open(f'{OUT}/summary.json', 'w'), indent=1)
    # ---- markdown tables
    for sc, R in res.items():
        md.append(f'\n## {sc}\n')
        md.append('### 复现\n```\n' + json.dumps(R['reproduction'], indent=1) + '\n```\n')
        md.append('### 全部臂（5 折均值，按 sRMSE 升序）\n\n| 臂 | sRMSE | Spearman | centered MSE | 折0 | 折1 | 折2 | 折3 | 折4 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|')
        for a, v in sorted(R['arms'].items(), key=lambda x: x[1]['score']):
            md.append(f"| {a} | {v['score']:.6f} | {fmt(v['spearman'], 4)} | {v['centered_mse']:.4f} | " + ' | '.join(f'{x:.4f}' for x in v['fold_scores']) + ' |')
        md.append('\n### 预先登记的比较与次要比较\n\n| 比较 | 新臂 | 基线 | 新/基线 5折均值 | 改善% | 胜折 | 逐折改善% | NB p | NB 95% | 去一折最小 | bootstrap 95% | P(>0) | 共同药面板 | 判定 |\n|---|---|---|---|---:|---|---|---:|---|---:|---|---:|---|---|')
        for c in R['comparisons'] + R['crosslayer_comparisons']:
            lofo = c.get('leave_one_fold_out_pct'); bci = c.get('boot_ci95_pct'); cp = c.get('common_panel_improvement_pct')
            md.append(f"| {c['comparison']} | {c['new']} | {c['base']} | {c['new_mean']:.6f} / {c['base_mean']:.6f} | {c['improvement_pct']:.3f} | {c['folds_won']} | "
                      + ', '.join(f'{x:.2f}' for x in c['per_fold_improvement_pct']) + f" | {fmt(c['nb_p'])} | [{fmt(c['nb_ci95_pct'][0], 2)}, {fmt(c['nb_ci95_pct'][1], 2)}] | "
                      + (fmt(min(lofo)) if lofo else '—') + ' | ' + (f'[{bci[0]:.2f}, {bci[1]:.2f}]' if bci else '—') + ' | ' + (fmt(c.get('boot_prob_positive')) if bci else '—')
                      + ' | ' + (f"{cp:.3f} ({c['common_panel_folds_won']})" if cp is not None else '—') + f" | {c['verdict']} |")
        md.append(f"\nP2 综合标签：**{R['P2_label']}**\n")
        md.append('### 收回教师增益 R(k)（本 run，RNA）\n\n| 臂 | 教师 | 臂 sRMSE | 教师 sRMSE | R（5折均值） | 逐折 |\n|---|---|---:|---:|---:|---|')
        for r in R['recovery']:
            md.append(f"| {r['arm']} | {r['teacher']} | {r['arm_mean']:.6f} | {r['teacher_mean']:.6f} | {r['fraction_of_means']:.3f} | " + ', '.join(f'{x:.3f}' for x in r['per_fold']) + ' |')
        md.append('\n### 跨层 R(k)：RNA（本 run，教师 = 矩阵 M+RNA）vs P（protein-panel run，教师 = M+全P）\n\n| k | M+k基因 sRMSE | R_RNA | R_RNA 逐折 | M+k蛋白 sRMSE | R_P | R_P 逐折 |\n|---|---:|---:|---|---:|---:|---|')
        rr = {(r['arm'], r['teacher']): r for r in R['recovery']}; rp = {r['arm']: r for r in R['recovery_protein_run']}
        for k in KSEL:
            a, b_ = rr[(f'm_panel_k{k}', 'teacher_saved')], rp[f'P::m_panel_k{k}']
            md.append(f"| {k} | {a['arm_mean']:.6f} | {a['fraction_of_means']:.3f} | " + ', '.join(f'{x:.2f}' for x in a['per_fold']) + f" | {b_['arm_mean']:.6f} | {b_['fraction_of_means']:.3f} | " + ', '.join(f'{x:.2f}' for x in b_['per_fold']) + ' |')
        a = rr[('m_panel_k128', 'teacher_saved')]
        md.append(f"| 128 | {a['arm_mean']:.6f} | {a['fraction_of_means']:.3f} | " + ', '.join(f'{x:.2f}' for x in a['per_fold']) + ' | — | — | — |')
        a, b_ = rr[('m_panel_sel', 'teacher_saved')], rp['P::m_panel_sel']
        md.append(f"| sel(8–64) | {a['arm_mean']:.6f} | {a['fraction_of_means']:.3f} | " + ', '.join(f'{x:.2f}' for x in a['per_fold']) + f" | {b_['arm_mean']:.6f} | {b_['fraction_of_means']:.3f} | " + ', '.join(f'{x:.2f}' for x in b_['per_fold']) + ' |')
        a, b_ = rr[('m_allcand', 'teacher_saved')], rp['P::m_allcand']
        md.append(f"| 全候选 | {a['arm_mean']:.6f} | {a['fraction_of_means']:.3f} | " + ', '.join(f'{x:.2f}' for x in a['per_fold']) + f" | {b_['arm_mean']:.6f} | {b_['fraction_of_means']:.3f} | " + ', '.join(f'{x:.2f}' for x in b_['per_fold']) + ' |')
        a = rr[('m_panel_pg_k32', 'teacher_saved')]
        md.append(f"| 双层 32P+32G | {a['arm_mean']:.6f} | {a['fraction_of_means']:.3f} | " + ', '.join(f'{x:.2f}' for x in a['per_fold']) + ' | — | — | — |')
        md.append(f"\n教师 sRMSE：RNA {rr[('m_panel_k8', 'teacher_saved')]['teacher_mean']:.6f}；P {rp['P::m_panel_k8']['teacher_mean']:.6f}；m_only {R['arms']['m_only']['score']:.6f}\n")
        md.append('### 入选分布与面板重叠\n```\n' + json.dumps(dict(selection=R['selection'], panels=R['panels']), indent=1, ensure_ascii=False) + '\n```\n')
    open(f'{OUT}/result_tables.md', 'w').write('\n'.join(md))
    for sc, R in res.items():
        print('==', sc, 'P2:', R['P2_label'])
        for c in R['comparisons']:
            print(f"  [{c['verdict']}] {c['comparison']}: {c['improvement_pct']:.3f}% wins={c['folds_won']} p={fmt(c['nb_p'])}")
        for r in R['recovery']:
            if r['teacher'] == 'teacher_saved':
                print(f"  R {r['arm']}: {r['fraction_of_means']:.3f}")


if __name__ == '__main__':
    main()

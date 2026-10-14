"""Pre-registered comparisons for the protein-panel run (PLAN.md sections 9-10).

usage: python analyze.py RUN_OUTPUT_DIR   (expects new_cell_main/, tissue_main/, *_repro/ inside)
Reads only this run's outputs and source-inputs.npz (tissues, drug ids, protein ids). Writes analysis.json there.
"""
import csv
import json
import sys
import numpy as np
from scipy import stats

OUT = sys.argv[1]
SOURCE = '/public/home/mengxl/creed-research-runs/gdsc-metabolite-full-protein-20260908-0001/output/result/source-inputs.npz'
KS = (8, 16, 32, 64); RKS = (16, 32); SEEDS = (0, 1, 2, 3, 4)
MUS = (0., .1, 1., 10., 100., 1000.); WS = (0., .25, 1.); NA = 4
B = 2000
SRC = np.load(SOURCE, allow_pickle=False)
TIS, DRUG, PROT = SRC['tissues'], SRC['drug_ids'], SRC['protein_ids']


def load_scenario(sc):
    d = f'{OUT}/{sc}_main'
    summ = json.load(open(f'{d}/summary.json'))
    rows = list(csv.DictReader(open(f'{d}/fold_metrics.csv')))
    folds = []
    for f in range(5):
        z = np.load(f'{d}/fold-{f}_predictions.npz', allow_pickle=False)
        s = np.load(f'{d}/fold-{f}_selection.npz', allow_pickle=False)
        folds.append(dict(z={k: z[k] for k in z.files}, sel={k: s[k] for k in s.files if not k.startswith(('loss__', 'count__'))}))
    info = json.load(open(f'{d}/bank_info.json'))
    return summ, rows, folds, info


def sq_err(fold, arm):
    z = fold['z']; t = z['truth']; obs = np.isfinite(t)
    return np.where(obs, ((z[arm] - t) / z['train_sd']) ** 2, 0.), obs


def weighted_scores(E, obs, use, CW, DW):
    """CW: (B, nq) cell weights, DW: (B, nd) drug weights -> (B,) fold scores."""
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


def common_drugs(folds):
    sets = [set(DRUG[fo['z']['panel']][fo['z']['evaluated']].tolist()) for fo in folds]
    return set.intersection(*sets)


class Scen:
    def __init__(self, sc):
        self.sc = sc
        self.summ, rows, self.folds, self.info = load_scenario(sc)
        self.fs = {}
        for r in rows:
            self.fs.setdefault(r['arm'], [None] * 5)[int(r['fold'])] = (float(r['score']), float(r['spearman']), float(r['centered_mse']))
        self.ratio = float(np.mean([len(fo['z']['held']) / len(fo['z']['fit']) for fo in self.folds]))
        self.W = build_weights(sc, self.folds, np.random.default_rng(20261009 + (0 if sc == 'new_cell' else 1)))
        self.common = common_drugs(self.folds)
        self.groups = {}
        pre = ['random'] if sc == 'tissue' else ['random', 'random_tissue']
        for p in pre:
            for k in RKS:
                self.groups[f'{p}_k{k}_mean'] = [f'{p}_k{k}_s{s}' for s in SEEDS]
        self._boot = {}; self._common = {}

    def members(self, arm):
        return self.groups.get(arm, [arm])

    def fold_scores(self, arm):
        return np.mean([[self.fs[a][f][0] for f in range(5)] for a in self.members(arm)], axis=0)

    def fold_metric(self, arm, i):
        return np.mean([[self.fs[a][f][i] for f in range(5)] for a in self.members(arm)], axis=0)

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


def compare(S, new, base, label, robust=True):
    n, b = S.fold_scores(new), S.fold_scores(base)
    imp = (b.mean() - n.mean()) / b.mean() * 100
    d = (b - n) / b * 100; wins = int((n < b).sum())
    se = np.sqrt((1 / 5 + S.ratio) * d.var(ddof=1)) if d.var(ddof=1) > 0 else np.nan
    tval = d.mean() / se if se and np.isfinite(se) else np.nan
    p = float(2 * stats.t.sf(abs(tval), 4)) if np.isfinite(tval) else np.nan
    q = stats.t.ppf(.975, 4)
    out = dict(comparison=label, scenario=S.sc, new=new, base=base, new_mean=float(n.mean()), base_mean=float(b.mean()),
               improvement_pct=float(imp), per_fold_improvement_pct=d.round(4).tolist(), folds_won=f'{wins}/5',
               nb_t=float(tval), nb_p=p, nb_ci95_pct=[float(d.mean() - q * se), float(d.mean() + q * se)],
               verdict=verdict(imp, wins),
               new_fold_scores=n.tolist(), base_fold_scores=b.tolist())
    if robust:
        lofo = []
        for f in range(5):
            keep = np.arange(5) != f
            lofo.append(float((b[keep].mean() - n[keep].mean()) / b[keep].mean() * 100))
        out['leave_one_fold_out_pct'] = [round(x, 4) for x in lofo]
        out['lofo_all_positive'] = bool(min(lofo) > 0)
        bn, bb = S.boot(new), S.boot(base)
        bi = (bb - bn) / bb * 100
        out['boot_ci95_pct'] = [float(np.percentile(bi, 2.5)), float(np.percentile(bi, 97.5))]
        out['boot_prob_positive'] = float((bi > 0).mean())
        if S.sc == 'tissue':
            cn, cb = S.common_scores(new), S.common_scores(base)
            out['common_panel_drugs'] = len(S.common)
            out['common_panel_improvement_pct'] = float((cb.mean() - cn.mean()) / cb.mean() * 100)
            out['common_panel_folds_won'] = f'{int((cn < cb).sum())}/5'
            out['common_panel_new_mean'] = float(cn.mean()); out['common_panel_base_mean'] = float(cb.mean())
    return out


def best(S, arms):
    return min(arms, key=lambda a: S.fold_scores(a).mean())


def recovery(S, arm, teacher):
    m, p, t = S.fold_scores('m_only'), S.fold_scores(arm), S.fold_scores(teacher)
    return dict(arm=arm, teacher=teacher, teacher_mean=float(t.mean()),
                fraction_of_means=float((m.mean() - p.mean()) / (m.mean() - t.mean())),
                per_fold=((m - p) / (m - t)).round(4).tolist())


def sel_dist(S, arm, decode):
    cnt = {}
    for fo in S.folds:
        s = fo['sel'][arm][fo['z']['evaluated']]
        for v in s:
            key = decode(int(v)); cnt[key] = cnt.get(key, 0) + 1
    return dict(sorted(cnt.items()))


def panel_overlap(S):
    out = {}
    for k in (8, 16, 32, 64):
        sets = [set(S.info[f'fold-{f}/outer']['top64'][:k]) for f in range(5)]
        jac = [len(sets[i] & sets[j]) / len(sets[i] | sets[j]) for i in range(5) for j in range(i + 1, 5)]
        out[f'k{k}_mean_pairwise_jaccard_outer'] = float(np.mean(jac))
        out[f'k{k}_in_all_5_outer'] = [str(PROT[i]) for i in sorted(set.intersection(*sets))]
    freq = {}
    for f in range(5):
        for i in S.info[f'fold-{f}/outer']['top64'][:16]:
            freq[str(PROT[i])] = freq.get(str(PROT[i]), 0) + 1
    out['outer_top16_frequency'] = dict(sorted(freq.items(), key=lambda x: -x[1]))
    out['candidate_counts'] = {k: v.get('candidate_count') for k, v in S.info.items()}
    return out


def main():
    res = {}
    for sc in ('new_cell', 'tissue'):
        S = Scen(sc)
        R = dict(reproduction=json.load(open(f'{OUT}/{sc}_main/summary.json'))['reproduction'],
                 repro_stage=json.load(open(f'{OUT}/{sc}_repro/summary.json'))['reproduction'])
        arms = sorted(S.fs) + sorted(S.groups)
        R['arms'] = {a: dict(score=float(S.fold_scores(a).mean()), spearman=float(np.nanmean(S.fold_metric(a, 1))),
                             centered_mse=float(S.fold_metric(a, 2).mean()), fold_scores=S.fold_scores(a).round(6).tolist())
                     for a in arms}
        C = []
        # P1
        C.append(compare(S, 'm_panel_sel', 'm_only', 'P1 面板价值：m_panel_sel vs m_only'))
        if sc == 'new_cell':
            C.append(compare(S, 'm_panel_tissue_sel', 'm_tissue', 'P1b 面板价值（含组织）：m_panel_tissue_sel vs m_tissue'))
        # P2
        for k in RKS:
            C.append(compare(S, f'm_panel_k{k}', f'random_k{k}_mean', f'P2 监督 vs 随机 k={k}'))
        rb = best(S, [f'random_k{k}_mean' for k in RKS])
        C.append(compare(S, 'm_panel_sel', rb, f'P2c m_panel_sel vs 最强随机均值（{rb}）'))
        if sc == 'new_cell':
            for k in RKS:
                C.append(compare(S, f'm_panel_tissue_k{k}', f'random_tissue_k{k}_mean', f'P2t（描述）含组织 监督 vs 随机 k={k}'))
        # P4
        C.append(compare(S, 'm_panel_sel__protein', 'm_panel_sel', 'P4 CREED P图增量：m_panel_sel__protein vs m_panel_sel'))
        direct = ['m_only'] + [f'm_panel_k{k}' for k in KS] + ['m_panel_sel']
        if sc == 'tissue':
            direct += ['vfm'] + [f'vfm_panel_k{k}' for k in KS] + ['vfm_panel_sel']
        bd = best(S, direct)
        C.append(compare(S, 'm_panel_sel__protein', bd, f'P4-保守 m_panel_sel__protein vs 同输入最强直接臂（{bd}）'))
        for g in ('mask', 'shuffle'):
            C.append(compare(S, 'm_panel_sel__protein', f'm_panel_sel__{g}', f'P4 对照 数值P图 vs {g} 图', robust=False))
        if sc == 'new_cell':
            C.append(compare(S, 'm_panel_tissue_sel__protein', 'm_panel_tissue_sel', 'P4b CREED P图增量（含组织）'))
            direct_t = direct + ['tissue_only', 'm_tissue'] + [f'm_panel_tissue_k{k}' for k in KS] + ['m_panel_tissue_sel']
            bdt = best(S, direct_t)
            C.append(compare(S, 'm_panel_tissue_sel__protein', bdt, f'P4b-保守 vs 同输入最强直接臂（{bdt}）'))
            for g in ('mask', 'shuffle'):
                C.append(compare(S, 'm_panel_tissue_sel__protein', f'm_panel_tissue_sel__{g}', f'P4b 对照 数值P图 vs {g} 图', robust=False))
            anyk = [a for a in direct_t + ['m_only__protein', 'm_tissue__protein']]
            ba = best(S, anyk)
            C.append(compare(S, 'm_panel_tissue_sel__protein', ba, f'附 m_panel_tissue_sel__protein vs 输入⊆的最强任意臂（{ba}）'))
            # P5
            C.append(compare(S, 'm_panel_tissue_sel', 'tissue_only', 'P5 m_panel_tissue_sel vs tissue_only'))
            C.append(compare(S, 'm_panel_tissue_sel__protein', 'm_tissue__protein', 'P5 m_panel_tissue_sel__protein vs m_tissue__protein'))
            C.append(compare(S, 'm_panel_tissue_sel__protein', 'tissue_only', 'P5 m_panel_tissue_sel__protein vs tissue_only'))
        else:
            C.append(compare(S, 'vfm_panel_sel__protein', 'vfm_panel_sel', 'P4b CREED P图增量（vfm 底座）'))
            C.append(compare(S, 'vfm_panel_sel__protein', bd, f'P4b-保守 vfm_panel_sel__protein vs 同输入最强直接臂（{bd}）'))
            for g in ('mask', 'shuffle'):
                C.append(compare(S, 'vfm_panel_sel__protein', f'vfm_panel_sel__{g}', f'P4b 对照 数值P图 vs {g} 图（vfm）', robust=False))
        # secondary
        mo = ['m_only', 'm_only__protein'] + (['vfm', 'vfm__protein'] if sc == 'tissue' else [])
        bm = best(S, mo)
        C.append(compare(S, 'm_panel_sel', bm, f'次要 m_panel_sel vs 同数据最强仅M臂（{bm}）'))
        C.append(compare(S, 'm_panel_sel__protein', bm, f'次要 m_panel_sel__protein vs 同数据最强仅M臂（{bm}）'))
        if sc == 'tissue':
            C.append(compare(S, 'vfm_panel_sel', 'vfm', '次要 vfm_panel_sel vs vfm'))
            C.append(compare(S, 'vfm_panel_sel__protein', 'vfm__protein', '次要 vfm_panel_sel__protein vs vfm__protein（当前候选）'))
        C.append(compare(S, 'm_allcand', 'm_panel_sel', '次要 全候选蛋白 vs 小面板', robust=False))
        C.append(compare(S, 'teacher_saved', 'm_panel_sel', '次要 实测全P教师 vs 小面板', robust=False))
        R['comparisons'] = C
        R['recovery'] = [recovery(S, a, t) for a in ('m_panel_sel', 'm_panel_sel__protein', 'm_panel_k8', 'm_panel_k16',
                                                     'm_panel_k32', 'm_panel_k64', 'm_allcand')
                         for t in ('teacher_saved', 'teacher_repro')]
        if sc == 'new_cell':
            R['recovery'] += [recovery(S, a, 'teacher_saved') for a in ('m_panel_tissue_sel', 'm_panel_tissue_sel__protein')]
        else:
            R['recovery'] += [recovery(S, a, 'teacher_saved') for a in ('vfm_panel_sel', 'vfm_panel_sel__protein')]
        R['selection'] = {
            'm_panel_sel_k': sel_dist(S, 'm_panel_sel', lambda v: f'k{KS[v // NA]}'),
            'm_panel_sel__protein_k': sel_dist(S, 'm_panel_sel__protein', lambda v: f'k{KS[v // (6 * NA)]}'),
            'm_panel_sel__protein_mu': sel_dist(S, 'm_panel_sel__protein', lambda v: f'mu{MUS[(v // NA) % 6]}'),
        }
        if sc == 'new_cell':
            R['selection'].update({
                'm_panel_tissue_sel_k': sel_dist(S, 'm_panel_tissue_sel', lambda v: f'k{KS[v // (3 * NA)]}'),
                'm_panel_tissue_sel_w': sel_dist(S, 'm_panel_tissue_sel', lambda v: f'w{WS[(v // NA) % 3]}'),
                'm_panel_tissue_sel__protein_mu': sel_dist(S, 'm_panel_tissue_sel__protein', lambda v: f'mu{MUS[(v // NA) % 6]}'),
                'm_panel_tissue_sel__protein_w': sel_dist(S, 'm_panel_tissue_sel__protein', lambda v: f'w{WS[(v // (6 * NA)) % 3]}'),
            })
        else:
            R['selection']['vfm_panel_sel__protein_mu'] = sel_dist(S, 'vfm_panel_sel__protein', lambda v: f'mu{MUS[(v // NA) % 6]}')
        R['panels'] = panel_overlap(S)
        R['nb_ratio_test_train'] = S.ratio
        R['common_panel_drugs'] = len(S.common)
        res[sc] = R
        print(json.dumps(dict(event='scenario_done', scenario=sc)), flush=True)
    json.dump(res, open(f'{OUT}/analysis.json', 'w'), indent=1, ensure_ascii=False)
    for sc, R in res.items():
        print('==', sc)
        for a, v in sorted(R['arms'].items(), key=lambda x: x[1]['score']):
            print(f"  {a:<34} {v['score']:.6f}  rho={v['spearman']:.4f}  cmse={v['centered_mse']:.4f}  {v['fold_scores']}")
        for c in R['comparisons']:
            print(f"  [{c['verdict']}] {c['comparison']}: {c['improvement_pct']:.3f}% wins={c['folds_won']} p={c['nb_p']:.3f} "
                  f"lofo_min={min(c.get('leave_one_fold_out_pct', [np.nan])):.3f} boot={c.get('boot_ci95_pct')}")
        for r in R['recovery']:
            print(f"  recovery {r['arm']} / {r['teacher']}: {r['fraction_of_means']:.3f} per_fold={r['per_fold']}")


if __name__ == '__main__':
    main()

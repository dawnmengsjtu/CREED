"""Pre-registered analysis for the CRISPR gene-effect endpoint replication (PLAN.md sections 7-9).

usage: python analyze.py OUTPUT_ROOT     (contains new_cell/ and tissue/ written by crispr_endpoint.py)
Writes OUTPUT_ROOT/{fold_metrics.csv, comparisons.csv, summary.json, result_tables.md}.
Reads only this run's outputs plus source-inputs.npz (tissue labels for the grouped bootstrap).
"""
import csv
import json
import sys

import numpy as np
from scipy import stats

ROOT = '/public/home/mengxl/creed-research-runs'
SOURCE = ROOT + '/gdsc-metabolite-full-protein-20260908-0001/output/result/source-inputs.npz'
TIS = np.load(SOURCE, allow_pickle=False)['tissues']

B = 2000
LAYERS = ('M', 'RNA', 'P')
PAIRS = (('M', 'RNA'), ('M', 'P'), ('RNA', 'P'), ('RNA', 'M'), ('P', 'M'), ('P', 'RNA'))
KS = (8, 16, 32, 64)
RKS = (16, 32)
SEEDS = (0, 1, 2, 3, 4)
SCEN_ID = {'new_cell': 0, 'tissue': 1}
ROOM_MIN = 0.3
TEACHER = {'P': 'M+Pval', 'G': 'M+RNA'}
PANEL_NAME = {'P': '蛋白面板', 'G': '基因面板'}

# GDSC drug-endpoint reference values (from the matrix and panel runs; for side-by-side comparison only)
GDSC_MAP = {'new_cell': {'RNA': (3.08, 0.34), 'P': (2.40, 0.29)}, 'tissue': {'RNA': (2.19, 0.33), 'P': (2.21, 0.33)}}
GDSC_PANEL = {'new_cell': {'P': (0.48, 0.56, 0.65, 0.72), 'G': (0.37, 0.56, 0.72, 0.76)},
              'tissue': {'P': (0.64, 0.57, 0.66, 0.74), 'G': (0.45, 0.64, 0.78, 0.76)}}


def meas_name(*layers):
    return '+'.join(l for l in LAYERS if l in layers)


class Scen:
    def __init__(self, root, sc):
        self.sc = sc
        d = f'{root}/{sc}'
        self.summ = json.load(open(f'{d}/summary.json'))
        self.fs = {}
        for r in csv.DictReader(open(f'{d}/fold_metrics.csv')):
            self.fs.setdefault(r['arm'], [None] * 5)[int(r['fold'])] = (
                float(r['score']), float(r['spearman']), float(r['centered_mse']))
        self.folds = [dict(np.load(f'{d}/fold-{f}_predictions.npz', allow_pickle=False)) for f in range(5)]
        self.ratio = float(np.mean([int(r['n_query_crispr']) / int(r['n_fit_crispr'])
                                    for r in csv.DictReader(open(f'{d}/fold_metrics.csv')) if r['arm'] == 'M']))
        self.ratio_all = float(np.mean([int(r['n_query']) / int(r['n_fit'])
                                        for r in csv.DictReader(open(f'{d}/fold_metrics.csv')) if r['arm'] == 'M']))
        self.groups = {}
        for tag in ('P', 'G'):
            for k in RKS:
                self.groups[f'random_{tag}_k{k}_mean'] = [f'random_{tag}_k{k}_s{s}' for s in SEEDS]
        gset = [set(z['gene_ids'][z['evaluated']].tolist()) for z in self.folds]
        self.common = set.intersection(*gset)
        self.masks = {'main': [z['evaluated'] for z in self.folds],
                      'sd03': [z['evaluated'] & (z['fit_sd_all'] >= 0.3) for z in self.folds],
                      'common': [z['evaluated'] & np.isin(z['gene_ids'], list(self.common)) for z in self.folds]}
        self.W = self._weights(np.random.default_rng(20261009 + SCEN_ID[sc]))
        self._boot, self._masked = {}, {}

    # ---- grouped bootstrap weights (targets shared across folds; cells grouped per scenario)
    def _weights(self, rng):
        ids = sorted({str(x) for z in self.folds for x in z['gene_ids']})
        pos = {g: i for i, g in enumerate(ids)}
        draw = rng.integers(0, len(ids), size=(B, len(ids)))
        gc = np.zeros((B, len(ids)))
        np.add.at(gc, (np.repeat(np.arange(B), len(ids)), draw.ravel()), 1)
        out = []
        for z in self.folds:
            GW = gc[:, [pos[str(x)] for x in z['gene_ids']]]
            held = z['held']
            tis = TIS[held]
            CW = np.zeros((B, len(held)))
            if self.sc == 'new_cell':
                grp = sorted(set(tis.tolist()))
                gi = np.array([grp.index(t) for t in tis])
                g = rng.integers(0, len(grp), size=(B, len(grp)))
                cc = np.zeros((B, len(grp)))
                np.add.at(cc, (np.repeat(np.arange(B), len(grp)), g.ravel()), 1)
                CW = cc[:, gi]
            else:
                for t in sorted(set(tis.tolist())):
                    cells = np.flatnonzero(tis == t)
                    g = rng.integers(0, len(cells), size=(B, len(cells)))
                    np.add.at(CW, (np.repeat(np.arange(B), len(cells)), cells[g].ravel()), 1)
            out.append((CW, GW))
        return out

    def members(self, arm):
        return self.groups.get(arm, [arm])

    def fold_scores(self, arm, i=0):
        return np.mean([[self.fs[a][f][i] for f in range(5)] for a in self.members(arm)], axis=0)

    def sq_err(self, f, arm):
        z = self.folds[f]
        t = z['truth']
        obs = np.isfinite(t)
        return np.where(obs, ((z['pred__' + arm] - t) / z['train_sd']) ** 2, 0.), obs

    def masked_scores(self, arm, which):
        key = (arm, which)
        if key not in self._masked:
            per = []
            for a in self.members(arm):
                s = []
                for f in range(5):
                    E, obs = self.sq_err(f, a)
                    use = self.masks[which][f]
                    mse = E.sum(0) / np.maximum(obs.sum(0), 1)
                    s.append(float(np.sqrt(mse[use].mean())))
                per.append(s)
            self._masked[key] = np.mean(per, axis=0)
        return self._masked[key]

    def boot(self, arm):
        if arm not in self._boot:
            per = []
            for a in self.members(arm):
                s = np.zeros((B, 5))
                for f in range(5):
                    E, obs = self.sq_err(f, a)
                    CW, GW = self.W[f]
                    num = CW @ E
                    den = CW @ obs.astype(float)
                    ok = (den > 0) & self.folds[f]['evaluated'][None, :] & (GW > 0)
                    mse = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
                    w = np.where(ok, GW, 0.)
                    s[:, f] = np.sqrt((w * mse).sum(1) / w.sum(1))
                per.append(s)
            self._boot[arm] = np.mean(per, axis=0).mean(1)
        return self._boot[arm]


def nb(d, ratio):
    v = float(np.var(d, ddof=1))
    se = float(np.sqrt((1 / 5 + ratio) * v)) if v > 0 else 0.
    t = float(np.mean(d) / se) if se > 0 else float('inf') * np.sign(np.mean(d))
    p = float(2 * stats.t.sf(abs(t), 4)) if np.isfinite(t) else 0.
    q = float(stats.t.ppf(.975, 4))
    return t, p, [float(np.mean(d) - q * se), float(np.mean(d) + q * se)], se


def gain(S, new, base, kind='', label='', robust=False):
    a, b = S.fold_scores(new), S.fold_scores(base)
    d = (b - a) / b * 100
    t, p, ci, _ = nb(d, S.ratio)
    out = dict(scenario=S.sc, kind=kind, comparison=label or f'{new} vs {base}', new=new, base=base,
               gain_pct=float(d.mean()), gain_pct_ratio_of_means=float((b.mean() - a.mean()) / b.mean() * 100),
               per_fold_pct=[round(float(x), 4) for x in d], wins=int((a < b).sum()),
               nb_t=t, nb_p=p, nb_ci95_pct=ci, new_mean=float(a.mean()), base_mean=float(b.mean()),
               new_fold_scores=[round(float(x), 6) for x in a], base_fold_scores=[round(float(x), 6) for x in b])
    if robust:
        lofo = [float((b[m].mean() - a[m].mean()) / b[m].mean() * 100) for m in (np.arange(5) != k for k in range(5))]
        out['lofo_pct'] = [round(x, 4) for x in lofo]
        out['lofo_min_pct'] = float(min(lofo))
        bn, bb = S.boot(new), S.boot(base)
        bi = (bb - bn) / bb * 100
        out['boot_ci95_pct'] = [float(np.percentile(bi, 2.5)), float(np.percentile(bi, 97.5))]
        out['boot_prob_positive'] = float((bi > 0).mean())
        for which in ('sd03', 'common'):
            an, ab = S.masked_scores(new, which), S.masked_scores(base, which)
            dm = (ab - an) / ab * 100
            out[f'{which}_gain_pct'] = float(dm.mean())
            out[f'{which}_wins'] = int((an < ab).sum())
    return out


def recovery(S, arm, teacher, which='main'):
    if which == 'main':
        m, p, t = S.fold_scores('M'), S.fold_scores(arm), S.fold_scores(teacher)
    else:
        m, p, t = S.masked_scores('M', which), S.masked_scores(arm, which), S.masked_scores(teacher, which)
    pf = (m - p) / (m - t)
    return dict(arm=arm, teacher=teacher, R=float((m.mean() - p.mean()) / (m.mean() - t.mean())),
                R_per_fold=[round(float(x), 4) for x in pf], R_mean_of_per_fold=float(pf.mean()),
                arm_mean=float(p.mean()), M_mean=float(m.mean()), teacher_mean=float(t.mean()))


def fmt(g):
    return f"{g['gain_pct']:+.3f}% ({g['wins']}/5, p={g['nb_p']:.3f})"


def analyze(root):
    out, comps, md = {}, [], []
    for sc in ('new_cell', 'tissue'):
        S = Scen(root, sc)
        res = dict(r_crispr=S.ratio, r_all_cells=S.ratio_all,
                   reproduction_gate=S.summ['reproduction_gate'],
                   evaluated_per_fold=S.summ['evaluated_per_fold'], dropped_per_fold=S.summ['dropped_per_fold'],
                   n_common_targets=len(S.common),
                   evaluated_per_bank={k: {b: dict(sd02=v[b]['evaluated_this_bank'], sd03=v[b]['evaluated_sd03_this_bank'],
                                                   n_fit_crispr=v[b]['n_fit_crispr'], n_query_crispr=v[b]['n_query_crispr'])
                                            for b in v} for k, v in S.summ['bank_info'].items()},
                   cells={}, triple={}, T={}, secondary={}, panels={}, criteria={})

        def add(g):
            comps.append({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in g.items()})
            return g

        md.append(f"\n## {sc}\n\n5 折均值 sRMSE（全部臂）\n\n| 臂 | sRMSE | 折0 | 折1 | 折2 | 折3 | 折4 | Spearman | centered MSE |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for a in S.fs:
            v = S.fold_scores(a)
            md.append(f"| {a} | {v.mean():.6f} | " + ' | '.join(f'{x:.4f}' for x in v)
                      + f" | {S.fold_scores(a, 1).mean():.4f} | {S.fold_scores(a, 2).mean():.4f} |")

        # ---- map
        for x in LAYERS:
            cells = {}
            for y in LAYERS:
                if y == x:
                    continue
                hd = add(gain(S, meas_name(x, y), x, 'headroom', f'room {x}->{y}', robust=(x == 'M')))
                rec = add(gain(S, f'{x}+v{y}', x, 'recovered', f'virtual {x}+v{y} vs {x}', robust=(x == 'M')))
                sh = add(gain(S, f'{x}+v{y}_shuf', x, 'shuffle', f'shuffle {x}+v{y}_shuf vs {x}'))
                vs = add(gain(S, f'{x}+v{y}', f'{x}+v{y}_shuf', 'virt_vs_shuf', f'{x}+v{y} vs shuffle', robust=(x == 'M')))
                r2 = [S.summ['bank_info'][str(f)][b]['generators'][f'{x}->{y}']['oof_r2_mean']
                      for f in range(5) for b in ('inner-0', 'inner-1', 'inner-2', 'outer')]
                cells[y] = dict(headroom=hd, recovered=rec, shuffle=sh, virt_vs_shuf=vs,
                                share=(rec['gain_pct'] / hd['gain_pct'] if hd['gain_pct'] > ROOM_MIN else None),
                                generator_oof_r2_mean=float(np.mean(r2)))
            res['cells'][x] = cells
            res['triple'][x] = add(gain(S, 'M+RNA+P', x, 'triple', f'M+RNA+P vs {x}'))

        md.append("\n地图（行 = 部署层 X，列 = 借用层 Y）。room = gain(X+Y vs X)；recovered = gain(X+vY vs X)；shuf = gain(X+vY_shuf vs X)；virt vs shuf；share = recovered/room。\n")
        md.append("| X \\ Y | 读出 | " + ' | '.join(LAYERS) + " |\n|---|---|" + '---|' * len(LAYERS))
        for x in LAYERS:
            def col(y, key):
                if y == x:
                    return '—'
                c = res['cells'][x][y]
                if key == 'share':
                    return '无空间' if c['share'] is None else f"{c['share'] * 100:.0f}%"
                if key == 'r2':
                    return f"{c['generator_oof_r2_mean']:.3f}"
                if key == 'gdsc':
                    g = GDSC_MAP.get(sc, {}).get(y) if x == 'M' else None
                    return '—' if g is None else f"room {g[0]:+.2f}% / share {g[1] * 100:.0f}%"
                return fmt(c[key])
            for key, name in (('headroom', 'room（实测 Y）'), ('recovered', 'recovered（虚拟 Y）'), ('shuffle', 'shuf 对照'),
                              ('virt_vs_shuf', '虚拟 vs 置乱'), ('share', 'share'), ('r2', '生成器 OOF R²'), ('gdsc', 'GDSC 药物端点')):
                md.append(f"| {x} | {name} | " + ' | '.join(col(y, key) for y in LAYERS) + ' |')

        if sc == 'new_cell':
            md.append("\n组织标签 T 行：\n\n| 比较 | 结果 |\n|---|---|")
            for x in LAYERS:
                g = add(gain(S, f'{x}+T', x, 'T', f'{x}+T vs {x}'))
                res['T'][x] = g
                md.append(f"| {x}+T vs {x} | {fmt(g)} |")
            for lab, a, b in (('M+vRNA+T vs M+T', 'M+vRNA+T', 'M+T'), ('M+vRNA+T vs M+vRNA', 'M+vRNA+T', 'M+vRNA'),
                              ('M vs T', 'M', 'T')):
                g = add(gain(S, a, b, 'T', lab))
                res['T'][lab] = g
                md.append(f"| {lab} | {fmt(g)} |")

        sec = {'M_Pval_room': add(gain(S, 'M+Pval', 'M', 'headroom_val', 'room M->Pval', robust=True)),
               'Pval_vs_P': add(gain(S, 'Pval', 'P', 'secondary', 'Pval vs P')),
               'null_vs_M': add(gain(S, 'M', 'null', 'secondary', 'M vs null'))}
        sec['share_val'] = (res['cells']['M']['P']['recovered']['gain_pct'] / sec['M_Pval_room']['gain_pct']
                            if sec['M_Pval_room']['gain_pct'] > ROOM_MIN else None)
        res['secondary'] = sec
        res['null_score'] = float(S.fold_scores('null').mean())

        # ---- panels
        md.append("\n面板规律（头在 0.5*K_M + 0.5*K_panel 上重训；教师 = M+Pval（蛋白）/ M+RNA（基因））\n")
        md.append("| 层 | k | sRMSE | vs M 改善% | 胜折 | NB p | R(k) | R(k) GDSC | R(k) SD≥0.3 | R(k) 共同集 |\n|---|---:|---:|---:|---|---:|---:|---:|---:|---:|")
        for tag in ('P', 'G'):
            pr = {}
            for i, k in enumerate(KS):
                arm = f'm_panel_{tag}_k{k}'
                g = add(gain(S, arm, 'M', 'panel', f'{arm} vs M', robust=(k in (32, 64))))
                rc = recovery(S, arm, TEACHER[tag])
                rc_sd = recovery(S, arm, TEACHER[tag], 'sd03')
                rc_cm = recovery(S, arm, TEACHER[tag], 'common')
                pr[k] = dict(vs_M=g, recovery=rc, recovery_sd03=rc_sd, recovery_common=rc_cm,
                             gdsc_R=GDSC_PANEL[sc][tag][i])
                md.append(f"| {PANEL_NAME[tag]} | {k} | {g['new_mean']:.6f} | {g['gain_pct']:+.3f}% | {g['wins']}/5 | "
                          f"{g['nb_p']:.3f} | {rc['R']:.3f} | {GDSC_PANEL[sc][tag][i]:.2f} | {rc_sd['R']:.3f} | {rc_cm['R']:.3f} |")
            for k in RKS:
                g = add(gain(S, f'm_panel_{tag}_k{k}', f'random_{tag}_k{k}_mean', 'panel_vs_random',
                             f'm_panel_{tag}_k{k} vs random_{tag}_k{k}_mean'))
                pr[f'vs_random_k{k}'] = g
                md.append(f"| {PANEL_NAME[tag]} | {k} | 监督 vs 随机 | {g['gain_pct']:+.3f}% | {g['wins']}/5 | {g['nb_p']:.3f} | — | — | — | — |")
            pr['teacher_room'] = add(gain(S, TEACHER[tag], 'M', 'teacher_room', f'{TEACHER[tag]} vs M'))
            res['panels'][tag] = pr

        # ---- pre-registered criteria
        r1a, r1b = res['cells']['M']['RNA']['headroom'], res['cells']['M']['P']['headroom']
        R1 = bool(r1a['gain_pct'] > 0 and r1a['wins'] >= 4 and r1b['gain_pct'] > 0 and r1b['wins'] >= 4)
        r2 = {}
        for y in ('RNA', 'P'):
            rec, vs = res['cells']['M'][y]['recovered'], res['cells']['M'][y]['virt_vs_shuf']
            r2[f'M+v{y}'] = bool(rec['wins'] >= 4 and rec['nb_p'] < 0.10 and vs['wins'] >= 4)
        R2 = bool(any(r2.values()))
        r3 = {f'{meas_name("M", y)} vs {y}': add(gain(S, meas_name('M', y), y, 'R3', f'M added to {y}'))
              for y in ('RNA', 'P')}
        R3 = {k: dict(gain_pct=v['gain_pct'], wins=v['wins'],
                      gdsc_pattern_holds=bool(v['gain_pct'] <= 0.3 or v['wins'] <= 2)) for k, v in r3.items()}
        R4d = {tag: dict(R32=res['panels'][tag][32]['recovery']['R'], R64=res['panels'][tag][64]['recovery']['R'])
               for tag in ('P', 'G')}
        R4 = bool(all(v['R32'] >= 0.5 and v['R64'] >= 0.5 for v in R4d.values()))
        n_pass = int(R1) + int(R2) + int(R4)
        res['criteria'] = dict(R1=dict(passed=R1, M_RNA=fmt(r1a), M_P=fmt(r1b)),
                               R2=dict(passed=R2, detail=r2), R3=R3, R4=dict(passed=R4, detail=R4d),
                               n_passed=n_pass,
                               verdict=('规律复现' if n_pass == 3 else ('部分复现' if n_pass == 2 else '不复现')))
        md.append(f"\n**预注册判据**：R1 {'过' if R1 else '不过'}；R2 {'过' if R2 else '不过'}（{r2}）；"
                  f"R4 {'过' if R4 else '不过'}（{R4d}）；R3（只报告）：" +
                  '；'.join(f"{k}: {v['gain_pct']:+.3f}%, {v['wins']}/5, GDSC 格局{'成立' if v['gdsc_pattern_holds'] else '不成立'}"
                            for k, v in R3.items()) +
                  f"\n\n**{sc} 判定 = {n_pass}/3 → {res['criteria']['verdict']}**")
        out[sc] = res

    json.dump(out, open(f'{root}/summary.json', 'w'), indent=1, default=float)
    keys = sorted({k for c in comps for k in c})
    with open(f'{root}/comparisons.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(comps)
    rows = []
    for sc in ('new_cell', 'tissue'):
        for r in csv.DictReader(open(f'{root}/{sc}/fold_metrics.csv')):
            rows.append(dict(scenario=sc, **r))
    with open(f'{root}/fold_metrics.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    open(f'{root}/result_tables.md', 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    analyze(sys.argv[1])

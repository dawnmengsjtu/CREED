"""Pre-registered analysis for the cross-layer HEADROOM x RECOVERABILITY matrix (PLAN.md section 8).

usage: python analyze.py OUTPUT_ROOT   (contains new_cell/ and tissue/ written by crosslayer.py)
Writes analysis.json, comparisons.csv, matrix_<scenario>.csv, matrix_tables.md into OUTPUT_ROOT.
"""
import csv
import json
import sys

import numpy as np
from scipy.stats import t as tdist

LAYERS = ('M', 'RNA', 'P')
PAIRS = (('M', 'RNA'), ('M', 'P'), ('RNA', 'P'), ('RNA', 'M'), ('P', 'M'), ('P', 'RNA'))
ROOM_MIN = 0.3


def meas_name(*layers):
    return '+'.join(l for l in LAYERS if l in layers)


def load_scenario(root, sc):
    rows = list(csv.DictReader(open(f'{root}/{sc}/fold_metrics.csv')))
    scores, spear, cmse, ratio = {}, {}, {}, []
    for r in rows:
        scores.setdefault(r['arm'], [None] * 5)[int(r['fold'])] = float(r['score'])
        spear.setdefault(r['arm'], [None] * 5)[int(r['fold'])] = float(r['spearman'])
        cmse.setdefault(r['arm'], [None] * 5)[int(r['fold'])] = float(r['centered_mse'])
        if r['arm'] == 'M':
            ratio.append(int(r['n_held']) / int(r['n_fit']))
    summary = json.load(open(f'{root}/{sc}/summary.json'))
    bank = json.load(open(f'{root}/{sc}/bank_info.json'))
    return dict(scores={a: np.array(v) for a, v in scores.items()}, spearman=spear, cmse=cmse,
                r=float(np.mean(ratio)), summary=summary, bank=bank)


def gain(D, new, base):
    a, b = D['scores'][new], D['scores'][base]
    d = (b - a) / b * 100
    m = float(d.mean()); v = float(d.var(ddof=1)); se = float(np.sqrt((1 / 5 + D['r']) * v))
    tstat = m / se if se > 0 else float('inf') * np.sign(m)
    p = float(2 * tdist.sf(abs(tstat), 4)) if np.isfinite(tstat) else 0.
    q = float(tdist.ppf(.975, 4))
    return dict(new=new, base=base, gain=m, per_fold=[round(float(x), 4) for x in d], wins=int((a < b).sum()),
                p=p, ci=[m - q * se, m + q * se], new_mean=float(a.mean()), base_mean=float(b.mean()))


def room_label(g):
    if g['gain'] >= 1 and g['wins'] >= 4:
        return '有空间'
    if g['gain'] >= ROOM_MIN:
        return '小幅空间'
    return '无空间'


def recovery_label(rec, vs):
    ok1 = rec['gain'] > 0 and rec['wins'] >= 4
    ok2 = vs['gain'] > 0 and vs['wins'] >= 4
    if ok1 and ok2:
        return '有恢复'
    if ok1:
        return '增益可由一般平滑解释'
    return '无恢复'


def share_label(share):
    if share is None:
        return '无空间'
    if share >= .5:
        return '大部分收回'
    if share >= .1:
        return '部分收回'
    return '基本未收回'


def fmt(g):
    return f"{g['gain']:+.3f}% ({g['wins']}/5, p={g['p']:.3f})"


def analyze(root):
    out, comps, md = {}, [], []
    for sc in ('new_cell', 'tissue'):
        D = load_scenario(root, sc)
        res = dict(r=D['r'], reproduction=D['summary']['reproduction'], cells={}, triple={}, T={}, secondary={}, generators={})
        md.append(f"\n## {sc}\n")
        md.append("5 折均值 sRMSE（所有臂）\n\n| 臂 | sRMSE | 折0 | 折1 | 折2 | 折3 | 折4 | Spearman | centered MSE |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for a, v in D['scores'].items():
            md.append(f"| {a} | {v.mean():.6f} | " + ' | '.join(f'{x:.4f}' for x in v) + f" | {np.mean(D['spearman'][a]):.4f} | {np.mean(D['cmse'][a]):.4f} |")
        md.append(f"\n矩阵（行 = 部署层 X，列 = 借用层 Y）。每格：headroom = gain(X+Y 对 X)；recovered = gain(X+vY 对 X)；shuf = gain(X+vY_shuf 对 X)；virt_vs_shuf = gain(X+vY 对 X+vY_shuf)；share = recovered/headroom。\n")
        md.append("| X \\ Y | 读出 | " + ' | '.join(LAYERS) + " |\n|---|---|" + '---|' * len(LAYERS))
        for x in LAYERS:
            cells = {}
            for y in LAYERS:
                if y == x:
                    continue
                hd = gain(D, meas_name(x, y), x)
                rec = gain(D, f'{x}+v{y}', x)
                sh = gain(D, f'{x}+v{y}_shuf', x)
                vs = gain(D, f'{x}+v{y}', f'{x}+v{y}_shuf')
                share = rec['gain'] / hd['gain'] if hd['gain'] > ROOM_MIN else None
                r2 = [D['bank'][k]['generators'][f'{x}->{y}']['oof_r2_mean'] for k in D['bank']]
                cell = dict(headroom=hd, recovered=rec, shuffle=sh, virt_vs_shuf=vs, share=share,
                            room_label=room_label(hd), recovery_label=recovery_label(rec, vs),
                            share_label=share_label(share) if room_label(hd) != '无空间' else '无空间',
                            generator_oof_r2_mean=float(np.mean(r2)))
                cells[y] = cell
                for tag, g in (('headroom', hd), ('recovered', rec), ('shuffle', sh), ('virt_vs_shuf', vs)):
                    comps.append(dict(scenario=sc, kind=tag, deployed=x, borrowed=y, **{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in g.items()}))
            res['cells'][x] = cells

            def col(y, key):
                if y == x:
                    return '—'
                c = cells[y]
                if key == 'share':
                    return '无空间' if c['share'] is None else f"{c['share'] * 100:.0f}% ({c['share_label']})"
                if key == 'labels':
                    return f"{c['room_label']} / {c['recovery_label']}"
                if key == 'r2':
                    return f"{c['generator_oof_r2_mean']:.3f}"
                return fmt(c[key])
            for key, name in (('headroom', 'headroom（实测 Y）'), ('recovered', 'recovered（虚拟 Y）'), ('shuffle', 'shuf 对照'),
                              ('virt_vs_shuf', '虚拟 对 置乱'), ('share', 'share'), ('r2', '生成器 OOF R²'), ('labels', '判定')):
                md.append(f"| {x} | {name} | " + ' | '.join(col(y, key) for y in LAYERS) + ' |')
            tri = gain(D, 'M+RNA+P', x)
            res['triple'][x] = tri
            comps.append(dict(scenario=sc, kind='triple', deployed=x, borrowed='others', **{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in tri.items()}))
        md.append("\n三层实测 M+RNA+P 对各单层：" + '；'.join(f"对 {x}: {fmt(res['triple'][x])}" for x in LAYERS))
        if sc == 'new_cell':
            md.append("\n组织标签 T 行（免费上下文层）：\n\n| X | X+T 对 X |\n|---|---|")
            for x in ('M', 'RNA', 'P', 'M+RNA+P'):
                g = gain(D, f'{x}+T', x); res['T'][x] = g
                comps.append(dict(scenario=sc, kind='T', deployed=x, borrowed='T', **{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in g.items()}))
                md.append(f"| {x} | {fmt(g)} |")
            for x in LAYERS:
                g = gain(D, x, 'T'); res['T'][f'{x}_vs_T'] = g
            md.append("单层对 T：" + '；'.join(f"{x} 对 T: {fmt(res['T'][f'{x}_vs_T'])}" for x in LAYERS))
        sec = {}
        for x in ('M', 'RNA'):
            hv = gain(D, f'{x}+Pval', x)
            rec = res['cells'][x]['P']['recovered']
            sec[x] = dict(headroom_val=hv, share_val=(rec['gain'] / hv['gain'] if hv['gain'] > ROOM_MIN else None))
            comps.append(dict(scenario=sc, kind='headroom_val', deployed=x, borrowed='Pval', **{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in hv.items()}))
        sec['Pval_vs_P'] = gain(D, 'Pval', 'P')
        sec['P_vs_Pval'] = gain(D, 'P', 'Pval')
        res['secondary'] = sec
        md.append("\n次要（掩码分解）：" + '；'.join(
            f"{x}: headroom_val {fmt(sec[x]['headroom_val'])}, share_val " + ('无空间' if sec[x]['share_val'] is None else f"{sec[x]['share_val'] * 100:.0f}%") for x in ('M', 'RNA'))
            + f"；P 对 Pval: {fmt(sec['P_vs_Pval'])}")
        for pair in [f'{x}->{y}' for x, y in PAIRS] + [f'{x}->{y}_shuf' for x, y in PAIRS]:
            vals = [D['bank'][k]['generators'][pair] for k in D['bank']]
            res['generators'][pair] = dict(oof_r2_mean=float(np.mean([v['oof_r2_mean'] for v in vals])),
                                           oof_r2_median=float(np.mean([v['oof_r2_median'] for v in vals])),
                                           lam_counts={str(l): int(sum(v['lam'] == l for v in vals)) for l in (.1, 1., 10., 100.)})
        md.append("\n生成器 OOF R²（20 个 bank 均值；目标列均值 / 中位数）：" + '；'.join(
            f"{p}: {v['oof_r2_mean']:.3f} / {v['oof_r2_median']:.3f}" for p, v in res['generators'].items() if not p.endswith('_shuf')))
        md.append("置乱对照 OOF R²：" + '；'.join(f"{p}: {v['oof_r2_mean']:.3f}" for p, v in res['generators'].items() if p.endswith('_shuf')))
        pc = [D['bank'][k]['P']['candidates'] for k in D['bank']]
        res['p_candidates'] = dict(min=int(min(pc)), max=int(max(pc)))
        res['null_drug'] = float(D['scores']['null_drug'].mean())
        out[sc] = res
        with open(f'{root}/matrix_{sc}.csv', 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['deployed', 'borrowed', 'headroom_gain', 'headroom_wins', 'headroom_p', 'recovered_gain', 'recovered_wins', 'recovered_p',
                        'shuffle_gain', 'shuffle_wins', 'virt_vs_shuf_gain', 'virt_vs_shuf_wins', 'virt_vs_shuf_p', 'share', 'room_label', 'recovery_label', 'share_label', 'generator_oof_r2'])
            for x in LAYERS:
                for y in LAYERS:
                    if y == x:
                        continue
                    c = res['cells'][x][y]
                    w.writerow([x, y, c['headroom']['gain'], c['headroom']['wins'], c['headroom']['p'], c['recovered']['gain'], c['recovered']['wins'], c['recovered']['p'],
                                c['shuffle']['gain'], c['shuffle']['wins'], c['virt_vs_shuf']['gain'], c['virt_vs_shuf']['wins'], c['virt_vs_shuf']['p'],
                                '' if c['share'] is None else c['share'], c['room_label'], c['recovery_label'], c['share_label'], c['generator_oof_r2_mean']])
    json.dump(out, open(f'{root}/analysis.json', 'w'), indent=1, default=float)
    with open(f'{root}/comparisons.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(comps[0])); w.writeheader(); w.writerows(comps)
    open(f'{root}/matrix_tables.md', 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    analyze(sys.argv[1])

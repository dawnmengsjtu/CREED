"""Pre-registered analysis for the generator-size learning curve (PLAN.md section 6).

usage: python analyze_curve.py OUTPUT_ROOT   (contains new_cell/ and tissue/ written by gencurve.py)
Writes OUTPUT_ROOT/{fold_metrics.csv, comparisons.csv, summary.json, curve_tables.md}.
Reads source-inputs only for the tissue labels of held cells (bootstrap groups).
"""
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import t as tdist

ROOT = os.environ.get('CREED_ROOT', '/public/home/mengxl/creed-research-runs')
SOURCE = ROOT + '/gdsc-metabolite-full-protein-20260908-0001/output/result/source-inputs.npz'
SCEN = ('new_cell', 'tissue')
LAYERS = {'RNA': ('f50', 'f75', 'f100', 'extra'), 'P': ('f50', 'f75', 'f100')}
NBOOT, SEED, ROOM_MIN = 2000, 20261009, 0.3


def load_scenario(root, sc):
    folds = []
    for f in range(5):
        z = np.load(f'{root}/{sc}/fold-{f}_predictions.npz', allow_pickle=False)
        folds.append({k: z[k] for k in z.files})
    rows = list(csv.DictReader(open(f'{root}/{sc}/fold_metrics.csv')))
    scores, spear, cmse, ratio = {}, {}, {}, []
    for r in rows:
        scores.setdefault(r['arm'], [None] * 5)[int(r['fold'])] = float(r['score'])
        spear.setdefault(r['arm'], [None] * 5)[int(r['fold'])] = float(r['spearman'])
        cmse.setdefault(r['arm'], [None] * 5)[int(r['fold'])] = float(r['centered_mse'])
        if r['arm'] == 'M':
            ratio.append(int(r['n_held']) / int(r['n_fit']))
    return dict(folds=folds, rows=rows, scores={a: np.array(v) for a, v in scores.items()}, spearman=spear, cmse=cmse,
                r=float(np.mean(ratio)), summary=json.load(open(f'{root}/{sc}/summary.json')),
                bank=json.load(open(f'{root}/{sc}/bank_info.json')))


def fold_score(d, arm):
    e = ((d['pred__' + arm] - d['truth']) / d['train_sd']) ** 2; obs = np.isfinite(d['truth'])
    mse = np.where(obs, e, 0).sum(0) / np.maximum(obs.sum(0), 1)
    return float(np.sqrt(mse[d['evaluated'].astype(bool)].mean()))


def nb(d, r):
    """Nadeau-Bengio corrected paired t on per-fold relative differences d (%), df = k-1."""
    d = np.asarray(d, float); k = len(d); m = float(d.mean()); v = float(d.var(ddof=1)) if k > 1 else 0.
    se = float(np.sqrt((1 / k + r) * v))
    if se == 0:
        return dict(t=None, p=(0. if m != 0 else 1.), ci=[m, m], se=0.)
    t = m / se; p = float(2 * tdist.sf(abs(t), k - 1)); q = float(tdist.ppf(.975, k - 1))
    return dict(t=float(t), p=p, ci=[m - q * se, m + q * se], se=se)


def gain(D, new, base, keep=None):
    a, b = D['scores'][new], D['scores'][base]
    if keep is not None:
        a, b = a[keep], b[keep]
    d = (b - a) / b * 100
    out = dict(new=new, base=base, gain=float(d.mean()), ratio_of_means=float((b.mean() - a.mean()) / b.mean() * 100),
               per_fold=[float(x) for x in d], wins=int((a < b).sum()), k=int(len(d)), new_mean=float(a.mean()), base_mean=float(b.mean()))
    out.update(nb(d, D['r']))
    if keep is None:
        out['lofo'] = [float(np.delete(d, f).mean()) for f in range(len(d))]
    return out


def boot_weights(D, S):
    """Common random numbers: multinomial weights over the union of drugs and over tissue groups of held cells (per fold)."""
    rng = np.random.default_rng(SEED)
    folds = D['folds']
    union = sorted(set().union(*[set(d['drug_ids'].tolist()) for d in folds]))
    uidx = {g: i for i, g in enumerate(union)}
    dw = rng.multinomial(len(union), np.full(len(union), 1 / len(union)), size=NBOOT).astype(float)
    cw = []
    for d in folds:
        t = S['tissues'][d['held']]; ut, inv = np.unique(t, return_inverse=True)
        cnt = rng.multinomial(len(ut), np.full(len(ut), 1 / len(ut)), size=NBOOT).astype(float)
        cw.append(cnt[:, inv])
    return dw, cw, uidx


def boot_scores(D, W, arm):
    dw, cw, uidx = W
    out = np.zeros((NBOOT, 5))
    for f, d in enumerate(D['folds']):
        e = ((d['pred__' + arm] - d['truth']) / d['train_sd']) ** 2; obs = np.isfinite(e)
        num = cw[f] @ np.where(obs, e, 0); den = cw[f] @ obs.astype(float)
        mse = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        v = dw[:, [uidx[g] for g in d['drug_ids']]] * d['evaluated'][None] * (den > 0)
        out[:, f] = np.sqrt((v * mse).sum(1) / v.sum(1))
    return out


def add_boot(g, D, W, cache):
    for a in (g['new'], g['base']):
        if a not in cache:
            cache[a] = boot_scores(D, W, a)
    br = ((cache[g['base']] - cache[g['new']]) / cache[g['base']] * 100).mean(1)
    g['boot'] = dict(reps=NBOOT, q025=float(np.quantile(br, .025)), q975=float(np.quantile(br, .975)), frac_positive=float((br > 0).mean()))
    return g


def fmt(g):
    return f"{g['gain']:+.3f}% ({g['wins']}/{g['k']}, p={g['p']:.3f})"


def analyze(root):
    S = np.load(SOURCE, allow_pickle=False); S = {'tissues': S['tissues']}
    out, comps, md, fm_rows = {}, [], [], []
    verdict_layer = {}
    for sc in SCEN:
        D = load_scenario(root, sc)
        W = boot_weights(D, S); cache = {}
        xcheck = max(abs(fold_score(d, a) - D['scores'][a][f]) for f, d in enumerate(D['folds']) for a in D['scores'])
        res = dict(r=D['r'], reproduction=D['summary']['reproduction'], metric_crosscheck_max_abs=float(xcheck), arms={}, layers={})
        for r in D['rows']:
            fm_rows.append(dict(scenario=sc, **r))
        md.append(f"\n## {sc}\n")
        md.append("5 折均值 sRMSE（所有臂；越低越好）\n\n| 臂 | sRMSE | 折0 | 折1 | 折2 | 折3 | 折4 | Spearman | centered MSE | 外折生成器 n（5 折） |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for a, v in D['scores'].items():
            ng = [D['bank'][f'fold-{f}/outer']['generators'][a]['n_gen'] for f in range(5)] if a in D['bank']['fold-0/outer']['generators'] else None
            res['arms'][a] = dict(score=float(v.mean()), fold_scores=v.tolist(), spearman=float(np.mean(D['spearman'][a])),
                                  centered_mse=float(np.mean(D['cmse'][a])), n_gen_outer=ng)
            md.append(f"| {a} | {v.mean():.6f} | " + ' | '.join(f'{x:.4f}' for x in v) + f" | {np.mean(D['spearman'][a]):.4f} | {np.mean(D['cmse'][a]):.4f} | "
                      + ('/'.join(map(str, ng)) if ng else '—') + ' |')

        def rec(cid, layer, g, role, share=None):
            comps.append(dict(scenario=sc, layer=layer, id=cid, role=role, new=g['new'], base=g['base'], new_mean=g['new_mean'], base_mean=g['base_mean'],
                              gain_pct=g['gain'], ratio_of_means_pct=g['ratio_of_means'], folds_won=g['wins'], nb_t=g['t'], nb_p=g['p'],
                              ci_low_pct=g['ci'][0], ci_high_pct=g['ci'][1], per_fold_pct=' '.join(f'{x:.4f}' for x in g['per_fold']),
                              lofo_min_pct=(min(g['lofo']) if 'lofo' in g else ''), lofo_max_pct=(max(g['lofo']) if 'lofo' in g else ''),
                              boot_q025_pct=(g['boot']['q025'] if 'boot' in g else ''), boot_q975_pct=(g['boot']['q975'] if 'boot' in g else ''),
                              boot_frac_positive=(g['boot']['frac_positive'] if 'boot' in g else ''), share=('' if share is None else share)))
            return g

        for Y, sizes in LAYERS.items():
            hd = rec('headroom', Y, gain(D, f'M+{Y}', 'M'), 'reference')
            room = hd['gain'] > ROOM_MIN
            L = dict(headroom=hd, sizes={}, r2_vs_share=[])
            md.append(f"\n### 借用层 {Y}：headroom = gain(M+{Y} 对 M) = {fmt(hd)}；去一折 [{min(hd['lofo']):.3f}, {max(hd['lofo']):.3f}]%\n")
            md.append("| 生成器训练集 | 外折 n_gen（5 折） | 内折 n_gen 范围 | gain(M+vY 对 M) | 胜折 | NB p | 95% CI | 去一折 min/max | share（均值定义） | share（均值比） | 逐折 share | 生成器 OOF R²（外折均值 / 20 bank 均值） | λ 计数 |\n|---|---|---|---:|---:|---:|---|---|---:|---:|---|---|---|")
            for s in sizes:
                arm = f'M+v{Y}_{s}'
                g = rec(f'recovered_{s}', Y, gain(D, arm, 'M'), 'recovered')
                share = (g['gain'] / hd['gain']) if room else None
                share_rm = (g['ratio_of_means'] / hd['ratio_of_means']) if room else None
                pf_share = [g['per_fold'][f] / hd['per_fold'][f] if hd['per_fold'][f] > 0 else float('nan') for f in range(5)]
                comps[-1]['share'] = '' if share is None else share
                ng_outer = [D['bank'][f'fold-{f}/outer']['generators'][arm]['n_gen'] for f in range(5)]
                ng_inner = [D['bank'][f'fold-{f}/{b}']['generators'][arm]['n_gen'] for f in range(5) for b in ('inner-0', 'inner-1', 'inner-2')]
                r2_outer = float(np.mean([D['bank'][f'fold-{f}/outer']['generators'][arm]['oof_r2_mean'] for f in range(5)]))
                r2_all = float(np.mean([D['bank'][k]['generators'][arm]['oof_r2_mean'] for k in D['bank']]))
                lam_counts = {str(l): int(sum(D['bank'][k]['generators'][arm]['lam'] == l for k in D['bank'])) for l in (.1, 1., 10., 100.)}
                L['sizes'][s] = dict(recovered=g, share=share, share_ratio_of_means=share_rm, per_fold_share=pf_share, n_gen_outer=ng_outer,
                                     n_gen_inner_min=int(min(ng_inner)), n_gen_inner_max=int(max(ng_inner)), oof_r2_outer_mean=r2_outer,
                                     oof_r2_all_banks_mean=r2_all, lam_counts=lam_counts)
                L['r2_vs_share'].append(dict(size=s, n_gen_outer_mean=float(np.mean(ng_outer)), oof_r2_outer_mean=r2_outer, share=share, gain=g['gain']))
                md.append(f"| {s} | {'/'.join(map(str, ng_outer))} | {min(ng_inner)}–{max(ng_inner)} | {g['gain']:+.3f}% | {g['wins']}/5 | {g['p']:.3f} | [{g['ci'][0]:.2f}, {g['ci'][1]:.2f}] | "
                          f"{min(g['lofo']):.3f} / {max(g['lofo']):.3f} | " + ('无空间' if share is None else f'{share * 100:.0f}%') + ' | '
                          + ('无空间' if share_rm is None else f'{share_rm * 100:.0f}%') + ' | ' + ' '.join('nan' if np.isnan(x) else f'{x * 100:.0f}%' for x in pf_share)
                          + f" | {r2_outer:.3f} / {r2_all:.3f} | " + ' '.join(f'{k}:{v}' for k, v in lam_counts.items()) + ' |')
            # L1: within-421 curve
            s1 = rec('L1_step_f75_vs_f50', Y, add_boot(gain(D, f'M+v{Y}_f75', f'M+v{Y}_f50'), D, W, cache), 'L1')
            s2 = rec('L1_step_f100_vs_f75', Y, add_boot(gain(D, f'M+v{Y}_f100', f'M+v{Y}_f75'), D, W, cache), 'L1')
            rising = bool(s1['wins'] >= 4 and s2['wins'] >= 4)
            lofo = []
            for f in range(5):
                keep = [i for i in range(5) if i != f]
                a1, a2 = gain(D, f'M+v{Y}_f75', f'M+v{Y}_f50', keep), gain(D, f'M+v{Y}_f100', f'M+v{Y}_f75', keep)
                lofo.append(dict(drop=f, step1_wins=a1['wins'], step2_wins=a2['wins'], step1_gain=a1['gain'], step2_gain=a2['gain'],
                                 rising=bool(a1['wins'] >= 3 and a2['wins'] >= 3)))
            # slopes: least-squares line through (n_gen_outer, per-fold share) per fold; also per-fold gain vs n
            def slope(xs, ys):
                xs, ys = np.asarray(xs, float), np.asarray(ys, float)
                ok = np.isfinite(ys)
                return float(np.polyfit(xs[ok], ys[ok], 1)[0]) if ok.sum() >= 2 else float('nan')
            sl = {}
            for tag, ss in (('3pt', ('f50', 'f75', 'f100')), ('4pt', sizes)):
                if tag == '4pt' and len(sizes) < 4:
                    continue
                per_share = [slope([L['sizes'][s]['n_gen_outer'][f] for s in ss], [L['sizes'][s]['per_fold_share'][f] * 100 for s in ss]) * 100 for f in range(5)]
                per_gain = [slope([L['sizes'][s]['n_gen_outer'][f] for s in ss], [L['sizes'][s]['recovered']['per_fold'][f] for s in ss]) * 100 for f in range(5)]
                mean_share = slope([np.mean(L['sizes'][s]['n_gen_outer']) for s in ss], [(L['sizes'][s]['share'] or np.nan) * 100 for s in ss]) * 100
                sl[tag] = dict(sizes=list(ss), share_slope_per_100_cells_per_fold=per_share,
                               share_slope_mean=float(np.nanmean(per_share)), share_slope_sd=float(np.nanstd(per_share, ddof=1)),
                               gain_slope_per_100_cells_per_fold=per_gain, gain_slope_mean=float(np.nanmean(per_gain)),
                               gain_slope_sd=float(np.nanstd(per_gain, ddof=1)), share_slope_on_fold_means=mean_share)
            L['L1'] = dict(step_f75_vs_f50=s1, step_f100_vs_f75=s2, rising=rising, label=('still rising at n=fit' if rising else 'flat'),
                           lofo=lofo, lofo_rising_count=int(sum(x['rising'] for x in lofo)), slopes=sl)
            md.append(f"\nL1（421 内曲线）：f75 对 f50 {fmt(s1)}，bootstrap 95% [{s1['boot']['q025']:.3f}, {s1['boot']['q975']:.3f}]%；"
                      f"f100 对 f75 {fmt(s2)}，bootstrap 95% [{s2['boot']['q025']:.3f}, {s2['boot']['q975']:.3f}]% → **{L['L1']['label']}**；"
                      f"去一折后仍 rising 的次数 {L['L1']['lofo_rising_count']}/5。")
            for tag, v in sl.items():
                md.append(f"share 对 n 斜率（{tag}，每 100 个生成器训练细胞的 share 百分点）：逐折 " + ' '.join(f'{x:+.1f}' for x in v['share_slope_per_100_cells_per_fold'])
                          + f"，均值 {v['share_slope_mean']:+.1f} ± {v['share_slope_sd']:.1f}；按 5 折均值 share 的斜率 {v['share_slope_on_fold_means']:+.1f}；"
                          f"gain 对 n 斜率（%/100 细胞）均值 {v['gain_slope_mean']:+.3f} ± {v['gain_slope_sd']:.3f}。")
            # L2: extra lines (RNA only)
            if 'extra' in sizes:
                e1 = rec('L2_extra_vs_f100', Y, add_boot(gain(D, 'M+vRNA_extra', 'M+vRNA_f100'), D, W, cache), 'L2')
                e2 = rec('L2_extra_vs_extra_shuf', Y, add_boot(gain(D, 'M+vRNA_extra', 'M+vRNA_extra_shuf'), D, W, cache), 'L2_control')
                e3 = rec('L2_extra_vs_extra_shufall', Y, add_boot(gain(D, 'M+vRNA_extra', 'M+vRNA_extra_shufall'), D, W, cache), 'L2_control')
                e4 = rec('L2_extra_shuf_vs_f100', Y, gain(D, 'M+vRNA_extra_shuf', 'M+vRNA_f100'), 'L2_control')
                e5 = rec('L2_extra_shufall_vs_M', Y, gain(D, 'M+vRNA_extra_shufall', 'M'), 'L2_control')
                e6 = rec('L2_extra_shuf_vs_M', Y, gain(D, 'M+vRNA_extra_shuf', 'M'), 'L2_control')
                helps = bool(e1['gain'] > 0 and e1['wins'] >= 4 and e1['p'] < .10)
                lofo2 = []
                for f in range(5):
                    keep = [i for i in range(5) if i != f]
                    a = gain(D, 'M+vRNA_extra', 'M+vRNA_f100', keep)
                    lofo2.append(dict(drop=f, gain=a['gain'], wins=a['wins'], p=a['p'], passes=bool(a['gain'] > 0 and a['wins'] >= 3 and a['p'] < .10)))
                ctrl = bool(e2['gain'] > 0 and e2['wins'] >= 4)
                L['L2'] = dict(extra_vs_f100=e1, extra_vs_extra_shuf=e2, extra_vs_extra_shufall=e3, extra_shuf_vs_f100=e4, extra_shufall_vs_M=e5, extra_shuf_vs_M=e6,
                               helps=helps, beats_paired_shuffle=ctrl, label=('extra lines help' if helps else 'no detectable help from extra lines'),
                               lofo=lofo2, lofo_pass_count=int(sum(x['passes'] for x in lofo2)))
                md.append(f"\nL2（额外 M+RNA 线）：extra 对 f100 {fmt(e1)}，95% CI [{e1['ci'][0]:.3f}, {e1['ci'][1]:.3f}]%，去一折 [{min(e1['lofo']):.3f}, {max(e1['lofo']):.3f}]%，"
                          f"成组 bootstrap 95% [{e1['boot']['q025']:.3f}, {e1['boot']['q975']:.3f}]%（正的比例 {e1['boot']['frac_positive']:.3f}）→ **{L['L2']['label']}**；"
                          f"去一折后仍通过 {L['L2']['lofo_pass_count']}/5。extra 对 extra_shuf（仅额外行配对置乱）{fmt(e2)}，bootstrap 95% [{e2['boot']['q025']:.3f}, {e2['boot']['q975']:.3f}]%；"
                          f"extra 对 extra_shufall（全部行配对置乱）{fmt(e3)}；extra_shuf 对 f100 {fmt(e4)}；extra_shufall 对 M {fmt(e5)}。")
            res['layers'][Y] = L
            verdict_layer.setdefault(Y, []).append(rising)
        out[sc] = res
    verdicts = {Y: ('data-sensitive' if any(v) else 'saturated within the available range') for Y, v in verdict_layer.items()}
    verdicts['L2'] = {sc: out[sc]['layers']['RNA']['L2']['label'] for sc in SCEN}
    out['verdicts'] = verdicts
    md.insert(0, "# 生成器规模学习曲线：预登记读出\n\n判定：" + '；'.join(f"{Y}: **{v}**" for Y, v in verdicts.items() if Y != 'L2')
              + '；L2: ' + '；'.join(f"{sc}: **{v}**" for sc, v in verdicts['L2'].items()))
    json.dump(out, open(f'{root}/summary.json', 'w'), indent=1, default=float)
    with open(f'{root}/comparisons.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(comps[0])); w.writeheader(); w.writerows(comps)
    with open(f'{root}/fold_metrics.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(fm_rows[0])); w.writeheader(); w.writerows(fm_rows)
    open(f'{root}/curve_tables.md', 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    analyze(sys.argv[1])

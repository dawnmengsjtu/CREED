"""Stage A step 2 analysis (PLAN.md sections 5-6, GO-A5). usage: python analyze_crispr.py OUT_ROOT"""
import json
import sys

import numpy as np

OUT = sys.argv[1]
SCEN = ('new_cell', 'tissue')
YS = ('RNA', 'P')


def rel(base, arm):
    return (base - arm) / base * 100


report, md = {}, ['# Stage A step 2 (CRISPR) result (auto-generated; numbers only)', '']
for scen in SCEN:
    R = [json.load(open(f'{OUT}/{scen}/fold-{f}/fold_result.json')) for f in range(5)]
    S = {}
    for res in R:
        for row in res['rows']:
            S.setdefault(row['arm'], []).append(row['score'])
    S = {a: np.array(v) for a, v in S.items()}
    rsel = {y: [r['r_selected'][y] for r in R] for y in YS}
    out = dict(reproduction_passed=all(r['reproduction_passed'] for r in R),
               reproduction_max=max(max(r['reproduction'].values()) for r in R), r_selected=rsel, Y={})
    md += [f'## {scen}', '', f"reproduction passed: {out['reproduction_passed']} (max {out['reproduction_max']:.2e}); M = {S['M'].mean():.6f}", '']
    for y in YS:
        star = lambda a: np.array([S[f'{a}_{y}_r{rsel[y][f]}'][f] for f in range(5)])
        starmean = lambda a: np.array([np.mean([S[f'{a}_{y}_r{rsel[y][f]}_s{s}'][f] for s in range(5)]) for f in range(5)])
        arms = {'M': S['M'], 'H086': S[f'H086_{y}'], 'A1*': star('A1'), 'A2': S[f'A2_{y}'], 'A3*': star('A3'),
                'A4*mean': starmean('A4'), 'A5*mean': starmean('A5'), 'A6*': star('A6'), 'A7*': star('A7'), 'A9b*': star('A9b'),
                'T512': S[f'T512_{y}'], 'CREED_M+Y': S[f'CREED_M+{y}']}
        room = rel(arms['M'], arms['CREED_M+Y'])
        tab = {}
        for a, v in arms.items():
            d_h = rel(arms['H086'], v); d_m = rel(arms['M'], v)
            tab[a] = dict(score=float(v.mean()), d_vs_H086=float(d_h.mean()), wins_vs_H086=int((d_h > 0).sum()),
                          d_vs_M=float(d_m.mean()), share=float(d_m.mean() / room.mean()), d_vs_H086_folds=d_h.tolist())
        out['Y'][y] = dict(room=float(room.mean()), arms=tab, GO_A5=bool(tab['A1*']['wins_vs_H086'] >= 4))
        md += [f'### {y} (room {room.mean():.3f}%, r* {rsel[y]}; GO-A5 {out["Y"][y]["GO_A5"]})', '',
               '| arm | sRMSE | Δ vs H086 % | wins | Δ vs M % | share |', '|---|---|---|---|---|---|']
        for a, v in tab.items():
            md.append(f"| {a} | {v['score']:.6f} | {v['d_vs_H086']:+.3f} | {v['wins_vs_H086']}/5 | {v['d_vs_M']:+.3f} | {v['share']:.3f} |")
        md.append('')
    report[scen] = out
json.dump(report, open(f'{OUT}/analysis_crispr.json', 'w'), indent=1)
open(f'{OUT}/RESULT_AUTO_CRISPR.md', 'w').write('\n'.join(md) + '\n')
print(json.dumps({s: {y: report[s]['Y'][y]['GO_A5'] for y in YS} for s in SCEN}))

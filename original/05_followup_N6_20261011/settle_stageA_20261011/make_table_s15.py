"""Table S15 of the CREED supplement, from the Stage A fold results (averaging only; no new computation).

build() returns (values, markdown_lines); running the file writes table_s15_values.json and table_s15.md.
tools/check_truth.py check (d) compares build() with the Table S15 body in supplemental_information.md.
Conventions: Δ = mean over folds of (S_H086 − S_arm)/S_H086 × 100 (negative = higher error than the standard virtual layer);
share = mean over folds of gain(arm vs M) ÷ mean over folds of gain(CREED_M+Y vs M)."""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
YS = ('RNA', 'P')
HEADER = ['| Endpoint | Scenario | Layer | Room | Standard share | A1 (r*): Δ, folds (share) | A2: Δ, folds | A3 (r*): Δ, folds | Random share | Permuted share | Shuffled share | Measured, r = 4 | Virtual, r = 4 |',
          '|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|']


def mean(v):
    return sum(v) / len(v)


def rel(b, a):
    return [(x - y) / x * 100 for x, y in zip(b, a)]


def load(root, scen):
    R = [json.load(open(os.path.join(HERE, root, scen, f'fold-{f}', 'fold_result.json'))) for f in range(5)]
    S = {}
    for r in R:
        for row in r['rows']:
            S.setdefault(row['arm'], []).append(row['score'])
    return R, S


def build():
    out = []
    for ep, root in (('Drug', 'output'), ('CRISPR', 'output_crispr')):
        for scen, lab in (('new_cell', 'New cell lines'), ('tissue', 'Held-out tissue')):
            R, S = load(root, scen)
            for y in YS:
                rs = [r['r_selected'][y] for r in R]
                st = lambda a: [S[f'{a}_{y}_r{rs[f]}'][f] for f in range(5)]
                sm = lambda a: [mean([S[f'{a}_{y}_r{rs[f]}_s{s}'][f] for s in range(5)]) for f in range(5)]
                M, H, room_arm = S['M'], S[f'H086_{y}'], S[f'CREED_M+{y}']
                room = mean(rel(M, room_arm))
                share = lambda v: mean(rel(M, v)) / room
                dh = lambda v: (mean(rel(H, v)), sum(d > 0 for d in rel(H, v)))
                out.append(dict(endpoint=ep, scenario=lab, layer=y, r_star=rs, room=room, h086_share=share(H),
                                A1=dh(st('A1')), A1_share=share(st('A1')), A2=dh(S[f'A2_{y}']), A3=dh(st('A3')),
                                A4_share=share(sm('A4')), A5_share=share(sm('A5')), A7_share=share(st('A7')),
                                A6star_share=share(st('A6')), A6r4_share=share(S[f'A6_{y}_r4']), A1r4_share=share(S[f'A1_{y}_r4'])))
    f2 = lambda x: f'{x:+.3f}%'
    pc = lambda x: f'{x * 100:.0f}%'
    lines = list(HEADER)
    for r in out:
        lines.append(f"| {r['endpoint']} | {r['scenario']} | {r['layer']} | {r['room']:+.3f}% | {pc(r['h086_share'])} | {f2(r['A1'][0])}, {r['A1'][1]}/5 ({pc(r['A1_share'])}) | {f2(r['A2'][0])}, {r['A2'][1]}/5 | {f2(r['A3'][0])}, {r['A3'][1]}/5 | {pc(r['A4_share'])} | {pc(r['A5_share'])} | {pc(r['A7_share'])} | {pc(r['A6r4_share'])} | {pc(r['A1r4_share'])} |")
    lines = [re.sub(r'(?<=[| (])-(?=\d)', '−', l) for l in lines]
    return out, lines


if __name__ == '__main__':
    values, lines = build()
    json.dump(values, open(os.path.join(HERE, 'table_s15_values.json'), 'w'), indent=1)
    open(os.path.join(HERE, 'table_s15.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

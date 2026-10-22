"""Plot table for the new Figure 2H: room retained at r = 4 by the function-weighted directions and by the controls.

Reads prospective_lowdim_20261022/server_output/analysis/analysis.json (PROSPECTIVE_LOWDIM_V1, run
prospective-lowdim-20261022-0001) and writes figure_2H/lowdim_controls.csv, one row per endpoint x scenario x layer x
comparison. Shares are copied verbatim (as %); nothing is recomputed. Standard library only.
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'server_output', 'analysis', 'analysis.json')
OUT = os.path.join(HERE, 'lowdim_controls.csv')
EP = [('drug', 'Drug response (GDSC2)', 'development, N6 arms re-run'),
      ('crispr', 'CRISPR gene effect', 'development, N6 arms re-run'),
      ('gdsc1', 'GDSC1 drug response', 'prospective, pre-registered')]
d = json.load(open(SRC))
rows = []
for ep, label, status in EP:
    for scen in ('new_cell', 'tissue'):
        for layer in ('RNA', 'P'):
            c = d['endpoints'][ep]['cells'][f'{scen}/{layer}']
            k = c['controls_r4']
            vals = dict(function_weighted=c['a6']['4'], pca=k['pca_share'], permuted=k['permuted_share'],
                        random=k['random_share'], virtual_same=c['a1']['4'])
            wins = dict(function_weighted='', pca=k['a6_beats_pca_folds'], permuted=k['a6_beats_permuted_folds'],
                        random=k['a6_beats_random_folds'], virtual_same='')
            for comp, v in vals.items():
                rows.append(dict(endpoint=ep, endpoint_label=label, status=status, scenario=scen, layer=layer,
                                 comparison=comp, share_pct=round(100 * v, 3), fw_beats_folds=wins[comp],
                                 room_pct=round(c['room'], 4),
                                 source=f'prospective-lowdim-20261022-0001/analysis/analysis.json endpoints.{ep}.cells.{scen}/{layer}'))
with open(OUT, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator='\n')
    w.writeheader(); w.writerows(rows)
print(len(rows), 'rows ->', OUT)

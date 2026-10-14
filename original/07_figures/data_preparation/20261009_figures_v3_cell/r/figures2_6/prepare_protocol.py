"""Data-bound counts for the five vector workflow schematics."""
from pathlib import Path
import numpy as np,json,csv
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'delivery_R_20261009/figures2_6'
z=np.load(ROOT/'data/raw/gdsc_metabolite_full_protein/source-inputs.npz',allow_pickle=False)
s=json.loads((ROOT/'data/raw/figure2_6_extra/cptac_shift_summary.json').read_text())
d=list(csv.DictReader(open(OUT/'figure3/plot_data/fold_metrics.csv')));counts=[int(r['drugs'])for r in d if r['arm']=='virtual_p']
flow=list(csv.DictReader(open(OUT/'figure6/plot_data/drug_flows.csv')))
gens=[np.load(ROOT/f'data/raw/gdsc_metabolite_full_protein/fold-{f}/outer/query-native.npz')['virtual_p'].shape[0] for f in range(5)];assert len(set(gens))==1
protocols={2:dict(cell_lines=len(z['sample_ids']),metabolites=z['m'].shape[1],generator_predictions=gens[0]),3:dict(cell_lines=len(z['sample_ids']),minimum_drugs=min(counts),maximum_drugs=max(counts),folds=len(counts)),4:dict(cell_lines=len(z['sample_ids']),folds=5,response_arms=10),5:dict(patient_count=s['patient_count'],cohorts={r['held_out_cohort'].split('_')[0].upper():r['query_patients']for r in s['records']if r['direction']=='rna_to_protein'},directions=len(s['directions'])),6:dict(full_drugs=len(z['drug_ids']),common_drugs=len(flow),folds=5)}
for f,p in protocols.items():(OUT/f'figure{f}/plot_data/protocol.json').write_text(json.dumps(p,indent=2))
print('Five data-bound schematic protocols prepared')

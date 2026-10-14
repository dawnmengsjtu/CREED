from pathlib import Path
import numpy as np,json
from run_extra import load,allowed_extra,extend_kernels
from run_recoverability import save_json

with np.load('/parent/source-inputs.npz') as f:s={k:f[k] for k in ('m','auc','sample_ids','metabolite_ids','drug_ids','tissues')}
e=load('/resources/m_gdsc_extra19.npz');rows=[]
for scenario in ('tissue','new_cell'):
 for fold in range(5):
  for bank in ('inner-0','inner-1','inner-2','outer'):
   old=Path('/previous')/(scenario+'_v2')/f'fold-{fold}'/bank
   z=load(old/'geometry.npz');ei=allowed_extra(s,e,z['fit'],scenario);assert len(ei)>0
   rows.append(dict(scenario=scenario,fold=fold,bank=bank,extra_ids=e['sample_ids'][ei].tolist()))
   if fold==0 and bank=='inner-0':
    inp=load(old/'graph_inputs.npz')
    gen=Path('/fmparent')/f'fold-{fold}'/bank if scenario=='tissue' else Path('/previous/new_cell')/f'fold-{fold}'/bank
    kernels,_=extend_kernels(s,e,z,inp,gen,ei)
    rows[-1]['min_eigenvalues']={b:float(np.linalg.eigvalsh(k).min()) for b,(k,h) in kernels.items()}
    assert min(rows[-1]['min_eigenvalues'].values())>-1e-8
    assert all(np.isfinite(k).all() and np.isfinite(h).all() for k,h in kernels.values())
save_json('/out/preflight_real.json',dict(status='PASS',rows=rows,query_outcomes_scored=False))
print(json.dumps(dict(status='PASS',banks=len(rows),extra_counts=sorted(set(len(r['extra_ids']) for r in rows)))))

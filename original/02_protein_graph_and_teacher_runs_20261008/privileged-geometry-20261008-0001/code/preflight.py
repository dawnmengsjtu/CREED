import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from run_geometry import load,run_bank,sha,SOURCE_SHA,effective,laplacian

parent=Path('/parent');fm=Path('/fmparent');out=Path('/out/preflight-bank')
s=load(parent/'source-inputs.npz');fs=load(fm/'source-inputs.npz')
assert sha(parent/'source-inputs.npz')==SOURCE_SHA
for key in ('m','auc','sample_ids','drug_ids','fold_ids','tissues'):np.testing.assert_array_equal(s[key],fs[key])
meta=json.loads((parent/'fold-0/inner-0/selection.json').read_text())
args=SimpleNamespace(parent=parent,fm_parent=fm,scenario='tissue')
grids=run_bank(s,fs,args,0,'inner-0',np.array(meta['fit']),np.array(meta['query']),np.array(meta['panel']),out)
for base in ('m_only','virtual_fm'):
    for graph in ('protein','mask','shuffle','metabolite'):
        np.testing.assert_array_equal(grids[base],grids[base+'__'+graph][:4])
record=json.loads((out/'record.json').read_text())
r=dict(status='PASS',all_four_alpha_old_baseline_parity=record['old_alpha_grid_parity_max'],
    zero_mu_grid_exact=True,source_identity_equal=True,complete_grid_arms=len(grids),
    seconds=record['seconds'])
Path('/out/preflight.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))

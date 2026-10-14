import json
from pathlib import Path
import numpy as np
from creed.science.masked_task_kernel import predict_masked
from run_recoverability import Experiment, load, kernel, predict_grid, cell_folds, ALPHAS, sha, SOURCE_SHA

p=Path('/parent');out=Path('/out');s=load(p/'source-inputs.npz')
assert sha(p/'source-inputs.npz')==SOURCE_SHA
ids=cell_folds(s['tissues'],20261008,outer=True)
np.testing.assert_array_equal(ids,json.loads(Path('/code/new_cell_fold_ids.json').read_text()))
rng=np.random.default_rng(47);x=rng.normal(size=(31,9));q=rng.normal(size=(7,9));y=rng.normal(size=(31,3));y[:4,0]=np.nan
k,h=kernel(x,q);grid=predict_grid(k,h,y)
for ai,a in enumerate(ALPHAS):np.testing.assert_allclose(grid[ai],predict_masked(k,h,y,a),rtol=1e-11,atol=1e-11)
meta=json.loads((p/'fold-0/inner-0/selection.json').read_text());fit=np.array(meta['fit']);query=np.array(meta['query']);panel=np.array(meta['panel'])
e=Experiment(s,p,'tissue',0,panel,out);old=e.raw(fit,query);e.raw_cache={};e.original={};new=e.raw(fit,query)
errors={name:float(np.max(np.abs(new[name]-old[name]))) for name in old}
for name in old:np.testing.assert_allclose(new[name],old[name],rtol=1e-11,atol=1e-11)
r={'status':'PASS','masked_solver_parity':True,'fixed_new_cell_folds_match_prior':True,'original_bank_four_alpha_reconstruction_max_abs':errors}
(out/'preflight.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))

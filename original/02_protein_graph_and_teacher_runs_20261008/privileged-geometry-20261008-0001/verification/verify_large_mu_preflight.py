"""Exercise the actual run_geometry functions, not a reimplementation, on tiny arrays.
AST loading avoids unrelated CREED imports; function source is unmodified.
"""
from pathlib import Path
import ast,json,hashlib
import numpy as np
from scipy.linalg import eigh,cho_factor,cho_solve
from scipy.spatial.distance import cdist
B=Path(__file__).resolve().parents[1];O=B/'verification';code=B/'code'
ns=dict(np=np,eigh=eigh,cho_factor=cho_factor,cho_solve=cho_solve,cdist=cdist,ALPHAS=(.1,1.,10.,100.))
for file,names in [('run_geometry.py',['laplacian','effective','extra_kernel']),('run_recoverability.py',['predict_grid','kernel','standardize','take','choose'])]:
 t=ast.parse((code/file).read_text());nodes=[n for n in t.body if isinstance(n,ast.FunctionDef)and n.name in names];assert len(nodes)==len(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(code/file),'exec'),ns)
rng=np.random.default_rng(8);checks=[]
def record(name,err,tol=1e-10):
 assert np.isfinite(err);checks.append(dict(check=name,error=float(err),tolerance=tol,passed=bool(err<=tol)))
n=33;d=8;nq=7
X=rng.normal(size=(n,d));Q=rng.normal(size=(nq,d));K=X@X.T;q=Q@X.T
P=rng.normal(size=(n,16));L,state=ns['laplacian'](P)
record('laplacian_symmetric',np.max(abs(L-L.T)),0);record('laplacian_rowsum',np.max(abs(L.sum(1))));record('laplacian_trace_n',abs(np.trace(L)-n));record('laplacian_psd',max(0,-np.linalg.eigvalsh(L).min()))
y=rng.normal(size=(n,4))+3;y[::4,0]=np.nan;y[::5,1]=np.nan;y[:24,2]=np.nan
oldgrid=ns['predict_grid'](K,q,y)
for mu in [0,.1,1,10,100,1000]:
 E,t=ns['effective'](K,q,L,mu);actual=ns['predict_grid'](E,t,y)
 if mu==0:record('mu0_every_prediction_exact',np.max(abs(actual-oldgrid)),0)
 R=np.eye(d)+mu*X.T@L@X
 record(f'mu{mu}/effective_vs_primal',np.max(abs(E-X@np.linalg.solve(R,X.T))))
 record(f'mu{mu}/query_vs_primal',np.max(abs(t-Q@np.linalg.solve(R,X.T))))
 for j in range(4):
  obs=np.flatnonzero(np.isfinite(y[:,j]));yy=y[obs,j];mean=yy.mean()
  for ai,alpha in enumerate(ns['ALPHAS']):
   expected=np.full(nq,mean)
   if len(obs)>=20:expected=Q@np.linalg.solve(X[obs].T@X[obs]+alpha*R,X[obs].T@(yy-mean))+mean
   record(f'mu{mu}/drug{j}/alpha{alpha}/masked_solve',np.max(abs(actual[ai,:,j]-expected)))
# Query-kernel averaging must commute with graph correction and prediction.
T=rng.normal(size=(n,12));Qs=rng.normal(size=(3,nq,12));Fk,Fh=ns['extra_kernel'](T,Qs)
mu=1;E,mean_h=ns['effective'](Fk,Fh,L,mu)
views=[]
for g in range(3):
 k,h=ns['extra_kernel'](T,Qs[g:g+1]);Eg,hg=ns['effective'](k,h,L,mu);views.append(ns['predict_grid'](Eg,hg,y))
actual=ns['predict_grid'](E,mean_h,y);record('mean_three_generator_cross_equals_mean_predictions',np.max(abs(actual-np.mean(views,axis=0))))
# Permuting the complete real-P rows gives the conjugated L for tie-free data.
perm=rng.permutation(n);Lp,_=ns['laplacian'](P[perm]);record('shuffle_graph_is_permuted_protein_graph',np.max(abs(Lp-L[np.ix_(perm,perm)])))
# Constant mask values still result in deterministic tie-broken edges. Record,
# do not call such a graph measured biological neighbourhood information.
Lc,_=ns['laplacian'](np.ones((n,4)));record('constant_mask_graph_finite',float(not np.isfinite(Lc).all()),0)
# Literal config order ensures exact ties favour mu0; no-validation fallback alpha1.
counts=np.ones((16,4));total=np.zeros_like(counts);sel=ns['choose'](total,counts,fallback=1);record('exact_tie_first_config_mu0',float(np.max(sel)),0)
counts[:,0]=0;sel=ns['choose'](total,counts,fallback=1);record('no_label_validation_fallback_mu0_alpha1',abs(int(sel[0])-1),0)
# Assert graph access in run_bank is fit-only at source level (runtime query-P
# perturbation verification should still be done on saved banks by main runner).
tree=ast.parse((code/'run_geometry.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run_bank');source=ast.unparse(fn)
assert "s['p'][np.ix_(a, pi)]" in source
assert "s['p'][q]" not in source and "s['p'][np.ix_(q" not in source
result={'status':'PASS'if all(c['passed']for c in checks)else'FAIL','n_checks':len(checks),'max_error':max(c['error']for c in checks),'checks':checks,'script_sha256':{f:hashlib.sha256((code/f).read_bytes()).hexdigest()for f in ['run_geometry.py','run_recoverability.py']},'limitations':['Small synthetic algebra checks only; no outcome performance claim.','Constant mask graph uses distance ties and therefore deterministic index-based neighbours.','Production inputs must contain finite filled P and at least two cells.'], 'static_run_bank_query_P_read_found':False}
(O/'large_mu_preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items()if k!='checks'},indent=2));assert result['status']=='PASS'

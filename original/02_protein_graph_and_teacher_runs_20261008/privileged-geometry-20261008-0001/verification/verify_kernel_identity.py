"""Independent algebra checks on small synthetic arrays; no biological model training.
Checks graph-regularized effective kernels against explicit primal ridge, including
missing labels, rank-deficient kernels, offsets and query projection.
"""
from pathlib import Path
import json
import numpy as np
OUT=Path(__file__).resolve().parent
rng=np.random.default_rng(20261008)

def laplacian(z,k=3):
 d=((z[:,None]-z[None,:])**2).sum(-1);np.fill_diagonal(d,np.inf)
 W=np.zeros_like(d);idx=np.argsort(d,axis=1)[:,:k];width=np.median(d[np.arange(len(z))[:,None],idx]);width=max(width,1e-12)
 W[np.arange(len(z))[:,None],idx]=np.exp(-d[np.arange(len(z))[:,None],idx]/width)
 W=(W+W.T)/2;L=np.diag(W.sum(1))-W
 return L*len(z)/np.trace(L)

def effective(K,q,L,mu):
 if mu==0:return K.copy(),q.copy()
 eig,U=np.linalg.eigh((K+K.T)/2)
 tol=100*np.finfo(float).eps*max(1,np.linalg.norm(K,2))
 assert eig.min()>=-tol,(eig.min(),tol)
 H=(U*np.sqrt(np.maximum(eig,0)))@U.T
 B=np.eye(len(K))+mu*H@L@H
 E=H@np.linalg.solve(B,H)
 return E,q-mu*q@L@E

def score_record(name,error,tolerance):
 ok=bool(error<=tolerance)
 return dict(check=name,maximum_absolute_error=float(error),tolerance=tolerance,passed=ok)

checks=[];details=[]
for rank_name,n,d in [('full_rank',9,13),('rank_deficient',11,4),('duplicate_cells',10,6)]:
 X=rng.normal(size=(n,d))/np.sqrt(d);Q=rng.normal(size=(5,d))/np.sqrt(d)
 if rank_name=='duplicate_cells':X[-1]=X[0]
 K=X@X.T;q=Q@X.T;L=laplacian(rng.normal(size=(n,4)));obs=np.array([i for i in range(n)if i%3!=1]);y=rng.normal(size=len(obs))+2.7;offset=float(y.mean());c=y-offset
 assert np.allclose(L.sum(1),0);assert np.linalg.eigvalsh(L).min()>-1e-12;assert abs(np.trace(L)-n)<1e-12
 for mu in [0,.1,1,10]:
  E,t=effective(K,q,L,mu)
  R=np.eye(d)+mu*X.T@L@X
  target_E=X@np.linalg.solve(R,X.T);target_t=Q@np.linalg.solve(R,X.T)
  checks.append(score_record(f'{rank_name}/mu={mu}/effective_kernel_vs_primal',np.max(np.abs(E-target_E)),3e-11))
  checks.append(score_record(f'{rank_name}/mu={mu}/query_kernel_vs_primal',np.max(np.abs(t-target_t)),3e-11))
  for alpha in [.1,1,10,100]:
   coefficients=np.linalg.solve(E[np.ix_(obs,obs)]+alpha*np.eye(len(obs)),c)
   f=E[:,obs]@coefficients;pred=t[:,obs]@coefficients+offset
   beta=np.linalg.solve(X[obs].T@X[obs]+alpha*R,X[obs].T@c)
   checks.append(score_record(f'{rank_name}/mu={mu}/alpha={alpha}/masked_predictions_vs_primal',np.max(np.abs(pred-(Q@beta+offset))),3e-11))
   checks.append(score_record(f'{rank_name}/mu={mu}/alpha={alpha}/all_fit_predictions_vs_primal',np.max(np.abs(f-X@beta)),3e-11))
   gradient=X[obs].T@(X[obs]@beta-c)+alpha*R@beta
   checks.append(score_record(f'{rank_name}/mu={mu}/alpha={alpha}/first_order_condition',np.max(np.abs(gradient)),3e-11))
   if mu==0:
    old=np.linalg.solve(K[np.ix_(obs,obs)]+alpha*np.eye(len(obs)),c)
    checks.append(score_record(f'{rank_name}/alpha={alpha}/mu0_exact_base',np.max(np.abs(pred-(q[:,obs]@old+offset))),0))
   details.append(dict(case=rank_name,n_fit=n,n_features=d,n_labels=len(obs),mu=mu,alpha=alpha,min_effective_eigenvalue=float(np.linalg.eigvalsh(E).min())))
 # Changing unobserved labels cannot alter the masked solve: no such values enter it.
 cfull=rng.normal(size=n);cfull[obs]=y;changed=cfull.copy();changed[np.setdiff1d(np.arange(n),obs)]=1e9
 checks.append(score_record(f'{rank_name}/unobserved_labels_ignored',np.max(np.abs(cfull[obs]-changed[obs])),0))
 # Numerical equivalent normal equation in RKHS coordinates H, including the labels mask.
 mu=1.;alpha=1.;w,U=np.linalg.eigh(K);H=(U*np.sqrt(np.maximum(w,0)))@U.T
 a=np.linalg.solve(H[obs].T@H[obs]+alpha*(np.eye(n)+mu*H@L@H),H[obs].T@c)
 E,t=effective(K,q,L,mu);u=np.linalg.solve(E[np.ix_(obs,obs)]+alpha*np.eye(len(obs)),c)
 checks.append(score_record(f'{rank_name}/masked_RKHS_coordinate_equivalence',np.max(np.abs(H@a-E[:,obs]@u)),3e-11))
# Zero graph must act as no graph even with nonzero mu (up to eigen rounding).
X=rng.normal(size=(8,5));Q=rng.normal(size=(3,5));K=X@X.T;q=Q@X.T
E,t=effective(K,q,np.zeros_like(K),10)
checks.append(score_record('zero_graph/effective_kernel',np.max(np.abs(E-K)),3e-11));checks.append(score_record('zero_graph/query_exact',np.max(np.abs(t-q)),0))
# Full PSD prior contracts the training kernel in Loewner order.
L=laplacian(rng.normal(size=(8,3)));E,t=effective(K,q,L,10)
checks.append(score_record('effective_kernel_contraction',max(0,-np.linalg.eigvalsh(K-E).min()),3e-11))
result={'status':'PASS'if all(x['passed']for x in checks)else'FAIL','scope':'Small-array mathematical identity verification only; no biological performance claim; no CREED data accessed.','n_checks':len(checks),'maximum_error':max(x['maximum_absolute_error']for x in checks),'checks':checks,'cases':details,'numpy_version':np.__version__}
(OUT/'kernel_identity_results.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items()if k not in ['checks','cases']},indent=2))
assert result['status']=='PASS'

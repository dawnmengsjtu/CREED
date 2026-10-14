"""Actual fixed extension vs a single explicit kernel-mean Gram construction.
Generator interface was separately replayed on all 120 saved real checkpoints.
Here stubs isolate new block algebra; no fitting, real extra labels or outcomes.
"""
import ast,argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import eigh,cho_factor,cho_solve
p=argparse.ArgumentParser();p.add_argument('--code',type=Path,required=True);p.add_argument('--old-code',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
ns=dict(np=np,cdist=cdist,eigh=eigh,cho_factor=cho_factor,cho_solve=cho_solve)
for file,names in [(a.code,('kernel_state','rbf','extend_blocks','extend_kernels','allowed_extra')),(a.old_code,('effective',))]:
    nodes=[n for n in ast.parse(file.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(nodes)==len(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(file),'exec'),ns)
rng=np.random.default_rng(20261008);checks=[]
def check(name,x,y,tol=1e-10):
    error=float(np.max(abs(np.asarray(x)-np.asarray(y)))) if np.asarray(x).size else 0.
    checks.append(dict(name=name,error=error,tolerance=tol,passed=bool(np.isfinite(error) and error<=tol)))
def kernel_all(points,oldfit):
    mean=oldfit.mean(0);sd=oldfit.std(0);sd=np.where(sd<1e-8,1.,sd);x=(points-mean)/sd;t=(oldfit-mean)/sd
    d=((t[:,None]-t[None])**2).sum(-1);u=d[np.triu_indices(len(t),1)];width=np.median(u[u>0]);dist=((x[:,None]-x[None])**2).sum(-1)
    return np.exp(-dist/(2*width))
n=25;nq=4;ne=3;dim=7;fdim=9
m=rng.normal(size=(n,dim));qm=rng.normal(size=(nq,dim));em=rng.normal(size=(ne,dim));ft=rng.normal(size=(n,fdim));fq=rng.normal(size=(3,nq,fdim))
weights=rng.normal(size=(3,dim,fdim));offsets=rng.normal(size=(3,fdim))
ns['load']=lambda path:dict(index=int(path.stem.split('-')[-1]))
ns['predict_generator']=lambda model,query:dict(virtual_fm=query@weights[model['index']]+offsets[model['index']])
fe=np.array([em@weights[g]+offsets[g] for g in range(3)])
points=np.concatenate([ft,fe.reshape(3*ne,fdim),fq.reshape(3*nq,fdim)])
G=kernel_all(points,ft);C=np.zeros((n+ne+nq,len(points)));C[:n,:n]=np.eye(n)
for j in range(ne):C[n+j,n+np.arange(3)*ne+j]=1/3
for j in range(nq):C[n+ne+j,n+3*ne+np.arange(3)*nq+j]=1/3
expectedF=C@G@C.T;expectedM=kernel_all(np.concatenate([m,em,qm]),m);expected=.5*(expectedM+expectedF)
z=dict(fit=np.arange(n),query=np.arange(n,n+nq),m_only_kernel=expectedM[:n,:n].copy(),m_only_cross=expectedM[n+ne:,:n].copy(),virtual_fm_kernel=expected[:n,:n].copy(),virtual_fm_cross=expected[n+ne:,:n].copy())
actual,got_fe=ns['extend_kernels'](dict(m=np.concatenate([m,qm])),dict(m=em),z,dict(fm_train=ft,fm_query_views=fq),Path('/unused'),np.arange(ne))
check('extra_views',got_fe,fe,0.)
for base,full in [('m_only',expectedM),('virtual_fm',expected)]:
    k,h=actual[base];check(base+'/explicit_kernel_mean_gram',k,full[:n+ne,:n+ne]);check(base+'/query_extra_9way',h,full[n+ne:,:n+ne]);check(base+'/old_kernel_exact',k[:n,:n],z[base+'_kernel'],0.);check(base+'/old_cross_exact',h[:,:n],z[base+'_cross'],0.)
    check(base+'/PSD_minimum',min(0.,float(np.linalg.eigvalsh(k).min())),0.)
    # Adding graph-isolated samples with no observed response leaves the old
    # masked solution unchanged; test with old/query blocks fixed exactly.
    w=rng.uniform(size=(n,n));w=(w+w.T)/2;np.fill_diagonal(w,0);l=np.diag(w.sum(1))-w;l*=n/np.trace(l);la=np.zeros_like(k);la[:n,:n]=l
    check(base+'/zero_extension_trace',np.trace(la),np.trace(l),0.)
    yy=rng.normal(size=n);yy-=yy.mean()
    for mu in (0.,1.,1000.):
        ko,ho=ns['effective'](z[base+'_kernel'],z[base+'_cross'],l,mu);ka,ha=ns['effective'](k,h,la,mu)
        pold=ho@np.linalg.solve(ko+np.eye(n),yy);paug=ha[:,:n]@np.linalg.solve(ka[:n,:n]+np.eye(n),yy)
        check(f'{base}/mu{mu}/unlabelled_extra_no_change',paug,pold,1e-8)
check('FM_mean_embedding_PSD',min(0.,float(np.linalg.eigvalsh(expectedF).min())),0.)
assert np.all(np.diag(expectedF)[n:n+ne]<1.)
# Hierarchical tissue exclusion: rows absent from current fit include both
# outer-held and inner-held tissues. New-cell mode has no tissue holdout and
# relies on upstream global identity exclusions before this function is called.
s=dict(m=np.zeros((6,2)),tissues=np.array(['Lung','Lung','Breast','Breast','Skin','Skin']))
e=dict(m=np.zeros((7,2)),auc=np.ones((7,3)),tissues=np.array(['Lung','Breast','Skin','Ovary','','Lung','Lung']))
e['m'][5,0]=np.nan;e['auc'][6]=np.nan
check('tissue_outer_excludes_skin_unknown_nonfinite',ns['allowed_extra'](s,e,np.array([0,1,2,3]),'tissue'),[0,1,3],0.)
check('tissue_inner_also_excludes_breast',ns['allowed_extra'](s,e,np.array([0,1]),'tissue'),[0,3],0.)
check('new_cell_same_tissue_allowed_after_global_identity_QA',ns['allowed_extra'](s,e,np.array([0,2,4]),'new_cell'),[0,1,2,3,4],0.)
result=dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',checks=len(checks),max_error=max(c['error'] for c in checks),script_sha256=hashlib.sha256(a.code.read_bytes()).hexdigest(),details=checks,limitations=['Synthetic new-block algebra with deterministic generator stubs; real generator inference separately replayed from all 120 saved files.','allowed_extra does not itself enforce global 608-linked-identity exclusion; the fixed upstream QA asset and hash must establish that condition.'])
a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='details'},indent=2));assert result['status']=='PASS'

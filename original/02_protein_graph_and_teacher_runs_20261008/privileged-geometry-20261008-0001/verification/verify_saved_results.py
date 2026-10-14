"""Independent verification of frozen privileged-geometry saved artifacts.
Reads only the authorized 421 development source and new/old saved banks. Does not
fit generators or refit full response models. Reconstructs graphs, selections,
final predictions and metrics; checks representative kernel solves per bank.
"""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from scipy.stats import spearmanr
P=argparse.ArgumentParser();P.add_argument('--run',type=Path,required=True);P.add_argument('--parent',type=Path,required=True);P.add_argument('--fm-parent',type=Path,required=True);P.add_argument('--report',type=Path,required=True);P.add_argument('--partial',action='store_true');a=P.parse_args()
start=time.perf_counter();checks=0;graphs_summary=[];banks=[];scores=[];selection_summary=[]
def load(p):
 with np.load(p,allow_pickle=False)as z:return {k:z[k]for k in z.files}
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def eq(x,y,atol=1e-10,rtol=1e-10):
 global checks
 np.testing.assert_allclose(x,y,atol=atol,rtol=rtol,equal_nan=True);checks+=1

def reconstruct_graph(x):
 assert len(x)>=2 and np.isfinite(x).all()
 sd=np.std(x,axis=0);sd[sd<1e-8]=1.;z=(x-np.mean(x,axis=0))/sd
 d=cdist(z,z,metric='sqeuclidean');off=d[np.triu_indices(len(x),1)];pos=off[off>0];width=np.median(pos)if len(pos)else 1.
 dd=d.copy();np.fill_diagonal(dd,np.inf);k=min(10,len(x)-1);order=np.argsort(dd,axis=1,kind='stable');nn=order[:,:k]
 w=np.zeros_like(d);rows=np.arange(len(x))[:,None];w[rows,nn]=np.exp(-d[rows,nn]/(2*width));w=np.maximum(w,w.T);l=np.diag(w.sum(1))-w;l*=len(x)/np.trace(l)
 # Equality at kth neighbour makes selection among equal-distance neighbours
 # depend on stable fit index. Also count duplicates independently.
 boundaries=np.sum(np.isclose(dd[np.arange(len(x)),order[:,k-1]],dd[np.arange(len(x)),order[:,k]],atol=1e-12,rtol=1e-12))if k<len(x)-1 else 0
 info=dict(all_features_constant=bool(np.all(np.std(x,0)<1e-8)),nonconstant_features=int(np.sum(np.std(x,0)>=1e-8)),zero_distance_pairs=int(np.sum(off==0)),nearest_boundary_tie_rows=int(boundaries),n=len(x),width=float(width))
 return l,info

def base_kernel(t,q):
 sd=t.std(0);sd=np.where(sd<1e-8,1.,sd);z=(t-t.mean(0))/sd;h=(q-t.mean(0))/sd;d=cdist(z,z,'sqeuclidean');v=d[np.triu_indices(len(t),1)];v=v[v>0];width=np.median(v)if len(v)else 1.
 return np.exp(-d/(2*width)),np.exp(-cdist(h,z,'sqeuclidean')/(2*width))

def inspect_bank(path,scenario,f,s):
 g=load(path/'geometry.npz');inp=load(path/'graph_inputs.npz');grids=load(path/'grids.npz');rec=js(path/'record.json');fit=g['fit'];query=g['query'];panel=g['panel'];pi=g['protein_indices']
 assert len(fit)>=2 and not(set(fit)&set(query));eq(fit,rec['fit'],0,0);eq(query,rec['query'],0,0);eq(panel,rec['panel'],0,0)
 np.testing.assert_array_equal(s['sample_ids'][fit],rec['fit_ids']);np.testing.assert_array_equal(s['sample_ids'][query],rec['query_ids'])
 if scenario=='tissue':assert not(set(s['tissues'][fit])&set(s['tissues'][query]))
 original=s['p'][np.ix_(fit,pi)];mask=np.isfinite(original);means=np.nanmean(original,axis=0);filled=np.where(mask,original,means)
 eq(g['protein_fill_mean'],means);eq(inp['protein'],filled);eq(inp['mask'],mask,0,0);eq(inp['metabolite'],s['m'][fit],0,0)
 # Fit-only independent reconstruction uses no query proteins.
 reconstructed={}
 for name,x in [('protein',filled),('mask',mask.astype(float)),('metabolite',s['m'][fit])]:
  l,info=reconstruct_graph(x);reconstructed[name]=l;eq(g['L_'+name],l);eig=np.linalg.eigvalsh(l)
  assert eig.min()>-1e-9;eq(l.sum(1),np.zeros(len(fit)));eq(np.trace(l),len(fit));eq(l,l.T,0,0)
  graphs_summary.append(dict(scenario=scenario,fold=f,bank=path.name,graph=name,min_eigenvalue=float(eig.min()),**info))
 perm=g['permutation'];rng=np.random.default_rng(2026100809);expected=np.arange(len(fit))
 for tissue in sorted(set(s['tissues'][fit].tolist())):
  ids=np.flatnonzero(s['tissues'][fit]==tissue);expected[ids]=rng.permutation(ids)
 eq(perm,expected,0,0);eq(g['L_shuffle'],reconstructed['protein'][np.ix_(perm,perm)])
 np.testing.assert_array_equal(s['tissues'][fit],s['tissues'][fit][perm])
 mk,mh=base_kernel(s['m'][fit],s['m'][query]);eq(g['m_only_kernel'],mk);eq(g['m_only_cross'],mh)
 fm_train=inp['fm_train'];fm_queries=inp['fm_query_views'];fks=[];fhs=[]
 assert len(fm_queries)==3
 for qview in fm_queries:
  fk,fh=base_kernel(fm_train,qview);fks.append(fk);fhs.append(fh)
 eq(g['virtual_fm_kernel'],.5*mk+.5*fks[0]);eq(g['virtual_fm_cross'],.5*mh+.5*np.mean(fhs,axis=0))
 if scenario=='tissue':
  old=a.fm_parent/f'fold-{f}'/path.name;tr=load(old/'train-features.npz');qr=load(old/'query-features.npz');refgrid=load(old/'per-alpha-predictions.npz')
  eq(fm_train,tr['virtual_fm'],0,0);eq(fm_queries,qr['virtual_fm'],0,0)
  for key in ['train','query']:assert sha(old/(key+'-features.npz'))==rec['provenance'][key+'_sha']
  assert sha(old/'selection.json')==rec['provenance']['split_sha']
  for base in ['m_only','virtual_fm']:eq(grids[base],refgrid[base],0,0)
 for base in ['m_only','virtual_fm']:
  k=g[base+'_kernel'];h=g[base+'_cross'];assert np.linalg.eigvalsh(k).min()>-1e-8
  for graph in ['protein','mask','shuffle','metabolite']:
   arr=grids[base+'__'+graph];assert arr.shape==(16,len(query),len(panel)) and np.isfinite(arr).all();eq(arr[:4],grids[base],0,0)
   # Representative real-data solve: all observed labels for three fixed drug
   # positions, mu=1 and alpha=1. General formula and all 16 configs are already
   # tested separately on small arrays; full saved selections are checked below.
   l=g['L_'+graph];mu=1.;R=np.eye(len(k))+mu*l@k
   ke=solve(R.T,k.T,assume_a='gen').T;he=solve(R.T,h.T,assume_a='gen').T
   for drug in sorted(set([0,len(panel)//2,len(panel)-1])):
    yy=s['auc'][fit,panel[drug]];obs=np.flatnonzero(np.isfinite(yy));mean=yy[obs].mean();pred=np.full(len(query),mean)
    if len(obs)>=20:pred=he[:,obs]@solve(ke[np.ix_(obs,obs)]+np.eye(len(obs)),yy[obs]-mean,assume_a='gen')+mean
    eq(pred,arr[9,:,drug],1e-8,1e-8)
 banks.append(dict(scenario=scenario,fold=f,bank=path.name,nfit=len(fit),nquery=len(query),nprotein=len(pi)))
 return g,grids

source=a.parent/'source-inputs.npz';assert sha(source)=='52d71517a8952e83a1603cd831f09449544102d0ca5ff829ff01be9bf79cd637';s=load(source);assert len(s['m'])==421
for scenario in ['tissue','new_cell']:
 out=a.run/'output'/scenario
 if not(out/'protocol.json').exists():
  assert a.partial;continue
 protocol=js(out/'protocol.json');assert protocol['source_sha256']==sha(source);assert protocol['script_sha256']==sha(a.run/'code/run_geometry.py');assert protocol['helper_sha256']==sha(a.run/'code/run_recoverability.py');assert protocol['fm_source_sha256']==sha(a.fm_parent/'source-inputs.npz')
 expected_configs=[[mu,alpha]for mu in [0,.1,1,10]for alpha in [.1,1,10,100]];assert protocol['configs']==expected_configs
 outerids=np.array(protocol['outer_fold_ids']);assert sorted(set(outerids))==list(range(5));seen=[]
 for f in range(5):
  dest=out/f'fold-{f}'
  if not(dest/'status.json').exists():
   assert a.partial;continue
  saved=js(dest/'selection.json');outerfit=np.flatnonzero(outerids!=f);held=np.flatnonzero(outerids==f);eq(saved['fit'],outerfit,0,0);eq(saved['held'],held,0,0);seen.extend(held.tolist());panel=np.array(saved['panel']);inner=np.array(saved['inner_ids']);sums={};cnts={}
  for j in range(3):
   g,grids=inspect_bank(dest/f'inner-{j}',scenario,f,s);eq(g['fit'],outerfit[inner!=j],0,0);eq(g['query'],outerfit[inner==j],0,0)
   truth=s['auc'][np.ix_(g['query'],panel)];sd=np.maximum(np.nanstd(s['auc'][np.ix_(g['fit'],panel)],axis=0),1e-8)
   for arm,grid in grids.items():
    valid=np.isfinite(truth);n=valid.sum(0);err=np.where(valid[None],((grid-truth[None])/sd)**2,0);loss=err.sum(1)/np.maximum(n,1)[None];count=np.tile((n>0).astype(int),(len(grid),1));sums[arm]=sums.get(arm,0)+loss;cnts[arm]=cnts.get(arm,0)+count
  selected={}
  for arm in sums:
   eq(sums[arm],saved['losses'][arm]);eq(cnts[arm],saved['counts'][arm],0,0);avg=np.divide(sums[arm],cnts[arm],out=np.full_like(sums[arm],np.inf),where=cnts[arm]>0);sel=avg.argmin(0);sel[np.all(cnts[arm]==0,axis=0)]=1;eq(sel,saved['selected'][arm],0,0);selected[arm]=sel
   if '__'in arm:selection_summary.append(dict(scenario=scenario,fold=f,arm=arm,mu_counts={str(mu):int(np.sum(sel//4==i))for i,mu in enumerate([0,.1,1,10])}))
  g,grids=inspect_bank(dest/'outer',scenario,f,s);eq(g['fit'],outerfit,0,0);eq(g['query'],held,0,0);p=load(dest/'predictions.npz');truth=s['auc'][np.ix_(held,panel)];sd=np.maximum(np.nanstd(s['auc'][np.ix_(outerfit,panel)],axis=0),1e-8);eq(p['truth'],truth,0,0);eq(p['train_sd'],sd);eq(p['panel'],panel,0,0);valid=np.isfinite(truth).sum(0)>=5;eq(p['evaluated'],valid,0,0)
  for arm,grid in grids.items():
   expected=grid[selected[arm],:,np.arange(len(panel))].T;eq(p[arm],expected)
  if scenario=='tissue':
   old=load(a.parent/f'fold-{f}'/'predictions.npz')
   for arm,ref in [('m_only','m_only'),('virtual_fm','frozen_virtual_fm')]:eq(p[arm],old[ref],0,0)
   for arm in ['virtual_p','measured_p','measured_mask','frozen_virtual_rna']:eq(p['old_'+arm],old[arm],0,0)
  arms=list(grids)+([k for k in p if k.startswith('old_')]if scenario=='tissue'else[])
  for arm in arms:
   mse=[];centered=[];rhos=[]
   for d in np.flatnonzero(valid):
    ok=np.isfinite(truth[:,d]);x=truth[ok,d];y=p[arm][ok,d];e=(y-x)/sd[d];mse.append(np.mean(e**2));centered.append(np.mean((e-e.mean())**2))
    if np.std(x)>0 and np.std(y)>1e-15:rhos.append(spearmanr(x,y)[0])
   scores.append(dict(scenario=scenario,fold=f,arm=arm,score=float(np.sqrt(np.mean(mse))),spearman=float(np.mean(rhos)),centered_mse=float(np.mean(centered))))
  print(json.dumps(dict(event='verified_fold',scenario=scenario,fold=f,checks=checks)),flush=True)
 if not a.partial:assert sorted(seen)==list(range(421));summary=js(out/'summary.json');assert summary['status']=='completed'
 if(out/'summary.json').exists():
  summary=js(out/'summary.json')
  for arm,row in summary['summary'].items():
   relevant=[r for r in scores if r['scenario']==scenario and r['arm']==arm]
   if len(relevant)==5:
    for field in ['score','spearman','centered_mse']:eq(np.mean([r[field]for r in relevant]),row[field])
result=dict(status='PARTIAL'if a.partial else'PASS',checks=checks,banks=len(banks),fold_metric_rows=len(scores),source_sha256=sha(source),code_sha256=sha(a.run/'code/run_geometry.py'),seconds=time.perf_counter()-start,bank_records=banks,graph_diagnostics=graphs_summary,selection_mu_counts=selection_summary,independently_recomputed_scores=scores,limitations=['No independent generator refitting; original bank hashes/arrays and current fit/query usage verified.','Representative three drug positions per graph/base bank directly recomputed at mu=1 alpha=1; all saved configs independently reselected using all inner-fold losses.','Graph no-query-P claim verified by reconstructing every graph only from source fit P plus source-code inspection; no production rerun with altered query P.'])
a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items()if k not in ['bank_records','graph_diagnostics','selection_mu_counts','independently_recomputed_scores']},indent=2))

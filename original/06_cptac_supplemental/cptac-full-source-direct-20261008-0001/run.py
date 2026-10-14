"""One full-source linear direct comparator; no old prediction mutation."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='2'
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0,'/oldcode/src')
from creed.science.masked_baseline import unique_feature_map,source_gene_labels,metrics
ALPHAS=np.array([.1,1.,10.,100.]);seed=20261008;out=Path('/out');start=time.time()
def save(name,d):
    (out/name).write_text(json.dumps(d,indent=2))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
config=dict(task='Full-source linear direct vs unchanged CPTAC anchor context',
    scope='Existing BRCA122 LUAD110 development, original mixed patient folds only',
    alpha=ALPHAS.tolist(),inner_folds=3,seed=seed,selection='one alpha per direction/outer fold; minimize pooled inner observed squared error in each inner training target scale',
    input='all mapped source coordinates with >=5 finite training rows and positive observed SD; standardized values plus missing indicators',
    fill='training median; fitted value mean/SD; centered availability indicators',
    kernel='linear Z Zt divided by mean fit diagonal; query uses same scale',
    solve='separate exact observed-label masks, free intercept via kernel centering; sum squared error plus alpha RKHS norm',
    unchanged='original outer target panels, original held patients, unchanged anchor and original baseline predictions',
    script_sha256=sha('/code/run.py'),training=True,status='design_frozen_before_fit')
save('config.json',config)
base=json.loads(Path('/old/summary.json').read_text());metadata=json.loads(Path('/data/summary.json').read_text())
hashes={r['cohort']:r['bundle_sha256'] for r in metadata['cohorts']};bundles=[]
for cohort in base['config']['cohorts']:
 p=Path('/data')/(cohort+'.npz');assert sha(p)==hashes[cohort]
 with np.load(p,allow_pickle=False) as z: bundles.append({k:z[k] for k in z.files if k.startswith(('rna__','protein__')) or k=='patient_ids'})
ids={};raw={}
for modality in ['rna','protein']:
 maps=[unique_feature_map(source_gene_labels(b[modality+'__feature_ids'],c,modality)) for b,c in zip(bundles,base['config']['cohorts'])]
 ids[modality]=np.array(sorted(set(maps[0])&set(maps[1])))
 raw[modality]=np.concatenate([np.where(b[modality+'__observed'],b[modality+'__values'],np.nan)[:,[m[g] for g in ids[modality]]] for b,m in zip(bundles,maps)])
patients=np.concatenate([b['patient_ids'] for b in bundles]);groups=np.concatenate([np.full(len(b['patient_ids']),i) for i,b in enumerate(bundles)])
assert len(patients)==232 and len(set(patients))==232
def features(x,q):
 n=np.isfinite(x).sum(0);mu=np.nansum(x,0)/np.maximum(n,1);sd=np.sqrt(np.sum(np.where(np.isfinite(x),x-mu,0)**2,0)/np.maximum(n,1))
 keep=(n>=5)&(sd>1e-8);x=x[:,keep];q=q[:,keep];fill=np.nanmedian(x,0)
 a=np.where(np.isfinite(x),x,fill);b=np.where(np.isfinite(q),q,fill);mu=a.mean(0);sd=a.std(0);sd=np.where(sd>1e-8,sd,1)
 a=np.column_stack(((a-mu)/sd,(~np.isfinite(x)).astype(float)));b=np.column_stack(((b-mu)/sd,(~np.isfinite(q)).astype(float)))
 center=a.mean(0);a-=center;b-=center;scale=np.sum(a*a)/len(a)
 assert scale>0
 return a@a.T/scale,b@a.T/scale,dict(source_columns=int(keep.sum()),kernel_scale=float(scale))
def predictions(x,q,y):
 k,h,meta=features(x,q);outpred=np.zeros((len(ALPHAS),len(q),y.shape[1]));masks={}
 for j in range(y.shape[1]):masks.setdefault(np.isfinite(y[:,j]).tobytes(),[]).append(j)
 for key,cols in masks.items():
  good=np.frombuffer(key,dtype=bool);ii=np.flatnonzero(good);yy=y[np.ix_(ii,cols)];ym=yy.mean(0)
  if len(ii)<2: raise ValueError('too few training targets')
  a=k[np.ix_(ii,ii)];b=h[:,ii];col=a.mean(0);grand=col.mean();ac=a-col-col[:,None]+grand;bc=b-b.mean(1)[:,None]-col+grand
  ev,u=eigh(ac,check_finite=False);ev=np.maximum(ev,0);uy=u.T@(yy-ym)
  for ai,alpha in enumerate(ALPHAS):outpred[ai][:,cols]=bc@(u@(uy/(ev[:,None]+alpha)))+ym
 meta['mask_groups']=len(masks)
 assert np.isfinite(outpred).all()
 return outpred,meta
records=[]
for rec in base['records']:
 if not rec['split'].startswith('mixed'):continue
 source,target=rec['direction'].split('_to_');name=rec['direction']+'__'+rec['split']
 old=np.load('/old/'+name+'.npz',allow_pickle=False);fit=old['train_indices'];held=old['test_indices'];lookup={g:i for i,g in enumerate(ids[target])};panel=np.array([lookup[g] for g in old['target_ids']])
 assert not set(patients[fit])&set(patients[held]);np.testing.assert_array_equal(patients[held],old['patient_ids_test'])
 X=raw[source];Y=raw[target][:,panel];loss=np.zeros(4);counts=0;inner=[]
 for fi,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=seed).split(fit,groups[fit])):
  a=fit[a];b=fit[b];mu=np.nanmean(Y[a],0);sd=np.nanstd(Y[a],0);sd=np.where(sd>1e-8,sd,1);y=(Y[a]-mu)/sd;t=(Y[b]-mu)/sd
  pred,meta=predictions(X[a],X[b],y);mask=np.isfinite(t);loss+=np.sum(np.where(mask[None],(pred-np.where(mask,t,0)[None])**2,0),axis=(1,2));counts+=mask.sum()
  inner.append(dict(fold=fi,fit=a.tolist(),held=b.tolist(),**meta));print(json.dumps(dict(name=name,inner=fi,elapsed=time.time()-start)),flush=True)
 ai=int(np.argmin(loss/counts));mu=old['target_mean'];sd=old['target_scale'];grid,meta=predictions(X[fit],X[held],(Y[fit]-mu)/sd);truth=(Y[held]-mu)/sd
 np.testing.assert_allclose(truth,old['truth'],atol=1e-12);pred=grid[ai]
 scores={a:metrics(truth,p) for a,p in [('full_source_linear',pred),('anchor_context',old['anchor_context']),('masked_pca_ridge',old['masked_pca_ridge']),('same_gene',old['same_gene'])]}
 np.savez_compressed(out/(name+'.npz'),prediction=pred,truth=truth,fit=fit,held=held,target_ids=old['target_ids'],target_mean=mu,target_scale=sd)
 row=dict(direction=rec['direction'],split=rec['split'],alpha=float(ALPHAS[ai]),inner_losses=(loss/counts).tolist(),inner=inner,metadata=meta,scores=scores)
 save(name+'.json',row);records.append(row);print(json.dumps(dict(name=name,complete=True,alpha=float(ALPHAS[ai]),scores=scores,elapsed=time.time()-start)),flush=True)
means={}
for direction in ['rna_to_protein','protein_to_rna']:
 rr=[r for r in records if r['direction']==direction]
 means[direction]={a:float(np.mean([r['scores'][a]['standardized_rmse'] for r in rr])) for a in rr[0]['scores']}
save('summary.json',dict(status='completed',records=records,means=means,elapsed_seconds=time.time()-start,config=config,source_summary_sha256=sha('/old/summary.json')))
print(json.dumps(dict(completed=True,means=means,elapsed=time.time()-start)),flush=True)

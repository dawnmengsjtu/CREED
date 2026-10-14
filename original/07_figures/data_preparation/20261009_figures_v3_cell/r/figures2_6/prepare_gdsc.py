"""Recompute Figures 2/3/4/6 from archived measurements and frozen predictions.
No training, synthetic measurements, outcome-driven example selection or p-values.
"""
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import csv,json,hashlib
ROOT=Path(__file__).resolve().parents[2];RAW=ROOT/'data/raw';OUT=ROOT/'delivery_R_20261009/figures2_6'
for f in [2,3,4,6]:(OUT/f'figure{f}'/'plot_data').mkdir(parents=True,exist_ok=True)
SOURCES=set()
def load(p):SOURCES.add(p);return np.load(p,allow_pickle=False)
def save(f,name,rows):
 rows=list(rows);assert rows,(f,name)
 path=OUT/f'figure{f}'/'plot_data'/f'{name}.csv'
 with path.open('w',newline='') as h:
  w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader()
  for r in rows:w.writerow({k:('' if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v) for k,v in r.items()})
def ranks(x):
 order=np.argsort(x,kind='stable');v=x[order];_,start,count=np.unique(v,return_index=True,return_counts=True)
 r=np.empty(len(x));r[order]=np.repeat(start+(count+1)/2,count);return r

def cor(x,y,rank=False):
 ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok]
 if len(x)<5 or np.ptp(x)<1e-12 or np.ptp(y)<1e-12:return np.nan
 if rank:x=ranks(x);y=ranks(y)
 x=x-x.mean();y=y-y.mean();return float(x@y/np.sqrt((x@x)*(y@y)))

src=load(RAW/'gdsc_metabolite_full_protein/source-inputs.npz');ids=src['sample_ids'];tissues=src['tissues'];folds=src['fold_ids'];auc=src['auc']
assert len(set(ids))==421
annotation_path=RAW/'figure2_6_extra/drug_annotations.json';SOURCES.add(annotation_path);annot=json.loads(annotation_path.read_text())
name=lambda did:annot.get(str(did),{}).get('names',[str(did)])[0]
# Example proteins: training-fold-0 variance order, restricted to eligibility in all folds.
bank=[]
for f in range(5):
 base=RAW/'gdsc_metabolite_full_protein'/f'fold-{f}'
 bank.append((load(base/'predictions.npz'),load(base/'outer/panel.npz'),load(base/'outer/measured-inputs.npz'),load(base/'outer/query-native.npz')))
common=set.intersection(*[set(z[1]['protein_ids']) for z in bank]);fit0=folds!=0
cv=np.array([p in common for p in src['protein_ids']]);v=np.nanvar(src['p'][fit0][:,cv],axis=0)
select=src['protein_ids'][cv][np.argsort(-v,kind='stable')[:20]].tolist();lab=lambda p:str(p).split(';')[-1].replace('_HUMAN','')
save(2,'protein_selection',[dict(protein_id=p,label=lab(p),rank=i+1,rule='fold0_training_variance_common_eligibility') for i,p in enumerate(select)])
feat=[];pfold=[];scatter=[];per_sample=[];inventory=[]
p_ar=['virtual_p','row_shuffle_p','observed_value_shuffle_p']
for f,(pred,panel,meas,native) in enumerate(bank):
 held=pred['held'];fit=pred['fit'];prot=panel['protein_ids'];idx=panel['indices'];truth=src['p'][np.ix_(held,idx)];mask=np.isfinite(truth)
 assert np.array_equal(mask,meas['measured_mask_query'][0].astype(bool))
 assert np.allclose(truth[mask],meas['measured_p_query'][0][mask])
 train=src['p'][np.ix_(fit,idx)];mu=np.nanmean(train,axis=0);sd=np.nanstd(train,axis=0);assert (sd>1e-8).all()
 arr={a:native[a].mean(axis=0) for a in p_ar};arr['train_mean']=np.broadcast_to(mu,truth.shape)
 metric={};corrs={}
 for a in arr:
  assert np.isfinite(arr[a]).all()
  errors=np.where(mask,((arr[a]-truth)/sd)**2,np.nan);rmse=float(np.sqrt(np.nanmean(errors)))
  metric[a]=rmse
  for i,si in enumerate(held):per_sample.append(dict(fold=f,sample_id=ids[si],arm=a,rmse=float(np.sqrt(np.nanmean(errors[i]))),n_observed=int(mask[i].sum())))
  if a in p_ar:
   corrs[a]=np.array([cor(truth[:,j],arr[a][:,j],True) for j in range(len(prot))])
 valid=np.logical_and.reduce([np.isfinite(corrs[a]) for a in p_ar])
 for a in arr:
  pfold.append(dict(fold=f,arm=a,rmse=metric[a],mean_rho=float(corrs[a][valid].mean()) if a in p_ar else np.nan,n_rho=int(valid.sum()) if a in p_ar else 0,n_features=len(prot),n_entries=int(mask.sum())))
  if a in p_ar:
   for j,pid in enumerate(prot):
    feat.append(dict(fold=f,arm=a,protein_id=pid,label=lab(pid),rho=float(corrs[a][j]),common_valid=bool(valid[j]),observed=int(mask[:,j].sum()),selected=pid in select))
 for j,pid in enumerate(prot):
  if pid in select[:6]:
   for i in np.where(mask[:,j])[0]:scatter.append(dict(fold=f,sample_id=ids[held[i]],protein_id=pid,label=lab(pid),observed_z=float((truth[i,j]-mu[j])/sd[j]),predicted_z=float((arr['virtual_p'][i,j]-mu[j])/sd[j])))
 inventory.append(dict(fold=f,held_n=len(held),eligible_proteins=len(prot),common_correlation_features=int(valid.sum()),observed_entries=int(mask.sum())))
save(2,'fold_metrics',pfold);save(2,'feature_rho',feat);save(2,'protein_scatter',scatter);save(2,'sample_metrics',per_sample);save(2,'inventory',inventory)
print('Figure 2 tables ready',flush=True)
# Functional scores: exact archived macro-drug standardized RMSE definition.
arms=['m_only','virtual_p','row_shuffle_p','observed_value_shuffle_p','ordinary_m','virtual_mask','measured_p','measured_mask','frozen_virtual_rna','frozen_virtual_fm']
fmetrics=[];drugs=[];samples=[];rankrows=[];cal=[];scores={};errors={};pred_by_fold={};panels={}
summary_path=RAW/'gdsc_metabolite_full_protein/summary.json';SOURCES.add(summary_path);summary=json.loads(summary_path.read_text());stored={(z['fold'],z['arm']):z['score'] for z in summary['fold_metrics']}
common_drugs=set.intersection(*[set(z[0]['drug_ids'].astype(str)) for z in bank])
# Only coverage and stable numeric ID order determine plotted examples.
counts={str(d):int(np.isfinite(auc[:,j]).sum()) for j,d in enumerate(src['drug_ids'])}
examples=sorted(common_drugs,key=lambda d:(-counts[d],int(d)))[:20]
save(3,'drug_selection',[dict(drug_id=d,name=name(d),observed=counts[d],rank=i+1) for i,d in enumerate(examples)])
for f,(pred,_,_,_) in enumerate(bank):
 held,fit,panel=(pred[k] for k in ['held','fit','panel']);truth=auc[np.ix_(held,panel)];sd=np.nanstd(auc[np.ix_(fit,panel)],axis=0);count=np.isfinite(truth).sum(axis=0);ev=count>=5
 assert np.array_equal(held,np.where(folds==f)[0])
 assert all(np.isfinite(pred[a]).all() for a in arms)
 panels[f]=pred['drug_ids'].astype(str);errors[f]={};pred_by_fold[f]=pred
 for a in arms:
  err=((pred[a]-truth)/sd)**2;errors[f][a]=err;dm=np.nanmean(err,axis=0);score=float(np.sqrt(dm[ev].mean()));assert np.isclose(score,stored[(f,a)],atol=1e-12,rtol=1e-12),(f,a,score,stored[(f,a)])
  scores[f,a]=score;fmetrics.append(dict(fold=f,arm=a,srmse=score,drugs=int(ev.sum()),observed_pairs=int(count[ev].sum())))
  for j,did in enumerate(panels[f]):
   if ev[j]:drugs.append(dict(fold=f,arm=a,drug_id=did,name=name(did),n=int(count[j]),smse=float(dm[j]),rho=cor(truth[:,j],pred[a][:,j],True)))
  for i,si in enumerate(held):samples.append(dict(fold=f,arm=a,sample_id=ids[si],tissue=tissues[si],smse=float(np.nanmean(err[i,ev])) if np.isfinite(truth[i,ev]).any() else np.nan,n=int(np.isfinite(truth[i,ev]).sum())))
  # Retrospective ranking on the same observed candidate pool for every arm, per cell line.
  if a in ['m_only','virtual_p','measured_p','frozen_virtual_fm']:
   for i,si in enumerate(held):
    ok=np.where(np.isfinite(truth[i])&ev)[0];order_true=ok[np.argsort(truth[i,ok],kind='stable')];order_pred=ok[np.argsort(pred[a][i,ok],kind='stable')]
    for k in [5,10,20,50]:
     if len(ok)>=k:rankrows.append(dict(fold=f,sample_id=ids[si],arm=a,k=k,recall=len(set(order_true[:k])&set(order_pred[:k]))/k,candidates=len(ok)))
 for j,did in enumerate(panels[f]):
  if did in examples[:6]:
   good=np.where(np.isfinite(truth[:,j]))[0];ordered=good[np.argsort(truth[good,j],kind='stable')]
   for b,block in enumerate(np.array_split(ordered,5)):
    if len(block):
     for a in ['m_only','virtual_p','measured_p']:
      cal.append(dict(drug_id=did,name=name(did),fold=f,bin=b+1,n=len(block),observed_auc=float(truth[block,j].mean()),predicted_auc=float(pred[a][block,j].mean()),arm=a))
save(3,'fold_metrics',fmetrics);save(3,'per_drug_metrics',drugs);save(3,'per_sample_metrics',samples);save(3,'ranking_recall',rankrows);save(3,'calibration_bins',cal)
# paired deltas, always positive when virtual protein improves on comparator.
lookup={(x['fold'],x['drug_id'],x['arm']):x for x in drugs};deltas=[]
for row in [x for x in drugs if x['arm']=='virtual_p']:
 for a in [a for a in arms if a!='virtual_p']:
  co=lookup[row['fold'],row['drug_id'],a]
  deltas.append(dict(fold=row['fold'],drug_id=row['drug_id'],name=row['name'],control=a,delta=co['smse']-row['smse'],n=row['n']))
save(3,'drug_deltas',deltas)
fold_delta=[dict(fold=f,control=a,delta=scores[f,a]-scores[f,'virtual_p']) for f in range(5) for a in arms if a!='virtual_p'];save(4,'fold_deltas',fold_delta)
save(4,'drug_deltas',deltas)
loo=[]
for a in arms:
 if a=='virtual_p':continue
 vals=np.array([scores[f,a]-scores[f,'virtual_p'] for f in range(5)])
 for omitted in range(5):loo.append(dict(control=a,omitted_fold=omitted,delta=float(np.delete(vals,omitted).mean()),full_mean=float(vals.mean())))
save(4,'leave_one_fold_out',loo)
sl={(x['sample_id'],x['arm']):x for x in samples};sdeltas=[]
for x in [z for z in samples if z['arm']=='virtual_p']:
 for a in ['m_only','virtual_mask','frozen_virtual_fm']:
  sdeltas.append(dict(sample_id=x['sample_id'],fold=x['fold'],tissue=x['tissue'],control=a,delta=sl[x['sample_id'],a]['smse']-x['smse']))
save(4,'sample_deltas',sdeltas)
headroom=[dict(fold=f,direct=scores[f,'m_only'],virtual=scores[f,'virtual_p'],measured=scores[f,'measured_p'],absolute_gain=scores[f,'m_only']-scores[f,'virtual_p'],measured_gap=scores[f,'m_only']-scores[f,'measured_p'],retained_fraction=(scores[f,'m_only']-scores[f,'virtual_p'])/(scores[f,'m_only']-scores[f,'measured_p'])) for f in range(5)]
assert all(z['measured_gap']>0 for z in headroom);save(3,'headroom',headroom)
print('Figures 3/4 exact-score replay passed (50 fold-arm values)',flush=True)
# Figure 6: existing putative-target annotations; no invented genetics or approval status.
# Exact whole strings define disjoint groups; comma-separated targets are not split.
def target(d):
 x=[v for v in annot.get(str(d),{}).get('putative_targets',[]) if v and v.strip().upper() not in ('NA','N/A','NONE','UNKNOWN')];return ' | '.join(sorted(set(x))) if x else 'Unannotated'
allids=src['drug_ids'].astype(str).tolist();categories=Counter(target(d) for d in allids);top=sorted(categories,key=lambda g:(-categories[g],g))[:7]
group=lambda d:target(d) if target(d) in top else 'Other annotations'
groups=top+['Other annotations'];annrows=[dict(drug_id=d,name=name(d),target_annotation=target(d),group=group(d),all_fold_panel=d in common_drugs) for d in allids];save(6,'drug_annotations',annrows)
eval_sets=[{z['drug_id'] for z in drugs if z['fold']==f and z['arm']=='m_only'} for f in range(5)];common_eval=sorted(set.intersection(*eval_sets),key=int)
flow=[]
for d in common_eval:
 direct=np.mean([lookup[f,d,'m_only']['smse'] for f in range(5)])
 virtual=np.mean([lookup[f,d,'virtual_p']['smse'] for f in range(5)])
 measured=np.mean([lookup[f,d,'measured_p']['smse'] for f in range(5)])
 flow.append(dict(drug_id=d,name=name(d),group=group(d),virtual_status='Lower error' if virtual<direct else 'Higher / equal',measured_status='Lower error' if measured<direct else 'Higher / equal',virtual_gain=direct-virtual,measured_gain=direct-measured))
save(6,'drug_flows',flow)
coverage=[dict(group=g,order=i+1,panel=label,n=sum(group(d)==g for d in dd)) for i,g in enumerate(groups) for label,dd in [('All drugs',allids),('All 5 folds',common_eval)]];save(6,'annotation_counts',coverage)
gd=[]
for f in range(5):
 for i,g in enumerate(groups):
  dd=[d for d in common_eval if group(d)==g]
  if dd:gd.append(dict(fold=f,group=g,order=i+1,n=len(dd),delta=float(np.mean([lookup[f,d,'m_only']['smse']-lookup[f,d,'virtual_p']['smse'] for d in dd]))))
save(6,'annotation_effects',gd)
# Copy required tables into each figure to keep its data contract self-contained.
import shutil
for f,n in [(4,'drug_selection')]:shutil.copyfile(OUT/'figure3/plot_data'/f'{n}.csv',OUT/f'figure{f}/plot_data'/f'{n}.csv')
receipt={'source_files':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(SOURCES)],'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'checks':{'aligned_samples':421,'fold_arm_score_replays':50,'all_score_replays_match':True,'protein_target_matches_raw_on_observed_entries':True},'selection_rules':{'protein_examples':'fold0-training variance, all-fold eligible intersection','drug_examples':'decreasing observed count, then numeric drug ID; no response-effect selection','drug_groups':'top7 exact annotation strings by full panel frequency'},'counts':{'common_eligible_proteins':len(common),'all_fold_evaluated_drugs':len(common_eval),'drug_annotation_groups':groups},'evidence_status':{'scientific_claim':False,'external_confirmation':False,'chainable':False},'evaluation':'archived development results; no new training or independent confirmation'}
(OUT/'GDSC_RAW_TO_TABLE_RECEIPT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt['counts']),flush=True)

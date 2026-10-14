"""Descriptive Figure 1 from actual archived arrays; NumPy + stdlib only.
Fold 0 is the illustrative holdout; feature selection/PCA fitting use folds 1-4.
No model performance is evaluated by this script.
"""
from pathlib import Path
from collections import Counter
import csv, hashlib, json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'delivery_R_20261009_v3/figure1'  # v3: provenance only - do NOT re-run; plot_data is frozen
DATA=OUT/'plot_data'; DATA.mkdir(parents=True,exist_ok=True)
raw=ROOT/'data/raw'
sources={k:raw/k/'source-inputs.npz' for k in ['gdsc_metabolite_query','gdsc_rna_inverse','gdsc_metabolite_full_protein']}
q,r,p=[np.load(sources[k],allow_pickle=False) for k in sources]
for z in (r,p):
 for k in ['m','auc','sample_ids','tissues','fold_ids']:
  eq=np.array_equal(q[k],z[k],equal_nan=True) if q[k].dtype.kind in 'fi' else np.array_equal(q[k],z[k])
  assert eq,k
assert len(set(q['sample_ids']))==len(q['sample_ids'])==421
for k in ['rna_repr','fm','m_repr']: assert np.array_equal(q[k],r[k],equal_nan=True),k
ids,tissue,folds=q['sample_ids'],q['tissues'],q['fold_ids']; train=folds!=0
assert train.sum()==328
counts=Counter(tissue.tolist()); tissue_order=sorted(counts,key=lambda t:(-counts[t],t));top=tissue_order[:8]
groups=np.array([t if t in top else 'Other tissues' for t in tissue]);group_order=top+['Other tissues']
short={'Haematopoietic and Lymphoid':'Haem. / lymphoid','Central Nervous System':'CNS','Large Intestine':'Large intestine','Peripheral Nervous System':'PNS','Biliary Tract':'Biliary tract','Head and Neck':'Head / neck','Soft Tissue':'Soft tissue'}
def save(name,rows):
 rows=list(rows);assert rows,name
 with (DATA/f'{name}.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def tissue_fields(t): return {'tissue':t,'tissue_label':short.get(t,t),'n':counts[t],'tissue_order':tissue_order.index(t)+1}
save('sample_metadata',[dict(sample_id=s,tissue=t,tissue_label=short.get(t,t),tissue_group=g,group_label=short.get(g,g),fold=int(f),partition='Held out' if f==0 else 'Training') for s,t,g,f in zip(ids,tissue,groups,folds)])
save('B_tissue_composition',[dict(group=g,label=short.get(g,g),n=int((groups==g).sum()),order=i+1) for i,g in enumerate(group_order)])
save('B_fold_composition',[dict(fold=f,n=int((folds==f).sum())) for f in range(5)])
save('asset_inventory',[dict(asset=k,features=a.shape[1],samples=a.shape[0],finite=int(np.isfinite(a).sum()),total=int(a.size),coverage=float(np.isfinite(a).mean())) for k,a in [('Metabolites',q['m']),('RNA',r['rna']),('Protein',p['p']),('Drug AUC',q['auc'])]])
snippets=[]
for name,a in [('Metabolites',q['m']),('Protein',p['p']),('Drug AUC',q['auc'])]:
 x=a[:6,:5].copy()
 for j in range(x.shape[1]):
  good=np.isfinite(x[:,j]);v=x[good,j]
  if len(v):x[good,j]=(v-v.min())/max(float(np.ptp(v)),1e-12)
 for i in range(6):
  for j in range(5):snippets.append(dict(asset=name,row=i,col=j,value=float(x[i,j]) if np.isfinite(x[i,j]) else ''))
save('A_real_array_snippets',snippets)
pcs={};pcmeta={};pcrows=[];selection=[]
for name,a in [('Metabolites',q['m']),('RNA',r['rna'])]:
 var=a[train].var(axis=0,ddof=1);keep=np.flatnonzero(var>1e-12)
 if name=='RNA':keep=keep[np.argsort(-var[keep],kind='stable')[:1000]]
 means=a[train][:,keep].mean(axis=0);sds=a[train][:,keep].std(axis=0,ddof=1)
 standardized=(a[:,keep]-means)/sds
 _,sing,vh=np.linalg.svd(standardized[train],full_matrices=False)
 load=vh[:2].T
 for j in range(2):
  if load[np.argmax(np.abs(load[:,j])),j]<0:load[:,j]*=-1
 coords=standardized@load;pcs[name]=coords;ratio=sing[:2]**2/(sing**2).sum()
 pcmeta[name]={'feature_count':len(keep),'training_n':int(train.sum()),'explained_variance':ratio.tolist()}
 for i in range(len(ids)):pcrows.append(dict(modality=name,sample_id=ids[i],tissue_group=groups[i],fold=int(folds[i]),partition='Training' if train[i] else 'Held out',pc1=float(coords[i,0]),pc2=float(coords[i,1])))
 if name=='RNA':
  for j in keep:selection.append(dict(feature=r['gene_ids'][j],training_variance=float(var[j])))
save('C_pca_coordinates',pcrows);save('C_rna_feature_selection',selection)
coverage_rows=[]
for name,a in [('Metabolites',q['m']),('RNA',r['rna']),('Protein',p['p']),('Drug AUC',q['auc'])]:
 for f in range(5):
  block=a[folds==f];observed=int(np.isfinite(block).sum())
  for status,n in [('Observed',observed),('Missing',block.size-observed)]:coverage_rows.append(dict(asset=name,fold=f,status=status,n=n,total=int(block.size),pct=n/block.size*100))
save('D_entry_coverage',coverage_rows)
pr=p['p'];trp=pr[train];coverage=np.isfinite(trp).mean(axis=0);eligible=np.flatnonzero(coverage>=.8)
var=np.nanvar(trp[:,eligible],axis=0,ddof=1);pk=eligible[np.argsort(-var,kind='stable')[:16]]
pm=np.nanmean(trp[:,pk],axis=0);ps=np.nanstd(trp[:,pk],axis=0,ddof=1);erows=[];psel=[]
for j,k in enumerate(pk):
 ident=str(p['protein_ids'][k]);label=ident.split(';')[-1].replace('_HUMAN','');psel.append(dict(protein_id=ident,label=label,rank=j+1,training_coverage=float(coverage[k])))
 for g in group_order:
  vals=pr[groups==g,k];good=np.isfinite(vals)
  erows.append(dict(protein_id=ident,label=label,rank=j+1,group=g,group_label=short.get(g,g),n=int(len(vals)),n_observed=int(good.sum()),observed_pct=float(good.mean()*100),mean_z=float((vals[good].mean()-pm[j])/ps[j]) if good.any() else ''))
save('E_protein_by_tissue',erows);save('E_protein_selection',psel)
flows=[]
for t in tissue_order:
 fs=np.unique(folds[tissue==t]);assert len(fs)==1,(t,fs)
 flows.append(dict(**tissue_fields(t),fold=int(fs[0]),group=t if t in top else 'Other tissues'))
assert sum(x['n'] for x in flows)==len(ids)
save('F_tissue_to_fold',flows)
mv=q['m'][train].var(axis=0,ddof=1);mk=np.argsort(-mv,kind='stable')[:20];gro=[];mselect=[]
for rank,k in enumerate(mk):
 means=np.array([q['m'][tissue==t,k].mean() for t in tissue_order]);scaled=(means-means.min())/max(float(np.ptp(means)),1e-12)
 mselect.append(dict(feature=str(q['metabolite_ids'][k]),rank=rank+1,training_variance=float(mv[k])))
 for t,mean,value in zip(tissue_order,means,scaled):gro.append(dict(**tissue_fields(t),feature=str(q['metabolite_ids'][k]),rank=rank+1,fold=int(np.unique(folds[tissue==t])[0]),mean=float(mean),scaled_mean=float(value)))
save('G_metabolite_tissue_means',gro);save('G_metabolite_selection',mselect)
hrows=[]
for name,a in [('Protein',pr),('Drug AUC',q['auc'])]:
 rates=np.isfinite(a).mean(axis=1)*100
 for i in range(len(ids)):hrows.append(dict(asset=name,sample_id=ids[i],fold=int(folds[i]),pc1=float(pcs['Metabolites'][i,0]),pc2=float(pcs['Metabolites'][i,1]),observed_pct=float(rates[i])))
save('H_sample_coverage_pca',hrows)
summary={'samples':len(ids),'tissues':len(counts),'folds':[int((folds==f).sum()) for f in range(5)],'protein_finite':int(np.isfinite(pr).sum()),'protein_total':int(pr.size),'auc_finite':int(np.isfinite(q['auc']).sum()),'auc_total':int(q['auc'].size),'pca':pcmeta,'primary_example':'metabolite -> virtual protein -> drug AUC','scientific_claim':False,'external_confirmation':False,'scope':'Descriptive analysis of archived development assets. No new performance evidence.','RNA_role':'paired contextual asset; not deployment input in the full-protein example','selection':'C/E/G fit or select on folds 1-4; fold 0 is illustrative Lung holdout'}
(DATA/'summary.json').write_text(json.dumps(summary,indent=2))
receipt={'sources':[{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in sources.values()],'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'checks':{'aligned_ids':True,'shared_inputs_equal':True,'unique_samples':421,'tissue_disjoint_folds':True,'flow_count':sum(x['n'] for x in flows)},'tables':[{'name':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(DATA.glob('*.csv'))]}
(OUT/'RAW_TO_TABLE_RECEIPT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(summary,indent=2))

"""Frozen CPTAC development transfer arrays -> exact-score replay -> plot tables."""
from pathlib import Path
import numpy as np,csv,json,hashlib
ROOT=Path(__file__).resolve().parents[2];RAW=ROOT/'data/raw/figure2_6_extra';OUT=ROOT/'delivery_R_20261009/figures2_6/figure5';(OUT/'plot_data').mkdir(parents=True,exist_ok=True)
def save(name,rows):
 with (OUT/'plot_data'/f'{name}.csv').open('w',newline='') as h:
  w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader()
  for r in rows:w.writerow({k:'' if v is None or isinstance(v,(float,np.floating)) and not np.isfinite(v) else v for k,v in r.items()})
def cor(x,y):
 ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok]
 if len(x)<5 or np.ptp(x)<=1e-12 or np.ptp(y)<=1e-12:return np.nan
 x=x-x.mean();y=y-y.mean();return float(x@y/np.sqrt((x@x)*(y@y)))
s=json.loads((RAW/'cptac_shift_summary.json').read_text());summary=[];features=[];patients=[];weights=[];sources=[RAW/'cptac_shift_summary.json'];checks=[]
for r in s['records']:
 p=RAW/'cptac'/f"{r['key']}.npz";z=np.load(p,allow_pickle=False);sources.append(p)
 base=dict(direction=r['direction'],cohort=r['held_out_cohort'].split('_')[0].upper());y=z['truth'];mask=np.isfinite(y)
 for arm in s['arms']:
  pred=z[arm];assert pred.shape==y.shape and np.isfinite(pred).all();err=pred-y
  rmse=float(np.sqrt(np.nanmean(err**2)));rho=np.array([cor(y[:,j],pred[:,j]) for j in range(y.shape[1])]);valid=np.isfinite(rho);record=r['arms'][arm]
  assert np.isclose(rmse,record['standardized_rmse'],rtol=1e-12,atol=1e-12)
  assert int(valid.sum())==record['valid_r_targets']
  if valid.any():assert np.isclose(rho[valid].mean(),record['mean_target_r'],rtol=1e-12,atol=1e-12)
  checks.append(dict(task=r['key'],arm=arm,rmse_replay=True,correlation_replay=True))
  summary.append(dict(**base,arm=arm,rmse=rmse,mean_r=float(rho[valid].mean()) if valid.any() else None,n_r=int(valid.sum()),n_entries=int(mask.sum()),targets=y.shape[1],patients=y.shape[0],before_rmse=r['uncorrected_arms'][arm]['standardized_rmse']))
  if arm in ['pooled_context','uniform_experts','gated_experts']:
   for j in np.where(valid)[0]:features.append(dict(**base,arm=arm,target_id=z['target_ids'][j],r=float(rho[j]),observed=int(mask[:,j].sum())))
   for i,pid in enumerate(z['patient_ids']):patients.append(dict(**base,arm=arm,patient_id=pid,rmse=float(np.sqrt(np.nanmean(err[i]**2))),bias=float(np.nanmean(err[i])),source_shift=float(z['source_shifts'][i]),source_coverage=float(z['source_coverage'][i]),n=int(mask[i].sum())))
 for j,dom in enumerate(z['domain_ids']):weights.append(dict(**base,expert=dom.split('_')[0].upper(),weight=float(z['weights'][:,j].mean()),patients=y.shape[0]))
 print('CPTAC replay passed',r['key'],flush=True)
save('summary',summary);save('feature_correlations',features);save('patient_metrics',patients);save('expert_weights',weights)
(OUT/'qa').mkdir(exist_ok=True)
(OUT/'qa/CPTAC_RAW_TO_TABLE_RECEIPT.json').write_text(json.dumps(dict(sources=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources],code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checks=checks,score_unit='pooled observed standardized entries; truth and predictions already use training target scaling',before_values='uncorrected_arms from original source-shift summary; not a fresh replay of uncorrected arrays',evidence='three reused development cohorts; not independent GDSC drug-response confirmation'),indent=2))

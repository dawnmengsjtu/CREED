"""Independent new assets/blocks/selection/score verification; no search or refits."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.linalg import solve,eigvalsh
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
p=argparse.ArgumentParser()
for arg in ('run','previous','fm-parent','source','report'):p.add_argument('--'+arg,type=Path,required=True)
a=p.parse_args();start=time.perf_counter();checks=0;banks=[];scores=[];maxsolve=0.;maxview=0.
BASES=('m_only','virtual_fm');GRAPHS=('protein','mask','shuffle','metabolite');MUS=(0.,.1,1.,10.,100.,1000.);ALPHAS=(.1,1.,10.,100.)
OLD=tuple(BASES)+tuple(b+'__'+g for b in BASES for g in GRAPHS);ARMS=tuple(prefix+b for prefix in ('original__','extra__') for b in OLD)
CONFIGS={prefix+b:[[mu,al] for mu in ((0.,) if '__' not in b else MUS) for al in ALPHAS] for prefix in ('original__','extra__') for b in OLD}
META=('fit','held','panel','drug_ids','fit_sample_ids','sample_ids','truth','train_sd','evaluated')
SOURCE='52d71517a8952e83a1603cd831f09449544102d0ca5ff829ff01be9bf79cd637';EXTRA='d4e7dc603f491217805a1d06b673f8c2662a8c0f2ad77ec875f7d7868526384c';CODE='535e4fb7c0d207fb4af7ffb61dd70b34835b40e1abdd433f5cc417179c31b4e6'
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def eq(x,y,atol=1e-10,rtol=1e-10):
    global checks
    np.testing.assert_allclose(x,y,atol=atol,rtol=rtol,equal_nan=True);checks+=1
def exact(x,y):
    global checks
    np.testing.assert_array_equal(x,y);checks+=1
def state(t):
    mean=t.mean(0);sd=t.std(0);sd[sd<1e-8]=1.;x=(t-mean)/sd;d=cdist(x,x,'sqeuclidean');v=d[np.triu_indices(len(t),1)];return mean,sd,np.median(v[v>0])
def rbf(x,y,st):
    mean,sd,width=st;return np.exp(-cdist((x-mean)/sd,(y-mean)/sd,'sqeuclidean')/(2*width))
def inspect(path,old,scenario,f,s,e):
    global maxsolve,maxview
    z=load(path/'geometry.npz');o=load(old/'geometry.npz');inp=load(old/'graph_inputs.npz');prior=load(old/'grids.npz');g=load(path/'grids.npz');rec=js(path/'record.json');fit,query,panel=o['fit'],o['query'],o['panel'];n=len(fit)
    for key in ('fit','query','panel'):exact(z[key],o[key]);exact(z[key],rec[key])
    assert rec['old_geometry_sha256']==sha(old/'geometry.npz') and rec['old_inputs_sha256']==sha(old/'graph_inputs.npz') and rec['old_grids_sha256']==sha(old/'grids.npz')
    good=np.isfinite(e['m']).all(1)&np.isfinite(e['auc']).any(1)
    if scenario=='tissue':
        excluded=np.setdiff1d(np.arange(421),fit);banned=set(s['tissues'][excluded].tolist());good&=np.array([bool(t) and t not in banned for t in e['tissues']])
    ei=np.flatnonzero(good);assert len(ei)>0;exact(ei,z['extra_indices']);exact(ei,rec['extra_indices']);exact(e['sample_ids'][ei],z['extra_sample_ids']);exact(e['sample_ids'][ei],rec['extra_sample_ids']);exact(e['tissues'][ei],rec['extra_tissues']);assert rec['extra_count']==len(ei)
    y=np.concatenate([s['auc'][np.ix_(fit,panel)],e['auc'][np.ix_(ei,panel)]])
    no_new=~np.isfinite(e['auc'][np.ix_(ei,panel)]).any(0);assert rec['no_extra_label_drugs']==int(no_new.sum());exact(rec['extra_observed_response_count'],np.isfinite(e['auc'][np.ix_(ei,panel)]).sum(0))
    gen=a.fm_parent/f'fold-{f}'/path.name if scenario=='tissue' else a.previous/'output/new_cell'/f'fold-{f}'/path.name
    fe=[]
    for j in range(3):
        file=gen/f'generator-{j}.npz';assert rec['generator_sha256'][j]==sha(file);m=load(file);assert str(m['schema'])=='creed.metabolite-query-generator.v1'
        i=int(np.flatnonzero(m['arms']=='virtual_fm')[0]);lo,hi=m['cuts'][i:i+2]
        kg=np.exp(-cdist((e['m'][ei]-m['mean'])/m['sd'],m['x'],'sqeuclidean')/(2*float(m['width'])))
        prediction=kg@m['coef'][:,lo:hi]+m['target_mean'][lo:hi];fe.append(prediction)
    fe=np.array(fe);maxview=max(maxview,float(np.max(abs(fe-z['extra_fm_views']))));eq(fe,z['extra_fm_views'])
    ms=state(s['m'][fit]);em=e['m'][ei];m_eo=rbf(em,s['m'][fit],ms);m_ee=rbf(em,em,ms);m_qe=rbf(s['m'][query],em,ms)
    fs=state(inp['fm_train']);eo=np.mean([rbf(v,inp['fm_train'],fs) for v in fe],0);ee=np.mean([rbf(v,w,fs) for v in fe for w in fe],0);qe=np.mean([rbf(v,w,fs) for v in inp['fm_query_views'] for w in fe],0)
    for base in BASES:
        k,h=z[base+'_kernel'],z[base+'_cross'];assert k.shape==(n+len(ei),n+len(ei)) and h.shape==(len(query),n+len(ei))
        exact(k[:n,:n],o[base+'_kernel']);exact(h[:,:n],o[base+'_cross']);eq(k,k.T);assert eigvalsh(k,subset_by_index=[0,0])[0]>-1e-8
        eo2,ee2,qe2=(m_eo,m_ee,m_qe) if base=='m_only' else (.5*(m_eo+eo),.5*(m_ee+ee),.5*(m_qe+qe))
        eq(k[n:,:n],eo2);eq(k[n:,n:],ee2);eq(h[:,n:],qe2)
        for graph in GRAPHS:
            l=z['L_'+graph];assert l.shape==k.shape;exact(l[:n,:n],o['L_'+graph]);exact(l[n:,:],np.zeros((len(ei),n+len(ei))));exact(l[:,n:],np.zeros((n+len(ei),len(ei))));eq(np.trace(l),np.trace(o['L_'+graph]),1e-10,1e-12)
            exact(g['extra__'+base+'__'+graph][:4],g['extra__'+base])
            if path.name=='outer' and f==0:
                R=np.eye(len(k))+1000.*l@k;ke=solve(R.T,k.T,assume_a='gen').T;he=solve(R.T,h.T,assume_a='gen').T
                for d in sorted(set([0,len(panel)//2,len(panel)-1])):
                    obs=np.flatnonzero(np.isfinite(y[:,d]));b=y[obs,d].mean();pred=np.full(len(query),b)
                    if len(obs)>=20:pred=he[:,obs]@solve(ke[np.ix_(obs,obs)]+np.eye(len(obs)),y[obs,d]-b,assume_a='gen')+b
                    error=float(np.max(abs(pred-g['extra__'+base+'__'+graph][21,:,d])));maxsolve=max(maxsolve,error);eq(pred,g['extra__'+base+'__'+graph][21,:,d],1e-7,1e-7)
    assert set(g)==set(ARMS)
    for arm in OLD:
        exact(g['original__'+arm],prior[arm]);exact(g['extra__'+arm][:,:,no_new],prior[arm][:,:,no_new])
    for arm,values in g.items():assert values.shape==(len(CONFIGS[arm]),len(query),len(panel)) and np.isfinite(values).all()
    banks.append(dict(scenario=scenario,fold=f,bank=path.name,nfit=n,nquery=len(query),extra_count=len(ei),extra_observed_responses=int(np.isfinite(e['auc'][np.ix_(ei,panel)]).sum()),no_extra_label_drugs=int(no_new.sum()),extra_indices=ei.tolist(),grids_sha256=sha(path/'grids.npz'),old_geometry_sha256=rec['old_geometry_sha256']))
    return o,g
assert sha(a.source)==SOURCE and sha(a.run/'code/run_extra.py')==CODE;s=load(a.source);assert len(s['m'])==421
resource=a.run/'resources';assert sha(resource/'m_gdsc_extra19.npz')==EXTRA;e=load(resource/'m_gdsc_extra19.npz');qa=js(resource/'m_gdsc_extra19_QA.json');config=js(resource/'m_gdsc_config.json')
assert qa['status']=='PASS_extraction_QC_and_old_train_parity' and qa['npz_sha256']==EXTRA and qa['config_sha256']==sha(resource/'m_gdsc_config.json') and qa['protected_numeric_rows_decoded']==0
assert qa['rows']==19 and e['m'].shape==(19,225) and e['auc'].shape==(19,295) and np.isfinite(e['m']).all() and np.isfinite(e['auc']).sum()==4485
exact(e['m_feature_ids'],s['metabolite_ids']);exact(e['drug_ids'],s['drug_ids']);exact(e['m_feature_ids'],config['m_feature_ids']);exact(e['drug_ids'],config['drug_ids'])
assert not(set(e['sample_ids'])&set(s['sample_ids'])) and not(set(e['sample_ids'])&set(config['excluded_old608']))
exact(e['sample_ids'],[c['sample_id'] for c in config['candidates']]);exact(e['model_ids'],[c['model_id'] for c in config['candidates']]);exact(e['patient_ids'],[c['patient_id'] for c in config['candidates']]);exact(e['ccle_ids'],[c['ccle_id'] for c in config['candidates']]);exact(e['tissues'],[c['tissue'] for c in config['candidates']])
obs=np.isfinite(e['auc']);assert np.isfinite(e['ln_ic50'][obs]).all() and np.isfinite(e['min_conc'][obs]).all() and np.isfinite(e['max_conc'][obs]).all();assert (e['min_conc'][obs]>0).all() and (e['min_conc'][obs]<=e['max_conc'][obs]).all()
for j,sid in enumerate(e['parity_sample_ids']):
    i=int(np.flatnonzero(s['sample_ids']==sid)[0]);exact(e['parity_m'][j],s['m'][i]);exact(e['parity_auc'][j],s['auc'][i])
assert js(a.previous/'verification/saved_results_v2_verification.json')['status']=='PASS'
for scenario in ('tissue','new_cell'):
    out=a.run/'output'/scenario;oldout=a.previous/'output'/(scenario+'_v2');protocol=js(out/'protocol.json');op=js(oldout/'protocol.json')
    assert protocol['source_sha256']==SOURCE and protocol['extra_sha256']==EXTRA and protocol['script_sha256']==CODE and protocol['helper_sha256']==sha(a.run/'code/run_recoverability.py')==sha(a.previous/'code/run_recoverability.py') and protocol['configs']==CONFIGS
    assert protocol['candidate_only'] and not protocol['external_confirmation'];exact(protocol['outer_fold_ids'],op['outer_fold_ids']);outer=np.array(protocol['outer_fold_ids']);seen=[]
    foldrows={(int(r['fold']),r['arm']):r for r in csv.DictReader((out/'fold_metrics.csv').open())};assert len(foldrows)==100
    drugrows={(int(r['fold']),r['arm'],r['drug_id']):r for r in csv.DictReader((out/'per_drug_metrics.csv').open())}
    for f in range(5):
        path=out/f'fold-{f}';old=oldout/f'fold-{f}';saved=js(path/'selection.json');meta=js(old/'selection.json');assert saved['configs']==CONFIGS
        for key in ('fit','held','panel','inner_ids'):exact(saved[key],meta[key])
        fit=np.flatnonzero(outer!=f);held=np.flatnonzero(outer==f);seen.extend(held);panel=np.array(meta['panel']);inner=np.array(meta['inner_ids']);exact(fit,meta['fit']);exact(held,meta['held']);totals={};counts={}
        for j in range(3):
            z,g=inspect(path/f'inner-{j}',old/f'inner-{j}',scenario,f,s,e);exact(z['fit'],fit[inner!=j]);exact(z['query'],fit[inner==j]);exact(z['panel'],panel)
            truth=s['auc'][np.ix_(z['query'],panel)];sd=np.maximum(np.nanstd(s['auc'][np.ix_(z['fit'],panel)],0),1e-8);valid=np.isfinite(truth);n=valid.sum(0)
            for arm,grid in g.items():
                ll=np.where(valid[None],((grid-truth[None])/sd)**2,0.).sum(1)/np.maximum(n,1)[None];nn=np.tile(n>0,(len(grid),1)).astype(int);totals[arm]=totals.get(arm,0)+ll;counts[arm]=counts.get(arm,0)+nn
        selected={}
        for arm in ARMS:
            eq(totals[arm],saved['losses'][arm]);exact(counts[arm],saved['counts'][arm]);avg=np.divide(totals[arm],counts[arm],out=np.full_like(totals[arm],np.inf),where=counts[arm]>0);ix=avg.argmin(0);ix[np.all(counts[arm]==0,0)]=1;exact(ix,saved['selected'][arm]);selected[arm]=ix
        z,g=inspect(path/'outer',old/'outer',scenario,f,s,e);exact(z['fit'],fit);exact(z['query'],held);exact(z['panel'],panel);pred=load(path/'predictions.npz');prior=load(old/'predictions.npz')
        for key in META:exact(pred[key],prior[key])
        truth=s['auc'][np.ix_(held,panel)];sd=np.maximum(np.nanstd(s['auc'][np.ix_(fit,panel)],0),1e-8);valid=np.isfinite(truth).sum(0)>=5;exact(pred['truth'],truth);eq(pred['train_sd'],sd);exact(pred['evaluated'],valid)
        for arm in ARMS:exact(pred[arm],g[arm][selected[arm],:,np.arange(len(panel))].T)
        for arm in OLD:exact(pred['original__'+arm],prior[arm])
        for arm in ARMS:
            mses=[];centered=[];rhos=[]
            for j,drug in enumerate(pred['drug_ids']):
                ok=np.isfinite(truth[:,j]);x=truth[ok,j];y=pred[arm][ok,j];err=(y-x)/sd[j];ms=float(np.mean(err**2)) if len(x) else 0.;rho=float(spearmanr(x,y)[0]) if valid[j] and np.std(x)>0 and np.std(y)>1e-15 else float('nan')
                row=drugrows[(f,arm,str(drug))];assert int(row['count'])==len(x) and (row['evaluated']=='True')==bool(valid[j]);eq(ms,float(row['standardized_mse']));eq(rho,float(row['spearman']))
                if valid[j]:
                    mses.append(ms);centered.append(float(np.mean((err-err.mean())**2)))
                    if np.isfinite(rho):rhos.append(rho)
            score=dict(scenario=scenario,fold=f,arm=arm,score=float(np.sqrt(np.mean(mses))),spearman=float(np.mean(rhos)),centered_mse=float(np.mean(centered)));row=foldrows[(f,arm)];assert int(row['drugs'])==int(valid.sum())
            for key in ('score','spearman','centered_mse'):eq(score[key],float(row[key]))
            scores.append(score)
        assert js(path/'status.json')['status']=='completed';print(json.dumps(dict(event='verified_fold',scenario=scenario,fold=f,checks=checks)),flush=True)
    assert sorted(seen)==list(range(421));summary=js(out/'summary.json');assert summary['status']=='completed'
    for arm,row in summary['summary'].items():
        rows=[r for r in scores if r['scenario']==scenario and r['arm']==arm];assert len(rows)==5
        for key in ('score','spearman','centered_mse'):eq(np.mean([r[key] for r in rows]),row[key])
result=dict(status='PASS',checks=checks,banks=len(banks),fold_metric_rows=len(scores),seconds=time.perf_counter()-start,max_extra_generator_replay_error=maxview,max_fixed_real_solve_error=maxsolve,source_sha256=SOURCE,extra_sha256=EXTRA,qa_sha256=sha(resource/'m_gdsc_extra19_QA.json'),code_sha256=CODE,verifier_sha256=sha(Path(__file__)),bank_records=banks,independently_recomputed_scores=scores,limitations=['Upstream candidate related-identity exclusion relies on frozen whitelist and resource QA; whitelist/IDs/old608 disjointness/QC/parity are checked here without rescanning protected data.','All new kernel blocks and extra generator views recomputed; downstream numerical refits limited to three fixed drugs at mu1000 alpha1 in fold0 outer of both scenarios, both bases, all four graphs.','Development evidence; original outer folds have already informed earlier work, so this is not external confirmation.'])
a.report.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('bank_records','independently_recomputed_scores')},indent=2))

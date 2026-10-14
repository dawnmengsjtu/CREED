"""Independent bounded verification for strong-teacher recoverability.

Replays saved model-selection grids and all final scores, checks the nested
fit/query graph and old prediction parity, and optionally performs ONE synthetic
masked-solver parity test. Does not refit real-data teachers or train new models.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

EXPECTED_SOURCE_SHA='52d71517a8952e83a1603cd831f09449544102d0ca5ff829ff01be9bf79cd637'
ALPHAS=(.1,1.,10.,100.)
BETAS=(0.,.25,.5,1.)
TEACHERS=('m_only','measured_p','measured_mask')
TARGETS=('protein_delta','mask_delta','value_delta','shuffled_delta','response_residual')
CONFIGS=tuple((k,a,b) for b in BETAS for k in ('rbf','linear') for a in ALPHAS)
RECOVERY_CONFIGS=tuple((k,a) for k in ('rbf','linear') for a in ALPHAS)
ARMS=TEACHERS+tuple('transfer_'+t for t in TARGETS)+('delta_fit_diagnostic',)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def read_npz(path):
    with np.load(path,allow_pickle=False) as d:return {k:d[k] for k in d.files}

def read_json(path):return json.loads(Path(path).read_text())
def read_csv(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def eq(actual,expected,label,atol=3e-11,rtol=3e-11):
    try:np.testing.assert_allclose(np.asarray(actual,float),np.asarray(expected,float),atol=atol,rtol=rtol,equal_nan=True)
    except AssertionError as e:raise AssertionError(label+': '+str(e)[:900]) from e

def same(actual,expected,label):
    if not np.array_equal(actual,expected):raise AssertionError(label+' differs')

def ids(x,label):
    a=np.asarray(x)
    if a.ndim!=1 or not np.issubdtype(a.dtype,np.integer) or len(a)!=len(set(a.tolist())) or (len(a) and (a.min()<0 or a.max()>=421)):
        raise AssertionError(label+' invalid sample indices')
    return a.astype(int)

def seed(*parts):return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'little')

def split_labels(tissues,n,scenario,outer=False):
    labels=np.asarray(tissues).astype(str);counts=Counter(labels.tolist())
    if scenario=='tissue':
        loads=[0]*n;mapping={}
        if len(counts)<n:raise AssertionError('too few tissues for nested grouping')
        for t in sorted(counts,key=lambda t:(-counts[t],t)):
            f=min(range(n),key=lambda f:(loads[f],f));mapping[t]=f;loads[f]+=counts[t]
        return np.array([mapping[t] for t in labels])
    result=np.full(len(labels),-1,int);loads=np.zeros(n,int)
    for t in sorted(counts,key=lambda t:(-counts[t],t)):
        ix=np.flatnonzero(labels==t);rng=np.random.default_rng(seed(20261008,'new_cell_folds' if outer else 'nested_cell_folds',t));rng.shuffle(ix)
        order=rng.permutation(n);order=order[np.argsort(loads[order],kind='stable')]
        for i,row in enumerate(ix):f=order[i%n];result[row]=f;loads[f]+=1
    return result

def loss_grid(pred,truth,scale):
    # Independent per-drug implementation; denominator is observed labels in
    # each validation fold, followed by equal averaging of supported folds.
    values=np.zeros((len(pred),truth.shape[1]));counts=np.zeros_like(values)
    for j in range(truth.shape[1]):
        obs=np.isfinite(truth[:,j])
        if obs.any():
            values[:,j]=np.mean(((pred[:,obs,j]-truth[obs,j])/scale[j])**2,axis=1);counts[:,j]=1
    return values,counts

def selected(total,count,fallback):
    score=np.divide(total,count,out=np.full_like(total,np.inf),where=count>0)
    r=np.argmin(score,axis=0);r[np.all(count==0,axis=0)]=fallback
    return r

def pick(grid,ix):return np.stack([grid[int(c),:,j] for j,c in enumerate(ix)],axis=1)

def scores(pred,truth,scale,use):
    mse=[];bias=[];rho=[];counts=[]
    for j in range(truth.shape[1]):
        obs=np.isfinite(truth[:,j]);counts.append(int(obs.sum()))
        if obs.any():
            e=(pred[obs,j]-truth[obs,j])/scale[j];mse.append(float(np.mean(e*e)));bias.append(float(np.mean(e)))
        else:mse.append(0.);bias.append(float('nan'))
        rho.append(float(spearmanr(pred[obs,j],truth[obs,j]).statistic) if use[j] and np.std(truth[obs,j])>0 and np.std(pred[obs,j])>1e-15 else float('nan'))
    mse=np.array(mse);bias=np.array(bias);rho=np.array(rho)
    return dict(score=float(np.sqrt(np.mean(mse[use]))),spearman=float(np.nanmean(rho[use])),centered_mse=float(np.mean((mse-bias*bias)[use])),mse=mse,bias_squared=bias*bias,rho=rho,count=np.array(counts))

def nested_graph(folder,source,panel,fit,held,scenario):
    traces=read_json(folder/'fit_trace.json');raw={};strong={};numeric=0;fixed=0;protein_fallbacks=[]
    for t in traces:
        a=ids(t['fit'],'trace fit');q=ids(t['query'],'trace query');key=(tuple(a),tuple(q))
        assert not set(a)&set(q)
        assert set(a)<=set(fit)
        assert set(q)<=set(fit) or (np.array_equal(a,fit) and np.array_equal(q,held))
        if scenario=='tissue':assert not set(source['tissues'][a])&set(source['tissues'][q])
        dest=raw if t['kind']=='raw' else strong
        assert key not in dest;dest[key]=t
        if t['kind']=='raw':
            assert t['protein_count']>0
            assert t['low_support_drugs']==int((np.isfinite(source['auc'][np.ix_(a,panel)]).sum(0)<20).sum())
            if t.get('protein_support_fallback',False):
                assert len(np.unique(source['tissues'][a]))<3 and not t.get('original_bank')
                pp=source['p'][a];observed=np.isfinite(pp).sum(0)
                with np.errstate(invalid='ignore',divide='ignore'):
                    mean=np.divide(np.nansum(pp,0),observed,out=np.zeros(pp.shape[1]),where=observed>0)
                    variance=np.divide(np.nansum((pp-mean)**2,0),observed,out=np.zeros(pp.shape[1]),where=observed>0)
                pi=np.flatnonzero((observed>=20)&(2*observed>=len(a))&(variance>1e-16))
                assert len(pi)==t['protein_count']
                protein_fallbacks.append(dict(fit=a.tolist(),query=q.tolist(),tissues=sorted(set(source['tissues'][a].tolist())),protein_indices=pi.tolist(),protein_count=len(pi)))
        else:
            for arm in TEACHERS:
                z=np.asarray(t['selected'][arm]);assert z.shape==(len(panel),) and np.all((z>=0)&(z<4))
            if t['kind']=='strong_fixed_alpha_support_fallback':
                assert scenario=='tissue' and len(np.unique(source['tissues'][a]))<3
                assert t['alpha']==1.0 and 'inner_ids' not in t and 'selection_file' not in t
                for arm in TEACHERS:same(t['selected'][arm],np.ones(len(panel),int),'fixed fallback alpha')
                fixed+=1
                continue
            assert t['kind']=='strong'
            same(t['inner_ids'],split_labels(source['tissues'][a],3,scenario),'teacher inner ids')
            if 'selection_file' in t:
                selection_path=Path(t['selection_file']);assert not selection_path.is_absolute() and '..' not in selection_path.parts
                z=read_npz(folder/selection_path)
                for key,value in [('fit',a),('query',q),('inner_ids',t['inner_ids'])]:same(z[key],value,'teacher selection '+key)
                for arm in TEACHERS:
                    loss=z[arm+'_loss'];count=z[arm+'_count']
                    assert loss.shape==count.shape==(4,len(panel)) and np.isfinite(loss).all() and np.all((count>=0)&(count<=3))
                    same(z[arm+'_selected'],t['selected'][arm],'deep teacher saved selection')
                    same(t['selected'][arm],selected(loss,count,1),'deep teacher alpha selection')
                numeric+=1
    for (aa,qq),t in strong.items():
        assert (aa,qq) in raw
        if t['kind']=='strong_fixed_alpha_support_fallback':continue
        a=np.array(aa);f=np.asarray(t['inner_ids'])
        for c in range(3):assert (tuple(a[f!=c]),tuple(a[f==c])) in raw
    assert len(raw)==read_json(folder/'status.json')['raw_fits']
    assert len(strong)==read_json(folder/'status.json')['strong_fits']
    return raw,strong,dict(raw_fits=len(raw),strong_fits=len(strong),teacher_selection_numeric_replays=numeric,fixed_alpha_support_fallbacks=fixed,protein_support_fallbacks=protein_fallbacks,all_teacher_selection_scores_saved=numeric+fixed==len(strong))

def bank(folder,name,source,panel,a,q,scenario,fold,strong):
    c=read_npz(folder/name/'contract.npz');g=read_npz(folder/name/'student_grids.npz')
    same(c['fit'],a,'bank fit');same(c['query'],q,'bank query');same(c['panel'],panel,'bank panel')
    f=split_labels(source['tissues'][a],3,scenario);same(c['oof_fold'],f,'student OOF ids')
    assert (tuple(a),tuple(q)) in strong
    perm=c['permutation'];same(np.sort(perm),np.arange(len(a)),'shuffle bijection');same(f[perm],f,'shuffle stays inside teacher query block')
    expected=np.arange(len(a))
    for k in range(3):
        pos=np.flatnonzero(f==k);assert (tuple(a[f!=k]),tuple(a[pos])) in strong
        # The pinned training image uses NumPy 1.x integer tuple repr. Convert
        # explicitly so verification stays identical under local NumPy 2.x.
        rng=np.random.default_rng(seed('delta_shuffle',scenario,fold,tuple(map(int,a)),k));expected[pos]=rng.permutation(pos)
    same(perm,expected,'fixed identity permutation')
    delta=c['oof_measured_p']-c['oof_m_only']
    targets=dict(protein_delta=delta,mask_delta=c['oof_measured_mask']-c['oof_m_only'],value_delta=c['oof_measured_p']-c['oof_measured_mask'],shuffled_delta=delta[perm],response_residual=source['auc'][np.ix_(a,panel)]-c['oof_m_only'])
    for t in TARGETS:
        eq(c['target_'+t],targets[t],'teacher target '+t)
        for k in ('rbf','linear'):
            assert g[t+'__'+k].shape==(4,len(q),len(panel)) and np.isfinite(g[t+'__'+k]).all()
    for arm in TEACHERS:
        assert c['oof_'+arm].shape==(len(a),len(panel)) and np.isfinite(c['oof_'+arm]).all()
        assert c['query_'+arm].shape==(len(q),len(panel)) and np.isfinite(c['query_'+arm]).all()
    return c,g

def verify_fold(root,parent,source,scenario,fold,outer_ids,fold_rows,drug_rows,recovery_rows):
    folder=root/f'fold-{fold}';p=read_npz(folder/'predictions.npz');a=np.flatnonzero(outer_ids!=fold);q=np.flatnonzero(outer_ids==fold)
    same(p['fit'],a,'outer fit');same(p['held'],q,'outer held');panel=p['panel']
    assert len(set(panel.tolist()))==len(panel) and np.all((panel>=0)&(panel<source['auc'].shape[1]))
    same(p['drug_ids'],source['drug_ids'][panel],'drug IDs')
    sample_ids_explicit='sample_ids' in p and 'fit_sample_ids' in p
    if 'sample_ids' in p:same(p['sample_ids'],source['sample_ids'][q],'held sample IDs')
    if 'fit_sample_ids' in p:same(p['fit_sample_ids'],source['sample_ids'][a],'fit sample IDs')
    if scenario=='new_cell':same(panel,np.arange(295),'new-cell full drug panel')
    parity={}
    if scenario=='tissue':
        old=read_npz(parent/f'fold-{fold}'/'predictions.npz')
        for k in ('fit','held','panel','drug_ids'):same(p[k],old[k],'old '+k)
        for arm in TEACHERS:
            same(p[arm],old[arm],'exact old frozen '+arm);parity[arm]=True
        assert read_json(folder/'original_parity.json')['status']=='PASS'
    raw,strong,trace_report=nested_graph(folder,source,panel,a,q,scenario)
    # Original-bank paths are checked via their final fold/bank components so
    # container mount names cannot disguise an unrelated source.
    for t in raw.values():
        if t.get('original_bank'):
            b=Path(t['original_bank']);meta=read_json(parent/b.parent.name/b.name/'selection.json')
            same(meta['fit'],t['fit'],'old raw-bank fit');same(meta['query'],t['query'],'old raw-bank query');same(meta['panel'],panel,'old raw-bank panel')
    choice=read_json(folder/'selection.json');assert choice['configs']==[list(x) for x in CONFIGS]
    assert choice['recovery_configs']==[list(x) for x in RECOVERY_CONFIGS]
    total={t:np.zeros((32,len(panel))) for t in TARGETS};count={t:np.zeros_like(total[t]) for t in TARGETS}
    rt=np.zeros((8,len(panel)));rn=np.zeros_like(rt);inner=split_labels(source['tissues'][a],3,scenario)
    for k in range(3):
        aa=a[inner!=k];vv=a[inner==k];c,g=bank(folder,f'inner-{k}',source,panel,aa,vv,scenario,fold,strong)
        scale=np.maximum(np.nanstd(source['auc'][np.ix_(aa,panel)],axis=0),1e-8);truth=source['auc'][np.ix_(vv,panel)]
        for t in TARGETS:
            pred=np.stack([c['query_m_only']+beta*g[t+'__'+kind][ALPHAS.index(alpha)] for kind,alpha,beta in CONFIGS]);l,n=loss_grid(pred,truth,scale);total[t]+=l;count[t]+=n
        pred=np.concatenate([g['protein_delta__'+kind] for kind in ('rbf','linear')]);l,n=loss_grid(pred,c['query_measured_p']-c['query_m_only'],scale);rt+=l;rn+=n
    for t in TARGETS:
        eq(choice['losses'][t],total[t],'student selection losses');eq(choice['counts'][t],count[t],'student selection counts')
        same(choice['selected'][t],selected(total[t],count[t],0),'functional selected config')
    eq(choice['recovery_losses'],rt,'recovery selection losses');eq(choice['recovery_counts'],rn,'recovery selection counts');same(choice['recovery_selected'],selected(rt,rn,1),'recovery selected config')
    c,g=bank(folder,'outer',source,panel,a,q,scenario,fold,strong)
    for arm in TEACHERS:eq(c['query_'+arm],p[arm],'outer teacher replay before exact copy')
    for t in TARGETS:
        grid=np.stack([beta*g[t+'__'+kind][ALPHAS.index(alpha)] for kind,alpha,beta in CONFIGS]);pred=p['m_only']+pick(grid,choice['selected'][t]);eq(p['transfer_'+t],pred,'selected outer prediction '+t)
    rec=pick(np.concatenate([g['protein_delta__'+k] for k in ('rbf','linear')]),choice['recovery_selected'])
    eq(p['predicted_delta'],rec,'selected recovery prediction');eq(p['delta_fit_diagnostic'],p['m_only']+rec,'diagnostic assembled prediction')
    delta=p['measured_p']-p['m_only'];mu=c['target_protein_delta'].mean(0)
    eq(p['teacher_delta'],delta,'teacher delta');eq(p['train_delta_mean'],mu,'train-only delta mean')
    truth=source['auc'][np.ix_(q,panel)];sd=np.maximum(np.nanstd(source['auc'][np.ix_(a,panel)],axis=0),1e-8);use=np.isfinite(truth).sum(0)>=5
    eq(p['truth'],truth,'bound truth');eq(p['train_sd'],sd,'training-only evaluation scale');same(p['evaluated'],use,'evaluation mask')
    summary={}
    for arm in ARMS:
        assert p[arm].shape==truth.shape and np.isfinite(p[arm]).all()
        sc=scores(p[arm],truth,sd,use);summary[arm]={k:sc[k] for k in ('score','spearman','centered_mse')};row=fold_rows[(fold,arm)]
        for k in summary[arm]:eq(float(row[k]),sc[k],'Y '+k)
        assert int(row['drugs'])==int(use.sum())
        for j,drug in enumerate(p['drug_ids']):
            d=drug_rows[(fold,arm,str(drug))];eq(float(d['standardized_mse']),sc['mse'][j],'per-drug MSE');eq(float(d['spearman']),sc['rho'][j],'per-drug rho');assert int(d['count'])==sc['count'][j];assert (d['evaluated'].lower()=='true')==bool(use[j])
    numerator=np.mean(((rec-delta)/sd)**2,axis=0);zero=np.mean((delta/sd)**2,axis=0);mean=np.mean(((delta-mu)/sd)**2,axis=0)
    rz=read_npz(folder/'recoverability.npz')
    for key,x in [('num',numerator),('zero',zero),('train_mean',mean)]:eq(rz[key],x,'delta '+key)
    rr=recovery_rows[fold];recover=dict(delta_mse=float(numerator[use].mean()),zero_mse=float(zero[use].mean()),train_mean_mse=float(mean[use].mean()),skill_vs_zero=float(1-numerator[use].mean()/zero[use].mean()),skill_vs_train_mean=float(1-numerator[use].mean()/mean[use].mean()))
    for k,v in recover.items():eq(float(rr[k]),v,'recovery ratio '+k)
    key='teacher_centered_mse_gain' if 'teacher_centered_mse_gain' in rr else 'teacher_risk_gain'
    eq(float(rr[key]),summary['m_only']['centered_mse']-summary['measured_p']['centered_mse'],'teacher centered risk gain')
    return dict(fold=fold,status='pass',samples=len(q),drugs=len(panel),explicit_sample_ids=sample_ids_explicit,old_outer_exact_parity=parity,trace=trace_report,functional_selection_replayed=True,recovery_selection_replayed=True,summary=summary,recovery=recover)

def solver_parity(runner):
    from creed.science.masked_task_kernel import predict_masked
    spec=importlib.util.spec_from_file_location('recoverability_runner_under_test',runner);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rng=np.random.default_rng(61008);x=rng.normal(size=(37,6));q=rng.normal(size=(11,6));k=np.exp(-cdist(x,x,'sqeuclidean')/10);h=np.exp(-cdist(q,x,'sqeuclidean')/10);y=rng.normal(size=(37,5));y[:5,1]=np.nan;y[:5,2]=np.nan;y[:12,3]=np.nan;y[:25,4]=np.nan
    actual=module.predict_grid(k,h,y);diff=[]
    for ai,a in enumerate(ALPHAS):
        expected=predict_masked(k,h,y,a);eq(actual[ai,:,:4],expected[:,:4],'old supported solver parity',atol=2e-12,rtol=2e-12);diff.append(float(np.max(np.abs(actual[ai,:,:4]-expected[:,:4]))));eq(actual[ai,:,4],np.nanmean(y[:,4]),'explicit low-support mean fallback')
    return dict(status='pass',max_abs_supported_difference=max(diff),arrays={'fit':37,'query':11,'outputs':5},old_low_support_nan_vs_new_fit_mean_explicit=True,real_data_refits=0)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--parent',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);ap.add_argument('--runner',type=Path,default=Path(__file__).with_name('run_recoverability.py'));ap.add_argument('--scenarios',nargs='+',default=['tissue','new_cell']);ap.add_argument('--scenario-dir',type=Path,help='Explicit directory for one scenario/version');ap.add_argument('--solver-parity',action='store_true');ap.add_argument('--allow-partial',action='store_true');args=ap.parse_args()
    report=dict(status='running',scope='saved selection replay, complete score recomputation, nested dependency trace, frozen outer parity; no real-data model refitting',errors=[],scenarios={})
    try:
        path=args.parent/'source-inputs.npz';assert digest(path)==EXPECTED_SOURCE_SHA;s=read_npz(path);assert s['m'].shape==(421,225) and s['p'].shape==(421,6692) and s['auc'].shape==(421,295);assert len(set(s['sample_ids'].tolist()))==421;assert len(set(s['drug_ids'].tolist()))==295
        report['source_sha256']=EXPECTED_SOURCE_SHA;report['runner_sha256']=digest(args.runner)
        assert args.scenario_dir is None or len(args.scenarios)==1
        if args.solver_parity:report['synthetic_solver_parity']=solver_parity(args.runner)
        for scenario in args.scenarios:
            root=args.scenario_dir if args.scenario_dir is not None else args.output/scenario
            if not (root/'protocol.json').exists():report['scenarios'][scenario]=dict(status='incomplete',reason='protocol absent');continue
            protocol=read_json(root/'protocol.json');assert protocol['source_sha256']==EXPECTED_SOURCE_SHA;assert protocol['script_sha256']==report['runner_sha256'];assert protocol['scenario']==scenario
            foldids=s['fold_ids'] if scenario=='tissue' else split_labels(s['tissues'],5,scenario,outer=True);same(protocol['outer_fold_ids'],foldids,'frozen scenario folds')
            rows=read_csv(root/'fold_metrics.csv') if (root/'fold_metrics.csv').exists() else [];fold_rows={(int(r['fold']),r['arm']):r for r in rows};assert len(fold_rows)==len(rows)
            rows=read_csv(root/'per_drug_metrics.csv') if (root/'per_drug_metrics.csv').exists() else [];drug_rows={(int(r['fold']),r['arm'],r['drug_id']):r for r in rows};assert len(drug_rows)==len(rows)
            rows=read_csv(root/'recoverability.csv') if (root/'recoverability.csv').exists() else [];recovery_rows={int(r['fold']):r for r in rows};assert len(recovery_rows)==len(rows)
            folds=[]
            for f in range(5):
                if not (root/f'fold-{f}'/'status.json').exists() or (f,'m_only') not in fold_rows:continue
                folds.append(verify_fold(root,args.parent,s,scenario,f,foldids,fold_rows,drug_rows,recovery_rows))
            state='pass' if len(folds)==5 and all(r['trace']['all_teacher_selection_scores_saved'] for r in folds) else 'incomplete'
            if len(folds)==5:
                assert sum(r['samples'] for r in folds)==421;summary=read_json(root/'summary.json');assert summary['status']=='completed' and summary['folds']==5
                for arm in ARMS:
                    for k in ('score','spearman','centered_mse'):eq(summary['summary'][arm][k],np.mean([r['summary'][arm][k] for r in folds]),'scenario summary '+arm+' '+k)
            report['scenarios'][scenario]=dict(status=state,completed_folds=len(folds),folds=folds)
        report['status']='pass' if all(v['status']=='pass' for v in report['scenarios'].values()) else 'incomplete'
    except Exception as e:
        report['status']='fail';report['errors'].append(type(e).__name__+': '+str(e))
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'status':report['status'],'report':str(args.report),'errors':report['errors'],'scenarios':{k:{'status':v['status'],'completed_folds':v.get('completed_folds',0)} for k,v in report['scenarios'].items()}}))
    if report['status']=='fail' or (report['status']=='incomplete' and not args.allow_partial):raise SystemExit(1)

if __name__=='__main__':main()

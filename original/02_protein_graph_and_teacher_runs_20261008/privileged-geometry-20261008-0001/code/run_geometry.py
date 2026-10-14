"""Privileged training-sample geometry with unchanged M-only query interfaces."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.linalg import eigh, cho_factor, cho_solve
from scipy.spatial.distance import cdist
from creed.science.metabolite_query import build_bank, group_folds
from creed.science.metabolite_protein_query import panel_for_bank
from run_recoverability import (load, sha, save_json, csv_save, kernel, predict_grid,
    metrics, losses, choose, take, cell_folds, ALPHAS, SOURCE_SHA)

MUS=(0.,.1,1.,10.)
CONFIGS=tuple((mu,alpha) for mu in MUS for alpha in ALPHAS)
GRAPHS=('protein','mask','shuffle','metabolite')
BASES=('m_only','virtual_fm')


def laplacian(x):
    mu=x.mean(0);sd=x.std(0);sd=np.where(sd<1e-8,1.,sd)
    z=(x-mu)/sd;d=cdist(z,z,'sqeuclidean')
    v=d[np.triu_indices(len(x),1)];v=v[v>0]
    width=float(np.median(v)) if len(v) else 1.
    order_d=d.copy();np.fill_diagonal(order_d,np.inf)
    neighbors=np.argsort(order_d,axis=1,kind='stable')[:,:min(10,len(x)-1)]
    w=np.zeros_like(d);rows=np.arange(len(x))[:,None]
    w[rows,neighbors]=np.exp(-d[rows,neighbors]/(2*width));w=np.maximum(w,w.T)
    l=np.diag(w.sum(1))-w;l*=len(x)/np.trace(l)
    assert np.max(np.abs(l.sum(1)))<1e-10
    return l,dict(mean=mu,sd=sd,width=width,neighbors=neighbors)


def effective(k,h,l,mu):
    if mu==0:return k,h
    vals,vecs=eigh((k+k.T)/2)
    if vals.min() < -1e-8:raise ValueError('indefinite original kernel')
    root=(vecs*np.sqrt(np.maximum(vals,0)))@vecs.T
    a=np.eye(len(k))+mu*root@l@root;a=(a+a.T)/2
    ke=root@cho_solve(cho_factor(a,lower=True),root);ke=(ke+ke.T)/2
    he=h-mu*h@l@ke
    assert np.isfinite(ke).all() and np.isfinite(he).all()
    return ke,he


def extra_kernel(t,queries):
    mu=t.mean(0);sd=t.std(0);sd=np.where(sd<1e-8,1.,sd)
    z=(t-mu)/sd;qs=(queries-mu)/sd
    d=cdist(z,z,'sqeuclidean');v=d[np.triu_indices(len(t),1)];v=v[v>0]
    width=np.median(v) if len(v) else 1.
    k=np.exp(-d/(2*width))
    # Averaging query kernels is equivalent to averaging predictions, because
    # each generator shares the identical task-training kernel and coefficients.
    hs=np.array([np.exp(-cdist(q,z,'sqeuclidean')/(2*width)) for q in qs])
    return k,hs.mean(0)


def run_bank(s,fs,args,f,name,a,q,panel,dest):
    start=time.perf_counter();dest.mkdir(parents=True)
    assert not set(a)&set(q)
    if args.scenario=='tissue':assert not set(s['tissues'][a])&set(s['tissues'][q])
    mk,mh=kernel(s['m'][a],s['m'][q])
    if args.scenario=='tissue':
        old=args.fm_parent/f'fold-{f}'/name
        meta=json.loads((old/'selection.json').read_text())
        np.testing.assert_array_equal(meta['fit'],a);np.testing.assert_array_equal(meta['query'],q)
        t=load(old/'train-features.npz')['virtual_fm']
        qs=load(old/'query-features.npz')['virtual_fm']
        fm_grids=load(old/'per-alpha-predictions.npz')
        pi=load(args.parent/f'fold-{f}'/name/'panel.npz')['indices']
        provenance=dict(reused_fm_bank=str(old),train_sha=sha(old/'train-features.npz'),
            query_sha=sha(old/'query-features.npz'),split_sha=sha(old/'selection.json'))
    else:
        bank=build_bank(s['m'][a],fs['rna_repr'][a],fs['fm'][a],fs['m_repr'][a],s['tissues'][a],s['m'][q])
        t=bank['train_features']['virtual_fm'];qs=bank['query_features']['virtual_fm']
        for g,model in enumerate(bank['models']):np.savez_compressed(dest/f'generator-{g}.npz',**model)
        provenance=dict(generator_splits=bank['generator_splits'])
        pi=panel_for_bank(s['p'][a],s['tissues'][a])['indices']
    assert len(pi)>0
    pk=s['p'][np.ix_(a,pi)];obs=np.isfinite(pk);means=np.nanmean(pk,axis=0)
    filled=np.where(obs,pk,means)
    lp,pstate=laplacian(filled);lm,mstate=laplacian(obs.astype(float));lx,xstate=laplacian(s['m'][a])
    rng=np.random.default_rng(2026100809);perm=np.arange(len(a))
    for tissue in sorted(set(s['tissues'][a].tolist())):
        pos=np.flatnonzero(s['tissues'][a]==tissue);perm[pos]=rng.permutation(pos)
    ls=lp[np.ix_(perm,perm)]
    graphs=dict(protein=lp,mask=lm,shuffle=ls,metabolite=lx)
    fk,fh=extra_kernel(t,qs)
    bases=dict(m_only=(mk,mh),virtual_fm=(.5*mk+.5*fk,.5*mh+.5*fh))
    y=s['auc'][np.ix_(a,panel)];grids={};parity={};geometry={}
    for base,(k,h) in bases.items():
        grid0=predict_grid(k,h,y);grids[base]=grid0
        if args.scenario=='tissue':
            expected=fm_grids[base]
            np.testing.assert_allclose(grid0,expected,rtol=1e-10,atol=1e-11)
            parity[base]=float(np.max(np.abs(grid0-expected)))
            grids[base]=expected
        geometry[base+'_kernel']=k;geometry[base+'_cross']=h
        for graph,l in graphs.items():
            vals=[grids[base]]
            for mu in MUS[1:]:
                ke,he=effective(k,h,l,mu);vals.append(predict_grid(ke,he,y))
            grids[base+'__'+graph]=np.concatenate(vals)
    np.savez_compressed(dest/'geometry.npz',fit=a,query=q,panel=panel,protein_indices=pi,
        permutation=perm,protein_fill_mean=means,**{'L_'+k:v for k,v in graphs.items()},**geometry)
    np.savez_compressed(dest/'graph_inputs.npz',protein=filled,mask=obs,metabolite=s['m'][a],
        fm_train=t,fm_query_views=qs)
    np.savez_compressed(dest/'grids.npz',**grids)
    save_json(dest/'record.json',dict(fold=f,bank=name,fit=a,query=q,panel=panel,
        fit_ids=s['sample_ids'][a],query_ids=s['sample_ids'][q],protein_indices=pi,
        graph_fit_only=True,query_input='metabolite only',provenance=provenance,
        old_alpha_grid_parity_max=parity,seconds=time.perf_counter()-start))
    print(json.dumps(dict(event='bank_done',scenario=args.scenario,fold=f,bank=name,
        seconds=round(time.perf_counter()-start,2))),flush=True)
    return grids


def run_fold(s,fs,args,f,foldids,scores,perdrug):
    a=np.flatnonzero(foldids!=f);q=np.flatnonzero(foldids==f)
    ref=load(args.parent/f'fold-{f}'/'predictions.npz') if args.scenario=='tissue' else None
    panel=ref['panel'] if ref is not None else np.arange(s['auc'].shape[1])
    inner=group_folds(s['tissues'][a],3)[0] if args.scenario=='tissue' else cell_folds(s['tissues'][a],20261008,nfolds=3)
    dest=args.output/f'fold-{f}';dest.mkdir()
    total={};count={}
    for j in range(3):
        aa,qq=a[inner!=j],a[inner==j]
        grids=run_bank(s,fs,args,f,f'inner-{j}',aa,qq,panel,dest/f'inner-{j}')
        for arm,grid in grids.items():
            ll,nn=losses(grid,s['auc'][np.ix_(qq,panel)],np.maximum(np.nanstd(s['auc'][np.ix_(aa,panel)],axis=0),1e-8))
            total[arm]=total.get(arm,0)+ll;count[arm]=count.get(arm,0)+nn
    selected={arm:choose(total[arm],count[arm],fallback=1) for arm in total}
    save_json(dest/'selection.json',dict(fit=a,held=q,panel=panel,inner_ids=inner,
        configs=CONFIGS,baseline_alphas=ALPHAS,losses=total,counts=count,selected=selected))
    grids=run_bank(s,fs,args,f,'outer',a,q,panel,dest/'outer')
    pred={arm:take(grid,selected[arm]) for arm,grid in grids.items()}
    parity={}
    if ref is not None:
        for arm,ref_arm in [('m_only','m_only'),('virtual_fm','frozen_virtual_fm')]:
            np.testing.assert_allclose(pred[arm],ref[ref_arm],rtol=1e-10,atol=1e-11)
            parity[arm]=float(np.max(np.abs(pred[arm]-ref[ref_arm])))
            pred[arm]=ref[ref_arm].copy()
        for arm in ('virtual_p','measured_p','measured_mask','frozen_virtual_rna'):pred['old_'+arm]=ref[arm]
    sd=np.maximum(np.nanstd(s['auc'][np.ix_(a,panel)],axis=0),1e-8)
    truth=s['auc'][np.ix_(q,panel)];valid=np.isfinite(truth).sum(0)>=5
    np.savez_compressed(dest/'predictions.npz',fit=a,held=q,panel=panel,drug_ids=s['drug_ids'][panel],
        fit_sample_ids=s['sample_ids'][a],sample_ids=s['sample_ids'][q],truth=truth,train_sd=sd,evaluated=valid,**pred)
    for arm,values in pred.items():
        m=metrics(truth,values,sd,valid)
        scores.append(dict(fold=f,arm=arm,score=m['score'],spearman=m['spearman'],centered_mse=m['centered_mse'],drugs=int(valid.sum())))
        for j in range(len(panel)):
            perdrug.append(dict(fold=f,arm=arm,drug_id=str(s['drug_ids'][panel[j]]),count=int(m['count'][j]),
                evaluated=bool(valid[j]),standardized_mse=float(m['mse'][j]),spearman=float(m['rho'][j])))
    save_json(dest/'status.json',dict(status='completed',original_baseline_parity=parity))
    print(json.dumps(dict(event='fold_done',scenario=args.scenario,fold=f,scores=[x for x in scores if x['fold']==f])),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--fm-parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scenario',choices=('tissue','new_cell'),required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    assert sha(args.parent/'source-inputs.npz')==SOURCE_SHA
    s=load(args.parent/'source-inputs.npz');fs=load(args.fm_parent/'source-inputs.npz')
    for key in ('m','auc','sample_ids','drug_ids','fold_ids','tissues'):np.testing.assert_array_equal(s[key],fs[key])
    ids=s['fold_ids'] if args.scenario=='tissue' else cell_folds(s['tissues'],20261008,outer=True)
    save_json(args.output/'protocol.json',dict(script_sha256=sha(__file__),helper_sha256=sha(Path(__file__).with_name('run_recoverability.py')),
        source_sha256=SOURCE_SHA,fm_source_sha256=sha(args.fm_parent/'source-inputs.npz'),outer_fold_ids=ids,
        scenario=args.scenario,configs=CONFIGS,graphs=GRAPHS,primary='virtual_fm__protein',
        graph_neighbors=10,graph_scaling='combinatorial Laplacian trace=n',query_input='M only',
        candidate_only=True,innovation_claim=False,external_confirmation=False))
    scores=[];perdrug=[]
    for f in range(5):
        run_fold(s,fs,args,f,ids,scores,perdrug)
        csv_save(args.output/'fold_metrics.csv',scores);csv_save(args.output/'per_drug_metrics.csv',perdrug)
    summary={arm:{k:float(np.mean([r[k] for r in scores if r['arm']==arm])) for k in ('score','spearman','centered_mse')}
        for arm in sorted(set(r['arm'] for r in scores))}
    save_json(args.output/'summary.json',dict(status='completed',scenario=args.scenario,folds=5,summary=summary,seconds=time.perf_counter()-start))
    print(json.dumps(dict(event='complete',scenario=args.scenario,summary=summary,seconds=time.perf_counter()-start)),flush=True)


if __name__=='__main__':main()

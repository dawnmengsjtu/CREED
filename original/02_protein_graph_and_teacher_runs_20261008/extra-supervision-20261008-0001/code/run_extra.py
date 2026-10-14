"""Add qualified M-response observations to frozen strong kernel/graph banks."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.linalg import eigh,cho_factor,cho_solve
from scipy.spatial.distance import cdist
from creed.science.metabolite_query import predict_generator
from run_recoverability import (load,sha,save_json,csv_save,predict_grid,losses,choose,take,
    metrics,ALPHAS,SOURCE_SHA)

MUS=(0.,.1,1.,10.,100.,1000.)
BASES=('m_only','virtual_fm');GRAPHS=('protein','mask','shuffle','metabolite')
BASE_CONFIGS={b:tuple((0.,a) for a in ALPHAS) for b in BASES}
BASE_CONFIGS.update({b+'__'+g:tuple((mu,a) for mu in MUS for a in ALPHAS) for b in BASES for g in GRAPHS})
CONFIGS={prefix+arm:c for prefix in ('original__','extra__') for arm,c in BASE_CONFIGS.items()}
META=('fit','held','panel','drug_ids','fit_sample_ids','sample_ids','truth','train_sd','evaluated')

def kernel_state(x):
    mean=x.mean(0);sd=x.std(0);sd=np.where(sd<1e-8,1.,sd);z=(x-mean)/sd
    d=cdist(z,z,'sqeuclidean');v=d[np.triu_indices(len(z),1)];v=v[v>0]
    if not len(v):raise ValueError('Degenerate frozen feature bank')
    return mean,sd,float(np.median(v))

def rbf(a,b,state):
    mean,sd,width=state
    return np.exp(-cdist((a-mean)/sd,(b-mean)/sd,'sqeuclidean')/(2*width))

def extend_blocks(old_k,old_h,extra_to_old,extra_to_extra,query_to_extra):
    k=np.block([[old_k,extra_to_old.T],[extra_to_old,extra_to_extra]])
    h=np.concatenate((old_h,query_to_extra),axis=1)
    return k,h

def extend_kernels(s,e,z,inp,generator_path,extra_indices):
    n=len(z['fit']);m=s['m'][z['fit']];q=s['m'][z['query']];em=e['m'][extra_indices]
    state=kernel_state(m)
    np.testing.assert_allclose(rbf(m,m,state),z['m_only_kernel'],rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(rbf(q,m,state),z['m_only_cross'],rtol=1e-11,atol=1e-12)
    mk,mh=extend_blocks(z['m_only_kernel'],z['m_only_cross'],rbf(em,m,state),rbf(em,em,state),rbf(q,em,state))
    ft,fq=inp['fm_train'],inp['fm_query_views'];fs=kernel_state(ft)
    fk_old=rbf(ft,ft,fs);fh_old=np.mean([rbf(v,ft,fs) for v in fq],axis=0)
    np.testing.assert_allclose(.5*z['m_only_kernel']+.5*fk_old,z['virtual_fm_kernel'],rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(.5*z['m_only_cross']+.5*fh_old,z['virtual_fm_cross'],rtol=1e-11,atol=1e-12)
    fe=np.array([predict_generator(load(generator_path/f'generator-{i}.npz'),em)['virtual_fm'] for i in range(3)])
    eo=np.mean([rbf(v,ft,fs) for v in fe],axis=0)
    ee=np.mean([rbf(a,b,fs) for a in fe for b in fe],axis=0);ee=(ee+ee.T)/2
    qe=np.mean([rbf(a,b,fs) for a in fq for b in fe],axis=0)
    fk,fh=extend_blocks(z['virtual_fm_kernel'],z['virtual_fm_cross'],.5*mk[n:,:n]+.5*eo,
        .5*mk[n:,n:]+.5*ee,.5*mh[:,n:]+.5*qe)
    for k,h,base in ((mk,mh,'m_only'),(fk,fh,'virtual_fm')):
        np.testing.assert_array_equal(k[:n,:n],z[base+'_kernel'])
        np.testing.assert_array_equal(h[:,:n],z[base+'_cross'])
    return dict(m_only=(mk,mh),virtual_fm=(fk,fh)),fe

def allowed_extra(s,e,fit,scenario):
    ok=np.isfinite(e['m']).all(1)&np.isfinite(e['auc']).any(1)
    if scenario=='tissue':
        excluded=np.setdiff1d(np.arange(len(s['m'])),fit)
        banned=set(s['tissues'][excluded].tolist())
        ok&=np.array([bool(t) and t not in banned for t in e['tissues']])
    return np.flatnonzero(ok)

def run_bank(s,e,args,old,dest,scenario,f,name):
    start=time.perf_counter();dest.mkdir()
    z=load(old/'geometry.npz');inp=load(old/'graph_inputs.npz');prior=load(old/'grids.npz')
    ei=allowed_extra(s,e,z['fit'],scenario);assert len(ei)>0
    gen=args.fm_parent/f'fold-{f}'/name if scenario=='tissue' else args.previous.parent/'new_cell'/f'fold-{f}'/name
    kernels,fe=extend_kernels(s,e,z,inp,gen,ei)
    y=np.concatenate((s['auc'][np.ix_(z['fit'],z['panel'])],e['auc'][np.ix_(ei,z['panel'])]))
    grids={'original__'+arm:prior[arm] for arm in BASE_CONFIGS};geometry={}
    for base,(k,h) in kernels.items():
        vals,vecs=eigh((k+k.T)/2);assert vals.min()>-1e-8
        root=(vecs*np.sqrt(np.maximum(vals,0)))@vecs.T
        grid0=predict_grid(k,h,y);grids['extra__'+base]=grid0
        geometry[base+'_kernel']=k;geometry[base+'_cross']=h
        for graph in GRAPHS:
            L=np.zeros_like(k);L[:len(z['fit']),:len(z['fit'])]=z['L_'+graph]
            geometry['L_'+graph]=L;parts=[grid0]
            for mu in MUS[1:]:
                a=np.eye(len(k))+mu*root@L@root;a=(a+a.T)/2
                ke=root@cho_solve(cho_factor(a,lower=True),root);ke=(ke+ke.T)/2
                he=h-mu*h@L@ke;parts.append(predict_grid(ke,he,y))
            grids['extra__'+base+'__'+graph]=np.concatenate(parts)
    no_extra_labels=~np.isfinite(e['auc'][np.ix_(ei,z['panel'])]).any(0)
    # Zero-weight extra nodes without labels cannot change that drug's objective.
    for arm in BASE_CONFIGS:grids['extra__'+arm][:,:,no_extra_labels]=prior[arm][:,:,no_extra_labels]
    np.savez_compressed(dest/'grids.npz',**grids)
    np.savez_compressed(dest/'geometry.npz',fit=z['fit'],query=z['query'],panel=z['panel'],extra_indices=ei,
        extra_sample_ids=e['sample_ids'][ei],extra_fm_views=fe,**geometry)
    save_json(dest/'record.json',dict(fit=z['fit'],query=z['query'],panel=z['panel'],extra_indices=ei,
        extra_sample_ids=e['sample_ids'][ei],extra_tissues=e['tissues'][ei],extra_count=len(ei),
        no_extra_label_drugs=int(no_extra_labels.sum()),
        extra_observed_response_count=np.isfinite(e['auc'][np.ix_(ei,z['panel'])]).sum(0),
        old_geometry_sha256=sha(old/'geometry.npz'),old_inputs_sha256=sha(old/'graph_inputs.npz'),
        old_grids_sha256=sha(old/'grids.npz'),generator_sha256=[sha(gen/f'generator-{i}.npz') for i in range(3)],
        new_graph_nodes='zero rows/cols; original trace unchanged',old_kernel_blocks_exact=True,
        seconds=time.perf_counter()-start))
    print(json.dumps(dict(event='bank_done',scenario=scenario,fold=f,bank=name,extra=len(ei),
        seconds=round(time.perf_counter()-start,2))),flush=True)
    return grids,z

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--extra',type=Path,required=True);p.add_argument('--extra-sha',required=True)
    p.add_argument('--previous',type=Path,required=True);p.add_argument('--fm-parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    assert sha(args.source)==SOURCE_SHA and sha(args.extra)==args.extra_sha
    with np.load(args.source,allow_pickle=False) as q:s={k:q[k] for k in ('m','auc','sample_ids','metabolite_ids','drug_ids','tissues')}
    e=load(args.extra)
    assert not (set(s['sample_ids'])&set(e['sample_ids']))
    np.testing.assert_array_equal(s['metabolite_ids'],e['m_feature_ids']);np.testing.assert_array_equal(s['drug_ids'],e['drug_ids'])
    assert np.isfinite(e['m']).all() and np.isfinite(e['auc']).any(1).all()
    op=json.loads((args.previous/'protocol.json').read_text());scenario=op['scenario']
    args.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    save_json(args.output/'protocol.json',dict(scenario=scenario,source_sha256=SOURCE_SHA,extra_sha256=args.extra_sha,
        script_sha256=sha(__file__),helper_sha256=sha(Path(__file__).with_name('run_recoverability.py')),
        configs=CONFIGS,outer_fold_ids=op['outer_fold_ids'],candidate_only=True,external_confirmation=False,
        main_base='virtual_fm' if scenario=='tissue' else 'm_only',query_input='M only',
        feature_transform='old fit-only moments and bandwidth; saved FM generators; old blocks exact',
        fm_extra='kernel mean embedding of 3 generator views: 3-way extra-old,9-way extra-extra/query-extra',
        graph='original graph padded with zeros, original trace not rescaled',
        selection='old 3 inner folds and original-fit SD; extra training labels available equally to all arms'))
    scores=[];perdrug=[]
    for f in range(5):
        old=args.previous/f'fold-{f}';dest=args.output/f'fold-{f}';dest.mkdir()
        so=json.loads((old/'selection.json').read_text());total={};count={}
        for j in range(3):
            grids,z=run_bank(s,e,args,old/f'inner-{j}',dest/f'inner-{j}',scenario,f,f'inner-{j}')
            truth=s['auc'][np.ix_(z['query'],z['panel'])]
            sd=np.maximum(np.nanstd(s['auc'][np.ix_(z['fit'],z['panel'])],axis=0),1e-8)
            for arm,g in grids.items():
                ll,nn=losses(g,truth,sd);total[arm]=total.get(arm,0)+ll;count[arm]=count.get(arm,0)+nn
        selected={arm:choose(total[arm],count[arm],fallback=1) for arm in CONFIGS}
        save_json(dest/'selection.json',dict(**{k:so[k] for k in ('fit','held','panel','inner_ids')},
            configs=CONFIGS,losses=total,counts=count,selected=selected))
        grids,z=run_bank(s,e,args,old/'outer',dest/'outer',scenario,f,'outer')
        previous=load(old/'predictions.npz');pred={arm:take(g,selected[arm]) for arm,g in grids.items()}
        for arm in BASE_CONFIGS:np.testing.assert_array_equal(pred['original__'+arm],previous[arm])
        np.savez_compressed(dest/'predictions.npz',**{k:previous[k] for k in META},**pred)
        for arm,values in pred.items():
            m=metrics(previous['truth'],values,previous['train_sd'],previous['evaluated'])
            scores.append(dict(fold=f,arm=arm,score=m['score'],spearman=m['spearman'],centered_mse=m['centered_mse'],
                drugs=int(previous['evaluated'].sum())))
            for j,drug in enumerate(previous['drug_ids']):
                perdrug.append(dict(fold=f,arm=arm,drug_id=str(drug),count=int(m['count'][j]),evaluated=bool(previous['evaluated'][j]),
                    standardized_mse=float(m['mse'][j]),spearman=float(m['rho'][j])))
        csv_save(args.output/'fold_metrics.csv',scores);csv_save(args.output/'per_drug_metrics.csv',perdrug)
        save_json(dest/'status.json',dict(status='completed',all_original_arms_exact=True))
        print(json.dumps(dict(event='fold_done',scenario=scenario,fold=f)),flush=True)
    summary={arm:{k:float(np.mean([r[k] for r in scores if r['arm']==arm])) for k in ('score','spearman','centered_mse')} for arm in CONFIGS}
    save_json(args.output/'summary.json',dict(status='completed',scenario=scenario,folds=5,summary=summary,seconds=time.perf_counter()-start))
    print(json.dumps(dict(event='complete',scenario=scenario,seconds=time.perf_counter()-start)),flush=True)

if __name__=='__main__':main()

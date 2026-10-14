"""One disclosed inner-boundary motivated extension; original outputs retained."""
import argparse,json,time
from pathlib import Path
import numpy as np
from run_geometry import (load,sha,save_json,csv_save,metrics,losses,choose,take,
    effective,predict_grid,ALPHAS,BASES,GRAPHS,SOURCE_SHA)

MUS=(0.,.1,1.,10.,100.,1000.)
CONFIGS=tuple((mu,alpha) for mu in MUS for alpha in ALPHAS)
META=('fit','held','panel','drug_ids','fit_sample_ids','sample_ids','truth','train_sd','evaluated')

def bank(s,old,dest,previous_name,f,name):
    start=time.perf_counter();dest.mkdir(parents=True)
    z=load(old/'geometry.npz');grids=load(old/'grids.npz')
    y=s['auc'][np.ix_(z['fit'],z['panel'])]
    for base in BASES:
        k,h=z[base+'_kernel'],z[base+'_cross']
        for graph in GRAPHS:
            arm=base+'__'+graph;parts=[grids[arm]]
            for mu in MUS[4:]:
                ke,he=effective(k,h,z['L_'+graph],mu);parts.append(predict_grid(ke,he,y))
            grids[arm]=np.concatenate(parts)
    np.savez_compressed(dest/'grids.npz',**grids)
    for fn in ('geometry.npz','graph_inputs.npz'):
        (dest/fn).symlink_to(Path('../../..')/previous_name/f'fold-{f}'/name/fn)
    record=json.loads((old/'record.json').read_text())
    record.update(extension_only=True,original_grid_sha256=sha(old/'grids.npz'),original_bank=str(old),
        added_mu=MUS[4:],seconds=time.perf_counter()-start)
    save_json(dest/'record.json',record)
    print(json.dumps(dict(event='extended_bank',scenario=previous_name,fold=f,bank=name,seconds=record['seconds'])),flush=True)
    return grids,z

def main():
    p=argparse.ArgumentParser();p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();assert sha(a.source)==SOURCE_SHA
    s=load(a.source);protocol=json.loads((a.previous/'protocol.json').read_text())
    protocol.update(extension_script_sha256=sha(__file__),configs=CONFIGS,
        previous_protocol_sha256=sha(a.previous/'protocol.json'),adaptation='v1 outer scores seen; only inner-boundary motivated mu100/1000 extension, equal for all controls')
    save_json(a.output/'protocol.json',protocol);scores=[];perdrug=[]
    for f in range(5):
        old=a.previous/f'fold-{f}';dest=a.output/f'fold-{f}';dest.mkdir()
        original_selection=json.loads((old/'selection.json').read_text());total={};count={}
        for j in range(3):
            grids,z=bank(s,old/f'inner-{j}',dest/f'inner-{j}',a.previous.name,f,f'inner-{j}')
            y=s['auc'][np.ix_(z['query'],z['panel'])]
            sd=np.maximum(np.nanstd(s['auc'][np.ix_(z['fit'],z['panel'])],axis=0),1e-8)
            for arm,g in grids.items():
                ll,nn=losses(g,y,sd);total[arm]=total.get(arm,0)+ll;count[arm]=count.get(arm,0)+nn
        selected={arm:choose(total[arm],count[arm],fallback=1) for arm in total}
        sel={k:original_selection[k] for k in ('fit','held','panel','inner_ids')}
        sel.update(configs=CONFIGS,baseline_alphas=ALPHAS,losses=total,counts=count,selected=selected)
        save_json(dest/'selection.json',sel)
        grids,z=bank(s,old/'outer',dest/'outer',a.previous.name,f,'outer')
        previous=load(old/'predictions.npz');pred={arm:take(g,selected[arm]) for arm,g in grids.items()}
        for base in BASES:
            np.testing.assert_allclose(pred[base],previous[base],rtol=1e-10,atol=1e-11)
            pred[base]=previous[base]
        pred.update({k:v for k,v in previous.items() if k.startswith('old_')})
        np.savez_compressed(dest/'predictions.npz',**{k:previous[k] for k in META},**pred)
        truth,sd,valid=previous['truth'],previous['train_sd'],previous['evaluated']
        for arm,values in pred.items():
            m=metrics(truth,values,sd,valid)
            scores.append(dict(fold=f,arm=arm,score=m['score'],spearman=m['spearman'],centered_mse=m['centered_mse'],drugs=int(valid.sum())))
            for j,drug in enumerate(previous['drug_ids']):
                perdrug.append(dict(fold=f,arm=arm,drug_id=str(drug),count=int(m['count'][j]),evaluated=bool(valid[j]),
                    standardized_mse=float(m['mse'][j]),spearman=float(m['rho'][j])))
        save_json(dest/'status.json',dict(status='completed',baseline_predictions_unchanged=True,extension_only=True))
        csv_save(a.output/'fold_metrics.csv',scores);csv_save(a.output/'per_drug_metrics.csv',perdrug)
        print(json.dumps(dict(event='fold_done',scenario=a.previous.name,fold=f)),flush=True)
    summary={arm:{k:float(np.mean([r[k] for r in scores if r['arm']==arm])) for k in ('score','spearman','centered_mse')}
        for arm in sorted(set(r['arm'] for r in scores))}
    save_json(a.output/'summary.json',dict(status='completed',scenario=a.previous.name,folds=5,summary=summary,seconds=time.perf_counter()-start))
    print(json.dumps(dict(event='complete',scenario=a.previous.name,summary=summary,seconds=time.perf_counter()-start)),flush=True)

if __name__=='__main__':main()

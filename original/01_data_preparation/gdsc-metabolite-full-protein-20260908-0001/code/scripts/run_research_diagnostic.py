"""Bounded, exploratory CREED diagnosis; never the formal Stage-A evaluator."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from creed.science import ccle_multifm_bidir_v5 as core
from creed.science import ccle_multifm_v5_direct_comparators as direct


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(truth, prediction, train):
    if truth.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError('invalid prediction')
    a = truth - truth.mean(axis=0)
    b = prediction - prediction.mean(axis=0)
    denominator = np.sqrt((a*a).sum(axis=0) * (b*b).sum(axis=0))
    # Axis reductions can leave roundoff residuals even for exactly repeated
    # nonzero values. Exact range checks keep constant controls undefined.
    valid = (denominator > 1e-12) & (np.ptp(truth,axis=0)>0) & (np.ptp(prediction,axis=0)>0)
    correlations = np.zeros(truth.shape[1])
    correlations[valid] = (a*b).sum(axis=0)[valid] / denominator[valid]
    # Zero correlation for constant predictions: never silently drop failures.
    scale = train.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    a = truth - truth.mean(axis=1, keepdims=True)
    b = prediction - prediction.mean(axis=1, keepdims=True)
    den = np.sqrt((a*a).sum(axis=1) * (b*b).sum(axis=1))
    row = np.divide((a*b).sum(axis=1), den, out=np.zeros(len(a)), where=den>1e-12)
    return dict(mean_feature_pearson=float(correlations.mean()),
                mean_sample_centered_pearson=float(row.mean()),
                train_scaled_rmse=float(np.sqrt(np.mean(((truth-prediction)/scale)**2))),
                constant_truth_features=int((truth.std(axis=0)<1e-12).sum()),
                undefined_correlation_features=int((~valid).sum()))


def synthetic():
    rng = np.random.default_rng(20260905)
    z = rng.normal(size=(421, 8))
    return dict(metabolites=np.ascontiguousarray(z @ rng.normal(size=(8,225))),
                proteins=np.ascontiguousarray(np.tanh(z @ rng.normal(size=(8,584)))),
                scgpt=np.ascontiguousarray(z @ rng.normal(size=(8,512)) + rng.normal(size=(421,512))*.01),
                esm=np.ascontiguousarray(rng.normal(size=(584,1280))))


def load_package(root):
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['scope'] != 'exploratory_training_421_only':
        raise ValueError('wrong package scope')
    arrays = {}
    for name, shape in dict(metabolites=(421,225),proteins=(421,584),scgpt=(421,512),esm=(584,1280)).items():
        path=root/(name+'.npy')
        if sha(path) != manifest['files'][path.name]['sha256']:
            raise ValueError('package hash changed: '+name)
        arrays[name]=np.load(path,allow_pickle=False)
        if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise ValueError('package shape/finite violation: '+name)
        arrays[name]=np.ascontiguousarray(arrays[name],dtype=np.float64)
    return arrays


def rbf_predict(x, y, held, alpha):
    from sklearn.metrics.pairwise import euclidean_distances
    from scipy.linalg import solve
    mean=x.mean(0); scale=x.std(0); scale=np.where(scale<1e-8,1.,scale)
    xx=(x-mean)/scale; hh=(held-mean)/scale
    d=euclidean_distances(xx,squared=True)
    positive=d[np.triu_indices(len(x),1)]
    width=float(np.median(positive[positive>0]))
    if not width>0: raise ValueError('degenerate RBF inputs')
    k=np.exp(-d/(2*width)); k.flat[::len(x)+1]+=alpha
    ym=y.mean(0)
    coef=solve(k,y-ym,assume_a='pos')
    return np.exp(-euclidean_distances(hh,xx,squared=True)/(2*width))@coef+ym


def selected_rbf(x,y,held):
    from sklearn.model_selection import KFold
    scores=[]
    for alpha in (.1,1.,10.,100.):
        losses=[]
        for fit,val in KFold(3,shuffle=True,random_state=905).split(x):
            pred=rbf_predict(x[fit],y[fit],x[val],alpha)
            scale=y[fit].std(0); scale=np.where(scale<1e-8,1.,scale)
            losses.append(float(np.mean(((pred-y[val])/scale)**2)))
        scores.append(float(np.mean(losses)))
    i=int(np.argmin(scores)); alpha=(.1,1.,10.,100.)[i]
    return rbf_predict(x,y,held,alpha), dict(alpha=alpha,inner_losses=scores)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path)
    parser.add_argument('--synthetic',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--arm',choices=['full','no_scgpt','no_esm_identity_only','baselines'],required=True)
    parser.add_argument('--fold',type=int,choices=range(5),default=0)
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.synthetic == (args.data is not None): parser.error('choose synthetic or data')
    if not args.synthetic and args.epochs!=50: parser.error('real diagnostic uses fixed 50 epochs')
    args.output.mkdir(parents=True,exist_ok=False)
    config=dict(arm=args.arm,fold=args.fold,epochs=args.epochs,device=args.device,
                synthetic=args.synthetic,scientific_claim=False,script_sha256=sha(__file__),
                core_sha256=sha(core.__file__))
    (args.output/'intent.json').write_text(json.dumps(config,indent=2))
    started=time.perf_counter()
    try:
        data=synthetic() if args.synthetic else load_package(args.data)
        if args.data:
            config['input_manifest_sha256']=sha(args.data/'manifest.json')
        fold=direct.stage_a_folds()[args.fold]
        fit=np.asarray(fold.fit_indices); held=np.asarray(fold.held_indices)
        m,p,s,e=[data[k] for k in ['metabolites','proteins','scgpt','esm']]
        predictions={}; selections={}
        if args.arm=='baselines':
            for direction,x,y in [('forward',m,p),('reverse',p,m)]:
                predictions[direction+'_direct'],selections[direction+'_direct']=selected_rbf(x[fit],y[fit],x[held])
                predictions[direction+'_measured_scgpt_oracle'],selections[direction+'_oracle']=selected_rbf(s[fit],y[fit],s[held])
                virtual,selection=selected_rbf(x[fit],s[fit],x[held])
                selections[direction+'_source_to_scgpt']=selection
                # Same fit-only second-stage model; no held target enters either fit.
                predictions[direction+'_bridge'],selections[direction+'_bridge']=selected_rbf(s[fit],y[fit],virtual)
                predictions[direction+'_virtual_scgpt']=virtual
        else:
            cfg=core.MultiFMBidirV5Config(epochs=args.epochs,device=args.device,seed=20260905,
                teacher_loss_weight=0. if args.arm=='no_scgpt' else .1)
            candidate='v5-noscgpt-k04-d032-q000' if args.arm=='no_scgpt' else 'v5-k04-d032-t010-q000'
            context=core.V5DropoutSeedContext(dropout_root=321870185,stage='stage_a',
                phase='A1' if args.arm=='full' else 'A2',family=args.arm,direction_or_joint='joint',
                candidate_id=candidate,outer_repeat=0,outer_fold=args.fold,inner_repeat='-',inner_fold='-',
                member=0,purpose='optimizer_fit',registry_positions=tuple(int(x) for x in fit))
            fitted=core.fit_multifm_bidir_v5_fold(metabolites=m[fit],proteins=p[fit],
                protein_mask=np.ones(p[fit].shape,dtype=bool),esm_embeddings=None if args.arm=='no_esm_identity_only' else e,
                scgpt=None if args.arm=='no_scgpt' else s[fit],config=cfg,control=args.arm,dropout_context=context)
            predictions['forward']=fitted.predict_forward(m[held]).proteins
            predictions['reverse']=fitted.predict_reverse(p[held],np.ones(p[held].shape,dtype=bool)).metabolites
            (args.output/'training.json').write_text(json.dumps([asdict(x) for x in fitted.loss_history]))
            weights={k:v.detach().cpu().numpy() for k,v in fitted._model.state_dict().items()}
            np.savez(args.output/'checkpoint.npz',**weights)
        scored={}
        for key,pred in predictions.items():
            y=s if key.endswith('virtual_scgpt') else p if key.startswith('forward') else m
            scored[key]=metrics(y[held],pred,y[fit])
        np.savez(args.output/'predictions.npz',fit_indices=fit,held_indices=held,**predictions)
        config.update(status='completed',wall_seconds=time.perf_counter()-started,metrics=scored,selections=selections)
        config['artifacts']={p.name:sha(p) for p in args.output.iterdir() if p.is_file()}
        (args.output/'result.json').write_text(json.dumps(config,indent=2,allow_nan=False))
        print(json.dumps(config,allow_nan=False),flush=True)
    except Exception as error:
        (args.output/'failure.json').write_text(json.dumps(dict(error=repr(error),wall_seconds=time.perf_counter()-started)))
        raise


if __name__=='__main__': main()

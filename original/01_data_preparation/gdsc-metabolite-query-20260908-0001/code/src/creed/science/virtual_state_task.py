"""Cross-fitted virtual layers condition actual task kernels; array-only primitives.

Virtual features add a supervised inductive bias, not new measured information.
The caller must rebuild generators within every task-selection training split.
"""
import numpy as np
from scipy.linalg import solve
from sklearn.metrics.pairwise import euclidean_distances


def stats(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('finite matrix required')
    mean=x.mean(0);sd=x.std(0);sd=np.where(sd<1e-8,1.,sd)
    return mean,sd


def fit_generator(rna,targets,genes,alpha=1.,seed=1708):
    """No downstream task labels accepted. Shuffle pairing, never feature identities."""
    x=np.asarray(rna[:,genes],dtype=float);mean,sd=stats(x);x=(x-mean)/sd
    d=euclidean_distances(x,squared=True);pos=d[np.triu_indices(len(x),1)]
    width=float(np.median(pos[pos>0]))
    if not width>0:raise ValueError('degenerate generator input')
    kernel=np.exp(-d/(2*width));kernel.flat[::len(x)+1]+=alpha
    names=tuple(targets);cuts=np.cumsum([0]+[targets[k].shape[1] for k in names])
    y=np.concatenate([targets[k] for k in names],axis=1).astype(float)
    if not np.isfinite(y).all():raise ValueError('nonfinite paired targets')
    perm=np.random.default_rng(seed).permutation(len(y));ym=y.mean(0)
    coef=solve(kernel,np.concatenate([y-ym,y[perm]-ym],axis=1),assume_a='pos')
    return dict(genes=genes,mean=mean,sd=sd,x=x,width=width,coef=coef,target_mean=ym,
                names=names,cuts=cuts,permutation=perm,alpha=alpha)


def generate(model,rna):
    x=(rna[:,model['genes']]-model['mean'])/model['sd']
    k=np.exp(-euclidean_distances(x,model['x'],squared=True)/(2*model['width']))
    result=k@model['coef'];width=len(model['target_mean']);out={}
    for prefix,offset in (('',0),('shuffled_',width)):
        for i,name in enumerate(model['names']):
            a,b=model['cuts'][i:i+2];out[prefix+name]=result[:,offset+a:offset+b]+model['target_mean'][a:b]
    return out


def fit_blocks(blocks):
    states=[stats(x) for x in blocks]
    z=np.concatenate([(x-m)/s/np.sqrt(x.shape[1]) for x,(m,s) in zip(blocks,states)],axis=1)/np.sqrt(len(blocks))
    return z,states


def apply_blocks(blocks,states):
    return np.concatenate([(x-m)/s/np.sqrt(x.shape[1]) for x,(m,s) in zip(blocks,states)],axis=1)/np.sqrt(len(blocks))


def component_kernel(x,q,kind):
    if kind=='ridge':return x@x.T,q@x.T,None
    if kind!='rbf':raise ValueError('unknown kernel')
    d=euclidean_distances(x,squared=True);pos=d[np.triu_indices(len(x),1)]
    width=float(np.median(pos[pos>0]))
    if not width>0:raise ValueError('degenerate task features')
    return np.exp(-d/(2*width)),np.exp(-euclidean_distances(q,x,squared=True)/(2*width)),width


def head_grid(rna,extra,qrna,qextra,y,configs):
    """Actual virtual values determine a task kernel; all scaling is training-only."""
    rx,rs=fit_blocks([rna]);rq=apply_blocks([qrna],rs)
    if extra:
        ex,es=fit_blocks(extra);eq=apply_blocks(qextra,es)
    else:ex=eq=None;es=None
    cache={};predictions=[]
    for kind,alpha,gamma in configs:
        if not 0<=gamma<=1 or (gamma and ex is None):raise ValueError('invalid virtual weight')
        if ('rna',kind) not in cache:cache['rna',kind]=component_kernel(rx,rq,kind)
        kr,hr,_=cache['rna',kind]
        if gamma:
            if ('extra',kind) not in cache:cache['extra',kind]=component_kernel(ex,eq,kind)
            ke,he,_=cache['extra',kind];k=(1-gamma)*kr+gamma*ke;h=(1-gamma)*hr+gamma*he
        else:k=kr.copy();h=hr
        k=k.copy();k.flat[::len(k)+1]+=alpha
        ym=float(np.mean(y));coef=solve(k,y-ym,assume_a='pos')
        predictions.append(h@coef+ym)
    return np.stack(predictions)


def fit_head(rna,extra,y,config):
    kind,alpha,gamma=config;rx,rs=fit_blocks([rna]);kr,_,rw=component_kernel(rx,rx,kind)
    state=dict(config=config,rna_x=rx,rna_states=rs,rna_width=rw)
    if gamma:
        ex,es=fit_blocks(extra);ke,_,ew=component_kernel(ex,ex,kind)
        state.update(extra_x=ex,extra_states=es,extra_width=ew);kernel=(1-gamma)*kr+gamma*ke
    else:kernel=kr
    kernel=kernel.copy();kernel.flat[::len(kernel)+1]+=alpha
    state['mean']=float(np.mean(y));state['coef']=solve(kernel,y-state['mean'],assume_a='pos')
    return state


def predict_head(state,rna,extra):
    kind,_,gamma=state['config'];q=apply_blocks([rna],state['rna_states'])
    def cross(a,b,width):return a@b.T if kind=='ridge' else np.exp(-euclidean_distances(a,b,squared=True)/(2*width))
    k=cross(q,state['rna_x'],state['rna_width'])
    if gamma:
        e=apply_blocks(extra,state['extra_states']);k=(1-gamma)*k+gamma*cross(e,state['extra_x'],state['extra_width'])
    return k@state['coef']+state['mean']

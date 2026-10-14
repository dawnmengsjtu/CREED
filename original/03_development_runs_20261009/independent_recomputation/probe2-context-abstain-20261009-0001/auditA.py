# Independent audit of Part A: recompute metric from saved predictions, re-verify reproduction, recompute NB t.
import json, csv, numpy as np
from scipy.stats import t as tdist, spearmanr
RUN='/public/home/mengxl/creed-research-runs'
OUT=RUN+'/probe2-context-abstain-20261009-0001/output'
EXT=RUN+'/extra-supervision-20261008-0001/output'
PP=RUN+'/protein-panel-20261009-0001/output'
SRC=np.load(RUN+'/gdsc-metabolite-full-protein-20260908-0001/output/result/source-inputs.npz',allow_pickle=False)
auc=SRC['auc']
summ=json.load(open(OUT+'/summary.json'))

def my_score(truth,pred,sd,use):
    # per fold per-drug mse over observed test cells of ((pred-truth)/sd)^2; sqrt(mean over evaluated drugs)
    obs=np.isfinite(truth); out=[]
    for j in np.flatnonzero(use):
        v=obs[:,j]
        out.append(np.mean(((pred[v,j]-truth[v,j])/sd[j])**2))
    return float(np.sqrt(np.mean(out)))

def nb(new,base,r):
    d=(np.asarray(base)-np.asarray(new))/np.asarray(base)*100
    m=d.mean(); v=d.var(ddof=1); se=np.sqrt((1/5+r)*v); t=m/se
    return float((base.mean()-new.mean())/base.mean()*100), int((np.asarray(new)<np.asarray(base)).sum()), float(t), float(2*tdist.sf(abs(t),4))

res={}
maxdiff_metric=0; maxdiff_repro=0
for sc in ('new_cell','tissue'):
    S={}; ratios=[]
    for f in range(5):
        Z=np.load(f'{OUT}/A/{sc}/fold-{f}_predictions.npz',allow_pickle=False)
        P=np.load(f'{EXT}/{sc}/fold-{f}/predictions.npz',allow_pickle=False)
        Q=np.load(f'{PP}/{sc}_main/fold-{f}_predictions.npz',allow_pickle=False)
        truth,sd,use=Z['truth'],Z['train_sd'],Z['evaluated']
        # independent truth / sd check from source-inputs
        fit,held,panel=Z['fit'],Z['held'],Z['panel']
        assert np.array_equal(P['fit'],fit) and np.array_equal(P['held'],held) and np.array_equal(P['panel'],panel)
        t2=auc[np.ix_(held,panel)]; assert np.array_equal(np.isfinite(t2),np.isfinite(truth)) and np.nanmax(np.abs(t2-truth))==0
        sd2=np.maximum(np.nanstd(auc[np.ix_(fit,panel)],axis=0),1e-8); assert np.max(np.abs(sd2-sd))==0
        assert np.array_equal(P['truth'],truth,equal_nan=True) and np.max(np.abs(P['train_sd']-sd))==0 and np.array_equal(P['evaluated'],use)
        ratios.append(len(held)/len(fit))
        # reproduction
        maxdiff_repro=max(maxdiff_repro, float(np.max(np.abs(Z['pred__m_only']-P['original__m_only']))),
                          float(np.max(np.abs(Z['pred__repro_extra_m_only']-P['extra__m_only']))),
                          max(float(np.max(np.abs(Z[f'pred__global_static_k{k}']-Q[f'm_panel_k{k}']))) for k in (8,16,32,64)),
                          float(np.max(np.abs(Z['pred__m_allcand']-Q['m_allcand']))),
                          max(float(np.max(np.abs(Z[f'pred__random_global_k{k}_s{s}']-Q[f'random_k{k}_s{s}']))) for k in (16,32) for s in (0,1,2)),
                          float(np.max(np.abs(Z['pred__teacher_fullP_saved']-Q['teacher_saved']))))
        for key in Z.files:
            if key.startswith('pred__'):
                a=key[6:]; S.setdefault(a,[None]*5)[f]=my_score(truth,Z[key],sd,use)
    r=float(np.mean(ratios))
    # compare to summary + fold_metrics.csv
    for a,v in S.items():
        saved=summ['A'][sc]['summary'][a]['folds']
        maxdiff_metric=max(maxdiff_metric, max(abs(x-y) for x,y in zip(v,saved)))
    def sc_(a):
        if a.endswith('_mean'):
            b=a[:-5]; return np.mean([S[f'{b}_s{s}'] for s in (0,1,2)],axis=0)
        return np.array(S[a])
    rows={}
    pols=['tissue_adaptive','cluster_adaptive'] if sc=='new_cell' else ['cluster_adaptive']
    for pol in pols:
        mr='matched_random_tissue' if pol=='tissue_adaptive' else 'matched_random_cluster'
        for k in (8,16,32,64):
            rows[f'{pol}_vs_global_static_k{k}']=nb(sc_(f'{pol}_k{k}'),sc_(f'global_static_k{k}'),r)
            rows[f'{pol}_vs_matched_random_k{k}']=nb(sc_(f'{pol}_k{k}'),sc_(f'{mr}_k{k}_mean'),r)
            rows[f'{mr}_mean_vs_random_global_mean_k{k}']=nb(sc_(f'{mr}_k{k}_mean'),sc_(f'random_global_k{k}_mean'),r)
    for k in (8,16,32,64):
        rows[f'global_static_vs_random_global_mean_k{k}']=nb(sc_(f'global_static_k{k}'),sc_(f'random_global_k{k}_mean'),r)
    # compare NB with summary
    nbdiff=0
    for name,(imp,w,t,p) in rows.items():
        c=summ['A'][sc]['comparisons'][name]
        nbdiff=max(nbdiff,abs(imp-c['improvement_pct']),abs(p-c['nb_p']),abs(w-c['folds_won']))
    print(sc,'r=',round(r,4),'nb_ratio_saved=',summ['A'][sc]['nb_ratio'],'max|NB diff|=',nbdiff)
    for a in ['m_only','m_allcand','teacher_fullP_saved']+[f'{p}_k{k}' for k in (8,16,32,64) for p in (['global_static','tissue_adaptive','cluster_adaptive','matched_random_tissue','matched_random_cluster','random_global'] if sc=='new_cell' else ['global_static','cluster_adaptive','matched_random_cluster','random_global'])]:
        key=a if a in S else a+'_mean'
        v=sc_(key); print(f'  {key:38s} mean={v.mean():.6f} folds={np.round(v,4).tolist()}')
    for name in [n for n in rows if ('k16' in n or 'k32' in n or 'k64' in n) and ('vs_global_static' in n or 'vs_matched_random' in n or 'global_static_vs' in n)]:
        imp,w,t,p=rows[name]; print(f'  {name:55s} imp%={imp:+.3f} wins={w}/5 t={t:+.3f} p={p:.4f}')
    # verdict check
    for pol in pols:
        i=all(rows[f'{pol}_vs_global_static_k{k}'][0]>0 and rows[f'{pol}_vs_global_static_k{k}'][1]>=4 and rows[f'{pol}_vs_global_static_k{k}'][3]<0.10 for k in (16,32))
        ii=all(rows[f'{pol}_vs_matched_random_k{k}'][0]>0 and rows[f'{pol}_vs_matched_random_k{k}'][1]>=3 for k in (16,32))
        print(f'  verdict {pol}: (i)={i} (ii)={ii} pass={i and ii}')
print('max |my metric - saved fold score| =',maxdiff_metric)
print('max reproduction diff (m_only, extra, m_panel_k, m_allcand, random, teacher) =',maxdiff_repro)
# selection file: check alpha counts/loss shapes and that selected alpha = argmin of inner loss
for sc in ('new_cell','tissue'):
    for f in range(5):
        Sel=np.load(f'{OUT}/A/{sc}/fold-{f}_selection.npz',allow_pickle=False)
        for a in ('global_static_k16','tissue_adaptive_k16','cluster_adaptive_k32'):
            if 'sel__'+a not in Sel.files: continue
            loss,cnt,sel=Sel['loss__'+a],Sel['count__'+a],Sel['sel__'+a]
            avg=np.divide(loss,cnt,out=np.full_like(loss,np.inf),where=cnt>0); idx=avg.argmin(0); idx[np.all(cnt==0,axis=0)]=1
            assert np.array_equal(idx,sel), (sc,f,a)
            assert cnt.max()<=3 and loss.shape[0]==4
print('selection = argmin of summed inner loss (fallback 1): OK; inner count max 3 banks')

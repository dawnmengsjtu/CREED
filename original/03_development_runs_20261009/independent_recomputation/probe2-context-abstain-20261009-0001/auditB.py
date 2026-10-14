# Independent audit of Part B: refit u1 from scratch (own implementation), u2/u3, per-cell error, risk-coverage, AURC, NB t.
import json, csv, hashlib, numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from scipy.stats import t as tdist
RUN='/public/home/mengxl/creed-research-runs'
OUT=RUN+'/probe2-context-abstain-20261009-0001/output'
EXT=RUN+'/extra-supervision-20261008-0001/output'
MX=RUN+'/crosslayer-matrix-20261009-0001/output'
S=np.load(RUN+'/gdsc-metabolite-full-protein-20260908-0001/output/result/source-inputs.npz',allow_pickle=False)
X=np.load(RUN+'/rna421-20260906-0002/rna_log_cpm.npy'); assert (np.load(RUN+'/rna421-20260906-0002/sample_ids.npy')==S['sample_ids']).all()
summ=json.load(open(OUT+'/summary.json'))
def seed_of(*p): return int.from_bytes(hashlib.sha256('|'.join(map(str,p)).encode()).digest()[:8],'little')
def zstd(x):
    mu=x.mean(0); sd=x.std(0); sd=np.where(sd<1e-8,1.,sd); return (x-mu)/sd
def submodels(K,H,Z,seed):
    n=len(K); folds=np.array_split(np.random.default_rng(seed).permutation(n),3); best=None
    for lam in (.1,1.,10.,100.):
        oof=np.empty_like(Z); sq=[]
        for held in folds:
            tr=np.setdiff1d(np.arange(n),held); mu=Z[tr].mean(0)
            c=cho_solve(cho_factor(K[np.ix_(tr,tr)]+lam*np.eye(len(tr)),lower=True),Z[tr]-mu)
            oof[held]=K[np.ix_(held,tr)]@c+mu; sq.append(H[:,tr]@c+mu)
        mse=((oof-Z)**2).mean()
        if best is None or mse<best[0]: best=(mse,lam,np.stack(sq))
    return best[1],best[2]
def ret_score(E,obs,use,keep):
    num=E[keep].sum(0); den=obs[keep].sum(0); ok=use&(den>0); return float(np.sqrt((num[ok]/den[ok]).mean()))
def curve(E,obs,use,order):
    n=len(order); cov={}
    for c in (1.0,0.9,0.8,0.7,0.5):
        nk=int(np.ceil(c*n)); keep=np.zeros(n,bool); keep[order[n-nk:]]=True; cov[c]=ret_score(E,obs,use,keep)
    xs=[];ys=[]
    for nk in range(int(np.ceil(.5*n)),n+1):
        keep=np.zeros(n,bool); keep[order[n-nk:]]=True; xs.append(nk/n); ys.append(ret_score(E,obs,use,keep))
    xs=np.array(xs); ys=np.array(ys)
    aurc=float(np.sum((ys[1:]+ys[:-1])/2*np.diff(xs))/(xs[-1]-xs[0]))
    return cov,aurc
def nb(new,base,r):
    new=np.asarray(new);base=np.asarray(base); d=(base-new)/base*100; m=d.mean(); se=np.sqrt((1/5+r)*d.var(ddof=1)); t=m/se
    return float((base.mean()-new.mean())/base.mean()*100), int((new<base).sum()), float(t), float(2*tdist.sf(abs(t),4))
SCID={'new_cell':0,'tissue':1}
for sc in ('new_cell','tissue'):
    pc=list(csv.DictReader(open(f'{OUT}/B/percell_{sc}.csv')))
    au={}; 
    for x in csv.DictReader(open(f'{OUT}/B/aurc_{sc}.csv')): au.setdefault((x['arm'],x['proxy']),[None]*5)[int(x['fold'])]=float(x['aurc'])
    rc={}
    for x in csv.DictReader(open(f'{OUT}/B/riskcov_{sc}.csv')): rc.setdefault((x['arm'],x['proxy'],float(x['coverage'])),[None]*5)[int(x['fold'])]=float(x['score'])
    AU={}; C8={}; C1={}; ratios=[]; maxu=0; maxau=0; maxrc=0
    for f in range(5):
        Z=np.load(f'{MX}/{sc}/fold-{f}_predictions.npz',allow_pickle=False); G=np.load(f'{EXT}/{sc}/fold-{f}/outer/geometry.npz',allow_pickle=False)
        fit,held=Z['fit'],Z['held']; n,nq=len(fit),len(held); assert np.array_equal(G['fit'],fit) and np.array_equal(G['query'],held)
        truth,sd,use=Z['truth'],Z['train_sd'],Z['evaluated']; obs=np.isfinite(truth)
        # independent truth/sd check
        assert np.nanmax(np.abs(S['auc'][np.ix_(held,Z['panel'])]-truth))==0 and np.max(np.abs(np.maximum(np.nanstd(S['auc'][np.ix_(fit,Z['panel'])],axis=0),1e-8)-sd))==0
        ratios.append(nq/n)
        KM,HM=G['m_only_kernel'][:n,:n],G['m_only_cross'][:,:n]
        # targets (fit only)
        xf=X[fit]; g=np.argsort(-xf.var(0),kind='stable')[:3000]; zr=zstd(xf[:,g])[:,:512]
        p=S['p']; idx=np.flatnonzero(np.isfinite(p[fit]).mean(0)>=.5); pf=p[np.ix_(fit,idx)]; mean=np.nanmean(pf,axis=0); vf=np.where(np.isfinite(pf),pf,mean)
        tsel=np.sort(np.argsort(-vf.var(0),kind='stable')[:512]); zp=zstd(vf)[:,tsel]
        seed=seed_of('oof',sc,f,'outer')
        u={}
        for layer,Zt in (('RNA',zr),('P',zp)):
            lam,sq=submodels(KM,HM,Zt,seed); u[f'u1_v{layer}']=sq.std(0,ddof=0).mean(1)
        u['u2']=1-HM.max(1)
        mf=S['m'][fit]; mu=mf.mean(0); msd=mf.std(0); msd=np.where(msd<1e-8,1.,msd)
        D=cdist((S['m'][held]-mu)/msd,(mf-mu)/msd); u['u3']=np.sort(D,axis=1)[:,:10].mean(1)
        # compare proxies with percell
        rows=[x for x in pc if int(x['fold'])==f]; assert [int(x['cell']) for x in rows]==held.tolist()
        for k,v in u.items(): maxu=max(maxu,float(np.max(np.abs(v-np.array([float(x[k]) for x in rows])))))
        orders={k:np.argsort(-v,kind='stable') for k,v in u.items()}
        for s in range(5): orders[f'random_s{s}']=np.random.default_rng([20261009,SCID[sc],f,5,s]).permutation(nq)
        for arm in ('M','M+vRNA','M+vP','M+RNA','M+P'):
            E=np.where(obs,((Z['pred__'+arm]-truth)/sd)**2,0.); eo=obs&use[None,:]; ecell=np.where(eo,E,0.).sum(1)/np.maximum(eo.sum(1),1)
            maxu=max(maxu,float(np.max(np.abs(ecell-np.array([float(x['e_'+arm]) for x in rows])))))
            od=dict(orders); od['oracle']=np.argsort(-ecell,kind='stable')
            for px,o in od.items():
                cov,a=curve(E,obs,use,o); AU.setdefault((arm,px),[0]*5)[f]=a; C8.setdefault((arm,px),[0]*5)[f]=cov[0.8]; C1.setdefault((arm,px),[0]*5)[f]=cov[1.0]
                maxau=max(maxau,abs(a-au[(arm,px)][f])); maxrc=max(maxrc,max(abs(cov[c]-rc[(arm,px,c)][f]) for c in cov))
    r=float(np.mean(ratios))
    print(f'== {sc}: r={r:.4f} (saved {summ["B"][sc]["nb_ratio"]:.4f}); max|proxy/e diff vs percell|={maxu:.2e}; max|AURC diff vs aurc.csv|={maxau:.2e}; max|riskcov diff|={maxrc:.2e}')
    def A_(arm,px): return np.mean([AU[(arm,f'random_s{s}')] for s in range(5)],axis=0) if px=='random' else np.array(AU[(arm,px)])
    def C_(dct,arm,px): return np.mean([dct[(arm,f'random_s{s}')] for s in range(5)],axis=0) if px=='random' else np.array(dct[(arm,px)])
    for arm in ('M+vRNA','M+vP','M','M+RNA','M+P'):
        for px in ('u1_vRNA','u1_vP','u2','u3','oracle'):
            imp,w,t,p=nb(A_(arm,px),A_(arm,'random'),r); full=C_(C1,arm,px); c8=C_(C8,arm,px); rel=(full.mean()-c8.mean())/full.mean()*100
            B1=imp>0 and w>=4 and p<0.10; B2=rel>=1.0
            sv=summ['B'][sc]['comparisons'][f'{arm}|{px}_vs_random']
            flag='' if (abs(imp-sv['improvement_pct'])<1e-6 and abs(p-sv['nb_p'])<1e-6 and abs(rel-sv['cov08_rel_reduction_pct'])<1e-6 and B1==sv['B1'] and B2==sv['B2']) else '  <-- MISMATCH'
            if arm in ('M+vRNA','M+vP') or px in ('u3',):
                print(f'  {arm:7s} {px:8s} AURC {A_(arm,px).mean():.5f} vs rand {A_(arm,"random").mean():.5f} imp%={imp:+.3f} wins={w}/5 t={t:+.3f} p={p:.4f} c0.8rel%={rel:+.3f} (c1.0 {full.mean():.4f} -> c0.8 {c8.mean():.4f}) B1={B1} B2={B2}{flag}')
            elif flag: print('  MISMATCH',arm,px)
        if arm in ('M+vRNA','M+vP'):
            for px in ('u1_vRNA','u1_vP','u3'):
                g1=(C_(C1,'M',px)-C_(C1,arm,px)); g8=(C_(C8,'M',px)-C_(C8,arm,px)); print(f'    gain {arm} {px}: full {g1.mean():+.4f} -> c0.8 {g8.mean():+.4f}; larger folds {(g8>g1).sum()}/5')

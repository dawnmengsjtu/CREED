import numpy as np, csv, json, sys
from scipy.stats import t as student_t
R='/public/home/mengxl/creed-research-runs/acquisition-probe-rna-20261009-0001/output'
MX='/public/home/mengxl/creed-research-runs/crosslayer-matrix-20261009-0001/output'
def score(truth,pred,sd,use):
    obs=np.isfinite(truth); n=obs.sum(0)
    mse=np.where(obs,((pred-truth)/sd)**2,0).sum(0)/np.maximum(n,1)
    return float(np.sqrt(mse[use].mean()))
def nb(new,base,ratio):
    n,b=np.asarray(new),np.asarray(base); d=(b-n)/b*100; imp=(b.mean()-n.mean())/b.mean()*100
    se=np.sqrt((1/5+ratio)*d.var(ddof=1)); t=d.mean()/se; p=2*student_t.sf(abs(t),4)
    return imp,int((n<b).sum()),t,p
out={}
for sc in ('new_cell','tissue'):
    d=f'{R}/{sc}_main'
    csvrows=list(csv.DictReader(open(f'{d}/fold_metrics.csv')))
    csvd={(r['head'],r['policy'],r['k'],int(r['fold'])):float(r['score']) for r in csvrows}
    keys=['none|m_only|M','none|matrix_vRNA|ref','none|null_drug|none','H_meas|measured_all|all','H_virt|measured_all|all','H_meas|virtual|0','H_virt|virtual|0']
    for h in ('H_meas','H_virt'):
        for k in (16,32):
            for p in ('adaptive','static','variance','mean_expr')+tuple(f'random_s{s}' for s in range(5)):
                keys.append(f'{h}|{p}|{k}')
    fs={k:[] for k in keys}; fs_csvdiff=0.; ratios=[]; mxdiff={'M+RNA':0.,'M+vRNA':0.}
    for f in range(5):
        P=np.load(f'{d}/fold-{f}_predictions.npz')
        truth,sd,use=P['truth'],P['train_sd'],P['evaluated']
        ratios.append(len(P['held'])/len(P['fit']))
        # cross-check truth/sd/use against extra-supervision predictions.npz
        E=np.load(f'/public/home/mengxl/creed-research-runs/extra-supervision-20261008-0001/output/{sc}/fold-{f}/predictions.npz')
        assert np.array_equal(E['evaluated'],use) and np.allclose(E['train_sd'],sd) and np.array_equal(np.isfinite(E['truth']),np.isfinite(truth)) and np.nanmax(np.abs(E['truth']-truth))==0
        assert np.max(np.abs(P['none|m_only|M']-E['original__m_only']))<=1e-8
        M=np.load(f'{MX}/{sc}/fold-{f}_predictions.npz')
        mxdiff['M+RNA']=max(mxdiff['M+RNA'],float(np.max(np.abs(P['H_meas|measured_all|all']-M['pred__M+RNA']))))
        mxdiff['M+vRNA']=max(mxdiff['M+vRNA'],float(np.max(np.abs(P['none|matrix_vRNA|ref']-M['pred__M+vRNA']))))
        for k in keys:
            s=score(truth,P[k],sd,use); fs[k].append(s)
            h,p,kk=k.split('|'); fs_csvdiff=max(fs_csvdiff,abs(s-csvd[(h,p,kk,f)]))
    ratio=float(np.mean(ratios))
    for h in ('H_meas','H_virt'):
        for k in (16,32):
            fs[f'{h}|random_mean|{k}']=list(np.mean([fs[f'{h}|random_s{s}|{k}'] for s in range(5)],axis=0))
            for f in range(5):
                fs_csvdiff=max(fs_csvdiff,abs(fs[f'{h}|random_mean|{k}'][f]-csvd[(h,'random_mean',str(k),f)]))
    print('==',sc,'ratio',ratio,'ratios',np.round(ratios,4).tolist(),'max|my score - csv|',fs_csvdiff,'matrix pred maxdiff',mxdiff)
    m=np.array(fs['none|m_only|M']); T=np.array(fs['H_meas|measured_all|all']); V=np.array(fs['none|matrix_vRNA|ref'])
    print('gain% headroom',((m-T)/m*100).mean(),'recovered',((m-V)/m*100).mean())
    for k in ['none|m_only|M','H_meas|measured_all|all','none|matrix_vRNA|ref','H_virt|virtual|0','H_meas|virtual|0','H_virt|measured_all|all']+[f'{h}|{p}|{k}' for h in ('H_meas','H_virt') for k in (16,32) for p in ('adaptive','static','random_mean','variance','mean_expr')]:
        print(f'{k:32s} mean={np.mean(fs[k]):.6f} folds={np.round(fs[k],4).tolist()}')
    teacher=T.mean()
    for h in ('H_meas','H_virt'):
        k0=np.mean(fs[f'{h}|virtual|0'])
        for p in ('adaptive','static','random_mean'):
            for k in (16,32):
                print(f'recovery {h} {p} k{k}: {(k0-np.mean(fs[f"{h}|{p}|{k}"]))/(k0-teacher)*100:.2f}%')
    for h in ('H_meas','H_virt'):
        for k in (16,32):
            for b in ('static','random_mean','variance','mean_expr'):
                imp,won,t,p=nb(fs[f'{h}|adaptive|{k}'],fs[f'{h}|{b}|{k}'],ratio)
                print(f'NB {h} k{k} adaptive vs {b}: imp={imp:+.4f}% won={won}/5 t={t:+.3f} p={p:.4f}')
A=json.load(open(f'{MX}/analysis.json')) if __import__('os').path.exists(f'{MX}/analysis.json') else None
print('matrix analysis.json keys', list(A.keys())[:20] if A else 'not found at output/analysis.json')

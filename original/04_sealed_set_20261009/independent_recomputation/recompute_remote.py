
import os
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[v]='8'
import json, csv, glob, hashlib, re, sys
import numpy as np
from scipy.stats import spearmanr, binomtest
RUN='/public/home/mengxl/creed-research-runs/internal-final-20261009-0001'
RAW='/public/home/mengxl/creed-research-runs/_raw_whitelist_20261009'
SRC='/public/home/mengxl/creed-research-runs/gdsc-metabolite-full-protein-20260908-0001/output/result/source-inputs.npz'
FIN = ["SIDM00079", "SIDM00083", "SIDM00084", "SIDM00085", "SIDM00087", "SIDM00113", "SIDM00116", "SIDM00120", "SIDM00123", "SIDM00193", "SIDM00212", "SIDM00214", "SIDM00220", "SIDM00238", "SIDM00274", "SIDM00279", "SIDM00282", "SIDM00286", "SIDM00325", "SIDM00346", "SIDM00349", "SIDM00353", "SIDM00356", "SIDM00363", "SIDM00382", "SIDM00401", "SIDM00405", "SIDM00421", "SIDM00425", "SIDM00435", "SIDM00446", "SIDM00447", "SIDM00449", "SIDM00453", "SIDM00457", "SIDM00463", "SIDM00466", "SIDM00478", "SIDM00484", "SIDM00493", "SIDM00505", "SIDM00506", "SIDM00515", "SIDM00521", "SIDM00523", "SIDM00524", "SIDM00528", "SIDM00556", "SIDM00570", "SIDM00594", "SIDM00602", "SIDM00606", "SIDM00608", "SIDM00609", "SIDM00614", "SIDM00658", "SIDM00670", "SIDM00677", "SIDM00678", "SIDM00686", "SIDM00695", "SIDM00702", "SIDM00704", "SIDM00706", "SIDM00713", "SIDM00715", "SIDM00734", "SIDM00742", "SIDM00748", "SIDM00749", "SIDM00763", "SIDM00772", "SIDM00782", "SIDM00783", "SIDM00796", "SIDM00841", "SIDM00852", "SIDM00853", "SIDM00860", "SIDM00866", "SIDM00875", "SIDM00881", "SIDM00885", "SIDM00887", "SIDM00898", "SIDM00899", "SIDM00925", "SIDM00939", "SIDM00947", "SIDM00950", "SIDM00955", "SIDM00961", "SIDM00986", "SIDM00989", "SIDM00997", "SIDM01003", "SIDM01016", "SIDM01020", "SIDM01023", "SIDM01025", "SIDM01031", "SIDM01036", "SIDM01059", "SIDM01064", "SIDM01068", "SIDM01071", "SIDM01072", "SIDM01076", "SIDM01084", "SIDM01086", "SIDM01087", "SIDM01088", "SIDM01098", "SIDM01105", "SIDM01124", "SIDM01127", "SIDM01160", "SIDM01169", "SIDM01173", "SIDM01181", "SIDM01186", "SIDM01194", "SIDM01197"]
OUT = {}
def shalist(ids): return hashlib.sha256(('\n'.join(sorted(ids))+'\n').encode()).hexdigest()
OUT['fin_sha']=shalist(FIN)
# ---------------- frozen model / train identity
Z=np.load(RUN+'/frozen_models/model.npz',allow_pickle=False)
S=np.load(SRC,allow_pickle=False)
train_ids=Z['sample_ids'].astype(str).tolist()
OUT['train']=dict(n=len(train_ids), sha_sorted=shalist(train_ids), equals_source_inputs=bool(np.array_equal(Z['sample_ids'].astype(str),S['sample_ids'].astype(str))),
                  overlap_with_fin=len(set(train_ids)&set(FIN)))
drug_ids=S['drug_ids'].astype(str).tolist()
auc421=S['auc']
def nanstd0(a):
    obs=np.isfinite(a); cnt=obs.sum(0); mean=np.where(obs,a,0).sum(0)/np.maximum(cnt,1)
    return np.sqrt(np.where(obs,(a-mean)**2,0).sum(0)/np.maximum(cnt,1))
my_sd_drug=np.maximum(nanstd0(auc421),1e-8); my_obs_drug=np.isfinite(auc421).sum(0)
OUT['train_sd_drug_vs_frozen']=float(np.max(np.abs(my_sd_drug-Z['drug_train_sd'])))
OUT['train_obs_drug_vs_frozen']=bool(np.array_equal(my_obs_drug,Z['drug_train_obs']))
cr_sd=Z['crispr_train_sd']; cr_obs=Z['crispr_train_obs']
# ---------------- re-derive truth AUC for the 123 from GDSC csv (123 rows only, as sealed_io.gdsc_extract)
W=set(FIN); vals={}; conflicts=set(); drugs_seen=set(); matched=0
with open(RAW+'/GDSC2_fitted_dose_response_24Jul22.csv',encoding='utf-8-sig',newline='') as fh:
    rd=csv.DictReader(fh)
    for row in rd:
        sid=row['SANGER_MODEL_ID']
        if sid not in W: continue
        assert row['DATASET']=='GDSC2'; matched+=1
        d=row['DRUG_ID']; drugs_seen.add(d)
        try: rec=tuple(float(row[k]) for k in ('AUC','LN_IC50','MIN_CONC','MAX_CONC'))
        except ValueError: continue
        if not np.isfinite(rec).all() or not (0<rec[2]<=rec[3]): continue
        key=(sid,d)
        if key in vals:
            if vals[key]!=rec: conflicts.add(key)
        else: vals[key]=rec
pos={d:j for j,d in enumerate(drug_ids)}; ridx={s:i for i,s in enumerate(FIN)}
T=np.full((123,len(drug_ids)),np.nan); outside=set()
for (sid,d),rec in vals.items():
    if (sid,d) in conflicts: continue
    if d in pos: T[ridx[sid],pos[d]]=rec[0]
    else: outside.add(d)
OUT['gdsc_rederive']=dict(matched_rows=matched, conflicts=len(conflicts), drugs_outside_295=sorted(outside), observed_pairs=int(np.isfinite(T).sum()), samples_with_response=int(np.isfinite(T).any(1).sum()))
# ---------------- tissues for the 123 (mmc2 identity sheet, metadata only)
import openpyxl
wb=openpyxl.load_workbook(RAW+'/mmc2.xlsx',read_only=True); ws=wb['Cell line level sample info']; rows=ws.iter_rows(values_only=True); next(rows)
hdr=[str(c) if c is not None else '' for c in next(rows)]; ci={k:hdr.index(k) for k in ('model_id','Tissue_type')}
tis={}
for r in rows:
    if r is None or r[ci['model_id']] is None: continue
    sid=str(r[ci['model_id']]).strip()
    if sid in W: tis[sid]='' if r[ci['Tissue_type']] is None else str(r[ci['Tissue_type']]).strip()
wb.close()
assert len(tis)==123
tissues=np.array([tis[s] for s in FIN])
toks=np.array([x if x else f'__unknown_q_{i}' for i,x in enumerate(tissues)])
OUT['eval_tissue_counts']={k:int(v) for k,v in zip(*np.unique(tissues,return_counts=True))}
# ---------------- my metrics on every arm npz
def my_metrics(pred,truth,sd,use):
    obs=np.isfinite(truth); n=obs.sum(0)
    e=np.where(obs,((pred-truth)/sd)**2,0.)
    mse=e.sum(0)/np.maximum(n,1)
    rho=np.full(truth.shape[1],np.nan)
    for j in np.flatnonzero(use):
        o=obs[:,j]
        if o.sum()>=2 and np.std(truth[o,j])>0 and np.std(pred[o,j])>1e-15:
            rho[j]=spearmanr(truth[o,j],pred[o,j]).statistic
    pm=np.where(obs,pred,0).sum(0)/np.maximum(n,1); tm=np.where(obs,truth,0).sum(0)/np.maximum(n,1)
    bias=(pm-tm)/sd
    return dict(score=float(np.sqrt(mse[use].mean())), spearman=float(np.nanmean(rho[use])) if np.isfinite(rho[use]).any() else float('nan'),
                centered_mse=float((mse-bias**2)[use].mean()), mse=mse, e=e, obs=obs, rho=rho)
summ=json.load(open(RUN+'/output/summary.json'))
prog=[json.loads(l) for l in open(RUN+'/output/progress.jsonl')]
RES={}; ARMS={}
for ep in ('drug','crispr'):
    files=sorted(glob.glob(f'{RUN}/output/arms/{ep}__*.npz'))
    ARMS[ep]={}; truth0=None; sd0=None; use0=None
    for f in files:
        z=np.load(f,allow_pickle=False); label=str(z['label'])
        arm=os.path.basename(f)[len(ep)+2:-4].replace('PLUS','+')
        truth=z['truth']; pred=z['pred']; sd=z['train_sd']
        if truth0 is None: truth0, sd0 = truth, sd
        same_truth=bool(np.array_equal(np.isnan(truth),np.isnan(truth0)) and np.nanmax(np.abs(truth-truth0))==0) and bool(np.array_equal(sd,sd0))
        if ep=='drug':
            my_sd=my_sd_drug; tr_obs=my_obs_drug
        else:
            my_sd=cr_sd; tr_obs=cr_obs
        use=(np.isfinite(truth).sum(0)>=5)&(tr_obs>=20)
        m=my_metrics(pred,truth,my_sd,use)
        sj=summ['analysis'][ep]['arms'][arm]; pj=[p for p in prog if p['endpoint']==ep and p['arm']==arm][0]
        ARMS[ep][arm]=dict(my_score=m['score'],my_spearman=m['spearman'],my_cmse=m['centered_mse'],
            npz_score=float(z['score']),summary_score=sj['score'],progress_score=pj['score'],
            npz_spearman=float(z['spearman']),summary_spearman=sj['spearman'],
            npz_cmse=float(z['centered_mse']),summary_cmse=sj['centered_mse'],
            d_score=abs(m['score']-sj['score']),d_score_npz=abs(m['score']-float(z['score'])),d_score_prog=abs(m['score']-pj['score']),
            d_spearman=abs(m['spearman']-sj['spearman']) if sj['spearman'] is not None else None,
            d_cmse=abs(m['centered_mse']-sj['centered_mse']),
            d_mse_max=float(np.max(np.abs(m['mse']-z['mse']))), d_rho_max=float(np.nanmax(np.abs(m['rho']-z['rho']))) if np.isfinite(m['rho']).any() else 0.,
            use_equal=bool(np.array_equal(use,z['evaluated'])), n_use=int(use.sum()), sd_equal=float(np.max(np.abs(my_sd-sd))), same_truth_as_first=same_truth, label=label, n_cells=int(truth.shape[0]))
        RES.setdefault(ep,{})[arm]=dict(m=m,use=use)
    if ep=='drug':
        OUT['truth_rederive_vs_npz']=dict(nan_equal=bool(np.array_equal(np.isnan(T),np.isnan(truth0))), max_abs_diff=float(np.nanmax(np.abs(T-truth0))) if np.isfinite(T).any() else None, n_obs_npz=int(np.isfinite(truth0).sum()))
OUT['arms']=ARMS
# ---------------- bootstrap, my own implementation (same seed, same draw order: per b, cell draws per stratum in np.unique order, then drug draws)
B=2000; SEED=20261009
STAT={}
for ep in ('drug','crispr'):
    mets={a:v['m'] for a,v in RES[ep].items()}; use=RES[ep]['M']['use']; U=np.flatnonzero(use)
    strata=[np.flatnonzero(toks==t) for t in np.unique(toks)]
    rng=np.random.default_rng(SEED); n=len(toks)
    Cw=np.zeros((B,n)); Wd=np.zeros((B,len(use)))
    for b in range(B):
        for s in strata:
            Cw[b,s]+=np.bincount(rng.integers(0,len(s),len(s)),minlength=len(s))
        Wd[b,U]=np.bincount(rng.integers(0,len(U),len(U)),minlength=len(U))
    def bscore(m):
        num=Cw@m['e']; den=Cw@m['obs'].astype(float)
        mse=np.divide(num,den,out=np.zeros_like(num),where=den>0)
        ww=Wd*(use[None]&(den>0))
        return np.sqrt((ww*mse).sum(1)/ww.sum(1))
    bs={a:bscore(m) for a,m in mets.items()}
    def gain(b,a): return (b-a)/b*100.
    def ci(x):
        x=np.asarray(x,float); x=x[np.isfinite(x)]
        return dict(lo=float(np.percentile(x,2.5)),hi=float(np.percentile(x,97.5)),prob_gt0=float(np.mean(x>0)),median=float(np.median(x)))
    def sign(a,b):
        A,Bm=mets[a]['mse'][use],mets[b]['mse'][use]; w=int((A<Bm).sum()); l=int((A>Bm).sum()); t=int((A==Bm).sum())
        return dict(wins=w,losses=l,ties=t,frac=w/(w+l),p=float(binomtest(w,w+l,.5).pvalue))
    comp={}
    def cmp(name,a,b):
        if a in mets and b in mets:
            comp[name]=dict(gain_pct=gain(mets[b]['score'],mets[a]['score']),boot=ci(gain(bs[b],bs[a])),sign=sign(a,b))
    for y in ('RNA','P'):
        cmp(f'M+v{y}_vs_M',f'M+v{y}','M'); cmp(f'M+v{y}_shuf_vs_M',f'M+v{y}_shuf','M'); cmp(f'M+v{y}_vs_shuf',f'M+v{y}',f'M+v{y}_shuf'); cmp(f'M+{y}_vs_M',f'M+{y}','M')
    cmp('M+Pval_vs_M','M+Pval','M')
    if ep=='drug':
        cmp('M+T_vs_M','M+T','M'); cmp('T_vs_M','T','M'); cmp('M+vRNA+T_vs_M','M+vRNA+T','M'); cmp('M+vRNA+T_vs_M+T','M+vRNA+T','M+T'); cmp('M+P_graph_vs_M','M+P_graph','M')
    for k in (16,32,64): cmp(f'M+P{k}_vs_M',f'M+P{k}','M'); cmp(f'M+G{k}_vs_M',f'M+G{k}','M')
    share={}
    for y in ('RNA','P'):
        room=gain(mets['M']['score'],mets[f'M+{y}']['score']); rec=gain(mets['M']['score'],mets[f'M+v{y}']['score'])
        broom=gain(bs['M'],bs[f'M+{y}']); brec=gain(bs['M'],bs[f'M+v{y}'])
        with np.errstate(divide='ignore',invalid='ignore'): bsh=np.where(np.abs(broom)>1e-12,brec/broom,np.nan)
        share[y]=dict(room_pct=room,recovered_pct=rec,share=rec/room,share_boot=ci(bsh))
    Rk={}
    for layer,teacher in (('P','M+Pval'),('G','M+RNA')):
        den=mets['M']['score']-mets[teacher]['score']; bden=bs['M']-bs[teacher]
        for k in (16,32,64):
            a=f'M+{layer}{k}'
            with np.errstate(divide='ignore',invalid='ignore'): br=np.where(np.abs(bden)>1e-12,(bs['M']-bs[a])/bden,np.nan)
            Rk[a]=dict(teacher=teacher,R=(mets['M']['score']-mets[a]['score'])/den,R_boot=ci(br))
        for k in (16,32):
            names=[f'M+{layer}rand{k}_s{s}' for s in range(5)]
            sc=float(np.mean([mets[a]['score'] for a in names])); bsc=np.mean([bs[a] for a in names],axis=0)
            with np.errstate(divide='ignore',invalid='ignore'): br=np.where(np.abs(bden)>1e-12,(bs['M']-bsc)/bden,np.nan)
            Rk[f'M+{layer}rand{k}_mean']=dict(teacher=teacher,R=(mets['M']['score']-sc)/den,R_boot=ci(br),score_mean=sc)
    arms_boot={a:ci(bs[a]) for a in mets}
    STAT[ep]=dict(n_strata=len(strata),n_use=int(use.sum()),comparisons=comp,share=share,panel_recovery=Rk,arms_boot=arms_boot,
                  arm_gain_vs_M={a:dict(gain_pct=gain(mets['M']['score'],mets[a]['score']),boot=ci(gain(bs['M'],bs[a])),sign=sign(a,'M')) for a in mets if a!='M'})
    # Monte-Carlo sensitivity: 3 alternative seeds for the borderline quantities
    alt={}
    for sd_ in (1,2,3):
        rng2=np.random.default_rng(sd_); Cw2=np.zeros((B,n)); Wd2=np.zeros((B,len(use)))
        for b in range(B):
            for s in strata: Cw2[b,s]+=np.bincount(rng2.integers(0,len(s),len(s)),minlength=len(s))
            Wd2[b,U]=np.bincount(rng2.integers(0,len(U),len(U)),minlength=len(U))
        def bscore2(m):
            num=Cw2@m['e']; den=Cw2@m['obs'].astype(float); mse=np.divide(num,den,out=np.zeros_like(num),where=den>0); ww=Wd2*(use[None]&(den>0)); return np.sqrt((ww*mse).sum(1)/ww.sum(1))
        b2={a:bscore2(mets[a]) for a in ('M','M+vRNA','M+vP','M+RNA','M+P','M+Pval','M+P32','M+P64','M+G32','M+G64','M+vRNA_shuf','M+vP_shuf')+(('M+T','M+P_graph') if ep=='drug' else ())}
        g=lambda a,b_: gain(b2[b_],b2[a])
        with np.errstate(divide='ignore',invalid='ignore'):
            alt[sd_]=dict(vRNA=ci(g('M+vRNA','M')),vP=ci(g('M+vP','M')),shareRNA=ci(g('M+vRNA','M')/g('M+RNA','M')),shareP=ci(g('M+vP','M')/g('M+P','M')),
                          R_P32=ci((b2['M']-b2['M+P32'])/(b2['M']-b2['M+Pval'])),R_P64=ci((b2['M']-b2['M+P64'])/(b2['M']-b2['M+Pval'])),
                          R_G32=ci((b2['M']-b2['M+G32'])/(b2['M']-b2['M+RNA'])),R_G64=ci((b2['M']-b2['M+G64'])/(b2['M']-b2['M+RNA'])),
                          vRNA_shuf=ci(g('M+vRNA_shuf','M')),vP_shuf=ci(g('M+vP_shuf','M')))
            if ep=='drug': alt[sd_].update(MT=ci(g('M+T','M')),Pgraph=ci(g('M+P_graph','M')))
    STAT[ep]['alt_seeds']=alt
OUT['stats']=STAT
# ---------------- SIDM tokens appearing anywhere in output/
toks_found=set()
for f in glob.glob(RUN+'/output/*')+glob.glob(RUN+'/output/arms/*'):
    if os.path.isdir(f): continue
    if f.endswith('.npz'):
        z=np.load(f,allow_pickle=False); toks_found|=set(re.findall(r'SIDM\d{5}',str(z['label'])))
    else:
        toks_found|=set(re.findall(r'SIDM\d{5}',open(f,encoding='utf-8',errors='replace').read()))
for f in glob.glob(RUN+'/logs/sealed*.log'):
    toks_found|=set(re.findall(r'SIDM\d{5}',open(f,encoding='utf-8',errors='replace').read()))
OUT['sidm_tokens_in_output_and_sealed_logs']=sorted(toks_found)
# frozen manifest tokens
man=open(RUN+'/frozen_models/FROZEN_MANIFEST.json').read(); OUT['sidm_tokens_in_manifest']=sorted(set(re.findall(r'SIDM\d{5}',man)))
OUT['inner_losses_sha_ok']=hashlib.sha256(open(RUN+'/frozen_models/inner_losses.npz','rb').read()).hexdigest()
OUT['frozen_crispr_cols']=int(len(Z['crispr_cols']))
print(json.dumps(OUT,default=lambda o: None if (isinstance(o,float) and not np.isfinite(o)) else (o.item() if hasattr(o,'item') else str(o))))

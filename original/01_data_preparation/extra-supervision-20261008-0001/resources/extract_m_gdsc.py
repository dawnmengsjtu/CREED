"""Frozen candidate-only M/GDSC extraction; emits one NPZ on stdout.

Only candidate rows plus three explicitly permitted old-training parity rows
are decoded numerically. Other source rows are inspected only for identifiers.
No fitting, imputation, normalization, filtering by outcome, or protected reads.
"""
from pathlib import Path
from collections import Counter,defaultdict
import csv,hashlib,json,sys
import numpy as np

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def prefix_fields(raw,last):
    out=[];start=0;i=0;quoted=False
    while i<len(raw):
        c=raw[i]
        if c==34:
            if quoted and i+1<len(raw) and raw[i+1]==34:i+=2;continue
            quoted=not quoted
        if c==44 and not quoted:
            out.append(raw[start:i]);start=i+1
            if len(out)>last:return out
        i+=1
    out.append(raw[start:].rstrip(b'\r\n'));assert len(out)>last and not quoted
    return out

def text(v):return next(csv.reader([v.decode('utf-8-sig')]))[0].strip()
def number(v):
    if v.strip().lower() in ('','na','nan','null','none'):return np.nan
    return float(v)

def main(c):
    cand=c['candidates'];parity=c['parity'];allrows=cand+parity
    ids=[r['sample_id'] for r in allrows];extra=[r['sample_id'] for r in cand]
    assert len(extra)==19 and len(set(ids))==len(ids)
    assert not set(extra)&set(c['excluded_old608'])
    assert len(c['m_feature_ids'])==225 and len(c['drug_ids'])==295
    bypair={(r['ccle_id'],r['model_id']):r['sample_id'] for r in allrows};idx={s:i for i,s in enumerate(ids)}
    base=Path('/home/node/.openclaw/workspace/bioagent/data/raw')
    mp=base/'ccle_metabolomics/CCLE_metabolomics_20190502.csv';yp=base/'gdsc/GDSC2_fitted_dose_response_24Jul22.csv'
    assert digest(mp)==c['m_sha256'] and digest(yp)==c['gdsc_sha256']
    m=np.full((len(ids),225),np.nan);seen=set();source_m_rows=0
    with mp.open('rb') as f:
        header=next(csv.reader([f.readline().decode('utf-8-sig')]))
        assert header[:2]==['CCLE_ID','DepMap_ID'] and header[2:]==c['m_feature_ids']
        for raw in f:
            if not raw.strip():continue
            source_m_rows+=1;p=prefix_fields(raw,1);pair=(text(p[0]),text(p[1]));sid=bypair.get(pair)
            if sid is None:continue
            assert sid not in seen;row=next(csv.reader([raw.decode('utf-8-sig')]));assert len(row)==227
            m[idx[sid]]=np.array([number(v) for v in row[2:]],float);seen.add(sid)
    assert seen==set(ids) and not np.isinf(m).any()
    drugidx={d:i for i,d in enumerate(c['drug_ids'])};values={};conflicts=set();invalid=0;matched=0;outside=0;duplicate=0;all_y_rows=0;out_of_panel=0
    batches=defaultdict(set);release=Counter();datasets=Counter()
    with yp.open('rb') as f:
        header=next(csv.reader([f.readline().decode('utf-8-sig')]));assert header[5]=='SANGER_MODEL_ID' and header[7]=='DRUG_ID'
        for raw in f:
            if not raw.strip():continue
            all_y_rows+=1;p=prefix_fields(raw,7);sid=text(p[5]);drug=text(p[7])
            if sid not in idx:continue
            if drug not in drugidx:out_of_panel+=1;continue
            fields=next(csv.reader([raw.decode('utf-8-sig')]));assert len(fields)==len(header);row=dict(zip(header,fields));matched+=1
            assert row['DATASET']=='GDSC2';datasets[row['DATASET']]+=1;release[row['WEBRELEASE']]+=1;batches[sid].add(row['WEBRELEASE'])
            try:record=tuple(float(row[k]) for k in ('AUC','LN_IC50','MIN_CONC','MAX_CONC'))
            except ValueError:invalid+=1;continue
            if not np.isfinite(record).all() or not (0<record[2]<=record[3]):invalid+=1;continue
            if not 0<=record[0]<=1:outside+=1
            key=(sid,drug)
            if key in values:
                if values[key]!=record:conflicts.add(key)
                else:duplicate+=1
            else:values[key]=record
    arrays={name:np.full((len(ids),295),np.nan) for name in ('auc','ln_ic50','min_conc','max_conc')}
    for (sid,drug),v in values.items():
        if (sid,drug) in conflicts:continue
        for k,n in enumerate(arrays):arrays[n][idx[sid],drugidx[drug]]=v[k]
    n=len(cand)
    qa=dict(status='candidate_extracted_pending_independent_parity',training_performed=False,protected_numeric_rows_decoded=0,
        source_m_rows=source_m_rows,source_gdsc_rows=all_y_rows,source_sha256={'m':c['m_sha256'],'gdsc':c['gdsc_sha256']},
        m_transform='identity: original source values preserved; no additional log or normalization',auc_semantics='native fitted GDSC2 AUC; lower values retained without reinterpretation',
        qc='finite AUC/LN_IC50/MIN_CONC/MAX_CONC and 0<MIN_CONC<=MAX_CONC; conflicting duplicate keys excluded; out-of-[0,1] AUC retained and counted',
        rows=n,m_shape=[n,225],auc_shape=[n,295],m_missing=int(np.isnan(m[:n]).sum()),auc_observed=int(np.isfinite(arrays['auc'][:n]).sum()),auc_missing=int(np.isnan(arrays['auc'][:n]).sum()),
        qc_counts_including_parity=dict(numeric_source_rows=matched,invalid_rows=invalid,duplicate_same=duplicate,conflicting_keys=len(conflicts),outside_unit_auc_retained=outside,out_of_panel_rows_skipped=out_of_panel),
        source_batches={'metabolites':'CCLE_metabolomics_20190502; within-file acquisition batch not given in identity columns','GDSC_dataset':dict(datasets),'GDSC_WEBRELEASE':dict(release)},
        sample_qa=[dict(sample_id=r['sample_id'],model_id=r['model_id'],patient_id=r['patient_id'],tissue=r['tissue'],m_missing=int(np.isnan(m[i]).sum()),auc_observed=int(np.isfinite(arrays['auc'][i]).sum()),auc_missing=int(np.isnan(arrays['auc'][i]).sum()),gdsc_webrelease=sorted(batches[r['sample_id']]),status='candidate_numeric_available' if r['tissue'] else 'numeric_available_tissue_unresolved') for i,r in enumerate(cand)],
        unresolved_tissue_ids=[r['sample_id'] for r in cand if not r['tissue']],parity_sample_ids=[r['sample_id'] for r in parity],numeric_whitelist=ids)
    out=dict(m=m[:n],sample_ids=np.array(extra),m_feature_ids=np.array(c['m_feature_ids']),drug_ids=np.array(c['drug_ids']),tissues=np.array([r['tissue'] for r in cand]),model_ids=np.array([r['model_id'] for r in cand]),patient_ids=np.array([r['patient_id'] for r in cand]),ccle_ids=np.array([r['ccle_id'] for r in cand]),gdsc_webrelease=np.array(['|'.join(sorted(batches[r['sample_id']])) for r in cand]),parity_sample_ids=np.array(qa['parity_sample_ids']),parity_m=m[n:],qa_json=np.array(json.dumps(qa)))
    out.update({k:v[:n] for k,v in arrays.items()});out.update({'parity_'+k:v[n:] for k,v in arrays.items()})
    np.savez_compressed(sys.stdout.buffer,**out)
    print(json.dumps({k:qa[k] for k in ('rows','m_missing','auc_observed','auc_missing','unresolved_tissue_ids')}),file=sys.stderr)

if __name__=='__main__':main(globals()['CONFIG'] if 'CONFIG' in globals() else json.loads(Path(sys.argv[1]).read_text()))

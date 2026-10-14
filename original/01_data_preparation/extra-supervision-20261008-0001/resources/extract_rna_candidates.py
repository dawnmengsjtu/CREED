"""Whitelisted extra M/RNA and RNA/GDSC values on the frozen old RNA panel.

Uses a byte-level column gate before numeric decoding. Source bytes for other
cell lines are traversed but their assay fields are never decoded. Emits one
NPZ containing the union RNA block and indexed M/GDSC sub-blocks on stdout.
"""
import csv,gzip,hashlib,json,sys
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from extract_m_gdsc import digest,prefix_fields,text,number

def all_byte_fields(raw):
    # This source's numeric data rows ordinarily have no quoted commas. Fall
    # back to the quote-aware byte scanner if row width signals otherwise.
    return raw.rstrip(b'\r\n').split(b',')

def selected_text(raw):return text(raw) if raw.strip() else ''

def main(c):
    rows=c['candidates'];ids=[r['sample_id'] for r in rows];parity=c['rna_parity_ids'];allids=ids+parity
    assert len(ids)==402 and len(set(allids))==len(allids)
    assert not set(ids)&set(c['excluded_old608'])
    m_ids=c['m_rna_ids'];y_ids=c['rna_gdsc_ids'];assert len(m_ids)==66 and len(y_ids)==354
    root=Path('/home/node/.openclaw/workspace/bioagent/data/raw');rp=root/'ccle/CCLE_RNAseq_RSEM_Expected_Counts.csv.gz';mp=root/'ccle_metabolomics/CCLE_metabolomics_20190502.csv';yp=root/'gdsc/GDSC2_fitted_dose_response_24Jul22.csv'
    for p,k in ((rp,'rna_sha256'),(mp,'m_sha256'),(yp,'gdsc_sha256')):assert digest(p)==c[k]
    common=np.array(c['common_source_gene_indices'],int);assert len(common)==34985 and len(set(common))==len(common)
    slot={int(g):i for i,g in enumerate(common)};counts=np.full((len(allids),len(common)),np.nan);missing=np.zeros(len(allids),int);genes=[];symbols=[];batches=[]
    with gzip.open(rp,'rb') as f:
        head=[next(csv.reader([f.readline().decode('utf-8-sig')])) for _ in range(4)]
        assert [r[0] for r in head]==['model_id','model_name','dataset_name','gene_id'];width=len(head[0]);lookup={s:j for j,s in enumerate(head[0][2:],2)}
        columns=[lookup[s] for s in allids];batches=[head[2][j] for j in columns]
        for raw in f:
            if not raw.strip():continue
            flds=all_byte_fields(raw)
            if len(flds)!=width:flds=prefix_fields(raw,width-1)
            assert len(flds)==width
            gi=len(genes);genes.append(selected_text(flds[0]));symbols.append(selected_text(flds[1]))
            # Only the frozen selected samples are decoded here.
            vals=np.array([number(selected_text(flds[j])) for j in columns],float)
            assert not np.isinf(vals).any() and not np.any(vals<0)
            missing+=np.isnan(vals)
            if gi in slot:counts[:,slot[gi]]=vals
    assert len(genes)==37279 and len(set(genes))==len(genes)
    geneids=np.array(genes)[common];assert hashlib.sha256(('\n'.join(geneids.tolist())+'\n').encode()).hexdigest()==c['common_gene_ids_sha256']
    good=np.isfinite(counts).all(1)&(np.nansum(counts,1)>0);expr=np.full_like(counts,np.nan)
    expr[good]=np.log1p(counts[good]/counts[good].sum(1)[:,None]*1e6)
    ridx={s:i for i,s in enumerate(ids)};midx=np.array([ridx[s] for s in m_ids]);yidx=np.array([ridx[s] for s in y_ids])
    mr={r['sample_id']:r for r in rows if r['has_m']};bypair={(r['ccle_id'],r['model_id']):i for i,s in enumerate(m_ids) for r in [mr[s]]};m=np.full((66,225),np.nan);seen=set()
    with mp.open('rb') as f:
        header=next(csv.reader([f.readline().decode('utf-8-sig')]));assert header[2:]==c['m_feature_ids']
        for raw in f:
            if not raw.strip():continue
            p=prefix_fields(raw,1);pair=(text(p[0]),text(p[1]));i=bypair.get(pair)
            if i is None:continue
            assert i not in seen;v=next(csv.reader([raw.decode('utf-8-sig')]));assert len(v)==227;m[i]=[number(z) for z in v[2:]];seen.add(i)
    assert len(seen)==66 and not np.isinf(m).any()
    yi={s:i for i,s in enumerate(y_ids)};di={s:i for i,s in enumerate(c['drug_ids'])};values={};conflicts=set();invalid=0;outside=0;duplicate=0;matched=0;release=defaultdict(set)
    with yp.open('rb') as f:
        header=next(csv.reader([f.readline().decode('utf-8-sig')]));assert header[5]=='SANGER_MODEL_ID' and header[7]=='DRUG_ID'
        for raw in f:
            if not raw.strip():continue
            p=prefix_fields(raw,7);sid=text(p[5]);drug=text(p[7])
            if sid not in yi or drug not in di:continue
            v=next(csv.reader([raw.decode('utf-8-sig')]));row=dict(zip(header,v));assert row['DATASET']=='GDSC2';matched+=1;release[sid].add(row['WEBRELEASE'])
            try:record=tuple(float(row[k]) for k in ('AUC','LN_IC50','MIN_CONC','MAX_CONC'))
            except ValueError:invalid+=1;continue
            if not np.isfinite(record).all() or not (0<record[2]<=record[3]):invalid+=1;continue
            if not 0<=record[0]<=1:outside+=1
            key=(sid,drug)
            if key in values:
                if values[key]!=record:conflicts.add(key)
                else:duplicate+=1
            else:values[key]=record
    y={k:np.full((354,295),np.nan) for k in ('auc','ln_ic50','min_conc','max_conc')}
    for (sid,drug),v in values.items():
        if (sid,drug) in conflicts:continue
        for k,name in enumerate(y):y[name][yi[sid],di[drug]]=v[k]
    n=len(ids);qa=dict(status='candidate_extracted_pending_independent_parity',training_performed=False,protected_numeric_rows_decoded=0,union_samples=n,m_rna_samples=66,rna_gdsc_samples=354,rna_common_genes=len(common),rna_source_genes=len(genes),rna_source_sha256=c['rna_sha256'],m_source_sha256=c['m_sha256'],gdsc_source_sha256=c['gdsc_sha256'],common_gene_ids_sha256=c['common_gene_ids_sha256'],normalization='ln(1 + counts / sample_total_of_frozen_34985_common_assayed_genes * 1e6); sample-local; same panel and rule as original 421',rna_common_missing=int(np.isnan(counts[:n]).sum()),rna_full_source_missing=int(missing[:n].sum()),rna_valid_original_transform=int(good[:n].sum()),rna_source_batches=dict(Counter(batches[:n])),m_missing=int(np.isnan(m).sum()),auc_observed=int(np.isfinite(y['auc']).sum()),auc_missing=int(np.isnan(y['auc']).sum()),gdsc_qc=dict(numeric_rows=matched,invalid_rows=invalid,duplicate_same=duplicate,conflicting_keys=len(conflicts),outside_unit_auc_retained=outside),numeric_rna_whitelist=allids,numeric_m_whitelist=m_ids,numeric_gdsc_whitelist=y_ids,
        sample_qa=[dict(sample_id=s,model_id=rows[i]['model_id'],patient_id=rows[i]['patient_id'],tissue=rows[i]['tissue'],rna_source=batches[i],rna_source_missing=int(missing[i]),rna_common_missing=int(np.isnan(counts[i]).sum()),rna_valid_original_transform=bool(good[i]),has_m=rows[i]['has_m'],has_gdsc=rows[i]['has_gdsc'],auc_observed=int(np.isfinite(y['auc'][yi[s]]).sum()) if s in yi else 0,gdsc_webrelease=sorted(release[s]) if s in yi else []) for i,s in enumerate(ids)])
    out=dict(sample_ids=np.array(ids),model_ids=np.array([r['model_id'] for r in rows]),patient_ids=np.array([r['patient_id'] for r in rows]),tissues=np.array([r['tissue'] for r in rows]),rna_source_names=np.array(batches[:n]),gene_ids=geneids,gene_symbols=np.array(symbols)[common],common_source_gene_indices=common,counts_common=counts[:n],rna_log_cpm=expr[:n],rna_valid_original_transform=good[:n],rna_full_source_missing=missing[:n],m=m,m_sample_ids=np.array(m_ids),m_rna_row_indices=midx,m_feature_ids=np.array(c['m_feature_ids']),gdsc_sample_ids=np.array(y_ids),gdsc_rna_row_indices=yidx,drug_ids=np.array(c['drug_ids']),parity_sample_ids=np.array(parity),parity_rna_log_cpm=expr[n:],parity_rna_source_names=np.array(batches[n:]),qa_json=np.array(json.dumps(qa)),**y)
    np.savez_compressed(sys.stdout.buffer,**out)
    print(json.dumps({k:qa[k] for k in ('union_samples','rna_common_missing','rna_valid_original_transform','m_missing','auc_observed','rna_source_batches')}),file=sys.stderr)

if __name__=='__main__':main(globals()['CONFIG'] if 'CONFIG' in globals() else json.loads(Path(sys.argv[1]).read_text()))

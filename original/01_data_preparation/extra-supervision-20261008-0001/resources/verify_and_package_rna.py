"""Verify source-scale parity and package two aligned candidate input blocks.

No training. Reads only the newly whitelisted extraction and frozen old-train
parity rows. Does not open protected or source response files.
"""
import csv,hashlib,json
from pathlib import Path
from collections import Counter
import numpy as np

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def main():
    z=load(ROOT/'rna_candidates402.npz');ref=load(ROOT/'rna_old_train_parity_reference.npz');m19=load(ROOT/'m_gdsc_extra19.npz');c=json.loads((ROOT/'rna_config.json').read_text());qa=json.loads(str(z['qa_json']))
    ids=z['sample_ids'];assert len(set(ids))==len(ids)==402 and not set(ids)&set(c['excluded_old608'])
    np.testing.assert_array_equal(z['gene_ids'],ref['gene_ids']);np.testing.assert_array_equal(z['common_source_gene_indices'],ref['common_source_gene_indices']);np.testing.assert_array_equal(z['parity_sample_ids'],ref['sample_ids']);np.testing.assert_array_equal(z['parity_rna_source_names'],ref['rna_source_names'])
    np.testing.assert_allclose(z['parity_rna_log_cpm'],ref['rna_log_cpm'],atol=2e-14,rtol=2e-14)
    parity_error=float(np.max(np.abs(z['parity_rna_log_cpm']-ref['rna_log_cpm'])))
    counts=z['counts_common'];assert counts.shape==(402,34985) and np.isfinite(counts).all() and np.all(counts>=0) and np.all(z['rna_valid_original_transform'])
    expected=np.log1p(counts/counts.sum(1)[:,None]*1e6);np.testing.assert_allclose(z['rna_log_cpm'],expected,atol=2e-14,rtol=2e-14)
    mi=z['m_rna_row_indices'];yi=z['gdsc_rna_row_indices'];np.testing.assert_array_equal(ids[mi],z['m_sample_ids']);np.testing.assert_array_equal(ids[yi],z['gdsc_sample_ids']);np.testing.assert_array_equal(z['m_feature_ids'],m19['m_feature_ids']);np.testing.assert_array_equal(z['drug_ids'],m19['drug_ids'])
    assert z['m'].shape==(66,225) and z['auc'].shape==(354,295) and np.isfinite(z['m']).all()
    cross=[]
    for i,s in enumerate(m19['sample_ids']):
        mm=np.flatnonzero(z['m_sample_ids']==s);yy=np.flatnonzero(z['gdsc_sample_ids']==s)
        if len(mm):np.testing.assert_array_equal(z['m'][mm[0]],m19['m'][i])
        if len(yy):np.testing.assert_array_equal(z['auc'][yy[0]],m19['auc'][i])
        if len(mm) and len(yy):cross.append(str(s))
    assert len(cross)==18
    metadata={k:z[k] for k in ('sample_ids','model_ids','patient_ids','tissues','rna_source_names')}
    shared=dict(gene_ids=z['gene_ids'],gene_symbols=z['gene_symbols'],common_source_gene_indices=z['common_source_gene_indices'],rna_source_sha256=np.array(c['rna_sha256']),m_source_sha256=np.array(c['m_sha256']),gdsc_source_sha256=np.array(c['gdsc_sha256']),rna_normalization=np.array(qa['normalization']))
    mp=ROOT/'m_rna_extra66.npz';yp=ROOT/'rna_gdsc_extra354.npz'
    np.savez_compressed(mp,**{k:v[mi] for k,v in metadata.items()},m=z['m'],m_feature_ids=z['m_feature_ids'],rna_log_cpm=z['rna_log_cpm'][mi],rna_counts_common=counts[mi],**shared)
    np.savez_compressed(yp,**{k:v[yi] for k,v in metadata.items()},drug_ids=z['drug_ids'],rna_log_cpm=z['rna_log_cpm'][yi],rna_counts_common=counts[yi],**{k:z[k] for k in ('auc','ln_ic50','min_conc','max_conc')},**shared)
    blocks={}
    for name,ix,path in [('M_RNA',mi,mp),('RNA_GDSC',yi,yp)]:
        blocks[name]=dict(samples=len(ix),rna_genes=34985,rna_batches=dict(Counter(z['rna_source_names'][ix].tolist())),tissue_unknown_ids=ids[ix][z['tissues'][ix]==''].tolist(),missing_rna=0,path=path.name,sha256=sha(path))
    qa.update(status='PASS_extraction_QC_and_old_train_parity',parity=dict(old_train_samples=ref['sample_ids'].tolist(),old_train_batches=ref['rna_source_names'].tolist(),max_absolute_error=parity_error,tolerance=2e-14,cross_package_M_GDSC_exact_samples=cross),blocks=blocks,canonical_npz_sha256=sha(ROOT/'rna_candidates402.npz'),extractor_sha256=sha(ROOT/'extract_rna_candidates.py'),config_sha256=sha(ROOT/'rna_config.json'),verification_sha256=sha(Path(__file__)),candidate_use_boundary='Numeric eligibility established; unresolved tissues and all outer/inner held-out groups must still be excluded. New batches require explicit handling. No training or independence claim.')
    (ROOT/'rna_candidates_QA.json').write_text(json.dumps(qa,indent=2)+'\n')
    with (ROOT/'rna_candidate_sample_QA.tsv').open('w') as f:
        rows=qa['sample_qa'];w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
    print(json.dumps(dict(status=qa['status'],parity=qa['parity'],blocks=blocks,auc_observed=qa['auc_observed'],auc_missing=qa['auc_missing'],gdsc_qc=qa['gdsc_qc']),indent=2))

if __name__=='__main__':main()

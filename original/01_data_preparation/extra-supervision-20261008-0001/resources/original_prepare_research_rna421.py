"""Extract only the 421 development training rows of assayed RNA expected counts."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import numpy as np
from prepare_research_421 import prefix,sha


def log_cpm(counts):
    counts=np.asarray(counts,dtype=np.float64)
    if counts.ndim!=2 or not np.isfinite(counts).all() or np.any(counts<0):
        raise ValueError('RNA counts must be complete, finite and nonnegative; no silent imputation')
    total=counts.sum(1)
    if np.any(total<=0): raise ValueError('empty RNA library')
    return np.log1p(counts/total[:,None]*1e6)


def common_panel(counts,training_partitions):
    if np.isinf(counts).any() or np.any(counts<0): raise ValueError('invalid RNA values')
    common=np.isfinite(counts).all(0)
    for fit in training_partitions:
        if not np.array_equal(common,np.isfinite(counts[fit]).all(0)):
            raise ValueError('common assay panel cannot be determined from each training fold alone')
    return np.flatnonzero(common)


def main():
    parser=argparse.ArgumentParser()
    for arg in ['source','source-manifest','rna-source','existing-data','output']: parser.add_argument('--'+arg,type=Path,required=True)
    args=parser.parse_args()
    source_hash=sha(args.source)
    if source_hash!='468767f19ea2c935285c0d0877475ea97cb70140662b307f39e841d91a81e5d1': raise ValueError('unexpected source')
    provenance=json.loads(args.source_manifest.read_text())
    if provenance['artifact']['sha256']!=source_hash or provenance['structure']['layers']['L1_rna_expected_counts']['semantic']!='untransformed_expected_counts':
        raise ValueError('RNA semantics not established')
    previous=json.loads((args.existing_data/'manifest.json').read_text())
    for name,record in previous['files'].items():
        if sha(args.existing_data/name)!=record['sha256']: raise ValueError('existing package changed')
    ids=prefix(args.source,'sample_ids',421); genes=prefix(args.source,'l1_gene_ids'); symbols=prefix(args.source,'l1_symbols')
    if not np.array_equal(ids,np.load(args.existing_data/'sample_ids.npy',allow_pickle=False)) or len(set(ids))!=421:
        raise ValueError('sample order changed')
    if not np.all(prefix(args.source,'partition_labels',421)=='train') or genes.shape!=(37279,) or len(set(genes))!=37279:
        raise ValueError('partition or gene identity changed')
    counts=prefix(args.source,'l1_values',421)
    if counts.shape!=(421,37279): raise ValueError('wrong RNA shape')
    from sklearn.model_selection import KFold
    original_missing=np.isnan(counts).sum(1)
    common=common_panel(counts,[fit for fit,_ in KFold(5,shuffle=True,random_state=2663416544).split(counts)])
    raw_source_hash=sha(args.rna_source)
    if raw_source_hash!=provenance['sources']['L1_rna_expected_counts']['sha256']: raise ValueError('RNA source identity differs')
    with gzip.open(args.rna_source,'rt',encoding='utf-8-sig',newline='') as f:
        reader=csv.reader(f); headers=[next(reader) for _ in range(4)]
    if [h[0] for h in headers]!=['model_id','model_name','dataset_name','gene_id']: raise ValueError('unexpected RNA metadata')
    lookup={sid:i for i,sid in enumerate(headers[0])}
    sources=np.asarray([headers[2][lookup[sid]] for sid in ids],dtype=np.str_)
    source_profile=[dict(source=str(source),rows=int((sources==source).sum()),
        missing_genes_per_row=sorted(set(int(x) for x in original_missing[sources==source]))) for source in np.unique(sources)]
    all_gene_ids=genes.copy(); counts=counts[:,common]; genes=genes[common]; symbols=symbols[common]
    expression=log_cpm(counts)
    profile=dict(rows=421,source_genes=37279,genes=len(common),removed_noncommon_genes=37279-len(common),
        source_missing_values=int(original_missing.sum()),source_profile=source_profile,common_panel_identical_in_all_outer_training_folds=True,
        duplicate_samples=0,duplicate_gene_ids=0,
        missing_values=int(np.isnan(counts).sum()),negative_values=int((counts<0).sum()),
        all_zero_genes=int((counts.sum(0)==0).sum()),zero_fraction=float((counts==0).mean()),
        library_total_quantiles=np.quantile(counts.sum(1),[0,.25,.5,.75,1]).tolist(),
        normalization='ln(1 + counts / sample_total_of_common_assayed_genes * 1e6); not full-library CPM or TPM; sample-local',
        sample_grain='cell-line identifier matched assays, not established same-cell or same-aliquot co-assay',
        biological_replicates_established=False)
    args.output.mkdir(parents=True,exist_ok=False)
    for name,value in dict(sample_ids=ids,gene_ids=genes,gene_symbols=symbols,all_source_gene_ids=all_gene_ids,
        common_source_gene_indices=common,rna_source_names=sources,rna_log_cpm=expression).items():
        np.save(args.output/(name+'.npy'),np.ascontiguousarray(value),allow_pickle=False)
    (args.output/'quality.json').write_text(json.dumps(profile,indent=2))
    if sha(args.source)!=source_hash: raise ValueError('source changed during preparation')
    manifest=dict(scope='exploratory_training_421_measured_rna',source_sha256=source_hash,
        source_manifest_sha256=sha(args.source_manifest),existing_manifest_sha256=sha(args.existing_data/'manifest.json'),
        source_rna_sha256=provenance['sources']['L1_rna_expected_counts']['sha256'],preparation_sha256=sha(Path(__file__)),
        audit_values_decoded=False,final_access=False,scientific_claim=False,
        files={x.name:dict(sha256=sha(x),bytes=x.stat().st_size) for x in args.output.iterdir()})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)); print(json.dumps(dict(manifest=manifest,quality=profile)),flush=True)


if __name__=='__main__': main()

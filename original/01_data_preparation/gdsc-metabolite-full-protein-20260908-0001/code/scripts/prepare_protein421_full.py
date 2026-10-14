"""Recover full measured protein panel for training-prefix only; retain NaNs."""
import argparse,json
from pathlib import Path
import numpy as np
from prepare_research_421 import prefix,sha
from creed.science.drug_target_state import target_map


def main():
    p=argparse.ArgumentParser()
    for k in ('visible','data','rna','gdsc','hgnc','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    assert sha(a.visible)=='468767f19ea2c935285c0d0877475ea97cb70140662b307f39e841d91a81e5d1'
    ids=prefix(a.visible,'sample_ids',421)
    np.testing.assert_array_equal(ids,np.load(a.data/'sample_ids.npy',allow_pickle=False))
    assert np.all(prefix(a.visible,'partition_labels',421)=='train')
    protein_ids=prefix(a.visible,'l2_protein_ids');values=prefix(a.visible,'l2_values',421)
    assert values.shape==(421,6692) and len(set(protein_ids))==6692
    assert not np.isinf(values).any()
    old_ids=np.load(a.data/'protein_ids.npy',allow_pickle=False);old=np.load(a.data/'proteins.npy',allow_pickle=False)
    lookup={p:i for i,p in enumerate(protein_ids)};indices=[lookup[p] for p in old_ids]
    np.testing.assert_array_equal(values[:,indices],old)
    assert sha(a.hgnc)=='e73e9259177884b5994fc81ed733c1b3d4df34c84290bc9dddc86e960d5d6419'
    mapping=target_map(a.hgnc,json.loads((a.gdsc/'drug_annotations.json').read_text()),
        np.load(a.rna/'gene_symbols.npy',allow_pickle=False),protein_ids)
    fractions=np.isfinite(values).mean(0)
    coverage={str(t):int((fractions>=t).sum()) for t in (0.,.2,.5,.8,.9,1.)}
    drug_coverage={str(t):sum(any(fractions[i]>=t for i in r['protein_indices']) for r in mapping['records'].values()) for t in (.2,.5,.8,.9,1.)}
    for name,x in dict(sample_ids=ids,protein_ids=protein_ids,proteins=values).items():np.save(a.output/(name+'.npy'),x,allow_pickle=False)
    (a.output/'mapping.json').write_text(json.dumps(mapping,indent=2))
    profile=dict(samples=421,proteins=6692,complete_old_panel_exactly_replayed=584,
        finite_cells=int(np.isfinite(values).sum()),missing_cells=int(np.isnan(values).sum()),
        proteins_at_observed_fraction=coverage,drug_ids_with_a_paired_target_at_fraction=drug_coverage,
        mapping_drug_ids=mapping['supported_drug_ids'],unique_paired_genes=mapping['unique_paired_genes'],
        reason_for_old_panel='421/421 measured and nonconstant in historical panel selection',
        missingness='measured source NaNs retained; no imputed values presented as observations',audit_rows_decoded=0,final_access=False)
    (a.output/'profile.json').write_text(json.dumps(profile,indent=2))
    manifest=dict(scope='exploratory 421 training prefix only, expanded missing-aware protein source',
        source_sha256=sha(a.visible),hgnc_sha256=sha(a.hgnc),scientific_claim=False,
        files={f.name:dict(sha256=sha(f),bytes=f.stat().st_size) for f in a.output.iterdir()})
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(profile),flush=True)


if __name__=='__main__':main()

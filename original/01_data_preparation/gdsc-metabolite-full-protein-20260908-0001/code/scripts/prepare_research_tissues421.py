"""Bind original tissue metadata to the isolated development sample order."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
import numpy as np
from run_research_diagnostic import sha
from creed.science.ccle_visible_lineage_registry import _read_mmc2_identity_records, PRODUCTION_CONTRACT


def main():
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True); p.add_argument('--data',type=Path,required=True); a=p.parse_args()
    assert sha(a.source)=='9327f665d259ef4d148d3a64e7c76cdeb51dce8dee06768efd6b7d044130e203'
    assert sha(a.data/'manifest.json')=='1b44f54110724f851c7240c252a5fec5abe5362a753c656b4b1e341041ba2936'
    manifest=json.loads((a.data/'manifest.json').read_text())
    assert sha(a.data/'sample_ids.npy')==manifest['files']['sample_ids.npy']['sha256']
    raw=a.source.read_bytes(); assert len(raw)==960365
    records,opened=_read_mmc2_identity_records(SimpleNamespace(raw=raw),contract=PRODUCTION_CONTRACT)
    ids=np.load(a.data/'sample_ids.npy',allow_pickle=False)
    assert len(ids)==len(set(ids))==421
    tissues=[records[str(i)].tissue_type for i in ids]
    assert all(isinstance(t,str) and t.strip()==t and t for t in tissues)
    assert len(set(tissues))==22
    print(json.dumps(dict(scientific_claim=False,source_sha256=sha(a.source),script_sha256=sha(Path(__file__)),
        sample_ids_sha256=sha(a.data/'sample_ids.npy'),sample_ids=ids.tolist(),tissues=tissues,
        tissue_counts=dict(sorted(Counter(tissues).items())),opened=opened,expression_read=False),indent=2))


if __name__=='__main__': main()

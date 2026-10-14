"""Materialize only training-prefix rows from already acquired development assets."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path
import numpy as np


def prefix(archive, member, rows=None):
    with zipfile.ZipFile(archive) as z, z.open(member+'.npy') as f:
        version=np.lib.format.read_magic(f)
        if version==(1,0): shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
        elif version==(2,0): shape,fortran,dtype=np.lib.format.read_array_header_2_0(f)
        else: raise ValueError('unsupported NPY version')
        if fortran or dtype.hasobject: raise ValueError('unsafe array format')
        if rows is not None:
            if shape[0]!=485 or rows!=421: raise ValueError('wrong development prefix')
            shape=(rows,*shape[1:])
        length=math.prod(shape)*dtype.itemsize
        raw=f.read(length)
        if len(raw)!=length: raise ValueError('short array prefix')
        return np.frombuffer(raw,dtype=dtype).reshape(shape).copy()


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    for key in ['visible','scgpt','panel','esm','esm-ids','output']:
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    sources={k:sha(getattr(args,k)) for k in ['visible','scgpt','panel','esm','esm_ids']}
    if sources['visible']!='468767f19ea2c935285c0d0877475ea97cb70140662b307f39e841d91a81e5d1':
        raise ValueError('visible source changed')
    if sources['scgpt']!='9624b09167c8a5a8e9ca0682c824e5d37c97cb196275bd5138b9ac464f6712ab':
        raise ValueError('scGPT source changed')
    ids=prefix(args.visible,'sample_ids',421)
    if len(set(ids))!=421 or not np.array_equal(ids,prefix(args.scgpt,'sample_ids',421)):
        raise ValueError('sample identity mismatch')
    for p in [args.visible,args.scgpt]:
        if not np.all(prefix(p,'partition_labels',421)=='train'):
            raise ValueError('non-training row')
    panel=json.loads(args.panel.read_text()); indices=np.asarray(panel['source_indices'])
    if indices.shape!=(584,) or len(set(indices))!=584 or not np.all(np.diff(indices)>0):
        raise ValueError('invalid protein panel')
    proteins=prefix(args.visible,'l2_protein_ids')[indices]
    esm_ids=json.loads(args.esm_ids.read_text())
    if proteins.tolist()!=panel['protein_ids'] or proteins.tolist()!=esm_ids:
        raise ValueError('ESM protein identity mismatch')
    arrays=dict(sample_ids=ids,protein_ids=proteins,
        metabolite_ids=prefix(args.visible,'l0_feature_names'),
        metabolites=prefix(args.visible,'l0_values',421),
        proteins=prefix(args.visible,'l2_values',421)[:,indices],
        scgpt=prefix(args.scgpt,'mean_embedding',421),
        esm=np.load(args.esm,allow_pickle=False))
    for k,shape in dict(metabolites=(421,225),proteins=(421,584),scgpt=(421,512),esm=(584,1280)).items():
        if arrays[k].shape!=shape or not np.isfinite(arrays[k]).all():
            raise ValueError('invalid shape/finite: '+k)
    args.output.mkdir(parents=True,exist_ok=False)
    for k,a in arrays.items(): np.save(args.output/(k+'.npy'),np.ascontiguousarray(a),allow_pickle=False)
    if sources!={k:sha(getattr(args,k)) for k in sources}: raise ValueError('source mutation')
    manifest=dict(scope='exploratory_training_421_only',sample_count=421,source_sha256=sources,
        preparation_sha256=sha(Path(__file__)),audit_values_decoded=False,final_access=False,
        files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in args.output.iterdir()})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest))


if __name__=='__main__': main()

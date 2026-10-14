"""Preflight (no model fitting, no outer scores): identity overlap and bank bookkeeping.

Reads only allowed inputs. Prints counts; writes preflight_identity.json into OUT_DIR.
"""
import collections
import json
import sys

import numpy as np

R = '/public/home/mengxl/creed-research-runs'
SOURCE = R + '/gdsc-metabolite-full-protein-20260908-0001/output/result/source-inputs.npz'
RNA421 = R + '/rna421-20260906-0002'
RES = R + '/extra-supervision-20261008-0001/resources'
EXT = R + '/extra-supervision-20261008-0001/output'
MODEL = R + '/privileged-geometry-20261008-0001/handoff/resources/model_ids_only.json'
OUT_DIR = sys.argv[1]

S = np.load(SOURCE, allow_pickle=False)
sid, stis = S['sample_ids'], S['tissues']
assert (np.load(RNA421 + '/sample_ids.npy') == sid).all()
src421 = np.load(RNA421 + '/rna_source_names.npy')
g421 = np.load(RNA421 + '/gene_ids.npy')
E = np.load(RES + '/rna_gdsc_extra354.npz', allow_pickle=False)
B = np.load(RES + '/m_rna_extra66.npz', allow_pickle=False)
M19 = np.load(RES + '/m_gdsc_extra19.npz', allow_pickle=False)
assert (E['gene_ids'] == g421).all() and (B['gene_ids'] == g421).all()
assert (E['drug_ids'] == S['drug_ids']).all() and (M19['drug_ids'] == S['drug_ids']).all()
assert (B['m_feature_ids'] == S['metabolite_ids']).all()

model = json.load(open(MODEL))['rows']
by_sid = collections.defaultdict(list)
for r in model:
    if r['SangerModelID']:
        by_sid[r['SangerModelID']].append(r)


def ident(ids):
    pat, ach, name, missing = set(), set(), set(), 0
    for s in ids:
        rows = by_sid.get(str(s), [])
        if len(rows) != 1:
            missing += 1
            continue
        r = rows[0]
        if r['PatientID']: pat.add(r['PatientID'])
        if r['ModelID']: ach.add(r['ModelID'])
        if r['StrippedCellLineName']: name.add(r['StrippedCellLineName'])
    return pat, ach, name, missing


p421, a421, n421, miss421 = ident(sid)
out = {'n421': len(sid), 'rna421_sources': dict(collections.Counter(src421.tolist())),
       'model_metadata_unique_for_421': len(sid) - miss421, 'model_metadata_missing_or_ambiguous_421': miss421}
pkgs = {'rna_gdsc_extra354': E, 'm_rna_extra66': B, 'm_gdsc_extra19': M19}
for name, P in pkgs.items():
    ids = P['sample_ids']
    pp, aa, nn, mm = ident(ids)
    rec = dict(n=len(ids), unique_sample_ids=len(set(ids.tolist())),
               sample_overlap_421=len(set(ids.tolist()) & set(sid.tolist())),
               own_patient_ids_unique=len(set(P['patient_ids'].tolist())),
               own_patient_overlap_421_modelcsv=len(set(P['patient_ids'].tolist()) & p421),
               modelcsv_patient_overlap_421=len(pp & p421),
               modelcsv_ach_overlap_421=len(aa & a421),
               own_model_ids_overlap_421=len(set(P['model_ids'].tolist()) & a421),
               stripped_name_overlap_421=len(nn & n421),
               modelcsv_missing_or_ambiguous=mm,
               tissues_known=int(np.sum(P['tissues'] != '')),
               tissues_not_in_421_vocab=sorted(set(P['tissues'].tolist()) - set(stis.tolist()) - {''}))
    if 'rna_source_names' in P.files:
        rec['rna_sources'] = dict(collections.Counter(P['rna_source_names'].tolist()))
    if 'auc' in P.files:
        rec['auc_observed'] = int(np.isfinite(P['auc']).sum())
        rec['rows_with_any_auc'] = int(np.isfinite(P['auc']).any(1).sum())
    out[name] = rec
# cross-package overlaps
s354, s66, s19 = (set(P['sample_ids'].tolist()) for P in (E, B, M19))
pt354, pt66, pt19 = (set(P['patient_ids'].tolist()) for P in (E, B, M19))
out['cross'] = dict(s66_and_s19=len(s66 & s19), s66_and_s354=len(s66 & s354), s19_and_s354=len(s19 & s354),
                    pt66_and_pt19=len(pt66 & pt19), pt66_and_pt354=len(pt66 & pt354), pt19_and_pt354=len(pt19 & pt354))
pseudo = np.array([i for i, s in enumerate(B['sample_ids']) if s not in s19 and s not in s354])
out['pseudo_candidates'] = dict(n=len(pseudo), patient_overlap_with_19=len(set(B['patient_ids'][pseudo].tolist()) & pt19),
                                patient_overlap_with_354=len(set(B['patient_ids'][pseudo].tolist()) & pt354),
                                sources=dict(collections.Counter(B['rna_source_names'][pseudo].tolist())),
                                tissues_known=int(np.sum(B['tissues'][pseudo] != '')))
out['tissue_counts_354'] = dict(collections.Counter(E['tissues'].tolist()).most_common())
out['tissue_counts_421'] = dict(collections.Counter(stis.tolist()).most_common())

# Bank bookkeeping (no outcomes scored): panel consistency and allowed-extra counts per bank.
banks = {}
for sc in ('new_cell', 'tissue'):
    for f in range(5):
        panels = []
        for b in ('inner-0', 'inner-1', 'inner-2', 'outer'):
            G = np.load(f'{EXT}/{sc}/fold-{f}/{b}/geometry.npz', allow_pickle=False)
            fit, query, panel = G['fit'], G['query'], G['panel']
            panels.append(panel)
            banned = set(stis[np.setdiff1d(np.arange(len(sid)), fit)].tolist())
            ok354 = np.isfinite(E['auc']).any(1)
            okp = np.ones(len(pseudo), bool)
            if sc == 'tissue':
                ok354 &= np.array([bool(t) and t not in banned for t in E['tissues']])
                okp &= np.array([bool(t) and t not in banned for t in B['tissues'][pseudo]])
            nsrc = collections.Counter(src421[fit].tolist())
            banks[f'{sc}/fold-{f}/{b}'] = dict(fit=len(fit), query=len(query), panel=len(panel),
                                               extra19=len(G['extra_indices']), extra354=int(ok354.sum()),
                                               pseudo48=int(okp.sum()), fit_sanger=nsrc.get('Sanger RNASeq', 0),
                                               fit_query_overlap=len(set(fit) & set(query)))
        P = np.load(f'{EXT}/{sc}/fold-{f}/predictions.npz', allow_pickle=False)
        assert all((p == P['panel']).all() for p in panels), (sc, f)
        assert (P['fit'] == np.load(f'{EXT}/{sc}/fold-{f}/outer/geometry.npz')['fit']).all()
        assert (P['held'] == np.load(f'{EXT}/{sc}/fold-{f}/outer/geometry.npz')['query']).all()
out['banks'] = banks
json.dump(out, open(f'{OUT_DIR}/preflight_identity.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in out.items() if k not in ('banks',)}, indent=1, ensure_ascii=False))
for k, v in banks.items():
    print(k, v)

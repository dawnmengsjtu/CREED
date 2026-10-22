"""PROSPECTIVE_LOWDIM_V1 step 1 (PLAN.md section 3): GDSC1 AUC matrix for the 421 development lines + data gate D0-D6.

No model is fitted and no prediction is scored here. The extractor is a line-by-line port of the frozen GDSC2 preparation
(gdsc421-20260906-0001/code/scripts/prepare_gdsc421.py, function extract) with the DATASET value as a parameter; gate D0
re-extracts GDSC2_fitted_dose_response_24Jul22.csv with it and requires the frozen source-inputs.npz AUC matrix back.
Exact SANGER_MODEL_ID and DRUG_ID joins only; no name matching anywhere except the descriptive name-overlap report (D6).
Reads only: source-inputs.npz, extra-supervision geometry (fit/query indices), the GDSC2 24Jul22 CSV, the GDSC1 file.
Never reads the 64 validation or the 123 sealed lines (the whitelist is the 421 development sample_ids).

usage: python prepare_gdsc1.py --gdsc1 FILE --gdsc1-sha SHA256 --gdsc2 FILE --out DIR
  FILE may be .csv or .xlsx (openpyxl if importable, otherwise a standard-library reader for the first worksheet).
exit code 0 = gate passed, 3 = gate failed (data_gate.json says which item); DIR must not exist.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import warnings
import zipfile
from collections import Counter, defaultdict
from xml.etree import ElementTree as ET

import numpy as np
from scipy.stats import spearmanr

ROOT = os.environ.get('CREED_ROOT', '/public/home/mengxl/creed-research-runs')
EXT = ROOT + '/extra-supervision-20261008-0001/output'
SOURCE = ROOT + '/gdsc-metabolite-full-protein-20260908-0001/output/result/source-inputs.npz'
SOURCE_SHA = '52d71517a8952e83a1603cd831f09449544102d0ca5ff829ff01be9bf79cd637'
GDSC2_SHA = '9d6dc82b9a072bb00b6139dcd6c4c11de1e047f210382fb726e92ab8dd0eae2a'
REQUIRED = {'SANGER_MODEL_ID', 'DATASET', 'DRUG_ID', 'DRUG_NAME', 'WEBRELEASE', 'AUC', 'LN_IC50', 'MIN_CONC', 'MAX_CONC', 'PUTATIVE_TARGET'}
INNER = ('inner-0', 'inner-1', 'inner-2')
# pre-registered gate thresholds (PLAN.md section 3.3)
MIN_LINES = 300            # D1: development lines with >= 1 finite GDSC1 AUC
MIN_PANEL = 100            # D2: drugs passing the panel rule in every outer fold
MIN_EVAL = 100             # D2: drugs evaluated (>= 5 observed test lines) in every outer fold
MIN_QUERY_LINES = 40       # D3: test lines with >= 1 GDSC1 AUC in every outer fold
MAX_CONFLICT_FRAC = .05    # D4: conflicting duplicate keys / observed pairs
MIN_DISJOINT = 50          # D6 (sensitivity only, not a stop): name-disjoint panel drugs in every outer fold
D0_TOL = 1e-12             # D0: max |re-extracted GDSC2 AUC - source-inputs AUC| (identical parsing gives 0.0)
PANEL_MIN_OUTER, PANEL_MIN_INNER, EVAL_MIN_QUERY = 80, 20, 5


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def cell_str(v):
    """Spreadsheet cell -> the string a CSV export would carry (integers without '.0')."""
    if v is None:
        return ''
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else repr(v)
    return str(v).strip()


def _col_index(ref):
    letters = re.match(r'[A-Z]+', ref).group(0)
    k = 0
    for ch in letters:
        k = k * 26 + ord(ch) - 64
    return k - 1


def _xlsx_rows_stdlib(path):
    """First worksheet of an .xlsx as lists of strings (shared, inline and numeric cells)."""
    ns = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
    with zipfile.ZipFile(path) as z:
        shared = []
        if 'xl/sharedStrings.xml' in z.namelist():
            for si in ET.parse(z.open('xl/sharedStrings.xml')).getroot().iter(ns + 'si'):
                shared.append(''.join(t.text or '' for t in si.iter(ns + 't')))
        sheets = sorted(n for n in z.namelist() if re.fullmatch(r'xl/worksheets/sheet\d+\.xml', n))
        assert sheets, 'no worksheet in xlsx'
        name = 'xl/worksheets/sheet1.xml' if 'xl/worksheets/sheet1.xml' in sheets else sheets[0]
        for ev, el in ET.iterparse(z.open(name), events=('end',)):
            if el.tag != ns + 'row':
                continue
            cells = {}
            for c in el.iter(ns + 'c'):
                t = c.get('t'); v = c.find(ns + 'v')
                if t == 's':
                    val = shared[int(v.text)]
                elif t == 'inlineStr':
                    val = ''.join(x.text or '' for x in c.iter(ns + 't'))
                else:
                    val = v.text if v is not None and v.text is not None else ''
                    if t not in ('str', 'b') and re.fullmatch(r'-?\d+\.0+', val or ''):
                        val = val.split('.')[0]
                cells[_col_index(c.get('r'))] = val.strip()
            el.clear()
            if cells:
                yield [cells.get(i, '') for i in range(max(cells) + 1)]


def iter_records(path):
    """Yields dict rows (string values) from a GDSC fitted dose-response CSV or XLSX."""
    if path.lower().endswith('.csv'):
        with open(path, encoding='utf-8-sig', newline='') as fh:
            reader = csv.DictReader(fh)
            if not REQUIRED.issubset(reader.fieldnames):
                raise ValueError('unexpected source schema')
            for row in reader:
                yield row
        return
    assert path.lower().endswith('.xlsx'), 'GDSC1 file must be .csv or .xlsx'
    try:
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.worksheets[0]
        ws.reset_dimensions()   # the GDSC1 27Oct23 xlsx stores no sheet dimension; without this read-only mode yields only A1
        rows = (list(map(cell_str, r)) for r in ws.iter_rows(values_only=True))
    except ImportError:
        rows = _xlsx_rows_stdlib(path)
    header = None
    for r in rows:
        if header is None:
            header = [h.strip() for h in r]
            if not REQUIRED.issubset(header):
                raise ValueError('unexpected source schema')
            continue
        if not any(x != '' for x in r):
            continue
        r = list(r) + [''] * (len(header) - len(r))
        yield dict(zip(header, r[:len(header)]))


def extract(records, ids, dataset):
    """Port of prepare_gdsc421.extract (frozen GDSC2 preparation) with `dataset` as a parameter; same QC, same outputs."""
    whitelist = set(ids); values = {}; conflicts = set(); names = defaultdict(set); targets = defaultdict(set)
    total = 0; matched = 0; invalid = 0; datasets = Counter(); releases = Counter(); duplicate_same = 0; outside_unit_auc = 0
    for row in records:
        total += 1; datasets[row['DATASET']] += 1; releases[row['WEBRELEASE']] += 1
        sid = row['SANGER_MODEL_ID']
        if sid not in whitelist:
            continue
        if row['DATASET'] != dataset:
            raise ValueError('mixed dataset')
        matched += 1; drug = row['DRUG_ID']; key = (sid, drug)
        names[drug].add(row['DRUG_NAME']); targets[drug].add(row['PUTATIVE_TARGET'])
        try:
            record = tuple(float(row[k]) for k in ('AUC', 'LN_IC50', 'MIN_CONC', 'MAX_CONC'))
        except ValueError:
            invalid += 1; continue
        if not np.isfinite(record).all() or not (0 < record[2] <= record[3]):
            invalid += 1; continue
        if not 0 <= record[0] <= 1:
            outside_unit_auc += 1
        if key in values:
            if values[key] != record:
                conflicts.add(key)
            else:
                duplicate_same += 1
        else:
            values[key] = record
    drugs = sorted(names, key=lambda x: int(x)); lookup = {v: i for i, v in enumerate(drugs)}; rows = {v: i for i, v in enumerate(ids)}
    auc = np.full((len(ids), len(drugs)), np.nan); ic50 = auc.copy()
    for (sid, drug), record in values.items():
        if (sid, drug) in conflicts:
            continue
        auc[rows[sid], lookup[drug]] = record[0]; ic50[rows[sid], lookup[drug]] = record[1]
    quality = dict(source_rows=total, whitelisted_source_rows=matched, datasets=dict(datasets), webreleases=dict(releases),
                   samples_requested=len(ids), samples_with_response=int(np.isfinite(auc).any(1).sum()), drug_ids=len(drugs),
                   observed_pairs=int(np.isfinite(auc).sum()), missing_pairs=int(np.isnan(auc).sum()), invalid_rows=invalid,
                   exact_duplicate_rows=duplicate_same, conflicting_keys=len(conflicts), outside_unit_auc_rows_retained=outside_unit_auc,
                   drug_names_ambiguous={k: sorted(v) for k, v in names.items() if len(v) != 1})
    annotations = {d: dict(names=sorted(names[d]), putative_targets=sorted(targets[d])) for d in drugs}
    return auc, ic50, np.array(drugs), annotations, quality


def norm_name(s):
    return re.sub(r'[^0-9a-z]', '', s.casefold())


def panel_rule(A, fit, inner_fits):
    """Drug-endpoint panel rule of the main analysis: >= 80 observed outer-fit labels, outer-fit SD > 1e-8, >= 20 observed
    labels in the fit set of each of the three inner banks (so every bank can fit every panel drug)."""
    yf = A[fit]
    cnt = np.isfinite(yf).sum(0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)          # all-NaN columns -> NaN SD -> excluded below
        sd = np.nanstd(yf, axis=0)
    keep = (cnt >= PANEL_MIN_OUTER) & (sd > 1e-8)
    for fi in inner_fits:
        keep &= np.isfinite(A[fi]).sum(0) >= PANEL_MIN_INNER
    return np.flatnonzero(keep)


def main():
    ap = argparse.ArgumentParser()
    for k in ('gdsc1', 'gdsc1-sha', 'gdsc2', 'out'):
        ap.add_argument('--' + k, required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=False)
    gate, items = {}, {}
    # inputs
    assert sha(SOURCE) == SOURCE_SHA, 'source-inputs sha mismatch'
    h1 = sha(a.gdsc1)
    items['D_file_sha'] = dict(expected=a.gdsc1_sha, observed=h1, passed=h1 == a.gdsc1_sha)
    S = np.load(SOURCE, allow_pickle=False)
    ids = [str(x) for x in S['sample_ids']]
    assert len(ids) == len(set(ids)) == 421
    # D0: the ported extractor reproduces the frozen GDSC2 matrix
    assert sha(a.gdsc2) == GDSC2_SHA, 'GDSC2 24Jul22 CSV sha mismatch'
    auc2, _, drugs2, ann2, q2 = extract(iter_records(a.gdsc2), ids, 'GDSC2')
    same_drugs = len(drugs2) == len(S['drug_ids']) and all(str(x) == str(y) for x, y in zip(drugs2, S['drug_ids']))
    same_nan = auc2.shape == S['auc'].shape and np.array_equal(np.isnan(auc2), np.isnan(S['auc']))
    d0 = float(np.nanmax(np.abs(auc2 - S['auc']))) if same_nan else float('inf')
    items['D0_extractor_reproduces_gdsc2'] = dict(drug_ids_equal=bool(same_drugs), nan_pattern_equal=bool(same_nan), max_abs_diff=d0,
                                                  passed=bool(same_drugs and same_nan and d0 <= D0_TOL))
    # GDSC1 extraction
    auc1, ic1, drugs1, ann1, q1 = extract(iter_records(a.gdsc1), ids, 'GDSC1')
    n_lines = int(np.isfinite(auc1).any(1).sum())
    items['D1_lines_with_gdsc1'] = dict(value=n_lines, threshold=MIN_LINES, passed=n_lines >= MIN_LINES)
    frac = q1['conflicting_keys'] / max(q1['observed_pairs'], 1)
    items['D4_file_integrity'] = dict(conflicting_keys=q1['conflicting_keys'], observed_pairs=q1['observed_pairs'], conflict_fraction=frac,
                                      ambiguous_drug_names=len(q1['drug_names_ambiguous']), invalid_rows=q1['invalid_rows'],
                                      passed=frac <= MAX_CONFLICT_FRAC)
    # D2/D3/D6 per scenario x outer fold (panel rule uses fit rows only; evaluated = >= 5 observed test labels)
    n2 = {norm_name(n) for d in drugs2 for n in ann2[d]['names']}
    in_gdsc2_name = np.array([any(norm_name(n) in n2 for n in ann1[d]['names']) for d in drugs1])
    in_gdsc2_id = np.isin(drugs1.astype(str), np.asarray(S['drug_ids']).astype(str))
    folds, panels = {}, {}
    for scen in ('new_cell', 'tissue'):
        for f in range(5):
            Go = np.load(f'{EXT}/{scen}/fold-{f}/outer/geometry.npz', allow_pickle=False)
            inner = [np.load(f'{EXT}/{scen}/fold-{f}/{b}/geometry.npz', allow_pickle=False)['fit'] for b in INNER]
            fit, query = Go['fit'], Go['query']
            assert max(fit.max(), query.max()) < 421
            pan = panel_rule(auc1, fit, inner)
            qobs = np.isfinite(auc1[np.ix_(query, pan)]).sum(0)
            ev = qobs >= EVAL_MIN_QUERY
            panels[f'{scen}__fold-{f}__panel'] = pan
            panels[f'{scen}__fold-{f}__evaluated'] = ev
            folds[f'{scen}/fold-{f}'] = dict(panel=int(len(pan)), evaluated=int(ev.sum()),
                                            query_lines_with_label=int(np.isfinite(auc1[query]).any(1).sum()),
                                            fit_lines_with_label=int(np.isfinite(auc1[fit]).any(1).sum()),
                                            panel_name_disjoint=int((~in_gdsc2_name[pan]).sum()),
                                            evaluated_name_disjoint=int((~in_gdsc2_name[pan] & ev).sum()))
    items['D2_drugs_per_fold'] = dict(min_panel=min(v['panel'] for v in folds.values()), min_evaluated=min(v['evaluated'] for v in folds.values()),
                                      thresholds=[MIN_PANEL, MIN_EVAL],
                                      passed=all(v['panel'] >= MIN_PANEL and v['evaluated'] >= MIN_EVAL for v in folds.values()))
    items['D3_test_lines_per_fold'] = dict(min=min(v['query_lines_with_label'] for v in folds.values()), threshold=MIN_QUERY_LINES,
                                           passed=all(v['query_lines_with_label'] >= MIN_QUERY_LINES for v in folds.values()))
    # D5 overlap with GDSC2 (descriptive; exact DRUG_ID and normalized-name overlap) + cross-screen concordance of matched names
    pairs = []
    by_name2 = defaultdict(list)
    for j, d in enumerate(drugs2):
        for n in ann2[d]['names']:
            by_name2[norm_name(n)].append(j)
    for i, d in enumerate(drugs1):
        for n in ann1[d]['names']:
            for j in by_name2.get(norm_name(n), []):
                o = np.isfinite(auc1[:, i]) & np.isfinite(auc2[:, j])
                rho = float(spearmanr(auc1[o, i], auc2[o, j]).statistic) if o.sum() >= 20 else float('nan')
                pairs.append(dict(gdsc1_id=str(d), gdsc2_id=str(drugs2[j]), name=n, shared_lines=int(o.sum()), spearman=rho))
    rhos = [p['spearman'] for p in pairs if np.isfinite(p['spearman'])]
    items['D5_overlap_with_gdsc2'] = dict(gdsc1_drug_ids=int(len(drugs1)), same_drug_id_as_gdsc2=int(in_gdsc2_id.sum()),
                                          name_matched_to_gdsc2=int(in_gdsc2_name.sum()), name_disjoint=int((~in_gdsc2_name).sum()),
                                          matched_pairs=len(pairs), median_cross_screen_spearman=float(np.median(rhos)) if rhos else float('nan'),
                                          passed=True)
    items['D6_name_disjoint_sensitivity'] = dict(min_panel_name_disjoint=min(v['panel_name_disjoint'] for v in folds.values()),
                                                 threshold=MIN_DISJOINT, evaluable=all(v['panel_name_disjoint'] >= MIN_DISJOINT for v in folds.values()),
                                                 passed=True)
    stop_items = ('D_file_sha', 'D0_extractor_reproduces_gdsc2', 'D1_lines_with_gdsc1', 'D2_drugs_per_fold', 'D3_test_lines_per_fold', 'D4_file_integrity')
    passed = all(items[k]['passed'] for k in stop_items)
    np.save(f'{a.out}/auc.npy', auc1, allow_pickle=False)
    np.save(f'{a.out}/ln_ic50.npy', ic1, allow_pickle=False)
    np.save(f'{a.out}/drug_ids.npy', drugs1.astype(str), allow_pickle=False)
    np.save(f'{a.out}/sample_ids.npy', np.array(ids), allow_pickle=False)
    np.save(f'{a.out}/name_in_gdsc2.npy', in_gdsc2_name, allow_pickle=False)
    np.savez_compressed(f'{a.out}/panels.npz', **panels)
    json.dump(ann1, open(f'{a.out}/drug_annotations.json', 'w'), indent=1)
    json.dump(dict(gdsc1=q1, gdsc2_reextracted=q2), open(f'{a.out}/quality.json', 'w'), indent=1)
    with open(f'{a.out}/name_matched_pairs.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['gdsc1_id', 'gdsc2_id', 'name', 'shared_lines', 'spearman']); w.writeheader(); w.writerows(pairs)
    files = {n: sha(f'{a.out}/{n}') for n in sorted(os.listdir(a.out))}
    gate = dict(protocol='PROSPECTIVE_LOWDIM_V1', gdsc1_file=os.path.abspath(a.gdsc1), gdsc1_sha256=h1, gdsc2_file=os.path.abspath(a.gdsc2),
                source_sha256=SOURCE_SHA, items=items, per_fold=folds, stop_items=list(stop_items), passed=bool(passed), files=files,
                endpoint='GDSC1 AUC, native fitted dose-response area; exact SANGER_MODEL_ID / DRUG_ID joins; missing kept missing')
    json.dump(gate, open(f'{a.out}/data_gate.json', 'w'), indent=1, default=float)
    print(json.dumps(dict(passed=passed, **{k: v.get('passed') for k, v in items.items()}), default=float), flush=True)
    sys.exit(0 if passed else 3)


if __name__ == '__main__':
    main()

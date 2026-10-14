"""Official-byte-checked extraction of CRISPR effects for safe development IDs."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from creed.science.functional_source_catalog import index_models


def checksum(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def extract(stream, mapping):
    header = next(csv.reader([stream.readline().decode('utf-8-sig')]))
    if header[0] not in ('', 'ModelID') or len(set(header[1:])) != len(header) - 1:
        raise ValueError('invalid CRISPR header')
    records = {}
    rows = 0
    for line in stream:
        prefix, delimiter, _ = line.partition(b',')
        if not delimiter:
            raise ValueError('invalid source identity delimiter')
        mid = prefix.decode('ascii').strip('"')
        rows += 1
        if mid not in mapping:
            continue
        if mid in records:
            raise ValueError('duplicate selected outcome row')
        fields = next(csv.reader([line.decode('utf-8')]))
        if len(fields) != len(header):
            raise ValueError('invalid selected outcome width')
        values = np.array([np.nan if v in ('', 'NA', 'NaN', 'nan') else float(v) for v in fields[1:]])
        if np.isinf(values).any():
            raise ValueError('infinite selected outcome')
        records[mid] = values
    return header[1:], records, rows


def main():
    p = argparse.ArgumentParser()
    for k in ('model', 'effect', 'safe_ids', 'bindings', 'output'):
        p.add_argument('--' + k.replace('_', '-'), type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    bindings = json.loads(a.bindings.read_text())
    for name, path in [('Model.csv', a.model), ('CRISPRGeneEffect.csv', a.effect)]:
        b = bindings['files'][name]
        if path.stat().st_size != b['bytes'] or checksum(path, 'md5') != b['md5']:
            raise ValueError('source does not match official24Q2 object: ' + name)
    if checksum(a.safe_ids) != '91624523bae5840eec817d3663f36d6dd77605a327d9a6351fa7ebea61ee8f28':
        raise ValueError('unexpected safe identity source')
    safe = np.load(a.safe_ids, allow_pickle=False).tolist()
    with a.model.open(encoding='utf-8-sig', newline='') as f:
        mapping, _, _ = index_models(list(csv.DictReader(f)), safe)
    with a.effect.open('rb') as f:
        genes, records, count = extract(f, mapping)
    ordered = sorted(records, key=lambda mid: safe.index(mapping[mid]['sanger_id']))
    if len(ordered) != 310:
        raise ValueError('catalog and extraction disagree on selected310')
    values = np.stack([records[mid] for mid in ordered])
    arrays = dict(effects=values, gene_ids=np.array(genes), model_ids=np.array(ordered),
                  sample_ids=np.array([mapping[m]['sanger_id'] for m in ordered]),
                  lineage=np.array([mapping[m]['lineage'] for m in ordered]),
                  patient_ids=np.array([mapping[m]['patient_id'] or m for m in ordered]))
    for name, x in arrays.items():
        np.save(a.output / (name + '.npy'), x, allow_pickle=False)
    summary = dict(samples=len(ordered), genes=len(genes), source_rows=count,
                   finite_values=int(np.isfinite(values).sum()), missing_values=int(np.isnan(values).sum()),
                   complete_genes=int(np.isfinite(values).all(0).sum()),
                   lineages=dict(Counter(arrays['lineage'])),
                   independent_patients=len(set(arrays['patient_ids'].tolist())),
                   official_source_bytes_verified=True, scientific_claim=False, external_confirmation=False,
                   source_sha256=checksum(a.effect), model_sha256=checksum(a.model),
                   bindings_sha256=checksum(a.bindings),
                   scope='Only exact selected310 of safe421 development identities; no expression source read.',
                   source_processing='DepMap24Q2 integrated copy-number/screen-quality corrected Chronos; not refitted upstream',
                   files={p.name: dict(sha256=checksum(p), bytes=p.stat().st_size) for p in a.output.glob('*.npy')})
    (a.output / 'manifest.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()

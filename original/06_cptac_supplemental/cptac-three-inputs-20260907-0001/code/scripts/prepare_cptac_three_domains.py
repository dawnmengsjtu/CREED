"""Read only three explicitly allowed development bundles; no GBM mount/read."""
import argparse
import json
from pathlib import Path

import numpy as np

from creed.science.cptac_transfer_inputs import sha, SCALES, rna_log_scale
from creed.science.masked_baseline import source_gene_labels, unique_feature_map

COHORTS = ('brca_cptac_2020', 'luad_cptac_2020', 'ucec_cptac_2020')
HASHES = ('8e65f0573edfba01f83b094b630f17d074d80feca623ae70e5e273979ea0bfd6',
          '4d5879681c305a1563260e03b20d42e974ffe634f4b5c6e8b3e1fdc62945b525',
          'e54338d68e401a90d98ae63d7fd6b5e88cb4daf8ccb578f1b2982323d8916083')
COUNTS = (122, 110, 95)


def main():
    parser = argparse.ArgumentParser()
    for key in ('blocks', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    bundles, records = [], []
    for cohort, expected, count in zip(COHORTS, HASHES, COUNTS):
        path = args.blocks / (cohort + '.npz')
        assert sha(path) == expected
        with np.load(path, allow_pickle=False) as source:
            bundle = {k: source[k] for k in source.files if k.startswith(('rna__', 'protein__'))
                      or k in ('sample_ids', 'patient_ids')}
        assert len(bundle['patient_ids']) == len(set(bundle['patient_ids'])) == count
        bundles.append(bundle)
        records.append(dict(cohort=cohort, patients=count, source_sha256=expected,
            rna_handling=SCALES[cohort], source_path=str(path)))
    output, feature_records = {}, {}
    for modality in ('rna', 'protein'):
        maps = [unique_feature_map(source_gene_labels(b[modality + '__feature_ids'], c, modality))
                for b, c in zip(bundles, COHORTS)]
        ids = np.array(sorted(set().union(*(set(m) for m in maps))))
        lookup = {gene: i for i, gene in enumerate(ids)}
        matrices = []
        for cohort, bundle, mapping in zip(COHORTS, bundles, maps):
            values = np.where(bundle[modality + '__observed'], bundle[modality + '__values'], np.nan)
            assert not np.isinf(values).any()
            if modality == 'rna':
                values = rna_log_scale(values, cohort)
            matrix = np.full((len(bundle['patient_ids']), len(ids)), np.nan)
            genes = sorted(mapping)
            matrix[:, [lookup[g] for g in genes]] = values[:, [mapping[g] for g in genes]]
            matrices.append(matrix)
        output[modality] = np.concatenate(matrices)
        output[modality + '_ids'] = ids
        feature_records[modality] = dict(union_genes=len(ids), cohort_unique_genes=[len(m) for m in maps],
            cohort_observed_values=[int(np.isfinite(m).sum()) for m in matrices])
    output['patient_ids'] = np.concatenate([b['patient_ids'] for b in bundles])
    output['sample_ids'] = np.concatenate([b['sample_ids'] for b in bundles])
    output['cohort'] = np.concatenate([np.full(len(b['patient_ids']), c) for b, c in zip(bundles, COHORTS)])
    assert len(output['patient_ids']) == len(set(output['patient_ids'])) == 327
    assert len(set(output['sample_ids'])) == 327
    np.savez_compressed(args.output / 'inputs.npz', **output)
    result = dict(status='prepared_development_only', patients=327, cohorts=records,
        feature_identity=feature_records, input_bundle_sha256=sha(args.output / 'inputs.npz'),
        producer_sha256=sha(Path(__file__)),
        boundaries=['only three allowlisted source bundles decoded', 'no GBM/test arrays accessed',
                    'gene union is schema only; model eligibility/scaling must use training values',
                    'source log scales retained; no cross-study physical-reference equivalence',
                    'duplicate/ambiguous gene identities excluded without alias guesses'])
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

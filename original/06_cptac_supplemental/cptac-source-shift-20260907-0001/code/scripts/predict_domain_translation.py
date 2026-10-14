"""Run a saved domain translator using only a measured source-modality file."""
import argparse
import json
from pathlib import Path

import numpy as np

from creed.science.domain_experts import load_domain_experts, predict_domain_experts


def main():
    parser = argparse.ArgumentParser()
    for key in ('model', 'query', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.query, allow_pickle=False) as query:
        assert set(query.files) == {'values', 'gene_ids', 'patient_ids', 'assay_schema'}
        schema = str(query['assay_schema'].item())
        ids = query['patient_ids']
        prediction = predict_domain_experts(load_domain_experts(args.model), query['values'],
            query['gene_ids'], ids, assay_schema=schema)
    np.savez_compressed(args.output / 'predictions.npz', patient_ids=ids,
        target_gene_ids=prediction['output_reference']['target_gene_ids'],
        weights=prediction['weights'], source_coverage=prediction['source_coverage'],
        disagreement=prediction['disagreement'], **prediction['native'],
        **{'standardized_' + k: v for k, v in prediction['standardized'].items()})
    summary = dict(status='source_only_prediction_complete', patients=len(ids),
        targets=len(prediction['output_reference']['target_gene_ids']),
        source_schema=schema, arms=list(prediction['native']),
        scientific_claim=False, caveat='Development research model; inferred values are not measured data; disagreement is uncalibrated.')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()

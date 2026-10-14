"""Compare separately executed source-only inference with the recorded models."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    for k in ('reference', 'inference', 'output'):
        p.add_argument('--' + k, type=Path, required=True)
    a = p.parse_args()
    reference = json.loads((a.reference / 'summary.json').read_text())
    records, total = [], 0
    for item in reference['records']:
        path = a.inference / item['key'] / 'predictions.npz'
        with np.load(path, allow_pickle=False) as new, np.load(a.reference / (item['key'] + '.npz'), allow_pickle=False) as old:
            np.testing.assert_array_equal(new['patient_ids'], old['patient_ids'])
            np.testing.assert_array_equal(new['target_gene_ids'], old['target_ids'])
            for arm in reference['arms']:
                np.testing.assert_allclose(new['standardized_' + arm], old[arm], atol=1e-9, rtol=1e-9)
                np.testing.assert_allclose(new[arm], old[arm] * old['target_sd'] + old['target_mean'], atol=1e-9, rtol=1e-9)
            np.testing.assert_allclose(new['weights'], old['weights'], atol=1e-10)
            np.testing.assert_allclose(new['disagreement'], old['disagreement'], atol=1e-10)
            count = int(old['gated_experts'].size)
            total += count
            records.append(dict(task=item['key'], patients=len(old['patient_ids']), targets=len(old['target_ids']),
                primary_predictions_replayed=count, arms=len(reference['arms']),
                inference_predictions_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    report = dict(status='source_only_replay_passed', scientific_claim=False, tasks=len(records),
        primary_prediction_values_replayed=total, native_and_standardized_arm_arrays=2 * len(records) * len(reference['arms']),
        records=records, caveat='Execution/replay verification only; predictive accuracy is reported separately.')
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

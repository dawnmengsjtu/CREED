"""Independently recompute domain-experiment evaluation; no model imports."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ARMS = ('train_mean', 'pooled_anchor', 'pooled_context', 'uniform_experts', 'gated_experts', 'reversed_gate')
COHORTS = ('brca_cptac_2020', 'luad_cptac_2020', 'ucec_cptac_2020')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for key in ('run', 'prepared', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    result = json.loads((args.run / 'summary.json').read_text())
    assert result['primary_arm'] == 'gated_experts' and len(result['records']) == 6
    assert sha(args.prepared / 'inputs.npz') == result['input_bundle_sha256']
    if result.get('source_shift'):
        assert args.reference is not None
        assert sha(args.reference / 'summary.json') == result['uncorrected_reference_sha256']
    with np.load(args.prepared / 'inputs.npz', allow_pickle=False) as z:
        data = dict(z)
    assert len(data['patient_ids']) == 327 and set(data['cohort']) == set(COHORTS)
    checked, hashes = [], {}
    for record in result['records']:
        source, target = record['direction'].split('_to_')
        path = args.run / (record['key'] + '.npz')
        hashes[path.name] = sha(path)
        with np.load(path, allow_pickle=False) as z:
            d = dict(z)
        fit = np.flatnonzero(data['cohort'] != record['held_out_cohort'])
        held = np.flatnonzero(data['cohort'] == record['held_out_cohort'])
        np.testing.assert_array_equal(d['fit'], fit)
        np.testing.assert_array_equal(d['held'], held)
        np.testing.assert_array_equal(d['patient_ids'], data['patient_ids'][held])
        assert not set(data['patient_ids'][fit]) & set(data['patient_ids'][held])
        train = data[target][fit]
        n = np.isfinite(train).sum(0)
        mean = np.nansum(train, axis=0) / np.maximum(n, 1)
        sd = np.sqrt(np.nansum((train - mean) ** 2, axis=0) / np.maximum(n, 1))
        panel = np.flatnonzero((n >= max(5, .8 * len(fit))) & (sd > 1e-8))
        np.testing.assert_array_equal(d['target_ids'], data[target + '_ids'][panel])
        np.testing.assert_allclose(d['target_mean'], mean[panel], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(d['target_sd'], sd[panel], rtol=1e-10, atol=1e-10)
        truth = (data[target][held][:, panel] - mean[panel]) / sd[panel]
        np.testing.assert_allclose(d['truth'], truth, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(d['weights'].sum(1), 1.)
        assert np.isfinite(d['weights']).all() and (d['weights'] >= 0).all()
        assert d['weights'].shape == (len(held), 2)
        experts = d['expert_predictions_standardized']
        np.testing.assert_allclose(d['uniform_experts'], experts.mean(1))
        np.testing.assert_allclose(d['gated_experts'], np.sum(experts * d['weights'][:, :, None], axis=1))
        np.testing.assert_allclose(d['reversed_gate'], np.sum(experts * d['weights'][:, ::-1, None], axis=1))
        np.testing.assert_allclose(d['train_mean'], 0., atol=1e-10)
        with np.load(args.run / 'query_inputs' / (record['key'] + '.npz'), allow_pickle=False) as q:
            assert set(q.files) == {'values', 'gene_ids', 'patient_ids', 'assay_schema'}
            np.testing.assert_array_equal(q['values'], data[source][held])
            np.testing.assert_array_equal(q['gene_ids'], data[source + '_ids'])
        if result.get('source_shift'):
            with np.load(args.reference / (record['key'] + '.npz'), allow_pickle=False) as previous:
                np.testing.assert_allclose(previous['truth'], truth)
                np.testing.assert_array_equal(previous['target_ids'], d['target_ids'])
                np.testing.assert_allclose(previous['weights'], d['weights'], rtol=1e-10, atol=1e-10)
                for arm in ARMS:
                    mask = np.isfinite(truth)
                    old_rmse = float(np.sqrt(np.mean((previous[arm][mask] - truth[mask]) ** 2)))
                    np.testing.assert_allclose(old_rmse, record['uncorrected_arms'][arm]['standardized_rmse'])
        for arm in ARMS:
            prediction = d[arm]
            assert prediction.shape == truth.shape and np.isfinite(prediction).all()
            mask = np.isfinite(truth)
            error = prediction[mask] - truth[mask]
            rmse = float(np.sqrt(np.mean(error ** 2)))
            bias = float(error.mean())
            old = record['arms'][arm]
            np.testing.assert_allclose(rmse, old['standardized_rmse'], rtol=1e-10)
            np.testing.assert_allclose(bias, old['standardized_bias'], rtol=1e-10, atol=1e-10)
            assert int(mask.sum()) == old['observed_scored_values']
            correlations = []
            for j in range(truth.shape[1]):
                use = mask[:, j]
                a, b = truth[use, j], prediction[use, j]
                if len(a) >= 5 and np.ptp(a) > 1e-12 and np.ptp(b) > 1e-12:
                    correlations.append(float(np.corrcoef(a, b)[0, 1]))
            assert len(correlations) == old['valid_r_targets']
            if correlations:
                np.testing.assert_allclose(np.mean(correlations), old['mean_target_r'], atol=1e-10)
                np.testing.assert_allclose(np.median(correlations), old['median_target_r'], atol=1e-10)
            checked.append(dict(direction=record['direction'], held=record['held_out_cohort'], arm=arm,
                standardized_rmse=rmse, standardized_bias=bias, observed_values=int(mask.sum())))
    summary = {direction: {arm: float(np.mean([r['standardized_rmse'] for r in checked
        if r['direction'] == direction and r['arm'] == arm])) for arm in ARMS}
        for direction in ('rna_to_protein', 'protein_to_rna')}
    for direction in summary:
        for arm in ARMS:
            np.testing.assert_allclose(summary[direction][arm], result['summary'][direction][arm], rtol=1e-10)
    report = dict(status='evaluation_checked', independent_model_refit=False, scientific_claim=False,
        input_sha256=result['input_bundle_sha256'], run_summary_sha256=sha(args.run / 'summary.json'),
        prediction_file_hashes=hashes, arm_task_metrics_recomputed=len(checked),
        source_shift=result.get('source_shift', False), frozen_reference_comparison_checked=bool(result.get('source_shift')),
        summary=summary, metrics=checked,
        boundaries=['evaluation replay, not independent model refit', 'all three cohorts reused development',
                    'source-only query files verified; inference-container replay recorded separately'])
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(dict(status=report['status'], summary=summary)), flush=True)


if __name__ == '__main__':
    main()

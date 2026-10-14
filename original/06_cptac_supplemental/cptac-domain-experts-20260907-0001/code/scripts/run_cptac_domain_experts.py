"""Six fixed development tasks with saved, source-only bidirectional models."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from creed.science.cptac_transfer_inputs import sha
from creed.science.masked_baseline import target_panel, metrics
from creed.science.domain_experts import (fit_domain_experts, predict_domain_experts,
    save_domain_experts, load_domain_experts)
from prepare_cptac_three_domains import COHORTS

ARMS = ('train_mean', 'pooled_anchor', 'pooled_context', 'uniform_experts', 'gated_experts', 'reversed_gate')
INPUT_SHA = '1b8b3eeb7e20c55c33ca788ca8a025213de7dbb003d21de5bbe36efc94b25bfd'
SUMMARY_SHA = 'b55c48d3b233b4d071d0820b0d4ec615c6acc5b9da0dc830a8aa15e2a19dfcac'


def main():
    parser = argparse.ArgumentParser()
    for key in ('prepared', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    assert sha(args.prepared / 'summary.json') == SUMMARY_SHA
    assert sha(args.prepared / 'inputs.npz') == INPUT_SHA
    with np.load(args.prepared / 'inputs.npz', allow_pickle=False) as source:
        data = dict(source)
    assert set(data['cohort']) == set(COHORTS)
    assert len(data['patient_ids']) == len(set(data['patient_ids'])) == 327
    intent = dict(scientific_claim=False, external_confirmation=False, primary_arm='gated_experts',
        cohorts=COHORTS, directions=['rna_to_protein', 'protein_to_rna'], arms=ARMS,
        alpha=.1, rank=16, max_predictors=1000, seed=20260907, patient_count=327,
        input_bundle_sha256=INPUT_SHA, prepared_summary_sha256=SUMMARY_SHA,
        code_sha256=sha(Path(__file__)),
        model_sha256=sha(Path(__file__).resolve().parents[1] / 'src/creed/science/domain_experts.py'),
        selection='fixed first domain-conditioned candidate; no tuning after held-cohort outcomes',
        metric='training-target-SD standardized pooled-observed-entry RMSE, equal three-cohort mean per direction',
        primary_comparisons=['pooled_context', 'uniform_experts', 'reversed_gate'],
        deployment='query source modality only; no target modality or target-cohort reference calibration; independent rows',
        boundary='three reused development cohorts only; no GBM; source log scales not common physical reference; fixed source gate is not a trained agent',
        uncertainty='expert disagreement is a diagnostic, not a calibrated confidence interval')
    (args.output / 'intent.json').write_text(json.dumps(intent, indent=2, allow_nan=False))
    (args.output / 'models').mkdir()
    (args.output / 'query_inputs').mkdir()
    records = []
    for source, target in (('rna', 'protein'), ('protein', 'rna')):
        direction = source + '_to_' + target
        schema = 'cptac_three_development_' + source + '_source_log_reference_v1'
        for held in COHORTS:
            key = direction + '__' + held
            fit = np.flatnonzero(data['cohort'] != held)
            query = np.flatnonzero(data['cohort'] == held)
            assert not set(data['patient_ids'][fit]) & set(data['patient_ids'][query])
            panel, mean, sd = target_panel(data[target][fit])
            model = fit_domain_experts(data[source][fit], data[target][fit][:, panel],
                data['cohort'][fit], data[source + '_ids'], data[target + '_ids'][panel],
                data['patient_ids'][fit], assay_schema=schema, alpha=.1, rank=16,
                max_predictors=1000, seed=20260907)
            predicted = predict_domain_experts(model, data[source][query], data[source + '_ids'],
                data['patient_ids'][query], assay_schema=schema)
            np.testing.assert_allclose(predicted['output_reference']['mean'], mean)
            np.testing.assert_allclose(predicted['output_reference']['sd'], sd)
            assert set(predicted['standardized']) == set(ARMS)
            save_domain_experts(model, args.output / 'models' / key)
            loaded = load_domain_experts(args.output / 'models' / key)
            order = np.arange(len(query))[::-1]
            genes = np.arange(len(data[source + '_ids']))[::-1]
            replay = predict_domain_experts(loaded, data[source][query[order]][:, genes],
                data[source + '_ids'][genes], data['patient_ids'][query[order]], assay_schema=schema)
            for arm in ARMS:
                np.testing.assert_allclose(predicted['standardized'][arm][order], replay['standardized'][arm], rtol=1e-9, atol=1e-9)
                assert np.isfinite(predicted['standardized'][arm]).all()
            single = predict_domain_experts(loaded, data[source][query[:1]], data[source + '_ids'],
                data['patient_ids'][query[:1]], assay_schema=schema)
            np.testing.assert_allclose(single['native']['gated_experts'], predicted['native']['gated_experts'][:1], rtol=1e-9, atol=1e-9)
            # Deliberately separate source-only runnable query artifact from scoring truth.
            np.savez_compressed(args.output / 'query_inputs' / (key + '.npz'), values=data[source][query],
                gene_ids=data[source + '_ids'], patient_ids=data['patient_ids'][query], assay_schema=np.array(schema))
            truth = (data[target][query][:, panel] - mean) / sd
            scores = {}
            for arm in ARMS:
                p = predicted['standardized'][arm]
                scores[arm] = metrics(truth, p)
                error = p - truth
                scores[arm]['standardized_bias'] = float(np.nanmean(error))
            out = dict(truth=truth, fit=fit, held=query, patient_ids=data['patient_ids'][query],
                target_ids=data[target + '_ids'][panel], target_mean=mean, target_sd=sd,
                weights=predicted['weights'], distance=predicted['distance'],
                source_coverage=predicted['source_coverage'], disagreement=predicted['disagreement'],
                expert_predictions_standardized=predicted['expert_predictions_standardized'],
                domain_ids=np.asarray(predicted['domain_ids']), **predicted['standardized'])
            np.savez_compressed(args.output / (key + '.npz'), **out)
            record = dict(key=key, direction=direction, held_out_cohort=held, train_patients=len(fit),
                query_patients=len(query), targets=len(panel), arms=scores,
                mean_source_coverage=float(np.mean(predicted['source_coverage'])),
                expert_domains=list(predicted['domain_ids']),
                mean_expert_weights=np.mean(predicted['weights'], axis=0).tolist(),
                save_reload_all_arm_replay=True, single_query_replay=True)
            records.append(record)
            print(json.dumps(dict(**record, seconds=time.perf_counter()-start)), flush=True)
    summary = {direction: {arm: float(np.mean([r['arms'][arm]['standardized_rmse'] for r in records
        if r['direction'] == direction])) for arm in ARMS} for direction in intent['directions']}
    gates = {direction: all(summary[direction]['gated_experts'] < summary[direction][arm]
        for arm in intent['primary_comparisons']) for direction in intent['directions']}
    result = dict(**intent, status='completed', records=records, summary=summary,
        descriptive_primary_gates=gates, seconds=time.perf_counter()-start)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(dict(summary=summary, descriptive_primary_gates=gates)), flush=True)


if __name__ == '__main__':
    main()

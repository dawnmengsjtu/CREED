"""Saved, source-only domain experts for paired development assays.

This is a regression component, not a full-layer/agent or causal claim. Domain
experts share a training-selected source panel, never query-domain labels.
"""
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np

from .anchored_context import fit_anchor, predict_anchor


ARMS = ('train_mean', 'pooled_anchor', 'pooled_context', 'uniform_experts',
        'gated_experts', 'reversed_gate')
FORMAT = 'creed.domain-experts.v1'


def _ids(values, name, *, unique=True):
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(name + ' must be one dimensional')
    result = [str(value) for value in array]
    if not result or any(not value.strip() or value != value.strip() or
                         value.upper() in {'NAN', 'NONE', 'NA'} for value in result):
        raise ValueError('invalid ' + name)
    if unique and len(set(result)) != len(result):
        raise ValueError('duplicate ' + name)
    return result


def _matrix(values, name):
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or np.isinf(result).any():
        raise ValueError(name + ' must be a matrix with finite values or NaN')
    return result


def _schema(value):
    if value is None:
        raise ValueError('assay_schema is required')
    try:
        return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))
    except (ValueError, TypeError) as error:
        raise ValueError('assay_schema must be JSON serializable') from error


def _stats(values):
    observed = np.isfinite(values)
    count = observed.sum(axis=0)
    mean = np.divide(np.where(observed, values, 0.).sum(0), count,
                     out=np.zeros(values.shape[1]), where=count > 0)
    delta = np.where(observed, values - mean, 0.)
    sd = np.sqrt(np.divide((delta * delta).sum(0), count,
                          out=np.zeros(values.shape[1]), where=count > 0))
    return count, mean, sd


def _panel(x, maximum):
    count, _, sd = _stats(x)
    eligible = np.flatnonzero((count >= max(5, .8 * len(x))) & (sd > 1e-8))
    return eligible[np.lexsort((eligible, -sd[eligible]))[:maximum]]


def _pca_fit(x, panel, rank):
    """Fit preprocessing on the fixed shared panel; never reselect by domain."""
    selected = x[:, panel]
    if not len(panel) or len(x) < 3:
        state = dict(panel=panel.copy(), fill=np.zeros(len(panel)), center=np.zeros(len(panel)),
            scale=np.ones(len(panel)), design_center=np.zeros(2 * len(panel)),
            basis=np.empty((0, 2 * len(panel))), zscale=np.empty(0))
        return np.empty((len(x), 0)), state
    count, _, _ = _stats(selected)
    fill = np.zeros(len(panel))
    for index in np.flatnonzero(count):
        fill[index] = np.median(selected[np.isfinite(selected[:, index]), index])
    raw = np.where(np.isfinite(selected), selected, fill)
    center, scale = raw.mean(0), raw.std(0)
    scale = np.where(scale > 1e-8, scale, 1.)
    design = np.concatenate(((raw - center) / scale, (~np.isfinite(selected)).astype(float)), axis=1)
    design_center = design.mean(0)
    centered = design - design_center
    if np.max(np.abs(centered), initial=0.) <= 1e-12:
        basis = np.empty((0, design.shape[1]))
    else:
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        components = min(rank, len(x) - 2, int(np.sum(singular > 1e-10)))
        basis = vt[:components]
    z = centered @ basis.T
    zscale = np.maximum(z.std(0), 1e-8)
    return z / zscale, dict(panel=panel.copy(), fill=fill, center=center, scale=scale,
                           design_center=design_center, basis=basis, zscale=zscale)


def _pca_predict(x, state):
    selected = x[:, state['panel']]
    raw = np.where(np.isfinite(selected), selected, state['fill'])
    design = np.concatenate(((raw - state['center']) / state['scale'],
                             (~np.isfinite(selected)).astype(float)), axis=1)
    return (design - state['design_center']) @ state['basis'].T / state['zscale']


def _safe_anchor(x, y, source_ids, target_ids, fallback, alpha):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        state = list(fit_anchor(x, y, source_ids, target_ids, alpha))
    count, mean, _ = _stats(y)
    local_mean = np.where(count > 0, mean, fallback)
    state[2] = np.where(np.isfinite(state[2]), state[2], local_mean)
    state[2][~state[4]] = local_mean[~state[4]]
    return tuple(state), count, local_mean


def _patient_folds(patient_ids, seed):
    if len(patient_ids) < 5:
        return []
    stable = np.argsort(np.asarray(patient_ids), kind='stable')
    shuffled = stable[np.random.default_rng(seed).permutation(len(stable))]
    allocation = np.empty(len(stable), dtype=int)
    allocation[shuffled] = np.arange(len(stable)) % 5
    return [(np.flatnonzero(allocation != fold), np.flatnonzero(allocation == fold)) for fold in range(5)]


def _fit_expert(x, y, source_ids, target_ids, patient_ids, panel, pooled_mean, alpha, rank, seed):
    anchor, count, local_mean = _safe_anchor(x, y, source_ids, target_ids, pooled_mean, alpha)
    target_supported = count >= 5
    residual = np.full_like(y, np.nan)
    traces = []
    for fit, held in _patient_folds(patient_ids, seed):
        # Unsupported OOF columns are excluded, never filled using held labels.
        subanchor, subcount, _ = _safe_anchor(x[fit], y[fit], source_ids, target_ids,
                                            np.zeros(y.shape[1]), alpha)
        predicted = predict_anchor(x[held], subanchor)
        values = y[held] - predicted
        values[:, subcount < 5] = np.nan
        residual[held] = values
        traces.append(dict(fit_patient_ids=[patient_ids[i] for i in fit],
            held_patient_ids=[patient_ids[i] for i in held],
            target_supported=(subcount >= 5)))
    z, pca = _pca_fit(x, panel, rank)
    design = np.column_stack((np.ones(len(x)), z))
    coefficients = np.zeros((design.shape[1], y.shape[1]))
    residual_count = np.isfinite(residual).sum(0)
    context_supported = target_supported & (residual_count >= 5)
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.
    for column in np.flatnonzero(context_supported):
        observed = np.isfinite(residual[:, column])
        a, b = design[observed], residual[observed, column]
        coefficients[:, column] = np.linalg.solve(a.T @ a / len(b) + penalty, a.T @ b / len(b))
    return dict(anchor=anchor, pca=pca, coefficients=coefficients,
        target_observed=count, target_supported=target_supported,
        context_supported=context_supported, residual_observed=residual_count,
        fallback_mean=local_mean, fallback_source=['domain_mean' if value else 'pooled_mean' for value in count],
        source_count=_stats(x)[0], patient_ids=list(patient_ids), oof_splits=traces,
        target_fit_scale='native assay reference; local expert parameters use own-domain targets only',
        alpha=float(alpha), rank=int(rank))


def _predict_expert(expert, x):
    anchor = predict_anchor(x, expert['anchor'])
    z = _pca_predict(x, expert['pca'])
    full = anchor + np.column_stack((np.ones(len(x)), z)) @ expert['coefficients']
    unsupported = ~expert['target_supported']
    anchor[:, unsupported] = expert['fallback_mean'][unsupported]
    full[:, unsupported] = expert['fallback_mean'][unsupported]
    if not np.isfinite(anchor).all() or not np.isfinite(full).all():
        raise ValueError('nonfinite expert prediction')
    return anchor, full


def fit_domain_experts(x_train, y_train, domain_ids, source_gene_ids, target_gene_ids,
                       patient_ids, *, assay_schema, alpha=.1, rank=16,
                       max_predictors=1000, seed=20260907):
    x, y = _matrix(x_train, 'x_train'), _matrix(y_train, 'y_train')
    source, target = _ids(source_gene_ids, 'source_gene_ids'), _ids(target_gene_ids, 'target_gene_ids')
    patients, domains = _ids(patient_ids, 'patient_ids'), _ids(domain_ids, 'domain_ids', unique=False)
    if x.shape != (len(patients), len(source)) or y.shape != (len(patients), len(target)) or len(domains) != len(patients):
        raise ValueError('training shape or identity mismatch')
    if not np.isfinite(alpha) or alpha <= 0 or rank < 1 or max_predictors < 1:
        raise ValueError('invalid regression configuration')
    schema = _schema(assay_schema)
    observed, mean, sd = _stats(y)
    if np.any(observed == 0):
        raise ValueError('target has no observed pooled training reference')
    reference = dict(mean=mean, sd=np.where(sd > 1e-8, sd, 1.),
        observed=observed, target_gene_ids=list(target), native_scale='caller-provided log/research assay reference')
    # This shared panel is selected once, before fitting any domain expert.
    panel = _panel(x, int(max_predictors))
    ordered_domains = sorted(set(domains))
    experts, gate_mean, gate_sd = [], [], []
    gate_eligible = np.ones(len(panel), dtype=bool)
    for domain in ordered_domains:
        rows = np.flatnonzero(np.asarray(domains) == domain)
        own_x, own_y = x[rows], y[rows]
        expert = _fit_expert(own_x, own_y, source, target, [patients[i] for i in rows],
                             panel, mean, alpha, rank, int(seed))
        expert['domain_id'] = domain
        experts.append(expert)
        count, center, scale = _stats(own_x[:, panel])
        gate_eligible &= (count >= 5) & (scale > 1e-8)
        gate_mean.append(center)
        gate_sd.append(scale)
    gate_panel = panel[gate_eligible]
    pooled = _fit_expert(x, y, source, target, patients, panel, mean, alpha, rank, int(seed))
    model = dict(format=FORMAT, scientific_claim=False, source_gene_ids=source,
        target_gene_ids=target, training_patient_ids=patients, assay_schema=json.loads(schema),
        assay_schema_canonical=schema, output_reference=reference, domain_ids=ordered_domains,
        context_panel=panel, context_gene_ids=[source[i] for i in panel],
        gate_panel=gate_panel, gate_gene_ids=[source[i] for i in gate_panel],
        gate_mean=np.asarray(gate_mean)[:, gate_eligible],
        gate_sd=np.asarray(gate_sd)[:, gate_eligible], experts=experts, pooled=pooled,
        settings=dict(alpha=float(alpha), rank=int(rank), max_predictors=int(max_predictors), seed=int(seed)),
        gate='softmax(-0.5 * mean squared source distance standardized by each training domain); shared observed source panel only',
        caveats=['Training-selected context panel is shared across experts.',
            'Expert parameters use own-domain inputs/targets, with declared pooled-mean fallback for absent targets.',
            'Five patient-held anchor OOF residuals train context; no query targets or query domain labels.',
            'Gate is fixed and source-only, not learned from held outcomes; disagreement is not calibrated uncertainty.'])
    return model


def predict_domain_experts(model, x_query, source_gene_ids, patient_ids, *, assay_schema):
    if model.get('format') != FORMAT or _schema(assay_schema) != model['assay_schema_canonical']:
        raise ValueError('model or assay schema mismatch')
    query = _matrix(x_query, 'x_query')
    genes, patients = _ids(source_gene_ids, 'source_gene_ids'), _ids(patient_ids, 'patient_ids')
    if query.shape != (len(patients), len(genes)):
        raise ValueError('query shape or identity mismatch')
    if set(patients) & set(model['training_patient_ids']):
        raise ValueError('query patient overlaps training patients')
    lookup = {gene: index for index, gene in enumerate(model['source_gene_ids'])}
    if set(genes) - set(lookup):
        raise ValueError('unrecognized query source gene identity')
    x = np.full((len(query), len(lookup)), np.nan)
    x[:, [lookup[gene] for gene in genes]] = query
    pooled_anchor, pooled_context = _predict_expert(model['pooled'], x)
    predictions = np.stack([_predict_expert(expert, x)[1] for expert in model['experts']], axis=1)
    panel = model['gate_panel']
    selected = x[:, panel]
    observed = np.isfinite(selected)
    n_observed = observed.sum(1)
    distances = np.zeros((len(query), len(model['experts'])))
    for index in range(len(model['experts'])):
        if len(panel):
            z = np.clip((selected - model['gate_mean'][index]) / model['gate_sd'][index], -1e100, 1e100)
            distances[:, index] = np.divide(np.where(observed, z ** 2, 0.).sum(1), n_observed,
                                            out=np.zeros(len(query)), where=n_observed > 0)
    log_weight = -.5 * distances
    weight = np.exp(log_weight - log_weight.max(1, keepdims=True))
    weight /= weight.sum(1, keepdims=True)
    reversed_weight = weight[:, ::-1]
    gated = np.einsum('nd,ndt->nt', weight, predictions)
    reference = model['output_reference']
    native = dict(train_mean=np.broadcast_to(reference['mean'], pooled_anchor.shape).copy(),
        pooled_anchor=pooled_anchor, pooled_context=pooled_context,
        uniform_experts=predictions.mean(1), gated_experts=gated,
        reversed_gate=np.einsum('nd,ndt->nt', reversed_weight, predictions))
    standardized = {name: (values - reference['mean']) / reference['sd'] for name, values in native.items()}
    disagreement = np.sqrt(np.einsum('nd,ndt->nt', weight, (predictions - gated[:, None, :]) ** 2))
    return dict(native=native, standardized=standardized, expert_predictions=predictions,
        expert_predictions_standardized=(predictions - reference['mean']) / reference['sd'],
        weights=weight, reversed_weights=reversed_weight, distance=distances,
        source_coverage=n_observed / len(panel) if len(panel) else np.zeros(len(query)),
        source_observed=n_observed, source_panel_size=len(panel), disagreement=disagreement,
        disagreement_standardized=disagreement / reference['sd'], output_reference=reference,
        patient_ids=patients, target_gene_ids=list(model['target_gene_ids']), domain_ids=list(model['domain_ids']),
        scientific_claim=False, uncertainty_calibrated=False)


def _encode(value, arrays):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError('object arrays cannot be saved')
        key = 'array_' + str(len(arrays))
        arrays[key] = value
        return {'__array__': key}
    if isinstance(value, dict):
        return {key: _encode(item, arrays) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item, arrays) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _decode(value, arrays):
    if isinstance(value, dict):
        if set(value) == {'__array__'}:
            return arrays[value['__array__']].copy()
        return {key: _decode(item, arrays) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item, arrays) for item in value]
    return value


def save_domain_experts(model, output_directory):
    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=False)
    arrays = {}
    payload = _encode(model, arrays)
    np.savez_compressed(root / 'arrays.npz', **arrays)
    archive_hash = hashlib.sha256((root / 'arrays.npz').read_bytes()).hexdigest()
    with (root / 'model.json').open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(dict(format=FORMAT, arrays_sha256=archive_hash, model=payload), handle,
                  sort_keys=True, indent=2, allow_nan=False)
    return root


def load_domain_experts(input_directory):
    root = Path(input_directory)
    payload = json.loads((root / 'model.json').read_text(encoding='utf-8'))
    if payload.get('format') != FORMAT:
        raise ValueError('unsupported model format')
    if hashlib.sha256((root / 'arrays.npz').read_bytes()).hexdigest() != payload['arrays_sha256']:
        raise ValueError('model array checksum mismatch')
    with np.load(root / 'arrays.npz', allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    model = _decode(payload['model'], arrays)
    if model.get('format') != FORMAT or _schema(model['assay_schema']) != model['assay_schema_canonical']:
        raise ValueError('inconsistent saved model metadata')
    return model

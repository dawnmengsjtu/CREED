"""Independent, stagewise full-protein M-query development verification.

No producer or bridge mathematics is imported. The prior M-query receipt is
inherited only for its bound preparation and fixed reference predictions.
Every new numerical stage is independently checked before its exact frozen
arrays are used at the next stage, avoiding positive-distance median rank
discontinuities caused by propagating otherwise harmless reconstruction bits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

import verify_gdsc_metabolite_query as old

require, same, close = old.require, old.same, old.close
digest, read_json, arrays = old.digest, old.read_json, old.arrays
BINDINGS_SHA = '457e45bcb80adec8cbf375ff4701453b252f9db8d87d2ac8568057a04292809a'
N_SAMPLES, M_WIDTH, PROTEIN_WIDTH = 421, 225, 6692
SOURCE_FIELDS = ('m', 'auc', 'sample_ids', 'metabolite_ids', 'drug_ids', 'tissues', 'fold_ids')
PREDICTION_FIELDS = ('fit', 'held', 'panel', 'drug_ids', 'm_only', 'virtual_rna', 'virtual_fm')
PROTEIN_FILES = ('manifest.json', 'proteins.npy', 'protein_ids.npy', 'sample_ids.npy')
VIRTUAL = ('virtual_p', 'row_shuffle_p', 'observed_value_shuffle_p', 'ordinary_m', 'virtual_mask')
DIAGNOSTIC = ('measured_p', 'measured_mask')
ARMS = ('m_only', *VIRTUAL, *DIAGNOSTIC)
FROZEN = dict(frozen_virtual_rna='virtual_rna', frozen_virtual_fm='virtual_fm')
SCORED_ARMS = (*ARMS, *FROZEN)
CONTROLS = ('m_only', 'row_shuffle_p', 'observed_value_shuffle_p', 'ordinary_m', 'virtual_mask', *FROZEN)
MARGIN = 1e-10
UPSTREAM = ('selection.json', 'panel.npz', 'generator-0.npz', 'generator-1.npz', 'generator-2.npz',
            'train-native.npz', 'query-native.npz', 'measured-inputs.npz', 'm-kernel.npz',
            'encodings.npz', 'encoded-train.npz', 'encoded-query.npz', 'extra-kernels.npz')
BANK_FILES = (*UPSTREAM, 'upstream-frozen.json', 'per-alpha-predictions.npz',
              'query-generator-predictions.npz', 'predictions-frozen.json')
CODE_FILES = dict(runner='scripts/run_gdsc_metabolite_full_protein.py',
    bridge='src/creed/science/metabolite_protein_query.py', parent_io='scripts/run_gdsc_metabolite_query.py',
    masked='src/creed/science/masked_task_kernel.py', coordinates='src/creed/science/metabolite_query.py',
    task='src/creed/science/virtual_state_task.py', parent_metric_check='scripts/run_gdsc_metabolite_task_metric.py',
    io='scripts/run_phospho_fewshot.py')


def parent_names():
    return {'source-inputs.npz', 'splits.json', *[f'fold-{f}/{n}' for f in range(5)
            for n in ('selection.json', 'predictions.npz')]}


def expected_artifacts():
    names = {'frozenconfig.json', 'bindings.json', 'splits.json', 'source-inputs.npz',
             'ordinary-projection.npz', 'all-predictions-frozen.json', 'scores.json'}
    for fold in range(5):
        prefix = f'fold-{fold}/'
        names.update(prefix + n for n in ('selection.json', 'm-only-replay.npz', 'selected.npz',
                                         'predictions.npz', 'scoring-reference.npz'))
        for bank in ('inner-0', 'inner-1', 'inner-2', 'outer'):
            names.update(prefix + bank + '/' + n for n in BANK_FILES)
            if bank != 'outer': names.add(prefix + bank + '/validation.npz')
    require(len(names) == 387, 'checker artifact contract')
    return names


def exact_json(actual, expected, name, *, tolerance=False):
    """Exact identities/structure; narrowly tolerant verified-parent scores."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), name + ': keys')
        for key in expected: exact_json(actual[key], expected[key], name + '/' + key, tolerance=tolerance)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), name + ': length')
        for i, value in enumerate(expected): exact_json(actual[i], value, name + '/' + str(i), tolerance=tolerance)
    elif isinstance(expected, (float, np.floating)) and tolerance:
        require(not isinstance(actual, bool) and isinstance(actual, (int, float))
                and np.isfinite(actual) and np.isfinite(expected), name + ': finite score')
        close(actual, expected, name, rtol=2e-14, atol=2e-14)
    else:
        require(type(actual) is type(expected) and actual == expected, name + ': exact value')


class Sources:
    def __init__(self, parent, verification, protein, source_bindings):
        self.root, self.protein = Path(parent).resolve(strict=True), Path(protein).resolve(strict=True)
        self.verification, self.binding_path = Path(verification), Path(source_bindings)
        require(digest(self.binding_path) == BINDINGS_SHA, 'exact source binding SHA')
        self.binding = read_json(self.binding_path)
        require(self.binding['schema'] == 'creed.gdsc-metabolite-full-protein.bound-sources.v1'
                and self.binding['scientific_claim'] is False, 'source binding schema/scope')
        require(set(self.binding['parent_files']) == parent_names()
                and set(self.binding['protein_files']) == set(PROTEIN_FILES), 'source object whitelist')
        self.summary, self.checked = read_json(self.root / 'summary.json'), read_json(self.verification)
        require(self.summary['status'] == 'completed' and self.checked['status'] == 'verified'
                and self.summary['scientific_claim'] is False and self.checked['scientific_claim'] is False,
                'verified development parent required')
        require(self.checked['producer_summary_sha256'] == self.binding['parent_summary_sha256']
                and self.checked['checked_artifacts'] == self.summary['artifacts'], 'parent receipt artifacts')
        exact_json(self.checked['mean_fold_scores'], self.summary['mean_fold_scores'], 'parent means', tolerance=True)
        exact_json(self.checked['fold_metrics'], self.summary['fold_metrics'], 'parent metrics', tolerance=True)
        require(all(self.summary['artifacts'].get(k) == v for k, v in self.binding['parent_files'].items()),
                'parent object binding inherited from receipt')
        self.parent_fields, self.protein_fields = {}, {}
        self.finish()

    def path(self, kind, name):
        require(kind in ('parent', 'protein') and name in self.binding[kind + '_files'], 'unlisted source object')
        root = self.root if kind == 'parent' else self.protein
        path = root / name
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root), 'source path boundary')
        require(digest(path) == self.binding[kind + '_files'][name], 'source SHA: ' + name)
        return path

    def npz(self, name):
        fields = SOURCE_FIELDS if name == 'source-inputs.npz' else PREDICTION_FIELDS if name in parent_names() and name.endswith('/predictions.npz') else None
        require(fields is not None, 'unlisted source NPZ fields')
        self.parent_fields[name] = list(fields)
        with np.load(self.path('parent', name), allow_pickle=False) as value:
            return {key: value[key] for key in fields}

    def json(self, name):
        require(name in parent_names() and name.endswith('.json'), 'unlisted source JSON')
        self.parent_fields[name] = 'json'
        return read_json(self.path('parent', name))

    def finish(self):
        require(digest(self.binding_path) == BINDINGS_SHA
                and digest(self.root / 'summary.json') == self.binding['parent_summary_sha256']
                and digest(self.verification) == self.binding['parent_verification_sha256'], 'parent binding changed')
        for kind in ('parent', 'protein'):
            for name in self.binding[kind + '_files']: self.path(kind, name)


def load_source(bound):
    source, split = bound.npz('source-inputs.npz'), bound.json('splits.json')
    manifest = read_json(bound.path('protein', 'manifest.json')); bound.protein_fields['manifest.json'] = 'json'
    require(manifest['scope'] == 'exploratory 421 training prefix only, expanded missing-aware protein source'
            and manifest['scientific_claim'] is False
            and manifest['source_sha256'] == '468767f19ea2c935285c0d0877475ea97cb70140662b307f39e841d91a81e5d1', 'safe protein lineage')
    values = {}
    for name in PROTEIN_FILES[1:]:
        path = bound.path('protein', name); member = manifest['files'][name]
        require(member['sha256'] == bound.binding['protein_files'][name]
                and member['bytes'] == path.stat().st_size, 'protein member SHA/bytes')
        values[name] = np.load(path, allow_pickle=False); bound.protein_fields[name] = 'npy'
    source.update(p=values['proteins.npy'], protein_ids=values['protein_ids.npy'])
    same(values['sample_ids.npy'], source['sample_ids'], 'protein patient order')
    for key in ('sample_ids', 'tissues', 'fold_ids'): same(source[key], split[key], 'split/' + key)
    for key, size in (('sample_ids', N_SAMPLES), ('metabolite_ids', M_WIDTH), ('protein_ids', PROTEIN_WIDTH), ('drug_ids', len(source['drug_ids']))):
        ids = source[key]
        require(ids.shape == (size,) and len(set(ids.tolist())) == size
                and all(isinstance(v, str) and v and v.strip() == v for v in ids.tolist()), 'source identities/' + key)
    require(source['m'].shape == (N_SAMPLES, M_WIDTH) and np.isfinite(source['m']).all()
            and source['p'].shape == (N_SAMPLES, PROTEIN_WIDTH) and not np.isinf(source['p']).any()
            and source['auc'].shape == (N_SAMPLES, len(source['drug_ids'])) and not np.isinf(source['auc']).any(), 'source shapes/nonfinite')
    same(old.tissue_partition(source['tissues'], 5)[0], source['fold_ids'], 'original tissue folds')
    return source, split


def direct_squared_distances(a, b=None):
    a = old.finite_matrix(a, 'direct distance source'); b = a if b is None else old.finite_matrix(b, 'direct distance query')
    require(a.shape[1] == b.shape[1], 'direct distance width')
    result = np.empty((len(a), len(b)))
    for i, row in enumerate(a):
        delta = b - row
        result[i] = np.einsum('ij,ij->i', delta, delta)
    require(np.isfinite(result).all(), 'direct distance overflow')
    return result


def zero_moments(value):
    counts, mean, sd = old.moments(value)
    return counts, np.where(counts, mean, 0.), np.where(counts, sd, 0.)


def panel_reference(p, tissues):
    folds, _ = old.tissue_partition(tissues, 3)
    counts, _, sd = zero_moments(p)
    splits = [dict(fold=g, fit=np.flatnonzero(folds != g), held_local=np.flatnonzero(folds == g)) for g in range(3)]
    moments = [zero_moments(p[s['fit']]) for s in splits]
    gc, gs = np.stack([v[0] for v in moments]), np.stack([v[2] for v in moments])
    a_fail, c_fail, s_fail = 2 * counts < len(p), (gc < 20).any(0), (gs <= 1e-8).any(0)
    keep = ~(a_fail | c_fail | s_fail)
    require(keep.any(), 'empty eligible full-protein panel')
    return dict(indices=np.flatnonzero(keep), eligible=keep, unsupported=~keep, a_counts=counts, a_sd=sd,
                generator_counts=gc, generator_sd=gs, fold_ids=folds, source_width=p.shape[1],
                a_coverage_failed=a_fail, generator_count_failed=c_fail, generator_sd_failed=s_fail), splits


def targets_reference(p, ordinary, tissues):
    observed = np.isfinite(p)
    same(np.isfinite(ordinary), observed, 'ordinary M observation mask')
    permutation, singleton, fixed = old.tissue_permutation(tissues)
    value_perm = np.full(p.shape, -1, int); value_fixed = np.zeros(p.shape[1], int); value_singleton = value_fixed.copy()
    shuffled = np.full(p.shape, np.nan); rng = np.random.Generator(np.random.PCG64(20260910))
    for column in range(p.shape[1]):
        for tissue in sorted(set(tissues.tolist())):
            rows = np.flatnonzero(observed[:, column] & (tissues == tissue))
            if not len(rows): continue
            donors = rng.permutation(rows)
            value_perm[rows, column] = donors; shuffled[rows, column] = p[donors, column]
            value_fixed[column] += np.count_nonzero(rows == donors); value_singleton[column] += len(rows) == 1
    targets = np.concatenate((p, p[permutation], shuffled, ordinary, observed.astype(float)), axis=1)
    mask = np.isfinite(targets); groups = {}; group_ids = np.empty(mask.shape[1], int)
    for j in range(mask.shape[1]):
        key = tuple(np.flatnonzero(mask[:, j]).tolist())
        if key not in groups: groups[key] = []
        groups[key].append(j)
    for i, columns in enumerate(groups.values()): group_ids[columns] = i
    return targets, groups, dict(target_observed=mask, observed_mask=observed, target_counts=mask.sum(0),
        target_group_ids=group_ids, group_count=len(groups), row_permutation=permutation, value_permutation=value_perm,
        value_fixed_points=value_fixed, value_singleton_rows=value_singleton, row_fixed_points=fixed, row_singleton_rows=singleton)


def generator_reference(m, p, ordinary, tissues, source_indices, source_width):
    z, mean, sd = old.standardized(m)
    dist = direct_squared_distances(z); width = old.positive_median(dist); kernel = np.exp(-dist / (2 * width))
    targets, groups, metadata = targets_reference(p, ordinary, tissues)
    count, target_mean, _ = zero_moments(targets); coef = np.zeros_like(targets)
    for row_tuple, columns in groups.items():
        rows = np.array(row_tuple, int); require(len(rows) >= 20, 'observed generator minimum')
        values, vectors = eigh(kernel[np.ix_(rows, rows)], driver='evd', check_finite=False)
        centered = targets[np.ix_(rows, columns)] - target_mean[columns]
        coef[np.ix_(rows, columns)] = vectors @ ((vectors.T @ centered) / (values + 1.)[:, None])
    return dict(schema='creed.metabolite-protein-query-generator.v1', mean=mean, sd=sd, x=z, width=width,
        coef=coef, target_mean=target_mean, **metadata, cuts=np.arange(6) * p.shape[1], arms=np.array(VIRTUAL),
        tissues=tissues, permutation_seed=20260910, alpha=1., feature_count=m.shape[1], protein_count=p.shape[1],
        source_protein_indices=source_indices, source_protein_count=source_width)


def predict_verified_generator(model, query):
    z = (query - model['mean']) / model['sd']
    cross = np.exp(-direct_squared_distances(z, model['x']) / (2 * float(model['width'])))
    result = cross @ model['coef'] + model['target_mean']
    return dict(zip(VIRTUAL, np.split(result, 5, axis=1)))


def measured_reference(p_fit, p_query):
    _, mean, _ = zero_moments(p_fit)
    require(np.isfinite(p_fit).any(0).all(), 'measured diagnostic observed training means')
    return dict(fill_mean=mean, measured_p_train=np.where(np.isfinite(p_fit), p_fit, mean),
                measured_mask_train=np.isfinite(p_fit).astype(float),
                measured_p_query=np.where(np.isfinite(p_query), p_query, mean)[None],
                measured_mask_query=np.isfinite(p_query).astype(float)[None])


def encoding_reference(train, queries):
    z, mean, sd = old.standardized(train)
    queries = np.asarray(queries, float)
    require(queries.ndim == 3 and queries.shape[2] == z.shape[1] and np.isfinite(queries).all(), 'encoded query dimensions')
    q = (queries - mean) / sd
    constant = bool(np.max(np.ptp(z, axis=0)) < 1e-12)
    return dict(mean=mean, sd=sd, constant=constant), z, q


def kernel_from_encoded(z, queries, constant):
    if constant:
        return 0., np.ones((len(z), len(z))), np.ones((len(queries), queries.shape[1], len(z)))
    dist = direct_squared_distances(z); width = old.positive_median(dist)
    return width, np.exp(-dist / (2 * width)), np.stack([np.exp(-direct_squared_distances(q, z) / (2 * width)) for q in queries])


def check_bank(files, prefix, source, fit, query, drug_panel, projection):
    require(not set(fit) & set(query) and not set(source['tissues'][fit]) & set(source['tissues'][query]), 'bank whole-tissue boundary')
    get = lambda name: files.array(prefix + '/' + name)
    panel, splits = panel_reference(source['p'][fit], source['tissues'][fit]); pi = panel['indices']
    old.json_check(files.json(prefix + '/selection.json'), dict(fit=fit.tolist(), query=query.tolist(), panel=drug_panel.tolist(),
        source_protein_indices=pi.tolist(), generator_splits=[{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in s.items()} for s in splits]), prefix + '/selection')
    old.compare_arrays(get('panel.npz'), dict(**panel, source_protein_ids=source['protein_ids'], protein_ids=source['protein_ids'][pi]), prefix + '/panel')
    m, p = source['m'][fit], source['p'][fit][:, pi]
    ordinary = np.where(np.isfinite(source['p'][fit]), m @ projection, np.nan)[:, pi]
    train = {a: np.empty((len(fit), len(pi))) for a in VIRTUAL}
    queries = {a: [] for a in VIRTUAL}
    for g, split in enumerate(splits):
        a, held = split['fit'], split['held_local']
        expected = generator_reference(m[a], p[a], ordinary[a], source['tissues'][fit[a]], pi, PROTEIN_WIDTH)
        saved = get(f'generator-{g}.npz')
        old.compare_arrays(saved, expected, prefix + f'/generator-{g}')
        # The independently checked exact model is now the input boundary.
        held_predictions = predict_verified_generator(saved, m[held])
        query_predictions = predict_verified_generator(saved, source['m'][query])
        for arm in VIRTUAL: train[arm][held] = held_predictions[arm]; queries[arm].append(query_predictions[arm])
    queries = {a: np.stack(v) for a, v in queries.items()}
    old.compare_arrays(get('train-native.npz'), train, prefix + '/train-native')
    old.compare_arrays(get('query-native.npz'), queries, prefix + '/query-native')
    train, queries = get('train-native.npz'), get('query-native.npz')
    measured = measured_reference(source['p'][fit][:, pi], source['p'][query][:, pi])
    old.compare_arrays(get('measured-inputs.npz'), measured, prefix + '/measured-inputs')
    measured = get('measured-inputs.npz')
    for arm in DIAGNOSTIC: train[arm], queries[arm] = measured[arm + '_train'], measured[arm + '_query']
    base = old.kernel_reference(m, (source['m'][query],), allow_constant=False)
    old.compare_arrays(get('m-kernel.npz'), dict(mean=base['mean'], sd=base['sd'], train=base['x'],
        query=(source['m'][query] - base['mean']) / base['sd'] / np.sqrt(M_WIDTH), width=base['width'],
        kernel=base['kernel'], cross=base['crosses'][0]), prefix + '/m-kernel')
    base = get('m-kernel.npz'); states, zs, qs, ks = get('encodings.npz'), get('encoded-train.npz'), get('encoded-query.npz'), get('extra-kernels.npz')
    require(set(zs) == set(qs) == set(VIRTUAL + DIAGNOSTIC), prefix + ': encoding arms')
    require(set(states) == {a + '_' + k for a in VIRTUAL + DIAGNOSTIC for k in ('mean', 'sd', 'constant', 'width')}, prefix + ': encoding state fields')
    require(set(ks) == {a + '_' + k for a in VIRTUAL + DIAGNOSTIC for k in ('train', 'query')}, prefix + ': extra kernel fields')
    y = source['auc'][np.ix_(fit, drug_panel)]
    components = {'m_only': old.masked_grid(base['kernel'], (base['cross'],), y)}
    for arm in VIRTUAL + DIAGNOSTIC:
        expected, _, _ = encoding_reference(train[arm], queries[arm])
        old.compare_arrays({k: states[arm + '_' + k] for k in expected}, expected, prefix + '/encoding/' + arm)
        # Standardization and kernels each get their own checked exact input.
        close(zs[arm], (train[arm] - states[arm + '_mean']) / states[arm + '_sd'], prefix + '/encoded-train/' + arm)
        close(qs[arm], (queries[arm] - states[arm + '_mean']) / states[arm + '_sd'], prefix + '/encoded-query/' + arm)
        width, kernel, cross = kernel_from_encoded(zs[arm], qs[arm], bool(states[arm + '_constant']))
        close(states[arm + '_width'], width, prefix + '/width/' + arm)
        close(ks[arm + '_train'], kernel, prefix + '/extra-kernel/' + arm)
        close(ks[arm + '_query'], cross, prefix + '/extra-cross/' + arm)
        components[arm] = old.masked_grid(.5 * base['kernel'] + .5 * ks[arm + '_train'],
            [.5 * base['cross'] + .5 * cross for cross in ks[arm + '_query']], y)
    files.check_frozen(prefix + '/upstream-frozen.json', [prefix + '/' + n for n in UPSTREAM])
    grids = {a: v.mean(0) for a, v in components.items()}
    old.compare_arrays(get('query-generator-predictions.npz'), components, prefix + '/components')
    old.compare_arrays(get('per-alpha-predictions.npz'), grids, prefix + '/alpha-grid')
    saved_components, saved_grids = get('query-generator-predictions.npz'), get('per-alpha-predictions.npz')
    for arm in ARMS: same(saved_grids[arm], saved_components[arm].mean(0), prefix + '/mean-after-head/' + arm)
    files.check_frozen(prefix + '/predictions-frozen.json', [prefix + '/' + n for n in ('per-alpha-predictions.npz', 'query-generator-predictions.npz')])
    return saved_grids, dict(selected_proteins=len(pi), unsupported_proteins=int(panel['unsupported'].sum()))


class ResultFiles(old.ResultFiles):
    def check_frozen(self, relative, names):
        expected = {Path(n).name: self.expected[n] for n in names}
        exact_json(self.json(relative), expected, relative)
        for name in names: self.path(name)


def check_fold(files, source, bound, fold, projection):
    prefix = f'fold-{fold}'; fit = np.flatnonzero(source['fold_ids'] != fold); held = np.flatnonzero(source['fold_ids'] == fold)
    inner, _ = old.tissue_partition(source['tissues'][fit], 3)
    parts = [dict(fit=fit[inner != g], held=fit[inner == g]) for g in range(3)]
    panel, _, _ = old.panel_reference(source['auc'], fit, [p['fit'] for p in parts])
    identity = dict(fit=fit, held=held, panel=panel, drug_ids=source['drug_ids'][panel])
    selection = dict(**{k: v.tolist() for k, v in identity.items()}, head_splits=[{k: v.tolist() for k, v in part.items()} for part in parts])
    old.json_check(files.json(prefix + '/selection.json'), selection, prefix + '/selection')
    old.json_check(bound.json(prefix + '/selection.json'), selection, prefix + '/parent-selection')
    reference = bound.npz(prefix + '/predictions.npz')
    old.compare_arrays(reference, identity, prefix + '/reference-identity', exact_keys=False)
    sums = {a + suffix: np.zeros((4, len(panel))) for a in ARMS for suffix in ('_loss', '_count')}; diagnostics = []
    for g, part in enumerate(parts):
        sub = prefix + f'/inner-{g}'
        grids, diag = check_bank(files, sub, source, part['fit'], part['held'], panel, projection)
        diagnostics.append(dict(fold=fold, bank=f'inner-{g}', **diag))
        expected = {}
        for arm in ARMS:
            loss, count, _ = old.loss_reference(grids[arm], source['auc'][np.ix_(part['held'], panel)], source['auc'][np.ix_(part['fit'], panel)])
            expected[arm + '_loss'], expected[arm + '_count'] = loss, count
        old.compare_arrays(files.array(sub + '/validation.npz'), expected, sub + '/validation')
        for key in sums: sums[key] += expected[key]
    chosen = {a: old.selection_reference(sums[a + '_loss'], sums[a + '_count']) for a in ARMS}
    old.compare_arrays(files.array(prefix + '/selected.npz'), dict(**chosen, **sums), prefix + '/selected')
    grids, diag = check_bank(files, prefix + '/outer', source, fit, held, panel, projection)
    diagnostics.append(dict(fold=fold, bank='outer', **diag))
    predictions = {a: old.selected_prediction(grids[a], chosen[a]) for a in ARMS}
    close(predictions['m_only'], reference['m_only'], prefix + '/exact-frozen-M', rtol=1e-12, atol=1e-12)
    old.compare_arrays(files.array(prefix + '/m-only-replay.npz'), dict(fit=fit, held=held, panel=panel,
        prediction=predictions['m_only'], frozen_prediction=reference['m_only'], selected=chosen['m_only'],
        loss=sums['m_only_loss'], count=sums['m_only_count']), prefix + '/m-only-replay')
    for arm, original in FROZEN.items(): predictions[arm] = reference[original]
    saved = files.array(prefix + '/predictions.npz')
    old.compare_arrays(saved, dict(**identity, **predictions), prefix + '/predictions')
    for arm, original in FROZEN.items():
        require(saved[arm].dtype == reference[original].dtype and saved[arm].tobytes() == reference[original].tobytes(), prefix + '/frozen-reference bytes')
    return saved, diagnostics, dict(fold=fold, predictions_sha256=files.expected[prefix + '/predictions.npz'],
        m_only_replayed=True, frozen_references_copied_exactly=True, fit_samples=len(fit), held_samples=len(held), drugs=len(panel))


def scores_reference(source, predictions, panel_diagnostics):
    rows, per_drug = [], []; seen = np.zeros(len(source['m']), int); references = []
    for fold, pred in enumerate(predictions):
        fit, held, panel = (pred[k] for k in ('fit', 'held', 'panel')); seen[held] += 1
        truth = source['auc'][np.ix_(held, panel)]; _, _, sd = old.moments(source['auc'][np.ix_(fit, panel)])
        count = np.isfinite(truth).sum(0)
        references.append(dict(truth=truth, train_sd=sd, observed=np.isfinite(truth), observed_counts=count, evaluated=count >= 5))
        for arm in SCORED_ARMS:
            metric = old.metric_reference(pred[arm], truth, sd)
            rows.append(dict(fold=fold, arm=arm, score=metric['score'], drug_ids=int(metric['evaluated'].sum()), observed_pairs=metric['observed_pairs']))
            per_drug.extend(dict(fold=fold, arm=arm, drug_id=str(drug), observed=int(count[j]), evaluated=bool(metric['evaluated'][j]),
                standardized_mse=float(metric['per_drug'][j]) if count[j] else None) for j, drug in enumerate(pred['drug_ids']))
    same(seen, np.ones(len(seen), int), 'five outer folds exactly once')
    means = {a: float(np.mean([r['score'] for r in rows if r['arm'] == a])) for a in SCORED_ARMS}
    lookup = {(r['fold'], r['arm']): r['score'] for r in rows}; comparisons = []; gates = dict(mean={}, sensitivity={})
    for control in CONTROLS:
        effects = [lookup[f, control] - lookup[f, 'virtual_p'] for f in range(5)]
        loo = [float(np.mean(effects[:f] + effects[f + 1:])) for f in range(5)]
        delta = means[control] - means['virtual_p']
        gates['mean'][control], gates['sensitivity'][control] = bool(delta > MARGIN), bool(min(loo) > MARGIN)
        comparisons.append(dict(candidate='virtual_p', control=control, mean_control_minus_candidate=delta,
            fold_effects=effects, leave_one_fold_out_effects=loo, folds_better=sum(d > MARGIN for d in effects)))
    return dict(mean_fold_scores=means, fold_metrics=rows, per_drug=per_drug, comparisons=comparisons, panel_diagnostics=panel_diagnostics), gates, references


def config_reference(code_root, source_bindings):
    code = Path(code_root)
    return dict(schema='creed.gdsc-metabolite-full-protein.development.v1', scientific_claim=False,
        external_confirmation=False, chainable=False, samples=N_SAMPLES, m_width=M_WIDTH, source_proteins=PROTEIN_WIDTH,
        arms=list(ARMS), frozen_references=FROZEN, primary='virtual_p', primary_controls=list(CONTROLS),
        alphas=old.ALPHAS.tolist(), extra_kernel_weight=.5, generator_alpha=1., min_a_protein_coverage=.5,
        min_generator_observed=20, min_generator_sd=1e-8, pairing_seed=20260910, ordinary_projection_seed=20260909,
        task_encoding='each arm own A-native/OOF mean and population SD once; full J; no second scaling',
        constant_extra_rule='training normalized max ptp <1e-12 makes train/all query kernels ones', comparison_margin=MARGIN,
        source_bindings_sha256=digest(source_bindings), protocol_sha256=digest(code / 'GDSC_METABOLITE_FULL_PROTEIN_NEXT.md'),
        code_sha256={k: digest(code / p) for k, p in CODE_FILES.items()})


def verify(*, result, parent, verification, protein, source_bindings, code_root, output):
    output = Path(output); require(not output.exists(), 'create-once verifier output already exists')
    result = Path(result).resolve(strict=True); summary = read_json(result / 'summary.json')
    require(summary['status'] == 'completed' and summary['scientific_claim'] is False, 'completed development producer')
    require(set(summary['artifacts']) == expected_artifacts(), 'exact 387 result artifacts')
    actual = {p.relative_to(result).as_posix() for p in result.rglob('*') if p.is_file()}
    require(actual == expected_artifacts() | {'summary.json'}, 'unexpected result files')
    files = ResultFiles(result, summary['artifacts'])
    bound = Sources(parent, verification, protein, source_bindings)
    config = config_reference(code_root, source_bindings)
    exact_json(files.json('frozenconfig.json'), config, 'frozen config')
    old.json_check(summary, config, 'summary config')
    exact_json(files.json('bindings.json'), bound.binding, 'source binding copy')
    source, split = load_source(bound)
    saved_source = files.array('source-inputs.npz')
    old.compare_arrays(saved_source, source, 'prepared source')
    for key in source: same(saved_source[key], source[key], 'prepared source exact/' + key)
    exact_json(files.json('splits.json'), split, 'original split copy')
    projection = np.random.Generator(np.random.PCG64(20260909)).standard_normal((M_WIDTH, PROTEIN_WIDTH)) / np.sqrt(M_WIDTH)
    projection_saved = files.array('ordinary-projection.npz')
    old.compare_arrays(projection_saved, dict(projection=projection, metabolite_ids=source['metabolite_ids'], protein_ids=source['protein_ids']), 'ordinary projection')
    same(projection_saved['projection'], projection, 'fixed projection exact')
    predictions, diagnostics, records = [], [], []
    for fold in range(5):
        pred, diag, record = check_fold(files, source, bound, fold, projection_saved['projection'])
        predictions.append(pred); diagnostics.extend(diag); records.append(record)
        print(json.dumps(dict(fold=fold, status='independently_checked')), flush=True)
    exact_json(files.json('all-predictions-frozen.json'), {str(r['fold']): r['predictions_sha256'] for r in records}, 'all prediction freeze')
    scores, gates, references = scores_reference(source, predictions, diagnostics)
    for fold, reference in enumerate(references): old.compare_arrays(files.array(f'fold-{fold}/scoring-reference.npz'), reference, f'fold-{fold}/scoring-reference')
    old.json_check(files.json('scores.json'), scores, 'scores'); old.json_check(summary, scores, 'summary scores')
    passed = dict(mean_gate_passed=all(gates['mean'].values()), sensitivity_gate_passed=all(gates['sensitivity'].values()),
                  global_gate_passed=all(gates['mean'].values()) and all(gates['sensitivity'].values()))
    exact_json(summary['gates'], gates, 'scientific gates'); old.json_check(summary, passed, 'scientific gate decisions')
    exact_json(summary['fold_records'], records, 'fold records')
    for new, prior in (('m_only', 'm_only'), *FROZEN.items()):
        close(scores['mean_fold_scores'][new], bound.summary['mean_fold_scores'][prior], 'frozen score/' + new, rtol=2e-14, atol=2e-14)
    exact_json(summary['parent_fields'], bound.parent_fields, 'parent field whitelist')
    exact_json(summary['protein_fields'], bound.protein_fields, 'protein field whitelist')
    exact_json(summary['parent_members'], bound.binding['parent_files'], 'parent member whitelist')
    exact_json(summary['protein_members'], bound.binding['protein_files'], 'protein member whitelist')
    require(summary['frozenconfig_sha256'] == files.expected['frozenconfig.json'], 'config SHA receipt')
    bound.finish()
    for name in expected_artifacts(): files.path(name)
    receipt = dict(status='verified', scientific_claim=False, external_confirmation=False, chainable=False,
        verification_mode='stagewise_verified_frozen_intermediates', producer_summary_sha256=digest(result / 'summary.json'),
        checked_artifacts=summary['artifacts'], source_bindings_sha256=digest(source_bindings),
        parent_summary_sha256=bound.binding['parent_summary_sha256'], parent_verification_sha256=bound.binding['parent_verification_sha256'],
        parent_fields=bound.parent_fields, protein_fields=bound.protein_fields, parent_members=bound.binding['parent_files'],
        protein_members=bound.binding['protein_files'], checked_banks=20, checked_generators=60, checked_inner_validations=15,
        checked_outer_folds=5, checked_artifact_count=387, checked_fold_metrics=50, checked_contrasts=7,
        **scores, gates=gates, **passed, checker_sha256=digest(__file__), independent_primitives_sha256=digest(old.__file__),
        limitations=['Bound prepared M/AUC source preparation and frozen RNA/FM predictions inherit the independent parent receipt; raw biological sources not re-read.',
            'Protein source member bytes, IDs, missing masks, all new reference models, predictions, heads, selection and scores independently checked.',
            'Exact checked intermediates are stage boundaries; this is not a bitwise end-to-end propagation of independently rounded arrays.',
            'Artifact freeze hashes and source whitelists are checked; saved files alone cannot prove wall-clock read ordering.',
            'Reused development; measured P/mask diagnostics require extra query assays; no uncertainty, full-layer recovery, or scientific-success claim.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream: json.dump(receipt, stream, indent=2, allow_nan=False)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('result', 'parent', 'verification', 'protein', 'source-bindings', 'code-root', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    verify(**vars(parser.parse_args()))

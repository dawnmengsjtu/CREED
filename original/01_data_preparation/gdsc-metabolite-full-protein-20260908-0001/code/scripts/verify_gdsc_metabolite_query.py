"""Independent metabolite-query GDSC reconstruction and task-utility checks.

No production model or runner is imported.  Kernel algebra, masked-label
regressions, projections, pairing controls, selection and scores are rebuilt
here from saved, identity-bound inputs.  Reused development only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

ARMS = ('m_only', 'virtual_rna', 'virtual_fm', 'ordinary_m', 'shuffle_rna',
        'shuffle_fm', 'measured_rna', 'measured_shuffle_rna')
VIRTUAL_ARMS = ('virtual_rna', 'virtual_fm', 'ordinary_m', 'shuffle_rna', 'shuffle_fm')
DEPLOYABLE_ARMS = ('m_only', *VIRTUAL_ARMS)
MEASURED_ARMS = ('measured_rna', 'measured_shuffle_rna')
ALPHAS = np.array([.1, 1., 10., 100.])
RNA_SEED, M_SEED, SHUFFLE_SEED = 20260908, 20260909, 20260910
WIDTH = 512
MARGIN = 1e-10
N_SAMPLES = 421
BINDINGS_SHA = 'b3e231b8065623770047d3f21653eceb867a06fb9a7af1977c21a1eca2ebba3f'
SCHEMA = 'creed.gdsc-metabolite-query.development.v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, message):
    try:
        np.testing.assert_array_equal(actual, expected)
    except AssertionError as error:
        raise ValueError(message) from error


def close(actual, expected, message, *, rtol=2e-7, atol=2e-9):
    try:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
    except AssertionError as error:
        raise ValueError(message) from error


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    def unique(pairs):
        output = {}
        for key, value in pairs:
            require(key not in output, 'duplicate JSON key: ' + key)
            output[key] = value
        return output
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream, object_pairs_hook=unique)


def arrays(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def finite_matrix(value, name):
    value = np.asarray(value, dtype=np.float64)
    require(value.ndim == 2 and value.shape[0] > 0 and value.shape[1] > 0
            and np.isfinite(value).all(), name + ': finite nonempty matrix')
    return value


def moments(value):
    """Observed-label moments, with no label imputation or silent zero SD."""
    y = np.asarray(value, dtype=np.float64)
    require(y.ndim == 2 and not np.isinf(y).any(), 'label matrix')
    count = np.isfinite(y).sum(axis=0)
    mean = np.full(y.shape[1], np.nan)
    sd = np.full(y.shape[1], np.nan)
    for column in np.flatnonzero(count):
        observed = y[np.isfinite(y[:, column]), column]
        mean[column] = observed.sum() / len(observed)
        sd[column] = np.sqrt(np.square(observed - mean[column]).sum() / len(observed))
    return count, mean, sd


def standardized(value):
    x = finite_matrix(value, 'training features')
    mean = np.add.reduce(x, axis=0) / len(x)
    sd = np.sqrt(np.add.reduce(np.square(x - mean), axis=0) / len(x))
    sd[sd < 1e-8] = 1.
    return (x - mean) / sd, mean, sd


def squared_distances(first, second=None):
    first = finite_matrix(first, 'distance first')
    second = first if second is None else finite_matrix(second, 'distance second')
    require(first.shape[1] == second.shape[1], 'distance feature order/width')
    distances = (np.einsum('ij,ij->i', first, first)[:, None]
                 + np.einsum('ij,ij->i', second, second)[None, :]
                 - 2. * (first @ second.T))
    np.maximum(distances, 0., out=distances)
    if first is second:
        np.fill_diagonal(distances, 0.)
    return distances


def positive_median(distances):
    require(distances.ndim == 2 and distances.shape[0] == distances.shape[1], 'square distances')
    upper = distances[np.triu_indices(len(distances), 1)]
    positive = np.sort(upper[upper > 0])
    require(len(positive) > 0, 'degenerate kernel features')
    middle = len(positive) // 2
    return float(positive[middle] if len(positive) % 2 else
                 (positive[middle - 1] + positive[middle]) / 2.)


def kernel_reference(train, queries, *, block_normalization=True, allow_constant=True):
    """Fit one input reference; each query is independently transformed."""
    z, mean, sd = standardized(train)
    queries = tuple(finite_matrix(q, 'query features') for q in queries)
    require(queries and all(q.shape[1] == z.shape[1] for q in queries), 'query feature widths')
    scale = np.sqrt(z.shape[1]) if block_normalization else 1.
    z = z / scale
    transformed = [(q - mean) / sd / scale for q in queries]
    constant = np.max(np.ptp(z, axis=0)) < 1e-12
    if constant:
        require(allow_constant, 'degenerate generator input')
        kernel = np.ones((len(z), len(z)))
        crosses = [np.ones((len(q), len(z))) for q in transformed]
        width = None
    else:
        distance = squared_distances(z)
        width = positive_median(distance)
        kernel = np.exp(-distance / (2. * width))
        crosses = [np.exp(-squared_distances(q, z) / (2. * width)) for q in transformed]
    return dict(kernel=kernel, crosses=crosses, width=width, mean=mean, sd=sd, x=z)


def projection_reference(value, feature_ids, seed, *, dimension=WIDTH):
    """Native-value fixed Gaussian map: no fitted all-patient statistic."""
    x = finite_matrix(value, 'projection source')
    ids = np.asarray(feature_ids, str)
    require(ids.shape == (x.shape[1],) and len(set(ids)) == len(ids)
            and all(v.strip() for v in ids), 'projection identity order')
    require(seed in (RNA_SEED, M_SEED) and dimension == WIDTH, 'frozen projection parameters')
    loading = np.random.Generator(np.random.PCG64(seed)).standard_normal((len(ids), WIDTH))
    loading /= np.sqrt(len(ids))
    return x @ loading, loading


def tissue_partition(labels, parts=3):
    labels = np.asarray(labels, str)
    require(labels.ndim == 1 and len(labels) and all(v.strip() for v in labels), 'tissue labels')
    counts = {label: int(np.count_nonzero(labels == label)) for label in set(labels.tolist())}
    require(parts >= 2 and len(counts) >= parts, 'insufficient tissue groups')
    totals = np.zeros(parts, dtype=int)
    assignment = {}
    for label in sorted(counts, key=lambda x: (-counts[x], x)):
        index = int(np.argmin(totals))
        assignment[label] = index
        totals[index] += counts[label]
    return np.array([assignment[label] for label in labels], dtype=int), assignment


def tissue_permutation(labels):
    labels = np.asarray(labels, str)
    require(labels.ndim == 1 and len(labels) and all(v.strip() for v in labels), 'shuffle tissues')
    rng = np.random.Generator(np.random.PCG64(SHUFFLE_SEED))
    permutation = np.arange(len(labels))
    singleton_rows = 0
    for label in sorted(set(labels.tolist())):
        rows = np.flatnonzero(labels == label)
        permutation[rows] = rng.permutation(rows)
        singleton_rows += int(len(rows) == 1)
    same(np.sort(permutation), np.arange(len(labels)), 'shuffle bijection')
    same(labels[permutation], labels, 'shuffle stays within tissue')
    return permutation, singleton_rows, int(np.count_nonzero(permutation == np.arange(len(labels))))


def masked_grid(kernel, crosses, labels, *, alphas=ALPHAS, minimum=20):
    """Independent spectral solves, grouped only by identical observed labels.

    Predictions retain generator axis.  Averaging is permitted only AFTER the
    nonlinear query kernels and each drug-specific head have been evaluated.
    """
    kernel = finite_matrix(kernel, 'head kernel')
    y = np.asarray(labels, dtype=np.float64)
    crosses = tuple(finite_matrix(cross, 'head cross kernel') for cross in crosses)
    require(y.ndim == 2 and kernel.shape == (len(y), len(y)) and not np.isinf(y).any(),
            'head label/kernel dimensions')
    require(crosses and all(c.shape[1] == len(y) and c.shape == crosses[0].shape for c in crosses),
            'head same-generator query dimensions')
    close(kernel, kernel.T, 'symmetric head kernel', rtol=1e-10, atol=1e-12)
    alphas = np.asarray(alphas, float)
    require(alphas.ndim == 1 and np.isfinite(alphas).all() and np.all(alphas > 0), 'positive head alpha')
    output = np.full((len(crosses), len(alphas), len(crosses[0]), y.shape[1]), np.nan)
    grouped = {}
    for column in range(y.shape[1]):
        mask = np.flatnonzero(np.isfinite(y[:, column]))
        if len(mask) >= minimum:
            grouped.setdefault(tuple(mask.tolist()), []).append(column)
    joined = np.concatenate(crosses, axis=0)
    for rows_tuple, columns in grouped.items():
        rows = np.array(rows_tuple, int)
        observed = y[np.ix_(rows, columns)]
        mean = np.add.reduce(observed, axis=0) / len(observed)
        gram = kernel[np.ix_(rows, rows)]
        values, vectors = eigh(gram, check_finite=False, driver='evd')
        require(values.min() > -1e-7, 'head kernel positive semidefinite')
        response = vectors.T @ (observed - mean)
        query_basis = joined[:, rows] @ vectors
        for index, alpha in enumerate(alphas):
            estimate = query_basis @ (response / (values + alpha)[:, None]) + mean
            for generator, chunk in enumerate(np.split(estimate, len(crosses))):
                output[generator, index][:, columns] = chunk
    return output


def task_grid(m_train, m_query, labels, extra_train=None, extra_queries=None):
    base = kernel_reference(m_train, (m_query,), allow_constant=False)
    if extra_train is None:
        require(extra_queries is None, 'M-only cannot accept target query values')
        result = masked_grid(base['kernel'], base['crosses'], labels)
    else:
        require(extra_queries is not None and len(extra_queries) in (1, 3), 'fixed query generator count')
        extra = kernel_reference(extra_train, extra_queries, allow_constant=False)
        mixture = .5 * base['kernel'] + .5 * extra['kernel']
        crosses = [.5 * base['crosses'][0] + .5 * cross for cross in extra['crosses']]
        result = masked_grid(mixture, crosses, labels)
    return result.mean(axis=0), result


def loss_reference(grid, truth, training_labels):
    grid, truth = np.asarray(grid, float), np.asarray(truth, float)
    require(grid.ndim == 3 and grid.shape[0] == len(ALPHAS) and grid.shape[1:] == truth.shape,
            'alpha prediction/truth shape')
    require(np.isfinite(grid).all(), 'finite alpha predictions before scoring')
    _, _, scale = moments(training_labels)
    require(np.isfinite(scale).all() and np.all(scale > 1e-8), 'nondegenerate inner training scales')
    observed = np.isfinite(truth)
    require(not np.isinf(truth).any(), 'inner truth infinity')
    count = observed.sum(axis=0)
    loss = np.zeros((len(ALPHAS), truth.shape[1]))
    for j in np.flatnonzero(count):
        loss[:, j] = np.square((grid[:, observed[:, j], j] - truth[observed[:, j], j]) / scale[j]).mean(axis=1)
    return loss, np.broadcast_to(count > 0, loss.shape).astype(int).copy(), scale


def selection_reference(losses, counts):
    losses, counts = np.asarray(losses, float), np.asarray(counts)
    require(losses.shape == counts.shape and losses.ndim == 2 and losses.shape[0] == len(ALPHAS)
            and np.isfinite(losses).all() and np.all(counts > 0), 'estimable matched alpha grid')
    return np.argmin(losses / counts, axis=0)


def selected_prediction(grid, selection):
    grid, selection = np.asarray(grid, float), np.asarray(selection)
    require(grid.ndim == 3 and selection.shape == (grid.shape[2],)
            and np.issubdtype(selection.dtype, np.integer) and np.all((selection >= 0) & (selection < len(ALPHAS))),
            'selected alpha indices')
    return grid[selection[None, :], np.arange(grid.shape[1])[:, None], np.arange(grid.shape[2])[None, :]]


def panel_reference(labels, fit, inner_fits):
    labels = np.asarray(labels, float)
    count, mean, sd = moments(labels[np.asarray(fit, int)])
    eligible = (count >= 80) & (sd > 1e-8)
    require(len(inner_fits) == 3, 'three original head inner folds')
    for inner in inner_fits:
        eligible &= np.isfinite(labels[np.asarray(inner, int)]).sum(axis=0) >= 20
    panel = np.flatnonzero(eligible)
    require(len(panel) > 0, 'nonempty original training drug panel')
    return panel, mean[panel], sd[panel]


def metric_reference(prediction, truth, train_sd):
    prediction, truth, train_sd = map(lambda a: np.asarray(a, float), (prediction, truth, train_sd))
    require(prediction.shape == truth.shape and prediction.ndim == 2
            and train_sd.shape == (truth.shape[1],) and np.all(train_sd > 1e-8)
            and np.isfinite(train_sd).all() and np.isfinite(prediction).all()
            and not np.isinf(truth).any(), 'matched metric inputs')
    observed = np.isfinite(truth)
    counts = observed.sum(axis=0)
    per_drug = np.full(truth.shape[1], np.nan)
    for j in np.flatnonzero(counts):
        per_drug[j] = np.square((prediction[observed[:, j], j] - truth[observed[:, j], j]) / train_sd[j]).mean()
    evaluated = counts >= 5
    require(evaluated.any(), 'some held drugs have five observations')
    score = float(np.sqrt(np.add.reduce(per_drug[evaluated]) / evaluated.sum()))
    return dict(score=score, per_drug=per_drug, counts=counts, evaluated=evaluated,
                observed_pairs=int(counts[evaluated].sum()))


def gate_reference(fold_scores):
    require(set(fold_scores) == set(ARMS), 'all eight gate arms')
    scores = {arm: np.asarray(fold_scores[arm], float) for arm in ARMS}
    require(all(x.shape == (5,) and np.isfinite(x).all() for x in scores.values()), 'five complete fold scores')
    means = {arm: float(value.sum() / 5) for arm, value in scores.items()}
    contrasts = {}
    for primary, controls in [('virtual_rna', ('m_only', 'ordinary_m', 'shuffle_rna')),
                              ('virtual_fm', ('m_only', 'virtual_rna', 'ordinary_m', 'shuffle_rna', 'shuffle_fm'))]:
        rows = {}
        for control in controls:
            difference = scores[control] - scores[primary]
            rows[control] = dict(control_minus_candidate=float(means[control] - means[primary]),
                                 fold_control_minus_candidate=difference.tolist(),
                                 favorable_folds=int(np.count_nonzero(difference > MARGIN)),
                                 passed=bool(means[control] - means[primary] > MARGIN))
        contrasts[primary] = rows
    return dict(mean_fold_scores=means, contrasts=contrasts,
                non_fm_utility_gate=all(x['passed'] for x in contrasts['virtual_rna'].values()),
                fm_contribution_gate=all(x['passed'] for x in contrasts['virtual_fm'].values()))


def json_check(actual, expected, context):
    """Validate an expected JSON tree, including list order and scalar types."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(expected) <= set(actual), context + ': fields')
        for key, value in expected.items():
            json_check(actual[key], value, context + '/' + key)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), context + ': length')
        for i, value in enumerate(expected):
            json_check(actual[i], value, context + '/' + str(i))
    elif isinstance(expected, bool):
        require(type(actual) is bool and actual == expected, context + ': bool')
    elif isinstance(expected, (float, np.floating)):
        require(not isinstance(actual, bool), context + ': number')
        close(actual, expected, context)
    else:
        require(actual == expected, context + ': value')


def compare_arrays(actual, expected, context, *, exact_keys=True):
    require((set(actual) == set(expected)) if exact_keys else (set(expected) <= set(actual)),
            context + ': array fields')
    for key, value in expected.items():
        value = np.asarray(value)
        if value.dtype.kind in 'biuUS':
            same(actual[key], value, context + '/' + key)
        else:
            close(actual[key], value, context + '/' + key)


class BoundFiles:
    """Only explicitly frozen logical members may be opened by this checker."""
    def __init__(self, roots, expected):
        self.roots = {key: Path(value).resolve(strict=True) for key, value in roots.items()}
        self.expected = dict(expected)
        self.used = {}

    def path(self, logical):
        require(isinstance(logical, str) and logical in self.expected and '\\' not in logical,
                'unbound source member: ' + str(logical))
        parts = logical.split('/')
        require(len(parts) >= 2 and parts[0] in self.roots
                and all(p and p not in ('.', '..') for p in parts), 'source path traversal')
        root = self.roots[parts[0]]
        value = root.joinpath(*parts[1:])
        require(value.is_file() and not value.is_symlink() and value.resolve().is_relative_to(root),
                'source path must remain regular and bound')
        actual = digest(value)
        expected = self.expected[logical]
        if isinstance(expected, dict):
            expected = expected['sha256']
        require(actual == expected, 'source SHA mismatch: ' + logical)
        self.used[logical] = actual
        return value

    def json(self, logical):
        return read_json(self.path(logical))

    def array(self, logical):
        return np.load(self.path(logical), allow_pickle=False)

    def fields(self, logical, names):
        with np.load(self.path(logical), allow_pickle=False) as source:
            require(set(names) <= set(source.files), 'bound source NPZ fields: ' + logical)
            return {key: source[key] for key in names}


def check_manifest_members(bound, key):
    """Check only the permitted input members, not unrelated assay packages."""
    manifest = bound.json(key + '/manifest.json')
    for logical in bound.expected:
        prefix = key + '/'
        if logical.startswith(prefix) and not logical.endswith('/manifest.json'):
            filename = logical[len(prefix):]
            require(filename in manifest['files'], 'member absent from prepared manifest: ' + logical)
            value = manifest['files'][filename]
            expected = bound.expected[logical]
            expected = expected['sha256'] if isinstance(expected, dict) else expected
            require(value['sha256'] == expected, 'prepared manifest differs from frozen member: ' + logical)
    return manifest


def generator_reference(m_train, target_blocks, tissues, m_queries):
    """Rebuild a complete fixed-alpha generator without any task labels."""
    require(set(target_blocks) == {'rna', 'fm', 'm'}, 'generator paired targets')
    require(all(np.asarray(value).shape == (len(m_train), WIDTH) for value in target_blocks.values()),
            'fixed generator target dimensions')
    targets = {key: finite_matrix(value, 'generator ' + key) for key, value in target_blocks.items()}
    permutation, singleton_rows, unchanged_rows = tissue_permutation(tissues)
    require(len(permutation) == len(m_train), 'generator training tissue order')
    state = kernel_reference(m_train, m_queries, block_normalization=False, allow_constant=False)
    # Targets are in native fixed coordinates: only an own-fit output mean is
    # removed.  No standard deviations, PCA bases or task AUC are transferred.
    mapped = dict(virtual_rna=targets['rna'], virtual_fm=targets['fm'], ordinary_m=targets['m'],
                  shuffle_rna=targets['rna'][permutation], shuffle_fm=targets['fm'][permutation])
    mean = {key: value.sum(0) / len(value) for key, value in mapped.items()}
    centered = np.concatenate([mapped[key] - mean[key] for key in VIRTUAL_ARMS], axis=1)
    eigenvalues, eigenvectors = eigh(state['kernel'], check_finite=False, driver='evd')
    coefficient = eigenvectors @ ((eigenvectors.T @ centered) / (eigenvalues + 1.)[:, None])
    coefficients = dict(zip(VIRTUAL_ARMS, np.split(coefficient, len(VIRTUAL_ARMS), axis=1)))
    predictions = [{key: cross @ coefficients[key] + mean[key] for key in VIRTUAL_ARMS}
                   for cross in state['crosses']]
    return dict(**state, permutation=permutation, singleton_rows=singleton_rows,
                unchanged_rows=unchanged_rows, target_means=mean, coefficients=coefficients,
                predictions=predictions)


def check_generator(model, m_train, target_blocks, tissues, m_queries):
    reference = generator_reference(m_train, target_blocks, tissues, m_queries)
    expected = dict(schema=np.array('creed.metabolite-query-generator.v1'),
        mean=reference['mean'], sd=reference['sd'], x=reference['x'], width=np.array(reference['width']),
        coef=np.concatenate([reference['coefficients'][a] for a in VIRTUAL_ARMS], axis=1),
        target_mean=np.concatenate([reference['target_means'][a] for a in VIRTUAL_ARMS]),
        arms=np.array(VIRTUAL_ARMS), cuts=np.arange(6) * WIDTH, permutation=reference['permutation'],
        tissues=np.asarray(tissues, str), alpha=np.array(1.), permutation_seed=np.array(SHUFFLE_SEED),
        fixed_points=np.array(reference['unchanged_rows']), singleton_rows=np.array(reference['singleton_rows']),
        feature_count=np.array(np.asarray(m_train).shape[1]), representation_width=np.array(WIDTH))
    compare_arrays(model, expected, 'fixed own-fit generator')
    return reference['predictions']


class ResultFiles:
    def __init__(self, root, expected):
        self.root = Path(root).resolve(strict=True)
        self.expected = dict(expected)
        self.used = {}

    def path(self, relative):
        require(relative in self.expected and '\\' not in relative, 'unrecorded result: ' + relative)
        parts = relative.split('/')
        require(all(part and part not in ('.', '..') for part in parts), 'result traversal')
        value = self.root.joinpath(*parts)
        require(value.is_file() and not value.is_symlink() and value.resolve().is_relative_to(self.root),
                'regular result file: ' + relative)
        actual = digest(value)
        expected = self.expected[relative]
        expected = expected['sha256'] if isinstance(expected, dict) else expected
        require(actual == expected, 'result SHA mismatch: ' + relative)
        self.used[relative] = actual
        return value

    def json(self, relative):
        return read_json(self.path(relative))

    def array(self, relative):
        return arrays(self.path(relative))

    def check_frozen(self, relative, names):
        value = self.json(relative)
        # Freeze documents use path-to-SHA maps; accept the explicitly named
        # wrapper when the writer also records a schema or stage marker.
        record = value.get('artifacts', value.get('hashes', value))
        require(isinstance(record, dict), 'frozen artifact mapping')
        for name in names:
            parent = str(Path(relative).parent).replace('\\', '/')
            full = name if name in self.expected else (parent + '/' + name)
            if full.startswith('./'):
                full = full[2:]
            key = full if full in record else name
            require(key in record, 'missing prediction freeze: ' + full)
            expected = record[key]
            expected = expected['sha256'] if isinstance(expected, dict) else expected
            require(digest(self.path(full)) == expected, 'prediction changed since freeze: ' + full)
        return value


def check_bank(result, prefix, source, fit, query, panel):
    """Every paired generator is reconstructed before any AUC head math."""
    fit, query, panel = (np.asarray(x, int) for x in (fit, query, panel))
    require(not set(fit) & set(query), 'bank train/query row disjointness')
    require(not set(source['tissues'][fit]) & set(source['tissues'][query]), 'bank whole-tissue exclusion')
    selection = result.json(prefix + '/selection.json')
    json_check(selection, dict(fit=fit.tolist(), query=query.tolist()), prefix + '/selection')
    folded, _ = tissue_partition(source['tissues'][fit], 3)
    expected_splits = [dict(fold=g, fit=np.flatnonzero(folded != g).tolist(),
                            held_local=np.flatnonzero(folded == g).tolist()) for g in range(3)]
    json_check(selection['generator_splits'], expected_splits, prefix + '/generator tissue splits')
    train = {a: np.full((len(fit), WIDTH), np.nan) for a in VIRTUAL_ARMS}
    queries = {a: np.full((3, len(query), WIDTH), np.nan) for a in VIRTUAL_ARMS}
    covered = np.zeros(len(fit), int)
    for generator, split in enumerate(expected_splits):
        a, b = np.array(split['fit'], int), np.array(split['held_local'], int)
        covered[b] += 1
        actual_fit, actual_held = fit[a], fit[b]
        expected = check_generator(result.array(prefix + f'/generator-{generator}.npz'),
            source['m'][actual_fit], dict(rna=source['rna_repr'][actual_fit],
                fm=source['fm'][actual_fit], m=source['m_repr'][actual_fit]),
            source['tissues'][actual_fit], [source['m'][actual_held], source['m'][query]])
        for arm in VIRTUAL_ARMS:
            train[arm][b] = expected[0][arm]
            queries[arm][generator] = expected[1][arm]
    same(covered, np.ones(len(fit), int), 'each head-training row exactly once OOF')
    compare_arrays(result.array(prefix + '/train-features.npz'), train, prefix + '/OOF')
    compare_arrays(result.array(prefix + '/query-features.npz'), queries, prefix + '/same-generator query')
    result.check_frozen(prefix + '/upstream-frozen.json',
        ['selection.json', 'generator-0.npz', 'generator-1.npz', 'generator-2.npz',
         'train-features.npz', 'query-features.npz'])
    y = source['auc'][np.ix_(fit, panel)]
    m_train, m_query = source['m'][fit], source['m'][query]
    grids = {}; members = {}
    grids['m_only'], _ = task_grid(m_train, m_query, y)
    for arm in VIRTUAL_ARMS:
        grids[arm], members[arm] = task_grid(m_train, m_query, y, train[arm], list(queries[arm]))
    # Actual query RNA is passed only into the two explicitly measured heads.
    # It is absent from every generator's input and from every virtual grid.
    for arm in MEASURED_ARMS:
        measured = source['rna_repr'][fit]
        if arm == 'measured_shuffle_rna':
            measured = measured[tissue_permutation(source['tissues'][fit])[0]]
        grids[arm], _ = task_grid(m_train, m_query, y, measured, [source['rna_repr'][query]])
    compare_arrays(result.array(prefix + '/per-alpha-predictions.npz'), grids, prefix + '/four-alpha head grid')
    compare_arrays(result.array(prefix + '/query-generator-predictions.npz'), members,
                   prefix + '/nonlinear generator members')
    compare_arrays(result.array(prefix + '/measured-permutation.npz'),
                   dict(permutation=tissue_permutation(source['tissues'][fit])[0]),
                   prefix + '/measured training-pair control')
    result.check_frozen(prefix + '/predictions-frozen.json',
                        ['per-alpha-predictions.npz', 'query-generator-predictions.npz', 'measured-permutation.npz'])
    return grids


def load_bound_source(bound):
    manifests = {}
    for key in ('data', 'rna', 'gdsc'):
        manifests[key] = check_manifest_members(bound, key)
    require(manifests['data']['scope'] == 'exploratory_training_421_only'
            and manifests['data']['sample_count'] == N_SAMPLES, 'prepared development scope')
    data_sha = bound.expected['data/manifest.json']
    require(manifests['rna']['existing_manifest_sha256'] == data_sha
            and manifests['gdsc']['existing_data_sha256'] == data_sha
            and manifests['gdsc']['source_sha256'] ==
                '9d6dc82b9a072bb00b6139dcd6c4c11de1e047f210382fb726e92ab8dd0eae2a',
            'prepared source lineage')
    for key in ('reference', 'measured-reference'):
        old = bound.json(key + '/result.json')
        require(old['status'] == 'completed' and old['scientific_claim'] is False,
                'completed checked development source reference')
    sample_ids = bound.array('data/sample_ids.npy').astype(str)
    require(sample_ids.shape == (N_SAMPLES,) and len(set(sample_ids)) == N_SAMPLES,
            'exact development sample identities')
    for name in ('rna', 'gdsc'):
        same(bound.array(name + '/sample_ids.npy'), sample_ids, name + ' sample order')
    m = finite_matrix(bound.array('data/metabolites.npy'), 'bound M')
    fm = finite_matrix(bound.array('data/scgpt.npy'), 'bound FM')
    raw_rna = finite_matrix(bound.array('rna/rna_log_cpm.npy'), 'bound native RNA')
    metabolite_ids = bound.array('data/metabolite_ids.npy').astype(str)
    gene_ids = bound.array('rna/gene_ids.npy').astype(str)
    drug_ids = bound.array('gdsc/drug_ids.npy').astype(str)
    auc = np.asarray(bound.array('gdsc/auc.npy'), float)
    require(m.shape == (N_SAMPLES, 225) and fm.shape == (N_SAMPLES, WIDTH)
            and raw_rna.shape[0] == N_SAMPLES and auc.shape == (N_SAMPLES, len(drug_ids))
            and not np.isinf(auc).any() and len(set(drug_ids)) == len(drug_ids), 'prepared source dimensions')
    rna_repr, _ = projection_reference(raw_rna, gene_ids, RNA_SEED)
    m_repr, _ = projection_reference(m, metabolite_ids, M_SEED)
    split = bound.json('reference/splits.json')
    json_check(bound.json('measured-reference/splits.json'), split, 'identical two inherited split files')
    same(split['sample_ids'], sample_ids, 'original source split identities')
    tissues = np.array(split['tissues'], str)
    fold_ids, _ = tissue_partition(tissues, 5)
    require(tissues.shape == (N_SAMPLES,), 'source tissue count')
    same(split['fold_ids'], fold_ids, 'original outer whole-tissue allocation')
    source = dict(m=m, rna_repr=rna_repr, fm=fm, m_repr=m_repr, auc=auc,
                  sample_ids=sample_ids, metabolite_ids=metabolite_ids, gene_ids=gene_ids,
                  drug_ids=drug_ids, tissues=tissues, fold_ids=fold_ids)
    return source, split, raw_rna.shape


def check_fold(result, bound, source, fold):
    prefix = 'fold-' + str(fold)
    fit = np.flatnonzero(source['fold_ids'] != fold)
    held = np.flatnonzero(source['fold_ids'] == fold)
    inner_ids, _ = tissue_partition(source['tissues'][fit], 3)
    head_splits = [dict(fit=fit[inner_ids != k].tolist(), held=fit[inner_ids == k].tolist()) for k in range(3)]
    panel, mean, sd = panel_reference(source['auc'], fit, [r['fit'] for r in head_splits])
    drug_ids = source['drug_ids'][panel]
    expected_selection = dict(fit=fit.tolist(), held=held.tolist(), panel=panel.tolist(),
                              drug_ids=drug_ids.tolist(), head_splits=head_splits)
    json_check(result.json(prefix + '/selection.json'), expected_selection, prefix + '/selection')
    for name in ('reference', 'measured-reference'):
        old = bound.json(name + f'/selection-{fold}.json')
        require(len(old['head_splits']) == 3, 'three inherited head splits')
        for index, split in enumerate(head_splits):
            json_check(old['head_splits'][index], split, 'original ' + name + '/head split')
    identity_keys = ('fit', 'held', 'panel', 'drug_ids', 'train_mean', 'train_sd')
    original = bound.fields('reference' + f'/predictions-{fold}.npz', identity_keys)
    old_m = bound.fields('measured-reference' + f'/predictions-{fold}.npz', (*identity_keys, 'metabolite_only'))
    expected_identity = dict(fit=fit, held=held, panel=panel, drug_ids=drug_ids, train_mean=mean, train_sd=sd)
    compare_arrays(original, expected_identity, 'original drug/reference identity', exact_keys=False)
    compare_arrays(old_m, expected_identity, 'M-only drug/reference identity', exact_keys=False)
    losses = {arm: np.zeros((len(ALPHAS), len(panel))) for arm in ARMS}
    counts = {arm: np.zeros((len(ALPHAS), len(panel)), int) for arm in ARMS}
    for index, split in enumerate(head_splits):
        a, b = np.array(split['fit'], int), np.array(split['held'], int)
        bank = check_bank(result, prefix + '/inner-' + str(index), source, a, b, panel)
        # This is the first label-scoring step for the recomputed bank.
        saved = {}
        for arm in ARMS:
            loss, count, _ = loss_reference(bank[arm], source['auc'][np.ix_(b, panel)],
                                           source['auc'][np.ix_(a, panel)])
            losses[arm] += loss; counts[arm] += count
            saved[arm + '_loss'] = loss; saved[arm + '_count'] = count
        compare_arrays(result.array(prefix + '/inner-' + str(index) + '/validation.npz'), saved,
                       prefix + '/inner-' + str(index) + '/validation')
    selected = {arm: selection_reference(losses[arm], counts[arm]) for arm in ARMS}
    saved_selection = {**selected, **{a + '_loss': losses[a] for a in ARMS},
                       **{a + '_count': counts[a] for a in ARMS}}
    compare_arrays(result.array(prefix + '/selected.npz'), saved_selection, prefix + '/selected')
    prior_selection = bound.json('measured-reference' + f'/selection-{fold}.json')
    close(losses['m_only'], prior_selection['losses']['metabolite_only'], 'old M-only four-alpha inner losses',
          rtol=1e-9, atol=1e-10)
    same(counts['m_only'], prior_selection['counts']['metabolite_only'], 'old M-only inner validation counts')
    same(selected['m_only'], prior_selection['selected']['metabolite_only'], 'old M-only chosen alpha')
    outer_bank = check_bank(result, prefix + '/outer', source, fit, held, panel)
    predictions = {arm: selected_prediction(outer_bank[arm], selected[arm]) for arm in ARMS}
    compare_arrays(result.array(prefix + '/predictions.npz'),
                   dict(fit=fit, held=held, panel=panel, drug_ids=drug_ids, **predictions),
                   prefix + '/selected outer predictions')
    close(predictions['m_only'], old_m['metabolite_only'], 'old M-only selected predictions', rtol=1e-9, atol=1e-10)
    return dict(fold=fold, fit=fit, held=held, panel=panel, drug_ids=drug_ids,
                train_mean=mean, train_sd=sd, predictions=predictions)


def evaluate_reference(completed, source):
    require(len(completed) == 5, 'all outer predictions complete before scores')
    fold_scores = {arm: [] for arm in ARMS}; metrics = []; per_drug = []
    for fold in completed:
        truth = source['auc'][np.ix_(fold['held'], fold['panel'])]
        for arm in ARMS:
            score = metric_reference(fold['predictions'][arm], truth, fold['train_sd'])
            fold_scores[arm].append(score['score'])
            metrics.append(dict(fold=fold['fold'], arm=arm, score=score['score'],
                drug_ids=int(score['evaluated'].sum()), observed_pairs=score['observed_pairs']))
            for index, drug in enumerate(fold['drug_ids']):
                value = score['per_drug'][index]
                per_drug.append(dict(fold=fold['fold'], arm=arm, drug_id=str(drug),
                    observed=int(score['counts'][index]), evaluated=bool(score['evaluated'][index]),
                    standardized_mse=float(value) if np.isfinite(value) else None))
    gates = gate_reference(fold_scores)
    return dict(**gates, fold_metrics=metrics, per_drug=per_drug)


def expected_result_files():
    output = {'frozenconfig.json', 'bindings.json', 'source-inputs.npz', 'splits.json',
              'projection.json', 'all-predictions-frozen.json', 'scores.json'}
    for fold in range(5):
        prefix = f'fold-{fold}'
        output.update(prefix + '/' + key for key in ('selection.json', 'selected.npz', 'predictions.npz'))
        for split in ('inner-0', 'inner-1', 'inner-2', 'outer'):
            bank = prefix + '/' + split
            output.update(bank + '/' + key for key in ('selection.json', 'generator-0.npz', 'generator-1.npz',
                'generator-2.npz', 'train-features.npz', 'query-features.npz', 'upstream-frozen.json',
                'per-alpha-predictions.npz', 'query-generator-predictions.npz', 'measured-permutation.npz',
                'predictions-frozen.json'))
            if split != 'outer':
                output.add(bank + '/validation.npz')
    return output


def expected_source_files():
    output = {f'data/{name}' for name in ('manifest.json', 'metabolites.npy', 'metabolite_ids.npy',
                                        'sample_ids.npy', 'scgpt.npy')}
    output.update(f'rna/{name}' for name in ('manifest.json', 'rna_log_cpm.npy', 'gene_ids.npy', 'sample_ids.npy'))
    output.update(f'gdsc/{name}' for name in ('manifest.json', 'auc.npy', 'drug_ids.npy', 'sample_ids.npy'))
    for key in ('reference', 'measured-reference'):
        output.update((key + '/splits.json', key + '/result.json'))
        for fold in range(5):
            output.update((key + f'/selection-{fold}.json', key + f'/predictions-{fold}.npz'))
    return output


def expected_config(code_root):
    root = Path(code_root).resolve(strict=True)
    paths = dict(runner='scripts/run_gdsc_metabolite_query.py', bridge='src/creed/science/metabolite_query.py',
                 task='src/creed/science/virtual_state_task.py', masked='src/creed/science/masked_task_kernel.py',
                 io='scripts/run_phospho_fewshot.py')
    return dict(schema=SCHEMA, scientific_claim=False, chainable=False, external_confirmation=False,
        samples=N_SAMPLES, arms=list(ARMS), primary_arm='virtual_fm', representation_width=WIDTH,
        rna_projection_seed=RNA_SEED, m_projection_seed=M_SEED, permutation_seed=SHUFFLE_SEED,
        alphas=ALPHAS.tolist(), generator_alpha=1., extra_kernel_weight=.5,
        query_generator_ensemble='mean of three task-head predictions, never mean input state or full-fit generator',
        comparison_margin=MARGIN, source_bindings_sha=BINDINGS_SHA,
        code_sha256={key: digest(root / value) for key, value in paths.items()},
        protocol_sha=digest(root / 'GDSC_METABOLITE_QUERY_NEXT.md'))


def producer_shaped_scores(evaluated):
    comparisons = []; gates = dict(non_fm={}, fm={})
    for group, primary in (('non_fm', 'virtual_rna'), ('fm', 'virtual_fm')):
        for control, row in evaluated['contrasts'][primary].items():
            means = evaluated['mean_fold_scores']
            comparisons.append(dict(group=group, candidate=primary, control=control,
                mean_control_minus_candidate=row['control_minus_candidate'],
                relative_improvement=1. - means[primary] / means[control],
                fold_effects=row['fold_control_minus_candidate'], folds_better=row['favorable_folds']))
            gates[group][control] = row['passed']
    scores = {key: evaluated[key] for key in ('mean_fold_scores', 'fold_metrics', 'per_drug')}
    scores['comparisons'] = comparisons
    return scores, gates


def check_loaded_field_contract(actual):
    require(set(actual) == expected_source_files(), 'record every and only permitted source member')
    identity = ['fit', 'held', 'panel', 'drug_ids', 'train_mean', 'train_sd']
    for key, value in actual.items():
        if key.endswith('.json'):
            require(value == 'json', 'source JSON read contract: ' + key)
        elif key.endswith('.npy'):
            require(value == 'npy', 'source NPY read contract: ' + key)
        else:
            expected = identity + (['metabolite_only'] if key.startswith('measured-reference/') else [])
            require(value == expected, 'source NPZ must not decode truth or extra assay values: ' + key)


def verify(*, result, data, rna, gdsc, reference, measured_reference, source_bindings, code_root, output=None):
    result, source_bindings = Path(result), Path(source_bindings)
    require(digest(source_bindings) == BINDINGS_SHA, 'exact pre-execution source binding SHA')
    binding = read_json(source_bindings)
    json_check(binding, dict(schema='creed.gdsc-metabolite-query.bound-sources.v1', scientific_claim=False),
               'frozen binding scope')
    require(set(binding['files']) == expected_source_files(), 'exact allowed source file list')
    bound = BoundFiles(dict(data=data, rna=rna, gdsc=gdsc, reference=reference,
                           **{'measured-reference': measured_reference}), binding['files'])
    for relative in bound.expected:
        bound.path(relative)
    summary_path = result / 'summary.json'
    summary_sha = digest(summary_path)
    summary = read_json(summary_path)
    require(summary['status'] == 'completed' and summary['scientific_claim'] is False,
            'completed non-scientific development producer')
    require(set(summary['artifacts']) == expected_result_files(), 'complete fixed artifact inventory')
    actual_files = {str(path.relative_to(result)).replace('\\', '/') for path in result.rglob('*') if path.is_file()}
    require(actual_files == expected_result_files() | {'summary.json'}, 'no missing or unexpected result files')
    files = ResultFiles(result, summary['artifacts'])
    config = files.json('frozenconfig.json')
    json_check(config, expected_config(code_root), 'frozen implementation/configuration')
    json_check(summary, config, 'summary frozen configuration')
    require(summary['frozenconfig_sha'] == files.used['frozenconfig.json'], 'summary configuration SHA')
    require(files.json('bindings.json') == binding, 'copied binding document')
    source, split, raw_rna_shape = load_bound_source(bound)
    require(files.json('splits.json') == split, 'exact inherited identity split artifact')
    compare_arrays(files.array('source-inputs.npz'), source, 'bound projection/identity/source arrays')
    json_check(files.json('projection.json'), dict(width=WIDTH, rna_seed=RNA_SEED, m_seed=M_SEED,
        rna_shape=list(raw_rna_shape), m_shape=list(source['m'].shape),
        policy='fixed Gaussian/sqrt(input width) on native coordinates; no data-fit projection moments'),
        'fixed native representation construction')
    completed = []
    for fold in range(5):
        completed.append(check_fold(files, bound, source, fold))
        print(json.dumps(dict(stage='independently_checked_fold', fold=fold,
                              generators=12, task_banks=4)), flush=True)
    freeze = files.json('all-predictions-frozen.json')
    require(set(freeze) == set(map(str, range(5))), 'all five predictions frozen before scores')
    for fold in range(5):
        require(freeze[str(fold)] == digest(files.path(f'fold-{fold}/predictions.npz')),
                'all-predictions freeze SHA')
    coverage = np.zeros(N_SAMPLES, int)
    for fold in completed:
        coverage[fold['held']] += 1
    same(coverage, np.ones(N_SAMPLES, int), 'each development cell line exactly once outer-held')
    evaluated = evaluate_reference(completed, source)
    scores, gates = producer_shaped_scores(evaluated)
    json_check(files.json('scores.json'), scores, 'independent all-drug/fold scores')
    json_check(summary, dict(**scores, gates=gates,
        non_fm_gate_passed=evaluated['non_fm_utility_gate'], fm_gate_passed=evaluated['fm_contribution_gate'],
        global_gate_passed=evaluated['fm_contribution_gate']), 'independent engineering gates')
    records = [dict(fold=fold['fold'], fit_patients=len(fold['fit']), held_patients=len(fold['held']),
        drugs=len(fold['panel']), m_only_reference_replayed=True,
        predictions_sha=digest(files.path(f"fold-{fold['fold']}/predictions.npz"))) for fold in completed]
    json_check(summary['fold_records'], records, 'per-fold completion and replay records')
    check_loaded_field_contract(summary['source_members_loaded'])
    # The prior mathematical check is inherited for source extraction, but this
    # run's M-only values, selected hyperparameters and all inner losses were
    # independently recomputed above, not merely copied from that reference.
    old_m = bound.json('measured-reference/result.json')
    close(scores['mean_fold_scores']['m_only'], old_m['summary']['metabolite_only'],
          'old M-only mean fold score', rtol=1e-9, atol=1e-10)
    require(set(files.used) == expected_result_files(), 'every saved artifact mathematically or structurally checked')
    require(digest(summary_path) == summary_sha and digest(source_bindings) == BINDINGS_SHA,
            'summary/source binding unchanged through independent check')
    receipt = dict(schema='creed.gdsc-metabolite-query.independent-check.v1', status='verified',
        scientific_claim=False, external_confirmation=False, chainable=False,
        producer_summary_sha256=summary_sha, source_bindings_sha256=BINDINGS_SHA,
        checker_sha256=digest(__file__), checked_source_members=bound.used,
        checked_artifacts=files.used, checked_generators=60, checked_task_banks=20,
        checked_task_alpha_grid_predictions=20 * len(ARMS) * len(ALPHAS),
        checked_fold_metrics=len(scores['fold_metrics']), checked_per_drug_metrics=len(scores['per_drug']),
        checked_m_only_reference_folds=5, patient_count=N_SAMPLES, **scores, gates=gates,
        non_fm_gate_passed=evaluated['non_fm_utility_gate'], fm_gate_passed=evaluated['fm_contribution_gate'],
        global_gate_passed=evaluated['fm_contribution_gate'],
        assumptions_and_limits=[
            'Numerical models and metrics independently rebuilt from exact bound prepared source arrays.',
            'Original raw-biological extraction, cached FM training provenance and culture-batch pairing are inherited, not re-established.',
            'All 421 identities are adaptively reused development; correlated drugs and folds are not independent confirmation.',
            'Actual query RNA is used only by explicit extra-measurement diagnostics; the virtual reconstructions below use query M alone.',
            'Artifact freeze hashes are checked retrospectively; runtime read ordering is supported separately by synthetic/canary tests.',
            'Neither an engineering pass nor a paired-shuffle contrast establishes full-transcriptome, causal, L0-L7, uncertainty or NMI success.'])
    if output is not None:
        with Path(output).open('x', encoding='utf-8') as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('result', 'data', 'rna', 'gdsc', 'reference', 'measured-reference', 'source-bindings', 'code-root', 'output'):
        parser.add_argument('--' + key, required=True, type=Path)
    receipt = verify(**vars(parser.parse_args()))
    print(json.dumps(dict(status=receipt['status'], global_gate_passed=receipt['global_gate_passed'],
                          checked_generators=receipt['checked_generators'])), flush=True)


if __name__ == '__main__':
    main()

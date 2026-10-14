"""Missing-aware native protein reference generators for M-only queries.

Panel selection is current head-A supervision. Every model then fits only its
own whole-tissue fitting rows. No function fits task-side scales or accepts
query protein values, query masks, or downstream response labels.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist

from creed.science.metabolite_query import group_folds, within_tissue_permutation


ARMS = ("virtual_p", "row_shuffle_p", "observed_value_shuffle_p", "ordinary_m", "virtual_mask")
SCHEMA = "creed.metabolite-protein-query-generator.v1"
PERMUTATION_SEED = 20260910
MIN_OBSERVED = 20
MIN_SD = 1e-8


def _matrix(values, name, *, rows=None, width=None, missing=False):
    raw = np.asarray(values)
    if (raw.ndim != 2 or raw.shape[1] == 0 or not np.issubdtype(raw.dtype, np.number)
            or np.iscomplexobj(raw) or (rows is not None and len(raw) != rows)
            or (width is not None and raw.shape[1] != width)):
        raise ValueError(f"{name}: invalid real numeric dimensions")
    value = raw.astype(np.float64)
    if np.isinf(value).any() or (not missing and not np.isfinite(value).all()):
        raise ValueError(f"{name}: invalid nonfinite values")
    return value


def _labels(values, rows):
    labels = np.asarray(values)
    if (labels.shape != (rows,) or rows == 0
            or any(not isinstance(x, str) or not x.strip() for x in labels.tolist())):
        raise ValueError("nonempty tissue labels must align with fitting rows")
    return labels.astype(str)


def _moments(values):
    observed = np.isfinite(values)
    counts = observed.sum(0)
    with np.errstate(over="ignore", invalid="ignore"):
        mean = np.divide(np.where(observed, values, 0).sum(0), counts,
                         out=np.zeros(values.shape[1]), where=counts > 0)
        squares = np.where(observed, values - mean, 0) ** 2
        sd = np.sqrt(np.divide(squares.sum(0), counts, out=np.zeros_like(mean), where=counts > 0))
    if not (np.isfinite(mean).all() and np.isfinite(sd).all()):
        raise ValueError("nonfinite observed-target moments")
    return counts, mean, sd


def panel_for_bank(p_fit, tissues_fit):
    """A-only panel with >=50% coverage and >=20/nonconstant in all 3 S_g.

    Return full-source diagnostics even if no column is eligible. build_bank
    treats that empty result as an explicit failure, never a relaxed panel.
    All split indices are local to A, in original row order.
    """
    protein = _matrix(p_fit, "A protein", missing=True)
    labels = _labels(tissues_fit, len(protein))
    folds, _ = group_folds(labels, nfolds=3)
    a_counts, _, a_sd = _moments(protein)
    splits, counts, sd = [], [], []
    for fold in range(3):
        fit, held = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        current_count, _, current_sd = _moments(protein[fit])
        counts.append(current_count); sd.append(current_sd)
        splits.append(dict(fold=fold, fit=fit, held_local=held))
    counts, sd = np.stack(counts), np.stack(sd)
    coverage_failed = 2 * a_counts < len(protein)
    count_failed = (counts < MIN_OBSERVED).any(0)
    sd_failed = (sd <= MIN_SD).any(0)
    eligible = ~(coverage_failed | count_failed | sd_failed)
    return dict(indices=np.flatnonzero(eligible), eligible=eligible, unsupported=~eligible,
                a_counts=a_counts, a_sd=a_sd, generator_counts=counts, generator_sd=sd,
                generator_splits=splits, fold_ids=folds, source_width=protein.shape[1],
                a_coverage_failed=coverage_failed, generator_count_failed=count_failed,
                generator_sd_failed=sd_failed)


def _value_permutation(observed, tissues):
    """Columns in source-J order; a separate single RNG, then sorted tissues."""
    permutation = np.full(observed.shape, -1, dtype=np.int64)
    fixed = np.zeros(observed.shape[1], dtype=np.int64)
    singletons = np.zeros_like(fixed)
    rng = np.random.default_rng(PERMUTATION_SEED)
    for column in range(observed.shape[1]):
        for tissue in sorted(set(tissues.tolist())):
            rows = np.flatnonzero(observed[:, column] & (tissues == tissue))
            if not len(rows):
                continue
            source = rng.permutation(rows)
            permutation[rows, column] = source
            fixed[column] += np.sum(source == rows)
            singletons[column] += len(rows) == 1
    return permutation, fixed, singletons


def _groups(observed):
    grouped = {}
    group_ids = np.empty(observed.shape[1], dtype=np.int64)
    for column in range(observed.shape[1]):
        key = observed[:, column].tobytes()
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(column)
    groups = list(grouped.values())
    for index, columns in enumerate(groups):
        group_ids[columns] = index
    return groups, group_ids


def fit_generator(m_fit, p_fit, ordinary_fit, tissues_fit):
    """Five native J-wide controls, shared M kernel and mask-grouped K+I.

    p_fit and ordinary_fit must already have the same ordered J columns and
    exactly the same observation mask. The source-ID binding belongs to the
    runner; standalone fit uses local J indices, build_bank retains full IDs'
    source indices. Zero coefficients outside observed rows are solver storage,
    not filled labels or observations.
    """
    m = _matrix(m_fit, "fit M")
    protein = _matrix(p_fit, "fit protein", rows=len(m), missing=True)
    ordinary = _matrix(ordinary_fit, "fit ordinary M", rows=len(m), width=protein.shape[1], missing=True)
    labels = _labels(tissues_fit, len(m))
    observed = np.isfinite(protein)
    if not np.array_equal(observed, np.isfinite(ordinary)):
        raise ValueError("ordinary targets must have exactly the protein observation mask")
    counts, _, sd_p = _moments(protein)
    if np.any(counts < MIN_OBSERVED) or np.any(sd_p <= MIN_SD):
        raise ValueError("generator protein targets violate fixed support/SD eligibility")
    with np.errstate(over="ignore", invalid="ignore"):
        mean, sd = m.mean(0), m.std(0)
        sd = np.where(sd < MIN_SD, 1.0, sd)
        x = (m - mean) / sd
    if not all(np.isfinite(v).all() for v in (mean, sd, x)):
        raise ValueError("nonfinite fitting M moments")
    distance = cdist(x, x, "sqeuclidean")
    positive = distance[np.triu_indices(len(x), 1)]
    positive = positive[positive > 0]
    if not len(positive) or not np.isfinite(distance).all():
        raise ValueError("degenerate or nonfinite M kernel geometry")
    width = float(np.median(positive))
    if not np.isfinite(width) or width <= 0:
        raise ValueError("invalid M RBF width")
    kernel = np.exp(-distance / (2 * width))
    row_permutation = within_tissue_permutation(labels)
    value_permutation, value_fixed, value_singletons = _value_permutation(observed, labels)
    shuffled_values = np.full(protein.shape, np.nan)
    for column in range(protein.shape[1]):
        rows = np.flatnonzero(observed[:, column])
        shuffled_values[rows, column] = protein[value_permutation[rows, column], column]
    targets = np.concatenate((protein, protein[row_permutation], shuffled_values, ordinary, observed.astype(float)), axis=1)
    target_observed = np.isfinite(targets)
    target_counts, target_mean, _ = _moments(targets)
    groups, group_ids = _groups(target_observed)
    coef = np.zeros(targets.shape)
    for columns in groups:
        rows = np.flatnonzero(target_observed[:, columns[0]])
        if len(rows) < MIN_OBSERVED:
            raise ValueError("insufficient observed reference targets")
        factor = cho_factor(kernel[np.ix_(rows, rows)] + np.eye(len(rows)), lower=True, check_finite=True)
        centered = targets[np.ix_(rows, columns)] - target_mean[columns]
        coef[np.ix_(rows, columns)] = cho_solve(factor, centered, check_finite=True)
    if not np.isfinite(coef).all():
        raise ValueError("nonfinite masked generator coefficients")
    tissue_counts = Counter(labels.tolist())
    j = protein.shape[1]
    return dict(schema=np.array(SCHEMA), mean=mean, sd=sd, x=x, width=np.array(width),
                coef=coef, target_mean=target_mean, target_observed=target_observed,
                observed_mask=observed, target_counts=target_counts, target_group_ids=group_ids,
                group_count=np.array(len(groups)), cuts=np.arange(6, dtype=np.int64) * j,
                arms=np.array(ARMS), row_permutation=row_permutation, value_permutation=value_permutation,
                value_fixed_points=value_fixed, value_singleton_rows=value_singletons,
                row_fixed_points=np.array(int(np.sum(row_permutation == np.arange(len(m))))),
                row_singleton_rows=np.array(sum(n for n in tissue_counts.values() if n == 1)),
                tissues=labels.copy(), permutation_seed=np.array(PERMUTATION_SEED), alpha=np.array(1.0),
                feature_count=np.array(m.shape[1]), protein_count=np.array(j),
                source_protein_indices=np.arange(j, dtype=np.int64), source_protein_count=np.array(j))


def _integer(model, key):
    value = np.asarray(model[key])
    if value.ndim != 0 or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{key} must be a saved integer scalar")
    return int(value)


def _validate_model(model):
    required = ("schema", "mean", "sd", "x", "width", "coef", "target_mean", "target_observed",
                "observed_mask", "target_counts", "target_group_ids", "group_count", "cuts", "arms",
                "row_permutation", "value_permutation", "value_fixed_points", "value_singleton_rows",
                "row_fixed_points", "row_singleton_rows", "tissues", "permutation_seed", "alpha",
                "feature_count", "protein_count", "source_protein_indices", "source_protein_count")
    if any(key not in model for key in required):
        raise ValueError("incomplete saved protein generator")
    if (np.asarray(model["schema"]).ndim != 0 or str(np.asarray(model["schema"]).item()) != SCHEMA
            or np.asarray(model["alpha"]).ndim != 0 or np.asarray(model["alpha"]).item() != 1.0
            or _integer(model, "permutation_seed") != PERMUTATION_SEED
            or not np.array_equal(model["arms"], ARMS)):
        raise ValueError("changed generator schema, arms, alpha or seed")
    p, j = _integer(model, "feature_count"), _integer(model, "protein_count")
    source_count = _integer(model, "source_protein_count")
    indices = np.asarray(model["source_protein_indices"])
    if (j <= 0 or source_count < j or indices.shape != (j,) or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= source_count) or np.any(np.diff(indices) <= 0)
            or not np.array_equal(model["cuts"], np.arange(6) * j)):
        raise ValueError("invalid protein coordinate order or cuts")
    x = _matrix(model["x"], "saved fitting M", width=p)
    labels = _labels(model["tissues"], len(x))
    observed = np.asarray(model["observed_mask"])
    all_observed = np.asarray(model["target_observed"])
    if observed.dtype != bool or observed.shape != (len(x), j) or all_observed.dtype != bool:
        raise ValueError("invalid saved observation masks")
    row_permutation = within_tissue_permutation(labels)
    value_permutation, value_fixed, value_singletons = _value_permutation(observed, labels)
    expected_mask = np.concatenate((observed, observed[row_permutation], observed, observed, np.ones_like(observed)), axis=1)
    groups, group_ids = _groups(expected_mask)
    tissue_counts = Counter(labels.tolist())
    if (not np.array_equal(model["row_permutation"], row_permutation)
            or not np.array_equal(model["value_permutation"], value_permutation)
            or not np.array_equal(model["value_fixed_points"], value_fixed)
            or not np.array_equal(model["value_singleton_rows"], value_singletons)
            or _integer(model, "row_fixed_points") != int(np.sum(row_permutation == np.arange(len(x))))
            or _integer(model, "row_singleton_rows") != sum(n for n in tissue_counts.values() if n == 1)
            or not np.array_equal(all_observed, expected_mask)
            or not np.array_equal(model["target_counts"], expected_mask.sum(0))
            or np.any(expected_mask.sum(0) < MIN_OBSERVED)
            or not np.array_equal(model["target_group_ids"], group_ids)
            or _integer(model, "group_count") != len(groups)):
        raise ValueError("changed saved mask, permutation, row order or grouping")
    coef = _matrix(model["coef"], "saved coefficients", rows=len(x), width=5 * j)
    mean, sd = np.asarray(model["mean"], float), np.asarray(model["sd"], float)
    target_mean = np.asarray(model["target_mean"], float)
    w = np.asarray(model["width"])
    if (mean.shape != (p,) or sd.shape != (p,) or not np.isfinite(mean).all()
            or not np.isfinite(sd).all() or np.any(sd <= 0) or w.ndim != 0
            or not np.isfinite(w).all() or w <= 0 or target_mean.shape != (5 * j,)
            or not np.isfinite(target_mean).all() or np.any(coef[~expected_mask] != 0)):
        raise ValueError("invalid saved moments, width or unobserved-row coefficients")
    return x, mean, sd, float(w), coef, target_mean, j


def predict_generator(model, m_query):
    """Return five unscaled native J-wide outputs using measured query M only."""
    x, mean, sd, width, coef, target_mean, j = _validate_model(model)
    query = _matrix(m_query, "query M", width=x.shape[1])
    with np.errstate(over="ignore", invalid="ignore"):
        q = (query - mean) / sd
    if not np.isfinite(q).all():
        raise ValueError("nonfinite standardized query M")
    distance = cdist(q, x, "sqeuclidean")
    if not np.isfinite(distance).all():
        raise ValueError("nonfinite query M distances")
    prediction = np.exp(-distance / (2 * width)) @ coef + target_mean
    if not np.isfinite(prediction).all():
        raise ValueError("nonfinite generated native states")
    return {arm: prediction[:, k * j:(k + 1) * j] for k, arm in enumerate(ARMS)}


def build_bank(m_fit, p_fit, tissues_fit, m_query, ordinary_projection):
    """A-local common panel, three tissue OOF models and their same query views."""
    m = _matrix(m_fit, "A M")
    protein = _matrix(p_fit, "A protein", rows=len(m), missing=True)
    labels = _labels(tissues_fit, len(m))
    query = _matrix(m_query, "query M", width=m.shape[1])
    projection = _matrix(ordinary_projection, "full ordinary projection", rows=m.shape[1], width=protein.shape[1])
    panel = panel_for_bank(protein, labels)
    indices = panel["indices"]
    if not len(indices):
        raise ValueError("empty common protein panel: candidate unestimable; no relaxation")
    ordinary_full = m @ projection
    if not np.isfinite(ordinary_full).all():
        raise ValueError("nonfinite ordinary M target projection")
    ordinary_full = np.where(np.isfinite(protein), ordinary_full, np.nan)
    p, ordinary = protein[:, indices], ordinary_full[:, indices]
    train = {arm: np.full((len(m), len(indices)), np.nan) for arm in ARMS}
    queries = {arm: [] for arm in ARMS}
    models = []
    for split in panel["generator_splits"]:
        fit, held = split["fit"], split["held_local"]
        model = fit_generator(m[fit], p[fit], ordinary[fit], labels[fit])
        model["source_protein_indices"] = indices.copy()
        model["source_protein_count"] = np.array(protein.shape[1])
        oof, q = predict_generator(model, m[held]), predict_generator(model, query)
        for arm in ARMS:
            train[arm][held] = oof[arm]
            queries[arm].append(q[arm])
        models.append(model)
    if not all(np.isfinite(v).all() for v in train.values()):
        raise ValueError("incomplete protein OOF bank")
    return dict(panel=panel, train_features=train,
                query_features={arm: np.stack(queries[arm]) for arm in ARMS},
                models=models, generator_splits=panel["generator_splits"])

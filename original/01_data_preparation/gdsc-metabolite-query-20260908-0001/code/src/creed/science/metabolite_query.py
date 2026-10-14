"""Fixed-coordinate, strictly training-only metabolite-query generators.

All five outputs are 512-dimensional representations, not recovered complete
biological layers. No function accepts query RNA, query FM values, or AUC.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist


ARMS = ("virtual_rna", "virtual_fm", "ordinary_m", "shuffle_rna", "shuffle_fm")
WIDTH = 512
PERMUTATION_SEED = 20260910
SCHEMA = "creed.metabolite-query-generator.v1"


def _matrix(values, label, *, rows=None, width=None):
    values = np.asarray(values, dtype=np.float64)
    if (values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all()
            or (rows is not None and len(values) != rows)
            or (width is not None and values.shape[1] != width)):
        raise ValueError(f"{label} has invalid dimensions or nonfinite values")
    return values


def _labels(labels):
    labels = np.asarray(labels)
    if (labels.ndim != 1 or not len(labels)
            or any(not isinstance(t, str) or not t.strip() for t in labels.tolist())):
        raise ValueError("invalid tissue labels")
    return labels.astype(str)


def project_fixed(values, feature_ids, seed):
    """Use N(0,1)/sqrt(P) in the supplied immutable feature-ID order.

    Projection row i belongs to feature_ids[i]. No observed mean, SD, PCA or
    gene selection enters construction. Callers must retain this ID order.
    """
    values = _matrix(values, "projection values")
    ids = np.asarray(feature_ids)
    if (ids.shape != (values.shape[1],)
            or any(not isinstance(s, str) or not s or s.strip() != s for s in ids.tolist())
            or len(set(ids.tolist())) != len(ids)):
        raise ValueError("projection requires exact unique feature IDs in column order")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("projection seed must be a nonnegative integer")
    projection = np.random.default_rng(int(seed)).standard_normal((len(ids), WIDTH)) / np.sqrt(len(ids))
    projected = values @ projection
    if not np.isfinite(projected).all():
        raise ValueError("nonfinite projected representation")
    return dict(values=projected, projection=projection, feature_ids=ids.astype(str), seed=int(seed))


def within_tissue_permutation(labels):
    """One fixed RNG; sorted tissue groups and current local row order."""
    labels = _labels(labels)
    rng = np.random.default_rng(PERMUTATION_SEED)
    permutation = np.arange(len(labels), dtype=np.int64)
    for tissue in sorted(set(labels.tolist())):
        rows = np.flatnonzero(labels == tissue)
        permutation[rows] = rng.permutation(rows)
    return permutation


def group_folds(labels, nfolds=3):
    """Exact existing tissue_folds grouping: size-descending greedy balancing."""
    labels = _labels(labels)
    if isinstance(nfolds, (bool, np.bool_)) or not isinstance(nfolds, (int, np.integer)):
        raise ValueError("nfolds must be an integer")
    counts = Counter(labels.tolist())
    if len(counts) < nfolds or nfolds < 2:
        raise ValueError("too few tissue groups")
    totals, mapping = [0] * nfolds, {}
    for tissue in sorted(counts, key=lambda t: (-counts[t], t)):
        fold = min(range(nfolds), key=lambda f: (totals[f], f))
        mapping[tissue] = fold
        totals[fold] += counts[tissue]
    return np.array([mapping[t] for t in labels], dtype=np.int64), mapping


def _fit_inputs(m_fit, rna_repr_fit, fm_fit, m_repr_fit, tissues_fit):
    m = _matrix(m_fit, "fit M")
    if len(m) < 2:
        raise ValueError("at least two generator fitting rows required")
    labels = _labels(tissues_fit)
    if len(labels) != len(m):
        raise ValueError("tissue identities must align with fitting rows")
    targets = tuple(_matrix(v, name, rows=len(m), width=WIDTH) for v, name in (
        (rna_repr_fit, "fit RNA representation"), (fm_fit, "fit FM representation"),
        (m_repr_fit, "fit ordinary M representation")))
    return m, targets, labels


def fit_generator(m_fit, rna_repr_fit, fm_fit, m_repr_fit, tissues_fit):
    """Fit shared RBF K+I with native centered targets and fit-only M moments."""
    m, (rna, fm, ordinary), labels = _fit_inputs(m_fit, rna_repr_fit, fm_fit, m_repr_fit, tissues_fit)
    mean, sd = m.mean(axis=0), m.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    if not (np.isfinite(mean).all() and np.isfinite(sd).all()):
        raise ValueError("nonfinite fit-only M moments")
    x = (m - mean) / sd
    distance = cdist(x, x, "sqeuclidean")
    upper = distance[np.triu_indices(len(x), 1)]
    positive = upper[upper > 0]
    if not len(positive):
        raise ValueError("degenerate generator M input: no positive pair distance")
    width = float(np.median(positive))
    if not np.isfinite(width) or width <= 0:
        raise ValueError("invalid generator kernel width")
    kernel = np.exp(-distance / (2 * width)) + np.eye(len(x))
    permutation = within_tissue_permutation(labels)
    target = np.concatenate((rna, fm, ordinary, rna[permutation], fm[permutation]), axis=1)
    target_mean = target.mean(axis=0)
    coef = solve(kernel, target - target_mean, assume_a="pos")
    if not (np.isfinite(target_mean).all() and np.isfinite(coef).all()):
        raise ValueError("nonfinite generator fit")
    counts = Counter(labels.tolist())
    return dict(schema=np.array(SCHEMA), mean=mean, sd=sd, x=x, width=np.array(width),
                coef=coef, target_mean=target_mean, arms=np.array(ARMS),
                cuts=np.arange(len(ARMS) + 1, dtype=np.int64) * WIDTH,
                permutation=permutation, tissues=labels.copy(), alpha=np.array(1.0),
                permutation_seed=np.array(PERMUTATION_SEED),
                fixed_points=np.array(int(np.sum(permutation == np.arange(len(labels))))),
                singleton_rows=np.array(sum(n for n in counts.values() if n == 1)),
                feature_count=np.array(m.shape[1]), representation_width=np.array(WIDTH))


def predict_generator(model, m_query):
    """Return the fixed five representations using measured query M alone."""
    required = ("schema", "mean", "sd", "x", "width", "coef", "target_mean", "arms", "cuts",
                "alpha", "feature_count", "representation_width")
    if any(key not in model for key in required):
        raise ValueError("incomplete saved generator state")
    if (str(np.asarray(model["schema"]).item()) != SCHEMA
            or float(np.asarray(model["alpha"]).item()) != 1.0
            or int(np.asarray(model["representation_width"]).item()) != WIDTH
            or not np.array_equal(model["arms"], np.array(ARMS))
            or not np.array_equal(model["cuts"], np.arange(len(ARMS) + 1) * WIDTH)):
        raise ValueError("saved generator coordinate schema or fixed alpha changed")
    p = int(np.asarray(model["feature_count"]).item())
    query = _matrix(m_query, "query M", width=p)
    x = _matrix(model["x"], "saved standardized fit M", width=p)
    mean, sd = np.asarray(model["mean"], float), np.asarray(model["sd"], float)
    coef = _matrix(model["coef"], "saved generator coefficients", rows=len(x), width=len(ARMS) * WIDTH)
    target_mean = np.asarray(model["target_mean"], float)
    width = float(np.asarray(model["width"]).item())
    if (mean.shape != (p,) or sd.shape != (p,) or not np.isfinite(mean).all()
            or not np.isfinite(sd).all() or np.any(sd <= 0)
            or target_mean.shape != (len(ARMS) * WIDTH,) or not np.isfinite(target_mean).all()
            or not np.isfinite(width) or width <= 0):
        raise ValueError("invalid saved generator moments or width")
    z = (query - mean) / sd
    kernel = np.exp(-cdist(z, x, "sqeuclidean") / (2 * width))
    prediction = kernel @ coef + target_mean
    if not np.isfinite(prediction).all():
        raise ValueError("nonfinite generated query representation")
    return {arm: prediction[:, i * WIDTH:(i + 1) * WIDTH] for i, arm in enumerate(ARMS)}


def build_bank(m_fit, rna_repr_fit, fm_fit, m_repr_fit, tissues_fit, m_query):
    """Three whole-tissue OOF generators and their SAME three query views.

    Both ``fit`` and ``held_local`` indices are relative to the supplied current
    head-training rows, in their original local order. No full-training refit is
    made, and query states are not averaged before a downstream nonlinear head.
    """
    m, (rna, fm, ordinary), labels = _fit_inputs(m_fit, rna_repr_fit, fm_fit, m_repr_fit, tissues_fit)
    query = _matrix(m_query, "query M", width=m.shape[1])
    folds, _ = group_folds(labels, nfolds=3)
    train_features = {arm: np.full((len(m), WIDTH), np.nan) for arm in ARMS}
    query_features = {arm: [] for arm in ARMS}
    models, splits = [], []
    for fold in range(3):
        fit, held = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        model = fit_generator(m[fit], rna[fit], fm[fit], ordinary[fit], labels[fit])
        oof, held_query = predict_generator(model, m[held]), predict_generator(model, query)
        for arm in ARMS:
            train_features[arm][held] = oof[arm]
            query_features[arm].append(held_query[arm])
        models.append(model)
        splits.append(dict(fold=fold, fit=fit, held_local=held))
    if any(not np.isfinite(values).all() for values in train_features.values()):
        raise ValueError("incomplete or nonfinite OOF bank")
    return dict(train_features=train_features,
                query_features={arm: np.stack(query_features[arm]) for arm in ARMS},
                models=models, generator_splits=splits)

"""Fixed rank-16, training-only metric from metabolite-task residuals.

The projection is supervised geometry on existing 512-coordinate representations,
not a new measurement or a refitted foundation model. Query labels are never an
input. Missing labels remain missing throughout the fit.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve


SCHEMA = "creed.task-residual-metric.v1"
FEATURES = 512
RANK_LIMIT = 16


def _features(values, label, *, fitting=False):
    raw = np.asarray(values)
    if (raw.ndim != 2 or raw.shape[1] != FEATURES
            or not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw)
            or (fitting and len(raw) == 0)):
        raise ValueError(f"{label} must have 512 real numeric feature coordinates")
    values = raw.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} must be finite")
    return values


def _targets(values, label, rows):
    raw = np.asarray(values)
    if (raw.ndim != 2 or len(raw) != rows or raw.shape[1] == 0
            or not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw)):
        raise ValueError(f"{label} must be an aligned real numeric target matrix")
    values = raw.astype(np.float64)
    if np.isinf(values).any():
        raise ValueError(f"{label} permits NaN missing values, not infinity")
    return values


def _basis(coef_eligible):
    """Canonical retained left subspace, dropping any tie block across rank 16."""
    matrix = np.asarray(coef_eligible, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != FEATURES or not np.isfinite(matrix).all():
        raise ValueError("eligible coefficient matrix must have 512 finite rows")
    projection = np.zeros((FEATURES, RANK_LIMIT))
    if matrix.shape[1] == 0:
        return dict(projection=projection, singular_values=np.empty(0), threshold=0.0,
                    numeric_rank=0, retained_rank=0, rank=0, tie_truncated=False,
                    rank_zero_fallback=True)
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    threshold = float(max(matrix.shape) * np.finfo(np.float64).eps * singular[0])
    numeric_rank = int(np.sum(singular > threshold))
    retained_rank = min(RANK_LIMIT, numeric_rank)
    tie_truncated = False
    if numeric_rank > RANK_LIMIT and singular[RANK_LIMIT - 1] - singular[RANK_LIMIT] <= threshold:
        retained_rank = RANK_LIMIT - 1
        while retained_rank > 0 and singular[retained_rank - 1] - singular[retained_rank] <= threshold:
            retained_rank -= 1
        tie_truncated = True
    for j in range(retained_rank):
        vector = u[:, j].copy()
        pivot = int(np.argmax(np.abs(vector)))
        if vector[pivot] < 0:
            vector *= -1
        projection[:, j] = vector
    return dict(projection=projection, singular_values=singular, threshold=threshold,
                numeric_rank=numeric_rank, retained_rank=retained_rank, rank=retained_rank,
                tie_truncated=tie_truncated, rank_zero_fallback=retained_rank == 0)


def fit_metric(features_A, y_A, oof_m_A):
    """Fit native standardized residual ridge directions using only head A.

    Feature means/SD use all A rows. Target SD uses all observed A labels, even
    when that row's OOF prediction is missing. Each eligible drug then centers
    features and residuals on its own valid rows and solves MSE + ||slope||^2,
    with a free intercept. Drugs have equal weight as columns of the final SVD.
    """
    features = _features(features_A, "features_A", fitting=True)
    y, oof = _targets(y_A, "y_A", len(features)), _targets(oof_m_A, "oof_m_A", len(features))
    if y.shape != oof.shape:
        raise ValueError("AUC and OOF target axes differ")
    with np.errstate(over="ignore", invalid="ignore"):
        mean, sd = features.mean(axis=0), features.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    if not (np.isfinite(mean).all() and np.isfinite(sd).all()):
        raise ValueError("nonfinite training feature moments")
    z = (features - mean) / sd
    observed = np.isfinite(y)
    valid = observed & np.isfinite(oof)
    observed_counts, valid_counts = observed.sum(axis=0), valid.sum(axis=0)
    target_sd = np.zeros(y.shape[1])
    for drug in range(y.shape[1]):
        if observed_counts[drug]:
            with np.errstate(over="ignore", invalid="ignore"):
                target_sd[drug] = y[observed[:, drug], drug].std()
    if not np.isfinite(target_sd).all():
        raise ValueError("nonfinite observed-AUC scale")
    scalable = target_sd > 1e-8
    eligible = (valid_counts >= 20) & scalable
    residual = np.full(y.shape, np.nan)
    for drug in np.flatnonzero(scalable):
        mask = valid[:, drug]
        with np.errstate(over="ignore", invalid="ignore"):
            residual[mask, drug] = (y[mask, drug] - oof[mask, drug]) / target_sd[drug]
        if not np.isfinite(residual[mask, drug]).all():
            raise ValueError("nonfinite standardized valid residual")
    coef, intercept = np.zeros((FEATURES, y.shape[1])), np.zeros(y.shape[1])
    for drug in np.flatnonzero(eligible):
        mask = valid[:, drug]
        x, target = z[mask], residual[mask, drug]
        x_mean, target_mean = x.mean(axis=0), float(target.mean())
        centered_x, centered_y = x - x_mean, target - target_mean
        # Exact constants must stay zero despite roundoff in their sample means.
        centered_x[:, np.ptp(x, axis=0) == 0] = 0
        if np.ptp(target) == 0:
            centered_y[:] = 0
        n = len(x)
        dual = solve(centered_x @ centered_x.T + n * np.eye(n), centered_y, assume_a="pos")
        coef[:, drug] = centered_x.T @ dual
        intercept[drug] = target_mean - x_mean @ coef[:, drug]
    if not (np.isfinite(coef).all() and np.isfinite(intercept).all()):
        raise ValueError("nonfinite residual ridge coefficients")
    state = _basis(coef[:, eligible])
    return dict(schema=np.array(SCHEMA), mean=mean, sd=sd, coef=coef, intercept=intercept,
                eligible=eligible, observed_counts=observed_counts, valid_counts=valid_counts,
                valid_mask=valid, target_sd=target_sd, residual=residual,
                rank_limit=np.array(RANK_LIMIT), **{"lambda": np.array(1.0)},
                **{key: np.asarray(value) for key, value in state.items()})


def transform_metric(model, features_Q):
    """Return Z @ U / sqrt(16), with no coordinate-wise post-standardization."""
    features = _features(features_Q, "features_Q")
    required = ("schema", "mean", "sd", "projection", "rank_limit", "lambda", "retained_rank", "rank")
    if any(key not in model for key in required):
        raise ValueError("incomplete metric model")
    if (str(np.asarray(model["schema"]).item()) != SCHEMA
            or float(np.asarray(model["lambda"]).item()) != 1.0
            or np.asarray(model["rank_limit"]).item() != RANK_LIMIT):
        raise ValueError("changed metric schema, lambda or rank limit")
    rank_value = np.asarray(model["retained_rank"])
    if (rank_value.ndim != 0 or not np.issubdtype(rank_value.dtype, np.integer)
            or not 0 <= int(rank_value) <= RANK_LIMIT or np.asarray(model["rank"]).item() != int(rank_value)):
        raise ValueError("invalid retained metric rank")
    rank = int(rank_value)
    mean, sd, projection = (np.asarray(model[k], float) for k in ("mean", "sd", "projection"))
    if (mean.shape != (FEATURES,) or sd.shape != (FEATURES,) or projection.shape != (FEATURES, RANK_LIMIT)
            or not np.isfinite(mean).all() or not np.isfinite(sd).all() or np.any(sd <= 0)
            or not np.isfinite(projection).all() or np.any(projection[:, rank:] != 0)
            or not np.allclose(projection[:, :rank].T @ projection[:, :rank], np.eye(rank), atol=1e-10, rtol=1e-10)):
        raise ValueError("invalid metric moments or retained orthonormal projection")
    with np.errstate(over="ignore", invalid="ignore"):
        projected = ((features - mean) / sd) @ projection / np.sqrt(RANK_LIMIT)
    if not np.isfinite(projected).all():
        raise ValueError("nonfinite query metric projection")
    return projected

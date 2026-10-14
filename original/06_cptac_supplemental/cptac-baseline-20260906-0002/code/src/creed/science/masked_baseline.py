"""Train-only low-rank multi-output ridge with exact missing-target masks."""
from collections import Counter
import re
import numpy as np


def unique_feature_map(ids):
    counts = Counter(str(x).strip() for x in ids)
    return {str(x).strip(): i for i, x in enumerate(ids)
            if str(x).strip().upper() not in {"", "NA", "NAN"} and counts[str(x).strip()] == 1}


def source_gene_labels(ids, cohort, modality):
    """Explicit source composite-ID projection; no gene aliases or isoform equivalence."""
    if modality == "rna":
        return [str(x).strip() for x in ids]
    labels = []
    for raw in ids:
        parts = str(raw).strip().split("|")
        valid = len(parts) == 2 and bool(re.fullmatch(r"[A-Za-z0-9_.-]+", parts[0]))
        if valid and cohort == "brca_cptac_2020":
            valid = bool(re.fullmatch(r"(?:NP|XP)_\d+(?:\.\d+)?", parts[1]))
        elif valid and cohort == "luad_cptac_2020":
            valid = parts[0] == parts[1]
        else:
            valid = False
        labels.append(parts[0] if valid else "")
    return labels


def target_panel(y, completeness=.8):
    count = np.isfinite(y).sum(axis=0)
    mean = np.nansum(y, axis=0) / np.maximum(count, 1)
    delta = np.where(np.isfinite(y), y - mean, 0.)
    sd = np.sqrt((delta * delta).sum(axis=0) / np.maximum(count, 1))
    valid = (count >= max(5, completeness * len(y))) & (sd > 1e-8)
    return np.flatnonzero(valid), mean[valid], sd[valid]


def fit_pca(x, max_features=1000, rank=16):
    indices, mean, sd = target_panel(x)
    if len(indices) == 0:
        raise ValueError("no eligible training predictors")
    order = np.argsort(-sd, kind="stable")[:max_features]
    indices, mean, sd = indices[order], mean[order], sd[order]
    selected = x[:, indices]
    fill = np.nanmedian(selected, axis=0)
    raw = np.where(np.isfinite(selected), selected, fill)
    center, scale = raw.mean(axis=0), raw.std(axis=0)
    scale = np.maximum(scale, 1e-8)
    # Missingness is a predictor, not a fictitious zero-valued measurement.
    indicators = ~np.isfinite(selected)
    design = np.concatenate(((raw - center) / scale, indicators.astype(float)), axis=1)
    design_center = design.mean(axis=0)
    _, _, vt = np.linalg.svd(design - design_center, full_matrices=False)
    basis = vt[:min(rank, len(x) - 2)]
    z = (design - design_center) @ basis.T
    zscale = np.maximum(z.std(axis=0), 1e-8)
    state = (indices, fill, center, scale, design_center, basis, zscale)
    return z / zscale, state


def transform_pca(x, state):
    indices, fill, center, scale, dc, basis, zs = state
    selected = x[:, indices]
    design = np.concatenate(((np.where(np.isfinite(selected), selected, fill) - center) / scale,
                             (~np.isfinite(selected)).astype(float)), axis=1)
    return (design - dc) @ basis.T / zs


def masked_ridge(z, y, alpha=.1):
    """Per-output observed-loss/n_observed plus ridge; intercept unpenalized."""
    design = np.column_stack((np.ones(len(z)), z))
    coeffs = []
    for start in range(0, y.shape[1], 256):
        block = y[:, start:start + 256]
        mask = np.isfinite(block).astype(float)
        counts = mask.sum(axis=0)
        if np.any(counts < 2):
            raise ValueError("target without sufficient observed training rows")
        gram = np.einsum("ni,nj,nk->kij", design, design, mask, optimize=True) / counts[:, None, None]
        penalty = np.eye(design.shape[1]) * alpha
        penalty[0, 0] = 0.
        rhs = design.T @ np.where(mask.astype(bool), block, 0.) / counts
        coeffs.append(np.linalg.solve(gram + penalty, rhs.T[..., None])[..., 0].T)
    return np.concatenate(coeffs, axis=1)


def predict_ridge(z, coefficients):
    return np.column_stack((np.ones(len(z)), z)) @ coefficients


def diagonal_predict(x_train, y_train, x_test, source_ids, target_ids, means, alpha=.1):
    mapping = {name: i for i, name in enumerate(source_ids)}
    result = np.broadcast_to(means, (len(x_test), len(means))).copy()
    covered = 0
    for j, name in enumerate(target_ids):
        if name not in mapping:
            continue
        i = mapping[name]
        ok = np.isfinite(x_train[:, i]) & np.isfinite(y_train[:, j])
        if ok.sum() < 5:
            continue
        xx, yy = x_train[ok, i], y_train[ok, j]
        xm, ym = xx.mean(), yy.mean()
        variance = np.mean((xx - xm) ** 2)
        if variance < 1e-8:
            continue
        slope = np.mean((xx - xm) * (yy - ym)) / (variance * (1 + alpha))
        test = np.where(np.isfinite(x_test[:, i]), x_test[:, i], xm)
        result[:, j] = ym + slope * (test - xm)
        covered += 1
    return result, covered


def metrics(truth, prediction):
    """Inputs are standardized with training-target statistics only."""
    rs = []
    for j in range(truth.shape[1]):
        ok = np.isfinite(truth[:, j]) & np.isfinite(prediction[:, j])
        a, b = truth[ok, j], prediction[ok, j]
        if len(a) < 5 or np.ptp(a) <= 1e-12 or np.ptp(b) <= 1e-12:
            continue
        a, b = a - a.mean(), b - b.mean()
        rs.append(float(a @ b / np.sqrt((a @ a) * (b @ b))))
    ok = np.isfinite(truth) & np.isfinite(prediction)
    return {"mean_target_r": float(np.mean(rs)) if rs else None,
            "median_target_r": float(np.median(rs)) if rs else None,
            "valid_r_targets": len(rs), "total_targets": truth.shape[1],
            "observed_scored_values": int(ok.sum()),
            "standardized_rmse": float(np.sqrt(np.mean((truth[ok] - prediction[ok]) ** 2))) if ok.any() else None}

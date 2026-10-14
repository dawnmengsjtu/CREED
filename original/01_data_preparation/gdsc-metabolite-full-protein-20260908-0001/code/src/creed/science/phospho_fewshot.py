"""Training-only semantic parameter transfer for few-shot phospho readouts.

ESM sequence and pathway annotations describe targets, not patient state. They
predict a prior on a common-coordinate RNA regression, never enter the patient
input by concatenation. Each novel target exposes only its five support rows.
This development estimator does not infer causal pathway activity or L3 truth.
"""
from __future__ import annotations

import hashlib
import numpy as np

ARMS = ('support_mean', 'direct', 'mean_prior', 'semantic', 'shuffle_esm', 'shuffle_reactome')


def stable_order(values, salt):
    values = [str(v) for v in values]
    if len(values) != len(set(values)):
        raise ValueError('stable identities must be unique')
    return sorted(values, key=lambda v: (hashlib.sha256((salt + '|' + v).encode()).hexdigest(), v))


def parent_is_heldout(symbol):
    return int(hashlib.sha256(('creed-phospho-parent-v1|' + str(symbol)).encode()).hexdigest(), 16) % 5 == 0


def column_reference(y):
    y = np.asarray(y, float)
    if y.ndim != 2 or np.isinf(y).any():
        raise ValueError('matrix must be finite or missing')
    mask = np.isfinite(y)
    count = mask.sum(0)
    mean = np.where(mask, y, 0).sum(0) / np.maximum(count, 1)
    sd = np.sqrt(np.where(mask, (y - mean) ** 2, 0).sum(0) / np.maximum(count, 1))
    return count, mean, sd


def fit_rna_reference(x, ids):
    x, ids = np.asarray(x, float), np.asarray(ids, str)
    if x.ndim != 2 or ids.shape != (x.shape[1],) or len(set(ids)) != len(ids):
        raise ValueError('RNA identity mismatch')
    count, mean, sd = column_reference(x)
    eligible = np.flatnonzero((count >= max(20, .8 * len(x))) & (sd > 1e-8))
    if len(eligible) < 16 or len(x) < 21:
        raise ValueError('insufficient training RNA for fixed PCA16')
    selected = eligible[np.argsort(-sd[eligible], kind='stable')[:1000]]
    safe_sd = np.where(sd > 1e-8, sd, 1.)
    z = np.where(np.isfinite(x), x - mean, 0) / safe_sd
    center = z[:, selected].mean(0)
    _, _, vt = np.linalg.svd(z[:, selected] - center, full_matrices=False)
    loading = vt[:16].T
    scores = (z[:, selected] - center) @ loading
    pc_sd = scores.std(0)
    pc_sd = np.where(pc_sd > 1e-8, pc_sd, 1.)
    return dict(rna_ids=ids, rna_mean=mean, rna_sd=safe_sd,
                rna_selected_indices=selected, rna_pca_center=center,
                rna_pca_loadings=loading, rna_pc_sd=pc_sd)


def transform_rna(x, model, ids):
    x = np.asarray(x, float)
    if not np.array_equal(np.asarray(ids, str), model['rna_ids']):
        raise ValueError('RNA order changed')
    if x.ndim != 2 or x.shape[1] != len(ids) or np.isinf(x).any():
        raise ValueError('invalid source RNA')
    missing = ~np.isfinite(x)
    z = np.where(missing, 0., x - model['rna_mean']) / model['rna_sd']
    pcs = ((z[:, model['rna_selected_indices']] - model['rna_pca_center'])
           @ model['rna_pca_loadings']) / model['rna_pc_sd']
    return dict(z=z, missing=missing, pcs=pcs)


def parent_design(source, rna_ids, parent):
    found = np.flatnonzero(np.asarray(rna_ids) == parent)
    n = len(source['z'])
    own = source['z'][:, found[0]] if len(found) else np.zeros(n)
    missing = source['missing'][:, found[0]] if len(found) else np.ones(n)
    return np.column_stack((np.ones(n), own, missing, source['pcs']))


def fit_ridge(design, y, prior=None, alpha=1.):
    """Mean squared loss + alpha * squared slope distance to the prior."""
    design, y = np.asarray(design, float), np.asarray(y, float)
    if design.ndim != 2 or y.shape != (len(design),) or not np.isfinite(design).all() or np.isinf(y).any():
        raise ValueError('invalid regression inputs')
    keep = np.isfinite(y)
    if keep.sum() < 3:
        raise ValueError('at least three labels required')
    p = design.shape[1] - 1
    prior = np.zeros(p) if prior is None else np.asarray(prior, float)
    if prior.shape != (p,) or not np.isfinite(prior).all() or alpha <= 0:
        raise ValueError('invalid slope prior or penalty')
    a, target = design[keep], y[keep]
    penalty = np.diag([0.] + [float(alpha)] * p)
    rhs = a.T @ target / len(a) + np.r_[0., alpha * prior]
    return np.linalg.solve(a.T @ a / len(a) + penalty, rhs)


def semantic_projection(base, novel, seed):
    """Base-only randomized PCA32 with train-score scaling; zero-padded rank."""
    base, novel = np.asarray(base, float), np.asarray(novel, float)
    if base.ndim != 2 or novel.ndim != 2 or base.shape[1] != novel.shape[1]:
        raise ValueError('semantic feature dimensions mismatch')
    if not np.isfinite(base).all() or not np.isfinite(novel).all():
        raise ValueError('semantic features must be finite')
    def unit(values):
        norm = np.linalg.norm(values, axis=1, keepdims=True)
        return np.divide(values, norm, out=np.zeros_like(values), where=norm > 0)
    b, q = unit(base), unit(novel)
    center = b.mean(0)
    centered = b - center
    rank = min(32, len(base)-1, base.shape[1])
    if rank < 1:
        raise ValueError('insufficient base targets for semantic prior')
    width = min(rank + 8, *centered.shape)
    omega = np.random.default_rng(seed).standard_normal((centered.shape[1], width))
    basis, _ = np.linalg.qr(centered @ omega, mode='reduced')
    for _ in range(2):
        basis, _ = np.linalg.qr(centered @ (centered.T @ basis), mode='reduced')
    _, _, vt = np.linalg.svd(basis.T @ centered, full_matrices=False)
    loading = np.zeros((base.shape[1], 32))
    loading[:, :rank] = vt[:rank].T
    score = centered @ loading
    sd = score.std(0)
    supported = sd > 1e-8
    loading[:, ~supported] = 0.
    sd = np.where(supported, sd, 1.)
    return (centered @ loading) / sd, ((q-center) @ loading) / sd, dict(center=center, loading=loading, sd=sd)


def row_derangement(size, seed):
    if size < 2:
        raise ValueError('two identities needed for a derangement')
    order = np.random.default_rng(seed).permutation(size)
    permutation = np.empty(size, dtype=int)
    permutation[order] = np.roll(order, 1)
    return permutation


def learn_semantic_prior(esm, pathway, base_slopes, nbase, seed):
    ebase, enovel, estate = semantic_projection(esm[:nbase], esm[nbase:], seed)
    pbase, pnovel, pstate = semantic_projection(pathway[:nbase], pathway[nbase:], seed+1)
    a = np.column_stack((np.ones(nbase), ebase, pbase))
    query = np.column_stack((np.ones(len(enovel)), enovel, pnovel))
    penalty = np.diag([0.] + [1.] * 64)
    coefficients = np.linalg.solve(a.T @ a / nbase + penalty, a.T @ base_slopes / nbase)
    state = {'coefficients': coefficients}
    for name, projection in (('esm', estate), ('pathway', pstate)):
        state.update({name + '_' + key: value for key, value in projection.items()})
    return query @ coefficients, state


def predict_saved(model, rna_query, rna_ids):
    source = transform_rna(rna_query, model, rna_ids)
    predictions = {}
    for arm in ARMS:
        relative = np.column_stack([parent_design(source, rna_ids, parent) @ model['coef_' + arm][i]
                                    for i, parent in enumerate(model['novel_parents'])])
        predictions[arm] = model['novel_mean'] + model['novel_sd'] * relative
    return predictions


def fit_predict_fewshot(*, rna_train, rna_query, rna_ids, base_y, base_parents,
                       support_y, novel_parents, support_indices, esm_base,
                       esm_novel, pathway_base, pathway_novel, seed=20260909):
    """Fit with full base labels and ONLY five-row novel support labels."""
    rna_train, rna_query = np.asarray(rna_train, float), np.asarray(rna_query, float)
    base_y, support_y = np.asarray(base_y, float), np.asarray(support_y, float)
    base_parents, novel_parents = np.asarray(base_parents, str), np.asarray(novel_parents, str)
    support_indices = np.asarray(support_indices, int)
    nbase, nnovel = len(base_parents), len(novel_parents)
    if nbase < 33 or nnovel < 1 or len(set(base_parents)) != nbase or len(set(novel_parents)) != nnovel:
        raise ValueError('require unique base/novel parent panels and at least 33 base targets')
    if set(base_parents) & set(novel_parents):
        raise ValueError('same parent appears in both supervision groups')
    if any(parent_is_heldout(g) for g in base_parents) or not all(parent_is_heldout(g) for g in novel_parents):
        raise ValueError('frozen parent holdout changed')
    if base_y.shape != (len(rna_train), nbase) or support_y.shape != (5, nnovel):
        raise ValueError('only five novel support rows may be supplied')
    if support_indices.shape != (5,) or len(set(support_indices)) != 5 or (support_indices < 0).any() or (support_indices >= len(rna_train)).any():
        raise ValueError('five distinct training support indices required')
    bc, bm, bs = column_reference(base_y)
    nc, nm, ns = column_reference(support_y)
    if (bc < 20).any() or (bs <= 1e-8).any() or (nc < 3).any() or (ns <= 1e-8).any():
        raise ValueError('unsupported training/support-only target panel')
    esm_base, esm_novel = np.asarray(esm_base, float), np.asarray(esm_novel, float)
    pathway_base, pathway_novel = np.asarray(pathway_base, float), np.asarray(pathway_novel, float)
    for values, rows in ((esm_base, nbase), (esm_novel, nnovel),
                         (pathway_base, nbase), (pathway_novel, nnovel)):
        if values.ndim != 2 or len(values) != rows:
            raise ValueError('base/novel semantic rows must match their own parent identities')
    esm = np.concatenate((esm_base, esm_novel)).astype(float)
    pathway = np.concatenate((pathway_base, pathway_novel)).astype(float)
    if len(esm) != nbase+nnovel or len(pathway) != nbase+nnovel or esm.shape[1] != 1280:
        raise ValueError('static parent feature order or ESM dimensions mismatch')
    model = fit_rna_reference(rna_train, rna_ids)
    train = transform_rna(rna_train, model, rna_ids)
    base_coef = np.stack([fit_ridge(parent_design(train, rna_ids, gene), (base_y[:, i]-bm[i])/bs[i])
                          for i, gene in enumerate(base_parents)])
    model.update(base_parents=base_parents, novel_parents=novel_parents,
                 base_mean=bm, base_sd=bs, base_count=bc, base_coef=base_coef,
                 novel_mean=nm, novel_sd=ns, novel_count=nc,
                 support_indices=support_indices, seed=np.array(seed))
    slope_count = base_coef.shape[1]-1
    priors = {'direct': np.zeros((nnovel, slope_count)),
              'mean_prior': np.broadcast_to(base_coef[:, 1:].mean(0), (nnovel, slope_count)).copy()}
    eperm, pperm = row_derangement(len(esm), seed+2), row_derangement(len(pathway), seed+3)
    model.update(esm_permutation=eperm, pathway_permutation=pperm)
    for arm, ee, pp in (('semantic', esm, pathway), ('shuffle_esm', esm[eperm], pathway),
                       ('shuffle_reactome', esm, pathway[pperm])):
        priors[arm], state = learn_semantic_prior(ee, pp, base_coef[:, 1:], nbase, seed+4)
        model.update({arm + '_' + k: v for k, v in state.items()})
    model['coef_support_mean'] = np.zeros((nnovel, slope_count+1))
    for arm, prior in priors.items():
        model['prior_' + arm] = prior
        model['coef_' + arm] = np.stack([fit_ridge(
            parent_design(train, rna_ids, gene)[support_indices],
            (support_y[:, i]-nm[i])/ns[i], prior[i]) for i, gene in enumerate(novel_parents)])
    predictions = predict_saved(model, rna_query, rna_ids)
    return dict(predictions=predictions, model=model, diagnostics=dict(
        base_parents=nbase, novel_parents=nnovel, support_patients=5,
        source_pca_rank=16, source_predictors=len(model['rna_selected_indices']),
        semantic_dimensions_per_source=32, ridge_penalty=1., semantic_penalty=1.,
        shuffled_esm_fixed_points=int((eperm == np.arange(len(eperm))).sum()),
        shuffled_pathway_fixed_points=int((pperm == np.arange(len(pperm))).sum()),
        target_reference='base: own training labels; novel: only shared five support patients',
        coordinate_order=['intercept', 'same_gene_RNA', 'same_gene_RNA_missing', 'training_RNA_PC1_to_PC16'],
        inference_inputs='query RNA and exact parent IDs only; static semantics encoded in training coefficients',
        semantic_shuffle='within-cohort base+novel parent row derangement, not a degree-preserving pathway rewiring',
        scientific_claim=False))

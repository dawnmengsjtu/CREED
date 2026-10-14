"""Model core for the sealed internal-final one-shot (CREED_INTERNAL_FINAL_ONESHOT_V1, PLAN.md §3–§5).

Every function here is a verbatim or line-equivalent copy of the reference implementations that produced the
development-fold results (crosslayer-matrix-20261009-0001/code/crosslayer.py, protein-panel-20261009-0001/code/protein_panel.py,
rna-panel-20261009-0001/code/rna_panel.py, crispr-endpoint-20261009-0001/code/crispr_endpoint.py,
extra-supervision-20261008-0001/code/run_recoverability.py). Nothing in this file reads any file: it only receives arrays.

Conventions (all frozen before any sealed value is read):
  * ALPHAS / LAMBDAS / MUS / WS grids, kernels, generator, panels: identical to the references.
  * Inner banks for hyper-parameter selection: cell_folds(tissues_of_fit, seed, nfolds=3) — the 'nested_cell_folds' rule of
    run_recoverability.py (verified on 2026-10-09 to reproduce every extra-supervision new_cell inner bank with seed 20261008);
    the one-shot replays it on the full 421 with seed 20261009.
  * L_protein: 10-nearest-neighbour (union-symmetrised) graph on the fit-only protein value kernel (>=50 % coverage proteins,
    fit-mean imputed, fit z-scored, RBF median width), W = K on the kNN support, L = D - W, trace normalised to n_fit
    (verified on 2026-10-09 to reproduce the saved L_protein of the extra-supervision banks to <= 1e-15).
"""
import hashlib
import json

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, spearmanr

ALPHAS = (.1, 1., 10., 100.)
LAMBDAS = (.1, 1., 10., 100.)
MUS = (0., .1, 1., 10., 100., 1000.)
WS = (0., .25, 1.)
TOP_GENES, TOP_TARGET, P_COVER, OOF_FOLDS = 3000, 512, .5, 3
PANEL_COVER, MIN_PAIR, KNN = .9, 10, 10
KS = (16, 32, 64)
RKS = (16, 32)
SEEDS = (0, 1, 2, 3, 4)
MIN_FIT = 20
PAIRS = (('M', 'RNA'), ('M', 'P'))
SEED = 20261009


# ---------------------------------------------------------------- utilities
def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_array(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'little')


def cell_folds(tissues, seed, nfolds=5, outer=False):
    """Verbatim: extra-supervision-20261008-0001/code/run_recoverability.py::cell_folds."""
    labels = np.asarray(tissues).astype(str)
    folds = np.full(len(labels), -1, int); loads = np.zeros(nfolds, int)
    for label in sorted(np.unique(labels), key=lambda t: (-np.sum(labels == t), t)):
        idx = np.flatnonzero(labels == label)
        rng = np.random.default_rng(stable_seed(seed, 'new_cell_folds' if outer else 'nested_cell_folds', label))
        rng.shuffle(idx)
        order = rng.permutation(nfolds)
        order = order[np.argsort(loads[order], kind='stable')]
        for j, row in enumerate(idx):
            f = int(order[j % nfolds]); folds[row] = f; loads[f] += 1
    assert np.all(folds >= 0) and np.all(loads > 0)
    return folds


def nanstd0(a):
    obs = np.isfinite(a); cnt = obs.sum(0)
    mean = np.divide(np.where(obs, a, 0.).sum(0), cnt, out=np.zeros(a.shape[1]), where=cnt > 0)
    var = np.divide(np.where(obs, (a - mean) ** 2, 0.).sum(0), cnt, out=np.zeros(a.shape[1]), where=cnt > 0)
    return np.sqrt(var)


def nanmean0(a):
    obs = np.isfinite(a); cnt = obs.sum(0)
    return np.divide(np.where(obs, a, 0.).sum(0), cnt, out=np.zeros(a.shape[1]), where=cnt > 0)


# ---------------------------------------------------------------- model core (reference-equivalent)
def predict_grid(k, h, y):
    """Per-target mean-centred masked KRR; < MIN_FIT labels -> fit mean for every alpha; zero labels -> 0 (count 0)."""
    pred = np.empty((len(ALPHAS), len(h), y.shape[1])); groups = {}
    for j in range(y.shape[1]):
        groups.setdefault(np.isfinite(y[:, j]).tobytes(), []).append(j)
    for cols in groups.values():
        obs = np.flatnonzero(np.isfinite(y[:, cols[0]]))
        if not len(obs):
            pred[:, :, cols] = 0.
            continue
        yy = y[np.ix_(obs, cols)]; mean = yy.mean(0)
        pred[:, :, cols] = mean
        if len(obs) < MIN_FIT:
            continue
        kk = k[np.ix_(obs, obs)]; hh = h[:, obs]
        for ai, alpha in enumerate(ALPHAS):
            coef = cho_solve(cho_factor(kk + alpha * np.eye(len(obs)), lower=True), yy - mean)
            pred[ai][:, cols] = hh @ coef + mean
    assert np.isfinite(pred).all()
    return pred


def bank_losses(grid, truth, sd):
    e = ((grid - truth[None]) / sd) ** 2
    n = np.isfinite(e).sum(1)
    return np.divide(np.nansum(e, axis=1), n, out=np.zeros_like(n, float), where=n > 0), (n > 0).astype(int)


def choose(total, count, fallback=1):
    avg = np.divide(total, count, out=np.full_like(total, np.inf), where=count > 0)
    idx = np.argmin(avg, axis=0); idx[np.all(count == 0, axis=0)] = fallback
    return idx


def take(grid, sel):
    return grid[sel, :, np.arange(grid.shape[2])].T


def grouped_spearman(truth, pred):
    """Per-column Spearman over each column's observed rows (== scipy.stats.spearmanr; NaN where a side is constant)."""
    rho = np.full(truth.shape[1], np.nan); groups = {}
    for j in range(truth.shape[1]):
        groups.setdefault(np.isfinite(truth[:, j]).tobytes(), []).append(j)
    for cols in groups.values():
        o = np.flatnonzero(np.isfinite(truth[:, cols[0]]))
        if len(o) < 2:
            continue
        T = truth[np.ix_(o, cols)]; P = pred[np.ix_(o, cols)]
        ok = (T.std(0) > 0) & (P.std(0) > 1e-15)
        if not ok.any():
            continue
        rt = rankdata(T[:, ok], axis=0); rp = rankdata(P[:, ok], axis=0)
        dt = rt - rt.mean(0); dp = rp - rp.mean(0)
        den = np.sqrt((dt * dt).sum(0) * (dp * dp).sum(0))
        r = np.divide((dt * dp).sum(0), den, out=np.full(int(ok.sum()), np.nan), where=den > 0)
        rho[np.array(cols)[ok]] = r
    return rho


def metrics(truth, pred, sd, use):
    """sRMSE over `use` targets; same-target Spearman; centered MSE; per-target arrays (reference formulae)."""
    obs = np.isfinite(truth); n = obs.sum(0)
    e = np.where(obs, ((pred - truth) / sd) ** 2, 0.)
    mse = e.sum(0) / np.maximum(n, 1)
    bias = (np.where(obs, pred, 0).sum(0) / np.maximum(n, 1) - nanmean0(truth)) / sd
    rho = grouped_spearman(truth, pred)
    return dict(score=float(np.sqrt(mse[use].mean())) if use.any() else float('nan'),
                spearman=float(np.nanmean(rho[use])) if use.any() and np.isfinite(rho[use]).any() else float('nan'),
                centered_mse=float((mse - bias * bias)[use].mean()) if use.any() else float('nan'),
                mse=mse, rho=rho, n=n, e=e, obs=obs)


def spearman_check(truth, pred, use):
    """Slow scipy reference for grouped_spearman (dry-run assertion only)."""
    rho = np.full(truth.shape[1], np.nan); obs = np.isfinite(truth)
    for j in np.flatnonzero(use):
        v = obs[:, j]
        if v.sum() >= 2 and np.std(truth[v, j]) > 0 and np.std(pred[v, j]) > 1e-15:
            rho[j] = spearmanr(truth[v, j], pred[v, j]).statistic
    return rho


# ---------------------------------------------------------------- kernels
def kernel_state(x):
    mean = x.mean(0); sd = x.std(0); sd = np.where(sd < 1e-8, 1., sd); z = (x - mean) / sd
    d = cdist(z, z, 'sqeuclidean'); v = d[np.triu_indices(len(z), 1)]; v = v[v > 0]
    if not len(v):
        raise ValueError('Degenerate feature bank')
    return mean, sd, float(np.median(v))


def rbf(a, b, state):
    mean, sd, width = state
    return np.exp(-cdist((a - mean) / sd, (b - mean) / sd, 'sqeuclidean') / (2 * width))


def binary_width(bf):
    d = cdist(bf, bf, 'sqeuclidean'); v = d[np.triu_indices(len(bf), 1)]; v = v[v > 0]
    return float(np.median(v)) if len(v) else float('nan')


def binary_rbf(bf, bq, width):
    if not np.isfinite(width):
        return np.ones((len(bf), len(bf))), np.ones((len(bq), len(bf)))
    return np.exp(-cdist(bf, bf, 'sqeuclidean') / (2 * width)), np.exp(-cdist(bq, bf, 'sqeuclidean') / (2 * width))


def same(a, b):
    return (a[:, None] == b[None, :]).astype(float)


def tokens(t, tag):
    return np.array([x if x else f'__unknown_{tag}_{i}' for i, x in enumerate(t)])


def build_laplacian(K, k=KNN):
    """kNN (union) graph on kernel K, W = K on the support, combinatorial Laplacian, trace normalised to n."""
    n = len(K); Kc = K.copy(); np.fill_diagonal(Kc, -np.inf)
    nn = np.argsort(-Kc, 1, kind='stable')[:, :k]
    A = np.zeros((n, n), bool); A[np.arange(n)[:, None], nn] = True; A = A | A.T
    W = np.where(A, K, 0.); np.fill_diagonal(W, 0.)
    L = np.diag(W.sum(1)) - W
    return L * n / np.trace(L)


def graph_pairs(K, H, L):
    """Reference P-graph formula: root=K^{1/2}; A=I+mu root L root; K_eff=root A^-1 root; H_eff=H-mu H L K_eff (mu in MUS[1:])."""
    vals, vecs = eigh((K + K.T) / 2)
    assert vals.min() > -1e-8
    root = (vecs * np.sqrt(np.maximum(vals, 0))) @ vecs.T
    out = []
    for mu in MUS[1:]:
        a = np.eye(len(K)) + mu * root @ L @ root; a = (a + a.T) / 2
        ke = root @ cho_solve(cho_factor(a, lower=True), root); ke = (ke + ke.T) / 2
        out.append((ke, H - mu * H @ L @ ke))
    return out


# ---------------------------------------------------------------- panel relevance (protein-panel / rna-panel rule)
def spearman_scores(P, Y, min_n=MIN_PAIR):
    """Verbatim protein_panel.py: sum_j |Spearman(P[:, i], Y[:, j])| on pairwise-complete rows; n<min_n or constant -> 0."""
    c = P.shape[1]; tot = np.zeros(c); pobs = np.isfinite(P)
    for j in range(Y.shape[1]):
        o = np.isfinite(Y[:, j])
        if o.sum() < min_n:
            continue
        X = P[o]; y = Y[o, j]; po = pobs[o]; cnt = po.sum(0)
        rx = rankdata(X, axis=0, nan_policy='omit')
        ry = rankdata(np.where(po, y[:, None], np.nan), axis=0, nan_policy='omit')
        with np.errstate(invalid='ignore'):
            mx = np.nanmean(rx, 0); my = np.nanmean(ry, 0)
        dx = np.where(po, rx - mx, 0.); dy = np.where(po, ry - my, 0.)
        num = (dx * dy).sum(0); den = np.sqrt((dx * dx).sum(0) * (dy * dy).sum(0))
        ok = (cnt >= min_n) & (den > 1e-12)
        r = np.zeros(c); r[ok] = num[ok] / den[ok]
        tot += np.abs(r)
    return tot


def _dense_block(Xsub, Ysub, min_n):
    rx = rankdata(Xsub, axis=0); dx = rx - rx.mean(0); dxx = (dx * dx).sum(0)
    ry = rankdata(Ysub, axis=0); dy = ry - ry.mean(0); dyy = (dy * dy).sum(0)
    num = dx.T @ dy; den = np.sqrt(np.outer(dxx, dyy))
    r = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    return np.abs(r).sum(1) if len(Xsub) >= min_n else np.zeros(Xsub.shape[1])


def panel_weights(Xf, Yf, min_n=MIN_PAIR):
    """Verbatim crispr_endpoint.py: exact sum_j |Spearman| grouped by target missingness pattern (== spearman_scores)."""
    n, c = Xf.shape; pobs_all = np.isfinite(Xf); tot = np.zeros(c); groups = {}
    for g in range(Yf.shape[1]):
        groups.setdefault(np.isfinite(Yf[:, g]).tobytes(), []).append(g)
    for cols in groups.values():
        o = np.flatnonzero(np.isfinite(Yf[:, cols[0]]))
        if len(o) < min_n:
            continue
        po = pobs_all[o]; Y = Yf[np.ix_(o, cols)]
        full = np.flatnonzero(po.all(0)); part = np.flatnonzero(~po.all(0))
        if len(full):
            tot[full] += _dense_block(Xf[np.ix_(o, full)], Y, min_n)
        if not len(part):
            continue
        Xp = Xf[np.ix_(o, part)]; pop = po[:, part]; cnt = pop.sum(0)
        rx = rankdata(Xp, axis=0, nan_policy='omit')
        with np.errstate(invalid='ignore'):
            mx = np.nanmean(rx, 0)
        dx = np.where(pop, rx - mx, 0.); dxx = (dx * dx).sum(0); zero = np.zeros(len(part))
        for t in range(Y.shape[1]):
            y = Y[:, t]; order = np.argsort(y, kind='stable'); Ps = pop[order]
            C = np.cumsum(Ps, axis=0).astype(float); ys = y[order]
            newblk = np.empty(len(ys), bool); newblk[0] = True
            np.not_equal(ys[1:], ys[:-1], out=newblk[1:])
            if newblk.all():
                ry = C
            else:
                starts = np.flatnonzero(newblk); ends = np.append(starts[1:], len(ys))
                Cb = np.vstack([np.zeros((1, len(part))), C]); avg = (Cb[starts] + Cb[ends] + 1.) / 2.
                ry = avg[np.cumsum(newblk) - 1]
            s = np.where(Ps, ry, 0.).sum(0); my = np.divide(s, cnt, out=zero.copy(), where=cnt > 0)
            dy = np.where(Ps, ry - my, 0.)
            num = (dx[order] * dy).sum(0); den = np.sqrt(dxx * (dy * dy).sum(0))
            ok = (cnt >= min_n) & (den > 1e-12)
            r = np.zeros(len(part)); r[ok] = num[ok] / den[ok]
            tot[part] += np.abs(r)
    return tot


def p_candidates(p_fit):
    obs = np.isfinite(p_fit); cnt = obs.sum(0); frac = cnt / len(p_fit)
    mean = np.divide(np.where(obs, p_fit, 0).sum(0), cnt, out=np.zeros(p_fit.shape[1]), where=cnt > 0)
    var = np.divide(np.where(obs, (p_fit - mean) ** 2, 0).sum(0), cnt, out=np.zeros(p_fit.shape[1]), where=cnt > 0)
    return np.flatnonzero((frac >= PANEL_COVER) & (var > 1e-12))


def random_panels(cand, seed_list):
    """protein-panel / rna-panel rule: one permutation of the (index-ascending) candidate pool per seed; first k (16 ⊂ 32)."""
    out = {}
    for s in SEEDS:
        perm = np.random.default_rng(list(seed_list) + [s]).permutation(cand)
        for k in RKS:
            out[f'rand{k}_s{s}'] = np.sort(perm[:k])
    return out


# ---------------------------------------------------------------- fit side (needs only fit arrays) and query side
def fit_layers(m, X, p, tis, gen_seed, shuf_seed):
    """Fit-only states of M, RNA, P, Pval, T plus the two virtual-layer generators (+ paired-shuffle controls).
    Returns a dict F that fully determines every deployable kernel; nothing here looks at a query row."""
    F = {}
    F['M_state'] = kernel_state(m)
    g = np.argsort(-X.var(0), kind='stable')[:TOP_GENES]
    F['RNA_genes'] = g
    F['RNA_state'] = kernel_state(X[:, g])
    idx = np.flatnonzero(np.isfinite(p).mean(0) >= P_COVER)
    pf = p[:, idx]; mean = np.nanmean(pf, axis=0)
    assert np.isfinite(mean).all()
    vf = np.where(np.isfinite(pf), pf, mean)
    F['P_idx'] = idx; F['P_impute_mean'] = mean; F['P_state'] = kernel_state(vf)
    F['P_mask_width'] = binary_width(np.isfinite(pf).astype(float))
    F['T_tokens'] = tokens(tis, 'tr')
    # fit-side kernels (needed for generators, L_protein and the fit Gram matrices)
    KM = rbf(m, m, F['M_state'])
    Kval = rbf(vf, vf, F['P_state'])
    F['L_protein'] = build_laplacian(Kval, KNN)
    # generator targets (crosslayer.py): Z_RNA = top-512 of the variance-ordered standardized RNA; Z_P = 512 highest-variance
    # standardized imputed proteins (column-sorted)
    zr = (X[:, g] - F['RNA_state'][0]) / F['RNA_state'][1]
    Z_RNA = zr[:, :TOP_TARGET]
    tsel = np.sort(np.argsort(-vf.var(0), kind='stable')[:TOP_TARGET])
    F['P_target_sel'] = tsel
    Z_P = ((vf - F['P_state'][0]) / F['P_state'][1])[:, tsel]
    n = len(m)
    perm = np.random.default_rng(shuf_seed).permutation(n)
    assert not np.array_equal(perm, np.arange(n))
    F['shuffle_perm'] = perm
    F['gen'] = {}
    for yl, Z in (('RNA', Z_RNA), ('P', Z_P)):
        for tag, target in (('', Z), ('_shuf', Z[perm])):
            oof, coef, mu, lam, r2m, r2med = generator_fit(KM, target, gen_seed)
            F['gen'][f'M->{yl}{tag}'] = dict(lam=lam, coef=coef, mu=mu, oof=oof, vstate=kernel_state(oof),
                                             oof_r2_mean=r2m, oof_r2_median=r2med, n_targets=int(Z.shape[1]))
    F['_fit_cache'] = dict(KM=KM, Kval=Kval, vf=vf, pmask=np.isfinite(pf).astype(float))
    return F


def generator_fit(KX, Z, seed):
    """crosslayer.py generator(): 3-fold cross-fitted KRR X->Z inside fit; lambda by mean OOF MSE; full-fit coefficients."""
    n = len(KX)
    folds = np.array_split(np.random.default_rng(seed).permutation(n), OOF_FOLDS)
    best = None
    for lam in LAMBDAS:
        oof = np.empty_like(Z)
        for held in folds:
            tr = np.setdiff1d(np.arange(n), held)
            mu = Z[tr].mean(0)
            coef = cho_solve(cho_factor(KX[np.ix_(tr, tr)] + lam * np.eye(len(tr)), lower=True), Z[tr] - mu)
            oof[held] = KX[np.ix_(held, tr)] @ coef + mu
        mse = float(((oof - Z) ** 2).mean())
        if best is None or mse < best[0]:
            best = (mse, lam, oof)
    _, lam, oof = best
    mu = Z.mean(0)
    coef = cho_solve(cho_factor(KX + lam * np.eye(n), lower=True), Z - mu)
    r2 = 1 - ((oof - Z) ** 2).sum(0) / np.maximum(((Z - Z.mean(0)) ** 2).sum(0), 1e-12)
    return oof, coef, mu, lam, float(r2.mean()), float(np.median(r2))


def fit_panels(F, X, p, y, rand_seed_list):
    """Supervised panels (protein-panel / rna-panel rules) on the fit cells for endpoint matrix y, plus random panels."""
    out = {}
    pc = p_candidates(p)
    wP = panel_weights(p[:, pc], y)
    orderP = pc[np.argsort(-wP, kind='stable')]
    gc = np.sort(F['RNA_genes'])
    wG = panel_weights(X[:, gc], y)
    orderG = gc[np.argsort(-wG, kind='stable')]
    out['P_cand'] = pc; out['P_weights'] = wP; out['G_cand'] = gc; out['G_weights'] = wG
    out['panels'] = {}
    for k in KS:
        out['panels'][f'P{k}'] = np.sort(orderP[:k]); out['panels'][f'G{k}'] = np.sort(orderG[:k])
    for name, idx in random_panels(pc, rand_seed_list).items():
        out['panels']['P' + name] = idx
    for name, idx in random_panels(gc, rand_seed_list).items():
        out['panels']['G' + name] = idx
    # panel kernel states (fit only)
    out['states'] = {}
    for name, idx in out['panels'].items():
        if name.startswith('P'):
            x = p[:, idx]; mean = np.nanmean(x, axis=0); assert np.isfinite(mean).all()
            out['states'][name] = (mean, kernel_state(np.where(np.isfinite(x), x, mean)))
        else:
            out['states'][name] = (None, kernel_state(X[:, idx]))
    return out


def panel_arm_names():
    names = [f'M+P{k}' for k in KS] + [f'M+G{k}' for k in KS]
    names += [f'M+Prand{k}_s{s}' for k in RKS for s in SEEDS] + [f'M+Grand{k}_s{s}' for k in RKS for s in SEEDS]
    return names


def base_kernels(F, PN, mf, Xf, pf, tf, mq, Xq, pq, tq):
    """All base (K_fit, H_query) pairs for the arms. mf/Xf/pf/tf are the fit arrays used in fit_layers; q = query arrays."""
    B = {}
    KM = rbf(mf, mf, F['M_state']); HM = rbf(mq, mf, F['M_state'])
    B['M'] = (KM, HM)
    tt, tqq = F['T_tokens'], tokens(tq, 'q')
    B['T'] = (same(tt, tt), same(tqq, tt))
    g = F['RNA_genes']
    KR = rbf(Xf[:, g], Xf[:, g], F['RNA_state']); HR = rbf(Xq[:, g], Xf[:, g], F['RNA_state'])
    B['M+RNA'] = (.5 * KM + .5 * KR, .5 * HM + .5 * HR)
    idx = F['P_idx']; mean = F['P_impute_mean']
    vf = np.where(np.isfinite(pf[:, idx]), pf[:, idx], mean); vq = np.where(np.isfinite(pq[:, idx]), pq[:, idx], mean)
    Kval = rbf(vf, vf, F['P_state']); Hval = rbf(vq, vf, F['P_state'])
    Kmask, Hmask = binary_rbf(np.isfinite(pf[:, idx]).astype(float), np.isfinite(pq[:, idx]).astype(float), F['P_mask_width'])
    B['M+Pval'] = (.5 * KM + .5 * Kval, .5 * HM + .5 * Hval)
    B['M+P'] = (.5 * KM + .5 * (.5 * Kval + .5 * Kmask), .5 * HM + .5 * (.5 * Hval + .5 * Hmask))
    for name, G in F['gen'].items():
        zq = HM @ G['coef'] + G['mu']
        Kv = rbf(G['oof'], G['oof'], G['vstate']); Hv = rbf(zq, G['oof'], G['vstate'])
        B['M+v' + name.split('->')[1]] = (.5 * KM + .5 * Kv, .5 * HM + .5 * Hv)
    for name, pidx in PN['panels'].items():
        mean, st = PN['states'][name]
        if name.startswith('P'):
            x = np.where(np.isfinite(pf[:, pidx]), pf[:, pidx], mean); xq = np.where(np.isfinite(pq[:, pidx]), pq[:, pidx], mean)
        else:
            x, xq = Xf[:, pidx], Xq[:, pidx]
        Kp = rbf(x, x, st); Hp = rbf(xq, x, st)
        B['M+' + name] = (.5 * KM + .5 * Kp, .5 * HM + .5 * Hp)
    return B


def arm_configs():
    """Arm -> (base, outer-config kind). Outer config: None (alpha only), 'w' (tissue weight), 'mu' (graph)."""
    A = {'null': None, 'M': ('M', None), 'T': ('T', None), 'M+T': ('M', 'w'),
         'M+vRNA': ('M+vRNA', None), 'M+vP': ('M+vP', None), 'M+vRNA+T': ('M+vRNA', 'w'),
         'M+vRNA_shuf': ('M+vRNA_shuf', None), 'M+vP_shuf': ('M+vP_shuf', None),
         'M+RNA': ('M+RNA', None), 'M+P': ('M+P', None), 'M+Pval': ('M+Pval', None),
         'M+P_graph': ('M', 'mu')}
    for a in panel_arm_names():
        A[a] = (a, None)
    return A


CRISPR_ARMS = ['null', 'M', 'M+vRNA', 'M+vP', 'M+vRNA_shuf', 'M+vP_shuf', 'M+RNA', 'M+P', 'M+Pval'] + panel_arm_names()


def config_labels(kind):
    if kind is None:
        return [f'alpha={a}' for a in ALPHAS]
    if kind == 'w':
        return [f'w={w}|alpha={a}' for w in WS for a in ALPHAS]
    if kind == 'mu':
        return [f'mu={mu}|alpha={a}' for mu in MUS for a in ALPHAS]
    raise ValueError(kind)


def json_ready(v):
    if isinstance(v, dict):
        return {str(k): json_ready(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_ready(x) for x in v]
    if isinstance(v, np.ndarray):
        return json_ready(v.tolist())
    if isinstance(v, np.generic):
        return json_ready(v.item())
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def dump_json(path, obj):
    with open(path, 'w') as fh:
        json.dump(json_ready(obj), fh, indent=1, ensure_ascii=False)

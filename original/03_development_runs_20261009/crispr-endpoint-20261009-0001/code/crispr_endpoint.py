"""CRISPR gene-effect endpoint replication of the cross-layer map + panel law (pre-registered in PLAN.md).

Same 421 split, same layer encodings, same generators, same heads as crosslayer-matrix-20261009-0001;
only the endpoint changes (GDSC AUC -> DepMap-style CRISPR gene effect, 310 x 18443 mapped into the 421 rows).

usage: python crispr_endpoint.py SCENARIO STAGE OUT_DIR
  STAGE = dryrun : fold-0 inner-0 only. Reproduction gate (M kernel + generator lambda/R2 vs the matrix run),
                   inner-grid sanity, and a brute-force check of the fast panel-weight routine. No outer score.
  STAGE = main   : all 5 folds, all 4 banks, outer scoring.
env: CREED_SKIP_SHA=1 skips input sha checks (dev only).
"""
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

ROOT = os.environ.get('CREED_ROOT', '/public/home/mengxl/creed-research-runs')
EXT = ROOT + '/extra-supervision-20261008-0001/output'
SOURCE = ROOT + '/gdsc-metabolite-full-protein-20260908-0001/output/result/source-inputs.npz'
RNA421 = ROOT + '/rna421-20260906-0002'
CRISPR = ROOT + '/crispr310-prepare-20260908-0001/output/result'
MATRIX = ROOT + '/crosslayer-matrix-20261009-0001/output'
SOURCE_SHA = '52d71517a8952e83a1603cd831f09449544102d0ca5ff829ff01be9bf79cd637'

ALPHAS = (.1, 1., 10., 100.)
LAMBDAS = (.1, 1., 10., 100.)
BANKS = ('inner-0', 'inner-1', 'inner-2', 'outer')
INNER = BANKS[:3]
BANK_ID = {b: i for i, b in enumerate(BANKS)}
SCEN_ID = {'new_cell': 0, 'tissue': 1}
TOP_GENES, TOP_TARGET, P_COVER, OOF_FOLDS = 3000, 512, .5, 3
PANEL_COVER, MIN_PAIR = .9, 10
KS = (8, 16, 32, 64)
RKS = (16, 32)
SEEDS = (0, 1, 2, 3, 4)
LAYER_ID = {'P': 0, 'G': 1}
MIN_FIT_CELLS, MIN_SD, MIN_SD_SENS = 20, .2, .3

LAYERS = ('M', 'RNA', 'P')
PAIRS = (('M', 'RNA'), ('M', 'P'), ('RNA', 'P'), ('RNA', 'M'), ('P', 'M'), ('P', 'RNA'))
MEASURED = {'M+RNA': ('M', 'RNA'), 'M+P': ('M', 'P'), 'M+Pval': ('M', 'Pval'),
            'RNA+P': ('RNA', 'P'), 'M+RNA+P': ('M', 'RNA', 'P')}
MEASURED_T = {'T': ('T',), 'M+T': ('M', 'T'), 'RNA+T': ('RNA', 'T'), 'P+T': ('P', 'T')}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'little')


def log(**kw):
    print(json.dumps(kw, default=float), flush=True)


# ---------------------------------------------------------------- model core (mirrors crosslayer.py)
def predict_grid(k, h, y):
    """Per-target mean-centred masked KRR; <MIN_FIT_CELLS labels -> fit mean for every alpha.
    A target with zero observed fit cells gets 0.0 (its inner loss count is then 0 and `choose` falls back)."""
    pred = np.empty((len(ALPHAS), len(h), y.shape[1]))
    groups, empty = {}, 0
    for j in range(y.shape[1]):
        groups.setdefault(np.isfinite(y[:, j]).tobytes(), []).append(j)
    for cols in groups.values():
        obs = np.flatnonzero(np.isfinite(y[:, cols[0]]))
        if not len(obs):
            pred[:, :, cols] = 0.
            empty += len(cols)
            continue
        yy = y[np.ix_(obs, cols)]
        mean = yy.mean(0)
        pred[:, :, cols] = mean
        if len(obs) < MIN_FIT_CELLS:
            continue
        kk = k[np.ix_(obs, obs)]
        hh = h[:, obs]
        for ai, alpha in enumerate(ALPHAS):
            coef = cho_solve(cho_factor(kk + alpha * np.eye(len(obs)), lower=True), yy - mean)
            pred[ai][:, cols] = hh @ coef + mean
    assert np.isfinite(pred).all()
    return pred, empty, len(groups)


def nanstd0(a):
    """Population std over observed entries per column (== np.nanstd), warning-free for all-NaN columns."""
    obs = np.isfinite(a)
    cnt = obs.sum(0)
    mean = np.divide(np.where(obs, a, 0.).sum(0), cnt, out=np.zeros(a.shape[1]), where=cnt > 0)
    var = np.divide(np.where(obs, (a - mean) ** 2, 0.).sum(0), cnt, out=np.zeros(a.shape[1]), where=cnt > 0)
    return np.sqrt(var)


def nanmean0(a):
    obs = np.isfinite(a)
    cnt = obs.sum(0)
    return np.divide(np.where(obs, a, 0.).sum(0), cnt, out=np.zeros(a.shape[1]), where=cnt > 0)


def bank_losses(grid, truth, sd):
    e = ((grid - truth[None]) / sd) ** 2
    n = np.isfinite(e).sum(1)
    return np.divide(np.nansum(e, axis=1), n, out=np.zeros_like(n, float), where=n > 0), (n > 0).astype(int)


def choose(total, count, fallback=1):
    avg = np.divide(total, count, out=np.full_like(total, np.inf), where=count > 0)
    idx = np.argmin(avg, axis=0)
    idx[np.all(count == 0, axis=0)] = fallback
    return idx


def take(grid, sel):
    return grid[sel, :, np.arange(grid.shape[2])].T


def grouped_spearman(truth, pred):
    """Per-column Spearman between truth and pred over each column's observed query rows.
    Identical to looping scipy.stats.spearmanr (Pearson on average ranks); NaN where either side is constant."""
    rho = np.full(truth.shape[1], np.nan)
    groups = {}
    for j in range(truth.shape[1]):
        groups.setdefault(np.isfinite(truth[:, j]).tobytes(), []).append(j)
    for cols in groups.values():
        o = np.flatnonzero(np.isfinite(truth[:, cols[0]]))
        if len(o) < 2:
            continue
        T = truth[np.ix_(o, cols)]
        P = pred[np.ix_(o, cols)]
        ok = (T.std(0) > 0) & (P.std(0) > 1e-15)
        if not ok.any():
            continue
        rt = rankdata(T[:, ok], axis=0)
        rp = rankdata(P[:, ok], axis=0)
        dt = rt - rt.mean(0)
        dp = rp - rp.mean(0)
        den = np.sqrt((dt * dt).sum(0) * (dp * dp).sum(0))
        r = np.divide((dt * dp).sum(0), den, out=np.full(int(ok.sum()), np.nan), where=den > 0)
        rho[np.array(cols)[ok]] = r
    return rho


def metrics(truth, pred, sd, use):
    obs = np.isfinite(truth)
    n = obs.sum(0)
    mse = np.where(obs, ((pred - truth) / sd) ** 2, 0).sum(0) / np.maximum(n, 1)
    bias = (np.where(obs, pred, 0).sum(0) / np.maximum(n, 1) - nanmean0(truth)) / sd
    rho = grouped_spearman(truth, pred)
    return dict(score=float(np.sqrt(mse[use].mean())), spearman=float(np.nanmean(rho[use])),
                centered_mse=float((mse - bias * bias)[use].mean()), mse=mse, rho=rho)


# ---------------------------------------------------------------- kernels (mirrors crosslayer.py)
def kernel_state(x):
    mean = x.mean(0)
    sd = x.std(0)
    sd = np.where(sd < 1e-8, 1., sd)
    z = (x - mean) / sd
    d = cdist(z, z, 'sqeuclidean')
    v = d[np.triu_indices(len(z), 1)]
    v = v[v > 0]
    if not len(v):
        raise ValueError('Degenerate feature bank')
    return mean, sd, float(np.median(v))


def rbf(a, b, state):
    mean, sd, width = state
    return np.exp(-cdist((a - mean) / sd, (b - mean) / sd, 'sqeuclidean') / (2 * width))


def rbf_pair(xf, xq):
    st = kernel_state(xf)
    return rbf(xf, xf, st), rbf(xq, xf, st), st


def binary_rbf(bf, bq):
    d = cdist(bf, bf, 'sqeuclidean')
    v = d[np.triu_indices(len(bf), 1)]
    v = v[v > 0]
    if not len(v):
        return np.ones_like(d), np.ones((len(bq), len(bf))), float('nan')
    width = float(np.median(v))
    return np.exp(-d / (2 * width)), np.exp(-cdist(bq, bf, 'sqeuclidean') / (2 * width)), width


def same(a, b):
    return (a[:, None] == b[None, :]).astype(float)


def tokens(t, tag):
    return np.array([x if x else f'__unknown_{tag}_{i}' for i, x in enumerate(t)])


def layer_inputs(S, X, fit, query):
    """Identical to crosslayer.py layer_inputs: M, RNA, P (value+mask), Pval."""
    L, info = {}, {}
    m = S['m']
    K, H, st = rbf_pair(m[fit], m[query])
    L['M'] = (K, H, (m[fit] - st[0]) / st[1])
    info['M'] = dict(features=int(m.shape[1]), width=st[2], target=int(m.shape[1]))
    xf = X[fit]
    g = np.argsort(-xf.var(0), kind='stable')[:TOP_GENES]
    K, H, st = rbf_pair(xf[:, g], X[np.ix_(query, g)])
    zf = (xf[:, g] - st[0]) / st[1]
    L['RNA'] = (K, H, zf[:, :TOP_TARGET])
    info['RNA'] = dict(features=int(len(g)), width=st[2], target=int(min(TOP_TARGET, len(g))), top_gene_index=int(g[0]))
    p = S['p']
    pf_all = p[fit]
    idx = np.flatnonzero(np.isfinite(pf_all).mean(0) >= P_COVER)
    pf, pq = p[np.ix_(fit, idx)], p[np.ix_(query, idx)]
    mean = np.nanmean(pf, axis=0)
    assert np.isfinite(mean).all()
    vf, vq = np.where(np.isfinite(pf), pf, mean), np.where(np.isfinite(pq), pq, mean)
    Kval, Hval, st = rbf_pair(vf, vq)
    Kmask, Hmask, mwidth = binary_rbf(np.isfinite(pf).astype(float), np.isfinite(pq).astype(float))
    tsel = np.sort(np.argsort(-vf.var(0), kind='stable')[:TOP_TARGET])
    zf = (vf - st[0]) / st[1]
    L['Pval'] = (Kval, Hval, None)
    L['P'] = (.5 * Kval + .5 * Kmask, .5 * Hval + .5 * Hmask, zf[:, tsel])
    info['P'] = dict(candidates=int(len(idx)), value_width=st[2], mask_width=mwidth, target=int(len(tsel)),
                     fit_observed_frac=float(np.isfinite(pf).mean()), query_observed_frac=float(np.isfinite(pq).mean()))
    L['RNA_pool'] = g
    return L, info


def generator(KX, HX, Z, seed):
    """Identical to crosslayer.py generator()."""
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
    zq = HX @ coef + mu
    r2 = 1 - ((oof - Z) ** 2).sum(0) / np.maximum(((Z - Z.mean(0)) ** 2).sum(0), 1e-12)
    return oof, zq, lam, float(r2.mean()), float(np.median(r2))


# ---------------------------------------------------------------- panel weights (exact, grouped by target mask)
def spearman_scores_slow(Pm, Y, min_n=MIN_PAIR):
    """Reference implementation, verbatim from protein_panel.py / rna_panel.py (used only as a dryrun check)."""
    c = Pm.shape[1]
    tot = np.zeros(c)
    pobs = np.isfinite(Pm)
    for j in range(Y.shape[1]):
        o = np.isfinite(Y[:, j])
        if o.sum() < min_n:
            continue
        Xs = Pm[o]
        y = Y[o, j]
        po = pobs[o]
        cnt = po.sum(0)
        rx = rankdata(Xs, axis=0, nan_policy='omit')
        ry = rankdata(np.where(po, y[:, None], np.nan), axis=0, nan_policy='omit')
        with np.errstate(invalid='ignore'):
            mx = np.nanmean(rx, 0)
            my = np.nanmean(ry, 0)
        dx = np.where(po, rx - mx, 0.)
        dy = np.where(po, ry - my, 0.)
        num = (dx * dy).sum(0)
        den = np.sqrt((dx * dx).sum(0) * (dy * dy).sum(0))
        ok = (cnt >= min_n) & (den > 1e-12)
        r = np.zeros(c)
        r[ok] = num[ok] / den[ok]
        tot += np.abs(r)
    return tot


def _dense_block(Xsub, Ysub, min_n):
    """sum_g |Spearman| for fully observed features (ranks of Y shared across features)."""
    rx = rankdata(Xsub, axis=0)
    dx = rx - rx.mean(0)
    dxx = (dx * dx).sum(0)
    ry = rankdata(Ysub, axis=0)
    dy = ry - ry.mean(0)
    dyy = (dy * dy).sum(0)
    num = dx.T @ dy
    den = np.sqrt(np.outer(dxx, dyy))
    r = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    return np.abs(r).sum(1) if len(Xsub) >= min_n else np.zeros(Xsub.shape[1])


def panel_weights(Xf, Yf, min_n=MIN_PAIR):
    """Exact sum over targets of |Spearman(feature, target)| on pairwise-complete fit rows.
    Mathematically identical to spearman_scores_slow, but grouped by target missingness pattern
    (the CRISPR matrix has only ~8 distinct patterns) and split into fully/partially observed features."""
    n, c = Xf.shape
    pobs_all = np.isfinite(Xf)
    tot = np.zeros(c)
    groups = {}
    for g in range(Yf.shape[1]):
        groups.setdefault(np.isfinite(Yf[:, g]).tobytes(), []).append(g)
    for cols in groups.values():
        o = np.flatnonzero(np.isfinite(Yf[:, cols[0]]))
        if len(o) < min_n:
            continue
        po = pobs_all[o]
        Y = Yf[np.ix_(o, cols)]
        full = np.flatnonzero(po.all(0))
        part = np.flatnonzero(~po.all(0))
        if len(full):
            tot[full] += _dense_block(Xf[np.ix_(o, full)], Y, min_n)
        if not len(part):
            continue
        Xp = Xf[np.ix_(o, part)]
        pop = po[:, part]
        cnt = pop.sum(0)
        rx = rankdata(Xp, axis=0, nan_policy='omit')
        with np.errstate(invalid='ignore'):
            mx = np.nanmean(rx, 0)
        dx = np.where(pop, rx - mx, 0.)
        dxx = (dx * dx).sum(0)
        zero = np.zeros(len(part))
        for t in range(Y.shape[1]):
            y = Y[:, t]
            order = np.argsort(y, kind='stable')
            Ps = pop[order]
            C = np.cumsum(Ps, axis=0).astype(float)
            ys = y[order]
            newblk = np.empty(len(ys), bool)
            newblk[0] = True
            np.not_equal(ys[1:], ys[:-1], out=newblk[1:])
            if newblk.all():
                ry = C
            else:
                starts = np.flatnonzero(newblk)
                ends = np.append(starts[1:], len(ys))
                Cb = np.vstack([np.zeros((1, len(part))), C])
                avg = (Cb[starts] + Cb[ends] + 1.) / 2.
                ry = avg[np.cumsum(newblk) - 1]
            s = np.where(Ps, ry, 0.).sum(0)
            my = np.divide(s, cnt, out=zero.copy(), where=cnt > 0)
            dy = np.where(Ps, ry - my, 0.)
            num = (dx[order] * dy).sum(0)
            den = np.sqrt(dxx * (dy * dy).sum(0))
            ok = (cnt >= min_n) & (den > 1e-12)
            r = np.zeros(len(part))
            r[ok] = num[ok] / den[ok]
            tot[part] += np.abs(r)
    return tot


def p_candidates(p_fit):
    obs = np.isfinite(p_fit)
    cnt = obs.sum(0)
    frac = cnt / len(p_fit)
    mean = np.divide(np.where(obs, p_fit, 0).sum(0), cnt, out=np.zeros(p_fit.shape[1]), where=cnt > 0)
    var = np.divide(np.where(obs, (p_fit - mean) ** 2, 0).sum(0), cnt, out=np.zeros(p_fit.shape[1]), where=cnt > 0)
    return np.flatnonzero((frac >= PANEL_COVER) & (var > 1e-12))


def protein_panel_kernel(p, fit, query, idx):
    x, xq = p[np.ix_(fit, idx)], p[np.ix_(query, idx)]
    mean = np.nanmean(x, axis=0)
    assert np.isfinite(mean).all()
    K, H, _ = rbf_pair(np.where(np.isfinite(x), x, mean), np.where(np.isfinite(xq), xq, mean))
    return K, H


def gene_panel_kernel(X, fit, query, idx):
    K, H, _ = rbf_pair(X[np.ix_(fit, idx)], X[np.ix_(query, idx)])
    return K, H


# ---------------------------------------------------------------- arms
def arm_list(scenario):
    arms = list(LAYERS) + ['Pval'] + list(MEASURED)
    if scenario == 'new_cell':
        arms += list(MEASURED_T) + ['M+vRNA+T']
    arms += [f'{x}+v{y}' for x, y in PAIRS] + [f'{x}+v{y}_shuf' for x, y in PAIRS]
    arms += [f'm_panel_P_k{k}' for k in KS] + [f'm_panel_G_k{k}' for k in KS]
    arms += [f'random_P_k{k}_s{s}' for k in RKS for s in SEEDS]
    arms += [f'random_G_k{k}_s{s}' for k in RKS for s in SEEDS]
    return arms


def average(L, parts):
    K = sum(L[p][0] for p in parts) / len(parts)
    H = sum(L[p][1] for p in parts) / len(parts)
    return K, H


# ---------------------------------------------------------------- one bank
def evaluated_mask(yf):
    cnt = np.isfinite(yf).sum(0)
    sd = nanstd0(yf)
    return (cnt >= MIN_FIT_CELLS) & (sd >= MIN_SD), cnt, sd


def run_bank(S, X, F, scenario, f, bank, cols, saved_gen, stage):
    G = np.load(f'{EXT}/{scenario}/fold-{f}/{bank}/geometry.npz', allow_pickle=False)
    fit, query = G['fit'], G['query']
    n = len(fit)
    assert fit.max() < len(S['sample_ids']) and query.max() < len(S['sample_ids'])
    y = F[np.ix_(fit, cols)]
    L, info = layer_inputs(S, X, fit, query)
    kdiff = max(float(np.max(np.abs(L['M'][0] - G['m_only_kernel'][:n, :n]))),
                float(np.max(np.abs(L['M'][1] - G['m_only_cross'][:, :n]))))
    np.testing.assert_allclose(L['M'][0], G['m_only_kernel'][:n, :n], rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(L['M'][1], G['m_only_cross'][:, :n], rtol=1e-11, atol=1e-12)
    if scenario == 'new_cell':
        ttr, tq = tokens(S['tissues'][fit], 'tr'), tokens(S['tissues'][query], 'q')
        L['T'] = (same(ttr, ttr), same(tq, ttr), None)

    grids, empties, ngroups = {}, {}, {}

    def emit(name, K, H):
        g, e, ng = predict_grid(K, H, y)
        grids[name] = g
        empties[name] = e
        ngroups[name] = ng

    for layer in list(LAYERS) + ['Pval']:
        emit(layer, L[layer][0], L[layer][1])
    for name, parts in MEASURED.items():
        emit(name, *average(L, parts))
    if scenario == 'new_cell':
        for name, parts in MEASURED_T.items():
            emit(name, *average(L, parts))

    # virtual layers + paired-shuffle controls (crosslayer.py rule)
    seed_oof = stable_seed('oof', scenario, f, bank)
    perm = np.random.default_rng(stable_seed('pair_shuffle', scenario, f, bank)).permutation(n)
    assert not np.array_equal(perm, np.arange(n))
    gen_info, vkern = {}, {}
    for x, yl in PAIRS:
        KX, HX, _ = L[x]
        Z = L[yl][2]
        for tag, target in (('', Z), ('_shuf', Z[perm])):
            oof, zq, lam, r2m, r2med = generator(KX, HX, target, seed_oof)
            Kv, Hv, st = rbf_pair(oof, zq)
            emit(f'{x}+v{yl}{tag}', .5 * KX + .5 * Kv, .5 * HX + .5 * Hv)
            gen_info[f'{x}->{yl}{tag}'] = dict(lam=lam, oof_r2_mean=r2m, oof_r2_median=r2med,
                                               virtual_width=st[2], n_targets=int(Z.shape[1]))
            if tag == '' and x == 'M' and yl == 'RNA':
                vkern['vRNA'] = (Kv, Hv)
    if scenario == 'new_cell':
        Kv, Hv = vkern['vRNA']
        emit('M+vRNA+T', (L['M'][0] + Kv + L['T'][0]) / 3., (L['M'][1] + Hv + L['T'][1]) / 3.)

    # reproduction gate 2: generator lambda / OOF R2 must match the matrix run bank-for-bank
    gd = dict(lam=0., r2_mean=0., r2_median=0., width=0.)
    for k, v in gen_info.items():
        sv = saved_gen[k]
        gd['lam'] = max(gd['lam'], abs(v['lam'] - sv['lam']))
        gd['r2_mean'] = max(gd['r2_mean'], abs(v['oof_r2_mean'] - sv['oof_r2_mean']))
        gd['r2_median'] = max(gd['r2_median'], abs(v['oof_r2_median'] - sv['oof_r2_median']))
        gd['width'] = max(gd['width'], abs(v['virtual_width'] - sv['virtual_width']))

    # ---- panels (fit-only selection on this bank's own evaluated targets)
    ev_b, cnt_b, sd_b = evaluated_mask(F[fit])
    ev_idx = np.flatnonzero(ev_b)
    pc = p_candidates(S['p'][fit])
    gc = L['RNA_pool']
    t_panel = time.perf_counter()
    wP = panel_weights(S['p'][np.ix_(fit, pc)], F[np.ix_(fit, ev_idx)])
    wG = panel_weights(X[np.ix_(fit, gc)], F[np.ix_(fit, ev_idx)])
    panel_seconds = time.perf_counter() - t_panel
    selP = pc[np.argsort(-wP, kind='stable')]
    selG = gc[np.argsort(-wG, kind='stable')]
    panels = {}
    for k in KS:
        panels[f'm_panel_P_k{k}'] = ('P', selP[:k])
        panels[f'm_panel_G_k{k}'] = ('G', selG[:k])
    for k in RKS:
        for s in SEEDS:
            for tag, pool in (('P', pc), ('G', gc)):
                rng = np.random.default_rng([20261009, SCEN_ID[scenario], f, BANK_ID[bank], s, LAYER_ID[tag]])
                panels[f'random_{tag}_k{k}_s{s}'] = (tag, rng.choice(pool, size=k, replace=False))
    for name, (tag, idx) in panels.items():
        Kp, Hp = protein_panel_kernel(S['p'], fit, query, idx) if tag == 'P' else gene_panel_kernel(X, fit, query, idx)
        emit(name, .5 * L['M'][0] + .5 * Kp, .5 * L['M'][1] + .5 * Hp)

    info.update(n_fit=int(n), n_query=int(len(query)),
                n_fit_crispr=int(np.isfinite(F[fit]).any(1).sum()), n_query_crispr=int(np.isfinite(F[query]).any(1).sum()),
                n_cols=int(len(cols)), evaluated_this_bank=int(ev_b.sum()),
                evaluated_sd03_this_bank=int(((cnt_b >= MIN_FIT_CELLS) & (sd_b >= MIN_SD_SENS)).sum()),
                kernel_block_max_abs_diff=kdiff, generator_max_abs_diff=gd, generators=gen_info,
                panel_candidates=dict(protein=int(len(pc)), gene=int(len(gc))),
                panel_seconds=round(panel_seconds, 2),
                panel_top8=dict(protein=[str(x) for x in S['protein_ids'][selP[:8]]], gene=[int(x) for x in selG[:8]]),
                empty_target_cols=int(max(empties.values())), missingness_groups=int(max(ngroups.values())))
    meta = dict(fit=fit, query=query, panel_idx={k: v[1] for k, v in panels.items()},
                panel_weights=dict(protein=wP, gene=wG), panel_pools=dict(protein=pc, gene=gc))
    if stage == 'dryrun':
        sl = min(40, len(ev_idx))
        sub = ev_idx[np.linspace(0, len(ev_idx) - 1, sl).astype(int)]
        chk = {}
        for tag, Xc, pool in (('protein', S['p'][np.ix_(fit, pc[:250])], pc[:250]), ('gene', X[np.ix_(fit, gc[:250])], gc[:250])):
            fastv = panel_weights(Xc, F[np.ix_(fit, sub)])
            slowv = spearman_scores_slow(Xc, F[np.ix_(fit, sub)])
            chk[tag] = float(np.max(np.abs(fastv - slowv)))
        info['panel_weight_fast_vs_slow_max_abs_diff'] = chk
    return grids, meta, info


# ---------------------------------------------------------------- data
def load_ctx():
    if os.environ.get('CREED_SKIP_SHA') != '1':
        assert sha(SOURCE) == SOURCE_SHA, 'source sha mismatch'
    S = np.load(SOURCE, allow_pickle=False)
    S = {k: S[k] for k in ('m', 'auc', 'p', 'tissues', 'sample_ids', 'protein_ids')}
    X = np.load(RNA421 + '/rna_log_cpm.npy')
    assert (np.load(RNA421 + '/sample_ids.npy') == S['sample_ids']).all(), 'RNA row order differs from source'
    assert X.shape[0] == len(S['sample_ids']) and np.isfinite(X).all() and np.isfinite(S['m']).all()
    E = np.load(CRISPR + '/effects.npy')
    csid = np.load(CRISPR + '/sample_ids.npy', allow_pickle=True)
    gid = np.load(CRISPR + '/gene_ids.npy', allow_pickle=True).astype(str)
    pos = {str(s): i for i, s in enumerate(S['sample_ids'])}
    rows = np.array([pos[str(s)] for s in csid])
    assert len(set(rows.tolist())) == len(rows) == E.shape[0] == 310
    F = np.full((len(S['sample_ids']), E.shape[1]), np.nan)
    F[rows] = E
    return S, X, F, gid, rows


def main():
    scenario, stage, out = sys.argv[1], sys.argv[2], sys.argv[3]
    assert scenario in ('new_cell', 'tissue') and stage in ('dryrun', 'main')
    os.makedirs(out, exist_ok=False)
    t0 = time.perf_counter()
    S, X, F, gid, crows = load_ctx()
    saved = json.load(open(f'{MATRIX}/{scenario}/bank_info.json'))
    arms = arm_list(scenario)
    json.dump(dict(scenario=scenario, stage=stage, script_sha256=sha(__file__), source_sha256=SOURCE_SHA,
                   alphas=ALPHAS, lambdas=LAMBDAS, top_genes=TOP_GENES, top_target=TOP_TARGET, p_cover=P_COVER,
                   panel_cover=PANEL_COVER, min_pair=MIN_PAIR, ks=KS, rks=RKS, seeds=SEEDS,
                   min_fit_cells=MIN_FIT_CELLS, min_sd=MIN_SD, oof_folds=OOF_FOLDS, pairs=PAIRS, arms=arms,
                   n_crispr=int(len(crows)), n_genes=int(F.shape[1]), rna_shape=list(X.shape)),
              open(f'{out}/protocol.json', 'w'), indent=1)
    gate = dict(kernel_block_max_abs_diff=0., generator_lam_max_abs_diff=0., generator_r2_mean_max_abs_diff=0.,
                generator_r2_median_max_abs_diff=0., generator_width_max_abs_diff=0.)
    fold_rows, pertarget_rows, bank_info_all = [], [], {}
    folds = (0,) if stage == 'dryrun' else range(5)
    for f in folds:
        Go = np.load(f'{EXT}/{scenario}/fold-{f}/outer/geometry.npz', allow_pickle=False)
        ev_outer, cnt_o, sd_o = evaluated_mask(F[Go['fit']])
        cols = np.flatnonzero(ev_outer)
        loss, outer, ometa, binfo = {}, None, None, {}
        for bank in (('inner-0',) if stage == 'dryrun' else BANKS):
            tb = time.perf_counter()
            grids, meta, info = run_bank(S, X, F, scenario, f, bank, cols, saved[f'fold-{f}/{bank}']['generators'], stage)
            gate['kernel_block_max_abs_diff'] = max(gate['kernel_block_max_abs_diff'], info['kernel_block_max_abs_diff'])
            for key, gk in (('lam', 'generator_lam_max_abs_diff'), ('r2_mean', 'generator_r2_mean_max_abs_diff'),
                            ('r2_median', 'generator_r2_median_max_abs_diff'), ('width', 'generator_width_max_abs_diff')):
                gate[gk] = max(gate[gk], info['generator_max_abs_diff'][key])
            truth_b = F[np.ix_(meta['query'], cols)]
            sd_b = np.maximum(nanstd0(F[np.ix_(meta['fit'], cols)]), 1e-8)
            if bank == 'outer':
                outer, ometa = grids, meta
            else:
                for a, g in grids.items():
                    loss.setdefault(a, {})[bank] = bank_losses(g, truth_b, sd_b)
            binfo[bank] = info
            log(event='bank_done', scenario=scenario, fold=f, bank=bank, n_cols=len(cols),
                kdiff=info['kernel_block_max_abs_diff'], gdiff=info['generator_max_abs_diff'],
                p_cand=info['panel_candidates']['protein'], panel_s=info['panel_seconds'],
                seconds=round(time.perf_counter() - tb, 1), total=round(time.perf_counter() - t0, 1))
        bank_info_all[f] = binfo
        if stage == 'dryrun':
            json.dump(dict(scenario=scenario, stage=stage, gate=gate, n_evaluated_outer=int(len(cols)),
                           bank_info=binfo, seconds=time.perf_counter() - t0),
                      open(f'{out}/summary.json', 'w'), indent=1, default=float)
            sl = {a: np.bincount(choose(loss[a]['inner-0'][0], loss[a]['inner-0'][1]), minlength=4).tolist() for a in arms}
            json.dump(sl, open(f'{out}/inner0_selection_counts.json', 'w'), indent=1)
            log(event='dryrun_done', gate=gate, n_evaluated_outer=int(len(cols)),
                fast_vs_slow=binfo['inner-0']['panel_weight_fast_vs_slow_max_abs_diff'],
                seconds=round(time.perf_counter() - t0, 1))
            return

        def summed(arm):
            return sum(loss[arm][b][0] for b in INNER), sum(loss[arm][b][1] for b in INNER)

        truth = F[np.ix_(ometa['query'], cols)]
        sd = np.maximum(nanstd0(F[np.ix_(ometa['fit'], cols)]), 1e-8)
        nq_obs = np.isfinite(truth).sum(0)
        use = nq_obs > 0                      # definitional: MSE undefined with zero observed query cells
        n_dropped = int((~use).sum())
        preds, sels = {}, {}
        for a in arms:
            sel = choose(*summed(a))
            sels[a] = sel
            preds[a] = take(outer[a], sel)
        preds['null'] = np.broadcast_to(nanmean0(F[np.ix_(ometa['fit'], cols)]), truth.shape).copy()
        np.savez_compressed(f'{out}/fold-{f}_predictions.npz', truth=truth, train_sd=sd, evaluated=use,
                            genes=cols, gene_ids=gid[cols], held=ometa['query'], fit=ometa['fit'],
                            n_query_obs=nq_obs, fit_obs_count=cnt_o[cols], fit_sd_all=sd_o[cols],
                            **{'pred__' + k: v for k, v in preds.items()})
        np.savez_compressed(f'{out}/fold-{f}_selection.npz',
                            **{'sel__' + a: v for a, v in sels.items()},
                            **{f'loss__{a}__{b}': v[0] for a, d in loss.items() for b, v in d.items()},
                            **{f'count__{a}__{b}': v[1] for a, d in loss.items() for b, v in d.items()},
                            **{f'panelidx__{a}': v for a, v in ometa['panel_idx'].items()},
                            panelw_protein=ometa['panel_weights']['protein'], panelw_gene=ometa['panel_weights']['gene'],
                            panelpool_protein=ometa['panel_pools']['protein'], panelpool_gene=ometa['panel_pools']['gene'])
        json.dump(binfo, open(f'{out}/fold-{f}_bank_info.json', 'w'), indent=1, default=float)
        for a, v in preds.items():
            mm = metrics(truth, v, sd, use)
            fold_rows.append(dict(fold=f, arm=a, score=mm['score'], spearman=mm['spearman'], centered_mse=mm['centered_mse'],
                                  targets=int(use.sum()), targets_dropped=n_dropped,
                                  n_fit=int(len(ometa['fit'])), n_query=int(len(ometa['query'])),
                                  n_fit_crispr=binfo['outer']['n_fit_crispr'], n_query_crispr=binfo['outer']['n_query_crispr']))
            for j in np.flatnonzero(use):
                pertarget_rows.append(dict(fold=f, arm=a, gene=str(gid[cols[j]]), train_sd=float(sd[j]),
                                           standardized_mse=float(mm['mse'][j]), spearman=float(mm['rho'][j])))
        log(event='fold_done', scenario=scenario, fold=f, n_evaluated=int(use.sum()), dropped=n_dropped,
            score_M=[r['score'] for r in fold_rows if r['fold'] == f and r['arm'] == 'M'][0],
            total=round(time.perf_counter() - t0, 1))
    with open(f'{out}/fold_metrics.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(fold_rows[0]))
        w.writeheader()
        w.writerows(fold_rows)
    with open(f'{out}/per_target_metrics.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(pertarget_rows[0]))
        w.writeheader()
        w.writerows(pertarget_rows)
    names = list(dict.fromkeys(r['arm'] for r in fold_rows))
    gate['passed'] = bool(gate['kernel_block_max_abs_diff'] <= 1e-11 and gate['generator_lam_max_abs_diff'] == 0.
                          and max(gate['generator_r2_mean_max_abs_diff'], gate['generator_r2_median_max_abs_diff'],
                                  gate['generator_width_max_abs_diff']) <= 1e-9)
    res = dict(scenario=scenario, stage=stage, reproduction_gate=gate,
               summary={a: {k: float(np.mean([r[k] for r in fold_rows if r['arm'] == a]))
                            for k in ('score', 'spearman', 'centered_mse')} for a in names},
               fold_scores={a: [r['score'] for r in fold_rows if r['arm'] == a] for a in names},
               evaluated_per_fold={int(f): int([r['targets'] for r in fold_rows if r['fold'] == f][0]) for f in range(5)},
               dropped_per_fold={int(f): int([r['targets_dropped'] for r in fold_rows if r['fold'] == f][0]) for f in range(5)},
               bank_info=bank_info_all, seconds=time.perf_counter() - t0)
    json.dump(res, open(f'{out}/summary.json', 'w'), indent=1, default=float)
    log(event='done', scenario=scenario, gate=gate, seconds=round(time.perf_counter() - t0, 1))


if __name__ == '__main__':
    main()

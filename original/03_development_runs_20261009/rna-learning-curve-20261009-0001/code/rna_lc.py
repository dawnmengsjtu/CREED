"""RNA-only learning curve + 354 RNA-GDSC extra-label gate + pseudo-label M student (pre-registered in PLAN.md).

Independent implementation (no project imports). Functions predict_grid / graph / choose / take / metrics follow the
reference implementation in tissue_kernel_20261009/tissue_kernel.py and extra-supervision code/run_*.py.
Reads only allowed inputs; writes only to OUT_DIR (must not exist).

usage: python rna_lc.py SCENARIO OUT_DIR
"""
import csv
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
from scipy.linalg import eigh, cho_factor, cho_solve
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

R = '/public/home/mengxl/creed-research-runs'
EXT = R + '/extra-supervision-20261008-0001/output'
RES = R + '/extra-supervision-20261008-0001/resources'
SOURCE = R + '/gdsc-metabolite-full-protein-20260908-0001/output/result/source-inputs.npz'
RNA421 = R + '/rna421-20260906-0002'
SOURCE_SHA = '52d71517a8952e83a1603cd831f09449544102d0ca5ff829ff01be9bf79cd637'
PKG_SHA = {'rna_gdsc_extra354.npz': 'b6323b4025e174217fb91afa4766dd5d134bef43189e3d86a9c8b5784e195102',
           'm_rna_extra66.npz': 'bd8d13f57b691f579dfcd6210a44f4ab1e05a8691ced62ff2baeb7a6a67883af',
           'm_gdsc_extra19.npz': 'd4e7dc603f491217805a1d06b673f8c2662a8c0f2ad77ec875f7d7868526384c'}

ALPHAS = (.1, 1., 10., 100.)
KS = (2000, 5000)
WS = (0., .25, 1.)              # tissue-kernel weights for rnat arms
MUS = (0., .1, 1., 10., 100., 1000.)
PWS = (0., .25, 1.)             # pseudo-label row weights
FRACS = (25, 50, 75, 100)
BANKS = ('inner-0', 'inner-1', 'inner-2', 'outer')
INNER = BANKS[:3]
MIN_SOURCE = 10


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'little')


def log(**kw):
    print(json.dumps(kw), flush=True)


# ---------------------------------------------------------------- model core (reference-equivalent)
def predict_grid(k, h, y):
    """Per-drug mean-centred masked KRR; <20 labels -> fit mean for every alpha."""
    pred = np.empty((len(ALPHAS), len(h), y.shape[1])); groups = {}
    for j in range(y.shape[1]):
        groups.setdefault(np.isfinite(y[:, j]).tobytes(), []).append(j)
    for cols in groups.values():
        obs = np.flatnonzero(np.isfinite(y[:, cols[0]]))
        if not len(obs):
            raise ValueError('No current-fit label')
        yy = y[np.ix_(obs, cols)]; mean = yy.mean(0)
        pred[:, :, cols] = mean
        if len(obs) < 20:
            continue
        kk = k[np.ix_(obs, obs)]; hh = h[:, obs]
        for ai, alpha in enumerate(ALPHAS):
            coef = cho_solve(cho_factor(kk + alpha * np.eye(len(obs)), lower=True), yy - mean)
            pred[ai][:, cols] = hh @ coef + mean
    assert np.isfinite(pred).all()
    return pred


def predict_grid_w(k, h, y, yp, wp):
    """Weighted variant: rows of y (real, NaN allowed) weight 1, appended pseudo rows yp (finite) weight wp>0.
    Solves (K_oo + alpha W^-1) c = y - weighted mean. <20 real labels -> real-label fit mean."""
    nr, npse = len(y), len(yp)
    assert k.shape[0] == nr + npse and wp > 0 and np.isfinite(yp).all()
    pred = np.empty((len(ALPHAS), len(h), y.shape[1])); groups = {}
    for j in range(y.shape[1]):
        groups.setdefault(np.isfinite(y[:, j]).tobytes(), []).append(j)
    for cols in groups.values():
        obs = np.flatnonzero(np.isfinite(y[:, cols[0]]))
        if not len(obs):
            raise ValueError('No current-fit label')
        if len(obs) < 20:
            pred[:, :, cols] = y[np.ix_(obs, cols)].mean(0)
            continue
        rows = np.concatenate([obs, nr + np.arange(npse)])
        yy = np.vstack([y[np.ix_(obs, cols)], yp[:, cols]])
        wts = np.concatenate([np.ones(len(obs)), np.full(npse, wp)])
        mean = (wts[:, None] * yy).sum(0) / wts.sum()
        kk = k[np.ix_(rows, rows)]; hh = h[:, rows]; inv = np.diag(1. / wts)
        for ai, alpha in enumerate(ALPHAS):
            coef = cho_solve(cho_factor(kk + alpha * inv, lower=True), yy - mean)
            pred[ai][:, cols] = hh @ coef + mean
    assert np.isfinite(pred).all()
    return pred


def graph_parts(k, h, L):
    """Effective (kernel, cross) for each mu, same formula as run_extra.py."""
    vals, vecs = eigh((k + k.T) / 2)
    assert vals.min() > -1e-8
    root = (vecs * np.sqrt(np.maximum(vals, 0))) @ vecs.T
    out = [(k, h)]
    for mu in MUS[1:]:
        a = np.eye(len(k)) + mu * root @ L @ root; a = (a + a.T) / 2
        ke = root @ cho_solve(cho_factor(a, lower=True), root); ke = (ke + ke.T) / 2
        he = h - mu * h @ L @ ke
        out.append((ke, he))
    return out


def kernel_state(x):
    mean = x.mean(0); sd = x.std(0); sd = np.where(sd < 1e-8, 1., sd); z = (x - mean) / sd
    d = cdist(z, z, 'sqeuclidean'); v = d[np.triu_indices(len(z), 1)]; v = v[v > 0]
    return mean, sd, float(np.median(v))


def rbf(a, b, state):
    mean, sd, width = state
    return np.exp(-cdist((a - mean) / sd, (b - mean) / sd, 'sqeuclidean') / (2 * width))


def bank_losses(grid, truth, sd):
    e = ((grid - truth[None]) / sd) ** 2
    n = np.isfinite(e).sum(1)
    return np.divide(np.nansum(e, axis=1), n, out=np.zeros_like(n, float), where=n > 0), (n > 0).astype(int)


def choose(total, count, fallback=1):
    avg = np.divide(total, count, out=np.full_like(total, np.inf), where=count > 0)
    idx = np.argmin(avg, axis=0); idx[np.all(count == 0, axis=0)] = fallback
    return idx


def take(grid, sel):
    return np.take_along_axis(grid, sel[None, None, :], axis=0)[0]


def metrics(truth, pred, sd, use):
    obs = np.isfinite(truth); n = obs.sum(0)
    mse = np.where(obs, ((pred - truth) / sd) ** 2, 0).sum(0) / np.maximum(n, 1)
    bias = (np.where(obs, pred, 0).sum(0) / np.maximum(n, 1) - np.nanmean(truth, axis=0)) / sd
    rho = np.full(len(sd), np.nan)
    for j in np.flatnonzero(use):
        v = obs[:, j]
        if np.std(truth[v, j]) > 0 and np.std(pred[v, j]) > 1e-15:
            rho[j] = spearmanr(truth[v, j], pred[v, j]).statistic
    return dict(score=float(np.sqrt(mse[use].mean())), spearman=float(np.nanmean(rho[use])),
                centered_mse=float((mse - bias * bias)[use].mean()), mse=mse, rho=rho)


# ---------------------------------------------------------------- features
def rna_kernels(xtr, str_, xqs, sqs, corr):
    """Fit-only per-source centring (corr='c') or none ('u'); top-k variance genes; fit z-score; RBF median width.
    Returns {k: (K, [H for each query block])}."""
    if corr == 'c':
        mu_all = xtr.mean(0); means = {}
        for s in np.unique(str_):
            r = str_ == s
            if r.sum() >= MIN_SOURCE:
                means[s] = xtr[r].mean(0)

        def apply(x, s):
            out = x.copy()
            for v in np.unique(s):
                out[s == v] -= means.get(v, mu_all)
            return out
        xtr = apply(xtr, str_); xqs = [apply(x, s) for x, s in zip(xqs, sqs)]
    order = np.argsort(-xtr.var(0), kind='stable')
    out = {}
    for k in KS:
        g = order[:k]; a = xtr[:, g]
        mean = a.mean(0); sd = a.std(0); sd = np.where(sd < 1e-8, 1., sd)
        z = (a - mean) / sd
        d = cdist(z, z, 'sqeuclidean'); v = d[np.triu_indices(len(z), 1)]; v = v[v > 0]; width = float(np.median(v))
        out[k] = (np.exp(-d / (2 * width)), [np.exp(-cdist((x[:, g] - mean) / sd, z, 'sqeuclidean') / (2 * width)) for x in xqs])
    return out


def tokens(t, tag):
    return np.array([x if x else f'__unknown_{tag}_{i}' for i, x in enumerate(t)])


def same(a, b):
    return (a[:, None] == b[None, :]).astype(float)


def allowed(tissues, auc_panel, fit, stis, scenario):
    ok = np.isfinite(auc_panel).any(1) if auc_panel is not None else np.ones(len(tissues), bool)
    if scenario == 'tissue':
        banned = set(stis[np.setdiff1d(np.arange(len(stis)), fit)].tolist())
        ok &= np.array([bool(t) and t not in banned for t in tissues])
    return np.flatnonzero(ok)


# ---------------------------------------------------------------- data
def load_ctx():
    assert sha(SOURCE) == SOURCE_SHA
    for name, h in PKG_SHA.items():
        assert sha(f'{RES}/{name}') == h, name
    S = np.load(SOURCE, allow_pickle=False)
    c = dict(m=S['m'], auc=S['auc'], stis=S['tissues'], sid=S['sample_ids'])
    c['X'] = np.load(RNA421 + '/rna_log_cpm.npy'); c['src'] = np.load(RNA421 + '/rna_source_names.npy')
    assert (np.load(RNA421 + '/sample_ids.npy') == c['sid']).all() and c['X'].shape == (421, 34985)
    E = np.load(f'{RES}/rna_gdsc_extra354.npz', allow_pickle=False)
    c.update(XE=E['rna_log_cpm'], srcE=E['rna_source_names'], aucE=E['auc'], tisE=E['tissues'])
    B = np.load(f'{RES}/m_rna_extra66.npz', allow_pickle=False)
    M19 = np.load(f'{RES}/m_gdsc_extra19.npz', allow_pickle=False)
    assert (E['drug_ids'] == S['drug_ids']).all() and (M19['drug_ids'] == S['drug_ids']).all()
    assert (E['gene_ids'] == np.load(RNA421 + '/gene_ids.npy')).all() and (B['gene_ids'] == E['gene_ids']).all()
    assert (B['m_feature_ids'] == S['metabolite_ids']).all() and (M19['m_feature_ids'] == S['metabolite_ids']).all()
    assert not set(E['sample_ids'].tolist()) & set(c['sid'].tolist())
    assert not set(B['sample_ids'].tolist()) & set(c['sid'].tolist())
    s19, s354 = set(M19['sample_ids'].tolist()), set(E['sample_ids'].tolist())
    pse = np.array([i for i, s in enumerate(B['sample_ids']) if s not in s19 and s not in s354])
    assert len(pse) == 48
    c.update(XB=B['rna_log_cpm'][pse], srcB=B['rna_source_names'][pse], mB=B['m'][pse], tisB=B['tissues'][pse],
             pse_ids=B['sample_ids'][pse], m19=M19['m'], auc19=M19['auc'])
    assert np.isfinite(c['XB']).all() and np.isfinite(c['mB']).all() and np.isfinite(c['XE']).all() and np.isfinite(c['X']).all()
    return c


def step1_bank(c, scenario, f, bank):
    m, auc, stis, X, src = c['m'], c['auc'], c['stis'], c['X'], c['src']
    XE, srcE, aucE, tisE, XB, srcB, tisB = (c[k] for k in ('XE', 'srcE', 'aucE', 'tisE', 'XB', 'srcB', 'tisB'))
    G = np.load(f'{EXT}/{scenario}/fold-{f}/{bank}/geometry.npz', allow_pickle=False)
    fit, query, panel, ei19, Lp = G['fit'], G['query'], G['panel'], G['extra_indices'], G['L_protein']
    n = len(fit)
    yfit = auc[np.ix_(fit, panel)]
    e354 = allowed(tisE, aucE[:, panel], fit, stis, scenario)
    ep = allowed(tisB, None, fit, stis, scenario)
    ytr = np.vstack([yfit, aucE[np.ix_(e354, panel)]])
    perm = np.random.default_rng(stable_seed('rna_lc', scenario, f, bank)).permutation(n)
    subs = {p: np.sort(perm[:math.ceil(p / 100 * n)]) for p in FRACS}
    assert (subs[100] == np.arange(n)).all()
    grids = {}
    # M learning curve (m_f100 = reproduction of original__m_only)
    for p in FRACS:
        rows = fit[subs[p]]; st = kernel_state(m[rows])
        grids[f'm_f{p}'] = predict_grid(rbf(m[rows], m[rows], st), rbf(m[query], m[rows], st), auc[np.ix_(rows, panel)])
    # RNA learning curve and extra
    feats_c100 = feats_cext = teacher = None
    for corr in ('c', 'u'):
        for p in FRACS:
            rows = fit[subs[p]]
            fk = rna_kernels(X[rows], src[rows], [X[query]], [src[query]], corr)
            grids[f'rna_{corr}_f{p}'] = np.concatenate([predict_grid(fk[k][0], fk[k][1][0], auc[np.ix_(rows, panel)]) for k in KS])
            if corr == 'c' and p == 100:
                feats_c100 = fk
        xtr = np.vstack([X[fit], XE[e354]]); strn = np.concatenate([src[fit], srcE[e354]])
        if corr == 'c':
            fk = rna_kernels(xtr, strn, [X[query], XB[ep]], [src[query], srcB[ep]], corr)
            parts, tparts = [], []
            for k in KS:
                g = predict_grid(fk[k][0], np.vstack([fk[k][1][0], fk[k][1][1]]), ytr)
                parts.append(g[:, :len(query)]); tparts.append(g[:, len(query):])
            grids['rna_c_extra'] = np.concatenate(parts)
            teacher = np.concatenate(tparts)
            feats_cext = fk
        else:
            fk = rna_kernels(xtr, strn, [X[query]], [src[query]], corr)
            grids['rna_u_extra'] = np.concatenate([predict_grid(fk[k][0], fk[k][1][0], ytr) for k in KS])
        del xtr
    # tissue references
    ttr = tokens(stis[fit], 'tr'); tq = tokens(stis[query], 'q')
    T, Tq = same(ttr, ttr), same(tq, ttr)
    grids['tissue_f100'] = predict_grid(T, Tq, yfit)
    tte = tokens(np.concatenate([stis[fit], tisE[e354]]), 'tr')
    Te, Tqe = same(tte, tte), same(tq, tte)
    grids['tissue_extra'] = predict_grid(Te, Tqe, ytr)
    # RNA + tissue (corrected); w=0 block reuses the rna_c grid
    for name, base_arm, fk, TT, TQ, yy in (('rnat_c_f100', 'rna_c_f100', feats_c100, T, Tq, yfit),
                                           ('rnat_c_extra', 'rna_c_extra', feats_cext, Te, Tqe, ytr)):
        parts = []
        for w in WS:
            for ki, k in enumerate(KS):
                if w == 0:
                    parts.append(grids[base_arm][ki * len(ALPHAS):(ki + 1) * len(ALPHAS)])
                else:
                    parts.append(predict_grid(fk[k][0] + w * TT, fk[k][1][0] + w * TQ, yy))
        grids[name] = np.concatenate(parts)
    meta = dict(fit=fit, query=query, panel=panel, ei19=ei19, Lp=Lp, e354=e354, ep=ep,
                ytr_mean=np.nanmean(ytr, axis=0), yfit_mean=np.nanmean(yfit, axis=0))
    return grids, meta, teacher


def step2_bank(c, scenario, f, bank, meta, teacher, tsel):
    m, auc, mB, m19, auc19 = c['m'], c['auc'], c['mB'], c['m19'], c['auc19']
    fit, query, panel, ei19, Lp, ep = (meta[k] for k in ('fit', 'query', 'panel', 'ei19', 'Lp', 'ep'))
    n, n19, npse = len(fit), len(ei19), len(ep)
    yp = take(teacher, tsel) if npse else np.zeros((0, len(panel)))
    rng = np.random.default_rng(stable_seed('pseudo_shuffle', scenario, f, bank))
    yp_shuf = yp[rng.permutation(npse)] if npse else yp
    yp_mean = np.broadcast_to(meta['ytr_mean'], yp.shape).copy()
    st = kernel_state(m[fit])
    mrows = np.vstack([m[fit], m19[ei19], mB[ep]])
    K = rbf(mrows, mrows, st); H = rbf(m[query], mrows, st)
    nr = n + n19
    yreal = np.vstack([auc[np.ix_(fit, panel)], auc19[np.ix_(ei19, panel)]])
    base = predict_grid(K[:nr, :nr], H[:, :nr], yreal)
    grids = {'m_extra19': base}
    for arm, lab in (('m_pseudo', yp), ('m_pseudo_shuffle', yp_shuf), ('m_pseudo_mean', yp_mean)):
        parts = [base]
        for w in PWS[1:]:
            parts.append(predict_grid_w(K, H, yreal, lab, w) if npse else base)
        grids[arm] = np.concatenate(parts)
    # P graph: w=0 block exactly as run_extra (nodes fit+19; drugs without extra labels from fit-only graph)
    assert Lp.shape == (nr, nr)
    g19 = np.concatenate([predict_grid(ke, he, yreal) for ke, he in graph_parts(K[:nr, :nr], H[:, :nr], Lp)])
    no_extra = ~np.isfinite(auc19[np.ix_(ei19, panel)]).any(0)
    if no_extra.any():
        g0 = np.concatenate([predict_grid(ke, he, yreal[:n]) for ke, he in graph_parts(K[:n, :n], H[:, :n], Lp[:n, :n])])
        g19[:, :, no_extra] = g0[:, :, no_extra]
    grids['mP_extra19'] = g19
    parts = [g19]
    if npse:
        Lext = np.zeros((nr + npse, nr + npse)); Lext[:nr, :nr] = Lp
        gp = graph_parts(K, H, Lext)
        for w in PWS[1:]:
            parts.append(np.concatenate([predict_grid_w(ke, he, yreal, yp, w) for ke, he in gp]))
    else:
        parts += [g19] * (len(PWS) - 1)
    grids['mP_pseudo'] = np.concatenate(parts)
    return grids, yp


SAVED = ('original__m_only', 'extra__m_only', 'extra__m_only__protein', 'original__m_only__protein', 'original__virtual_fm',
         'original__virtual_fm__protein', 'extra__virtual_fm', 'extra__virtual_fm__protein')


def main():
    scenario, out_dir = sys.argv[1], sys.argv[2]
    assert scenario in ('new_cell', 'tissue')
    os.makedirs(out_dir, exist_ok=False)
    t0 = time.perf_counter()
    c = load_ctx()
    json.dump(dict(scenario=scenario, script_sha256=sha(__file__), source_sha256=SOURCE_SHA, packages=PKG_SHA,
                   alphas=ALPHAS, ks=KS, ws=WS, mus=MUS, pws=PWS, fracs=FRACS, min_source=MIN_SOURCE,
                   pseudo_sample_ids=c['pse_ids'].tolist()), open(f'{out_dir}/protocol.json', 'w'), indent=1)
    fold_rows, perdrug_rows, repro = [], [], {}
    for f in range(5):
        P = np.load(f'{EXT}/{scenario}/fold-{f}/predictions.npz', allow_pickle=False)
        loss, outer, metas, teachers = {}, {}, {}, {}

        def record(bank, grids, meta):
            if bank == 'outer':
                outer.update(grids)
            else:
                truth = c['auc'][np.ix_(meta['query'], meta['panel'])]
                sd = np.maximum(np.nanstd(c['auc'][np.ix_(meta['fit'], meta['panel'])], axis=0), 1e-8)
                for a, g in grids.items():
                    loss.setdefault(a, {})[bank] = bank_losses(g, truth, sd)

        def summed(arm, banks):
            return sum(loss[arm][b][0] for b in banks), sum(loss[arm][b][1] for b in banks)

        for bank in BANKS:
            tb = time.perf_counter()
            grids, meta, teacher = step1_bank(c, scenario, f, bank)
            metas[bank], teachers[bank] = meta, teacher
            record(bank, grids, meta)
            if bank == 'outer':
                q, d = len(meta['query']), len(meta['panel'])
                outer['null_f100'] = np.broadcast_to(meta['yfit_mean'], (1, q, d)).copy()
                outer['null_extra354'] = np.broadcast_to(meta['ytr_mean'], (1, q, d)).copy()
            log(event='bank_step1_done', scenario=scenario, fold=f, bank=bank, extra354=len(meta['e354']),
                pseudo=len(meta['ep']), seconds=round(time.perf_counter() - tb, 1), total=round(time.perf_counter() - t0, 1))
        teacher_sel = {}
        for bank in BANKS:
            tb = time.perf_counter()
            used = INNER if bank == 'outer' else tuple(b for b in INNER if b != bank)
            teacher_sel[bank] = choose(*summed('rna_c_extra', used), fallback=1)
            grids, yp = step2_bank(c, scenario, f, bank, metas[bank], teachers[bank], teacher_sel[bank])
            record(bank, grids, metas[bank])
            if bank == 'outer':
                np.savez_compressed(f'{out_dir}/fold-{f}_teacher_pseudo.npz', pseudo_sample_ids=c['pse_ids'][metas[bank]['ep']],
                                    teacher_labels=yp, teacher_selected=teacher_sel[bank], panel=metas[bank]['panel'])
            log(event='bank_step2_done', scenario=scenario, fold=f, bank=bank, pseudo=len(metas[bank]['ep']),
                seconds=round(time.perf_counter() - tb, 1), total=round(time.perf_counter() - t0, 1))

        # selection on inner banks, single outer evaluation
        truth, sd, use = P['truth'], P['train_sd'], P['evaluated']
        assert (P['panel'] == metas['outer']['panel']).all() and (P['held'] == metas['outer']['query']).all()
        preds, selected = {}, {}
        for a, g in outer.items():
            if a.startswith('null_'):
                preds[a] = g[0]; continue
            sel = choose(*summed(a, INNER), fallback=1)
            selected[a] = sel; preds[a] = take(g, sel)
        rp = dict(m_f100__original__m_only=float(np.max(np.abs(preds['m_f100'] - P['original__m_only']))),
                  m_extra19__extra__m_only=float(np.max(np.abs(preds['m_extra19'] - P['extra__m_only']))),
                  mP_extra19__extra__m_only__protein=float(np.max(np.abs(preds['mP_extra19'] - P['extra__m_only__protein']))))
        repro[f] = rp
        log(event='reproduction', scenario=scenario, fold=f, **rp)
        for a in SAVED:
            preds['saved__' + a] = P[a]
        np.savez_compressed(f'{out_dir}/fold-{f}_predictions.npz', truth=truth, train_sd=sd, evaluated=use, panel=P['panel'],
                            drug_ids=P['drug_ids'], held=P['held'], fit=P['fit'],
                            **{'pred__' + k: v for k, v in preds.items()}, **{'sel__' + k: v for k, v in selected.items()},
                            **{f'teacher_sel__{b}': v for b, v in teacher_sel.items()})
        np.savez_compressed(f'{out_dir}/fold-{f}_inner_losses.npz',
                            **{f'{a}__{b}__loss': v[0] for a, d in loss.items() for b, v in d.items()},
                            **{f'{a}__{b}__count': v[1] for a, d in loss.items() for b, v in d.items()})
        for a, v in preds.items():
            mm = metrics(truth, v, sd, use)
            fold_rows.append(dict(fold=f, arm=a, score=mm['score'], spearman=mm['spearman'], centered_mse=mm['centered_mse'],
                                  drugs=int(use.sum())))
            for j in range(len(sd)):
                perdrug_rows.append(dict(fold=f, arm=a, drug_id=str(P['drug_ids'][j]), evaluated=bool(use[j]),
                                         standardized_mse=float(mm['mse'][j]), spearman=float(mm['rho'][j])))
        log(event='fold_done', scenario=scenario, fold=f, total=round(time.perf_counter() - t0, 1))
    with open(f'{out_dir}/fold_metrics.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(fold_rows[0])); w.writeheader(); w.writerows(fold_rows)
    with open(f'{out_dir}/per_drug_metrics.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(perdrug_rows[0])); w.writeheader(); w.writerows(perdrug_rows)
    arms = list(dict.fromkeys(r['arm'] for r in fold_rows))
    summary = {a: {k: float(np.mean([r[k] for r in fold_rows if r['arm'] == a])) for k in ('score', 'spearman', 'centered_mse')}
               for a in arms}
    folds = {a: [r['score'] for r in fold_rows if r['arm'] == a] for a in arms}
    maxdiff = max(v for d in repro.values() for v in d.values())
    out = dict(scenario=scenario, reproduction=dict(per_fold=repro, max_abs_diff=maxdiff, passed=bool(maxdiff <= 1e-8)),
               summary=summary, fold_scores=folds, seconds=time.perf_counter() - t0)
    json.dump(out, open(f'{out_dir}/summary.json', 'w'), indent=1)
    log(event='done', scenario=scenario, reproduction_max=maxdiff, seconds=round(time.perf_counter() - t0, 1))


if __name__ == '__main__':
    main()

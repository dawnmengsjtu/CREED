"""SETTLE-F Stage A: functional-metric virtual layers on CREED's identical banks (pre-registered in PLAN.md).

Reuses the frozen CREED cross-layer code (code/crosslayer.py, sha recorded in PLAN.md) for data loading, layer
preprocessing, the KRR generator and the masked per-drug KRR head. Only the state representation fed to the head changes.

usage: python stageA.py SCENARIO STAGE OUT_DIR [FOLD]
  STAGE = dryrun : fold-0 inner banks only; reproduction of saved CREED inner losses + gradient finite-difference test
  STAGE = main   : one outer fold (FOLD), all four banks, outer scoring
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('crosslayer', os.path.join(HERE, 'crosslayer.py'))
CL = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(CL)

XL_OUT = CL.ROOT + '/crosslayer-matrix-20261009-0001/output'
RS = (4, 8, 16, 32, 64)
NSEED = 5
YS = ('RNA', 'P')
G0_PAIRS = (('P', 'RNA'), ('RNA', 'P'), ('RNA', 'M'), ('P', 'M'))
MIN_OBS = 20


def log(**kw):
    print(json.dumps(kw), flush=True)


# ---------------------------------------------------------------- measured Z incl. query rows (mirrors layer_inputs)
def measured_z(S, X, fit, query, L):
    m = S['m']; st = CL.kernel_state(m[fit])
    out = {'M': ((m[fit] - st[0]) / st[1], (m[query] - st[0]) / st[1])}
    xf = X[fit]; g = np.argsort(-xf.var(0), kind='stable')[:CL.TOP_GENES]
    st = CL.kernel_state(xf[:, g])
    out['RNA'] = (((xf[:, g] - st[0]) / st[1])[:, :CL.TOP_TARGET], ((X[np.ix_(query, g)] - st[0]) / st[1])[:, :CL.TOP_TARGET])
    p = S['p']; pf_all = p[fit]
    idx = np.flatnonzero(np.isfinite(pf_all).mean(0) >= CL.P_COVER)
    pf, pq = p[np.ix_(fit, idx)], p[np.ix_(query, idx)]
    mean = np.nanmean(pf, axis=0)
    vf, vq = np.where(np.isfinite(pf), pf, mean), np.where(np.isfinite(pq), pq, mean)
    st = CL.kernel_state(vf)
    tsel = np.sort(np.argsort(-vf.var(0), kind='stable')[:CL.TOP_TARGET])
    out['P'] = (((vf - st[0]) / st[1])[:, tsel], ((vq - st[0]) / st[1])[:, tsel])
    for k in ('M', 'RNA', 'P'):
        np.testing.assert_array_equal(out[k][0], L[k][2])
    return out


# ---------------------------------------------------------------- kernels without per-column standardization
def gw_pair(sf, sq):
    """RBF with one global width (median positive squared distance among fit rows); no per-column standardization."""
    d = cdist(sf, sf, 'sqeuclidean'); v = d[np.triu_indices(len(sf), 1)]; v = v[v > 0]
    if not len(v):
        raise ValueError('Degenerate state')
    w = float(np.median(v))
    return np.exp(-d / (2 * w)), np.exp(-cdist(sq, sf, 'sqeuclidean') / (2 * w)), w


# ---------------------------------------------------------------- teacher + EGOP
def teacher_alpha(K, y, sd, folds):
    """Per-target alpha chosen by 3-fold CV inside the fit set (CREED choose rule, fallback index 1)."""
    tot = np.zeros((len(CL.ALPHAS), y.shape[1])); cnt = np.zeros_like(tot)
    n = len(K)
    for held in folds:
        tr = np.setdiff1d(np.arange(n), held)
        ok = np.isfinite(y[tr]).sum(0) > 0
        yy = y[tr][:, ok]
        g = CL.predict_grid(K[np.ix_(tr, tr)], K[np.ix_(held, tr)], yy)
        l, c = CL.bank_losses(g, y[held][:, ok], sd[ok])
        tot[:, ok] += l; cnt[:, ok] += c
    return CL.choose(tot, cnt, fallback=1)


def egop(KB, Z, y, sd, folds, perm=None):
    """Expected gradient outer product of the teacher f_d(base, z) w.r.t. raw z (base fixed), labels in sd units.
    KB = base kernel on fit rows; Z = measured fit-standardized target (n x p). perm permutes label rows (control)."""
    if perm is not None:
        y = y[perm]
    stz = CL.kernel_state(Z)
    Kz = CL.rbf(Z, Z, stz)
    K = .5 * KB + .5 * Kz
    alpha = teacher_alpha(K, y, sd, folds)
    _, sdz, wz = stz
    n, p = Z.shape
    G = np.zeros((p, p)); used = 0
    for d in range(y.shape[1]):
        o = np.flatnonzero(np.isfinite(y[:, d]))
        if len(o) < MIN_OBS:
            continue
        ys = y[o, d] / sd[d]
        beta = cho_solve(cho_factor(K[np.ix_(o, o)] + CL.ALPHAS[alpha[d]] * np.eye(len(o)), lower=True), ys - ys.mean())
        C = .5 * Kz[:, o] * beta[None, :]
        grad = (C @ Z[o] - C.sum(1)[:, None] * Z) / (sdz ** 2 * wz)
        G += grad.T @ grad; used += 1
    G /= (n * max(used, 1))
    ev, U = np.linalg.eigh(G)
    order = np.argsort(-ev); ev, U = np.clip(ev[order], 0, None), U[:, order]
    return dict(G=G, ev=ev, U=U, alpha=alpha, used=used, stz=stz, K=K, Kz=Kz)


def wr(E, r):
    return E['U'][:, :r] * np.sqrt(E['ev'][:r])[None, :]


def fr2(Z, Zhat, G):
    e = Z - Zhat; c = Z - Z.mean(0)
    return float(1 - np.einsum('ij,jk,ik->i', e, G, e).mean() / np.einsum('ij,jk,ik->i', c, G, c).mean())


def r2_mean(Z, Zhat):
    return float((1 - ((Z - Zhat) ** 2).sum(0) / np.maximum(((Z - Z.mean(0)) ** 2).sum(0), 1e-12)).mean())


def rand_orth(p, r, seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((p, r)))
    return q


# ---------------------------------------------------------------- one bank
def run_bank(S, X, scenario, f, bank, fr2_pairs=False):
    G = np.load(f'{CL.EXT}/{scenario}/fold-{f}/{bank}/geometry.npz', allow_pickle=False)
    fit, query, panel = G['fit'], G['query'], G['panel']
    n = len(fit)
    y = S['auc'][np.ix_(fit, panel)]
    sd = np.maximum(np.nanstd(y, axis=0), 1e-8)
    L, info = CL.layer_inputs(S, X, fit, query)
    np.testing.assert_allclose(L['M'][0], G['m_only_kernel'][:n, :n], rtol=1e-11, atol=1e-12)
    Zm = measured_z(S, X, fit, query, L)
    KM, HM = L['M'][0], L['M'][1]
    seed_oof = CL.stable_seed('oof', scenario, f, bank)
    folds = np.array_split(np.random.default_rng(seed_oof).permutation(n), CL.OOF_FOLDS)
    perm = np.random.default_rng(CL.stable_seed('pair_shuffle', scenario, f, bank)).permutation(n)
    grids = {'M': CL.predict_grid(KM, HM, y)}
    diag = {}
    for yl in YS:
        Z, Zq = Zm[yl]
        oof, zq, lam, r2m, _ = CL.generator(KM, HM, Z, seed_oof)
        oofs, zqs, _, _, _ = CL.generator(KM, HM, Z[perm], seed_oof)
        Kv, Hv, _ = CL.rbf_pair(oof, zq)
        grids[f'H086_{yl}'] = CL.predict_grid(.5 * KM + .5 * Kv, .5 * HM + .5 * Hv, y)
        Kt, Ht, _ = CL.rbf_pair(Z, Zq)
        grids[f'T512_{yl}'] = CL.predict_grid(.5 * KM + .5 * Kt, .5 * HM + .5 * Ht, y)
        Ka, Ha, _ = gw_pair(oof, zq)
        grids[f'A2_{yl}'] = CL.predict_grid(.5 * KM + .5 * Ka, .5 * HM + .5 * Ha, y)
        E = egop(KM, Z, y, sd, folds)
        Ep = [egop(KM, Z, y, sd, folds, perm=np.random.default_rng(CL.stable_seed('labperm', scenario, f, bank, yl, s)).permutation(n))
              for s in range(NSEED)]
        _, _, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
        for r in RS:
            W = wr(E, r)
            def head(sf, sq):
                K2, H2, _ = gw_pair(sf, sq)
                return CL.predict_grid(.5 * KM + .5 * K2, .5 * HM + .5 * H2, y)
            grids[f'A1_{yl}_r{r}'] = head(oof @ W, zq @ W)
            grids[f'A6_{yl}_r{r}'] = head(Z @ W, Zq @ W)
            grids[f'A7_{yl}_r{r}'] = head(oofs @ W, zqs @ W)
            V = Vt[:r].T; mu = Z.mean(0)
            grids[f'A3_{yl}_r{r}'] = head((oof - mu) @ V, (zq - mu) @ V)
            for s in range(NSEED):
                Q = rand_orth(Z.shape[1], r, CL.stable_seed('randsub', scenario, f, bank, yl, r, s))
                grids[f'A4_{yl}_r{r}_s{s}'] = head(oof @ Q, zq @ Q)
                Wp = wr(Ep[s], r)
                grids[f'A5_{yl}_r{r}_s{s}'] = head(oof @ Wp, zq @ Wp)
        # split-half stability of the metric subspace (drugs split in two, seeded)
        half = np.random.default_rng(CL.stable_seed('splithalf', scenario, f, bank, yl)).permutation(y.shape[1])
        h1, h2 = np.sort(half[:len(half) // 2]), np.sort(half[len(half) // 2:])
        E1, E2 = egop(KM, Z, y[:, h1], sd[h1], folds), egop(KM, Z, y[:, h2], sd[h2], folds)
        cos2 = {r: float(np.sum((E1['U'][:, :r].T @ E2['U'][:, :r]) ** 2) / r) for r in RS}
        evn = E['ev'] / max(E['ev'].sum(), 1e-300)
        diag[yl] = dict(gen_lambda=lam, r2_mean_oof=r2m, r2_mean_oof_check=r2_mean(Z, oof), fr2_oof=fr2(Z, oof, E['G']),
                        fr2_query=fr2(Zq, zq, E['G']), r2_mean_query=r2_mean(Zq, zq), egop_drugs=E['used'],
                        ev_frac_top=[float(evn[:r].sum()) for r in RS], splithalf_cos2=cos2,
                        fr2_oof_perm=[fr2(Z, oof, e['G']) for e in Ep])
    if fr2_pairs:
        for base, tgt in G0_PAIRS:
            KB, HB = L[base][0], L[base][1]
            Z, Zq = Zm[tgt]
            oof, zq, lam, r2m, _ = CL.generator(KB, HB, Z, seed_oof)
            E = egop(KB, Z, y, sd, folds)
            diag[f'{base}->{tgt}'] = dict(gen_lambda=lam, r2_mean_oof=r2m, fr2_oof=fr2(Z, oof, E['G']),
                                         fr2_query=fr2(Zq, zq, E['G']), r2_mean_query=r2_mean(Zq, zq))
    meta = dict(fit=fit, query=query, panel=panel)
    return grids, meta, dict(n_fit=int(n), n_query=int(len(query)), diag=diag)


# ---------------------------------------------------------------- finite-difference test of the EGOP gradient (T4)
def fd_test(S, X, scenario):
    G = np.load(f'{CL.EXT}/{scenario}/fold-0/inner-0/geometry.npz', allow_pickle=False)
    fit, query, panel = G['fit'], G['query'], G['panel']
    n = len(fit); y = S['auc'][np.ix_(fit, panel)]; sd = np.maximum(np.nanstd(y, axis=0), 1e-8)
    L, _ = CL.layer_inputs(S, X, fit, query)
    Z = L['RNA'][2]; KM = L['M'][0]
    folds = np.array_split(np.random.default_rng(1).permutation(n), 3)
    E = egop(KM, Z, y, sd, folds)
    stz = E['stz']; mean, sdz, wz = stz
    rng = np.random.default_rng(20261011); errs = []
    for _ in range(20):
        d = int(rng.integers(y.shape[1]))
        o = np.flatnonzero(np.isfinite(y[:, d]))
        if len(o) < MIN_OBS:
            continue
        ys = y[o, d] / sd[d]
        beta = cho_solve(cho_factor(E['K'][np.ix_(o, o)] + CL.ALPHAS[E['alpha'][d]] * np.eye(len(o)), lower=True), ys - ys.mean())
        i = int(rng.integers(n)); c = int(rng.integers(Z.shape[1]))
        C = .5 * E['Kz'][i, o] * beta
        g = float((C @ Z[o, c] - C.sum() * Z[i, c]) / (sdz[c] ** 2 * wz))
        def f(zi):
            kz = np.exp(-cdist(((zi - mean) / sdz)[None], (Z[o] - mean) / sdz, 'sqeuclidean')[0] / (2 * wz))
            return float((.5 * KM[i, o] + .5 * kz) @ beta)
        h = 1e-5; zp, zm = Z[i].copy(), Z[i].copy(); zp[c] += h; zm[c] -= h
        num = (f(zp) - f(zm)) / (2 * h)
        errs.append(abs(g - num) / max(abs(num), 1e-8))
    return dict(n=len(errs), max_rel_err=float(max(errs)), median_rel_err=float(np.median(errs)))


# ---------------------------------------------------------------- driver
def main():
    scenario, stage, out = sys.argv[1], sys.argv[2], sys.argv[3]
    assert scenario in ('new_cell', 'tissue') and stage in ('dryrun', 'main')
    os.makedirs(out, exist_ok=False)
    t0 = time.perf_counter()
    S, X, _ = CL.load_ctx()
    json.dump(dict(scenario=scenario, stage=stage, script_sha256=CL.sha(__file__), crosslayer_sha256=CL.sha(os.path.join(HERE, 'crosslayer.py')),
                   rs=RS, nseed=NSEED, ys=YS, g0_pairs=G0_PAIRS), open(f'{out}/protocol.json', 'w'), indent=1)
    if stage == 'dryrun':
        saved = np.load(f'{XL_OUT}/{scenario}/fold-0_inner_losses.npz', allow_pickle=False)
        rep = {}
        for bank in CL.INNER:
            grids, meta, info = run_bank(S, X, scenario, 0, bank, fr2_pairs=(bank == 'inner-0'))
            truth = S['auc'][np.ix_(meta['query'], meta['panel'])]
            sd = np.maximum(np.nanstd(S['auc'][np.ix_(meta['fit'], meta['panel'])], axis=0), 1e-8)
            for mine, theirs in (('M', 'M'), ('H086_RNA', 'M+vRNA'), ('H086_P', 'M+vP')):
                l, _ = CL.bank_losses(grids[mine], truth, sd)
                rep[f'{bank}/{mine}'] = float(np.max(np.abs(l - saved[f'{theirs}__{bank}__loss'])))
            json.dump(info, open(f'{out}/{bank}_info.json', 'w'), indent=1)
            log(event='bank_done', bank=bank, arms=len(grids), seconds=round(time.perf_counter() - t0, 1))
        fd = fd_test(S, X, scenario)
        res = dict(reproduction_max_abs_diff=rep, reproduction_passed=bool(max(rep.values()) <= 1e-10), fd_test=fd,
                   fd_passed=bool(fd['max_rel_err'] < 1e-5), seconds=time.perf_counter() - t0)
        json.dump(res, open(f'{out}/summary.json', 'w'), indent=1)
        log(event='dryrun_done', **res)
        return
    f = int(sys.argv[4])
    loss, infos = {}, {}
    for bank in CL.BANKS:
        grids, meta, info = run_bank(S, X, scenario, f, bank, fr2_pairs=(bank == 'outer'))
        infos[bank] = info
        if bank == 'outer':
            outer, ometa = grids, meta
        else:
            truth = S['auc'][np.ix_(meta['query'], meta['panel'])]
            sd = np.maximum(np.nanstd(S['auc'][np.ix_(meta['fit'], meta['panel'])], axis=0), 1e-8)
            for a, g in grids.items():
                loss.setdefault(a, {})[bank] = CL.bank_losses(g, truth, sd)
        log(event='bank_done', scenario=scenario, fold=f, bank=bank, arms=len(grids), seconds=round(time.perf_counter() - t0, 1))
    arms = list(outer)
    # inner scores (teacher-only r selection uses these for T512 / A6 / M only)
    inner_score = {}
    for a in arms:
        tot = sum(loss[a][b][0] for b in CL.INNER); cnt = sum(loss[a][b][1] for b in CL.INNER)
        sel = CL.choose(tot, cnt, fallback=1)
        avg = np.divide(tot, cnt, out=np.full_like(tot, np.nan), where=cnt > 0)
        v = avg[sel, np.arange(avg.shape[1])]
        inner_score[a] = (sel, v)
    common = np.all([np.isfinite(inner_score[a][1]) for a in arms], axis=0)
    iscore = {a: float(np.sqrt(inner_score[a][1][common].mean())) for a in arms}
    rsel = {}
    for yl in YS:
        full = iscore['M'] - iscore[f'T512_{yl}']
        ok = [r for r in RS if iscore['M'] - iscore[f'A6_{yl}_r{r}'] >= .9 * full]
        rsel[yl] = int(min(ok)) if ok else int(max(RS))
    P = np.load(f'{XL_OUT}/{scenario}/fold-{f}_predictions.npz', allow_pickle=False)
    truth, sd, use = P['truth'], P['train_sd'], P['evaluated']
    assert (P['panel'] == ometa['panel']).all() and (P['held'] == ometa['query']).all() and (P['fit'] == ometa['fit']).all()
    preds = {a: CL.take(outer[a], inner_score[a][0]) for a in arms}
    repro = {mine: float(np.max(np.abs(preds[mine] - P[f'pred__{theirs}'])))
             for mine, theirs in (('M', 'M'), ('H086_RNA', 'M+vRNA'), ('H086_P', 'M+vP'))}
    rows = []
    for a, v in preds.items():
        mm = CL.metrics(truth, v, sd, use)
        rows.append(dict(fold=f, arm=a, score=mm['score'], spearman=mm['spearman'], centered_mse=mm['centered_mse'], inner_score=iscore[a]))
    for a in ('M+RNA', 'M+P'):
        mm = CL.metrics(truth, P[f'pred__{a}'], sd, use)
        rows.append(dict(fold=f, arm='CREED_' + a, score=mm['score'], spearman=mm['spearman'], centered_mse=mm['centered_mse'], inner_score=float('nan')))
    np.savez_compressed(f'{out}/predictions.npz', truth=truth, train_sd=sd, evaluated=use, **{'pred__' + k: v for k, v in preds.items()})
    res = dict(scenario=scenario, fold=f, reproduction=repro, reproduction_passed=bool(max(repro.values()) <= 1e-8),
               r_selected=rsel, inner_scores=iscore, rows=rows, bank_info=infos, seconds=time.perf_counter() - t0)
    json.dump(res, open(f'{out}/fold_result.json', 'w'), indent=1)
    log(event='fold_done', scenario=scenario, fold=f, reproduction=repro, r_selected=rsel, seconds=round(time.perf_counter() - t0, 1))


if __name__ == '__main__':
    main()

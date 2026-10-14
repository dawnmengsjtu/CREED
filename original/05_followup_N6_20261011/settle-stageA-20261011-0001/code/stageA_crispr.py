"""SETTLE-F Stage A, step 2 (PLAN.md section 5): CRISPR gene-effect endpoint.

Reuses the frozen CREED CRISPR endpoint code (code/crispr_endpoint.py) for data loading, layer preprocessing, the KRR
generator, target selection (evaluated_mask on the outer fit) and the masked KRR head. Arms as in stageA.py, plus A9b
(the drug-endpoint metric G_drug of the same bank transferred to the CRISPR head).

usage: python stageA_crispr.py SCENARIO STAGE OUT_DIR [FOLD]
  STAGE = dryrun : fold-0 inner-0 only; reproduction of the CREED CRISPR M / M+vRNA / M+vP inner grids is not saved by
                   that run, so the dryrun checks generator lambda/R2 against its bank_info and an EGOP finite-difference test
  STAGE = main   : one outer fold, all four banks, outer scoring; asserts M / H086 outer predictions == CREED CRISPR run
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
_spec = importlib.util.spec_from_file_location('crispr_endpoint', os.path.join(HERE, 'crispr_endpoint.py'))
CE = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(CE)

CR_OUT = CE.ROOT + '/crispr-endpoint-20261009-0001/output'
RS = (4, 8, 16, 32, 64)
NSEED = 5
YS = ('RNA', 'P')
MIN_OBS = 20


def log(**kw):
    print(json.dumps(kw, default=float), flush=True)


def pg(K, H, y):
    return CE.predict_grid(K, H, y)[0]


def measured_z(S, X, fit, query, L):
    xf = X[fit]; g = np.argsort(-xf.var(0), kind='stable')[:CE.TOP_GENES]
    st = CE.kernel_state(xf[:, g])
    out = {'RNA': (((xf[:, g] - st[0]) / st[1])[:, :CE.TOP_TARGET], ((X[np.ix_(query, g)] - st[0]) / st[1])[:, :CE.TOP_TARGET])}
    p = S['p']; pf_all = p[fit]
    idx = np.flatnonzero(np.isfinite(pf_all).mean(0) >= CE.P_COVER)
    pf, pq = p[np.ix_(fit, idx)], p[np.ix_(query, idx)]
    mean = np.nanmean(pf, axis=0)
    vf, vq = np.where(np.isfinite(pf), pf, mean), np.where(np.isfinite(pq), pq, mean)
    st = CE.kernel_state(vf)
    tsel = np.sort(np.argsort(-vf.var(0), kind='stable')[:CE.TOP_TARGET])
    out['P'] = (((vf - st[0]) / st[1])[:, tsel], ((vq - st[0]) / st[1])[:, tsel])
    for k in YS:
        np.testing.assert_array_equal(out[k][0], L[k][2])
    return out


def gw_pair(sf, sq):
    d = cdist(sf, sf, 'sqeuclidean'); v = d[np.triu_indices(len(sf), 1)]; v = v[v > 0]
    if not len(v):
        raise ValueError('Degenerate state')
    w = float(np.median(v))
    return np.exp(-d / (2 * w)), np.exp(-cdist(sq, sf, 'sqeuclidean') / (2 * w)), w


def teacher_alpha(K, y, sd, folds):
    tot = np.zeros((len(CE.ALPHAS), y.shape[1])); cnt = np.zeros_like(tot)
    n = len(K)
    for held in folds:
        tr = np.setdiff1d(np.arange(n), held)
        g = pg(K[np.ix_(tr, tr)], K[np.ix_(held, tr)], y[tr])
        l, c = CE.bank_losses(g, y[held], sd)
        tot += l; cnt += c
    return CE.choose(tot, cnt, fallback=1)


def egop(KB, Z, y, sd, folds, perm=None):
    """As stageA.egop; targets sharing an observation pattern and alpha are solved together."""
    if perm is not None:
        y = y[perm]
    stz = CE.kernel_state(Z)
    Kz = CE.rbf(Z, Z, stz)
    K = .5 * KB + .5 * Kz
    alpha = teacher_alpha(K, y, sd, folds)
    _, sdz, wz = stz
    n, p = Z.shape
    G = np.zeros((p, p)); used = 0
    groups = {}
    for d in range(y.shape[1]):
        groups.setdefault((np.isfinite(y[:, d]).tobytes(), int(alpha[d])), []).append(d)
    for (pat, ai), cols in groups.items():
        o = np.flatnonzero(np.frombuffer(pat, dtype=bool))
        if len(o) < MIN_OBS:
            continue
        ys = y[np.ix_(o, cols)] / sd[cols]
        B = cho_solve(cho_factor(K[np.ix_(o, o)] + CE.ALPHAS[ai] * np.eye(len(o)), lower=True), ys - ys.mean(0))
        Kh = .5 * Kz[:, o] / wz
        Zo = Z[o] / sdz ** 2; Zi = Z / sdz ** 2
        for k in range(B.shape[1]):
            C = Kh * B[:, k][None, :]
            grad = C @ Zo - C.sum(1)[:, None] * Zi
            G += grad.T @ grad; used += 1
    G /= (n * max(used, 1))
    ev, U = np.linalg.eigh(G)
    order = np.argsort(-ev); ev, U = np.clip(ev[order], 0, None), U[:, order]
    return dict(G=G, ev=ev, U=U, alpha=alpha, used=used, stz=stz, K=K, Kz=Kz)


def wr(E, r):
    return E['U'][:, :r] * np.sqrt(E['ev'][:r])[None, :]


def rand_orth(p, r, seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((p, r)))
    return q


def run_bank(S, X, F, scenario, f, bank, cols, keep_grids):
    G = np.load(f'{CE.EXT}/{scenario}/fold-{f}/{bank}/geometry.npz', allow_pickle=False)
    fit, query, panel = G['fit'], G['query'], G['panel']
    n = len(fit)
    y = F[np.ix_(fit, cols)]
    sd = np.maximum(CE.nanstd0(y), 1e-8)
    yd = S['auc'][np.ix_(fit, panel)]
    sdd = np.maximum(np.nanstd(yd, axis=0), 1e-8)
    L, _ = CE.layer_inputs(S, X, fit, query)
    np.testing.assert_allclose(L['M'][0], G['m_only_kernel'][:n, :n], rtol=1e-11, atol=1e-12)
    Zm = measured_z(S, X, fit, query, L)
    KM, HM = L['M'][0], L['M'][1]
    seed_oof = CE.stable_seed('oof', scenario, f, bank)
    folds = np.array_split(np.random.default_rng(seed_oof).permutation(n), CE.OOF_FOLDS)
    perm = np.random.default_rng(CE.stable_seed('pair_shuffle', scenario, f, bank)).permutation(n)
    truth_b = F[np.ix_(query, cols)]
    out = {}

    def emit(name, K, H):
        g = pg(K, H, y)
        out[name] = g if keep_grids else CE.bank_losses(g, truth_b, sd)

    def head(name, sf, sq):
        K2, H2, _ = gw_pair(sf, sq)
        emit(name, .5 * KM + .5 * K2, .5 * HM + .5 * H2)

    emit('M', KM, HM)
    diag = {}
    for yl in YS:
        Z, Zq = Zm[yl]
        oof, zq, lam, r2m, _ = CE.generator(KM, HM, Z, seed_oof)[:5]
        oofs, zqs = CE.generator(KM, HM, Z[perm], seed_oof)[:2]
        Kv, Hv, _ = CE.rbf_pair(oof, zq)
        emit(f'H086_{yl}', .5 * KM + .5 * Kv, .5 * HM + .5 * Hv)
        Kt, Ht, _ = CE.rbf_pair(Z, Zq)
        emit(f'T512_{yl}', .5 * KM + .5 * Kt, .5 * HM + .5 * Ht)
        head(f'A2_{yl}', oof, zq)
        E = egop(KM, Z, y, sd, folds)
        Ed = egop(KM, Z, yd, sdd, folds)
        Ep = [egop(KM, Z, y, sd, folds, perm=np.random.default_rng(CE.stable_seed('labperm_crispr', scenario, f, bank, yl, s)).permutation(n))
              for s in range(NSEED)]
        _, _, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
        for r in RS:
            W = wr(E, r)
            head(f'A1_{yl}_r{r}', oof @ W, zq @ W)
            head(f'A6_{yl}_r{r}', Z @ W, Zq @ W)
            head(f'A7_{yl}_r{r}', oofs @ W, zqs @ W)
            Wd = wr(Ed, r)
            head(f'A9b_{yl}_r{r}', oof @ Wd, zq @ Wd)
            V = Vt[:r].T; mu = Z.mean(0)
            head(f'A3_{yl}_r{r}', (oof - mu) @ V, (zq - mu) @ V)
            for s in range(NSEED):
                Q = rand_orth(Z.shape[1], r, CE.stable_seed('randsub', scenario, f, bank, yl, r, s))
                head(f'A4_{yl}_r{r}_s{s}', oof @ Q, zq @ Q)
                Wp = wr(Ep[s], r)
                head(f'A5_{yl}_r{r}_s{s}', oof @ Wp, zq @ Wp)
        diag[yl] = dict(gen_lambda=lam, r2_mean_oof=r2m, egop_targets=E['used'], egop_drugs=Ed['used'])
    return out, dict(fit=fit, query=query), dict(n_fit=int(n), n_query=int(len(query)), n_cols=int(len(cols)), diag=diag)


def fd_test(S, X, F, scenario, cols):
    G = np.load(f'{CE.EXT}/{scenario}/fold-0/inner-0/geometry.npz', allow_pickle=False)
    fit, query = G['fit'], G['query']
    n = len(fit); y = F[np.ix_(fit, cols)]; sd = np.maximum(CE.nanstd0(y), 1e-8)
    L, _ = CE.layer_inputs(S, X, fit, query)
    Z = L['RNA'][2]; KM = L['M'][0]
    folds = np.array_split(np.random.default_rng(1).permutation(n), 3)
    E = egop(KM, Z, y[:, :400], sd[:400], folds)
    mean, sdz, wz = E['stz']
    # reference: per-target loop (stageA.py formulation) on the same 400 targets
    Gref = np.zeros_like(E['G']); used = 0
    for d in range(400):
        o = np.flatnonzero(np.isfinite(y[:, d]))
        if len(o) < MIN_OBS:
            continue
        ys = y[o, d] / sd[d]
        beta = cho_solve(cho_factor(E['K'][np.ix_(o, o)] + CE.ALPHAS[E['alpha'][d]] * np.eye(len(o)), lower=True), ys - ys.mean())
        C = .5 * E['Kz'][:, o] * beta[None, :]
        grad = (C @ Z[o] - C.sum(1)[:, None] * Z) / (sdz ** 2 * wz)
        Gref += grad.T @ grad; used += 1
    Gref /= (n * used)
    return dict(grouped_vs_loop_rel=float(np.abs(E['G'] - Gref).max() / np.abs(Gref).max()), used=used)


def main():
    scenario, stage, out = sys.argv[1], sys.argv[2], sys.argv[3]
    assert scenario in ('new_cell', 'tissue') and stage in ('dryrun', 'main')
    os.makedirs(out, exist_ok=False)
    t0 = time.perf_counter()
    S, X, F, gid, crows = CE.load_ctx()
    json.dump(dict(scenario=scenario, stage=stage, script_sha256=CE.sha(__file__), crispr_endpoint_sha256=CE.sha(os.path.join(HERE, 'crispr_endpoint.py')),
                   rs=RS, nseed=NSEED, ys=YS), open(f'{out}/protocol.json', 'w'), indent=1)
    f = 0 if stage == 'dryrun' else int(sys.argv[4])
    Go = np.load(f'{CE.EXT}/{scenario}/fold-{f}/outer/geometry.npz', allow_pickle=False)
    cols = np.flatnonzero(CE.evaluated_mask(F[Go['fit']])[0])
    if stage == 'dryrun':
        fd = fd_test(S, X, F, scenario, cols)
        res = dict(fd=fd, fd_passed=bool(fd['grouped_vs_loop_rel'] < 1e-8), n_cols=int(len(cols)))
        losses, meta, info = run_bank(S, X, F, scenario, 0, 'inner-0', cols, keep_grids=False)
        saved = json.load(open(f'{CE.MATRIX}/{scenario}/bank_info.json'))['fold-0/inner-0']['generators']
        res['gen_check'] = {y: dict(lam=info['diag'][y]['gen_lambda'] - saved[f'M->{y}']['lam'],
                                    r2=info['diag'][y]['r2_mean_oof'] - saved[f'M->{y}']['oof_r2_mean']) for y in YS}
        res['arms'] = len(losses); res['seconds'] = time.perf_counter() - t0
        json.dump(res, open(f'{out}/summary.json', 'w'), indent=1, default=float)
        log(event='dryrun_done', **res)
        return
    loss, infos = {}, {}
    for bank in CE.BANKS:
        res_b, meta, info = run_bank(S, X, F, scenario, f, bank, cols, keep_grids=(bank == 'outer'))
        infos[bank] = info
        if bank == 'outer':
            outer, ometa = res_b, meta
        else:
            for a, v in res_b.items():
                loss.setdefault(a, {})[bank] = v
        log(event='bank_done', scenario=scenario, fold=f, bank=bank, seconds=round(time.perf_counter() - t0, 1))
    arms = list(outer)
    sel, iv = {}, {}
    for a in arms:
        tot = sum(loss[a][b][0] for b in CE.INNER); cnt = sum(loss[a][b][1] for b in CE.INNER)
        sel[a] = CE.choose(tot, cnt, fallback=1)
        avg = np.divide(tot, cnt, out=np.full_like(tot, np.nan), where=cnt > 0)
        iv[a] = avg[sel[a], np.arange(avg.shape[1])]
    common = np.all([np.isfinite(iv[a]) for a in arms], axis=0)
    iscore = {a: float(np.sqrt(iv[a][common].mean())) for a in arms}
    rsel = {}
    for yl in YS:
        full = iscore['M'] - iscore[f'T512_{yl}']
        ok = [r for r in RS if iscore['M'] - iscore[f'A6_{yl}_r{r}'] >= .9 * full]
        rsel[yl] = int(min(ok)) if ok else int(max(RS))
    P = np.load(f'{CR_OUT}/{scenario}/fold-{f}_predictions.npz', allow_pickle=False)
    truth, sd, use = P['truth'], P['train_sd'], P['evaluated']
    preds = {a: CE.take(outer[a], sel[a]) for a in arms}
    assert preds['M'].shape == truth.shape
    repro = {mine: float(np.max(np.abs(preds[mine] - P[f'pred__{theirs}'])))
             for mine, theirs in (('M', 'M'), ('H086_RNA', 'M+vRNA'), ('H086_P', 'M+vP'))}
    rows = []
    for a, v in preds.items():
        mm = CE.metrics(truth, v, sd, use)
        rows.append(dict(fold=f, arm=a, score=mm['score'], spearman=mm['spearman'], centered_mse=mm['centered_mse'], inner_score=iscore[a]))
    for a in ('M+RNA', 'M+P'):
        mm = CE.metrics(truth, P[f'pred__{a}'], sd, use)
        rows.append(dict(fold=f, arm='CREED_' + a, score=mm['score'], spearman=mm['spearman'], centered_mse=mm['centered_mse'], inner_score=float('nan')))
    res = dict(scenario=scenario, fold=f, endpoint='crispr', reproduction=repro, reproduction_passed=bool(max(repro.values()) <= 1e-8),
               r_selected=rsel, inner_scores=iscore, rows=rows, bank_info=infos, seconds=time.perf_counter() - t0)
    json.dump(res, open(f'{out}/fold_result.json', 'w'), indent=1, default=float)
    log(event='fold_done', scenario=scenario, fold=f, reproduction=repro, r_selected=rsel, seconds=round(time.perf_counter() - t0, 1))


if __name__ == '__main__':
    main()

"""Probe 2, part B: abstention by M-derived uncertainty on the saved virtual-layer arms (pre-registered in PLAN.md).

Analysis-only on crosslayer-matrix-20261009-0001 saved outer predictions. u1 refits the matrix run's M->Y generators
inside fit with the matrix code's recipe (same OOF split seed, same lambda rule) and uses the 3 cross-fitting fold models
as sub-generators. Reads only allowed inputs; writes only to OUT_DIR.

usage: python abstain.py SCENARIO OUT_DIR
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

RUN = '/public/home/mengxl/creed-research-runs'
EXT = RUN + '/extra-supervision-20261008-0001/output'
SOURCE = RUN + '/gdsc-metabolite-full-protein-20260908-0001/output/result/source-inputs.npz'
RNA421 = RUN + '/rna421-20260906-0002'
MX = RUN + '/crosslayer-matrix-20261009-0001/output'
ARMS = ('M', 'M+vRNA', 'M+vP', 'M+RNA', 'M+P')
LAMBDAS = (.1, 1., 10., 100.)
TOP_GENES, TOP_TARGET, P_COVER, OOF_FOLDS = 3000, 512, .5, 3
COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.5)
RSEEDS = (0, 1, 2, 3, 4)
KNN = 10
SCEN_ID = {'new_cell': 0, 'tissue': 1}


def log(**kw):
    print(json.dumps(kw), flush=True)


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'little')


def kernel_state(x):
    mean = x.mean(0); sd = x.std(0); sd = np.where(sd < 1e-8, 1., sd); z = (x - mean) / sd
    d = cdist(z, z, 'sqeuclidean'); v = d[np.triu_indices(len(z), 1)]; v = v[v > 0]
    return mean, sd, float(np.median(v))


def rbf(a, b, state):
    mean, sd, width = state
    return np.exp(-cdist((a - mean) / sd, (b - mean) / sd, 'sqeuclidean') / (2 * width))


def targets(S, X, fit):
    """Z_RNA, Z_P exactly as crosslayer.layer_inputs (fit-only)."""
    xf = X[fit]
    g = np.argsort(-xf.var(0), kind='stable')[:TOP_GENES]
    st = kernel_state(xf[:, g])
    z_rna = ((xf[:, g] - st[0]) / st[1])[:, :TOP_TARGET]
    p = S['p']; pf_all = p[fit]
    idx = np.flatnonzero(np.isfinite(pf_all).mean(0) >= P_COVER)
    pf = p[np.ix_(fit, idx)]; mean = np.nanmean(pf, axis=0)
    vf = np.where(np.isfinite(pf), pf, mean)
    st = kernel_state(vf)
    tsel = np.sort(np.argsort(-vf.var(0), kind='stable')[:TOP_TARGET])
    z_p = ((vf - st[0]) / st[1])[:, tsel]
    return {'RNA': z_rna, 'P': z_p}, dict(p_candidates=int(len(idx)), rna_top_gene_index=int(g[0]))


def generator_sub(KX, HX, Z, seed):
    """Same OOF split / lambda rule as crosslayer.generator; additionally returns the 3 fold-models' query predictions."""
    n = len(KX)
    folds = np.array_split(np.random.default_rng(seed).permutation(n), OOF_FOLDS)
    best = None
    for lam in LAMBDAS:
        oof = np.empty_like(Z); subq = []
        for held in folds:
            tr = np.setdiff1d(np.arange(n), held)
            mu = Z[tr].mean(0)
            coef = cho_solve(cho_factor(KX[np.ix_(tr, tr)] + lam * np.eye(len(tr)), lower=True), Z[tr] - mu)
            oof[held] = KX[np.ix_(held, tr)] @ coef + mu
            subq.append(HX[:, tr] @ coef + mu)
        mse = float(((oof - Z) ** 2).mean())
        if best is None or mse < best[0]:
            best = (mse, lam, oof, np.stack(subq))
    _, lam, oof, subq = best
    r2 = 1 - ((oof - Z) ** 2).sum(0) / np.maximum(((Z - Z.mean(0)) ** 2).sum(0), 1e-12)
    return lam, float(r2.mean()), subq


def retained_score(E, obs, use, keep):
    """E: (nq, nd) squared standardized errors (0 where unobserved); keep: bool mask of retained cells."""
    num = E[keep].sum(0); den = obs[keep].sum(0)
    ok = use & (den > 0)
    return float(np.sqrt((num[ok] / den[ok]).mean()))


def curve(E, obs, use, order):
    """order: cells sorted most-uncertain first. Returns dict coverage->score and AURC over [0.5, 1]."""
    n = len(order); out = {}
    for c in COVERAGES:
        nk = int(np.ceil(c * n)); keep = np.zeros(n, bool); keep[order[n - nk:]] = True
        out[c] = retained_score(E, obs, use, keep)
    nks = np.arange(int(np.ceil(.5 * n)), n + 1)
    xs, ys = [], []
    for nk in nks:
        keep = np.zeros(n, bool); keep[order[n - nk:]] = True
        xs.append(nk / n); ys.append(retained_score(E, obs, use, keep))
    xs, ys = np.array(xs), np.array(ys)
    aurc = float(np.trapz(ys, xs) / (xs[-1] - xs[0]))
    return out, aurc


def main():
    scenario, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    t0 = time.perf_counter()
    S = np.load(SOURCE, allow_pickle=False)
    S = {k: S[k] for k in ('m', 'auc', 'p', 'tissues', 'sample_ids')}
    X = np.load(RNA421 + '/rna_log_cpm.npy')
    assert (np.load(RNA421 + '/sample_ids.npy') == S['sample_ids']).all()
    binfo = json.load(open(f'{MX}/{scenario}/bank_info.json'))
    rows, arows, prows, gen_check = [], [], [], {}
    for f in range(5):
        Z = np.load(f'{MX}/{scenario}/fold-{f}_predictions.npz', allow_pickle=False)
        G = np.load(f'{EXT}/{scenario}/fold-{f}/outer/geometry.npz', allow_pickle=False)
        fit, held = Z['fit'], Z['held']; n, nq = len(fit), len(held)
        assert np.array_equal(G['fit'], fit) and np.array_equal(G['query'], held)
        truth, sd, use = Z['truth'], Z['train_sd'], Z['evaluated']
        obs = np.isfinite(truth)
        KM, HM = G['m_only_kernel'][:n, :n], G['m_only_cross'][:, :n]
        # proxies
        T, tinfo = targets(S, X, fit)
        seed = stable_seed('oof', scenario, f, 'outer')
        u = {}
        for layer in ('RNA', 'P'):
            lam, r2, subq = generator_sub(KM, HM, T[layer], seed)
            ref = binfo[f'fold-{f}/outer']['generators'][f'M->{layer}']
            gen_check[f'fold-{f}/M->{layer}'] = dict(lam=lam, lam_saved=ref['lam'], r2=r2, r2_saved=ref['oof_r2_mean'],
                                                      lam_identical=bool(lam == ref['lam']), r2_abs_diff=abs(r2 - ref['oof_r2_mean']))
            u[f'u1_v{layer}'] = subq.std(0, ddof=0).mean(1)
        u['u2'] = 1 - HM.max(1)
        mu = S['m'][fit].mean(0); msd = S['m'][fit].std(0); msd = np.where(msd < 1e-8, 1., msd)
        D = cdist((S['m'][held] - mu) / msd, (S['m'][fit] - mu) / msd)
        u['u3'] = np.sort(D, axis=1)[:, :KNN].mean(1)
        errs = {}
        for arm in ARMS:
            E = np.where(obs, ((Z['pred__' + arm] - truth) / sd) ** 2, 0.)
            eval_obs = obs & use[None, :]
            errs[arm] = (E, np.where(eval_obs, E, 0.).sum(1) / np.maximum(eval_obs.sum(1), 1))
        for i in range(nq):
            prows.append(dict(scenario=scenario, fold=f, cell=int(held[i]), sample_id=str(S['sample_ids'][held[i]]), tissue=str(S['tissues'][held[i]]),
                              **{k: float(v[i]) for k, v in u.items()}, **{'e_' + a: float(errs[a][1][i]) for a in ARMS}))
        orders = {k: np.argsort(-v, kind='stable') for k, v in u.items()}
        for s in RSEEDS:
            orders[f'random_s{s}'] = np.random.default_rng([20261009, SCEN_ID[scenario], f, 5, s]).permutation(nq)
        for arm in ARMS:
            E, e_cell = errs[arm]
            ords = dict(orders); ords['oracle'] = np.argsort(-e_cell, kind='stable')
            for proxy, order in ords.items():
                cov, aurc = curve(E, obs, use, order)
                for c, sc in cov.items():
                    rows.append(dict(scenario=scenario, arm=arm, proxy=proxy, coverage=c, fold=f, score=sc))
                arows.append(dict(scenario=scenario, arm=arm, proxy=proxy, fold=f, aurc=aurc, n_query=nq, n_fit=n))
        log(event='fold_done', scenario=scenario, fold=f, gen=gen_check[f'fold-{f}/M->RNA'], p_candidates=tinfo['p_candidates'],
            seconds=round(time.perf_counter() - t0, 1))
    for name, rr in (('riskcov', rows), ('aurc', arows), ('percell', prows)):
        with open(f'{out}/{name}_{scenario}.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)
    json.dump(dict(scenario=scenario, generator_check=gen_check, seconds=time.perf_counter() - t0), open(f'{out}/gen_check_{scenario}.json', 'w'), indent=1)
    log(event='done', scenario=scenario, all_lambda_identical=all(v['lam_identical'] for v in gen_check.values()),
        max_r2_diff=max(v['r2_abs_diff'] for v in gen_check.values()), seconds=round(time.perf_counter() - t0, 1))


if __name__ == '__main__':
    main()

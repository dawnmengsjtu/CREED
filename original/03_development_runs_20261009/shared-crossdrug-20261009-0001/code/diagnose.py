"""POST-HOC descriptive diagnostic (written AFTER outer scores existed; not pre-registered; selects nothing).

Question: how much of the standardized drug response is a shared "general sensitivity" factor, how well can the
deployable kernel predict that factor for held-out cells, and does the per-drug baseline already carry it implicitly?
Usage: python diagnose.py OUT_DIR
"""
import json
import os
import sys
import numpy as np
import shared_crossdrug as sc

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=False)
S = np.load(sc.SOURCE, allow_pickle=False); E = np.load(sc.EXTRA, allow_pickle=False)
auc = S['auc']
RUN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = []
for scen, kern, base_arm, shared_arm in (('new_cell', 'm_only', 'base_m', 'shared_m'), ('tissue', 'virtual_fm', 'base_fm', 'shared_fm')):
    for f in range(5):
        G = np.load(f'{sc.RUN}/output/{scen}/fold-{f}/outer/geometry.npz', allow_pickle=False)
        fit, query, panel, ei = G['fit'], G['query'], G['panel'], G['extra_indices']
        y = np.vstack([auc[np.ix_(fit, panel)], E['auc'][np.ix_(ei, panel)]])
        st = sc.label_state(y, False)
        mu, sd, elig = st['mu'], st['sd'], st['elig']
        s = st['fam']['mean']['F'][:, 0]; fok = st['fok']
        Z = st['z']
        # share of fit variance explained by the mean factor (per-drug R^2, averaged over eligible drugs)
        r2_fit = []
        for d in np.flatnonzero(elig):
            o = np.isfinite(Z[:, d]) & fok
            r2_fit.append(np.corrcoef(Z[o, d], s[o])[0, 1] ** 2)
        P = np.load(f'{RUN_DIR}/output_{scen}/fold-{f}_predictions.npz', allow_pickle=False)
        truth = P['truth']; assert np.array_equal(P['held'], query)
        Zq = (truth - mu) / sd
        Zqe = np.where(np.isfinite(Zq[:, elig]), Zq[:, elig], np.nan)
        okq = np.isfinite(Zqe).sum(1) > 0
        sq = np.full(len(query), np.nan); sq[okq] = np.nanmean(Zqe[okq], axis=1)
        r2_oracle = []
        for d in np.flatnonzero(elig & P['evaluated']):
            o = np.isfinite(Zq[:, d]) & okq
            if o.sum() > 2 and np.std(Zq[o, d]) > 0:
                r2_oracle.append(np.corrcoef(Zq[o, d], sq[o])[0, 1] ** 2)
        k, h = G[f'{kern}_kernel'], G[f'{kern}_cross']
        fh = sc.factor_krr(k, h, s[:, None], fok)[:, :, 0]
        r_hat = {str(a): float(np.corrcoef(fh[i][okq], sq[okq])[0, 1]) for i, a in enumerate(sc.ALPHAS)}
        sd_ratio = {str(a): float(np.std(fh[i][okq]) / np.std(sq[okq])) for i, a in enumerate(sc.ALPHAS)}
        implied = {}
        for arm in (base_arm, shared_arm):
            zp = (P[arm] - mu) / sd
            imp = zp[:, elig].mean(1)
            implied[arm] = dict(r=float(np.corrcoef(imp[okq], sq[okq])[0, 1]), sd_ratio=float(np.std(imp[okq]) / np.std(sq[okq])))
        rows.append(dict(scenario=scen, fold=f, kernel=kern, fit_r2_mean_factor=float(np.mean(r2_fit)),
                         query_oracle_r2_mean_factor=float(np.mean(r2_oracle)),
                         factor_pred_r=r_hat, factor_pred_sd_ratio=sd_ratio, implied_factor=implied))
        print(json.dumps(rows[-1]), flush=True)
json.dump(rows, open(f'{OUT}/diagnostic.json', 'w'), indent=1)

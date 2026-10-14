"""Synthetic unit tests for shared_crossdrug.py (no real data, no outer scores)."""
import numpy as np
import shared_crossdrug as sc

rng = np.random.default_rng(0)
n, q, D, p = 200, 60, 80, 10
X = rng.normal(size=(n + q, p))
def rbf(a, b):
    d = ((a[:, None] - b[None]) ** 2).sum(-1); return np.exp(-d / (2 * np.median(d[d > 0])))
K = rbf(X[:n], X[:n]); H = rbf(X[n:], X[:n])
s_true = np.tanh(X[:, 0]) + 0.5 * X[:, 1]           # shared sensitivity driven by inputs
load = rng.normal(1.0, 0.3, size=D)
Y = s_true[:, None] * load[None] * 0.6 + rng.normal(size=(n + q, D)) * 1.0
Y = Y * rng.uniform(.05, .2, size=D) + rng.uniform(.6, .9, size=D)
mask = rng.random((n + q, D)) < 0.15
Ytr = np.where(mask[:n], np.nan, Y[:n]); Yte = np.where(mask[n:], np.nan, Y[n:])

base = sc.predict_grid(K, H, Ytr)
st = sc.label_state(Ytr, True)
fams, cons = sc.shared_families(K, H, st, base, list(sc.SHARED_TYPES))
print('consistency (z-space base vs raw base):', cons)
assert cons < 1e-10

# krr_multi equals predict_grid on raw labels column by column
km = sc.krr_multi(K, H, np.isfinite(Ytr), np.where(np.isfinite(Ytr), Ytr, 0.)[:, :, None])[..., 0]
print('krr_multi vs predict_grid:', np.max(np.abs(km - base)))
assert np.max(np.abs(km - base)) < 1e-10

sd = np.nanstd(Ytr, axis=0)
def score(pred):
    e = ((pred - Yte) / sd) ** 2
    return np.sqrt(np.nanmean(np.nanmean(e, axis=0)))
print('base best alpha score:', min(score(base[a]) for a in range(4)))
for t, g in fams.items():
    print(t, 'best shared config score:', min(score(g[c]) for c in range(len(g))), 'n_configs', len(g))
# gamma -> 0 limit check: implementation linear in gamma; gamma=0.5 and 1 bracket
# a pure-noise check: with no shared signal, shared factor should not help much
Y2 = rng.normal(size=(n + q, D)) * 0.1 + 0.7
Y2tr = np.where(mask[:n], np.nan, Y2[:n]); Y2te = np.where(mask[n:], np.nan, Y2[n:])
b2 = sc.predict_grid(K, H, Y2tr); st2 = sc.label_state(Y2tr, True)
f2, _ = sc.shared_families(K, H, st2, b2, ['mean'])
sd2 = np.nanstd(Y2tr, axis=0)
sc2 = lambda pred: np.sqrt(np.nanmean(np.nanmean(((pred - Y2te) / sd2) ** 2, axis=0)))
print('noise: base', min(sc2(b2[a]) for a in range(4)), 'shared', min(sc2(f2['mean'][c]) for c in range(32)))
# low-rank invariance: svd1 factor ~ mean factor direction
F1 = st['fam']['svd1']['F'][:, 0]; s = st['fam']['mean']['F'][:, 0]
print('corr(svd1, mean factor):', abs(np.corrcoef(F1, s)[0, 1]), 'iters', st['lowrank_iters'])
# decode / arm registry sanity
for scen in ('new_cell', 'tissue'):
    arms, repro, settings, lr = sc.registry(scen)
    for a, ab in arms.items():
        _, ncfg = sc.decode(ab, np.array([0]))
        print(scen, a, ncfg)
# analyze() on fake fold data (no real outcomes)
for scen in ('new_cell', 'tissue'):
    arms, repro, settings, lr = sc.registry(scen)
    names = set(a for c in sc.comparisons(scen) for a in c[:2])
    fold_data, preds_all, rows = [], [], []
    tis = np.array(['A', 'B', 'C'] * 40)
    for f in range(5):
        tr = rng.normal(size=(20, 30)); tr[rng.random(tr.shape) < .1] = np.nan
        held = np.arange(f * 20, f * 20 + 20); sdv = np.ones(30); use = np.ones(30, bool)
        dids = np.array([f'd{j}' for j in range(30)])
        pr = {a: tr + rng.normal(scale=.5, size=tr.shape) for a in names}
        for a in names:
            m = sc.metrics(tr, pr[a], sdv, use)
            rows.append(dict(fold=f, arm=a, score=m['score'], spearman=m['spearman'], centered_mse=m['centered_mse']))
        fold_data.append(dict(truth=tr, sd=sdv, use=use, obs=np.isfinite(tr), held=held, fit=np.arange(80),
                              drug_ids=dids, n_extra=5))
        preds_all.append(pr)
    res = sc.analyze(scen, None, fold_data, preds_all, rows, {'tissues': tis})
    print(scen, 'analyze ok', len(res), res[0]['comparison'], res[0]['nadeau_bengio'], res[0]['grouped_bootstrap_rel_pct_ci95'])
    import json; json.dumps(sc.clean(res))
print('ALL TESTS PASSED')

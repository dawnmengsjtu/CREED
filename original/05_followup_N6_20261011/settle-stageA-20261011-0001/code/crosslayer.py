"""Cross-layer HEADROOM x RECOVERABILITY matrix on identical folds (pre-registered in PLAN.md).

Independent implementation (no project imports). predict_grid / bank_losses / choose / take / metrics / kernel_state / rbf
follow the local reference implementations (tissue_kernel_20261009/tissue_kernel.py, rna_learning_curve_20261009/rna_lc.py),
which were themselves verified against the extra-supervision run. Reads only allowed inputs; writes only to OUT_DIR
(must not exist).

usage: python crosslayer.py SCENARIO STAGE OUT_DIR
  STAGE = dryrun : fold-0 inner banks only (no outer bank, no outer score); inner-grid reproduction vs saved grids
  STAGE = main   : all folds, all banks, outer scoring
env (dev smoke test only): CREED_ROOT overrides the run root; CREED_SKIP_SHA=1 skips the input sha checks.
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
from scipy.stats import spearmanr

ROOT = os.environ.get('CREED_ROOT', '/public/home/mengxl/creed-research-runs')
EXT = ROOT + '/extra-supervision-20261008-0001/output'
EXTRA = ROOT + '/extra-supervision-20261008-0001/resources/m_gdsc_extra19.npz'
SOURCE = ROOT + '/gdsc-metabolite-full-protein-20260908-0001/output/result/source-inputs.npz'
RNA421 = ROOT + '/rna421-20260906-0002'
SOURCE_SHA = '52d71517a8952e83a1603cd831f09449544102d0ca5ff829ff01be9bf79cd637'
EXTRA_SHA = 'd4e7dc603f491217805a1d06b673f8c2662a8c0f2ad77ec875f7d7868526384c'

ALPHAS = (.1, 1., 10., 100.)
LAMBDAS = (.1, 1., 10., 100.)
BANKS = ('inner-0', 'inner-1', 'inner-2', 'outer')
INNER = BANKS[:3]
TOP_GENES, TOP_TARGET, P_COVER, OOF_FOLDS = 3000, 512, .5, 3
LAYERS = ('M', 'RNA', 'P')
PAIRS = (('M', 'RNA'), ('M', 'P'), ('RNA', 'P'), ('RNA', 'M'), ('P', 'M'), ('P', 'RNA'))
MEASURED = {'M+RNA': ('M', 'RNA'), 'M+P': ('M', 'P'), 'RNA+P': ('RNA', 'P'), 'M+RNA+P': ('M', 'RNA', 'P'),
            'Pval': ('Pval',), 'M+Pval': ('M', 'Pval'), 'RNA+Pval': ('RNA', 'Pval')}
MEASURED_T = {'T': ('T',), 'M+T': ('M', 'T'), 'RNA+T': ('RNA', 'T'), 'P+T': ('P', 'T'), 'M+RNA+P+T': ('M', 'RNA', 'P', 'T')}
REPRO = 'repro_M_extra19'


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


# ---------------------------------------------------------------- kernels
def kernel_state(x):
    """Fit moments + median positive squared distance (identical to run_extra.py / reference)."""
    mean = x.mean(0); sd = x.std(0); sd = np.where(sd < 1e-8, 1., sd); z = (x - mean) / sd
    d = cdist(z, z, 'sqeuclidean'); v = d[np.triu_indices(len(z), 1)]; v = v[v > 0]
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
    """RBF on raw 0/1 indicator vectors (no standardization); width = median positive pairwise Hamming distance."""
    d = cdist(bf, bf, 'sqeuclidean'); v = d[np.triu_indices(len(bf), 1)]; v = v[v > 0]
    if not len(v):
        return np.ones_like(d), np.ones((len(bq), len(bf))), float('nan')
    width = float(np.median(v))
    return np.exp(-d / (2 * width)), np.exp(-cdist(bq, bf, 'sqeuclidean') / (2 * width)), width


def same(a, b):
    return (a[:, None] == b[None, :]).astype(float)


def tokens(t, tag):
    return np.array([x if x else f'__unknown_{tag}_{i}' for i, x in enumerate(t)])


# ---------------------------------------------------------------- layers (fit-only preprocessing)
def layer_inputs(S, X, fit, query):
    """Returns {layer: (K_fit, H_query, Z_target or None)} and an info dict. Z = standardized fit features (<=512)."""
    L, info = {}, {}
    # M: all 225 metabolites
    m = S['m']
    K, H, st = rbf_pair(m[fit], m[query])
    L['M'] = (K, H, (m[fit] - st[0]) / st[1])
    info['M'] = dict(features=int(m.shape[1]), width=st[2], target=int(m.shape[1]))
    # RNA: top-3000 fit-variance genes (stable order), fit z-score, RBF median width
    xf = X[fit]
    g = np.argsort(-xf.var(0), kind='stable')[:TOP_GENES]
    K, H, st = rbf_pair(xf[:, g], X[np.ix_(query, g)])
    zf = (xf[:, g] - st[0]) / st[1]
    L['RNA'] = (K, H, zf[:, :TOP_TARGET])          # g is variance-descending, so the first 512 are the top-512
    info['RNA'] = dict(features=int(len(g)), width=st[2], target=int(min(TOP_TARGET, len(g))), top_gene_index=int(g[0]))
    # P: proteins observed in >=50% of fit cells; values (fit-mean imputed, fit z-scored, RBF) and mask (binary RBF)
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
    return L, info


def generator(KX, HX, Z, seed):
    """Cross-fitted KRR X->Z inside fit. OOF predictions for fit rows (3 folds), full-fit generator for query rows.
    lambda chosen inside fit by mean OOF MSE over target columns. Returns (Z_oof_fit, Z_query, lambda, r2_mean, r2_median)."""
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


def arm_list(scenario):
    arms = list(LAYERS) + list(MEASURED)
    if scenario == 'new_cell':
        arms += list(MEASURED_T)
    arms += [f'{x}+v{y}' for x, y in PAIRS] + [f'{x}+v{y}_shuf' for x, y in PAIRS]
    arms.append(REPRO)
    return arms


def average(L, parts):
    K = sum(L[p][0] for p in parts) / len(parts); H = sum(L[p][1] for p in parts) / len(parts)
    return K, H


# ---------------------------------------------------------------- one bank
def run_bank(S, X, E, scenario, f, bank):
    G = np.load(f'{EXT}/{scenario}/fold-{f}/{bank}/geometry.npz', allow_pickle=False)
    fit, query, panel, ei = G['fit'], G['query'], G['panel'], G['extra_indices']
    n = len(fit)
    auc, tis = S['auc'], S['tissues']
    y = auc[np.ix_(fit, panel)]
    L, info = layer_inputs(S, X, fit, query)
    # kernel block check against the saved original-data M kernel (reproduction gate 1)
    kdiff = max(float(np.max(np.abs(L['M'][0] - G['m_only_kernel'][:n, :n]))),
                float(np.max(np.abs(L['M'][1] - G['m_only_cross'][:, :n]))))
    np.testing.assert_allclose(L['M'][0], G['m_only_kernel'][:n, :n], rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(L['M'][1], G['m_only_cross'][:, :n], rtol=1e-11, atol=1e-12)
    if scenario == 'new_cell':
        ttr, tq = tokens(tis[fit], 'tr'), tokens(tis[query], 'q')
        L['T'] = (same(ttr, ttr), same(tq, ttr), None)
    grids = {}
    for layer in LAYERS:
        grids[layer] = predict_grid(L[layer][0], L[layer][1], y)
    for name, parts in MEASURED.items():
        grids[name] = predict_grid(*average(L, parts), y)
    if scenario == 'new_cell':
        for name, parts in MEASURED_T.items():
            grids[name] = predict_grid(*average(L, parts), y)
    # virtual layers + pairing-shuffled controls
    seed_oof = stable_seed('oof', scenario, f, bank)
    perm = np.random.default_rng(stable_seed('pair_shuffle', scenario, f, bank)).permutation(n)
    assert not np.array_equal(perm, np.arange(n))
    gen_info = {}
    for x, yl in PAIRS:
        KX, HX, _ = L[x]; Z = L[yl][2]
        for tag, target in (('', Z), ('_shuf', Z[perm])):
            oof, zq, lam, r2m, r2med = generator(KX, HX, target, seed_oof)
            Kv, Hv, st = rbf_pair(oof, zq)
            grids[f'{x}+v{yl}{tag}'] = predict_grid(.5 * KX + .5 * Kv, .5 * HX + .5 * Hv, y)
            gen_info[f'{x}->{yl}{tag}'] = dict(lam=lam, oof_r2_mean=r2m, oof_r2_median=r2med, virtual_width=st[2],
                                               n_targets=int(Z.shape[1]))
    # reproduction arm: saved extended M kernel (fit + extra 19) with extra labels (not part of any matrix)
    ye = np.vstack([y, E['auc'][np.ix_(ei, panel)]])
    assert G['m_only_kernel'].shape == (n + len(ei), n + len(ei))
    grids[REPRO] = predict_grid(G['m_only_kernel'], G['m_only_cross'], ye)
    info.update(n_fit=int(n), n_query=int(len(query)), n_panel=int(len(panel)), n_extra_saved=int(len(ei)),
                kernel_block_max_abs_diff=kdiff, generators=gen_info)
    meta = dict(fit=fit, query=query, panel=panel)
    return grids, meta, info


# ---------------------------------------------------------------- data
def load_ctx():
    if os.environ.get('CREED_SKIP_SHA') != '1':
        assert sha(SOURCE) == SOURCE_SHA, 'source sha mismatch'
        assert sha(EXTRA) == EXTRA_SHA, 'extra19 sha mismatch'
    S = np.load(SOURCE, allow_pickle=False)
    S = {k: S[k] for k in ('m', 'auc', 'p', 'tissues', 'sample_ids', 'drug_ids', 'protein_ids', 'metabolite_ids')}
    X = np.load(RNA421 + '/rna_log_cpm.npy')
    assert (np.load(RNA421 + '/sample_ids.npy') == S['sample_ids']).all(), 'RNA row order differs from source'
    assert X.shape[0] == len(S['sample_ids']) and np.isfinite(X).all() and np.isfinite(S['m']).all()
    E = np.load(EXTRA, allow_pickle=False)
    assert (E['drug_ids'] == S['drug_ids']).all()
    E = {'auc': E['auc']}
    return S, X, E


def main():
    scenario, stage, out = sys.argv[1], sys.argv[2], sys.argv[3]
    assert scenario in ('new_cell', 'tissue') and stage in ('dryrun', 'main')
    os.makedirs(out, exist_ok=False)
    t0 = time.perf_counter()
    S, X, E = load_ctx()
    arms = arm_list(scenario)
    json.dump(dict(scenario=scenario, stage=stage, script_sha256=sha(__file__), source_sha256=SOURCE_SHA, extra_sha256=EXTRA_SHA,
                   alphas=ALPHAS, lambdas=LAMBDAS, top_genes=TOP_GENES, top_target=TOP_TARGET, p_cover=P_COVER, oof_folds=OOF_FOLDS,
                   pairs=PAIRS, arms=arms, rna_shape=list(X.shape)), open(f'{out}/protocol.json', 'w'), indent=1)
    repro = dict(kernel_block_max_abs_diff=0., grid_M_vs_original__m_only=0., grid_repro_vs_extra__m_only=0.,
                 pred_M_vs_original__m_only=0., pred_repro_vs_extra__m_only=0.)
    fold_rows, perdrug_rows, bank_info, sel_counts = [], [], {}, {}
    for f in ((0,) if stage == 'dryrun' else range(5)):
        loss, outer, ometa = {}, None, None
        for bank in BANKS:
            if stage == 'dryrun' and bank == 'outer':
                continue
            tb = time.perf_counter()
            grids, meta, info = run_bank(S, X, E, scenario, f, bank)
            saved = np.load(f'{EXT}/{scenario}/fold-{f}/{bank}/grids.npz', allow_pickle=False)
            d1 = float(np.max(np.abs(grids['M'] - saved['original__m_only'])))
            d2 = float(np.max(np.abs(grids[REPRO] - saved['extra__m_only'])))
            repro['grid_M_vs_original__m_only'] = max(repro['grid_M_vs_original__m_only'], d1)
            repro['grid_repro_vs_extra__m_only'] = max(repro['grid_repro_vs_extra__m_only'], d2)
            repro['kernel_block_max_abs_diff'] = max(repro['kernel_block_max_abs_diff'], info['kernel_block_max_abs_diff'])
            if bank == 'outer':
                outer, ometa = grids, meta
            else:
                truth = S['auc'][np.ix_(meta['query'], meta['panel'])]
                sd = np.maximum(np.nanstd(S['auc'][np.ix_(meta['fit'], meta['panel'])], axis=0), 1e-8)
                for a, g in grids.items():
                    loss.setdefault(a, {})[bank] = bank_losses(g, truth, sd)
            bank_info[f'fold-{f}/{bank}'] = info
            log(event='bank_done', scenario=scenario, fold=f, bank=bank, grid_diff_M=d1, grid_diff_extra=d2,
                p_candidates=info['P']['candidates'], seconds=round(time.perf_counter() - tb, 1), total=round(time.perf_counter() - t0, 1))

        def summed(arm):
            return sum(loss[arm][b][0] for b in INNER), sum(loss[arm][b][1] for b in INNER)

        if stage == 'dryrun':
            sels = {a: choose(*summed(a), fallback=1) for a in arms}
            np.savez_compressed(f'{out}/dryrun_fold-0_inner_selection.npz', **sels)
            json.dump(bank_info, open(f'{out}/bank_info.json', 'w'), indent=1)
            res = dict(scenario=scenario, stage=stage, reproduction=repro, selection_counts={a: np.bincount(s, minlength=4).tolist() for a, s in sels.items()},
                       seconds=time.perf_counter() - t0)
            json.dump(res, open(f'{out}/summary.json', 'w'), indent=1)
            log(event='dryrun_done', **{k: v for k, v in res.items() if k != 'selection_counts'})
            return
        P = np.load(f'{EXT}/{scenario}/fold-{f}/predictions.npz', allow_pickle=False)
        truth, sd, use = P['truth'], P['train_sd'], P['evaluated']
        assert (P['panel'] == ometa['panel']).all() and (P['held'] == ometa['query']).all() and (P['fit'] == ometa['fit']).all()
        preds, sels = {}, {}
        for a in arms:
            sel = choose(*summed(a), fallback=1)
            sels[a] = sel; preds[a] = take(outer[a], sel)
        repro['pred_M_vs_original__m_only'] = max(repro['pred_M_vs_original__m_only'], float(np.max(np.abs(preds['M'] - P['original__m_only']))))
        repro['pred_repro_vs_extra__m_only'] = max(repro['pred_repro_vs_extra__m_only'], float(np.max(np.abs(preds[REPRO] - P['extra__m_only']))))
        preds['null_drug'] = np.broadcast_to(np.nanmean(S['auc'][np.ix_(ometa['fit'], ometa['panel'])], axis=0), truth.shape).copy()
        np.savez_compressed(f'{out}/fold-{f}_predictions.npz', truth=truth, train_sd=sd, evaluated=use, panel=P['panel'],
                            drug_ids=P['drug_ids'], held=P['held'], fit=P['fit'],
                            **{'pred__' + k: v for k, v in preds.items()}, **{'sel__' + k: v for k, v in sels.items()})
        np.savez_compressed(f'{out}/fold-{f}_inner_losses.npz',
                            **{f'{a}__{b}__loss': v[0] for a, d in loss.items() for b, v in d.items()},
                            **{f'{a}__{b}__count': v[1] for a, d in loss.items() for b, v in d.items()})
        sel_counts[f] = {a: np.bincount(s, minlength=len(ALPHAS)).tolist() for a, s in sels.items()}
        for a, v in preds.items():
            mm = metrics(truth, v, sd, use)
            fold_rows.append(dict(fold=f, arm=a, score=mm['score'], spearman=mm['spearman'], centered_mse=mm['centered_mse'],
                                  drugs=int(use.sum()), n_fit=int(len(P['fit'])), n_held=int(len(P['held']))))
            for j in range(len(sd)):
                perdrug_rows.append(dict(fold=f, arm=a, drug_id=str(P['drug_ids'][j]), evaluated=bool(use[j]),
                                         standardized_mse=float(mm['mse'][j]), spearman=float(mm['rho'][j])))
        log(event='fold_done', scenario=scenario, fold=f, total=round(time.perf_counter() - t0, 1),
            score_M=fold_rows[-len(preds)]['score'], pred_diff_M=repro['pred_M_vs_original__m_only'])
    with open(f'{out}/fold_metrics.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(fold_rows[0])); w.writeheader(); w.writerows(fold_rows)
    with open(f'{out}/per_drug_metrics.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(perdrug_rows[0])); w.writeheader(); w.writerows(perdrug_rows)
    json.dump(bank_info, open(f'{out}/bank_info.json', 'w'), indent=1)
    names = list(dict.fromkeys(r['arm'] for r in fold_rows))
    summary = {a: {k: float(np.mean([r[k] for r in fold_rows if r['arm'] == a])) for k in ('score', 'spearman', 'centered_mse')} for a in names}
    folds = {a: [r['score'] for r in fold_rows if r['arm'] == a] for a in names}
    passed = bool(max(repro['grid_M_vs_original__m_only'], repro['grid_repro_vs_extra__m_only'],
                      repro['pred_M_vs_original__m_only'], repro['pred_repro_vs_extra__m_only']) <= 1e-8)
    res = dict(scenario=scenario, stage=stage, reproduction=dict(**repro, passed=passed), summary=summary, fold_scores=folds,
               selection_counts=sel_counts, seconds=time.perf_counter() - t0)
    json.dump(res, open(f'{out}/summary.json', 'w'), indent=1)
    log(event='done', scenario=scenario, reproduction=res['reproduction'], seconds=round(time.perf_counter() - t0, 1))


if __name__ == '__main__':
    main()

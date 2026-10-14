"""Tissue-kernel augmentation for the new-cell scenario (pre-registered in PLAN.md).

Independent re-implementation of the saved masked KRR / P-graph prediction (no project imports).
Reads saved geometry banks of extra-supervision-20261008-0001; writes only to OUT_DIR.
"""
import csv
import json
import os
import sys
import time
import numpy as np
from scipy.linalg import eigh, cho_factor, cho_solve
from scipy.stats import spearmanr

EXT = '/public/home/mengxl/creed-research-runs/extra-supervision-20261008-0001/output/new_cell'
SOURCE = '/public/home/mengxl/creed-research-runs/gdsc-metabolite-full-protein-20260908-0001/output/result/source-inputs.npz'
EXTRA = '/public/home/mengxl/creed-research-runs/extra-supervision-20261008-0001/resources/m_gdsc_extra19.npz'
OUT_DIR = sys.argv[1]
ALPHAS = (.1, 1., 10., 100.)
MUS = (0., .1, 1., 10., 100., 1000.)
WS = (0., .25, 1.)
BANKS = ('inner-0', 'inner-1', 'inner-2', 'outer')


def predict_grid(k, h, y):
    pred = np.empty((len(ALPHAS), len(h), y.shape[1])); groups = {}
    for j in range(y.shape[1]):
        groups.setdefault(np.isfinite(y[:, j]).tobytes(), []).append(j)
    for cols in groups.values():
        obs = np.flatnonzero(np.isfinite(y[:, cols[0]]))
        yy = y[np.ix_(obs, cols)]; mean = yy.mean(0)
        pred[:, :, cols] = mean
        if len(obs) < 20:
            continue
        kk = k[np.ix_(obs, obs)]; hh = h[:, obs]
        for ai, alpha in enumerate(ALPHAS):
            coef = cho_solve(cho_factor(kk + alpha * np.eye(len(obs)), lower=True), yy - mean)
            pred[ai][:, cols] = hh @ coef + mean
    return pred


def graph_grids(k, h, y, L):
    vals, vecs = eigh((k + k.T) / 2)
    root = (vecs * np.sqrt(np.maximum(vals, 0))) @ vecs.T
    parts = [predict_grid(k, h, y)]
    for mu in MUS[1:]:
        a = np.eye(len(k)) + mu * root @ L @ root; a = (a + a.T) / 2
        ke = root @ cho_solve(cho_factor(a, lower=True), root); ke = (ke + ke.T) / 2
        he = h - mu * h @ L @ ke
        parts.append(predict_grid(ke, he, y))
    return np.concatenate(parts)


def tissue_tokens(t, tag):
    return np.array([x if x else f'__unknown_{tag}_{i}' for i, x in enumerate(t)])


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
                centered_mse=float((mse - bias * bias)[use].mean()), mse=mse)


def tissue_drug_mean(y, ttr, tq, min_n=3):
    drug_mean = np.nanmean(y, axis=0)
    out = np.broadcast_to(drug_mean, (len(tq), y.shape[1])).copy()
    for t in np.unique(tq):
        rows = ttr == t
        if not rows.any():
            continue
        cnt = np.isfinite(y[rows]).sum(0)
        with np.errstate(invalid='ignore'):
            tm = np.nanmean(y[rows], axis=0) if rows.sum() > 0 else drug_mean
        ok = cnt >= min_n
        q = tq == t
        out[np.ix_(q, ok)] = tm[ok]
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=False)
    t0 = time.perf_counter()
    S = np.load(SOURCE, allow_pickle=False); E = np.load(EXTRA, allow_pickle=False)
    auc, stis = S['auc'], S['tissues']
    assert (E['drug_ids'] == S['drug_ids']).all()
    families = {'repro_m': None, 'repro_mP': None, 'tissue_only': None, 'm_tissue': None, 'm_tissue__protein': None}
    fold_rows, repro = [], {'grid_max_abs_diff': 0.0, 'pred_max_abs_diff': 0.0}
    selected_w = {'m_tissue': [], 'm_tissue__protein': []}
    for f in range(5):
        total = {k: 0 for k in families}; count = {k: 0 for k in families}
        outer_grids = None
        for bank in BANKS:
            G = np.load(f'{EXT}/fold-{f}/{bank}/geometry.npz', allow_pickle=False)
            fit, query, panel, ei = G['fit'], G['query'], G['panel'], G['extra_indices']
            K, H, L = G['m_only_kernel'], G['m_only_cross'], G['L_protein']
            y = np.vstack([auc[np.ix_(fit, panel)], E['auc'][np.ix_(ei, panel)]])
            ttr = tissue_tokens(np.concatenate([stis[fit], E['tissues'][ei]]), 'tr')
            tq = tissue_tokens(stis[query], 'q')
            T = (ttr[:, None] == ttr[None, :]).astype(float); Tq = (tq[:, None] == ttr[None, :]).astype(float)
            mt, mtp = [], []
            for w in WS:
                mtp.append(graph_grids(K + w * T, H + w * Tq, y, L))
                mt.append(mtp[-1][:len(ALPHAS)])
            grids = {'repro_m': mt[0], 'repro_mP': mtp[0], 'tissue_only': predict_grid(T, Tq, y),
                     'm_tissue': np.concatenate(mt), 'm_tissue__protein': np.concatenate(mtp)}
            saved = np.load(f'{EXT}/fold-{f}/{bank}/grids.npz', allow_pickle=False)
            repro['grid_max_abs_diff'] = max(repro['grid_max_abs_diff'],
                                             float(np.max(np.abs(grids['repro_m'] - saved['extra__m_only']))),
                                             float(np.max(np.abs(grids['repro_mP'] - saved['extra__m_only__protein']))))
            if bank == 'outer':
                outer_grids, outer_meta = grids, (fit, ei, panel, ttr, tq, y)
            else:
                truth = auc[np.ix_(query, panel)]
                sd = np.maximum(np.nanstd(auc[np.ix_(fit, panel)], axis=0), 1e-8)
                for k_, g in grids.items():
                    ll, nn = bank_losses(g, truth, sd); total[k_] = total[k_] + ll; count[k_] = count[k_] + nn
            print(json.dumps(dict(event='bank_done', fold=f, bank=bank, seconds=round(time.perf_counter() - t0, 1))), flush=True)
        P = np.load(f'{EXT}/fold-{f}/predictions.npz', allow_pickle=False)
        truth, sd, use = P['truth'], P['train_sd'], P['evaluated']
        preds = {}
        for k_ in families:
            sel = choose(total[k_], count[k_], fallback=1)
            preds[k_] = take(outer_grids[k_], sel)
            if k_ == 'm_tissue':
                selected_w[k_].extend((sel // len(ALPHAS)).tolist())
            if k_ == 'm_tissue__protein':
                selected_w[k_].extend((sel // (len(ALPHAS) * len(MUS))).tolist())
        repro['pred_max_abs_diff'] = max(repro['pred_max_abs_diff'],
                                         float(np.max(np.abs(preds['repro_m'] - P['extra__m_only']))),
                                         float(np.max(np.abs(preds['repro_mP'] - P['extra__m_only__protein']))))
        _, _, _, ttr, tq, y = outer_meta
        preds['null_drug'] = np.broadcast_to(np.nanmean(y, axis=0), truth.shape)
        preds['null_tissue_drug'] = tissue_drug_mean(y, ttr, tq)
        for arm, v in preds.items():
            m = metrics(truth, v, sd, use)
            fold_rows.append(dict(fold=f, arm=arm, score=m['score'], spearman=m['spearman'], centered_mse=m['centered_mse']))
        np.savez_compressed(f'{OUT_DIR}/fold-{f}_predictions.npz', **{k: np.asarray(v) for k, v in preds.items()})
    with open(f'{OUT_DIR}/fold_metrics.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(fold_rows[0])); w.writeheader(); w.writerows(fold_rows)
    arms = sorted({r['arm'] for r in fold_rows})
    summary = {a: {k: float(np.mean([r[k] for r in fold_rows if r['arm'] == a])) for k in ('score', 'spearman', 'centered_mse')} for a in arms}
    folds = {a: [r['score'] for r in fold_rows if r['arm'] == a] for a in arms}
    wdist = {k: {str(WS[i]): int(np.sum(np.array(v) == i)) for i in range(len(WS))} for k, v in selected_w.items()}
    out = dict(reproduction=repro, summary=summary, fold_scores=folds, selected_tissue_weight_counts=wdist,
               seconds=time.perf_counter() - t0)
    json.dump(out, open(f'{OUT_DIR}/summary.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()

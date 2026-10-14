"""Probe 2, part A: context-adaptive measured protein panels with RETRAINED heads (pre-registered in PLAN.md).

Independent implementation; model pieces follow protein_panel.py / crosslayer.py (reference-equivalent).
Reads only: extra-supervision banks (geometry/grids/predictions, fold_metrics.csv, resources/m_gdsc_extra19.npz),
source-inputs.npz, protein-panel-20261009-0001 outputs (reproduction + reference columns).
Writes only to OUT_DIR.

usage: python probe2.py SCENARIO dryrun OUT_DIR          # fold-0 inner-0 bank only; no outer score
       python probe2.py SCENARIO fold   OUT_DIR FOLD     # one fold, all 4 banks, selection, outer predictions, metrics
"""
import json
import os
import sys
import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, spearmanr

RUN = '/public/home/mengxl/creed-research-runs'
EXT = RUN + '/extra-supervision-20261008-0001/output'
EXTRA = RUN + '/extra-supervision-20261008-0001/resources/m_gdsc_extra19.npz'
SOURCE = RUN + '/gdsc-metabolite-full-protein-20260908-0001/output/result/source-inputs.npz'
PP = RUN + '/protein-panel-20261009-0001/output'
SCEN_ID = {'new_cell': 0, 'tissue': 1}
BANK_ID = {'inner-0': 0, 'inner-1': 1, 'inner-2': 2, 'outer': 3}
BANKS = ('inner-0', 'inner-1', 'inner-2', 'outer')
ALPHAS = (.1, 1., 10., 100.)
KS = (8, 16, 32, 64)
SEEDS = (0, 1, 2)
COVER, MIN_PAIR, MIN_CTX, N0, KM_K, KM_SEED, KM_INIT, KM_ITER = 0.9, 10, 8, 10., 5, 20261009, 10, 100


def log(**kw):
    print(json.dumps(kw), flush=True)


# ---------------------------------------------------------------- model pieces (reference-equivalent)
def predict_grid(k, h, y):
    pred = np.empty((len(ALPHAS), len(h), y.shape[1])); groups = {}
    for j in range(y.shape[1]):
        groups.setdefault(np.isfinite(y[:, j]).tobytes(), []).append(j)
    for cols in groups.values():
        obs = np.flatnonzero(np.isfinite(y[:, cols[0]]))
        if not len(obs):
            raise ValueError('no fit label')
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


def rbf_kernel(x, q):
    mu = x.mean(0); sd = x.std(0); sd = np.where(sd < 1e-8, 1., sd)
    z, h = (x - mu) / sd, (q - mu) / sd
    d = cdist(z, z, 'sqeuclidean'); pos = d[np.triu_indices(len(z), 1)]; pos = pos[pos > 0]
    if not len(pos):
        return np.ones_like(d), np.ones((len(q), len(x)))
    width = np.median(pos)
    return np.exp(-d / (2 * width)), np.exp(-cdist(h, z, 'sqeuclidean') / (2 * width))


def protein_kernel(p, fit, query, idx):
    x, xq = p[np.ix_(fit, idx)], p[np.ix_(query, idx)]
    mean = np.nanmean(x, axis=0)
    assert np.isfinite(mean).all()
    return rbf_kernel(np.where(np.isfinite(x), x, mean), np.where(np.isfinite(xq), xq, mean))


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
                centered_mse=float((mse - bias * bias)[use].mean()))


def tissue_tokens(t, tag):
    return np.array([x if x else f'__unknown_{tag}_{i}' for i, x in enumerate(t)])


# ---------------------------------------------------------------- relevance (fit cells only; identical to protein_panel.py)
def spearman_scores(P, Y, min_n=MIN_PAIR):
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


def candidates(p_fit):
    obs = np.isfinite(p_fit); cnt = obs.sum(0); frac = cnt / len(p_fit)
    mean = np.divide(np.where(obs, p_fit, 0).sum(0), cnt, out=np.zeros(p_fit.shape[1]), where=cnt > 0)
    var = np.divide(np.where(obs, (p_fit - mean) ** 2, 0).sum(0), cnt, out=np.zeros(p_fit.shape[1]), where=cnt > 0)
    return np.flatnonzero((frac >= COVER) & (var > 1e-12))


def order_by(score, cand):
    """descending score, stable (ties -> ascending candidate column index)."""
    return cand[np.argsort(-score, kind='stable')]


# ---------------------------------------------------------------- k-means (fit-only, fixed seed)
def kmeans(Z, K=KM_K, seed=KM_SEED, n_init=KM_INIT, max_iter=KM_ITER):
    rng = np.random.default_rng(seed); n = len(Z); best = None
    for _ in range(n_init):
        C = np.empty((K, Z.shape[1])); C[0] = Z[rng.integers(n)]
        for c in range(1, K):
            d2 = cdist(Z, C[:c], 'sqeuclidean').min(1)
            C[c] = Z[rng.choice(n, p=d2 / d2.sum())]
        labels = None
        for _it in range(max_iter):
            D = cdist(Z, C, 'sqeuclidean'); new = D.argmin(1)
            if labels is not None and np.array_equal(new, labels):
                break
            labels = new
            for c in range(K):
                m = labels == c
                C[c] = Z[m].mean(0) if m.any() else Z[D.min(1).argmax()]
        D = cdist(Z, C, 'sqeuclidean'); labels = D.argmin(1); inertia = float(D[np.arange(n), labels].sum())
        if best is None or inertia < best[0]:
            best = (inertia, C.copy(), labels.copy())
    return best


# ---------------------------------------------------------------- one bank
def run_bank(S, E, scenario, f, bank, emit, info):
    t0 = time.perf_counter()
    G = np.load(f'{EXT}/{scenario}/fold-{f}/{bank}/geometry.npz', allow_pickle=False)
    fit, query, panel, ei = G['fit'], G['query'], G['panel'], G['extra_indices']; n = len(fit); nq = len(query)
    auc, p, m, tis = S['auc'], S['p'], S['m'], S['tissues']
    y = auc[np.ix_(fit, panel)]
    KM, HM = G['m_only_kernel'][:n, :n].copy(), G['m_only_cross'][:, :n].copy()
    # reproduction arms
    emit('M', predict_grid(KM, HM, y))
    assert G['m_only_kernel'].shape == (n + len(ei), n + len(ei))
    emit('XM', predict_grid(G['m_only_kernel'], G['m_only_cross'], np.vstack([y, E['auc'][np.ix_(ei, panel)]])))
    # candidate pool + global relevance
    cand = candidates(p[fit])
    w_global = spearman_scores(p[np.ix_(fit, cand)], y)
    order_g = order_by(w_global, cand)
    info.update(n_fit=int(n), n_query=int(nq), candidate_count=int(len(cand)), top64=order_g[:64].tolist())
    # contexts: name -> (fit mask, query mask); eligible contexts have >= MIN_CTX fit cells
    contexts = {}
    if scenario == 'new_cell':
        ttr, tq = tissue_tokens(tis[fit], 'tr'), tissue_tokens(tis[query], 'q')
        cnt = {t: int((ttr == t).sum()) for t in sorted(set(ttr.tolist()))}
        elig = [t for t in sorted(cnt) if cnt[t] >= MIN_CTX]
        contexts['tissue'] = ({t: (ttr == t, tq == t) for t in elig}, np.isin(tq, elig, invert=True))
        info['tissue'] = dict(fit_counts=cnt, eligible=elig, query_counts={t: int((tq == t).sum()) for t in sorted(set(tq.tolist()))},
                              query_to_global=int(np.isin(tq, elig, invert=True).sum()))
    mu = m[fit].mean(0); sd = m[fit].std(0); sd = np.where(sd < 1e-8, 1., sd)
    Zf, Zq = (m[fit] - mu) / sd, (m[query] - mu) / sd
    inertia, C, lab = kmeans(Zf)
    labq = cdist(Zq, C, 'sqeuclidean').argmin(1)
    csz = {int(c): int((lab == c).sum()) for c in range(KM_K)}
    celig = [c for c in range(KM_K) if csz[c] >= MIN_CTX]
    contexts['cluster'] = ({str(c): (lab == c, labq == c) for c in celig}, np.isin(labq, celig, invert=True))
    info['cluster'] = dict(inertia=inertia, fit_sizes=csz, eligible=celig, query_sizes={int(c): int((labq == c).sum()) for c in range(KM_K)},
                           query_to_global=int(np.isin(labq, celig, invert=True).sum()),
                           tissue_by_cluster={int(c): {t: int(v) for t, v in zip(*np.unique(tis[fit][lab == c], return_counts=True))} for c in range(KM_K)})
    # context relevance (shrunk toward global)
    orders = {}
    for pol, (ctx, _) in contexts.items():
        orders[pol] = {}
        for name, (fm, _qm) in ctx.items():
            nc = float(fm.sum())
            w_ctx = spearman_scores(p[np.ix_(fit[fm], cand)], y[fm])
            w = (nc / (nc + N0)) * w_ctx + (N0 / (nc + N0)) * w_global
            orders[pol][name] = order_by(w, cand)
    info['relevance_seconds'] = round(time.perf_counter() - t0, 1)
    # heads: one per distinct panel, trained on all fit cells
    heads = {}

    def head(idx):
        key = tuple(int(i) for i in idx)
        if key not in heads:
            pk, ph = protein_kernel(p, fit, query, np.array(key))
            heads[key] = predict_grid(.5 * KM + .5 * pk, .5 * HM + .5 * ph, y)
        return heads[key]

    def policy_grid(panels, ctx, to_global):
        """panels: {'__global__': idx, name: idx}; ctx: {name: (fm, qm)}; to_global: query mask using the global slot."""
        pred = np.empty((len(ALPHAS), nq, y.shape[1])); covered = np.zeros(nq, bool)
        for name, (_fm, qm) in ctx.items():
            if qm.any():
                pred[:, qm, :] = head(panels[name])[:, qm, :]; covered |= qm
        if to_global.any():
            pred[:, to_global, :] = head(panels['__global__'])[:, to_global, :]; covered |= to_global
        assert covered.all()
        return pred

    overlap = {}
    for k in KS:
        g_idx = np.sort(order_g[:k])
        emit(f'G{k}', head(g_idx))
        for pol, (ctx, to_global) in contexts.items():
            tag = 'T' if pol == 'tissue' else 'C'
            panels = {'__global__': g_idx}
            for name in ctx:
                panels[name] = np.sort(orders[pol][name][:k])
            emit(f'{tag}{k}', policy_grid(panels, ctx, to_global))
            if k == 16:
                overlap[pol] = {name: len(set(panels[name].tolist()) & set(g_idx.tolist())) / len(set(panels[name].tolist()) | set(g_idx.tolist()))
                                for name in ctx}
    info['jaccard_vs_global_k16'] = overlap
    info['heads_adaptive'] = len(heads)
    # matched random (same contexts, same number of heads), 3 seeds
    for pol, (ctx, to_global) in contexts.items():
        tag, pid = ('RT', 1) if pol == 'tissue' else ('RC', 2)
        for s in SEEDS:
            rng = np.random.default_rng([20261009, SCEN_ID[scenario], f, BANK_ID[bank], pid, s])
            perms = {'__global__': rng.permutation(cand)}
            for name in ctx:                       # ctx keys are sorted tissue names / cluster ids 0..4
                perms[name] = rng.permutation(cand)
            for k in KS:
                emit(f'{tag}{k}s{s}', policy_grid({nm: np.sort(pm[:k]) for nm, pm in perms.items()}, ctx, to_global))
    # random global (same seeds as protein-panel random_k16/32 for s<=2), 3 seeds
    rand = {}
    for s in SEEDS:
        perm = np.random.default_rng([20261009, SCEN_ID[scenario], f, BANK_ID[bank], s]).permutation(cand)
        for k in KS:
            idx = np.sort(perm[:k]); rand[f'RG{k}s{s}'] = idx.tolist()
            emit(f'RG{k}s{s}', head(idx))
    info['random_global_panels'] = rand
    # k = all end (candidate-pool teacher)
    pk, ph = protein_kernel(p, fit, query, cand)
    emit('A', predict_grid(.5 * KM + .5 * pk, .5 * HM + .5 * ph, y))
    info.update(heads_total=len(heads), bank_seconds=round(time.perf_counter() - t0, 1))
    return G


def arm_keys(scenario):
    A = {'m_only': 'M', 'repro_extra_m_only': 'XM', 'm_allcand': 'A'}
    for k in KS:
        A[f'global_static_k{k}'] = f'G{k}'
        if scenario == 'new_cell':
            A[f'tissue_adaptive_k{k}'] = f'T{k}'
        A[f'cluster_adaptive_k{k}'] = f'C{k}'
        for s in SEEDS:
            if scenario == 'new_cell':
                A[f'matched_random_tissue_k{k}_s{s}'] = f'RT{k}s{s}'
            A[f'matched_random_cluster_k{k}_s{s}'] = f'RC{k}s{s}'
            A[f'random_global_k{k}_s{s}'] = f'RG{k}s{s}'
    return A


def load_source():
    S = np.load(SOURCE, allow_pickle=False)
    S = {k: S[k] for k in ('m', 'auc', 'p', 'tissues', 'sample_ids', 'drug_ids', 'protein_ids')}
    E = np.load(EXTRA, allow_pickle=False)
    assert (E['drug_ids'] == S['drug_ids']).all()
    return S, {'auc': E['auc']}


def main():
    scenario, stage, out = sys.argv[1], sys.argv[2], sys.argv[3]
    assert scenario in ('new_cell', 'tissue') and stage in ('dryrun', 'fold')
    os.makedirs(out, exist_ok=True)
    t0 = time.perf_counter()
    S, E = load_source()
    arms = arm_keys(scenario); keys = sorted(set(arms.values()))
    if stage == 'dryrun':
        f, bank = 0, 'inner-0'
        store, info = {}, {}
        run_bank(S, E, scenario, f, bank, lambda k, g: store.__setitem__(k, g), info)
        saved = np.load(f'{EXT}/{scenario}/fold-{f}/{bank}/grids.npz', allow_pickle=False)
        d1 = float(np.max(np.abs(store['M'] - saved['original__m_only'])))
        d2 = float(np.max(np.abs(store['XM'] - saved['extra__m_only'])))
        pp = json.load(open(f'{PP}/{scenario}_main/bank_info.json'))[f'fold-{f}/{bank}']
        same_top = pp['top64'] == info['top64']
        same_rand = all(pp['random_panels'][f'R{k}s{s}'] == info['random_global_panels'][f'RG{k}s{s}'] for k in (16, 32) for s in SEEDS)
        G = np.load(f'{EXT}/{scenario}/fold-{f}/{bank}/geometry.npz', allow_pickle=False)
        truth = S['auc'][np.ix_(G['query'], G['panel'])]; sd = np.maximum(np.nanstd(S['auc'][np.ix_(G['fit'], G['panel'])], axis=0), 1e-8)
        inner_scores = {}
        for a, k in arms.items():          # engineering check only: inner-bank score of alpha index 1 (not a selection)
            inner_scores[a] = metrics(truth, store[k][1], sd, np.isfinite(truth).sum(0) > 0)['score']
        res = dict(scenario=scenario, stage=stage, fold=f, bank=bank, grid_diff_original_m_only=d1, grid_diff_extra_m_only=d2,
                   top64_identical_to_protein_panel=same_top, random_panels_identical_to_protein_panel=same_rand,
                   keys_emitted=len(store), keys_needed=len(keys), missing=[k for k in keys if k not in store],
                   info=info, inner_alpha1_scores=inner_scores, seconds=time.perf_counter() - t0)
        json.dump(res, open(f'{out}/dryrun_{scenario}.json', 'w'), indent=1)
        log(event='dryrun_done', scenario=scenario, d1=d1, d2=d2, same_top=same_top, same_rand=same_rand, missing=res['missing'],
            candidates=info['candidate_count'], heads=info['heads_total'], seconds=round(res['seconds'], 1))
        return
    f = int(sys.argv[4])
    total = {k: 0 for k in keys}; count = {k: 0 for k in keys}; outer = {}
    repro = dict(grid_original_m_only=0., grid_extra_m_only=0.)
    bank_info = {}
    for bank in BANKS:
        info = {}
        if bank == 'outer':
            def emit(key, grid):
                outer[key] = grid
        else:
            Gq = np.load(f'{EXT}/{scenario}/fold-{f}/{bank}/geometry.npz', allow_pickle=False)
            truth = S['auc'][np.ix_(Gq['query'], Gq['panel'])]
            sd = np.maximum(np.nanstd(S['auc'][np.ix_(Gq['fit'], Gq['panel'])], axis=0), 1e-8)

            def emit(key, grid, truth=truth, sd=sd):
                ll, nn = bank_losses(grid, truth, sd); total[key] = total[key] + ll; count[key] = count[key] + nn
                if key in ('M', 'XM'):
                    outer[f'{bank}:{key}'] = grid
        G = run_bank(S, E, scenario, f, bank, emit, info)
        saved = np.load(f'{EXT}/{scenario}/fold-{f}/{bank}/grids.npz', allow_pickle=False)
        gm = outer['M'] if bank == 'outer' else outer.pop(f'{bank}:M'); gx = outer['XM'] if bank == 'outer' else outer.pop(f'{bank}:XM')
        repro['grid_original_m_only'] = max(repro['grid_original_m_only'], float(np.max(np.abs(gm - saved['original__m_only']))))
        repro['grid_extra_m_only'] = max(repro['grid_extra_m_only'], float(np.max(np.abs(gx - saved['extra__m_only']))))
        pp = json.load(open(f'{PP}/{scenario}_main/bank_info.json'))[f'fold-{f}/{bank}']
        info['top64_identical_to_protein_panel'] = bool(pp['top64'] == info['top64'])
        info['random_panels_identical_to_protein_panel'] = bool(all(pp['random_panels'][f'R{k}s{s}'] == info['random_global_panels'][f'RG{k}s{s}'] for k in (16, 32) for s in SEEDS))
        bank_info[f'fold-{f}/{bank}'] = info
        log(event='bank_done', scenario=scenario, fold=f, bank=bank, candidates=info['candidate_count'], heads=info['heads_total'],
            bank_seconds=info['bank_seconds'], total=round(time.perf_counter() - t0, 1))
    # cache heavy results before finalisation
    np.savez_compressed(f'{out}/fold-{f}_cache.npz', **{'outer__' + k: v for k, v in outer.items()},
                        **{'total__' + k: v for k, v in total.items()}, **{'count__' + k: v for k, v in count.items()})
    P = np.load(f'{EXT}/{scenario}/fold-{f}/predictions.npz', allow_pickle=False)
    truth, sd, use = P['truth'], P['train_sd'], P['evaluated']
    assert np.array_equal(P['fit'], G['fit']) and np.array_equal(P['held'], G['query']) and np.array_equal(P['panel'], G['panel'])
    preds, sels = {}, {}
    for a, k in arms.items():
        sel = choose(total[k], count[k], fallback=1); sels[a] = sel; preds[a] = take(outer[k], sel)
    repro['pred_original_m_only'] = float(np.max(np.abs(preds['m_only'] - P['original__m_only'])))
    repro['pred_extra_m_only'] = float(np.max(np.abs(preds['repro_extra_m_only'] - P['extra__m_only'])))
    Z = np.load(f'{PP}/{scenario}_main/fold-{f}_predictions.npz', allow_pickle=False)
    assert np.array_equal(Z['fit'], P['fit']) and np.array_equal(Z['held'], P['held']) and np.array_equal(Z['panel'], P['panel'])
    repro['pred_global_static_vs_m_panel_k'] = {k: float(np.max(np.abs(preds[f'global_static_k{k}'] - Z[f'm_panel_k{k}']))) for k in KS}
    repro['pred_m_allcand_vs_saved'] = float(np.max(np.abs(preds['m_allcand'] - Z['m_allcand'])))
    repro['pred_random_global_vs_saved'] = {f'k{k}_s{s}': float(np.max(np.abs(preds[f'random_global_k{k}_s{s}'] - Z[f'random_k{k}_s{s}']))) for k in (16, 32) for s in SEEDS}
    preds['teacher_fullP_saved'] = Z['teacher_saved']
    preds['null_drug'] = np.broadcast_to(np.nanmean(S['auc'][np.ix_(P['fit'], P['panel'])], axis=0), truth.shape).copy()
    # metric reproduction on saved predictions
    import csv
    pp_csv = {(int(r['fold']), r['arm']): float(r['score']) for r in csv.DictReader(open(f'{PP}/{scenario}_main/fold_metrics.csv'))}
    ex_csv = {(int(r['fold']), r['arm']): float(r['score']) for r in csv.DictReader(open(f'{EXT}/{scenario}/fold_metrics.csv'))}
    repro['metric_vs_protein_panel_csv'] = max(abs(metrics(truth, Z[a], sd, use)['score'] - pp_csv[(f, a)]) for a in ['m_only', 'm_allcand', 'teacher_saved'] + [f'm_panel_k{k}' for k in KS])
    repro['metric_vs_extra_supervision_csv'] = max(abs(metrics(truth, P[a], sd, use)['score'] - ex_csv[(f, a)]) for a in ('original__m_only', 'extra__m_only'))
    rows = []
    for a, v in preds.items():
        mm = metrics(truth, v, sd, use)
        rows.append(dict(scenario=scenario, arm=a, fold=f, n_fit=int(len(P['fit'])), n_query=int(len(P['held'])), **mm))
    np.savez_compressed(f'{out}/fold-{f}_predictions.npz', truth=truth, train_sd=sd, evaluated=use, fit=P['fit'], held=P['held'], panel=P['panel'],
                        **{'pred__' + a: v for a, v in preds.items()})
    np.savez_compressed(f'{out}/fold-{f}_selection.npz', **{'sel__' + a: s for a, s in sels.items()},
                        **{'loss__' + a: total[k] for a, k in arms.items()}, **{'count__' + a: count[k] for a, k in arms.items()})
    json.dump(rows, open(f'{out}/fold-{f}_metrics.json', 'w'), indent=1)
    json.dump(repro, open(f'{out}/fold-{f}_repro.json', 'w'), indent=1)
    json.dump(bank_info, open(f'{out}/fold-{f}_bank_info.json', 'w'), indent=1)
    os.remove(f'{out}/fold-{f}_cache.npz')
    log(event='fold_done', scenario=scenario, fold=f, repro=repro, score_m_only=[r['score'] for r in rows if r['arm'] == 'm_only'][0],
        seconds=round(time.perf_counter() - t0, 1))


if __name__ == '__main__':
    main()

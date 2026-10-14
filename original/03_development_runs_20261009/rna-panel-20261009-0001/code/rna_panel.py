"""Few-layer deployment with a SMALL MEASURED RNA panel: M (+ tissue label in new_cell) + k genes (pre-registered in PLAN.md).

Independent implementation (model functions copied from the local protein_panel.py / crosslayer.py references; no project imports).
Reads only: saved banks of extra-supervision-20261008-0001, source-inputs.npz, rna421 log CPM, the crosslayer-matrix run's saved
predictions / inner losses (teacher + reproduction gate) and the protein-panel run's saved predictions (m_tissue / tissue_only check).
Writes only to OUT_DIR (must not exist).

usage: python rna_panel.py SCENARIO STAGE OUT_DIR
  STAGE = dryrun : fold-0 inner-0 bank only -> inner grid reproduction, inner-loss check vs crosslayer run, Spearman path check (no outer score)
  STAGE = repro  : reproduction arms only (m_only, m_allcand, [tissue_only, m_tissue, m_allcand_tissue] / [vfm, vfm_allcand]), all folds
  STAGE = main   : all pre-registered arms, all folds
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
from scipy.stats import rankdata, spearmanr

ROOT = '/public/home/mengxl/creed-research-runs'
EXT = ROOT + '/extra-supervision-20261008-0001/output'
SOURCE = ROOT + '/gdsc-metabolite-full-protein-20260908-0001/output/result/source-inputs.npz'
RNA421 = ROOT + '/rna421-20260906-0002'
MATRIX = ROOT + '/crosslayer-matrix-20261009-0001/output'
PPANEL = ROOT + '/protein-panel-20261009-0001/output'
SOURCE_SHA = '52d71517a8952e83a1603cd831f09449544102d0ca5ff829ff01be9bf79cd637'

ALPHAS = (.1, 1., 10., 100.)
WS_NEW = (0., .25, 1.)
KS = (8, 16, 32, 64, 128)
KSEL = (8, 16, 32, 64)
RKS = (16, 32)
SEEDS = (0, 1, 2, 3, 4)
TOP_GENES = 3000
P_COVER = 0.9
MIN_PAIR = 10
PG_K = 32
BANKS = ('inner-0', 'inner-1', 'inner-2', 'outer')
SCEN_ID = {'new_cell': 0, 'tissue': 1}
BANK_ID = {'inner-0': 0, 'inner-1': 1, 'inner-2': 2, 'outer': 3}
GATE = 1e-8


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


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


# ---------------------------------------------------------------- kernels (identical to crosslayer.py)
def kernel_state(x):
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


def same(a, b):
    return (a[:, None] == b[None, :]).astype(float)


def tokens(t, tag):
    return np.array([x if x else f'__unknown_{tag}_{i}' for i, x in enumerate(t)])


# ---------------------------------------------------------------- relevance scores (protein-panel rule)
def spearman_scores(P, Y, min_n=MIN_PAIR):
    """sum_j |Spearman(P[:, i], Y[:, j])| on pairwise-complete rows; pairs with n<min_n or a constant side add 0.
    Verbatim copy of protein_panel.py (handles NaN in P)."""
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


def spearman_scores_dense(X_, Y, min_n=MIN_PAIR):
    """Same rule for a NaN-free feature matrix (every feature observed on every row)."""
    assert np.isfinite(X_).all()
    c = X_.shape[1]; tot = np.zeros(c)
    for j in range(Y.shape[1]):
        o = np.isfinite(Y[:, j])
        if o.sum() < min_n:
            continue
        rx = rankdata(X_[o], axis=0); ry = rankdata(Y[o, j])
        dx = rx - rx.mean(0); dy = ry - ry.mean()
        num = (dx * dy[:, None]).sum(0); den = np.sqrt((dx * dx).sum(0) * (dy * dy).sum())
        ok = den > 1e-12
        r = np.zeros(c); r[ok] = num[ok] / den[ok]
        tot += np.abs(r)
    return tot


def p_candidates(p_fit):
    obs = np.isfinite(p_fit); cnt = obs.sum(0); frac = cnt / len(p_fit)
    mean = np.divide(np.where(obs, p_fit, 0).sum(0), cnt, out=np.zeros(p_fit.shape[1]), where=cnt > 0)
    var = np.divide(np.where(obs, (p_fit - mean) ** 2, 0).sum(0), cnt, out=np.zeros(p_fit.shape[1]), where=cnt > 0)
    return np.flatnonzero((frac >= P_COVER) & (var > 1e-12))


def protein_kernel(p, fit, query, idx):
    x, xq = p[np.ix_(fit, idx)], p[np.ix_(query, idx)]
    mean = np.nanmean(x, axis=0)
    assert np.isfinite(mean).all()
    K, H, st = rbf_pair(np.where(np.isfinite(x), x, mean), np.where(np.isfinite(xq), xq, mean))
    return K, H


# ---------------------------------------------------------------- arms
def wkey(w):
    return f'w{w}'


def arm_keys(scenario):
    ws = WS_NEW if scenario == 'new_cell' else (0.,)
    A = {}
    A['m_only'] = ['M|w0.0']
    for k in KS:
        A[f'm_panel_k{k}'] = [f'G{k}|w0.0']
    A['m_panel_sel'] = [f'G{k}|w0.0' for k in KSEL]
    for k in RKS:
        for s in SEEDS:
            A[f'random_k{k}_s{s}'] = [f'R{k}s{s}|w0.0']
    A['m_allcand'] = ['A|w0.0']
    A['m_panel_pg_k32'] = ['PG32|w0.0']
    if scenario == 'new_cell':
        A['tissue_only'] = ['T']
        A['m_tissue'] = [f'M|{wkey(w)}' for w in ws]
        for k in KS:
            A[f'm_panel_tissue_k{k}'] = [f'G{k}|{wkey(w)}' for w in ws]
        A['m_panel_tissue_sel'] = [f'G{k}|{wkey(w)}' for k in KSEL for w in ws]
        for k in RKS:
            for s in SEEDS:
                A[f'random_tissue_k{k}_s{s}'] = [f'R{k}s{s}|{wkey(w)}' for w in ws]
        A['m_allcand_tissue'] = [f'A|{wkey(w)}' for w in ws]
    else:
        A['vfm'] = ['V|w0.0']
        for k in KS:
            A[f'vfm_panel_k{k}'] = [f'VG{k}|w0.0']
        A['vfm_panel_sel'] = [f'VG{k}|w0.0' for k in KSEL]
        A['vfm_allcand'] = ['VA|w0.0']
    return A


REPRO_ARMS = {'new_cell': ('m_only', 'm_allcand', 'tissue_only', 'm_tissue', 'm_allcand_tissue'),
              'tissue': ('m_only', 'm_allcand', 'vfm', 'vfm_allcand')}
SAVED_NAME = {'m_only': 'original__m_only', 'vfm': 'original__virtual_fm'}      # extra-supervision grids + predictions
MATRIX_NAME = {'m_allcand': 'M+RNA', 'm_only': 'M'}                              # crosslayer-matrix predictions + inner losses
PPANEL_NAME = {'new_cell': ('m_tissue', 'tissue_only'), 'tissue': ('vfm',)}      # protein-panel saved predictions


# ---------------------------------------------------------------- one bank
def run_bank(S, X, scenario, f, bank, need, check_spearman=False):
    """Returns ({key: grid} for keys in `need`, info)."""
    G = np.load(f'{EXT}/{scenario}/fold-{f}/{bank}/geometry.npz', allow_pickle=False)
    fit, query, panel = G['fit'], G['query'], G['panel']; n = len(fit)
    auc, m, p, tis = S['auc'], S['m'], S['p'], S['tissues']
    y = auc[np.ix_(fit, panel)]
    ws = WS_NEW if scenario == 'new_cell' else (0.,)
    info = dict(n_fit=int(n), n_query=int(len(query)), n_panel=int(len(panel)))
    grids = {}

    def emit(key, K, H):
        if key in need:
            grids[key] = predict_grid(K, H, y)

    # M kernel recomputed exactly as crosslayer.py; checked against the saved original-data block
    KM, HM, stm = rbf_pair(m[fit], m[query])
    kd = max(float(np.max(np.abs(KM - G['m_only_kernel'][:n, :n]))), float(np.max(np.abs(HM - G['m_only_cross'][:, :n]))))
    np.testing.assert_allclose(KM, G['m_only_kernel'][:n, :n], rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(HM, G['m_only_cross'][:, :n], rtol=1e-11, atol=1e-12)
    info['m_kernel_block_max_abs_diff'] = kd; info['m_width'] = stm[2]
    if scenario == 'new_cell':
        ttr, tq = tokens(tis[fit], 'tr'), tokens(tis[query], 'q')
        T, Tq = same(ttr, ttr), same(tq, ttr)
    else:
        T = Tq = None
    if scenario == 'tissue':
        KV, HV = G['virtual_fm_kernel'][:n, :n].copy(), G['virtual_fm_cross'][:, :n].copy()

    def add_base(name, K, H):
        for w in ws:
            Kw, Hw = (K + w * T, H + w * Tq) if w else (K, H)
            emit(f'{name}|{wkey(w)}', Kw, Hw)

    add_base('M', KM, HM)
    if scenario == 'new_cell':
        emit('T', T, Tq)
    if scenario == 'tissue':
        emit('V|w0.0', KV, HV)
    # RNA candidate pool = crosslayer.py RNA feature set (top-3000 fit-variance genes, stable order)
    xf = X[fit]
    g = np.argsort(-xf.var(0), kind='stable')[:TOP_GENES]
    KR, HR, str_ = rbf_pair(xf[:, g], X[np.ix_(query, g)])
    info['rna'] = dict(features=int(len(g)), width=str_[2], top_gene_index=int(g[0]))
    add_base('A', .5 * KM + .5 * KR, .5 * HM + .5 * HR)
    if scenario == 'tissue':
        emit('VA|w0.0', .5 * KV + .5 * KR, .5 * HV + .5 * HR)
    want_panel = any(k.split('|')[0][0] in 'GRVP' and not k.startswith('VA') and not k.startswith('V|') for k in need)
    if not want_panel:
        return grids, info
    # supervised relevance (fit only); candidate pool sorted by gene column index for the tie rule
    cand = np.sort(g)
    t0 = time.perf_counter()
    score = spearman_scores_dense(X[np.ix_(fit, cand)], y)
    info['spearman_seconds'] = round(time.perf_counter() - t0, 1)
    if check_spearman:
        t1 = time.perf_counter()
        ref = spearman_scores(X[np.ix_(fit, cand)], y)
        info['spearman_general_seconds'] = round(time.perf_counter() - t1, 1)
        info['spearman_path_max_abs_diff'] = float(np.max(np.abs(ref - score)))
        assert np.array_equal(np.argsort(-ref, kind='stable')[:128], np.argsort(-score, kind='stable')[:128])
    order = cand[np.argsort(-score, kind='stable')]
    info['top128'] = order[:128].tolist(); info['top128_score'] = np.sort(score)[::-1][:128].round(6).tolist()
    panels = {}
    for k in KS:
        idx = np.sort(order[:k]); panels[f'G{k}'] = idx
        Kp, Hp, st = rbf_pair(X[np.ix_(fit, idx)], X[np.ix_(query, idx)])
        add_base(f'G{k}', .5 * KM + .5 * Kp, .5 * HM + .5 * Hp)
        if scenario == 'tissue':
            emit(f'VG{k}|w0.0', .5 * KV + .5 * Kp, .5 * HV + .5 * Hp)
        if k == PG_K:
            Kg32, Hg32 = Kp, Hp
    for s in SEEDS:
        rng = np.random.default_rng([20261009, SCEN_ID[scenario], f, BANK_ID[bank], s])
        perm = rng.permutation(cand)
        for k in RKS:
            idx = np.sort(perm[:k]); panels[f'R{k}s{s}'] = idx
            Kp, Hp, st = rbf_pair(X[np.ix_(fit, idx)], X[np.ix_(query, idx)])
            add_base(f'R{k}s{s}', .5 * KM + .5 * Kp, .5 * HM + .5 * Hp)
    # descriptive two-layer arm: 32 proteins by the protein-panel rule + 32 genes
    if 'PG32|w0.0' in need:
        t0 = time.perf_counter()
        pc = p_candidates(p[fit])
        ps = spearman_scores(p[np.ix_(fit, pc)], y)
        porder = pc[np.argsort(-ps, kind='stable')]
        pidx = np.sort(porder[:PG_K]); panels['P32'] = pidx
        Kp32, Hp32 = protein_kernel(p, fit, query, pidx)
        emit('PG32|w0.0', .5 * KM + .25 * Kp32 + .25 * Kg32, .5 * HM + .25 * Hp32 + .25 * Hg32)
        info['protein'] = dict(candidates=int(len(pc)), spearman_seconds=round(time.perf_counter() - t0, 1), top32=pidx.tolist())
    info['panels'] = {k: v.tolist() for k, v in panels.items()}
    return grids, info


# ---------------------------------------------------------------- data
def load_ctx():
    assert sha(SOURCE) == SOURCE_SHA, 'source sha mismatch'
    S = np.load(SOURCE, allow_pickle=False)
    S = {k: S[k] for k in ('m', 'auc', 'p', 'tissues', 'sample_ids', 'drug_ids', 'protein_ids')}
    X = np.load(RNA421 + '/rna_log_cpm.npy')
    assert (np.load(RNA421 + '/sample_ids.npy') == S['sample_ids']).all(), 'RNA row order differs from source'
    assert X.shape[0] == len(S['sample_ids']) and np.isfinite(X).all() and np.isfinite(S['m']).all()
    return S, X


def main():
    scenario, stage, out = sys.argv[1], sys.argv[2], sys.argv[3]
    assert scenario in ('new_cell', 'tissue') and stage in ('dryrun', 'repro', 'main')
    os.makedirs(out, exist_ok=False)
    t0 = time.perf_counter()
    S, X = load_ctx()
    arms = arm_keys(scenario)
    if stage == 'repro':
        arms = {a: arms[a] for a in REPRO_ARMS[scenario]}
    need = sorted({k for v in arms.values() for k in v})
    json.dump(dict(scenario=scenario, stage=stage, script_sha256=sha(__file__), source_sha256=SOURCE_SHA, alphas=ALPHAS, ws=WS_NEW,
                   ks=KS, ksel=KSEL, rks=RKS, seeds=SEEDS, top_genes=TOP_GENES, p_cover=P_COVER, min_pair=MIN_PAIR,
                   arms=arms, rna_shape=list(X.shape)), open(f'{out}/protocol.json', 'w'), indent=1)
    repro = dict(m_kernel_block_max_abs_diff=0., grid_max_abs_diff={}, pred_max_abs_diff={}, inner_loss_vs_matrix_max_abs_diff={},
                 matrix_pred_max_abs_diff={}, ppanel_pred_max_abs_diff={}, fold_split_identical=True)
    fold_rows, sel_record = [], {}
    folds = (0,) if stage == 'dryrun' else range(5)
    for f in folds:
        total = {k: 0 for k in need}; count = {k: 0 for k in need}
        outer, ometa, bank_info = None, None, {}
        ML = np.load(f'{MATRIX}/{scenario}/fold-{f}_inner_losses.npz', allow_pickle=False)
        for bank in BANKS:
            if stage == 'dryrun' and bank in ('inner-1', 'inner-2', 'outer'):
                continue
            tb = time.perf_counter()
            grids, info = run_bank(S, X, scenario, f, bank, set(need), check_spearman=(stage == 'dryrun'))
            repro['m_kernel_block_max_abs_diff'] = max(repro['m_kernel_block_max_abs_diff'], info['m_kernel_block_max_abs_diff'])
            saved = np.load(f'{EXT}/{scenario}/fold-{f}/{bank}/grids.npz', allow_pickle=False)
            for a, sname in SAVED_NAME.items():
                if a in arms:
                    d = float(np.max(np.abs(np.concatenate([grids[k] for k in arms[a]]) - saved[sname])))
                    repro['grid_max_abs_diff'][a] = max(repro['grid_max_abs_diff'].get(a, 0.), d)
            G = np.load(f'{EXT}/{scenario}/fold-{f}/{bank}/geometry.npz', allow_pickle=False)
            if bank == 'outer':
                outer, ometa = grids, dict(fit=G['fit'], query=G['query'], panel=G['panel'])
            else:
                truth = S['auc'][np.ix_(G['query'], G['panel'])]
                sd = np.maximum(np.nanstd(S['auc'][np.ix_(G['fit'], G['panel'])], axis=0), 1e-8)
                for k, g in grids.items():
                    ll, nn = bank_losses(g, truth, sd); total[k] = total[k] + ll; count[k] = count[k] + nn
                for a, mname in MATRIX_NAME.items():
                    if a in arms:
                        ll, nn = bank_losses(grids[arms[a][0]], truth, sd)
                        d = float(np.max(np.abs(ll - ML[f'{mname}__{bank}__loss'])))
                        assert np.array_equal(nn, ML[f'{mname}__{bank}__count'])
                        repro['inner_loss_vs_matrix_max_abs_diff'][a] = max(repro['inner_loss_vs_matrix_max_abs_diff'].get(a, 0.), d)
            bank_info[f'fold-{f}/{bank}'] = info
            log(event='bank_done', scenario=scenario, fold=f, bank=bank, seconds=round(time.perf_counter() - tb, 1),
                total=round(time.perf_counter() - t0, 1), spearman_seconds=info.get('spearman_seconds'),
                grid_diff=repro['grid_max_abs_diff'], inner_loss_diff=repro['inner_loss_vs_matrix_max_abs_diff'])
        if stage == 'dryrun':
            sels = {a: choose(np.concatenate([total[k] for k in v]), np.concatenate([count[k] for k in v])) for a, v in arms.items()}
            np.savez_compressed(f'{out}/dryrun_fold-0_inner-0_selection.npz', **sels)
            json.dump(bank_info, open(f'{out}/fold-0_bank_info.json', 'w'), indent=1)
            res = dict(scenario=scenario, stage=stage, reproduction=repro, bank_info=bank_info,
                       selection_counts={a: np.bincount(s, minlength=4).tolist() for a, s in sels.items()}, seconds=time.perf_counter() - t0)
            json.dump(res, open(f'{out}/summary.json', 'w'), indent=1)
            log(event='dryrun_done', scenario=scenario, reproduction=repro, seconds=round(time.perf_counter() - t0, 1))
            return
        P = np.load(f'{EXT}/{scenario}/fold-{f}/predictions.npz', allow_pickle=False)
        truth, sd, use = P['truth'], P['train_sd'], P['evaluated']
        assert (P['panel'] == ometa['panel']).all() and (P['held'] == ometa['query']).all() and (P['fit'] == ometa['fit']).all()
        preds, sels = {}, {}
        for a, keys in arms.items():
            tot = np.concatenate([total[k] for k in keys]); cnt = np.concatenate([count[k] for k in keys])
            sel = choose(tot, cnt, fallback=1)
            preds[a] = take(np.concatenate([outer[k] for k in keys]), sel); sels[a] = sel
        for a, sname in SAVED_NAME.items():
            if a in preds:
                d = float(np.max(np.abs(preds[a] - P[sname])))
                repro['pred_max_abs_diff'][a] = max(repro['pred_max_abs_diff'].get(a, 0.), d)
        C = np.load(f'{MATRIX}/{scenario}/fold-{f}_predictions.npz', allow_pickle=False)
        same_split = (C['fit'] == P['fit']).all() and (C['held'] == P['held']).all() and (C['panel'] == P['panel']).all()
        repro['fold_split_identical'] &= bool(same_split)
        for a, mname in MATRIX_NAME.items():
            if a in preds:
                d = float(np.max(np.abs(preds[a] - C['pred__' + mname])))
                repro['matrix_pred_max_abs_diff'][a] = max(repro['matrix_pred_max_abs_diff'].get(a, 0.), d)
        PPd = f'{PPANEL}/{scenario}_main/fold-{f}_predictions.npz'
        if os.path.exists(PPd):
            PP = np.load(PPd, allow_pickle=False)
            assert (PP['fit'] == P['fit']).all() and (PP['held'] == P['held']).all() and (PP['panel'] == P['panel']).all()
            for a in PPANEL_NAME[scenario]:
                if a in preds:
                    d = float(np.max(np.abs(preds[a] - PP[a])))
                    repro['ppanel_pred_max_abs_diff'][a] = max(repro['ppanel_pred_max_abs_diff'].get(a, 0.), d)
        preds['teacher_saved'] = C['pred__M+RNA']
        preds['null_drug'] = np.broadcast_to(np.nanmean(S['auc'][np.ix_(P['fit'], P['panel'])], axis=0), truth.shape).copy()
        for a, v in preds.items():
            mm = metrics(truth, v, sd, use)
            fold_rows.append(dict(fold=f, arm=a, score=mm['score'], spearman=mm['spearman'], centered_mse=mm['centered_mse'],
                                  drugs=int(use.sum()), n_fit=int(len(P['fit'])), n_held=int(len(P['held']))))
        np.savez_compressed(f'{out}/fold-{f}_predictions.npz', truth=truth, train_sd=sd, evaluated=use, fit=P['fit'], held=P['held'],
                            panel=P['panel'], drug_ids=P['drug_ids'], **preds)
        panel_arrays = {f'panel__{b.split("/")[1]}__{k}': np.asarray(v, dtype=np.int64) for b, inf in bank_info.items() for k, v in inf.get('panels', {}).items()}
        np.savez_compressed(f'{out}/fold-{f}_selection.npz', **{a: s for a, s in sels.items()},
                            **{'loss__' + a: np.concatenate([total[k] for k in v]) for a, v in arms.items()},
                            **{'count__' + a: np.concatenate([count[k] for k in v]) for a, v in arms.items()}, **panel_arrays)
        json.dump(bank_info, open(f'{out}/fold-{f}_bank_info.json', 'w'), indent=1)
        sel_record[f] = {a: np.bincount(s, minlength=len(arms[a]) * len(ALPHAS)).tolist() for a, s in sels.items()}
        gate = repro['matrix_pred_max_abs_diff'].get('m_allcand', np.inf) <= GATE
        log(event='fold_done', scenario=scenario, fold=f, total=round(time.perf_counter() - t0, 1), repro=repro, gate_B_passed=gate,
            score_m_only=[r['score'] for r in fold_rows if r['fold'] == f and r['arm'] == 'm_only'][0],
            score_m_allcand=[r['score'] for r in fold_rows if r['fold'] == f and r['arm'] == 'm_allcand'][0])
        if not gate:
            json.dump(dict(scenario=scenario, stage=stage, reproduction=repro, failed_fold=f), open(f'{out}/GATE_FAILED.json', 'w'), indent=1)
            log(event='gate_failed', scenario=scenario, fold=f, repro=repro)
            sys.exit(2)
    with open(f'{out}/fold_metrics.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(fold_rows[0])); w.writeheader(); w.writerows(fold_rows)
    names = list(dict.fromkeys(r['arm'] for r in fold_rows))
    summary = {a: {k: float(np.mean([r[k] for r in fold_rows if r['arm'] == a])) for k in ('score', 'spearman', 'centered_mse')} for a in names}
    fscores = {a: [r['score'] for r in fold_rows if r['arm'] == a] for a in names}
    passed = dict(gate_A=bool(max(list(repro['grid_max_abs_diff'].values()) + list(repro['pred_max_abs_diff'].values())) <= GATE),
                  gate_B=bool(repro['matrix_pred_max_abs_diff'].get('m_allcand', np.inf) <= GATE))
    configs = {a: [f'{k}|alpha{al}' for k in v for al in ALPHAS] for a, v in arms.items()}
    res = dict(scenario=scenario, stage=stage, reproduction=dict(**repro, passed=passed), summary=summary, fold_scores=fscores,
               selection_counts=sel_record, configs=configs, seconds=time.perf_counter() - t0)
    json.dump(res, open(f'{out}/summary.json', 'w'), indent=1)
    log(event='done', scenario=scenario, stage=stage, reproduction=res['reproduction'], seconds=round(time.perf_counter() - t0, 1))


if __name__ == '__main__':
    main()

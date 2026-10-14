"""Acquisition probe, RNA version (pre-registered in PLAN.md): at an equal budget of k measured GENES per query cell,
does per-sample ADAPTIVE choice of which genes to measure recover more than a static or random panel?

Mirrors the protein probe (acquisition-probe-20261009-0001/code/acquisition_probe.py) with the borrowed layer changed
from P to RNA. Model pieces (predict_grid / bank_losses / choose / take / metrics / kernel_state / rbf / generator /
stable_seed) are copied verbatim from the reference implementations (protein probe, crosslayer.py).
Reads only: extra-supervision banks, source-inputs.npz, rna421, the matrix run's saved M+RNA / M+vRNA predictions and
inner losses (reproduction only). Writes only to OUT_DIR.

usage: python acquisition_probe_rna.py SCENARIO STAGE OUT_DIR
  STAGE = repro   : reproduce saved m_only arms and the matrix run's M+RNA (k=all) and M+vRNA (matrix recipe) outer
                    predictions; no new arms
  STAGE = dryrun  : fold-0 inner banks only (generator alpha + head alpha selection, inner-loss check); no outer bank
  STAGE = main    : all folds, all pre-registered arms
  STAGE = analyze : SCENARIO ignored; OUT_DIR is the parent holding new_cell_main/ and tissue_main/
"""
import csv
import hashlib
import json
import os
import sys
import time
import warnings
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, spearmanr, t as student_t

RUN = os.environ.get('CREED_ROOT', '/public/home/mengxl/creed-research-runs')   # env override: synthetic smoke test only
EXT = RUN + '/extra-supervision-20261008-0001/output'
EXTRA = RUN + '/extra-supervision-20261008-0001/resources/m_gdsc_extra19.npz'
SOURCE = RUN + '/gdsc-metabolite-full-protein-20260908-0001/output/result/source-inputs.npz'
RNA421 = RUN + '/rna421-20260906-0002'
MATRIX = RUN + '/crosslayer-matrix-20261009-0001/output'
SOURCE_SHA = '52d71517a8952e83a1603cd831f09449544102d0ca5ff829ff01be9bf79cd637'
EXTRA_SHA = 'd4e7dc603f491217805a1d06b673f8c2662a8c0f2ad77ec875f7d7868526384c'
RNA_SHA = '5eb953ea12089eea6cdc5a7892646d3b75532429cfbeeefa8832d9b4a5f718cf'
# registered matrix-run headline gains (mean over folds of per-fold relative improvement, %) — PLAN section 3
MATRIX_TARGETS = {'new_cell': dict(headroom=3.078810120332858, recovered=1.0555808085535539),
                  'tissue': dict(headroom=2.1947192412669287, recovered=0.716058946743318)}
SCEN_ID = {'new_cell': 0, 'tissue': 1}
BANK_ID = {'inner-0': 0, 'inner-1': 1, 'inner-2': 2, 'outer': 3}
BANKS = ('inner-0', 'inner-1', 'inner-2', 'outer')

ALPHAS = (.1, 1., 10., 100.)
LAMBDAS = (.1, 1., 10., 100.)
KS = (8, 16, 32, 64, 128, 256)
SEEDS = (0, 1, 2, 3, 4)
TOP_GENES, TOP_TARGET, OOF_FOLDS = 3000, 512, 3
MIN_PAIR = 10
N_SUB = 3
HEADS = ('H_meas', 'H_virt')
STATIC_POLICIES = ('static', 'variance', 'mean_expr', 'oracle_static') + tuple(f'random_s{s}' for s in SEEDS)
CELL_POLICIES = ('adaptive', 'adaptive_surprise')
POLICIES = STATIC_POLICIES + CELL_POLICIES
PRIMARY_SAVE = {('H_meas', 'virtual', 0), ('H_meas', 'measured_all', 'all'), ('H_virt', 'virtual', 0), ('H_virt', 'measured_all', 'all')}
for _h in HEADS:
    for _k in (16, 32):
        for _p in ('adaptive', 'static', 'variance', 'mean_expr') + tuple(f'random_s{s}' for s in SEEDS):
            PRIMARY_SAVE.add((_h, _p, _k))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- model pieces (copied from reference)
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
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sp = float(np.nanmean(rho[use]))
    return dict(score=float(np.sqrt(mse[use].mean())), spearman=sp,
                centered_mse=float((mse - bias * bias)[use].mean()), mse=mse)


def spearman_scores(X, Y, min_n=MIN_PAIR):
    """sum_j |Spearman(X[:, i], Y[:, j])| over the rows where Y[:, j] is observed (X has no missing values);
    drugs with n<min_n add 0; a constant side gives 0. Drugs sharing an observation pattern share one ranking."""
    c = X.shape[1]; tot = np.zeros(c); groups = {}
    for j in range(Y.shape[1]):
        groups.setdefault(np.isfinite(Y[:, j]).tobytes(), []).append(j)
    for cols in groups.values():
        o = np.isfinite(Y[:, cols[0]])
        if o.sum() < min_n:
            continue
        rx = rankdata(X[o], axis=0); dx = rx - rx.mean(0); sx = np.sqrt((dx * dx).sum(0))
        for j in cols:
            ry = rankdata(Y[o, j]); dy = ry - ry.mean(); sy = np.sqrt((dy * dy).sum())
            den = sx * sy
            ok = den > 1e-12
            r = np.zeros(c); r[ok] = (dx[:, ok] * dy[:, None]).sum(0) / den[ok]
            tot += np.abs(r)
    return tot


# ---------------------------------------------------------------- matrix-run pieces (copied verbatim from crosslayer.py)
def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'little')


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


def matrix_generator(KX, HX, Z, seed):
    """Cross-fitted KRR X->Z inside fit (matrix recipe). OOF predictions for fit rows (3 folds), full-fit generator
    for query rows; lambda chosen inside fit by mean OOF MSE over target columns."""
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


def matrix_vrna_grid(B, scenario, f, bank):
    """matrix run's M+vRNA arm on this bank: generator on the top-512 standardized candidate genes, virtual kernel on OOF."""
    Z = B['Z_fit'][:, :TOP_TARGET]           # candidates are variance-descending, so the first 512 are the top-512
    oof, zq, lam, r2m, r2med = matrix_generator(B['KM'], B['HM'], Z, stable_seed('oof', scenario, f, bank))
    Kv, Hv, st = rbf_pair(oof, zq)
    return predict_grid(.5 * B['KM'] + .5 * Kv, .5 * B['HM'] + .5 * Hv, B['y']), dict(lam=lam, oof_r2_mean=r2m, oof_r2_median=r2med, virtual_width=st[2])


# ---------------------------------------------------------------- probe virtual-RNA generator (mirrors protein probe)
def sub_split(n, scenario, f, bank):
    rng = np.random.default_rng([20261009, SCEN_ID[scenario], f, BANK_ID[bank], 7])
    return np.array_split(rng.permutation(n), N_SUB)


def krr_fit_predict(K_tr, Y_tr, alpha, *H_list):
    mean = Y_tr.mean(0)
    coef = cho_solve(cho_factor(K_tr + alpha * np.eye(len(K_tr)), lower=True), Y_tr - mean)
    return [h @ coef + mean for h in H_list]


def generator(KM, HM, Z_fit, alpha, parts):
    """3 sub-generators on 2/3 of the fit rows each. Returns OOF virtual for fit rows (n_fit x c) and the
    three sub-generator predictions for query rows (3 x n_q x c)."""
    n = len(Z_fit); V_fit = np.empty_like(Z_fit); V_q = np.empty((N_SUB, len(HM), Z_fit.shape[1]))
    for g, part in enumerate(parts):
        tr = np.setdiff1d(np.arange(n), part)
        oof, q = krr_fit_predict(KM[np.ix_(tr, tr)], Z_fit[tr], alpha, KM[np.ix_(part, tr)], HM[:, tr])
        V_fit[part] = oof; V_q[g] = q
    return V_fit, V_q


# ---------------------------------------------------------------- response head (fixed per bank)
class Head:
    """0.5*K_M + 0.5*RBF(R_fit) response head. R_fit given in the units the head is fitted on; queries in the same units.
    Standardization by fit moments of R_fit, median positive squared distance bandwidth (same recipe as kernel_state)."""

    def __init__(self, KM, HM, R_fit, y):
        mu = R_fit.mean(0); sd = R_fit.std(0); self.mu, self.sd = mu, np.where(sd < 1e-8, 1., sd)
        self.Z = (R_fit - self.mu) / self.sd
        d = cdist(self.Z, self.Z, 'sqeuclidean'); pos = d[np.triu_indices(len(self.Z), 1)]; pos = pos[pos > 0]
        self.width = float(np.median(pos)) if len(pos) else None
        KR = np.exp(-d / (2 * self.width)) if self.width else np.ones_like(d)
        self.K = .5 * KM + .5 * KR; self.HM = HM; self.y = y
        self.C = None

    def cross(self, Q):
        HR = np.exp(-cdist((Q - self.mu) / self.sd, self.Z, 'sqeuclidean') / (2 * self.width)) if self.width else np.ones((len(Q), len(self.Z)))
        return .5 * self.HM + .5 * HR

    def grid(self, Q):
        return predict_grid(self.K, self.cross(Q), self.y)

    def freeze(self, sel):
        """per-drug alpha index -> combined dual coefficient matrix (n_fit x n_drugs) and offsets; prediction = H @ C + m."""
        y = self.y; n, m = y.shape; C = np.zeros((n, m)); off = np.empty(m); groups = {}
        for j in range(m):
            groups.setdefault(np.isfinite(y[:, j]).tobytes(), []).append(j)
        for cols in groups.values():
            obs = np.flatnonzero(np.isfinite(y[:, cols[0]])); yy = y[np.ix_(obs, cols)]; mean = yy.mean(0)
            off[cols] = mean
            if len(obs) < 20:
                continue
            kk = self.K[np.ix_(obs, obs)]
            for ai, alpha in enumerate(ALPHAS):
                sub = [c for c in cols if sel[c] == ai]
                if not sub:
                    continue
                coef = cho_solve(cho_factor(kk + alpha * np.eye(len(obs)), lower=True), y[np.ix_(obs, sub)] - off[sub])
                C[np.ix_(obs, sub)] = coef
        self.C, self.off = C, off

    def predict(self, Q):
        return self.cross(Q) @ self.C + self.off


# ---------------------------------------------------------------- bank preparation
def load_bank(S, X, scenario, f, bank):
    G = np.load(f'{EXT}/{scenario}/fold-{f}/{bank}/geometry.npz', allow_pickle=False)
    fit, query, panel = G['fit'], G['query'], G['panel']; n = len(fit)
    KM = G['m_only_kernel'][:n, :n].copy(); HM = G['m_only_cross'][:, :n].copy()
    # kernel block check: saved M kernel block == RBF median kernel recomputed from the raw M of the fit rows
    Km, Hm, _ = rbf_pair(S['m'][fit], S['m'][query])
    kdiff = max(float(np.max(np.abs(Km - KM))), float(np.max(np.abs(Hm - HM))))
    np.testing.assert_allclose(Km, KM, rtol=1e-11, atol=1e-12); np.testing.assert_allclose(Hm, HM, rtol=1e-11, atol=1e-12)
    B = dict(fit=fit, query=query, panel=panel, n=n, G=G, KM=KM, HM=HM, y=S['auc'][np.ix_(fit, panel)], kernel_block_max_abs_diff=kdiff)
    # candidate pool: fit-only top-3000 variance genes (matrix recipe), fit standardization
    xf = X[fit]
    cand = np.argsort(-xf.var(0), kind='stable')[:TOP_GENES]
    Xf = xf[:, cand]; Xq = X[np.ix_(query, cand)]
    mu = Xf.mean(0); sd = Xf.std(0); sd = np.where(sd < 1e-8, 1., sd)
    B.update(cand=cand, cvar=Xf.var(0), cmean=Xf.mean(0), mu=mu, sd=sd, X_fit_raw=Xf, Xq_raw=Xq,
             Z_fit=(Xf - mu) / sd, Zq=(Xq - mu) / sd)
    return B


def generator_loss(B, scenario, f, bank):
    """inner-bank RNA-reconstruction loss (standardized query genes, all entries) of the 3-sub-generator mean, per alpha."""
    parts = sub_split(B['n'], scenario, f, bank); out = []
    for alpha in ALPHAS:
        _, V_q = generator(B['KM'], B['HM'], B['Z_fit'], alpha, parts)
        out.append(float(np.mean((V_q.mean(0) - B['Zq']) ** 2)))
    return out


def build_heads(B, scenario, f, bank, alpha_gen):
    parts = sub_split(B['n'], scenario, f, bank)
    V_fit, V_q3 = generator(B['KM'], B['HM'], B['Z_fit'], alpha_gen, parts)
    B.update(V_fit=V_fit, V_q3=V_q3, V_q=V_q3.mean(0), U_q=V_q3.std(0))
    B['H_meas'] = Head(B['KM'], B['HM'], B['X_fit_raw'], B['y'])
    B['H_virt'] = Head(B['KM'], B['HM'], V_fit, B['y'])


def head_input(B, head, Q_std):
    """Q_std: query RNA in measured-standardized units (virtual mean with acquired entries overridden)."""
    return Q_std * B['sd'] + B['mu'] if head == 'H_meas' else Q_std


def repro_grids(E, B):
    """saved-arm grids on this bank: original__m_only (fit block) and extra__m_only (full saved kernel + extra19 labels)."""
    G = B['G']; ei = G['extra_indices']
    y_ext = np.vstack([B['y'], E['auc'][np.ix_(ei, B['panel'])]])
    assert G['m_only_kernel'].shape == (B['n'] + len(ei), B['n'] + len(ei))
    return {'original__m_only': predict_grid(B['KM'], B['HM'], B['y']),
            'extra__m_only': predict_grid(G['m_only_kernel'], G['m_only_cross'], y_ext)}


# ---------------------------------------------------------------- outer-bank policies
def policy_indices(B, S, X, GS, f, scenario, info):
    """returns dict policy -> dict k -> index array (k,) for static policies or (n_q, k) for per-cell policies."""
    fit, cand, y = B['fit'], B['cand'], B['y']; n_q = len(B['query'])
    t0 = time.perf_counter()
    w = spearman_scores(B['X_fit_raw'], y)
    t1 = time.perf_counter()
    w_or = spearman_scores(X[:, cand], S['auc'][:, B['panel']])
    info.update(spearman_seconds=round(t1 - t0, 1), oracle_spearman_seconds=round(time.perf_counter() - t1, 1))
    orders = {'static': np.argsort(-w, kind='stable'), 'variance': np.argsort(-B['cvar'], kind='stable'),
              'mean_expr': np.argsort(-B['cmean'], kind='stable'), 'oracle_static': np.argsort(-w_or, kind='stable')}
    for s in SEEDS:
        rng = np.random.default_rng([20261009, SCEN_ID[scenario], f, BANK_ID['outer'], s])
        orders[f'random_s{s}'] = rng.permutation(len(cand))
    A = w[None, :] * B['U_q']; Su = w[None, :] * np.abs(B['V_q'])
    cell_orders = {'adaptive': np.argsort(-A, axis=1, kind='stable'), 'adaptive_surprise': np.argsort(-Su, axis=1, kind='stable')}
    idx = {}
    for pol, o in orders.items():
        idx[pol] = {k: o[:k] for k in KS}
    for pol, o in cell_orders.items():
        idx[pol] = {k: o[:, :k] for k in KS}
    sym = GS[cand]
    info.update(w_static_top16=sym[orders['static'][:16]].tolist(), w_static_top16_score=w[orders['static'][:16]].round(5).tolist(),
                oracle_top16=sym[orders['oracle_static'][:16]].tolist(), variance_top16=sym[orders['variance'][:16]].tolist(),
                mean_expr_top16=sym[orders['mean_expr'][:16]].tolist(),
                static_oracle_overlap16=int(len(np.intersect1d(orders['static'][:16], orders['oracle_static'][:16]))),
                static_oracle_overlap64=int(len(np.intersect1d(orders['static'][:64], orders['oracle_static'][:64]))),
                static_variance_overlap64=int(len(np.intersect1d(orders['static'][:64], orders['variance'][:64]))),
                u_q_mean=float(B['U_q'].mean()), u_q_median=float(np.median(B['U_q'])),
                corr_w_meanU=float(np.corrcoef(w, B['U_q'].mean(0))[0, 1]), corr_w_var=float(np.corrcoef(w, B['cvar'])[0, 1]),
                w_mean=float(w.mean()), w_max=float(w.max()))
    for pol in CELL_POLICIES:
        chosen = idx[pol][16]; counts = np.bincount(chosen.ravel(), minlength=len(cand))
        top = np.argsort(-counts, kind='stable')[:16]
        info[f'{pol}_k16_most_chosen'] = [(str(sym[t]), int(counts[t])) for t in top]
        info[f'{pol}_k16_distinct_genes'] = int((counts > 0).sum())
        info[f'{pol}_k16_mean_overlap_with_static16'] = float(np.mean([len(np.intersect1d(r, orders['static'][:16])) for r in chosen]))
        info[f'{pol}_k16_mean_overlap_with_static64'] = float(np.mean([len(np.intersect1d(r, orders['static'][:64])) for r in chosen]))
    return idx, w


def build_query(B, sel):
    Q = B['V_q'].copy()
    if sel is None:
        return Q
    if sel.ndim == 1:
        Q[:, sel] = B['Zq'][:, sel]
    else:
        rows = np.arange(len(Q))[:, None]; Q[rows, sel] = B['Zq'][rows, sel]
    return Q


# ---------------------------------------------------------------- driver
def run_fold(S, X, GS, E, scenario, f, stage, repro, bank_info, log):
    t0 = time.perf_counter()
    banks = {b: load_bank(S, X, scenario, f, b) for b in (BANKS[:3] if stage == 'dryrun' else BANKS)}
    for b, B in banks.items():
        bank_info[f'fold-{f}/{b}'] = dict(n_fit=int(B['n']), n_query=int(len(B['query'])), n_panel=int(len(B['panel'])),
                                          candidate_count=int(len(B['cand'])), top_gene=str(GS[B['cand'][0]]),
                                          kernel_block_max_abs_diff=B['kernel_block_max_abs_diff'])
        repro['kernel_block_max_abs_diff'] = max(repro.get('kernel_block_max_abs_diff', 0.), B['kernel_block_max_abs_diff'])
    inner = {}
    for b in BANKS[:3]:
        B = banks[b]
        inner[b] = dict(truth=S['auc'][np.ix_(B['query'], B['panel'])], sd=np.maximum(np.nanstd(B['y'], axis=0), 1e-8))
    # --- saved m_only arms (reproduction) on every bank
    saved_arms = ('original__m_only', 'extra__m_only')
    total = {a: 0 for a in saved_arms}; count = {a: 0 for a in saved_arms}; outer_saved = {}
    if stage in ('repro', 'main'):
        for b, B in banks.items():
            grids = repro_grids(E, B)
            saved = np.load(f'{EXT}/{scenario}/fold-{f}/{b}/grids.npz', allow_pickle=False)
            for a, g in grids.items():
                d = float(np.max(np.abs(g - saved[a])))
                repro['grid_max_abs_diff'][a] = max(repro['grid_max_abs_diff'].get(a, 0.), d)
                if b == 'outer':
                    outer_saved[a] = g
                else:
                    ll, nn = bank_losses(g, inner[b]['truth'], inner[b]['sd']); total[a] = total[a] + ll; count[a] = count[a] + nn
    # --- matrix-recipe M+vRNA arm (reference / reproduction) and H_meas k=all grids on inner banks vs matrix inner losses
    mv_tot = 0; mv_cnt = 0; mv_outer = None
    ML = np.load(f'{MATRIX}/{scenario}/fold-{f}_inner_losses.npz', allow_pickle=False)
    for b, B in banks.items():
        g, ginfo = matrix_vrna_grid(B, scenario, f, b)
        bank_info[f'fold-{f}/{b}']['matrix_generator'] = ginfo
        if b == 'outer':
            mv_outer = g
        else:
            ll, nn = bank_losses(g, inner[b]['truth'], inner[b]['sd']); mv_tot = mv_tot + ll; mv_cnt = mv_cnt + nn
            d = float(np.max(np.abs(ll - ML[f'M+vRNA__{b}__loss'])))
            repro['inner_loss_max_abs_diff_vs_matrix']['M+vRNA'] = max(repro['inner_loss_max_abs_diff_vs_matrix'].get('M+vRNA', 0.), d)
    sel_mv = choose(mv_tot, mv_cnt)
    # H_meas (measured RNA head) alpha selection: inner query input = measured candidate genes (k = all) == matrix M+RNA
    hm_tot = 0; hm_cnt = 0
    for b in BANKS[:3]:
        B = banks[b]; H = Head(B['KM'], B['HM'], B['X_fit_raw'], B['y'])
        ll, nn = bank_losses(H.grid(B['Xq_raw']), inner[b]['truth'], inner[b]['sd']); hm_tot = hm_tot + ll; hm_cnt = hm_cnt + nn
        d = float(np.max(np.abs(ll - ML[f'M+RNA__{b}__loss'])))
        repro['inner_loss_max_abs_diff_vs_matrix']['M+RNA'] = max(repro['inner_loss_max_abs_diff_vs_matrix'].get('M+RNA', 0.), d)
        bank_info[f'fold-{f}/{b}']['head_meas_width'] = H.width
    sel_hm = choose(hm_tot, hm_cnt)
    if stage == 'repro':
        P = np.load(f'{EXT}/{scenario}/fold-{f}/predictions.npz', allow_pickle=False)
        truth, sd, use = P['truth'], P['train_sd'], P['evaluated']
        rows = []
        for a in saved_arms:
            pred = take(outer_saved[a], choose(total[a], count[a]))
            repro['pred_max_abs_diff'][a] = max(repro['pred_max_abs_diff'].get(a, 0.), float(np.max(np.abs(pred - P[a]))))
            rows.append(dict(fold=f, arm=a, score=metrics(truth, pred, sd, use)['score'], score_of_saved_pred=metrics(truth, P[a], sd, use)['score']))
        # matrix two ends
        MP = np.load(f'{MATRIX}/{scenario}/fold-{f}_predictions.npz', allow_pickle=False)
        assert np.array_equal(MP['fit'], P['fit']) and np.array_equal(MP['held'], P['held']) and np.array_equal(MP['panel'], P['panel'])
        B = banks['outer']; H = Head(B['KM'], B['HM'], B['X_fit_raw'], B['y']); H.freeze(sel_hm)
        pred_all = H.predict(B['Xq_raw']); pred_mv = take(mv_outer, sel_mv)
        for a, pred, key, selk, mysel in (('M+RNA', pred_all, 'pred__M+RNA', 'sel__M+RNA', sel_hm), ('M+vRNA', pred_mv, 'pred__M+vRNA', 'sel__M+vRNA', sel_mv)):
            repro['matrix_pred_max_abs_diff'][a] = max(repro['matrix_pred_max_abs_diff'].get(a, 0.), float(np.max(np.abs(pred - MP[key]))))
            repro['matrix_alpha_selection_mismatches'][a] = repro['matrix_alpha_selection_mismatches'].get(a, 0) + int((mysel != MP[selk]).sum())
            rows.append(dict(fold=f, arm=a, score=metrics(truth, pred, sd, use)['score'], score_of_saved_pred=metrics(truth, MP[key], sd, use)['score']))
        rows.append(dict(fold=f, arm='M', score=metrics(truth, take(outer_saved['original__m_only'], choose(total['original__m_only'], count['original__m_only'])), sd, use)['score'],
                         score_of_saved_pred=metrics(truth, P['original__m_only'], sd, use)['score']))
        log(dict(event='fold_done', fold=f, seconds=round(time.perf_counter() - t0, 1)))
        return rows
    # --- generator alpha (inner banks only)
    gl = np.zeros(len(ALPHAS))
    for b in BANKS[:3]:
        l = generator_loss(banks[b], scenario, f, b); gl += np.array(l); bank_info[f'fold-{f}/{b}']['generator_loss'] = l
    ai = int(np.argmin(gl)); alpha_gen = ALPHAS[ai]
    bank_info[f'fold-{f}'] = dict(generator_alpha=alpha_gen, generator_loss_sum=gl.round(6).tolist())
    log(dict(event='generator_alpha', fold=f, alpha=alpha_gen, loss_sum=gl.round(5).tolist(), seconds=round(time.perf_counter() - t0, 1)))
    # --- heads on every bank; per-drug alpha on inner banks
    htot = {h: 0 for h in HEADS}; hcnt = {h: 0 for h in HEADS}
    for b, B in banks.items():
        build_heads(B, scenario, f, b, alpha_gen)
        bank_info[f'fold-{f}/{b}'].update(virtual_oof_r2=float(1 - np.mean((B['V_fit'] - B['Z_fit']) ** 2)),
                                           virtual_query_mse=float(np.mean((B['V_q'] - B['Zq']) ** 2)),
                                           head_meas_width=B['H_meas'].width, head_virt_width=B['H_virt'].width)
        if b == 'outer':
            continue
        for h in HEADS:
            Q = B['Zq'] if h == 'H_meas' else B['V_q']
            g = B[h].grid(head_input(B, h, Q))
            ll, nn = bank_losses(g, inner[b]['truth'], inner[b]['sd']); htot[h] = htot[h] + ll; hcnt[h] = hcnt[h] + nn
        log(dict(event='bank_done', fold=f, bank=b, seconds=round(time.perf_counter() - t0, 1)))
    sel = {h: choose(htot[h], hcnt[h]) for h in HEADS}
    assert np.array_equal(sel['H_meas'], sel_hm), 'H_meas selection differs between the two equivalent code paths'
    sel['original__m_only'] = choose(total['original__m_only'], count['original__m_only']) if stage == 'main' else None
    bank_info[f'fold-{f}']['alpha_counts'] = {h: np.bincount(sel[h], minlength=len(ALPHAS)).tolist() for h in HEADS}
    bank_info[f'fold-{f}']['alpha_counts']['matrix_vRNA'] = np.bincount(sel_mv, minlength=len(ALPHAS)).tolist()
    if stage == 'dryrun':
        log(dict(event='dryrun_fold_done', fold=f, alpha_counts=bank_info[f'fold-{f}']['alpha_counts'], seconds=round(time.perf_counter() - t0, 1)))
        return []
    # --- outer bank
    B = banks['outer']; info = bank_info[f'fold-{f}/outer']
    P = np.load(f'{EXT}/{scenario}/fold-{f}/predictions.npz', allow_pickle=False)
    truth, sd, use = P['truth'], P['train_sd'], P['evaluated']
    assert np.array_equal(P['fit'], B['fit']) and np.array_equal(P['held'], B['query']) and np.array_equal(P['panel'], B['panel'])
    MP = np.load(f'{MATRIX}/{scenario}/fold-{f}_predictions.npz', allow_pickle=False)
    assert np.array_equal(MP['fit'], P['fit']) and np.array_equal(MP['held'], P['held']) and np.array_equal(MP['panel'], P['panel'])
    for h in HEADS:
        B[h].freeze(sel[h])
    idx, w = policy_indices(B, S, X, GS, f, scenario, info)
    preds = {}
    pred_m = take(outer_saved['original__m_only'], sel['original__m_only'])
    repro['pred_max_abs_diff']['original__m_only'] = max(repro['pred_max_abs_diff'].get('original__m_only', 0.), float(np.max(np.abs(pred_m - P['original__m_only']))))
    pred_e = take(outer_saved['extra__m_only'], choose(total['extra__m_only'], count['extra__m_only']))
    repro['pred_max_abs_diff']['extra__m_only'] = max(repro['pred_max_abs_diff'].get('extra__m_only', 0.), float(np.max(np.abs(pred_e - P['extra__m_only']))))
    preds[('none', 'm_only', 'M')] = pred_m
    preds[('none', 'null_drug', 'none')] = np.broadcast_to(np.nanmean(B['y'], axis=0), truth.shape).copy()
    preds[('none', 'matrix_vRNA', 'ref')] = take(mv_outer, sel_mv)
    for h in HEADS:
        preds[(h, 'virtual', 0)] = B[h].predict(head_input(B, h, build_query(B, None)))
        preds[(h, 'measured_all', 'all')] = B[h].predict(head_input(B, h, B['Zq']))
        for pol in POLICIES:
            for k in KS:
                preds[(h, pol, k)] = B[h].predict(head_input(B, h, build_query(B, idx[pol][k])))
    for a, key, selk, mysel in (('M+RNA', ('H_meas', 'measured_all', 'all'), 'sel__M+RNA', sel['H_meas']), ('M+vRNA', ('none', 'matrix_vRNA', 'ref'), 'sel__M+vRNA', sel_mv)):
        repro['matrix_pred_max_abs_diff'][a] = max(repro['matrix_pred_max_abs_diff'].get(a, 0.), float(np.max(np.abs(preds[key] - MP['pred__' + a]))))
        repro['matrix_alpha_selection_mismatches'][a] = repro['matrix_alpha_selection_mismatches'].get(a, 0) + int((mysel != MP[selk]).sum())
    rows = []
    for (h, pol, k), v in preds.items():
        m = metrics(truth, v, sd, use)
        rows.append(dict(scenario=scenario, head=h, policy=pol, k=k, fold=f, score=m['score'], spearman=m['spearman'], centered_mse=m['centered_mse']))
    for h in HEADS:
        for k in KS:
            rs = [r for r in rows if r['head'] == h and r['k'] == k and r['policy'].startswith('random_s')]
            rows.append(dict(scenario=scenario, head=h, policy='random_mean', k=k, fold=f, score=float(np.mean([r['score'] for r in rs])),
                             spearman=float(np.mean([r['spearman'] for r in rs])), centered_mse=float(np.mean([r['centered_mse'] for r in rs]))))
    np.savez_compressed(f'{OUT}/fold-{f}_predictions.npz', truth=truth, train_sd=sd, evaluated=use, fit=P['fit'], held=P['held'], panel=P['panel'],
                        **{f'{h}|{pol}|{k}': v for (h, pol, k), v in preds.items() if (h, pol, k) in PRIMARY_SAVE or h == 'none'})
    np.savez_compressed(f'{OUT}/fold-{f}_selection.npz', alpha_H_meas=sel['H_meas'], alpha_H_virt=sel['H_virt'], alpha_m_only=sel['original__m_only'],
                        alpha_matrix_vRNA=sel_mv, generator_alpha=alpha_gen, candidates=B['cand'], w_static=w,
                        adaptive_k16=idx['adaptive'][16], adaptive_k32=idx['adaptive'][32], adaptive_surprise_k16=idx['adaptive_surprise'][16],
                        static_k256=idx['static'][256], variance_k256=idx['variance'][256], mean_expr_k256=idx['mean_expr'][256], oracle_k256=idx['oracle_static'][256],
                        U_q=B['U_q'].astype(np.float32), V_q=B['V_q'].astype(np.float32),
                        **{f'random_s{s}_k256': idx[f'random_s{s}'][256] for s in SEEDS})
    log(dict(event='fold_done', fold=f, seconds=round(time.perf_counter() - t0, 1)))
    return rows


def load_ctx():
    if os.environ.get('CREED_SKIP_SHA') != '1':   # synthetic smoke test only
        assert sha(SOURCE) == SOURCE_SHA, 'source sha mismatch'
        assert sha(EXTRA) == EXTRA_SHA, 'extra19 sha mismatch'
        assert sha(RNA421 + '/rna_log_cpm.npy') == RNA_SHA, 'rna sha mismatch'
    S = np.load(SOURCE, allow_pickle=False)
    S = {k: S[k] for k in ('m', 'auc', 'tissues', 'sample_ids', 'drug_ids')}
    X = np.load(RNA421 + '/rna_log_cpm.npy')
    assert (np.load(RNA421 + '/sample_ids.npy') == S['sample_ids']).all(), 'RNA row order differs from source'
    assert X.shape[0] == len(S['sample_ids']) and X.shape[1] >= TOP_GENES and np.isfinite(X).all() and np.isfinite(S['m']).all()
    GS = np.load(RNA421 + '/gene_symbols.npy', allow_pickle=False); assert len(GS) == X.shape[1]
    E = np.load(EXTRA, allow_pickle=False); assert (E['drug_ids'] == S['drug_ids']).all(); E = {'auc': E['auc']}
    return S, X, GS, E


def gain_pct(new, base):
    n, b = np.asarray(new), np.asarray(base)
    return float(((b - n) / b * 100).mean())


def main_stage(scenario, stage):
    t0 = time.perf_counter()
    logf = open(f'{OUT}/stage.log', 'a')

    def log(d):
        s = json.dumps(d); print(s, flush=True); logf.write(s + '\n'); logf.flush()
    S, X, GS, E = load_ctx()
    json.dump(dict(scenario=scenario, stage=stage, script_sha256=sha(__file__), source_sha256=SOURCE_SHA, extra_sha256=EXTRA_SHA, rna_sha256=RNA_SHA,
                   alphas=ALPHAS, lambdas=LAMBDAS, ks=KS, seeds=SEEDS, top_genes=TOP_GENES, top_target=TOP_TARGET, n_sub=N_SUB, heads=HEADS,
                   policies=POLICIES, matrix_targets=MATRIX_TARGETS), open(f'{OUT}/protocol.json', 'w'), indent=1)
    repro = dict(grid_max_abs_diff={}, pred_max_abs_diff={}, matrix_pred_max_abs_diff={}, matrix_alpha_selection_mismatches={}, inner_loss_max_abs_diff_vs_matrix={})
    bank_info, rows = {}, []
    folds = (0,) if stage == 'dryrun' else range(5)
    for f in folds:
        rows += run_fold(S, X, GS, E, scenario, f, stage, repro, bank_info, log)
    json.dump(bank_info, open(f'{OUT}/bank_info.json', 'w'), indent=1)
    if stage == 'dryrun':
        repro['passed_inner'] = bool(max(repro['grid_max_abs_diff'].values(), default=0.) <= 1e-8 and max(repro['inner_loss_max_abs_diff_vs_matrix'].values()) <= 1e-8)
        json.dump(dict(scenario=scenario, stage=stage, reproduction=repro, seconds=time.perf_counter() - t0), open(f'{OUT}/summary.json', 'w'), indent=1)
        log(dict(event='dryrun_done', reproduction=repro, seconds=round(time.perf_counter() - t0, 1)))
        return
    if stage == 'repro':
        saved = {}
        with open(f'{EXT}/{scenario}/fold_metrics.csv') as fh:
            for r in csv.DictReader(fh):
                saved[(int(r['fold']), r['arm'])] = float(r['score'])
        md = 0.; md2 = 0.
        for r in rows:
            if r['arm'] in ('original__m_only', 'extra__m_only'):
                md = max(md, abs(r['score'] - saved[(r['fold'], r['arm'])])); md2 = max(md2, abs(r['score_of_saved_pred'] - saved[(r['fold'], r['arm'])]))
        repro['fold_score_max_abs_diff_my_pred_vs_saved_csv'] = md
        repro['fold_score_max_abs_diff_my_metric_on_saved_pred_vs_saved_csv'] = md2
        fs = lambda a: [r['score'] for r in sorted(rows, key=lambda r: r['fold']) if r['arm'] == a]
        g_head = gain_pct(fs('M+RNA'), fs('M')); g_rec = gain_pct(fs('M+vRNA'), fs('M'))
        repro['matrix_gain_pct'] = {'headroom_M+RNA_vs_M': g_head, 'recovered_M+vRNA_vs_M': g_rec,
                                    'headroom_abs_diff_vs_registered': abs(g_head - MATRIX_TARGETS[scenario]['headroom']),
                                    'recovered_abs_diff_vs_registered': abs(g_rec - MATRIX_TARGETS[scenario]['recovered'])}
        repro['passed'] = bool(max(repro['pred_max_abs_diff'].values()) <= 1e-8 and max(repro['grid_max_abs_diff'].values()) <= 1e-8
                               and max(repro['matrix_pred_max_abs_diff'].values()) <= 1e-6 and max(repro['inner_loss_max_abs_diff_vs_matrix'].values()) <= 1e-8
                               and repro['matrix_gain_pct']['headroom_abs_diff_vs_registered'] <= 1e-6 and repro['matrix_gain_pct']['recovered_abs_diff_vs_registered'] <= 1e-6)
        json.dump(dict(scenario=scenario, stage=stage, reproduction=repro, rows=rows, seconds=time.perf_counter() - t0), open(f'{OUT}/summary.json', 'w'), indent=1)
        log(dict(event='done', reproduction=repro, seconds=round(time.perf_counter() - t0, 1)))
        return
    with open(f'{OUT}/fold_metrics.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['scenario', 'head', 'policy', 'k', 'fold', 'score', 'spearman', 'centered_mse']); w.writeheader(); w.writerows(rows)
    fs = lambda h, pol, k: [r['score'] for r in sorted(rows, key=lambda r: r['fold']) if r['head'] == h and r['policy'] == pol and str(r['k']) == str(k)]
    g_head = gain_pct(fs('H_meas', 'measured_all', 'all'), fs('none', 'm_only', 'M')); g_rec = gain_pct(fs('none', 'matrix_vRNA', 'ref'), fs('none', 'm_only', 'M'))
    repro['matrix_gain_pct'] = {'headroom_M+RNA_vs_M': g_head, 'recovered_M+vRNA_vs_M': g_rec,
                                'headroom_abs_diff_vs_registered': abs(g_head - MATRIX_TARGETS[scenario]['headroom']),
                                'recovered_abs_diff_vs_registered': abs(g_rec - MATRIX_TARGETS[scenario]['recovered'])}
    repro['passed'] = bool(max(repro['pred_max_abs_diff'].values()) <= 1e-8 and max(repro['grid_max_abs_diff'].values()) <= 1e-8
                           and max(repro['matrix_pred_max_abs_diff'].values()) <= 1e-6 and max(repro['inner_loss_max_abs_diff_vs_matrix'].values()) <= 1e-8
                           and repro['matrix_gain_pct']['headroom_abs_diff_vs_registered'] <= 1e-6 and repro['matrix_gain_pct']['recovered_abs_diff_vs_registered'] <= 1e-6)
    keys = sorted({(r['head'], r['policy'], str(r['k'])) for r in rows})
    summary = {}
    for h, pol, k in keys:
        rs = sorted([r for r in rows if r['head'] == h and r['policy'] == pol and str(r['k']) == k], key=lambda r: r['fold'])
        summary['|'.join((h, pol, k))] = {m: float(np.mean([r[m] for r in rs])) for m in ('score', 'spearman', 'centered_mse')} | {'fold_scores': [r['score'] for r in rs]}
    json.dump(dict(scenario=scenario, stage=stage, reproduction=repro, summary=summary, seconds=time.perf_counter() - t0), open(f'{OUT}/summary.json', 'w'), indent=1)
    log(dict(event='done', reproduction=repro, seconds=round(time.perf_counter() - t0, 1)))


# ---------------------------------------------------------------- analysis (pre-registered comparisons)
def nb_compare(new, base, ratio):
    n, b = np.asarray(new), np.asarray(base)
    imp = float((b.mean() - n.mean()) / b.mean() * 100)
    d = (b - n) / b * 100; wins = int((n < b).sum())
    var = d.var(ddof=1)
    se = float(np.sqrt((1 / 5 + ratio) * var)) if var > 0 else float('nan')
    tval = float(d.mean() / se) if se and np.isfinite(se) else float('nan')
    p = float(2 * student_t.sf(abs(tval), 4)) if np.isfinite(tval) else float('nan')
    q = student_t.ppf(.975, 4)
    return dict(new_mean=float(n.mean()), base_mean=float(b.mean()), improvement_pct=imp, gain_pct_fold_mean=float(d.mean()),
                per_fold_improvement_pct=d.round(4).tolist(), folds_won=wins, nb_t=tval, nb_p=p,
                nb_ci95_pct=[float(d.mean() - q * se), float(d.mean() + q * se)] if np.isfinite(se) else None,
                new_fold_scores=n.tolist(), base_fold_scores=b.tolist())


def analyze(parent):
    out = {}
    for sc in ('new_cell', 'tissue'):
        d = f'{parent}/{sc}_main'
        if not os.path.exists(f'{d}/fold_metrics.csv'):
            continue
        rows = list(csv.DictReader(open(f'{d}/fold_metrics.csv')))
        for r in rows:
            r['fold'] = int(r['fold']); r['score'] = float(r['score']); r['spearman'] = float(r['spearman']); r['centered_mse'] = float(r['centered_mse'])
        ratio = float(np.mean([len(np.load(f'{d}/fold-{f}_predictions.npz')['held']) / len(np.load(f'{d}/fold-{f}_predictions.npz')['fit']) for f in range(5)]))

        def fs(h, pol, k):
            rs = sorted([r for r in rows if r['head'] == h and r['policy'] == pol and r['k'] == str(k)], key=lambda r: r['fold'])
            assert len(rs) == 5, (h, pol, k, len(rs))
            return np.array([r['score'] for r in rs])

        def mean_of(h, pol, k, m='score'):
            rs = [r for r in rows if r['head'] == h and r['policy'] == pol and r['k'] == str(k)]
            return float(np.mean([r[m] for r in rs]))
        teacher = fs('H_meas', 'measured_all', 'all').mean()
        curves = {}; k5080 = {}
        for h in HEADS:
            k0 = fs(h, 'virtual', 0).mean(); denom = k0 - teacher
            curves[h] = {}; k5080[h] = {}
            for pol in POLICIES + ('random_mean',):
                seq = [k0] + [fs(h, pol, k).mean() for k in KS] + [fs(h, 'measured_all', 'all').mean()]
                rec = [float((k0 - x) / denom) if denom != 0 else float('nan') for x in seq]
                klab = ['0'] + [str(k) for k in KS] + ['all']
                curves[h][pol] = dict(k=klab, srmse=[float(x) for x in seq], recovery=rec, monotone_nonincreasing_srmse=bool(np.all(np.diff(seq) <= 1e-12)))
                kk = {}
                for thr, name in ((.5, 'k50'), (.8, 'k80')):
                    hit = [klab[i] for i in range(1, len(klab)) if rec[i] >= thr]
                    kk[name] = hit[0] if hit else None
                k5080[h][pol] = kk
        comps = {}
        for h in HEADS:
            for k in (16, 32):
                for base in ('static', 'random_mean', 'variance', 'mean_expr'):
                    comps[f'{h}|k{k}|adaptive_vs_{base}'] = nb_compare(fs(h, 'adaptive', k), fs(h, base, k), ratio)
                for base in ('static', 'random_mean'):
                    comps[f'{h}|k{k}|adaptive_surprise_vs_{base}'] = nb_compare(fs(h, 'adaptive_surprise', k), fs(h, base, k), ratio)
                comps[f'{h}|k{k}|static_vs_random_mean'] = nb_compare(fs(h, 'static', k), fs(h, 'random_mean', k), ratio)
                comps[f'{h}|k{k}|variance_vs_random_mean'] = nb_compare(fs(h, 'variance', k), fs(h, 'random_mean', k), ratio)
                comps[f'{h}|k{k}|mean_expr_vs_random_mean'] = nb_compare(fs(h, 'mean_expr', k), fs(h, 'random_mean', k), ratio)
                comps[f'{h}|k{k}|oracle_static_vs_static'] = nb_compare(fs(h, 'oracle_static', k), fs(h, 'static', k), ratio)
                comps[f'{h}|k{k}|adaptive_vs_oracle_static'] = nb_compare(fs(h, 'adaptive', k), fs(h, 'oracle_static', k), ratio)
                comps[f'{h}|k{k}|adaptive_vs_m_only'] = nb_compare(fs(h, 'adaptive', k), fs('none', 'm_only', 'M'), ratio)
                comps[f'{h}|k{k}|static_vs_m_only'] = nb_compare(fs(h, 'static', k), fs('none', 'm_only', 'M'), ratio)
                comps[f'{h}|k{k}|adaptive_vs_k0_virtual'] = nb_compare(fs(h, 'adaptive', k), fs(h, 'virtual', 0), ratio)
            for k in KS:
                comps[f'{h}|k{k}|adaptive_vs_static_allk'] = nb_compare(fs(h, 'adaptive', k), fs(h, 'static', k), ratio)
                comps[f'{h}|k{k}|adaptive_vs_random_mean_allk'] = nb_compare(fs(h, 'adaptive', k), fs(h, 'random_mean', k), ratio)
        comps['H_virt|k0|virtual_vs_m_only'] = nb_compare(fs('H_virt', 'virtual', 0), fs('none', 'm_only', 'M'), ratio)
        comps['none|ref|matrix_vRNA_vs_m_only'] = nb_compare(fs('none', 'matrix_vRNA', 'ref'), fs('none', 'm_only', 'M'), ratio)
        comps['H_virt|k0|virtual_vs_matrix_vRNA'] = nb_compare(fs('H_virt', 'virtual', 0), fs('none', 'matrix_vRNA', 'ref'), ratio)
        comps['H_meas|kall|teacher_vs_m_only'] = nb_compare(fs('H_meas', 'measured_all', 'all'), fs('none', 'm_only', 'M'), ratio)
        comps['H_virt|kall|measured_into_virtual_head_vs_H_meas_kall'] = nb_compare(fs('H_virt', 'measured_all', 'all'), fs('H_meas', 'measured_all', 'all'), ratio)
        comps['H_meas|k0|virtual_into_measured_head_vs_H_virt_k0'] = nb_compare(fs('H_meas', 'virtual', 0), fs('H_virt', 'virtual', 0), ratio)
        verdict = {}
        for h in HEADS:
            cond = {}
            for base, tag in (('static', 'cond1_adaptive_beats_static'), ('random_mean', 'cond2_adaptive_beats_random_mean')):
                ok = True; detail = {}
                for k in (16, 32):
                    c = comps[f'{h}|k{k}|adaptive_vs_{base}']
                    good = c['improvement_pct'] > 0 and c['folds_won'] >= 4 and np.isfinite(c['nb_p']) and c['nb_p'] < 0.10
                    detail[f'k{k}'] = dict(improvement_pct=c['improvement_pct'], folds_won=c['folds_won'], nb_p=c['nb_p'], satisfied=bool(good))
                    ok &= bool(good)
                cond[tag] = dict(satisfied=ok, detail=detail)
            cond['cond3_adaptive_recovery_monotone'] = dict(satisfied=curves[h]['adaptive']['monotone_nonincreasing_srmse'], srmse_curve=curves[h]['adaptive']['srmse'])
            cond['all_satisfied'] = bool(all(cond[c]['satisfied'] for c in ('cond1_adaptive_beats_static', 'cond2_adaptive_beats_random_mean', 'cond3_adaptive_recovery_monotone')))
            verdict[h] = cond
        rand_range = {}
        for h in HEADS:
            for k in KS:
                per_seed = [fs(h, f'random_s{s}', k).mean() for s in SEEDS]
                rand_range[f'{h}|k{k}'] = dict(mean=float(np.mean(per_seed)), min=float(np.min(per_seed)), max=float(np.max(per_seed)), per_seed=per_seed)
        table = {}
        for key in sorted({(r['head'], r['policy'], r['k']) for r in rows}):
            h, pol, k = key
            table['|'.join(key)] = dict(score=mean_of(h, pol, k), spearman=mean_of(h, pol, k, 'spearman'), centered_mse=mean_of(h, pol, k, 'centered_mse'),
                                        fold_scores=fs(h, pol, k).tolist())
        summ = json.load(open(f'{d}/summary.json'))
        bank = json.load(open(f'{d}/bank_info.json'))
        gen = dict(generator_alpha_per_fold=[bank[f'fold-{f}']['generator_alpha'] for f in range(5)],
                   probe_virtual_oof_r2_mean_outer=float(np.mean([bank[f'fold-{f}/outer']['virtual_oof_r2'] for f in range(5)])),
                   probe_virtual_query_mse_mean_outer=float(np.mean([bank[f'fold-{f}/outer']['virtual_query_mse'] for f in range(5)])),
                   matrix_generator_oof_r2_mean_all_banks=float(np.mean([bank[k]['matrix_generator']['oof_r2_mean'] for k in bank if '/' in k])),
                   u_q_mean_outer=[bank[f'fold-{f}/outer']['u_q_mean'] for f in range(5)],
                   corr_w_meanU_outer=[bank[f'fold-{f}/outer']['corr_w_meanU'] for f in range(5)],
                   corr_w_var_outer=[bank[f'fold-{f}/outer']['corr_w_var'] for f in range(5)],
                   static_oracle_overlap16=[bank[f'fold-{f}/outer']['static_oracle_overlap16'] for f in range(5)],
                   adaptive_k16_distinct_genes=[bank[f'fold-{f}/outer']['adaptive_k16_distinct_genes'] for f in range(5)],
                   adaptive_k16_mean_overlap_with_static16=[bank[f'fold-{f}/outer']['adaptive_k16_mean_overlap_with_static16'] for f in range(5)],
                   static_top16_fold0=bank['fold-0/outer']['w_static_top16'], adaptive_k16_most_chosen_fold0=bank['fold-0/outer']['adaptive_k16_most_chosen'])
        out[sc] = dict(nb_ratio_test_train=ratio, teacher_srmse=float(teacher), reproduction=summ['reproduction'], table=table, curves=curves, k50_k80=k5080,
                       comparisons=comps, pass_conditions=verdict, random_range=rand_range, diagnostics=gen,
                       scenario_pass_primary_head=verdict['H_meas']['all_satisfied'], scenario_pass_secondary_head=verdict['H_virt']['all_satisfied'])
    out['PASS'] = bool(any(out[sc]['scenario_pass_primary_head'] for sc in out if sc in ('new_cell', 'tissue')))
    out['PASS_secondary_head_H_virt'] = bool(any(out[sc]['scenario_pass_secondary_head'] for sc in out if sc in ('new_cell', 'tissue')))
    os.makedirs(f'{parent}/analysis', exist_ok=True)
    json.dump(out, open(f'{parent}/analysis/summary.json', 'w'), indent=1)
    print(json.dumps(dict(PASS=out['PASS'], PASS_secondary=out['PASS_secondary_head_H_virt'],
                          per_scenario={sc: dict(primary=out[sc]['scenario_pass_primary_head'], secondary=out[sc]['scenario_pass_secondary_head'],
                                                 reproduction_passed=out[sc]['reproduction']['passed']) for sc in out if sc in ('new_cell', 'tissue')})))


if __name__ == '__main__':
    scenario, stage, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
    assert stage in ('repro', 'dryrun', 'main', 'analyze')
    if stage == 'analyze':
        analyze(OUT)
    else:
        assert scenario in ('new_cell', 'tissue')
        os.makedirs(OUT, exist_ok=False)
        main_stage(scenario, stage)

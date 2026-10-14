"""Acquisition probe (pre-registered in PLAN.md): at an equal budget of k measured proteins per query cell,
does per-sample ADAPTIVE choice of which proteins to measure recover more than a static or random panel?

Independent implementation (model pieces copied from the local protein_panel.py / tissue_kernel.py references).
Reads only: extra-supervision banks (original-data top-left blocks; the extra rows only for the extra__m_only
reproduction check), source-inputs.npz, strong-teacher saved outer predictions (reference column only),
protein-panel saved m_allcand fold scores (cross-check only). Writes only to OUT_DIR.

usage: python acquisition_probe.py SCENARIO STAGE OUT_DIR
  STAGE = repro   : reproduce saved m_only outer predictions (all folds); no new arms
  STAGE = dryrun  : fold-0 inner banks only (generator alpha + head alpha selection); no outer bank
  STAGE = main    : all folds, all pre-registered arms
  STAGE = analyze : SCENARIO ignored; OUT_DIR is the parent holding new_cell_main/ and tissue_main/
"""
import csv
import json
import os
import sys
import time
import warnings
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, spearmanr, t as student_t

RUN = '/public/home/mengxl/creed-research-runs'
EXT = RUN + '/extra-supervision-20261008-0001/output'
EXTRA = RUN + '/extra-supervision-20261008-0001/resources/m_gdsc_extra19.npz'
SOURCE = RUN + '/gdsc-metabolite-full-protein-20260908-0001/output/result/source-inputs.npz'
ST = RUN + '/strong-teacher-recoverability-20261008-0001/output'
ST_DIR = {'new_cell': 'new_cell', 'tissue': 'tissue_v2'}
PP = RUN + '/protein-panel-20261009-0001/output'
SCEN_ID = {'new_cell': 0, 'tissue': 1}
BANK_ID = {'inner-0': 0, 'inner-1': 1, 'inner-2': 2, 'outer': 3}
BANKS = ('inner-0', 'inner-1', 'inner-2', 'outer')

ALPHAS = (.1, 1., 10., 100.)
KS = (8, 16, 32, 64)
SEEDS = (0, 1, 2, 3, 4)
COVER = 0.9
MIN_PAIR = 10
N_SUB = 3
HEADS = ('H_meas', 'H_virt')
POLICIES = ('static', 'variance', 'adaptive', 'adaptive_surprise', 'oracle_static') + tuple(f'random_s{s}' for s in SEEDS)
PRIMARY_SAVE = {('H_meas', 'virtual', 0), ('H_meas', 'measured_all', 'all'), ('H_virt', 'virtual', 0), ('H_virt', 'measured_all', 'all')}
for _h in HEADS:
    for _k in (16, 32):
        for _p in ('adaptive', 'static', 'variance') + tuple(f'random_s{s}' for s in SEEDS):
            PRIMARY_SAVE.add((_h, _p, _k))


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


def spearman_scores(P, Y, min_n=MIN_PAIR):
    """sum_j |Spearman(P[:, i], Y[:, j])| on pairwise-complete rows; pairs with n<min_n or a constant side add 0."""
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


def candidate_stats(p_fit):
    """candidate columns (coverage >= COVER in fit, observed variance > 1e-12), their fit observed mean and variance."""
    obs = np.isfinite(p_fit); cnt = obs.sum(0); frac = cnt / len(p_fit)
    mean = np.divide(np.where(obs, p_fit, 0).sum(0), cnt, out=np.zeros(p_fit.shape[1]), where=cnt > 0)
    var = np.divide(np.where(obs, (p_fit - mean) ** 2, 0).sum(0), cnt, out=np.zeros(p_fit.shape[1]), where=cnt > 0)
    cand = np.flatnonzero((frac >= COVER) & (var > 1e-12))
    return cand, mean[cand], var[cand]


def tissue_tokens(t, tag):
    return np.array([x if x else f'__unknown_{tag}_{i}' for i, x in enumerate(t)])


# ---------------------------------------------------------------- virtual-P generator
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
    """0.5*K_M + 0.5*RBF(P_fit) response head. P_fit given in the units the head is fitted on; queries in the same units.
    Standardization by fit moments of P_fit, median positive squared distance bandwidth (same recipe as protein_kernel)."""

    def __init__(self, KM, HM, P_fit, y):
        mu = P_fit.mean(0); sd = P_fit.std(0); self.mu, self.sd = mu, np.where(sd < 1e-8, 1., sd)
        self.Z = (P_fit - self.mu) / self.sd
        d = cdist(self.Z, self.Z, 'sqeuclidean'); pos = d[np.triu_indices(len(self.Z), 1)]; pos = pos[pos > 0]
        self.width = float(np.median(pos)) if len(pos) else None
        KP = np.exp(-d / (2 * self.width)) if self.width else np.ones_like(d)
        self.K = .5 * KM + .5 * KP; self.HM = HM; self.y = y
        self.C = None

    def cross(self, Q):
        HP = np.exp(-cdist((Q - self.mu) / self.sd, self.Z, 'sqeuclidean') / (2 * self.width)) if self.width else np.ones((len(Q), len(self.Z)))
        return .5 * self.HM + .5 * HP

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
def load_bank(S, scenario, f, bank):
    G = np.load(f'{EXT}/{scenario}/fold-{f}/{bank}/geometry.npz', allow_pickle=False)
    fit, query, panel = G['fit'], G['query'], G['panel']; n = len(fit)
    B = dict(fit=fit, query=query, panel=panel, n=n, G=G,
             KM=G['m_only_kernel'][:n, :n].copy(), HM=G['m_only_cross'][:, :n].copy(),
             y=S['auc'][np.ix_(fit, panel)])
    p = S['p']
    cand, cmean, cvar = candidate_stats(p[fit])
    pf = p[np.ix_(fit, cand)]; pq = p[np.ix_(query, cand)]
    pf = np.where(np.isfinite(pf), pf, cmean); pq_imp = np.where(np.isfinite(pq), pq, cmean)
    sd = pf.std(0); sd = np.where(sd < 1e-8, 1., sd); mu = pf.mean(0)
    B.update(cand=cand, cvar=cvar, mu=mu, sd=sd, P_fit_raw=pf, Z_fit=(pf - mu) / sd, Zq_imp=(pq_imp - mu) / sd,
             Zq_obs=np.isfinite(pq), Zq_raw=(pq - mu) / sd)
    return B


def generator_loss(B, scenario, f, bank):
    """inner-bank P-reconstruction loss (observed standardized query entries) of the 3-sub-generator mean, per alpha."""
    parts = sub_split(B['n'], scenario, f, bank); out = []
    for alpha in ALPHAS:
        _, V_q = generator(B['KM'], B['HM'], B['Z_fit'], alpha, parts)
        e = (V_q.mean(0) - B['Zq_raw']) ** 2
        out.append(float(np.nanmean(np.where(B['Zq_obs'], e, np.nan))))
    return out


def build_heads(B, scenario, f, bank, alpha_gen):
    parts = sub_split(B['n'], scenario, f, bank)
    V_fit, V_q3 = generator(B['KM'], B['HM'], B['Z_fit'], alpha_gen, parts)
    B.update(V_fit=V_fit, V_q3=V_q3, V_q=V_q3.mean(0), U_q=V_q3.std(0))
    B['H_meas'] = Head(B['KM'], B['HM'], B['P_fit_raw'], B['y'])
    B['H_virt'] = Head(B['KM'], B['HM'], V_fit, B['y'])


def head_input(B, head, Q_std):
    """Q_std: query P in measured-standardized units (virtual mean with acquired entries overridden)."""
    return Q_std * B['sd'] + B['mu'] if head == 'H_meas' else Q_std


def repro_grids(S, E, scenario, B):
    """saved-arm grids on this bank: original__m_only (fit block) and, for new_cell, extra__m_only (full saved kernel)."""
    out = {'original__m_only': predict_grid(B['KM'], B['HM'], B['y'])}
    if scenario == 'new_cell':
        G = B['G']; ei = G['extra_indices']
        y_ext = np.vstack([B['y'], E['auc'][np.ix_(ei, B['panel'])]])
        out['extra__m_only'] = predict_grid(G['m_only_kernel'], G['m_only_cross'], y_ext)
    return out


# ---------------------------------------------------------------- outer-bank policies
def policy_indices(B, S, f, scenario, info):
    """returns dict policy -> dict k -> index array (k,) for static policies or (n_q, k) for per-cell policies."""
    fit, cand, y = B['fit'], B['cand'], B['y']; p = S['p']; n_q = len(B['query'])
    t0 = time.perf_counter()
    w = spearman_scores(p[np.ix_(fit, cand)], y)
    t1 = time.perf_counter()
    w_or = spearman_scores(p[:, cand], S['auc'][:, B['panel']])
    info.update(spearman_seconds=round(t1 - t0, 1), oracle_spearman_seconds=round(time.perf_counter() - t1, 1))
    orders = {'static': np.argsort(-w, kind='stable'), 'variance': np.argsort(-B['cvar'], kind='stable'),
              'oracle_static': np.argsort(-w_or, kind='stable')}
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
    pid = S['protein_ids']
    info.update(w_static_top16=pid[cand[orders['static'][:16]]].tolist(), w_static_top16_score=w[orders['static'][:16]].round(5).tolist(),
                oracle_top16=pid[cand[orders['oracle_static'][:16]]].tolist(), variance_top16=pid[cand[orders['variance'][:16]]].tolist(),
                static_oracle_overlap16=int(len(np.intersect1d(orders['static'][:16], orders['oracle_static'][:16]))),
                static_oracle_overlap64=int(len(np.intersect1d(orders['static'][:64], orders['oracle_static'][:64]))),
                u_q_mean=float(B['U_q'].mean()), u_q_median=float(np.median(B['U_q'])),
                corr_w_meanU=float(np.corrcoef(w, B['U_q'].mean(0))[0, 1]))
    for pol in ('adaptive', 'adaptive_surprise'):
        chosen = idx[pol][16]; counts = np.bincount(chosen.ravel(), minlength=len(cand))
        top = np.argsort(-counts, kind='stable')[:16]
        info[f'{pol}_k16_most_chosen'] = [(pid[cand[t]], int(counts[t])) for t in top]
        info[f'{pol}_k16_distinct_proteins'] = int((counts > 0).sum())
        info[f'{pol}_k16_mean_overlap_with_static16'] = float(np.mean([len(np.intersect1d(r, orders['static'][:16])) for r in chosen]))
        info[f'{pol}_k16_mean_overlap_with_static64'] = float(np.mean([len(np.intersect1d(r, orders['static'][:64])) for r in chosen]))
    return idx


def build_query(B, sel):
    Q = B['V_q'].copy()
    if sel is None:
        return Q
    if sel.ndim == 1:
        Q[:, sel] = B['Zq_imp'][:, sel]
    else:
        rows = np.arange(len(Q))[:, None]; Q[rows, sel] = B['Zq_imp'][rows, sel]
    return Q


# ---------------------------------------------------------------- driver
def run_fold(S, E, scenario, f, stage, repro, bank_info, log):
    t0 = time.perf_counter()
    banks = {b: load_bank(S, scenario, f, b) for b in (BANKS[:3] if stage == 'dryrun' else BANKS)}
    for b, B in banks.items():
        bank_info[f'fold-{f}/{b}'] = dict(n_fit=int(B['n']), n_query=int(len(B['query'])), candidate_count=int(len(B['cand'])))
    # inner losses for the saved m_only arm (reproduction) and the two heads
    inner = {}
    for b in BANKS[:3]:
        B = banks[b]
        inner[b] = dict(truth=S['auc'][np.ix_(B['query'], B['panel'])],
                        sd=np.maximum(np.nanstd(B['y'], axis=0), 1e-8))
    saved_arms = ('original__m_only', 'extra__m_only') if scenario == 'new_cell' else ('original__m_only',)
    total = {a: 0 for a in saved_arms}; count = {a: 0 for a in saved_arms}; outer_saved = {}
    if stage in ('repro', 'main'):
        for b, B in banks.items():
            grids = repro_grids(S, E, scenario, B)
            saved = np.load(f'{EXT}/{scenario}/fold-{f}/{b}/grids.npz', allow_pickle=False)
            for a, g in grids.items():
                d = float(np.max(np.abs(g - saved[a])))
                repro['grid_max_abs_diff'][a] = max(repro['grid_max_abs_diff'].get(a, 0.), d)
                if b == 'outer':
                    outer_saved[a] = g
                else:
                    ll, nn = bank_losses(g, inner[b]['truth'], inner[b]['sd']); total[a] = total[a] + ll; count[a] = count[a] + nn
    if stage == 'repro':
        P = np.load(f'{EXT}/{scenario}/fold-{f}/predictions.npz', allow_pickle=False)
        truth, sd, use = P['truth'], P['train_sd'], P['evaluated']
        rows = []
        for a in saved_arms:
            pred = take(outer_saved[a], choose(total[a], count[a]))
            repro['pred_max_abs_diff'][a] = max(repro['pred_max_abs_diff'].get(a, 0.), float(np.max(np.abs(pred - P[a]))))
            rows.append(dict(fold=f, arm=a, score=metrics(truth, pred, sd, use)['score'], score_of_saved_pred=metrics(truth, P[a], sd, use)['score']))
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
                                           virtual_query_mse_observed=float(np.nanmean(np.where(B['Zq_obs'], (B['V_q'] - B['Zq_raw']) ** 2, np.nan))),
                                           head_meas_width=B['H_meas'].width, head_virt_width=B['H_virt'].width)
        if b == 'outer':
            continue
        for h in HEADS:
            Q = B['Zq_imp'] if h == 'H_meas' else B['V_q']
            g = B[h].grid(head_input(B, h, Q))
            ll, nn = bank_losses(g, inner[b]['truth'], inner[b]['sd']); htot[h] = htot[h] + ll; hcnt[h] = hcnt[h] + nn
        log(dict(event='bank_done', fold=f, bank=b, candidates=int(len(B['cand'])), seconds=round(time.perf_counter() - t0, 1)))
    sel = {h: choose(htot[h], hcnt[h]) for h in HEADS}
    sel['original__m_only'] = choose(total['original__m_only'], count['original__m_only']) if stage == 'main' else None
    bank_info[f'fold-{f}']['alpha_counts'] = {h: np.bincount(sel[h], minlength=len(ALPHAS)).tolist() for h in HEADS}
    if stage == 'dryrun':
        log(dict(event='dryrun_fold_done', fold=f, alpha_counts=bank_info[f'fold-{f}']['alpha_counts'], seconds=round(time.perf_counter() - t0, 1)))
        return []
    # --- outer bank
    B = banks['outer']; info = bank_info[f'fold-{f}/outer']
    P = np.load(f'{EXT}/{scenario}/fold-{f}/predictions.npz', allow_pickle=False)
    truth, sd, use = P['truth'], P['train_sd'], P['evaluated']
    assert np.array_equal(P['fit'], B['fit']) and np.array_equal(P['held'], B['query']) and np.array_equal(P['panel'], B['panel'])
    for h in HEADS:
        B[h].freeze(sel[h])
    idx = policy_indices(B, S, f, scenario, info)
    preds = {}
    pred_m = take(outer_saved['original__m_only'], sel['original__m_only'])
    repro['pred_max_abs_diff']['original__m_only'] = max(repro['pred_max_abs_diff'].get('original__m_only', 0.), float(np.max(np.abs(pred_m - P['original__m_only']))))
    preds[('none', 'm_only', 'M')] = pred_m
    preds[('none', 'null_drug', 'none')] = np.broadcast_to(np.nanmean(B['y'], axis=0), truth.shape).copy()
    T = np.load(f'{ST}/{ST_DIR[scenario]}/fold-{f}/predictions.npz', allow_pickle=False)
    assert np.array_equal(T['fit'], P['fit']) and np.array_equal(T['held'], P['held']) and np.array_equal(T['panel'], P['panel'])
    preds[('none', 'teacher_saved', 'full_P')] = T['measured_p']
    for h in HEADS:
        preds[(h, 'virtual', 0)] = B[h].predict(head_input(B, h, build_query(B, None)))
        preds[(h, 'measured_all', 'all')] = B[h].predict(head_input(B, h, B['Zq_imp']))
        for pol in POLICIES:
            for k in KS:
                preds[(h, pol, k)] = B[h].predict(head_input(B, h, build_query(B, idx[pol][k])))
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
                        generator_alpha=alpha_gen, candidates=B['cand'], adaptive_k16=idx['adaptive'][16], adaptive_k32=idx['adaptive'][32],
                        static_k64=idx['static'][64], variance_k64=idx['variance'][64], oracle_k64=idx['oracle_static'][64],
                        U_q=B['U_q'], V_q=B['V_q'], Zq_imp=B['Zq_imp'],
                        **{f'random_s{s}_k64': idx[f'random_s{s}'][64] for s in SEEDS})
    log(dict(event='fold_done', fold=f, seconds=round(time.perf_counter() - t0, 1)))
    return rows


def main_stage(scenario, stage):
    t0 = time.perf_counter()
    logf = open(f'{OUT}/stage.log', 'a')

    def log(d):
        s = json.dumps(d); print(s, flush=True); logf.write(s + '\n'); logf.flush()
    S = np.load(SOURCE, allow_pickle=False)
    S = {k: S[k] for k in ('m', 'auc', 'p', 'tissues', 'sample_ids', 'drug_ids', 'protein_ids')}
    E = None
    if scenario == 'new_cell':
        E = np.load(EXTRA, allow_pickle=False); E = {k: E[k] for k in ('auc', 'sample_ids', 'drug_ids')}
        assert (E['drug_ids'] == S['drug_ids']).all()
    repro = dict(grid_max_abs_diff={}, pred_max_abs_diff={})
    bank_info, rows = {}, []
    folds = (0,) if stage == 'dryrun' else range(5)
    for f in folds:
        rows += run_fold(S, E, scenario, f, stage, repro, bank_info, log)
    json.dump(bank_info, open(f'{OUT}/bank_info.json', 'w'), indent=1)
    if stage == 'dryrun':
        log(dict(event='dryrun_done', seconds=round(time.perf_counter() - t0, 1)))
        return
    if stage == 'repro':
        saved = {}
        with open(f'{EXT}/{scenario}/fold_metrics.csv') as fh:
            for r in csv.DictReader(fh):
                saved[(int(r['fold']), r['arm'])] = float(r['score'])
        md = 0.; md2 = 0.
        for r in rows:
            md = max(md, abs(r['score'] - saved[(r['fold'], r['arm'])])); md2 = max(md2, abs(r['score_of_saved_pred'] - saved[(r['fold'], r['arm'])]))
        repro['fold_score_max_abs_diff_my_pred_vs_saved_csv'] = md
        repro['fold_score_max_abs_diff_my_metric_on_saved_pred_vs_saved_csv'] = md2
        repro['passed'] = bool(max(repro['pred_max_abs_diff'].values()) <= 1e-8 and max(repro['grid_max_abs_diff'].values()) <= 1e-8)
        json.dump(dict(scenario=scenario, stage=stage, reproduction=repro, rows=rows, seconds=time.perf_counter() - t0), open(f'{OUT}/summary.json', 'w'), indent=1)
        log(dict(event='done', reproduction=repro, seconds=round(time.perf_counter() - t0, 1)))
        return
    with open(f'{OUT}/fold_metrics.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['scenario', 'head', 'policy', 'k', 'fold', 'score', 'spearman', 'centered_mse']); w.writeheader(); w.writerows(rows)
    # cross-check H_meas k=all vs protein-panel saved m_allcand fold scores (not a gate)
    pp = json.load(open(f'{PP}/{scenario}_main/summary.json'))['fold_scores']['m_allcand']
    mine = [r['score'] for r in sorted(rows, key=lambda r: r['fold']) if r['head'] == 'H_meas' and r['policy'] == 'measured_all']
    repro['H_meas_kall_vs_protein_panel_m_allcand_fold_score_max_abs_diff'] = float(np.max(np.abs(np.array(mine) - np.array(pp))))
    repro['passed'] = bool(max(repro['pred_max_abs_diff'].values()) <= 1e-8 and max(repro['grid_max_abs_diff'].values()) <= 1e-8)
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
    return dict(new_mean=float(n.mean()), base_mean=float(b.mean()), improvement_pct=imp, per_fold_improvement_pct=d.round(4).tolist(),
                folds_won=wins, nb_t=tval, nb_p=p, nb_ci95_pct=[float(d.mean() - q * se), float(d.mean() + q * se)] if np.isfinite(se) else None,
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
        R = {}; curves = {}
        for h in HEADS:
            k0 = fs(h, 'virtual', 0).mean(); denom = k0 - teacher
            curves[h] = {}
            for pol in POLICIES + ('random_mean',):
                seq = [k0] + [fs(h, pol, k).mean() for k in KS] + [fs(h, 'measured_all', 'all').mean()]
                curves[h][pol] = dict(k=['0'] + [str(k) for k in KS] + ['all'], srmse=[float(x) for x in seq],
                                      recovery=[float((k0 - x) / denom) if denom != 0 else float('nan') for x in seq],
                                      monotone_nonincreasing_srmse=bool(np.all(np.diff(seq) <= 1e-12)))
        comps = {}
        for h in HEADS:
            for k in (16, 32):
                for base in ('static', 'random_mean', 'variance'):
                    comps[f'{h}|k{k}|adaptive_vs_{base}'] = nb_compare(fs(h, 'adaptive', k), fs(h, base, k), ratio)
                for base in ('static', 'random_mean'):
                    comps[f'{h}|k{k}|adaptive_surprise_vs_{base}'] = nb_compare(fs(h, 'adaptive_surprise', k), fs(h, base, k), ratio)
                comps[f'{h}|k{k}|static_vs_random_mean'] = nb_compare(fs(h, 'static', k), fs(h, 'random_mean', k), ratio)
                comps[f'{h}|k{k}|oracle_static_vs_static'] = nb_compare(fs(h, 'oracle_static', k), fs(h, 'static', k), ratio)
                comps[f'{h}|k{k}|adaptive_vs_oracle_static'] = nb_compare(fs(h, 'adaptive', k), fs(h, 'oracle_static', k), ratio)
                comps[f'{h}|k{k}|adaptive_vs_m_only'] = nb_compare(fs(h, 'adaptive', k), fs('none', 'm_only', 'M'), ratio)
                comps[f'{h}|k{k}|static_vs_m_only'] = nb_compare(fs(h, 'static', k), fs('none', 'm_only', 'M'), ratio)
        comps['H_virt|k0|virtual_vs_m_only'] = nb_compare(fs('H_virt', 'virtual', 0), fs('none', 'm_only', 'M'), ratio)
        comps['H_meas|kall|teacher_vs_m_only'] = nb_compare(fs('H_meas', 'measured_all', 'all'), fs('none', 'm_only', 'M'), ratio)
        comps['H_virt|kall|measured_into_virtual_head_vs_H_meas_kall'] = nb_compare(fs('H_virt', 'measured_all', 'all'), fs('H_meas', 'measured_all', 'all'), ratio)
        comps['H_meas|k0|virtual_into_measured_head_vs_H_virt_k0'] = nb_compare(fs('H_meas', 'virtual', 0), fs('H_virt', 'virtual', 0), ratio)
        # pre-registered PASS conditions, per head
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
        out[sc] = dict(nb_ratio_test_train=ratio, teacher_srmse=float(teacher), reproduction=summ['reproduction'], table=table, curves=curves,
                       comparisons=comps, pass_conditions=verdict, random_range=rand_range,
                       scenario_pass_primary_head=verdict['H_meas']['all_satisfied'], scenario_pass_secondary_head=verdict['H_virt']['all_satisfied'])
    out['PASS'] = bool(any(out[sc]['scenario_pass_primary_head'] for sc in out if sc in ('new_cell', 'tissue')))
    out['PASS_secondary_head_H_virt'] = bool(any(out[sc]['scenario_pass_secondary_head'] for sc in out if sc in ('new_cell', 'tissue')))
    os.makedirs(f'{parent}/analysis', exist_ok=True)
    json.dump(out, open(f'{parent}/analysis/summary.json', 'w'), indent=1)
    print(json.dumps(dict(PASS=out['PASS'], PASS_secondary=out['PASS_secondary_head_H_virt'],
                          per_scenario={sc: dict(primary=out[sc]['scenario_pass_primary_head'], secondary=out[sc]['scenario_pass_secondary_head']) for sc in out if sc in ('new_cell', 'tissue')})))


if __name__ == '__main__':
    scenario, stage, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
    assert stage in ('repro', 'dryrun', 'main', 'analyze')
    if stage == 'analyze':
        analyze(OUT)
    else:
        assert scenario in ('new_cell', 'tissue')
        os.makedirs(OUT, exist_ok=False)
        main_stage(scenario, stage)

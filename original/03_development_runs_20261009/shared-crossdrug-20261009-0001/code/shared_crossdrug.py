"""Shared cross-drug cell-sensitivity factor (Direction B), pre-registered in PLAN.md.

Independent implementation (no project imports). Baseline masked KRR / P-graph / tissue-kernel code copied
from the local reference /Users/xlzhexue/Desktop/proj/creed/tissue_kernel_20261009/tissue_kernel.py.
Reads only the frozen banks of extra-supervision-20261008-0001, the 421 source table, the 19-sample extra
table and the tissue-kernel run predictions (for reproduction only). Writes only to OUT_DIR.

Usage: python shared_crossdrug.py SCENARIO OUT_DIR MODE
  SCENARIO: new_cell | tissue
  MODE: repro -> only existing baseline arms (reproduction check; no new arm is computed)
        full  -> all pre-registered arms, selection on inner banks, outer evaluation, analysis
"""
import csv
import json
import os
import sys
import time
import numpy as np
from scipy.linalg import eigh, cho_factor, cho_solve
from scipy.stats import spearmanr, t as student_t

RUN = '/public/home/mengxl/creed-research-runs/extra-supervision-20261008-0001'
SOURCE = '/public/home/mengxl/creed-research-runs/gdsc-metabolite-full-protein-20260908-0001/output/result/source-inputs.npz'
EXTRA = RUN + '/resources/m_gdsc_extra19.npz'
TK = '/public/home/mengxl/creed-research-runs/tissue-kernel-20261009-0001/output'
ALPHAS = (.1, 1., 10., 100.)
MUS = (0., .1, 1., 10., 100., 1000.)
WS = (0., .25, 1.)
GAMMAS = (.5, 1.)
RANKS = (1, 2, 3)
BANKS = ('inner-0', 'inner-1', 'inner-2', 'outer')
MIN_OBS = 20
SEED = 20261009
N_BOOT = 2000
SHARED_TYPES = ('mean',) + tuple(f'svd{r}' for r in RANKS) + ('perm',)


# ----------------------------------------------------------------------------- arm registry
def registry(scenario):
    if scenario == 'new_cell':
        S = lambda w, mu: ('m_only', 'extra', w, mu)
        lr = ['base'] + [f'svd{r}' for r in RANKS]
        arms = {
            'base_m': [(S(0., 0.), ['base'])],
            'base_mP': [(S(0., mu), ['base']) for mu in MUS],
            'base_mt': [(S(w, 0.), ['base']) for w in WS],
            'base_mtP': [(S(w, mu), ['base']) for w in WS for mu in MUS],
            'shared_m': [(S(0., 0.), ['base', 'mean'])],
            'sharedLR_m': [(S(0., 0.), lr)],
            'shared_mt': [(S(w, 0.), ['base', 'mean']) for w in WS],
            'sharedLR_mt': [(S(w, 0.), lr) for w in WS],
            'shared_mP': [(S(0., mu), ['base', 'mean']) for mu in MUS],
            'shared_mtP': [(S(w, mu), ['base', 'mean']) for w in WS for mu in MUS],
            'sharedPerm_m': [(S(0., 0.), ['base', 'perm'])],
            'sharedPerm_mt': [(S(w, 0.), ['base', 'perm']) for w in WS],
        }
        repro = {'base_m': ('saved', 'extra__m_only'), 'base_mP': ('saved', 'extra__m_only__protein'),
                 'base_mt': ('tk', 'm_tissue'), 'base_mtP': ('tk', 'm_tissue__protein'),
                 'tissue_only': ('tk', 'tissue_only')}
        settings = [S(w, mu) for w in WS for mu in MUS]
        lr_settings = {S(w, 0.) for w in WS}
    else:
        F = lambda d, mu: ('virtual_fm', d, 0., mu)
        M0 = ('m_only', 'extra', 0., 0.)
        arms = {
            'base_m': [(M0, ['base'])],
            'base_fm': [(F('extra', 0.), ['base'])],
            'base_fmP': [(F('extra', mu), ['base']) for mu in MUS],
            'base_fm_orig': [(F('original', 0.), ['base'])],
            'base_fmP_orig': [(F('original', mu), ['base']) for mu in MUS],
            'shared_m': [(M0, ['base', 'mean'])],
            'shared_fm': [(F('extra', 0.), ['base', 'mean'])],
            'sharedLR_fm': [(F('extra', 0.), ['base'] + [f'svd{r}' for r in RANKS])],
            'shared_fmP': [(F('extra', mu), ['base', 'mean']) for mu in MUS],
            'shared_fm_orig': [(F('original', 0.), ['base', 'mean'])],
            'shared_fmP_orig': [(F('original', mu), ['base', 'mean']) for mu in MUS],
            'sharedPerm_fm': [(F('extra', 0.), ['base', 'perm'])],
        }
        repro = {'base_m': ('saved', 'extra__m_only'), 'base_fm': ('saved', 'extra__virtual_fm'),
                 'base_fmP': ('saved', 'extra__virtual_fm__protein'),
                 'base_fm_orig': ('saved', 'original__virtual_fm'),
                 'base_fmP_orig': ('saved', 'original__virtual_fm__protein')}
        settings = [M0] + [F('extra', mu) for mu in MUS] + [F('original', mu) for mu in MUS]
        lr_settings = {F('extra', 0.)}
    return arms, repro, settings, lr_settings


def comparisons(scenario):
    """(new, base, role). Roles: primary / secondary / context. Fixed before any outer score."""
    if scenario == 'new_cell':
        return [('shared_m', 'base_m', 'primary'), ('shared_mt', 'base_mt', 'primary'),
                ('sharedLR_m', 'base_m', 'secondary'), ('sharedLR_mt', 'base_mt', 'secondary'),
                ('shared_mP', 'base_mP', 'secondary'), ('shared_mtP', 'base_mtP', 'secondary'),
                ('shared_m', 'base_mP', 'context'), ('shared_mP', 'base_m', 'context'),
                ('shared_mt', 'tissue_only', 'context'), ('shared_mtP', 'base_mt', 'context'),
                ('sharedPerm_m', 'base_m', 'control'), ('sharedPerm_mt', 'base_mt', 'control'),
                ('shared_m', 'sharedPerm_m', 'control'), ('shared_mt', 'sharedPerm_mt', 'control')]
    return [('shared_fm', 'base_fm', 'primary'),
            ('shared_fm', 'base_fm_orig', 'secondary'), ('shared_m', 'base_m', 'secondary'),
            ('sharedLR_fm', 'base_fm', 'secondary'), ('shared_fmP', 'base_fmP', 'secondary'),
            ('shared_fm_orig', 'base_fm_orig', 'secondary'), ('shared_fmP_orig', 'base_fmP_orig', 'secondary'),
            ('shared_fmP_orig', 'base_fm_orig', 'context'), ('shared_fmP', 'base_fm_orig', 'context'),
            ('sharedPerm_fm', 'base_fm', 'control'), ('shared_fm', 'sharedPerm_fm', 'control')]


# ----------------------------------------------------------------------------- reference KRR (copied)
def predict_grid(k, h, y):
    pred = np.empty((len(ALPHAS), len(h), y.shape[1])); groups = {}
    for j in range(y.shape[1]):
        groups.setdefault(np.isfinite(y[:, j]).tobytes(), []).append(j)
    for cols in groups.values():
        obs = np.flatnonzero(np.isfinite(y[:, cols[0]]))
        yy = y[np.ix_(obs, cols)]; mean = yy.mean(0)
        pred[:, :, cols] = mean
        if len(obs) < MIN_OBS:
            continue
        kk = k[np.ix_(obs, obs)]; hh = h[:, obs]
        for ai, alpha in enumerate(ALPHAS):
            coef = cho_solve(cho_factor(kk + alpha * np.eye(len(obs)), lower=True), yy - mean)
            pred[ai][:, cols] = hh @ coef + mean
    return pred


def graph_root(k):
    vals, vecs = eigh((k + k.T) / 2)
    return (vecs * np.sqrt(np.maximum(vals, 0))) @ vecs.T


def effective_kernel(k, h, L, mu, root):
    if mu == 0:
        return k, h
    a = np.eye(len(k)) + mu * root @ L @ root; a = (a + a.T) / 2
    ke = root @ cho_solve(cho_factor(a, lower=True), root); ke = (ke + ke.T) / 2
    he = h - mu * h @ L @ ke
    return ke, he


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
                centered_mse=float((mse - bias * bias)[use].mean()), mse=mse, rho=rho, count=n)


# ----------------------------------------------------------------------------- shared cross-drug factor
def lowrank_factors(ze, r, iters=300, tol=1e-7):
    """Hard-impute rank-r SVD of the standardized fit label matrix; missing entries only imputed inside fit."""
    obs = np.isfinite(ze); X = np.where(obs, ze, 0.); prev = None; it = 0
    for it in range(iters):
        U, sv, vt = np.linalg.svd(X, full_matrices=False)
        low = (U[:, :r] * sv[:r]) @ vt[:r]
        X = np.where(obs, ze, low)
        if prev is not None and np.linalg.norm(low - prev) <= tol * np.linalg.norm(low):
            break
        prev = low
    U, sv, vt = np.linalg.svd(X, full_matrices=False)
    return U[:, :r] * sv[:r], it + 1


def label_state(y, need_lowrank):
    """Fit-only standardization, shared factors, per-drug loadings and factor deviations (kernel independent)."""
    obs = np.isfinite(y); nobs = obs.sum(0)
    assert (nobs > 0).all()
    mu = np.nanmean(y, axis=0)
    sd = np.maximum(np.nanstd(y, axis=0), 1e-8)
    z = (y - mu) / sd
    elig = nobs >= MIN_OBS
    ze = np.where(obs[:, elig], z[:, elig], np.nan)
    cnt = np.isfinite(ze).sum(1); fok = cnt > 0
    s = np.full(len(y), np.nan)
    s[fok] = np.nanmean(ze[fok], axis=1)
    factors = {'mean': s[:, None]}
    iters = {}
    if need_lowrank:
        for r in RANKS:
            factors[f'svd{r}'], iters[f'svd{r}'] = lowrank_factors(ze, r)
        # control: the same mean factor with its cell correspondence destroyed (fixed permutation)
        rows = np.flatnonzero(fok); sp = np.full(len(y), np.nan)
        sp[rows] = s[np.random.default_rng(SEED).permutation(rows)]
        factors['perm'] = sp[:, None]
    fam = {}
    n, D = y.shape
    for name, F in factors.items():
        r = F.shape[1]; B = np.zeros((D, r)); Fbar = np.zeros((D, r)); dev = np.zeros((n, D))
        for d in np.flatnonzero(elig):
            rows = obs[:, d] & fok
            X = F[rows]; xb = X.mean(0); Xc = X - xb
            zz = z[rows, d]
            b = np.linalg.lstsq(Xc, zz - zz.mean(), rcond=None)[0]
            B[d] = b; Fbar[d] = xb
            dev[rows, d] = Xc @ b
        fam[name] = dict(F=F, B=B, Fbar=Fbar, dev=dev)
    return dict(obs=obs, mu=mu, sd=sd, z=z, elig=elig, fok=fok, fam=fam, lowrank_iters=iters,
                factor_cells=int(fok.sum()))


def krr_multi(k, h, obs, rhs):
    """Mean-centred masked KRR (same groups/fallback as predict_grid) for several right-hand sides.
    rhs: (n, D, m). Returns (len(ALPHAS), q, D, m)."""
    q = h.shape[0]; n, D, m = rhs.shape
    out = np.empty((len(ALPHAS), q, D, m)); groups = {}
    for j in range(D):
        groups.setdefault(obs[:, j].tobytes(), []).append(j)
    for cols in groups.values():
        o = np.flatnonzero(obs[:, cols[0]])
        yy = rhs[o][:, cols, :].reshape(len(o), -1); mean = yy.mean(0)
        if len(o) < MIN_OBS:
            out[:, :, cols, :] = mean.reshape(len(cols), m)
            continue
        kk = k[np.ix_(o, o)]; hh = h[:, o]
        for ai, alpha in enumerate(ALPHAS):
            coef = cho_solve(cho_factor(kk + alpha * np.eye(len(o)), lower=True), yy - mean)
            out[ai][:, cols, :] = (hh @ coef + mean).reshape(q, len(cols), m)
    return out


def factor_krr(k, h, F, fok):
    o = np.flatnonzero(fok); Fo = F[o]; m = Fo.mean(0)
    kk = k[np.ix_(o, o)]; hh = h[:, o]; out = []
    for alpha in ALPHAS:
        coef = cho_solve(cho_factor(kk + alpha * np.eye(len(o)), lower=True), Fo - m)
        out.append(hh @ coef + m)
    return np.array(out)  # (A, q, r)


def shared_families(k, h, st, base_grid, types):
    """Return {type: (len(GAMMAS)*A*A, q, D) grid} ordered gamma, alpha_s, alpha_r; plus consistency diff.
    prediction = mu_d + sd_d * (gamma * (Fhat_q - Fbar_d) B_d + KRR_alpha_r(z_d - gamma * (F - Fbar_d) B_d))."""
    z = np.where(st['obs'], st['z'], 0.)
    rhs = np.stack([z] + [st['fam'][t]['dev'] for t in types], axis=2)
    kr = krr_multi(k, h, st['obs'], rhs)  # (A, q, D, 1+T)
    mu, sd, elig = st['mu'], st['sd'], st['elig']
    zbase = mu + sd * kr[:, :, :, 0]
    consistency = float(np.max(np.abs(zbase - base_grid)))
    out = {}
    for ti, t in enumerate(types):
        fm = st['fam'][t]
        fh = factor_krr(k, h, fm['F'], st['fok'])  # (A, q, r)
        parts = []
        for g in GAMMAS:
            for ai_s in range(len(ALPHAS)):
                shared = fh[ai_s] @ fm['B'].T - np.sum(fm['Fbar'] * fm['B'], axis=1)[None, :]
                for ai_r in range(len(ALPHAS)):
                    zz = g * shared + kr[ai_r, :, :, 0] - g * kr[ai_r, :, :, 1 + ti]
                    p = mu + sd * zz
                    p[:, ~elig] = base_grid[ai_r][:, ~elig]
                    parts.append(p)
        out[t] = np.array(parts)
    return out, consistency


def decode(arm_blocks, sel):
    """Map selected config indices to (setting, family, gamma, alpha_s, alpha_r)."""
    table = []
    for setting, fams in arm_blocks:
        for fam in fams:
            if fam == 'base':
                table += [(setting, 'base', 0., None, a) for a in ALPHAS]
            else:
                table += [(setting, fam, g, a_s, a_r) for g in GAMMAS for a_s in ALPHAS for a_r in ALPHAS]
    return [table[i] for i in sel], len(table)


# ----------------------------------------------------------------------------- bank processing
def run_bank(scenario, f, bank, S, E, mode, settings, lr_settings):
    auc, stis = S['auc'], S['tissues']
    G = np.load(f'{RUN}/output/{scenario}/fold-{f}/{bank}/geometry.npz', allow_pickle=False)
    fit, query, panel, ei = G['fit'], G['query'], G['panel'], G['extra_indices']
    n0 = len(fit)
    y_ext = np.vstack([auc[np.ix_(fit, panel)], E['auc'][np.ix_(ei, panel)]])
    datas = sorted({s[1] for s in settings})
    ys = {'extra': y_ext, 'original': y_ext[:n0]}
    states = {}
    if mode == 'full':
        for d in datas:
            need_lr = any(s[1] == d for s in lr_settings)
            states[d] = label_state(ys[d], need_lr)
    blocks = {}; diag = dict(consistency=0.0)
    ttr = tissue_tokens(np.concatenate([stis[fit], E['tissues'][ei]]), 'tr')
    tq = tissue_tokens(stis[query], 'q')
    kgroups = {}
    for s in settings:
        kgroups.setdefault(s[:3], []).append(s[3])
    for (base, data, w), mus in kgroups.items():
        K, H, L = G[f'{base}_kernel'], G[f'{base}_cross'], G['L_protein']
        if w:
            T = (ttr[:, None] == ttr[None, :]).astype(float); Tq = (tq[:, None] == ttr[None, :]).astype(float)
            K = K + w * T; H = H + w * Tq
        if data == 'original':
            K, H, L = K[:n0, :n0], H[:, :n0], L[:n0, :n0]
        y = ys[data]
        root = graph_root(K) if any(m > 0 for m in mus) else None
        for mu in mus:
            ke, he = effective_kernel(K, H, L, mu, root)
            setting = (base, data, w, mu)
            bg = predict_grid(ke, he, y)
            blocks[(setting, 'base')] = bg
            if mode == 'full':
                types = ['mean'] + ([f'svd{r}' for r in RANKS] + ['perm'] if setting in lr_settings else [])
                fams, cons = shared_families(ke, he, states[data], bg, types)
                diag['consistency'] = max(diag['consistency'], cons)
                for t, g in fams.items():
                    blocks[(setting, t)] = g
    if scenario == 'new_cell':
        T = (ttr[:, None] == ttr[None, :]).astype(float); Tq = (tq[:, None] == ttr[None, :]).astype(float)
        blocks[('tissue_only', 'base')] = predict_grid(T, Tq, y_ext)
    if mode == 'full':
        diag['factor_cells'] = {d: st['factor_cells'] for d, st in states.items()}
        diag['lowrank_iters'] = {d: st['lowrank_iters'] for d, st in states.items()}
    meta = dict(fit=fit, query=query, panel=panel, ei=ei, y_ext=y_ext, n_extra=len(ei))
    return blocks, meta, diag


def arm_grid(blocks, arm_blocks):
    return np.concatenate([blocks[(setting, fam)] for setting, fams in arm_blocks for fam in fams])


# ----------------------------------------------------------------------------- analysis
def nb_test(rel, ratio):
    rel = np.asarray(rel, float); k = len(rel)
    mean = rel.mean(); var = rel.var(ddof=1)
    se = np.sqrt((1. / k + ratio) * var)
    tval = mean / se if se > 0 else np.inf * np.sign(mean)
    p = float(2 * student_t.sf(abs(tval), k - 1)) if np.isfinite(tval) else 0.0
    half = student_t.ppf(.975, k - 1) * se
    return dict(mean_rel_pct=float(mean), t=float(tval), p=p, ci95=[float(mean - half), float(mean + half)],
                variance_factor=float(1. / k + ratio))


def fold_scores_weighted(e, obs, use, cw, dw):
    """cw: (B, q) cell weights; dw: (B, D) drug weights. Returns (B,) fold sRMSE."""
    num = cw @ np.where(obs, e, 0.)
    den = cw @ obs.astype(float)
    mse = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    valid = (den > 0) & use[None, :]
    wd = dw * valid
    tot = wd.sum(1)
    return np.sqrt(np.divide((wd * mse).sum(1), tot, out=np.full(len(tot), np.nan), where=tot > 0))


def analyze(scenario, out_dir, fold_data, preds, fold_rows, S):
    comps = comparisons(scenario)
    arms_scores = {}
    for r in fold_rows:
        arms_scores.setdefault(r['arm'], {})[r['fold']] = r
    rng = np.random.default_rng(SEED)
    drug_union = sorted(set().union(*[set(fd['drug_ids'][fd['use']].tolist()) for fd in fold_data]))
    du_index = {d: i for i, d in enumerate(drug_union)}
    dcount = rng.multinomial(len(drug_union), np.full(len(drug_union), 1. / len(drug_union)), size=N_BOOT)
    boot_w = []
    for fd in fold_data:
        if scenario == 'tissue':
            groups = S['tissues'][fd['held']]
        else:
            groups = fd['held'].astype(str)
        uniq, inv = np.unique(groups, return_inverse=True)
        gcount = rng.multinomial(len(uniq), np.full(len(uniq), 1. / len(uniq)), size=N_BOOT)
        cw = gcount[:, inv].astype(float)
        cols = np.array([du_index.get(d, -1) for d in fd['drug_ids']])
        dw = np.where(cols[None, :] >= 0, dcount[:, np.maximum(cols, 0)], 0).astype(float)
        boot_w.append((cw, dw))
    common = set.intersection(*[set(fd['drug_ids'][fd['use']].tolist()) for fd in fold_data])
    results = []
    for new, base, role in comps:
        a = np.array([arms_scores[new][f]['score'] for f in range(5)])
        b = np.array([arms_scores[base][f]['score'] for f in range(5)])
        rel = (b - a) / b * 100
        ratio = float(np.mean([len(fd['held']) / (len(fd['fit']) + (0 if new.endswith('_orig') else fd['n_extra']))
                               for fd in fold_data]))
        nb = nb_test(rel, ratio)
        lofo = [float((np.delete(b, i).mean() - np.delete(a, i).mean()) / np.delete(b, i).mean() * 100) for i in range(5)]
        bs_a = np.zeros(N_BOOT); bs_b = np.zeros(N_BOOT)
        for f, fd in enumerate(fold_data):
            cw, dw = boot_w[f]
            for arr, key in ((bs_a, new), (bs_b, base)):
                e = ((preds[f][key] - fd['truth']) / fd['sd']) ** 2
                arr += fold_scores_weighted(e, fd['obs'], fd['use'], cw, dw) / 5
        bs_rel = (bs_b - bs_a) / bs_b * 100
        cm = []
        for f, fd in enumerate(fold_data):
            use_c = fd['use'] & np.isin(fd['drug_ids'], list(common))
            ma = metrics(fd['truth'], preds[f][new], fd['sd'], use_c)['score']
            mb = metrics(fd['truth'], preds[f][base], fd['sd'], use_c)['score']
            cm.append((ma, mb))
        cm = np.array(cm)
        imp = float((b.mean() - a.mean()) / b.mean() * 100)
        wins = int((a < b).sum())
        verdict = ('pass_push_line' if imp >= 1 and wins >= 4 else 'small_positive_below_line' if imp > 0 else 'no_gain')
        results.append(dict(
            comparison=f'{new} vs {base}', role=role, new=new, base=base,
            new_mean=float(a.mean()), base_mean=float(b.mean()), improvement_pct=imp,
            new_folds=a.tolist(), base_folds=b.tolist(), per_fold_rel_pct=rel.tolist(), folds_won=wins,
            nadeau_bengio=nb, test_train_ratio=ratio, lofo_improvement_pct=lofo, lofo_all_positive=bool(min(lofo) > 0),
            grouped_bootstrap_rel_pct_ci95=[float(np.nanpercentile(bs_rel, 2.5)), float(np.nanpercentile(bs_rel, 97.5))],
            grouped_bootstrap_frac_positive=float(np.mean(bs_rel > 0)),
            common_panel_drugs=len(common),
            common_panel_improvement_pct=float((cm[:, 1].mean() - cm[:, 0].mean()) / cm[:, 1].mean() * 100),
            common_panel_folds_won=int((cm[:, 0] < cm[:, 1]).sum()),
            new_spearman=float(np.mean([arms_scores[new][f]['spearman'] for f in range(5)])),
            base_spearman=float(np.mean([arms_scores[base][f]['spearman'] for f in range(5)])),
            new_centered_mse=float(np.mean([arms_scores[new][f]['centered_mse'] for f in range(5)])),
            base_centered_mse=float(np.mean([arms_scores[base][f]['centered_mse'] for f in range(5)])),
            verdict=verdict))
    return results


# ----------------------------------------------------------------------------- main
def clean(v):
    if isinstance(v, np.ndarray): return clean(v.tolist())
    if isinstance(v, np.generic): return clean(v.item())
    if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)): return [clean(x) for x in v]
    if isinstance(v, float) and not np.isfinite(v): return None
    return v


def main():
    scenario, out_dir, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    assert scenario in ('new_cell', 'tissue') and mode in ('repro', 'full')
    os.makedirs(out_dir, exist_ok=False)
    t0 = time.perf_counter()
    S = np.load(SOURCE, allow_pickle=False); E = np.load(EXTRA, allow_pickle=False)
    S = {k: S[k] for k in ('auc', 'tissues', 'drug_ids', 'sample_ids')}
    E = {k: E[k] for k in ('auc', 'tissues', 'drug_ids', 'sample_ids')}
    assert (E['drug_ids'] == S['drug_ids']).all()
    arms, repro_map, settings, lr_settings = registry(scenario)
    if mode == 'repro':
        arms = {k: v for k, v in arms.items() if k.startswith('base_')}
    arm_names = list(arms) + (['tissue_only'] if scenario == 'new_cell' else [])
    arms_all = dict(arms)
    if scenario == 'new_cell':
        arms_all['tissue_only'] = [('tissue_only', ['base'])]
    repro = {a: dict(grid_max_abs_diff=0.0, pred_max_abs_diff=0.0) for a in repro_map}
    fold_rows, perdrug_rows, fold_data, preds_all, sel_summary, diags = [], [], [], [], {}, []
    for f in range(5):
        total = {}; count = {}; outer_blocks = None
        saved_grid_keys = {v[1] for v in repro_map.values() if v[0] == 'saved'}
        for bank in BANKS:
            blocks, meta, diag = run_bank(scenario, f, bank, S, E, mode, settings, lr_settings)
            diags.append(dict(fold=f, bank=bank, **diag))
            saved = np.load(f'{RUN}/output/{scenario}/fold-{f}/{bank}/grids.npz', allow_pickle=False)
            for a, (src, key) in repro_map.items():
                if src == 'saved':
                    g = arm_grid(blocks, arms_all[a])
                    repro[a]['grid_max_abs_diff'] = max(repro[a]['grid_max_abs_diff'], float(np.max(np.abs(g - saved[key]))))
            if bank == 'outer':
                outer_blocks, outer_meta = blocks, meta
            else:
                truth = S['auc'][np.ix_(meta['query'], meta['panel'])]
                sd = np.maximum(np.nanstd(S['auc'][np.ix_(meta['fit'], meta['panel'])], axis=0), 1e-8)
                for a, ab in arms_all.items():
                    ll, nn = bank_losses(arm_grid(blocks, ab), truth, sd)
                    total[a] = total.get(a, 0) + ll; count[a] = count.get(a, 0) + nn
            print(json.dumps(dict(event='bank_done', scenario=scenario, fold=f, bank=bank,
                                  consistency=diag.get('consistency'), seconds=round(time.perf_counter() - t0, 1))), flush=True)
        P = np.load(f'{RUN}/output/{scenario}/fold-{f}/predictions.npz', allow_pickle=False)
        truth, sd, use = P['truth'], P['train_sd'], P['evaluated']
        assert np.array_equal(P['held'], outer_meta['query']) and np.array_equal(P['panel'], outer_meta['panel'])
        tkp = np.load(f'{TK}/fold-{f}_predictions.npz', allow_pickle=False) if scenario == 'new_cell' else None
        preds = {}
        for a, ab in arms_all.items():
            sel = choose(total[a], count[a], fallback=1)
            preds[a] = take(arm_grid(outer_blocks, ab), sel)
            dec, ncfg = decode(ab, sel) if a != 'tissue_only' else ([], 4)
            if dec:
                fams = [d[1] for d in dec]; gam = [d[2] for d in dec]; als = [d[3] for d in dec]
                ws = [d[0][2] for d in dec]; mus_ = [d[0][3] for d in dec]
                ss = sel_summary.setdefault(a, dict(n_configs=ncfg, family={}, gamma={}, alpha_s={}, w={}, mu={}))
                for key, vals in (('family', fams), ('gamma', gam), ('alpha_s', als), ('w', ws), ('mu', mus_)):
                    for v in vals:
                        ss[key][str(v)] = ss[key].get(str(v), 0) + 1
        preds['null_drug'] = np.broadcast_to(np.nanmean(outer_meta['y_ext'], axis=0), truth.shape).copy()
        for a, (src, key) in repro_map.items():
            ref = P[key] if src == 'saved' else tkp[key]
            repro[a]['pred_max_abs_diff'] = max(repro[a]['pred_max_abs_diff'], float(np.max(np.abs(preds[a] - ref))))
        for a, v in preds.items():
            m = metrics(truth, v, sd, use)
            fold_rows.append(dict(fold=f, arm=a, score=m['score'], spearman=m['spearman'],
                                  centered_mse=m['centered_mse'], drugs=int(use.sum())))
            for j, drug in enumerate(P['drug_ids']):
                perdrug_rows.append(dict(fold=f, arm=a, drug_id=str(drug), count=int(m['count'][j]),
                                         evaluated=bool(use[j]), standardized_mse=float(m['mse'][j]),
                                         spearman=float(m['rho'][j])))
        np.savez_compressed(f'{out_dir}/fold-{f}_predictions.npz', truth=truth, train_sd=sd, evaluated=use,
                            held=P['held'], panel=P['panel'], drug_ids=P['drug_ids'],
                            **{k: np.asarray(v) for k, v in preds.items()})
        fold_data.append(dict(truth=truth, sd=sd, use=use, obs=np.isfinite(truth), held=P['held'], fit=P['fit'],
                              drug_ids=P['drug_ids'], n_extra=outer_meta['n_extra']))
        preds_all.append(preds)
        print(json.dumps(dict(event='fold_done', fold=f, seconds=round(time.perf_counter() - t0, 1))), flush=True)
    with open(f'{out_dir}/fold_metrics.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(fold_rows[0])); w.writeheader(); w.writerows(fold_rows)
    with open(f'{out_dir}/per_drug_metrics.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(perdrug_rows[0])); w.writeheader(); w.writerows(perdrug_rows)
    names = sorted({r['arm'] for r in fold_rows})
    summary = {a: {k: float(np.mean([r[k] for r in fold_rows if r['arm'] == a])) for k in ('score', 'spearman', 'centered_mse')}
               for a in names}
    folds = {a: [r['score'] for r in fold_rows if r['arm'] == a] for a in names}
    repro_ok = all(v['pred_max_abs_diff'] <= 1e-8 and v['grid_max_abs_diff'] <= 1e-8 for v in repro.values())
    out = dict(scenario=scenario, mode=mode, reproduction=repro, reproduction_passed=repro_ok,
               summary=summary, fold_scores=folds, selection=sel_summary, diagnostics=diags)
    if mode == 'full':
        out['comparisons'] = analyze(scenario, out_dir, fold_data, preds_all, fold_rows, S) if repro_ok else 'NOT RUN: reproduction failed'
    out['seconds'] = time.perf_counter() - t0
    with open(f'{out_dir}/summary.json', 'w') as fh:
        json.dump(clean(out), fh, indent=1, ensure_ascii=False)
    print(json.dumps(dict(event='complete', reproduction=repro, reproduction_passed=repro_ok,
                          seconds=round(out['seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()

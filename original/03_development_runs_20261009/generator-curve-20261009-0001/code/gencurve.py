"""Generator-size learning curve of the free virtual layer (M row of the cross-layer matrix), pre-registered in PLAN.md.

M+vRNA / M+vP with the X->Y generator trained on nested 50% / 75% / 100% subsets of the bank's fit cells, and (M->RNA only)
on fit + the identity-audited extra M+RNA lines (m_rna_extra66.npz). Heads, folds, kernels, targets, alpha selection and metric
are UNCHANGED from crosslayer-matrix-20261009-0001: predict_grid / bank_losses / choose / take / metrics / kernel_state / rbf /
rbf_pair / binary_rbf / average / generator cross-fitting are copied verbatim from its code/crosslayer.py
(sha256 c1095b941bdb92b335175291fc5a039f99d7c62b0fdb339e80db2817d11f0da3); layer_inputs is the same computation and
additionally returns the raw target matrices. Only the generator-training set changes.

usage: python gencurve.py SCENARIO STAGE OUT_DIR
  STAGE = dryrun : fold-0 inner-0 only (generator lambda / OOF R2 / virtual width reproduction at f100 vs the matrix bank_info,
                   inner grid check for M) + extra-line counts for every fold/bank of the scenario vs INVENTORY. No outer bank,
                   no outer score.
  STAGE = main   : all folds, all banks, outer scoring; hard gates vs the matrix outer predictions (<= 1e-8).
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
SOURCE = ROOT + '/gdsc-metabolite-full-protein-20260908-0001/output/result/source-inputs.npz'
RNA421 = ROOT + '/rna421-20260906-0002'
EXTRA66 = ROOT + '/extra-supervision-20261008-0001/resources/m_rna_extra66.npz'
RNA_CONFIG = ROOT + '/extra-supervision-20261008-0001/resources/rna_config.json'
MATRIX = ROOT + '/crosslayer-matrix-20261009-0001/output'
SOURCE_SHA = '52d71517a8952e83a1603cd831f09449544102d0ca5ff829ff01be9bf79cd637'
EXTRA66_SHA = 'bd8d13f57b691f579dfcd6210a44f4ab1e05a8691ced62ff2baeb7a6a67883af'
RNA_CONFIG_SHA = '995a42ab7cbf4d9d9c5db3eb9ecf79d1fe8887bc43841846b9202b4e48ed5a6d'

ALPHAS = (.1, 1., 10., 100.)
LAMBDAS = (.1, 1., 10., 100.)
BANKS = ('inner-0', 'inner-1', 'inner-2', 'outer')
INNER = BANKS[:3]
TOP_GENES, TOP_TARGET, P_COVER, OOF_FOLDS = 3000, 512, .5, 3
SCEN_ID = {'new_cell': 0, 'tissue': 1}
BANK_ID = {'inner-0': 0, 'inner-1': 1, 'inner-2': 2, 'outer': 3}
SIZES = (('f50', .5), ('f75', .75), ('f100', 1.))
BLANK = ('SIDM00015', 'SIDM00037', 'SIDM00248', 'SIDM00271')
GATE = 1e-8
# allowed extra M+RNA lines per bank (INVENTORY.md section 3; tissue: blank tissue + the bank's query tissues excluded)
EXPECTED_EXTRA = {'new_cell': {f: {b: 66 for b in BANKS} for f in range(5)},
                  'tissue': {0: {'outer': 51, 'inner-0': 42, 'inner-1': 49, 'inner-2': 44},
                             1: {'outer': 44, 'inner-0': 51, 'inner-1': 47, 'inner-2': 44},
                             2: {'outer': 52, 'inner-0': 48, 'inner-1': 42, 'inner-2': 44},
                             3: {'outer': 51, 'inner-0': 46, 'inner-1': 40, 'inner-2': 49},
                             4: {'outer': 50, 'inner-0': 51, 'inner-1': 40, 'inner-2': 45}}}
VIRTUAL_ARMS = ['M+vRNA_f50', 'M+vRNA_f75', 'M+vRNA_f100', 'M+vRNA_extra', 'M+vRNA_extra_shuf', 'M+vRNA_extra_shufall',
                'M+vP_f50', 'M+vP_f75', 'M+vP_f100']
ARMS = ['M', 'M+RNA', 'M+P'] + VIRTUAL_ARMS
# my arm -> matrix arm whose outer predictions it must reproduce
GATE_PAIRS = {'M': 'M', 'M+RNA': 'M+RNA', 'M+P': 'M+P', 'M+vRNA_f100': 'M+vRNA', 'M+vP_f100': 'M+vP', 'null_drug': 'null_drug'}
GEN_GATE = {'M+vRNA_f100': 'M->RNA', 'M+vP_f100': 'M->P'}


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


# ---------------------------------------------------------------- model core (verbatim from crosslayer.py)
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


# ---------------------------------------------------------------- kernels (verbatim)
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


def average(L, parts):
    K = sum(L[p][0] for p in parts) / len(parts); H = sum(L[p][1] for p in parts) / len(parts)
    return K, H


# ---------------------------------------------------------------- layers (same computation as crosslayer.layer_inputs; also returns raw targets)
def layer_inputs(S, X, fit, query):
    """Returns {layer: (K_fit, H_query, Z_target)}, info, and raw = {'RNA': (xf[:, g], g, cols), 'P': (vf, idx, cols)} where
    the matrix's target is ((raw - fit_mean) / fit_sd)[:, cols] (fit stats with the kernel_state rule)."""
    L, info, raw = {}, {}, {}
    m = S['m']
    K, H, st = rbf_pair(m[fit], m[query])
    L['M'] = (K, H, (m[fit] - st[0]) / st[1])
    info['M'] = dict(features=int(m.shape[1]), width=st[2], target=int(m.shape[1]))
    xf = X[fit]
    g = np.argsort(-xf.var(0), kind='stable')[:TOP_GENES]
    xg = xf[:, g]
    K, H, st = rbf_pair(xg, X[np.ix_(query, g)])
    zf = (xg - st[0]) / st[1]
    L['RNA'] = (K, H, zf[:, :TOP_TARGET])
    raw['RNA'] = (xg, g, np.arange(TOP_TARGET))
    info['RNA'] = dict(features=int(len(g)), width=st[2], target=int(min(TOP_TARGET, len(g))), top_gene_index=int(g[0]))
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
    L['P'] = (.5 * Kval + .5 * Kmask, .5 * Hval + .5 * Hmask, zf[:, tsel])
    raw['P'] = (vf, idx, tsel)
    info['P'] = dict(candidates=int(len(idx)), value_width=st[2], mask_width=mwidth, target=int(len(tsel)),
                     fit_observed_frac=float(np.isfinite(pf).mean()), query_observed_frac=float(np.isfinite(pq).mean()))
    return L, info, raw


# ---------------------------------------------------------------- generator (cross-fitting verbatim; returns coefficients)
def zstats(x):
    mean = x.mean(0); sd = x.std(0)
    return mean, np.where(sd < 1e-8, 1., sd)


def fit_generator(KX, Z, seed):
    """3-way cross-fitted KRR on the generator-training rows (identical to crosslayer.generator), lambda by mean OOF MSE.
    Returns (Z_oof, coef, mu, lambda, r2_mean, r2_median) of the full-set generator."""
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
    r2 = 1 - ((oof - Z) ** 2).sum(0) / np.maximum(((Z - Z.mean(0)) ** 2).sum(0), 1e-12)
    return oof, coef, mu, lam, float(r2.mean()), float(np.median(r2))


def virtual_layer(Kall, Hall, raw_all, cols, n_fit, gen, seed, perm=None):
    """Virtual Y features for all n_fit fit rows and the query rows from a generator trained on rows `gen` of the stacked
    (fit rows first, then extra rows) arrays. Kall: M kernel among stacked rows (fit-only kernel state); Hall: query x stacked.
    Targets: z-scored with the generator-training rows' mean/sd (kernel_state rule) over the full column set, then [:, cols].
    Fit rows inside gen get their OOF prediction; fit rows outside gen and the query rows get the full-set generator's prediction."""
    mean, sd = zstats(raw_all[gen])
    Z = ((raw_all[gen] - mean) / sd)[:, cols]
    if perm is not None:
        Z = Z[perm]
    oof, coef, mu, lam, r2m, r2med = fit_generator(Kall[np.ix_(gen, gen)], Z, seed)
    ingen = np.zeros(len(Kall), bool); ingen[gen] = True
    pos = np.full(len(Kall), -1); pos[gen] = np.arange(len(gen))
    fit_in, fit_out = np.flatnonzero(ingen[:n_fit]), np.flatnonzero(~ingen[:n_fit])
    V = np.empty((n_fit, Z.shape[1]))
    V[fit_in] = oof[pos[fit_in]]
    if len(fit_out):
        V[fit_out] = Kall[np.ix_(fit_out, gen)] @ coef + mu
    Vq = Hall[:, gen] @ coef + mu
    return V, Vq, dict(n_gen=int(len(gen)), n_fit_in_gen=int(len(fit_in)), n_fit_out_gen=int(len(fit_out)),
                       n_extra_in_gen=int(len(gen) - len(fit_in)), lam=lam, oof_r2_mean=r2m, oof_r2_median=r2med,
                       n_targets=int(Z.shape[1]))


def allowed_extra(scenario, query_tissues, extra_tissues):
    if scenario == 'new_cell':
        return np.arange(len(extra_tissues))
    held = set(np.unique(query_tissues).tolist())
    return np.flatnonzero([(t != '') and (t not in held) for t in extra_tissues])


# ---------------------------------------------------------------- one bank
def run_bank(S, X, E, scenario, f, bank, MB):
    G = np.load(f'{EXT}/{scenario}/fold-{f}/{bank}/geometry.npz', allow_pickle=False)
    fit, query, panel = G['fit'], G['query'], G['panel']
    n = len(fit)
    y = S['auc'][np.ix_(fit, panel)]
    L, info, raw = layer_inputs(S, X, fit, query)
    kdiff = max(float(np.max(np.abs(L['M'][0] - G['m_only_kernel'][:n, :n]))),
                float(np.max(np.abs(L['M'][1] - G['m_only_cross'][:, :n]))))
    np.testing.assert_allclose(L['M'][0], G['m_only_kernel'][:n, :n], rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(L['M'][1], G['m_only_cross'][:, :n], rtol=1e-11, atol=1e-12)
    KX, HX = L['M'][0], L['M'][1]
    grids = {'M': predict_grid(KX, HX, y), 'M+RNA': predict_grid(*average(L, ('M', 'RNA')), y),
             'M+P': predict_grid(*average(L, ('M', 'P')), y)}
    # nested generator-training subsets of the fit cells (one permutation per bank)
    sub = np.random.default_rng([20261009, SCEN_ID[scenario], f, BANK_ID[bank], 7]).permutation(n)
    gens = {tag: np.sort(sub[:int(np.ceil(frac * n))]) for tag, frac in SIZES}
    assert np.array_equal(gens['f100'], np.arange(n)) and set(gens['f50'].tolist()) <= set(gens['f75'].tolist())
    # allowed extra M+RNA lines for this bank; M kernel extended with the fit-only kernel state
    ok = allowed_extra(scenario, S['tissues'][query], E['tissues'])
    assert len(ok) == EXPECTED_EXTRA[scenario][f][bank], (scenario, f, bank, len(ok))
    st = kernel_state(S['m'][fit]); me = E['m'][ok]
    Kall = np.block([[KX, rbf(S['m'][fit], me, st)], [rbf(me, S['m'][fit], st), rbf(me, me, st)]])
    Hall = np.hstack([HX, rbf(S['m'][query], me, st)])
    xg, g, rcols = raw['RNA']
    rna_all = np.vstack([xg, E['rna'][np.ix_(ok, g)]])
    vf, _, pcols = raw['P']
    seed_oof = stable_seed('oof', scenario, f, bank)
    gen_info = {}

    def add(arm, V, Vq, gi):
        Kv, Hv, vst = rbf_pair(V, Vq)
        grids[arm] = predict_grid(.5 * KX + .5 * Kv, .5 * HX + .5 * Hv, y)
        gi['virtual_width'] = vst[2]; gen_info[arm] = gi

    for tag, _ in SIZES:
        add(f'M+vRNA_{tag}', *virtual_layer(KX, HX, xg, rcols, n, gens[tag], seed_oof))
        add(f'M+vP_{tag}', *virtual_layer(KX, HX, vf, pcols, n, gens[tag], seed_oof))
    ge = np.arange(n + len(ok))
    add('M+vRNA_extra', *virtual_layer(Kall, Hall, rna_all, rcols, n, ge, seed_oof))
    pe = np.arange(n, n + len(ok)); perm = np.arange(n + len(ok))
    perm[pe] = pe[np.random.default_rng(stable_seed('pair_shuffle_extra', scenario, f, bank)).permutation(len(pe))]
    assert not np.array_equal(perm, np.arange(len(perm)))
    add('M+vRNA_extra_shuf', *virtual_layer(Kall, Hall, rna_all, rcols, n, ge, seed_oof, perm))
    permall = np.random.default_rng(stable_seed('pair_shuffle', scenario, f, bank)).permutation(n + len(ok))
    assert not np.array_equal(permall, np.arange(len(permall)))
    add('M+vRNA_extra_shufall', *virtual_layer(Kall, Hall, rna_all, rcols, n, ge, seed_oof, permall))
    # f100 generator reproduction vs the matrix run's bank_info
    gdiff = 0.
    for arm, key in GEN_GATE.items():
        ref = MB[f'fold-{f}/{bank}']['generators'][key]
        d = max(abs(gen_info[arm]['oof_r2_mean'] - ref['oof_r2_mean']), abs(gen_info[arm]['oof_r2_median'] - ref['oof_r2_median']),
                abs(gen_info[arm]['virtual_width'] - ref['virtual_width']))
        gen_info[arm]['matrix_lam_equal'] = bool(gen_info[arm]['lam'] == ref['lam'])
        gen_info[arm]['matrix_max_abs_diff'] = d
        assert gen_info[arm]['matrix_lam_equal']
        gdiff = max(gdiff, d)
    assert set(grids) == set(ARMS)
    info.update(n_fit=int(n), n_query=int(len(query)), n_panel=int(len(panel)), n_extra_allowed=int(len(ok)),
                extra_sample_ids=E['sample_ids'][ok].tolist(), gen_sizes={tag: int(len(gens[tag])) for tag, _ in SIZES},
                kernel_block_max_abs_diff=kdiff, generator_f100_max_abs_diff=gdiff, generators=gen_info)
    return grids, dict(fit=fit, query=query, panel=panel), info


# ---------------------------------------------------------------- data
def load_ctx():
    if os.environ.get('CREED_SKIP_SHA') != '1':
        assert sha(SOURCE) == SOURCE_SHA, 'source sha mismatch'
        assert sha(EXTRA66) == EXTRA66_SHA, 'extra66 sha mismatch'
        assert sha(RNA_CONFIG) == RNA_CONFIG_SHA, 'rna_config sha mismatch'
    S = np.load(SOURCE, allow_pickle=False)
    S = {k: S[k] for k in ('m', 'auc', 'p', 'tissues', 'sample_ids', 'drug_ids', 'protein_ids', 'metabolite_ids')}
    X = np.load(RNA421 + '/rna_log_cpm.npy')
    assert (np.load(RNA421 + '/sample_ids.npy') == S['sample_ids']).all(), 'RNA row order differs from source'
    assert X.shape[0] == len(S['sample_ids']) and np.isfinite(X).all() and np.isfinite(S['m']).all()
    E = np.load(EXTRA66, allow_pickle=False)
    assert (E['gene_ids'] == np.load(RNA421 + '/gene_ids.npy', allow_pickle=False)).all(), 'extra RNA gene panel differs'
    assert E['rna_log_cpm'].shape[1] == X.shape[1] and E['m'].shape[1] == S['m'].shape[1]
    assert np.isfinite(E['m']).all() and np.isfinite(E['rna_log_cpm']).all()
    ids = E['sample_ids'].tolist()
    cfg = json.load(open(RNA_CONFIG))
    assert sorted(ids) == sorted(cfg['m_rna_ids']) and len(set(ids)) == 66, 'extra ids differ from rna_config m_rna_ids'
    assert not (set(ids) & set(S['sample_ids'].tolist())) and not (set(ids) & set(cfg['excluded_old608']))
    assert sorted(E['sample_ids'][E['tissues'] == ''].tolist()) == sorted(BLANK)
    E = {'m': E['m'], 'rna': E['rna_log_cpm'], 'tissues': E['tissues'], 'sample_ids': E['sample_ids']}
    return S, X, E


def count_extra(S, E, scenario):
    rows = {}
    for f in range(5):
        for bank in BANKS:
            G = np.load(f'{EXT}/{scenario}/fold-{f}/{bank}/geometry.npz', allow_pickle=False)
            ok = allowed_extra(scenario, S['tissues'][G['query']], E['tissues'])
            rows[f'fold-{f}/{bank}'] = dict(n=int(len(ok)), expected=EXPECTED_EXTRA[scenario][f][bank],
                                            held_tissues=sorted(set(S['tissues'][G['query']].tolist())))
            assert len(ok) == EXPECTED_EXTRA[scenario][f][bank], (scenario, f, bank, len(ok))
    return rows


def main():
    scenario, stage, out = sys.argv[1], sys.argv[2], sys.argv[3]
    assert scenario in SCEN_ID and stage in ('dryrun', 'main')
    os.makedirs(out, exist_ok=False)
    t0 = time.perf_counter()
    S, X, E = load_ctx()
    MB = json.load(open(f'{MATRIX}/{scenario}/bank_info.json'))
    json.dump(dict(scenario=scenario, stage=stage, script_sha256=sha(__file__), source_sha256=SOURCE_SHA, extra66_sha256=EXTRA66_SHA,
                   rna_config_sha256=RNA_CONFIG_SHA, alphas=ALPHAS, lambdas=LAMBDAS, top_genes=TOP_GENES, top_target=TOP_TARGET,
                   p_cover=P_COVER, oof_folds=OOF_FOLDS, sizes=SIZES, arms=ARMS + ['null_drug'], gate=GATE, rna_shape=list(X.shape),
                   subset_seed='np.random.default_rng([20261009, scenario_id, fold, bank_id, 7])', expected_extra=EXPECTED_EXTRA[scenario]),
              open(f'{out}/protocol.json', 'w'), indent=1)
    counts = count_extra(S, E, scenario)
    json.dump(counts, open(f'{out}/extra_counts.json', 'w'), indent=1)
    log(event='extra_counts_ok', scenario=scenario, outer=[counts[f'fold-{f}/outer']['n'] for f in range(5)])
    repro = dict(kernel_block_max_abs_diff=0., generator_f100_max_abs_diff=0., grid_M_vs_matrix_saved=0.,
                 **{f'pred_{a}_vs_matrix_{b}': 0. for a, b in GATE_PAIRS.items()}, selection_equal=True)
    fold_rows, perdrug_rows, bank_info, sel_counts = [], [], {}, {}
    for f in ((0,) if stage == 'dryrun' else range(5)):
        loss, outer, ometa, fold_bank = {}, None, None, {}
        for bank in BANKS:
            if stage == 'dryrun' and bank != 'inner-0':
                continue
            tb = time.perf_counter()
            grids, meta, info = run_bank(S, X, E, scenario, f, bank, MB)
            saved = np.load(f'{EXT}/{scenario}/fold-{f}/{bank}/grids.npz', allow_pickle=False)
            d1 = float(np.max(np.abs(grids['M'] - saved['original__m_only'])))
            repro['grid_M_vs_matrix_saved'] = max(repro['grid_M_vs_matrix_saved'], d1)
            repro['kernel_block_max_abs_diff'] = max(repro['kernel_block_max_abs_diff'], info['kernel_block_max_abs_diff'])
            repro['generator_f100_max_abs_diff'] = max(repro['generator_f100_max_abs_diff'], info['generator_f100_max_abs_diff'])
            if bank == 'outer':
                outer, ometa = grids, meta
            else:
                truth = S['auc'][np.ix_(meta['query'], meta['panel'])]
                sd = np.maximum(np.nanstd(S['auc'][np.ix_(meta['fit'], meta['panel'])], axis=0), 1e-8)
                for a, g in grids.items():
                    loss.setdefault(a, {})[bank] = bank_losses(g, truth, sd)
            bank_info[f'fold-{f}/{bank}'] = info; fold_bank[bank] = info
            log(event='bank_done', scenario=scenario, fold=f, bank=bank, grid_diff_M=d1, gen_f100_diff=info['generator_f100_max_abs_diff'],
                n_extra=info['n_extra_allowed'], gen_sizes=info['gen_sizes'],
                lam={a: v['lam'] for a, v in info['generators'].items()}, r2={a: round(v['oof_r2_mean'], 4) for a, v in info['generators'].items()},
                seconds=round(time.perf_counter() - tb, 1), total=round(time.perf_counter() - t0, 1))
        with open(f'{os.path.dirname(os.path.abspath(out))}/extra_ids_{scenario}_fold-{f}.txt', 'w') as fh:
            fh.write(f'# allowed extra M+RNA lines per bank, {scenario} fold-{f}: ' +
                     ', '.join(f"{b}={fold_bank[b]['n_extra_allowed']}" for b in fold_bank) + '\n')
            for b in fold_bank:
                for s in fold_bank[b]['extra_sample_ids']:
                    fh.write(f'{b}\t{s}\n')
        json.dump(fold_bank, open(f'{out}/fold-{f}_bank_info.json', 'w'), indent=1)

        def summed(arm):
            return sum(loss[arm][b][0] for b in INNER), sum(loss[arm][b][1] for b in INNER)

        if stage == 'dryrun':
            json.dump(bank_info, open(f'{out}/bank_info.json', 'w'), indent=1)
            res = dict(scenario=scenario, stage=stage, reproduction=repro, seconds=time.perf_counter() - t0,
                       generators={a: {k: v for k, v in g.items()} for a, g in fold_bank['inner-0']['generators'].items()})
            json.dump(res, open(f'{out}/summary.json', 'w'), indent=1)
            log(event='dryrun_done', scenario=scenario, reproduction=repro, seconds=round(res['seconds'], 1))
            return
        P = np.load(f'{EXT}/{scenario}/fold-{f}/predictions.npz', allow_pickle=False)
        truth, sd, use = P['truth'], P['train_sd'], P['evaluated']
        assert (P['panel'] == ometa['panel']).all() and (P['held'] == ometa['query']).all() and (P['fit'] == ometa['fit']).all()
        preds, sels = {}, {}
        for a in ARMS:
            sel = choose(*summed(a), fallback=1)
            sels[a] = sel; preds[a] = take(outer[a], sel)
        preds['null_drug'] = np.broadcast_to(np.nanmean(S['auc'][np.ix_(ometa['fit'], ometa['panel'])], axis=0), truth.shape).copy()
        # hard gates vs the matrix run's saved outer predictions
        MX = np.load(f'{MATRIX}/{scenario}/fold-{f}_predictions.npz', allow_pickle=False)
        for k in ('truth', 'train_sd', 'evaluated', 'panel', 'held', 'fit'):
            assert np.array_equal(MX[k], P[k], equal_nan=(k in ('truth', 'train_sd'))), k  # truth has NaN (unobserved)
        for a, b in GATE_PAIRS.items():
            repro[f'pred_{a}_vs_matrix_{b}'] = max(repro[f'pred_{a}_vs_matrix_{b}'], float(np.max(np.abs(preds[a] - MX['pred__' + b]))))
            if a != 'null_drug':
                repro['selection_equal'] = bool(repro['selection_equal'] and np.array_equal(sels[a], MX['sel__' + b]))
        np.savez_compressed(f'{out}/fold-{f}_predictions.npz', truth=truth, train_sd=sd, evaluated=use, panel=P['panel'],
                            drug_ids=P['drug_ids'], held=P['held'], fit=P['fit'],
                            **{'pred__' + k: v for k, v in preds.items()}, **{'sel__' + k: v for k, v in sels.items()})
        np.savez_compressed(f'{out}/fold-{f}_inner_losses.npz',
                            **{f'{a}__{b}__loss': v[0] for a, d in loss.items() for b, v in d.items()},
                            **{f'{a}__{b}__count': v[1] for a, d in loss.items() for b, v in d.items()})
        sel_counts[f] = {a: np.bincount(s, minlength=len(ALPHAS)).tolist() for a, s in sels.items()}
        for a, v in preds.items():
            mm = metrics(truth, v, sd, use)
            gi = fold_bank['outer']['generators'].get(a)
            fold_rows.append(dict(fold=f, arm=a, score=mm['score'], spearman=mm['spearman'], centered_mse=mm['centered_mse'],
                                  drugs=int(use.sum()), n_fit=int(len(P['fit'])), n_held=int(len(P['held'])),
                                  n_gen_outer=(gi['n_gen'] if gi else ''), lam_outer=(gi['lam'] if gi else ''),
                                  oof_r2_outer=(gi['oof_r2_mean'] if gi else '')))
            for j in range(len(sd)):
                perdrug_rows.append(dict(fold=f, arm=a, drug_id=str(P['drug_ids'][j]), evaluated=bool(use[j]),
                                         standardized_mse=float(mm['mse'][j]), spearman=float(mm['rho'][j])))
        log(event='fold_done', scenario=scenario, fold=f, total=round(time.perf_counter() - t0, 1),
            scores={r['arm']: round(r['score'], 6) for r in fold_rows if r['fold'] == f},
            gates={k: v for k, v in repro.items() if k.startswith('pred_')})
    with open(f'{out}/fold_metrics.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(fold_rows[0])); w.writeheader(); w.writerows(fold_rows)
    with open(f'{out}/per_drug_metrics.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(perdrug_rows[0])); w.writeheader(); w.writerows(perdrug_rows)
    json.dump(bank_info, open(f'{out}/bank_info.json', 'w'), indent=1)
    names = list(dict.fromkeys(r['arm'] for r in fold_rows))
    summary = {a: {k: float(np.mean([r[k] for r in fold_rows if r['arm'] == a])) for k in ('score', 'spearman', 'centered_mse')} for a in names}
    folds = {a: [r['score'] for r in fold_rows if r['arm'] == a] for a in names}
    passed = bool(max(v for k, v in repro.items() if k.startswith('pred_') or k.startswith('grid_')) <= GATE
                  and repro['generator_f100_max_abs_diff'] <= 1e-9)
    res = dict(scenario=scenario, stage=stage, reproduction=dict(**repro, passed=passed), summary=summary, fold_scores=folds,
               selection_counts=sel_counts, seconds=time.perf_counter() - t0)
    json.dump(res, open(f'{out}/summary.json', 'w'), indent=1)
    log(event='done', scenario=scenario, reproduction=res['reproduction'], seconds=round(time.perf_counter() - t0, 1))


if __name__ == '__main__':
    main()

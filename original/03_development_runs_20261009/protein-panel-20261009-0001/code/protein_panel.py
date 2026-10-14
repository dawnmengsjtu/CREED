"""Few-layer deployment: M (+ tissue label in new_cell) + a SMALL MEASURED protein panel (pre-registered in PLAN.md).

Independent implementation (functions copied from the local tissue_kernel.py reference; no project imports).
Reads only: saved banks of extra-supervision-20261008-0001 (original-data top-left blocks), source-inputs.npz,
privileged-geometry protein_indices and strong-teacher predictions (teacher reproduction only).
Writes only to OUT_DIR (a new directory inside the protein-panel run).

usage: python protein_panel.py SCENARIO STAGE OUT_DIR
  STAGE = dryrun : fold-0 inner banks only (no outer bank, no outer score) -> engineering check
  STAGE = repro  : saved arms (m_only, m_only__protein, [vfm, vfm__protein]) + teacher repro, all folds
  STAGE = main   : all pre-registered arms, all folds
"""
import json
import os
import sys
import time
import numpy as np
from scipy.linalg import eigh, cho_factor, cho_solve
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, spearmanr

RUN = '/public/home/mengxl/creed-research-runs'
EXT = RUN + '/extra-supervision-20261008-0001/output'
SOURCE = RUN + '/gdsc-metabolite-full-protein-20260908-0001/output/result/source-inputs.npz'
PG = RUN + '/privileged-geometry-20261008-0001/output'
ST = RUN + '/strong-teacher-recoverability-20261008-0001/output'
ST_DIR = {'new_cell': 'new_cell', 'tissue': 'tissue_v2'}
SCEN_ID = {'new_cell': 0, 'tissue': 1}
BANK_ID = {'inner-0': 0, 'inner-1': 1, 'inner-2': 2, 'outer': 3}

ALPHAS = (.1, 1., 10., 100.)
MUS = (0., .1, 1., 10., 100., 1000.)
WS_NEW = (0., .25, 1.)
KS = (8, 16, 32, 64)
RKS = (16, 32)
SEEDS = (0, 1, 2, 3, 4)
GRAPHS = ('protein', 'mask', 'shuffle')
BANKS = ('inner-0', 'inner-1', 'inner-2', 'outer')
COVER = 0.9
MIN_PAIR = 10


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


def rbf_kernel(x, q):
    """Fit-moment standardization, RBF with median positive squared distance among fit rows."""
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
                centered_mse=float((mse - bias * bias)[use].mean()), mse=mse)


def tissue_tokens(t, tag):
    return np.array([x if x else f'__unknown_{tag}_{i}' for i, x in enumerate(t)])


# ---------------------------------------------------------------- panel selection (fit cells only)
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


def candidates(p_fit):
    obs = np.isfinite(p_fit); cnt = obs.sum(0); frac = cnt / len(p_fit)
    mean = np.divide(np.where(obs, p_fit, 0).sum(0), cnt, out=np.zeros(p_fit.shape[1]), where=cnt > 0)
    var = np.divide(np.where(obs, (p_fit - mean) ** 2, 0).sum(0), cnt, out=np.zeros(p_fit.shape[1]), where=cnt > 0)
    return np.flatnonzero((frac >= COVER) & (var > 1e-12))


# ---------------------------------------------------------------- arms
def wkey(w):
    return f'w{w}'


def arm_keys(scenario):
    ws = WS_NEW if scenario == 'new_cell' else (0.,)

    def g(base, w, graph):
        return [f'{base}|{wkey(w)}'] + [f'{base}|{wkey(w)}|{graph}|{mu}' for mu in MUS[1:]]
    A = {}
    A['m_only'] = ['M|w0.0']
    A['m_only__protein'] = g('M', 0., 'protein')
    for k in KS:
        A[f'm_panel_k{k}'] = [f'P{k}|w0.0']
    A['m_panel_sel'] = [f'P{k}|w0.0' for k in KS]
    for gr in GRAPHS:
        A[f'm_panel_sel__{gr}'] = sum((g(f'P{k}', 0., gr) for k in KS), [])
    for k in RKS:
        for s in SEEDS:
            A[f'random_k{k}_s{s}'] = [f'R{k}s{s}|w0.0']
    A['m_allcand'] = ['A|w0.0']
    A['teacher_repro'] = ['TP']
    if scenario == 'new_cell':
        A['tissue_only'] = ['T']
        A['m_tissue'] = [f'M|{wkey(w)}' for w in ws]
        A['m_tissue__protein'] = sum((g('M', w, 'protein') for w in ws), [])
        for k in KS:
            A[f'm_panel_tissue_k{k}'] = [f'P{k}|{wkey(w)}' for w in ws]
        A['m_panel_tissue_sel'] = [f'P{k}|{wkey(w)}' for k in KS for w in ws]
        for gr in GRAPHS:
            A[f'm_panel_tissue_sel__{gr}'] = sum((g(f'P{k}', w, gr) for k in KS for w in ws), [])
        for k in RKS:
            for s in SEEDS:
                A[f'random_tissue_k{k}_s{s}'] = [f'R{k}s{s}|{wkey(w)}' for w in ws]
        A['m_allcand_tissue'] = [f'A|{wkey(w)}' for w in ws]
    else:
        A['vfm'] = ['V|w0.0']
        A['vfm__protein'] = g('V', 0., 'protein')
        for k in KS:
            A[f'vfm_panel_k{k}'] = [f'VP{k}|w0.0']
        A['vfm_panel_sel'] = [f'VP{k}|w0.0' for k in KS]
        for gr in GRAPHS:
            A[f'vfm_panel_sel__{gr}'] = sum((g(f'VP{k}', 0., gr) for k in KS), [])
        A['vfm_allcand'] = ['VA|w0.0']
    return A


REPRO_ARMS = {'new_cell': ('m_only', 'm_only__protein', 'teacher_repro'),
              'tissue': ('m_only', 'm_only__protein', 'vfm', 'vfm__protein', 'teacher_repro')}
SAVED_NAME = {'m_only': 'original__m_only', 'm_only__protein': 'original__m_only__protein',
              'vfm': 'original__virtual_fm', 'vfm__protein': 'original__virtual_fm__protein'}


# ---------------------------------------------------------------- one bank
def run_bank(S, scenario, f, bank, stage, emit, info):
    G = np.load(f'{EXT}/{scenario}/fold-{f}/{bank}/geometry.npz', allow_pickle=False)
    fit, query, panel = G['fit'], G['query'], G['panel']; n = len(fit)
    auc, p, tis = S['auc'], S['p'], S['tissues']
    y = auc[np.ix_(fit, panel)]
    KM, HM = G['m_only_kernel'][:n, :n].copy(), G['m_only_cross'][:, :n].copy()
    Ls = {}
    for gr in GRAPHS:
        L = G['L_' + gr]
        assert np.all(L[n:] == 0) and np.all(L[:, n:] == 0)
        Ls[gr] = L[:n, :n].copy()
    full = stage in ('main', 'dryrun')
    ws = WS_NEW if scenario == 'new_cell' else (0.,)
    if scenario == 'new_cell':
        ttr, tq = tissue_tokens(tis[fit], 'tr'), tissue_tokens(tis[query], 'q')
        T = (ttr[:, None] == ttr[None, :]).astype(float); Tq = (tq[:, None] == ttr[None, :]).astype(float)
    else:
        T = Tq = None

    def add_base(name, K, H, graphs):
        for w in ws:
            Kw, Hw = (K + w * T, H + w * Tq) if w else (K, H)
            emit(f'{name}|{wkey(w)}', predict_grid(Kw, Hw, y))
            if not graphs:
                continue
            vals, vecs = eigh((Kw + Kw.T) / 2)
            assert vals.min() > -1e-8
            root = (vecs * np.sqrt(np.maximum(vals, 0))) @ vecs.T
            for gr in graphs:
                L = Ls[gr]
                for mu in MUS[1:]:
                    a = np.eye(n) + mu * root @ L @ root; a = (a + a.T) / 2
                    ke = root @ cho_solve(cho_factor(a, lower=True), root); ke = (ke + ke.T) / 2
                    he = Hw - mu * Hw @ L @ ke
                    emit(f'{name}|{wkey(w)}|{gr}|{mu}', predict_grid(ke, he, y))

    # --- saved / reference bases
    if full:
        add_base('M', KM, HM, ('protein',))
    else:
        emit('M|w0.0', predict_grid(KM, HM, y))
        vals, vecs = eigh((KM + KM.T) / 2); root = (vecs * np.sqrt(np.maximum(vals, 0))) @ vecs.T
        for mu in MUS[1:]:
            a = np.eye(n) + mu * root @ Ls['protein'] @ root; a = (a + a.T) / 2
            ke = root @ cho_solve(cho_factor(a, lower=True), root); ke = (ke + ke.T) / 2
            emit(f'M|w0.0|protein|{mu}', predict_grid(ke, HM - mu * HM @ Ls['protein'] @ ke, y))
    if scenario == 'tissue':
        KV, HV = G['virtual_fm_kernel'][:n, :n].copy(), G['virtual_fm_cross'][:, :n].copy()
        vals, vecs = eigh((KV + KV.T) / 2); root = (vecs * np.sqrt(np.maximum(vals, 0))) @ vecs.T
        emit('V|w0.0', predict_grid(KV, HV, y))
        for mu in MUS[1:]:
            a = np.eye(n) + mu * root @ Ls['protein'] @ root; a = (a + a.T) / 2
            ke = root @ cho_solve(cho_factor(a, lower=True), root); ke = (ke + ke.T) / 2
            emit(f'V|w0.0|protein|{mu}', predict_grid(ke, HV - mu * HV @ Ls['protein'] @ ke, y))
    # teacher reproduction: 0.5 M + 0.5 measured-P RBF on the teacher's own eligible protein list
    PGG = np.load(f'{PG}/{scenario}/fold-{f}/{bank}/geometry.npz', allow_pickle=False)
    assert np.array_equal(PGG['fit'], fit) and np.array_equal(PGG['query'], query)
    pk, ph = protein_kernel(p, fit, query, PGG['protein_indices'])
    emit('TP', predict_grid(.5 * KM + .5 * pk, .5 * HM + .5 * ph, y))
    info.update(n_fit=int(n), n_query=int(len(query)), teacher_protein_count=int(len(PGG['protein_indices'])))
    if not full:
        return G
    # --- new arms
    t0 = time.perf_counter()
    cand = candidates(p[fit])
    score = spearman_scores(p[np.ix_(fit, cand)], y)
    order = cand[np.argsort(-score, kind='stable')]
    info.update(candidate_count=int(len(cand)), spearman_seconds=round(time.perf_counter() - t0, 1),
                top64=order[:64].tolist(), top64_score=np.sort(score)[::-1][:64].round(6).tolist())
    if scenario == 'new_cell':
        emit('T', predict_grid(T, Tq, y))
    for k in KS:
        idx = np.sort(order[:k])
        pk, ph = protein_kernel(p, fit, query, idx)
        add_base(f'P{k}', .5 * KM + .5 * pk, .5 * HM + .5 * ph, GRAPHS)
        if scenario == 'tissue':
            add_base(f'VP{k}', .5 * KV + .5 * pk, .5 * HV + .5 * ph, GRAPHS)
    rand = {}
    for s in SEEDS:
        rng = np.random.default_rng([20261009, SCEN_ID[scenario], f, BANK_ID[bank], s])
        perm = rng.permutation(cand)
        for k in RKS:
            idx = np.sort(perm[:k]); rand[f'R{k}s{s}'] = idx.tolist()
            pk, ph = protein_kernel(p, fit, query, idx)
            add_base(f'R{k}s{s}', .5 * KM + .5 * pk, .5 * HM + .5 * ph, ())
    info['random_panels'] = rand
    pk, ph = protein_kernel(p, fit, query, cand)
    add_base('A', .5 * KM + .5 * pk, .5 * HM + .5 * ph, ())
    if scenario == 'tissue':
        emit('VA|w0.0', predict_grid(.5 * KV + .5 * pk, .5 * HV + .5 * ph, y))
    return G


# ---------------------------------------------------------------- driver
def main():
    scenario, stage, out = sys.argv[1], sys.argv[2], sys.argv[3]
    assert scenario in ('new_cell', 'tissue') and stage in ('dryrun', 'repro', 'main')
    os.makedirs(out, exist_ok=False)
    t0 = time.perf_counter()
    S = np.load(SOURCE, allow_pickle=False)
    S = {k: S[k] for k in ('m', 'auc', 'p', 'tissues', 'sample_ids', 'drug_ids', 'protein_ids')}
    arms = arm_keys(scenario)
    if stage == 'repro':
        arms = {a: arms[a] for a in REPRO_ARMS[scenario]}
    needed = sorted({k for v in arms.values() for k in v})
    repro_keys = {k for a in SAVED_NAME if a in arms for k in arms[a]}
    repro = dict(grid_max_abs_diff={}, pred_max_abs_diff={}, teacher_pred_max_abs_diff=0.0,
                 teacher_m_only_vs_saved_max_abs_diff=0.0, fold_split_identical=True)
    fold_rows, sel_record, bank_info = [], {}, {}
    folds = (0,) if stage == 'dryrun' else range(5)
    for f in folds:
        total = {k: 0 for k in needed}; count = {k: 0 for k in needed}
        outer = {}
        for bank in BANKS:
            if stage == 'dryrun' and bank == 'outer':
                continue
            info = {}; rstore = {}
            if bank == 'outer':
                def emit(key, grid):
                    if key in total:
                        outer[key] = grid
                    if key in repro_keys:
                        rstore[key] = grid
            else:
                Gq = np.load(f'{EXT}/{scenario}/fold-{f}/{bank}/geometry.npz', allow_pickle=False)
                truth = S['auc'][np.ix_(Gq['query'], Gq['panel'])]
                sd = np.maximum(np.nanstd(S['auc'][np.ix_(Gq['fit'], Gq['panel'])], axis=0), 1e-8)

                def emit(key, grid, truth=truth, sd=sd):
                    if key in total:
                        ll, nn = bank_losses(grid, truth, sd); total[key] = total[key] + ll; count[key] = count[key] + nn
                    if key in repro_keys:
                        rstore[key] = grid
            G = run_bank(S, scenario, f, bank, stage, emit, info)
            # reproduction of saved grids (all inner banks + outer bank)
            saved = np.load(f'{EXT}/{scenario}/fold-{f}/{bank}/grids.npz', allow_pickle=False)
            for a in SAVED_NAME:
                if a in arms:
                    mine = np.concatenate([rstore[k] for k in arms[a]])
                    d = float(np.max(np.abs(mine - saved[SAVED_NAME[a]])))
                    repro['grid_max_abs_diff'][a] = max(repro['grid_max_abs_diff'].get(a, 0.), d)
            bank_info[f'fold-{f}/{bank}'] = info
            print(json.dumps(dict(event='bank_done', fold=f, bank=bank, candidates=info.get('candidate_count'),
                                  seconds=round(time.perf_counter() - t0, 1))), flush=True)
        if stage == 'dryrun':
            sel = {a: choose(np.concatenate([total[k] for k in v]), np.concatenate([count[k] for k in v])) for a, v in arms.items()}
            np.savez_compressed(f'{out}/dryrun_fold-0_inner_selection.npz', **sel)
            break
        P = np.load(f'{EXT}/{scenario}/fold-{f}/predictions.npz', allow_pickle=False)
        truth, sd, use = P['truth'], P['train_sd'], P['evaluated']
        assert np.array_equal(P['fit'], G['fit']) and np.array_equal(P['held'], G['query'])
        preds, sels = {}, {}
        for a, keys in arms.items():
            tot = np.concatenate([total[k] for k in keys]); cnt = np.concatenate([count[k] for k in keys])
            sel = choose(tot, cnt, fallback=1)
            preds[a] = take(np.concatenate([outer[k] for k in keys]), sel)
            sels[a] = sel
        for a in SAVED_NAME:
            if a in preds:
                d = float(np.max(np.abs(preds[a] - P[SAVED_NAME[a]])))
                repro['pred_max_abs_diff'][a] = max(repro['pred_max_abs_diff'].get(a, 0.), d)
        T = np.load(f'{ST}/{ST_DIR[scenario]}/fold-{f}/predictions.npz', allow_pickle=False)
        same = np.array_equal(T['fit'], P['fit']) and np.array_equal(T['held'], P['held']) and np.array_equal(T['panel'], P['panel'])
        repro['fold_split_identical'] &= bool(same)
        repro['teacher_pred_max_abs_diff'] = max(repro['teacher_pred_max_abs_diff'], float(np.max(np.abs(preds['teacher_repro'] - T['measured_p']))))
        repro['teacher_m_only_vs_saved_max_abs_diff'] = max(repro['teacher_m_only_vs_saved_max_abs_diff'], float(np.max(np.abs(T['m_only'] - P['original__m_only']))))
        preds['teacher_saved'] = T['measured_p']
        fit_y = S['auc'][np.ix_(P['fit'], P['panel'])]
        preds['null_drug'] = np.broadcast_to(np.nanmean(fit_y, axis=0), truth.shape).copy()
        for a, v in preds.items():
            m = metrics(truth, v, sd, use)
            fold_rows.append(dict(fold=f, arm=a, score=m['score'], spearman=m['spearman'], centered_mse=m['centered_mse']))
        np.savez_compressed(f'{out}/fold-{f}_predictions.npz', truth=truth, train_sd=sd, evaluated=use, fit=P['fit'],
                            held=P['held'], panel=P['panel'], **preds)
        np.savez_compressed(f'{out}/fold-{f}_selection.npz', **{a: s for a, s in sels.items()},
                            **{'loss__' + a: np.concatenate([total[k] for k in v]) for a, v in arms.items()},
                            **{'count__' + a: np.concatenate([count[k] for k in v]) for a, v in arms.items()})
        sel_record[f] = {a: np.bincount(s, minlength=len(arms[a]) * len(ALPHAS)).tolist() for a, s in sels.items()}
        print(json.dumps(dict(event='fold_done', fold=f, seconds=round(time.perf_counter() - t0, 1))), flush=True)
    json.dump(bank_info, open(f'{out}/bank_info.json', 'w'))
    if stage == 'dryrun':
        print(json.dumps(dict(event='dryrun_done', seconds=round(time.perf_counter() - t0, 1),
                              candidates={k: v.get('candidate_count') for k, v in bank_info.items()})), flush=True)
        return
    import csv
    with open(f'{out}/fold_metrics.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(fold_rows[0])); w.writeheader(); w.writerows(fold_rows)
    names = sorted({r['arm'] for r in fold_rows})
    summary = {a: {k: float(np.mean([r[k] for r in fold_rows if r['arm'] == a])) for k in ('score', 'spearman', 'centered_mse')} for a in names}
    folds_ = {a: [r['score'] for r in fold_rows if r['arm'] == a] for a in names}
    configs = {a: [f'{k}|alpha{al}' for k in v for al in ALPHAS] for a, v in arms.items()}
    res = dict(scenario=scenario, stage=stage, reproduction=repro, summary=summary, fold_scores=folds_,
               selection_counts=sel_record, configs=configs, seconds=time.perf_counter() - t0)
    json.dump(res, open(f'{out}/summary.json', 'w'), indent=1)
    print(json.dumps(dict(event='done', reproduction=repro, seconds=round(time.perf_counter() - t0, 1))), flush=True)


if __name__ == '__main__':
    main()

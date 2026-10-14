"""Push-line checks for existing CREED candidates (pre-registered in PLAN.md). Analysis only: no model training.

Independent implementation (metric/selection functions follow the local reference implementation
tissue_kernel_20261009/tissue_kernel.py). Reads only saved outer predictions, saved grids/geometry of the
extra-supervision run, the tissue-kernel run outputs, the 421 source table, the 19-sample extra table and the
fold_metrics.csv of the strong-teacher run (background reference for the distillation condition).
Writes only to OUT_DIR = argv[1].
"""
import csv
import json
import os
import sys
import time
import warnings
import numpy as np
from scipy.stats import spearmanr, t as tdist

RUNS = '/public/home/mengxl/creed-research-runs'
EXT = f'{RUNS}/extra-supervision-20261008-0001'
TK = f'{RUNS}/tissue-kernel-20261009-0001/output'
ST = f'{RUNS}/strong-teacher-recoverability-20261008-0001/output'
SOURCE = f'{RUNS}/gdsc-metabolite-full-protein-20260908-0001/output/result/source-inputs.npz'
EXTRA = f'{EXT}/resources/m_gdsc_extra19.npz'
OUT = sys.argv[1]
SEED = 20261009
B = 2000
NF = 5
SCENARIOS = ('new_cell', 'tissue')
INNER = ('inner-0', 'inner-1', 'inner-2')
META = {'fit', 'held', 'panel', 'drug_ids', 'fit_sample_ids', 'sample_ids', 'truth', 'train_sd', 'evaluated'}
TK_NAME = {'repro_m': 'tk_repro_m', 'repro_mP': 'tk_repro_mP', 'tissue_only': 'tissue_only', 'm_tissue': 'm_tissue',
           'm_tissue__protein': 'm_tissue__protein', 'null_drug': 'tk_null_drug', 'null_tissue_drug': 'null_tissue_drug'}
PRIMARY = [
    dict(id='P1', tag='(a)', scenario='new_cell', deploy='M', cand='extra__m_only__protein', base='extra__m_only',
         mask='extra__m_only__mask', struct=('extra__m_only__shuffle', 'extra__m_only__metabolite')),
    dict(id='P2', tag='(b)', scenario='new_cell', deploy='M+tissue label', cand='m_tissue__protein', base='m_tissue',
         mask=None, struct=()),
    dict(id='P3', tag='(c1)', scenario='tissue', deploy='M (virtual FM)', cand='original__virtual_fm__protein',
         base='original__virtual_fm', mask='original__virtual_fm__mask',
         struct=('original__virtual_fm__shuffle', 'original__virtual_fm__metabolite')),
    dict(id='P4', tag='(c2)', scenario='tissue', deploy='M (virtual FM)', cand='extra__virtual_fm__protein',
         base='extra__virtual_fm', mask='extra__virtual_fm__mask',
         struct=('extra__virtual_fm__shuffle', 'extra__virtual_fm__metabolite')),
]
SECONDARY = [
    ('S1', 'new_cell', 'extra__m_only__protein', 'original__m_only__protein'),
    ('S2', 'tissue', 'extra__virtual_fm__protein', 'original__virtual_fm'),
    ('S3a', 'new_cell', 'm_tissue__protein', 'tissue_only'),
    ('S3b', 'new_cell', 'm_tissue__protein', 'null_tissue_drug'),
    ('S3c', 'new_cell', 'extra__m_only__protein', 'tissue_only'),
    ('S4', 'new_cell', 'm_tissue__protein', 'extra__m_only__protein'),
]
T0 = time.perf_counter()


def log(**kw):
    kw['seconds'] = round(time.perf_counter() - T0, 1)
    print(json.dumps(kw, default=float), flush=True)


# ---------------------------------------------------------------- metric (matches the reference implementation)
def metrics(truth, pred, sd, use):
    obs = np.isfinite(truth); n = obs.sum(0)
    e = np.where(obs, ((pred - truth) / sd) ** 2, 0.0)
    mse = e.sum(0) / np.maximum(n, 1)
    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        bias = (np.where(obs, pred, 0).sum(0) / np.maximum(n, 1) - np.nanmean(truth, axis=0)) / sd
        rho = np.full(len(sd), np.nan)
        for j in np.flatnonzero(use):
            v = obs[:, j]
            if np.std(truth[v, j]) > 0 and np.std(pred[v, j]) > 1e-15:
                rho[j] = spearmanr(truth[v, j], pred[v, j]).statistic
        sp = float(np.nanmean(rho[use]))
    return dict(score=float(np.sqrt(mse[use].mean())), spearman=sp,
                centered_mse=float((mse - bias * bias)[use].mean()), mse=mse, e=e, obs=obs)


# ---------------------------------------------------------------- selection rule (reference implementation)
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


# ---------------------------------------------------------------- statistics
def rel(base, cand):
    return (np.mean(base, axis=-1) - np.mean(cand, axis=-1)) / np.mean(base, axis=-1)


def nb_test(base, cand, ratio):
    d = (base - cand) / base * 100
    m = float(d.mean()); var = float(d.var(ddof=1))
    se = float(np.sqrt((1 / NF + ratio) * var))
    tval = m / se if se > 0 else float('inf')
    p = float(2 * tdist.sf(abs(tval), NF - 1))
    q = float(tdist.ppf(0.975, NF - 1))
    return dict(t=tval, p=p, ci_pct=[m - q * se, m + q * se], mean_fold_rel_pct=m)


def compare(fs, cand, base, ratio):
    b, c = np.asarray(fs[base]), np.asarray(fs[cand])
    keep = [[k for k in range(NF) if k != drop] for drop in range(NF)]
    return dict(cand=cand, base=base, base_mean=float(b.mean()), cand_mean=float(c.mean()),
                abs_diff=float(b.mean() - c.mean()), rel_pct=float(rel(b, c) * 100),
                fold_rel_pct=((b - c) / b * 100).tolist(), folds_won=int((c < b).sum()),
                lofo_rel_pct=[float(rel(b[k], c[k]) * 100) for k in keep], nb=nb_test(b, c, ratio))


def boot_fold_scores(e, obs, use, C, W):
    num = C @ e; den = C @ obs
    v = use[None] & (den > 0)
    mse = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    ww = W * v
    return np.sqrt((ww * mse).sum(1) / ww.sum(1))


def draw_weights(rng, n_universe, clusters, two_stage):
    """clusters[f] = list of index arrays (test-cell positions) per tissue cluster, sorted by tissue label."""
    W = np.zeros((B, n_universe)); C = [np.zeros((B, sum(len(x) for x in cl))) for cl in clusters]
    for b in range(B):
        W[b] = np.bincount(rng.integers(0, n_universe, n_universe), minlength=n_universe)
        for f, cl in enumerate(clusters):
            for k in rng.integers(0, len(cl), len(cl)):
                members = cl[k]
                if two_stage:
                    np.add.at(C[f][b], members[rng.integers(0, len(members), len(members))], 1.0)
                else:
                    C[f][b, members] += 1.0
    return W, C


def main():
    os.makedirs(OUT, exist_ok=False)
    S = np.load(SOURCE, allow_pickle=False); X = np.load(EXTRA, allow_pickle=False)
    auc, tissues, src_ids = S['auc'], S['tissues'], S['sample_ids']
    assert (X['drug_ids'] == S['drug_ids']).all()
    repro = dict(score_max_abs_diff=0.0, spearman_max_abs_diff=0.0, centered_mse_max_abs_diff=0.0, drugs_mismatch=0,
                 tk_score_max_abs_diff=0.0, tk_spearman_max_abs_diff=0.0, tk_centered_mse_max_abs_diff=0.0,
                 tk_alignment_max_abs_diff=0.0, tk_null_vs_null_extra_max_abs_diff=0.0,
                 truth_exact=True, train_sd_max_abs_diff=0.0, grid_selection={}, grid_selection_constant_count=True)
    data = {}
    fold_rows = []
    for sc in SCENARIOS:
        saved = {}
        with open(f'{EXT}/output/{sc}/fold_metrics.csv') as fh:
            for r in csv.DictReader(fh):
                saved[(int(r['fold']), r['arm'])] = r
        folds = []
        for f in range(NF):
            P = np.load(f'{EXT}/output/{sc}/fold-{f}/predictions.npz', allow_pickle=False)
            fit, held, panel = P['fit'], P['held'], P['panel']
            truth, sd, use = P['truth'], P['train_sd'], P['evaluated']
            assert (src_ids[held] == P['sample_ids']).all() and (S['drug_ids'][panel] == P['drug_ids']).all()
            st = auc[np.ix_(held, panel)]
            repro['truth_exact'] &= bool(np.array_equal(np.isnan(st), np.isnan(truth)) and
                                         np.array_equal(np.nan_to_num(st), np.nan_to_num(truth)))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                sd_chk = np.maximum(np.nanstd(auc[np.ix_(fit, panel)], axis=0), 1e-8)
            repro['train_sd_max_abs_diff'] = max(repro['train_sd_max_abs_diff'], float(np.max(np.abs(sd_chk - sd))))
            arms = [k for k in P.files if k not in META]
            preds = {a: P[a] for a in arms}
            # null models (descriptive): per-drug training mean, original and +extra.
            rec = json.load(open(f'{EXT}/output/{sc}/fold-{f}/outer/record.json'))
            rows = [int(np.flatnonzero(X['sample_ids'] == i)[0]) for i in rec['extra_sample_ids']]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                fit_y = auc[np.ix_(fit, panel)]
                preds['null_original'] = np.broadcast_to(np.nanmean(fit_y, axis=0), truth.shape)
                preds['null_extra'] = np.broadcast_to(
                    np.nanmean(np.vstack([fit_y, X['auc'][np.ix_(rows, panel)]]), axis=0), truth.shape)
            if sc == 'new_cell':
                T = np.load(f'{TK}/fold-{f}_predictions.npz', allow_pickle=False)
                for k, name in TK_NAME.items():
                    preds[name] = T[k]
                repro['tk_alignment_max_abs_diff'] = max(
                    repro['tk_alignment_max_abs_diff'],
                    float(np.max(np.abs(T['repro_m'] - P['extra__m_only']))),
                    float(np.max(np.abs(T['repro_mP'] - P['extra__m_only__protein']))))
                repro['tk_null_vs_null_extra_max_abs_diff'] = max(
                    repro['tk_null_vs_null_extra_max_abs_diff'],
                    float(np.max(np.abs(T['null_drug'] - preds['null_extra']))))
            M = {}
            for a, v in preds.items():
                m = metrics(truth, v, sd, use); M[a] = m
                row = dict(scenario=sc, fold=f, arm=a, score=m['score'], spearman=m['spearman'],
                           centered_mse=m['centered_mse'], drugs=int(use.sum()), saved_score='')
                if (f, a) in saved:
                    r = saved[(f, a)]
                    repro['score_max_abs_diff'] = max(repro['score_max_abs_diff'], abs(m['score'] - float(r['score'])))
                    if np.isfinite(m['spearman']):
                        repro['spearman_max_abs_diff'] = max(repro['spearman_max_abs_diff'],
                                                             abs(m['spearman'] - float(r['spearman'])))
                    repro['centered_mse_max_abs_diff'] = max(repro['centered_mse_max_abs_diff'],
                                                             abs(m['centered_mse'] - float(r['centered_mse'])))
                    repro['drugs_mismatch'] += int(int(r['drugs']) != int(use.sum()))
                    row['saved_score'] = float(r['score'])
                fold_rows.append(row)
            # held-cell tissue clusters (sorted by label; empty label = own cluster)
            ht = tissues[held]
            labs = np.array([t if t else f'__unknown_{i}' for i, t in enumerate(ht)])
            clusters = [np.flatnonzero(labs == t) for t in np.unique(labs)]
            folds.append(dict(M=M, use=use, drug_ids=P['drug_ids'], n_held=len(held), n_fit=len(fit), arms=arms,
                              clusters=clusters, P=P, held=held, fit=fit, panel=panel))
            log(event='fold_loaded', scenario=sc, fold=f, n_clusters=len(clusters))
        data[sc] = folds
    # tissue-kernel saved fold metrics
    with open(f'{TK}/fold_metrics.csv') as fh:
        for r in csv.DictReader(fh):
            m = data['new_cell'][int(r['fold'])]['M'][TK_NAME[r['arm']]]
            repro['tk_score_max_abs_diff'] = max(repro['tk_score_max_abs_diff'], abs(m['score'] - float(r['score'])))
            if np.isfinite(m['spearman']) and r['spearman'] not in ('', 'nan', 'NaN'):
                repro['tk_spearman_max_abs_diff'] = max(repro['tk_spearman_max_abs_diff'],
                                                        abs(m['spearman'] - float(r['spearman'])))
            repro['tk_centered_mse_max_abs_diff'] = max(repro['tk_centered_mse_max_abs_diff'],
                                                        abs(m['centered_mse'] - float(r['centered_mse'])))
    log(event='score_reproduction', **{k: v for k, v in repro.items() if k != 'grid_selection'})

    # ------------------------------------------------------------ prediction-level reproduction from saved grids
    for sc in SCENARIOS:
        for f in range(NF):
            fd = data[sc][f]; P = fd['P']; arms = fd['arms']
            base = f'{EXT}/output/{sc}/fold-{f}'
            pos_of = {int(g): i for i, g in enumerate(fd['panel'])}
            total, count = {}, {}
            for bank in INNER:
                G = np.load(f'{base}/{bank}/geometry.npz', allow_pickle=False)
                fit, query, panel = G['fit'], G['query'], G['panel']
                truth = auc[np.ix_(query, panel)]
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    sd = np.maximum(np.nanstd(auc[np.ix_(fit, panel)], axis=0), 1e-8)
                src_cols = np.array([j for j, g in enumerate(panel) if int(g) in pos_of])
                dst_cols = np.array([pos_of[int(panel[j])] for j in src_cols])
                gz = np.load(f'{base}/{bank}/grids.npz', allow_pickle=False)
                for a in arms:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        ll, nn = bank_losses(gz[a], truth, sd)
                    if a not in total:
                        total[a] = np.zeros((ll.shape[0], len(fd['panel']))); count[a] = np.zeros_like(total[a], int)
                    total[a][:, dst_cols] += ll[:, src_cols]; count[a][:, dst_cols] += nn[:, src_cols]
            G = np.load(f'{base}/outer/geometry.npz', allow_pickle=False)
            assert (G['fit'] == fd['fit']).all() and (G['query'] == fd['held']).all() and (G['panel'] == fd['panel']).all()
            gz = np.load(f'{base}/outer/grids.npz', allow_pickle=False)
            for a in arms:
                repro['grid_selection_constant_count'] &= bool(np.all(count[a] == count[a][:1]))
                pred = take(gz[a], choose(total[a], count[a], fallback=1))
                d = float(np.max(np.abs(pred - P[a])))
                key = f'{sc}/{a}'
                repro['grid_selection'][key] = max(repro['grid_selection'].get(key, 0.0), d)
            log(event='grid_selection_done', scenario=sc, fold=f,
                max_diff=max(repro['grid_selection'][f'{sc}/{a}'] for a in arms))
    gs = repro['grid_selection']
    repro['grid_selection_max_abs_diff'] = max(gs.values())
    repro['grid_selection_failed_arms'] = {k: v for k, v in gs.items() if v > 1e-8}
    repro['score_check_passed'] = bool(repro['score_max_abs_diff'] <= 1e-8 and repro['tk_score_max_abs_diff'] <= 1e-8
                                       and repro['drugs_mismatch'] == 0)
    repro['alignment_check_passed'] = bool(repro['tk_alignment_max_abs_diff'] <= 1e-8)
    repro['prediction_check_passed'] = bool(repro['grid_selection_max_abs_diff'] <= 1e-8)
    log(event='reproduction', **{k: v for k, v in repro.items() if k != 'grid_selection'})
    summary = dict(reproduction=repro)
    if not (repro['score_check_passed'] and repro['alignment_check_passed'] and repro['truth_exact']
            and repro['train_sd_max_abs_diff'] <= 1e-12):
        json.dump(summary, open(f'{OUT}/summary.json', 'w'), indent=1, default=float)
        log(event='STOP_reproduction_failed')
        return

    # ------------------------------------------------------------ fold scores per arm
    fs, fs_common, arm_summary, ratio, common_info = {}, {}, {}, {}, {}
    for sc in SCENARIOS:
        folds = data[sc]
        names = list(folds[0]['M'])
        fs[sc] = {a: np.array([folds[f]['M'][a]['score'] for f in range(NF)]) for a in names}
        arm_summary[sc] = {a: dict(score=float(fs[sc][a].mean()), fold_scores=fs[sc][a].tolist(),
                                   spearman=float(np.mean([folds[f]['M'][a]['spearman'] for f in range(NF)])),
                                   centered_mse=float(np.mean([folds[f]['M'][a]['centered_mse'] for f in range(NF)])))
                           for a in names}
        ratio[sc] = float(np.mean([folds[f]['n_held'] / folds[f]['n_fit'] for f in range(NF)]))
        common = set.intersection(*[set(folds[f]['drug_ids'][folds[f]['use']]) for f in range(NF)])
        union = sorted(set.union(*[set(folds[f]['drug_ids']) for f in range(NF)]))
        common_info[sc] = dict(n_common=len(common), n_union=len(union))
        fs_common[sc] = {}
        for a in names:
            v = []
            for f in range(NF):
                uc = folds[f]['use'] & np.isin(folds[f]['drug_ids'], sorted(common))
                v.append(float(np.sqrt(folds[f]['M'][a]['mse'][uc].mean())))
            fs_common[sc][a] = np.array(v)
        data[sc + '_union'] = union
    log(event='fold_scores_ready', ratio=ratio, common=common_info)

    # ------------------------------------------------------------ bootstrap (primary: tissue-cluster x drug; sensitivity: two-stage)
    boot = {}
    for mode, seed in (('primary', SEED), ('two_stage', [SEED, 2])):
        for sc in SCENARIOS:
            folds = data[sc]; union = data[sc + '_union']
            rng = np.random.default_rng(seed)
            W, C = draw_weights(rng, len(union), [fd['clusters'] for fd in folds], two_stage=(mode == 'two_stage'))
            uidx = {d: i for i, d in enumerate(union)}
            out = {}
            for a in fs[sc]:
                cols = []
                for f, fd in enumerate(folds):
                    m = fd['M'][a]
                    Wf = W[:, [uidx[d] for d in fd['drug_ids']]]
                    cols.append(boot_fold_scores(m['e'], m['obs'].astype(float), fd['use'], C[f], Wf))
                    if mode == 'primary' and a == next(iter(fs[sc])) and f == 0:
                        one = boot_fold_scores(m['e'], m['obs'].astype(float), fd['use'],
                                               np.ones((1, fd['n_held'])), np.ones((1, len(fd['drug_ids']))))
                        assert abs(one[0] - m['score']) < 1e-12
                out[a] = np.stack(cols, axis=1)
            boot[(mode, sc)] = out
            log(event='bootstrap_done', mode=mode, scenario=sc)

    def boot_rel(sc, cand, base):
        res = {}
        for mode in ('primary', 'two_stage'):
            r = rel(boot[(mode, sc)][base], boot[(mode, sc)][cand]) * 100
            res[mode] = dict(ci95_pct=[float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))],
                             prob_gt0=float(np.mean(r > 0)), median_pct=float(np.median(r)))
            boot_draws[f'{mode}|{sc}|{cand}|vs|{base}'] = r
        return res

    boot_draws = {}

    # ------------------------------------------------------------ comparisons
    comps, rows_cmp = {}, []

    def full_compare(cid, sc, cand, base, kind):
        c = compare(fs[sc], cand, base, ratio[sc])
        c['bootstrap'] = boot_rel(sc, cand, base)
        c['common_panel'] = compare(fs_common[sc], cand, base, ratio[sc])
        c['common_panel']['n_drugs'] = common_info[sc]['n_common']
        c['spearman_delta'] = arm_summary[sc][cand]['spearman'] - arm_summary[sc][base]['spearman']
        c['centered_mse_delta'] = arm_summary[sc][cand]['centered_mse'] - arm_summary[sc][base]['centered_mse']
        c.update(id=cid, scenario=sc, kind=kind)
        comps[cid] = c
        rows_cmp.append(dict(id=cid, kind=kind, scenario=sc, cand=cand, base=base, base_mean=c['base_mean'],
                             cand_mean=c['cand_mean'], rel_pct=c['rel_pct'], folds_won=c['folds_won'],
                             fold_rel_pct=';'.join(f'{x:.4f}' for x in c['fold_rel_pct']),
                             lofo_rel_pct=';'.join(f'{x:.4f}' for x in c['lofo_rel_pct']),
                             nb_p=c['nb']['p'], nb_ci_lo=c['nb']['ci_pct'][0], nb_ci_hi=c['nb']['ci_pct'][1],
                             boot_lo=c['bootstrap']['primary']['ci95_pct'][0],
                             boot_hi=c['bootstrap']['primary']['ci95_pct'][1],
                             boot2_lo=c['bootstrap']['two_stage']['ci95_pct'][0],
                             boot2_hi=c['bootstrap']['two_stage']['ci95_pct'][1],
                             common_rel_pct=c['common_panel']['rel_pct'], common_folds_won=c['common_panel']['folds_won'],
                             spearman_delta=c['spearman_delta'], centered_mse_delta=c['centered_mse_delta']))
        return c

    pass_rows = []
    for p in PRIMARY:
        sc = p['scenario']
        main_c = full_compare(p['id'], sc, p['cand'], p['base'], 'primary')
        mask_c = full_compare(p['id'] + '_mask', sc, p['cand'], p['mask'], 'numeric_vs_mask') if p['mask'] else None
        struct_c = [full_compare(f"{p['id']}_struct{i}", sc, p['cand'], s, 'structure_control')
                    for i, s in enumerate(p['struct'])]
        cond = {}
        cond['R1'] = ('pass' if main_c['rel_pct'] >= 1.0 else 'fail', f"{main_c['rel_pct']:.4f}%")
        cond['R2'] = ('pass' if main_c['folds_won'] >= 4 else 'fail', f"{main_c['folds_won']}/5")
        if mask_c:
            cond['R3'] = ('pass' if mask_c['cand_mean'] < mask_c['base_mean'] else 'fail',
                          f"{mask_c['cand_mean']:.6f} vs mask {mask_c['base_mean']:.6f} ({mask_c['rel_pct']:.4f}%, {mask_c['folds_won']}/5)")
            cond['R4b'] = ('pass' if min(mask_c['lofo_rel_pct']) > 0 else 'fail',
                           'min LOFO ' + f"{min(mask_c['lofo_rel_pct']):.4f}%")
        else:
            cond['R3'] = ('not_evaluable', 'no saved mask arm')
            cond['R4b'] = ('not_evaluable', 'no saved mask arm')
        cond['R4a'] = ('pass' if min(main_c['lofo_rel_pct']) > 0 else 'fail',
                       'min LOFO ' + f"{min(main_c['lofo_rel_pct']):.4f}%")
        lo, hi = main_c['bootstrap']['primary']['ci95_pct']
        cond['R5'] = ('pass' if lo > 0 else 'fail', f'[{lo:.4f}, {hi:.4f}]%')
        if sc == 'tissue':
            cp = main_c['common_panel']
            cond['R6'] = ('pass' if cp['rel_pct'] >= 1.0 and cp['folds_won'] >= 4 else 'fail',
                          f"{cp['n_drugs']} drugs: {cp['rel_pct']:.4f}%, {cp['folds_won']}/5")
        else:
            cond['R6'] = ('not_applicable', 'new_cell: common panel = full 295 panel')
        if struct_c:
            ok = all(s['cand_mean'] < s['base_mean'] for s in struct_c)
            cond['R7'] = ('pass' if ok else 'fail', '; '.join(f"{s['base'].split('__')[-1]} {s['rel_pct']:.4f}% {s['folds_won']}/5" for s in struct_c))
        else:
            cond['R7'] = ('not_evaluable', 'no saved structure-control arm')
        cond['R8'] = ('not_evaluated', 'no matched (same input, same data, same budget) distillation arm')
        evaluable = [k for k in ('R1', 'R2', 'R3', 'R4a', 'R4b', 'R5', 'R6', 'R7') if cond[k][0] in ('pass', 'fail')]
        all_pass = all(cond[k][0] == 'pass' for k in evaluable)
        any_ne = any(cond[k][0] == 'not_evaluable' for k in ('R3', 'R4b', 'R7'))
        if all_pass and not any_ne:
            grade = 'pass_except_distillation'
        elif main_c['rel_pct'] <= 0:
            grade = 'no_gain'
        elif (main_c['folds_won'] >= 4 and cond['R4a'][0] == 'pass' and cond['R5'][0] == 'pass'
              and cond['R3'][0] in ('pass', 'not_evaluable')):
            grade = 'small_but_reliable' + ('' if cond['R3'][0] == 'pass' else '_no_mask_control')
        else:
            grade = 'small_unstable'
        for k, (status, detail) in cond.items():
            pass_rows.append(dict(id=p['id'], tag=p['tag'], scenario=sc, deploy=p['deploy'], cand=p['cand'],
                                  base=p['base'], condition=k, status=status, detail=detail))
        pass_rows.append(dict(id=p['id'], tag=p['tag'], scenario=sc, deploy=p['deploy'], cand=p['cand'], base=p['base'],
                              condition='GRADE', status=grade,
                              detail='push line = pass only if all evaluable R1-R7 pass with none not_evaluable'))
        comps[p['id']]['conditions'] = cond; comps[p['id']]['grade'] = grade
        log(event='primary_done', id=p['id'], grade=grade, rel_pct=main_c['rel_pct'], folds_won=main_c['folds_won'])
    for cid, sc, cand, base in SECONDARY:
        full_compare(cid, sc, cand, base, 'secondary')

    # ------------------------------------------------------------ distillation background (strong-teacher run, fold scores only)
    distill = {}
    for sc, sub in (('new_cell', 'new_cell'), ('tissue', 'tissue_v2')):
        st = {}
        with open(f'{ST}/{sub}/fold_metrics.csv') as fh:
            for r in csv.DictReader(fh):
                st.setdefault(r['arm'], [None] * NF)[int(r['fold'])] = float(r['score'])
        st = {k: np.array(v) for k, v in st.items() if None not in v}
        match = float(np.max(np.abs(st['m_only'] - fs[sc]['original__m_only'])))
        entry = dict(m_only_fold_score_max_abs_diff=match, matched_folds=bool(match <= 1e-8),
                     st_scores={k: v.tolist() for k, v in st.items() if k in
                                ('m_only', 'transfer_protein_delta', 'transfer_mask_delta', 'measured_p')})
        if match <= 1e-8:
            ext = dict(fs[sc]); ext.update({'st__' + k: v for k, v in st.items()})
            pairs = [('original__m_only__protein', 'st__transfer_protein_delta'),
                     ('st__transfer_protein_delta', 'original__m_only'),
                     ('original__m_only__protein', 'original__m_only')]
            if sc == 'tissue':
                pairs.append(('original__virtual_fm__protein', 'st__transfer_protein_delta'))
            entry['comparisons'] = [compare(ext, c, b, ratio[sc]) for c, b in pairs]
        distill[sc] = entry
    log(event='distillation_background', **{sc: dict(match=v['matched_folds']) for sc, v in distill.items()})

    # ------------------------------------------------------------ write outputs
    with open(f'{OUT}/fold_scores.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(fold_rows[0])); w.writeheader(); w.writerows(fold_rows)
    with open(f'{OUT}/comparisons.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_cmp[0])); w.writeheader(); w.writerows(rows_cmp)
    with open(f'{OUT}/pass_fail.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(pass_rows[0])); w.writeheader(); w.writerows(pass_rows)
    np.savez_compressed(f'{OUT}/bootstrap.npz', **{k.replace('|', '__'): v for k, v in boot_draws.items()})
    summary.update(arms=arm_summary, common_panel={sc: dict(**common_info[sc], fold_scores={a: v.tolist() for a, v in fs_common[sc].items()}) for sc in SCENARIOS},
                   n_test_over_n_train=ratio, comparisons=comps, distillation_background=distill,
                   bootstrap=dict(B=B, seed=SEED, two_stage_seed=[SEED, 2],
                                  n_clusters={sc: [len(fd['clusters']) for fd in data[sc]] for sc in SCENARIOS}),
                   seconds=time.perf_counter() - T0)
    json.dump(summary, open(f'{OUT}/summary.json', 'w'), indent=1, default=float)
    log(event='done')


if __name__ == '__main__':
    main()

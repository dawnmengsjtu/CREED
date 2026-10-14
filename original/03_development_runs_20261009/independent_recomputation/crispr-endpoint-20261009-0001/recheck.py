"""Independent read-only recomputation for crispr-endpoint-20261009-0001. Writes nothing."""
import csv, json, glob, os, sys
import numpy as np
from scipy import stats
from scipy.stats import rankdata

ROOT = '/public/home/mengxl/creed-research-runs'
RUN = ROOT + '/crispr-endpoint-20261009-0001'
OUT = RUN + '/output'
EXT = ROOT + '/extra-supervision-20261008-0001/output'
MATRIX = ROOT + '/crosslayer-matrix-20261009-0001/output'
CRISPR = ROOT + '/crispr310-prepare-20260908-0001/output/result'
SOURCE = ROOT + '/gdsc-metabolite-full-protein-20260908-0001/output/result/source-inputs.npz'
BANKS = ('inner-0', 'inner-1', 'inner-2', 'outer')
rep = {}

# ---------------- build F (421 x genes) from raw effects
S = np.load(SOURCE, allow_pickle=False)
sid = S['sample_ids']
E = np.load(CRISPR + '/effects.npy')
csid = np.load(CRISPR + '/sample_ids.npy', allow_pickle=True)
gid = np.load(CRISPR + '/gene_ids.npy', allow_pickle=True).astype(str)
pos = {str(s): i for i, s in enumerate(sid)}
rows = np.array([pos[str(s)] for s in csid])
assert len(set(rows.tolist())) == 310 == E.shape[0]
F = np.full((len(sid), E.shape[1]), np.nan)
F[rows] = E
rep['F_shape'] = list(F.shape)
rep['nan_frac_effects'] = float(np.isnan(E).mean())


def nanstd_pop(a):
    # population nanstd per column (== np.nanstd default ddof=0), zero for empty columns
    with np.errstate(invalid='ignore'):
        s = np.nanstd(a, axis=0)
    return np.where(np.isfinite(s), s, 0.)


# ---------------- (1) per-arm per-fold sRMSE from predictions.npz, truth/sd rebuilt from F
fm_rows = list(csv.DictReader(open(OUT + '/fold_metrics.csv')))
rep['fold_metrics_rows'] = len(fm_rows)
my = {}  # (sc, fold, arm) -> (score, spearman, cmse)
maxdiff = dict(score=0., spearman=0., centered_mse=0., truth=0., train_sd=0.)
nq_per_fold, nf_per_fold = {}, {}
per_fold_check = {}
for sc in ('new_cell', 'tissue'):
    for f in range(5):
        z = np.load(f'{OUT}/{sc}/fold-{f}_predictions.npz', allow_pickle=False)
        G = np.load(f'{EXT}/{sc}/fold-{f}/outer/geometry.npz', allow_pickle=False)
        fit, held, genes = z['fit'], z['held'], z['genes']
        assert np.array_equal(fit, G['fit']) and np.array_equal(held, G['query']), (sc, f)
        truth = F[np.ix_(held, genes)]
        sd = np.maximum(nanstd_pop(F[np.ix_(fit, genes)]), 1e-8)
        maxdiff['truth'] = max(maxdiff['truth'], float(np.nanmax(np.abs(truth - z['truth']))))
        assert np.array_equal(np.isfinite(truth), np.isfinite(z['truth']))
        maxdiff['train_sd'] = max(maxdiff['train_sd'], float(np.max(np.abs(sd - z['train_sd']))))
        obs = np.isfinite(truth)
        nq = obs.sum(0)
        use = nq > 0
        assert use.all(), 'some evaluated target has 0 query obs'
        # evaluated rule check on outer bank
        cnt_all = np.isfinite(F[fit]).sum(0)
        sd_all = nanstd_pop(F[fit])
        ev = np.flatnonzero((cnt_all >= 20) & (sd_all >= 0.2))
        assert np.array_equal(ev, genes), ('evaluated set mismatch', sc, f)
        assert np.array_equal(z['evaluated'], use)
        nq_per_fold[(sc, f)] = int(np.isfinite(F[held]).any(1).sum())
        nf_per_fold[(sc, f)] = int(np.isfinite(F[fit]).any(1).sum())
        arms = [k[6:] for k in z.files if k.startswith('pred__')]
        for a in arms:
            p = z['pred__' + a]
            e = np.where(obs, ((p - truth) / sd) ** 2, 0.)
            mse = e.sum(0) / nq
            score = float(np.sqrt(mse.mean()))
            bias = (np.where(obs, p, 0.).sum(0) / nq - np.where(obs, truth, 0.).sum(0) / nq) / sd
            cmse = float((mse - bias ** 2).mean())
            # spearman via scipy loop on a subsample of columns for speed check? do all columns exactly
            tn = np.where(obs, truth, np.nan)
            pn = np.where(obs, p, np.nan)
            rt = rankdata(tn, axis=0, nan_policy='omit')
            rp = rankdata(pn, axis=0, nan_policy='omit')
            with np.errstate(invalid='ignore'):
                dt = np.where(obs, rt - np.nanmean(rt, 0), 0.)
                dp = np.where(obs, rp - np.nanmean(rp, 0), 0.)
                den = np.sqrt((dt * dt).sum(0) * (dp * dp).sum(0))
                tsd = np.nanstd(tn, 0); psd = np.nanstd(pn, 0)
            ok = (nq >= 2) & (tsd > 0) & (psd > 1e-15) & (den > 0)
            rho = np.where(ok, (dt * dp).sum(0) / np.where(den > 0, den, 1.), np.nan)
            my[(sc, f, a)] = (score, float(np.nanmean(rho)), cmse)
        per_fold_check[f'{sc}/{f}'] = dict(n_targets=int(len(genes)), n_query=int(len(held)), n_fit=int(len(fit)),
                                           n_query_crispr=nq_per_fold[(sc, f)], n_fit_crispr=nf_per_fold[(sc, f)],
                                           min_query_obs=int(nq.min()))
        print('fold done', sc, f, flush=True)

n_cmp = 0
for r in fm_rows:
    k = (r['scenario'], int(r['fold']), r['arm'])
    s, rho, c = my[k]
    maxdiff['score'] = max(maxdiff['score'], abs(s - float(r['score'])))
    maxdiff['spearman'] = max(maxdiff['spearman'], abs(rho - float(r['spearman'])))
    maxdiff['centered_mse'] = max(maxdiff['centered_mse'], abs(c - float(r['centered_mse'])))
    assert int(r['targets']) == per_fold_check[f"{k[0]}/{k[1]}"]['n_targets']
    assert int(r['targets_dropped']) == 0
    assert int(r['n_query_crispr']) == nq_per_fold[(k[0], k[1])] and int(r['n_fit_crispr']) == nf_per_fold[(k[0], k[1])]
    n_cmp += 1
rep['metric_recompute'] = dict(n_rows_compared=n_cmp, max_abs_diff=maxdiff, per_fold=per_fold_check)
assert n_cmp == len(my), (n_cmp, len(my))

# ---------------- (2) comparisons: gain%, wins, NB t/p, CI
def fold_scores(sc, arm):
    if arm.startswith('random_') and arm.endswith('_mean'):
        base = arm[:-5]
        return np.mean([[my[(sc, f, f'{base}_s{s}')][0] for f in range(5)] for s in range(5)], axis=0)
    return np.array([my[(sc, f, arm)][0] for f in range(5)])

ratio = {sc: float(np.mean([nq_per_fold[(sc, f)] / nf_per_fold[(sc, f)] for f in range(5)])) for sc in ('new_cell', 'tissue')}
rep['r_crispr'] = ratio
cmp_rows = list(csv.DictReader(open(OUT + '/comparisons.csv')))
cd = dict(gain=0., wins=0, t=0., p=0., ci=0., ratio_of_means=0., lofo_min=0.)
q975 = stats.t.ppf(.975, 4)
cmp_table = []
for r in cmp_rows:
    sc = r['scenario']
    a, b = fold_scores(sc, r['new']), fold_scores(sc, r['base'])
    d = (b - a) / b * 100
    g = d.mean()
    v = d.var(ddof=1)
    se = np.sqrt((1 / 5 + ratio[sc]) * v)
    t = g / se
    p = 2 * stats.t.sf(abs(t), 4)
    wins = int((a < b).sum())
    rom = (b.mean() - a.mean()) / b.mean() * 100
    cd['gain'] = max(cd['gain'], abs(g - float(r['gain_pct'])))
    cd['wins'] = max(cd['wins'], abs(wins - int(r['wins'])))
    cd['t'] = max(cd['t'], abs(t - float(r['nb_t'])))
    cd['p'] = max(cd['p'], abs(p - float(r['nb_p'])))
    cd['ratio_of_means'] = max(cd['ratio_of_means'], abs(rom - float(r['gain_pct_ratio_of_means'])))
    ci = json.loads(r['nb_ci95_pct'])
    cd['ci'] = max(cd['ci'], abs(g - q975 * se - ci[0]), abs(g + q975 * se - ci[1]))
    if r.get('lofo_min_pct'):
        lofo = min((b[m].mean() - a[m].mean()) / b[m].mean() * 100 for m in (np.arange(5) != k for k in range(5)))
        cd['lofo_min'] = max(cd['lofo_min'], abs(lofo - float(r['lofo_min_pct'])))
    cmp_table.append(dict(sc=sc, comparison=r['comparison'], gain=round(float(g), 4), wins=wins, t=round(float(t), 3), p=round(float(p), 5)))
rep['comparisons_recompute'] = dict(n_rows=len(cmp_rows), max_abs_diff=cd)
rep['key_comparisons'] = [c for c in cmp_table if any(s in c['comparison'] for s in
                          ('room M->', 'virtual M+v', 'M+vRNA vs shuffle', 'M+vP vs shuffle', 'M added', 'M vs null', 'vs random', 'm_panel_P_k32 vs M', 'm_panel_P_k64 vs M', 'm_panel_G_k32 vs M', 'm_panel_G_k64 vs M', 'M vs T'))]

# R(k) and verdict logic
summ = json.load(open(OUT + '/summary.json'))
Rk = {}
Rdiff = 0.
for sc in ('new_cell', 'tissue'):
    m = fold_scores(sc, 'M').mean()
    Rk[sc] = {}
    for tag, teacher in (('P', 'M+Pval'), ('G', 'M+RNA')):
        tm = fold_scores(sc, teacher).mean()
        Rk[sc][tag] = {}
        for k in (8, 16, 32, 64):
            pm = fold_scores(sc, f'm_panel_{tag}_k{k}').mean()
            R = (m - pm) / (m - tm)
            Rk[sc][tag][k] = round(float(R), 4)
            Rdiff = max(Rdiff, abs(R - summ[sc]['panels'][tag][str(k)]['recovery']['R']))
    # criteria
    def g(comp):
        return [c for c in cmp_table if c['sc'] == sc and c['comparison'] == comp][0]
    r1 = all(g(f'room M->{y}')['gain'] > 0 and g(f'room M->{y}')['wins'] >= 4 for y in ('RNA', 'P'))
    r2 = any(g(f'virtual M+v{y} vs M')['wins'] >= 4 and g(f'virtual M+v{y} vs M')['p'] < 0.10 and g(f'M+v{y} vs shuffle')['wins'] >= 4 for y in ('RNA', 'P'))
    r4 = all(Rk[sc][tag][32] >= 0.5 and Rk[sc][tag][64] >= 0.5 for tag in ('P', 'G'))
    r3 = {y: dict(gain=g(f'M added to {y}')['gain'], wins=g(f'M added to {y}')['wins']) for y in ('RNA', 'P')}
    Rk[sc]['criteria'] = dict(R1=r1, R2=r2, R4=r4, n=int(r1) + int(r2) + int(r4), R3=r3,
                              reported=summ[sc]['criteria']['n_passed'], reported_verdict=summ[sc]['criteria']['verdict'])
rep['Rk'] = Rk
rep['Rk_max_abs_diff_vs_summary'] = Rdiff

# ---------------- (3) evaluated rule per bank, fit/query disjointness, generator reproduction
bank_chk = {}
disjoint_viol, inner_in_outer_viol, ev_mismatch, sd03_mismatch, maxidx = 0, 0, 0, 0, 0
gen_diff = dict(lam=0., oof_r2_mean=0., oof_r2_median=0., virtual_width=0.)
n_gen = 0
for sc in ('new_cell', 'tissue'):
    msaved = json.load(open(f'{MATRIX}/{sc}/bank_info.json'))
    for f in range(5):
        bi = json.load(open(f'{OUT}/{sc}/fold-{f}_bank_info.json'))
        Go = np.load(f'{EXT}/{sc}/fold-{f}/outer/geometry.npz', allow_pickle=False)
        ofit, oq = set(Go['fit'].tolist()), set(Go['query'].tolist())
        for b in BANKS:
            G = np.load(f'{EXT}/{sc}/fold-{f}/{b}/geometry.npz', allow_pickle=False)
            fit, q = G['fit'], G['query']
            maxidx = max(maxidx, int(fit.max()), int(q.max()))
            if set(fit.tolist()) & set(q.tolist()):
                disjoint_viol += 1
            if b != 'outer' and not (set(fit.tolist()) <= ofit and set(q.tolist()) <= ofit):
                inner_in_outer_viol += 1
            cnt = np.isfinite(F[fit]).sum(0)
            sd = nanstd_pop(F[fit])
            n02 = int(((cnt >= 20) & (sd >= 0.2)).sum())
            n03 = int(((cnt >= 20) & (sd >= 0.3)).sum())
            if n02 != bi[b]['evaluated_this_bank']:
                ev_mismatch += 1
            if n03 != bi[b]['evaluated_sd03_this_bank']:
                sd03_mismatch += 1
            bank_chk[f'{sc}/f{f}/{b}'] = dict(n_fit=int(len(fit)), n_query=int(len(q)), ev02=n02, ev03=n03,
                                              reported02=bi[b]['evaluated_this_bank'], n_fit_crispr=int(np.isfinite(F[fit]).any(1).sum()),
                                              n_query_crispr=int(np.isfinite(F[q]).any(1).sum()), empty_cols=bi[b]['empty_target_cols'])
            for k, v in bi[b]['generators'].items():
                sv = msaved[f'fold-{f}/{b}']['generators'][k]
                for key in gen_diff:
                    gen_diff[key] = max(gen_diff[key], abs(v[key] - sv[key]))
                n_gen += 1
rep['bank_checks'] = dict(n_banks=len(bank_chk), fit_query_overlap_violations=disjoint_viol,
                          inner_not_subset_of_outer_fit=inner_in_outer_viol, evaluated_sd02_mismatch=ev_mismatch,
                          evaluated_sd03_mismatch=sd03_mismatch, max_index=maxidx,
                          max_empty_target_cols=max(v['empty_cols'] for v in bank_chk.values()),
                          generator_entries_compared=n_gen, generator_max_abs_diff_vs_matrix=gen_diff)
rep['bank_table'] = {k: (v['ev02'], v['ev03'], v['n_fit_crispr'], v['n_query_crispr']) for k, v in bank_chk.items()}

# common set sizes
for sc in ('new_cell', 'tissue'):
    sets = [set(np.load(f'{OUT}/{sc}/fold-{f}_predictions.npz', allow_pickle=False)['gene_ids'].tolist()) for f in range(5)]
    rep[f'common_{sc}'] = len(set.intersection(*sets))

# panel selection sanity: outer panel idx within pool; pool rule recheck for outer fold 0
sel_chk = {}
for sc in ('new_cell', 'tissue'):
    z = np.load(f'{OUT}/{sc}/fold-0_selection.npz', allow_pickle=False)
    G = np.load(f'{EXT}/{sc}/fold-0/outer/geometry.npz', allow_pickle=False)
    fit = G['fit']
    p = S['p'][fit]
    obs = np.isfinite(p)
    frac = obs.mean(0)
    with np.errstate(invalid='ignore'):
        var = np.nanvar(p, axis=0)
    pc = np.flatnonzero((frac >= 0.9) & (var > 1e-12))
    X = np.load(ROOT + '/rna421-20260906-0002/rna_log_cpm.npy')
    gc = np.argsort(-X[fit].var(0), kind='stable')[:3000]
    sel_chk[sc] = dict(protein_pool_match=bool(np.array_equal(pc, z['panelpool_protein'])),
                       gene_pool_match=bool(np.array_equal(gc, z['panelpool_gene'])),
                       protein_pool_n=int(len(pc)),
                       top64_P_is_argsort_of_weights=bool(np.array_equal(z['panelidx__m_panel_P_k64'], z['panelpool_protein'][np.argsort(-z['panelw_protein'], kind='stable')][:64])),
                       top64_G_is_argsort_of_weights=bool(np.array_equal(z['panelidx__m_panel_G_k64'], z['panelpool_gene'][np.argsort(-z['panelw_gene'], kind='stable')][:64])),
                       random_in_pool=bool(all(np.isin(z[f'panelidx__random_{t}_k{k}_s{s}'], z[f'panelpool_{"protein" if t=="P" else "gene"}']).all() for t in 'PG' for k in (16, 32) for s in range(5))))
rep['selection_checks_fold0_outer'] = sel_chk

print(json.dumps(rep, indent=1, default=float))

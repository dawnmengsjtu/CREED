"""Pre-registered analysis for probe 2 (PLAN.md sections 2, 3.5, 3.6, 4.3, 4.4).

usage: python analyze.py OUT_ROOT     (OUT_ROOT/A/{new_cell,tissue}/fold-*_{metrics,repro,bank_info}.json, OUT_ROOT/B/*.csv)
Writes OUT_ROOT/A/fold_metrics.csv, OUT_ROOT/B/riskcov.csv, OUT_ROOT/B/aurc.csv, OUT_ROOT/summary.json, OUT_ROOT/tables.md.
"""
import csv
import glob
import json
import sys
import numpy as np
from scipy.stats import t as tdist

OUT = sys.argv[1]
SCEN = ('new_cell', 'tissue')
KS = (8, 16, 32, 64); SEEDS = (0, 1, 2); PRIMARY_KS = (16, 32)
COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.5)
ARMS_B = ('M', 'M+vRNA', 'M+vP', 'M+RNA', 'M+P')
PROXIES = ('u1_vRNA', 'u1_vP', 'u2', 'u3', 'oracle', 'random')
PRIMARY_PROXY = {'M+vRNA': 'u1_vRNA', 'M+vP': 'u1_vP'}


def nb(new, base, r):
    """Nadeau-Bengio corrected paired t on per-fold relative improvement (base - new)/base."""
    new, base = np.asarray(new, float), np.asarray(base, float)
    d = (base - new) / base * 100
    m = float(d.mean()); v = float(d.var(ddof=1)); se = float(np.sqrt((1 / 5 + r) * v))
    t = m / se if se > 0 else (np.inf if m > 0 else (-np.inf if m < 0 else np.nan))
    p = float(2 * tdist.sf(abs(t), 4)) if np.isfinite(t) else (0. if m != 0 else 1.)
    q = float(tdist.ppf(.975, 4))
    return dict(improvement_pct=float((base.mean() - new.mean()) / base.mean() * 100), per_fold_pct=[round(float(x), 4) for x in d],
                folds_won=int((new < base).sum()), nb_t=float(t) if np.isfinite(t) else None, nb_p=p,
                ci95_pct=[m - q * se, m + q * se], new_mean=float(new.mean()), base_mean=float(base.mean()),
                new_folds=[float(x) for x in new], base_folds=[float(x) for x in base])


# ---------------------------------------------------------------- part A
def load_A(sc):
    rows = []
    for f in range(5):
        rows += json.load(open(f'{OUT}/A/{sc}/fold-{f}_metrics.json'))
    fs = {}
    for r in rows:
        fs.setdefault(r['arm'], {})[r['fold']] = r
    assert all(sorted(v) == list(range(5)) for v in fs.values()), 'missing folds'
    r = float(np.mean([fs['m_only'][f]['n_query'] / fs['m_only'][f]['n_fit'] for f in range(5)]))
    repro = [json.load(open(f'{OUT}/A/{sc}/fold-{f}_repro.json')) for f in range(5)]
    binfo = {}
    for f in range(5):
        binfo.update(json.load(open(f'{OUT}/A/{sc}/fold-{f}_bank_info.json')))
    return rows, fs, r, repro, binfo


def scores(fs, arm):
    if arm.endswith('_mean'):
        base = arm[:-5]
        return np.mean([[fs[f'{base}_s{s}'][f]['score'] for f in range(5)] for s in SEEDS], axis=0)
    return np.array([fs[arm][f]['score'] for f in range(5)])


def part_A():
    res = {}
    for sc in SCEN:
        rows, fs, r, repro, binfo = load_A(sc)
        R = dict(grid_original_m_only=max(x['grid_original_m_only'] for x in repro), grid_extra_m_only=max(x['grid_extra_m_only'] for x in repro),
                 pred_original_m_only=max(x['pred_original_m_only'] for x in repro), pred_extra_m_only=max(x['pred_extra_m_only'] for x in repro),
                 pred_global_static_vs_m_panel_k={k: max(x['pred_global_static_vs_m_panel_k'][str(k)] for x in repro) for k in KS},
                 pred_m_allcand_vs_saved=max(x['pred_m_allcand_vs_saved'] for x in repro),
                 pred_random_global_vs_saved=max(v for x in repro for v in x['pred_random_global_vs_saved'].values()),
                 metric_vs_protein_panel_csv=max(x['metric_vs_protein_panel_csv'] for x in repro),
                 metric_vs_extra_supervision_csv=max(x['metric_vs_extra_supervision_csv'] for x in repro),
                 top64_identical_all_banks=all(v['top64_identical_to_protein_panel'] for v in binfo.values()),
                 random_panels_identical_all_banks=all(v['random_panels_identical_to_protein_panel'] for v in binfo.values()))
        R['passed'] = bool(max(R['grid_original_m_only'], R['grid_extra_m_only'], R['pred_original_m_only'], R['pred_extra_m_only'],
                               max(R['pred_global_static_vs_m_panel_k'].values()), R['pred_m_allcand_vs_saved'], R['pred_random_global_vs_saved']) <= 1e-8
                           and R['metric_vs_protein_panel_csv'] <= 1e-12 and R['metric_vs_extra_supervision_csv'] <= 1e-12)
        arms = sorted(fs)
        groups = {}
        pols = ['tissue_adaptive', 'cluster_adaptive'] if sc == 'new_cell' else ['cluster_adaptive']
        for k in KS:
            if sc == 'new_cell':
                groups[f'matched_random_tissue_k{k}_mean'] = 1
            groups[f'matched_random_cluster_k{k}_mean'] = 1; groups[f'random_global_k{k}_mean'] = 1
        summary = {}
        for a in arms + list(groups):
            s = scores(fs, a)
            if a in groups:
                sp = np.mean([[fs[f'{a[:-5]}_s{x}'][f]['spearman'] for f in range(5)] for x in SEEDS], axis=0)
                cm = np.mean([[fs[f'{a[:-5]}_s{x}'][f]['centered_mse'] for f in range(5)] for x in SEEDS], axis=0)
            else:
                sp = np.array([fs[a][f]['spearman'] for f in range(5)]); cm = np.array([fs[a][f]['centered_mse'] for f in range(5)])
            summary[a] = dict(score=float(s.mean()), spearman=float(np.nanmean(sp)), centered_mse=float(cm.mean()), folds=[float(x) for x in s])
        # recovery fractions
        m0 = scores(fs, 'm_only'); tA = scores(fs, 'm_allcand'); tF = scores(fs, 'teacher_fullP_saved')
        rec = {}
        for a in list(summary):
            if a in ('m_only', 'repro_extra_m_only', 'null_drug', 'teacher_fullP_saved') or a.startswith(('matched_random', 'random_global')) and not a.endswith('_mean'):
                continue
            s = scores(fs, a)
            rec[a] = dict(vs_allcand=float((m0.mean() - s.mean()) / (m0.mean() - tA.mean())), vs_fullP=float((m0.mean() - s.mean()) / (m0.mean() - tF.mean())),
                          per_fold_vs_allcand=[round(float(x), 4) for x in (m0 - s) / (m0 - tA)])
        # comparisons
        comp = {}
        for pol in pols:
            mr = 'matched_random_tissue' if pol == 'tissue_adaptive' else 'matched_random_cluster'
            for k in KS:
                comp[f'{pol}_vs_global_static_k{k}'] = nb(scores(fs, f'{pol}_k{k}'), scores(fs, f'global_static_k{k}'), r)
                comp[f'{pol}_vs_matched_random_k{k}'] = nb(scores(fs, f'{pol}_k{k}'), scores(fs, f'{mr}_k{k}_mean'), r)
                comp[f'{mr}_mean_vs_random_global_mean_k{k}'] = nb(scores(fs, f'{mr}_k{k}_mean'), scores(fs, f'random_global_k{k}_mean'), r)
        for k in KS:
            comp[f'global_static_vs_random_global_mean_k{k}'] = nb(scores(fs, f'global_static_k{k}'), scores(fs, f'random_global_k{k}_mean'), r)
            comp[f'global_static_vs_m_only_k{k}'] = nb(scores(fs, f'global_static_k{k}'), scores(fs, 'm_only'), r)
        comp['m_allcand_vs_m_only'] = nb(scores(fs, 'm_allcand'), scores(fs, 'm_only'), r)
        # pass criteria per policy
        verdicts = {}
        for pol in pols:
            c1 = {k: comp[f'{pol}_vs_global_static_k{k}'] for k in PRIMARY_KS}
            c2 = {k: comp[f'{pol}_vs_matched_random_k{k}'] for k in PRIMARY_KS}
            i = all(c1[k]['improvement_pct'] > 0 and c1[k]['folds_won'] >= 4 and c1[k]['nb_p'] < 0.10 for k in PRIMARY_KS)
            ii = all(c2[k]['improvement_pct'] > 0 and c2[k]['folds_won'] >= 3 for k in PRIMARY_KS)
            verdicts[pol] = dict(cond_i_beats_global_static=bool(i), cond_ii_beats_matched_random=bool(ii), passes=bool(i and ii),
                                 detail={f'k{k}': dict(vs_global=dict(imp=c1[k]['improvement_pct'], wins=c1[k]['folds_won'], p=c1[k]['nb_p']),
                                                       vs_matched=dict(imp=c2[k]['improvement_pct'], wins=c2[k]['folds_won'], p=c2[k]['nb_p'])) for k in PRIMARY_KS})
        # selection + context diagnostics
        diag = {}
        for f in range(5):
            o = binfo[f'fold-{f}/outer']
            diag[f'fold-{f}'] = dict(candidates=o['candidate_count'], heads=o['heads_total'], bank_seconds=o['bank_seconds'],
                                     cluster_sizes=o['cluster']['fit_sizes'], cluster_query_sizes=o['cluster']['query_sizes'],
                                     cluster_eligible=o['cluster']['eligible'], cluster_query_to_global=o['cluster']['query_to_global'],
                                     cluster_jaccard_vs_global_k16=o['jaccard_vs_global_k16'].get('cluster'),
                                     cluster_tissue_composition=o['cluster']['tissue_by_cluster'])
            if sc == 'new_cell':
                diag[f'fold-{f}'].update(tissue_eligible=o['tissue']['eligible'], tissue_query_to_global=o['tissue']['query_to_global'],
                                         tissue_jaccard_vs_global_k16=o['jaccard_vs_global_k16'].get('tissue'))
        res[sc] = dict(reproduction=R, nb_ratio=r, summary=summary, recovery=rec, comparisons=comp, verdicts=verdicts, diagnostics=diag)
    passA = any(v['passes'] for sc in SCEN for v in res[sc]['verdicts'].values())
    res['PASS_A'] = bool(passA)
    res['reproduction_passed_all'] = bool(all(res[sc]['reproduction']['passed'] for sc in SCEN))
    return res


# ---------------------------------------------------------------- part B
def part_B():
    res = {}
    for sc in SCEN:
        rc = list(csv.DictReader(open(f'{OUT}/B/riskcov_{sc}.csv'))); au = list(csv.DictReader(open(f'{OUT}/B/aurc_{sc}.csv')))
        gen = json.load(open(f'{OUT}/B/gen_check_{sc}.json'))['generator_check']
        cov = {}; aurc = {}
        for x in rc:
            cov.setdefault((x['arm'], x['proxy']), {}).setdefault(float(x['coverage']), [None] * 5)[int(x['fold'])] = float(x['score'])
        ratio = []
        for x in au:
            aurc.setdefault((x['arm'], x['proxy']), [None] * 5)[int(x['fold'])] = float(x['aurc'])
            if x['arm'] == 'M' and x['proxy'] == 'u2':
                ratio.append(int(x['n_query']) / int(x['n_fit']))
        r = float(np.mean(ratio))

        def A(arm, proxy):
            if proxy == 'random':
                return np.mean([aurc[(arm, f'random_s{s}')] for s in range(5)], axis=0)
            return np.array(aurc[(arm, proxy)])

        def C(arm, proxy, c):
            if proxy == 'random':
                return np.mean([cov[(arm, f'random_s{s}')][c] for s in range(5)], axis=0)
            return np.array(cov[(arm, proxy)][c])
        table = {}; comp = {}; verdict = {}
        for arm in ARMS_B:
            for proxy in PROXIES:
                table[f'{arm}|{proxy}'] = dict(aurc=float(A(arm, proxy).mean()), aurc_folds=[float(x) for x in A(arm, proxy)],
                                               coverage={str(c): float(C(arm, proxy, c).mean()) for c in COVERAGES},
                                               coverage_folds={str(c): [float(x) for x in C(arm, proxy, c)] for c in COVERAGES})
                if proxy != 'random':
                    cmp = nb(A(arm, proxy), A(arm, 'random'), r)
                    full, c8 = C(arm, proxy, 1.0), C(arm, proxy, 0.8)
                    rel = float((full.mean() - c8.mean()) / full.mean() * 100)
                    cmp.update(cov08_rel_reduction_pct=rel, cov08_per_fold_pct=[round(float(x), 4) for x in (full - c8) / full * 100],
                               B1=bool(cmp['improvement_pct'] > 0 and cmp['folds_won'] >= 4 and cmp['nb_p'] < 0.10), B2=bool(rel >= 1.0))
                    if arm in ('M+vRNA', 'M+vP'):
                        gM_full, gM_8 = C('M', proxy, 1.0), C('M', proxy, 0.8); gV_full, gV_8 = full, c8
                        G1, G8 = gM_full - gV_full, gM_8 - gV_8
                        cmp.update(gain_full=float(G1.mean()), gain_cov08=float(G8.mean()), gain_larger_on_retained=bool(G8.mean() > G1.mean()),
                                   gain_larger_folds=int((G8 > G1).sum()))
                    comp[f'{arm}|{proxy}_vs_random'] = cmp
            if arm in PRIMARY_PROXY:
                c = comp[f'{arm}|{PRIMARY_PROXY[arm]}_vs_random']
                verdict[arm] = dict(primary_proxy=PRIMARY_PROXY[arm], B1=c['B1'], B2=c['B2'], passes=bool(c['B1'] and c['B2']),
                                    aurc_improvement_pct=c['improvement_pct'], folds_won=c['folds_won'], nb_p=c['nb_p'], cov08_rel_reduction_pct=c['cov08_rel_reduction_pct'],
                                    gain_larger_on_retained=c['gain_larger_on_retained'])
        # per-cell proxy/error correlations (descriptive)
        pc = list(csv.DictReader(open(f'{OUT}/B/percell_{sc}.csv')))
        from scipy.stats import spearmanr
        corr = {}
        for proxy in ('u1_vRNA', 'u1_vP', 'u2', 'u3'):
            for arm in ARMS_B:
                per = []
                for f in range(5):
                    u = np.array([float(x[proxy]) for x in pc if int(x['fold']) == f]); e = np.array([float(x['e_' + arm]) for x in pc if int(x['fold']) == f])
                    per.append(float(spearmanr(u, e).statistic))
                corr[f'{proxy}|{arm}'] = dict(mean=float(np.mean(per)), per_fold=[round(x, 3) for x in per])
        res[sc] = dict(generator_check=gen, all_lambda_identical=all(v['lam_identical'] for v in gen.values()),
                       max_r2_abs_diff=max(v['r2_abs_diff'] for v in gen.values()), nb_ratio=r, table=table, comparisons=comp, verdicts=verdict,
                       proxy_error_spearman=corr)
    res['PASS_B'] = bool(any(v['passes'] for sc in SCEN for v in res[sc]['verdicts'].values()))
    return res


def write_csvs():
    rows = []
    for sc in SCEN:
        for f in range(5):
            rows += json.load(open(f'{OUT}/A/{sc}/fold-{f}_metrics.json'))
    with open(f'{OUT}/A/fold_metrics.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['scenario', 'arm', 'fold', 'score', 'spearman', 'centered_mse', 'n_fit', 'n_query']); w.writeheader(); w.writerows(rows)
    for name in ('riskcov', 'aurc'):
        rr = []
        for sc in SCEN:
            rr += list(csv.DictReader(open(f'{OUT}/B/{name}_{sc}.csv')))
        with open(f'{OUT}/B/{name}.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)


def tables(resA, resB):
    L = []
    for sc in SCEN:
        S = resA[sc]['summary']; L.append(f'\n### A {sc}: all arms (5-fold mean sRMSE; folds 0-4)\n')
        L.append('| arm | sRMSE | Spearman | cMSE | f0 | f1 | f2 | f3 | f4 | R(allcand) | R(fullP) |'); L.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for a in sorted(S, key=lambda x: S[x]['score']):
            rec = resA[sc]['recovery'].get(a, {})
            L.append(f"| {a} | {S[a]['score']:.6f} | {S[a]['spearman']:.4f} | {S[a]['centered_mse']:.4f} | " + ' | '.join(f'{x:.4f}' for x in S[a]['folds'])
                     + f" | {rec.get('vs_allcand', float('nan')):.3f} | {rec.get('vs_fullP', float('nan')):.3f} |")
        L.append(f'\n### A {sc}: comparisons\n'); L.append('| comparison | new | base | imp% | wins | NB p | CI95 | per-fold % |'); L.append('|---|---:|---:|---:|---|---:|---|---|')
        for name, c in resA[sc]['comparisons'].items():
            L.append(f"| {name} | {c['new_mean']:.6f} | {c['base_mean']:.6f} | {c['improvement_pct']:+.3f} | {c['folds_won']}/5 | {c['nb_p']:.3f} | [{c['ci95_pct'][0]:.2f}, {c['ci95_pct'][1]:.2f}] | {c['per_fold_pct']} |")
    for sc in SCEN:
        T = resB[sc]['table']; L.append(f'\n### B {sc}: risk-coverage (5-fold mean retained-set sRMSE) and AURC\n')
        L.append('| arm | proxy | c=1.0 | c=0.9 | c=0.8 | c=0.7 | c=0.5 | AURC |'); L.append('|---|---|---:|---:|---:|---:|---:|---:|')
        for key, v in T.items():
            arm, proxy = key.split('|')
            L.append(f"| {arm} | {proxy} | " + ' | '.join(f"{v['coverage'][str(c)]:.4f}" for c in COVERAGES) + f" | {v['aurc']:.5f} |")
        L.append(f'\n### B {sc}: proxy vs random (AURC) and coverage-0.8 condition\n')
        L.append('| arm | proxy | AURC imp% | wins | NB p | c0.8 rel.red% | B1 | B2 | gain full | gain c0.8 | larger on retained |'); L.append('|---|---|---:|---|---:|---:|---|---|---:|---:|---|')
        for key, c in resB[sc]['comparisons'].items():
            arm, rest = key.split('|'); proxy = rest[:-10]
            L.append(f"| {arm} | {proxy} | {c['improvement_pct']:+.3f} | {c['folds_won']}/5 | {c['nb_p']:.3f} | {c['cov08_rel_reduction_pct']:+.3f} | {c['B1']} | {c['B2']} | "
                     f"{c.get('gain_full', float('nan')):.4f} | {c.get('gain_cov08', float('nan')):.4f} | {c.get('gain_larger_on_retained', '')} |")
    return '\n'.join(L)


def main():
    write_csvs()
    resA, resB = part_A(), part_B()
    out = dict(PASS_A=resA['PASS_A'], PASS_B=resB['PASS_B'], reproduction_passed_all=resA['reproduction_passed_all'], A=resA, B=resB)
    json.dump(out, open(f'{OUT}/summary.json', 'w'), indent=1)
    open(f'{OUT}/tables.md', 'w').write(tables(resA, resB))
    print(json.dumps(dict(PASS_A=resA['PASS_A'], PASS_B=resB['PASS_B'], repro=resA['reproduction_passed_all'],
                          verdictsA={sc: resA[sc]['verdicts'] for sc in SCEN}, verdictsB={sc: resB[sc]['verdicts'] for sc in SCEN}), indent=1))


if __name__ == '__main__':
    main()

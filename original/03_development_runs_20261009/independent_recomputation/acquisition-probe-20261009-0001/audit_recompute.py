import csv, json, math
import numpy as np
from scipy.stats import t as st

R = '/public/home/mengxl/creed-research-runs/acquisition-probe-20261009-0001/output'
ARMS = ['H_meas|virtual|0', 'H_meas|measured_all|all',
        'H_meas|static|16', 'H_meas|adaptive|16', 'H_meas|variance|16',
        'H_meas|static|32', 'H_meas|adaptive|32', 'H_meas|variance|32',
        'H_virt|virtual|0', 'H_virt|measured_all|all',
        'H_virt|static|16', 'H_virt|adaptive|16', 'H_virt|static|32', 'H_virt|adaptive|32',
        'none|m_only|M', 'none|teacher_saved|full_P', 'none|null_drug|none']
RAND = {h: {k: [f'{h}|random_s{s}|{k}' for s in range(5)] for k in (16, 32)} for h in ('H_meas', 'H_virt')}


def srmse(truth, pred, sd, use):
    # independent loop implementation of the pre-registered metric
    vals = []
    for j in range(truth.shape[1]):
        if not use[j]:
            continue
        o = np.isfinite(truth[:, j])
        e = (pred[o, j] - truth[o, j]) / sd[j]
        vals.append(np.mean(e * e))
    return math.sqrt(float(np.mean(vals)))


def nb(new, base, ratio):
    new, base = np.asarray(new), np.asarray(base)
    d = (base - new) / base * 100
    imp = (base.mean() - new.mean()) / base.mean() * 100
    var = d.var(ddof=1)
    se = math.sqrt((0.2 + ratio) * var)
    tv = d.mean() / se
    p = 2 * st.sf(abs(tv), 4)
    return dict(imp=round(float(imp), 4), wins=int((new < base).sum()), t=round(float(tv), 3), p=round(float(p), 4))


out = {}
for sc in ('new_cell', 'tissue'):
    d = f'{R}/{sc}_main'
    fold = {a: [] for a in ARMS}
    rfold = {h: {k: [] for k in (16, 32)} for h in RAND}
    ratios = []
    keys_seen = None
    for f in range(5):
        P = np.load(f'{d}/fold-{f}_predictions.npz', allow_pickle=False)
        truth, sd, use = P['truth'], P['train_sd'], P['evaluated']
        ratios.append(len(P['held']) / len(P['fit']))
        keys_seen = sorted(k for k in P.files if '|' in k)
        for a in ARMS:
            fold[a].append(srmse(truth, P[a], sd, use))
        for h in RAND:
            for k in (16, 32):
                rfold[h][k].append(float(np.mean([srmse(truth, P[a], sd, use) for a in RAND[h][k]])))
    ratio = float(np.mean(ratios))
    # compare with saved fold_metrics.csv
    saved = {}
    for r in csv.DictReader(open(f'{d}/fold_metrics.csv')):
        saved[(r['head'], r['policy'], r['k'], int(r['fold']))] = float(r['score'])
    maxd = 0.0
    for a in ARMS:
        h, pol, k = a.split('|')
        for f in range(5):
            maxd = max(maxd, abs(fold[a][f] - saved[(h, pol, k, f)]))
    for h in RAND:
        for k in (16, 32):
            for f in range(5):
                maxd = max(maxd, abs(rfold[h][k][f] - saved[(h, 'random_mean', str(k), f)]))
    res = dict(ratio=ratio, max_abs_diff_vs_saved_csv=maxd, n_saved_pred_keys=len(keys_seen),
               headline={a: round(float(np.mean(fold[a])), 6) for a in ARMS},
               headline_random_mean={f'{h}|random_mean|{k}': round(float(np.mean(rfold[h][k])), 6) for h in RAND for k in (16, 32)},
               folds={a: [round(x, 6) for x in fold[a]] for a in ARMS},
               folds_random_mean={f'{h}|{k}': [round(x, 6) for x in rfold[h][k]] for h in RAND for k in (16, 32)})
    comps = {}
    for h in ('H_meas', 'H_virt'):
        for k in (16, 32):
            comps[f'{h}|k{k}|adaptive_vs_static'] = nb(fold[f'{h}|adaptive|{k}'], fold[f'{h}|static|{k}'], ratio)
            comps[f'{h}|k{k}|adaptive_vs_random_mean'] = nb(fold[f'{h}|adaptive|{k}'], rfold[h][k], ratio)
            if h == 'H_meas':
                comps[f'{h}|k{k}|adaptive_vs_variance'] = nb(fold[f'{h}|adaptive|{k}'], fold[f'{h}|variance|{k}'], ratio)
                comps[f'{h}|k{k}|static_vs_random_mean'] = nb(fold[f'{h}|static|{k}'], rfold[h][k], ratio)
    comps['H_virt|k0|virtual_vs_m_only'] = nb(fold['H_virt|virtual|0'], fold['none|m_only|M'], ratio)
    comps['H_meas|kall|teacher_vs_m_only'] = nb(fold['H_meas|measured_all|all'], fold['none|m_only|M'], ratio)
    res['comparisons'] = comps
    # monotone check for adaptive H_meas needs k=8,64 which are not saved as predictions -> use csv
    seq = []
    for k in ('0', '8', '16', '32', '64', 'all'):
        pol = 'virtual' if k == '0' else ('measured_all' if k == 'all' else 'adaptive')
        seq.append(float(np.mean([saved[('H_meas', pol, k, f)] for f in range(5)])))
    res['H_meas_adaptive_curve_from_csv'] = [round(x, 6) for x in seq]
    res['H_meas_adaptive_monotone'] = bool(np.all(np.diff(seq) <= 1e-12))
    out[sc] = res
print(json.dumps(out, indent=1))

"""PROSPECTIVE_LOWDIM_V1 fold worker (pre-registered in PLAN.md).

Imports the frozen code unchanged from code/frozen/ (checksums in FROZEN_SHA256SUMS and PLAN.md section 4):
  stageA.py / stageA_crispr.py (N6: EGOP, function-weighted directions W_r, single-width RBF, measured_z), which load
  crosslayer.py / crispr_endpoint.py (layer encodings, generator, masked KRR head, inner-bank selection, metrics);
  protein_panel.py / rna_panel.py (supervised-sum panel rule and random-panel recipe of the drug-response panels).
Only three things are new: the GDSC1 label matrix (prepare_gdsc1.py), controls of the MEASURED layer projected on r = 4
directions (PCA, random, label-permuted), and panel arms chosen from the function-weighted directions.

usage: python lowdim.py ENDPOINT SCENARIO STAGE OUT_DIR [FOLD]
  ENDPOINT = drug (GDSC2 AUC) | crispr (CRISPR gene effect) | gdsc1 (GDSC1 AUC)
  STAGE = dryrun : fold-0 inner-0 bank only; reproduction / engineering checks; no outer prediction, no outer score
          main   : one outer fold, all four banks; outer PREDICTIONS + reproduction gate; writes no score
          score  : scores a finished main fold (OUT_DIR = that fold's directory); run.sh calls it only after every gate passed
"""
import importlib.util
import json
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = os.path.join(HERE, 'frozen')


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(FROZEN, name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SA = _load('stageA')            # loads frozen/crosslayer.py as SA.CL
SAC = _load('stageA_crispr')    # loads frozen/crispr_endpoint.py as SAC.CE
PP = _load('protein_panel')
RP = _load('rna_panel')
CL, CE = SA.CL, SAC.CE

ROOT = CL.ROOT
RUN = os.environ.get('LOWDIM_RUN', ROOT + '/prospective-lowdim-20261022-0001')
XL_OUT = ROOT + '/crosslayer-matrix-20261009-0001/output'
N6_RUN = ROOT + '/settle-stageA-20261011-0001'
CR_OUT = ROOT + '/crispr-endpoint-20261009-0001/output'
PP_OUT = ROOT + '/protein-panel-20261009-0001/output'
RP_OUT = ROOT + '/rna-panel-20261009-0001/output'
GDSC1_DIR = RUN + '/data_gdsc1'

ENDPOINTS = ('drug', 'crispr', 'gdsc1')
YS = ('RNA', 'P')
RS = (4, 8, 16, 32, 64)          # same rank grid as N6
R0 = 4                           # the pre-registered rank for controls and panel leverage
NSEED = 5
KS = (8, 16, 32, 64)
TAG_LAYER = {'P': 'P', 'G': 'RNA'}   # panel tag -> measured layer whose directions guide it
LAYER_ID = {'P': 0, 'G': 1}
SCEN_ID = {'new_cell': 0, 'tissue': 1}
BANK_ID = {'inner-0': 0, 'inner-1': 1, 'inner-2': 2, 'outer': 3}
SEED_RND512 = 20261022
GATE_PRED, GATE_SCORE, GATE_INNER, GATE_GEN = 1e-8, 1e-10, 1e-10, 1e-9
PERM_TAG = {'drug': 'labperm', 'crispr': 'labperm_crispr', 'gdsc1': 'labperm_gdsc1'}


def log(**kw):
    print(json.dumps(kw, default=float), flush=True)


# ---------------------------------------------------------------- data
def load_ctx(ep):
    S, X, _ = CL.load_ctx()                       # sha-checked source-inputs + RNA (extra19 unused)
    ctx = dict(S=S, X=X)
    if ep == 'crispr':
        S2, _, F, gid, _ = CE.load_ctx()
        assert np.array_equal(S2['sample_ids'], S['sample_ids']) and np.array_equal(S2['p'], S['p'], equal_nan=True)
        ctx.update(F=F, gid=gid)
    if ep == 'gdsc1':
        gate = json.load(open(f'{GDSC1_DIR}/data_gate.json'))
        assert gate['passed'] is True, 'GDSC1 data gate did not pass'
        for name in ('auc.npy', 'drug_ids.npy', 'sample_ids.npy', 'panels.npz', 'name_in_gdsc2.npy'):
            assert CL.sha(f'{GDSC1_DIR}/{name}') == gate['files'][name], name
        assert np.array_equal(np.load(f'{GDSC1_DIR}/sample_ids.npy'), S['sample_ids'].astype(str))
        ctx.update(A1=np.load(f'{GDSC1_DIR}/auc.npy'), A1_panels=np.load(f'{GDSC1_DIR}/panels.npz'),
                   A1_drugs=np.load(f'{GDSC1_DIR}/drug_ids.npy'), A1_disjoint=~np.load(f'{GDSC1_DIR}/name_in_gdsc2.npy'))
    return ctx


def geometry(scen, f, bank):
    return np.load(f'{CL.EXT}/{scen}/fold-{f}/{bank}/geometry.npz', allow_pickle=False)


def fold_spec(ctx, ep, scen, f):
    """Label matrix and target columns of one outer fold (the same columns in all four banks)."""
    Go = geometry(scen, f, 'outer')
    if ep == 'drug':
        return dict(Y=ctx['S']['auc'], cols=Go['panel'])
    if ep == 'crispr':
        return dict(Y=ctx['F'], cols=np.flatnonzero(CE.evaluated_mask(ctx['F'][Go['fit']])[0]))
    A = ctx['A1']
    inner = [geometry(scen, f, b)['fit'] for b in CL.INNER]
    yf = A[Go['fit']]; cnt = np.isfinite(yf).sum(0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)              # all-NaN columns -> NaN SD -> excluded
        sd = np.nanstd(yf, axis=0)
    keep = (cnt >= 80) & (sd > 1e-8)
    for fi in inner:
        keep &= np.isfinite(A[fi]).sum(0) >= 20
    cols = np.flatnonzero(keep)
    assert np.array_equal(cols, ctx['A1_panels'][f'{scen}__fold-{f}__panel']), 'GDSC1 panel differs from the data gate'
    return dict(Y=A, cols=cols)


def label_sd(ep, y):
    if ep == 'crispr':
        return np.maximum(CE.nanstd0(y), 1e-8)
    return np.maximum(np.nanstd(y, axis=0), 1e-8)


def z_pools(S, X, fit):
    """Column indices (into X / S['p']) of the 512 measured target columns Z (crosslayer.layer_inputs rule)."""
    g = np.argsort(-X[fit].var(0), kind='stable')[:CL.TOP_GENES]
    p = S['p']; idx = np.flatnonzero(np.isfinite(p[fit]).mean(0) >= CL.P_COVER)
    pf = p[np.ix_(fit, idx)]; mean = np.nanmean(pf, axis=0); vf = np.where(np.isfinite(pf), pf, mean)
    tsel = np.sort(np.argsort(-vf.var(0), kind='stable')[:CL.TOP_TARGET])
    return {'RNA': g[:CL.TOP_TARGET], 'P': idx[tsel]}


def check_pools(S, X, fit, pools, Zm):
    """The pools index exactly the columns of the measured targets Z (re-standardized column by column)."""
    def zs(v):
        sd = v.std(0)
        return (v - v.mean(0)) / np.where(sd < 1e-8, 1., sd)
    xr = X[np.ix_(fit, pools['RNA'])]
    pp = S['p'][np.ix_(fit, pools['P'])]
    vp = np.where(np.isfinite(pp), pp, np.nanmean(pp, axis=0))
    for yl, v in (('RNA', xr), ('P', vp)):
        assert len(pools[yl]) == Zm[yl][0].shape[1]
        np.testing.assert_allclose(zs(v), Zm[yl][0], rtol=0, atol=1e-9)


def leverage(E, r):
    """Function-weighted leverage of each measured column on the top-r directions: ||row j of W_r||^2 = sum_i ev_i U_ji^2."""
    return (E['U'][:, :r] ** 2) @ E['ev'][:r]


def cos2(E1, E2, r):
    return float(np.sum((E1['U'][:, :r].T @ E2['U'][:, :r]) ** 2) / r)


# ---------------------------------------------------------------- one bank
def run_bank(ctx, spec, ep, scen, f, bank, emit, xl_gen):
    S, X = ctx['S'], ctx['X']
    G = geometry(scen, f, bank)
    fit, query = G['fit'], G['query']; n = len(fit)
    if ep in ('drug', 'gdsc1'):
        assert np.array_equal(G['panel'], geometry(scen, f, 'outer')['panel'])
    Y, cols = spec['Y'], spec['cols']
    y = Y[np.ix_(fit, cols)]
    sd = label_sd(ep, y)
    if ep == 'crispr':                     # the CRISPR N6 step used crispr_endpoint.py / stageA_crispr.py
        def predict(K, H):
            return CE.predict_grid(K, H, y)[0]
        C, A = CE, SAC
    else:                                  # drug-response N6 used crosslayer.py / stageA.py
        def predict(K, H):
            return CL.predict_grid(K, H, y)
        C, A = CL, SA
    L, _ = C.layer_inputs(S, X, fit, query)
    Zm = A.measured_z(S, X, fit, query, L)
    egop = A.egop
    np.testing.assert_allclose(L['M'][0], G['m_only_kernel'][:n, :n], rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(L['M'][1], G['m_only_cross'][:, :n], rtol=1e-11, atol=1e-12)
    pools = z_pools(S, X, fit)
    check_pools(S, X, fit, pools, Zm)
    KM, HM = L['M'][0], L['M'][1]
    seed_oof = CL.stable_seed('oof', scen, f, bank)
    folds = np.array_split(np.random.default_rng(seed_oof).permutation(n), CL.OOF_FOLDS)
    perm = np.random.default_rng(CL.stable_seed('pair_shuffle', scen, f, bank)).permutation(n)
    info = dict(n_fit=int(n), n_query=int(len(query)), n_cols=int(len(cols)), gen={}, diag={})

    def head(name, sf, sq):
        K2, H2, _ = A.gw_pair(sf, sq)
        emit(name, predict(.5 * KM + .5 * K2, .5 * HM + .5 * H2))

    emit('M', predict(KM, HM))
    if ep == 'gdsc1':
        for name, parts in (('M+RNA', ('M', 'RNA')), ('M+P', ('M', 'P')), ('M+Pval', ('M', 'Pval'))):
            emit(name, predict(*CL.average(L, parts)))
    lev, levx = {}, {}
    if ep in ('crispr', 'gdsc1'):
        yd = S['auc'][np.ix_(fit, G['panel'])]
        sdd = np.maximum(np.nanstd(yd, axis=0), 1e-8)
    for yl in YS:
        Z, Zq = Zm[yl]
        oof, zq, lam, r2m, r2med = C.generator(KM, HM, Z, seed_oof)
        Kv, Hv, st = C.rbf_pair(oof, zq)
        sv = xl_gen[f'M->{yl}']
        info['gen'][yl] = dict(lam=abs(lam - sv['lam']), r2_mean=abs(r2m - sv['oof_r2_mean']),
                               r2_median=abs(r2med - sv['oof_r2_median']), width=abs(st[2] - sv['virtual_width']))
        emit(f'H086_{yl}', predict(.5 * KM + .5 * Kv, .5 * HM + .5 * Hv))
        if ep == 'gdsc1':
            oofs, zqs, lams, r2ms, r2meds = C.generator(KM, HM, Z[perm], seed_oof)
            Ks, Hs, sts = C.rbf_pair(oofs, zqs)
            ss = xl_gen[f'M->{yl}_shuf']
            info['gen'][f'{yl}_shuf'] = dict(lam=abs(lams - ss['lam']), r2_mean=abs(r2ms - ss['oof_r2_mean']),
                                             r2_median=abs(r2meds - ss['oof_r2_median']), width=abs(sts[2] - ss['virtual_width']))
            emit(f'H086_{yl}_shuf', predict(.5 * KM + .5 * Ks, .5 * HM + .5 * Hs))
            Kt, Ht, _ = CL.rbf_pair(Z, Zq)
            emit(f'T512_{yl}', predict(.5 * KM + .5 * Kt, .5 * HM + .5 * Ht))
        E = egop(KM, Z, y, sd, folds)
        Ep = [egop(KM, Z, y, sd, folds, perm=np.random.default_rng(CL.stable_seed(PERM_TAG[ep], scen, f, bank, yl, s)).permutation(n))
              for s in range(NSEED)]
        for r in RS:
            W = A.wr(E, r)
            head(f'A6_{yl}_r{r}', Z @ W, Zq @ W)            # measured layer on the r function-weighted directions
            head(f'A1_{yl}_r{r}', oof @ W, zq @ W)          # virtual layer on the same directions
        _, _, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
        V = Vt[:R0].T; mu = Z.mean(0)
        head(f'A6pca_{yl}_r{R0}', (Z - mu) @ V, (Zq - mu) @ V)
        for s in range(NSEED):
            Q = A.rand_orth(Z.shape[1], R0, CL.stable_seed('randsub', scen, f, bank, yl, R0, s))
            head(f'A6rand_{yl}_r{R0}_s{s}', Z @ Q, Zq @ Q)
            Wp = A.wr(Ep[s], R0)
            head(f'A6perm_{yl}_r{R0}_s{s}', Z @ Wp, Zq @ Wp)
            head(f'A5_{yl}_r{R0}_s{s}', oof @ Wp, zq @ Wp)  # = N6 arm A5 at r = 4 (reproduction arm for drug / CRISPR)
        lev[yl] = leverage(E, R0)
        evn = E['ev'] / max(E['ev'].sum(), 1e-300)
        d = dict(egop_used=int(E['used']), ev_frac_top=[float(evn[:r].sum()) for r in RS])
        if ep == 'crispr':
            Ed = egop(KM, Z, yd, sdd, folds)              # drug-response metric on the CRISPR bank (N6 arm A9b)
            Wd = A.wr(Ed, R0)
            head(f'A9b_{yl}_r{R0}', oof @ Wd, zq @ Wd)
            levx[yl] = leverage(Ed, R0)
            d['cos2_crispr_vs_drug'] = {r: cos2(E, Ed, r) for r in RS}
        if ep == 'gdsc1':
            E2 = egop(KM, Z, yd, sdd, folds)              # GDSC2 metric of the same bank
            d['cos2_gdsc1_vs_gdsc2'] = {r: cos2(E, E2, r) for r in RS}
            half = np.random.default_rng(CL.stable_seed('splithalf_gdsc1', scen, f, bank, yl)).permutation(y.shape[1])
            h1, h2 = np.sort(half[:len(half) // 2]), np.sort(half[len(half) // 2:])
            E1, Eh = egop(KM, Z, y[:, h1], sd[h1], folds), egop(KM, Z, y[:, h2], sd[h2], folds)
            d['splithalf_cos2'] = {r: cos2(E1, Eh, r) for r in RS}
        info['diag'][yl] = d
    run_panels(ctx, ep, scen, f, bank, G, fit, query, L, y, predict, emit, pools, lev, levx, info)
    return info


# ---------------------------------------------------------------- panels
def run_panels(ctx, ep, scen, f, bank, G, fit, query, L, y, predict, emit, pools, lev, levx, info):
    S, X = ctx['S'], ctx['X']; p = S['p']; n = len(fit)
    if ep == 'crispr':
        F = ctx['F']
        ev_idx = np.flatnonzero(CE.evaluated_mask(F[fit])[0])
        Yw = F[np.ix_(fit, ev_idx)]
        cand = {'P': CE.p_candidates(p[fit]), 'G': L['RNA_pool']}
        w = {'P': CE.panel_weights(p[np.ix_(fit, cand['P'])], Yw), 'G': CE.panel_weights(X[np.ix_(fit, cand['G'])], Yw)}
        order = {t: cand[t][np.argsort(-w[t], kind='stable')] for t in 'PG'}

        def sup_k(t, k):
            return order[t][:k]                                     # crispr_endpoint.py keeps weight order

        def rnd_k(t, k, s):
            rng = np.random.default_rng([20261009, SCEN_ID[scen], f, BANK_ID[bank], s, LAYER_ID[t]])
            return rng.choice(cand[t], size=k, replace=False)

        def w512(t, pool):
            return CE.panel_weights((p if t == 'P' else X)[np.ix_(fit, pool)], Yw)

        def kern(t, idx):
            return CE.protein_panel_kernel(p, fit, query, idx) if t == 'P' else CE.gene_panel_kernel(X, fit, query, idx)

        base = {'P': (L['M'][0], L['M'][1]), 'G': (L['M'][0], L['M'][1])}
    else:
        candP = PP.candidates(p[fit])
        gsel = np.argsort(-X[fit].var(0), kind='stable')[:RP.TOP_GENES]
        cand = {'P': candP, 'G': np.sort(gsel)}
        w = {'P': PP.spearman_scores(p[np.ix_(fit, candP)], y), 'G': RP.spearman_scores_dense(X[np.ix_(fit, cand['G'])], y)}
        order = {t: cand[t][np.argsort(-w[t], kind='stable')] for t in 'PG'}
        perms = {(t, s): np.random.default_rng([20261009, SCEN_ID[scen], f, BANK_ID[bank], s]).permutation(cand[t])
                 for t in 'PG' for s in range(NSEED)}

        def sup_k(t, k):
            return np.sort(order[t][:k])

        def rnd_k(t, k, s):
            return np.sort(perms[(t, s)][:k])                       # nested within seed; k = 16, 32 = the saved panels

        def w512(t, pool):
            return PP.spearman_scores(p[np.ix_(fit, pool)], y) if t == 'P' else RP.spearman_scores_dense(X[np.ix_(fit, pool)], y)

        def kern(t, idx):
            if t == 'P':
                return PP.protein_kernel(p, fit, query, idx)
            K, H, _ = RP.rbf_pair(X[np.ix_(fit, idx)], X[np.ix_(query, idx)])
            return K, H

        # protein_panel.py used the saved M kernel block; rna_panel.py recomputed it (equal to <= 1e-11)
        base = {'P': (G['m_only_kernel'][:n, :n].copy(), G['m_only_cross'][:, :n].copy()), 'G': (L['M'][0], L['M'][1])}
    rec = {}
    for t in 'PG':
        yl = TAG_LAYER[t]; pool = pools[yl]; KB, HB = base[t]

        def ph(name, idx):
            Kp, Hp = kern(t, idx)
            emit(name, predict(.5 * KB + .5 * Kp, .5 * HB + .5 * Hp))
            rec[name] = np.asarray(idx).tolist()

        o_dir = pool[np.argsort(-lev[yl], kind='stable')]
        o_s512 = pool[np.argsort(-w512(t, pool), kind='stable')]
        o_dx = pool[np.argsort(-levx[yl], kind='stable')] if yl in levx else None
        for k in KS:
            ph(f'SUP_{t}_k{k}', sup_k(t, k))
            ph(f'DIR_{t}_k{k}', np.sort(o_dir[:k]))
            ph(f'SUP512_{t}_k{k}', np.sort(o_s512[:k]))
            if o_dx is not None:
                ph(f'DIRx_{t}_k{k}', np.sort(o_dx[:k]))
            for s in range(NSEED):
                ph(f'RND_{t}_k{k}_s{s}', rnd_k(t, k, s))
                prm = np.random.default_rng([SEED_RND512, SCEN_ID[scen], f, BANK_ID[bank], s, LAYER_ID[t]]).permutation(pool)
                ph(f'RND512_{t}_k{k}_s{s}', np.sort(prm[:k]))
        info.setdefault('panel_check', {})[t] = dict(sup_top64=order[t][:64].tolist(),
                                                     sup_top8_ids=[str(x) for x in S['protein_ids'][order[t][:8]]] if t == 'P' else [int(x) for x in order[t][:8]],
                                                     candidates=int(len(cand[t])), pool=int(len(pool)),
                                                     dir_in_sup_top64=int(np.isin(o_dir[:64], order[t][:64]).sum()))
    info['panels'] = rec


# ---------------------------------------------------------------- fold driver
def xl_generators(scen):
    return json.load(open(f'{XL_OUT}/{scen}/bank_info.json'))


def outer_truth(ctx, ep, scen, f, spec, Go):
    """truth / train_sd / evaluated of the outer fold, from the saved runs for drug and CRISPR (as N6 did)."""
    if ep == 'drug':
        P = np.load(f'{XL_OUT}/{scen}/fold-{f}_predictions.npz', allow_pickle=False)
        assert (P['panel'] == spec['cols']).all() and (P['held'] == Go['query']).all() and (P['fit'] == Go['fit']).all()
        return P['truth'], P['train_sd'], P['evaluated']
    if ep == 'crispr':
        P = np.load(f'{CR_OUT}/{scen}/fold-{f}_predictions.npz', allow_pickle=False)
        assert (P['genes'] == spec['cols']).all() and (P['held'] == Go['query']).all() and (P['fit'] == Go['fit']).all()
        return P['truth'], P['train_sd'], P['evaluated']
    A = spec['Y']; cols = spec['cols']
    truth = A[np.ix_(Go['query'], cols)]
    sd = np.maximum(np.nanstd(A[np.ix_(Go['fit'], cols)], axis=0), 1e-8)
    use = np.isfinite(truth).sum(0) >= 5
    assert np.array_equal(use, ctx['A1_panels'][f'{scen}__fold-{f}__evaluated'])
    return truth, sd, use


def gate_outer(ctx, ep, scen, f, preds, truth, sd, use):
    """Reproduction gate G2-G4 (PLAN.md section 9): saved outer predictions / scores of earlier runs."""
    pairs, diffs, sdiffs = {}, {}, {}
    if ep == 'drug':
        P = np.load(f'{XL_OUT}/{scen}/fold-{f}_predictions.npz', allow_pickle=False)
        N = np.load(f'{N6_RUN}/output/{scen}/fold-{f}/predictions.npz', allow_pickle=False)
        Ps = np.load(f'{PP_OUT}/{scen}_main/fold-{f}_predictions.npz', allow_pickle=False)
        Rs = np.load(f'{RP_OUT}/{scen}/fold-{f}_predictions.npz', allow_pickle=False)
        for Q in (Ps, Rs):
            assert (Q['panel'] == P['panel']).all() and (Q['held'] == P['held']).all() and (Q['fit'] == P['fit']).all()
        pairs.update({'M': P['pred__M'], 'H086_RNA': P['pred__M+vRNA'], 'H086_P': P['pred__M+vP']})
        for yl in YS:
            pairs[f'A6_{yl}_r4'] = N[f'pred__A6_{yl}_r4']; pairs[f'A1_{yl}_r4'] = N[f'pred__A1_{yl}_r4']
            for s in range(NSEED):
                pairs[f'A5_{yl}_r4_s{s}'] = N[f'pred__A5_{yl}_r4_s{s}']
        for k in KS:
            pairs[f'SUP_P_k{k}'] = Ps[f'm_panel_k{k}']; pairs[f'SUP_G_k{k}'] = Rs[f'm_panel_k{k}']
        for k in (16, 32):
            for s in range(NSEED):
                pairs[f'RND_P_k{k}_s{s}'] = Ps[f'random_k{k}_s{s}']; pairs[f'RND_G_k{k}_s{s}'] = Rs[f'random_k{k}_s{s}']
    elif ep == 'crispr':
        C = np.load(f'{CR_OUT}/{scen}/fold-{f}_predictions.npz', allow_pickle=False)
        pairs.update({'M': C['pred__M'], 'H086_RNA': C['pred__M+vRNA'], 'H086_P': C['pred__M+vP']})
        for t in 'PG':
            for k in KS:
                pairs[f'SUP_{t}_k{k}'] = C[f'pred__m_panel_{t}_k{k}']
            for k in (16, 32):
                for s in range(NSEED):
                    pairs[f'RND_{t}_k{k}_s{s}'] = C[f'pred__random_{t}_k{k}_s{s}']
        R = json.load(open(f'{N6_RUN}/output_crispr/{scen}/fold-{f}/fold_result.json'))
        saved = {r['arm']: r['score'] for r in R['rows']}
        names = [f'{a}_{yl}_r4' for a in ('A6', 'A1', 'A9b') for yl in YS] + [f'A5_{yl}_r4_s{s}' for yl in YS for s in range(NSEED)]
        for a in names:
            sdiffs[a] = abs(CE.metrics(truth, preds[a], sd, use)['score'] - saved[a])
    for a, v in pairs.items():
        assert v.shape == preds[a].shape, a
        diffs[a] = float(np.max(np.abs(preds[a] - v)))
    ok = all(v <= GATE_PRED for v in diffs.values()) and all(v <= GATE_SCORE for v in sdiffs.values())
    return dict(pred_max_abs_diff=diffs, score_abs_diff=sdiffs, passed=bool(ok))


def gen_gate(infos):
    worst = dict(lam=0., r2_mean=0., r2_median=0., width=0.)
    for inf in infos:
        for g in inf['gen'].values():
            for k in worst:
                worst[k] = max(worst[k], g[k])
    worst['passed'] = bool(worst['lam'] == 0. and max(worst['r2_mean'], worst['r2_median'], worst['width']) <= GATE_GEN)
    return worst


def main_fold(ep, scen, f, out):
    t0 = time.perf_counter()
    ctx = load_ctx(ep)
    spec = fold_spec(ctx, ep, scen, f)
    xlg = xl_generators(scen)
    Go = geometry(scen, f, 'outer')
    loss, preds, sels, infos = {}, {}, {}, {}
    for bank in CL.BANKS:
        Gb = geometry(scen, f, bank)
        if bank != 'outer':
            truth_b = spec['Y'][np.ix_(Gb['query'], spec['cols'])]
            sd_b = label_sd(ep, spec['Y'][np.ix_(Gb['fit'], spec['cols'])])

            def emit(name, grid, truth_b=truth_b, sd_b=sd_b, bank=bank):
                assert bank not in loss.setdefault(name, {}), f'duplicate arm {name}'
                loss[name][bank] = CL.bank_losses(grid, truth_b, sd_b)
        else:
            def emit(name, grid):
                assert name not in preds, f'duplicate arm {name}'
                tot = sum(loss[name][b][0] for b in CL.INNER); cnt = sum(loss[name][b][1] for b in CL.INNER)
                sels[name] = CL.choose(tot, cnt, fallback=1)
                preds[name] = CL.take(grid, sels[name])
        infos[bank] = run_bank(ctx, spec, ep, scen, f, bank, emit, xlg[f'fold-{f}/{bank}']['generators'])
        log(event='bank_done', endpoint=ep, scenario=scen, fold=f, bank=bank, arms=len(preds) or len(loss),
            seconds=round(time.perf_counter() - t0, 1))
    assert set(preds) == set(loss), 'arm sets differ between banks'
    truth, sd, use = outer_truth(ctx, ep, scen, f, spec, Go)
    gate = dict(generators=gen_gate(infos.values()))
    if ep in ('drug', 'crispr'):
        gate['outer'] = gate_outer(ctx, ep, scen, f, preds, truth, sd, use)
    gate['passed'] = bool(all(v['passed'] for v in gate.values() if isinstance(v, dict)))
    np.savez_compressed(f'{out}/predictions.npz', truth=truth, train_sd=sd, evaluated=use, cols=spec['cols'], fit=Go['fit'],
                        held=Go['query'], **{'pred__' + a: v for a, v in preds.items()}, **{'sel__' + a: v for a, v in sels.items()})
    json.dump(infos, open(f'{out}/bank_info.json', 'w'), indent=1, default=float)
    res = dict(endpoint=ep, scenario=scen, fold=f, stage='main', gate=gate, arms=sorted(preds), seconds=time.perf_counter() - t0)
    json.dump(res, open(f'{out}/gate.json', 'w'), indent=1, default=float)
    if not gate['passed']:
        json.dump(res, open(f'{out}/GATE_FAILED.json', 'w'), indent=1, default=float)
        log(event='gate_failed', endpoint=ep, scenario=scen, fold=f, gate=gate)
        sys.exit(2)
    log(event='fold_predicted', endpoint=ep, scenario=scen, fold=f, gate_passed=True, seconds=round(time.perf_counter() - t0, 1))


def reference_rows(ep, scen, f):
    """Saved reference predictions scored beside the arms (room denominators and panel teachers)."""
    if ep == 'drug':
        P = np.load(f'{XL_OUT}/{scen}/fold-{f}_predictions.npz', allow_pickle=False)
        Ps = np.load(f'{PP_OUT}/{scen}_main/fold-{f}_predictions.npz', allow_pickle=False)
        return {'CREED_M+RNA': P['pred__M+RNA'], 'CREED_M+P': P['pred__M+P'], 'TEACHER_P': Ps['teacher_saved'], 'TEACHER_G': P['pred__M+RNA']}
    if ep == 'crispr':
        C = np.load(f'{CR_OUT}/{scen}/fold-{f}_predictions.npz', allow_pickle=False)
        return {'CREED_M+RNA': C['pred__M+RNA'], 'CREED_M+P': C['pred__M+P'], 'TEACHER_P': C['pred__M+Pval'], 'TEACHER_G': C['pred__M+RNA']}
    return {}


def score_fold(ep, scen, f, out):
    """Scores one finished fold. run.sh calls this only after every main-stage gate of the run passed."""
    t0 = time.perf_counter()
    g = json.load(open(f'{out}/gate.json'))
    assert g['gate']['passed'] is True and not os.path.exists(f'{out}/GATE_FAILED.json')   # ADDENDUM 1: gate.json keeps the verdict under 'gate'
    assert not os.path.exists(f'{out}/fold_result.json'), 'already scored'
    Z = np.load(f'{out}/predictions.npz', allow_pickle=False)
    truth, sd, use = Z['truth'], Z['train_sd'], Z['evaluated']
    preds = {k[6:]: Z[k] for k in Z.files if k.startswith('pred__')}
    ref = reference_rows(ep, scen, f)
    if ep == 'gdsc1':
        ref = {'CREED_M+RNA': preds['M+RNA'], 'CREED_M+P': preds['M+P'], 'TEACHER_P': preds['M+Pval'], 'TEACHER_G': preds['M+RNA']}
    metrics = CE.metrics if ep == 'crispr' else CL.metrics
    info = json.load(open(f'{out}/bank_info.json'))
    rows, mse, rho = [], {}, {}
    for a, v in list(preds.items()) + list(ref.items()):
        mm = metrics(truth, v, sd, use)
        rows.append(dict(fold=f, arm=a, score=mm['score'], spearman=mm['spearman'], centered_mse=mm['centered_mse']))
        mse[a], rho[a] = mm['mse'], mm['rho']
    np.savez_compressed(f'{out}/per_target.npz', cols=Z['cols'], evaluated=use, n_query_obs=np.isfinite(truth).sum(0),
                        **{'mse__' + a: v for a, v in mse.items()}, **{'rho__' + a: v for a, v in rho.items()})
    res = dict(endpoint=ep, scenario=scen, fold=f, gate=g['gate'], rows=rows, n_fit=info['outer']['n_fit'], n_query=info['outer']['n_query'],
               n_targets_evaluated=int(use.sum()), diag={b: v['diag'] for b, v in info.items()}, seconds=time.perf_counter() - t0)
    json.dump(res, open(f'{out}/fold_result.json', 'w'), indent=1, default=float)
    log(event='fold_scored', endpoint=ep, scenario=scen, fold=f, seconds=round(time.perf_counter() - t0, 1))


def dryrun(ep, scen, out):
    """fold-0 inner-0 only: every arm is built, inner-0 losses only; no outer bank is touched."""
    t0 = time.perf_counter()
    ctx = load_ctx(ep)
    spec = fold_spec(ctx, ep, scen, 0)
    xlg = xl_generators(scen)
    Gb = geometry(scen, 0, 'inner-0')
    truth_b = spec['Y'][np.ix_(Gb['query'], spec['cols'])]
    sd_b = label_sd(ep, spec['Y'][np.ix_(Gb['fit'], spec['cols'])])
    loss = {}

    def emit(name, grid):
        assert name not in loss, f'duplicate arm {name}'
        loss[name] = CL.bank_losses(grid, truth_b, sd_b)

    info = run_bank(ctx, spec, ep, scen, 0, 'inner-0', emit, xlg['fold-0/inner-0']['generators'])
    chk = dict(generators=gen_gate([info]))
    if ep == 'drug':
        saved = np.load(f'{XL_OUT}/{scen}/fold-0_inner_losses.npz', allow_pickle=False)
        d = {mine: float(np.max(np.abs(loss[mine][0] - saved[f'{theirs}__inner-0__loss'])))
             for mine, theirs in (('M', 'M'), ('H086_RNA', 'M+vRNA'), ('H086_P', 'M+vP'))}
        chk['inner_losses_vs_crosslayer'] = dict(max_abs_diff=d, passed=bool(max(d.values()) <= GATE_INNER))
        fd = SA.fd_test(ctx['S'], ctx['X'], scen)
        chk['egop_finite_difference'] = dict(**fd, passed=bool(fd['max_rel_err'] < 1e-5))
        ppi = json.load(open(f'{PP_OUT}/{scen}_main/bank_info.json'))['fold-0/inner-0']['top64']
        sel = np.load(f'{RP_OUT}/{scen}/fold-0_selection.npz', allow_pickle=False)
        rpok = all(np.array_equal(np.sort(np.array(info['panel_check']['G']['sup_top64'][:k])), sel[f'panel__inner-0__G{k}']) for k in KS)
        chk['panel_selection_vs_saved'] = dict(protein_top64_equal=info['panel_check']['P']['sup_top64'] == list(ppi), gene_panels_equal=bool(rpok),
                                               passed=bool(info['panel_check']['P']['sup_top64'] == list(ppi) and rpok))
    if ep == 'crispr':
        fd = SAC.fd_test(ctx['S'], ctx['X'], ctx['F'], scen, spec['cols'])
        chk['egop_grouped_vs_loop'] = dict(**fd, passed=bool(fd['grouped_vs_loop_rel'] < 1e-8))
        bi = json.load(open(f'{CR_OUT}/{scen}/fold-0_bank_info.json'))['inner-0']['panel_top8']
        ok = info['panel_check']['P']['sup_top8_ids'] == bi['protein'] and info['panel_check']['G']['sup_top8_ids'] == bi['gene']
        chk['panel_selection_vs_saved'] = dict(passed=bool(ok))
    passed = bool(all(v['passed'] for v in chk.values()))
    res = dict(endpoint=ep, scenario=scen, stage='dryrun', checks=chk, passed=passed, arms=len(loss),
               bank_info=info, seconds=time.perf_counter() - t0)
    json.dump(res, open(f'{out}/summary.json', 'w'), indent=1, default=float)
    log(event='dryrun_done', endpoint=ep, scenario=scen, passed=passed, arms=len(loss), seconds=round(time.perf_counter() - t0, 1))
    if not passed:
        sys.exit(2)


def main():
    ep, scen, stage, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    assert ep in ENDPOINTS and scen in SCEN_ID and stage in ('dryrun', 'main', 'score')
    if stage == 'score':
        score_fold(ep, scen, int(sys.argv[5]), out)
        return
    os.makedirs(out, exist_ok=False)
    json.dump(dict(protocol='PROSPECTIVE_LOWDIM_V1', endpoint=ep, scenario=scen, stage=stage, script_sha256=CL.sha(__file__),
                   frozen_sha256={n: CL.sha(os.path.join(FROZEN, n)) for n in sorted(os.listdir(FROZEN)) if n.endswith('.py')},
                   rs=RS, r0=R0, ks=KS, nseed=NSEED, threads=os.environ.get('OMP_NUM_THREADS')),
              open(f'{out}/protocol.json', 'w'), indent=1)
    if stage == 'dryrun':
        dryrun(ep, scen, out)
    else:
        main_fold(ep, scen, int(sys.argv[5]), out)


if __name__ == '__main__':
    main()

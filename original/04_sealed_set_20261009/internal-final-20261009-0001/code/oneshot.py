"""One-shot runner for the sealed internal-final confirmation (protocol CREED_INTERNAL_FINAL_ONESHOT_V1, PLAN.md §3–§8).

usage:
  python oneshot.py freeze                      # 421-only: inner banks, hyper-parameters, panels, generators -> frozen_models/
  python oneshot.py gates                       # code-equivalence gates against the development runs (fold-0 banks)
  python oneshot.py dryrun                      # 401 pseudo-train / 20 pseudo-eval lines of the 421 -> dryrun/ (no sealed value)
  python oneshot.py sealed --gdsc GDSC2_CSV [--crispr CRISPRGeneEffect_CSV]   # THE single sealed evaluation (needs the receipt)

The sealed mode refuses to run (exit code 3, before any raw file is opened) unless RELEASE_RECEIPT.json exists and its recorded
sha256 of PLAN.md, of every file in code/ and of every file in frozen_models/ match the files on disk and the recorded
numpy / scipy versions match the environment. It also refuses if output/ already exists (no retry, PLAN §7).
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '8')

import argparse
import datetime
import json
import platform
import sys
import time
import traceback

import numpy as np
import scipy
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import creed_core as cc
import sealed_io as sio

PROTOCOL_ID = 'CREED_INTERNAL_FINAL_ONESHOT_V1'
ROOT = os.environ.get('CREED_ROOT', '/public/home/mengxl/creed-research-runs')
RUN = os.environ.get('CREED_RUN', ROOT + '/internal-final-20261009-0001')
SOURCE = ROOT + '/gdsc-metabolite-full-protein-20260908-0001/output/result/source-inputs.npz'
SOURCE_SHA = '52d71517a8952e83a1603cd831f09449544102d0ca5ff829ff01be9bf79cd637'
RNA421 = ROOT + '/rna421-20260906-0002'
CRISPR310 = ROOT + '/crispr310-prepare-20260908-0001/output/result'
RAWDIR = ROOT + '/_raw_whitelist_20261009'
SPLIT_IDS = ROOT + '/generator-fullpool-20261009-0001/resources/original_identity_split_ids.json'
EXT = ROOT + '/extra-supervision-20261008-0001/output/new_cell'
MATRIX = ROOT + '/crosslayer-matrix-20261009-0001/output/new_cell'
PPANEL = ROOT + '/protein-panel-20261009-0001/output/new_cell_main'
RPANEL = ROOT + '/rna-panel-20261009-0001/output/new_cell'
ENV_EXPECTED = dict(numpy='2.0.2', scipy='1.13.1')
B_BOOT = 2000
MIN_EVAL_OBS, MIN_TRAIN_OBS = 5, 20
CR_MIN_SD = .2
T0 = time.perf_counter()
LOGFH = None


def log(**kw):
    kw['t'] = round(time.perf_counter() - T0, 1)
    line = json.dumps(cc.json_ready(kw), ensure_ascii=False)
    print(line, flush=True)
    if LOGFH is not None:
        LOGFH.write(line + '\n'); LOGFH.flush()


def refuse(msg, code=3):
    log(event='REFUSED', reason=msg)
    print(f'REFUSED: {msg}', file=sys.stderr, flush=True)
    sys.exit(code)


# ---------------------------------------------------------------- 421 development inputs (always allowed)
class Data:
    def __init__(self, m, X, p, tis, ids):
        self.m, self.X, self.p, self.tis, self.ids = m, X, p, tis, ids

    def sub(self, idx):
        return Data(self.m[idx], self.X[idx], self.p[idx], self.tis[idx], self.ids[idx])


def load_train():
    assert cc.sha_file(SOURCE) == SOURCE_SHA, 'source-inputs sha mismatch'
    Z = np.load(SOURCE, allow_pickle=False)
    S = {k: Z[k] for k in ('m', 'auc', 'p', 'tissues', 'sample_ids', 'drug_ids', 'protein_ids', 'metabolite_ids')}
    X = np.load(RNA421 + '/rna_log_cpm.npy', allow_pickle=False)
    R = dict(sample_ids=np.load(RNA421 + '/sample_ids.npy', allow_pickle=False), gene_ids=np.load(RNA421 + '/gene_ids.npy', allow_pickle=False),
             all_gene_ids=np.load(RNA421 + '/all_source_gene_ids.npy', allow_pickle=False),
             common=np.load(RNA421 + '/common_source_gene_indices.npy', allow_pickle=False), rna=X)
    assert (R['sample_ids'] == S['sample_ids']).all() and X.shape == (421, 34985) and np.isfinite(X).all() and np.isfinite(S['m']).all()
    C = dict(effects=np.load(CRISPR310 + '/effects.npy', allow_pickle=False),
             gene_ids=np.load(CRISPR310 + '/gene_ids.npy', allow_pickle=True).astype(str),
             sample_ids=np.load(CRISPR310 + '/sample_ids.npy', allow_pickle=True).astype(str),
             model_ids=np.load(CRISPR310 + '/model_ids.npy', allow_pickle=True).astype(str))
    pos = {s: i for i, s in enumerate(S['sample_ids'].tolist())}
    rows = np.array([pos[s] for s in C['sample_ids'].tolist()])
    assert len(set(rows.tolist())) == len(rows) == C['effects'].shape[0] == 310
    F = np.full((421, C['effects'].shape[1]), np.nan); F[rows] = C['effects']
    split = json.load(open(SPLIT_IDS))
    train, val, fin = (split[k] for k in ('train', 'validation', 'internal_final'))
    assert len(train) == 421 and len(val) == 64 and len(fin) == 123 and sorted(train) == sorted(S['sample_ids'].tolist())
    assert not (set(train) & set(val)) and not (set(train) & set(fin)) and not (set(val) & set(fin))
    D = Data(S['m'], X, S['p'], S['tissues'], S['sample_ids'])
    return S, D, R, C, F, dict(train=train, validation=val, internal_final=fin)


def raw_paths(S, R):
    return dict(metabolomics=RAWDIR + '/CCLE_metabolomics_20190502.csv', rna=RAWDIR + '/CCLE_RNAseq_RSEM_Expected_Counts.csv.gz',
                mmc2=RAWDIR + '/mmc2.xlsx', mmc3=RAWDIR + '/mmc3.xlsx', common=R['common'], all_gene_ids=R['all_gene_ids'],
                metabolite_ids=S['metabolite_ids'], protein_ids=S['protein_ids'])


# ---------------------------------------------------------------- endpoint definitions
def crispr_targets(Ffit):
    cnt = np.isfinite(Ffit).sum(0); sd = cc.nanstd0(Ffit)
    return np.flatnonzero((cnt >= MIN_TRAIN_OBS) & (sd >= CR_MIN_SD))


DRUG_ARMS = list(cc.arm_configs())
CR_ARMS = list(cc.CRISPR_ARMS)


# ---------------------------------------------------------------- one bank (fit arrays -> grids on query arrays)
def bank_compute(Df, Dq, y_drug, Fcr_fit, cr_cols, gen_seed, shuf_seed, rand_seed_list, want_crispr, emit):
    """Fits every layer / generator / panel on Df only, then emits (arm, grid) for the query arrays Dq.
    `emit(endpoint, arm, grid)` is called as soon as each grid exists (PLAN §7 crash semantics)."""
    Fb = cc.fit_layers(Df.m, Df.X, Df.p, Df.tis, gen_seed, shuf_seed)
    PN = cc.fit_panels(Fb, Df.X, Df.p, y_drug, rand_seed_list)
    B = cc.base_kernels(Fb, PN, Df.m, Df.X, Df.p, Df.tis, Dq.m, Dq.X, Dq.p, Dq.tis)
    cache = {}; A = cc.arm_configs(); graph_cache = None
    for arm in DRUG_ARMS:
        if arm == 'null':
            continue
        g, graph_cache = one_grid(arm, A, B, Fb, y_drug, cache, graph_cache)
        emit('drug', arm, g)
    out = dict(F=Fb, PN=PN)
    if want_crispr:
        ev_b = crispr_targets(Fcr_fit)
        PNc = cc.fit_panels(Fb, Df.X, Df.p, Fcr_fit[:, ev_b], rand_seed_list)
        Bc = cc.base_kernels(Fb, PNc, Df.m, Df.X, Df.p, Df.tis, Dq.m, Dq.X, Dq.p, Dq.tis)
        yc = Fcr_fit[:, cr_cols]; cache_c = {}
        for arm in CR_ARMS:
            if arm == 'null':
                continue
            g, _ = one_grid(arm, A, Bc, Fb, yc, cache_c, None)
            emit('crispr', arm, g)
        out['PNc'] = PNc; out['cr_evaluated_bank'] = ev_b
    return out


def one_grid(arm, A, B, Fb, y, cache, graph_cache):
    base, kind = A[arm]
    K, H = B[base]

    def pg(key, K_, H_):
        if key not in cache:
            cache[key] = cc.predict_grid(K_, H_, y)
        return cache[key]

    if kind is None:
        return pg(base, K, H), graph_cache
    if kind == 'w':
        T, Tq = B['T']
        return np.concatenate([pg(base, K, H)] + [pg(f'{base}|w{w}', K + w * T, H + w * Tq) for w in cc.WS[1:]]), graph_cache
    if graph_cache is None:
        graph_cache = cc.graph_pairs(K, H, Fb['L_protein'])
    return np.concatenate([pg(base, K, H)] + [pg(f'{base}|mu{mu}', ke, he) for mu, (ke, he) in zip(cc.MUS[1:], graph_cache)]), graph_cache


# ---------------------------------------------------------------- inner-bank selection on a training set
def inner_selection(D, auc, Fcr, cr_cols, seed_tag, want_crispr, inner_seed=cc.SEED):
    """3 inner banks = cell_folds(tissues, inner_seed, nfolds=3) replayed on the training set. Per-target config selection."""
    inner = cc.cell_folds(D.tis, inner_seed, nfolds=3)
    total = {'drug': {}, 'crispr': {}}; count = {'drug': {}, 'crispr': {}}; bank_info = {}
    for c in range(3):
        fit, q = np.flatnonzero(inner != c), np.flatnonzero(inner == c)
        Df, Dq = D.sub(fit), D.sub(q)
        truth = {'drug': auc[q]}; sd = {'drug': np.maximum(cc.nanstd0(auc[fit]), 1e-8)}
        if want_crispr:
            truth['crispr'] = Fcr[np.ix_(q, cr_cols)]; sd['crispr'] = np.maximum(cc.nanstd0(Fcr[np.ix_(fit, cr_cols)]), 1e-8)

        def emit(ep, arm, grid):
            ll, nn = cc.bank_losses(grid, truth[ep], sd[ep])
            total[ep][arm] = total[ep].get(arm, 0) + ll; count[ep][arm] = count[ep].get(arm, 0) + nn

        tb = time.perf_counter()
        res = bank_compute(Df, Dq, auc[fit], Fcr[fit] if want_crispr else None, cr_cols,
                           cc.stable_seed('oof', seed_tag, 'inner', c), cc.stable_seed('pair_shuffle', seed_tag, 'inner', c),
                           [cc.SEED, 9, c], want_crispr, emit)
        bank_info[f'inner-{c}'] = dict(n_fit=int(len(fit)), n_query=int(len(q)), generators={k: {kk: v[kk] for kk in ('lam', 'oof_r2_mean', 'oof_r2_median', 'n_targets')} | {'virtual_width': v['vstate'][2]} for k, v in res['F']['gen'].items()},
                                       P_candidates=int(len(res['PN']['P_cand'])), panels={k: v.tolist() for k, v in res['PN']['panels'].items() if not k.startswith(('Prand', 'Grand'))},
                                       crispr_evaluated_bank=int(len(res['cr_evaluated_bank'])) if want_crispr else None,
                                       seconds=round(time.perf_counter() - tb, 1))
        log(event='inner_bank_done', bank=c, n_fit=len(fit), n_query=len(q), seconds=round(time.perf_counter() - tb, 1))
    sel = {ep: {a: cc.choose(total[ep][a], count[ep][a], fallback=1) for a in total[ep]} for ep in total}
    return dict(inner_ids=inner, sel=sel, total=total, count=count, bank_info=bank_info)


# ---------------------------------------------------------------- frozen model (de)serialisation
def kz(s):
    return s.replace('->', '_to_').replace('+', 'PLUS').replace('|', '_').replace('/', '_')


def pack_full(F, PN, PNc, sel, cr_cols, y_drug, Fcr, D):
    """Everything that defines the deployed models, as flat numpy arrays."""
    A = {}
    A['sample_ids'] = D.ids; A['tissues'] = D.tis
    for i, nm in enumerate(('mean', 'sd', 'width')):
        for layer in ('M', 'RNA', 'P'):
            A[f'{layer}_state_{nm}'] = np.asarray(F[f'{layer}_state'][i])
    A['RNA_genes'] = F['RNA_genes']; A['P_idx'] = F['P_idx']; A['P_impute_mean'] = F['P_impute_mean']
    A['P_mask_width'] = np.asarray(F['P_mask_width']); A['P_target_sel'] = F['P_target_sel']; A['shuffle_perm'] = F['shuffle_perm']
    A['L_protein'] = F['L_protein']; A['T_tokens'] = F['T_tokens']
    for name, G in F['gen'].items():
        k = 'gen_' + kz(name)
        A[k + '_lam'] = np.asarray(G['lam']); A[k + '_coef'] = G['coef']; A[k + '_mu'] = G['mu']; A[k + '_oof'] = G['oof']
        A[k + '_vstate_mean'] = G['vstate'][0]; A[k + '_vstate_sd'] = G['vstate'][1]; A[k + '_vstate_width'] = np.asarray(G['vstate'][2])
        A[k + '_oof_r2'] = np.array([G['oof_r2_mean'], G['oof_r2_median']])
    for ep, P in (('drug', PN), ('crispr', PNc)):
        if P is None:
            continue
        A[f'{ep}_P_cand'] = P['P_cand']; A[f'{ep}_P_weights'] = P['P_weights']; A[f'{ep}_G_cand'] = P['G_cand']; A[f'{ep}_G_weights'] = P['G_weights']
        for name, idx in P['panels'].items():
            A[f'{ep}_panel_{name}'] = idx
            mean, st = P['states'][name]
            if mean is not None:
                A[f'{ep}_panelstate_{name}_impute'] = mean
            A[f'{ep}_panelstate_{name}_mean'] = st[0]; A[f'{ep}_panelstate_{name}_sd'] = st[1]; A[f'{ep}_panelstate_{name}_width'] = np.asarray(st[2])
    for ep in sel:
        for arm, s in sel[ep].items():
            A[f'{ep}_sel_{kz(arm)}'] = s
    A['drug_train_sd'] = np.maximum(cc.nanstd0(y_drug), 1e-8); A['drug_train_mean'] = cc.nanmean0(y_drug); A['drug_train_obs'] = np.isfinite(y_drug).sum(0)
    if Fcr is not None:
        A['crispr_cols'] = cr_cols; yc = Fcr[:, cr_cols]
        A['crispr_train_sd'] = np.maximum(cc.nanstd0(yc), 1e-8); A['crispr_train_mean'] = cc.nanmean0(yc); A['crispr_train_obs'] = np.isfinite(yc).sum(0)
    return A


def compare_packed(A, Z, atol=1e-10):
    """Max abs difference / equality per key between a freshly built pack and a frozen npz; raises on mismatch."""
    diffs = {}
    for k, v in A.items():
        if k not in Z.files:
            raise RuntimeError(f'frozen file lacks key {k}')
        w = Z[k]
        if v.dtype.kind in 'USib':
            if not np.array_equal(np.asarray(v), w):
                raise RuntimeError(f'frozen mismatch (exact) at {k}')
            diffs[k] = 0.
        else:
            d = float(np.nanmax(np.abs(np.asarray(v, float) - w))) if np.size(v) else 0.
            if not (np.array_equal(np.isnan(np.asarray(v, float)), np.isnan(w)) and d <= atol):
                raise RuntimeError(f'frozen mismatch at {k}: max abs diff {d}')
            diffs[k] = d
    extra = sorted(set(Z.files) - set(A))
    if extra:
        raise RuntimeError(f'frozen file has unexpected keys {extra[:5]}')
    return max(diffs.values())


# ---------------------------------------------------------------- full fit on a training set (freeze or rebuild)
def fit_training_set(D, auc, Fcr, seed_tag, want_crispr, with_inner=True, inner_seed=cc.SEED):
    cr_cols = crispr_targets(Fcr) if want_crispr else np.zeros(0, int)
    sel_pack = inner_selection(D, auc, Fcr, cr_cols, seed_tag, want_crispr, inner_seed) if with_inner else None
    F = cc.fit_layers(D.m, D.X, D.p, D.tis, cc.stable_seed('oof', seed_tag, 'full'), cc.stable_seed('pair_shuffle', seed_tag, 'full'))
    PN = cc.fit_panels(F, D.X, D.p, auc, [cc.SEED, 9])
    PNc = cc.fit_panels(F, D.X, D.p, Fcr[:, cr_cols], [cc.SEED, 9]) if want_crispr else None
    return dict(F=F, PN=PN, PNc=PNc, cr_cols=cr_cols, sel_pack=sel_pack)


def manifest_from(M, D, S, gene_ids, want_crispr, label):
    F, PN, PNc, sp = M['F'], M['PN'], M['PNc'], M['sel_pack']
    A = cc.arm_configs()
    man = dict(protocol=PROTOCOL_ID, label=label, created=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
               environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
               n_train=int(len(D.ids)), grids=dict(alphas=cc.ALPHAS, lambdas=cc.LAMBDAS, mus=cc.MUS, ws=cc.WS, ks=cc.KS, random_ks=cc.RKS, seeds=cc.SEEDS),
               rules=dict(inner_banks='cell_folds(tissues_train, seed, nfolds=3) [run_recoverability.py nested_cell_folds rule, verified to reproduce the extra-supervision new_cell inner banks]; seed=20261009 on the training set',
                          selection='per target: argmin over configs of summed standardized inner-bank MSE (choose, fallback index 1 = alpha 1 / w 0 / mu 0); <20 observed fit labels -> fit mean',
                          generator='crosslayer.py generator(): 3-fold cross-fitted KRR M->Z, lambda by mean OOF MSE; OOF rows define the virtual kernel state; full-fit coefficients for query rows',
                          generator_seeds=dict(full=['sha256("oof|<tag>|full")[:8] little-endian', 'sha256("pair_shuffle|<tag>|full")'], inner=['sha256("oof|<tag>|inner|c")', 'sha256("pair_shuffle|<tag>|inner|c")']),
                          tissue='T = same-tissue indicator; arms with kind w use K + w*T, w in {0,0.25,1} selected jointly with alpha (protein-panel/tissue-kernel rule)',
                          graph='L_protein = 10-NN union graph on the fit protein value kernel (>=50% coverage, fit-mean imputed, fit z-scored, RBF median width), W=K on support, L=D-W, trace=n_fit; K_eff/H_eff per the reference formula; mu selected jointly with alpha',
                          panels='protein: coverage>=0.9 & var>1e-12 candidates, score=sum_targets |Spearman| (pairs>=10), top-k stable, kernel on fit-mean-imputed values; gene: top-3000 variance pool sorted by index, same score; random: rng=[20261009,9,s] (+ [c] inner bank c) permutation of the pool, first k',
                          crispr_targets=f'genes with >=20 observed training lines and SD>={CR_MIN_SD} on the training set'),
               generators={k: dict(lam=v['lam'], oof_r2_mean=v['oof_r2_mean'], oof_r2_median=v['oof_r2_median'], virtual_width=v['vstate'][2], n_targets=v['n_targets'], coef_sha256=cc.sha_array(v['coef'])) for k, v in F['gen'].items()},
               layers=dict(M=dict(width=F['M_state'][2]), RNA=dict(width=F['RNA_state'][2], top_gene_index=int(F['RNA_genes'][0])),
                           P=dict(candidates=int(len(F['P_idx'])), value_width=F['P_state'][2], mask_width=F['P_mask_width'])),
               L_protein_sha256=cc.sha_array(F['L_protein']),
               arms={ep: {a: dict(configs=cc.config_labels(A[a][1]) if a != 'null' else None,
                                  selected_counts=np.bincount(sp['sel'][ep][a], minlength=len(cc.config_labels(A[a][1]))).tolist() if (sp and a != 'null') else None)
                          for a in (DRUG_ARMS if ep == 'drug' else CR_ARMS)} for ep in (('drug', 'crispr') if want_crispr else ('drug',))},
               panels=dict(drug={k: dict(indices=v.tolist(), ids=(S['protein_ids'][v].tolist() if k.startswith('P') else gene_ids[v].tolist()))
                                 for k, v in PN['panels'].items()}),
               panel_candidates=dict(drug=dict(protein=int(len(PN['P_cand'])), gene=int(len(PN['G_cand'])))),
               inner_bank_info=sp['bank_info'] if sp else None, inner_fold_sizes=np.bincount(sp['inner_ids']).tolist() if sp else None)
    if want_crispr:
        man['crispr'] = dict(n_targets=int(len(M['cr_cols'])), panel_candidates=dict(protein=int(len(PNc['P_cand'])), gene=int(len(PNc['G_cand']))),
                             panels={k: dict(indices=v.tolist(), ids=(S['protein_ids'][v].tolist() if k.startswith('P') else gene_ids[v].tolist())) for k, v in PNc['panels'].items()})
    return man


def write_frozen(M, D, S, gene_ids, auc, Fcr, want_crispr, out_dir, label):
    os.makedirs(out_dir, exist_ok=False)
    A = pack_full(M['F'], M['PN'], M['PNc'], M['sel_pack']['sel'], M['cr_cols'], auc, Fcr if want_crispr else None, D)
    np.savez(out_dir + '/model.npz', **A)
    sp = M['sel_pack']
    np.savez(out_dir + '/inner_losses.npz', inner_ids=sp['inner_ids'],
             **{f'{ep}_loss_{kz(a)}': np.asarray(v) for ep in sp['total'] for a, v in sp['total'][ep].items()},
             **{f'{ep}_count_{kz(a)}': np.asarray(v) for ep in sp['count'] for a, v in sp['count'][ep].items()})
    man = manifest_from(M, D, S, gene_ids, want_crispr, label)
    man['files'] = {f: cc.sha_file(f'{out_dir}/{f}') for f in ('model.npz', 'inner_losses.npz')}
    cc.dump_json(out_dir + '/FROZEN_MANIFEST.json', man)
    log(event='frozen_written', dir=out_dir, files=man['files'])
    return A


# ---------------------------------------------------------------- evaluation on an evaluation set
def evaluate_set(M, D, auc, Fcr, Dq, truth_drug, truth_cr, want_crispr, out_dir, label):
    """Deployed predictions for every arm on Dq; each arm's predictions + scores are written as soon as they exist."""
    os.makedirs(out_dir + '/arms', exist_ok=True)
    F, PN, PNc, sel, cr_cols = M['F'], M['PN'], M['PNc'], M['sel_pack']['sel'], M['cr_cols']
    A = cc.arm_configs()
    res = {}
    for ep in (('drug', 'crispr') if want_crispr else ('drug',)):
        if ep == 'drug':
            y, truth, arms, P = auc, truth_drug, DRUG_ARMS, PN
        else:
            y, truth, arms, P = Fcr[:, cr_cols], truth_cr[:, cr_cols], CR_ARMS, PNc
        sd = np.maximum(cc.nanstd0(y), 1e-8)
        use = (np.isfinite(truth).sum(0) >= MIN_EVAL_OBS) & (np.isfinite(y).sum(0) >= MIN_TRAIN_OBS)
        B = cc.base_kernels(F, P, D.m, D.X, D.p, D.tis, Dq.m, Dq.X, Dq.p, Dq.tis)
        cache = {}; graph_cache = None; preds = {}; mets = {}; status = {}
        for arm in arms:
            try:
                if arm == 'null':
                    pred = np.broadcast_to(cc.nanmean0(y), truth.shape).copy()
                else:
                    g, graph_cache = one_grid(arm, A, B, F, y, cache, graph_cache)
                    pred = cc.take(g, sel[ep][arm])
                m = cc.metrics(truth, pred, sd, use)
                preds[arm] = pred; mets[arm] = m; status[arm] = 'computed'
                np.savez_compressed(f'{out_dir}/arms/{ep}__{kz(arm)}.npz', pred=pred, truth=truth, train_sd=sd, evaluated=use, mse=m['mse'], rho=m['rho'],
                                    score=np.asarray(m['score']), spearman=np.asarray(m['spearman']), centered_mse=np.asarray(m['centered_mse']),
                                    selected=sel[ep][arm] if arm != 'null' else np.zeros(0, int), label=np.asarray(label))
                with open(out_dir + '/progress.jsonl', 'a') as fh:
                    fh.write(json.dumps(dict(endpoint=ep, arm=arm, score=m['score'], spearman=m['spearman'], centered_mse=m['centered_mse'],
                                             targets=int(use.sum()), t=round(time.perf_counter() - T0, 1))) + '\n')
                log(event='arm_done', endpoint=ep, arm=arm, score=round(m['score'], 6), targets=int(use.sum()))
            except Exception as exc:          # PLAN §7: computed arms stay final; this arm is reported as not computed
                status[arm] = 'not_computed: ' + repr(exc)
                log(event='arm_failed', endpoint=ep, arm=arm, error=repr(exc), trace=traceback.format_exc()[-800:])
        res[ep] = dict(preds=preds, metrics=mets, status=status, truth=truth, sd=sd, use=use, n_eval=int(len(Dq.ids)))
    return res


# ---------------------------------------------------------------- statistics (PLAN §5–§6)
def draw_weights(rng, strata, use, B):
    n = sum(len(s) for s in strata); U = np.flatnonzero(use)
    Cw = np.zeros((B, n)); W = np.zeros((B, len(use)))
    for b in range(B):
        for s in strata:
            Cw[b, s] += np.bincount(rng.integers(0, len(s), len(s)), minlength=len(s))
        W[b, U] = np.bincount(rng.integers(0, len(U), len(U)), minlength=len(U))
    return Cw, W


def boot_scores(m, use, Cw, W):
    num = Cw @ m['e']; den = Cw @ m['obs'].astype(float)
    mse = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    ww = W * (use[None] & (den > 0))
    return np.sqrt((ww * mse).sum(1) / ww.sum(1))


def rel_gain(base, arm):
    return (base - arm) / base * 100.


def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if not len(x):
        return dict(lo=None, hi=None, prob_gt0=None, median=None)
    return dict(lo=float(np.percentile(x, 2.5)), hi=float(np.percentile(x, 97.5)), prob_gt0=float(np.mean(x > 0)), median=float(np.median(x)))


def sign_test(m_arm, m_base, use):
    a, b = m_arm[use], m_base[use]
    win = int((a < b).sum()); lose = int((a > b).sum()); tie = int((a == b).sum()); n = win + lose
    p = float(binomtest(win, n, .5).pvalue) if n else None
    return dict(wins=win, losses=lose, ties=tie, fraction_better=(win / n if n else None), p_two_sided=p)


def analyze(R, tissues_q, want_crispr, label):
    out = dict(label=label, bootstrap=dict(B=B_BOOT, seed=cc.SEED, cells='stratified by tissue (empty tissue = own stratum)', drugs='by ID among evaluated targets'))
    for ep, E in R.items():
        mets, use = E['metrics'], E['use']
        if 'M' not in mets:
            out[ep] = dict(error='baseline M not computed', status=E['status']); continue
        toks = cc.tokens(tissues_q, 'q'); strata = [np.flatnonzero(toks == t) for t in np.unique(toks)]
        rng = np.random.default_rng(cc.SEED)
        Cw, W = draw_weights(rng, strata, use, B_BOOT)
        bs = {a: boot_scores(m, use, Cw, W) for a, m in mets.items()}
        arms_tab = {}
        for a, m in mets.items():
            row = dict(score=m['score'], spearman=m['spearman'], centered_mse=m['centered_mse'], boot_score=ci(bs[a]))
            if a != 'M':
                g = rel_gain(bs['M'], bs[a])
                row.update(gain_vs_M_pct=rel_gain(mets['M']['score'], m['score']), gain_vs_M_boot=ci(g), sign_test_vs_M=sign_test(m['mse'], mets['M']['mse'], use))
            arms_tab[a] = row
        comp = {}

        def compare(name, arm, base):
            if arm in mets and base in mets:
                comp[name] = dict(arm=arm, base=base, gain_pct=rel_gain(mets[base]['score'], mets[arm]['score']),
                                  boot=ci(rel_gain(bs[base], bs[arm])), sign_test=sign_test(mets[arm]['mse'], mets[base]['mse'], use))
            else:
                comp[name] = dict(arm=arm, base=base, status='not_computed')
            return comp[name]

        for y in ('RNA', 'P'):
            compare(f'M+v{y}_vs_M', f'M+v{y}', 'M'); compare(f'M+v{y}_shuf_vs_M', f'M+v{y}_shuf', 'M')
            compare(f'M+v{y}_vs_shuf', f'M+v{y}', f'M+v{y}_shuf'); compare(f'M+{y}_vs_M', f'M+{y}', 'M')
        compare('M+Pval_vs_M', 'M+Pval', 'M')
        if ep == 'drug':
            compare('M+T_vs_M', 'M+T', 'M'); compare('T_vs_M', 'T', 'M'); compare('M+vRNA+T_vs_M', 'M+vRNA+T', 'M'); compare('M+vRNA+T_vs_M+T', 'M+vRNA+T', 'M+T')
            compare('M+P_graph_vs_M', 'M+P_graph', 'M')
        for k in cc.KS:
            compare(f'M+P{k}_vs_M', f'M+P{k}', 'M'); compare(f'M+G{k}_vs_M', f'M+G{k}', 'M')
        # recovered share and panel recovery R(k)
        share = {}
        for y in ('RNA', 'P'):
            if all(a in mets for a in (f'M+v{y}', f'M+{y}')):
                room = rel_gain(mets['M']['score'], mets[f'M+{y}']['score']); rec = rel_gain(mets['M']['score'], mets[f'M+v{y}']['score'])
                broom = rel_gain(bs['M'], bs[f'M+{y}']); brec = rel_gain(bs['M'], bs[f'M+v{y}'])
                with np.errstate(divide='ignore', invalid='ignore'):
                    bshare = np.where(np.abs(broom) > 1e-12, brec / broom, np.nan)
                share[y] = dict(room_pct=room, recovered_pct=rec, share=(rec / room if abs(room) > 1e-12 else None), share_boot=ci(bshare))
        Rk = {}
        for layer, teacher in (('P', 'M+Pval'), ('G', 'M+RNA')):
            if teacher not in mets:
                continue
            den = mets['M']['score'] - mets[teacher]['score']; bden = bs['M'] - bs[teacher]
            for k in cc.KS:
                a = f'M+{layer}{k}'
                if a in mets:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        br = np.where(np.abs(bden) > 1e-12, (bs['M'] - bs[a]) / bden, np.nan)
                    Rk[a] = dict(teacher=teacher, R=((mets['M']['score'] - mets[a]['score']) / den if abs(den) > 1e-12 else None), R_boot=ci(br),
                                 teacher_gain_pct=rel_gain(mets['M']['score'], mets[teacher]['score']))
            for k in cc.RKS:
                names = [f'M+{layer}rand{k}_s{s}' for s in cc.SEEDS if f'M+{layer}rand{k}_s{s}' in mets]
                if names:
                    sc = float(np.mean([mets[a]['score'] for a in names])); bsc = np.mean([bs[a] for a in names], axis=0)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        br = np.where(np.abs(bden) > 1e-12, (bs['M'] - bsc) / bden, np.nan)
                    Rk[f'M+{layer}rand{k}_mean'] = dict(teacher=teacher, R=((mets['M']['score'] - sc) / den if abs(den) > 1e-12 else None), R_boot=ci(br),
                                                        score_mean_over_seeds=sc, seeds=len(names), gain_vs_M_pct=rel_gain(mets['M']['score'], sc))
        out[ep] = dict(n_eval_cells=E['n_eval'], n_targets_evaluated=int(use.sum()), n_targets_total=int(len(use)), n_strata=len(strata),
                       arms=arms_tab, comparisons=comp, share=share, panel_recovery=Rk, status=E['status'])
    out['F'] = f_criteria(out, want_crispr)
    return out


def f_criteria(out, want_crispr):
    d = out.get('drug', {}); comp = d.get('comparisons', {}); share = d.get('share', {}); Rk = d.get('panel_recovery', {})

    def pos_ci(c):
        return bool(c.get('gain_pct') is not None and c['gain_pct'] > 0 and c['boot']['lo'] is not None and c['boot']['lo'] > 0)

    def pick(name):
        return comp.get(name, {})

    F = {}
    F['F1'] = dict(claim='free virtual RNA layer beats M alone', rule='gain(M+vRNA vs M) > 0 and bootstrap 95% lower bound > 0', evidence=pick('M+vRNA_vs_M'), confirmed=pos_ci(pick('M+vRNA_vs_M')))
    F['F2'] = dict(claim='free virtual P layer beats M alone', rule='gain(M+vP vs M) > 0 and bootstrap 95% lower bound > 0', evidence=pick('M+vP_vs_M'), confirmed=pos_ci(pick('M+vP_vs_M')))
    sh = {y: pick(f'M+v{y}_shuf_vs_M') for y in ('RNA', 'P')}
    F['F3'] = dict(claim='paired-shuffle controls do not beat M', rule='for each of vRNA_shuf, vP_shuf: bootstrap interval of gain vs M contains 0 or gain < 0', evidence=sh,
                   confirmed=all(c.get('boot', {}).get('lo') is not None and (c['boot']['lo'] <= 0 or c['gain_pct'] < 0) for c in sh.values()))
    shares_ok = {y: (share.get(y, {}).get('share') is not None and 0.15 <= share[y]['share'] <= 0.55) for y in ('RNA', 'P')}
    F['F4'] = dict(claim='recovered share consistent with development folds', rule='share = gain(M+vY vs M)/gain(M+Y vs M) in [0.15, 0.55] for Y=RNA and Y=P, and F1 and F2 confirmed',
                   evidence=share, share_in_range=shares_ok, confirmed=bool(all(shares_ok.values()) and F['F1']['confirmed'] and F['F2']['confirmed']))
    rk_ok = {a: (Rk.get(a, {}).get('R') is not None and Rk[a]['R'] >= .5) for a in ('M+P32', 'M+P64', 'M+G32', 'M+G64')}
    F['F5'] = dict(claim='panel law', rule='protein and gene panels: R(32) >= 0.5 and R(64) >= 0.5 (teacher M+Pval / M+RNA)', evidence={a: Rk.get(a) for a in rk_ok}, confirmed=bool(all(rk_ok.values())))
    F['F6'] = dict(claim='tissue label has value', rule='gain(M+T vs M) > 0 and bootstrap lower bound > 0', evidence=pick('M+T_vs_M'), confirmed=pos_ci(pick('M+T_vs_M')))
    c7 = pick('M+P_graph_vs_M')
    F['F7'] = dict(claim='training-time protein graph', rule='gain(M+P graph vs M) > 0 (direction; interval reported)', evidence=c7, confirmed=bool(c7.get('gain_pct') is not None and c7['gain_pct'] > 0))
    n_conf = sum(int(F[k]['confirmed']) for k in ('F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7'))
    F['total'] = f'{n_conf}/7 confirmed'
    if want_crispr and 'crispr' in out and 'comparisons' in out['crispr']:
        c = out['crispr']['comparisons']; R2 = out['crispr']['panel_recovery']
        f1 = pos_ci(c.get('M+vRNA_vs_M', {})); f2 = pos_ci(c.get('M+vP_vs_M', {}))
        rk2 = {a: (R2.get(a, {}).get('R') is not None and R2[a]['R'] >= .5) for a in ('M+P32', 'M+P64', 'M+G32', 'M+G64')}
        F['F8'] = dict(claim='F1, F2, F5 reproduce on the CRISPR endpoint', rule='same rules on CRISPR gene effect', F1=f1, F2=f2, F5=bool(all(rk2.values())), F5_detail=rk2,
                       evidence=dict(vRNA=c.get('M+vRNA_vs_M'), vP=c.get('M+vP_vs_M'), panels={a: R2.get(a) for a in rk2}), confirmed=bool(f1 and f2 and all(rk2.values())),
                       n_targets=out['crispr']['n_targets_evaluated'], n_cells=out['crispr']['n_eval_cells'])
    elif want_crispr:
        F['F8'] = dict(claim='F1, F2, F5 reproduce on the CRISPR endpoint', status='not_evaluated', reason=out.get('crispr', {}).get('error', 'CRISPR endpoint unavailable'))
    else:
        F['F8'] = dict(claim='F1, F2, F5 reproduce on the CRISPR endpoint', status='not_evaluated', reason='no CRISPR file supplied')
    return F


def fmt(x, nd=4):
    return '—' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{nd}f}'


def fmt_ci(c):
    return '—' if not c or c.get('lo') is None else f'[{c["lo"]:.2f}, {c["hi"]:.2f}] P(>0)={c["prob_gt0"]:.3f}'


def result_markdown(an, meta):
    L = []
    lab = an['label']
    L.append(f'# {lab}：封存内部最终集一次性确认——结果草稿\n')
    L.append(f'协议 {PROTOCOL_ID}；生成时间 {datetime.datetime.now().astimezone().isoformat(timespec="seconds")}。')
    if 'DRYRUN' in lab:
        L.append('**本文件是 DRYRUN（只用 421 开发细胞的 401/20 伪划分），不含任何封存值，不是结果。**')
    L.append(f'\n训练细胞 {meta["n_train"]}；评估细胞 {meta["n_eval"]}；评估药物 = 评估集观测 ≥ {MIN_EVAL_OBS} 且训练集观测 ≥ {MIN_TRAIN_OBS}。所有结果一律报告（PLAN §6，无合格线）。\n')
    F = an['F']
    L.append('## 预注册确认项（PLAN §6）\n')
    L.append('| 编号 | 主张 | 判据 | 结果 | 关键数字 |'); L.append('|---|---|---|---|---|')
    for k in ('F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8'):
        f = F.get(k, {})
        if 'confirmed' in f:
            ev = f.get('evidence', {})
            if k in ('F1', 'F2', 'F6', 'F7'):
                num = f'gain {fmt(ev.get("gain_pct"), 3)}%，95% {fmt_ci(ev.get("boot"))}'
            elif k == 'F3':
                num = '; '.join(f'{y}: {fmt(c.get("gain_pct"), 3)}% {fmt_ci(c.get("boot"))}' for y, c in ev.items())
            elif k == 'F4':
                num = '; '.join(f'{y}: room {fmt(v.get("room_pct"), 3)}%, rec {fmt(v.get("recovered_pct"), 3)}%, share {fmt(v.get("share"), 3)} {fmt_ci(v.get("share_boot"))}' for y, v in ev.items())
            elif k == 'F5':
                num = '; '.join(f'{a}: R={fmt((v or {}).get("R"), 3)} {fmt_ci((v or {}).get("R_boot"))}' for a, v in ev.items())
            else:
                num = f'F1={f.get("F1")}, F2={f.get("F2")}, F5={f.get("F5")}（{f.get("n_cells")} 细胞, {f.get("n_targets")} 靶标）'
            L.append(f'| {k} | {f["claim"]} | {f["rule"]} | {"确认" if f["confirmed"] else "未确认"} | {num} |')
        else:
            L.append(f'| {k} | {f.get("claim", "")} | — | 未评估 | {f.get("reason", "")} |')
    L.append(f'\n**总判定：{F["total"]}**（F8 另列）。\n')
    for ep in ('drug', 'crispr'):
        if ep not in an or 'arms' not in an[ep]:
            if ep in an:
                L.append(f'## {ep} 终点：{an[ep].get("error")}\n')
            continue
        d = an[ep]
        L.append(f'## {"药敏 (GDSC2 AUC)" if ep == "drug" else "CRISPR 基因效应"} 终点：{d["n_eval_cells"]} 细胞，{d["n_targets_evaluated"]}/{d["n_targets_total"]} 靶标，{d["n_strata"]} 个组织层\n')
        L.append('| 臂 | sRMSE | 95% bootstrap | Spearman | centered MSE | 相对 M 改善% | 95% 区间, P(>0) | 符号检验（胜/负/平，p） |'); L.append('|---|---|---|---|---|---|---|---|')
        for a, r in d['arms'].items():
            st = r.get('sign_test_vs_M') or {}
            L.append(f'| {a} | {fmt(r["score"], 5)} | [{fmt(r["boot_score"]["lo"])}, {fmt(r["boot_score"]["hi"])}] | {fmt(r["spearman"])} | {fmt(r["centered_mse"])} | '
                     f'{fmt(r.get("gain_vs_M_pct"), 3)} | {fmt_ci(r.get("gain_vs_M_boot"))} | {st.get("wins", "—")}/{st.get("losses", "—")}/{st.get("ties", "—")}，p={fmt(st.get("p_two_sided"), 3)} |')
        nc = [a for a, s in d['status'].items() if s != 'computed']
        if nc:
            L.append(f'\n未计算的臂：{nc}')
        L.append('\n### 追回份额与面板 R(k)\n')
        for y, v in d['share'].items():
            L.append(f'- M→{y}: room {fmt(v["room_pct"], 3)}%，recovered {fmt(v["recovered_pct"], 3)}%，share {fmt(v["share"], 3)} {fmt_ci(v["share_boot"])}')
        for a, v in d['panel_recovery'].items():
            L.append(f'- {a}（教师 {v["teacher"]}，教师增益 {fmt(v.get("teacher_gain_pct"), 3) if "teacher_gain_pct" in v else "—"}%）: R = {fmt(v["R"], 3)} {fmt_ci(v["R_boot"])}' + (f'，种子均值 sRMSE {fmt(v["score_mean_over_seeds"], 5)}' if 'score_mean_over_seeds' in v else ''))
        L.append('\n### 其他预注册对比\n')
        for n, c in d['comparisons'].items():
            if 'gain_pct' in c:
                L.append(f'- {n}: {fmt(c["gain_pct"], 3)}%，95% {fmt_ci(c["boot"])}，符号 {c["sign_test"]["wins"]}/{c["sign_test"]["losses"]}/{c["sign_test"]["ties"]} p={fmt(c["sign_test"]["p_two_sided"], 3)}')
            else:
                L.append(f'- {n}: 未计算')
        L.append('')
    L.append('## 边界声明\n')
    L.append('- 这是同一 CCLE 细胞系宇宙、同一协议内的封存集一次性确认，不是外部验证（PLAN §10）。')
    L.append('- 没有折，不做 Nadeau–Bengio 检验；不确定性来自分组 bootstrap（B=2000，种子 20261009，细胞按组织分层、药物按 ID 同时重采样）与逐药配对符号检验。')
    L.append('- 运行后不得基于结果改任何算子、臂、指标或判定规则（PLAN §7）。')
    L.append(f'\n输入与门：{json.dumps(cc.json_ready(meta.get("gates", {})), ensure_ascii=False)}')
    return '\n'.join(L) + '\n'


# ---------------------------------------------------------------- receipt gate (sealed mode only)
def receipt_gate():
    rp = RUN + '/RELEASE_RECEIPT.json'
    if not os.path.exists(rp):
        refuse(f'RELEASE_RECEIPT.json is absent ({rp}); the sealed evaluation cannot start (PLAN §7–§8). No raw file was opened.')
    try:
        rec = json.load(open(rp))
    except Exception as exc:
        refuse(f'RELEASE_RECEIPT.json unreadable: {exc!r}')
    if rec.get('protocol_id') != PROTOCOL_ID:
        refuse(f'receipt protocol_id {rec.get("protocol_id")!r} != {PROTOCOL_ID}')
    plan_sha = cc.sha_file(RUN + '/PLAN.md')
    if rec.get('plan_sha256') != plan_sha:
        refuse(f'PLAN.md sha {plan_sha} differs from the receipt ({rec.get("plan_sha256")})')
    code_dir = os.path.dirname(os.path.abspath(__file__))
    code_files = sorted(f for f in os.listdir(code_dir) if f.endswith('.py'))
    code_sha = {f: cc.sha_file(f'{code_dir}/{f}') for f in code_files}
    if rec.get('code_sha256') != code_sha:
        refuse(f'code sha256 differ from the receipt: receipt={rec.get("code_sha256")} disk={code_sha}')
    fz = RUN + '/frozen_models'
    fz_files = sorted(os.listdir(fz)) if os.path.isdir(fz) else []
    fz_sha = {f: cc.sha_file(f'{fz}/{f}') for f in fz_files}
    if rec.get('frozen_models_sha256') != fz_sha:
        refuse(f'frozen_models sha256 differ from the receipt: receipt={rec.get("frozen_models_sha256")} disk={fz_sha}')
    env = dict(numpy=np.__version__, scipy=scipy.__version__)
    if rec.get('environment', {}).get('numpy') != env['numpy'] or rec.get('environment', {}).get('scipy') != env['scipy'] or env != ENV_EXPECTED:
        refuse(f'environment {env} differs from the receipt {rec.get("environment")} / expected {ENV_EXPECTED}')
    ids = json.load(open(SPLIT_IDS))['internal_final']
    ids_sha = cc.sha_bytes(('\n'.join(sorted(ids)) + '\n').encode())
    if rec.get('internal_final_ids_sha256') != ids_sha or not ids_sha.startswith('e8f278c4'):
        refuse(f'internal_final ID-list sha {ids_sha} differs from the receipt ({rec.get("internal_final_ids_sha256")})')
    if os.path.exists(RUN + '/output'):
        refuse('output/ already exists: the sealed evaluation runs exactly once (PLAN §7, no retry)')
    return rec, dict(plan_sha256=plan_sha, code_sha256=code_sha, frozen_models_sha256=fz_sha, environment=env, internal_final_ids_sha256=ids_sha)


# ---------------------------------------------------------------- modes
def do_freeze(args):
    S, D, R, C, Fcr, ids = load_train()
    out = RUN + '/frozen_models'
    if os.path.exists(out):
        refuse(f'{out} exists; freezing happens once (remove manually only before registration of SHA256SUMS.v2)', 4)
    log(event='freeze_start', n_train=421, crispr_lines=int(np.isfinite(Fcr).any(1).sum()))
    M = fit_training_set(D, S['auc'], Fcr, 'internal_final_421', want_crispr=True)
    write_frozen(M, D, S, R['gene_ids'], S['auc'], Fcr, True, out, 'FROZEN on 421 development lines only')
    log(event='freeze_done', crispr_targets=int(len(M['cr_cols'])), generators={k: v['lam'] for k, v in M['F']['gen'].items()})


def rebuild_and_verify(S, D, auc, Fcr, frozen_dir, seed_tag, want_crispr):
    """Deterministically rebuild the fit side on the training set and compare with the frozen npz; selections are loaded."""
    Z = np.load(frozen_dir + '/model.npz', allow_pickle=False)
    M = fit_training_set(D, auc, Fcr, seed_tag, want_crispr, with_inner=False)
    sel = {'drug': {}, 'crispr': {}}
    for k in Z.files:
        for ep in ('drug', 'crispr'):
            if k.startswith(f'{ep}_sel_'):
                sel[ep][k[len(f'{ep}_sel_'):]] = Z[k]
    inv = {kz(a): a for a in set(DRUG_ARMS) | set(CR_ARMS)}
    sel = {ep: {inv[a]: v for a, v in d.items()} for ep, d in sel.items()}
    M['sel_pack'] = dict(sel=sel)
    A = pack_full(M['F'], M['PN'], M['PNc'], sel, M['cr_cols'], auc, Fcr if want_crispr else None, D)
    maxdiff = compare_packed(A, Z)
    log(event='frozen_verified', max_abs_diff=maxdiff, keys=len(A))
    return M, maxdiff


def do_dryrun(args):
    out = RUN + '/dryrun'
    os.makedirs(out, exist_ok=True)
    for sub in ('output', 'frozen_models_dryrun'):
        if os.path.exists(f'{out}/{sub}'):
            refuse(f'{out}/{sub} exists; remove the previous dry run first', 4)
    S, D, R, C, Fcr, ids = load_train()
    rng = np.random.default_rng(cc.SEED)
    perm = rng.permutation(421); ev, tr = np.sort(perm[:20]), np.sort(perm[20:])
    Dtr, Dev = D.sub(tr), D.sub(ev)
    auc_tr, auc_ev = S['auc'][tr], S['auc'][ev]; F_tr, F_ev = Fcr[tr], Fcr[ev]
    label = 'DRYRUN, 421-only (401 pseudo-train / 20 pseudo-eval, seed 20261009)'
    log(event='dryrun_start', label=label, pseudo_eval_ids=D.ids[ev].tolist(), crispr_in_eval=int(np.isfinite(F_ev).any(1).sum()))
    meta = dict(n_train=401, n_eval=20, gates={})
    # (1) loaders on synthetic endpoint files + identity-gated omics loader on the 5 parity lines (all 421-only values)
    sg = out + '/synthetic_gdsc2_421only.csv'; sio.write_synthetic_gdsc(sg, S)
    gp = sio.gdsc_parity_421(sg, S); meta['gates']['gdsc_parity_synthetic'] = {k: v for k, v in gp.items() if k != 'quality'}
    assert gp['passed'], gp
    auc_ev_loaded, q = sio.gdsc_load_ids(sg, D.ids[ev].tolist(), S['drug_ids'])
    assert np.array_equal(np.isnan(auc_ev_loaded), np.isnan(auc_ev)) and np.nanmax(np.abs(auc_ev_loaded - auc_ev)) == 0.
    sc = out + '/synthetic_crispr_310only.csv'; sio.write_synthetic_crispr(sc, C)
    cp, Fload = sio.crispr_parity_310(sc, RAWDIR + '/Model.csv', S, C); meta['gates']['crispr_parity_synthetic'] = cp
    assert cp['passed'], cp
    log(event='dryrun_loaders_ok', gdsc=meta['gates']['gdsc_parity_synthetic'], crispr=cp)
    Dom = sio.load_omics(raw_paths(S, R), sio.PARITY, forbidden=ids['validation'], universe=ids['train'] + ids['internal_final'], log=log)
    op = sio.omics_parity(Dom, S, R); meta['gates']['omics_parity_5_train_lines'] = op
    assert op['passed'], op
    log(event='dryrun_omics_loader_ok', **op)
    # (2) freeze on the 401, (3) rebuild/verify, (4) evaluate on the 20, (5) statistics
    M = fit_training_set(Dtr, auc_tr, F_tr, 'dryrun_401', want_crispr=True)
    fz = out + '/frozen_models_dryrun'
    write_frozen(M, Dtr, S, R['gene_ids'], auc_tr, F_tr, True, fz, label)
    M2, md = rebuild_and_verify(S, Dtr, auc_tr, F_tr, fz, 'dryrun_401', True)
    for ep in M['sel_pack']['sel']:
        for a, v in M['sel_pack']['sel'][ep].items():
            assert np.array_equal(v, M2['sel_pack']['sel'][ep][a])
    meta['gates']['frozen_rebuild_max_abs_diff'] = md
    R_ = evaluate_set(M2, Dtr, auc_tr, F_tr, Dev, auc_ev, F_ev, True, out + '/output', label)
    # spearman path check (grouped vs scipy loop) on the M arm
    m = R_['drug']['metrics']['M']; ref = cc.spearman_check(R_['drug']['truth'], R_['drug']['preds']['M'], R_['drug']['use'])
    both = np.isfinite(ref) & np.isfinite(m['rho']); meta['gates']['spearman_path_max_abs_diff'] = float(np.max(np.abs(ref[both] - m['rho'][both]))) if both.any() else 0.
    # panel-weight path check (fast exact routine vs verbatim slow routine) on a slice
    pc = M['PN']['P_cand'][:200]
    meta['gates']['panel_weight_fast_vs_slow'] = float(np.max(np.abs(cc.panel_weights(Dtr.p[:, pc], auc_tr) - cc.spearman_scores(Dtr.p[:, pc], auc_tr))))
    an = analyze(R_, Dev.tis, True, label)
    cc.dump_json(out + '/output/summary.json', dict(analysis=an, meta=meta))
    open(out + '/output/RESULT_DRAFT.md', 'w').write(result_markdown(an, meta))
    log(event='dryrun_done', F=an['F'], gates=meta['gates'])


def do_gates(args):
    """Code-equivalence gates against the development runs on the extra-supervision fold-0 banks."""
    out = RUN + '/dryrun/gates.json'
    os.makedirs(RUN + '/dryrun', exist_ok=True)
    S, D, R, C, Fcr, ids = load_train()
    G = dict(inner_rule_replay={}, m_kernel_block_max_abs_diff=0., L_protein_max_abs_diff=0., generator=dict(lam_mismatch=0, r2_mean=0., r2_median=0., width=0.),
             protein_panel_top16_equal={}, protein_panel_top64_score_max_abs_diff=0., gene_panel_top16_equal={}, random_panel_equal={}, prediction_max_abs_diff={})
    # inner-rule replay on all 5 folds (seed 20261008 reproduces the saved banks)
    outer = cc.cell_folds(S['tissues'], 20261008, outer=True)
    for f in range(5):
        a = np.flatnonzero(outer != f); inner = cc.cell_folds(S['tissues'][a], 20261008, nfolds=3)
        ok = [bool(np.array_equal(np.load(f'{EXT}/fold-{f}/inner-{c}/geometry.npz')['fit'], a[inner != c]) and np.array_equal(np.load(f'{EXT}/fold-{f}/inner-{c}/geometry.npz')['query'], a[inner == c])) for c in range(3)]
        G['inner_rule_replay'][f'fold-{f}'] = ok
    saved_gen = json.load(open(MATRIX + '/bank_info.json'))
    pb = json.load(open(PPANEL + '/bank_info.json'))
    rb = json.load(open(RPANEL + '/fold-0_bank_info.json')) if os.path.exists(RPANEL + '/fold-0_bank_info.json') else None
    BANK_ID = {'inner-0': 0, 'inner-1': 1, 'inner-2': 2, 'outer': 3}
    total = {}; count = {}; outer_grids = {}
    arms_cmp = ['M', 'M+RNA', 'M+P', 'M+Pval', 'M+vRNA', 'M+vP', 'M+vRNA_shuf', 'M+vP_shuf', 'T', 'M+T', 'M+P_graph'] + [f'M+P{k}' for k in cc.KS] + [f'M+G{k}' for k in cc.KS] + [f'M+Prand16_s{s}' for s in cc.SEEDS]
    for bank in ('inner-0', 'inner-1', 'inner-2', 'outer'):
        Gz = np.load(f'{EXT}/fold-0/{bank}/geometry.npz', allow_pickle=False)
        fit, q = Gz['fit'], Gz['query']; n = len(fit)
        Df, Dq = D.sub(fit), D.sub(q); y = S['auc'][fit]
        Fb = cc.fit_layers(Df.m, Df.X, Df.p, Df.tis, cc.stable_seed('oof', 'new_cell', 0, bank), cc.stable_seed('pair_shuffle', 'new_cell', 0, bank))
        PN = cc.fit_panels(Fb, Df.X, Df.p, y, [20261009, 0, 0, BANK_ID[bank]])
        B = cc.base_kernels(Fb, PN, Df.m, Df.X, Df.p, Df.tis, Dq.m, Dq.X, Dq.p, Dq.tis)
        G['m_kernel_block_max_abs_diff'] = max(G['m_kernel_block_max_abs_diff'], float(np.max(np.abs(B['M'][0] - Gz['m_only_kernel'][:n, :n]))), float(np.max(np.abs(B['M'][1] - Gz['m_only_cross'][:, :n]))))
        G['L_protein_max_abs_diff'] = max(G['L_protein_max_abs_diff'], float(np.max(np.abs(Fb['L_protein'] - Gz['L_protein'][:n, :n]))))
        sg = saved_gen[f'fold-0/{bank}']['generators']
        for k, v in Fb['gen'].items():
            G['generator']['lam_mismatch'] += int(v['lam'] != sg[k]['lam'])
            G['generator']['r2_mean'] = max(G['generator']['r2_mean'], abs(v['oof_r2_mean'] - sg[k]['oof_r2_mean']))
            G['generator']['r2_median'] = max(G['generator']['r2_median'], abs(v['oof_r2_median'] - sg[k]['oof_r2_median']))
            G['generator']['width'] = max(G['generator']['width'], abs(v['vstate'][2] - sg[k]['virtual_width']))
        top = pb[f'fold-0/{bank}']['top64']; order = PN['P_cand'][np.argsort(-PN['P_weights'], kind='stable')]
        G['protein_panel_top16_equal'][bank] = bool(np.array_equal(order[:16], np.array(top[:16]))) and bool(np.array_equal(order[:64], np.array(top)))
        G['protein_panel_top64_score_max_abs_diff'] = max(G['protein_panel_top64_score_max_abs_diff'], float(np.max(np.abs(np.sort(PN['P_weights'])[::-1][:64] - np.array(pb[f'fold-0/{bank}']['top64_score'])))))
        G['random_panel_equal'][bank] = all(np.array_equal(PN['panels'][f'Prand{k}_s{s}'], np.array(pb[f'fold-0/{bank}']['random_panels'][f'R{k}s{s}'])) for k in cc.RKS for s in cc.SEEDS)
        if rb is not None and f'fold-0/{bank}' in rb:
            gorder = PN['G_cand'][np.argsort(-PN['G_weights'], kind='stable')]
            G['gene_panel_top16_equal'][bank] = bool(np.array_equal(gorder[:16], np.array(rb[f'fold-0/{bank}']['top128'][:16])))
            G['random_panel_equal'][bank] &= all(np.array_equal(PN['panels'][f'Grand{k}_s{s}'], np.array(rb[f'fold-0/{bank}']['panels'][f'R{k}s{s}'])) for k in cc.RKS for s in cc.SEEDS)
        cache = {}; gc_ = None; A = cc.arm_configs()
        for arm in arms_cmp:
            g, gc_ = one_grid(arm, A, B, Fb, y, cache, gc_)
            if bank == 'outer':
                outer_grids[arm] = g
            else:
                ll, nn = cc.bank_losses(g, S['auc'][q], np.maximum(cc.nanstd0(y), 1e-8))
                total[arm] = total.get(arm, 0) + ll; count[arm] = count.get(arm, 0) + nn
        log(event='gate_bank_done', bank=bank, m_kernel=G['m_kernel_block_max_abs_diff'], L=G['L_protein_max_abs_diff'], gen=G['generator'], top16=G['protein_panel_top16_equal'][bank])
    preds = {a: cc.take(outer_grids[a], cc.choose(total[a], count[a], fallback=1)) for a in arms_cmp}
    Pm = np.load(MATRIX + '/fold-0_predictions.npz', allow_pickle=False)
    for a in ('M', 'M+RNA', 'M+P', 'M+Pval', 'M+vRNA', 'M+vP', 'M+vRNA_shuf', 'M+vP_shuf', 'T'):
        G['prediction_max_abs_diff'][f'matrix:{a}'] = float(np.max(np.abs(preds[a] - Pm['pred__' + a])))
    Pp = np.load(PPANEL + '/fold-0_predictions.npz', allow_pickle=False)
    for a, s in (('M+T', 'm_tissue'), ('M+P_graph', 'm_only__protein'), ('T', 'tissue_only')):
        G['prediction_max_abs_diff'][f'protein_panel:{a}'] = float(np.max(np.abs(preds[a] - Pp[s])))
    for k in cc.KS:
        G['prediction_max_abs_diff'][f'protein_panel:M+P{k}'] = float(np.max(np.abs(preds[f'M+P{k}'] - Pp[f'm_panel_k{k}'])))
    for s in cc.SEEDS:
        G['prediction_max_abs_diff'][f'protein_panel:M+Prand16_s{s}'] = float(np.max(np.abs(preds[f'M+Prand16_s{s}'] - Pp[f'random_k16_s{s}'])))
    if os.path.exists(RPANEL + '/fold-0_predictions.npz'):
        Pr = np.load(RPANEL + '/fold-0_predictions.npz', allow_pickle=False)
        for k in cc.KS:
            G['prediction_max_abs_diff'][f'rna_panel:M+G{k}'] = float(np.max(np.abs(preds[f'M+G{k}'] - Pr[f'm_panel_k{k}'])))
    G['passed'] = bool(all(all(v) for v in G['inner_rule_replay'].values()) and G['m_kernel_block_max_abs_diff'] <= 1e-11 and G['L_protein_max_abs_diff'] <= 1e-12
                       and G['generator']['lam_mismatch'] == 0 and max(G['generator']['r2_mean'], G['generator']['r2_median'], G['generator']['width']) <= 1e-9
                       and all(G['protein_panel_top16_equal'].values()) and all(G['random_panel_equal'].values())
                       and max(G['prediction_max_abs_diff'].values()) <= 1e-8)
    cc.dump_json(out, G)
    log(event='gates_done', **G)


def do_sealed(args):
    rec, hashes = receipt_gate()
    out = RUN + '/output'
    os.makedirs(out, exist_ok=False)
    global LOGFH
    os.makedirs(RUN + '/logs', exist_ok=True)
    LOGFH = open(RUN + f'/logs/sealed_{datetime.datetime.now():%Y%m%d_%H%M%S}.log', 'a')
    log(event='receipt_gate_passed', **hashes)
    S, D, R, C, Fcr, ids = load_train()
    want_cr = bool(args.crispr)
    # 1. rebuild + verify frozen model (421 only)
    M, md = rebuild_and_verify(S, D, S['auc'], Fcr, RUN + '/frozen_models', 'internal_final_421', True)
    meta = dict(n_train=421, n_eval=123, gates=dict(frozen_rebuild_max_abs_diff=md, receipt=hashes))
    # 2. endpoint parity on development lines only
    gp = sio.gdsc_parity_421(args.gdsc, S); meta['gates']['gdsc_parity_421'] = {k: v for k, v in gp.items() if k != 'quality'}
    log(event='gdsc_parity_421', **meta['gates']['gdsc_parity_421'])
    if not gp['passed']:
        refuse('GDSC2 AUC parity on the 421 failed; no internal_final value was decoded', 5)
    cr_ok = False
    if want_cr:
        try:
            cp, _ = sio.crispr_parity_310(args.crispr, RAWDIR + '/Model.csv', S, C); meta['gates']['crispr_parity_310'] = cp; cr_ok = bool(cp['passed'])
            log(event='crispr_parity_310', **cp)
        except Exception as exc:
            meta['gates']['crispr_parity_310'] = dict(passed=False, error=repr(exc)); log(event='crispr_parity_failed', error=repr(exc))
        if not cr_ok:
            log(event='crispr_endpoint_skipped', reason='parity on the 310 development lines failed; drug endpoint continues')
    cc.dump_json(out + '/pre_sealed_checkpoint.json', meta)
    # 3. sealed values (first and only time): omics for 123 (+5 parity lines), AUC for 123, CRISPR for 123
    fin = ids['internal_final']
    Dom = sio.load_omics(raw_paths(S, R), list(fin) + sio.PARITY, forbidden=ids['validation'], universe=ids['train'] + ids['internal_final'], log=log)
    op = sio.omics_parity(Dom, S, R); meta['gates']['omics_parity_5_train_lines'] = op
    log(event='omics_parity', **op)
    if not op['passed']:
        refuse('omics parity on the 5 development lines failed after loading; stopping before any score', 6)
    keep = np.array([s in set(fin) for s in Dom['sample_ids']])
    assert keep.sum() == 123 and set(Dom['sample_ids'][keep].tolist()) == set(fin) and not (set(Dom['sample_ids'].tolist()) & set(ids['validation']))
    Dq = Data(Dom['m'][keep], Dom['rna'][keep], Dom['p'][keep], Dom['tissues'][keep], Dom['sample_ids'][keep])
    auc_q, q = sio.gdsc_load_ids(args.gdsc, Dq.ids.tolist(), S['drug_ids']); meta['gates']['gdsc_123'] = q
    log(event='sealed_auc_loaded', samples_with_response=q['samples_with_response'], observed_pairs=q['observed_pairs'], outside_295=len(q['drugs_outside_frozen_295']))
    F_q = None
    if cr_ok:
        try:
            F_q, ci_ = sio.crispr_load(args.crispr, RAWDIR + '/Model.csv', Dq.ids.tolist(), C['gene_ids'].tolist(), forbidden=ids['validation'])
            meta['gates']['crispr_123'] = ci_; log(event='sealed_crispr_loaded', **ci_)
        except Exception as exc:
            cr_ok = False; meta['gates']['crispr_123'] = dict(error=repr(exc)); log(event='crispr_123_failed', error=repr(exc))
    meta['eval_tissues'] = dict(zip(*np.unique(Dq.tis, return_counts=True)))
    meta['eval_rna_batch'] = dict(zip(*np.unique(Dom['rna_batch'][keep], return_counts=True)))
    # 4. all arms, one invocation; each arm written as soon as computed
    R_ = evaluate_set(M, D, S['auc'], Fcr, Dq, auc_q, F_q, cr_ok, out, 'SEALED internal_final 123')
    an = analyze(R_, Dq.tis, cr_ok, 'SEALED internal_final 123')
    cc.dump_json(out + '/summary.json', dict(analysis=an, meta=meta, receipt=rec))
    open(out + '/RESULT_DRAFT.md', 'w').write(result_markdown(an, meta))
    log(event='sealed_done', F=an['F'])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=('freeze', 'gates', 'dryrun', 'sealed'))
    p.add_argument('--gdsc', help='GDSC2 fitted dose-response CSV (sealed mode)')
    p.add_argument('--crispr', help='DepMap CRISPRGeneEffect.csv (sealed mode, optional)')
    args = p.parse_args()
    if args.mode == 'sealed':
        if not args.gdsc:
            refuse('--gdsc is required in sealed mode (checked before the receipt gate; nothing was opened)')
        do_sealed(args)
    elif args.mode == 'freeze':
        do_freeze(args)
    elif args.mode == 'gates':
        do_gates(args)
    else:
        do_dryrun(args)


if __name__ == '__main__':
    main()

"""Nested, finite transfer of a strong measured-protein teacher to M-only.

No query outcomes enter fitting, tuning, or pseudo-target generation.
Original tissue outer predictions remain the base; all candidates are separate.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from creed.science.metabolite_query import group_folds
from creed.science.metabolite_protein_query import panel_for_bank

ALPHAS = (.1, 1., 10., 100.)
BETAS = (0., .25, .5, 1.)
TEACHERS = ('m_only', 'measured_p', 'measured_mask')
TARGETS = ('protein_delta', 'mask_delta', 'value_delta', 'shuffled_delta', 'response_residual')
CONFIGS = tuple((k, a, b) for b in BETAS for k in ('rbf', 'linear') for a in ALPHAS)
RECOVERY_CONFIGS = tuple((k, a) for k in ('rbf', 'linear') for a in ALPHAS)
SOURCE_SHA = '52d71517a8952e83a1603cd831f09449544102d0ca5ff829ff01be9bf79cd637'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def clean(v):
    if isinstance(v, np.ndarray): return clean(v.tolist())
    if isinstance(v, np.generic): return clean(v.item())
    if isinstance(v, dict): return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)): return [clean(x) for x in v]
    if isinstance(v, float) and not np.isfinite(v): return None
    return v


def save_json(path, v):
    Path(path).write_text(json.dumps(clean(v), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def csv_save(path, rows):
    with Path(path).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'little')


def cell_folds(tissues, seed, nfolds=5, outer=False):
    """Outer folds reproduce the already frozen functional-operator algorithm."""
    labels = np.asarray(tissues).astype(str)
    folds = np.full(len(labels), -1, int); loads = np.zeros(nfolds, int)
    for label in sorted(np.unique(labels), key=lambda t: (-np.sum(labels == t), t)):
        idx = np.flatnonzero(labels == label)
        rng = np.random.default_rng(stable_seed(seed, 'new_cell_folds' if outer else 'nested_cell_folds', label))
        rng.shuffle(idx)
        order = rng.permutation(nfolds)
        order = order[np.argsort(loads[order], kind='stable')]
        for j, row in enumerate(idx):
            f = int(order[j % nfolds]); folds[row] = f; loads[f] += 1
    assert np.all(folds >= 0) and np.all(loads > 0)
    return folds


def standardize(x, q):
    mu = x.mean(0); sd = x.std(0); sd = np.where(sd < 1e-8, 1., sd)
    return (x - mu) / sd, (q - mu) / sd


def kernel(x, q, kind='rbf'):
    z, h = standardize(x, q)
    if kind == 'linear':
        return z @ z.T / z.shape[1], h @ z.T / z.shape[1]
    d = cdist(z, z, 'sqeuclidean'); positive = d[np.triu_indices(len(z), 1)]
    positive = positive[positive > 0]
    if not len(positive): return np.ones_like(d), np.ones((len(q), len(x)))
    width = np.median(positive)
    return np.exp(-d / (2 * width)), np.exp(-cdist(h, z, 'sqeuclidean') / (2 * width))


def predict_grid(k, h, y):
    """Same mean-centred masked KRR as old head; low support returns fit mean."""
    pred = np.empty((4, len(h), y.shape[1])); groups = {}
    for j in range(y.shape[1]): groups.setdefault(np.isfinite(y[:, j]).tobytes(), []).append(j)
    for cols in groups.values():
        obs = np.flatnonzero(np.isfinite(y[:, cols[0]]))
        if not len(obs): raise ValueError('No current-fit label; stop rather than borrow query labels')
        yy = y[np.ix_(obs, cols)]; mean = yy.mean(0)
        pred[:, :, cols] = mean
        if len(obs) < 20: continue
        kk = k[np.ix_(obs, obs)]; hh = h[:, obs]
        for ai, alpha in enumerate(ALPHAS):
            coef = cho_solve(cho_factor(kk + alpha * np.eye(len(obs)), lower=True), yy - mean)
            pred[ai][:, cols] = hh @ coef + mean
    assert np.isfinite(pred).all()
    return pred


def losses(pred, truth, scale):
    e = ((pred - truth[None]) / scale) ** 2
    n = np.isfinite(e).sum(1)
    return np.divide(np.nansum(e, axis=1), n, out=np.zeros_like(n, float), where=n > 0), (n > 0).astype(int)


def choose(total, count, fallback=1):
    avg = np.divide(total, count, out=np.full_like(total, np.inf), where=count > 0)
    idx = np.argmin(avg, axis=0); idx[np.all(count == 0, axis=0)] = fallback
    return idx


def take(grid, selected):
    return np.take_along_axis(grid, selected[None, None, :], axis=0)[0]


class Experiment:
    def __init__(self, source, parent, scenario, fold, panel, out):
        self.s, self.parent, self.scenario, self.fold, self.panel, self.out = source, parent, scenario, fold, panel, out
        self.y = source['auc'][:, panel]
        self.raw_cache = {}; self.strong_cache = {}; self.trace = []; self.original = {}
        if scenario == 'tissue':
            for name in ('inner-0', 'inner-1', 'inner-2', 'outer'):
                p = parent / f'fold-{fold}' / name
                meta = json.loads((p / 'selection.json').read_text())
                assert meta['panel'] == panel.tolist()
                self.original[self.key(meta['fit'], meta['query'])] = p

    @staticmethod
    def key(a, q): return (tuple(map(int, a)), tuple(map(int, q)))

    def split(self, a):
        if self.scenario == 'tissue': return group_folds(self.s['tissues'][a], 3)[0]
        return cell_folds(self.s['tissues'][a], 20261008, nfolds=3)

    def raw(self, a, q):
        key = self.key(a, q)
        if key in self.raw_cache: return self.raw_cache[key]
        assert not set(a) & set(q)
        if self.scenario == 'tissue': assert not set(self.s['tissues'][a]) & set(self.s['tissues'][q])
        original = self.original.get(key)
        if original is not None:
            z = load(original / 'per-alpha-predictions.npz')
            result = {k: z[k] for k in TEACHERS}
            pi = load(original / 'panel.npz')['indices']
        else:
            # Eligibility is exactly the original training-only protein rule,
            # including its tissue-based support checks even in the cell scenario.
            pi = panel_for_bank(self.s['p'][a], self.s['tissues'][a])['indices']
            if not len(pi): raise ValueError('No eligible protein in current fit')
            m, mh = kernel(self.s['m'][a], self.s['m'][q])
            p, pq = self.s['p'][np.ix_(a, pi)], self.s['p'][np.ix_(q, pi)]
            obs, oq = np.isfinite(p), np.isfinite(pq); mean = np.nanmean(p, axis=0)
            pk, ph = kernel(np.where(obs, p, mean), np.where(oq, pq, mean))
            mk, xh = kernel(obs.astype(float), oq.astype(float))
            result = {name: predict_grid(k, h, self.y[a]) for name, k, h in (
                ('m_only', m, mh), ('measured_p', .5*m+.5*pk, .5*mh+.5*ph),
                ('measured_mask', .5*m+.5*mk, .5*mh+.5*xh))}
        assert all(np.isfinite(v).all() for v in result.values())
        self.trace.append(dict(kind='raw', fit=a, query=q, protein_count=len(pi),
            low_support_drugs=int(np.sum(np.isfinite(self.y[a]).sum(0) < 20)),
            original_bank=str(original) if original else None))
        self.raw_cache[key] = result
        return result

    def strong(self, a, q):
        key = self.key(a, q)
        if key in self.strong_cache: return self.strong_cache[key]
        f = self.split(a); total = {k: np.zeros((4, len(self.panel))) for k in TEACHERS}
        count = {k: np.zeros((4, len(self.panel))) for k in TEACHERS}
        for c in range(3):
            aa, vv = a[f != c], a[f == c]
            grid = self.raw(aa, vv)
            scale = np.maximum(np.nanstd(self.y[aa], axis=0), 1e-8)
            for arm in TEACHERS:
                l, n = losses(grid[arm], self.y[vv], scale); total[arm] += l; count[arm] += n
        selected = {k: choose(total[k], count[k]) for k in TEACHERS}
        grid = self.raw(a, q); result = {k: take(grid[k], selected[k]) for k in TEACHERS}
        saved = self.out / 'teacher_selection'; saved.mkdir(exist_ok=True)
        selection_file = saved / f'{len(self.strong_cache):04d}.npz'
        np.savez_compressed(selection_file,fit=a,query=q,inner_ids=f,
            **{k+'_loss':v for k,v in total.items()},**{k+'_count':v for k,v in count.items()},
            **{k+'_selected':v for k,v in selected.items()})
        self.trace.append(dict(kind='strong', fit=a, query=q, inner_ids=f, selected=selected,
            selection_file=str(selection_file.relative_to(self.out))))
        self.strong_cache[key] = result
        return result

    def student(self, a, q, name):
        f = self.split(a); oof = {k: np.full((len(a), len(self.panel)), np.nan) for k in TEACHERS}
        permutation = np.arange(len(a))
        for c in range(3):
            pos = np.flatnonzero(f == c); aa, vv = a[f != c], a[pos]
            t = self.strong(aa, vv)
            for arm in TEACHERS: oof[arm][pos] = t[arm]
            # Whole drug vectors shuffled only within this common teacher-held block.
            rng = np.random.default_rng(stable_seed('delta_shuffle', self.scenario, self.fold, tuple(a), c))
            permutation[pos] = rng.permutation(pos)
        delta = oof['measured_p'] - oof['m_only']
        targets = dict(protein_delta=delta, mask_delta=oof['measured_mask']-oof['m_only'],
            value_delta=oof['measured_p']-oof['measured_mask'], shuffled_delta=delta[permutation],
            response_residual=self.y[a]-oof['m_only'])
        assert all(np.isfinite(x).all() for k, x in targets.items() if k != 'response_residual')
        grids = {}
        for kind in ('rbf', 'linear'):
            k, h = kernel(self.s['m'][a], self.s['m'][q], kind)
            for target, y in targets.items(): grids[(target, kind)] = predict_grid(k, h, y)
        base = self.strong(a, q)
        dest = self.out / name; dest.mkdir()
        np.savez_compressed(dest / 'contract.npz', fit=a, query=q, panel=self.panel,
            oof_fold=f, permutation=permutation, **{'oof_'+k:v for k,v in oof.items()},
            **{'target_'+k:v for k,v in targets.items()}, **{'query_'+k:v for k,v in base.items()})
        np.savez_compressed(dest / 'student_grids.npz', **{t+'__'+k:v for (t,k),v in grids.items()})
        return grids, base, targets


def metrics(truth, prediction, sd, use):
    mask = np.isfinite(truth); n = mask.sum(0)
    mse = np.nansum(((prediction-truth)/sd)**2, axis=0) / np.maximum(n, 1)
    bias = (np.divide(np.where(mask,prediction,0).sum(0), n, out=np.zeros_like(sd), where=n>0)
            - np.nanmean(truth,axis=0)) / sd
    rho = np.full(len(sd), np.nan)
    for j in np.flatnonzero(use):
        valid = mask[:,j]
        if np.std(truth[valid,j]) > 0 and np.std(prediction[valid,j]) > 1e-15:
            rho[j] = spearmanr(truth[valid,j], prediction[valid,j]).statistic
    return dict(score=float(np.sqrt(mse[use].mean())), spearman=float(np.nanmean(rho[use])),
        centered_mse=float((mse-bias*bias)[use].mean()), mse=mse, rho=rho, count=n)


def run_fold(source, args, f, outer_ids, scores, recoveries, perdrug):
    a, q = np.flatnonzero(outer_ids != f), np.flatnonzero(outer_ids == f)
    if args.scenario == 'tissue':
        # Original drug panel is fixed, including original fold-specific eligibility.
        ref_indices = load(args.parent / f'fold-{f}' / 'predictions.npz')
        panel = ref_indices['panel']; np.testing.assert_array_equal(a, ref_indices['fit']); np.testing.assert_array_equal(q, ref_indices['held'])
    else: panel = np.arange(source['auc'].shape[1])
    out = args.output / f'fold-{f}'; out.mkdir()
    exp = Experiment(source, args.parent, args.scenario, f, panel, out)
    inner = exp.split(a); ndrug = len(panel)
    total = {k: np.zeros((len(CONFIGS), ndrug)) for k in TARGETS}
    count = {k: np.zeros_like(v) for k,v in total.items()}
    rt = np.zeros((len(RECOVERY_CONFIGS), ndrug)); rn = np.zeros_like(rt)
    for c in range(3):
        aa, vv = a[inner != c], a[inner == c]
        grids, b, _ = exp.student(aa, vv, f'inner-{c}')
        scale = np.maximum(np.nanstd(exp.y[aa], axis=0), 1e-8)
        for target in TARGETS:
            pred = np.stack([b['m_only'] + beta * grids[(target,kind)][ALPHAS.index(alpha)] for kind,alpha,beta in CONFIGS])
            l,n = losses(pred, exp.y[vv], scale); total[target] += l; count[target] += n
        pred = np.concatenate([grids[('protein_delta',k)] for k in ('rbf','linear')])
        l,n = losses(pred, b['measured_p']-b['m_only'], scale); rt += l; rn += n
        print(json.dumps(dict(event='inner_done',scenario=args.scenario,fold=f,inner=c,raw_fits=len(exp.raw_cache))),flush=True)
    selected = {k:choose(total[k],count[k],fallback=0) for k in TARGETS}
    rselected = choose(rt,rn)
    # Freeze inner choices on disk before generating the outer candidate predictions.
    save_json(out/'selection.json',dict(selected=selected, recovery_selected=rselected,configs=CONFIGS,
        recovery_configs=RECOVERY_CONFIGS, losses=total, counts=count, recovery_losses=rt, recovery_counts=rn))
    grids, base, targets = exp.student(a,q,'outer')
    parity = {}
    if args.scenario == 'tissue':
        for k in TEACHERS:
            parity[k] = float(np.max(np.abs(base[k]-ref_indices[k])))
            np.testing.assert_allclose(base[k],ref_indices[k],rtol=1e-11,atol=1e-11)
            base[k] = ref_indices[k].copy()  # exact saved baseline; never replaced by a worse refit
        save_json(out/'original_parity.json',dict(status='PASS',max_absolute_errors=parity))
    pred = dict(base)
    correction = {}
    for target in TARGETS:
        gg = np.stack([beta*grids[(target,kind)][ALPHAS.index(alpha)] for kind,alpha,beta in CONFIGS])
        correction[target] = take(gg,selected[target]); pred['transfer_'+target] = base['m_only']+correction[target]
    recovery_grid = np.concatenate([grids[('protein_delta',k)] for k in ('rbf','linear')])
    g = take(recovery_grid,rselected); pred['delta_fit_diagnostic'] = base['m_only']+g
    sd = np.maximum(np.nanstd(exp.y[a],axis=0),1e-8); truth=exp.y[q]; use=np.isfinite(truth).sum(0)>=5
    np.savez_compressed(out/'predictions.npz',fit=a,held=q,panel=panel,drug_ids=source['drug_ids'][panel],
        sample_ids=source['sample_ids'][q],fit_sample_ids=source['sample_ids'][a],
        truth=truth,train_sd=sd,evaluated=use,teacher_delta=base['measured_p']-base['m_only'],
        predicted_delta=g,train_delta_mean=targets['protein_delta'].mean(0),**pred)
    for arm,p in pred.items():
        m=metrics(truth,p,sd,use)
        scores.append(dict(fold=f,arm=arm,score=m['score'],spearman=m['spearman'],centered_mse=m['centered_mse'],drugs=int(use.sum())))
        for j in range(ndrug):
            perdrug.append(dict(fold=f,arm=arm,drug_id=str(source['drug_ids'][panel[j]]),count=int(m['count'][j]),
                evaluated=bool(use[j]),standardized_mse=float(m['mse'][j]),spearman=float(m['rho'][j])))
    delta=base['measured_p']-base['m_only']; mean=targets['protein_delta'].mean(0)
    denom0=((delta/sd)**2).mean(0); denomm=(((delta-mean)/sd)**2).mean(0); num=(((g-delta)/sd)**2).mean(0)
    # Aggregate same-scale squared errors; avoid averaging unstable near-zero per-drug R2.
    recoveries.append(dict(fold=f,delta_mse=float(num[use].mean()),zero_mse=float(denom0[use].mean()),
        train_mean_mse=float(denomm[use].mean()),skill_vs_zero=float(1-num[use].mean()/denom0[use].mean()),
        skill_vs_train_mean=float(1-num[use].mean()/denomm[use].mean()),
        teacher_risk_gain=float(np.nanmean([r['centered_mse'] for r in scores if r['fold']==f and r['arm']=='m_only'])-
            np.nanmean([r['centered_mse'] for r in scores if r['fold']==f and r['arm']=='measured_p']))))
    np.savez_compressed(out/'recoverability.npz',num=num,zero=denom0,train_mean=denomm)
    save_json(out/'fit_trace.json',exp.trace)
    save_json(out/'status.json',dict(status='completed',raw_fits=len(exp.raw_cache),strong_fits=len(exp.strong_cache)))
    print(json.dumps(dict(event='fold_done',scenario=args.scenario,fold=f,scores=[r for r in scores if r['fold']==f],recovery=recoveries[-1])),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scenario',choices=('tissue','new_cell'),required=True);p.add_argument('--folds',type=int,default=5)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    assert sha(args.parent/'source-inputs.npz')==SOURCE_SHA
    source=load(args.parent/'source-inputs.npz'); assert source['m'].shape==(421,225) and source['p'].shape==(421,6692)
    foldids=source['fold_ids'] if args.scenario=='tissue' else cell_folds(source['tissues'],20261008,outer=True)
    protocol=dict(scenario=args.scenario,source_sha256=SOURCE_SHA,script_sha256=sha(__file__),outer_fold_ids=foldids,
        alpha_grid=ALPHAS,beta_grid=BETAS,student_configs=CONFIGS,targets=TARGETS,
        teacher='old M RBF and 0.5M+0.5measured P/mask RBF; per-drug nested 3-fold alpha selection',
        pseudo_targets='strict three-fold crossfit separately inside EACH student fitting partition; each teacher tuned within own fit',
        student='M only, two fixed kernel families; alpha/family/beta selected from inner response outcomes',
        recovery_diagnostic='separate alpha/family choice on inner held teacher-delta error, beta1',
        shuffle='fixed whole drug vectors permuted WITHIN each common teacher held block',
        protein_eligibility='original panel_for_bank with train-only coverage and 3 tissue support checks',
        low_support='if nested teacher or observed-residual label count<20, use current-fit mean for every alpha; report counts',
        no_innovation_claim=True,scientific_claim=False,external_confirmation=False,
        note='Repeated development data. Failure is not an information-theoretic bound. Zero correction is an inner candidate, not a guarantee on unseen data.')
    save_json(args.output/'protocol.json',protocol)
    scores=[];recoveries=[];perdrug=[]
    for f in range(args.folds):
        run_fold(source,args,f,foldids,scores,recoveries,perdrug)
        csv_save(args.output/'fold_metrics.csv',scores);csv_save(args.output/'recoverability.csv',recoveries)
        csv_save(args.output/'per_drug_metrics.csv',perdrug)
    summary={a:{k:float(np.mean([r[k] for r in scores if r['arm']==a])) for k in ('score','spearman','centered_mse')}
        for a in sorted(set(r['arm'] for r in scores))}
    save_json(args.output/'summary.json',dict(status='completed',scenario=args.scenario,folds=args.folds,
        summary=summary,recoverability=recoveries,seconds=time.perf_counter()-start))
    print(json.dumps(dict(event='done',scenario=args.scenario,summary=summary,seconds=time.perf_counter()-start)),flush=True)


if __name__=='__main__':main()

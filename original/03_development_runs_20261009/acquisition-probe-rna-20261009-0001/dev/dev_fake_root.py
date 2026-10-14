"""Synthetic fake root for an engineering smoke test of acquisition_probe_rna.py (reads NO real data).
Builds the file layout the probe expects (extra-supervision banks, source-inputs, rna421, matrix saved arms) from random
numbers, using the probe module's own functions so the saved arms are self-consistent.
usage: python dev_fake_root.py FAKE_ROOT
"""
import csv
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acquisition_probe_rna as ap

N, NM, D, G, NE = 150, 225, 40, 3200, 6
rng = np.random.default_rng(0)


def write_root(root):
    os.makedirs(f'{root}/gdsc-metabolite-full-protein-20260908-0001/output/result', exist_ok=True)
    os.makedirs(f'{root}/extra-supervision-20261008-0001/resources', exist_ok=True)
    os.makedirs(f'{root}/rna421-20260906-0002', exist_ok=True)
    lat = rng.normal(size=(N, 5))
    m = lat @ rng.normal(size=(5, NM)) + rng.normal(size=(N, NM))
    X = np.log1p(np.exp(lat @ rng.normal(size=(5, G)) * .5 + rng.normal(size=(N, G)) + 3))
    auc = lat @ rng.normal(size=(5, D)) * .3 + rng.normal(size=(N, D)) * .5 + .7
    auc[rng.random((N, D)) < .15] = np.nan
    sids = np.array([f'SIDM{i:05d}' for i in range(N)]); dids = np.array([f'D{j:03d}' for j in range(D)])
    tissues = rng.choice(['lung', 'breast', 'skin', 'blood'], size=N)
    np.savez(f'{root}/gdsc-metabolite-full-protein-20260908-0001/output/result/source-inputs.npz', m=m, auc=auc, tissues=tissues, sample_ids=sids, drug_ids=dids)
    e_auc = rng.normal(size=(NE, D)) * .5 + .7; e_auc[rng.random((NE, D)) < .3] = np.nan
    e_m = rng.normal(size=(NE, NM))
    np.savez(f'{root}/extra-supervision-20261008-0001/resources/m_gdsc_extra19.npz', auc=e_auc, drug_ids=dids, m=e_m)
    np.save(f'{root}/rna421-20260906-0002/rna_log_cpm.npy', X); np.save(f'{root}/rna421-20260906-0002/sample_ids.npy', sids)
    np.save(f'{root}/rna421-20260906-0002/gene_symbols.npy', np.array([f'GENE{g}' for g in range(G)]))
    S = dict(m=m, auc=auc); E = dict(auc=e_auc, m=e_m)
    for sc in ('new_cell', 'tissue'):
        ext = f'{root}/extra-supervision-20261008-0001/output/{sc}'; mat = f'{root}/crosslayer-matrix-20261009-0001/output/{sc}'
        os.makedirs(mat, exist_ok=True)
        perm = rng.permutation(N); folds = np.array_split(perm, 5); fm_rows = []
        for f in range(5):
            held = np.sort(folds[f]); fit = np.sort(np.setdiff1d(np.arange(N), held))
            panel = np.arange(D) if sc == 'new_cell' else np.sort(rng.choice(D, D - 3, replace=False))
            ip = rng.permutation(fit); iparts = np.array_split(ip, 3)
            banks = {}
            for j in range(3):
                q = np.sort(iparts[j]); banks[f'inner-{j}'] = (np.sort(np.setdiff1d(fit, q)), q)
            banks['outer'] = (fit, held)
            tot = {}; cnt = {}; mtot = {}; mcnt = {}; outer = {}
            for bank, (bf, bq) in banks.items():
                d = f'{ext}/fold-{f}/{bank}'; os.makedirs(d, exist_ok=True)
                ei = np.arange(NE) if sc == 'new_cell' else np.arange(NE - 2)
                st = ap.kernel_state(m[bf])
                mk = ap.rbf(np.vstack([m[bf], e_m[ei]]), np.vstack([m[bf], e_m[ei]]), st); mh = ap.rbf(m[bq], np.vstack([m[bf], e_m[ei]]), st)
                y = auc[np.ix_(bf, panel)]; ye = np.vstack([y, e_auc[np.ix_(ei, panel)]])
                g0 = ap.predict_grid(mk[:len(bf), :len(bf)], mh[:, :len(bf)], y); g1 = ap.predict_grid(mk, mh, ye)
                np.savez(f'{d}/geometry.npz', fit=bf, query=bq, panel=panel, extra_indices=ei, m_only_kernel=mk, m_only_cross=mh)
                np.savez(f'{d}/grids.npz', original__m_only=g0, extra__m_only=g1)
                # matrix arms on this bank
                B = dict(fit=bf, query=bq, panel=panel, n=len(bf), KM=mk[:len(bf), :len(bf)].copy(), HM=mh[:, :len(bf)].copy(), y=y)
                xf = X[bf]; cand = np.argsort(-xf.var(0), kind='stable')[:ap.TOP_GENES]; Xf = xf[:, cand]; Xq = X[np.ix_(bq, cand)]
                mu = Xf.mean(0); sd = Xf.std(0); sd = np.where(sd < 1e-8, 1., sd); B['Z_fit'] = (Xf - mu) / sd
                gm, _ = ap.matrix_vrna_grid(B, sc, f, bank)
                H = ap.Head(B['KM'], B['HM'], Xf, y); gr = H.grid(Xq)
                if bank == 'outer':
                    outer = dict(m=g0, e=g1, mv=gm, mr=gr)
                else:
                    truth = auc[np.ix_(bq, panel)]; sdd = np.maximum(np.nanstd(y, axis=0), 1e-8)
                    for a, g in (('o', g0), ('e', g1), ('M+RNA', gr), ('M+vRNA', gm)):
                        ll, nn = ap.bank_losses(g, truth, sdd); tot[a] = tot.get(a, 0) + ll; cnt[a] = cnt.get(a, 0) + nn
                        if a in ('M+RNA', 'M+vRNA'):
                            mtot[f'{a}__{bank}__loss'] = ll; mcnt[f'{a}__{bank}__count'] = nn
            truth = auc[np.ix_(held, panel)]; tsd = np.maximum(np.nanstd(auc[np.ix_(fit, panel)], axis=0), 1e-8)
            use = (np.isfinite(truth).sum(0) >= 3) & (np.nanstd(truth, axis=0) > 0)
            po = ap.take(outer['m'], ap.choose(tot['o'], cnt['o'])); pe = ap.take(outer['e'], ap.choose(tot['e'], cnt['e']))
            np.savez(f'{ext}/fold-{f}/predictions.npz', fit=fit, held=held, panel=panel, truth=truth, train_sd=tsd, evaluated=use, original__m_only=po, extra__m_only=pe)
            for a, p in (('original__m_only', po), ('extra__m_only', pe)):
                fm_rows.append(dict(fold=f, arm=a, score=ap.metrics(truth, p, tsd, use)['score']))
            sr = ap.choose(tot['M+RNA'], cnt['M+RNA']); sv = ap.choose(tot['M+vRNA'], cnt['M+vRNA'])
            np.savez(f'{mat}/fold-{f}_predictions.npz', fit=fit, held=held, panel=panel, **{'pred__M+RNA': ap.take(outer['mr'], sr), 'pred__M+vRNA': ap.take(outer['mv'], sv),
                                                                                     'sel__M+RNA': sr, 'sel__M+vRNA': sv})
            np.savez(f'{mat}/fold-{f}_inner_losses.npz', **mtot, **mcnt)
        with open(f'{ext}/fold_metrics.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['fold', 'arm', 'score']); w.writeheader(); w.writerows(fm_rows)
    print('fake root written', root)


if __name__ == '__main__':
    write_root(sys.argv[1])

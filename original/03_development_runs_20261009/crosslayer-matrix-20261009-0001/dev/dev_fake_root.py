"""Dev-only smoke test: build a tiny synthetic run root with the same file layout, then crosslayer.py can run against it
with CREED_ROOT=<root> CREED_SKIP_SHA=1. No real data is read. The 'saved' grids/predictions are produced with the same
reference functions, so the reproduction gates test plumbing (shapes, keys, selection, ordering), not science.

usage: python dev_fake_root.py FAKE_ROOT
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crosslayer as C  # noqa: E402

root = sys.argv[1]
rng = np.random.default_rng(0)
N, NM, NG, NP, ND, NE = 70, 12, 90, 60, 15, 6
m = rng.normal(size=(N, NM)); X = rng.normal(size=(N, NG)) + m[:, :1] * .5
p = rng.normal(size=(N, NP)) + m[:, 1:2] * .5; p[rng.random(p.shape) < .35] = np.nan
p[:, :5] = np.nan; p[rng.random(N) < .5, 5:8] = np.nan  # some proteins below 50% coverage
auc = m @ rng.normal(size=(NM, ND)) * .2 + rng.normal(size=(N, ND)); auc[rng.random(auc.shape) < .1] = np.nan
auc[:, 0] = np.nan; auc[rng.random(N) < .6, 0] = 1.  # one sparse drug (<20 labels in inner fits)
tissues = rng.choice(['A', 'B', 'C', 'D'], N)
sid = np.array([f'S{i:03d}' for i in range(N)]); did = np.array([f'D{j:02d}' for j in range(ND)])
os.makedirs(f'{root}/gdsc-metabolite-full-protein-20260908-0001/output/result'); os.makedirs(f'{root}/rna421-20260906-0002')
os.makedirs(f'{root}/extra-supervision-20261008-0001/resources')
np.savez(f'{root}/gdsc-metabolite-full-protein-20260908-0001/output/result/source-inputs.npz', m=m, auc=auc, p=p, tissues=tissues,
         sample_ids=sid, drug_ids=did, protein_ids=np.array([f'P{i}' for i in range(NP)]), metabolite_ids=np.array([f'M{i}' for i in range(NM)]), fold_ids=np.arange(N) % 5)
np.save(f'{root}/rna421-20260906-0002/rna_log_cpm.npy', X); np.save(f'{root}/rna421-20260906-0002/sample_ids.npy', sid)
em = rng.normal(size=(NE, NM)); eauc = rng.normal(size=(NE, ND)); eauc[rng.random(eauc.shape) < .3] = np.nan
np.savez(f'{root}/extra-supervision-20261008-0001/resources/m_gdsc_extra19.npz', m=em, auc=eauc, drug_ids=did, tissues=rng.choice(['A', 'B'], NE),
         sample_ids=np.array([f'E{i}' for i in range(NE)]), m_feature_ids=np.array([f'M{i}' for i in range(NM)]))
for sc in ('new_cell', 'tissue'):
    for f in range(5):
        held = np.flatnonzero(np.arange(N) % 5 == f); fit = np.setdiff1d(np.arange(N), held)
        panel = np.arange(ND) if sc == 'new_cell' else np.arange(ND - (f % 2))
        banks = {'outer': (fit, held)}
        for j in range(3):
            q = fit[np.arange(len(fit)) % 3 == j]; banks[f'inner-{j}'] = (np.setdiff1d(fit, q), q)
        total = {}; count = {}; outer = {}
        for bank, (bf, bq) in banks.items():
            d = f'{root}/extra-supervision-20261008-0001/output/{sc}/fold-{f}/{bank}'; os.makedirs(d)
            ei = np.arange(NE) if sc == 'new_cell' else np.arange(NE - 2)
            st = C.kernel_state(m[bf]); mm = np.vstack([m[bf], em[ei]])
            K = C.rbf(mm, mm, st); H = C.rbf(m[bq], mm, st)
            np.savez(f'{d}/geometry.npz', fit=bf, query=bq, panel=panel, extra_indices=ei, m_only_kernel=K, m_only_cross=H,
                     L_protein=np.zeros_like(K))
            n = len(bf)
            g0 = C.predict_grid(K[:n, :n], H[:, :n], auc[np.ix_(bf, panel)])
            g1 = C.predict_grid(K, H, np.vstack([auc[np.ix_(bf, panel)], eauc[np.ix_(ei, panel)]]))
            np.savez(f'{d}/grids.npz', original__m_only=g0, extra__m_only=g1)
            if bank == 'outer':
                outer = {'original__m_only': g0, 'extra__m_only': g1}
            else:
                truth = auc[np.ix_(bq, panel)]; sd = np.maximum(np.nanstd(auc[np.ix_(bf, panel)], axis=0), 1e-8)
                for k, g in (('original__m_only', g0), ('extra__m_only', g1)):
                    ll, nn = C.bank_losses(g, truth, sd); total[k] = total.get(k, 0) + ll; count[k] = count.get(k, 0) + nn
        truth = auc[np.ix_(held, panel)]; sd = np.maximum(np.nanstd(auc[np.ix_(fit, panel)], axis=0), 1e-8)
        use = np.isfinite(truth).sum(0) >= 3
        preds = {k: C.take(outer[k], C.choose(total[k], count[k], 1)) for k in outer}
        np.savez(f'{root}/extra-supervision-20261008-0001/output/{sc}/fold-{f}/predictions.npz', fit=fit, held=held, panel=panel,
                 drug_ids=did[panel], truth=truth, train_sd=sd, evaluated=use, **preds)
print('fake root ready', root)

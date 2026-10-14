"""Development test on INNER banks only (no outer scoring): timing + grid reproduction vs saved inner grids."""
import sys
import time
import numpy as np

sys.path.insert(0, sys.argv[1])
import rna_lc as r  # noqa: E402

c = r.load_ctx()
for scenario, f, bank in ((sys.argv[2], int(sys.argv[3]), sys.argv[4]),):
    t = time.perf_counter()
    grids, meta, teacher = r.step1_bank(c, scenario, f, bank)
    t1 = time.perf_counter() - t
    saved = np.load(f'{r.EXT}/{scenario}/fold-{f}/{bank}/grids.npz')
    print('step1 seconds', round(t1, 1), {k: v.shape for k, v in grids.items()}, 'teacher', teacher.shape)
    print('m_f100 vs saved original__m_only grid', float(np.max(np.abs(grids['m_f100'] - saved['original__m_only']))))
    t = time.perf_counter()
    g2, yp = r.step2_bank(c, scenario, f, bank, meta, teacher, np.ones(len(meta['panel']), int))
    print('step2 seconds', round(time.perf_counter() - t, 1), {k: v.shape for k, v in g2.items()})
    print('m_extra19 vs saved extra__m_only grid', float(np.max(np.abs(g2['m_extra19'] - saved['extra__m_only']))))
    print('mP_extra19 vs saved extra__m_only__protein grid', float(np.max(np.abs(g2['mP_extra19'] - saved['extra__m_only__protein']))))
    print('m_pseudo w0 block == base', float(np.max(np.abs(g2['m_pseudo'][:4] - g2['m_extra19']))))
    print('mP_pseudo w0 block == mP_extra19', float(np.max(np.abs(g2['mP_pseudo'][:24] - g2['mP_extra19']))))
    print('rnat w0 == rna_c', float(np.max(np.abs(grids['rnat_c_extra'][:8] - grids['rna_c_extra']))))
    print('yp range', float(np.min(yp)), float(np.max(yp)), 'n pseudo', yp.shape)

"""Minimal closed loop: reproduce saved new-cell scores, then score a per-drug training-mean null model.

Independent re-implementation of the saved metric (does not import project code).
Run on the server via stdin; reads only saved run outputs, the 421 source table and the 19-sample extra table.
Writes nothing; prints summary numbers only.
"""
import csv
import json
import sys
import numpy as np

RUN = '/public/home/mengxl/creed-research-runs/extra-supervision-20261008-0001'
SCENARIO = sys.argv[1] if len(sys.argv) > 1 else 'new_cell'
OUT = RUN + '/output/' + SCENARIO
SOURCE = '/public/home/mengxl/creed-research-runs/gdsc-metabolite-full-protein-20260908-0001/output/result/source-inputs.npz'
EXTRA = RUN + '/resources/m_gdsc_extra19.npz'
ARMS = {
    'original__m_only': 'M 直接基线（原数据）',
    'extra__m_only': 'M 直接基线（加数据）',
    'original__m_only__protein': '旧 P 图（原数据）',
    'extra__m_only__protein': '当前候选（加数据 P 图）',
} if SCENARIO == 'new_cell' else {
    'original__m_only': 'M 直接基线（原数据）',
    'original__virtual_fm': 'FM 直接基线（原数据）',
    'original__virtual_fm__protein': '当前候选 FM+P 图（原数据）',
    'extra__virtual_fm': 'FM 直接基线（加数据）',
    'extra__virtual_fm__protein': 'FM+P 图（加数据）',
}


def fold_score(pred, truth, sd, use):
    """sqrt of the mean, over evaluated drugs, of per-drug mean squared standardized error."""
    obs = np.isfinite(truth)
    err = np.where(obs, ((pred - truth) / sd) ** 2, 0.0)
    per_drug = err.sum(0) / np.maximum(obs.sum(0), 1)
    return float(np.sqrt(per_drug[use].mean())), per_drug


src = np.load(SOURCE, allow_pickle=False)
auc = src['auc']
ext = np.load(EXTRA, allow_pickle=False)
assert (ext['drug_ids'] == src['drug_ids']).all()
reported = json.load(open(OUT + '/summary.json'))['summary']
saved_fold = {}
with open(OUT + '/fold_metrics.csv') as fh:
    for r in csv.DictReader(fh):
        saved_fold[(int(r['fold']), r['arm'])] = float(r['score'])

mine = {a: [] for a in ARMS}
null = {'null_original': [], 'null_extra': []}
per_drug = {k: [] for k in list(ARMS) + list(null)}
max_fold_diff = 0.0
sanity = []
for f in range(5):
    p = np.load(f'{OUT}/fold-{f}/predictions.npz', allow_pickle=False)
    rec = json.load(open(f'{OUT}/fold-{f}/outer/record.json'))
    fit, held, panel = p['fit'], p['held'], p['panel']
    truth, sd, use = p['truth'], p['train_sd'], p['evaluated']

    # Sanity: saved truth / train_sd agree with the source table.
    src_truth = auc[np.ix_(held, panel)]
    same_truth = np.array_equal(np.isnan(truth), np.isnan(src_truth)) and np.allclose(
        np.nan_to_num(truth), np.nan_to_num(src_truth), atol=0, rtol=0)
    fit_y = auc[np.ix_(fit, panel)]
    sd_check = np.maximum(np.nanstd(fit_y, axis=0), 1e-8)
    sanity.append((f, same_truth, float(np.max(np.abs(sd_check - sd)))))

    for a in ARMS:
        s, d = fold_score(p[a], truth, sd, use)
        mine[a].append(s)
        per_drug[a].append(d[use])
        max_fold_diff = max(max_fold_diff, abs(s - saved_fold[(f, a)]))

    # Null models: per-drug mean of observed training labels.
    mu0 = np.nanmean(fit_y, axis=0)
    ids = list(rec['extra_sample_ids'])
    rows = [int(np.flatnonzero(ext['sample_ids'] == i)[0]) for i in ids]
    mu1 = np.nanmean(np.vstack([fit_y, ext['auc'][np.ix_(rows, panel)]]), axis=0)
    for key, mu in (('null_original', mu0), ('null_extra', mu1)):
        s, d = fold_score(np.broadcast_to(mu, truth.shape), truth, sd, use)
        null[key].append(s)
        per_drug[key].append(d[use])

print('== 健全性检查（逐折：真值与源表一致？train_sd 与源表重算的最大差）')
for f, ok, dsd in sanity:
    print(f'  fold {f}: truth_match={ok}  max|sd差|={dsd:.3e}')

print('\n== 复现检查（我的实现 vs 保存值）')
print(f'  逐折最大绝对差: {max_fold_diff:.3e}')
for a, name in ARMS.items():
    m = float(np.mean(mine[a]))
    print(f'  {name:<16} 我的={m:.9f}  保存={reported[a]["score"]:.9f}  差={abs(m - reported[a]["score"]):.3e}')

print('\n== 零模型（逐药训练均值）')
for key, name in (('null_original', '零模型（原数据）'), ('null_extra', '零模型（加数据）')):
    print(f'  {name}: 5折均值={np.mean(null[key]):.9f}  逐折={[round(x, 6) for x in null[key]]}')

print('\n== 相对零模型（同数据比较；改善% = (零模型-模型)/零模型）')
pairs = [(a, 'null_extra' if a.startswith('extra__') else 'null_original') for a in ARMS]
for a, nk in pairs:
    mm, nn = np.array(mine[a]), np.array(null[nk])
    imp = (nn.mean() - mm.mean()) / nn.mean() * 100
    wins = int((mm < nn).sum())
    print(f'  {ARMS[a]:<16} 模型={mm.mean():.6f}  零模型={nn.mean():.6f}  改善={imp:.4f}%  优于零模型的折数={wins}/5')
    print(f'      逐折改善%={[round(x, 4) for x in ((nn - mm) / nn * 100)]}')

print('\n== 附加描述（未预先登记）：逐药技能分数 1 - MSE_model/MSE_null，5折合并')
for a, nk in pairs:
    dm = np.concatenate(per_drug[a]); dn = np.concatenate(per_drug[nk])
    skill = 1 - dm / dn
    print(f'  {ARMS[a]:<16} 药-折数={len(skill)}  技能>0占比={np.mean(skill > 0) * 100:.1f}%  '
          f'中位数={np.median(skill):.4f}  均值={np.mean(skill):.4f}  >0.05占比={np.mean(skill > 0.05) * 100:.1f}%')

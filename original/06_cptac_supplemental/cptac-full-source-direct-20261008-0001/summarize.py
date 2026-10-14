"""Summarize the fixed comparator run; never fits or changes predictions."""
import csv
import json
from pathlib import Path
from statistics import mean

root = Path(__file__).resolve().parent
d = json.loads((root / 'summary.json').read_text())
old = json.loads((root / 'old_summary.json').read_text())
assert d['status'] == 'completed' and len(d['records']) == 10
lookup = {(r['direction'], r['split']): r for r in old['records']}
rows = []
for r in d['records']:
    o = lookup[r['direction'], r['split']]
    for arm, metrics in r['scores'].items():
        rows.append(dict(direction=r['direction'], split=r['split'], arm=arm,
                         alpha=r['alpha'] if arm == 'full_source_linear' else '',
                         full_source_columns=r['metadata']['source_columns'],
                         identity_modelled_targets=o['same_gene_modelled_targets'],
                         original_targets=o['targets'],
                         identity_coverage=o['same_gene_modelled_targets']/o['targets'],
                         **metrics))
with (root / 'fold_metrics.csv').open('w') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

lines = ['# CPTAC 同完整源面板线性 direct：固定一次公平比较', '',
         '日期：2026-10-08。状态：原定10个outer fits全部完成。旧预测、模型和主图均未修改。', '',
         '**范围：旧BRCA/LUAD共232人的mixed患者五折，两个方向；不是现Figure5的327人、三个队列迁移任务。** 后者已做过source-only shift与expert native target scale，本次没有重新评价、替换或追加该分支。简单源校正不是新的待提算法。', '',
         '## 主要结果', '',
         '数值为五折标准化RMSE等权均值，越低越好。比较单位、目标、患者完全沿用原outer折。', '',
         '| 方向 | 旧PCA16 direct | 旧anchor+context | 新full-source linear direct | anchor相对新direct改善 | anchor胜出折数 |',
         '|---|---:|---:|---:|---:|---:|']
results = {}
for direction in ['rna_to_protein', 'protein_to_rna']:
    m = d['means'][direction]
    records = [r for r in d['records'] if r['direction'] == direction]
    wins = sum(r['scores']['anchor_context']['standardized_rmse'] < r['scores']['full_source_linear']['standardized_rmse'] for r in records)
    delta = 100*(1-m['anchor_context']/m['full_source_linear'])
    closed = 100*(m['masked_pca_ridge']-m['full_source_linear'])/(m['masked_pca_ridge']-m['anchor_context'])
    results[direction] = dict(anchor_relative_improvement_vs_full_source_percent=delta,
                              anchor_wins=wins, old_pca_gap_closed_by_direct_percent=closed,
                              full_source_mean_target_r=mean(r['scores']['full_source_linear']['mean_target_r'] for r in records),
                              anchor_mean_target_r=mean(r['scores']['anchor_context']['mean_target_r'] for r in records))
    lines.append(f"| {direction} | {m['masked_pca_ridge']:.6f} | {m['anchor_context']:.6f} | {m['full_source_linear']:.6f} | {delta:+.3f}% | {wins}/5 |")
lines += ['', '## 全部逐折数值', '',
          '| 方向 | 折 | 旧anchor+context | full-source linear | 新direct−anchor | alpha | 同基因anchor覆盖 |',
          '|---|---:|---:|---:|---:|---:|---:|']
for r in d['records']:
    o = lookup[r['direction'], r['split']]
    a = r['scores']['anchor_context']['standardized_rmse']
    f = r['scores']['full_source_linear']['standardized_rmse']
    lines.append(f"| {r['direction']} | {r['split'].split('_')[-1]} | {a:.9f} | {f:.9f} | {f-a:+.9f} | {r['alpha']} | {o['same_gene_modelled_targets']}/{o['targets']} ({100*o['same_gene_modelled_targets']/o['targets']:.3f}%) |")
lines += ['', '## 能保留什么，不能保留什么', '']
for direction, x in results.items():
    lines.append(f"- **{direction}**：同完整源面板的简单direct解释掉旧PCA→anchor差距的{x['old_pca_gap_closed_by_direct_percent']:.2f}%；anchor对新direct相对RMSE改善{x['anchor_relative_improvement_vs_full_source_percent']:+.3f}%，胜{x['anchor_wins']}/5折。平均每目标Pearson，新direct {x['full_source_mean_target_r']:.6f}，anchor {x['anchor_mean_target_r']:.6f}。")
lines += [
    '- 旧7.96%/4.20%是相对top1000→PCA16 baseline，不能继续当作相对公平全源direct的专属双向优势。此次比较不证明全局SOTA；它只排除一个明确的输入覆盖/瓶颈混淆。',
    '- 上述“旧差距解释比例”只是两条RMSE差值之比，不能当作输入覆盖贡献的因果百分比；新direct同时取消旧PCA瓶颈并使用预定inner选alpha。此次没有逐项干预它们。',
    '- 两方向身份锚覆盖不对称：RNA→蛋白约95.2%，蛋白→RNA约53.0%。这是原summary的`same_gene_modelled_targets`，不是猜测。不能把两方向差异直接解释为生物学因果方向；是否由覆盖率导致，当前比较不能独立识别。',
    '- 可以保留RNA→蛋白上的定向增益（以实际表格为准）与identity+context的设计经验；不应再扩写为双向都胜、功能层已经保真或新癌种泛化成立。没有更改目标/划分、选择药物、补新预训练模型或进行测试侧校准。', '',
    '## 固定实现与公平性', '',
    '- 输入采用相同原始source panel全部映射坐标，训练中至少5个观测且方差非零；不先压到top1000/PCA16。用训练中位数填缺失、训练均值/标准差标准化，附显式missing indicators。',
    '- 线性ridge按目标的真实可见训练标签拟合，独立free intercept；缺失目标不填零。通过observed-mask分组进行完全等价的线性核求解，核除以训练平均对角尺度。',
    '- 仅alpha=(0.1, 1, 10, 100)，每个方向/outer折用3个预固定inner患者折选择一个alpha。所有inner变换只拟合inner训练患者；全部原outer目标支持与原保存尺度保持一致。query truth逐值核对至1e-12。',
    '- 所有10折均选择0.1（如配置及逐折表所示）。这是预定搜索边界，未因结果追加更小alpha、重新划分或更换方法；不能据此宣称ridge已充分调优。',
    '- 求解器以独立primal ridge方程在含输入/标签缺失的玩具数据上校验，全部4个alpha最大差5.996e-15；见SOLVER_CHECK.json。',
    f"- 运行资源2 CPU、6 GiB内存上限、无网络容器；10个原定outer fits及其inner选择总用时{d['elapsed_seconds']:.1f}秒。原始数据、旧源码和旧预测只读挂载。", '',
    '## 单位与外推边界', '',
    '原manifest中BRCA RNA是log2 FPKM（源零值先换NA），LUAD是log2 normalized RPKM；蛋白均为log2 MS丰度比，但现成资料未证明其参考分母相同。本次仅沿用旧mixed任务做同输入比较，没有把不同癌种重新定标。跨癌种量值失败仍混杂测量兼容性与生物差异，不能直接冠以新的科学难题。现Figure5分支已实施过源校正和native-scale，其结果不在本次范围。', '',
    '## 可复核产物', '',
    '- `config.json`与`run.py`在fit前保存；`summary.json`含inner患者索引、候选损失、选择alpha、全部指标。',
    '- `fold_metrics.csv`为10折×4臂共40行；`old_summary.json`为只读拷贝，含原anchor覆盖。',
    '- `run.log`为实际运行日志。服务器新预测位于`/public/home/mengxl/creed-research-runs/cptac-full-source-direct-20261008-0001/output/`；未覆盖旧run。',
    '- `summarize.py`仅读已有summary，不进行任何训练。',
]
(root/'RESULT.md').write_text('\n'.join(lines)+'\n')
(root/'comparison.json').write_text(json.dumps(results, indent=2)+'\n')
print(json.dumps(dict(means=d['means'],comparison=results,elapsed_seconds=d['elapsed_seconds']),indent=2))

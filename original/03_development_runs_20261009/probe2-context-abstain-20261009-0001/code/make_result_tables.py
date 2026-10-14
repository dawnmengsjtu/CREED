"""Render the appendix tables of RESULT.md from summary.json (stdlib only; no new statistics).

usage: python3 make_result_tables.py summary.json > result_tables.md
"""
import json
import sys

S = json.load(open(sys.argv[1]))
SCEN = ('new_cell', 'tissue'); CN = {'new_cell': '新细胞', 'tissue': '整组织'}
COV = ('1.0', '0.9', '0.8', '0.7', '0.5')
L = []


def arm_input(a):
    if a in ('m_only', 'repro_extra_m_only'):
        return '仅 M' if a == 'm_only' else 'M（+19 样本，复现）'
    if a == 'm_allcand':
        return 'M+全候选约 1800 蛋白（k=all）'
    if a == 'teacher_fullP_saved':
        return 'M+全 P 约 3350 蛋白（参照）'
    if a == 'null_drug':
        return '无'
    if a.startswith('tissue_adaptive') or a.startswith('matched_random_tissue'):
        return 'M+组织标签+k 蛋白'
    return 'M+k 蛋白'


for sc in SCEN:
    A = S['A'][sc]; summ = A['summary']; rec = A['recovery']
    L.append(f'\n### A-{CN[sc]}：全部臂（5 折均值 sRMSE，按分数排序；R = 收回比例）\n')
    L.append('| 臂 | 部署输入 | sRMSE | Spearman | cMSE | fold0 | fold1 | fold2 | fold3 | fold4 | R(k=all) | R(全P) |')
    L.append('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for a in sorted(summ, key=lambda x: summ[x]['score']):
        if (a.startswith('matched_random') or a.startswith('random_global')) and not a.endswith('_mean'):
            continue
        r = rec.get(a)
        rs = f"{r['vs_allcand']:.3f} | {r['vs_fullP']:.3f}" if r else '— | —'
        L.append(f"| {a} | {arm_input(a)} | {summ[a]['score']:.6f} | {summ[a]['spearman']:.4f} | {summ[a]['centered_mse']:.4f} | "
                 + ' | '.join(f'{x:.4f}' for x in summ[a]['folds']) + f' | {rs} |')
    L.append(f'\n### A-{CN[sc]}：随机臂各种子\n')
    L.append('| 臂 | sRMSE | fold0 | fold1 | fold2 | fold3 | fold4 |'); L.append('|---|---:|---:|---:|---:|---:|---:|')
    for a in sorted(summ):
        if (a.startswith('matched_random') or a.startswith('random_global')) and not a.endswith('_mean'):
            L.append(f"| {a} | {summ[a]['score']:.6f} | " + ' | '.join(f'{x:.4f}' for x in summ[a]['folds']) + ' |')
    L.append(f'\n### A-{CN[sc]}：全部比较（NB 校正配对 t，df=4，r={A["nb_ratio"]:.3f}）\n')
    L.append('| 比较 | 新臂均值 | 基线均值 | 改善% | 胜折 | NB p | 95% 区间(%) | 逐折改善% |'); L.append('|---|---:|---:|---:|---|---:|---|---|')
    for name, c in A['comparisons'].items():
        L.append(f"| {name} | {c['new_mean']:.6f} | {c['base_mean']:.6f} | {c['improvement_pct']:+.3f} | {c['folds_won']}/5 | {c['nb_p']:.3f} | "
                 f"[{c['ci95_pct'][0]:.2f}, {c['ci95_pct'][1]:.2f}] | {', '.join(f'{x:+.2f}' for x in c['per_fold_pct'])} |")
    L.append(f'\n### A-{CN[sc]}：外折上下文诊断\n')
    L.append('| 折 | 候选数 | 头数 | 簇大小(fit) | 簇大小(query) | 簇面板 vs 全局 Jaccard@16 | 组织：有面板的组织数 / 回退全局的 query 数 | 组织面板 vs 全局 Jaccard@16 |')
    L.append('|---|---:|---:|---|---|---|---|---|')
    for f in range(5):
        d = A['diagnostics'][f'fold-{f}']
        cj = ', '.join(f'{k}:{v:.2f}' for k, v in (d['cluster_jaccard_vs_global_k16'] or {}).items())
        if 'tissue_eligible' in d:
            tj = ', '.join(f'{k[:6]}:{v:.2f}' for k, v in (d['tissue_jaccard_vs_global_k16'] or {}).items())
            ts = f"{len(d['tissue_eligible'])} / {d['tissue_query_to_global']}"
        else:
            tj, ts = '—', '—'
        L.append(f"| {f} | {d['candidates']} | {d['heads']} | {d['cluster_sizes']} | {d['cluster_query_sizes']} | {cj} | {ts} | {tj} |")
for sc in SCEN:
    B = S['B'][sc]; T = B['table']
    L.append(f'\n### B-{CN[sc]}：风险–覆盖（保留集 sRMSE 的 5 折均值）与 AURC\n')
    L.append('| 臂 | 代理 | c=1.0 | c=0.9 | c=0.8 | c=0.7 | c=0.5 | AURC |'); L.append('|---|---|---:|---:|---:|---:|---:|---:|')
    for key, v in T.items():
        arm, proxy = key.split('|')
        L.append(f'| {arm} | {proxy} | ' + ' | '.join(f"{v['coverage'][c]:.4f}" for c in COV) + f" | {v['aurc']:.5f} |")
    L.append(f'\n### B-{CN[sc]}：代理 vs 随机（AURC）、覆盖 0.8 条件、增益–覆盖（r={B["nb_ratio"]:.3f}）\n')
    L.append('| 臂 | 代理 | AURC 改善% | 胜折 | NB p | 逐折 AURC 改善% | c=0.8 相对降低% | B1 | B2 | 增益(全集) | 增益(c=0.8) | 保留集增益更大 |')
    L.append('|---|---|---:|---|---:|---|---:|---|---|---:|---:|---|')
    for key, c in B['comparisons'].items():
        arm, rest = key.split('|'); proxy = rest[:-10]
        g1 = f"{c['gain_full']:+.4f}" if 'gain_full' in c else '—'; g8 = f"{c['gain_cov08']:+.4f}" if 'gain_cov08' in c else '—'
        gl = f"{c['gain_larger_on_retained']}（{c['gain_larger_folds']}/5 折）" if 'gain_larger_on_retained' in c else '—'
        L.append(f"| {arm} | {proxy} | {c['improvement_pct']:+.3f} | {c['folds_won']}/5 | {c['nb_p']:.3f} | {', '.join(f'{x:+.2f}' for x in c['per_fold_pct'])} | "
                 f"{c['cov08_rel_reduction_pct']:+.3f} | {'满足' if c['B1'] else '不满足'} | {'满足' if c['B2'] else '不满足'} | {g1} | {g8} | {gl} |")
    L.append(f'\n### B-{CN[sc]}：代理与逐细胞误差的 Spearman（逐折均值；描述性）\n')
    L.append('| 代理 | ' + ' | '.join(S['B'][sc]['table'] and ['M', 'M+vRNA', 'M+vP', 'M+RNA', 'M+P']) + ' |'); L.append('|---|---:|---:|---:|---:|---:|')
    for proxy in ('u1_vRNA', 'u1_vP', 'u2', 'u3'):
        L.append(f'| {proxy} | ' + ' | '.join(f"{B['proxy_error_spearman'][f'{proxy}|{a}']['mean']:+.3f}" for a in ('M', 'M+vRNA', 'M+vP', 'M+RNA', 'M+P')) + ' |')
    L.append(f'\n### B-{CN[sc]}：生成器重拟合核对\n')
    L.append('| bank/对 | λ（我的 / 保存） | OOF R² 均值（我的 / 保存） |'); L.append('|---|---|---|')
    for k, v in B['generator_check'].items():
        L.append(f"| {k} | {v['lam']} / {v['lam_saved']} | {v['r2']:.6f} / {v['r2_saved']:.6f} |")
print('\n'.join(L))

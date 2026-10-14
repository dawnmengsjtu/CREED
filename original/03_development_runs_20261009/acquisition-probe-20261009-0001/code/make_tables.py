"""Render Markdown tables for RESULT.md from the analysis summary.json / bank_info (formatting only, no new statistics)."""
import json
A = json.load(open('summary.json'))
SC_NAME = {'new_cell': '新细胞', 'tissue': '整组织'}
HEAD_NAME = {'H_meas': 'H_meas（实测头，主头）', 'H_virt': 'H_virt（虚拟头，次头）', 'none': '参照'}
POL_ORDER = ['m_only', 'null_drug', 'teacher_saved', 'virtual', 'measured_all', 'random_mean', 'random_s0', 'random_s1', 'random_s2', 'random_s3', 'random_s4',
             'variance', 'static', 'adaptive', 'adaptive_surprise', 'oracle_static']
POL_NAME = {'m_only': 'm_only（仅 M，复现 original__m_only）', 'null_drug': 'null_drug（逐药 fit 均值）', 'teacher_saved': 'teacher_saved（全 P 教师约 3350 蛋白，保存值）',
            'virtual': 'k=0 全虚拟', 'measured_all': 'k=all 全实测', 'random_mean': 'random（5 种子均值）', 'variance': 'variance', 'static': 'static',
            'adaptive': '**adaptive**', 'adaptive_surprise': 'adaptive_surprise', 'oracle_static': 'oracle_static（泄漏参照）'}
out = []
P = out.append
P('### 附表 A：复现检查\n')
P('| 项目 | 新细胞 | 整组织 |\n|---|---:|---:|')
r1, r2 = A['new_cell']['reproduction'], A['tissue']['reproduction']
P(f"| original__m_only 内外折网格最大差 | {r1['grid_max_abs_diff']['original__m_only']:.1e} | {r2['grid_max_abs_diff']['original__m_only']:.1e} |")
P(f"| extra__m_only 网格最大差（含额外 19 行） | {r1['grid_max_abs_diff']['extra__m_only']:.1e} | — |")
P(f"| original__m_only 外折选择后预测最大差 | {r1['pred_max_abs_diff']['original__m_only']:.1e} | {r2['pred_max_abs_diff']['original__m_only']:.1e} |")
P(f"| H_meas k=all 折分数 vs protein-panel m_allcand | {r1['H_meas_kall_vs_protein_panel_m_allcand_fold_score_max_abs_diff']:.1e} | {r2['H_meas_kall_vs_protein_panel_m_allcand_fold_score_max_abs_diff']:.1e} |")
P('')
for sc in ('new_cell', 'tissue'):
    S = A[sc]; T = S['table']
    P(f'### 附表 B-{SC_NAME[sc]}：全部臂、全部折（sRMSE；5 折均值 / Spearman / centered MSE / fold0–4）\n')
    P('| 头 | 策略 | k | sRMSE | Spearman | centered MSE | fold0 | fold1 | fold2 | fold3 | fold4 |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    keys = []
    for h in ('none', 'H_meas', 'H_virt'):
        for pol in POL_ORDER:
            for k in ('M', 'none', 'full_P', '0', 'all', '8', '16', '32', '64'):
                key = f'{h}|{pol}|{k}'
                if key in T:
                    keys.append((h, pol, k, key))
    for h, pol, k, key in keys:
        v = T[key]; sp = '—' if v['spearman'] != v['spearman'] else f"{v['spearman']:.4f}"
        P(f"| {h} | {POL_NAME.get(pol, pol)} | {k} | {v['score']:.6f} | {sp} | {v['centered_mse']:.4f} | " + ' | '.join(f'{x:.4f}' for x in v['fold_scores']) + ' |')
    P('')
    P(f'### 附表 C-{SC_NAME[sc]}：收回曲线（sRMSE 5 折均值 / 收回比例 R；教师 = H_meas k=all = {S["teacher_srmse"]:.6f}）\n')
    P('| 头 | 策略 | k=0 | k=8 | k=16 | k=32 | k=64 | k=all | 单调非增 |\n|---|---|---:|---:|---:|---:|---:|---:|---|')
    for h in ('H_meas', 'H_virt'):
        for pol in ('random_mean', 'variance', 'static', 'adaptive', 'adaptive_surprise', 'oracle_static'):
            c = S['curves'][h][pol]
            cells = [f"{s:.5f} / {r:+.3f}" for s, r in zip(c['srmse'], c['recovery'])]
            P(f"| {h} | {POL_NAME[pol]} | " + ' | '.join(cells) + f" | {'是' if c['monotone_nonincreasing_srmse'] else '否'} |")
    P('')
    P(f'### 附表 D-{SC_NAME[sc]}：主比较（C1–C3）与次要比较（Nadeau–Bengio 校正配对 t，df=4，r={S["nb_ratio_test_train"]:.3f}）\n')
    P('| 比较 | 头 | k | 新臂 | 基线 | 改善% | 胜折 | 逐折改善% | t | p | 95%CI(%) |\n|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|')
    order = ['adaptive_vs_static', 'adaptive_vs_random_mean', 'adaptive_vs_variance', 'adaptive_surprise_vs_static', 'adaptive_surprise_vs_random_mean',
             'static_vs_random_mean', 'oracle_static_vs_static', 'adaptive_vs_oracle_static', 'adaptive_vs_m_only', 'static_vs_m_only']
    tag = {'adaptive_vs_static': 'C1 adaptive vs static', 'adaptive_vs_random_mean': 'C2 adaptive vs random_mean', 'adaptive_vs_variance': 'C3 adaptive vs variance',
           'adaptive_surprise_vs_static': '次 surprise vs static', 'adaptive_surprise_vs_random_mean': '次 surprise vs random_mean', 'static_vs_random_mean': '次 static vs random_mean',
           'oracle_static_vs_static': '次 oracle_static vs static', 'adaptive_vs_oracle_static': '次 adaptive vs oracle_static', 'adaptive_vs_m_only': '次 adaptive vs m_only', 'static_vs_m_only': '次 static vs m_only'}
    for h in ('H_meas', 'H_virt'):
        for k in (16, 32):
            for o in order:
                c = S['comparisons'][f'{h}|k{k}|{o}']
                ci = '—' if c['nb_ci95_pct'] is None else f"[{c['nb_ci95_pct'][0]:+.3f}, {c['nb_ci95_pct'][1]:+.3f}]"
                P(f"| {tag[o]} | {h} | {k} | {c['new_mean']:.6f} | {c['base_mean']:.6f} | {c['improvement_pct']:+.3f} | {c['folds_won']}/5 | " +
                  ', '.join(f'{x:+.2f}' for x in c['per_fold_improvement_pct']) + f" | {c['nb_t']:+.2f} | {c['nb_p']:.3f} | {ci} |")
    for key, name in (('H_virt|k0|virtual_vs_m_only', '两端 H_virt k=0 vs m_only'), ('H_meas|kall|teacher_vs_m_only', '两端 教师(H_meas k=all) vs m_only'),
                      ('H_virt|kall|measured_into_virtual_head_vs_H_meas_kall', '两端 H_virt k=all vs H_meas k=all'), ('H_meas|k0|virtual_into_measured_head_vs_H_virt_k0', '两端 H_meas k=0 vs H_virt k=0')):
        c = S['comparisons'][key]; ci = '—' if c['nb_ci95_pct'] is None else f"[{c['nb_ci95_pct'][0]:+.3f}, {c['nb_ci95_pct'][1]:+.3f}]"
        P(f"| {name} | — | — | {c['new_mean']:.6f} | {c['base_mean']:.6f} | {c['improvement_pct']:+.3f} | {c['folds_won']}/5 | " +
          ', '.join(f'{x:+.2f}' for x in c['per_fold_improvement_pct']) + f" | {c['nb_t']:+.2f} | {c['nb_p']:.3f} | {ci} |")
    P('')
    P(f'### 附表 E-{SC_NAME[sc]}：预先登记的通过条件逐项\n')
    P('| 头 | 条件 | k=16 | k=32 | 满足 |\n|---|---|---|---|---|')
    for h in ('H_meas', 'H_virt'):
        pc = S['pass_conditions'][h]
        for c, nm in (('cond1_adaptive_beats_static', '1 adaptive 优于 static（改善>0，≥4/5 折，p<0.10）'), ('cond2_adaptive_beats_random_mean', '2 adaptive 优于 random_mean（同上）')):
            d = pc[c]['detail']
            cells = [f"改善 {d[kk]['improvement_pct']:+.3f}%，{d[kk]['folds_won']}/5，p={d[kk]['nb_p']:.3f} → {'满足' if d[kk]['satisfied'] else '不满足'}" for kk in ('k16', 'k32')]
            P(f"| {h} | {nm} | {cells[0]} | {cells[1]} | {'是' if pc[c]['satisfied'] else '否'} |")
        m = pc['cond3_adaptive_recovery_monotone']
        P(f"| {h} | 3 adaptive 收回曲线单调非减 | sRMSE 序列 {', '.join(f'{x:.5f}' for x in m['srmse_curve'])} | | {'是' if m['satisfied'] else '否'} |")
        P(f"| {h} | **全部满足** | | | **{'是' if pc['all_satisfied'] else '否'}** |")
    P('')
    P(f'### 附表 F-{SC_NAME[sc]}：随机面板 5 种子范围（5 折均值 sRMSE）\n')
    P('| 头 | k | 均值 | 最小 | 最大 | 各种子 |\n|---|---:|---:|---:|---:|---|')
    for key, v in S['random_range'].items():
        h, k = key.split('|')
        P(f"| {h} | {k[1:]} | {v['mean']:.6f} | {v['min']:.6f} | {v['max']:.6f} | {', '.join(f'{x:.5f}' for x in v['per_seed'])} |")
    P('')
    b = json.load(open(f'bank_info_{sc}.json'))
    P(f'### 附表 G-{SC_NAME[sc]}：外折诊断（生成器、头、策略重叠）\n')
    P('| 折 | 候选数 | 生成器 alpha | 虚拟 P 的 OOF R² | 查询虚拟 P 的 MSE（标准化） | u 均值/中位 | corr(w, 平均 u) | static∩oracle 前16/前64 | adaptive k16：每细胞与 static 前16/前64 重叠 | adaptive k16 覆盖蛋白数 | surprise k16 与 static 前16 重叠 | H_meas alpha 计数(0.1,1,10,100) | H_virt alpha 计数 |\n|---:|---:|---:|---:|---:|---|---:|---|---|---:|---:|---|---|')
    for f in range(5):
        fo, o = b[f'fold-{f}'], b[f'fold-{f}/outer']
        P(f"| {f} | {o['candidate_count']} | {fo['generator_alpha']} | {o['virtual_oof_r2']:.3f} | {o['virtual_query_mse_observed']:.3f} | {o['u_q_mean']:.3f}/{o['u_q_median']:.3f} | {o['corr_w_meanU']:+.2f} | {o['static_oracle_overlap16']}/{o['static_oracle_overlap64']} | {o['adaptive_k16_mean_overlap_with_static16']:.2f}/{o['adaptive_k16_mean_overlap_with_static64']:.2f} | {o['adaptive_k16_distinct_proteins']} | {o['adaptive_surprise_k16_mean_overlap_with_static16']:.2f} | {fo['alpha_counts']['H_meas']} | {fo['alpha_counts']['H_virt']} |")
    P('')
    P(f'各折 static 前 8 蛋白：')
    for f in range(5):
        P(f"- fold{f}: " + ', '.join(x.split(';')[1].replace('_HUMAN', '') for x in b[f'fold-{f}/outer']['w_static_top16'][:8]) +
          '；adaptive k16 最常选：' + ', '.join(f"{x[0].split(';')[1].replace('_HUMAN', '')}({x[1]})" for x in b[f'fold-{f}/outer']['adaptive_k16_most_chosen'][:5]))
    P('')
open('tables.md', 'w').write('\n'.join(out))
print(len(out), 'lines')

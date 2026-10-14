# 原 608 之外全部身份安全 M+RNA 配对池：提取结果（generator-fullpool-20261009-0001）

日期：2026-10-09。计划见 [PLAN.md](../PLAN.md)（登记 2026-10-09T17:50:56+08:00，原 sha256 `c9234493…`，`SHA256SUMS.v1`；两次运行前程序修正 ADDENDUM-A/B 后 sha256 `78b08ed7…`→`48b94bb1…`，`SHA256SUMS.v2/.v3`，三个版本全部保留）。主运行脚本 `code/extract_fullpool.py` sha256 `6073a957…`，8 线程，32 s。本轮**只提取与计数，没有训练、没有外折分数**。

- 服务器：`/public/home/mengxl/creed-research-runs/generator-fullpool-20261009-0001/`（PLAN.md、PLAN_REGISTERED_AT.txt、SHA256SUMS.v1/.v2/.v3、code/、resources/、logs/（两次失败尝试的日志）、run.log、pack/{m_rna_fullpool.npz, qa.json, EXTRACTION.md}、OUTPUT_SHA256SUMS）。
- 本地：`/Users/xlzhexue/Desktop/proj/creed/generator_fullpool_20261009/`（同名文件 + pack/qa.json、pack/m_rna_fullpool.npz）。
- 只读了 PI 放行的 5 个原始文件（运行前断言 sha256 与验收报告一致）与计划列出的既有资源；mmc3 只读了 `Prot matrix excl single-peptide` 第一列的样本 ID。64/123 封存线：ID 只用于排除，数值字段从未被 `float()`（M CSV、RNA gz 均按身份前缀/表头做行列门控），没有写、没有打印。

## 0. 一句话结论

**原 608 之外、身份安全、同时有 M 与 RNA 的细胞系全集 = 67 条（新细胞场景可用 67；整组织留出场景各外折可用 52 / 45 / 53 / 51 / 51）。可行性阈值 150 不满足（67 < 150），生成器扩容按预设判定不可行。** 已审计的 66 条是这 67 条的严格子集，且 M、RNA 数值与 66 包逐元素完全一致（最大差 0）；唯一新增的一条 `SIDM01095`（RH30_SOFT_TISSUE）来自 mmc2 的一条多值 `BROAD_ID` 记录，身份有歧义标记（见 §2.4）。M∩P 全集：按 608 协议的 mmc2 口径为 609（608 之外只有 `SIDM00046`，它没有 RNA 列）；最宽松的桥接口径也只有 612——"608 基本就是整个 M∩P 全集"成立。

## 1. 硬门：全部通过

| 门 | 结果 | 判据 |
|---|---:|---|
| 5 条 421 训练线（SIDM00018, 00080, 00082, 00396, 00685；Sanger+Broad）M 向量 vs `source-inputs m` | 最大绝对差 **0**，NaN 模式相同 | ≤ 1e-9 |
| 同 5 条 RNA log-CPM vs `rna421/rna_log_cpm.npy` | 最大绝对差 **1.78e-15** | ≤ 1e-9 |
| 225 代谢物列顺序 == `source-inputs metabolite_ids` == `rna_config.m_feature_ids` | 相同 | 断言 |
| 源 37,279 个 gene_id == `all_source_gene_ids.npy`；冻结 34,985 索引上的 gene_id == `rna421/gene_ids.npy` | 相同 | 断言 |
| 421 ⊂ 608；421+64+123 的并集 == `excluded_old608` | 成立 | 断言 |
| 信息性：与 66 包重叠的 66 条，M / RNA 最大绝对差 | **0 / 0** | — |

变换：M 原值 float64，不取 log、不标准化、不插补；RNA = `ln(1 + counts / 该样本在冻结 34,985 基因面板上的总计数 × 1e6)`，存 float32（`rna_counts_common` 同存）。池内 M 缺失 0，RNA 面板缺失 0，67 条全部满足"有限、非负、正库总量"。

## 2. 身份桥与排除

### 2.1 身份表
- mmc2 `Cell line level sample info`：949 个 SIDM，942 有 `BROAD_ID`、942 有 `CCLE_ID`；3 条字段为分号多值（`SIDM01095`: ACH-000833;ACH-001741;ACH-001189；`SIDM00096`: ACH-000338 重复两次；`SIDM00400`: ACH-002392;ACH-002303）。拆分后没有一个 ACH / CCLE_ID 对应多个 SIDM。
- Model.csv：1,959 个 ModelID，1,216 有 `SangerModelID`；`SIDM00117` 对应两个 ACH（歧义，不用）。
- 桥：1,217 对 SIDM↔ACH（mmc2 942 + Model.csv 补充 275）。4 处 mmc2 与 Model.csv 冲突（`SIDM01095, SIDM00399, SIDM00603, SIDM00114`），全部 mmc2 胜。
- 组织：SIDM 在 mmc2 → `Tissue_type`；否则 `OncotreeLineage → Tissue_type` 经验映射（两表共有线学习，多数类 ≥ 90% 且 n ≥ 3，共 22 条映射，如 Bowel→Large Intestine、CNS/Brain→Central Nervous System、Pleura→Lung）；`Esophagus/Stomach`（57% / 43%）、`Uterus`（63%）不满足阈值 → 留空。池内组织来源：mmc2 3、lineage 映射 60、空 4。**与 66 包的 `tissues` 逐条相同（差异 0）**，空值恰为 `SIDM00015, SIDM00037, SIDM00248, SIDM00271`。

### 2.2 M 行解析（928 行）
- 桥接口径：732 行两条路径（DepMap_ID、CCLE_ID）一致解析，1 行只有 DepMap_ID 解析，195 行不可映射（不猜），0 冲突；733 个唯一 SIDM。无同一 SIDM 对应多条源行。
- 仅 mmc2 口径：整串处理（608 协议口径）609 个 SIDM，608 之外只有 `SIDM00046`；分号拆分 611（多 `SIDM00096`、`SIDM01095`）。608 ⊂ 两种口径。

### 2.3 候选与排除
| 步骤 | 数目 |
|---|---:|
| M 行解析出 SIDM 且 SIDM 在 RNA 表头（1,047 列） | 683 |
| 排除 SIDM ∈ 608 | −608 |
| 排除与 608 线同 ACH | 0 |
| 排除与 608 线同 PatientID | −8（`SIDM00093, 00147, 00228, 00452, 00593, 00897, 00979, 01022`） |
| 排除与 608 线同 CCLE 名 / 同规范化细胞名 | 0 |
| 池内同 ACH / 同 PatientID 去重 | 0 |
| **池** | **67** |

### 2.4 与已审计 66 条的关系
- 66 ⊂ 67；66 条的 ACH、PatientID、组织与 66 包完全一致；M / RNA 数值完全一致。
- 新增 1 条：`SIDM01095`（RH30_SOFT_TISSUE，M 行 DepMap_ID = ACH-000833，PatientID PT-lbYOig，Soft Tissue，Sanger RNA，在 mmc3 蛋白表头中）。它在 2026-10-08 审计中落选的原因可以复现：审计用 Model.csv 桥接，Model.csv 给 `SIDM01095` 的 ACH 是 ACH-001189，与 M 行的 ACH-000833 不符，故未配上；本轮 mmc2 的多值字段拆分后 ACH-000833 直接落到 `SIDM01095`。mmc2 把三个 DepMap 模型挂在同一个 Sanger 模型下，**DepMap 层面身份有歧义**。它不与任何 608 线共享 ACH / PatientID / 名字，所以按 PLAN 规则留在池内（`qa.json sample_table.in_audited66 = false` 可识别）；若下游要与审计口径严格一致，去掉它即为 66。

## 3. 池的构成
- RNA 批次：Broad 58 / Sanger 9（421 为 367 / 54）。
- 组织：Haem 17、Lung 11、Large Intestine 8、CNS 5、Breast 4、Skin 4、空 4、Ovary 3、Liver 3、Bladder 2、PNS 2、Pancreas 2、Bone 1、Soft Tissue 1。
- 在 mmc2 中 3 条、在 mmc3 蛋白表头中 3 条（`SIDM00096, SIDM00969, SIDM01095`）；其余 64 条只能经 Model.csv 桥接，在"mmc2 为唯一权威"的协议下与蛋白层无合法连接。

## 4. 逐折可用数（整组织留出场景；组织空值 4 条按既定规则排除）
| 外折 | 外折留出组织 | 因留出组织排除 | **外折可用** | inner-0 / 1 / 2（留出组织按 INVENTORY §3.2 转录） |
|---|---|---:|---:|---|
| 0 | Lung | 11 | **52** | 42 / 50 / 45 |
| 1 | Bone, Endometrium, Haem | 18 | **45** | 51 / 48 / 45 |
| 2 | Bladder, Breast, Kidney, Pancreas, PNS, Prostate | 10 | **53** | 48 / 43 / 45 |
| 3 | Biliary Tract, Head and Neck, Large Intestine, Ovary, Soft Tissue, Stomach | 12 | **51** | 47 / 41 / 50 |
| 4 | CNS, Esophagus, Liver, Other tissue, Skin, Thyroid | 12 | **51** | 51 / 41 / 46 |

新细胞场景每折 67（去掉组织空值则 63）。与 66 包相比每折最多 +1（`SIDM01095` 为 Soft Tissue，fold 3 被留出）。

## 5. M∩P 全集（只用身份，蛋白值未读）
| 口径 | M | M∩P | 其中 608 之外 | M∩RNA∩P 608 之外 |
|---|---:|---:|---|---|
| 仅 mmc2，多值字段整串（608 协议） | 609 | **609** | `SIDM00046`（无 RNA 列） | 0 |
| 仅 mmc2，多值字段拆分 | 611 | 611 | `SIDM00046, SIDM00096, SIDM01095` | `SIDM00096, SIDM01095` |
| mmc2 + Model.csv 桥 | 733 | 612 | 上述 + `SIDM00969` | `SIDM00096, SIDM00969, SIDM01095` |

P 表头 949、RNA 表头 1,047。即便最宽松口径，608 也占 M∩P 的 608/612 = 99.3%；按协议口径 608/609。M→P 生成器没有可扩容的配对。

## 6. 阴性与注意事项
- **67 < 150：生成器扩容不可行**；这与 INVENTORY 的结论一致，并把"原始源不可读"这一盲点关闭了——现在是穷举后的 67，不是沿用的 66。
- 本轮三次运行：前两次在身份阶段因我的断言过严停止（多值 `BROAD_ID`；"恰 609"），都在任何数值解码之前，修正写入 PLAN ADDENDUM-A/B，哈希版本保留。
- `SIDM01095` 身份歧义如上；池内 64 条经 Model.csv 桥接的线不在 mmc2 中，其 SIDM↔ACH 不是 mmc2 权威，与 66 包同一等级。
- 本文件与 qa.json 不含任何 608 线的 PatientID / 名字列表，只有数目。

## 7. 产物
`pack/m_rna_fullpool.npz`：`sample_ids`(67, SIDM)、`ach_ids`、`patient_ids`、`tissues`、`lineages`、`ccle_ids`、`rna_source_names`、`m (67,225) float64`、`rna (67,34985) float32`、`rna_counts_common (67,34985) float32`、`metabolite_ids (225)`、`gene_ids (34985)`、`common_source_gene_indices`、`parity_sample_ids/parity_m/parity_rna`（5 条 421 训练校验线）、`qa_json`。`pack/qa.json`：计数、排除清单、硬门、逐折、全集、逐样本表。

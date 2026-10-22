# Methods S1. CREED analysis and figure code, related to STAR Methods

## Purpose

This archive holds the original code of the CREED study: the data-preparation scripts that built the frozen development bundle and the legacy virtual representation, the 2026-10-08 runs whose saved fold, inner-bank, graph and teacher files the later runs read, every 2026-10-09 development run with its analysis script, the one-shot sealed-set run and its independent recomputation, the follow-up N6 with the script that builds Table S15, the CPTAC supplemental analysis, the figure data-preparation and plotting code, and the scripts that built the submission package. It contains code only: no data array, model file or run output. Registered plans, code checksums (SHA256SUMS) and run records are in Data S2 (development runs and N6) and Data S3 (sealed-set run).

All files are verbatim copies of the files that were executed (or, for the package builders, of the current builders). Nothing was edited, so absolute paths of the compute server (`/public/home/mengxl/...`), of the containers (`/code`, `/oldcode`, `/parent`, `/home/node/...`) and of the authors' workstation remain in the code, and some comments, logs and plan references are in Chinese. Files listed in a SHA256SUMS file of Data S2 or Data S3 can be checked against it.

## Repository and portable reproduction

The repository at https://github.com/dawnmengsjtu/CREED contains the code snapshot, portable launchers, pinned dependencies and validation instructions. It is publicly accessible without login. The repository snapshot uploaded on 2026-10-14 predates the submission-material link update: analysis and figure code are identical, while package builders and this README may differ. The GitHub integrity and synthetic smoke tests passed for commit d4b2f56497aaaf684a46beaf1ef920ab5452e55f. Full historical and sealed-set reproducibility remains subject to the limitations recorded in the repository. No software license has been added.

## Layout

Folders by component. Inside each, the folder names are the run directories of the compute server (`/public/home/mengxl/creed-research-runs/<run>/`) or the figure working directories (`/public/home/mengxl/workspaces/creed-research/figure_work/<dir>/`), and the paths below them are unchanged, so a path written in the code maps onto this layout. Files kept only in the authors' project repository (`null_baseline_20261009/`, `settle_stageA_20261011/`, `submission_CellSystems_20261011/_tools/`) keep their repository folder, and the independent recomputations are grouped under `independent_recomputation/<run>/`.

| Folder | Content | Files |
|---|---|---:|
| `01_data_preparation/` | Data preparation: identity protocol, frozen 421-line development bundle, aligned RNA, GDSC2 AUC, full proteome, CRISPR gene effect, legacy scGPT representation, whitelisted extra-line packages | 68 |
| `02_protein_graph_and_teacher_runs_20261008/` | Measured-layer teachers, protein sample-graph runs and the +19-line extra-supervision run whose saved fold, inner-bank, graph and prediction files every later run reads | 25 |
| `03_development_runs_20261009/` | The pre-registered development runs of 2026-10-09 (one folder per run) and the independent read-only recomputations of five of them | 48 |
| `04_sealed_set_20261009/` | One-shot confirmation on the sealed internal-final set and its independent recomputation | 7 |
| `05_followup_N6_20261011/` | Pre-registered follow-up N6 (function-weighted virtual layer) and the script that builds Table S15 | 9 |
| `06_cptac_supplemental/` | CPTAC cross-cohort layer-transfer analysis (Figure S5) and the full-source direct comparator | 38 |
| `07_figures/` | Figure data preparation, per-panel R scripts, renderer and assembler of Figures 1-6, Figures S1-S5 and the graphical abstract | 122 |
| `08_package_builders/` | Scripts that build the submission package (manuscript and Document S1 .docx, Document S1 PDF, Excel tables, Data S1-S10, this archive, upload zip) | 10 |
| `09_prospective_test_20261022/` | Pre-registered prospective test of the low-dimensional structure on GDSC1 (run prospective-lowdim-20261022-0001), its addendum, and the builder of the Figure 2H plot table | 10 |
| `SOURCES.tsv` | Origin, SHA-256 and size of every file | 1 |

`SOURCES.tsv` gives the origin, SHA-256 and size of every file at the time it was copied. For runs that executed from a code snapshot (01 and 06), only the files reachable by import from the entry scripts are included (static import closure, with `PYTHONPATH=/code/src:/code/scripts` as in the launchers); the rest of each snapshot is unrelated to this paper. Where a run kept a second byte-identical copy of a file (the `dev/` folders of three 2026-10-09 runs), it is not repeated.

## What each script produced

Figure, table and Data numbers are those of this paper; STAR Methods sections are named by their headings.

### `01_data_preparation/`

**`01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/`** (36 files)

Run that wrote the frozen 421-line development bundle `source-inputs.npz` (SHA-256 52d71517a8952e83a1603cd831f09449544102d0ca5ff829ff01be9bf79cd637; key resources table) from the parent bundle of `gdsc-metabolite-query-20260908-0001` and the full proteome. `code/` is the run's own code snapshot (commit in `source-commit.txt`), reduced to the files reachable by import from the entry scripts below (static import closure over `code/scripts` and `code/src`, as executed with `PYTHONPATH=/code/src:/code/scripts`). The same snapshot was mounted read-only as `/oldcode` by the 2026-10-08 runs in `02_...`, which import `creed.science.metabolite_query`, `metabolite_protein_query` and `masked_task_kernel` from it.
- `launch.sh` (producer and verifier modes; Docker, `--network none`) and `code/GDSC_METABOLITE_FULL_PROTEIN_BINDINGS.json` (SHA-256 of every mounted input).
- `code/scripts/run_gdsc_metabolite_full_protein.py`: writes `source-inputs.npz` (metabolites 421 x 225, proteome 421 x 6,692 with missingness, GDSC2 AUC 421 x 295, tissue labels, identifiers, fold definitions) and the frozen reference arms. STAR Methods: Frozen development inputs and whitelisted extra lines; Omics layers and preprocessing; key resources table.
- `code/scripts/verify_gdsc_metabolite_full_protein.py`, `verify_gdsc_metabolite_query.py`: the verifier mode of `launch.sh`.
- `code/scripts/prepare_research_421.py`: bundle `data421-0002` (metabolites 421 x 225 on the source scale, the 512-dimensional scGPT embedding used as the legacy FM target). Its SHA-256 equals the `preparation_sha256` recorded in that bundle's manifest. STAR Methods: Omics layers (Metabolites); Legacy virtual representation.
- `code/scripts/prepare_research_rna421.py`: the aligned RNA bundle `rna421-20260906-0002` (421 x 34,985 gene identifiers, ln(1 + within-panel CPM)); SHA-256 equals the recorded `preparation_sha256`. STAR Methods: Omics layers (Transcriptome).
- `code/scripts/prepare_research_tissues421.py`: tissue labels of the 421 lines; SHA-256 equals the `script_sha256` of `tissues421-20260906-0001.json`. STAR Methods: Omics layers (Tissue label).
- `code/scripts/prepare_protein421_full.py`: the full proteome `protein421-full-20260907-0001` (6,692 proteins, missing values kept). STAR Methods: Omics layers (Proteome). The run record stores no script hash; this is the snapshot version.
- `code/scripts/run_ccle_scgpt_sample_embedding.py` with `code/src/creed/science/scgpt_encoder.py` and `ccle_values.py`: the frozen whole-human scGPT cell embedding (archive SHA-256 9624b09167c8...) that `prepare_research_421.py` reads. STAR Methods: Legacy virtual representation. The record of the embedding run is not on the compute server; this is the snapshot version.
- `code/scripts/build_ccle_metabolite_identity_v2.py`, `build_ccle_visible_lineage_registry.py`, `build_ccle_triple_split_v1.py`, `materialize_ccle_visible_v1.py`: identity resolution, lineage registry, split freeze and visible-partition materialization, the steps behind the 608-line split CREED_CCLE_L0_L1_L2_SPLIT_V1 (Data S1) and the development-visible archive (SHA-256 468767f1...) that `prepare_research_421.py`, `prepare_research_rna421.py` and `prepare_protein421_full.py` read. STAR Methods: Cell-line identity protocol and data split; Figure 1. These steps predate the run directories and their run records were not located; snapshot versions.
- All other files under `code/`: helpers imported by the scripts above.

**`01_data_preparation/gdsc-metabolite-query-20260908-0001/`** (12 files)

Run that fitted the three cross-fitted metabolite-to-FM generators of every held-out-tissue bank (the legacy virtual representation; target: the scGPT embedding of `data421-0002`) and wrote the parent bundle read by `run_gdsc_metabolite_full_protein.py`; the new-cell-line generators were fitted with the same `build_bank` function by `run_geometry.py` (`02_.../privileged-geometry-20261008-0001/`). Own code snapshot (commit in `source-commit.txt`), reduced to the import closure of `run_gdsc_metabolite_query.py` and its verifier; every file is byte-identical to the copy in `gdsc-metabolite-full-protein-20260908-0001/code/`. `launch.sh` records the mounts (data421, rna421, gdsc421 and the two earlier reference runs). STAR Methods: Virtual layers (Legacy virtual representation).

**`01_data_preparation/gdsc421-20260906-0001/`** (7 files)

`run.sh` and `code/scripts/prepare_gdsc421.py` (with its import closure): GDSC2 fitted AUC for the 421 lines on the fixed 295-drug panel (exact SANGER_MODEL_ID and DRUG_ID; invalid and conflicting records excluded; missing pairs kept missing). STAR Methods: Drug-response endpoint. Run in the image `creed-vcp-runtime:py3112-torch220-esm321-v9`, as `run.sh` records.

**`01_data_preparation/crispr310-prepare-20260908-0001/`** (5 files)

`code/scripts/prepare_crispr_development.py` (with its import closure) and `code/DEPMAP24Q2_SOURCE_BINDINGS.json` (its SHA-256 equals the `bindings_sha256` in the output manifest): the CRISPR gene-effect matrix of the 310 screened development lines x 18,443 genes from DepMap 24Q2. STAR Methods: CRISPR gene-effect endpoint; Figure 5; Tables S10-S12. The launcher of this run was not retained.

**`01_data_preparation/extra-supervision-20261008-0001/resources/`** (8 files)

Extraction of the three whitelisted extra-line packages (19 M-AUC lines: `extract_m_gdsc.py`, `verify_m_gdsc.py`, `m_gdsc_gateway_payload.py`; 66 M-RNA and 354 RNA-AUC lines: `extract_rna_candidates.py`, `verify_and_package_rna.py`, `rna_gateway_payload.py`). The two `*_gateway_payload.py` files are the self-contained readers that were executed; they embed the frozen whitelist as JSON (cell-line, DepMap model and patient identifiers, tissue, metabolite-feature, drug and gene-index identifiers; no assay value). `original_prepare_gdsc421.py` and `original_prepare_research_rna421.py` are byte-identical copies of the preparation scripts in this component, used for source-scale parity. STAR Methods: Frozen development inputs and whitelisted extra lines; Data S1.

### `02_protein_graph_and_teacher_runs_20261008/`

**`02_protein_graph_and_teacher_runs_20261008/strong-teacher-recoverability-20261008-0001/`** (8 files)

Measured-proteome teachers ("M + measured P values") whose saved predictions the panel, acquisition and push-line runs reproduce (teacher of R(k), Table S5; STAR Methods: Supervised measured panels and random-panel controls). `launch_scenario.sh` ran `code/run_recoverability.py` (new cell lines); `launch_tissue_v2.sh` ran `code_v2/run_recoverability.py` (held-out tissue), whose only change is the support fallback for teacher folds with fewer than three training tissues, the mid-run revision disclosed in STAR Methods (Quantification and statistical analysis, Pre-registration and reproduction gates). `code/verify_results.py` replays the saved selections and scores; `code/preflight.py` is the preflight; `new_cell_fold_ids.json` is the new-cell-line fold assignment of the 421 lines (fold numbers only).

**`02_protein_graph_and_teacher_runs_20261008/privileged-geometry-20261008-0001/`** (10 files)

Protein sample-graph regularization with an M-only query interface. `launch.sh` ran `code/run_geometry.py` (mu grid 0-10; it reuses the held-out-tissue FM banks of `gdsc-metabolite-query-20260908-0001` and fits the new-cell-line M-to-FM generators); `launch_extension.sh` ran `code/extend_geometry.py`, the disclosed extension to mu = 100 and 1000. Their saved protein indices and graphs are read by the later runs. `verification/`: independent checks of the saved results, the effective-kernel algebra and the large-mu preflight. STAR Methods: Protein sample-graph regularization; Figure 3; Figure S2; Data S5 (through `pushline_checks.py`).

**`02_protein_graph_and_teacher_runs_20261008/extra-supervision-20261008-0001/`** (7 files)

`launch.sh` ran `code/run_extra.py` (imports `run_recoverability.py`): the "+19" arms with the 19 whitelisted M-AUC lines added as training rows. This run wrote the saved fold, inner-bank, kernel-grid and prediction files (`output/<scenario>/fold-*/...`) that every 2026-10-09 run, the sealed-set freeze and N6 load and assert unchanged. `verification/`: independent verification of the saved assets, selections and scores (`verify_saved_extra.py`, launched by `launch_verification.sh`) and the extension preflight. STAR Methods: Cross-validation design; Frozen development inputs and whitelisted extra lines.

### `03_development_runs_20261009/`

**`03_development_runs_20261009/crosslayer-matrix-20261009-0001/`** (5 files)

`code/crosslayer.py` (all arms, both scenarios), `code/analyze.py` (pre-registered analysis), `code/run.sh`; `dev/dev_fake_root.py` builds the synthetic root of the smoke test; `posthoc/posthoc_saved_compare.py` is the post-hoc, descriptive comparison with the saved project arms (including the legacy FM candidate). Produced: Figure 2A-G, Figure S2A-C, Table S2, Data S4, and the predictions that the Figure 3 and Figure S4 data stages and Table S9 use. STAR Methods: Per-drug masked kernel ridge head; Virtual layers (including the paired-shuffle control and headroom, recovered gain and recovered share).

**`03_development_runs_20261009/generator-curve-20261009-0001/`** (3 files)

`code/gencurve.py`, `code/analyze_curve.py`, `code/run.sh`: generator-size learning curve (f50/f75/f100 and the +66 extra M-RNA arm). Produced: Figure 2G, Figure S1A-C, Table S3. STAR Methods: Generator-size learning curve and pairing universe.

**`03_development_runs_20261009/generator-fullpool-20261009-0001/`** (2 files)

`code/extract_fullpool.py`, `code/run.sh`: identity-only enumeration of the pairing universe outside the 608 split lines (the final code file is the one registered in SHA256SUMS.v3; the two earlier versions corrected before the run were not retained). Produced: Figure 2H, Figure S1D, Table S4, Table S8 row H093. STAR Methods: Generator-size learning curve and pairing universe.

**`03_development_runs_20261009/protein-panel-20261009-0001/`** (2 files)

`code/protein_panel.py`, `code/analyze.py`: supervised measured protein panels, random-panel controls and graph-on-panel arms. Produced: Figure 4B-D, Figure S3A, Table S5, Data S6 (and the graph-on-panel row H083 of Table S8). STAR Methods: Supervised measured panels and random-panel controls.

**`03_development_runs_20261009/rna-panel-20261009-0001/`** (4 files)

`code/rna_panel.py`, `code/analyze.py`, `code/run.sh`, `code/job_dryrun.sh`: the gene-panel version. Produced: the gene rows of Figure 4B-D, Table S6, Data S7. STAR Methods: Supervised measured panels and random-panel controls.

**`03_development_runs_20261009/acquisition-probe-20261009-0001/`** (2 files)

`code/acquisition_probe.py`, `code/make_tables.py` (formatting only): N1, per-sample adaptive protein acquisition. Produced: Figure 4E, Table S7 (N1), Data S8 (part a), Table S8 row H087. STAR Methods: Pre-registered negative controls (N1).

**`03_development_runs_20261009/acquisition-probe-rna-20261009-0001/`** (3 files)

`code/acquisition_probe_rna.py`, `code/run.sh`; `dev/dev_fake_root.py` (synthetic smoke test): the transcriptome mirror of N1. Produced: Data S8 (part b), Table S7, Table S8 row H088. STAR Methods: Pre-registered negative controls (N1). The `dev/` copy of `acquisition_probe_rna.py` is byte-identical to `code/` and is not repeated.

**`03_development_runs_20261009/probe2-context-abstain-20261009-0001/`** (6 files)

`code/probe2.py` (part A, N2 context-adaptive panels), `code/abstain.py` (part B, N3 abstention), `code/analyze.py`, `code/make_result_tables.py` (formatting only), `code/run.sh`, `code/job_dryrun.sh`. Produced: Figure 4F-H, Figure S3B, Table S7 (N2, N3), Data S9, Data S10, Table S8 row H089. STAR Methods: Pre-registered negative controls (N2, N3).

**`03_development_runs_20261009/crispr-endpoint-20261009-0001/`** (3 files)

`code/crispr_endpoint.py`, `code/analyze.py`, `code/run.sh`: the cross-layer map and the panel law on the CRISPR gene-effect endpoint. Produced: Figure 5, Tables S10-S12, the development CRISPR values of Figure 6. STAR Methods: CRISPR gene-effect endpoint; Pre-registered negative controls (N5).

**`03_development_runs_20261009/pushline-checks-20261009-0001/`** (1 file)

`code/pushline_checks.py` (analysis only, no training): controls, decomposition, leave-one-fold-out and bootstrap checks of the protein sample-graph and tissue-label effects. Produced: Data S5; with the cross-layer and tissue-kernel outputs, the plot tables of Figure 3 (A, E, H) and Figure S2D-F (through `prepare_fig4_v2.py`). STAR Methods: Protein sample-graph regularization; Quantification and statistical analysis (Robustness).

**`03_development_runs_20261009/tissue-kernel-20261009-0001/`** (1 file)

`code/tissue_kernel.py`: tissue-kernel augmentation K_M + w T in new cell lines. Produced: Figure S2D (free tissue label, with the cross-layer run; through `prepare_fig4_v2.py`). STAR Methods: Per-drug masked kernel ridge head (tissue kernel).

**`03_development_runs_20261009/shared-crossdrug-20261009-0001/`** (3 files)

`code/shared_crossdrug.py`, `code/test_synthetic.py` (synthetic unit tests), `code/diagnose.py` (post-hoc, descriptive, not pre-registered). Produced: Table S8 row H084. STAR Methods: Pre-registered negative controls (N4).

**`03_development_runs_20261009/rna-learning-curve-20261009-0001/`** (6 files)

`code/rna_lc.py`, `code/analyze.py`, `code/run.sh`, `code/preflight_identity.py`, `code/test_inner.py` (inner banks only), `code/fake_outputs.py` (synthetic outputs to test `analyze.py`): RNA learning curve, the 354 RNA-AUC extra-label gate and the RNA-teacher pseudo-label control. Produced: Table S8 row H085. STAR Methods: Pre-registered negative controls (N4). The `dev/` copies are byte-identical to `code/` and are not repeated.

**`03_development_runs_20261009/null_baseline_20261009/`** (1 file)

`null_baseline.py`: reproduction of the saved new-cell-line scores and the per-drug training-mean null; run read-only through ssh standard input, no server run directory. Produced: Table S1, Note S1. STAR Methods: Quantification and statistical analysis (Null models).

**`03_development_runs_20261009/independent_recomputation/`** (6 files)

Read-only recomputations written separately from the run code and executed through ssh standard input (`python -I`): `crispr-endpoint-20261009-0001/recheck.py`, `rna-panel-20261009-0001/audit_metric.py`, `probe2-context-abstain-20261009-0001/auditA.py` and `auditB.py`, `acquisition-probe-20261009-0001/audit_recompute.py`, `acquisition-probe-rna-20261009-0001/audit.py`. They rescore saved predictions from saved truth and training scales and recompute the pre-registered comparisons. STAR Methods: Quantification and statistical analysis (Independent recomputation).

### `04_sealed_set_20261009/`

**`04_sealed_set_20261009/internal-final-20261009-0001/`** (6 files)

`code/oneshot.py` (freeze, gates, dry run and the single sealed run), `code/creed_core.py` (model core copied from the development reference implementations), `code/sealed_io.py` (identity-gated loaders; only whitelisted identities are decoded); `logs/run_gates_dryrun.sh`, `logs/run_dryrun.sh`, `logs/run_freeze.sh` are the launchers kept with the logs. The three code files are those bound by SHA256SUMS.v2 and the release receipt (Data S3). Produced: Figure 6, Table 1, Tables S13 and S14. STAR Methods: One-shot confirmation on the sealed internal-final set; Quantification and statistical analysis (Sealed-set statistics).

**`04_sealed_set_20261009/independent_recomputation/`** (1 file)

`recompute_remote.py`: the separately written read-only recomputation named in Data S3 (RECOMPUTE.md), executed once through ssh standard input with `python -I`; it re-derives the sealed truth from the raw files for the 123 identities it lists (identifiers only), recomputes every score, bootstrap, sign test and verdict, and checks hashes, timeline and identities. STAR Methods: Quantification and statistical analysis (Independent recomputation).

### `05_followup_N6_20261011/`

**`05_followup_N6_20261011/settle-stageA-20261011-0001/`** (8 files)

`code/stageA.py` and `code/stageA_crispr.py` (N6 on the drug-response and CRISPR endpoints), `code/analyze.py`, `code/analyze_crispr.py`, `code/run.sh`, `code/run_crispr.sh`. `code/crosslayer.py` and `code/crispr_endpoint.py` are the frozen development files that N6 imports; they are byte-identical to the copies in `03_.../crosslayer-matrix-20261009-0001/code/` and `.../crispr-endpoint-20261009-0001/code/`. Produced: Table S15, Note S3. STAR Methods: Pre-registered negative controls (N6).

**`05_followup_N6_20261011/settle_stageA_20261011/`** (1 file)

`make_table_s15.py`: builds Table S15 from the per-fold N6 results (averaging only); also in Data S2.

### `09_prospective_test_20261022/`

**`09_prospective_test_20261022/prospective-lowdim-20261022-0001/`** (9 files)

`code/lowdim.py` (fold worker: inner-bank dry run, outer predictions with the reproduction gates, scoring), `code/prepare_gdsc1.py` (GDSC1 extraction and data gate), `code/analyze.py` (pre-registered analysis), `code/run.sh` (the single invocation), `code/register.sh`, `code/preflight.sh` and `code/FROZEN_SHA256SUMS`. The run imports six frozen files from `code/frozen/`; they are byte-identical to `crosslayer.py`, `crispr_endpoint.py`, `stageA.py` and `stageA_crispr.py` in `05_followup_N6_20261011/settle-stageA-20261011-0001/code/` and to `protein_panel.py` and `rna_panel.py` of the protein- and gene-panel runs in `03_development_runs_20261009/` (checksums in `code/FROZEN_SHA256SUMS`), and are not repeated here. `code_addendum1/resume_after_addendum1.sh` and `code_addendum1/synthetic_test.py` belong to ADDENDUM 1 (score-stage software error fixed before any score). Executed with the CellFM Python, four threads per process. STAR Methods: Prospective test of the low-dimensional structure; Note S4.

**`09_prospective_test_20261022/prospective_lowdim_20261022/figure_2H/`** (1 file)

`build_fig2h_controls.py`: builds the Figure 2H plot table (`lowdim_controls.csv`) from the run's `analysis/analysis.json` (values copied, no recomputation).

### `06_cptac_supplemental/`

**`06_cptac_supplemental/cptac-blocks-20260906-0001/`** (2 files)

`code/scripts/prepare_cptac_blocks.py` with `code/src/creed/science/cptac_blocks.py`: source-native per-cohort bundles from the cBioPortal study archives (the script also writes bundles of cohorts that this paper does not use). STAR Methods: Cross-cohort layer transfer in CPTAC cohorts; Figure S5.

**`06_cptac_supplemental/cptac-baseline-20260906-0002/`** (4 files)

`code/scripts/run_cptac_bidirectional_baseline.py` with `code/src/creed/science/masked_baseline.py`, `verify_cptac_baseline.py` and the design note `code/experiments/CPTAC_BIDIRECTIONAL_BASELINE_20260906.md`: the BRCA + LUAD bidirectional baseline that defined the five cohort-stratified patient folds (scikit-learn `StratifiedKFold` on cohort labels, seed 20260906) and the target panels of the mixed-fold task. `cptac-anchor-20260906-0001` reads its saved folds (asserting the SHA-256 of its summary, 156e35f7...), and `cptac-full-source-direct-20261008-0001` reads them through the anchor run's outputs (`source_summary_sha256` f095caa4...). The launcher was not retained. STAR Methods: Cross-cohort layer transfer in CPTAC cohorts; Discussion.

**`06_cptac_supplemental/cptac-anchor-20260906-0001/`** (4 files)

`code/scripts/run_cptac_anchored_context.py` (same-gene anchor plus context) and `verify_cptac_baseline.py`: the anchor + context predictions of the BRCA + LUAD mixed-fold task against which `cptac-full-source-direct-20261008-0001` compares. STAR Methods: Cross-cohort layer transfer in CPTAC cohorts.

**`06_cptac_supplemental/cptac-three-inputs-20260907-0001/`** (5 files)

`code/scripts/prepare_cptac_three_domains.py` (SHA-256 equals the `producer_sha256` of the run summary): the BRCA, LUAD and UCEC input bundle (327 patients) of the transfer tasks. STAR Methods: Cross-cohort layer transfer in CPTAC cohorts.

**`06_cptac_supplemental/cptac-domain-experts-20260907-0001/`** (9 files)

`run.sh` and `code/scripts/run_cptac_domain_experts.py`: the six transfer tasks without shift correction, the reference run mounted by the source-shift run. STAR Methods: Cross-cohort layer transfer in CPTAC cohorts.

**`06_cptac_supplemental/cptac-source-shift-20260907-0001/`** (12 files)

`run.sh` and `code/scripts/run_cptac_domain_experts.py --source-shift`: the six tasks with source-only shift correction (training mean, pooled anchor, pooled context, uniform, gated and reversed-gate experts). `check_domain_expert_outputs.py` recomputes the evaluation, `predict_domain_translation.py` runs the saved models on source-only inputs and `check_domain_source_only.py` compares the two (checker launch lines were not retained). Produced: Figure S5 (through `prepare_cptac.py`). STAR Methods: Cross-cohort layer transfer in CPTAC cohorts.

**`06_cptac_supplemental/cptac-full-source-direct-20261008-0001/`** (2 files)

`run.py`: full-source linear direct comparator for the BRCA + LUAD mixed five-fold task (imports `creed.science.masked_baseline`; every code snapshot from 2026-09-07 on carries the same `masked_baseline.py`, SHA-256 c5bf00ac..., as in `cptac-source-shift-20260907-0001/code/src/`; the launcher was not retained). `summarize.py` (from the run record in `research_plans/20261008/strategy/positive_core/cptac_full_source_reader/`) summarizes it without fitting. Used for the RNA-to-protein comparison in the Discussion.

### `07_figures/`

**`07_figures/data_preparation/`** (12 files)

Figure data stage: Python scripts that turn saved run outputs into the plot tables (`out/<figure>/plot_data/*.csv`) read by the R panels; each writes a `*_DATA_SOURCES.md` with the SHA-256 of its inputs. Copies from `20261009_figures_v3_cell/r/figures2_6/`, byte-identical to the copies in `20261009_figures2_6_v2/r/figures2_6/` where they were executed (`run.sh data` of that folder runs `prepare_gdsc.py`, `prepare_cptac.py` and `prepare_protocol.py`; the `prepare_*_v2.py` scripts were run one by one). Final figure numbers: `prepare_fig2_v2.py` -> Figure 2 and Figure S2B-C; `prepare_fig3_v2.py` -> Figure 3 (B, C, D, F, G) and Figure S2A; `prepare_fig4_v2.py` -> Figure 3 (A, E, H) and Figure S2D-F; `prepare_fig5_v2.py` -> Figure 4 and Figure S3; `prepare_fig6_v2.py` -> Figure S4; `prepare_fig7_v2.py` -> Figure 5; `prepare_fig8_v2.py` -> Figure 6 (it asserts the plot tables against the published sealed-set values); `prepare_figS1_v2.py` -> Figure S1 (it also built the pairing-universe panel that was Figure 2H until 2026-10-22; Figure 2H is now built from folder 09); `prepare_cptac.py` -> Figure S5; `prepare_gdsc.py`, `prepare_protocol.py` -> GDSC and schematic count tables of the earlier figure set reused by the above.

**`07_figures/figure1/`** (17 files)

Figure 1, built with the code of the v3 figure set: `r/figure1/A.R`-`H.R` (panels), `common.R`, `render_panels.R`, `assemble.R`, `run.sh`, `check_delivery.py` and `_tools/` (vector-bound checks), with `r/cell_style.R` (identical to the v4 copy). Its plot tables are a frozen copy of the accepted 2026-10-07 tables; `prepare_data.py` is the data stage that built them (kept for provenance, not re-run). STAR Methods: Software and computation.

**`07_figures/figures_v4/`** (92 files)

Figures 2-6 and Figures S1-S5: `r/render_v4.R <fig> [panel ...]` renders each panel independently at its slot size from `r/<fig>/layout.csv` (PDF, PNG and RDS), and `r/assemble_v4.R <fig>` places the saved panels on the 174-mm page; `r/v4_common.R` and `r/cell_style.R` hold the shared reader, export and style code. In `r/<fig>/`, `A.R`-`H.R` are the panels, `LEG_*.R` and `KEY.R` the legend and key slots, `common.R` the figure's own helpers; `Figure4/prep/extract_probe2_*.py` and `Figure6/prep_dev_folds.R` extract the per-fold tables of Figure 4G-H and Figure 6F-G from saved outputs (no recomputation). The server folder names keep the pre-v1.8 supplementary numbers: `r/FigureS4/` is Figure S1, `r/FigureS1/` is Figure S2, `r/FigureS3/` is Figure S3, `r/FigureS2/` is Figure S4 and `r/FigureS5/` is Figure S5; `r/Figure2/`-`r/Figure6/` are Figures 2-6. Superseded variants kept on the server (`_old/`, `_old_v4/`, `_v4r0/`, `_bak/` and the layout-trial folders) are not included.

**`07_figures/graphical_abstract/`** (1 file)

`graphical_abstract.R`: the graphical abstract.

### `08_package_builders/`

**`08_package_builders/`** (10 files)

Builders of the submission package, refreshed from `_tools/` at every build of this archive: `build_cell_docx.py` (manuscript, Document S1, cover letter and highlights .docx from the Markdown masters), `build_xlsx.py` (Excel Tables S3, S4, S8 and S11-S14), `build_data_zips.py` (Data S1-S10), `build_methods_s1.py` (this archive), `build_si_pdf.py` (Document S1 PDF: WebKit print of the Document S1 master; imports build_cell_docx.py), `build_submission_zip.py` (upload zip with SHA-256 manifest), `build_docx_v18_original.py` (earlier builder kept for provenance), and the render checks `render_check.sh`, `html2pdf.swift` and `pages_export.applescript`.

### `SOURCES.tsv`

**`SOURCES.tsv`** (1 file)

Origin of every file: `server:` absolute path on the compute server, `repo:` path in the authors' local project repository, or `session-scratchpad:` file name in the working directory of the session that wrote the independent recomputations; with the SHA-256 and size at the time of copying.


## How the runs were executed

- 2026-10-09 development runs, sealed-set run and N6: on CPU. Their launchers (`run.sh`, `job_dryrun.sh`, `logs/run_*.sh`) call the interpreter of the conda environment CellFM (`/public/home/mengxl/.conda/envs/CellFM/bin/python`; Python 3.9.23, NumPy 2.0.2, SciPy 1.13.1, openpyxl 3.1.5, as that environment reports on 2026-10-11) and fix the thread counts (8 in total per run: one 8-thread process, two scenarios in parallel with 4 threads each, or eight concurrent single-thread jobs; N6, four threads per process). Five runs kept no launcher (protein-panel, acquisition-probe, pushline-checks, tissue-kernel, shared-crossdrug); the usage line at the top of `protein_panel.py`, `acquisition_probe.py` and `shared_crossdrug.py` gives the command form. `null_baseline.py` and the independent recomputations were run through ssh standard input. Plan and code checksums were frozen before each main run (SHA256SUMS in Data S2 and Data S3).
- Data preparation, the 2026-10-08 runs and the CPTAC analysis (run directories of 2026-09-05 to 2026-10-08): in Docker containers without network access, with code and inputs mounted read-only, as each `launch*.sh` or `run.sh` records; image `sha256:0a450c02d469c8bad7d43c248fee99789e2f408472b0e5a2d1e118e04c250b9a` (local image ID, not in a public registry; key resources table), named in some launchers by its tag `creed-vcp-runtime:py3112-torch220-esm321-v9` (the same image; `docker image inspect`, 2026-10-11). Inside it: Python 3.11.2, NumPy 1.26.4, SciPy 1.16.3, scikit-learn 1.7.2, PyTorch 2.2.0+cu121 (queried 2026-10-11). Exception: the three whitelisted extra-line packages in `01_data_preparation/extra-supervision-20261008-0001/resources/` were extracted by their self-contained `*_gateway_payload.py` scripts (frozen whitelists in `*_config.json`) through `docker exec` in the long-running `openclaw-gateway` service container (alpine/openclaw image, bridge network, workspace mounted read-write), as the resources README records; the frozen scGPT embedding ran on GPU in a separate Python 3.10 runtime with PyTorch 2.3.0+cu121 and scGPT 0.2.4 (the runtime receipt that `run_ccle_scgpt_sample_embedding.py` validates), and no execution records were located for the identity-resolution steps (folder notes above). Container records on the compute server (`docker ps -a`, 2026-10-11) show the CPTAC blocks, baseline, anchor, three-inputs, domain-experts and source-shift runs in this image with no network; for `cptac-full-source-direct-20261008-0001` no container was kept, and its run record states a no-network container with 2 CPUs. The 2026-10-08 runs imported the code snapshot of `gdsc-metabolite-full-protein-20260908-0001` mounted as `/oldcode`. Some launchers of these earlier runs were not retained; this is stated in the folder notes above.
- Figures: R 4.4.3 in the conda environment creed-r (`/public/home/mengxl/.conda/envs/creed-r/bin/Rscript`) with ggplot2 4.0.3, cowplot 1.2.0, ggalluvial 0.12.6, dplyr 1.2.1, tidyr 1.3.2, readr 2.2.0, svglite 2.2.2, jsonlite 2.0.0 and gtable 0.3.6; data-preparation scripts with the CellFM Python (`prepare_fig2_v2.py`, `prepare_fig4_v2.py` and `prepare_fig8_v2.py` also use pandas 2.3.2). From the figure working directory: `Rscript r/render_v4.R Figure3` renders every panel of a figure, then `Rscript r/assemble_v4.R Figure3` assembles the page.
- Package builders: Python 3 standard library only, on the authors' workstation (render checks: macOS Quick Look, a Swift/WebKit HTML-to-PDF helper and pdftoppm).

## Scans

Every file was scanned for credentials, tokens, private keys, e-mail addresses and IPv4 addresses (none found; `render_check.sh` names the compute server by its SSH host alias) and for data values. No file contains an assay value of any cell line or patient. Files contain identifiers only where the code needs them (cell-line, model and patient identifiers in the whitelists of `01_.../resources/*_gateway_payload.py` and in the loaders of the sealed-set run; the 123 sealed identifiers in `recompute_remote.py`). `prepare_fig8_v2.py` and `build_cell_docx.py` contain published sealed-set summary values (as in Table 1 and Figure 6) as consistency checks or cover-letter text.

## SHA-256 of every file

All files of this archive except this README.

| SHA-256 | Bytes | Path |
|---|---:|---|
| ace630ab88eed0e4d71b06ffcc2a893358e3e50241e0e1fb7253f8449f14fff7 | 1097 | `01_data_preparation/crispr310-prepare-20260908-0001/code/DEPMAP24Q2_SOURCE_BINDINGS.json` |
| 13ec95e97273108c960998892563bbdcd460d7a6fa9439da7475b32872cedca8 | 4455 | `01_data_preparation/crispr310-prepare-20260908-0001/code/scripts/prepare_crispr_development.py` |
| 09f9413a3832c6604e8b6b3b46d2088ba665d424c04f35501a0dbc1f7a5bcdb9 | 1352 | `01_data_preparation/crispr310-prepare-20260908-0001/code/src/creed/__init__.py` |
| e0d1c2ba913d54d81d30e6676813b52acbba400e917f3122591f7adc7652f17d | 1106 | `01_data_preparation/crispr310-prepare-20260908-0001/code/src/creed/science/__init__.py` |
| 0ef3134d8cb18cee72e613d52c5ebd4237c2dc6989df027616da3c3e105f05fc | 6527 | `01_data_preparation/crispr310-prepare-20260908-0001/code/src/creed/science/functional_source_catalog.py` |
| b9a25cb6dcc9de39655add5223c2ba8e4be981eef38f4654360eb4de1bebac5b | 6935 | `01_data_preparation/extra-supervision-20261008-0001/resources/extract_m_gdsc.py` |
| 7c9add36db97c3917b3cc383cf58616cf86f148503d7e4490fd3eda123dce833 | 7912 | `01_data_preparation/extra-supervision-20261008-0001/resources/extract_rna_candidates.py` |
| 555e97913783ad0992cfa567f30013d1c004590371ebc653e8988170e27aaca9 | 31652 | `01_data_preparation/extra-supervision-20261008-0001/resources/m_gdsc_gateway_payload.py` |
| 91d19ceb82801fa73389b36d6c2e095b518bd872d6d7055d9e0ebe0ed5ace064 | 5139 | `01_data_preparation/extra-supervision-20261008-0001/resources/original_prepare_gdsc421.py` |
| e3ad2ccac1d5438bd839671fd44070cf9e7fa3e07d803b63a752c4e0f7c216bd | 5704 | `01_data_preparation/extra-supervision-20261008-0001/resources/original_prepare_research_rna421.py` |
| 12ec06476d3500f522788b6169444094a58dfd6e7f0b4d8dd86176b6855bf551 | 290878 | `01_data_preparation/extra-supervision-20261008-0001/resources/rna_gateway_payload.py` |
| 5ccee585fb56c1b7814511fc0e3ce133103bc5f840455409ffae1aa94cd96cb7 | 4750 | `01_data_preparation/extra-supervision-20261008-0001/resources/verify_and_package_rna.py` |
| b39d51082bc7e3370fd78a209df0f3046285902fca5459aff89c737a7c60752e | 1906 | `01_data_preparation/extra-supervision-20261008-0001/resources/verify_m_gdsc.py` |
| 457e45bcb80adec8cbf375ff4701453b252f9db8d87d2ac8568057a04292809a | 1856 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/code/GDSC_METABOLITE_FULL_PROTEIN_BINDINGS.json` |
| 2dcabd5bcfa7c988e4fe102e13a7163c47bd2fcf80b78214636613c9e5d4d829 | 21223 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/code/scripts/build_ccle_metabolite_identity_v2.py` |
| 30a94a8aa553af9ff38ebb9241a2abc7b17bb9c09d58548a3f5945343176718d | 7672 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/code/scripts/build_ccle_triple_split_v1.py` |
| ec513fee45e4ce05dcbefe4e29175d64698491f86f36894f91b8b03b5cae75d7 | 3007 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/code/scripts/build_ccle_visible_lineage_registry.py` |
| 9e021a9445f911b89bf4e048d9f9bb9fdb23614428586a81237187f041e12e03 | 15173 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/code/scripts/materialize_ccle_visible_v1.py` |
| 31d8d8f2ae1e4377c512b11e0c82b0cd1422072c67df739efacd980f242faa18 | 3073 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/code/scripts/prepare_protein421_full.py` |
| 7e56a12f4da3b7b62d1799161595991fc7680cc5685da810219643146f648a60 | 3803 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/code/scripts/prepare_research_421.py` |
| e3ad2ccac1d5438bd839671fd44070cf9e7fa3e07d803b63a752c4e0f7c216bd | 5704 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/code/scripts/prepare_research_rna421.py` |
| 4171385bf120ace8af55e220da64ad188935f8952f40f9676d0d0f7234923b83 | 1646 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/code/scripts/prepare_research_tissues421.py` |
| 7aba5ced3a50de5ac1c4651e2fe6facf25d9a65a121901e7a2f67b7435f3d0ee | 32909 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/code/scripts/run_ccle_scgpt_sample_embedding.py` |
| 591667b43b50dca15512a062ce5161ba15e550cf2b3e1bb6d3b4b43342c7203f | 25826 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/code/scripts/run_gdsc_metabolite_full_protein.py` |
| 8b3d441c33651d3ee04e2f7e5f9e80100531aa2b1bd534d0a1645a6038617291 | 19712 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/code/scripts/run_gdsc_metabolite_query.py` |
| b56849aaca1310526114077aa104b9ca4a700d09605942cbc7b556bf2307afe1 | 20380 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/code/scripts/run_gdsc_metabolite_task_metric.py` |
| 298488a8385681b0391530a6e7dfa55c244c874ccfaa73c3c004e9738c4ccef6 | 33355 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/code/scripts/run_phospho_fewshot.py` |
| e74fc2dfda38902c0492a2a36087d6e07cc692bba1fceb202161b5aeb96fc9af | 8930 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/code/scripts/run_research_diagnostic.py` |
| 75f759aeb187bffea42fb1210eeea1d556f0a6513cca285c26bc83991c6b0ac0 | 33671 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/code/scripts/verify_gdsc_metabolite_full_protein.py` |
| 9f9a0fd8cf6e2e02b7188c28299c87cd5cc79f0b8c6825ed48aeb5a302c6e46d | 46938 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/code/scripts/verify_gdsc_metabolite_query.py` |
| 09f9413a3832c6604e8b6b3b46d2088ba665d424c04f35501a0dbc1f7a5bcdb9 | 1352 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/code/src/creed/__init__.py` |
| e0d1c2ba913d54d81d30e6676813b52acbba400e917f3122591f7adc7652f17d | 1106 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/code/src/creed/science/__init__.py` |
| e869bf9dab9826f26bde79ac514d854e14c53d74b2258327559e351e1d4a695b | 33922 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/code/src/creed/science/ccle.py` |
| 0e5cf0c956de95b99059b5d8e00fa9b873d10884cc93fb6ef1852fe2dfa830a6 | 12554 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/code/src/creed/science/ccle_metabolite_identity.py` |
| 4e1002435dd007c4ff3562ea65d93291749d05765f1ad7aea75a6df8e549d42b | 134926 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/code/src/creed/science/ccle_multifm_bidir_v5.py` |
| 998bf0012bbc9edcb3accfe975bc6ea412aea2aa7c719322d139a28f656174c3 | 133828 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/code/src/creed/science/ccle_multifm_v5_direct_comparators.py` |
| 0cea67cbbb9d254349ff4f938d2266be60493fb27b6569cc9a5699a8e6e24c72 | 37227 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/code/src/creed/science/ccle_values.py` |
| 0debf3323c992372af53bb0d1a5fba104a06735711b538cec3321105191b2811 | 102179 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/code/src/creed/science/ccle_visible_lineage_registry.py` |
| e70f8538c009317fa05d2e11c4bdf53975fec194a1fc23145cc1ab54bb354008 | 3260 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/code/src/creed/science/drug_target_state.py` |
| fd65e4e7cc7d93b2c0c253d40a461ebc30f1dfc0def73801210c0d69082ced99 | 919 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/code/src/creed/science/masked_task_kernel.py` |
| f20564f3be1860e1d4f25de3f565d3a45b2430842ccf17993feeb6a72fb5a7e6 | 16143 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/code/src/creed/science/metabolite_protein_query.py` |
| f1f2b88d44066c07be975073a590899bbfbefaceb9765874912b6633c3de922b | 9693 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/code/src/creed/science/metabolite_query.py` |
| 303c0c1c49b226d293024407cce5254322661da6d51b2f2de50b61336906a7b2 | 26523 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/code/src/creed/science/metabolite_registry.py` |
| a4bbbbb861f7b702271db756d5d1ef6c8618a168830586ff54b98d6e8822c656 | 12407 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/code/src/creed/science/phospho_fewshot.py` |
| bba5d4d3e3f22771fb4c46e2cc866c8d6d550263a6c94ccf1da97d566f7c2e55 | 52342 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/code/src/creed/science/scgpt_encoder.py` |
| af538251e5247682d93682b01684c5bf1601fa94c250c22c6a6d49feebd3c5d9 | 8163 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/code/src/creed/science/task_residual_metric.py` |
| 591e5646ed940cc61f337c3d389d0931e053112b607da5ad8022d10d658fa2eb | 4777 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/code/src/creed/science/virtual_state_task.py` |
| 028e6fad415a381453c836d4ebe8a0058148a2c2ebd0f20870eafc68d5d0a9a4 | 2363 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/launch.sh` |
| a4fb1fc064d41cdcfef1a390a24d7fe28d590140950e740734f1b20a3db216b8 | 41 | `01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/source-commit.txt` |
| b3e231b8065623770047d3f21653eceb867a06fb9a7af1977c21a1eca2ebba3f | 4007 | `01_data_preparation/gdsc-metabolite-query-20260908-0001/code/GDSC_METABOLITE_QUERY_BINDINGS.json` |
| 8b3d441c33651d3ee04e2f7e5f9e80100531aa2b1bd534d0a1645a6038617291 | 19712 | `01_data_preparation/gdsc-metabolite-query-20260908-0001/code/scripts/run_gdsc_metabolite_query.py` |
| 298488a8385681b0391530a6e7dfa55c244c874ccfaa73c3c004e9738c4ccef6 | 33355 | `01_data_preparation/gdsc-metabolite-query-20260908-0001/code/scripts/run_phospho_fewshot.py` |
| 9f9a0fd8cf6e2e02b7188c28299c87cd5cc79f0b8c6825ed48aeb5a302c6e46d | 46938 | `01_data_preparation/gdsc-metabolite-query-20260908-0001/code/scripts/verify_gdsc_metabolite_query.py` |
| 09f9413a3832c6604e8b6b3b46d2088ba665d424c04f35501a0dbc1f7a5bcdb9 | 1352 | `01_data_preparation/gdsc-metabolite-query-20260908-0001/code/src/creed/__init__.py` |
| e0d1c2ba913d54d81d30e6676813b52acbba400e917f3122591f7adc7652f17d | 1106 | `01_data_preparation/gdsc-metabolite-query-20260908-0001/code/src/creed/science/__init__.py` |
| fd65e4e7cc7d93b2c0c253d40a461ebc30f1dfc0def73801210c0d69082ced99 | 919 | `01_data_preparation/gdsc-metabolite-query-20260908-0001/code/src/creed/science/masked_task_kernel.py` |
| f1f2b88d44066c07be975073a590899bbfbefaceb9765874912b6633c3de922b | 9693 | `01_data_preparation/gdsc-metabolite-query-20260908-0001/code/src/creed/science/metabolite_query.py` |
| a4bbbbb861f7b702271db756d5d1ef6c8618a168830586ff54b98d6e8822c656 | 12407 | `01_data_preparation/gdsc-metabolite-query-20260908-0001/code/src/creed/science/phospho_fewshot.py` |
| 591e5646ed940cc61f337c3d389d0931e053112b607da5ad8022d10d658fa2eb | 4777 | `01_data_preparation/gdsc-metabolite-query-20260908-0001/code/src/creed/science/virtual_state_task.py` |
| 5e3d25a2ce82e3e836505782d1b351e115360f42c491aa6fc742368d51c27a82 | 2636 | `01_data_preparation/gdsc-metabolite-query-20260908-0001/launch.sh` |
| c647d4c6b365fcbe1031f91a9ab5a7a803f695031f8f1dd496ef0459d7c73300 | 41 | `01_data_preparation/gdsc-metabolite-query-20260908-0001/source-commit.txt` |
| 91d19ceb82801fa73389b36d6c2e095b518bd872d6d7055d9e0ebe0ed5ace064 | 5139 | `01_data_preparation/gdsc421-20260906-0001/code/scripts/prepare_gdsc421.py` |
| e74fc2dfda38902c0492a2a36087d6e07cc692bba1fceb202161b5aeb96fc9af | 8930 | `01_data_preparation/gdsc421-20260906-0001/code/scripts/run_research_diagnostic.py` |
| 09f9413a3832c6604e8b6b3b46d2088ba665d424c04f35501a0dbc1f7a5bcdb9 | 1352 | `01_data_preparation/gdsc421-20260906-0001/code/src/creed/__init__.py` |
| e0d1c2ba913d54d81d30e6676813b52acbba400e917f3122591f7adc7652f17d | 1106 | `01_data_preparation/gdsc421-20260906-0001/code/src/creed/science/__init__.py` |
| 4e1002435dd007c4ff3562ea65d93291749d05765f1ad7aea75a6df8e549d42b | 134926 | `01_data_preparation/gdsc421-20260906-0001/code/src/creed/science/ccle_multifm_bidir_v5.py` |
| 998bf0012bbc9edcb3accfe975bc6ea412aea2aa7c719322d139a28f656174c3 | 133828 | `01_data_preparation/gdsc421-20260906-0001/code/src/creed/science/ccle_multifm_v5_direct_comparators.py` |
| 62242b0248f24cb8d02a3d2529f2aaa0d39db69009f41ece9ff7257322bfe172 | 825 | `01_data_preparation/gdsc421-20260906-0001/run.sh` |
| 99951c278dc45254e667a1f915be876a23957a283e89e265bc25c4c91e18ec15 | 1391 | `02_protein_graph_and_teacher_runs_20261008/extra-supervision-20261008-0001/code/preflight_real.py` |
| 535e4fb7c0d207fb4af7ffb61dd70b34835b40e1abdd433f5cc417179c31b4e6 | 10467 | `02_protein_graph_and_teacher_runs_20261008/extra-supervision-20261008-0001/code/run_extra.py` |
| 37aadf83f3b5f4001567367e27daaffa7dc76a74b9d645b3073801e1ae31c7f7 | 20735 | `02_protein_graph_and_teacher_runs_20261008/extra-supervision-20261008-0001/code/run_recoverability.py` |
| 850adc639bf0c2c0d1dc0f8b97d956a52a02a93028aac275703dc7f4a49dd47e | 1290 | `02_protein_graph_and_teacher_runs_20261008/extra-supervision-20261008-0001/launch.sh` |
| 42803d501a536689539be1da63e186b2c9f5bbd9b590b73eb71af86982b7f409 | 976 | `02_protein_graph_and_teacher_runs_20261008/extra-supervision-20261008-0001/verification/launch_verification.sh` |
| 790cfb1392f0073bbd76fbfd39c166fff67a51ed7c343560ca0a07e60a7a198d | 5618 | `02_protein_graph_and_teacher_runs_20261008/extra-supervision-20261008-0001/verification/preflight_extension.py` |
| 2529291b62df396c07984b10bda1b7da95ccdb8fd33f306903473fc70f767c49 | 13671 | `02_protein_graph_and_teacher_runs_20261008/extra-supervision-20261008-0001/verification/verify_saved_extra.py` |
| 9dfa8ce52945fdeea17501c83a7661d724830f0a4947c2043965bba26272bd08 | 5112 | `02_protein_graph_and_teacher_runs_20261008/privileged-geometry-20261008-0001/code/extend_geometry.py` |
| 38600c6cc5b8fa254d30c6d548ce5d25ae94b57ea61bc825664b2a0cac2d7066 | 1247 | `02_protein_graph_and_teacher_runs_20261008/privileged-geometry-20261008-0001/code/preflight.py` |
| 6d925ed89052f21761b11eb8aa59aaeb764702c52f533a08c95bda988e717d88 | 10472 | `02_protein_graph_and_teacher_runs_20261008/privileged-geometry-20261008-0001/code/run_geometry.py` |
| 37aadf83f3b5f4001567367e27daaffa7dc76a74b9d645b3073801e1ae31c7f7 | 20735 | `02_protein_graph_and_teacher_runs_20261008/privileged-geometry-20261008-0001/code/run_recoverability.py` |
| 35e2177cb3ea6d68f098220099dbcf031462cafaf152e5d9702341973944a09d | 1001 | `02_protein_graph_and_teacher_runs_20261008/privileged-geometry-20261008-0001/launch.sh` |
| 11c1038965d106239e6c7fb5c602f49eb4abbdce40399e914c012f09feb59889 | 903 | `02_protein_graph_and_teacher_runs_20261008/privileged-geometry-20261008-0001/launch_extension.sh` |
| 019c0db5b2f97eb1c3af799e3ddcec1004fc1a5381e6b427a9ffd324149472ca | 4951 | `02_protein_graph_and_teacher_runs_20261008/privileged-geometry-20261008-0001/verification/verify_kernel_identity.py` |
| 7e84781c736dca514dca1ffb80e4e9e856996d8e97f6c66e9fc2e954f0023fa5 | 4725 | `02_protein_graph_and_teacher_runs_20261008/privileged-geometry-20261008-0001/verification/verify_large_mu_preflight.py` |
| 84e516638a9b8095c86813e5ef4e9d9bb3a89171c703d4d593607dc4fda926fd | 11364 | `02_protein_graph_and_teacher_runs_20261008/privileged-geometry-20261008-0001/verification/verify_saved_results.py` |
| 3b11adab4ef9f812949da2e11b2d8cf73abbb1092ae300906e1bcb94f667c7a1 | 11969 | `02_protein_graph_and_teacher_runs_20261008/privileged-geometry-20261008-0001/verification/verify_saved_results_v2.py` |
| ee2e1d1bb0b9a6deb191e1b9c26d1b3d22a2165972e695546641c9f2c53b7ea7 | 1264 | `02_protein_graph_and_teacher_runs_20261008/strong-teacher-recoverability-20261008-0001/code/new_cell_fold_ids.json` |
| 37a4117bf09fd7e796a2aa06205b8a939279393933817fbe061e32c530155cc0 | 1405 | `02_protein_graph_and_teacher_runs_20261008/strong-teacher-recoverability-20261008-0001/code/preflight.py` |
| 31ada939b9544fe01838791acac2c1531bc816892fdad33c3afa0c922af1733e | 19137 | `02_protein_graph_and_teacher_runs_20261008/strong-teacher-recoverability-20261008-0001/code/run_recoverability.py` |
| c9082ffedd5b5451b48825617b0ae13cb49d41fe2c54650c410f807c256119b1 | 21390 | `02_protein_graph_and_teacher_runs_20261008/strong-teacher-recoverability-20261008-0001/code/verify_results.py` |
| ee2e1d1bb0b9a6deb191e1b9c26d1b3d22a2165972e695546641c9f2c53b7ea7 | 1264 | `02_protein_graph_and_teacher_runs_20261008/strong-teacher-recoverability-20261008-0001/code_v2/new_cell_fold_ids.json` |
| 37aadf83f3b5f4001567367e27daaffa7dc76a74b9d645b3073801e1ae31c7f7 | 20735 | `02_protein_graph_and_teacher_runs_20261008/strong-teacher-recoverability-20261008-0001/code_v2/run_recoverability.py` |
| 1e62688f657cbade3a2c9ae4da6ab7583568704502d1e9b2e84f23d38a69f000 | 891 | `02_protein_graph_and_teacher_runs_20261008/strong-teacher-recoverability-20261008-0001/launch_scenario.sh` |
| d589e4500f6b542d1729926db1ca78fb67b1c7d0fc8c427101ff43da8da189d6 | 811 | `02_protein_graph_and_teacher_runs_20261008/strong-teacher-recoverability-20261008-0001/launch_tissue_v2.sh` |
| 32e7d5949a7bd7e145b44c8eb1bdb8996cb6e149a49cc0dd07679d64cb6341f1 | 33126 | `03_development_runs_20261009/acquisition-probe-20261009-0001/code/acquisition_probe.py` |
| 26990b9c648dc21d27ea7e8f6935f9d7fd5161c68780e770c8729c66544f9eb1 | 9160 | `03_development_runs_20261009/acquisition-probe-20261009-0001/code/make_tables.py` |
| 678449a5579f12c0093b4083cce2b968bfb734beeece8fefb5f2f0b82ba8727e | 46617 | `03_development_runs_20261009/acquisition-probe-rna-20261009-0001/code/acquisition_probe_rna.py` |
| 884b044ab48e662644129542077fa5618bc0e9eea03ba470a4a906aa0e6b7bc5 | 772 | `03_development_runs_20261009/acquisition-probe-rna-20261009-0001/code/run.sh` |
| 473da1622b555c002185c44eb5559502cae330fe80dbbdba9bbcbdbdcbd64f04 | 6195 | `03_development_runs_20261009/acquisition-probe-rna-20261009-0001/dev/dev_fake_root.py` |
| 792b495e4f17aff6f001a225bf0ed570e51ca048cdf9e3012357ca82b4929596 | 18152 | `03_development_runs_20261009/crispr-endpoint-20261009-0001/code/analyze.py` |
| 58f7e0a5125095b1facde16ce8a04b6db1ae70ca33a96076a4aa023670059b56 | 30736 | `03_development_runs_20261009/crispr-endpoint-20261009-0001/code/crispr_endpoint.py` |
| a746906321d7a23a25173f410b7b078ec22e8b8d2c846873855de63accc7d61a | 1204 | `03_development_runs_20261009/crispr-endpoint-20261009-0001/code/run.sh` |
| 06d2d94b0baa176b16655d997d8f9886492bc6feb24a1140df2e5bf2b15bc0bc | 10330 | `03_development_runs_20261009/crosslayer-matrix-20261009-0001/code/analyze.py` |
| c1095b941bdb92b335175291fc5a039f99d7c62b0fdb339e80db2817d11f0da3 | 20014 | `03_development_runs_20261009/crosslayer-matrix-20261009-0001/code/crosslayer.py` |
| 8f30868693e7521b64c3adbe557d788dfa707091a2440023367ee4332f4dcb0b | 552 | `03_development_runs_20261009/crosslayer-matrix-20261009-0001/code/run.sh` |
| 595cc52eb4875e49cd49bc3381f79f74ce35d872e72420a8349e95919650a89a | 4389 | `03_development_runs_20261009/crosslayer-matrix-20261009-0001/dev/dev_fake_root.py` |
| ca2e3edecf0d807aae9d4d70f4330a9e55c97eacf0bd8b9a6dec4ed4d88ca1d2 | 5433 | `03_development_runs_20261009/crosslayer-matrix-20261009-0001/posthoc/posthoc_saved_compare.py` |
| 65cd229d888503967dc41b0c3d1501bebbc0bbb5fa01b66aa941914750a0ade5 | 17187 | `03_development_runs_20261009/generator-curve-20261009-0001/code/analyze_curve.py` |
| bac60ffc984993c0a2c8b7f1c806dabb66bf66da16b93a573c34bd4f45cf9c6c | 27520 | `03_development_runs_20261009/generator-curve-20261009-0001/code/gencurve.py` |
| eea835ec61a71da9ac058876f730cce5151bff25eab3351642211d48cf2c68a3 | 641 | `03_development_runs_20261009/generator-curve-20261009-0001/code/run.sh` |
| 6073a957506d08e095625fd38f24323050c5d666e8a5e1c22841f22810e33f27 | 34089 | `03_development_runs_20261009/generator-fullpool-20261009-0001/code/extract_fullpool.py` |
| 036f9ce7391bf0a2d77cc071d49f58e2b85ef23fc55a29858ce63a602ab26c0f | 371 | `03_development_runs_20261009/generator-fullpool-20261009-0001/code/run.sh` |
| 6f0bb2dd536fb85ed3508184585daba19a4d1c24a4b05020bc0ee30c53c4ab23 | 4556 | `03_development_runs_20261009/independent_recomputation/acquisition-probe-20261009-0001/audit_recompute.py` |
| b54004741aed69aa51461d130a74dd0ed097713ea5876cef47f3824cf609b458 | 4304 | `03_development_runs_20261009/independent_recomputation/acquisition-probe-rna-20261009-0001/audit.py` |
| 76feda159111bccdb21a6861e7dc10e7db175ed0348166b8d069301a5622a9a4 | 13589 | `03_development_runs_20261009/independent_recomputation/crispr-endpoint-20261009-0001/recheck.py` |
| f29d693cb8db0efb5f95b392a9a2c04eb0dae5ee818ab20624cac40e904a382a | 6565 | `03_development_runs_20261009/independent_recomputation/probe2-context-abstain-20261009-0001/auditA.py` |
| f209a65f60d5651e02279ea5ff324627812895b488cbabf7ab36117dad9e2326 | 7232 | `03_development_runs_20261009/independent_recomputation/probe2-context-abstain-20261009-0001/auditB.py` |
| 5b913b4883a5a61e1b350faa89e69be1fb192eadd54edd8010bbb15a7a6d6bd0 | 10185 | `03_development_runs_20261009/independent_recomputation/rna-panel-20261009-0001/audit_metric.py` |
| ca02862d10983fb5535c1bf45ad8f8508e6bdea66a412901615fcba0124261a3 | 5376 | `03_development_runs_20261009/null_baseline_20261009/null_baseline.py` |
| 48570702b1f66c75122ead8fb95a56b7510e6bf2ae66916027071cf644c6dcd5 | 8346 | `03_development_runs_20261009/probe2-context-abstain-20261009-0001/code/abstain.py` |
| 46e139ce2eb92fb962b4bc8092b1d9b84e090365cd4b2962aaa86e851ffff681 | 17118 | `03_development_runs_20261009/probe2-context-abstain-20261009-0001/code/analyze.py` |
| 50a7842e4283b156e52276ad29e336f486dc37c34aeef5392bbd10b0ac1d29d6 | 590 | `03_development_runs_20261009/probe2-context-abstain-20261009-0001/code/job_dryrun.sh` |
| bcbc584a1581d6e72641fcf0077c4f9fa43fd100008ba9293f1898b9bd0dbe67 | 6026 | `03_development_runs_20261009/probe2-context-abstain-20261009-0001/code/make_result_tables.py` |
| 2292ce1c099de2a98acd4be299431ced0f3047dcb615c8fc53e3858773592ec4 | 21531 | `03_development_runs_20261009/probe2-context-abstain-20261009-0001/code/probe2.py` |
| 9b51799c35afdfe415d8ff274d9fe5bfc4693d60f5106bed4fbc060dfc4f28ce | 981 | `03_development_runs_20261009/probe2-context-abstain-20261009-0001/code/run.sh` |
| f9ef4d6d36bfba0eaad5d55c90dc8108bbc1f1e50bb59048d8862d77f9537475 | 16162 | `03_development_runs_20261009/protein-panel-20261009-0001/code/analyze.py` |
| c38955610489c859d7c7b7d9c49de0c6bedf94427ff785f228cd6194d3b4f1dc | 19945 | `03_development_runs_20261009/protein-panel-20261009-0001/code/protein_panel.py` |
| 3560010736e3e4d208ad992a6fe12d7e73e598eb902447d797b1846dd417ee19 | 27670 | `03_development_runs_20261009/pushline-checks-20261009-0001/code/pushline_checks.py` |
| a4c036ace728cd3c056efd7cb28789e99191945a57334247f75ae8ed1a8161f4 | 9987 | `03_development_runs_20261009/rna-learning-curve-20261009-0001/code/analyze.py` |
| 5c307eb86753b89ef8d68e53376ac2dfc993f7a0f34f82392666a044e06d0062 | 2438 | `03_development_runs_20261009/rna-learning-curve-20261009-0001/code/fake_outputs.py` |
| 259484186ae29d3ebb2cb221d9f1fecb87451b569d8b5134f69567e092dbc8d4 | 6336 | `03_development_runs_20261009/rna-learning-curve-20261009-0001/code/preflight_identity.py` |
| 18aaf95daac30e60bb605ecf1a7a6b14389e35a121520cf2ac067415922a13cf | 22166 | `03_development_runs_20261009/rna-learning-curve-20261009-0001/code/rna_lc.py` |
| ba6339eacf1ae8e0c3405af2d2845cb8debc5ba3ad8dc71a3e4ec986983f42b8 | 488 | `03_development_runs_20261009/rna-learning-curve-20261009-0001/code/run.sh` |
| b70681327069cf264981677622a97d6a6ee628bcc34a5772d25b3ecf695cc319 | 1601 | `03_development_runs_20261009/rna-learning-curve-20261009-0001/code/test_inner.py` |
| 68a60254ff7dd77df742bf16ce6c86ea477696e495b2e17faea81a111e79069e | 21673 | `03_development_runs_20261009/rna-panel-20261009-0001/code/analyze.py` |
| 3c1b10843d3c932fbf885c0805c41d1d4f39abb030b4cd1e39e74c25c7f163f4 | 618 | `03_development_runs_20261009/rna-panel-20261009-0001/code/job_dryrun.sh` |
| dea8e99d553382ab693cf03249696761aaa43f6fbcc956d4af215d8b4c8ffa39 | 24099 | `03_development_runs_20261009/rna-panel-20261009-0001/code/rna_panel.py` |
| d5c62f64e88b5227eba499d6474a7d5ebf626e35633a290e861b971b7ca1534f | 1872 | `03_development_runs_20261009/rna-panel-20261009-0001/code/run.sh` |
| 05d4d49e90d71f1d0c1a929bfe2ba556d822307bd30a7f1e28a6ec29088aa6c4 | 3182 | `03_development_runs_20261009/shared-crossdrug-20261009-0001/code/diagnose.py` |
| b3f621ad74409b0f88103c850d8020863a5eb8204126003c890b87a297ccff57 | 28196 | `03_development_runs_20261009/shared-crossdrug-20261009-0001/code/shared_crossdrug.py` |
| b579925124ed5af64e2f3977c3a40857181b8344bc56be037ee67a35f7feda9b | 3928 | `03_development_runs_20261009/shared-crossdrug-20261009-0001/code/test_synthetic.py` |
| 65aa7207aeb9775b5f603625996cd639148cc4dfc6ea28357c699cc4a387a9e5 | 8557 | `03_development_runs_20261009/tissue-kernel-20261009-0001/code/tissue_kernel.py` |
| 5edeb78f4b32816a5252e000dbbe830ae3a89b93a5b028de8527f6db0fbe6471 | 14804 | `04_sealed_set_20261009/independent_recomputation/recompute_remote.py` |
| ae72a12cfe85b63e6b6da01d4b4d867721c8829ba801a81c891395ad43f1cd6d | 21769 | `04_sealed_set_20261009/internal-final-20261009-0001/code/creed_core.py` |
| 50192cc4a0cb9d43c00f92630a49cdebf728603d62c74168a19595a5d84ba8c5 | 55872 | `04_sealed_set_20261009/internal-final-20261009-0001/code/oneshot.py` |
| 24679bce963299660b619049883645d9e94bf0fd8f959145271be2712cb6ae85 | 24271 | `04_sealed_set_20261009/internal-final-20261009-0001/code/sealed_io.py` |
| 72844173d0215fdb2ec74efe056abcd59dd2debcb3f34c8133191e16d6cbf8ce | 305 | `04_sealed_set_20261009/internal-final-20261009-0001/logs/run_dryrun.sh` |
| 414f58c7d7528394dc7db9aa905e284717d35ba8a6915196d67fc3bb4f808cd8 | 287 | `04_sealed_set_20261009/internal-final-20261009-0001/logs/run_freeze.sh` |
| 4f05686a6cd858a720b44ea7fd4904087825cddf5d6b429b40defd6498db854d | 405 | `04_sealed_set_20261009/internal-final-20261009-0001/logs/run_gates_dryrun.sh` |
| d609e2306ee38d284937a7a653607f50663b08f08665a201771954822ef4a16a | 9293 | `05_followup_N6_20261011/settle-stageA-20261011-0001/code/analyze.py` |
| 5c5cddbfc8d2663055cce3940ac1b0bcb010e914dcb85d0f422de7d900c24043 | 2717 | `05_followup_N6_20261011/settle-stageA-20261011-0001/code/analyze_crispr.py` |
| 58f7e0a5125095b1facde16ce8a04b6db1ae70ca33a96076a4aa023670059b56 | 30736 | `05_followup_N6_20261011/settle-stageA-20261011-0001/code/crispr_endpoint.py` |
| c1095b941bdb92b335175291fc5a039f99d7c62b0fdb339e80db2817d11f0da3 | 20014 | `05_followup_N6_20261011/settle-stageA-20261011-0001/code/crosslayer.py` |
| 5b5cb08eb4500425f017abf83c1c4e5933d14744fc841fc532f148744e0cdcfb | 625 | `05_followup_N6_20261011/settle-stageA-20261011-0001/code/run.sh` |
| 5ebdf3525f8ae08d218085699f57f8c3c2f96843758372e6885b3108284a1fbd | 679 | `05_followup_N6_20261011/settle-stageA-20261011-0001/code/run_crispr.sh` |
| f94ab897588ab58cbec2adf4197553ae9b8eb730f8164c4c46563e165cb8dfc6 | 15705 | `05_followup_N6_20261011/settle-stageA-20261011-0001/code/stageA.py` |
| 9a74d2a4288c8f79b2861e69eeba08f82b7a893d5b00f240f262ed4e3d21f397 | 12522 | `05_followup_N6_20261011/settle-stageA-20261011-0001/code/stageA_crispr.py` |
| 0faa58199c5bc68145ab57bdde8ef9dca2b80576c9f823d72807c97e662be8e7 | 3341 | `05_followup_N6_20261011/settle_stageA_20261011/make_table_s15.py` |
| adeba45bed5655f2ffd5d6b91c65ee4bc85feadeb360e704ef02def6d6b141b9 | 7387 | `06_cptac_supplemental/cptac-anchor-20260906-0001/code/scripts/run_cptac_anchored_context.py` |
| 9922104cf21ebf815410f84fe559d3d11274db688502d9d9c3bb0a4e06c00c90 | 7461 | `06_cptac_supplemental/cptac-anchor-20260906-0001/code/scripts/verify_cptac_baseline.py` |
| 629234d3ff08b21e5d0ff9536c5deab2f318ba1f05823c2f897e18056f8c5663 | 5750 | `06_cptac_supplemental/cptac-anchor-20260906-0001/code/src/creed/science/anchored_context.py` |
| 10bd7e7d7954760177848bac97932b744dfb28396b583ad3207706712cb69da3 | 5568 | `06_cptac_supplemental/cptac-anchor-20260906-0001/code/src/creed/science/masked_baseline.py` |
| 60e5580bf810bafcef3aeb4da3585c49404b4ee07ac1c447ae7fa8ef78584996 | 2316 | `06_cptac_supplemental/cptac-baseline-20260906-0002/code/experiments/CPTAC_BIDIRECTIONAL_BASELINE_20260906.md` |
| e1258c356dd487d29b46183c47eeccebccbfc7b8ef72328e90ca5f1420e357a3 | 7289 | `06_cptac_supplemental/cptac-baseline-20260906-0002/code/scripts/run_cptac_bidirectional_baseline.py` |
| 18ec83c04faab5d2e1273c1eb8bb315016d9a1306b8c487cfc2ecd217e8f27fd | 5789 | `06_cptac_supplemental/cptac-baseline-20260906-0002/code/scripts/verify_cptac_baseline.py` |
| 10bd7e7d7954760177848bac97932b744dfb28396b583ad3207706712cb69da3 | 5568 | `06_cptac_supplemental/cptac-baseline-20260906-0002/code/src/creed/science/masked_baseline.py` |
| 7a8b1358e1afc7bcf8afd59b0bb17dfedde27b1a35f7053f311d8a304f735cb0 | 2622 | `06_cptac_supplemental/cptac-blocks-20260906-0001/code/scripts/prepare_cptac_blocks.py` |
| 055f5e1e7774916a4ad98b700d553b24aff4f6ccd932515c66607f8983f9d34a | 10603 | `06_cptac_supplemental/cptac-blocks-20260906-0001/code/src/creed/science/cptac_blocks.py` |
| 411236f9052c7f507b0ddd1149c5813cddce69ec8fb85a118a75aaebe7f9e0c7 | 3985 | `06_cptac_supplemental/cptac-domain-experts-20260907-0001/code/scripts/prepare_cptac_three_domains.py` |
| 59f5937fa3bc80ce5e91c308f99bb5bee7c8276ac2eeb0801c211aa3265c9e8a | 7564 | `06_cptac_supplemental/cptac-domain-experts-20260907-0001/code/scripts/run_cptac_domain_experts.py` |
| 09f9413a3832c6604e8b6b3b46d2088ba665d424c04f35501a0dbc1f7a5bcdb9 | 1352 | `06_cptac_supplemental/cptac-domain-experts-20260907-0001/code/src/creed/__init__.py` |
| e0d1c2ba913d54d81d30e6676813b52acbba400e917f3122591f7adc7652f17d | 1106 | `06_cptac_supplemental/cptac-domain-experts-20260907-0001/code/src/creed/science/__init__.py` |
| dcf581e8abdec3ce240aae960e910cdaf74d92ceb4c562ea4964af15f4a99c64 | 5885 | `06_cptac_supplemental/cptac-domain-experts-20260907-0001/code/src/creed/science/anchored_context.py` |
| d815542a5a36bea2b65c22cd71a9b2f7bae379597aeb7c6a268f9164ed9567d4 | 6032 | `06_cptac_supplemental/cptac-domain-experts-20260907-0001/code/src/creed/science/cptac_transfer_inputs.py` |
| 7f79e2797c7c95d8c1be0baab24152fd52cbd7b898be443129988701b4fc93f8 | 16355 | `06_cptac_supplemental/cptac-domain-experts-20260907-0001/code/src/creed/science/domain_experts.py` |
| c5bf00ac071cc58e67f414011dbd191464d892b022388d7153354f046c526bb8 | 5607 | `06_cptac_supplemental/cptac-domain-experts-20260907-0001/code/src/creed/science/masked_baseline.py` |
| fa6d7f3f6668fbbbe638d8088648d7856d0ef32a8440f80266a6f80f09985be5 | 757 | `06_cptac_supplemental/cptac-domain-experts-20260907-0001/run.sh` |
| 664fbb5fe9b566da27e4d9a5430323be2ec841822a32178441f6a143ac019841 | 6605 | `06_cptac_supplemental/cptac-full-source-direct-20261008-0001/run.py` |
| 6eccea3ccae9f911278ece4f43479b2a5e6f98ee66fa2b458ce4943e9f29ad23 | 7818 | `06_cptac_supplemental/cptac-full-source-direct-20261008-0001/summarize.py` |
| fa95688ca211b1f26f6cd830b648a109ffebdf64f88a1b190197fc5af0eaa298 | 7111 | `06_cptac_supplemental/cptac-source-shift-20260907-0001/code/scripts/check_domain_expert_outputs.py` |
| a770c5f97da72994f3b08f23626403121ecd3dce2d7e34aa9f4c1b4c857e4845 | 2188 | `06_cptac_supplemental/cptac-source-shift-20260907-0001/code/scripts/check_domain_source_only.py` |
| f7bb5fefb5a369dee8dac9d8e89910c3c0f4f22649664aa15d98f88dee8d2eea | 1793 | `06_cptac_supplemental/cptac-source-shift-20260907-0001/code/scripts/predict_domain_translation.py` |
| 411236f9052c7f507b0ddd1149c5813cddce69ec8fb85a118a75aaebe7f9e0c7 | 3985 | `06_cptac_supplemental/cptac-source-shift-20260907-0001/code/scripts/prepare_cptac_three_domains.py` |
| d4cc1e56c3fbbd2f99a9827590d090884459038f34e92141f819b606b29763a6 | 10275 | `06_cptac_supplemental/cptac-source-shift-20260907-0001/code/scripts/run_cptac_domain_experts.py` |
| 09f9413a3832c6604e8b6b3b46d2088ba665d424c04f35501a0dbc1f7a5bcdb9 | 1352 | `06_cptac_supplemental/cptac-source-shift-20260907-0001/code/src/creed/__init__.py` |
| e0d1c2ba913d54d81d30e6676813b52acbba400e917f3122591f7adc7652f17d | 1106 | `06_cptac_supplemental/cptac-source-shift-20260907-0001/code/src/creed/science/__init__.py` |
| dcf581e8abdec3ce240aae960e910cdaf74d92ceb4c562ea4964af15f4a99c64 | 5885 | `06_cptac_supplemental/cptac-source-shift-20260907-0001/code/src/creed/science/anchored_context.py` |
| d815542a5a36bea2b65c22cd71a9b2f7bae379597aeb7c6a268f9164ed9567d4 | 6032 | `06_cptac_supplemental/cptac-source-shift-20260907-0001/code/src/creed/science/cptac_transfer_inputs.py` |
| d00f2c0f5cf2d1c09837eeced2c0f00ec9f48a12ccb5832e6599b7196348299f | 18900 | `06_cptac_supplemental/cptac-source-shift-20260907-0001/code/src/creed/science/domain_experts.py` |
| c5bf00ac071cc58e67f414011dbd191464d892b022388d7153354f046c526bb8 | 5607 | `06_cptac_supplemental/cptac-source-shift-20260907-0001/code/src/creed/science/masked_baseline.py` |
| 41abebd89a2456064a61ec81f576bc69b723523993b3504daf4adc12d2d16857 | 892 | `06_cptac_supplemental/cptac-source-shift-20260907-0001/run.sh` |
| 411236f9052c7f507b0ddd1149c5813cddce69ec8fb85a118a75aaebe7f9e0c7 | 3985 | `06_cptac_supplemental/cptac-three-inputs-20260907-0001/code/scripts/prepare_cptac_three_domains.py` |
| 09f9413a3832c6604e8b6b3b46d2088ba665d424c04f35501a0dbc1f7a5bcdb9 | 1352 | `06_cptac_supplemental/cptac-three-inputs-20260907-0001/code/src/creed/__init__.py` |
| e0d1c2ba913d54d81d30e6676813b52acbba400e917f3122591f7adc7652f17d | 1106 | `06_cptac_supplemental/cptac-three-inputs-20260907-0001/code/src/creed/science/__init__.py` |
| d815542a5a36bea2b65c22cd71a9b2f7bae379597aeb7c6a268f9164ed9567d4 | 6032 | `06_cptac_supplemental/cptac-three-inputs-20260907-0001/code/src/creed/science/cptac_transfer_inputs.py` |
| c5bf00ac071cc58e67f414011dbd191464d892b022388d7153354f046c526bb8 | 5607 | `06_cptac_supplemental/cptac-three-inputs-20260907-0001/code/src/creed/science/masked_baseline.py` |
| e5a118f466bde6b9e65b2d2bf417c12b1f25e6ae35b590d0dd59b6376e58aa07 | 1583 | `07_figures/data_preparation/20261009_figures2_6_v2/r/figures2_6/run.sh` |
| 02418ad8c732dcc9071c098183be6a97694aa01cf9266960d9989ae7f3e844e7 | 3461 | `07_figures/data_preparation/20261009_figures_v3_cell/r/figures2_6/prepare_cptac.py` |
| 3f6cfcc55d30541512cdf29998411ed21fdea4541f182a32d55acfa1710f56c5 | 8517 | `07_figures/data_preparation/20261009_figures_v3_cell/r/figures2_6/prepare_fig2_v2.py` |
| 3b1182b9afa60bac4a7c8b1a93384fcc52829f330c5b3be6f640c34f90fdbf35 | 11829 | `07_figures/data_preparation/20261009_figures_v3_cell/r/figures2_6/prepare_fig3_v2.py` |
| 56e949051cfca54333392316ab31b31de2332838a45d513f74f8263f8e35e61f | 13953 | `07_figures/data_preparation/20261009_figures_v3_cell/r/figures2_6/prepare_fig4_v2.py` |
| d0ffeabfcbc08bb31cdda6bde6712d49119599fa773e80f42600743a2803e6d1 | 15920 | `07_figures/data_preparation/20261009_figures_v3_cell/r/figures2_6/prepare_fig5_v2.py` |
| f4edfdbec04459ad5d461df0ae6b6decc526e9778372146f65eb20e752de86f0 | 12336 | `07_figures/data_preparation/20261009_figures_v3_cell/r/figures2_6/prepare_fig6_v2.py` |
| 7bf115f11151259f937d90a66805209da201a60171a5cb34575818dca0e56cd6 | 23672 | `07_figures/data_preparation/20261009_figures_v3_cell/r/figures2_6/prepare_fig7_v2.py` |
| c9eab6b2870846ea03f7659753c7d77e06fd5b016722aa30168bb761d82fc267 | 34602 | `07_figures/data_preparation/20261009_figures_v3_cell/r/figures2_6/prepare_fig8_v2.py` |
| 00df6ba689db47646a70ac10b631d5a7dd1a40e79672d490339914da9b2c2a8e | 24532 | `07_figures/data_preparation/20261009_figures_v3_cell/r/figures2_6/prepare_figS1_v2.py` |
| 6dd5f76d0edcb4e57070cebf0ea61eda87aee722932f9dd5648d5c0ff0f7487b | 12975 | `07_figures/data_preparation/20261009_figures_v3_cell/r/figures2_6/prepare_gdsc.py` |
| 8ad94c6adf369273866a725ee268d9a4076a56c92186d21f07ed1b9c8b8f2d8a | 1458 | `07_figures/data_preparation/20261009_figures_v3_cell/r/figures2_6/prepare_protocol.py` |
| fdd7cdbca7dfdfb929cc407cb0523b850af2aaf8dec2ba8e58bd160c0d571b1e | 15200 | `07_figures/figure1/20261009_figures_v3_cell/r/cell_style.R` |
| 04b27ce7bfaeb3f2826904626424d8a7a54131978499d7c9f15b5792fe339a8e | 2601 | `07_figures/figure1/20261009_figures_v3_cell/r/figure1/A.R` |
| 7be0907b0f535693dc2fc068c4b5ba6fe59869db125e61dbdaab8d21d415f221 | 1517 | `07_figures/figure1/20261009_figures_v3_cell/r/figure1/B.R` |
| 9ad19a160385fcd60c08badc070ba18ff9d57cbae58efc11828897ff3fa66211 | 1639 | `07_figures/figure1/20261009_figures_v3_cell/r/figure1/C.R` |
| 717001bfe4196f773eb6aa904417a95e36232109427620796ede6984d2b3a8f3 | 1025 | `07_figures/figure1/20261009_figures_v3_cell/r/figure1/D.R` |
| 313af50f519894c47b0d33707096593f93ce1a12e4212c9fb90357d2c42e4246 | 1237 | `07_figures/figure1/20261009_figures_v3_cell/r/figure1/E.R` |
| 854e87313c33475d786cd4c48327217d2d42b40ab2d648fc1b0e05b269735f31 | 2540 | `07_figures/figure1/20261009_figures_v3_cell/r/figure1/F.R` |
| 0aeb7180fa2db22c11a93cdf99a7341da765d92ebb7c55b4f2650f337c3caa3d | 2058 | `07_figures/figure1/20261009_figures_v3_cell/r/figure1/G.R` |
| 03bab2e9517b9bb43e967558c3314ef9e376c87ca737207e1406e2c9e184a2ed | 1520 | `07_figures/figure1/20261009_figures_v3_cell/r/figure1/H.R` |
| 60ca938e7da89ce0b444c87606f75c11f06cca6db8dad4cf9893571876e93e0c | 353 | `07_figures/figure1/20261009_figures_v3_cell/r/figure1/_tools/crop.py` |
| 10530c9068cfdbb01b3938e25d46227804a60daa6ef37f6637fc4d24e2d01426 | 763 | `07_figures/figure1/20261009_figures_v3_cell/r/figure1/_tools/extent.py` |
| 519e21c183cfe59a119c3efa5df79c275fc30a0a48ed3f4fbd0c8b82a6712a37 | 557 | `07_figures/figure1/20261009_figures_v3_cell/r/figure1/assemble.R` |
| b9f9902338df81077e99168a29dfd23d3935732b72116f3a3f7cccb3bf5136d1 | 3544 | `07_figures/figure1/20261009_figures_v3_cell/r/figure1/check_delivery.py` |
| 46c5983d08b730b3266831a396f9cee9c466e8bbef860ce41f4311a879e427b3 | 2260 | `07_figures/figure1/20261009_figures_v3_cell/r/figure1/common.R` |
| c7fed02b0c79f656c63becb376626d1c5f5ee532f0f98c0ca522b0ff6a476478 | 7950 | `07_figures/figure1/20261009_figures_v3_cell/r/figure1/prepare_data.py` |
| b0c09da9196f97dd7c1fc6c3c6d8dcfaa32074aed256ffbf3ac6b03a114e9031 | 715 | `07_figures/figure1/20261009_figures_v3_cell/r/figure1/render_panels.R` |
| ae48e095933bc2c2df5006776d96d28dc7d80a1cb9cd25328467fc9f7732ef6f | 934 | `07_figures/figure1/20261009_figures_v3_cell/r/figure1/run.sh` |
| 314937cdbbc0886e9bc99dd4d1a3204ecd24bfa50ab60a9ce1591f04e920ee4a | 5036 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure2/A.R` |
| 1115ddd80c8d90788de87fc401ed5ea7585a018057d1842617db605ebf3ed4e7 | 4639 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure2/B.R` |
| e0c155321fc17234d23dde600a447ce499e2c5648377218bc130b91f2542520c | 7265 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure2/C.R` |
| 697dc3dc7921cea139e54a2bc5744b18de5599379e0afabbebb4f6c30acfec4b | 3701 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure2/D.R` |
| 7ce550b5d5c0724a4a0d377b5343fc726cd9dcf36e4af3f4223e446f67e2ee54 | 3583 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure2/E.R` |
| 437833de89d7f287f4edd8ca275c864d587eb19d256841c69fecb99557b9bbe8 | 2622 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure2/F.R` |
| 9347f3958ef9a5c47fe440717792335adedf1ca7502d44ac94352c6be7ec18f8 | 2695 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure2/G.R` |
| 74c39ac2cf60fe5e86dffb240e962f34639fbf71b3a896df2fbe3243b2accc0d | 3606 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure2/H.R` |
| c99fa39f62fc45517395b0a2069bf7bee6b9a5f1e50338cde01434553b668470 | 2021 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure2/common.R` |
| 8f594502183032f46bff4bf076f6f50944248548f9350e7ba89e8d8eb5f73de1 | 248 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure2/layout.csv` |
| 5b879c05baf3febb61d3d2f34b5cc2e43a9066ffc2369b3d144697046f953708 | 6429 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure3/A.R` |
| 8ca6a1d61778006df6a9324106eb2d1b348d9cf20378baa948fb63688dfffa31 | 3691 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure3/B.R` |
| 7eee6e60c824d504b997062e0949ce3d3ca5c201f830f43249ea73bc209c7128 | 3005 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure3/C.R` |
| 18390eb8faead27441d6a2cab20c3f9f65eecb12e0cbb3adcdbc8be6fecc6021 | 7021 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure3/D.R` |
| 3f7b80343228e8da402ffbe3bd2219410ef1548b6ac8bb2e7c99d057f493b37b | 6962 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure3/E.R` |
| ba4941b94cc024e36ed104cd3253f7ab976b7c7a79d1c5147520d17ca64216e0 | 2933 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure3/F.R` |
| 240f7e20ef25b0e1ba4a41398fd1f18cfe8f1aa8213a5782ab8e91084cd2e69a | 3578 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure3/G.R` |
| eafc6e8be1d0e24cfd6185198ac70e22be0555355c205bf2490af4667d9d19ce | 3743 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure3/H.R` |
| 6f5537c19288c8cdfe16175ce966693bb016aadcd834a3c0b945e6ebb1dcc675 | 4136 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure3/common.R` |
| 5e53cd22ebd1218471d05ba4a0c7411ca6e4439aaeb694e7f7d6c28cdbe79acf | 191 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure3/layout.csv` |
| 34853cc3999ca3638b8bd4f1052f5fe0aba381cb3db0c909cbec037948570f88 | 4042 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure4/A.R` |
| 185358ce6ee6164bdcc151192140970a019c22cee38a0748be10732e6a1c4709 | 5436 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure4/B.R` |
| 7fe6058226f0158ed72c807364cfd1397257bf708a15c321e4d27ca0ec30ed7d | 3440 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure4/C.R` |
| f4f2193a0109a6d310b7a97478e412eb1a7bc9d70af35a72aa14f0287ea8287a | 2373 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure4/D.R` |
| 2baf952fb1f4aa504b8ffbe83a62162e4b287e755fc3356a0fbb7fc3f6446ca3 | 2901 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure4/E.R` |
| 072437777a32be3b5916d16ace2dae699c661446176028393de779a21b22bb23 | 1984 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure4/F.R` |
| aba43f36fda05633d940391ef6c9dee76852860bb76f7ef7ffd39706b8fdd1d4 | 4876 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure4/G.R` |
| 66e40d30c49e73c8e15dbbb0686ab7d25e3e05e4e6fc3efa71380113d5efee67 | 8477 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure4/H.R` |
| dc4f2cac77cc0d24bfeaeb8a8e0e2b1e361bdc0414e739f483ae9a7fa6f97040 | 2076 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure4/KEY.R` |
| 18ea031eef26b9f5d3a399204fb9a57028a0de30fb43f241aa7bda400db45e82 | 3529 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure4/common.R` |
| d87bbb8f7d83304e07faeb2b2183d91d2cf5b85407895c7fd13e654ac8284b44 | 212 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure4/layout.csv` |
| 61fa24942f4edbd78b9dcc4316f8dc03d583a15ac3aebd6c91e7f860727ecf08 | 2467 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure4/prep/extract_probe2_cmp.py` |
| 51c18f478e6791812d51e643a69e38e9ac780a973c9f3b810d76167c748dc7b4 | 2752 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure4/prep/extract_probe2_fold.py` |
| 54a0ef89563ae744c05709618ad64dd4297333293a168351da4f460549dae095 | 5506 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure5/A.R` |
| 85351d46b7475c13af5207ba185330d7009604e0c43de9959e5a769c84a31a38 | 4143 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure5/B.R` |
| 88bbb3c2915e4dbb8a9d7f5fe76f328265b04cc3921141e42f052a681ed8d139 | 10177 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure5/C.R` |
| 0d2c26a05d8f30a9fef130441549427a2238e606e980558036abe061961505a3 | 3168 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure5/D.R` |
| 8117aabc285266d631263a4376f770c7a6c80d5f076a5f3c34ee55007b324135 | 2401 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure5/E.R` |
| 4ea7145b528ded30349af9c750eccade96f5f130cda3311520a60f33761c43a1 | 1960 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure5/F.R` |
| e439e0b9205143be1099038d5e06e65d2d9d5956500282eef8dab0dc57248fa5 | 4437 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure5/G.R` |
| 438870eed94929c87827df7844c5a66f2e13d56a2fa3ed78fdc9bcf73dff77ce | 4926 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure5/H.R` |
| a52454c627f5a38c31c6a5331edb137e1bc06193c25963f1504b99013f64119c | 3260 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure5/common.R` |
| 8c18d313114e2717916f9c4022e51a0f0083225bff1f4db3aa8ffed9d477412c | 202 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure5/layout.csv` |
| cfb267b7ca7457c5b193954d365051fcd157f554fee2675d511e2489a87ce146 | 5370 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure6/A.R` |
| 64a3323e3c0b0b70342cc7a57cf3a7e497c85a670f3403d297f4c3391c08ec95 | 7922 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure6/B.R` |
| 0f39a629b17b288984128b708d8b4adbdc99a7f0b172193ce7d4365dd5c59f37 | 3787 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure6/C.R` |
| cdf4b2564dfb2e55246ef18ec20489b65c7d500647ad5aea9b462060f9faffd7 | 6443 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure6/D.R` |
| 81e1ff2f166ce4f2381b6be8911c6549803c7a3adfc38dae9ac3c14f93d6b259 | 3354 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure6/E.R` |
| 439a9c18d00188ae89a14eea2b217c2adf9751dacaba3e0b5016d0be3c67eee6 | 3530 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure6/F.R` |
| 3d7601e8dfc3fe1acab78f02a4a26b5c779d740f596fa3a1d88285411e4619c7 | 7072 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure6/G.R` |
| f6dc4757d863d1e1ccb3f8b5f6535ba5237989f3c82cc7471084a0a64eed732c | 4906 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure6/H.R` |
| c81ed04d46639abea25e905eaca6d2b666a1dc639d517499b68c1f674d45867e | 6360 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure6/common.R` |
| 7811445d959342ea3f2e99bc30b9ee966032b306859c997a9cefe143199b61cf | 192 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure6/layout.csv` |
| 811422d016dcc4fb7a4d98663f5be6f537343be0295fc2c842c975955e8fb6c7 | 3122 | `07_figures/figures_v4/20261010_figures_v4_six/r/Figure6/prep_dev_folds.R` |
| b0f34f01a3b427313b446b48b0b7a567e709271c68313e62aa0dc018937f43cb | 6040 | `07_figures/figures_v4/20261010_figures_v4_six/r/FigureS1/A.R` |
| 3057060bd15fe2a9dadc4ef18dbe5e156a85f35f985d08f98703c29bf661d069 | 682 | `07_figures/figures_v4/20261010_figures_v4_six/r/FigureS1/B.R` |
| c0e93d303968bb7a845ae4941ee3bfc70b6dd252a36db305366a9d2ee31f6e07 | 576 | `07_figures/figures_v4/20261010_figures_v4_six/r/FigureS1/C.R` |
| 5e117252bc43e76b98fad5aaae6b91b61ff0567500c0173e3e13930f139eda74 | 1984 | `07_figures/figures_v4/20261010_figures_v4_six/r/FigureS1/D.R` |
| 42684de63fe5c36c2abc59321c30a31bd36f6bec4f2fbb87be9c2d246e13c1f6 | 1783 | `07_figures/figures_v4/20261010_figures_v4_six/r/FigureS1/E.R` |
| f5be50cb483511d03370a48d7cd7e50e0e9161b82abc7c408d5efc1444419113 | 1973 | `07_figures/figures_v4/20261010_figures_v4_six/r/FigureS1/F.R` |
| c3f4b39894ad17bc8670ea4741db8575ad0cc98ed42d0dade60ddb16d8e2e2ab | 701 | `07_figures/figures_v4/20261010_figures_v4_six/r/FigureS1/LEG_ARM.R` |
| 9e0579b4b25a3d6ff784419e7be524a1319d241f7d27e868e56ca70e87f5223f | 7565 | `07_figures/figures_v4/20261010_figures_v4_six/r/FigureS1/common.R` |
| 81be36d5c318ac019c6b4ee95b4da0412ae32cec6a3bb05d13879a70489b9e96 | 179 | `07_figures/figures_v4/20261010_figures_v4_six/r/FigureS1/layout.csv` |
| 37ab13e09277d328c07625adf7c9591de575b046d41a278257c4940501e24e27 | 1468 | `07_figures/figures_v4/20261010_figures_v4_six/r/FigureS2/A.R` |
| 9925713880c9aa04471ef8a05e2f7f29ce6e6f1ef69f311187dd5f4092c9350e | 1515 | `07_figures/figures_v4/20261010_figures_v4_six/r/FigureS2/B.R` |
| 6f4f2eb49769a471bf3eb616a7d08bce113d874879bb298c84e569092abbca89 | 2198 | `07_figures/figures_v4/20261010_figures_v4_six/r/FigureS2/C.R` |
| 6e15db2facc64aea08a15594eb630b0acce0daf0c6d4dcf914d6ce7d04aeedc0 | 1620 | `07_figures/figures_v4/20261010_figures_v4_six/r/FigureS2/D.R` |
| cea053782dbc0dff3fe293ea183919048eee09f7f2d620a55a9d7c265533a3ba | 1381 | `07_figures/figures_v4/20261010_figures_v4_six/r/FigureS2/LEG_AB.R` |
| 1d13dc33b13daeb714908b5c914e6c19ba8f590114e086c80efae8e536baa583 | 2758 | `07_figures/figures_v4/20261010_figures_v4_six/r/FigureS2/common.R` |
| c8fffe61b3ad9d7adcd66c0adbd8df50f5f59d6a540f63ff9eafb6849eb5402c | 149 | `07_figures/figures_v4/20261010_figures_v4_six/r/FigureS2/layout.csv` |
| 846ba33b99c4ebf02ca755d42754c7149c0f607acf5cd104b06b9e36c9b07f84 | 2691 | `07_figures/figures_v4/20261010_figures_v4_six/r/FigureS3/A.R` |
| 0873a8853de93205113e13863ee0854c3d272db6d4661201a3bda22aeab5549d | 2724 | `07_figures/figures_v4/20261010_figures_v4_six/r/FigureS3/B.R` |
| f01ff949f51843c3d6af034c87b4bc1e262da5604cf118d2c5b4974ec096eacd | 93 | `07_figures/figures_v4/20261010_figures_v4_six/r/FigureS3/layout.csv` |
| 63b90a4831b05645b98e517de82e3b63beb94b8c951c799e0b77c53a6b46d2da | 617 | `07_figures/figures_v4/20261010_figures_v4_six/r/FigureS4/A.R` |
| ec9bcc9a36fa41660de2ebe6ccd5727efef2520ae70b8c3f6275257db4b615a2 | 200 | `07_figures/figures_v4/20261010_figures_v4_six/r/FigureS4/B.R` |
| 7441358bd7a028f4ba34bcd638179e3bdb3a2a5e66d5288a76bdf1c8b8fdbc71 | 1148 | `07_figures/figures_v4/20261010_figures_v4_six/r/FigureS4/C.R` |
| 16ea9f2a733f5ce21373d5f678eca3d2693411992622af504cd4275f98969dc9 | 2568 | `07_figures/figures_v4/20261010_figures_v4_six/r/FigureS4/D.R` |
| a1b595965b4f9d394757c5c5e1e8b0d1337b73e873da1cfd96332944736dd9dd | 140 | `07_figures/figures_v4/20261010_figures_v4_six/r/FigureS4/LEG_AB.R` |
| 4b74ecdfd80c5c66ddf6f23fd1cf0d861821a66b58ef8f89ddccc458bbf5999f | 6446 | `07_figures/figures_v4/20261010_figures_v4_six/r/FigureS4/common.R` |
| d37fb5a38ce4369ca62c26f4d5e545f795396708452c933f146664e75a8d1e8b | 142 | `07_figures/figures_v4/20261010_figures_v4_six/r/FigureS4/layout.csv` |
| 594d96566080705d13e41ccd8b4f4cf56dd42f41186c82113d11ac69b087cdc4 | 3183 | `07_figures/figures_v4/20261010_figures_v4_six/r/FigureS5/A.R` |
| 3259b055cea39d3217389cdfc801f0494e92fdc13a39904ee7389c55e6bb8107 | 1632 | `07_figures/figures_v4/20261010_figures_v4_six/r/FigureS5/B.R` |
| b58728b0d3c1269d772359eaca7b46aab2db36ba1b116cbdf6f32ebea51bd594 | 1837 | `07_figures/figures_v4/20261010_figures_v4_six/r/FigureS5/C.R` |
| 710cd892acd852228b2bdb807f83d569968bc661b8c3a31adfeca6569c70fab2 | 1548 | `07_figures/figures_v4/20261010_figures_v4_six/r/FigureS5/D.R` |
| ace871ddcdbc5a1257860ed94bf65ba5b767dbd1e63a686c5259595408894c3e | 1104 | `07_figures/figures_v4/20261010_figures_v4_six/r/FigureS5/E.R` |
| 103241adfe6983b43bac10c9fc906ccd4d97954eb89834aaed82a9059ede457c | 1098 | `07_figures/figures_v4/20261010_figures_v4_six/r/FigureS5/F.R` |
| 2fc816c37730ad901f01b1ff489fffacfd3c4c3d5c91f87fa5c96ec6ee1d7106 | 3793 | `07_figures/figures_v4/20261010_figures_v4_six/r/FigureS5/common.R` |
| f2589e175ea5917a266242ab5e0aa2db2441ce81ef90898766bf8e92c5830463 | 167 | `07_figures/figures_v4/20261010_figures_v4_six/r/FigureS5/layout.csv` |
| 191e8ec6f4b9b7d4e1234aecd3ca6ea6b5394a180f1df40fac93a9f8e2d7ff8d | 5976 | `07_figures/figures_v4/20261010_figures_v4_six/r/assemble_v4.R` |
| fdd7cdbca7dfdfb929cc407cb0523b850af2aaf8dec2ba8e58bd160c0d571b1e | 15200 | `07_figures/figures_v4/20261010_figures_v4_six/r/cell_style.R` |
| 328b4413319f681fd787c4884331f356822fcd44157c1622dfc5cb43d0b51bfa | 3306 | `07_figures/figures_v4/20261010_figures_v4_six/r/render_v4.R` |
| 33087e79d9dfcf7ade6667a37ed27774fe7f3e620d0ac94934e587defb61dadc | 10290 | `07_figures/figures_v4/20261010_figures_v4_six/r/v4_common.R` |
| a1ce47b2b33a514d517e2466d9605690e4c8c96e70d7354ef7e51c5aefa93030 | 9987 | `07_figures/graphical_abstract/20261010_figures_v4_six/graphical_abstract/graphical_abstract.R` |
| ee5f339e1a42b974b512066b8fe161295518c19100afe829143d838cd62f19c5 | 38276 | `08_package_builders/submission_CellSystems_20261011/_tools/build_cell_docx.py` |
| 91bbefffcf63826c02f8ddefb8a22dfcffc7bae40bb072157b6378c04045a8d8 | 57106 | `08_package_builders/submission_CellSystems_20261011/_tools/build_data_zips.py` |
| 2a85d2f510bb8827dfc6c2d42e0a8d86c0e6cc307546af4132eb3ee1b6c829eb | 33535 | `08_package_builders/submission_CellSystems_20261011/_tools/build_docx_v18_original.py` |
| 3222ba0fc09d941eaa6e07875487d13097773e23f8b5bc4912591f522bc119d9 | 46043 | `08_package_builders/submission_CellSystems_20261011/_tools/build_methods_s1.py` |
| 5ec4d3e93a05e0b8a1651b0e5aa919c0140581071f7100d25d5d7427a79d7ae6 | 8780 | `08_package_builders/submission_CellSystems_20261011/_tools/build_si_pdf.py` |
| c1ce14546a8b0a7bd6dcc1801e93b282dd28a5db9fe050c149bebd0cf04a2db5 | 3085 | `08_package_builders/submission_CellSystems_20261011/_tools/build_submission_zip.py` |
| cadcb2effe6935013a24ad297a2b6c9e7f19b4ff6d0520c8b28e09db3ad040db | 19273 | `08_package_builders/submission_CellSystems_20261011/_tools/build_xlsx.py` |
| 05088d3d80c4cd9a9057ed6c4993b67a570e9f3b08c008d16e0e5f6574606fff | 1886 | `08_package_builders/submission_CellSystems_20261011/_tools/html2pdf.swift` |
| dfd6017c9726065298da02ca7f1fd14a5da8838970a59e071295baa29d375e40 | 264 | `08_package_builders/submission_CellSystems_20261011/_tools/pages_export.applescript` |
| 13718cec45bafa907ba193d0e0c1233219a34a8c79a2fbb69b8c3c5e58682004 | 1262 | `08_package_builders/submission_CellSystems_20261011/_tools/render_check.sh` |
| 7ddb3358f7320118f39256c0b4aed210bd840cac82300a3306be9a020a1ec763 | 558 | `09_prospective_test_20261022/prospective-lowdim-20261022-0001/code/FROZEN_SHA256SUMS` |
| c4a78e643785d55973c462efed5cc8b7f7028b13bb6c0c7546adb64862231493 | 20769 | `09_prospective_test_20261022/prospective-lowdim-20261022-0001/code/analyze.py` |
| ff5a5d3381bd34dcbba71b86e7db1dfef2503afc3c08b3ce4372972f1ca01958 | 29279 | `09_prospective_test_20261022/prospective-lowdim-20261022-0001/code/lowdim.py` |
| fa3f9113896bbd26cf47c35d78f9981ee5bb76081b9cce1a33c53318c143f223 | 2970 | `09_prospective_test_20261022/prospective-lowdim-20261022-0001/code/preflight.sh` |
| c30dd39855affd9dfe2ccb9d6abbe2fb6fcd8cfb41de31a7e886ee7d06794161 | 17065 | `09_prospective_test_20261022/prospective-lowdim-20261022-0001/code/prepare_gdsc1.py` |
| f92010d3da6fcb98c58298f0aa0b21ca4eb4118835a490c5cf842d2ccfe893b8 | 1776 | `09_prospective_test_20261022/prospective-lowdim-20261022-0001/code/register.sh` |
| 626755aa768e63b20914b651d3f0377dcff4a93d03ed1708119df7851471c14b | 4093 | `09_prospective_test_20261022/prospective-lowdim-20261022-0001/code/run.sh` |
| ef014e460475a7b47baf6445d1c993b30aa056ffc6295943447ee675595de893 | 2548 | `09_prospective_test_20261022/prospective-lowdim-20261022-0001/code_addendum1/resume_after_addendum1.sh` |
| 3de4f8f1ba3f1277426fad78788d1823a2541ee2ac4fb788a85c83759b12214d | 4127 | `09_prospective_test_20261022/prospective-lowdim-20261022-0001/code_addendum1/synthetic_test.py` |
| de422103aabdddbdb9e0430b96f8ab2b3104862dbce74415992eefc5f923d627 | 2106 | `09_prospective_test_20261022/prospective_lowdim_20261022/figure_2H/build_fig2h_controls.py` |
| 897e37d00d258f2e112ca01f6d507a900d92498d8586bbdab5990ae8cdc4c89a | 87961 | `SOURCES.tsv` |

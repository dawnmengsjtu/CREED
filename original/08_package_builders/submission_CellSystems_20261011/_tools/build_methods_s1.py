#!/usr/bin/env python3
"""Build Methods S1 (08_Methods_S1_code.zip): the CREED analysis and figure code, related to STAR Methods.

Source: the verbatim code mirror ~/Desktop/proj/creed/code_methods_s1/ (one folder per component; inside it the
run or figure directories of the compute server with their relative paths unchanged; SOURCES.tsv gives the origin
and SHA-256 of every file at the time it was copied). Before zipping, this script
  1. refreshes the package-builder component (08_package_builders/...) from this _tools folder, because the
     builders change with the package (code files only: .py .sh .swift .applescript);
  2. checks that every mirrored file matches its SOURCES.tsv row (SHA-256 and size), that no file is unlisted, and
     that every file copied from the local repo is still byte-identical to its origin (a stale copy stops the build);
  3. refuses data arrays and binaries (.npz .npy .pkl .pt .h5 .parquet ..., NUL bytes) and scans every file for
     credentials, private keys, e-mail addresses and IPv4 addresses (any hit stops the build);
  4. writes README.md (purpose, layout, what each script produced, how the runs were executed, SHA-256 manifest)
     and zips README.md + the mirror deterministically (sorted entries, fixed timestamps and permissions, DEFLATE 9).
Nothing is read from the compute server. Stdlib only.

usage: python3 -I -B build_methods_s1.py            # build and verify
       python3 -I -B build_methods_s1.py --check    # rebuild in memory and compare with the existing zip
"""
import os, re, sys, hashlib, zipfile, io, shutil

ROOT = '/Users/xlzhexue/Desktop/proj/creed'
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.environ.get('CREED_SUBMISSION_DIR', os.path.dirname(HERE))
MIRROR = os.path.join(ROOT, 'code_methods_s1')
OUT = os.path.join(PKG, '08_Methods_S1_code.zip')
TITLE = 'Methods S1. CREED analysis and figure code, related to STAR Methods'
FIXED_DATE = (2026, 10, 11, 0, 0, 0)
SIZE_TARGET = 2 * 1024 * 1024
# package-builder folders mirrored into 08_package_builders/<repo-relative path>/ at every build
BUILDER_DIRS = ['submission_CellSystems_20261011/_tools']
BUILDER_EXT = ('.py', '.sh', '.swift', '.applescript')
FORBIDDEN_EXT = ('.npz', '.npy', '.pkl', '.pickle', '.pt', '.pth', '.ckpt', '.h5', '.hdf5', '.parquet', '.feather',
                 '.rds', '.rda', '.rdata', '.xlsx', '.docx', '.pdf', '.png', '.tif', '.tiff', '.jpg', '.zip', '.tar', '.gz')
SKIP_NAMES = ('.DS_Store',)

# ----------------------------------------------------------------------------------------------- README content
COMPONENTS = [
 ('01_data_preparation', 'Data preparation: identity protocol, frozen 421-line development bundle, aligned RNA, GDSC2 AUC, full proteome, CRISPR gene effect, legacy scGPT representation, whitelisted extra-line packages'),
 ('02_protein_graph_and_teacher_runs_20261008', 'Measured-layer teachers, protein sample-graph runs and the +19-line extra-supervision run whose saved fold, inner-bank, graph and prediction files every later run reads'),
 ('03_development_runs_20261009', 'The pre-registered development runs of 2026-10-09 (one folder per run) and the independent read-only recomputations of five of them'),
 ('04_sealed_set_20261009', 'One-shot confirmation on the sealed internal-final set and its independent recomputation'),
 ('05_followup_N6_20261011', 'Pre-registered follow-up N6 (function-weighted virtual layer) and the script that builds Table S15'),
 ('06_cptac_supplemental', 'CPTAC cross-cohort layer-transfer analysis (Figure S5) and the full-source direct comparator'),
 ('07_figures', 'Figure data preparation, per-panel R scripts, renderer and assembler of Figures 1-6, Figures S1-S5 and the graphical abstract'),
 ('08_package_builders', 'Scripts that build the submission package (manuscript and Document S1 .docx, Document S1 PDF, Excel tables, Data S1-S10, this archive, upload zip)'),
]

# (folder prefix inside the archive, markdown text). Every file must fall under one of these prefixes.
DESCRIBE = [
 ('01_data_preparation/gdsc-metabolite-full-protein-20260908-0001/', """\
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
- All other files under `code/`: helpers imported by the scripts above."""),
 ('01_data_preparation/gdsc-metabolite-query-20260908-0001/', """\
Run that fitted the three cross-fitted metabolite-to-FM generators of every held-out-tissue bank (the legacy virtual representation; target: the scGPT embedding of `data421-0002`) and wrote the parent bundle read by `run_gdsc_metabolite_full_protein.py`; the new-cell-line generators were fitted with the same `build_bank` function by `run_geometry.py` (`02_.../privileged-geometry-20261008-0001/`). Own code snapshot (commit in `source-commit.txt`), reduced to the import closure of `run_gdsc_metabolite_query.py` and its verifier; every file is byte-identical to the copy in `gdsc-metabolite-full-protein-20260908-0001/code/`. `launch.sh` records the mounts (data421, rna421, gdsc421 and the two earlier reference runs). STAR Methods: Virtual layers (Legacy virtual representation)."""),
 ('01_data_preparation/gdsc421-20260906-0001/', """\
`run.sh` and `code/scripts/prepare_gdsc421.py` (with its import closure): GDSC2 fitted AUC for the 421 lines on the fixed 295-drug panel (exact SANGER_MODEL_ID and DRUG_ID; invalid and conflicting records excluded; missing pairs kept missing). STAR Methods: Drug-response endpoint. Run in the image `creed-vcp-runtime:py3112-torch220-esm321-v9`, as `run.sh` records."""),
 ('01_data_preparation/crispr310-prepare-20260908-0001/', """\
`code/scripts/prepare_crispr_development.py` (with its import closure) and `code/DEPMAP24Q2_SOURCE_BINDINGS.json` (its SHA-256 equals the `bindings_sha256` in the output manifest): the CRISPR gene-effect matrix of the 310 screened development lines x 18,443 genes from DepMap 24Q2. STAR Methods: CRISPR gene-effect endpoint; Figure 5; Tables S10-S12. The launcher of this run was not retained."""),
 ('01_data_preparation/extra-supervision-20261008-0001/resources/', """\
Extraction of the three whitelisted extra-line packages (19 M-AUC lines: `extract_m_gdsc.py`, `verify_m_gdsc.py`, `m_gdsc_gateway_payload.py`; 66 M-RNA and 354 RNA-AUC lines: `extract_rna_candidates.py`, `verify_and_package_rna.py`, `rna_gateway_payload.py`). The two `*_gateway_payload.py` files are the self-contained readers that were executed; they embed the frozen whitelist as JSON (cell-line, DepMap model and patient identifiers, tissue, metabolite-feature, drug and gene-index identifiers; no assay value). `original_prepare_gdsc421.py` and `original_prepare_research_rna421.py` are byte-identical copies of the preparation scripts in this component, used for source-scale parity. STAR Methods: Frozen development inputs and whitelisted extra lines; Data S1."""),
 ('02_protein_graph_and_teacher_runs_20261008/strong-teacher-recoverability-20261008-0001/', """\
Measured-proteome teachers ("M + measured P values") whose saved predictions the panel, acquisition and push-line runs reproduce (teacher of R(k), Table S5; STAR Methods: Supervised measured panels and random-panel controls). `launch_scenario.sh` ran `code/run_recoverability.py` (new cell lines); `launch_tissue_v2.sh` ran `code_v2/run_recoverability.py` (held-out tissue), whose only change is the support fallback for teacher folds with fewer than three training tissues, the mid-run revision disclosed in STAR Methods (Quantification and statistical analysis, Pre-registration and reproduction gates). `code/verify_results.py` replays the saved selections and scores; `code/preflight.py` is the preflight; `new_cell_fold_ids.json` is the new-cell-line fold assignment of the 421 lines (fold numbers only)."""),
 ('02_protein_graph_and_teacher_runs_20261008/privileged-geometry-20261008-0001/', """\
Protein sample-graph regularization with an M-only query interface. `launch.sh` ran `code/run_geometry.py` (mu grid 0-10; it reuses the held-out-tissue FM banks of `gdsc-metabolite-query-20260908-0001` and fits the new-cell-line M-to-FM generators); `launch_extension.sh` ran `code/extend_geometry.py`, the disclosed extension to mu = 100 and 1000. Their saved protein indices and graphs are read by the later runs. `verification/`: independent checks of the saved results, the effective-kernel algebra and the large-mu preflight. STAR Methods: Protein sample-graph regularization; Figure 3; Figure S2; Data S5 (through `pushline_checks.py`)."""),
 ('02_protein_graph_and_teacher_runs_20261008/extra-supervision-20261008-0001/', """\
`launch.sh` ran `code/run_extra.py` (imports `run_recoverability.py`): the "+19" arms with the 19 whitelisted M-AUC lines added as training rows. This run wrote the saved fold, inner-bank, kernel-grid and prediction files (`output/<scenario>/fold-*/...`) that every 2026-10-09 run, the sealed-set freeze and N6 load and assert unchanged. `verification/`: independent verification of the saved assets, selections and scores (`verify_saved_extra.py`, launched by `launch_verification.sh`) and the extension preflight. STAR Methods: Cross-validation design; Frozen development inputs and whitelisted extra lines."""),
 ('03_development_runs_20261009/crosslayer-matrix-20261009-0001/', """\
`code/crosslayer.py` (all arms, both scenarios), `code/analyze.py` (pre-registered analysis), `code/run.sh`; `dev/dev_fake_root.py` builds the synthetic root of the smoke test; `posthoc/posthoc_saved_compare.py` is the post-hoc, descriptive comparison with the saved project arms (including the legacy FM candidate). Produced: Figure 2A-G, Figure S2A-C, Table S2, Data S4, and the predictions that the Figure 3 and Figure S4 data stages and Table S9 use. STAR Methods: Per-drug masked kernel ridge head; Virtual layers (including the paired-shuffle control and headroom, recovered gain and recovered share)."""),
 ('03_development_runs_20261009/generator-curve-20261009-0001/', """\
`code/gencurve.py`, `code/analyze_curve.py`, `code/run.sh`: generator-size learning curve (f50/f75/f100 and the +66 extra M-RNA arm). Produced: Figure 2G, Figure S1A-C, Table S3. STAR Methods: Generator-size learning curve and pairing universe."""),
 ('03_development_runs_20261009/generator-fullpool-20261009-0001/', """\
`code/extract_fullpool.py`, `code/run.sh`: identity-only enumeration of the pairing universe outside the 608 split lines (the final code file is the one registered in SHA256SUMS.v3; the two earlier versions corrected before the run were not retained). Produced: Figure 2H, Figure S1D, Table S4, Table S8 row H093. STAR Methods: Generator-size learning curve and pairing universe."""),
 ('03_development_runs_20261009/protein-panel-20261009-0001/', """\
`code/protein_panel.py`, `code/analyze.py`: supervised measured protein panels, random-panel controls and graph-on-panel arms. Produced: Figure 4B-D, Figure S3A, Table S5, Data S6 (and the graph-on-panel row H083 of Table S8). STAR Methods: Supervised measured panels and random-panel controls."""),
 ('03_development_runs_20261009/rna-panel-20261009-0001/', """\
`code/rna_panel.py`, `code/analyze.py`, `code/run.sh`, `code/job_dryrun.sh`: the gene-panel version. Produced: the gene rows of Figure 4B-D, Table S6, Data S7. STAR Methods: Supervised measured panels and random-panel controls."""),
 ('03_development_runs_20261009/acquisition-probe-20261009-0001/', """\
`code/acquisition_probe.py`, `code/make_tables.py` (formatting only): N1, per-sample adaptive protein acquisition. Produced: Figure 4E, Table S7 (N1), Data S8 (part a), Table S8 row H087. STAR Methods: Pre-registered negative controls (N1)."""),
 ('03_development_runs_20261009/acquisition-probe-rna-20261009-0001/', """\
`code/acquisition_probe_rna.py`, `code/run.sh`; `dev/dev_fake_root.py` (synthetic smoke test): the transcriptome mirror of N1. Produced: Data S8 (part b), Table S7, Table S8 row H088. STAR Methods: Pre-registered negative controls (N1). The `dev/` copy of `acquisition_probe_rna.py` is byte-identical to `code/` and is not repeated."""),
 ('03_development_runs_20261009/probe2-context-abstain-20261009-0001/', """\
`code/probe2.py` (part A, N2 context-adaptive panels), `code/abstain.py` (part B, N3 abstention), `code/analyze.py`, `code/make_result_tables.py` (formatting only), `code/run.sh`, `code/job_dryrun.sh`. Produced: Figure 4F-H, Figure S3B, Table S7 (N2, N3), Data S9, Data S10, Table S8 row H089. STAR Methods: Pre-registered negative controls (N2, N3)."""),
 ('03_development_runs_20261009/crispr-endpoint-20261009-0001/', """\
`code/crispr_endpoint.py`, `code/analyze.py`, `code/run.sh`: the cross-layer map and the panel law on the CRISPR gene-effect endpoint. Produced: Figure 5, Tables S10-S12, the development CRISPR values of Figure 6. STAR Methods: CRISPR gene-effect endpoint; Pre-registered negative controls (N5)."""),
 ('03_development_runs_20261009/pushline-checks-20261009-0001/', """\
`code/pushline_checks.py` (analysis only, no training): controls, decomposition, leave-one-fold-out and bootstrap checks of the protein sample-graph and tissue-label effects. Produced: Data S5; with the cross-layer and tissue-kernel outputs, the plot tables of Figure 3 (A, E, H) and Figure S2D-F (through `prepare_fig4_v2.py`). STAR Methods: Protein sample-graph regularization; Quantification and statistical analysis (Robustness)."""),
 ('03_development_runs_20261009/tissue-kernel-20261009-0001/', """\
`code/tissue_kernel.py`: tissue-kernel augmentation K_M + w T in new cell lines. Produced: Figure S2D (free tissue label, with the cross-layer run; through `prepare_fig4_v2.py`). STAR Methods: Per-drug masked kernel ridge head (tissue kernel)."""),
 ('03_development_runs_20261009/shared-crossdrug-20261009-0001/', """\
`code/shared_crossdrug.py`, `code/test_synthetic.py` (synthetic unit tests), `code/diagnose.py` (post-hoc, descriptive, not pre-registered). Produced: Table S8 row H084. STAR Methods: Pre-registered negative controls (N4)."""),
 ('03_development_runs_20261009/rna-learning-curve-20261009-0001/', """\
`code/rna_lc.py`, `code/analyze.py`, `code/run.sh`, `code/preflight_identity.py`, `code/test_inner.py` (inner banks only), `code/fake_outputs.py` (synthetic outputs to test `analyze.py`): RNA learning curve, the 354 RNA-AUC extra-label gate and the RNA-teacher pseudo-label control. Produced: Table S8 row H085. STAR Methods: Pre-registered negative controls (N4). The `dev/` copies are byte-identical to `code/` and are not repeated."""),
 ('03_development_runs_20261009/null_baseline_20261009/', """\
`null_baseline.py`: reproduction of the saved new-cell-line scores and the per-drug training-mean null; run read-only through ssh standard input, no server run directory. Produced: Table S1, Note S1. STAR Methods: Quantification and statistical analysis (Null models)."""),
 ('03_development_runs_20261009/independent_recomputation/', """\
Read-only recomputations written separately from the run code and executed through ssh standard input (`python -I`): `crispr-endpoint-20261009-0001/recheck.py`, `rna-panel-20261009-0001/audit_metric.py`, `probe2-context-abstain-20261009-0001/auditA.py` and `auditB.py`, `acquisition-probe-20261009-0001/audit_recompute.py`, `acquisition-probe-rna-20261009-0001/audit.py`. They rescore saved predictions from saved truth and training scales and recompute the pre-registered comparisons. STAR Methods: Quantification and statistical analysis (Independent recomputation)."""),
 ('04_sealed_set_20261009/internal-final-20261009-0001/', """\
`code/oneshot.py` (freeze, gates, dry run and the single sealed run), `code/creed_core.py` (model core copied from the development reference implementations), `code/sealed_io.py` (identity-gated loaders; only whitelisted identities are decoded); `logs/run_gates_dryrun.sh`, `logs/run_dryrun.sh`, `logs/run_freeze.sh` are the launchers kept with the logs. The three code files are those bound by SHA256SUMS.v2 and the release receipt (Data S3). Produced: Figure 6, Table 1, Tables S13 and S14. STAR Methods: One-shot confirmation on the sealed internal-final set; Quantification and statistical analysis (Sealed-set statistics)."""),
 ('04_sealed_set_20261009/independent_recomputation/', """\
`recompute_remote.py`: the separately written read-only recomputation named in Data S3 (RECOMPUTE.md), executed once through ssh standard input with `python -I`; it re-derives the sealed truth from the raw files for the 123 identities it lists (identifiers only), recomputes every score, bootstrap, sign test and verdict, and checks hashes, timeline and identities. STAR Methods: Quantification and statistical analysis (Independent recomputation)."""),
 ('05_followup_N6_20261011/settle-stageA-20261011-0001/', """\
`code/stageA.py` and `code/stageA_crispr.py` (N6 on the drug-response and CRISPR endpoints), `code/analyze.py`, `code/analyze_crispr.py`, `code/run.sh`, `code/run_crispr.sh`. `code/crosslayer.py` and `code/crispr_endpoint.py` are the frozen development files that N6 imports; they are byte-identical to the copies in `03_.../crosslayer-matrix-20261009-0001/code/` and `.../crispr-endpoint-20261009-0001/code/`. Produced: Table S15, Note S3. STAR Methods: Pre-registered negative controls (N6)."""),
 ('05_followup_N6_20261011/settle_stageA_20261011/', """\
`make_table_s15.py`: builds Table S15 from the per-fold N6 results (averaging only); also in Data S2."""),
 ('06_cptac_supplemental/cptac-blocks-20260906-0001/', """\
`code/scripts/prepare_cptac_blocks.py` with `code/src/creed/science/cptac_blocks.py`: source-native per-cohort bundles from the cBioPortal study archives (the script also writes bundles of cohorts that this paper does not use). STAR Methods: Cross-cohort layer transfer in CPTAC cohorts; Figure S5."""),
 ('06_cptac_supplemental/cptac-baseline-20260906-0002/', """\
`code/scripts/run_cptac_bidirectional_baseline.py` with `code/src/creed/science/masked_baseline.py`, `verify_cptac_baseline.py` and the design note `code/experiments/CPTAC_BIDIRECTIONAL_BASELINE_20260906.md`: the BRCA + LUAD bidirectional baseline that defined the five cohort-stratified patient folds (scikit-learn `StratifiedKFold` on cohort labels, seed 20260906) and the target panels of the mixed-fold task. `cptac-anchor-20260906-0001` reads its saved folds (asserting the SHA-256 of its summary, 156e35f7...), and `cptac-full-source-direct-20261008-0001` reads them through the anchor run's outputs (`source_summary_sha256` f095caa4...). The launcher was not retained. STAR Methods: Cross-cohort layer transfer in CPTAC cohorts; Discussion."""),
 ('06_cptac_supplemental/cptac-anchor-20260906-0001/', """\
`code/scripts/run_cptac_anchored_context.py` (same-gene anchor plus context) and `verify_cptac_baseline.py`: the anchor + context predictions of the BRCA + LUAD mixed-fold task against which `cptac-full-source-direct-20261008-0001` compares. STAR Methods: Cross-cohort layer transfer in CPTAC cohorts."""),
 ('06_cptac_supplemental/cptac-three-inputs-20260907-0001/', """\
`code/scripts/prepare_cptac_three_domains.py` (SHA-256 equals the `producer_sha256` of the run summary): the BRCA, LUAD and UCEC input bundle (327 patients) of the transfer tasks. STAR Methods: Cross-cohort layer transfer in CPTAC cohorts."""),
 ('06_cptac_supplemental/cptac-domain-experts-20260907-0001/', """\
`run.sh` and `code/scripts/run_cptac_domain_experts.py`: the six transfer tasks without shift correction, the reference run mounted by the source-shift run. STAR Methods: Cross-cohort layer transfer in CPTAC cohorts."""),
 ('06_cptac_supplemental/cptac-source-shift-20260907-0001/', """\
`run.sh` and `code/scripts/run_cptac_domain_experts.py --source-shift`: the six tasks with source-only shift correction (training mean, pooled anchor, pooled context, uniform, gated and reversed-gate experts). `check_domain_expert_outputs.py` recomputes the evaluation, `predict_domain_translation.py` runs the saved models on source-only inputs and `check_domain_source_only.py` compares the two (checker launch lines were not retained). Produced: Figure S5 (through `prepare_cptac.py`). STAR Methods: Cross-cohort layer transfer in CPTAC cohorts."""),
 ('06_cptac_supplemental/cptac-full-source-direct-20261008-0001/', """\
`run.py`: full-source linear direct comparator for the BRCA + LUAD mixed five-fold task (imports `creed.science.masked_baseline`; every code snapshot from 2026-09-07 on carries the same `masked_baseline.py`, SHA-256 c5bf00ac..., as in `cptac-source-shift-20260907-0001/code/src/`; the launcher was not retained). `summarize.py` (from the run record in `research_plans/20261008/strategy/positive_core/cptac_full_source_reader/`) summarizes it without fitting. Used for the RNA-to-protein comparison in the Discussion."""),
 ('07_figures/data_preparation/', """\
Figure data stage: Python scripts that turn saved run outputs into the plot tables (`out/<figure>/plot_data/*.csv`) read by the R panels; each writes a `*_DATA_SOURCES.md` with the SHA-256 of its inputs. Copies from `20261009_figures_v3_cell/r/figures2_6/`, byte-identical to the copies in `20261009_figures2_6_v2/r/figures2_6/` where they were executed (`run.sh data` of that folder runs `prepare_gdsc.py`, `prepare_cptac.py` and `prepare_protocol.py`; the `prepare_*_v2.py` scripts were run one by one). Final figure numbers: `prepare_fig2_v2.py` -> Figure 2 and Figure S2B-C; `prepare_fig3_v2.py` -> Figure 3 (B, C, D, F, G) and Figure S2A; `prepare_fig4_v2.py` -> Figure 3 (A, E, H) and Figure S2D-F; `prepare_fig5_v2.py` -> Figure 4 and Figure S3; `prepare_fig6_v2.py` -> Figure S4; `prepare_fig7_v2.py` -> Figure 5; `prepare_fig8_v2.py` -> Figure 6 (it asserts the plot tables against the published sealed-set values); `prepare_figS1_v2.py` -> Figure 2H and Figure S1; `prepare_cptac.py` -> Figure S5; `prepare_gdsc.py`, `prepare_protocol.py` -> GDSC and schematic count tables of the earlier figure set reused by the above."""),
 ('07_figures/figure1/', """\
Figure 1, built with the code of the v3 figure set: `r/figure1/A.R`-`H.R` (panels), `common.R`, `render_panels.R`, `assemble.R`, `run.sh`, `check_delivery.py` and `_tools/` (vector-bound checks), with `r/cell_style.R` (identical to the v4 copy). Its plot tables are a frozen copy of the accepted 2026-10-07 tables; `prepare_data.py` is the data stage that built them (kept for provenance, not re-run). STAR Methods: Software and computation."""),
 ('07_figures/figures_v4/', """\
Figures 2-6 and Figures S1-S5: `r/render_v4.R <fig> [panel ...]` renders each panel independently at its slot size from `r/<fig>/layout.csv` (PDF, PNG and RDS), and `r/assemble_v4.R <fig>` places the saved panels on the 174-mm page; `r/v4_common.R` and `r/cell_style.R` hold the shared reader, export and style code. In `r/<fig>/`, `A.R`-`H.R` are the panels, `LEG_*.R` and `KEY.R` the legend and key slots, `common.R` the figure's own helpers; `Figure4/prep/extract_probe2_*.py` and `Figure6/prep_dev_folds.R` extract the per-fold tables of Figure 4G-H and Figure 6F-G from saved outputs (no recomputation). The server folder names keep the pre-v1.8 supplementary numbers: `r/FigureS4/` is Figure S1, `r/FigureS1/` is Figure S2, `r/FigureS3/` is Figure S3, `r/FigureS2/` is Figure S4 and `r/FigureS5/` is Figure S5; `r/Figure2/`-`r/Figure6/` are Figures 2-6. Superseded variants kept on the server (`_old/`, `_old_v4/`, `_v4r0/`, `_bak/` and the layout-trial folders) are not included."""),
 ('07_figures/graphical_abstract/', """\
`graphical_abstract.R`: the graphical abstract."""),
 ('08_package_builders/', """\
Builders of the submission package, refreshed from `_tools/` at every build of this archive: `build_cell_docx.py` (manuscript, Document S1, cover letter and highlights .docx from the Markdown masters), `build_xlsx.py` (Excel Tables S3, S4, S8 and S11-S14), `build_data_zips.py` (Data S1-S10), `build_methods_s1.py` (this archive), `build_si_pdf.py` (Document S1 PDF: WebKit print of the Document S1 master; imports build_cell_docx.py), `build_submission_zip.py` (upload zip with SHA-256 manifest), `build_docx_v18_original.py` (earlier builder kept for provenance), and the render checks `render_check.sh`, `html2pdf.swift` and `pages_export.applescript`."""),
 ('SOURCES.tsv', """\
Origin of every file: `server:` absolute path on the compute server, `repo:` path in the authors' local project repository, or `session-scratchpad:` file name in the working directory of the session that wrote the independent recomputations; with the SHA-256 and size at the time of copying."""),
]

README_HEAD = """# {title}

## Purpose

This archive holds the original code of the CREED study: the data-preparation scripts that built the frozen development bundle and the legacy virtual representation, the 2026-10-08 runs whose saved fold, inner-bank, graph and teacher files the later runs read, every 2026-10-09 development run with its analysis script, the one-shot sealed-set run and its independent recomputation, the follow-up N6 with the script that builds Table S15, the CPTAC supplemental analysis, the figure data-preparation and plotting code, and the scripts that built the submission package. It contains code only: no data array, model file or run output. Registered plans, code checksums (SHA256SUMS) and run records are in Data S2 (development runs and N6) and Data S3 (sealed-set run).

All files are verbatim copies of the files that were executed (or, for the package builders, of the current builders). Nothing was edited, so absolute paths of the compute server (`/public/home/mengxl/...`), of the containers (`/code`, `/oldcode`, `/parent`, `/home/node/...`) and of the authors' workstation remain in the code, and some comments, logs and plan references are in Chinese. Files listed in a SHA256SUMS file of Data S2 or Data S3 can be checked against it.

## Layout

Folders by component. Inside each, the folder names are the run directories of the compute server (`/public/home/mengxl/creed-research-runs/<run>/`) or the figure working directories (`/public/home/mengxl/workspaces/creed-research/figure_work/<dir>/`), and the paths below them are unchanged, so a path written in the code maps onto this layout. Files kept only in the authors' project repository (`null_baseline_20261009/`, `settle_stageA_20261011/`, `submission_CellSystems_20261011/_tools/`) keep their repository folder, and the independent recomputations are grouped under `independent_recomputation/<run>/`.

| Folder | Content | Files |
|---|---|---:|
{component_rows}

`SOURCES.tsv` gives the origin, SHA-256 and size of every file at the time it was copied. For runs that executed from a code snapshot (01 and 06), only the files reachable by import from the entry scripts are included (static import closure, with `PYTHONPATH=/code/src:/code/scripts` as in the launchers); the rest of each snapshot is unrelated to this paper. Where a run kept a second byte-identical copy of a file (the `dev/` folders of three 2026-10-09 runs), it is not repeated.

## What each script produced

Figure, table and Data numbers are those of this paper; STAR Methods sections are named by their headings.

{describe}

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
{manifest}
"""

# ----------------------------------------------------------------------------------------------- helpers
def sha(b):
    return hashlib.sha256(b).hexdigest()

def mirror_files():
    out = []
    for dp, dn, fn in os.walk(MIRROR):
        dn[:] = sorted(d for d in dn if d != '__pycache__')
        for f in sorted(fn):
            if f in SKIP_NAMES:
                continue
            out.append(os.path.relpath(os.path.join(dp, f), MIRROR))
    return sorted(out)

def read_sources():
    rows = {}
    with open(os.path.join(MIRROR, 'SOURCES.tsv'), encoding='utf-8') as fh:
        head = fh.readline().rstrip('\n').split('\t')
        assert head == ['archive_path', 'sha256', 'bytes', 'origin', 'note'], head
        for line in fh:
            a, h, n, o, note = line.rstrip('\n').split('\t')
            rows[a] = dict(sha256=h, bytes=int(n), origin=o, note=note)
    return rows

def write_sources(rows):
    lines = ['archive_path\tsha256\tbytes\torigin\tnote']
    for a in sorted(rows):
        r = rows[a]
        lines.append('\t'.join([a, r['sha256'], str(r['bytes']), r['origin'], r['note']]))
    with open(os.path.join(MIRROR, 'SOURCES.tsv'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')

def refresh_builders(rows):
    """Mirror the current package builders into 08_package_builders/<dir>/; returns a change list."""
    changes = []
    for d in BUILDER_DIRS:
        src_dir = os.path.join(ROOT, d)
        arc_dir = '08_package_builders/' + d
        want = sorted(f for f in os.listdir(src_dir)
                      if os.path.isfile(os.path.join(src_dir, f)) and f.endswith(BUILDER_EXT))
        for a in [a for a in rows if a.startswith(arc_dir + '/')]:
            if os.path.basename(a) not in want:
                os.remove(os.path.join(MIRROR, a)); del rows[a]; changes.append('removed ' + a)
        for f in want:
            a = arc_dir + '/' + f
            data = open(os.path.join(src_dir, f), 'rb').read()
            dst = os.path.join(MIRROR, a)
            old = open(dst, 'rb').read() if os.path.isfile(dst) else None
            if old != data:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, 'wb') as fh:
                    fh.write(data)
                changes.append(('updated ' if old is not None else 'added ') + a)
            rows[a] = dict(sha256=sha(data), bytes=len(data), origin='repo:' + d + '/' + f, note='')
    return changes

TLD = r'(?:com|org|net|edu|gov|mil|int|io|ai|cn|uk|de|fr|jp|kr|ch|nl|se|au|ca|us|info|biz|me|co)'
SCAN = {
    'private key': re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----|\bssh-(?:rsa|ed25519|dss) AAAA'),
    # written with character classes so that this line does not match itself
    'token': re.compile(r'\b(?:gh[pos]_[A-Za-z0-9]{20,}|github[_]pat_|glpat[-]|xox[abprs]-|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_\-]{30}|sk-[A-Za-z0-9]{20,}|sk[-]ant-|hf_[A-Za-z0-9]{30,})'),
    'credential assignment': re.compile(r'''(?i)\b(?:password|passwd|secret|api[_-]?key|apikey|access[_-]?key|auth[_-]?token)\b\s*[:=]\s*['"][^'"]{4,}['"]'''),
    'authorization header': re.compile(r'(?i)\bauthorization\s*:\s*\S|\bbearer\s+[A-Za-z0-9._\-]{16,}'),
    'URL with credentials': re.compile(r'[a-z][a-z0-9+.-]*://[^/\s:@]+:[^/\s@]+@'),
    'e-mail address': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.' + TLD + r'\b'),
    'IPv4 address': re.compile(r'(?<![\d.])(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?![\d.])'),
}

def scan(path, data):
    hits = []
    if b'\x00' in data:
        hits.append((path, 0, 'binary content'))
        return hits
    text = data.decode('utf-8', 'replace')
    for i, line in enumerate(text.splitlines(), 1):
        for k, rx in SCAN.items():
            m = rx.search(line)
            if m:
                hits.append((path, i, k + ': ' + m.group(0)[:60]))
    return hits

def describe_for(path):
    best = None
    for prefix, _ in DESCRIBE:
        if path == prefix or path.startswith(prefix):
            if best is None or len(prefix) > len(best):
                best = prefix
    return best

# ----------------------------------------------------------------------------------------------- build
def collect(refresh=True):
    rows = read_sources()
    changes = refresh_builders(rows) if refresh else []
    if refresh:
        write_sources(rows)
    files = [f for f in mirror_files() if f != 'SOURCES.tsv']
    problems = []
    if set(files) != set(rows):
        problems += ['file not in SOURCES.tsv: ' + f for f in sorted(set(files) - set(rows))]
        problems += ['listed but missing: ' + f for f in sorted(set(rows) - set(files))]
    blobs = {}
    for f in files:
        data = open(os.path.join(MIRROR, f), 'rb').read()
        blobs[f] = data
        r = rows.get(f)
        if r and (sha(data) != r['sha256'] or len(data) != r['bytes']):
            problems.append('SHA-256/size differs from SOURCES.tsv: ' + f)
        if r and r['origin'].startswith('repo:'):
            origin = os.path.join(ROOT, r['origin'][5:])
            if not os.path.isfile(origin):
                problems.append('repo origin missing: ' + r['origin'])
            elif open(origin, 'rb').read() != data:
                problems.append('stale copy (origin changed since copying): ' + f + ' <- ' + r['origin'])
        if f.lower().endswith(FORBIDDEN_EXT) or '__pycache__' in f or f.endswith('.pyc') or '.bak' in os.path.basename(f):
            problems.append('forbidden file type: ' + f)
        if describe_for(f) is None:
            problems.append('no README description covers: ' + f)
    blobs['SOURCES.tsv'] = open(os.path.join(MIRROR, 'SOURCES.tsv'), 'rb').read()
    hits = []
    for f, data in blobs.items():
        hits += scan(f, data)
    return rows, blobs, changes, problems, hits

def make_readme(blobs):
    counts = {}
    for f in blobs:
        top = f.split('/')[0]
        counts[top] = counts.get(top, 0) + 1
    comp_rows = '\n'.join('| `%s/` | %s | %d |' % (c, text, counts.get(c, 0)) for c, text in COMPONENTS)
    comp_rows += '\n| `SOURCES.tsv` | Origin, SHA-256 and size of every file | 1 |'
    desc = []
    current = None
    for prefix, text in DESCRIBE:
        top = prefix.split('/')[0]
        if top != current:
            desc.append(('### `%s/`\n' % top) if top in dict(COMPONENTS) else ('### `%s`\n' % top))
            current = top
        n = sum(1 for f in blobs if f == prefix or f.startswith(prefix))
        desc.append('**`%s`** (%d file%s)\n\n%s\n' % (prefix, n, '' if n == 1 else 's', text))
    manifest = '\n'.join('| %s | %d | `%s` |' % (sha(blobs[f]), len(blobs[f]), f) for f in sorted(blobs))
    return README_HEAD.format(title=TITLE, component_rows=comp_rows, describe='\n'.join(desc), manifest=manifest)

def zip_bytes(blobs, readme):
    buf = io.BytesIO()
    entries = [('README.md', readme.encode('utf-8'))] + [(f, blobs[f]) for f in sorted(blobs)]
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, data in entries:
            zi = zipfile.ZipInfo(name, date_time=FIXED_DATE)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.create_system = 3
            mode = 0o755 if name.endswith('.sh') else 0o644
            zi.external_attr = (0o100000 | mode) << 16
            z.writestr(zi, data, compresslevel=9)
    return buf.getvalue(), entries

def verify_zip(path, entries):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        names = z.namelist()
        assert names == [n for n, _ in entries], 'entry order/names differ'
        for n, data in entries:
            assert z.read(n) == data, n
    return len(names)

def main():
    check = '--check' in sys.argv
    rows, blobs, changes, problems, hits = collect(refresh=not check)
    for c in changes:
        print('refreshed:', c)
    if problems or hits:
        for p in problems:
            print('PROBLEM:', p)
        for h in hits:
            print('SCAN HIT: %s:%d %s' % h)
        raise SystemExit('build stopped: %d problems, %d scan hits' % (len(problems), len(hits)))
    readme = make_readme(blobs)
    data, entries = zip_bytes(blobs, readme)
    if check:
        old = open(OUT, 'rb').read() if os.path.isfile(OUT) else b''
        print('identical to existing zip:', old == data, '(%d bytes)' % len(data))
        raise SystemExit(0 if old == data else 1)
    tmp = OUT + '.tmp'
    with open(tmp, 'wb') as fh:
        fh.write(data)
    n = verify_zip(tmp, entries)
    os.replace(tmp, OUT)
    size = os.path.getsize(OUT)
    print('%s: %d entries (README.md + %d files), %d bytes (%.2f MB), sha256 %s' % (
        os.path.basename(OUT), n, n - 1, size, size / 1048576, sha(data)))
    if size > SIZE_TARGET:
        print('WARNING: archive is larger than 2 MB')

if __name__ == '__main__':
    main()

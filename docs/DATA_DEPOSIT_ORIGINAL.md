# CREED data deposit: development inputs, run outputs and frozen sealed-set models

This deposit holds the data files that the paper *Measured once, used as many: quantifying how much multi-layer value one measured layer recovers* (Cell Systems, submitted) promises to deposit:

- the frozen 421-line development inputs;
- the outer-fold predictions, per-fold truth and training scales of every development arm;
- per-drug metrics where the runs wrote them, and the selected panels per bank;
- the frozen sealed-set model arrays;
- the CPTAC transfer and reconstruction run outputs.

The code that produced and reads these files is the paper's Methods S1. Pre-registration files, run summaries and the sealed-set release chain are in Data S1–S10.

Assembled on 2026-10-11 from the compute-server run directories. Every file is a byte-identical copy (SHA-256 checked after copying).

## Layout

The deposit is split into one archive per run directory, because the repository limits the number of files per record. Unpack every `<run>.tar` into one folder. This gives:

```
runs/<run directory>/...      the files, at the same relative paths as in the original run directory
MANIFEST.tsv                  one row per file: path, bytes, SHA-256, category, what it supports in the paper
SHA256SUMS                    sha256sum -c SHA256SUMS (run from the folder that contains runs/)
RAW_SOURCES.tsv               public source files the data preparation read: location, size, SHA-256 (not redistributed)
TARS_SHA256SUMS               SHA-256 of the archives themselves
```

The same record also holds three files of the paper's supplemental information, byte-identical to the published versions: `08_Methods_S1_code.zip` (Methods S1: the original analysis and figure code, with its own README and per-file SHA-256), `Data_S2.zip` (pre-registration files and reproduction-gate logs) and `Data_S3.zip` (sealed-set release chain). `RECORD_SHA256SUMS` lists the SHA-256 of every file in the record.

The deposit has 788 files in 27 run directories, 7.21 GB in total.

## Running the Methods S1 code against this deposit

The paths mirror the original run root, `/public/home/mengxl/creed-research-runs/`.

- `crosslayer.py`, `crispr_endpoint.py`, `gencurve.py`, `acquisition_probe_rna.py` and `oneshot.py` read the environment variable `CREED_ROOT`. Set it to the `runs/` folder of this deposit.
- `protein_panel.py`, `probe2.py`, `abstain.py`, `acquisition_probe.py`, `pushline_checks.py` and `rna_panel.py` hard-code the original run root in one constant. Change that line to the `runs/` folder.
- The ID arrays load with `np.load(..., allow_pickle=False)`. Some scripts pass `allow_pickle=True`; this is not needed.

## What is where

| Paper item | Files |
|---|---|
| Frozen development inputs (421 lines × 225 metabolites, 6,692 proteins with missingness retained, 295-drug GDSC2 AUC, tissues, held-out-tissue fold ids) | `runs/gdsc-metabolite-full-protein-20260908-0001/output/result/source-inputs.npz` (SHA-256 52d71517…, as in the key resources table), `splits.json`, `bindings.json`, `frozenconfig.json` |
| Aligned RNA bundle (421 × 34,985 genes, ln(1 + CPM); 367 Broad and 54 Sanger lines) | `runs/rna421-20260906-0002/` |
| CRISPR development matrix (DepMap 24Q2 Chronos; 310 of the 421 lines × 18,443 genes) | `runs/crispr310-prepare-20260908-0001/output/result/` |
| GDSC2 drug annotations (drug ids → names, putative targets) | `runs/gdsc421-20260906-0001/result/` |
| Whitelisted extra-line packages (19 M–AUC, 66 M–RNA, 354 RNA–AUC lines; outside the 608 split identities) | `runs/extra-supervision-20261008-0001/resources/*.npz` (the 2026-10-09 code checks their SHA-256) |
| Fold and inner-bank indices, per-fold truth and training scales; inner-bank grids used by the reproduction gates | `runs/extra-supervision-20261008-0001/output/{new_cell,tissue}/` (`protocol.json`, `fold-*/selection.json`, `fold-*/predictions.npz`, `fold-*/{inner-0,inner-1,inner-2,outer}/{grids,geometry}.npz`) |
| 2026-10-08 measured-teacher, protein sample-graph and extra-supervision runs | `runs/strong-teacher-recoverability-20261008-0001/`, `runs/privileged-geometry-20261008-0001/`, `runs/extra-supervision-20261008-0001/` |
| 2026-10-09 development runs (Figures 2–5, Tables S2–S12) | `runs/crosslayer-matrix-…`, `generator-curve-…`, `protein-panel-…`, `rna-panel-…`, `acquisition-probe-…`, `acquisition-probe-rna-…`, `probe2-context-abstain-…`, `crispr-endpoint-…`, `tissue-kernel-…`, `pushline-checks-…`, `shared-crossdrug-…`, `rna-learning-curve-…` |
| Follow-up N6 (Table S15, Note S3) | `runs/settle-stageA-20261011-0001/output/{new_cell,tissue}/fold-*/predictions.npz` |
| Frozen sealed-set model arrays (fitted on the 421 development lines only) | `runs/internal-final-20261009-0001/frozen_models/{model.npz,inner_losses.npz,FROZEN_MANIFEST.json}` and `SHA256SUMS.v2` |
| CPTAC transfer (Figure S5) and reconstruction comparison (Discussion) | `runs/cptac-*/` |

`MANIFEST.tsv` gives, for every file, the figures, tables or sections it supports.

## Notes for readers checking the results

- **Cell lines.** Every file covers only the 421 development lines (or a subset), plus the whitelisted extra lines that are also listed in Data S1. No value of the 64 validation lines or the 123 sealed internal-final lines is in this deposit. All 788 files were scanned for the Sanger model identifiers of those 187 lines on 2026-10-11, with 0 hits. The sealed-set predictions and truth are not part of this deposit; the sealed-set summary and its independent recomputation are in Data S3.
- **Folds.** `source-inputs.npz['fold_ids']` holds the held-out-tissue folds only. The new-cell-line folds are recomputed by `run_recoverability.py` (`cell_folds(tissues, 20261008)`) and saved in `runs/extra-supervision-20261008-0001/output/new_cell/protocol.json` (`outer_fold_ids`) and in the `fit`/`held` arrays of `fold-*/predictions.npz`.
- **Row identity.** Most prediction files carry held indices or sample ids. Three do not:
  - The tissue-kernel prediction files have no row indices or truth.
  - The N6 prediction files have truth but no held indices.
  - The shared-crossdrug files have held indices but no fit indices.

  For all of these, take the row order from the `held` and `sample_ids` arrays of the extra-supervision predictions of the same fold. The tissue-kernel files also need truth from there.
- **Per-drug metrics.** Some runs wrote no per-drug file: protein-panel, acquisition-probe (protein and gene), probe2 part A, rna-panel, tissue-kernel and N6. Their per-drug metrics can be recomputed from the predictions, which embed truth and training scales; tissue-kernel needs truth from the extra-supervision predictions. The N6 CRISPR step saved per-fold results only (`fold_result.json`, in Data S2), not outer predictions.
- **Protein sample-graph versions.** `privileged-geometry-20261008-0001/output/new_cell` and `tissue` are the registered μ grid. `new_cell_v2` and `tissue_v2` add μ = 100 and 1000, an extension made after the first outer-fold summaries had been seen (STAR Methods). The extra-supervision run builds on v2. For the measured-teacher run, `tissue_v2` is the complete held-out-tissue run; an interrupted first attempt was not deposited.
- **Frozen sealed-set arrays.** `model.npz` holds more than head coefficients. It also holds per-line derived development values: the out-of-fold virtual RNA and P (421 × 512), the protein sample graph (421 × 421) and the shuffle permutations. `FROZEN_MANIFEST.json` lists every array with its SHA-256. `SHA256SUMS.v2` is the registered checksum file of the protocol.
- **CPTAC.**
  - The five cohort-stratified patient folds of the 232-patient reconstruction comparison were defined by `cptac-baseline-20260906-0002`. Its prediction archives are not repeated here, because every array in them is identical to the copy inside the `cptac-anchor-20260906-0001` archives. The anchor run asserts the SHA-256 of the baseline summary (156e35f7…), and the full-source comparator reads the folds through the anchor outputs (`source_summary_sha256` f095caa4…).
  - Per-patient gating weights are in `cptac-source-shift-20260907-0001/result/*.npz`.
  - The CPTAC files hold patient-level values re-packaged from the public cBioPortal studies brca_cptac_2020, luad_cptac_2020 and ucec_cptac_2020, which the cBioPortal datahub distributes under the ODC Open Database License (ODbL). The GBM and RCC bundles of the original input folder were never read and are not included.
- **Identifiers.** The CRISPR matrix and the extra-line packages contain public DepMap model (ACH-) and patient (PT-) identifiers as distributed by DepMap. Data S1 lists only Sanger model identifiers.
- **Checksums.** Several runs wrote no output checksum list: the 2026-10-08 runs, crosslayer-matrix, tissue-kernel, shared-crossdrug, acquisition-probe-rna, the CPTAC runs, and the predictions of protein-panel and acquisition-probe. For those files, `SHA256SUMS` (computed on 2026-10-11) is the reference. Where a run did write a checksum list, every deposited file matches it.
- **Software.** The 2026-10-09 runs, the sealed-set run and N6 used Python 3.9.23 with NumPy 2.0.2 and SciPy 1.13.1. Data preparation (except the extraction of the extra-line packages, which ran in a separate network-connected service container, and the scGPT embedding, which ran on GPU with Python 3.10, PyTorch 2.3.0 and scGPT 0.2.4), the 2026-10-08 runs and the CPTAC runs used the Docker image listed in the key resources table (local image ID sha256:0a450c02…; Python 3.11.2, NumPy 1.26.4, SciPy 1.16.3, scikit-learn 1.7.2, PyTorch 2.2.0). That image is not in a public registry.

## Upstream data and terms

The derived matrices come from public resources. Cite the original sources, and follow their terms of use, when reusing them:

- CCLE metabolomics (Li et al., 2019);
- Cell Model Passports RNA-seq release 20191101 (van der Meer et al., 2019);
- the ProCan-DepMapSanger proteome (Gonçalves et al., 2022);
- GDSC2 release 8.4 (Iorio et al., 2016; Yang et al., 2013);
- DepMap 24Q2 (CC BY 4.0);
- the cBioPortal CPTAC studies (ODbL; Krug et al., 2020; Gillette et al., 2020; Dou et al., 2020).

`RAW_SOURCES.tsv` (also in Data S1 as `raw_source_checksums.tsv`) gives the exact public files, their sizes and SHA-256.

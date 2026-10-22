# Validation performed on 2026-10-14 and 2026-10-23

The 2026-10-23 update brought the export in line with the 2026-10-22 submission package: Methods S1 component `09_prospective_test_20261022` (the pre-registered prospective test), the new Figure 2H panel script, updated package builders, Data S2 with the prospective-test records, the 28th data archive and 45 more run records. The checks marked 2026-10-23 were run again on the compute server with the study's own environment (Python 3.9.23, NumPy 2.0.2, SciPy 1.13.1) against the archives of the deposit upload folder. Checks marked 2026-10-14 cover inputs that did not change.

| Check | Date | Result | Evidence |
|---|---|---|---|
| Original source integrity | 2026-10-23 | All 337 payload files match Methods S1 `SOURCES.tsv`; all Python files parse | `validation/code_integrity.json` |
| Supplement integrity | 2026-10-23 | Data S1–S10 ZIP checksums and CRCs pass | `metadata/SUPPLEMENT_SHA256SUMS` |
| Data archives | 2026-10-23 | 28 archives, 918 files, 10,622,250,085 bytes unpacked and verified | `validation/data_integrity.json` |
| Run records | 2026-10-23 | 219 small records (174 from 2026-10-14, 45 added), anchored to the server inventories or Data S3; 1,137 files verified with the records installed | `metadata/RECORDS.tsv`, `validation/data_with_records.json` |
| Synthetic execution | 2026-10-23 | Both scenarios × 5 folds; original shared-crossdrug tests | `validation/synthetic.json` |
| Real prediction replay | 2026-10-23 | 145 new-cell and 120 tissue fold/arm sRMSE values, zero difference (same result as 2026-10-14) | `validation/metric_replay.json` |
| Prospective analysis replay | 2026-10-23 | Original `analyze.py` rerun on the deposit and records: all five analysis files byte-identical to the deposited ones | `validation/prospective_analysis_replay.json` |
| Credential scan | 2026-10-23 | 942 files and ZIP members, 0 matches for the three high-confidence patterns | `validation/credential_scan.json` |
| Fresh cross-layer fitting | 2026-10-14 | Both scenarios × 5 folds; 320 + 270 arrays compared, max difference 5.43e-14 | `validation/real_crosslayer.json` |
| Plot table generation | 2026-10-14 | Eight historical data stages, 59 CSV files; original checks pass | `validation/figure_data.json` |

The 2026-10-14 numerical checks used an isolated venv on macOS arm64 with Python 3.9.6, NumPy 2.0.2 and SciPy 1.13.1. The historical Python patch version was 3.9.23; a conda recipe pins that version. The test workstation emitted NumPy `matmul` RuntimeWarnings. All reproduced arrays had the same finite/nonfinite masks as the archived arrays and agreed to the tolerance reported above; the warnings were not hidden or used to skip checks. Other BLAS/platform combinations may produce different rounding.

## Remaining limitations

- The Zenodo DOI was reserved but not publicly accessible at the 2026-10-23 check (HTTP 404). No authenticated upload or publication was performed by the release-preparation scripts.
- The project owner explicitly deferred software licensing on 2026-10-14. This export adds no software license grant. The final author citation remains pending, so the citation file is deliberately a template.
- This check starts at the frozen development inputs. It does not rebuild every raw upstream assay bundle or the historical scGPT embedding. The prospective test's GDSC1 extraction was not rerun from the source file, which is not redistributed.
- The original local Docker image is not available from a public registry. The new CPU Dockerfile and the recorded optional environments are untested recipes, not an exact reconstruction of it.
- The early experiments described only in Table S8 parts A and C are not completely represented in Methods S1. This exception is already stated in the manuscript.
- Not every development, N6, CPTAC or prospective-test model was freshly retrained. These sources are included; the strongest new runtime evidence is for the cross-layer matrix (refit) and for the prospective analysis (rerun from deposited predictions).
- The 64 validation and 123 sealed cell-line assay values and sealed predictions are not distributed. Public readers can audit the protocol, frozen development models, release receipt and summary, not independently rescore those inaccessible values.
- N6 CRISPR retained fold results, not its outer predictions.
- The historical plot-data stages are runnable, but final V4 data assembly, Figure 1's historical raw snapshots, and R figure rendering (including the new Figure 2H) are not fully verified. See `FIGURES.md`.
- The GitHub Actions workflow passed for commit d4b2f56 (2026-10-14). It has not yet run for this update.

The workflow follows the documented v7 usage of [actions/checkout](https://github.com/actions/checkout) and [actions/setup-python](https://github.com/actions/setup-python), checked during the 2026-10-14 preparation. Package versions from the tested venv are recorded in `env/validation-lock.txt`.

No original scientific code, split, frozen data array, analysis threshold, or numerical manuscript claim was changed to obtain a passing result. New script copies change file paths only. Newly generated results are kept outside the release and outside the input deposit.

# Validation performed on 2026-10-14

| Check | Result | Evidence |
|---|---|---|
| Original source integrity | All 327 payload files match Methods S1 `SOURCES.tsv` | `validation/code_integrity.json` |
| Supplement integrity | Data S1–S10 ZIP checksums and CRCs pass | `metadata/SUPPLEMENT_SHA256SUMS` |
| Data archives | 27 archives, 788 files, 7,205,749,506 bytes verified | `validation/data_integrity.json` |
| Run records | 174 small records added, anchored to the server inventory or Data S3 | `metadata/RECORDS.tsv` |
| Synthetic execution | Both scenarios × 5 folds; original shared-crossdrug tests | `validation/synthetic.json` |
| Real prediction replay | 145 new-cell and 120 tissue fold/arm sRMSE values, zero difference | `validation/metric_replay.json` |
| Fresh cross-layer fitting | Both scenarios × 5 folds; 320 + 270 arrays compared, max difference 5.43e-14 | `validation/real_crosslayer.json` |
| Plot table generation | Eight historical data stages, 59 CSV files; original checks pass | `validation/figure_data.json` |

The new numerical environment was an isolated venv on macOS arm64 using Python 3.9.6, NumPy 2.0.2, and SciPy 1.13.1. The historical Python patch version was 3.9.23; a conda recipe pins that version. The test workstation emitted NumPy `matmul` RuntimeWarnings. All reproduced arrays had the same finite/nonfinite masks as the archived arrays and agreed to the tolerance reported above; the warnings were not hidden or used to skip checks. Other BLAS/platform combinations may produce different rounding.

## Remaining limitations

- The Zenodo DOI was reserved but not publicly accessible at the preparation check. No authenticated upload or publication was performed by the release-preparation scripts.
- The project owner explicitly deferred software licensing on 2026-10-14. This preparation export adds no software license grant. The final author citation remains pending, so the citation file is deliberately a template.
- This check starts at the frozen development inputs. It does not rebuild every raw upstream assay bundle or the historical scGPT embedding.
- The original local Docker image is not available from a public registry. The new CPU Dockerfile and the recorded optional environments are untested recipes, not an exact reconstruction of it.
- The early experiments described only in Table S8 parts A and C are not completely represented in Methods S1. This exception is already stated in the manuscript.
- Not every development, N6, or CPTAC model was freshly retrained. These sources are included; the strongest new runtime evidence here is for the cross-layer matrix.
- The 64 validation and 123 sealed cell-line assay values and sealed predictions are not distributed. Public readers can audit the protocol, frozen development models, release receipt and summary, not independently rescore those inaccessible values.
- N6 CRISPR retained fold results, not its outer predictions.
- The historical plot-data stages are now runnable, but final V4 data assembly, Figure 1's historical raw snapshots, and R figure rendering are not fully verified. See `FIGURES.md`.
- GitHub Actions has been configured but has not run on GitHub. Local execution is the evidence above.

The workflow follows the currently documented v7 usage of [actions/checkout](https://github.com/actions/checkout) and [actions/setup-python](https://github.com/actions/setup-python), checked during preparation. Package versions from the tested venv are recorded in `env/validation-lock.txt`.

No original scientific code, split, frozen data array, analysis threshold, or numerical manuscript claim was changed to obtain a passing result. New script copies change file paths only. Newly generated results are kept outside the release and outside the input deposit.

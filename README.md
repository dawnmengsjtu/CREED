# CREED

Research code for **Measured once, used as many: quantifying how much multi-layer value a single omics layer can recover in functional prediction**.

CREED evaluates how much of the gain from a second measured omics layer can be recovered by a cross-fitted virtual layer or a small measured panel. The study uses a fixed development cohort of 421 cancer cell lines, six directed layer pairs, two deployment scenarios, drug-response and CRISPR endpoints, and a separate sealed-set evaluation.

This is a publication-preparation export dated **2026-10-14**. It preserves the 327 original Methods S1 payload files and adds portable launchers, pinned development dependencies, data checks, and runnable validation. **It does not yet establish end-to-end reproducibility of every historical experiment or of the final figure PDFs.** See [validation and limitations](docs/VALIDATION.md).

[中文说明](README.zh-CN.md) · [reproduction guide](docs/REPRODUCING.md) · [figure data](docs/FIGURES.md) · [release checklist](docs/RELEASE_CHECKLIST.md)

## Quick start: no study data required

Use Python 3.9. The recorded study version was 3.9.23; the clean local validation used 3.9.6, NumPy 2.0.2, and SciPy 1.13.1.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_release.py
python scripts/smoke_test.py
```

The smoke test generates synthetic inputs in a temporary directory, runs both scenarios through all five folds, and checks the original reproduction gates. It also runs the original shared-crossdrug synthetic tests. It is an engineering check, not evidence for the paper's results. It took about 13 seconds on the preparation workstation; other systems may differ.

Alternatively, use `conda env create -f env/development.yml`. A new CPU [Dockerfile](Dockerfile) is supplied, but its build has not been tested; it is not the historical private CUDA image.

## Study data and a real-data check

The reserved data DOI is [10.5281/zenodo.23299706](https://doi.org/10.5281/zenodo.23299706). Its unauthenticated public endpoint returned HTTP 404 on 2026-10-14. Until the authors publish the record, use the locally supplied directory containing the 27 registered `.tar` files. This GitHub export does not contain the 7.21 GB data payload.

```bash
# Once the record is public:
python scripts/data.py download --dest downloads

# Or use the archive directory supplied by the authors:
python scripts/data.py unpack --archives /path/to/archive-directory --dest data
python scripts/data.py verify --data-dir data
python scripts/replay_metrics.py --data-root data/runs
```

`unpack` checks every archive and all 788 deposited files, then installs the 174 small run records bundled here into their original run-relative paths. Existing different records are never replaced. The records complement the array deposit and overlap it where the same metadata was already deposited.

The independent metric replay checks **265 fold/arm sRMSE values** against the stored prediction arrays. A separate real-data validation retrained every cross-layer arm in both scenarios and compared **590 arrays**, with a maximum absolute numeric difference of **5.43 × 10⁻¹⁴**. Instructions to repeat that validation are in [REPRODUCING.md](docs/REPRODUCING.md).

## Repository contents

| Path | Contents |
|---|---|
| `original/` | Unmodified Methods S1 source snapshot, original README, and per-file `SOURCES.tsv` |
| `scripts/` | New portable launchers, data staging, integrity checks, and validation |
| `requirements.txt`, `env/` | Pinned development dependencies and recorded optional environments |
| `metadata/` | Data manifests, checksums, raw-source locations, and 174 traceable run records |
| `supplement/` | Data S1–S10, byte-identical to the 2026-10-14 submission package |
| `validation/` | Machine-readable results of checks actually performed |
| `docs/` | Reproduction commands, figure dependencies, limitations, and release steps |

Do not edit `original/` to change paths. `scripts/run.py` creates a separate script copy, changes the server run root, and records the original/adapted SHA-256 in `ADAPTATIONS.json`. It preserves the original scientific algorithms and input hash checks. Output goes to a fresh directory outside the data root.

## Scope, citation, and license

The sealed evaluation's code, freeze records, and summaries are preserved, but the 64 validation and 123 sealed cell-line assay values are not distributed. Sealed-set scoring cannot be rerun from this public deposit. The N6 CRISPR step saved fold results, not outer predictions. Some earlier experiments summarized only in Table S8 parts A and C are not fully archived.

The project owner explicitly chose to defer software licensing on 2026-10-14 and prepare the release package only. This export adds no software license grant. A final author list is also pending. [CITATION.cff.template](CITATION.cff.template) is a template to complete before renaming it `CITATION.cff`; it is not a valid final citation. Data and third-party components retain their own terms; see [upstream sources](metadata/RAW_SOURCES.tsv) and the original data-deposit documentation.

# Reproducing the development analyses

Run all commands from the repository root. Use a fresh environment, install `requirements.txt`, and first run `scripts/verify_release.py` and `scripts/smoke_test.py` as described in the README. All new tools use CPU and standard Python; the numerical stack is NumPy 2.0.2/SciPy 1.13.1.

## 1. Stage and verify inputs

Obtain all 28 registered archives. The reserved Zenodo record was not public at the preparation check; `scripts/data.py download` reports this clearly rather than pretending that the files were downloaded.

```bash
python scripts/data.py unpack --archives /path/to/archive-directory --dest data
python scripts/data.py verify --data-dir data --report work/data-verification.json
python scripts/replay_metrics.py --data-root data/runs --report work/metric-replay.json
```

The destination of `unpack` must not exist. It verifies archive SHA-256, refuses links and unregistered members, validates extracted bytes, and installs the bundled run records. For an already unpacked deposit, use `python scripts/data.py records --data-dir data` before `verify`; this only adds missing records and checks existing ones. `metadata/RECORDS.tsv` records the source and SHA-256 of each added record. Most are matched to the captured server inventories (2026-10-11, and 2026-10-22 for the prospective test); three sealed summaries are taken directly from Data S3. No assay values are reconstructed from summaries.

The runnable root is `data/runs`, not `data`. Use about 25 GB of free space for both the downloaded archives (10.6 GB) and extracted files, plus room for new outputs. Keep large arrays outside Git.

## 2. Retrain the full cross-layer matrix

```bash
python scripts/run.py crosslayer --data-root data/runs --scenario new_cell --stage main --out outputs/crosslayer/new_cell
python scripts/run.py crosslayer --data-root data/runs --scenario tissue --stage main --out outputs/crosslayer/tissue
python scripts/compare_crosslayer.py --data-root data/runs --results-root outputs/crosslayer --report work/crosslayer-comparison.json
```

Each output contains `code/` (the path-adapted source and hash record), `run.log`, and `results/`. All five folds and the original reproduction gates are run. The local preparation check took about 75 and 63 seconds respectively with one numerical thread per process; these are measured examples, not runtime guarantees.

For an initial check confined to fold 0's inner banks, use `--stage dryrun` and another new output directory. This deliberately does not compute outer scores.

To run the original statistical analysis on the new results, create its expected two-scenario layout without touching the stored data:

```bash
mkdir -p outputs/crosslayer/analysis
cp -R outputs/crosslayer/new_cell/results outputs/crosslayer/analysis/new_cell
cp -R outputs/crosslayer/tissue/results outputs/crosslayer/analysis/tissue
python original/03_development_runs_20261009/crosslayer-matrix-20261009-0001/code/analyze.py outputs/crosslayer/analysis
```

This writes `analysis.json`, comparison tables, and matrix tables in the analysis directory.

## 3. Other development entry points

The same launcher supports these original algorithms. Availability of a launcher is not a claim that every algorithm was retrained in the preparation check.

| Analysis | Stages | Additional argument / evidence |
|---|---|---|
| `crosslayer` | `dryrun`, `main` | Both scenarios fully retrained and compared |
| `generator-curve` | `dryrun`, `main` | Original source; not freshly retrained in this release check |
| `protein-panel`, `rna-panel` | `dryrun`, `repro`, `main` | Original source; read saved teacher, graph, or matrix assets |
| `crispr` | `dryrun`, `main` | Original source; not freshly retrained in this release check |
| `acquisition-protein`, `acquisition-rna` | `dryrun`, `main` | Original source; not freshly retrained in this release check |
| `context-panel` | `dryrun`, `fold` | `fold` requires `--fold 0` through `--fold 4` |
| `n6-drug`, `n6-crispr` | `dryrun`, `main` | `main` requires one `--fold` per invocation |

Example:

```bash
python scripts/run.py protein-panel --data-root data/runs --scenario new_cell --stage dryrun --out outputs/protein-dryrun
```

All stages write into a fresh directory. The original N6 code runs one fold per main invocation. Its per-fold outputs and analyses must be organized as documented in the original Methods S1 README; this release does not silently merge or replace frozen records. The launcher intentionally has no sealed-set scoring command.

The tissue-label, shared-crossdrug, RNA learning-curve, abstention, independent recomputation, earlier teacher/graph, CPTAC, and package-builder sources remain in `original/`; consult its detailed README for entry points and expected historical layouts. They are not all covered by the portable launcher. Scripts using container mounts such as `/oldcode`, `/code`, or historical raw inputs still require the recorded bindings. Some early inputs and execution records are unavailable.

## 4. Prospective test of the low-dimensional structure

The pre-registered prospective test (run `prospective-lowdim-20261022-0001`, Methods S1 component `09_prospective_test_20261022`) is in the deposit as one archive: the GDSC1 matrix and annotations, plus the outer predictions, per-target metrics, bank records and fold protocols of all 30 folds (GDSC1, and the drug-response and CRISPR re-runs). Its plan, addendum, gates and logs are in Data S2. The per-fold results and the five analysis files are bundled here as run records.

```bash
python scripts/replay_prospective.py --data-root data/runs --out work/prospective-replay --report work/prospective-replay.json
```

This copies the original `code/` folder, changes only the run root, links the deposited `output/` and `data_gdsc1/` folders read-only into a fresh run directory, runs the original `analyze.py` there, and compares the five outputs (`analysis.json`, `RESULT_AUTO.md`, `cells_A.csv`, `figure_2H_prospective.csv`, `panel_comparisons.csv`) byte for byte with the records. The analysis also reads the N6 per-fold results of both endpoints, which are bundled records too.

Refitting the prospective models (`lowdim.py`) is not wrapped by `run.py`. `lowdim.py` and `prepare_gdsc1.py` read `CREED_ROOT`; the registered stage order is in `code/run.sh` of that component and in the plan (`PLAN.md`, Data S2). The GDSC1 source file is not redistributed; its location and SHA-256 are in the deposit README (`docs/DATA_DEPOSIT_ORIGINAL.md`).

## 5. Interpret the checks correctly

`replay_metrics.py` recomputes sRMSE from saved predictions, not models from raw assays. `run.py crosslayer --stage main` does fit the models anew against the frozen development inputs. Neither validates the source-scale upstream preparation, independent external generalization, or sealed-set scores. Original path adaptation changes script hashes; the unmodified source snapshot remains available for matching the pre-registration hashes. Do not set `CREED_SKIP_SHA` or disable Python assertions for real-data work.

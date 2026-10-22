# Figure data and rendering

The source includes all retained Python data stages, R panel scripts, layouts, the renderer and the assembler. The previously missing connection between the array deposit and the small run records is now supplied by `scripts/data.py unpack`.

## Verified data-stage command

```bash
python -m pip install -r env/figures-python.txt
python scripts/prepare_figure_data.py --data-root data/runs --out work/figure-data
```

The destination must not exist. This copies the original data-stage scripts, adapts the run and figure roots, reconstructs two historical raw-file paths using byte-identical deposited files, and runs the original checks. It generated **59 CSV files in eight stages** in the preparation check. Logs, `INPUT_BINDINGS.json`, path-adaptation hashes, and `report.json` are retained in the work directory.

| Stage | Original script | Historical output folder / final use |
|---|---|---|
| `map` | `prepare_fig2_v2.py` | `figure2/plot_data`: cross-layer map, final Figure 2 |
| `generator` | `prepare_figS1_v2.py` | `figure8/plot_data`: learning curves and universe; final Figure 2G and Figure S1 (the universe panel is Figure S1D) |
| `drug-detail` | `prepare_fig3_v2.py` | `figure3/plot_data`: per-drug, sample and ranking tables; final Figure 3 and Figure S2 |
| `graph-controls` | `prepare_fig4_v2.py` | `figure4/plot_data`: graph/tissue controls; final Figure 3 and Figure S2 |
| `panels` | `prepare_fig5_v2.py` | `figure5/plot_data`: panel and acquisition/abstention tables; final Figure 4 and Figure S3 |
| `panel-detail` | `prepare_fig6_v2.py` | `figure6/plot_data`: drug annotations and panel details; final Figure S4 |
| `crispr` | `prepare_fig7_v2.py` | `figure7/plot_data`: final Figure 5 |
| `sealed-summary` | `prepare_fig8_v2.py` | `figure9/plot_data`: final Figure 6, using the saved Data S3 summary |

Paths above are relative to `work/figure-data/staging/delivery_R_20261009/figures2_6/`. Check actual directory names in the generated report/tree if using a different historical script. `--stages` can select a subset, but dependencies remain: for example, `crispr` reads tables produced by `map` and `panels`.

## Figure 2H (2026-10-22)

Figure 2H shows the measured-layer controls of the prospective test. Its plot table `lowdim_controls.csv` is built by `original/09_prospective_test_20261022/prospective_lowdim_20261022/figure_2H/build_fig2h_controls.py`, which copies values verbatim from the run's `analysis/analysis.json` (a bundled run record; `scripts/replay_prospective.py` regenerates it). The script reads that file at `../server_output/analysis/analysis.json` relative to itself, the layout of the authors' working copy; point `SRC` at `data/runs/prospective-lowdim-20261022-0001/analysis/analysis.json` instead. The panel script is `r/Figure2/H.R`. It was not rendered in this release check.

## Final V4 layout is a separate stage

The final R scripts are at `original/07_figures/figures_v4/20261010_figures_v4_six/r/`. They expect `out/<Figure>/plot_data` alongside `r/`, containing a selected mixture of these historical tables and several extra tables. The original frozen V4 `plot_data` directories were not included in the source archive. The preparation check did not recreate every final input or render the final PDF pages.

Examples of additional work required:

- Figure 1's original preparation script reads three historical raw bundles, including `gdsc_metabolite_query` and `gdsc_rna_inverse`. Those exact bundles are not in the current deposit. The current frozen M/P/AUC and aligned RNA matrices do not by themselves recreate the original snapshot provenance.
- Figures 2, 4, 5 and 6 reuse Figure 1's `A_real_array_snippets.csv`; Figure 5 uses prefixed copies of Figure 2 tables and additional CRISPR fold tables.
- Figure 4 adds `g_cov50_fold`, `h_aurc_fold` and `g_n3_vs_random` through the two Python extractors in `r/Figure4/prep/`.
- Figure 6 adds `dev_folds.csv` through `r/Figure6/prep_dev_folds.R` and uses the saved sealed summary, not newly evaluated sealed values.
- Figure S5 uses the separate original `prepare_cptac.py` data stage; it was not included in the eight-stage run above.

Historical R source folder names predate the final supplementary numbering: source `FigureS4` maps to final S1, `FigureS1` to S2, `FigureS3` to S3, `FigureS2` to S4, and `FigureS5` to S5. Keep the historical names while rendering; rename delivered artifacts only according to this mapping.

Once the inputs are assembled, run the original `render_v4.R` and `assemble_v4.R` with their documented arguments. Do not call the historical `run.sh` as an all-in-one current pipeline: some historical render/check files referenced by it are not retained in this archive.

Recorded R dependencies are in `env/figures-packages.tsv`. R 4.4.3 and its graphics dependencies were unavailable in the local release-preparation environment, so final graphics installation, rendering, and visual QA remain unverified.

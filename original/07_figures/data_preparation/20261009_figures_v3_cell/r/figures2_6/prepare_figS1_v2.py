#!/usr/bin/env python
"""Data stage for Supplementary Figure S1 (FIG id 8 of the V2 pipeline).

Panel A  generator learning curve: recovery share of the measured-layer gap vs the
         number of generator training cells n, four series (virtual RNA / virtual
         protein x new cell / held-out tissue), 5-fold means + per-fold points.
Panel B  the pairing universe ladder (how many cell lines could ever be added).

Inputs (READ ONLY):
  RUNC/output/{summary.json,fold_metrics.csv,comparisons.csv}   generator-curve-20261009-0001
  RUNF/pack/qa.json                                            generator-fullpool-20261009-0001

Every number written here is asserted against the binding tables of
  generator_curve_20261009/DECISION.md   (section 2 learning-curve table, section 2 slopes)
  generator_fullpool_20261009/pack/EXTRACTION.md (sections 1, 2.3, 5)
so a silent drift in an upstream file fails the run instead of reaching the figure.

Writes only into
  $V2/delivery_R_20261009/figures2_6/figure8/plot_data/
"""
import csv
import hashlib
import json
import os
import sys

import numpy as np

V2 = "/public/home/mengxl/workspaces/creed-research/figure_work/20261009_figures2_6_v2"
RUNC = "/public/home/mengxl/creed-research-runs/generator-curve-20261009-0001"
RUNF = "/public/home/mengxl/creed-research-runs/generator-fullpool-20261009-0001"
OUT = os.path.join(V2, "delivery_R_20261009/figures2_6/figure8/plot_data")

SRC = {
    "summary.json": os.path.join(RUNC, "output/summary.json"),
    "fold_metrics.csv": os.path.join(RUNC, "output/fold_metrics.csv"),
    "comparisons.csv": os.path.join(RUNC, "output/comparisons.csv"),
    "qa.json": os.path.join(RUNF, "pack/qa.json"),
    "EXTRACTION.md": os.path.join(RUNF, "pack/EXTRACTION.md"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(name, rows, cols):
    path = os.path.join(OUT, name + ".csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return path, len(rows)


def ols_slope(x, y):
    """Least-squares slope of y on x (per unit x)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc * xc).sum())


# ----------------------------------------------------------------- inputs
os.makedirs(OUT, exist_ok=True)
summary = json.load(open(SRC["summary.json"]))
qa = json.load(open(SRC["qa.json"]))
comparisons = list(csv.DictReader(open(SRC["comparisons.csv"])))
fold_metrics = list(csv.DictReader(open(SRC["fold_metrics.csv"])))

checks = []


def check(label, got, want, tol=0.0):
    ok = abs(float(got) - float(want)) <= tol
    checks.append((label, got, want, ok))
    if not ok:
        raise SystemExit("CHECK FAILED %s: got %r want %r (tol %r)" % (label, got, want, tol))


# ============================================================== PANEL A
# series definition: (scenario, layer) -> labels used in the figure
SERIES = [
    ("new_cell", "RNA", "vRNA_new_cell", "Virtual RNA, new cell lines", "Virtual RNA", "New cell lines"),
    ("new_cell", "P", "vP_new_cell", "Virtual protein, new cell lines", "Virtual protein", "New cell lines"),
    ("tissue", "RNA", "vRNA_tissue", "Virtual RNA, held-out tissue", "Virtual RNA", "Held-out tissue"),
    ("tissue", "P", "vP_tissue", "Virtual protein, held-out tissue", "Virtual protein", "Held-out tissue"),
]
SIZE_ORDER = ["f50", "f75", "f100", "extra"]
SIZE_LABEL = {"f50": "50% of fit cells", "f75": "75% of fit cells",
              "f100": "100% of fit cells", "extra": "100% + extra paired lines"}

# binding values, transcribed from DECISION.md section 2 (learning-curve table)
DEC = {
    # (scenario, layer): headroom_pct, {size: (n_mean, gain_pct, share_pct_rounded, wins, p_rounded)}
    ("new_cell", "RNA"): (3.079, {
        "f50": (168.8, 0.698, 23, 5, 0.004), "f75": (252.8, 0.912, 30, 5, 0.020),
        "f100": (336.8, 1.056, 34, 5, 0.005), "extra": (402.8, 1.200, 39, 5, 0.002)}),
    ("new_cell", "P"): (2.401, {
        "f50": (168.8, 0.426, 18, 5, 0.004), "f75": (252.8, 0.563, 23, 5, 0.001),
        "f100": (336.8, 0.700, 29, 5, 0.000)}),
    ("tissue", "RNA"): (2.195, {
        "f50": (168.6, 0.608, 28, 5, 0.024), "f75": (252.8, 0.715, 33, 5, 0.032),
        "f100": (336.8, 0.716, 33, 5, 0.018), "extra": (386.4, 0.778, 35, 5, 0.032)}),
    ("tissue", "P"): (2.207, {
        "f50": (168.6, 0.392, 18, 5, 0.018), "f75": (252.8, 0.643, 29, 5, 0.005),
        "f100": (336.8, 0.729, 33, 5, 0.013)}),
}
# Slopes. RESULT.md line 148 defines both as the MEAN +- SD of the five per-fold
# least-squares slopes through f50/f75/f100; DECISION.md section 2 prints the gain slope
# rounded. gain slope = pp of relative sRMSE improvement per 100 generator cells;
# share slope = share percentage points per 100 generator cells. Different units.
DEC_GAIN_SLOPE = {  # (mean, sd) from RESULT.md line 148, printed label from DECISION.md s2
    "vRNA_new_cell": (0.213, 0.078, "+0.21"), "vP_new_cell": (0.163, 0.049, "+0.16"),
    "vRNA_tissue": (0.065, 0.049, "+0.07"), "vP_tissue": (0.203, 0.183, "+0.20")}
DEC_SHARE_SLOPE = {"vRNA_new_cell": (7.3, 3.7), "vP_new_cell": (7.3, 3.2),
                   "vRNA_tissue": (3.6, 3.4), "vP_tissue": (10.0, 7.6)}

mean_rows, fold_rows, slope_rows = [], [], []
for scen, layer, key, series_label, layer_label, scen_label in SERIES:
    L = summary[scen]["layers"][layer]
    headroom_pct = L["headroom"]["gain"]
    dec_head, dec_sizes = DEC[(scen, layer)]
    check("headroom %s" % key, headroom_pct, dec_head, 5e-4)
    head_fold = L["headroom"]["per_fold"]

    ns, gains, shares_pp = [], [], []
    for size in SIZE_ORDER:
        if size not in L["sizes"]:
            continue
        S = L["sizes"][size]
        rec = S["recovered"]
        n_mean = float(np.mean(S["n_gen_outer"]))
        dn_mean, dgain, dshare, dwins, dp = dec_sizes[size]
        check("n_gen %s %s" % (key, size), n_mean, dn_mean, 5e-2)
        check("gain %s %s" % (key, size), rec["gain"], dgain, 5e-4)
        check("share %s %s" % (key, size), round(S["share"] * 100), dshare, 0.0)
        check("wins %s %s" % (key, size), rec["wins"], dwins, 0.0)
        check("p %s %s" % (key, size), round(rec["p"], 3), dp, 0.0)
        # share = 5-fold-mean gain / 5-fold-mean headroom (the DECISION definition)
        check("share==gain/headroom %s %s" % (key, size),
              S["share"], rec["gain"] / headroom_pct, 1e-12)

        mean_rows.append(dict(
            series=key, series_label=series_label, layer_label=layer_label,
            scenario=scen, scenario_label=scen_label, layer=layer, size=size,
            size_label=SIZE_LABEL[size], n_gen_mean=round(n_mean, 1),
            share=S["share"], share_pct=100.0 * S["share"], gain_pct=rec["gain"],
            headroom_pct=headroom_pct, folds_won=rec["wins"], folds_total=rec["k"],
            p_nb=rec["p"], is_extra=int(size == "extra"),
            oof_r2=S["oof_r2_outer_mean"]))
        for f, (nf, sf, gf) in enumerate(zip(S["n_gen_outer"], S["per_fold_share"],
                                             rec["per_fold"])):
            # per-fold share = per-fold gain / per-fold headroom
            check("per-fold share %s %s f%d" % (key, size, f), sf, gf / head_fold[f], 1e-9)
            fold_rows.append(dict(
                series=key, series_label=series_label, layer_label=layer_label,
                scenario=scen, scenario_label=scen_label, layer=layer, size=size,
                fold=f, n_gen_fold=nf, share=sf, share_pct=100.0 * sf,
                gain_pct=gf, headroom_fold_pct=head_fold[f],
                is_extra=int(size == "extra")))
        ns.append(n_mean)
        gains.append(rec["gain"])
        shares_pp.append(100.0 * S["share"])

    # slopes are fitted on the three nested generator sizes (f50/f75/f100) only:
    # the extra arm adds lines from a different pool, so it is not part of the law fit
    nested = [i for i, s in enumerate(SIZE_ORDER[:3]) if s in L["sizes"]]
    gain_pooled = 100.0 * ols_slope([ns[i] for i in nested], [gains[i] for i in nested])
    gain_fold_slopes, share_fold_slopes = [], []
    for f in range(5):
        xs = [L["sizes"][SIZE_ORDER[i]]["n_gen_outer"][f] for i in nested]
        gain_fold_slopes.append(100.0 * ols_slope(
            xs, [L["sizes"][SIZE_ORDER[i]]["recovered"]["per_fold"][f] for i in nested]))
        share_fold_slopes.append(100.0 * ols_slope(
            xs, [100.0 * L["sizes"][SIZE_ORDER[i]]["per_fold_share"][f] for i in nested]))
    gn_m, gn_s = float(np.mean(gain_fold_slopes)), float(np.std(gain_fold_slopes, ddof=1))
    sh_m, sh_s = float(np.mean(share_fold_slopes)), float(np.std(share_fold_slopes, ddof=1))
    dec_m, dec_s, dec_label = DEC_GAIN_SLOPE[key]
    check("gain slope mean %s" % key, round(gn_m, 3), dec_m, 0.0)
    check("gain slope sd %s" % key, round(gn_s, 3), dec_s, 0.0)
    # DECISION s2 prints the slope rounded from the 3-decimal RESULT.md value
    check("gain slope label %s" % key, float(dec_label), round(gn_m, 3), 5.1e-3)
    check("share slope mean %s" % key, round(sh_m, 1), DEC_SHARE_SLOPE[key][0], 0.0)
    check("share slope sd %s" % key, round(sh_s, 1), DEC_SHARE_SLOPE[key][1], 0.0)
    slope_rows.append(dict(
        series=key, series_label=series_label, layer_label=layer_label,
        scenario=scen, scenario_label=scen_label,
        gain_slope_pp_per100=gn_m, gain_slope_sd_pp_per100=gn_s,
        gain_slope_label=dec_label, gain_slope_pooled_pp_per100=gain_pooled,
        share_slope_mean_pp_per100=sh_m, share_slope_sd_pp_per100=sh_s,
        share_slope_label="%+.1f ± %.1f" % (sh_m, sh_s),
        n_fit_sizes=len(nested), headroom_pct=DEC[(scen, layer)][0],
        n_last=max(ns), share_pct_last=shares_pp[-1]))

# fold_metrics.csv cross-check: the arm-level 5-fold-mean sRMSE behind every gain
ARM = {("new_cell", "RNA"): "M+vRNA", ("new_cell", "P"): "M+vP",
       ("tissue", "RNA"): "M+vRNA", ("tissue", "P"): "M+vP"}
fm = {}
for r in fold_metrics:
    fm[(r["scenario"], r["arm"], int(r["fold"]))] = float(r["score"])
for scen, layer, key, *_ in SERIES:
    base = [fm[(scen, "M", f)] for f in range(5)]
    for size in SIZE_ORDER:
        arm = "%s_%s" % (ARM[(scen, layer)], size)
        if (scen, arm, 0) not in fm:
            continue
        new = [fm[(scen, arm, f)] for f in range(5)]
        # gain = mean over folds of the per-fold relative sRMSE improvement
        per_fold = [100.0 * (b - n) / b for b, n in zip(base, new)]
        rows_f = [x for x in fold_rows if x["series"] == key and x["size"] == size]
        for f, g in enumerate(per_fold):
            check("fold_metrics per-fold gain %s %s f%d" % (key, size, f),
                  g, rows_f[f]["gain_pct"], 1e-9)
        row = [m for m in mean_rows if m["series"] == key and m["size"] == size][0]
        check("fold_metrics gain %s %s" % (key, size),
              float(np.mean(per_fold)), row["gain_pct"], 1e-9)

# comparisons.csv cross-check of the same gains / shares
for r in comparisons:
    if r["role"] != "recovered":
        continue
    key = {("new_cell", "RNA"): "vRNA_new_cell", ("new_cell", "P"): "vP_new_cell",
           ("tissue", "RNA"): "vRNA_tissue", ("tissue", "P"): "vP_tissue"}[(r["scenario"], r["layer"])]
    size = r["id"].replace("recovered_", "")
    row = [m for m in mean_rows if m["series"] == key and m["size"] == size][0]
    check("comparisons gain %s %s" % (key, size), float(r["gain_pct"]), row["gain_pct"], 1e-12)
    check("comparisons share %s %s" % (key, size), float(r["share"]), row["share"], 1e-12)

# L1 / L2 verdict rows (DECISION section 3) for the caption and the extra-point note
L2 = summary["new_cell"]["layers"]["RNA"]["L2"]
verdict_rows = [
    dict(scenario="new_cell", layer="RNA", test="L1", verdict="flat",
         detail="f75-f50 +0.215% (3/5, p=0.206); f100-f75 +0.145% (4/5, p=0.149)"),
    dict(scenario="new_cell", layer="P", test="L1", verdict="rising",
         detail="f75-f50 +0.137% (4/5, p=0.188); f100-f75 +0.138% (4/5, p=0.132)"),
    dict(scenario="tissue", layer="RNA", test="L1", verdict="flat",
         detail="f75-f50 +0.107% (4/5, p=0.598); f100-f75 +0.001% (2/5, p=0.995)"),
    dict(scenario="tissue", layer="P", test="L1", verdict="flat",
         detail="f75-f50 +0.252% (4/5, p=0.111); f100-f75 +0.087% (3/5, p=0.647)"),
    dict(scenario="new_cell", layer="RNA", test="L2", verdict="extra lines help (weak)",
         detail="extra-f100 +0.146% (5/5, p=0.058), CI [-0.01,+0.30], bootstrap P+ 0.958"),
    dict(scenario="tissue", layer="RNA", test="L2", verdict="not detectable",
         detail="extra-f100 +0.062% (4/5, p=0.577)"),
]

# ============================================================== PANEL B
U = qa["universe"]
sealed = qa["sealed"]
pool = qa["pool"]
check("RNA header", U["RNA_header"], 1047)
check("P header", U["P_header"], 949)
check("M source rows", qa["metabolomics"]["rows"], 928)
check("M unique SIDM (bridge)", qa["metabolomics"]["unique_sidm_bridge"], 733)
check("M and RNA", pool["candidates_with_rna"], 683)
check("M and RNA (universe)", U["bridge"]["M_and_RNA"]["n"], 683)
check("sealed union", sealed["union"], 608)
check("sealed train", sealed["train"], 421)
check("sealed validation", sealed["validation"], 64)
check("sealed internal_final", sealed["internal_final"], 123)
check("train+val+final == 608", sealed["train"] + sealed["validation"] + sealed["internal_final"], 608)
check("excluded 608", pool["excluded"]["sealed_608"], 608)
check("same-patient exclusions", pool["excluded"]["shares_patient_with_608"], 8)
check("pool n", pool["n"], 67)
check("683-608-8 == 67", 683 - 608 - 8, 67)
check("audited 66 subset of pool", int(qa["audited66"]["subset_of_pool"]), 1)
check("pool minus audited66 == 1", len(qa["audited66"]["pool_not_in_66"]), 1)
check("feasibility threshold", qa["feasibility"]["threshold"], 150)
check("feasibility proceed", int(qa["feasibility"]["proceed"]), 0)
# EXTRACTION section 5: M and P universe = 609 under the protocol (mmc2, whole-string)
# convention; 611 (mmc2 split) and 612 (mmc2 + Model.csv bridge) under looser bridges;
# M and RNA and P outside the 608 = 0 under the protocol convention
check("M (protocol convention)", qa["m_mmc2_only"]["whole_string_convention"], 609)
check("M (protocol figure)", qa["m_mmc2_only"]["protocol_figure"], 609)
check("M and P outside 608 (protocol)", len(qa["m_mmc2_only"]["whole_outside_608"]), 1)
check("M split convention", qa["m_mmc2_only"]["split_convention"], 611)
check("M and P (bridge)", U["bridge"]["M_and_P"]["n"], 612)
check("extraction gate M diff", qa["gates"]["m_max_abs_diff"], 0.0)
check("extraction gate RNA diff", qa["gates"]["rna_max_abs_diff"], 1.8e-15, 5e-17)
check("extraction gate parity lines", len(qa["gates"]["parity_ids"]), 5)
check("extraction gates passed", int(qa["gates"]["passed"]), 1)
MRP_PROTOCOL = 608  # 609 M-and-P minus SIDM00046, which has no RNA column
assert qa["m_mmc2_only"]["whole_outside_608"] == ["SIDM00046"]
assert "not_rna_candidate" in json.dumps(U["mmc2_only"]["M_and_P"]["outside_excluded_reasons"])

# one row per bar segment, top (order 1) to bottom (order 8)
ladder = [
    dict(order=1, bar="RNA (transcriptome) lines", seg="total", value=U["RNA_header"],
         fill_key="layer", n_label="1,047", seg_label="",
         note="columns of the CCLE RNA-seq table"),
    dict(order=2, bar="Metabolomics source rows", seg="total", value=qa["metabolomics"]["rows"],
         fill_key="layer", n_label="928", seg_label="",
         note="rows of the CCLE metabolomics table"),
    dict(order=3, bar="\u2192 mapped to unique cell lines", seg="total",
         value=qa["metabolomics"]["unique_sidm_bridge"], fill_key="mapped",
         n_label="733", seg_label="",
         note="195 source rows unmappable, 0 bridge conflicts"),
    dict(order=4, bar="M ∩ RNA", seg="total", value=pool["candidates_with_rna"],
         fill_key="pair", n_label="683", seg_label="",
         note="both layers present on disk"),
    dict(order=5, bar="M ∩ P (identity only)", seg="total",
         value=qa["m_mmc2_only"]["protocol_figure"], fill_key="pair", n_label="609",
         seg_label="",
         note="protocol convention (mmc2); 611–612 under looser bridges"),
    dict(order=6, bar="M ∩ RNA ∩ P, used", seg="train", value=sealed["train"],
         fill_key="train", n_label="421", seg_label="421",
         note="608 under the protocol convention: 421 development,",
         note2="64 + 123 sealed; the one M ∩ P line outside has no RNA"),
    dict(order=6, bar="M ∩ RNA ∩ P, used", seg="validation", value=sealed["validation"],
         fill_key="sealed_v", n_label="64", seg_label="64", note="", note2=""),
    dict(order=6, bar="M ∩ RNA ∩ P, used", seg="internal_final",
         value=sealed["internal_final"], fill_key="sealed_f", n_label="123",
         seg_label="123", note="", note2=""),
    dict(order=7, bar="Extra identity-safe M + RNA", seg="total", value=pool["n"],
         fill_key="extra", n_label="67", seg_label="",
         note="683 − 608 used/sealed − 8 same PatientID; 66 audited,",
         note2="+ 1 ambiguous. 67 < 150 threshold → scale-up infeasible"),
    dict(order=8, bar="Extra M + RNA + P", seg="total", value=0, fill_key="zero",
         n_label="0", seg_label="",
         note="no extra M + RNA + P line exists under the protocol convention"),
]
for r in ladder:
    r["bar_total"] = sum(x["value"] for x in ladder if x["order"] == r["order"])
    r["y"] = 9 - r["order"]
    r.setdefault("note2", "")
    # cumulative left edge of each segment within its bar
    r["x0"] = sum(x["value"] for x in ladder
                  if x["order"] == r["order"] and ladder.index(x) < ladder.index(r))
    r["x1"] = r["x0"] + r["value"]
check("ladder 608 bar", [r for r in ladder if r["order"] == 6][0]["bar_total"], MRP_PROTOCOL)

notes = dict(
    identity_authority="identity authority: ProCan mmc2; bridge: DepMap Model.csv",
    mrp_protocol=MRP_PROTOCOL,
    mp_protocol=qa["m_mmc2_only"]["protocol_figure"],
    mp_loose_split=qa["m_mmc2_only"]["split_convention"],
    mp_loose_bridge=U["bridge"]["M_and_P"]["n"],
    extra_mrna=pool["n"], extra_mrna_audited=pool["n"] - 1,
    extra_mrnap=0, same_patient_excluded=pool["excluded"]["shares_patient_with_608"],
    feasibility_threshold=qa["feasibility"]["threshold"],
    feasibility_proceed=bool(qa["feasibility"]["proceed"]),
    extra_point_label="all identity-safe extra lines (67 incl. one ambiguous)",
    extra_used_in_curve=66,
    extra_tissue_per_fold_curve=[51, 44, 52, 51, 50],
    extra_tissue_per_fold_pool67=[qa["usable_tissue_per_fold"]["fold-%d" % f] for f in range(5)],
    gate_m_diff=qa["gates"]["m_max_abs_diff"],
    gate_rna_diff=qa["gates"]["rna_max_abs_diff"],
    gate_parity_lines=len(qa["gates"]["parity_ids"]),
    third=100.0 / 3.0,
    l2_new_cell_rna=dict(
        label=L2["label"], helps=bool(L2["helps"]),
        beats_paired_shuffle=bool(L2["beats_paired_shuffle"]),
        delta_pct=L2["extra_vs_f100"]["gain"], wins=L2["extra_vs_f100"]["wins"],
        p=L2["extra_vs_f100"]["p"], ci=L2["extra_vs_f100"]["ci"],
        vs_shuffle_pct=L2["extra_vs_extra_shuf"]["gain"],
        vs_shuffle_p=L2["extra_vs_extra_shuf"]["p"]),
)
check("L2 new_cell RNA delta", L2["extra_vs_f100"]["gain"], 0.146, 5e-4)
check("L2 new_cell RNA p", round(L2["extra_vs_f100"]["p"], 3), 0.058, 0.0)
check("L2 new_cell RNA wins", L2["extra_vs_f100"]["wins"], 5, 0.0)
check("L2 vs paired shuffle", L2["extra_vs_extra_shuf"]["gain"], 0.427, 5e-4)
check("L2 tissue RNA delta", summary["tissue"]["layers"]["RNA"]["L2"]["extra_vs_f100"]["gain"],
      0.062, 5e-4)

# ----------------------------------------------------------------- write
written = []
written.append(write_csv("a_curve_mean", mean_rows, [
    "series", "series_label", "layer_label", "scenario", "scenario_label", "layer",
    "size", "size_label", "n_gen_mean", "share", "share_pct", "gain_pct",
    "headroom_pct", "folds_won", "folds_total", "p_nb", "is_extra", "oof_r2"]))
written.append(write_csv("a_curve_fold", fold_rows, [
    "series", "series_label", "layer_label", "scenario", "scenario_label", "layer",
    "size", "fold", "n_gen_fold", "share", "share_pct", "gain_pct",
    "headroom_fold_pct", "is_extra"]))
written.append(write_csv("a_slopes", slope_rows, [
    "series", "series_label", "layer_label", "scenario", "scenario_label",
    "gain_slope_pp_per100", "gain_slope_sd_pp_per100", "gain_slope_label",
    "gain_slope_pooled_pp_per100", "share_slope_mean_pp_per100",
    "share_slope_sd_pp_per100", "share_slope_label", "n_fit_sizes", "headroom_pct",
    "n_last", "share_pct_last"]))
written.append(write_csv("a_verdicts", verdict_rows, ["scenario", "layer", "test", "verdict", "detail"]))
written.append(write_csv("b_ladder", ladder, [
    "order", "y", "bar", "seg", "value", "x0", "x1", "bar_total", "fill_key",
    "n_label", "seg_label", "note", "note2"]))
with open(os.path.join(OUT, "figS1_notes.json"), "w") as fh:
    json.dump(notes, fh, indent=1)
written.append((os.path.join(OUT, "figS1_notes.json"), len(notes)))

doc = os.path.join(OUT, "FIGS1_DATA_SOURCES.md")
with open(doc, "w") as fh:
    fh.write("# Supplementary Figure S1 (FIG 8) - data sources\n\n")
    fh.write("Written by `r/figures2_6/prepare_figS1_v2.py` (sha256 "
             "%s) on this host; python %s.\n\n" % (sha256(os.path.abspath(__file__))[:12],
                                                   sys.version.split()[0]))
    fh.write("## Inputs (read only)\n\n| file | sha256 |\n|---|---|\n")
    for k in ("summary.json", "fold_metrics.csv", "comparisons.csv", "qa.json", "EXTRACTION.md"):
        fh.write("| `%s` | `%s` |\n" % (SRC[k], sha256(SRC[k])))
    fh.write("\n`summary.json`, `fold_metrics.csv`, `comparisons.csv` are the outputs of "
             "`generator-curve-20261009-0001` (learning curve); `qa.json` / `EXTRACTION.md` "
             "of `generator-fullpool-20261009-0001` (pairing universe). No protected set was "
             "opened: the pairing universe is read as counts from `qa.json` only.\n")
    fh.write("\n## Outputs\n\n| table | rows |\n|---|---:|\n")
    for p, n in written:
        fh.write("| `%s` | %d |\n" % (os.path.basename(p), n))
    fh.write("\n## Cross-checks against the binding tables (all passed)\n\n")
    fh.write("%d assertions. Panel A against DECISION.md section 2 (n, gain, share, "
             "folds won, p, gain slope, share slope) and the share identity "
             "share = 5-fold-mean gain / 5-fold-mean headroom; the per-fold shares against "
             "per-fold gain / per-fold headroom; the gains re-derived independently from "
             "`fold_metrics.csv` arm sRMSE and from `comparisons.csv`. Panel B against "
             "EXTRACTION.md sections 1, 2.3 and 5 and `qa.json`.\n\n" % len(checks))
    fh.write("| check | value | expected |\n|---|---:|---:|\n")
    for label, got, want, _ in checks:
        fh.write("| %s | %s | %s |\n" % (label, got, want))
    fh.write("\n## Reading notes written into the panels\n\n")
    fh.write("* Panel A y axis is the **recovery share** (gain / measured-layer headroom); "
             "the slope printed in the panel is the **gain** slope in percentage points of "
             "relative sRMSE improvement per 100 generator cells (DECISION.md section 2). The two "
             "have different units - the share slopes are %s.\n"
             % "; ".join("%s %s" % (r["series"], r["share_slope_label"]) for r in slope_rows))
    fh.write("* The +66 point is the whole remaining pairing universe: 67 identity-safe extra "
             "M+RNA lines exist, 66 were audited and used in the curve arm, the 67th "
             "(`SIDM01095`) is identity-ambiguous at the DepMap level. For the held-out-tissue "
             "scenario the extra arm adds only the tissue-eligible subset (44-52 per fold), so "
             "its mean n is 386.4 and not 402.8.\n")
    fh.write("* Virtual protein has no extra point: 0 extra M+RNA+P lines exist under the "
             "protocol convention, so the generator scale-up was infeasible below the "
             "pre-registered 150-line threshold and only the +66 curve point exists.\n")
print("WROTE", len(written), "tables into", OUT)
print("CHECKS PASSED:", len(checks))
print("DOC", doc)

"""POST-HOC (not pre-registered; descriptive only). Compare crosslayer arms with saved project arms on identical folds,
check M+Pval against the strong-teacher measured_p predictions, LOFO minima for primary cells, alpha-boundary counts.
Reads only: this run's output, extra-supervision predictions.npz, strong-teacher predictions.npz. Writes only to posthoc/."""
import json
import numpy as np
from scipy.stats import t as tdist

R = "/public/home/mengxl/creed-research-runs"
RUN = R + "/crosslayer-matrix-20261009-0001"
OUT = RUN + "/output"
EXT = R + "/extra-supervision-20261008-0001/output"
ST = R + "/strong-teacher-recoverability-20261008-0001/output"
ST_DIR = {"new_cell": "new_cell", "tissue": "tissue_v2"}
LAYERS = ("M", "RNA", "P")


def score(pred, truth, sd, use):
    obs = np.isfinite(truth)
    mse = np.where(obs, ((pred - truth) / sd) ** 2, 0).sum(0) / np.maximum(obs.sum(0), 1)
    return float(np.sqrt(mse[use].mean()))


def nb(a, b, r):
    a, b = np.array(a), np.array(b)
    d = (b - a) / b * 100; m = d.mean(); se = np.sqrt((1 / 5 + r) * d.var(ddof=1))
    t = m / se if se > 0 else np.inf; p = float(2 * tdist.sf(abs(t), 4)); q = tdist.ppf(.975, 4)
    return dict(gain=round(float(m), 4), wins=int((a < b).sum()), p=round(p, 4),
                ci=[round(float(m - q * se), 3), round(float(m + q * se), 3)], per_fold=[round(float(x), 3) for x in d],
                lofo_min=round(float(min(np.delete(d, i).mean() for i in range(5))), 4), a_mean=round(float(a.mean()), 6), b_mean=round(float(b.mean()), 6))


res = {}
for sc in ("new_cell", "tissue"):
    mine, saved, ratio, teacher_diff = {}, {}, [], 0.0
    for f in range(5):
        Z = np.load(f"{OUT}/{sc}/fold-{f}_predictions.npz", allow_pickle=False)
        P = np.load(f"{EXT}/{sc}/fold-{f}/predictions.npz", allow_pickle=False)
        T = np.load(f"{ST}/{ST_DIR[sc]}/fold-{f}/predictions.npz", allow_pickle=False)
        truth, sd, use = Z["truth"], Z["train_sd"], Z["evaluated"]
        assert np.array_equal(P["held"], Z["held"]) and np.array_equal(T["held"], Z["held"]) and np.array_equal(T["panel"], Z["panel"])
        ratio.append(len(Z["held"]) / len(Z["fit"]))
        for k in Z.files:
            if k.startswith("pred__"):
                mine.setdefault(k[6:], []).append(score(Z[k], truth, sd, use))
        for k in ("original__m_only", "original__m_only__protein", "original__virtual_fm", "original__virtual_fm__protein",
                  "extra__m_only__protein", "extra__virtual_fm__protein"):
            saved.setdefault(k, []).append(score(P[k], truth, sd, use))
        saved.setdefault("strong_teacher_measured_p", []).append(score(T["measured_p"], truth, sd, use))
        teacher_diff = max(teacher_diff, float(np.max(np.abs(Z["pred__M+Pval"] - T["measured_p"]))))
    r = float(np.mean(ratio))
    out = dict(r=r, teacher_pred_max_abs_diff=teacher_diff, saved_means={k: round(float(np.mean(v)), 6) for k, v in saved.items()},
               saved_folds={k: [round(x, 4) for x in v] for k, v in saved.items()}, comparisons={})
    S = json.load(open(f"{OUT}/{sc}/summary.json"))
    for a, fm in S["fold_scores"].items():
        assert np.allclose(fm, mine[a]), a
    pairs = [("M+vRNA", "original__m_only__protein"), ("M+vP", "original__m_only__protein"), ("M+vRNA", "original__virtual_fm"),
             ("M+vP", "original__virtual_fm"), ("M+vRNA", "original__virtual_fm__protein"), ("M+vP", "original__virtual_fm__protein"),
             ("M+vRNA", "extra__m_only__protein"), ("M+vP", "extra__m_only__protein"), ("M+vRNA", "M+vP"), ("P", "RNA"),
             ("RNA+P", "RNA"), ("RNA+Pval", "RNA"), ("M+Pval", "M+P"), ("M+Pval", "strong_teacher_measured_p"), ("M", "null_drug"),
             ("RNA", "null_drug"), ("P", "null_drug")]
    if sc == "new_cell":
        pairs += [("M+vRNA", "T"), ("M+vP", "T"), ("M+T", "T"), ("M+T", "M+vRNA"), ("M+vRNA", "M+T")]
    for a, b in pairs:
        fa = mine[a]; fb = mine[b] if b in mine else saved[b]
        out["comparisons"]["%s vs %s" % (a, b)] = nb(fa, fb, r)
    for x in LAYERS:
        for y in LAYERS:
            if x == y:
                continue
            out["comparisons"]["recovered %s->%s" % (x, y)] = nb(mine["%s+v%s" % (x, y)], mine[x], r)
            out["comparisons"]["headroom %s->%s" % (x, y)] = nb(mine["+".join(l for l in LAYERS if l in (x, y))], mine[x], r)
    sel = S["selection_counts"]; tot = {}
    for f, d in sel.items():
        for a, c in d.items():
            tot[a] = [u + v for u, v in zip(tot.get(a, [0, 0, 0, 0]), c)]
    out["alpha_selection_counts_outer"] = {a: dict(counts=c, frac_alpha_0_1=round(c[0] / sum(c), 3), frac_alpha_100=round(c[3] / sum(c), 3))
                                           for a, c in tot.items()}
    res[sc] = out
json.dump(res, open(RUN + "/posthoc/posthoc_saved_compare.json", "w"), indent=1)
for sc, o in res.items():
    print("==", sc, "r=%.4f" % o["r"], "teacher_pred_max_abs_diff=%.3e" % o["teacher_pred_max_abs_diff"])
    print("saved means:", o["saved_means"])
    for k, v in o["comparisons"].items():
        print("  %-46s gain=%+.4f%% wins=%d/5 p=%.4f CI=%s lofo_min=%+.4f per_fold=%s  (a=%.6f b=%.6f)" % (
            k, v["gain"], v["wins"], v["p"], v["ci"], v["lofo_min"], v["per_fold"], v["a_mean"], v["b_mean"]))
    print("  alpha boundary (frac at 0.1, frac at 100):", {a: (v["frac_alpha_0_1"], v["frac_alpha_100"]) for a, v in o["alpha_selection_counts_outer"].items()})

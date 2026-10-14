"""Independent metric replay and target/split check for the CPTAC baseline."""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def replay(y, pred):
    mask = np.isfinite(y) & np.isfinite(pred)
    n = mask.sum(axis=0)
    a = np.where(mask, y, 0.)
    b = np.where(mask, pred, 0.)
    am, bm = a.sum(axis=0) / np.maximum(n, 1), b.sum(axis=0) / np.maximum(n, 1)
    da, db = np.where(mask, y - am, 0.), np.where(mask, pred - bm, 0.)
    yrange = np.max(np.where(mask, y, -np.inf), axis=0) - np.min(np.where(mask, y, np.inf), axis=0)
    prange = np.max(np.where(mask, pred, -np.inf), axis=0) - np.min(np.where(mask, pred, np.inf), axis=0)
    good = (n >= 5) & (yrange > 1e-12) & (prange > 1e-12)
    rr = ((da * db).sum(axis=0)[good] /
          np.sqrt((da * da).sum(axis=0)[good] * (db * db).sum(axis=0)[good]))
    return {"mean_target_r": float(rr.mean()) if rr.size else None,
            "median_target_r": float(np.median(rr)) if rr.size else None,
            "valid_r_targets": int(good.sum()), "total_targets": y.shape[1],
            "observed_scored_values": int(mask.sum()),
            "standardized_rmse": float(np.sqrt(np.mean((y[mask] - pred[mask]) ** 2)))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists()
    summary = json.loads((args.run / "summary.json").read_text())
    bundles = []
    for c in summary["config"]["cohorts"]:
        with np.load(args.data / (c + ".npz"), allow_pickle=False) as z:
            bundles.append({k: z[k] for k in z.files if k.startswith(("rna__", "protein__")) or k in ("patient_ids", "sample_ids")})
    patients = np.concatenate([b["patient_ids"] for b in bundles])
    samples = np.concatenate([b["sample_ids"] for b in bundles])
    source = {}
    for mod in ("rna", "protein"):
        maps = []
        for b, cohort in zip(bundles, summary["config"]["cohorts"]):
            labels = [str(g).strip() for g in b[mod + "__feature_ids"]]
            if mod == "protein":
                decoded = []
                for g in labels:
                    parts = g.split("|")
                    valid = len(parts) == 2 and re.fullmatch(r"[A-Za-z0-9_.-]+", parts[0])
                    if valid and cohort.startswith("brca"):
                        valid = re.fullmatch(r"(?:NP|XP)_\d+(?:\.\d+)?", parts[1])
                    elif valid and cohort.startswith("luad"):
                        valid = parts[0] == parts[1]
                    else:
                        valid = False
                    decoded.append(parts[0] if valid else "")
                labels = decoded
            counts = Counter(labels)
            maps.append({g: i for i, g in enumerate(labels) if counts[g] == 1 and g.upper() not in ("", "NA", "NAN")})
        ids = sorted(set(maps[0]) & set(maps[1]))
        yy = np.concatenate([np.where(b[mod + "__observed"], b[mod + "__values"], np.nan)[:, [m[g] for g in ids]] for b, m in zip(bundles, maps)])
        source[mod] = (np.array(ids), yy)
    checked, verified, comparisons, strata = 0, [], [], []
    for r in summary["records"]:
        file = args.run / (r["direction"] + "__" + r["split"] + ".npz")
        with np.load(file, allow_pickle=False) as z:
            train, test = z["train_indices"], z["test_indices"]
            assert not set(patients[train]) & set(patients[test])
            np.testing.assert_array_equal(patients[test], z["patient_ids_test"])
            np.testing.assert_array_equal(samples[test], z["sample_ids_test"])
            ids, native = source[r["direction"].split("_to_")[1]]
            counts = np.isfinite(native[train]).sum(axis=0)
            mean = np.nansum(native[train], axis=0) / np.maximum(counts, 1)
            diff = np.where(np.isfinite(native[train]), native[train] - mean, 0.)
            sd = np.sqrt((diff ** 2).sum(axis=0) / np.maximum(counts, 1))
            panel = (counts >= max(5, .8 * len(train))) & (sd > 1e-8)
            np.testing.assert_array_equal(ids[panel], z["target_ids"])
            np.testing.assert_allclose(mean[panel], z["target_mean"], atol=1e-12)
            np.testing.assert_allclose(sd[panel], z["target_scale"], atol=1e-12)
            yy = (native[test][:, panel] - mean[panel]) / sd[panel]
            np.testing.assert_allclose(yy, z["truth"], atol=1e-12, equal_nan=True)
            predictions = {arm: z[arm] for arm in r["arms"]}
            for arm, expected in r["arms"].items():
                assert np.isfinite(predictions[arm]).all()
                actual = replay(yy, predictions[arm])
                for key, value in expected.items():
                    if value is None:
                        assert actual[key] is None
                    else:
                        assert np.isclose(value, actual[key], atol=1e-10, rtol=1e-10), (r["split"], arm, key)
                    checked += 1
            # Same-gene coverage is smaller than the full RNA panel. Compare r
            # on the identical jointly-defined target set, not unequal subsets.
            compared_arms = [a for a in ("same_gene", "masked_pca_ridge", "anchor_context", "anchor_context_pathway", "anchor_context_shuffled_pathway") if a in predictions]
            shared = []
            for j in range(yy.shape[1]):
                ok = np.isfinite(yy[:, j])
                if ok.sum() >= 5 and np.ptp(yy[ok, j]) > 1e-12 and all(np.ptp(predictions[a][ok, j]) > 1e-12 for a in compared_arms):
                    shared.append(j)
            comparisons.append({"direction": r["direction"], "split": r["split"],
                                "common_defined_targets": len(shared),
                                **{a: replay(yy[:, shared], predictions[a][:, shared]) for a in compared_arms}})
            if r["split"].startswith("mixed"):
                first_count = len(bundles[0]["patient_ids"])
                for cohort_index, cohort in enumerate(summary["config"]["cohorts"]):
                    subset = (test < first_count) if cohort_index == 0 else (test >= first_count)
                    strata.append({"direction": r["direction"], "split": r["split"], "cohort": cohort,
                                   "n_test": int(subset.sum()), "arms": {a: replay(yy[subset], predictions[a][subset]) for a in r["arms"]}})
            verified.append({"direction": r["direction"], "split": r["split"], "prediction_sha256": digest(file)})
    output = {"status": "passed", "summary_sha256": digest(args.run / "summary.json"),
              "replayed_scalar_fields": checked, "verified_splits": verified,
              "matched_target_comparisons": comparisons, "within_cohort_scores": strata,
              "boundary": "patient disjointness, exact raw-derived target panels/scales/truth and metrics verified; model fits not independently retrained"}
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps({k: v for k, v in output.items() if k not in ("verified_splits", "matched_target_comparisons", "within_cohort_scores")}))


if __name__ == "__main__":
    main()

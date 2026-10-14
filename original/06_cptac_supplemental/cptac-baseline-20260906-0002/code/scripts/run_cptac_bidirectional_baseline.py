"""BRCA/LUAD fixed first-pass baseline. Not a new blind test or FM claim."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from creed.science.masked_baseline import (
    unique_feature_map, source_gene_labels, target_panel, fit_pca, transform_pca, masked_ridge,
    predict_ridge, diagonal_predict, metrics,
)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(exist_ok=False)
    config = {"cohorts": ["brca_cptac_2020", "luad_cptac_2020"], "seed": 20260906,
              "rank": 16, "alpha": .1, "max_predictors": 1000,
              "target_training_completeness": .8,
              "preprocessing": "source log2 values unchanged; all fitted transformations training-only",
              "identity": "RNA symbols unchanged; BRCA protein SYMBOL|NP/XP accession and LUAD SYMBOL|SYMBOL projected to source gene label; ambiguous/duplicate gene labels entirely excluded",
              "selection": "fixed first-pass settings; no test-based model selection",
              "status": "development_baseline_not_FM_or_agent_efficacy",
              "summary_sha256": sha(args.data / "summary.json"),
              "code_sha256": sha(Path(__file__)),
              "module_sha256": sha(Path(__file__).resolve().parents[1] / "src/creed/science/masked_baseline.py")}
    (args.output / "intent.json").write_text(json.dumps(config, indent=2))
    summary = json.loads((args.data / "summary.json").read_text())
    expected = {c["cohort"]: c["bundle_sha256"] for c in summary["cohorts"]}
    bundles = []
    for c in config["cohorts"]:
        path = args.data / (c + ".npz")
        assert sha(path) == expected[c]
        with np.load(path, allow_pickle=False) as z:
            bundles.append({k: z[k] for k in z.files if k.startswith(("rna__", "protein__")) or k in ("sample_ids", "patient_ids")})
    arrays, ids, identity = {}, {}, {}
    for modality in ("rna", "protein"):
        labels = [source_gene_labels(b[modality + "__feature_ids"], c, modality) for b, c in zip(bundles, config["cohorts"])]
        maps = [unique_feature_map(g) for g in labels]
        common = sorted(set(maps[0]) & set(maps[1]))
        if not common:
            raise ValueError(f"no shared gene identifiers for {modality}")
        ids[modality] = np.array(common)
        identity[modality] = {"shared_unique_features": len(common),
                              "cohort_unique_counts": [len(m) for m in maps],
                              "unparsed_rows": [sum(not x for x in g) for g in labels],
                              "duplicate_or_empty_rows_excluded": [len(g) - len(m) for g, m in zip(labels, maps)]}
        arrays[modality] = np.concatenate([
            np.where(b[modality + "__observed"], b[modality + "__values"], np.nan)[:, [m[g] for g in common]]
            for b, m in zip(bundles, maps)])
    samples = np.concatenate([b["sample_ids"] for b in bundles])
    patients = np.concatenate([b["patient_ids"] for b in bundles])
    cohorts = np.concatenate([np.full(len(b["sample_ids"]), i) for i, b in enumerate(bundles)])
    assert len(set(patients)) == len(patients), "this baseline splitter requires one sample per patient"
    splitter = StratifiedKFold(5, shuffle=True, random_state=config["seed"])
    splits = [(f"mixed_fold_{i}", tr, te) for i, (tr, te) in enumerate(splitter.split(samples, cohorts))]
    splits += [("brca_to_luad", np.flatnonzero(cohorts == 0), np.flatnonzero(cohorts == 1)),
               ("luad_to_brca", np.flatnonzero(cohorts == 1), np.flatnonzero(cohorts == 0))]
    (args.output / "identity.json").write_text(json.dumps(identity, indent=2))
    results = []
    for source, target in (("rna", "protein"), ("protein", "rna")):
        x, y = arrays[source], arrays[target]
        for split_name, train, test in splits:
            assert not set(patients[train]) & set(patients[test])
            panel, means, scales = target_panel(y[train])
            yt = (y[train][:, panel] - means) / scales
            truth = (y[test][:, panel] - means) / scales
            ztrain, state = fit_pca(x[train], config["max_predictors"], config["rank"])
            ztest = transform_pca(x[test], state)
            coef = masked_ridge(ztrain, yt, config["alpha"])
            pred = predict_ridge(ztest, coef)
            diag, coverage = diagonal_predict(x[train], y[train][:, panel], x[test], ids[source], ids[target][panel], means)
            diag = (diag - means) / scales
            rng = np.random.default_rng(config["seed"] + len(results))
            permutation = rng.permutation(len(train))
            shuffled = predict_ridge(ztest, masked_ridge(ztrain, yt[permutation], config["alpha"]))
            context_mean = np.zeros_like(truth)
            for c in np.unique(cohorts[test]):
                rows = cohorts[train] == c
                if rows.any():
                    counts = np.isfinite(yt[rows]).sum(axis=0)
                    context_mean[cohorts[test] == c] = np.nansum(yt[rows], axis=0) / np.maximum(counts, 1)
            predictions = {"train_mean": np.zeros_like(truth), "cohort_mean": context_mean,
                           "same_gene": diag, "masked_pca_ridge": pred, "shuffled_target": shuffled}
            record = {"direction": source + "_to_" + target, "split": split_name,
                      "n_train": len(train), "n_test": len(test), "targets": len(panel),
                      "same_gene_modelled_targets": coverage,
                      "source_features": len(state[0]), "arms": {a: metrics(truth, v) for a, v in predictions.items()}}
            key = record["direction"] + "__" + split_name
            np.savez_compressed(args.output / (key + ".npz"), truth=truth,
                                train_indices=train, test_indices=test, patient_ids_test=patients[test],
                                sample_ids_test=samples[test], target_ids=ids[target][panel],
                                target_mean=means, target_scale=scales, **predictions)
            results.append(record)
            print(json.dumps(record), flush=True)
    final = {"config": config, "identity": identity, "n_patients": len(patients), "records": results,
             "caveat": "same two previously inspected cohorts; reciprocal cohort transfer is development, not prospective external validation"}
    (args.output / "summary.json").write_text(json.dumps(final, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import sys, traceback
        detail = {"error": str(exc), "type": type(exc).__name__, "traceback": traceback.format_exc()}
        print(json.dumps(detail), flush=True)
        if "--output" in sys.argv:
            out = Path(sys.argv[sys.argv.index("--output") + 1])
            if out.is_dir() and not (out / "failure.json").exists():
                (out / "failure.json").write_text(json.dumps(detail, indent=2))
        raise

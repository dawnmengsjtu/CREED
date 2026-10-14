"""Independently replay cross-layer sRMSE from deposited predictions (no fitting)."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from common import save_json, sha256, manifest, records_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    root = args.data_root.resolve()
    expected = {r["path"]: r["sha256"] for r in manifest() + records_manifest()}
    checked = set()

    def verified(path):
        relative = "runs/" + str(path.relative_to(root))
        if relative not in expected or sha256(path) != expected[relative]:
            raise RuntimeError("Unverified input: " + relative)
        checked.add(relative)
        return path

    results = {}
    for scenario in ("new_cell", "tissue"):
        run = root / "crosslayer-matrix-20261009-0001/output" / scenario
        with verified(run / "fold_metrics.csv").open() as f:
            rows = list(csv.DictReader(f))
        error, count = 0.0, 0
        for fold in range(5):
            with np.load(verified(run / ("fold-%s_predictions.npz" % fold)), allow_pickle=False) as z:
                truth, sd, use = z["truth"], z["train_sd"], z["evaluated"]
                for row in rows:
                    if int(row["fold"]) != fold:
                        continue
                    pred = z["pred__" + row["arm"]]
                    observed = np.isfinite(truth)
                    sq = np.where(observed, ((pred - truth) / sd) ** 2, 0)
                    mse = sq.sum(axis=0) / np.maximum(observed.sum(axis=0), 1)
                    score = float(np.sqrt(mse[use].mean()))
                    if not np.isfinite(score):
                        raise RuntimeError("Nonfinite score")
                    error = max(error, abs(score - float(row["score"])))
                    count += 1
        if error > 1e-10 or count != len(rows):
            raise RuntimeError("Metric replay failed: %s, %s" % (scenario, error))
        results[scenario] = {"fold_arm_scores": count, "max_absolute_score_error": error}
    result = {"passed": True, "kind": "real deposited predictions; no retraining",
              "verified_input_files": len(checked), "results": results}
    if args.report:
        save_json(args.report, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

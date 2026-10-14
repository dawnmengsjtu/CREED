"""Compare every prediction-archive array from a fresh cross-layer fit with the deposit."""
import argparse
import json
from pathlib import Path

import numpy as np
from common import manifest, save_json, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--results-root", type=Path, required=True,
                   help="Directory with new_cell/results and tissue/results from run.py")
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    expected = {row["path"]: row["sha256"] for row in manifest()}
    results = {}
    for scenario in ("new_cell", "tissue"):
        max_diff, count = 0.0, 0
        for fold in range(5):
            name = "fold-%s_predictions.npz" % fold
            relative = "crosslayer-matrix-20261009-0001/output/" + scenario + "/" + name
            frozen = args.data_root / relative
            if sha256(frozen) != expected["runs/" + relative]:
                raise RuntimeError("Original prediction archive hash mismatch: " + relative)
            fresh = args.results_root / scenario / "results" / name
            with np.load(frozen, allow_pickle=False) as a, np.load(fresh, allow_pickle=False) as b:
                if set(a.files) != set(b.files):
                    raise RuntimeError("Array names differ: " + name)
                for key in a.files:
                    x, y = a[key], b[key]
                    if x.shape != y.shape:
                        raise RuntimeError("Array shapes differ: " + key)
                    if x.dtype.kind in "fc":
                        mask = np.isfinite(x)
                        if not np.array_equal(mask, np.isfinite(y)):
                            raise RuntimeError("Finite masks differ: " + key)
                        if mask.any():
                            max_diff = max(max_diff, float(np.max(np.abs(x[mask] - y[mask]))))
                        if not np.array_equal(np.isnan(x), np.isnan(y)):
                            raise RuntimeError("NaN masks differ: " + key)
                    elif not np.array_equal(x, y):
                        raise RuntimeError("Non-floating array differs: " + key)
                    count += 1
        summary = json.loads((args.results_root / scenario / "results/summary.json").read_text())
        if max_diff > 1e-8 or summary["reproduction"]["passed"] is not True:
            raise RuntimeError("Fresh-fit comparison failed: %s / %g" % (scenario, max_diff))
        results[scenario] = {"prediction_archive_arrays_compared": count,
                             "maximum_absolute_difference": max_diff, "folds": 5,
                             "gates": summary["reproduction"]}
    result = {"passed": True, "kind": "fresh fitting of all cross-layer arms in both scenarios; comparison against all original prediction arrays",
              "results": results}
    if args.report:
        save_json(args.report, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Create source-native development bundles on A6000; never overwrite a run."""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from creed.science.cptac_blocks import COHORT_RNA, digest, load_cohort


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=False, exist_ok=False)
    intent = {"started_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "CPTAC source-native partial-modality development data",
              "cohorts": list(COHORT_RNA), "source": str(args.source),
              "python": platform.python_version(), "numpy": np.__version__,
              "code": {str(p): digest(p) for p in [Path(__file__),
                       Path(__file__).resolve().parents[1] / "src/creed/science/cptac_blocks.py"]}}
    (args.output / "intent.json").write_text(json.dumps(intent, indent=2), encoding="utf-8")
    profiles = []
    try:
        for cohort in COHORT_RNA:
            profile, bundle = load_cohort(args.source / cohort)
            target = args.output / (cohort + ".npz")
            with target.open("xb") as handle:
                np.savez_compressed(handle, **bundle)
            profile["bundle_sha256"] = digest(target)
            (args.output / (cohort + ".json")).write_text(json.dumps(profile, indent=2), encoding="utf-8")
            profiles.append(profile)
            print(json.dumps({"cohort": cohort, "samples": profile["sample_union"],
                              "patients": profile["patients_in_union"],
                              "all_modalities": profile["all_present_modality_intersection"]}), flush=True)
        summary = {"status": "prepared_development_only", "cohorts": profiles,
                   "finished_utc": datetime.now(timezone.utc).isoformat(),
                   "claim": "data readiness only; no predictive result",
                   "patient_grouping": "preserve raw patient IDs across cohorts; review repeated IDs before split"}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except Exception as exc:
        (args.output / "failure.json").write_text(json.dumps({"error_type": type(exc).__name__,
                      "error": str(exc), "completed_cohorts": [p["cohort"] for p in profiles]}, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()

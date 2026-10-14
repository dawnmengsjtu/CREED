"""Exercise the original cross-layer pipeline on synthetic inputs, not paper results."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

from common import ORIGINAL, runtime_env, save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    start = time.monotonic()
    cross = ORIGINAL / "03_development_runs_20261009/crosslayer-matrix-20261009-0001"
    results = {}
    with tempfile.TemporaryDirectory(prefix="creed-synthetic-") as tmp:
        root = Path(tmp)
        env = runtime_env(root / "runs")
        env["CREED_SKIP_SHA"] = "1"  # Only the generated synthetic fixture gets this bypass.
        env["PYTHONPATH"] = str(cross / "code")
        commands = [([str(cross / "dev/dev_fake_root.py"), str(root / "runs")], "fixture")]
        commands += [([str(cross / "code/crosslayer.py"), sc, "main", str(root / sc)], sc)
                     for sc in ("new_cell", "tissue")]
        for command, name in commands:
            result = subprocess.run([sys.executable] + command, env=env, text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            if name != "fixture":
                s = json.loads((root / name / "summary.json").read_text())
                if s["reproduction"]["passed"] is not True:
                    raise RuntimeError("Synthetic reproduction gate failed: " + name)
                results[name] = {"reproduction": s["reproduction"], "arms": len(s["summary"]), "folds": 5}
        shared = ORIGINAL / "03_development_runs_20261009/shared-crossdrug-20261009-0001/code/test_synthetic.py"
        result = subprocess.run([sys.executable, str(shared)], env=env, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        results["shared_crossdrug"] = result.stdout.strip()
    import numpy, scipy
    result = {"passed": True, "kind": "synthetic engineering check; not scientific replication",
              "python": platform.python_version(), "platform": platform.platform(),
              "numpy": numpy.__version__, "scipy": scipy.__version__, "results": results,
              "seconds": round(time.monotonic() - start, 2)}
    if args.report:
        save_json(args.report, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Rerun the prospective test's original analysis on the deposit and compare it with the deposited analysis files."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from common import ORIGINAL, adapt_tree, records_manifest, runtime_env, save_json, sha256

RUN = "prospective-lowdim-20261022-0001"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True, help="Verified deposit's runs/ directory, with run records")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    data_root, out = args.data_root.resolve(), args.out.resolve()
    if not (data_root / RUN).is_dir():
        p.error("--data-root must point to the unpacked runs/ directory")
    if out == data_root or data_root in out.parents:
        p.error("--out must be outside the frozen data root")
    if out.exists():
        p.error("--out must not already exist")
    out.mkdir(parents=True)
    code = adapt_tree(ORIGINAL / "09_prospective_test_20261022" / RUN / "code", out / "code", data_root)
    run = out / "run"
    run.mkdir()
    # Read-only views of the deposit; analyze.py writes only run/analysis/.
    for name in ("output", "data_gdsc1"):
        os.symlink(data_root / RUN / name, run / name)
    start = time.monotonic()
    with (out / "analyze.log").open("w") as log:
        result = subprocess.run([sys.executable, str(code / "analyze.py"), str(run)], env=runtime_env(data_root),
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        print((out / "analyze.log").read_text()[-8000:], file=sys.stderr)
        sys.exit(result.returncode)
    expected = {r["path"]: r["sha256"] for r in records_manifest() if r["path"].startswith("runs/%s/analysis/" % RUN)}
    files = {}
    for path, digest in sorted(expected.items()):
        replayed = run / "analysis" / Path(path).name
        got = sha256(replayed) if replayed.is_file() else None
        files[Path(path).name] = {"record_sha256": digest, "replayed_sha256": got, "identical": got == digest}
    report = {"passed": bool(files) and all(f["identical"] for f in files.values()),
              "kind": "original analyze.py of %s rerun on the deposit; outputs compared byte for byte with the "
                      "bundled analysis records" % RUN,
              "python": platform.python_version(), "seconds": round(time.monotonic() - start, 2), "files": files}
    if args.report:
        save_json(args.report, report)
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()

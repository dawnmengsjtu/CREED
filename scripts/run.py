"""Portable launchers for development analyses; output must be a fresh directory."""
import argparse
from pathlib import Path
import subprocess
import sys

from common import ORIGINAL, adapt_tree, runtime_env

D = "03_development_runs_20261009/"
N = "05_followup_N6_20261011/settle-stageA-20261011-0001/code/"
ENTRIES = {
    "crosslayer": (D + "crosslayer-matrix-20261009-0001/code/crosslayer.py", ("dryrun", "main")),
    "generator-curve": (D + "generator-curve-20261009-0001/code/gencurve.py", ("dryrun", "main")),
    "protein-panel": (D + "protein-panel-20261009-0001/code/protein_panel.py", ("dryrun", "repro", "main")),
    "rna-panel": (D + "rna-panel-20261009-0001/code/rna_panel.py", ("dryrun", "repro", "main")),
    "crispr": (D + "crispr-endpoint-20261009-0001/code/crispr_endpoint.py", ("dryrun", "main")),
    "acquisition-protein": (D + "acquisition-probe-20261009-0001/code/acquisition_probe.py", ("dryrun", "main")),
    "acquisition-rna": (D + "acquisition-probe-rna-20261009-0001/code/acquisition_probe_rna.py", ("dryrun", "main")),
    "context-panel": (D + "probe2-context-abstain-20261009-0001/code/probe2.py", ("dryrun", "fold")),
    "n6-drug": (N + "stageA.py", ("dryrun", "main")),
    "n6-crispr": (N + "stageA_crispr.py", ("dryrun", "main")),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("analysis", choices=ENTRIES)
    p.add_argument("--data-root", type=Path, required=True, help="Verified deposit's runs/ directory")
    p.add_argument("--scenario", choices=("new_cell", "tissue"), required=True)
    p.add_argument("--stage", default="dryrun")
    p.add_argument("--fold", type=int, choices=range(5))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    entry, stages = ENTRIES[args.analysis]
    if args.stage not in stages:
        p.error("Valid stages for %s: %s" % (args.analysis, ", ".join(stages)))
    needs_fold = args.stage == "fold" or (args.analysis.startswith("n6-") and args.stage == "main")
    if needs_fold != (args.fold is not None):
        p.error("--fold is required only for context-panel fold and N6 main stages")
    data_root, out = args.data_root.resolve(), args.out.resolve()
    if not data_root.is_dir():
        p.error("--data-root must point to the unpacked runs/ directory")
    # Protect frozen input/output assets even if the caller supplies an unsafe output path.
    if out == data_root or data_root in out.parents:
        p.error("--out must be outside the frozen data root")
    if out.exists():
        p.error("--out must not already exist")
    out.mkdir(parents=True)
    source = ORIGINAL / entry
    code = adapt_tree(source.parent, out / "code", data_root)
    cmd = [sys.executable, str(code / source.name), args.scenario, args.stage, str(out / "results")]
    if needs_fold:
        cmd.append(str(args.fold))
    env = runtime_env(data_root)
    with (out / "run.log").open("w") as log:
        result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    print("Log:", out / "run.log")
    print("Results:", out / "results")
    if result.returncode:
        print((out / "run.log").read_text()[-8000:], file=sys.stderr)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

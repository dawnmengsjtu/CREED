"""Regenerate available plot tables with original checks in a new work directory.

This is the historical V2 data stage, NOT a claim to reproduce all final PDFs.
See docs/FIGURES.md for the mapping to the final figures and remaining assets.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import shutil

from common import ORIGINAL, adapt_tree, runtime_env, save_json, sha256

STAGES = {
    "map": "prepare_fig2_v2.py",
    "generator": "prepare_figS1_v2.py",
    "drug-detail": "prepare_fig3_v2.py",
    "graph-controls": "prepare_fig4_v2.py",
    "panels": "prepare_fig5_v2.py",
    "panel-detail": "prepare_fig6_v2.py",
    "crispr": "prepare_fig7_v2.py",
    "sealed-summary": "prepare_fig8_v2.py",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    args = p.parse_args()
    data_root, out = args.data_root.resolve(), args.out.resolve()
    if out == data_root or data_root in out.parents:
        p.error("--out must be outside the frozen data root")
    out.mkdir(parents=True, exist_ok=False)
    source = ORIGINAL / "07_figures/data_preparation/20261009_figures_v3_cell/r/figures2_6"
    code = adapt_tree(source, out / "code", data_root)
    # These are byte-identical assets in the deposit; reconstruct the historical
    # figure input layout without modifying data_root or inventing derived values.
    bindings = {
        "data/raw/figure2_6_extra/drug_annotations.json": "gdsc421-20260906-0001/result/drug_annotations.json",
        "data/raw/gdsc_metabolite_full_protein/source-inputs.npz":
            "gdsc-metabolite-full-protein-20260908-0001/output/result/source-inputs.npz",
    }
    for relative, source_relative in bindings.items():
        target = out / "staging" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(data_root / source_relative, target)
    save_json(out / "INPUT_BINDINGS.json", [
        {"figure_path": key, "deposit_path": val, "sha256": sha256(data_root / val)}
        for key, val in bindings.items()])
    old = "/public/home/mengxl/workspaces/creed-research/figure_work/20261009_figures2_6_v2"
    replacement = str(out / "staging").replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
    replacements = []
    for path in code.glob("*.py"):
        before = sha256(path)
        text = path.read_text()
        path.write_text(text.replace(old, replacement))
        replacements.append({"file": path.name, "before": before, "after": sha256(path),
                             "figure_root_replacements": text.count(old)})
    save_json(code / "FIGURE_ROOT_ADAPTATIONS.json", replacements)
    results = {}
    # Execute selected stages in dependency order, regardless of CLI order.
    for stage, filename in STAGES.items():
        if stage not in args.stages:
            continue
        command = [sys.executable, str(code / filename)]
        if stage == "panels":
            command.append("--gene-panel-ready")
        with (out / (stage + ".log")).open("w") as log:
            result = subprocess.run(command, env=runtime_env(data_root), stdout=log, stderr=subprocess.STDOUT)
        results[stage] = {"passed": result.returncode == 0, "log": stage + ".log"}
        if result.returncode:
            results[stage]["error_tail"] = (out / (stage + ".log")).read_text()[-1800:]
        print(stage, "PASS" if not result.returncode else "FAILED", flush=True)
    files = sorted((out / "staging").rglob("*.csv"))
    report = {"kind": "historical plot-table data stages; final R rendering not run",
              "passed": all(r["passed"] for r in results.values()), "stages": results,
              "csv_files": len(files)}
    save_json(out / "report.json", report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

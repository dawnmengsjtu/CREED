"""Verify original code and supplements without installing scientific dependencies."""
import argparse
import ast
import csv
import json
from pathlib import Path
import zipfile

from common import ROOT, ORIGINAL, checksum_list, records_manifest, save_json, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    release_manifest = ROOT / "RELEASE_SHA256SUMS"
    if release_manifest.exists():
        for name, digest in checksum_list(release_manifest).items():
            if sha256(ROOT / name) != digest:
                raise RuntimeError("Release file changed: " + name)
    with (ORIGINAL / "SOURCES.tsv").open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for row in rows:
        path = ORIGINAL / row["archive_path"]
        if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            raise RuntimeError("Original source changed: " + row["archive_path"])
    python_files = list(ORIGINAL.rglob("*.py")) + list((ROOT / "scripts").glob("*.py"))
    for path in python_files:
        ast.parse(path.read_text(), filename=str(path))
    for row in records_manifest():
        path = ROOT / "metadata/records" / Path(row["path"]).relative_to("runs")
        if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            raise RuntimeError("Run record changed: " + row["path"])
    supplements = checksum_list(ROOT / "metadata/SUPPLEMENT_SHA256SUMS")
    for name, digest in supplements.items():
        path = ROOT / name
        if sha256(path) != digest:
            raise RuntimeError("Supplement changed: " + name)
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
            if bad:
                raise RuntimeError("ZIP CRC failed: " + bad)
    result = {"passed": True, "original_payload_files": len(rows),
              "python_files_parsed": len(python_files), "supplement_archives": len(supplements),
              "run_records": len(records_manifest())}
    if args.report:
        save_json(args.report, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

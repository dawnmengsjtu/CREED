"""Shared release helpers; original scientific code is in original/."""
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "original"
SERVER_ROOT = "/public/home/mengxl/creed-research-runs"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def manifest():
    with (ROOT / "metadata/MANIFEST.tsv").open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def records_manifest():
    with (ROOT / "metadata/RECORDS.tsv").open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def checksum_list(path):
    return {line.split(maxsplit=1)[1].strip(): line.split()[0]
            for line in Path(path).read_text().splitlines() if line.strip()}


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def runtime_env(data_root=None):
    env = os.environ.copy()
    # Real-data launchers never inherit the synthetic hash bypass.
    env.pop("CREED_SKIP_SHA", None)
    env.pop("PYTHONOPTIMIZE", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    if data_root is not None:
        env["CREED_ROOT"] = str(Path(data_root).resolve())
    return env


def adapt_tree(source, target, data_root):
    """Copy a script tree and replace only the original run-root path.

    Record every file's before/after hashes. Do not touch the source snapshot.
    Container /oldcode and historical figure roots are NOT rewritten here.
    """
    source, target, data_root = Path(source), Path(target), Path(data_root).resolve()
    target.mkdir(parents=True, exist_ok=False)
    replacement = str(data_root).replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
    replacement = replacement.replace("\n", "\\n").replace("\r", "\\r")
    rows = []
    for src in sorted(source.rglob("*")):
        if not src.is_file() or "__pycache__" in src.parts:
            continue
        dst = target / src.relative_to(source)
        dst.parent.mkdir(parents=True, exist_ok=True)
        data = src.read_bytes()
        # Only Python is executed by the portable launcher. Preserve everything else.
        count = data.count(SERVER_ROOT.encode()) if src.suffix == ".py" else 0
        dst.write_bytes(data.replace(SERVER_ROOT.encode(), replacement.encode()) if count else data)
        rows.append({"file": str(src.relative_to(source)), "source_sha256": sha256(src),
                     "adapted_sha256": sha256(dst), "path_replacements": count})
    save_json(target / "ADAPTATIONS.json", {"data_root": str(data_root), "files": rows})
    return target

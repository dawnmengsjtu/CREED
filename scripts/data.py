"""Download, safely unpack, and verify the 27 registered data archives."""
import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.error
import urllib.request

from common import ROOT, checksum_list, manifest, records_manifest, save_json, sha256


def download(destination):
    api = "https://zenodo.org/api/records/23299706"
    try:
        with urllib.request.urlopen(api, timeout=30) as response:
            record = json.load(response)
    except urllib.error.HTTPError as e:
        raise SystemExit("Zenodo record is not publicly accessible (HTTP %s). "
                         "The DOI is reserved; obtain the archive directory from the authors." % e.code)
    available = {entry["key"]: entry for entry in record["files"]}
    expected = checksum_list(ROOT / "metadata/TARS_SHA256SUMS")
    destination.mkdir(parents=True, exist_ok=True)
    for name, digest in expected.items():
        path = destination / name
        if path.is_file() and sha256(path) == digest:
            continue
        if name not in available:
            raise SystemExit("Missing archive on public record: " + name)
        partial = path.with_suffix(".tar.partial")
        link = available[name]["links"]["self"]
        with urllib.request.urlopen(link, timeout=120) as src, partial.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        if sha256(partial) != digest:
            partial.unlink()
            raise SystemExit("Archive hash mismatch: " + name)
        partial.replace(path)
        print("downloaded", name, flush=True)


def verify(directory):
    rows = manifest() + records_manifest()
    failures = []
    for row in rows:
        p = directory / row["path"]
        if not p.is_file() or p.stat().st_size != int(row["bytes"]) or sha256(p) != row["sha256"]:
            failures.append(row["path"])
    result = {"files_checked": len(rows), "bytes": sum(int(r["bytes"]) for r in rows),
              "passed": not failures, "failures": failures}
    if failures:
        raise RuntimeError(json.dumps(result))
    return result


def install_records(destination):
    rows = records_manifest()
    for row in rows:
        relative = PurePosixPath(row["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "runs":
            raise RuntimeError("Invalid record path: " + row["path"])
        source = ROOT / "metadata/records" / Path(*relative.parts[1:])
        if sha256(source) != row["sha256"]:
            raise RuntimeError("Release record hash mismatch: " + row["path"])
        target = destination / row["path"]
        if target.exists():
            if sha256(target) != row["sha256"]:
                raise RuntimeError("Refusing to replace a different record: " + str(target))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    return len(rows)


def unpack(archives, destination):
    destination.mkdir(parents=True, exist_ok=False)
    expected_files = {r["path"]: r for r in manifest()}
    seen = set()
    for name, digest in checksum_list(ROOT / "metadata/TARS_SHA256SUMS").items():
        archive = archives / name
        if not archive.is_file() or sha256(archive) != digest:
            raise RuntimeError("Missing archive or hash mismatch: " + str(archive))
        with tarfile.open(archive) as tf:
            for member in tf:
                p = PurePosixPath(member.name)
                if p.is_absolute() or ".." in p.parts:
                    raise RuntimeError("Unsafe archive path: " + member.name)
                if member.isdir():
                    continue
                key = str(p)
                if not member.isfile() or key not in expected_files or key in seen:
                    raise RuntimeError("Unregistered or duplicate archive member: " + key)
                if member.size != int(expected_files[key]["bytes"]):
                    raise RuntimeError("Size mismatch: " + key)
                target = destination / key
                target.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(member) as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
                if sha256(target) != expected_files[key]["sha256"]:
                    raise RuntimeError("Data hash mismatch: " + key)
                seen.add(key)
        print("verified and unpacked", name, flush=True)
    if seen != set(expected_files):
        raise RuntimeError("Missing data files: " + repr(set(expected_files) - seen))
    records = install_records(destination)
    return {"archives_checked": 27, "files_checked": len(seen), "release_records_checked": records, "passed": True,
            "bytes": sum(int(r["bytes"]) for r in expected_files.values())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("download"); a.add_argument("--dest", type=Path, default=ROOT / "downloads")
    a = sub.add_parser("unpack"); a.add_argument("--archives", type=Path, required=True)
    a.add_argument("--dest", type=Path, required=True); a.add_argument("--report", type=Path)
    a = sub.add_parser("verify"); a.add_argument("--data-dir", type=Path, required=True)
    a.add_argument("--report", type=Path)
    a = sub.add_parser("records"); a.add_argument("--data-dir", type=Path, required=True)
    args = p.parse_args()
    if args.command == "download":
        download(args.dest); return
    if args.command == "records":
        print("Installed/verified records:", install_records(args.data_dir)); return
    result = unpack(args.archives, args.dest) if args.command == "unpack" else verify(args.data_dir)
    if args.report:
        save_json(args.report, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

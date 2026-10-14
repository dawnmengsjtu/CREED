#!/usr/bin/env python3
"""Create-once custody materialisation of frozen CCLE development values."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping

repository = Path(__file__).resolve().parents[1]
source_root = repository / "src"
if not source_root.is_dir():  # pragma: no cover - packaging/runtime failure
    raise RuntimeError("materializer must run from a complete CREED source snapshot")
sys.path.insert(0, str(source_root))

from creed.science.ccle import canonical_json_sha256, sha256_file  # noqa: E402
from creed.science.ccle_values import (  # noqa: E402
    EXPECTED_REAL_FEATURE_COUNTS,
    FROZEN_REAL_SPLIT_PAYLOAD_SHA256,
    PROTOCOL_ID,
    SEALED_ARTIFACT_ROLE,
    SEALED_MANIFEST_SCHEMA,
    VISIBLE_ARTIFACT_ROLE,
    VISIBLE_MANIFEST_SCHEMA,
    CCLEValueCustodyError,
    array_manifest,
    partition_arrays,
    read_frozen_ccle_values,
    validate_frozen_split_and_sources,
    write_npz_create_once,
)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json_create_once(path: Path, payload: object) -> None:
    serialised = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialised)
        handle.flush()
        os.fsync(handle.fileno())


def _canonical_manifest(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["integrity"] = {"canonical_payload_sha256": canonical_json_sha256(payload)}
    return result


def _git_commit(path: Path) -> str | None:
    try:
        completed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={path.resolve()}",
                "-C",
                str(path),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit if len(commit) == 40 else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the frozen CCLE split and source hashes, then physically isolate "
            "development-visible values from the sealed internal-final partition."
        )
    )
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--metabolomics", type=Path, required=True)
    parser.add_argument("--rna", type=Path, required=True)
    parser.add_argument("--mmc2", type=Path, required=True)
    parser.add_argument("--mmc3", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protein-sheet", default="Prot matrix excl single-peptide")
    parser.add_argument(
        "--expected-l0-features",
        type=int,
        default=EXPECTED_REAL_FEATURE_COUNTS["L0_metabolomics"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--expected-l1-features",
        type=int,
        default=EXPECTED_REAL_FEATURE_COUNTS["L1_rna_expected_counts"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--expected-l2-features",
        type=int,
        default=EXPECTED_REAL_FEATURE_COUNTS["L2_protein_mmc3"],
        help=argparse.SUPPRESS,
    )
    return parser


def _seal_tree(root: Path) -> None:
    if not root.exists():
        return
    files = sorted((path for path in root.rglob("*") if path.is_file()), reverse=True)
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for path in files:
        path.chmod(0o444)
    for path in directories:
        path.chmod(0o555)
    root.chmod(0o555)


def _safe_failure(exc: BaseException) -> str:
    if isinstance(exc, CCLEValueCustodyError):
        return str(exc)
    return "unexpected custody failure; no assay value is included in this receipt"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root: Path = args.output_root
    try:
        output_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    except FileExistsError:
        print(f"STOP_CREATE_ONCE output root exists: {output_root}", file=sys.stderr)
        return 73
    visible_root = output_root / "visible"
    sealed_root = output_root / "sealed_internal_final"
    visible_root.mkdir(mode=0o700)
    sealed_root.mkdir(mode=0o700)

    values_module = repository / "src" / "creed" / "science" / "ccle_values.py"
    code = {
        "materializer_sha256": sha256_file(Path(__file__)),
        "ccle_values_sha256": sha256_file(values_module),
        "git_commit": _git_commit(repository),
    }
    source_paths = {
        "L0_metabolomics": args.metabolomics,
        "L1_rna_expected_counts": args.rna,
        "identity_mmc2": args.mmc2,
        "L2_protein_mmc3": args.mmc3,
    }
    intent = {
        "schema": "creed.ccle-value-materialization-intent.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "intent",
        "started_at_utc": _utc_now(),
        "split": str(args.split),
        "source_paths": {key: str(value) for key, value in source_paths.items()},
        "output_root": str(output_root),
        "code": code,
        "custody": {
            "source_hashes_verified_before_value_access": True,
            "visible_partitions": ["train", "validation"],
            "internal_final_role": SEALED_ARTIFACT_ROLE,
            "model_invocations_allowed": 0,
            "preprocessing_allowed": False,
            "imputation_allowed": False,
            "automatic_retry": False,
        },
    }
    _write_json_create_once(output_root / "intent.json", intent)
    intent_sha256 = sha256_file(output_root / "intent.json")

    try:
        contract = validate_frozen_split_and_sources(
            args.split, source_paths=source_paths
        )
        expected_counts = {
            "L0_metabolomics": args.expected_l0_features,
            "L1_rna_expected_counts": args.expected_l1_features,
            "L2_protein_mmc3": args.expected_l2_features,
        }
        if contract.split_payload_sha256 == FROZEN_REAL_SPLIT_PAYLOAD_SHA256:
            if expected_counts != dict(EXPECTED_REAL_FEATURE_COUNTS):
                raise CCLEValueCustodyError(
                    "real frozen split requires exact 225/37279/6692 feature counts"
                )
            if args.protein_sheet != "Prot matrix excl single-peptide":
                raise CCLEValueCustodyError(
                    "real frozen split requires the frozen mmc3 protein sheet"
                )
        values = read_frozen_ccle_values(
            contract,
            expected_feature_counts=expected_counts,
            protein_sheet=args.protein_sheet,
        )
        for role, source in contract.source_paths.items():
            if (
                source.stat().st_size != contract.source_bytes[role]
                or sha256_file(source) != contract.source_sha256[role]
            ):
                raise CCLEValueCustodyError(
                    f"source changed during custody read for role {role}"
                )

        visible_ids = contract.visible_sample_ids
        visible_labels = ("train",) * len(contract.train_sample_ids) + (
            "validation",
        ) * len(contract.validation_sample_ids)
        visible_arrays = partition_arrays(values, visible_ids, visible_labels)
        visible_npz = visible_root / "ccle_l0_l1_l2_visible.npz"
        write_npz_create_once(visible_npz, visible_arrays)
        visible_structure = array_manifest(visible_arrays)
        visible_payload = {
            "schema": VISIBLE_MANIFEST_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "artifact_role": VISIBLE_ARTIFACT_ROLE,
            "status": "complete",
            "internal_final_released": False,
            "split": {
                "file_sha256": contract.split_file_sha256,
                "payload_sha256": contract.split_payload_sha256,
                "visible_order": "frozen train sample_ids then frozen validation sample_ids",
                "partition_counts": {
                    "train": len(contract.train_sample_ids),
                    "validation": len(contract.validation_sample_ids),
                },
                "internal_final_count": len(contract.internal_final_sample_ids),
                "internal_final_sample_ids_sha256": hashlib.sha256(
                    "".join(
                        f"{value}\n"
                        for value in sorted(contract.internal_final_sample_ids)
                    ).encode("utf-8")
                ).hexdigest(),
            },
            "sources": {
                role: {
                    "sha256": contract.source_sha256[role],
                    "bytes": contract.source_bytes[role],
                }
                for role in sorted(contract.source_paths)
            },
            "code": code,
            "artifact": {
                "filename": visible_npz.name,
                "sha256": sha256_file(visible_npz),
                "bytes": visible_npz.stat().st_size,
            },
            "structure": visible_structure,
            "scientific_claim_authority": False,
        }
        visible_manifest = _canonical_manifest(visible_payload)
        visible_manifest_path = visible_root / "manifest.json"
        _write_json_create_once(visible_manifest_path, visible_manifest)

        final_ids = contract.internal_final_sample_ids
        sealed_arrays = partition_arrays(
            values, final_ids, ("internal_final",) * len(final_ids)
        )
        sealed_npz = sealed_root / "ccle_l0_l1_l2_internal_final.npz"
        write_npz_create_once(sealed_npz, sealed_arrays)
        sealed_structure = array_manifest(sealed_arrays)
        sealed_payload = {
            "schema": SEALED_MANIFEST_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "artifact_role": SEALED_ARTIFACT_ROLE,
            "status": "sealed_unreleased",
            "internal_final_released": False,
            "split": {
                "file_sha256": contract.split_file_sha256,
                "payload_sha256": contract.split_payload_sha256,
                "partition_count": len(final_ids),
                "sample_ids_sha256": sealed_structure["sample_ids_sha256"],
            },
            "sources": {
                role: {
                    "sha256": contract.source_sha256[role],
                    "bytes": contract.source_bytes[role],
                }
                for role in sorted(contract.source_paths)
            },
            "code": code,
            "artifact": {
                "filename": sealed_npz.name,
                "sha256": sha256_file(sealed_npz),
                "bytes": sealed_npz.stat().st_size,
            },
            "structure": sealed_structure,
            "scientific_loader_access": "forbidden_fail_closed",
            "scientific_claim_authority": False,
        }
        sealed_manifest = _canonical_manifest(sealed_payload)
        sealed_manifest_path = sealed_root / "manifest.json"
        _write_json_create_once(sealed_manifest_path, sealed_manifest)

        receipt = {
            "schema": "creed.ccle-value-materialization-receipt.v1",
            "protocol_id": PROTOCOL_ID,
            "status": "complete",
            "completed_at_utc": _utc_now(),
            "source_hashes_verified_before_value_access": True,
            "model_invocations": 0,
            "automatic_retries": 0,
            "preprocessing_performed": False,
            "missing_values_filled": False,
            "visible": {
                "sample_count": visible_structure["sample_count"],
                "layer_shapes": {
                    role: layer["shape"]
                    for role, layer in visible_structure["layers"].items()
                },
                "artifact_sha256": visible_payload["artifact"]["sha256"],
                "manifest_sha256": sha256_file(visible_manifest_path),
            },
            "sealed_internal_final": {
                "sample_count": sealed_structure["sample_count"],
                "layer_shapes": {
                    role: layer["shape"]
                    for role, layer in sealed_structure["layers"].items()
                },
                "artifact_sha256": sealed_payload["artifact"]["sha256"],
                "manifest_sha256": sha256_file(sealed_manifest_path),
                "values_logged_or_reported": False,
                "released": False,
            },
            "code": code,
            "intent_sha256": intent_sha256,
            "scientific_claim_authority": False,
        }
        receipt_path = output_root / "receipt.json"
        _write_json_create_once(receipt_path, receipt)
        receipt_sha256 = sha256_file(receipt_path)
        _seal_tree(output_root)
    except BaseException as exc:
        failure = {
            "schema": "creed.ccle-value-materialization-receipt.v1",
            "protocol_id": PROTOCOL_ID,
            "status": "failed",
            "completed_at_utc": _utc_now(),
            "error_type": type(exc).__name__,
            "error": _safe_failure(exc),
            "model_invocations": 0,
            "automatic_retries": 0,
            "internal_final_values_logged_or_reported": False,
            "internal_final_released": False,
            "scientific_claim_authority": False,
        }
        try:
            _write_json_create_once(output_root / "receipt.json", failure)
        finally:
            _seal_tree(output_root)
        print(
            "CCLE custody materialization failed without value disclosure",
            file=sys.stderr,
        )
        return 2 if isinstance(exc, CCLEValueCustodyError) else 1

    public_summary = {
        "status": "complete",
        "output_root": str(output_root),
        "visible_sample_count": visible_structure["sample_count"],
        "visible_layer_shapes": {
            role: layer["shape"] for role, layer in visible_structure["layers"].items()
        },
        "visible_artifact_sha256": visible_payload["artifact"]["sha256"],
        "sealed_internal_final_sample_count": sealed_structure["sample_count"],
        "sealed_internal_final_layer_shapes": {
            role: layer["shape"] for role, layer in sealed_structure["layers"].items()
        },
        "sealed_internal_final_artifact_sha256": sealed_payload["artifact"]["sha256"],
        "internal_final_values_logged_or_reported": False,
        "internal_final_released": False,
        "scientific_claim_authority": False,
        "receipt_sha256": receipt_sha256,
    }
    print(json.dumps(public_summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

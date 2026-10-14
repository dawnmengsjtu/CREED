#!/usr/bin/env python3
"""Create-once metadata-only freeze of the CREED CCLE triple-layer split."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

try:
    from creed.science.ccle import (
        CCLEContractError,
        build_ccle_split_manifest,
        load_ccle_triple_inventory,
    )
except ModuleNotFoundError as exc:
    # The metadata-only freezer can run in a custody runtime that deliberately
    # omits numerical science packages imported by ``creed.science.__init__``.
    # Load this one stdlib-only contract module without weakening its checks.
    if exc.name is None or not (
        exc.name == "numpy" or exc.name == "creed" or exc.name.startswith("creed.")
    ):
        raise
    module_path = (
        Path(__file__).resolve().parents[1] / "src" / "creed" / "science" / "ccle.py"
    )
    spec = importlib.util.spec_from_file_location("creed_ccle_contract", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load CCLE contract module: {module_path}") from exc
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    CCLEContractError = module.CCLEContractError
    build_ccle_split_manifest = module.build_ccle_split_manifest
    load_ccle_triple_inventory = module.load_ccle_triple_inventory


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json_create_once(path: Path, payload: object) -> None:
    serialised = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, indent=2
    ) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialised)
        handle.flush()
        os.fsync(handle.fileno())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze CCLE L0-L1-L2 sample identities without reading any assay values."
        )
    )
    parser.add_argument("--metabolomics", type=Path, required=True)
    parser.add_argument("--rna", type=Path, required=True)
    parser.add_argument("--mmc2", type=Path, required=True)
    parser.add_argument("--mmc3", type=Path, required=True)
    parser.add_argument(
        "--protein-sheet",
        default="Prot matrix excl single-peptide",
        help="mmc3 sheet whose header/sample IDs define L2 (values remain unread)",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--frozen-on", default="2026-08-30")
    parser.add_argument("--expected-triple-count", type=int, default=608)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root: Path = args.output_root
    try:
        output_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    except FileExistsError:
        print(f"STOP_CREATE_ONCE output root exists: {output_root}", file=sys.stderr)
        return 73

    intent = {
        "schema": "creed.ccle-l0-l1-l2-split-intent.v1",
        "protocol_id": "CREED_CCLE_L0_L1_L2_SPLIT_V1",
        "status": "intent",
        "started_at_utc": _utc_now(),
        "output_root": str(output_root),
        "sources": {
            "L0_metabolomics": str(args.metabolomics),
            "L1_rna_expected_counts": str(args.rna),
            "identity_mmc2": str(args.mmc2),
            "L2_protein_mmc3": str(args.mmc3),
        },
        "protein_sheet": args.protein_sheet,
        "expected_triple_count": args.expected_triple_count,
        "seed": args.seed,
        "allowed_reads": [
            "full source bytes for SHA-256 only",
            "schemas and headers",
            "sample identifiers",
            "mmc2 non-expression identity and tissue/lineage metadata",
        ],
        "forbidden_reads": [
            "metabolite abundance values",
            "RNA expected-count values",
            "protein abundance values",
        ],
        "model_invocations_allowed": 0,
        "automatic_retry": False,
    }
    _write_json_create_once(output_root / "intent.json", intent)

    try:
        inventory = load_ccle_triple_inventory(
            metabolomics_path=args.metabolomics,
            rna_path=args.rna,
            mmc2_path=args.mmc2,
            mmc3_path=args.mmc3,
            protein_sheet=args.protein_sheet,
            expected_triple_count=args.expected_triple_count,
            minimum_triple_count=args.expected_triple_count,
        )
        manifest = build_ccle_split_manifest(
            inventory, seed=args.seed, frozen_on=args.frozen_on
        )
        _write_json_create_once(output_root / "split.json", manifest)
        split_file_bytes = (output_root / "split.json").read_bytes()
        receipt = {
            "schema": "creed.ccle-l0-l1-l2-split-receipt.v1",
            "protocol_id": "CREED_CCLE_L0_L1_L2_SPLIT_V1",
            "status": "complete",
            "completed_at_utc": _utc_now(),
            "assay_values_read": False,
            "model_invocations": 0,
            "automatic_retries": 0,
            "exact_triple_samples": len(inventory.triple_sidms),
            "manifest_payload_sha256": manifest["integrity"][
                "canonical_payload_sha256"
            ],
            "split_file_sha256": hashlib.sha256(split_file_bytes).hexdigest(),
            "source_sha256": {
                role: source.sha256
                for role, source in sorted(inventory.source_fingerprints.items())
            },
            "internal_final_released": False,
            "scientific_claim_authority": False,
        }
        _write_json_create_once(output_root / "receipt.json", receipt)
        for path in output_root.iterdir():
            path.chmod(0o444)
        output_root.chmod(0o555)
    except Exception as exc:
        failure = {
            "schema": "creed.ccle-l0-l1-l2-split-receipt.v1",
            "protocol_id": "CREED_CCLE_L0_L1_L2_SPLIT_V1",
            "status": "failed",
            "completed_at_utc": _utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "assay_values_read": False,
            "model_invocations": 0,
            "automatic_retries": 0,
            "internal_final_released": False,
            "scientific_claim_authority": False,
        }
        try:
            _write_json_create_once(output_root / "receipt.json", failure)
        finally:
            for path in output_root.iterdir():
                path.chmod(0o444)
            output_root.chmod(0o555)
        print(f"CCLE split freeze failed: {exc}", file=sys.stderr)
        return 2 if isinstance(exc, CCLEContractError) else 1

    print(
        json.dumps(
            {
                "status": "complete",
                "output_root": str(output_root),
                "triple_samples": len(inventory.triple_sidms),
                "partition_counts": {
                    name: payload["count"]
                    for name, payload in manifest["partitions"].items()
                },
                "manifest_payload_sha256": manifest["integrity"][
                    "canonical_payload_sha256"
                ],
                "internal_final_released": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Create or independently verify the frozen visible CCLE lineage registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from creed.science import ccle as identity_module
from creed.science import ccle_visible_lineage_registry as registry_module
from creed.science.ccle_visible_lineage_registry import (
    CCLEVisibleLineageRegistryError,
    produce_visible_lineage_registry,
    verify_visible_lineage_registry,
)


def _add_sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mmc2", type=Path, required=True)
    parser.add_argument("--acquisition-receipt", type=Path, required=True)
    parser.add_argument("--visible-artifact", type=Path, required=True)
    parser.add_argument("--visible-manifest", type=Path, required=True)
    parser.add_argument("--visible-receipt", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create or independently verify an identity/lineage-only registry "
            "from frozen ProCan mmc2 metadata and the two visible NPZ identity "
            "members. Expression members and internal-final custody are forbidden."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    _add_sources(create)
    create.add_argument("--output-root", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--registry-root", type=Path, required=True)
    _add_sources(verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    code_paths = {
        "registry_module": Path(registry_module.__file__).resolve(),
        "registry_cli": Path(__file__).resolve(),
        "identity_module": Path(identity_module.__file__).resolve(),
        "registry_protocol": (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "CCLE_VISIBLE_LINEAGE_REGISTRY_V1.json"
        ),
    }
    common = {
        "mmc2": args.mmc2,
        "acquisition_receipt": args.acquisition_receipt,
        "visible_artifact": args.visible_artifact,
        "visible_manifest": args.visible_manifest,
        "visible_receipt": args.visible_receipt,
        "code_paths": code_paths,
    }
    try:
        if args.command == "create":
            result = produce_visible_lineage_registry(
                output_root=args.output_root, **common
            )
        else:
            result = verify_visible_lineage_registry(
                registry_root=args.registry_root, **common
            )
    except (OSError, CCLEVisibleLineageRegistryError) as exc:
        print(f"STOP_CCLE_VISIBLE_LINEAGE_REGISTRY {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

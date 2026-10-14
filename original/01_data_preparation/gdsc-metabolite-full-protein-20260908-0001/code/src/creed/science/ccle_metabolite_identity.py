"""Fail-closed contract for the frozen CCLE L0 metabolite identities.

The contract distinguishes an authoritative single-compound resolution from
mixed chromatographic peaks and sum-composition lipid features.  Only an exact
name or an authority-returned exact synonym may carry a unique structure.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Mapping

from creed.science.metabolite_registry import normalized_lexical_name


IDENTITY_SCHEMA_VERSION = 2
FROZEN_FEATURE_MANIFEST_SHA256 = (
    "bd8f624b4a4cd48c74671eb4b0102a3d6374a648d9148b2b55e15f5857d30594"
)
FROZEN_DATASET_SOURCE_SHA256 = (
    "7c1d24aa575f4c58a29019026b5df8e6d1142a56925aba32ff3f1d1d5a7fd0ac"
)
FROZEN_IDENTITY_REGISTRY_SHA256 = (
    "1dc713efe4c1b0264dcd11a0466b77a012c3690ff3b98f7910879e491fa2f959"
)
LIPID_TOTAL_COMPOSITION = re.compile(
    r"^C(?P<carbons>\d+):(?P<double_bonds>\d+) "
    r"(?P<class>LPC|LPE|PC(?:-[AB])?|SM|DAG|CE|TAG)$"
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
CHEBI_PATTERN = re.compile(r"^CHEBI:\d+$")


class CCLEIdentityError(ValueError):
    """Raised when a CCLE identity artifact violates the V2 boundary."""


class FeatureIdentityClass(StrEnum):
    EXACT = "EXACT"
    AUTHORITY_ALIAS = "AUTHORITY_ALIAS"
    COMPOSITE = "COMPOSITE"
    AGGREGATE = "AGGREGATE"
    UNKNOWN = "UNKNOWN"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lexical_feature_class(raw_name: str) -> FeatureIdentityClass | None:
    """Classify only explicit non-singleton notation; never infer a molecule."""

    if "/" in raw_name:
        return FeatureIdentityClass.COMPOSITE
    if LIPID_TOTAL_COMPOSITION.fullmatch(raw_name) or raw_name.startswith("hexoses ("):
        return FeatureIdentityClass.AGGREGATE
    return None


def _require_nonempty_string(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise CCLEIdentityError(f"{field} must be a non-empty string")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise CCLEIdentityError(f"{field} must be a lowercase SHA-256 digest")
    return value


def validate_identity_record(record: Mapping[str, Any]) -> None:
    """Reject structures on non-unique features and unproven unique matches."""

    if record.get("schema_version") != IDENTITY_SCHEMA_VERSION:
        raise CCLEIdentityError("unsupported identity schema version")
    if record.get("dataset_id") != "CCLE_METABOLOMICS_2019":
        raise CCLEIdentityError("unexpected dataset identity")
    raw_name = _require_nonempty_string(record, "raw_name")
    if record.get("normalized_name") != normalized_lexical_name(raw_name):
        raise CCLEIdentityError("normalized_name violates lexical-only policy")
    try:
        identity_class = FeatureIdentityClass(str(record.get("identity_class")))
    except ValueError as error:
        raise CCLEIdentityError("invalid identity_class") from error

    explicit_class = lexical_feature_class(raw_name)
    if explicit_class is not None and identity_class is not explicit_class:
        raise CCLEIdentityError("explicit composite/aggregate notation was promoted")
    if explicit_class is None and identity_class in {
        FeatureIdentityClass.COMPOSITE,
        FeatureIdentityClass.AGGREGATE,
    }:
        raise CCLEIdentityError("non-singleton class lacks explicit source notation")

    evidence = record.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise CCLEIdentityError("identity evidence must be a non-empty list")
    for index, item in enumerate(evidence):
        if not isinstance(item, Mapping):
            raise CCLEIdentityError("identity evidence entries must be objects")
        _require_nonempty_string(item, "evidence_type")
        if "raw_sha256" in item:
            _require_sha256(item["raw_sha256"], f"evidence[{index}].raw_sha256")

    unique_fields = (
        "pubchem_cid",
        "pubchem_title",
        "canonical_smiles",
        "isomeric_smiles",
        "pubchem_record_url",
    )
    if identity_class in {
        FeatureIdentityClass.EXACT,
        FeatureIdentityClass.AUTHORITY_ALIAS,
    }:
        if not isinstance(record.get("pubchem_cid"), int) or record["pubchem_cid"] <= 0:
            raise CCLEIdentityError("resolved identity requires a positive PubChem CID")
        for field in unique_fields[1:]:
            _require_nonempty_string(record, field)
        if record.get("pubchem_exact_synonym_verified") is not True:
            raise CCLEIdentityError(
                "resolved identity lacks exact PubChem synonym proof"
            )
        if identity_class is FeatureIdentityClass.EXACT:
            if normalized_lexical_name(
                record["pubchem_title"]
            ) != normalized_lexical_name(raw_name):
                raise CCLEIdentityError("EXACT record title is not lexically exact")
        elif normalized_lexical_name(
            record["pubchem_title"]
        ) == normalized_lexical_name(raw_name):
            raise CCLEIdentityError("AUTHORITY_ALIAS record is actually an exact title")
        chebi_id = record.get("chebi_id")
        if chebi_id is not None and (
            not isinstance(chebi_id, str) or CHEBI_PATTERN.fullmatch(chebi_id) is None
        ):
            raise CCLEIdentityError("invalid ChEBI identifier")
    else:
        forbidden = [field for field in unique_fields if record.get(field) is not None]
        if record.get("chebi_id") is not None:
            forbidden.append("chebi_id")
        if record.get("pubchem_exact_synonym_verified") is not False:
            raise CCLEIdentityError(
                "non-unique identity claims exact synonym verification"
            )
        if forbidden:
            raise CCLEIdentityError(
                f"non-unique feature carries unique identity fields: {sorted(forbidden)}"
            )

    structure_ready = identity_class in {
        FeatureIdentityClass.EXACT,
        FeatureIdentityClass.AUTHORITY_ALIAS,
    }
    if record.get("structure_ready") is not structure_ready:
        raise CCLEIdentityError("structure_ready disagrees with identity_class")


def identity_jsonl_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    checked: list[dict[str, Any]] = []
    for record in records:
        validate_identity_record(record)
        checked.append(dict(record))
    checked.sort(key=lambda item: (item["normalized_name"], item["raw_name"]))
    return b"".join(
        (
            json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for item in checked
    )


def load_identity_jsonl(
    path: str | Path,
    *,
    expected_feature_count: int | None = None,
) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise CCLEIdentityError(f"invalid JSON at line {line_number}") from error
        if not isinstance(record, dict):
            raise CCLEIdentityError(f"identity line {line_number} is not an object")
        validate_identity_record(record)
        records.append(record)
    if not records:
        raise CCLEIdentityError("identity registry is empty")
    raw_names = [record["raw_name"] for record in records]
    if len(raw_names) != len(set(raw_names)):
        raise CCLEIdentityError("duplicate CCLE feature identity")
    if expected_feature_count is not None and len(records) != expected_feature_count:
        raise CCLEIdentityError("identity registry feature count mismatch")
    if identity_jsonl_bytes(records) != Path(path).read_bytes():
        raise CCLEIdentityError(
            "identity registry is not canonically sorted/serialized"
        )
    return tuple(records)


def load_frozen_ccle_identity_registry(
    registry_path: str | Path,
    *,
    feature_manifest_path: str | Path,
) -> tuple[dict[str, Any], ...]:
    """Load the checked registry in original CCLE feature order.

    A self-consistent replacement registry is insufficient: downstream FM
    extraction must bind both the checked-in 225-name source manifest and the
    exact reviewed registry bytes.  Returning source order also prevents a
    sorted JSONL order from silently reindexing the assay columns.
    """

    registry = Path(registry_path)
    feature_manifest = Path(feature_manifest_path)
    if not registry.is_file() or not feature_manifest.is_file():
        raise CCLEIdentityError("frozen registry or feature manifest is missing")
    if sha256_file(registry) != FROZEN_IDENTITY_REGISTRY_SHA256:
        raise CCLEIdentityError("CCLE identity registry hash mismatch")
    if sha256_file(feature_manifest) != FROZEN_FEATURE_MANIFEST_SHA256:
        raise CCLEIdentityError("CCLE feature manifest hash mismatch")
    try:
        manifest = json.loads(feature_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CCLEIdentityError("CCLE feature manifest is invalid") from error
    if not isinstance(manifest, dict):
        raise CCLEIdentityError("CCLE feature manifest must be an object")
    features = manifest.get("features")
    if (
        manifest.get("dataset_id") != "CCLE_METABOLOMICS_2019"
        or manifest.get("source_sha256") != FROZEN_DATASET_SOURCE_SHA256
        or manifest.get("feature_count") != 225
        or not isinstance(features, list)
        or len(features) != 225
        or any(not isinstance(item, str) or not item for item in features)
        or len(set(features)) != len(features)
    ):
        raise CCLEIdentityError("CCLE feature manifest violates the frozen contract")

    records = load_identity_jsonl(registry, expected_feature_count=225)
    by_name = {str(record["raw_name"]): record for record in records}
    if set(by_name) != set(features):
        raise CCLEIdentityError(
            "identity registry does not cover the frozen feature set"
        )
    for record in records:
        frozen_evidence = [
            item
            for item in record["evidence"]
            if item.get("evidence_type") == "frozen_dataset_feature_name"
        ]
        if len(frozen_evidence) != 1:
            raise CCLEIdentityError(
                "identity record requires one frozen dataset-name evidence entry"
            )
        evidence = frozen_evidence[0]
        if (
            evidence.get("feature_manifest_sha256") != FROZEN_FEATURE_MANIFEST_SHA256
            or evidence.get("dataset_source_sha256") != FROZEN_DATASET_SOURCE_SHA256
        ):
            raise CCLEIdentityError("identity evidence does not bind the frozen source")
    return tuple(by_name[str(name)] for name in features)


def identity_summary(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    items = tuple(records)
    for item in items:
        validate_identity_record(item)
    counts = Counter(str(item["identity_class"]) for item in items)
    return {
        "counts_by_identity_class": {
            status.value: counts.get(status.value, 0) for status in FeatureIdentityClass
        },
        "feature_count": len(items),
        "matching_policy": {
            "automatic_fuzzy_resolution": False,
            "authority_alias_requires_exact_returned_synonym": True,
            "composite_or_aggregate_unique_structure": False,
        },
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "structure_ready_count": sum(item["structure_ready"] is True for item in items),
        "unique_chebi_primary_count": sum(
            item.get("chebi_id") is not None for item in items
        ),
        "unique_pubchem_cid_count": len(
            {
                item["pubchem_cid"]
                for item in items
                if item.get("pubchem_cid") is not None
            }
        ),
    }

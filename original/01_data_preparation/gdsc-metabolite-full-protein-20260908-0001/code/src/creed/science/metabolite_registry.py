"""Auditable cross-cohort metabolite identity registry.

The registry deliberately performs only deterministic lexical equality and
explicit, provenance-bearing alias resolution.  It never promotes a fuzzy or
punctuation-derived match to a unique chemical identity.
"""

from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


BUILDER_ID = "CREED_METABOLITE_IDENTITY_REGISTRY_V1"
REGISTRY_SCHEMA_VERSION = 1


class RegistryError(ValueError):
    """Raised when an input cannot support an auditable identity decision."""


class IdentityStatus(StrEnum):
    EXACT = "EXACT"
    CURATED_ALIAS = "CURATED_ALIAS"
    AMBIGUOUS = "AMBIGUOUS"
    UNKNOWN = "UNKNOWN"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalized_lexical_name(value: str) -> str:
    """Return the only automatic name normalization permitted by V1.

    NFKC normalization, surrounding/internal whitespace normalization, and
    Unicode case folding are lexical operations.  Punctuation, underscores,
    stereochemistry, salts, abbreviations, and biochemical synonyms are left
    untouched and therefore require an explicit curated alias.
    """

    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class SourceDescriptor:
    role: str
    source_name: str
    source_sha256: str
    path: str
    dataset_id: str | None = None

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        role: str,
        dataset_id: str | None = None,
    ) -> "SourceDescriptor":
        source = Path(path).resolve()
        if not source.is_file():
            raise RegistryError(f"missing {role} input: {source}")
        return cls(
            role=role,
            source_name=source.name,
            source_sha256=sha256_file(source),
            path=str(source),
            dataset_id=dataset_id,
        )

    def receipt_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "path": self.path,
            "role": self.role,
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True, slots=True)
class DatasetFeature:
    dataset_id: str
    dataset_local_id: str
    raw_name: str
    source_name: str
    source_sha256: str
    source_locator: str

    def __post_init__(self) -> None:
        for field_name in (
            "dataset_id",
            "dataset_local_id",
            "raw_name",
            "source_name",
            "source_sha256",
            "source_locator",
        ):
            if not str(getattr(self, field_name)).strip():
                raise RegistryError(f"empty DatasetFeature.{field_name}")


@dataclass(frozen=True, slots=True)
class ReferenceRecord:
    canonical_id: str
    canonical_name: str
    smiles: str | None
    source_name: str
    source_sha256: str
    source_locator: str

    def candidate_dict(self) -> dict[str, Any]:
        return {
            "canonical_id": self.canonical_id,
            "canonical_name": self.canonical_name,
            "smiles": self.smiles,
            "structure_available": self.smiles is not None,
            "evidence_source": {
                "source_locator": self.source_locator,
                "source_name": self.source_name,
                "source_sha256": self.source_sha256,
            },
        }


@dataclass(frozen=True, slots=True)
class CuratedAlias:
    dataset_id: str
    raw_name: str
    canonical_id: str
    curation_id: str
    evidence_source: str
    source_name: str
    source_sha256: str
    source_locator: str

    def evidence_dict(self) -> dict[str, Any]:
        return {
            "canonical_id": self.canonical_id,
            "curation_id": self.curation_id,
            "evidence_source": self.evidence_source,
            "evidence_type": "explicit_curated_alias",
            "source_locator": self.source_locator,
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True, slots=True)
class ParsedFeatures:
    features: tuple[DatasetFeature, ...]
    source: SourceDescriptor


@dataclass(frozen=True, slots=True)
class ReferenceCatalog:
    records_by_id: Mapping[str, ReferenceRecord]
    ids_by_normalized_name: Mapping[str, tuple[str, ...]]
    source: SourceDescriptor

    @classmethod
    def from_csv(cls, path: str | Path) -> "ReferenceCatalog":
        source = SourceDescriptor.from_path(path, role="structure_reference")
        records: dict[str, ReferenceRecord] = {}
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            _require_columns(reader.fieldnames, ("hmdb_id", "name", "smiles"), path)
            for row_number, row in enumerate(reader, start=2):
                canonical_id = (row.get("hmdb_id") or "").strip()
                canonical_name = (row.get("name") or "").strip()
                smiles_text = (row.get("smiles") or "").strip()
                if not canonical_id or not canonical_name:
                    raise RegistryError(
                        f"empty hmdb_id/name in {Path(path).name}:{row_number}"
                    )
                record = ReferenceRecord(
                    canonical_id=canonical_id,
                    canonical_name=canonical_name,
                    smiles=smiles_text or None,
                    source_name=source.source_name,
                    source_sha256=source.source_sha256,
                    source_locator=f"row:{row_number}",
                )
                existing = records.get(canonical_id)
                if existing is not None and (
                    existing.canonical_name != record.canonical_name
                    or existing.smiles != record.smiles
                ):
                    raise RegistryError(
                        f"conflicting reference rows for {canonical_id}"
                    )
                records.setdefault(canonical_id, record)
        if not records:
            raise RegistryError("structure reference contains no records")
        by_name: defaultdict[str, list[str]] = defaultdict(list)
        for record in records.values():
            by_name[normalized_lexical_name(record.canonical_name)].append(
                record.canonical_id
            )
        return cls(
            records_by_id=dict(sorted(records.items())),
            ids_by_normalized_name={
                name: tuple(sorted(ids)) for name, ids in sorted(by_name.items())
            },
            source=source,
        )

    def exact_candidates(self, raw_name: str) -> tuple[ReferenceRecord, ...]:
        identifiers = self.ids_by_normalized_name.get(
            normalized_lexical_name(raw_name), ()
        )
        return tuple(self.records_by_id[item] for item in identifiers)


@dataclass(frozen=True, slots=True)
class AliasCatalog:
    aliases_by_key: Mapping[tuple[str, str], tuple[CuratedAlias, ...]]
    source: SourceDescriptor | None

    @classmethod
    def empty(cls) -> "AliasCatalog":
        return cls(aliases_by_key={}, source=None)

    @classmethod
    def from_csv(cls, path: str | Path) -> "AliasCatalog":
        source = SourceDescriptor.from_path(path, role="curated_alias_manifest")
        grouped: defaultdict[tuple[str, str], list[CuratedAlias]] = defaultdict(list)
        required = (
            "dataset_id",
            "raw_name",
            "canonical_id",
            "curation_id",
            "evidence_source",
        )
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            _require_columns(reader.fieldnames, required, path)
            for row_number, row in enumerate(reader, start=2):
                values = {key: (row.get(key) or "").strip() for key in required}
                missing = [key for key, value in values.items() if not value]
                if missing:
                    raise RegistryError(
                        f"empty alias fields {missing} in {Path(path).name}:{row_number}"
                    )
                alias = CuratedAlias(
                    **values,
                    source_name=source.source_name,
                    source_sha256=source.source_sha256,
                    source_locator=f"row:{row_number}",
                )
                key = (
                    alias.dataset_id,
                    normalized_lexical_name(alias.raw_name),
                )
                grouped[key].append(alias)
        aliases = {
            key: tuple(
                sorted(
                    values,
                    key=lambda item: (
                        item.canonical_id,
                        item.curation_id,
                        item.evidence_source,
                    ),
                )
            )
            for key, values in sorted(grouped.items())
        }
        return cls(aliases_by_key=aliases, source=source)

    def candidates_for(self, feature: DatasetFeature) -> tuple[CuratedAlias, ...]:
        return self.aliases_by_key.get(
            (feature.dataset_id, normalized_lexical_name(feature.raw_name)), ()
        )

    @property
    def row_count(self) -> int:
        return sum(len(items) for items in self.aliases_by_key.values())


@dataclass(frozen=True, slots=True)
class RegistryRecord:
    dataset_id: str
    dataset_local_id: str
    raw_name: str
    normalized_name: str
    identity_status: IdentityStatus
    canonical_id: str | None
    canonical_name: str | None
    dataset_local_token: str | None
    candidate_structures: tuple[ReferenceRecord, ...]
    evidence_sources: tuple[Mapping[str, Any], ...]
    source_file_sha256: str

    def sort_key(self) -> tuple[str, str, str]:
        return (self.dataset_id, self.normalized_name, self.dataset_local_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_structures": [
                candidate.candidate_dict() for candidate in self.candidate_structures
            ],
            "canonical_id": self.canonical_id,
            "canonical_name": self.canonical_name,
            "dataset_id": self.dataset_id,
            "dataset_local_id": self.dataset_local_id,
            "dataset_local_token": self.dataset_local_token,
            "evidence_sources": [dict(item) for item in self.evidence_sources],
            "identity_status": self.identity_status.value,
            "normalized_name": self.normalized_name,
            "raw_name": self.raw_name,
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "source_file_sha256": self.source_file_sha256,
        }


def _require_columns(
    actual: Sequence[str] | None,
    required: Sequence[str],
    path: str | Path,
) -> None:
    if actual is None:
        raise RegistryError(f"missing header in {path}")
    missing = [name for name in required if name not in actual]
    if missing:
        raise RegistryError(f"missing columns {missing} in {path}")


def load_ccle_features(
    path: str | Path,
    *,
    dataset_id: str = "CCLE_METABOLOMICS_2019",
) -> ParsedFeatures:
    source = SourceDescriptor.from_path(
        path,
        role="dataset_feature_source",
        dataset_id=dataset_id,
    )
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise RegistryError(f"empty CCLE CSV: {path}") from error
    if header[:2] != ["CCLE_ID", "DepMap_ID"]:
        raise RegistryError(
            "CCLE metabolomics header must begin with CCLE_ID,DepMap_ID"
        )
    names = [item.strip() for item in header[2:]]
    if not names or any(not name for name in names):
        raise RegistryError("CCLE metabolite feature names must be non-empty")
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise RegistryError(f"duplicate CCLE metabolite names: {duplicates[:5]}")
    return ParsedFeatures(
        features=tuple(
            DatasetFeature(
                dataset_id=dataset_id,
                dataset_local_id=name,
                raw_name=name,
                source_name=source.source_name,
                source_sha256=source.source_sha256,
                source_locator=f"header_column:{column}",
            )
            for column, name in enumerate(names, start=3)
        ),
        source=source,
    )


def load_gbm_features(
    path: str | Path,
    *,
    dataset_id: str = "CPTAC_GBM_2021_METABOLOME",
) -> ParsedFeatures:
    source = SourceDescriptor.from_path(
        path,
        role="dataset_feature_source",
        dataset_id=dataset_id,
    )
    features: list[DatasetFeature] = []
    seen_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        _require_columns(reader.fieldnames, ("ENTITY_STABLE_ID", "NAME"), path)
        for row_number, row in enumerate(reader, start=2):
            local_id = (row.get("ENTITY_STABLE_ID") or "").strip()
            raw_name = (row.get("NAME") or "").strip()
            if not local_id or not raw_name:
                raise RegistryError(
                    f"empty GBM stable ID/name in {Path(path).name}:{row_number}"
                )
            if local_id in seen_ids:
                raise RegistryError(f"duplicate GBM stable ID: {local_id}")
            seen_ids.add(local_id)
            features.append(
                DatasetFeature(
                    dataset_id=dataset_id,
                    dataset_local_id=local_id,
                    raw_name=raw_name,
                    source_name=source.source_name,
                    source_sha256=source.source_sha256,
                    source_locator=f"row:{row_number}",
                )
            )
    if not features:
        raise RegistryError("GBM metabolomics table contains no features")
    return ParsedFeatures(features=tuple(features), source=source)


def _dataset_local_token(feature: DatasetFeature) -> str:
    payload = (
        f"{feature.dataset_id}\0{feature.dataset_local_id}\0{feature.raw_name}"
    ).encode("utf-8")
    return f"LOCAL::{feature.dataset_id}::{sha256_bytes(payload)[:24]}"


def resolve_feature(
    feature: DatasetFeature,
    reference: ReferenceCatalog,
    aliases: AliasCatalog | None = None,
) -> RegistryRecord:
    aliases = aliases or AliasCatalog.empty()
    dataset_evidence: dict[str, Any] = {
        "dataset_id": feature.dataset_id,
        "dataset_local_id": feature.dataset_local_id,
        "evidence_type": "dataset_feature",
        "source_locator": feature.source_locator,
        "source_name": feature.source_name,
        "source_sha256": feature.source_sha256,
    }
    exact = reference.exact_candidates(feature.raw_name)
    if exact:
        status = IdentityStatus.EXACT if len(exact) == 1 else IdentityStatus.AMBIGUOUS
        match_evidence: tuple[Mapping[str, Any], ...] = (
            {
                "evidence_type": "normalized_lexical_exact",
                "normalization_rule": "NFKC_WHITESPACE_CASEFOLD_ONLY",
                "reference_source_name": reference.source.source_name,
                "reference_source_sha256": reference.source.source_sha256,
            },
        )
        candidates = exact
    else:
        alias_rows = aliases.candidates_for(feature)
        if alias_rows:
            missing = sorted(
                {
                    alias.canonical_id
                    for alias in alias_rows
                    if alias.canonical_id not in reference.records_by_id
                }
            )
            if missing:
                raise RegistryError(
                    f"curated alias points to absent reference IDs: {missing}"
                )
            candidate_ids = sorted({alias.canonical_id for alias in alias_rows})
            candidates = tuple(reference.records_by_id[item] for item in candidate_ids)
            status = (
                IdentityStatus.CURATED_ALIAS
                if len(candidates) == 1
                else IdentityStatus.AMBIGUOUS
            )
            match_evidence = tuple(alias.evidence_dict() for alias in alias_rows)
        else:
            candidates = ()
            status = IdentityStatus.UNKNOWN
            match_evidence = (
                {
                    "evidence_type": "no_permitted_identity_match",
                    "matching_policy": (
                        "normalized lexical exact or explicit curated alias only; "
                        "no fuzzy unique resolution"
                    ),
                    "reference_source_name": reference.source.source_name,
                    "reference_source_sha256": reference.source.source_sha256,
                },
            )
    resolved = candidates[0] if len(candidates) == 1 else None
    return RegistryRecord(
        dataset_id=feature.dataset_id,
        dataset_local_id=feature.dataset_local_id,
        raw_name=feature.raw_name,
        normalized_name=normalized_lexical_name(feature.raw_name),
        identity_status=status,
        canonical_id=resolved.canonical_id if resolved is not None else None,
        canonical_name=resolved.canonical_name if resolved is not None else None,
        dataset_local_token=(
            _dataset_local_token(feature) if status is IdentityStatus.UNKNOWN else None
        ),
        candidate_structures=tuple(
            sorted(candidates, key=lambda item: item.canonical_id)
        ),
        evidence_sources=(dataset_evidence, *match_evidence),
        source_file_sha256=feature.source_sha256,
    )


def build_registry(
    features: Iterable[DatasetFeature],
    reference: ReferenceCatalog,
    aliases: AliasCatalog | None = None,
) -> tuple[RegistryRecord, ...]:
    aliases = aliases or AliasCatalog.empty()
    feature_list = tuple(features)
    keys = [(item.dataset_id, item.dataset_local_id) for item in feature_list]
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicates:
        raise RegistryError(f"duplicate dataset-local feature keys: {duplicates[:5]}")
    return tuple(
        sorted(
            (resolve_feature(item, reference, aliases) for item in feature_list),
            key=RegistryRecord.sort_key,
        )
    )


def registry_jsonl_bytes(records: Iterable[RegistryRecord]) -> bytes:
    ordered = sorted(tuple(records), key=RegistryRecord.sort_key)
    return b"".join(canonical_json_bytes(record.to_dict()) for record in ordered)


def registry_summary(
    records: Sequence[RegistryRecord],
    *,
    reference_count: int,
    alias_row_count: int,
    alias_manifest_supplied: bool,
) -> dict[str, Any]:
    status_names = tuple(item.value for item in IdentityStatus)
    totals = Counter(record.identity_status.value for record in records)
    by_dataset: dict[str, dict[str, int]] = {}
    for dataset_id in sorted({record.dataset_id for record in records}):
        dataset_counts = Counter(
            record.identity_status.value
            for record in records
            if record.dataset_id == dataset_id
        )
        by_dataset[dataset_id] = {
            status: dataset_counts.get(status, 0) for status in status_names
        }
        by_dataset[dataset_id]["TOTAL"] = sum(dataset_counts.values())
    structure_ready = sum(
        record.canonical_id is not None
        and bool(record.candidate_structures)
        and record.candidate_structures[0].smiles is not None
        for record in records
    )
    resolved_ids_by_dataset = {
        dataset_id: sorted(
            {
                record.canonical_id
                for record in records
                if record.dataset_id == dataset_id
                and record.canonical_id is not None
                and bool(record.candidate_structures)
                and record.candidate_structures[0].smiles is not None
            }
        )
        for dataset_id in sorted({record.dataset_id for record in records})
    }
    pairwise_overlap: list[dict[str, Any]] = []
    dataset_ids = sorted(resolved_ids_by_dataset)
    for left_index, left in enumerate(dataset_ids):
        for right in dataset_ids[left_index + 1 :]:
            shared = sorted(
                set(resolved_ids_by_dataset[left])
                & set(resolved_ids_by_dataset[right])
            )
            pairwise_overlap.append(
                {
                    "canonical_ids": shared,
                    "dataset_a": left,
                    "dataset_b": right,
                    "structure_ready_canonical_id_count": len(shared),
                }
            )
    return {
        "alias_manifest_supplied": alias_manifest_supplied,
        "alias_row_count": alias_row_count,
        "builder_id": BUILDER_ID,
        "claim_authority": "none",
        "counts_by_dataset": by_dataset,
        "counts_by_status": {status: totals.get(status, 0) for status in status_names},
        "matching_policy": {
            "automatic": "NFKC_WHITESPACE_CASEFOLD_ONLY",
            "curated_alias_requires_explicit_manifest": True,
            "fuzzy_unique_resolution": False,
            "punctuation_or_synonym_rewrite": False,
        },
        "pairwise_structure_ready_overlap": pairwise_overlap,
        "reference_record_count": reference_count,
        "registry_record_count": len(records),
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "status": "development_only_non_authorizing",
        "structure_ready_unique_record_count": structure_ready,
    }


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_registry_bundle_from_files(
    *,
    ccle_path: str | Path,
    gbm_path: str | Path,
    reference_path: str | Path,
    output_directory: str | Path,
    curated_alias_path: str | Path | None = None,
    ccle_dataset_id: str = "CCLE_METABOLOMICS_2019",
    gbm_dataset_id: str = "CPTAC_GBM_2021_METABOLOME",
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build one create-once, non-authorizing registry artifact bundle."""

    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started_at = generated_at or _utc_now()
    try:
        ccle = load_ccle_features(ccle_path, dataset_id=ccle_dataset_id)
        gbm = load_gbm_features(gbm_path, dataset_id=gbm_dataset_id)
        reference = ReferenceCatalog.from_csv(reference_path)
        aliases = (
            AliasCatalog.from_csv(curated_alias_path)
            if curated_alias_path is not None
            else AliasCatalog.empty()
        )
        records = build_registry(
            (*ccle.features, *gbm.features),
            reference,
            aliases,
        )
        registry_payload = registry_jsonl_bytes(records)
        summary = registry_summary(
            records,
            reference_count=len(reference.records_by_id),
            alias_row_count=aliases.row_count,
            alias_manifest_supplied=aliases.source is not None,
        )
        summary_payload = canonical_json_bytes(summary)
        _write_exclusive(output / "registry.jsonl", registry_payload)
        _write_exclusive(output / "summary.json", summary_payload)

        sources = [ccle.source, gbm.source, reference.source]
        if aliases.source is not None:
            sources.append(aliases.source)
        receipt = {
            "artifact_sha256": {
                "registry.jsonl": sha256_bytes(registry_payload),
                "summary.json": sha256_bytes(summary_payload),
            },
            "builder_id": BUILDER_ID,
            "claim_authority": "none",
            "completed_at": started_at,
            "input_sources": [
                source.receipt_dict()
                for source in sorted(
                    sources,
                    key=lambda item: (
                        item.role,
                        item.dataset_id or "",
                        item.source_name,
                    ),
                )
            ],
            "record_count": len(records),
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "status": "completed_development_registry_non_authorizing",
        }
        receipt_payload = canonical_json_bytes(receipt)
        _write_exclusive(output / "receipt.json", receipt_payload)
        hashes = {
            "receipt.json": sha256_bytes(receipt_payload),
            "registry.jsonl": sha256_bytes(registry_payload),
            "summary.json": sha256_bytes(summary_payload),
        }
        checksum_payload = "".join(
            f"{digest}  {name}\n" for name, digest in sorted(hashes.items())
        ).encode("ascii")
        _write_exclusive(output / "SHA256SUMS.txt", checksum_payload)
        return receipt
    except Exception as error:
        failure = {
            "builder_id": BUILDER_ID,
            "claim_authority": "none",
            "error": str(error),
            "error_type": type(error).__name__,
            "failed_at": started_at,
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "status": "failed_development_registry_non_authorizing",
        }
        failure_path = output / "failure.json"
        if not failure_path.exists():
            _write_exclusive(failure_path, canonical_json_bytes(failure))
        raise

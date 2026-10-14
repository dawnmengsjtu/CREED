#!/usr/bin/env python3
"""Build a create-once, source-backed CCLE metabolite identity V2 bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from creed.science.ccle_metabolite_identity import (
    IDENTITY_SCHEMA_VERSION,
    LIPID_TOTAL_COMPOSITION,
    FeatureIdentityClass,
    identity_jsonl_bytes,
    identity_summary,
    lexical_feature_class,
)
from creed.science.metabolite_registry import normalized_lexical_name


CHEBI_PATTERN = re.compile(r"^CHEBI:(\d+)$", re.IGNORECASE)


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


class QuerySource:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.source_id = self.root.name
        self.receipt_path = self.root / "receipt.json"
        self.index_path = self.root / "query_index.json"
        self.requests_path = self.root / "requests.jsonl"
        for path in (self.receipt_path, self.index_path, self.requests_path):
            if not path.is_file():
                raise ValueError(f"missing query-source artifact: {path}")
        self.receipt = json.loads(self.receipt_path.read_text(encoding="utf-8"))
        self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.requests = {
            item["request_id"]: item
            for item in (
                json.loads(line)
                for line in self.requests_path.read_text(encoding="utf-8").splitlines()
            )
        }
        if len(self.requests) != len(
            self.requests_path.read_text(encoding="utf-8").splitlines()
        ):
            raise ValueError(f"duplicate request ID in {self.source_id}")

    def property_request(self, feature_name: str) -> Mapping[str, Any] | None:
        request_id = self.index.get("feature_property_request_ids", {}).get(
            feature_name
        )
        return self.requests.get(request_id) if request_id is not None else None

    def synonym_request(self, cid: int) -> Mapping[str, Any] | None:
        request_id = self.index.get("pubchem_synonym_request_ids", {}).get(str(cid))
        return self.requests.get(request_id) if request_id is not None else None

    def chebi_request(self, chebi_id: str) -> Mapping[str, Any] | None:
        request_id = self.index.get("chebi_request_ids", {}).get(chebi_id)
        return self.requests.get(request_id) if request_id is not None else None

    def raw_path(self, request: Mapping[str, Any]) -> Path | None:
        request_id = str(request["request_id"])
        candidates = tuple((self.root / "raw").rglob(f"{request_id}.json"))
        if not candidates:
            return None
        if len(candidates) != 1:
            raise ValueError(f"ambiguous raw payload for {request_id}")
        path = candidates[0]
        observed = sha256_file(path)
        if request.get("raw_sha256") != observed:
            raise ValueError(f"raw payload hash mismatch for {request_id}")
        return path

    def raw_json(self, request: Mapping[str, Any]) -> Mapping[str, Any] | None:
        path = self.raw_path(request)
        if path is None:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError(f"raw payload is not an object: {path}")
        return value

    def evidence(
        self,
        request: Mapping[str, Any],
        *,
        evidence_type: str,
    ) -> dict[str, Any]:
        path = self.raw_path(request)
        envelope = request.get("skill_envelope", {})
        sources = envelope.get("sources", []) if isinstance(envelope, Mapping) else []
        urls = sorted(
            {
                value
                for source in sources
                if isinstance(source, Mapping)
                for value in (source.get("canonical_url"), source.get("url"))
                if isinstance(value, str) and value
            }
        )
        evidence = {
            "evidence_type": evidence_type,
            "record_urls": urls,
            "request_id": request["request_id"],
            "request_source_id": self.source_id,
        }
        if path is not None:
            evidence["raw_path"] = path.relative_to(self.root).as_posix()
            evidence["raw_sha256"] = request["raw_sha256"]
        return evidence

    def receipt_record(self) -> dict[str, Any]:
        return {
            "query_index_sha256": sha256_file(self.index_path),
            "receipt_sha256": sha256_file(self.receipt_path),
            "requests_sha256": sha256_file(self.requests_path),
            "source_id": self.source_id,
            "source_status": self.receipt.get("status"),
            "source_raw_root_digest": self.receipt.get("raw_root_digest"),
        }


def _property_rows(
    source: QuerySource,
    request: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    raw = source.raw_json(request)
    if raw is None:
        return []
    rows = raw.get("PropertyTable", {}).get("Properties", [])
    return [row for row in rows if isinstance(row, Mapping)]


def _synonyms(
    source: QuerySource,
    request: Mapping[str, Any],
) -> list[str]:
    raw = source.raw_json(request)
    if raw is None:
        return []
    rows = raw.get("InformationList", {}).get("Information", [])
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
        return []
    return [str(value) for value in rows[0].get("Synonym", [])]


def _component_labels(raw_name: str) -> list[str]:
    if raw_name.startswith("cis/trans-"):
        suffix = raw_name.removeprefix("cis/trans-")
        return [f"cis-{suffix}", f"trans-{suffix}"]
    return [item.strip() for item in raw_name.split("/")]


def _aggregate_descriptor(raw_name: str) -> dict[str, Any]:
    lipid = LIPID_TOTAL_COMPOSITION.fullmatch(raw_name)
    if lipid is not None:
        lipid_class = lipid.group("class")
        peak_label = None
        if lipid_class.startswith("PC-"):
            peak_label = lipid_class.removeprefix("PC-")
            lipid_class = "PC"
        return {
            "aggregate_type": "lipid_sum_composition",
            "double_bond_total": int(lipid.group("double_bonds")),
            "lipid_class": lipid_class,
            "peak_label": peak_label,
            "total_acyl_carbons": int(lipid.group("carbons")),
        }
    return {
        "aggregate_type": "hexose_class_signal",
        "chromatography": raw_name.removeprefix("hexoses (").removesuffix(")"),
    }


def _base_record(
    *,
    raw_name: str,
    feature_manifest: Mapping[str, Any],
    feature_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "aggregate_descriptor": None,
        "candidate_components": [],
        "candidate_pubchem_records": [],
        "canonical_smiles": None,
        "chebi_candidates": [],
        "chebi_id": None,
        "chebi_record_url": None,
        "dataset_id": "CCLE_METABOLOMICS_2019",
        "evidence": [
            {
                "dataset_source_sha256": feature_manifest["source_sha256"],
                "evidence_type": "frozen_dataset_feature_name",
                "feature_manifest_sha256": feature_manifest_sha256,
            }
        ],
        "identity_class": FeatureIdentityClass.UNKNOWN.value,
        "isomeric_smiles": None,
        "normalized_name": normalized_lexical_name(raw_name),
        "pubchem_cid": None,
        "pubchem_exact_synonym_verified": False,
        "pubchem_record_url": None,
        "pubchem_title": None,
        "raw_name": raw_name,
        "resolution_reason": None,
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "structure_ready": False,
    }


def _choose_property_request(
    raw_name: str,
    sources: tuple[QuerySource, ...],
) -> tuple[QuerySource, Mapping[str, Any], list[Mapping[str, Any]]] | None:
    attempts: list[tuple[QuerySource, Mapping[str, Any], list[Mapping[str, Any]]]] = []
    for source in sources:
        request = source.property_request(raw_name)
        if request is not None:
            attempts.append((source, request, _property_rows(source, request)))
    for attempt in reversed(attempts):
        if len(attempt[2]) == 1:
            return attempt
    return attempts[-1] if attempts else None


def _chebi_candidates(
    *,
    source: QuerySource,
    synonyms: Iterable[str],
    pubchem_inchikey: str | None,
) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for synonym in sorted(set(synonyms)):
        match = CHEBI_PATTERN.fullmatch(synonym.strip())
        if match is None:
            continue
        requested_id = f"CHEBI:{match.group(1)}"
        request = source.chebi_request(requested_id)
        if request is None:
            continue
        raw = source.raw_json(request)
        if raw is None:
            continue
        actual_id = raw.get("chebi_accession")
        structure = raw.get("default_structure", {})
        if not isinstance(actual_id, str) or CHEBI_PATTERN.fullmatch(actual_id) is None:
            continue
        if not isinstance(structure, Mapping):
            structure = {}
        chebi_inchikey = structure.get("standard_inchi_key") or raw.get("inchikey")
        released = raw.get("is_released") is True
        matches = (
            isinstance(pubchem_inchikey, str)
            and isinstance(chebi_inchikey, str)
            and pubchem_inchikey == chebi_inchikey
        )
        candidates[actual_id] = {
            "chebi_id": actual_id,
            "chebi_name": raw.get("name"),
            "inchikey_matches_pubchem": matches,
            "is_released": released,
            "record_url": f"https://www.ebi.ac.uk/chebi/searchId.do?chebiId={actual_id}",
            "request_evidence": source.evidence(
                request,
                evidence_type="chebi_targeted_compound_record",
            ),
        }
    return [candidates[key] for key in sorted(candidates)]


def build_record(
    *,
    raw_name: str,
    feature_manifest: Mapping[str, Any],
    feature_manifest_sha256: str,
    sources: tuple[QuerySource, ...],
) -> dict[str, Any]:
    record = _base_record(
        raw_name=raw_name,
        feature_manifest=feature_manifest,
        feature_manifest_sha256=feature_manifest_sha256,
    )
    lexical_class = lexical_feature_class(raw_name)
    if lexical_class is FeatureIdentityClass.COMPOSITE:
        record.update(
            {
                "candidate_components": _component_labels(raw_name),
                "identity_class": FeatureIdentityClass.COMPOSITE.value,
                "resolution_reason": "source label explicitly joins multiple analytes or isomers",
            }
        )
        record["evidence"].append(
            {
                "evidence_type": "explicit_slash_delimited_nonunique_feature",
                "matching_policy": "no unique molecular structure",
            }
        )
        return record
    if lexical_class is FeatureIdentityClass.AGGREGATE:
        record.update(
            {
                "aggregate_descriptor": _aggregate_descriptor(raw_name),
                "identity_class": FeatureIdentityClass.AGGREGATE.value,
                "resolution_reason": "source label is a class or lipid sum-composition measurement",
            }
        )
        record["evidence"].append(
            {
                "evidence_type": "explicit_aggregate_feature_notation",
                "matching_policy": "no unique molecular structure",
            }
        )
        return record

    chosen = _choose_property_request(raw_name, sources)
    if chosen is None:
        record["resolution_reason"] = "no PubChem property request was recorded"
        return record
    source, property_request, rows = chosen
    record["evidence"].append(
        source.evidence(
            property_request,
            evidence_type="pubchem_exact_name_property_query",
        )
    )
    if len(rows) != 1:
        record["candidate_pubchem_records"] = sorted(
            (
                {
                    "pubchem_cid": row.get("CID"),
                    "pubchem_title": row.get("Title"),
                    "record_url": (
                        f"https://pubchem.ncbi.nlm.nih.gov/compound/{row['CID']}"
                        if isinstance(row.get("CID"), int)
                        else None
                    ),
                }
                for row in rows
            ),
            key=lambda item: (item["pubchem_cid"] is None, item["pubchem_cid"] or 0),
        )
        record["resolution_reason"] = (
            "PubChem exact-name request returned no record"
            if not rows
            else "PubChem exact-name request returned multiple records"
        )
        return record

    property_row = rows[0]
    cid = property_row.get("CID")
    if not isinstance(cid, int):
        record["resolution_reason"] = "PubChem record lacks an integer CID"
        return record
    synonym_request = source.synonym_request(cid)
    if synonym_request is None:
        record["resolution_reason"] = (
            "unique PubChem record lacks a captured synonym response"
        )
        return record
    synonyms = _synonyms(source, synonym_request)
    record["evidence"].append(
        source.evidence(
            synonym_request,
            evidence_type="pubchem_cid_synonym_record",
        )
    )
    exact_synonym = normalized_lexical_name(raw_name) in {
        normalized_lexical_name(value) for value in synonyms
    }
    canonical_smiles = property_row.get("ConnectivitySMILES") or property_row.get(
        "CanonicalSMILES"
    )
    isomeric_smiles = property_row.get("SMILES") or property_row.get("IsomericSMILES")
    if not exact_synonym:
        record["resolution_reason"] = (
            "source name is absent from returned PubChem synonyms"
        )
        return record
    if not isinstance(canonical_smiles, str) or not canonical_smiles:
        record["resolution_reason"] = "unique PubChem record lacks canonical SMILES"
        return record
    if not isinstance(isomeric_smiles, str) or not isomeric_smiles:
        isomeric_smiles = canonical_smiles

    title = str(property_row.get("Title") or "")
    identity_class = (
        FeatureIdentityClass.EXACT
        if normalized_lexical_name(title) == normalized_lexical_name(raw_name)
        else FeatureIdentityClass.AUTHORITY_ALIAS
    )
    chebi_candidates = _chebi_candidates(
        source=source,
        synonyms=synonyms,
        pubchem_inchikey=property_row.get("InChIKey"),
    )
    valid_chebi = [
        candidate
        for candidate in chebi_candidates
        if candidate["is_released"] and candidate["inchikey_matches_pubchem"]
    ]
    unique_chebi = valid_chebi[0] if len(valid_chebi) == 1 else None
    record.update(
        {
            "canonical_smiles": canonical_smiles,
            "chebi_candidates": chebi_candidates,
            "chebi_id": unique_chebi["chebi_id"] if unique_chebi else None,
            "chebi_record_url": unique_chebi["record_url"] if unique_chebi else None,
            "identity_class": identity_class.value,
            "isomeric_smiles": isomeric_smiles,
            "pubchem_cid": cid,
            "pubchem_exact_synonym_verified": True,
            "pubchem_record_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
            "pubchem_title": title,
            "resolution_reason": (
                "PubChem title is lexically exact"
                if identity_class is FeatureIdentityClass.EXACT
                else "source name is an exact PubChem-returned synonym"
            ),
            "structure_ready": True,
        }
    )
    if unique_chebi is not None:
        record["evidence"].append(unique_chebi["request_evidence"])
    return record


def replay_request_rows(sources: Iterable[QuerySource]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sources:
        for request_id, item in sorted(source.requests.items()):
            request = dict(item["request"])
            request.pop("raw_output_path", None)
            rows.append(
                {
                    "expected_raw_sha256": item.get("raw_sha256"),
                    "feature_names": item.get("feature_names", []),
                    "request": request,
                    "request_id": request_id,
                    "source_id": source.source_id,
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--query-source", action="append", required=True)
    parser.add_argument("--output-directory", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    feature_path = Path(args.feature_manifest).resolve()
    feature_manifest = json.loads(feature_path.read_text(encoding="utf-8"))
    features = feature_manifest.get("features")
    if not isinstance(features, list) or len(features) != 225:
        raise ValueError("feature manifest must contain the frozen 225 features")
    if len(features) != len(set(features)):
        raise ValueError("feature manifest contains duplicate names")
    sources = tuple(QuerySource(path) for path in args.query_source)
    feature_sha256 = sha256_file(feature_path)
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        records = [
            build_record(
                raw_name=str(raw_name),
                feature_manifest=feature_manifest,
                feature_manifest_sha256=feature_sha256,
                sources=sources,
            )
            for raw_name in features
        ]
        registry_payload = identity_jsonl_bytes(records)
        summary = identity_summary(records)
        summary.update(
            {
                "claim_authority": "none",
                "dataset_source_sha256": feature_manifest["source_sha256"],
                "feature_manifest_sha256": feature_sha256,
                "query_sources": [source.receipt_record() for source in sources],
                "status": "completed_identity_registry_non_authorizing",
            }
        )
        summary_payload = canonical_json(summary)
        replay_payload = b"".join(
            canonical_json(row) for row in replay_request_rows(sources)
        )
        write_exclusive(output / "registry.jsonl", registry_payload)
        write_exclusive(output / "summary.json", summary_payload)
        write_exclusive(output / "replay_requests.jsonl", replay_payload)
        receipt = {
            "artifact_sha256": {
                "registry.jsonl": hashlib.sha256(registry_payload).hexdigest(),
                "replay_requests.jsonl": hashlib.sha256(replay_payload).hexdigest(),
                "summary.json": hashlib.sha256(summary_payload).hexdigest(),
            },
            "claim_authority": "none",
            "completed_at": started_at,
            "feature_count": len(records),
            "input_feature_manifest_sha256": feature_sha256,
            "query_sources": [source.receipt_record() for source in sources],
            "schema_version": IDENTITY_SCHEMA_VERSION,
            "status": "completed_identity_registry_non_authorizing",
        }
        receipt_payload = canonical_json(receipt)
        write_exclusive(output / "receipt.json", receipt_payload)
        artifacts = sorted(path for path in output.iterdir() if path.is_file())
        write_exclusive(
            output / "SHA256SUMS.txt",
            b"".join(
                f"{sha256_file(path)}  {path.name}\n".encode("ascii")
                for path in artifacts
            ),
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        failure = {
            "claim_authority": "none",
            "error": str(error),
            "error_type": type(error).__name__,
            "failed_at": started_at,
            "schema_version": IDENTITY_SCHEMA_VERSION,
            "status": "failed_identity_registry_non_authorizing",
        }
        if not (output / "failure.json").exists():
            write_exclusive(output / "failure.json", canonical_json(failure))
        raise


if __name__ == "__main__":
    raise SystemExit(main())

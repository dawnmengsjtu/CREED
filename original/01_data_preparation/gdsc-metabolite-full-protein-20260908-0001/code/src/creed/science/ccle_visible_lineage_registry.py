"""Create and independently verify the frozen visible CCLE lineage registry.

The only decoded source cells are the non-expression identity columns in the
``Cell line level sample info`` sheet of ProCan Table S1 (``mmc2.xlsx``).  The
only decoded members of the visible CCLE NPZ are ``sample_ids.npy`` and
``partition_labels.npy``.  Assay members and the sealed internal-final cohort
are outside this module's authority.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import posixpath
import re
import stat
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Mapping, Sequence

import numpy as np

from creed.science import ccle as ccle_identity
from creed.science.ccle import (
    CCLEContractError,
    CCLEIdentityRecord,
    infer_identity_groups,
    lineage_stratum,
)


class CCLEVisibleLineageRegistryError(ValueError):
    """Raised when a source or lineage registry violates the frozen contract."""


INTENT_SCHEMA = "creed.ccle-visible-lineage-registry-intent.v1"
REGISTRY_SCHEMA = "creed.ccle-visible-lineage-registry.v1"
PRODUCER_SCHEMA = "creed.ccle-visible-lineage-registry-producer.v1"
VERIFIER_SCHEMA = "creed.ccle-visible-lineage-registry-verifier.v1"
VERIFIER_INTENT_SCHEMA = "creed.ccle-visible-lineage-registry-verifier-intent.v1"
VERIFIER_ATTEMPT_SCHEMA = "creed.ccle-visible-lineage-registry-verifier-attempt.v1"
PROTOCOL_ID = "CREED_CCLE_VISIBLE_LINEAGE_REGISTRY_V1"
SHEET_NAME = "Cell line level sample info"
ROW_FIELDS = (
    "sample_id",
    "partition",
    "identity_group",
    "tissue_type",
    "cancer_type",
    "cancer_subtype",
    "haem_lineage",
    "lineage_stratum",
)
MMC2_IDENTITY_COLUMNS = (
    "model_id",
    "Project_Identifier",
    "Cell_line",
    "Source",
    "Identifier",
    "Tissue_type",
    "Cancer_type",
    "Cancer_subtype",
    "Haem_lineage",
    "BROAD_ID",
    "CCLE_ID",
)
# The frozen worksheet is not contiguous: column F is Gender and is outside
# this registry's identity/lineage authority.  Freeze the actual permitted
# coordinates so the targeted XML reader never materializes F-column values.
MMC2_IDENTITY_COLUMN_NUMBERS = (1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
VISIBLE_IDENTITY_MEMBERS = ("sample_ids.npy", "partition_labels.npy")
LINEAGE_RULE = (
    "normalised Tissue_type plus '|haem:' and non-empty Haem_lineage; "
    "otherwise normalised Tissue_type plus '|cancer:' and Cancer_type"
)
IDENTITY_GROUP_RULE = (
    "same source Identifier, exact normalised Cell_line, or conservative "
    "explicit parental/resistant/derived/clone family"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SIDM_RE = re.compile(r"SIDM[0-9]+")
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_CODE_BYTES = 8 * 1024 * 1024
_MAX_MMC2_BYTES = 8 * 1024 * 1024
_MAX_VISIBLE_BYTES = 512 * 1024 * 1024
_MAX_ZIP_ENTRIES = 128
_MAX_XLSX_ENTRIES = 2_048
_MAX_XLSX_METADATA_BYTES = 4 * 1024 * 1024
_MAX_XLSX_SHEET_BYTES = 32 * 1024 * 1024
_MAX_XLSX_SHARED_STRINGS_BYTES = 64 * 1024 * 1024
_CELL_REFERENCE_RE = re.compile(r"([A-Z]+)([1-9][0-9]*)")
_XLSX_ESCAPE_RE = re.compile(r"_x([0-9A-Fa-f]{4})_")
_WORKBOOK_RELATIONSHIP_NAMESPACE = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)
_CODE_ROLES = frozenset(
    {"registry_module", "registry_cli", "identity_module", "registry_protocol"}
)


@dataclass(frozen=True, slots=True)
class FileSpec:
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if self.bytes < 1 or _SHA256_RE.fullmatch(self.sha256) is None:
            raise CCLEVisibleLineageRegistryError("invalid frozen file specification")


@dataclass(frozen=True, slots=True)
class MemberSpec:
    filename: str
    bytes: int
    sha256: str
    dtype: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.filename not in VISIBLE_IDENTITY_MEMBERS
            or self.bytes < 1
            or _SHA256_RE.fullmatch(self.sha256) is None
            or not self.dtype
            or not self.shape
            or any(value < 1 for value in self.shape)
        ):
            raise CCLEVisibleLineageRegistryError(
                "invalid frozen visible-member specification"
            )


@dataclass(frozen=True, slots=True)
class LineageContract:
    sample_count: int = 485
    train_count: int = 421
    validation_count: int = 64
    mmc2_record_count: int = 949
    identity_group_count: int = 485
    non_singleton_identity_group_count: int = 0
    lineage_strata_count: int = 35


@dataclass(frozen=True, slots=True)
class FrozenSources:
    mmc2: FileSpec
    acquisition_receipt: FileSpec
    visible_artifact: FileSpec
    visible_manifest: FileSpec
    visible_receipt: FileSpec
    sample_ids_member: MemberSpec
    partition_labels_member: MemberSpec
    canonical_records: FileSpec


@dataclass(frozen=True, slots=True)
class LineageRecord:
    sample_id: str
    partition: str
    identity_group: str
    tissue_type: str | None
    cancer_type: str | None
    cancer_subtype: str | None
    haem_lineage: str | None
    lineage_stratum: str

    @property
    def document(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "partition": self.partition,
            "identity_group": self.identity_group,
            "tissue_type": self.tissue_type,
            "cancer_type": self.cancer_type,
            "cancer_subtype": self.cancer_subtype,
            "haem_lineage": self.haem_lineage,
            "lineage_stratum": self.lineage_stratum,
        }


@dataclass(frozen=True, slots=True)
class VerifiedVisibleLineageRegistry:
    records: tuple[LineageRecord, ...]
    bindings: Mapping[str, object]


PRODUCTION_CONTRACT = LineageContract()
PRODUCTION_FROZEN_SOURCES = FrozenSources(
    mmc2=FileSpec(
        bytes=960_365,
        sha256="9327f665d259ef4d148d3a64e7c76cdeb51dce8dee06768efd6b7d044130e203",
    ),
    acquisition_receipt=FileSpec(
        bytes=794,
        sha256="40c7ae056b611c254c2ff1dbfce03dd8e75883b5a20b6fb6bb10ee71091cd020",
    ),
    visible_artifact=FileSpec(
        bytes=40_313_525,
        sha256="468767f19ea2c935285c0d0877475ea97cb70140662b307f39e841d91a81e5d1",
    ),
    visible_manifest=FileSpec(
        bytes=3_437,
        sha256="c716a05e5d95bce9eae30f0f90b7b5220236586adb73420ea459672fac03c898",
    ),
    visible_receipt=FileSpec(
        bytes=1_683,
        sha256="016bcd266a61772b3445f2afceb1ad0ff6868014900dd12e7442c5f6077c53a0",
    ),
    sample_ids_member=MemberSpec(
        filename="sample_ids.npy",
        bytes=17_588,
        sha256="0865b09b2d248d77f65d5dc7d4c3815a333048181515cb03bd7589a309f244a3",
        dtype="<U9",
        shape=(485,),
    ),
    partition_labels_member=MemberSpec(
        filename="partition_labels.npy",
        bytes=19_528,
        sha256="b95d4cbb7e3bf0ce3a44583b6d86becc6d8caea3227b5671cc4164380b5923ed",
        dtype="<U10",
        shape=(485,),
    ),
    canonical_records=FileSpec(
        bytes=125_275,
        sha256="f53a78f5dcdf1bc612170cae2b2eb975b79cf06759ae32b6258a29afbb80fe5d",
    ),
)


@dataclass(frozen=True, slots=True)
class _PinnedFile:
    path: Path
    raw: bytes
    dev: int
    ino: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.raw).hexdigest()

    @property
    def binding(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "dev": self.dev,
            "ino": self.ino,
            "mode": self.mode,
            "bytes": self.size,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class _ArchiveIdentity:
    path: Path
    dev: int
    ino: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @property
    def binding(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "dev": self.dev,
            "ino": self.ino,
            "mode": self.mode,
            "bytes": self.size,
        }


def completion_receipt_path(registry_root: str | Path) -> Path:
    """Return the sole adjacent verifier-completion path for a registry root."""

    root = Path(registry_root)
    return root.with_name(f"{root.name}-complete.json")


def verifier_attempt_root_path(registry_root: str | Path) -> Path:
    """Return the adjacent create-once independent-verifier attempt root."""

    root = Path(registry_root)
    return root.with_name(f"{root.name}-verification")


def produce_visible_lineage_registry(**kwargs: object) -> dict[str, object]:
    """Create one identity-only lineage registry with create-once semantics."""

    return _produce_visible_lineage_registry(**kwargs)


def verify_visible_lineage_registry(**kwargs: object) -> dict[str, object]:
    """Independently replay sources and create the adjacent completion leaf."""

    return _verify_visible_lineage_registry(**kwargs)


def load_verified_visible_lineage_registry(
    *,
    registry_root: str | Path,
    completion_receipt: str | Path,
    contract: LineageContract = PRODUCTION_CONTRACT,
    frozen: FrozenSources = PRODUCTION_FROZEN_SOURCES,
) -> VerifiedVisibleLineageRegistry:
    """Load a completed registry without reopening any source workbook/NPZ."""

    return _load_verified_visible_lineage_registry(
        registry_root=registry_root,
        completion_receipt=completion_receipt,
        contract=contract,
        frozen=frozen,
    )


def _canonical_json_bytes(value: object) -> bytes:
    try:
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
    except (TypeError, ValueError) as exc:
        raise CCLEVisibleLineageRegistryError(
            "lineage document is not canonical JSON"
        ) from exc


def _absolute(path: str | Path, role: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise CCLEVisibleLineageRegistryError(f"{role} must be an absolute path")
    return candidate.absolute()


def _descriptor_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _path_identity(value: os.stat_result) -> tuple[int, ...]:
    stable = (value.st_dev, value.st_ino, value.st_size)
    if os.name == "nt":
        return stable
    return (*stable, value.st_mtime_ns, value.st_ctime_ns)


def _reject_symlink_components(
    path: Path,
    role: str,
    *,
    include_leaf: bool,
) -> None:
    candidate = path.absolute()
    parts = candidate.parts
    if not parts:
        raise CCLEVisibleLineageRegistryError(f"{role} path is empty")
    current = Path(parts[0])
    stop = len(parts) if include_leaf else len(parts) - 1
    for part in parts[1:stop]:
        current = current / part
        try:
            observed = os.lstat(current)
        except OSError as exc:
            raise CCLEVisibleLineageRegistryError(f"{role} ancestor is absent") from exc
        if stat.S_ISLNK(observed.st_mode):
            raise CCLEVisibleLineageRegistryError(
                f"{role} contains a symbolic-link component"
            )
        if current != candidate and not stat.S_ISDIR(observed.st_mode):
            raise CCLEVisibleLineageRegistryError(f"{role} ancestor is not a directory")


def _reject_forbidden_path(path: Path, role: str) -> None:
    if any(
        "sealed" in part
        or "internal_final" in part
        or "internal-final" in part
        or part == "d4"
        for part in (value.lower() for value in path.parts)
    ):
        raise CCLEVisibleLineageRegistryError(
            f"{role} names forbidden sealed/internal-final custody"
        )


def _read_pinned_file(
    path: str | Path,
    *,
    role: str,
    maximum_bytes: int,
    expected: FileSpec | None = None,
    required_mode: int | None = None,
    reject_final: bool = False,
) -> _PinnedFile:
    source = _absolute(path, role)
    if reject_final:
        _reject_forbidden_path(source, role)
    _reject_symlink_components(source, role, include_leaf=True)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise CCLEVisibleLineageRegistryError(f"cannot pin {role}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CCLEVisibleLineageRegistryError(f"{role} is not a regular file")
        mode = stat.S_IMODE(before.st_mode)
        if required_mode is not None and os.name != "nt" and mode != required_mode:
            raise CCLEVisibleLineageRegistryError(f"{role} permission seal changed")
        if before.st_size < 1 or before.st_size > maximum_bytes:
            raise CCLEVisibleLineageRegistryError(f"{role} exceeds its byte bound")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1_048_576, remaining))
            if not chunk:
                raise CCLEVisibleLineageRegistryError(f"short read from {role}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise CCLEVisibleLineageRegistryError(f"{role} grew during read")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _descriptor_identity(before) != _descriptor_identity(after):
        raise CCLEVisibleLineageRegistryError(f"{role} changed during read")
    try:
        path_after = os.lstat(source)
    except OSError as exc:
        raise CCLEVisibleLineageRegistryError(f"{role} disappeared after read") from exc
    if stat.S_ISLNK(path_after.st_mode) or _path_identity(path_after) != _path_identity(
        after
    ):
        raise CCLEVisibleLineageRegistryError(f"{role} path identity changed")
    pinned = _PinnedFile(
        path=source,
        raw=b"".join(chunks),
        dev=before.st_dev,
        ino=before.st_ino,
        mode=mode,
        size=before.st_size,
        mtime_ns=before.st_mtime_ns,
        ctime_ns=before.st_ctime_ns,
    )
    if expected is not None and (
        pinned.size != expected.bytes or pinned.sha256 != expected.sha256
    ):
        raise CCLEVisibleLineageRegistryError(f"{role} frozen identity changed")
    return pinned


def _json_object(pinned: _PinnedFile, role: str) -> dict[str, object]:
    try:
        value = json.loads(pinned.raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CCLEVisibleLineageRegistryError(f"{role} is invalid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise CCLEVisibleLineageRegistryError(f"{role} must be a JSON object")
    return value


def _exclusive_write(path: Path, raw: bytes, *, mode: int = 0o600) -> _PinnedFile:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, mode)
    except OSError as exc:
        raise CCLEVisibleLineageRegistryError(
            f"create-once output already exists or is unsafe: {path}"
        ) from exc
    try:
        written = 0
        while written < len(raw):
            count = os.write(descriptor, raw[written:])
            if count < 1:
                raise OSError("short create-once write")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    observed = _read_pinned_file(
        path,
        role="new create-once output",
        maximum_bytes=max(1, len(raw)),
    )
    if observed.raw != raw:
        raise CCLEVisibleLineageRegistryError(
            "create-once output posterior verification failed"
        )
    return observed


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _seal_root(root: Path) -> None:
    for child in root.iterdir():
        observed = os.lstat(child)
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
            raise CCLEVisibleLineageRegistryError(
                "lineage registry root contains a non-regular artifact"
            )
        child.chmod(0o444)
    _fsync_directory(root)
    root.chmod(0o555)
    _fsync_directory(root.parent)


def _preserve_failed_attempt(
    root: Path,
    *,
    schema: str,
    stage: str,
    status: str,
) -> None:
    """Best-effort create-once failure preservation without deleting artifacts."""

    try:
        observed = os.lstat(root)
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            return
        root.chmod(0o700)
        existing: dict[str, object] = {}
        for child in sorted(root.iterdir(), key=lambda value: value.name):
            child_stat = os.lstat(child)
            if stat.S_ISLNK(child_stat.st_mode) or not stat.S_ISREG(child_stat.st_mode):
                existing[child.name] = {"regular_non_symlink": False}
                continue
            try:
                binding = _read_pinned_file(
                    child,
                    role=f"failed-attempt existing artifact {child.name}",
                    maximum_bytes=_MAX_JSON_BYTES,
                ).binding
            except BaseException:
                # O_EXCL has already consumed the name when a failed write
                # leaves a zero-byte or unreadable regular inode.  Record the
                # lstat-only identity and continue so this partial child cannot
                # suppress failure.json or leave the attempt reusable.
                existing[child.name] = {
                    "regular_non_symlink": True,
                    "binding_complete": False,
                    "path": str(child),
                    "dev": child_stat.st_dev,
                    "ino": child_stat.st_ino,
                    "mode": stat.S_IMODE(child_stat.st_mode),
                    "bytes": child_stat.st_size,
                }
            else:
                existing[child.name] = {
                    **binding,
                    "regular_non_symlink": True,
                    "binding_complete": True,
                }
        failure_path = root / "failure.json"
        if not os.path.lexists(failure_path):
            failure = {
                "schema": schema,
                "protocol_id": PROTOCOL_ID,
                "status": status,
                "failed_stage": stage,
                "existing_artifacts_before_failure_receipt": existing,
                "scientific_claim_authority": False,
                "expression_or_abundance_values_decoded": False,
                "internal_final_accessed": False,
                "model_weights_accessed": False,
                "model_invocations": 0,
                "automatic_retry": False,
            }
            _exclusive_write(failure_path, _canonical_json_bytes(failure), mode=0o444)
        _seal_root(root)
    except BaseException:
        # Preserve the original transaction failure.  The caller never retries
        # or removes this attempt, even when best-effort sealing itself fails.
        return


def _bind_code_files(code_paths: Mapping[str, str | Path]) -> dict[str, object]:
    if set(code_paths) != set(_CODE_ROLES):
        raise CCLEVisibleLineageRegistryError(
            "code path registry must contain exactly registry_module, "
            "registry_cli, identity_module, and registry_protocol"
        )
    expected_paths = {
        "registry_module": Path(__file__).resolve(),
        "identity_module": Path(ccle_identity.__file__).resolve(),
        "registry_cli": (
            Path(__file__).resolve().parents[3]
            / "scripts"
            / "build_ccle_visible_lineage_registry.py"
        ),
        "registry_protocol": (
            Path(__file__).resolve().parents[3]
            / "experiments"
            / "CCLE_VISIBLE_LINEAGE_REGISTRY_V1.json"
        ),
    }
    result: dict[str, object] = {}
    for role, path in sorted(code_paths.items()):
        source = _absolute(path, f"code file {role}")
        expected = expected_paths.get(role)
        if expected is not None and source.resolve() != expected:
            raise CCLEVisibleLineageRegistryError(
                f"{role} code role does not name the executing source"
            )
        result[role] = _read_pinned_file(
            source,
            role=f"code file {role}",
            maximum_bytes=_MAX_CODE_BYTES,
        ).binding
    return result


def _require_current_code_binding(value: object, role: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(_CODE_ROLES):
        raise CCLEVisibleLineageRegistryError(f"{role} code registry changed")
    current = {
        "registry_module": _read_pinned_file(
            Path(__file__).resolve(),
            role="currently executing registry module",
            maximum_bytes=_MAX_CODE_BYTES,
        ).binding,
        "identity_module": _read_pinned_file(
            Path(ccle_identity.__file__).resolve(),
            role="currently executing CCLE identity module",
            maximum_bytes=_MAX_CODE_BYTES,
        ).binding,
        "registry_cli": _read_pinned_file(
            Path(__file__).resolve().parents[3]
            / "scripts"
            / "build_ccle_visible_lineage_registry.py",
            role="current lineage registry CLI",
            maximum_bytes=_MAX_CODE_BYTES,
        ).binding,
        "registry_protocol": _read_pinned_file(
            Path(__file__).resolve().parents[3]
            / "experiments"
            / "CCLE_VISIBLE_LINEAGE_REGISTRY_V1.json",
            role="current lineage registry protocol",
            maximum_bytes=_MAX_CODE_BYTES,
        ).binding,
    }
    if any(value.get(name) != binding for name, binding in current.items()):
        raise CCLEVisibleLineageRegistryError(
            f"currently executing code differs from {role} binding"
        )


def _validate_contract(contract: LineageContract) -> None:
    if (
        contract.sample_count < 2
        or contract.train_count < 1
        or contract.validation_count < 1
        or contract.train_count + contract.validation_count != contract.sample_count
        or contract.mmc2_record_count < contract.sample_count
        or contract.identity_group_count < 1
        or contract.identity_group_count > contract.sample_count
        or contract.non_singleton_identity_group_count < 0
        or contract.non_singleton_identity_group_count > contract.identity_group_count
        or contract.lineage_strata_count < 1
        or contract.lineage_strata_count > contract.sample_count
    ):
        raise CCLEVisibleLineageRegistryError("lineage contract is invalid")


def _contract_document(contract: LineageContract) -> dict[str, object]:
    return {
        "sample_count": contract.sample_count,
        "train_count": contract.train_count,
        "validation_count": contract.validation_count,
        "mmc2_record_count": contract.mmc2_record_count,
        "identity_group_count": contract.identity_group_count,
        "non_singleton_identity_group_count": (
            contract.non_singleton_identity_group_count
        ),
        "lineage_strata_count": contract.lineage_strata_count,
        "record_fields": list(ROW_FIELDS),
        "lineage_rule": LINEAGE_RULE,
        "identity_group_rule": IDENTITY_GROUP_RULE,
        "canonicalization": (
            "UTF-8 JSON list; sort_keys=True; separators=(',', ':'); "
            "ensure_ascii=False; allow_nan=False; one terminal LF; missing "
            "metadata retained as null"
        ),
    }


def _frozen_document(frozen: FrozenSources) -> dict[str, object]:
    def file_spec(value: FileSpec) -> dict[str, object]:
        return {"bytes": value.bytes, "sha256": value.sha256}

    def member_spec(value: MemberSpec) -> dict[str, object]:
        return {
            "filename": value.filename,
            "bytes": value.bytes,
            "sha256": value.sha256,
            "dtype": value.dtype,
            "shape": list(value.shape),
        }

    return {
        "mmc2": file_spec(frozen.mmc2),
        "acquisition_receipt": file_spec(frozen.acquisition_receipt),
        "visible_artifact": file_spec(frozen.visible_artifact),
        "visible_manifest": file_spec(frozen.visible_manifest),
        "visible_receipt": file_spec(frozen.visible_receipt),
        "sample_ids_member": member_spec(frozen.sample_ids_member),
        "partition_labels_member": member_spec(frozen.partition_labels_member),
        "canonical_records": file_spec(frozen.canonical_records),
    }


def _intent_access_policy() -> dict[str, object]:
    return {
        "mmc2_sheet_allowed": SHEET_NAME,
        "mmc2_columns_allowed": list(MMC2_IDENTITY_COLUMNS),
        "mmc2_column_numbers_allowed": list(MMC2_IDENTITY_COLUMN_NUMBERS),
        "visible_npz_members_allowed": list(VISIBLE_IDENTITY_MEMBERS),
        "expression_or_abundance_members_allowed": [],
        "internal_final_allowed": False,
        "model_weights_allowed": False,
        "model_invocations": 0,
        "automatic_retry": False,
    }


def _producer_access_policy() -> dict[str, object]:
    return {
        "mmc2_full_file_hashed": True,
        "mmc2_sheet_decoded": SHEET_NAME,
        "mmc2_columns_decoded": list(MMC2_IDENTITY_COLUMNS),
        "mmc2_column_numbers_decoded": list(MMC2_IDENTITY_COLUMN_NUMBERS),
        "mmc2_value_decoder": (
            "raw-byte cell-coordinate filter then selected-fragment XML decode"
        ),
        "mmc2_other_sheets_or_cells_materialized": False,
        "visible_npz_members_decoded": list(VISIBLE_IDENTITY_MEMBERS),
        "visible_archive_expression_members_opened_or_decoded": [],
        "expression_or_abundance_values_decoded": False,
        "internal_final_accessed": False,
        "model_weights_accessed": False,
        "model_invocations": 0,
        "automatic_retry": False,
    }


def _verifier_access_policy() -> dict[str, object]:
    return _producer_access_policy()


def _verified_contract_document(
    contract: LineageContract, frozen: FrozenSources
) -> dict[str, object]:
    return {
        "source_replay_independent_of_producer_payload": True,
        "sample_count": contract.sample_count,
        "train_count": contract.train_count,
        "validation_count": contract.validation_count,
        "identity_group_count": contract.identity_group_count,
        "non_singleton_identity_group_count": (
            contract.non_singleton_identity_group_count
        ),
        "lineage_strata_count": contract.lineage_strata_count,
        "record_fields_exact": list(ROW_FIELDS),
        "canonical_records_bytes": frozen.canonical_records.bytes,
        "canonical_records_sha256": frozen.canonical_records.sha256,
        "records_exact_equal": True,
    }


def _validate_source_adjacency(
    *,
    mmc2: Path,
    acquisition_receipt: Path,
    visible_artifact: Path,
    visible_manifest: Path,
    visible_receipt: Path,
) -> None:
    if acquisition_receipt != mmc2.with_name("receipt.json"):
        raise CCLEVisibleLineageRegistryError(
            "acquisition receipt is not adjacent to frozen mmc2.xlsx"
        )
    if visible_manifest != visible_artifact.with_name("manifest.json"):
        raise CCLEVisibleLineageRegistryError(
            "visible manifest is not adjacent to visible artifact"
        )
    if visible_receipt != visible_artifact.parent.parent / "receipt.json":
        raise CCLEVisibleLineageRegistryError(
            "visible receipt is not the custody root receipt"
        )


def _validate_acquisition_receipt(
    receipt_file: _PinnedFile,
    *,
    frozen: FrozenSources,
) -> None:
    receipt = _json_object(receipt_file, "mmc2 acquisition receipt")
    mmc2 = receipt.get("mmc2")
    if (
        receipt.get("schema") != "creed.source-acquisition.receipt.v1"
        or receipt.get("status") != "complete"
        or receipt.get("pmcid") != "PMC9387775"
        or receipt.get("doi") != "10.1016/j.ccell.2022.06.010"
        or receipt.get("curl_invocations") != 1
        or receipt.get("automatic_retries") != 0
        or not isinstance(mmc2, Mapping)
        or mmc2.get("name") != "mmc2.xlsx"
        or mmc2.get("bytes") != frozen.mmc2.bytes
        or mmc2.get("sha256") != frozen.mmc2.sha256
    ):
        raise CCLEVisibleLineageRegistryError(
            "mmc2 acquisition receipt contract changed"
        )


def _validate_visible_metadata(
    manifest_file: _PinnedFile,
    receipt_file: _PinnedFile,
    archive: _ArchiveIdentity,
    *,
    contract: LineageContract,
    frozen: FrozenSources,
) -> None:
    manifest = _json_object(manifest_file, "visible manifest")
    integrity = manifest.get("integrity")
    payload = dict(manifest)
    payload.pop("integrity", None)
    payload_sha = hashlib.sha256(_canonical_json_bytes(payload)[:-1]).hexdigest()
    split_record = manifest.get("split")
    artifact_record = manifest.get("artifact")
    if (
        not isinstance(integrity, Mapping)
        or integrity.get("canonical_payload_sha256") != payload_sha
        or manifest.get("schema") != "creed.ccle-visible-materialization.v1"
        or manifest.get("protocol_id") != "CREED_CCLE_L0_L1_L2_SPLIT_V1"
        or manifest.get("artifact_role") != "development_visible"
        or manifest.get("status") != "complete"
        or manifest.get("internal_final_released") is not False
        or manifest.get("scientific_claim_authority") is not False
        or not isinstance(split_record, Mapping)
        or split_record.get("visible_order")
        != "frozen train sample_ids then frozen validation sample_ids"
        or split_record.get("partition_counts")
        != {"train": contract.train_count, "validation": contract.validation_count}
        or not isinstance(artifact_record, Mapping)
        or artifact_record.get("filename") != archive.path.name
        or artifact_record.get("bytes") != archive.size
        or artifact_record.get("sha256") != frozen.visible_artifact.sha256
    ):
        raise CCLEVisibleLineageRegistryError("visible manifest contract changed")
    receipt = _json_object(receipt_file, "visible materialization receipt")
    visible = receipt.get("visible")
    sealed = receipt.get("sealed_internal_final")
    if (
        receipt.get("schema") != "creed.ccle-value-materialization-receipt.v1"
        or receipt.get("protocol_id") != "CREED_CCLE_L0_L1_L2_SPLIT_V1"
        or receipt.get("status") != "complete"
        or receipt.get("source_hashes_verified_before_value_access") is not True
        or receipt.get("model_invocations") != 0
        or receipt.get("automatic_retries") != 0
        or receipt.get("preprocessing_performed") is not False
        or receipt.get("missing_values_filled") is not False
        or receipt.get("scientific_claim_authority") is not False
        or not isinstance(visible, Mapping)
        or visible.get("sample_count") != contract.sample_count
        or visible.get("artifact_sha256") != frozen.visible_artifact.sha256
        or visible.get("manifest_sha256") != manifest_file.sha256
        or not isinstance(sealed, Mapping)
        or sealed.get("values_logged_or_reported") is not False
        or sealed.get("released") is not False
    ):
        raise CCLEVisibleLineageRegistryError(
            "visible materialization receipt contract changed"
        )


def _clean_cell(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _require_unique_optional_field(
    records: Mapping[str, CCLEIdentityRecord],
    *,
    attribute: str,
    label: str,
) -> None:
    owners: dict[str, str] = {}
    for sample_id, record in records.items():
        value = getattr(record, attribute)
        if value is None:
            continue
        prior = owners.get(value)
        if prior is not None and prior != sample_id:
            raise CCLEVisibleLineageRegistryError(f"{label} is not one-to-one")
        owners[value] = sample_id


def _xml_local_name(tag: object) -> str:
    return str(tag).rsplit("}", 1)[-1]


def _xlsx_member_bytes(
    archive: zipfile.ZipFile,
    name: str,
    *,
    maximum_bytes: int,
    role: str,
) -> bytes:
    try:
        info = archive.getinfo(name)
    except KeyError as exc:
        raise CCLEVisibleLineageRegistryError(f"XLSX is missing {role}") from exc
    if (
        info.is_dir()
        or info.flag_bits & 0x1
        or info.file_size < 1
        or info.file_size > maximum_bytes
    ):
        raise CCLEVisibleLineageRegistryError(f"XLSX {role} is unsafe")
    try:
        with archive.open(info, "r") as member:
            return _read_exact(member, info.file_size, f"XLSX {role}")
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise CCLEVisibleLineageRegistryError(f"cannot read XLSX {role}") from exc


def _safe_xlsx_target(target: str) -> str:
    if not target or "\\" in target or "\x00" in target:
        raise CCLEVisibleLineageRegistryError("XLSX sheet target is unsafe")
    if target.startswith("/"):
        candidate = posixpath.normpath(target.lstrip("/"))
    else:
        candidate = posixpath.normpath(posixpath.join("xl", target))
    if (
        candidate.startswith("../")
        or candidate == ".."
        or not candidate.startswith("xl/worksheets/")
        or not candidate.endswith(".xml")
    ):
        raise CCLEVisibleLineageRegistryError("XLSX sheet target escapes scope")
    return candidate


def _target_sheet_member(archive: zipfile.ZipFile) -> str:
    workbook_raw = _xlsx_member_bytes(
        archive,
        "xl/workbook.xml",
        maximum_bytes=_MAX_XLSX_METADATA_BYTES,
        role="workbook metadata",
    )
    relationships_raw = _xlsx_member_bytes(
        archive,
        "xl/_rels/workbook.xml.rels",
        maximum_bytes=_MAX_XLSX_METADATA_BYTES,
        role="workbook relationships",
    )
    try:
        workbook = ET.fromstring(workbook_raw)
        relationships = ET.fromstring(relationships_raw)
    except ET.ParseError as exc:
        raise CCLEVisibleLineageRegistryError(
            "XLSX workbook metadata is invalid XML"
        ) from exc
    relationship_id: str | None = None
    for element in workbook.iter():
        if (
            _xml_local_name(element.tag) == "sheet"
            and element.attrib.get("name") == SHEET_NAME
        ):
            relationship_id = element.attrib.get(
                f"{{{_WORKBOOK_RELATIONSHIP_NAMESPACE}}}id"
            )
            break
    if not relationship_id:
        raise CCLEVisibleLineageRegistryError(f"frozen mmc2 is missing {SHEET_NAME!r}")
    targets = [
        element.attrib.get("Target")
        for element in relationships
        if _xml_local_name(element.tag) == "Relationship"
        and element.attrib.get("Id") == relationship_id
        and element.attrib.get("TargetMode", "Internal") == "Internal"
    ]
    if len(targets) != 1 or not isinstance(targets[0], str):
        raise CCLEVisibleLineageRegistryError(
            "XLSX target-sheet relationship is ambiguous"
        )
    return _safe_xlsx_target(targets[0])


def _column_number(reference: str) -> tuple[int, int]:
    match = _CELL_REFERENCE_RE.fullmatch(reference)
    if match is None:
        raise CCLEVisibleLineageRegistryError(
            "XLSX target sheet has an invalid cell reference"
        )
    column = 0
    for character in match.group(1):
        column = column * 26 + ord(character) - ord("A") + 1
    return column, int(match.group(2))


def _xlsx_unescape(value: str) -> str:
    return _XLSX_ESCAPE_RE.sub(lambda match: chr(int(match.group(1), 16)), value)


def _cell_text(element: ET.Element) -> str:
    return _xlsx_unescape(
        "".join(
            node.text or ""
            for node in element.iter()
            if _xml_local_name(node.tag) == "t"
        )
    )


def _raw_xml_tag_end(raw: bytes, start: int, role: str) -> int:
    """Return the exclusive end of one raw start tag without decoding values."""

    quote: int | None = None
    for index in range(start + 1, len(raw)):
        value = raw[index]
        if quote is not None:
            if value == quote:
                quote = None
            continue
        if value in (ord('"'), ord("'")):
            quote = value
        elif value == ord(">"):
            return index + 1
    raise CCLEVisibleLineageRegistryError(f"{role} has an unterminated XML tag")


def _raw_xml_qname(start_tag: bytes, role: str) -> bytes:
    content = start_tag[1:].lstrip()
    match = re.match(rb"[A-Za-z_][A-Za-z0-9_.:-]*", content)
    if match is None:
        raise CCLEVisibleLineageRegistryError(f"{role} has an invalid XML name")
    return match.group(0)


def _raw_xml_root_wrapper(
    raw: bytes,
    *,
    expected_local_name: bytes,
    role: str,
) -> tuple[bytes, bytes]:
    """Extract only the root start tag needed to parse selected fragments."""

    cursor = 0
    while True:
        start = raw.find(b"<", cursor)
        if start < 0:
            raise CCLEVisibleLineageRegistryError(f"{role} lacks an XML root")
        if raw.startswith(b"<?", start):
            end = raw.find(b"?>", start + 2)
            if end < 0:
                raise CCLEVisibleLineageRegistryError(
                    f"{role} has an unterminated XML declaration"
                )
            cursor = end + 2
            continue
        if raw.startswith(b"<!--", start):
            end = raw.find(b"-->", start + 4)
            if end < 0:
                raise CCLEVisibleLineageRegistryError(
                    f"{role} has an unterminated XML comment"
                )
            cursor = end + 3
            continue
        if raw.startswith(b"<!", start) or raw.startswith(b"</", start):
            raise CCLEVisibleLineageRegistryError(
                f"{role} contains an unsupported XML declaration"
            )
        end = _raw_xml_tag_end(raw, start, role)
        start_tag = raw[start:end]
        qname = _raw_xml_qname(start_tag, role)
        if qname.rsplit(b":", 1)[-1] != expected_local_name:
            raise CCLEVisibleLineageRegistryError(f"{role} root element changed")
        if start_tag[:-1].rstrip().endswith(b"/"):
            raise CCLEVisibleLineageRegistryError(f"{role} root is empty")
        return start_tag, b"</" + qname + b">"


def _raw_xml_element_offsets(
    raw: bytes,
    *,
    local_name: bytes,
    role: str,
) -> Iterator[tuple[int, int, int]]:
    """Yield raw element offsets while never UTF-8/XML-decoding their bodies.

    The frozen source hash makes this a parser for one immutable OOXML object,
    not a general untrusted XML parser.  Declarations with value-bearing
    alternate syntax are rejected; unselected element bodies remain bytes.
    """

    if b"<![CDATA[" in raw or re.search(rb"<!DOCTYPE", raw, re.IGNORECASE):
        raise CCLEVisibleLineageRegistryError(
            f"{role} uses forbidden value-bearing XML declarations"
        )
    cursor = 0
    while True:
        start = raw.find(b"<", cursor)
        if start < 0:
            return
        if raw.startswith(b"<?", start):
            end = raw.find(b"?>", start + 2)
            if end < 0:
                raise CCLEVisibleLineageRegistryError(
                    f"{role} has an unterminated processing instruction"
                )
            cursor = end + 2
            continue
        if raw.startswith(b"<!--", start):
            end = raw.find(b"-->", start + 4)
            if end < 0:
                raise CCLEVisibleLineageRegistryError(
                    f"{role} has an unterminated XML comment"
                )
            cursor = end + 3
            continue
        if raw.startswith((b"</", b"<!"), start):
            cursor = _raw_xml_tag_end(raw, start, role)
            continue
        start_tag_end = _raw_xml_tag_end(raw, start, role)
        start_tag = raw[start:start_tag_end]
        qname = _raw_xml_qname(start_tag, role)
        if qname.rsplit(b":", 1)[-1] != local_name:
            cursor = start_tag_end
            continue
        if start_tag[:-1].rstrip().endswith(b"/"):
            yield start, start_tag_end, start_tag_end
            cursor = start_tag_end
            continue
        closing = re.compile(rb"</" + re.escape(qname) + rb"[\t\r\n ]*>")
        match = closing.search(raw, start_tag_end)
        if match is None:
            raise CCLEVisibleLineageRegistryError(
                f"{role} has an unterminated {local_name.decode('ascii')} element"
            )
        yield start, match.end(), start_tag_end
        cursor = match.end()


def _parse_selected_xml_fragment(
    *,
    root_start: bytes,
    root_close: bytes,
    fragment: bytes,
    expected_local_name: str,
    role: str,
) -> ET.Element:
    """Decode exactly one already-authorized XML fragment."""

    try:
        wrapper = ET.fromstring(root_start + fragment + root_close)
    except ET.ParseError as exc:
        raise CCLEVisibleLineageRegistryError(
            f"selected {role} fragment is invalid XML"
        ) from exc
    selected = [
        element
        for element in list(wrapper)
        if _xml_local_name(element.tag) == expected_local_name
    ]
    if len(selected) != 1:
        raise CCLEVisibleLineageRegistryError(
            f"selected {role} fragment changed structure"
        )
    return selected[0]


def _selected_shared_strings(raw: bytes, selected: set[int]) -> dict[int, str]:
    if not selected:
        return {}
    root_start, root_close = _raw_xml_root_wrapper(
        raw,
        expected_local_name=b"sst",
        role="XLSX shared-string table",
    )
    result: dict[int, str] = {}
    count = 0
    for index, (start, end, _start_tag_end) in enumerate(
        _raw_xml_element_offsets(
            raw,
            local_name=b"si",
            role="XLSX shared-string table",
        )
    ):
        count += 1
        if index not in selected:
            continue
        element = _parse_selected_xml_fragment(
            root_start=root_start,
            root_close=root_close,
            fragment=raw[start:end],
            expected_local_name="si",
            role="shared string",
        )
        result[index] = _cell_text(element)
    if selected and max(selected) >= count:
        raise CCLEVisibleLineageRegistryError(
            "XLSX identity cells reference absent shared strings"
        )
    if set(result) != selected:
        raise CCLEVisibleLineageRegistryError(
            "XLSX identity cells reference absent shared strings"
        )
    return result


def _target_identity_cells(
    sheet_raw: bytes,
) -> tuple[dict[tuple[int, int], tuple[str, object]], set[int]]:
    root_start, root_close = _raw_xml_root_wrapper(
        sheet_raw,
        expected_local_name=b"worksheet",
        role="XLSX target sheet",
    )
    cells: dict[tuple[int, int], tuple[str, object]] = {}
    shared_indices: set[int] = set()
    for start, end, start_tag_end in _raw_xml_element_offsets(
        sheet_raw,
        local_name=b"c",
        role="XLSX target sheet",
    ):
        # Cell coordinates are ASCII schema metadata.  The body remains raw
        # bytes unless the coordinate is one of the frozen A:E,G:L columns.
        start_tag = sheet_raw[start:start_tag_end]
        match = re.search(
            rb"(?:^|[\t\r\n ])r[\t\r\n ]*=[\t\r\n ]*(['\"])([A-Z]+[1-9][0-9]*)\1",
            start_tag,
        )
        if match is None:
            raise CCLEVisibleLineageRegistryError(
                "XLSX target-sheet cell lacks a coordinate"
            )
        try:
            reference = match.group(2).decode("ascii")
        except UnicodeError as exc:
            raise CCLEVisibleLineageRegistryError(
                "XLSX target-sheet coordinate is not ASCII"
            ) from exc
        column, row = _column_number(reference)
        if row < 2 or column not in MMC2_IDENTITY_COLUMN_NUMBERS:
            continue
        element = _parse_selected_xml_fragment(
            root_start=root_start,
            root_close=root_close,
            fragment=sheet_raw[start:end],
            expected_local_name="c",
            role="identity cell",
        )
        key = (row, column)
        if key in cells:
            raise CCLEVisibleLineageRegistryError(
                "XLSX target sheet repeats an identity cell"
            )
        if any(_xml_local_name(node.tag) == "f" for node in element):
            raise CCLEVisibleLineageRegistryError(
                "XLSX identity cells may not contain formulas"
            )
        cell_type = element.attrib.get("t", "n")
        if cell_type == "s":
            values = [node.text for node in element if _xml_local_name(node.tag) == "v"]
            if len(values) != 1 or values[0] is None:
                raise CCLEVisibleLineageRegistryError(
                    "XLSX shared-string identity cell is invalid"
                )
            try:
                shared_index = int(values[0])
            except ValueError as exc:
                raise CCLEVisibleLineageRegistryError(
                    "XLSX shared-string index is invalid"
                ) from exc
            if shared_index < 0:
                raise CCLEVisibleLineageRegistryError(
                    "XLSX shared-string index is negative"
                )
            shared_indices.add(shared_index)
            cells[key] = ("shared", shared_index)
        elif cell_type == "inlineStr":
            cells[key] = ("literal", _cell_text(element))
        elif cell_type in {"str", "n"}:
            values = [node.text for node in element if _xml_local_name(node.tag) == "v"]
            if len(values) > 1:
                raise CCLEVisibleLineageRegistryError(
                    "XLSX literal identity cell is ambiguous"
                )
            cells[key] = ("literal", values[0] if values else None)
        else:
            raise CCLEVisibleLineageRegistryError(
                "XLSX identity cell has an unsupported type"
            )
    return cells, shared_indices


def _read_mmc2_identity_records(
    source: _PinnedFile,
    *,
    contract: LineageContract,
) -> tuple[dict[str, CCLEIdentityRecord], tuple[str, ...]]:
    try:
        with zipfile.ZipFile(io.BytesIO(source.raw), "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if (
                len(infos) < 3
                or len(infos) > _MAX_XLSX_ENTRIES
                or len(names) != len(set(names))
                or any(
                    not name
                    or name.startswith(("/", "\\"))
                    or "\\" in name
                    or name == ".."
                    or name.startswith("../")
                    or "/../" in name
                    for name in names
                )
            ):
                raise CCLEVisibleLineageRegistryError(
                    "frozen mmc2 XLSX central directory is unsafe"
                )
            sheet_member = _target_sheet_member(archive)
            sheet_raw = _xlsx_member_bytes(
                archive,
                sheet_member,
                maximum_bytes=_MAX_XLSX_SHEET_BYTES,
                role="allowed identity sheet",
            )
            cells, shared_indices = _target_identity_cells(sheet_raw)
            shared: dict[int, str] = {}
            opened = [
                "xl/workbook.xml",
                "xl/_rels/workbook.xml.rels",
                sheet_member,
            ]
            if shared_indices:
                shared_raw = _xlsx_member_bytes(
                    archive,
                    "xl/sharedStrings.xml",
                    maximum_bytes=_MAX_XLSX_SHARED_STRINGS_BYTES,
                    role="selected shared strings",
                )
                shared = _selected_shared_strings(shared_raw, shared_indices)
                opened.append("xl/sharedStrings.xml:selected_indices_only")
    except CCLEVisibleLineageRegistryError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise CCLEVisibleLineageRegistryError(
            "frozen mmc2 is not a safe XLSX workbook"
        ) from exc

    def value(row: int, column: int) -> str | None:
        token = cells.get((row, column))
        if token is None:
            return None
        kind, payload = token
        if kind == "shared":
            return _clean_cell(shared[int(payload)])
        return _clean_cell(payload)

    headers = tuple(value(2, column) for column in MMC2_IDENTITY_COLUMN_NUMBERS)
    if headers != MMC2_IDENTITY_COLUMNS:
        raise CCLEVisibleLineageRegistryError(
            "frozen mmc2 identity-column order changed"
        )
    rows = sorted({row for row, column in cells if row >= 3 and column == 1})
    records: dict[str, CCLEIdentityRecord] = {}
    for row in rows:
        values = {
            name: value(row, column)
            for name, column in zip(
                MMC2_IDENTITY_COLUMNS, MMC2_IDENTITY_COLUMN_NUMBERS, strict=True
            )
        }
        model_id = values["model_id"]
        if model_id is None:
            continue
        if _SIDM_RE.fullmatch(model_id) is None:
            raise CCLEVisibleLineageRegistryError(
                "frozen mmc2 contains an invalid model_id"
            )
        if model_id in records:
            raise CCLEVisibleLineageRegistryError(
                "frozen mmc2 contains a duplicate model_id"
            )
        records[model_id] = CCLEIdentityRecord(
            model_id=model_id,
            project_identifier=values["Project_Identifier"],
            cell_line=values["Cell_line"],
            source=values["Source"],
            identifier=values["Identifier"],
            tissue_type=values["Tissue_type"],
            cancer_type=values["Cancer_type"],
            cancer_subtype=values["Cancer_subtype"],
            haem_lineage=values["Haem_lineage"],
            broad_id=values["BROAD_ID"],
            ccle_id=values["CCLE_ID"],
        )
    if len(records) != contract.mmc2_record_count:
        raise CCLEVisibleLineageRegistryError(
            "frozen mmc2 identity-record count changed"
        )
    _require_unique_optional_field(records, attribute="broad_id", label="mmc2 BROAD_ID")
    _require_unique_optional_field(records, attribute="ccle_id", label="mmc2 CCLE_ID")
    return records, tuple(opened)


def _read_exact(member: BinaryIO, count: int, role: str) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = member.read(min(1_048_576, remaining))
        if not chunk:
            raise CCLEVisibleLineageRegistryError(f"short read from {role}")
        chunks.append(chunk)
        remaining -= len(chunk)
    if member.read(1):
        raise CCLEVisibleLineageRegistryError(f"{role} member grew")
    return b"".join(chunks)


def _decode_identity_member(
    archive: zipfile.ZipFile,
    spec: MemberSpec,
) -> np.ndarray:
    try:
        info = archive.getinfo(spec.filename)
    except KeyError as exc:
        raise CCLEVisibleLineageRegistryError(
            f"visible archive is missing {spec.filename}"
        ) from exc
    if (
        info.file_size != spec.bytes
        or info.file_size < 1
        or info.file_size > 4 * 1024 * 1024
        or info.flag_bits & 0x1
        or info.is_dir()
    ):
        raise CCLEVisibleLineageRegistryError(
            f"visible {spec.filename} member metadata changed"
        )
    try:
        with archive.open(info, "r") as member:
            raw = _read_exact(member, info.file_size, spec.filename)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise CCLEVisibleLineageRegistryError(
            f"cannot read visible identity member {spec.filename}"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != spec.sha256:
        raise CCLEVisibleLineageRegistryError(
            f"visible {spec.filename} member identity changed"
        )
    try:
        value = np.load(io.BytesIO(raw), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise CCLEVisibleLineageRegistryError(
            f"visible {spec.filename} is not a safe NPY"
        ) from exc
    if (
        not isinstance(value, np.ndarray)
        or value.dtype.hasobject
        or value.dtype != np.dtype(spec.dtype)
        or value.shape != spec.shape
        or not value.flags.c_contiguous
    ):
        raise CCLEVisibleLineageRegistryError(
            f"visible {spec.filename} array contract changed"
        )
    return np.asarray(value).copy()


def _read_visible_identity_members(
    path: Path,
    *,
    frozen: FrozenSources,
) -> tuple[np.ndarray, np.ndarray, _ArchiveIdentity]:
    source = _absolute(path, "visible identity archive")
    _reject_forbidden_path(source, "visible identity archive")
    _reject_symlink_components(source, "visible identity archive", include_leaf=True)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise CCLEVisibleLineageRegistryError(
            "cannot open visible identity archive"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size != frozen.visible_artifact.bytes
            or before.st_size > _MAX_VISIBLE_BYTES
        ):
            raise CCLEVisibleLineageRegistryError(
                "visible identity archive size/type changed"
            )
        try:
            with os.fdopen(os.dup(descriptor), "rb") as handle:
                with zipfile.ZipFile(handle, "r") as archive:
                    infos = archive.infolist()
                    names = [info.filename for info in infos]
                    if (
                        len(infos) < len(VISIBLE_IDENTITY_MEMBERS)
                        or len(infos) > _MAX_ZIP_ENTRIES
                        or len(names) != len(set(names))
                        or any(
                            not name
                            or name.startswith(("/", "\\"))
                            or "/" in name
                            or "\\" in name
                            or name in {".", ".."}
                            for name in names
                        )
                    ):
                        raise CCLEVisibleLineageRegistryError(
                            "visible NPZ central directory is unsafe"
                        )
                    sample_ids = _decode_identity_member(
                        archive, frozen.sample_ids_member
                    )
                    partition_labels = _decode_identity_member(
                        archive, frozen.partition_labels_member
                    )
        except CCLEVisibleLineageRegistryError:
            raise
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            raise CCLEVisibleLineageRegistryError(
                "visible artifact is not a safe frozen NPZ"
            ) from exc
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _descriptor_identity(before) != _descriptor_identity(after):
        raise CCLEVisibleLineageRegistryError(
            "visible identity archive changed during member reads"
        )
    path_after = os.lstat(source)
    if stat.S_ISLNK(path_after.st_mode) or _path_identity(path_after) != _path_identity(
        after
    ):
        raise CCLEVisibleLineageRegistryError(
            "visible identity archive path identity changed"
        )
    identity = _ArchiveIdentity(
        path=source,
        dev=before.st_dev,
        ino=before.st_ino,
        mode=stat.S_IMODE(before.st_mode),
        size=before.st_size,
        mtime_ns=before.st_mtime_ns,
        ctime_ns=before.st_ctime_ns,
    )
    return sample_ids, partition_labels, identity


def _records_document(records: Sequence[LineageRecord]) -> list[dict[str, object]]:
    return [record.document for record in records]


def _validate_records(
    records: Sequence[LineageRecord],
    *,
    contract: LineageContract,
    frozen: FrozenSources,
) -> bytes:
    if len(records) != contract.sample_count:
        raise CCLEVisibleLineageRegistryError("lineage sample count changed")
    sample_ids = tuple(record.sample_id for record in records)
    partitions = tuple(record.partition for record in records)
    identity_groups = tuple(record.identity_group for record in records)
    strata = tuple(record.lineage_stratum for record in records)
    expected_partitions = ("train",) * contract.train_count + (
        "validation",
    ) * contract.validation_count
    if (
        len(set(sample_ids)) != contract.sample_count
        or any(_SIDM_RE.fullmatch(value) is None for value in sample_ids)
        or partitions != expected_partitions
        or any(not value or value.strip() != value for value in identity_groups)
        or len(set(identity_groups)) != contract.identity_group_count
        or any(not value or value.strip() != value for value in strata)
        or len(set(strata)) != contract.lineage_strata_count
    ):
        raise CCLEVisibleLineageRegistryError(
            "lineage records violate sample/partition/group/stratum contract"
        )
    for record in records:
        if tuple(record.document) != ROW_FIELDS:
            raise CCLEVisibleLineageRegistryError("lineage record fields changed")
        for value in (
            record.tissue_type,
            record.cancer_type,
            record.cancer_subtype,
            record.haem_lineage,
        ):
            if value is not None and (not value or value.strip() != value):
                raise CCLEVisibleLineageRegistryError(
                    "lineage metadata is not clean optional text"
                )
    raw = _canonical_json_bytes(_records_document(records))
    if (
        len(raw) != frozen.canonical_records.bytes
        or hashlib.sha256(raw).hexdigest() != frozen.canonical_records.sha256
    ):
        raise CCLEVisibleLineageRegistryError(
            "canonical lineage payload differs from the frozen identity"
        )
    return raw


def _build_records(
    *,
    identities: Mapping[str, CCLEIdentityRecord],
    sample_ids_raw: np.ndarray,
    partition_labels_raw: np.ndarray,
    contract: LineageContract,
    frozen: FrozenSources,
) -> tuple[tuple[LineageRecord, ...], list[dict[str, object]]]:
    sample_ids = tuple(str(value) for value in sample_ids_raw.tolist())
    partitions = tuple(str(value) for value in partition_labels_raw.tolist())
    if len(sample_ids) != contract.sample_count or len(partitions) != len(sample_ids):
        raise CCLEVisibleLineageRegistryError(
            "visible identity arrays have an invalid sample count"
        )
    missing = [sample_id for sample_id in sample_ids if sample_id not in identities]
    if missing:
        raise CCLEVisibleLineageRegistryError(
            "visible SIDMs are absent from frozen mmc2 identity metadata"
        )
    try:
        identity_groups, non_singleton = infer_identity_groups(identities, sample_ids)
        records = tuple(
            LineageRecord(
                sample_id=sample_id,
                partition=partition,
                identity_group=identity_groups[sample_id],
                tissue_type=identities[sample_id].tissue_type,
                cancer_type=identities[sample_id].cancer_type,
                cancer_subtype=identities[sample_id].cancer_subtype,
                haem_lineage=identities[sample_id].haem_lineage,
                lineage_stratum=lineage_stratum(identities[sample_id]),
            )
            for sample_id, partition in zip(sample_ids, partitions, strict=True)
        )
    except CCLEContractError as exc:
        raise CCLEVisibleLineageRegistryError(
            "CCLE identity grouping or lineage derivation failed"
        ) from exc
    if len(non_singleton) != contract.non_singleton_identity_group_count:
        raise CCLEVisibleLineageRegistryError(
            "non-singleton identity-group count changed"
        )
    _validate_records(records, contract=contract, frozen=frozen)
    return records, non_singleton


def _replay_sources(
    *,
    mmc2: str | Path,
    acquisition_receipt: str | Path,
    visible_artifact: str | Path,
    visible_manifest: str | Path,
    visible_receipt: str | Path,
    contract: LineageContract,
    frozen: FrozenSources,
) -> tuple[tuple[LineageRecord, ...], dict[str, object]]:
    mmc2_path = _absolute(mmc2, "frozen mmc2 identity workbook")
    acquisition_path = _absolute(acquisition_receipt, "mmc2 acquisition receipt")
    visible_path = _absolute(visible_artifact, "visible identity archive")
    manifest_path = _absolute(visible_manifest, "visible manifest")
    visible_receipt_path = _absolute(visible_receipt, "visible materialization receipt")
    for source, role in (
        (mmc2_path, "frozen mmc2 identity workbook"),
        (acquisition_path, "mmc2 acquisition receipt"),
        (visible_path, "visible identity archive"),
        (manifest_path, "visible manifest"),
        (visible_receipt_path, "visible materialization receipt"),
    ):
        _reject_forbidden_path(source, role)
    _validate_source_adjacency(
        mmc2=mmc2_path,
        acquisition_receipt=acquisition_path,
        visible_artifact=visible_path,
        visible_manifest=manifest_path,
        visible_receipt=visible_receipt_path,
    )
    mmc2_file = _read_pinned_file(
        mmc2_path,
        role="frozen mmc2 identity workbook",
        maximum_bytes=_MAX_MMC2_BYTES,
        expected=frozen.mmc2,
        reject_final=True,
    )
    acquisition_file = _read_pinned_file(
        acquisition_path,
        role="mmc2 acquisition receipt",
        maximum_bytes=_MAX_JSON_BYTES,
        expected=frozen.acquisition_receipt,
        reject_final=True,
    )
    _validate_acquisition_receipt(acquisition_file, frozen=frozen)
    manifest_file = _read_pinned_file(
        manifest_path,
        role="visible manifest",
        maximum_bytes=_MAX_JSON_BYTES,
        expected=frozen.visible_manifest,
        reject_final=True,
    )
    visible_receipt_file = _read_pinned_file(
        visible_receipt_path,
        role="visible materialization receipt",
        maximum_bytes=_MAX_JSON_BYTES,
        expected=frozen.visible_receipt,
        reject_final=True,
    )
    sample_ids, partitions, archive = _read_visible_identity_members(
        visible_path, frozen=frozen
    )
    _validate_visible_metadata(
        manifest_file,
        visible_receipt_file,
        archive,
        contract=contract,
        frozen=frozen,
    )
    identities, xlsx_members_opened = _read_mmc2_identity_records(
        mmc2_file, contract=contract
    )
    records, non_singleton = _build_records(
        identities=identities,
        sample_ids_raw=sample_ids,
        partition_labels_raw=partitions,
        contract=contract,
        frozen=frozen,
    )
    bindings = {
        "mmc2": {
            **mmc2_file.binding,
            "xlsx_members_opened": list(xlsx_members_opened),
            "other_sheet_xml_opened": False,
            "shared_strings_materialized_only_for_allowed_cells": True,
            "value_decoder": (
                "raw-byte cell-coordinate filter then selected-fragment XML decode"
            ),
        },
        "acquisition_receipt": acquisition_file.binding,
        "visible_manifest": manifest_file.binding,
        "visible_receipt": visible_receipt_file.binding,
        "visible_artifact": {
            **archive.binding,
            "declared_sha256": frozen.visible_artifact.sha256,
            "full_archive_sha256_recomputed": False,
            "reason_full_archive_not_read": (
                "access contract permits decoding only two identity members"
            ),
        },
        "visible_identity_members": {
            frozen.sample_ids_member.filename: {
                "bytes": frozen.sample_ids_member.bytes,
                "sha256": frozen.sample_ids_member.sha256,
            },
            frozen.partition_labels_member.filename: {
                "bytes": frozen.partition_labels_member.bytes,
                "sha256": frozen.partition_labels_member.sha256,
            },
        },
        "non_singleton_identity_groups": non_singleton,
    }
    return records, bindings


def _validate_file_binding_document(
    value: object,
    *,
    expected: FileSpec,
    role: str,
    extra_keys: frozenset[str] = frozenset(),
) -> Mapping[str, object]:
    required = {"path", "dev", "ino", "mode", "bytes", "sha256"}
    if not isinstance(value, Mapping) or set(value) != required | set(extra_keys):
        raise CCLEVisibleLineageRegistryError(f"{role} binding schema changed")
    path_raw = value.get("path")
    if not isinstance(path_raw, str):
        raise CCLEVisibleLineageRegistryError(f"{role} binding path changed")
    path = _absolute(Path(path_raw), f"{role} binding path")
    _reject_forbidden_path(path, f"{role} binding path")
    if (
        type(value.get("dev")) is not int
        or type(value.get("ino")) is not int
        or type(value.get("mode")) is not int
        or int(value["dev"]) < 0
        or int(value["ino"]) < 0
        or int(value["mode"]) < 0
        or value.get("bytes") != expected.bytes
        or value.get("sha256") != expected.sha256
    ):
        raise CCLEVisibleLineageRegistryError(f"{role} frozen binding changed")
    return value


def _validate_source_bindings_document(
    value: object,
    *,
    contract: LineageContract,
    frozen: FrozenSources,
) -> Mapping[str, object]:
    expected_keys = {
        "mmc2",
        "acquisition_receipt",
        "visible_manifest",
        "visible_receipt",
        "visible_artifact",
        "visible_identity_members",
        "non_singleton_identity_groups",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise CCLEVisibleLineageRegistryError("source-binding registry changed")
    mmc2 = _validate_file_binding_document(
        value.get("mmc2"),
        expected=frozen.mmc2,
        role="mmc2 source",
        extra_keys=frozenset(
            {
                "xlsx_members_opened",
                "other_sheet_xml_opened",
                "shared_strings_materialized_only_for_allowed_cells",
                "value_decoder",
            }
        ),
    )
    opened = mmc2.get("xlsx_members_opened")
    if (
        not isinstance(opened, list)
        or len(opened) not in {3, 4}
        or opened[:2] != ["xl/workbook.xml", "xl/_rels/workbook.xml.rels"]
        or not isinstance(opened[2], str)
        or not opened[2].startswith("xl/worksheets/")
        or not opened[2].endswith(".xml")
        or (
            len(opened) == 4
            and opened[3] != "xl/sharedStrings.xml:selected_indices_only"
        )
        or mmc2.get("other_sheet_xml_opened") is not False
        or mmc2.get("shared_strings_materialized_only_for_allowed_cells") is not True
        or mmc2.get("value_decoder")
        != "raw-byte cell-coordinate filter then selected-fragment XML decode"
    ):
        raise CCLEVisibleLineageRegistryError(
            "mmc2 targeted-XLSX access binding changed"
        )
    acquisition = _validate_file_binding_document(
        value.get("acquisition_receipt"),
        expected=frozen.acquisition_receipt,
        role="acquisition receipt source",
    )
    manifest = _validate_file_binding_document(
        value.get("visible_manifest"),
        expected=frozen.visible_manifest,
        role="visible manifest source",
    )
    receipt = _validate_file_binding_document(
        value.get("visible_receipt"),
        expected=frozen.visible_receipt,
        role="visible receipt source",
    )
    archive = value.get("visible_artifact")
    archive_keys = {
        "path",
        "dev",
        "ino",
        "mode",
        "bytes",
        "declared_sha256",
        "full_archive_sha256_recomputed",
        "reason_full_archive_not_read",
    }
    if not isinstance(archive, Mapping) or set(archive) != archive_keys:
        raise CCLEVisibleLineageRegistryError(
            "visible archive source binding schema changed"
        )
    archive_path_raw = archive.get("path")
    if not isinstance(archive_path_raw, str):
        raise CCLEVisibleLineageRegistryError("visible archive source path changed")
    archive_path = _absolute(Path(archive_path_raw), "visible archive binding path")
    _reject_forbidden_path(archive_path, "visible archive binding path")
    if (
        type(archive.get("dev")) is not int
        or type(archive.get("ino")) is not int
        or type(archive.get("mode")) is not int
        or int(archive["dev"]) < 0
        or int(archive["ino"]) < 0
        or int(archive["mode"]) < 0
        or archive.get("bytes") != frozen.visible_artifact.bytes
        or archive.get("declared_sha256") != frozen.visible_artifact.sha256
        or archive.get("full_archive_sha256_recomputed") is not False
        or archive.get("reason_full_archive_not_read")
        != "access contract permits decoding only two identity members"
    ):
        raise CCLEVisibleLineageRegistryError("visible archive frozen binding changed")
    member_bindings = value.get("visible_identity_members")
    expected_members = {
        frozen.sample_ids_member.filename: {
            "bytes": frozen.sample_ids_member.bytes,
            "sha256": frozen.sample_ids_member.sha256,
        },
        frozen.partition_labels_member.filename: {
            "bytes": frozen.partition_labels_member.bytes,
            "sha256": frozen.partition_labels_member.sha256,
        },
    }
    non_singleton = value.get("non_singleton_identity_groups")
    if (
        member_bindings != expected_members
        or not isinstance(non_singleton, list)
        or len(non_singleton) != contract.non_singleton_identity_group_count
    ):
        raise CCLEVisibleLineageRegistryError(
            "visible member/group source binding changed"
        )
    _validate_source_adjacency(
        mmc2=Path(str(mmc2["path"])),
        acquisition_receipt=Path(str(acquisition["path"])),
        visible_artifact=archive_path,
        visible_manifest=Path(str(manifest["path"])),
        visible_receipt=Path(str(receipt["path"])),
    )
    return value


def _produce_visible_lineage_registry(
    *,
    mmc2: str | Path,
    acquisition_receipt: str | Path,
    visible_artifact: str | Path,
    visible_manifest: str | Path,
    visible_receipt: str | Path,
    output_root: str | Path,
    code_paths: Mapping[str, str | Path],
    contract: LineageContract = PRODUCTION_CONTRACT,
    frozen: FrozenSources = PRODUCTION_FROZEN_SOURCES,
) -> dict[str, object]:
    _validate_contract(contract)
    output = _absolute(output_root, "lineage registry output root")
    completion = completion_receipt_path(output)
    verifier_attempt = verifier_attempt_root_path(output)
    _reject_forbidden_path(output, "lineage registry output root")
    _reject_symlink_components(
        output, "lineage registry output root", include_leaf=False
    )
    if (
        os.path.lexists(output)
        or os.path.lexists(completion)
        or os.path.lexists(verifier_attempt)
    ):
        raise CCLEVisibleLineageRegistryError(
            "create-once lineage registry output/verifier/completion already exists"
        )
    created = False
    stage = "create_producer_root"
    try:
        output.mkdir(mode=0o700, parents=False, exist_ok=False)
        created = True
        stage = "fsync_producer_parent"
        _fsync_directory(output.parent)
        stage = "write_producer_intent"
        intent = {
            "schema": INTENT_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "status": "started_identity_lineage_only_non_authorizing",
            "output_root": str(output),
            "lineage_contract": _contract_document(contract),
            "frozen_sources": _frozen_document(frozen),
            "access_policy": _intent_access_policy(),
            "scientific_claim_authority": False,
        }
        _exclusive_write(
            output / "intent.json", _canonical_json_bytes(intent), mode=0o444
        )
        stage = "replay_frozen_identity_sources"
        records, source_bindings = _replay_sources(
            mmc2=mmc2,
            acquisition_receipt=acquisition_receipt,
            visible_artifact=visible_artifact,
            visible_manifest=visible_manifest,
            visible_receipt=visible_receipt,
            contract=contract,
            frozen=frozen,
        )
        _validate_source_bindings_document(
            source_bindings, contract=contract, frozen=frozen
        )
        stage = "write_canonical_lineage_records"
        records_raw = _validate_records(records, contract=contract, frozen=frozen)
        _exclusive_write(output / "lineage_records.json", records_raw, mode=0o444)
        stage = "bind_producer_code_and_protocol"
        code_bindings = _bind_code_files(code_paths)
        intent_file = _read_pinned_file(
            output / "intent.json",
            role="lineage producer intent",
            maximum_bytes=_MAX_JSON_BYTES,
        )
        records_file = _read_pinned_file(
            output / "lineage_records.json",
            role="canonical lineage records",
            maximum_bytes=_MAX_JSON_BYTES,
            expected=frozen.canonical_records,
        )
        receipt = {
            "schema": PRODUCER_SCHEMA,
            "records_schema": REGISTRY_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "status": "complete_identity_lineage_only_non_authorizing",
            "verification_complete": False,
            "scientific_claim_authority": False,
            "sample_count": contract.sample_count,
            "partition_counts": {
                "train": contract.train_count,
                "validation": contract.validation_count,
            },
            "identity_group_count": contract.identity_group_count,
            "lineage_strata_count": contract.lineage_strata_count,
            "producer_outputs": {
                "intent.json": intent_file.binding,
                "lineage_records.json": records_file.binding,
            },
            "source_bindings": source_bindings,
            "code_bindings": code_bindings,
            "access_policy": _producer_access_policy(),
        }
        stage = "write_producer_receipt"
        _exclusive_write(
            output / "receipt.json", _canonical_json_bytes(receipt), mode=0o444
        )
        stage = "validate_producer_file_set"
        if {child.name for child in output.iterdir()} != {
            "intent.json",
            "lineage_records.json",
            "receipt.json",
        }:
            raise CCLEVisibleLineageRegistryError(
                "producer lineage registry contains unexpected files"
            )
        stage = "seal_producer_root"
        _seal_root(output)
        # A posterior consumer pass binds the sealed inode, modes, file set,
        # canonical payload, receipt, and currently executing producer code.
        stage = "posterior_verify_producer_root"
        posterior = _load_producer_registry(output, contract=contract, frozen=frozen)
        return {
            "status": "complete_identity_lineage_only_non_authorizing",
            "registry_root": str(output),
            "completion_receipt": str(completion),
            "sample_count": len(posterior.records),
            "verification_complete": False,
            "scientific_claim_authority": False,
        }
    except BaseException:
        if created:
            _preserve_failed_attempt(
                output,
                schema=PRODUCER_SCHEMA,
                stage=stage,
                status="failed_identity_lineage_only_non_authorizing",
            )
        raise


def _lineage_record_from_document(value: object) -> LineageRecord:
    if not isinstance(value, Mapping) or tuple(sorted(value)) != tuple(
        sorted(ROW_FIELDS)
    ):
        raise CCLEVisibleLineageRegistryError("lineage record schema changed")
    required = (
        "sample_id",
        "partition",
        "identity_group",
        "lineage_stratum",
    )
    if any(not isinstance(value.get(name), str) for name in required):
        raise CCLEVisibleLineageRegistryError("lineage record required text changed")
    optional = ("tissue_type", "cancer_type", "cancer_subtype", "haem_lineage")
    if any(
        field is not None and not isinstance(field, str)
        for field in (value.get(name) for name in optional)
    ):
        raise CCLEVisibleLineageRegistryError(
            "lineage record optional metadata changed"
        )
    return LineageRecord(
        sample_id=str(value["sample_id"]),
        partition=str(value["partition"]),
        identity_group=str(value["identity_group"]),
        tissue_type=value.get("tissue_type"),  # type: ignore[arg-type]
        cancer_type=value.get("cancer_type"),  # type: ignore[arg-type]
        cancer_subtype=value.get("cancer_subtype"),  # type: ignore[arg-type]
        haem_lineage=value.get("haem_lineage"),  # type: ignore[arg-type]
        lineage_stratum=str(value["lineage_stratum"]),
    )


def _load_producer_registry(
    root: Path,
    *,
    contract: LineageContract,
    frozen: FrozenSources,
) -> VerifiedVisibleLineageRegistry:
    _reject_symlink_components(root, "lineage registry root", include_leaf=True)
    try:
        observed = os.lstat(root)
    except OSError as exc:
        raise CCLEVisibleLineageRegistryError(
            "lineage registry root is absent"
        ) from exc
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
        raise CCLEVisibleLineageRegistryError("lineage registry root is unsafe")
    if os.name != "nt" and stat.S_IMODE(observed.st_mode) != 0o555:
        raise CCLEVisibleLineageRegistryError(
            "lineage registry root is not permission sealed"
        )
    children = tuple(root.iterdir())
    expected_names = {"intent.json", "lineage_records.json", "receipt.json"}
    if {child.name for child in children} != expected_names or any(
        child.is_symlink() or not child.is_file() for child in children
    ):
        raise CCLEVisibleLineageRegistryError(
            "lineage registry producer file set changed"
        )
    pins = {
        name: _read_pinned_file(
            root / name,
            role=f"lineage registry {name}",
            maximum_bytes=_MAX_JSON_BYTES,
            required_mode=0o444,
            expected=(
                frozen.canonical_records if name == "lineage_records.json" else None
            ),
        )
        for name in sorted(expected_names)
    }
    intent = _json_object(pins["intent.json"], "lineage registry intent")
    receipt = _json_object(pins["receipt.json"], "lineage producer receipt")
    try:
        records_document = json.loads(pins["lineage_records.json"].raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CCLEVisibleLineageRegistryError(
            "canonical lineage records are invalid JSON"
        ) from exc
    if not isinstance(records_document, list):
        raise CCLEVisibleLineageRegistryError(
            "canonical lineage payload is not a JSON list"
        )
    records = tuple(_lineage_record_from_document(value) for value in records_document)
    _validate_records(records, contract=contract, frozen=frozen)
    if (
        set(intent)
        != {
            "schema",
            "protocol_id",
            "status",
            "output_root",
            "lineage_contract",
            "frozen_sources",
            "access_policy",
            "scientific_claim_authority",
        }
        or intent.get("schema") != INTENT_SCHEMA
        or intent.get("protocol_id") != PROTOCOL_ID
        or intent.get("status") != "started_identity_lineage_only_non_authorizing"
        or intent.get("scientific_claim_authority") is not False
        or intent.get("output_root") != str(root)
        or intent.get("lineage_contract") != _contract_document(contract)
        or intent.get("frozen_sources") != _frozen_document(frozen)
        or intent.get("access_policy") != _intent_access_policy()
    ):
        raise CCLEVisibleLineageRegistryError("lineage registry intent changed")
    outputs = receipt.get("producer_outputs")
    access = receipt.get("access_policy")
    code_bindings = receipt.get("code_bindings")
    source_bindings = receipt.get("source_bindings")
    if (
        set(receipt)
        != {
            "schema",
            "records_schema",
            "protocol_id",
            "status",
            "verification_complete",
            "scientific_claim_authority",
            "sample_count",
            "partition_counts",
            "identity_group_count",
            "lineage_strata_count",
            "producer_outputs",
            "source_bindings",
            "code_bindings",
            "access_policy",
        }
        or receipt.get("schema") != PRODUCER_SCHEMA
        or receipt.get("records_schema") != REGISTRY_SCHEMA
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("status") != "complete_identity_lineage_only_non_authorizing"
        or receipt.get("verification_complete") is not False
        or receipt.get("scientific_claim_authority") is not False
        or receipt.get("sample_count") != contract.sample_count
        or receipt.get("partition_counts")
        != {"train": contract.train_count, "validation": contract.validation_count}
        or receipt.get("identity_group_count") != contract.identity_group_count
        or receipt.get("lineage_strata_count") != contract.lineage_strata_count
        or outputs
        != {
            "intent.json": pins["intent.json"].binding,
            "lineage_records.json": pins["lineage_records.json"].binding,
        }
        or access != _producer_access_policy()
    ):
        raise CCLEVisibleLineageRegistryError(
            "lineage registry producer receipt changed"
        )
    source_bindings = _validate_source_bindings_document(
        source_bindings, contract=contract, frozen=frozen
    )
    _require_current_code_binding(code_bindings, "producer")
    root_after = os.lstat(root)
    if _path_identity(root_after) != _path_identity(observed):
        raise CCLEVisibleLineageRegistryError(
            "lineage registry root changed during posterior read"
        )
    bindings = {
        "registry_root": {
            "path": str(root),
            "dev": observed.st_dev,
            "ino": observed.st_ino,
            "mode": stat.S_IMODE(observed.st_mode),
        },
        **{name: pin.binding for name, pin in pins.items()},
        "source_bindings": source_bindings,
        "producer_code_bindings": code_bindings,
    }
    return VerifiedVisibleLineageRegistry(records=records, bindings=bindings)


def _load_verifier_attempt(
    attempt: Path,
    *,
    producer: VerifiedVisibleLineageRegistry,
    contract: LineageContract,
    frozen: FrozenSources,
) -> dict[str, object]:
    _reject_symlink_components(
        attempt, "lineage verifier attempt root", include_leaf=True
    )
    try:
        observed = os.lstat(attempt)
    except OSError as exc:
        raise CCLEVisibleLineageRegistryError(
            "lineage verifier attempt root is absent"
        ) from exc
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
        raise CCLEVisibleLineageRegistryError("lineage verifier attempt root is unsafe")
    if os.name != "nt" and stat.S_IMODE(observed.st_mode) != 0o555:
        raise CCLEVisibleLineageRegistryError(
            "lineage verifier attempt root is not permission sealed"
        )
    expected_names = {"intent.json", "receipt.json"}
    children = tuple(attempt.iterdir())
    if {child.name for child in children} != expected_names or any(
        child.is_symlink() or not child.is_file() for child in children
    ):
        raise CCLEVisibleLineageRegistryError(
            "lineage verifier attempt file set changed"
        )
    pins = {
        name: _read_pinned_file(
            attempt / name,
            role=f"lineage verifier attempt {name}",
            maximum_bytes=_MAX_JSON_BYTES,
            required_mode=0o444,
        )
        for name in sorted(expected_names)
    }
    intent = _json_object(pins["intent.json"], "lineage verifier intent")
    receipt = _json_object(pins["receipt.json"], "lineage verifier receipt")
    producer_files = {
        name: producer.bindings[name]
        for name in ("intent.json", "lineage_records.json", "receipt.json")
    }
    if (
        set(intent)
        != {
            "schema",
            "protocol_id",
            "status",
            "scientific_claim_authority",
            "registry_root",
            "producer_files",
            "lineage_contract",
            "frozen_sources",
            "access_policy",
        }
        or intent.get("schema") != VERIFIER_INTENT_SCHEMA
        or intent.get("protocol_id") != PROTOCOL_ID
        or intent.get("status")
        != "started_independent_identity_lineage_verification_non_authorizing"
        or intent.get("scientific_claim_authority") is not False
        or intent.get("registry_root") != producer.bindings.get("registry_root")
        or intent.get("producer_files") != producer_files
        or intent.get("lineage_contract") != _contract_document(contract)
        or intent.get("frozen_sources") != _frozen_document(frozen)
        or intent.get("access_policy") != _intent_access_policy()
    ):
        raise CCLEVisibleLineageRegistryError("lineage verifier attempt intent changed")
    source_bindings = _validate_source_bindings_document(
        receipt.get("independent_source_bindings"),
        contract=contract,
        frozen=frozen,
    )
    if source_bindings != producer.bindings.get("source_bindings"):
        raise CCLEVisibleLineageRegistryError(
            "lineage verifier attempt source differs from producer"
        )
    code_bindings = receipt.get("code_bindings")
    if (
        set(receipt)
        != {
            "schema",
            "protocol_id",
            "status",
            "scientific_claim_authority",
            "producer_files",
            "verifier_outputs",
            "independent_source_bindings",
            "verified_contract",
            "code_bindings",
            "access_policy",
        }
        or receipt.get("schema") != VERIFIER_ATTEMPT_SCHEMA
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("status")
        != "complete_independent_identity_lineage_verification_non_authorizing"
        or receipt.get("scientific_claim_authority") is not False
        or receipt.get("producer_files") != producer_files
        or receipt.get("verifier_outputs")
        != {"intent.json": pins["intent.json"].binding}
        or receipt.get("verified_contract")
        != _verified_contract_document(contract, frozen)
        or receipt.get("access_policy") != _verifier_access_policy()
    ):
        raise CCLEVisibleLineageRegistryError(
            "lineage verifier attempt receipt changed"
        )
    _require_current_code_binding(code_bindings, "verifier attempt")
    root_after = os.lstat(attempt)
    if _path_identity(root_after) != _path_identity(observed):
        raise CCLEVisibleLineageRegistryError(
            "lineage verifier attempt changed during posterior read"
        )
    return {
        "root": {
            "path": str(attempt),
            "dev": observed.st_dev,
            "ino": observed.st_ino,
            "mode": stat.S_IMODE(observed.st_mode),
        },
        "intent.json": pins["intent.json"].binding,
        "receipt.json": pins["receipt.json"].binding,
        "source_bindings": source_bindings,
        "code_bindings": code_bindings,
    }


def _verify_visible_lineage_registry(
    *,
    registry_root: str | Path,
    mmc2: str | Path,
    acquisition_receipt: str | Path,
    visible_artifact: str | Path,
    visible_manifest: str | Path,
    visible_receipt: str | Path,
    code_paths: Mapping[str, str | Path],
    contract: LineageContract = PRODUCTION_CONTRACT,
    frozen: FrozenSources = PRODUCTION_FROZEN_SOURCES,
) -> dict[str, object]:
    _validate_contract(contract)
    root = _absolute(registry_root, "lineage registry root")
    completion = completion_receipt_path(root)
    attempt = verifier_attempt_root_path(root)
    _reject_forbidden_path(root, "lineage registry root")
    _reject_symlink_components(
        completion, "adjacent lineage completion receipt", include_leaf=False
    )
    if os.path.lexists(completion) or os.path.lexists(attempt):
        raise CCLEVisibleLineageRegistryError(
            "create-once lineage verifier attempt/completion already exists"
        )
    created = False
    stage = "create_verifier_attempt_root"
    try:
        attempt.mkdir(mode=0o700, parents=False, exist_ok=False)
        created = True
        stage = "fsync_verifier_attempt_parent"
        _fsync_directory(attempt.parent)
        # Loading the sealed producer is part of the single verifier attempt,
        # not a retryable preflight.  Once the adjacent root exists, every
        # producer-read, source-replay, or publication failure is preserved in
        # that create-once root and permanently consumes this verifier attempt.
        stage = "posterior_load_sealed_producer"
        producer = _load_producer_registry(root, contract=contract, frozen=frozen)
        producer_files = {
            name: producer.bindings[name]
            for name in ("intent.json", "lineage_records.json", "receipt.json")
        }
        verifier_intent = {
            "schema": VERIFIER_INTENT_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "status": (
                "started_independent_identity_lineage_verification_non_authorizing"
            ),
            "scientific_claim_authority": False,
            "registry_root": producer.bindings["registry_root"],
            "producer_files": producer_files,
            "lineage_contract": _contract_document(contract),
            "frozen_sources": _frozen_document(frozen),
            "access_policy": _intent_access_policy(),
        }
        stage = "write_verifier_intent"
        _exclusive_write(
            attempt / "intent.json",
            _canonical_json_bytes(verifier_intent),
            mode=0o444,
        )
        stage = "independently_replay_frozen_identity_sources"
        replay_records, source_bindings = _replay_sources(
            mmc2=mmc2,
            acquisition_receipt=acquisition_receipt,
            visible_artifact=visible_artifact,
            visible_manifest=visible_manifest,
            visible_receipt=visible_receipt,
            contract=contract,
            frozen=frozen,
        )
        _validate_source_bindings_document(
            source_bindings, contract=contract, frozen=frozen
        )
        if replay_records != producer.records:
            raise CCLEVisibleLineageRegistryError(
                "independent lineage replay differs from producer payload"
            )
        if source_bindings != producer.bindings.get("source_bindings"):
            raise CCLEVisibleLineageRegistryError(
                "independent lineage replay source binding differs from producer"
            )
        stage = "bind_verifier_code_and_protocol"
        code_bindings = _bind_code_files(code_paths)
        verifier_intent_file = _read_pinned_file(
            attempt / "intent.json",
            role="lineage verifier intent",
            maximum_bytes=_MAX_JSON_BYTES,
        )
        verifier_receipt = {
            "schema": VERIFIER_ATTEMPT_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "status": (
                "complete_independent_identity_lineage_verification_non_authorizing"
            ),
            "scientific_claim_authority": False,
            "producer_files": producer_files,
            "verifier_outputs": {"intent.json": verifier_intent_file.binding},
            "independent_source_bindings": source_bindings,
            "verified_contract": _verified_contract_document(contract, frozen),
            "code_bindings": code_bindings,
            "access_policy": _verifier_access_policy(),
        }
        stage = "write_verifier_receipt"
        _exclusive_write(
            attempt / "receipt.json",
            _canonical_json_bytes(verifier_receipt),
            mode=0o444,
        )
        stage = "validate_verifier_file_set"
        if {child.name for child in attempt.iterdir()} != {
            "intent.json",
            "receipt.json",
        }:
            raise CCLEVisibleLineageRegistryError(
                "lineage verifier attempt contains unexpected files"
            )
        stage = "seal_verifier_attempt_root"
        _seal_root(attempt)
        stage = "posterior_verify_verifier_attempt"
        attempt_bindings = _load_verifier_attempt(
            attempt,
            producer=producer,
            contract=contract,
            frozen=frozen,
        )
        completion_document = {
            "schema": VERIFIER_SCHEMA,
            "records_schema": REGISTRY_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "status": "complete",
            "verification_complete": True,
            "authoritative_for_visible_lineage_registry_integrity": True,
            "scientific_claim_authority": False,
            "registry_root": producer.bindings["registry_root"],
            "producer_files": producer_files,
            "verifier_attempt": {
                name: attempt_bindings[name]
                for name in ("root", "intent.json", "receipt.json")
            },
            "independent_source_bindings": source_bindings,
            "verified_contract": _verified_contract_document(contract, frozen),
            "code_bindings": code_bindings,
            "access_policy": _verifier_access_policy(),
        }
        stage = "publish_adjacent_completion"
        _exclusive_write(
            completion, _canonical_json_bytes(completion_document), mode=0o444
        )
        completion.chmod(0o444)
        _fsync_directory(completion.parent)
        stage = "posterior_verify_completed_registry"
        verified = _load_verified_visible_lineage_registry(
            registry_root=root,
            completion_receipt=completion,
            contract=contract,
            frozen=frozen,
        )
        return {
            "status": "complete",
            "registry_root": str(root),
            "verifier_attempt_root": str(attempt),
            "completion_receipt": str(completion),
            "sample_count": len(verified.records),
            "internal_final_accessed": False,
            "scientific_claim_authority": False,
        }
    except BaseException:
        if created:
            _preserve_failed_attempt(
                attempt,
                schema=VERIFIER_ATTEMPT_SCHEMA,
                stage=stage,
                status="failed_independent_identity_lineage_verification_non_authorizing",
            )
        raise


def _load_verified_visible_lineage_registry(
    *,
    registry_root: str | Path,
    completion_receipt: str | Path,
    contract: LineageContract = PRODUCTION_CONTRACT,
    frozen: FrozenSources = PRODUCTION_FROZEN_SOURCES,
) -> VerifiedVisibleLineageRegistry:
    _validate_contract(contract)
    root = _absolute(registry_root, "lineage registry root")
    completion_path = _absolute(
        completion_receipt, "lineage registry completion receipt"
    )
    _reject_forbidden_path(root, "lineage registry root")
    if completion_path != completion_receipt_path(root):
        raise CCLEVisibleLineageRegistryError(
            "completion receipt is not the registry root's adjacent sibling"
        )
    producer = _load_producer_registry(root, contract=contract, frozen=frozen)
    attempt_path = verifier_attempt_root_path(root)
    attempt_bindings = _load_verifier_attempt(
        attempt_path,
        producer=producer,
        contract=contract,
        frozen=frozen,
    )
    completion = _read_pinned_file(
        completion_path,
        role="lineage registry completion receipt",
        maximum_bytes=_MAX_JSON_BYTES,
        required_mode=0o444,
    )
    document = _json_object(completion, "lineage registry completion receipt")
    producer_files = document.get("producer_files")
    verified_contract = document.get("verified_contract")
    access = document.get("access_policy")
    code_bindings = document.get("code_bindings")
    source_bindings = document.get("independent_source_bindings")
    verifier_attempt = document.get("verifier_attempt")
    if (
        set(document)
        != {
            "schema",
            "records_schema",
            "protocol_id",
            "status",
            "verification_complete",
            "authoritative_for_visible_lineage_registry_integrity",
            "scientific_claim_authority",
            "registry_root",
            "producer_files",
            "verifier_attempt",
            "independent_source_bindings",
            "verified_contract",
            "code_bindings",
            "access_policy",
        }
        or document.get("schema") != VERIFIER_SCHEMA
        or document.get("records_schema") != REGISTRY_SCHEMA
        or document.get("protocol_id") != PROTOCOL_ID
        or document.get("status") != "complete"
        or document.get("verification_complete") is not True
        or document.get("authoritative_for_visible_lineage_registry_integrity")
        is not True
        or document.get("scientific_claim_authority") is not False
        or document.get("registry_root") != producer.bindings.get("registry_root")
        or verifier_attempt
        != {
            name: attempt_bindings[name]
            for name in ("root", "intent.json", "receipt.json")
        }
        or producer_files
        != {
            name: producer.bindings[name]
            for name in ("intent.json", "lineage_records.json", "receipt.json")
        }
        or verified_contract != _verified_contract_document(contract, frozen)
        or access != _verifier_access_policy()
    ):
        raise CCLEVisibleLineageRegistryError(
            "lineage registry independent completion contract changed"
        )
    source_bindings = _validate_source_bindings_document(
        source_bindings, contract=contract, frozen=frozen
    )
    if source_bindings != producer.bindings.get("source_bindings"):
        raise CCLEVisibleLineageRegistryError(
            "completed verifier source binding differs from producer"
        )
    if source_bindings != attempt_bindings.get(
        "source_bindings"
    ) or code_bindings != attempt_bindings.get("code_bindings"):
        raise CCLEVisibleLineageRegistryError(
            "completion differs from sealed verifier attempt"
        )
    _require_current_code_binding(code_bindings, "verifier")
    completion_after = _read_pinned_file(
        completion_path,
        role="lineage registry completion posterior reread",
        maximum_bytes=_MAX_JSON_BYTES,
        required_mode=0o444,
    )
    if completion_after.binding != completion.binding:
        raise CCLEVisibleLineageRegistryError(
            "lineage completion changed during posterior reread"
        )
    producer_after = _load_producer_registry(root, contract=contract, frozen=frozen)
    attempt_after = _load_verifier_attempt(
        attempt_path,
        producer=producer_after,
        contract=contract,
        frozen=frozen,
    )
    if (
        producer_after.records != producer.records
        or producer_after.bindings != producer.bindings
        or attempt_after != attempt_bindings
    ):
        raise CCLEVisibleLineageRegistryError(
            "producer registry changed during completion posterior read"
        )
    bindings = dict(producer.bindings)
    bindings["completion_receipt"] = completion.binding
    bindings["verifier_source_bindings"] = source_bindings
    bindings["verifier_code_bindings"] = code_bindings
    return VerifiedVisibleLineageRegistry(
        records=producer.records,
        bindings=bindings,
    )


__all__ = [
    "CCLEVisibleLineageRegistryError",
    "FileSpec",
    "FrozenSources",
    "IDENTITY_GROUP_RULE",
    "INTENT_SCHEMA",
    "LINEAGE_RULE",
    "LineageContract",
    "LineageRecord",
    "MMC2_IDENTITY_COLUMNS",
    "MMC2_IDENTITY_COLUMN_NUMBERS",
    "MemberSpec",
    "PRODUCER_SCHEMA",
    "PRODUCTION_CONTRACT",
    "PRODUCTION_FROZEN_SOURCES",
    "PROTOCOL_ID",
    "REGISTRY_SCHEMA",
    "ROW_FIELDS",
    "SHEET_NAME",
    "VERIFIER_SCHEMA",
    "VERIFIER_ATTEMPT_SCHEMA",
    "VERIFIER_INTENT_SCHEMA",
    "VISIBLE_IDENTITY_MEMBERS",
    "VerifiedVisibleLineageRegistry",
    "completion_receipt_path",
    "load_verified_visible_lineage_registry",
    "produce_visible_lineage_registry",
    "verify_visible_lineage_registry",
    "verifier_attempt_root_path",
]

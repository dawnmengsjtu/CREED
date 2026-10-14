"""Auditable sample-level scGPT encoding for explicitly labelled bulk assays.

The whole-human scGPT ``embed_data`` helper rank-bins the non-zero values in
each sample.  The adapter supports the legacy GBM FPKM-UQ abundance contract
and the CCLE RSEM expected-count contract as two distinct input semantics.  It
does not relabel either assay as raw single-cell counts.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


SCGPT_VERSION = "0.2.4"
GBM_INPUT_SEMANTICS = "bulk_fpkm_uq_nonnegative_abundance"
CCLE_INPUT_SEMANTICS = "bulk_rsem_expected_counts_nonnegative"
# Backwards-compatible name used by the frozen GBM runner and its receipts.
INPUT_SEMANTICS = GBM_INPUT_SEMANTICS
SUPPORTED_INPUT_SEMANTICS = frozenset(
    {GBM_INPUT_SEMANTICS, CCLE_INPUT_SEMANTICS}
)
VALUE_ADAPTER = "native_per_sample_nonzero_rank_binning_51"
SAMPLE_LOCAL_TOKEN_SAMPLER = "sha256_sample_seed_numpy_pcg64_v1"
SAMPLE_LOCAL_SEED_SCHEMA = "creed.scgpt.sample-local-token-seed.v1"
SAMPLE_LOCAL_BINNING = "scgpt_0.2.4_quantile_digitize_sample_rng_v1"
SCGPT_MODEL_INITIALIZATION_SEED = 1729
SCGPT_ACTIVE_INFERENCE_PREFIXES = (
    "encoder.",
    "value_encoder.",
    "transformer_encoder.",
)
MISSING_POLICY = "omit_missing_gene_token"
GBM_ASSAY_SHIFT = "bulk_tumor_fpkm_uq_to_single_cell_pretrained_scgpt"
CCLE_ASSAY_SHIFT = (
    "bulk_cell_line_rsem_expected_counts_to_single_cell_pretrained_scgpt"
)
# Backwards-compatible name used by the frozen GBM protocol.
ASSAY_SHIFT = GBM_ASSAY_SHIFT
ASSAY_SHIFT_BY_INPUT_SEMANTICS = {
    GBM_INPUT_SEMANTICS: GBM_ASSAY_SHIFT,
    CCLE_INPUT_SEMANTICS: CCLE_ASSAY_SHIFT,
}
SCGPT_INFERENCE_CONTRACT_SCHEMA = "creed.scgpt-inference-contract.v1"
OFFICIAL_WHOLE_HUMAN_MODEL_SHA256 = (
    "6cb5d451ab5c4b33eb673adbe4fddc61d2389df1b89b7651a9fe2e557572b922"
)
OFFICIAL_WHOLE_HUMAN_VOCAB_SHA256 = (
    "acca93d114ca62c3f0f50debbd23e8c87f0714f4737764454f6b2b13f2e8580f"
)
OFFICIAL_WHOLE_HUMAN_ARGS_SHA256 = (
    "c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60"
)
OFFICIAL_WHOLE_HUMAN_ASSET_HASHES = {
    "best_model_pt_sha256": OFFICIAL_WHOLE_HUMAN_MODEL_SHA256,
    "vocab_json_sha256": OFFICIAL_WHOLE_HUMAN_VOCAB_SHA256,
    "args_json_sha256": OFFICIAL_WHOLE_HUMAN_ARGS_SHA256,
}


class ScGPTInputError(ValueError):
    """Raised when an input, model, or output violates the frozen contract."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def identifier_sequence_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _derive_sample_local_seed(
    ensemble_seed: int, sample_id: str, *, domain: str
) -> int:
    """Derive a domain-separated 64-bit seed for one sample and member."""

    if isinstance(ensemble_seed, (bool, np.bool_)) or int(ensemble_seed) != ensemble_seed:
        raise ScGPTInputError("ensemble seed must be an integer")
    normalized = str(sample_id)
    if not normalized:
        raise ScGPTInputError("sample ID is required for row-local token sampling")
    digest = hashlib.sha256()
    if not domain:
        raise ScGPTInputError("sample-local RNG domain is required")
    for value in (
        SAMPLE_LOCAL_SEED_SCHEMA,
        domain,
        str(int(ensemble_seed)),
        normalized,
    ):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], "big", signed=False)


def derive_sample_local_seed(ensemble_seed: int, sample_id: str) -> int:
    """Derive the stable gene-selection RNG seed for one sample and member."""

    return _derive_sample_local_seed(
        ensemble_seed, sample_id, domain="gene_selection"
    )


def derive_sample_local_binning_seed(ensemble_seed: int, sample_id: str) -> int:
    """Derive the stable tie-breaking RNG seed for one sample and member."""

    return _derive_sample_local_seed(
        ensemble_seed, sample_id, domain="expression_binning_ties"
    )


def _int64_array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def sample_local_gene_indices(
    positive_gene_indices: np.ndarray,
    *,
    sample_id: str,
    ensemble_seed: int,
    maximum_gene_tokens: int,
) -> tuple[np.ndarray, int]:
    """Select one row's genes without consulting any other row or global RNG."""

    positive = np.asarray(positive_gene_indices, dtype=np.int64)
    if positive.ndim != 1 or np.any(positive < 0):
        raise ScGPTInputError("positive gene indices must be one-dimensional")
    if len(np.unique(positive)) != len(positive):
        raise ScGPTInputError("positive gene indices must be unique")
    if maximum_gene_tokens < 1:
        raise ScGPTInputError("maximum_gene_tokens must be positive")
    derived_seed = derive_sample_local_seed(ensemble_seed, sample_id)
    if len(positive) <= maximum_gene_tokens:
        selected = positive.copy()
    else:
        generator = np.random.default_rng(derived_seed)
        order = generator.permutation(len(positive))[:maximum_gene_tokens]
        selected = positive[order]
    selected.setflags(write=False)
    return selected, derived_seed


def sample_local_binning(
    row: np.ndarray,
    *,
    n_bins: int,
    sample_id: str,
    ensemble_seed: int,
) -> tuple[np.ndarray, int]:
    """Reproduce scGPT 0.2.4 binning with an injected row-local tie RNG."""

    values = np.asarray(row)
    if values.ndim != 1 or values.size == 0:
        raise ScGPTInputError("sample-local binning requires one nonempty row")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ScGPTInputError("sample-local binning row must be finite and nonnegative")
    if isinstance(n_bins, (bool, np.bool_)) or int(n_bins) != n_bins or n_bins < 2:
        raise ScGPTInputError("sample-local bin count must be an integer >= 2")
    derived_seed = derive_sample_local_binning_seed(ensemble_seed, sample_id)
    if float(np.max(values)) == 0.0:
        return np.zeros_like(values, dtype=np.int64), derived_seed

    def digitize(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
        left = np.digitize(x, bins)
        right = np.digitize(x, bins, right=True)
        generator = np.random.default_rng(derived_seed)
        random_values = generator.random(len(x))
        return np.ceil(random_values * (right - left) + left).astype(np.int64)

    if float(np.min(values)) <= 0.0:
        nonzero_indices = values.nonzero()
        nonzero = values[nonzero_indices]
        bins = np.quantile(nonzero, np.linspace(0, 1, int(n_bins) - 1))
        result = np.zeros_like(values, dtype=np.int64)
        result[nonzero_indices] = digitize(nonzero, bins)
    else:
        bins = np.quantile(values, np.linspace(0, 1, int(n_bins) - 1))
        result = digitize(values, bins)
    return result, derived_seed


def sample_local_tokenize_row(
    row: np.ndarray,
    vocabulary_gene_ids: np.ndarray,
    *,
    sample_id: str,
    ensemble_seed: int,
    max_length: int,
    cls_token_id: int,
    pad_token_id: int,
    pad_value: float,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    """Construct one complete scGPT token row without shared mutable state."""

    values = np.asarray(row, dtype=np.float32)
    gene_ids = np.asarray(vocabulary_gene_ids, dtype=np.int64)
    if values.ndim != 1 or gene_ids.ndim != 1 or values.shape != gene_ids.shape:
        raise ScGPTInputError("sample token row and vocabulary IDs are misaligned")
    if max_length < 2:
        raise ScGPTInputError("sample token max_length must include CLS and a gene")
    positive = np.flatnonzero(values > 0.0)
    all_positive_binned, binning_seed = sample_local_binning(
        values[positive],
        n_bins=51,
        sample_id=sample_id,
        ensemble_seed=ensemble_seed,
    )
    selected, selection_seed = sample_local_gene_indices(
        positive,
        sample_id=sample_id,
        ensemble_seed=ensemble_seed,
        maximum_gene_tokens=max_length - 1,
    )
    selected_positions = np.searchsorted(positive, selected)
    if not np.array_equal(positive[selected_positions], selected):
        raise ScGPTInputError("sample-local gene selection left the positive row")
    genes = np.full(max_length, int(pad_token_id), dtype=np.int64)
    expressions = np.full(max_length, float(pad_value), dtype=np.float32)
    genes[0] = int(cls_token_id)
    width = len(selected)
    genes[1 : width + 1] = gene_ids[selected]
    expressions[1 : width + 1] = all_positive_binned[selected_positions]
    audit = {
        "selection_seed": selection_seed,
        "binning_seed": binning_seed,
        "selected_gene_indices_sha256": _int64_array_sha256(selected),
        "token_gene_ids_sha256": _int64_array_sha256(genes),
        "token_values_sha256": _int64_array_sha256(
            expressions.astype(np.int64, copy=False)
        ),
    }
    return genes, expressions, audit


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


def _assay_label(input_semantics: str) -> str:
    if input_semantics == GBM_INPUT_SEMANTICS:
        return "FPKM-UQ abundance"
    if input_semantics == CCLE_INPUT_SEMANTICS:
        return "RSEM expected counts"
    return "expression"


@dataclass(frozen=True, slots=True)
class ExpressionPayload:
    sample_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    values: np.ndarray
    source_sha256: str
    input_semantics: str = INPUT_SEMANTICS

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ScGPTInputError("expression values must be a two-dimensional matrix")
        if self.values.shape != (len(self.sample_ids), len(self.gene_ids)):
            raise ScGPTInputError("expression matrix does not match sample/gene schema")
        _validate_identifiers(self.sample_ids, "sample")
        _validate_identifiers(self.gene_ids, "gene")
        if not _is_sha256(self.source_sha256):
            raise ScGPTInputError("source_sha256 must be a SHA-256 hex digest")
        if self.input_semantics not in SUPPORTED_INPUT_SEMANTICS:
            raise ScGPTInputError("expression payload has unsupported input semantics")


@dataclass(frozen=True, slots=True)
class ScGPTModelContract:
    model_dir: Path
    model_sha256: str
    vocab_sha256: str
    args_sha256: str
    vocab: Mapping[str, int]
    args: Mapping[str, Any]
    inference: Mapping[str, Any] = field(default_factory=dict)
    inference_contract_path: Path | None = None
    inference_contract_sha256: str | None = None

    def __post_init__(self) -> None:
        declared = self.inference if self.inference else self.args
        object.__setattr__(self, "inference", _validate_inference_metadata(declared))


@dataclass(frozen=True, slots=True)
class CCLEScGPTTokenInput:
    """Symbol-token view of CCLE counts with Ensembl provenance retained."""

    payload: ExpressionPayload
    mapping_receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PreparedScGPTInput:
    sample_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    values: np.ndarray
    source_sha256: str
    input_semantics: str
    sample_schema_sha256: str
    input_gene_schema_sha256: str
    matched_gene_schema_sha256: str
    input_gene_count: int
    matched_gene_count: int
    unmatched_gene_count: int
    missing_value_count: int
    observed_zero_count: int
    positive_gene_count_min: int
    positive_gene_count_median: float
    positive_gene_count_max: int


@dataclass(frozen=True, slots=True)
class ScGPTEmbeddingConfig:
    max_length: int = 1200
    batch_size: int = 16
    seeds: tuple[int, ...] = (3407, 3408, 3409, 3410)
    device: str = "cuda"
    use_fast_transformer: bool = False
    minimum_vocab_genes: int = 1200
    minimum_positive_genes_per_sample: int = 1200
    input_semantics: str = INPUT_SEMANTICS
    value_adapter: str = VALUE_ADAPTER
    missing_policy: str = MISSING_POLICY

    def __post_init__(self) -> None:
        if self.input_semantics not in SUPPORTED_INPUT_SEMANTICS:
            supported = ", ".join(sorted(SUPPORTED_INPUT_SEMANTICS))
            raise ScGPTInputError(
                "unsupported input semantics; explicitly label the input as "
                f"FPKM-UQ or RSEM expected counts using one of: {supported}"
            )
        if self.value_adapter != VALUE_ADAPTER:
            raise ScGPTInputError("only native 51-bin non-zero rank binning is allowed")
        if self.missing_policy != MISSING_POLICY:
            raise ScGPTInputError("only omission of missing gene tokens is allowed")
        if self.max_length < 2 or self.batch_size < 1:
            raise ScGPTInputError("max_length and batch_size must be positive")
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise ScGPTInputError(
                "at least two unique ensemble seeds are required for uncertainty"
            )
        if any(seed < 0 for seed in self.seeds):
            raise ScGPTInputError("ensemble seeds must be non-negative")
        if self.minimum_vocab_genes < self.max_length:
            raise ScGPTInputError("minimum_vocab_genes cannot be below max_length")
        if self.minimum_positive_genes_per_sample < self.max_length:
            raise ScGPTInputError(
                "minimum positive genes per sample cannot be below max_length"
            )

    def receipt_payload(self) -> dict[str, Any]:
        return {
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "seeds": list(self.seeds),
            "device": self.device,
            "use_fast_transformer": self.use_fast_transformer,
            "minimum_vocab_genes": self.minimum_vocab_genes,
            "minimum_positive_genes_per_sample": (
                self.minimum_positive_genes_per_sample
            ),
            "input_semantics": self.input_semantics,
            "value_adapter": self.value_adapter,
            "missing_policy": self.missing_policy,
            "token_sampler": SAMPLE_LOCAL_TOKEN_SAMPLER,
            "value_binning": SAMPLE_LOCAL_BINNING,
            "sample_seed_schema": SAMPLE_LOCAL_SEED_SCHEMA,
            "sample_seed_inputs": ["rng_domain", "ensemble_seed", "exact_sample_id"],
            "cross_sample_rng_state_allowed": False,
            "assay_shift": ASSAY_SHIFT_BY_INPUT_SEMANTICS[self.input_semantics],
            "scientific_scope": "development_representation_not_in_distribution_claim",
        }


@dataclass(frozen=True, slots=True)
class EmbeddingEnsemble:
    sample_ids: tuple[str, ...]
    member_embeddings: np.ndarray
    mean_embedding: np.ndarray
    per_dimension_standard_deviation: np.ndarray
    sample_l2_uncertainty: np.ndarray
    backend_metadata: Mapping[str, Any]


class SampleEmbeddingBackend(Protocol):
    def metadata(self, model_dir: Path) -> Mapping[str, Any]: ...

    def embed(
        self,
        values: np.ndarray,
        gene_ids: tuple[str, ...],
        sample_ids: tuple[str, ...],
        model_dir: Path,
        config: ScGPTEmbeddingConfig,
        seed: int,
    ) -> np.ndarray: ...


def _validate_identifiers(values: Sequence[str], label: str) -> None:
    if not values:
        raise ScGPTInputError(f"{label} identifier sequence is empty")
    normalized = tuple(str(value).strip() for value in values)
    if any(not value for value in normalized):
        raise ScGPTInputError(f"{label} identifiers must be non-empty")
    if len(set(normalized)) != len(normalized):
        raise ScGPTInputError(f"{label} identifiers must be unique")


def _parse_value(raw: str, *, row: int, column: int) -> float:
    value = raw.strip()
    if value in {"", "NA", "NaN", "nan", "N/A"}:
        return float("nan")
    try:
        parsed = float(value)
    except ValueError as error:
        raise ScGPTInputError(
            f"non-numeric expression at row {row}, column {column}"
        ) from error
    if np.isinf(parsed):
        raise ScGPTInputError(
            f"infinite expression at row {row}, column {column}"
        )
    return parsed


def load_gene_by_sample_tsv(path: str | Path) -> ExpressionPayload:
    """Load a create-once gene-by-sample TSV without changing its semantics."""

    source = Path(path)
    if not source.is_file():
        raise ScGPTInputError(f"missing expression input: {source}")
    genes: list[str] = []
    rows: list[list[float]] = []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration as error:
            raise ScGPTInputError("expression TSV is empty") from error
        if len(header) < 3:
            raise ScGPTInputError("expression TSV requires a gene column and samples")
        sample_ids = tuple(value.strip() for value in header[1:])
        _validate_identifiers(sample_ids, "sample")
        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ScGPTInputError(
                    f"row {row_number} has {len(row)} columns; expected {len(header)}"
                )
            gene = row[0].strip()
            if not gene:
                raise ScGPTInputError(f"empty gene identifier at row {row_number}")
            genes.append(gene)
            rows.append(
                [
                    _parse_value(value, row=row_number, column=column)
                    for column, value in enumerate(row[1:], start=2)
                ]
            )
    gene_ids = tuple(genes)
    _validate_identifiers(gene_ids, "gene")
    values = np.asarray(rows, dtype=np.float64).T
    return ExpressionPayload(
        sample_ids=sample_ids,
        gene_ids=gene_ids,
        values=values,
        source_sha256=sha256_file(source),
    )


def adapt_ccle_expected_counts_to_scgpt_tokens(
    *,
    sample_ids: Sequence[str],
    gene_ids: Sequence[str],
    symbols: Sequence[str],
    values: np.ndarray,
    source_sha256: str,
) -> CCLEScGPTTokenInput:
    """Map source Ensembl genes to scGPT symbols without changing count semantics.

    ``gene_ids`` remain the primary CCLE feature identities.  Symbols are only
    token annotations for scGPT.  Empty symbols are dropped explicitly and all
    source genes sharing a symbol are summed as additive expected counts.  A
    missing value in any member of a duplicate group propagates to the grouped
    token; it is not silently treated as zero.
    """

    normalized_samples = tuple(str(value).strip() for value in sample_ids)
    normalized_gene_ids = tuple(str(value).strip() for value in gene_ids)
    if not all(isinstance(value, str) for value in symbols):
        raise ScGPTInputError("CCLE source symbol annotations must be strings")
    normalized_symbols = tuple(value.strip() for value in symbols)
    _validate_identifiers(normalized_samples, "sample")
    _validate_identifiers(normalized_gene_ids, "Ensembl gene")
    if len(normalized_symbols) != len(normalized_gene_ids):
        raise ScGPTInputError("CCLE symbol annotations do not match Ensembl genes")
    source_values = np.asarray(values, dtype=np.float64)
    expected_shape = (len(normalized_samples), len(normalized_gene_ids))
    if source_values.ndim != 2 or source_values.shape != expected_shape:
        raise ScGPTInputError("CCLE expected-count matrix does not match its schema")
    if np.any(np.isinf(source_values)):
        raise ScGPTInputError("RSEM expected counts contain infinite values")
    finite = np.isfinite(source_values)
    if np.any(source_values[finite] < 0):
        raise ScGPTInputError("RSEM expected counts must be non-negative or missing")
    if not _is_sha256(source_sha256):
        raise ScGPTInputError("source_sha256 must be a SHA-256 hex digest")

    indices_by_symbol: dict[str, list[int]] = {}
    empty_symbol_gene_ids: list[str] = []
    for index, (gene_id, symbol) in enumerate(
        zip(normalized_gene_ids, normalized_symbols, strict=True)
    ):
        if not symbol:
            empty_symbol_gene_ids.append(gene_id)
            continue
        indices_by_symbol.setdefault(symbol, []).append(index)
    if not indices_by_symbol:
        raise ScGPTInputError("CCLE source has no non-empty scGPT symbol annotations")

    token_symbols = tuple(indices_by_symbol)
    aggregated = np.empty(
        (len(normalized_samples), len(token_symbols)), dtype=np.float64
    )
    mapping: list[dict[str, object]] = []
    duplicate_groups = 0
    duplicate_source_gene_count = 0
    for token_index, symbol in enumerate(token_symbols):
        source_indices = indices_by_symbol[symbol]
        source_ids = [normalized_gene_ids[index] for index in source_indices]
        if len(source_indices) > 1:
            duplicate_groups += 1
            duplicate_source_gene_count += len(source_indices)
        # np.sum intentionally propagates NaN.  nansum would silently reinterpret
        # a missing expected count as an observed zero.
        aggregated[:, token_index] = np.sum(
            source_values[:, source_indices], axis=1, dtype=np.float64
        )
        mapping.append({"symbol": symbol, "source_gene_ids": source_ids})
    if np.any(np.isinf(aggregated)):
        raise ScGPTInputError("summed RSEM expected counts overflowed")

    mapping_receipt: dict[str, Any] = {
        "primary_gene_identity": "source_ensembl_gene_id",
        "scgpt_token_identity": "trimmed_source_symbol_annotation",
        "symbol_whitespace_policy": "strip_leading_and_trailing_whitespace",
        "symbol_case_policy": "preserve_source_case_exactly",
        "symbol_order_policy": "first_source_gene_occurrence",
        "silent_symbol_overwrite_allowed": False,
        "source_gene_count": len(normalized_gene_ids),
        "source_ensembl_gene_ids_sha256": identifier_sequence_sha256(
            normalized_gene_ids
        ),
        "source_symbol_annotations_sha256": identifier_sequence_sha256(
            normalized_symbols
        ),
        "empty_symbol_policy": "drop_before_scgpt_tokenization",
        "empty_symbol_gene_count": len(empty_symbol_gene_ids),
        "empty_symbol_ensembl_gene_ids_sha256": identifier_sequence_sha256(
            empty_symbol_gene_ids
        ),
        "nonempty_symbol_source_gene_count": (
            len(normalized_gene_ids) - len(empty_symbol_gene_ids)
        ),
        "unique_scgpt_symbol_count": len(token_symbols),
        "scgpt_symbol_schema_sha256": identifier_sequence_sha256(token_symbols),
        "duplicate_symbol_policy": (
            "sum_additive_rsem_expected_counts_in_source_gene_order"
        ),
        "duplicate_missing_value_policy": (
            "propagate_missing_if_any_source_gene_is_missing"
        ),
        "duplicate_symbol_group_count": duplicate_groups,
        "source_genes_in_duplicate_symbol_groups": duplicate_source_gene_count,
        "collapsed_duplicate_source_gene_count": (
            len(normalized_gene_ids)
            - len(empty_symbol_gene_ids)
            - len(token_symbols)
        ),
        "token_to_source_ensembl_mapping_sha256": canonical_json_sha256(mapping),
        "source_missing_value_count": int(np.count_nonzero(~finite)),
        "adapted_missing_value_count": int(np.count_nonzero(~np.isfinite(aggregated))),
    }
    payload = ExpressionPayload(
        sample_ids=normalized_samples,
        gene_ids=token_symbols,
        values=aggregated,
        source_sha256=source_sha256,
        input_semantics=CCLE_INPUT_SEMANTICS,
    )
    return CCLEScGPTTokenInput(payload=payload, mapping_receipt=mapping_receipt)


def _load_inference_contract(
    path: str | Path,
    *,
    asset_hashes: Mapping[str, str],
) -> tuple[Path, str, Mapping[str, Any]]:
    source = Path(path).resolve()
    if not source.is_file():
        raise ScGPTInputError("scGPT inference contract is missing")
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ScGPTInputError("scGPT inference contract is invalid JSON") from error
    if not isinstance(document, dict):
        raise ScGPTInputError("scGPT inference contract must be a JSON object")
    integrity = document.get("integrity")
    if not isinstance(integrity, Mapping):
        raise ScGPTInputError("scGPT inference contract lacks integrity")
    payload = dict(document)
    payload.pop("integrity", None)
    if integrity.get("canonical_payload_sha256") != canonical_json_sha256(payload):
        raise ScGPTInputError("scGPT inference contract payload hash mismatch")
    if (
        document.get("schema") != SCGPT_INFERENCE_CONTRACT_SCHEMA
        or document.get("status") != "frozen"
    ):
        raise ScGPTInputError("unexpected scGPT inference contract identity")
    if document.get("assets") != dict(asset_hashes):
        raise ScGPTInputError("scGPT inference contract does not bind model assets")
    evidence = document.get("evidence")
    if (
        not isinstance(evidence, Mapping)
        or not isinstance(evidence.get("source"), str)
        or not evidence["source"].strip()
        or not _is_sha256(evidence.get("source_sha256"))
    ):
        raise ScGPTInputError(
            "scGPT inference contract lacks hash-bound source evidence"
        )
    evidence_source = Path(evidence["source"])
    if not evidence_source.is_absolute():
        evidence_source = source.parent / evidence_source
    if (
        not evidence_source.is_file()
        or sha256_file(evidence_source) != evidence["source_sha256"]
    ):
        raise ScGPTInputError("scGPT inference-contract source evidence mismatch")
    inference = document.get("inference")
    if not isinstance(inference, Mapping):
        raise ScGPTInputError("scGPT inference contract lacks inference metadata")
    return source, sha256_file(source), dict(inference)


def _validate_inference_metadata(inference: Mapping[str, Any]) -> dict[str, Any]:
    try:
        n_bins = int(inference.get("n_bins", -1))
        max_seq_len = int(inference.get("max_seq_len", -1))
    except (TypeError, ValueError) as error:
        raise ScGPTInputError("invalid scGPT inference metadata") from error
    if inference.get("input_style") != "binned" or n_bins != 51:
        raise ScGPTInputError("whole-human model must declare binned 51-bin inputs")
    if max_seq_len != 1200:
        raise ScGPTInputError("whole-human model must declare max_seq_len=1200")
    value_adapter = inference.get("value_adapter", VALUE_ADAPTER)
    if value_adapter != VALUE_ADAPTER:
        raise ScGPTInputError("inference contract declares an unsupported value adapter")
    return {
        "input_style": "binned",
        "n_bins": n_bins,
        "max_seq_len": max_seq_len,
        "value_adapter": value_adapter,
    }


def load_model_contract(
    model_dir: str | Path,
    *,
    inference_contract_path: str | Path | None = None,
) -> ScGPTModelContract:
    root = Path(model_dir).resolve()
    paths = {
        "model": root / "best_model.pt",
        "vocab": root / "vocab.json",
        "args": root / "args.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ScGPTInputError(f"model directory lacks required files: {missing}")
    vocab = json.loads(paths["vocab"].read_text(encoding="utf-8"))
    args = json.loads(paths["args"].read_text(encoding="utf-8"))
    if not isinstance(vocab, dict) or not vocab:
        raise ScGPTInputError("vocab.json must contain a non-empty token mapping")
    try:
        embedding_dimension = int(args.get("embsize", -1))
    except (TypeError, ValueError) as error:
        raise ScGPTInputError("whole-human model lacks a valid embsize") from error
    if embedding_dimension <= 0:
        raise ScGPTInputError("whole-human model lacks a valid embsize")
    asset_hashes = {
        "best_model_pt_sha256": sha256_file(paths["model"]),
        "vocab_json_sha256": sha256_file(paths["vocab"]),
        "args_json_sha256": sha256_file(paths["args"]),
    }
    inference_keys = ("input_style", "n_bins", "max_seq_len")
    args_declares_inference = all(key in args for key in inference_keys)
    contract_path: Path | None = None
    contract_sha256: str | None = None
    if inference_contract_path is None:
        if not args_declares_inference:
            raise ScGPTInputError(
                "args.json lacks inference metadata; a frozen hash-bound "
                "inference contract is required"
            )
        inference = _validate_inference_metadata(args)
    else:
        contract_path, contract_sha256, declared = _load_inference_contract(
            inference_contract_path, asset_hashes=asset_hashes
        )
        inference = _validate_inference_metadata(declared)
        if args_declares_inference:
            args_inference = _validate_inference_metadata(args)
            if inference != args_inference:
                raise ScGPTInputError(
                    "inference contract conflicts with args.json metadata"
                )
    return ScGPTModelContract(
        model_dir=root,
        model_sha256=asset_hashes["best_model_pt_sha256"],
        vocab_sha256=asset_hashes["vocab_json_sha256"],
        args_sha256=asset_hashes["args_json_sha256"],
        vocab={str(key): int(value) for key, value in vocab.items()},
        args=args,
        inference=inference,
        inference_contract_path=contract_path,
        inference_contract_sha256=contract_sha256,
    )


def require_official_whole_human_model(contract: ScGPTModelContract) -> None:
    """Reject derived or incomplete asset triplets in production runners."""

    observed = {
        "best_model_pt_sha256": contract.model_sha256,
        "vocab_json_sha256": contract.vocab_sha256,
        "args_json_sha256": contract.args_sha256,
    }
    if observed != OFFICIAL_WHOLE_HUMAN_ASSET_HASHES:
        raise ScGPTInputError(
            "model directory is not the frozen original whole-human scGPT asset triplet"
        )


def prepare_scgpt_input(
    payload: ExpressionPayload,
    contract: ScGPTModelContract,
    config: ScGPTEmbeddingConfig,
) -> PreparedScGPTInput:
    if payload.input_semantics != config.input_semantics:
        raise ScGPTInputError(
            "expression payload semantics do not match the scGPT configuration"
        )
    if config.max_length > int(contract.inference["max_seq_len"]):
        raise ScGPTInputError("requested max_length exceeds the checkpoint contract")
    values = np.asarray(payload.values, dtype=np.float64)
    if np.any(np.isinf(values)):
        raise ScGPTInputError(
            f"{_assay_label(config.input_semantics)} input contains infinite values"
        )
    finite = np.isfinite(values)
    if np.any(values[finite] < 0):
        raise ScGPTInputError(
            f"{_assay_label(config.input_semantics)} input must be non-negative "
            "or missing"
        )

    matched_indices = np.asarray(
        [index for index, gene in enumerate(payload.gene_ids) if gene in contract.vocab],
        dtype=np.int64,
    )
    if len(matched_indices) < config.minimum_vocab_genes:
        raise ScGPTInputError(
            f"only {len(matched_indices)} genes match the scGPT vocabulary; "
            f"minimum is {config.minimum_vocab_genes}"
        )
    matched_genes = tuple(payload.gene_ids[int(index)] for index in matched_indices)
    matched_values = values[:, matched_indices].copy()
    missing = ~np.isfinite(matched_values)
    missing_count = int(missing.sum())
    matched_values[missing] = 0.0
    positive_per_sample = np.count_nonzero(matched_values > 0.0, axis=1)
    if np.any(positive_per_sample < config.minimum_positive_genes_per_sample):
        bad = [
            payload.sample_ids[index]
            for index in np.flatnonzero(
                positive_per_sample < config.minimum_positive_genes_per_sample
            )
        ]
        raise ScGPTInputError(
            "samples lack enough observed positive vocabulary genes for the frozen "
            f"1200-token adapter: {bad[:5]}"
        )
    return PreparedScGPTInput(
        sample_ids=payload.sample_ids,
        gene_ids=matched_genes,
        values=matched_values.astype(np.float32, copy=False),
        source_sha256=payload.source_sha256,
        input_semantics=payload.input_semantics,
        sample_schema_sha256=identifier_sequence_sha256(payload.sample_ids),
        input_gene_schema_sha256=identifier_sequence_sha256(payload.gene_ids),
        matched_gene_schema_sha256=identifier_sequence_sha256(matched_genes),
        input_gene_count=len(payload.gene_ids),
        matched_gene_count=len(matched_genes),
        unmatched_gene_count=len(payload.gene_ids) - len(matched_genes),
        missing_value_count=missing_count,
        observed_zero_count=int(np.count_nonzero(matched_values == 0.0))
        - missing_count,
        positive_gene_count_min=int(positive_per_sample.min()),
        positive_gene_count_median=float(np.median(positive_per_sample)),
        positive_gene_count_max=int(positive_per_sample.max()),
    )


class OfficialScGPTBackend:
    """Frozen scGPT 0.2.4 inference with row-local token sampling.

    The upstream ``embed_data`` helper draws every row's gene subset from one
    process-global torch RNG stream.  That makes a row's selected token set
    depend on which rows preceded it.  CREED instead reproduces the official
    non-zero filtering, 51-bin transform, model construction and CLS pooling,
    while deriving the sampling RNG from only ``(ensemble seed, sample ID)``.
    GPU batching is therefore an execution detail rather than part of the
    representation definition.
    """

    def __init__(self) -> None:
        self._invariance_audits: list[dict[str, Any]] = []
        self._checkpoint_coverages: list[dict[str, Any]] = []

    def metadata(self, model_dir: Path) -> Mapping[str, Any]:
        import scgpt
        import torch
        from scgpt.model import TransformerModel
        from scgpt.preprocess import binning
        from scgpt.tokenizer import GeneVocab
        from scgpt.utils import load_pretrained

        actual = str(getattr(scgpt, "__version__", "unknown"))
        if actual != SCGPT_VERSION:
            raise ScGPTInputError(
                f"scGPT version mismatch: expected {SCGPT_VERSION}, observed {actual}"
            )
        components = {
            "transformer_model": TransformerModel,
            "binning": binning,
            "gene_vocab": GeneVocab,
            "load_pretrained": load_pretrained,
        }
        source_bindings: dict[str, Mapping[str, str]] = {}
        for name, component in components.items():
            source = Path(inspect.getsourcefile(component) or "").resolve()
            if not source.is_file():
                raise ScGPTInputError(f"cannot bind installed scGPT {name} source")
            source_bindings[name] = {
                "path": str(source),
                "sha256": sha256_file(source),
            }
        adapter_source = Path(__file__).resolve()
        return {
            "backend": "creed.sample_local_scgpt_transformer_v1",
            "model_mode": "eval",
            "maximum_length_policy": "keep_cls_plus_sample_local_gene_subset",
            "token_sampler": SAMPLE_LOCAL_TOKEN_SAMPLER,
            "value_binning": SAMPLE_LOCAL_BINNING,
            "sample_seed_schema": SAMPLE_LOCAL_SEED_SCHEMA,
            "sample_seed_inputs": ["rng_domain", "ensemble_seed", "exact_sample_id"],
            "ensemble_axis": "sample_local_gene_selection_and_binning_tie_seed",
            "row_tokenization_dependency": "same_sample_only",
            "gpu_batching_role": "execution_only_no_cross_sample_operator",
            "inference_precision": "float32_no_autocast",
            "scgpt_version": actual,
            "scgpt_module": str(Path(scgpt.__file__).resolve()),
            "official_component_sources": source_bindings,
            "adapter_source": str(adapter_source),
            "adapter_source_sha256": sha256_file(adapter_source),
            "torch_version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "model_dir": str(model_dir.resolve()),
            "sample_local_invariance_audits": list(self._invariance_audits),
            "active_checkpoint_coverage": list(self._checkpoint_coverages),
        }

    def embed(
        self,
        values: np.ndarray,
        gene_ids: tuple[str, ...],
        sample_ids: tuple[str, ...],
        model_dir: Path,
        config: ScGPTEmbeddingConfig,
        seed: int,
    ) -> np.ndarray:
        import torch
        from scgpt.model import TransformerModel
        from scgpt.tokenizer import GeneVocab
        from scgpt.utils import load_pretrained

        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise ScGPTInputError("CUDA was required but is unavailable")
        device = torch.device(config.device)
        # Initialization must not vary across tokenization ensemble members.
        # Any inactive randomly initialized task head is irrelevant to _encode;
        # every active _encode parameter is required below to come from the
        # frozen checkpoint with an exact shape match.
        torch.manual_seed(SCGPT_MODEL_INITIALIZATION_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SCGPT_MODEL_INITIALIZATION_SEED)

        model_dir = Path(model_dir)
        with (model_dir / "args.json").open("r", encoding="utf-8") as handle:
            model_configs = json.load(handle)
        vocab = GeneVocab.from_file(model_dir / "vocab.json")
        for token in ("<pad>", "<cls>", "<eoc>"):
            if token not in vocab:
                vocab.append_token(token)
        vocab.set_default_index(vocab["<pad>"])
        if any(gene not in vocab for gene in gene_ids):
            raise ScGPTInputError("prepared scGPT gene schema left the frozen vocabulary")
        vocabulary_gene_ids = np.asarray(vocab(list(gene_ids)), dtype=np.int64)
        model = TransformerModel(
            ntoken=len(vocab),
            d_model=model_configs["embsize"],
            nhead=model_configs["nheads"],
            d_hid=model_configs["d_hid"],
            nlayers=model_configs["nlayers"],
            nlayers_cls=model_configs["n_layers_cls"],
            n_cls=1,
            vocab=vocab,
            dropout=model_configs["dropout"],
            pad_token=model_configs["pad_token"],
            pad_value=model_configs["pad_value"],
            do_mvc=True,
            do_dab=False,
            use_batch_labels=False,
            domain_spec_batchnorm=False,
            explicit_zero_prob=False,
            use_fast_transformer=config.use_fast_transformer,
            fast_transformer_backend="flash",
            pre_norm=False,
        )
        if (
            getattr(model, "dsbn", None) is not None
            or getattr(model, "bn", None) is not None
        ):
            raise ScGPTInputError("sample-local scGPT inference forbids batch normalization")
        checkpoint = torch.load(model_dir / "best_model.pt", map_location=device)
        normalized_checkpoint = {
            key.replace("Wqkv.", "in_proj_"): value
            for key, value in checkpoint.items()
        }
        model_state = model.state_dict()
        active_keys = tuple(
            sorted(
                key
                for key in model_state
                if key.startswith(SCGPT_ACTIVE_INFERENCE_PREFIXES)
            )
        )
        missing = tuple(key for key in active_keys if key not in normalized_checkpoint)
        mismatched = tuple(
            key
            for key in active_keys
            if key in normalized_checkpoint
            and tuple(normalized_checkpoint[key].shape) != tuple(model_state[key].shape)
        )
        if missing or mismatched:
            raise ScGPTInputError(
                "frozen checkpoint does not cover every active scGPT _encode parameter"
            )
        coverage = {
            "ensemble_seed": int(seed),
            "model_initialization_seed": SCGPT_MODEL_INITIALIZATION_SEED,
            "active_prefixes": list(SCGPT_ACTIVE_INFERENCE_PREFIXES),
            "active_parameter_tensor_count": len(active_keys),
            "loaded_active_parameter_tensor_count": len(active_keys),
            "missing_active_parameter_tensor_count": 0,
            "shape_mismatched_active_parameter_tensor_count": 0,
            "active_parameter_keys_sha256": identifier_sequence_sha256(active_keys),
            "wqkv_to_in_proj_normalization_applied": True,
        }
        self._checkpoint_coverages.append(coverage)
        load_pretrained(model, checkpoint, verbose=False)
        model.to(device)
        model.eval()

        matrix = np.asarray(values, dtype=np.float32)
        token_genes = np.full(
            (len(sample_ids), config.max_length),
            int(vocab[model_configs["pad_token"]]),
            dtype=np.int64,
        )
        token_values = np.full(
            (len(sample_ids), config.max_length),
            float(model_configs["pad_value"]),
            dtype=np.float32,
        )
        token_genes[:, 0] = int(vocab["<cls>"])
        token_audits: list[Mapping[str, Any]] = []
        for row_index, sample_id in enumerate(sample_ids):
            genes, expressions, token_audit = sample_local_tokenize_row(
                matrix[row_index],
                vocabulary_gene_ids,
                sample_id=sample_id,
                ensemble_seed=seed,
                max_length=config.max_length,
                cls_token_id=int(vocab["<cls>"]),
                pad_token_id=int(vocab[model_configs["pad_token"]]),
                pad_value=float(model_configs["pad_value"]),
            )
            token_genes[row_index] = genes
            token_values[row_index] = expressions
            token_audits.append(token_audit)

        # Prove the complete pre-GPU token row is independent of global RNG,
        # sample order, and companion values before any batched model call.
        target = 0
        if len(sample_ids) > 1:
            companion = matrix[1].copy()
            companion[companion > 0.0] *= 1.5
            sample_local_tokenize_row(
                companion,
                vocabulary_gene_ids,
                sample_id=sample_ids[1],
                ensemble_seed=seed,
                max_length=config.max_length,
                cls_token_id=int(vocab["<cls>"]),
                pad_token_id=int(vocab[model_configs["pad_token"]]),
                pad_value=float(model_configs["pad_value"]),
            )
        audited_genes, audited_values, audited_tokens = sample_local_tokenize_row(
            matrix[target],
            vocabulary_gene_ids,
            sample_id=sample_ids[target],
            ensemble_seed=seed,
            max_length=config.max_length,
            cls_token_id=int(vocab["<cls>"]),
            pad_token_id=int(vocab[model_configs["pad_token"]]),
            pad_value=float(model_configs["pad_value"]),
        )
        token_exact = bool(
            np.array_equal(audited_genes, token_genes[target])
            and np.array_equal(audited_values, token_values[target])
            and audited_tokens == token_audits[target]
        )
        if not token_exact:
            raise ScGPTInputError("pre-GPU sample-local token invariance audit failed")

        def infer(gene_batch: np.ndarray, value_batch: np.ndarray) -> np.ndarray:
            output = np.empty(
                (len(gene_batch), int(model_configs["embsize"])), dtype=np.float32
            )
            # Full float32 avoids batch-shape-dependent autocast kernel choices
            # that can otherwise obscure the row-local dependency audit.
            with torch.no_grad():
                for start in range(0, len(gene_batch), config.batch_size):
                    stop = min(start + config.batch_size, len(gene_batch))
                    genes_tensor = torch.from_numpy(gene_batch[start:stop]).to(device)
                    values_tensor = torch.from_numpy(value_batch[start:stop]).to(device)
                    padding = genes_tensor.eq(int(vocab[model_configs["pad_token"]]))
                    encoded = model._encode(
                        genes_tensor,
                        values_tensor,
                        src_key_padding_mask=padding,
                        batch_labels=None,
                    )[:, 0, :]
                    output[start:stop] = encoded.float().cpu().numpy()
            norms = np.linalg.norm(output, axis=1, keepdims=True)
            if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
                raise ScGPTInputError("scGPT returned a zero or invalid embedding")
            return output / norms

        result = infer(token_genes, token_values)
        if not np.all(np.isfinite(result)):
            raise ScGPTInputError("scGPT returned non-finite embeddings")

        # A small formal runtime audit exercises the row-locality claim through
        # the actual loaded model: the same target tokens are evaluated alone,
        # under reordered companions, and after companion-only perturbations.
        alone = infer(token_genes[target : target + 1], token_values[target : target + 1])[0]
        companions = [index for index in range(1, min(len(sample_ids), 3))]
        context_indices = companions[:1] + [target] + companions[1:]
        context_genes = token_genes[context_indices].copy()
        context_values = token_values[context_indices].copy()
        target_position = context_indices.index(target)
        reordered = infer(context_genes, context_values)[target_position]
        for position in range(len(context_indices)):
            if position != target_position:
                context_values[position, 1:] = np.roll(
                    context_values[position, 1:], position + 1
                )
        perturbed = infer(context_genes, context_values)[target_position]
        differences = {
            "main_vs_alone_max_abs": float(np.max(np.abs(result[target] - alone))),
            "alone_vs_reordered_max_abs": float(np.max(np.abs(alone - reordered))),
            "alone_vs_companion_perturbed_max_abs": float(
                np.max(np.abs(alone - perturbed))
            ),
        }
        tolerance = 1e-5
        if any(value > tolerance for value in differences.values()):
            raise ScGPTInputError(
                "sample-local runtime invariance audit failed: "
                + json.dumps(differences, sort_keys=True)
            )
        self._invariance_audits.append(
            {
                "ensemble_seed": int(seed),
                "target_sample_id": sample_ids[target],
                "absolute_tolerance": tolerance,
                "passed": True,
                "pre_gpu_token_arrays_exact_after_repeat_and_companion_perturbation": (
                    token_exact
                ),
                **differences,
                **token_audits[target],
            }
        )
        return result


def embed_scgpt_ensemble(
    prepared: PreparedScGPTInput,
    contract: ScGPTModelContract,
    config: ScGPTEmbeddingConfig,
    backend: SampleEmbeddingBackend,
) -> EmbeddingEnsemble:
    metadata_before = dict(backend.metadata(contract.model_dir))
    dynamic_metadata = {
        "sample_local_invariance_audits",
        "active_checkpoint_coverage",
    }
    static_metadata_before = {
        key: value
        for key, value in metadata_before.items()
        if key not in dynamic_metadata
    }
    members = []
    expected_dimension = int(contract.args["embsize"])
    for seed in config.seeds:
        embedding = np.asarray(
            backend.embed(
                prepared.values,
                prepared.gene_ids,
                prepared.sample_ids,
                contract.model_dir,
                config,
                seed,
            ),
            dtype=np.float32,
        )
        expected_shape = (len(prepared.sample_ids), expected_dimension)
        if embedding.shape != expected_shape:
            raise ScGPTInputError(
                f"backend returned {embedding.shape}; expected {expected_shape}"
            )
        if not np.all(np.isfinite(embedding)):
            raise ScGPTInputError("backend returned non-finite embeddings")
        members.append(embedding)
    # Runtime row-locality audits are populated by the production backend while
    # embedding, so metadata is captured only after all ensemble members finish.
    metadata = dict(backend.metadata(contract.model_dir))
    static_metadata_after = {
        key: value for key, value in metadata.items() if key not in dynamic_metadata
    }
    if static_metadata_after != static_metadata_before:
        raise ScGPTInputError("scGPT backend identity changed during inference")
    static_identity_sha256 = canonical_json_sha256(static_metadata_before)
    metadata["static_backend_identity_sha256"] = static_identity_sha256
    metadata["static_backend_identity_verified_before_and_after"] = True
    member_array = np.stack(members)
    mean = member_array.mean(axis=0, dtype=np.float64)
    norms = np.linalg.norm(mean, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ScGPTInputError("ensemble mean contains a zero or invalid sample vector")
    mean = (mean / norms).astype(np.float32)
    deviations = member_array.astype(np.float64) - mean[np.newaxis, :, :]
    uncertainty = np.sqrt(np.mean(np.sum(deviations**2, axis=2), axis=0))
    return EmbeddingEnsemble(
        sample_ids=prepared.sample_ids,
        member_embeddings=member_array,
        mean_embedding=mean,
        per_dimension_standard_deviation=member_array.std(axis=0, ddof=1),
        sample_l2_uncertainty=uncertainty.astype(np.float32),
        backend_metadata=metadata,
    )


def prepared_input_receipt(prepared: PreparedScGPTInput) -> dict[str, Any]:
    return {
        "source_sha256": prepared.source_sha256,
        "input_semantics": prepared.input_semantics,
        "sample_count": len(prepared.sample_ids),
        "sample_schema_sha256": prepared.sample_schema_sha256,
        "input_gene_count": prepared.input_gene_count,
        "input_gene_schema_sha256": prepared.input_gene_schema_sha256,
        "matched_gene_count": prepared.matched_gene_count,
        "matched_gene_schema_sha256": prepared.matched_gene_schema_sha256,
        "unmatched_gene_count": prepared.unmatched_gene_count,
        "missing_value_count_mapped_to_token_omission": prepared.missing_value_count,
        "observed_zero_count": prepared.observed_zero_count,
        "positive_gene_count_per_sample": {
            "minimum": prepared.positive_gene_count_min,
            "median": prepared.positive_gene_count_median,
            "maximum": prepared.positive_gene_count_max,
        },
    }

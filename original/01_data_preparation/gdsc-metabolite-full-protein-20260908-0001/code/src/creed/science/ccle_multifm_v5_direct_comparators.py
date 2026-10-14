"""Array-only direct comparators for the non-authorizing CREED V5 draft.

This module implements the three direct families declared by
``CCLE_MULTIFM_BIDIRECTIONAL_V5_DRAFT.json``:

* float64 RBF kernel ridge over the complete 60-schedule registry;
* independently scaled PLS2 over the complete four-component registry;
* a three-member, one-hidden-layer GELU MLP over the declared AdamW
  weight-decay registry, with a hidden width that meets the maximum
  inference-path parameter count of all 16 full-model candidates.

The public full-OOF entry points are synthetic scaffolds over the fixed
421-row axes. A generic analytic-only mask fixture can exercise RBF/PLS mask
grouping, but is not Stage C. Randomized reverse-mask MLP execution is blocked
until a child protocol freezes its mask and seed namespaces. No entry point
accepts evaluation labels, paths, partition names, authority flags, or retry
controls.

The module is intentionally producer-side and non-authorizing.  It does not
implement activation, persistence, receipts, an independent verifier,
scientific gates, or a claim API.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import random
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping, TypeAlias

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


class CCLEMultiFMV5DirectComparatorError(ValueError):
    """Raised when the V5 direct-comparator contract cannot be satisfied."""


Direction: TypeAlias = Literal["forward", "reverse"]
Family: TypeAlias = Literal["RBF_kernel_ridge", "PLS2", "capacity_matched_direct_MLP"]

PROTOCOL_ID = "CREED_CCLE_MULTIFM_BIDIRECTIONAL_V5_STAGE_A_421_FEASIBILITY_V1"
STAGE_A_ROW_COUNT = 421
METABOLITE_COUNT = 225
PROTEIN_COUNT = 584
STAGE_A_FOLD_SEED = 2663416544
STAGE_A_FOLDS = 5

RBF_BANDWIDTH_MULTIPLIERS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
RBF_RIDGE_ALPHAS: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
RBF_SOURCE_STANDARDIZE: tuple[bool, ...] = (False, True)
PLS2_COMPONENTS: tuple[int, ...] = (4, 8, 16, 32)
DIRECT_MLP_WEIGHT_DECAYS: tuple[float, ...] = (0.00001, 0.0001, 0.001)
DIRECT_MLP_MEMBER_ROOTS: tuple[int, ...] = (
    328727162,
    1476026374,
    4034537809,
)
EARLY_STOP_MEMBER_ROOTS: tuple[int, ...] = (
    1923259244,
    2948505754,
    2292500555,
)


def _exact_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CCLEMultiFMV5DirectComparatorError(
            f"{label} must be an exact integer >= {minimum}"
        )
    return value


def _exact_float(value: object, label: str) -> float:
    if type(value) is not float or not np.isfinite(value):
        raise CCLEMultiFMV5DirectComparatorError(
            f"{label} must be an exact finite float"
        )
    return value


def _direction(value: object) -> Direction:
    if type(value) is not str or value not in {"forward", "reverse"}:
        raise CCLEMultiFMV5DirectComparatorError(
            "direction must be exactly 'forward' or 'reverse'"
        )
    return value  # type: ignore[return-value]


def _family(value: object) -> Family:
    allowed = {"RBF_kernel_ridge", "PLS2", "capacity_matched_direct_MLP"}
    if type(value) is not str or value not in allowed:
        raise CCLEMultiFMV5DirectComparatorError(
            "family is not in the fixed V5 direct-family registry"
        )
    return value  # type: ignore[return-value]


def _native_little_f64(dtype: np.dtype[object]) -> bool:
    return dtype == np.dtype("<f8") and dtype.isnative


def _native_little_i64(dtype: np.dtype[object]) -> bool:
    return dtype == np.dtype("<i8") and dtype.isnative


def _require_f64_matrix(
    value: object,
    label: str,
    *,
    rows: int | None = None,
    columns: int | None = None,
    minimum_rows: int = 1,
) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise CCLEMultiFMV5DirectComparatorError(
            f"{label} must be an exact numpy.ndarray"
        )
    matrix = value
    if not _native_little_f64(matrix.dtype):
        raise CCLEMultiFMV5DirectComparatorError(
            f"{label} must have native little-endian float64 dtype"
        )
    if not matrix.flags.c_contiguous or matrix.ndim != 2:
        raise CCLEMultiFMV5DirectComparatorError(
            f"{label} must be a C-contiguous two-dimensional matrix"
        )
    if matrix.shape[0] < minimum_rows or matrix.shape[1] < 1:
        raise CCLEMultiFMV5DirectComparatorError(f"{label} has an invalid shape")
    if rows is not None and matrix.shape[0] != rows:
        raise CCLEMultiFMV5DirectComparatorError(
            f"{label} must contain exactly {rows} rows"
        )
    if columns is not None and matrix.shape[1] != columns:
        raise CCLEMultiFMV5DirectComparatorError(
            f"{label} must contain exactly {columns} columns"
        )
    if not np.all(np.isfinite(matrix)):
        raise CCLEMultiFMV5DirectComparatorError(f"{label} must be entirely finite")
    return matrix


def _require_masked_f64_matrix(
    value: object,
    masks: object,
    label: str,
    *,
    columns: int,
) -> tuple[np.ndarray, np.ndarray]:
    if type(value) is not np.ndarray:
        raise CCLEMultiFMV5DirectComparatorError(
            f"{label} must be an exact numpy.ndarray"
        )
    matrix = value
    if (
        not _native_little_f64(matrix.dtype)
        or not matrix.flags.c_contiguous
        or matrix.ndim != 2
        or matrix.shape[0] < 1
        or matrix.shape[1] != columns
    ):
        raise CCLEMultiFMV5DirectComparatorError(
            f"{label} must be C-contiguous little-endian float64 with fixed width"
        )
    if np.any(np.isinf(matrix)):
        raise CCLEMultiFMV5DirectComparatorError(f"{label} contains infinity")
    if type(masks) is not np.ndarray:
        raise CCLEMultiFMV5DirectComparatorError(
            "source_masks must be an exact numpy.ndarray"
        )
    mask = masks
    if (
        mask.dtype != np.bool_
        or mask.ndim != 2
        or not mask.flags.c_contiguous
        or mask.shape != matrix.shape
    ):
        raise CCLEMultiFMV5DirectComparatorError(
            "source_masks must be a C-contiguous bool matrix matching source rows"
        )
    if not np.array_equal(mask, np.isfinite(matrix)):
        raise CCLEMultiFMV5DirectComparatorError(
            "source_masks must equal the exact finite-value mask"
        )
    if np.any(np.sum(mask, axis=1) == 0):
        raise CCLEMultiFMV5DirectComparatorError("an empty reverse source mask is STOP")
    return matrix, mask


def _readonly(
    value: np.ndarray, *, dtype: np.dtype[object] | None = None
) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _validate_pair(source: object, target: object) -> tuple[np.ndarray, np.ndarray]:
    x = _require_f64_matrix(source, "source", rows=STAGE_A_ROW_COUNT)
    y = _require_f64_matrix(target, "target", rows=STAGE_A_ROW_COUNT)
    if x.shape[0] != y.shape[0]:  # defensive; exact rows already bind this
        raise CCLEMultiFMV5DirectComparatorError("source/target sample axes differ")
    return x, y


def _validate_directional_widths(
    source: np.ndarray, target: np.ndarray, direction: Direction
) -> None:
    expected = (
        (METABOLITE_COUNT, PROTEIN_COUNT)
        if direction == "forward"
        else (PROTEIN_COUNT, METABOLITE_COUNT)
    )
    if source.shape[1] != expected[0] or target.shape[1] != expected[1]:
        raise CCLEMultiFMV5DirectComparatorError(
            "endpoint widths do not match the requested fixed direction"
        )


def full_candidate_ids() -> tuple[str, ...]:
    """Return all 16 V5 candidates in the protocol's normative order."""

    result: list[str] = []
    for latent_slots in (4, 8):
        for dimension in (32, 64):
            for teacher_token in ("t010", "t050"):
                for dropout_token in ("q000", "q005"):
                    result.append(
                        f"v5-k{latent_slots:02d}-d{dimension:03d}-"
                        f"{teacher_token}-{dropout_token}"
                    )
    return tuple(result)


def _check_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CCLEMultiFMV5DirectComparatorError(
            f"{label} must be one lowercase SHA-256"
        )
    return value


def _canonical_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CCLEMultiFMV5DirectComparatorError(
            "parameter inventory is not canonical JSON"
        ) from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CCLEMultiFMV5DirectComparatorError(
            "value is not canonical-JSON serializable"
        ) from exc


@dataclass(frozen=True, slots=True)
class CoreNamedParameterInventory:
    """The five protocol-relevant fields from one exact-core named receipt."""

    name: str
    shape: tuple[int, ...]
    numel: int
    inference_path: bool
    training_only: bool

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name or "|" in self.name:
            raise CCLEMultiFMV5DirectComparatorError(
                "named parameter has an invalid canonical name"
            )
        if type(self.shape) is not tuple or not self.shape:
            raise CCLEMultiFMV5DirectComparatorError(
                "named parameter shape must be a nonempty tuple"
            )
        product = 1
        for dimension in self.shape:
            product *= _exact_int(dimension, "parameter shape dimension", minimum=1)
        count = _exact_int(self.numel, "named parameter numel", minimum=1)
        if product != count:
            raise CCLEMultiFMV5DirectComparatorError(
                "named parameter shape/numel mismatch"
            )
        if (
            type(self.inference_path) is not bool
            or type(self.training_only) is not bool
        ):
            raise CCLEMultiFMV5DirectComparatorError(
                "parameter path flags must be exact booleans"
            )
        if self.inference_path == self.training_only:
            raise CCLEMultiFMV5DirectComparatorError(
                "parameter must belong to exactly one inference/training-only path"
            )
        if self.training_only != self.name.startswith("teacher_head."):
            raise CCLEMultiFMV5DirectComparatorError(
                "only the teacher head may be training-only"
            )

    def canonical_object(self) -> dict[str, object]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "numel": self.numel,
            "inference_path": self.inference_path,
            "training_only": self.training_only,
        }


@dataclass(frozen=True, slots=True)
class CoreCandidateParameterInventory:
    candidate_id: str
    latent_slots: int
    model_dimension: int
    teacher_loss_weight: float
    source_token_dropout: float
    total: int
    trainable: int
    inference_path: int
    training_only: int
    named: tuple[CoreNamedParameterInventory, ...]

    def __post_init__(self) -> None:
        if self.candidate_id not in full_candidate_ids():
            raise CCLEMultiFMV5DirectComparatorError(
                "candidate inventory ID is outside the exact 16-candidate registry"
            )
        if self.latent_slots not in (4, 8) or self.model_dimension not in (32, 64):
            raise CCLEMultiFMV5DirectComparatorError(
                "candidate inventory architecture is outside the protocol grid"
            )
        if self.teacher_loss_weight not in (0.1, 0.5):
            raise CCLEMultiFMV5DirectComparatorError(
                "candidate inventory teacher weight changed"
            )
        if self.source_token_dropout not in (0.0, 0.05):
            raise CCLEMultiFMV5DirectComparatorError(
                "candidate inventory dropout changed"
            )
        expected_id = (
            f"v5-k{self.latent_slots:02d}-d{self.model_dimension:03d}-"
            f"{'t010' if self.teacher_loss_weight == 0.1 else 't050'}-"
            f"{'q000' if self.source_token_dropout == 0.0 else 'q005'}"
        )
        if self.candidate_id != expected_id:
            raise CCLEMultiFMV5DirectComparatorError(
                "candidate ID/configuration binding is inconsistent"
            )
        if type(self.named) is not tuple or not self.named:
            raise CCLEMultiFMV5DirectComparatorError(
                "candidate has no exact-core named parameter inventory"
            )
        if any(type(item) is not CoreNamedParameterInventory for item in self.named):
            raise CCLEMultiFMV5DirectComparatorError(
                "candidate named inventory has a foreign record"
            )
        names = tuple(item.name for item in self.named)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise CCLEMultiFMV5DirectComparatorError(
                "named parameter inventory must be unique and sorted"
            )
        required_exact = {
            "metabolite_identity",
            "protein_identity_projection.weight",
            "teacher_head.weight",
            "teacher_head.bias",
        }
        required_prefixes = (
            "metabolite_encoder.",
            "protein_encoder.",
            "shared_pma.",
            "shared_latent_core.",
            "metabolite_decoder.",
            "protein_decoder.",
        )
        if not required_exact.issubset(names) or any(
            not any(name.startswith(prefix) for name in names)
            for prefix in required_prefixes
        ):
            raise CCLEMultiFMV5DirectComparatorError(
                "candidate inventory is not the exact core parameter graph"
            )
        total = sum(item.numel for item in self.named)
        inference = sum(item.numel for item in self.named if item.inference_path)
        training = sum(item.numel for item in self.named if item.training_only)
        for label, value in (
            ("total", self.total),
            ("trainable", self.trainable),
            ("inference_path", self.inference_path),
            ("training_only", self.training_only),
        ):
            _exact_int(value, label, minimum=1)
        if (
            self.total != total
            or self.trainable != total
            or self.inference_path != inference
            or self.training_only != training
            or total != inference + training
        ):
            raise CCLEMultiFMV5DirectComparatorError(
                "candidate aggregate counts differ from its named inventory"
            )

    def canonical_object(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "latent_slots": self.latent_slots,
            "model_dimension": self.model_dimension,
            "teacher_loss_weight": self.teacher_loss_weight,
            "source_token_dropout": self.source_token_dropout,
            "total": self.total,
            "trainable": self.trainable,
            "inference_path": self.inference_path,
            "training_only": self.training_only,
            "named": [item.canonical_object() for item in self.named],
        }


@dataclass(frozen=True, slots=True)
class StaticVerifierBinding:
    inventory_sha256: str
    source_archive_sha256: str
    independent_verifier_receipt_sha256: str
    production_runtime_approval_sha256: str
    scientific_claim_authority: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        values = tuple(
            _check_sha256(getattr(self, label), label)
            for label in (
                "inventory_sha256",
                "source_archive_sha256",
                "independent_verifier_receipt_sha256",
                "production_runtime_approval_sha256",
            )
        )
        if len(set(values)) != len(values):
            raise CCLEMultiFMV5DirectComparatorError(
                "static/runtime binding hashes must not alias"
            )


@dataclass(frozen=True, slots=True)
class FullCandidateParameterBudget:
    """Exact-core inventory; never authoritative merely by construction."""

    candidates: tuple[CoreCandidateParameterInventory, ...]
    inventory_sha256: str
    static_verifier_binding: StaticVerifierBinding | None = None
    scientific_claim_authority: bool = field(init=False, default=False)
    execution_authority: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if (
            type(self.candidates) is not tuple
            or tuple(item.candidate_id for item in self.candidates)
            != full_candidate_ids()
        ):
            raise CCLEMultiFMV5DirectComparatorError(
                "budget must contain all 16 exact-core candidates in normative order"
            )
        if any(
            type(item) is not CoreCandidateParameterInventory
            for item in self.candidates
        ):
            raise CCLEMultiFMV5DirectComparatorError(
                "budget contains a foreign candidate inventory"
            )
        canonical = {
            "schema": "creed.ccle-multifm-v5-exact-core-inventory.v1",
            "protocol_id": PROTOCOL_ID,
            "candidates": [item.canonical_object() for item in self.candidates],
        }
        if _check_sha256(
            self.inventory_sha256, "inventory_sha256"
        ) != _canonical_sha256(canonical):
            raise CCLEMultiFMV5DirectComparatorError(
                "inventory SHA-256 does not bind the exact named inventory"
            )
        grouped: dict[tuple[int, int], tuple[CoreNamedParameterInventory, ...]] = {}
        for candidate in self.candidates:
            key = (candidate.latent_slots, candidate.model_dimension)
            prior = grouped.setdefault(key, candidate.named)
            if candidate.named != prior:
                raise CCLEMultiFMV5DirectComparatorError(
                    "same-shape candidates have inconsistent named inventories"
                )
        if self.static_verifier_binding is not None:
            if type(self.static_verifier_binding) is not StaticVerifierBinding:
                raise CCLEMultiFMV5DirectComparatorError(
                    "static verifier binding has a foreign type"
                )
            if self.static_verifier_binding.inventory_sha256 != self.inventory_sha256:
                raise CCLEMultiFMV5DirectComparatorError(
                    "static verifier binding targets another inventory"
                )

    @property
    def candidate_totals(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (item.candidate_id, item.inference_path) for item in self.candidates
        )

    @property
    def maximum(self) -> int:
        return max(item.inference_path for item in self.candidates)

    @property
    def production_static_binding_present(self) -> bool:
        return self.static_verifier_binding is not None


def _budget_from_candidates(
    candidates: tuple[CoreCandidateParameterInventory, ...],
    *,
    binding: StaticVerifierBinding | None = None,
) -> FullCandidateParameterBudget:
    canonical = {
        "schema": "creed.ccle-multifm-v5-exact-core-inventory.v1",
        "protocol_id": PROTOCOL_ID,
        "candidates": [item.canonical_object() for item in candidates],
    }
    return FullCandidateParameterBudget(
        candidates=candidates,
        inventory_sha256=_canonical_sha256(canonical),
        static_verifier_binding=binding,
    )


def export_exact_core_full_candidate_parameter_budget() -> FullCandidateParameterBudget:
    """Build all 16 full graphs and export their real exact-core receipts.

    This performs static graph construction only.  It opens no endpoint matrix,
    identifier, or model weight asset and grants no production authority.
    """

    from creed.science import ccle_multifm_bidir_v5 as core

    torch = _import_torch()
    effective_esm = np.zeros((PROTEIN_COUNT, 1280), dtype=np.float64)
    candidates: list[CoreCandidateParameterInventory] = []
    candidate_index = 0
    for latent_slots in (4, 8):
        for model_dimension in (32, 64):
            for teacher_loss_weight in (0.1, 0.5):
                for source_token_dropout in (0.0, 0.05):
                    config = core.MultiFMBidirV5Config(
                        latent_slots=latent_slots,
                        model_dimension=model_dimension,
                        teacher_loss_weight=teacher_loss_weight,
                        source_token_dropout=source_token_dropout,
                        epochs=1,
                        seed=910000 + candidate_index,
                        intervention_seed=920000 + candidate_index,
                    )
                    model = core._build_model(
                        torch,
                        metabolite_count=METABOLITE_COUNT,
                        protein_count=PROTEIN_COUNT,
                        esm_feature_count=1280,
                        effective_esm=effective_esm,
                        config=config,
                        control="full",
                    )
                    initialization_seeds = core._initialize_model(
                        torch, model, config.seed
                    )
                    count = core._parameter_count(model, initialization_seeds)
                    if type(count) is not core.ParameterCount or any(
                        type(item) is not core.NamedParameterReceipt
                        for item in count.named
                    ):
                        raise CCLEMultiFMV5DirectComparatorError(
                            "exact core returned a foreign parameter receipt"
                        )
                    named = tuple(
                        CoreNamedParameterInventory(
                            name=item.name,
                            shape=item.shape,
                            numel=item.numel,
                            inference_path=item.inference_path,
                            training_only=item.training_only,
                        )
                        for item in count.named
                    )
                    candidate_id = full_candidate_ids()[candidate_index]
                    candidates.append(
                        CoreCandidateParameterInventory(
                            candidate_id=candidate_id,
                            latent_slots=latent_slots,
                            model_dimension=model_dimension,
                            teacher_loss_weight=teacher_loss_weight,
                            source_token_dropout=source_token_dropout,
                            total=count.total,
                            trainable=count.trainable,
                            inference_path=count.inference_path,
                            training_only=count.training_only,
                            named=named,
                        )
                    )
                    candidate_index += 1
    return _budget_from_candidates(tuple(candidates))


def bind_parameter_budget_to_static_verifier(
    budget: FullCandidateParameterBudget,
    binding: StaticVerifierBinding,
) -> FullCandidateParameterBudget:
    if type(budget) is not FullCandidateParameterBudget:
        raise CCLEMultiFMV5DirectComparatorError("budget has a foreign type")
    if type(binding) is not StaticVerifierBinding:
        raise CCLEMultiFMV5DirectComparatorError("binding has a foreign type")
    if binding.inventory_sha256 != budget.inventory_sha256:
        raise CCLEMultiFMV5DirectComparatorError(
            "static verifier did not bind this exact inventory"
        )
    return _budget_from_candidates(budget.candidates, binding=binding)


def direct_mlp_parameter_count(
    input_count: int, output_count: int, hidden_width: int
) -> int:
    inputs = _exact_int(input_count, "input_count", minimum=1)
    outputs = _exact_int(output_count, "output_count", minimum=1)
    hidden = _exact_int(hidden_width, "hidden_width", minimum=1)
    return hidden * (inputs + outputs + 1) + outputs


def minimum_capacity_hidden_width(
    input_count: int, output_count: int, budget: FullCandidateParameterBudget
) -> int:
    """Return the smallest hidden width whose path count is at least B_max."""

    if type(budget) is not FullCandidateParameterBudget:
        raise CCLEMultiFMV5DirectComparatorError(
            "budget must be a validated FullCandidateParameterBudget"
        )
    inputs = _exact_int(input_count, "input_count", minimum=1)
    outputs = _exact_int(output_count, "output_count", minimum=1)
    required = _exact_int(budget.maximum, "B_max", minimum=1)
    denominator = inputs + outputs + 1
    hidden = max(1, math.ceil((required - outputs) / denominator))
    count = direct_mlp_parameter_count(inputs, outputs, hidden)
    if count < required:
        raise CCLEMultiFMV5DirectComparatorError("hidden width undershot B_max")
    if (
        hidden > 1
        and direct_mlp_parameter_count(inputs, outputs, hidden - 1) >= required
    ):
        raise CCLEMultiFMV5DirectComparatorError("hidden width is not minimal")
    return hidden


@dataclass(frozen=True, order=True, slots=True)
class RBFSchedule:
    bandwidth_multiplier: float
    ridge_alpha: float
    source_standardize: bool

    def __post_init__(self) -> None:
        _exact_float(self.bandwidth_multiplier, "bandwidth_multiplier")
        _exact_float(self.ridge_alpha, "ridge_alpha")
        if self.bandwidth_multiplier not in RBF_BANDWIDTH_MULTIPLIERS:
            raise CCLEMultiFMV5DirectComparatorError(
                "RBF bandwidth is outside the fixed schedule"
            )
        if self.ridge_alpha not in RBF_RIDGE_ALPHAS:
            raise CCLEMultiFMV5DirectComparatorError(
                "RBF alpha is outside the fixed schedule"
            )
        if type(self.source_standardize) is not bool:
            raise CCLEMultiFMV5DirectComparatorError(
                "RBF source_standardize must be an exact bool"
            )


@dataclass(frozen=True, order=True, slots=True)
class PLS2Schedule:
    components: int

    def __post_init__(self) -> None:
        _exact_int(self.components, "PLS2 components", minimum=1)
        if self.components not in PLS2_COMPONENTS:
            raise CCLEMultiFMV5DirectComparatorError(
                "PLS2 components are outside the fixed schedule"
            )


@dataclass(frozen=True, order=True, slots=True)
class DirectMLPSchedule:
    weight_decay: float

    def __post_init__(self) -> None:
        _exact_float(self.weight_decay, "direct MLP weight_decay")
        if self.weight_decay not in DIRECT_MLP_WEIGHT_DECAYS:
            raise CCLEMultiFMV5DirectComparatorError(
                "direct MLP weight_decay is outside the fixed schedule"
            )


Schedule: TypeAlias = RBFSchedule | PLS2Schedule | DirectMLPSchedule


def rbf_schedule_registry() -> tuple[RBFSchedule, ...]:
    return tuple(
        RBFSchedule(multiplier, alpha, standardized)
        for multiplier in RBF_BANDWIDTH_MULTIPLIERS
        for alpha in RBF_RIDGE_ALPHAS
        for standardized in RBF_SOURCE_STANDARDIZE
    )


def pls2_schedule_registry() -> tuple[PLS2Schedule, ...]:
    return tuple(PLS2Schedule(components) for components in PLS2_COMPONENTS)


def direct_mlp_schedule_registry() -> tuple[DirectMLPSchedule, ...]:
    return tuple(
        DirectMLPSchedule(weight_decay) for weight_decay in DIRECT_MLP_WEIGHT_DECAYS
    )


def schedule_registry(family: Family) -> tuple[Schedule, ...]:
    checked = _family(family)
    if checked == "RBF_kernel_ridge":
        return rbf_schedule_registry()
    if checked == "PLS2":
        return pls2_schedule_registry()
    return direct_mlp_schedule_registry()


@dataclass(frozen=True, slots=True)
class StageAFold:
    fold: int
    fit_indices: tuple[int, ...]
    held_indices: tuple[int, ...]


def stage_a_folds() -> tuple[StageAFold, ...]:
    """Construct and independently validate the fixed five-fold registry."""

    positions = np.arange(STAGE_A_ROW_COUNT, dtype=np.int64)
    splitter = KFold(
        n_splits=STAGE_A_FOLDS,
        shuffle=True,
        random_state=STAGE_A_FOLD_SEED,
    )
    folds: list[StageAFold] = []
    held_count = np.zeros(STAGE_A_ROW_COUNT, dtype=np.int64)
    for fold, (fit, held) in enumerate(splitter.split(positions)):
        fit = np.asarray(fit, dtype=np.int64)
        held = np.asarray(held, dtype=np.int64)
        _validate_fold_indices(fit, held, fold)
        held_count[held] += 1
        folds.append(
            StageAFold(
                fold=fold,
                fit_indices=tuple(int(item) for item in fit),
                held_indices=tuple(int(item) for item in held),
            )
        )
    if not np.array_equal(held_count, np.ones(STAGE_A_ROW_COUNT, dtype=np.int64)):
        raise CCLEMultiFMV5DirectComparatorError(
            "fixed five-fold held coverage is not exactly one per row"
        )
    if tuple(len(item.held_indices) for item in folds) != (85, 84, 84, 84, 84):
        raise CCLEMultiFMV5DirectComparatorError("fixed fold sizes changed")
    return tuple(folds)


def _validate_fold_indices(fit: np.ndarray, held: np.ndarray, fold: int) -> None:
    _exact_int(fold, "fold", minimum=0)
    for value, label in ((fit, "fit_indices"), (held, "held_indices")):
        if (
            type(value) is not np.ndarray
            or not _native_little_i64(value.dtype)
            or value.ndim != 1
            or not value.flags.c_contiguous
            or value.size == 0
            or np.any(value < 0)
            or np.any(value >= STAGE_A_ROW_COUNT)
            or not np.all(value[1:] > value[:-1])
        ):
            raise CCLEMultiFMV5DirectComparatorError(
                f"{label} is not a canonical sorted int64 index vector"
            )
    if np.intersect1d(fit, held).size:
        raise CCLEMultiFMV5DirectComparatorError("fit and held rows overlap")
    combined = np.sort(np.concatenate((fit, held)))
    if not np.array_equal(combined, np.arange(STAGE_A_ROW_COUNT, dtype=np.int64)):
        raise CCLEMultiFMV5DirectComparatorError(
            "fit and held rows do not partition the fixed registry"
        )


def median_target_pearson(truth: object, prediction: object) -> float:
    """Compute the protocol's complete-panel float64 median Pearson metric."""

    y = _require_f64_matrix(truth, "truth", minimum_rows=3)
    p = _require_f64_matrix(prediction, "prediction", minimum_rows=3)
    if y.shape != p.shape:
        raise CCLEMultiFMV5DirectComparatorError("truth/prediction axes differ")
    y_centered = y - np.mean(y, axis=0, dtype=np.float64)
    p_centered = p - np.mean(p, axis=0, dtype=np.float64)
    y_ss = np.sum(y_centered * y_centered, axis=0, dtype=np.float64)
    p_ss = np.sum(p_centered * p_centered, axis=0, dtype=np.float64)
    denominator = np.sqrt(y_ss * p_ss)
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0.0):
        raise CCLEMultiFMV5DirectComparatorError(
            "a fixed target has nonpositive variance; target dropping is forbidden"
        )
    correlations = (
        np.sum(y_centered * p_centered, axis=0, dtype=np.float64) / denominator
    )
    if not np.all(np.isfinite(correlations)):
        raise CCLEMultiFMV5DirectComparatorError("target correlation is non-finite")
    metric = float(np.median(correlations))
    if not np.isfinite(metric):
        raise CCLEMultiFMV5DirectComparatorError("panel metric is non-finite")
    return metric


@dataclass(frozen=True, slots=True)
class RBFModel:
    schedule: RBFSchedule
    source_mean: np.ndarray | None
    source_scale: np.ndarray | None
    fit_source: np.ndarray
    target_mean: np.ndarray
    coefficients: np.ndarray
    base_median: float
    sigma2: float


def fit_rbf(source_fit: object, target_fit: object, schedule: RBFSchedule) -> RBFModel:
    """Fit the exact Cholesky-only float64 RBF comparator."""

    x = _require_f64_matrix(source_fit, "source_fit", minimum_rows=2)
    y = _require_f64_matrix(target_fit, "target_fit", minimum_rows=2)
    if x.shape[0] != y.shape[0]:
        raise CCLEMultiFMV5DirectComparatorError("RBF fit sample axes differ")
    if type(schedule) is not RBFSchedule:
        raise CCLEMultiFMV5DirectComparatorError("RBF schedule has the wrong type")
    if schedule.source_standardize:
        scaler = StandardScaler(with_mean=True, with_std=True, copy=True)
        transformed = scaler.fit_transform(x)
        source_mean = np.asarray(scaler.mean_, dtype=np.float64)
        source_scale = np.asarray(scaler.scale_, dtype=np.float64)
        if not np.all(np.isfinite(source_mean)) or not np.all(
            np.isfinite(source_scale)
        ):
            raise CCLEMultiFMV5DirectComparatorError("RBF source scale is non-finite")
        if np.any(source_scale <= 0.0):
            raise CCLEMultiFMV5DirectComparatorError("RBF source scale is nonpositive")
    else:
        transformed = np.array(x, dtype=np.float64, order="C", copy=True)
        source_mean = None
        source_scale = None
    transformed = np.ascontiguousarray(transformed, dtype=np.float64)
    distances = cdist(transformed, transformed, metric="sqeuclidean")
    if distances.dtype != np.float64 or not np.all(np.isfinite(distances)):
        raise CCLEMultiFMV5DirectComparatorError("RBF fit distances are non-finite")
    upper = distances[np.triu_indices(distances.shape[0], k=1)]
    positive = upper[np.isfinite(upper) & (upper > 0.0)]
    if positive.size == 0:
        raise CCLEMultiFMV5DirectComparatorError(
            "RBF has no positive finite upper-triangle training distance"
        )
    base_median = float(np.median(positive))
    sigma2 = float(schedule.bandwidth_multiplier * base_median)
    if not np.isfinite(base_median) or base_median <= 0.0:
        raise CCLEMultiFMV5DirectComparatorError("RBF base median is invalid")
    if not np.isfinite(sigma2) or sigma2 <= 0.0:
        raise CCLEMultiFMV5DirectComparatorError("RBF sigma2 is invalid")
    kernel = np.exp(-distances / (2.0 * sigma2))
    np.fill_diagonal(kernel, np.float64(1.0))
    kernel = 0.5 * (kernel + kernel.T)
    target_mean = np.mean(y, axis=0, dtype=np.float64)
    centered = y - target_mean
    system = kernel + schedule.ridge_alpha * np.eye(kernel.shape[0], dtype=np.float64)
    try:
        factor = cho_factor(
            system,
            lower=True,
            overwrite_a=False,
            check_finite=True,
        )
        coefficients = cho_solve(
            factor,
            centered,
            overwrite_b=False,
            check_finite=True,
        )
    except Exception as exc:
        raise CCLEMultiFMV5DirectComparatorError(
            "RBF Cholesky fit failed; no fallback is allowed"
        ) from exc
    if not np.all(np.isfinite(coefficients)):
        raise CCLEMultiFMV5DirectComparatorError("RBF coefficients are non-finite")
    return RBFModel(
        schedule=schedule,
        source_mean=None if source_mean is None else _readonly(source_mean),
        source_scale=None if source_scale is None else _readonly(source_scale),
        fit_source=_readonly(transformed),
        target_mean=_readonly(target_mean),
        coefficients=_readonly(np.ascontiguousarray(coefficients, dtype=np.float64)),
        base_median=base_median,
        sigma2=sigma2,
    )


def predict_rbf(model: RBFModel, source: object) -> np.ndarray:
    if type(model) is not RBFModel:
        raise CCLEMultiFMV5DirectComparatorError("model must be an exact RBFModel")
    x = _require_f64_matrix(source, "source", minimum_rows=1)
    if x.shape[1] != model.fit_source.shape[1]:
        raise CCLEMultiFMV5DirectComparatorError("RBF source width changed")
    if model.schedule.source_standardize:
        if model.source_mean is None or model.source_scale is None:
            raise CCLEMultiFMV5DirectComparatorError("RBF scaler state is incomplete")
        transformed = (x - model.source_mean) / model.source_scale
    else:
        if model.source_mean is not None or model.source_scale is not None:
            raise CCLEMultiFMV5DirectComparatorError("unexpected RBF scaler state")
        transformed = np.array(x, dtype=np.float64, order="C", copy=True)
    distances = cdist(
        np.ascontiguousarray(transformed, dtype=np.float64),
        model.fit_source,
        metric="sqeuclidean",
    )
    kernel = np.exp(-distances / (2.0 * model.sigma2))
    prediction = kernel @ model.coefficients + model.target_mean
    prediction = np.ascontiguousarray(prediction, dtype=np.float64)
    if prediction.shape != (x.shape[0], model.target_mean.shape[0]):
        raise CCLEMultiFMV5DirectComparatorError("RBF prediction axes changed")
    if not np.all(np.isfinite(prediction)):
        raise CCLEMultiFMV5DirectComparatorError("RBF prediction is non-finite")
    return _readonly(prediction)


@dataclass(frozen=True, slots=True)
class PLS2Model:
    schedule: PLS2Schedule
    source_scaler: StandardScaler
    target_scaler: StandardScaler
    estimator: PLSRegression
    n_iter: tuple[int, ...]
    source_width: int
    target_width: int


def fit_pls2(
    source_fit: object, target_fit: object, schedule: PLS2Schedule
) -> PLS2Model:
    """Fit PLS2 with independent current-fit X/Y scalers and strict convergence."""

    x = _require_f64_matrix(source_fit, "source_fit", minimum_rows=2)
    y = _require_f64_matrix(target_fit, "target_fit", minimum_rows=2)
    if x.shape[0] != y.shape[0]:
        raise CCLEMultiFMV5DirectComparatorError("PLS2 fit sample axes differ")
    if type(schedule) is not PLS2Schedule:
        raise CCLEMultiFMV5DirectComparatorError("PLS2 schedule has the wrong type")
    capacity = min(x.shape[0] - 1, x.shape[1])
    if schedule.components > capacity:
        raise CCLEMultiFMV5DirectComparatorError(
            "PLS2 component count exceeds current comparator-fit capacity"
        )
    x_scaler = StandardScaler(with_mean=True, with_std=True, copy=True).fit(x)
    y_scaler = StandardScaler(with_mean=True, with_std=True, copy=True).fit(y)
    x_scaled = np.ascontiguousarray(x_scaler.transform(x), dtype=np.float64)
    y_scaled = np.ascontiguousarray(y_scaler.transform(y), dtype=np.float64)
    if not np.all(np.isfinite(x_scaled)) or not np.all(np.isfinite(y_scaled)):
        raise CCLEMultiFMV5DirectComparatorError(
            "PLS2 scaled fit values are non-finite"
        )
    estimator = PLSRegression(
        n_components=schedule.components,
        scale=False,
        max_iter=500,
        tol=1e-6,
        copy=True,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        try:
            estimator.fit(x_scaled, y_scaled)
        except Exception as exc:
            raise CCLEMultiFMV5DirectComparatorError("PLS2 fit failed") from exc
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise CCLEMultiFMV5DirectComparatorError("PLS2 emitted ConvergenceWarning")
    raw_iterations = getattr(estimator, "n_iter_", None)
    if not isinstance(raw_iterations, (list, tuple, np.ndarray)):
        raise CCLEMultiFMV5DirectComparatorError("PLS2 n_iter_ is missing")
    n_iter = tuple(int(value) for value in raw_iterations)
    if len(n_iter) != schedule.components:
        raise CCLEMultiFMV5DirectComparatorError(
            "PLS2 n_iter_ must contain one integer per fitted component"
        )
    for raw, converted in zip(raw_iterations, n_iter, strict=True):
        if type(raw) is not int or raw != converted:
            raise CCLEMultiFMV5DirectComparatorError(
                "PLS2 n_iter_ must contain exact Python integers"
            )
        if converted < 1 or converted >= 500:
            raise CCLEMultiFMV5DirectComparatorError(
                "PLS2 convergence is ambiguous at max_iter"
            )
    score = float(estimator.score(x_scaled, y_scaled))
    if not np.isfinite(score):
        raise CCLEMultiFMV5DirectComparatorError("PLS2 fit score is non-finite")
    return PLS2Model(
        schedule=schedule,
        source_scaler=x_scaler,
        target_scaler=y_scaler,
        estimator=estimator,
        n_iter=n_iter,
        source_width=x.shape[1],
        target_width=y.shape[1],
    )


def predict_pls2(model: PLS2Model, source: object) -> np.ndarray:
    if type(model) is not PLS2Model:
        raise CCLEMultiFMV5DirectComparatorError("model must be an exact PLS2Model")
    x = _require_f64_matrix(source, "source", minimum_rows=1)
    if x.shape[1] != model.source_width:
        raise CCLEMultiFMV5DirectComparatorError("PLS2 source width changed")
    x_scaled = np.ascontiguousarray(model.source_scaler.transform(x), dtype=np.float64)
    try:
        prediction_scaled = model.estimator.predict(x_scaled, copy=True)
    except TypeError:  # sklearn versions before the explicit copy keyword
        prediction_scaled = model.estimator.predict(x_scaled)
    prediction_scaled = np.asarray(prediction_scaled, dtype=np.float64)
    if prediction_scaled.ndim == 1:
        prediction_scaled = prediction_scaled[:, None]
    prediction = model.target_scaler.inverse_transform(prediction_scaled)
    prediction = np.ascontiguousarray(prediction, dtype=np.float64)
    if prediction.shape != (x.shape[0], model.target_width):
        raise CCLEMultiFMV5DirectComparatorError("PLS2 prediction axes changed")
    if not np.all(np.isfinite(prediction)):
        raise CCLEMultiFMV5DirectComparatorError("PLS2 prediction is non-finite")
    return _readonly(prediction)


@dataclass(frozen=True, slots=True)
class _EndpointScale:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, value: np.ndarray) -> np.ndarray:
        result = (value - self.mean) / self.scale
        if not np.all(np.isfinite(result)):
            raise CCLEMultiFMV5DirectComparatorError(
                "direct MLP standardized values are non-finite"
            )
        return np.ascontiguousarray(result, dtype=np.float64)

    def inverse(self, value: np.ndarray) -> np.ndarray:
        result = value * self.scale + self.mean
        if not np.all(np.isfinite(result)):
            raise CCLEMultiFMV5DirectComparatorError(
                "direct MLP inverse-scaled prediction is non-finite"
            )
        return np.ascontiguousarray(result, dtype=np.float64)


def _fit_endpoint_scale(value: np.ndarray) -> _EndpointScale:
    mean = np.mean(value, axis=0, dtype=np.float64)
    centered = value - mean
    variance = np.mean(centered * centered, axis=0, dtype=np.float64)
    raw_scale = np.sqrt(variance)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(raw_scale)):
        raise CCLEMultiFMV5DirectComparatorError("endpoint scale is non-finite")
    scale = np.where(raw_scale < 1e-8, np.float64(1.0), raw_scale)
    if np.any(scale <= 0.0) or not np.all(np.isfinite(scale)):
        raise CCLEMultiFMV5DirectComparatorError("endpoint scale is nonpositive")
    return _EndpointScale(mean=_readonly(mean), scale=_readonly(scale))


def _derive_seed(root_seed: int, *tokens: object) -> int:
    root = _exact_int(root_seed, "root_seed", minimum=0)
    if root >= 2**63:
        raise CCLEMultiFMV5DirectComparatorError("root_seed must be below 2^63")
    encoded: list[str] = [str(root)]
    for token in tokens:
        if type(token) is int:
            if token < 0:
                raise CCLEMultiFMV5DirectComparatorError("seed index is negative")
            text = str(token)
        elif type(token) is str and token and "|" not in token:
            text = token
        else:
            raise CCLEMultiFMV5DirectComparatorError("invalid canonical seed token")
        encoded.append(text)
    digest = hashlib.sha256("|".join(encoded).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**63)


def canonical_schedule_id(
    schedule: Schedule, *, direction: Direction | None = None
) -> str:
    """Map a schedule to the Stage-A control-plane's canonical task token."""

    if type(schedule) is RBFSchedule:
        bandwidth_tokens = {
            0.25: "0p25",
            0.5: "0p50",
            1.0: "1p00",
            2.0: "2p00",
            4.0: "4p00",
        }
        alpha_tokens = {
            0.001: "0p001",
            0.01: "0p010",
            0.1: "0p100",
            1.0: "1p000",
            10.0: "10p000",
            100.0: "100p000",
        }
        return (
            f"rbf-b{bandwidth_tokens[schedule.bandwidth_multiplier]}-"
            f"a{alpha_tokens[schedule.ridge_alpha]}-"
            f"s{int(schedule.source_standardize)}"
        )
    if type(schedule) is PLS2Schedule:
        return f"pls-c{schedule.components:03d}"
    if type(schedule) is DirectMLPSchedule:
        if direction not in ("forward", "reverse"):
            raise CCLEMultiFMV5DirectComparatorError(
                "direct MLP canonical ID requires a direction"
            )
        tokens = {
            0.00001: "0p00001",
            0.0001: "0p00010",
            0.001: "0p00100",
        }
        inputs, outputs, hidden = (
            (225, 584, 666) if direction == "forward" else (584, 225, 667)
        )
        return (
            f"mlp-in{inputs:04d}-out{outputs:04d}-h{hidden:06d}-"
            f"wd{tokens[schedule.weight_decay]}"
        )
    raise CCLEMultiFMV5DirectComparatorError("unknown schedule type")


def canonical_schedule_settings(
    schedule: Schedule, *, direction: Direction | None = None
) -> tuple[tuple[str, object], ...]:
    """Return the canonical control-plane settings tuple for one schedule."""

    if type(schedule) is RBFSchedule:
        return (
            ("bandwidth_multiplier", schedule.bandwidth_multiplier),
            ("ridge_alpha", schedule.ridge_alpha),
            ("source_standardize", schedule.source_standardize),
        )
    if type(schedule) is PLS2Schedule:
        return (("components", schedule.components),)
    if type(schedule) is DirectMLPSchedule:
        if direction not in ("forward", "reverse"):
            raise CCLEMultiFMV5DirectComparatorError(
                "direct MLP canonical settings require a direction"
            )
        inputs, outputs, hidden = (
            (225, 584, 666) if direction == "forward" else (584, 225, 667)
        )
        return (
            ("input_width", inputs),
            ("output_width", outputs),
            ("hidden_width", hidden),
            ("weight_decay", schedule.weight_decay),
        )
    raise CCLEMultiFMV5DirectComparatorError("unknown schedule type")


def _mlp_candidate_id(schedule: DirectMLPSchedule, direction: Direction) -> str:
    return canonical_schedule_id(schedule, direction=direction)


def _per_fit_seed(
    root_seed: int,
    *,
    stage: str,
    phase: str,
    direction: Direction,
    candidate_id: str,
    fold: int | str,
    member: int,
    purpose: str,
) -> int:
    checked_root = _exact_int(root_seed, "root_seed", minimum=0)
    if checked_root >= 2**32:
        raise CCLEMultiFMV5DirectComparatorError("root_seed must be smaller than 2**32")
    checked_direction = _direction(direction)
    if type(stage) is not str or not stage or "|" in stage:
        raise CCLEMultiFMV5DirectComparatorError("stage is not a canonical token")
    if phase != "A1":
        raise CCLEMultiFMV5DirectComparatorError(
            "direct MLP seed phase must be the registered A1 token"
        )
    if type(candidate_id) is not str or not candidate_id or "|" in candidate_id:
        raise CCLEMultiFMV5DirectComparatorError(
            "candidate_id is not a canonical token"
        )
    if fold != "-":
        _exact_int(fold, "fold", minimum=0)
    _exact_int(member, "member", minimum=0)
    if type(purpose) is not str or not purpose or "|" in purpose:
        raise CCLEMultiFMV5DirectComparatorError("purpose is not a canonical token")
    preimage = "|".join(
        (
            str(checked_root),
            stage,
            phase,
            "capacity_matched_direct_MLP",
            checked_direction,
            candidate_id,
            "0",
            str(fold),
            "-",
            "-",
            str(member),
            purpose,
        )
    )
    return int.from_bytes(
        hashlib.sha256(preimage.encode("utf-8")).digest()[:8],
        byteorder="big",
        signed=False,
    ) % (2**63)


def _nested_split(sample_count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    count = _exact_int(sample_count, "sample_count", minimum=1)
    nested_seed = _exact_int(seed, "nested seed", minimum=0)
    if nested_seed >= 2**63:
        raise CCLEMultiFMV5DirectComparatorError("nested seed must be below 2^63")
    permutation = np.random.Generator(np.random.PCG64(nested_seed)).permutation(count)
    early_count = math.ceil(count / 8)
    early = np.sort(permutation[:early_count]).astype(np.int64, copy=False)
    optimizer = np.sort(permutation[early_count:]).astype(np.int64, copy=False)
    early = np.ascontiguousarray(early)
    optimizer = np.ascontiguousarray(optimizer)
    if early.size < 16 or optimizer.size < 32:
        raise CCLEMultiFMV5DirectComparatorError(
            "nested direct MLP split violates minimum row counts"
        )
    if np.intersect1d(early, optimizer).size:
        raise CCLEMultiFMV5DirectComparatorError("nested split overlaps")
    if not np.array_equal(
        np.sort(np.concatenate((early, optimizer))),
        np.arange(count, dtype=np.int64),
    ):
        raise CCLEMultiFMV5DirectComparatorError("nested split loses rows")
    return optimizer, early


def _import_torch() -> object:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - project runtime supplies torch
        raise CCLEMultiFMV5DirectComparatorError(
            "PyTorch is required for the direct MLP comparator"
        ) from exc
    return torch


def _build_direct_mlp(
    torch: object,
    *,
    input_count: int,
    output_count: int,
    hidden_width: int,
    model_seed: int,
    device: str = "cpu",
) -> object:
    nn = torch.nn  # type: ignore[attr-defined]
    functional = torch.nn.functional  # type: ignore[attr-defined]

    class DirectMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Raw Parameters avoid nn.Linear.reset_parameters consuming the
            # mutable global RNG before our per-name initialization.
            self.input_weight = nn.Parameter(
                torch.empty(hidden_width, input_count, dtype=torch.float32)
            )
            self.input_bias = nn.Parameter(
                torch.empty(hidden_width, dtype=torch.float32)
            )
            self.output_weight = nn.Parameter(
                torch.empty(output_count, hidden_width, dtype=torch.float32)
            )
            self.output_bias = nn.Parameter(
                torch.empty(output_count, dtype=torch.float32)
            )
            self.gelu_approximate = "none"

        def forward(self, source: object) -> object:
            hidden = functional.linear(source, self.input_weight, self.input_bias)
            activated = functional.gelu(hidden, approximate=self.gelu_approximate)
            return functional.linear(activated, self.output_weight, self.output_bias)

    model = DirectMLP()
    total = sum(int(parameter.numel()) for parameter in model.parameters())
    expected = direct_mlp_parameter_count(input_count, output_count, hidden_width)
    if total != expected:
        raise CCLEMultiFMV5DirectComparatorError(
            "direct MLP named parameter count differs from the analytic count"
        )
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        generator = torch.Generator(device="cpu")  # type: ignore[attr-defined]
        generator.manual_seed(_derive_seed(model_seed, "initialization", name))
        with torch.no_grad():  # type: ignore[attr-defined]
            if name.endswith("_weight"):
                nn.init.xavier_uniform_(parameter, gain=1.0, generator=generator)
            elif name.endswith("_bias"):
                nn.init.zeros_(parameter)
            else:  # pragma: no cover - fixed Linear registry
                raise CCLEMultiFMV5DirectComparatorError(
                    "direct MLP has an unknown parameter"
                )
    if type(device) is not str or (device != "cpu" and not device.startswith("cuda:")):
        raise CCLEMultiFMV5DirectComparatorError("invalid direct MLP device")
    if device.startswith("cuda:") and not torch.cuda.is_available():  # type: ignore[attr-defined]
        raise CCLEMultiFMV5DirectComparatorError("requested CUDA device is unavailable")
    return model.to(device)


def _batch_order(model_fit_seed: int, epoch: int, sample_count: int) -> np.ndarray:
    _exact_int(epoch, "epoch", minimum=1)
    _exact_int(sample_count, "sample_count", minimum=1)
    preimage = f"{model_fit_seed}|batch_order|{epoch}"
    seed = int.from_bytes(
        hashlib.sha256(preimage.encode("utf-8")).digest()[:8],
        byteorder="big",
        signed=False,
    ) % (2**63)
    order = np.random.Generator(np.random.PCG64(seed)).permutation(sample_count)
    return np.ascontiguousarray(order, dtype=np.int64)


def _make_optimizer(torch: object, model: object, weight_decay: float) -> object:
    optimizer = torch.optim.AdamW(  # type: ignore[attr-defined]
        model.parameters(),
        lr=0.0003,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
        amsgrad=False,
        foreach=False,
        fused=False,
        capturable=False,
        maximize=False,
        differentiable=False,
    )
    defaults = optimizer.defaults
    exact = {
        "lr": 0.0003,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": weight_decay,
        "amsgrad": False,
        "foreach": False,
        "fused": False,
        "capturable": False,
        "maximize": False,
        "differentiable": False,
    }
    for name, expected in exact.items():
        observed = defaults.get(name)
        if type(observed) is not type(expected) or observed != expected:
            raise CCLEMultiFMV5DirectComparatorError(
                f"AdamW field {name} differs from the fixed lifecycle"
            )
    return optimizer


def _train_epoch(
    torch: object,
    model: object,
    optimizer: object,
    source: np.ndarray,
    target: np.ndarray,
    *,
    epoch: int,
    model_fit_seed: int,
) -> DirectEpochTrace:
    model.train()
    device = next(model.parameters()).device
    order = _batch_order(model_fit_seed, epoch, source.shape[0])
    full_source = torch.from_numpy(  # type: ignore[attr-defined]
        np.ascontiguousarray(source, dtype=np.float32)
    ).to(device=device)
    full_target = torch.from_numpy(  # type: ignore[attr-defined]
        np.ascontiguousarray(target, dtype=np.float32)
    ).to(device=device)
    batches: list[DirectBatchTrace] = []
    for start in range(0, order.size, 32):
        batch = order[start : start + 32]
        indices = torch.from_numpy(  # type: ignore[attr-defined]
            np.ascontiguousarray(batch, dtype=np.int64)
        ).to(device=device)
        x = torch.index_select(full_source, 0, indices)  # type: ignore[attr-defined]
        y = torch.index_select(full_target, 0, indices)  # type: ignore[attr-defined]
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = torch.nn.functional.mse_loss(  # type: ignore[attr-defined]
            prediction, y, reduction="mean"
        )
        if not bool(torch.isfinite(loss).item()):  # type: ignore[attr-defined]
            raise CCLEMultiFMV5DirectComparatorError("direct MLP loss is non-finite")
        loss.backward()
        parameters = tuple(model.parameters())
        for parameter in parameters:
            if parameter.grad is None or not bool(
                torch.isfinite(parameter.grad).all().item()  # type: ignore[attr-defined]
            ):
                raise CCLEMultiFMV5DirectComparatorError(
                    "direct MLP gradient is absent or non-finite"
                )
        torch.nn.utils.clip_grad_norm_(  # type: ignore[attr-defined]
            parameters,
            max_norm=1.0,
            error_if_nonfinite=True,
        )
        optimizer.step()
        for parameter in parameters:
            if not bool(torch.isfinite(parameter).all().item()):  # type: ignore[attr-defined]
                raise CCLEMultiFMV5DirectComparatorError(
                    "direct MLP parameter became non-finite"
                )
        batches.append(
            DirectBatchTrace(
                start=start,
                stop=start + len(batch),
                row_indices=tuple(int(item) for item in batch),
                loss=float(loss.detach().cpu().item()),
            )
        )
    order_bytes = np.asarray(order, dtype="<i8").tobytes(order="C")
    return DirectEpochTrace(
        epoch=epoch,
        batch_order=tuple(int(item) for item in order),
        batch_order_sha256=_raw_sha256(order_bytes),
        batches=tuple(batches),
        validation_loss=None,
        model_state_sha256=_model_state_sha256(model),
        optimizer_state_sha256=_optimizer_state_sha256(optimizer, model),
    )


def _validation_mse(
    torch: object, model: object, source: np.ndarray, target: np.ndarray
) -> float:
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():  # type: ignore[attr-defined]
        x = torch.from_numpy(  # type: ignore[attr-defined]
            np.ascontiguousarray(source, dtype=np.float32)
        ).to(device=device)
        y = torch.from_numpy(  # type: ignore[attr-defined]
            np.ascontiguousarray(target, dtype=np.float32)
        ).to(device=device)
        loss = torch.nn.functional.mse_loss(model(x), y, reduction="mean")  # type: ignore[attr-defined]
    value = float(loss.detach().cpu().item())
    if not np.isfinite(value):
        raise CCLEMultiFMV5DirectComparatorError(
            "direct MLP early-stop loss is non-finite"
        )
    return value


@dataclass(frozen=True, slots=True)
class DirectMLPMemberReceipt:
    member: int
    hidden_width: int
    parameter_count: int
    parameter_excess: int
    nested_split_seed: int
    selection_model_seed: int
    refit_model_seed: int
    optimizer_fit_indices: tuple[int, ...]
    early_stop_indices: tuple[int, ...]
    best_epoch: int
    terminal_selection_loss: float


@dataclass(frozen=True, slots=True)
class DirectMLPModel:
    schedule: DirectMLPSchedule
    model: object
    source_scale: _EndpointScale
    target_scale: _EndpointScale
    source_width: int
    target_width: int
    hidden_width: int
    parameter_count: int
    member: int
    best_epoch: int


@dataclass(frozen=True, slots=True)
class DirectAttemptRuntime:
    mode: Literal["synthetic_cpu_non_authorizing", "production_cuda"]
    device: str
    fresh_process: bool
    process_fit_count: int
    torch_version: str
    cuda_version: str | None
    cudnn_version: int | None
    device_uuid: str | None
    image_sha256: str
    source_sha256: str
    deterministic_algorithms: bool
    scientific_claim_authority: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if self.mode == "synthetic_cpu_non_authorizing":
            if self.device != "cpu":
                raise CCLEMultiFMV5DirectComparatorError(
                    "synthetic runtime is CPU-only"
                )
        elif self.mode == "production_cuda":
            raise CCLEMultiFMV5DirectComparatorError(
                "production execution belongs exclusively to the direct worker adapter"
            )
        else:
            raise CCLEMultiFMV5DirectComparatorError(
                "attempt runtime mode is unregistered"
            )
        if not self.fresh_process or self.process_fit_count != 1:
            raise CCLEMultiFMV5DirectComparatorError(
                "each attempt requires one fit in a fresh process"
            )
        _check_sha256(self.image_sha256, "image_sha256")
        _check_sha256(self.source_sha256, "source_sha256")
        if not self.deterministic_algorithms:
            raise CCLEMultiFMV5DirectComparatorError(
                "deterministic algorithms are required"
            )


def _require_synthetic_runtime_before_input(runtime: object, role: str) -> DirectAttemptRuntime:
    """Keep draft math entry points from touching arrays in production mode."""

    if type(runtime) is not DirectAttemptRuntime or runtime.mode != "synthetic_cpu_non_authorizing":
        raise CCLEMultiFMV5DirectComparatorError(
            f"UNBOUND_DRAFT_STOP: {role} belongs to the external worker adapter"
        )
    global _SYNTHETIC_PROCESS_FIT_COUNT
    _SYNTHETIC_PROCESS_FIT_COUNT += 1
    if _SYNTHETIC_PROCESS_FIT_COUNT != 1:
        raise CCLEMultiFMV5DirectComparatorError(
            "synthetic process attempted more than one fit; a new runtime object "
            "does not constitute a fresh process"
        )
    return runtime


# A runtime dataclass is only a receipt-shaped value; constructing another one
# cannot reset process history.  Production uses the external worker's own
# process guard, while these draft/synthetic entry points fail closed globally.
_SYNTHETIC_PROCESS_FIT_COUNT = 0


@dataclass(frozen=True, slots=True)
class DirectBatchTrace:
    start: int
    stop: int
    row_indices: tuple[int, ...]
    loss: float


@dataclass(frozen=True, slots=True)
class DirectEpochTrace:
    epoch: int
    batch_order: tuple[int, ...]
    batch_order_sha256: str
    batches: tuple[DirectBatchTrace, ...]
    validation_loss: float | None
    model_state_sha256: str
    optimizer_state_sha256: str


@dataclass(frozen=True, slots=True)
class DirectRNGStateReceipt:
    python_sha256: str
    numpy_sha256: str
    torch_cpu_sha256: str
    torch_cuda_sha256: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DirectTensorArtifact:
    name: str
    dtype: str
    shape: tuple[int, ...]
    raw_hex: str
    sha256: str


def _tensor_artifact(name: str, tensor: object) -> DirectTensorArtifact:
    raw = _tensor_bytes(tensor)
    return DirectTensorArtifact(
        name,
        str(tensor.dtype),
        tuple(int(item) for item in tensor.shape),
        raw.hex(),
        _raw_sha256(raw),
    )


def _model_checkpoint(model: object) -> tuple[DirectTensorArtifact, ...]:
    return tuple(
        _tensor_artifact(name, tensor)
        for name, tensor in sorted(model.state_dict().items())
    )


def _optimizer_checkpoint(
    optimizer: object, model: object
) -> tuple[DirectTensorArtifact, ...]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    result: list[DirectTensorArtifact] = []
    for parameter, state in sorted(
        optimizer.state.items(), key=lambda item: names[id(item[0])]
    ):
        for key, value in sorted(state.items()):
            if hasattr(value, "detach"):
                result.append(_tensor_artifact(f"{names[id(parameter)]}.{key}", value))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class DirectSeedDerivationReceipt:
    root_role: str
    root_value: int
    full_preimage: str
    preimage_sha256: str
    derivation: str
    derived_value: int

    def __post_init__(self) -> None:
        _exact_int(self.root_value, "seed root", minimum=0)
        _exact_int(self.derived_value, "derived seed", minimum=0)
        _check_sha256(self.preimage_sha256, "seed preimage SHA-256")
        if self.preimage_sha256 != _raw_sha256(self.full_preimage.encode("utf-8")):
            raise CCLEMultiFMV5DirectComparatorError("seed preimage hash differs")
        if self.derivation != "uint64_be_first8_mod_2^63":
            raise CCLEMultiFMV5DirectComparatorError("seed derivation is unregistered")
        expected = int.from_bytes(
            hashlib.sha256(self.full_preimage.encode("utf-8")).digest()[:8], "big"
        ) % (2**63)
        if self.derived_value != expected:
            raise CCLEMultiFMV5DirectComparatorError("derived seed value differs")


def _validate_production_seed_receipt(
    receipt: DirectSeedDerivationReceipt | None, *, value: int, purpose: str
) -> DirectSeedDerivationReceipt:
    if type(receipt) is not DirectSeedDerivationReceipt:
        raise CCLEMultiFMV5DirectComparatorError(
            "production attempt requires an exact seed derivation receipt"
        )
    fields = receipt.full_preimage.split("|")
    if (
        len(fields) != 12
        or fields[3] != "capacity_matched_direct_MLP"
        or fields[-1] != purpose
    ):
        raise CCLEMultiFMV5DirectComparatorError("seed preimage lifecycle differs")
    if fields[0] != str(receipt.root_value) or receipt.derived_value != value:
        raise CCLEMultiFMV5DirectComparatorError("seed receipt does not bind value")
    return receipt


def _raw_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tensor_bytes(tensor: object) -> bytes:
    array = tensor.detach().cpu().contiguous().numpy()  # type: ignore[attr-defined]
    return array.tobytes(order="C")


def _model_state_sha256(model: object) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        raw = _tensor_bytes(tensor)
        header = _canonical_json_bytes(
            {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        )
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def _optimizer_state_sha256(optimizer: object, model: object) -> str:
    payload = _optimizer_state_payload(optimizer, model)
    return _raw_sha256(_canonical_json_bytes(payload))


def _optimizer_state_payload(optimizer: object, model: object) -> dict[str, object]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    payload: dict[str, object] = {"defaults": {}, "groups": [], "state": []}
    payload["defaults"] = {
        key: list(value) if type(value) is tuple else value
        for key, value in sorted(optimizer.defaults.items())
        if type(value) in {bool, int, float, str, tuple} or value is None
    }
    for group in optimizer.param_groups:
        payload["groups"].append(  # type: ignore[union-attr]
            {
                "parameters": [names[id(item)] for item in group["params"]],
                "fields": {
                    key: list(value) if type(value) is tuple else value
                    for key, value in sorted(group.items())
                    if key != "params"
                    and (type(value) in {bool, int, float, str, tuple} or value is None)
                },
            }
        )
    for parameter, state in sorted(
        optimizer.state.items(), key=lambda item: names[id(item[0])]
    ):
        encoded: dict[str, object] = {"parameter": names[id(parameter)]}
        for key, value in sorted(state.items()):
            if hasattr(value, "detach"):
                encoded[key] = {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "sha256": _raw_sha256(_tensor_bytes(value)),
                }
            elif type(value) in {bool, int, float, str} or value is None:
                encoded[key] = value
            else:
                raise CCLEMultiFMV5DirectComparatorError(
                    "optimizer state contains an unregistered value"
                )
        payload["state"].append(encoded)  # type: ignore[union-attr]
    return payload


def _rng_state_receipt(torch: object) -> DirectRNGStateReceipt:
    cuda: tuple[str, ...] = ()
    if torch.cuda.is_available():  # type: ignore[attr-defined]
        cuda = tuple(
            _raw_sha256(_tensor_bytes(item)) for item in torch.cuda.get_rng_state_all()
        )  # type: ignore[attr-defined]
    return DirectRNGStateReceipt(
        python_sha256=_raw_sha256(pickle.dumps(random.getstate(), protocol=5)),
        numpy_sha256=_raw_sha256(pickle.dumps(np.random.get_state(), protocol=5)),
        torch_cpu_sha256=_raw_sha256(_tensor_bytes(torch.get_rng_state())),  # type: ignore[attr-defined]
        torch_cuda_sha256=cuda,
    )


def _array_sha256(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    header = _canonical_json_bytes(
        {"dtype": value.dtype.str, "shape": list(value.shape), "order": "C"}
    )
    digest.update(len(header).to_bytes(8, "little"))
    digest.update(header)
    digest.update(np.ascontiguousarray(value).tobytes(order="C"))
    return digest.hexdigest()


def _runtime_binding_sha256(runtime: DirectAttemptRuntime) -> str:
    return _canonical_sha256(
        {
            name: getattr(runtime, name)
            for name in runtime.__dataclass_fields__
            if name != "scientific_claim_authority"
        }
    )


def _verify_attempt_runtime(runtime: DirectAttemptRuntime, torch: object) -> None:
    if runtime.mode != "production_cuda":
        return
    if str(torch.__version__) != runtime.torch_version:
        raise CCLEMultiFMV5DirectComparatorError("bound torch version differs")
    if not torch.cuda.is_available() or str(torch.version.cuda) != runtime.cuda_version:
        raise CCLEMultiFMV5DirectComparatorError(
            "bound production CUDA runtime is unavailable"
        )
    if int(torch.backends.cudnn.version()) != runtime.cudnn_version:
        raise CCLEMultiFMV5DirectComparatorError("bound cuDNN version differs")
    device_index = int(runtime.device.split(":", 1)[1])
    properties = torch.cuda.get_device_properties(device_index)
    observed_uuid = getattr(properties, "uuid", None)
    if observed_uuid is None or str(observed_uuid) != runtime.device_uuid:
        raise CCLEMultiFMV5DirectComparatorError("bound CUDA device UUID differs")
    image = os.environ.get("CREED_IMAGE_SHA256")
    if image != runtime.image_sha256:
        raise CCLEMultiFMV5DirectComparatorError("bound image hash differs")
    observed_source = _raw_sha256(Path(__file__).read_bytes())
    if observed_source != runtime.source_sha256:
        raise CCLEMultiFMV5DirectComparatorError("bound source hash differs")


@dataclass(frozen=True, slots=True)
class DirectOptimizerFitOutput:
    best_epoch: int
    terminal_loss: float
    optimizer_fit_indices: tuple[int, ...]
    early_stop_indices: tuple[int, ...]
    selection_binding_sha256: str
    runtime: DirectAttemptRuntime
    epoch_trace: tuple[DirectEpochTrace, ...] = ()
    initial_model_state_sha256: str = ""
    initial_optimizer_state_sha256: str = ""
    final_model_state_sha256: str = ""
    final_optimizer_state_sha256: str = ""
    selected_model_state_sha256: str = ""
    selected_optimizer_state_sha256: str = ""
    rng_before: DirectRNGStateReceipt | None = None
    rng_after: DirectRNGStateReceipt | None = None
    source_fit_sha256: str = ""
    target_fit_sha256: str = ""
    seed_full_preimage: str = ""
    seed_preimage_sha256: str = ""
    seed_value: int = 0
    split_seed_full_preimage: str = ""
    split_seed_preimage_sha256: str = ""
    split_seed_value: int = 0
    runtime_binding_sha256: str = ""
    receipt_sha256: str = ""
    selected_model_checkpoint: tuple[DirectTensorArtifact, ...] = ()
    selected_optimizer_checkpoint: tuple[DirectTensorArtifact, ...] = ()
    held_prediction: None = field(init=False, default=None)
    scientific_claim_authority: bool = field(init=False, default=False)


@dataclass(frozen=True, slots=True)
class SyntheticDirectSelectionParent:
    selection_binding_sha256: str
    best_epoch: int
    synthetic_structurally_bound: bool
    optimizer_receipt_sha256: str | None = None
    synthetic_parent_binding_sha256: str | None = None
    selected_model_state_sha256: str | None = None
    selected_optimizer_state_sha256: str | None = None

    def __post_init__(self) -> None:
        _check_sha256(self.selection_binding_sha256, "selection_binding_sha256")
        _exact_int(self.best_epoch, "best_epoch", minimum=1)
        if not self.synthetic_structurally_bound:
            raise CCLEMultiFMV5DirectComparatorError(
                "synthetic selection parent is not structurally bound"
            )
        if self.optimizer_receipt_sha256 is not None:
            _check_sha256(self.optimizer_receipt_sha256, "optimizer_receipt_sha256")
        if self.synthetic_parent_binding_sha256 is not None:
            _check_sha256(
                self.synthetic_parent_binding_sha256, "synthetic_parent_binding_sha256"
            )
        if self.synthetic_structurally_bound:
            for value, label in (
                (self.optimizer_receipt_sha256, "optimizer_receipt_sha256"),
                (
                    self.synthetic_parent_binding_sha256,
                    "synthetic_parent_binding_sha256",
                ),
                (self.selected_model_state_sha256, "selected_model_state_sha256"),
                (
                    self.selected_optimizer_state_sha256,
                    "selected_optimizer_state_sha256",
                ),
            ):
                _check_sha256(value, label)
            expected = _canonical_sha256(
                {
                    "status": "synthetic_structurally_bound",
                    "optimizer_receipt_sha256": self.optimizer_receipt_sha256,
                    "selection_binding_sha256": self.selection_binding_sha256,
                    "best_epoch": self.best_epoch,
                    "selected_model_state_sha256": self.selected_model_state_sha256,
                    "selected_optimizer_state_sha256": self.selected_optimizer_state_sha256,
                }
            )
            if self.synthetic_parent_binding_sha256 != expected:
                raise CCLEMultiFMV5DirectComparatorError(
                    "synthetic binding hash does not bind parent fields"
                )


@dataclass(frozen=True, slots=True)
class DirectRefitOutput:
    model: DirectMLPModel
    prediction: np.ndarray
    parent_selection_sha256: str
    epochs: int
    runtime: DirectAttemptRuntime
    epoch_trace: tuple[DirectEpochTrace, ...] = ()
    initial_model_state_sha256: str = ""
    initial_optimizer_state_sha256: str = ""
    final_model_state_sha256: str = ""
    final_optimizer_state_sha256: str = ""
    rng_before: DirectRNGStateReceipt | None = None
    rng_after: DirectRNGStateReceipt | None = None
    source_fit_sha256: str = ""
    target_fit_sha256: str = ""
    source_predict_sha256: str = ""
    prediction_sha256: str = ""
    seed_full_preimage: str = ""
    seed_preimage_sha256: str = ""
    seed_value: int = 0
    runtime_binding_sha256: str = ""
    receipt_sha256: str = ""
    parent_optimizer_receipt_sha256: str = ""
    synthetic_parent_binding_sha256: str = ""
    scientific_claim_authority: bool = field(init=False, default=False)


@dataclass(frozen=True, slots=True)
class _SelectionExecution:
    best_epoch: int
    terminal_loss: float
    optimizer_indices: np.ndarray
    early_indices: np.ndarray
    epoch_trace: tuple[DirectEpochTrace, ...]
    initial_model_sha256: str
    initial_optimizer_sha256: str
    final_model_sha256: str
    final_optimizer_sha256: str
    selected_model_sha256: str
    selected_optimizer_sha256: str
    selected_model_checkpoint: tuple[DirectTensorArtifact, ...]
    selected_optimizer_checkpoint: tuple[DirectTensorArtifact, ...]
    validation_predictions: tuple[DirectTensorArtifact, ...]
    initial_model_checkpoint: tuple[DirectTensorArtifact, ...]
    initial_optimizer_checkpoint: tuple[DirectTensorArtifact, ...]
    final_model_checkpoint: tuple[DirectTensorArtifact, ...]
    final_optimizer_checkpoint: tuple[DirectTensorArtifact, ...]
    initial_optimizer_payload: dict[str, object]
    selected_optimizer_payload: dict[str, object]
    final_optimizer_payload: dict[str, object]


def synthetic_direct_selection_parent(
    output: DirectOptimizerFitOutput, *, synthetic_parent_binding_sha256: str
) -> SyntheticDirectSelectionParent:
    if type(output) is not DirectOptimizerFitOutput:
        raise CCLEMultiFMV5DirectComparatorError("optimizer output type changed")
    verify_direct_optimizer_fit_output(output)
    _check_sha256(synthetic_parent_binding_sha256, "synthetic parent binding SHA-256")
    if (
        output.held_prediction is not None
        or len(output.epoch_trace) < output.best_epoch
    ):
        raise CCLEMultiFMV5DirectComparatorError("optimizer output is incomplete")
    if (
        output.selected_model_state_sha256
        != output.epoch_trace[output.best_epoch - 1].model_state_sha256
    ):
        raise CCLEMultiFMV5DirectComparatorError("selected model state is inconsistent")
    if not output.selected_model_checkpoint or not output.selected_optimizer_checkpoint:
        raise CCLEMultiFMV5DirectComparatorError("selected raw checkpoint is absent")
    for item in (
        *output.selected_model_checkpoint,
        *output.selected_optimizer_checkpoint,
    ):
        try:
            raw = bytes.fromhex(item.raw_hex)
        except ValueError as exc:
            raise CCLEMultiFMV5DirectComparatorError(
                "checkpoint hex is malformed"
            ) from exc
        if item.raw_hex != item.raw_hex.lower() or _raw_sha256(raw) != item.sha256:
            raise CCLEMultiFMV5DirectComparatorError("checkpoint raw hash differs")
    if synthetic_parent_binding_sha256 != _synthetic_parent_binding_sha256(output):
        raise CCLEMultiFMV5DirectComparatorError(
            "synthetic binding hash does not bind the optimizer receipt"
        )
    return SyntheticDirectSelectionParent(
        output.selection_binding_sha256,
        output.best_epoch,
        True,
        output.receipt_sha256,
        synthetic_parent_binding_sha256,
        output.selected_model_state_sha256,
        output.selected_optimizer_state_sha256,
    )


def _synthetic_parent_binding_sha256(output: DirectOptimizerFitOutput) -> str:
    return _canonical_sha256(
        {
            "status": "synthetic_structurally_bound",
            "optimizer_receipt_sha256": output.receipt_sha256,
            "selection_binding_sha256": output.selection_binding_sha256,
            "best_epoch": output.best_epoch,
            "selected_model_state_sha256": output.selected_model_state_sha256,
            "selected_optimizer_state_sha256": output.selected_optimizer_state_sha256,
        }
    )


def _rng_payload(value: DirectRNGStateReceipt | None) -> object:
    if value is None:
        return None
    return {
        "python": value.python_sha256,
        "numpy": value.numpy_sha256,
        "torch_cpu": value.torch_cpu_sha256,
        "torch_cuda": list(value.torch_cuda_sha256),
    }


def _optimizer_output_sha256(output: DirectOptimizerFitOutput) -> str:
    return _canonical_sha256(
        {
            "best_epoch": output.best_epoch,
            "terminal_loss": output.terminal_loss,
            "optimizer_fit_indices": list(output.optimizer_fit_indices),
            "early_stop_indices": list(output.early_stop_indices),
            "selection_binding_sha256": output.selection_binding_sha256,
            "runtime_binding_sha256": output.runtime_binding_sha256,
            "source_fit_sha256": output.source_fit_sha256,
            "target_fit_sha256": output.target_fit_sha256,
            "seed_full_preimage": output.seed_full_preimage,
            "seed_preimage_sha256": output.seed_preimage_sha256,
            "seed_value": output.seed_value,
            "split_seed_full_preimage": output.split_seed_full_preimage,
            "split_seed_preimage_sha256": output.split_seed_preimage_sha256,
            "split_seed_value": output.split_seed_value,
            "initial_model": output.initial_model_state_sha256,
            "initial_optimizer": output.initial_optimizer_state_sha256,
            "final_model": output.final_model_state_sha256,
            "final_optimizer": output.final_optimizer_state_sha256,
            "selected_model": output.selected_model_state_sha256,
            "selected_optimizer": output.selected_optimizer_state_sha256,
            "selected_model_checkpoint": [
                [item.name, item.dtype, list(item.shape), item.sha256]
                for item in output.selected_model_checkpoint
            ],
            "selected_optimizer_checkpoint": [
                [item.name, item.dtype, list(item.shape), item.sha256]
                for item in output.selected_optimizer_checkpoint
            ],
            "rng_before": _rng_payload(output.rng_before),
            "rng_after": _rng_payload(output.rng_after),
            "epochs": [
                {
                    "epoch": item.epoch,
                    "order": item.batch_order_sha256,
                    "batches": [
                        [batch.start, batch.stop, list(batch.row_indices), batch.loss]
                        for batch in item.batches
                    ],
                    "validation": item.validation_loss,
                    "model": item.model_state_sha256,
                    "optimizer": item.optimizer_state_sha256,
                }
                for item in output.epoch_trace
            ],
        }
    )


def verify_direct_optimizer_fit_output(output: DirectOptimizerFitOutput) -> None:
    if (
        type(output) is not DirectOptimizerFitOutput
        or output.held_prediction is not None
    ):
        raise CCLEMultiFMV5DirectComparatorError(
            "optimizer output type/held contract changed"
        )
    if output.receipt_sha256 != _optimizer_output_sha256(output):
        raise CCLEMultiFMV5DirectComparatorError("optimizer receipt self-hash differs")
    if len(output.epoch_trace) < output.best_epoch:
        raise CCLEMultiFMV5DirectComparatorError("optimizer epoch trace is incomplete")
    if (
        output.selected_model_state_sha256
        != output.epoch_trace[output.best_epoch - 1].model_state_sha256
    ):
        raise CCLEMultiFMV5DirectComparatorError("selected model state is inconsistent")


def _refit_output_sha256(output: DirectRefitOutput) -> str:
    return _canonical_sha256(
        {
            "parent_selection_sha256": output.parent_selection_sha256,
            "parent_optimizer_receipt_sha256": output.parent_optimizer_receipt_sha256,
            "synthetic_parent_binding_sha256": output.synthetic_parent_binding_sha256,
            "epochs": output.epochs,
            "runtime_binding_sha256": output.runtime_binding_sha256,
            "source_fit_sha256": output.source_fit_sha256,
            "target_fit_sha256": output.target_fit_sha256,
            "source_predict_sha256": output.source_predict_sha256,
            "prediction_sha256": output.prediction_sha256,
            "seed_full_preimage": output.seed_full_preimage,
            "seed_preimage_sha256": output.seed_preimage_sha256,
            "seed_value": output.seed_value,
            "initial_model": output.initial_model_state_sha256,
            "initial_optimizer": output.initial_optimizer_state_sha256,
            "final_model": output.final_model_state_sha256,
            "final_optimizer": output.final_optimizer_state_sha256,
            "rng_before": _rng_payload(output.rng_before),
            "rng_after": _rng_payload(output.rng_after),
            "epochs_trace": [
                [
                    item.epoch,
                    item.batch_order_sha256,
                    item.model_state_sha256,
                    item.optimizer_state_sha256,
                ]
                for item in output.epoch_trace
            ],
        }
    )


def verify_direct_refit_output(output: DirectRefitOutput) -> None:
    if type(output) is not DirectRefitOutput:
        raise CCLEMultiFMV5DirectComparatorError("refit output type changed")
    if output.receipt_sha256 != _refit_output_sha256(output):
        raise CCLEMultiFMV5DirectComparatorError("refit receipt self-hash differs")
    if len(output.epoch_trace) != output.epochs:
        raise CCLEMultiFMV5DirectComparatorError("refit epoch trace is incomplete")
    if output.prediction_sha256 != _array_sha256(output.prediction):
        raise CCLEMultiFMV5DirectComparatorError("refit prediction hash differs")
    if output.final_model_state_sha256 != _model_state_sha256(output.model.model):
        raise CCLEMultiFMV5DirectComparatorError("refit final model hash differs")


def direct_mlp_optimizer_fit_attempt(
    source_fit: object,
    target_fit: object,
    *,
    schedule: DirectMLPSchedule,
    hidden_width: int,
    selection_seed: int,
    split_seed: int,
    runtime: DirectAttemptRuntime,
    selection_seed_receipt: DirectSeedDerivationReceipt | None = None,
    split_seed_receipt: DirectSeedDerivationReceipt | None = None,
) -> DirectOptimizerFitOutput:
    runtime = _require_synthetic_runtime_before_input(runtime, "optimizer")
    torch = _import_torch()
    _verify_attempt_runtime(runtime, torch)
    if runtime.mode == "production_cuda":
        selection_seed_receipt = _validate_production_seed_receipt(
            selection_seed_receipt, value=selection_seed, purpose="optimizer_fit"
        )
        split_seed_receipt = _validate_production_seed_receipt(
            split_seed_receipt, value=split_seed, purpose="early_stop_split"
        )
    source = _require_f64_matrix(source_fit, "source_fit")
    target = _require_f64_matrix(target_fit, "target_fit", rows=source.shape[0])
    rng_before = _rng_state_receipt(torch)
    execution = _execute_selection(
        source,
        target,
        schedule=schedule,
        hidden_width=hidden_width,
        selection_seed=selection_seed,
        split_seed=split_seed,
    )
    rng_after = _rng_state_receipt(torch)
    binding_payload = {
        "best_epoch": execution.best_epoch,
        "terminal_loss": execution.terminal_loss,
        "optimizer_rows": execution.optimizer_indices.tolist(),
        "early_rows": execution.early_indices.tolist(),
        "weight_decay": schedule.weight_decay,
        "input_width": source.shape[1],
        "output_width": target.shape[1],
        "hidden_width": hidden_width,
        "selected_model_state_sha256": execution.selected_model_sha256,
        "selected_optimizer_state_sha256": execution.selected_optimizer_sha256,
    }
    binding = _canonical_sha256(binding_payload)
    seed_preimage = (
        selection_seed_receipt.full_preimage
        if selection_seed_receipt is not None
        else f"synthetic_cpu_non_authorizing|optimizer_fit|{selection_seed}"
    )
    split_preimage = (
        split_seed_receipt.full_preimage
        if split_seed_receipt is not None
        else f"synthetic_cpu_non_authorizing|early_stop_split|{split_seed}"
    )
    fields = {
        **binding_payload,
        "selection_binding_sha256": binding,
        "runtime_binding_sha256": _runtime_binding_sha256(runtime),
        "source_fit_sha256": _array_sha256(source),
        "target_fit_sha256": _array_sha256(target),
        "seed_full_preimage": seed_preimage,
        "seed_value": selection_seed,
        "epoch_trace_sha256": _canonical_sha256(
            [
                {
                    "epoch": item.epoch,
                    "order": item.batch_order_sha256,
                    "model": item.model_state_sha256,
                    "optimizer": item.optimizer_state_sha256,
                    "validation": item.validation_loss,
                }
                for item in execution.epoch_trace
            ]
        ),
    }
    output = DirectOptimizerFitOutput(
        execution.best_epoch,
        execution.terminal_loss,
        tuple(map(int, execution.optimizer_indices)),
        tuple(map(int, execution.early_indices)),
        binding,
        runtime,
        execution.epoch_trace,
        execution.initial_model_sha256,
        execution.initial_optimizer_sha256,
        execution.final_model_sha256,
        execution.final_optimizer_sha256,
        execution.selected_model_sha256,
        execution.selected_optimizer_sha256,
        rng_before,
        rng_after,
        fields["source_fit_sha256"],
        fields["target_fit_sha256"],
        seed_preimage,
        _raw_sha256(seed_preimage.encode()),
        selection_seed,
        split_preimage,
        _raw_sha256(split_preimage.encode()),
        split_seed,
        fields["runtime_binding_sha256"],
        "",
        execution.selected_model_checkpoint,
        execution.selected_optimizer_checkpoint,
    )
    output = replace(output, receipt_sha256=_optimizer_output_sha256(output))
    verify_direct_optimizer_fit_output(output)
    return output


def direct_mlp_refit_attempt(
    source_fit: object,
    target_fit: object,
    source_predict: object,
    *,
    schedule: DirectMLPSchedule,
    hidden_width: int,
    refit_seed: int,
    member: int,
    parent: SyntheticDirectSelectionParent,
    runtime: DirectAttemptRuntime,
    refit_seed_receipt: DirectSeedDerivationReceipt | None = None,
) -> DirectRefitOutput:
    runtime = _require_synthetic_runtime_before_input(runtime, "refit")
    torch = _import_torch()
    _verify_attempt_runtime(runtime, torch)
    if runtime.mode == "production_cuda":
        refit_seed_receipt = _validate_production_seed_receipt(
            refit_seed_receipt, value=refit_seed, purpose="refit"
        )
    if (
        parent.optimizer_receipt_sha256 is None
        or parent.synthetic_parent_binding_sha256 is None
    ):
        raise CCLEMultiFMV5DirectComparatorError(
            "synthetic refit requires optimizer receipt and parent binding hashes"
        )
    source = _require_f64_matrix(source_fit, "source_fit")
    target = _require_f64_matrix(target_fit, "target_fit", rows=source.shape[0])
    predict = _require_f64_matrix(
        source_predict, "source_predict", columns=source.shape[1]
    )
    source_scale, target_scale = (
        _fit_endpoint_scale(source),
        _fit_endpoint_scale(target),
    )
    rng_before = _rng_state_receipt(torch)
    model = _build_direct_mlp(
        torch,
        input_count=source.shape[1],
        output_count=target.shape[1],
        hidden_width=hidden_width,
        model_seed=refit_seed,
        device=runtime.device,
    )
    optimizer = _make_optimizer(torch, model, schedule.weight_decay)
    initial_model = _model_state_sha256(model)
    initial_optimizer = _optimizer_state_sha256(optimizer, model)
    x, y = source_scale.transform(source), target_scale.transform(target)
    traces: list[DirectEpochTrace] = []
    for epoch in range(1, parent.best_epoch + 1):
        trace = _train_epoch(
            torch, model, optimizer, x, y, epoch=epoch, model_fit_seed=refit_seed
        )
        if trace is None:
            raise CCLEMultiFMV5DirectComparatorError("refit epoch trace is absent")
        traces.append(trace)
    wrapped = DirectMLPModel(
        schedule,
        model,
        source_scale,
        target_scale,
        source.shape[1],
        target.shape[1],
        hidden_width,
        direct_mlp_parameter_count(source.shape[1], target.shape[1], hidden_width),
        member,
        parent.best_epoch,
    )
    prediction = predict_direct_mlp(wrapped, predict)
    rng_after = _rng_state_receipt(torch)
    seed_preimage = (
        refit_seed_receipt.full_preimage
        if refit_seed_receipt is not None
        else f"synthetic_cpu_non_authorizing|refit|{refit_seed}"
    )
    prediction_hash = _array_sha256(prediction)
    runtime_hash = _runtime_binding_sha256(runtime)
    output = DirectRefitOutput(
        wrapped,
        _readonly(prediction),
        parent.selection_binding_sha256,
        parent.best_epoch,
        runtime,
        tuple(traces),
        initial_model,
        initial_optimizer,
        _model_state_sha256(model),
        _optimizer_state_sha256(optimizer, model),
        rng_before,
        rng_after,
        _array_sha256(source),
        _array_sha256(target),
        _array_sha256(predict),
        prediction_hash,
        seed_preimage,
        _raw_sha256(seed_preimage.encode()),
        refit_seed,
        runtime_hash,
        "",
    )
    output = replace(
        output,
        parent_optimizer_receipt_sha256=parent.optimizer_receipt_sha256,
        synthetic_parent_binding_sha256=parent.synthetic_parent_binding_sha256,
    )
    output = replace(output, receipt_sha256=_refit_output_sha256(output))
    verify_direct_refit_output(output)
    return output


def _execute_selection(
    source_fit: np.ndarray,
    target_fit: np.ndarray,
    *,
    schedule: DirectMLPSchedule,
    hidden_width: int,
    selection_seed: int,
    split_seed: int,
    device: str = "cpu",
) -> _SelectionExecution:
    torch = _import_torch()
    optimizer_indices, early_indices = _nested_split(source_fit.shape[0], split_seed)
    source_scale = _fit_endpoint_scale(source_fit[optimizer_indices])
    target_scale = _fit_endpoint_scale(target_fit[optimizer_indices])
    x_optimizer = source_scale.transform(source_fit[optimizer_indices])
    y_optimizer = target_scale.transform(target_fit[optimizer_indices])
    x_early = source_scale.transform(source_fit[early_indices])
    y_early = target_scale.transform(target_fit[early_indices])
    model = _build_direct_mlp(
        torch,
        input_count=source_fit.shape[1],
        output_count=target_fit.shape[1],
        hidden_width=hidden_width,
        model_seed=selection_seed,
        device=device,
    )
    optimizer = _make_optimizer(torch, model, schedule.weight_decay)
    initial_model = _model_state_sha256(model)
    initial_optimizer = _optimizer_state_sha256(optimizer, model)
    initial_model_checkpoint = _model_checkpoint(model)
    initial_optimizer_checkpoint = _optimizer_checkpoint(optimizer, model)
    initial_optimizer_payload = _optimizer_state_payload(optimizer, model)
    incumbent: float | None = None
    best_epoch: int | None = None
    stale = 0
    terminal = math.nan
    traces: list[DirectEpochTrace] = []
    selected_model = ""
    selected_optimizer = ""
    selected_model_checkpoint: tuple[DirectTensorArtifact, ...] = ()
    selected_optimizer_checkpoint: tuple[DirectTensorArtifact, ...] = ()
    selected_optimizer_payload: dict[str, object] = {}
    validation_predictions: list[DirectTensorArtifact] = []
    for epoch in range(1, 1001):
        trace = _train_epoch(
            torch,
            model,
            optimizer,
            x_optimizer,
            y_optimizer,
            epoch=epoch,
            model_fit_seed=selection_seed,
        )
        terminal = _validation_mse(torch, model, x_early, y_early)
        with torch.no_grad():
            validation_prediction = (
                model(torch.as_tensor(x_early, dtype=torch.float32, device=device))
                .detach()
                .cpu()
            )
        validation_predictions.append(
            _tensor_artifact(
                f"validation_prediction.epoch_{epoch:04d}", validation_prediction
            )
        )
        if trace is None:  # test seam; production always returns a complete trace
            trace = DirectEpochTrace(
                epoch,
                tuple(
                    int(item)
                    for item in _batch_order(selection_seed, epoch, len(x_optimizer))
                ),
                _raw_sha256(
                    _batch_order(selection_seed, epoch, len(x_optimizer))
                    .astype("<i8")
                    .tobytes()
                ),
                (),
                terminal,
                _model_state_sha256(model),
                _optimizer_state_sha256(optimizer, model),
            )
        else:
            trace = DirectEpochTrace(
                trace.epoch,
                trace.batch_order,
                trace.batch_order_sha256,
                trace.batches,
                terminal,
                trace.model_state_sha256,
                trace.optimizer_state_sha256,
            )
        traces.append(trace)
        if epoch < 50:
            continue
        if incumbent is None:
            incumbent = terminal
            best_epoch = epoch
            selected_model = trace.model_state_sha256
            selected_optimizer = trace.optimizer_state_sha256
            selected_model_checkpoint = _model_checkpoint(model)
            selected_optimizer_checkpoint = _optimizer_checkpoint(optimizer, model)
            selected_optimizer_payload = _optimizer_state_payload(optimizer, model)
            stale = 0
        elif terminal < incumbent - 1e-5:
            incumbent = terminal
            best_epoch = epoch
            selected_model = trace.model_state_sha256
            selected_optimizer = trace.optimizer_state_sha256
            selected_model_checkpoint = _model_checkpoint(model)
            selected_optimizer_checkpoint = _optimizer_checkpoint(optimizer, model)
            selected_optimizer_payload = _optimizer_state_payload(optimizer, model)
            stale = 0
        else:
            stale += 1
        if stale >= 40:
            break
    if best_epoch is None or incumbent is None or not np.isfinite(incumbent):
        raise CCLEMultiFMV5DirectComparatorError(
            "direct MLP did not select an eligible finite epoch"
        )
    return _SelectionExecution(
        best_epoch,
        terminal,
        optimizer_indices,
        early_indices,
        tuple(traces),
        initial_model,
        initial_optimizer,
        _model_state_sha256(model),
        _optimizer_state_sha256(optimizer, model),
        selected_model,
        selected_optimizer,
        selected_model_checkpoint,
        selected_optimizer_checkpoint,
        tuple(validation_predictions),
        initial_model_checkpoint,
        initial_optimizer_checkpoint,
        _model_checkpoint(model),
        _optimizer_checkpoint(optimizer, model),
        initial_optimizer_payload,
        selected_optimizer_payload,
        _optimizer_state_payload(optimizer, model),
    )


def _select_best_epoch(
    source_fit: np.ndarray,
    target_fit: np.ndarray,
    *,
    schedule: DirectMLPSchedule,
    hidden_width: int,
    selection_seed: int,
    split_seed: int,
) -> tuple[int, float, np.ndarray, np.ndarray]:
    result = _execute_selection(
        source_fit,
        target_fit,
        schedule=schedule,
        hidden_width=hidden_width,
        selection_seed=selection_seed,
        split_seed=split_seed,
    )
    return (
        result.best_epoch,
        result.terminal_loss,
        result.optimizer_indices,
        result.early_indices,
    )


def _fit_direct_mlp_member(
    source_fit: np.ndarray,
    target_fit: np.ndarray,
    *,
    schedule: DirectMLPSchedule,
    budget: FullCandidateParameterBudget,
    direction: Direction,
    stage: str,
    fold: int | str,
    member: int,
) -> tuple[DirectMLPModel, DirectMLPMemberReceipt]:
    if stage != "synthetic_stage_a_scaffold":
        raise CCLEMultiFMV5DirectComparatorError(
            "combined selection/refit is synthetic-only; production requires two physical attempts"
        )
    if type(schedule) is not DirectMLPSchedule:
        raise CCLEMultiFMV5DirectComparatorError("direct MLP schedule type changed")
    checked_direction = _direction(direction)
    _exact_int(member, "member", minimum=0)
    if member >= len(DIRECT_MLP_MEMBER_ROOTS):
        raise CCLEMultiFMV5DirectComparatorError("direct MLP member is not registered")
    hidden = minimum_capacity_hidden_width(
        source_fit.shape[1], target_fit.shape[1], budget
    )
    parameter_count = direct_mlp_parameter_count(
        source_fit.shape[1], target_fit.shape[1], hidden
    )
    if parameter_count < budget.maximum:
        raise CCLEMultiFMV5DirectComparatorError("direct MLP is below B_max")
    candidate_id = _mlp_candidate_id(schedule, checked_direction)
    split_seed = _per_fit_seed(
        EARLY_STOP_MEMBER_ROOTS[member],
        stage=stage,
        phase="A1",
        direction=checked_direction,
        candidate_id=candidate_id,
        fold=fold,
        member=member,
        purpose="early_stop_split",
    )
    selection_seed = _per_fit_seed(
        DIRECT_MLP_MEMBER_ROOTS[member],
        stage=stage,
        phase="A1",
        direction=checked_direction,
        candidate_id=candidate_id,
        fold=fold,
        member=member,
        purpose="optimizer_fit",
    )
    refit_seed = _per_fit_seed(
        DIRECT_MLP_MEMBER_ROOTS[member],
        stage=stage,
        phase="A1",
        direction=checked_direction,
        candidate_id=candidate_id,
        fold=fold,
        member=member,
        purpose="refit",
    )
    best_epoch, terminal, optimizer_indices, early_indices = _select_best_epoch(
        source_fit,
        target_fit,
        schedule=schedule,
        hidden_width=hidden,
        selection_seed=selection_seed,
        split_seed=split_seed,
    )
    source_scale = _fit_endpoint_scale(source_fit)
    target_scale = _fit_endpoint_scale(target_fit)
    x_refit = source_scale.transform(source_fit)
    y_refit = target_scale.transform(target_fit)
    torch = _import_torch()
    model = _build_direct_mlp(
        torch,
        input_count=source_fit.shape[1],
        output_count=target_fit.shape[1],
        hidden_width=hidden,
        model_seed=refit_seed,
    )
    optimizer = _make_optimizer(torch, model, schedule.weight_decay)
    for epoch in range(1, best_epoch + 1):
        _train_epoch(
            torch,
            model,
            optimizer,
            x_refit,
            y_refit,
            epoch=epoch,
            model_fit_seed=refit_seed,
        )
    receipt = DirectMLPMemberReceipt(
        member=member,
        hidden_width=hidden,
        parameter_count=parameter_count,
        parameter_excess=parameter_count - budget.maximum,
        nested_split_seed=split_seed,
        selection_model_seed=selection_seed,
        refit_model_seed=refit_seed,
        optimizer_fit_indices=tuple(int(item) for item in optimizer_indices),
        early_stop_indices=tuple(int(item) for item in early_indices),
        best_epoch=best_epoch,
        terminal_selection_loss=terminal,
    )
    return (
        DirectMLPModel(
            schedule=schedule,
            model=model,
            source_scale=source_scale,
            target_scale=target_scale,
            source_width=source_fit.shape[1],
            target_width=target_fit.shape[1],
            hidden_width=hidden,
            parameter_count=parameter_count,
            member=member,
            best_epoch=best_epoch,
        ),
        receipt,
    )


def predict_direct_mlp(model: DirectMLPModel, source: object) -> np.ndarray:
    if type(model) is not DirectMLPModel:
        raise CCLEMultiFMV5DirectComparatorError(
            "model must be an exact DirectMLPModel"
        )
    x = _require_f64_matrix(source, "source", minimum_rows=1)
    if x.shape[1] != model.source_width:
        raise CCLEMultiFMV5DirectComparatorError("direct MLP source width changed")
    standardized = model.source_scale.transform(x)
    torch = _import_torch()
    model.model.eval()
    device = next(model.model.parameters()).device
    with torch.no_grad():  # type: ignore[attr-defined]
        tensor = torch.from_numpy(  # type: ignore[attr-defined]
            np.ascontiguousarray(standardized, dtype=np.float32)
        ).to(device=device)
        output = model.model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)  # type: ignore[attr-defined]
        raw = output.detach().cpu().numpy()
    prediction = model.target_scale.inverse(np.asarray(raw, dtype=np.float64))
    if prediction.shape != (x.shape[0], model.target_width):
        raise CCLEMultiFMV5DirectComparatorError("direct MLP prediction axes changed")
    return _readonly(prediction)


@dataclass(frozen=True, slots=True)
class ComparatorFoldReceipt:
    family: Family
    direction: Direction
    schedule: Schedule
    fold: int
    fit_indices: tuple[int, ...]
    held_indices: tuple[int, ...]
    rbf_base_median: float | None = None
    rbf_sigma2: float | None = None
    pls_n_iter: tuple[int, ...] | None = None
    mlp_members: tuple[DirectMLPMemberReceipt, ...] = ()
    held_excluded_from_every_fit: bool = field(init=False, default=True)


@dataclass(frozen=True, slots=True)
class CandidateOOFResult:
    family: Family
    direction: Direction
    schedule: Schedule
    metric: float
    ensemble_prediction: np.ndarray
    member_predictions: tuple[np.ndarray, ...]
    folds: tuple[ComparatorFoldReceipt, ...]
    scientific_claim_authority: bool = field(init=False, default=False)
    execution_authority: bool = field(init=False, default=False)


@dataclass(frozen=True, slots=True)
class DirectFamilySelection:
    family: Family
    direction: Direction
    candidates: tuple[CandidateOOFResult, ...]
    selected_schedule: Schedule
    selected_metric: float
    selected_prediction: np.ndarray
    prediction_source: str
    execution_scope: str
    oof_provenance_status: str
    label_access_status: str
    complete_schedule_registry: bool = field(init=False, default=True)
    scientific_claim_authority: bool = field(init=False, default=False)
    execution_authority: bool = field(init=False, default=False)


def _schedule_type_for_family(family: Family) -> type[Schedule]:
    if family == "RBF_kernel_ridge":
        return RBFSchedule
    if family == "PLS2":
        return PLS2Schedule
    return DirectMLPSchedule


def _schedule_selection_key(candidate: CandidateOOFResult) -> tuple[object, ...]:
    metric = candidate.metric
    schedule = candidate.schedule
    if type(metric) is not float or not np.isfinite(metric):
        raise CCLEMultiFMV5DirectComparatorError("candidate metric is invalid")
    if type(schedule) is RBFSchedule:
        return (
            -metric,
            abs(schedule.bandwidth_multiplier - 1.0),
            schedule.ridge_alpha,
            schedule.source_standardize,
        )
    if type(schedule) is PLS2Schedule:
        return (-metric, schedule.components)
    if type(schedule) is DirectMLPSchedule:
        return (-metric, schedule.weight_decay)
    raise CCLEMultiFMV5DirectComparatorError("candidate schedule type is unknown")


def _select_complete_candidates(
    family: Family,
    direction: Direction,
    candidates: tuple[CandidateOOFResult, ...],
    *,
    prediction_source: str,
    verified_internal_oof: bool,
    execution_scope: str,
) -> DirectFamilySelection:
    expected = schedule_registry(family)
    if len(candidates) != len(expected):
        raise CCLEMultiFMV5DirectComparatorError(
            "candidate set does not contain the complete family schedule"
        )
    observed = tuple(candidate.schedule for candidate in candidates)
    if observed != expected or len(set(observed)) != len(observed):
        raise CCLEMultiFMV5DirectComparatorError(
            "candidate schedule is missing, duplicated, or reordered"
        )
    expected_type = _schedule_type_for_family(family)
    shape: tuple[int, int] | None = None
    for candidate in candidates:
        if (
            type(candidate) is not CandidateOOFResult
            or candidate.family != family
            or candidate.direction != direction
            or type(candidate.schedule) is not expected_type
            or candidate.schedule not in expected
        ):
            raise CCLEMultiFMV5DirectComparatorError(
                "candidate identity differs from the fixed family registry"
            )
        prediction = _require_f64_matrix(
            candidate.ensemble_prediction,
            "candidate ensemble prediction",
            rows=STAGE_A_ROW_COUNT,
            minimum_rows=STAGE_A_ROW_COUNT,
        )
        if shape is None:
            shape = prediction.shape
        elif prediction.shape != shape:
            raise CCLEMultiFMV5DirectComparatorError(
                "candidate prediction axes differ within a family"
            )
    selected = min(candidates, key=_schedule_selection_key)
    return DirectFamilySelection(
        family=family,
        direction=direction,
        candidates=candidates,
        selected_schedule=selected.schedule,
        selected_metric=selected.metric,
        selected_prediction=selected.ensemble_prediction,
        prediction_source=prediction_source,
        execution_scope=execution_scope,
        oof_provenance_status=(
            "verified_internal_fixed_folds"
            if verified_internal_oof
            else "unverified_caller_supplied"
        ),
        label_access_status=(
            "verified_fold_held_labels_used_only_after_prediction"
            if verified_internal_oof
            else "unverified_caller_supplied"
        ),
    )


def rank_complete_oof_predictions(
    *,
    family: Family,
    direction: Direction,
    truth: object,
    candidate_predictions: Mapping[Schedule, np.ndarray],
) -> DirectFamilySelection:
    """Recompute metrics and rank a complete caller-supplied OOF schedule.

    This is useful for producer/verifier fixtures, but the returned object is
    permanently non-authorizing and records that its arrays were supplied by
    the caller.  It never accepts a caller-supplied metric or winning label.
    """

    checked_family = _family(family)
    checked_direction = _direction(direction)
    y = _require_f64_matrix(
        truth,
        "truth",
        rows=STAGE_A_ROW_COUNT,
        minimum_rows=STAGE_A_ROW_COUNT,
    )
    expected = schedule_registry(checked_family)
    if not isinstance(candidate_predictions, Mapping):
        raise CCLEMultiFMV5DirectComparatorError(
            "candidate_predictions must be a mapping"
        )
    if set(candidate_predictions) != set(expected) or len(candidate_predictions) != len(
        expected
    ):
        raise CCLEMultiFMV5DirectComparatorError(
            "candidate predictions must contain the complete exact schedule"
        )
    candidates: list[CandidateOOFResult] = []
    for schedule in expected:
        prediction = _require_f64_matrix(
            candidate_predictions[schedule],
            "candidate prediction",
            rows=STAGE_A_ROW_COUNT,
            columns=y.shape[1],
            minimum_rows=STAGE_A_ROW_COUNT,
        )
        owned = _readonly(prediction)
        candidates.append(
            CandidateOOFResult(
                family=checked_family,
                direction=checked_direction,
                schedule=schedule,
                metric=median_target_pearson(y, owned),
                ensemble_prediction=owned,
                member_predictions=(),
                folds=(),
            )
        )
    return _select_complete_candidates(
        checked_family,
        checked_direction,
        tuple(candidates),
        prediction_source="caller_supplied_array_only_non_authorizing",
        verified_internal_oof=False,
        execution_scope="unverified_caller_supplied",
    )


def _run_candidate_oof(
    source: np.ndarray,
    target: np.ndarray,
    *,
    family: Family,
    direction: Direction,
    schedule: Schedule,
    budget: FullCandidateParameterBudget | None,
    stage: str,
) -> CandidateOOFResult:
    folds = stage_a_folds()
    prediction = np.empty_like(target)
    coverage = np.zeros(STAGE_A_ROW_COUNT, dtype=np.int64)
    member_predictions: list[np.ndarray] = []
    if family == "capacity_matched_direct_MLP":
        if type(budget) is not FullCandidateParameterBudget:
            raise CCLEMultiFMV5DirectComparatorError(
                "direct MLP requires the exact full-candidate budget"
            )
        if type(schedule) is not DirectMLPSchedule:
            raise CCLEMultiFMV5DirectComparatorError("MLP schedule type mismatch")
        member_predictions = [np.empty_like(target) for _ in DIRECT_MLP_MEMBER_ROOTS]
    fold_receipts: list[ComparatorFoldReceipt] = []
    for split in folds:
        fit = np.ascontiguousarray(np.asarray(split.fit_indices, dtype=np.int64))
        held = np.ascontiguousarray(np.asarray(split.held_indices, dtype=np.int64))
        _validate_fold_indices(fit, held, split.fold)
        x_fit = np.ascontiguousarray(source[fit], dtype=np.float64)
        y_fit = np.ascontiguousarray(target[fit], dtype=np.float64)
        x_held = np.ascontiguousarray(source[held], dtype=np.float64)
        if family == "RBF_kernel_ridge":
            if type(schedule) is not RBFSchedule:
                raise CCLEMultiFMV5DirectComparatorError("RBF schedule type mismatch")
            model = fit_rbf(x_fit, y_fit, schedule)
            held_prediction = predict_rbf(model, x_held)
            receipt = ComparatorFoldReceipt(
                family=family,
                direction=direction,
                schedule=schedule,
                fold=split.fold,
                fit_indices=split.fit_indices,
                held_indices=split.held_indices,
                rbf_base_median=model.base_median,
                rbf_sigma2=model.sigma2,
            )
        elif family == "PLS2":
            if type(schedule) is not PLS2Schedule:
                raise CCLEMultiFMV5DirectComparatorError("PLS2 schedule type mismatch")
            model = fit_pls2(x_fit, y_fit, schedule)
            held_prediction = predict_pls2(model, x_held)
            receipt = ComparatorFoldReceipt(
                family=family,
                direction=direction,
                schedule=schedule,
                fold=split.fold,
                fit_indices=split.fit_indices,
                held_indices=split.held_indices,
                pls_n_iter=model.n_iter,
            )
        else:
            if type(schedule) is not DirectMLPSchedule or budget is None:
                raise CCLEMultiFMV5DirectComparatorError(
                    "MLP fit contract is incomplete"
                )
            mlp_receipts: list[DirectMLPMemberReceipt] = []
            for member in range(len(DIRECT_MLP_MEMBER_ROOTS)):
                model, member_receipt = _fit_direct_mlp_member(
                    x_fit,
                    y_fit,
                    schedule=schedule,
                    budget=budget,
                    direction=direction,
                    stage=stage,
                    fold=split.fold,
                    member=member,
                )
                member_prediction = predict_direct_mlp(model, x_held)
                member_predictions[member][held] = member_prediction
                # Local nested indices map only through the sorted fold-fit vector.
                optimizer_global = tuple(
                    int(fit[item]) for item in member_receipt.optimizer_fit_indices
                )
                early_global = tuple(
                    int(fit[item]) for item in member_receipt.early_stop_indices
                )
                if set(optimizer_global) & set(split.held_indices) or set(
                    early_global
                ) & set(split.held_indices):
                    raise CCLEMultiFMV5DirectComparatorError(
                        "outer-held row entered the MLP nested lifecycle"
                    )
                mlp_receipts.append(
                    DirectMLPMemberReceipt(
                        member=member_receipt.member,
                        hidden_width=member_receipt.hidden_width,
                        parameter_count=member_receipt.parameter_count,
                        parameter_excess=member_receipt.parameter_excess,
                        nested_split_seed=member_receipt.nested_split_seed,
                        selection_model_seed=member_receipt.selection_model_seed,
                        refit_model_seed=member_receipt.refit_model_seed,
                        optimizer_fit_indices=optimizer_global,
                        early_stop_indices=early_global,
                        best_epoch=member_receipt.best_epoch,
                        terminal_selection_loss=member_receipt.terminal_selection_loss,
                    )
                )
            held_prediction = np.mean(
                np.stack([values[held] for values in member_predictions], axis=0),
                axis=0,
                dtype=np.float64,
            )
            receipt = ComparatorFoldReceipt(
                family=family,
                direction=direction,
                schedule=schedule,
                fold=split.fold,
                fit_indices=split.fit_indices,
                held_indices=split.held_indices,
                mlp_members=tuple(mlp_receipts),
            )
        if held_prediction.shape != (held.size, target.shape[1]) or not np.all(
            np.isfinite(held_prediction)
        ):
            raise CCLEMultiFMV5DirectComparatorError(
                "held prediction is incomplete or non-finite"
            )
        prediction[held] = held_prediction
        coverage[held] += 1
        fold_receipts.append(receipt)
    if not np.array_equal(coverage, np.ones(STAGE_A_ROW_COUNT, dtype=np.int64)):
        raise CCLEMultiFMV5DirectComparatorError("OOF coverage is not exactly one")
    if not np.all(np.isfinite(prediction)):
        raise CCLEMultiFMV5DirectComparatorError("OOF prediction is non-finite")
    frozen_members: tuple[np.ndarray, ...] = ()
    if member_predictions:
        for values in member_predictions:
            if not np.all(np.isfinite(values)):
                raise CCLEMultiFMV5DirectComparatorError(
                    "direct MLP member OOF prediction is incomplete"
                )
        frozen_members = tuple(_readonly(values) for values in member_predictions)
        reconstructed = np.mean(
            np.stack(frozen_members, axis=0), axis=0, dtype=np.float64
        )
        if not np.array_equal(prediction, reconstructed):
            raise CCLEMultiFMV5DirectComparatorError(
                "direct MLP ensemble is not the arithmetic member mean"
            )
    owned_prediction = _readonly(prediction)
    return CandidateOOFResult(
        family=family,
        direction=direction,
        schedule=schedule,
        metric=median_target_pearson(target, owned_prediction),
        ensemble_prediction=owned_prediction,
        member_predictions=frozen_members,
        folds=tuple(fold_receipts),
    )


def run_stage_a_family_oof(
    *,
    source: object,
    target: object,
    family: Family,
    direction: Direction,
    budget: FullCandidateParameterBudget | None = None,
    execution_scope: str,
) -> DirectFamilySelection:
    """Synthetic whole-family scaffold; never a production worker lifecycle."""

    if execution_scope != "synthetic_scaffold_non_authorizing":
        raise CCLEMultiFMV5DirectComparatorError(
            "whole-family OOF is synthetic scaffold only; production must use "
            "one StageAFitTask/member worker per fresh process"
        )

    checked_family = _family(family)
    checked_direction = _direction(direction)
    x, y = _validate_pair(source, target)
    _validate_directional_widths(x, y, checked_direction)
    if checked_family == "capacity_matched_direct_MLP":
        if type(budget) is not FullCandidateParameterBudget:
            raise CCLEMultiFMV5DirectComparatorError(
                "direct MLP Stage-A run requires the exact 16-candidate budget"
            )
    elif budget is not None and type(budget) is not FullCandidateParameterBudget:
        raise CCLEMultiFMV5DirectComparatorError("invalid optional budget")
    candidates = tuple(
        _run_candidate_oof(
            x,
            y,
            family=checked_family,
            direction=checked_direction,
            schedule=schedule,
            budget=budget,
            stage="synthetic_stage_a_scaffold",
        )
        for schedule in schedule_registry(checked_family)
    )
    return _select_complete_candidates(
        checked_family,
        checked_direction,
        candidates,
        prediction_source="internally_generated_fixed_five_fold_oof",
        verified_internal_oof=True,
        execution_scope="synthetic_scaffold_non_authorizing",
    )


@dataclass(frozen=True, slots=True)
class BidirectionalDirectOOFResult:
    forward: tuple[DirectFamilySelection, ...]
    reverse: tuple[DirectFamilySelection, ...]
    scientific_claim_authority: bool = field(init=False, default=False)
    execution_authority: bool = field(init=False, default=False)


def run_stage_a_bidirectional_direct_oof(
    *,
    metabolites: object,
    proteins: object,
    budget: FullCandidateParameterBudget,
    execution_scope: str,
) -> BidirectionalDirectOOFResult:
    """Synthetic six-atom scaffold; never a production worker lifecycle."""

    if execution_scope != "synthetic_scaffold_non_authorizing":
        raise CCLEMultiFMV5DirectComparatorError(
            "bidirectional whole-family OOF is synthetic scaffold only"
        )

    m = _require_f64_matrix(
        metabolites,
        "metabolites",
        rows=STAGE_A_ROW_COUNT,
        columns=METABOLITE_COUNT,
    )
    p = _require_f64_matrix(
        proteins,
        "proteins",
        rows=STAGE_A_ROW_COUNT,
        columns=PROTEIN_COUNT,
    )
    if type(budget) is not FullCandidateParameterBudget:
        raise CCLEMultiFMV5DirectComparatorError("invalid full-candidate budget")
    families: tuple[Family, ...] = (
        "RBF_kernel_ridge",
        "PLS2",
        "capacity_matched_direct_MLP",
    )
    forward = tuple(
        run_stage_a_family_oof(
            source=m,
            target=p,
            family=family,
            direction="forward",
            budget=budget,
            execution_scope=execution_scope,
        )
        for family in families
    )
    reverse = tuple(
        run_stage_a_family_oof(
            source=p,
            target=m,
            family=family,
            direction="reverse",
            budget=budget,
            execution_scope=execution_scope,
        )
        for family in families
    )
    return BidirectionalDirectOOFResult(forward=forward, reverse=reverse)


@dataclass(frozen=True, slots=True)
class MaskFamilyRefit:
    family: Family
    mask_sha256: str
    observed_source_count: int
    row_indices: tuple[int, ...]
    selection: DirectFamilySelection
    prediction: np.ndarray
    mlp_members: tuple[DirectMLPMemberReceipt, ...] = ()
    all_training_rows_used: bool = field(init=False, default=True)
    prediction_labels_accessed: bool = field(init=False, default=False)
    scientific_claim_authority: bool = field(init=False, default=False)


@dataclass(frozen=True, slots=True)
class ReverseMaskReselectionResult:
    mask_refits: tuple[MaskFamilyRefit, ...]
    family_predictions: tuple[tuple[Family, np.ndarray], ...]
    prediction_rows: int
    training_rows: int = field(init=False, default=STAGE_A_ROW_COUNT)
    target_count: int = field(init=False, default=METABOLITE_COUNT)
    evidence_status: str = field(
        init=False, default="generic_analytic_non_stage_c_non_authorizing"
    )
    exact_stage_c: bool = field(init=False, default=False)
    scientific_claim_authority: bool = field(init=False, default=False)
    execution_authority: bool = field(init=False, default=False)


def _refit_selected_family(
    source_train: np.ndarray,
    target_train: np.ndarray,
    source_predict: np.ndarray,
    *,
    selection: DirectFamilySelection,
    budget: FullCandidateParameterBudget | None,
    mask_sha256: str,
) -> tuple[np.ndarray, tuple[DirectMLPMemberReceipt, ...]]:
    schedule = selection.selected_schedule
    if selection.family == "RBF_kernel_ridge":
        if type(schedule) is not RBFSchedule:
            raise CCLEMultiFMV5DirectComparatorError(
                "selected RBF schedule changed type"
            )
        prediction = predict_rbf(
            fit_rbf(source_train, target_train, schedule), source_predict
        )
        return prediction, ()
    if selection.family == "PLS2":
        if type(schedule) is not PLS2Schedule:
            raise CCLEMultiFMV5DirectComparatorError(
                "selected PLS2 schedule changed type"
            )
        prediction = predict_pls2(
            fit_pls2(source_train, target_train, schedule), source_predict
        )
        return prediction, ()
    del budget, mask_sha256
    raise CCLEMultiFMV5DirectComparatorError(
        "stochastic reverse-mask MLP is blocked until a Stage-C child protocol "
        "registers mask_id, stage, phase, purpose, seed roots and preimages"
    )


def reselect_refit_reverse_masks_analytic(
    *,
    training_proteins: object,
    training_metabolites: object,
    masked_protein_source: object,
    source_masks: object,
) -> ReverseMaskReselectionResult:
    """Generic RBF/PLS mask grouping; explicitly not an exact Stage-C run.

    Only complete 421-row training endpoints and the masked prediction source
    are accepted. Prediction row count is generic and carries no cohort
    meaning. The still-unfrozen stochastic MLP family is intentionally absent.
    """

    proteins = _require_f64_matrix(
        training_proteins,
        "training_proteins",
        rows=STAGE_A_ROW_COUNT,
        columns=PROTEIN_COUNT,
    )
    metabolites = _require_f64_matrix(
        training_metabolites,
        "training_metabolites",
        rows=STAGE_A_ROW_COUNT,
        columns=METABOLITE_COUNT,
    )
    masked, masks = _require_masked_f64_matrix(
        masked_protein_source,
        source_masks,
        "masked_protein_source",
        columns=PROTEIN_COUNT,
    )
    families: tuple[Family, ...] = (
        "RBF_kernel_ridge",
        "PLS2",
    )
    assembled = {
        family: np.empty((masked.shape[0], METABOLITE_COUNT), dtype=np.float64)
        for family in families
    }
    assigned = {
        family: np.zeros(masked.shape[0], dtype=np.int64) for family in families
    }
    unique: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[bytes] = set()
    for row in range(masked.shape[0]):
        key = masks[row].tobytes(order="C")
        if key in seen:
            continue
        seen.add(key)
        mask = np.ascontiguousarray(masks[row], dtype=np.bool_)
        rows = np.flatnonzero(np.all(masks == mask, axis=1)).astype(np.int64)
        unique.append((mask, np.ascontiguousarray(rows)))
    refits: list[MaskFamilyRefit] = []
    for mask, rows in unique:
        observed = np.flatnonzero(mask).astype(np.int64)
        if observed.size == 0:
            raise CCLEMultiFMV5DirectComparatorError("empty mask is STOP")
        source_train = np.ascontiguousarray(proteins[:, observed], dtype=np.float64)
        source_predict = np.ascontiguousarray(
            masked[np.ix_(rows, observed)], dtype=np.float64
        )
        if not np.all(np.isfinite(source_predict)):
            raise CCLEMultiFMV5DirectComparatorError(
                "observed mask extraction contains a non-finite value"
            )
        mask_sha256 = hashlib.sha256(
            mask.astype(np.uint8).tobytes(order="C")
        ).hexdigest()
        for family in families:
            candidates = tuple(
                _run_candidate_oof(
                    source_train,
                    metabolites,
                    family=family,
                    direction="reverse",
                    schedule=schedule,
                    budget=None,
                    stage="generic_analytic_mask_fixture",
                )
                for schedule in schedule_registry(family)
            )
            selection = _select_complete_candidates(
                family,
                "reverse",
                candidates,
                prediction_source="mask_specific_fixed_five_fold_oof",
                verified_internal_oof=True,
                execution_scope="generic_analytic_non_stage_c_non_authorizing",
            )
            prediction, member_receipts = _refit_selected_family(
                source_train,
                metabolites,
                source_predict,
                selection=selection,
                budget=None,
                mask_sha256=mask_sha256,
            )
            if prediction.shape != (rows.size, METABOLITE_COUNT):
                raise CCLEMultiFMV5DirectComparatorError(
                    "mask-specific prediction axes changed"
                )
            assembled[family][rows] = prediction
            assigned[family][rows] += 1
            refits.append(
                MaskFamilyRefit(
                    family=family,
                    mask_sha256=mask_sha256,
                    observed_source_count=int(observed.size),
                    row_indices=tuple(int(item) for item in rows),
                    selection=selection,
                    prediction=_readonly(prediction),
                    mlp_members=member_receipts,
                )
            )
    frozen_predictions: list[tuple[Family, np.ndarray]] = []
    for family in families:
        if not np.array_equal(
            assigned[family], np.ones(masked.shape[0], dtype=np.int64)
        ):
            raise CCLEMultiFMV5DirectComparatorError(
                "mask-specific row assignment is not exactly one"
            )
        if not np.all(np.isfinite(assembled[family])):
            raise CCLEMultiFMV5DirectComparatorError(
                "assembled mask-specific prediction is non-finite"
            )
        frozen_predictions.append((family, _readonly(assembled[family])))
    return ReverseMaskReselectionResult(
        mask_refits=tuple(refits),
        family_predictions=tuple(frozen_predictions),
        prediction_rows=masked.shape[0],
    )


def reselect_refit_reverse_masks(
    *,
    training_proteins: object,
    training_metabolites: object,
    masked_protein_source: object,
    source_masks: object,
    budget: FullCandidateParameterBudget,
) -> ReverseMaskReselectionResult:
    """Fail closed until the separately frozen Stage-C child exists."""

    del (
        training_proteins,
        training_metabolites,
        masked_protein_source,
        source_masks,
        budget,
    )
    raise CCLEMultiFMV5DirectComparatorError(
        "Stage-C stochastic reverse-mask execution is blocked: the child "
        "protocol has not registered mask_id, stage/phase/purpose, seed roots, "
        "preimage receipts, runtime binding or create-once evidence roots"
    )


def immutable_family_prediction_map(
    result: ReverseMaskReselectionResult,
) -> Mapping[Family, np.ndarray]:
    """Expose the assembled predictions without granting mutation or authority."""

    if type(result) is not ReverseMaskReselectionResult:
        raise CCLEMultiFMV5DirectComparatorError(
            "result must be an exact ReverseMaskReselectionResult"
        )
    return MappingProxyType(dict(result.family_predictions))


__all__ = [
    "BidirectionalDirectOOFResult",
    "CCLEMultiFMV5DirectComparatorError",
    "CandidateOOFResult",
    "ComparatorFoldReceipt",
    "CoreCandidateParameterInventory",
    "CoreNamedParameterInventory",
    "DirectFamilySelection",
    "DirectMLPModel",
    "DirectAttemptRuntime",
    "DirectOptimizerFitOutput",
    "DirectRefitOutput",
    "SyntheticDirectSelectionParent",
    "DirectMLPMemberReceipt",
    "DirectMLPSchedule",
    "FullCandidateParameterBudget",
    "MaskFamilyRefit",
    "PLS2Model",
    "PLS2Schedule",
    "RBFSchedule",
    "RBFModel",
    "ReverseMaskReselectionResult",
    "StaticVerifierBinding",
    "bind_parameter_budget_to_static_verifier",
    "canonical_schedule_id",
    "canonical_schedule_settings",
    "direct_mlp_parameter_count",
    "direct_mlp_optimizer_fit_attempt",
    "direct_mlp_refit_attempt",
    "direct_mlp_schedule_registry",
    "export_exact_core_full_candidate_parameter_budget",
    "fit_pls2",
    "fit_rbf",
    "full_candidate_ids",
    "immutable_family_prediction_map",
    "median_target_pearson",
    "minimum_capacity_hidden_width",
    "pls2_schedule_registry",
    "predict_direct_mlp",
    "predict_pls2",
    "predict_rbf",
    "rank_complete_oof_predictions",
    "rbf_schedule_registry",
    "reselect_refit_reverse_masks",
    "reselect_refit_reverse_masks_analytic",
    "run_stage_a_bidirectional_direct_oof",
    "run_stage_a_family_oof",
    "schedule_registry",
    "stage_a_folds",
]

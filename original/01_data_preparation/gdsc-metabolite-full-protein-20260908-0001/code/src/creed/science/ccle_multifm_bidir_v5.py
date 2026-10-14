"""Protocol-level, array-only V5 single-fold model core.

This module implements only the exact network and one caller-defined fit.  It
has no path, cohort, fold-construction, OOF, activation, result-selection, or
claim-authority API.  In particular, it cannot name or open any audit or final
partition.

The two inference graphs are deliberately narrow::

    measured metabolite tokens -> metabolite ISAB -> shared PMA/core slots
        -> ESM-conditioned protein query decoder -> proteins

    observed (protein abundance, ESM identity) tokens -> protein ISAB
        -> shared PMA/core slots -> metabolite query decoder -> metabolites

scGPT is a fold-fit, training-only teacher.  Neither inference method accepts
scGPT or the opposite measured endpoint, and no raw source-to-output skip is
present.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import numpy as np


V5Control = Literal[
    "full",
    "no_scgpt",
    "no_esm_identity_only",
    "scgpt_row_deranged",
    "scgpt_covariance_random",
    "esm2_deranged",
    "esm2_covariance_random",
    "route_shuffle",
]
V5_CONTROLS: tuple[str, ...] = (
    "full",
    "no_scgpt",
    "no_esm_identity_only",
    "scgpt_row_deranged",
    "scgpt_covariance_random",
    "esm2_deranged",
    "esm2_covariance_random",
    "route_shuffle",
)
V5_METABOLITE_COUNT = 225
V5_PROTEIN_COUNT = 584
V5_SCGPT_DIMENSION = 512
V5_ESM2_DIMENSION = 1280

_SEMANTIC_ESM_CONTROLS = frozenset(V5_CONTROLS) - {"no_esm_identity_only"}
_REAL_SCGPT_CONTROLS = frozenset(V5_CONTROLS) - {
    "no_scgpt",
    "scgpt_covariance_random",
}
_NO_INTERVENTION_SEED_CONTROLS = frozenset({"full", "no_scgpt", "no_esm_identity_only"})
_REQUIRED_INTERVENTION_SEED_CONTROLS = frozenset(V5_CONTROLS) - (
    _NO_INTERVENTION_SEED_CONTROLS
)
_CUDA_ENVIRONMENT = {
    "PYTHONHASHSEED": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}
_DEVICE_PATTERN = re.compile(r"(?:cpu|cuda:[0-9]+)\Z")
_DROPOUT_ROOT_BY_MEMBER = (321870185, 1660126547, 659023361)
_DIRECT_MLP_SHAPE_BY_DIRECTION = {
    "forward": (V5_METABOLITE_COUNT, V5_PROTEIN_COUNT, 666),
    "reverse": (V5_PROTEIN_COUNT, V5_METABOLITE_COUNT, 667),
}
_DIRECT_MLP_WEIGHT_DECAY_TOKENS = ("0p00001", "0p00010", "0p00100")


class CCLEMultiFMV5Error(ValueError):
    """Raised when an exact V5 single-fold contract is violated."""


def _exact_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:  # bool and NumPy ints fail closed
        raise CCLEMultiFMV5Error(f"{label} must be an exact integer >= {minimum}")
    return value


def _exact_float(value: object, label: str) -> float:
    if type(value) is not float or not np.isfinite(value):
        raise CCLEMultiFMV5Error(f"{label} must be an exact finite float")
    return value


def _validate_direct_candidate_id(direction: str, candidate_id: str) -> None:
    """Validate the separate Stage-A direct-MLP lifecycle registry."""

    if direction not in _DIRECT_MLP_SHAPE_BY_DIRECTION:
        raise CCLEMultiFMV5Error("direct lifecycle direction is unregistered")
    if type(candidate_id) is not str:
        raise CCLEMultiFMV5Error("direct candidate_id must be an exact string")
    inputs, outputs, hidden = _DIRECT_MLP_SHAPE_BY_DIRECTION[direction]
    registered = {
        f"mlp-in{inputs:04d}-out{outputs:04d}-h{hidden:06d}-wd{token}"
        for token in _DIRECT_MLP_WEIGHT_DECAY_TOKENS
    }
    if candidate_id not in registered:
        raise CCLEMultiFMV5Error(
            "capacity direct candidate_id is not a registered token"
        )


@dataclass(frozen=True)
class MultiFMBidirV5Config:
    """Exact architecture and optimizer fields for one fold-local fit.

    ``epochs`` is supplied by the outer, separately governed lifecycle.  This
    core does not implement early stopping or choose it.  The architecture
    fields otherwise admit only values in the V5 candidate grid.
    """

    latent_slots: int = 4
    model_dimension: int = 32
    teacher_dimension: int = 32
    inducing_points: int = 16
    isab_blocks_per_modality: int = 2
    latent_core_blocks: int = 2
    attention_heads: int = 4
    ffn_multiplier: int = 2
    layer_norm_epsilon: float = 1e-5
    source_token_dropout: float = 0.0
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.0003
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0001
    gradient_clip_norm: float = 1.0
    cross_loss_weight: float = 1.0
    self_loss_weight: float = 0.25
    teacher_loss_weight: float = 0.1
    state_alignment_weight: float = 0.1
    teacher_whitening_epsilon: float = 1e-6
    scale_epsilon: float = 1e-8
    seed: int = 20260902
    intervention_seed: int | None = None
    device: str = "cpu"
    deterministic_algorithms: bool = True
    cpu_threads: int = 1
    interop_threads: int = 1

    def __post_init__(self) -> None:
        for label in (
            "latent_slots",
            "model_dimension",
            "teacher_dimension",
            "inducing_points",
            "isab_blocks_per_modality",
            "latent_core_blocks",
            "attention_heads",
            "ffn_multiplier",
            "epochs",
            "batch_size",
            "cpu_threads",
            "interop_threads",
        ):
            _exact_int(getattr(self, label), label, minimum=1)
        value = _exact_int(self.seed, "seed")
        if value >= 2**63:
            raise CCLEMultiFMV5Error("seed must be smaller than 2^63")
        for label in (
            "layer_norm_epsilon",
            "source_token_dropout",
            "learning_rate",
            "adam_beta1",
            "adam_beta2",
            "adam_epsilon",
            "weight_decay",
            "gradient_clip_norm",
            "cross_loss_weight",
            "self_loss_weight",
            "teacher_loss_weight",
            "state_alignment_weight",
            "teacher_whitening_epsilon",
            "scale_epsilon",
        ):
            _exact_float(getattr(self, label), label)

        exact_fields: dict[str, object] = {
            "teacher_dimension": 32,
            "inducing_points": 16,
            "isab_blocks_per_modality": 2,
            "latent_core_blocks": 2,
            "attention_heads": 4,
            "ffn_multiplier": 2,
            "layer_norm_epsilon": 1e-5,
            "batch_size": 32,
            "learning_rate": 0.0003,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 1.0,
            "cross_loss_weight": 1.0,
            "self_loss_weight": 0.25,
            "state_alignment_weight": 0.1,
            "teacher_whitening_epsilon": 1e-6,
            "scale_epsilon": 1e-8,
            "cpu_threads": 1,
            "interop_threads": 1,
        }
        for label, expected in exact_fields.items():
            if getattr(self, label) != expected:
                raise CCLEMultiFMV5Error(
                    f"{label} is definitionally fixed at {expected!r}"
                )
        if self.latent_slots not in (4, 8):
            raise CCLEMultiFMV5Error("latent_slots must be one of the V5 grid {4,8}")
        if self.model_dimension not in (32, 64):
            raise CCLEMultiFMV5Error(
                "model_dimension must be one of the V5 grid {32,64}"
            )
        if self.epochs > 1000:
            raise CCLEMultiFMV5Error("epochs cannot exceed the V5 maximum of 1000")
        if self.model_dimension % self.attention_heads:
            raise CCLEMultiFMV5Error("model_dimension must divide attention_heads")
        if self.source_token_dropout not in (0.0, 0.05):
            raise CCLEMultiFMV5Error(
                "source_token_dropout must be one of the V5 grid {0.0,0.05}"
            )
        if self.teacher_loss_weight not in (0.0, 0.1, 0.5):
            raise CCLEMultiFMV5Error("teacher_loss_weight must be 0.0, 0.1, or 0.5")
        if not (0.0 < self.adam_beta1 < 1.0 and 0.0 < self.adam_beta2 < 1.0):
            raise CCLEMultiFMV5Error("Adam beta values must lie strictly in (0,1)")
        if self.learning_rate <= 0.0 or self.adam_epsilon <= 0.0:
            raise CCLEMultiFMV5Error("optimizer learning_rate/epsilon must be positive")
        if self.weight_decay < 0.0 or self.gradient_clip_norm <= 0.0:
            raise CCLEMultiFMV5Error("weight_decay/gradient clip is invalid")
        if (
            type(self.deterministic_algorithms) is not bool
            or not self.deterministic_algorithms
        ):
            raise CCLEMultiFMV5Error("deterministic_algorithms must be exactly True")
        if type(self.device) is not str or not _DEVICE_PATTERN.fullmatch(self.device):
            raise CCLEMultiFMV5Error("device must be exactly 'cpu' or 'cuda:<index>'")


@dataclass(frozen=True)
class V5DropoutSeedContext:
    """Canonical single-fit fields needed by the protocol dropout preimage."""

    dropout_root: int
    stage: str
    phase: str
    family: str
    direction_or_joint: str
    candidate_id: str
    outer_repeat: int | str
    outer_fold: int | str
    inner_repeat: int | str
    inner_fold: int | str
    member: int
    purpose: str
    registry_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        root = _exact_int(self.dropout_root, "dropout_root")
        if root >= 2**32:
            raise CCLEMultiFMV5Error("dropout_root must be an unsigned 32-bit root")
        _exact_int(self.member, "member")
        for label in (
            "stage",
            "phase",
            "family",
            "direction_or_joint",
            "candidate_id",
            "purpose",
        ):
            value = getattr(self, label)
            if type(value) is not str or not value or "|" in value:
                raise CCLEMultiFMV5Error(
                    f"{label} must be a non-empty canonical token without '|'"
                )
        if self.stage != "stage_a":
            raise CCLEMultiFMV5Error("dropout stage must be exactly 'stage_a'")
        if self.phase not in {"A1", "A2"}:
            raise CCLEMultiFMV5Error("dropout phase must be exactly A1 or A2")
        if self.direction_or_joint not in {"forward", "reverse", "joint", "-"}:
            raise CCLEMultiFMV5Error("dropout direction token is unregistered")
        if self.purpose not in {"optimizer_fit", "refit"}:
            raise CCLEMultiFMV5Error(
                "dropout purpose must be optimizer_fit or refit; early-stop is forbidden"
            )
        for label in (
            "outer_repeat",
            "outer_fold",
            "inner_repeat",
            "inner_fold",
        ):
            index = getattr(self, label)
            if index != "-":
                _exact_int(index, label)
        if type(self.registry_positions) is not tuple or not self.registry_positions:
            raise CCLEMultiFMV5Error(
                "registry_positions must be a non-empty tuple of exact integers"
            )
        previous = -1
        for position in self.registry_positions:
            _exact_int(position, "registry position")
            if position >= 2**63:
                raise CCLEMultiFMV5Error("registry position must be smaller than 2^63")
            if position <= previous:
                raise CCLEMultiFMV5Error(
                    "registry_positions must be unique and strictly increasing"
                )
            previous = position
        if self.outer_repeat != 0:
            raise CCLEMultiFMV5Error("Stage-A dropout outer_repeat must be exactly 0")
        if type(self.outer_fold) is not int or not 0 <= self.outer_fold <= 4:
            raise CCLEMultiFMV5Error("Stage-A dropout outer_fold must be an exact 0..4")
        if self.inner_repeat != "-" or self.inner_fold != "-":
            raise CCLEMultiFMV5Error(
                "Stage-A dropout inner indices must be exactly '-'"
            )
        if not 0 <= self.member <= 2:
            raise CCLEMultiFMV5Error("Stage-A dropout member must be an exact 0..2")
        if self.dropout_root != _DROPOUT_ROOT_BY_MEMBER[self.member]:
            raise CCLEMultiFMV5Error("dropout_root is not registered for this member")
        if self.family not in V5_CONTROLS:
            raise CCLEMultiFMV5Error("dropout family is not a registered neural family")
        if self.family == "full":
            valid_route = self.phase == "A1" and self.direction_or_joint == "joint"
        else:
            valid_route = self.phase == "A2" and self.direction_or_joint == "joint"
        if not valid_route:
            raise CCLEMultiFMV5Error(
                "dropout phase/direction combination is not registered for this family"
            )


@dataclass(frozen=True)
class DropoutSeedReceipt:
    root_role: str
    root_value: int
    stage: str
    phase: str
    family: str
    direction_or_joint: str
    candidate_id: str
    outer_repeat: int | str
    outer_fold: int | str
    inner_repeat: int | str
    inner_fold: int | str
    member: int
    purpose: str
    epoch: int
    modality: str
    registry_position: int
    full_preimage: str
    preimage_sha256: str
    derivation: str
    derived_value: int
    scientific_authority: bool = False


@dataclass(frozen=True)
class DropoutSeedManifest:
    """Compact lossless recipe and digest for every consumed dropout seed."""

    context: V5DropoutSeedContext
    schema_version: str
    root_role: str
    root_value: int
    first_epoch: int
    last_epoch: int
    epoch_count: int
    modality_order: tuple[str, ...]
    positions_dtype: str
    positions_shape: tuple[int, ...]
    positions_byte_count: int
    positions_raw_encoding: str
    positions_raw_hex: str
    positions_sha256: str
    seed_table_dtype: str
    seed_table_shape: tuple[int, ...]
    seed_table_byte_count: int
    seed_table_raw_encoding: str
    seed_table_raw_hex: str
    seed_table_sha256: str
    seed_count: int
    exact_derived_count: int
    stream_item_count: int
    formula_version: str
    stream_encoding: str
    stream_sha256: str
    all_preimages_unique: bool
    scientific_authority: bool = False


@dataclass(frozen=True)
class ArrayReceipt:
    shape: tuple[int, ...]
    dtype: str
    sha256: str


@dataclass(frozen=True)
class FeatureScaler:
    """Observed-only float64 statistics fit within the supplied fit rows."""

    mean: np.ndarray
    scale: np.ndarray

    def transform(self, matrix: np.ndarray, mask: np.ndarray) -> np.ndarray:
        safe = np.zeros(matrix.shape, dtype=np.float64)
        safe[mask] = matrix[mask]
        safe -= self.mean
        safe /= self.scale
        safe[~mask] = 0.0  # storage only; masked tokens never enter attention
        return np.ascontiguousarray(safe, dtype=np.float32)

    def inverse(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float64)
        return values * self.scale + self.mean


@dataclass(frozen=True)
class PCATeacherReceipt:
    input: ArrayReceipt
    mean: ArrayReceipt
    singular_values: ArrayReceipt
    components: ArrayReceipt
    canonical_sign_indices: ArrayReceipt
    eigenvalues: ArrayReceipt
    whitening_scale: ArrayReceipt
    transformed_fit: ArrayReceipt
    scipy_lapack_driver: str = "gesvd"
    full_matrices: bool = False
    check_finite: bool = True
    canonical_sign_rule: str = "largest_abs_loading_positive_lowest_index_tie"


@dataclass(frozen=True)
class WhitenedPCATeacher:
    """Float64, centered, canonical SciPy-gesvd teacher transform."""

    mean: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray
    eigenvalues: np.ndarray
    whitening_scale: np.ndarray
    receipt: PCATeacherReceipt

    @property
    def output_dimension(self) -> int:
        return int(self.components.shape[1])

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        values = _finite_matrix(matrix, "scGPT teacher transform")
        if values.shape[1] != self.mean.shape[0]:
            raise CCLEMultiFMV5Error("scGPT teacher feature count changed")
        state = ((values - self.mean) @ self.components) / self.whitening_scale
        if not np.isfinite(state).all():
            raise CCLEMultiFMV5Error("scGPT teacher transform is non-finite")
        return _readonly(state, dtype=np.float64)


@dataclass(frozen=True)
class DeterminismReceipt:
    seed: int
    device: str
    torch_version: str
    cuda_available: bool
    fresh_cuda_process_verified: bool
    deterministic_algorithms: bool
    warn_only: bool
    cudnn_deterministic: bool | None
    cudnn_benchmark: bool | None
    cuda_matmul_tf32: bool | None
    cudnn_tf32: bool | None
    flash_sdp_enabled: bool | None
    mem_efficient_sdp_enabled: bool | None
    math_sdp_enabled: bool | None
    default_dtype: str
    cublas_workspace_config: str | None
    cpu_threads: int
    interop_threads: int
    required_environment: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class NamedParameterReceipt:
    name: str
    shape: tuple[int, ...]
    numel: int
    dtype: str
    initialization_seed: int
    initialization_sha256: str
    inference_path: bool
    training_only: bool


@dataclass(frozen=True)
class ParameterCount:
    total: int
    trainable: int
    inference_path: int
    training_only: int
    by_module: tuple[tuple[str, int], ...]
    named: tuple[NamedParameterReceipt, ...]
    learned_preprocessing_parameters_included: bool = False


@dataclass(frozen=True)
class ControlParameterReceipt:
    control: str
    full_esm_projection_parameters: int
    effective_identity_parameters: int
    identity_capacity_excess: int
    learned_identity_width: int | None
    teacher_head_parameters: int
    teacher_head_active: bool
    teacher_head_in_optimizer_registry: bool
    teacher_head_optimizer_state_created: bool
    teacher_head_initial_sha256: str
    teacher_head_final_sha256: str
    teacher_head_unchanged: bool
    initial_model_sha256: str
    final_model_sha256: str
    same_prediction_interface: bool
    scientific_authority: bool = False


@dataclass(frozen=True)
class RouteShuffleIndexReceipt:
    permutation: tuple[int, ...]
    self_metabolite_source: tuple[int, ...]
    self_metabolite_target: tuple[int, ...]
    self_protein_source: tuple[int, ...]
    self_protein_target: tuple[int, ...]
    cross_forward_metabolite_source: tuple[int, ...]
    cross_forward_protein_target: tuple[int, ...]
    cross_reverse_protein_source: tuple[int, ...]
    cross_reverse_metabolite_target: tuple[int, ...]
    anchor_metabolite_source: tuple[int, ...]
    anchor_protein_source: tuple[int, ...]
    anchor_teacher_target: tuple[int, ...]
    state_metabolite_source: tuple[int, ...]
    state_protein_source: tuple[int, ...]


@dataclass(frozen=True)
class ControlTransformReceipt:
    control: str
    intervention_seed: int | None
    scgpt_input_opened: bool
    esm_input_opened: bool
    effective_teacher: ArrayReceipt | None
    effective_esm: ArrayReceipt | None
    permutation: tuple[int, ...] | None
    route_indices: RouteShuffleIndexReceipt | None
    transform: str
    scientific_authority: bool = False


@dataclass(frozen=True)
class EpochLossReceipt:
    epoch: int
    cross: float
    self_reconstruction: float
    teacher: float
    state_alignment: float
    total: float


@dataclass(frozen=True)
class EarlyStopLossReceipt:
    """One exact, dropout-free nested-validation observation."""

    epoch: int
    total: float
    eligible: bool
    improved: bool
    incumbent_epoch: int | None
    incumbent_loss: float | None
    consecutive_without_improvement: int
    stop: bool
    model_state_sha256: str


@dataclass(frozen=True)
class RNGStateReceipt:
    python_sha256: str
    numpy_sha256: str
    torch_cpu_sha256: str
    torch_cuda_sha256: str | None


@dataclass(frozen=True)
class EpochStateReceipt:
    epoch: int
    before_model_state_sha256: str
    before_optimizer_state_sha256: str
    before_rng_state: RNGStateReceipt
    training_loss: EpochLossReceipt
    validation_loss: EarlyStopLossReceipt | None
    model_state_sha256: str
    optimizer_state_sha256: str
    rng_state: RNGStateReceipt


@dataclass(frozen=True)
class NamedTensorState:
    name: str
    shape: tuple[int, ...]
    dtype: str
    raw_bytes: bytes
    sha256: str


@dataclass(frozen=True)
class TensorSafeCheckpoint:
    epoch: int
    tensors: tuple[NamedTensorState, ...]
    state_sha256: str


@dataclass(frozen=True)
class RawValidationEvidence:
    epoch: int
    tensors: tuple[NamedTensorState, ...]
    evidence_sha256: str


@dataclass(frozen=True)
class V5OptimizerAttemptArtifact:
    completed_epochs: int
    best_epoch: int
    best_loss: float
    training_trace: tuple[EpochLossReceipt, ...]
    validation_trace: tuple[EarlyStopLossReceipt, ...]
    epoch_states: tuple[EpochStateReceipt, ...]
    raw_validation: tuple[RawValidationEvidence, ...]
    selected_validation_predictions: tuple[NamedTensorState, ...]
    best_checkpoint: TensorSafeCheckpoint
    dropout_seed_manifest: DropoutSeedManifest
    selection_state_sha256: str
    early_stop_trace_sha256: str
    scientific_authority: bool = False


@dataclass(frozen=True)
class OptimizerParameterGroupReceipt:
    index: int
    ordered_named_parameters: tuple[str, ...]


@dataclass(frozen=True)
class OptimizerReceipt:
    optimizer_class: str
    optimizer_module: str
    torch_version: str
    learning_rate: float
    betas: tuple[float, float]
    epsilon: float
    weight_decay: float
    amsgrad: bool
    foreach: bool
    fused: bool
    capturable: bool
    maximize: bool
    differentiable: bool
    gradient_clip_global_norm: float
    error_if_nonfinite: bool
    parameter_group_count: int
    parameter_groups: tuple[OptimizerParameterGroupReceipt, ...]
    canonical_field_json: str
    canonical_field_sha256: str
    scientific_authority: bool = False


@dataclass(frozen=True)
class V5ForwardPrediction:
    proteins: np.ndarray
    z_m: np.ndarray


@dataclass(frozen=True)
class V5ReversePrediction:
    metabolites: np.ndarray
    z_p: np.ndarray


@dataclass
class FittedMultiFMBidirV5Fold:
    """One fitted fold; deliberately non-authoritative and array-only."""

    config: MultiFMBidirV5Config
    dropout_context: V5DropoutSeedContext | None
    dropout_seed_manifest: DropoutSeedManifest
    control: V5Control
    metabolite_scaler: FeatureScaler
    protein_scaler: FeatureScaler
    scgpt_teacher: WhitenedPCATeacher | None
    effective_esm_tokens: np.ndarray | None
    parameter_count: ParameterCount
    control_parameters: ControlParameterReceipt
    control_transform: ControlTransformReceipt
    determinism: DeterminismReceipt
    optimizer_receipt: OptimizerReceipt
    loss_history: tuple[EpochLossReceipt, ...]
    architecture_contract: dict[str, Any]
    _model: Any = field(repr=False)
    _torch: Any = field(repr=False)

    @property
    def metabolite_count(self) -> int:
        return int(self.metabolite_scaler.mean.shape[0])

    @property
    def protein_count(self) -> int:
        return int(self.protein_scaler.mean.shape[0])

    def predict_forward(
        self,
        metabolites: np.ndarray,
        metabolite_mask: np.ndarray | None = None,
        *,
        batch_size: int | None = None,
    ) -> V5ForwardPrediction:
        """Predict proteins from measured metabolites only."""

        values, mask = _matrix_and_mask(
            metabolites,
            metabolite_mask,
            "forward metabolites",
            expected_features=self.metabolite_count,
        )
        if not bool(mask.all()):
            raise CCLEMultiFMV5Error(
                "forward metabolite inference requires the complete 225 panel"
            )
        scaled = self.metabolite_scaler.transform(values, mask)
        outputs: list[np.ndarray] = []
        states: list[np.ndarray] = []
        torch = self._torch
        device = torch.device(self.config.device)
        size = _prediction_batch_size(batch_size, self.config.batch_size)
        self._model.eval()
        with torch.inference_mode():
            for start in range(0, len(scaled), size):
                stop = min(start + size, len(scaled))
                source = torch.from_numpy(scaled[start:stop]).to(device)
                source_mask = torch.from_numpy(mask[start:stop]).to(device)
                prediction, state = self._model.forward_from_metabolites(
                    source, source_mask
                )
                outputs.append(prediction.cpu().numpy())
                states.append(state.cpu().numpy())
        scaled_output = np.concatenate(outputs, axis=0)
        if not np.isfinite(scaled_output).all():
            raise CCLEMultiFMV5Error("non-finite V5 forward prediction")
        return V5ForwardPrediction(
            proteins=_readonly(self.protein_scaler.inverse(scaled_output)),
            z_m=_readonly(np.concatenate(states, axis=0), dtype=np.float32),
        )

    def predict_reverse(
        self,
        proteins: np.ndarray,
        protein_mask: np.ndarray,
        *,
        batch_size: int | None = None,
    ) -> V5ReversePrediction:
        """Predict metabolites from observed protein tokens and exact mask."""

        values, mask = _matrix_and_mask(
            proteins,
            protein_mask,
            "reverse proteins",
            expected_features=self.protein_count,
            require_mask=True,
        )
        scaled = self.protein_scaler.transform(values, mask)
        outputs: list[np.ndarray] = []
        states: list[np.ndarray] = []
        torch = self._torch
        device = torch.device(self.config.device)
        size = _prediction_batch_size(batch_size, self.config.batch_size)
        self._model.eval()
        with torch.inference_mode():
            for start in range(0, len(scaled), size):
                stop = min(start + size, len(scaled))
                source = torch.from_numpy(scaled[start:stop]).to(device)
                source_mask = torch.from_numpy(mask[start:stop]).to(device)
                prediction, state = self._model.forward_from_proteins(
                    source, source_mask
                )
                outputs.append(prediction.cpu().numpy())
                states.append(state.cpu().numpy())
        scaled_output = np.concatenate(outputs, axis=0)
        if not np.isfinite(scaled_output).all():
            raise CCLEMultiFMV5Error("non-finite V5 reverse prediction")
        return V5ReversePrediction(
            metabolites=_readonly(self.metabolite_scaler.inverse(scaled_output)),
            z_p=_readonly(np.concatenate(states, axis=0), dtype=np.float32),
        )


def _prediction_batch_size(requested: int | None, default: int) -> int:
    value = default if requested is None else requested
    return _exact_int(value, "prediction batch_size", minimum=1)


def _readonly(value: np.ndarray, *, dtype: Any | None = None) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


def _array_receipt(value: np.ndarray) -> ArrayReceipt:
    array = np.ascontiguousarray(value)
    if array.dtype.itemsize > 1 and array.dtype.byteorder not in ("<", "|"):
        array = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<")))
    digest = hashlib.sha256(array.tobytes(order="C")).hexdigest()
    return ArrayReceipt(
        tuple(int(item) for item in array.shape), str(array.dtype), digest
    )


def _finite_matrix(value: np.ndarray, label: str) -> np.ndarray:
    matrix = np.asarray(value)
    if matrix.dtype.kind not in "fiu" or matrix.dtype.kind == "b":
        raise CCLEMultiFMV5Error(f"{label} must have a real numeric dtype")
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise CCLEMultiFMV5Error(f"{label} must be a non-empty two-dimensional array")
    if not np.isfinite(matrix).all():
        raise CCLEMultiFMV5Error(f"{label} contains non-finite values")
    return np.ascontiguousarray(matrix)


def _matrix_and_mask(
    value: np.ndarray,
    mask: np.ndarray | None,
    label: str,
    *,
    expected_features: int | None = None,
    require_mask: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.dtype.kind == "b":
        raise CCLEMultiFMV5Error(f"{label} must have a real numeric dtype")
    matrix = np.asarray(raw, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise CCLEMultiFMV5Error(f"{label} must be a non-empty two-dimensional array")
    if expected_features is not None and matrix.shape[1] != expected_features:
        raise CCLEMultiFMV5Error(
            f"{label} has {matrix.shape[1]} features; expected {expected_features}"
        )
    if mask is None:
        if require_mask:
            raise CCLEMultiFMV5Error(f"{label} requires an explicit mask")
        observed = np.ones(matrix.shape, dtype=bool)
    else:
        raw_mask = np.asarray(mask)
        if raw_mask.dtype != np.bool_ or raw_mask.shape != matrix.shape:
            raise CCLEMultiFMV5Error(
                f"{label} mask must be a boolean array with the same shape"
            )
        observed = np.ascontiguousarray(raw_mask)
    if np.any(observed.sum(axis=1) == 0):
        raise CCLEMultiFMV5Error(f"{label} has a row with no observed features")
    if not np.isfinite(matrix[observed]).all():
        raise CCLEMultiFMV5Error(f"{label} has non-finite observed values")
    return np.ascontiguousarray(matrix), observed


def _fit_scaler(
    matrix: np.ndarray,
    mask: np.ndarray,
    *,
    epsilon: float,
    label: str,
) -> FeatureScaler:
    counts = mask.sum(axis=0)
    if np.any(counts == 0):
        raise CCLEMultiFMV5Error(f"{label} has a feature with no fit observations")
    safe = np.zeros(matrix.shape, dtype=np.float64)
    safe[mask] = matrix[mask]
    mean = safe.sum(axis=0) / counts
    centered = np.zeros(matrix.shape, dtype=np.float64)
    centered[mask] = safe[mask] - np.broadcast_to(mean, matrix.shape)[mask]
    variance = np.square(centered).sum(axis=0) / counts
    scale = np.sqrt(variance)
    scale[scale < epsilon] = 1.0
    if not (np.isfinite(mean).all() and np.isfinite(scale).all()):
        raise CCLEMultiFMV5Error(f"{label} scaler is non-finite")
    return FeatureScaler(_readonly(mean), _readonly(scale))


def _canonicalize_svd_signs(
    u: np.ndarray, vh: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    canonical_u = np.ascontiguousarray(u, dtype=np.float64).copy()
    canonical_vh = np.ascontiguousarray(vh, dtype=np.float64).copy()
    anchor_indices = np.empty(canonical_vh.shape[0], dtype=np.int64)
    for component in range(canonical_vh.shape[0]):
        absolute = np.abs(canonical_vh[component])
        maximum = float(absolute.max())
        anchors = np.flatnonzero(absolute == maximum)
        if maximum == 0.0 or len(anchors) == 0:
            raise CCLEMultiFMV5Error("unstable SVD sign canonicalization")
        anchor = int(anchors[0])
        anchor_indices[component] = anchor
        if canonical_vh[component, anchor] < 0.0:
            canonical_vh[component] *= -1.0
            canonical_u[:, component] *= -1.0
        if canonical_vh[component, anchor] <= 0.0:
            raise CCLEMultiFMV5Error("SVD canonical sign invariant failed")
    return canonical_u, canonical_vh, anchor_indices


def _scipy_svd(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from scipy import linalg
    except ImportError as exc:  # pragma: no cover - runtime dependency failure
        raise CCLEMultiFMV5Error("SciPy is required for exact V5 gesvd") from exc
    try:
        u, singular, vh = linalg.svd(
            values,
            full_matrices=False,
            check_finite=True,
            lapack_driver="gesvd",
            overwrite_a=False,
        )
    except Exception as exc:  # pragma: no cover - LAPACK error is runtime-specific
        raise CCLEMultiFMV5Error("SciPy gesvd failed without fallback") from exc
    if not (
        np.isfinite(u).all() and np.isfinite(singular).all() and np.isfinite(vh).all()
    ):
        raise CCLEMultiFMV5Error("SciPy gesvd returned non-finite values")
    u, vh, anchors = _canonicalize_svd_signs(u, vh)
    return u, np.ascontiguousarray(singular, dtype=np.float64), vh, anchors


def _fit_scgpt_teacher(
    matrix: np.ndarray,
    dimension: int,
    whitening_epsilon: float,
) -> WhitenedPCATeacher:
    values = _finite_matrix(matrix, "fit-fold scGPT teacher")
    if values.shape[0] <= dimension or min(values.shape) < dimension:
        raise CCLEMultiFMV5Error(
            "fit-fold scGPT dimensions must support the 32-coordinate teacher"
        )
    mean = values.mean(axis=0, dtype=np.float64)
    centered = values - mean
    _u, singular_values, vh, sign_indices = _scipy_svd(centered)
    retained_singular = singular_values[:dimension]
    if len(retained_singular) != dimension or np.any(retained_singular <= 0.0):
        raise CCLEMultiFMV5Error(
            "fit-fold scGPT matrix has fewer than 32 positive components"
        )
    components = np.ascontiguousarray(vh[:dimension].T)
    eigenvalues = np.square(retained_singular) / (values.shape[0] - 1)
    whitening_scale = np.sqrt(eigenvalues + whitening_epsilon)
    transformed = (centered @ components) / whitening_scale
    if not np.isfinite(transformed).all():
        raise CCLEMultiFMV5Error("fit-fold scGPT whitening is non-finite")
    placeholder = PCATeacherReceipt(
        input=_array_receipt(values),
        mean=_array_receipt(mean),
        singular_values=_array_receipt(retained_singular),
        components=_array_receipt(components),
        canonical_sign_indices=_array_receipt(sign_indices[:dimension]),
        eigenvalues=_array_receipt(eigenvalues),
        whitening_scale=_array_receipt(whitening_scale),
        transformed_fit=_array_receipt(transformed),
    )
    return WhitenedPCATeacher(
        mean=_readonly(mean),
        components=_readonly(components),
        singular_values=_readonly(retained_singular),
        eigenvalues=_readonly(eigenvalues),
        whitening_scale=_readonly(whitening_scale),
        receipt=placeholder,
    )


def _sattolo_permutation(size: int, seed: int) -> np.ndarray:
    _exact_int(size, "permutation size", minimum=2)
    _exact_int(seed, "permutation seed")
    permutation = np.arange(size, dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(seed))
    for index in range(size - 1, 0, -1):
        other = int(rng.integers(0, index))
        permutation[index], permutation[other] = (
            permutation[other],
            permutation[index],
        )
    if np.any(permutation == np.arange(size)) or len(np.unique(permutation)) != size:
        raise CCLEMultiFMV5Error("Sattolo intervention is not a derangement")
    return permutation


def _route_shuffle_indices(row_count: int, seed: int) -> RouteShuffleIndexReceipt:
    permutation_array = _sattolo_permutation(row_count, seed)
    identity = tuple(range(row_count))
    permutation = tuple(int(item) for item in permutation_array)
    return RouteShuffleIndexReceipt(
        permutation=permutation,
        self_metabolite_source=identity,
        self_metabolite_target=identity,
        self_protein_source=identity,
        self_protein_target=identity,
        cross_forward_metabolite_source=identity,
        cross_forward_protein_target=permutation,
        cross_reverse_protein_source=permutation,
        cross_reverse_metabolite_target=identity,
        anchor_metabolite_source=identity,
        anchor_protein_source=permutation,
        anchor_teacher_target=identity,
        state_metabolite_source=identity,
        state_protein_source=permutation,
    )


def _canonical_qr_basis(values: np.ndarray) -> np.ndarray:
    try:
        from scipy import linalg
    except ImportError as exc:  # pragma: no cover
        raise CCLEMultiFMV5Error("SciPy is required for exact V5 QR") from exc
    try:
        q, r = linalg.qr(
            values,
            mode="economic",
            pivoting=False,
            check_finite=True,
            overwrite_a=False,
        )
    except Exception as exc:  # pragma: no cover
        raise CCLEMultiFMV5Error("SciPy QR failed without fallback") from exc
    diagonal = np.diag(r)
    if len(diagonal) != values.shape[1] or np.any(diagonal == 0.0):
        raise CCLEMultiFMV5Error("QR control is rank-deficient")
    signs = np.where(diagonal > 0.0, 1.0, -1.0)
    q = q * signs
    if np.any(np.diag(r) * signs <= 0.0):
        raise CCLEMultiFMV5Error("QR positive-diagonal invariant failed")
    return np.ascontiguousarray(q, dtype=np.float64)


def _covariance_random_teacher(reference: np.ndarray, seed: int) -> np.ndarray:
    values = _finite_matrix(reference, "scGPT covariance control reference")
    rows, dimension = values.shape
    if rows <= dimension:
        raise CCLEMultiFMV5Error(
            "scGPT covariance-random control requires rows > teacher dimension"
        )
    mean = values.mean(axis=0, dtype=np.float64)
    centered = values - mean
    _u, singular, vh, _anchors = _scipy_svd(centered)
    if len(singular) < dimension or np.any(singular[:dimension] <= 0.0):
        raise CCLEMultiFMV5Error("scGPT covariance control reference lacks rank 32")
    rng = np.random.Generator(np.random.PCG64(seed))
    gaussian = rng.standard_normal((rows, dimension), dtype=np.float64)
    gaussian -= gaussian.mean(axis=0)
    q = _canonical_qr_basis(gaussian)
    state = mean + (q * singular[:dimension]) @ vh[:dimension]
    tolerance = 1e-10 + 1e-10 * np.abs(mean)
    if not np.all(np.abs(state.mean(axis=0) - mean) <= tolerance):
        raise CCLEMultiFMV5Error("random teacher mean invariant failed")
    reference_covariance = centered.T @ centered / (rows - 1)
    state_centered = state - state.mean(axis=0)
    state_covariance = state_centered.T @ state_centered / (rows - 1)
    covariance_tolerance = 1e-10 + 1e-10 * np.abs(reference_covariance)
    if not np.all(
        np.abs(state_covariance - reference_covariance) <= covariance_tolerance
    ):
        raise CCLEMultiFMV5Error("random teacher covariance invariant failed")
    if np.array_equal(state, values):
        raise CCLEMultiFMV5Error("random teacher did not alter row information")
    return np.ascontiguousarray(state)


def _covariance_random_esm(esm: np.ndarray, seed: int) -> np.ndarray:
    values = _finite_matrix(esm, "ESM2 covariance control input")
    rows, columns = values.shape
    if rows < 2:
        raise CCLEMultiFMV5Error("at least two ESM2 rows are required")
    mean = values.mean(axis=0, dtype=np.float64)
    centered = values - mean
    _u, singular, vh, _anchors = _scipy_svd(centered)
    rank_width = min(rows - 1, columns)
    if len(singular) < rank_width or vh.shape[0] < rank_width:
        raise CCLEMultiFMV5Error("ESM2 SVD lacks the required row-space width")
    rng = np.random.Generator(np.random.PCG64(seed))
    gaussian = rng.standard_normal((rows, rank_width), dtype=np.float64)
    gaussian -= gaussian.mean(axis=0)
    q = _canonical_qr_basis(gaussian)
    randomized = mean + (q * singular[:rank_width]) @ vh[:rank_width]
    if np.array_equal(randomized, values):
        raise CCLEMultiFMV5Error("ESM2 covariance control preserved row assignments")
    if not np.allclose(randomized.mean(axis=0), mean, atol=1e-10, rtol=1e-10):
        raise CCLEMultiFMV5Error("ESM2 covariance control mean mismatch")
    if not np.allclose(
        np.cov(randomized, rowvar=False),
        np.cov(values, rowvar=False),
        atol=1e-10,
        rtol=1e-10,
    ):
        raise CCLEMultiFMV5Error("ESM2 covariance control covariance mismatch")
    return np.ascontiguousarray(randomized)


def _prepare_esm_control(
    esm_embeddings: np.ndarray | None,
    *,
    control: V5Control,
    protein_count: int,
    seed: int | None,
) -> tuple[np.ndarray | None, tuple[int, ...] | None, str, bool]:
    if control == "no_esm_identity_only":
        if esm_embeddings is not None:
            raise CCLEMultiFMV5Error(
                "no_esm_identity_only requires esm_embeddings is exactly None"
            )
        return None, None, "learned_identity_capacity_control", False
    if esm_embeddings is None:
        raise CCLEMultiFMV5Error(f"control {control!r} requires ESM2 embeddings")
    esm = _finite_matrix(esm_embeddings, "ESM2 embeddings")
    if esm.shape[0] != protein_count:
        raise CCLEMultiFMV5Error("ESM2 row count must equal protein feature count")
    if control == "esm2_deranged":
        seed = _exact_int(seed, "intervention_seed")
        permutation = _sattolo_permutation(protein_count, seed)
        return (
            esm[permutation].copy(),
            tuple(int(item) for item in permutation),
            "sattolo_identity_derangement",
            True,
        )
    if control == "esm2_covariance_random":
        seed = _exact_int(seed, "intervention_seed")
        randomized = _covariance_random_esm(esm, seed)
        return randomized, None, "gesvd_qr_covariance_random", True
    return esm.copy(), None, "identity_bound_esm2", True


def _prepare_teacher_control(
    scgpt: np.ndarray | None,
    *,
    control: V5Control,
    rows: int,
    config: MultiFMBidirV5Config,
    intervention_seed: int | None,
) -> tuple[
    WhitenedPCATeacher | None, np.ndarray | None, tuple[int, ...] | None, str, bool
]:
    if control == "no_scgpt":
        if scgpt is not None:
            raise CCLEMultiFMV5Error("no_scgpt requires scgpt is exactly None")
        return None, None, None, "inert_same_shape_teacher_head", False
    if scgpt is None:
        raise CCLEMultiFMV5Error(f"control {control!r} requires fit-fold scGPT")
    values = _finite_matrix(scgpt, "fit-fold scGPT")
    if values.shape[0] != rows:
        raise CCLEMultiFMV5Error("scGPT and endpoint fit row counts differ")
    if values.shape[1] != V5_SCGPT_DIMENSION:
        raise CCLEMultiFMV5Error("scGPT feature count must be exactly 512")
    if control == "scgpt_covariance_random":
        seed = _exact_int(intervention_seed, "intervention_seed")
        teacher = _fit_scgpt_teacher(
            values, config.teacher_dimension, config.teacher_whitening_epsilon
        )
        real_state = teacher.transform(values)
        state = _covariance_random_teacher(real_state, seed)
        return (
            teacher,
            state,
            None,
            "whitened_teacher_gesvd_qr_covariance_random",
            True,
        )
    teacher = _fit_scgpt_teacher(
        values, config.teacher_dimension, config.teacher_whitening_epsilon
    )
    state = teacher.transform(values)
    if control == "scgpt_row_deranged":
        seed = _exact_int(intervention_seed, "intervention_seed")
        permutation = _sattolo_permutation(rows, seed)
        state = np.ascontiguousarray(state[permutation])
        return (
            teacher,
            state,
            tuple(int(item) for item in permutation),
            "fold_local_whitened_teacher_sattolo_derangement",
            True,
        )
    return teacher, state, None, "fold_local_whitened_scgpt", True


def _derive_seed(root: int, *parts: object) -> int:
    material = "|".join((str(root), *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63)


def _verify_fresh_cuda_environment(torch_already_imported: bool) -> None:
    if torch_already_imported:
        raise CCLEMultiFMV5Error(
            "CUDA V5 must start in a fresh process before torch is imported"
        )
    mismatches = [
        f"{name}={os.environ.get(name)!r} expected {expected!r}"
        for name, expected in _CUDA_ENVIRONMENT.items()
        if os.environ.get(name) != expected
    ]
    if mismatches:
        raise CCLEMultiFMV5Error(
            "CUDA fresh-process environment mismatch: " + "; ".join(mismatches)
        )


def _require_and_configure_torch(
    config: MultiFMBidirV5Config,
) -> tuple[Any, DeterminismReceipt]:
    torch_already_imported = "torch" in sys.modules
    is_cuda = config.device.startswith("cuda:")
    if is_cuda:
        _verify_fresh_cuda_environment(torch_already_imported)
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise CCLEMultiFMV5Error("PyTorch is required to fit V5") from exc
    if torch.get_default_dtype() is not torch.float32:
        raise CCLEMultiFMV5Error("torch global default dtype must be exactly float32")
    if is_cuda and not torch.cuda.is_available():
        raise CCLEMultiFMV5Error("a CUDA device was requested but CUDA is unavailable")
    device = torch.device(config.device)
    if device.type == "cuda" and (
        device.index is None or device.index >= torch.cuda.device_count()
    ):
        raise CCLEMultiFMV5Error("requested CUDA device index does not exist")

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as exc:
            raise CCLEMultiFMV5Error(
                "torch interop threads were not fixed before parallel work"
            ) from exc
    random.seed(config.seed)
    np.random.seed(config.seed % (2**32))
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision("highest")

    cudnn_deterministic: bool | None = None
    cudnn_benchmark: bool | None = None
    cuda_tf32: bool | None = None
    cudnn_tf32: bool | None = None
    flash_enabled: bool | None = None
    mem_enabled: bool | None = None
    math_enabled: bool | None = None
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        cudnn_deterministic = bool(torch.backends.cudnn.deterministic)
        cudnn_benchmark = bool(torch.backends.cudnn.benchmark)
        cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        cuda_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            flash_enabled = bool(torch.backends.cuda.flash_sdp_enabled())
            mem_enabled = bool(torch.backends.cuda.mem_efficient_sdp_enabled())
            math_enabled = bool(torch.backends.cuda.math_sdp_enabled())
    if flash_enabled is True or mem_enabled is True or math_enabled is False:
        raise CCLEMultiFMV5Error("math-SDP-only runtime invariant failed")
    receipt = DeterminismReceipt(
        seed=config.seed,
        device=str(device),
        torch_version=str(torch.__version__),
        cuda_available=bool(torch.cuda.is_available()),
        fresh_cuda_process_verified=is_cuda,
        deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
        warn_only=bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        cudnn_deterministic=cudnn_deterministic,
        cudnn_benchmark=cudnn_benchmark,
        cuda_matmul_tf32=cuda_tf32,
        cudnn_tf32=cudnn_tf32,
        flash_sdp_enabled=flash_enabled,
        mem_efficient_sdp_enabled=mem_enabled,
        math_sdp_enabled=math_enabled,
        default_dtype=str(torch.get_default_dtype()),
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        cpu_threads=int(torch.get_num_threads()),
        interop_threads=int(torch.get_num_interop_threads()),
        required_environment=tuple(sorted(_CUDA_ENVIRONMENT.items())),
    )
    return torch, receipt


def _build_model(
    torch: Any,
    *,
    metabolite_count: int,
    protein_count: int,
    esm_feature_count: int,
    effective_esm: np.ndarray | None,
    config: MultiFMBidirV5Config,
    control: V5Control,
) -> Any:
    """Build the byte-exact pre-norm Set Transformer graph on CPU."""

    nn = torch.nn
    functional = torch.nn.functional
    d = config.model_dimension
    eps = config.layer_norm_epsilon

    class PreNormMAB(nn.Module):
        """H=Q+MHA(LNq(Q),LNk(K),LNv(V)); O=H+FFN(LNh(H))."""

        def __init__(self) -> None:
            super().__init__()
            self.ln_q = nn.LayerNorm(d, eps=eps, elementwise_affine=True, bias=True)
            self.ln_k = nn.LayerNorm(d, eps=eps, elementwise_affine=True, bias=True)
            self.ln_v = nn.LayerNorm(d, eps=eps, elementwise_affine=True, bias=True)
            self.attention = nn.MultiheadAttention(
                embed_dim=d,
                num_heads=config.attention_heads,
                dropout=0.0,
                bias=True,
                add_bias_kv=False,
                add_zero_attn=False,
                kdim=d,
                vdim=d,
                batch_first=True,
            )
            self.ln_h = nn.LayerNorm(d, eps=eps, elementwise_affine=True, bias=True)
            self.ffn = nn.Sequential(
                nn.Linear(d, config.ffn_multiplier * d, bias=True),
                nn.GELU(approximate="none"),
                nn.Linear(config.ffn_multiplier * d, d, bias=True),
            )

        def forward(
            self,
            query: Any,
            key: Any,
            value: Any,
            *,
            key_padding_mask: Any | None = None,
            query_padding_mask: Any | None = None,
        ) -> Any:
            attended, weights = self.attention(
                self.ln_q(query),
                self.ln_k(key),
                self.ln_v(value),
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            if weights is not None:
                raise CCLEMultiFMV5Error("MAB unexpectedly returned attention weights")
            hidden = query + attended
            output = hidden + self.ffn(self.ln_h(hidden))
            if query_padding_mask is not None:
                output = output.masked_fill(query_padding_mask.unsqueeze(-1), 0.0)
            return output

    class ISABBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inducing_points = nn.Parameter(
                torch.empty(config.inducing_points, d, dtype=torch.float32)
            )
            self.induce = PreNormMAB()
            self.return_to_source = PreNormMAB()

        def forward(self, source: Any, observed: Any) -> Any:
            inducing = self.inducing_points.unsqueeze(0).expand(source.shape[0], -1, -1)
            padding = ~observed
            summary = self.induce(
                inducing,
                source,
                source,
                key_padding_mask=padding,
            )
            return self.return_to_source(
                source,
                summary,
                summary,
                query_padding_mask=padding,
            )

    class SetEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(
                ISABBlock() for _ in range(config.isab_blocks_per_modality)
            )

        def forward(self, source: Any, observed: Any) -> Any:
            hidden = source
            for block in self.blocks:
                hidden = block(hidden, observed)
            return hidden

    class SharedPMA(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.slot_queries = nn.Parameter(
                torch.empty(config.latent_slots, d, dtype=torch.float32)
            )
            self.mab = PreNormMAB()

        def forward(self, source: Any, observed: Any) -> Any:
            slots = self.slot_queries.unsqueeze(0).expand(source.shape[0], -1, -1)
            return self.mab(
                slots,
                source,
                source,
                key_padding_mask=~observed,
            )

    class PreNormLatentBlock(nn.Module):
        """Two-LN shared self-attention block required by the protocol."""

        def __init__(self) -> None:
            super().__init__()
            self.ln_attention = nn.LayerNorm(
                d, eps=eps, elementwise_affine=True, bias=True
            )
            self.attention = nn.MultiheadAttention(
                embed_dim=d,
                num_heads=config.attention_heads,
                dropout=0.0,
                bias=True,
                add_bias_kv=False,
                add_zero_attn=False,
                kdim=d,
                vdim=d,
                batch_first=True,
            )
            self.ln_ffn = nn.LayerNorm(d, eps=eps, elementwise_affine=True, bias=True)
            self.ffn = nn.Sequential(
                nn.Linear(d, config.ffn_multiplier * d, bias=True),
                nn.GELU(approximate="none"),
                nn.Linear(config.ffn_multiplier * d, d, bias=True),
            )

        def forward(self, state: Any) -> Any:
            normalized = self.ln_attention(state)
            attended, weights = self.attention(
                normalized, normalized, normalized, need_weights=False
            )
            if weights is not None:
                raise CCLEMultiFMV5Error(
                    "latent core unexpectedly returned attention weights"
                )
            residual = state + attended
            return residual + self.ffn(self.ln_ffn(residual))

    class LatentCore(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(
                PreNormLatentBlock() for _ in range(config.latent_core_blocks)
            )

        def forward(self, state: Any) -> Any:
            for block in self.blocks:
                state = block(state)
            return state

    class QueryDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mab = PreNormMAB()
            self.scalar_head = nn.Linear(d, 1, bias=True)

        def forward(self, query: Any, state: Any) -> Any:
            batch_query = query.unsqueeze(0).expand(state.shape[0], -1, -1)
            decoded = self.mab(batch_query, state, state)
            return self.scalar_head(decoded).squeeze(-1)

    class V5Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.metabolite_identity = nn.Parameter(
                torch.empty(metabolite_count, d, dtype=torch.float32)
            )
            self.metabolite_abundance_weight = nn.Parameter(
                torch.empty(d, dtype=torch.float32)
            )
            self.metabolite_abundance_bias = nn.Parameter(
                torch.empty(d, dtype=torch.float32)
            )
            self.protein_abundance_weight = nn.Parameter(
                torch.empty(d, dtype=torch.float32)
            )
            self.protein_abundance_bias = nn.Parameter(
                torch.empty(d, dtype=torch.float32)
            )
            self.metabolite_query_offset = nn.Parameter(
                torch.empty(d, dtype=torch.float32)
            )
            self.protein_query_offset = nn.Parameter(
                torch.empty(d, dtype=torch.float32)
            )

            self.metabolite_token_norm = nn.LayerNorm(
                d, eps=eps, elementwise_affine=True, bias=True
            )
            self.protein_token_norm = nn.LayerNorm(
                d, eps=eps, elementwise_affine=True, bias=True
            )
            self.metabolite_query_norm = nn.LayerNorm(
                d, eps=eps, elementwise_affine=True, bias=True
            )
            self.protein_query_norm = nn.LayerNorm(
                d, eps=eps, elementwise_affine=True, bias=True
            )

            self.no_esm_identity_only = control == "no_esm_identity_only"
            if self.no_esm_identity_only:
                identity_width = math.ceil(
                    (esm_feature_count * d) / (protein_count + d)
                )
                self.protein_identity_width = identity_width
                self.protein_identity_table = nn.Parameter(
                    torch.empty(protein_count, identity_width, dtype=torch.float32)
                )
                self.protein_identity_projection = nn.Linear(
                    identity_width, d, bias=True
                )
                self.register_buffer("effective_esm", None)
            else:
                self.protein_identity_width = None
                self.register_buffer(
                    "effective_esm",
                    torch.from_numpy(
                        np.ascontiguousarray(effective_esm, dtype=np.float32)
                    ),
                    persistent=True,
                )
                self.protein_identity_projection = nn.Linear(
                    esm_feature_count, d, bias=True
                )

            self.metabolite_encoder = SetEncoder()
            self.protein_encoder = SetEncoder()
            self.shared_pma = SharedPMA()
            self.shared_latent_core = LatentCore()
            self.metabolite_decoder = QueryDecoder()
            self.protein_decoder = QueryDecoder()
            self.teacher_head = nn.Linear(d, config.teacher_dimension, bias=True)

        def projected_protein_identities(self) -> Any:
            if self.no_esm_identity_only:
                raw = self.protein_identity_table
            else:
                raw = functional.layer_norm(
                    self.effective_esm,
                    (esm_feature_count,),
                    weight=None,
                    bias=None,
                    eps=eps,
                )
            return functional.gelu(
                self.protein_identity_projection(raw), approximate="none"
            )

        def tokenize_metabolites(
            self,
            abundance: Any,
            observed: Any,
            identity_indices: Any | None = None,
        ) -> Any:
            identities = self.metabolite_identity
            if identity_indices is not None:
                identities = identities[identity_indices]
            tokens = (
                identities.unsqueeze(0)
                + abundance.unsqueeze(-1) * self.metabolite_abundance_weight
                + self.metabolite_abundance_bias
            )
            normalized = self.metabolite_token_norm(tokens)
            return normalized.masked_fill(~observed.unsqueeze(-1), 0.0)

        def tokenize_proteins(
            self,
            abundance: Any,
            observed: Any,
            identity_indices: Any | None = None,
        ) -> Any:
            identities = self.projected_protein_identities()
            if identity_indices is not None:
                identities = identities[identity_indices]
            tokens = (
                identities.unsqueeze(0)
                + abundance.unsqueeze(-1) * self.protein_abundance_weight
                + self.protein_abundance_bias
            )
            normalized = self.protein_token_norm(tokens)
            return normalized.masked_fill(~observed.unsqueeze(-1), 0.0)

        def metabolite_queries(self) -> Any:
            return self.metabolite_query_norm(
                self.metabolite_identity + self.metabolite_query_offset
            )

        def protein_queries(self) -> Any:
            return self.protein_query_norm(
                self.projected_protein_identities() + self.protein_query_offset
            )

        def encode_metabolites(
            self,
            abundance: Any,
            observed: Any,
            identity_indices: Any | None = None,
        ) -> Any:
            if identity_indices is not None:
                canonical = torch.argsort(identity_indices)
                abundance = abundance[:, canonical]
                observed = observed[:, canonical]
                identity_indices = identity_indices[canonical]
            tokens = self.tokenize_metabolites(
                abundance, observed, identity_indices=identity_indices
            )
            hidden = self.metabolite_encoder(tokens, observed)
            return self.shared_latent_core(self.shared_pma(hidden, observed))

        def encode_proteins(
            self,
            abundance: Any,
            observed: Any,
            identity_indices: Any | None = None,
        ) -> Any:
            if identity_indices is not None:
                canonical = torch.argsort(identity_indices)
                abundance = abundance[:, canonical]
                observed = observed[:, canonical]
                identity_indices = identity_indices[canonical]
            tokens = self.tokenize_proteins(
                abundance, observed, identity_indices=identity_indices
            )
            hidden = self.protein_encoder(tokens, observed)
            return self.shared_latent_core(self.shared_pma(hidden, observed))

        def decode_metabolites(self, state: Any) -> Any:
            return self.metabolite_decoder(self.metabolite_queries(), state)

        def decode_proteins(self, state: Any) -> Any:
            return self.protein_decoder(self.protein_queries(), state)

        def teacher_state(self, state: Any) -> Any:
            return self.teacher_head(state.mean(dim=1))

        def forward_from_metabolites(self, abundance: Any, observed: Any) -> Any:
            state = self.encode_metabolites(abundance, observed)
            return self.decode_proteins(state), state

        def forward_from_proteins(self, abundance: Any, observed: Any) -> Any:
            state = self.encode_proteins(abundance, observed)
            return self.decode_metabolites(state), state

    return V5Network()


_NORMAL_PARAMETER_SUFFIXES = (
    "metabolite_identity",
    "metabolite_abundance_weight",
    "metabolite_abundance_bias",
    "protein_abundance_weight",
    "protein_abundance_bias",
    "metabolite_query_offset",
    "protein_query_offset",
    "protein_identity_table",
    "inducing_points",
    "slot_queries",
)


def _initialize_model(torch: Any, model: Any, root_seed: int) -> dict[str, int]:
    """Initialize every tensor from its own qualified-name seed."""

    modules = dict(model.named_modules())
    initialization_seeds: dict[str, int] = {}
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if parameter.device.type != "cpu" or parameter.dtype is not torch.float32:
            raise CCLEMultiFMV5Error(
                f"parameter {name!r} was not initialized as CPU float32"
            )
        seed = _derive_seed(root_seed, "initialization", name)
        initialization_seeds[name] = seed
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        module_name, _, local_name = name.rpartition(".")
        owner = modules[module_name]
        with torch.no_grad():
            if isinstance(owner, torch.nn.LayerNorm):
                if local_name == "weight":
                    torch.nn.init.ones_(parameter)
                elif local_name == "bias":
                    torch.nn.init.zeros_(parameter)
                else:  # pragma: no cover - current PyTorch registry
                    raise CCLEMultiFMV5Error(f"unknown LayerNorm tensor {name!r}")
            elif isinstance(owner, torch.nn.Linear):
                if local_name == "weight":
                    torch.nn.init.xavier_uniform_(
                        parameter, gain=1.0, generator=generator
                    )
                elif local_name == "bias":
                    torch.nn.init.zeros_(parameter)
                else:  # pragma: no cover
                    raise CCLEMultiFMV5Error(f"unknown Linear tensor {name!r}")
            elif isinstance(owner, torch.nn.MultiheadAttention):
                if local_name in {
                    "in_proj_weight",
                    "q_proj_weight",
                    "k_proj_weight",
                    "v_proj_weight",
                }:
                    torch.nn.init.xavier_uniform_(
                        parameter, gain=1.0, generator=generator
                    )
                elif local_name in {"in_proj_bias", "bias_k", "bias_v"}:
                    torch.nn.init.zeros_(parameter)
                else:  # out_proj is owned by its Linear child
                    raise CCLEMultiFMV5Error(f"unknown MHA tensor {name!r}")
            elif any(name.endswith(suffix) for suffix in _NORMAL_PARAMETER_SUFFIXES):
                torch.nn.init.normal_(
                    parameter, mean=0.0, std=0.02, generator=generator
                )
            else:
                raise CCLEMultiFMV5Error(
                    f"parameter {name!r} has no byte-exact initializer"
                )
        if not torch.isfinite(parameter).all():
            raise CCLEMultiFMV5Error(f"parameter {name!r} initialized non-finite")
    return initialization_seeds


def _tensor_collection_sha256(named_tensors: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        digest.update(name.encode("utf-8"))
        digest.update(b"|")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"|")
        digest.update(",".join(str(item) for item in array.shape).encode("ascii"))
        digest.update(b"|")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _tensor_sha256(tensor: Any) -> str:
    array = np.ascontiguousarray(tensor.detach().cpu().numpy())
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _optimizer_state_sha256(model: Any, optimizer: Any) -> str:
    """Hash AdamW state without pickle or framework object identifiers."""

    digest = hashlib.sha256()
    parameter_names = {id(value): name for name, value in model.named_parameters()}
    for group_index, group in enumerate(optimizer.param_groups):
        digest.update(f"group|{group_index}|".encode("ascii"))
        fields = {
            key: value
            for key, value in group.items()
            if key != "params" and isinstance(value, (bool, int, float, str, tuple))
        }
        digest.update(
            json.dumps(
                fields,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        )
        for parameter in group["params"]:
            name = parameter_names[id(parameter)]
            digest.update(name.encode("utf-8"))
            state = optimizer.state.get(parameter, {})
            for key in sorted(state):
                digest.update(str(key).encode("utf-8"))
                value = state[key]
                if hasattr(value, "detach"):
                    array = np.ascontiguousarray(value.detach().cpu().numpy())
                    digest.update(str(array.dtype).encode("ascii"))
                    digest.update(",".join(map(str, array.shape)).encode("ascii"))
                    digest.update(array.tobytes(order="C"))
                else:
                    digest.update(
                        json.dumps(
                            value,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ).encode("utf-8")
                    )
    return digest.hexdigest()


def _rng_state_receipt(torch: Any) -> RNGStateReceipt:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    numpy_payload = {
        "bit_generator": numpy_state[0],
        "keys": [int(value) for value in numpy_state[1]],
        "position": int(numpy_state[2]),
        "has_gauss": int(numpy_state[3]),
        "cached_gaussian": float(numpy_state[4]),
    }
    cpu = np.ascontiguousarray(torch.get_rng_state().cpu().numpy())
    cuda_sha: str | None = None
    if torch.cuda.is_available():
        digest = hashlib.sha256()
        for state in torch.cuda.get_rng_state_all():
            digest.update(np.ascontiguousarray(state.cpu().numpy()).tobytes(order="C"))
        cuda_sha = digest.hexdigest()
    return RNGStateReceipt(
        python_sha256=_canonical_json_sha256(python_state),
        numpy_sha256=_canonical_json_sha256(numpy_payload),
        torch_cpu_sha256=hashlib.sha256(cpu.tobytes(order="C")).hexdigest(),
        torch_cuda_sha256=cuda_sha,
    )


def _tensor_safe_checkpoint(model: Any, epoch: int) -> TensorSafeCheckpoint:
    tensors: list[NamedTensorState] = []
    for name, tensor in sorted(model.state_dict().items()):
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        raw = array.tobytes(order="C")
        tensors.append(
            NamedTensorState(
                name=name,
                shape=tuple(int(value) for value in array.shape),
                dtype=str(array.dtype),
                raw_bytes=raw,
                sha256=hashlib.sha256(raw).hexdigest(),
            )
        )
    return TensorSafeCheckpoint(
        epoch=epoch,
        tensors=tuple(tensors),
        state_sha256=_tensor_collection_sha256(model.state_dict().items()),
    )


def _named_tensor_state(name: str, tensor: Any) -> NamedTensorState:
    array = np.ascontiguousarray(tensor.detach().cpu().numpy())
    raw = array.tobytes(order="C")
    return NamedTensorState(
        name=name,
        shape=tuple(int(value) for value in array.shape),
        dtype=str(array.dtype),
        raw_bytes=raw,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _parameter_count(
    model: Any, initialization_seeds: dict[str, int]
) -> ParameterCount:
    module_counts: list[tuple[str, int]] = []
    for name, module in model.named_modules():
        count = int(sum(value.numel() for value in module.parameters(recurse=False)))
        if count:
            module_counts.append((name if name else "<root>", count))
    by_module = tuple(module_counts)
    named: list[NamedParameterReceipt] = []
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        training_only = name.startswith("teacher_head.")
        named.append(
            NamedParameterReceipt(
                name=name,
                shape=tuple(int(item) for item in parameter.shape),
                numel=int(parameter.numel()),
                dtype=str(parameter.dtype),
                initialization_seed=initialization_seeds[name],
                initialization_sha256=_tensor_sha256(parameter),
                inference_path=not training_only,
                training_only=training_only,
            )
        )
    total = sum(item.numel for item in named)
    trainable = int(
        sum(value.numel() for value in model.parameters() if value.requires_grad)
    )
    inference = sum(item.numel for item in named if item.inference_path)
    training = sum(item.numel for item in named if item.training_only)
    if total != trainable or total != inference + training:
        raise CCLEMultiFMV5Error("internal parameter accounting mismatch")
    if sum(count for _name, count in by_module) != total:
        raise CCLEMultiFMV5Error("module-level parameter accounting mismatch")
    return ParameterCount(
        total=total,
        trainable=trainable,
        inference_path=inference,
        training_only=training,
        by_module=by_module,
        named=tuple(named),
    )


def _optimizer_receipt(
    torch: Any,
    optimizer: Any,
    model: Any,
    config: MultiFMBidirV5Config,
) -> OptimizerReceipt:
    """Read back and bind the real AdamW object; never trust constructor inputs."""

    if optimizer.__class__ is not torch.optim.AdamW:
        raise CCLEMultiFMV5Error("optimizer class is not exactly torch.optim.AdamW")
    defaults = optimizer.defaults
    expected_scalar_defaults = {
        "lr": config.learning_rate,
        "eps": config.adam_epsilon,
        "weight_decay": config.weight_decay,
    }
    for field_name, expected_value in expected_scalar_defaults.items():
        observed_value = defaults.get(field_name)
        if type(observed_value) is not float or observed_value != expected_value:
            raise CCLEMultiFMV5Error(
                f"AdamW default {field_name} differs from the protocol"
            )
    observed_betas = defaults.get("betas")
    expected_betas = (config.adam_beta1, config.adam_beta2)
    if (
        type(observed_betas) is not tuple
        or len(observed_betas) != 2
        or any(type(value) is not float for value in observed_betas)
        or observed_betas != expected_betas
    ):
        raise CCLEMultiFMV5Error("AdamW default betas differ from the protocol")
    expected_boolean_defaults = {
        "amsgrad": False,
        "foreach": False,
        "fused": False,
        "capturable": False,
        "maximize": False,
        "differentiable": False,
    }
    for field_name, expected_value in expected_boolean_defaults.items():
        observed_value = defaults.get(field_name)
        if type(observed_value) is not bool or observed_value is not expected_value:
            raise CCLEMultiFMV5Error(
                f"AdamW default {field_name} must remain explicitly false"
            )
    if len(optimizer.param_groups) != 1:
        raise CCLEMultiFMV5Error("AdamW must have exactly one parameter group")

    names_by_identity = {
        id(parameter): name for name, parameter in model.named_parameters()
    }
    model_names = tuple(name for name, _parameter in model.named_parameters())
    seen: set[int] = set()
    group_receipts: list[OptimizerParameterGroupReceipt] = []
    for group_index, group in enumerate(optimizer.param_groups):
        for field_name, expected_value in {
            **expected_scalar_defaults,
            "betas": expected_betas,
            **expected_boolean_defaults,
        }.items():
            if group.get(field_name) != expected_value:
                raise CCLEMultiFMV5Error(
                    f"AdamW group {group_index} field {field_name} differs"
                )
        group_names: list[str] = []
        for parameter in group["params"]:
            identity = id(parameter)
            if identity not in names_by_identity:
                raise CCLEMultiFMV5Error(
                    "AdamW contains a parameter outside the exact model registry"
                )
            if identity in seen:
                raise CCLEMultiFMV5Error("AdamW parameter membership is duplicated")
            seen.add(identity)
            group_names.append(names_by_identity[identity])
        group_receipts.append(
            OptimizerParameterGroupReceipt(
                index=group_index,
                ordered_named_parameters=tuple(group_names),
            )
        )
    if tuple(group_receipts[0].ordered_named_parameters) != model_names:
        raise CCLEMultiFMV5Error(
            "AdamW ordered named-parameter membership differs from the model"
        )

    optimizer_class = optimizer.__class__.__name__
    optimizer_module = optimizer.__class__.__module__
    torch_version = str(torch.__version__)
    field_object = {
        "amsgrad": defaults["amsgrad"],
        "betas": list(defaults["betas"]),
        "capturable": defaults["capturable"],
        "class": optimizer_class,
        "differentiable": defaults["differentiable"],
        "eps": defaults["eps"],
        "foreach": defaults["foreach"],
        "fused": defaults["fused"],
        "lr": defaults["lr"],
        "maximize": defaults["maximize"],
        "module": optimizer_module,
        "param_group_count": len(group_receipts),
        "param_groups": [
            {
                "index": group.index,
                "ordered_named_parameters": list(group.ordered_named_parameters),
            }
            for group in group_receipts
        ],
        "torch_version": torch_version,
        "weight_decay": defaults["weight_decay"],
    }
    try:
        canonical = json.dumps(
            field_object,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:  # pragma: no cover - guarded fields
        raise CCLEMultiFMV5Error(
            "optimizer field object is not finite canonical JSON"
        ) from exc
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return OptimizerReceipt(
        optimizer_class=optimizer_class,
        optimizer_module=optimizer_module,
        torch_version=torch_version,
        learning_rate=defaults["lr"],
        betas=defaults["betas"],
        epsilon=defaults["eps"],
        weight_decay=defaults["weight_decay"],
        amsgrad=defaults["amsgrad"],
        foreach=defaults["foreach"],
        fused=defaults["fused"],
        capturable=defaults["capturable"],
        maximize=defaults["maximize"],
        differentiable=defaults["differentiable"],
        gradient_clip_global_norm=config.gradient_clip_norm,
        error_if_nonfinite=True,
        parameter_group_count=len(group_receipts),
        parameter_groups=tuple(group_receipts),
        canonical_field_json=canonical,
        canonical_field_sha256=digest,
    )


def _masked_mse(torch: Any, prediction: Any, target: Any, mask: Any) -> Any:
    weights = mask.to(dtype=prediction.dtype)
    denominator = weights.sum()
    if float(denominator.detach().cpu()) <= 0.0:
        raise CCLEMultiFMV5Error("masked MSE has no observed target")
    return (torch.square(prediction - target) * weights).sum() / denominator


def _loss_only_layer_norm(torch: Any, state: Any, epsilon: float) -> Any:
    return torch.nn.functional.layer_norm(
        state,
        (state.shape[-1],),
        weight=None,
        bias=None,
        eps=epsilon,
    )


def _v5_loss_components(
    *,
    torch: Any,
    model: Any,
    config: MultiFMBidirV5Config,
    m_batch: Any,
    p_cross_batch: Any,
    p_self_batch: Any,
    m_source_mask_batch: Any,
    p_cross_source_mask_batch: Any,
    p_self_source_mask_batch: Any,
    m_target_mask_batch: Any,
    p_cross_target_mask_batch: Any,
    p_self_target_mask_batch: Any,
    teacher_target: Any | None,
    shared_protein_route: bool,
    raw_sink: dict[str, Any] | None = None,
    capture_raw_targets: bool = False,
) -> tuple[Any, Any, Any, Any, Any]:
    """Single normative loss implementation shared by train and validation."""

    z_m = model.encode_metabolites(m_batch, m_source_mask_batch)
    z_p_cross = model.encode_proteins(p_cross_batch, p_cross_source_mask_batch)
    z_p_self = (
        z_p_cross
        if shared_protein_route
        else model.encode_proteins(p_self_batch, p_self_source_mask_batch)
    )
    cross_p = model.decode_proteins(z_m)
    cross_m = model.decode_metabolites(z_p_cross)
    self_m = model.decode_metabolites(z_m)
    self_p = model.decode_proteins(z_p_self)
    cross_loss = 0.5 * (
        _masked_mse(torch, cross_p, p_cross_batch, p_cross_target_mask_batch)
        + _masked_mse(torch, cross_m, m_batch, m_target_mask_batch)
    )
    self_loss = 0.5 * (
        _masked_mse(torch, self_m, m_batch, m_target_mask_batch)
        + _masked_mse(torch, self_p, p_self_batch, p_self_target_mask_batch)
    )
    state_loss = torch.nn.functional.mse_loss(
        _loss_only_layer_norm(torch, z_m, config.layer_norm_epsilon),
        _loss_only_layer_norm(torch, z_p_cross, config.layer_norm_epsilon),
    )
    if teacher_target is None:
        teacher_loss = torch.zeros((), dtype=torch.float32, device=m_batch.device)
    else:
        teacher_loss = 0.5 * (
            torch.nn.functional.mse_loss(model.teacher_state(z_m), teacher_target)
            + torch.nn.functional.mse_loss(
                model.teacher_state(z_p_cross), teacher_target
            )
        )
    total_loss = (
        config.cross_loss_weight * cross_loss
        + config.self_loss_weight * self_loss
        + config.teacher_loss_weight * teacher_loss
        + config.state_alignment_weight * state_loss
    )
    if raw_sink is not None:
        # Verifier-only state evidence.  These are the two pre-normalization
        # predictions whose layer-normalized residual defines ``state``.
        if capture_raw_targets:
            raw_sink["validator_state_m"] = z_m
            raw_sink["validator_state_p"] = z_p_cross
        residuals = {
            "cross_p": (cross_p, p_cross_batch, p_cross_target_mask_batch),
            "cross_m": (cross_m, m_batch, m_target_mask_batch),
            "self_m": (self_m, m_batch, m_target_mask_batch),
            "self_p": (self_p, p_self_batch, p_self_target_mask_batch),
            "state": (
                _loss_only_layer_norm(torch, z_m, config.layer_norm_epsilon),
                _loss_only_layer_norm(torch, z_p_cross, config.layer_norm_epsilon),
                torch.ones_like(z_m, dtype=torch.bool),
            ),
        }
        if teacher_target is not None:
            residuals["teacher_m"] = (
                model.teacher_state(z_m),
                teacher_target,
                torch.ones_like(teacher_target, dtype=torch.bool),
            )
            residuals["teacher_p"] = (
                model.teacher_state(z_p_cross),
                teacher_target,
                torch.ones_like(teacher_target, dtype=torch.bool),
            )
        for name, (prediction, target, mask) in residuals.items():
            squared = torch.square(prediction - target) * mask.to(prediction.dtype)
            flattened = squared.reshape(squared.shape[0], -1)
            raw_sink[f"{name}_row_sse"] = flattened.sum(dim=1)
            raw_sink[f"{name}_feature_sse"] = flattened.sum(dim=0)
            raw_sink[f"{name}_count"] = mask.sum().reshape(())
            raw_sink[f"_{name}_prediction"] = prediction
            if capture_raw_targets:
                raw_sink[f"_{name}_target"] = target
    return cross_loss, self_loss, teacher_loss, state_loss, total_loss


def _drop_source_tokens(
    mask: np.ndarray,
    *,
    probability: float,
    context: V5DropoutSeedContext | None,
    epoch: int,
    modality: str,
) -> np.ndarray:
    if context is None:
        raise CCLEMultiFMV5Error(
            "positive source token dropout requires V5DropoutSeedContext"
        )
    if context.family not in V5_CONTROLS:
        raise CCLEMultiFMV5Error(
            "dropout is restricted to registered set-model controls"
        )
    if len(context.registry_positions) != mask.shape[0]:
        raise CCLEMultiFMV5Error(
            "dropout registry_positions length differs from fit rows"
        )
    if modality not in {"M", "P"}:
        raise CCLEMultiFMV5Error("dropout modality must be exactly 'M' or 'P'")
    dropped = np.ascontiguousarray(mask).copy()
    for row in range(mask.shape[0]):
        observed = np.flatnonzero(mask[row])
        drop_count = int(math.floor(probability * len(observed)))
        if drop_count >= len(observed):
            raise CCLEMultiFMV5Error("source token dropout removed an entire set")
        seed = _dropout_seed_receipt(
            context,
            epoch=epoch,
            modality=modality,
            registry_position=context.registry_positions[row],
        ).derived_value
        rng = np.random.Generator(np.random.PCG64(seed))
        selected = rng.permutation(observed)[:drop_count]
        dropped[row, selected] = False
    if np.any(dropped.sum(axis=1) == 0):
        raise CCLEMultiFMV5Error("source token dropout produced an empty set")
    return dropped


def _dropout_seed_receipt(
    context: V5DropoutSeedContext,
    *,
    epoch: int,
    modality: str,
    registry_position: int,
) -> DropoutSeedReceipt:
    """Materialize the complete protocol preimage for one row/modality/epoch."""

    if context.family not in V5_CONTROLS:
        raise CCLEMultiFMV5Error(
            "dropout receipt requires a registered set-model family"
        )
    epoch = _exact_int(epoch, "dropout epoch")
    if epoch < 1:
        raise CCLEMultiFMV5Error("dropout epoch must be a positive exact integer")
    if modality not in {"M", "P"}:
        raise CCLEMultiFMV5Error("dropout modality must be exactly 'M' or 'P'")
    registry_position = _exact_int(registry_position, "registry position")
    if registry_position not in context.registry_positions:
        raise CCLEMultiFMV5Error(
            "dropout registry position is not bound by the fit context"
        )
    preimage = "|".join(
        (
            str(context.dropout_root),
            context.stage,
            context.phase,
            context.family,
            context.direction_or_joint,
            context.candidate_id,
            str(context.outer_repeat),
            str(context.outer_fold),
            str(context.inner_repeat),
            str(context.inner_fold),
            str(context.member),
            context.purpose,
            str(epoch),
            modality,
            str(registry_position),
        )
    )
    encoded = preimage.encode("utf-8")
    digest = hashlib.sha256(encoded).digest()
    return DropoutSeedReceipt(
        root_role=f"source_token_dropout_member.{context.member}",
        root_value=context.dropout_root,
        stage=context.stage,
        phase=context.phase,
        family=context.family,
        direction_or_joint=context.direction_or_joint,
        candidate_id=context.candidate_id,
        outer_repeat=context.outer_repeat,
        outer_fold=context.outer_fold,
        inner_repeat=context.inner_repeat,
        inner_fold=context.inner_fold,
        member=context.member,
        purpose=context.purpose,
        epoch=epoch,
        modality=modality,
        registry_position=registry_position,
        full_preimage=preimage,
        preimage_sha256=digest.hex(),
        derivation="uint64_be_first8_mod_2^63",
        derived_value=_dropout_derived_value(encoded),
    )


def _dropout_derived_value(full_preimage_utf8: bytes) -> int:
    return int.from_bytes(hashlib.sha256(full_preimage_utf8).digest()[:8], "big") % (
        2**63
    )


def _validate_dropout_context_for_fit(
    context: V5DropoutSeedContext | None,
    *,
    control: V5Control,
    row_count: int,
    config: MultiFMBidirV5Config,
) -> None:
    if context is None:
        return
    if context.family != control:
        raise CCLEMultiFMV5Error("dropout family differs from the fitted control")
    expected_phase = "A1" if control == "full" else "A2"
    if context.phase != expected_phase:
        raise CCLEMultiFMV5Error("dropout phase differs from the fitted control family")
    if context.direction_or_joint != "joint":
        raise CCLEMultiFMV5Error("bidirectional fit dropout direction must be joint")
    if len(context.registry_positions) != row_count:
        raise CCLEMultiFMV5Error(
            "dropout registry_positions length differs from fit rows"
        )
    q_token = "q000" if config.source_token_dropout == 0.0 else "q005"
    if control == "no_scgpt":
        expected_candidate = (
            f"v5-noscgpt-k{config.latent_slots:02d}-"
            f"d{config.model_dimension:03d}-{q_token}"
        )
    else:
        teacher_token = "t010" if config.teacher_loss_weight == 0.1 else "t050"
        expected_candidate = (
            f"v5-k{config.latent_slots:02d}-d{config.model_dimension:03d}-"
            f"{teacher_token}-{q_token}"
        )
    if context.candidate_id != expected_candidate:
        raise CCLEMultiFMV5Error(
            "dropout candidate_id is not the registered token for this fit"
        )


def _dropout_seed_manifest(
    context: V5DropoutSeedContext | None,
    *,
    epochs: int,
) -> DropoutSeedManifest:
    if context is None:
        raise CCLEMultiFMV5Error("dropout manifest requires its exact seed context")
    if context.family not in V5_CONTROLS:
        raise CCLEMultiFMV5Error(
            "dropout manifest is restricted to full and seven set-model controls"
        )
    epochs = _exact_int(epochs, "dropout manifest epochs", minimum=1)
    positions = np.asarray(context.registry_positions, dtype="<i8")
    if not positions.flags.c_contiguous or positions.dtype.str != "<i8":
        raise CCLEMultiFMV5Error("dropout positions are not canonical C-order <i8")
    position_bytes = positions.tobytes(order="C")
    seed_table = np.empty((epochs, 2, len(positions)), dtype="<u8", order="C")
    stream = hashlib.sha256()
    seen_preimages: set[str] = set()
    count = 0
    for epoch_index, epoch in enumerate(range(1, epochs + 1)):
        for modality_index, modality in enumerate(("M", "P")):
            for position_index, position in enumerate(context.registry_positions):
                receipt = _dropout_seed_receipt(
                    context,
                    epoch=epoch,
                    modality=modality,
                    registry_position=position,
                )
                encoded = receipt.full_preimage.encode("utf-8")
                stream.update(len(encoded).to_bytes(8, "little"))
                stream.update(encoded)
                stream.update(receipt.derived_value.to_bytes(8, "little"))
                seed_table[epoch_index, modality_index, position_index] = (
                    receipt.derived_value
                )
                if receipt.full_preimage in seen_preimages:
                    raise CCLEMultiFMV5Error(
                        "dropout preimage stream is not exact-once"
                    )
                seen_preimages.add(receipt.full_preimage)
                count += 1
    expected = epochs * 2 * len(context.registry_positions)
    if count != expected:
        raise CCLEMultiFMV5Error("dropout manifest count differs from consumption")
    if not seed_table.flags.c_contiguous or seed_table.dtype.str != "<u8":
        raise CCLEMultiFMV5Error("dropout seed table is not canonical C-order <u8")
    seed_bytes = seed_table.tobytes(order="C")
    return DropoutSeedManifest(
        context=context,
        schema_version="creed.ccle-multifm-v5.dropout-seed-manifest.v1",
        root_role=f"source_token_dropout_member.{context.member}",
        root_value=context.dropout_root,
        first_epoch=1,
        last_epoch=epochs,
        epoch_count=epochs,
        modality_order=("M", "P"),
        positions_dtype="<i8",
        positions_shape=positions.shape,
        positions_byte_count=len(position_bytes),
        positions_raw_encoding="lowercase_hex",
        positions_raw_hex=position_bytes.hex(),
        positions_sha256=hashlib.sha256(position_bytes).hexdigest(),
        seed_table_dtype="<u8",
        seed_table_shape=seed_table.shape,
        seed_table_byte_count=len(seed_bytes),
        seed_table_raw_encoding="lowercase_hex",
        seed_table_raw_hex=seed_bytes.hex(),
        seed_table_sha256=hashlib.sha256(seed_bytes).hexdigest(),
        seed_count=count,
        exact_derived_count=count,
        stream_item_count=count,
        formula_version="creed.ccle-multifm-v5.dropout-seed-formula.v1",
        stream_encoding="uint64_le_length || utf8_preimage || uint64_le_value",
        stream_sha256=stream.hexdigest(),
        all_preimages_unique=True,
    )


def _epoch_order(row_count: int, root_seed: int, epoch: int) -> np.ndarray:
    seed = _derive_seed(root_seed, "batch_order", epoch)
    return np.random.Generator(np.random.PCG64(seed)).permutation(row_count)


def _validate_control_loss_contract(
    control: V5Control, config: MultiFMBidirV5Config
) -> None:
    if control not in V5_CONTROLS:
        raise CCLEMultiFMV5Error(f"unsupported V5 control: {control!r}")
    if control == "no_scgpt":
        if config.teacher_loss_weight != 0.0:
            raise CCLEMultiFMV5Error(
                "no_scgpt definitionally requires teacher_loss_weight=0.0"
            )
    elif config.teacher_loss_weight not in (0.1, 0.5):
        raise CCLEMultiFMV5Error(
            f"control {control!r} requires teacher_loss_weight in {{0.1,0.5}}"
        )


def _validate_intervention_seed(
    control: V5Control, config: MultiFMBidirV5Config
) -> int | None:
    """Bind one physical-attempt intervention seed without derivation or reuse."""

    seed = config.intervention_seed
    if control in _NO_INTERVENTION_SEED_CONTROLS:
        if seed is not None:
            raise CCLEMultiFMV5Error(
                f"control {control!r} requires intervention_seed is exactly None"
            )
        return None
    if control not in _REQUIRED_INTERVENTION_SEED_CONTROLS:
        raise CCLEMultiFMV5Error(f"unsupported V5 control: {control!r}")
    value = _exact_int(seed, "intervention_seed")
    if value >= 2**63:
        raise CCLEMultiFMV5Error("intervention_seed must be smaller than 2^63")
    return value


@dataclass
class V5TrainingSession:
    """One prepared physical fit with explicit epoch and validation boundaries."""

    config: MultiFMBidirV5Config
    control: V5Control
    dropout_context: V5DropoutSeedContext
    m_values: np.ndarray
    m_mask: np.ndarray
    p_values: np.ndarray
    p_mask: np.ndarray
    m_scaled: np.ndarray
    p_scaled: np.ndarray
    metabolite_scaler: FeatureScaler
    protein_scaler: FeatureScaler
    teacher: WhitenedPCATeacher | None
    teacher_state: np.ndarray | None
    effective_esm: np.ndarray | None
    intervention_permutation: tuple[int, ...] | None
    route_indices: RouteShuffleIndexReceipt | None
    route_permutation: np.ndarray | None
    teacher_transform: str
    esm_transform: str
    scgpt_opened: bool
    esm_opened: bool
    intervention_seed: int | None
    torch: Any
    model: Any
    optimizer: Any
    determinism: DeterminismReceipt
    parameter_count: ParameterCount
    optimizer_receipt: OptimizerReceipt
    initial_model_sha256: str
    initial_teacher_sha256: str
    history: list[EpochLossReceipt] = field(default_factory=list)

    @property
    def completed_epochs(self) -> int:
        return len(self.history)

    def train_one_epoch(self) -> EpochLossReceipt:
        epoch = self.completed_epochs + 1
        if epoch > 1000:
            raise CCLEMultiFMV5Error("training session exceeded epoch 1000")
        torch = self.torch
        device = torch.device(self.config.device)
        m_source_mask = _drop_source_tokens(
            self.m_mask,
            probability=self.config.source_token_dropout,
            context=self.dropout_context,
            epoch=epoch,
            modality="M",
        )
        p_source_mask = _drop_source_tokens(
            self.p_mask,
            probability=self.config.source_token_dropout,
            context=self.dropout_context,
            epoch=epoch,
            modality="P",
        )
        row_count = self.m_values.shape[0]
        order = _epoch_order(row_count, self.config.seed, epoch)
        aggregates = {
            key: 0.0 for key in ("cross", "self", "teacher", "state", "total")
        }
        teacher_float32 = (
            None
            if self.teacher_state is None
            else np.ascontiguousarray(self.teacher_state, dtype=np.float32)
        )
        self.model.train()
        for start in range(0, row_count, self.config.batch_size):
            index = order[start : start + self.config.batch_size]
            protein_index = (
                index
                if self.route_permutation is None
                else self.route_permutation[index]
            )
            m_batch = torch.from_numpy(self.m_scaled[index]).to(device)
            p_cross_batch = torch.from_numpy(self.p_scaled[protein_index]).to(device)
            p_self_batch = torch.from_numpy(self.p_scaled[index]).to(device)
            m_source = torch.from_numpy(m_source_mask[index]).to(device)
            p_cross_source = torch.from_numpy(p_source_mask[protein_index]).to(device)
            p_self_source = torch.from_numpy(p_source_mask[index]).to(device)
            m_target = torch.from_numpy(self.m_mask[index]).to(device)
            p_cross_target = torch.from_numpy(self.p_mask[protein_index]).to(device)
            p_self_target = torch.from_numpy(self.p_mask[index]).to(device)
            teacher_target = (
                None
                if teacher_float32 is None
                else torch.from_numpy(teacher_float32[index]).to(device)
            )
            self.optimizer.zero_grad(set_to_none=True)
            components = _v5_loss_components(
                torch=torch,
                model=self.model,
                config=self.config,
                m_batch=m_batch,
                p_cross_batch=p_cross_batch,
                p_self_batch=p_self_batch,
                m_source_mask_batch=m_source,
                p_cross_source_mask_batch=p_cross_source,
                p_self_source_mask_batch=p_self_source,
                m_target_mask_batch=m_target,
                p_cross_target_mask_batch=p_cross_target,
                p_self_target_mask_batch=p_self_target,
                teacher_target=teacher_target,
                shared_protein_route=self.route_permutation is None,
            )
            if any(not bool(torch.isfinite(value).item()) for value in components):
                raise CCLEMultiFMV5Error("non-finite V5 loss component")
            components[-1].backward()
            for name, parameter in self.model.named_parameters():
                if parameter.grad is not None and not bool(
                    torch.isfinite(parameter.grad).all().item()
                ):
                    raise CCLEMultiFMV5Error(f"non-finite gradient for {name!r}")
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            if not bool(torch.isfinite(norm).item()):
                raise CCLEMultiFMV5Error("non-finite global gradient norm")
            self.optimizer.step()
            for name, parameter in self.model.named_parameters():
                if not bool(torch.isfinite(parameter).all().item()):
                    raise CCLEMultiFMV5Error(f"non-finite parameter {name!r}")
            batch_rows = len(index)
            for key, value in zip(
                ("cross", "self", "teacher", "state", "total"), components
            ):
                aggregates[key] += float(value.detach().cpu()) * batch_rows
        receipt = EpochLossReceipt(
            epoch=epoch,
            cross=aggregates["cross"] / row_count,
            self_reconstruction=aggregates["self"] / row_count,
            teacher=aggregates["teacher"] / row_count,
            state_alignment=aggregates["state"] / row_count,
            total=aggregates["total"] / row_count,
        )
        self.history.append(receipt)
        return receipt

    def evaluate_early_stop(
        self,
        *,
        metabolites: np.ndarray,
        proteins: np.ndarray,
        protein_mask: np.ndarray,
        scgpt: np.ndarray | None,
        metabolite_mask: np.ndarray | None = None,
        intervention_seed: int | None = None,
        capture_raw: bool = False,
        capture_targets: bool = False,
    ) -> (
        float | tuple[float, tuple[NamedTensorState, ...], tuple[NamedTensorState, ...]]
    ):
        """Evaluate the exact complete nested loss in one dropout-free batch."""

        if capture_targets and not capture_raw:
            raise CCLEMultiFMV5Error("residual targets require raw capture")
        m_values, m_mask = _matrix_and_mask(
            metabolites, metabolite_mask, "early-stop metabolites"
        )
        p_values, p_mask = _matrix_and_mask(
            proteins, protein_mask, "early-stop proteins", require_mask=True
        )
        if (
            m_values.shape[0] != p_values.shape[0]
            or m_values.shape[1:] != (V5_METABOLITE_COUNT,)
            or p_values.shape[1:] != (V5_PROTEIN_COUNT,)
            or not bool(m_mask.all())
            or not bool(p_mask.all())
        ):
            raise CCLEMultiFMV5Error("early-stop endpoint contract changed")
        requires_early_seed = self.control in SCGPT_EARLY_INTERVENTION_CONTROLS
        if requires_early_seed:
            seed = _exact_int(intervention_seed, "early-stop intervention_seed")
            if seed >= 2**63:
                raise CCLEMultiFMV5Error("early-stop intervention_seed is too large")
        elif intervention_seed is not None:
            raise CCLEMultiFMV5Error(
                "this family must reuse selection identity or exact None at early-stop"
            )
        else:
            seed = None
        teacher_state: np.ndarray | None = None
        if self.control == "no_scgpt":
            if scgpt is not None:
                raise CCLEMultiFMV5Error("no_scgpt early-stop input must be unmounted")
        else:
            if self.teacher is None or scgpt is None:
                raise CCLEMultiFMV5Error("early-stop scGPT/teacher artifact is absent")
            if (
                type(scgpt) is not np.ndarray
                or scgpt.ndim != 2
                or scgpt.shape != (m_values.shape[0], V5_SCGPT_DIMENSION)
            ):
                raise CCLEMultiFMV5Error(
                    "early-stop scGPT must be exactly [n_early, 512]"
                )
            teacher_state = self.teacher.transform(scgpt)
            if self.control == "scgpt_row_deranged":
                teacher_state = np.ascontiguousarray(
                    teacher_state[_sattolo_permutation(len(teacher_state), seed)]
                )
            elif self.control == "scgpt_covariance_random":
                teacher_state = _covariance_random_teacher(teacher_state, seed)
        route_permutation: np.ndarray | None = None
        if self.control == "route_shuffle":
            route_permutation = np.asarray(
                _route_shuffle_indices(len(m_values), seed).permutation,
                dtype=np.int64,
            )
        m_scaled = self.metabolite_scaler.transform(m_values, m_mask)
        p_scaled = self.protein_scaler.transform(p_values, p_mask)
        all_m = np.ones_like(m_mask, dtype=bool)
        all_p = np.ones_like(p_mask, dtype=bool)
        index = np.arange(len(m_values))
        protein_index = index if route_permutation is None else route_permutation
        torch = self.torch
        device = torch.device(self.config.device)
        self.model.eval()
        with torch.no_grad():
            teacher_target = (
                None
                if teacher_state is None
                else torch.from_numpy(
                    np.ascontiguousarray(teacher_state, dtype=np.float32)
                ).to(device)
            )
            raw_sink: dict[str, Any] | None = {} if capture_raw else None
            components = _v5_loss_components(
                torch=torch,
                model=self.model,
                config=self.config,
                m_batch=torch.from_numpy(m_scaled).to(device),
                p_cross_batch=torch.from_numpy(p_scaled[protein_index]).to(device),
                p_self_batch=torch.from_numpy(p_scaled).to(device),
                m_source_mask_batch=torch.from_numpy(all_m).to(device),
                p_cross_source_mask_batch=torch.from_numpy(all_p[protein_index]).to(
                    device
                ),
                p_self_source_mask_batch=torch.from_numpy(all_p).to(device),
                m_target_mask_batch=torch.from_numpy(m_mask).to(device),
                p_cross_target_mask_batch=torch.from_numpy(p_mask[protein_index]).to(
                    device
                ),
                p_self_target_mask_batch=torch.from_numpy(p_mask).to(device),
                teacher_target=teacher_target,
                shared_protein_route=route_permutation is None,
                raw_sink=raw_sink,
                capture_raw_targets=capture_targets,
            )
        value = float(components[-1].detach().cpu())
        if not np.isfinite(value):
            raise CCLEMultiFMV5Error("non-finite early-stop total loss")
        if not capture_raw:
            return value
        if raw_sink is None:
            raise AssertionError("raw validation sink absent")
        tensors_list: list[NamedTensorState] = []
        predictions: list[NamedTensorState] = []
        for name, tensor in sorted(raw_sink.items()):
            if name.startswith("_"):
                prediction_state = _named_tensor_state(name[1:], tensor)
                predictions.append(prediction_state)
                if name.endswith("_target"):
                    continue
                # Complete per-epoch predictions are required by the independent
                # validator; the digest remains for producer-summary closure.
                if capture_targets:
                    tensors_list.append(prediction_state)
                prediction = np.frombuffer(
                    prediction_state.raw_bytes, dtype=np.dtype(prediction_state.dtype)
                ).reshape(prediction_state.shape)
                digest = hashlib.sha256(prediction.tobytes(order="C")).digest()
                tensors_list.append(
                    _named_tensor_state(
                        f"{name[1:]}_sha256",
                        self.torch.from_numpy(
                            np.frombuffer(digest, dtype=np.uint8).copy()
                        ),
                    )
                )
            else:
                tensors_list.append(_named_tensor_state(name, tensor))
        tensors = tuple(tensors_list)
        return value, tensors, tuple(predictions)

    def snapshot_state(
        self, *, epoch: int, include_tensors: bool = False
    ) -> tuple[str, str, RNGStateReceipt, TensorSafeCheckpoint | None]:
        model_sha = _tensor_collection_sha256(self.model.state_dict().items())
        optimizer_sha = _optimizer_state_sha256(self.model, self.optimizer)
        rng = _rng_state_receipt(self.torch)
        checkpoint = (
            _tensor_safe_checkpoint(self.model, epoch) if include_tensors else None
        )
        return model_sha, optimizer_sha, rng, checkpoint

    def finalize_fitted(self) -> FittedMultiFMBidirV5Fold:
        if not self.history:
            raise CCLEMultiFMV5Error("cannot finalize an untrained V5 session")
        model = self.model
        model.eval()
        final_teacher_hash = _tensor_collection_sha256(
            model.teacher_head.named_parameters()
        )
        final_model_hash = _tensor_collection_sha256(model.named_parameters())
        teacher_active = self.control != "no_scgpt"
        teacher_unchanged = self.initial_teacher_sha256 == final_teacher_hash
        teacher_parameters = tuple(model.teacher_head.parameters())
        optimizer_parameters = tuple(
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        )
        teacher_in_optimizer = all(
            any(parameter is candidate for candidate in optimizer_parameters)
            for parameter in teacher_parameters
        )
        teacher_optimizer_state_created = any(
            parameter in self.optimizer.state for parameter in teacher_parameters
        )
        if not teacher_in_optimizer:
            raise CCLEMultiFMV5Error(
                "teacher head is absent from the common optimizer registry"
            )
        if self.control == "no_scgpt" and not teacher_unchanged:
            raise CCLEMultiFMV5Error(
                "no_scgpt inert teacher head changed from initialization"
            )
        if self.control == "no_scgpt" and teacher_optimizer_state_created:
            raise CCLEMultiFMV5Error(
                "no_scgpt inert teacher head acquired optimizer state"
            )
        if self.control != "no_scgpt" and teacher_unchanged:
            raise CCLEMultiFMV5Error(
                "active teacher head did not change during fitting"
            )
        esm_feature_count = (
            V5_ESM2_DIMENSION
            if self.effective_esm is None
            else int(self.effective_esm.shape[1])
        )
        full_identity_parameters = (
            esm_feature_count * self.config.model_dimension
            + self.config.model_dimension
        )
        if self.control == "no_esm_identity_only":
            identity_width = int(model.protein_identity_width)
            effective_identity_parameters = (
                self.p_values.shape[1] * identity_width
                + identity_width * self.config.model_dimension
                + self.config.model_dimension
            )
        else:
            identity_width = None
            effective_identity_parameters = full_identity_parameters
        excess = effective_identity_parameters - full_identity_parameters
        if excess < 0:
            raise CCLEMultiFMV5Error(
                "no-ESM identity control is below full-arm capacity"
            )
        parameter_receipt = ControlParameterReceipt(
            control=self.control,
            full_esm_projection_parameters=full_identity_parameters,
            effective_identity_parameters=effective_identity_parameters,
            identity_capacity_excess=excess,
            learned_identity_width=identity_width,
            teacher_head_parameters=int(
                sum(value.numel() for value in model.teacher_head.parameters())
            ),
            teacher_head_active=teacher_active,
            teacher_head_in_optimizer_registry=teacher_in_optimizer,
            teacher_head_optimizer_state_created=teacher_optimizer_state_created,
            teacher_head_initial_sha256=self.initial_teacher_sha256,
            teacher_head_final_sha256=final_teacher_hash,
            teacher_head_unchanged=teacher_unchanged,
            initial_model_sha256=self.initial_model_sha256,
            final_model_sha256=final_model_hash,
            same_prediction_interface=True,
        )
        transform_receipt = ControlTransformReceipt(
            control=self.control,
            intervention_seed=self.intervention_seed,
            scgpt_input_opened=self.scgpt_opened,
            esm_input_opened=self.esm_opened,
            effective_teacher=(
                None
                if self.teacher_state is None
                else _array_receipt(self.teacher_state)
            ),
            effective_esm=(
                None
                if self.effective_esm is None
                else _array_receipt(self.effective_esm)
            ),
            permutation=self.intervention_permutation,
            route_indices=self.route_indices,
            transform=(
                f"teacher={self.teacher_transform};esm={self.esm_transform}"
                + (
                    ";route=sattolo_cross_pairing"
                    if self.route_permutation is not None
                    else ""
                )
            ),
        )
        architecture_contract = {
            "stage": "single_fold_exact_core_non_authorizing",
            "array_only": True,
            "set_architecture": "two_modality_specific_2xISAB_16_inducing",
            "shared_pma_module_and_slots": True,
            "shared_latent_core_blocks": 2,
            "pre_norm_mab": True,
            "latent_shape": (
                "batch",
                self.config.latent_slots,
                self.config.model_dimension,
            ),
            "raw_source_to_output_skip": False,
            "inference_scgpt_allowed": False,
            "opposite_endpoint_inference_allowed": False,
            "math_sdp_only": True,
            "network_dtype": "float32",
            "dropout_manifest_q0_full_derivation": True,
            "dropout_manifest_attempt_terminal_binding_implemented": False,
            "route_shuffle_explicit_index_receipt": self.route_indices is not None,
            "scientific_authority": False,
            "submission_authority": False,
            "publication_authority": False,
            "tensor_only_inference_serializer_implemented": False,
            "independent_fx_static_verifier_implemented": False,
        }
        return FittedMultiFMBidirV5Fold(
            config=self.config,
            dropout_context=self.dropout_context,
            dropout_seed_manifest=_dropout_seed_manifest(
                self.dropout_context, epochs=self.completed_epochs
            ),
            control=self.control,
            metabolite_scaler=self.metabolite_scaler,
            protein_scaler=self.protein_scaler,
            scgpt_teacher=self.teacher,
            effective_esm_tokens=(
                None if self.effective_esm is None else _readonly(self.effective_esm)
            ),
            parameter_count=self.parameter_count,
            control_parameters=parameter_receipt,
            control_transform=transform_receipt,
            determinism=self.determinism,
            optimizer_receipt=self.optimizer_receipt,
            loss_history=tuple(self.history),
            architecture_contract=architecture_contract,
            _model=model,
            _torch=self.torch,
        )


@dataclass
class V5CheckpointInferenceSession:
    """Checkpoint-only evaluator with no optimizer, fit, or backward surface.

    The object intentionally carries only the state consumed by the normative
    dropout-free early-stop evaluator.  It delegates to the same loss path as
    :class:`V5TrainingSession`, while making an optimizer impossible to reach
    from this process boundary.
    """

    config: MultiFMBidirV5Config
    control: V5Control
    metabolite_scaler: FeatureScaler
    protein_scaler: FeatureScaler
    teacher: WhitenedPCATeacher | None
    torch: Any
    model: Any
    checkpoint_sha256: str

    def evaluate_early_stop(
        self,
        *,
        metabolites: np.ndarray,
        proteins: np.ndarray,
        protein_mask: np.ndarray,
        scgpt: np.ndarray | None,
        metabolite_mask: np.ndarray | None = None,
        intervention_seed: int | None = None,
        capture_raw: bool = False,
        capture_targets: bool = False,
    ) -> (
        float | tuple[float, tuple[NamedTensorState, ...], tuple[NamedTensorState, ...]]
    ):
        return V5TrainingSession.evaluate_early_stop(
            self,  # type: ignore[arg-type]
            metabolites=metabolites,
            proteins=proteins,
            protein_mask=protein_mask,
            scgpt=scgpt,
            metabolite_mask=metabolite_mask,
            intervention_seed=intervention_seed,
            capture_raw=capture_raw,
            capture_targets=capture_targets,
        )


SCGPT_EARLY_INTERVENTION_CONTROLS = frozenset(
    {"scgpt_row_deranged", "scgpt_covariance_random", "route_shuffle"}
)


def prepare_multifm_bidir_v5_checkpoint_inference(
    *,
    metabolites: np.ndarray,
    proteins: np.ndarray,
    protein_mask: np.ndarray,
    esm_embeddings: np.ndarray | None,
    scgpt: np.ndarray | None,
    checkpoint_tensors: Mapping[str, np.ndarray],
    expected_checkpoint_sha256: str,
    config: MultiFMBidirV5Config,
    control: V5Control,
    metabolite_mask: np.ndarray | None = None,
) -> V5CheckpointInferenceSession:
    """Rebuild preprocessing and load one selected checkpoint for inference.

    This is the independent counterpart to the optimizer process.  It never
    initializes a protocol model, constructs an optimizer, performs a fit, or
    exposes ``backward``.  Every tensor in the exact architecture state must be
    supplied and the aggregate state hash must match the externally bound
    selected checkpoint.
    """

    if (
        type(expected_checkpoint_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", expected_checkpoint_sha256) is None
    ):
        raise CCLEMultiFMV5Error("expected checkpoint hash is not lowercase SHA-256")
    if not isinstance(checkpoint_tensors, Mapping) or not checkpoint_tensors:
        raise CCLEMultiFMV5Error("checkpoint tensor mapping is absent")
    _validate_control_loss_contract(control, config)
    intervention_seed = _validate_intervention_seed(control, config)
    m_values, m_mask = _matrix_and_mask(
        metabolites, metabolite_mask, "checkpoint fit metabolites"
    )
    p_values, p_mask = _matrix_and_mask(
        proteins, protein_mask, "checkpoint fit proteins", require_mask=True
    )
    if m_values.shape[0] != p_values.shape[0]:
        raise CCLEMultiFMV5Error("checkpoint fit endpoint row counts differ")
    if (
        m_values.shape[1:] != (V5_METABOLITE_COUNT,)
        or p_values.shape[1:] != (V5_PROTEIN_COUNT,)
        or not bool(m_mask.all())
        or not bool(p_mask.all())
    ):
        raise CCLEMultiFMV5Error("checkpoint fit endpoint contract changed")
    if m_values.shape[0] <= config.teacher_dimension and control != "no_scgpt":
        raise CCLEMultiFMV5Error("checkpoint fit rows do not support teacher PCA")

    effective_esm, _, _, _ = _prepare_esm_control(
        esm_embeddings,
        control=control,
        protein_count=p_values.shape[1],
        seed=intervention_seed,
    )
    esm_feature_count = (
        V5_ESM2_DIMENSION if effective_esm is None else int(effective_esm.shape[1])
    )
    if esm_feature_count != V5_ESM2_DIMENSION:
        raise CCLEMultiFMV5Error("checkpoint ESM2 feature count changed")
    teacher, _, _, _, _ = _prepare_teacher_control(
        scgpt,
        control=control,
        rows=m_values.shape[0],
        config=config,
        intervention_seed=intervention_seed,
    )
    metabolite_scaler = _fit_scaler(
        m_values,
        m_mask,
        epsilon=config.scale_epsilon,
        label="checkpoint fit metabolites",
    )
    protein_scaler = _fit_scaler(
        p_values,
        p_mask,
        epsilon=config.scale_epsilon,
        label="checkpoint fit proteins",
    )
    torch, _ = _require_and_configure_torch(config)
    model = _build_model(
        torch,
        metabolite_count=V5_METABOLITE_COUNT,
        protein_count=V5_PROTEIN_COUNT,
        esm_feature_count=esm_feature_count,
        effective_esm=effective_esm,
        config=config,
        control=control,
    )
    expected_state = model.state_dict()
    parameter_names = {name for name, _ in model.named_parameters()}
    if set(checkpoint_tensors) != set(expected_state):
        missing = sorted(set(expected_state) - set(checkpoint_tensors))
        extra = sorted(set(checkpoint_tensors) - set(expected_state))
        raise CCLEMultiFMV5Error(
            f"checkpoint tensor registry changed: missing={missing!r}; extra={extra!r}"
        )
    loaded: dict[str, Any] = {}
    for name, expected in sorted(expected_state.items()):
        value = checkpoint_tensors[name]
        if type(value) is not np.ndarray or value.dtype.hasobject:
            raise CCLEMultiFMV5Error(
                f"checkpoint tensor {name!r} is not numeric ndarray"
            )
        array = np.ascontiguousarray(value)
        expected_array = np.ascontiguousarray(expected.detach().cpu().numpy())
        if array.shape != expected_array.shape or array.dtype != expected_array.dtype:
            raise CCLEMultiFMV5Error(f"checkpoint tensor {name!r} shape/dtype changed")
        if not np.isfinite(array).all():
            raise CCLEMultiFMV5Error(f"checkpoint tensor {name!r} is non-finite")
        if name not in parameter_names and not np.array_equal(array, expected_array):
            raise CCLEMultiFMV5Error(
                f"checkpoint persistent input buffer {name!r} differs from frozen input"
            )
        loaded[name] = torch.from_numpy(array.copy())
    incompatible = model.load_state_dict(loaded, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise CCLEMultiFMV5Error("checkpoint strict state load changed")
    model = model.to(device=torch.device(config.device), dtype=torch.float32)
    model.eval()
    actual_checkpoint_sha256 = _tensor_collection_sha256(model.state_dict().items())
    if actual_checkpoint_sha256 != expected_checkpoint_sha256:
        raise CCLEMultiFMV5Error("checkpoint aggregate differs from external selection")
    return V5CheckpointInferenceSession(
        config=config,
        control=control,
        metabolite_scaler=metabolite_scaler,
        protein_scaler=protein_scaler,
        teacher=teacher,
        torch=torch,
        model=model,
        checkpoint_sha256=actual_checkpoint_sha256,
    )


def prepare_multifm_bidir_v5_training(
    *,
    metabolites: np.ndarray,
    proteins: np.ndarray,
    protein_mask: np.ndarray,
    esm_embeddings: np.ndarray | None,
    scgpt: np.ndarray | None,
    config: MultiFMBidirV5Config,
    control: V5Control,
    dropout_context: V5DropoutSeedContext,
    metabolite_mask: np.ndarray | None = None,
) -> V5TrainingSession:
    """Prepare one fit exactly once; no held or early-stop row is accepted here."""

    _validate_control_loss_contract(control, config)
    intervention_seed = _validate_intervention_seed(control, config)
    m_values, m_mask = _matrix_and_mask(metabolites, metabolite_mask, "fit metabolites")
    p_values, p_mask = _matrix_and_mask(
        proteins, protein_mask, "fit proteins", require_mask=True
    )
    if m_values.shape[0] != p_values.shape[0]:
        raise CCLEMultiFMV5Error("metabolite and protein fit row counts differ")
    _validate_dropout_context_for_fit(
        dropout_context, control=control, row_count=m_values.shape[0], config=config
    )
    if not bool(m_mask.all()) or not bool(p_mask.all()):
        raise CCLEMultiFMV5Error(
            "V5 fit endpoints must be complete; missing masks are inference-only"
        )
    if m_values.shape[1] != V5_METABOLITE_COUNT:
        raise CCLEMultiFMV5Error("metabolite feature count must be exactly 225")
    if p_values.shape[1] != V5_PROTEIN_COUNT:
        raise CCLEMultiFMV5Error("protein feature count must be exactly 584")
    if m_values.shape[0] <= config.teacher_dimension and control != "no_scgpt":
        raise CCLEMultiFMV5Error("fit rows must exceed the teacher dimension")
    effective_esm, esm_permutation, esm_transform, esm_opened = _prepare_esm_control(
        esm_embeddings,
        control=control,
        protein_count=p_values.shape[1],
        seed=intervention_seed,
    )
    esm_feature_count = (
        V5_ESM2_DIMENSION if effective_esm is None else int(effective_esm.shape[1])
    )
    if esm_feature_count != V5_ESM2_DIMENSION:
        raise CCLEMultiFMV5Error("ESM2 feature count must be exactly 1280")
    teacher, teacher_state, teacher_permutation, teacher_transform, scgpt_opened = (
        _prepare_teacher_control(
            scgpt,
            control=control,
            rows=m_values.shape[0],
            config=config,
            intervention_seed=intervention_seed,
        )
    )
    intervention_permutation = teacher_permutation or esm_permutation
    route_indices: RouteShuffleIndexReceipt | None = None
    route_permutation: np.ndarray | None = None
    if control == "route_shuffle":
        route_indices = _route_shuffle_indices(
            m_values.shape[0], _exact_int(intervention_seed, "intervention_seed")
        )
        route_permutation = np.asarray(route_indices.permutation, dtype=np.int64)
        intervention_permutation = route_indices.permutation
    m_scaler = _fit_scaler(
        m_values, m_mask, epsilon=config.scale_epsilon, label="fit metabolites"
    )
    p_scaler = _fit_scaler(
        p_values, p_mask, epsilon=config.scale_epsilon, label="fit proteins"
    )
    m_scaled = m_scaler.transform(m_values, m_mask)
    p_scaled = p_scaler.transform(p_values, p_mask)
    torch, determinism = _require_and_configure_torch(config)
    model = _build_model(
        torch,
        metabolite_count=m_values.shape[1],
        protein_count=p_values.shape[1],
        esm_feature_count=esm_feature_count,
        effective_esm=effective_esm,
        config=config,
        control=control,
    )
    initialization_seeds = _initialize_model(torch, model, config.seed)
    counts = _parameter_count(model, initialization_seeds)
    initial_model_hash = _tensor_collection_sha256(model.named_parameters())
    initial_teacher_hash = _tensor_collection_sha256(
        model.teacher_head.named_parameters()
    )
    device = torch.device(config.device)
    model = model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
        amsgrad=False,
        foreach=False,
        fused=False,
        capturable=False,
        maximize=False,
        differentiable=False,
    )
    optimizer_receipt = _optimizer_receipt(torch, optimizer, model, config)
    return V5TrainingSession(
        config=config,
        control=control,
        dropout_context=dropout_context,
        m_values=m_values,
        m_mask=m_mask,
        p_values=p_values,
        p_mask=p_mask,
        m_scaled=m_scaled,
        p_scaled=p_scaled,
        metabolite_scaler=m_scaler,
        protein_scaler=p_scaler,
        teacher=teacher,
        teacher_state=teacher_state,
        effective_esm=effective_esm,
        intervention_permutation=intervention_permutation,
        route_indices=route_indices,
        route_permutation=route_permutation,
        teacher_transform=teacher_transform,
        esm_transform=esm_transform,
        scgpt_opened=scgpt_opened,
        esm_opened=esm_opened,
        intervention_seed=intervention_seed,
        torch=torch,
        model=model,
        optimizer=optimizer,
        determinism=determinism,
        parameter_count=counts,
        optimizer_receipt=optimizer_receipt,
        initial_model_sha256=initial_model_hash,
        initial_teacher_sha256=initial_teacher_hash,
    )


def run_multifm_bidir_v5_optimizer_attempt(
    session: V5TrainingSession,
    *,
    early_metabolites: np.ndarray,
    early_proteins: np.ndarray,
    early_protein_mask: np.ndarray,
    early_scgpt: np.ndarray | None,
    early_metabolite_mask: np.ndarray | None = None,
    early_intervention_seed: int | None = None,
    maximum_epochs: int = 1000,
) -> V5OptimizerAttemptArtifact:
    """Run one optimizer-fit process through the exact registered stop rule."""

    maximum_epochs = _exact_int(maximum_epochs, "maximum_epochs", minimum=50)
    if maximum_epochs > 1000 or session.completed_epochs:
        raise CCLEMultiFMV5Error(
            "optimizer attempt requires one fresh session and at most 1000 epochs"
        )
    validation: list[EarlyStopLossReceipt] = []
    raw_validation: list[RawValidationEvidence] = []
    epoch_states: list[EpochStateReceipt] = []
    incumbent_loss: float | None = None
    incumbent_epoch: int | None = None
    incumbent_checkpoint: TensorSafeCheckpoint | None = None
    incumbent_predictions: tuple[NamedTensorState, ...] | None = None
    without_improvement = 0
    for epoch in range(1, maximum_epochs + 1):
        before_model, before_optimizer, before_rng, _ = session.snapshot_state(
            epoch=epoch - 1
        )
        training = session.train_one_epoch()
        evaluated = session.evaluate_early_stop(
            metabolites=early_metabolites,
            proteins=early_proteins,
            protein_mask=early_protein_mask,
            scgpt=early_scgpt,
            metabolite_mask=early_metabolite_mask,
            intervention_seed=early_intervention_seed,
            capture_raw=True,
            capture_targets=True,
        )
        if type(evaluated) is not tuple:
            raise AssertionError("optimizer validation did not return raw evidence")
        value, raw_tensors, full_predictions = evaluated
        raw_validation.append(
            RawValidationEvidence(
                epoch=epoch,
                tensors=raw_tensors,
                evidence_sha256=_canonical_json_sha256(
                    [
                        {
                            "dtype": item.dtype,
                            "name": item.name,
                            "sha256": item.sha256,
                            "shape": item.shape,
                        }
                        for item in raw_tensors
                    ]
                ),
            )
        )
        eligible = epoch >= 50
        improved = False
        if eligible:
            if incumbent_loss is None:
                improved = True
            elif value < incumbent_loss - 1e-5:
                improved = True
            if improved:
                incumbent_loss = value
                incumbent_epoch = epoch
                without_improvement = 0
            else:
                without_improvement += 1
        stop = eligible and without_improvement >= 40
        model_sha, optimizer_sha, rng, checkpoint = session.snapshot_state(
            epoch=epoch, include_tensors=improved
        )
        if improved:
            if checkpoint is None:
                raise AssertionError("improved epoch lacks tensor-safe checkpoint")
            incumbent_checkpoint = checkpoint
            incumbent_predictions = full_predictions
        observation = EarlyStopLossReceipt(
            epoch=epoch,
            total=value,
            eligible=eligible,
            improved=improved,
            incumbent_epoch=incumbent_epoch,
            incumbent_loss=incumbent_loss,
            consecutive_without_improvement=without_improvement,
            stop=stop,
            model_state_sha256=model_sha,
        )
        validation.append(observation)
        epoch_states.append(
            EpochStateReceipt(
                epoch=epoch,
                before_model_state_sha256=before_model,
                before_optimizer_state_sha256=before_optimizer,
                before_rng_state=before_rng,
                training_loss=training,
                validation_loss=observation,
                model_state_sha256=model_sha,
                optimizer_state_sha256=optimizer_sha,
                rng_state=rng,
            )
        )
        if stop:
            break
    if (
        incumbent_loss is None
        or incumbent_epoch is None
        or incumbent_checkpoint is None
        or incumbent_predictions is None
    ):
        raise CCLEMultiFMV5Error("optimizer attempt produced no eligible finite epoch")
    trace_payload = [
        {
            "epoch": item.epoch,
            "total": item.total,
            "eligible": item.eligible,
            "improved": item.improved,
            "incumbent_epoch": item.incumbent_epoch,
            "incumbent_loss": item.incumbent_loss,
            "consecutive_without_improvement": item.consecutive_without_improvement,
            "stop": item.stop,
            "model_state_sha256": item.model_state_sha256,
        }
        for item in validation
    ]
    return V5OptimizerAttemptArtifact(
        completed_epochs=session.completed_epochs,
        best_epoch=incumbent_epoch,
        best_loss=incumbent_loss,
        training_trace=tuple(session.history),
        validation_trace=tuple(validation),
        epoch_states=tuple(epoch_states),
        raw_validation=tuple(raw_validation),
        selected_validation_predictions=incumbent_predictions,
        best_checkpoint=incumbent_checkpoint,
        dropout_seed_manifest=_dropout_seed_manifest(
            session.dropout_context, epochs=session.completed_epochs
        ),
        selection_state_sha256=incumbent_checkpoint.state_sha256,
        early_stop_trace_sha256=_canonical_json_sha256(trace_payload),
    )


def fit_multifm_bidir_v5_fold(
    *,
    metabolites: np.ndarray,
    proteins: np.ndarray,
    protein_mask: np.ndarray,
    esm_embeddings: np.ndarray | None,
    scgpt: np.ndarray | None,
    config: MultiFMBidirV5Config = MultiFMBidirV5Config(),
    control: V5Control = "full",
    metabolite_mask: np.ndarray | None = None,
    dropout_context: V5DropoutSeedContext | None = None,
) -> FittedMultiFMBidirV5Fold:
    """Backward-compatible fixed-epoch wrapper over ``V5TrainingSession``."""

    if dropout_context is None:
        raise CCLEMultiFMV5Error(
            "every set-model fit requires its exact dropout seed context"
        )
    session = prepare_multifm_bidir_v5_training(
        metabolites=metabolites,
        proteins=proteins,
        protein_mask=protein_mask,
        esm_embeddings=esm_embeddings,
        scgpt=scgpt,
        config=config,
        control=control,
        metabolite_mask=metabolite_mask,
        dropout_context=dropout_context,
    )
    for _ in range(config.epochs):
        session.train_one_epoch()
    return session.finalize_fitted()


__all__ = [
    "ArrayReceipt",
    "CCLEMultiFMV5Error",
    "ControlParameterReceipt",
    "ControlTransformReceipt",
    "DeterminismReceipt",
    "DropoutSeedManifest",
    "DropoutSeedReceipt",
    "EpochLossReceipt",
    "EarlyStopLossReceipt",
    "EpochStateReceipt",
    "FeatureScaler",
    "FittedMultiFMBidirV5Fold",
    "MultiFMBidirV5Config",
    "NamedParameterReceipt",
    "OptimizerReceipt",
    "OptimizerParameterGroupReceipt",
    "PCATeacherReceipt",
    "ParameterCount",
    "RouteShuffleIndexReceipt",
    "RNGStateReceipt",
    "RawValidationEvidence",
    "NamedTensorState",
    "TensorSafeCheckpoint",
    "V5Control",
    "V5DropoutSeedContext",
    "V5ForwardPrediction",
    "V5CheckpointInferenceSession",
    "V5OptimizerAttemptArtifact",
    "V5ReversePrediction",
    "V5TrainingSession",
    "V5_CONTROLS",
    "V5_ESM2_DIMENSION",
    "V5_METABOLITE_COUNT",
    "V5_PROTEIN_COUNT",
    "V5_SCGPT_DIMENSION",
    "WhitenedPCATeacher",
    "fit_multifm_bidir_v5_fold",
    "prepare_multifm_bidir_v5_checkpoint_inference",
    "prepare_multifm_bidir_v5_training",
    "run_multifm_bidir_v5_optimizer_attempt",
]

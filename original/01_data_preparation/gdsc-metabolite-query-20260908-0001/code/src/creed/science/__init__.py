"""Optional scientific evaluation APIs.

The base CREED package intentionally has no NumPy dependency. Scientific
symbols are loaded only when requested so provenance and registry modules stay
usable in the minimal governance runtime.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_METRIC_EXPORTS = frozenset(
    {
        "BootstrapDeltaCI",
        "CorrelationSummary",
        "CrossLayerMetricSummary",
        "MetricInputError",
        "OracleGapClosure",
        "StandardizedErrorSummary",
        "evaluate_cross_layer_predictions",
        "neutralized_flat_pearson",
        "oracle_gap_closure",
        "patient_target_bootstrap_delta",
        "standardized_errors",
        "summarize_centered_profile_pearson",
        "summarize_per_target_pearson",
    }
)

__all__ = sorted(_METRIC_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _METRIC_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    metrics = import_module(f"{__name__}.metrics")
    return getattr(metrics, name)

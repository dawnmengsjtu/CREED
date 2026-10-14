"""CREED: cross-layer discovery under executable evidence control.

The package root stays dependency-light and loads its public symbols lazily.
This lets isolated scientific runtimes import ``creed.science`` without first
loading unrelated governance or completion modules.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "CompletionResult",
    "FusionPolicy",
    "LAYER_SPECS",
    "Layer",
    "LayerSpec",
    "LayerState",
    "MeasuredStateReceipt",
    "StateKind",
    "SparseCompletionError",
    "SparseLayerCompleter",
]

_EXPORT_MODULE = {
    "CompletionResult": "completion",
    "FusionPolicy": "completion",
    "SparseCompletionError": "completion",
    "SparseLayerCompleter": "completion",
    "LAYER_SPECS": "layers",
    "Layer": "layers",
    "LayerSpec": "layers",
    "LayerState": "layers",
    "MeasuredStateReceipt": "layers",
    "StateKind": "layers",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})

__version__ = "0.1.0"


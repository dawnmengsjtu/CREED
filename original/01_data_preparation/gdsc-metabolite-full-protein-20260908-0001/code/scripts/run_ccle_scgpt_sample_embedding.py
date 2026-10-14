#!/usr/bin/env python3
"""Create one hash-bound scGPT embedding of frozen visible CCLE RNA counts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import stat
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Mapping

import numpy as np


repository = Path(__file__).resolve().parents[1]
source_root = repository / "src"
if not source_root.is_dir():  # pragma: no cover - packaging/runtime failure
    raise RuntimeError("runner must execute from a complete CREED source snapshot")
sys.path.insert(0, str(source_root))

from creed.science import ccle_values  # noqa: E402
from creed.science import scgpt_encoder as encoder  # noqa: E402


PROTOCOL_ID = "CREED_CCLE_SCGPT_VISIBLE_L1_SAMPLE_EMBEDDING_V1"
RUN_SCHEMA = "creed.ccle-scgpt-visible-embedding-receipt.v1"
UNCERTAINTY_SCHEMA = "creed.ccle-scgpt-embedding-uncertainty.v1"
OUTPUT_FILENAME = "ccle_scgpt_sample_embeddings.npz"
OFFICIAL_RUNTIME_MANIFEST_SHA256 = (
    "cfd3b501168608fe3410950401916b5104127c2f3fc72aa8c0cd3168e8964eb5"
)
OFFICIAL_PYTHON_RUNTIME_RECEIPT_SHA256 = (
    "3696637ffce5bf25cb043a9a21c3b25cd729a482c5fcf4c02d9d4c67481dea5b"
)
OFFICIAL_PYTHON_RUNTIME_RECEIPT_BYTES = 1_354
EXPECTED_VISIBLE_PARTITION_COUNTS = {"train": 421, "validation": 64}
EXPECTED_INTERNAL_FINAL_COUNT = 123


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _write_json_create_once(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _write_npz_create_once(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":  # Windows unit-test hosts do not expose directory fsync.
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _git_identity(path: Path) -> tuple[str, bool]:
    git_prefix = ["git", "-c", f"safe.directory={path}", "-C", str(path)]
    commit = subprocess.run(
        [*git_prefix, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if len(commit) != 40:
        raise RuntimeError("cannot bind a valid git commit")
    dirty = bool(
        subprocess.run(
            [*git_prefix, "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    )
    return commit, dirty


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("numpy", "pandas", "anndata", "torch", "scgpt"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _environment_receipt() -> dict[str, object]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "byteorder": sys.byteorder,
        "packages": _package_versions(),
        "relevant_environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
        },
    }


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers"
        ) from error
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visible-artifact", type=Path, required=True)
    parser.add_argument("--visible-manifest", type=Path)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-runtime-manifest", type=Path, required=True)
    parser.add_argument(
        "--python-runtime-receipt",
        type=Path,
        default=Path(
            "/home/node/.openclaw/workspace/creed-private/fm-runtime/"
            "scgpt-pytorch23-runtime-v1-0002/receipt.json"
        ),
    )
    parser.add_argument("--inference-contract", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--seeds", type=_parse_seeds, default=(3407, 3408, 3409, 3410)
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--input-semantics",
        choices=(encoder.CCLE_INPUT_SEMANTICS,),
        default=encoder.CCLE_INPUT_SEMANTICS,
    )
    return parser


def _reject_forbidden_artifact_path(path: Path) -> None:
    resolved = path.resolve()
    if any(
        "sealed" in part.lower() or "internal_final" in part.lower()
        for part in resolved.parts
    ):
        raise ccle_values.CCLEValueCustodyError(
            "CCLE scGPT runner refuses sealed/internal-final paths"
        )


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _safe_failure(error: BaseException) -> str:
    if isinstance(
        error,
        (ccle_values.CCLEValueCustodyError, encoder.ScGPTInputError),
    ):
        return str(error)
    return "unexpected execution failure; no assay value is included in this receipt"


def _seal_output_permissions(output: Path) -> None:
    """Permission-seal a completed or failed create-once attempt."""

    for artifact in output.iterdir():
        if artifact.is_file():
            artifact.chmod(0o444)
    output.chmod(0o555)
    if stat.S_IMODE(output.stat().st_mode) != 0o555:
        raise RuntimeError("output directory permission seal verification failed")
    if any(
        stat.S_IMODE(artifact.stat().st_mode) != 0o444
        for artifact in output.iterdir()
        if artifact.is_file()
    ):
        raise RuntimeError("output file permission seal verification failed")


def _file_binding(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": encoder.sha256_file(path),
    }


def _validate_model_runtime_manifest(
    path: Path, model_dir: Path
) -> dict[str, object]:
    source = path.resolve()
    if source.parent != model_dir.resolve():
        raise encoder.ScGPTInputError(
            "model runtime manifest must be in the model asset directory"
        )
    if not source.is_file():
        raise encoder.ScGPTInputError("model runtime manifest is missing")
    binding = _file_binding(source)
    if binding["sha256"] != OFFICIAL_RUNTIME_MANIFEST_SHA256:
        raise encoder.ScGPTInputError(
            "model runtime manifest is not the frozen original runtime manifest"
        )
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise encoder.ScGPTInputError("model runtime manifest is invalid") from error
    if not isinstance(document, dict):
        raise encoder.ScGPTInputError("model runtime manifest must be an object")
    if (
        document.get("schema") != "creed.scgpt-original-runtime.v1"
        or document.get("status") != "complete_read_only_runtime"
        or document.get("claim_authority") != "none"
    ):
        raise encoder.ScGPTInputError("unexpected model runtime manifest identity")
    expected_contract = {
        "input_style": "binned",
        "n_bins": 51,
        "max_seq_len": 1200,
        "embedding_dimension": 512,
    }
    if document.get("contract") != expected_contract:
        raise encoder.ScGPTInputError("model runtime manifest contract mismatch")
    files = document.get("files")
    expected_hashes = {
        "best_model.pt": encoder.OFFICIAL_WHOLE_HUMAN_MODEL_SHA256,
        "vocab.json": encoder.OFFICIAL_WHOLE_HUMAN_VOCAB_SHA256,
        "args.json": encoder.OFFICIAL_WHOLE_HUMAN_ARGS_SHA256,
    }
    if not isinstance(files, Mapping) or set(files) != set(expected_hashes):
        raise encoder.ScGPTInputError("model runtime manifest file set mismatch")
    for filename, expected_hash in expected_hashes.items():
        record = files[filename]
        asset = model_dir / filename
        if (
            not isinstance(record, Mapping)
            or set(record) != {"bytes", "sha256"}
            or not isinstance(record.get("bytes"), int)
            or record["bytes"] <= 0
            or record.get("sha256") != expected_hash
            or not asset.is_file()
            or asset.stat().st_size != record["bytes"]
            or encoder.sha256_file(asset) != expected_hash
        ):
            raise encoder.ScGPTInputError(
                f"model runtime manifest asset mismatch for {filename}"
            )
    return {
        **binding,
        "schema": document["schema"],
        "status": document["status"],
        "claim_authority": document["claim_authority"],
        "contract": expected_contract,
        "source_host_root": document.get("source_host_root"),
        "expected_sha256": OFFICIAL_RUNTIME_MANIFEST_SHA256,
    }


def _validate_python_runtime_receipt(path: Path) -> dict[str, object]:
    source = path.resolve()
    if not source.is_file():
        raise encoder.ScGPTInputError("Python runtime receipt is missing")
    binding = _file_binding(source)
    if (
        binding["bytes"] != OFFICIAL_PYTHON_RUNTIME_RECEIPT_BYTES
        or binding["sha256"] != OFFICIAL_PYTHON_RUNTIME_RECEIPT_SHA256
    ):
        raise encoder.ScGPTInputError("Python runtime receipt identity changed")
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise encoder.ScGPTInputError("Python runtime receipt is invalid") from error
    validated = document.get("validated", {})
    if (
        document.get("schema") != "creed.scgpt-runtime-build-receipt.v1"
        or document.get("status") != "complete"
        or document.get("automatic_retries") != 0
        or document.get("scientific_claim_authority") is not False
        or validated.get("python") != "3.10"
        or validated.get("scgpt") != "0.2.4"
        or validated.get("torch") != "2.3.0+cu121"
        or validated.get("cuda_available") is not True
        or validated.get("import_smoke") is not True
    ):
        raise encoder.ScGPTInputError("unexpected Python runtime receipt contract")
    environment_root = Path(str(document.get("environment_root", ""))).resolve()
    if Path(sys.executable).resolve().parent.parent != environment_root:
        raise encoder.ScGPTInputError("active Python is outside the bound scGPT runtime")
    return {
        **binding,
        "schema": document["schema"],
        "status": document["status"],
        "environment_root": str(environment_root),
        "validated": dict(validated),
        "expected_sha256": OFFICIAL_PYTHON_RUNTIME_RECEIPT_SHA256,
    }


def _verify_file_binding(path: Path, binding: Mapping[str, object]) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != binding["bytes"]
        or encoder.sha256_file(path) != binding["sha256"]
    ):
        raise encoder.ScGPTInputError("a bound input changed during scGPT execution")


def _model_receipt(
    contract: encoder.ScGPTModelContract,
    runtime_manifest: Mapping[str, object],
) -> dict[str, object]:
    return {
        "asset_identity": "original_whole_human_scgpt_triplet",
        "model_dir": str(contract.model_dir),
        "best_model_pt_sha256": contract.model_sha256,
        "vocab_json_sha256": contract.vocab_sha256,
        "args_json_sha256": contract.args_sha256,
        "official_asset_hashes_required": dict(
            encoder.OFFICIAL_WHOLE_HUMAN_ASSET_HASHES
        ),
        "runtime_manifest": dict(runtime_manifest),
        "input_style": contract.inference["input_style"],
        "n_bins": int(contract.inference["n_bins"]),
        "max_seq_len": int(contract.inference["max_seq_len"]),
        "value_adapter": contract.inference["value_adapter"],
        "embedding_dimension": int(contract.args["embsize"]),
        "inference_metadata_source": (
            "args.json"
            if contract.inference_contract_path is None
            else "separate_frozen_hash_bound_contract"
        ),
        "inference_contract": (
            None
            if contract.inference_contract_path is None
            else {
                "path": str(contract.inference_contract_path),
                "sha256": contract.inference_contract_sha256,
            }
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = args.visible_artifact.resolve()
    manifest = (
        args.visible_manifest.resolve()
        if args.visible_manifest is not None
        else artifact.with_name("manifest.json")
    )
    split_path = args.split.resolve()
    model_dir = args.model_dir.resolve()
    runtime_manifest_path = args.model_runtime_manifest.resolve()
    python_runtime_receipt_path = args.python_runtime_receipt.resolve()
    output = args.output_directory.resolve()
    try:
        _reject_forbidden_artifact_path(artifact)
        _reject_forbidden_artifact_path(manifest)
        _reject_forbidden_artifact_path(output)
    except ccle_values.CCLEValueCustodyError as error:
        print(str(error), file=sys.stderr)
        return 2

    config = encoder.ScGPTEmbeddingConfig(
        max_length=args.max_length,
        batch_size=args.batch_size,
        seeds=tuple(args.seeds),
        device=args.device,
        use_fast_transformer=False,
        minimum_vocab_genes=args.max_length,
        minimum_positive_genes_per_sample=args.max_length,
        input_semantics=args.input_semantics,
    )
    config_payload = config.receipt_payload()
    commit, dirty = _git_identity(repository)
    if dirty:
        print("CCLE scGPT embedding requires a clean git snapshot", file=sys.stderr)
        return 2
    code = {
        "git_commit": commit,
        "git_dirty": dirty,
        "runner_sha256": encoder.sha256_file(Path(__file__).resolve()),
        "encoder_sha256": encoder.sha256_file(
            repository / "src" / "creed" / "science" / "scgpt_encoder.py"
        ),
        "ccle_values_sha256": encoder.sha256_file(
            repository / "src" / "creed" / "science" / "ccle_values.py"
        ),
    }
    environment = _environment_receipt()

    try:
        output.mkdir(mode=0o700, parents=True, exist_ok=False)
        _fsync_directory(output.parent)
    except FileExistsError:
        print(f"STOP_CREATE_ONCE output directory exists: {output}", file=sys.stderr)
        return 73
    intent = {
        "schema": RUN_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "started_development_visible_only",
        "started_at_utc": _utc_now(),
        "visible_artifact": str(artifact),
        "visible_manifest": str(manifest),
        "frozen_split": str(split_path),
        "model_dir": str(model_dir),
        "model_runtime_manifest": str(runtime_manifest_path),
        "python_runtime_receipt": str(python_runtime_receipt_path),
        "output_directory": str(output),
        "configuration": config_payload,
        "configuration_sha256": encoder.canonical_json_sha256(config_payload),
        "code": code,
        "environment": environment,
        "custody": {
            "allowed_partitions": ["train", "validation"],
            "expected_visible_partition_counts": EXPECTED_VISIBLE_PARTITION_COUNTS,
            "expected_internal_final_count": EXPECTED_INTERNAL_FINAL_COUNT,
            "sealed_or_internal_final_loading_allowed": False,
            "parameter_fitting_allowed": False,
            "automatic_retry": False,
        },
        "scientific_claim_authority": False,
    }
    _write_json_create_once(output / "intent.json", intent)

    try:
        split = ccle_values.validate_frozen_split(
            split_path, require_real_payload=True
        )
        split_counts = {
            "train": len(split.train_sample_ids),
            "validation": len(split.validation_sample_ids),
        }
        if (
            split_counts != EXPECTED_VISIBLE_PARTITION_COUNTS
            or len(split.internal_final_sample_ids) != EXPECTED_INTERNAL_FINAL_COUNT
        ):
            raise ccle_values.CCLEValueCustodyError(
                "frozen CCLE partition counts differ from the 485-sample "
                "development contract"
            )
        runtime_manifest = _validate_model_runtime_manifest(
            runtime_manifest_path, model_dir
        )
        python_runtime = _validate_python_runtime_receipt(
            python_runtime_receipt_path
        )
        contract = encoder.load_model_contract(
            model_dir,
            inference_contract_path=args.inference_contract,
        )
        encoder.require_official_whole_human_model(contract)
        if (
            contract.inference["input_style"]
            != runtime_manifest["contract"]["input_style"]
            or int(contract.inference["n_bins"])
            != runtime_manifest["contract"]["n_bins"]
            or int(contract.inference["max_seq_len"])
            != runtime_manifest["contract"]["max_seq_len"]
            or int(contract.args["embsize"])
            != runtime_manifest["contract"]["embedding_dimension"]
        ):
            raise encoder.ScGPTInputError(
                "loaded model metadata conflicts with the runtime manifest"
            )
        visible = ccle_values.load_visible_ccle_artifact(
            artifact,
            split_path=split_path,
            manifest_path=manifest,
        )
        artifact_binding = _file_binding(artifact)
        manifest_binding = _file_binding(manifest)
        split_binding = _file_binding(split_path)
        model_bindings = {
            name: _file_binding(model_dir / filename)
            for name, filename in {
                "model": "best_model.pt",
                "vocab": "vocab.json",
                "args": "args.json",
            }.items()
        }
        token_input = encoder.adapt_ccle_expected_counts_to_scgpt_tokens(
            sample_ids=visible.sample_ids,
            gene_ids=visible.l1_gene_ids,
            symbols=visible.l1_symbols,
            values=visible.l1_values,
            source_sha256=str(artifact_binding["sha256"]),
        )
        prepared = encoder.prepare_scgpt_input(
            token_input.payload, contract, config
        )
        result = encoder.embed_scgpt_ensemble(
            prepared, contract, config, encoder.OfficialScGPTBackend()
        )
        if result.sample_ids != split.visible_sample_ids:
            raise encoder.ScGPTInputError(
                "scGPT result differs from frozen train/validation sample order"
            )

        _verify_file_binding(artifact, artifact_binding)
        _verify_file_binding(manifest, manifest_binding)
        _verify_file_binding(split_path, split_binding)
        for name, filename in {
            "model": "best_model.pt",
            "vocab": "vocab.json",
            "args": "args.json",
        }.items():
            _verify_file_binding(model_dir / filename, model_bindings[name])
        _verify_file_binding(runtime_manifest_path, runtime_manifest)
        _verify_file_binding(python_runtime_receipt_path, python_runtime)
        if contract.inference_contract_path is not None:
            _verify_file_binding(
                contract.inference_contract_path,
                {
                    "bytes": contract.inference_contract_path.stat().st_size,
                    "sha256": contract.inference_contract_sha256,
                },
            )

        output_arrays = {
            "sample_ids": np.asarray(result.sample_ids, dtype=np.str_),
            "partition_labels": np.asarray(
                visible.partition_labels, dtype=np.str_
            ),
            "seeds": np.asarray(config.seeds, dtype=np.int64),
            "member_embeddings": result.member_embeddings,
            "mean_embedding": result.mean_embedding,
            "per_dimension_standard_deviation": (
                result.per_dimension_standard_deviation
            ),
            "sample_l2_uncertainty": result.sample_l2_uncertainty,
        }
        output_path = output / OUTPUT_FILENAME
        _write_npz_create_once(output_path, output_arrays)
        uncertainty = {
            "schema": UNCERTAINTY_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "status": "complete_uncalibrated_diagnostic",
            "seeds": list(config.seeds),
            "ensemble_member_count": len(config.seeds),
            "sample_count": len(result.sample_ids),
            "embedding_dimension": int(result.mean_embedding.shape[1]),
            "sample_schema_sha256": encoder.identifier_sequence_sha256(
                result.sample_ids
            ),
            "embedding_artifact": _file_binding(output_path),
            "configuration_sha256": encoder.canonical_json_sha256(config_payload),
            "estimands": {
                "per_dimension_standard_deviation": {
                    "ddof": 1,
                    "shape": list(result.per_dimension_standard_deviation.shape),
                    "array_sha256": _array_sha256(
                        result.per_dimension_standard_deviation
                    ),
                },
                "sample_l2_uncertainty": {
                    "definition": (
                        "root mean squared L2 distance of CREED sample-local "
                        "gene-selection and expression-binning-tie ensemble "
                        "members from the unit-normalized ensemble mean"
                    ),
                    "shape": list(result.sample_l2_uncertainty.shape),
                    "array_sha256": _array_sha256(result.sample_l2_uncertainty),
                    "median": float(np.median(result.sample_l2_uncertainty)),
                    "maximum": float(np.max(result.sample_l2_uncertainty)),
                },
            },
            "calibration_status": "uncalibrated_diagnostic_only",
            "uncertainty_scope": (
                "sample_local_gene_selection_and_binning_tie_sensitivity_not_"
                "model_epistemic_uncertainty"
            ),
            "coverage_claim_allowed": False,
            "propagated_to_downstream_predictive_distribution": False,
            "scientific_claim_authority": False,
        }
        uncertainty_path = output / "uncertainty_receipt.json"
        _write_json_create_once(uncertainty_path, uncertainty)
        partition_counts = {
            label: visible.partition_labels.count(label)
            for label in ("train", "validation")
        }
        # Close the long-run TOCTOU window before producing the final receipt.
        for path, binding in (
            (artifact, artifact_binding),
            (manifest, manifest_binding),
            (split_path, split_binding),
            (runtime_manifest_path, runtime_manifest),
            (python_runtime_receipt_path, python_runtime),
        ):
            _verify_file_binding(path, binding)
        for name, filename in {
            "model": "best_model.pt",
            "vocab": "vocab.json",
            "args": "args.json",
        }.items():
            _verify_file_binding(model_dir / filename, model_bindings[name])
        current_commit, current_dirty = _git_identity(repository)
        current_code = {
            "git_commit": current_commit,
            "git_dirty": current_dirty,
            "runner_sha256": encoder.sha256_file(Path(__file__).resolve()),
            "encoder_sha256": encoder.sha256_file(
                repository / "src" / "creed" / "science" / "scgpt_encoder.py"
            ),
            "ccle_values_sha256": encoder.sha256_file(
                repository / "src" / "creed" / "science" / "ccle_values.py"
            ),
        }
        if current_code != code:
            raise encoder.ScGPTInputError(
                "git or bound scGPT source changed during execution"
            )
        receipt = {
            "schema": RUN_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "status": "complete_development_visible_only",
            "completed_at_utc": _utc_now(),
            "input_semantics": encoder.CCLE_INPUT_SEMANTICS,
            "assay_shift": encoder.CCLE_ASSAY_SHIFT,
            "scientific_interpretation": (
                "Frozen whole-human scGPT checkpoint representation of CCLE bulk "
                "RSEM expected counts. CREED preserves the checkpoint, model "
                "construction, non-zero quantile binning, and CLS pooling semantics "
                "while intentionally replacing embed_data global MT19937/torch RNG "
                "and autocast with domain-separated per-sample PCG64 tokenization "
                "and float32 inference. The input is not relabelled as raw or "
                "single-cell counts."
            ),
            "source_bindings": {
                "frozen_split": {
                    **split_binding,
                    "payload_sha256": split.split_payload_sha256,
                    "visible_sample_ids_sha256": (
                        encoder.identifier_sequence_sha256(split.visible_sample_ids)
                    ),
                    "internal_final_sample_count": len(
                        split.internal_final_sample_ids
                    ),
                    "internal_final_values_loaded": False,
                },
                "visible_npz": artifact_binding,
                "visible_manifest": manifest_binding,
            },
            "development_scope": {
                "encoded_partitions": ["train", "validation"],
                "partition_counts": partition_counts,
                "expected_partition_counts": EXPECTED_VISIBLE_PARTITION_COUNTS,
                "expected_visible_sample_count": sum(
                    EXPECTED_VISIBLE_PARTITION_COUNTS.values()
                ),
                "sample_count": len(result.sample_ids),
                "frozen_checkpoint_parameter_fitting_performed": False,
                "validation_used_for_parameter_fitting": False,
                "validation_use": (
                    "frozen representation encoding only; no parameter fitting, "
                    "selection, early stopping, or adaptation"
                ),
                "internal_final_loaded": False,
            },
            "execution_strategy": {
                "sample_local_tokens_precomputed_before_gpu_batching": True,
                "gpu_batch_size": config.batch_size,
                "maximum_tokens_per_sample": config.max_length,
                "ensemble_members_executed_sequentially": True,
                "ensemble_axis": "sample_local_gene_selection_and_binning_tie_seed",
                "token_sampler": encoder.SAMPLE_LOCAL_TOKEN_SAMPLER,
                "value_binning": encoder.SAMPLE_LOCAL_BINNING,
                "sample_seed_schema": encoder.SAMPLE_LOCAL_SEED_SCHEMA,
                "sample_seed_inputs": [
                    "rng_domain",
                    "ensemble_seed",
                    "exact_sample_id",
                ],
                "cross_sample_rng_state_used": False,
                "gpu_batching_role": "execution_only_no_cross_sample_operator",
                "official_scgpt_model_mode": "eval",
                "automatic_oom_retry_or_batch_size_change": False,
            },
            "input": encoder.prepared_input_receipt(prepared),
            "gene_token_mapping": dict(token_input.mapping_receipt),
            "model": _model_receipt(contract, runtime_manifest),
            "configuration": config_payload,
            "configuration_sha256": encoder.canonical_json_sha256(config_payload),
            "backend": dict(result.backend_metadata),
            "python_runtime": python_runtime,
            "output": {
                **_file_binding(output_path),
                "arrays": {
                    name: {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "sha256": _array_sha256(value),
                    }
                    for name, value in output_arrays.items()
                },
                "all_embedding_and_uncertainty_values_finite": bool(
                    np.all(np.isfinite(result.member_embeddings))
                    and np.all(np.isfinite(result.mean_embedding))
                    and np.all(
                        np.isfinite(result.per_dimension_standard_deviation)
                    )
                    and np.all(np.isfinite(result.sample_l2_uncertainty))
                ),
                "mean_embedding_unit_normalized": bool(
                    np.allclose(
                        np.linalg.norm(result.mean_embedding, axis=1),
                        1.0,
                        rtol=1e-5,
                        atol=1e-6,
                    )
                ),
                "uncertainty_receipt": _file_binding(uncertainty_path),
            },
            "code": code,
            "environment": environment,
            "intent_sha256": encoder.sha256_file(output / "intent.json"),
            "automatic_retries": 0,
            "scientific_claim_authority": False,
            "requested_permission_seal": {
                "files_mode": "0444",
                "directory_mode": "0555",
                "verified_by_producer_after_receipt_write": False,
                "independent_seal_receipt_required_for_downstream_use": True,
                "physical_immutability_claimed": False,
            },
        }
        receipt_path = output / "receipt.json"
        _write_json_create_once(receipt_path, receipt)
    except BaseException as error:
        failure = {
            "schema": RUN_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "status": "failed_development_visible_only",
            "completed_at_utc": _utc_now(),
            "error_type": type(error).__name__,
            "error": _safe_failure(error),
            "sealed_or_internal_final_values_loaded": False,
            "automatic_retries": 0,
            "scientific_claim_authority": False,
        }
        _write_json_create_once(output / "failure.json", failure)
        _seal_output_permissions(output)
        print("CCLE scGPT embedding failed closed", file=sys.stderr)
        return 2 if isinstance(
            error, (ccle_values.CCLEValueCustodyError, encoder.ScGPTInputError)
        ) else 1

    _seal_output_permissions(output)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": str(output_path),
                "output_sha256": receipt["output"]["sha256"],
                "receipt": str(receipt_path),
                "receipt_sha256": encoder.sha256_file(receipt_path),
                "internal_final_loaded": False,
                "scientific_claim_authority": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

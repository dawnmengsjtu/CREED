"""Source-native, partially observed CPTAC blocks for development research.

This loader does not normalize, impute, merge duplicate genes or declare new
biological layers. Clinical outcomes never enter the returned predictors.
"""
from __future__ import annotations

import csv
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class CPTACSchemaError(ValueError):
    pass


COHORT_RNA = {
    "brca_cptac_2020": "data_mrna_seq_fpkm.txt",
    "gbm_cptac_2021": "data_mrna_seq_fpkm.txt",
    "luad_cptac_2020": "data_mrna_seq_rpkm.txt",
    "rcc_cptac_gdc": "data_mrna_seq_tpm.txt",
    "ucec_cptac_2020": "data_mrna_seq_rsem.txt",
}
MODALITIES = {
    "rna": (1, None),
    "protein": (2, "data_protein_quantification.txt"),
    "phosphoprotein": (2, "data_phosphoprotein_quantification.txt"),
    "metabolite": (0, "data_metabolome_quantification.txt"),
    "lipid_negative": (0, "data_lipidome_negative_quantification.txt"),
    "lipid_positive": (0, "data_lipidome_positive_quantification.txt"),
}
ANNOTATIONS = {"Entrez_Gene_Id", "NAME", "DESCRIPTION", "DETAILS", "GENE_SYMBOL", "PHOSPHOSITES"}
MISSING = {"", "NA", "N/A", "NAN", "NULL", "[NOT AVAILABLE]"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_meta(path: Path) -> dict[str, str]:
    meta = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip() and not line.startswith("#"):
            if ":" not in line:
                raise CPTACSchemaError(f"invalid meta line: {path.name}")
            key, value = line.split(":", 1)
            if key.strip() in meta:
                raise CPTACSchemaError(f"duplicate metadata key: {key}")
            meta[key.strip()] = value.strip()
    return meta


def read_patients(path: Path) -> dict[str, str]:
    """Retain only identity columns; no outcome or expression-derived covariates."""
    pairs = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = csv.reader((line for line in f if not line.startswith("#")), delimiter="\t")
        header = next(rows)
        if len(set(header)) != len(header):
            raise CPTACSchemaError("duplicate clinical header")
        s, p = header.index("SAMPLE_ID"), header.index("PATIENT_ID")
        for row in rows:
            if not row:
                continue
            if len(row) != len(header):
                raise CPTACSchemaError("ragged clinical row")
            sample, patient = row[s].strip(), row[p].strip()
            if not sample or not patient or sample in pairs:
                raise CPTACSchemaError("empty or duplicated clinical sample identity")
            pairs[sample] = patient
    return pairs


def parse_value(token: str) -> tuple[float, int]:
    token = token.strip()
    if token.upper() in MISSING:
        return float("nan"), 0
    censor = -1 if token.startswith("<") else (1 if token.startswith(">") else 0)
    numeric = token.lstrip("<>= ") if censor else token
    try:
        value = float(numeric)
    except ValueError as exc:
        raise CPTACSchemaError(f"unrecognized assay value {token!r}") from exc
    if not np.isfinite(value):
        raise CPTACSchemaError(f"nonfinite value not an explicit missing token: {token!r}")
    return value, censor


@dataclass
class AssayBlock:
    name: str
    layer: int
    samples: tuple[str, ...]
    feature_ids: tuple[str, ...]
    values: np.ndarray
    censor: np.ndarray
    source: dict

    @property
    def observed(self) -> np.ndarray:
        return np.isfinite(self.values) & (self.censor == 0)

    def profile(self) -> dict:
        seen = self.observed
        counts = Counter(self.feature_ids)
        finite = self.values[seen]
        return {
            "layer": self.layer, "shape_samples_features": list(self.values.shape),
            "exact_observed_values": int(seen.sum()),
            "missing_values": int(np.isnan(self.values).sum()),
            "censored_values": int((self.censor != 0).sum()),
            "exact_observed_fraction": float(seen.mean()),
            "all_missing_samples": int((~seen.any(axis=1)).sum()),
            "all_missing_features": int((~seen.any(axis=0)).sum()),
            "empty_feature_ids": sum(not k or k.upper() in MISSING for k in self.feature_ids),
            "duplicate_feature_ids": sum(v > 1 for v in counts.values()),
            "duplicate_extra_rows": sum(v - 1 for v in counts.values()),
            "exact_value_quantiles": np.quantile(finite, [0, .25, .5, .75, 1]).tolist() if finite.size else None,
            "negative_exact_values": int((finite < 0).sum()),
            "source": self.source,
        }


def read_assay(path: Path, name: str, layer: int, patients: dict[str, str]) -> AssayBlock:
    if path.is_symlink() or "zscores" in path.name or "reconstructed" in path.name:
        raise CPTACSchemaError("only explicit original regular assay tables are accepted")
    meta_path = path.with_name("meta_" + path.name.removeprefix("data_"))
    meta = read_meta(meta_path)
    if meta.get("data_filename") != path.name:
        raise CPTACSchemaError("source metadata does not name the assay file")
    before_hash = digest(path)
    features, values, censor_rows = [], [], []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        if len(set(header)) != len(header):
            raise CPTACSchemaError("duplicate assay columns")
        unknown = set(header[1:]) - ANNOTATIONS - set(patients)
        if unknown:
            raise CPTACSchemaError(f"unregistered assay columns: {sorted(unknown)[:5]}")
        sample_indices = [i for i, col in enumerate(header) if i and col in patients]
        if not sample_indices:
            raise CPTACSchemaError("assay has no registered samples")
        for row in reader:
            if not row:
                continue
            if len(row) != len(header):
                raise CPTACSchemaError(f"ragged assay row in {path.name}")
            parsed = [parse_value(row[i]) for i in sample_indices]
            features.append(row[0].strip())
            values.append([x for x, _ in parsed])
            censor_rows.append([c for _, c in parsed])
    if not features:
        raise CPTACSchemaError("empty assay")
    if digest(path) != before_hash:
        raise CPTACSchemaError("source changed while loading")
    return AssayBlock(name, layer, tuple(header[i] for i in sample_indices), tuple(features),
                      np.asarray(values, dtype=np.float64).T,
                      np.asarray(censor_rows, dtype=np.int8).T,
                      {"path": str(path), "sha256": before_hash,
                       "bytes": path.stat().st_size, "metadata_sha256": digest(meta_path),
                       "metadata": meta, "transform_applied": "none_source_native",
                       "feature_key": header[0],
                       "training_semantics_status": "requires_assay_specific_review"})


def align_blocks(blocks: dict[str, AssayBlock], patients: dict[str, str]) -> tuple[tuple[str, ...], dict]:
    """Union alignment: absent assays are NaN/false, never fake zero measurements."""
    samples = tuple(sorted(set().union(*(set(b.samples) for b in blocks.values()))))
    index = {s: i for i, s in enumerate(samples)}
    aligned = {}
    for name, b in blocks.items():
        values = np.full((len(samples), len(b.feature_ids)), np.nan)
        censor = np.zeros(values.shape, dtype=np.int8)
        present = np.zeros(len(samples), dtype=bool)
        rows = [index[s] for s in b.samples]
        values[rows], censor[rows], present[rows] = b.values, b.censor, True
        aligned[name] = {"values": values, "observed": np.isfinite(values) & (censor == 0),
                         "censor": censor, "assay_present": present,
                         "feature_ids": np.asarray(b.feature_ids, dtype=str)}
    if any(s not in patients for s in samples):
        raise CPTACSchemaError("sample has no patient")
    return samples, aligned


def load_cohort(root: Path) -> tuple[dict, dict]:
    if root.name not in COHORT_RNA:
        raise CPTACSchemaError("cohort not in explicit development allowlist")
    clinical = root / "data_clinical_sample.txt"
    clinical_hash = digest(clinical)
    patients = read_patients(clinical)
    if digest(clinical) != clinical_hash:
        raise CPTACSchemaError("clinical source changed while loading")
    blocks = {}
    absent = []
    for name, (layer, filename) in MODALITIES.items():
        path = root / (filename or COHORT_RNA[root.name])
        if not path.exists():
            absent.append(name)
            continue
        blocks[name] = read_assay(path, name, layer, patients)
    if not blocks:
        raise CPTACSchemaError("no assays")
    samples, arrays = align_blocks(blocks, patients)
    present = {name: {s for s, ok in zip(samples, a["observed"].any(axis=1)) if ok}
               for name, a in arrays.items()}
    patterns = Counter("+".join(name for name in blocks if s in present[name]) or "none" for s in samples)
    all_modality = set.intersection(*present.values())
    patient_ids = [patients[s] for s in samples]
    profile = {
        "cohort": root.name, "status": "development_only_not_a_new_blind_test",
        "sample_union": len(samples), "patients_in_union": len(set(patient_ids)),
        "multiple_sample_patients": sum(v > 1 for v in Counter(patient_ids).values()),
        "clinical_rows": len(patients), "clinical_sha256": clinical_hash,
        "all_present_modality_intersection": len(all_modality),
        "observed_modality_patterns": dict(sorted(patterns.items())), "absent_modalities": absent,
        "pairwise_sample_overlap": {a: {b: len(present[a] & present[b]) for b in blocks} for a in blocks},
        "assays": {name: block.profile() for name, block in blocks.items()},
        "license_file_exists": (root / "LICENSE").is_file(),
        "license_status": "source_terms_review_required_before_release",
        "no_imputation_no_scaling_no_target_feature_selection": True,
    }
    bundle = {"sample_ids": np.asarray(samples, dtype=str),
              "patient_ids": np.asarray(patient_ids, dtype=str),
              "cohort": np.asarray(root.name)}
    for name, a in arrays.items():
        bundle.update({name + "__" + key: val for key, val in a.items()})
    return profile, bundle

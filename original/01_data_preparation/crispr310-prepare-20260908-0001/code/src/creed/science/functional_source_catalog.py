"""Outcome-blind identity catalog for existing CTRP and DepMap development sources.

No outcome conversion, statistics, or model selection belongs in this module.
Line scanners inspect identifiers, delimiters and line boundaries only. They
physically traverse source bytes, but never decode numeric response fields.
Only exact public Sanger IDs and collision-free, explicit model aliases link
sources. These are development identities, never independent confirmation.
"""
from collections import Counter, defaultdict
import csv
import re


def alias_key(value):
    # Conservative punctuation normalization, not edit-distance/fuzzy matching.
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def index_models(rows, safe_ids):
    safe_ids = list(safe_ids)
    if len(safe_ids) != len(set(safe_ids)) or not safe_ids:
        raise ValueError("safe identity list must be nonempty and unique")
    if any(not re.fullmatch(r"SIDM\d+", s) for s in safe_ids):
        raise ValueError("only exact Sanger model IDs are accepted")
    rows = list(rows)
    mids = Counter(r["ModelID"] for r in rows)
    sids = Counter(r["SangerModelID"] for r in rows if r["SangerModelID"])
    aliases = defaultdict(set)
    mapped = {}
    for row in rows:
        mid, sid = row["ModelID"], row["SangerModelID"]
        for field in ("CellLineName", "StrippedCellLineName"):
            key = alias_key(row[field])
            if key:
                aliases[key].add(mid)
        if sid in safe_ids and mids[mid] == 1 and sids[sid] == 1:
            mapped[mid] = {"sanger_id": sid, "patient_id": row["PatientID"],
                           "lineage": row["OncotreeLineage"],
                           "cell_line_name": row["CellLineName"]}
    return mapped, aliases, {
        "model_metadata_rows": len(rows), "safe_ids_requested": len(safe_ids),
        "exact_unambiguous_safe_models": len(mapped),
        "safe_ids_without_unique_model": sorted(set(safe_ids) - {r["sanger_id"] for r in mapped.values()}),
        "ambiguous_alias_keys": sum(len(v) > 1 for v in aliases.values()),
    }


def index_ctrp(cell_rows, experiment_rows, mapped_models, aliases):
    cells, experiments = list(cell_rows), list(experiment_rows)
    cell_counts = Counter(r["master_ccl_id"] for r in cells)
    exp_counts = Counter(r["experiment_id"] for r in experiments)
    cell_map = {}
    reasons = Counter()
    for row in cells:
        key = row["master_ccl_id"]
        candidates = aliases.get(alias_key(row["ccl_name"]), set())
        if cell_counts[key] != 1:
            reasons["duplicate_cell_key"] += 1
        elif len(candidates) != 1:
            reasons["absent_or_ambiguous_alias"] += 1
        else:
            mid = next(iter(candidates))
            if mid in mapped_models:
                cell_map[key] = {"model_id": mid, **mapped_models[mid]}
            else:
                reasons["not_safe_development_model"] += 1
    exp_map = {}
    for row in experiments:
        if row["master_ccl_id"] in cell_map and exp_counts[row["experiment_id"]] == 1:
            exp_map[row["experiment_id"]] = {
                **cell_map[row["master_ccl_id"]], "master_ccl_id": row["master_ccl_id"],
                "snp_fp_status": row["snp_fp_status"], "culture_media": row["culture_media"],
                "experiment_date": row["experiment_date"],
            }
    return cell_map, exp_map, {
        "ctrp_cell_metadata_rows": len(cells), "ctrp_experiment_metadata_rows": len(experiments),
        "safe_ctrp_cell_keys": len(cell_map), "safe_ctrp_experiments": len(exp_map),
        "safe_ctrp_distinct_models": len({r["model_id"] for r in exp_map.values()}),
        "excluded_cell_reasons": dict(reasons),
        "fingerprint_status_counts": dict(Counter(r["snp_fp_status"] for r in exp_map.values())),
    }


def scan_effect_identifiers(stream, mapped_models):
    header = next(csv.reader([stream.readline().decode("utf-8-sig")]))
    if header[0] not in ("", "ModelID") or len(header) < 2:
        raise ValueError("unexpected CRISPR source header")
    genes = header[1:]
    if len(set(genes)) != len(genes) or any(not re.fullmatch(r".+ \(\d+\)", g) for g in genes):
        raise ValueError("invalid or duplicate CRISPR gene identity")
    counts = Counter()
    total = 0
    for line in stream:
        if not line.strip():
            raise ValueError("blank CRISPR source row")
        prefix, separator, _ = line.partition(b",")
        if not separator:
            raise ValueError("missing CRISPR identity delimiter")
        mid = prefix.decode("ascii").strip('"')
        if not re.fullmatch(r"ACH-\d+", mid):
            raise ValueError("unexpected CRISPR model identity")
        total += 1
        if mid in mapped_models:
            counts[mid] += 1
    unique = sorted(mid for mid, n in counts.items() if n == 1)
    return unique, {"crispr_source_rows": total, "crispr_gene_columns": len(genes),
                    "safe_crispr_unique_models": len(unique),
                    "safe_crispr_duplicate_models_excluded": sum(n != 1 for n in counts.values()),
                    "response_values_decoded": 0}


def scan_ctrp_identifiers(stream, exp_map):
    header = stream.readline().decode("utf-8-sig").rstrip("\r\n").split("\t")
    if len(header) != 17 or header[0] != "experiment_id" or header[-1] != "master_cpd_id":
        raise ValueError("unexpected CTRP response header")
    total = selected = 0
    pairs = Counter()
    compounds = set()
    for line in stream:
        prefix, separator, rest = line.partition(b"\t")
        if not separator:
            raise ValueError("missing CTRP identity delimiter")
        experiment = prefix.decode("ascii")
        total += 1
        if experiment not in exp_map:
            continue
        if line.count(b"\t") != 16:
            raise ValueError("unexpected selected CTRP row width")
        compound = rest.rsplit(b"\t", 1)[-1].strip().decode("ascii")
        if not compound.isdigit():
            raise ValueError("invalid CTRP compound identity")
        selected += 1
        compounds.add(compound)
        pairs[(exp_map[experiment]["sanger_id"], compound)] += 1
    return {"ctrp_source_rows": total, "safe_ctrp_curve_rows": selected,
            "safe_ctrp_distinct_pairs": len(pairs), "safe_ctrp_compounds": len(compounds),
            "safe_ctrp_samples_with_curve": len({sid for sid, _ in pairs}),
            "pairs_with_multiple_experimental_rows": sum(n > 1 for n in pairs.values()),
            "response_values_decoded": 0}

"""Explicit source-scale handling and static ESM identity projection.

Log-scale compatibility is NOT common-reference harmonization. No target-cohort
centering, target imputation, gene-alias heuristics or invented FM rows.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import sparse

COHORTS = ["brca_cptac_2020", "luad_cptac_2020", "gbm_cptac_2021", "ucec_cptac_2020"]
README_HASHES = {
    COHORTS[0]: "7c47bd9daaa989cb508ab584f87cbc06285c07d82051369c81a9a0e3c46f626a",
    COHORTS[1]: "978d9cdc260bc2bd8b9084f1bec0ea5723d8a490bc9140b20ef784e04a64bcdb",
    COHORTS[2]: "d358de7d9cf310944ed7d73db4ee5562eeb53fc9b3adda865d8aef3f8c364918",
    COHORTS[3]: "6cdd067cf45875214e6afedb047acd058e9fd99c700b67fded0f384f7cf7a4a6",
}
SCALES = {
    COHORTS[0]: "source log2 FPKM unchanged; source zeros already missing",
    COHORTS[1]: "source log2 upper-quartile normalized RPKM unchanged",
    COHORTS[2]: "source positive linear FPKM-UQ converted once by log2(x); no fitted centering",
    COHORTS[3]: "source log2(upper-quantile normalized RSEM + 1) unchanged",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def rna_log_scale(values, cohort):
    if cohort not in SCALES:
        raise ValueError("unreviewed RNA scale")
    result = np.array(values, dtype=float, copy=True)
    if cohort == "gbm_cptac_2021":
        if np.any(result[np.isfinite(result)] <= 0):
            raise ValueError("GBM nonpositive observed RNA requires explicit handling")
        result = np.log2(result)
    return result


def project_esm(esm_root, hgnc, wanted):
    root = Path(esm_root)
    if sha(root / "receipt.json") != "df42c8c3f9eac2734b7851d3af7f39231af650853c7607aed08267f3e1588c75":
        raise ValueError("unrecognized ESM source receipt")
    receipt = json.loads((root / "receipt.json").read_text())
    for name, expected in receipt["outputs"].items():
        assert sha(root / name) == expected["sha256"]
        assert (root / name).stat().st_size == expected["bytes"]
    assert sha(hgnc) == "e73e9259177884b5994fc81ed733c1b3d4df34c84290bc9dddc86e960d5d6419"
    values = np.load(root / "esm2_embeddings.npy", allow_pickle=False)
    present = np.load(root / "present_mask.npy", allow_pickle=False)
    accessions = json.loads((root / "uniprot_accessions.json").read_text())
    assert values.shape == (6692, 1280) and present.shape == (6692,)
    assert int(present.sum()) == 6691 and np.isfinite(values[present]).all()
    assert np.isnan(values[~present]).all() and len(set(accessions)) == 6692
    accession_index = {a: i for i, a in enumerate(accessions) if present[i]}
    gene_indices, ambiguous = {}, []
    with open(hgnc, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["status"] != "Approved" or row["symbol"] not in wanted:
                continue
            indices = sorted({accession_index[a] for a in row["uniprot_ids"].split("|") if a in accession_index})
            if len(indices) == 1:
                gene_indices[row["symbol"]] = indices[0]
            elif len(indices) > 1:
                ambiguous.append(row["symbol"])
    # One accession assigned to multiple approved genes is also ambiguous.
    from collections import Counter
    counts = Counter(gene_indices.values())
    gene_indices = {g: i for g, i in gene_indices.items() if counts[i] == 1}
    genes = np.array(sorted(gene_indices))
    indices = np.array([gene_indices[g] for g in genes])
    projected = values[indices]
    assert np.isfinite(projected).all() and np.all(np.linalg.norm(projected, axis=1) > 0)
    return genes, projected, indices, {
        "source_receipt_sha256": sha(root / "receipt.json"), "source_outputs": receipt["outputs"],
        "source_present": 6691, "source_total": 6692, "projected_genes": len(genes),
        "multi_available_accession_genes_excluded": len(ambiguous), "loaded_model": receipt["loaded_model"],
        "semantics": "static sequence identity, not patient state; absent genes have no FM edges",
        "mapping": "exact approved HGNC symbols and unique available accessions; no aliases or isoform collapsing",
    }


def esm_peer_graph(source_ids, target_ids, esm_genes, embeddings, k=16, permutation=None):
    """Positive cosine top-k source neighbors; exact self gene always excluded."""
    lookup = {g: i for i, g in enumerate(esm_genes)}
    source_index = np.array([i for i, g in enumerate(source_ids) if g in lookup], dtype=int)
    target_index = np.array([i for i, g in enumerate(target_ids) if g in lookup], dtype=int)
    if not len(source_index) or not len(target_index):
        return sparse.csr_matrix((len(source_ids), len(target_ids)))
    unit = embeddings.astype(np.float64)
    unit = unit / np.linalg.norm(unit, axis=1, keepdims=True)
    sx = unit[[lookup[source_ids[i]] for i in source_index]]
    ty = unit[[lookup[target_ids[i]] for i in target_index]]
    rows, cols, weights = [], [], []
    for start in range(0, len(target_index), 256):
        target_part = target_index[start:start+256]
        similarity = ty[start:start+256] @ sx.T
        similarity[np.asarray(target_ids)[target_part, None] == np.asarray(source_ids)[source_index][None, :]] = -np.inf
        take = min(k, max(0, len(source_index)-1))
        if take == 0:
            continue
        nearest = np.argsort(-similarity, axis=1, kind="stable")[:, :take]
        score = np.take_along_axis(similarity, nearest, axis=1)
        rr = source_index[nearest]
        cc = np.broadcast_to(target_part[:, None], rr.shape)
        if permutation is not None:
            rr = np.asarray(permutation)[rr]
        ok = (score > 0) & (np.asarray(source_ids)[rr] != np.asarray(target_ids)[cc])
        rows.extend(rr[ok]); cols.extend(cc[ok]); weights.extend(score[ok])
    return sparse.csr_matrix((weights, (rows, cols)), shape=(len(source_ids), len(target_ids)))

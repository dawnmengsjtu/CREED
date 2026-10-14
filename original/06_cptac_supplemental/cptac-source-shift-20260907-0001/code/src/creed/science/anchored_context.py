"""Development candidate: cross-fitted identity anchors and typed pathway peers.

Static annotations are predictive inputs, never independently measured pathway
truth. No foundation model, language agent, alias inference or random fallback.
"""
import csv
from collections import defaultdict
import numpy as np
from scipy import sparse


def fit_anchor(x, y, source_ids, target_ids, alpha=.1):
    mapping = {str(g): i for i, g in enumerate(source_ids)}
    index = np.array([mapping.get(str(g), -1) for g in target_ids])
    xx = x[:, np.maximum(index, 0)].copy()
    xx[:, index < 0] = np.nan
    ok = np.isfinite(xx) & np.isfinite(y)
    n = ok.sum(0)
    xm = np.where(ok, xx, 0.).sum(0) / np.maximum(n, 1)
    ym = np.where(ok, y, 0.).sum(0) / np.maximum(n, 1)
    dx, dy = np.where(ok, xx - xm, 0.), np.where(ok, y - ym, 0.)
    var = (dx * dx).sum(0) / np.maximum(n, 1)
    supported = (index >= 0) & (n >= 5) & (var >= 1e-8)
    slope = np.zeros(y.shape[1])
    slope[supported] = (dx * dy).sum(0)[supported] / n[supported] / var[supported] / (1 + alpha)
    ym[~supported] = np.nanmean(y, axis=0)[~supported]
    return index, xm, ym, slope, supported


def predict_anchor(x, state):
    index, xm, ym, slope, _ = state
    xx = x[:, np.maximum(index, 0)]
    return ym + np.where(np.isfinite(xx), xx - xm, 0.) * slope


def gene_pathways(hgnc, reactome):
    """Exact approved symbols and exact accession join; retain positive edges only."""
    accession_genes = defaultdict(set)
    with open(hgnc, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["status"] == "Approved":
                for acc in row["uniprot_ids"].split("|"):
                    if acc:
                        accession_genes[acc].add(row["symbol"])
    result = defaultdict(set)
    with open(reactome, newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) != 6:
                raise ValueError("invalid Reactome row")
            if row[5] != "Homo sapiens":
                continue
            # Ambiguous accession-to-gene joins are excluded, not replicated.
            genes = accession_genes.get(row[0], set())
            if len(genes) == 1:
                result[next(iter(genes))].add(row[1])
    return dict(result)


def peer_graph(source_ids, target_ids, annotations, permutation=None):
    """Count shared lowest-level pathways, excluding the target's own gene."""
    paths = sorted(set().union(*annotations.values()))
    lookup = {p: i for i, p in enumerate(paths)}
    def membership(ids):
        pairs = [(i, lookup[p]) for i, g in enumerate(ids) for p in annotations.get(str(g), ())]
        if not pairs:
            return sparse.csr_matrix((len(ids), len(paths)))
        rows, cols = zip(*pairs)
        return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ids), len(paths)))
    weights = (membership(source_ids) @ membership(target_ids).T).tocoo()
    rows = weights.row if permutation is None else np.asarray(permutation)[weights.row]
    keep = np.asarray(source_ids)[rows] != np.asarray(target_ids)[weights.col]
    graph = sparse.csr_matrix((weights.data[keep], (rows[keep], weights.col[keep])),
                              shape=(len(source_ids), len(target_ids)))
    return graph


def pathway_features(x_train, x_test, graph):
    """Training-only scaling/eligibility; weighted measured peers, never targets."""
    n = np.isfinite(x_train).sum(0)
    mean = np.nansum(x_train, axis=0) / np.maximum(n, 1)
    d = np.where(np.isfinite(x_train), x_train - mean, 0.)
    sd = np.sqrt((d * d).sum(0) / np.maximum(n, 1))
    valid = (n >= max(5, .8 * len(x_train))) & (sd > 1e-8)
    graph = graph.multiply(valid[:, None]).tocsr()
    graph.eliminate_zeros()
    supported = np.asarray((graph > 0).sum(0)).ravel() >= 3
    graph = graph.multiply(supported[None, :]).tocsr()
    def project(x):
        observed = np.isfinite(x)
        z = np.where(observed, x - mean, 0.) / np.maximum(sd, 1e-8)
        numerator = z @ graph
        denominator = observed.astype(float) @ graph
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    train, test = project(x_train), project(x_test)
    center, scale = train.mean(0), train.std(0)
    scale = np.where(scale > 1e-8, scale, 1.)
    return (train - center) / scale, (test - center) / scale, supported


def context_predict(ztrain, residual, ztest, local_train=None, local_test=None, alpha=.1):
    """Shared sample context plus optional output-specific pathway context."""
    a = np.column_stack((np.ones(len(ztrain)), ztrain))
    b = np.column_stack((np.ones(len(ztest)), ztest))
    predictions = []
    for start in range(0, residual.shape[1], 128):
        y = residual[:, start:start + 128]
        aa = np.broadcast_to(a, (y.shape[1], *a.shape))
        bb = np.broadcast_to(b, (y.shape[1], *b.shape))
        if local_train is not None:
            lt = local_train[:, start:start + 128]
            le = local_test[:, start:start + 128]
            if lt.ndim == 2:
                lt, le = lt[..., None], le[..., None]
            aa = np.concatenate((aa, lt.transpose(1, 0, 2)), axis=2)
            bb = np.concatenate((bb, le.transpose(1, 0, 2)), axis=2)
        mask = np.isfinite(y).T
        count = mask.sum(1)
        if np.any(count < 2):
            raise ValueError("insufficient residual targets")
        gram = np.einsum("kni,knj,kn->kij", aa, aa, mask, optimize=True) / count[:, None, None]
        rhs = np.einsum("kni,kn->ki", aa, np.where(mask, y.T, 0.), optimize=True) / count[:, None]
        penalty = np.eye(aa.shape[2]) * alpha
        penalty[0, 0] = 0.
        coef = np.linalg.solve(gram + penalty, rhs[..., None])[..., 0]
        predictions.append(np.einsum("kni,ki->nk", bb, coef))
    return np.concatenate(predictions, axis=1)

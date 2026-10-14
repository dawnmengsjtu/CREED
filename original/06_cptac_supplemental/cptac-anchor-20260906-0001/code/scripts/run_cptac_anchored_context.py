"""Fixed development experiment on existing CPTAC splits; never a blind test."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from creed.science.masked_baseline import unique_feature_map, source_gene_labels, fit_pca, transform_pca, metrics
from creed.science.anchored_context import fit_anchor, predict_anchor, gene_pathways, peer_graph, pathway_features, context_predict


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ("data", "baseline", "hgnc", "reactome", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    base = json.loads((args.baseline / "summary.json").read_text())
    config = dict(base["config"])
    config.update(status="development_identity_context_pathway_candidate_not_FM_or_agent_efficacy",
                  inner_folds=5, knowledge="exact approved HGNC symbol to unambiguous accession to Reactome V97 human lowest-level positive membership",
                  pathway_feature="weighted non-self source-gene peers; at least 3 train-eligible peers; weights count shared pathways",
                  negative_control="fixed source-node relabeling, then self-edge exclusion; not perfectly degree-preserving after exclusion",
                  selection="fixed first candidate after inspecting prior baseline; all cohorts development; no claim of unbiased external validation")
    config["input_hashes"] = {"hgnc": sha(args.hgnc), "reactome": sha(args.reactome), "baseline_summary": sha(args.baseline / "summary.json")}
    assert config["input_hashes"]["hgnc"] == "e73e9259177884b5994fc81ed733c1b3d4df34c84290bc9dddc86e960d5d6419"
    assert config["input_hashes"]["reactome"] == "e34fe56a5129d88d8cd67300c279d7e81fa1d3bf86fa0577cf91032c326eaecc"
    assert config["input_hashes"]["baseline_summary"] == "156e35f79d2444b10953abfcb674fee453afdf5485d28a31b472f1a15e77253e"
    config["code_hashes"] = {p.name: sha(p) for p in [Path(__file__), Path(__file__).resolve().parents[1] / "src/creed/science/anchored_context.py", Path(__file__).resolve().parents[1] / "src/creed/science/masked_baseline.py"]}
    # Prior module/script fingerprints remain explicitly baseline provenance only.
    config["baseline_code_sha256"] = config.pop("code_sha256")
    config["baseline_module_sha256"] = config.pop("module_sha256")
    (args.output / "intent.json").write_text(json.dumps(config, indent=2))
    annotations = gene_pathways(args.hgnc, args.reactome)
    manifest = json.loads((args.data / "summary.json").read_text())
    expected = {c["cohort"]: c["bundle_sha256"] for c in manifest["cohorts"]}
    bundles = []
    for c in config["cohorts"]:
        path = args.data / (c + ".npz")
        assert sha(path) == expected[c]
        with np.load(path, allow_pickle=False) as b:
            bundles.append({k: b[k] for k in b.files if k.startswith(("rna__", "protein__")) or k in ("patient_ids", "sample_ids")})
    arrays, ids = {}, {}
    for mod in ("rna", "protein"):
        maps = [unique_feature_map(source_gene_labels(b[mod + "__feature_ids"], c, mod)) for b, c in zip(bundles, config["cohorts"])]
        ids[mod] = np.array(sorted(set(maps[0]) & set(maps[1])))
        arrays[mod] = np.concatenate([np.where(b[mod + "__observed"], b[mod + "__values"], np.nan)[:, [m[g] for g in ids[mod]]] for b, m in zip(bundles, maps)])
    patients = np.concatenate([b["patient_ids"] for b in bundles])
    assert len(set(patients)) == len(patients)
    cohorts = np.concatenate([np.full(len(b["patient_ids"]), i) for i, b in enumerate(bundles)])
    graphs = {}
    for source, target in (("rna", "protein"), ("protein", "rna")):
        permutation = np.random.default_rng(config["seed"]).permutation(len(ids[source]))
        graphs[source] = (peer_graph(ids[source], ids[target], annotations), peer_graph(ids[source], ids[target], annotations, permutation))
    results = []
    for prior in base["records"]:
        key = prior["direction"] + "__" + prior["split"]
        source, target = prior["direction"].split("_to_")
        with np.load(args.baseline / (key + ".npz"), allow_pickle=False) as old:
            saved = {k: old[k] for k in old.files}
        train, test = saved["train_indices"], saved["test_indices"]
        assert not set(patients[train]) & set(patients[test])
        lookup = {g: j for j, g in enumerate(ids[target])}
        panel = np.array([lookup[g] for g in saved["target_ids"]])
        xtrain, xtest = arrays[source][train], arrays[source][test]
        yt = (arrays[target][train][:, panel] - saved["target_mean"]) / saved["target_scale"]
        np.testing.assert_allclose(saved["truth"], (arrays[target][test][:, panel] - saved["target_mean"]) / saved["target_scale"], atol=1e-12)
        state = fit_anchor(xtrain, yt, ids[source], saved["target_ids"])
        anchor = predict_anchor(xtest, state)
        np.testing.assert_allclose(anchor, saved["same_gene"], atol=1e-10)
        oof = np.full_like(yt, np.nan)
        inner_ids = np.full(len(train), -1)
        inner = StratifiedKFold(config["inner_folds"], shuffle=True, random_state=config["seed"])
        for fold, (itr, ite) in enumerate(inner.split(train, cohorts[train])):
            inner_state = fit_anchor(xtrain[itr], yt[itr], ids[source], saved["target_ids"])
            oof[ite] = predict_anchor(xtrain[ite], inner_state)
            inner_ids[ite] = fold
        assert np.isfinite(oof).all() and (inner_ids >= 0).all()
        ztrain, pca = fit_pca(xtrain, config["max_predictors"], config["rank"])
        ztest = transform_pca(xtest, pca)
        residual = yt - oof
        predictions = {a: saved[a] for a in prior["arms"]}
        predictions["anchor_context"] = anchor + context_predict(ztrain, residual, ztest)
        coverage = {}
        for name, graph in zip(("anchor_context_pathway", "anchor_context_shuffled_pathway"), graphs[source]):
            local_train, local_test, supported = pathway_features(xtrain, xtest, graph[:, panel])
            predictions[name] = anchor + context_predict(ztrain, residual, ztest, local_train, local_test)
            coverage[name] = int(supported.sum())
            saved[name + "__supported"] = supported
        record = dict(prior)
        record["pathway_supported_targets"] = coverage
        record["arms"] = {a: metrics(saved["truth"], v) for a, v in predictions.items()}
        saved.update(predictions)
        saved["anchor_oof_train"] = oof
        saved["inner_fold_ids"] = inner_ids
        np.savez_compressed(args.output / (key + ".npz"), **saved)
        results.append(record)
        print(json.dumps(record), flush=True)
    (args.output / "summary.json").write_text(json.dumps({"config": config, "records": results, "n_patients": len(patients), "annotation_genes": len(annotations), "caveat": "Development only. Pathway scores inferred from input are not a measured extra layer. No FM or dynamic agent efficacy demonstrated."}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        print(traceback.format_exc(), flush=True)
        raise

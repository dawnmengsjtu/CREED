"""Eight fixed arms: metabolite-only query, virtual RNA/FM, and measured diagnostics."""
import argparse
from pathlib import Path
import time
import numpy as np

from creed.science import metabolite_query as bridge
from creed.science import virtual_state_task as task
from creed.science import masked_task_kernel as masked
from run_phospho_fewshot import sha, save_json, save_npz, read_json

BINDINGS_SHA = 'b3e231b8065623770047d3f21653eceb867a06fb9a7af1977c21a1eca2ebba3f'
N_SAMPLES = 421
ARMS = ('m_only', 'virtual_rna', 'virtual_fm', 'ordinary_m', 'shuffle_rna', 'shuffle_fm',
        'measured_rna', 'measured_shuffle_rna')
VIRTUAL_ARMS = ARMS[1:6]
ALPHAS = (.1, 1., 10., 100.)
MARGIN = 1e-10
SOURCE_FILES = {
    'data': ('manifest.json', 'metabolites.npy', 'metabolite_ids.npy', 'sample_ids.npy', 'scgpt.npy'),
    'rna': ('manifest.json', 'rna_log_cpm.npy', 'gene_ids.npy', 'sample_ids.npy'),
    'gdsc': ('manifest.json', 'auc.npy', 'drug_ids.npy', 'sample_ids.npy'),
    **{a: ('splits.json', 'result.json', *[n for f in range(5) for n in
        (f'selection-{f}.json', f'predictions-{f}.npz')]) for a in ('reference', 'measured-reference')},
}


def arrays(path, keys=None):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in (z.files if keys is None else keys)}


class BoundSources:
    def __init__(self, roots, binding):
        self.roots = {k: Path(v).resolve() for k, v in roots.items()}
        self.binding_path = Path(binding).resolve()
        if sha(self.binding_path) != BINDINGS_SHA:
            raise ValueError('exact pre-run source bindings required')
        self.document = read_json(self.binding_path)
        expected = {a + '/' + f for a, names in SOURCE_FILES.items() for f in names}
        if set(self.document['files']) != expected or set(roots) != set(SOURCE_FILES):
            raise ValueError('exact source object whitelist required')
        self.loaded = {}
        self.finish()

    def path(self, relative):
        if relative not in self.document['files']:
            raise ValueError('unlisted source object')
        alias, filename = relative.split('/', 1)
        path = self.roots[alias] / filename
        if path.is_symlink() or path.resolve().parent != self.roots[alias]:
            raise ValueError('source escapes bound root')
        return path

    def json(self, relative):
        self.loaded[relative] = 'json'
        return read_json(self.path(relative))

    def npy(self, relative):
        self.loaded[relative] = 'npy'
        return np.load(self.path(relative), allow_pickle=False)

    def npz(self, relative, fields):
        allowed = {'fit', 'held', 'panel', 'drug_ids', 'train_mean', 'train_sd'}
        if relative.startswith('measured-reference/'):
            allowed.add('metabolite_only')
        if not set(fields) <= allowed:
            raise ValueError('parent truth or unsupported prediction field requested')
        self.loaded[relative] = list(fields)
        return arrays(self.path(relative), fields)

    def finish(self):
        if sha(self.binding_path) != BINDINGS_SHA:
            raise ValueError('source binding changed')
        for relative, digest in self.document['files'].items():
            if sha(self.path(relative)) != digest:
                raise ValueError('bound source changed: ' + relative)


def load_inputs(bound):
    dm, rm, gm = [bound.json(a + '/manifest.json') for a in ('data', 'rna', 'gdsc')]
    if (dm['scope'] != 'exploratory_training_421_only' or dm['sample_count'] != N_SAMPLES
            or rm['existing_manifest_sha256'] != bound.document['files']['data/manifest.json']
            or gm['existing_data_sha256'] != bound.document['files']['data/manifest.json']
            or gm['source_sha256'] != '9d6dc82b9a072bb00b6139dcd6c4c11de1e047f210382fb726e92ab8dd0eae2a'):
        raise ValueError('prepared development package lineage/scope mismatch')
    for alias, manifest in (('data', dm), ('rna', rm), ('gdsc', gm)):
        for name in SOURCE_FILES[alias]:
            if name == 'manifest.json':
                continue
            row = manifest['files'][name]; path = bound.path(alias + '/' + name)
            if row['sha256'] != bound.document['files'][alias + '/' + name] or row['bytes'] != path.stat().st_size:
                raise ValueError('package member manifest mismatch')
    split = bound.json('reference/splits.json')
    if split != bound.json('measured-reference/splits.json'):
        raise ValueError('reference identity splits disagree')
    for alias in ('reference', 'measured-reference'):
        result = bound.json(alias + '/result.json')
        if result['status'] != 'completed' or result['scientific_claim'] is not False:
            raise ValueError('completed development references required')
    ids = bound.npy('data/sample_ids.npy')
    for actual in (bound.npy('rna/sample_ids.npy'), bound.npy('gdsc/sample_ids.npy'), split['sample_ids']):
        np.testing.assert_array_equal(actual, ids)
    if len(ids) != N_SAMPLES or len(set(ids)) != N_SAMPLES:
        raise ValueError('exact unique development identities required')
    labels = np.array(split['tissues']); folds = np.array(split['fold_ids'])
    np.testing.assert_array_equal(bridge.group_folds(labels, 5)[0], folds)
    m, rna, fm, y = [bound.npy(p) for p in ('data/metabolites.npy', 'rna/rna_log_cpm.npy', 'data/scgpt.npy', 'gdsc/auc.npy')]
    mids, genes, drugs = [bound.npy(p) for p in ('data/metabolite_ids.npy', 'rna/gene_ids.npy', 'gdsc/drug_ids.npy')]
    if (m.shape != (N_SAMPLES, 225) or fm.shape != (N_SAMPLES, 512)
            or rna.shape != (N_SAMPLES, len(genes)) or y.shape != (N_SAMPLES, len(drugs))
            or len(mids) != 225 or not all(np.isfinite(v).all() for v in (m, rna, fm)) or np.isinf(y).any()
            or any(len(set(v)) != len(v) for v in (mids, genes, drugs))):
        raise ValueError('prepared array dimensions/identity/value mismatch')
    rp = bridge.project_fixed(rna, genes, 20260908)
    mp = bridge.project_fixed(m, mids, 20260909)
    source = dict(m=m, rna_repr=rp['values'], fm=fm, m_repr=mp['values'], auc=y,
        sample_ids=ids, metabolite_ids=mids, gene_ids=genes, drug_ids=drugs, tissues=labels, fold_ids=folds)
    projection = dict(width=512, rna_seed=20260908, m_seed=20260909,
        rna_shape=list(rna.shape), m_shape=list(m.shape),
        policy='fixed Gaussian/sqrt(input width) on native coordinates; no data-fit projection moments')
    return source, split, projection


def task_grid(m_train, extra_train, m_query, extra_queries, y_train):
    """Only permitted task-training labels; no query outcome or other measured layer."""
    mz, ms = task.fit_blocks([m_train]); mq = task.apply_blocks([m_query], ms)
    km, hm, _ = task.component_kernel(mz, mq, 'rbf')
    if extra_train is None:
        kernels = [(km, hm)]
    else:
        ez, es = task.fit_blocks([extra_train]); kernels = []
        for values in extra_queries:
            eq = task.apply_blocks([values], es)
            ke, he, _ = task.component_kernel(ez, eq, 'rbf')
            kernels.append(((km + ke) / 2, (hm + he) / 2))
    by_generator = np.stack([np.stack([masked.predict_masked(k, h, y_train, a) for a in ALPHAS]) for k, h in kernels])
    if not np.isfinite(by_generator).all():
        raise ValueError('nonfinite fixed task predictions')
    return by_generator.mean(0), by_generator


def run_bank(source, fit, query, panel, root):
    root.mkdir()
    fit, query = np.asarray(fit, int), np.asarray(query, int)
    if set(fit) & set(query) or set(source['tissues'][fit]) & set(source['tissues'][query]):
        raise ValueError('task patient/tissue overlap')
    bank = bridge.build_bank(source['m'][fit], source['rna_repr'][fit], source['fm'][fit],
        source['m_repr'][fit], source['tissues'][fit], source['m'][query])
    save_json(root / 'selection.json', dict(fit=fit, query=query, generator_splits=bank['generator_splits']))
    for k, model in enumerate(bank['models']):
        save_npz(root / f'generator-{k}.npz', **model)
    save_npz(root / 'train-features.npz', **bank['train_features'])
    save_npz(root / 'query-features.npz', **bank['query_features'])
    save_json(root / 'upstream-frozen.json', {p.name: sha(p) for p in root.iterdir() if p.is_file()})
    train_y = source['auc'][np.ix_(fit, panel)]
    predictions, components = {}, {}
    predictions['m_only'], _ = task_grid(source['m'][fit], None, source['m'][query], None, train_y)
    for arm in VIRTUAL_ARMS:
        predictions[arm], components[arm] = task_grid(source['m'][fit], bank['train_features'][arm],
            source['m'][query], bank['query_features'][arm], train_y)
    # Only these two explicit diagnostics consume actual held RNA representations.
    permutation = bridge.within_tissue_permutation(source['tissues'][fit])
    for arm, rna_rows in (('measured_rna', fit), ('measured_shuffle_rna', fit[permutation])):
        predictions[arm], _ = task_grid(source['m'][fit], source['rna_repr'][rna_rows], source['m'][query],
            [source['rna_repr'][query]], train_y)
    save_npz(root / 'per-alpha-predictions.npz', **predictions)
    save_npz(root / 'query-generator-predictions.npz', **components)
    save_npz(root / 'measured-permutation.npz', permutation=permutation)
    save_json(root / 'predictions-frozen.json', {name: sha(root / name) for name in
        ('per-alpha-predictions.npz', 'query-generator-predictions.npz', 'measured-permutation.npz')})
    return predictions


def validation(predictions, truth, scale):
    result = {}
    if not np.isfinite(scale).all() or np.any(scale <= 1e-8):
        raise ValueError('degenerate task-training target scale')
    for arm in ARMS:
        error = ((predictions[arm] - truth[None, :, :]) / scale) ** 2
        n = np.isfinite(error).sum(1)
        result[arm + '_loss'] = np.divide(np.nansum(error, axis=1), n,
            out=np.zeros_like(n, dtype=float), where=n > 0)
        result[arm + '_count'] = (n > 0).astype(int)
    return result


def run_fold(source, bound, f, root):
    root.mkdir()
    fields = ('fit', 'held', 'panel', 'drug_ids', 'train_mean', 'train_sd')
    inherited = bound.npz(f'reference/predictions-{f}.npz', fields)
    fit, held = np.flatnonzero(source['fold_ids'] != f), np.flatnonzero(source['fold_ids'] == f)
    np.testing.assert_array_equal(inherited['fit'], fit); np.testing.assert_array_equal(inherited['held'], held)
    inner = bridge.group_folds(source['tissues'][fit], 3)[0]
    parts = [dict(fit=fit[inner != k], held=fit[inner == k]) for k in range(3)]
    old_parts = bound.json(f'reference/selection-{f}.json')['head_splits']
    for actual, expected in zip(parts, old_parts):
        np.testing.assert_array_equal(actual['fit'], expected['fit']); np.testing.assert_array_equal(actual['held'], expected['held'])
    y = source['auc']
    panel_mask = (np.isfinite(y[fit]).sum(0) >= 80) & (np.nanstd(y[fit], axis=0) > 1e-8)
    for part in parts:
        panel_mask &= np.isfinite(y[part['fit']]).sum(0) >= 20
    panel = np.flatnonzero(panel_mask)
    if not len(panel):
        raise ValueError('no training-eligible original drug panel')
    np.testing.assert_array_equal(inherited['panel'], panel)
    np.testing.assert_array_equal(inherited['drug_ids'], source['drug_ids'][panel])
    np.testing.assert_allclose(inherited['train_mean'], np.nanmean(y[np.ix_(fit, panel)], axis=0), rtol=1e-10)
    np.testing.assert_allclose(inherited['train_sd'], np.nanstd(y[np.ix_(fit, panel)], axis=0), rtol=1e-10)
    save_json(root / 'selection.json', dict(fit=fit, held=held, panel=panel, drug_ids=source['drug_ids'][panel], head_splits=parts))
    totals = {a + suffix: np.zeros((4, len(panel))) for a in ARMS for suffix in ('_loss', '_count')}
    for h, part in enumerate(parts):
        a, b = part['fit'], part['held']; sub = root / f'inner-{h}'
        predictions = run_bank(source, a, b, panel, sub)
        # The full four-alpha predictions have already been persisted.
        values = validation(predictions, y[np.ix_(b, panel)], np.nanstd(y[np.ix_(a, panel)], axis=0))
        save_npz(sub / 'validation.npz', **values)
        for key in totals:
            totals[key] += values[key]
    selected = {}
    for arm in ARMS:
        if not (totals[arm + '_count'] > 0).all():
            raise ValueError('missing all inner validation evidence for a drug')
        selected[arm] = np.argmin(totals[arm + '_loss'] / totals[arm + '_count'], axis=0)
    save_npz(root / 'selected.npz', **selected, **totals)
    predictions = run_bank(source, fit, held, panel, root / 'outer')
    chosen = {a: np.take_along_axis(predictions[a], selected[a][None, None, :], axis=0)[0] for a in ARMS}
    save_npz(root / 'predictions.npz', fit=fit, held=held, panel=panel, drug_ids=source['drug_ids'][panel], **chosen)
    prior = bound.npz(f'measured-reference/predictions-{f}.npz', (*fields, 'metabolite_only'))
    for key in fields:
        np.testing.assert_array_equal(prior[key], inherited[key])
    prior_selection = bound.json(f'measured-reference/selection-{f}.json')
    np.testing.assert_array_equal(selected['m_only'], prior_selection['selected']['metabolite_only'])
    for name, suffix in (('losses', '_loss'), ('counts', '_count')):
        np.testing.assert_allclose(totals['m_only' + suffix], prior_selection[name]['metabolite_only'], rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(chosen['m_only'], prior['metabolite_only'], rtol=1e-9, atol=1e-10)
    return dict(fold=f, fit_patients=len(fit), held_patients=len(held), drugs=len(panel),
        m_only_reference_replayed=True, predictions_sha=sha(root / 'predictions.npz'))


def score_all(source, root):
    metrics, per_drug, seen = [], [], np.zeros(len(source['sample_ids']), int)
    for f in range(5):
        pred = arrays(root / f'fold-{f}' / 'predictions.npz'); fit, held, panel = [pred[k] for k in ('fit', 'held', 'panel')]
        seen[held] += 1
        truth = source['auc'][np.ix_(held, panel)]
        scale = np.nanstd(source['auc'][np.ix_(fit, panel)], axis=0)
        counts = np.isfinite(truth).sum(0); eligible = counts >= 5
        if not eligible.any():
            raise ValueError('no held-observed original drug evaluation panel')
        for arm in ARMS:
            error = ((pred[arm] - truth) / scale) ** 2
            mse = np.divide(np.nansum(error, axis=0), counts, out=np.full(len(panel), np.nan), where=counts > 0)
            score = float(np.sqrt(np.mean(mse[eligible])))
            metrics.append(dict(fold=f, arm=arm, score=score, drug_ids=int(eligible.sum()),
                observed_pairs=int(counts[eligible].sum())))
            for j in range(len(panel)):
                per_drug.append(dict(fold=f, arm=arm, drug_id=str(source['drug_ids'][panel[j]]),
                    observed=int(counts[j]), evaluated=bool(eligible[j]), standardized_mse=float(mse[j]) if counts[j] else None))
    if not np.all(seen == 1):
        raise ValueError('each development patient must be predicted exactly once')
    means = {a: float(np.mean([r['score'] for r in metrics if r['arm'] == a])) for a in ARMS}
    groups = dict(non_fm=('virtual_rna', ('m_only', 'ordinary_m', 'shuffle_rna')),
        fm=('virtual_fm', ('m_only', 'virtual_rna', 'ordinary_m', 'shuffle_rna', 'shuffle_fm')))
    comparisons, gates = [], {}
    lookup = {(r['fold'], r['arm']): r['score'] for r in metrics}
    for name, (candidate, controls) in groups.items():
        gates[name] = {}
        for control in controls:
            effects = [lookup[f, control] - lookup[f, candidate] for f in range(5)]
            delta = means[control] - means[candidate]
            gates[name][control] = bool(delta > MARGIN)
            comparisons.append(dict(group=name, candidate=candidate, control=control,
                mean_control_minus_candidate=delta, relative_improvement=1 - means[candidate] / means[control],
                fold_effects=effects, folds_better=int(sum(v > MARGIN for v in effects))))
    return dict(mean_fold_scores=means, fold_metrics=metrics, per_drug=per_drug, comparisons=comparisons), gates


def run(*, data, rna, gdsc, reference, measured_reference, source_bindings, output):
    started = time.perf_counter(); output = Path(output).resolve()
    bound = BoundSources(dict(data=data, rna=rna, gdsc=gdsc, reference=reference,
        **{'measured-reference': measured_reference}), source_bindings)
    output.mkdir(parents=True, exist_ok=False)
    config = dict(schema='creed.gdsc-metabolite-query.development.v1', scientific_claim=False, chainable=False,
        external_confirmation=False, samples=N_SAMPLES, arms=list(ARMS), primary_arm='virtual_fm',
        representation_width=512, rna_projection_seed=20260908, m_projection_seed=20260909,
        permutation_seed=20260910, alphas=list(ALPHAS), generator_alpha=1., extra_kernel_weight=.5,
        query_generator_ensemble='mean of three task-head predictions, never mean input state or full-fit generator',
        comparison_margin=MARGIN, source_bindings_sha=BINDINGS_SHA,
        code_sha256={name: sha(path) for name, path in dict(runner=__file__, bridge=bridge.__file__,
            task=task.__file__, masked=masked.__file__, io=save_npz.__code__.co_filename).items()},
        protocol_sha=sha(Path(__file__).resolve().parents[1] / 'GDSC_METABOLITE_QUERY_NEXT.md'),
        caveats=['all421 are adaptively reused development cell lines, not independent external validation',
            'shared cell-line identity does not establish matched culture batch',
            '512-dimensional predicted representations are not full-transcriptome or L0-L7 recovery',
            'measured arms use extra query RNA and cannot rescue deployable-arm failure',
            'single tissue-preserving permutation is not a formal null distribution',
            'correlated drugs/folds do not establish significance or clinical treatment benefit'])
    save_json(output / 'frozenconfig.json', config); save_json(output / 'bindings.json', bound.document)
    source, split, projection = load_inputs(bound)
    save_npz(output / 'source-inputs.npz', **source)
    save_json(output / 'splits.json', split); save_json(output / 'projection.json', projection)
    records = []
    for f in range(5):
        record = run_fold(source, bound, f, output / f'fold-{f}'); records.append(record)
        print(str(dict(**record, elapsed=time.perf_counter() - started)), flush=True)
    save_json(output / 'all-predictions-frozen.json', {str(r['fold']): r['predictions_sha'] for r in records})
    scores, gates = score_all(source, output)
    save_json(output / 'scores.json', scores)
    bound.finish()
    result = dict(**config, status='completed', **scores, gates=gates,
        non_fm_gate_passed=bool(all(gates['non_fm'].values())), fm_gate_passed=bool(all(gates['fm'].values())),
        global_gate_passed=bool(all(gates['fm'].values())), fold_records=records,
        source_members_loaded=bound.loaded, frozenconfig_sha=sha(output / 'frozenconfig.json'),
        artifacts={str(p.relative_to(output)).replace('\\', '/'): sha(p) for p in output.rglob('*') if p.is_file()},
        elapsed_seconds=time.perf_counter() - started)
    save_json(output / 'summary.json', result)
    print(str(dict(status='completed', mean_fold_scores=scores['mean_fold_scores'], gates=gates,
        elapsed_seconds=result['elapsed_seconds'])), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('data', 'rna', 'gdsc', 'reference', 'measured-reference', 'source-bindings', 'output'):
        parser.add_argument('--' + key, required=True)
    run(**vars(parser.parse_args()))

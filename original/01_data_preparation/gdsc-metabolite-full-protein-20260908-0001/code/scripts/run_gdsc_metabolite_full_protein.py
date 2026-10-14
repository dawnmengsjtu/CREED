"""Fixed full-source protein-reference task candidate; query input is M only."""
import argparse
from pathlib import Path
import time

import numpy as np
from scipy.spatial.distance import cdist

from creed.science import metabolite_protein_query as bridge
from creed.science import masked_task_kernel as masked
import run_gdsc_metabolite_query as parent_io
from run_gdsc_metabolite_task_metric import require_verified_metrics
from run_phospho_fewshot import sha, read_json, save_json, save_npz

BINDINGS_SHA = '457e45bcb80adec8cbf375ff4701453b252f9db8d87d2ac8568057a04292809a'
BINDING_SCHEMA = 'creed.gdsc-metabolite-full-protein.bound-sources.v1'
N_SAMPLES, M_WIDTH, PROTEIN_WIDTH = 421, 225, 6692
SOURCE_FIELDS = ('m', 'auc', 'sample_ids', 'metabolite_ids', 'drug_ids', 'tissues', 'fold_ids')
PREDICTION_FIELDS = ('fit', 'held', 'panel', 'drug_ids', 'm_only', 'virtual_rna', 'virtual_fm')
PROTEIN_FILES = ('manifest.json', 'proteins.npy', 'protein_ids.npy', 'sample_ids.npy')
ALPHAS, MARGIN = (.1, 1., 10., 100.), 1e-10
VIRTUAL = tuple(bridge.ARMS)
DEPLOYABLE = ('m_only', *VIRTUAL)
DIAGNOSTIC = ('measured_p', 'measured_mask')
ARMS = (*DEPLOYABLE, *DIAGNOSTIC)
FROZEN = dict(frozen_virtual_rna='virtual_rna', frozen_virtual_fm='virtual_fm')
SCORED_ARMS = (*ARMS, *FROZEN)
CONTROLS = (*[a for a in DEPLOYABLE if a != 'virtual_p'], *FROZEN)


def allowed_parent_files():
    return {'source-inputs.npz', 'splits.json', *[f'fold-{f}/{n}' for f in range(5)
            for n in ('selection.json', 'predictions.npz')]}


class Sources:
    """Only twelve parent artifacts and four protein objects are readable."""
    def __init__(self, parent, verification, protein, source_bindings):
        self.root, self.protein = Path(parent).resolve(strict=True), Path(protein).resolve(strict=True)
        self.verification, self.binding_path = Path(verification), Path(source_bindings)
        if sha(self.binding_path) != BINDINGS_SHA:
            raise ValueError('exact full-protein source binding required')
        self.binding = read_json(self.binding_path)
        if (self.binding.get('schema') != BINDING_SCHEMA or self.binding.get('scientific_claim') is not False
                or set(self.binding['parent_files']) != allowed_parent_files()
                or set(self.binding['protein_files']) != set(PROTEIN_FILES)):
            raise ValueError('exact source object whitelist required')
        if (sha(self.root / 'summary.json') != self.binding['parent_summary_sha256']
                or sha(self.verification) != self.binding['parent_verification_sha256']):
            raise ValueError('parent summary/receipt SHA mismatch')
        self.summary, checked = read_json(self.root / 'summary.json'), read_json(self.verification)
        if (self.summary['status'] != 'completed' or self.summary['scientific_claim'] is not False
                or checked['status'] != 'verified' or checked['scientific_claim'] is not False
                or checked['producer_summary_sha256'] != self.binding['parent_summary_sha256']
                or checked['checked_artifacts'] != self.summary['artifacts']):
            raise ValueError('independently verified development parent required')
        require_verified_metrics(self.summary, checked)
        if any(self.summary['artifacts'].get(name) != digest for name, digest in self.binding['parent_files'].items()):
            raise ValueError('parent artifact binding differs from checked receipt')
        self.parent_fields, self.protein_fields = {}, {}
        self.finish()

    def path(self, kind, name):
        allowed = self.binding['parent_files' if kind == 'parent' else 'protein_files'] if kind in ('parent', 'protein') else {}
        if name not in allowed:
            raise ValueError('unlisted source object')
        root = self.root if kind == 'parent' else self.protein
        path = root / name
        if path.is_symlink() or root not in path.resolve().parents or sha(path) != allowed[name]:
            raise ValueError('source path or SHA mismatch: ' + name)
        return path

    def json(self, name):
        if not name.endswith('.json'):
            raise ValueError('unlisted parent JSON')
        self.parent_fields[name] = 'json'
        return read_json(self.path('parent', name))

    def npz(self, name):
        fields = SOURCE_FIELDS if name == 'source-inputs.npz' else PREDICTION_FIELDS if name.endswith('/predictions.npz') else None
        if fields is None:
            raise ValueError('unlisted parent array fields')
        self.parent_fields[name] = list(fields)
        return parent_io.arrays(self.path('parent', name), fields)

    def protein_npy(self, name):
        if name not in PROTEIN_FILES or not name.endswith('.npy'):
            raise ValueError('unlisted protein array')
        self.protein_fields[name] = 'npy'
        return np.load(self.path('protein', name), allow_pickle=False)

    def finish(self):
        if (sha(self.binding_path) != BINDINGS_SHA
                or sha(self.root / 'summary.json') != self.binding['parent_summary_sha256']
                or sha(self.verification) != self.binding['parent_verification_sha256']):
            raise ValueError('bound source changed')
        for kind in ('parent', 'protein'):
            for name in self.binding[kind + '_files']:
                self.path(kind, name)


def _identities(values, size, name):
    values = np.asarray(values)
    if (values.shape != (size,) or len(set(values.tolist())) != size
            or any(not isinstance(x, str) or not x or x.strip() != x for x in values.tolist())):
        raise ValueError('invalid unique ordered ' + name)


def load_source(bound):
    source, split = bound.npz('source-inputs.npz'), bound.json('splits.json')
    manifest = read_json(bound.path('protein', 'manifest.json'))
    bound.protein_fields['manifest.json'] = 'json'
    if (manifest['scope'] != 'exploratory 421 training prefix only, expanded missing-aware protein source'
            or manifest['scientific_claim'] is not False
            or manifest['source_sha256'] != '468767f19ea2c935285c0d0877475ea97cb70140662b307f39e841d91a81e5d1'):
        raise ValueError('wrong safe full-protein source lineage')
    for name in PROTEIN_FILES[1:]:
        item = manifest['files'][name]; path = bound.path('protein', name)
        if item['sha256'] != bound.binding['protein_files'][name] or item['bytes'] != path.stat().st_size:
            raise ValueError('protein member manifest SHA/bytes mismatch')
    source['p'] = bound.protein_npy('proteins.npy')
    source['protein_ids'] = bound.protein_npy('protein_ids.npy')
    np.testing.assert_array_equal(bound.protein_npy('sample_ids.npy'), source['sample_ids'])
    for key in ('sample_ids', 'tissues', 'fold_ids'):
        np.testing.assert_array_equal(source[key], split[key])
    for key, size in (('sample_ids', N_SAMPLES), ('metabolite_ids', M_WIDTH), ('protein_ids', PROTEIN_WIDTH),
                      ('drug_ids', len(source['drug_ids']))):
        _identities(source[key], size, key)
    if (source['m'].shape != (N_SAMPLES, M_WIDTH) or not np.isfinite(source['m']).all()
            or source['p'].shape != (N_SAMPLES, PROTEIN_WIDTH) or np.isinf(source['p']).any()
            or source['auc'].shape != (N_SAMPLES, len(source['drug_ids'])) or np.isinf(source['auc']).any()):
        raise ValueError('exact safe M/P/AUC shapes and finite/missing conventions required')
    np.testing.assert_array_equal(bridge.group_folds(source['tissues'], 5)[0], source['fold_ids'])
    return source, split


def encode_extra(train, queries):
    """One A-only native mean/SD; direct full-J Euclidean geometry thereafter."""
    train, queries = np.asarray(train, float), np.asarray(queries, float)
    if (train.ndim != 2 or not train.shape[1] or queries.ndim != 3 or queries.shape[2] != train.shape[1]
            or not np.isfinite(train).all() or not np.isfinite(queries).all()):
        raise ValueError('finite aligned native extra states required')
    mean, sd = train.mean(0), train.std(0); sd = np.where(sd < 1e-8, 1., sd)
    z, q = (train - mean) / sd, (queries - mean) / sd
    if not all(np.isfinite(v).all() for v in (mean, sd, z, q)):
        raise ValueError('nonfinite once-only extra encoding')
    constant = bool(np.max(np.ptp(z, axis=0)) < 1e-12)
    if constant:
        width = 0.; kernel = np.ones((len(z), len(z))); cross = np.ones((len(q), q.shape[1], len(z)))
    else:
        distance = cdist(z, z, 'sqeuclidean'); upper = distance[np.triu_indices(len(z), 1)]
        positive = upper[upper > 0]
        if not len(positive):
            raise ValueError('nonconstant extra state has no positive distances')
        width = float(np.median(positive))
        kernel = np.exp(-distance / (2 * width))
        cross = np.stack([np.exp(-cdist(x, z, 'sqeuclidean') / (2 * width)) for x in q])
    return dict(mean=mean, sd=sd, width=np.array(width), constant=np.array(constant)), z, q, kernel, cross


def measured_features(p_fit, p_query):
    """Only this diagnostic interface accepts query protein values/missingness."""
    observed = np.isfinite(p_fit); query_observed = np.isfinite(p_query)
    if np.isinf(p_fit).any() or np.isinf(p_query).any() or not observed.any(0).all():
        raise ValueError('invalid measured-reference protein inputs')
    means = np.nanmean(p_fit, axis=0)
    return (dict(measured_p=np.where(observed, p_fit, means), measured_mask=observed.astype(float)),
            dict(measured_p=np.where(query_observed, p_query, means)[None], measured_mask=query_observed.astype(float)[None]), means)


def extra_grid(m_kernel, m_cross, extra_kernel, extra_cross, y):
    kernel = .5 * m_kernel + .5 * extra_kernel
    components = np.stack([np.stack([masked.predict_masked(kernel, .5 * m_cross + .5 * h, y, alpha)
                                    for alpha in ALPHAS]) for h in extra_cross])
    if not np.isfinite(components).all():
        raise ValueError('nonfinite masked task predictions')
    return components.mean(0), components


def validation(predictions, truth, scale):
    if not np.isfinite(scale).all() or np.any(scale <= 1e-8):
        raise ValueError('nondegenerate A-only AUC scales required')
    result = {}
    for arm, values in predictions.items():
        if not np.isfinite(values).all():
            raise ValueError('nonfinite prediction cannot shrink the scoring mask')
        error = ((values - truth[None]) / scale) ** 2; count = np.isfinite(error).sum(1)
        result[arm + '_loss'] = np.divide(np.nansum(error, axis=1), count, out=np.zeros_like(count, float), where=count > 0)
        result[arm + '_count'] = (count > 0).astype(int)
    return result


def check_m_baseline(source, reference, parts, root):
    panel = reference['panel']; total = np.zeros((4, len(panel))); count = np.zeros_like(total)
    for part in parts:
        a, b = part['fit'], part['held']
        grid, _ = parent_io.task_grid(source['m'][a], None, source['m'][b], None, source['auc'][np.ix_(a, panel)])
        value = validation({'m_only': grid}, source['auc'][np.ix_(b, panel)], np.nanstd(source['auc'][np.ix_(a, panel)], axis=0))
        total += value['m_only_loss']; count += value['m_only_count']
    if not (count > 0).all():
        raise ValueError('missing baseline inner validation evidence')
    selected = np.argmin(total / count, axis=0)
    grid, _ = parent_io.task_grid(source['m'][reference['fit']], None, source['m'][reference['held']], None,
                                source['auc'][np.ix_(reference['fit'], panel)])
    prediction = np.take_along_axis(grid, selected[None, None], axis=0)[0]
    np.testing.assert_allclose(prediction, reference['m_only'], rtol=1e-12, atol=1e-12)
    save_npz(root / 'm-only-replay.npz', fit=reference['fit'], held=reference['held'], panel=panel,
             prediction=prediction, frozen_prediction=reference['m_only'], selected=selected, loss=total, count=count)


def run_bank(source, fit, query, panel, ordinary_projection, root):
    root.mkdir()
    if set(fit) & set(query) or set(source['tissues'][fit]) & set(source['tissues'][query]):
        raise ValueError('task query overlaps fitting tissues')
    bank = bridge.build_bank(source['m'][fit], source['p'][fit], source['tissues'][fit], source['m'][query], ordinary_projection)
    pi = bank['panel']['indices']
    save_json(root / 'selection.json', dict(fit=fit, query=query, panel=panel, source_protein_indices=pi,
        generator_splits=bank['generator_splits']))
    save_npz(root / 'panel.npz', **{k: v for k, v in bank['panel'].items() if k != 'generator_splits'},
             source_protein_ids=source['protein_ids'], protein_ids=source['protein_ids'][pi])
    for g, model in enumerate(bank['models']): save_npz(root / f'generator-{g}.npz', **model)
    save_npz(root / 'train-native.npz', **bank['train_features'])
    save_npz(root / 'query-native.npz', **bank['query_features'])
    train, queries = dict(bank['train_features']), dict(bank['query_features'])
    measured_train, measured_query, fill_mean = measured_features(source['p'][fit][:, pi], source['p'][query][:, pi])
    train.update(measured_train); queries.update(measured_query)
    save_npz(root / 'measured-inputs.npz', fill_mean=fill_mean, **{a + '_train': v for a, v in measured_train.items()},
             **{a + '_query': v for a, v in measured_query.items()})
    mz, ms = parent_io.task.fit_blocks([source['m'][fit]]); mq = parent_io.task.apply_blocks([source['m'][query]], ms)
    mk, mh, mw = parent_io.task.component_kernel(mz, mq, 'rbf')
    save_npz(root / 'm-kernel.npz', mean=ms[0][0], sd=ms[0][1], train=mz, query=mq, width=mw, kernel=mk, cross=mh)
    encoded_train, encoded_query, encoding, kernels = {}, {}, {}, {}
    grids, components = {}, {}
    grids['m_only'], components['m_only'] = parent_io.task_grid(source['m'][fit], None, source['m'][query], None,
                                                              source['auc'][np.ix_(fit, panel)])
    for arm in (*VIRTUAL, *DIAGNOSTIC):
        state, z, q, k, h = encode_extra(train[arm], queries[arm])
        encoding.update({arm + '_' + key: value for key, value in state.items()})
        encoded_train[arm], encoded_query[arm] = z, q
        kernels[arm + '_train'], kernels[arm + '_query'] = k, h
        grids[arm], components[arm] = extra_grid(mk, mh, k, h, source['auc'][np.ix_(fit, panel)])
    save_npz(root / 'encodings.npz', **encoding)
    save_npz(root / 'encoded-train.npz', **encoded_train); save_npz(root / 'encoded-query.npz', **encoded_query)
    save_npz(root / 'extra-kernels.npz', **kernels)
    save_json(root / 'upstream-frozen.json', {p.name: sha(p) for p in root.iterdir() if p.is_file()})
    save_npz(root / 'per-alpha-predictions.npz', **grids)
    save_npz(root / 'query-generator-predictions.npz', **components)
    save_json(root / 'predictions-frozen.json', {name: sha(root / name) for name in
        ('per-alpha-predictions.npz', 'query-generator-predictions.npz')})
    return grids


def run_fold(bound, source, fold, projection, root):
    root.mkdir(); ref = bound.npz(f'fold-{fold}/predictions.npz'); old = bound.json(f'fold-{fold}/selection.json')
    fit, held = np.flatnonzero(source['fold_ids'] != fold), np.flatnonzero(source['fold_ids'] == fold)
    np.testing.assert_array_equal(ref['fit'], fit); np.testing.assert_array_equal(ref['held'], held)
    inner = bridge.group_folds(source['tissues'][fit], 3)[0]
    parts = [dict(fit=fit[inner != g], held=fit[inner == g]) for g in range(3)]
    if len(old['head_splits']) != 3: raise ValueError('three frozen head splits required')
    for actual, expected in zip(parts, old['head_splits']):
        for key in ('fit', 'held'): np.testing.assert_array_equal(actual[key], expected[key])
    y = source['auc']; keep = (np.isfinite(y[fit]).sum(0) >= 80) & (np.nanstd(y[fit], axis=0) > 1e-8)
    for part in parts: keep &= np.isfinite(y[part['fit']]).sum(0) >= 20
    panel = np.flatnonzero(keep)
    if not len(panel): raise ValueError('empty original drug panel')
    for expected in (ref['panel'], old['panel']): np.testing.assert_array_equal(panel, expected)
    for key, value in (('fit', fit), ('held', held), ('drug_ids', source['drug_ids'][panel])):
        np.testing.assert_array_equal(old[key], value)
    np.testing.assert_array_equal(ref['drug_ids'], source['drug_ids'][panel])
    save_json(root / 'selection.json', dict(fit=fit, held=held, panel=panel, drug_ids=source['drug_ids'][panel], head_splits=parts))
    check_m_baseline(source, ref, parts, root)
    total = {a + suffix: np.zeros((4, len(panel))) for a in ARMS for suffix in ('_loss', '_count')}
    for g, part in enumerate(parts):
        a, b = part['fit'], part['held']; sub = root / f'inner-{g}'
        grids = run_bank(source, a, b, panel, projection, sub)
        losses = validation(grids, y[np.ix_(b, panel)], np.nanstd(y[np.ix_(a, panel)], axis=0))
        save_npz(sub / 'validation.npz', **losses)
        for key in total: total[key] += losses[key]
    chosen = {}
    for arm in ARMS:
        if not (total[arm + '_count'] > 0).all(): raise ValueError('missing all inner validation evidence')
        chosen[arm] = np.argmin(total[arm + '_loss'] / total[arm + '_count'], axis=0)
    save_npz(root / 'selected.npz', **chosen, **total)
    grids = run_bank(source, fit, held, panel, projection, root / 'outer')
    predictions = {a: np.take_along_axis(grids[a], chosen[a][None, None], axis=0)[0] for a in ARMS}
    np.testing.assert_allclose(predictions['m_only'], ref['m_only'], rtol=1e-12, atol=1e-12)
    predictions.update({name: ref[original].copy() for name, original in FROZEN.items()})
    save_npz(root / 'predictions.npz', fit=fit, held=held, panel=panel, drug_ids=ref['drug_ids'], **predictions)
    saved = parent_io.arrays(root / 'predictions.npz', list(FROZEN))
    for name, original in FROZEN.items():
        if saved[name].dtype != ref[original].dtype or saved[name].tobytes() != ref[original].tobytes():
            raise ValueError('frozen generic reference changed')
    return dict(fold=fold, predictions_sha256=sha(root / 'predictions.npz'), m_only_replayed=True,
                frozen_references_copied_exactly=True, fit_samples=len(fit), held_samples=len(held), drugs=len(panel))


def score_all(source, root):
    rows, per_drug, panel_diagnostics = [], [], []; seen = np.zeros(N_SAMPLES, int)
    for fold in range(5):
        pred = parent_io.arrays(root / f'fold-{fold}/predictions.npz')
        fit, held, panel = (pred[k] for k in ('fit', 'held', 'panel')); seen[held] += 1
        truth = source['auc'][np.ix_(held, panel)]; scale = np.nanstd(source['auc'][np.ix_(fit, panel)], axis=0)
        count = np.isfinite(truth).sum(0); evaluated = count >= 5
        if not evaluated.any() or not np.isfinite(scale).all() or np.any(scale <= 1e-8):
            raise ValueError('unscorable original evaluation panel')
        save_npz(root / f'fold-{fold}/scoring-reference.npz', truth=truth, train_sd=scale,
                 observed=np.isfinite(truth), observed_counts=count, evaluated=evaluated)
        for arm in SCORED_ARMS:
            if not np.isfinite(pred[arm]).all(): raise ValueError('nonfinite prediction cannot shrink evaluation')
            mse = np.divide(np.nansum(((pred[arm] - truth) / scale) ** 2, axis=0), count,
                            out=np.full(len(panel), np.nan), where=count > 0)
            rows.append(dict(fold=fold, arm=arm, score=float(np.sqrt(mse[evaluated].mean())),
                             drug_ids=int(evaluated.sum()), observed_pairs=int(count[evaluated].sum())))
            per_drug.extend(dict(fold=fold, arm=arm, drug_id=str(drug), observed=int(count[j]),
                evaluated=bool(evaluated[j]), standardized_mse=float(mse[j]) if count[j] else None) for j, drug in enumerate(pred['drug_ids']))
        for bank in ('inner-0', 'inner-1', 'inner-2', 'outer'):
            state = parent_io.arrays(root / f'fold-{fold}/{bank}/panel.npz', ['indices', 'unsupported'])
            panel_diagnostics.append(dict(fold=fold, bank=bank, selected_proteins=len(state['indices']),
                                          unsupported_proteins=int(state['unsupported'].sum())))
    if not np.all(seen == 1): raise ValueError('five-fold sample coverage changed')
    means = {arm: float(np.mean([r['score'] for r in rows if r['arm'] == arm])) for arm in SCORED_ARMS}
    lookup = {(r['fold'], r['arm']): r['score'] for r in rows}; comparisons = []; gates = dict(mean={}, sensitivity={})
    for control in CONTROLS:
        effects = [lookup[f, control] - lookup[f, 'virtual_p'] for f in range(5)]
        loo = [float(np.mean([v for j, v in enumerate(effects) if j != f])) for f in range(5)]
        delta = means[control] - means['virtual_p']; gates['mean'][control] = bool(delta > MARGIN)
        gates['sensitivity'][control] = bool(min(loo) > MARGIN)
        comparisons.append(dict(candidate='virtual_p', control=control, mean_control_minus_candidate=delta,
            fold_effects=effects, leave_one_fold_out_effects=loo, folds_better=sum(v > MARGIN for v in effects)))
    return dict(mean_fold_scores=means, fold_metrics=rows, per_drug=per_drug, comparisons=comparisons,
                panel_diagnostics=panel_diagnostics), gates


def run(*, parent, verification, protein, source_bindings, output):
    start = time.perf_counter(); bound = Sources(parent, verification, protein, source_bindings)
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    code = Path(__file__).resolve().parents[1]
    paths = dict(runner=__file__, bridge=bridge.__file__, parent_io=parent_io.__file__, masked=masked.__file__,
        coordinates=code / 'src/creed/science/metabolite_query.py', task=parent_io.task.__file__,
        parent_metric_check=code / 'scripts/run_gdsc_metabolite_task_metric.py', io=code / 'scripts/run_phospho_fewshot.py')
    config = dict(schema='creed.gdsc-metabolite-full-protein.development.v1', scientific_claim=False,
        external_confirmation=False, chainable=False, samples=N_SAMPLES, m_width=M_WIDTH, source_proteins=PROTEIN_WIDTH,
        arms=list(ARMS), frozen_references=FROZEN, primary='virtual_p', primary_controls=list(CONTROLS),
        alphas=list(ALPHAS), extra_kernel_weight=.5, generator_alpha=1., min_a_protein_coverage=.5,
        min_generator_observed=20, min_generator_sd=1e-8, pairing_seed=20260910, ordinary_projection_seed=20260909,
        task_encoding='each arm own A-native/OOF mean and population SD once; full J; no second scaling',
        constant_extra_rule='training normalized max ptp <1e-12 makes train/all query kernels ones',
        comparison_margin=MARGIN, source_bindings_sha256=sha(source_bindings),
        protocol_sha256=sha(code / 'GDSC_METABOLITE_FULL_PROTEIN_NEXT.md'), code_sha256={k: sha(p) for k, p in paths.items()})
    save_json(output / 'frozenconfig.json', config); save_json(output / 'bindings.json', bound.binding)
    source, split = load_source(bound)
    save_json(output / 'splits.json', split); save_npz(output / 'source-inputs.npz', **source)
    projection = np.random.default_rng(20260909).standard_normal((M_WIDTH, PROTEIN_WIDTH)) / np.sqrt(M_WIDTH)
    save_npz(output / 'ordinary-projection.npz', projection=projection, metabolite_ids=source['metabolite_ids'], protein_ids=source['protein_ids'])
    records = []
    for fold in range(5):
        record = run_fold(bound, source, fold, projection, output / f'fold-{fold}'); records.append(record)
        print(str(dict(**record, elapsed_seconds=time.perf_counter() - start)), flush=True)
    save_json(output / 'all-predictions-frozen.json', {str(r['fold']): r['predictions_sha256'] for r in records})
    scores, gates = score_all(source, output); save_json(output / 'scores.json', scores)
    for new, old in (('m_only', 'm_only'), *FROZEN.items()):
        if not np.isclose(scores['mean_fold_scores'][new], bound.summary['mean_fold_scores'][old], rtol=2e-14, atol=2e-14):
            raise ValueError('frozen parent task score replay mismatch')
    bound.finish()
    result = dict(**config, status='completed', **scores, gates=gates,
        mean_gate_passed=all(gates['mean'].values()), sensitivity_gate_passed=all(gates['sensitivity'].values()),
        global_gate_passed=all(gates['mean'].values()) and all(gates['sensitivity'].values()), fold_records=records,
        parent_fields=bound.parent_fields, protein_fields=bound.protein_fields,
        parent_members=bound.binding['parent_files'], protein_members=bound.binding['protein_files'],
        frozenconfig_sha256=sha(output / 'frozenconfig.json'), elapsed_seconds=time.perf_counter() - start,
        artifacts={str(p.relative_to(output)).replace('\\', '/'): sha(p) for p in output.rglob('*') if p.is_file()},
        caveats=['adaptive reused development; not independent validation', 'native protein units unresolved',
            'missingness may be MNAR or batch-related; controls do not prove protein biology',
            'measured P/mask diagnostics use extra query assays and cannot rescue deployable gate',
            'native eligible-P predictions are not full6692 recovery or calibrated chainable layers',
            'generic RNA/FM use512 coordinates; strong comparisons, not matched-width ablations',
            'no causal, clinical, FM necessity, L0-L7 or NMI success claim'])
    save_json(output / 'summary.json', result)
    print(str(dict(status='completed', mean_fold_scores=scores['mean_fold_scores'], gates=gates)), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('parent', 'verification', 'protein', 'source-bindings', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    run(**vars(parser.parse_args()))

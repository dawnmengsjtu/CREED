"""Task-residual-directed geometry, with fixed equal-supervision controls."""
import argparse
from pathlib import Path
import time
import numpy as np
from scipy.spatial.distance import cdist

from creed.science import task_residual_metric as metric
from creed.science import virtual_state_task as task
from creed.science import masked_task_kernel as masked
from run_gdsc_metabolite_query import arrays
from run_phospho_fewshot import sha, read_json, save_json, save_npz

BINDINGS_SHA = '2bdc4f8a6831c57cc9b0e93c53ca94b9c20346cde62a134b61f2c773c15c0d23'
N_SAMPLES = 421
ARM_MAP = dict(adapted_rna='virtual_rna', adapted_fm='virtual_fm', adapted_ordinary_m='ordinary_m',
               adapted_shuffle_rna='shuffle_rna', adapted_shuffle_fm='shuffle_fm')
ARMS = tuple(ARM_MAP)
ALPHAS = (.1, 1., 10., 100.)
MARGIN = 1e-10
SOURCE_FIELDS = ('m', 'auc', 'sample_ids', 'drug_ids', 'tissues', 'fold_ids')
MODEL_FIELDS = ('schema', 'mean', 'sd', 'x', 'width', 'alpha')


def require_verified_metrics(summary, checked):
    """Pinned independent arithmetic can differ by a few float64 roundoff bits.

    Identities, counts and field sets remain exact. Only numeric scores allow
    a tolerance far below the fixed scientific comparison margin (1e-10).
    """
    def score(left, right):
        if (isinstance(left, bool) or isinstance(right, bool)
                or not isinstance(left, (int, float)) or not isinstance(right, (int, float))
                or not np.isfinite(left) or not np.isfinite(right)
                or not np.isclose(left, right, rtol=2e-14, atol=2e-14)):
            raise ValueError('parent verified metric mismatch')

    means, verified_means = summary['mean_fold_scores'], checked['mean_fold_scores']
    if set(means) != set(verified_means):
        raise ValueError('parent verified metric identities mismatch')
    for arm in means:
        score(means[arm], verified_means[arm])
    rows, verified_rows = summary['fold_metrics'], checked['fold_metrics']
    if len(rows) != len(verified_rows):
        raise ValueError('parent verified metric row count mismatch')
    for row, verified in zip(rows, verified_rows):
        if (set(row) != set(verified)
                or {k: v for k, v in row.items() if k != 'score'}
                != {k: v for k, v in verified.items() if k != 'score'}):
            raise ValueError('parent verified metric identities/counts mismatch')
        score(row['score'], verified['score'])


def allowed_parent_files():
    names = {'source-inputs.npz', 'splits.json'}
    for f in range(5):
        prefix = f'fold-{f}'
        names.update(prefix + '/' + n for n in ('selection.json', 'predictions.npz'))
        for bank in ('inner-0', 'inner-1', 'inner-2', 'outer'):
            names.add(prefix + '/' + bank + '/selection.json')
            names.update(prefix + '/' + bank + f'/generator-{g}.npz' for g in range(3))
            names.update(prefix + '/' + bank + '/' + n for n in ('train-features.npz', 'query-features.npz'))
    return names


def parent_fields(name):
    if name.endswith('.json'):
        return 'json'
    if name == 'source-inputs.npz':
        return list(SOURCE_FIELDS)
    if '/generator-' in name:
        return list(MODEL_FIELDS)
    if name.endswith('/predictions.npz'):
        return ['fit', 'held', 'panel', 'drug_ids']
    if name.endswith(('train-features.npz', 'query-features.npz')):
        return list(ARM_MAP.values())
    raise ValueError('unlisted parent array fields')


class Parent:
    def __init__(self, root, verification, binding):
        self.root = Path(root).resolve(strict=True)
        self.verification, self.binding_path = Path(verification), Path(binding)
        if sha(binding) != BINDINGS_SHA:
            raise ValueError('exact task-metric source binding required')
        self.binding = read_json(binding)
        if (sha(self.root / 'summary.json') != self.binding['parent_summary_sha256']
                or sha(verification) != self.binding['parent_verification_sha256']):
            raise ValueError('parent summary/receipt SHA mismatch')
        self.summary, checked = read_json(self.root / 'summary.json'), read_json(verification)
        if (checked['status'] != 'verified' or checked['scientific_claim'] is not False
                or checked['producer_summary_sha256'] != self.binding['parent_summary_sha256']
                or checked['checked_artifacts'] != self.summary['artifacts']
                or self.summary['status'] != 'completed'):
            raise ValueError('independently verified development parent required')
        require_verified_metrics(self.summary, checked)
        self.used, self.fields = {}, {}

    def path(self, name):
        if name not in allowed_parent_files():
            raise ValueError('unlisted parent object')
        path = self.root / name
        if path.is_symlink() or self.root not in path.resolve().parents:
            raise ValueError('parent path escapes root')
        expected = self.summary['artifacts'][name]
        if sha(path) != expected:
            raise ValueError('parent artifact SHA mismatch: ' + name)
        self.used[name] = expected
        return path

    def json(self, name):
        self.fields[name] = 'json'
        return read_json(self.path(name))

    def npz(self, name):
        fields = parent_fields(name)
        if fields == 'json':
            raise ValueError('not an array object')
        self.fields[name] = fields
        return arrays(self.path(name), fields)

    def finish(self):
        if (sha(self.binding_path) != BINDINGS_SHA
                or sha(self.root / 'summary.json') != self.binding['parent_summary_sha256']
                or sha(self.verification) != self.binding['parent_verification_sha256']):
            raise ValueError('bound parent changed')
        for name in list(self.used):
            self.path(name)


def load_source(parent):
    source, split = parent.npz('source-inputs.npz'), parent.json('splits.json')
    if (source['m'].shape != (N_SAMPLES, 225) or not np.isfinite(source['m']).all()
            or source['auc'].shape != (N_SAMPLES, len(source['drug_ids'])) or np.isinf(source['auc']).any()
            or len(set(source['sample_ids'])) != N_SAMPLES):
        raise ValueError('exact prepared M/task dimensions required')
    for key in ('sample_ids', 'tissues', 'fold_ids'):
        np.testing.assert_array_equal(source[key], split[key])
    return source


def baseline_oof(parent, source, prefix, fit, panel, selection):
    result = np.full((len(fit), len(panel)), np.nan)
    seen = np.zeros(len(fit), int)
    for g, split in enumerate(selection['generator_splits']):
        a, b = np.array(split['fit'], int), np.array(split['held_local'], int)
        if set(a) & set(b) or set(source['tissues'][fit[a]]) & set(source['tissues'][fit[b]]):
            raise ValueError('residual baseline tissue overlap')
        state = parent.npz(prefix + f'/generator-{g}.npz')
        if str(state['schema']) != 'creed.metabolite-query-generator.v1' or float(state['alpha']) != 1.:
            raise ValueError('frozen parent M generator reference required')
        x, mean, sd, width = state['x'], state['mean'], state['sd'], float(state['width'])
        np.testing.assert_allclose(x, (source['m'][fit[a]] - mean) / sd, rtol=1e-10, atol=1e-10)
        q = (source['m'][fit[b]] - mean) / sd
        k = np.exp(-cdist(x, x, 'sqeuclidean') / (2 * width))
        h = np.exp(-cdist(q, x, 'sqeuclidean') / (2 * width))
        result[b] = masked.predict_masked(k, h, source['auc'][np.ix_(fit[a], panel)], 1., min_observed=20)
        seen[b] += 1
    if not np.all(seen == 1):
        raise ValueError('residual OOF coverage mismatch')
    return result


def projected_grid(m_train, m_query, projected_train, projected_queries, y, retained_rank):
    """No per-axis restandardization after the supervised projection."""
    mz, state = task.fit_blocks([m_train])
    mq = task.apply_blocks([m_query], state)
    km, hm, _ = task.component_kernel(mz, mq, 'rbf')
    if len(projected_queries) != 3:
        raise ValueError('same three query-generator states required')
    if retained_rank:
        distance = cdist(projected_train, projected_train, 'sqeuclidean')
        upper = distance[np.triu_indices(len(distance), 1)]
        positive = upper[upper > 0]
        if not len(positive):
            raise ValueError('nonzero-rank projection has degenerate training distances')
        width = float(np.median(positive))
        kernel = (km + np.exp(-distance / (2 * width))) / 2
        crosses = [(hm + np.exp(-cdist(q, projected_train, 'sqeuclidean') / (2 * width))) / 2
                   for q in projected_queries]
    else:
        kernel, crosses = km, [hm, hm, hm]
    components = np.stack([np.stack([masked.predict_masked(kernel, h, y, a) for a in ALPHAS]) for h in crosses])
    if not np.isfinite(components).all():
        raise ValueError('nonfinite task-metric predictions')
    return components.mean(0), components


def run_bank(parent, source, prefix, fit, query, panel, root):
    root.mkdir()
    selected = parent.json(prefix + '/selection.json')
    np.testing.assert_array_equal(selected['fit'], fit)
    np.testing.assert_array_equal(selected['query'], query)
    if set(fit) & set(query) or set(source['tissues'][fit]) & set(source['tissues'][query]):
        raise ValueError('task bank tissue/query overlap')
    train, queries = parent.npz(prefix + '/train-features.npz'), parent.npz(prefix + '/query-features.npz')
    oof = baseline_oof(parent, source, prefix, fit, panel, selected)
    save_npz(root / 'baseline-oof.npz', oof_m=oof)
    train_y = source['auc'][np.ix_(fit, panel)]
    projected, projected_query, models = {}, {}, {}
    for arm, original in ARM_MAP.items():
        models[arm] = metric.fit_metric(train[original], train_y, oof)
        projected[arm] = metric.transform_metric(models[arm], train[original])
        projected_query[arm] = np.stack([metric.transform_metric(models[arm], q) for q in queries[original]])
        save_npz(root / ('metric-' + arm + '.npz'), **models[arm])
    save_npz(root / 'train-features.npz', **projected)
    save_npz(root / 'query-features.npz', **projected_query)
    save_json(root / 'upstream-frozen.json', {p.name: sha(p) for p in root.iterdir() if p.is_file()})
    predictions, components = {}, {}
    for arm in ARMS:
        predictions[arm], components[arm] = projected_grid(source['m'][fit], source['m'][query],
            projected[arm], projected_query[arm], train_y, int(models[arm]['retained_rank']))
    save_npz(root / 'per-alpha-predictions.npz', **predictions)
    save_npz(root / 'query-generator-predictions.npz', **components)
    save_json(root / 'predictions-frozen.json', {n: sha(root / n) for n in
        ('per-alpha-predictions.npz', 'query-generator-predictions.npz')})
    return predictions


def validation(predictions, truth, scale):
    if not np.isfinite(scale).all() or np.any(scale <= 1e-8):
        raise ValueError('nondegenerate head-training scales required')
    values = {}
    for arm in ARMS:
        error = ((predictions[arm] - truth[None]) / scale) ** 2
        counts = np.isfinite(error).sum(1)
        values[arm + '_loss'] = np.divide(np.nansum(error, axis=1), counts,
            out=np.zeros_like(counts, dtype=float), where=counts > 0)
        values[arm + '_count'] = (counts > 0).astype(int)
    return values


def run_fold(parent, source, fold, root):
    root.mkdir()
    prefix = f'fold-{fold}'
    selection = parent.json(prefix + '/selection.json')
    fit, held, panel = [np.array(selection[k], int) for k in ('fit', 'held', 'panel')]
    inherited = parent.npz(prefix + '/predictions.npz')
    for key, value in dict(fit=fit, held=held, panel=panel, drug_ids=source['drug_ids'][panel]).items():
        np.testing.assert_array_equal(inherited[key], value)
    np.testing.assert_array_equal(fit, np.flatnonzero(source['fold_ids'] != fold))
    np.testing.assert_array_equal(held, np.flatnonzero(source['fold_ids'] == fold))
    totals = {a + s: np.zeros((4, len(panel))) for a in ARMS for s in ('_loss', '_count')}
    for h, split in enumerate(selection['head_splits']):
        a, b = np.array(split['fit'], int), np.array(split['held'], int)
        sub = root / f'inner-{h}'
        grids = run_bank(parent, source, prefix + f'/inner-{h}', a, b, panel, sub)
        vals = validation(grids, source['auc'][np.ix_(b, panel)], np.nanstd(source['auc'][np.ix_(a, panel)], axis=0))
        save_npz(sub / 'validation.npz', **vals)
        for key in totals:
            totals[key] += vals[key]
    chosen = {}
    for arm in ARMS:
        if not (totals[arm + '_count'] > 0).all():
            raise ValueError('missing inner drug validation evidence')
        chosen[arm] = np.argmin(totals[arm + '_loss'] / totals[arm + '_count'], axis=0)
    save_npz(root / 'selected.npz', **chosen, **totals)
    grids = run_bank(parent, source, prefix + '/outer', fit, held, panel, root / 'outer')
    pred = {a: np.take_along_axis(grids[a], chosen[a][None, None, :], axis=0)[0] for a in ARMS}
    save_npz(root / 'predictions.npz', fit=fit, held=held, panel=panel, drug_ids=source['drug_ids'][panel], **pred)
    return dict(fold=fold, predictions_sha=sha(root / 'predictions.npz'))


def score_all(source, output, original):
    rows, per_drug, diagnostics = [], [], []
    seen = np.zeros(N_SAMPLES, int)
    for fold in range(5):
        pred = arrays(output / f'fold-{fold}/predictions.npz')
        fit, held, panel = [pred[k] for k in ('fit', 'held', 'panel')]
        seen[held] += 1
        truth = source['auc'][np.ix_(held, panel)]
        scale = np.nanstd(source['auc'][np.ix_(fit, panel)], axis=0)
        counts = np.isfinite(truth).sum(0)
        eligible = counts >= 5
        if not eligible.any():
            raise ValueError('empty original evaluation panel')
        for arm in ARMS:
            mse = np.divide(np.nansum(((pred[arm] - truth) / scale) ** 2, axis=0), counts,
                            out=np.full(len(panel), np.nan), where=counts > 0)
            rows.append(dict(fold=fold, arm=arm, score=float(np.sqrt(mse[eligible].mean())),
                drug_ids=int(eligible.sum()), observed_pairs=int(counts[eligible].sum())))
            for j, drug in enumerate(pred['drug_ids']):
                per_drug.append(dict(fold=fold, arm=arm, drug_id=str(drug), observed=int(counts[j]),
                    evaluated=bool(eligible[j]), standardized_mse=float(mse[j]) if counts[j] else None))
        for bank in ('inner-0', 'inner-1', 'inner-2', 'outer'):
            for arm in ARMS:
                model = arrays(output / f'fold-{fold}/{bank}/metric-{arm}.npz')
                diagnostics.append(dict(fold=fold, bank=bank, arm=arm, retained_rank=int(model['retained_rank']),
                    numeric_rank=int(model['numeric_rank']), tie_truncated=bool(model['tie_truncated']),
                    eligible_drugs=int(model['eligible'].sum())))
    if not np.all(seen == 1):
        raise ValueError('exact five-fold prediction coverage required')
    means = {a: float(np.mean([r['score'] for r in rows if r['arm'] == a])) for a in ARMS}
    lookup = {(r['fold'], r['arm']): r['score'] for r in rows}
    for name in ('m_only', 'virtual_fm', 'virtual_rna'):
        means['parent_' + name] = original['mean_fold_scores'][name]
        lookup.update({(r['fold'], 'parent_' + name): r['score'] for r in original['fold_metrics'] if r['arm'] == name})
    groups = dict(fm=('adapted_fm', ('adapted_rna', 'adapted_ordinary_m', 'adapted_shuffle_rna',
        'adapted_shuffle_fm', 'parent_m_only', 'parent_virtual_fm')),
        rna=('adapted_rna', ('parent_virtual_rna', 'parent_m_only', 'adapted_ordinary_m', 'adapted_shuffle_rna')))
    gates = dict(fm_mean={}, fm_sensitivity={}, rna_mean={})
    comparisons = []
    for group, (candidate, controls) in groups.items():
        for control in controls:
            effects = [lookup[f, control] - lookup[f, candidate] for f in range(5)]
            loo = [float(np.mean([v for j, v in enumerate(effects) if j != f])) for f in range(5)]
            delta = means[control] - means[candidate]
            gates[group + '_mean'][control] = bool(delta > MARGIN)
            if group == 'fm':
                gates['fm_sensitivity'][control] = bool(all(v > MARGIN for v in loo))
            comparisons.append(dict(group=group, candidate=candidate, control=control,
                mean_control_minus_candidate=delta, fold_effects=effects, folds_better=sum(v > MARGIN for v in effects),
                leave_one_fold_out_effects=loo))
    return dict(mean_fold_scores=means, fold_metrics=rows, per_drug=per_drug, comparisons=comparisons,
                projection_diagnostics=diagnostics), gates


def run(*, parent, parent_verification, source_bindings, output):
    start = time.perf_counter()
    bound = Parent(parent, parent_verification, source_bindings)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    code = Path(__file__).resolve().parents[1]
    paths = dict(runner=__file__, metric=metric.__file__, task=task.__file__, masked=masked.__file__,
                 io=code / 'scripts/run_phospho_fewshot.py', arrays=code / 'scripts/run_gdsc_metabolite_query.py')
    config = dict(schema='creed.gdsc-metabolite-task-metric.development.v1', scientific_claim=False,
        external_confirmation=False, chainable=False, samples=N_SAMPLES, arms=list(ARMS), arm_map=ARM_MAP,
        rank_limit=16, projection_lambda=1., residual_baseline_alpha=1., residual_min_observed=20,
        alphas=list(ALPHAS), extra_kernel_weight=.5, comparison_margin=MARGIN,
        projection_post_axis_standardization=False, original_generators_retrained=False,
        source_bindings_sha256=BINDINGS_SHA, parent_summary_sha256=bound.binding['parent_summary_sha256'],
        parent_verification_sha256=bound.binding['parent_verification_sha256'],
        protocol_sha256=sha(code / 'GDSC_METABOLITE_TASK_METRIC_NEXT.md'),
        code_sha256={k: sha(v) for k, v in paths.items()})
    save_json(output / 'frozenconfig.json', config)
    save_json(output / 'bindings.json', bound.binding)
    source = load_source(bound)
    records = []
    for fold in range(5):
        record = run_fold(bound, source, fold, output / f'fold-{fold}')
        records.append(record)
        print(str(dict(**record, elapsed_seconds=time.perf_counter() - start)), flush=True)
    save_json(output / 'all-predictions-frozen.json', {str(r['fold']): r['predictions_sha'] for r in records})
    scores, gates = score_all(source, output, bound.summary)
    save_json(output / 'scores.json', scores)
    bound.finish()
    if set(bound.used) != allowed_parent_files():
        raise ValueError('exact parent read inventory incomplete')
    result = dict(**config, status='completed', **scores, gates=gates,
        fm_gate_passed=all(gates['fm_mean'].values()), fm_fold_sensitivity_passed=all(gates['fm_sensitivity'].values()),
        rna_gate_passed=all(gates['rna_mean'].values()),
        global_gate_passed=all(gates['fm_mean'].values()) and all(gates['fm_sensitivity'].values()),
        fold_records=records, parent_members=bound.used, parent_fields=bound.fields,
        frozenconfig_sha256=sha(output / 'frozenconfig.json'),
        artifacts={str(p.relative_to(output)).replace('\\', '/'): sha(p) for p in output.rglob('*') if p.is_file()},
        caveats=['task-supervised training geometry, not end-to-end label-free OOF features',
                 'all identities and outcomes are adaptively reused development, not external confirmation',
                 'fold omission is a sensitivity screen, not a formal statistical test',
                 'no full RNA, L0-L7, FM mechanism or NMI completion from an engineering pass'],
        elapsed_seconds=time.perf_counter() - start)
    save_json(output / 'summary.json', result)
    print(str(dict(status='completed', mean_fold_scores=scores['mean_fold_scores'], gates=gates,
                   elapsed_seconds=result['elapsed_seconds'])), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('parent', 'parent-verification', 'source-bindings', 'output'):
        parser.add_argument('--' + key, required=True)
    run(**vars(parser.parse_args()))

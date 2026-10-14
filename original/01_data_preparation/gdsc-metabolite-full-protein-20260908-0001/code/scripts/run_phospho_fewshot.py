"""Known-cohort RNA to representative phospho-entity few-shot transfer.

Entities and five support patients are selected by identities before target
values are examined. Novel-parent training labels outside those five rows are
never supplied to fitting, eligibility, references, scores or saved inputs.
Phosphorylation entities remain L2 measurements; cohorts do not align sites.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np


COHORTS = ('brca_cptac_2020', 'luad_cptac_2020', 'ucec_cptac_2020')
BLOCK_HASHES = dict(zip(COHORTS, (
    '8e65f0573edfba01f83b094b630f17d074d80feca623ae70e5e273979ea0bfd6',
    '4d5879681c305a1563260e03b20d42e974ffe634f4b5c6e8b3e1fdc62945b525',
    'e54338d68e401a90d98ae63d7fd6b5e88cb4daf8ccb578f1b2982323d8916083')))
RAW_HASHES = dict(zip(COHORTS, (
    'aef2c0d99e94a3b93a5aef12a4d98a300d7fbaa19e84645d6e7e89fa831a533d',
    'c829b182a33274dd35983f1e59c38f02c49d9f6619daf80b8f3c89c9a9646f13',
    '8536e9925369a980795205fa33b088a8d9373770d0c04e0cc263fb9404c27759')))
COUNTS = dict(zip(COHORTS, (122, 110, 95)))
INPUT_SHA = '1b8b3eeb7e20c55c33ca788ca8a025213de7dbb003d21de5bbe36efc94b25bfd'
RNA_SUMMARY_SHA = 'b55c48d3b233b4d071d0820b0d4ec615c6acc5b9da0dc830a8aa15e2a19dfcac'
HGNC_SHA = 'e73e9259177884b5994fc81ed733c1b3d4df34c84290bc9dddc86e960d5d6419'
REACTOME_SHA = 'e34fe56a5129d88d8cd67300c279d7e81fa1d3bf86fa0577cf91032c326eaecc'
ESM_HASHES = {
    'receipt.json': 'df42c8c3f9eac2734b7851d3af7f39231af650853c7607aed08267f3e1588c75',
    'esm2_embeddings.npy': '757d78cb7ac8cef045ab979daa337f7938dc4667cc7aa296aa23061db429b37d',
    'present_mask.npy': '251a5f92134b9f9b6d58bf3ea0cbbd958962bec2499d624651acec95527876b5',
    'uniprot_accessions.json': '0f0f1f561feb6c5f036f3fd56a9fedd459156143338f09bdb4e87057f9011791',
}
ARMS = ('support_mean', 'direct', 'mean_prior', 'semantic', 'shuffle_esm', 'shuffle_reactome')
BASELINES = ('direct', 'mean_prior', 'shuffle_esm', 'shuffle_reactome', 'support_mean')
ENTITY_SALT = 'creed-phospho-entity-v1'
SUPPORT_SALT = 'creed-phospho-support-v1|'
SEED = 20260909


def math_api():
    from creed.science import phospho_fewshot
    if tuple(phospho_fewshot.ARMS) != ARMS:
        raise ValueError('few-shot math arm contract changed')
    return phospho_fewshot


def sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('source must be a regular non-symlink file')
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def save_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(jsonable(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def save_npz(path, **values):
    arrays = {key: np.asarray(value) for key, value in values.items()}
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError('saved arrays must not require pickle')
    with Path(path).open('xb') as handle:
        np.savez_compressed(handle, **arrays)


def approved_hgnc(path):
    approved = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            if row['status'] != 'Approved':
                continue
            symbol = row['symbol']
            if symbol in approved:
                raise ValueError('duplicated approved HGNC symbol')
            approved[symbol] = tuple(value for value in row['uniprot_ids'].split('|') if value)
    return approved


def global_esm_mapping(approved, accessions, present):
    """Resolve unique available accessions globally before any cohort subset."""
    if len(accessions) != len(present) or len(set(accessions)) != len(accessions):
        raise ValueError('ESM accession identity mismatch')
    available = {accession: index for index, accession in enumerate(accessions) if present[index]}
    candidates, multiple = {}, []
    for symbol, values in approved.items():
        matches = sorted({available[value] for value in values if value in available})
        if len(matches) == 1:
            candidates[symbol] = matches[0]
        elif len(matches) > 1:
            multiple.append(symbol)
    counts = Counter(candidates.values())
    mapping = {symbol: index for symbol, index in candidates.items() if counts[index] == 1}
    return mapping, dict(globally_unique_available_parents=len(mapping),
        multiple_available_accession_genes=len(multiple),
        shared_available_accession_genes=sum(counts[index] > 1 for index in candidates.values()))


def human_pathways(path, approved):
    accession_genes = defaultdict(set)
    for symbol, accessions in approved.items():
        for accession in accessions:
            accession_genes[accession].add(symbol)
    result, human_rows = defaultdict(set), 0
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.reader(handle, delimiter='\t'):
            if len(row) != 6:
                raise ValueError('Reactome source must have six columns')
            if row[5] != 'Homo sapiens':
                continue
            human_rows += 1
            symbols = accession_genes.get(row[0], set())
            if len(symbols) == 1:
                result[next(iter(symbols))].add(row[1])
    return dict(result), human_rows


def read_parent_metadata(path, feature_ids):
    """Decode identity columns only, not numeric sample fields in raw tables."""
    entities, parents = [], []
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        header = handle.readline().rstrip('\r\n').split('\t')
        if len(set(header)) != len(header):
            raise ValueError('duplicate raw phospho columns')
        entity_column, gene_column = header.index('ENTITY_STABLE_ID'), header.index('GENE_SYMBOL')
        last = max(entity_column, gene_column)
        for line in handle:
            if not line.strip():
                continue
            # At most the leading annotation fields are split; the numerical
            # remainder is never parsed or used for eligibility or modelling.
            fields = line.rstrip('\r\n').split('\t', last + 1)
            if len(fields) <= last:
                raise ValueError('ragged phospho identity fields')
            entities.append(fields[entity_column].strip())
            parents.append(fields[gene_column].strip())
    np.testing.assert_array_equal(entities, feature_ids)
    if len(set(entities)) != len(entities):
        raise ValueError('duplicate raw phospho entity identity')
    return np.asarray(parents)


def select_entities(feature_ids, parent_symbols, approved, esm_mapping):
    """One metadata-hash-selected entity per exact approved parent, no labels."""
    api = math_api()
    feature_ids, parent_symbols = np.asarray(feature_ids), np.asarray(parent_symbols)
    if feature_ids.shape != parent_symbols.shape or feature_ids.ndim != 1 or len(set(feature_ids)) != len(feature_ids):
        raise ValueError('entity/parent metadata identity mismatch')
    by_parent = defaultdict(list)
    unknown_rows = 0
    for entity, parent in zip(feature_ids, parent_symbols):
        if parent in approved:
            by_parent[str(parent)].append(str(entity))
        else:
            unknown_rows += 1
    lookup = {str(entity): index for index, entity in enumerate(feature_ids)}
    selected, missing_esm = [], []
    for parent in sorted(by_parent):
        chosen = api.stable_order(by_parent[parent], ENTITY_SALT)[0]
        row = dict(parent=parent, entity_id=chosen, source_feature_index=lookup[chosen],
            entity_candidates=len(by_parent[parent]), heldout_parent=bool(api.parent_is_heldout(parent)))
        # Entity choice is fixed even for parents later excluded by static
        # availability or target eligibility; a better-observed site is never substituted.
        if parent in esm_mapping:
            selected.append(row)
        else:
            missing_esm.append(row)
    return selected, dict(raw_entities=len(feature_ids), approved_parents=len(by_parent),
        unknown_or_unapproved_entity_rows=unknown_rows, missing_unique_static_esm_parents=missing_esm,
        retained_static_parents=len(selected), entity_selection_salt=ENTITY_SALT)


def choose_support(patient_ids, cohort):
    patient_ids = [str(value) for value in patient_ids]
    if len(patient_ids) != len(set(patient_ids)) or len(patient_ids) < 5:
        raise ValueError('five unique training support patients are required')
    ids = math_api().stable_order(patient_ids, SUPPORT_SALT + cohort)[:5]
    return np.array([patient_ids.index(value) for value in ids], dtype=int), ids


def target_moments(values):
    values = np.asarray(values, float)
    if values.ndim != 2 or np.isinf(values).any():
        raise ValueError('target inputs require finite or missing values')
    seen = np.isfinite(values)
    count = seen.sum(0)
    mean = np.divide(np.where(seen, values, 0.).sum(0), count,
                     out=np.zeros(values.shape[1]), where=count > 0)
    delta = np.where(seen, values - mean, 0.)
    sd = np.sqrt(np.divide((delta * delta).sum(0), count,
                          out=np.zeros(values.shape[1]), where=count > 0))
    return count, mean, sd


def eligible_inputs(base_y, support_y):
    """No novel target's non-support training values can enter this function."""
    if len(support_y) != 5:
        raise ValueError('support set must contain exactly five predetermined rows')
    base_count, base_mean, base_sd = target_moments(base_y)
    support_count, support_mean, support_sd = target_moments(support_y)
    return dict(base_mask=(base_count >= 20) & (base_sd > 1e-8),
        novel_mask=(support_count >= 3) & (support_sd > 1e-8),
        base_count=base_count, base_mean=base_mean, base_sd=base_sd,
        support_count=support_count, support_mean=support_mean, support_sd=support_sd)


def extract_allowed_targets(values, observed, censor, selected, training_rows, support_rows, query_rows):
    """Extract only base training, novel five-support and novel query targets.

    The in-memory source archive may contain other labels; these are never
    returned or examined by any target statistic or model call.
    """
    values, observed, censor = np.asarray(values), np.asarray(observed), np.asarray(censor)
    if values.shape != observed.shape or values.shape != censor.shape or observed.dtype != np.bool_:
        raise ValueError('phospho value/observed/censor schema mismatch')
    base = [row for row in selected if not row['heldout_parent']]
    novel = [row for row in selected if row['heldout_parent']]
    base_columns = np.array([row['source_feature_index'] for row in base], dtype=int)
    novel_columns = np.array([row['source_feature_index'] for row in novel], dtype=int)
    def subset(rows, columns):
        ix = np.ix_(np.asarray(rows, int), columns)
        raw, seen, censored = values[ix], observed[ix], censor[ix]
        if not np.array_equal(seen, np.isfinite(raw) & (censored == 0)):
            raise ValueError('selected source observed mask is inconsistent')
        return np.where(seen, raw, np.nan)
    return base, novel, subset(training_rows, base_columns), subset(support_rows, novel_columns), subset(query_rows, novel_columns)


def pathway_arrays(base_parents, novel_parents, annotations):
    pathway_ids = sorted(set().union(*(annotations.get(parent, set()) for parent in list(base_parents) + list(novel_parents))))
    if not pathway_ids:
        raise ValueError('no Reactome pathway features for eligible parent identities')
    lookup = {value: index for index, value in enumerate(pathway_ids)}
    def encode(parents):
        values = np.zeros((len(parents), len(pathway_ids)), dtype=np.float64)
        for row, parent in enumerate(parents):
            for pathway in annotations.get(parent, ()):
                values[row, lookup[pathway]] = 1.
        return values
    return encode(base_parents), encode(novel_parents), pathway_ids


def score_predictions(prediction, truth, support_sd, parents, entities):
    prediction, truth, support_sd = np.asarray(prediction), np.asarray(truth), np.asarray(support_sd)
    if prediction.shape != truth.shape or truth.ndim != 2 or support_sd.shape != (truth.shape[1],):
        raise ValueError('few-shot score shape mismatch')
    if not np.isfinite(prediction).all() or np.isinf(truth).any() or not np.isfinite(support_sd).all() or np.any(support_sd <= 1e-8):
        raise ValueError('invalid prediction or support-fitted scale')
    if len(parents) != len(set(parents)) or len(parents) != truth.shape[1] or len(entities) != len(parents):
        raise ValueError('scores require one unique entity per parent')
    records, native, standardized = [], [], []
    for column, parent in enumerate(parents):
        observed = np.isfinite(truth[:, column])
        n = int(observed.sum())
        rmse = float(np.sqrt(np.mean((prediction[observed, column] - truth[observed, column]) ** 2))) if n else None
        normalized = rmse / float(support_sd[column]) if n else None
        if n:
            native.append(rmse); standardized.append(normalized)
        records.append(dict(parent=str(parent), entity_id=str(entities[column]), query_observations=n,
            native_rmse=rmse, support_standardized_rmse=normalized))
    return dict(query_patients=len(truth), eligible_novel_parents=truth.shape[1],
        parents_with_query_observations=len(native), parents_without_query_observations=truth.shape[1] - len(native),
        observed_query_entries=int(np.isfinite(truth).sum()),
        mean_parent_native_rmse=float(np.mean(native)) if native else None,
        mean_parent_support_standardized_rmse=float(np.mean(standardized)) if standardized else None,
        per_parent=records, primary_definition='Mean of per-entity native RMSE, one entity per parent; not pooled-entry RMSE.',
        missing_query_policy='Targets without an observed query value are unscorable, never reselected; eligibility and coverage retained.')


def strict_gates(cohort_records):
    per_cohort = {}
    for cohort in COHORTS:
        record = cohort_records.get(cohort, {})
        scores = record.get('scores', {})
        semantic = scores.get('semantic', {}).get('mean_parent_native_rmse')
        per_cohort[cohort] = {arm: bool(semantic is not None and scores.get(arm, {}).get('mean_parent_native_rmse') is not None
            and semantic < scores[arm]['mean_parent_native_rmse']) for arm in BASELINES}
    return dict(per_cohort=per_cohort,
        all_three_cohorts_all_five_baselines=all(flag for row in per_cohort.values() for flag in row.values()),
        interpretation='Strict fixed development comparison, not significance, independent confirmation or full CREED success.')


def run(*, rna, calibration, blocks, raw, esm, hgnc, reactome, output):
    started = time.perf_counter()
    api = math_api()
    rna, calibration, blocks, raw, esm, output = [Path(path).absolute() for path in (rna, calibration, blocks, raw, esm, output)]
    hgnc, reactome = Path(hgnc).absolute(), Path(reactome).absolute()
    output.mkdir(parents=True, exist_ok=False)
    expected = {rna / 'inputs.npz': INPUT_SHA, rna / 'summary.json': RNA_SUMMARY_SHA,
                hgnc: HGNC_SHA, reactome: REACTOME_SHA}
    expected.update({esm / name: value for name, value in ESM_HASHES.items()})
    expected.update({blocks / (cohort + '.npz'): BLOCK_HASHES[cohort] for cohort in COHORTS})
    expected.update({raw / cohort / 'data_phosphoprotein_quantification.txt': RAW_HASHES[cohort] for cohort in COHORTS})
    for path, value in expected.items():
        if sha(path) != value:
            raise ValueError('frozen source hash mismatch: ' + path.name)
    calibration_summary, split = read_json(calibration / 'summary.json'), read_json(calibration / 'splits.json')
    if calibration_summary['status'] != 'completed' or calibration_summary['scientific_claim'] is not False:
        raise ValueError('existing development calibration status mismatch')
    if calibration_summary['input_bundle_sha256'] != INPUT_SHA or calibration_summary['splits_sha256'] != sha(calibration / 'splits.json'):
        raise ValueError('existing 261/66 split binding mismatch')
    expected[calibration / 'summary.json'] = sha(calibration / 'summary.json')
    expected[calibration / 'splits.json'] = sha(calibration / 'splits.json')
    allowed_rna_keys = ('rna', 'rna_ids', 'patient_ids', 'sample_ids', 'cohort')
    with np.load(rna / 'inputs.npz', allow_pickle=False) as archive:
        data = {key: archive[key] for key in allowed_rna_keys}
    if data['rna'].shape != (327, 30670) or len(set(data['patient_ids'])) != 327 or len(set(data['rna_ids'])) != 30670:
        raise ValueError('frozen RNA dimensions/identities mismatch')
    np.testing.assert_array_equal(data['patient_ids'], split['patients'])
    np.testing.assert_array_equal(data['cohort'], split['cohort_ids'])
    training, query = np.asarray(split['training_indices']), np.asarray(split['validation_indices'])
    if len(training) != 261 or len(query) != 66 or set(training) & set(query) or set(training) | set(query) != set(range(327)):
        raise ValueError('only the existing disjoint 261/66 development split is permitted')
    np.testing.assert_array_equal(data['patient_ids'][training], split['training_patient_ids'])
    np.testing.assert_array_equal(data['patient_ids'][query], split['validation_patient_ids'])
    if set(data['cohort']) != set(COHORTS):
        raise ValueError('unexpected cohort')
    approved = approved_hgnc(hgnc)
    receipt = read_json(esm / 'receipt.json')
    for name in ('esm2_embeddings.npy', 'present_mask.npy', 'uniprot_accessions.json'):
        if receipt['outputs'][name]['sha256'] != ESM_HASHES[name]:
            raise ValueError('ESM receipt/known output identity mismatch')
        if receipt['outputs'][name]['bytes'] != (esm / name).stat().st_size:
            raise ValueError('ESM output size mismatch')
    embeddings = np.load(esm / 'esm2_embeddings.npy', allow_pickle=False)
    present = np.load(esm / 'present_mask.npy', allow_pickle=False)
    accessions = read_json(esm / 'uniprot_accessions.json')
    if embeddings.shape != (6692, 1280) or present.shape != (6692,) or present.dtype != np.bool_ or int(present.sum()) != 6691:
        raise ValueError('ESM projection dimensions/coverage changed')
    if not np.isfinite(embeddings[present]).all() or not np.isnan(embeddings[~present]).all():
        raise ValueError('ESM presence mask and values disagree')
    esm_mapping, esm_audit = global_esm_mapping(approved, accessions, present)
    if len(esm_mapping) != 6673:
        raise ValueError('global unique-parent ESM mapping changed')
    pathways, human_rows = human_pathways(reactome, approved)
    if human_rows != 54699:
        raise ValueError('Reactome human source row count changed')
    config = dict(schema='creed.phospho-parent-fewshot.development.v1', scientific_claim=False,
        biological_layer='L2 phosphoprotein measurement; no L3 assertion', cohorts=list(COHORTS),
        arms=list(ARMS), primary_arm='semantic', primary_baselines=list(BASELINES), seed=SEED,
        parent_split='int(SHA256(creed-phospho-parent-v1|exact approved HGNC symbol),16) % 5 == 0 is held-out; all entities of a held parent excluded from base',
        entity_selection='One exact ENTITY_STABLE_ID per cohort/parent, smallest stable metadata hash before labels; never replace an ineligible selection.',
        entity_salt=ENTITY_SALT, support_salt_prefix=SUPPORT_SALT, support_patients_per_cohort=5,
        support_selection='Fixed five lowest-hash training patient IDs per cohort, shared by all novel targets; no replacement or query use.',
        base_eligibility='At least 20 finite training targets and population SD > 1e-8.',
        novel_eligibility='At least 3 finite labels and population SD > 1e-8 among the fixed five support patients only.',
        design='intercept + own-parent RNA + own-RNA-missing indicator + shared training-only RNA PCA16; cohort-specific preprocessing',
        prior='ESM32 and Reactome32 feature projections fitted only on eligible base parents; ridge slope prior and support adaptation, lambda=1, unpenalized intercept.',
        semantic_identity='Frozen parent-protein sequence embedding plus positive exact-human Reactome membership; not measured phosphosite state.',
        primary_metric='Within each cohort, mean of per-entity native RMSE, one representative entity per parent; all three cohorts must strictly beat all five controls.',
        secondary_metric='Same per-entity errors divided by that target fixed-support SD, then averaged across scorable parents.',
        no_validation_selection=True, no_new_patient_split=True, no_new_uncertainty=True,
        sources={str(path): value for path, value in expected.items()}, esm_identity_audit=esm_audit,
        approved_hgnc_parents=len(approved), reactome_human_rows=human_rows,
        code_sha256=sha(Path(__file__)), math_sha256=sha(Path(api.__file__)),
        boundaries=['Only the three frozen known development cohorts; no GBM, audit64, final123 or D4.',
            'Novel-parent non-support training labels do not enter fitting, eligibility, normalization, saved training inputs or scoring.',
            'Query labels are separate scoring truth, not model arguments or a selection source.',
            'Entities differ across cohorts, and same parent does not establish shared phosphosite identity.',
            'No independent, clinical, causal, physical-harmonization, full-agent or full CREED claim.'])
    save_json(output / 'frozenconfig.json', config)
    save_json(output / 'splits.json', split)
    records = {}
    for cohort in COHORTS:
        root = output / cohort
        root.mkdir()
        cohort_all = np.flatnonzero(data['cohort'] == cohort)
        fit_global = training[data['cohort'][training] == cohort]
        query_global = query[data['cohort'][query] == cohort]
        local_lookup = {int(row): index for index, row in enumerate(cohort_all)}
        fit_local = np.array([local_lookup[int(row)] for row in fit_global], dtype=int)
        query_local = np.array([local_lookup[int(row)] for row in query_global], dtype=int)
        support_indices, support_ids = choose_support(data['patient_ids'][fit_global], cohort)
        support_local = fit_local[support_indices]
        block_path = blocks / (cohort + '.npz')
        with np.load(block_path, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive['patient_ids'], data['patient_ids'][cohort_all])
            np.testing.assert_array_equal(archive['sample_ids'], data['sample_ids'][cohort_all])
            if len(cohort_all) != COUNTS[cohort]:
                raise ValueError('cohort source population changed')
            feature_ids = archive['phosphoprotein__feature_ids']
            parents = read_parent_metadata(raw / cohort / 'data_phosphoprotein_quantification.txt', feature_ids)
            selected, metadata_audit = select_entities(feature_ids, parents, approved, esm_mapping)
            # Persist label-independent choices before decoding target arrays.
            save_json(root / 'identity-selection.json', dict(cohort=cohort, selected=selected,
                audit=metadata_audit, training_patient_ids=data['patient_ids'][fit_global],
                query_patient_ids=data['patient_ids'][query_global], support_patient_ids=support_ids,
                support_indices_within_training=support_indices, source_block_sha256=BLOCK_HASHES[cohort],
                raw_parent_metadata_sha256=RAW_HASHES[cohort]))
            base, novel, base_y, support_y, query_truth = extract_allowed_targets(
                archive['phosphoprotein__values'], archive['phosphoprotein__observed'], archive['phosphoprotein__censor'],
                selected, fit_local, support_local, query_local)
        eligibility = eligible_inputs(base_y, support_y)
        save_npz(root / 'eligibility-inputs.npz', base_y=base_y, support_y=support_y,
            base_parents=np.array([row['parent'] for row in base]),
            novel_parents=np.array([row['parent'] for row in novel]),
            base_entity_ids=np.array([row['entity_id'] for row in base]),
            novel_entity_ids=np.array([row['entity_id'] for row in novel]),
            training_patient_ids=data['patient_ids'][fit_global], support_patient_ids=np.array(support_ids))
        base_mask, novel_mask = eligibility['base_mask'], eligibility['novel_mask']
        base_eligible = [row for row, keep in zip(base, base_mask) if keep]
        novel_eligible = [row for row, keep in zip(novel, novel_mask) if keep]
        eligibility_record = dict(base_candidates=base, novel_candidates=novel,
            **eligibility, base_eligible=len(base_eligible), novel_eligible=len(novel_eligible),
            missing_query_values_not_used_for_eligibility=True,
            novel_non_support_training_labels_supplied=False, support_ids=support_ids)
        save_json(root / 'eligibility.json', eligibility_record)
        if len(base_eligible) < 33 or not novel_eligible:
            records[cohort] = dict(status='not_estimable', reason='Need at least 33 eligible base and one novel target; no resampling or relaxation.',
                base_eligible=len(base_eligible), novel_eligible=len(novel_eligible), support_patient_ids=support_ids)
            print(json.dumps(dict(cohort=cohort, status='not_estimable')), flush=True)
            continue
        base_parents, novel_parents = [row['parent'] for row in base_eligible], [row['parent'] for row in novel_eligible]
        if (set(base_parents) & set(novel_parents)
                or any(api.parent_is_heldout(parent) for parent in base_parents)
                or not all(api.parent_is_heldout(parent) for parent in novel_parents)):
            raise ValueError('parent-heldout boundary violated')
        pathway_base, pathway_novel, pathway_ids = pathway_arrays(base_parents, novel_parents, pathways)
        base_esm_indices = np.array([esm_mapping[parent] for parent in base_parents], dtype=int)
        novel_esm_indices = np.array([esm_mapping[parent] for parent in novel_parents], dtype=int)
        # Bind every saved feature row to its exact parent and accession, so an
        # independent replay can verify row identity, not just matrix dimensions.
        for parent, index in zip(base_parents + novel_parents, np.r_[base_esm_indices, novel_esm_indices]):
            if not present[index] or accessions[index] not in approved[parent]:
                raise ValueError('parent-to-ESM row identity mismatch')
        allowed = dict(rna_train=data['rna'][fit_global], rna_ids=data['rna_ids'],
            base_y=base_y[:, base_mask], base_parents=np.array(base_parents),
            support_y=support_y[:, novel_mask], novel_parents=np.array(novel_parents), support_indices=support_indices,
            esm_base=embeddings[base_esm_indices], esm_novel=embeddings[novel_esm_indices],
            pathway_base=pathway_base, pathway_novel=pathway_novel)
        save_npz(root / 'training-allowed.npz', **allowed,
            training_patient_ids=data['patient_ids'][fit_global], support_patient_ids=np.array(support_ids),
            base_entity_ids=np.array([row['entity_id'] for row in base_eligible]),
            novel_entity_ids=np.array([row['entity_id'] for row in novel_eligible]), pathway_ids=np.array(pathway_ids),
            base_esm_indices=base_esm_indices, novel_esm_indices=novel_esm_indices,
            base_esm_accessions=np.array(accessions)[base_esm_indices],
            novel_esm_accessions=np.array(accessions)[novel_esm_indices])
        save_npz(root / 'query-source.npz', rna_query=data['rna'][query_global], rna_ids=data['rna_ids'],
                 patient_ids=data['patient_ids'][query_global])
        # Save outcomes separately; they are not sent to the fitting function.
        truth = query_truth[:, novel_mask]
        save_npz(root / 'query-truth.npz', truth=truth, patient_ids=data['patient_ids'][query_global],
            novel_parents=np.array(novel_parents), entity_ids=np.array([row['entity_id'] for row in novel_eligible]),
            support_mean=eligibility['support_mean'][novel_mask], support_sd=eligibility['support_sd'][novel_mask])
        fitted = api.fit_predict_fewshot(**allowed, rna_query=data['rna'][query_global], seed=SEED)
        if set(fitted['predictions']) != set(ARMS):
            raise ValueError('few-shot prediction arm set changed')
        save_npz(root / 'model.npz', **fitted['model'])
        with np.load(root / 'model.npz', allow_pickle=False) as archive:
            saved_model = {key: archive[key] for key in archive.files}
        replay = api.predict_saved(saved_model, data['rna'][query_global], data['rna_ids'])
        scores = {}
        for arm in ARMS:
            predicted = np.asarray(fitted['predictions'][arm])
            if predicted.shape != truth.shape or not np.isfinite(predicted).all():
                raise ValueError('invalid few-shot query prediction')
            np.testing.assert_allclose(replay[arm], predicted, rtol=1e-9, atol=1e-10)
            scores[arm] = score_predictions(predicted, truth, eligibility['support_sd'][novel_mask], novel_parents,
                                            [row['entity_id'] for row in novel_eligible])
        save_npz(root / 'predictions.npz', **fitted['predictions'], patient_ids=data['patient_ids'][query_global],
            novel_parents=np.array(novel_parents), entity_ids=np.array([row['entity_id'] for row in novel_eligible]))
        save_json(root / 'diagnostics.json', fitted['diagnostics'])
        artifact_names = ('identity-selection.json', 'eligibility.json', 'eligibility-inputs.npz', 'training-allowed.npz', 'query-source.npz',
                          'query-truth.npz', 'model.npz', 'predictions.npz', 'diagnostics.json')
        records[cohort] = dict(status='completed', training_patients=len(fit_global), query_patients=len(query_global),
            base_eligible=len(base_eligible), novel_eligible=len(novel_eligible), support_patient_ids=support_ids,
            support_indices_within_training=support_indices.tolist(), support_count=5,
            novel_non_support_training_labels_supplied=False, all_eligible_query_targets_retained=True,
            static_parent_esm_present=True, base_parents_with_reactome=int(np.any(pathway_base, axis=1).sum()),
            novel_parents_with_reactome=int(np.any(pathway_novel, axis=1).sum()),
            pathway_features=len(pathway_ids), source_only_saved_model_replay=True, scores=scores,
            artifacts={name: dict(path=str((root / name).relative_to(output)).replace('\\', '/'), sha256=sha(root / name))
                       for name in artifact_names})
        print(json.dumps(dict(cohort=cohort, status='completed', base=len(base_eligible), novel=len(novel_eligible),
            primary_scores={arm: scores[arm]['mean_parent_native_rmse'] for arm in ARMS}, seconds=time.perf_counter()-started)), flush=True)
    for path, value in expected.items():
        if sha(path) != value:
            raise ValueError('frozen source changed during run')
    result = dict(**config, status='completed' if all(records[c]['status'] == 'completed' for c in COHORTS) else 'not_estimable',
        cohort_results=records, descriptive_primary_gates=strict_gates(records),
        frozenconfig_sha256=sha(output / 'frozenconfig.json'), elapsed_seconds=time.perf_counter()-started,
        caveats=['Known reused development cohorts and the old patient split, not an external validation.',
            'Five support labels per novel parent are additional supervision, not zero-shot biological inference.',
            'ESM describes a parent protein sequence, not a phosphosite measurement; Reactome is static annotation.',
            'Representative entities are cohort-specific and are not asserted to be the same site across studies.',
            'No L3, independent causal mechanism, clinical deployment, uncertainty calibration or full CREED claim.'])
    save_json(output / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('rna', 'calibration', 'blocks', 'raw', 'esm', 'hgnc', 'reactome', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    result = run(**vars(parser.parse_args()))
    print(json.dumps(dict(status=result['status'], gates=result['descriptive_primary_gates'])), flush=True)


if __name__ == '__main__':
    main()

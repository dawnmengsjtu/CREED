#!/usr/bin/env python3
"""Build Data S1-S10 (ZIP archives) of the Cell Systems package.  python3 -B _tools/build_data_zips.py

Every file is listed explicitly below (manuscript_cellsystems/data_files_plan.md describes the contents).
- Run records are byte-for-byte copies from the local read-only mirrors (<run>_20261009/, settle_stageA_20261011/,
  internal_final_20261009/, handoff_20261009/evidence/) or from data_exports_20261011/server_copies/ (files copied
  from the compute server on 2026-10-11). Each is placed under a short run folder at its path in the server run
  directory, and its SHA-256 is checked against data_exports_20261011/server_run_index.tsv (listing of the server run
  directories made on 2026-10-11), so that the run's own checksum lists can be checked from the run folder.
- Derived files (identity lists, panel membership, Bonferroni flag, translation) come from
  data_exports_20261011/{server_derived,derived}/ (see data_exports_20261011/README.md) and are labelled in the README.
- Each archive gets README.txt: title and description parsed from supplemental_information.md, notes, and the SHA-256
  of every file. Data S2 and S3 also get CHECKSUM_INDEX.tsv, the status of every entry of every checksum list.
- Output is deterministic: fixed timestamps and permissions, sorted entries, README first. Nothing is read from the
  server here; the build stops if any copy differs from the server listing or any stated fact fails its check.
"""
import collections, csv, hashlib, io, json, os, re, sys, textwrap, zipfile

ROOT = '/Users/xlzhexue/Desktop/proj/creed'
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PKG, '04_Data_S')
SI_MD = os.path.join(ROOT, 'manuscript_cellsystems', 'supplemental_information.md')
MAIN_MD = os.path.join(ROOT, 'manuscript_cellsystems', 'main_text.md')
EXP = 'data_exports_20261011/'
X = EXP + 'server_copies/'
ZIP_TIME = (2026, 10, 11, 0, 0, 0)
SCEN = ('new_cell', 'tissue')

# run folder in the archives -> server run directory (/public/home/<user>/creed-research-runs/<dir>/)
RUNS = {
    'crosslayer_matrix': 'crosslayer-matrix-20261009-0001', 'protein_panel': 'protein-panel-20261009-0001',
    'rna_panel': 'rna-panel-20261009-0001', 'acquisition_probe': 'acquisition-probe-20261009-0001',
    'acquisition_probe_rna': 'acquisition-probe-rna-20261009-0001',
    'probe2_context_abstain': 'probe2-context-abstain-20261009-0001', 'crispr_endpoint': 'crispr-endpoint-20261009-0001',
    'generator_curve': 'generator-curve-20261009-0001', 'generator_fullpool': 'generator-fullpool-20261009-0001',
    'null_baseline': None, 'pushline_checks': 'pushline-checks-20261009-0001',
    'tissue_kernel': 'tissue-kernel-20261009-0001', 'shared_crossdrug': 'shared-crossdrug-20261009-0001',
    'rna_learning_curve': 'rna-learning-curve-20261009-0001', 'settle_stageA': 'settle-stageA-20261011-0001',
    'internal_final': 'internal-final-20261009-0001',
    'a_protein': 'acquisition-probe-20261009-0001', 'b_gene': 'acquisition-probe-rna-20261009-0001',
    'development_protocols_20261008/recoverability': 'strong-teacher-recoverability-20261008-0001',
    'development_protocols_20261008/privileged_geometry': 'privileged-geometry-20261008-0001',
    'development_protocols_20261008/extra_supervision': 'extra-supervision-20261008-0001',
}

# ---------------------------------------------------------------------------------------------------- file lists
# An entry is (path inside Data_S#/, source relative to ROOT, origin); origin is ('server', run folder, server path),
# ('local', note) for records that exist only in the authors' copy, or ('derived', note).


def srv(folder, pairs):
    return [('%s/%s' % (folder, rel), src, ('server', folder, rel)) for rel, src in pairs]


def same(folder, local_dir, rels):
    return srv(folder, [(r, local_dir + r) for r in rels])


CL, PP, RP = 'crosslayer_matrix_20261009/', 'protein_panel_20261009/', 'rna_panel_20261009/'
AP, AR, P2 = 'acquisition_probe_20261009/', 'acquisition_probe_rna_20261009/', 'probe2_context_abstain_20261009/'
CE, GC, GF = 'crispr_endpoint_20261009/', 'generator_curve_20261009/', 'generator_fullpool_20261009/'
NB, PC, TK = 'null_baseline_20261009/', 'pushline_checks_20261009/', 'tissue_kernel_20261009/'
SC, RL, N6 = 'shared_crossdrug_20261009/', 'rna_learning_curve_20261009/', 'settle_stageA_20261011/'
IF, EV = 'internal_final_20261009/', 'handoff_20261009/evidence/'

S1 = ([('identity_split/original_identity_split.json', EV + 'original_identity_split.json',
        ('local', 'identity protocol record, frozen 2026-08-30'))]
      + [('whitelist_packages/%s' % f, EXP + 'server_derived/' + f, ('derived', 'identity fields of the package QA'))
         for f in ('extra19_M_AUC_ids.tsv', 'extra66_M_RNA_ids.tsv', 'extra354_RNA_AUC_ids.tsv')]
      + srv('generator_curve', [('output/extra_ids_%s_fold-%d.txt' % (s, f), GC + 'server_outputs/extra_ids_%s_fold-%d.txt' % (s, f))
                                for s in SCEN for f in range(5)])
      + [('raw_source_checksums.tsv', EXP + 'derived/raw_source_checksums.tsv', ('derived', 'checksums of the public source files the code read'))])

S2 = (srv('crosslayer_matrix', [('PLAN.md', CL + 'PLAN.md'),
                                ('PLAN_REGISTERED_AT.txt', CL + 'server_outputs/dryrun/PLAN_REGISTERED_AT.txt'),
                                ('SHA256SUMS', CL + 'server_outputs/SHA256SUMS'),
                                ('SHA256SUMS_WRITTEN_AT.txt', CL + 'server_outputs/SHA256SUMS_WRITTEN_AT.txt'),
                                ('run.log', CL + 'server_outputs/run.log'),
                                ('posthoc/SHA256SUMS_posthoc', CL + 'server_outputs/posthoc/SHA256SUMS_posthoc'),
                                ('posthoc/posthoc_saved_compare.json', CL + 'server_outputs/posthoc/posthoc_saved_compare.json')])
      + srv('protein_panel', [('PLAN.md', PP + 'PLAN.md'), ('SHA256SUMS', PP + 'SHA256SUMS_server.txt'),
                              ('OUTPUT_SHA256SUMS', X + 'protein-panel-20261009-0001/OUTPUT_SHA256SUMS'),
                              ('output/analysis.json', PP + 'summary.json')])
      + srv('rna_panel', [('PLAN.md', RP + 'PLAN.md')] + [(f, RP + 'server_outputs/' + f) for f in (
          'PLAN_REGISTERED_AT.txt', 'SHA256SUMS', 'SHA256SUMS_WRITTEN_AT.txt', 'OUTPUT_SHA256SUMS', 'output/summary.json')])
      + srv('acquisition_probe', [('PLAN.md', AP + 'PLAN.md'), ('SHA256SUMS', AP + 'SHA256SUMS_server.txt'),
                                  ('OUTPUT_SHA256SUMS', AP + 'OUTPUT_SHA256SUMS_server.txt'),
                                  ('output/new_cell_repro/summary.json', AP + 'repro_new_cell.json'),
                                  ('output/tissue_repro/summary.json', AP + 'repro_tissue.json'),
                                  ('output/analysis/summary.json', AP + 'summary.json')])
      + srv('acquisition_probe_rna', [('PLAN.md', AR + 'PLAN.md'),
                                      ('PLAN_REGISTERED_AT.txt', AR + 'server_outputs/PLAN_REGISTERED_AT.txt'),
                                      ('SHA256SUMS', AR + 'server_outputs/SHA256SUMS'),
                                      ('output/new_cell_repro/summary.json', AR + 'server_outputs/repro/new_cell_repro_summary.json'),
                                      ('output/tissue_repro/summary.json', AR + 'server_outputs/repro/tissue_repro_summary.json'),
                                      ('output/analysis/summary.json', AR + 'summary.json')])
      + srv('probe2_context_abstain', [('PLAN.md', P2 + 'PLAN.md'),
                                       ('PLAN_REGISTERED_AT.txt', P2 + 'server_outputs/PLAN_REGISTERED_AT.txt'),
                                       ('SHA256SUMS', P2 + 'server_outputs/SHA256SUMS'),
                                       ('SHA256SUMS_WRITTEN_AT.txt', P2 + 'server_outputs/SHA256SUMS_WRITTEN_AT.txt'),
                                       ('OUTPUT_SHA256SUMS', P2 + 'OUTPUT_SHA256SUMS_server.txt'),
                                       ('output/summary.json', P2 + 'server_outputs/summary.json')]
            + [('output/A/%s/fold-%d_repro.json' % (s, f), P2 + 'server_outputs/A/%s/fold-%d_repro.json' % (s, f))
               for s in SCEN for f in range(5)])
      + srv('crispr_endpoint', [('PLAN.md', CE + 'PLAN.md'),
                                ('PLAN_REGISTERED_AT.txt', X + 'crispr-endpoint-20261009-0001/PLAN_REGISTERED_AT.txt'),
                                ('SHA256SUMS', CE + 'SHA256SUMS'), ('OUTPUT_SHA256SUMS', CE + 'OUTPUT_SHA256SUMS'),
                                ('CODE_FIXES_BEFORE_MAIN.txt', CE + 'CODE_FIXES_BEFORE_MAIN.txt'),
                                ('output/summary.json', CE + 'summary.json')])
      + srv('generator_curve', [('PLAN.md', GC + 'PLAN.md')] + [(f, GC + 'server_outputs/' + f) for f in (
          'PLAN_REGISTERED_AT.txt', 'SHA256SUMS', 'SHA256SUMS_WRITTEN_AT.txt', 'OUTPUT_SHA256SUMS')]
            + [('output/summary.json', GC + 'summary.json')])
      + same('generator_fullpool', GF, ('PLAN.md', 'PLAN_REGISTERED_AT.txt', 'SHA256SUMS.v1', 'SHA256SUMS.v2', 'SHA256SUMS.v3',
                                        'SHA256SUMS_WRITTEN_AT.v1.txt', 'SHA256SUMS_WRITTEN_AT.v2.txt',
                                        'SHA256SUMS_WRITTEN_AT.v3.txt', 'OUTPUT_SHA256SUMS', 'OUTPUT_SHA256SUMS_WRITTEN_AT.txt'))
      + [('null_baseline/' + f, NB + f, ('local', 'read-only rescoring; no server run directory'))
         for f in ('PLAN.md', 'results_new_cell.txt', 'results_tissue.txt')]
      + srv('pushline_checks', [('PLAN.md', PC + 'PLAN.md'), ('SHA256SUMS', X + 'pushline-checks-20261009-0001/SHA256SUMS'),
                                ('OUTPUT_SHA256SUMS', X + 'pushline-checks-20261009-0001/OUTPUT_SHA256SUMS'),
                                ('output/summary.json', PC + 'summary.json')])
      + srv('tissue_kernel', [('PLAN.md', TK + 'PLAN.md'), ('SHA256SUMS', X + 'tissue-kernel-20261009-0001/SHA256SUMS'),
                              ('output/summary.json', X + 'tissue-kernel-20261009-0001/output/summary.json')])
      + srv('shared_crossdrug', [('PLAN.md', SC + 'PLAN.md'), ('SHA256SUMS', SC + 'server_outputs/SHA256SUMS'),
                                 ('output_new_cell/summary.json', SC + 'server_outputs/summary_new_cell.json'),
                                 ('output_tissue/summary.json', SC + 'server_outputs/summary_tissue.json'),
                                 ('repro.log', SC + 'server_outputs/repro.log')])
      + srv('rna_learning_curve', [('PLAN.md', RL + 'PLAN.md'),
                                   ('PLAN_REGISTERED_AT.txt', X + 'rna-learning-curve-20261009-0001/PLAN_REGISTERED_AT.txt'),
                                   ('SHA256SUMS', RL + 'server_outputs/SHA256SUMS'), ('run.log', RL + 'server_outputs/run.log'),
                                   ('output/new_cell/summary.json', RL + 'server_outputs/summary_new_cell.json'),
                                   ('output/tissue/summary.json', RL + 'server_outputs/summary_tissue.json')])
      # follow-up N6: PLAN.md as registered; server_record/ holds the server's top-level files (its PLAN.md after both addenda)
      + [('settle_stageA/PLAN.md', N6 + 'PLAN.md', ('local', 'plan as registered = first 7,812 bytes of server_record/PLAN.md'))]
      + [('settle_stageA/server_record/' + f, N6 + 'server_record/' + f, ('server', 'settle_stageA', f)) for f in (
          'PLAN.md', 'PLAN_REGISTERED_AT.txt', 'PLAN_SHA_AFTER_ADDENDUM2.txt', 'SHA256SUMS', 'SHA256SUMS_CRISPR', 'run.log',
          'run_crispr.log')]
      + [('settle_stageA/server_record/SHA256SUMS_WRITTEN_AT.txt', N6 + 'server_record/PLAN_REGISTERED_AT.txt',
          ('server', 'settle_stageA', 'SHA256SUMS_WRITTEN_AT.txt'))]  # same 25 bytes as PLAN_REGISTERED_AT.txt on the server
      + [('settle_stageA/server_record/OUTPUT_SHA256SUMS_server.txt', N6 + 'server_record/OUTPUT_SHA256SUMS_server.txt',
          ('local', 'SHA-256 listing of the outputs, computed on the server after the run'))]
      + [('settle_stageA/server_record/dryrun/SHA256_and_mtime_server.txt', N6 + 'server_record/dryrun/SHA256_and_mtime_server.txt',
          ('local', 'SHA-256 and modification times of the dry-run summaries, computed on the server'))]
      + [('settle_stageA/server_record/dryrun/%s_summary.json' % d, N6 + 'server_record/dryrun/%s_summary.json' % d,
          ('server', 'settle_stageA', 'dev/%s/summary.json' % d)) for d in ('dryrun_new_cell', 'dryrun_tissue', 'crispr_dryrun_new_cell')]
      + same('settle_stageA', N6, ['output/RESULT_AUTO.md', 'output/analysis.json', 'output_crispr/RESULT_AUTO_CRISPR.md',
                                   'output_crispr/analysis_crispr.json']
             + ['%s/%s/fold-%d/fold_result.json' % (o, s, f) for o in ('output', 'output_crispr') for s in SCEN for f in range(5)]
             + ['code/' + c for c in ('analyze.py', 'analyze_crispr.py', 'run.sh', 'run_crispr.sh', 'stageA.py', 'stageA_crispr.py')])
      + [('settle_stageA/' + f, N6 + f, ('local', 'written after the run to build Table S15'))
         for f in ('make_table_s15.py', 'table_s15.md', 'table_s15_values.json')]
      # protocols of the three 2026-10-08 development runs whose folds, banks, teacher and graph design later runs reuse
      + srv('development_protocols_20261008/recoverability', [('PROTOCOL.md', X + 'strong-teacher-recoverability-20261008-0001/PROTOCOL.md'),
                                                              ('handoff/PROTOCOL.md', EV + 'experiments/execution_20261008_recoverability/PROTOCOL.md'),
                                                              ('handoff/README.md', EV + 'experiments/execution_20261008_recoverability/README.md')])
      + srv('development_protocols_20261008/privileged_geometry', [('PROTOCOL.md', EV + 'experiments/execution_20261008_privileged_geometry/PROTOCOL.md'),
                                                                   ('EXTENSION.md', X + 'privileged-geometry-20261008-0001/EXTENSION.md'),
                                                                   ('handoff/README.md', EV + 'experiments/execution_20261008_privileged_geometry/README.md')])
      + srv('development_protocols_20261008/extra_supervision', [('PROTOCOL.md', EV + 'experiments/execution_20261008_extra_supervision/PROTOCOL.md'),
                                                                 ('README.md', EV + 'experiments/execution_20261008_extra_supervision/README.md')]))

S3 = (srv('internal_final', [(f, IF + f) for f in ('PLAN.md', 'PLAN_REGISTERED_AT.txt', 'SHA256SUMS.v1', 'SHA256SUMS.v2',
                                                  'FREEZE_REPORT.md', 'output/pre_sealed_checkpoint.json', 'output/progress.jsonl',
                                                  'output/summary.json', 'output/RESULT_DRAFT.md', 'frozen_models/FROZEN_MANIFEST.json')]
                         + [('RELEASE_RECEIPT.json', IF + 'output/RELEASE_RECEIPT.json'), ('logs/sealed_run.log', IF + 'output/sealed_run.log')])
      + [('review/' + f, IF + f, ('local', 'written by the reviewers outside the compute server')) for f in ('FREEZE_REVIEW.md', 'RECOMPUTE.md')])

S4 = (srv('crosslayer_matrix', [('output/' + f, CL + 'server_outputs/' + f) for f in (
    'comparisons.csv', 'matrix_new_cell.csv', 'matrix_tissue.csv', 'matrix_tables.md', 'analysis.json')]
    + [('output/%s/%s' % (s, f), CL + 'server_outputs/%s/%s' % (s, f)) for s in SCEN
       for f in ('fold_metrics.csv', 'per_drug_metrics.csv', 'bank_info.json', 'summary.json')])
    + [('bonferroni_primary24.csv', EXP + 'derived/bonferroni_primary24.csv', ('derived', 'from output/comparisons.csv'))])

S5 = srv('pushline_checks', [('output/' + f, PC + f) for f in ('comparisons.csv', 'fold_scores.csv', 'pass_fail.csv', 'summary.json')])

S6 = (srv('protein_panel', [('output/%s_main/%s' % (s, f), PP + '%s_%s.%s' % (f.split('.')[0], s, f.split('.')[1]))
                            for s in SCEN for f in ('fold_metrics.csv', 'summary.json', 'bank_info.json')]
          + [('output/analysis.json', PP + 'analysis_server.json')])
      + [('panel_membership.csv', EXP + 'derived/panel_membership.csv', ('derived', 'bank_info indices mapped to protein identifiers'))])

S7 = (srv('rna_panel', [('output/' + f, RP + 'server_outputs/output/' + f) for f in (
    'fold_metrics.csv', 'summary.json', 'analysis.json', 'result_tables.md')]
    + [('output/%s/fold-%d_bank_info.json' % (s, f), RP + 'server_outputs/output/%s/fold-%d_bank_info.json' % (s, f))
       for s in SCEN for f in range(5)])
      + [('gene_panel_membership.csv', EXP + 'derived/gene_panel_membership.csv', ('derived', 'bank_info indices mapped to gene and protein identifiers'))])

S8 = (srv('a_protein', [('output/%s_main/fold_metrics.csv' % s, AP + 'fold_metrics_%s.csv' % s) for s in SCEN]
          + [('output/%s_main/summary.json' % s, AP + 'summary_%s.json' % s) for s in SCEN]
          + [('output/analysis/summary.json', AP + 'summary.json'), ('output/analysis/tables.md', AP + 'tables.md')])
      + srv('b_gene', [('output/%s_main/fold_metrics.csv' % s, AR + 'server_outputs/%s_main/fold_metrics.csv' % s) for s in SCEN]
            + [('output/analysis/summary.json', AR + 'server_outputs/analysis/summary.json')]))

S9 = (srv('probe2_context_abstain', [('output/A/fold_metrics.csv', P2 + 'server_outputs/A/fold_metrics.csv')])
      + [('result_tables.md', P2 + 'result_tables.md', ('local', 'rendered from output/summary.json by code/make_result_tables.py')),
         ('audit_recomputation.txt', EXP + 'derived/audit_recomputation.txt', ('derived', "translation of the run's audit appendix"))])

S10 = srv('probe2_context_abstain', [('output/B/' + f, P2 + 'server_outputs/B/' + f) for f in (
    'aurc.csv', 'aurc_new_cell.csv', 'aurc_tissue.csv', 'riskcov.csv', 'riskcov_new_cell.csv', 'riskcov_tissue.csv',
    'percell_new_cell.csv', 'percell_tissue.csv', 'gen_check_new_cell.json', 'gen_check_tissue.json')])

PLAN = collections.OrderedDict([('S1', S1), ('S2', S2), ('S3', S3), ('S4', S4), ('S5', S5), ('S6', S6), ('S7', S7),
                                ('S8', S8), ('S9', S9), ('S10', S10)])

# plan versions named in checksum lists or STAR Methods: (run folder, file, prefix length or None, sha256 prefix, meaning)
PLAN_VERSIONS = [
    ('generator_curve', 'PLAN.md', 15135, '7b796fbe', 'registered plan (STAR Methods), before ADDENDUM-A'),
    ('generator_curve', 'PLAN.md', None, 'b64cebf8', 'after ADDENDUM-A; the hash in SHA256SUMS'),
    ('generator_fullpool', 'PLAN.md', 7485, 'c9234493', 'SHA256SUMS.v1 (registered)'),
    ('generator_fullpool', 'PLAN.md', 8265, '78b08ed7', 'SHA256SUMS.v2 (first pre-run correction)'),
    ('generator_fullpool', 'PLAN.md', None, '48b94bb1', 'SHA256SUMS.v3 (second pre-run correction)'),
    ('settle_stageA', 'PLAN.md', None, '1a498cac', 'server_record/SHA256SUMS (registered)'),
    ('settle_stageA', 'server_record/PLAN.md', 7812, '1a498cac', 'the same registered plan'),
    ('settle_stageA', 'server_record/PLAN.md', 8574, 'b2a06e7b', 'server_record/SHA256SUMS_CRISPR (after the CRISPR-step addendum)'),
    ('settle_stageA', 'server_record/PLAN.md', None, 'cffc1590', 'server_record/PLAN_SHA_AFTER_ADDENDUM2.txt (after both addenda)'),
]

GLOSSARY_S4 = [('有空间', 'headroom'), ('小幅空间', 'small headroom'), ('无空间', 'no headroom'), ('有恢复', 'recovered'),
               ('无恢复', 'not recovered'), ('部分收回', 'partly recovered'), ('基本未收回', 'essentially not recovered')]

# phrases the SI descriptions must contain for the READMEs to describe these archives correctly (warning only)
SI_REQUIRED = {'S2': ['where recorded']}


# ---------------------------------------------------------------------------------------------------- helpers
def sha(b):
    return hashlib.sha256(b).hexdigest()


def read(rel):
    with open(os.path.join(ROOT, rel), 'rb') as f:
        return f.read()


def wrap(text, indent=''):
    return textwrap.fill(text, width=118, initial_indent=indent, subsequent_indent=' ' * len(indent) if indent else '',
                         break_long_words=False, break_on_hyphens=False)


def bullets(items):
    return [wrap(t, '- ') for t in items]


def server_index():
    idx = {}
    with open(os.path.join(ROOT, EXP, 'server_run_index.tsv'), newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            idx[(r['server_run'], r['path'])] = r['sha256']
    return idx


def si_descriptions():
    md = open(SI_MD, encoding='utf-8').read()
    out = {}
    for m in re.finditer(r'(?m)^\*\*Data (S\d+)\. (.+?)\*\* (.+)$', md):
        out[m.group(1)] = (m.group(2), m.group(3).strip())
    return out


def main_titles():
    md = open(MAIN_MD, encoding='utf-8').read()
    return dict(re.findall(r'(?m)^Data (S\d+)\. (.*)$', md))


HEX = re.compile(r'^([0-9a-f]{64}) [ *](.+?)\s*$')


def parse_list(data):
    """(line number, sha256, path) of every checksum line; '#' comment lines and other text are skipped."""
    out = []
    for i, line in enumerate(data.decode('utf-8').splitlines(), 1):
        m = HEX.match(line)
        if m:
            path, note = m.group(2), ''
            c = re.match(r'(.+?)\s+#\s.*$', path)
            if c:
                path, note = c.group(1), 'the list line has a trailing comment'
            out.append((i, m.group(1), path, note))
    return out


def is_list(arc):
    b = arc.rsplit('/', 1)[-1]
    return b.startswith(('SHA256SUMS', 'OUTPUT_SHA256SUMS', 'PLAN_SHA_')) and '_WRITTEN_AT' not in b or b == 'SHA256_and_mtime_server.txt'


# lists whose paths are relative to a sub-folder of the run directory
LIST_BASE = {'acquisition_probe/OUTPUT_SHA256SUMS': 'acquisition_probe/output'}


def run_folder(arc):
    for k in sorted(RUNS, key=len, reverse=True):
        if arc.startswith(k + '/'):
            return k
    return None


# ---------------------------------------------------------------------------------------------------- checks
def check_entries(idx):
    seen = {}
    for tid, entries in PLAN.items():
        arcs = [a for a, _, _ in entries]
        dup = [a for a, c in collections.Counter(arcs).items() if c > 1]
        assert not dup, (tid, dup)
        for arc, src, origin in entries:
            data = read(src)
            if origin[0] == 'server':
                key = (RUNS[origin[1]], origin[2])
                assert key in idx, ('not in server listing', tid, arc, key)
                assert idx[key] == sha(data), ('differs from server', tid, arc, src)
            seen.setdefault(tid, {})[arc] = data
        keys = [o if o[0] == 'server' else ('src', s) for _, s, o in entries]
        assert len(set(keys)) == len(keys), ('same file twice', tid)
        by = collections.defaultdict(list)
        for (arc, src, origin) in entries:
            by[sha(seen[tid][arc])].append(origin)
        for h, os_ in by.items():
            assert len(os_) == 1 or all(o[0] == 'server' and o[2].endswith('.txt') and len(seen[tid][a]) < 100
                                        for o in os_ for a, _, o2 in entries if o2 == o), ('duplicate content', tid, os_)
    return seen


def check_facts(files):
    s2 = files['S2']
    for folder, fn, n, pre, _ in PLAN_VERSIONS:
        b = s2['%s/%s' % (folder, fn)]
        assert sha(b if n is None else b[:n]).startswith(pre), (folder, fn, n, pre)
    assert s2['settle_stageA/PLAN.md'] == s2['settle_stageA/server_record/PLAN.md'][:7812]
    assert sha(s2['probe2_context_abstain/output/summary.json']).startswith('4b09a68b')
    # N6 imported the frozen cross-layer and CRISPR-endpoint code unchanged
    lists = {a: {p: h for _, h, p, _ in parse_list(b)} for a, b in s2.items() if is_list(a)}
    assert lists['settle_stageA/server_record/SHA256SUMS']['code/crosslayer.py'] == lists['crosslayer_matrix/SHA256SUMS']['code/crosslayer.py']
    assert lists['settle_stageA/server_record/SHA256SUMS_CRISPR']['code/crispr_endpoint.py'] == lists['crispr_endpoint/SHA256SUMS']['code/crispr_endpoint.py']
    # Data S2 has every registration and checksum file of the server run directories it covers
    idx = server_index()
    pat = re.compile(r'(^|/)(PLAN_REGISTERED_AT|SHA256SUMS|OUTPUT_SHA256SUMS|PLAN_SHA_|CODE_FIXES)')
    have = {(RUNS[o[1]], o[2]) for _, _, o in PLAN['S2'] if o[0] == 'server'}
    for folder in {o[1] for _, _, o in PLAN['S2'] if o[0] == 'server' and not o[1].startswith('development')}:
        for (run, path) in idx:
            if run == RUNS[folder] and pat.search(path) and not path.startswith(('dev/', 'dryrun/', 'logs/')):
                assert (run, path) in have, ('server record missing from Data S2', run, path)
    # runs without registration-time file / without output list, as stated in the README
    noreg = sorted(f for f in {a.split('/')[0] for a in s2} if RUNS.get(f) and f + '/PLAN_REGISTERED_AT.txt' not in s2
                   and f + '/server_record/PLAN_REGISTERED_AT.txt' not in s2 and not f.startswith('development'))
    assert noreg == ['acquisition_probe', 'protein_panel', 'pushline_checks', 'shared_crossdrug', 'tissue_kernel'], noreg
    noout = sorted(f for f in {a.split('/')[0] for a in s2} if RUNS.get(f) and not f.startswith('development')
                   and not any(a.startswith(f + '/') and 'OUTPUT_SHA256SUMS' in a for a in s2))
    assert noout == ['acquisition_probe_rna', 'crosslayer_matrix', 'rna_learning_curve', 'shared_crossdrug', 'tissue_kernel'], noout
    rl = s2['rna_learning_curve/run.log'].decode().splitlines()[:7]
    assert all(l.endswith(': OK') for l in rl) and [l.split(':')[0] for l in rl] == [p for _, _, p, _ in parse_list(s2['rna_learning_curve/SHA256SUMS'])][:7]
    cl = s2['crosslayer_matrix/run.log'].decode()
    assert '2026-10-09T13:25:31+0800' in cl and '2026-10-09T13:32:47+0800' in cl
    assert s2['probe2_context_abstain/OUTPUT_SHA256SUMS'].decode().count('\n') == 62
    assert read(P2 + 'server_outputs/OUTPUT_SHA256SUMS').decode().count('\n') == 59
    # identity lists: counts, overlaps, blank tissues
    ids = {}
    for f in ('extra19_M_AUC_ids.tsv', 'extra66_M_RNA_ids.tsv', 'extra354_RNA_AUC_ids.tsv'):
        rows = list(csv.DictReader(io.StringIO(files['S1']['whitelist_packages/' + f].decode()), delimiter='\t'))
        ids[f] = rows
    n = {k: len(v) for k, v in ids.items()}
    assert list(n.values()) == [19, 66, 354], n
    sets = [set(r['sample_id'] for r in v) for v in ids.values()]
    facts = dict(all3=len(sets[0] & sets[1] & sets[2]), union=len(sets[0] | sets[1] | sets[2]),
                 blank={k: sum(1 for r in v if not r['tissue']) for k, v in ids.items()})
    assert all('SIDM00271' in s for s in sets)
    assert all(r['tissue'] == '' for v in ids.values() for r in v if r['sample_id'] == 'SIDM00271')
    m66 = sets[1]
    for f in range(5):
        for s in SCEN:
            t = files['S1']['generator_curve/output/extra_ids_%s_fold-%d.txt' % (s, f)].decode()
            assert set(re.findall(r'SIDM\d{5}', t)) <= m66
    d = json.loads(files['S1']['identity_split/original_identity_split.json'])
    for part in d['partitions'].values():
        assert sha(('\n'.join(part['sample_ids']) + '\n').encode()) == part['sample_ids_sha256']
    e = {k: v for k, v in d.items() if k != 'integrity'}
    assert sha(json.dumps(e, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()) == d['integrity']['canonical_payload_sha256']
    assert d['frozen_on'] == '2026-08-30' and d['internal_final_release_gate']['released'] is False
    assert [d['partitions'][p]['count'] for p in ('train', 'validation', 'internal_final')] == [421, 64, 123]
    assert len(d['split']['strata']) == 35 and d['split']['seed'] == 20260830
    # Bonferroni file = the 24 primary rows of comparisons.csv, values unchanged
    comp = list(csv.DictReader(io.StringIO(files['S4']['crosslayer_matrix/output/comparisons.csv'].decode())))
    prim = [r for r in comp if r['kind'] in ('headroom', 'recovered')]
    bon = list(csv.DictReader(io.StringIO(files['S4']['bonferroni_primary24.csv'].decode())))
    assert len(bon) == 24 and all(all(b[k] == p[k] for k in p) for b, p in zip(bon, prim))
    assert sum(b['significant'] == 'TRUE' for b in bon) == 4
    assert all((b['significant'] == 'TRUE') == (float(b['p']) < 0.05 / 24) for b in bon)
    # S3 CRISPR count note
    sj = files['S3']['internal_final/output/summary.json'].decode()
    assert sj.count('"n_eval_cells": 123') == 2 and '"n_eval_cells": 95' not in sj
    assert '95' in files['S3']['review/RECOMPUTE.md'].decode()
    # result_tables.md of Data S9 is the rendering of the hash-locked summary.json (checked by rerunning the renderer)
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as td:
        sj = os.path.join(td, 'summary.json')
        open(sj, 'wb').write(s2['probe2_context_abstain/output/summary.json'])
        out = subprocess.run([sys.executable, '-I', '-B', os.path.join(ROOT, P2 + 'make_result_tables.py'), sj],
                             capture_output=True, check=True).stdout
    assert out == files['S9']['result_tables.md']
    assert sha(read(P2 + 'make_result_tables.py')).startswith('bcbc584a')
    return facts


# ---------------------------------------------------------------------------------------------------- checksum index
def checksum_index(tid, files, all_by_sha):
    rows, counts = [], collections.Counter()
    arcs = files[tid]
    for arc in sorted(a for a in arcs if is_list(a)):
        rf = LIST_BASE.get(arc, run_folder(arc))
        for line, h, p, note in parse_list(arcs[arc]):
            target = '%s/%s' % (rf, p)
            here = arcs.get(target)
            elsewhere = [x for x in all_by_sha.get(h, []) if x != 'Data_%s/%s' % (tid, target)]
            if here is not None and sha(here) == h:
                st, det = 'OK', ('also at ' + ', '.join(elsewhere)) if elsewhere else ''
            elif here is not None:
                st, det = 'FAILED', 'different version at this path'
                for cand in sorted(a for a in arcs if a.startswith(run_folder(arc) + '/') and a.rsplit('/', 1)[-1] == p.rsplit('/', 1)[-1]):
                    b = arcs[cand]
                    if sha(b) == h:
                        det = 'other plan version: identical to Data_%s/%s' % (tid, cand)
                        break
                    cut = [i + 1 for i, c in enumerate(b) if c == 10]
                    hit = [n for n in cut if sha(b[:n]) == h]
                    if hit:
                        det = 'earlier plan version: first %d bytes of Data_%s/%s' % (hit[0], tid, cand)
                        break
                assert det != 'different version at this path', (arc, p)
            elif elsewhere:
                st, det = 'ELSEWHERE', 'same bytes at ' + ', '.join(elsewhere)
            else:
                st, det = 'NOT_INCLUDED', ''
            counts[st] += 1
            rows.append(['Data_%s/%s' % (tid, arc), line, h, p, st, '; '.join(x for x in (det, note) if x)])
    buf = io.StringIO()
    w = csv.writer(buf, delimiter='\t', lineterminator='\n')
    w.writerow(['list', 'line', 'sha256', 'listed_path', 'status', 'detail'])
    w.writerows(rows)
    return buf.getvalue().encode(), counts, rows


# ---------------------------------------------------------------------------------------------------- README text
def scan(files):
    """Extensions, NaN in JSON, absolute paths and Chinese text in an archive."""
    ext = sorted({('.' + a.rsplit('.', 1)[1]) if '.' in a.rsplit('/', 1)[-1] else 'no extension' for a in files})
    nan = sorted(a for a, b in files.items() if a.endswith('.json') and re.search(rb'\bNaN\b', b))
    paths = {k: sorted(a for a, b in files.items() if k.encode() in b) for k in ('/public/home/', '/Users/', '/home/node')}
    cjk = sorted(a for a, b in files.items() if re.search('[一-鿿]', b.decode('utf-8', 'replace')))
    return ext, nan, paths, cjk


def common_notes(tid, files, derived, own):
    ext, nan, paths, cjk = scan(files)
    cjk = [a for a in cjk if own.get(a, ('server',))[0] != 'derived']
    lists = [a for a in files if is_list(a)]
    n = []
    if derived:
        n.append('Run records are byte-for-byte copies (not edited). Files that are not copies of files of the server run '
                 'directories (derived tables, renderings, and records kept outside the compute server) are described '
                 'under "Files" below.')
    else:
        n.append('Files are byte-for-byte copies of the run records (not edited).')
    noext = [a for a in files if '.' not in a.rsplit('/', 1)[-1]]
    assert all(is_list(a) for a in noext), noext
    vers = [e for e in ext if re.match(r'^\.v\d$', e)]
    exts = [e for e in ext if e != 'no extension' and e not in vers]
    if noext or vers:
        exts.append('checksum lists without extension' + (' or with .v1-.v3' if vers else ''))
    n.append('Plain-text formats only (%s).' % ', '.join(exts) + (
        ' JSON files use NaN for missing values (Python json syntax; strict parsers need NaN replaced by null).' if nan else ''))
    if cjk:
        if tid in ('S2', 'S3'):
            n.append('Most registered plans and audit records are in Chinese, as registered; they are not translated.')
        else:
            n.append('Chinese text, as written in the original records and not translated, occurs in: %s.' % ', '.join(cjk))
    kinds = []
    if paths['/public/home/']:
        kinds.append('of the compute server (/public/home/<user>/...)')
    if paths['/Users/']:
        kinds.append("of the authors' local workstation (/Users/<user>/...)")
    if paths['/home/node']:
        kinds.append('of an automated-agent workspace (/home/node/...)')
    if kinds:
        n.append('Some files contain absolute file paths %s; they are left unedited so that the files stay byte-identical '
                 'to the original records.' % (', '.join(kinds[:-1]) + ' and ' + kinds[-1] if len(kinds) > 1 else kinds[0]))
    if lists:
        n.append('Run folders have the layout of the run directory on the compute server, so each checksum list can be '
                 'checked from its run folder (e.g. cd Data_%s/<run folder>; shasum -a 256 -c <list>). Entries for files '
                 'not included here (%s) are reported as missing. Lines beginning with # are comments written by the runs; '
                 'checkers report them as improperly formatted. CHECKSUM_INDEX.tsv (derived by the archive builder) gives '
                 'the status of every entry: OK (verifies here), FAILED (a different version is at the listed path%s), '
                 'ELSEWHERE (the same bytes are at the archive path given), NOT_INCLUDED.' % (
                     tid, 'code files, .npz arrays, RESULT.md files and other run files' if tid == 'S2' else
                     'the code files and the frozen model arrays', '; see Plan versions' if tid == 'S2' else ''))
    else:
        rfs = sorted({run_folder(a) for a in files if run_folder(a)})
        n.append('The run folder%s %s %s the layout of the run directory on the compute server; the runs\' checksum lists '
                 'are in Data S2 (Data_S2/CHECKSUM_INDEX.tsv shows which list covers each file of this archive).'
                 % ('s' if len(rfs) > 1 else '', ', '.join(f + '/' for f in rfs), 'have' if len(rfs) > 1 else 'has'))
    return n


def runs_table(files):
    rows = []
    for folder in sorted({a.split('/')[0] for a in files if not a.startswith('development_protocols')}):
        if folder in ('CHECKSUM_INDEX.tsv',):
            continue
        reg = files.get(folder + '/PLAN_REGISTERED_AT.txt') or files.get(folder + '/server_record/PLAN_REGISTERED_AT.txt')
        regs = reg.decode().strip().splitlines()[-1].split()[0] if reg else 'none (registration date stated in PLAN.md)'
        if folder == 'crispr_endpoint':
            regs = reg.decode().strip().splitlines()[1].split()[0]
        lists = sorted(a[len(folder) + 1:] for a in files if a.startswith(folder + '/') and is_list(a))
        rows.append('%s = %s; registration time: %s; checksum lists: %s' % (
            folder, RUNS[folder] or 'no server run directory', regs, ', '.join(lists) or 'none'))
    return rows


def s1_files(facts):
    b = facts['blank']
    return [
        'identity_split/original_identity_split.json: the identity protocol record CREED_CCLE_L0_L1_L2_SPLIT_V1, frozen on '
        '2026-08-30, before the sealed-set release (released = false is expected; the release receipt is in Data S3). '
        'sample_ids_sha256 = SHA-256 of the IDs, in the stored order, joined by \\n with a trailing \\n; '
        'canonical_payload_sha256 = SHA-256 of the JSON without the integrity block, with sorted keys and compact '
        'separators (UTF-8). TEMPLAR and OX are internal project code names.',
        'whitelist_packages/extra19_M_AUC_ids.tsv, extra66_M_RNA_ids.tsv, extra354_RNA_AUC_ids.tsv (derived): the '
        'identities of the three whitelisted extra-sample packages (columns sample_id, model_id, tissue, package), '
        'exported on 2026-10-11 from the package QA files of the compute server (m_gdsc_extra19_QA.json sample_qa; '
        'rna_candidate_sample_QA.tsv rows with has_m or has_gdsc, in the order of the package configuration), with '
        'patient identifiers and every count or value field dropped. Tissue is blank where the package records give '
        'none: SIDM00271 (tissue unresolved; STAR Methods) in all three packages, and %d further M–RNA and %d further '
        'RNA–AUC lines. The packages overlap (%d lines are in all three; %d distinct '
        'lines in total) and were never summed.' % (b['extra66_M_RNA_ids.tsv'] - 1, b['extra354_RNA_AUC_ids.tsv'] - 1,
                                                     facts['all3'], facts['union']),
        'Legacy file name: identity_split/original_identity_split.json and the code of Methods S1 name the RNA source '
        'CCLE_RNAseq_RSEM_Expected_Counts.csv.gz (role L1_rna_expected_counts). That file is a locally gzip-compressed '
        'copy of the Cell Model Passports release 20191101 read-count table rnaseq_read_count_20191101.csv (key '
        'resources table); the RSEM name is historical.',
        'raw_source_checksums.tsv (derived): for every public source file that the data-preparation code read, the '
        'public location, size, SHA-256 (and the MD5 that DepMap publishes), what it feeds and notes, computed on the '
        'compute server on 2026-10-11. The files themselves are public and are not redistributed here.',
        'generator_curve/output/extra_ids_{new_cell,tissue}_fold-{0..4}.txt: the per-bank subsets of the 66-line M–RNA '
        'package used by the generator-size curve (run generator-curve-20261009-0001; the first line of each file gives '
        'the count per bank).',
    ]


def s2_files(files, counts, rows):
    fails = [r for r in rows if r[4] == 'FAILED']
    assert fails and all('plan version' in r[5] for r in fails), fails
    t = [wrap('Run folders (folder = server run directory under /public/home/<user>/creed-research-runs/; registration '
              'time from PLAN_REGISTERED_AT.txt; checksum lists present):')] + [wrap(r, '  ') for r in runs_table(files)]
    t += [''] + bullets([
        'Data S2 holds every plan-registration and checksum file that exists in these server run directories '
        '(PLAN_REGISTERED_AT, SHA256SUMS*, OUTPUT_SHA256SUMS*, *_WRITTEN_AT), plus the reproduction-gate summaries and '
        'logs. protein_panel, acquisition_probe, pushline_checks, tissue_kernel and shared_crossdrug have no '
        'PLAN_REGISTERED_AT.txt: their plans state the registration date (2026-10-09) and their SHA256SUMS hold the plan '
        'hash (shared_crossdrug/SHA256SUMS also notes the time it was written, 2026-10-09T11:35:05+0800). '
        'crosslayer_matrix, acquisition_probe_rna, tissue_kernel and shared_crossdrug wrote no output checksum list; '
        'rna_learning_curve appended its output hashes to SHA256SUMS (see below).',
        'null_baseline: read-only rescoring of the saved predictions of extra-supervision-20261008-0001, with no server '
        'run directory. PLAN.md was registered in its text (2026-10-09, before any prediction value was read) and not '
        'hashed; results_new_cell.txt and results_tissue.txt are the printed results.',
        'rna_learning_curve/SHA256SUMS: the first 7 entries (PLAN.md and code) were verified at the start of the main run '
        '(run.log lines 1-7, OK); RESULT.md and the 4 output entries were appended after the run. code/fake_outputs.py '
        'writes random synthetic outputs, used only to test analyze.py before any outer score.',
        'crosslayer_matrix/run.log gives the main-run window (13:25:31-13:32:47); crispr_endpoint/CODE_FIXES_BEFORE_MAIN.txt '
        'records the code fixes made before the main run.',
        'probe2_context_abstain/OUTPUT_SHA256SUMS is the server file (62 lines); an earlier 59-line local copy is not '
        'included. No file appears twice in this archive; reproduction-gate summaries that are also result tables appear '
        'in Data S4-S10 as well (same bytes, same path in the run folder).',
        'Paths in the lists are relative to the server run directory, i.e. to the run folder here. Exceptions: '
        'acquisition_probe/OUTPUT_SHA256SUMS is relative to the run\'s output/ folder (check it from '
        'acquisition_probe/output/), and settle_stageA/server_record/dryrun/SHA256_and_mtime_server.txt to the server '
        'folder dev/ (the files are saved here under other names, see below).',
        'settle_stageA = follow-up N6 (run settle-stageA-20261011-0001). PLAN.md is the plan as registered; server_record/ '
        'holds copies of the files at the top of the server run directory, among them PLAN.md after both addenda. '
        'server_record/OUTPUT_SHA256SUMS_server.txt and server_record/dryrun/SHA256_and_mtime_server.txt are SHA-256 '
        'listings computed on the server after the run and saved by the authors (no file of that name exists on the '
        'server); the dry-run summaries are the server files dev/<name>/summary.json saved as <name>_summary.json. '
        'make_table_s15.py, table_s15.md and table_s15_values.json were written after the run to build Table S15 from '
        'the 20 fold_result.json files. The frozen cross-layer and CRISPR-endpoint code that N6 imported '
        '(code/crosslayer.py, code/crispr_endpoint.py) has the same SHA-256 as in those runs\' SHA256SUMS.',
        'development_protocols_20261008/: protocols of the three 2026-10-08 development runs whose folds, inner banks, '
        'measured teacher and protein sample-graph design the later runs reuse (recoverability = '
        'strong-teacher-recoverability-20261008-0001, privileged_geometry = privileged-geometry-20261008-0001, '
        'extra_supervision = extra-supervision-20261008-0001). These runs wrote no checksum list or registration-time '
        'file. handoff/ is the run\'s link to its planning folder. recoverability/PROTOCOL.md (run directory, written '
        'before the run) and recoverability/handoff/PROTOCOL.md (the same protocol plus the mid-run technical revision '
        'v2 for held-out-tissue teacher folds with fewer than three training tissues) differ only by that section; '
        'privileged_geometry/EXTENSION.md records the later extension of the mu grid to 100 and 1000.',
    ])
    t += ['', wrap('Checksum verification (each list checked from its run folder; counts of CHECKSUM_INDEX.tsv): %s. The %d '
                   'FAILED entries are plan versions other than the PLAN.md at the listed path (%s); each is reproduced as '
                   'shown under Plan versions. NOT_INCLUDED entries are code files, .npz arrays, RESULT.md files, logs and '
                   'other run files that are not part of the Data S files.' % (
                       ', '.join('%s %d' % (k, counts[k]) for k in ('OK', 'FAILED', 'ELSEWHERE', 'NOT_INCLUDED')), len(fails),
                       '; '.join('%s line %s' % (r[0][len('Data_S2/'):], r[1]) for r in fails))),
          '', wrap('Plan versions (SHA-256 prefix = the file or byte prefix that reproduces it, e.g. head -c 15135 '
                   'generator_curve/PLAN.md | shasum -a 256):')]
    for folder, fn, nbytes, pre, what in PLAN_VERSIONS:
        t.append(wrap('%s... = %s %s/%s: %s' % (pre, 'full file' if nbytes is None else 'first %d bytes of' % nbytes, folder, fn, what), '  '))
    return t


def s3_files(counts, rows):
    ni = sorted(r[3] for r in rows if r[4] == 'NOT_INCLUDED')
    assert set(ni) == {'code/creed_core.py', 'code/oneshot.py', 'code/sealed_io.py', 'frozen_models/inner_losses.npz',
                       'frozen_models/model.npz'} and counts['FAILED'] == 0, ni
    return bullets([
        'internal_final/ has the layout of the server run directory internal-final-20261009-0001 for the files included '
        '(plan, registration time, SHA256SUMS.v1 and .v2, FREEZE_REPORT.md, RELEASE_RECEIPT.json, '
        'output/pre_sealed_checkpoint.json, logs/sealed_run.log, output/progress.jsonl, output/summary.json, '
        'output/RESULT_DRAFT.md, frozen_models/FROZEN_MANIFEST.json).',
        'review/FREEZE_REVIEW.md (read-only freeze review) and review/RECOMPUTE.md (independent recomputation report) were '
        'written by the reviewers outside the compute server; they are not listed in any checksum file.',
        'The frozen model arrays (model.npz, inner_losses.npz; SHA-256 in SHA256SUMS.v2) are not included and will be '
        'deposited at Zenodo before publication (reserved DOI: https://doi.org/10.5281/zenodo.23299706).',
        'CRISPR evaluated lines = 95: output/summary.json (n_eval_cells = 123) and output/RESULT_DRAFT.md count all 123 '
        'sealed lines, of which 95 have DepMap 24Q2 CRISPR gene effects (corrected in review/RECOMPUTE.md). Items F1-F8 '
        'in these files are P1-P8 in the paper.',
        'Checksum verification from internal_final/ (counts of CHECKSUM_INDEX.tsv): %s; NOT_INCLUDED are the three code '
        'files and the two frozen arrays.' % ', '.join('%s %d' % (k, counts[k]) for k in ('OK', 'FAILED', 'ELSEWHERE', 'NOT_INCLUDED')),
    ])


def s4_files():
    return bullets([
        'crosslayer_matrix/ = run crosslayer-matrix-20261009-0001; output/comparisons.csv holds all comparisons with '
        'per-fold values, wins, Nadeau-Bengio p and CI.',
        'bonferroni_primary24.csv (derived, not a run record): the 24 primary comparisons (kind headroom or recovered) '
        'of output/comparisons.csv, every value copied as is, plus threshold = 0.05/24 and significant = TRUE when '
        'p < threshold (4 rows; two-sided test, the sign is given by gain).',
        'Labels in output/matrix_new_cell.csv and output/matrix_tissue.csv (output/matrix_tables.md is in Chinese): ' +
        '; '.join('%s = %s' % g for g in GLOSSARY_S4) + ' (room_label, recovery_label and share_label; STAR Methods).',
    ])


def s6_files():
    return bullets([
        'protein_panel/ = run protein-panel-20261009-0001. output/*_main/bank_info.json give each bank\'s panels as column '
        'indices into the 6,692-column proteome; they are not listed in the run\'s OUTPUT_SHA256SUMS.',
        'panel_membership.csv (derived, not a run record): every index list of output/*_main/bank_info.json mapped to '
        'protein identifiers (protein_ids of the frozen 421-line source bundle, exported on 2026-10-11). Columns: '
        'scenario, fold, bank (inner-0..2, outer), panel (top64 = candidates ranked by relevance; R16s0..R32s4 = random '
        'panels, k = 16 or 32, seeds 0-4), position (rank 1-64 for top64; position in the stored, column-sorted list for '
        'random panels), column_index, protein_id, top64_score. The k-protein panel of a bank is the first k rows of its '
        'top64. Indices are 0-based: mapping the outer-bank top-k panels this way reproduces every identifier list in '
        'output/analysis.json (panels.k*_in_all_5_outer), and 1-based mapping does not.',
    ])


def s7_files():
    return bullets([
        'rna_panel/ = run rna-panel-20261009-0001. output/*/fold-*_bank_info.json give each bank\'s panels as column '
        'indices (genes: into the 34,985-gene aligned RNA bundle; P32: into the 6,692-column proteome).',
        'gene_panel_membership.csv (derived, not a run record): every index list of the bank_info files mapped to '
        'identifiers (gene_id of the aligned RNA bundle, with the symbol as annotation; protein_id of the source bundle; '
        'exported on 2026-10-11). Columns: scenario, fold, bank, panel (top128 = genes ranked by relevance; G8-G128 = '
        'gene panels, the first k of top128; R16s0..R32s4 = random gene panels; P32 = the 32-protein part of the '
        'two-layer arm), position (rank for top128; position in the stored, column-sorted list otherwise), column_index, '
        'layer (RNA or protein), feature_id, symbol, score (top128_score). Indices are 0-based: mapping the outer-bank '
        'panels this way reproduces the gene symbols in output/analysis.json (panels.k*_in_all_5_outer and '
        'outer_top16_fold0), and 1-based mapping does not; every P32 equals the first 32 proteins of the protein-panel '
        'run\'s top64 for the same bank (Data S6).',
    ])


def s8_files():
    return bullets(['a_protein/ = run acquisition-probe-20261009-0001 (protein version); b_gene/ = run '
                    'acquisition-probe-rna-20261009-0001 (gene version).'])


def s9_files():
    return bullets([
        'probe2_context_abstain/ = run probe2-context-abstain-20261009-0001 (part A: context-specific panels).',
        'result_tables.md (rendered, not a server file): the tables of the run\'s output/summary.json (SHA-256 4b09a68b..., '
        'in Data S2) rendered by the run\'s code/make_result_tables.py (bcbc584a..., listed in its OUTPUT_SHA256SUMS); '
        'rerunning python3 make_result_tables.py summary.json reproduces it byte for byte.',
        'audit_recomputation.txt (derived): English translation of the run\'s audit appendix (RESULT.md section "附：独立审计结论"), '
        'every value copied as is, followed by the Chinese original.',
    ])


def s10_files():
    return bullets(['probe2_context_abstain/ = run probe2-context-abstain-20261009-0001 (part B: abstention).'])


# ---------------------------------------------------------------------------------------------------- build
def zinfo(name):
    zi = zipfile.ZipInfo(name, date_time=ZIP_TIME)
    zi.compress_type = zipfile.ZIP_DEFLATED
    zi.create_system = 3
    zi.external_attr = 0o100644 << 16
    return zi


def coverage(tid, files, list_sha):
    miss = sorted(a for a, b in files.items() if sha(b) not in list_sha)
    if not miss:
        return ['Every file of this archive is listed in a run checksum list in Data S2 (see Data_S2/CHECKSUM_INDEX.tsv).']
    s2 = PLAN['S2']
    s2folder = {'a_protein': 'acquisition_probe', 'b_gene': 'acquisition_probe_rna'}
    noout = sorted({run_folder(a) for a in miss if run_folder(a) and not any(
        x.startswith(s2folder.get(run_folder(a), run_folder(a)) + '/') and 'OUTPUT_SHA256SUMS' in x for x, _, _ in s2)})
    why = (' (%s wrote no output checksum list)' % ', '.join(RUNS[f] for f in noout)) if noout else ''
    if len(miss) == len(files):
        return ['No file of this archive is listed in a run checksum list%s.' % why]
    return ['Not listed in any run checksum list%s: %s.' % (why, ', '.join(miss))]


def build():
    os.makedirs(OUT, exist_ok=True)
    idx = server_index()
    files = check_entries(idx)
    facts = check_facts(files)
    all_by_sha = collections.defaultdict(list)
    for tid, fs in files.items():
        for a, b in fs.items():
            all_by_sha[sha(b)].append('Data_%s/%s' % (tid, a))
    extra, stats = {}, {}
    for tid in ('S2', 'S3'):
        data, counts, rows = checksum_index(tid, files, all_by_sha)
        extra[tid] = data
        stats[tid] = (counts, rows)
    list_sha = {h for tid in ('S2', 'S3') for r in stats[tid][1] for h in [r[2]]}
    desc, mt = si_descriptions(), main_titles()
    warnings = []
    report = []
    for tid, entries in PLAN.items():
        title, text = desc[tid]
        if mt.get(tid, '') + '.' != title:
            warnings.append('title of Data %s differs between supplemental_information.md and main_text.md' % tid)
        for phrase in SI_REQUIRED.get(tid, []):
            if phrase not in text:
                warnings.append('Data %s description in supplemental_information.md lacks "%s"; apply the returned change '
                                'and rebuild' % (tid, phrase))
        fs = dict(files[tid])
        if tid in extra:
            fs['CHECKSUM_INDEX.tsv'] = extra[tid]
        derived = any(o[0] != 'server' for _, _, o in entries) or tid in extra
        own = {a: o for a, _, o in entries}
        readme = ['Data %s. %s' % (tid, title), '', wrap(text), '', 'About this archive'] + bullets(common_notes(tid, fs, derived, own))
        body = {'S1': lambda: bullets(s1_files(facts)), 'S2': lambda: s2_files(files['S2'], *stats['S2']),
                'S3': lambda: s3_files(*stats['S3']), 'S4': s4_files, 'S5': lambda: bullets(['pushline_checks/ = run pushline-checks-20261009-0001.']),
                'S6': s6_files, 'S7': s7_files, 'S8': s8_files, 'S9': s9_files, 'S10': s10_files}[tid]()
        readme += ['', 'Files'] + body
        if tid not in ('S2', 'S3'):
            readme += bullets(coverage(tid, files[tid], list_sha))
        readme += ['', 'SHA-256 of every file in this archive:'] + ['%s  Data_%s/%s' % (sha(fs[a]), tid, a) for a in sorted(fs)]
        text_out = '\n'.join(readme) + '\n'
        assert not re.search(r'TODO|\[\[|Still to be added|data_files_plan|PENDING|placeholder', text_out), tid
        path = os.path.join(OUT, 'Data_%s.zip' % tid)
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr(zinfo('Data_%s/README.txt' % tid), text_out.encode('utf-8'))
            for a in sorted(fs):
                z.writestr(zinfo('Data_%s/%s' % (tid, a)), fs[a])
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
        report.append((tid, os.path.getsize(path), len(fs) + 1))
    return report, warnings, stats


if __name__ == '__main__':
    report, warnings, stats = build()
    for tid, size, n in report:
        print('Data_%-4s %8d bytes  %3d files (incl. README.txt)' % (tid, size, n))
    for tid, (counts, rows) in stats.items():
        print('Data_%s checksum entries: %s' % (tid, dict(sorted(counts.items()))))
        for r in rows:
            if r[4] == 'FAILED':
                print('   FAILED  %s line %s  %s  (%s)' % (r[0], r[1], r[3], r[5]))
    for w in warnings:
        print('WARNING: ' + w)

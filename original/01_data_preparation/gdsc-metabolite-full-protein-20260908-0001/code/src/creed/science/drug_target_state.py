"""Conservative source-annotation identity links, not proof of drug mechanism."""
import csv
import re
from collections import defaultdict


def gene_feature_map(hgnc_path, rna_symbols, protein_ids):
    approved = {}; synonyms = defaultdict(set); accessions = defaultdict(set)
    with open(hgnc_path, encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            if row['status'] != 'Approved':
                continue
            symbol = row['symbol']; approved[symbol] = symbol
            for field in ('alias_symbol', 'prev_symbol'):
                for token in row[field].split('|'):
                    if token: synonyms[token].add(symbol)
            for token in row['uniprot_ids'].split('|'):
                if token: accessions[token].add(symbol)
    def resolve(token):
        if token in approved: return token
        candidates = synonyms.get(token, set())
        return next(iter(candidates)) if len(candidates) == 1 else None
    rna = defaultdict(list); protein = defaultdict(list)
    for i, name in enumerate(rna_symbols):
        symbol = resolve(str(name))
        if symbol: rna[symbol].append(i)
    for i, name in enumerate(protein_ids):
        symbols = accessions.get(str(name).split(';')[0], set())
        if len(symbols) == 1: protein[next(iter(symbols))].append(i)
    return resolve, rna, protein


def identity_pairs(hgnc_path, rna_symbols, protein_ids):
    """All unique same-gene links, independent of drug labels or expression."""
    _, rna, protein = gene_feature_map(hgnc_path, rna_symbols, protein_ids)
    return sorted([dict(symbol=s, rna_index=v[0], protein_index=protein[s][0])
                   for s,v in rna.items() if len(v)==1 and len(protein[s])==1],
                  key=lambda r:r['protein_index'])


def target_map(hgnc_path, annotations, rna_symbols, protein_ids):
    resolve, rna, protein = gene_feature_map(hgnc_path, rna_symbols, protein_ids)
    records = {}
    for drug, item in annotations.items():
        tokens = sorted({s.strip() for x in item['putative_targets'] for s in re.split('[,;]', x) if s.strip()})
        resolved = {token: resolve(token) for token in tokens}
        symbols = sorted({x for x in resolved.values() if x})
        both = [s for s in symbols if len(rna[s]) == 1 and len(protein[s]) == 1]
        records[drug] = dict(raw_targets=item['putative_targets'], names=item['names'], tokens=resolved,
            paired_symbols=both, rna_indices=[rna[s][0] for s in both], protein_indices=[protein[s][0] for s in both],
            unmatched_tokens=[t for t, s in resolved.items() if not s],
            canonical_symbols=symbols, supported=bool(both))
    return dict(records=records, drug_ids=len(records),
        supported_drug_ids=sum(x['supported'] for x in records.values()),
        mapped_protein_features=sum(len(v) for v in protein.values()),
        unique_paired_genes=sum(len(v)==1 and len(protein[k])==1 for k,v in rna.items()),
        semantics='HGNC identity mapping of GDSC2 putative annotations; not independent pharmacological validation',
        rules='exact approved first; globally unique alias/previous otherwise; unique RNA and protein feature per paired gene; no family expansion')

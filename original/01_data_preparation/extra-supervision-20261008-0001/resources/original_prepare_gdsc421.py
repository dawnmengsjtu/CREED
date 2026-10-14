"""Extract GDSC2 outcomes for exact 421 development IDs; no name/fuzzy matching.

Numeric outcomes outside the supplied whitelist are never interpreted. Missing
measurements stay NaN; conflicting replicate keys are excluded, not averaged.
"""
import argparse
import csv
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from run_research_diagnostic import sha

SOURCE_SHA='9d6dc82b9a072bb00b6139dcd6c4c11de1e047f210382fb726e92ab8dd0eae2a'


def extract(source,ids):
    whitelist=set(ids);values={};conflicts=set();names=defaultdict(set);targets=defaultdict(set)
    total=0;matched=0;invalid=0;datasets=Counter();releases=Counter();duplicate_same=0;outside_unit_auc=0
    with Path(source).open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle)
        required={'SANGER_MODEL_ID','DATASET','DRUG_ID','DRUG_NAME','WEBRELEASE','AUC','LN_IC50','MIN_CONC','MAX_CONC','PUTATIVE_TARGET'}
        if not required.issubset(reader.fieldnames):raise ValueError('unexpected source schema')
        for row in reader:
            total+=1;datasets[row['DATASET']]+=1;releases[row['WEBRELEASE']]+=1
            sid=row['SANGER_MODEL_ID']
            if sid not in whitelist:continue
            if row['DATASET']!='GDSC2':raise ValueError('mixed dataset')
            matched+=1;drug=row['DRUG_ID'];key=(sid,drug)
            names[drug].add(row['DRUG_NAME']);targets[drug].add(row['PUTATIVE_TARGET'])
            try:record=tuple(float(row[k]) for k in ('AUC','LN_IC50','MIN_CONC','MAX_CONC'))
            except ValueError:invalid+=1;continue
            if not np.isfinite(record).all() or not (0<record[2]<=record[3]):
                invalid+=1;continue
            if not 0<=record[0]<=1:outside_unit_auc+=1
            if key in values:
                if values[key]!=record:conflicts.add(key)
                else:duplicate_same+=1
            else:values[key]=record
    drugs=sorted(names,key=lambda x:int(x));lookup={v:i for i,v in enumerate(drugs)};rows={v:i for i,v in enumerate(ids)}
    auc=np.full((len(ids),len(drugs)),np.nan);ic50=auc.copy()
    for (sid,drug),record in values.items():
        if (sid,drug) in conflicts:continue
        auc[rows[sid],lookup[drug]]=record[0];ic50[rows[sid],lookup[drug]]=record[1]
    quality=dict(source_rows=total,whitelisted_source_rows=matched,datasets=dict(datasets),webreleases=dict(releases),
        samples_requested=len(ids),samples_with_response=int(np.isfinite(auc).any(1).sum()),drug_ids=len(drugs),
        observed_pairs=int(np.isfinite(auc).sum()),missing_pairs=int(np.isnan(auc).sum()),invalid_rows=invalid,
        exact_duplicate_rows=duplicate_same,conflicting_keys=len(conflicts),outside_unit_auc_rows_retained=outside_unit_auc,
        drug_names_ambiguous={k:sorted(v) for k,v in names.items() if len(v)!=1},
        observations_per_drug={k:int(np.isfinite(auc[:,lookup[k]]).sum()) for k in drugs},
        conflicting_pairs=[list(k) for k in sorted(conflicts)])
    annotations={drug:dict(names=sorted(names[drug]),putative_targets=sorted(targets[drug])) for drug in drugs}
    return auc,ic50,np.array(drugs),annotations,quality


def main():
    p=argparse.ArgumentParser()
    for k in ('source','data','output'):p.add_argument('--'+k,required=True,type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    assert sha(args.source)==SOURCE_SHA
    assert sha(args.data/'manifest.json')=='1b44f54110724f851c7240c252a5fec5abe5362a753c656b4b1e341041ba2936'
    manifest=json.loads((args.data/'manifest.json').read_text())
    assert sha(args.data/'sample_ids.npy')==manifest['files']['sample_ids.npy']['sha256']
    ids=np.load(args.data/'sample_ids.npy',allow_pickle=False);assert len(ids)==len(set(ids))==421
    auc,ic50,drugs,annotations,quality=extract(args.source,ids.tolist())
    for name,value in dict(sample_ids=ids,drug_ids=drugs,auc=auc,ln_ic50=ic50).items():np.save(args.output/(name+'.npy'),value,allow_pickle=False)
    (args.output/'drug_annotations.json').write_text(json.dumps(annotations,indent=2))
    (args.output/'quality.json').write_text(json.dumps(quality,indent=2))
    out=dict(scope='GDSC2 exact-ID outcomes for isolated 421 development samples only',scientific_claim=False,
             source_sha256=SOURCE_SHA,existing_data_sha256=sha(args.data/'manifest.json'),
             primary_endpoint='AUC, native fitted dose-response area; not a patient treatment recommendation',
             secondary_endpoint='LN_IC50 retained, not evaluated by the primary plan',
             sample_link='exact SANGER_MODEL_ID only; not old fuzzy name mapping',
             missing='NaN; never imputed outcomes',replicates='conflicting AUC/IC50/dose-range duplicate keys excluded',
             files={f.name:dict(sha256=sha(f),bytes=f.stat().st_size) for f in args.output.iterdir() if f.is_file()})
    (args.output/'manifest.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in quality.items() if k not in ('observations_per_drug','conflicting_pairs','drug_names_ambiguous')}),flush=True)


if __name__=='__main__':main()

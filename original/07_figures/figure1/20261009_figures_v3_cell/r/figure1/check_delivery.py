"""Check table denominators and exported vector bounds, independently of R layout code."""
from pathlib import Path
import csv, json, hashlib
import numpy as np
import pymupdf as fitz
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'delivery_R_20261007/figure1_data_overview'; DATA=OUT/'plot_data'
def read(n):return list(csv.DictReader((DATA/(n+'.csv')).open()))
meta=read('sample_metadata');fold_n={f:sum(int(x['fold'])==f for x in meta) for f in range(5)}
assert len(meta)==len({x['sample_id'] for x in meta})==421
assert sum(int(x['n']) for x in read('B_tissue_composition'))==421
assert sum(int(x['n']) for x in read('B_fold_composition'))==421
flows=read('F_tissue_to_fold');assert len(flows)==22 and sum(int(x['n']) for x in flows)==421
for x in flows:
 samples=[z for z in meta if z['tissue']==x['tissue']]
 assert len(samples)==int(x['n']) and {z['fold'] for z in samples}=={x['fold']}
d=read('D_entry_coverage');h=read('H_sample_coverage_pca')
for a in ['Protein','Drug AUC']:
 for f in range(5):
  hh=[float(z['observed_pct']) for z in h if z['asset']==a and int(z['fold'])==f]
  dd=next(z for z in d if z['asset']==a and int(z['fold'])==f and z['status']=='Observed')
  assert len(hh)==fold_n[f] and abs(np.mean(hh)-float(dd['pct']))<1e-9
for z in read('E_protein_by_tissue'):
 assert int(z['n_observed'])<=int(z['n'])
 assert abs(float(z['observed_pct'])-100*int(z['n_observed'])/int(z['n']))<1e-9
pca=read('C_pca_coordinates');assert len(pca)==842
assert all(np.isfinite([float(z['pc1']),float(z['pc2'])]).all() for z in pca)
assert all(z['partition']==('Held out' if z['fold']=='0' else 'Training') for z in pca)
for z in read('G_metabolite_tissue_means'):assert 0<=float(z['scaled_mean'])<=1
assert len(read('G_metabolite_tissue_means'))==20*22
outbounds=[];pdfs=[]
for path in sorted((OUT/'panels').glob('*.pdf'))+list((OUT/'main').glob('*.pdf')):
 doc=fitz.open(path);assert len(doc)==1
 page=doc[0];texts=[]
 for block in page.get_text('dict')['blocks']:
  for line in block.get('lines',[]):
   for span in line['spans']:
    texts.append(span['text']);r=fitz.Rect(span['bbox'])
    if r.x0<-.4 or r.y0<-.4 or r.x1>page.rect.width+.4 or r.y1>page.rect.height+.4:outbounds.append({'file':path.name,'text':span['text'],'bbox':list(r)})
 assert texts,path
 assert not any(t.strip() in ['NA','NaN','Inf'] for t in texts),path
 pdfs.append({'file':str(path.relative_to(OUT)),'width_mm':page.rect.width/72*25.4,'height_mm':page.rect.height/72*25.4,'text_spans':len(texts),'raster_images':len(page.get_images())})
 assert len(page.get_images())==0,path
assert not outbounds,outbounds
main=Image.open(OUT/'main/Figure1_CREED_data_overview.png');assert main.size==(4251,7364)
# No raster substitution in editable SVG; all plotted content remains vector.
for path in list((OUT/'panels').glob('*.svg'))+list((OUT/'main').glob('*.svg')):assert '<image ' not in path.read_text()
receipt={'status':'PASS','sample_count':421,'tissue_count':22,'fold_counts':fold_n,'checks':['unique IDs','composition counts','tissue-to-fold exclusivity','D and H coverage consistency','E observed denominators','PCA count and finite coordinates','G range and dimensions','PDF text inside page','no literal missing-value labels','PDF and SVG vector content','600 dpi PNG dimensions'],'pdfs':pdfs,'png_size':main.size,'manual_visual_review':'See VISUAL_REVIEW.md; automated checks do not establish publication suitability.'}
(OUT/'qa/AUTOMATED_CHECKS.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))

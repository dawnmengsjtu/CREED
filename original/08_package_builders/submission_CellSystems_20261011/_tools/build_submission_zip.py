"""Assemble the upload-ready submission zip (deterministic) with a SHA-256 manifest.

Included: 00 README (Chinese guide), 01 manuscript (.docx), 02 Document S1 (.pdf for upload, .docx editable source),
03 Excel tables, 04 Data S1–S10, 05 main-figure PDFs, 06 cover letter, 07 Highlights/eTOC, 08 Methods S1.
Excluded on purpose: TIFFs (initial-submission size limit; regenerate for revision), the graphical abstract (optional at
first submission and not yet compliant with the GA guide), QA_CHECKLIST.md, _tools/, _render_check/.
Usage: python3 -I -B build_submission_zip.py [output.zip]
"""
import hashlib
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.environ.get('CREED_SUBMISSION_DIR', os.path.dirname(HERE))
ROOT = os.path.dirname(PKG)
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'CREED_CellSystems_submission_20261011.zip')
TOP = os.path.basename(PKG) if os.environ.get('CREED_SUBMISSION_DIR') else 'CREED_CellSystems_submission_20261011'
FILES = ['00_README_投稿包说明.md', '01_Manuscript_CREED_CellSystems.docx', '02_Supplemental_Information_DocumentS1.pdf',
         '02_Supplemental_Information_DocumentS1.docx']
FILES += ['03_Tables_S/Table_S%s.xlsx' % n for n in ('3', '4', '8', '11', '12', '13', '14')]
FILES += ['04_Data_S/Data_S%d.zip' % n for n in range(1, 11)]
FILES += ['05_Figures/Figure%d.pdf' % n for n in range(1, 7)]
FILES += ['06_Cover_Letter_CREED.docx', '07_Highlights_eTOC.docx', '08_Methods_S1_code.zip']
STAMP = tuple(map(int, os.environ.get('CREED_SUBMISSION_DATE', '2026-10-11').split('-'))) + (0, 0, 0)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    rows, total = [], 0
    for f in FILES:
        p = os.path.join(PKG, f)
        if not os.path.isfile(p):
            sys.exit('missing deliverable: ' + f)
        rows.append((f, os.path.getsize(p), sha(p)))
        total += os.path.getsize(p)
    manifest = 'SHA-256\tbytes\tfile\n' + ''.join('%s\t%d\t%s\n' % (h, n, f) for f, n, h in rows)
    if os.path.exists(OUT):
        os.remove(OUT)
    with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f, n, h in rows:
            zi = zipfile.ZipInfo('%s/%s' % (TOP, f), STAMP)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, open(os.path.join(PKG, f), 'rb').read())
        zi = zipfile.ZipInfo('%s/SHA256SUMS.tsv' % TOP, STAMP)
        zi.compress_type = zipfile.ZIP_DEFLATED
        zi.external_attr = 0o644 << 16
        z.writestr(zi, manifest.encode())
    with zipfile.ZipFile(OUT) as z:
        assert z.testzip() is None
        for f, n, h in rows:
            assert hashlib.sha256(z.read('%s/%s' % (TOP, f))).hexdigest() == h
    print('%s: %d files, deliverables %.2f MB, zip %.2f MB, sha256 %s' % (OUT, len(rows) + 1, total / 1e6, os.path.getsize(OUT) / 1e6, sha(OUT)))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Stdlib-only Markdown -> .docx writer for the CREED manuscript (v1.8, figures_v4).

Builds
  (a) CREED_manuscript_v2_CellSystems.docx : title, abstract, main text (Results tables and
      Table 1 as real Word tables), Discussion, Methods, then a 'Figures' section with each
      main figure (Figures 1-6) on its own page followed by its legend, then the
      Supplementary Information list.
  (b) CREED_Supplementary_Information.docx : Figures S1-S5, each on its own page with its
      legend, then every table of SUPPLEMENTARY_TABLES.md as real Word tables.

Input text = the clean manuscript (no [src: ...] tags). Images = PNGs downsampled with
'sips -Z 2400' into ./img/ (Figure1.png ... FigureS5.png). Images are inline, 16.5 cm wide
(EMU = cm x 360000), aspect ratio kept. Only zipfile, struct, re, os, sys, datetime used.

v1.8: Supplementary Figures S1-S5 are numbered by first citation (figures_v4/RENUMBERING_v4.md §6; img/ renamed
to match). Submission filters: the main text gets two exact edits (no internal paths); the SI tables are built
from SUPPLEMENTARY_TABLES.md minus its internal notes, [src: ...] tags, provenance sentences and every 'Source'
column (the master file keeps all provenance). After writing, both files are re-read and checked for internal
paths, [src:, <!--, and PENDING outside genuine [[PENDING: ...]] markers (none at all in the SI); the build exits
non-zero if a check fails.
"""
import os, re, struct, sys, zipfile, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
MS = '/Users/xlzhexue/Desktop/proj/creed/manuscript_v0'
CLEAN = os.path.join(MS, 'CREED_manuscript_v0_clean.md')
SUPT = os.path.join(MS, 'SUPPLEMENTARY_TABLES.md')
IMG = os.path.join(HERE, 'img')
OUT_MAIN = os.path.join(MS, 'CREED_manuscript_v2_CellSystems.docx')
OUT_SI = os.path.join(MS, 'CREED_Supplementary_Information.docx')

EMU_PER_CM = 360000
IMG_W_CM = 16.5

# ---------------------------------------------------------------- helpers

def esc(s):
    return (s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
             .replace('"', '&quot;'))

def png_size(path):
    with open(path, 'rb') as f:
        head = f.read(24)
    if head[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('not a PNG: ' + path)
    w, h = struct.unpack('>II', head[16:24])
    return w, h

def run(text, bold=False, italic=False, code=False, size=None, highlight=False):
    rpr = ''
    if code:
        rpr += '<w:rFonts w:ascii="Courier New" w:hAnsi="Courier New" w:cs="Courier New"/>'
    if bold:
        rpr += '<w:b/><w:bCs/>'
    if italic:
        rpr += '<w:i/><w:iCs/>'
    if highlight:
        rpr += '<w:highlight w:val="yellow"/>'
    if size:
        rpr += '<w:sz w:val="%d"/><w:szCs w:val="%d"/>' % (size, size)
    rpr = '<w:rPr>%s</w:rPr>' % rpr if rpr else ''
    return '<w:r>%s<w:t xml:space="preserve">%s</w:t></w:r>' % (rpr, esc(text))

TOKEN = re.compile(r'(\*\*.+?\*\*|`[^`]+`|(?<![\w*])\*(?!\s)[^*\n]+?(?<!\s)\*(?![\w*])|\[\[PENDING[^\]]*\]\])')

def inline_runs(text, size=None, base_bold=False):
    """Markdown inline -> runs: **bold**, *italic*, `code`, [[PENDING]] highlighted."""
    text = text.replace('\\*', '∗STAR∗')
    out = []
    pos = 0
    for m in TOKEN.finditer(text):
        if m.start() > pos:
            out.append(('t', text[pos:m.start()]))
        tok = m.group(0)
        if tok.startswith('**'):
            out.append(('b', tok[2:-2]))
        elif tok.startswith('`'):
            out.append(('c', tok[1:-1]))
        elif tok.startswith('[['):
            out.append(('p', tok))
        else:
            out.append(('i', tok[1:-1]))
        pos = m.end()
    if pos < len(text):
        out.append(('t', text[pos:]))
    runs = []
    for kind, s in out:
        s = s.replace('∗STAR∗', '*')
        if kind == 'b':
            # nested italics inside bold
            for k2, s2 in _split_italic(s):
                runs.append(run(s2, bold=True, italic=(k2 == 'i'), size=size))
        elif kind == 'i':
            runs.append(run(s, italic=True, bold=base_bold, size=size))
        elif kind == 'c':
            runs.append(run(s, code=True, bold=base_bold, size=size))
        elif kind == 'p':
            runs.append(run(s, bold=base_bold, size=size, highlight=True))
        else:
            runs.append(run(s, bold=base_bold, size=size))
    return ''.join(runs)

def _split_italic(s):
    parts = []
    pos = 0
    for m in re.finditer(r'(?<![\w*])\*(?!\s)[^*]+?(?<!\s)\*(?![\w*])', s):
        if m.start() > pos:
            parts.append(('t', s[pos:m.start()]))
        parts.append(('i', m.group(0)[1:-1]))
        pos = m.end()
    if pos < len(s):
        parts.append(('t', s[pos:]))
    return parts

def para(text, style=None, align=None, size=None, keep_next=False, indent=None, bold=False,
         space_after=None):
    ppr = ''
    if style:
        ppr += '<w:pStyle w:val="%s"/>' % style
    if keep_next:
        ppr += '<w:keepNext/>'
    if space_after is not None:
        ppr += '<w:spacing w:after="%d"/>' % space_after
    if indent:
        ppr += '<w:ind w:left="%d" w:hanging="%d"/>' % indent
    if align:
        ppr += '<w:jc w:val="%s"/>' % align
    ppr = '<w:pPr>%s</w:pPr>' % ppr if ppr else ''
    return '<w:p>%s%s</w:p>' % (ppr, inline_runs(text, size=size, base_bold=bold))

def page_break():
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'

def table_xml(rows, size=18):
    """rows: list of lists of cell strings; first row = header."""
    ncol = max(len(r) for r in rows)
    total = 9638  # 17.0 cm text width in twips
    colw = total // ncol
    b = 'w:val="single" w:sz="4" w:space="0" w:color="000000"'
    x = ['<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="5000" w:type="pct"/>'
         '<w:tblBorders><w:top %s/><w:left %s/><w:bottom %s/><w:right %s/>'
         '<w:insideH %s/><w:insideV %s/></w:tblBorders><w:tblLayout w:type="autofit"/>'
         '<w:tblCellMar><w:left w:w="57" w:type="dxa"/><w:right w:w="57" w:type="dxa"/></w:tblCellMar>'
         '<w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="1" w:lastColumn="0" w:noHBand="0" w:noVBand="1"/>'
         '</w:tblPr>' % (b, b, b, b, b, b)]
    x.append('<w:tblGrid>' + ''.join('<w:gridCol w:w="%d"/>' % colw for _ in range(ncol)) + '</w:tblGrid>')
    for ri, r in enumerate(rows):
        r = r + [''] * (ncol - len(r))
        trpr = '<w:trPr><w:tblHeader/></w:trPr>' if ri == 0 else ''
        x.append('<w:tr>%s' % trpr)
        for c in r:
            shade = '<w:shd w:val="clear" w:color="auto" w:fill="E7E6E6"/>' if ri == 0 else ''
            x.append('<w:tc><w:tcPr><w:tcW w:w="%d" w:type="dxa"/>%s</w:tcPr>' % (colw, shade))
            x.append('<w:p><w:pPr><w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/></w:pPr>%s</w:p>'
                     % inline_runs(c, size=size, base_bold=(ri == 0)))
            x.append('</w:tc>')
        x.append('</w:tr>')
    x.append('</w:tbl>')
    x.append('<w:p><w:pPr><w:spacing w:after="120"/></w:pPr></w:p>')
    return ''.join(x)

def split_row(line):
    s = line.strip()
    if s.startswith('|'):
        s = s[1:]
    if s.endswith('|'):
        s = s[:-1]
    # split on unescaped pipes
    cells = re.split(r'(?<!\\)\|', s)
    return [c.strip().replace('\\|', '|') for c in cells]

# ---------------------------------------------------------------- markdown -> body XML

def md_blocks(md, heading_map=None, table_size=18, para_size=None):
    """Convert a markdown string to a list of WordprocessingML block strings."""
    heading_map = heading_map or {1: 'Title', 2: 'Heading1', 3: 'Heading2', 4: 'Heading3'}
    md = re.sub(r'<!--.*?-->', '', md, flags=re.S)
    lines = md.split('\n')
    out = []
    i = 0
    buf = []

    def flush():
        if buf:
            out.append(para(' '.join(x.strip() for x in buf), size=para_size))
            buf.clear()

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s:
            flush(); i += 1; continue
        if re.fullmatch(r'-{3,}|\*{3,}', s):
            flush(); i += 1; continue
        m = re.match(r'^(#{1,6})\s+(.*)$', s)
        if m:
            flush()
            lvl = len(m.group(1))
            out.append(para(m.group(2).strip(), style=heading_map.get(lvl, 'Heading3'), keep_next=True))
            i += 1; continue
        if s.startswith('|'):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                r = lines[i].strip()
                if not re.fullmatch(r'\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?', r):
                    rows.append(split_row(r))
                i += 1
            out.append(table_xml(rows, size=table_size))
            continue
        lm = re.match(r'^(\s*)([-*]|\d+\.)\s+(.*)$', ln)
        if lm:
            flush()
            depth = len(lm.group(1).expandtabs(2)) // 2
            bullet = '•' if lm.group(2) in ('-', '*') else lm.group(2)
            left = 360 + 360 * depth
            out.append(para(bullet + '\t' + lm.group(3), indent=(left, 360), size=para_size))
            i += 1; continue
        buf.append(ln)
        i += 1
    flush()
    return out

def image_block(rid, idx, path, name):
    w, h = png_size(path)
    cx = int(IMG_W_CM * EMU_PER_CM)
    cy = int(round(cx * h / w))
    return ('<w:p><w:pPr><w:spacing w:before="0" w:after="120"/><w:jc w:val="center"/></w:pPr><w:r><w:drawing>'
            '<wp:inline distT="0" distB="0" distL="0" distR="0">'
            '<wp:extent cx="%d" cy="%d"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
            '<wp:docPr id="%d" name="%s"/>'
            '<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            '<pic:pic><pic:nvPicPr><pic:cNvPr id="%d" name="%s"/><pic:cNvPicPr/></pic:nvPicPr>'
            '<pic:blipFill><a:blip r:embed="%s"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            '<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="%d" cy="%d"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>'
            '</a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'
            % (cx, cy, idx, esc(name), idx, esc(name), rid, cx, cy))

# ---------------------------------------------------------------- submission filters (v1.8)
# The markdown masters keep their provenance; these filters act only on what goes into the
# Word files. Every exact edit must match the stated number of times, so a change in the
# master makes the build stop instead of silently letting an internal note through.

MAIN_SUBMISSION_EDITS = [
    # Supplementary Information list: internal file location of the compiled tables
    ('**Supplementary tables** (compiled in manuscript_v0/SUPPLEMENTARY_TABLES.md; numbered in order of first citation)',
     '**Supplementary tables** (compiled in the Supplementary Information document; numbered in order of first citation)', 1),
    # local script path inside a genuine [[PENDING]] marker (Methods, CPTAC)
    ('to be written from figure_pipeline/r/figures2_6/prepare_cptac.py and the source-shift run protocol',
     'to be written from prepare_cptac.py and the source-shift run protocol', 1),
    # Supplementary Information list: figure-set version history
    ('The former protein-reconstruction figure (v3 Supplementary Figure S2) is not part of the v4 set; '
     'its generator diagnostics are in Supplementary Data 4. ', '', 1),
    # internal ledger file names (Methods N4; Table S4 entry of the Supplementary Information list)
    (', plus the 2026-10-09 runs appended as H080–H091 in registries_additions_20261009.tsv)',
     ', plus the 2026-10-09 runs appended as H080–H091)', 1),
    ('every EXPERIMENT_LEDGER.tsv row with completed_negative status plus the 2026-10-09 rows H080–H093 of '
     'registries_additions_20261009.tsv.',
     'every experiment-ledger row with completed_negative status plus the 2026-10-09 rows H080–H093.', 1),
]

# old supplementary names in the Supplementary Information list (workflow history):
# 14 '(formerly [Supplementary ]Table S-…)' notes and 2 that carry further content after ';'
MAIN_SUBMISSION_REGEX = [
    (r' \(formerly (?:Supplementary )?Table S-[\w-]+\)', '', 14),
    (r'\(formerly Table S-[\w-]+; ', '(', 2),
]

def apply_regex(md, rules, label):
    for pat, rep, n in rules:
        md, k = re.subn(pat, rep, md)
        if k != n:
            sys.exit('%s submission regex matched %d times (expected %d): %s' % (label, k, n, pat))
    return md

SI_DROP_PARAGRAPHS = (
    'Numbering (v1.7',            # old-to-new table-number map (workflow history)
)

SI_SUBMISSION_EDITS = [
    # header note: keep only its scope sentence
    ('Companion to `CREED_manuscript_v0.md`. Every value is copied from the file named in the `Source` column; '
     'nothing was recomputed. Ledger status labels are those of `handoff_20261009/registries/EXPERIMENT_LEDGER.tsv` '
     '(H001–H079) and `registries_additions_20261009.tsv` (H080–H094). ', '', 1),
    # Table S1
    (' [tissue_kernel_20261009/RESULT.md §2, §4]', '', 1),
    (' [CREED_系统诊断报告_20261009.md §2 根因3]', '', 1),
    # Table S2
    ('Source: `generator_curve_20261009/RESULT.md` §4 (all columns) and §2 (generator sizes and out-of-fold R²); '
     '`generator_curve_20261009/DECISION.md` §2 (five-fold mean n, rounded share table). Nothing was recomputed. ', '', 1),
    ("Absolute per-arm sRMSE values are in the run's `fold_metrics.csv` and in RESULT.md §3; the f100 arms",
     "Absolute per-arm sRMSE values are in the run's `fold_metrics.csv`; the f100 arms", 1),
    # Table S3
    ('Source: `generator_fullpool_20261009/pack/EXTRACTION.md` §1–§5 and `generator_fullpool_20261009/pack/qa.json`. ', '', 1),
    ('This closes the earlier blind spot that the raw sources had never been read exhaustively: the 67 is an enumeration, not an inherited count',
     'The 67 is an exhaustive enumeration of the raw sources, not an inherited count', 1),
    # Table S4
    (' (21 rows, matching the 21 closed negatives counted in CREED_系统诊断报告_20261009.md §4)', ' (21 rows)', 1),
    ('; see `registries_additions_20261009.tsv`.', '.', 1),
    # Table S5
    ('Run `crispr-endpoint-20261009-0001`. ', '', 1),
    (' Source: crispr_endpoint_20261009/result_tables.md (new_cell and tissue "5 折均值 sRMSE（全部臂）" tables); '
     'key rows and the per-fold values also in crispr_endpoint_20261009/RESULT.md §3.', '', 1),
    # Table S6
    ('fixed in `crispr_endpoint_20261009/PLAN.md` §8 before', 'fixed in the pre-registered plan before', 1),
    (' Source: crispr_endpoint_20261009/RESULT.md §6; crispr_endpoint_20261009/DECISION.md §4.', '', 1),
    (' Source: crispr_endpoint_20261009/RESULT.md §7.', '', 2),
    (' Source: crispr_endpoint_20261009/RESULT.md §5 预注册的阴性; protein_panel_20261009/RESULT.md §8 比较 tables.', '', 1),
    # Table S7
    (' (formerly embedded in v3 Figure 6A)', '', 1),
    (' Per-fold values (10 folds × 6 arms) are in the v3 Figure 6 (now Figure S4) plot table `arm_scores.csv`.', '', 1),
    (' (cross-layer-matrix RESULT table, not recomputed here)', ' (from the cross-layer matrix analysis)', 1),
    # Tables S8 and S9
    ('Protocol `CREED_INTERNAL_FINAL_ONESHOT_V1` (internal_final_20261009/PLAN.md, registered',
     'Protocol `CREED_INTERNAL_FINAL_ONESHOT_V1` (registered', 2),
    (" Values are copied row by row from the frozen runner's `internal_final_20261009/output/RESULT_DRAFT.md` "
     "(generated from `output/summary.json`); nothing was recomputed here.", '', 2),
    ('The independent recomputation (`internal_final_20261009/RECOMPUTE.md`) reproduced',
     'The independent recomputation reproduced', 2),
    (' Source: internal_final_20261009/output/RESULT_DRAFT.md §药敏; internal_final_20261009/output/summary.json '
     '§analysis.drug; internal_final_20261009/RECOMPUTE.md §2.1, §3.1.', '', 1),
    ('(PLAN §6, no pass line)', '(pre-registered protocol; no pass line)', 1),
    (' Source: internal_final_20261009/output/summary.json §analysis.F; internal_final_20261009/RECOMPUTE.md §4 '
     '(independent re-derivation, no disagreement).',
     ' The independent recomputation re-derived these verdicts with no disagreement.', 1),
    ("; the runner's draft wrote '123 cells' for this endpoint and the independent recomputation corrected it to 95)", ')', 1),
    (' Source: internal_final_20261009/output/RESULT_DRAFT.md §CRISPR; internal_final_20261009/output/summary.json '
     '§analysis.crispr, §meta.gates.crispr_123; internal_final_20261009/RECOMPUTE.md §0, §2.2, §3.2.', '', 1),
    (' Source: internal_final_20261009/output/summary.json §analysis.F.F8; internal_final_20261009/RECOMPUTE.md §4.', '', 1),
]

def apply_edits(md, edits, label):
    for old, new, n in edits:
        k = md.count(old)
        if k != n:
            sys.exit('%s submission edit matched %d times (expected %d): %r' % (label, k, n, old[:90]))
        md = md.replace(old, new)
    return md

def drop_columns(md, names=('Source',)):
    """Remove every table column whose header cell is one of `names` (provenance columns)."""
    lines = md.split('\n')
    out, i = [], 0
    while i < len(lines):
        if not lines[i].strip().startswith('|'):
            out.append(lines[i]); i += 1; continue
        j = i
        while j < len(lines) and lines[j].strip().startswith('|'):
            j += 1
        block = [split_row(l) for l in lines[i:j]]
        drop = {k for k, c in enumerate(block[0]) if c in names}
        for row in block:
            out.append('| ' + ' | '.join(c for k, c in enumerate(row) if k not in drop) + ' |')
        i = j
    return '\n'.join(out)

def si_submission(md):
    md = re.sub(r'^# .*\n', '', md, count=1)                 # file title (the docx has its own heading)
    paras = re.split(r'(\n\s*\n)', md)
    md = ''.join(p for p in paras if not p.lstrip().startswith(SI_DROP_PARAGRAPHS))
    md = apply_edits(md, SI_SUBMISSION_EDITS, 'SI')
    md = re.sub(r'\s*\[src:[^\]]*\]', '', md)                 # provenance tags
    md = drop_columns(md, ('Source',))                        # provenance columns
    return md

# ---------------------------------------------------------------- docx text checks

FORBIDDEN_BOTH = ['/Users/', '/public/', 'creed-research', '[src:', '<!--', 'figures_v', 'manuscript_v0',
                  'figure_pipeline', 'handoff_2026', 'scratchpad']
PATHLIKE = re.compile(r'\b[\w\-]+_2026\d{4}/|\b[\w\-]+/[\w\-/]+\.(?:md|py|R|tsv|json|csv|txt)\b')
SI_EXTRA = re.compile(r'\b[\w\-]+\.(?:md|py|R|tsv|txt)\b|-2026\d{4}-\d{4}|RECOMPUTE|RESULT_DRAFT|DECISION|'
                      r'\bv3\b|v1\.7|formerly|PENDING|Companion to|Numbering \(')
MAIN_EXTRA = re.compile(r'formerly|\bv[1-4] (?:set|figure|Supplementary|legend|caption)|\bthe v[1-4]\b|'
                        r'(?:[Ff]igures?|set) v[1-4]\b|\b[\w\-]+\.tsv\b|_caption|caption file')
SI_DATA_FILES = {'Model.csv', 'fold_metrics.csv', 'comparisons.csv', 'summary.json'}
PATH_OK = {'mmc2/Model.csv'}  # 'conflicts between the mmc2 and Model.csv source tables', not a path

def docx_text(path):
    with zipfile.ZipFile(path) as z:
        x = z.read('word/document.xml').decode('utf-8')
    x = x.replace('</w:p>', '\n').replace('<w:tab/>', '\t')
    x = re.sub(r'<[^>]+>', '', x)
    return (x.replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"').replace('&amp;', '&'))

def check_docx(path, label, strict):
    t = docx_text(path)
    bad = [s for s in FORBIDDEN_BOTH if s in t] + [m for m in PATHLIKE.findall(t) if m not in PATH_OK]
    pend = re.findall(r'\[\[PENDING:[^\]]*\]\]', t)
    if t.count('PENDING') != len(pend):
        bad.append('PENDING outside a [[PENDING: ...]] marker')
    if not strict:
        bad += MAIN_EXTRA.findall(t)
    if strict:
        bad += SI_EXTRA.findall(t)
        bad += [f for f in re.findall(r'\b[\w\-]+\.(?:csv|json)\b', t) if f not in SI_DATA_FILES]
    print('%s check: %s; [[PENDING]] markers: %d' % (label, 'PASS' if not bad else 'FAIL ' + repr(sorted(set(bad))), len(pend)))
    return not bad, pend

# ---------------------------------------------------------------- package parts

NS = ('xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
      'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
      'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
      'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
      'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"')

STYLES = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:docDefaults>
 <w:rPrDefault><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="Times New Roman" w:cs="Times New Roman"/><w:sz w:val="22"/><w:szCs w:val="22"/><w:lang w:val="en-US" w:eastAsia="zh-CN"/></w:rPr></w:rPrDefault>
 <w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="276" w:lineRule="auto"/></w:pPr></w:pPrDefault>
</w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/><w:pPr><w:jc w:val="both"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:jc w:val="left"/><w:spacing w:before="0" w:after="240"/></w:pPr><w:rPr><w:b/><w:bCs/><w:sz w:val="34"/><w:szCs w:val="34"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:jc w:val="left"/><w:spacing w:before="360" w:after="120"/><w:outlineLvl w:val="0"/></w:pPr><w:rPr><w:b/><w:bCs/><w:sz w:val="30"/><w:szCs w:val="30"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:jc w:val="left"/><w:spacing w:before="240" w:after="120"/><w:outlineLvl w:val="1"/></w:pPr><w:rPr><w:b/><w:bCs/><w:sz w:val="26"/><w:szCs w:val="26"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:jc w:val="left"/><w:spacing w:before="200" w:after="80"/><w:outlineLvl w:val="2"/></w:pPr><w:rPr><w:b/><w:bCs/><w:i/><w:iCs/><w:sz w:val="23"/><w:szCs w:val="23"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Legend"><w:name w:val="Legend"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:after="100" w:line="252" w:lineRule="auto"/></w:pPr><w:rPr><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr></w:style>
<w:style w:type="table" w:default="1" w:styleId="TableNormal"><w:name w:val="Normal Table"/><w:uiPriority w:val="99"/><w:semiHidden/><w:tblPr><w:tblInd w:w="0" w:type="dxa"/><w:tblCellMar><w:top w:w="0" w:type="dxa"/><w:left w:w="108" w:type="dxa"/><w:bottom w:w="0" w:type="dxa"/><w:right w:w="108" w:type="dxa"/></w:tblCellMar></w:tblPr></w:style>
<w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:basedOn w:val="TableNormal"/><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/><w:jc w:val="left"/></w:pPr><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4" w:space="0" w:color="000000"/><w:left w:val="single" w:sz="4" w:space="0" w:color="000000"/><w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/><w:right w:val="single" w:sz="4" w:space="0" w:color="000000"/><w:insideH w:val="single" w:sz="4" w:space="0" w:color="000000"/><w:insideV w:val="single" w:sz="4" w:space="0" w:color="000000"/></w:tblBorders></w:tblPr></w:style>
</w:styles>'''

SETTINGS = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:defaultTabStop w:val="360"/><w:compat><w:compatSetting w:name="compatibilityMode" w:uri="http://schemas.microsoft.com/office/word" w:val="15"/></w:compat></w:settings>'''

def write_docx(path, blocks, images, title):
    """blocks: list of XML strings (may contain {IMG:n} placeholders already resolved);
    images: list of (rid, target_name, source_path)."""
    sect = ('<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
            '<w:pgMar w:top="1418" w:right="1134" w:bottom="1418" w:left="1134" w:header="709" w:footer="709" w:gutter="0"/>'
            '</w:sectPr>')
    doc = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<w:document %s><w:body>%s%s</w:body></w:document>'
           % (NS, ''.join(blocks), sect))
    rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
            '<Relationship Id="rIdSettings" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>']
    for rid, name, _ in images:
        rels.append('<Relationship Id="%s" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/%s"/>' % (rid, name))
    rels.append('</Relationships>')
    ctypes = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
              '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
              '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
              '<Default Extension="xml" ContentType="application/xml"/>'
              '<Default Extension="png" ContentType="image/png"/>'
              '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
              '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
              '<Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>'
              '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
              '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
              '</Types>')
    root_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                 '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                 '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
                 '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
                 '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
                 '</Relationships>')
    now = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    core = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            '<dc:title>%s</dc:title><dc:creator>CREED</dc:creator>'
            '<dcterms:created xsi:type="dcterms:W3CDTF">%s</dcterms:created>'
            '<dcterms:modified xsi:type="dcterms:W3CDTF">%s</dcterms:modified></cp:coreProperties>' % (esc(title), now, now))
    app = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
           '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
           '<Application>build_docx.py (stdlib)</Application></Properties>')
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', ctypes)
        z.writestr('_rels/.rels', root_rels)
        z.writestr('docProps/core.xml', core)
        z.writestr('docProps/app.xml', app)
        z.writestr('word/document.xml', doc)
        z.writestr('word/styles.xml', STYLES)
        z.writestr('word/settings.xml', SETTINGS)
        z.writestr('word/_rels/document.xml.rels', ''.join(rels))
        for rid, name, src in images:
            z.write(src, 'word/media/' + name, compress_type=zipfile.ZIP_STORED)

# ---------------------------------------------------------------- manuscript splitting

def sections(md):
    """Split on '## ' headings -> ordered list of (heading, text including heading)."""
    md = re.sub(r'<!--.*?-->', '', md, flags=re.S)
    parts = re.split(r'(?m)^(?=## )', md)
    res = []
    for p in parts:
        m = re.match(r'## (.*)', p)
        res.append((m.group(1).strip() if m else '', p))
    return res

def legend_blocks(section_text):
    """'## Figure legends' text -> {fig_id: markdown legend}."""
    out = {}
    chunks = re.split(r'(?m)^(?=\*\*Figure S?\d+\.)', section_text)
    for c in chunks:
        m = re.match(r'\*\*Figure (S?\d+)\.', c)
        if m:
            out[m.group(1)] = c.strip()
    return out

def legend_xml(md):
    paras = [p.strip() for p in re.split(r'\n\s*\n', md) if p.strip()]
    x = []
    for k, p in enumerate(paras):
        p = ' '.join(l.strip() for l in p.split('\n'))
        if k == 0:
            x.append(para(p, style='Legend', keep_next=True))
        else:
            x.append(para(p, style='Legend'))
    return x

def build_main():
    md = open(CLEAN, encoding='utf-8').read()
    md = apply_edits(md, MAIN_SUBMISSION_EDITS, 'main')
    md = apply_regex(md, MAIN_SUBMISSION_REGEX, 'main')
    secs = sections(md)
    blocks = []
    legends = {}
    si_text = ''
    for head, text in secs:
        if head == 'Figure legends':
            legends.update(legend_blocks(text)); continue
        if head == 'Supplementary figure legends':
            continue
        if head.startswith('Supplementary Information'):
            si_text = text; continue
        blocks.extend(md_blocks(text))
    images = []
    blocks.append(page_break())
    blocks.append(para('Figures', style='Heading1'))
    for n in range(1, 7):
        fid = str(n)
        src = os.path.join(IMG, 'Figure%s.png' % fid)
        rid = 'rIdImg%d' % n
        name = 'image%d.png' % n
        images.append((rid, name, src))
        if n > 1:
            blocks.append(page_break())
        blocks.append(image_block(rid, n, src, 'Figure %s' % fid))
        blocks.extend(legend_xml(legends[fid]))
    if si_text:
        blocks.append(page_break())
        blocks.extend(md_blocks(si_text))
    title = re.match(r'# (.*)', md).group(1)
    write_docx(OUT_MAIN, blocks, images, title)
    return len(images)

def build_si():
    md = open(CLEAN, encoding='utf-8').read()
    title = re.match(r'# (.*)', md).group(1)
    legends = {}
    for head, text in sections(md):
        if head == 'Supplementary figure legends':
            legends.update(legend_blocks(text))
    blocks = [para('Supplementary Information', style='Title'),
              para(title, bold=True),
              para('Contents: Figures S1–S5 with legends; Tables S1–S9. Supplementary Data 1–10, Supplementary Notes 1–2 and Supplementary Methods are listed in the main manuscript (Supplementary Information list).'),
              para('Supplementary Figures', style='Heading1')]
    images = []
    for k, fid in enumerate(['S1', 'S2', 'S3', 'S4', 'S5'], 1):
        src = os.path.join(IMG, 'Figure%s.png' % fid)
        rid = 'rIdImg%d' % k
        images.append((rid, 'image%d.png' % k, src))
        blocks.append(page_break())
        blocks.append(image_block(rid, k, src, 'Figure %s' % fid))
        blocks.extend(legend_xml(legends[fid]))
    blocks.append(page_break())
    blocks.append(para('Supplementary Tables', style='Heading1'))
    st = si_submission(open(SUPT, encoding='utf-8').read())
    with open(os.path.join(HERE, 'SI_tables_submission_view.md'), 'w', encoding='utf-8') as f:
        f.write(st)  # inspection copy in the scratch dir; not a source file
    blocks.extend(md_blocks(st, heading_map={1: 'Heading1', 2: 'Heading2', 3: 'Heading3', 4: 'Heading3'},
                            table_size=15, para_size=20))
    write_docx(OUT_SI, blocks, images, 'Supplementary Information — ' + title)
    return len(images)

if __name__ == '__main__':
    a = build_main()
    b = build_si()
    print('main images', a, '->', OUT_MAIN)
    print('SI images', b, '->', OUT_SI)
    ok1, pend = check_docx(OUT_MAIN, 'main docx', strict=False)
    for p in pend:
        print('   ', p)
    ok2, _ = check_docx(OUT_SI, 'SI docx', strict=True)
    if not (ok1 and ok2):
        sys.exit('submission check failed')

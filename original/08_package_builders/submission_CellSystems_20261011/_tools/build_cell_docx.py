#!/usr/bin/env python3
"""Stdlib-only Markdown -> .docx builder for the CREED Cell Systems submission (2026-10-11).

Extended from scratchpad/v4_docx/build_docx.py (v1.8; original kept as build_docx_v18_original.py).

Builds, into the submission folder:
  01_Manuscript_CREED_CellSystems.docx      main_text.md + star_methods.md, Cell Press order
  02_Supplemental_Information_DocumentS1.docx  Document S1 part of supplemental_information.md
  06_Cover_Letter_CREED.docx                cover letter (text in this file, numbers from Table 1 / Summary)
  07_Highlights_eTOC.docx                   title, Summary, Highlights, eTOC blurb, keywords

Formatting (manuscript): A4, 2.54 cm top/bottom, 2.0 cm left/right (17.0 cm text width, so figures
fit at 16.5 cm), Arial 12 pt, double spacing, continuous line numbers (w:lnNumType), page number
in the footer, Heading 1-3 styles, real Word tables with single black borders and no shading,
superscript (^x^), italics, bold. Placeholders [[PENDING: ...]] / [[AUTHOR: ...]] are kept
verbatim and highlighted so the authors can find them; ==text== is highlighted without markers (statements
the authors must confirm, e.g. the AI-use sentences).

Only os, re, struct, sys, zipfile, datetime are used. Text is taken verbatim from the masters;
the only content edits are the structural ones listed in STRUCTURAL_EDITS (headings, the
STAR Methods assembly marker) - no number is touched.
"""
import os, re, struct, sys, zipfile, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.environ.get('CREED_SUBMISSION_DIR', os.path.dirname(HERE))
MS = '/Users/xlzhexue/Desktop/proj/creed/manuscript_cellsystems'
MAIN_MD = os.path.join(MS, 'main_text.md')
STAR_MD = os.path.join(MS, 'star_methods.md')
SI_MD = os.path.join(MS, 'supplemental_information.md')
IMG = os.path.join(HERE, 'img')

OUT_MAIN = os.path.join(PKG, '01_Manuscript_CREED_CellSystems.docx')
OUT_SI = os.path.join(PKG, '02_Supplemental_Information_DocumentS1.docx')
OUT_COVER = os.path.join(PKG, '06_Cover_Letter_CREED.docx')
OUT_HL = os.path.join(PKG, '07_Highlights_eTOC.docx')

EMU_PER_CM = 360000
IMG_W_CM = 16.5
FONT = 'Arial'

# ---------------------------------------------------------------- inline markdown

def esc(s):
    return (s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;'))

def run(text, bold=False, italic=False, sup=False, code=False, size=None, highlight=False, sub=False):
    rpr = ''
    if code:
        rpr += '<w:rFonts w:ascii="Courier New" w:hAnsi="Courier New" w:cs="Courier New"/>'
    if bold:
        rpr += '<w:b/><w:bCs/>'
    if italic:
        rpr += '<w:i/><w:iCs/>'
    if size:
        rpr += '<w:sz w:val="%d"/><w:szCs w:val="%d"/>' % (size, size)
    if highlight:
        rpr += '<w:highlight w:val="yellow"/>'
    if sup:
        rpr += '<w:vertAlign w:val="superscript"/>'
    elif sub:
        rpr += '<w:vertAlign w:val="subscript"/>'
    rpr = '<w:rPr>%s</w:rPr>' % rpr if rpr else ''
    parts = text.split('\t')
    out = []
    for k, p in enumerate(parts):
        if k:
            out.append('<w:r>%s<w:tab/></w:r>' % rpr)
        if p:
            out.append('<w:r>%s<w:t xml:space="preserve">%s</w:t></w:r>' % (rpr, esc(p)))
    return ''.join(out)

STAR = ''  # stands in for an escaped \*
TOKENS = [
    ('ph', re.compile(r'\[\[[A-Z]+:?[^\]]*\]\]')),
    ('hl', re.compile(r'==(.+?)==')),
    ('b', re.compile(r'\*\*(.+?)\*\*')),
    ('c', re.compile(r'`([^`]+)`')),
    ('sup', re.compile(r'\^([^\^\s]{1,40})\^')),
    ('sub', re.compile(r'~([^~\s]{1,40})~')),
    ('i', re.compile(r'(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])')),
]

def inline_runs(text, size=None, bold=False, italic=False):
    text = text.replace('\\*', STAR)
    return _inline(text, size, bold, italic)

def _inline(text, size, bold, italic, sup=False, sub=False, hl=False):
    out = []
    pos = 0
    while pos < len(text):
        best = None
        for kind, rx in TOKENS:
            m = rx.search(text, pos)
            if m and (best is None or m.start() < best[1].start()):
                best = (kind, m)
        if best is None:
            break
        kind, m = best
        if m.start() > pos:
            out.append(run(text[pos:m.start()].replace(STAR, '*'), bold, italic, sup, size=size, sub=sub, highlight=hl))
        if kind == 'ph':
            out.append(run(m.group(0).replace(STAR, '*'), bold, italic, sup, size=size, highlight=True))
        elif kind == 'hl':
            out.append(_inline(m.group(1), size, bold, italic, sup, sub, True))
        elif kind == 'b':
            out.append(_inline(m.group(1), size, True, italic, sup, hl=hl))
        elif kind == 'i':
            out.append(_inline(m.group(1), size, bold, True, sup, hl=hl))
        elif kind == 'sup':
            out.append(_inline(m.group(1), size, bold, italic, True, hl=hl))
        elif kind == 'sub':
            out.append(_inline(m.group(1), size, bold, italic, False, True, hl=hl))
        elif kind == 'c':
            out.append(run(m.group(1).replace(STAR, '*'), bold, italic, sup, code=True, size=size, highlight=hl))
        pos = m.end()
    if pos < len(text):
        out.append(run(text[pos:].replace(STAR, '*'), bold, italic, sup, size=size, sub=sub, highlight=hl))
    return ''.join(out)

def ppr(style=None, align=None, keep_next=False, keep_lines=False, indent=None, space_before=None,
        space_after=None, line=None, no_ln=False, page_break_before=False):
    x = ''
    if style:
        x += '<w:pStyle w:val="%s"/>' % style
    if keep_next:
        x += '<w:keepNext/>'
    if keep_lines:
        x += '<w:keepLines/>'
    if page_break_before:
        x += '<w:pageBreakBefore/>'
    if no_ln:
        x += '<w:suppressLineNumbers/>'
    if space_before is not None or space_after is not None or line is not None:
        a = ''
        if space_before is not None:
            a += ' w:before="%d"' % space_before
        if space_after is not None:
            a += ' w:after="%d"' % space_after
        if line is not None:
            a += ' w:line="%d" w:lineRule="auto"' % line
        x += '<w:spacing%s/>' % a
    if indent:
        x += '<w:ind w:left="%d" w:hanging="%d"/>' % indent
    if align:
        x += '<w:jc w:val="%s"/>' % align
    return '<w:pPr>%s</w:pPr>' % x if x else ''

def para(text, size=None, bold=False, italic=False, **kw):
    return '<w:p>%s%s</w:p>' % (ppr(**kw), inline_runs(text, size=size, bold=bold, italic=italic))

PB = '\x00PAGEBREAK'

def page_break():
    return PB

def resolve_breaks(blocks):
    out, pending = [], False
    for b in blocks:
        if b == PB:
            pending = True
            continue
        if pending:
            if b.startswith('<w:p><w:pPr>'):
                # schema order inside pPr: pStyle, keepNext, keepLines, pageBreakBefore, ...
                m = re.match(r'<w:p><w:pPr>((?:<w:pStyle [^>]*/>)?(?:<w:keepNext/>)?(?:<w:keepLines/>)?)', b)
                if '<w:pageBreakBefore/>' not in b:
                    b = b[:m.end()] + '<w:pageBreakBefore/>' + b[m.end():]
            elif b.startswith('<w:p>'):
                b = '<w:p><w:pPr><w:pageBreakBefore/></w:pPr>' + b[len('<w:p>'):]
            else:
                out.append('<w:p><w:pPr><w:pageBreakBefore/><w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/><w:suppressLineNumbers/></w:pPr></w:p>')
            pending = False
        out.append(b)
    return out

# ---------------------------------------------------------------- tables

def split_row(line):
    s = line.strip()
    if s.startswith('|'):
        s = s[1:]
    if s.endswith('|') and not s.endswith('\\|'):
        s = s[:-1]
    return [c.strip().replace('\\|', '|') for c in re.split(r'(?<!\\)\|', s)]

SEP_RX = re.compile(r'\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?')

def visible_len(s):
    s = re.sub(r'\*\*|\^|`|\\', '', s)
    return len(s)

def table_xml(rows, aligns=None, size=18, total=9638, no_ln=True, krt=False):
    """rows: header first. aligns: per column 'left'/'right'. total: table width in twips."""
    ncol = max(len(r) for r in rows)
    rows = [r + [''] * (ncol - len(r)) for r in rows]
    aligns = (aligns or []) + ['left'] * ncol
    # column widths: every column gets at least its longest word (so no word is broken), the rest of
    # the width is shared in proportion to the (capped) length of the longest cell
    char = size * 10 * 0.56 + 6                      # approx. average Arial glyph width in twips
    mins, extra = [], []
    for c in range(ncol):
        cells = [rows[r][c] for r in range(len(rows))]
        longest = max(visible_len(x) for x in cells) if cells else 1
        word = max((visible_len(w) for x in cells for w in x.split()), default=1)
        word = min(word, 22)
        mins.append((word + 1.5) * char + 114)
        extra.append(max(min(longest, 45) - word, 0) + 1)
    if krt:
        mins, extra = [0, 0, 0], [44, 20, 36]
    if sum(mins) > total:
        mins = [m * total / sum(mins) for m in mins]
    rest = total - sum(mins)
    colw = [int(mins[c] + rest * extra[c] / sum(extra)) for c in range(ncol)]
    b = 'w:val="single" w:sz="4" w:space="0" w:color="000000"'
    x = ['<w:tbl><w:tblPr><w:tblW w:w="%d" w:type="dxa"/>'
         '<w:tblBorders><w:top %s/><w:left %s/><w:bottom %s/><w:right %s/>'
         '<w:insideH %s/><w:insideV %s/></w:tblBorders><w:tblLayout w:type="fixed"/>'
         '<w:tblCellMar><w:top w:w="15" w:type="dxa"/><w:left w:w="57" w:type="dxa"/><w:bottom w:w="15" w:type="dxa"/><w:right w:w="57" w:type="dxa"/></w:tblCellMar>'
         '<w:tblLook w:val="0000" w:firstRow="0" w:lastRow="0" w:firstColumn="0" w:lastColumn="0" w:noHBand="1" w:noVBand="1"/>'
         '</w:tblPr>' % ((sum(colw),) + (b,) * 6)]
    x.append('<w:tblGrid>' + ''.join('<w:gridCol w:w="%d"/>' % w for w in colw) + '</w:tblGrid>')
    cp = ppr(space_before=0, space_after=0, line=240, no_ln=no_ln)
    for ri, r in enumerate(rows):
        x.append('<w:tr>%s' % ('<w:trPr><w:cantSplit/><w:tblHeader/></w:trPr>' if ri == 0 else '<w:trPr><w:cantSplit/></w:trPr>'))
        # category row (KRT): bold first cell, all other cells empty -> one merged cell
        if ri > 0 and all(not c for c in r[1:]) and r[0].startswith('**') and r[0].endswith('**'):
            x.append('<w:tc><w:tcPr><w:tcW w:w="%d" w:type="dxa"/><w:gridSpan w:val="%d"/></w:tcPr>' % (sum(colw), ncol))
            x.append('<w:p>%s%s</w:p></w:tc></w:tr>' % (cp, inline_runs(r[0], size=size)))
            continue
        for ci, c in enumerate(r):
            al = aligns[ci] if ri > 0 else 'left'
            x.append('<w:tc><w:tcPr><w:tcW w:w="%d" w:type="dxa"/></w:tcPr>' % colw[ci])
            pp = ppr(space_before=0, space_after=0, line=240, no_ln=no_ln, align=('right' if al == 'right' else None))
            x.append('<w:p>%s%s</w:p></w:tc>' % (pp, inline_runs(c, size=size, bold=(ri == 0))))
        x.append('</w:tr>')
    x.append('</w:tbl>')
    x.append('<w:p>%s</w:p>' % ppr(space_before=0, space_after=120, line=240, no_ln=no_ln))
    return ''.join(x)

# ---------------------------------------------------------------- markdown -> body XML

def md_blocks(md, heading_map, table_size=18, para_kw=None, list_kw=None, krt_first_table=False, total=9638):
    para_kw = para_kw or {}
    md = re.sub(r'<!--.*?-->', '', md, flags=re.S)
    lines = md.split('\n')
    out, buf = [], []
    i = 0
    first_table = [krt_first_table]

    def flush():
        if buf:
            out.append(para(' '.join(x.strip() for x in buf), **para_kw))
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
            rows, aligns = [], None
            while i < len(lines) and lines[i].strip().startswith('|'):
                r = lines[i].strip()
                if SEP_RX.fullmatch(r):
                    aligns = ['right' if c.strip().endswith(':') else 'left' for c in split_row(r)]
                else:
                    rows.append(split_row(r))
                i += 1
            out.append(table_xml(rows, aligns, size=table_size, krt=first_table[0], total=total))
            first_table[0] = False
            continue
        lm = re.match(r'^(\s*)([-*]|\d+\.)\s+(.*)$', ln)
        if lm:
            flush()
            depth = len(lm.group(1).expandtabs(2)) // 2
            bullet = '•' if lm.group(2) in ('-', '*') else lm.group(2)
            left = 425 + 360 * depth
            kw = dict(para_kw)
            kw.update(list_kw or {})
            out.append(para(bullet + '\t' + lm.group(3), indent=(left, 425), **kw))
            i += 1; continue
        buf.append(ln)
        i += 1
    flush()
    return out

def png_size(path):
    with open(path, 'rb') as f:
        head = f.read(24)
    if head[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('not a PNG: ' + path)
    return struct.unpack('>II', head[16:24])

def image_block(rid, idx, path, name, width_cm=IMG_W_CM):
    w, h = png_size(path)
    cx = int(width_cm * EMU_PER_CM)
    cy = int(round(cx * h / w))
    return ('<w:p>%s<w:r><w:drawing>'
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
            % (ppr(space_before=0, space_after=120, line=240, align='center', no_ln=True, keep_next=True),
               cx, cy, idx, esc(name), idx, esc(name), rid, cx, cy))

# ---------------------------------------------------------------- package parts

NS = ('xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
      'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
      'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
      'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
      'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"')

def styles_xml(body_pt=12, line=480, after=0):
    sz = body_pt * 2
    f = FONT
    def ps(sid, name, rpr, pp, based='Normal', nxt='Normal', outline=None):
        ol = '<w:outlineLvl w:val="%d"/>' % outline if outline is not None else ''
        return ('<w:style w:type="paragraph" w:styleId="%s"><w:name w:val="%s"/><w:basedOn w:val="%s"/>'
                '<w:next w:val="%s"/><w:qFormat/><w:pPr>%s%s</w:pPr><w:rPr>%s</w:rPr></w:style>'
                % (sid, name, based, nxt, pp, ol, rpr))
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="%s" w:hAnsi="%s" w:eastAsia="%s" w:cs="%s"/>'
            '<w:sz w:val="%d"/><w:szCs w:val="%d"/><w:lang w:val="en-US" w:eastAsia="zh-CN"/></w:rPr></w:rPrDefault>'
            '<w:pPrDefault><w:pPr><w:spacing w:before="0" w:after="%d" w:line="%d" w:lineRule="auto"/></w:pPr></w:pPrDefault>'
            '</w:docDefaults>'
            '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/>'
            '<w:pPr><w:spacing w:before="0" w:after="%d" w:line="%d" w:lineRule="auto"/><w:jc w:val="left"/></w:pPr>'
            '<w:rPr><w:rFonts w:ascii="%s" w:hAnsi="%s" w:eastAsia="%s" w:cs="%s"/><w:sz w:val="%d"/><w:szCs w:val="%d"/></w:rPr></w:style>'
            % (f, f, f, f, sz, sz, after, line, after, line, f, f, f, f, sz, sz)
            + ps('Title', 'Title', '<w:b/><w:bCs/><w:sz w:val="32"/><w:szCs w:val="32"/>',
                 '<w:spacing w:before="0" w:after="240"/>')
            + ps('Heading1', 'heading 1', '<w:b/><w:bCs/><w:sz w:val="28"/><w:szCs w:val="28"/>',
                 '<w:keepNext/><w:spacing w:before="240" w:after="0"/>', outline=0)
            + ps('Heading2', 'heading 2', '<w:b/><w:bCs/><w:sz w:val="%d"/><w:szCs w:val="%d"/>' % (sz, sz),
                 '<w:keepNext/><w:spacing w:before="120" w:after="0"/>', outline=1)
            + ps('Heading3', 'heading 3', '<w:b/><w:bCs/><w:i/><w:iCs/><w:sz w:val="%d"/><w:szCs w:val="%d"/>' % (sz, sz),
                 '<w:keepNext/><w:spacing w:before="120" w:after="0"/>', outline=2)
            + '<w:style w:type="paragraph" w:styleId="Footer"><w:name w:val="footer"/><w:basedOn w:val="Normal"/>'
              '<w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>'
              '<w:rPr><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr></w:style>'
            + '<w:style w:type="character" w:default="1" w:styleId="DefaultParagraphFont"><w:name w:val="Default Paragraph Font"/><w:uiPriority w:val="1"/><w:semiHidden/></w:style>'
            + '<w:style w:type="character" w:styleId="LineNumber"><w:name w:val="line number"/><w:basedOn w:val="DefaultParagraphFont"/>'
              '<w:rPr><w:rFonts w:ascii="%s" w:hAnsi="%s"/><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr></w:style>' % (f, f)
            + '<w:style w:type="table" w:default="1" w:styleId="TableNormal"><w:name w:val="Normal Table"/><w:uiPriority w:val="99"/><w:semiHidden/>'
              '<w:tblPr><w:tblInd w:w="0" w:type="dxa"/><w:tblCellMar><w:top w:w="0" w:type="dxa"/><w:left w:w="108" w:type="dxa"/>'
              '<w:bottom w:w="0" w:type="dxa"/><w:right w:w="108" w:type="dxa"/></w:tblCellMar></w:tblPr></w:style>'
            + '</w:styles>')

SETTINGS = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:defaultTabStop w:val="425"/><w:characterSpacingControl w:val="doNotCompress"/>'
            '<w:compat><w:compatSetting w:name="compatibilityMode" w:uri="http://schemas.microsoft.com/office/word" w:val="15"/></w:compat>'
            '</w:settings>')

FOOTER = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
          '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
          'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
          '<w:p><w:pPr><w:pStyle w:val="Footer"/><w:jc w:val="center"/></w:pPr>'
          '<w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
          '<w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>1</w:t></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r>'
          '</w:p></w:ftr>')

def write_docx(path, blocks, images, title, line_numbers=True, body_pt=12, line=480, after=0,
               margins=(1440, 1134, 1440, 1134)):
    top, right, bottom, left = margins
    ln = '<w:lnNumType w:countBy="1" w:distance="284" w:restart="continuous"/>' if line_numbers else ''
    sect = ('<w:sectPr><w:footerReference w:type="default" r:id="rIdFooter1"/>'
            '<w:pgSz w:w="11906" w:h="16838"/>'
            '<w:pgMar w:top="%d" w:right="%d" w:bottom="%d" w:left="%d" w:header="709" w:footer="567" w:gutter="0"/>'
            '%s<w:pgNumType w:start="1"/><w:cols w:space="425"/></w:sectPr>' % (top, right, bottom, left, ln))
    doc = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<w:document %s><w:body>%s%s</w:body></w:document>'
           % (NS, ''.join(resolve_breaks(blocks)), sect))
    rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
            '<Relationship Id="rIdSettings" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>'
            '<Relationship Id="rIdFooter1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>']
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
              '<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>'
              '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
              '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
              '</Types>')
    root_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                 '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                 '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
                 '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
                 '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
                 '</Relationships>')
    now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    core = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            '<dc:title>%s</dc:title><dc:creator></dc:creator>'
            '<dcterms:created xsi:type="dcterms:W3CDTF">%s</dcterms:created>'
            '<dcterms:modified xsi:type="dcterms:W3CDTF">%s</dcterms:modified></cp:coreProperties>' % (esc(title), now, now))
    app = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
           '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
           '<Application>Microsoft Office Word</Application></Properties>')
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', ctypes)
        z.writestr('_rels/.rels', root_rels)
        z.writestr('docProps/core.xml', core)
        z.writestr('docProps/app.xml', app)
        z.writestr('word/document.xml', doc)
        z.writestr('word/styles.xml', styles_xml(body_pt, line, after))
        z.writestr('word/settings.xml', SETTINGS)
        z.writestr('word/footer1.xml', FOOTER)
        z.writestr('word/_rels/document.xml.rels', ''.join(rels))
        for rid, name, src in images:
            z.write(src, 'word/media/' + name, compress_type=zipfile.ZIP_STORED)

# ---------------------------------------------------------------- markdown splitting

def sections(md, level=2):
    md = re.sub(r'<!--.*?-->', '', md, flags=re.S)
    parts = re.split(r'(?m)^(?=%s )' % ('#' * level), md)
    res = []
    for p in parts:
        m = re.match(r'#{%d} (.*)' % level, p)
        res.append((m.group(1).strip() if m else '', p))
    return res

def strip_heading(text):
    return re.sub(r'^#{1,6} .*\n', '', text, count=1)

def figure_chunks(text):
    """'**Figure N. title**' chunks -> list of (id, title, legend markdown without the title)."""
    out = []
    for c in re.split(r'(?m)^(?=\*\*Figure S?\d+\.)', text):
        m = re.match(r'\*\*Figure (S?\d+)\.(.*?)\*\*[ \t]*\n', c)
        if m:
            out.append((m.group(1), 'Figure %s.%s' % (m.group(1), m.group(2)), c[m.end():].strip()))
    return out

# ---------------------------------------------------------------- 01 manuscript

H_MAIN = {1: 'Title', 2: 'Heading1', 3: 'Heading2', 4: 'Heading3'}
H_STAR = {1: 'Heading1', 2: 'Heading2', 3: 'Heading3', 4: 'Heading3'}

def build_main():
    md = open(MAIN_MD, encoding='utf-8').read()
    star = open(STAR_MD, encoding='utf-8').read()
    secs = sections(md)
    head = dict(secs)
    title_page = secs[0][1]
    title = re.match(r'# (.*)', title_page).group(1).strip()
    blocks = []
    # title page
    tp = re.sub(r'^# .*\n', '', title_page, count=1)
    blocks.append(para(title, style='Title', line=360))
    for p in [x.strip() for x in re.split(r'\n\s*\n', tp) if x.strip()]:
        blocks.append(para(' '.join(l.strip() for l in p.split('\n'))))
    order_counted = []
    skip = {'Highlights', 'eTOC blurb'}          # delivered in 07_Highlights_eTOC.docx
    figs = []
    for name, text in secs[1:]:
        if name in skip:
            continue
        if name == 'Summary':
            blocks.append(page_break())
        if name in ('Figure titles and legends', 'Tables', 'STAR Methods'):
            blocks.append(page_break())
        if name == 'Figure titles and legends':
            blocks.append(para(name, style='Heading1', keep_next=True))
            pre = text.split('**Figure 1.', 1)[0]
            for fid, ftitle, legend in figure_chunks(text):
                figs.append((fid, ftitle))
                blocks.append(para('**%s**' % ftitle, keep_next=True, space_before=240))
                blocks.extend(md_blocks(legend, H_MAIN))
            continue
        if name == 'Tables':
            blocks.append(para(name, style='Heading1', keep_next=True))
            body = strip_heading(text)
            blocks.extend(md_blocks(body, H_MAIN, table_size=16))
            continue
        if name == 'STAR Methods':
            assert '[[ASSEMBLY:' in text, 'STAR Methods assembly marker missing'
            blocks.append(para('STAR★Methods', style='Heading1', keep_next=True))
            sbody = re.sub(r'<!--.*?-->', '', star, flags=re.S)
            sbody = re.sub(r'(?m)^# STAR★METHODS\s*\n', '', sbody, count=1)
            blocks.extend(md_blocks(sbody, H_STAR, table_size=16, krt_first_table=True))
            continue
        if name == 'References':
            blocks.append(para(name, style='Heading1', keep_next=True))
            blocks.extend(md_blocks(strip_heading(text), H_MAIN, list_kw={}))
            continue
        blocks.extend(md_blocks(text, H_MAIN))
    # figures, one per page, for review convenience
    images = []
    for n, (fid, ftitle) in enumerate(figs, 1):
        src = os.path.join(IMG, 'Figure%s.png' % fid)
        rid, nm = 'rIdImg%d' % n, 'image%d.png' % n
        images.append((rid, nm, src))
        blocks.append(page_break())
        blocks.append(para('**%s**' % ftitle, no_ln=True, line=240, space_after=120, keep_next=True))
        blocks.append(image_block(rid, n, src, 'Figure %s' % fid))
    write_docx(OUT_MAIN, blocks, images, title)
    return title, len(images)

# ---------------------------------------------------------------- 02 Document S1

def build_si(title):
    md = open(SI_MD, encoding='utf-8').read()
    md = re.sub(r'<!--.*?-->', '', md, flags=re.S)
    doc_s1 = md.split('\n# Document S1\n', 1)[1].split('\n# Excel tables', 1)[0]
    index = md.split('## Index of supplemental items', 1)[1].split('\n---', 1)[0]
    s1_line = re.search(r'\*\*(Document S1\.[^*]*)\*\*', index).group(1)
    small = dict(size=20)
    blocks = [para('Supplemental information', style='Title', line=276),
              para('**%s**' % title, line=276, space_after=200),
              para('[[PENDING: author names]]', line=276, space_after=200),
              para(s1_line, line=276, space_after=200)]
    secs = sections(doc_s1, 2)
    images = []
    k = 0
    contents = []
    for name, text in secs:
        if not name:
            continue
        body = strip_heading(text)
        if name == 'Supplemental figures':
            for fid, ftitle, legend in figure_chunks(body):
                k += 1
                contents.append(ftitle)
                src = os.path.join(IMG, 'Figure%s.png' % fid)
                rid, nm = 'rIdImg%d' % k, 'image%d.png' % k
                images.append((rid, nm, src))
                blocks.append(page_break())
                blocks.append(image_block(rid, k, src, 'Figure %s' % fid))
                blocks.append(para('**%s**' % ftitle, keep_next=True, space_after=80, **small))
                blocks.extend(md_blocks(legend, H_STAR, para_kw=dict(space_after=80, **small)))
        elif name == 'Supplemental tables':
            for c in re.split(r'(?m)^(?=\*\*Table S\d+\.)', body):
                m = re.match(r'\*\*(Table S\d+\..*?)\*\*[ \t]*\n', c)
                if not m:
                    continue
                contents.append(m.group(1))
                blocks.append(page_break())
                blocks.append(para('**%s**' % m.group(1), keep_next=True, space_after=80, **small))
                blocks.extend(md_blocks(c[m.end():], H_STAR, table_size=15, para_kw=dict(space_after=80, **small)))
        elif name == 'Supplemental notes':
            blocks.append(page_break())
            for c in re.split(r'(?m)^(?=\*\*Note S\d+\.)', body):
                m = re.match(r'\*\*(Note S\d+\..*?)\*\*[ \t]*\n', c)
                if not m:
                    continue
                contents.append(m.group(1))
                blocks.append(para('**%s**' % m.group(1), keep_next=True, space_before=200, space_after=80, **small))
                blocks.extend(md_blocks(c[m.end():], H_STAR, para_kw=dict(space_after=80, **small)))
    # contents list on the title page
    blocks[4:4] = [para('**Contents**', line=276, space_after=60, **small)] + \
                  [para(c, line=276, space_after=0, **small) for c in contents]
    write_docx(OUT_SI, blocks, images, 'Supplemental information - ' + title, line_numbers=False,
               body_pt=10, line=276, after=80)
    return len(images), contents

# ---------------------------------------------------------------- 06 cover letter

COVER = [
    '[[PENDING: submission date]]',
    'The Editors, *Cell Systems*',
    'Dear Editors,',
    'We submit our manuscript "{title}" for consideration as an Article in *Cell Systems*.',
    'Multi-omic models are trained on paired molecular layers, yet a laboratory or clinic usually measures one layer at deployment. '
    'Anyone designing such a study has to decide whether a second layer is worth measuring, how much of it a model trained on paired '
    'data can supply without a new assay, and how many additional molecules close the rest of the gap. We answer these questions '
    'on paired data with a functional endpoint, using one kernel ridge family throughout so that arms differ only in the information they receive.',
    'Using 421 cancer cell lines with paired metabolome, transcriptome, proteome and drug response, we find that the functional value '
    'of a second layer is concentrated in a few directions. Measured RNA nearly doubles the predictive skill of a metabolite-only model, '
    'and four function-weighted directions of the measured layer carry most of that gain; a pre-registered test on a third endpoint, GDSC1, '
    'confirmed this, with random and label-permuted directions carrying little of it. This structure explains what a single layer '
    'can recover. A free cross-fitted virtual layer recovers about one third on development folds, whatever its reconstruction accuracy, '
    'and neither re-representing it nor building it on a single-cell foundation model raised that share. Measuring 8–64 selected '
    'proteins or genes recovers 37–78% of the room, random panels of the same size capture most of this, and tailoring panels per '
    'sample or per context added nothing. We release the ranked panels and retrained heads as a resource. The share and the panel law '
    'replicate on CRISPR gene effects.',
    'Every claim the design could test was then evaluated once, under a protocol registered beforehand, on 123 cell lines that had '
    'remained sealed throughout development. Six of eight pre-registered predictions held, and every gain kept its direction. The two '
    'misses were ratio criteria that fell just short: the sealed lines had more room for a second layer, so the recovered proteome share '
    'was 0.137 against a band starting at 0.15, and protein-panel recovery was 0.490 and 0.489 against 0.5. Both are reported in Table 1.',
    'The work shows how functional information is distributed across molecular layers and how many measurements recover it, a '
    'systems-level question within the scope of *Cell Systems*. It gives design rules to groups that plan multi-omic studies, and it '
    'argues that virtual-omics models, now judged mostly by reconstruction accuracy, should be judged by the functional value they '
    'deliver. Registered plans, checksums and run outputs accompany the results. Code is supplied as Methods S1; the repository at '
    'https://github.com/dawnmengsjtu/CREED is publicly accessible.',
    'This manuscript has not been published and is not under consideration elsewhere, and all authors approved its submission. '
    'Preprint: [[PENDING: preprint server and DOI, or "none"]]. Suggested reviewers: [[PENDING: names, affiliations, e-mails]]. '
    'Reviewers to exclude: [[PENDING: names and reasons, or "none"]]. Competing interests: [[PENDING: as in the Declaration of interests]].',
    'Sincerely,',
    '[[PENDING: corresponding author name, affiliation and e-mail]], on behalf of all authors',
]

def build_cover(title):
    blocks = []
    for k, p in enumerate(COVER):
        p = p.replace('{title}', title)
        blocks.append(para(p, align='both' if 3 <= k <= 8 else None, space_after=110))
    write_docx(OUT_COVER, blocks, [], 'Cover letter', line_numbers=False, body_pt=10, line=252, after=110,
               margins=(1134, 1134, 1134, 1134))
    return len(' '.join(COVER).split())

# ---------------------------------------------------------------- 07 highlights / eTOC

def build_highlights(title):
    md = open(MAIN_MD, encoding='utf-8').read()
    secs = dict(sections(md))
    blocks = [para('Title', style='Heading1', line=276), para(title, line=276, space_after=160)]
    out = {}
    for name in ('Highlights', 'eTOC blurb', 'Summary', 'Keywords'):
        body = strip_heading(secs[name]).strip()
        out[name] = body
        blocks.append(para(name if name != 'eTOC blurb' else 'eTOC blurb (In brief)', style='Heading1', line=276))
        blocks.extend(md_blocks(body, H_STAR, para_kw=dict(line=276, space_after=120)))
    write_docx(OUT_HL, blocks, [], 'Highlights and eTOC blurb', line_numbers=False, body_pt=12, line=276, after=120)
    return out

# ---------------------------------------------------------------- checks

FORBIDDEN = ['/Users/', '/public/', 'creed-research', '[src:', '<!--', 'scratchpad', 'manuscript_v0',
             'figures_v', 'handoff_2026', 'ASSEMBLY', '-->']

def docx_text(path):
    with zipfile.ZipFile(path) as z:
        x = z.read('word/document.xml').decode('utf-8')
    x = x.replace('</w:p>', '\n').replace('<w:tab/>', '\t')
    x = re.sub(r'<[^>]+>', '', x)
    return x.replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"').replace('&amp;', '&')

def check(path):
    import xml.dom.minidom
    with zipfile.ZipFile(path) as z:
        bad_zip = z.testzip()
        for nm in z.namelist():
            if nm.endswith('.xml') or nm.endswith('.rels'):
                xml.dom.minidom.parseString(z.read(nm))
    t = docx_text(path)
    bad = [s for s in FORBIDDEN if s in t]
    leftover = re.findall(r'\*\*|(?<!\w)\^\S|\\\*|\w\^[{(\d−-]|_\{|~[^~\s]{1,40}~|\w\^\w|==', t)
    print('%-48s zip=%s xml=ok forbidden=%s md-leftovers=%d placeholders=%d' % (
        os.path.basename(path), 'ok' if bad_zip is None else bad_zip, bad or 'none', len(leftover),
        len(re.findall(r'\[\[[A-Z]+', t))))
    return not bad and bad_zip is None

if __name__ == '__main__':
    title, n1 = build_main()
    n2, contents = build_si(title)
    w = build_cover(title)
    build_highlights(title)
    print('manuscript figures:', n1, '| Document S1 figures:', n2, '| cover letter words:', w)
    ok = all(check(p) for p in (OUT_MAIN, OUT_SI, OUT_COVER, OUT_HL))
    if not ok:
        sys.exit('check failed')

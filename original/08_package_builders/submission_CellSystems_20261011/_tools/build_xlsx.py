#!/usr/bin/env python3
"""Stdlib-only OOXML (.xlsx) writer for the CREED Excel tables (Tables S3, S4, S8, S11, S12, S13, S14).

Source: the "Excel tables" part of manuscript_cellsystems/supplemental_information.md, whose table
bodies are verbatim copies of the source tables (TABLE_RENUMBERING.md, data_files_plan.md).

One table per file (Cell Press rule). Sheet "Legend" = title + legend + every text block of the table
in order; then one sheet per Markdown table (header row bold, no formulas, no shading, no colour).
Numeric cells are stored as numbers with a number format that shows exactly the source digits
(e.g. "0.940" -> 0.94 with format 0.000); every other cell is kept as text. Bracketed interval cells of
Tables S13/S14 Part A and the per-fold share cell of Table S3 are split into columns only when every
cell matches a strict pattern (regex full match, so each part is an exact substring of the source).
"""
import os, re, sys, zipfile, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
SI_MD = '/Users/xlzhexue/Desktop/proj/creed/manuscript_cellsystems/supplemental_information.md'
OUTDIR = os.path.join(PKG, '03_Tables_S')

def esc(s):
    return (s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;'))

def plain(s):
    """Markdown inline -> plain text (no number is touched)."""
    s = s.replace('\\*', '')
    s = re.sub(r'\*\*(.+?)\*\*', r'\1', s)
    s = re.sub(r'(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])', r'\1', s)
    s = re.sub(r'`([^`]+)`', r'\1', s)
    s = re.sub(r'\^([^\^\s]{1,40})\^', r'\1', s)
    return s.replace('', '*').replace('\\|', '|')

def split_row(line):
    s = line.strip()
    if s.startswith('|'):
        s = s[1:]
    if s.endswith('|'):
        s = s[:-1]
    return [c.strip() for c in re.split(r'(?<!\\)\|', s)]

SEP_RX = re.compile(r'\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?')
NUM_RX = re.compile(r'^([−-])?(\d+)(\.\d+)?$')
NUMC_RX = re.compile(r'^([−-])?(\d{1,3}(?:,\d{3})+)(\.\d+)?$')

def as_number(s):
    """Return (value_text, format_code) if s is a plain number, else None."""
    m = NUM_RX.match(s)
    comma = False
    if not m:
        m = NUMC_RX.match(s)
        comma = True
    if not m:
        return None
    sign, ip, fp = m.group(1), m.group(2), m.group(3) or ''
    digits = ip.replace(',', '')
    if len(digits) > 1 and digits.startswith('0') and not comma:
        return None                       # identifiers such as 039839 stay text
    if len(digits) + len(fp) > 15:
        return None
    if sign and not any(ch in '123456789' for ch in digits + fp):
        return None                       # keep a signed zero such as -0.00 as text (its sign carries meaning)
    val = ('-' if sign else '') + digits + fp
    dec = len(fp) - 1 if fp else 0
    base = '#,##0' if comma else '0'
    fmt = base + ('.' + '0' * dec if dec else '')
    return val, fmt

# ---------------------------------------------------------------- split rules

SPLITS = {
    '95% bootstrap interval of sRMSE': (re.compile(r'\[(\S+), (\S+)\]'),
                                        ['sRMSE, 2.5th percentile', 'sRMSE, 97.5th percentile']),
    '95% interval of the gain; P(> 0)': (re.compile(r'\[(\S+), (\S+)\] P\(>0\)=(\S+)'),
                                         ['Gain, 2.5th percentile', 'Gain, 97.5th percentile', 'P(gain > 0)']),
    'Sign test vs M (wins/losses/ties, two-sided p)': (re.compile(r'(\d+)/(\d+)/(\d+), p=(\S+)'),
                                                       ['Sign test wins', 'Sign test losses', 'Sign test ties', 'Sign test p (two-sided)']),
    'Per-fold shares': (re.compile(r'(\d+), (\d+), (\d+), (\d+), (\d+)'),
                        ['Per-fold share F0 (%)', 'Per-fold share F1 (%)', 'Per-fold share F2 (%)', 'Per-fold share F3 (%)', 'Per-fold share F4 (%)']),
}
EMPTY = {'—', '', '-'}

def apply_splits(rows, log):
    head = rows[0]
    for col_name, (rx, new_heads) in SPLITS.items():
        if col_name not in head:
            continue
        ci = head.index(col_name)
        body = [r[ci] for r in rows[1:]]
        if not all(c in EMPTY or rx.fullmatch(c) for c in body):
            log.append('split skipped (non-matching cell): ' + col_name)
            continue
        new = [head[:ci] + new_heads + head[ci + 1:]]
        for r in rows[1:]:
            c = r[ci]
            parts = list(rx.fullmatch(c).groups()) if c not in EMPTY else [c] * len(new_heads)
            new.append(r[:ci] + parts + r[ci + 1:])
        rows = new
        head = rows[0]
        log.append('split: %s -> %d columns' % (col_name, len(new_heads)))
    return rows

# ---------------------------------------------------------------- xlsx writer

def col_letter(i):
    s = ''
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s

class Book:
    def __init__(self):
        self.sheets = []       # (name, rows[list of list of (kind,value,style)], widths)
        self.fmts = {}         # format code -> numFmtId

    def fmt_id(self, code):
        if code == '0':
            return 1
        if code not in self.fmts:
            self.fmts[code] = 164 + len(self.fmts)
        return self.fmts[code]

    def add_sheet(self, name, rows, header=True, wrap_col0=False):
        name = re.sub(r'[\[\]:*?/\\]', '', name)[:31]
        base, k = name, 2
        while any(n == name for n, _, _ in self.sheets):
            name = (base[:28] + ' %d' % k); k += 1
        self.sheets.append((name, (rows, header, wrap_col0), None))
        return name

    def styles(self):
        # cellXfs: 0 normal text, 1 bold header (wrap), 2 wrapped text; then one per number format
        self._xf = {}
        xfs = ['<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>',
               '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1" applyAlignment="1"><alignment wrapText="1" vertical="top"/></xf>',
               '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment wrapText="1" vertical="top"/></xf>']
        codes = ['0'] + list(self.fmts)
        for c in codes:
            self._xf[c] = len(xfs)
            xfs.append('<xf numFmtId="%d" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>' % self.fmt_id(c))
        nf = ''.join('<numFmt numFmtId="%d" formatCode="%s"/>' % (i, esc(c)) for c, i in self.fmts.items())
        return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                + ('<numFmts count="%d">%s</numFmts>' % (len(self.fmts), nf) if self.fmts else '')
                + '<fonts count="2"><font><sz val="11"/><name val="Arial"/><family val="2"/></font>'
                  '<font><b/><sz val="11"/><name val="Arial"/><family val="2"/></font></fonts>'
                  '<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>'
                  '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
                  '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
                  '<cellXfs count="%d">%s</cellXfs>'
                  '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
                  '</styleSheet>' % (len(xfs), ''.join(xfs)))

    def sheet_xml(self, rows, header, wrap_col0):
        ncol = max((len(r) for r in rows), default=1)
        widths = []
        for c in range(ncol):
            lens = [len(r[c]) for r in rows if c < len(r)]
            w = min(max(lens + [6]) + 2, 60)
            if wrap_col0 and c == 0:
                w = 120
            widths.append(w)
        out = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
               '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
               'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">']
        if header:
            out.append('<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>')
        out.append('<sheetFormatPr defaultRowHeight="15"/><cols>' + ''.join(
            '<col min="%d" max="%d" width="%d" customWidth="1"/>' % (i + 1, i + 1, w) for i, w in enumerate(widths)) + '</cols><sheetData>')
        for ri, r in enumerate(rows):
            cells = []
            for ci, v in enumerate(r):
                ref = '%s%d' % (col_letter(ci), ri + 1)
                if v == '':
                    continue
                num = None if (header and ri == 0) else as_number(v)
                if num:
                    val, code = num
                    cells.append('<c r="%s" s="%d"><v>%s</v></c>' % (ref, self._xf[code], val))
                else:
                    st = 1 if (header and ri == 0) or (not header and ri == 0 and wrap_col0) else (2 if wrap_col0 else 0)
                    cells.append('<c r="%s" s="%d" t="inlineStr"><is><t xml:space="preserve">%s</t></is></c>' % (ref, st, esc(v)))
            out.append('<row r="%d">%s</row>' % (ri + 1, ''.join(cells)))
        out.append('</sheetData><pageMargins left="0.7" right="0.7" top="0.75" bottom="0.75" header="0.3" footer="0.3"/></worksheet>')
        return ''.join(out)

    def save(self, path, title):
        # register formats first
        for _, (rows, header, _w), _ in self.sheets:
            for ri, r in enumerate(rows):
                if header and ri == 0:
                    continue
                for v in r:
                    n = as_number(v)
                    if n:
                        self.fmt_id(n[1])
        styles = self.styles()
        n = len(self.sheets)
        wb = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
              '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
              'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
              + ''.join('<sheet name="%s" sheetId="%d" r:id="rId%d"/>' % (esc(nm), i + 1, i + 1)
                        for i, (nm, _, _) in enumerate(self.sheets))
              + '</sheets></workbook>')
        wbrels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                  '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                  + ''.join('<Relationship Id="rId%d" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet%d.xml"/>' % (i + 1, i + 1) for i in range(n))
                  + '<Relationship Id="rId%d" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>' % (n + 1)
                  + '</Relationships>')
        ct = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
              '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
              '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
              '<Default Extension="xml" ContentType="application/xml"/>'
              '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
              '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
              + ''.join('<Override PartName="/xl/worksheets/sheet%d.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' % (i + 1) for i in range(n))
              + '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
                '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
                '</Types>')
        rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
                '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
                '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
                '</Relationships>')
        now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        core = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
                'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
                'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>%s</dc:title>'
                '<dcterms:created xsi:type="dcterms:W3CDTF">%s</dcterms:created></cp:coreProperties>' % (esc(title), now))
        app = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
               '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"><Application>Microsoft Excel</Application></Properties>')
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('[Content_Types].xml', ct)
            z.writestr('_rels/.rels', rels)
            z.writestr('docProps/core.xml', core)
            z.writestr('docProps/app.xml', app)
            z.writestr('xl/workbook.xml', wb)
            z.writestr('xl/_rels/workbook.xml.rels', wbrels)
            z.writestr('xl/styles.xml', styles)
            for i, (nm, (rows, header, wrap0), _) in enumerate(self.sheets):
                z.writestr('xl/worksheets/sheet%d.xml' % (i + 1), self.sheet_xml(rows, header, wrap0))

# ---------------------------------------------------------------- parse the Excel-table part

LABEL_RX = re.compile(r'^\*{1,2}(Part [A-F]\b[^*]*|Slopes[^*]*|Pre-registered criterion (L\d)[^*]*|Limits of this table[^*]*)\*{1,2}')

def sheet_label(lab):
    m = re.match(r'Part ([A-F])', lab)
    if m:
        return 'Part ' + m.group(1)
    m = re.match(r'Pre-registered criterion (L\d)', lab)
    if m:
        return 'Criterion ' + m.group(1)
    if lab.startswith('Slopes'):
        return 'Slopes'
    if lab.startswith('Limits'):
        return 'Limits'
    return lab[:31]

def parse_blocks(text):
    """-> list of ('p', text) | ('t', rows) | ('li', text)"""
    lines = text.split('\n')
    out, buf, i = [], [], 0
    def flush():
        if buf:
            out.append(('p', ' '.join(x.strip() for x in buf)))
            buf.clear()
    while i < len(lines):
        s = lines[i].strip()
        if not s or re.fullmatch(r'-{3,}', s):
            flush(); i += 1; continue
        if s.startswith('|'):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                if not SEP_RX.fullmatch(lines[i].strip()):
                    rows.append(split_row(lines[i]))
                i += 1
            out.append(('t', rows)); continue
        if re.match(r'^[-*]\s+', s) and not s.startswith('**') and not re.match(r'^\*[^ ]', s):
            flush(); out.append(('li', re.sub(r'^[-*]\s+', '', s))); i += 1; continue
        buf.append(s); i += 1
    flush()
    return out

def build():
    md = open(SI_MD, encoding='utf-8').read()
    md = re.sub(r'<!--.*?-->', '', md, flags=re.S)
    part = md.split('\n# Excel tables', 1)[1]
    chunks = re.split(r'(?m)^(?=\*\*Table S\d+\.)', part)
    os.makedirs(OUTDIR, exist_ok=True)
    report = []
    for c in chunks:
        m = re.match(r'\*\*(Table (S\d+)\..*?)\*\*\s*\(delivered as a separate Excel file; reproduced here for review\)[ \t]*\n', c)
        if not m:
            continue
        title, tid = m.group(1), m.group(2)
        blocks = parse_blocks(c[m.end():])
        book = Book()
        legend = [[title]]
        log = []
        segs = []                  # (label, [blocks])
        cur_label, cur = None, []
        for kind, val in blocks:
            if kind == 'p' and LABEL_RX.match(val):
                segs.append((cur_label, cur))
                cur_label, cur = LABEL_RX.match(val).group(1).strip().rstrip('.'), [(kind, val)]
            else:
                cur.append((kind, val))
        segs.append((cur_label, cur))
        sheet_plan = []
        ntab = 0
        for label, bl in segs:
            has_table = any(k == 't' for k, _ in bl)
            if label is None:
                # legend paragraphs before the first labelled segment
                for k, v in bl:
                    if k == 't':
                        ntab += 1
                        sheet_plan.append(('Table', None, v, []))
                    else:
                        legend.append([plain(v)])
                continue
            name = sheet_label(label)
            texts = [plain(v) for k, v in bl if k != 't']
            tables = [v for k, v in bl if k == 't']
            if not tables:
                sheet_plan.append((name, texts, None, []))
            else:
                for j, t in enumerate(tables):
                    sheet_plan.append((name if len(tables) == 1 else '%s (%d)' % (name, j + 1), texts if j == 0 else [], t, []))
        # sheets
        legend.append([''])
        legend.append(['Sheets in this file:'])
        names = []
        for name, texts, rows, _ in sheet_plan:
            names.append(name)
        for nm in names:
            legend.append([' - ' + nm])
        book.add_sheet('Legend', legend, header=False, wrap_col0=True)
        for name, texts, rows, _ in sheet_plan:
            if rows is None:
                book.add_sheet(name, [[t] for t in texts], header=False, wrap_col0=True)
                continue
            rows = [[plain(x) for x in r] for r in rows]
            rows = apply_splits(rows, log)
            sheet_rows = rows
            if texts:
                # caption/notes of a labelled table go below the table, after one empty row
                sheet_rows = rows + [['']] + [[t] for t in texts]
            book.add_sheet(name, sheet_rows, header=True)
        fn = os.path.join(OUTDIR, 'Table_%s.xlsx' % tid)
        book.save(fn, title)
        report.append((os.path.basename(fn), title, len(book.sheets), log))
    return report

if __name__ == '__main__':
    import xml.dom.minidom
    rep = build()
    for fn, title, ns, log in rep:
        p = os.path.join(OUTDIR, fn)
        with zipfile.ZipFile(p) as z:
            assert z.testzip() is None
            for n in z.namelist():
                if n.endswith('.xml') or n.endswith('.rels'):
                    xml.dom.minidom.parseString(z.read(n))
        print('%-16s sheets=%d  %s  %s' % (fn, ns, title[:70], '; '.join(log)))

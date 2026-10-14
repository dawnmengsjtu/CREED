"""Build 02_Supplemental_Information_DocumentS1.pdf from the same master as the Document S1 .docx.

Reads manuscript_cellsystems/supplemental_information.md (Document S1 part) with the section helpers of
build_cell_docx.py, writes a paginated HTML (title page and contents; each supplemental figure on its own page
with title and legend; each table from a new page; notes), and prints it to A4 PDF with WebKit (stdlib + macOS
swiftc). Usage: python3 -I -B build_si_pdf.py
"""
import html
import importlib.util
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('build_cell_docx', os.path.join(HERE, 'build_cell_docx.py'))
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)
PKG = os.environ.get('CREED_SUBMISSION_DIR', os.path.dirname(HERE))
OUT = os.path.join(PKG, '02_Supplemental_Information_DocumentS1.pdf')

SWIFT = r'''
import Cocoa
import WebKit
let a = CommandLine.arguments
let inURL = URL(fileURLWithPath: a[1]); let outURL = URL(fileURLWithPath: a[2])
class D: NSObject, WKNavigationDelegate {
  var window: NSWindow?
  func webView(_ w: WKWebView, didFinish n: WKNavigation!) {
    DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
      let pi = NSPrintInfo()
      pi.paperSize = NSSize(width: 595.3, height: 841.9)
      pi.topMargin = 48; pi.bottomMargin = 48; pi.leftMargin = 45; pi.rightMargin = 45
      pi.horizontalPagination = .fit; pi.verticalPagination = .automatic
      pi.jobDisposition = .save
      pi.dictionary()[NSPrintInfo.AttributeKey.jobSavingURL] = outURL
      let op = w.printOperation(with: pi)
      op.showsPrintPanel = false; op.showsProgressPanel = false
      op.view?.frame = w.bounds
      op.runModal(for: self.window!, delegate: self, didRun: #selector(self.done(_:success:contextInfo:)), contextInfo: nil)
    }
  }
  @objc func done(_ op: NSPrintOperation, success: Bool, contextInfo: UnsafeMutableRawPointer?) { print(success ? "ok" : "fail"); exit(success ? 0 : 1) }
}
let app = NSApplication.shared
app.setActivationPolicy(.prohibited)
let d = D()
let wv = WKWebView(frame: NSRect(x: 0, y: 0, width: 505, height: 746))
let win = NSWindow(contentRect: wv.frame, styleMask: [.borderless], backing: .buffered, defer: false)
win.contentView = wv; d.window = win; wv.navigationDelegate = d
wv.loadFileURL(inURL, allowingReadAccessTo: URL(fileURLWithPath: "/"))
DispatchQueue.main.asyncAfter(deadline: .now() + 180) { print("timeout"); exit(2) }
app.run()
'''

CSS = '''
body { font-family: Arial, Helvetica, sans-serif; font-size: 9pt; line-height: 1.32; color: #000; margin: 0; }
h1 { font-size: 15pt; margin: 0 0 10pt 0; }
h2 { font-size: 11pt; margin: 12pt 0 6pt 0; }
p { margin: 0 0 5pt 0; text-align: left; }
.title { font-size: 12pt; font-weight: bold; margin: 0 0 8pt 0; }
.ph { background: #ffff00; }
.newpage { break-before: page; page-break-before: always; }
.fig img { display: block; width: 100%; max-height: 540pt; object-fit: contain; margin: 0 0 6pt 0; }
.cap { font-weight: bold; margin: 4pt 0 4pt 0; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 8pt 0; font-size: 6.6pt; line-height: 1.2; page-break-inside: auto; }
tr { page-break-inside: avoid; }
th, td { border: 0.4pt solid #8a8a8a; padding: 1.2pt 2.4pt; vertical-align: top; }
th { background: #eeeeee; font-weight: bold; }
td.r, th.r { text-align: right; }
ul { margin: 0 0 5pt 14pt; padding: 0; }
li { margin: 0 0 2pt 0; }
sup, sub { font-size: 70%; line-height: 0; }
.toc p { margin: 0 0 2pt 0; }
'''


def inline(t):
    t = html.escape(t, quote=False).replace('\\*', '&#42;')
    t = re.sub(r'\[\[([A-Z]+:?[^\]]*)\]\]', lambda m: '<span class="ph">[[%s]]</span>' % m.group(1), t)
    t = re.sub(r'==(.+?)==', r'<span class="ph">\1</span>', t)
    t = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', t)
    t = re.sub(r'`([^`]+)`', r'<code>\1</code>', t)
    t = re.sub(r'\^([^\^\s]{1,40})\^', r'<sup>\1</sup>', t)
    t = re.sub(r'~([^~\s]{1,40})~', r'<sub>\1</sub>', t)
    t = re.sub(r'(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])', r'<i>\1</i>', t)
    return t


def table_html(lines):
    rows = [[c.strip() for c in l.strip().strip('|').split('|')] for l in lines]
    head, sep, body = rows[0], rows[1], rows[2:]
    al = ['r' if s.strip().endswith(':') and not s.strip().startswith(':') else '' for s in sep]
    out = ['<table><thead><tr>']
    out += ['<th class="%s">%s</th>' % (al[i] if i < len(al) else '', inline(c)) for i, c in enumerate(head)]
    out.append('</tr></thead><tbody>')
    for r in body:
        out.append('<tr>' + ''.join('<td class="%s">%s</td>' % (al[i] if i < len(al) else '', inline(c)) for i, c in enumerate(r)) + '</tr>')
    out.append('</tbody></table>')
    return ''.join(out)


def blocks(md):
    out = []
    for chunk in re.split(r'\n\s*\n', md.strip()):
        lines = [l for l in chunk.split('\n') if l.strip()]
        if not lines:
            continue
        if len(lines) == 1 and re.fullmatch(r'\s*(-{3,}|\*{3,})\s*', lines[0]):
            continue
        if all(l.lstrip().startswith('|') for l in lines) and len(lines) >= 2:
            out.append(table_html(lines))
        elif all(l.lstrip().startswith('- ') for l in lines):
            out.append('<ul>' + ''.join('<li>%s</li>' % inline(l.lstrip()[2:]) for l in lines) + '</ul>')
        else:
            out.append('<p>%s</p>' % inline(' '.join(l.strip() for l in lines)))
    return '\n'.join(out)


def main():
    md = open(B.SI_MD, encoding='utf-8').read()
    md = re.sub(r'<!--.*?-->', '', md, flags=re.S)
    doc_s1 = md.split('\n# Document S1\n', 1)[1].split('\n# Excel tables', 1)[0]
    index = md.split('## Index of supplemental items', 1)[1].split('\n---', 1)[0]
    s1_line = re.search(r'\*\*(Document S1\.[^*]*)\*\*', index).group(1)
    main_md = open(os.path.join(os.path.dirname(B.SI_MD), 'main_text.md'), encoding='utf-8').read()
    title = re.search(r'(?m)^# (.+)$', main_md).group(1).strip()
    parts, contents = [], []
    for name, text in B.sections(doc_s1, 2):
        if not name:
            continue
        body = B.strip_heading(text)
        if name == 'Supplemental figures':
            for fid, ftitle, legend in B.figure_chunks(body):
                contents.append(ftitle)
                src = os.path.join(B.IMG, 'Figure%s.png' % fid)
                assert os.path.exists(src), src
                parts.append('<div class="newpage fig"><img src="file://%s" alt="Figure %s"><p class="cap">%s</p>%s</div>' % (src, fid, inline(ftitle), blocks(legend)))
        elif name == 'Supplemental tables':
            for c in re.split(r'(?m)^(?=\*\*Table S\d+\.)', body):
                m = re.match(r'\*\*(Table S\d+\..*?)\*\*[ \t]*\n', c)
                if not m:
                    continue
                contents.append(m.group(1))
                parts.append('<div class="newpage"><p class="cap">%s</p>%s</div>' % (inline(m.group(1)), blocks(c[m.end():])))
        elif name == 'Supplemental notes':
            notes = []
            for c in re.split(r'(?m)^(?=\*\*Note S\d+\.)', body):
                m = re.match(r'\*\*(Note S\d+\..*?)\*\*[ \t]*\n', c)
                if not m:
                    continue
                contents.append(m.group(1))
                notes.append('<p class="cap">%s</p>%s' % (inline(m.group(1)), blocks(c[m.end():])))
            parts.append('<div class="newpage">%s</div>' % '\n'.join(notes))
    front = ('<h1>Supplemental information</h1><p class="title">%s</p><p><span class="ph">[[PENDING: author names]]</span></p><p>%s</p>'
             '<h2>Contents</h2><div class="toc">%s</div>') % (inline(title), inline(s1_line), ''.join('<p>%s</p>' % inline(c) for c in contents))
    doc = '<!doctype html><html><head><meta charset="utf-8"><style>%s</style></head><body>%s\n%s</body></html>' % (CSS, front, '\n'.join(parts))
    tmp = tempfile.mkdtemp(prefix='si_pdf_')
    hp = os.path.join(tmp, 'DocumentS1.html')
    open(hp, 'w', encoding='utf-8').write(doc)
    sw = os.path.join(tmp, 'h2p.swift'); exe = os.path.join(tmp, 'h2p')
    open(sw, 'w').write(SWIFT)
    subprocess.run(['swiftc', '-O', sw, '-o', exe], check=True, capture_output=True)
    if os.path.exists(OUT):
        os.remove(OUT)
    r = subprocess.run([exe, hp, OUT], capture_output=True, text=True, timeout=240)
    if r.stdout.strip() != 'ok' or not os.path.exists(OUT):
        sys.exit('PDF print failed: %s %s' % (r.stdout, r.stderr))
    left = re.findall(r'\*\*|(?<!\w)\^\S|~[^~\s]{1,40}~|_\{|==', re.sub(r'<[^>]+>', '', doc))
    left += re.findall(r'<p>\s*[-*]{3,}\s*</p>', doc)
    print('02 PDF:', OUT, os.path.getsize(OUT), 'bytes;', len(contents), 'items; markdown leftovers:', len(left))


if __name__ == '__main__':
    main()

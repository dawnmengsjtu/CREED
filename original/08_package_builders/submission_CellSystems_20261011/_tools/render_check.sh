#!/bin/zsh
# Render check used on 2026-10-11 (Pages did not answer AppleEvents): docx -> Quick Look (Apple OfficeImport) HTML
# -> WebKit paginated A4 PDF (html2pdf.swift) -> server pdftoppm -r 60 -> page PNGs in _render_check/png.
set -e
PKG=${0:A:h:h}; SP=${TMPDIR:-/tmp}/cell_render; rm -rf $SP; mkdir -p $SP/ql
swiftc -O $PKG/_tools/html2pdf.swift -o $SP/html2pdf
cd $PKG
for f in 01_Manuscript_CREED_CellSystems 02_Supplemental_Information_DocumentS1 06_Cover_Letter_CREED 07_Highlights_eTOC; do
  (qlmanage -p -o $SP/ql $f.docx >/dev/null 2>&1 &); done
sleep 15; pkill -f 'qlmanage -p' || true
for f in 01_Manuscript_CREED_CellSystems 02_Supplemental_Information_DocumentS1 06_Cover_Letter_CREED 07_Highlights_eTOC; do
  $SP/html2pdf $SP/ql/$f.docx.qlpreview/Preview.html $PKG/_render_check/$f.pdf; done
R=/public/home/mengxl/workspaces/creed-research/figure_work/20261010_figures_v4_six/_render_check/cell_pkg_20261011
ssh -o BatchMode=yes lab-a6000 "mkdir -p $R && rm -f $R/*"
scp -q -o BatchMode=yes _render_check/*.pdf lab-a6000:$R/
ssh -o BatchMode=yes lab-a6000 "cd $R && for f in *.pdf; do pdftoppm -png -r 60 \$f \${f%.pdf}; done"
rm -rf _render_check/png; mkdir -p _render_check/png
scp -q -o BatchMode=yes "lab-a6000:$R/*.png" _render_check/png/

# v3 (Cell Systems) shared file for Figure 1 (data overview, panels A-H). SHARED - the Figure 1 agent edits A.R..H.R only.
# Style contract: ../cell_style.R + ../../V3_STYLE_CONTRACT.md. Page 174 x 235 mm, two 84 mm columns (x = 1, x = 89).
suppressPackageStartupMessages({library(ggplot2);library(grid);library(cowplot);library(ggalluvial);library(dplyr);library(tidyr);library(readr);library(svglite);library(jsonlite)})
SCRIPT_DIR<-normalizePath(dirname(sub('^--file=','',commandArgs(FALSE)[grep('^--file=',commandArgs(FALSE))][1])))
source(file.path(SCRIPT_DIR,'..','cell_style.R'))
ROOT<-normalizePath(file.path(SCRIPT_DIR,'../..'))
OUT<-file.path(ROOT,'delivery_R_20261009_v3/figure1');DATA<-file.path(OUT,'plot_data')   # plot_data = frozen copy of the accepted 20261007 tables
for(d in c('panels','main','qa','panel_data'))dir.create(file.path(OUT,d),recursive=TRUE,showWarnings=FALSE)
read_table<-function(n)read_csv(file.path(DATA,paste0(n,'.csv')),show_col_types=FALSE)
S<-fromJSON(file.path(DATA,'summary.json'))
# Eight slots re-packed in two columns (reading order A B / C D / E F / G H keeps the accepted v2 arrangement: E and G right, F and H left)
LAYOUT<-data.frame(panel=LETTERS[1:8],x_mm=c(1,89,1,89,89,1,89,1),y_mm=c(1,1,69,69,105,133,167,200),w_mm=84,h_mm=c(65,65,61,33,59,64,67,34))
PAGE<-c(w=174,h=235)
BASE<-theme_cell()
TISSUE<-read_table('B_tissue_composition');TISSUE_COL<-setNames(TISSUE_PAL[seq_len(nrow(TISSUE))],TISSUE$group)   # 8 Okabe-Ito hues + gray for 'Other tissues'
TISSUE_LABEL<-setNames(TISSUE$label,TISSUE$group)
FOLD_COL<-FOLD_RAMP
# Layer colours for the data-overview schematics: M gray, RNA blue, P purple; functional target (AUC) vermilion; held-out vs training = black / gray
MEASURED<-LAYER_COL[['M']];VIRTUAL<-LAYER_COL[['P']];RNA_COL<-LAYER_COL[['RNA']];TARGET<-OI[['vermilion']]
PARTITION_COL<-c(`Held out`='#000000',Training='#BDBDBD')
PAL<-c(unname(OI[c('blue','purple','vermilion','skyblue','green','orange','yellow')]),'#7F7F7F','#BDBDBD')   # legacy name; prefer the named vectors above
slot<-function(id)LAYOUT[LAYOUT$panel==id,]
export_panel<-function(p,id){row<-slot(id);stopifnot(nrow(row)==1);export_panel_cell(p,id,row$w_mm,row$h_mm,file.path(OUT,'panels'),data_dir=file.path(OUT,'panel_data'))}

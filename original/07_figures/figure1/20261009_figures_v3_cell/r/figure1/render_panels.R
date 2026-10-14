a<-commandArgs(FALSE);script<-sub('^--file=','',a[grep('^--file=',a)][1]);source(file.path(dirname(script),'common.R'))
ids<-commandArgs(TRUE);if(!length(ids))ids<-LETTERS[1:8]
for(id in ids){stopifnot(id%in%LETTERS[1:8]);source(file.path(SCRIPT_DIR,paste0(id,'.R')));export_panel(get(paste0('make_',id))(),id)}
write_csv(LAYOUT,file.path(OUT,'ASSEMBLY_LAYOUT_MM.csv'))
writeLines(c('Renderer: R; child grobs saved at final physical sizes','Assembly reads RDS; no raster resizing','Style: r/cell_style.R (theme_cell, Okabe-Ito contract); see V3_STYLE_CONTRACT.md','Typography: Helvetica; 8 pt panel letters, 7 pt axis titles, 6 pt ticks/legend; no in-panel titles or prose'),file.path(OUT,'qa/STYLE_CONTRACT.txt'))

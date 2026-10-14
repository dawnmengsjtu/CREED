a<-commandArgs(FALSE);script<-sub('^--file=','',a[grep('^--file=',a)][1]);source(file.path(dirname(script),'common.R'))
# Usage: assemble.R [blank]   -- 'blank' writes qa/Figure1_v3_blank.{pdf,png} with slot outlines only (no RDS needed)
blank<-length(commandArgs(TRUE))>0&&commandArgs(TRUE)[1]=='blank'
write_csv(LAYOUT,file.path(OUT,'qa/layout_mm.csv'))
if(blank)assemble_cell(LAYOUT,PAGE,file.path(OUT,'panels'),file.path(OUT,'qa','Figure1_v3_blank'),blank=TRUE) else assemble_cell(LAYOUT,PAGE,file.path(OUT,'panels'),file.path(OUT,'main','Figure1_v3'))

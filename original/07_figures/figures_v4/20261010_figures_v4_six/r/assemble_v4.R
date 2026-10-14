# assemble_v4.R <fig> [blank]   e.g.  Rscript r/assemble_v4.R Figure3
# Reads r/<fig>/layout.csv (first row PAGE,0,0,<w>,<h>; then panel,x_mm,y_mm,w_mm,h_mm[,align_v,align_h]), places each
# out/<fig>/panels/<panel>.rds at its native size (refuses a panel whose stored size differs from its slot), checks
# overlaps and page bounds (hard stops), runs the Figure-1-standard layout QA (soft: printed + qa/layout_check.txt),
# and writes out/<fig>/main/<fig>.{pdf,png} (cairo PDF, fonts embedded; 600 dpi RGB PNG).
# 'blank' draws the slot grid only -> out/<fig>/qa/<fig>_blank.{pdf,png} (no RDS needed).
source(file.path(dirname(normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))),'v4_common.R'))
args<-commandArgs(TRUE);if(!length(args))stop('usage: assemble_v4.R <fig> [blank]')
FIG<-check_fig_v4(args[1]);blank<-length(args)>1&&args[2]=='blank'
OUT<-ensure_out_v4(FIG);LAY<-read_layout(FIG);S<-LAY$slots;page<-LAY$page;W<-page[['w']];H<-page[['h']]
# ---- hard checks: page size, bounds, overlaps, native panel size
if(W>174+1e-9||H>235+1e-9)stop(sprintf('page %g x %g mm exceeds 174 x 235 mm',W,H))
oob<-S$panel[S$x_mm<0|S$y_mm<0|S$x_mm+S$w_mm>W+1e-9|S$y_mm+S$h_mm>H+1e-9|S$w_mm<=0|S$h_mm<=0]
if(length(oob))stop('slots outside the page: ',paste(oob,collapse=' '))
if(nrow(S)>1)for(i in 1:(nrow(S)-1))for(j in (i+1):nrow(S)){a<-S[i,];b<-S[j,]
 if(a$x_mm<b$x_mm+b$w_mm-1e-9&&b$x_mm<a$x_mm+a$w_mm-1e-9&&a$y_mm<b$y_mm+b$h_mm-1e-9&&b$y_mm<a$y_mm+a$h_mm-1e-9)stop('slots overlap: ',a$panel,' and ',b$panel)}
PD<-file.path(OUT,'panels')
if(!blank){miss<-S$panel[!file.exists(file.path(PD,paste0(S$panel,'.rds')))];if(length(miss))stop('missing panel RDS (render first): ',paste(miss,collapse=' '))
 G<-setNames(lapply(S$panel,function(id)readRDS(file.path(PD,paste0(id,'.rds')))),S$panel)
 for(id in S$panel){sz<-attr(G[[id]],'size_mm');s<-S[S$panel==id,]
  if(is.null(sz))stop('panel ',id,' RDS has no size_mm attribute (not exported by export_panel_v4); re-render it')
  if(abs(sz[['w']]-s$w_mm)>1e-6||abs(sz[['h']]-s$h_mm)>1e-6)stop(sprintf('panel %s was rendered at %g x %g mm but its slot is %g x %g mm; re-render (render_v4.R %s %s)',id,sz[['w']],sz[['h']],s$w_mm,s$h_mm,FIG,id))}}
# ---- soft QA against the Figure 1 standard (LAYOUT_GUIDE.md)
qa<-character(0);note<-function(lvl,...)qa<<-c(qa,sprintf('%s %s',lvl,paste0(...)))
L<-S[is_lettered(S$panel),]
main<-is_main_v4(FIG)
if(main){if(!identical(sort(L$panel),LETTERS[1:8]))note('FAIL','main figure needs exactly lettered panels A-H; has ',paste(sort(L$panel),collapse=''))
 if(abs(W-174)>1e-9)note('FAIL','main figure width must be 174 mm (is ',W,')')
 if(H<220)note('FAIL','main figure height ',H,' mm < 220 mm') else if(H<225)note('WARN','main figure height ',H,' mm < 225 mm target')}else if(H<80)note('FAIL','supplementary height ',H,' mm < 80 mm')
ys<-sort(unique(L$y_mm));rk<-numeric(0);st<- -Inf;k<-0;for(y in ys){if(y-st>5){k<-k+1;st<-y};rk[as.character(y)]<-k}   # tops within 5 mm = one row
ro<-L$panel[order(rk[as.character(L$y_mm)],L$x_mm)];if(!identical(ro,sort(L$panel)))note('FAIL','reading order (top->bottom, left->right by slot top) is ',paste(ro,collapse=''),'; letters must follow it')
GX<-c(1,44,89,132);GW<-c(41,84,172,129,127)
off<-L$panel[!(round(L$x_mm,3)%in%GX)];if(length(off))note('WARN','left edge off the grid (x in 1/44/89/132): ',paste(off,collapse=' '))
offw<-L$panel[!(round(L$w_mm,3)%in%GW)];if(length(offw))note('WARN','width off the grid (41/84/172 mm): ',paste(offw,collapse=' '))
area<-L$w_mm*L$h_mm/(W*H);big<-L$panel[area>.35];if(length(big))note('FAIL','panel larger than 35% of page: ',paste(sprintf('%s (%.0f%%)',big,100*area[area>.35]),collapse=' '))
# side-by-side lettered panels whose vertical spans overlap should share a top (letters on one baseline)
if(nrow(L)>1)for(i in 1:(nrow(L)-1))for(j in (i+1):nrow(L)){a<-L[i,];b<-L[j,]
 if(a$y_mm<b$y_mm+b$h_mm&&b$y_mm<a$y_mm+a$h_mm&&abs(a$y_mm-b$y_mm)>1e-6&&abs(a$y_mm-b$y_mm)<15)note('WARN',sprintf('%s (top %g) and %s (top %g) sit side by side with tops %g mm apart; align rows',a$panel,a$y_mm,b$panel,b$y_mm,abs(a$y_mm-b$y_mm)))}
# unused page area (coverage of the page by slots, ignoring 1 mm margins and 3-4 mm gutters)
cov<-sum(S$w_mm*S$h_mm)/((W-2)*(H-2));if(cov<.80)note('WARN',sprintf('slots cover only %.0f%% of the printable page; fill the page or reduce height',100*cov))
rows<-sort(unique(L$y_mm));note('INFO','row tops (mm): ',paste(rows,collapse=' '),'; page ',W,' x ',H,' mm; ',nrow(L),' lettered + ',nrow(S)-nrow(L),' unlettered slots; coverage ',sprintf('%.0f%%',100*cov))
if(!any(grepl('^(FAIL|WARN)',qa)))note('PASS','layout meets the Figure 1 grid rules that can be checked automatically')
writeLines(qa,file.path(OUT,'qa','layout_check.txt'));cat(paste0('QA ',qa,'\n'),sep='')
write.csv(rbind(data.frame(panel='PAGE',x_mm=0,y_mm=0,w_mm=W,h_mm=H),S[,c('panel','x_mm','y_mm','w_mm','h_mm')]),file.path(OUT,'qa','layout_mm.csv'),row.names=FALSE)
# ---- draw
draw<-function(){grid.newpage();grid.rect(gp=gpar(fill='white',col=NA))
 for(i in seq_len(nrow(S))){z<-S[i,];pushViewport(viewport(x=unit(z$x_mm,'mm'),y=unit(H-z$y_mm,'mm'),width=unit(z$w_mm,'mm'),height=unit(z$h_mm,'mm'),just=c('left','top'),clip='on'))
  if(blank){grid.rect(gp=gpar(fill='#F4F4F4',col='#9E9E9E',lwd=.5,lty=3));if(is_lettered(z$panel))grid.draw(panel_letter_grob(z$panel))
   grid.text(sprintf('%s  %g x %g mm',if(is_lettered(z$panel))'' else z$panel,z$w_mm,z$h_mm),gp=gpar(fontfamily=CELL_FONT,fontsize=6,col='#7F7F7F'))}else grid.draw(G[[z$panel]])
  popViewport()}}
stem<-if(blank)file.path(OUT,'qa',paste0(FIG,'_blank'))else file.path(OUT,'main',FIG)
cairo_pdf(paste0(stem,'.pdf'),width=W/25.4,height=H/25.4,family=CELL_FONT);draw();invisible(dev.off())
png(paste0(stem,'.png'),width=W,height=H,units='mm',res=600,type='cairo',bg='white');draw();invisible(dev.off())
cat(sprintf('ASSEMBLED %s %g x %g mm from %d slots%s -> %s.{pdf,png}\n',FIG,W,H,nrow(S),if(blank)' (BLANK)' else '',stem))

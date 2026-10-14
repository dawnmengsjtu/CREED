# cell_style.R - the ONE style contract for the CREED v3 (Cell Systems) figures.
# Sourced by r/figures2_6/common.R and r/figure1/common.R. Figure agents never edit this file;
# they use the names defined here and call export_panel() (defined in common.R, which wraps
# export_panel_cell() below). Full prose contract: V3_STYLE_CONTRACT.md next to the r/ directory.
suppressPackageStartupMessages({library(ggplot2);library(grid)})
stopifnot(packageVersion('ggplot2')>='3.4.0')

# ---------------------------------------------------------------- typography / geometry (pt, mm)
CELL_FONT<-'Helvetica'               # fontconfig substitutes Liberation Sans (metric clone) on the server
PT_PANEL_LETTER<-8;PT_AXIS_TITLE<-7;PT_TICK<-6;PT_LEGEND<-6;PT_STRIP<-6.5;PT_MIN<-6
LW_AXIS<-0.4/.pt;LW_TICK<-0.3/.pt;LW_LINE<-0.5/.pt;LW_ERR<-0.4/.pt;LW_THIN<-0.3/.pt   # ggplot linewidth units
TICK_LEN<-unit(1,'mm')
PT_SMALL<-1.2;PT_LARGE<-1.8           # ggplot point size (~ diameter in mm): per-fold points small, means large
GEOM_TEXT_MIN<-PT_MIN/.pt             # geom_text(size=) must be >= this (2.11) -> 6 pt
PANEL_GAP_MM<-3
INK<-'#000000';LINE<-'#000000';MISSING<-'#E5E5E5';GRID_OFF<-element_blank()

# ---------------------------------------------------------------- colour contract (Okabe-Ito, colour-blind safe)
OI<-c(black='#000000',orange='#E69F00',skyblue='#56B4E9',green='#009E73',yellow='#F0E442',blue='#0072B2',vermilion='#D55E00',purple='#CC79A7')
alpha_hex<-function(col,a)grDevices::adjustcolor(col,alpha.f=a)   # 8-digit hex with alpha; cairo honours it
# Deployment arms (the five arms that recur in every figure). Keys are the v3 canonical arm ids.
ARM_COL_CELL<-c(m_only='#7F7F7F',            # M only
                m_vrna='#0072B2',            # M + virtual RNA
                m_vp='#CC79A7',              # M + virtual P
                measured=alpha_hex('#000000',.6),   # measured reference arms (M + RNA, M + P): black 60 %
                shuffle='#BDBDBD',           # paired-shuffle controls
                tissue='#E69F00',            # tissue-label arms
                null='#000000')              # null model line (dashed), training-mean predictor
ARM_LABEL_CELL<-c(m_only='M only',m_vrna='M + virtual RNA',m_vp='M + virtual P',measured='M + measured',shuffle='Shuffle',tissue='M + tissue',null='Null')
# Virtual layers (used wherever a layer, not an arm, is coloured: map axes, recovery share rows, generators)
LAYER_COL<-c(M='#7F7F7F',RNA='#0072B2',P='#CC79A7')
# Supervised panels and their random-panel controls (35 % alpha of the layer colour)
PANEL_COL<-c(protein='#D55E00',gene='#56B4E9')
PANEL_RANDOM_COL<-c(protein_random=alpha_hex('#D55E00',.35),gene_random=alpha_hex('#56B4E9',.35))
# Folds F0-F4: gray-to-black ramp, never rainbow
FOLD_RAMP<-setNames(c('#BDBDBD','#969696','#737373','#525252','#000000'),as.character(0:4))
# Endpoints. Rule 1 (default): same colour, GDSC filled / CRISPR hollow. Rule 2 (both endpoints in one
# panel and colour is free): GDSC blue / CRISPR vermilion. A figure picks ONE rule and the legend states it.
ENDPOINT_SHAPE<-c(GDSC=16,CRISPR=1)
ENDPOINT_SHAPE_FILLABLE<-c(GDSC=21,CRISPR=21)   # use with fill = colour for GDSC, fill = 'white' for CRISPR
ENDPOINT_COL<-c(GDSC='#0072B2',CRISPR='#D55E00')
# Tissue groups (Fig 1 rings/alluvial; 8 named tissues + Other): the 8 Okabe-Ito hues, Other = light gray
TISSUE_PAL<-c(OI[['orange']],OI[['skyblue']],OI[['green']],OI[['yellow']],OI[['blue']],OI[['vermilion']],OI[['purple']],OI[['black']],'#BDBDBD')
# Heat-map scales
GAIN_DIV<-c(low='#0072B2',mid='#FFFFFF',high='#D55E00')    # signed gains
SHARE_SEQ<-c(low='#FFFFFF',high='#009E73')                   # shares / fractions

# ---------------------------------------------------------------- theme
theme_cell<-function(base_size=PT_AXIS_TITLE,legend_pos='bottom'){
 theme_classic(base_family=CELL_FONT,base_size=base_size)+theme(
  text=element_text(colour=INK,family=CELL_FONT),
  line=element_line(colour=INK),
  axis.text=element_text(size=PT_TICK,colour=INK),
  axis.title=element_text(size=PT_AXIS_TITLE,colour=INK,face='plain'),
  axis.title.x=element_text(margin=margin(t=1.5,unit='mm')),axis.title.y=element_text(margin=margin(r=1.5,unit='mm')),
  axis.line=element_line(linewidth=LW_AXIS,colour=INK,lineend='square'),
  axis.ticks=element_line(linewidth=LW_TICK,colour=INK),axis.ticks.length=TICK_LEN,
  panel.grid=element_blank(),panel.grid.major=element_blank(),panel.grid.minor=element_blank(),
  panel.background=element_blank(),plot.background=element_blank(),
  panel.border=element_blank(),
  panel.spacing=unit(PANEL_GAP_MM,'mm'),
  strip.background=element_blank(),strip.text=element_text(size=PT_STRIP,face='bold',colour=INK,margin=margin(0,0,1.2,0,'mm')),
  strip.placement='outside',
  legend.position=legend_pos,legend.direction=if(legend_pos=='bottom')'horizontal' else 'vertical',
  legend.title=element_blank(),legend.text=element_text(size=PT_LEGEND,colour=INK),
  legend.key=element_blank(),legend.key.size=unit(2.4,'mm'),legend.key.spacing.x=unit(2,'mm'),legend.key.spacing.y=unit(.6,'mm'),
  legend.background=element_blank(),legend.box.background=element_blank(),legend.margin=margin(0,0,0,0),legend.box.spacing=unit(1,'mm'),
  plot.title=element_blank(),plot.subtitle=element_blank(),plot.caption=element_blank(),plot.tag=element_blank(),
  plot.margin=margin(5,1.5,1,1.5,'mm'))   # 5 mm top: room for the 8 pt panel letter stamped by export_panel_cell
}
legend_inside<-function(x,y,just=c(0,1))theme(legend.position='inside',legend.position.inside=c(x,y),legend.justification=just,legend.direction='vertical')
legend_title_on<-function()theme(legend.title=element_text(size=PT_LEGEND,colour=INK))   # only when ambiguous; add AFTER BASE in the + chain

# ---------------------------------------------------------------- scale helpers
scale_colour_arm<-function(...)scale_colour_manual(values=ARM_COL_CELL,labels=ARM_LABEL_CELL,name=NULL,...)
scale_fill_arm<-function(...)scale_fill_manual(values=ARM_COL_CELL,labels=ARM_LABEL_CELL,name=NULL,...)
scale_colour_layer<-function(...)scale_colour_manual(values=LAYER_COL,name=NULL,...)
scale_fill_layer<-function(...)scale_fill_manual(values=LAYER_COL,name=NULL,...)
scale_colour_panel<-function(with_random=TRUE,...)scale_colour_manual(values=if(with_random)c(PANEL_COL,PANEL_RANDOM_COL)else PANEL_COL,name=NULL,...)
scale_fill_panel<-function(with_random=TRUE,...)scale_fill_manual(values=if(with_random)c(PANEL_COL,PANEL_RANDOM_COL)else PANEL_COL,name=NULL,...)
scale_colour_fold<-function(...)scale_colour_manual(values=FOLD_RAMP,labels=paste0('F',0:4),name=NULL,...)
scale_fill_fold<-function(...)scale_fill_manual(values=FOLD_RAMP,labels=paste0('F',0:4),name=NULL,...)
scale_endpoint<-function(rule=1,...)if(rule==1)list(scale_shape_manual(values=ENDPOINT_SHAPE,name=NULL,...))else list(scale_shape_manual(values=ENDPOINT_SHAPE,name=NULL,...),scale_colour_manual(values=ENDPOINT_COL,name=NULL,...))
fold_x<-function(...)scale_x_continuous(breaks=0:4,labels=paste0('F',0:4),...)
guide_cell_bar<-function(horizontal=TRUE,len=20)guide_colourbar(barwidth=if(horizontal)unit(len,'mm')else unit(2,'mm'),barheight=if(horizontal)unit(2,'mm')else unit(len,'mm'),ticks.colour=INK,ticks.linewidth=LW_TICK,frame.colour=INK,frame.linewidth=LW_TICK,title.position='top',direction=if(horizontal)'horizontal'else'vertical')
# Signed gain: diverging blue-white-vermilion, symmetric limits
scale_fill_gain<-function(limit,breaks=NULL,name=NULL,horizontal=TRUE,...){if(is.null(breaks)){b<-signif(limit,1);if(b>limit)b<-b/2;breaks<-c(-b,0,b)};scale_fill_gradient2(low=GAIN_DIV[['low']],mid=GAIN_DIV[['mid']],high=GAIN_DIV[['high']],midpoint=0,limits=c(-limit,limit),breaks=breaks,name=name,na.value=MISSING,oob=scales::squish,guide=guide_cell_bar(horizontal),...)}   # values beyond +-limit are clamped to the end colour, never dropped
# Share / fraction: sequential white-green
scale_fill_share<-function(limits=c(0,1),breaks=waiver(),name=NULL,horizontal=TRUE,...)scale_fill_gradient(low=SHARE_SEQ[['low']],high=SHARE_SEQ[['high']],limits=limits,breaks=breaks,name=name,na.value=MISSING,oob=scales::squish,guide=guide_cell_bar(horizontal),...)
# Thin uncapped error bars / mean points in contract geometry
geom_err<-function(mapping=NULL,data=NULL,...)geom_errorbar(mapping=mapping,data=data,width=0,linewidth=LW_ERR,...)
geom_err_h<-function(mapping=NULL,data=NULL,...)geom_errorbar(mapping=mapping,data=data,width=0,linewidth=LW_ERR,orientation='y',...)
geom_fold_points<-function(mapping=NULL,data=NULL,...)geom_point(mapping=mapping,data=data,size=PT_SMALL,colour='#7F7F7F',alpha=.8,shape=16,...)
geom_null_line<-function(y)geom_hline(yintercept=y,linetype='22',linewidth=LW_LINE,colour=INK)

# ---------------------------------------------------------------- schematic helpers (mm from the panel's TOP-LEFT; h = slot height)
dtxt<-function(t,x,y,size=PT_TICK,bold=FALSE,col=INK,just='centre',h=88,rot=0){stopifnot(size>=PT_MIN);textGrob(t,x=unit(x,'mm'),y=unit(h-y,'mm'),just=just,rot=rot,gp=gpar(fontfamily=CELL_FONT,fontsize=size,fontface=if(bold)'bold'else'plain',col=col,lineheight=1.1))}
drect<-function(x,y,w,ht,fill='white',col=INK,lwd=0.5,h=88,lty=1,r=0)if(r>0)roundrectGrob(x=unit(x,'mm'),y=unit(h-y,'mm'),width=unit(w,'mm'),height=unit(ht,'mm'),just=c('left','top'),r=unit(r,'mm'),gp=gpar(fill=fill,col=col,lwd=lwd,lty=lty))else rectGrob(x=unit(x,'mm'),y=unit(h-y,'mm'),width=unit(w,'mm'),height=unit(ht,'mm'),just=c('left','top'),gp=gpar(fill=fill,col=col,lwd=lwd,lty=lty))
dline<-function(x,y,x2,y2,arrow=FALSE,h=88,lty=1,col=INK,lwd=0.5)segmentsGrob(x0=unit(x,'mm'),y0=unit(h-y,'mm'),x1=unit(x2,'mm'),y1=unit(h-y2,'mm'),gp=gpar(col=col,lwd=lwd,lty=lty,fill=col),arrow=if(arrow)grid::arrow(length=unit(1.2,'mm'),type='closed')else NULL)
dgroup<-function(g)gTree(children=do.call(gList,g))
# Vector matrix icon; tiles encode a supplied observed numeric matrix only (no invented values)
dmatrix<-function(values,x,y,step=1.8,col=LAYER_COL[['M']],h=88){ramp<-colorRampPalette(c('#FFFFFF',col))(100);r<-range(values,na.rm=TRUE);z<-(values-r[1])/max(diff(r),1e-8);g<-list();for(i in seq_len(nrow(z)))for(j in seq_len(ncol(z)))g<-c(g,list(drect(x+(j-1)*step,y+(i-1)*step,step*.86,step*.86,if(is.na(z[i,j]))MISSING else ramp[1+round(z[i,j]*99)],col=NA,h=h)));g}
# Word budget check for schematics (contract: <= 25 words in a design panel)
grob_words<-function(g){n<-0;walk<-function(x){if(inherits(x,'text'))n<<-n+length(strsplit(paste(as.character(x$label),collapse=' '),'[[:space:]]+')[[1]]);if(!is.null(x$children))for(ch in x$children)walk(ch)};walk(g);n}

# ---------------------------------------------------------------- guard + export
.lab_present<-function(v)!is.null(v)&&!(is.character(v)&&all(!nzchar(trimws(v))))
.plot_labels<-function(p){l<-tryCatch(p@labels,error=function(e)NULL);if(is.null(l))l<-tryCatch(p$labels,error=function(e)NULL);l}
cell_guard<-function(p,id='?'){
 if(inherits(p,'ggplot')){
  l<-.plot_labels(p);bad<-c('title','subtitle','caption','tag')[vapply(c('title','subtitle','caption','tag'),function(k).lab_present(l[[k]]),TRUE)]
  if(length(bad))stop(sprintf('CELL_STYLE GUARD: panel %s has in-panel %s (%s). Remove it with labs(%s) and move the text to the figure legend.',id,paste(bad,collapse='/'),paste(sprintf('%s="%s"',bad,vapply(bad,function(k)substr(paste(l[[k]],collapse=' '),1,40),'')),collapse='; '),paste0(bad,'=NULL',collapse=',')),call.=FALSE)
  for(j in seq_along(p$layers)){ly<-p$layers[[j]];cl<-class(ly$geom)[1];sz<-ly$aes_params$size
   if(cl%in%c('GeomText','GeomLabel','GeomTextRepel','GeomLabelRepel')&&!is.null(sz)&&any(sz*.pt<PT_MIN-1e-6))stop(sprintf('CELL_STYLE GUARD: panel %s layer %d (%s) prints text at %.1f pt (< %g pt). Use size >= GEOM_TEXT_MIN (%.2f).',id,j,cl,min(sz)*.pt,PT_MIN,GEOM_TEXT_MIN),call.=FALSE)}
 }
 invisible(TRUE)}
# Panel letter: 8 pt bold uppercase, top-left, outside the plotting area (the theme reserves a 5 mm top margin)
panel_letter_grob<-function(id)textGrob(toupper(id),x=unit(.6,'mm'),y=unit(1,'npc')-unit(.8,'mm'),just=c('left','top'),gp=gpar(fontfamily=CELL_FONT,fontsize=PT_PANEL_LETTER,fontface='bold',col=INK))
# export_panel_cell(p, id, w_mm, h_mm, out_dir): refuses titles/subtitles/captions, stamps the letter, writes
# <id>.rds (for assembly), <id>.pdf (cairo_pdf, fonts embedded) and <id>.png (600 dpi, RGB) at the slot size.
export_panel_cell<-function(p,id,w_mm,h_mm,out_dir,data_dir=NULL,label=TRUE){
 cell_guard(p,id)
 if(!is.null(data_dir)&&inherits(p,'ggplot')){dir.create(data_dir,recursive=TRUE,showWarnings=FALSE)
  if(is.data.frame(p$data))readr::write_csv(p$data,file.path(data_dir,paste0(id,'_display.csv')))
  for(j in seq_along(p$layers)){ld<-p$layers[[j]]$data;if(is.data.frame(ld))readr::write_csv(ld,file.path(data_dir,paste0(id,'_layer',j,'_source.csv')))}}
 g<-if(inherits(p,'ggplot'))ggplotGrob(p)else p
 kids<-list(rectGrob(gp=gpar(fill='white',col=NA)),g);if(label)kids<-c(kids,list(panel_letter_grob(id)))
 final<-gTree(children=do.call(gList,kids))
 dir.create(out_dir,recursive=TRUE,showWarnings=FALSE)
 saveRDS(final,file.path(out_dir,paste0(id,'.rds')))
 cairo_pdf(file.path(out_dir,paste0(id,'.pdf')),width=w_mm/25.4,height=h_mm/25.4,family=CELL_FONT);grid.newpage();grid.draw(final);dev.off()
 png(file.path(out_dir,paste0(id,'.png')),width=w_mm,height=h_mm,units='mm',res=600,type='cairo',bg='white');grid.newpage();grid.draw(final);dev.off()
 if(!inherits(p,'ggplot')){nw<-grob_words(final)-1;if(nw>25)warning(sprintf('CELL_STYLE: schematic %s has %d words (> 25)',id,nw),call.=FALSE)}
 cat('PANEL',id,'RENDERED',w_mm,'x',h_mm,'mm\n');invisible(final)}
# Assemble a page from slot RDS files (LAYOUT: panel,x_mm,y_mm,w_mm,h_mm; page: c(w=,h=)); blank=TRUE draws slot outlines only.
assemble_cell<-function(LAYOUT,page,panel_dir,out_stem,blank=FALSE){
 for(i in 1:(nrow(LAYOUT)-1))for(j in (i+1):nrow(LAYOUT)){a<-LAYOUT[i,];b<-LAYOUT[j,];if(a$x_mm<b$x_mm+b$w_mm&&b$x_mm<a$x_mm+a$w_mm&&a$y_mm<b$y_mm+b$h_mm&&b$y_mm<a$y_mm+a$h_mm)stop('slots overlap: ',a$panel,' ',b$panel)}
 stopifnot(all(LAYOUT$x_mm+LAYOUT$w_mm<=page['w']),all(LAYOUT$y_mm+LAYOUT$h_mm<=page['h']),page['w']<=174,page['h']<=235)
 if(!blank)stopifnot(all(file.exists(file.path(panel_dir,paste0(LAYOUT$panel,'.rds')))))
 draw<-function(){grid.newpage();grid.rect(gp=gpar(fill='white',col=NA))
  for(i in seq_len(nrow(LAYOUT))){z<-LAYOUT[i,];pushViewport(viewport(x=unit(z$x_mm,'mm'),y=unit(page['h']-z$y_mm,'mm'),width=unit(z$w_mm,'mm'),height=unit(z$h_mm,'mm'),just=c('left','top'),clip='on'))
   if(blank){grid.rect(gp=gpar(fill='#F4F4F4',col='#9E9E9E',lwd=.5,lty=3));grid.draw(panel_letter_grob(z$panel));grid.text(sprintf('%g x %g mm',z$w_mm,z$h_mm),gp=gpar(fontfamily=CELL_FONT,fontsize=6,col='#9E9E9E'))}else grid.draw(readRDS(file.path(panel_dir,paste0(z$panel,'.rds'))))
   popViewport()}}
 cairo_pdf(paste0(out_stem,'.pdf'),width=page['w']/25.4,height=page['h']/25.4,family=CELL_FONT);draw();dev.off()
 png(paste0(out_stem,'.png'),width=page['w'],height=page['h'],units='mm',res=600,type='cairo',bg='white');draw();dev.off()
 cat('ASSEMBLED',basename(out_stem),page['w'],'x',page['h'],'mm from',nrow(LAYOUT),'slots',if(blank)'(BLANK)' else '','\n')}

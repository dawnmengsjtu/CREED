# v4_common.R - shared tooling for the CREED v4 "six main figures" build (SHARED: figure agents do not edit).
# Sources cell_style.R (the v3 Cell Systems contract, copied unchanged) and adds:
#   read_pd(fig, name)                         -> out/<fig>/plot_data/<name>.csv
#   read_layout(fig) / slot_v4(fig, id)        -> r/<fig>/layout.csv (first row PAGE,0,0,<w>,<h>)
#   export_panel_v4(p, fig, id, w_mm, h_mm)    -> out/<fig>/panels/<id>.{rds,pdf,png}, letter stamped top-left
#   fix_panel_margins(p, left=, right=, top=, bottom=)  -> gtable whose plot area starts at fixed mm offsets
#   align_v4(plots, align, axis)               -> cowplot::align_plots wrapper (guard first)
# Grid (Figure 1 standard): 174 mm page, columns x = 1 and x = 89 (84 mm, 4 mm gutter), 41 mm sub-columns at
# x = 1, 44, 89, 132 (2 mm gap), full width 172 mm at x = 1, 3 mm row gaps. See ../LAYOUT_GUIDE.md.
suppressPackageStartupMessages({library(ggplot2);library(grid);library(gtable);library(cowplot);library(dplyr);library(tidyr);library(readr);library(jsonlite)})
.v4_script<-function(){a<-commandArgs(FALSE);f<-sub('^--file=','',a[grep('^--file=',a)][1]);if(is.na(f))NA_character_ else normalizePath(f)}
if(!exists('V4_R')){V4_R<-Sys.getenv('V4_R',unset=NA);if(is.na(V4_R))V4_R<-dirname(.v4_script())}
V4_R<-normalizePath(V4_R);V4_ROOT<-normalizePath(file.path(V4_R,'..'))
source(file.path(V4_R,'cell_style.R'))
# Sandbox (smoke tests): V4_SANDBOX=_smoke puts scripts in r/_smoke/<fig> and outputs in out/_smoke/<fig>.
V4_SANDBOX<-Sys.getenv('V4_SANDBOX',unset='')
FIG_NAMES_V4<-c(paste0('Figure',2:6),paste0('FigureS',1:5))
check_fig_v4<-function(fig){if(!fig%in%FIG_NAMES_V4)stop('figure must be one of ',paste(FIG_NAMES_V4,collapse=', '),' (got "',fig,'")',call.=FALSE);invisible(fig)}
is_main_v4<-function(fig)grepl('^Figure[0-9]+$',fig)
fig_rdir<-function(fig)if(nzchar(V4_SANDBOX))file.path(V4_R,V4_SANDBOX,fig)else file.path(V4_R,fig)
fig_out<-function(fig)if(nzchar(V4_SANDBOX))file.path(V4_ROOT,'out',V4_SANDBOX,fig)else file.path(V4_ROOT,'out',fig)
ensure_out_v4<-function(fig){for(d in c('plot_data','panels','panel_data','main','qa'))dir.create(file.path(fig_out(fig),d),recursive=TRUE,showWarnings=FALSE);invisible(fig_out(fig))}

# ---------------------------------------------------------------- data
read_pd<-function(fig,name){f<-file.path(fig_out(fig),'plot_data',paste0(sub('\\.csv$','',name),'.csv'));if(!file.exists(f))stop('read_pd: missing ',f,call.=FALSE);read_csv(f,show_col_types=FALSE,na=c('','NA'))}
read_summary_v4<-function(fig){f<-file.path(fig_out(fig),'plot_data','summary.json');if(!file.exists(f))stop('read_summary_v4: copy summary.json into ',dirname(f),' first',call.=FALSE);fromJSON(f)}

# ---------------------------------------------------------------- layout
read_layout<-function(fig){f<-file.path(fig_rdir(fig),'layout.csv');if(!file.exists(f))stop('missing layout ',f,call.=FALSE)
 L<-read.csv(f,stringsAsFactors=FALSE,strip.white=TRUE,colClasses=c(panel='character'))
 need<-c('panel','x_mm','y_mm','w_mm','h_mm');if(!all(need%in%names(L)))stop('layout.csv needs columns ',paste(need,collapse=','),call.=FALSE)
 if(L$panel[1]!='PAGE')stop('layout.csv: first row must be PAGE,0,0,<w>,<h>',call.=FALSE)
 for(k in need[-1])L[[k]]<-as.numeric(L[[k]])
 for(k in c('align_v','align_h'))if(!k%in%names(L))L[[k]]<-'' else L[[k]][is.na(L[[k]])]<-''
 page<-c(w=L$w_mm[1],h=L$h_mm[1]);S<-L[-1,,drop=FALSE];rownames(S)<-NULL
 if(anyDuplicated(S$panel))stop('layout.csv: duplicated panel ids',call.=FALSE)
 list(page=page,slots=S)}
slot_v4<-function(fig,id){S<-read_layout(fig)$slots;r<-S[S$panel==id,];if(nrow(r)!=1)stop('panel ',id,' not in ',fig,' layout.csv',call.=FALSE);r}
is_lettered<-function(id)grepl('^[A-Z]$',id)   # single capital = lettered panel; any other id (e.g. LEG_CD) = unlabelled shared legend / spacer

# ---------------------------------------------------------------- alignment helpers
.nullpdf<-function(){if(dev.cur()==1){pdf(NULL);TRUE}else FALSE}
.mm<-function(u)convertWidth(u,'mm',valueOnly=TRUE)
.mmh<-function(u)convertHeight(u,'mm',valueOnly=TRUE)
# Force the plot area of a ggplot to start `left` mm from the slot's left edge (and/or end `right` mm from its right,
# start `top` mm below the slot top, end `bottom` mm above its bottom). Stacked panels given the same `left` share
# one left plot edge even when their y tick labels differ. Returns a gtable (guard is run on the ggplot first).
fix_panel_margins<-function(p,left=NULL,right=NULL,top=NULL,bottom=NULL,id='?'){
 if(inherits(p,'ggplot'))cell_guard(p,id);opened<-.nullpdf();on.exit(if(opened)dev.off())
 g<-if(inherits(p,'ggplot'))ggplotGrob(p)else p
 pl<-g$layout[grepl('^panel',g$layout$name),];if(!nrow(pl))stop('fix_panel_margins: no panel in grob',call.=FALSE)
 adj<-function(units,idx,target,first,what){if(!length(idx)){if(target>0)warning(what,': no columns to pad');return(units)}
  cur<-sum(vapply(idx,function(i){u<-units[i];if(what%in%c('left','right')).mm(u)else .mmh(u)},0))
  if(cur>target+1e-6)warning(sprintf('fix_panel_margins (%s): content needs %.1f mm > requested %.1f mm; widen the margin',what,cur,target),call.=FALSE)
  units[first]<-units[first]+unit(max(target-cur,0),'mm');units}
 l0<-min(pl$l);r0<-max(pl$r);t0<-min(pl$t);b0<-max(pl$b)
 if(!is.null(left))g$widths<-adj(g$widths,seq_len(l0-1),left,1,'left')
 if(!is.null(right))g$widths<-adj(g$widths,if(r0<length(g$widths))(r0+1):length(g$widths)else integer(0),right,length(g$widths),'right')
 if(!is.null(top))g$heights<-adj(g$heights,seq_len(t0-1),top,1,'top')
 if(!is.null(bottom))g$heights<-adj(g$heights,if(b0<length(g$heights))(b0+1):length(g$heights)else integer(0),bottom,length(g$heights),'bottom')
 g}
# cowplot alignment across panels exported to different slots (same-width slots for align='v', same-height for 'h').
align_v4<-function(plots,align='v',axis=if(align=='v')'l' else 'tb'){
 for(n in names(plots))if(inherits(plots[[n]],'ggplot'))cell_guard(plots[[n]],n)
 opened<-.nullpdf();on.exit(if(opened)dev.off())
 out<-cowplot::align_plots(plotlist=plots,align=align,axis=axis);names(out)<-names(plots);out}

# ---------------------------------------------------------------- export
# export_panel_v4(p, fig, id, w_mm, h_mm): guard (no title/subtitle/caption/tag, no text < 6 pt), stamp the 8 pt bold
# letter top-left (single-capital ids only), write rds+pdf+png at exactly w_mm x h_mm. `grob=` passes a pre-aligned
# gtable built from `p` (p is still used for the guard and the panel_data dump).
export_panel_v4<-function(p,fig,id,w_mm,h_mm,grob=NULL,label=is_lettered(id)){
 cell_guard(p,id);if(!is.null(grob)&&inherits(grob,'ggplot'))cell_guard(grob,id)
 out<-ensure_out_v4(fig)
 if(inherits(p,'ggplot')){dd<-file.path(out,'panel_data')
  if(is.data.frame(p$data))write_csv(p$data,file.path(dd,paste0(id,'_display.csv')))
  for(j in seq_along(p$layers)){ld<-p$layers[[j]]$data;if(is.data.frame(ld))write_csv(ld,file.path(dd,paste0(id,'_layer',j,'_source.csv')))}}
 opened<-.nullpdf()
 g<-if(!is.null(grob)){if(inherits(grob,'ggplot'))ggplotGrob(grob)else grob}else if(inherits(p,'ggplot'))ggplotGrob(p)else p
 if(opened)dev.off()
 kids<-list(rectGrob(gp=gpar(fill='white',col=NA)),g);if(label)kids<-c(kids,list(panel_letter_grob(id)))
 final<-gTree(children=do.call(gList,kids),name=paste0('panel_',id))
 attr(final,'size_mm')<-c(w=w_mm,h=h_mm);attr(final,'v4_fig')<-fig
 pd<-file.path(out,'panels');saveRDS(final,file.path(pd,paste0(id,'.rds')))
 cairo_pdf(file.path(pd,paste0(id,'.pdf')),width=w_mm/25.4,height=h_mm/25.4,family=CELL_FONT);grid.newpage();grid.draw(final);invisible(dev.off())
 png(file.path(pd,paste0(id,'.png')),width=w_mm,height=h_mm,units='mm',res=600,type='cairo',bg='white');grid.newpage();grid.draw(final);invisible(dev.off())
 if(!inherits(p,'ggplot')&&is.null(grob)){nw<-grob_words(final)-label;if(nw>25)warning(sprintf('CELL_STYLE: schematic %s has %d words (> 25)',id,nw),call.=FALSE)}
 cat(sprintf('V4 %s PANEL %s RENDERED %g x %g mm\n',fig,id,w_mm,h_mm));invisible(final)}

# ---------------------------------------------------------------- v3 compatibility names (so v3 panel scripts port with minimal edits)
BASE<-theme_cell()
FOLD_COL<-FOLD_RAMP
ARM_LABEL<-c(virtual_p='M + virtual P',row_shuffle_p='Row shuffle',observed_value_shuffle_p='Value shuffle',train_mean='Null',m_only='M only',ordinary_m='Random projection',virtual_mask='Virtual mask',measured_p='M + measured P',measured_mask='Measured mask',frozen_virtual_rna='M + virtual RNA',frozen_virtual_fm='M + frozen FM',
            m_vrna='M + virtual RNA',m_vp='M + virtual P',measured='M + measured',shuffle='Shuffle',tissue='M + tissue',null='Null')
ARM_COL<-c(virtual_p=ARM_COL_CELL[['m_vp']],row_shuffle_p=ARM_COL_CELL[['shuffle']],observed_value_shuffle_p=ARM_COL_CELL[['shuffle']],train_mean=ARM_COL_CELL[['null']],m_only=ARM_COL_CELL[['m_only']],
           ordinary_m=alpha_hex('#7F7F7F',.35),virtual_mask=alpha_hex(ARM_COL_CELL[['m_vp']],.35),measured_p=ARM_COL_CELL[['measured']],measured_mask=alpha_hex('#000000',.3),frozen_virtual_rna=ARM_COL_CELL[['m_vrna']],frozen_virtual_fm=OI[['green']],
           ARM_COL_CELL)
arm_colour<-function(...)scale_colour_manual(values=ARM_COL,labels=ARM_LABEL,name=NULL,...)
arm_fill<-function(...)scale_fill_manual(values=ARM_COL,labels=ARM_LABEL,name=NULL,...)
div_colour<-function(name=NULL,limit=NULL,horizontal=TRUE)if(is.null(limit))scale_fill_gradient2(low=GAIN_DIV[['low']],mid=GAIN_DIV[['mid']],high=GAIN_DIV[['high']],midpoint=0,name=name,na.value=MISSING,guide=guide_cell_bar(horizontal))else scale_fill_gain(limit,name=name,horizontal=horizontal)
small_legend<-theme(legend.position='bottom',legend.direction='horizontal')
# Shared legend helper: pull the legend out of one ggplot (for an unlabelled LEG_* slot under the panels it serves)
legend_only_v4<-function(p){opened<-.nullpdf();on.exit(if(opened)dev.off());g<-cowplot::get_plot_component(p,'guide-box-bottom',return_all=TRUE);if(is.list(g)&&!inherits(g,'grob'))g<-g[[1]];g}
# Typographic minus on numeric axes (Figure 1 prints U+2212, not a hyphen): scale_y_continuous(labels = num_minus)
num_minus<-scales::label_number(style_negative='minus',big.mark='')
num_minus_acc<-function(accuracy)scales::label_number(accuracy=accuracy,style_negative='minus',big.mark='')

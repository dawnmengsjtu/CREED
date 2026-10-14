# render_v4.R <fig> [ID ...]   e.g.  Rscript r/render_v4.R Figure3 A B C      (no IDs = every slot in layout.csv)
# For each ID: sources r/<fig>/<ID>.R (defines make_<ID>()), calls it with the slot size from r/<fig>/layout.csv in
# SLOT_W / SLOT_H (mm), and exports out/<fig>/panels/<ID>.{rds,pdf,png} via export_panel_v4().
# make_<ID>() returns a ggplot (built on BASE), a grob (schematic / composite), or list(plot=<ggplot>, grob=<gtable>)
# when it pre-aligned itself (e.g. with fix_panel_margins()). Optional r/<fig>/common.R is sourced once first.
# layout.csv optional columns align_v / align_h: panels sharing a non-empty group name are rebuilt together and aligned
# with cowplot (align_v: same left plot edge for stacked panels; align_h: same plot top/bottom for side-by-side panels).
source(file.path(dirname(normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))),'v4_common.R'))
args<-commandArgs(TRUE);if(!length(args))stop('usage: render_v4.R <fig> [ID ...]')
FIG<-check_fig_v4(args[1]);ensure_out_v4(FIG);LAY<-read_layout(FIG);SL<-LAY$slots;PAGE<-LAY$page
DATA<-file.path(fig_out(FIG),'plot_data');read_table<-function(n)read_pd(FIG,n)   # v3-compatible reader
ids<-if(length(args)>1)args[-1]else SL$panel
bad<-setdiff(ids,SL$panel);if(length(bad))stop('not in layout.csv: ',paste(bad,collapse=' '))
fc<-file.path(fig_rdir(FIG),'common.R');if(file.exists(fc))source(fc)
# expand to whole alignment groups
grp_members<-function(col,id){g<-SL[[col]][SL$panel==id];if(!nzchar(g))character(0)else SL$panel[SL[[col]]==g]}
build<-ids;repeat{nb<-unique(c(build,unlist(lapply(build,function(i)c(grp_members('align_v',i),grp_members('align_h',i))))));if(length(nb)==length(build))break;build<-nb}
build<-SL$panel[SL$panel%in%build]
if(length(setdiff(build,ids)))cat('render_v4: also rebuilding alignment-group members',paste(setdiff(build,ids),collapse=' '),'\n')
P<-list();G<-list()
for(id in build){f<-file.path(fig_rdir(FIG),paste0(id,'.R'));if(!file.exists(f))stop('missing panel script ',f)
 s<-SL[SL$panel==id,];SLOT_W<-s$w_mm;SLOT_H<-s$h_mm;ID<-id
 source(f,local=FALSE);fn<-paste0('make_',id);if(!exists(fn,mode='function'))stop(f,' must define ',fn,'()')
 r<-get(fn)();if(is.list(r)&&!inherits(r,'ggplot')&&!inherits(r,'grob')&&!is.null(r$plot)){P[[id]]<-r$plot;if(!is.null(r$grob))G[[id]]<-r$grob}else P[[id]]<-r}
for(col in c('align_v','align_h')){gs<-unique(SL[[col]][SL$panel%in%build&nzchar(SL[[col]])])
 for(g in gs){m<-SL$panel[SL[[col]]==g];m<-m[vapply(m,function(i)inherits(G[[i]],'gtable')||inherits(P[[i]],'ggplot'),TRUE)]
  if(length(m)<2){cat('render_v4: group',g,'has <2 alignable ggplots, skipped\n');next}
  if(col=='align_v'&&length(unique(SL$w_mm[SL$panel%in%m]))>1)warning('align_v group ',g,' mixes slot widths; only the left edge is aligned',call.=FALSE)
  if(col=='align_h'&&length(unique(SL$h_mm[SL$panel%in%m]))>1)warning('align_h group ',g,' mixes slot heights; plot tops align but bottoms will not',call.=FALSE)
  inp<-setNames(lapply(m,function(i)if(!is.null(G[[i]]))G[[i]]else P[[i]]),m);al<-align_v4(inp,align=if(col=='align_v')'v' else 'h');for(i in m)G[[i]]<-al[[i]];cat('render_v4: aligned',col,g,':',paste(m,collapse=' '),'\n')}}
for(id in build){s<-SL[SL$panel==id,];export_panel_v4(P[[id]],FIG,id,s$w_mm,s$h_mm,grob=G[[id]])}

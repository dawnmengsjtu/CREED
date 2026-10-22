# Figure 2H (2026-10-22) - the room is carried by a few function-weighted directions: measured-layer controls at r = 4
# (plot_data/lowdim_controls.csv, built by prospective_lowdim_20261022/figure_2H/build_fig2h_controls.py from
# prospective-lowdim-20261022-0001/analysis/analysis.json; values verbatim, five-fold means).
# One row per endpoint x borrowed layer: drug response (GDSC2), CRISPR gene effect, and GDSC1 drug response, the endpoint
# of the pre-registered prospective test (light band). x = share of the room (%) kept when the measured layer is projected
# on 4 directions: function-weighted (circles, layer colour), the leading principal components (triangles), a
# label-permuted metric (squares) and random orthogonal subspaces (crosses, five-seed mean), all gray but the first;
# diamonds = the virtual layer projected on the same 4 function-weighted directions. Filled = new cell lines, open =
# held-out tissue (page key in A), nudged apart. Dashed line = 100%, the full measured layer, drawn over the data rows only.
# Shape key at the top.
# No panel title. Plot top/bottom fixed (F2_BOT_TOP / F2_BOT_BOTTOM) to align with G.
make_H<-function(){
 d<-read_table('lowdim_controls')
 stopifnot(nrow(d)==60,all(is.finite(d$share_pct)),
           all(d$comparison%in%c('function_weighted','pca','permuted','random','virtual_same')))
 ROW<-c('drug|RNA'=6,'drug|P'=5,'crispr|RNA'=4,'crispr|P'=3,'gdsc1|RNA'=2,'gdsc1|P'=1)
 SH<-c(function_weighted=21,pca=24,permuted=22,random=4,virtual_same=23)
 d<-d%>%mutate(y=ROW[paste(endpoint,layer,sep='|')]+ifelse(scenario=='new_cell',.18,-.18),
   lc=ifelse(layer=='RNA',ARM_COL_CELL[['m_vrna']],ARM_COL_CELL[['m_vp']]),
   col=ifelse(comparison%in%c('function_weighted','virtual_same'),lc,F2_OTHER),
   fill=f2_fill(col,scenario),shp=SH[comparison])
 XMIN<- -10;XMAX<-150;XR<- -12;XG<- -33;XB<- -29
 rows<-tibble::tibble(y=ROW,lab=sub('.*\\|','',names(ROW)))%>%mutate(col=ifelse(lab=='RNA',ARM_COL_CELL[['m_vrna']],ARM_COL_CELL[['m_vp']]))
 grp<-tibble::tibble(lab=c('GDSC2','CRISPR','GDSC1'),top=c(6,4,2),bot=c(5,3,1))%>%mutate(y=(top+bot)/2)
 key<-tibble::tibble(comp=c('function_weighted','virtual_same','pca','permuted','random'),
   x=c(-6,64,-6,34,74),y=c(7.62,7.62,7.12,7.12,7.12),
   lab=c('Function-weighted','Virtual, same directions','PCA','Permuted','Random'))%>%mutate(shp=SH[comp])
 p<-ggplot(d)+
  annotate('rect',xmin=XMIN,xmax=XMAX,ymin=.52,ymax=2.48,fill='#F2F2F2',colour=NA)+
  annotate('segment',x=100,xend=100,y=.52,yend=6.55,linetype='22',linewidth=LW_LINE,colour=INK)+
  geom_point(aes(share_pct,y,colour=col,fill=fill,shape=shp),size=PT_SMALL+.25,stroke=.45)+
  geom_point(data=key,aes(x,y,shape=shp),colour=INK,fill='white',size=PT_SMALL+.25,stroke=.45)+
  geom_text(data=key,aes(x=x+4,y=y,label=lab),hjust=0,size=6/.pt,family=CELL_FONT)+
  geom_text(data=rows,aes(x=XR,y=y,label=lab,colour=col),hjust=1,size=6/.pt,family=CELL_FONT,fontface='bold')+
  geom_segment(data=grp,aes(x=XB,xend=XB,y=top+.35,yend=bot-.35),linewidth=LW_THIN)+
  geom_text(data=grp,aes(x=XG,y=y,label=lab),hjust=1,size=6/.pt,family=CELL_FONT)+
  scale_colour_identity()+scale_fill_identity()+scale_shape_identity()+
  scale_x_continuous(breaks=c(0,50,100,150),expand=c(0,0))+
  scale_y_continuous(expand=c(0,0))+
  coord_cartesian(xlim=c(XMIN,XMAX),ylim=c(.5,7.95),clip='off')+
  labs(x='Share of room at r = 4 (%)',y=NULL)+BASE+
  theme(axis.line.y=element_blank(),axis.text.y=element_blank(),axis.ticks.y=element_blank(),
        plot.margin=margin(5,2,1,19,'mm'))
 list(plot=p,grob=fix_panel_margins(p,top=F2_BOT_TOP,bottom=F2_BOT_BOTTOM,id=ID))}

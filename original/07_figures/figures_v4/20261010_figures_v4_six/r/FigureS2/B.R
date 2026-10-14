# FigureS2 B <- v3 Figure 6C: per-fold group means of the paired per-drug difference in standardized MSE,
# M only minus M + virtual RNA (positive = lower error with virtual RNA). Diverging scale +-0.2 chosen from the data
# (max |delta| = 0.196), so no tile is clamped. The horizontal colour bar is drawn in the shared slot LEG_AB (under B),
# so B carries no legend and its facets are as wide as A's.
make_B<-function(legend=FALSE){d<-annotation_data('annotation_effects')%>%filter(arm=='m_vrna')%>%mutate(y=group_y(group),fold=factor(paste0('F',fold),levels=paste0('F',0:4)));lim<-.2
 stopifnot(max(abs(d$delta),na.rm=TRUE)<=lim)
 ggplot(d,aes(fold,y,fill=delta))+geom_tile(colour='white',linewidth=LW_LINE,height=1,width=1)+facet_wrap(~scenario,nrow=1)+
  scale_fill_gain(lim,name='Δ sMSE',horizontal=TRUE,labels=num_minus_acc(0.1))+
  scale_x_discrete(expand=c(0,0))+scale_y_groups()+coord_groups()+labs(x='Outer fold',y=NULL)+BASE+legend_title_on()+
  guides(fill=guide_colourbar(direction='horizontal',theme=theme(legend.key.width=unit(18,'mm'),legend.key.height=unit(2,'mm'),
    legend.title.position='left',legend.title=element_text(size=PT_LEGEND,colour=INK,vjust=1,margin=margin(0,1.5,0,0,'mm')),
    legend.ticks=element_line(colour=INK,linewidth=LW_TICK),legend.frame=element_rect(colour=INK,linewidth=LW_TICK))))+
  theme(axis.line=element_blank(),axis.ticks.y=element_blank(),axis.ticks.x=element_line(colour=NA),panel.spacing=S2_FACET_GAP,
        legend.position=if(legend)'bottom' else 'none')}

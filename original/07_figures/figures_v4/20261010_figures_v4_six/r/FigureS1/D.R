# FigureS1 D (v4) <- v3 Figure 4F: the free tissue label in the new-cell-line scenario, per-fold gain in sRMSE over the
# per-drug training-mean null (points = held-out fold, F0-F4 ramp keyed under E; black bar = five-fold mean).
# Horizontal (arms as rows, shared across the two runs; facets side by side: Matrix run | Tissue-kernel run), full
# width, compact rows (about 4 mm apart). x axis 1.5-6.5 % (the null, 0 %, is stated in the legend). The black bar is
# drawn above the fold points and is taller than the fold stack (S1_BAR_D > S1_FOLD_DODGE_D stack). Left plot edge =
# S1_LEFT (shared with B above and E below).
make_D<-function(){
 d<-read_table('fig4_tissue_label')
 d$arm_label<-sub('Drug x tissue mean','Drug × tissue mean',d$arm_label)
 lv<-c('Drug × tissue mean','T only','M only','M + virtual RNA','M + T','M + T + P graph')            # top -> bottom
 yl<-c('Tissue mean','T only','M only','M + virtual RNA','M + T','M + T + P graph')
 d$arm_label<-factor(d$arm_label,levels=rev(lv))
 d$run<-factor(d$run,levels=c('matrix','tissue_kernel'),labels=c('Matrix run','Tissue-kernel run'))
 d$y<-as.integer(d$arm_label);d$y_pt<-d$y-(d$fold-2)*S1_FOLD_DODGE_D
 s<-d%>%distinct(run,arm,arm_label,y,mean_gain_vs_null_pct)
 p<-ggplot(d,aes(gain_vs_null_pct,y_pt))+
  geom_point(aes(colour=factor(fold)),size=PT_SMALL)+
  geom_segment(data=s,aes(x=mean_gain_vs_null_pct,xend=mean_gain_vs_null_pct,y=y-S1_BAR_D,yend=y+S1_BAR_D),colour=INK,linewidth=LW_LINE*1.5,inherit.aes=FALSE)+
  facet_wrap(~run,ncol=2,axes='all',axis.labels='margins')+
  scale_colour_manual(values=FOLD_RAMP,guide='none')+
  scale_y_continuous(breaks=seq_along(lv),labels=rev(yl),limits=c(.4,length(lv)+.6),expand=expansion(mult=0))+
  scale_x_continuous(limits=c(1.5,6.5),breaks=2:6,expand=expansion(mult=0))+
  labs(x='Gain in sRMSE vs null (%)',y=NULL)+
  BASE+theme(panel.spacing.x=unit(S1_FACET_GAP,'mm'))
 list(plot=p,grob=fix_panel_margins(p,left=S1_LEFT,top=S1_TOP_D,bottom=S1_BOT_D,id='D'))}

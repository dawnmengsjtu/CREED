# Figure 1C (v3): training-fitted PCA (folds 1-4, n = 328) with the fold-0 Lung hold-out (n = 93) projected.
# Metabolites: all 225 features; RNA: 1,000 highest-training-variance features. Axis % = training variance explained.
make_C <- function() {
 d<-read_table('C_pca_coordinates');d$partition<-factor(d$partition,c('Training','Held out'));d$tissue_group<-factor(d$tissue_group,TISSUE$group)
 minus_lab<-function(x)sub('^-','\u2212',as.character(x))   # true minus sign on negative ticks
 one<-function(m,strip=TRUE){z<-d[d$modality==m,];ev<-100*S$pca[[m]]$explained_variance
  ggplot(z,aes(pc1,pc2,colour=tissue_group))+geom_point(size=.5,alpha=.85,stroke=0,shape=16)+facet_wrap(~partition,nrow=1)+
   scale_colour_manual(values=TISSUE_COL,breaks=TISSUE$group,labels=TISSUE$label,drop=FALSE)+
   scale_x_continuous(labels=minus_lab)+scale_y_continuous(labels=minus_lab)+
   labs(x=sprintf('PC 1 (%.1f%%)',ev[1]),y=sprintf('%s\nPC 2 (%.1f%%)',m,ev[2]))+BASE+
   theme(legend.position='none',plot.margin=margin(if(strip)5 else 1.5,1.5,1,1.5,'mm'),strip.text=if(strip)element_text(size=PT_STRIP,face='bold',colour=INK,margin=margin(0,0,1.2,0,'mm')) else element_blank())+
   guides(colour=guide_legend(nrow=3,byrow=FALSE,override.aes=list(size=1.6,alpha=1)))}
 p1<-one('Metabolites');p2<-one('RNA',strip=FALSE)
 leg<-cowplot::get_plot_component(p1+theme(legend.position='bottom',legend.justification='left'),'guide-box-bottom')
 cowplot::ggdraw()+cowplot::draw_plot(cowplot::plot_grid(p1,p2,ncol=1,align='v',axis='lr',rel_heights=c(1.13,1)),x=0,y=.135,width=1,height=.865)+
  cowplot::draw_grob(leg,x=.04,y=.005,width=.94,height=.125)
}

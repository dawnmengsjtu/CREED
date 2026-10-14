# Figure 1G (v3): tissue means of the 20 highest-training-variance metabolites, rescaled 0-1 within each feature
# across the 22 tissues; rows = complete-linkage clustering of Euclidean distances; columns grouped by outer fold,
# ordered by decreasing tissue n. 'TDCA / TCDCA' = taurodeoxycholate/taurochenodeoxycholate; alpha/beta set in Greek.
# Source singleton tissue 'Other tissue' is labelled 'Unspecified' (distinct from the 14-tissue aggregate 'Other tissues').
# Strip row sits level with the panel letter (top margin 1.7 mm) so the right column is tight and the rotated labels end >= 2 mm above the page bottom.
# 2026-10-11 (audit A4-02): strip.clip = 'off' so the 'F0' strip label over the single Lung column is not clipped in the PNG.
make_G <- function() {
 d<-read_table('G_metabolite_tissue_means');d$tissue_label[d$tissue=='Other tissue']<-'Unspecified';features<-unique(d$feature);tissues<-unique(d$tissue)
 mat<-sapply(tissues,function(t)d$scaled_mean[d$tissue==t]);rownames(mat)<-features
 ord<-hclust(dist(mat))$order;d$feature<-factor(d$feature,rev(features[ord]))
 tx<-d%>%distinct(tissue,tissue_label,fold,n)%>%arrange(fold,desc(n),tissue);d$tissue<-factor(d$tissue,tx$tissue);d$fold<-factor(paste0('F',d$fold),paste0('F',0:4))
 labels<-setNames(tx$tissue_label,tx$tissue)
 ylab<-function(x){x<-ifelse(x=='taurodeoxycholate/taurochenodeoxycholate','TDCA / TCDCA',x);x<-sub('^alpha-','α-',x);sub('-beta-','-β-',x)}
 ggplot(d,aes(tissue,feature,fill=scaled_mean))+geom_tile(colour='white',linewidth=0.5/.pt)+facet_grid(~fold,scales='free_x',space='free_x')+
  scale_fill_share(limits=c(0,1),breaks=c(0,.5,1),name='Scaled\nmean',horizontal=FALSE)+
  scale_x_discrete(labels=labels,expand=c(0,0))+scale_y_discrete(expand=c(0,0),labels=ylab)+
  labs(x=NULL,y=NULL)+BASE+legend_title_on()+
  theme(axis.line=element_blank(),axis.ticks=element_blank(),axis.text.x=element_text(angle=90,hjust=1,vjust=.5),panel.spacing=unit(.7,'mm'),strip.clip='off',legend.position='right',legend.direction='vertical',plot.margin=margin(1.7,1,2,1,'mm'))
}

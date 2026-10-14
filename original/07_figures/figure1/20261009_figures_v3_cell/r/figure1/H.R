# Figure 1H (v3): per-cell-line observed-feature share for protein and drug AUC on the metabolite PCA of C,
# split by outer fold; one 0-100 % scale for both rows; coordinates are PCA scores, not spatial positions.
# Ramp floor raised: 0 % = pale green (35 % along the white-to-green share ramp) instead of light grey, so the protein
# row (23-66 %) reads as clearly green; limits stay 0-100 (nothing clamped).
make_H <- function() {
 d<-read_table('H_sample_coverage_pca');d$asset<-factor(d$asset,c('Protein','Drug AUC'));d$fold<-factor(paste0('F',d$fold),paste0('F',0:4))
 floor<-colorRampPalette(c(SHARE_SEQ[['low']],SHARE_SEQ[['high']]))(101)[36]
 ggplot(d,aes(pc1,pc2,colour=observed_pct))+geom_point(size=.45,stroke=0,shape=16)+facet_grid(asset~fold)+
  scale_colour_gradient(low=floor,high=SHARE_SEQ[['high']],limits=c(0,100),breaks=c(0,50,100),name='Observed (%)',
   guide=guide_colourbar(barwidth=unit(18,'mm'),barheight=unit(2,'mm'),ticks.colour=INK,ticks.linewidth=LW_TICK,frame.colour=INK,frame.linewidth=LW_TICK,direction='horizontal'))+
  labs(x=NULL,y=NULL)+BASE+legend_title_on()+
  theme(axis.line=element_blank(),axis.ticks=element_blank(),axis.text=element_blank(),axis.title.x=element_blank(),axis.title.y=element_blank(),strip.text.y=element_text(angle=0,hjust=0),
   panel.spacing=unit(1,'mm'),legend.position='bottom',legend.justification='left',legend.title.position='left',legend.title=element_text(size=PT_LEGEND,colour=INK,margin=margin(r=1.5,unit='mm')),plot.margin=margin(5,1,.5,1,'mm'))
}

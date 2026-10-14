# Figure 1E (v3): 16 highest-training-variance proteins (>= 80 % training coverage) by tissue group.
# Fill = tissue-group mean of training-standardised values (display range -1.6..1.6); area = % lines observed.
# Labels are UniProt entry-name prefixes (_HUMAN dropped), not reconciled gene symbols.
make_E <- function() {
 d<-read_table('E_protein_by_tissue');sel<-read_table('E_protein_selection');d$label<-factor(d$label,rev(sel$label));d$group<-factor(d$group,TISSUE$group)
 ggplot(d,aes(group,label))+geom_point(aes(fill=mean_z,size=observed_pct),shape=21,colour='#4D4D4D',stroke=.2)+
  scale_size_area(max_size=2.6,limits=c(0,100),breaks=c(25,50,100),name='Observed (%)')+
  scale_fill_gain(1.6,name='Mean z',horizontal=FALSE,labels=function(x)sub('^-','\u2212',as.character(x)))+
  scale_x_discrete(labels=TISSUE_LABEL)+labs(x=NULL,y=NULL)+BASE+legend_title_on()+
  theme(axis.text.x=element_text(angle=45,hjust=1,vjust=1),axis.line=element_blank(),axis.ticks=element_blank(),legend.position='right',legend.direction='vertical',legend.box='vertical',legend.box.just='left',legend.spacing.y=unit(2.5,'mm'),plot.margin=margin(5,1,1,1.5,'mm'))+
  guides(size=guide_legend(override.aes=list(fill='white',colour='#4D4D4D'),order=2))
}

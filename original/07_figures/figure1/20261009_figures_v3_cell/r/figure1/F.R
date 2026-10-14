# Figure 1F (v3): tissue -> outer-fold assignment; flow width = unique cell lines (total 421); folds are tissue-disjoint.
# Flow colours = tissue group of B/C. Only the eight named tissues of B/C/E are labelled (with counts); the 14 smaller
# tissues keep their own strata and grey flows and share one bracket 'Other (14 tissues)'; no leader lines.
# The singleton source category 'Other tissue' is displayed as 'Unspecified' (not the 14-tissue aggregate 'Other tissues').
make_F <- function() {
 d<-read_table('F_tissue_to_fold');d$tissue_label[d$tissue=='Other tissue']<-'Unspecified'
 d$tissue<-factor(d$tissue,levels=rev(d$tissue));d$fold<-factor(paste0('F',d$fold),levels=paste0('F',4:0))
 p<-ggplot(d,aes(axis1=tissue,axis2=fold,y=n))+geom_alluvium(aes(fill=group),width=.13,alpha=.75,knot.pos=.4)+geom_stratum(width=.13,fill='#F0F0F0',colour='#737373',linewidth=LW_THIN)+
  scale_fill_manual(values=TISSUE_COL,guide='none')+scale_x_discrete(limits=c('Tissue','Outer fold'),expand=c(.35,0))+scale_y_continuous(expand=expansion(mult=.01))+
  coord_cartesian(clip='off')+theme_void(base_family=CELL_FONT)+theme(plot.margin=margin(3,4.5,1.5,18,'mm'),axis.text.x=element_text(size=PT_AXIS_TITLE,colour=INK,margin=margin(t=1,unit='mm')))
 strata<-ggplot_build(p)$data[[2]];strata$tissue<-as.character(strata$stratum)
 named<-setdiff(TISSUE$group,'Other tissues');map<-setNames(d$tissue_label,as.character(d$tissue))
 left<-strata[strata$x==1,];oth<-left[!left$tissue%in%named,];left<-left[left$tissue%in%named,]
 left<-data.frame(y=left$y,label=paste0(map[left$tissue],'  ',round(left$count)))   # plain columns only (panel_data export)
 sx<-1-.13/2;bx<-sx-.03;tk<-.015;y0<-min(oth$ymin)+1.5;y1<-max(oth$ymax)-1.5   # bracket just left of the 14 small strata
 br<-data.frame(x=c(bx,bx,bx),xend=c(bx,bx+tk,bx+tk),y=c(y0,y0,y1),yend=c(y1,y0,y1))
 other<-data.frame(y=(y0+y1)/2,label=sprintf('Other (%d tissues)',nrow(oth)))
 right<-strata[strata$x==2,];right<-data.frame(y=right$y,label=paste0(right$tissue,' (',round(right$count),')'))
 p+geom_text(data=left,aes(x=sx-.02,y=y,label=label),inherit.aes=FALSE,hjust=1,size=GEOM_TEXT_MIN,family=CELL_FONT,colour=INK)+
  geom_segment(data=br,aes(x=x,y=y,xend=xend,yend=yend),inherit.aes=FALSE,colour=INK,linewidth=LW_THIN,lineend='square')+
  geom_text(data=other,aes(x=bx-.02,y=y,label=label),inherit.aes=FALSE,hjust=1,size=GEOM_TEXT_MIN,family=CELL_FONT,colour=INK)+
  geom_text(data=right,aes(x=2.1,y=y,label=label),inherit.aes=FALSE,hjust=0,size=GEOM_TEXT_MIN,family=CELL_FONT,colour=INK)
}

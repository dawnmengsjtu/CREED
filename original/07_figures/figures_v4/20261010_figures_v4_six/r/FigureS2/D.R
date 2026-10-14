# FigureS2 D <- v3 Figure 6E (rotated): share of drugs per annotation group whose five-fold mean error is lower than with
# M only (filled points, four arms) and the same share within single folds (small light-gray points). Groups are rows,
# in the order and at the heights used by A and B; the four arms are offset vertically within each row. The arm key
# serves D only and stays under it.
make_D<-function(){g<-annotation_data('drug_arm_gains');w<-annotation_data('fold_win_fraction')%>%mutate(pct=frac*100)
 off<-setNames(c(.3,.1,-.1,-.3),names(S2_ARM_LABEL))
 s<-g%>%group_by(scenario,group,arm)%>%summarise(n=n(),n_lower=sum(status=='Lower error'),pct=mean(status=='Lower error')*100,.groups='drop')%>%mutate(y=group_y(group)+off[as.character(arm)])
 w<-w%>%filter(arm%in%names(S2_ARM_LABEL))%>%mutate(y=group_y(group)+off[as.character(arm)])
 ggplot(s,aes(pct,y))+geom_vline(xintercept=50,linetype='22',linewidth=LW_THIN,colour=INK)+
  geom_point(data=w,size=.7,alpha=.8,shape=16,colour='#C8C8C8')+
  geom_point(aes(fill=arm),size=PT_SMALL,shape=21,colour='white',stroke=.2)+
  facet_wrap(~scenario,nrow=1)+
  scale_fill_manual(values=S2_ARM_COL,breaks=names(S2_ARM_LABEL),labels=unname(S2_ARM_LABEL),name=NULL)+
  scale_x_continuous(breaks=c(0,50,100))+scale_y_groups()+coord_groups(xlim=c(-4,104))+
  labs(x='Improved drugs (%)',y=NULL)+BASE+
  guides(fill=guide_legend(nrow=2,byrow=TRUE,override.aes=list(size=1.6,shape=21,colour='white',stroke=.2)))+
  theme(axis.line.y=element_blank(),axis.ticks.y=element_blank(),panel.spacing=S2_FACET_GAP,legend.box.spacing=unit(1,'mm'),legend.key.size=unit(2.6,'mm'))}

# S5 F (v1 5F): mean patient-specific gating weight given to each of the two source-cohort experts.
# All weights are drawn (coordinate clipping, no scale limit that drops stacked segments); weights > 8% are printed
# as rounded percentages.
make_F<-function(){d<-cp_read('expert_weights')%>%mutate(expert=factor(expert,levels=CP_COHORTS))
 ggplot(d,aes(weight*100,cohort,fill=expert))+
  geom_col(width=.6,colour='white',linewidth=.5/.pt,position=position_stack(reverse=TRUE))+
  geom_text(aes(label=ifelse(weight>.08,sprintf('%.0f',weight*100),'')),position=position_stack(vjust=.5,reverse=TRUE),size=GEOM_TEXT_MIN,family=CELL_FONT,colour=INK)+
  facet_grid(direction~.,labeller=cp_dir_labeller)+
  scale_fill_manual(values=CP_COHORT_COL,name='Source expert',drop=FALSE)+
  scale_y_discrete(limits=rev(CP_COHORTS))+scale_x_continuous(breaks=c(0,50,100),expand=c(0,0))+coord_cartesian(xlim=c(0,100),clip='off')+
  labs(x='Mean gating weight (%)',y='Held-out cohort')+BASE+CP_STRIP+legend_title_on()+
  CP_LEGEND_TOP+theme(legend.title=element_text(size=PT_LEGEND,margin=margin(r=1.5,unit='mm')))}

# S5 B (v1 5B): paired cohort-level standardized RMSE before (hollow ring) and after (filled dot) source-only shift
# correction, for the context, uniform-expert and gated-expert predictors; colour = target layer of the direction.
# The ring is 1.5x the dot and is drawn after it (data order within the layer), so when the two values coincide the
# result is a dot inside a ring: both marks are always visible, and 'unchanged' differs from 'missing'.
CP_PT_BEFORE<-PT_LARGE*1.5;CP_PT_AFTER<-PT_LARGE
make_B<-function(){d<-cp_read('summary')%>%filter(arm%in%CP_THREE)%>%mutate(arm=factor(arm,levels=rev(CP_THREE)))
 long<-d%>%select(direction,cohort,arm,before_rmse,rmse)%>%pivot_longer(c(before_rmse,rmse),names_to='version',values_to='value')%>%
  mutate(version=factor(version,levels=c('before_rmse','rmse')))%>%arrange(desc(version))   # corrected dots first, rings on top
 ggplot()+geom_segment(data=d,aes(x=before_rmse,xend=rmse,y=arm,yend=arm),colour='#BDBDBD',linewidth=LW_LINE)+
  geom_point(data=long,aes(value,arm,shape=version,size=version,colour=direction),stroke=.45)+
  cp_facet()+cp_srmse_x()+scale_y_discrete(labels=CP_PRED_LABEL)+
  scale_shape_manual(values=c(before_rmse=1,rmse=16),labels=c(before_rmse='Before correction',rmse='Corrected'),name=NULL)+
  scale_size_manual(values=c(before_rmse=CP_PT_BEFORE,rmse=CP_PT_AFTER),labels=c(before_rmse='Before correction',rmse='Corrected'),name=NULL)+
  scale_colour_manual(values=CP_DIR_COL,guide='none')+guides(shape=guide_legend(override.aes=list(colour=INK)))+
  labs(x="Standardized RMSE",y=NULL)+BASE+CP_STRIP+CP_LEGEND_TOP+theme(panel.spacing.x=unit(2.5,"mm"))}

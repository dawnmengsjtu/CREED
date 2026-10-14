# S5 D (v1 5D): per-patient standardized RMSE (observed target entries only) for the three predictors in B,
# log axis shared with B (same column ranges, same mm per decade in every column).
# Violins (width-scaled) with a black interquartile bar and a white median notch.
# UCEC protein -> RNA: the per-patient spread (8.78-9.03) is narrower than a line on this axis, so the violin would be a
# hairline; there the patients are drawn as vertically jittered points (true x, jitter in y only) with a black median tick
# drawn under the points and longer than the jitter band, so it shows above and below the strip.
cp_tight<-function(d)d$cohort=='UCEC'&d$direction=='protein_to_rna'
make_D<-function(){d<-cp_read('patient_metrics')%>%filter(arm%in%CP_THREE)%>%mutate(arm=factor(arm,levels=rev(CP_THREE)))
 dv<-d[!cp_tight(d),];dp<-d[cp_tight(d),]
 ggplot(d,aes(rmse,arm))+
  geom_violin(data=dv,aes(fill=direction,colour=direction),scale='width',width=.85,linewidth=LW_THIN)+
  stat_summary(data=dp,geom='errorbar',orientation='y',fun.min=median,fun.max=median,fun=median,colour=INK,linewidth=.3,width=.84)+
  geom_point(data=dp,aes(colour=direction),position=position_jitter(width=0,height=.24,seed=11),shape=16,size=.55,alpha=.8)+
  cp_iqr_median(dv)+
  cp_facet()+cp_srmse_x()+scale_y_discrete(labels=CP_PRED_LABEL)+
  scale_fill_manual(values=sapply(CP_DIR_COL,alpha_hex,.35),guide='none')+scale_colour_manual(values=CP_DIR_COL,guide='none')+
  labs(x='Per-patient standardized RMSE',y=NULL)+BASE+CP_STRIP+theme(panel.spacing.x=unit(2.5,'mm'))}

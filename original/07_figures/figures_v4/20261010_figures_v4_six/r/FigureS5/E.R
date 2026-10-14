# S5 E (v1 5E): per-target Pearson r (targets with >= 5 observed pairs and non-constant truth and prediction) for the
# three predictors in B; dashed line at r = 0. Same summary glyph as D (black IQR bar, white median notch).
# The axis runs 0.06 beyond +/-1 so the most extreme tails (r = -0.984) do not touch the axis line.
make_E<-function(){d<-cp_read('feature_correlations')%>%filter(arm%in%CP_THREE,!is.na(r))%>%mutate(arm=factor(arm,levels=rev(CP_THREE)))
 ggplot(d,aes(r,arm))+geom_vline(xintercept=0,linetype='22',linewidth=LW_THIN,colour='#7F7F7F')+
  geom_violin(aes(fill=direction,colour=direction),scale='width',width=.85,linewidth=LW_THIN)+
  cp_iqr_median()+
  facet_grid(direction~cohort,labeller=cp_dir_labeller)+
  scale_x_continuous(limits=c(-1,1),breaks=c(-0.5,0,0.5),labels=c('−0.5','0','0.5'),expand=expansion(add=.06))+scale_y_discrete(labels=CP_PRED_LABEL)+
  scale_fill_manual(values=sapply(CP_DIR_COL,alpha_hex,.35),guide='none')+scale_colour_manual(values=CP_DIR_COL,guide='none')+
  labs(x='Per-target Pearson r',y=NULL)+BASE+CP_STRIP+theme(panel.spacing.x=unit(2.5,'mm'))}

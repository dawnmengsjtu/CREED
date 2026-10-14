# FigureS1 LEG_ARM (v4, unlettered): the page's single arm key, one row, directly under B and C. It serves A
# (binned observed vs predicted AUC) and B/C (per-fold sRMSE); the Null key (dashed line) applies to B and C.
make_LEG_ARM<-function(){
 d<-s1_srmse_data('new_cell')
 p<-ggplot(d,aes(x,score,colour=arm,shape=arm,linetype=arm,group=arm))+geom_line(linewidth=LW_LINE)+geom_point(data=~filter(.x,arm!='null'),size=S1_PT_KEY,stroke=.3)+
  s1_arm_scales()+BASE+theme(legend.position='bottom',legend.key.width=unit(6,'mm'),legend.key.spacing.x=unit(4,'mm'))+
  guides(colour=guide_legend(nrow=1),shape=guide_legend(nrow=1),linetype=guide_legend(nrow=1))
 cell_guard(p,'LEG_ARM');legend_only_v4(p)}

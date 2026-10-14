# Figure S4C (v4 fix 1): least-squares slope of the gain over the three nested generator sizes, mean +- SD over the five
# outer folds (a_slopes: gain_slope_pp_per100 +- gain_slope_sd_pp_per100), per scenario; colour = virtual layer (LEG_AB).
make_C<-function(){
 s<-s4_prep(read_table('a_slopes'));m<-s$gain_slope_pp_per100;sd<-s$gain_slope_sd_pp_per100
 stopifnot(max(m+sd)<.42,min(m-sd)>0)
 s$xi<-as.integer(s$scenario)+ifelse(s$layer=='RNA',-.18,.18)
 p<-ggplot(s,aes(xi,gain_slope_pp_per100,colour=arm))+
  geom_err(aes(ymin=gain_slope_pp_per100-gain_slope_sd_pp_per100,ymax=gain_slope_pp_per100+gain_slope_sd_pp_per100))+
  geom_point(size=PT_LARGE,shape=16)+
  scale_colour_manual(values=ARM_COL_CELL[unname(S4_ARM)],labels=ARM_LABEL_CELL[unname(S4_ARM)],name=NULL)+
  scale_x_continuous(breaks=1:2,labels=c('New\ncell lines','Held-out\ntissue'))+
  scale_y_continuous(breaks=seq(0,.4,.1),labels=num_minus_acc(.1))+
  coord_cartesian(xlim=c(.45,2.55),ylim=c(0,.42),expand=FALSE)+
  labs(x=NULL,y='Gain slope (% per 100 lines)')+BASE+theme(legend.position='none',axis.text.x=element_text(lineheight=.95))
 s4_fix(p,top=S4_TOP2,left=S4_LEFT)}

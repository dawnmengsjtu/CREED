# Figure 3H (<- v4 3G) - grouped dot plot: recall of each cell line's observed top-k most sensitive drugs among its observed
# candidates (ranking_recall.csv), averaged over cell lines within a fold and then over the five folds (v3 aggregation);
# every arm ranks the same candidates. Within each k the arms sit side by side in a fixed order (M only, M + virtual RNA,
# M + virtual P, M + measured RNA, M + measured P; colours and symbols keyed in A); dashed black segment = training-mean
# null (five-fold mean). A light range bar behind each k runs from the null to the better of the two measured references
# (a re-arrangement of the plotted five-fold means, no new statistic). Scenarios as rows with top strips (as B, E, F).
# Fix round 2 (2026-10-10): the five per-fold mean recalls per arm and k (the intermediate step of the aggregation:
# mean over cell lines within a fold) are drawn as small light points in the arm colour and symbol, spread horizontally
# around each five-fold mean (fold order F0-F4, left to right, 0.035 group widths apart). The y range now holds every
# per-fold value, so the held-out-tissue y break of the previous round is gone; the new-cell-line facet starts at 0.565 so
# the k = 10 null and M only marks (0.596-0.599) sit clear of the axis line.
make_H<-function(){arms<-c('M','M+vRNA','M+vP','M+RNA','M+P','null_drug')
 d<-read_pd(FIG,'ranking_recall')%>%group_by(scenario,fold,arm,k)%>%summarise(recall=mean(recall),.groups='drop')
 m<-d%>%group_by(scenario,arm,k)%>%summarise(recall=mean(recall),folds=n(),.groups='drop')%>%mutate(kx=match(k,sort(unique(k))))
 stopifnot(all(m$folds==5),nrow(m)==48,nrow(d)==240)
 off<-setNames(c(seq(-.3,.3,length.out=5),0),arms);m$x<-m$kx+off[m$arm];W<-.37
 FJ<-.035   # per-fold horizontal spread (group widths)
 d<-d%>%mutate(kx=match(k,sort(unique(k))),x=kx+off[arm]+(fold-2)*FJ)
 write_csv(d%>%select(scenario,fold,arm,k,fold_mean_recall=recall),file.path(fig_out(FIG),'panel_data','H_fold_means.csv'))
 rb<-m%>%group_by(scenario,kx)%>%summarise(lo=recall[arm=='null_drug'],hi=max(recall[arm%in%c('M+RNA','M+P')]),.groups='drop')
 write_csv(rb,file.path(fig_out(FIG),'panel_data','H_range_bars.csv'))
 fp<-filter(d,arm!='null_drug')
 rngs<-fp%>%group_by(scenario)%>%summarise(lo=min(recall),hi=max(recall));print(rngs)
 lim<-data.frame(scenario=c('new_cell','new_cell','tissue','tissue'),x=c(.6,4.4,.6,4.4),y=c(.565,.735,.530,.712))
 stopifnot(all(fp$recall>ifelse(fp$scenario=='new_cell',.565,.530)),all(fp$recall<ifelse(fp$scenario=='new_cell',.735,.712)))
 sc<-function(z)factor(F3_SC[z],levels=F3_SC);m$scenario<-sc(m$scenario);rb$scenario<-sc(rb$scenario);lim$scenario<-sc(lim$scenario);fp$scenario<-sc(fp$scenario)
 pts<-filter(m,arm!='null_drug');nul<-filter(m,arm=='null_drug')
 p<-ggplot(pts,aes(x,recall))+geom_blank(data=lim,aes(x,y),inherit.aes=FALSE)+
  geom_rect(data=rb,aes(xmin=kx-W,xmax=kx+W,ymin=lo,ymax=hi),fill=F3_RANGE_FILL,colour=NA,inherit.aes=FALSE)+
  geom_segment(data=nul,aes(x=kx-W,xend=kx+W,y=recall,yend=recall),linetype='22',linewidth=LW_LINE,colour=INK)+
  geom_point(data=fp,aes(colour=arm,shape=arm),size=.55,alpha=.45,stroke=0)+
  geom_point(aes(colour=arm,shape=arm),size=1.3)+
  facet_wrap(~scenario,ncol=1,scales='free_y')+
  scale_colour_manual(values=F3_COL,guide='none')+scale_shape_manual(values=F3_SHAPE,guide='none')+
  scale_x_continuous(breaks=1:4,labels=sort(unique(m$k)),expand=c(0,0))+
  scale_y_continuous(breaks=c(.55,.60,.65,.70),labels=function(b)formatC(b,format='f',digits=2),expand=c(0,0))+
  labs(x='k (top-k drugs)',y='Top-k recall')+BASE+
  theme(panel.spacing.y=unit(2.2,'mm'),plot.margin=margin(1,1.2,.6,.6,'mm'))
 cell_guard(p,'H')
 list(plot=p,grob=fix_panel_margins(p,top=F3_TOP,bottom=8.2,id='H'))}

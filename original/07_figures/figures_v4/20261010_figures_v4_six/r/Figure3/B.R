# Figure 3B (<- v4 3E) - lollipop: comparator - virtual arm difference in fold sRMSE (fig4_controls.csv delta; > 0 = the
# virtual arm has the lower error). Within a comparator the upper lollipop is M + virtual RNA (blue) and the lower
# M + virtual P (pink). Stem from 0 to the five-fold mean, large dot = five-fold mean, small light dots = single folds
# (arm colour, no fold ramp). Scenarios as facets.
# Fix round 2 (2026-10-10): rows top to bottom = Shuffle, M only, M + measured (same layer), T only, M + T.
#   - Only like-for-like measured references are kept: virtual RNA against measured RNA, virtual P against measured P,
#     merged into one 'M + measured (same layer)' row (the cross-layer pairs vP-vs-measured-RNA and vRNA-vs-measured-P
#     in fig4_controls.csv are no longer drawn).
#   - T only and M + T (not run in held-out tissue: tissue unseen in training) are the last two rows, so the #F0F0F0
#     not-run shade is one trailing strip at the bottom of the held-out facet (explained in the legend; no in-panel text).
make_B<-function(){
 d<-read_pd(FIG,'fig4_controls')
 stopifnot(all(d$comparator%in%c('shuffle','m_only','tissue_label','m_plus_t','measured_rna','measured_p')))
 # keep only the matched measured pair for each virtual layer
 d<-d%>%filter(!(comparator=='measured_rna'&virtual_layer!='vRNA'),!(comparator=='measured_p'&virtual_layer!='vP'))%>%
  mutate(row=ifelse(comparator%in%c('measured_rna','measured_p'),'measured_same',comparator))
 lv<-c('shuffle','m_only','measured_same','tissue_label','m_plus_t')
 lab<-c(shuffle='Shuffle',m_only='M only',measured_same='M + measured\n(same layer)',tissue_label='T only',m_plus_t='M + T')
 d$arm<-ifelse(d$virtual_layer=='vRNA','M+vRNA','M+vP')
 d$y<-(length(lv)+1-match(d$row,lv))+ifelse(d$arm=='M+vRNA',.2,-.2)
 m<-d%>%group_by(scenario,arm,row,comparator,y)%>%summarise(delta=mean(delta),n=n(),.groups='drop');stopifnot(all(m$n==5))
 stopifnot(nrow(filter(m,row=='measured_same'))==4,all(with(filter(m,row=='measured_same'),(arm=='M+vRNA')==(comparator=='measured_rna'))))
 write_csv(m%>%select(scenario,row,comparator,arm,fivefold_mean_delta=delta),file.path(fig_out(FIG),'panel_data','B_fivefold_means.csv'))
 run<-d%>%distinct(scenario,row)
 nr<-expand.grid(scenario=names(F3_SC),row=lv,stringsAsFactors=FALSE)%>%anti_join(run,by=c('scenario','row'))%>%mutate(y=length(lv)+1-match(row,lv))
 stopifnot(nrow(nr)==2,all(nr$scenario=='tissue'),setequal(nr$y,1:2))   # trailing strip: the two bottom rows
 sc<-function(z)factor(F3_SC[z],levels=F3_SC);d$scenario<-sc(d$scenario);m$scenario<-sc(m$scenario);nr$scenario<-sc(nr$scenario)
 p<-ggplot(d,aes(delta,y))+
  geom_rect(data=nr,aes(xmin=-Inf,xmax=Inf,ymin=y-.5,ymax=y+.5),fill=F3_NOTRUN,colour=NA,inherit.aes=FALSE)+
  geom_vline(xintercept=0,linewidth=LW_THIN,colour=F3_ZERO)+
  geom_segment(data=m,aes(x=0,xend=delta,y=y,yend=y,colour=arm),linewidth=LW_LINE*1.6)+
  geom_point(aes(colour=arm),size=.75,shape=16,alpha=.45)+
  geom_point(data=m,aes(colour=arm),size=PT_LARGE,shape=16)+
  facet_wrap(~scenario,ncol=2,axes='all',axis.labels='margins')+
  scale_colour_manual(values=F3_COL,guide='none')+
  scale_y_continuous(breaks=length(lv):1,labels=unname(lab[lv]),limits=c(.5,length(lv)+.5),expand=c(0,0))+   # rows fill the panel; the not-run strip reaches the x axis
  scale_x_continuous(breaks=c(-.02,0,.02),labels=num_minus_acc(0.01),expand=expansion(mult=.08))+
  labs(x='Δ fold sRMSE (comparator − virtual)',y=NULL)+
  BASE+theme(axis.ticks.y=element_blank(),axis.text.y=element_text(lineheight=.9),panel.spacing=unit(3,'mm'),plot.margin=margin(1,1.2,.8,.6,'mm'))
 cell_guard(p,'B')
 list(plot=p,grob=fix_panel_margins(p,top=F3_TOP,id='B'))}

# Figure 6D (old v4-flex E; fix round 2, 2026-10-10) - every CRISPR arm of the sealed run (95 lines with a DepMap 24Q2
# screen, 2,456 gene-effect targets). Bars start at 0 (gain over M only is zero for M). Bar = sealed gain; its 95 %
# grouped-bootstrap interval is drawn in white inside the bar (lower bound -> estimate) and in the bar colour above it
# (estimate -> upper bound). Open circle (white fill, group-colour outline, as in B) on every bar that has a
# new-cell-line (development-fold) value: M + virtual RNA / P (forest.csv dev_gain_pct) and M + measured RNA / P
# (b_share.csv dev_room_pct, the development room = gain of M + measured RNA / P; their sealed room_pct is asserted
# equal to the forest.csv gain of the same bars).
# Random panels: 35 % fill, no interval, the five seeds as small dots. Paired shuffles: dots with intervals near 0
# (bars would be slivers) and an inset with an expanded y axis around 0 (dotted call-out), which shows that only
# M + shuffled virtual P has its interval entirely below 0. Groups: virtual layers, paired shuffles, protein panels,
# gene panels, measured layers (shaded). The endpoint is named in the y-axis title. Source: forest.csv /
# forest_seeds.csv (endpoint crispr), verbatim.
make_D<-function(){
 d<-f6_forest_rows('crispr');stopifnot(nrow(d)==17)
 tick<-c('M+vRNA'='RNA','M+vP'='P','M+vRNA_shuf'='RNA','M+vP_shuf'='P','M+P16'='16','M+P32'='32','M+P64'='64','M+Prand16_mean'='16','M+Prand32_mean'='32',
         'M+G16'='16','M+G32'='32','M+G64'='64','M+Grand16_mean'='16','M+Grand32_mean'='32','M+RNA'='RNA','M+P'='P','M+Pval'='P values')   # final polish 2026-10-11: was 'P (no mask)'
 d$sub<-ifelse(d$group=='panel',paste0('panel_',d$layer),d$group)
 ord<-c('virtual','shuffle','panel_protein','panel_gene','reference')
 d<-d%>%mutate(sub=factor(sub,levels=ord),rnd=kind=='random_mean')%>%arrange(sub,rnd,row_order)
 gap<-.8;d$x<-seq_len(nrow(d))+gap*(as.integer(d$sub)-1)
 d$x[d$arm=='M+Pval']<-d$x[d$arm=='M+Pval']+.8          # a little air for the one-line 'P values' label
 d$tick<-unname(tick[d$arm]);stopifnot(!any(is.na(d$tick)))
 d$edge<-F6_COL[sub('_random','',d$colkey)];d$fill<-F6_COL[d$colkey]
 sh<-d[d$group=='shuffle',];bars<-d[d$group!='shuffle',];bi<-bars[!bars$rnd,]
 sd<-read_table('forest_seeds')%>%filter(endpoint=='crispr');sd$x<-d$x[match(sd$mean_arm,d$arm)];sd$col<-d$edge[match(sd$mean_arm,d$arm)];stopifnot(nrow(sd)==20)
 dv<-d[!is.na(d$dev_gain_pct),c('arm','x','edge','dev_gain_pct','gain_pct')];stopifnot(nrow(dv)==2)
 bs<-read_table('b_share')%>%filter(endpoint=='crispr');ma<-ifelse(bs$layer=='RNA','M+RNA','M+P');mi<-match(ma,d$arm)
 stopifnot(length(mi)==2,!any(is.na(mi)),all(abs(bs$room_pct-d$gain_pct[mi])<1e-9))
 dv<-rbind(dv,data.frame(arm=ma,x=d$x[mi],edge=d$edge[mi],dev_gain_pct=bs$dev_room_pct,gain_pct=d$gain_pct[mi]))
 write_csv(dv,file.path(fig_out(FIG),'panel_data','D_dev_circles_display.csv'))
 gl<-d%>%group_by(sub)%>%summarise(x0=min(x),x1=max(x),x=(min(x)+max(x))/2,.groups='drop')%>%
  mutate(lab=c(virtual='Virtual',shuffle='Shuffled',panel_protein='Protein panels',panel_gene='Gene panels',reference='Measured')[as.character(sub)])
 rb<-d%>%filter(rnd)%>%group_by(sub)%>%summarise(x0=min(x)-.35,x1=max(x)+.35,.groups='drop')
 ylim<-c(-.3,3.0);xr<-c(.35,max(d$x)+1.45)
 yt<- -.5;yb<- -.86;yg<- -1.2            # tick labels, random bracket, group labels (data units below the axis, clip off)
 ref<-gl[gl$sub=='reference',]
 # inset: the two paired shuffles on an expanded y axis around 0
 shi<-sh%>%mutate(xi=c(1,2))
 ins<-ggplot(shi)+
  geom_hline(yintercept=0,colour=INK,linewidth=LW_THIN)+
  geom_errorbar(aes(x=xi,ymin=lo,ymax=hi),width=.16,linewidth=.55/.pt*1.3,colour=F6_SHUF)+
  geom_point(aes(xi,gain_pct),shape=16,size=1.7,colour=F6_SHUF)+
  geom_text(aes(xi,.075,label=tick),size=GEOM_TEXT_MIN,vjust=0,colour=INK)+
  scale_y_continuous(breaks=c(-.2,-.1,0),labels=num_minus_acc(.1))+scale_x_continuous(breaks=NULL)+
  coord_cartesian(xlim=c(.45,2.55),ylim=c(-.21,.12),expand=FALSE,clip='off')+
  labs(x=NULL,y=NULL)+
  BASE+theme(axis.line.x=element_blank(),axis.ticks.x=element_blank(),plot.margin=margin(0,0,0,0,'mm'),
             panel.border=element_rect(fill=NA,colour=F6_RULE,linewidth=LW_THIN))
 ig<-ggplotGrob(ins)
 ix<-c(.9,5.25);iy<-c(1.28,2.86)               # inset placement (main data units, includes its y tick labels)
 cx<-c(min(sh$x)-.42,max(sh$x)+.42);cy<-c(-.27,.16)  # call-out box around the shuffles in the main plot
 p<-ggplot(d)+
  annotate('rect',xmin=ref$x0-.6,xmax=xr[2],ymin=ylim[1],ymax=ylim[2],fill=F6_BAND,colour=NA)+
  geom_hline(yintercept=0,colour=INK,linewidth=LW_AXIS)+
  geom_col(data=bars,aes(x,gain_pct),fill=bars$fill,colour=bars$edge,width=.72,linewidth=.35)+
  geom_segment(data=bi,aes(x=x,xend=x,y=lo,yend=gain_pct),colour='white',linewidth=LW_ERR*1.3)+
  geom_segment(data=bi,aes(x=x,xend=x,y=gain_pct,yend=hi),colour=bi$edge,linewidth=LW_ERR*1.3)+
  geom_point(data=sd,aes(x,gain_pct),colour=sd$col,size=.7,shape=16)+
  geom_point(data=dv,aes(x,dev_gain_pct),shape=21,fill='white',colour='white',size=1.55,stroke=1.15)+
  geom_point(data=dv,aes(x,dev_gain_pct),shape=21,fill='white',colour=dv$edge,size=1.55,stroke=.6)+
  geom_errorbar(data=sh,aes(x=x,ymin=lo,ymax=hi),width=0,linewidth=LW_ERR,colour=F6_SHUF)+
  geom_point(data=sh,aes(x,gain_pct),shape=16,size=1.3,colour=F6_SHUF)+
  annotate('rect',xmin=cx[1],xmax=cx[2],ymin=cy[1],ymax=cy[2],fill=NA,colour='#9E9E9E',linewidth=LW_THIN,linetype='11')+
  annotate('segment',x=cx,xend=c(ix[1]+.95,ix[2]),y=cy[2],yend=iy[1]+.02,colour='#9E9E9E',linewidth=LW_THIN,linetype='11')+
  annotation_custom(ig,xmin=ix[1],xmax=ix[2],ymin=iy[1],ymax=iy[2])+
  geom_text(aes(x,yt,label=tick,hjust=ifelse(arm=='M+Pval',.47,.5)),size=GEOM_TEXT_MIN,vjust=1,colour=INK)+
  geom_segment(data=rb,aes(x=x0,xend=x1,y=yb+.06,yend=yb+.06),colour='#7F7F7F',linewidth=LW_THIN)+
  geom_text(data=rb,aes((x0+x1)/2,yb,label='random'),size=GEOM_TEXT_MIN,vjust=1,colour='#4D4D4D')+
  geom_text(data=gl,aes(x,yg,label=lab),size=F6_GROUP_PT/.pt,fontface='bold',vjust=1,colour=INK)+
  scale_y_continuous(breaks=0:3,labels=num_minus)+scale_x_continuous(breaks=NULL)+
  coord_cartesian(xlim=xr,ylim=ylim,expand=FALSE,clip='off')+
  labs(x=NULL,y='Gain over M only (%), CRISPR')+
  BASE+theme(axis.line.x=element_blank(),axis.ticks.x=element_blank(),legend.position='none',plot.margin=margin(5,.5,1,1.5,'mm'))
 f6_fix(p,left=12,top=F6_TOP,bottom=12.5)}

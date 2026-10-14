# Figure 5B - absolute CRISPR sRMSE per fold (the CRISPR counterpart of Figure 2B): null (per-gene fit-set mean), M only,
# M + virtual RNA (deployable from metabolites alone) and M + measured RNA (reference, not deployable); one facet per
# scenario. Arm colours: null #BDBDBD (baseline gray, fix round; was black), M only #7F7F7F, M + virtual RNA #0072B2, M +
# measured RNA #4D4D4D. Points = folds (arm colour; filled = new cell lines, open = held-out tissue), joined per fold in one light gray
# and labelled directly at the right end (F0-F4). Crossbars = five-fold means (d_abs_mean srmse_mean) in the arm colour.
# Held-out tissue: fold F1 (haematopoietic and lymphoid lines, 1.13-1.15 for every arm) sits in a strip above a broken axis
# at the same sRMSE-per-mm scale (display y = 1.045 + (sRMSE - 1.125) above the break); the means include it.
# Final polish (2026-10-11): two-line arm labels ('M + virtual\nRNA', 'M + measured\nRNA', was three lines). The arms sit
# at x = 1, 2, 3 and 4.25: a deliberate extra gap before the non-deployable measured reference (the gap grammar of
# Figure 3A), which is also what lets the two-line labels clear each other at the 6 pt house minimum. Crossbars +-0.22
# (were +-0.30); fold labels 0.45 right of the measured arm (were 0.30) and spread at least 0.0095 sRMSE apart (were
# 0.0085), so no label touches a mark; x padding, facet spacing and side margins tightened to give the facets the width.
make_B<-function(){
 lv<-c('null','M','M+vRNA','M+RNA');lab<-c('Null','M only','M + virtual\nRNA','M + measured\nRNA')
 XP<-c(1,2,3,4.25);CB<-.22;XL<-XP[4]+.45;X0<-.7;X1<-XP[4]+.27
 BRK<-1.035;UP0<-1.125;UPY<-1.045
 tr<-function(v)ifelse(v>1.09,UPY+(v-UP0),v);itr<-function(v)ifelse(v>BRK+.002,v-UPY+UP0,v)
 col<-c(null=F5_NULL,M=LAYER_COL[['M']],`M+vRNA`=LAYER_COL[['RNA']],`M+RNA`=F5_MEAS)
 d<-read_table('d_abs_fold')%>%filter(arm%in%lv)%>%mutate(a=XP[match(arm,lv)],sc=factor(F5_SC[scenario],levels=F5_SC),
   x=a+(fold-2)*.07,y=tr(srmse),col=col[arm],fill=f5_fill(col,scenario))
 m<-read_table('d_abs_mean')%>%filter(arm%in%lv)%>%mutate(a=XP[match(arm,lv)],sc=factor(F5_SC[scenario],levels=F5_SC),col=col[arm])
 chk<-d%>%group_by(scenario,arm)%>%summarise(v=mean(srmse),.groups='drop')%>%inner_join(m,by=c('scenario','arm'))
 stopifnot(nrow(d)==40,nrow(m)==8,max(abs(chk$v-chk$srmse_mean))<1e-9,all(m$srmse_mean<BRK),
           all((d$srmse>1.09)==(d$scenario=='tissue'&d$fold==1)),all(d$srmse<1.17),all(d$srmse>.92))
 fl<-d%>%filter(arm=='M+RNA')%>%mutate(lab=paste0('F',fold),ly=y)
 for(s in levels(fl$sc)){k<-fl$sc==s&fl$y<BRK;fl$ly[k]<-f5_spread(fl$y[k],.0095)}
 lim<-tibble::tibble(sc=factor(F5_SC,levels=F5_SC),x=XP[1],y=c(.935,.915),y2=c(1.04,1.075))
 bm<-tibble::tibble(sc=factor(F5_SC[['tissue']],levels=F5_SC))
 ggplot(d,aes(x,y))+
  geom_blank(data=lim,aes(x,y))+geom_blank(data=lim,aes(x,y2))+
  geom_line(aes(group=interaction(sc,fold)),colour=F5_FOLDLINE,linewidth=LW_THIN)+
  geom_segment(data=m,aes(x=a-CB,xend=a+CB,y=srmse_mean,yend=srmse_mean,colour=col),linewidth=LW_LINE*2.6,lineend='butt')+
  geom_point(aes(colour=col,fill=fill),shape=24,size=PT_SMALL+.3,stroke=.45)+
  geom_text(data=fl,aes(x=XL,y=ly,label=lab),hjust=0,size=6/.pt,family=CELL_FONT,colour=F5_FOLDLAB)+
  geom_segment(data=bm,aes(x=X0-.07,xend=X0+.07,y=BRK,yend=BRK+.003),inherit.aes=FALSE,linewidth=LW_AXIS)+
  geom_segment(data=bm,aes(x=X0-.07,xend=X0+.07,y=UPY-.003,yend=UPY),inherit.aes=FALSE,linewidth=LW_AXIS)+
  geom_rect(data=bm,aes(xmin=X0-.03,xmax=X0+.03,ymin=BRK+.0018,ymax=UPY-.0018),inherit.aes=FALSE,fill='white',colour=NA)+
  facet_wrap(~sc,scales='free_y')+scale_colour_identity()+scale_fill_identity()+
  scale_x_continuous(breaks=XP,labels=lab,expand=c(0,0))+
  scale_y_continuous(breaks=function(l)if(l[2]>1.06)c(.94,.97,1,1.03,tr(1.14))else c(.94,.97,1,1.03),
                     labels=function(b)formatC(itr(b),format='f',digits=2),expand=c(0,0))+
  coord_cartesian(xlim=c(X0,X1),clip='off')+
  labs(x=NULL,y='CRISPR sRMSE')+BASE+
  theme(panel.spacing.x=unit(6.2,'mm'),plot.margin=margin(5,4.6,1,.8,'mm'),axis.text.x=element_text(margin=margin(t=.8,unit='mm'),lineheight=.95))}

# Figure 5E - the absolute room shrinks on CRISPR: bars from zero = five-fold-mean room of the four metabolite-row cells
# (a_map room_gain), GDSC and CRISPR side by side, M->RNA (blue) and M->P (pink) within each endpoint; one facet per scenario
# (bars filled = new cell lines, open = held-out tissue). Points = per-fold rooms (b_share_fold room_fold) in light gray with
# the endpoint glyph. The CRISPR/GDSC room ratios (a_map room_ratio_crispr_over_gdsc) are printed as a two-line stack centred
# over the CRISPR bar pair, M->RNA (blue) above M->P (pink), clear of the per-fold triangles. Bars stay (not dot + crossbar)
# so this panel keeps a form of its own: B is the page's dot + crossbar panel. y axis ends at 3.62 (largest per-fold room
# 3.59 %, GDSC M->RNA new cell lines F2).
make_E<-function(){
 m<-read_table('a_map')%>%filter(deployed=='M')%>%transmute(endpoint,scenario,pair,room=room_gain,ratio=room_ratio_crispr_over_gdsc)
 f<-read_table('b_share_fold')%>%transmute(endpoint,scenario,pair,fold,room=room_fold)
 dx<-c(`M→RNA`=-.2,`M→P`=.2)
 m<-m%>%mutate(sc=factor(F5_SC[scenario],levels=F5_SC),x=match(endpoint,F5_EP)+dx[pair],col=F5_PAIR_COL[pair],fill=f5_fill(col,scenario))
 f<-f%>%mutate(sc=factor(F5_SC[scenario],levels=F5_SC),x=match(endpoint,F5_EP)+dx[pair]+(fold-2)*.045)
 top<-f%>%filter(endpoint=='CRISPR')%>%group_by(scenario)%>%summarise(top=max(room),.groups='drop')
 r<-m%>%filter(endpoint=='CRISPR')%>%left_join(top,by='scenario')%>%
  mutate(lab=paste0('×',sprintf('%.2f',ratio)),x=2,y=pmax(top,room)+ifelse(pair=='M→P',.30,.66))
 YM<-3.62
 stopifnot(nrow(m)==8,nrow(f)==40,all(!is.na(r$ratio)),max(f$room)<YM,max(r$y)<YM-.3)
 ggplot(m,aes(x,room))+
  geom_col(aes(fill=fill,colour=col),width=.34,linewidth=LW_LINE)+
  geom_point(data=f,aes(x,room,shape=endpoint),size=PT_SMALL,fill=F5_FOLDPT,colour='white',stroke=.25)+
  geom_text(data=r,aes(x,y,label=lab,colour=col),size=6/.pt,family=CELL_FONT,vjust=0,hjust=.5)+
  facet_wrap(~sc)+scale_fill_identity()+scale_colour_identity()+scale_shape_manual(values=F5_SHAPE,guide='none')+
  scale_x_continuous(breaks=1:2,labels=F5_EP,expand=c(0,0))+
  scale_y_continuous(breaks=0:3,expand=c(0,0))+coord_cartesian(xlim=c(.5,2.5),ylim=c(0,YM),clip='off')+
  labs(x=NULL,y='Room (%)')+BASE+
  theme(panel.spacing.x=unit(3,'mm'),plot.margin=margin(5,1.5,1,1,'mm'),axis.text.x=element_text(margin=margin(t=.8,unit='mm')))}

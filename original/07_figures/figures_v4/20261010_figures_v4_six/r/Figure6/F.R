# Figure 6F (fix round 2, 2026-10-10) - room and recovered gain from development to sealed as a slope chart (the
# room-versus-recovery view; its end values repeat B and D on purpose, the form is new). For each endpoint x layer, two
# segments from the new-cell-line value (open, left) to the sealed value (filled, right): measured = gain of
# M + measured RNA / P over M only (the room; #4D4D4D) and virtual = gain of M + virtual RNA / P over M only (the
# recovered gain; layer colour). The five development folds (F0-F4, left to right) are small light-grey dots just left
# of each open circle (their mean is the open circle). Direct labels on the CRISPR RNA pair only. Thin whiskers on the
# sealed ends = 95 % grouped-bootstrap intervals of the same gains (forest.csv lo / hi of M + measured RNA / P and
# M + virtual RNA / P; their estimates are asserted equal to b_share.csv).
# Sources: b_share.csv (dev_room_pct, room_pct, dev_recovered_pct, recovered_pct), forest.csv and dev_folds.csv
# (per-fold room / recovered; fold means asserted equal to b_share.csv dev values in prep_dev_folds.R and again here).
make_F<-function(){b<-f6_share_rows();dx<-.3
 sl<-bind_rows(b%>%transmute(endpoint,layer,ep,lay,base,what='measured',col=F6_MEAS,y0=dev_room_pct,y1=room_pct),
               b%>%transmute(endpoint,layer,ep,lay,base,what='virtual',col=col,y0=dev_recovered_pct,y1=recovered_pct))
 sl$x0<-sl$base-dx;sl$x1<-sl$base+dx
 fo<-read_table('forest');arm<-ifelse(sl$what=='measured',ifelse(sl$lay=='RNA','M+RNA','M+P'),ifelse(sl$lay=='RNA','M+vRNA','M+vP'))
 m<-match(paste(sl$endpoint,arm),paste(fo$endpoint,fo$arm));stopifnot(!any(is.na(m)),all(abs(fo$gain_pct[m]-sl$y1)<1e-9))
 sl$lo<-fo$lo[m];sl$hi<-fo$hi[m]
 fd<-read_table('dev_folds')%>%filter(quantity%in%c('room','recovered'))%>%mutate(what=ifelse(quantity=='room','measured','virtual'))
 fd<-fd%>%left_join(sl%>%dplyr::select(endpoint,layer,what,base,y0),by=c('endpoint','layer','what'))
 stopifnot(nrow(fd)==40,!any(is.na(fd$base)))
 chk<-fd%>%group_by(endpoint,layer,what)%>%summarise(m=mean(value),y0=y0[1],.groups='drop');stopifnot(all(abs(chk$m-chk$y0)<5e-4))
 fd$x<-fd$base-dx-.22+(fd$fold-2)*.035
 pts<-bind_rows(sl%>%transmute(ep,lay,what,col,x=x0,y=y0,open=TRUE),sl%>%transmute(ep,lay,what,col,x=x1,y=y1,open=FALSE))
 write_csv(sl,file.path(fig_out(FIG),'panel_data','F_slopes_display.csv'));write_csv(fd,file.path(fig_out(FIG),'panel_data','F_folds_display.csv'))
 f1<-sl[sl$ep=='CRISPR'&sl$lay=='RNA',]                     # direct labels on the CRISPR RNA pair (free space above it)
 lab<-data.frame(x=c(f1$base[1]+.3,f1$base[1]),y=c(3.2,1.25),lab=c('measured','virtual'))
 xr<-c(.25,6.45)
 p<-ggplot(sl)+
  geom_hline(yintercept=0,colour=INK,linewidth=LW_AXIS)+
  geom_point(data=fd,aes(x,value),colour=F6_FOLD,size=.75,shape=16)+
  geom_segment(aes(x=x1,xend=x1,y=lo,yend=hi),colour=sl$col,linewidth=LW_ERR)+
  geom_segment(aes(x=x0,xend=x1,y=y0,yend=y1),colour=sl$col,linewidth=LW_LINE*1.5)+
  geom_point(data=pts,aes(x,y),shape=21,fill=ifelse(pts$open,'white',pts$col),colour=pts$col,size=1.7,stroke=.55)+
  geom_text(data=lab,aes(x,y,label=lab),size=GEOM_TEXT_MIN,hjust=.5,vjust=.5,colour=INK)+
  f6_xlabels(b,c(0,6))+
  scale_y_continuous(breaks=0:6,labels=num_minus)+scale_x_continuous(breaks=NULL)+
  coord_cartesian(xlim=xr,ylim=c(0,6),expand=FALSE,clip='off')+
  labs(x=NULL,y='Gain over M only (%)')+
  BASE+theme(axis.line.x=element_blank(),axis.ticks.x=element_blank(),legend.position='none')
 f6_fix(p,left=10,top=F6_TOP4,bottom=F6_BOT4)}

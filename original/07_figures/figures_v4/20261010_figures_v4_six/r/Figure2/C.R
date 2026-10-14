# Figure 2C - the cross-layer map (hero), fix round + polish 2026-10-10. Rows = deployed layer, columns = borrowed layer,
# scenario blocks side by side. Every off-diagonal tile is a small card:
#   tile fill = room (xl_matrix headroom_gain, % sRMSE, five-fold mean) on a single-hue gray scale (white -> gray, the
#     room colour of A and D); negative-room tiles are white with gray hatching (key: hatched swatch '< 0');
#   room value printed top right; bubble (top left) = recovered share (xl_matrix share) where room >= 0.3 % and share > 0,
#     in the directed pair's colour, filled = new cell lines, open = held-out tissue; tiles with room >= 0 but no bubble
#     (all three such tiles have share <= 0, i.e. nothing recovered) carry a tiny gray dot in the bubble position instead;
#   white inset (bottom) = per-fold room F0-F4 (xl_pair_folds headroom_fold) as five small bars from a zero hairline, in
#     the measured-reference gray (room = measured gain), one shared mini scale (-2.4 to +3.6 %) for all tiles.
#   Diagonal tiles = the single layer alone: five-fold mean sRMSE from panel B (xl_fold_scores), in the layer colour on
#     a light tint of that colour.
# The legend row is drawn as one grob centred under the map (room bar + hatch swatch, inset key, share bubbles); the
# diagonal tiles need no key (stated in the caption).
make_C<-function(){lv<-c('M','RNA','P')
 m<-read_table('xl_matrix')%>%mutate(x=match(borrowed,lv),y=4-match(deployed,lv),sc=factor(F2_SC[scenario],levels=F2_SC),
  room=headroom_gain,lab=sub('^-','−',sprintf('%+.2f',room)),pair=paste0(deployed,'→',borrowed),
  bub=!is.na(share)&share>0&room>=0.3,dot=room>=0&!bub,col=F2_PAIR_COL[pair],bfill=f2_fill(col,scenario),rf=ifelse(room<0,NA,room))
 f<-read_table('xl_pair_folds')%>%mutate(x=match(borrowed,lv),y=4-match(deployed,lv),sc=factor(F2_SC[scenario],levels=F2_SC))
 s<-read_table('xl_fold_scores')%>%filter(arm%in%lv)%>%group_by(scenario,arm)%>%summarise(v=mean(score),.groups='drop')%>%
  mutate(x=match(arm,lv),y=4-x,sc=factor(F2_SC[scenario],levels=F2_SC),lab=sprintf('%.3f',v),col=LAYER_COL[arm],tint=alpha_hex(LAYER_COL[arm],.14))
 stopifnot(nrow(m)==12,nrow(f)==60,nrow(s)==6,max(m$room)<3.2,all(m$blocked==(m$room<0)),
           all(f$headroom_fold> -2.4),all(f$headroom_fold<3.6),sum(m$dot)==3,all(m$share[m$dot]<=0))
 RMAX<-3.2;RAMP_HI<-'#7A7A7A'
 # inset geometry (data units; a tile spans +-0.5)
 IX<-.42;IB<- -.455;IT<- -.075;LO<- -2.4;HI<-3.6;k<-(IT-IB-.05)/(HI-LO);y0<-IB+.025-LO*k
 fx<-c(-.3,-.15,0,.15,.3);bw<-.055
 ins<-m%>%select(sc,x,y)
 fb<-f%>%mutate(xc=x+fx[fold+1],yb=y+y0,yt=y+y0+headroom_fold*k)
 hl<-m%>%filter(room<0)%>%select(sc,x,y)%>%tidyr::crossing(k=seq(-.875,.875,by=.175))%>%
  mutate(xs=pmax(-.5,k-.5),xe=pmin(.5,k+.5),ys=xs-k,ye=xe-k)%>%filter(xe>xs)%>%mutate(xs=x+xs,xe=x+xe,ys=y+ys,ye=y+ye)
 b<-m%>%filter(bub);dt<-m%>%filter(dot)
 MAXSZ<-5.6
 p<-ggplot(m,aes(x,y))+
  geom_tile(aes(fill=rf),colour='white',linewidth=.6)+
  geom_segment(data=hl,aes(x=xs,xend=xe,y=ys,yend=ye),colour='#BDBDBD',linewidth=LW_THIN)+
  geom_tile(data=s,aes(x,y),fill=s$tint,colour='white',linewidth=.6)+
  geom_text(data=s,aes(x,y,label=lab),colour=s$col,size=6.5/.pt,family=CELL_FONT,fontface='bold')+
  geom_rect(data=ins,aes(xmin=x-IX,xmax=x+IX,ymin=y+IB,ymax=y+IT),fill='white',colour=ifelse(m$room<0,'#BDBDBD',NA),linewidth=LW_THIN)+
  geom_segment(data=ins,aes(x=x-IX+.03,xend=x+IX-.03,y=y+y0,yend=y+y0),colour='#9E9E9E',linewidth=LW_THIN)+
  geom_rect(data=fb,aes(xmin=xc-bw,xmax=xc+bw,ymin=pmin(yb,yt),ymax=pmax(yb,yt)),fill=F2_MEAS_SOLID,colour=NA)+
  geom_point(data=b,aes(x=x-.24,y=y+.2,size=share),shape=21,fill=b$bfill,colour=ifelse(b$scenario=='new_cell','white',b$col),stroke=ifelse(b$scenario=='new_cell',.4,.9))+
  geom_point(data=dt,aes(x=x-.24,y=y+.2),shape=16,size=.9,colour='#9E9E9E')+
  geom_text(aes(x=ifelse(bub|dot,x+.17,x),y=y+.2,label=lab),size=6.5/.pt,family=CELL_FONT,colour=INK)+
  facet_wrap(~sc)+
  scale_fill_gradient(low='#FFFFFF',high=RAMP_HI,limits=c(0,RMAX),na.value='white',guide='none')+
  scale_size_area(max_size=MAXSZ,limits=c(0,.4),guide='none')+
  scale_x_continuous(breaks=1:3,labels=lv,expand=c(0,0))+scale_y_continuous(breaks=1:3,labels=rev(lv),expand=c(0,0))+
  coord_cartesian(clip='off')+
  labs(x='Borrowed layer',y='Deployed layer')+BASE+
  theme(axis.line=element_blank(),axis.ticks=element_blank(),legend.position='none',panel.spacing.x=unit(3,'mm'),
        axis.text.y=element_text(margin=margin(r=1,unit='mm')),axis.text.x=element_text(margin=margin(t=1,unit='mm')),
        plot.margin=margin(5,1.5,.5,1.5,'mm'))
 # ---------------- legend row (one grob, mm units, left to right)
 LH<-8.4;yc<-4.6   # row height, centre line (mm from the bottom); polish: 0.6 mm less bottom white (3 mm gutter to E)
 tx<-function(t,x,y=yc,just='left',size=PT_LEGEND,col=INK)textGrob(t,x=unit(x,'mm'),y=unit(y,'mm'),just=just,gp=gpar(fontfamily=CELL_FONT,fontsize=size,col=col))
 rc<-function(x,y,w,h,fill,col=NA,lwd=.5)rectGrob(x=unit(x,'mm'),y=unit(y,'mm'),width=unit(w,'mm'),height=unit(h,'mm'),just=c('left','bottom'),gp=gpar(fill=fill,col=col,lwd=lwd))
 sg<-function(x,y,x2,y2,col=INK,lwd=.5)segmentsGrob(unit(x,'mm'),unit(y,'mm'),unit(x2,'mm'),unit(y2,'mm'),gp=gpar(col=col,lwd=lwd))
 pt<-function(x,y,sz,fill,col='white',stroke=.4)pointsGrob(unit(x,'mm'),unit(y,'mm'),pch=21,size=unit(1,'char'),gp=gpar(fontsize=sz*.pt+stroke*.stroke/2,lwd=stroke*.stroke/2,fill=fill,col=col))
 L<-list();x<-0
 # room: title, hatched '< 0' swatch, gray bar 0-3
 L<-c(L,list(tx('Room (%)',x)));x<-x+10.2
 bh<-2.2;by<-yc-.2
 L<-c(L,list(rc(x,by,2.6,bh,'white','#9E9E9E',.4)),lapply(c(.6,1.3,2.0),function(o)sg(x+o-.5,by,x+o+.5,by+bh,'#BDBDBD',.5)),list(tx('< 0',x+1.3,by-1.6,'centre')))
 x<-x+4.6;bw2<-14;ramp<-colorRampPalette(c('#FFFFFF',RAMP_HI))(64)
 L<-c(L,list(rasterGrob(matrix(ramp,nrow=1),x=unit(x,'mm'),y=unit(by,'mm'),width=unit(bw2,'mm'),height=unit(bh,'mm'),just=c('left','bottom'),interpolate=TRUE),rc(x,by,bw2,bh,NA,INK,.4)))
 for(v in 0:3){xv<-x+bw2*v/RMAX;L<-c(L,list(sg(xv,by,xv,by-.6,INK,.4),tx(as.character(v),xv,by-1.6,'centre')))}
 x<-x+bw2+3.0
 # inset key: a mini gray tile with the white per-fold inset, labelled F0-F4
 L<-c(L,list(rc(x,by-1.1,6.4,4.4,'#B5B5B5'),rc(x+.5,by-.7,5.4,2.2,'white')))
 hk<-c(1.0,.6,1.3,.8,1.1);for(i in 1:5)L<-c(L,list(rc(x+.85+(i-1)*1.05,by-.45,.5,hk[i],F2_MEAS_SOLID)))
 L<-c(L,list(tx('F0–F4',x+7.2,by+.4)));x<-x+14.9
 # share bubbles (same area scale as the map)
 L<-c(L,list(tx('Recovered share (%)',x)));x<-x+20.9
 for(v in c(.1,.3)){sz<-MAXSZ*sqrt(v/.4);d<-sz*.75;L<-c(L,list(pt(x+d/2,yc,sz,'#7F7F7F'),tx(as.character(v*100),x+d+.8,yc)));x<-x+d+4.6}
 WL<-x-4.6+3.2   # row width: last bubble label ('30', ~2.4 mm) + a small end pad
 # centre the row under the map (between the left edge of the first and the right edge of the last tile column)
 opened<-if(dev.cur()==1){pdf(NULL);TRUE}else FALSE
 g<-ggplotGrob(p);if(opened)dev.off()
 pl<-g$layout[grepl('^panel',g$layout$name),]
 leg<-gTree(children=do.call(gList,L),vp=viewport(x=unit(.5,'npc'),width=unit(WL,'mm'),just='centre'))
 g<-gtable::gtable_add_rows(g,unit(LH,'mm'),pos=-1);g<-gtable::gtable_add_grob(g,leg,t=nrow(g),l=min(pl$l),r=max(pl$r),clip='off',name='c_legend')
 list(plot=p,grob=g)}

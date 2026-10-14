# Figure 5C (hero) - the cross-layer map on both endpoints, in the Figure 2C card grammar. Four 3 x 3 blocks in one row:
# GDSC (new cell lines, held-out tissue) | CRISPR (new cell lines, held-out tissue); rows = deployed layer, columns =
# borrowed layer. Every off-diagonal tile:
#   fill  = room (a_map room_gain, % sRMSE, five-fold mean) on the single-hue gray scale of Figure 2C (white -> gray,
#           0-3.2 %); negative-room tiles white with gray hatching;
#   value = room, printed top right;
#   bubble (top left) = recovered share (a_map share) where room >= 0.3 % and share > 0, in the directed pair's colour
#           (M->RNA blue, M->P pink, other pairs gray), filled = new cell lines, open = held-out tissue;
#   white inset (bottom) = per-fold room F0-F4 as five bars from a zero hairline (GDSC: f2_xl_pair_folds headroom_fold;
#           CRISPR: f5_crispr_pair_folds headroom_fold), one shared mini scale (-2.4 to +3.6 %) for all 24 tiles, so the
#           CRISPR rooms read directly as shorter bars. Fix round: taller tiles and insets (1.07 mm per % instead of 0.77),
#           and five fold ticks on the zero line, so near-zero tiles read as five near-zero folds, not as empty boxes.
# Legend: one centred line under the map (centred on the plot area), inset key labelled 'F0-F4' as in Figure 2C.
# Diagonal tiles = the layer alone, five-fold mean sRMSE (GDSC f2_xl_fold_scores, CRISPR f5_crispr_layer_fold_scores),
# in the layer colour on a light tint. Endpoint headers carry the endpoint glyph (circle GDSC, triangle CRISPR).
# Final polish (2026-10-11), as Figure 2C: the "Layer alone (sRMSE)" legend key is gone (the caption explains the diagonal
# tiles), and the six off-diagonal tiles with room >= 0 but no bubble carry a tiny gray dot in the bubble position, with the
# value moved right as in the bubble tiles. GDSC: three tiles with share <= 0 (as in Figure 2C). CRISPR: RNA->P in both
# scenarios (room 0.187 and 0.299 %, below the 0.3 % rule, so no share is defined) and P->RNA held-out tissue (share < 0).
make_C<-function(){lv<-c('M','RNA','P')
 OFF<-c(GDSC.new_cell=0,GDSC.tissue=3.14,CRISPR.new_cell=6.78,CRISPR.tissue=9.92)
 m<-read_table('a_map')%>%transmute(endpoint,scenario,deployed,borrowed,room=room_gain,share)%>%
  mutate(key=paste(endpoint,scenario,sep='.'),x=match(borrowed,lv)+OFF[key],y=4-match(deployed,lv),pair=paste0(deployed,'→',borrowed),
   lab=f5_slab(room),bub=!is.na(share)&share>0&room>=0.3,dot=room>=0&!bub,col=F5_PAIR_COL[pair],bfill=f5_fill(col,scenario),rf=ifelse(room<0,NA,room))
 gx<-read_table('f2_xl_pair_folds')%>%transmute(endpoint='GDSC',scenario,deployed,borrowed,fold,room_fold=headroom_fold,room_mean=headroom_gain)
 cx<-read_table('f5_crispr_pair_folds')%>%transmute(endpoint,scenario,deployed,borrowed,fold,room_fold=headroom_fold,room_mean=headroom_gain)
 f<-bind_rows(gx,cx)%>%mutate(key=paste(endpoint,scenario,sep='.'),x=match(borrowed,lv)+OFF[key],y=4-match(deployed,lv))
 chk<-f%>%distinct(endpoint,scenario,deployed,borrowed,room_mean)%>%inner_join(m,by=c('endpoint','scenario','deployed','borrowed'))
 s<-bind_rows(read_table('f2_xl_fold_scores')%>%filter(arm%in%lv)%>%transmute(endpoint='GDSC',scenario,arm,score),
              read_table('f5_crispr_layer_fold_scores')%>%transmute(endpoint='CRISPR',scenario,arm,score))%>%
  group_by(endpoint,scenario,arm)%>%summarise(v=mean(score),n=n(),.groups='drop')%>%
  mutate(key=paste(endpoint,scenario,sep='.'),x=match(arm,lv)+OFF[key],y=4-match(arm,lv),lab=sprintf('%.3f',v),col=LAYER_COL[arm],tint=alpha_hex(LAYER_COL[arm],.14))
 stopifnot(nrow(m)==24,nrow(f)==120,nrow(chk)==24,max(abs(chk$room_mean-chk$room))<1e-9,nrow(s)==12,all(s$n==5),
           max(m$room)<3.2,all(f$room_fold> -2.4),all(f$room_fold<3.6),
           sum(m$dot)==6,all((is.na(m$share)&m$room<.3)[m$dot]|(m$share<=0)[m$dot]))
 RMAX<-3.2;RAMP_HI<-'#7A7A7A'
 IX<-.42;IB<- -.465;IT<- -.03;LO<- -2.4;HI<-3.6;k<-(IT-IB-.05)/(HI-LO);y0<-IB+.025-LO*k
 fx<-c(-.3,-.15,0,.15,.3);bw<-.055
 ins<-m%>%select(x,y)
 tk<-ins%>%tidyr::crossing(fo=fx)%>%mutate(xc=x+fo)
 fb<-f%>%mutate(xc=x+fx[fold+1],yb=y+y0,yt=y+y0+room_fold*k)
 hl<-m%>%filter(room<0)%>%select(x,y)%>%tidyr::crossing(kk=seq(-.875,.875,by=.175))%>%
  mutate(xs=pmax(-.5,kk-.5),xe=pmin(.5,kk+.5),ys=xs-kk,ye=xe-kk)%>%filter(xe>xs)%>%mutate(xs=x+xs,xe=x+xe,ys=y+ys,ye=y+ye)
 b<-m%>%filter(bub);dt<-m%>%filter(dot)
 MAXSZ<-4.6
 # headers: scenario over each block, endpoint (with glyph) over each pair of blocks
 sch<-tibble::tibble(x=2+OFF,lab=rep(F5_SC,2))
 eph<-tibble::tibble(ep=F5_EP,x1=.5+OFF[c(1,3)],x2=3.5+OFF[c(2,4)])%>%mutate(xc=(x1+x2)/2)
 RH<-(SLOT_H-11.5-6.4-7.4)/3   # approx. tile height (mm): header offsets stay fixed in mm when the slot grows
 YS<-3.5+3.4/RH;YR<-3.5+7.1/RH;YE<-3.5+10/RH
 ax<-tibble::tibble(x=as.vector(outer(1:3,OFF,'+')),lab=rep(lv,4))
 p<-ggplot(m,aes(x,y))+
  geom_tile(aes(fill=rf),colour='white',linewidth=.6)+
  geom_segment(data=hl,aes(x=xs,xend=xe,y=ys,yend=ye),colour='#BDBDBD',linewidth=LW_THIN)+
  geom_tile(data=s,aes(x,y),fill=s$tint,colour='white',linewidth=.6)+
  geom_text(data=s,aes(x,y,label=lab),colour=s$col,size=6/.pt,family=CELL_FONT,fontface='bold')+
  geom_rect(data=ins,aes(xmin=x-IX,xmax=x+IX,ymin=y+IB,ymax=y+IT),fill='white',colour=NA)+
  geom_segment(data=ins,aes(x=x-IX+.03,xend=x+IX-.03,y=y+y0,yend=y+y0),colour='#9E9E9E',linewidth=LW_THIN)+
  geom_segment(data=tk,aes(x=xc,xend=xc,y=y+y0-.022,yend=y+y0+.022),colour='#9E9E9E',linewidth=LW_THIN)+
  geom_rect(data=fb,aes(xmin=xc-bw,xmax=xc+bw,ymin=pmin(yb,yt),ymax=pmax(yb,yt)),fill=F5_MEAS,colour=NA)+
  geom_point(data=b,aes(x=x-.27,y=y+.21,size=share),shape=21,fill=b$bfill,colour=ifelse(b$scenario=='new_cell','white',b$col),stroke=ifelse(b$scenario=='new_cell',.35,.75))+
  geom_point(data=dt,aes(x=x-.27,y=y+.21),shape=16,size=.9,colour='#9E9E9E')+
  geom_text(aes(x=ifelse(bub|dot,x+.15,x),y=y+.21,label=lab),size=6/.pt,family=CELL_FONT,colour=INK)+
  geom_text(data=sch,aes(x=x,y=YS,label=lab),size=PT_STRIP/.pt,family=CELL_FONT,fontface='bold',colour=INK,vjust=.5)+
  geom_segment(data=eph,aes(x=x1,xend=x2,y=YR,yend=YR),colour=INK,linewidth=LW_LINE)+
  geom_text(data=eph,aes(x=xc+.14,y=YE,label=ep),size=PT_AXIS_TITLE/.pt,family=CELL_FONT,fontface='bold',colour=INK,hjust=0,vjust=.5)+
  geom_point(data=eph,aes(x=xc-.08,y=YE,shape=ep),size=1.9,fill='#525252',colour='#525252',stroke=.4)+
  scale_shape_manual(values=F5_SHAPE,guide='none')+
  scale_fill_gradient(low='#FFFFFF',high=RAMP_HI,limits=c(0,RMAX),na.value='white',guide='none')+
  scale_size_area(max_size=MAXSZ,limits=c(0,.4),guide='none')+
  scale_x_continuous(breaks=ax$x,labels=ax$lab,expand=c(0,0))+scale_y_continuous(breaks=1:3,labels=rev(lv),expand=c(0,0))+
  coord_cartesian(xlim=c(.5,3.5+OFF[[4]]),ylim=c(.5,3.5),clip='off')+
  labs(x='Borrowed layer',y='Deployed layer')+BASE+
  theme(axis.line=element_blank(),axis.ticks=element_blank(),legend.position='none',
        axis.text.y=element_text(margin=margin(r=1,unit='mm')),axis.text.x=element_text(margin=margin(t=1,unit='mm')),
        axis.title.x=element_text(margin=margin(t=1,unit='mm')),
        plot.margin=margin(11.5,1.5,.5,1,"mm"))
 # ---------------- legend row (one grob, mm units, one centred line under the plot area) - same keys as Figure 2C
 LH<-6.4;yc<-3.9
 opened<-if(dev.cur()==1){pdf(NULL);TRUE}else FALSE
 tw<-function(t,size=PT_LEGEND)convertWidth(grobWidth(textGrob(t,gp=gpar(fontfamily=CELL_FONT,fontsize=size))),'mm',valueOnly=TRUE)
 tx<-function(t,x,y=yc,just='left',size=PT_LEGEND,col=INK)textGrob(t,x=unit(x,'mm'),y=unit(y,'mm'),just=just,gp=gpar(fontfamily=CELL_FONT,fontsize=size,col=col))
 rc<-function(x,y,w,h,fill,col=NA,lwd=.5)rectGrob(x=unit(x,'mm'),y=unit(y,'mm'),width=unit(w,'mm'),height=unit(h,'mm'),just=c('left','bottom'),gp=gpar(fill=fill,col=col,lwd=lwd))
 sg<-function(x,y,x2,y2,col=INK,lwd=.5)segmentsGrob(unit(x,'mm'),unit(y,'mm'),unit(x2,'mm'),unit(y2,'mm'),gp=gpar(col=col,lwd=lwd))
 pt<-function(x,y,sz,fill,col='white',stroke=.4)pointsGrob(unit(x,'mm'),unit(y,'mm'),pch=21,size=unit(1,'char'),gp=gpar(fontsize=sz*.pt+stroke*.stroke/2,lwd=stroke*.stroke/2,fill=fill,col=col))
 L<-list();x<-0
 L<-c(L,list(tx('Room (%)',x)));x<-x+tw('Room (%)')+1.2
 bh<-2.2;by<-yc-.2
 L<-c(L,list(rc(x,by,2.6,bh,'white','#9E9E9E',.4)),lapply(c(.6,1.3,2.0),function(o)sg(x+o-.5,by,x+o+.5,by+bh,'#BDBDBD',.5)),list(tx('< 0',x+1.3,by-1.7,'centre')))
 x<-x+5.4;bw2<-17;ramp<-colorRampPalette(c('#FFFFFF',RAMP_HI))(64)
 L<-c(L,list(rasterGrob(matrix(ramp,nrow=1),x=unit(x,'mm'),y=unit(by,'mm'),width=unit(bw2,'mm'),height=unit(bh,'mm'),just=c('left','bottom'),interpolate=TRUE),rc(x,by,bw2,bh,NA,INK,.4)))
 for(v in 0:3){xv<-x+bw2*v/RMAX;L<-c(L,list(sg(xv,by,xv,by-.6,INK,.4),tx(as.character(v),xv,by-1.7,'centre')))}
 x<-x+bw2+5.5
 L<-c(L,list(rc(x,by-1.1,6.4,4.4,'#B5B5B5'),rc(x+.5,by-.7,5.4,2.2,'white'),sg(x+.7,by-.1,x+5.7,by-.1,'#9E9E9E',.3)))
 hk<-c(1.0,.6,1.3,.8,1.1);for(i in 1:5)L<-c(L,list(rc(x+.85+(i-1)*1.05,by-.1,.5,hk[i],F5_MEAS)))
 L<-c(L,list(tx('F0–F4',x+7.4,yc)));x<-x+7.4+tw('F0–F4')+5.5
 L<-c(L,list(tx("Recovered share (%)",x)));x<-x+tw('Recovered share (%)')+1.6
 for(v in c(.1,.2,.3)){sz<-MAXSZ*sqrt(v/.4);d<-sz*.75;L<-c(L,list(pt(x+d/2,yc,sz,'#7F7F7F'),tx(as.character(v*100),x+d+.8,yc)));x<-x+d+.8+tw('30')+2.6}
 xend<-x-2.6   # end of the last bubble label (the "Layer alone" key was removed in the final polish)
 if(opened)dev.off()
 g<-f5_grob(p)
 # centre the legend line on the plot area (panel columns of the gtable)
 pl<-g$layout[grepl('^panel',g$layout$name),];opened<-if(dev.cur()==1){pdf(NULL);TRUE}else FALSE
 wmm<-vapply(seq_along(g$widths),function(i){u<-g$widths[i];if(grepl('null',as.character(u)))NA_real_ else convertWidth(u,'mm',valueOnly=TRUE)},0)
 if(opened)dev.off()
 pw<-SLOT_W-sum(wmm,na.rm=TRUE);pleft<-sum(wmm[seq_len(min(pl$l)-1)],na.rm=TRUE);shift<-pleft+pw/2-xend/2
 leg<-gTree(children=do.call(gList,L),vp=viewport(x=unit(shift,'mm'),y=unit(0,'npc'),width=unit(xend,'mm'),height=unit(1,'npc'),just=c('left','bottom')))
 g<-gtable::gtable_add_rows(g,unit(LH,'mm'),pos=-1);g<-gtable::gtable_add_grob(g,leg,t=nrow(g),l=1,r=ncol(g),clip='off',name='c_legend')
 list(plot=p,grob=g)}

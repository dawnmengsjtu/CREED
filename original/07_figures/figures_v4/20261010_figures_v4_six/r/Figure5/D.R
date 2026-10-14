# Figure 5D - CRISPR per-fold room and recovery for the four metabolite-row cells (mirror of Figure 2D, narrow and tall).
# Four blocks top to bottom, in order of decreasing room (M->RNA new cell lines, M->RNA held-out tissue, M->P new cell lines,
# M->P held-out tissue), each with a bold title (pair in its colour + scenario). Per fold (b_share_fold, endpoint CRISPR):
# light track from 0 to the measured gain (room_fold) ending in the measured-reference dot (#4D4D4D), overlaid by a
# pair-coloured bar from 0 to the virtual gain (recovered_fold) ending in the pair-coloured triangle (CRISPR glyph; filled =
# new cell lines, open = held-out tissue); the short dark-gray (#4D4D4D, the measured colour) tick marks one third of that
# fold's room. Fix round: the four five-fold 'Mean' rows were dropped (C prints the means), the axis ends at 2.4 % (largest
# room 2.32 %) and the block gaps are tighter.
make_D<-function(){
 f<-read_table('b_share_fold')%>%filter(endpoint=='CRISPR')%>%transmute(scenario,pair,row=paste0('F',fold),room=room_fold,rec=recovered_fold)
 blocks<-tibble::tibble(pair=c('M→RNA','M→RNA','M→P','M→P'),scenario=c('new_cell','tissue','new_cell','tissue'))%>%mutate(b=row_number())
 pos<-list();yy<-0;tt<-numeric(4)
 for(i in 1:4){tt[i]<-yy;yy<-yy-(if(i==1)1.75 else 1.15);pos[[i]]<-setNames(yy-(0:4),paste0('F',0:4));yy<-yy-4-1.3}
 d<-f%>%left_join(blocks,by=c('pair','scenario'))%>%
  mutate(yy=mapply(function(b,r)pos[[b]][[r]],b,row),col=F5_PAIR_COL[pair],fill=f5_fill(col,scenario))
 XM<-2.4
 stopifnot(nrow(d)==20,all(d$room>0),all(d$rec>0),max(d$room)<XM-.05)
 ax<-d%>%distinct(yy,row)%>%arrange(yy)
 ti<-blocks%>%mutate(y=tt[b],col=F5_PAIR_COL[pair],sc=F5_SC[scenario])
 tag<-d%>%filter(b==1,row=='F0')%>%mutate(y=tt[1]-.85)
 TW<-.62
 ggplot(d,aes(y=yy))+
  geom_segment(aes(x=0,xend=room,yend=yy),colour=F5_TRACK,linewidth=2.7,lineend='butt')+
  geom_segment(aes(x=0,xend=rec,yend=yy,colour=col),linewidth=2.7,lineend='butt')+
  geom_segment(aes(x=room/3,xend=room/3,y=yy-.38,yend=yy+.38),colour=F5_MEAS,linewidth=LW_LINE*1.2)+
  geom_point(aes(x=room),shape=21,fill=F5_MEAS,colour='white',stroke=.3,size=PT_LARGE+.3)+
  geom_point(aes(x=rec,fill=fill,colour=col),shape=24,stroke=.6,size=PT_LARGE+.1)+
  geom_text(data=ti,aes(x=0,y=y,label=pair,colour=col),hjust=0,vjust=.5,size=PT_STRIP/.pt,fontface='bold',family=CELL_FONT)+
  geom_text(data=ti,aes(x=TW,y=y,label=sc),hjust=0,vjust=.5,size=PT_STRIP/.pt,fontface='bold',family=CELL_FONT,colour=INK)+
  geom_text(data=tag,aes(x=room,y=y,label='measured'),hjust=.5,size=6/.pt,family=CELL_FONT,colour='#525252')+
  geom_text(data=tag,aes(x=room/3,y=y,label='1/3'),hjust=.5,size=6/.pt,family=CELL_FONT,colour='#525252')+
  scale_colour_identity()+scale_fill_identity()+
  scale_x_continuous(breaks=c(0,1,2),expand=c(0,0))+scale_y_continuous(breaks=ax$yy,labels=ax$row,expand=c(0,0))+
  coord_cartesian(xlim=c(0,XM),ylim=c(min(d$yy)-.75,.45),clip='off')+
  labs(x='Gain over M only (%)',y=NULL)+BASE+
  theme(axis.line.y=element_blank(),axis.ticks.y=element_blank(),axis.text.y=element_text(size=PT_TICK,margin=margin(r=.8,unit='mm')),
        plot.margin=margin(5,2.5,1,1,'mm'))}

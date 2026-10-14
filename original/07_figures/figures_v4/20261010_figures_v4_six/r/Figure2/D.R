# Figure 2D - per-fold room and recovery for the four metabolite-row cells (fix round + polish 2026-10-10), narrow and tall.
# Four blocks top to bottom (M->RNA new cell lines, M->RNA held-out tissue, M->P new cell lines, M->P held-out tissue),
# each with a horizontal bold title above it (pair in its colour + scenario). For each fold (xl_pair_folds): a light track
# from 0 to the measured gain (room, headroom_fold) ending in the measured-reference dot, overlaid by a pair-coloured bar
# from 0 to the virtual gain (recovery, recovered_fold) ending in the pair-coloured dot (filled = new cell lines, open =
# held-out tissue); the short black tick marks one third of that fold's room. 'Mean' rows use the five-fold means
# (headroom_gain / recovered_gain). The two marks are named once, directly, on the first Mean row ('measured', '1/3').
# Polish: x runs 0-3.6 (ticks 0-3), the tightest range that keeps every measured dot on scale (largest per-fold room =
# 3.586 %, M->RNA new cell lines F2; 0-3.3 would cut F1 3.506 and F2 3.586).
make_D<-function(){
 f<-read_table('xl_pair_folds')%>%filter(deployed=='M')%>%mutate(row=paste0('F',fold),room=headroom_fold,rec=recovered_fold)
 mm<-f%>%distinct(scenario,pair,headroom_gain,recovered_gain)%>%transmute(scenario,pair,row='Mean',room=headroom_gain,rec=recovered_gain)
 blocks<-tibble::tibble(pair=c('M→RNA','M→RNA','M→P','M→P'),scenario=c('new_cell','tissue','new_cell','tissue'))%>%mutate(b=row_number())
 # vertical positions (data units, top = 0): title, [tag row in block 1], Mean, F0-F4; extra gap between the two pairs
 pos<-list();yy<-0;tt<-numeric(4)
 for(i in 1:4){tt[i]<-yy;yy<-yy-(if(i==1)1.75 else 1.05);pos[[i]]<-c(Mean=yy,setNames(yy-1.2-(0:4),paste0('F',0:4)));yy<-yy-1.2-4-(if(i==2)1.75 else 1.35)}
 d<-bind_rows(f%>%select(scenario,pair,row,room,rec),mm)%>%left_join(blocks,by=c('pair','scenario'))%>%
  mutate(yy=mapply(function(b,r)pos[[b]][[r]],b,row),col=F2_PAIR_COL[pair],fill=f2_fill(col,scenario))
 stopifnot(nrow(d)==24,all(d$room>0),all(d$rec>0),max(d$room)<3.6)
 ax<-d%>%distinct(yy,row)%>%arrange(yy)
 ti<-blocks%>%mutate(y=tt[b],col=F2_PAIR_COL[pair],sc=F2_SC[scenario])
 tag<-d%>%filter(b==1,row=='Mean')%>%mutate(y=tt[1]-.82)
 XM<-3.6;TW<-.86   # x extent; offset of the scenario words after the pair name (data units)
 ggplot(d,aes(y=yy))+
  geom_segment(aes(x=0,xend=room,yend=yy),colour='#D9D9D9',linewidth=2.5,lineend='butt')+
  geom_segment(aes(x=0,xend=rec,yend=yy,colour=col),linewidth=2.5,lineend='butt')+
  geom_segment(aes(x=room/3,xend=room/3,y=yy-.36,yend=yy+.36),colour=INK,linewidth=LW_LINE)+
  geom_point(aes(x=room),shape=21,fill=F2_MEAS_SOLID,colour='white',stroke=.3,size=PT_LARGE+.3)+
  geom_point(aes(x=rec,fill=fill,colour=col),shape=21,stroke=.65,size=PT_LARGE+.3)+
  geom_text(data=ti,aes(x=0,y=y,label=pair,colour=col),hjust=0,vjust=.5,size=PT_STRIP/.pt,fontface='bold',family=CELL_FONT)+
  geom_text(data=ti,aes(x=TW,y=y,label=sc),hjust=0,vjust=.5,size=PT_STRIP/.pt,fontface='bold',family=CELL_FONT,colour=INK)+
  geom_text(data=tag,aes(x=room,y=y,label='measured'),hjust=.5,size=6/.pt,family=CELL_FONT,colour='#525252')+
  geom_text(data=tag,aes(x=room/3,y=y,label='1/3'),hjust=.5,size=6/.pt,family=CELL_FONT,colour='#525252')+
  scale_colour_identity()+scale_fill_identity()+
  scale_x_continuous(breaks=0:3,expand=c(0,0))+scale_y_continuous(breaks=ax$yy,labels=ax$row,expand=c(0,0))+
  coord_cartesian(xlim=c(0,XM),ylim=c(min(d$yy)-.7,.45),clip='off')+
  labs(x='Gain over M only (%)',y=NULL)+BASE+
  theme(axis.line.y=element_blank(),axis.ticks.y=element_blank(),axis.text.y=element_text(size=PT_TICK,margin=margin(r=.8,unit='mm')),
        plot.margin=margin(5,2.5,1,1,'mm'))}

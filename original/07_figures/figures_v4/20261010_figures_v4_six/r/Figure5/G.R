# Figure 5G - the panel law on both endpoints, protein and gene panels in one plot (old D and E merged). x = panel size k
# (log2), y = recovery R(k); one facet per scenario. Lines + large symbols = R from five-fold-mean sRMSE (c_panel_curve),
# colour = panel layer (protein #D55E00, gene #56B4E9, as in Figure 4), glyph = endpoint (circle GDSC, triangle CRISPR),
# filled = new cell lines, open = held-out tissue. Small symbols = per-fold R (c_panel_fold) in the series colour at 35 %
# alpha (open with a 55 %-alpha outline in held-out tissue), drawn at the series offset so each fold is tied to its series;
# large symbols on top. Fix round: the four series are offset by -0.36/-0.12/+0.12/+0.36 on log2(k) (was +-0.07/+-0.21),
# for legibility only. Gray reference lines: R = 1 (M + measured layer, the teacher) and R = 0 (M only), named once at the
# right.
# Final polish (2026-10-11): the main y axis ends at 1.1. A thin strip above a broken axis (about 3.5 mm, compressed
# scale: display = 1.17 + 0.3 x (R - 1.2), tick 1.5) holds the only two per-fold values above 1.1, both held-out tissue F1:
# GDSC gene panel k = 32 (R = 1.56) and CRISPR protein panel k = 64 (R = 1.25). The break is drawn on the axis line (white
# gap + two parallel diagonal strokes), as in Figures 2B and 5B; the recovered height goes to the main plot area. The
# protein / gene colour key moved from the former empty band (1.38-1.52) to the empty top-left corner of the new-cell-line
# facet, under the R = 1 line. Axis titles follow Figure 4 ('Recovery fraction R(k)', 'Panel size, k (measured molecules)').
make_G<-function(){
 cv<-read_table('c_panel_curve');fd<-read_table('c_panel_fold')
 TOP<-1.1;BK<-1.15;S0<-1.17;SC<-.3;YT<-1.30
 tr<-function(v)ifelse(v>BK,S0+SC*(v-1.2),v);itr<-function(v)ifelse(v>BK,1.2+(v-S0)/SC,v)
 dx<-c(protein.GDSC=-.36,protein.CRISPR=-.12,gene.GDSC=.12,gene.CRISPR=.36)
 prep<-function(d)d%>%mutate(sc=factor(F5_SC[scenario],levels=F5_SC),s=paste(layer,endpoint,sep='.'),x=log2(k)+dx[s],col=PANEL_COL[layer],y=tr(R))
 cv<-prep(cv)%>%mutate(fill=f5_fill(col,scenario))
 fd<-prep(fd)%>%mutate(pfill=ifelse(scenario=='new_cell',alpha_hex(col,.35),'white'),pcol=ifelse(scenario=='new_cell',alpha_hex(col,.35),alpha_hex(col,.55)))
 hi<-fd%>%filter(R>TOP)
 stopifnot(nrow(cv)==32,nrow(fd)==160,max(fd$R)<itr(YT)-.02,min(fd$R)> -.15,max(cv$R)<TOP,
           nrow(hi)==2,all(hi$scenario=='tissue'),all(hi$fold==1),all(hi$R>1.2),!any(fd$R>TOP&fd$R<=1.2))
 ref<-tibble::tibble(sc=factor(F5_SC[['tissue']],levels=F5_SC),y=c(1,0),lab=c('Measured','M only'))
 key<-tibble::tibble(sc=factor(F5_SC[['new_cell']],levels=F5_SC),x=2.58,y=c(.945,.855),lab=c('Protein panel','Gene panel'),col=PANEL_COL[c('protein','gene')])
 # axis break on the new-cell-line facet's y axis line (the only drawn y axis): white gap + two parallel diagonal strokes
 bm<-tibble::tibble(sc=factor(F5_SC[['new_cell']],levels=F5_SC),x=2.5,yc=(TOP+BK)/2)
 ggplot(cv,aes(x,y))+
  geom_hline(yintercept=1,linetype='22',linewidth=LW_LINE,colour='#7F7F7F')+
  geom_hline(yintercept=0,linetype='11',linewidth=LW_LINE,colour='#7F7F7F')+
  geom_point(data=fd,aes(x,y,shape=endpoint),size=PT_SMALL-.1,fill=fd$pfill,colour=fd$pcol,stroke=.3)+
  geom_line(aes(group=s,colour=col),linewidth=LW_LINE)+
  geom_point(aes(shape=endpoint,fill=fill,colour=col),size=PT_LARGE-.1,stroke=.6)+
  geom_text(data=ref,aes(x=6.62,y=y,label=lab),hjust=0,size=6/.pt,family=CELL_FONT,colour='#7F7F7F')+
  geom_text(data=key,aes(x=x,y=y,label=lab,colour=col),hjust=0,size=6/.pt,family=CELL_FONT)+
  geom_rect(data=bm,aes(xmin=x-.09,xmax=x+.09,ymin=yc-.026,ymax=yc+.026),inherit.aes=FALSE,fill='white',colour=NA)+
  geom_segment(data=bm,aes(x=x-.2,xend=x+.2,y=yc-.026-.024,yend=yc-.026+.024),inherit.aes=FALSE,linewidth=LW_AXIS,lineend='butt')+
  geom_segment(data=bm,aes(x=x-.2,xend=x+.2,y=yc+.026-.024,yend=yc+.026+.024),inherit.aes=FALSE,linewidth=LW_AXIS,lineend='butt')+
  facet_wrap(~sc)+scale_fill_identity()+scale_colour_identity()+scale_shape_manual(values=F5_SHAPE,guide='none')+
  scale_x_continuous(breaks=3:6,labels=c('8','16','32','64'),expand=c(0,0))+
  scale_y_continuous(breaks=c(0,.5,1,tr(1.5)),labels=function(b)num_minus_acc(0.1)(itr(b)),expand=c(0,0))+
  coord_cartesian(xlim=c(2.5,6.55),ylim=c(-.15,YT),clip='off')+
  labs(x='Panel size, k (measured molecules)',y='Recovery fraction R(k)')+BASE+
  theme(panel.spacing.x=unit(3,'mm'),plot.margin=margin(5,10,1,1,'mm'))}

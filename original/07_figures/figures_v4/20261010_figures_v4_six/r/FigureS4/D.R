# Figure S4D (v4 fix 1) <- v3 Figure S1B: the pairing universe as a ladder of horizontal bars (cell-line counts, one linear
# axis). Bar totals printed at the bar ends; the used 608 split into development (ink) and sealed (hollow) segments with
# their counts; a horizontal Development / Sealed key in the empty band right of the two short bottom bars.
# 2026-10-11 (audits A5-22, A5-23): three-way key with Figure 2H's naming - Development (ink), Validation (excluded; mid
# gray, the 64 lines whose values were seen in earlier development) and Internal-final (sealed; open, the 123 sealed
# lines). The panel prints no bridge-conflict text.
make_D<-function(){
 d<-as.data.frame(read_table('b_ladder'));d$order<-as.integer(d$order)
 d$fill<-ifelse(d$order==6,S4_USED_FILL[d$seg],S4_LADDER_FILL[as.character(d$order)])
 d$outline<-ifelse(d$order==6&d$seg!='train',INK,NA)
 BARH<-.32;XMAX<-1130;YL<-c(.4,8.6)
 bars<-d[d$value>0,]
 lab<-d[!duplicated(d$y),c('y','order','bar_total')];lab$txt<-formatC(lab$bar_total,big.mark=',',format='d')
 seg<-d[!is.na(d$seg_label),];seg$xm<-(seg$x0+seg$x1)/2;seg$txtcol<-ifelse(seg$seg=='train','#FFFFFF',INK)
 KEY_LV<-c('Development','Validation (excluded)','Internal-final (sealed)')
 key<-data.frame(x=-1e4,y=-1e4,set=factor(KEY_LV,levels=KEY_LV))
 p<-ggplot()+
  geom_rect(data=bars,aes(xmin=x0,xmax=x1,ymin=y-BARH,ymax=y+BARH,fill=fill,colour=outline),linewidth=LW_AXIS)+
  geom_text(data=lab,aes(x=bar_total+10,y=y,label=txt),hjust=0,vjust=.5,size=GEOM_TEXT_MIN,family=CELL_FONT,fontface='bold',colour=INK)+
  geom_text(data=seg,aes(x=xm,y=y,label=seg_label,colour=txtcol),hjust=.5,vjust=.5,size=GEOM_TEXT_MIN,family=CELL_FONT)+
  geom_point(data=key,aes(x,y,shape=set),size=2.6,stroke=LW_AXIS*.pt/2.0,colour=INK)+
  scale_fill_identity()+scale_colour_identity()+
  scale_shape_manual(values=setNames(c(15,22,0),KEY_LV),name=NULL,
    guide=guide_legend(override.aes=list(fill=unname(S4_USED_FILL[c('train','validation','internal_final')]))))+
  scale_x_continuous(breaks=c(0,500,1000),labels=c('0','500','1,000'),expand=c(0,0))+
  scale_y_continuous(breaks=8:1,labels=unname(S4_LADDER_LABEL[as.character(1:8)]),expand=c(0,0))+
  coord_cartesian(xlim=c(0,XMAX),ylim=YL)+
  labs(x='Cell lines',y=NULL)+
  BASE+legend_inside(160/XMAX,(1.5-YL[1])/diff(YL),just=c(0,.5))+
  theme(legend.direction='horizontal',axis.line.y=element_blank(),axis.ticks.y=element_blank(),axis.text.y=element_text(hjust=1),
        legend.key.width=unit(3,'mm'),legend.key.height=unit(3,'mm'),legend.key.spacing.x=unit(4,'mm'))
 s4_fix(p,top=S4_TOP2)}

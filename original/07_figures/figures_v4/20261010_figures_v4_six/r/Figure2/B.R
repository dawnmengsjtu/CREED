# Figure 2B - single-layer baselines as a dot-and-crossbar strip (fix round + polish, 2026-10-10): per-fold sRMSE of the
# null model and of M, RNA and P alone (xl_fold_scores, arms null_drug / M / RNA / P), one facet per scenario, free y.
# Points = folds (arm colour; filled = new cell lines, open = held-out tissue, the page-wide scenario fill), dodged by
# fold and joined per fold in one light gray; folds are labelled directly at the right end of each line (F0-F4).
# Crossbar = five-fold mean in the arm colour. Null is neutral dark gray (#555555), not black: dark gray/black means the
# measured reference in D; M only keeps the mid-gray layer colour. Held-out tissue fold F1 (all four arms 1.93-1.99) sits
# in a short strip above a broken axis (display y = 1.18 + (sRMSE - 1.92) above the break); the five-fold means include
# it. The break is drawn in mm on the axis line (white gap + two parallel diagonal strokes, each 2.5 mm long).
# Polish: categories evenly spaced but wider (tighter x padding, 5 mm facet spacing, 3.5 mm right margin) so the four
# tick labels have visible gaps.
make_B<-function(){
 lv<-c('null_drug','M','RNA','P');lab<-c('Null','M only','RNA only','P only')
 BRK<-1.15;UP0<-1.92;UPY<-1.18                                     # lower segment ends at 1.15; 1.92 is drawn at 1.18
 NULL_COL<-'#555555';HW<-.25;XL<-.6;XR<-4.27                        # crossbar half-width; x limits (data units)
 tr<-function(v)ifelse(v>1.5,UPY+(v-UP0),v);itr<-function(v)ifelse(v>BRK+.01,v-UPY+UP0,v)
 d<-read_table('xl_fold_scores')%>%filter(arm%in%lv)%>%mutate(a=match(arm,lv),arm=factor(arm,levels=lv,labels=lab),
   sc=factor(F2_SC[scenario],levels=F2_SC),fold=factor(fold),x=a+(as.integer(as.character(fold))-2)*.075,y=tr(score))
 m<-d%>%group_by(sc,scenario,arm,a)%>%summarise(score=mean(score),.groups='drop')
 stopifnot(nrow(d)==40,all(d$score>.83),all(m$score<BRK),all(d$score<1.15|d$score>UP0),all(d$score<2.0),
           all((d$score>1.5)==(d$scenario=='tissue'&d$fold=='1')))
 col<-c(Null=NULL_COL,`M only`=LAYER_COL[['M']],`RNA only`=LAYER_COL[['RNA']],`P only`=LAYER_COL[['P']])
 d<-d%>%mutate(col=col[as.character(arm)],fill=ifelse(scenario=='new_cell',col,'white'))
 # direct fold labels at the right end of each fold line (spread so they do not touch)
 fl<-d%>%filter(arm=='P only')%>%mutate(lab=paste0('F',fold),ly=y)
 for(s in levels(fl$sc)){k<-fl$sc==s&fl$y<BRK;fl$ly[k]<-f2_spread(fl$y[k],.028)}
 # facet limits (free y): new cell lines 0.85-1.105; held-out tissue 0.815-1.255 with the break at 1.15-1.18
 lim<-tibble::tibble(sc=factor(F2_SC,levels=F2_SC),x=1,y=c(.85,.815),y2=c(1.105,1.255))
 p<-ggplot(d,aes(x,y))+
  geom_blank(data=lim,aes(x,y))+geom_blank(data=lim,aes(x,y2))+
  geom_line(aes(group=interaction(sc,fold)),colour=F2_FOLDLINE,linewidth=LW_THIN)+
  geom_segment(data=m,aes(x=a-HW,xend=a+HW,y=score,yend=score,colour=col[as.character(arm)]),linewidth=LW_LINE*2.6,lineend='butt')+
  geom_point(aes(colour=col,fill=fill),shape=21,size=PT_SMALL+.25,stroke=.45)+
  geom_text(data=fl,aes(x=XR+.05,y=ly,label=lab),hjust=0,size=6/.pt,family=CELL_FONT,colour=F2_FOLDLAB)+
  facet_wrap(~sc,scales='free_y')+scale_colour_identity()+scale_fill_identity()+
  scale_x_continuous(breaks=1:4,labels=lab,expand=c(0,0))+
  scale_y_continuous(breaks=function(l)if(l[2]>1.2)c(.9,1,1.1,tr(1.95))else c(.9,1,1.1),
                     labels=function(b)formatC(itr(b),format='f',digits=2),expand=c(0,0))+
  coord_cartesian(xlim=c(XL,XR),clip='off')+
  labs(x=NULL,y='sRMSE')+BASE+
  theme(panel.spacing.x=unit(5,'mm'),plot.margin=margin(5,3.5,1,1.5,'mm'),axis.text.x=element_text(margin=margin(t=.8,unit='mm')))
 # ---- axis break on the held-out tissue y axis, drawn in mm over the axis line (left edge of that facet's panel)
 opened<-if(dev.cur()==1){pdf(NULL);TRUE}else FALSE
 bt<-ggplot_build(p);g<-ggplotGrob(p);if(opened)dev.off()
 k<-which(levels(d$sc)==F2_SC[['tissue']]);yr<-bt$layout$panel_params[[k]]$y.range
 yn<-((BRK+UPY)/2-yr[1])/diff(yr)
 pl<-g$layout[grepl('^panel',g$layout$name),];pl<-pl[order(pl$l),][k,]
 mm<-function(v)unit(v,'mm');Y<-function(o)unit(yn,'npc')+mm(o)
 brk<-gTree(children=gList(
  rectGrob(x=unit(0,'npc'),y=unit(yn,'npc'),width=mm(1.2),height=mm(1.2),gp=gpar(fill='white',col=NA)),
  segmentsGrob(unit(0,'npc')-mm(1.1),Y(-.6-.55),unit(0,'npc')+mm(1.1),Y(-.6+.55),gp=gpar(col=INK,lwd=.5,lineend='butt')),
  segmentsGrob(unit(0,'npc')-mm(1.1),Y(.6-.55),unit(0,'npc')+mm(1.1),Y(.6+.55),gp=gpar(col=INK,lwd=.5,lineend='butt'))),name='b_axis_break')
 g<-gtable::gtable_add_grob(g,brk,t=pl$t,l=pl$l,b=pl$b,r=pl$r,clip='off',z=Inf,name='b_axis_break')
 list(plot=p,grob=g)}

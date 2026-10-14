# Figure 6E (fix round 2, 2026-10-10) - panel recovery R(k) on the sealed set as a line plot:
# R(k) = gain(M + k-molecule panel) / gain(saved teacher; M + measured P values for protein panels, M + measured RNA
# for gene panels). Final polish (2026-10-11): axis titles as in Figures 4 and 5 ('Recovery fraction R(k)', 'Panel size,
# k (measured molecules)'; were 'Panel recovery R(k)', 'Panel size k'). One facet per endpoint (the only facet strips on the page). Supervised panels: filled circles joined
# by solid lines, with 95 % grouped-bootstrap intervals; random panels (mean of five seeds, k = 16 and 32): squares,
# 35 % fill, with their intervals and no joining line. Within each k, from left to right: random protein, supervised
# protein, supervised gene, random gene, 2.3 mm apart. The pre-registered criterion R >= 0.5 (k = 32 and 64; items P5
# and P8) is a light band from k = 32 to k = 64 with a dashed lower edge at 0.5. The development (new-cell-line) R(32)
# and R(64) and the P5 / P8 verdicts are in G (development -> sealed), so E carries sealed values only.
# Plot top aligned with D. Source: b_panel_R.csv (R, lo, hi, criterion), verbatim.
make_E<-function(){d<-read_table('b_panel_R');stopifnot(nrow(d)==20)
 LEFT<-10.5;RIGHT<-1.6;GAPF<-2.5;xl<-c(.5,3.5)
 fw<-(SLOT_W-LEFT-RIGHT-GAPF)/2;upm<-diff(xl)/fw;dg<-2.3*upm          # 2.3 mm between neighbouring series
 off<-c(protein_random_mean=-1.5,protein_supervised=-.5,gene_supervised=.5,gene_random_mean=1.5)*dg
 d<-d%>%mutate(ep=factor(ifelse(endpoint=='drug','GDSC','CRISPR'),levels=c('GDSC','CRISPR')),
   xk=log2(k)-3,x=xk+off[paste(layer,series,sep='_')],rnd=series=='random_mean',
   colkey=paste0(layer,ifelse(rnd,'_random','')),edge=F6_COL[layer])
 sup<-d[!d$rnd,];rnd<-d[d$rnd,]
 crit<-unique(d$criterion);stopifnot(length(crit)==1,crit==.5)
 yl<-c(0,.95);bx<-c(2-2.2*dg,3+2.2*dg)
 band<-data.frame(ep=factor(c('GDSC','CRISPR'),levels=c('GDSC','CRISPR')),x0=bx[1],x1=bx[2],y0=crit,y1=yl[2])
 write_csv(d,file.path(fig_out(FIG),'panel_data','E_display.csv'))
 p<-ggplot()+
  geom_rect(data=band,aes(xmin=x0,xmax=x1,ymin=y0,ymax=y1),fill='#F4F4F4',colour=NA)+
  geom_segment(data=band,aes(x=x0,xend=x1,y=y0,yend=y0),linetype='22',linewidth=LW_LINE,colour=INK)+
  geom_line(data=sup,aes(x,R,group=layer,colour=layer),linewidth=LW_LINE)+
  geom_err(data=d,aes(x=x,ymin=lo,ymax=hi),colour=d$edge)+
  geom_point(data=rnd,aes(x,R),shape=22,fill=F6_COL[rnd$colkey],colour=rnd$edge,size=1.45,stroke=.45)+
  geom_point(data=sup,aes(x,R),shape=16,colour=sup$edge,size=1.6)+
  geom_text(data=data.frame(ep=factor('GDSC',levels=c('GDSC','CRISPR')),x=3.45,y=c(.13,.05),lab=c('Protein panels','Gene panels')),
            aes(x,y,label=lab),colour=c(F6_COL[['protein']],F6_COL[['gene']]),hjust=1,size=GEOM_TEXT_MIN)+
  scale_colour_manual(values=c(protein=F6_COL[['protein']],gene=F6_COL[['gene']]),guide='none')+
  facet_wrap(~ep,nrow=1)+
  scale_x_continuous(breaks=1:3,labels=c(16,32,64))+scale_y_continuous(breaks=seq(0,.8,.2),labels=num_minus_acc(.1))+
  coord_cartesian(xlim=xl,ylim=yl,expand=FALSE)+
  labs(x='Panel size, k (measured molecules)',y='Recovery fraction R(k)')+
  BASE+theme(legend.position='none',strip.text=element_text(size=PT_STRIP,face='bold',margin=margin(0,0,1.2,0,'mm')),panel.spacing.x=unit(GAPF,'mm'))
 f6_fix(p,left=LEFT,right=RIGHT,top=F6_TOP,bottom=9)}

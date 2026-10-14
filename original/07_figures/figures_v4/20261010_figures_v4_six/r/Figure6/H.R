# Figure 6H (fix round 2, 2026-10-10) - sealed-set gain against new-cell-line (development-fold) gain for every arm
# that has both (GDSC: M + tissue label, M + virtual RNA, M + virtual P, the two paired shuffles, M + protein graph;
# CRISPR: M + virtual RNA, M + virtual P). y = sealed gain over M only with its 95 % grouped-bootstrap interval, read
# from summary.json (analysis/<endpoint>/comparisons/<arm>_vs_M); x = development-fold gain (forest.csv dev_gain_pct;
# no interval). Colour = arm (as in B and D); circle = GDSC, triangle = CRISPR (key in A); all symbols are sealed
# estimates, so filled. Dashed line = identity. Equal aspect (same mm per % on both axes) on x -0.35..1.6 and
# y -0.45..1.45; the y axis is broken above 1.45 and the one point above it (M + tissue label, sealed 1.854) sits on a
# 4 mm compressed strip above the break. Labels: M + tissue label and Shuffle controls directly beside their points;
# the three arms in the crowded centre (M + virtual RNA, M + virtual P, M + protein graph) are named once each in the
# empty upper-left corner, each entry showing the arm colour as circle + triangle (both endpoints of that arm).
make_H<-function(){s<-read_summary_v4('Figure6');fo<-read_table('forest')%>%filter(kind=='arm',!is.na(dev_gain_pct))
 d<-do.call(rbind,lapply(seq_len(nrow(fo)),function(i){r<-fo[i,];c<-s$analysis[[r$endpoint]]$comparisons[[paste0(r$arm,'_vs_M')]]
  stopifnot(!is.null(c),abs(c$gain_pct-r$gain_pct)<1e-9)
  data.frame(endpoint=r$endpoint,arm=r$arm,dev=r$dev_gain_pct,sealed=c$gain_pct,lo=c$boot$lo,hi=c$boot$hi)}))
 key<-c('M+T'='tissue','M+vRNA'='vrna','M+vP'='vp','M+vRNA_shuf'='shuffle','M+vP_shuf'='shuffle','M+P_graph'='graph')
 d$colkey<-unname(key[d$arm]);stopifnot(!any(is.na(d$colkey)),nrow(d)==8)
 d<-d[order(d$endpoint=='crispr'),]
 xl<-c(-.35,1.6);yl<-c(-.45,1.45);ys<-c(1.62,2.82)
 LEFT<-11;TOPs<-F6_TOP4;HS<-4;GAP<-1.4;BOT<-F6_BOT4
 PHm<-SLOT_H-TOPs-HS-GAP-BOT;mmu<-PHm/diff(yl);PWm<-diff(xl)*mmu;RIGHT<-SLOT_W-LEFT-PWm   # equal aspect
 stopifnot(RIGHT>=1.5)
 up<-d$sealed>yl[2];stopifnot(sum(up)==1,d$arm[up]=='M+T')
 write_csv(d,file.path(fig_out(FIG),'panel_data','H_display.csv'))
 # colour key (upper-left corner, empty of data): swatch square + arm name, one entry per arm (both endpoints)
 kx<-xl[1]+.06;ky<-c(1.33,1.215,1.10)
 kk<-data.frame(x=kx,y=ky,colkey=c('vrna','vp','graph'),lab=c('M + virtual RNA','M + virtual P','M + protein graph'))
 sh<-d[d$colkey=='shuffle',]
 pm<-ggplot(d)+
  geom_hline(yintercept=0,colour='#E6E6E6',linewidth=LW_THIN)+geom_vline(xintercept=0,colour='#E6E6E6',linewidth=LW_THIN)+
  annotate('segment',x=xl[1],xend=yl[2],y=xl[1],yend=yl[2],linetype='22',linewidth=LW_LINE,colour=INK)+
  geom_segment(aes(x=dev,xend=dev,y=lo,yend=pmin(hi,yl[2]),colour=colkey),linewidth=LW_ERR)+
  geom_point(data=d[!up,],aes(dev,sealed,colour=colkey,shape=endpoint,size=endpoint),stroke=0)+
  geom_point(data=kk,aes(x,y,colour=colkey),shape=16,size=1.45)+
  geom_point(data=kk,aes(x+.075,y,colour=colkey),shape=17,size=1.6)+
  geom_text(data=kk,aes(x+.15,y,label=lab),hjust=0,vjust=.5,size=GEOM_TEXT_MIN,colour=INK)+
  annotate('text',x=max(sh$dev)+.11,y=-.13,label='Shuffle controls',hjust=0,vjust=.5,size=GEOM_TEXT_MIN,colour=INK)+
  annotate('segment',x=xl[1]-.035,xend=xl[1]+.035,y=yl[2]-.03,yend=yl[2]+.03,colour=INK,linewidth=LW_AXIS)+
  scale_colour_manual(values=F6_COL,guide='none')+scale_shape_manual(values=c(drug=16,crispr=17),guide='none')+
  scale_size_manual(values=c(drug=1.75,crispr=1.95),guide='none')+
  scale_x_continuous(breaks=c(0,.5,1,1.5),labels=num_minus_acc(.1))+scale_y_continuous(breaks=c(0,.5,1),labels=num_minus_acc(.1))+
  coord_cartesian(xlim=xl,ylim=yl,expand=FALSE,clip='off')+
  labs(x='Gain over M only (%), new cell lines',y='Gain over M only (%), sealed')+
  BASE+theme(legend.position='none',plot.margin=margin(0,0,0,0,'mm'))
 du<-d[up,]
 ps<-ggplot(du)+
  geom_segment(aes(x=dev,xend=dev,y=ys[1],yend=hi,colour=colkey),linewidth=LW_ERR)+
  geom_point(aes(dev,sealed,colour=colkey),shape=16,size=1.75,stroke=0)+
  annotate('text',x=du$dev-.07,y=du$sealed,label='M + tissue label',hjust=1,vjust=.5,size=GEOM_TEXT_MIN,colour=INK)+
  annotate('segment',x=xl[1]-.035,xend=xl[1]+.035,y=ys[1]-.12,yend=ys[1]+.12,colour=INK,linewidth=LW_AXIS)+
  scale_colour_manual(values=F6_COL,guide='none')+
  scale_x_continuous(breaks=NULL)+scale_y_continuous(breaks=2,labels=num_minus_acc(.1))+
  coord_cartesian(xlim=xl,ylim=ys,expand=FALSE,clip='off')+
  labs(x=NULL,y=NULL)+
  BASE+theme(axis.line.x=element_blank(),axis.ticks.x=element_blank(),axis.text.x=element_blank(),legend.position='none',plot.margin=margin(0,0,0,0,'mm'))
 gs<-fix_panel_margins(ps,left=LEFT,right=RIGHT,top=TOPs,bottom=GAP/2,id='H');gm<-fix_panel_margins(pm,left=LEFT,right=RIGHT,top=GAP/2,bottom=BOT,id='H')
 g<-cowplot::plot_grid(gs,gm,ncol=1,rel_heights=c(TOPs+HS+GAP/2,GAP/2+PHm+BOT))
 list(plot=pm,grob=g)}

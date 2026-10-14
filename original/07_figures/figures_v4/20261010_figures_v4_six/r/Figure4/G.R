# Figure 4G - the pre-registered N3 test itself: AURC lowering against Random ranking (probe-2 comparisons,
# g_n3_vs_random = summary.json B.<scenario>.comparisons['<arm>|<proxy>_vs_random'], re-arranged by
# prep/extract_probe2_cmp.py; u1 = the arm-matched sub-generator disagreement, the pre-registered primary proxy).
# Lollipop + pip chart. Two blocks (bold titles above: New cell lines, Held-out tissue); in each block three proxies
# (Generator disagreement, Distance in M, Oracle), two rows per proxy (upper = M + virtual RNA, lower = M + virtual P).
# Left part: lollipop from 0 to improvement_pct (= 100 x (1 - AURC[proxy] / AURC[Random]) of the five-fold means);
# head = proxy symbol of the page key (circle / triangle / diamond), arm colour for the two real proxies, Oracle in
# the oracle mid grey; filled = New cell lines, open = Held-out tissue. The x axis is broken after 5.2 % (upper segment
# 15.8-18.2 % holds the Oracle rows); Oracle stems are interrupted at the break. Black line at 0 = Random (reference; drawn per block).
# Right part: five pips per row = folds F0-F4 in order; coloured = AURC below Random in that fold (per_fold_pct > 0),
# pale grey = not. The number of coloured pips is folds_won (4, 4, 3, 4 for Generator disagreement; 5 elsewhere).
make_G<-function(){d<-read_table('g_n3_vs_random');stopifnot(nrow(d)==60,!anyNA(d$per_fold_pct))
 chk<-d%>%group_by(scenario,arm,proxy)%>%summarise(n=n(),w=sum(won),fw=unique(folds_won),.groups='drop');stopifnot(all(chk$n==5),all(chk$w==chk$fw))
 PR<-c('u1','u3','oracle');PP<-1.9;PA<-.75;PAD<-.45;GAP<-1.5
 BLK<-2*PP+PA+2*PAD+GAP
 yrow<-function(sc,px,arm)ifelse(sc=='new_cell',0,BLK)+(match(px,PR)-1)*PP+ifelse(arm=='M+vRNA',0,PA)
 m<-d%>%group_by(scenario,arm,proxy)%>%summarise(v=unique(improvement_pct),fw=unique(folds_won),.groups='drop')%>%as.data.frame()
 stopifnot(nrow(m)==12)
 m$y<-yrow(m$scenario,m$proxy,m$arm);m$col<-ifelse(m$proxy=='oracle',F4_ORC,F4_ARMCOL[m$arm]);m$fill<-ifelse(m$scenario=='new_cell',m$col,'white')
 m$sh<-c(u1=21,u3=24,oracle=23)[m$proxy]
 # broken x axis: main 0-5.2 %, upper segment 15.8-18.2 % drawn from x = 6.0
 BRK<-5.2;G0<-6.0;C0<-15.8;XHI<-18.2;tx<-function(v)ifelse(v<=BRK,v,G0+(v-C0))
 stopifnot(all(m$v>0),all(m$v<=BRK|(m$v>=C0&m$v<=XHI)))
 m$xt<-tx(m$v)
 st<-rbind(transform(m,x0=0,x1=pmin(xt,BRK)),transform(m[m$v>BRK,],x0=G0,x1=xt))
 YL<-c(-PAD,2*BLK-GAP-PAD)
 ybr<-c(sapply(c('new_cell','tissue'),function(s)yrow(s,PR,'M+vRNA')+PA/2))
 ylab<-rep(c('Generator\ndisagreement','Distance\nin M','Oracle'),2)
 tt<-data.frame(x=-.25,y=c(-PAD-.45,BLK-PAD-.45),lab=unname(F4_SC))
 z0<-data.frame(y0=c(-PAD,BLK-PAD),y1=c(2*PP+PA+PAD,BLK+2*PP+PA+PAD))   # zero (= Random) line per block, broken under the block titles
 YA<-YL[2];ax<-data.frame(x=c(-.25,G0),xend=c(BRK,tx(XHI)))
 bk<-data.frame(x=c(BRK-.12,G0-.12),xend=c(BRK+.12,G0+.12))
 p1<-ggplot(m)+
  geom_segment(data=z0,aes(x=0,xend=0,y=y0,yend=y1),linewidth=LW_THIN,colour=INK)+
  geom_segment(data=st,aes(x=x0,xend=x1,y=y,yend=y,colour=col),linewidth=LW_LINE*1.4,lineend='butt')+
  geom_point(aes(xt,y,colour=col,fill=fill,shape=sh),size=PT_LARGE-.2,stroke=.7)+
  f4_block_title(tt)+
  geom_segment(data=ax,aes(x=x,xend=xend,y=YA,yend=YA),linewidth=LW_AXIS,colour=INK,lineend='square')+
  geom_segment(data=bk,aes(x=x,xend=xend,y=YA+.3,yend=YA-.3),linewidth=LW_AXIS,colour=INK)+
  coord_cartesian(clip='off')+
  scale_colour_identity()+scale_fill_identity()+scale_shape_identity()+
  scale_x_continuous(breaks=tx(c(0,2,4,16,18)),labels=c('0','2','4','16','18'),limits=c(-.25,tx(XHI)),expand=expansion(0),oob=scales::oob_keep)+
  scale_y_reverse(breaks=ybr,labels=ylab,limits=rev(YL),expand=expansion(0),oob=scales::oob_keep)+
  labs(x='AURC lowering vs\nRandom (%)',y=NULL)+BASE+
  theme(axis.line=element_blank(),axis.ticks.y=element_blank(),axis.text.y=element_text(lineheight=.95))
 # pips: folds F0-F4 per row
 d$y<-yrow(d$scenario,d$proxy,d$arm);d$col<-ifelse(d$proxy=='oracle',F4_ORC,F4_ARMCOL[d$arm])
 d$pf<-ifelse(d$won==1,d$col,'#E5E5E5')
 p2<-ggplot(d)+
  geom_point(aes(fold,y,fill=pf,colour=pf),shape=22,stroke=0,size=1.35)+
  coord_cartesian(clip='off')+scale_fill_identity()+scale_colour_identity()+
  scale_x_continuous(breaks=0:4,labels=c('F0','','','','F4'),limits=c(-.55,4.55),expand=expansion(0))+
  scale_y_reverse(limits=rev(YL),expand=expansion(0),oob=scales::oob_keep)+
  labs(x='Folds\nwon',y=NULL)+BASE+
  theme(axis.line.y=element_blank(),axis.ticks.y=element_blank(),axis.text.y=element_blank(),axis.line.x=element_line(linewidth=LW_AXIS,colour=INK))
 for(p in list(p1,p2))cell_guard(p,'G')
 W2<-12.5
 g1<-fix_panel_margins(p1,left=15.5,right=1.5,top=8.8,bottom=11,id='G');g2<-fix_panel_margins(p2,left=1.5,right=1.5,top=8.8,bottom=11,id='G')
 write_csv(d,file.path(fig_out(FIG),'panel_data','G_pips_source.csv'))
 list(plot=p1,grob=f4_row(list(g1,g2),c(SLOT_W-W2,W2)))}

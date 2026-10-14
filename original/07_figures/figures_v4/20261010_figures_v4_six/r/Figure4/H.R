# Figure 4H - pre-registered negative N3, abstention: risk-coverage curves (left) and the per-fold rows behind them
# (right), one full-width panel. Three pre-aligned parts share plot-area tops and bottoms (top 8.8 mm, bottom 11 mm).
# Left (curves, e_riskcov_mean, five-fold means): x = coverage retained (reversed, axis line ends at 0.5); y = sRMSE of
# the retained cells relative to full coverage in the same arm and fold (%; negative = lower error = better).
# Colour = arm (page key); line type + symbol = proxy (page key): Generator disagreement (u1; solid, circles), Distance
# in M (u3; dashed, triangles), Random ranking (light grey, mean of five seeds), Oracle ranking by the true error (mid
# grey, dotted, diamonds). Only the two arm-coloured proxies are dodged (+/-0.008 on x); Random and Oracle are drawn
# undodged for both arms with M + virtual P on top (the two arms differ by <= 0.13 percentage points, so they coincide).
# Filled = New cell lines facet, open = Held-out tissue facet. Proxy names are written in the gap between the facets,
# at the New cell lines line ends (clip off). One 'Coverage retained' title centred under both facets.
# Right (per fold; the only per-fold N3 view): rows = Mean and F0-F4 within each scenario block (bold block titles),
# two lines per row (upper = M + virtual RNA, lower = M + virtual P, 1.34 mm apart). Each line: a light segment from
# Random to Oracle (the room), Random (light-grey circle), Oracle (mid-grey diamond), Generator disagreement (arm circle),
# Distance in M (arm triangle); Mean-row symbols larger. Two x facets share the rows:
#   'Error change at 50% coverage (%)' (g_cov50_fold rel_pct; Mean row = rel_pct_mean = the 0.5 end points of the curves);
#   'AURC (sRMSE)' (h_aurc_fold aurc; Mean row = aurc_mean = e_aurc), broken after 1.135: the right segment (1.44-2.05,
#   compressed 5x) holds the Held-out tissue F1 row (AURC 1.49-2.01 for every proxy and arm).
make_H<-function(){
 # ---------------- curves
 d<-read_table('e_riskcov_mean')
 d$sc<-f4_sc(d$scenario);stopifnot(!anyNA(d$sc),all(d$arm%in%names(F4_ARM)),nrow(d)==80)
 rc<-d[d$proxy%in%c('random','oracle'),]%>%group_by(scenario,proxy,coverage)%>%summarise(dd=diff(range(rel_pct_mean)),.groups='drop')
 stopifnot(max(rc$dd)<=0.13)   # Random / Oracle: arms coincide within 0.13 percentage points (max 0.127, Held-out tissue Oracle at 0.9)
 d$x<-d$coverage+ifelse(d$proxy%in%c('u1','u3'),ifelse(d$arm=='M+vRNA',-.008,.008),0)
 d$col<-ifelse(d$proxy=='random',F4_RND,ifelse(d$proxy=='oracle',F4_ORC,F4_ARMCOL[d$arm]))
 d$fill<-ifelse(d$scenario=='new_cell',d$col,'white');d$lt<-F4_PXLT[d$proxy];d$sh<-F4_PXSH_OPEN[d$proxy]
 d$g<-paste(d$arm,d$proxy)
 ord<-c(random=1,oracle=2,u1=3,u3=4);d<-d[order(d$arm=='M+vP',ord[d$proxy]),]
 bg<-d[d$proxy%in%c('random','oracle'),];fg<-d[d$proxy%in%c('u1','u3'),]
 L<-d[d$coverage==.5&d$scenario=='new_cell',]%>%group_by(sc,proxy)%>%summarise(y=mean(rel_pct_mean),.groups='drop')
 L$y<-L$y+c(random=1.6,u1=-1.0,u3=0,oracle=0)[L$proxy]
 L$lab<-c(random='Random',u1='Generator\ndisagreement',u3='Distance in M',oracle='Oracle')[L$proxy]
 X0<-1.03;X1<-.47
 ax<-data.frame(sc=factor(F4_SC,levels=F4_SC),x=X0,xend=.5)
 pc<-ggplot(d,aes(x,rel_pct_mean))+
  geom_hline(yintercept=0,linetype='12',linewidth=LW_THIN,colour=INK)+
  geom_line(data=bg,aes(group=g,colour=col,linetype=lt),linewidth=LW_LINE)+
  geom_point(data=bg,aes(colour=col,fill=fill,shape=sh),size=PT_SMALL,stroke=.45)+
  geom_line(data=fg[fg$arm=='M+vRNA',],aes(group=g,colour=col,linetype=lt),linewidth=LW_LINE*1.4)+
  geom_point(data=fg[fg$arm=='M+vRNA',],aes(colour=col,fill=fill,shape=sh),size=PT_LARGE-.35,stroke=.6)+
  geom_line(data=fg[fg$arm=='M+vP',],aes(group=g,colour=col,linetype=lt),linewidth=LW_LINE*1.4)+
  geom_point(data=fg[fg$arm=='M+vP',],aes(colour=col,fill=fill,shape=sh),size=PT_LARGE-.35,stroke=.6)+
  geom_text(data=L,aes(x=.462,y=y,label=lab),colour=INK,hjust=0,lineheight=.95,size=6/.pt,family=CELL_FONT)+
  geom_segment(data=ax,aes(x=x,xend=xend,y=-28,yend=-28),inherit.aes=FALSE,linewidth=LW_AXIS,colour=INK,lineend='square')+
  facet_grid(.~sc)+
  coord_cartesian(clip='off')+
  scale_colour_identity()+scale_fill_identity()+scale_linetype_identity()+scale_shape_identity()+
  scale_x_reverse(breaks=c(1,.9,.8,.7,.5),labels=c('1.0','0.9','0.8','0.7','0.5'),limits=c(X0,X1),expand=expansion(0),oob=scales::oob_keep)+
  scale_y_continuous(labels=num_minus,breaks=seq(-25,0,5),limits=c(-28,3),expand=expansion(0))+
  labs(x='Coverage retained',y='Error change vs\nfull coverage (%)')+BASE+
  theme(axis.line.x=element_blank(),panel.spacing.x=unit(15,'mm'))
 # ---------------- per-fold rows (shared by the two right-hand x facets)
 RW<-c('Mean','F0','F1','F2','F3','F4');YP<-c(Mean=0,F0=1.3,F1=2.3,F2=3.3,F3=4.3,F4=5.3);PAD<-.5;GAP<-1.3;OFF<-.25
 BLK<-YP[['F4']]+2*PAD+GAP
 rows<-function(z,val,mval){m<-unique(z[,c('scenario','arm','proxy',mval)]);names(m)[4]<-'v';m$fold<-'Mean'
  z$fold<-paste0('F',z$fold);z$v<-z[[val]];z<-rbind(as.data.frame(z[,c('scenario','arm','proxy','fold','v')]),as.data.frame(m[,c('scenario','arm','proxy','fold','v')]))
  z$y<-ifelse(z$scenario=='new_cell',0,BLK)+YP[z$fold]+ifelse(z$arm=='M+vRNA',-OFF,OFF);z$mean<-z$fold=='Mean'
  z$col<-ifelse(z$proxy=='random',F4_RND,ifelse(z$proxy=='oracle',F4_ORC,F4_ARMCOL[z$arm]))
  z$fill<-ifelse(z$scenario=='new_cell',z$col,'white');z$sh<-F4_PXSH_OPEN[z$proxy]
  z$sz<-ifelse(z$mean,1.4,1.1)*ifelse(z$proxy=='random',.9,1)
  stopifnot(nrow(z)==96,!anyNA(z$y),!anyNA(z$v))
  z<-z[order(ord[z$proxy],z$arm=='M+vP'),];z}
 room<-function(z)z%>%group_by(scenario,arm,fold,y)%>%summarise(lo=v[proxy=='oracle'],hi=v[proxy=='random'],.groups='drop')
 YL<-c(-PAD,2*BLK-GAP-PAD)
 ybr<-c(YP,BLK+YP);ylab<-rep(RW,2)
 rtheme<-theme(axis.line=element_blank(),axis.ticks.y=element_blank())
 # 50% coverage
 a5<-rows(read_table('g_cov50_fold'),'rel_pct','rel_pct_mean');r5<-room(a5)
 tt<-data.frame(x=-42,y=c(-PAD-.42,BLK-PAD-.42),lab=unname(F4_SC))
 ax5<-data.frame(x=-42,xend=11)
 p5<-ggplot(a5)+
  geom_vline(xintercept=0,linetype='12',linewidth=LW_THIN,colour=INK)+
  geom_segment(data=r5,aes(x=lo,xend=hi,y=y,yend=y),colour=F4_ROOM,linewidth=LW_LINE)+
  geom_point(aes(v,y,colour=col,fill=fill,shape=sh,size=sz),stroke=.5)+
  f4_block_title(tt)+
  geom_segment(data=ax5,aes(x=x,xend=xend,y=YL[2],yend=YL[2]),linewidth=LW_AXIS,colour=INK,lineend='square')+
  coord_cartesian(clip='off')+
  scale_colour_identity()+scale_fill_identity()+scale_shape_identity()+scale_size_identity()+
  scale_x_continuous(breaks=c(-40,-20,0),labels=num_minus,limits=c(-42,11),expand=expansion(0))+
  scale_y_reverse(breaks=ybr,labels=ylab,limits=rev(YL),expand=expansion(0),oob=scales::oob_keep)+
  labs(x='Error change at\n50% coverage (%)',y=NULL)+BASE+rtheme
 # AURC, broken shared axis
 aa<-rows(read_table('h_aurc_fold'),'aurc','aurc_mean')
 BRK<-1.135;G0<-1.155;C0<-1.44;CF<-5;tx<-function(v)ifelse(v<=BRK,v,G0+(v-C0)/CF)
 stopifnot(all(aa$v<=BRK|aa$v>=C0),all(aa$v[aa$v>BRK]<=2.05),all(aa$v[aa$mean]<=BRK))
 aa$xt<-tx(aa$v);ra<-room(transform(aa,v=xt))
 TB<-c(.8,.9,1.0,1.1,1.5,2.0)
 axa<-data.frame(x=c(.74,G0-.002),xend=c(BRK+.002,tx(2.05)))
 bka<-data.frame(x=c(BRK-.006,G0-.006),xend=c(BRK+.006,G0+.006))
 pa<-ggplot(aa)+
  geom_segment(data=ra,aes(x=lo,xend=hi,y=y,yend=y),colour=F4_ROOM,linewidth=LW_LINE)+
  geom_point(aes(xt,y,colour=col,fill=fill,shape=sh,size=sz),stroke=.5)+
  geom_segment(data=axa,aes(x=x,xend=xend,y=YL[2],yend=YL[2]),linewidth=LW_AXIS,colour=INK,lineend='square')+
  geom_segment(data=bka,aes(x=x,xend=xend,y=YL[2]+.3,yend=YL[2]-.3),linewidth=LW_AXIS,colour=INK)+
  coord_cartesian(clip='off')+
  scale_colour_identity()+scale_fill_identity()+scale_shape_identity()+scale_size_identity()+
  scale_x_continuous(breaks=tx(TB),labels=c('0.8','0.9','1.0','','1.5','2.0'),limits=c(.74,tx(2.05)),expand=expansion(0))+
  scale_y_reverse(limits=rev(YL),expand=expansion(0),oob=scales::oob_keep)+
  labs(x='AURC (sRMSE)',y=NULL)+BASE+rtheme+theme(axis.text.y=element_blank())
 for(p in list(pc,p5,pa))cell_guard(p,'H')
 write_csv(a5,file.path(fig_out(FIG),'panel_data','H_cov50_source.csv'));write_csv(aa,file.path(fig_out(FIG),'panel_data','H_aurc_source.csv'))
 W<-c(96,40,36);stopifnot(sum(W)==SLOT_W)
 g1<-fix_panel_margins(pc,top=8.8,bottom=11,id='H')
 g2<-fix_panel_margins(p5,left=8,right=1.5,top=8.8,bottom=11,id='H');g3<-fix_panel_margins(pa,left=2.5,right=2,top=8.8,bottom=11,id='H')
 list(plot=pc,grob=f4_row(list(g1,g2,g3),W))}

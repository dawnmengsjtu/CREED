# Figure 3D (hero; <- v4 3B merged with the right half of v4 3D) - per-drug, per-fold delta sMSE (M only - arm) for the 20
# coverage-selected drugs, for all four added-layer arms (M + virtual RNA, M + virtual P, M + measured RNA, M + measured P)
# in both scenarios: 20 rows x 40 columns (scenario > arm > fold F0-F4). Signed purple-green scale (PRGn stops), symmetric
# at the largest |delta| among the virtual-arm tiles (0.36), so no virtual tile is clamped; measured-arm tiles beyond it are
# drawn in the end colour with a small white centre dot. Fix round: the ramp is square-root spaced (signed sqrt transform;
# legend breaks +/-0.05, 0.15, 0.36) so the small virtual-arm values (0.02-0.1) are visible; the column header is two-level
# ('M + virtual' / 'M + measured' spanning RNA | P), with RNA / P in the arm colour. Marginal dot strip above each column =
# percentage of all evaluated drugs (295 / 281-285 per fold, drug_wins.csv) with lower sMSE than under M only, in the arm
# colour and symbol (the symbols are keyed in A).
# Fix round 2 (2026-10-10): the off-scale handling is keyed in the figure - a small end-colour tile with the white centre
# dot sits under the colour bar, labelled '> 0.36' (all 14 clipped tiles are measured-arm tiles above +0.36; none below
# -0.36). It is a one-entry legend drawn by a data-free layer with a custom key glyph (draw_key_f3clip).
# 2026-10-11 (audit A4-13): the key's white dot is a pch-16 point drawn exactly like the tile dots (geom_point size .55,
# default stroke .5), and the RNA / P header text layer no longer contributes a key (its draw_key_text 'a' glyph was
# printed over the dot and made it look split).
F3_CLIP_PT<-.55
draw_key_f3clip<-function(data,params,size)grobTree(rectGrob(gp=gpar(fill=data$colour,col=NA)),
  pointsGrob(x=.5,y=.5,pch=16,gp=gpar(col='white',fill='white',fontsize=F3_CLIP_PT*.pt+.5*.stroke/2,lwd=.5*.stroke/2)))
F3_PRGN<-c(F3_DIV[['low']],'#C2A5CF',F3_DIV[['mid']],'#A6DBA0',F3_DIV[['high']])   # ColorBrewer PRGn stops; no blue (RNA colour)
F3_SSQRT<-scales::new_transform('signed_sqrt',function(x)sign(x)*sqrt(abs(x)),function(x)sign(x)*x^2)
make_D<-function(){arms<-c('M+vRNA','M+vP','M+RNA','M+P')
 sel<-read_pd(FIG,'drug_selection')%>%arrange(rank);stopifnot(nrow(sel)==20)
 d<-read_pd(FIG,'drug_deltas')%>%filter(arm%in%arms)%>%semi_join(sel,by='drug_id');stopifnot(nrow(d)==800,!anyNA(d$delta))
 lim<-max(abs(d$delta[d$arm%in%c('M+vRNA','M+vP')]));stopifnot(abs(lim-0.3615455)<1e-6)
 gA<-.28;gS<-1.0
 colx<-function(sc,arm,fold)(sc=='tissue')*(20+3*gA+gS)+(match(arm,arms)-1)*(5+gA)+fold+1
 d<-d%>%mutate(x=colx(scenario,arm,fold),y=21-match(drug_id,sel$drug_id),clamp=abs(delta)>lim,fillv=pmax(pmin(delta,lim),-lim))
 stopifnot(sum(d$clamp)==14,all(d$delta[d$clamp]>lim),all(d$arm[d$clamp]%in%c('M+RNA','M+P')))   # off-scale key '> 0.36' covers every clipped tile
 # marginal strip: pct 55-90 mapped to y 21.55-25.15
 S0<-21.2;S1<-24.9;pm<-function(p)S0+(p-57)/33*(S1-S0)
 w<-read_pd(FIG,'drug_wins')%>%filter(arm%in%arms)%>%mutate(x=colx(scenario,arm,fold),pct=100*fraction,yy=pm(pct));stopifnot(nrow(w)==40,all(w$pct>57&w$pct<90))
 HCOL<-c(`M+vRNA`=LAYER_COL[['RNA']],`M+vP`=LAYER_COL[['P']],`M+RNA`=F3_MEAS,`M+P`=F3_MEAS)
 blk<-expand.grid(arm=arms,scenario=names(F3_SC),stringsAsFactors=FALSE)%>%mutate(x0=colx(scenario,arm,0)-.5,x1=colx(scenario,arm,4)+.5,xm=(x0+x1)/2,
   lab=ifelse(arm%in%c('M+vRNA','M+RNA'),'RNA','P'),col=HCOL[arm])
 grp<-expand.grid(grp=c('virtual','measured'),scenario=names(F3_SC),stringsAsFactors=FALSE)%>%
   mutate(a0=ifelse(grp=='virtual','M+vRNA','M+RNA'),a1=ifelse(grp=='virtual','M+vP','M+P'),x0=colx(scenario,a0,0)-.5,x1=colx(scenario,a1,4)+.5,xm=(x0+x1)/2,
   lab=ifelse(grp=='virtual','M + virtual','M + measured'))
 scb<-blk%>%group_by(scenario)%>%summarise(x0=min(x0),x1=max(x1),.groups='drop')%>%mutate(xm=(x0+x1)/2,lab=F3_SC[scenario])
 xb<-sort(unique(d$x));xl<-rep(paste0('F',0:4),8)
 yb<-c(1:20,pm(c(60,75,90)));yl<-c(rev(sel$name),'60','75','90')
 YA<-25.4;YG<-26.62;YS<-27.86   # member (RNA / P) / group ('M + virtual') / scenario header baselines
 p<-ggplot(d)+
  geom_tile(aes(x,y,fill=fillv),colour='white',linewidth=.18)+
  geom_point(data=filter(d,clamp),aes(x,y),shape=16,size=F3_CLIP_PT,colour='white')+
  # strip guides: faint reference lines at 60 / 75 / 90 per arm block, then the per-fold dots
  geom_segment(data=tidyr::crossing(blk,p=c(60,75,90)),aes(x=x0,xend=x1,y=pm(p),yend=pm(p)),colour='#E3E3E3',linewidth=LW_THIN)+
  geom_point(data=w,aes(x,yy,colour=arm,shape=arm),size=1.15,show.legend=FALSE)+
  geom_point(data=data.frame(x=NA_real_,y=NA_real_,k='clip'),aes(x,y,colour=k),na.rm=TRUE,key_glyph=draw_key_f3clip,inherit.aes=FALSE)+
  geom_text(data=blk,aes(xm,YA,label=lab,colour=I(col)),size=GEOM_TEXT_MIN,family=CELL_FONT,vjust=0,show.legend=FALSE)+
  geom_segment(data=grp,aes(x=x0+.25,xend=x1-.25,y=YG-.14,yend=YG-.14),linewidth=LW_THIN,colour='#7F7F7F')+
  geom_text(data=grp,aes(xm,YG,label=lab),size=GEOM_TEXT_MIN,family=CELL_FONT,colour=INK,vjust=0)+
  geom_text(data=scb,aes(xm,YS,label=lab),size=PT_STRIP/.pt,fontface='bold',family=CELL_FONT,colour=INK,vjust=0)+
  geom_segment(data=scb,aes(x=x0+.1,xend=x1-.1,y=YS-.16,yend=YS-.16),linewidth=LW_THIN,colour=INK)+
  annotate('text',x=-1.05,y=(S0+S1)/2,label='Drugs with\nlower error (%)',hjust=1,vjust=.5,size=GEOM_TEXT_MIN,family=CELL_FONT,colour=INK,lineheight=.95)+
  scale_fill_gradientn(colours=F3_PRGN,values=c(0,.25,.5,.75,1),limits=c(-lim,lim),transform=F3_SSQRT,breaks=c(-.36,-.15,-.05,0,.05,.15,.36),
    labels=c('−0.36','−0.15','−0.05','0','0.05','0.15','0.36'),
    name='Δ sMSE (M only − arm)',guide=guide_colourbar(order=1,direction='vertical',barwidth=unit(2,'mm'),barheight=unit(32,"mm"),ticks.colour=INK,ticks.linewidth=LW_TICK,
    frame.colour=INK,frame.linewidth=LW_TICK,title.position='right',title.theme=element_text(angle=-90,size=PT_LEGEND,hjust=.5,family=CELL_FONT)))+
  scale_colour_manual(values=c(F3_COL[arms],clip=F3_DIV[['high']]),breaks='clip',labels=sprintf('> %.2f',floor(lim*100)/100),name=NULL,
    guide=guide_legend(order=2,theme=theme(legend.key.width=unit(2,'mm'),legend.key.height=unit(2,'mm'),legend.key.spacing.x=unit(.9,'mm'))))+
  scale_shape_manual(values=F3_SHAPE[arms],guide='none')+
  scale_x_continuous(breaks=xb,labels=xl,expand=c(0,0))+scale_y_continuous(breaks=yb,labels=yl,expand=c(0,0))+
  coord_cartesian(xlim=c(.5,max(blk$x1)),ylim=c(.5,S1+.15),clip='off')+labs(x=NULL,y=NULL)+
  BASE+legend_title_on()+theme(axis.line=element_blank(),axis.ticks=element_blank(),
   axis.text.x=element_text(size=PT_TICK,margin=margin(t=.5,unit='mm')),axis.text.y=element_text(size=PT_TICK,margin=margin(r=.8,unit='mm')),
   legend.position='right',legend.justification=c(0,.38),legend.box='vertical',legend.box.just='left',legend.spacing.y=unit(2.2,'mm'),
   legend.box.spacing=unit(1.6,'mm'),legend.margin=margin(0,0,0,0),
   plot.margin=margin(10.2,.5,.5,.6,'mm'))
 cell_guard(p,'D')
 list(plot=p,grob=fix_panel_margins(p,left=19.5,id='D'))}

# Figure 6G (fix round 2, 2026-10-10; merges the old G (recovered share) with the ratio rows of the old gauge C) - every
# pre-registered ratio item, development -> sealed, as dumbbell rows. One row per estimate; the open symbol is the
# new-cell-line (development) value, the filled symbol the sealed value, joined by a light-grey connector; the thin
# full-colour line with end caps is the sealed 95 % grouped-bootstrap interval. Circle = GDSC, triangle = CRISPR.
# Upper block, x = 'Recovered share (%)' (display rescaling of the share fraction x 100; ticks 0-60): recovered gain of
# M + virtual RNA / P over the room of M + measured RNA / P (blue / pink). GDSC rows carry the five development folds as
# light-grey ticks (per-fold shares exist for GDSC only); dashed lines at 15 and 55 = the pre-registered P4 range over the
# GDSC rows; dotted grey line = 1/3, the Figure 2 reference. The CRISPR shares are descriptive (no item).
# Lower block, x = 'R(k)': supervised 32- and 64-molecule panels (vermilion proteins, light-blue genes) for GDSC (P5) and
# CRISPR (P8); dashed line = criterion 0.5. Right-hand column: bracket, P-code with tick (confirmed) or cross (not
# confirmed), endpoint. Rows are labelled in house names, one baseline each. The k = 16 and random panels are in E.
# Sources: b_share.csv (share, lo, hi, dev_share, band_lo, band_hi), b_panel_R.csv (R, lo, hi, dev_R, criterion),
# dev_folds.csv (share_fold, GDSC), protocol.json (verdicts, band, R_criterion; verdicts asserted = d_scoreboard.csv).
make_G<-function(){v<-f6_verdicts();pr<-fromJSON(file.path(DATA,'protocol.json'));band<-unlist(pr$band)*100;rc<-pr$R_criterion
 bs<-read_table('b_share');br<-read_table('b_panel_R');fd<-read_table('dev_folds')%>%filter(quantity=='share')
 stopifnot(all(abs(bs$band_lo*100-band[1])<1e-9),all(abs(bs$band_hi*100-band[2])<1e-9),rc==.5,all(br$criterion==rc))
 LBL<-17.6;RC<-9.6;PW<-SLOT_W-LBL-RC;TOP<-F6_TOP4;BOT<-F6_BOT4;AX<-7.3;GAP<-1.3;CAP<-.95
 nuS<-4.4;nuR<-8.4;u<-(SLOT_H-TOP-AX-GAP-BOT)/(nuS+nuR)              # mm per row (shared by both blocks)
 lat<-function(n,grp){y<-numeric(n);cur<-0;for(i in seq_len(n)){if(i>1&&grp[i]!=grp[i-1])cur<-cur-.4;y[i]<-cur-.5;cur<-cur-1};y}
 # ---- share rows (x 100 for display)
 S<-bs%>%mutate(ep=ifelse(endpoint=='drug','GDSC','CRISPR'),ord=match(paste(endpoint,layer),c('drug RNA','drug P','crispr RNA','crispr P')))%>%arrange(ord)%>%
  transmute(ep,layer,label=ifelse(layer=='RNA','M + virtual RNA','M + virtual P'),colkey=ifelse(layer=='RNA','vrna','vp'),
            dev=dev_share*100,est=share*100,lo=lo*100,hi=hi*100,shp=ifelse(ep=='GDSC','g','c'))
 S$y<-lat(4,S$ep)
 fdS<-fd%>%mutate(ep=ifelse(endpoint=='drug','GDSC','CRISPR'))%>%left_join(S%>%dplyr::select(ep,layer,y),by=c('ep','layer'))%>%mutate(x=value*100)
 stopifnot(nrow(fdS)==10,all(fdS$ep=='GDSC'),!any(is.na(fdS$y)))
 # ---- R(k) rows: supervised k = 32, 64
 R<-br%>%filter(series=='supervised',k%in%c(32,64))%>%mutate(ep=ifelse(endpoint=='drug','GDSC','CRISPR'),
   ord=match(endpoint,c('drug','crispr'))*10+match(layer,c('protein','gene'))*2+(k==64))%>%arrange(ord)%>%
  transmute(ep,layer,k,label=paste(k,ifelse(layer=='protein','proteins','genes')),colkey=layer,dev=dev_R,est=R,lo,hi,shp=ifelse(ep=='GDSC','g','c'))
 stopifnot(nrow(R)==8,!any(is.na(R$dev)))
 R$y<-lat(8,R$ep)
 write_csv(bind_rows(S%>%mutate(block='share_pct'),R%>%mutate(block='R')),file.path(fig_out(FIG),'panel_data','G_dumbbell_display.csv'))
 write_csv(fdS,file.path(fig_out(FIG),'panel_data','G_share_folds_display.csv'))
 grp<-function(d,item){d%>%group_by(ep)%>%summarise(y0=min(y)-.42,y1=max(y)+.42,yc=mean(y),.groups='drop')%>%mutate(item=item[ep])}
 mk<-function(d,xlim,breaks,acc,xt,refs,items,folds=NULL,top,bot,extra=list()){nu<- -min(d$y)+.5;upm<-diff(xlim)/PW;uy<-1/u
  d$col<-F6_COL[d$colkey];x_lab<-xlim[1]-(LBL-.4)*upm
  gi<-grp(d,items);gi$id<-ifelse(is.na(gi$item),'',f6_item(gi$item));gi$ok<-ifelse(is.na(gi$item),NA,unname(v[gi$item]))
  xb<-xlim[2]+1.0*upm;xt0<-xlim[2]+1.8*upm
  two<-!is.na(gi$item)                               # P-code + glyph on line 1, endpoint on line 2
  gl<-do.call(rbind,lapply(which(two),function(i)f6_glyph_segs(xt0+4.3*upm,gi$yc[i]+1.05*uy,gi$ok[i],upm,uy)))
  tx<-rbind(data.frame(x=xt0,y=gi$yc[two]+1.05*uy,lab=gi$id[two],bold=TRUE),
            data.frame(x=xt0,y=ifelse(two,gi$yc-1.15*uy,gi$yc),lab=gi$ep,bold=FALSE))
  p<-ggplot(d)+refs+extra+
   geom_segment(aes(x=dev,xend=est,y=y,yend=y),colour='#D6D6D6',linewidth=2.1/.pt,lineend='butt')+
   (if(!is.null(folds))geom_segment(data=folds,aes(x=x,xend=x,y=y-.5/u*.9,yend=y+.5/u*.9),colour='#8C8C8C',linewidth=.55/.pt,inherit.aes=FALSE))+
   geom_errorbar(aes(xmin=lo,xmax=hi,y=y),orientation='y',width=CAP/u,colour=d$col,linewidth=LW_ERR)+
   geom_point(aes(dev,y,shape=shp),fill='white',colour=d$col,size=ifelse(d$shp=='c',1.55,1.45),stroke=.55)+
   geom_point(aes(est,y,shape=shp),fill=d$col,colour=d$col,size=ifelse(d$shp=='c',1.7,1.6),stroke=.3)+
   geom_text(aes(x=x_lab,y=y,label=label),hjust=0,vjust=.5,size=GEOM_TEXT_MIN,colour=INK)+
   geom_segment(data=gi,aes(x=xb,xend=xb,y=y0,yend=y1),colour=F6_KEY,linewidth=LW_THIN,inherit.aes=FALSE)+
   geom_text(data=tx[tx$bold,],aes(x,y,label=lab),hjust=0,vjust=.5,size=F6_GROUP_PT/.pt,fontface='bold',colour=INK,inherit.aes=FALSE)+
   geom_text(data=tx[!tx$bold,],aes(x,y,label=lab),hjust=0,vjust=.5,size=GEOM_TEXT_MIN,colour=INK,inherit.aes=FALSE)+
   (if(!is.null(gl))geom_segment(data=gl,aes(x=x,y=y,xend=xe,yend=ye),colour=F6_KEY,linewidth=.32,lineend='round',inherit.aes=FALSE))+
   scale_shape_manual(values=c(g=21,c=24),guide='none')+
   scale_x_continuous(breaks=breaks,labels=num_minus_acc(acc))+scale_y_continuous(breaks=NULL)+
   coord_cartesian(xlim=xlim,ylim=c(-nu,0),expand=FALSE,clip='off')+
   labs(x=xt,y=NULL)+
   BASE+theme(axis.line.y=element_blank(),axis.ticks.y=element_blank(),axis.text.y=element_blank(),axis.title.y=element_blank(),legend.position='none',
              plot.margin=margin(0,0,0,0,'mm'),axis.title.x=element_text(margin=margin(t=.8,unit='mm')))
  list(p=p,nu=nu)}
 gS<-S%>%filter(ep=='GDSC');yS<-c(min(gS$y)-.5,max(gS$y)+.5)
 refS<-list(annotate('segment',x=100/3,xend=100/3,y=-nuS,yend=0,linetype='11',linewidth=LW_LINE,colour='#8C8C8C'),
            annotate('text',x=100/3,y=.35,label='1/3',vjust=0,size=GEOM_TEXT_MIN,colour='#6E6E6E'),
            annotate('segment',x=band,xend=band,y=yS[1],yend=yS[2],linetype='22',linewidth=LW_LINE,colour=INK))
 refR<-list(annotate('segment',x=rc,xend=rc,y=-nuR,yend=0,linetype='22',linewidth=LW_LINE,colour=INK))
 A1<-mk(S,c(0,60),seq(0,60,20),1,'Recovered share (%)',refS,c(GDSC='F4',CRISPR=NA),folds=fdS)
 A2<-mk(R,c(0,1),seq(0,1,.2),.1,'R(k)',refR,c(GDSC='F5',CRISPR='F8'))
 stopifnot(abs(A1$nu-nuS)<1e-9,abs(A2$nu-nuR)<1e-9)
 g1<-fix_panel_margins(A1$p,left=LBL,right=RC,top=TOP,bottom=AX,id='G');g2<-fix_panel_margins(A2$p,left=LBL,right=RC,top=GAP,bottom=BOT,id='G')
 h1<-TOP+nuS*u+AX;h2<-GAP+nuR*u+BOT;stopifnot(abs(h1+h2-SLOT_H)<1e-6)
 g<-cowplot::plot_grid(g1,g2,ncol=1,rel_heights=c(h1,h2))
 list(plot=A1$p,grob=g)}

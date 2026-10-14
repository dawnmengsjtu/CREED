# Figure 4E - pre-registered negative N1, per-sample acquisition with a fixed head (f_acquisition_curve, five-fold means).
# Tile heat map: rows = head and policy; the top row 'Retrained (B)' is the acquisition probe's own retrained-head static
# protein panel (the B design re-run inside the probe, so its values differ slightly from B), set apart by a gap and a
# thin rule; below it the four fixed-head policies (static, adaptive, variance, random [mean of five seeds]).
# Columns = k within scenario. Fill = R(k) on a square-root single-hue scale (white -> protein-panel vermilion); every
# tile prints its value. A thin left bracket labelled 'Fixed head' (6 pt) groups the four fixed-head policy rows; it sits
# left of their row labels, inside the label column set by 'Retrained (B)' (positions in tile units, clip off).
make_E<-function(){d<-read_table('f_acquisition_curve')
 P<-c(retrained='Retrained (B)',static='Static',adaptive='Adaptive',variance='Variance',random='Random')
 d$pol<-ifelse(d$head=='retrained','retrained',d$policy);d$row<-factor(P[d$pol],levels=rev(P))
 d$hd<-factor(ifelse(d$head=='retrained','Retrained head','Fixed head'),levels=c('Retrained head','Fixed head'))
 d$sc<-f4_sc(d$scenario);d$kf<-factor(d$k,levels=c(8,16,32,64));stopifnot(nrow(d)==40,!anyNA(d$row),!anyNA(d$sc))
 d$lab<-f4_lab(d$R,2);d$tc<-ifelse(d$R>.35,'white',INK)
 # bracket in tile units: x = 0 is the left tile edge minus 0.5; 1 unit = one tile (~6.1 mm in the 68-mm slot)
 BX<- -1.2;br<-data.frame(hd=factor('Fixed head',levels=levels(d$hd)),sc=factor(F4_SC[1],levels=F4_SC))
 brs<-rbind(cbind(br,x=BX,xend=BX,y=.62,yend=4.38),cbind(br,x=BX,xend=BX+.12,y=.62,yend=.62),cbind(br,x=BX,xend=BX+.12,y=4.38,yend=4.38))
 brt<-cbind(br,x=BX-.2,y=2.5,lab='Fixed head')
 rule<-merge(data.frame(hd=factor('Retrained head',levels=levels(d$hd))),data.frame(sc=factor(F4_SC,levels=F4_SC)))
 p<-ggplot(d,aes(kf,row))+
  geom_tile(aes(fill=R),colour='white',linewidth=.5)+
  geom_text(aes(label=lab,colour=tc),size=6/.pt,family=CELL_FONT)+
  geom_hline(data=rule,aes(yintercept=.5-.17),linewidth=LW_THIN,colour=INK)+
  geom_segment(data=brs,aes(x=x,xend=xend,y=y,yend=yend),inherit.aes=FALSE,linewidth=LW_THIN,colour=INK,lineend='butt')+
  geom_text(data=brt,aes(x=x,y=y,label=lab),inherit.aes=FALSE,angle=90,vjust=0,size=6/.pt,family=CELL_FONT,colour=INK)+
  facet_grid(hd~sc,scales='free_y',space='free_y')+
  coord_cartesian(xlim=c(.5,4.5),expand=FALSE,clip='off')+
  scale_colour_identity()+
  scale_fill_gradient(low='#FFFFFF',high=F4_PROT,limits=c(0,.75),trans='sqrt',guide='none')+
  scale_x_discrete(expand=expansion(0))+scale_y_discrete(expand=expansion(0))+
  labs(x='Panel size, k (measured molecules)',y=NULL)+BASE+
  theme(axis.line=element_blank(),axis.ticks=element_blank(),strip.text.y=element_blank(),panel.spacing.y=unit(2.2,"mm"),panel.spacing.x=unit(2.5,"mm"))
 f4_fix(p,top=8.8,bottom=11)}

# Figure 2H - pairing universe as an UpSet-style ladder (fix round + polish 2026-10-10; b_ladder.csv, the table behind the
# supplementary universe panel). One row per count, grouped Source / Mapped / Paired / Extra; a dot matrix (columns M,
# RNA, P; filled in the layer colour = layer present, light gray = absent, joined by a vertical line) states which layers
# each row needs; horizontal bars carry the counts, printed verbatim (n_label / seg_label). The three-layer bar is split
# into the 421 development lines and the 64 validation + 123 sealed internal-final lines in dark / mid / light gray (one-to-one key). Pair bars use
# the pair colours (M + RNA blue, M + P pink; the 67 extra M + RNA lines are M + RNA, so the same solid blue); the RNA
# source row uses the RNA layer colour, the M mapped row the M gray. Polish: the 928 metabolomics source rows (rows, not
# lines) are a light-gray bar with a dashed M-gray outline instead of a white outlined bar. Plot top/bottom fixed
# (6.2 / 9.5 mm) to align with G.
# 2026-10-11 (audits A4-16, A5-22): key reads 'Development' / 'Validation (excluded)' / 'Internal-final (sealed)': the 64
# validation lines (values seen in earlier development, excluded from every analysis) are no longer called sealed; only
# the 123 internal-final lines are sealed. The key moved 60 count units left (swatches 700-745, text from 762) so the
# longer labels keep a right margin inside the slot.
make_H<-function(){
 b<-read_table('b_ladder')
 stopifnot(nrow(b)==10,sum(b$value[b$order==6])==608)
 b<-b%>%mutate(yy=9-order,key=case_when(order==1~'rna',order==2~'m',order==3~'m2',order==4~'mr',order==5~'mp',
   seg=='train'~'dev',seg=='validation'~'val',seg=='internal_final'~'fin',order==7~'mr2',TRUE~'zero'))
 fc<-c(rna=LAYER_COL[['RNA']],m=MISSING,m2=LAYER_COL[['M']],mr=ARM_COL_CELL[['m_vrna']],mp=ARM_COL_CELL[['m_vp']],
       dev='#525252',val='#969696',fin='#D9D9D9',mr2=ARM_COL_CELL[['m_vrna']],zero='white')
 # layer membership per row (order 1-8)
 mem<-list(`1`='RNA',`2`='M',`3`='M',`4`=c('M','RNA'),`5`=c('M','P'),`6`=c('M','RNA','P'),`7`=c('M','RNA'),`8`=c('M','RNA','P'))
 DX<-c(M=-335,RNA=-232,P=-130)
 dots<-tidyr::expand_grid(order=1:8,layer=names(DX))%>%mutate(yy=9-order,x=DX[layer],on=mapply(function(o,l)l%in%mem[[as.character(o)]],order,layer),
   col=ifelse(on,LAYER_COL[layer],MISSING))
 conn<-dots%>%filter(on)%>%group_by(yy)%>%summarise(x0=min(x),x1=max(x),.groups='drop')%>%filter(x1>x0)
 grp<-tibble::tibble(lab=c('Source','Mapped','Paired','Extra'),top=c(8,6,5,2),bot=c(7,6,3,1))%>%mutate(y=(top+bot)/2)
 ends<-b%>%group_by(yy)%>%summarise(x=max(x1),lab=format(max(bar_total),big.mark=','),.groups='drop')
 seg<-b%>%filter(order==6)%>%mutate(xm=(x0+x1)/2,tc=ifelse(seg=='train','white',INK))
 key<-tibble::tibble(key=c('dev','val','fin'),lab=c('Development','Validation (excluded)','Internal-final (sealed)'),y=c(2.3,1.7,1.1))
 XL<- -395
 p<-ggplot(b)+
  geom_segment(data=conn,aes(x=x0,xend=x1,y=yy,yend=yy),colour='#7F7F7F',linewidth=LW_LINE)+
  geom_point(data=dots,aes(x,yy,colour=col),size=PT_LARGE,shape=16)+
  geom_rect(data=b%>%filter(key!='m'),aes(xmin=x0,xmax=x1,ymin=yy-.34,ymax=yy+.34,fill=key),colour=NA)+
  geom_rect(data=b%>%filter(key=='m'),aes(xmin=x0,xmax=x1,ymin=yy-.34,ymax=yy+.34,fill=key),colour=LAYER_COL[['M']],linetype='22',linewidth=LW_LINE)+
  geom_text(data=ends,aes(x=x+18,y=yy,label=lab),hjust=0,size=6/.pt,family=CELL_FONT)+
  geom_text(data=seg,aes(x=xm,y=yy,label=as.character(value),colour=tc),size=6/.pt,family=CELL_FONT)+
  geom_rect(data=key,aes(xmin=700,xmax=745,ymin=y-.2,ymax=y+.2,fill=key))+
  geom_text(data=key,aes(x=762,y=y,label=lab),hjust=0,size=6/.pt,family=CELL_FONT)+
  geom_text(data=grp,aes(x=XL-28,y=y,label=lab),hjust=1,size=6/.pt,family=CELL_FONT)+
  geom_segment(data=grp%>%filter(top>bot),aes(x=XL,xend=XL,y=top+.3,yend=bot-.3),linewidth=LW_THIN)+
  annotate('text',x=DX,y=-.22,label=names(DX),colour=LAYER_COL[names(DX)],size=6/.pt,family=CELL_FONT,fontface='bold')+
  annotate('segment',x=0,xend=1200,y=.5,yend=.5,linewidth=LW_AXIS)+
  scale_fill_manual(values=fc,guide='none')+scale_colour_identity()+
  scale_x_continuous(breaks=c(0,500,1000),labels=c('0','500','1,000'),expand=c(0,0))+
  scale_y_continuous(expand=c(0,0))+
  coord_cartesian(xlim=c(XL-10,1200),ylim=c(.5,8.5),clip='off')+
  labs(x='Count',y=NULL)+BASE+
  theme(axis.line=element_blank(),axis.text.y=element_blank(),axis.ticks.y=element_blank(),
        axis.title.x=element_text(hjust=(600-XL+10)/(1200-XL+10)),plot.margin=margin(5,6,1,12,'mm'))
 list(plot=p,grob=fix_panel_margins(p,top=F2_BOT_TOP,bottom=F2_BOT_BOTTOM,id=ID))}

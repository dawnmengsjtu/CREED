# Figure 3A - comparison design in the Figure 1A tile vocabulary, and the one symbol key for the page (fix round,
# 2026-10-10). Rows = the arms used in B-H; columns = what the model sees in training and at deployment (gray box).
# Tiles: M = gray mini-matrix; virtual RNA / P = tinted tile with a dashed outline in the layer colour; paired shuffle =
# the same dashed tile in light gray; tissue label T = orange tile; protein sample graph = pink graph glyph (training
# only); measured RNA / P = dark-gray tile (#4D4D4D, the measured-reference colour) with a solid outline in the layer
# colour. Flow (as in Figure 2A): M -> generator -> dashed blue / pink arrows into the virtual tiles, once, at the top of
# the training column. Key: the marker used in B-H sits just left of each arm's label (gray dot M only, blue dot,
# pink dot, dark-gray square / triangle for the measured references); the last row keys the dashed training-mean null.
# The virtual and measured pairs share one group label ('M + virtual' | RNA, P) to keep the panel at 20 words.
# No dashed separator: the measured references are set off by a 2 mm gap (dashes mean only virtual and null).
# Fix round 2 (2026-10-10): (i) a 'T only' row (orange T tile in the added-layer slot, no M matrix) keys B's T-only
# comparator; to stay at 20 words 'Training-mean null' became 'Null' and 'P graph' became 'Graph' (the C family includes
# an M graph); (ii) the Training column has the same pale gray rounded box as Deployment, as the two stages in Figure 2A;
# (iii) the generator moved into the top-left corner beside the Training header (input: elbow arrow from the M tile of the
# M-only row; outputs: dashed blue / pink arrows into the virtual tiles), so the tile rows start directly under the column
# headers and the empty top-left block is gone; (iv) 1 mm between the null's dashed segment and its text.
make_A<-function(){H<-SLOT_H;tx<-function(...)dtxt(...,h=H);rc<-function(...)drect(...,h=H);ln<-function(...)dline(...,h=H)
 TL<-4.0;GAP<-.9;step<-4.65;y1<-6.6;SEP<-1.6
 Mc<-LAYER_COL[['M']];Rc<-LAYER_COL[['RNA']];Pc<-LAYER_COL[['P']]
 mini<-function(x,y){v<-c(.9,.35,.7,.5,.95,.25,.6,.8,.4);ramp<-colorRampPalette(c('#FFFFFF',Mc))(100);s<-TL/3
  lapply(seq_along(v),function(i){r<-(i-1)%/%3;c<-(i-1)%%3;rc(x+c*s,y+r*s,s*.86,s*.86,fill=ramp[round(v[i]*99)+1],col=NA)})}
 tile<-function(x,y,key){
  if(key=='')return(list())
  if(key=='M')return(mini(x,y))
  if(key=='vRNA')return(list(rc(x,y,TL,TL,fill=f3_tint(Rc,.28),col=Rc,lwd=.7,lty=2)))
  if(key=='vP')return(list(rc(x,y,TL,TL,fill=f3_tint(Pc,.28),col=Pc,lwd=.7,lty=2)))
  if(key=='shuf')return(list(rc(x,y,TL,TL,fill=f3_tint(F3_SHUF,.35),col='#9E9E9E',lwd=.7,lty=2)))
  if(key=='T')return(list(rc(x,y,TL,TL,fill=F3_TISS,col=NA)))
  if(key=='mRNA')return(list(rc(x+.2,y+.2,TL-.4,TL-.4,fill=F3_MEAS,col=Rc,lwd=1.1)))
  if(key=='mP')return(list(rc(x+.2,y+.2,TL-.4,TL-.4,fill=F3_MEAS,col=Pc,lwd=1.1)))
  if(key=='graph'){cx<-x+c(.9,TL-.9,.9,TL-.9,TL/2);cy<-y+c(.9,.9,TL-.9,TL-.9,TL/2);i0<-c(5,5,5,5,1,2,3,1);i1<-c(1,2,3,4,2,4,4,3)
   return(list(rc(x,y,TL,TL,fill='white',col=Pc,lwd=.5),
     segmentsGrob(x0=unit(cx[i0],'mm'),y0=unit(H-cy[i0],'mm'),x1=unit(cx[i1],'mm'),y1=unit(H-cy[i1],'mm'),gp=gpar(col=Pc,lwd=.6)),
     circleGrob(x=unit(cx,'mm'),y=unit(H-cy,'mm'),r=unit(.42,'mm'),gp=gpar(fill=Pc,col=NA))))}}
 # marker glyph exactly as ggplot draws it in B-H (size 1.5, stroke 0.5)
 glyph<-function(x,y,arm,size=1.5)pointsGrob(x=unit(x,'mm'),y=unit(H-y,'mm'),pch=F3_SHAPE[[arm]],
   gp=gpar(col=F3_COL[[arm]],fill=F3_COL[[arm]],fontsize=size*.pt+.5*.stroke/2,lwd=.5*.stroke/2))
 arrowl<-function(x,y,col,lty=1,lwd=.6)linesGrob(x=unit(x,'mm'),y=unit(H-y,'mm'),gp=gpar(col=col,fill=col,lwd=lwd,lty=lty,linejoin='mitre'),
   arrow=grid::arrow(length=unit(.85,'mm'),angle=28,type='closed'))
 # rows: label, group label, glyph arm, training tiles (slot 1 = M, slot 2 = added layer), deployment tiles
 rows<-list(list('M only',NA,'M','M','M'),
            list('RNA','M + virtual','M+vRNA',c('M','vRNA'),c('M','vRNA')),list('P','M + virtual','M+vP',c('M','vP'),c('M','vP')),
            list('Shuffle',NA,NA,c('M','shuf'),c('M','shuf')),list('T only',NA,NA,c('','T'),c('','T')),list('M + T',NA,NA,c('M','T'),c('M','T')),
            list('Graph',NA,NA,c('M','graph'),'M'),
            list('RNA','M + measured','M+RNA',c('M','mRNA'),c('M','mRNA')),list('P','M + measured','M+P',c('M','mP'),c('M','mP')))
 xg<-1.4;xl<-3.0;xbr<-17.1;xmg<-18.05;xml<-19.25          # glyph / label / group bracket / member glyph / member label
 wcol<-2*TL+GAP;xtr<-25.8;PADR<-1.45;xte<-xtr+wcol+PADR+.5+1.2;ctr<-xtr+wcol/2;cte<-xte+wcol/2
 ys<-y1+(seq_along(rows)-1)*step;ys[8:9]<-ys[8:9]+SEP;yc<-ys+TL/2
 ybot<-ys[9]+TL+1.2;ynull<-ybot+2.3
 stopifnot(xte+wcol+1.2<=SLOT_W-.2,ynull+1.2<=H)
 g<-list(rc(xtr-1.2,.6,wcol+1.2+PADR,ybot-.6,fill='#F2F2F2',col=NA,r=1.2),rc(xte-1.2,.6,wcol+2.4,ybot-.6,fill='#F2F2F2',col=NA,r=1.2),
         tx('Training',ctr,2.5,PT_TICK,TRUE),tx('Deployment',cte,2.5,PT_TICK,TRUE))
 # generator in the top-left corner beside the Training header: M (row 1, training) -> generator -> virtual RNA / P
 gx0<-12.2;gx1<-22.6;gy0<-3.4;gy1<-6.6;gyc<-(gy0+gy1)/2;c2<-xtr+TL+GAP+TL/2;xpk<-xtr+wcol+1.0;gxin<-17.4
 g<-c(g,list(rc(gx0,gy0,gx1-gx0,gy1-gy0,fill='white',col=INK,lwd=.6,r=.6),tx('generator',(gx0+gx1)/2,gyc,PT_TICK),
  arrowl(c(xtr-.15,gxin,gxin),c(yc[1],yc[1],gy1+.05),INK,lwd=.5),
  arrowl(c(gx1,c2,c2),c(gyc+.45,gyc+.45,ys[2]-.15),Rc,lty=2),
  arrowl(c(gx1,xpk,xpk,xtr+wcol+.15),c(gyc-.45,gyc-.45,yc[3],yc[3]),Pc,lty=2)))
 for(i in seq_along(rows)){r<-rows[[i]];y<-ys[i]
  if(is.na(r[[2]])){g<-c(g,list(tx(r[[1]],xl,yc[i],PT_TICK,just='left')));if(!is.na(r[[3]]))g<-c(g,list(glyph(xg,yc[i],r[[3]])))}
  else g<-c(g,list(glyph(xmg,yc[i],r[[3]]),tx(r[[1]],xml,yc[i],PT_TICK,just='left')))
  for(j in seq_along(r[[4]]))g<-c(g,tile(xtr+(j-1)*(TL+GAP),y,r[[4]][j]))
  for(j in seq_along(r[[5]]))g<-c(g,tile(xte+(j-1)*(TL+GAP),y,r[[5]][j]))}
 for(k in list(2:3,8:9))g<-c(g,list(tx(rows[[k[1]]][[2]],xl,mean(yc[k]),PT_TICK,just='left'),
   ln(xbr,ys[k[1]]+.2,xbr,ys[k[2]]+TL-.2,col='#7F7F7F',lwd=.5)))
 # training-mean null key: short black dashed segment (as in E and H), 1 mm clear of its text
 g<-c(g,list(segmentsGrob(x0=unit(xg-1.1,'mm'),y0=unit(H-ynull,'mm'),x1=unit(xl-1.0,'mm'),y1=unit(H-ynull,'mm'),gp=gpar(col=INK,lwd=.5,lty='22')),
  tx('Null',xl,ynull,PT_TICK,just='left')))
 dgroup(g)}

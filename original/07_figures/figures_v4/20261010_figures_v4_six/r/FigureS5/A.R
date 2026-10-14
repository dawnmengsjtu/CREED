# S5 A: minimal transfer-design schematic (<= 25 words).
# Left: leave-one-cohort-out grid (rows = the three transfer tasks; light cohort tiles = training cohorts, solid
# cohort tile = held-out cohort). Right: the two transfer directions applied to the held-out cohort, with Figure 1A's line
# convention (solid = measured input layer, dashed = virtual / predicted target layer; key at top right as in Figure 1A).
# Source-only shift correction re-centres each patient's measured input (source) layer before prediction, using source
# values only: its connector ends on the two solid input boxes (2026-10-11, resolver cptac_methods#6; it previously ended
# on the dashed target boxes). To make room for the connector on the left (x = 52, 2.4 mm right of the grid arrow's
# tip), the two box pairs moved 2.3 mm right (x1 51.8 -> 54.1, x2 67.7 -> 70).
# Dashed outlines are used for 'virtual' only (never for 'held out'), as in Figure 1A.
make_A<-function(){H<-SLOT_H;tint<-function(c)alpha_hex(c,.22)
 box<-function(x,y,w,ht,fill,col,lab,lty=1,bold=FALSE)list(drect(x,y,w,ht,fill,col,lwd=.6,h=H,lty=lty,r=.6),dtxt(lab,x+w/2,y+ht/2,6.5,bold=bold,h=H))
 arr<-function(x,y,x2,y2,col=INK)dline(x,y,x2,y2,TRUE,h=H,col=col,lwd=.6)
 g<-list()
 # line key (top right, in the letter band, as in Figure 1A)
 ly<-2.9;g<-c(g,list(dline(50.4,ly,54.9,ly,h=H,lwd=.6),dtxt('measured',55.9,ly,6.5,just='left',h=H),
  dline(70.2,ly,74.7,ly,h=H,lwd=.6,lty=2),dtxt('virtual',75.7,ly,6.5,just='left',h=H)))
 # leave-one-cohort-out grid (rows = transfer tasks)
 tw<-11.5;th<-8.4;gx<-c(5.5,19.5,33.5);gy<-c(12.5,22.4,32.3)
 for(j in 1:3){co<-CP_COHORTS[j];cc<-CP_COHORT_COL[[co]];g<-c(g,list(dtxt(co,gx[j]+tw/2,9,6.5,TRUE,h=H)))
  for(i in 1:3){held<-(i==j);g<-c(g,list(drect(gx[j],gy[i],tw,th,if(held)cc else tint(cc),col=cc,lwd=.6,h=H,r=.5)))}}
 # key under the grid: each swatch shows the three cohort colours exactly as drawn (light = training, solid = held out)
 ky<-49.25;sw<-function(x,held){w<-4.8/3;c(lapply(1:3,function(k){cc<-CP_COHORT_COL[[k]];drect(x+(k-1)*w,ky-1.6,w,3.2,if(held)cc else tint(cc),col=NA,h=H)}),
  list(drect(x,ky-1.6,4.8,3.2,NA,col='#7F7F7F',lwd=.4,h=H)))}
 g<-c(g,sw(5.5,FALSE),list(dtxt('Training',11.4,ky,6.5,just='left',h=H)),sw(25,TRUE),list(dtxt('Held out',30.9,ky,6.5,just='left',h=H)))
 # grid -> held-out cohort
 g<-c(g,list(arr(45.6,26.6,49.6,26.6)))
 R<-LAYER_COL[['RNA']];P<-LAYER_COL[['P']]
 bw<-11.5;x1<-54.1;x2<-70;yA<-15.4;yB<-28.6;bh<-7.4
 g<-c(g,list(drect(50,7,33,33.7,'#F2F2F2',col=NA,h=H,r=1.2),dtxt('Held-out cohort',52,10.4,6.5,TRUE,just='left',h=H)),
  box(x1,yA,bw,bh,tint(R),R,'RNA'),list(arr(x1+bw+.3,yA+bh/2,x2-.4,yA+bh/2)),box(x2,yA,bw,bh,'white',P,'Protein',lty=2),
  box(x1,yB,bw,bh,tint(P),P,'Protein'),list(arr(x1+bw+.3,yB+bh/2,x2-.4,yB+bh/2)),box(x2,yB,bw,bh,'white',R,'RNA',lty=2))
 # source-only shift correction -> both measured (solid) input boxes, via a connector on their left-hand side
 bx<-52;cy<-45.5
 g<-c(g,list(dline(bx,cy,bx,yA+bh/2,h=H,lwd=.6),arr(bx,yA+bh/2,x1-.25,yA+bh/2),arr(bx,yB+bh/2,x1-.25,yB+bh/2),
  drect(50,cy,33,7.5,'white',col=INK,lwd=.6,h=H,r=.6),dtxt('Source-only shift correction',66.5,ky,6.5,h=H)))
 dgroup(g)}

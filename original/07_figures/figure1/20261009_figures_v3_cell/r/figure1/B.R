# Figure 1B (v3): one paired cell-line panel. Outer ring = tissue (8 largest + Other), inner ring = outer fold.
# Independent marginal rings of the same 421 lines (angular alignment carries no tissue-to-fold meaning; see F).
make_B <- function() {
 H<-65;cx<-24;cy<-35
 ring<-function(d,keys,cols,r1,r2){start<-0;out<-list()
  for(i in seq_len(nrow(d))){end<-start+d$n[i]/sum(d$n)*2*pi;t<-seq(start,end,length.out=120)
   out<-c(out,list(polygonGrob(x=unit(c(cx+r2*sin(t),cx+r1*sin(rev(t))),'mm'),y=unit(c(H-(cy-r2*cos(t)),H-(cy-r1*cos(rev(t)))),'mm'),gp=gpar(fill=cols[[keys[i]]],col='white',lwd=.6))));start<-end}
  out}
 f<-read_table('B_fold_composition')
 g<-c(ring(TISSUE,TISSUE$group,TISSUE_COL,16.5,22),ring(f,as.character(f$fold),FOLD_COL,10.2,15.5),
  list(dtxt('421',cx,cy-1.5,10,TRUE,h=H),dtxt('cell lines',cx,cy+2.8,6,h=H)))
 x0<-52;xr<-82;pitch<-3.1
 g<-c(g,list(dtxt('Tissue',x0,9,6.5,TRUE,just='left',h=H)))
 for(i in seq_len(nrow(TISSUE))){yy<-12.6+(i-1)*pitch;g<-c(g,list(drect(x0,yy-.9,1.8,1.8,TISSUE_COL[[TISSUE$group[i]]],col=NA,h=H),dtxt(TISSUE$label[i],x0+3,yy,6,just='left',h=H),dtxt(as.character(TISSUE$n[i]),xr,yy,6,just='right',h=H)))}
 y1<-12.6+nrow(TISSUE)*pitch+2.4
 g<-c(g,list(dtxt('Outer fold',x0,y1,6.5,TRUE,just='left',h=H)))
 for(i in seq_len(nrow(f))){yy<-y1+3.6+(i-1)*pitch;g<-c(g,list(drect(x0,yy-.9,1.8,1.8,FOLD_COL[[as.character(f$fold[i])]],col=NA,h=H),dtxt(paste0('F',f$fold[i]),x0+3,yy,6,just='left',h=H),dtxt(as.character(f$n[i]),xr,yy,6,just='right',h=H)))}
 dgroup(g)
}

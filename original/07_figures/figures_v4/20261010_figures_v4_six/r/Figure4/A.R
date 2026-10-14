# Figure 4A - panel design in the Figure 1A / 2A icon vocabulary (116 x 20 mm band; the page key sits in the KEY slot).
# Left box (selection): the borrowed layer (Figure 1A RNA block behind the real protein excerpt) is ranked on training
# cells (the selected top-k bars in borrowed-layer pink, the rest light grey); the top k molecules form the panel:
# protein panel = two real protein columns in a protein-panel vermilion frame, gene panel = two columns of RNA tiles
# (flat RNA tint, as the Figure 1A RNA block; no RNA excerpt exists in the snippet table) in a gene-panel sky frame.
# Right box (deployment): M + the k measured molecules, head retrained, read out as R(k): a light track from M only
# (dashed line ending at its label) to the teacher (M + measured layer; dark-grey dot), with the protein and gene bars
# drawn part-way and labelled directly (Figure 2A's room/recovery encoding). The bars fill the box height. The left box
# starts at x = 4.2 mm so that the panel letter sits on white, as in Figure 2A. 16 words.
make_A<-function(){
 H<-SLOT_H;a<-read_table('A_real_array_snippets')
 R<-function(x,y,w,ht,...)drect(x,y,w,ht,...,h=H)
 Tt<-function(t,x,y,...)dtxt(t,x,y,...,h=H)
 Ln<-function(x,y,x2,y2,...)dline(x,y,x2,y2,...,h=H)
 Pt<-function(x,y,fill,col,d=1.6,lwd=.8)circleGrob(x=unit(x,'mm'),y=unit(H-y,'mm'),r=unit(d/2,'mm'),gp=gpar(fill=fill,col=col,lwd=lwd))
 GM<-LAYER_COL[['M']];VR<-LAYER_COL[['RNA']];VP<-LAYER_COL[['P']]
 tint<-function(c,a=.22)alpha_hex(c,a)
 st<-1.8;ts<-st*.84;iw<-function(n)n*st-st*.16;ih<-6*st-st*.16
 tiles<-function(asset,x,y,col,cols=0:4){z<-a[a$asset==asset&a$col%in%cols,];ramp<-colorRampPalette(c('#FFFFFF',col))(120)[21:120]
  z$cc<-match(z$col,cols)-1
  lapply(seq_len(nrow(z)),function(i)R(x+z$cc[i]*st,y+z$row[i]*st,ts,ts,if(is.na(z$value[i]))MISSING else ramp[1+round(z$value[i]*99)],col=NA))}
 flat<-function(x,y,fill,ncol=2)unlist(lapply(0:(ncol-1),function(j)lapply(0:5,function(i)R(x+j*st,y+i*st,ts,ts,fill,col=NA))),recursive=FALSE)
 arr<-function(x,y,x2,y2)Ln(x,y,x2,y2,arrow=TRUE,col=INK,lwd=.6)
 yI<-3.0;yL<-16.6;ym<-yI+ih/2
 g<-list(R(4.2,1.0,50.8,18.0,'#F2F2F2',col=NA,r=1.2),R(57,1.0,58,18.0,'#F2F2F2',col=NA,r=1.2))
 # ---- selection: borrowed layer -> ranked on training -> top k (protein panel | gene panel)
 xb<-6.8;o<-1.1
 g<-c(g,list(R(xb+o,yI-o,iw(5)+.5,ih+.5,tint(VR),col=VR,lwd=.5),R(xb-.25,yI-.25,iw(5)+.5,ih+.5,'white',col=NA)),
  tiles('Protein',xb,yI,VP),list(Tt("borrowed layer",xb+iw(5)/2+.3,yL,size=6.5)))
 xr<-23.8;bw<-1.15;hb<-c(10.5,9.4,8.6,6.2,5.4,4.9,4.1,3.6,3.0,2.5,2.1,1.7)
 g<-c(g,list(arr(xb+iw(5)+2.0,ym,xr-1.3,ym)),
  lapply(seq_along(hb),function(i)R(xr+(i-1)*bw,yI+ih-hb[i],.85,hb[i],if(i<=3)VP else '#D9D9D9',col=NA)),
  list(Ln(xr-.3,yI+ih+.25,xr+12*bw,yI+ih+.25,col=INK,lwd=.5),Tt('ranked on training',xr+6.8,yL,size=6.5),
       arr(xr+12*bw+0.9,ym,41.3,ym)))
 xk<-42.6;wk<-iw(2);xg<-xk+wk+2.4
 g<-c(g,list(R(xk-.5,yI-.5,wk+1,ih+1,'white',col=F4_PROT,lwd=.9)),tiles('Protein',xk,yI,VP,cols=c(1,3)),
  list(R(xg-.5,yI-.5,wk+1,ih+1,'white',col=F4_GENE,lwd=.9)),flat(xg,yI,tint(VR,.35)),
  list(Tt('top k',(xk+xg+wk)/2,yL,size=6.5)))
 # ---- deployment: M + k measured (head retrained) -> R(k) bars, labelled protein / gene; bars span the icon height
 xm<-60.2;xp<-xm+iw(5)+1.5
 g<-c(g,tiles('Metabolites',xm,yI,GM),list(R(xp-.5,yI-.5,wk+1,ih+1,'white',col=F4_PROT,lwd=.9)),
  tiles('Protein',xp,yI,VP,cols=c(1,3)),list(Tt('M + k',(xm+xp+wk)/2,yL,size=6.5)))
 x0<-87.4;x1<-104.2;yt<-3.2;yp<-7.0;yg<-10.9;bh<-2.4;L<-x1-x0
 g<-c(g,list(arr(xp+wk+1.6,ym,79.6,ym),Tt('R(k)',83.0,ym,size=6.5),
  Ln(x0,yt-.6,x0,yL-1.5,lty=3,col='#737373',lwd=.5),Ln(x1,yt-.6,x1,yt+2.9,lty=3,col='#737373',lwd=.5),
  R(x0,yt,L,2.0,'#D9D9D9',col=NA),Pt(x1,yt+1.0,F4_REF,'white',2.1,lwd=.6),Tt('teacher',x1+1.7,yt+1.0,size=6.5,just='left'),
  R(x0,yp,L*.62,bh,F4_PROT,col=NA),Tt('protein',x0+L*.62+1.0,yp+bh/2,size=6,just='left'),
  R(x0,yg,L*.56,bh,F4_GENE,col=NA),Tt('gene',x0+L*.56+1.0,yg+bh/2,size=6,just='left'),
  Tt('M only',x0,yL,size=6.5)))
 dgroup(g)}

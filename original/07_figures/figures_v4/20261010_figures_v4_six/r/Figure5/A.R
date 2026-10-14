# Figure 5A - design schematic in the Figure 1A / 2A icon vocabulary (fix round: compact box, no in-panel note).
# M (metabolite excerpt, gray ramp) feeds the cross-fitted generator, whose virtual RNA / P (dashed outlines; RNA block
# behind the shrunk protein excerpt, as in 2A) joins M as the deployed input of two response heads trained on the same
# lines and folds. Head icons, both 5 x 6 cell grids like every other icon: GDSC = the Figure 2A drug-AUC excerpt redrawn in
# the dark gray #525252 of the endpoint glyphs (vermilion is the protein panel on this page, G/H); CRISPR = a schematic grid
# in four sequential tints of bluish green #009E73 (a hue used nowhere else on the page; the tints encode no values).
# Each head is named at its right with the page-wide endpoint glyph (circle GDSC, triangle CRISPR). Bottom line: the
# page-wide scenario key in the neutral square (filled = new cell lines, open = held-out tissue), so shape means only
# endpoint and fill means only scenario.
# Final polish (2026-10-11): the GDSC head icon read as a second metabolite (M) grid, because M is also a gray grid three
# tiles away. It is now an AUC-readout icon, still in the endpoint-glyph grays and still a 5 x 6 grid: the bottom two rows
# in #4D4D4D, the top four rows in two light tints of it (a fixed schematic pattern; the tints encode no values), and a
# 0.5 pt #4D4D4D frame, so it reads as the endpoint icon and not a layer icon. Vermilion stays reserved for the protein
# panel (G, H and Figure 4).
make_A<-function(){
 H<-SLOT_H;a<-read_table('f2_A_real_array_snippets')
 kx<-SLOT_W/62;ky<-SLOT_H/50
 R<-function(x,y,w,ht,...)drect(x*kx,y*ky,w*kx,ht*ky,...,h=H)
 Ra<-function(x,y,w,ht,...)drect(x*kx,y*ky,w,ht,...,h=H)
 Tt<-function(t,x,y,...)dtxt(t,x*kx,y*ky,...,h=H)
 Ln<-function(x,y,x2,y2,...)dline(x*kx,y*ky,x2*kx,y2*ky,...,h=H)
 GM<-LAYER_COL[['M']];VR<-LAYER_COL[['RNA']];VP<-LAYER_COL[['P']];GD<-F5_KEY;CR<-F5_CRISPR_TILE
 tint<-function(c,a=.22)alpha_hex(c,a)
 tiles<-function(asset,x,y,col,step,shrink=1){z<-a[a$asset==asset,];ramp<-colorRampPalette(c('#FFFFFF',col))(120)[21:120]
  v<-if(shrink<1)0.5+shrink*(z$value-0.5)else z$value
  lapply(seq_len(nrow(z)),function(i)Ra(x+z$col[i]*step/kx,y+z$row[i]*step/ky,step*.84,step*.84,if(is.na(v[i]))MISSING else ramp[1+round(v[i]*99)],col=NA))}
 # CRISPR: schematic 5 x 6 grid, four sequential tints (fixed pattern, no values)
 crispr<-function(x,y,step){lev<-matrix(c(3,1,4,2,3, 2,4,1,3,1, 4,2,3,1,4, 1,3,2,4,2, 2,4,1,3,4, 4,2,3,1,2),nrow=6,byrow=TRUE)
  tn<-colorRampPalette(c('#FFFFFF',CR))(100)[c(28,50,74,100)]
  unlist(lapply(0:5,function(r)lapply(0:4,function(cc)Ra(x+cc*step/kx,y+r*step/ky,step*.84,step*.84,tn[lev[r+1,cc+1]],col=NA))),recursive=FALSE)}
 iw<-function(step)5*step-step*.16; ih<-function(step)6*step-step*.16
 gdsc<-function(x,y,step){lev<-matrix(c(1,2,1,1,2, 2,1,2,1,1, 1,1,2,2,1, 2,1,1,2,2),nrow=4,byrow=TRUE)
  tn<-c(alpha_hex(GD,.16),alpha_hex(GD,.34));DK<-F5_MEAS
  c(unlist(lapply(0:5,function(r)lapply(0:4,function(cc)Ra(x+cc*step/kx,y+r*step/ky,step*.84,step*.84,if(r>=4)DK else tn[lev[r+1,cc+1]],col=NA))),recursive=FALSE),
    list(Ra(x-.45/kx,y-.45/ky,iw(step)+.9,ih(step)+.9,NA,col=DK,lwd=.5/.75)))}
 pair<-function(x,y,step){o<-1.25;w<-iw(step);h<-ih(step)
  c(list(Ra(x+o/kx,y-o/ky,w+.6,h+.6,tint(VR,.10),col=VR,lwd=.5,lty=2),Ra(x-.3/kx,y-.3/ky,w+.6,h+.6,'white',col=NA)),
    tiles('Protein',x,y,VP,step,shrink=.45),list(Ra(x-.3/kx,y-.3/ky,w+.6,h+.6,NA,col=VP,lwd=.5,lty=2)))}
 arr<-function(x,y,x2,y2,lty=1,col=INK)Ln(x,y,x2,y2,arrow=TRUE,col=col,lwd=.6,lty=lty)
 glyph<-function(x,y,shape,fill,col=F5_KEY,d=1.8){
  if(shape=='circle')circleGrob(x=unit(x*kx,'mm'),y=unit(H-y*ky,'mm'),r=unit(d/2,'mm'),gp=gpar(fill=fill,col=col,lwd=.7))
  else if(shape=='square')rectGrob(x=unit(x*kx,'mm'),y=unit(H-y*ky,'mm'),width=unit(d*.86,'mm'),height=unit(d*.86,'mm'),gp=gpar(fill=fill,col=col,lwd=.7))
  else polygonGrob(x=unit(x*kx+c(-d/2,d/2,0)*1.08,'mm'),y=unit(H-y*ky+c(-d*.42,-d*.42,d*.5),'mm'),gp=gpar(fill=fill,col=col,lwd=.7))}
 s<-1.6;yI<-8;wI<-iw(s)/kx;hI<-ih(s)/ky
 xm<-4;xg1<-17;xg2<-33;xv<-38.6;yH<-30
 cM<-xm+wI/2;cV<-xv+wI/2+.6;cJ<-(cM+cV)/2;xh1<-cJ-14-wI/2;xh2<-cJ+14-wI/2
 g<-list(R(1,4.6,60,44.6,'#F2F2F2',col=NA,r=1.2))
 # row 1: M -> generator -> virtual RNA / P
 g<-c(g,tiles('Metabolites',xm,yI,GM,s),
  list(arr(xm+wI+1.2,yI+hI/2,xg1-.4,yI+hI/2),R(xg1,yI+hI/2-3,xg2-xg1,6,'white',col=INK,lwd=.6,r=.6),Tt('generator',(xg1+xg2)/2,yI+hI/2,size=6.5),
       arr(xg2+.4,yI+hI/2,xv-1,yI+hI/2)),
  pair(xv,yI,s),
  list(Tt('M',cM,yI+hI+2.8,size=6.5),Tt('virtual RNA / P',cV,yI+hI+2.8,size=6.5)))
 # joint input M + virtual RNA / P -> two heads on the same lines and folds
 y0<-yI+hI+4.3;yj<-y0+2;ys<-yj+2.4
 g<-c(g,list(Ln(cM,y0,cM,yj,col=INK,lwd=.6),Ln(cV,y0,cV,yj,col=INK,lwd=.6),Ln(cM,yj,cV,yj,col=INK,lwd=.6),Ln(cJ,yj,cJ,ys,col=INK,lwd=.6),
  Ln(xh1+wI/2,ys,xh2+wI/2,ys,col=INK,lwd=.6),arr(xh1+wI/2,ys,xh1+wI/2,yH-.8),arr(xh2+wI/2,ys,xh2+wI/2,yH-.8)))
 ym<-yH+hI/2
 g<-c(g,gdsc(xh1,yH,s),crispr(xh2,yH,s),
  list(glyph(xh1+wI+2.3,ym,'circle',F5_KEY),Tt('GDSC',xh1+wI+3.7,ym,size=6.5,just='left'),
       glyph(xh2+wI+2.3,ym,'triangle',F5_KEY),Tt('CRISPR',xh2+wI+3.7,ym,size=6.5,just='left')))
 # page-wide scenario key (neutral squares: fill = scenario only)
 yk<-45.1;k1<-10.2;k2<-34.6
 g<-c(g,list(glyph(k1,yk,'square',F5_KEY,d=1.8),Tt('New cell lines',k1+1.6,yk,size=6,just='left'),
  glyph(k2,yk,'square','white',d=1.8),Tt('Held-out tissue',k2+1.6,yk,size=6,just='left')))
 dgroup(g)}

# Figure 2A - design schematic in the Figure 1A icon vocabulary (fix round, 2026-10-10). Tile icons are the real
# column-scaled 6 x 5 excerpts used in Figure 1A (A_real_array_snippets): M = metabolite excerpt (gray ramp), measured
# RNA / P = Figure 1A's RNA block (tinted blue box) behind the protein excerpt (pink ramp), virtual RNA / P = the same
# pair with dashed outlines and the protein excerpt shrunk toward its mean (an imperfect reconstruction), drug AUC =
# the AUC excerpt (vermilion ramp, Figure 1A's endpoint colour). Training: M + measured RNA / P fit the generator, whose
# virtual RNA / P feeds the AUC head. Deployment: the three arms evenly spaced; under them the room bar (D's encoding:
# light track ending in the measured-reference dot) and the two recovery bars (virtual RNA / P colours) drawn to one
# third of the room; the page-wide scenario key (filled = new cell lines, open = held-out tissue) sits beside them.
# Designed on a 76 x 54 mm canvas; positions scale with the slot, icon tiles keep their absolute size.
make_A<-function(){
 H<-SLOT_H;a<-read_table('A_real_array_snippets')
 kx<-SLOT_W/76;ky<-SLOT_H/54
 R<-function(x,y,w,ht,...)drect(x*kx,y*ky,w*kx,ht*ky,...,h=H)
 Ra<-function(x,y,w,ht,...)drect(x*kx,y*ky,w,ht,...,h=H)            # absolute size, scaled position
 Tt<-function(t,x,y,...)dtxt(t,x*kx,y*ky,...,h=H)
 Ln<-function(x,y,x2,y2,...)dline(x*kx,y*ky,x2*kx,y2*ky,...,h=H)
 Pt<-function(x,y,fill,col,d=1.6,lwd=.8)circleGrob(x=unit(x*kx,'mm'),y=unit(H-y*ky,'mm'),r=unit(d/2,'mm'),gp=gpar(fill=fill,col=col,lwd=lwd))
 GM<-LAYER_COL[['M']];VR<-LAYER_COL[['RNA']];VP<-LAYER_COL[['P']];AUC<-OI[['vermilion']]
 tint<-function(c,a=.22)alpha_hex(c,a)
 # one 6 x 5 excerpt; x,y = top-left (design units); step = tile pitch (mm)
 tiles<-function(asset,x,y,col,step,shrink=1){z<-a[a$asset==asset,];ramp<-colorRampPalette(c('#FFFFFF',col))(120)[21:120]
  v<-if(shrink<1)0.5+shrink*(z$value-0.5)else z$value
  lapply(seq_len(nrow(z)),function(i)Ra(x+z$col[i]*step/kx,y+z$row[i]*step/ky,step*.84,step*.84,if(is.na(v[i]))MISSING else ramp[1+round(v[i]*99)],col=NA))}
 iw<-function(step)5*step-step*.16; ih<-function(step)6*step-step*.16
 # RNA / P pair: RNA block (Figure 1A) behind, protein excerpt in front; virtual = dashed outlines + shrunk excerpt
 pair<-function(x,y,step,virtual=FALSE){o<-1.25;w<-iw(step);h<-ih(step);lty<-if(virtual)2 else 1
  c(list(Ra(x+o/kx,y-o/ky,w+.6,h+.6,if(virtual)tint(VR,.10)else tint(VR),col=VR,lwd=.5,lty=lty),
         Ra(x-.3/kx,y-.3/ky,w+.6,h+.6,'white',col=NA)),
    tiles('Protein',x,y,VP,step,shrink=if(virtual).45 else 1),
    if(virtual)list(Ra(x-.3/kx,y-.3/ky,w+.6,h+.6,NA,col=VP,lwd=.5,lty=2))else list())}
 arr<-function(x,y,x2,y2,lty=1,col=INK)Ln(x,y,x2,y2,arrow=TRUE,col=col,lwd=.6,lty=lty)
 g<-list()
 # ---- training: M + measured RNA / P -> generator -> virtual RNA / P -> drug AUC head
 g<-c(g,list(R(1,2.4,74,20.6,'#F2F2F2',col=NA,r=1.2),Tt('Training',3,5.2,size=6.5,bold=TRUE,just='left')))
 s1<-1.5;yI<-8.6;xm<-5.2;xp<-15.2;xv<-47.6;xa<-63.6
 g<-c(g,tiles('Metabolites',xm,yI,GM,s1),pair(xp,yI,s1),
  list(arr(25.3,13,28.9,13),R(29.4,10,13,6,'white',col=INK,lwd=.6,r=.6),Tt('generator',35.9,13,size=6.5),arr(42.9,13,46.6,13)),
  pair(xv,yI,s1,virtual=TRUE),list(arr(57.6,13,62.6,13)),tiles('Drug AUC',xa,yI,AUC,s1),
  list(Tt('M',xm+iw(s1)/2/kx,20.3,size=6.5),Tt('RNA / P',xp+iw(s1)/2/kx+.3,20.3,size=6.5),
       Tt('virtual RNA / P',xv+iw(s1)/2/kx+.3,20.3,size=6.5),Tt('AUC',xa+iw(s1)/2/kx,20.3,size=6.5)))
 # ---- deployment: three arms, evenly spaced
 g<-c(g,list(R(1,25,74,28.4,'#F2F2F2',col=NA,r=1.2),Tt('Deployment',3,27.8,size=6.5,bold=TRUE,just='left')))
 s2<-1.25;yA<-31.2;c0<-11;c1<-37;c2<-61.6;wM<-iw(s2)/kx;wP<-(iw(s2)+1.25)/kx;gap<-.7
 g<-c(g,list(arr(35.9,16.3,c1+gap/2+wP/2-.2,yA-1.6,lty=2,col=VR)),
  tiles('Metabolites',c0-wM/2,yA,GM,s2),
  tiles('Metabolites',c1-(wM+gap+wP)/2,yA,GM,s2),pair(c1-(wM+gap+wP)/2+wM+gap,yA,s2,virtual=TRUE),
  tiles('Metabolites',c2-(wM+gap+wP)/2,yA,GM,s2),pair(c2-(wM+gap+wP)/2+wM+gap,yA,s2),
  list(Tt('M only',c0,41.2,size=6.5),Tt('M + virtual RNA / P',c1,41.2,size=6.5),Tt('M + measured RNA / P',c2,41.2,size=6.5)))
 # ---- room and recovery (gain over M only): room = M only -> M + measured; recovery drawn to one third of the room
 x0<-c0;xr<-c2;x1<-x0+(xr-x0)/3;yb<-44.6
 g<-c(g,list(Ln(x0,42.9,x0,52.2,lty=3,col='#737373',lwd=.5),Ln(xr,42.9,xr,yb-.2,lty=3,col='#737373',lwd=.5),
  R(x0,yb,xr-x0,2.2,'#D9D9D9',col=NA),Pt(xr,yb+1.1,F2_MEAS_SOLID,'white',2,lwd=.6),Tt('room',xr+2,yb+1.1,size=6.5,just='left'),
  R(x0,yb+3.3,x1-x0,1.8,VR,col=NA),Pt(x1,yb+4.2,VR,VR,1.7),R(x0,yb+6.1,x1-x0,1.8,VP,col=NA),Pt(x1,yb+7.0,VP,VP,1.7),
  Tt('recovery',x1+2.2,yb+5.6,size=6.5,just='left')))
 # ---- page-wide scenario key (filled / open), once for the whole figure
 kx0<-49;g<-c(g,list(Pt(kx0,yb+4.2,'#525252','#525252',1.7,lwd=.7),Tt('New cell lines',kx0+1.8,yb+4.2,size=6,just='left'),
  Pt(kx0,yb+7.0,'white','#525252',1.7,lwd=.7),Tt('Held-out tissue',kx0+1.8,yb+7.0,size=6,just='left')))
 dgroup(g)}

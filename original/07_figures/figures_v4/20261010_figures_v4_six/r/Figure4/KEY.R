# Figure 4 page key (unlettered KEY slot beside A, 52 x 20 mm): drawn once for the page, three titled groups (as the
# 'Tissue' / 'Outer fold' keys of Figure 1B).
# Column 1: 'Scenario' (filled = New cell lines, open = Held-out tissue; B, C, F, G, H) and 'Arm' (M + virtual RNA /
# M + virtual P; G, H). Column 2: 'Abstention proxy' (symbol + line type; G, H): Generator disagreement (u1), Distance in
# M (u3), Random ranking, Oracle ranking. Proxy glyphs are neutral here because colour = arm in G and H.
make_KEY<-function(){
 H<-SLOT_H
 Tt<-function(t,x,y)dtxt(t,x,y,size=6,just='left',h=H)
 Hd<-function(t,x,y)dtxt(t,x,y,size=6.5,bold=TRUE,just='left',h=H)
 Ln<-function(x,y,x2,col,lty=1,lwd=.9)segmentsGrob(x0=unit(x,'mm'),y0=unit(H-y,'mm'),x1=unit(x2,'mm'),y1=unit(H-y,'mm'),gp=gpar(col=col,lwd=lwd,lty=lty,lineend='butt'))
 Sh<-function(x,y,pch,fill,col,size=1.7)pointsGrob(x=unit(x,'mm'),y=unit(H-y,'mm'),pch=pch,size=unit(size,'mm'),gp=gpar(col=col,fill=fill,lwd=.9))
 y1<-c(2.2,5.3,8.3,11.6,14.7,17.7)          # column 1: title, 2 entries, title, 2 entries
 y2<-c(2.2,5.3,8.4,11.5,14.6)               # column 2: title, 4 entries
 c1s<-1.2;c1e<-4.8;c1m<-3.0;c1t<-5.8
 c2s<-22.4;c2e<-27.0;c2m<-24.7;c2t<-28.0
 g<-list(
  Hd('Scenario',c1s,y1[1]),
  Sh(c1m,y1[2],21,F4_NEU,F4_NEU),Tt('New cell lines',c1t,y1[2]),
  Sh(c1m,y1[3],21,'white',F4_NEU),Tt('Held-out tissue',c1t,y1[3]),
  Hd('Arm',c1s,y1[4]),
  Ln(c1s,y1[5],c1e,F4_VR,lwd=1.4),Sh(c1m,y1[5],21,F4_VR,F4_VR,1.5),Tt('M + virtual RNA',c1t,y1[5]),
  Ln(c1s,y1[6],c1e,F4_VP,lwd=1.4),Sh(c1m,y1[6],21,F4_VP,F4_VP,1.5),Tt('M + virtual P',c1t,y1[6]),
  Hd('Abstention proxy',c2s,y2[1]),
  Ln(c2s,y2[2],c2e,F4_NEU,lwd=1.1),Sh(c2m,y2[2],21,F4_NEU,F4_NEU,1.5),Tt('Generator disagreement',c2t,y2[2]),
  Ln(c2s,y2[3],c2e,F4_NEU,lty='42',lwd=1.1),Sh(c2m,y2[3],24,F4_NEU,F4_NEU,1.5),Tt('Distance in M',c2t,y2[3]),
  Ln(c2s,y2[4],c2e,F4_RND,lwd=1.1),Sh(c2m,y2[4],21,F4_RND,F4_RND,1.2),Tt('Random',c2t,y2[4]),
  Ln(c2s,y2[5],c2e,F4_ORC,lty='13',lwd=1.1),Sh(c2m,y2[5],23,F4_ORC,F4_ORC,1.4),Tt('Oracle',c2t,y2[5]))
 dgroup(g)}

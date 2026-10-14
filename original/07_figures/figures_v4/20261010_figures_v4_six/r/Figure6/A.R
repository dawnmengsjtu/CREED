# Figure 6A - protocol of the one-shot confirmation in the Figure 1A icon vocabulary, and the page symbol key
# (fix round 2, 2026-10-10). Development zone (open circle = development / new-cell-line values everywhere on the page):
# the 421 development lines as one open count block -> the frozen model (box with 'frozen' on its edge): M tile ->
# virtual RNA / P (dashed = virtual) -> drug AUC tile (light-orange tint of Figure 1A). The release receipt is the gate
# (a badge on the release arrow, regular type like 'Evaluate once'). Sealed zone (filled circle = sealed-set values):
# the 123 sealed lines as one filled count block -> M tile (the only input at deployment, as in Figure 1A; same x as the
# development M tile) -> 'Evaluate once' -> the two endpoints, GDSC (drug AUC tile, circle; same x as the development
# AUC tile) and CRISPR (own knocked-out grid tile: outlined cells, knocked-out cells struck through; triangle) - the
# endpoint shapes used in B, G and H. Symbol keys and count blocks in #4D4D4D (black is kept for text and arrows).
# Tile icons are the real Figure 1A excerpts (A_real_array_snippets); the CRISPR tile is a fixed icon. 13 words.
make_A<-function(){H<-SLOT_H;a<-read_table('A_real_array_snippets');g<-list()
 pr<-fromJSON(file.path(DATA,'protocol.json'));stopifnot(pr$n_train==421,pr$n_eval==123)   # counts printed in the blocks
 GM<-LAYER_COL[['M']];VR<-LAYER_COL[['RNA']];VP<-LAYER_COL[['P']];K<-F6_KEY;TXT<-6.5
 R<-function(x,y,w,ht,...)drect(x,y,w,ht,...,h=H);Tt<-function(t,x,y,...)dtxt(t,x,y,...,h=H)
 Ar<-function(x,y,x2,y2,...)dline(x,y,x2,y2,arrow=TRUE,lwd=.6,...,h=H)
 Ln<-function(x,y,x2,y2,col=K,lwd=.5)segmentsGrob(unit(x,'mm'),unit(H-y,'mm'),unit(x2,'mm'),unit(H-y2,'mm'),gp=gpar(col=col,lwd=lwd,lineend='butt'))
 Pt<-function(x,y,pch,fill=K,col=K,sz=5.6)pointsGrob(x=unit(x,'mm'),y=unit(H-y,'mm'),pch=pch,gp=gpar(col=col,fill=fill,fontsize=sz,lwd=1))
 tiles<-function(asset,x,y,col,step,shrink=1,lo='#FFFFFF'){z<-a[a$asset==asset,];ramp<-colorRampPalette(c(lo,col))(120)[21:120]
  v<-if(shrink<1)0.5+shrink*(z$value-0.5)else z$value
  lapply(seq_len(nrow(z)),function(i)R(x+z$col[i]*step,y+z$row[i]*step,step*.84,step*.84,if(is.na(v[i]))MISSING else ramp[1+round(v[i]*99)],col=NA))}
 iw<-function(s)5*s-s*.16;ih<-function(s)6*s-s*.16
 vpair<-function(x,y,s){w<-iw(s);h<-ih(s);c(list(R(x+1,y-1,w+.5,h+.5,alpha_hex(VR,.10),col=VR,lwd=.5,lty=2),R(x-.25,y-.25,w+.5,h+.5,'white',col=NA)),
   tiles('Protein',x,y,VP,s,shrink=.45),list(R(x-.25,y-.25,w+.5,h+.5,NA,col=VP,lwd=.5,lty=2)))}
 # CRISPR tile: 6 x 5 outlined cells (own outline style), knocked-out cells struck through, solid outer frame
 crispr<-function(x,y,s){ko<-data.frame(c=c(0,1,2,3,4,1,3),r=c(2,4,0,5,1,1,3));out<-list()
  for(r in 0:5)for(cc in 0:4){x0<-x+cc*s;y0<-y+r*s;out[[length(out)+1]]<-R(x0,y0,s*.84,s*.84,'white',col=K,lwd=.35)
   if(any(ko$c==cc&ko$r==r))out[[length(out)+1]]<-Ln(x0+.1,y0+s*.84-.1,x0+s*.84-.1,y0+.1,lwd=.6)}
  c(out,list(R(x-.35,y-.35,iw(s)+.7,ih(s)+.7,NA,col=K,lwd=.6)))}
 # count block: three stacked sheets; open (white, outlined) = development, filled (#4D4D4D) = sealed
 block<-function(n,x,y,open){w<-8;h<-5.2;out<-list()
  for(k in 2:0)out[[length(out)+1]]<-R(x+k*.75,y-k*.75,w,h,if(open)'white' else K,col=if(open)K else 'white',lwd=.5,r=.5)
  c(out,list(Tt(n,x+w/2,y+h/2,size=TXT,bold=TRUE,col=if(open)INK else 'white'),Tt('lines',x+w/2,y+h+2.2,size=6)))}
 s<-1.12;ZW<-SLOT_W-2      # zone width
 xm<-17.6;xv<-27.4;xa<-38.4                      # x of the M tile, virtual pair and AUC tile (development row)
 ex0<-25.7;ew<-10.8;xr<-ex0+ew/2                  # 'Evaluate once' box (sealed row); xr = x of the release arrow
 # ---- development zone
 g<-c(g,list(R(1,1,ZW,19.6,F6_BAND,col=NA,r=1.2),Tt('Development',4.4,3.7,size=TXT,bold=TRUE,just='left'),Pt(21.3,3.7,21,fill='white')))
 yc<-11.6;yI<-yc-ih(s)/2;bx0<-15.9;bx1<-xa+iw(s)+1.6;by0<-yI-1.7
 g<-c(g,block(as.character(pr$n_train),3,yc-2.2,TRUE))
 g<-c(g,list(Ar(13.2,yc,15.3,yc)),tiles('Metabolites',xm,yI,GM,s),list(Ar(xm+iw(s)+.6,yc,xv-.7,yc)),vpair(xv,yI,s),
   list(Ar(xv+iw(s)+1.9,yc,xa-.7,yc)),tiles('Drug AUC',xa,yI,F6_AUC_TINT,s),
   list(R(bx0,by0,bx1-bx0,ih(s)+3.4,NA,col=INK,lwd=.55,r=.8)))
 # 'frozen' on the top edge of the model box (knocked out of the border with the zone colour)
 fx<-(xv+xa+iw(s))/2+2.2
 g<-c(g,list(R(fx-4.3,by0-1.1,8.6,2.2,F6_BAND,col=NA),Tt('frozen',fx,by0,size=6)))
 # ---- sealed zone
 g<-c(g,list(R(1,25.2,ZW,19.4,F6_BAND,col=NA,r=1.2),Tt('Sealed',4.4,27.9,size=TXT,bold=TRUE,just='left'),Pt(13.3,27.9,21,fill=K)))
 yc2<-35.2;yT<-yc2-ih(s)/2
 g<-c(g,block(as.character(pr$n_eval),3,yc2-2.2,FALSE))
 g<-c(g,list(Ar(13.2,yc2,15.3,yc2)),tiles('Metabolites',xm,yT,GM,s),list(Tt('M',xm+iw(s)/2,yT+ih(s)+2.1,size=6)),
   list(Ar(xm+iw(s)+.6,yc2,ex0-.5,yc2),R(ex0,yc2-3.3,ew,6.6,'white',col=INK,lwd=.55,r=.8),Tt('Evaluate\nonce',xr,yc2,size=TXT),
        Ar(ex0+ew+.4,yc2,xa-.6,yc2)))
 xc<-46.3
 g<-c(g,tiles('Drug AUC',xa,yT,F6_AUC_TINT,s),crispr(xc,yT,s),
  list(Pt(xa-2.8,42.3,16,sz=5.4),Tt('GDSC',xa-1.8,42.3,size=6,just='left'),Pt(xc-1.0,42.3,17,sz=5.4),Tt('CRISPR',xc,42.3,size=6,just='left')))
 # ---- receipt gate: badge on the release arrow (frozen model -> single sealed evaluation)
 g<-c(g,list(Ar(xr,by0+ih(s)+3.4,xr,yc2-3.4),R(xr-6,20.6,12,3.6,'white',col=K,lwd=.5,r=.8),Tt('Receipt',xr,22.4,size=TXT)))
 dgroup(g)}

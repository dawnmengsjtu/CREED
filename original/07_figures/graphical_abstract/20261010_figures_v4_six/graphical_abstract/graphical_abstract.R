# CREED graphical abstract (Cell Press spec: one square panel, 1200 x 1200 px at 300 dpi = 4 in = 101.6 mm;
# Arial 8-12 pt; TIFF / PDF / JPG; reads top-left -> right -> down; no data items, text used sparingly).
# Colour contract and schematic helpers: r/cell_style.R (LAYER_COL, PANEL_COL, MISSING, drect/dline/dtxt).
# Tile icons: the real column-scaled 6 x 5 excerpts used in Figures 1A / 2A (A_real_array_snippets); RNA has no
# excerpt (Figure 1A draws it as a plain block), so its tiles are uniform (no value encoded).
# Numbers (all from the tagged master CREED_manuscript_v0.md, Abstract / Results 2, 6, 8):
#   virtual layer recovers "one fifth to one third" of the multi-layer room (dev 0.29-0.34; sealed 0.14-0.19)
#   8-64 measured proteins recovered 48-74 %; CRISPR share 30-34 % with room 40-60 % smaller
#   sealed set: 123 lines, one-shot; P1-P3 and P6-P8 confirmed, P4 and P5 not.
# Room only in the metabolite row (M -> RNA / P); RNA <-> P small room, not recovered (drawn as short unfilled stubs).
V4<-'/public/home/mengxl/workspaces/creed-research/figure_work/20261010_figures_v4_six'
GA<-file.path(V4,'graphical_abstract')
source(file.path(V4,'r','cell_style.R'))
CELL_FONT<-'Arial'                       # fontconfig: Arial -> Liberation Sans (metric-identical clone of Arial)
W<-101.6;H<-101.6                        # 4 x 4 in
snip<-read.csv(file.path(GA,'_data','A_real_array_snippets.csv'))
GM<-LAYER_COL[['M']];VR<-LAYER_COL[['RNA']];VP<-LAYER_COL[['P']];PROT<-PANEL_COL[['protein']];MEAS<-'#666666'
TXT<-9;LAB<-8.5;HEAD<-10
T<-function(t,x,y,size=TXT,bold=FALSE,col=INK,just='left')dtxt(t,x,y,size=size,bold=bold,col=col,just=just,h=H)
Rr<-function(x,y,w,ht,fill='white',col=NA,lwd=.5,lty=1,r=0)drect(x,y,w,ht,fill=fill,col=col,lwd=lwd,lty=lty,r=r,h=H)
Ln<-function(x,y,x2,y2,arrow=FALSE,col=INK,lwd=.8,lty=1)dline(x,y,x2,y2,arrow=arrow,col=col,lwd=lwd,lty=lty,h=H)
Pt<-function(x,y,d,fill,col=NA,lwd=.6)circleGrob(x=unit(x,'mm'),y=unit(H-y,'mm'),r=unit(d/2,'mm'),gp=gpar(fill=fill,col=col,lwd=lwd))
Poly<-function(x,y,col=INK,lwd=1,fill=NA)polylineGrob(x=unit(x,'mm'),y=unit(H-y,'mm'),gp=gpar(col=col,lwd=lwd,fill=fill,lineend='round',linejoin='round'))
ramp_of<-function(col)colorRampPalette(c('#FFFFFF',col))(120)[21:120]
# 6 x 5 tile stack, top-left (x, y), pitch s; asset = excerpt name or NULL (uniform tiles); shrink < 1 = virtual
tiles<-function(asset,x,y,col,s,shrink=1,uniform=.55,a=1){rp<-ramp_of(col);if(a<1)rp<-alpha_hex(rp,a);g<-list()
 if(is.null(asset)){for(i in 0:5)for(j in 0:4)g<-c(g,list(Rr(x+j*s,y+i*s,s*.84,s*.84,rp[1+round((0.5+shrink*(uniform-.5))*99)])));return(g)}
 z<-snip[snip$asset==asset,];v<-if(shrink<1)0.5+shrink*(z$value-0.5)else z$value
 lapply(seq_len(nrow(z)),function(i)Rr(x+z$col[i]*s,y+z$row[i]*s,s*.84,s*.84,if(is.na(v[i]))MISSING else rp[1+round(v[i]*99)]))}
sw<-function(s)5*s-.16*s;sh<-function(s)6*s-.16*s
outline<-function(x,y,s,col,lty)Rr(x-.5,y-.5,sw(s)+1,sh(s)+1,NA,col=col,lwd=.7,lty=lty,r=.4)
# small square mini-tile (3 x 3) for row heads
mini<-function(asset,x,y,col,s=1.05){rp<-ramp_of(col);g<-list()
 if(is.null(asset)){for(i in 0:2)for(j in 0:2)g<-c(g,list(Rr(x+j*s,y+i*s,s*.84,s*.84,rp[55])));return(g)}
 z<-snip[snip$asset==asset&snip$row<3&snip$col<3,];lapply(seq_len(nrow(z)),function(i)Rr(x+z$col[i]*s,y+z$row[i]*s,s*.84,s*.84,if(is.na(z$value[i]))MISSING else rp[1+round(z$value[i]*99)]))}
# cell-line icon: three cells with nuclei
cells<-function(x,y,k=1)list(Pt(x,y+.4*k,3.0*k,'white','#525252',.6),Pt(x+2.3*k,y-.5*k,2.6*k,'white','#525252',.6),Pt(x+1.7*k,y+1.9*k,2.3*k,'white','#525252',.6),
 Pt(x,y+.4*k,1.1*k,'#BDBDBD'),Pt(x+2.3*k,y-.5*k,.95*k,'#BDBDBD'),Pt(x+1.7*k,y+1.9*k,.85*k,'#BDBDBD'))
# dose-response icon (drug response): L axes + falling sigmoid
sigmoid<-function(x,y,w=6.4,ht=6,lty=1){xx<-seq(0,1,length.out=40);yy<-1/(1+exp(10*(xx-.5)))
 list(Poly(c(x,x,x+w),c(y,y+ht,y+ht),col=INK,lwd=.7),
  polylineGrob(x=unit(x+.6+xx*(w-1),'mm'),y=unit(H-(y+ht-.6-yy*(ht-1.6)),'mm'),gp=gpar(col=INK,lwd=1.3,lty=lty,lineend='round')))}
# room track: light track of length L from x0, measured-reference dot at the end; fills = list(col, lo, hi, dy, hh)
track<-function(x0,y,L,fills=list(),th=2.4,dot=TRUE){g<-list(Rr(x0,y-th/2,L,th,'#E6E6E6',r=th/2))
 for(f in fills){g<-c(g,list(Rr(x0,y-th/2+f$dy,L*f$lo,f$hh,f$col),Rr(x0+L*f$lo,y-th/2+f$dy,L*(f$hi-f$lo),f$hh,alpha_hex(f$col,.38))))}
 g<-c(g,list(Ln(x0,y-th/2-.7,x0,y+th/2+.7,col='#525252',lwd=.8)))
 if(dot)g<-c(g,list(Pt(x0+L,y,2.6,MEAS,'white',.6)));g}
tick<-function(x,y,col=INK)Poly(c(x-1,x-.25,x+1.15),c(y,y+.8,y-.9),col=col,lwd=1.5)
cross<-function(x,y,col='#7F7F7F')list(Ln(x-.85,y-.85,x+.85,y+.85,col=col,lwd=1.4),Ln(x-.85,y+.85,x+.85,y-.85,col=col,lwd=1.4))
padlock<-function(x,y){th<-seq(0,pi,length.out=30);list(Poly(x+2.1+1.35*cos(th),y+.2-1.6*sin(th),col='#525252',lwd=1.4),
 Poly(c(x+.75,x+.75),c(y+.2,y+1.2),col='#525252',lwd=1.4),Poly(c(x+3.45,x+3.45),c(y+.2,y+1.2),col='#525252',lwd=1.4),
 Rr(x,y+1.1,4.2,3.6,'#525252',r=.5),Pt(x+2.1,y+2.6,.9,'white'))}
scissors<-function(x,y){list(Ln(x+.7,y+3.4,x+3.9,y-.6,col='#525252',lwd=1.3),Ln(x+3.3,y+3.4,x+.1,y-.6,col='#525252',lwd=1.3),
 Pt(x+.5,y+4.2,1.7,'white','#525252',1.1),Pt(x+3.5,y+4.2,1.7,'white','#525252',1.1))}

# width (mm) of a string on a cairo device (Arial -> Liberation Sans metrics)
cairo_pdf(tempfile(fileext='.pdf'),width=4,height=4,family=CELL_FONT)
strw<-function(t,size=TXT,bold=FALSE)convertWidth(grobWidth(textGrob(t,gp=gpar(fontfamily=CELL_FONT,fontsize=size,fontface=if(bold)'bold'else'plain'))),'mm',valueOnly=TRUE)
Tmix<-function(parts,bolds,x,y,size=TXT){g<-list();for(i in seq_along(parts)){g<-c(g,list(T(parts[i],x,y,size,bolds[i])));x<-x+strw(sub(' +$','',parts[i]),size,bolds[i])+if(grepl(' $',parts[i])).278*size*25.4/72 else 0};g}
g<-list(Rr(0,0,W,H,'white'))
# ------------------------------------------------------------------ top band: Train -> Deploy (left to right)
yb<-3;hb<-25;s<-1.65;ty<-11.4;bxT<-3;bwT<-45.3;bxD<-53.3;bwD<-45.3
g<-c(g,list(Rr(bxT,yb,bwT,hb,'#F2F2F2',r=1.5),Rr(bxD,yb,bwD,hb,'#F2F2F2',r=1.5)))
g<-c(g,list(Ln(bxT+bwT+.6,ty+sh(s)/2,bxD-.6,ty+sh(s)/2,arrow=TRUE,lwd=1.1)))
g<-c(g,list(T('Train',bxT+2.5,yb+3.8,HEAD,TRUE)),cells(bxT+bwT-7.4,yb+3.7,.95))
xs<-bxT+c(2.6,13.2,23.8);sgT<-bxT+37.4;sgD<-bxD+38.3
g<-c(g,tiles('Metabolites',xs[1],ty,GM,s),tiles(NULL,xs[2],ty,VR,s),tiles('Protein',xs[3],ty,VP,s))
for(i in 1:3)g<-c(g,list(T(c('M','RNA','P')[i],xs[i]+sw(s)/2,ty+sh(s)+3.4,LAB,just='centre',col=INK)))
g<-c(g,list(Ln(xs[3]+sw(s)+1.1,ty+sh(s)/2,sgT-1.0,ty+sh(s)/2,arrow=TRUE,lwd=.8)),sigmoid(sgT,ty+1.4,w=5.6))
# Deploy: measured M only; virtual RNA / P dashed and shrunk toward their means; predicted response dashed
g<-c(g,list(T('Deploy:',bxD+2.5,yb+3.8,HEAD,TRUE)))
wD<-strw('Deploy: ',HEAD,TRUE)
g<-c(g,list(T('metabolites only',bxD+2.5+wD,yb+3.8,HEAD)))
xd<-bxD+c(2.6,15.0,25.0)
g<-c(g,tiles('Metabolites',xd[1],ty,GM,s),tiles(NULL,xd[2],ty,VR,s,shrink=.35,a=.6),list(outline(xd[2],ty,s,VR,2)),
 tiles('Protein',xd[3],ty,VP,s,shrink=.35,a=.6),list(outline(xd[3],ty,s,VP,2)))
g<-c(g,list(Ln(xd[1]+sw(s)+.7,ty+sh(s)/2,xd[2]-1.0,ty+sh(s)/2,arrow=TRUE,lwd=.8)))
g<-c(g,list(T('virtual',(xd[2]+xd[3]+sw(s))/2,ty+sh(s)+3.4,LAB,just='centre')))
g<-c(g,list(Ln(xd[3]+sw(s)+1.2,ty+sh(s)/2,sgD-1.0,ty+sh(s)/2,arrow=TRUE,lwd=.8)),sigmoid(sgD,ty+1.4,w=5.6,lty=2))
# ------------------------------------------------------------------ three takeaways (top to bottom)
x0<-15.5;L<-33;xt<-61.5
sep<-function(y)Ln(3,y,W-3,y,col='#D9D9D9',lwd=.6)
# (i) room only when M is deployed; virtual RNA / P recover 1/5-1/3 of it; RNA / P rows: small room, not recovered
yi<-c(36.8,43.2,49.6)
g<-c(g,mini('Metabolites',5.2,yi[1]-1.5,GM),mini(NULL,5.2,yi[2]-1.5,VR),mini('Protein',5.2,yi[3]-1.5,VP))
g<-c(g,track(x0,yi[1],L,list(list(col=VR,lo=1/5,hi=1/3,dy=0,hh=1.2),list(col=VP,lo=1/5,hi=1/3,dy=1.2,hh=1.2))))
g<-c(g,track(x0,yi[2],L*.3,dot=TRUE),track(x0,yi[3],L*.3,dot=TRUE))
g<-c(g,list(T('room',x0+L+2.4,yi[1],LAB,col='#525252')))
g<-c(g,list(T('Virtual layer',xt,yi[2]-2.2)),Tmix(c('recovers ','1/5–1/3'),c(FALSE,TRUE),xt,yi[2]+2.2))
g<-c(g,list(sep(55.6)))
# (ii) 8-64 measured proteins recover 48-74 % of the room
yii<-62.9
g<-c(g,mini('Metabolites',5.2,yii-1.5,GM))
for(i in 0:1)for(j in 0:3)g<-c(g,list(Rr(9.2+j*1.05,yii-1.5+i*1.05+.5,.88,.88,PROT)))
g<-c(g,track(x0,yii,L,list(list(col=PROT,lo=.48,hi=.74,dy=0,hh=2.4))))
g<-c(g,Tmix(c('8–64 ','measured proteins'),c(TRUE,FALSE),xt,yii-2.2),Tmix(c('recover ','48–74%'),c(FALSE,TRUE),xt,yii+2.2))
g<-c(g,list(sep(70.2)))
# (iii) CRISPR endpoint: room 40-60 % smaller (track half as long), same share (30-34 %); sealed set: P1-P8
yc<-77.4
g<-c(g,scissors(6.4,yc-2.6))
g<-c(g,track(x0,yc,L*.5,list(list(col=VR,lo=.30,hi=.34,dy=0,hh=1.2),list(col=VP,lo=.30,hi=.34,dy=1.2,hh=1.2))))
g<-c(g,list(T('CRISPR: replicated',xt,yc)))
g<-c(g,list(sep(84.6)))
ys<-91.8
g<-c(g,padlock(5.6,ys-2.2))
ok<-c(TRUE,TRUE,TRUE,FALSE,FALSE,TRUE,TRUE,TRUE);bw<-3.5;gap<-.68
for(k in 1:8){bx<-x0+(k-1)*(bw+gap)
 g<-c(g,list(Rr(bx,ys-bw/2,bw,bw,if(ok[k])'white' else '#F2F2F2',col=if(ok[k])INK else '#9E9E9E',lwd=.6,r=.4)))
 g<-c(g,if(ok[k])list(tick(bx+bw/2,ys))else cross(bx+bw/2,ys))}
g<-c(g,Tmix(c('123 ','sealed lines:'),c(TRUE,FALSE),xt,ys-2.2),list(T('confirmed once',xt,ys+2.2)))
invisible(dev.off())
GAg<-dgroup(g)
nw<-grob_words(GAg);cat('WORDS',nw,'\n');stopifnot(nw<=25)
dir.create(file.path(GA,'out'),showWarnings=FALSE)
stem<-file.path(GA,'out','GraphicalAbstract')
cairo_pdf(paste0(stem,'.pdf'),width=W/25.4,height=H/25.4,family=CELL_FONT);grid.newpage();grid.draw(GAg);dev.off()
png(paste0(stem,'.png'),width=1200,height=1200,res=300,type='cairo',bg='white');grid.newpage();grid.draw(GAg);dev.off()
tiff(paste0(stem,'.tif'),width=1200,height=1200,res=300,type='cairo',compression='lzw',bg='white');grid.newpage();grid.draw(GAg);dev.off()
png(paste0(stem,'_preview2x.png'),width=2400,height=2400,res=600,type='cairo',bg='white');grid.newpage();grid.draw(GAg);dev.off()
cat('DONE\n')

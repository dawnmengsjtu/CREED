# Figure 1A (v3, Cell Systems): few measured layers -> multi-layer functional prediction. Schematic, <= 25 words.
# Solid boxes/tiles = measured assets; dashed boxes = virtual (inferred) or predicted. Tile icons are real
# column-scaled 6 x 5 excerpts of the archived matrices (A_real_array_snippets); grey tiles = unavailable entries.
make_A <- function() {
 H<-65;a<-read_table('A_real_array_snippets')
 M<-LAYER_COL[['M']];P<-LAYER_COL[['P']];R<-LAYER_COL[['RNA']];A<-TARGET
 tint<-function(c)alpha_hex(c,.22)
 icon<-function(asset,x,y,col,step=2){z<-a[a$asset==asset,];ramp<-colorRampPalette(c('#FFFFFF',col))(120)[21:120]
  lapply(seq_len(nrow(z)),function(i)drect(x+z$col[i]*step,y+z$row[i]*step,step*.84,step*.84,if(is.na(z$value[i]))MISSING else ramp[1+round(z$value[i]*99)],col=NA,h=H))}
 box<-function(x,y,w,ht,fill,col,lab,lty=1)list(drect(x,y,w,ht,fill,col,lwd=.6,h=H,lty=lty,r=.6),dtxt(lab,x+w/2,y+ht/2,6.5,h=H))
 arr<-function(x,y,x2,y2,col=INK)dline(x,y,x2,y2,TRUE,h=H,col=col,lwd=.6)
 seg<-function(x,y,x2,y2,col=INK)dline(x,y,x2,y2,FALSE,h=H,col=col,lwd=.6)
 xs<-c(8,28,48,68);lab<-c('Metabolites (M)','RNA','Protein (P)','Drug AUC')
 g<-c(icon('Metabolites',xs[1],6,M),list(drect(xs[2],6,9.7,11.7,tint(R),col=R,lwd=.5,h=H)),icon('Protein',xs[3],6,P),icon('Drug AUC',xs[4],6,A))
 for(i in 1:4)g<-c(g,list(dtxt(lab[i],xs[i]+4.85,21,6.5,h=H)))
 # key (top right, inside the letter band): solid = measured, dashed = virtual
 g<-c(g,list(seg(50,3,54,3),dtxt('measured',55,3,6,just='left',h=H),dline(68,3,72,3,FALSE,h=H,lty=2,lwd=.6),dtxt('virtual',73,3,6,just='left',h=H)))
 # training folds: generator M -> virtual P supervised by measured P; response head M + virtual P -> measured AUC
 g<-c(g,list(arr(42.8,23.6,42.8,26.6),drect(3,27,78,21.5,'#F2F2F2',col=NA,h=H,r=1.2),dtxt('Training folds',5,29.6,6.5,TRUE,just='left',h=H)),
  box(36,31.6,12,4.6,tint(P),P,'P'),list(arr(42,36.2,42,38.6,col=P)),
  box(8,38.8,12,6,tint(M),M,'M'),box(34,38.8,16,6,'white',P,'virtual P',lty=2),box(64,38.8,14,6,tint(A),A,'AUC'),
  list(arr(20,41.8,33.6,41.8),arr(50,41.8,63.6,41.8),seg(14,44.8,14,46.7),seg(14,46.7,71,46.7),arr(71,46.7,71,45.1)))
 # deployment: metabolite input only -> virtual P -> predicted AUC
 g<-c(g,list(arr(42.8,48.8,42.8,51.3),drect(3,51.7,78,13,'#F2F2F2',col=NA,h=H,r=1.2),dtxt('Deployment',5,54.3,6.5,TRUE,just='left',h=H)),
  box(8,56.2,12,6,tint(M),M,'M'),box(34,56.2,16,6,'white',P,'virtual P',lty=2),box(64,56.2,14,6,'white',A,'AUC',lty=2),
  list(arr(20,59.2,33.6,59.2),arr(50,59.2,63.6,59.2),seg(14,62.2,14,63.9),seg(14,63.9,71,63.9),arr(71,63.9,71,62.5)))
 dgroup(g)
}

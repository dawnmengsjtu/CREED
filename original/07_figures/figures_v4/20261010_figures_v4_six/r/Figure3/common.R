# Figure 3 (v4 flexible-layout redesign, 2026-10-10) "Functional utility of the virtual layers and of training-time
# information" - shared constants, sourced once by render_v4.R. Follows the approved Figure 2 approach:
#   eight different plot forms; colour threading M grey / virtual RNA #0072B2 / virtual P #CC79A7 / measured references
#   dark grey #4D4D4D (square = RNA, triangle = P) / shuffle #BDBDBD / tissue label #E69F00 / null = dashed black;
#   folds are never a gray ramp: per-fold marks are small light points in the arm colour; scenario = facet, or
#   filled (new cell lines) / open (held-out tissue) where a panel has no facets (C only, keyed there).
#   Signed per-drug values use a purple-green scale (never blue-orange: blue is the RNA colour).
# Fix round (2026-10-10): A is the page key (arm glyphs + null segment); dashes mean only 'virtual' (tile outline) and
#   'null' (line); off-scale values sit in strips past an axis break drawn with f3_add_break() (white gap + '//').
F3_SC<-c(new_cell='New cell lines',tissue='Held-out tissue')
F3_ARMS<-c('M','M+vRNA','M+vP','M+RNA','M+P')
F3_LAB<-c(M='M only',`M+vRNA`='M + virtual RNA',`M+vP`='M + virtual P',`M+RNA`='M + measured RNA',`M+P`='M + measured P',null_drug='Null')
F3_MEAS<-'#4D4D4D'
F3_COL<-c(M=LAYER_COL[['M']],`M+vRNA`=LAYER_COL[['RNA']],`M+vP`=LAYER_COL[['P']],`M+RNA`=F3_MEAS,`M+P`=F3_MEAS,null_drug=INK)
F3_SHAPE<-c(M=16,`M+vRNA`=16,`M+vP`=16,`M+RNA`=15,`M+P`=17)
F3_SHUF<-ARM_COL_CELL[['shuffle']];F3_TISS<-ARM_COL_CELL[['tissue']];F3_ZERO<-'#9E9E9E'
F3_DIV<-c(low='#762A83',mid='#F7F7F7',high='#1B7837')   # purple (arm worse) - green (arm better)
f3_tint<-function(col,a=.35)alpha_hex(col,a)
f3_num<-function(x,acc=0.01)sub('^-','−',formatC(x,format='f',digits=-log10(acc)))
f3_grob<-function(p){opened<-if(dev.cur()==1){pdf(NULL);TRUE}else FALSE;on.exit(if(opened)dev.off());ggplotGrob(p)}
F3_TOP<-6.4   # plot-area top offset (letter band + one strip line) shared by side-by-side panels
# Figure 1 tissue short names (singleton source category 'Other tissue' shown as 'Unspecified', as in Figure 1F/G)
F3_TISSUE<-c(`Biliary Tract`='Biliary tract',`Central Nervous System`='CNS',`Haematopoietic and Lymphoid`='Haem. / lymphoid',`Head and Neck`='Head / neck',
             `Large Intestine`='Large intestine',`Peripheral Nervous System`='PNS',`Soft Tissue`='Soft tissue',`Other tissue`='Unspecified')
f3_tissue<-function(t)ifelse(t%in%names(F3_TISSUE),F3_TISSUE[t],t)

# Axis-break mark (fix round): a white gap over the axis line plus two parallel slanted strokes, drawn on the gtable in
# the panel cell (exact mm, independent of the data aspect). side 'b' = bottom x axis, 'l' = left y axis; pos = npc
# position along that axis inside the panel.
f3_break_grob<-function(side,pos,gap=.8,len=.8,rise=.75,lwd=.4){
 if(side=='b'){x<-unit(pos,'npc');y<-unit(0,'npc')
  gTree(children=gList(rectGrob(x=x,y=y,width=unit(gap,'mm'),height=unit(1.1,'mm'),gp=gpar(fill='white',col=NA)),
   segmentsGrob(x0=x+unit(c(-gap/2,gap/2)-len/2,'mm'),y0=y-unit(rise,'mm'),x1=x+unit(c(-gap/2,gap/2)+len/2,'mm'),y1=y+unit(rise,'mm'),gp=gpar(col=INK,lwd=lwd,lineend='butt'))))}
 else{x<-unit(0,'npc');y<-unit(pos,'npc')
  gTree(children=gList(rectGrob(x=x,y=y,width=unit(1.1,'mm'),height=unit(gap,'mm'),gp=gpar(fill='white',col=NA)),
   segmentsGrob(x0=x-unit(rise,'mm'),y0=y+unit(c(-gap/2,gap/2)-len/2,'mm'),x1=x+unit(rise,'mm'),y1=y+unit(c(-gap/2,gap/2)+len/2,'mm'),gp=gpar(col=INK,lwd=lwd,lineend='butt'))))}}
f3_panels<-function(g,by='l'){p<-g$layout[grepl('^panel',g$layout$name),];p$name[order(p[[by]],p[[if(by=='l')'t' else 'l']])]}
f3_add_grob<-function(g,panel,grob,name){lay<-g$layout[g$layout$name==panel,];stopifnot(nrow(lay)==1)
 gtable::gtable_add_grob(g,grob,t=lay$t,l=lay$l,b=lay$b,r=lay$r,z=Inf,clip='off',name=name)}
f3_add_break<-function(g,panel,side,pos,...)f3_add_grob(g,panel,f3_break_grob(side,pos,...),paste0('axis-break-',panel,'-',side,'-',sprintf('%.4f',pos)))
F3_RANGE_FILL<-'#E3E3E3'   # H: null-to-best-measured range bar (the light track of Figure 2A's room bar)
F3_NOTRUN<-'#F0F0F0'       # B: comparator not run in that scenario

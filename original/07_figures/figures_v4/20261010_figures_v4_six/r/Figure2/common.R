# Figure 2 (v4 pilot, 2026-10-10) - "A cross-layer recoverability map". Shared constants, sourced once by render_v4.R.
# Colour threading (one colour per directed pair / arm, identical in every panel where it appears):
#   M->RNA and M + virtual RNA = #0072B2; M->P and M + virtual P = #CC79A7; measured references = black 60 %;
#   paired shuffles = #BDBDBD; the four other directed pairs = one muted gray. Scenario = fill (filled = new cell
#   lines, open = held-out tissue) or facet; never a new colour. Numbers are read verbatim from plot_data/.
F2_SC<-c(new_cell='New cell lines',tissue='Held-out tissue')
F2_PAIRS<-c('M→RNA','M→P','RNA→M','RNA→P','P→M','P→RNA')
F2_OTHER<-'#8C8C8C'
F2_PAIR_COL<-c(`M→RNA`=ARM_COL_CELL[['m_vrna']],`M→P`=ARM_COL_CELL[['m_vp']],`RNA→M`=F2_OTHER,`RNA→P`=F2_OTHER,`P→M`=F2_OTHER,`P→RNA`=F2_OTHER)
F2_MEAS<-ARM_COL_CELL[['measured']];F2_SHUF<-ARM_COL_CELL[['shuffle']]
F2_MEAS_SOLID<-'#666666'   # black 60 % on white, opaque (for dots drawn over tracks)
F2_FOLDLINE<-F2_SHUF;F2_FOLDLAB<-F2_OTHER   # per-fold joining lines / fold labels: grays already used in E and F (no fold ramp)
F2_SHAPE<-c(new_cell=21,tissue=21)                       # fillable circle; fill = colour (new cell lines) or white (tissue)
f2_fill<-function(col,sc)ifelse(sc=='new_cell',col,'white')
f2_lab<-function(x,acc=0.01)sub('^-','−',formatC(x,format='f',digits=-log10(acc)))
f2_fix<-function(p,...)list(plot=p,grob=fix_panel_margins(p,...,id=ID))
F2_BOT_TOP<-6.2;F2_BOT_BOTTOM<-9.5   # bottom row G | H: fixed plot-area top / bottom (mm) so the two plot areas align
F2_TOP<-6.2   # plot top offset shared by side-by-side panels (letter band 5 mm + strip)
# simple 1-D label repel: keep label order, enforce a minimum gap (data units), centred on the original positions
f2_spread<-function(y,gap){o<-order(y);z<-y[o];for(i in seq_along(z)[-1])if(z[i]-z[i-1]<gap)z[i]<-z[i-1]+gap
 z<-z-(mean(z)-mean(y[o]));for(i in seq_along(z)[-1])if(z[i]-z[i-1]<gap)z[i]<-z[i-1]+gap;r<-y;r[o]<-z;r}

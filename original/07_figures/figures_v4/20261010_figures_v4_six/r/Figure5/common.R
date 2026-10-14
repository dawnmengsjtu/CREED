# Figure 5 (v4 flexible-layout redesign, 2026-10-10; fix round 2026-10-10) - "The map and the panel law replicate on a
# second functional endpoint". Same approach as the approved Figure 2: eight different plot forms, one hero (C, the
# two-endpoint map), icon schematic in the Figure 1A vocabulary, colour threading with no collisions, page-wide keys drawn
# once (in A).
# Encodings (identical in every panel where they appear):
#   endpoint = SHAPE: GDSC circle (21), CRISPR triangle (24); endpoint is never a colour. Square (22) = no endpoint: the
#              neutral glyph of the scenario key in A and of the paired GDSC x CRISPR gains in H.
#   scenario = FILL: filled = new cell lines, open (white) = held-out tissue; or facet.
#   colour   = directed pair / arm / panel layer: M->RNA and M + virtual RNA #0072B2; M->P #CC79A7; the other four pairs one
#              muted gray; measured references #4D4D4D (also the one-third markers, which are fractions of the measured
#              room); null = #BDBDBD (baseline gray); protein panel #D55E00; gene panel #56B4E9. Endpoint icons in A:
#              GDSC = drug-AUC excerpt in the dark gray #525252 of the endpoint glyphs (the vermilion of Figures 1A/2A is
#              the protein panel on this page); CRISPR = bluish-green grid (#009E73, used nowhere else on the page).
#   per-fold values = one light gray or the series colour at low alpha (no fold ramp); room = gray, as in Figure 2.
# Numbers are read verbatim from plot_data/ (v3 Figure 7 tables, Figure 2 v4 tables copied with an f2_ prefix, and the
# f5_crispr_* extracts of crispr-endpoint-20261009-0001/output/{comparisons,fold_metrics}.csv). Nothing is recomputed
# except five-fold means of listed per-fold scores and ratios already present as columns.
F5_SC<-c(new_cell='New cell lines',tissue='Held-out tissue')
F5_EP<-c('GDSC','CRISPR')
F5_SHAPE<-c(GDSC=21,CRISPR=24)
F5_OTHER<-'#8C8C8C'
F5_PAIR_COL<-c(`M→RNA`=LAYER_COL[['RNA']],`M→P`=LAYER_COL[['P']],`RNA→M`=F5_OTHER,`RNA→P`=F5_OTHER,`P→M`=F5_OTHER,`P→RNA`=F5_OTHER)
F5_MEAS<-'#4D4D4D'            # measured reference (room end / M + measured RNA / one-third markers)
F5_NULL<-'#BDBDBD'            # null arm (baseline gray)
F5_KEY<-'#525252'             # endpoint glyphs in keys and headers; GDSC icon in A
F5_TRACK<-'#D9D9D9'           # room track
F5_FOLDPT<-'#ABABAB'          # per-fold points (one light gray)
F5_FOLDLINE<-'#DCDCDC'        # per-fold joining lines (lighter than the null gray)
F5_FOLDLAB<-'#7F7F7F'
F5_CRISPR_TILE<-OI[['green']] # CRISPR endpoint grid in A only (bluish green is not used anywhere else on the page)
f5_fill<-function(col,sc)ifelse(sc=='new_cell',col,'white')
f5_lab<-function(x,digits=2)sub('^-','−',formatC(x,format='f',digits=digits))
f5_slab<-function(x,digits=2)sub('^-','−',sprintf(paste0('%+.',digits,'f'),x))
f5_fix<-function(p,...)list(plot=p,grob=fix_panel_margins(p,...,id=ID))
f5_spread<-function(y,gap){o<-order(y);z<-y[o];for(i in seq_along(z)[-1])if(z[i]-z[i-1]<gap)z[i]<-z[i-1]+gap
 z<-z-(mean(z)-mean(y[o]));for(i in seq_along(z)[-1])if(z[i]-z[i-1]<gap)z[i]<-z[i-1]+gap;r<-y;r[o]<-z;r}
f5_grob<-function(p){opened<-if(dev.cur()==1){pdf(NULL);TRUE}else FALSE;on.exit(if(opened)dev.off());ggplotGrob(p)}

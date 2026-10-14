# Figure 4 (v4 flexible layout, fix round 2, 2026-10-10): "How much must be measured: the panel law and pre-registered
# negative controls". Numbers are read verbatim from the v3 Figure 5 plot_data tables in out/Figure4/plot_data/, plus
# three probe-2 tables re-arranged from the finished probe-2 summary.json (no new statistics):
#   g_cov50_fold, h_aurc_fold (prep/extract_probe2_fold.py; five-fold means checked against e_riskcov_mean / e_aurc) and
#   g_n3_vs_random (prep/extract_probe2_cmp.py; the pre-registered N3 comparisons against Random, checked against the
#   verdicts, e_aurc and h_aurc_fold).
# Layout V3 of fix round 2: 20-mm schematic band + page key, hero B beside stacked C/D, three-across row E | F | G
# (G = the N3 test itself), full-width H (risk-coverage curves | per-fold rows); see Figure4_v4_PANELMAP.md.
# Colour threading (one meaning per colour, page-wide):
#   protein panels #D55E00 (all protein-panel arms, any k), gene panels #56B4E9 (random panels in B = the same hue, pale);
#   M + virtual RNA #0072B2, M + virtual P #CC79A7; measured teacher (M + measured P values) dark grey #4D4D4D only;
#   random ranking #BDBDBD; oracle ranking mid grey #969696 (dotted); tissue-adaptive #E69F00 (tissue label);
#   cluster-adaptive dark purple #762A83 (purple-green scheme; used nowhere else); null = shuffle/null grey #BDBDBD (D);
#   folds = one light grey #A6A6A6 or a light tint of the series colour; panel size k = one sequential green ramp
#   #A1D99B -> #00441B (C only); N1 heat map = white -> vermilion; black = reference lines only.
# Scenario: filled = New cell lines, open = Held-out tissue (key drawn once, in the KEY slot beside A).
F4_SC<-c(new_cell='New cell lines',tissue='Held-out tissue')
f4_sc<-function(x)factor(unname(F4_SC[x]),levels=F4_SC)
F4_PROT<-PANEL_COL[['protein']];F4_GENE<-PANEL_COL[['gene']]
F4_VR<-ARM_COL_CELL[['m_vrna']];F4_VP<-ARM_COL_CELL[['m_vp']]
F4_REF<-'#4D4D4D';F4_ORC<-'#969696';F4_RND<-'#BDBDBD';F4_TIS<-'#E69F00';F4_CLU<-'#762A83';F4_FOLD<-'#A6A6A6';F4_NEU<-'#525252'
F4_NULL<-'#BDBDBD';F4_ROOM<-'#E3E3E3'
F4_K<-c(`8`='#A1D99B',`16`='#41AB5D',`32`='#006D2C',`64`='#00441B');F4_KLINE<-'#00441B'
# N3 proxies: house descriptions (u1 / u3 codes live only in the legend text)
F4_PX<-c(random='Random',u1='Generator disagreement',u3='Distance in M',oracle='Oracle')
F4_PXSH<-c(random=16,u1=16,u3=17,oracle=18)            # filled shapes; open versions 21/24/23 for Held-out tissue
F4_PXSH_OPEN<-c(random=21,u1=21,u3=24,oracle=23)
F4_PXLT<-c(random='solid',u1='solid',u3='42',oracle='13')
F4_ARM<-c(`M+vRNA`='M + virtual RNA',`M+vP`='M + virtual P');F4_ARMCOL<-c(`M+vRNA`=F4_VR,`M+vP`=F4_VP)
f4_fill<-function(col,sc)ifelse(sc=='new_cell',col,'white')
f4_lab<-function(x,d=2)sub('^-','−',formatC(x,format='f',digits=d))
f4_fix<-function(p,...)list(plot=p,grob=fix_panel_margins(p,...,id=ID))
F4_KX<-8   # x position of 'all' (full candidate pool) on the log2 k axis: one tick spacing after k = 128, with a break mark
# Bold block title above a block of rows (G, H right): horizontal, left-aligned, the strip size of the other panels.
f4_block_title<-function(d)geom_text(data=d,aes(x=x,y=y,label=lab),hjust=0,vjust=0,size=PT_STRIP/.pt,fontface='bold',family=CELL_FONT,colour=INK,inherit.aes=FALSE)
# Put several pre-aligned gtables side by side in one slot (widths in mm), drawing unclipped so block titles may overhang.
f4_row<-function(grobs,widths){g<-gtable::gtable_row(ID,grobs,widths=unit(widths,'mm'),height=unit(SLOT_H,'mm'));g$layout$clip<-'off';g}

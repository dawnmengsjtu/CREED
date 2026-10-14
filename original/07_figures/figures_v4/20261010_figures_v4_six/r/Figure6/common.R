# Figure 6 (v4, fix round 2, 2026-10-10) - "One-shot confirmation on the sealed internal-final set".
# Shared constants, sourced once by render_v4.R. Eight panels, eight plot forms:
#   A protocol icon schematic (+ the page symbol key) | B GDSC forest (hero) with a win-fraction strip column and the
#   verdict column of the gain items (P1, P2, P3, P6, P7) | C the sealed set (unit chart, 123 lines by tissue stratum) |
#   D CRISPR gains (bars + intervals, shuffle inset) | E panel recovery R(k) on the sealed set (line plot) |
#   F room and recovered gain, development -> sealed (slope chart, development folds as dots) |
#   G ratio items, development -> sealed (dumbbell rows: recovered share P4 and R(k) P5 / P8, with verdicts) |
#   H sealed vs development gain (scatter, broken y axis).
# Pre-registered items are called P1-P8 on the page (summary.json / protocol.json keys F1-F8; renamed so that
# they cannot be read as folds F0-F4).
# Symbol rule, defined once in A: filled = sealed set; open = new cell lines (development folds).
# Endpoint (wherever both endpoints share a plot: B strip column, G, H): circle = GDSC, triangle = CRISPR.
# Colour threading (one colour per arm class, the same in every panel):
#   tissue label #E69F00 | M + virtual RNA #0072B2 | M + virtual P #CC79A7 | paired shuffles #BDBDBD |
#   M + protein graph #009E73 | protein panels #D55E00 | gene panels #56B4E9 (random panels: same hue, 35 % fill; tints
#   are used for nothing else) | measured references #4D4D4D | development folds: one light grey (#A6A6A6).
#   Symbol keys, verdict ticks and crosses, count blocks and unit squares: #4D4D4D. Black = axes, text and dashed
#   reference / threshold lines only. No diverging scale anywhere.
# Data: plot_data/forest.csv, forest_seeds.csv, b_share.csv, b_panel_R.csv, d_scoreboard.csv, protocol.json (v3
# Figure 8 tables, copied verbatim), summary.json (internal_final_20261009/output), A_real_array_snippets.csv
# (Figure 1A icon excerpts, copied from Figure 2) and dev_folds.csv (per-fold development values copied verbatim by
# prep_dev_folds.R from out/Figure2/plot_data/xl_pair_folds.csv and the CRISPR endpoint run's comparisons.csv).
# Nothing is recomputed; every printed number is read from these files.
F6_COL<-c(tissue='#E69F00',vrna='#0072B2',vp='#CC79A7',shuffle='#BDBDBD',graph='#009E73',
          protein='#D55E00',gene='#56B4E9',protein_random=alpha_hex('#D55E00',.35),gene_random=alpha_hex('#56B4E9',.35),
          measured='#4D4D4D')
F6_MEAS<-'#4D4D4D';F6_KEY<-'#4D4D4D';F6_SHUF<-'#BDBDBD';F6_RULE<-'#D9D9D9';F6_BAND<-'#F2F2F2';F6_FOLD<-'#A6A6A6'
F6_AUC_TINT<-'#E9A97C'   # light-orange tint of the Figure 1A drug AUC tile (vermilion mixed ~50 % with white)
F6_ROW_LABEL<-c('M+T'='M + tissue label','T'='Tissue label alone','M+vRNA+T'='M + virtual RNA + tissue label',
  'M+vRNA'='M + virtual RNA','M+vP'='M + virtual P',
  'M+vRNA_shuf'='M + shuffled virtual RNA','M+vP_shuf'='M + shuffled virtual P','M+P_graph'='M + protein graph',
  'M+P16'='16 proteins','M+P32'='32 proteins','M+P64'='64 proteins','M+G16'='16 genes','M+G32'='32 genes','M+G64'='64 genes',
  'M+Prand16_mean'='16 proteins, random','M+Prand32_mean'='32 proteins, random','M+Grand16_mean'='16 genes, random','M+Grand32_mean'='32 genes, random',
  'M+RNA'='M + measured RNA','M+P'='M + measured P','M+Pval'='M + measured P values')   # final polish 2026-10-11: was 'M + measured P (no mask)'
F6_GROUP_LABEL<-c(tissue='Tissue label',virtual='Virtual layers',shuffle='Shuffle controls',graph='Protein graph',panel='Measured panels',reference='Measured layers')
f6_colkey<-function(d)dplyr::case_when(d$group=='panel'~paste0(d$layer,ifelse(d$kind=='random_mean','_random','')),
  d$group=='reference'~'measured',d$arm=='M+vRNA'~'vrna',d$arm=='M+vP'~'vp',d$arm=='M+vRNA+T'~'tissue',TRUE~d$group)
# display regrouping only (as v3): M + virtual RNA + tissue label is listed as the third tissue-label row
f6_regroup<-function(d){i<-d$arm=='M+vRNA+T';if(any(i)){d$group[i]<-'tissue';d$group_order[i]<-d$group_order[d$arm=='M+T'][1];d$row_order[i]<-3};d}
f6_forest_rows<-function(ep){d<-read_table('forest')%>%filter(endpoint==ep);d<-f6_regroup(d)%>%arrange(group_order,row_order)
 d$colkey<-f6_colkey(d);d$row_label<-unname(F6_ROW_LABEL[d$arm]);stopifnot(!any(is.na(d$row_label)));d}
f6_fix<-function(p,...)list(plot=p,grob=fix_panel_margins(p,...,id=ID))
f6_num<-function(x,d=2)sub('^-','−',formatC(x,format='f',digits=d))
f6_item<-function(f)sub('^F','P',f)                # F1-F8 (source keys) -> P1-P8 (page names)
# verdicts of the eight items: protocol.json, asserted equal to d_scoreboard.csv -> confirmed
f6_verdicts<-function(){pr<-fromJSON(file.path(DATA,'protocol.json'));sb<-read_table('d_scoreboard');items<-paste0('F',1:8)
 v<-unlist(pr$verdicts[items]);stopifnot(length(v)==8,identical(unname(v),as.logical(sb$confirmed[match(items,sb$item)])));v}
# tick (confirmed) / cross (not confirmed) as segments in data units; ux, uy = data units per mm on x and y
f6_glyph_segs<-function(x,y,ok,ux,uy){if(ok)data.frame(x=c(x-.8*ux,x-.22*ux),y=c(y,y-.62*uy),xe=c(x-.22*ux,x+.85*ux),ye=c(y-.62*uy,y+.72*uy))else
  data.frame(x=c(x-.62*ux,x-.62*ux),y=c(y-.62*uy,y+.62*uy),xe=c(x+.62*ux,x+.62*ux),ye=c(y+.62*uy,y-.62*uy))}
F6_GROUP_PT<-PT_STRIP                              # one style for group labels in B and D: 6.5 pt bold
F6_TOP<-8.5                                        # slot top -> plot top for the D / E row (letter band 5 mm + strip)
F6_TOP4<-7.5;F6_BOT4<-10                           # slot top / bottom -> plot area for the F / G / H row
F6_XB<-c(1,2.5,4.4,5.9)   # x base of GDSC RNA, GDSC P, CRISPR RNA, CRISPR P in F
f6_share_rows<-function(){b<-read_table('b_share');stopifnot(nrow(b)==4)
 b%>%mutate(ep=ifelse(endpoint=='drug','GDSC','CRISPR'),lay=ifelse(layer=='RNA','RNA','P'),col=ifelse(layer=='RNA',F6_COL[['vrna']],F6_COL[['vp']]),
   base=F6_XB[match(paste(endpoint,layer),c('drug RNA','drug P','crispr RNA','crispr P'))])}
f6_xlabels<-function(b,yr){u<-diff(yr)/(SLOT_H-F6_TOP4-F6_BOT4);y1<-yr[1]-1.4*u;y2<-yr[1]-4.3*u
 eg<-b%>%group_by(ep)%>%summarise(x=mean(base),.groups='drop')
 list(geom_text(data=b,aes(base,y1,label=lay),size=GEOM_TEXT_MIN,vjust=1,colour=INK,inherit.aes=FALSE),
      geom_text(data=eg,aes(x,y2,label=ep),size=F6_GROUP_PT/.pt,fontface='bold',vjust=1,colour=INK,inherit.aes=FALSE))}

# Figure 5H - what does not transfer: which molecules. Gain (%) of the supervised panel over a random panel of the same
# size (mean of five seeds), GDSC (x) against CRISPR (y) on the same fold (old F and G merged; d_random_fold paired by
# scenario, layer, k and fold - a re-arrangement, nothing recomputed). Glyph = the neutral square (22) of the page (each
# point carries both endpoints). Large squares = five-fold means of the per-fold gains (d_random_mean gain_pct_fold_mean);
# small translucent squares = per fold. Colour = panel layer (protein #D55E00, gene #56B4E9); filled = new cell lines,
# open = held-out tissue; k = symbol SIZE (smaller = 16, larger = 32; fix round, was diamond/square), keyed once above the
# plot area, outside it. Solid lines = zero (random panel); dashed = identity, labelled 'y = x' at its upper end.
# Final polish (2026-10-11): per-fold squares at 35 % alpha with no outline (new cell lines; held-out tissue: open, 55 %-alpha
# outline as in G), mean squares on top with a white halo (0.6 pt), so the eight means read above the folds. Axes
# x = -0.3 to 1.1 and y = -0.35 to 1.05 (equal ranges, so the identity line runs at 45 degrees; were -0.6 to 1.55 and
# -0.6 to 1.05). The three per-fold values outside the x range (all held-out tissue, gene panel: F4 k = 16, GDSC -0.44 %;
# F4 k = 32, GDSC -0.53 %; F1 k = 32, GDSC +1.46 %) are pinned to the axis edge as half-squares at their true y (clipped
# by the panel edge; listed in panel_data/H_pinned_outliers.csv). Axis titles in the house pattern of Figure 6
# ('Gain over random (%), GDSC').
make_H<-function(){
 rf<-read_table('d_random_fold');rm<-read_table('d_random_mean')
 w<-rf%>%select(endpoint,scenario,layer,k,fold,gain_pct)%>%tidyr::pivot_wider(names_from=endpoint,values_from=gain_pct)
 wm<-rm%>%select(endpoint,scenario,layer,k,gain_pct_fold_mean)%>%tidyr::pivot_wider(names_from=endpoint,values_from=gain_pct_fold_mean)
 SZ_F<-c(`16`=.85,`32`=1.35);SZ_M<-c(`16`=1.45,`32`=2.3)
 XL<-c(-.3,1.1);YL<-c(-.35,1.05)
 prep<-function(d)d%>%mutate(col=PANEL_COL[layer],fill=f5_fill(col,scenario),kk=as.character(k))
 w<-prep(w)%>%mutate(sz=SZ_F[kk],out=GDSC<XL[1]|GDSC>XL[2],
   pfill=ifelse(scenario=='new_cell',alpha_hex(col,.35),'transparent'),pcol=ifelse(scenario=='new_cell','transparent',alpha_hex(col,.55)))
 wm<-prep(wm)%>%mutate(sz=SZ_M[kk])
 wi<-w%>%filter(!out);wo<-w%>%filter(out)
 stopifnot(nrow(w)==40,nrow(wm)==16-8,!anyNA(w$GDSC),!anyNA(w$CRISPR),nrow(wo)==3,all(wo$scenario=='tissue'),all(wo$layer=='gene'),
           all(w$CRISPR>YL[1]+.04),all(w$CRISPR<YL[2]-.04),all(wi$GDSC>XL[1]+.03),all(wi$GDSC<XL[2]-.03),
           all(wm$GDSC>XL[1]&wm$GDSC<XL[2]),all(wm$CRISPR>YL[1]&wm$CRISPR<YL[2]))
 write_csv(wo%>%select(scenario,layer,k,fold,GDSC,CRISPR)%>%mutate(pinned_to=ifelse(GDSC<XL[1],XL[1],XL[2])),
           file.path(fig_out(FIG),'panel_data','H_pinned_outliers.csv'))
 # k key above the plot area (outside it, in the band that aligns this panel's plot top with G's strip)
 YK<-YL[2]+.10;key<-tibble::tibble(x=c(-.216,.306),y=YK,kk=c('16','32'),lab=c('k = 16','k = 32'))%>%mutate(sz=SZ_M[kk])
 HALO<-.6+2*.6/(.stroke/2*.75)   # mean stroke (0.6) plus 0.6 pt of white on each side, in ggplot stroke units
 p<-ggplot(wi,aes(GDSC,CRISPR))+
  geom_hline(yintercept=0,linewidth=LW_THIN,colour=INK)+geom_vline(xintercept=0,linewidth=LW_THIN,colour=INK)+
  geom_abline(slope=1,intercept=0,linetype='22',linewidth=LW_LINE,colour='#7F7F7F')+
  annotate('text',x=.92,y=1.0,label='y = x',hjust=1,vjust=.5,size=6/.pt,family=CELL_FONT,colour='#7F7F7F')+
  geom_point(aes(size=I(sz)),shape=22,fill=wi$pfill,colour=wi$pcol,stroke=.3)+
  geom_point(data=wm,aes(size=I(sz)),shape=22,fill='white',colour='white',stroke=HALO)+
  geom_point(data=wm,aes(size=I(sz),fill=fill,colour=col),shape=22,stroke=.6)+
  geom_point(data=key,aes(x,y,size=I(sz)),shape=22,fill='#7F7F7F',colour='white',stroke=.4)+
  geom_text(data=key,aes(x+.059,y,label=lab),hjust=0,size=6/.pt,family=CELL_FONT,colour=INK)+
  scale_fill_identity()+scale_colour_identity()+
  scale_x_continuous(breaks=c(0,.5,1),labels=num_minus_acc(0.1),expand=c(0,0))+
  scale_y_continuous(breaks=c(0,.5,1),labels=num_minus_acc(0.1),expand=c(0,0))+
  coord_cartesian(xlim=XL,ylim=YL,clip='off')+
  labs(x='Gain over random (%), GDSC',y='Gain over random (%), CRISPR')+BASE+
  theme(plot.margin=margin(5,1.5,1,1,'mm'))
 # pinned outliers: per-fold squares centred on the panel edge at their true y, clipped by the panel (half-squares)
 g<-f5_grob(p);pl<-g$layout[grepl('^panel',g$layout$name),]
 side<-.665*wo$sz   # mm: side of a ggplot shape-22 square of that size
 hs<-rectGrob(x=unit(ifelse(wo$GDSC<XL[1],0,1),'npc'),y=unit((wo$CRISPR-YL[1])/diff(YL),'npc'),width=unit(side,'mm'),height=unit(side,'mm'),
   gp=gpar(fill=NA,col=wo$pcol,lwd=.3*.stroke/2))
 g<-gtable::gtable_add_grob(g,hs,t=pl$t,l=pl$l,b=pl$b,r=pl$r,clip='on',name='h_pinned')
 list(plot=p,grob=g)}

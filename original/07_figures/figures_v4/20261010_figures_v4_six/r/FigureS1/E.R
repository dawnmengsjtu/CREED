# FigureS1 E (v4) <- v3 Figure 4D: leave-one-fold-out sensitivity of the four main effects (points = omitted fold,
# grey-to-black F0-F4; black bar = all-five-fold value). Facets side by side (New cell lines | Held-out tissue), each
# closed (own axis line at zero gain, tick labels on the margins). Carries the page's only fold key (F0-F4, one row),
# which also serves D directly above. Fold points are offset vertically by fold (S1_FOLD_DODGE, F0 top to F4 bottom)
# for legibility only; the black bar is drawn ABOVE the points and is taller than the fold stack (S1_BAR_E).
# Left plot edge = S1_LEFT (shared with B and D); plot top/bottom shared with F, so the two keys sit on one baseline.
make_E<-function(){
 d<-read_table('fig4_lofo')
 d$y<-match(d$effect,S1_EFF);d$y_pt<-d$y-(d$omitted_fold-2)*S1_FOLD_DODGE;d$scenario<-factor(S1_SCEN[d$scenario],levels=S1_SCEN)
 s<-d%>%group_by(scenario,effect,y)%>%summarise(full=first(full_mean_pct),.groups='drop')
 p<-ggplot(d,aes(rel_pct,y))+
  geom_point(aes(y=y_pt,colour=factor(omitted_fold)),size=PT_SMALL)+
  geom_segment(data=s,aes(x=full,xend=full,y=y-S1_BAR_E,yend=y+S1_BAR_E),colour=INK,linewidth=LW_LINE*1.5)+
  facet_wrap(~scenario,ncol=2,axes='all',axis.labels='margins')+
  scale_colour_manual(values=FOLD_RAMP,labels=paste0('F',0:4),name=NULL)+
  scale_y_continuous(breaks=seq_along(S1_EFF),labels=S1_EFF_LAB[S1_EFF],expand=expansion(add=.6))+
  scale_x_continuous(breaks=seq(0,1.2,.4),limits=c(0,1.3),expand=expansion(mult=c(0,.03)),labels=num_minus_acc(.1))+
  labs(x='Gain in sRMSE (%)',y=NULL)+
  BASE+theme(axis.text.y=element_text(lineheight=.9),panel.spacing.x=unit(S1_FACET_GAP,'mm'),legend.margin=margin(1,0,0,0,'mm'))
 list(plot=p,grob=fix_panel_margins(p,left=S1_LEFT,top=S1_TOP_EF,bottom=S1_BOT_EF,id='E'))}

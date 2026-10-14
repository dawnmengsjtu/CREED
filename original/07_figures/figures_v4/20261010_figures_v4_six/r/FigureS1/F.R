# FigureS1 F (v4) <- v3 Figure 4E: interval summaries of the same four effects, copied from the run tables.
# Thin black line = Nadeau-Bengio corrected 95% CI; thick grey bar = grouped bootstrap 95% (protein-graph rows only);
# circle = five-fold mean, black outline, filled in the effect colour; dashed line = zero. Same rows as E beside it
# (same y scale, same plot top and bottom), so the row labels are printed once (in E) and F has no y text: the
# width goes to the intervals.
make_F<-function(){
 d<-read_table('fig4_intervals')
 d$effect<-factor(d$effect,levels=S1_EFF);d$scenario<-factor(S1_SCEN[d$scenario],levels=S1_SCEN)
 d$fill<-unname(S1_EFF_COL[as.character(d$effect)])
 kinds<-c('Bootstrap 95%','Corrected 95% CI')
 seg<-bind_rows(d%>%filter(!is.na(boot_lo))%>%transmute(scenario,effect,lo=boot_lo,hi=boot_hi,kind=kinds[1]),
                d%>%transmute(scenario,effect,lo=nb_lo,hi=nb_hi,kind=kinds[2]))   # bootstrap drawn first (beneath)
 seg$kind<-factor(seg$kind,levels=rev(kinds))
 p<-ggplot(d,aes(y=effect))+
  geom_vline(xintercept=0,linetype='22',linewidth=LW_THIN,colour=INK)+
  geom_segment(data=seg,aes(x=lo,xend=hi,yend=effect,linewidth=kind,colour=kind))+
  geom_point(aes(x=mean_pct,fill=fill),shape=21,colour=INK,size=PT_LARGE,stroke=.3)+scale_fill_identity()+
  facet_wrap(~scenario,ncol=2,axes='all',axis.labels='margins')+
  scale_linewidth_manual(values=setNames(c(LW_ERR,1.3),rev(kinds)),name=NULL)+
  scale_colour_manual(values=setNames(c(INK,'#BDBDBD'),rev(kinds)),name=NULL)+
  scale_x_continuous(breaks=seq(0,1.5,.5),limits=c(-.3,1.62),expand=expansion(mult=0),labels=num_minus_acc(.1))+
  scale_y_discrete(labels=S1_EFF_LAB,expand=expansion(add=.6))+
  labs(x='Gain in sRMSE (%)',y=NULL)+
  BASE+theme(axis.text.y=element_blank(),panel.spacing.x=unit(S1_FACET_GAP,'mm'),legend.margin=margin(1,0,0,0,'mm'),legend.key.width=unit(5,'mm'))
 list(plot=p,grob=fix_panel_margins(p,left=S1_LEFT_F,top=S1_TOP_EF,bottom=S1_BOT_EF,id='F'))}

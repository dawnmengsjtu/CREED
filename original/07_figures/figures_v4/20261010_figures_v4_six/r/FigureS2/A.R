# FigureS2 A <- v3 Figure 6B: drugs per annotation group evaluated in all five folds, one facet per scenario
# (new cell lines 295, held-out tissue 281). Log10 x axis (breaks 1 / 10 / 100; bars start at 1) so the 2-5-drug named
# groups are readable next to Unannotated (38) and Other (218-230); exact counts printed at the bar ends.
# Bars coloured by annotation tone (opaque TONE_COL); the tone key is drawn once in the shared slot LEG_AB (serves A and C).
A_XMAX<-700   # upper x limit (drugs): leaves room for the printed count right of the longest bar inside each facet
make_A<-function(legend=FALSE){d<-annotation_data('annotation_counts')%>%filter(panel_key%in%c('new_cell','tissue'))%>%
  mutate(scenario=factor(SCEN_LABEL[panel_key],levels=SCEN_LABEL),y=group_y(group),tone=group_tone(group))
 stopifnot(all(d$n>=1),max(d$n)<A_XMAX/2)
 ggplot(d,aes(n,y))+geom_col(aes(fill=tone),orientation='y',width=.72)+
  geom_text(aes(label=n),hjust=-.25,size=GEOM_TEXT_MIN,family=CELL_FONT)+
  facet_wrap(~scenario,nrow=1)+scale_fill_manual(values=TONE_COL,drop=FALSE,name=NULL)+
  scale_x_log10(breaks=c(1,10,100),labels=c('1','10','100'))+scale_y_groups()+coord_groups(xlim=c(1,A_XMAX))+
  labs(x='Drugs',y=NULL)+BASE+guides(fill=guide_legend(nrow=1))+
  theme(axis.line.y=element_blank(),axis.ticks.y=element_blank(),panel.spacing=S2_FACET_GAP,
        legend.position=if(legend)'bottom' else 'none',legend.key.size=unit(2.6,'mm'),legend.key.spacing.x=unit(3,'mm'))}

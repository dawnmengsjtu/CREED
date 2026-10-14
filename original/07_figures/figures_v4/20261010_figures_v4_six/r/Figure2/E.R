# Figure 2E - paired-shuffle slopegraph (fix round + polish, 2026-10-10). For each of the six directed pairs and each fold
# (xl_pair_folds), a line from the gain with the row-shuffled virtual layer (shuffle_fold) to the gain with the virtual
# layer (recovered_fold), both relative to the deployed layer alone (M only for M->RNA and M->P); one facet per scenario.
# M->RNA / M->P in their colours, the other four pairs in one muted gray. Thick lines = five-fold means (shuffle_gain /
# recovered_gain from xl_matrix). Polish: the three direct labels are no longer a ggplot layer (which stretched the x
# scale of both facets); they sit in their own gtable column right of the held-out tissue facet, belonging to neither
# facet, at the held-out-tissue means. Both facet plot areas therefore have identical width and the same x range.
make_E<-function(){
 f<-read_table('xl_pair_folds')%>%transmute(scenario,pair,fold,Shuffled=shuffle_fold,Virtual=recovered_fold)
 m<-read_table('xl_matrix')%>%transmute(scenario,pair,Shuffled=shuffle_gain,Virtual=recovered_gain)
 lg<-function(z)z%>%pivot_longer(c(Shuffled,Virtual),names_to='arm',values_to='gain')%>%
  mutate(sc=factor(F2_SC[scenario],levels=F2_SC),x=ifelse(arm=='Shuffled',1,2),hi=pair%in%c('M→RNA','M→P'),col=F2_PAIR_COL[pair],
         fill=ifelse(arm=='Shuffled',F2_SHUF,f2_fill(col,scenario)))
 F<-lg(f)%>%mutate(g=paste(pair,fold));M<-lg(m)
 stopifnot(nrow(F)==120,nrow(M)==24)
 oth<-M%>%filter(x==2,!hi)%>%group_by(sc)%>%summarise(gain=mean(range(gain)),.groups='drop')%>%mutate(lab='Other\npairs',col=F2_OTHER)
 L<-bind_rows(M%>%filter(x==2,hi)%>%transmute(sc,gain,lab=pair,col),oth)%>%filter(sc==F2_SC[['tissue']])%>%mutate(ly=f2_spread(gain,.2))
 p<-ggplot(F,aes(x,gain))+
  geom_hline(yintercept=0,linewidth=LW_THIN,colour='#9E9E9E')+
  geom_line(data=F%>%filter(!hi),aes(group=g),colour=F2_OTHER,alpha=.35,linewidth=LW_THIN)+
  geom_line(data=F%>%filter(hi),aes(group=g,colour=col),alpha=.45,linewidth=LW_THIN)+
  geom_point(data=F,aes(colour=ifelse(x==1,'#9E9E9E',col),fill=fill),shape=21,size=PT_SMALL-.2,stroke=.3,alpha=.8)+
  geom_line(data=M%>%filter(!hi),aes(group=pair),colour=F2_OTHER,linewidth=LW_LINE*1.6)+
  geom_line(data=M%>%filter(hi),aes(group=pair,colour=col),linewidth=LW_LINE*2.2)+
  geom_point(data=M,aes(colour=ifelse(x==1,'#7F7F7F',col),fill=fill),shape=21,size=PT_LARGE,stroke=.55)+
  facet_grid(.~sc)+scale_colour_identity()+scale_fill_identity()+
  scale_x_continuous(breaks=1:2,labels=c('Shuffled','Virtual'),expand=expansion(add=c(.42,.42)))+
  scale_y_continuous(labels=num_minus_acc(0.1),breaks=seq(-1,1.5,.5))+
  coord_cartesian(clip='off')+
  labs(x=NULL,y='Gain over single layer (%)')+BASE+theme(panel.spacing.x=unit(3,'mm'),plot.margin=margin(5,1,1,1.5,'mm'))
 # ---- label column: its own gtable column after the last facet (width LW mm), y in the shared panel scale
 LWD<-9.4
 opened<-if(dev.cur()==1){pdf(NULL);TRUE}else FALSE
 bt<-ggplot_build(p);g<-ggplotGrob(p);if(opened)dev.off()
 yr<-bt$layout$panel_params[[1]]$y.range;stopifnot(all(abs(bt$layout$panel_params[[2]]$y.range-yr)<1e-9))
 pl<-g$layout[grepl('^panel',g$layout$name),];r0<-max(pl$r)
 g<-gtable::gtable_add_cols(g,unit(LWD,'mm'),pos=r0)
 lab<-gTree(children=do.call(gList,lapply(seq_len(nrow(L)),function(i)textGrob(L$lab[i],x=unit(1.3,'mm'),y=unit((L$ly[i]-yr[1])/diff(yr),'npc'),
  just=c('left','centre'),gp=gpar(fontfamily=CELL_FONT,fontsize=6,col=L$col[i],lineheight=.9)))),name='e_labels')
 g<-gtable::gtable_add_grob(g,lab,t=min(pl$t),b=max(pl$b),l=r0+1,clip='off',name='e_labels')
 list(plot=p,grob=g)}

# FigureS1 A (v4) <- v3 Figure 3C: observed vs predicted AUC (five observed-AUC bins per fold, pooled across folds
# weighted by cell count) for the six most covered drugs, as a full-width 2 x 6 strip of small multiples
# (columns = drugs, named once on top; rows = New cell lines / Held-out tissue, labelled at the right).
# Composite gtable: the two facet blocks are stacked with gtable rbind (equal panel heights, common columns), the
# shared axis titles and row labels are added as gtable cells. Guard applied to both blocks (no titles; text >= 6 pt).
# Fix round 2 (display only): (i) one x range per drug (union of its two rows, via geom_blank) -> both rows of a drug
# have identical x limits, breaks and labels; (ii) one y label accuracy per drug column (the larger number of decimals
# the two rows need, found in a first ggplot_build pass); (iii) smaller markers (S1_PT_A) so six series stay separable.
# Arm key: drawn once in slot LEG_ARM under B/C (serves A, B, C). Free y per facet; no identity line.
s1_breaks<-function(lims){for(n in 2:6){b<-pretty(lims,n);b<-b[b>=lims[1]-1e-12&b<=lims[2]+1e-12];if(length(b)>=2){if(length(b)>3)b<-b[seq(1,length(b),by=2)];return(b)}};seq(lims[1],lims[2],length.out=2)}
# x axis: a facet is ~18.6 mm wide and a 6 pt label like '0.70' ~4.1 mm, so three labels need a step of >= 0.36 of the
# data range; otherwise keep only the outer two (both still inside the facet).
s1_breaks_x<-function(lims){b<-s1_breaks(lims);if(length(b)==3&&diff(b)[1]/diff(range(lims))<.36)b<-b[c(1,3)];b}
s1_need_dd<-function(b){b<-b[is.finite(b)];if(!length(b))return(2);for(dd in 2:5){s<-formatC(b,format='f',digits=dd);if(all(abs(as.numeric(s)-b)<1e-9)&&!anyDuplicated(s))return(dd)};5}
s1_labels<-function(x){o<-rep(NA_character_,length(x));f<-is.finite(x);o[f]<-formatC(x[f],format='f',digits=s1_need_dd(x));o}
s1_bkey<-function(b){b<-b[is.finite(b)];paste(sprintf('%.9f',b),collapse='|')}
make_A<-function(){arms<-S1_ARMS[1:5]
 sel<-read_table('drug_selection')%>%filter(rank<=6)%>%arrange(rank)
 d<-read_table('calibration_bins')%>%filter(drug_id%in%sel$drug_id)%>%group_by(scenario,drug_id,name,arm,bin)%>%
  summarise(x=weighted.mean(observed_auc,n),y=weighted.mean(predicted_auc,n),cells=sum(n),.groups='drop')%>%
  mutate(name=factor(name,levels=sel$name),arm=factor(arm,levels=arms),scenario_label=S1_SCEN[scenario])
 xr<-d%>%group_by(name)%>%summarise(x0=min(x),x1=max(x),.groups='drop')
 xb<-d%>%group_by(scenario,name)%>%summarise(y=median(y),.groups='drop')%>%left_join(xr,by='name')%>%
  tidyr::pivot_longer(c(x0,x1),names_to='end',values_to='x')%>%select(scenario,name,x,y)
 block<-function(sc,top,ylab=s1_labels){p<-ggplot(filter(d,scenario==sc),aes(x,y,colour=arm,shape=arm,linetype=arm))+
   geom_blank(data=filter(xb,scenario==sc),aes(x,y),inherit.aes=FALSE)+
   geom_line(linewidth=LW_LINE)+geom_point(size=S1_PT_A,stroke=.25)+
   facet_wrap(~name,ncol=6,scales='free')+coord_cartesian(clip='off')+s1_arm_scales(arms)+
   scale_x_continuous(breaks=s1_breaks_x,labels=s1_labels,expand=expansion(mult=.1))+
   scale_y_continuous(breaks=s1_breaks,labels=ylab,expand=expansion(mult=.1))+labs(x=NULL,y=NULL)+
   BASE+theme(legend.position='none',panel.spacing.x=unit(2.2,'mm'),strip.clip='off',
    plot.margin=if(top)margin(5,0,0,0,'mm')else margin(1.6,0,0,0,'mm'))
  if(!top)p<-p+theme(strip.text=element_blank(),strip.background=element_blank())
  cell_guard(p,'A');p}
 opened<-.nullpdf();on.exit(if(opened)dev.off())
 # pass 1: y breaks of every facet -> decimals each drug column needs (max over its two rows)
 info<-bind_rows(lapply(c('new_cell','tissue'),function(sc){bp<-ggplot_build(block(sc,sc=='new_cell'));L<-bp$layout$layout
  bind_rows(lapply(seq_len(nrow(L)),function(i){b<-bp$layout$panel_params[[i]]$y$breaks;tibble(scenario=sc,name=as.character(L$name[i]),key=s1_bkey(b),dd=s1_need_dd(b))}))}))
 info<-info%>%group_by(name)%>%mutate(dd=max(dd))%>%ungroup()
 clash<-info%>%group_by(key)%>%summarise(n=n_distinct(dd),.groups='drop')%>%filter(n>1)
 if(nrow(clash))warning('A: identical y breaks in drug columns that need different accuracies: ',paste(clash$key,collapse=' ; '),call.=FALSE)
 ymap<-tapply(info$dd,info$key,max)
 ylab<-function(x){o<-rep(NA_character_,length(x));f<-is.finite(x);dd<-ymap[s1_bkey(x)];if(is.na(dd))dd<-s1_need_dd(x);o[f]<-formatC(x[f],format='f',digits=dd);o}
 cat('A: y decimals per drug column:',paste(sprintf('%s=%d',unique(info$name),info$dd[!duplicated(info$name)]),collapse=', '),'\n')
 g1<-ggplotGrob(block('new_cell',TRUE,ylab));g2<-ggplotGrob(block('tissue',FALSE,ylab))
 w<-grid::unit.pmax(g1$widths,g2$widths);g1$widths<-w;g2$widths<-w
 n1<-nrow(g1);g<-rbind(g1,g2,size='first');g$widths<-w
 pr<-function(gt,off=0){l<-gt$layout[grepl('^panel',gt$layout$name),];list(t=min(l$t)+off,b=max(l$b)+off,l=min(l$l),r=max(l$r))}
 p1<-pr(g1);p2<-pr(g2,n1)
 tg<-function(t,rot=0,bold=FALSE,size=PT_AXIS_TITLE)textGrob(t,rot=rot,gp=gpar(fontfamily=CELL_FONT,fontsize=size,fontface=if(bold)'bold'else'plain',col=INK))
 # right: 1.6 mm gap + 3 mm rotated row labels + 0.4 mm edge; left: 3.6 mm y title; bottom: 3.6 mm x title + 0.6 mm
 g<-gtable::gtable_add_cols(g,unit(1.6,'mm'));g<-gtable::gtable_add_cols(g,unit(3,'mm'));rc<-ncol(g);g<-gtable::gtable_add_cols(g,unit(.4,'mm'))
 g<-gtable::gtable_add_grob(g,tg(S1_SCEN[['new_cell']],rot=-90,bold=TRUE,size=PT_STRIP),t=p1$t,b=p1$b,l=rc,r=rc,clip='off',name='row1')
 g<-gtable::gtable_add_grob(g,tg(S1_SCEN[['tissue']],rot=-90,bold=TRUE,size=PT_STRIP),t=p2$t,b=p2$b,l=rc,r=rc,clip='off',name='row2')
 g<-gtable::gtable_add_cols(g,unit(3.6,'mm'),pos=0)
 g<-gtable::gtable_add_grob(g,tg('Predicted AUC',rot=90),t=p1$t,b=p2$b,l=1,r=1,clip='off',name='ytitle')
 g<-gtable::gtable_add_rows(g,unit(3.6,'mm'));g<-gtable::gtable_add_grob(g,tg('Observed AUC'),t=nrow(g),b=nrow(g),l=p1$l+1,r=p1$r+1,clip='off',name='xtitle')
 g<-gtable::gtable_add_rows(g,unit(.6,'mm'))
 pfull<-ggplot(d,aes(x,y,colour=arm))+geom_point()+BASE   # guard + panel_data dump of both scenarios (A_display.csv)
 list(plot=pfull,grob=g)}

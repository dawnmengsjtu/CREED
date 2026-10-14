# FigureS3 B (v4) <- v3 Figure 5D, lower row only (vs matched random panels): pre-registered negative N2. Context-adaptive
# panels (per tissue, New cell lines only; per k-means cluster of M) versus matched random panels of the same size.
# Gray points = per-fold relative improvement (positive favours the adaptive panel); black bars = improvement of the
# five-fold means. Values from plot_data d_context_fold / d_context_summary (v3 copy, reference == matched_random;
# nothing recomputed). Strips are nested: the scenario (house names) spans its variants and is drawn as one extra gtable
# row above the variant strips (ggh4x is not installed), so "New cell lines" is printed once over its two variants.
make_B<-function(){fd<-read_table('d_context_fold');sm<-read_table('d_context_summary')
 VAR<-c(tissue_adaptive='Tissue-adaptive',cluster_adaptive='Cluster-adaptive');SC<-c('New cell lines','Held-out tissue');SCN<-setNames(SC,c('New cell','Whole tissue'))
 FAC<-c('New cell lines|Tissue-adaptive','New cell lines|Cluster-adaptive','Held-out tissue|Cluster-adaptive')
 prep<-function(d){d<-d[d$reference=='matched_random',];d$scenario_label<-unname(SCN[d$scenario_label]);d$variant_label<-unname(VAR[d$variant])
  d$facet<-factor(paste(d$scenario_label,d$variant_label,sep='|'),levels=FAC);d$kf<-factor(d$k,levels=c(16,32,64));d}
 fd<-prep(fd);sm<-prep(sm);stopifnot(!anyNA(fd$facet),!anyNA(sm$facet),nrow(fd)==45,nrow(sm)==9)
 p<-ggplot(fd,aes(x=kf,y=improvement_pct))+geom_hline(yintercept=0,linewidth=LW_THIN,colour=INK)+
  geom_fold_points()+
  geom_errorbar(data=sm,aes(x=kf,ymin=improvement_pct_of_means,ymax=improvement_pct_of_means),width=.6,linewidth=.7/.pt,colour=INK,inherit.aes=FALSE)+
  facet_grid(.~facet,labeller=as_labeller(function(x)sub('^.*[|]','',x)),axes='all',axis.labels='margins')+
  scale_y_continuous(breaks=seq(-1,1.5,.5),labels=num_minus_acc(0.1))+
  labs(x='Panel size, k',y='Improvement vs random (%)')+BASE+theme(panel.spacing.x=unit(2.5,'mm'))
 opened<-.nullpdf();on.exit(if(opened)dev.off())
 g<-ggplotGrob(p);st<-g$layout[grepl('^strip-t',g$layout$name),];st<-st[order(st$l),];row<-unique(st$t);stopifnot(length(row)==1,nrow(st)==3)
 g<-gtable_add_rows(g,unit(3.8,'mm'),pos=row-1)
 span<-function(lab)gTree(children=gList(
  textGrob(lab,y=unit(2.3,'mm'),gp=gpar(fontfamily=CELL_FONT,fontsize=PT_STRIP,fontface='bold',col=INK)),
  segmentsGrob(x0=unit(.6,'mm'),x1=unit(1,'npc')-unit(.6,'mm'),y0=unit(.4,'mm'),y1=unit(.4,'mm'),gp=gpar(col=INK,lwd=LW_AXIS*.pt,lineend='butt'))))
 g<-gtable_add_grob(g,span(SC[1]),t=row,l=st$l[1],r=st$r[2],clip='off',name='strip-span-1')
 g<-gtable_add_grob(g,span(SC[2]),t=row,l=st$l[3],r=st$r[3],clip='off',name='strip-span-2')
 list(plot=p,grob=g)}

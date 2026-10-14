# Figure 4F - pre-registered negative N2: context-adaptive protein panels against the global static panel (k = 16, 32, 64),
# both variants on one axis, dodged within k: tissue-adaptive (New cell lines only; tissue-label orange), cluster-adaptive
# (k-means clusters of M; dark purple) for New cell lines (filled) and Held-out tissue (open).
# Points = per-fold relative gain (d_context_fold improvement_pct, reference = global_static); crossbars = gain of the
# five-fold means (d_context_summary improvement_pct_of_means), drawn in the colour of their own points. Variants
# labelled directly.
make_F<-function(){f<-read_table('d_context_fold');s<-read_table('d_context_summary')
 f<-f[f$reference=='global_static',];s<-s[s$reference=='global_static',]
 prep<-function(z){z$col<-ifelse(z$variant=='tissue_adaptive',F4_TIS,F4_CLU)
  z$fill<-ifelse(z$scenario=='new_cell',z$col,'white')
  off<-ifelse(z$variant=='tissue_adaptive',-.28,ifelse(z$scenario=='new_cell',0,.28))
  z$xc<-match(z$k,c(16,32,64))+off;z}
 f<-prep(f);s<-prep(s);stopifnot(nrow(f)==45,nrow(s)==9)
 f$x<-f$xc+(as.numeric(f$fold)-2)*.035
 lab<-data.frame(x=3.5,y=c(-.58,-.73),txt=c('Tissue-adaptive','Cluster-adaptive'),col=c(F4_TIS,F4_CLU))
 p<-ggplot()+
  geom_hline(yintercept=0,linewidth=LW_AXIS,colour=INK)+
  geom_point(data=f,aes(x,improvement_pct,colour=col,fill=fill),shape=21,size=PT_SMALL,stroke=.5,alpha=.9)+
  geom_errorbar(data=s,aes(x=xc,ymin=improvement_pct_of_means,ymax=improvement_pct_of_means,colour=col),width=.24,linewidth=LW_LINE*2)+
  geom_text(data=lab,aes(x,y,label=txt,colour=col),hjust=1,size=6/.pt,family=CELL_FONT)+
  scale_colour_identity()+scale_fill_identity()+
  scale_x_continuous(breaks=1:3,labels=c('16','32','64'),limits=c(.55,3.5),expand=expansion(0))+
  scale_y_continuous(labels=num_minus_acc(0.1),breaks=seq(-.8,.8,.4),limits=c(-.9,.75),expand=expansion(0))+
  labs(x='Panel size, k\n(measured molecules)',y='Gain over global panel (%)')+BASE
 f4_fix(p,top=8.8,bottom=11)}

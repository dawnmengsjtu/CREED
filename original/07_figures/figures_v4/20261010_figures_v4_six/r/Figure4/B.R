# Figure 4B (hero) - the panel law R(k). Rows = borrowed layer (protein panels, gene panels), columns = scenario.
# Ranked (supervised) panels: line + large symbols = R from five-fold mean sRMSE (b_recovery_curve); small tinted points
# just left of each k = per-fold R (b_recovery_fold). Random panels of the same size: pale line + open symbols with the
# seed range (R_lo..R_hi), small pale points just right of each k = per-fold R. 'all' = full candidate pool (teacher
# side), placed one tick spacing after k = 128 with an axis break mark (the k axis is log2 up to 128).
# Broken y axis: the gene row has a narrow upper strip (1.48-1.62) for the single per-fold value above the main range
# (Held-out tissue, F1, k = 32, R = 1.56); the gap between the strip and the gene row is the axis break.
# Direct labels: protein row at the curve ends (the empty k = 128 column; protein panels stop at k = 64); gene row: Ranked
# inside, Random at the end of the random line (just right of the open k = 32 symbol and its fold points), as in the protein row.
make_B<-function(){cv<-read_table('b_recovery_curve');fd<-read_table('b_recovery_fold')
 xpos<-function(k,kn)ifelse(k=='all',F4_KX,log2(kn))
 lay<-function(s)ifelse(grepl('^protein',s),'Protein panels','Gene panels')
 typ<-function(s)ifelse(grepl('random',s),'Random','Ranked')
 prep<-function(z){z$layer<-lay(z$series);z$type<-typ(z$series);z$sc<-f4_sc(z$scenario);z$x0<-xpos(z$k,z$k_num);z}
 cv<-prep(cv);fd<-prep(fd);stopifnot(!anyNA(cv$sc),!anyNA(fd$sc),!anyNA(cv$x0),!anyNA(fd$x0))
 YHI<-1.4;fd$row<-ifelse(fd$layer=='Protein panels','Protein panels',ifelse(fd$R>YHI,'hi','Gene panels'))
 cv$row<-cv$layer;RL<-c('Protein panels','hi','Gene panels')
 cv$row<-factor(cv$row,levels=RL);fd$row<-factor(fd$row,levels=RL)
 COL<-c(`Protein panels`=F4_PROT,`Gene panels`=F4_GENE)
 cv$col<-COL[cv$layer];fd$col<-COL[fd$layer]
 fd$x<-fd$x0+ifelse(fd$type=='Ranked',-.22,.22)+(as.numeric(fd$fold)-2)*.025
 sup<-cv[cv$type=='Ranked',];rnd<-cv[cv$type=='Random',]
 lim<-data.frame(row=factor(rep(RL,each=2),levels=RL),y=c(-.1,1.18,1.48,1.62,-.1,1.18),x=3)
 lim<-merge(lim,data.frame(sc=factor(F4_SC,levels=F4_SC)))
 refs<-merge(data.frame(row=factor(c('Protein panels','Gene panels'),levels=RL)),data.frame(sc=factor(F4_SC,levels=F4_SC)))
 # line-end labels for the protein row (New cell lines), at the k = 64 ends of the ranked and random curves
 e<-function(z)z$R[z$layer=='Protein panels'&z$scenario=='new_cell'&z$k=='64']
 lab<-data.frame(row=factor(rep('Protein panels',2),levels=RL),sc=factor(rep(F4_SC[1],2),levels=F4_SC),
  x=c(6.42,6.42),y=c(e(sup)+.06,e(rnd)-.07),txt=c('Ranked','Random'),col=c(F4_PROT,alpha_hex(F4_PROT,.8)))
 g32<-rnd$R[rnd$layer=='Gene panels'&rnd$scenario=='new_cell'&rnd$k=='32'];stopifnot(length(g32)==1)
 lab<-rbind(lab,data.frame(row=factor(rep('Gene panels',2),levels=RL),sc=factor(rep(F4_SC[1],2),levels=F4_SC),
  x=c(2.62,5.40),y=c(.80,g32-.01),txt=c('Ranked','Random'),col=c(F4_GENE,alpha_hex(F4_GENE,.9))))
 # x axis drawn by hand in the bottom (gene) row so it can carry a break mark between k = 128 and 'all'
 xa<-merge(data.frame(row=factor('Gene panels',levels=RL),x=c(2.55,7.62),xend=c(7.38,8.45),y=-.1),data.frame(sc=factor(F4_SC,levels=F4_SC)))
 xb<-merge(data.frame(row=factor('Gene panels',levels=RL),x=c(7.31,7.55),xend=c(7.45,7.69),y=-.14,yend=-.06),data.frame(sc=factor(F4_SC,levels=F4_SC)))
 ggplot()+
  geom_hline(data=refs,aes(yintercept=0),linetype='12',linewidth=LW_THIN,colour=INK)+
  geom_hline(data=refs,aes(yintercept=1),linetype='22',linewidth=LW_LINE,colour=F4_REF)+
  geom_point(data=fd[fd$type=='Random',],aes(x,R,colour=col),size=PT_SMALL-.3,shape=16,alpha=.35)+
  geom_point(data=fd[fd$type=='Ranked',],aes(x,R,colour=col),size=PT_SMALL-.2,shape=16,alpha=.55)+
  geom_errorbar(data=rnd,aes(x=x0,ymin=R_lo,ymax=R_hi,colour=col),width=0,linewidth=LW_ERR,alpha=.6)+
  geom_line(data=rnd,aes(x0,R,colour=col,group=interaction(series,sc)),linewidth=LW_LINE,linetype='42',alpha=.6)+
  geom_point(data=rnd,aes(x0,R,colour=col),fill='white',shape=21,size=PT_LARGE,stroke=.6)+
  geom_line(data=sup[sup$k!='all',],aes(x0,R,colour=col,group=interaction(series,sc)),linewidth=LW_LINE*1.6)+
  geom_point(data=sup,aes(x0,R,colour=col),size=PT_LARGE+.2,shape=16)+
  geom_text(data=lab,aes(x,y,label=txt,colour=col),size=6/.pt,family=CELL_FONT,hjust=0)+
  geom_segment(data=xa,aes(x=x,xend=xend,y=y,yend=y),linewidth=LW_AXIS,colour=INK,lineend='square')+
  geom_segment(data=xb,aes(x=x,xend=xend,y=y,yend=yend),linewidth=LW_AXIS,colour=INK)+
  geom_blank(data=lim,aes(x,y))+
  facet_grid(row~sc,scales='free_y',space='free_y',labeller=labeller(row=c(`Protein panels`='Protein panels',hi='',`Gene panels`='Gene panels')))+
  coord_cartesian(clip='off')+
  scale_colour_identity()+
  scale_x_continuous(breaks=c(3:7,F4_KX),labels=c('8','16','32','64','128','all'),limits=c(2.55,F4_KX+.45),expand=expansion(0))+
  scale_y_continuous(breaks=c(0,.5,1,1.5),minor_breaks=seq(0,1.5,.25),labels=num_minus_acc(0.1),expand=expansion(0),limits=function(r)c(max(r[1],-.1),r[2]),oob=scales::oob_keep)+
  guides(y=guide_axis(minor.ticks=TRUE))+
  labs(x='Panel size, k (measured molecules)',y='Recovery fraction R(k)')+BASE+
  theme(axis.line.x=element_blank(),axis.minor.ticks.length=rel(.5),
   panel.spacing.y=unit(c(4,1.2),'mm'),panel.spacing.x=unit(4,'mm'),strip.text.y=element_text(angle=-90,size=PT_STRIP,face='bold',margin=margin(0,0,0,1.2,'mm')))}

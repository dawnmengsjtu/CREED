# Figure 5F - ... yet the recovered share holds: one lane per endpoint (circle GDSC above, triangle CRISPR below) for each
# metabolite-row pair (M->RNA, M->P), two stacked facets with top strips (New cell lines, Held-out tissue), like B, C, E and
# G (fix round; was rotated right strips). Large symbols = share of the fold-averaged gains (b_share_mean share_pct), in the
# pair colour, filled = new cell lines, open = held-out tissue; small light-gray symbols = per-fold shares
# (b_share_fold share_fold_pct). Printed at the right: b_share_mean share_label. Dashed dark-gray (#4D4D4D) line = one third.
make_F<-function(){
 lane<-c(`M→RNA.GDSC`=4.1,`M→RNA.CRISPR`=3.3,`M→P.GDSC`=1.9,`M→P.CRISPR`=1.1)
 m<-read_table('b_share_mean')%>%mutate(sc=factor(F5_SC[scenario],levels=F5_SC),y=lane[paste(pair,endpoint,sep='.')],col=F5_PAIR_COL[pair],fill=f5_fill(col,scenario))
 f<-read_table('b_share_fold')%>%mutate(sc=factor(F5_SC[scenario],levels=F5_SC),y=lane[paste(pair,endpoint,sep='.')])
 stopifnot(nrow(m)==8,nrow(f)==40,all(f$share_fold_pct>15),all(f$share_fold_pct<62))
 XV<-64.4
 ggplot(m,aes(share_pct,y))+
  geom_vline(xintercept=100/3,linetype='22',linewidth=LW_LINE,colour=F5_MEAS)+
  geom_point(data=f,aes(share_fold_pct,y,shape=endpoint),size=PT_SMALL,fill=F5_FOLDPT,colour='white',stroke=.25)+
  geom_point(aes(shape=endpoint,fill=fill,colour=col),size=PT_LARGE+.2,stroke=.65)+
  geom_text(aes(x=XV,y=y,label=share_label,colour=col),hjust=0,size=6/.pt,family=CELL_FONT)+
  facet_wrap(~sc,ncol=1)+scale_fill_identity()+scale_colour_identity()+scale_shape_manual(values=F5_SHAPE,guide='none')+
  scale_x_continuous(breaks=c(20,40,60),expand=c(0,0))+scale_y_continuous(breaks=c(3.7,1.5),labels=c('M→RNA','M→P'),expand=c(0,0))+
  coord_cartesian(xlim=c(15,63),ylim=c(.55,4.65),clip='off')+
  labs(x='Recovered share (%)',y=NULL)+BASE+
  theme(panel.spacing.y=unit(1.2,'mm'),axis.ticks.y=element_blank(),plot.margin=margin(5,6.5,1,1,'mm'))}

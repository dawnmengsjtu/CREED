# Figure 2F - recovery versus reconstruction (fix round 2026-10-10). One point per directed pair per scenario
# (xl_matrix: x = generator out-of-fold R2, generator_oof_r2; y = recovered gain over the deployed layer alone,
# recovered_gain), filled = new cell lines, open = held-out tissue (the page-wide key is in A), the two scenarios of a
# pair joined by a thin line; per-fold points (xl_pair_folds oof_r2_fold, recovered_fold) small behind. Pairs are
# labelled directly next to their points with short leaders that stop before the symbols; M->RNA / M->P in their
# colours, the rest one muted gray. x limits 0.05-0.52 (all points lie in 0.06-0.51).
# 2026-10-11 (audit A4-07): x title writes R² as one italic run with the superscript-two glyph at the 7 pt title size
# (the plotmath superscript was drawn at 4.9 pt, below the 6 pt minimum); vphantom(italic(R)^2) keeps the old title
# height, so the E | F plot areas (align_h r3) do not move.
make_F<-function(){
 m<-read_table('xl_matrix')%>%mutate(col=F2_PAIR_COL[pair],fill=f2_fill(col,scenario))
 f<-read_table('xl_pair_folds')%>%mutate(col=F2_PAIR_COL[pair],fill=f2_fill(col,scenario))
 stopifnot(min(f$oof_r2_fold)>.05,max(f$oof_r2_fold)<.52,min(f$recovered_fold)>-.9,max(f$recovered_fold)<1.4)
 lab<-tibble::tribble(~pair,~lx,~ly,~hj,
  'M→RNA',0.322,1.02,0, 'M→P',0.163,0.86,0, 'RNA→M',0.143,-0.06,0, 'P→M',0.149,-0.55,0, 'RNA→P',0.338,-0.37,0, 'P→RNA',0.500,0.40,1)%>%
  mutate(col=F2_PAIR_COL[pair])
 led<-tibble::tribble(~pair,~x1,~y1,~x2,~y2,
  'M→RNA',0.319,1.00,0.277,0.93, 'RNA→P',0.340,-0.33,0.327,-0.235)%>%
  mutate(col=F2_PAIR_COL[pair])
 ggplot(m,aes(generator_oof_r2,recovered_gain))+
  geom_hline(yintercept=0,linewidth=LW_THIN,colour='#9E9E9E')+
  geom_point(data=f,aes(oof_r2_fold,recovered_fold,colour=col,fill=fill),shape=21,size=PT_SMALL-.3,stroke=.3,alpha=.55)+
  geom_line(aes(group=pair,colour=col),linewidth=LW_LINE)+
  geom_point(aes(colour=col,fill=fill),shape=21,size=PT_LARGE+.3,stroke=.6)+
  geom_segment(data=led,aes(x1,y1,xend=x2,yend=y2,colour=col),linewidth=LW_THIN)+
  geom_text(data=lab,aes(lx,ly,label=pair,colour=col,hjust=hj),size=6.5/.pt,family=CELL_FONT)+
  scale_colour_identity()+scale_fill_identity()+
  scale_x_continuous(limits=c(0.05,0.52),breaks=seq(0.1,0.5,0.1),labels=num_minus_acc(0.1),expand=c(0,0))+
  scale_y_continuous(labels=num_minus_acc(0.5),breaks=seq(-1,1.5,.5),limits=c(-0.9,1.4),expand=c(0,0))+
  coord_cartesian(clip='off')+
  labs(x=expression('Generator out-of-fold '*italic('R²')*vphantom(italic(R)^2)),y='Recovered gain (%)')+BASE+theme(plot.margin=margin(5,2.5,1,1.5,'mm'))}

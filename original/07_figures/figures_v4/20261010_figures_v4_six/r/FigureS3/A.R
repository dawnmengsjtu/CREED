# FigureS3 A (v4) <- v3 Figure 5C, lower facet only: paired difference in sRMSE from M only, per fold, for the null,
# M only, M + 16 P, M + 64 P and M + measured P values; facets New cell lines | Held-out tissue (house names; the plot_data labels
# "New cell" / "Whole tissue" are mapped). Folds F0-F4 = FOLD_RAMP joined within fold; black bars = five-fold means
# (delta_mean); dashed line = null (delta_mean of the null arm); thin solid line = zero. Values from plot_data
# c_abs_fold / c_abs_mean (v3 copy, unchanged; nothing recomputed). The v3 top facet (absolute sRMSE, with the off-scale
# Held-out tissue F1 triangles) is not part of this panel, so no off-scale marker is needed: every delta is on scale.
# Fold legend drawn once, horizontally above the strips: it fills the band that align_h (group r1) opens above A's
# strips to match B's two-row nested strips, so A's plot top still coincides with B's.
# 2026-10-11 (audits A3-12, A5-06): the panel-law teacher (plot_data label 'M + full proteome'; value kernel on the
# proteins observed in >= 50% of fit lines) is labelled 'M + measured P values', as in Figure 4, distinct from the
# value + mask reference 'M + measured P' of Figure 2 and Table S2.
make_A<-function(){fd<-read_table('c_abs_fold');mn<-read_table('c_abs_mean')
 SC<-c('New cell lines','Held-out tissue');SCN<-setNames(SC,c('New cell','Whole tissue'))
 short<-c('Null (per-drug mean)'='Null','M only'='M only','M + 16 proteins'='M + 16 P','M + 64 proteins'='M + 64 P','M + full proteome'='M + measured P values')
 prep<-function(d){d$arm_s<-factor(unname(short[d$arm_label]),levels=short);d$scenario_label<-factor(unname(SCN[d$scenario_label]),levels=SC);d}
 fd<-prep(fd);mn<-prep(mn);fd$fold<-factor(fd$fold,levels=0:4)
 stopifnot(!anyNA(fd$arm_s),!anyNA(fd$scenario_label),nrow(fd)==50,nrow(mn)==10)
 nul<-mn[mn$arm=='null_drug',]
 ggplot(fd,aes(x=arm_s,y=delta_vs_m_only))+
  geom_hline(yintercept=0,linewidth=LW_THIN,colour=INK)+
  geom_hline(data=nul,aes(yintercept=delta_mean),linetype='22',linewidth=LW_LINE,colour=INK)+
  geom_line(aes(group=fold,colour=fold),linewidth=LW_THIN,alpha=.9)+
  geom_point(aes(colour=fold),size=PT_SMALL)+
  geom_errorbar(data=mn,aes(x=arm_s,ymin=delta_mean,ymax=delta_mean),width=.55,linewidth=.7/.pt,colour=INK,inherit.aes=FALSE)+
  facet_grid(.~scenario_label,axes='all',axis.labels='margins')+scale_colour_fold()+
  scale_y_continuous(breaks=seq(-0.04,0.04,0.02),labels=num_minus_acc(0.01))+
  labs(x=NULL,y='Δ sRMSE vs M only')+BASE+
  theme(legend.position='top',legend.direction='horizontal',legend.box.spacing=unit(.8,'mm'),
        axis.text.x=element_text(angle=35,hjust=1,vjust=1),panel.spacing.x=unit(3,'mm'))}

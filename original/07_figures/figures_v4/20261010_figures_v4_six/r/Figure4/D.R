# Figure 4D - what the measured panel buys in absolute terms: paired gain over M only in the same fold,
# Gain over M only (sRMSE) = sRMSE[M only] - sRMSE[arm], i.e. the sign flip of the source difference (c_abs_mean
# delta_mean, c_abs_fold delta_vs_m_only = sRMSE[arm] - sRMSE[M only]); positive = lower error than M only (better), as
# in F and Figure 2D. Rows: Null, M + 16 measured P, M + 64 measured P and M + measured P values (the saved protein teacher of
# B: the measured protein values without the missingness-mask features; Figure 6 shows the same arm under the same name). Horizontal bars = five-fold mean; points = the five folds, one light grey
# with a white rim. Null = shuffle/null grey #BDBDBD (no outline); both protein-panel arms in full protein-panel
# vermilion (k is read from the label); M + measured P values = dark-grey measured teacher.
# Final polish (2026-10-11): the teacher row is named 'M + measured P values' (was 'M + measured P', which is the house name of
# a different arm, sRMSE 0.988 vs 0.986 for this teacher; see Figure S3 legend); the legend no longer uses 'no mask'. Black = the zero reference line only.
make_D<-function(){fd<-read_table('c_abs_fold');mn<-read_table('c_abs_mean')
 A<-c(null_drug='Null',m_panel_k16='M + 16 measured P',m_panel_k64='M + 64 measured P',teacher_saved='M + measured P values')
 key<-function(z){z$a<-A[z$arm];z<-z[!is.na(z$a),];z$a<-factor(z$a,levels=rev(A));z$sc<-f4_sc(z$scenario);z}
 fd<-key(fd);mn<-key(mn);stopifnot(nrow(fd)==40,nrow(mn)==8)
 mn$gain<- -mn$delta_mean;fd$gain<- -fd$delta_vs_m_only
 FILL<-c(Null=F4_NULL,`M + 16 measured P`=F4_PROT,`M + 64 measured P`=F4_PROT,`M + measured P values`=F4_REF)
 fd$y<-as.numeric(fd$a)+(as.numeric(as.character(fd$fold))-2)*.11
 ggplot(mn,aes(gain,a))+
  geom_col(aes(fill=a),colour=NA,width=.7,orientation='y')+
  geom_vline(xintercept=0,linewidth=LW_AXIS,colour=INK)+
  geom_point(data=fd,aes(gain,y),fill=F4_FOLD,colour='white',size=PT_SMALL-.1,shape=21,stroke=.35)+
  facet_wrap(~sc,nrow=1)+
  scale_fill_manual(values=FILL,guide='none')+
  scale_x_continuous(labels=num_minus_acc(0.01),breaks=c(-.03,0,.03),limits=c(-.044,.044),expand=expansion(0))+
  scale_y_discrete(expand=expansion(add=.5))+
  labs(x='Gain over M only (sRMSE)',y=NULL)+BASE+
  theme(axis.line.y=element_blank(),axis.ticks.y=element_blank(),panel.spacing.x=unit(3,'mm'))}

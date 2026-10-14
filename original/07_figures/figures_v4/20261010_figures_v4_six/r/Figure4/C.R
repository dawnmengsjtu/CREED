# Figure 4C - layer-agnostic panel law: gene-panel R(k) against protein-panel R(k) at the same k (k = 8-64, ranked panels).
# Small symbols = one fold (b_recovery_fold; same fold, same k, both layers); large symbols = R from five-fold mean sRMSE
# (b_recovery_curve), joined in k order by 0.5-pt dark-green lines (solid = New cell lines, dashed = Held-out tissue).
# Colour = k (one sequential green ramp #A1D99B -> #00441B; key = equal-size swatches above the plot at right, outside
# the plot area); filled = New cell lines, open = Held-out tissue. Dashed black diagonal = equal recovery (reference).
# Ticks 0.0 / 0.5 / 1.0 (minor 0.25) as in B.
# Broken y axis: a ~4 mm upper strip holds the one per-fold gene value above the main range (Held-out tissue, F1,
# k = 32: gene 1.56, protein 0.96).
make_C<-function(){cv<-read_table('b_recovery_curve');fd<-read_table('b_recovery_fold')
 K<-c('8','16','32','64');S<-c('protein_supervised','gene_supervised')
 wide<-function(z,keep){z<-z[z$series%in%S&z$k%in%K,c(keep,'series','R')];z<-tidyr::pivot_wider(z,names_from=series,values_from=R)
  z$kf<-factor(z$k,levels=K);z$sc<-f4_sc(z$scenario);as.data.frame(z)}
 f<-wide(fd,c('scenario','k','fold'));m<-wide(cv,c('scenario','k'))
 stopifnot(nrow(f)==40,nrow(m)==8,!anyNA(f$gene_supervised),!anyNA(m$protein_supervised))
 YHI<-1.3;f$row<-factor(ifelse(f$gene_supervised>YHI,'hi','main'),levels=c('hi','main'));m$row<-factor('main',levels=c('hi','main'))
 f$fill<-ifelse(f$scenario=='new_cell',F4_K[as.character(f$k)],'white');m$fill<-ifelse(m$scenario=='new_cell',F4_K[as.character(m$k)],'white')
 m<-m[order(m$scenario,as.numeric(m$k)),];m$lt<-ifelse(m$scenario=='new_cell','solid','22')
 lim<-data.frame(row=factor(c('hi','hi','main','main'),levels=c('hi','main')),protein_supervised=.5,gene_supervised=c(1.49,1.61,-.06,1.1))
 diag<-data.frame(row=factor('main',levels=c('hi','main')))
 ggplot(f,aes(protein_supervised,gene_supervised))+
  geom_abline(data=diag,aes(slope=1,intercept=0),linetype="22",linewidth=LW_THIN,colour=INK)+
  geom_point(aes(colour=F4_K[as.character(k)],fill=fill),shape=21,size=PT_SMALL,stroke=.5,alpha=.9)+
  geom_path(data=m,aes(group=scenario,linetype=lt),colour=F4_KLINE,linewidth=LW_LINE)+
  geom_point(data=m,aes(colour=F4_K[as.character(k)],fill=fill),shape=21,size=PT_LARGE+.4,stroke=.8)+
  geom_blank(data=lim)+
  facet_grid(row~.,scales='free_y',space='free_y')+
  scale_linetype_identity()+
  scale_colour_identity(guide='legend',breaks=unname(F4_K),labels=names(F4_K),name='k')+scale_fill_identity()+
  guides(colour=guide_legend(override.aes=list(shape=22,fill=unname(F4_K),colour=unname(F4_K),size=2.4,stroke=0,alpha=1),nrow=1,title.position='left'),
   x=guide_axis(minor.ticks=TRUE),y=guide_axis(minor.ticks=TRUE))+
  scale_x_continuous(limits=c(-.06,1.1),breaks=c(0,.5,1),minor_breaks=seq(0,1,.25),labels=num_minus_acc(0.1),expand=expansion(0))+
  scale_y_continuous(breaks=c(0,.5,1,1.5),minor_breaks=seq(0,1,.25),labels=num_minus_acc(0.1),expand=expansion(0))+
  labs(x='Protein-panel R(k)',y='Gene-panel R(k)')+BASE+
  legend_title_on()+theme(strip.text.y=element_blank(),panel.spacing.y=unit(1.6,'mm'),axis.minor.ticks.length=rel(.5),
   legend.position='top',legend.justification=c(1,.5),legend.direction='horizontal',legend.box.spacing=unit(.8,'mm'),
   legend.key.size=unit(2.4,'mm'),legend.key.spacing.x=unit(.8,'mm'),legend.margin=margin(-3.2,0,0,0,'mm'),legend.background=element_blank())}

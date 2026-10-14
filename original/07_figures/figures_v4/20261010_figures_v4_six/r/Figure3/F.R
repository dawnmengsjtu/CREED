# Figure 3F (<- v4 3F) - dumbbell: mean within-cell delta sMSE (M only - arm) by tissue (per_sample_metrics.csv, cells with
# a finite sMSE under every arm), virtual arm joined to its measured reference: upper dumbbell RNA (blue dot -> dark-gray
# square), lower dumbbell P (pink dot -> dark-gray triangle). Tissues (Figure 1 short names) ordered by evaluable cell
# lines, counts in a right-hand column headed n (identical in both scenarios). Scenarios as facets, one x range.
make_F<-function(){arms<-c('M+vRNA','M+vP','M+RNA','M+P')
 w<-read_pd(FIG,'per_sample_metrics')%>%filter(arm%in%c('M',arms))%>%select(scenario,sample_id,tissue,arm,smse)%>%pivot_wider(names_from=arm,values_from=smse)%>%filter(if_all(c('M',all_of(arms)),is.finite))
 d<-w%>%group_by(scenario,tissue)%>%summarise(across(all_of(arms),~mean(M-.x)),n=n(),.groups='drop')
 nn<-d%>%distinct(scenario,tissue,n)%>%pivot_wider(names_from=scenario,values_from=n,names_prefix='n_');stopifnot(all(nn$n_new_cell==nn$n_tissue),sum(nn$n_new_cell)==419)
 nn<-nn%>%arrange(n_new_cell,tissue);ord<-nn$tissue
 dd<-bind_rows(d%>%transmute(scenario,tissue,layer='RNA',v=`M+vRNA`,m=`M+RNA`),d%>%transmute(scenario,tissue,layer='P',v=`M+vP`,m=`M+P`))%>%
  mutate(row=match(tissue,ord),y=row+ifelse(layer=='RNA',.19,-.19),vc=ifelse(layer=='RNA',LAYER_COL[['RNA']],LAYER_COL[['P']]),ms=ifelse(layer=='RNA',15,17),
   scenario=factor(F3_SC[scenario],levels=F3_SC))
 stopifnot(!anyNA(dd$row),min(c(dd$v,dd$m))>-.07,max(c(dd$v,dd$m))<.17)
 p<-ggplot(dd,aes(y=y))+geom_vline(xintercept=0,linewidth=LW_THIN,colour=F3_ZERO)+
  geom_segment(aes(x=v,xend=m,yend=y,colour=vc),linewidth=LW_LINE,alpha=.45)+
  geom_point(aes(x=m,shape=ms),colour=F3_MEAS,size=.95)+
  geom_point(aes(x=v,colour=vc),shape=16,size=1.05)+
  facet_wrap(~scenario,ncol=2,axes='all',axis.labels='margins')+scale_colour_identity()+scale_shape_identity()+
  scale_y_continuous(breaks=seq_along(ord),labels=unname(f3_tissue(ord)),expand=expansion(add=.6),sec.axis=sec_axis(~.,breaks=seq_along(ord),labels=as.character(nn$n_new_cell)))+
  scale_x_continuous(limits=c(-.07,.17),breaks=c(0,.1),labels=c('0','0.1'),expand=c(0,0))+
  labs(x='Δ sMSE (M only − arm)',y=NULL)+BASE+theme(panel.spacing=unit(2.4,'mm'),axis.ticks.y=element_blank(),axis.line.y.right=element_blank(),
   axis.text.y.right=element_text(size=PT_TICK,colour=INK,hjust=1,margin=margin(l=.5,unit='mm')),plot.margin=margin(1,.6,.6,.6,'mm'))
 cell_guard(p,'F')
 g<-f3_grob(p);lay<-g$layout;ar<-lay[grepl('^axis-r',lay$name)&vapply(seq_len(nrow(lay)),function(i)!inherits(g$grobs[[i]],'zeroGrob'),TRUE),]
 ar<-ar[which.max(ar$l),];st<-lay[grep('^strip-t',lay$name),]
 g<-gtable::gtable_add_grob(g,textGrob('n',x=unit(1,'npc'),y=unit(1.2,'mm'),hjust=1,vjust=0,gp=gpar(fontfamily=CELL_FONT,fontsize=PT_STRIP,fontface='bold',col=INK)),
   t=min(st$t),l=ar$l,r=ar$r,name='count-header',clip='off')
 list(plot=p,grob=fix_panel_margins(g,top=F3_TOP,id='F'))}

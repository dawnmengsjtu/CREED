# Figure 3C (<- v4 3H) - bar chart (the figure's only bars; baseline 0): gain in sRMSE from a training-time sample-graph
# regulariser over the same-input no-graph arm (deployment input M only). Bar = five-fold mean gain (fig4_graph_summary
# mean_pct; the numeric-P value is the run's verbatim comparison), small points = single folds (fig4_graph rel_gain_pct).
# Scenario = filled bar + filled points (new cell lines, upper) vs open bar + open points (held-out tissue, lower); the
# page's only filled/open key, drawn in dark gray #4D4D4D (fix round: not the M-graph gray). Graph colours: M graph = M gray,
# shuffled P = shuffle gray, masked P = light pink, numeric P = P pink.
# Fix round 2 (2026-10-10): the plot-area top is aligned with B (F3_TOP = 6.4 mm below the slot top), so the key sits in
# the top-right band beside the letter (where B has its facet strips) and the M graph row starts at the plot top; the
# y expansion was cut so the four graph groups fill the plot height.
F3C_COL<-c(metabolite=LAYER_COL[['M']],shuffle=ARM_COL_CELL[['shuffle']],mask='#E3B5CF',protein=LAYER_COL[['P']])
make_C<-function(){
 d<-read_pd(FIG,'fig4_graph')%>%filter(arm!='no_graph');s<-read_pd(FIG,'fig4_graph_summary')
 arms<-names(F3C_COL);lab<-c(metabolite='M graph',shuffle='Shuffled P graph',mask='Masked P graph',protein='Numeric P graph')
 yy<-setNames(rev(seq_along(arms)),arms);off<-c(new_cell=.2,tissue=-.2)
 s$gain<-ifelse(s$arm=='protein',s$mean_pct_verbatim,s$mean_pct);stopifnot(!anyNA(s$gain),nrow(s)==8,nrow(d)==40)
 s<-s%>%mutate(y=yy[arm]+off[scenario],col=F3C_COL[arm],fill=ifelse(scenario=='new_cell',alpha_hex(col,.5),'white'),sc=F3_SC[scenario])
 d<-d%>%mutate(y=yy[arm]+off[scenario],col=F3C_COL[arm],fill=ifelse(scenario=='new_cell',col,'white'),sc=factor(F3_SC[scenario],levels=F3_SC))
 p<-ggplot(d,aes(rel_gain_pct,y))+
  geom_rect(data=s,aes(xmin=0,xmax=gain,ymin=y-.17,ymax=y+.17,fill=fill,colour=col),linewidth=LW_THIN,inherit.aes=FALSE)+
  geom_vline(xintercept=0,linewidth=LW_AXIS,colour=INK)+
  geom_point(aes(fill=fill,colour=col,shape=sc),size=1.05,stroke=.45)+
  scale_fill_identity()+scale_colour_identity()+
  scale_shape_manual(values=c(21,21),name=NULL,guide=guide_legend(override.aes=list(fill=c(F3_MEAS,'white'),colour=F3_MEAS,size=1.6,stroke=.5)))+
  scale_y_continuous(breaks=yy,labels=unname(lab[names(yy)]),expand=expansion(add=c(.32,.32)))+
  scale_x_continuous(breaks=c(0,.25,.5),labels=c('0','0.25','0.50'),expand=expansion(mult=c(.06,.04)))+
  labs(x='Gain over no-graph arm (%)',y=NULL)+
  BASE+theme(axis.ticks.y=element_blank(),axis.line.y=element_blank(),legend.position='top',legend.direction='vertical',legend.justification=c(1,1),
   legend.margin=margin(0,0,0,0),legend.box.spacing=unit(.6,'mm'),legend.key.size=unit(2.4,'mm'),legend.key.spacing.y=unit(0,'mm'),
   legend.text=element_text(size=PT_LEGEND,margin=margin(l=.6,unit='mm')),plot.margin=margin(1,1.5,.8,.6,'mm'))
 cell_guard(p,'C')
 list(plot=p,grob=fix_panel_margins(p,top=F3_TOP,id='C'))}

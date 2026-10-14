# FigureS2 C <- v3 Figure 6D (reduced): paired drug-level outcomes vs M only (five-fold mean standardized MSE lower or not)
# for three arms; ribbons follow single drugs and carry the annotation tone (opaque, same colours as A; key in LEG_AB).
# Strata are light gray with a thin mid-gray outline (Figure 1F style). "Lower" / "Not lower" are y-axis labels placed,
# per facet, at the centre of the y band that lies inside the Lower (resp. Not lower) stratum on ALL three axes
# (computed from drug_flows), so each label lines up with its stratum on every axis; the panel shares A's left plot edge.
make_C<-function(){st<-function(x)factor(ifelse(x=='Lower error','Lower','Not lower'),levels=c('Lower','Not lower'))
 d<-annotation_data('drug_flows')%>%transmute(scenario,tone=group_tone(group),a1=st(status_m_vrna),a2=st(status_m_panel32),a3=st(status_m_rna_measured))%>%count(scenario,tone,a1,a2,a3,name='n')
 # strata stack Lower on top of Not lower with no gap: Not lower spans [0, nl_k] on axis k, Lower spans [nl_k, tot]
 l0<-d%>%group_by(scenario)%>%summarise(tot=sum(n),nl1=sum(n[a1=='Not lower']),nl2=sum(n[a2=='Not lower']),nl3=sum(n[a3=='Not lower']),.groups='drop')%>%
  mutate(nl_min=pmin(nl1,nl2,nl3),nl_max=pmax(nl1,nl2,nl3),y_not=nl_min/2,y_low=(nl_max+tot)/2)
 brk<-function(lim){k<-which.min(abs(l0$tot-max(lim)));c(l0$y_not[k],l0$y_low[k])}
 ggplot(d,aes(axis1=a1,axis2=a2,axis3=a3,y=n))+
  geom_alluvium(aes(fill=tone),alpha=1,width=.42,colour=NA,knot.pos=.35)+
  geom_stratum(width=.42,fill=STRATUM_FILL,colour=STRATUM_LINE,linewidth=LW_THIN)+
  geom_text(stat='stratum',aes(label=after_stat(count)),size=GEOM_TEXT_MIN,family=CELL_FONT)+
  facet_wrap(~scenario,ncol=1,scales='free_y')+scale_fill_manual(values=TONE_COL,name=NULL,drop=FALSE,guide='none')+
  scale_x_discrete(limits=unname(S2_ARM_LABEL[c('m_vrna','m_panel32','m_rna_measured')]),expand=expansion(add=c(.22,.42)))+
  scale_y_continuous(breaks=brk,labels=function(b)ifelse(b==min(b),'Not lower','Lower'),expand=expansion(mult=c(.01,.01)))+
  labs(x=NULL,y=NULL)+BASE+
  theme(axis.line=element_blank(),axis.ticks=element_blank(),axis.text.x=element_text(hjust=.5),panel.spacing=unit(3,'mm'),legend.position='none')}

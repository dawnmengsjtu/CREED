# Figure 2G - generator-size learning curve (fix round 2026-10-10; a_curve_mean / a_curve_fold): recovered share (%) of
# the room versus the number of generator training cells (fold mean of n_gen; 50 %, 75 %, 100 % of the fit cells, plus
# the extra identity-safe paired lines for virtual RNA, joined by a dotted segment). M + virtual RNA and M + virtual P in
# their colours, filled = new cell lines, open = held-out tissue (page key in A); per-fold shares small behind; dashed
# line = one third of the room. y starts at 5 % (all per-fold shares >= 8.5 %). Direct labels and the '1/3' tag sit at
# the right ends of the lines (x = 424, just past the plot area). Plot top/bottom fixed (6.2 / 9.5 mm) to align with H.
# 2026-10-11 (audit A4-17): x title 'Paired generator lines' (was 'Generator training cells'; the legend counts lines).
make_G<-function(){
 lab_s<-c(RNA='M + virtual RNA',P='M + virtual P')
 m<-read_table('a_curve_mean')%>%mutate(col=ifelse(layer=='RNA',ARM_COL_CELL[['m_vrna']],ARM_COL_CELL[['m_vp']]),fill=f2_fill(col,scenario),g=paste(layer,scenario))
 f<-read_table('a_curve_fold')%>%mutate(col=ifelse(layer=='RNA',ARM_COL_CELL[['m_vrna']],ARM_COL_CELL[['m_vp']]),fill=f2_fill(col,scenario))
 stopifnot(nrow(m)==14,nrow(f)==70,min(f$share_pct)>5,max(f$share_pct)<66)
 core<-m%>%filter(is_extra==0);ext<-m%>%group_by(g)%>%filter(any(is_extra==1))%>%filter(size%in%c('f100','extra'))%>%ungroup()
 XR<-420;XL<-424
 lab<-tibble::tibble(x=XL,y=c(42,25.5),lab=unname(lab_s[c('RNA','P')]),col=c(ARM_COL_CELL[['m_vrna']],ARM_COL_CELL[['m_vp']]))
 f<-f%>%mutate(xj=n_gen_fold+ifelse(scenario=='new_cell',-4,4)+ifelse(layer=='RNA',-2,2))
 p<-ggplot(m,aes(n_gen_mean,share_pct))+
  geom_hline(yintercept=100/3,linetype='22',linewidth=LW_LINE,colour=INK)+
  annotate('text',x=XL,y=100/3,label='1/3',hjust=0,size=6/.pt,family=CELL_FONT)+
  geom_point(data=f,aes(xj,share_pct,colour=col,fill=fill),shape=21,size=PT_SMALL-.3,stroke=.3,alpha=.5)+
  geom_line(data=core,aes(group=g,colour=col),linewidth=LW_LINE*1.4)+
  geom_line(data=ext,aes(group=g,colour=col),linewidth=LW_LINE*1.4,linetype='11')+
  geom_point(aes(colour=col,fill=fill),shape=21,size=PT_LARGE+.2,stroke=.6)+
  geom_text(data=lab,aes(x=x,y=y,label=lab,colour=col),hjust=0,size=6.5/.pt,family=CELL_FONT)+
  scale_colour_identity()+scale_fill_identity()+
  scale_x_continuous(breaks=c(200,300,400),expand=c(0,0))+
  scale_y_continuous(breaks=seq(20,60,20),expand=c(0,0))+
  coord_cartesian(xlim=c(150,XR),ylim=c(5,66),clip='off')+
  labs(x='Paired generator lines',y='Recovered share (%)')+BASE+theme(plot.margin=margin(5,19,1,1.5,'mm'))
 list(plot=p,grob=fix_panel_margins(p,top=F2_BOT_TOP,bottom=F2_BOT_BOTTOM,id=ID))}

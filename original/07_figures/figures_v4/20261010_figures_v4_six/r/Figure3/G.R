# Figure 3G (new view of the per-cell data behind v4 3F) - mirrored histogram: per-cell-line delta sMSE (M only - arm) for
# the 419 cell lines with a finite sMSE under every arm (per_sample_metrics.csv; each line is scored once, in its held-out
# fold). Upward bars = M + virtual RNA (blue), downward = M + virtual P (pink); dark-gray outline = the matching measured
# reference (M + measured RNA up, M + measured P down). Bins 0.025 wide on -0.2 to 0.4. Fix round: values below -0.2 or
# above 0.4 are no longer folded into the outer bins; they are counted in one tail bin each, set off past an axis break
# (white gap + '//') and ticked '< -0.2' / '> 0.4' (counts in the legend). Scenarios as rows with top strips (as B, E, F);
# the zero line is solid gray (dashes mean only virtual and null on this page).
make_G<-function(){arms<-c('M+vRNA','M+vP','M+RNA','M+P')
 w<-read_pd(FIG,'per_sample_metrics')%>%filter(arm%in%c('M',arms))%>%select(scenario,sample_id,arm,smse)%>%pivot_wider(names_from=arm,values_from=smse)%>%filter(if_all(c('M',all_of(arms)),is.finite))
 stopifnot(all(table(w$scenario)==419))
 lo<- -.2;hi<-.4;bw<-.025;br<-seq(lo,hi,bw);gp<-.02;nb<-length(br)-1
 xmin<-lo-gp-bw-.006;xmax<-hi+gp+bw+.006
 l<-w%>%pivot_longer(all_of(arms),names_to='arm',values_to='a')%>%mutate(delta=M-a,
   bin=ifelse(delta<lo,0L,ifelse(delta>hi,nb+1L,findInterval(delta,br,rightmost.closed=TRUE))))
 pooled<-l%>%group_by(scenario,arm)%>%summarise(below=sum(delta<lo),above=sum(delta>hi),.groups='drop')
 write_csv(pooled,file.path(fig_out(FIG),'panel_data','G_pooled_tail_counts.csv'))
 h<-l%>%count(scenario,arm,bin)%>%complete(scenario,arm,bin=0:(nb+1),fill=list(n=0))%>%
  mutate(x0=ifelse(bin==0,lo-gp-bw,ifelse(bin==nb+1,hi+gp,br[pmax(pmin(bin,nb),1)])),x1=x0+bw,
   sgn=ifelse(arm%in%c('M+vRNA','M+RNA'),1,-1),yv=sgn*n,virtual=arm%in%c('M+vRNA','M+vP'),layer=ifelse(sgn>0,'RNA','P'),tail=bin%in%c(0,nb+1),
   scenario=factor(F3_SC[scenario],levels=F3_SC))
 stopifnot(all((h%>%group_by(scenario,arm)%>%summarise(s=sum(n),.groups='drop'))$s==419))
 hv<-filter(h,virtual,n>0);hm<-filter(h,!virtual)
 # measured outline: closed step path over the main range per (scenario, layer); tail bins as their own outlines
 st<-hm%>%filter(!tail)%>%arrange(scenario,layer,bin)%>%group_by(scenario,layer)%>%reframe(x=c(lo,as.vector(rbind(x0,x1)),hi),y=c(0,rep(yv,each=2),0))
 mt<-hm%>%filter(tail,n>0)
 yl<-range(h$yv)*1.06
 p<-ggplot()+
  geom_rect(data=hv,aes(xmin=x0+.002,xmax=x1-.002,ymin=0,ymax=yv,fill=layer),colour=NA)+
  geom_path(data=st,aes(x,y,group=layer),colour=F3_MEAS,linewidth=LW_LINE)+
  geom_rect(data=mt,aes(xmin=x0,xmax=x1,ymin=0,ymax=yv),fill=NA,colour=F3_MEAS,linewidth=LW_LINE)+
  geom_hline(yintercept=0,linewidth=LW_THIN,colour=INK)+
  geom_vline(xintercept=0,linewidth=LW_THIN,colour=F3_ZERO)+
  facet_wrap(~scenario,ncol=1)+
  scale_fill_manual(values=c(RNA=LAYER_COL[['RNA']],P=LAYER_COL[['P']]),guide='none')+
  scale_x_continuous(breaks=c(lo-gp-bw/2,0,.2,hi+gp+bw/2),labels=c('< −0.2','0','0.2','> 0.4'),expand=c(0,0))+
  scale_y_continuous(labels=function(v)as.character(abs(v)),breaks=c(-100,0,100))+
  coord_cartesian(xlim=c(xmin,xmax),ylim=yl,expand=FALSE,clip='off')+
  labs(x='Δ sMSE (M only − arm)',y='Cell lines')+BASE+
  theme(panel.spacing.y=unit(2.2,'mm'),plot.margin=margin(1,2.6,.6,.6,'mm'))
 cell_guard(p,'G')
 g<-f3_grob(p);pn<-f3_panels(g,'t');bot<-pn[length(pn)];npx<-function(x)(x-xmin)/(xmax-xmin)
 for(xb in c(lo-gp/2,hi+gp/2))g<-f3_add_break(g,bot,'b',npx(xb))
 list(plot=p,grob=fix_panel_margins(g,top=F3_TOP,bottom=8.2,id='G'))}

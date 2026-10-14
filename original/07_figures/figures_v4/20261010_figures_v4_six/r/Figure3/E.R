# Figure 3E (<- v4 3C) - scatter: per-drug M-only RMSE (log x) against its reduction by each virtual arm (drug_rmse.csv;
# drugs evaluated in all five folds: 295 new cell lines, 281 held-out tissue). Each scenario has its own x window fitted to
# the bulk of its drugs at the same mm per log unit (new cell lines 0.82-1.42, held-out tissue 0.94-1.60). Fix round: the
# x axis is broken just past the window (white gap + '//') and the drugs beyond it (2 and 14) are drawn as filled,
# arm-coloured, right-pointing triangles in a narrow strip past the break, ticked '> 1.42' / '> 1.6' (the 1.4 tick of
# the new-cell-line window is dropped so it does not collide with '> 1.42'; no open marks:
# open = held-out tissue elsewhere on the page). Dashed black = fold-mean sRMSE of the training-mean null (headroom.csv
# null, mean over folds); gray line = zero.
# Fix round 2 (2026-10-10): axis titles follow the page convention of D, F and G ('Δ RMSE (M only − arm)', 'Per-drug RMSE,
# M only'; RMSE kept because the source metric is per-drug RMSE). In the off-scale strip each arm has its own sub-column
# (virtual RNA blue on the left, virtual P pink on the right); inside a sub-column, triangles that would overlap are
# spread over up to five lanes 0.5 mm apart (greedy, in order of reduction), so every one of the 2 / 14 drugs per arm is
# visible. The strip was widened to hold the two sub-columns.
# Final polish (2026-10-11): axis titles in house metric naming, 'Δ sRMSE (M only − arm)' and 'Per-drug sRMSE, M only'. The
# per-drug value is sqrt(mean over the five folds of the per-drug MSE after scaling by the training SD) (prepare_fig3_v2.py,
# drug_rmse), i.e. the per-drug sRMSE, on the same scale as the null sRMSE line; D, F and G already use 'Δ sMSE (M only − arm)'.
F3E_WIN<-list(new_cell=c(.82,1.42),tissue=c(.94,1.60))
F3E_PAD<-.006;F3E_BRK<-.011;F3E_STRIP<-.036;F3E_END<-.060   # log10 units: left pad; break, strip centre, right edge past the window
F3E_SUB<-1.6;F3E_LANE<-.5;F3E_TRI_W<-1.0;F3E_TRI_H<-1.1;F3E_MINDY<-.6   # mm: sub-column half-distance, lane step, triangle size, min vertical gap per lane
f3e_lanes<-function(ymm){o<-order(ymm);lane<-integer(length(ymm));last<-list();ord<-c(0,1,-1,2,-2)
 for(i in o){ok<-FALSE;for(L in ord){k<-as.character(L);if(is.null(last[[k]])||all(abs(ymm[i]-last[[k]])>=F3E_MINDY)){lane[i]<-L;last[[k]]<-c(last[[k]],ymm[i]);ok<-TRUE;break}}
  if(!ok){dmin<-vapply(as.character(ord),function(k)min(abs(ymm[i]-last[[k]])),0);L<-ord[which.max(dmin)];lane[i]<-L;last[[as.character(L)]]<-c(last[[as.character(L)]],ymm[i])}}
 lane}
make_E<-function(){arms<-c('M+vRNA','M+vP')
 lo<-unlist(lapply(F3E_WIN,`[`,1));hi<-unlist(lapply(F3E_WIN,`[`,2))
 d<-read_pd(FIG,'drug_rmse')%>%filter(arm%in%c('M',arms))%>%pivot_wider(names_from=arm,values_from=rmse)%>%pivot_longer(all_of(arms),names_to='arm',values_to='virtual')%>%
  mutate(reduction=M-virtual,beyond=M>hi[scenario],x=ifelse(beyond,hi[scenario]*10^F3E_STRIP,M),arm=factor(arm,levels=arms))
 stopifnot(all(d$M>=lo[d$scenario]))
 nb<-d%>%filter(arm=='M+vRNA')%>%group_by(scenario)%>%summarise(n=n(),beyond=sum(beyond),maxM=max(M));stopifnot(identical(as.integer(nb$n),c(295L,281L)),identical(as.integer(nb$beyond),c(2L,14L)))
 cat(sprintf('E: beyond-window max M-only RMSE %s = %.4f\n',nb$scenario,nb$maxM))
 write_csv(d%>%filter(beyond)%>%select(scenario,drug_id,arm,M,reduction),file.path(fig_out(FIG),'panel_data','E_beyond_window_drugs.csv'))
 sN<-hi[['new_cell']]*10^F3E_STRIP;sT<-hi[['tissue']]*10^F3E_STRIP
 rng<-data.frame(scenario=names(F3E_WIN),x0=lo*10^-F3E_PAD,x1=hi*10^F3E_END)
 brk_fun<-function(lim){L<-if(max(lim)<.6)10^lim else lim;if(min(L)<mean(c(rng$x0[1],rng$x0[2])))c(.9,1,1.2,sN)else c(1,1.2,1.4,sT)}   # facet told apart by its left edge (0.81 vs 0.93)
 lab_fun<-function(b)ifelse(abs(b-sN)<1e-9,'> 1.42',ifelse(abs(b-sT)<1e-9,'> 1.6',formatC(b,format='f',digits=1)))
 yr<-range(d$reduction);yl<-c(min(yr[1]-.04*diff(yr),-.042),yr[2]+.04*diff(yr))
 sc<-function(z)factor(F3_SC[z],levels=F3_SC)
 dd<-d%>%mutate(scenario=sc(scenario))
 nul<-read_pd(FIG,'headroom')%>%distinct(scenario,fold,null)%>%group_by(scenario)%>%summarise(null=mean(null),.groups='drop')%>%mutate(scenario=sc(scenario))
 win<-data.frame(scenario=sc(rep(rng$scenario,2)),x=c(rng$x0,rng$x1),y=0)
 p<-ggplot(dd,aes(x,reduction,colour=arm))+geom_blank(data=win,aes(x,y),inherit.aes=FALSE)+
  geom_hline(yintercept=0,linewidth=LW_THIN,colour=F3_ZERO)+
  geom_vline(data=nul,aes(xintercept=null),linetype='22',linewidth=LW_LINE,colour=INK)+
  geom_point(data=filter(dd,!beyond),size=.75,alpha=.55,shape=16)+
  geom_blank(data=filter(dd,beyond))+      # beyond-window drugs: drawn as triangles on the gtable below (exact mm shape)
  facet_wrap(~scenario,ncol=2,scales='free_x',axes='all',axis.labels='all_x')+
  scale_x_log10(breaks=brk_fun,labels=lab_fun,expand=c(0,0))+
  scale_y_continuous(breaks=c(-.04,0,.04,.08),labels=num_minus_acc(0.01))+
  scale_colour_manual(values=F3_COL[arms],guide='none')+coord_cartesian(ylim=yl,expand=FALSE,clip='off')+
  labs(x='Per-drug sRMSE, M only',y='Δ sRMSE (M only − arm)')+BASE+theme(panel.spacing=unit(4,'mm'),plot.margin=margin(1,1.5,.6,.6,'mm'))
 cell_guard(p,'E')
 g<-f3_grob(p);pn<-f3_panels(g,'l')
 lw<-log10(rng$x1)-log10(rng$x0);g$widths[g$layout$l[match(pn,g$layout$name)]]<-unit(lw/sum(lw)*2,'null')
 g<-fix_panel_margins(g,top=F3_TOP,id='E')
 # panel height in mm (slot height minus every fixed row) for the lane test
 opened<-if(dev.cur()==1){pdf(NULL);TRUE}else FALSE
 hs<-g$heights;isnull<-vapply(seq_along(hs),function(j)grid::unitType(hs[j])=='null',TRUE)
 ph<-SLOT_H-sum(vapply(which(!isnull),function(j)convertHeight(hs[j],'mm',valueOnly=TRUE),0));if(opened)dev.off()
 lanes_out<-list()
 for(i in seq_along(pn)){s<-rng$scenario[i];L0<-log10(rng$x0[i]);L1<-log10(rng$x1[i]);npx<-function(x)(log10(x)-L0)/(L1-L0)
  g<-f3_add_break(g,pn[i],'b',npx(hi[[s]]*10^F3E_BRK))
  b<-d%>%filter(scenario==s,beyond)%>%arrange(arm);if(!nrow(b))next
  xn<-npx(hi[[s]]*10^F3E_STRIP);yn<-(b$reduction-yl[1])/diff(yl);n<-nrow(b)
  b$sub<-ifelse(b$arm=='M+vRNA',-F3E_SUB,F3E_SUB);b$lane<-0L
  for(a in arms){k<-which(b$arm==a);if(length(k))b$lane[k]<-f3e_lanes(yn[k]*ph)}
  xo<-b$sub+b$lane*F3E_LANE;lanes_out[[s]]<-b%>%mutate(x_offset_mm=xo,panel_h_mm=ph)
  tri<-polygonGrob(x=unit(rep(xn,each=3),'npc')+unit(rep(xo,each=3)+rep(c(-.45,-.45,.55)*F3E_TRI_W,n),'mm'),
    y=unit(rep(yn,each=3),'npc')+unit(rep(c(.5,-.5,0)*F3E_TRI_H,n),'mm'),
    id=rep(seq_len(n),each=3),gp=gpar(fill=F3_COL[as.character(b$arm)],col='white',lwd=.25))
  g<-f3_add_grob(g,pn[i],tri,paste0('beyond-',s))}
 write_csv(bind_rows(lanes_out)%>%select(scenario,drug_id,arm,M,reduction,lane,x_offset_mm,panel_h_mm),file.path(fig_out(FIG),'panel_data','E_beyond_window_lanes.csv'))
 cat(sprintf('E: panel height %.1f mm; lanes used %s\n',ph,paste(sapply(lanes_out,function(z)paste(range(z$lane),collapse='..')),collapse=' / ')))
 list(plot=p,grob=g)}

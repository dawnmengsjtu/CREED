# Figure 6B (hero, fix round 2, 2026-10-10) - ONE dense GDSC forest: every GDSC arm of the sealed run (21 rows in six
# groups) as gain over M only with its 95 % grouped-bootstrap interval (filled symbol, sealed set); the new-cell-line
# (development-fold) value, where one exists, as an open circle just under its row; random panels = square (mean of five
# seeds, 35 % fill) with the five seeds as small dots and no interval; measured layers on a shaded band (references, not
# deployable). x axis -0.5 to 6; the endpoint is named in the axis title (no in-panel header).
# Middle strip column: per-target win fraction of the paired sign test, wins / (wins + losses), ties excluded, read
# verbatim from summary.json analysis/<endpoint>/arms/<arm>/sign_test_vs_M/fraction_better, for GDSC (circle, upper)
# and CRISPR (triangle, lower) in the same row; random panels: the five seeds as 1.4-size symbols with a white rim on a
# thin min-max line (no mean is drawn: none exists in the sources); dashed line = one half.
# Right-hand verdict column (moved here from the former gauge C): P-code and tick / cross beside the rows each gain
# item tests - P1 M + virtual RNA, P2 M + virtual P, P3 the two paired shuffles (bracket), P6 M + tissue label,
# P7 M + protein graph. Verdicts from protocol.json, asserted equal to d_scoreboard.csv. Win counts: legend / Table
# S-sealed. Sources: forest.csv / forest_seeds.csv (verbatim), summary.json, protocol.json, d_scoreboard.csv.
make_B<-function(){s<-read_summary_v4('Figure6');v<-f6_verdicts()
 d<-f6_forest_rows('drug');stopifnot(nrow(d)==21)
 # row lattice: header unit + one unit per row, 0.55-unit gap (with a rule) between groups
 y<-numeric(nrow(d));hdr<-data.frame();sep<-c();cur<-0;prev<-''
 for(i in seq_len(nrow(d))){g<-d$group[i]
  if(g!=prev){if(prev!=''){sep<-c(sep,cur-.275);cur<-cur-.55};hdr<-rbind(hdr,data.frame(y=cur-.5,label=F6_GROUP_LABEL[[g]],grp=g));cur<-cur-1;prev<-g}
  y[i]<-cur-.5;cur<-cur-1}
 d$y<-y;NU<- -cur
 TOPB<-6.4;BOTB<-9;u<-(SLOT_H-TOPB-BOTB)/NU          # mm per row unit
 WM<-21.5;WV<-8.6                                    # strip-column and verdict-column widths (mm)
 WF<-SLOT_W-WM-WV;lc<-33;mr<-.5;xlim<-c(-.5,6);pw<-WF-lc-mr;upm<-diff(xlim)/pw
 x_lab<-xlim[1]-(lc-.6)*upm;x_row<-x_lab+1.4*upm;x_end<-xlim[2]+mr*upm
 sd<-read_table('forest_seeds')%>%filter(endpoint=='drug');sd$y<-d$y[match(sd$mean_arm,d$arm)];sd$colkey<-d$colkey[match(sd$mean_arm,d$arm)]
 sd$col<-alpha_hex(F6_COL[sub('_random','',sd$colkey)],.75);stopifnot(nrow(sd)==20,!any(is.na(sd$y)))
 da<-d[d$kind!='random_mean',];dr<-d[d$kind=='random_mean',];dv<-d[!is.na(d$dev_gain_pct),];stopifnot(nrow(dv)==6)
 band_top<-hdr$y[hdr$grp=='reference']+.5+.275
 dv$yd<-dv$y-.36
 p1<-ggplot()+
  annotate('rect',xmin=x_lab,xmax=x_end,ymin=-NU,ymax=band_top,fill=F6_BAND,colour=NA)+
  annotate('segment',x=x_lab,xend=x_end,y=sep,yend=sep,colour=F6_RULE,linewidth=LW_THIN)+
  annotate('segment',x=0,xend=0,y=-NU,yend=0,colour=INK,linewidth=LW_AXIS)+
  geom_err_h(data=da,aes(xmin=lo,xmax=hi,y=y,colour=colkey))+
  geom_point(data=sd,aes(gain_pct,y),colour=sd$col,size=.75,shape=16)+
  geom_point(data=da,aes(gain_pct,y,colour=colkey),shape=16,size=1.75)+
  geom_point(data=dr,aes(gain_pct,y,fill=colkey,colour=sub('_random','',colkey)),shape=22,size=1.55,stroke=.5)+
  geom_point(data=dv,aes(dev_gain_pct,yd,colour=colkey),shape=21,fill='white',size=1.45,stroke=.55)+
  geom_text(data=hdr,aes(x=x_lab,y=y,label=label),hjust=0,vjust=.5,size=F6_GROUP_PT/.pt,fontface='bold',colour=INK)+
  geom_text(data=d,aes(x=x_row,y=y,label=row_label),hjust=0,vjust=.5,size=GEOM_TEXT_MIN,colour=INK)+
  scale_colour_manual(values=F6_COL,guide='none')+scale_fill_manual(values=F6_COL,guide='none')+
  scale_x_continuous(breaks=0:6,labels=num_minus)+scale_y_continuous(breaks=NULL)+
  coord_cartesian(xlim=xlim,ylim=c(-NU,0),expand=FALSE,clip='off')+
  labs(x='Gain over M only (%), GDSC',y=NULL)+
  BASE+theme(axis.line.y=element_blank(),axis.ticks.y=element_blank(),axis.text.y=element_blank(),axis.title.y=element_blank(),
             legend.position='none',plot.margin=margin(0,mr,0,lc,'mm'))
 # ---- strip column: win fractions, both endpoints (ties excluded), every random seed
 seeds_of<-function(a)if(grepl('rand.*_mean$',a))paste0(sub('_mean$','',a),'_s',0:4)else a
 rows<-list();for(ep in c('drug','crispr'))for(i in seq_len(nrow(d)))for(a in seeds_of(d$arm[i])){st<-s$analysis[[ep]]$arms[[a]]$sign_test_vs_M
  if(is.null(st))next;rows[[length(rows)+1]]<-data.frame(endpoint=ep,row_arm=d$arm[i],arm=a,frac=st$fraction_better,wins=st$wins,losses=st$losses,ties=st$ties,
   y=d$y[i],colkey=sub('_random','',d$colkey[i]),seed=grepl('_s[0-4]$',a))}
 w<-do.call(rbind,rows);stopifnot(all(abs(w$frac-w$wins/(w$wins+w$losses))<1e-12),sum(w$endpoint=='drug')==37,sum(w$endpoint=='crispr')==33)
 off_g<-.5/u;off_c<- -.55/u                         # GDSC 0.5 mm above, CRISPR 0.55 mm below the row line
 w$yy<-w$y+ifelse(w$endpoint=='drug',off_g,off_c);w$col<-F6_COL[w$colkey]
 ws<-w[w$seed,];wa<-w[!w$seed,]
 wr<-ws%>%group_by(row_arm,endpoint,yy,colkey)%>%summarise(lo=min(frac),hi=max(frac),.groups='drop')   # display extent of the five seeds
 write_csv(w,file.path(fig_out(FIG),'panel_data','B_winfraction_display.csv'))
 xm<-c(.27,.93)
 p2<-ggplot()+
  annotate('rect',xmin=-Inf,xmax=Inf,ymin=-NU,ymax=band_top,fill=F6_BAND,colour=NA)+
  annotate('segment',x=-Inf,xend=Inf,y=sep,yend=sep,colour=F6_RULE,linewidth=LW_THIN)+
  annotate('segment',x=.5,xend=.5,y=-NU,yend=0,linetype='22',linewidth=LW_LINE,colour=INK)+
  geom_segment(data=wr,aes(x=lo,xend=hi,y=yy,yend=yy),colour=F6_COL[wr$colkey],linewidth=LW_THIN)+
  geom_point(data=ws,aes(frac,yy,shape=endpoint),fill=alpha_hex(ws$col,.85),colour='white',size=ifelse(ws$endpoint=='drug',1.4,1.5),stroke=.22)+
  geom_point(data=wa,aes(frac,yy,shape=endpoint),fill=wa$col,colour=wa$col,size=ifelse(wa$endpoint=='drug',1.45,1.6),stroke=0)+
  scale_shape_manual(values=c(drug=21,crispr=24),guide='none')+
  scale_x_continuous(breaks=c(.3,.5,.7,.9),labels=num_minus_acc(.1))+scale_y_continuous(breaks=NULL)+
  coord_cartesian(xlim=xm,ylim=c(-NU,0),expand=FALSE,clip='off')+
  labs(x='Win fraction',y=NULL)+
  BASE+theme(axis.line.y=element_blank(),axis.ticks.y=element_blank(),axis.text.y=element_blank(),axis.title.y=element_blank(),
             legend.position='none',plot.margin=margin(0,1.6,0,3.0,'mm'))
 # ---- verdict column: P-code + tick / cross beside the tested rows (P3 spans the two shuffle rows with a bracket)
 VL<-.9;VR<-.2;vw<-WV-VL-VR;ux<-1/vw;uy<-1/u
 it<-data.frame(item=c('F1','F2','F3','F6','F7'),a1=c('M+vRNA','M+vP','M+vRNA_shuf','M+T','M+P_graph'),a2=c('M+vRNA','M+vP','M+vP_shuf','M+T','M+P_graph'))
 it$y1<-d$y[match(it$a1,d$arm)];it$y2<-d$y[match(it$a2,d$arm)];it$yc<-(it$y1+it$y2)/2;it$id<-f6_item(it$item);it$ok<-unname(v[it$item])
 stopifnot(!any(is.na(it$y1)),!any(is.na(it$y2)))
 gx<-ux*5.7                                          # glyph centre (mm from the column's plot-left edge)
 gs<-do.call(rbind,lapply(seq_len(nrow(it)),function(i)f6_glyph_segs(gx,it$yc[i],it$ok[i],ux,uy)))
 br<-it[it$y1!=it$y2,]
 p3<-ggplot()+
  geom_segment(data=br,aes(x=0,xend=0,y=y1+.32,yend=y2-.32),colour=F6_KEY,linewidth=LW_THIN)+
  geom_text(data=it,aes(x=ux*.8,y=yc,label=id),hjust=0,vjust=.5,size=F6_GROUP_PT/.pt,fontface='bold',colour=INK)+
  geom_segment(data=gs,aes(x=x,y=y,xend=xe,yend=ye),colour=F6_KEY,linewidth=.32,lineend='round')+
  scale_x_continuous(breaks=NULL)+scale_y_continuous(breaks=NULL)+
  coord_cartesian(xlim=c(0,1),ylim=c(-NU,0),expand=FALSE,clip='off')+
  labs(x=NULL,y=NULL)+
  theme_void()+theme(plot.margin=margin(0,VR,0,VL,'mm'))
 g1<-fix_panel_margins(p1,top=TOPB,bottom=BOTB,id='B');g2<-fix_panel_margins(p2,top=TOPB,bottom=BOTB,id='B')
 g3<-fix_panel_margins(p3,left=VL,right=VR,top=TOPB,bottom=BOTB,id='B')
 g<-cowplot::plot_grid(g1,g2,g3,nrow=1,rel_widths=c(WF,WM,WV))
 list(plot=p1,grob=g)}

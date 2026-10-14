# Figure 6C (new form, fix round 2, 2026-10-10; replaces the verdict gauge, whose verdicts moved to B and G) - the
# sealed set as unit charts. Left: one square per sealed line (123), one row per tissue stratum (the 20 strata within
# which the grouped bootstrap resamples cells), rows ordered by size, squares in groups of five; filled #4D4D4D =
# sealed set (the filled-symbol rule of A); count at the end of each row; tissue names as in Figure 1. Right (in the
# space the short rows leave): the same 123 lines as a 10-wide waffle, dark = the 95 lines with a DepMap 24Q2 CRISPR
# screen (the lines that score in D, E and the CRISPR rows of G), light grey (#E5E5E5, the missing-data grey of
# Figure 1) = no screen. Sources: summary.json meta/eval_tissues (asserted to sum to meta/n_eval = 123, with 20 strata =
# analysis/drug/n_strata) and meta/gates/crispr_123 (lines_with_crispr = 95 of mapped_models = 123).
make_C<-function(){s<-read_summary_v4('Figure6')
 et<-s$meta$eval_tissues;d<-data.frame(tissue=names(et),n=as.integer(unlist(et)),stringsAsFactors=FALSE)
 stopifnot(sum(d$n)==s$meta$n_eval,s$meta$n_eval==123,nrow(d)==20,nrow(d)==s$analysis$drug$n_strata)
 nm<-c('Haematopoietic and Lymphoid'='Haem. / lymphoid','Central Nervous System'='CNS','Large Intestine'='Large intestine',
  'Head and Neck'='Head / neck','Peripheral Nervous System'='PNS','Soft Tissue'='Soft tissue')
 d$label<-ifelse(d$tissue%in%names(nm),nm[d$tissue],d$tissue)
 d<-d[order(-d$n,d$label),];d$row<-seq_len(nrow(d))
 write_csv(d,file.path(fig_out(FIG),'panel_data','C_units_display.csv'))
 TOP<-5.6;BOT<-1.2;LBL<-18.2;RGT<-4.4               # mm: letter band, bottom, label column, count column
 pw<-SLOT_W-LBL-RGT;ph<-SLOT_H-TOP-BOT;pitch<-ph/nrow(d)
 sq<-1.0;step<-1.16;g5<-.42                           # square side, square pitch, extra gap after every 5 squares (mm)
 xmm<-function(k)(k-1)*step+((k-1)%/%5)*g5          # left edge of square k (mm from the plot's left edge)
 u<-do.call(rbind,lapply(seq_len(nrow(d)),function(i)data.frame(row=d$row[i],k=seq_len(d$n[i]))))
 u$x0<-xmm(u$k);u$x1<-u$x0+sq;u$yc<--(u$row-.5)*pitch;u$y0<-u$yc-sq/2;u$y1<-u$yc+sq/2
 d$yc<--(d$row-.5)*pitch;d$xe<-xmm(d$n)+sq+.9
 stopifnot(max(d$xe)+3.5<=pw+RGT)
 cg<-s$meta$gates$crispr_123;ncr<-cg$lines_with_crispr;stopifnot(ncr==95,cg$mapped_models==s$meta$n_eval)
 wp<-1.95;ws<-1.62;wx0<-15.6;wy0<- -11.8;nc<-10    # waffle pitch, square side, left edge and top (mm)
 w<-data.frame(i=seq_len(s$meta$n_eval));w$col<-ifelse(w$i<=ncr,F6_MEAS,MISSING)
 w$x0<-wx0+((w$i-1)%%nc)*wp;w$y1<-wy0-((w$i-1)%/%nc)*wp;w$x1<-w$x0+ws;w$y0<-w$y1-ws
 wb<-min(w$y0);stopifnot(wx0+nc*wp<=pw+RGT-.2)
 wk<-data.frame(x=wx0,y=wb-c(2.3,5.0),fill=c(F6_MEAS,MISSING),lab=c(paste0('Screened (',ncr,')'),'Not screened'))
 write_csv(w,file.path(fig_out(FIG),'panel_data','C_crispr_waffle_display.csv'))
 p<-ggplot()+
  geom_rect(data=w,aes(xmin=x0,xmax=x1,ymin=y0,ymax=y1),fill=w$col,colour=NA)+
  annotate('text',x=wx0,y=wy0+2.0,label='CRISPR screen',hjust=0,vjust=.5,size=F6_GROUP_PT/.pt,fontface='bold',colour=INK)+
  geom_rect(data=wk,aes(xmin=x,xmax=x+ws,ymin=y-ws/2,ymax=y+ws/2),fill=wk$fill,colour=NA)+
  geom_text(data=wk,aes(x=x+ws+1.0,y=y,label=lab),hjust=0,vjust=.5,size=GEOM_TEXT_MIN,colour=INK)+
  geom_rect(data=u,aes(xmin=x0,xmax=x1,ymin=y0,ymax=y1),fill=F6_MEAS,colour=NA)+
  geom_text(data=d,aes(x=-1.2,y=yc,label=label),hjust=1,vjust=.5,size=GEOM_TEXT_MIN,colour=INK)+
  geom_text(data=d,aes(x=xe,y=yc,label=n),hjust=0,vjust=.5,size=GEOM_TEXT_MIN,colour=INK)+
  scale_x_continuous(breaks=NULL)+scale_y_continuous(breaks=NULL)+
  coord_cartesian(xlim=c(0,pw),ylim=c(-ph,0),expand=FALSE,clip='off')+
  labs(x=NULL,y=NULL)+
  theme_void()+theme(plot.margin=margin(0,0,0,0,'mm'))
 f6_fix(p,left=LBL,right=RGT,top=TOP,bottom=BOT)}

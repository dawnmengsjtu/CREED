# Figure S4 (v4, fix round 1) "Generator size and the pairing universe" - shared keys. Sourced once by render_v4.R.
# Panels: A/B = per-fold gain of M + virtual RNA and M + virtual P over M only against generator training cells, New
# cell lines / Held-out tissue (not the share: Figure 2H shows the share); LEG_AB = one shared key; C = gain slope (mean +- SD over folds, a_slopes);
# D = pairing-universe ladder (v3 Figure S1B). Values verbatim from plot_data; nothing recomputed.
# Alignment: fix_panel_margins() with common offsets: A, B, C share left plot edge S4_LEFT; A/B share top and bottom
# (row 1); C/D share top and bottom (row 2).
S4_SCENARIO<-c(new_cell='New cell lines',tissue='Held-out tissue')
S4_ARM<-c(RNA='m_vrna',P='m_vp')                 # layer -> canonical arm key (colour ARM_COL_CELL, label ARM_LABEL_CELL)
S4_LEFT<-12;S4_TOP1<-12.5;S4_TOP2<-6;S4_BOT<-8.5
s4_fix<-function(p,top,bottom=S4_BOT,left=NULL)list(plot=p,grob=fix_panel_margins(p,left=left,top=top,bottom=bottom,id=ID))
s4_prep<-function(d){d<-as.data.frame(d)
 if(!'layer'%in%names(d)&&'series'%in%names(d))d$layer<-ifelse(grepl('^vRNA_',d$series),'RNA','P')   # a_slopes has no layer column
 if('scenario'%in%names(d))d$scenario<-factor(S4_SCENARIO[d$scenario],levels=unname(S4_SCENARIO))
 d$arm<-factor(S4_ARM[d$layer],levels=unname(S4_ARM))
 d}
# A / B: per-fold gain trajectories over the nested generator sizes (50, 75, 100% of fit cells) + the extra-line arm.
# One panel per scenario (A = New cell lines, B = Held-out tissue, as the facets of Figure 2H); inside it one facet per
# virtual layer with free x ranges but equal mm per cell (space = 'free_x'), so the P facet (no extra arm) carries no
# blank tail and visual slopes compare across all four facets. Scenario = merged header above the two layer strips.
S4_XR<-data.frame(arm=factor(c('m_vrna','m_vrna','m_vp','m_vp'),levels=unname(S4_ARM)),x=c(150,420,150,355),gain_pct=.5)
s4_curve<-function(scn){
 lab<-S4_SCENARIO[[scn]]
 mn<-s4_prep(read_table('a_curve_mean'));fd<-s4_prep(read_table('a_curve_fold'))
 mn<-mn[mn$scenario==lab,];fd<-fd[fd$scenario==lab,];mn$x<-mn$n_gen_mean;fd$x<-fd$n_gen_fold
 stopifnot(nrow(fd)>0,max(fd$gain_pct)<1.5,min(fd$gain_pct)>0,max(mn$gain_pct)<1.5,
           all(fd$x[fd$arm=='m_vp']<355),all(fd$x>150),all(fd$x<420))
 core<-mn[mn$is_extra==0,];ext<-mn[mn$is_extra==1,];fcore<-fd[fd$is_extra==0,];fext<-fd[fd$is_extra==1,]
 f100<-fcore[fcore$size=='f100',c('arm','fold','x','gain_pct')];names(f100)[3:4]<-c('x0','y0')
 fseg<-merge(fext[,c('arm','fold','x','gain_pct')],f100,by=c('arm','fold'))
 m100<-core[core$size=='f100',c('arm','x','gain_pct')];names(m100)[2:3]<-c('x0','y0')
 mseg<-merge(ext[,c('arm','x','gain_pct')],m100,by='arm')
 p<-ggplot(core,aes(x,gain_pct,colour=arm))+
  geom_blank(data=S4_XR,inherit.aes=FALSE,aes(x,gain_pct))+
  geom_line(data=fcore,aes(group=fold),alpha=.45,linewidth=LW_THIN)+
  geom_segment(data=fseg,aes(x=x0,y=y0,xend=x,yend=gain_pct),alpha=.45,linewidth=LW_THIN,linetype='22')+
  geom_point(data=fd,alpha=.45,size=PT_SMALL,shape=16)+
  geom_line(linewidth=LW_LINE)+
  geom_segment(data=mseg,aes(x=x0,y=y0,xend=x,yend=gain_pct),linewidth=LW_LINE,linetype='22')+
  geom_point(size=PT_LARGE,shape=16)+
  geom_point(data=ext,fill='white',shape=21,size=PT_LARGE,stroke=.6)+
  facet_grid(~arm,scales='free_x',space='free_x',labeller=as_labeller(ARM_LABEL_CELL[unname(S4_ARM)]))+
  scale_colour_manual(values=ARM_COL_CELL[unname(S4_ARM)],guide='none')+
  scale_x_continuous(breaks=c(200,300,400),expand=c(0,0))+
  scale_y_continuous(breaks=seq(0,1.5,.5),labels=num_minus_acc(.1))+
  coord_cartesian(ylim=c(0,1.5),expand=FALSE)+
  labs(x='Paired generator lines',y='Gain over M only (%)')+BASE+theme(legend.position='none')
 g<-s4_add_header(p,lab)
 list(plot=p,grob=fix_panel_margins(g,left=S4_LEFT,top=S4_TOP1,bottom=S4_BOT,id=ID))}
# merged scenario header (6.5 pt bold, as a strip) spanning both layer facets, with a thin rule marking the span
s4_add_header<-function(p,label){opened<-.nullpdf();on.exit(if(opened)dev.off())
 g<-ggplotGrob(p);pl<-g$layout[grepl('^panel',g$layout$name),];st<-g$layout[grepl('^strip-t',g$layout$name),];t0<-min(st$t)
 g<-gtable::gtable_add_rows(g,unit(3.6,'mm'),pos=t0-1)
 hdr<-grobTree(textGrob(label,y=unit(1.3,'mm'),vjust=0,gp=gpar(fontfamily=CELL_FONT,fontsize=PT_STRIP,fontface='bold',col=INK)),
               segmentsGrob(x0=unit(0,'npc'),x1=unit(1,'npc'),y0=unit(.5,'mm'),y1=unit(.5,'mm'),gp=gpar(col=INK,lwd=LW_THIN*.pt,lineend='butt')))
 gtable::gtable_add_grob(g,hdr,t=t0,l=min(pl$l),r=max(pl$r),clip='off',name='strip-header')}
# LEG_AB: one key for A, B (and the colours of C). Series keys = line through point (as drawn for the five-fold means);
# fold key = thin translucent line + small point; extra-line key = dashed segment through the open symbol.
s4_key_plot<-function(){
 lv<-c('rna','p','fold','extra');lab<-c(rna=ARM_LABEL_CELL[['m_vrna']],p=ARM_LABEL_CELL[['m_vp']],fold='Each fold (F0–F4)',extra='+ extra M ∩ RNA lines')
 d<-data.frame(x=rep(1:2,4),y=1,k=factor(rep(lv,each=2),levels=lv))
 sc<-function(f,v)f(values=v,labels=lab,breaks=lv,name=NULL)
 ggplot(d,aes(x,y,group=k,colour=k,linetype=k,shape=k,alpha=k,size=k,linewidth=k))+geom_line()+geom_point(fill='white',stroke=.6)+
  sc(scale_colour_manual,c(rna=ARM_COL_CELL[['m_vrna']],p=ARM_COL_CELL[['m_vp']],fold='#7F7F7F',extra=ARM_COL_CELL[['m_vrna']]))+
  sc(scale_linetype_manual,c(rna='solid',p='solid',fold='solid',extra='22'))+
  sc(scale_shape_manual,c(rna=16,p=16,fold=16,extra=21))+
  sc(scale_alpha_manual,c(rna=1,p=1,fold=.6,extra=1))+
  sc(scale_size_manual,c(rna=PT_LARGE,p=PT_LARGE,fold=PT_SMALL,extra=PT_LARGE))+
  sc(scale_linewidth_manual,c(rna=LW_LINE,p=LW_LINE,fold=LW_THIN,extra=LW_LINE))+
  BASE+theme(legend.key.width=unit(6,'mm'),legend.key.spacing.x=unit(5,'mm'))+guides(colour=guide_legend(nrow=1))}
# D: ladder labels (metabolites abbreviated M throughout, as in the coloured rows) and fills
S4_LADDER_LABEL<-c('1'='RNA lines','2'='M source rows','3'='M lines','4'='M ∩ RNA','5'='M ∩ P',
                   '6'='M ∩ RNA ∩ P','7'='Extra M ∩ RNA','8'='Extra M ∩ RNA ∩ P')
S4_LADDER_FILL<-c('1'='#BDBDBD','2'='#BDBDBD','3'='#BDBDBD','4'=ARM_COL_CELL[['m_vrna']],'5'=ARM_COL_CELL[['m_vp']],'7'=ARM_COL_CELL[['m_vrna']])
S4_USED_FILL<-c(train=INK,validation='#969696',internal_final='#FFFFFF')   # 2026-10-11: validation mid gray (as in Figure 2H) so the key is three-way

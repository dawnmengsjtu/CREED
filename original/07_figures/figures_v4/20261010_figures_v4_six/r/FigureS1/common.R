# FigureS1 (v4, fix round 2) shared definitions: "Robustness of the virtual-layer effects".
# Page = two blocks, each closed by the one key it needs (key drawn once, directly under the panels it serves):
#   arm-coloured block : A observed vs predicted AUC (v3 Fig 3C, full width)
#                        B per-fold sRMSE, new cell lines (v3 Fig 2E upper facet)       | C per-fold sRMSE, held-out
#                        tissue (v3 Fig 2F upper facet; broken y axis, F1 in a short upper segment)
#                        LEG_ARM: arm key for A, B, C (Null = dashed line, B and C only)
#   fold-coloured block: D free tissue label (v3 Fig 4F, full width)
#                        E leave-one-fold-out (v3 Fig 4D; carries the fold key for D and E) | F interval summaries
#                        (v3 Fig 4E; same rows as E, so no y text)
# Plot tables copied verbatim into out/FigureS1/plot_data/ (nothing recomputed). Every change is display-only.
S1_ARMS<-c('M','M+vRNA','M+vP','M+RNA','M+P','null')
S1_LAB<-c(M='M only',`M+vRNA`='M + virtual RNA',`M+vP`='M + virtual P',`M+RNA`='M + measured RNA',`M+P`='M + measured P',null='Null')
S1_COL<-c(M=ARM_COL_CELL[['m_only']],`M+vRNA`=ARM_COL_CELL[['m_vrna']],`M+vP`=ARM_COL_CELL[['m_vp']],`M+RNA`=ARM_COL_CELL[['measured']],`M+P`=ARM_COL_CELL[['measured']],null=ARM_COL_CELL[['null']])
S1_SHP<-c(M=16,`M+vRNA`=16,`M+vP`=16,`M+RNA`=15,`M+P`=17,null=NA)
S1_LTY<-c(M='solid',`M+vRNA`='solid',`M+vP`='solid',`M+RNA`='solid',`M+P`='solid',null='22')
S1_SCEN<-c(new_cell='New cell lines',tissue='Held-out tissue')
s1_arm_scales<-function(arms=S1_ARMS)list(
 scale_colour_manual(values=S1_COL[arms],labels=S1_LAB[arms],limits=arms,name=NULL),
 scale_shape_manual(values=S1_SHP[arms],labels=S1_LAB[arms],limits=arms,name=NULL),
 scale_linetype_manual(values=S1_LTY[arms],labels=S1_LAB[arms],limits=arms,name=NULL))
# Marker sizes (ggplot size ~ diameter in mm): A facets are 18.6 x 17 mm with six series -> small markers.
S1_PT_A<-0.9;S1_PT_SR<-1.0;S1_PT_KEY<-1.2
# B/C: horizontal offset per arm (fold units, display only) so coincident arms stay separable; 0.1 unit ~ 1.25 mm.
S1_ARM_DODGE<-c(M=-0.2,`M+vRNA`=-0.1,`M+vP`=0,`M+RNA`=0.1,`M+P`=0.2,null=0)
S1_BREAK_GAP<-1.8    # C: gap between the two y segments (mm)
# Effects of E and F (rows, top to bottom: virtual RNA, virtual P, numeric P graph, numeric vs masked P graph)
S1_EFF<-c('protein_vs_mask','protein_graph','virtual_p','virtual_rna')     # bottom -> top
S1_EFF_LAB<-c(virtual_rna='M + virtual RNA\nvs M only',virtual_p='M + virtual P\nvs M only',protein_graph='Numeric P graph\nvs no graph',protein_vs_mask='Numeric P graph\nvs masked P graph')
S1_EFF_COL<-c(virtual_rna=ARM_COL_CELL[['m_vrna']],virtual_p=ARM_COL_CELL[['m_vp']],protein_graph=PANEL_COL[['protein']],protein_vs_mask=PANEL_COL[['protein']])
# Grid alignment (mm from the slot edge to the plot area).
S1_LEFT<-21.2        # one left plot edge for the left column (B, D, E) and for C (mirror of B)
S1_LEFT_F<-3         # F: no y text (rows are E's), only the axis ticks
S1_TOP_BC<-5.6;S1_BOT_BC<-8.4     # B, C: letter band | fold ticks + 'Held-out fold'
S1_TOP_D<-8.6;S1_BOT_D<-8.6       # D: letter band + facet strip | ticks + x title
S1_TOP_EF<-8.6;S1_BOT_EF<-13      # E, F: letter band + facet strip | ticks + x title + one-row key
S1_FACET_GAP<-3
S1_FOLD_DODGE<-0.11  # E: fold points offset vertically by fold (F0 top ... F4 bottom), +-0.22 row; x unchanged
S1_FOLD_DODGE_D<-0.075  # D: same, rows are 4 mm apart -> +-0.15 row
S1_BAR_E<-0.40;S1_BAR_D<-0.34     # half-height of the black reference bar (row units): taller than the fold stack

# ---- B and C: per-fold sRMSE, one builder (identical left structure -> strip and y title at identical positions).
s1_srmse_data<-function(scen)read_table('xl_fold_scores')%>%filter(scenario==scen,arm%in%c(S1_ARMS[1:5],'null_drug'))%>%
 mutate(arm=factor(ifelse(arm=='null_drug','null',arm),levels=S1_ARMS),x=fold+unname(S1_ARM_DODGE[as.character(arm)]),scenario_label=S1_SCEN[[scen]])%>%
 select(scenario,scenario_label,fold,arm,x,score)
# Arms: points joined by lines between adjacent folds that lie in the same y segment (column grp); the null is a short
# dashed reference bar at each fold spanning the arm cluster (x +- S1_NULL_HALF), so no line ever crosses an axis break.
S1_NULL_HALF<-0.3
s1_srmse_seg<-function(d,ylim,breaks,bottom){
 p<-ggplot(d,aes(x,score,colour=arm,linetype=arm,shape=arm))+
  geom_segment(data=~filter(.x,arm=='null'),aes(x=fold-S1_NULL_HALF,xend=fold+S1_NULL_HALF,y=score,yend=score),linewidth=LW_LINE)+
  geom_line(data=~filter(.x,arm!='null'),aes(group=grp),linewidth=LW_LINE)+geom_point(data=~filter(.x,arm!='null'),size=S1_PT_SR,stroke=.3)+
  s1_arm_scales()+scale_x_continuous(breaks=0:4,labels=paste0('F',0:4))+
  scale_y_continuous(breaks=breaks,labels=num_minus_acc(.01))+
  coord_cartesian(xlim=c(-.45,4.45),ylim=ylim,expand=FALSE)+
  labs(x=if(bottom)'Held-out fold' else NULL,y=NULL)+BASE+theme(legend.position='none',plot.margin=margin(0,1.5,0,0,'mm'))
 if(!bottom)p<-p+theme(axis.text.x=element_blank(),axis.ticks.x=element_blank(),axis.line.x=element_blank(),axis.title.x=element_blank(),axis.ticks.length.x=unit(0,'mm'))
 p}
# segs: list of list(ylim=, breaks=), top -> bottom. Segment heights proportional to their y range, so every segment
# has the same data units per mm; a pair of slanted marks on the y axis line flags each break.
s1_srmse_grob<-function(scen,segs,id){
 d<-s1_srmse_data(scen);n<-length(segs)
 d$seg<-vapply(d$score,function(v){k<-which(vapply(segs,function(z)v>=z$ylim[1]&v<=z$ylim[2],TRUE));if(length(k)!=1)stop(id,': value ',v,' outside every y segment');k},1L)
 d<-d%>%arrange(arm,fold)%>%group_by(arm)%>%mutate(grp=paste(arm,cumsum(c(1,diff(seg)!=0))))%>%ungroup()
 ps<-lapply(seq_len(n),function(i)s1_srmse_seg(d,segs[[i]]$ylim,segs[[i]]$breaks,bottom=i==n));for(p in ps)cell_guard(p,id)
 opened<-.nullpdf();on.exit(if(opened)dev.off())
 gs<-lapply(ps,ggplotGrob);w<-do.call(grid::unit.pmax,lapply(gs,function(g)g$widths));for(i in seq_len(n))gs[[i]]$widths<-w
 for(i in seq_len(n)){pr<-gs[[i]]$layout$t[grepl('^panel',gs[[i]]$layout$name)];gs[[i]]$heights[pr]<-unit(diff(segs[[i]]$ylim),'null')}
 g<-gs[[1]];brk<-integer(0)
 if(n>1)for(i in 2:n){g<-gtable::gtable_add_rows(g,unit(S1_BREAK_GAP,'mm'));brk<-c(brk,nrow(g));g<-rbind(g,gs[[i]],size='first')}
 pl<-g$layout[grepl('^panel',g$layout$name),];pc<-min(pl$l);pt<-min(pl$t);pb<-max(pl$b)
 mk<-function(yc)segmentsGrob(x0=unit(0,'npc')-unit(.8,'mm'),x1=unit(0,'npc')+unit(.8,'mm'),y0=yc-unit(.3,'mm'),y1=yc+unit(.3,'mm'),gp=gpar(col=INK,lwd=LW_AXIS*.pt,lineend='butt'))
 for(r in brk)g<-gtable::gtable_add_grob(g,gTree(children=gList(mk(unit(0,'npc')+unit(.3,'mm')),mk(unit(1,'npc')-unit(.3,'mm')))),t=r,l=pc,clip='off',name=paste0('ybreak',r))
 tg<-function(t,size,bold)textGrob(t,rot=90,gp=gpar(fontfamily=CELL_FONT,fontsize=size,fontface=if(bold)'bold'else'plain',col=INK))
 g<-gtable::gtable_add_cols(g,unit(3.8,'mm'),pos=0);g<-gtable::gtable_add_grob(g,tg('sRMSE',PT_AXIS_TITLE,FALSE),t=pt,b=pb,l=1,r=1,clip='off',name='ytitle')
 g<-gtable::gtable_add_cols(g,unit(3.2,'mm'),pos=0);g<-gtable::gtable_add_grob(g,tg(S1_SCEN[[scen]],PT_STRIP,TRUE),t=pt,b=pb,l=1,r=1,clip='off',name='scenario')
 g<-gtable::gtable_add_cols(g,unit(0,'mm'),pos=0)   # spacer: fix_panel_margins pads this column only
 g<-fix_panel_margins(g,left=S1_LEFT,top=S1_TOP_BC,bottom=S1_BOT_BC,id=id)
 pdump<-ggplot(d,aes(x,score,colour=arm))+geom_point()+BASE    # guard + panel_data dump (x = fold + display offset)
 list(plot=pdump,grob=g)}

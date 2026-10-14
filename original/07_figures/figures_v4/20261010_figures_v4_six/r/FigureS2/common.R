# FigureS2 (v4) shared helpers - adapted from v3 r/figures2_6/figure6/common_annotation.R (same group order, labels and colours).
# Group order/levels come from annotation_counts.csv; the abbreviations below only shorten the exact annotation strings for display.
suppressPackageStartupMessages(library(ggalluvial))
GROUP_LABEL<-c(Unannotated='Unannotated',`MEK1, MEK2`='MEK1/2',`PARP1, PARP2`='PARP1/2',TOP1='TOP1',`AKT1, AKT2, AKT3`='AKT1/2/3',EGFR='EGFR',HSP90='HSP90',`Other annotations`='Other')
group_levels<-function(){cnt<-read_table('annotation_counts');cnt%>%distinct(group,order)%>%arrange(order)%>%pull(group)}
glabel<-function(g){g<-as.character(g);l<-unname(GROUP_LABEL[g]);ifelse(is.na(l),g,l)}
# Annotation tones: Unannotated = mid gray, the six named groups = Okabe-Ito green, Other = light gray (v3 TONE_COL).
# Drawn opaque everywhere (A bars, C ribbons, LEG_AB key), so the key swatches are exactly the colours in A and C.
TONE_COL<-c(Unannotated='#7F7F7F',`Named groups`=OI[['green']],Other='#BDBDBD')
group_tone<-function(g){g<-as.character(g);factor(ifelse(g=='Unannotated','Unannotated',ifelse(g=='Other annotations','Other','Named groups')),levels=names(TONE_COL))}
SCEN_LABEL<-c(new_cell='New cell lines',tissue='Held-out tissue')
# The 32-protein supervised panel is labelled 'M + 32 P', the form used for the 16/64-protein panels in Figure 4 and Figure S3
# ('M + 16 P', 'M + 64 P'). It is not the house 'M + measured P' arm (full measured proteome), so that label would mislabel it.
S2_ARM_LABEL<-c(m_vrna='M + virtual RNA',m_vp='M + virtual P',m_panel32='M + 32 P',m_rna_measured='M + measured RNA')
S2_ARM_COL<-c(m_vrna=ARM_COL_CELL[['m_vrna']],m_vp=ARM_COL_CELL[['m_vp']],m_panel32=PANEL_COL[['protein']],m_rna_measured=ARM_COL_CELL[['measured']])
annotation_data<-function(n){d<-read_table(n);if('group'%in%names(d))d$group<-factor(d$group,levels=group_levels());if('scenario'%in%names(d))d$scenario<-factor(SCEN_LABEL[d$scenario],levels=SCEN_LABEL);if('arm'%in%names(d))d$arm<-factor(d$arm,levels=names(S2_ARM_LABEL));d}
# Rows: groups top-to-bottom in annotation_counts order (Unannotated first, Other last) at integer y = 8..1, so the
# row bands of A, B and D sit at identical heights inside equally tall plot areas (y range 0.5-8.5, no expansion).
group_y<-function(g){lv<-group_levels();length(lv)+1-match(as.character(g),lv)}
scale_y_groups<-function()scale_y_continuous(breaks=seq_along(group_levels()),labels=rev(glabel(group_levels())),expand=c(0,0))
coord_groups<-function(xlim=NULL)coord_cartesian(xlim=xlim,ylim=c(.5,length(group_levels())+.5),expand=FALSE,clip='off')
S2_FACET_GAP<-unit(4,'mm')
STRATUM_FILL<-'#EFEFEF';STRATUM_LINE<-'#737373'   # Figure 1F stratum style (light gray fill, mid-gray thin outline)

# FigureS5 (v4) common: cross-cohort layer transfer in three CPTAC development cohorts.
# Restyle of v1 Figure 5 (20261007_figures2_6/r/figures2_6/figure5/*.R) from its v1 plot_data (copied verbatim).
# Colour roles in this figure (Okabe-Ito, cell_style.R):
#   target layer of each transfer direction = LAYER_COL (RNA -> protein: P purple; protein -> RNA: RNA blue)
#   cohort identity (schematic tiles, gating-weight sources) = Fig 1 tissue colours where they exist
#     (BRCA = Breast green, LUAD = Lung orange) and vermilion for UCEC
#   before/after source-shift correction = hollow/filled point (shape, not colour)
CP_PRED<-c('train_mean','pooled_anchor','pooled_context','uniform_experts','gated_experts','reversed_gate')
CP_PRED_LABEL<-c(train_mean='Mean',pooled_anchor='Anchor',pooled_context='Context',uniform_experts='Uniform',gated_experts='Gated',reversed_gate='Reversed')
CP_THREE<-c('pooled_context','uniform_experts','gated_experts')
CP_DIR<-c('rna_to_protein','protein_to_rna')
CP_DIR_LABEL<-c(rna_to_protein='RNA → protein',protein_to_rna='Protein → RNA')
CP_DIR_COL<-c(rna_to_protein=LAYER_COL[['P']],protein_to_rna=LAYER_COL[['RNA']])
CP_COHORTS<-c('BRCA','LUAD','UCEC')
CP_COHORT_COL<-c(BRCA=OI[['green']],LUAD=OI[['orange']],UCEC=OI[['vermilion']])
cp_read<-function(name){d<-read_pd(FIG,name);d$direction<-factor(d$direction,levels=CP_DIR);d$cohort<-factor(d$cohort,levels=CP_COHORTS);d}
cp_dir_labeller<-labeller(direction=CP_DIR_LABEL)
# shared log sRMSE axis for B (cohort-level) and D (per-patient). One x range per cohort column, identical in B and D
# (display ranges only, chosen to contain every plotted value plus the radius of B's before-correction ring:
# BRCA/LUAD <= 2.71, UCEC <= 9.03, minimum 0.526).
# facet_grid(..., scales='free_x', space='free_x') makes column width proportional to its log range, so every column has
# the same mm per decade: 0.5, 1 and 2 sit at the same offset from each column's left edge, and the UCEC column is
# simply wider because its axis extends to 10.5 (it is not a different scale).
CP_XR<-data.frame(cohort=factor(rep(CP_COHORTS,each=2),levels=CP_COHORTS),x=c(0.45,3.3,0.45,3.3,0.45,10.5))
CP_SRMSE_BRK<-c(0.5,1,2,4,8)
cp_srmse_x<-function()list(geom_blank(data=CP_XR,aes(x=x),inherit.aes=FALSE),scale_x_log10(breaks=CP_SRMSE_BRK,labels=c('0.5','1','2','4','8'),expand=c(0,0)))
cp_facet<-function()facet_grid(direction~cohort,labeller=cp_dir_labeller,scales='free_x',space='free_x')
# distribution summary drawn on the violins of D and E: black bar = interquartile range, white notch = median.
# (Replaces the box plot, whose outlined box collapsed to a black sliver inside the narrowest violins; the notch never
# hides the bar: it is 0.17 mm wide, so the bar stays visible on both sides even for the narrowest IQR, 0.3 mm.)
CP_IQR_LW<-0.95;CP_MED_LW<-0.22;CP_MED_H<-.12
q25<-function(v)quantile(v,.25,names=FALSE);q75<-function(v)quantile(v,.75,names=FALSE)
cp_iqr_median<-function(data=NULL)list(
 stat_summary(data=data,geom='linerange',orientation='y',fun.min=q25,fun.max=q75,fun=median,colour=INK,linewidth=CP_IQR_LW),
 stat_summary(data=data,geom='errorbar',orientation='y',fun.min=median,fun.max=median,fun=median,colour='white',linewidth=CP_MED_LW,width=CP_MED_H))
# legend in the 5 mm letter band, right-justified (as Figure 1D 'Missing'): frees the bottom of the panel
CP_LEGEND_TOP<-theme(legend.position='top',legend.justification=c(1,1),legend.box.spacing=unit(.8,'mm'),plot.margin=margin(1.2,1.5,1,1.5,'mm'))
# right-hand direction strips: same text, same width in every faceted panel
CP_STRIP<-theme(strip.text.y.right=element_text(angle=-90,size=PT_STRIP,face='bold',margin=margin(0,0,0,1.2,'mm')),
                strip.text.x=element_text(size=PT_STRIP,face='bold',margin=margin(0,0,1.2,0,'mm')))

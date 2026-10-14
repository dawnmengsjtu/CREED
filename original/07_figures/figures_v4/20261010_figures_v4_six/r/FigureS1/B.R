# FigureS1 B (v4) <- v3 Figure 2E, upper facet: per-fold sRMSE of the M-deployed arms in new cell lines, with the
# per-drug training-mean null (dashed). Built by s1_srmse_grob() (common.R), shared with C, so the scenario label, the
# y title and the tick labels (all two decimals) sit at identical positions in B and C. Arms are offset horizontally
# within each fold (S1_ARM_DODGE, display only) and drawn with small markers, so coincident arms stay separable.
# Left plot edge = S1_LEFT (shared with D and E below); plot top/bottom shared with C. Arm key: LEG_ARM, directly below.
make_B<-function()s1_srmse_grob('new_cell',list(list(ylim=c(.865,1.105),breaks=c(.9,1,1.1))),'B')

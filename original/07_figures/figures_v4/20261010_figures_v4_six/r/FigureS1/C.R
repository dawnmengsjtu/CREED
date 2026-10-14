# FigureS1 C (v4) <- v3 Figure 2F, upper facet: as B for held-out tissue. Fold F1 (sRMSE about 1.94-1.99) sits far
# above the other folds (0.85-0.95), so the y axis is broken (display only): F1 in a short upper segment
# (1.935-2.005), F0 and F2-F4 in the lower segment (0.845-0.955). Both segments have the same data units per mm;
# slanted marks on the y axis line flag the break; lines crossing it are clipped at the segment edges.
make_C<-function()s1_srmse_grob('tissue',list(list(ylim=c(1.935,2.005),breaks=c(1.95,2)),list(ylim=c(.845,.955),breaks=c(.85,.9,.95))),'C')

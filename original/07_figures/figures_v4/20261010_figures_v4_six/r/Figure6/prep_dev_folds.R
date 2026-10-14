# Figure 6 (fix round 2, 2026-10-10) - extract the per-fold development (new-cell-line) values used in F and G.
# Nothing is recomputed: every value is copied verbatim from an existing table.
#   GDSC  : out/Figure2/plot_data/xl_pair_folds.csv (new_cell, deployed M, borrowed RNA / P):
#           headroom_fold (room, %), recovered_fold (recovered gain, %), share_fold (recovered share, fraction).
#   CRISPR: creed-research-runs/crispr-endpoint-20261009-0001/output/comparisons.csv (scenario new_cell):
#           per_fold_pct of 'room M->RNA', 'virtual M+vRNA vs M', 'room M->P', 'virtual M+vP vs M' (%).
#           (No per-fold CRISPR shares exist in the permitted tables, so none are drawn.)
# Check: the mean of the five fold values equals the development value already printed on the page
# (b_share.csv dev_room_pct / dev_recovered_pct, 3 decimals).
# Usage: Rscript r/Figure6/prep_dev_folds.R   ->  out/Figure6/plot_data/dev_folds.csv
suppressPackageStartupMessages({library(readr);library(dplyr)})
ROOT<-'/public/home/mengxl/workspaces/creed-research/figure_work/20261010_figures_v4_six'
PD<-file.path(ROOT,'out','Figure6','plot_data')
xl<-read_csv(file.path(ROOT,'out','Figure2','plot_data','xl_pair_folds.csv'),show_col_types=FALSE)
xl<-xl%>%filter(scenario=='new_cell',deployed=='M',borrowed%in%c('RNA','P'))%>%arrange(borrowed,fold)
stopifnot(nrow(xl)==10,all(table(xl$borrowed)==5))
g<-bind_rows(
 xl%>%transmute(endpoint='drug',layer=borrowed,quantity='room',fold,value=headroom_fold),
 xl%>%transmute(endpoint='drug',layer=borrowed,quantity='recovered',fold,value=recovered_fold),
 xl%>%transmute(endpoint='drug',layer=borrowed,quantity='share',fold,value=share_fold))%>%
 mutate(source='out/Figure2/plot_data/xl_pair_folds.csv (new_cell, M<-RNA / M<-P)')
cc<-read_csv('/public/home/mengxl/creed-research-runs/crispr-endpoint-20261009-0001/output/comparisons.csv',show_col_types=FALSE)%>%filter(scenario=='new_cell')
want<-c('room M->RNA'='RNA room','virtual M+vRNA vs M'='RNA recovered','room M->P'='P room','virtual M+vP vs M'='P recovered')
cr<-do.call(rbind,lapply(names(want),function(k){r<-cc[cc$comparison==k,];stopifnot(nrow(r)==1)
 v<-as.numeric(strsplit(gsub('\\[|\\]| ','',r$per_fold_pct),',')[[1]]);stopifnot(length(v)==5)
 lq<-strsplit(want[[k]],' ')[[1]]
 data.frame(endpoint='crispr',layer=lq[1],quantity=lq[2],fold=0:4,value=v,
            source=paste0('crispr-endpoint-20261009-0001/output/comparisons.csv (new_cell, "',k,'", per_fold_pct)'))}))
d<-bind_rows(g,cr)
bs<-read_csv(file.path(PD,'b_share.csv'),show_col_types=FALSE)
for(i in seq_len(nrow(bs))){e<-bs$endpoint[i];l<-bs$layer[i]
 m_room<-mean(d$value[d$endpoint==e&d$layer==l&d$quantity=='room']);m_rec<-mean(d$value[d$endpoint==e&d$layer==l&d$quantity=='recovered'])
 stopifnot(abs(m_room-bs$dev_room_pct[i])<5e-4,abs(m_rec-bs$dev_recovered_pct[i])<5e-4)
 cat(sprintf('%s %s: fold-mean room %.4f vs %.3f; recovered %.4f vs %.3f  OK\n',e,l,m_room,bs$dev_room_pct[i],m_rec,bs$dev_recovered_pct[i]))}
stopifnot(nrow(d)==50)
write_csv(d,file.path(PD,'dev_folds.csv'))
cat('wrote',file.path(PD,'dev_folds.csv'),nrow(d),'rows\n')

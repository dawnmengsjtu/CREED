# Figure 1D (v3): share of finite entries per outer fold in the archived RNA, protein and drug-AUC matrices
# (denominator = cell lines in fold x features). Metabolites (100 % finite) omitted. Coloured = observed, grey = missing.
make_D <- function() {
 d<-read_table('D_entry_coverage')%>%filter(asset!='Metabolites');d$asset<-factor(d$asset,c('RNA','Protein','Drug AUC'));d$fold<-factor(paste0('F',d$fold),paste0('F',0:4))
 d$key<-factor(ifelse(d$status=='Missing','Missing',as.character(d$asset)),c('Missing','RNA','Protein','Drug AUC'))
 ggplot(d,aes(fold,pct,fill=key))+geom_col(width=.72)+facet_wrap(~asset,nrow=1)+
  scale_fill_manual(values=c(Missing=MISSING,RNA=LAYER_COL[['RNA']],Protein=LAYER_COL[['P']],`Drug AUC`=TARGET),breaks='Missing')+
  scale_y_continuous(breaks=c(0,50,100),expand=expansion(mult=c(0,.02)))+
  labs(x=NULL,y='Entries (%)')+BASE+theme(legend.position='top',legend.justification='right',legend.direction='horizontal',plot.margin=margin(1.2,1.5,1,1.5,'mm'),legend.box.spacing=unit(.6,'mm'))
}

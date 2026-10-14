# S5 C (v1 5C): corrected standardized RMSE of all six predictors (rows = held-out cohort, facets = direction).
# Log10 grey scale; tile numbers are the untransformed values (2 dp).
# Predictor labels sit on the top axis (horizontal), in the band that D uses for its cohort strips, and the colour bar
# sits under the heat map, in the band that D uses for its x axis: no empty band above the tiles, no labels hanging below.
make_C<-function(){d<-cp_read('summary')%>%mutate(arm=factor(arm,levels=CP_PRED),cohort=factor(cohort,levels=rev(CP_COHORTS)))
 ggplot(d,aes(arm,cohort,fill=rmse))+geom_tile(colour='white',linewidth=.5/.pt)+
  geom_text(aes(label=sprintf('%.2f',rmse),colour=rmse>3),size=GEOM_TEXT_MIN,family=CELL_FONT)+
  scale_colour_manual(values=c('FALSE'=INK,'TRUE'='white'),guide='none')+
  facet_grid(direction~.,labeller=cp_dir_labeller)+
  scale_fill_gradient(low='#FFFFFF',high='#525252',trans='log10',limits=c(0.45,10.5),breaks=c(0.5,1,2,4,8),labels=c('0.5','1','2','4','8'),name='Standardized RMSE',
                      guide=guide_colourbar(barwidth=unit(26,'mm'),barheight=unit(2,'mm'),ticks.colour=INK,ticks.linewidth=LW_TICK,frame.colour=INK,frame.linewidth=LW_TICK,direction='horizontal'))+
  scale_x_discrete(labels=CP_PRED_LABEL,expand=c(0,0),position='top')+scale_y_discrete(expand=c(0,0))+
  labs(x=NULL,y='Held-out cohort')+BASE+CP_STRIP+legend_title_on()+
  theme(legend.position='bottom',legend.direction='horizontal',legend.justification=c(.5,1),legend.box.spacing=unit(1.6,'mm'),legend.margin=margin(0,0,0,0),
        legend.title.position='left',legend.title=element_text(size=PT_LEGEND,vjust=1,margin=margin(r=2,t=0,unit='mm')),
        axis.line=element_blank(),axis.ticks=element_blank(),
        axis.text.x.top=element_text(size=PT_TICK,angle=0,hjust=.5,vjust=0,margin=margin(b=1.2,unit='mm')))}

# FigureS2 LEG_AB (unlettered, 172 x 5 mm, between row A/B and row C/D): the shared keys drawn once.
#   left half  - annotation tone key (opaque TONE_COL), serves the bars in A and the ribbons in C; centred under A's plot area
#   right half - Delta sMSE colour bar of B (horizontal); centred under B's plot area
# Plot-area centres are measured from the A and B grobs (fixed widths left/right of the panel columns), not hard-coded.
for(.f in c('A.R','B.R'))if(!exists(paste0('make_',sub('\\.R$','',.f)),mode='function'))source(file.path(fig_rdir(FIG),.f),local=FALSE)
make_LEG_AB<-function(){opened<-dev.cur()==1;if(opened)pdf(NULL);on.exit(if(opened)dev.off())
 mid<-function(p,slot_w){g<-ggplotGrob(p);pl<-g$layout[grepl('^panel',g$layout$name),]
  lw<-sum(convertWidth(g$widths[seq_len(min(pl$l)-1)],'mm',valueOnly=TRUE));rw<-sum(convertWidth(g$widths[(max(pl$r)+1):length(g$widths)],'mm',valueOnly=TRUE))
  (lw+slot_w-rw)/2}
 cA<-mid(make_A(),84);cB<-88+mid(make_B(),84)
 la<-legend_only_v4(make_A(TRUE));lb<-legend_only_v4(make_B(TRUE))
 # both keys are top-justified in the strip, so the tone swatches and the colour bar sit on one line (bar labels below it)
 top<-function(g,x)grobTree(g,vp=viewport(x=unit(x,'mm'),y=unit(1,'npc')-unit(.3,'mm'),width=unit(84,'mm'),height=grobHeight(g),just=c('centre','top')))
 gTree(children=gList(top(la,cA),top(lb,cB)),name='LEG_AB')}

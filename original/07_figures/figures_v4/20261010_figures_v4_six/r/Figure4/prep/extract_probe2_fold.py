# Figure 4 fix round (2026-10-10): per-fold N3 values for panels G and H, re-arranged (no new statistics) from the
# finished probe-2 run summary (READ only):
#   /public/home/mengxl/creed-research-runs/probe2-context-abstain-20261009-0001/output/summary.json  B.<scenario>.table[<arm>|<proxy>]
#     aurc_folds              -> h_aurc_fold.csv  (per-fold AURC; its mean is e_aurc.aurc_mean, checked)
#     coverage_folds[1.0/0.5] -> g_cov50_fold.csv (per-fold retained sRMSE at full and 50% coverage; rel_pct = 100*(c50/c100-1),
#                                the per-fold quantity whose five-fold mean is e_riskcov_mean.rel_pct_mean at 0.5, checked)
#   B/aurc.csv per-fold rows are used as a cross-check for the single-proxy rows (random = mean of five seeds per fold).
# u1 = the arm-matched sub-generator disagreement (u1_vRNA for M+vRNA, u1_vP for M+vP), as in the v3 e_aurc table.
import json, csv, sys
import numpy as np
P="/public/home/mengxl/creed-research-runs/probe2-context-abstain-20261009-0001/output"
PD=sys.argv[1]
S=json.load(open(P+"/summary.json"))["B"]
ea={(r["scenario"],r["arm"],r["proxy"]):float(r["aurc_mean"]) for r in csv.DictReader(open(PD+"/e_aurc.csv"))}
er={(r["scenario"],r["arm"],r["proxy"]):float(r["rel_pct_mean"]) for r in csv.DictReader(open(PD+"/e_riskcov_mean.csv")) if r["coverage"] in ("0.5","0.50")}
raw={}
for r in csv.DictReader(open(P+"/B/aurc.csv")): raw[(r["scenario"],r["arm"],r["proxy"],int(r["fold"]))]=float(r["aurc"])
A=[];C=[]
for sc in ("new_cell","tissue"):
  for arm in ("M+vRNA","M+vP"):
    u1="u1_vRNA" if arm=="M+vRNA" else "u1_vP"
    for px,key in (("u1",u1),("u3","u3"),("random","random"),("oracle","oracle")):
      t=S[sc]["table"][arm+"|"+key]; f=t["aurc_folds"]
      assert abs(np.mean(f)-ea[(sc,arm,px)])<1e-12 and abs(t["aurc"]-ea[(sc,arm,px)])<1e-12,(sc,arm,px)
      for i,v in enumerate(f):
        if px=="random":
          rv=np.mean([raw[(sc,arm,"random_s%d"%s,i)] for s in range(5)])
        else: rv=raw[(sc,arm,key,i)]
        assert abs(rv-v)<1e-12,(sc,arm,px,i,rv,v)
        A.append(dict(scenario=sc,arm=arm,proxy=px,fold=i,aurc=v,aurc_mean=t["aurc"]))
      c1=t["coverage_folds"]["1.0"]; c5=t["coverage_folds"]["0.5"]
      rel=[100*(b/a-1) for a,b in zip(c1,c5)]
      assert abs(np.mean(rel)-er[(sc,arm,px)])<1e-9,(sc,arm,px,np.mean(rel),er[(sc,arm,px)])
      for i in range(5): C.append(dict(scenario=sc,arm=arm,proxy=px,fold=i,srmse_full=c1[i],srmse_cov50=c5[i],rel_pct=rel[i],rel_pct_mean=er[(sc,arm,px)]))
for name,rows in (("h_aurc_fold",A),("g_cov50_fold",C)):
  with open(PD+"/"+name+".csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
  print(name,len(rows),"rows; all checks passed")

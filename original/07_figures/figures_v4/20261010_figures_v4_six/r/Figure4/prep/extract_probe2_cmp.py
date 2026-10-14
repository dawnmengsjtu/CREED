# Figure 4 fix round 2 (2026-10-10): the pre-registered N3 test drawn directly in panel G. Re-arranged (no new statistics)
# from the finished probe-2 run summary (READ only):
#   /public/home/mengxl/creed-research-runs/probe2-context-abstain-20261009-0001/output/summary.json
#     B.<scenario>.comparisons["<arm>|<proxy>_vs_random"] -> g_n3_vs_random.csv
#   improvement_pct = AURC lowering vs Random (%), folds_won, nb_p, per_fold_pct (F0-F4; won = per_fold_pct > 0).
# u1 = the arm-matched sub-generator disagreement (u1_vRNA for M+vRNA, u1_vP for M+vP), as in e_aurc / the verdicts.
# Checks: improvement_pct and folds_won of u1 equal the pre-registered verdict; new_mean / base_mean equal e_aurc
# aurc_mean of the proxy / Random; folds_won equals the count of positive per_fold_pct; per_fold_pct equals
# 100*(1 - AURC[proxy] / AURC[Random]) of the same fold in h_aurc_fold (to the stored 4 decimals).
import json, csv, sys
P="/public/home/mengxl/creed-research-runs/probe2-context-abstain-20261009-0001/output"
PD=sys.argv[1]
S=json.load(open(P+"/summary.json"))["B"]
ea={(r["scenario"],r["arm"],r["proxy"]):float(r["aurc_mean"]) for r in csv.DictReader(open(PD+"/e_aurc.csv"))}
hf={(r["scenario"],r["arm"],r["proxy"],int(r["fold"])):float(r["aurc"]) for r in csv.DictReader(open(PD+"/h_aurc_fold.csv"))}
rows=[]
for sc in ("new_cell","tissue"):
  for arm in ("M+vRNA","M+vP"):
    u1="u1_vRNA" if arm=="M+vRNA" else "u1_vP"
    for px,key in (("u1",u1),("u3","u3"),("oracle","oracle")):
      c=S[sc]["comparisons"][arm+"|"+key+"_vs_random"]
      if px=="u1":
        v=S[sc]["verdicts"][arm]; assert v["primary_proxy"]==key
        assert abs(v["aurc_improvement_pct"]-c["improvement_pct"])<1e-12 and v["folds_won"]==c["folds_won"],(sc,arm)
      assert abs(c["new_mean"]-ea[(sc,arm,px)])<1e-12 and abs(c["base_mean"]-ea[(sc,arm,"random")])<1e-12,(sc,arm,px)
      pf=c["per_fold_pct"]; assert c["folds_won"]==sum(x>0 for x in pf),(sc,arm,px)
      for i,x in enumerate(pf):
        chk=100*(1-hf[(sc,arm,px,i)]/hf[(sc,arm,"random",i)]); assert abs(chk-x)<6e-5,(sc,arm,px,i,chk,x)
        rows.append(dict(scenario=sc,arm=arm,proxy=px,fold=i,per_fold_pct=x,won=int(x>0),improvement_pct=c["improvement_pct"],folds_won=c["folds_won"],nb_p=c["nb_p"]))
with open(PD+"/g_n3_vs_random.csv","w",newline="") as fh:
  w=csv.DictWriter(fh,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
print("g_n3_vs_random",len(rows),"rows; all checks passed")

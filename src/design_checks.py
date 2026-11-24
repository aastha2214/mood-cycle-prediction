import pyreadstat, pandas as pd, numpy as np
from scipy import stats
q,_=pyreadstat.read_sav("data/RV20_Questionnaires_MC_20230519.sav")
i=q[q.include==1].copy()
i["long"]=i.groupby("VPN").Session.transform("nunique")>1
for nm,g in [("all included",i),("longitudinal",i[i.long]),("cross-only",i[~i.long])]:
    print(f"\n[{nm}] rows={len(g)} women={g.VPN.nunique()}")
    for h in ["estradiol","progesterone"]:
        for t,f in [("raw",lambda x:x),("log1p",np.log1p)]:
            x=f(g[h]); print(f"  {h:12s} {t:5s} skew={stats.skew(x):5.2f} kurt={stats.kurtosis(x):5.2f} max_z={((x-x.mean())/x.std()).max():4.1f} zeros={int((g[h]==0).sum())}")
    print("  corr E2-P raw spearman=%.2f pearson=%.2f | log1p pearson=%.2f"%(stats.spearmanr(g.estradiol,g.progesterone)[0],g.estradiol.corr(g.progesterone),np.log1p(g.estradiol).corr(np.log1p(g.progesterone))))
print("\nP log1p vs raw: raw P>300 rows:",(i.progesterone>300).sum(),"; log1p(300)=%.2f log1p(median 45)=%.2f"%(np.log1p(300),np.log1p(45)))
print("Pearson corr of P with phase(luteal dummy): raw %.2f log1p %.2f"%(i.progesterone.corr((i.Phase==2).astype(float)),np.log1p(i.progesterone).corr((i.Phase==2).astype(float))))
print("\nMissing among included (n=%d): "%len(i),i[["estradiol","progesterone","PMS","BDI_RW","BAI_RW","Affect_pos","DRSP_psychisch","STAI","Phase"]].isna().sum().to_dict())
L=i[i.long]; b=L[L.Phase==0].VPN.unique()
print("Long women with a menses row:",len(b),"/",L.VPN.nunique(),"; non-menses rows with baseline:",((L.Phase!=0)&L.VPN.isin(b)).sum(),"/",(L.Phase!=0).sum())
print("Target SD (included): ",i[["Affect_pos","DRSP_psychisch","STAI"]].std().round(3).to_dict())
# between- vs within-person share for longitudinal
for t in ["Affect_pos","DRSP_psychisch","STAI"]:
    gm=L.groupby("VPN")[t].transform("mean"); print(f"  {t}: within-woman SD={np.sqrt(((L[t]-gm)**2).mean()):.3f} vs total SD={L[t].std():.3f} (long, n={len(L)})")

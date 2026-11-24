import pyreadstat, pandas as pd, numpy as np
q,_=pyreadstat.read_sav("data/RV20_Questionnaires_MC_20230519.sav")
q["nsess"]=q.groupby("VPN").Session.transform("nunique")
i=q[q.include==1]
print("include==1: women",i.VPN.nunique(),"| women with 3 sess",i[i.nsess==3].VPN.nunique(),"| rows in 3-sess women",(i.nsess==3).sum(),"| cross (nsess==1) women",i[i.nsess==1].VPN.nunique())
print("non-included women with nsess3:",q[(q.include==0)&(q.nsess==3)].VPN.nunique())
print("Session1 rows incl:",(i.Session==1).sum())
lw=i[i.nsess>1]; print("long rows phases",lw.Phase.value_counts().sort_index().to_dict(), "hormone-free HC? HC1_type nonblank among included:",(i.HC1_type.str.strip()!="").sum())
print("PSST items per row non-null:",q[[f"PSST{k}" for k in range(1,15)]].notna().all(axis=1).sum(), q.PSST1.dropna().unique())
print("PMS vs sum PSST1-14 corr:",np.corrcoef(*q[["PMS"]].join(q[[f"PSST{k}" for k in range(1,15)]].sum(axis=1).rename("s")).dropna().T.values)[0,1].round(3))
print(i[["estradiol","progesterone"]].describe().round(2).T)

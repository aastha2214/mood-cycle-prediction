"""Does within-woman estradiol still relate to DRSP once phase is in the model? (E2 and phase are strongly collinear within a woman.)"""
import sys, warnings; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from src.data import *
warnings.filterwarnings("ignore"); df = load_analysis(); rows = []
for which in ["long", "pooled"]:
    d = subset(df, which).copy()
    for c in ["DRSP_psychisch", "DRSP", "estradiol"]: d["z_" + c] = (d[c] - d[c].mean()) / d[c].std(ddof=1)
    m = d.groupby("VPN").z_estradiol.transform("mean"); d["zb"] = m; d["zw"] = d.z_estradiol - m
    print(f"[{which}] n={len(d)} rows, {d.VPN.nunique()} women; corr(within E2, luteal/periov dummies) = {d.zw.corr(d.ph2):+.2f} / {d.zw.corr(d.ph1):+.2f}")
    for y in ["DRSP_psychisch", "DRSP"]:
        for nm, rhs in [("E2 only", "zw + zb"), ("E2 + phase", "zw + zb + ph1 + ph2")]:
            r = smf.mixedlm(f"z_{y} ~ {rhs}", d, groups=d.VPN).fit(reml=True); b, se = r.fe_params["zw"], r.bse_fe["zw"]
            rows.append(dict(sample=which, n_rows=len(d), n_women=d.VPN.nunique(), outcome=y, model=nm, b_within_E2=b, lo=b - 1.96 * se, hi=b + 1.96 * se, p=r.pvalues["zw"]))
            print(f"   {y:15s} {nm:11s} within-E2 b = {b:+.2f} [{b-1.96*se:+.2f}, {b+1.96*se:+.2f}] p={r.pvalues['zw']:.3f}")
pd.DataFrame(rows).to_csv("results/phase5_e2_given_phase.csv", index=False)

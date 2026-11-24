"""Phase 4: LOPO nested CV for baseline/ridge/RF/GB on feature sets A,B,C,D,D2 x 3 targets x {pooled, longitudinal}; plus cross->longitudinal transfer."""
import sys, time; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from src.data import *; from src.cvlib import *
df = load_analysis(); rows = []; t0 = time.time()
assert set(df[df["sample"] == "cross"].VPN).isdisjoint(df[df["sample"] == "long"].VPN)       # disjoint samples
def add(sample, tn, model, fs, d, pred, hp):
    rows.append(pd.DataFrame(dict(sample=sample, target=tn, model=model, fset=fs, VPN=d.VPN.values, Session=d.Session.values, y=d[TARGETS[tn]].values, pred=pred, hp=hp)))
for which in ["pooled", "long"]:
    d = subset(df, which)
    for tn, tc in TARGETS.items():
        p, hp = lopo(d, tc, FEATURE_SETS["A"], "baseline"); add(which, tn, "baseline", "-", d, p, hp)
        for fs in ["A", "B", "C", "D", "D2"]:
            for m in ["ridge", "rf", "gb"]:
                p, hp = lopo(d, tc, FEATURE_SETS[fs], m); add(which, tn, m, fs, d, p, hp)
        print(f"{which:6s} {tn:10s} done ({len(d)} rows, {d.VPN.nunique()} women) [{time.time()-t0:.0f}s]", flush=True)
# cross-sectional (train) -> longitudinal (test): disjoint women
cr, lg = df[df["sample"] == "cross"], df[df["sample"] == "long"]
for tn, tc in TARGETS.items():
    p, hp = predict_one("baseline", cr[FEATURE_SETS["A"]].values, cr[tc].values, cr.VPN.values, lg[FEATURE_SETS["A"]].values, SEED); add("cross2long", tn, "baseline", "-", lg, p, hp)
    for fs in ["A", "B", "C", "D", "D2"]:
        for m in ["ridge", "rf", "gb"]:
            p, hp = predict_one(m, cr[FEATURE_SETS[fs]].values, cr[tc].values, cr.VPN.values, lg[FEATURE_SETS[fs]].values, SEED); add("cross2long", tn, m, fs, lg, p, hp)
print(f"cross2long done: train {len(cr)} rows/{cr.VPN.nunique()} women -> test {len(lg)} rows/{lg.VPN.nunique()} women [{time.time()-t0:.0f}s]")
pd.concat(rows).to_csv("results/phase4_oof.csv", index=False); print("saved results/phase4_oof.csv")

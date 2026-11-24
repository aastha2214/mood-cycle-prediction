"""Analysis dataset: Questionnaires file, include==1, complete rows (see docs/design.md)."""
from pathlib import Path
import numpy as np, pandas as pd, pyreadstat
ROOT = Path(__file__).resolve().parent.parent
SEED = 42
TARGETS = {"pos_affect": "Affect_pos", "drsp_psych": "DRSP_psychisch", "stai": "STAI"}
FEATURE_SETS = {  # A phase only, B hormones only, C = A+B, D = C + traits
    "A": ["ph1", "ph2"], "B": ["logE2", "logP"],
    "C": ["ph1", "ph2", "logE2", "logP"],
    "D": ["ph1", "ph2", "logE2", "logP", "PMS", "BDI_RW", "BAI_RW"],
    "D2": ["ph1", "ph2", "logE2", "logP", "BDI_RW", "BAI_RW"]}   # D without PSST (tests PSST-DRSP overlap)
RAW_D = ["ph1", "ph2", "estradiol", "progesterone", "PMS", "BDI_RW", "BAI_RW"]  # un-transformed hormones

def load_analysis():
    q, _ = pyreadstat.read_sav(str(ROOT / "data/RV20_Questionnaires_MC_20230519.sav"))
    q = q[q.include == 1].copy()
    q["sample"] = np.where(q.groupby("VPN").VPN.transform("size") > 1, "long", "cross")
    q = q.dropna(subset=["PMS"]).copy()          # 1 row with missing PSST -> 271 common rows
    q["ph1"] = (q.Phase == 1).astype(float); q["ph2"] = (q.Phase == 2).astype(float)
    q["logE2"] = np.log1p(q.estradiol); q["logP"] = np.log1p(q.progesterone)
    return q.reset_index(drop=True)

def subset(df, which):  # which: "pooled" | "long"
    return df if which == "pooled" else df[df["sample"] == "long"].reset_index(drop=True)

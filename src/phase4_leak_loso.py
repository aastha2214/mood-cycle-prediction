"""Phase 4 extras: (1) shuffled-target leakage test; (2) leave-one-SESSION-out experiment on the longitudinal sample (how much personal history helps)."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from src.data import *; from src.cvlib import *
df = load_analysis(); rng = np.random.default_rng(SEED); B = 2000
# ---- (1) leakage test: permute the target across rows; a leak-free pipeline must NOT beat the baseline
print("Shuffled-target test (pooled, ridge, set D, 3 permutations; expect skill <= ~0). n=%d rows, %d women" % (len(df), df.VPN.nunique()))
leak = []
for tn, tc in TARGETS.items():
    for k in range(3):
        d = df.copy(); d[tc] = np.random.default_rng(SEED + k).permutation(d[tc].values)
        p, _ = lopo(d, tc, FEATURE_SETS["D"], "ridge"); pb, _ = lopo(d, tc, FEATURE_SETS["D"], "baseline")
        leak.append(dict(target=tn, perm=k, R2_oos=1 - ((p - d[tc]) ** 2).sum() / ((pb - d[tc]) ** 2).sum()))
leak = pd.DataFrame(leak); leak.to_csv("results/phase4_shuffle_test.csv", index=False); print(leak.groupby("target").R2_oos.agg(["mean", "min", "max"]).round(4).to_string())
# ---- (2) leave-one-session-out, longitudinal sample only
L = subset(df, "long"); g = L.VPN.values; N = len(L)
def pm_feature(y, g, h):          # personal mean of the woman's other sessions, EXCLUDING the row itself and the held-out row h
    pm = np.full(len(y), np.nan)
    for i in range(len(y)):
        m = (g == g[i]) & (np.arange(len(y)) != i) & (np.arange(len(y)) != h)
        if m.any(): pm[i] = y[m].mean()
    return pm
def loso_one(h, tc, y, X, g):
    tr = np.arange(len(y)) != h; out = {}
    out["global mean"] = y[tr].mean(); out["personal mean (other sessions)"] = y[(g == g[h]) & tr].mean()
    pm = pm_feature(y, g, h); pm = np.where(np.isnan(pm), y[tr].mean(), pm)
    for fs in ["C", "D"]:
        cols = FEATURE_SETS[fs]; Xf = L[cols].values
        out[f"ridge {fs} (no history)"] = predict_one("ridge", Xf[tr], y[tr], g[tr], Xf[[h]], SEED + h)[0][0]
        Xh = np.column_stack([Xf, pm]); out[f"ridge {fs} + personal mean"] = predict_one("ridge", Xh[tr], y[tr], g[tr], Xh[[h]], SEED + h)[0][0]
    return out
rows = []
for tn, tc in TARGETS.items():
    y = L[tc].values; X = None
    res = Parallel(n_jobs=10)(delayed(loso_one)(h, tc, y, X, g) for h in range(N)); P = pd.DataFrame(res)
    women = np.unique(g); W = np.random.default_rng(SEED).multinomial(len(women), np.ones(len(women)) / len(women), size=B).astype(float)
    idx = {w: np.where(g == w)[0] for w in women}; n_w = np.array([len(idx[w]) for w in women])
    def agg(err): return np.array([err[idx[w]].sum() for w in women])
    base_sse = agg((P["global mean"].values - y) ** 2)
    for mname in P.columns:
        e = P[mname].values - y; sae, sse = agg(np.abs(e)), agg(e ** 2)
        f = lambda wt: dict(MAE=wt @ sae / (wt @ n_w), RMSE=np.sqrt(wt @ sse / (wt @ n_w)), R2_oos=1 - wt @ sse / (wt @ base_sse))
        pt = f(np.ones(len(women))); bs = pd.DataFrame([f(w) for w in W])
        rows.append(dict(target=tn, model=mname, n_rows=N, n_women=len(women), **{k: pt[k] for k in pt}, **{k + "_lo": np.percentile(bs[k], 2.5) for k in pt}, **{k + "_hi": np.percentile(bs[k], 97.5) for k in pt}))
R = pd.DataFrame(rows); R.to_csv("results/phase4_loso.csv", index=False)
print("\nLeave-one-SESSION-out, longitudinal only (n=%d rows, %d women): RMSE [95%% CI] and R2 vs global mean" % (N, len(women)))
for t, g_ in R.groupby("target"):
    print(f"  {t}"); [print(f"    {r.model:34s} RMSE {r.RMSE:.3f} [{r.RMSE_lo:.3f},{r.RMSE_hi:.3f}]  R2_oos {r.R2_oos:+.3f} [{r.R2_oos_lo:+.2f},{r.R2_oos_hi:+.2f}]") for r in g_.itertuples()]

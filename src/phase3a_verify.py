"""3a: ridge via normal equations and SVD vs scikit-learn; also 3e numerical MAP check."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from src.data import *; from src.mathcore.ridge import *
df = load_analysis(); lams = [0, 1e-4, 1e-2, 1e-1, 1, 10]; rows = []
for which in ["pooled", "long"]:
    d = subset(df, which)
    for tn, tc in TARGETS.items():
        for fs, cols in [("D", FEATURE_SETS["D"]), ("D-raw", RAW_D)]:
            X = d[cols].values; y = d[tc].values
            mu, sd = standardize_fit(X); Z = (X - mu) / sd; Zc, yc, zm, ym = center(Z, y)
            for lam in lams:
                wn = ridge_normal(Zc, yc, lam) if lam > 0 else None; ws = ridge_svd(Zc, yc, lam)
                sk = (Ridge(alpha=len(y) * lam).fit(Z, y) if lam > 0 else LinearRegression().fit(Z, y))
                dn = np.abs(wn - sk.coef_).max() if wn is not None else np.nan
                ds = np.abs(ws - sk.coef_).max(); di = abs(ym - zm @ ws - sk.intercept_)
                rows.append(dict(sample=which, n=len(y), target=tn, feats=fs, lam=lam, d_normal=dn, d_svd=ds, d_intercept=di))
r = pd.DataFrame(rows); r.to_csv("results/phase3a_sklearn_check.csv", index=False)
print("3a ridge vs sklearn: configs =", len(r), "| max |w_normal - sklearn| = %.2e | max |w_svd - sklearn| = %.2e | max intercept diff = %.2e" %
      (r.d_normal.max(), r.d_svd.max(), r.d_intercept.max()))
assert r.d_normal.max() < 1e-8 and r.d_svd.max() < 1e-8 and r.d_intercept.max() < 1e-8
# 3e: gradient of the negative log posterior vanishes at the ridge solution when n*lam = sigma^2/tau^2
d = subset(df, "pooled"); X = d[FEATURE_SETS["D"]].values; y = d["DRSP_psychisch"].values; n = len(y)
mu, sd = standardize_fit(X); Zc, yc, _, _ = center((X - mu) / sd, y); lam = 0.1; w = ridge_svd(Zc, yc, lam)
tau2 = 2.0; sig2 = n * lam * tau2                      # implied noise variance for a chosen prior variance
g_post = -(Zc.T @ (yc - Zc @ w)) / sig2 + w / tau2     # grad of -log p(w|y) = RSS/(2 sig2) + |w|^2/(2 tau2)
print("3e MAP check (pooled, n=%d): max|grad of neg-log-posterior at ridge w| = %.2e (lam=%.2f <-> sigma^2/tau^2 = n*lam = %.2f)" % (n, np.abs(g_post).max(), lam, n * lam))
assert np.abs(g_post).max() < 1e-10
pd.DataFrame([dict(sample="pooled", n_rows=n, n_women=d.VPN.nunique(), target="drsp_psych", lam=lam, n_times_lam=n * lam, tau2=tau2, sigma2=sig2, max_abs_grad_neg_log_posterior=float(np.abs(g_post).max()))]).to_csv("results/phase3e_map_check.csv", index=False)

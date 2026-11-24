"""3b addendum: tuned heavy-ball momentum vs GD on the badly conditioned (unscaled) problems."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from src.data import *; from src.mathcore.ridge import *; from src.mathcore.optim import *; from src.plotstyle import *
df = load_analysis(); LAM, TOL, CAP = 0.01, 1e-6, 60000; rows = []; curves = {}
def iters_to_tol(L, Ls):
    sub = (L - Ls) / (L[0] - Ls); return int(np.argmax(sub < TOL)) if (sub < TOL).any() else None
for which in ["pooled", "long"]:
    d = subset(df, which); y = d["DRSP_psychisch"].values; n = len(y); nw = d.VPN.nunique(); print(f"\n[{which}] n={n} rows, {nw} women")
    for vn, cols in [("raw (raw hormones, unscaled)", RAW_D), ("log1p hormones, unscaled", FEATURE_SETS["D"])]:
        Xc, yc, _, _ = center(d[cols].values, y); ev = np.linalg.eigvalsh(hessian(Xc, LAM)); L_, mu = ev[-1], ev[0]; kap = L_ / mu
        beta = ((np.sqrt(kap) - 1) / (np.sqrt(kap) + 1)) ** 2; lr = 4 / (np.sqrt(L_) + np.sqrt(mu)) ** 2
        Ls = loss(ridge_svd(Xc, yc, LAM), Xc, yc, LAM); res = {}
        _, Lg = gd(Xc, yc, LAM, 1 / L_, CAP); res["GD (lr=1/L)"] = Lg
        _, Lh = momentum(Xc, yc, LAM, lr, beta, CAP); res["Heavy ball (tuned)"] = Lh
        out = {k: iters_to_tol(v, Ls) for k, v in res.items()}; pred_hb = np.log(TOL) / np.log(beta)
        print(f"  {vn}: kappa={kap:,.0f}, beta={beta:.5f}, lr={lr:.4g} | iterations to 1e-6: GD {out['GD (lr=1/L)'] or '>'+str(CAP)}, tuned heavy-ball {out['Heavy ball (tuned)']} (theory ~ {pred_hb:.0f})")
        rows.append(dict(sample=which, n=n, problem=vn, kappa=kap, beta=beta, lr=lr, gd_iters=out["GD (lr=1/L)"], hb_iters=out["Heavy ball (tuned)"], hb_theory=pred_hb))
        curves[(which, vn)] = (res, Ls)
pd.DataFrame(rows).to_csv("results/phase3b_momentum_tuned.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for a, vn in zip(ax, ["raw (raw hormones, unscaled)", "log1p hormones, unscaled"]):
    res, Ls = curves[("pooled", vn)]
    for k, c, ls in [("GD (lr=1/L)", C["GD"], "-"), ("Heavy ball (tuned)", C["Momentum"], "--")]:
        a.semilogy(np.maximum(res[k] - Ls, 1e-16)[:6000], color=c, ls=ls, label=k)
    a.set_title(f"{vn}\npooled n=271 rows, 160 women"); a.set_xlabel("iterations (first 6000 shown)")
ax[0].set_ylabel("L(w_t) - L(w*)"); ax[0].legend(frameon=False); fig.savefig("results/fig_3b_momentum_tuned.png"); plt.close(fig)

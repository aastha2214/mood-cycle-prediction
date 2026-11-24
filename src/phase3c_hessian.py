"""3c: Hessian eigenvalues, condition number raw vs standardised, link to GD convergence (DRSP psychological, set D)."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from src.data import *; from src.mathcore.ridge import *; from src.mathcore.optim import *; from src.plotstyle import *
df = load_analysis(); TOL = 1e-6; CAP = 60000; rows = []; store = {}
variants = {"raw (unscaled, raw hormones)": (RAW_D, False), "log1p hormones, unscaled": (FEATURE_SETS["D"], False), "log1p + standardised": (FEATURE_SETS["D"], True)}
for which in ["pooled", "long"]:
    d = subset(df, which); y = d["DRSP_psychisch"].values; n = len(y); nw = d.VPN.nunique()
    print(f"\n[{which}] n={n} rows, {nw} women")
    for vn, (cols, std) in variants.items():
        X = d[cols].values; X = (X - X.mean(0)) / X.std(0) if std else X; Xc, yc, _, _ = center(X, y)
        ev0 = np.linalg.eigvalsh(hessian(Xc, 0.0)); store[(which, vn)] = ev0
        for lam in [0.0, 0.01, 0.1, 1.0]:
            ev = ev0 + 2 * lam; rows.append(dict(sample=which, n=n, variant=vn, lam=lam, lam_min=ev[0], lam_max=ev[-1], kappa=ev[-1] / ev[0]))
        # GD with lr = 1/lmax at lam=0.01
        lam = 0.01; Hs = hessian(Xc, lam); ev = np.linalg.eigvalsh(Hs); lr = 1 / ev[-1]; ws = ridge_svd(Xc, yc, lam); Ls = loss(ws, Xc, yc, lam)
        w = np.zeros(Xc.shape[1]); L0 = loss(w, Xc, yc, lam) - Ls; hit = None
        for t in range(1, CAP + 1):
            w = w - lr * grad(w, Xc, yc, lam)
            if (loss(w, Xc, yc, lam) - Ls) / L0 < TOL: hit = t; break
        kap = ev[-1] / ev[0]; pred = np.log(TOL) / (2 * np.log(1 - 1 / kap))
        rows[-3].update(gd_iters=hit, gd_pred=pred)  # attach to the lam=0.01 row
        print(f"  {vn:30s} kappa(lam=0)={ev0[-1]/ev0[0]:9.1f} | lam=0.01: eig {ev[0]:.4g}..{ev[-1]:.4g}, kappa={kap:8.1f} | GD(lr=1/lmax) iters to 1e-6: observed {hit if hit else '>'+str(CAP)}, predicted <= {pred:.0f}")
res = pd.DataFrame(rows); res.to_csv("results/phase3c_condition.csv", index=False)
print("\nKappa vs lambda (pooled):"); print(res[res["sample"] == "pooled"].pivot(index="variant", columns="lam", values="kappa").round(1).to_string())
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
cols = {"raw (unscaled, raw hormones)": C["SGD"], "log1p hormones, unscaled": C["Momentum"], "log1p + standardised": C["GD"]}
for vn, c in cols.items():
    ev = store[("pooled", vn)]; ax[0].semilogy(range(1, len(ev) + 1), ev[::-1], marker="o", color=c, label=vn)
ax[0].set_xlabel("eigenvalue rank"); ax[0].set_ylabel("eigenvalue of Hessian (lam=0)"); ax[0].set_title("Hessian spectrum, pooled n=271"); ax[0].legend(frameon=False, fontsize=8)
lams = np.logspace(-4, 1, 40)
for vn, c in cols.items():
    ev = store[("pooled", vn)]; ax[1].loglog(lams, (ev[-1] + 2 * lams) / (ev[0] + 2 * lams), color=c, label=vn)
ax[1].set_xlabel("lambda"); ax[1].set_ylabel("condition number kappa(lambda)"); ax[1].set_title("Ridge shrinks kappa: (l_max+2 lam)/(l_min+2 lam)")
fig.savefig("results/fig_3c_hessian.png"); plt.close(fig)

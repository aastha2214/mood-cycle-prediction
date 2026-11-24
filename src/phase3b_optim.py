"""3b: GD / SGD / momentum / Adam on the ridge loss; loss curves; learning-rate study (log1p+standardised feature set D, DRSP psychological)."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from src.data import *; from src.mathcore.ridge import *; from src.mathcore.optim import *; from src.plotstyle import *
df = load_analysis(); LAM = 0.01; PASSES = 300; TOL = 1e-6; rows = []; curves = {}
def setup(which):
    d = subset(df, which); X = d[FEATURE_SETS["D"]].values; y = d["DRSP_psychisch"].values
    mu, sd = standardize_fit(X); Zc, yc, _, _ = center((X - mu) / sd, y); ws = ridge_svd(Zc, yc, LAM)
    ev = np.linalg.eigvalsh(hessian(Zc, LAM)); return Zc, yc, ws, ev, len(y), d.VPN.nunique()
def run(name, Zc, yc, lr, beta=0.9):
    if name == "GD": return gd(Zc, yc, LAM, lr, PASSES)
    if name == "Momentum": return momentum(Zc, yc, LAM, lr, beta, PASSES)
    if name == "Adam": return adam(Zc, yc, LAM, lr, PASSES)
    return sgd(Zc, yc, LAM, lr, PASSES, batch=16, decay=0.0)
grids = {"GD": [0.1, 0.5, 1.0, 1.9, 2.1], "Momentum": [0.1, 0.5, 1.0, 1.5, 2.1], "SGD": [0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
         "Adam": [0.001, 0.01, 0.05, 0.1, 0.5]}   # GD/Momentum/SGD: multiples of 1/lmax(H); Adam: absolute step
for which in ["pooled", "long"]:
    Zc, yc, ws, ev, n, nw = setup(which); lmax, lmin = ev[-1], ev[0]; Ls = loss(ws, Zc, yc, LAM); L0 = loss(np.zeros(Zc.shape[1]), Zc, yc, LAM)
    print(f"\n[{which}] n={n} rows, {nw} women | H eigenvalues {lmin:.3f}..{lmax:.3f}, kappa={lmax/lmin:.1f} | L(w*)={Ls:.4f}, L(0)={L0:.4f}")
    base = {"GD": 1 / lmax, "Momentum": 1 / lmax, "SGD": 1 / lmax, "Adam": 1.0}
    for name, g in grids.items():
        for m in g:
            _, L = run(name, Zc, yc, m * base[name]); sub = (L - Ls) / (L[0] - Ls)
            ok = np.isfinite(L).all() and sub[-1] < 1
            hit = int(np.argmax(sub < TOL)) if (sub < TOL).any() else None
            rows.append(dict(sample=which, n=n, optimizer=name, lr_mult=m, lr=m * base[name], final_rel_subopt=sub[-1] if ok else np.inf, passes_to_tol=hit))
    # headline curves at "good" settings
    sets = {"GD": 1 / lmax, "Momentum": 1 / lmax, "Adam": 0.05, "SGD": 0.25 / lmax}
    curves[which] = {k: run(k, Zc, yc, v)[1] for k, v in sets.items()}
    for k, L in curves[which].items(): print(f"  {k:9s} lr={sets[k]:.4g}: rel. suboptimality after {PASSES} passes = {(L[-1]-Ls)/(L[0]-Ls):.2e}")
    # theory check for GD: contraction per step of the loss gap = rho^2, rho = max|1 - lr*lambda_i|
    for lrn, lr in [("1/lmax", 1 / lmax), ("2/(lmax+lmin) optimal", 2 / (lmax + lmin))]:
        rho = np.max(np.abs(1 - lr * ev)); pred = np.log(TOL) / (2 * np.log(rho)); _, L = gd(Zc, yc, LAM, lr, 4000)
        sub = (L - Ls) / (L[0] - Ls); obs = int(np.argmax(sub < TOL))
        print(f"  GD theory, lr={lrn}: rho={rho:.4f}; iterations to 1e-6 predicted <= {pred:.0f}, observed {obs}")
    w_all = {k: run(k, Zc, yc, v)[0] for k, v in sets.items()}
    print("  max|w - w*| after %d passes: " % PASSES + ", ".join(f"{k} {np.abs(w-ws).max():.1e}" for k, w in w_all.items()))
res = pd.DataFrame(rows); res.to_csv("results/phase3b_lr_study.csv", index=False)
print("\nLR study (pooled): final rel. suboptimality after 300 passes; inf = diverged/NaN (GD/Mom/SGD lr in units of 1/lmax; Adam absolute):")
for o in grids:
    q = res[(res["sample"] == "pooled") & (res.optimizer == o)]
    print(f"  {o:9s}", "  ".join(f"{r.lr_mult:g}:{r.final_rel_subopt:.1e}" for r in q.itertuples()))
# Figures
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for a, which in zip(ax, ["pooled", "long"]):
    Zc, yc, ws, ev, n, nw = setup(which); Ls = loss(ws, Zc, yc, LAM)
    for k, L in curves[which].items(): a.semilogy(np.maximum(L - Ls, 1e-16), color=C[k], ls=LS[k], label=k)
    a.set_title(f"{which}: n={n} rows, {nw} women"); a.set_xlabel("passes over the data (GD/Mom/Adam: iterations; SGD: epochs)")
ax[0].set_ylabel("L(w_t) - L(w*)"); ax[0].legend(frameon=False); fig.suptitle("Ridge loss gap vs passes (DRSP psychological, feature set D, lambda=0.01)", y=1.02)
fig.savefig("results/fig_3b_loss_curves.png"); plt.close(fig)
fig, ax = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
for a, o in zip(ax, grids):
    for which, ls in [("pooled", "-"), ("long", "--")]:
        q = res[(res["sample"] == which) & (res.optimizer == o)]; y = q.final_rel_subopt.replace(np.inf, np.nan)
        a.loglog(q.lr_mult, y, ls=ls, marker="o", color=C[o], label=which)
        for r in q.itertuples():
            if not np.isfinite(r.final_rel_subopt): a.axvline(r.lr_mult, color=C["grey"], alpha=.3)
    a.set_title(o + (" (lr / (1/lmax))" if o != "Adam" else " (absolute lr)")); a.set_xlabel("learning rate")
ax[0].set_ylabel("rel. suboptimality @300 passes"); ax[0].legend(frameon=False); fig.savefig("results/fig_3b_lr_study.png"); plt.close(fig)

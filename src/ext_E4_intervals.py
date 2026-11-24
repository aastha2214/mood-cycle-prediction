"""E4: prediction intervals under leave-one-participant-out and their EMPIRICAL COVERAGE at 80/90/95%.
Methods: (1) Bayesian predictive interval from E2 (mean +/- z*sd); (2) jackknife-conformal interval for ridge: inside every training set, the absolute residuals of each training woman are obtained by refitting WITHOUT that woman
(ridge with the lambda chosen by nested CV for the fold), and the interval is ridge prediction +/- the conformal quantile of those residuals; (3) reference: constant Gaussian (training mean +/- z*training SD).
Coverage = share of held-out sessions whose observed value lies inside the interval, with 95% CIs from resampling women; width = mean interval width in the units of the target."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from src.plotstyle import *; from src.data import *; from src.cvlib import predict_one; from src.mathcore.ridge import fit_predict_path; from src.extlib import boot_mean
df = load_analysis(); SETS = ["A", "C", "D"]; LEVELS = [0.80, 0.90, 0.95]; ZQ = {l: stats.norm.ppf(0.5 + l / 2) for l in LEVELS}
def conformal_fold(Xtr, ytr, gtr, Xte, seed):
    pred, hp = predict_one("ridge", Xtr, ytr, gtr, Xte, seed); lam = float(hp); res = []
    for v in np.unique(gtr):
        m = gtr != v; assert set(gtr[m]).isdisjoint(gtr[~m]); res.append(np.abs(ytr[~m] - fit_predict_path(Xtr[m], ytr[m], Xtr[~m], np.array([lam]))[0]))
    r = np.sort(np.concatenate(res)); n = len(r); q = {l: r[min(int(np.ceil(l * (n + 1))) - 1, n - 1)] for l in LEVELS}; return pred, q, n
def one(w, X, y, g):
    te = g == w; tr = ~te; assert set(g[tr]).isdisjoint(g[te]); pred, q, n = conformal_fold(X[tr], y[tr], g[tr], X[te], SEED + int(w)); return np.where(te)[0], pred, q, y[tr].mean(), y[tr].std(ddof=1)
def run_cfg(sample, target, fs):
    tc = TARGETS[target]; cols = FEATURE_SETS[fs]; assert tc not in cols
    if sample == "cross2long":
        tr, te = df[df["sample"] == "cross"], df[df["sample"] == "long"]; pred, q, n = conformal_fold(tr[cols].values, tr[tc].values, tr.VPN.values, te[cols].values, SEED); d = te
        outs = [(np.arange(len(te)), pred, q, tr[tc].mean(), tr[tc].std(ddof=1))]
    else:
        d = subset(df, sample); X, y, g = d[cols].values, d[tc].values, d.VPN.values; outs = Parallel(n_jobs=10)(delayed(one)(w, X, y, g) for w in np.unique(g))
    rows = []
    for idx, pred, q, bm, bs in outs:
        for k, i in enumerate(idx):
            r = dict(sample=sample, target=target, fset=fs, VPN=d.VPN.values[i], Session=d.Session.values[i], y=d[tc].values[i], ridge_pred=pred[k], base_mu=bm, base_sd=bs)
            for l in LEVELS: r[f"q{int(l*100)}"] = q[l]
            rows.append(r)
    return pd.DataFrame(rows)
jobs = [(s, t, f) for s in ["pooled", "long", "cross2long"] for t in TARGETS for f in SETS]
R = pd.concat([run_cfg(*j) for j in jobs]); E2 = pd.read_csv("results/extE2_oof.csv")[["sample", "target", "fset", "VPN", "Session", "pred", "sd"]].rename(columns={"pred": "bayes_pred", "sd": "bayes_sd"})
R = R.merge(E2, on=["sample", "target", "fset", "VPN", "Session"], how="left", validate="one_to_one"); assert R.bayes_sd.notna().all()
# ---- long format with the three methods
M = {"bayes_predictive": ("bayes_pred", lambda r, l: ZQ[l] * r.bayes_sd), "jackknife_conformal": ("ridge_pred", lambda r, l: r[f"q{int(l*100)}"]), "constant_gaussian": ("base_mu", lambda r, l: ZQ[l] * r.base_sd)}
out = []
for name, (cen, half) in M.items():
    for l in LEVELS:
        h = half(R, l); c = R[cen]; out.append(R[["sample", "target", "fset", "VPN", "Session", "y"]].assign(method=name, level=l, lo=c - h, hi=c + h, width=2 * h, covered=((R.y >= c - h) & (R.y <= c + h)).astype(float)))
I = pd.concat(out); I.to_csv("results/extE4_intervals_oof.csv", index=False)
rows = []
for (s, t, f, m, l), g in I.groupby(["sample", "target", "fset", "method", "level"]):
    cv, cl, ch = boot_mean(g, "covered"); wv, wl, wh = boot_mean(g, "width")
    rows.append(dict(sample=s, target=t, fset=f, method=m, level=l, n_rows=len(g), n_women=g.VPN.nunique(), coverage=cv, cov_lo=cl, cov_hi=ch, mean_width=wv, width_lo=wl, width_hi=wh, covers_nominal=bool(cl <= l <= ch)))
C = pd.DataFrame(rows); ref = C[C.method == "constant_gaussian"].set_index(["sample", "target", "fset", "level"]).mean_width
C["width_ratio_vs_constant"] = [w / ref.loc[(r.sample, r.target, r.fset, r.level)] for w, r in zip(C.mean_width, C.itertuples())]; C.to_csv("results/extE4_coverage.csv", index=False)
print(f"E4 intervals: {len(R)} held-out rows over {len(jobs)} configurations x 3 methods x 3 levels")
for s, g in C.groupby("sample"):
    print(f"\n{s}: n={g.n_rows.iloc[0]} rows, {g.n_women.iloc[0]} women (test). coverage by method and nominal level (mean over {len(g)//(3*3)} configs), share of configs whose CI contains nominal")
    for m, gm in g.groupby("method"):
        print(f"  {m:20s} " + "  ".join(f"{l:.2f}: cov {gm[gm.level==l].coverage.mean():.3f}, ok {gm[gm.level==l].covers_nominal.mean():.2f}, width {gm[gm.level==l].mean_width.mean():.2f}" for l in LEVELS))

# ---- figure: empirical vs nominal coverage (points = the 9 target x feature-set configurations, line = their mean)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True); col = {"bayes_predictive": "#0072B2", "jackknife_conformal": "#D55E00", "constant_gaussian": "#888888"}
for a, s in zip(ax, ["pooled", "long", "cross2long"]):
    g = C[C["sample"] == s]; n = g.n_rows.iloc[0]; w = g.n_women.iloc[0]
    for m, gm in g.groupby("method"):
        a.plot(LEVELS, [gm[gm.level == l].coverage.mean() for l in LEVELS], "-o", color=col[m], label=m.replace("_", " "))
        for l in LEVELS: a.scatter([l] * len(gm[gm.level == l]), gm[gm.level == l].coverage, color=col[m], s=8, alpha=.35)
    a.plot([0.75, 1], [0.75, 1], color="#333", lw=.8); a.set_title(f"{s}: {n} test rows, {w} women", fontsize=9); a.set_xlabel("nominal level")
ax[0].set_ylabel("empirical coverage"); ax[0].legend(frameon=False, fontsize=7); fig.savefig("results/fig_E4_coverage.png"); plt.close(fig)

"""3d: ridge path, bias-variance (bootstrap over women), lambda chosen by nested grouped CV."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from src.data import *; from src.mathcore.ridge import *; from src.plotstyle import *
df = load_analysis(); LAMS = np.logspace(-4, 1, 26); rng = np.random.default_rng(SEED); FS = FEATURE_SETS["D"]
def inner_folds(women, k, rng):
    w = rng.permutation(women); return np.array_split(w, k)
def nested_lopo(d, tcol, cols, rng):
    X = d[cols].values; y = d[tcol].values; g = d.VPN.values; women = np.unique(g); n = len(y)
    oof = np.zeros(n); base = np.zeros(n); chosen = []
    for w in women:
        te = g == w; tr = ~te; trw = np.unique(g[tr]); sse = np.zeros(len(LAMS))
        for f in inner_folds(trw, 5, rng):
            iv = np.isin(g, f) & tr; it = tr & ~iv
            P = fit_predict_path(X[it], y[it], X[iv], LAMS); sse += ((P - y[iv]) ** 2).sum(1)
        j = int(np.argmin(sse)); chosen.append(LAMS[j])
        oof[te] = fit_predict_path(X[tr], y[tr], X[te], LAMS[[j]])[0]; base[te] = y[tr].mean()
    return oof, base, np.array(chosen)
rows = []
print("3d nested CV: outer = leave-one-woman-out, inner = 5-fold by woman, lambda grid 1e-4..10 (26 values), feature set D (log1p+std)")
for which in ["pooled", "long"]:
    d = subset(df, which); n = len(d); nw = d.VPN.nunique()
    for tn, tc in TARGETS.items():
        oof, base, ch = nested_lopo(d, tc, FS, np.random.default_rng(SEED)); y = d[tc].values
        mae, rmse = np.abs(oof - y).mean(), np.sqrt(((oof - y) ** 2).mean()); mae_b, rmse_b = np.abs(base - y).mean(), np.sqrt(((base - y) ** 2).mean())
        r2 = 1 - ((oof - y) ** 2).sum() / ((base - y) ** 2).sum()
        rows.append(dict(sample=which, n_rows=n, n_women=nw, target=tn, ridge_MAE=mae, base_MAE=mae_b, ridge_RMSE=rmse, base_RMSE=rmse_b, R2_vs_mean=r2,
                         lam_median=np.median(ch), lam_q25=np.percentile(ch, 25), lam_q75=np.percentile(ch, 75), frac_at_grid_max=(ch == LAMS[-1]).mean()))
res = pd.DataFrame(rows); res.to_csv("results/phase3d_nested_cv.csv", index=False); print(res.round(3).to_string(index=False))
# ---- ridge path + bias-variance (DRSP psychological, set D)
fig, ax = plt.subplots(2, 3, figsize=(14, 8)); names = ["phase: peri-ov", "phase: luteal", "log E2", "log P", "PSST", "BDI", "BAI"]
pal = ["#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00", "#CC79A7", "#333333"]; sty = ["-", "--", "-", "--", "-", "--", "-"]
bv_rows = []
for r, which in enumerate(["pooled", "long"]):
    d = subset(df, which); X = d[FS].values; y = d["DRSP_psychisch"].values; g = d.VPN.values; women = np.unique(g); n = len(y)
    mu, sd = standardize_fit(X); Zc, yc, _, _ = center((X - mu) / sd, y); W = ridge_path_svd(Zc, yc, LAMS)
    for j in range(W.shape[1]): ax[r, 0].semilogx(LAMS, W[:, j], color=pal[j], ls=sty[j], label=names[j])
    ax[r, 0].axhline(0, color="#888", lw=.8); ax[r, 0].set_title(f"{which}: ridge path\n(n={n} rows, {len(women)} women)"); ax[r, 0].set_xlabel("lambda"); ax[r, 0].set_ylabel("coefficient (per SD of feature)")
    if r == 0: ax[r, 0].legend(frameon=False, fontsize=7, ncol=2)
    # bias-variance: R random splits of women (25% held out); B bootstrap fits over training women
    R, B = 10, 100; acc = np.zeros((3, len(LAMS)))
    for _ in range(R):
        pw = rng.permutation(women); te_w = pw[:len(women) // 4]; tr_w = pw[len(women) // 4:]; te = np.isin(g, te_w)
        preds = np.zeros((B, len(LAMS), te.sum()))
        for b in range(B):
            pick = rng.choice(tr_w, len(tr_w)); idx = np.concatenate([np.where(g == w)[0] for w in pick])
            preds[b] = fit_predict_path(X[idx], y[idx], X[te], LAMS)
        mp = preds.mean(0); var = preds.var(0).mean(1); b2 = ((mp - y[te]) ** 2).mean(1)  # b2 = bias^2 + irreducible noise (not separable on real data)
        acc += np.array([var, b2, var + b2])
    acc /= R; bv_rows += [dict(sample=which, n_rows=n, n_women=len(women), lam=l, variance=a, bias2_plus_noise=b, expected_sq_err=c) for l, a, b, c in zip(LAMS, *acc)]
    ax[r, 1].semilogx(LAMS, acc[0], color=C["SGD"], ls="--", label="variance"); ax[r, 1].semilogx(LAMS, acc[1], color=C["GD"], ls="-.", label="bias$^2$ + noise")
    ax[r, 1].semilogx(LAMS, acc[2], color=C["ink"], label="expected squared error"); ax[r, 1].axvline(LAMS[acc[2].argmin()], color=C["grey"], lw=.8)
    ax[r, 1].set_title(f"{which}: bias-variance\n(10 splits x 100 woman-bootstraps)"); ax[r, 1].set_xlabel("lambda"); ax[r, 1].set_ylabel("DRSP$^2$ units"); ax[r, 1].legend(frameon=False, fontsize=8)
    ch_best = LAMS[acc[2].argmin()]; print(f"3d bias-variance [{which}] n={n}: variance falls {acc[0][0]:.2f} -> {acc[0][-1]:.2f}; bias2+noise rises {acc[1][0]:.1f} -> {acc[1][-1]:.1f}; min expected sq err {acc[2].min():.1f} at lambda={ch_best:.3g}")
    ch = res[(res["sample"] == which) & (res.target == "drsp_psych")]
    ax[r, 2].axis("off"); ax[r, 2].text(0, .9, f"Nested CV (DRSP, set D)\nlambda median {ch.lam_median.iloc[0]:.3g}\nIQR {ch.lam_q25.iloc[0]:.3g} - {ch.lam_q75.iloc[0]:.3g}\nridge MAE {ch.ridge_MAE.iloc[0]:.2f} vs mean-baseline {ch.base_MAE.iloc[0]:.2f}\nR2 vs mean {ch.R2_vs_mean.iloc[0]:.3f}", va="top", fontsize=10)
fig.savefig("results/fig_3d_path_biasvar.png"); plt.close(fig); pd.DataFrame(bv_rows).to_csv("results/phase3d_bias_variance.csv", index=False)

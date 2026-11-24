"""Phase 6a negative control: predict emotion-recognition accuracy (and speed) from hormones with the Phase 4 protocol. The paper found no link, so skill should be ~0.
Session accuracy = % correct over the session's 60 trials; speed = median RT (ms) of correct trials. No RT-outlier cut (the authors' cutoff uses global RT statistics; skipped to avoid any global preprocessing)."""
import sys, warnings; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd, pyreadstat, statsmodels.formula.api as smf
from scipy import stats
from src.data import *; from src.cvlib import *; from src.evalutils import boot; from src.plotstyle import *
warnings.filterwarnings("ignore")
e, _ = pyreadstat.read_sav(str(ROOT / "data/RV20_EmotionRecognition_MC_20230519.sav"), usecols=["VPN", "Session", "RT", "Hit"])
er = e.groupby(["VPN", "Session"]).apply(lambda g: pd.Series(dict(er_accuracy=100 * g.Hit.mean(), er_speed=g.RT[g.Hit == 1].median(), n_trials=len(g))), include_groups=False).reset_index()
df = load_analysis().merge(er, on=["VPN", "Session"], how="left"); assert df.er_accuracy.notna().all() and (df.n_trials == 60).all()
print(f"Merged emotion-recognition outcomes: {len(df)} rows, {df.VPN.nunique()} women (all 271 analysis sessions have 60 trials). accuracy mean {df.er_accuracy.mean():.1f}% (SD {df.er_accuracy.std():.1f}); speed median {df.er_speed.median():.0f} ms")
T = {"er_accuracy": "er_accuracy", "er_speed": "er_speed"}; SETS = ["A", "B", "C", "D"]; rows = []
def add(sample, tn, model, fs, d, pred, hp): rows.append(pd.DataFrame(dict(sample=sample, target=tn, model=model, fset=fs, VPN=d.VPN.values, Session=d.Session.values, y=d[tn].values, pred=pred, hp=hp)))
for which in ["pooled", "long"]:
    d = subset(df, which)
    for tn in T:
        p, hp = lopo(d, tn, FEATURE_SETS["A"], "baseline"); add(which, tn, "baseline", "-", d, p, hp)
        for fs in SETS:
            for m in ["ridge", "rf", "gb"]: p, hp = lopo(d, tn, FEATURE_SETS[fs], m); add(which, tn, m, fs, d, p, hp)
cr, lg = df[df["sample"] == "cross"], df[df["sample"] == "long"]
for tn in T:
    p, hp = predict_one("baseline", cr[FEATURE_SETS["A"]].values, cr[tn].values, cr.VPN.values, lg[FEATURE_SETS["A"]].values, SEED); add("cross2long", tn, "baseline", "-", lg, p, hp)
    for fs in SETS:
        for m in ["ridge", "rf", "gb"]: p, hp = predict_one(m, cr[FEATURE_SETS[fs]].values, cr[tn].values, cr.VPN.values, lg[FEATURE_SETS[fs]].values, SEED); add("cross2long", tn, m, fs, lg, p, hp)
oof = pd.concat(rows); oof.to_csv("results/phase6a_oof.csv", index=False); M, D = boot(oof); M.to_csv("results/phase6a_metrics.csv", index=False); D.to_csv("results/phase6a_deltas.csv", index=False)
print("\nNEGATIVE CONTROL: R2_oos vs global-mean baseline, point [95% woman-bootstrap CI]")
for (s, t), g in M[M.model != "baseline"].groupby(["sample", "target"]):
    b = M[(M["sample"] == s) & (M.target == t) & (M.model == "baseline")].iloc[0]; print(f"\n{s} | {t} | n={b.n_rows} rows, {b.n_women} test women | baseline RMSE {b.RMSE:.3f}")
    for f in SETS: print(f"  {f}  " + "  ".join(f"{m}: {g[(g.model==m)&(g.fset==f)].R2_oos.iloc[0]:+.3f}[{g[(g.model==m)&(g.fset==f)].R2_oos_lo.iloc[0]:+.2f},{g[(g.model==m)&(g.fset==f)].R2_oos_hi.iloc[0]:+.2f}]" for m in ["ridge", "rf", "gb"]))
print("\nHormone comparisons (RMSE; 18 tests per family = 3 samples x 2 targets x 3 models). Positive delta = first set worse")
for c in ["B vs baseline", "C vs A"]:
    g = D[D.comparison == c]; print(f"  {c}: first set better (delta<0, CI excl. 0): {((g.delta_RMSE<0)&(g.hi<0)).sum()} | worse: {((g.delta_RMSE>0)&(g.lo>0)).sum()} | raw p<0.05: {(g.p_two_sided<.05).sum()}, Holm<0.05: {(g.p_holm<.05).sum()}, BH<0.05: {(g.p_fdr_bh<.05).sum()} | smallest one-sided 'improves' p: {g.p_improve.min():.3f}")
# mixed-model check (authors: no phase / hormone effect on recognition)
print("\nMixed models (random intercept, z-scored; Wald p) as in the paper's analysis, accuracy and speed:")
MM = []
for which in ["long", "pooled"]:
    d = subset(df, which).copy()
    for c in ["er_accuracy", "er_speed", "estradiol", "progesterone"]: d["z_" + c] = (d[c] - d[c].mean()) / d[c].std(ddof=1)
    for y in ["er_accuracy", "er_speed"]:
        r = smf.mixedlm(f"z_{y} ~ ph1 + ph2", d, groups=d.VPN).fit(reml=True); b = r.fe_params[["ph1", "ph2"]].values; S = r.cov_params().loc[["ph1", "ph2"], ["ph1", "ph2"]].values; pj = stats.chi2.sf(b @ np.linalg.solve(S, b), 2)
        h = smf.mixedlm(f"z_{y} ~ z_estradiol * z_progesterone", d, groups=d.VPN).fit(reml=True)
        MM.append(dict(sample=which, n_rows=len(d), n_women=d.VPN.nunique(), outcome=y, phase_joint_p=pj, E2_b=h.fe_params["z_estradiol"], E2_p=h.pvalues["z_estradiol"], P_b=h.fe_params["z_progesterone"], P_p=h.pvalues["z_progesterone"]))
MM = pd.DataFrame(MM); MM.to_csv("results/phase6a_mixed.csv", index=False); print(MM.round(3).to_string(index=False))
fig, ax = plt.subplots(2, 3, figsize=(13, 6), sharex=True); cm = {"ridge": C["GD"], "rf": C["Momentum"], "gb": C["SGD"]}
for i, t in enumerate(T):
    for j, s in enumerate(["pooled", "long", "cross2long"]):
        a = ax[i, j]
        for q, m in enumerate(["ridge", "rf", "gb"]):
            g = M[(M["sample"] == s) & (M.target == t) & (M.model == m)].set_index("fset").loc[SETS]; x = np.arange(len(SETS)) + (q - 1) * .22
            a.errorbar(x, g.R2_oos, yerr=[g.R2_oos - g.R2_oos_lo, g.R2_oos_hi - g.R2_oos], fmt="o", ms=4, color=cm[m], label=m, capsize=2, lw=1.2)
        b = M[(M["sample"] == s) & (M.target == t)].iloc[0]; a.axhline(0, color=C["ink"], lw=1); a.set_title(f"{t} | {s}\n{b.n_rows} rows, {b.n_women} test women", fontsize=9); a.set_xticks(range(len(SETS))); a.set_xticklabels(SETS)
        if j == 0: a.set_ylabel("R2 vs global-mean baseline")
ax[0, 0].legend(frameon=False, fontsize=8); fig.suptitle("Negative control: emotion recognition from phase/hormones/traits (expected: no skill)", y=1.0); fig.savefig("results/fig_6a_negctrl.png"); plt.close(fig)

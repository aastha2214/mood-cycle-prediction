"""E7: CROSS-SAMPLE validation. Two disjoint groups of women from the same study: the 95 single-session women and the 65 longitudinal women (176 sessions).
Direction 1: train on the single-session women, test on the longitudinal women. Direction 2: train on the longitudinal women, test on the single-session women.
Models: training-sample mean (baseline), ridge (scikit-learn), ridge from scratch, random forest, gradient boosting; feature sets A, B, C, D, D2; three targets.
Hyper-parameters by inner CV grouped by woman INSIDE the training sample only; standardisation uses TRAINING-sample statistics only. Bootstrap CIs resample the women of the TEST sample."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from statsmodels.stats.multitest import multipletests
from src.data import *; from src.cvlib import predict_one, LAMS; from src.extlib import boot_configs; from src.evalutils import _stats, _met
from src.plotstyle import *
df = load_analysis(); cross, long_ = df[df["sample"] == "cross"], df[df["sample"] == "long"]; SETS = ["A", "B", "C", "D2", "D"]; B = 2000
DIRS = {"cross_to_long": (cross, long_), "long_to_cross": (long_, cross)}
def ridge_sklearn(Xtr, ytr, gtr, Xte, seed, k_inner=5):
    """scikit-learn Ridge on StandardScaler features; same inner folds as the from-scratch version; alpha = n_rows * lambda (my loss is (1/n)RSS + lambda ||w||^2)."""
    folds = np.array_split(np.random.default_rng(seed).permutation(np.unique(gtr)), k_inner); sse = np.zeros(len(LAMS))
    for f in folds:
        iv = np.isin(gtr, f); it = ~iv; assert set(gtr[it]).isdisjoint(gtr[iv])
        for j, lam in enumerate(LAMS): sse[j] += ((make_pipeline(StandardScaler(), Ridge(alpha=it.sum() * lam)).fit(Xtr[it], ytr[it]).predict(Xtr[iv]) - ytr[iv]) ** 2).sum()
    j = int(np.argmin(sse)); return make_pipeline(StandardScaler(), Ridge(alpha=len(ytr) * LAMS[j])).fit(Xtr, ytr).predict(Xte), f"{LAMS[j]:.3g}"
rows = []; sizes = []
for dname, (tr, te) in DIRS.items():
    assert set(tr.VPN).isdisjoint(set(te.VPN)), "a woman appears in both train and test"; assert not (set(tr.VPN) & set(te.VPN)) and len(set(tr.VPN) | set(te.VPN)) == len(set(tr.VPN)) + len(set(te.VPN))
    assert not te.duplicated(["VPN", "Session"]).any() and not tr.duplicated(["VPN", "Session"]).any()
    sizes.append(dict(direction=dname, train_rows=len(tr), train_women=tr.VPN.nunique(), test_rows=len(te), test_women=te.VPN.nunique(), women_in_both=len(set(tr.VPN) & set(te.VPN))))
    for t, tc in TARGETS.items():
        ytr, gtr = tr[tc].values, tr.VPN.values; assert tc not in sum(FEATURE_SETS.values(), [])
        base, _ = predict_one("baseline", tr[FEATURE_SETS["A"]].values, ytr, gtr, te[FEATURE_SETS["A"]].values, SEED); rows.append(te[["VPN", "Session"]].assign(sample=dname, target=t, model="baseline", fset="-", y=te[tc].values, pred=base, hp="mean"))
        for fs in SETS:
            Xtr, Xte = tr[FEATURE_SETS[fs]].values, te[FEATURE_SETS[fs]].values
            for m in ["ridge", "ridge_scratch", "rf", "gb"]:
                p, hp = ridge_sklearn(Xtr, ytr, gtr, Xte, SEED) if m == "ridge" else predict_one("ridge" if m == "ridge_scratch" else m, Xtr, ytr, gtr, Xte, SEED)
                rows.append(te[["VPN", "Session"]].assign(sample=dname, target=t, model=m, fset=fs, y=te[tc].values, pred=p, hp=hp))
oof = pd.concat(rows)[["sample", "target", "model", "fset", "VPN", "Session", "y", "pred", "hp"]]; oof.to_csv("results/ext_e7_oof.csv", index=False); pd.DataFrame(sizes).to_csv("results/ext_e7_sizes.csv", index=False)
# ---- parity of scikit-learn ridge and from-scratch ridge
a = oof[oof.model == "ridge"].set_index(["sample", "target", "fset", "VPN", "Session"]).pred; b_ = oof[oof.model == "ridge_scratch"].set_index(["sample", "target", "fset", "VPN", "Session"]).pred
pd.DataFrame([dict(n_predictions=len(a), max_abs_pred_diff=float((a - b_.loc[a.index]).abs().max()))]).to_csv("results/ext_e7_ridge_parity.csv", index=False)
# ---- consistency with Phase 4: direction 1 repeats the cross-sample experiment of Phase 4 (same seed, same code path) for the from-scratch ridge, random forest and gradient boosting
p4 = pd.read_csv("results/phase4_oof.csv"); p4 = p4[p4["sample"] == "cross2long"].assign(model=lambda x: x.model.replace({"ridge": "ridge_scratch"}))
mm = oof[oof["sample"] == "cross_to_long"].merge(p4, on=["target", "model", "fset", "VPN", "Session"], suffixes=("", "_p4"))
pd.DataFrame([dict(n_predictions_compared=len(mm), max_abs_pred_diff_vs_phase4=float((mm.pred - mm.pred_p4).abs().max()))]).to_csv("results/ext_e7_phase4_check.csv", index=False)
# ---- metrics with women-bootstrap CIs (test women resampled); paired differences hormones vs no hormones
MODELS_ALL = ["ridge", "ridge_scratch", "rf", "gb"]; comps = [((m, "C"), (m, "A")) for m in MODELS_ALL] + [((m, "B"), ("baseline", "-")) for m in MODELS_ALL]
M, D = boot_configs(oof, comps); M.to_csv("results/ext_e7_metrics.csv", index=False); D.to_csv("results/ext_e7_deltas.csv", index=False)
# ---- hormone tests: bootstrap p-values (same resamples as boot_configs), Holm and Benjamini-Hochberg over ALL 36 hormone comparisons and within each family of 18
rng = np.random.default_rng(SEED); T = []
for (dname, t), blk in oof.groupby(["sample", "target"]):
    women = np.sort(blk.VPN.unique()); W = rng.multinomial(len(women), np.ones(len(women)) / len(women), size=B).astype(float); S = {k: _stats(g).loc[women] for k, g in blk.groupby(["model", "fset"])}
    rm = lambda k: np.sqrt((W @ S[k].sse.values) / (W @ S[k].n.values)); pt = lambda k: np.sqrt(S[k].sse.sum() / S[k].n.sum())
    for m in ["ridge", "rf", "gb"]:
        for fam, k1, k2 in [("C vs A", (m, "C"), (m, "A")), ("B vs baseline", (m, "B"), ("baseline", "-"))]:
            d = rm(k1) - rm(k2); lo, hi = np.percentile(d, [2.5, 97.5]); n0, n1 = (d <= 0).sum(), (d >= 0).sum()
            T.append(dict(direction=dname, target=t, model=m, family=fam, n_test_rows=int(S[k1].n.sum()), n_test_women=len(women), delta_RMSE=pt(k1) - pt(k2), ci_lo=lo, ci_hi=hi, p_two_sided=min(1, 2 * (min(n0, n1) + 1) / (B + 1)), p_improve_one_sided=(n1 + 1) / (B + 1)))
T = pd.DataFrame(T)
for col, nm in [("p_two_sided", ""), ("p_improve_one_sided", "_improve")]:
    T["p_holm_all" + nm] = multipletests(T[col], method="holm")[1]; T["p_fdr_all" + nm] = multipletests(T[col], method="fdr_bh")[1]
    for fam, g in T.groupby("family"): T.loc[g.index, "p_holm_family" + nm] = multipletests(g[col], method="holm")[1]; T.loc[g.index, "p_fdr_family" + nm] = multipletests(g[col], method="fdr_bh")[1]
T.to_csv("results/ext_e7_hormone_tests.csv", index=False)
chk = D[(D.metric == "RMSE")].assign(model=lambda x: x["first"].str.split("/").str[0], fam=lambda x: np.where(x["second"].str.startswith("baseline"), "B vs baseline", "C vs A")).merge(T, left_on=["sample", "target", "model", "fam"], right_on=["direction", "target", "model", "family"], suffixes=("", "_t"))
assert np.allclose(chk.delta, chk.delta_RMSE) and np.allclose(chk.lo, chk.ci_lo) and np.allclose(chk.hi, chk.ci_hi), "bootstrap CIs differ between the two code paths"
# ---- figure: out-of-sample skill by direction, target, feature set and model
fig, ax = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True); cm = {"ridge": C["GD"], "ridge_scratch": "#56B4E9", "rf": C["Momentum"], "gb": C["SGD"]}
for i, dname in enumerate(DIRS):
    for j, t in enumerate(TARGETS):
        a_ = ax[i, j]
        for q, m in enumerate(MODELS_ALL):
            g = M[(M["sample"] == dname) & (M.target == t) & (M.model == m)].set_index("fset").loc[SETS]; x = np.arange(len(SETS)) + (q - 1.5) * .17
            a_.errorbar(x, g.R2_oos, yerr=[g.R2_oos - g.R2_oos_lo, g.R2_oos_hi - g.R2_oos], fmt="o", ms=3.5, color=cm[m], label=m, capsize=2, lw=1)
        a_.axhline(0, color=C["ink"], lw=1); r = M[(M["sample"] == dname) & (M.target == t)].iloc[0]; a_.set_title(f"{dname.replace('_', ' ')} | {t}\ntest: {r.n_rows} rows, {r.n_women} women", fontsize=9); a_.set_xticks(range(len(SETS))); a_.set_xticklabels(SETS)
        if j == 0: a_.set_ylabel("R2 vs training-sample mean")
ax[0, 0].legend(frameon=False, fontsize=7); fig.savefig("results/fig_e7_cross_sample.png"); plt.close(fig)
print("E7 cross-sample validation:"); print(pd.DataFrame(sizes).to_string(index=False)); print("ridge parity (sklearn vs scratch): max |diff| = %.1e" % pd.read_csv("results/ext_e7_ridge_parity.csv").max_abs_pred_diff.iloc[0])
for dname in DIRS:
    g = M[(M["sample"] == dname) & (M.model == "ridge")]
    print(f"\n{dname} (test {g.n_rows.iloc[0]} rows, {g.n_women.iloc[0]} women): ridge R2_oos by set [DRSP / pos.affect / anxiety]")
    for f in SETS: print(f"  {f:2s} " + " / ".join(f"{g[(g.target==t)&(g.fset==f)].R2_oos.iloc[0]:+.3f}" for t in ["drsp_psych", "pos_affect", "stai"]))
print(f"\nhormone comparisons: {len(T)} tests; raw p<0.05: {(T.p_two_sided<.05).sum()}, Holm(all)<0.05: {(T.p_holm_all<.05).sum()}, FDR(all)<0.05: {(T.p_fdr_all<.05).sum()}; hormones significantly BETTER (one-sided raw p<0.05): {(T.p_improve_one_sided<.05).sum()}; min one-sided p {T.p_improve_one_sided.min():.3f}")

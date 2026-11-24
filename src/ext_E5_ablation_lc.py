"""E5: (a) ablation: drop each feature group from set D (and each group alone); (b) learning curves: leave-one-woman-out with the training set restricted to k randomly chosen OTHER women.
Ablation uses ridge (nested CV) in all samples and random forest (drop-one sets, pooled and longitudinal). Learning curves: ridge, sets A, C, D, with mean TRAIN R2 as well. CIs resample women."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from src.plotstyle import *; from src.data import *; from src.cvlib import lopo, predict_one; from src.extlib import boot_configs, _ci
df = load_analysis(); FULL = FEATURE_SETS["D"]; GROUPS = {"phase": ["ph1", "ph2"], "hormones": ["logE2", "logP"], "PSST": ["PMS"], "BDI": ["BDI_RW"], "BAI": ["BAI_RW"]}
SETS = {f"D-{g}": [c for c in FULL if c not in cols] for g, cols in GROUPS.items()}; SETS.update({f"only-{g}": cols for g, cols in GROUPS.items()}); SETS["only-questionnaires"] = ["PMS", "BDI_RW", "BAI_RW"]
DROP = [f"D-{g}" for g in GROUPS]; cols_ = ["sample", "target", "model", "fset", "VPN", "Session", "y", "pred"]; rows = []
def addrows(sample, t, model, fs, d, pred): rows.append(pd.DataFrame(dict(sample=sample, target=t, model=model, fset=fs, VPN=d.VPN.values, Session=d.Session.values, y=d[TARGETS[t]].values, pred=pred)))
for sample in ["pooled", "long", "cross2long"]:
    for t, tc in TARGETS.items():
        for fs, cols in SETS.items():
            if sample == "cross2long": tr, te = df[df["sample"] == "cross"], df[df["sample"] == "long"]; p, _ = predict_one("ridge", tr[cols].values, tr[tc].values, tr.VPN.values, te[cols].values, SEED); addrows(sample, t, "ridge", fs, te, p)
            else: d = subset(df, sample); p, _ = lopo(d, tc, cols, "ridge"); addrows(sample, t, "ridge", fs, d, p)
        if sample != "cross2long":
            d = subset(df, sample)
            for fs in DROP: p, _ = lopo(d, tc, SETS[fs], "rf"); addrows(sample, t, "rf", fs, d, p)
oof = pd.concat(rows); oof.to_csv("results/extE5_ablation_oof.csv", index=False)
p4 = pd.read_csv("results/phase4_oof.csv"); p4 = p4[((p4.model == "baseline") | (p4.model.isin(["ridge", "rf"]) & (p4.fset == "D")))][cols_]
comps = [((m, f), (m, "D")) for m in ["ridge", "rf"] for f in DROP]
M, D = boot_configs(pd.concat([p4, oof[cols_]]), comps); M.to_csv("results/extE5_ablation_metrics.csv", index=False); D.to_csv("results/extE5_ablation_deltas.csv", index=False)
# ---- learning curves
KS = {"pooled": [10, 20, 40, 80, 159], "long": [10, 20, 30, 40, 64]}; LSETS = ["A", "C", "D"]
def lc_fold(w, X, y, g, k):
    te = g == w; tr = ~te; pool = np.unique(g[tr]); pick = np.random.default_rng(SEED + int(w) * 1000 + k).choice(pool, size=k, replace=False); m = np.isin(g, pick); assert set(g[m]).isdisjoint(g[te]) and m.sum() > 0
    P, _ = predict_one("ridge", X[m], y[m], g[m], np.vstack([X[te], X[m]]), SEED + int(w)); pt, ptr = P[:te.sum()], P[te.sum():]; b = y[m].mean()
    return w, float(((pt - y[te]) ** 2).sum()), float(((b - y[te]) ** 2).sum()), int(te.sum()), float(1 - ((ptr - y[m]) ** 2).sum() / ((y[m] - b) ** 2).sum()), float(np.abs(pt - y[te]).sum())
lc = []; rng = np.random.default_rng(SEED)
for sample, ks in KS.items():
    d = subset(df, sample); g = d.VPN.values; women = np.unique(g)
    for t, tc in TARGETS.items():
        for fs in LSETS:
            X, y = d[FEATURE_SETS[fs]].values, d[tc].values
            for k in ks:
                res = pd.DataFrame(Parallel(n_jobs=10)(delayed(lc_fold)(w, X, y, g, k) for w in women), columns=["w", "sse", "sse_base", "n", "train_r2", "sae"]); W = np.random.default_rng(SEED).multinomial(len(women), np.ones(len(women)) / len(women), size=2000).astype(float)
                f = lambda wt: dict(R2_oos=1 - wt @ res.sse.values / (wt @ res.sse_base.values), RMSE=np.sqrt(wt @ res.sse.values / (wt @ res.n.values)), train_R2=(wt @ res.train_r2.values) / wt.sum(1) if wt.ndim > 1 else res.train_r2.mean())
                pt, bt = f(np.ones(len(women))), f(W); r = dict(sample=sample, target=t, fset=fs, n_train_women=k, n_test_rows=int(res.n.sum()), n_test_women=len(women), RMSE_baseline=float(np.sqrt(res.sse_base.sum() / res.n.sum())))
                for q in ["R2_oos", "RMSE", "train_R2"]: r[q] = pt[q]; r[q + "_lo"], r[q + "_hi"] = _ci(bt[q])
                lc.append(r)
L = pd.DataFrame(lc); L.to_csv("results/extE5_learning_curves.csv", index=False)
# identity check: k = all other women reproduces the Phase 4 ridge predictions' R2_oos
chk = []
for sample, ks in KS.items():
    for t in TARGETS:
        a = L[(L["sample"] == sample) & (L.target == t) & (L.fset == "D") & (L.n_train_women == ks[-1])].R2_oos.iloc[0]; b = pd.read_csv("results/phase4_metrics.csv").query("sample == @sample and target == @t and model == 'ridge' and fset == 'D'").R2_oos.iloc[0]; chk.append(dict(sample=sample, target=t, R2_oos_lc_all_women=a, R2_oos_phase4=b, abs_diff=abs(a - b)))
pd.DataFrame(chk).to_csv("results/extE5_lc_identity_check.csv", index=False)
# ---- figures
fig, ax = plt.subplots(2, 3, figsize=(13, 6.5), sharex="row"); colr = {"A": "#888888", "C": "#009E73", "D": "#0072B2"}
for i, sample in enumerate(KS):
    for j, t in enumerate(TARGETS):
        a = ax[i, j]
        for fs in LSETS:
            g = L[(L["sample"] == sample) & (L.target == t) & (L.fset == fs)]; a.plot(g.n_train_women, g.R2_oos, "-o", color=colr[fs], label=f"set {fs}: test"); a.fill_between(g.n_train_women, g.R2_oos_lo, g.R2_oos_hi, color=colr[fs], alpha=.12)
            if fs == "D": a.plot(g.n_train_women, g.train_R2, "--", color=colr[fs], label="set D: train")
        a.axhline(0, color="#333", lw=.8); a.set_xscale("log"); a.set_title(f"{sample} | {t}\\n{int(g.n_test_rows.iloc[0])} test rows, {int(g.n_test_women.iloc[0])} women", fontsize=9); a.set_xlabel("training women")
        if j == 0: a.set_ylabel("R2 (test: vs mean baseline)")
ax[0, 0].legend(frameon=False, fontsize=7); fig.savefig("results/fig_E5_learning_curves.png"); plt.close(fig)
fig, ax = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
for i, sample in enumerate(["pooled", "long"]):
    for j, t in enumerate(TARGETS):
        a = ax[i, j]; g = D[(D["sample"] == sample) & (D.target == t) & (D.metric == "RMSE") & (D["first"].str.startswith("ridge/"))].set_index("first").loc[[f"ridge/{f}" for f in DROP]]
        a.errorbar(range(len(DROP)), g.delta, yerr=[g.delta - g.lo, g.hi - g.delta], fmt="o", color="#0072B2", capsize=3); a.axhline(0, color="#333", lw=.8); a.set_xticks(range(len(DROP))); a.set_xticklabels([f.replace("D-", "drop\\n") for f in DROP], fontsize=8)
        a.set_title(f"{sample} | {t}\\n{int(g.n_rows.iloc[0])} rows, {int(g.n_women.iloc[0])} women", fontsize=9)
        if j == 0: a.set_ylabel("RMSE change when the group is dropped")
fig.savefig("results/fig_E5_ablation.png"); plt.close(fig)
print(f"E5: ablation {len(oof)} out-of-fold rows; learning curves {len(L)} points; k=all identity check max |diff| = {pd.DataFrame(chk).abs_diff.max():.1e}")
dd = D[D.metric == "RMSE"]
for s_ in ["pooled", "long"]:
    g = dd[(dd["sample"] == s_) & dd["first"].str.startswith("ridge/")]; print(f"\n{s_}: dropping a group from set D (ridge), change in RMSE [CI excludes 0]; n={g.n_rows.iloc[0]} rows, {g.n_women.iloc[0]} women")
    for t in TARGETS: print(f"  {t:10s} " + "  ".join(f"{r.first.split('/')[1]}: {r.delta:+.3f}{'*' if r.excludes_0 else ''}" for r in g[g.target == t].itertuples()))

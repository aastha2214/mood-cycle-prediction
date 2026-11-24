"""Phase 4 metrics: MAE/RMSE/R2 from out-of-fold predictions with women-level bootstrap CIs and paired differences."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from src.data import SEED; from src.plotstyle import *
B = 2000; oof = pd.read_csv("results/phase4_oof.csv"); rng = np.random.default_rng(SEED); M, D = [], []
def stats(g):
    e = g.pred - g.y; grp = g.groupby("VPN")
    return pd.DataFrame(dict(n=grp.size(), sae=e.abs().groupby(g.VPN).sum(), sse=(e ** 2).groupby(g.VPN).sum(), sy=grp.y.sum(), sy2=(g.y ** 2).groupby(g.VPN).sum()))
def met(S, wt, base):
    N = wt @ S.n.values; sse = wt @ S.sse.values; sst = wt @ S.sy2.values - (wt @ S.sy.values) ** 2 / N
    return dict(MAE=wt @ S.sae.values / N, RMSE=np.sqrt(sse / N), R2=1 - sse / sst, R2_oos=1 - sse / (wt @ base.sse.values))
def ci(x): return np.percentile(x, [2.5, 97.5])
for (sample, target), blk in oof.groupby(["sample", "target"]):
    women = np.sort(blk.VPN.unique()); W = rng.multinomial(len(women), np.ones(len(women)) / len(women), size=B).astype(float); one = np.ones((1, len(women)))
    cfg = {(m, f): stats(g).loc[women] for (m, f), g in blk.groupby(["model", "fset"])}; base = cfg[("baseline", "-")]
    boot = {k: met(S, W, base) for k, S in cfg.items()}; pt = {k: met(S, one[0], base) for k, S in cfg.items()}
    for (m, f), S in cfg.items():
        r = dict(sample=sample, target=target, model=m, fset=f, n_rows=int(S.n.sum()), n_women=len(women))
        for k in ["MAE", "RMSE", "R2", "R2_oos"]:
            lo, hi = ci(boot[(m, f)][k]); r[k] = pt[(m, f)][k]; r[k + "_lo"], r[k + "_hi"] = lo, hi
        r["MAE_woman_mean"], r["MAE_woman_sd"] = (S.sae / S.n).mean(), (S.sae / S.n).std(); r["RMSE_woman_mean"], r["RMSE_woman_sd"] = np.sqrt(S.sse / S.n).mean(), np.sqrt(S.sse / S.n).std()
        M.append(r)
    pairs = [("A", "-"), ("B", "-"), ("C", "-"), ("D", "-"), ("D2", "-"), ("C", "A"), ("C", "B"), ("D", "C"), ("D2", "C"), ("D", "D2")]
    for m in ["ridge", "rf", "gb"]:
        for f1, f2 in pairs:
            k1 = (m, f1); k2 = ("baseline", "-") if f2 == "-" else (m, f2)
            for k in ["RMSE", "MAE"]:
                bd = boot[k1][k] - boot[k2][k]; lo, hi = ci(bd)
                D.append(dict(sample=sample, target=target, model=m, comparison=f"{f1} vs {'baseline' if f2=='-' else f2}", metric=k, n_rows=int(cfg[k1].n.sum()), n_women=len(women),
                              delta=pt[k1][k] - pt[k2][k], lo=lo, hi=hi, excludes_0=bool(lo > 0 or hi < 0)))
M = pd.DataFrame(M); D = pd.DataFrame(D); M.to_csv("results/phase4_metrics.csv", index=False); D.to_csv("results/phase4_deltas.csv", index=False)
pd.set_option("display.width", 220)
print("R2_oos = 1 - SSE_model/SSE_global-mean-baseline (out-of-fold, LOPO); point [95%% woman-bootstrap CI]; B=%d" % B)
for (s, t), g in M[M.model != "baseline"].groupby(["sample", "target"]):
    b = M[(M["sample"] == s) & (M.target == t) & (M.model == "baseline")].iloc[0]
    print(f"\n{s} | {t} | n={b.n_rows} rows, {b.n_women} women | baseline MAE {b.MAE:.3f} RMSE {b.RMSE:.3f} R2 {b.R2:.3f}")
    for f in ["A", "B", "C", "D2", "D"]:
        print(f"  {f:2s} " + "  ".join(f"{m}: {g[(g.model==m)&(g.fset==f)].R2_oos.iloc[0]:+.3f}[{g[(g.model==m)&(g.fset==f)].R2_oos_lo.iloc[0]:+.2f},{g[(g.model==m)&(g.fset==f)].R2_oos_hi.iloc[0]:+.2f}]" for m in ["ridge", "rf", "gb"]))
print("\nKey paired differences in RMSE (negative = first set better); * = 95% CI excludes 0")
k = D[(D.metric == "RMSE") & D.comparison.isin(["C vs A", "D vs D2"])]
for (s, t, c), g in k.groupby(["sample", "target", "comparison"]):
    print(f"  {s:10s}{t:10s}{c:8s} " + "  ".join(f"{r.model}: {r.delta:+.3f}[{r.lo:+.3f},{r.hi:+.3f}]{'*' if r.excludes_0 else ''}" for r in g.itertuples()))
# figure: skill vs baseline
fig, ax = plt.subplots(3, 3, figsize=(13, 9), sharex=True); cm = {"ridge": C["GD"], "rf": C["Momentum"], "gb": C["SGD"]}; sets = ["A", "B", "C", "D2", "D"]
for i, s in enumerate(["pooled", "long", "cross2long"]):
    for j, t in enumerate(["pos_affect", "drsp_psych", "stai"]):
        a = ax[i, j]
        for q, m in enumerate(["ridge", "rf", "gb"]):
            g = M[(M["sample"] == s) & (M.target == t) & (M.model == m)].set_index("fset").loc[sets]; x = np.arange(len(sets)) + (q - 1) * .22
            a.errorbar(x, g.R2_oos, yerr=[g.R2_oos - g.R2_oos_lo, g.R2_oos_hi - g.R2_oos], fmt="o", ms=4, color=cm[m], label=m, capsize=2, lw=1.2)
        a.axhline(0, color=C["ink"], lw=1); b = M[(M["sample"] == s) & (M.target == t)].iloc[0]; a.set_title(f"{s} | {t}\n{b.n_rows} rows, {b.n_women} test women", fontsize=9)
        a.set_xticks(range(len(sets))); a.set_xticklabels(sets)
        if j == 0: a.set_ylabel("R2 vs global-mean baseline")
ax[0, 0].legend(frameon=False, fontsize=8); fig.savefig("results/fig_4_skill.png"); plt.close(fig)

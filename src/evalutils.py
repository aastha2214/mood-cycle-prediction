"""Bootstrap (resampling WOMEN) metrics for out-of-fold regression predictions + paired differences with p-values and Holm/BH adjustment (same logic as phase4_report.py)."""
import numpy as np, pandas as pd
from statsmodels.stats.multitest import multipletests
from .data import SEED
def _stats(g):
    e = g.pred - g.y; return pd.DataFrame(dict(n=g.groupby("VPN").size(), sae=e.abs().groupby(g.VPN).sum(), sse=(e ** 2).groupby(g.VPN).sum(), sy=g.groupby("VPN").y.sum(), sy2=(g.y ** 2).groupby(g.VPN).sum()))
def _met(S, wt, base):
    N = wt @ S.n.values; sse = wt @ S.sse.values; sst = wt @ S.sy2.values - (wt @ S.sy.values) ** 2 / N
    return dict(MAE=wt @ S.sae.values / N, RMSE=np.sqrt(sse / N), R2=1 - sse / sst, R2_oos=1 - sse / (wt @ base.sse.values))
def boot(oof, B=2000, seed=SEED, pairs=(("C", "A"), ("B", "-"), ("A", "-"), ("D", "-"), ("D", "C"))):
    rng = np.random.default_rng(seed); M, D = [], []; ci = lambda x: np.percentile(x, [2.5, 97.5])
    for (sample, target), blk in oof.groupby(["sample", "target"]):
        women = np.sort(blk.VPN.unique()); W = rng.multinomial(len(women), np.ones(len(women)) / len(women), size=B).astype(float); one = np.ones(len(women))
        cfg = {k: _stats(g).loc[women] for k, g in blk.groupby(["model", "fset"])}; base = cfg[("baseline", "-")]
        bt = {k: _met(S, W, base) for k, S in cfg.items()}; pt = {k: _met(S, one, base) for k, S in cfg.items()}
        for (m, f), S in cfg.items():
            r = dict(sample=sample, target=target, model=m, fset=f, n_rows=int(S.n.sum()), n_women=len(women))
            for k in ["MAE", "RMSE", "R2", "R2_oos"]: r[k] = pt[(m, f)][k]; r[k + "_lo"], r[k + "_hi"] = ci(bt[(m, f)][k])
            M.append(r)
        for m in ["ridge", "rf", "gb"]:
            for f1, f2 in pairs:
                k1 = (m, f1); k2 = ("baseline", "-") if f2 == "-" else (m, f2); d = bt[k1]["RMSE"] - bt[k2]["RMSE"]; lo, hi = ci(d); a, b = (d <= 0).sum(), (d >= 0).sum()
                D.append(dict(sample=sample, target=target, model=m, comparison=f"{f1} vs {'baseline' if f2 == '-' else f2}", n_rows=int(cfg[k1].n.sum()), n_women=len(women), delta_RMSE=pt[k1]["RMSE"] - pt[k2]["RMSE"], lo=lo, hi=hi,
                              p_two_sided=min(1, 2 * (min(a, b) + 1) / (B + 1)), p_improve=(b + 1) / (B + 1)))
    M, D = pd.DataFrame(M), pd.DataFrame(D)
    for c, g in D.groupby("comparison"):
        D.loc[g.index, "p_holm"] = multipletests(g.p_two_sided, method="holm")[1]; D.loc[g.index, "p_fdr_bh"] = multipletests(g.p_two_sided, method="fdr_bh")[1]
    return M, D

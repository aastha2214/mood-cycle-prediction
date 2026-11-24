"""Shared bootstrap helpers for the Phase 7 extensions (women are resampled, never rows). Same machinery/seed as Phase 4."""
import numpy as np, pandas as pd
from .evalutils import _stats, _met
from .data import SEED
def _ci(x): return np.percentile(x, [2.5, 97.5])
def boot_configs(oof, comps, B=2000, seed=SEED):
    """oof: columns sample,target,model,fset,VPN,Session,y,pred (must contain baseline/'-'). comps: list of ((model,fset),(model,fset)); delta = first minus second (RMSE and MAE)."""
    rng = np.random.default_rng(seed); M, D = [], []
    for (sample, target), blk in oof.groupby(["sample", "target"]):
        women = np.sort(blk.VPN.unique()); W = rng.multinomial(len(women), np.ones(len(women)) / len(women), size=B).astype(float); one = np.ones(len(women))
        cfg = {k: _stats(g).loc[women] for k, g in blk.groupby(["model", "fset"])}; base = cfg[("baseline", "-")]
        bt = {k: _met(S, W, base) for k, S in cfg.items()}; pt = {k: _met(S, one, base) for k, S in cfg.items()}
        for (m, f), S in cfg.items():
            r = dict(sample=sample, target=target, model=m, fset=f, n_rows=int(S.n.sum()), n_women=len(women))
            for k in ["MAE", "RMSE", "R2", "R2_oos"]: r[k] = pt[(m, f)][k]; r[k + "_lo"], r[k + "_hi"] = _ci(bt[(m, f)][k])
            M.append(r)
        for k1, k2 in comps:
            if k1 in cfg and k2 in cfg:
                for met in ["RMSE", "MAE"]:
                    d = bt[k1][met] - bt[k2][met]; lo, hi = _ci(d)
                    D.append(dict(sample=sample, target=target, first=f"{k1[0]}/{k1[1]}", second=f"{k2[0]}/{k2[1]}", metric=met, n_rows=int(cfg[k1].n.sum()), n_women=len(women), delta=pt[k1][met] - pt[k2][met], lo=lo, hi=hi,
                                  excludes_0=bool(lo > 0 or hi < 0)))
    return pd.DataFrame(M), pd.DataFrame(D)
def boot_mean(df, col, B=2000, seed=SEED):
    """Row-level quantity averaged over rows; CI by resampling WOMEN (per-woman sums and counts). Returns (mean, lo, hi)."""
    g = df.groupby("VPN")[col].agg(["sum", "count"]); rng = np.random.default_rng(seed); W = rng.multinomial(len(g), np.ones(len(g)) / len(g), size=B).astype(float)
    b = (W @ g["sum"].values) / (W @ g["count"].values); lo, hi = _ci(b); return float(g["sum"].sum() / g["count"].sum()), float(lo), float(hi)

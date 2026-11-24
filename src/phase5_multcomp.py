"""Holm and Benjamini-Hochberg correction for the 27 paired 'do hormones help?' comparisons from Phase 4 (C vs A, RMSE), and a second family B vs baseline.
p-values come from the SAME woman-bootstrap as Phase 4 (same seed/order => identical resamples): two-sided p = 2*min(P(d*<=0), P(d*>=0)) with +1 smoothing."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from statsmodels.stats.multitest import multipletests
from src.data import SEED
B = 2000; oof = pd.read_csv("results/phase4_oof.csv"); rng = np.random.default_rng(SEED); rows = []   # one-sided p: H0 delta>=0 (no improvement) => p = P(d*>=0), small when bootstrap deltas are mostly negative
def stats_(g):
    e = g.pred - g.y; return (e ** 2).groupby(g.VPN).sum(), g.groupby("VPN").size()
for (sample, target), blk in oof.groupby(["sample", "target"]):
    women = np.sort(blk.VPN.unique()); W = rng.multinomial(len(women), np.ones(len(women)) / len(women), size=B).astype(float)   # same draw order as phase4_report.py
    S = {k: stats_(g) for k, g in blk.groupby(["model", "fset"])}
    rm = lambda k: np.sqrt((W @ S[k][0].loc[women].values) / (W @ S[k][1].loc[women].values))
    pt = lambda k: np.sqrt(S[k][0].loc[women].sum() / S[k][1].loc[women].sum())
    for m in ["ridge", "rf", "gb"]:
        for fam, k1, k2 in [("C vs A", (m, "C"), (m, "A")), ("B vs baseline", (m, "B"), ("baseline", "-"))]:
            d = rm(k1) - rm(k2); lo_cnt, hi_cnt = (d <= 0).sum(), (d >= 0).sum()
            rows.append(dict(family=fam, sample=sample, target=target, model=m, n_rows=int(S[k1][1].sum()), n_women=len(women), delta_RMSE=pt(k1) - pt(k2),
                             ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5), p_two_sided=min(1, 2 * (min(lo_cnt, hi_cnt) + 1) / (B + 1)), p_improve_one_sided=(hi_cnt + 1) / (B + 1)))
R = pd.DataFrame(rows)
for fam, g in R.groupby("family"):
    i = g.index
    R.loc[i, "p_holm"] = multipletests(g.p_two_sided, method="holm")[1]; R.loc[i, "p_fdr_bh"] = multipletests(g.p_two_sided, method="fdr_bh")[1]
    R.loc[i, "p_improve_holm"] = multipletests(g.p_improve_one_sided, method="holm")[1]; R.loc[i, "p_improve_fdr_bh"] = multipletests(g.p_improve_one_sided, method="fdr_bh")[1]
R.to_csv("results/phase5_multcomp.csv", index=False); pd.set_option("display.width", 220)
for fam, g in R.groupby("family"):
    print(f"\n{fam}: {len(g)} tests (3 samples x 3 targets x 3 models); delta RMSE < 0 means the first set is better")
    print(f"  raw two-sided p<0.05: {(g.p_two_sided<.05).sum()} | Holm<0.05: {(g.p_holm<.05).sum()} | BH-FDR<0.05: {(g.p_fdr_bh<.05).sum()}  [those are mostly cases where the first set is WORSE]")
    print(f"  one-sided 'improves' raw p<0.05: {(g.p_improve_one_sided<.05).sum()} | Holm<0.05: {(g.p_improve_holm<.05).sum()} | BH-FDR<0.05: {(g.p_improve_fdr_bh<.05).sum()}; smallest improvement p raw {g.p_improve_one_sided.min():.3f}, Holm {g.p_improve_holm.min():.3f}")
    print(f"  direction: delta RMSE > 0 (first set worse) in {(g.delta_RMSE>0).sum()}/{len(g)}")
g = R[R.family == "C vs A"]; print("\nC vs A, rows with raw two-sided p<0.05 (worse = hormones hurt):"); print(g[g.p_two_sided < .05][["sample", "target", "model", "delta_RMSE", "p_two_sided", "p_holm", "p_fdr_bh"]].round(4).to_string(index=False))

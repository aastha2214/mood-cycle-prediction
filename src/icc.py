"""ICC(1) of each target: ANOVA estimator (unbalanced, n0 adjusted) + REML random-intercept model + woman-bootstrap CI."""
import sys, warnings; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from src.data import load_analysis, subset, TARGETS, SEED
warnings.filterwarnings("ignore")

def icc_anova(y, g):
    df = pd.DataFrame({"y": y, "g": g}); s = df.groupby("g").y.agg(["size", "mean"])
    N, k = len(df), len(s); gm = df.y.mean()
    ssb = (s["size"] * (s["mean"] - gm) ** 2).sum(); ssw = ((df.y - df.g.map(s["mean"])) ** 2).sum()
    msb, msw = ssb / (k - 1), ssw / (N - k); n0 = (N - (s["size"] ** 2).sum() / N) / (k - 1)
    return (msb - msw) / (msb + (n0 - 1) * msw)

def icc_reml(y, g):
    d = pd.DataFrame({"y": y, "g": g}); m = smf.mixedlm("y~1", d, groups=d.g).fit(reml=True)
    vu = float(m.cov_re.iloc[0, 0]); return vu / (vu + m.scale)

if __name__ == "__main__":
    df = load_analysis(); rng = np.random.default_rng(SEED); rows = []
    for which in ["pooled", "long"]:
        d = subset(df, which); W = d.VPN.unique()
        for nm, col in TARGETS.items():
            a = icc_anova(d[col].values, d.VPN.values); r = icc_reml(d[col].values, d.VPN.values)
            bs = []
            for _ in range(500):
                pick = rng.choice(W, len(W)); parts = [d[d.VPN == w].assign(g=i) for i, w in enumerate(pick)]
                b = pd.concat(parts); bs.append(icc_anova(b[col].values, b.g.values))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append(dict(sample=which, target=nm, n_rows=len(d), n_women=len(W), ICC_anova=a, ICC_reml=r, ci_lo=lo, ci_hi=hi))
    out = pd.DataFrame(rows).round(3); print(out.to_string(index=False)); out.to_csv("results/icc.csv", index=False)

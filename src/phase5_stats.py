"""Phase 5: mixed-effects models (random intercept per woman, REML) + permutation tests. Outcomes/predictors z-scored within each sample (like the authors' scale())."""
import sys, warnings; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from joblib import Parallel, delayed
from scipy import stats
from src.data import *
warnings.filterwarnings("ignore"); rng = np.random.default_rng(SEED); BP = 5000
OUT = {"pos_affect": "Affect_pos", "neg_affect": "Affect_neg", "drsp_psych": "DRSP_psychisch", "drsp_total": "DRSP", "stai": "STAI"}
df = load_analysis(); MIX, PERM = [], []
def prep(which):
    d = subset(df, which).copy()
    for c in list(OUT.values()) + ["estradiol", "progesterone", "PMS", "BDI_RW", "BAI_RW", "logE2", "logP"]: d["z_" + c] = (d[c] - d[c].mean()) / d[c].std(ddof=1)
    for c in ["estradiol", "progesterone", "logE2"]:
        m = d.groupby("VPN")["z_" + c].transform("mean"); d["zb_" + c] = m; d["zw_" + c] = d["z_" + c] - m
    return d
def fit(d, y, rhs):
    return smf.mixedlm(f"z_{OUT[y]} ~ {rhs}", d, groups=d["VPN"]).fit(reml=True)
def add(sample, d, y, model, res, terms, extra=None):
    n, nw = len(d), d.VPN.nunique(); sd = d[OUT[y]].std(ddof=1); fe = res.fe_params; se = res.bse_fe; vu = float(res.cov_re.iloc[0, 0]); ve = res.scale
    for t in terms:
        MIX.append(dict(sample=sample, n_rows=n, n_women=nw, outcome=y, model=model, term=t, b_z=fe[t], se=se[t], lo=fe[t] - 1.96 * se[t], hi=fe[t] + 1.96 * se[t], p=res.pvalues[t], b_raw=fe[t] * sd, icc_resid=vu / (vu + ve)))
    return res
def joint(res, names):
    b = res.fe_params[names].values; S = res.cov_params().loc[names, names].values; x = float(b @ np.linalg.solve(S, b)); return x, stats.chi2.sf(x, len(names))
def contrast(res, a, b):
    c = res.fe_params[a] - res.fe_params[b]; S = res.cov_params(); v = S.loc[a, a] + S.loc[b, b] - 2 * S.loc[a, b]; se = np.sqrt(v); return c, se, 2 * stats.norm.sf(abs(c / se))
def r2(res):
    vf = np.var(res.model.exog @ res.fe_params.values);  # fixed-effects part only (fittedvalues would include random effects)
    vu = float(res.cov_re.iloc[0, 0]); ve = res.scale; return vf / (vf + vu + ve), (vf + vu) / (vf + vu + ve)
R2 = []
for which in ["long", "pooled"]:
    d = prep(which); n, nw = len(d), d.VPN.nunique()
    for y in OUT:
        res = add(which, d, y, "phase", fit(d, y, "ph1 + ph2"), ["ph1", "ph2"]); c, se, p = contrast(res, "ph2", "ph1"); x, pj = joint(res, ["ph1", "ph2"]); m, cnd = r2(res)
        MIX.append(dict(sample=which, n_rows=n, n_women=nw, outcome=y, model="phase", term="ph2-ph1", b_z=c, se=se, lo=c - 1.96 * se, hi=c + 1.96 * se, p=p, b_raw=c * d[OUT[y]].std(ddof=1), icc_resid=np.nan))
        MIX.append(dict(sample=which, n_rows=n, n_women=nw, outcome=y, model="phase", term="phase joint (chi2 df=2)", b_z=x, se=np.nan, lo=np.nan, hi=np.nan, p=pj, b_raw=np.nan, icc_resid=np.nan))
        R2.append(dict(sample=which, n_rows=n, n_women=nw, outcome=y, marginal_R2_phase=m, conditional_R2=cnd, wald_p_phase=pj))
        add(which, d, y, "E2*P", fit(d, y, "z_estradiol * z_progesterone"), ["z_estradiol", "z_progesterone", "z_estradiol:z_progesterone"])
    for y in ["drsp_psych", "drsp_total"]:
        add(which, d, y, "E2 within/between", fit(d, y, "zw_estradiol + zb_estradiol"), ["zw_estradiol", "zb_estradiol"])
        add(which, d, y, "E2+P within/between", fit(d, y, "zw_estradiol + zb_estradiol + zw_progesterone + zb_progesterone"), ["zw_estradiol", "zb_estradiol", "zw_progesterone", "zb_progesterone"])
        add(which, d, y, "log1p E2 within/between", fit(d, y, "zw_logE2 + zb_logE2"), ["zw_logE2", "zb_logE2"])
        res = add(which, d, y, "phase x PSST", fit(d, y, "(ph1 + ph2) * z_PMS"), ["z_PMS", "ph1:z_PMS", "ph2:z_PMS"]); x, pj = joint(res, ["ph1:z_PMS", "ph2:z_PMS"])
        MIX.append(dict(sample=which, n_rows=n, n_women=nw, outcome=y, model="phase x PSST", term="interaction joint (chi2 df=2)", b_z=x, se=np.nan, lo=np.nan, hi=np.nan, p=pj, b_raw=np.nan, icc_resid=np.nan))
    for y in ["pos_affect", "neg_affect"]:
        add(which, d, y, "phase x BDI", fit(d, y, "(ph1 + ph2) * z_BDI_RW"), ["z_BDI_RW"])
        add(which, d, y, "BDI (main effect only)", fit(d, y, "z_BDI_RW"), ["z_BDI_RW"])
# ---------- permutation tests ----------
def perms(d, B):  # restricted permutations: within woman for repeat-measured women; among single-session women for the rest
    g = d.VPN.values; idx = np.arange(len(g)); P = np.tile(idx, (B, 1)); cnt = pd.Series(g).map(pd.Series(g).value_counts()).values
    for w in np.unique(g[cnt > 1]):
        ii = np.where(g == w)[0]
        for b in range(B): P[b, ii] = rng.permutation(ii)
    si = np.where(cnt == 1)[0]
    if len(si) > 1:
        for b in range(B): P[b, si] = rng.permutation(si)
    return P
for which in ["long", "pooled"]:
    d = prep(which); N = len(d); g = d.VPN.values; nw = d.VPN.nunique(); Dm = d[["ph1", "ph2"]].values
    Gp = (g[:, None] == g[None, :]).astype(float); Pw = Gp / Gp.sum(1, keepdims=True)         # projector on woman means
    Pu = np.full((N, N), 1 / N)
    proj = Pw if which == "long" else Pu                                                      # long: within-woman (fixed-effect) F; pooled: ordinary F, stratified permutations
    PM = perms(d, BP); dfres = N - (nw if which == "long" else 1) - 2
    def phaseF(Dp, yv):
        Dt = Dp - proj @ Dp; yt = yv - proj @ yv; XtX = np.einsum("...ni,...nj->...ij", Dt, Dt); Xty = np.einsum("...ni,n->...i", Dt, yt)
        beta = np.linalg.solve(XtX, Xty[..., None])[..., 0]; rss1 = yt @ yt - np.einsum("...i,...i->...", beta, Xty); return (yt @ yt - rss1) / 2 / (rss1 / dfres)
    for y in OUT:
        yv = d["z_" + OUT[y]].values; obs = phaseF(Dm, yv); nullF = phaseF(Dm[PM], yv); PERM.append(dict(sample=which, n_rows=N, n_women=nw, outcome=y, test="phase omnibus F (%s)" % ("within-woman" if which == "long" else "ordinary F, stratified perms"), stat=obs, perm_p=(1 + (nullF >= obs).sum()) / (BP + 1), B=BP))
    if which == "pooled":   # permutation test with the same statistic as the mixed model (Wald chi2 of the 2 phase terms), B=1000, stratified permutations
        B2 = 1000; Dm_ = Dm.copy()
        def wald_all(Dp):
            dd = d.copy(); dd[["ph1", "ph2"]] = Dp; return [joint(fit(dd, y, "ph1 + ph2"), ["ph1", "ph2"])[0] for y in OUT]
        obs = wald_all(Dm_); nul = np.array(Parallel(n_jobs=10)(delayed(wald_all)(Dm_[PM[b]]) for b in range(B2)))
        for k, y in enumerate(OUT): PERM.append(dict(sample=which, n_rows=N, n_women=nw, outcome=y, test="phase Wald chi2 of the mixed model, stratified perms", stat=obs[k], perm_p=(1 + (nul[:, k] >= obs[k]).sum()) / (B2 + 1), B=B2))
    # estradiol within-woman slope (permute E2 within woman; only repeat-measured women carry within information, so pooled == long)
    Wm = Gp / Gp.sum(1, keepdims=True); x = d["z_estradiol"].values; xt = x - Wm @ x
    for y in OUT:
        yv = d["z_" + OUT[y]].values; yt = yv - Wm @ yv; obs = (xt @ yt) / (xt @ xt); XP = x[PM]; XPt = XP - XP @ Wm.T; nul = (XPt @ yt) / (XPt * XPt).sum(1)
        PERM.append(dict(sample=which, n_rows=N, n_women=nw, outcome=y, test="E2 within-woman slope (|slope|)", stat=obs, perm_p=(1 + (np.abs(nul) >= abs(obs)).sum()) / (BP + 1), B=BP))
    # estradiol between-woman slope on woman means (permute woman-mean E2 across women)
    wm = d.groupby("VPN")[["z_estradiol"] + ["z_" + v for v in OUT.values()]].mean(); xb = wm["z_estradiol"].values
    for y in OUT:
        yb = wm["z_" + OUT[y]].values; sl = lambda xx, yy: ((xx - xx.mean()) @ (yy - yy.mean())) / ((xx - xx.mean()) @ (xx - xx.mean())); obs = sl(xb, yb)
        nul = np.array([sl(rng.permutation(xb), yb) for _ in range(BP)])
        PERM.append(dict(sample=which, n_rows=N, n_women=nw, outcome=y, test="E2 between-woman slope on woman means (|slope|)", stat=obs, perm_p=(1 + (np.abs(nul) >= abs(obs)).sum()) / (BP + 1), B=BP))
Mx = pd.DataFrame(MIX); Pm = pd.DataFrame(PERM); Rr = pd.DataFrame(R2)
Mx.to_csv("results/phase5_mixed.csv", index=False); Pm.to_csv("results/phase5_perm.csv", index=False); Rr.to_csv("results/phase5_r2_phase.csv", index=False)
# descriptive hormones by phase (longitudinal)
L = subset(df, "long"); desc = L.groupby("Phase").agg(n=("VPN", "size"), women=("VPN", "nunique"), E2_mean=("estradiol", "mean"), E2_median=("estradiol", "median"), P_mean=("progesterone", "mean"), P_median=("progesterone", "median")); desc.to_csv("results/phase5_hormones_by_phase_long.csv")
print("Mixed models fitted:", len(Mx), "rows;", "permutation tests:", len(Pm), "; B =", BP); print(desc.round(2).to_string())

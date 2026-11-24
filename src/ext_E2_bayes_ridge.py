"""E2: Bayesian ridge FROM SCRATCH (Gaussian prior on weights, evidence maximisation of the prior precision alpha and noise precision beta inside every training fold),
compared with the Phase 4 ridge under the same leave-one-participant-out protocol; calibration of the predictive variance. Intercept = training mean (unpenalised), features standardised on training rows only."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import BayesianRidge
from src.data import *; from src.mathcore.ridge import standardize_fit; from src.extlib import *
df = load_analysis(); SETS = ["A", "B", "C", "D2", "D"]; ALPHA_MAX = 1e12
def bayes_ridge_fit(Zc, yc, max_iter=20000, tol=1e-10):
    """Posterior N(m, S) for y = Z w + noise, prior w ~ N(0, alpha^-1 I), noise precision beta. MacKay fixed-point updates (Bishop PRML 3.92-3.95):
    gamma = sum_i beta s_i^2/(alpha + beta s_i^2); alpha <- gamma/(m'm); beta <- (N - gamma)/||y - Z m||^2. Returns the fitted posterior and the log evidence."""
    N, p = Zc.shape; U, s, Vt = np.linalg.svd(Zc, full_matrices=False); Uty = U.T @ yc; alpha, beta = 1.0, 1.0 / yc.var(); conv = False
    for it in range(1, max_iter + 1):
        mt = beta * s * Uty / (alpha + beta * s ** 2); gamma = np.sum(beta * s ** 2 / (alpha + beta * s ** 2)); r = yc - U @ (s * mt)
        a_new = min(gamma / max(mt @ mt, 1e-300), ALPHA_MAX); b_new = (N - gamma) / (r @ r)
        done = abs(a_new - alpha) <= tol * alpha and abs(b_new - beta) <= tol * beta; alpha, beta = a_new, b_new
        if done: conv = True; break
    mt = beta * s * Uty / (alpha + beta * s ** 2); m = Vt.T @ mt; r = yc - Zc @ m; Sdiag = 1.0 / (alpha + beta * s ** 2); S = (Vt.T * Sdiag) @ Vt
    logev = p / 2 * np.log(alpha) + N / 2 * np.log(beta) - (beta / 2 * r @ r + alpha / 2 * m @ m) - 0.5 * np.sum(np.log(alpha + beta * s ** 2)) - N / 2 * np.log(2 * np.pi)
    return dict(m=m, S=S, alpha=alpha, beta=beta, gamma=float(np.sum(beta * s ** 2 / (alpha + beta * s ** 2))), n_iter=it, converged=conv, log_evidence=float(logev))
def fit_predict(Xtr, ytr, Xte):
    mu, sd = standardize_fit(Xtr); Z = (Xtr - mu) / sd; zm = Z.mean(0); Zc = Z - zm; ym = ytr.mean(); fit = bayes_ridge_fit(Zc, ytr - ym)
    Zt = (Xte - mu) / sd - zm; mean = Zt @ fit["m"] + ym; var = 1.0 / fit["beta"] + np.einsum("ij,jk,ik->i", Zt, fit["S"], Zt)
    return mean, np.sqrt(var), fit, ytr.mean(), ytr.std(ddof=1)
def run_cfg(sample, target, fs):
    tc = TARGETS[target]; cols = FEATURE_SETS[fs]; rows = []
    if sample == "cross2long": tr, te = df[df["sample"] == "cross"], df[df["sample"] == "long"]; folds = [(tr, te)]
    else:
        d = subset(df, sample); folds = []
        for w in np.unique(d.VPN.values):
            tr, te = d[d.VPN != w], d[d.VPN == w]; assert set(tr.VPN).isdisjoint(te.VPN); folds.append((tr, te))   # group integrity
    assert tc not in cols
    for tr, te in folds:
        mean, sd, fit, bm, bs = fit_predict(tr[cols].values, tr[tc].values, te[cols].values)
        rows.append(pd.DataFrame(dict(sample=sample, target=target, model="bayes_ridge", fset=fs, VPN=te.VPN.values, Session=te.Session.values, y=te[tc].values, pred=mean, sd=sd, base_mu=bm, base_sd=bs, alpha=fit["alpha"], beta=fit["beta"],
                                      gamma=fit["gamma"], n_iter=fit["n_iter"], converged=fit["converged"], n_train=len(tr), lam_equiv=fit["alpha"] / (fit["beta"] * len(tr)))))
    return pd.concat(rows)
jobs = [(s, t, f) for s in ["pooled", "long", "cross2long"] for t in TARGETS for f in SETS]
oof = pd.concat(Parallel(n_jobs=10)(delayed(run_cfg)(*j) for j in jobs)); oof.to_csv("results/extE2_oof.csv", index=False)
# ---- check against scikit-learn BayesianRidge (pure evidence maximisation: all Gamma hyper-priors set to 0), full samples
chk = []
for s in ["pooled", "long"]:
    d = subset(df, s)
    for t, tc in TARGETS.items():
        for fs in SETS:
            X, y = d[FEATURE_SETS[fs]].values, d[tc].values; mu, sd = standardize_fit(X); Z = (X - mu) / sd; Zc = Z - Z.mean(0); fit = bayes_ridge_fit(Zc, y - y.mean())
            sk = BayesianRidge(max_iter=100000, tol=1e-13, alpha_1=0, alpha_2=0, lambda_1=0, lambda_2=0, fit_intercept=True).fit(Z, y)
            chk.append(dict(sample=s, n_rows=len(y), n_women=d.VPN.nunique(), target=t, fset=fs, max_abs_diff_weights=np.abs(fit["m"] - sk.coef_).max(), rel_diff_alpha=abs(fit["alpha"] - sk.lambda_) / sk.lambda_, rel_diff_beta=abs(fit["beta"] - sk.alpha_) / sk.alpha_,
                            sklearn_converged_iters=int(sk.n_iter_), mine_iters=fit["n_iter"], eff_params_gamma=fit["gamma"], alpha_finite=bool(fit["alpha"] < 1e6)))
pd.DataFrame(chk).to_csv("results/extE2_sklearn_check.csv", index=False)
# ---- compare with ridge (Phase 4 out-of-fold predictions, identical protocol) and baseline
p4 = pd.read_csv("results/phase4_oof.csv"); p4 = p4[p4.model.isin(["baseline", "ridge"])]
bayes = oof[["sample", "target", "model", "fset", "VPN", "Session", "y", "pred"]]
M, D = boot_configs(pd.concat([p4[["sample", "target", "model", "fset", "VPN", "Session", "y", "pred"]], bayes]), [(("bayes_ridge", f), ("ridge", f)) for f in SETS])
M.to_csv("results/extE2_metrics_all.csv", index=False); D.to_csv("results/extE2_vs_ridge.csv", index=False)
# ---- calibration of the predictive variance: standardised squared error (target 1) and log predictive density versus a constant-variance Gaussian baseline
oof["z2"] = ((oof.y - oof.pred) / oof.sd) ** 2; oof["nlpd"] = 0.5 * np.log(2 * np.pi * oof.sd ** 2) + 0.5 * oof.z2
oof["nlpd_base"] = 0.5 * np.log(2 * np.pi * oof.base_sd ** 2) + 0.5 * ((oof.y - oof.base_mu) / oof.base_sd) ** 2; oof["d_nlpd"] = oof.nlpd - oof.nlpd_base
rows = []
for (s, t, f), g in oof.groupby(["sample", "target", "fset"]):
    z, zl, zh = boot_mean(g, "z2"); n, nl, nh = boot_mean(g, "nlpd"); dn, dl, dh = boot_mean(g, "d_nlpd"); ridge_hp = p4[(p4["sample"] == s) & (p4.target == t) & (p4.model == "ridge") & (p4.fset == f)].hp.astype(float)
    rows.append(dict(sample=s, target=t, fset=f, n_rows=len(g), n_women=g.VPN.nunique(), mean_z2=z, mean_z2_lo=zl, mean_z2_hi=zh, nlpd=n, nlpd_lo=nl, nlpd_hi=nh, d_nlpd_vs_const_gauss=dn, d_nlpd_lo=dl, d_nlpd_hi=dh,
                     sd_pred_min=g.sd.min(), sd_pred_max=g.sd.max(), rmse_equiv_sd_const=g.base_sd.median(), median_lambda_equiv=g.lam_equiv.median(), median_lambda_ridge_cv=ridge_hp.median(), median_eff_params=g.gamma.median(),
                     median_iters=g.n_iter.median(), n_nonconverged_folds=int((~g.converged).sum())))
C = pd.DataFrame(rows); C.to_csv("results/extE2_calibration.csv", index=False)
# ---- concise console summary
sk = pd.DataFrame(chk); fin = sk[sk.alpha_finite]; print(f"E2 Bayesian ridge: {len(oof)} out-of-fold rows over {len(jobs)} configurations. sklearn parity ({len(sk)} full-sample configs): max |dw| {sk.max_abs_diff_weights.max():.1e}, max rel diff beta {sk.rel_diff_beta.max():.1e}; max rel diff alpha {fin.rel_diff_alpha.max():.1e} over the {len(fin)} configs with finite alpha ({len(sk)-len(fin)} have alpha -> infinity: weights shrunk to zero, alpha not identifiable); non-converged folds: {int(C.n_nonconverged_folds.sum())}")
mm = M[M.model == "bayes_ridge"]
for (s,), g in mm.groupby(["sample"]):
    print(f"\n{s} (n={g.n_rows.iloc[0]} rows, {g.n_women.iloc[0]} women): R2_oos Bayes ridge by set [DRSP / pos.affect / anxiety]")
    for f in SETS: print(f"  {f:2s} " + " / ".join(f"{g[(g.target==t)&(g.fset==f)].R2_oos.iloc[0]:+.3f}" for t in ["drsp_psych", "pos_affect", "stai"]))
dd = D[D.metric == "RMSE"]; print(f"\nBayes ridge minus ridge, RMSE: |delta| max {dd.delta.abs().max():.4f}; CI excludes 0 in {int(dd.excludes_0.sum())} of {len(dd)} comparisons")
print(f"Calibration: mean z^2 (target 1) ranges {C.mean_z2.min():.2f}-{C.mean_z2.max():.2f}; predictive sd varies little (max/min per config up to {(C.sd_pred_max/C.sd_pred_min).max():.2f}).")

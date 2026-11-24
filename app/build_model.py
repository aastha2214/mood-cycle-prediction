"""Builds app/model_artifacts.json: the final model of the E6 demo, trained on ALL 160 women (271 sessions) with the validated feature set D.
Run once from the project root:  ./.venv/bin/python -I app/build_model.py   (reads data through src/data.py, writes only app/model_artifacts.json; touches no existing file).
Ridge: lambda chosen by the same inner cross-validation grouped by woman as in the nested CV (src/cvlib.py), solution by SVD (src/mathcore/ridge.py), features log1p-transformed and standardised with the training statistics.
Bayesian ridge (E2): Gaussian prior, evidence maximisation of alpha and beta (copy of the function in src/ext_E2_bayes_ridge.py, checked here against scikit-learn)."""
import sys, json; from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
import numpy as np
from scipy import stats
from sklearn.linear_model import BayesianRidge
from src.data import load_analysis, FEATURE_SETS, TARGETS, SEED
from src.cvlib import predict_one
from src.mathcore.ridge import standardize_fit, ridge_svd, fit_predict_path
def bayes_ridge_fit(Zc, yc, max_iter=20000, tol=1e-10, alpha_max=1e12):   # same algorithm as E2 (MacKay fixed point)
    N, p = Zc.shape; U, s, Vt = np.linalg.svd(Zc, full_matrices=False); Uty = U.T @ yc; alpha, beta = 1.0, 1.0 / yc.var()
    for it in range(1, max_iter + 1):
        mt = beta * s * Uty / (alpha + beta * s ** 2); gamma = np.sum(beta * s ** 2 / (alpha + beta * s ** 2)); r = yc - U @ (s * mt)
        a_new = min(gamma / max(mt @ mt, 1e-300), alpha_max); b_new = (N - gamma) / (r @ r); done = abs(a_new - alpha) <= tol * alpha and abs(b_new - beta) <= tol * beta; alpha, beta = a_new, b_new
        if done: break
    mt = beta * s * Uty / (alpha + beta * s ** 2); return Vt.T @ mt, (Vt.T * (1.0 / (alpha + beta * s ** 2))) @ Vt, alpha, beta
d = load_analysis(); cols = FEATURE_SETS["D"]; assert cols == ["ph1", "ph2", "logE2", "logP", "PMS", "BDI_RW", "BAI_RW"]
X, g = d[cols].values, d.VPN.values; mu, sd = standardize_fit(X); Z = (X - mu) / sd; zm = Z.mean(0); Zc = Z - zm
LEVEL = 0.90; art = dict(n_rows=len(d), n_women=int(d.VPN.nunique()), features=cols, mu=mu.tolist(), sd=sd.tolist(), zm=zm.tolist(), level=LEVEL, z_level=float(stats.norm.ppf(0.5 + LEVEL / 2)),
                         inputs={"estradiol": dict(min=float(d.estradiol.min()), max=float(d.estradiol.max()), median=float(d.estradiol.median())), "progesterone": dict(min=float(d.progesterone.min()), max=float(d.progesterone.max()), median=float(d.progesterone.median())),
                                 "PMS": dict(min=float(d.PMS.min()), max=float(d.PMS.max()), median=float(d.PMS.median())), "BDI_RW": dict(min=float(d.BDI_RW.min()), max=float(d.BDI_RW.max()), median=float(d.BDI_RW.median())), "BAI_RW": dict(min=float(d.BAI_RW.min()), max=float(d.BAI_RW.max()), median=float(d.BAI_RW.median()))},
                         targets={})
for t, tc in TARGETS.items():
    y = d[tc].values; ym = float(y.mean()); _, hp = predict_one("ridge", X, y, g, X[:1], SEED); lam = float(hp); w = ridge_svd(Zc, y - ym, lam)
    chk = fit_predict_path(X, y, X, np.array([lam]))[0]; assert np.allclose(chk, Zc @ w + ym, atol=1e-9), "ridge solution differs from the pipeline used in the validation"
    m, S, alpha, beta = bayes_ridge_fit(Zc, y - ym); sk = BayesianRidge(max_iter=100000, tol=1e-13, alpha_1=0, alpha_2=0, lambda_1=0, lambda_2=0).fit(Z, y); assert np.abs(m - sk.coef_).max() < 1e-6 and abs(beta - sk.alpha_) / beta < 1e-6, "Bayesian ridge differs from scikit-learn"
    art["targets"][t] = dict(column=tc, ybar=ym, lam=lam, w=w.tolist(), m=m.tolist(), S=S.tolist(), alpha=float(alpha), beta=float(beta), obs_min=float(y.min()), obs_max=float(y.max()), obs_sd=float(y.std(ddof=1)))
(ROOT / "app/model_artifacts.json").write_text(json.dumps(art, indent=1)); print(f"app/model_artifacts.json written: model trained on {art['n_rows']} sessions of {art['n_women']} women; lambdas " + ", ".join(f"{t}={v['lam']:.3g}" for t, v in art["targets"].items()))

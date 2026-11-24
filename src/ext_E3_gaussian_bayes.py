"""E3: Gaussian Bayes classifier FROM SCRATCH for cycle phase (0 menses, 1 peri-ovulatory, 2 luteal) from log1p estradiol and log1p progesterone.
Class priors and class-conditional Gaussians are estimated in the training fold; Bayes' theorem gives the posterior; the label is the posterior mode. Variants: full class covariances (QDA, primary), shared covariance (LDA), diagonal (naive Bayes);
empirical priors (primary) or equal priors. No hyper-parameters, hence no inner CV. Compared with the Phase 6b logistic regression out-of-fold predictions (same folds, same protocol)."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from src.data import *; from src.extlib import _ci
df = load_analysis(); cols = FEATURE_SETS["B"]; NAMES = ["menses", "peri-ov", "luteal"]; VARIANTS = [("full", "empirical"), ("full", "equal"), ("shared", "empirical"), ("shared", "equal"), ("diag", "empirical"), ("diag", "equal")]; B = 2000
def gnb_fit(X, y, cov):
    K = 3; pri = np.array([(y == k).mean() for k in range(K)]); mu = np.array([X[y == k].mean(0) for k in range(K)]); S = [np.cov(X[y == k].T, ddof=0) for k in range(K)]   # maximum-likelihood covariances (divide by n_k), as scikit-learn's QDA does
    if cov == "shared": Sp = sum(np.sum(y == k) * S[k] for k in range(K)) / len(y); S = [Sp] * K
    if cov == "diag": S = [np.diag(np.diag(s)) for s in S]
    return dict(prior=pri, mu=mu, S=S)
def gnb_posterior(m, X, priors):
    """Bayes: p(k|x) = pi_k N(x; mu_k, S_k) / sum_j pi_j N(x; mu_j, S_j), computed in log space."""
    lp = []
    for k in range(3):
        d = X - m["mu"][k]; Si = np.linalg.inv(m["S"][k]); _, ld = np.linalg.slogdet(m["S"][k]); lp.append(np.log(priors[k]) - 0.5 * (np.einsum("ij,jk,ik->i", d, Si, d) + ld + X.shape[1] * np.log(2 * np.pi)))
    lp = np.array(lp).T; lp -= lp.max(1, keepdims=True); p = np.exp(lp); return p / p.sum(1, keepdims=True)
def predict(trX, trY, teX, cov, pri_mode):
    m = gnb_fit(trX, trY, cov); pri = m["prior"] if pri_mode == "empirical" else np.ones(3) / 3; return gnb_posterior(m, teX, pri), m["prior"]
rows = []
def add(sample, name, te, P, base_prior):
    for i in range(len(te)): rows.append(dict(sample=sample, model=name, VPN=te.VPN.values[i], Session=te.Session.values[i], y=int(te.Phase.values[i]), pred=int(P[i].argmax()), p0=P[i, 0], p1=P[i, 1], p2=P[i, 2], prior_ll=base_prior[i]))
for sample in ["pooled", "long", "cross2long"]:
    if sample == "cross2long": tr, te = df[df["sample"] == "cross"], df[df["sample"] == "long"]; folds = [(tr, te)]
    else:
        d = subset(df, sample); folds = []
        for w in np.unique(d.VPN.values):
            tr, te = d[d.VPN != w], d[d.VPN == w]; assert set(tr.VPN).isdisjoint(te.VPN); folds.append((tr, te))
    for cov, pm in VARIANTS:
        for tr, te in folds:
            P, prior = predict(tr[cols].values, tr.Phase.values.astype(int), te[cols].values, cov, pm); y = te.Phase.values.astype(int)
            add(sample, f"gauss_bayes_{cov}_{pm}", te, P, np.log(prior[y]))
oof = pd.DataFrame(rows); oof.to_csv("results/extE3_oof.csv", index=False)
# ---- parity with scikit-learn QDA (full covariance, empirical priors) on the full pooled sample (posterior probabilities)
d = subset(df, "pooled"); X, y = d[cols].values, d.Phase.values.astype(int); mine, _ = predict(X, y, X, "full", "empirical"); sk = QuadraticDiscriminantAnalysis(reg_param=0.0).fit(X, y).predict_proba(X)
pd.DataFrame([dict(sample="pooled", n_rows=len(y), n_women=d.VPN.nunique(), max_abs_posterior_diff=float(np.abs(mine - sk).max()), same_label_rate=float((mine.argmax(1) == sk.argmax(1)).mean()))]).to_csv("results/extE3_sklearn_check.csv", index=False)
# ---- fitted parameters (descriptive, full pooled sample)
m = gnb_fit(X, y, "full"); par = []
for k in range(3):
    par.append(dict(sample="pooled", n_rows=len(y), class_=NAMES[k], prior=m["prior"][k], mean_logE2=m["mu"][k][0], mean_logP=m["mu"][k][1], sd_logE2=np.sqrt(m["S"][k][0, 0]), sd_logP=np.sqrt(m["S"][k][1, 1]), corr=m["S"][k][0, 1] / np.sqrt(m["S"][k][0, 0] * m["S"][k][1, 1])))
pd.DataFrame(par).to_csv("results/extE3_params.csv", index=False)
# ---- metrics: balanced accuracy, accuracy, log loss with woman-bootstrap CIs; paired difference to logistic regression (Phase 6b out-of-fold predictions)
lr = pd.read_csv("results/phase6b_oof.csv"); lr = lr[lr.model == "logreg"].assign(p0=np.nan, p1=np.nan, p2=np.nan, prior_ll=np.nan)[oof.columns.tolist()]; lr["model"] = "logreg"; allp = pd.concat([oof, lr])
def cm_by_woman(g, women): return np.array([[[((g.y == i) & (g.pred == j) & (g.VPN == w)).sum() for j in range(3)] for i in range(3)] for w in women])
M, CMr = [], []; rng = np.random.default_rng(SEED)
for sample, blk in allp.groupby("sample"):
    women = np.sort(blk.VPN.unique()); W = rng.multinomial(len(women), np.ones(len(women)) / len(women), size=B).astype(float); one = np.ones(len(women)); bs = {}; pt = {}
    for mod, g in blk.groupby("model"):
        c = cm_by_woman(g, women); tot = lambda wt: np.tensordot(wt, c, axes=1); rec = lambda t: np.diag(t) / np.maximum(t.sum(1), 1)
        pt[mod] = tot(one); bs[mod] = np.array([rec(tot(W[b])).mean() for b in range(B)]); t = pt[mod]; r = rec(t)
        ll = None
        if not g.p0.isna().any():
            pcorr = g[["p0", "p1", "p2"]].values[np.arange(len(g)), g.y.values]; g = g.assign(ll=-np.log(np.maximum(pcorr, 1e-12)), ll_prior=-g.prior_ll); gw = g.groupby("VPN")[["ll", "ll_prior"]].sum().loc[women]; cnt = g.groupby("VPN").size().loc[women].values
            ll = (W @ gw.ll.values) / (W @ cnt); llp = (W @ gw.ll_prior.values) / (W @ cnt); ll_pt = gw.ll.sum() / cnt.sum(); llp_pt = gw.ll_prior.sum() / cnt.sum()
        M.append(dict(sample=sample, model=mod, n_rows=int(t.sum()), n_women=len(women), bal_acc=r.mean(), bal_acc_lo=_ci(bs[mod])[0], bal_acc_hi=_ci(bs[mod])[1], acc=np.trace(t) / t.sum(), recall_menses=r[0], recall_peri_ov=r[1], recall_luteal=r[2],
                      log_loss=ll_pt if ll is not None else np.nan, log_loss_lo=_ci(ll)[0] if ll is not None else np.nan, log_loss_hi=_ci(ll)[1] if ll is not None else np.nan,
                      log_loss_prior_only=llp_pt if ll is not None else np.nan, d_log_loss_vs_prior=(ll_pt - llp_pt) if ll is not None else np.nan, d_ll_lo=_ci(ll - llp)[0] if ll is not None else np.nan, d_ll_hi=_ci(ll - llp)[1] if ll is not None else np.nan))
        for i in range(3):
            for j in range(3): CMr.append(dict(sample=sample, model=mod, true=NAMES[i], pred=NAMES[j], count=int(t[i, j])))
    for mod in bs:
        if mod != "logreg":
            d_ = bs[mod] - bs["logreg"]; lo, hi = _ci(d_); M[[i for i, r in enumerate(M) if r["sample"] == sample and r["model"] == mod][0]].update(d_bal_acc_vs_logreg=np.mean(np.diag(pt[mod]) / pt[mod].sum(1)) - np.mean(np.diag(pt["logreg"]) / pt["logreg"].sum(1)), d_lo=lo, d_hi=hi)
Mdf = pd.DataFrame(M); Mdf.to_csv("results/extE3_metrics.csv", index=False); pd.DataFrame(CMr).to_csv("results/extE3_confusion.csv", index=False)
print(f"E3 Gaussian Bayes: sklearn QDA parity max |posterior diff| {pd.read_csv('results/extE3_sklearn_check.csv').max_abs_posterior_diff.iloc[0]:.1e}")
for s, g in Mdf.groupby("sample"):
    print(f"\n{s}: n={g.n_rows.iloc[0]} test rows, {g.n_women.iloc[0]} women; balanced accuracy [95% CI] (chance 0.333)")
    for r in g.sort_values("bal_acc", ascending=False).itertuples(): print(f"  {r.model:34s} {r.bal_acc:.3f} [{r.bal_acc_lo:.3f}, {r.bal_acc_hi:.3f}]" + (f"  vs logreg {r.d_bal_acc_vs_logreg:+.3f} [{r.d_lo:+.3f}, {r.d_hi:+.3f}]" if r.model != "logreg" else ""))

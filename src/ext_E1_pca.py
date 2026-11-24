"""E1: PCA FROM SCRATCH via SVD. (a) description: explained variance, loadings, numerical rank, collinearity of estradiol and progesterone;
(b) prediction: ridge on all principal components (must equal ridge on raw features: rotation invariance), principal-component regression (number of components chosen by nested CV) and random forest on components,
versus the same models on raw features (Phase 4). PCA (standardise, centre, SVD) is refitted on TRAINING rows at every level (outer folds and inner folds); test rows are only projected."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from src.data import *; from src.mathcore.ridge import standardize_fit, ridge_path_svd; from src.cvlib import LAMS, RF_GRID, _tree; from src.extlib import *
df = load_analysis(); SETS = ["B", "C", "D2", "D"]; FEATN = {"ph1": "peri-ov", "ph2": "luteal", "logE2": "log E2", "logP": "log P", "PMS": "PSST", "BDI_RW": "BDI", "BAI_RW": "BAI"}
def pca_fit(X):
    """Standardise with TRAINING statistics, centre, SVD: Zc = U S Vt. Returns the projector (mu, sd, zm, V) and the singular values."""
    mu, sd = standardize_fit(X); Z = (X - mu) / sd; zm = Z.mean(0); U, s, Vt = np.linalg.svd(Z - zm, full_matrices=False); return (mu, sd, zm, Vt.T), s
def pca_transform(proj, X): mu, sd, zm, V = proj; return ((X - mu) / sd - zm) @ V
# -------------------------------------------------------------- (a) description (whole sample; descriptive only, no prediction)
S_rows, L_rows, K_rows = [], [], []
for sample in ["pooled", "long"]:
    d = subset(df, sample)
    for fs in SETS:
        cols = FEATURE_SETS[fs]; X = d[cols].values; mu, sd = standardize_fit(X); Z = (X - mu) / sd; Zc = Z - Z.mean(0); U, s, Vt = np.linalg.svd(Zc, full_matrices=False); ev = s ** 2 / (len(X) - 1); evr = ev / ev.sum()
        rank = int((s > s.max() * max(Zc.shape) * np.finfo(float).eps).sum()); kappa = float(ev.max() / ev.min())
        for j in range(len(s)): S_rows.append(dict(sample=sample, n_rows=len(d), n_women=d.VPN.nunique(), fset=fs, n_features=len(cols), component=j + 1, singular_value=s[j], eigenvalue=ev[j], explained_var_ratio=evr[j], cum_explained=evr[:j + 1].sum(), numerical_rank=rank, kappa_corr_matrix=kappa))
        for j in range(min(3, len(s))):
            for q, c in enumerate(cols): L_rows.append(dict(sample=sample, fset=fs, component=j + 1, feature=FEATN[c], loading=Vt[j, q]))
    for tf, (a, b) in {"raw": ("estradiol", "progesterone"), "log1p": ("logE2", "logP")}.items():
        for grp, g in [("all rows", d)] + [(nm, d[d.Phase == k]) for k, nm in enumerate(["menses", "peri-ovulatory", "luteal"])]:
            r = np.corrcoef(g[a], g[b])[0, 1]; rho = stats.spearmanr(g[a], g[b])[0]; K_rows.append(dict(sample=sample, scale=tf, group=grp, n_rows=len(g), n_women=g.VPN.nunique(), pearson_r=r, spearman_rho=rho, pc1_share_of_two_hormones=(1 + abs(r)) / 2))
pd.DataFrame(S_rows).to_csv("results/extE1_pca_summary.csv", index=False); pd.DataFrame(L_rows).to_csv("results/extE1_loadings.csv", index=False); pd.DataFrame(K_rows).to_csv("results/extE1_hormone_corr.csv", index=False)
# -------------------------------------------------------------- (b) prediction with PCA inside every training fold
def path_noscale(Ptr, ytr, Pte, lams):
    pm = Ptr.mean(0); W = ridge_path_svd(Ptr - pm, ytr - ytr.mean(), lams); return ((Pte - pm) @ W.T).T + ytr.mean()
def ols_k(Ptr, ytr, Pte, k):
    pm = Ptr.mean(0); w = np.linalg.lstsq(Ptr[:, :k] - pm[:k], ytr - ytr.mean(), rcond=None)[0]; return (Pte[:, :k] - pm[:k]) @ w + ytr.mean()
def predict_pcs(model, Xtr, ytr, gtr, Xte, seed, k_inner=5):
    p = Xtr.shape[1]; grid = {"ridge_pcs": list(LAMS), "pcr": list(range(1, p + 1)), "rf_pcs": RF_GRID}[model]; sse = np.zeros(len(grid))
    for f in np.array_split(np.random.default_rng(seed).permutation(np.unique(gtr)), k_inner):
        iv = np.isin(gtr, f); it = ~iv; assert set(gtr[it]).isdisjoint(gtr[iv]); proj, _ = pca_fit(Xtr[it]); Pt, Pv = pca_transform(proj, Xtr[it]), pca_transform(proj, Xtr[iv])   # PCA fitted on INNER-TRAIN only
        if model == "ridge_pcs": sse += ((path_noscale(Pt, ytr[it], Pv, LAMS) - ytr[iv]) ** 2).sum(1)
        elif model == "pcr": sse += np.array([((ols_k(Pt, ytr[it], Pv, k) - ytr[iv]) ** 2).sum() for k in grid])
        else: sse += np.array([((_tree("rf", hp, seed).fit(Pt, ytr[it]).predict(Pv) - ytr[iv]) ** 2).sum() for hp in grid])
    j = int(np.argmin(sse)); hp = grid[j]; proj, _ = pca_fit(Xtr); Pt, Pe = pca_transform(proj, Xtr), pca_transform(proj, Xte)
    if model == "ridge_pcs": return path_noscale(Pt, ytr, Pe, LAMS[[j]])[0], f"{hp:.3g}"
    if model == "pcr": return ols_k(Pt, ytr, Pe, hp), str(hp)
    return _tree("rf", hp, seed).fit(Pt, ytr).predict(Pe), str(hp)
def one_fold(w, X, y, g, model):
    te = g == w; tr = ~te; assert set(g[tr]).isdisjoint(g[te]); p, hp = predict_pcs(model, X[tr], y[tr], g[tr], X[te], SEED + int(w)); return np.where(te)[0], p, hp
def run(sample, target, fs, model):
    tc = TARGETS[target]; cols = FEATURE_SETS[fs]; assert tc not in cols
    if sample == "cross2long":
        tr, te = df[df["sample"] == "cross"], df[df["sample"] == "long"]; p, hp = predict_pcs(model, tr[cols].values, tr[tc].values, tr.VPN.values, te[cols].values, SEED); hps = [hp] * len(te); d = te
    else:
        d = subset(df, sample); X, y, g = d[cols].values, d[tc].values, d.VPN.values; p = np.zeros(len(y)); hps = [None] * len(y)
        for idx, pp, hp in Parallel(n_jobs=10)(delayed(one_fold)(w, X, y, g, model) for w in np.unique(g)):
            p[idx] = pp
            for i in idx: hps[i] = hp
    return pd.DataFrame(dict(sample=sample, target=target, model=model, fset=fs, VPN=d.VPN.values, Session=d.Session.values, y=d[tc].values, pred=p, hp=hps))
oof = pd.concat([run(s, t, f, m) for s in ["pooled", "long", "cross2long"] for t in TARGETS for f in SETS for m in ["ridge_pcs", "pcr", "rf_pcs"]]); oof.to_csv("results/extE1_oof.csv", index=False)
p4 = pd.read_csv("results/phase4_oof.csv"); p4 = p4[p4.model.isin(["baseline", "ridge", "rf"]) & p4.fset.isin(SETS + ["-"])]
cols_ = ["sample", "target", "model", "fset", "VPN", "Session", "y", "pred"]
comps = [((f"{a}", f), (b, f)) for f in SETS for a, b in [("ridge_pcs", "ridge"), ("pcr", "ridge"), ("rf_pcs", "rf")]]
M, D = boot_configs(pd.concat([p4[cols_], oof[cols_]]), comps); M.to_csv("results/extE1_metrics_all.csv", index=False); D.to_csv("results/extE1_deltas.csv", index=False)
# identity check: ridge on ALL principal components equals ridge on raw standardised features (rotation invariance)
m = oof[oof.model == "ridge_pcs"].merge(p4[p4.model == "ridge"][["sample", "target", "fset", "VPN", "Session", "pred", "hp"]], on=["sample", "target", "fset", "VPN", "Session"], suffixes=("_pcs", "_raw"))
idc = m.groupby(["sample", "target", "fset"]).apply(lambda g: pd.Series(dict(n_rows=len(g), n_women=g.VPN.nunique(), max_abs_pred_diff=(g.pred_pcs - g.pred_raw).abs().max(), same_lambda_rate=(g.hp_pcs.astype(float).round(6) == g.hp_raw.astype(float).round(6)).mean())), include_groups=False).reset_index()
idc.to_csv("results/extE1_identity_check.csv", index=False)
k = oof[oof.model == "pcr"].groupby(["sample", "target", "fset"]).hp.apply(lambda x: float(pd.to_numeric(x).median())).reset_index().rename(columns={"hp": "median_k_components"}); k.to_csv("results/extE1_pcr_k.csv", index=False)
sm = pd.DataFrame(S_rows); print("E1 PCA: pooled set D explained variance ratio by component:", " ".join(f"{v:.2f}" for v in sm[(sm['sample'] == 'pooled') & (sm.fset == 'D')].explained_var_ratio))
kc = pd.DataFrame(K_rows); print("log E2 vs log P, pooled all rows: r=%.2f (n=%d rows)" % (kc[(kc['sample'] == 'pooled') & (kc.scale == 'log1p') & (kc.group == 'all rows')].pearson_r.iloc[0], kc[(kc['sample'] == 'pooled') & (kc.group == 'all rows')].n_rows.iloc[0]))
print(f"identity check ridge-on-all-PCs vs ridge-on-raw: max |prediction difference| over {len(idc)} configs = {idc.max_abs_pred_diff.max():.1e}; same lambda chosen in {idc.same_lambda_rate.mean():.3f} of folds on average")
dd = D[(D.metric == "RMSE")]
for (a, b), g in dd.groupby(["first", "second"], sort=False):
    pass
for nm in ["pcr", "rf_pcs"]:
    g = dd[dd["first"].str.startswith(nm)]; print(f"{nm} minus raw model, RMSE: median delta {g.delta.median():+.4f}, range [{g.delta.min():+.4f}, {g.delta.max():+.4f}], CI excludes 0 in {int(g.excludes_0.sum())} of {len(g)}")

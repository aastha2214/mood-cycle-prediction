"""E7 (distribution shift): compare the two groups of women (95 single-session women = 'cross', 65 longitudinal women with 176 sessions = 'long') on every feature and target.
Standardised mean difference (long minus cross; pooled SD; 95% CI resampling women within each group), Kolmogorov-Smirnov test on the row level (descriptive: sessions of one woman are not independent) and on the first session of each woman
(one value per woman: her FIRST session, so that both groups contribute one independent observation per woman), Holm over variables. Shift summary per direction/target, multivariate covariate shift (AUC of a grouped-CV logistic classifier), and a comparison of regression coefficients between the groups (relationship shift)."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
from src.data import *; from src.plotstyle import *
df = load_analysis(); cr, lg = df[df["sample"] == "cross"], df[df["sample"] == "long"]; B = 2000; rng = np.random.default_rng(SEED)
VARS = {"ph1": ("feature", "peri-ovulatory phase (0/1)"), "ph2": ("feature", "luteal phase (0/1)"), "estradiol": ("feature", "estradiol, raw pg/ml"), "progesterone": ("feature", "progesterone, raw pg/ml"), "logE2": ("feature", "log1p estradiol"), "logP": ("feature", "log1p progesterone"),
        "PMS": ("feature", "PSST"), "BDI_RW": ("feature", "BDI"), "BAI_RW": ("feature", "BAI"), "Affect_pos": ("target", "positive affect"), "DRSP_psychisch": ("target", "DRSP psychological"), "STAI": ("target", "state anxiety")}
def smd(a, b): sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)); return (b.mean() - a.mean()) / sp
def groups(d, v): return [d.loc[d.VPN == w, v].values for w in d.VPN.unique()]
rows = []
for v, (role, lab) in VARS.items():
    a, b = cr[v].values, lg[v].values; ga, gb = groups(cr, v), groups(lg, v); boot = []
    for _ in range(B):
        ra = np.concatenate([ga[i] for i in rng.integers(0, len(ga), len(ga))]); rb = np.concatenate([gb[i] for i in rng.integers(0, len(gb), len(gb))]); boot.append(smd(ra, rb))
    fa, fb = cr.sort_values("Session").groupby("VPN")[v].first().values, lg.sort_values("Session").groupby("VPN")[v].first().values; ks_r = stats.ks_2samp(a, b); ks_w = stats.ks_2samp(fa, fb)
    rows.append(dict(variable=v, label=lab, role=role, n_rows_cross=len(a), n_women_cross=len(ga), n_rows_long=len(b), n_women_long=len(gb), mean_cross=a.mean(), sd_cross=a.std(ddof=1), mean_long=b.mean(), sd_long=b.std(ddof=1),
                     smd_long_minus_cross=smd(a, b), smd_lo=np.percentile(boot, 2.5), smd_hi=np.percentile(boot, 97.5), ks_D_rows=ks_r.statistic, ks_p_rows_descriptive=ks_r.pvalue, ks_D_first=ks_w.statistic, ks_p_first=ks_w.pvalue, n_first_cross=len(fa), n_first_long=len(fb)))
S = pd.DataFrame(rows); S["ks_p_first_holm"] = multipletests(S.ks_p_first, method="holm")[1]; S["ks_p_first_fdr"] = multipletests(S.ks_p_first, method="fdr_bh")[1]; S.to_csv("results/ext_e7_shift_variables.csv", index=False)
# ---- hormones within phase (is the hormone shift only a phase-mix shift?)
ph_rows = []
for k, nm in enumerate(["menses", "peri-ovulatory", "luteal"]):
    for v in ["logE2", "logP"]:
        a, b = cr[cr.Phase == k], lg[lg.Phase == k]; ga, gb = groups(a, v), groups(b, v); bs = []
        for _ in range(B):
            ra = np.concatenate([ga[i] for i in rng.integers(0, len(ga), len(ga))]); rb = np.concatenate([gb[i] for i in rng.integers(0, len(gb), len(gb))]); bs.append(rb.mean() - ra.mean())
        ph_rows.append(dict(phase=nm, variable=v, n_rows_cross=len(a), n_women_cross=a.VPN.nunique(), n_rows_long=len(b), n_women_long=b.VPN.nunique(), mean_cross=a[v].mean(), mean_long=b[v].mean(), diff_long_minus_cross=b[v].mean() - a[v].mean(), diff_lo=np.percentile(bs, 2.5), diff_hi=np.percentile(bs, 97.5)))
pd.DataFrame(ph_rows).to_csv("results/ext_e7_shift_by_phase.csv", index=False)
tab = pd.crosstab(df["sample"], df.Phase); chi = stats.chi2_contingency(tab.values); pd.DataFrame([dict(n_rows_cross=int(tab.loc["cross"].sum()), n_rows_long=int(tab.loc["long"].sum()), chi2=chi[0], dof=chi[2], p_rows_descriptive=chi[1])]).to_csv("results/ext_e7_phase_mix.csv", index=False)
# ---- summary per direction and target: how much of the baseline error is the shift in the mean, and the change in spread
o = pd.read_csv("results/ext_e7_oof.csv"); sm = []
for dname, (tr, te) in {"cross_to_long": (cr, lg), "long_to_cross": (lg, cr)}.items():
    for t, tc in TARGETS.items():
        bl = o[(o["sample"] == dname) & (o.target == t) & (o.model == "baseline")]; mse = float(((bl.y - bl.pred) ** 2).mean()); shift = float((te[tc].mean() - tr[tc].mean()) ** 2)
        sm.append(dict(direction=dname, target=t, train_rows=len(tr), train_women=tr.VPN.nunique(), test_rows=len(te), test_women=te.VPN.nunique(), train_mean=tr[tc].mean(), test_mean=te[tc].mean(), train_sd=tr[tc].std(ddof=1), test_sd=te[tc].std(ddof=1),
                       sd_ratio_test_over_train=te[tc].std(ddof=1) / tr[tc].std(ddof=1), baseline_rmse=np.sqrt(mse), mean_shift_share_of_baseline_mse=shift / mse))
pd.DataFrame(sm).to_csv("results/ext_e7_shift_summary.csv", index=False)
# ---- multivariate covariate shift: can a classifier tell the two groups apart from the features? (grouped 5-fold CV, balanced classes; AUC with CI resampling women)
auc = []; allw = df.VPN.values; ylab = (df["sample"] == "long").astype(int).values
for fs in ["A", "B", "C", "D"]:
    X = df[FEATURE_SETS[fs]].values; score = np.zeros(len(df)); women = np.unique(allw)
    for f in np.array_split(np.random.default_rng(SEED).permutation(women), 5):
        te_ = np.isin(allw, f); m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)).fit(X[~te_], ylab[~te_]); score[te_] = m.decision_function(X[te_])
    bs = []
    for _ in range(B):
        pick = np.random.default_rng(rng.integers(1 << 31)).choice(women, len(women)); idx = np.concatenate([np.where(allw == w)[0] for w in pick])
        if 0 < ylab[idx].mean() < 1: bs.append(roc_auc_score(ylab[idx], score[idx]))
    auc.append(dict(fset=fs, n_rows=len(df), n_women=len(women), auc=roc_auc_score(ylab, score), auc_lo=np.percentile(bs, 2.5), auc_hi=np.percentile(bs, 97.5)))
pd.DataFrame(auc).to_csv("results/ext_e7_shift_auc.csv", index=False)
# ---- relationship shift: OLS coefficients of set D (features and outcome standardised with POOLED statistics) in each group, difference with CI
feats = FEATURE_SETS["D"]; mu, sd = df[feats].mean(), df[feats].std(ddof=0); rows = []
def ols(d, tc, y_mu, y_sd):
    X = np.column_stack([np.ones(len(d)), (d[feats] - mu) / sd]); y = (d[tc].values - y_mu) / y_sd; return np.linalg.lstsq(X, y, rcond=None)[0][1:]
for t, tc in TARGETS.items():
    ym, ys = df[tc].mean(), df[tc].std(ddof=0); b0, b1 = ols(cr, tc, ym, ys), ols(lg, tc, ym, ys); ga, gb = [cr[cr.VPN == w] for w in cr.VPN.unique()], [lg[lg.VPN == w] for w in lg.VPN.unique()]; bs = []
    for _ in range(B):
        ra = pd.concat([ga[i] for i in rng.integers(0, len(ga), len(ga))]); rb = pd.concat([gb[i] for i in rng.integers(0, len(gb), len(gb))]); bs.append(ols(rb, tc, ym, ys) - ols(ra, tc, ym, ys))
    bs = np.array(bs)
    for k, f in enumerate(feats): rows.append(dict(target=t, feature=f, n_rows_cross=len(cr), n_women_cross=cr.VPN.nunique(), n_rows_long=len(lg), n_women_long=lg.VPN.nunique(), coef_cross=b0[k], coef_long=b1[k], diff_long_minus_cross=b1[k] - b0[k],
                                                  diff_lo=np.percentile(bs[:, k], 2.5), diff_hi=np.percentile(bs[:, k], 97.5)))
Cf = pd.DataFrame(rows); Cf["ci_excludes_0"] = (Cf.diff_lo > 0) | (Cf.diff_hi < 0); Cf.to_csv("results/ext_e7_coef_shift.csv", index=False)
# ---- figure: standardised mean differences
fig, ax = plt.subplots(figsize=(7.5, 4.8)); S2 = S.iloc[::-1].reset_index(drop=True); cc = ["#D55E00" if r == "target" else "#0072B2" for r in S2.role]
ax.errorbar(S2.smd_long_minus_cross, range(len(S2)), xerr=[S2.smd_long_minus_cross - S2.smd_lo, S2.smd_hi - S2.smd_long_minus_cross], fmt="none", ecolor="#888", capsize=2)
ax.scatter(S2.smd_long_minus_cross, range(len(S2)), c=cc, zorder=3); ax.axvline(0, color="#333", lw=.8); ax.set_yticks(range(len(S2))); ax.set_yticklabels(S2.label, fontsize=8)
ax.set_xlabel("standardised mean difference, longitudinal minus single-session women (95% CI, women resampled)"); ax.set_title(f"Shift between the two groups: {len(lg)} sessions of {lg.VPN.nunique()} women vs {len(cr)} women\n(blue = features, orange = targets)", fontsize=9)
fig.savefig("results/fig_e7_shift.png"); plt.close(fig)
print(f"E7 shift: {len(S)} variables; groups: {cr.VPN.nunique()} single-session women ({len(cr)} rows), {lg.VPN.nunique()} longitudinal women ({len(lg)} rows)")
print(S[["label", "smd_long_minus_cross", "smd_lo", "smd_hi", "ks_p_first_holm"]].round(3).to_string(index=False))
print(pd.DataFrame(sm)[["direction", "target", "sd_ratio_test_over_train", "mean_shift_share_of_baseline_mse"]].round(3).to_string(index=False)); print(pd.DataFrame(auc).round(3).to_string(index=False))
print("coefficient differences with CI excluding 0:", int(Cf.ci_excludes_0.sum()), "of", len(Cf))

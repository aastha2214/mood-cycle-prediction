"""Overfitting check: mean TRAIN R2 versus TEST R2 under leave-one-participant-out for ridge, random forest and gradient boosting on feature set D (pooled, 271 rows / 160 women).
Per outer fold the model is refitted exactly as in Phase 4 (same seeds, nested CV); it is then evaluated on its own training rows (train R2 = 1 - SSE/SST around the training mean) and on the held-out woman.
Test R2 = pooled out-of-fold R2 (1 - SSE/SST over all held-out rows; a single held-out woman has too few rows for a per-fold R2). Train R2 = mean over the folds. CIs resample women (paired)."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from src.data import *; from src.cvlib import predict_one; from src.extlib import _ci
df = load_analysis(); d = subset(df, "pooled"); cols = FEATURE_SETS["D"]; g = d.VPN.values; women = np.unique(g); B = 2000
def fold(w, X, y, model):
    te = g == w; tr = ~te; assert set(g[tr]).isdisjoint(g[te]); P, _ = predict_one(model, X[tr], y[tr], g[tr], np.vstack([X[te], X[tr]]), SEED + int(w)); pt, ptr = P[:te.sum()], P[te.sum():]
    return w, np.where(te)[0], pt, float(1 - ((ptr - y[tr]) ** 2).sum() / ((y[tr] - y[tr].mean()) ** 2).sum())
rows, p4 = [], pd.read_csv("results/phase4_oof.csv"); W = np.random.default_rng(SEED).multinomial(len(women), np.ones(len(women)) / len(women), size=B).astype(float)
for t, tc in TARGETS.items():
    y = d[tc].values; X = d[cols].values
    for model in ["ridge", "rf", "gb"]:
        out = Parallel(n_jobs=10)(delayed(fold)(w, X, y, model) for w in women); pred = np.zeros(len(y)); tr2 = []
        for w, idx, pt, r2 in out: pred[idx] = pt; tr2.append(r2)
        ref = p4[(p4["sample"] == "pooled") & (p4.target == t) & (p4.model == model) & (p4.fset == "D")].sort_values(["VPN", "Session"]); order = np.lexsort((d.Session.values, d.VPN.values))
        maxdiff = float(np.abs(pred[order] - ref.pred.values).max()); tr2 = np.array(tr2)
        e = pd.DataFrame(dict(VPN=g, sse=(pred - y) ** 2, n=1, sy=y, sy2=y ** 2)).groupby("VPN").sum().loc[women]; sse, n, sy, sy2 = e.sse.values, e.n.values, e.sy.values, e.sy2.values
        te_r2 = lambda wt: 1 - (wt @ sse) / (wt @ sy2 - (wt @ sy) ** 2 / (wt @ n)); one = np.ones(len(women)); pt_te = te_r2(one); bt_te = te_r2(W); bt_tr = (W @ tr2) / W.sum(1); gap_b = bt_tr - bt_te
        rows.append(dict(sample="pooled", target=t, model=model, fset="D", n_rows=len(y), n_women=len(women), train_R2_mean=tr2.mean(), train_R2_sd=tr2.std(ddof=1), train_lo=_ci(bt_tr)[0], train_hi=_ci(bt_tr)[1], test_R2=pt_te, test_lo=_ci(bt_te)[0], test_hi=_ci(bt_te)[1],
                         gap=tr2.mean() - pt_te, gap_lo=_ci(gap_b)[0], gap_hi=_ci(gap_b)[1], max_abs_pred_diff_vs_phase4=maxdiff))
R = pd.DataFrame(rows); R.to_csv("results/extOF_train_test.csv", index=False)
print(f"Train vs test R2, set D, pooled (n={R.n_rows.iloc[0]} rows, {R.n_women.iloc[0]} women); max |test prediction - Phase 4| = {R.max_abs_pred_diff_vs_phase4.max():.1e}")
print(R[["target", "model", "train_R2_mean", "test_R2", "gap", "gap_lo", "gap_hi"]].round(3).to_string(index=False))

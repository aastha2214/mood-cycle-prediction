"""Leave-one-participant-out (LOPO) CV with nested, woman-grouped hyperparameter selection. No woman is ever in both train and test."""
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from .data import SEED
from .mathcore.ridge import fit_predict_path
LAMS = np.logspace(-4, 1, 26)
RF_GRID = [3, 10, 25]                              # min_samples_leaf (100 trees)
GB_GRID = [(1, 50), (1, 150), (2, 50), (2, 150)]   # (max_depth, n_estimators), learning_rate 0.05
MODELS = ["baseline", "ridge", "rf", "gb"]

def _tree(model, hp, seed):
    if model == "rf": return RandomForestRegressor(n_estimators=100, min_samples_leaf=hp, random_state=seed, n_jobs=1)
    return GradientBoostingRegressor(max_depth=hp[0], n_estimators=hp[1], learning_rate=0.05, random_state=seed)

def predict_one(model, Xtr, ytr, gtr, Xte, seed, k_inner=5):
    """Fit on training rows only (hyperparameters chosen by k-fold CV grouped by woman inside the training rows) and predict Xte."""
    if model == "baseline": return np.full(len(Xte), ytr.mean()), "mean"
    grid = list(LAMS) if model == "ridge" else (RF_GRID if model == "rf" else GB_GRID)
    folds = np.array_split(np.random.default_rng(seed).permutation(np.unique(gtr)), k_inner); sse = np.zeros(len(grid))
    for f in folds:
        iv = np.isin(gtr, f); it = ~iv; assert set(gtr[it]).isdisjoint(gtr[iv])
        if model == "ridge": sse += ((fit_predict_path(Xtr[it], ytr[it], Xtr[iv], LAMS) - ytr[iv]) ** 2).sum(1)
        else:
            for j, hp in enumerate(grid): sse[j] += ((_tree(model, hp, seed).fit(Xtr[it], ytr[it]).predict(Xtr[iv]) - ytr[iv]) ** 2).sum()
    j = int(np.argmin(sse)); hp = grid[j]
    if model == "ridge": return fit_predict_path(Xtr, ytr, Xte, LAMS[[j]])[0], f"{hp:.3g}"
    return _tree(model, hp, seed).fit(Xtr, ytr).predict(Xte), str(hp)

def _one(w, X, y, g, model):
    te = g == w; tr = ~te; assert set(g[tr]).isdisjoint(g[te])          # leakage guard: group integrity
    p, hp = predict_one(model, X[tr], y[tr], g[tr], X[te], SEED + int(w)); return np.where(te)[0], p, hp

def lopo(d, tcol, cols, model, n_jobs=10):
    assert tcol not in cols and not any(c in cols for c in ["Affect_pos", "Affect_neg", "Affect_Diff", "DRSP", "DRSP_psychisch", "DRSP_physisch", "STAI", "PANAS_pos", "PANAS_neg"]), "target-derived feature"
    assert not d.duplicated(["VPN", "Session"]).any()
    X, y, g = d[cols].values, d[tcol].values, d.VPN.values; pred = np.zeros(len(y)); hps = [None] * len(y)
    out = Parallel(n_jobs=n_jobs if model != "baseline" else 1)(delayed(_one)(w, X, y, g, model) for w in np.unique(g))
    for idx, p, hp in out:
        pred[idx] = p
        for i in idx: hps[i] = hp
    return pred, hps

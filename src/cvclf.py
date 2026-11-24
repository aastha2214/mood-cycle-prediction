"""Leave-one-woman-out classification with nested, woman-grouped hyperparameter choice (inner metric: balanced accuracy)."""
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_sample_weight
from .data import SEED
GRIDS = {"logreg": [0.01, 0.1, 1, 10, 100], "rf": [3, 10, 25], "gb": [(1, 50), (1, 150), (2, 50), (2, 150)]}
def bacc(y, p, classes=(0, 1, 2)): return float(np.mean([(p[y == c] == c).mean() for c in classes if (y == c).any()]))
def _fit(model, hp, X, y, seed):
    w = compute_sample_weight("balanced", y)
    if model == "logreg": return make_pipeline(StandardScaler(), LogisticRegression(C=hp, max_iter=2000)).fit(X, y, logisticregression__sample_weight=w)
    if model == "rf": return RandomForestClassifier(n_estimators=100, min_samples_leaf=hp, class_weight="balanced", random_state=seed, n_jobs=1).fit(X, y)
    return GradientBoostingClassifier(max_depth=hp[0], n_estimators=hp[1], learning_rate=0.05, random_state=seed).fit(X, y, sample_weight=w)
def predict_one_clf(model, Xtr, ytr, gtr, Xte, seed, k=5, fixed_hp=None):
    if model == "baseline": return np.full(len(Xte), np.bincount(ytr.astype(int)).argmax()), "majority"
    grid = GRIDS[model]
    if fixed_hp is not None: hp = fixed_hp
    else:
        folds = np.array_split(np.random.default_rng(seed).permutation(np.unique(gtr)), k); sc = []
        for hp_ in grid:
            pred = np.zeros(len(ytr)); 
            for f in folds:
                iv = np.isin(gtr, f); it = ~iv; assert set(gtr[it]).isdisjoint(gtr[iv]); pred[iv] = _fit(model, hp_, Xtr[it], ytr[it], seed).predict(Xtr[iv])
            sc.append(bacc(ytr, pred))
        hp = grid[int(np.argmax(sc))]
    return _fit(model, hp, Xtr, ytr, seed).predict(Xte), str(hp)
def _one(w, X, y, g, model, fixed_hp):
    te = g == w; tr = ~te; assert set(g[tr]).isdisjoint(g[te]); p, hp = predict_one_clf(model, X[tr], y[tr], g[tr], X[te], SEED + int(w), fixed_hp=fixed_hp); return np.where(te)[0], p, hp
def lopo_clf(X, y, g, model, n_jobs=10, fixed_hp=None):
    pred = np.zeros(len(y), dtype=int); hps = [None] * len(y)
    for idx, p, hp in Parallel(n_jobs=n_jobs if model != "baseline" else 1)(delayed(_one)(w, X, y, g, model, fixed_hp) for w in np.unique(g)):
        pred[idx] = p
        for i in idx: hps[i] = hp
    return pred, hps

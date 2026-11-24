"""Ridge regression from scratch (NumPy only).
Loss on centred data:  L(w) = (1/n)||y - Xw||^2 + lam ||w||^2   (intercept = mean, unpenalised)
=> normal equations (X'X + n lam I) w = X'y  ;  sklearn Ridge(alpha = n*lam) is the same problem."""
import numpy as np

def center(X, y):
    mu, ym = X.mean(0), y.mean(); return X - mu, y - ym, mu, ym

def standardize_fit(X):
    mu, sd = X.mean(0), X.std(0); sd = np.where(sd == 0, 1.0, sd); return mu, sd

def ridge_normal(Xc, yc, lam):
    n, p = Xc.shape
    return np.linalg.solve(Xc.T @ Xc + n * lam * np.eye(p), Xc.T @ yc)

def svd_parts(Xc):
    return np.linalg.svd(Xc, full_matrices=False)

def ridge_svd(Xc, yc, lam, parts=None):
    """w = V diag(s/(s^2+n lam)) U'y.  lam=0 gives the pseudo-inverse solution V S^+ U'y (small s truncated)."""
    n = Xc.shape[0]; U, s, Vt = parts if parts is not None else svd_parts(Xc)
    if lam == 0:
        keep = s > s.max() * max(Xc.shape) * np.finfo(float).eps; d = np.zeros_like(s); d[keep] = 1 / s[keep]
    else:
        d = s / (s ** 2 + n * lam)
    return Vt.T @ (d * (U.T @ yc))

def ridge_path_svd(Xc, yc, lams):
    parts = svd_parts(Xc); return np.array([ridge_svd(Xc, yc, l, parts) for l in lams])

def loss(w, Xc, yc, lam):
    r = yc - Xc @ w; return r @ r / len(yc) + lam * w @ w

def grad(w, Xc, yc, lam):
    return -2 / len(yc) * Xc.T @ (yc - Xc @ w) + 2 * lam * w

def hessian(Xc, lam):
    return 2 / len(Xc) * Xc.T @ Xc + 2 * lam * np.eye(Xc.shape[1])

def fit_predict(Xtr, ytr, Xte, lam, standardize=True):
    """Full pipeline: scale with TRAIN stats only, centre, SVD ridge, predict."""
    mu, sd = standardize_fit(Xtr) if standardize else (Xtr.mean(0), np.ones(Xtr.shape[1]))
    A = (Xtr - mu) / sd; Ac, yc, _, ym = center(A, ytr)
    w = ridge_svd(Ac, yc, lam); return ((Xte - mu) / sd - A.mean(0)) @ w + ym

def fit_predict_path(Xtr, ytr, Xte, lams):
    """Ridge predictions for every lambda at once (one SVD). TRAIN statistics only for scaling/centring. Returns (len(lams), n_test)."""
    mu, sd = standardize_fit(Xtr); A = (Xtr - mu) / sd; Ac, yc, am, ym = center(A, ytr)
    W = ridge_path_svd(Ac, yc, lams); return (((Xte - mu) / sd - am) @ W.T).T + ym

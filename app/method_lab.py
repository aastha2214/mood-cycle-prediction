"""Illustrative optimisation and shrinkage examples in pure NumPy. Synthetic problems only: no study data is used here."""
import numpy as np

TOL = 1e-6            # relative loss gap at which an optimiser counts as converged
CAP = 100_000         # largest number of iterations that is run
DIM = 20              # dimension of the synthetic quadratic
N_ILLUSTRATIVE = 100  # number of observations n in the shrinkage factor
KAPPAS = sorted({int(round(x)) for x in np.logspace(np.log10(2), 5, 90)} | {2, 10, 100, 1000, 10_000, 100_000})   # log-spaced choices

def problem(kappa, dim=DIM):
    """Eigenvalues (1/kappa ... 1, so Lmax = 1 and mu = 1/kappa) and a start at which every direction contributes equally to the initial loss."""
    lam = np.geomspace(1.0 / kappa, 1.0, dim); return lam, 1.0 / np.sqrt(lam)

def step_sizes(kappa):
    """(gradient descent step 1/Lmax, momentum step, momentum coefficient) for Lmax = 1 and mu = 1/kappa."""
    lmax, mu = 1.0, 1.0 / kappa
    return 1.0 / lmax, 4.0 / (np.sqrt(lmax) + np.sqrt(mu)) ** 2, ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** 2

def descend(lam, w0, step, beta=0.0, tol=TOL, cap=CAP):
    """Heavy-ball momentum on f(w) = 1/2 sum(lam w^2); beta = 0 is gradient descent. Returns (loss gap relative to the start per iteration, iterations to reach tol or None)."""
    w, v = np.array(w0, float), np.zeros(len(lam)); start = 0.5 * lam @ (w * w); gap = np.empty(cap + 1); gap[0] = 1.0
    for t in range(1, cap + 1):
        v = beta * v - step * lam * w; w = w + v; gap[t] = 0.5 * (lam @ (w * w)) / start
        if gap[t] <= tol: return gap[: t + 1], t
    return gap, None

def compare_methods(kappa, tol=TOL, cap=CAP):
    """Gradient descent and tuned momentum on the same problem."""
    lam, w0 = problem(kappa); eta_gd, eta_hb, beta = step_sizes(kappa)
    gd, gd_iters = descend(lam, w0, eta_gd, 0.0, tol, cap); hb, hb_iters = descend(lam, w0, eta_hb, beta, tol, cap)
    return dict(kappa=kappa, gd=gd, hb=hb, gd_iters=gd_iters, hb_iters=hb_iters, eta_gd=eta_gd, eta_hb=eta_hb, beta=beta, cap=cap, tol=tol)

def shrinkage(singular_values, lams, n=N_ILLUSTRATIVE):
    """Ridge shrinkage factor s^2 / (s^2 + n lambda): one row per singular value, one column per lambda."""
    s2 = np.asarray(singular_values, float)[:, None] ** 2; return s2 / (s2 + n * np.asarray(lams, float)[None, :])

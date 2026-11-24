"""GD, SGD, momentum, Adam on the ridge loss (centred data). Each returns (w, losses) with one loss per iteration (SGD: per epoch)."""
import numpy as np
from .ridge import loss, grad

def gd(Xc, yc, lam, lr, iters, w0=None):
    w = np.zeros(Xc.shape[1]) if w0 is None else w0.copy(); L = [loss(w, Xc, yc, lam)]
    for _ in range(iters):
        w = w - lr * grad(w, Xc, yc, lam); L.append(loss(w, Xc, yc, lam))
    return w, np.array(L)

def momentum(Xc, yc, lam, lr, beta, iters):
    w = np.zeros(Xc.shape[1]); v = np.zeros_like(w); L = [loss(w, Xc, yc, lam)]
    for _ in range(iters):
        v = beta * v - lr * grad(w, Xc, yc, lam); w = w + v; L.append(loss(w, Xc, yc, lam))
    return w, np.array(L)

def adam(Xc, yc, lam, lr, iters, b1=0.9, b2=0.999, eps=1e-8):
    w = np.zeros(Xc.shape[1]); m = np.zeros_like(w); v = np.zeros_like(w); L = [loss(w, Xc, yc, lam)]
    for t in range(1, iters + 1):
        g = grad(w, Xc, yc, lam); m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        w = w - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps); L.append(loss(w, Xc, yc, lam))
    return w, np.array(L)

def sgd(Xc, yc, lam, lr, epochs, batch=16, decay=0.0, seed=42):
    """Mini-batch SGD; unbiased gradient -2/|B| X_B'(y_B - X_B w) + 2 lam w; lr_t = lr/(1+decay*t)."""
    rng = np.random.default_rng(seed); n, p = Xc.shape; w = np.zeros(p); L = [loss(w, Xc, yc, lam)]; t = 0
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch):
            b = idx[s:s + batch]; g = -2 / len(b) * Xc[b].T @ (yc[b] - Xc[b] @ w) + 2 * lam * w
            w = w - lr / (1 + decay * t) * g; t += 1
        L.append(loss(w, Xc, yc, lam))
    return w, np.array(L)

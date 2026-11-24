> **Working notes: the checked numbers are in docs/report.md**

# 3e  Ridge as the MAP estimate under a Gaussian prior
Code check: `src/phase3a_verify.py` (second part).

**Model.** y = Xw + eps, eps ~ N(0, sigma^2 I) (likelihood), prior w ~ N(0, tau^2 I) (we believe weights are small and symmetric around 0). Bayes: p(w|y) is proportional to p(y|w) p(w).
**Derivation.** -log p(w|y) = (1/(2 sigma^2)) ||y - Xw||^2 + (1/(2 tau^2)) ||w||^2 + const.
Multiply by 2 sigma^2 (does not move the minimum): minimise ||y - Xw||^2 + (sigma^2/tau^2) ||w||^2. Divide by n to match 3a's loss: (1/n)||y - Xw||^2 + lambda||w||^2 with
**n * lambda = sigma^2 / tau^2**, i.e. lambda = sigma^2/(n tau^2).
In words: ridge = "least squares with a prior belief that weights are small". A tight prior (small tau^2) or noisy data (large sigma^2) gives a larger lambda and more shrinkage; with a flat prior (tau^2 -> infinity) lambda -> 0 and we recover ordinary least squares (maximum likelihood). Because the posterior is Gaussian, its mode (MAP) equals its mean: w_MAP = (X'X + (sigma^2/tau^2) I)^-1 X'y.
**Numerical check** (pooled, n=271, 160 women, DRSP psychological, set D, lambda = 0.1 so n*lambda = 27.1, tau^2 = 2 chosen, sigma^2 = 54.2 implied): the gradient of the negative log posterior at the closed-form ridge solution has maximum absolute value 1.8e-14, i.e. zero. (Phase 7 E2 builds the full Bayesian version, with the posterior variance.)

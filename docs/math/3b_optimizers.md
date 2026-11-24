> **Working notes: the checked numbers are in docs/report.md**

# 3b  Solving the same ridge loss with GD, SGD, momentum, Adam
Code: `src/mathcore/optim.py`, experiments: `src/phase3b_optim.py`. Figures: `results/fig_3b_loss_curves.png`, `results/fig_3b_lr_study.png`. Table: `results/phase3b_lr_study.csv`.
Setting: DRSP psychological, feature set D (log1p hormones, standardised), lambda = 0.01. Pooled n=271 rows / 160 women (kappa = 6.6) and longitudinal n=176 / 65 (kappa = 8.1). Reference solution w* = closed form from 3a.

**Updates** (g = gradient of L; lr = learning rate):
- GD: w <- w - lr g. "Walk downhill in proportion to the slope."
- SGD: same, but g is computed from a random mini-batch (here 16 rows), an unbiased but noisy estimate of the full gradient. Cheap per step, but never settles exactly with a constant lr.
- Momentum (heavy ball): v <- beta v - lr g; w <- w + v. Keeps a running velocity, which smooths zig-zags. Here beta = 0.9, lr = 1/lambda_max (not tuned).
- Adam: keeps running averages of g (m) and g^2 (v), bias-corrected, and steps by lr * m_hat/(sqrt(v_hat)+eps): every weight gets its own effective step size. beta1 = 0.9, beta2 = 0.999.

**Results (300 passes over the data; relative suboptimality (L-L*)/(L0-L*)).**
| optimizer (pooled) | setting | result |
|---|---|---|
| GD | lr = 1/lambda_max = 0.257 | machine precision, ||w-w*|| 4e-15 (reaches 1e-6 after 29 iterations) |
| Momentum | lr = 1/lambda_max, beta 0.9 | 5e-15; ||w-w*|| 2.9e-7 (oscillates, slower than GD here since untuned) |
| Adam | lr = 0.05 | 6.5e-6, still creeping down; ||w-w*|| 1.7e-2 |
| SGD | lr = 0.25/lambda_max, batch 16 | stalls near 7e-3 (noise floor), ||w-w*|| 0.38 |
Longitudinal sample gives the same ordering (GD 0, momentum 3e-15, Adam 5.8e-5, SGD 3.5e-3).

**Learning-rate study (pooled; see csv for both samples).**
- GD: converges for lr from 0.1 to 1.9 x (1/lambda_max); **diverges at 2.1 x**, as theory says (stable only if lr < 2/lambda_max).
- Momentum with beta = 0.9 tolerates larger steps (stable at 2.1 x here; the heavy-ball limit is 2(1+beta)/lambda_max).
- SGD: with constant lr the final error grows with lr (3e-6 at 0.01 x up to 2e-1 at 1 x): bigger steps = faster start but a higher noise floor. Remedy: decaying lr (`decay` argument, not used in the headline run).
- Adam: lr 0.001 is too slow (0.75 left after 300 passes); 0.1 and 0.5 reach 1e-13 to 1e-14.

**GD vs theory** (see 3c): the loss gap shrinks by about rho^2 per step with rho = max_i |1 - lr*lambda_i|. Predicted iterations to 1e-6 (upper estimate from the slowest direction): 42 (lr=1/lambda_max) and 23 (optimal lr = 2/(lambda_max+lambda_min)); observed 29 and 18. Longitudinal: predicted 53 / 28, observed 39 / 21. Observed <= predicted, as expected, since the starting error need not have full weight on the slowest direction.

## Addendum: tuned heavy-ball momentum (`src/phase3b2_momentum_tuned.py`, `results/fig_3b_momentum_tuned.png`)
For a quadratic loss with Hessian eigenvalues between mu (smallest) and L (largest), kappa = L/mu, the best heavy-ball settings are lr = 4/(sqrt(L)+sqrt(mu))^2 and beta = ((sqrt(kappa)-1)/(sqrt(kappa)+1))^2. Plain GD needs about kappa steps; tuned momentum needs about sqrt(kappa). In plain words: GD walks down the steepest direction and zig-zags across a long narrow valley, while the "velocity" term lets the ball roll along the valley floor instead of bouncing off the walls. (The untuned beta = 0.9 in the main table above does not get this benefit.)
Iterations until the loss gap is below 1e-6 of its start (lambda = 0.01, DRSP psychological, GD with lr = 1/L):
| Problem (unscaled) | kappa | GD | tuned heavy ball | rough theory ln(1e-6)/ln(beta) |
|---|---|---|---|---|
| pooled (271 rows / 160 women), raw hormones | 64,182 | not reached in 60,000 | 1,634 | ~875 |
| pooled, log1p hormones | 1,088 | 4,385 | 212 | ~114 |
| longitudinal (176 / 65), raw hormones | 75,939 | not reached in 60,000 | 1,821 | ~952 |
| longitudinal, log1p hormones | 1,040 | 4,671 | 206 | ~111 |
Observed counts are about 2x the rough theory (the theory is the asymptotic rate; the early oscillating phase costs extra), but the gain over GD is large and of the predicted order (about sqrt(kappa) vs kappa). Standardising (kappa about 7) makes the problem easy for plain GD without any tuning.

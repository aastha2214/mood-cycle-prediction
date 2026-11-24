> **Working notes: the checked numbers are in docs/report.md**

# 3c  Gradient, Hessian, conditioning and convergence speed
Code: `src/phase3c_hessian.py`; table `results/phase3c_condition.csv`; figure `results/fig_3c_hessian.png`.

**Gradient and Hessian of the ridge loss** L(w) = (1/n)||y - Xw||^2 + lambda||w||^2 (centred data):
grad L = -(2/n) X'(y - Xw) + 2 lambda w,   H = Hess L = (2/n) X'X + 2 lambda I.
H does not depend on w: the loss is a quadratic bowl. H is symmetric positive semi-definite for lambda = 0 and positive definite for lambda > 0, so the bowl has exactly one minimum (strictly convex). Eigenvalues of H are 2(s_i^2/n + lambda): ridge adds 2 lambda to each, lifting the flattest direction off zero.

**Condition number** kappa = lambda_max(H)/lambda_min(H) = (lambda_max + 2 lambda)/(lambda_min + 2 lambda) measures how elongated the bowl is: long, narrow valleys have large kappa.
**Link to GD.** With step lr = 1/lambda_max, the error along eigendirection i shrinks by (1 - lambda_i/lambda_max) per step, so the slowest direction gives rho = 1 - 1/kappa and about kappa * ln(1/eps) / 2 steps to reach a loss gap of eps. Large kappa means slow GD; the best constant step 2/(lambda_max+lambda_min) gives rho = (kappa-1)/(kappa+1).

**Results (feature set D, DRSP psychological, lambda = 0.01).** Pooled n=271 / 160 women; longitudinal n=176 / 65.
| Features | kappa (pooled, lambda=0) | kappa (long, lambda=0) | GD iterations to 1e-6 pooled: observed / predicted bound | long: observed / predicted bound |
|---|---|---|---|---|
| raw (unscaled, raw E2 and P) | 71,796 | 85,408 | > 60,000 (not reached) / 443,352 | > 60,000 / 524,563 |
| log1p hormones, unscaled | 1,277 | 1,224 | 4,385 / 7,513 | 4,671 / 7,183 |
| log1p + standardised | 6.8 | 8.4 | 29 / 42 | 39 / 53 |
Reason: unscaled columns have wildly different variances (progesterone up to hundreds, phase dummies 0/1), so the bowl is stretched; standardising puts every column on variance 1, making H nearly round. Raising lambda also helps (pooled, standardised: kappa 6.8 -> 5.3 at lambda 0.1 -> 2.3 at lambda 1; raw: 71,796 -> 5,583 at lambda 1). Note this is why standardising is a requirement for gradient methods, not just for a fair ridge penalty. Closed-form solutions (3a) are unaffected by kappa except through numerical precision.

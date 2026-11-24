> **Working notes: the checked numbers are in docs/report.md**

# 3a  Ridge regression: normal equations and SVD pseudo-inverse
Code: `src/mathcore/ridge.py` (`ridge_normal`, `ridge_svd`), check: `src/phase3a_verify.py`.

**Setup.** n sessions, p features, X (n x p), y (n). Centre X and y (subtract column means). The mean of y then becomes the intercept and is not penalised. Ridge loss:
L(w) = (1/n) ||y - Xw||^2 + lambda ||w||^2.

**Normal equations.** Set the gradient to zero: -(2/n) X'(y - Xw) + 2 lambda w = 0, so (X'X + n lambda I) w = X'y, i.e. w = (X'X + n lambda I)^-1 X'y.
In words: ordinary least squares asks X'X to be invertible; adding n*lambda to its diagonal makes it always invertible and pulls the weights toward zero. (scikit-learn's `alpha` equals n*lambda; that is the only convention difference.)

**SVD form.** Write X = U S V' (singular values s_i). Then X'X + n lambda I = V (S^2 + n lambda I) V', so
w = V diag( s_i / (s_i^2 + n lambda) ) U'y.
Each direction i of the data is shrunk by the factor s_i^2/(s_i^2 + n lambda): directions with large singular values (well determined by the data) are barely touched, weak directions are suppressed. For lambda = 0 the factor is 1/s_i, which is the Moore-Penrose pseudo-inverse solution w = X^+ y = V S^+ U'y (tiny s_i are dropped, giving the minimum-norm least-squares solution when X is rank deficient).

**Check against scikit-learn** (`Ridge(alpha=n*lambda)`, `LinearRegression` for lambda=0). 72 configurations: pooled (n=271 rows, 160 women) and longitudinal (n=176 rows, 65 women) x 3 targets x 2 feature versions (log1p set D, raw-hormone set D) x lambda in {0, 1e-4, 1e-2, 0.1, 1, 10}.
Max absolute coefficient difference: normal equations 2.7e-15, SVD 1.1e-14; intercept difference 0. Asserted < 1e-8 in the script.

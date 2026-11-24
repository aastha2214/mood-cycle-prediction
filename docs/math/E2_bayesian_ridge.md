> **Working notes: the checked numbers are in docs/report.md**

# E2  Bayesian ridge regression (posterior over weights, predictive mean and variance)
Code: `src/ext_E2_bayes_ridge.py`. Results: `results/extE2_*.csv`; report section 8.2.

**Model.** Centre the data (the intercept is the training mean). Assume y = Zw + noise with noise ~ N(0, 1/beta) and a prior w ~ N(0, (1/alpha) I). Here beta is the noise precision (1/sigma^2) and alpha the prior precision (1/tau^2).

**Posterior over the weights.** Bayes' theorem gives another Gaussian: w | data ~ N(m, S) with S = (alpha I + beta Z'Z)^-1 and m = beta S Z'y. In words: S says how well each direction of the weight space is pinned down by the data (directions with little data keep a wide posterior); m is the best guess. **Link to ridge:** m is exactly the ridge solution with n*lambda = alpha/beta (see note 3e). The Bayesian version additionally keeps S.

**Predictive distribution for a new row x.** y_new | x ~ N(x'm, 1/beta + x'Sx). Two parts: 1/beta is the noise of a single observation; x'Sx is the uncertainty about the weights, which is larger for unusual x. (Because the intercept is fixed at the training mean, its uncertainty is ignored.)

**Choosing alpha and beta without cross-validation: evidence maximisation.** The evidence is the probability of the training outcomes after averaging over all weights allowed by the prior: p(y | alpha, beta) = integral p(y | w, beta) p(w | alpha) dw. In log form, with N rows and p features:
log p = (p/2) log alpha + (N/2) log beta - beta/2 ||y - Zm||^2 - alpha/2 m'm - (1/2) log|alpha I + beta Z'Z| - (N/2) log 2 pi.
The first terms reward fitting the data and the log-determinant term penalises complexity automatically (an Occam factor): large alpha = simple model. Setting the derivatives to zero gives updates that are iterated to a fixed point (MacKay): gamma = sum_i beta s_i^2/(alpha + beta s_i^2) (the number of well-determined parameters, from the singular values s_i of Z), then alpha = gamma/(m'm) and 1/beta = ||y - Zm||^2/(N - gamma). Everything is computed from the **training rows of each fold only**; no inner CV loop is needed.

**When there is no signal,** m'm shrinks, alpha grows without bound and the weights go to zero: the model returns the training mean with a variance equal to the residual variance. That is the correct answer, not a failure.

**Calibration check.** If the predictive distribution is right, the standardised error z = (y - mean)/sd has mean square 1. Averages above 1 mean the model is over-confident, below 1 under-confident. The log predictive density (negative log-likelihood of the held-out values, in nats) is compared with a Gaussian that just uses the training mean and variance.

**Limits.** Rows are treated as independent although sessions of one woman are correlated (the unexplained between-woman variance is absorbed into 1/beta); the prior is the same for all weights; the Gaussian noise is an assumption.

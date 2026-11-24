> **Working notes: the checked numbers are in docs/report.md**

# E4  Prediction intervals and their coverage
Code: `src/ext_E4_intervals.py`. Results: `results/extE4_*.csv`, `results/fig_E4_coverage.png`; report section 8.4.

**What an interval promises.** A "90% prediction interval" for a new session should contain the observed value in about 90 out of 100 sessions. This is **coverage**. A confidence interval for the *mean* is much narrower and answers a different question; a prediction interval must include the noise of a single observation. We also look at the **width** (sharpness): of two intervals with the same coverage, the narrower is more useful.

**Bayesian predictive interval.** From E2 the predictive distribution of a new outcome is N(mean, sd^2) with sd^2 = 1/beta + x'Sx. The central interval of level L is mean +/- z sd, where z is the normal quantile (1.28, 1.64, 1.96 for 80%, 90%, 95%). It is correct only if the model (Gaussian noise, independent rows, same population) is correct.

**Conformal (jackknife) interval.** Idea: do not trust the model's own variance; measure its errors. Inside each training set, refit the model leaving out one training woman at a time and record the absolute errors on the woman left out (her sessions together, so a woman's own sessions never help predict herself). Sort these n errors; the interval for a new session is the point prediction +/- the ceil(L(n+1))-th smallest error. If the new woman is "exchangeable" with the training women, the chance that her error is below that quantile is at least about L, whatever the error distribution (no Gaussian assumption). Caveats: exact guarantees need the refit-without-her model to be the same as the final model (here the quantile comes from leave-one-woman-out models, the prediction from the model on all training women, so coverage is approximate); several sessions of one woman are correlated, so the coverage per woman is more variable than per session; if the test women differ from the training women (a different sample) the guarantee is lost.

**Reference interval.** Training mean +/- z x training SD: the interval one would give without any predictors. Models help only if their intervals are narrower at the same coverage.

**Empirical coverage and honesty.** For every method and level, coverage is the share of held-out sessions inside the interval, with a 95% interval from resampling women. We call coverage "off" when this interval excludes the nominal level, under-coverage (too narrow) or over-coverage (too wide, conservative).

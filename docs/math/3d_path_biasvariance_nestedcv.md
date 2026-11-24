> **Working notes: the checked numbers are in docs/report.md**

# 3d  Ridge path, bias-variance trade-off, lambda by nested CV
Code: `src/phase3d_path_cv.py`; figure `results/fig_3d_path_biasvar.png`; tables `results/phase3d_*.csv`. Feature set D (log1p, standardised inside each training set), 26 values of lambda from 1e-4 to 10. Everything reported for pooled (n=271 rows, 160 women) and longitudinal (n=176, 65 women).

**Ridge path.** Plot of each coefficient as lambda grows: all shrink smoothly to 0 (left panels). In DRSP psychological, BDI/PSST/BAI carry the largest standardised coefficients; hormone coefficients are small (log E2 about -0.5, log P about -0.2 pooled at lambda near 0).

**Bias-variance.** Expected squared error at a new woman = (bias)^2 + variance + noise. On real data the true function is unknown, so I estimate it empirically: repeat 10 times {hold out 25% of women; draw 100 bootstrap resamples of the *remaining women*; fit ridge on each; predict held-out rows}. Variance = spread of predictions across bootstraps; "bias^2 + noise" = squared error of the *average* prediction (these two cannot be separated without knowing the noise). By the algebraic identity E[(f_b - y)^2] = Var(f_b) + (E[f_b] - y)^2, the two parts add up to the expected squared error of a single bootstrap fit.
Pooled: variance falls from 2.58 (lambda=1e-4) to 0.52 (lambda=10) while bias^2+noise rises from 48.5 to 74.6; the total is lowest (50.7) at lambda about 0.1. Longitudinal: variance 5.42 -> 1.09, bias^2+noise 67.0 -> 86.8, minimum 69.1 at lambda about 0.4. Here variance is small relative to bias^2+noise (the latter is mostly irreducible noise from unexplained mood variation), so the U-shape is shallow: ridge only helps a little with 7 features and 160+ women.

**Nested CV for lambda.** Outer loop = leave-one-woman-out (160 folds pooled, 65 longitudinal). In each outer fold the held-out woman is untouched; lambda is picked by 5-fold CV *by woman* inside the remaining women; then ridge is refit on all remaining women with that lambda and predicts the held-out woman. Scaling is fitted on training data only at every level. Out-of-fold results (feature set D; R2 is relative to the training-mean baseline, i.e. 1 - SSE_model/SSE_mean):
| Sample | Target | Ridge MAE | Mean-baseline MAE | R2 vs mean | lambda median (IQR) |
|---|---|---|---|---|---|
| pooled | positive affect | 0.598 | 0.639 | 0.096 | 0.40 (0.25-0.40) |
| pooled | DRSP psych. | 5.579 | 7.334 | 0.364 | 0.16 (0.16-0.25) |
| pooled | state anxiety | 0.309 | 0.339 | 0.153 | 0.40 (0.25-0.40) |
| long | positive affect | 0.657 | 0.682 | 0.050 | 1.00 (1.00-1.59) |
| long | DRSP psych. | 6.579 | 7.897 | 0.270 | 0.25 (0.25-0.40) |
| long | state anxiety | 0.347 | 0.375 | 0.110 | 0.63 (0.40-0.63) |
No chosen lambda sat at the grid maximum. These are *demonstration* numbers for the method; Phase 4 repeats the evaluation properly for all feature sets and models with confidence intervals. Caveat to examine there: PSST is itself a premenstrual-symptom questionnaire, so its strong link with DRSP may be partly measuring the same construct.

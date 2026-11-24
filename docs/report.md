# Mood Prediction from Menstrual Cycle Data: Ridge Regression, Optimization and Bayesian Methods from Scratch

*PAI103 project report (learning project, not a medical tool). Every number below is loaded from the saved result files by `src/build_report.py`; see `docs/numbers_provenance.md`. The authors' published values appear only where labelled **authors' reported (reference only)**. Extension sections (Phase 7) can be added as markdown files in `docs/extensions/`; they are inserted at the marker near the end.*

## Summary of findings

**The question.** Mood is often said to follow the menstrual cycle. This project asks a practical version: if we know a woman's cycle phase and the estradiol and progesterone in her saliva, can we predict her mood better than by simply guessing the average, for a woman the model has never seen? Data: open study data of Pletzer & Noachtar (2023), 271 rows, 160 women (of which 176 rows, 65 women are women with up to three sessions).

**Five key results.**
1. *Hormones add nothing to cycle phase for unseen women.* Of 27 paired comparisons (phase + hormones versus phase alone) 0 showed an improvement and 15 were worse; the smallest uncorrected p-value for an improvement was 0.334.
2. *Phase alone helps only a little, and only for premenstrual symptoms.* Share of the error removed compared with guessing the average (ridge, pooled): DRSP psychological symptoms +0.051 [+0.005, +0.095], positive affect +0.021 [-0.009, +0.051], state anxiety -0.001 [-0.001, -0.000].
3. *Questionnaires, and above all the woman's own history, carry the predictable signal.* With BDI, BAI and PSST added the DRSP figure rises to +0.360 [+0.247, +0.457]; giving the model a woman's own mean from her other sessions lifted a phase-plus-hormone model from +0.034 [-0.020, +0.084] to +0.324 [+0.152, +0.459] (longitudinal women).
4. *Group-level effects are real but small, and they happen within a woman.* In-sample, DRSP symptoms are lower in the luteal phase than at menses (-0.51 [-0.75, -0.27] standard deviations, as in the authors' paper), and the estradiol link is within a woman (-0.48 [-0.68, -0.29]) but absent between women (-0.04 [-0.27, +0.19]). Phase explains only 0.085 of the variance, which is why significant averages do not give individual predictions.
5. *The checks behave as they should.* Hormones identify the cycle phase (balanced accuracy 0.70 [0.65, 0.75] versus chance 0.333); emotion-recognition accuracy, which should be unrelated, is not predicted (best R2 +0.032), with one small, uncorrected caveat for reaction speed.

The extensions (section 8) agree: the Bayesian ridge predicts like ridge (largest RMSE difference 0.121); 90% prediction intervals for new women of the same population cover 0.893 (Bayesian) and 0.900 (conformal) of sessions but are wide and fail across samples; dropping the hormones from the full model costs nothing; and random forests and boosting fit their training women far better than new women (train-minus-test R2 up to 0.265) without testing better than ridge. Cross-sample validation (E7: training on the single-session women and testing on the longitudinal women, and the reverse) gives the same picture: the questionnaire model transfers in both directions (DRSP, ridge, set D: R2 +0.292 [+0.158, +0.409] and +0.499 [+0.331, +0.634]), while hormones improved prediction in neither direction after Holm and FDR correction (smallest adjusted p for an improvement 1.000); this is validation across two groups of women of the same study, not across datasets.

**Three main limitations.** (1) *Small samples:* only 65 women contribute repeated sessions and 95 contribute one, so intervals are wide and small effects cannot be excluded. (2) *Noisy hormone measurements:* salivary immunoassays have limited validity, which could hide a real effect. (3) *Concurrent, not forecasting:* hormones and mood are measured in the same session, so the models describe a state rather than predict tomorrow. This is a learning project, not a medical tool.

## Abstract

**Question.** Do salivary estradiol and progesterone improve the prediction of mood beyond cycle phase, in women the model has never seen? **Data.** Open data of Pletzer & Noachtar (2023): 271 sessions from 160 women (176 sessions from 65 longitudinal women with up to three sessions at menses, peri-ovulation and mid-luteal phase, and 95 sessions from 95 cross-sectional women with one session each). **Methods.** Ridge regression implemented from scratch (normal equations, SVD, gradient methods), compared with random forests and gradient boosting, under leave-one-woman-out cross-validation with nested hyper-parameter selection and women-level bootstrap intervals; mixed-effects models and permutation tests; a negative control and a sanity check. **Results.** Phase alone explained little out of sample (R2 versus the mean baseline, DRSP psychological symptoms, ridge: +0.051 [+0.005, +0.095]; positive affect +0.021 [-0.009, +0.051]; state anxiety -0.001 [-0.001, -0.000]). Adding hormones to phase never helped: in 27 paired comparisons none showed a significant improvement and 15 were significantly worse. Questionnaire scores (BDI, BAI, PSST) carried most of the predictable signal (DRSP, ridge, set D: +0.360 [+0.247, +0.457]). In-sample, phase and within-woman estradiol effects were clear (mixed models, longitudinal sample, DRSP psychological: luteal minus menses -0.51 [-0.75, -0.27] SD), consistent with the authors, and the estradiol effect was entirely within-woman. Hormones did identify the phase (balanced accuracy 0.70 [0.65, 0.75] versus chance 0.333), and failed to predict emotion-recognition accuracy as expected. Conclusion: on this sample, hormones add no detectable predictive value for mood in unseen women beyond cycle phase; significant group-level effects do not translate into individual-level prediction.

## 1. Introduction

Mood is often said to vary with the menstrual cycle. A statistical effect on average ("mood is better in the peri-ovulatory phase than at menses") is different from the ability to predict one woman's mood from her cycle information. This project asks the predictive question on open data of Pletzer & Noachtar (2023, *Hormones and Behavior*), where the authors reported group-level phase and estradiol effects on mood. The course context (Mathematics Behind AI) is the second aim: the core methods are implemented from scratch in NumPy (least squares and ridge via normal equations and SVD, gradient and Hessian, condition number, GD / SGD / momentum / Adam, ridge as a Bayesian MAP estimate) and then used in a leakage-aware evaluation. The research question is: **do salivary estradiol and progesterone improve prediction of mood beyond cycle phase alone, on women the model has never seen?**

## 2. Data

**Source.** OSF project dmhrw (four files, SHA-256 checked in `data/MANIFEST.md`). The modelling table is the questionnaires file: one row per session of one woman, 333 rows and 184 women. I applied the authors' own filter (`include == 1`): 272 rows, 161 women; and dropped the single row with a missing PSST score, leaving **271 rows, 160 women** (the *pooled* sample).

**Samples (disjoint by construction).** *Longitudinal*: women with more than one included session, 176 rows, 65 women; rows per phase menses/peri-ovulatory/luteal = 65/51/60 (46 women have all three sessions, 19 have two, because the filter acts per session). *Cross-sectional*: women with exactly one included session, 95 rows, 95 women; rows per phase 38/18/39. The paper reports 65 longitudinal and 153 cross-sectional women (**authors' reported**). My longitudinal count is 65; the paper's cross-sectional figure corresponds to all first sessions (153 rows, equal to the paper's figure, which includes 58 first sessions of longitudinal women), not to my disjoint definition (95 women) which I chose to prevent leakage in the cross-to-longitudinal experiment.

**Variables.** Phase coded 0/1/2 = menses / peri-ovulatory / luteal (confirmed from the authors' R script labels). Hormones: salivary estradiol and progesterone (pg/ml), log1p-transformed and standardised inside training folds for modelling. Trait scores: PSST (`PMS`), BDI, BAI. Targets: positive affect (`Affect_pos`), DRSP psychological symptoms (`DRSP_psychisch`; the column `DRSP_physical` is an exact duplicate of it and is not used), state anxiety (`STAI`).

**Hormone distributions.** Estradiol skewness 1.13 raw versus 0.22 after log1p; progesterone 2.79 versus -0.81; 9 / 7 exact zeros (so `log1p`, not `log`). My longitudinal phase means reproduce the paper's table (see Results).

**Reproducibility.** Every number and figure in this report is reproduced by `./run_all.sh` (fixed seed; see README).

**How much of each target is a stable person-level trait? (ICC)** The intraclass correlation is the share of variance that lies between women (one-way ANOVA estimator for unbalanced data, with a 95% bootstrap interval resampling women):

| Target | Pooled ICC [95% CI] (n rows / women) | Longitudinal ICC [95% CI] (n rows / women) |
|---|---|---|
| positive affect | 0.37 [0.19, 0.54] (271 / 160) | 0.43 [0.23, 0.57] (176 / 65) |
| DRSP psychological symptoms | 0.31 [0.12, 0.45] (271 / 160) | 0.39 [0.20, 0.53] (176 / 65) |
| state anxiety | 0.52 [0.34, 0.66] (271 / 160) | 0.57 [0.42, 0.71] (176 / 65) |

A large ICC means much of the variance is a stable person level that a model for an unseen woman can only reach through person-level predictors (trait scores), not through phase or hormones. The pooled ICC mixes two kinds of women (within-woman variance is only identified from the longitudinal women) and is the less trustworthy of the two.

## 3. Methods

**Feature sets.** A: phase (two dummies, menses = reference). B: log1p estradiol and log1p progesterone. C: A + B. D: C + PSST + BDI + BAI. D2: D without PSST (added to test whether PSST overlaps with the DRSP outcome). Every set uses the same 271 complete rows.

**Mathematical core (NumPy only).** Ridge loss L(w) = (1/n)||y - Xw||^2 + lambda||w||^2 on centred data (intercept unpenalised). (a) Normal equations (X'X + n lambda I)w = X'y and the SVD form w = V diag(s_i/(s_i^2 + n lambda)) U'y (pseudo-inverse at lambda = 0), checked against scikit-learn; (b) the same loss solved by gradient descent, mini-batch SGD, heavy-ball momentum and Adam; (c) gradient, Hessian H = (2/n)X'X + 2 lambda I, eigenvalues and condition number kappa = lambda_max/lambda_min, raw versus standardised; (d) ridge path, bias-variance by bootstrap over women, lambda by nested CV; (e) ridge as the MAP estimate under a Gaussian prior, with n lambda = sigma^2/tau^2. Notes in `docs/math/`.

**Models and validation.** Baseline: the training-set mean. Ridge (own implementation, 26 values of lambda), random forest (3 leaf sizes, 100 trees), gradient boosting (4 depth/trees settings, learning rate 0.05). Outer loop: **leave-one-participant-out** (all sessions of a woman are held out together, so no woman is ever in both training and test); inner loop: 5-fold cross-validation grouped by woman inside the training women for hyper-parameters (**nested CV**). Scaling and centring use training rows only. Random seed 42. Additional transfer experiment: train on the cross-sectional women, test on the longitudinal women. Metrics: MAE, RMSE, R2 (pooled out-of-fold), and R2 versus the mean baseline (R2_oos = 1 - SSE_model/SSE_baseline). Uncertainty: percentile bootstrap resampling **women** (B = 2000); paired differences use the same resampled women. Per-woman MAE/RMSE mean +/- SD are in `results/phase4_tables.md` (R2 has no per-woman value). Leakage checklist: `docs/leakage_checklist.md`.

**Statistics.** Random-intercept mixed models (REML, statsmodels) on z-scored outcomes (as in the authors' R script), Wald tests, Holm adjustment over the three phase contrasts; permutation tests that keep the design (phase labels shuffled within woman, estradiol shuffled within woman for the within test, woman-mean estradiol shuffled across women for the between test). Estradiol split into a within-woman part (value minus the woman's own mean) and a between-woman part (her mean). Multiple comparisons: Holm and Benjamini-Hochberg FDR over the paired hormone comparisons. **Credibility checks:** a negative control (emotion-recognition accuracy and speed from the same inputs) and a sanity check (cycle phase from hormones, balanced accuracy and confusion matrix).

## 4. Results

### 4.1 Mathematical core

**Ridge solvers agree with scikit-learn.** Across 72 configurations (two samples x three targets x two feature versions x six lambdas) the largest coefficient difference was 2.7e-15 (normal equations) and 1.1e-14 (SVD).

**Conditioning drives gradient-descent speed.** Hessian condition number kappa (lambda = 0) and iterations plain GD (step 1/lambda_max, lambda = 0.01) needs to shrink the loss gap to 1e-6:

| Features | kappa pooled | GD iterations pooled | kappa longitudinal | GD iterations longitudinal |
|---|---|---|---|---|
| raw, unscaled | 71,796 | not reached | 85,407 | not reached |
| log1p hormones, unscaled | 1,277 | 4,385 | 1,224 | 4,671 |
| log1p + standardised | 6.8 | 29 | 8.4 | 39 |

(Sample sizes: pooled 271 rows, 160 women; longitudinal 176 rows, 65 women. 'not reached' = not converged within the iteration cap.) Standardising brings kappa to a few units, so GD converges in a few dozen steps; on unscaled data it is hopelessly slow. **Tuned heavy-ball momentum** (beta = ((sqrt(kappa)-1)/(sqrt(kappa)+1))^2, lr = 4/(sqrt(L)+sqrt(mu))^2) reaches the same accuracy in 1,634 iterations on the raw problem (kappa 64,182) where GD did not converge, and 212 versus 4,385 for GD on the log1p problem: about sqrt(kappa) steps instead of kappa. GD with learning rate 2.1 / lambda_max diverged, as theory (stability needs lr < 2/lambda_max) predicts, while 1.9 / lambda_max converged (relative suboptimality 2.2e-16). SGD with a constant step stalls at a noise floor that grows with the step (3.1e-06 at 0.01/lambda_max versus 2.0e-01 at 1/lambda_max). Figures: `results/fig_3b_*.png`, `fig_3c_hessian.png`.

**Bias-variance and lambda.** Bootstrapping women and averaging over held-out splits, variance falls and bias-squared-plus-noise rises as lambda grows (bias and irreducible noise cannot be separated on real data); the expected squared error is smallest at lambda = 0.1 (pooled, 271 rows, 160 women) and 0.4 (longitudinal, 176 rows, 65 women); the curve is shallow. Figure `results/fig_3d_path_biasvar.png`. **Ridge as MAP:** with n lambda = sigma^2/tau^2 the gradient of the negative log posterior at the ridge solution has maximum absolute value 1.8e-14 (n = 271 rows).

### 4.2 Prediction for unseen women (leave-one-participant-out, nested CV)

**Pooled sample** (leave-one-woman-out; the numbers are for the held-out women). Cells: R2 versus the training-mean baseline [95% bootstrap CI over women].

| model | set | positive affect (271 rows, 160 women) | DRSP psychological symptoms (271 rows, 160 women) | state anxiety (271 rows, 160 women) |
|---|---|---|---|---|
| baseline (mean) RMSE | - | 0.763 | 8.892 | 0.413 |
| ridge | A | +0.021 [-0.009, +0.051] | +0.051 [+0.005, +0.095] | -0.001 [-0.001, -0.000] |
| ridge | B | -0.005 [-0.012, -0.000] | -0.002 [-0.020, +0.014] | -0.001 [-0.002, -0.000] |
| ridge | C | +0.001 [-0.028, +0.029] | +0.037 [-0.004, +0.076] | -0.002 [-0.002, -0.001] |
| ridge | D2 | +0.102 [+0.004, +0.186] | +0.339 [+0.231, +0.438] | +0.149 [+0.048, +0.245] |
| ridge | D | +0.093 [-0.005, +0.175] | +0.360 [+0.247, +0.457] | +0.154 [+0.048, +0.248] |
| rf | A | +0.023 [-0.012, +0.058] | +0.052 [+0.004, +0.101] | -0.008 [-0.015, -0.001] |
| rf | B | -0.007 [-0.046, +0.032] | -0.027 [-0.081, +0.024] | -0.044 [-0.084, -0.006] |
| rf | C | +0.001 [-0.039, +0.040] | +0.005 [-0.049, +0.056] | -0.045 [-0.084, -0.008] |
| rf | D2 | +0.068 [-0.029, +0.156] | +0.373 [+0.260, +0.473] | +0.108 [-0.024, +0.229] |
| rf | D | +0.078 [-0.015, +0.161] | +0.377 [+0.259, +0.472] | +0.131 [-0.002, +0.248] |
| gb | A | +0.023 [-0.009, +0.055] | +0.051 [+0.006, +0.096] | -0.006 [-0.010, -0.002] |
| gb | B | -0.010 [-0.039, +0.016] | +0.001 [-0.030, +0.030] | -0.023 [-0.052, +0.005] |
| gb | C | -0.007 [-0.034, +0.017] | +0.019 [-0.015, +0.054] | -0.023 [-0.052, +0.005] |
| gb | D2 | +0.088 [-0.006, +0.172] | +0.380 [+0.275, +0.469] | +0.124 [+0.022, +0.219] |
| gb | D | +0.087 [-0.004, +0.167] | +0.377 [+0.278, +0.461] | +0.094 [-0.004, +0.184] |

**Longitudinal sample** (leave-one-woman-out; the numbers are for the held-out women). Cells: R2 versus the training-mean baseline [95% bootstrap CI over women].

| model | set | positive affect (176 rows, 65 women) | DRSP psychological symptoms (176 rows, 65 women) | state anxiety (176 rows, 65 women) |
|---|---|---|---|---|
| baseline (mean) RMSE | - | 0.805 | 9.487 | 0.439 |
| ridge | A | +0.027 [-0.009, +0.062] | +0.055 [+0.004, +0.110] | -0.004 [-0.020, +0.009] |
| ridge | B | -0.005 [-0.013, +0.001] | -0.003 [-0.047, +0.030] | -0.002 [-0.004, +0.000] |
| ridge | C | -0.007 [-0.029, +0.016] | +0.033 [-0.015, +0.084] | -0.001 [-0.005, +0.002] |
| ridge | D2 | +0.061 [-0.040, +0.145] | +0.249 [+0.114, +0.365] | +0.108 [-0.023, +0.219] |
| ridge | D | +0.037 [-0.069, +0.117] | +0.259 [+0.111, +0.382] | +0.127 [-0.016, +0.244] |
| rf | A | +0.030 [-0.012, +0.071] | +0.055 [-0.001, +0.115] | +0.003 [-0.020, +0.023] |
| rf | B | -0.024 [-0.065, +0.014] | -0.009 [-0.081, +0.060] | -0.044 [-0.079, -0.009] |
| rf | C | -0.015 [-0.056, +0.023] | -0.022 [-0.092, +0.049] | -0.042 [-0.077, -0.009] |
| rf | D2 | +0.077 [-0.049, +0.187] | +0.289 [+0.146, +0.421] | +0.081 [-0.063, +0.216] |
| rf | D | +0.075 [-0.049, +0.185] | +0.284 [+0.132, +0.422] | +0.120 [-0.038, +0.266] |
| gb | A | +0.030 [-0.008, +0.068] | +0.054 [+0.004, +0.110] | +0.000 [-0.020, +0.018] |
| gb | B | -0.033 [-0.069, +0.001] | -0.027 [-0.083, +0.024] | -0.010 [-0.041, +0.018] |
| gb | C | -0.041 [-0.077, -0.008] | -0.034 [-0.091, +0.018] | -0.010 [-0.041, +0.018] |
| gb | D2 | +0.120 [+0.025, +0.215] | +0.297 [+0.151, +0.432] | +0.131 [+0.003, +0.254] |
| gb | D | +0.118 [+0.029, +0.204] | +0.291 [+0.158, +0.411] | +0.111 [-0.028, +0.247] |

**Cross-sectional to longitudinal sample** (training set: 95 rows, 95 women; test set: 176 rows, 65 women). Cells: R2 versus the training-mean baseline [95% bootstrap CI over women].

| model | set | positive affect (176 rows, 65 women) | DRSP psychological symptoms (176 rows, 65 women) | state anxiety (176 rows, 65 women) |
|---|---|---|---|---|
| baseline (mean) RMSE | - | 0.797 | 9.398 | 0.435 |
| ridge | A | +0.012 [-0.012, +0.037] | +0.054 [+0.016, +0.099] | -0.002 [-0.004, +0.000] |
| ridge | B | -0.010 [-0.044, +0.023] | +0.001 [-0.001, +0.002] | -0.001 [-0.003, +0.001] |
| ridge | C | +0.006 [-0.022, +0.035] | +0.022 [+0.008, +0.040] | -0.003 [-0.006, +0.000] |
| ridge | D2 | +0.111 [+0.005, +0.206] | +0.252 [+0.116, +0.378] | +0.118 [-0.001, +0.226] |
| ridge | D | +0.107 [+0.004, +0.201] | +0.292 [+0.158, +0.409] | +0.128 [+0.023, +0.226] |
| rf | A | +0.002 [-0.000, +0.003] | +0.056 [+0.013, +0.105] | -0.002 [-0.009, +0.004] |
| rf | B | -0.016 [-0.064, +0.040] | -0.040 [-0.092, +0.010] | +0.000 [-0.011, +0.012] |
| rf | C | -0.015 [-0.063, +0.040] | -0.015 [-0.077, +0.045] | -0.001 [-0.012, +0.010] |
| rf | D2 | +0.040 [-0.100, +0.158] | +0.241 [+0.088, +0.362] | +0.166 [+0.061, +0.259] |
| rf | D | +0.030 [-0.112, +0.149] | +0.248 [+0.101, +0.367] | +0.147 [+0.035, +0.245] |
| gb | A | +0.008 [-0.022, +0.042] | +0.055 [+0.016, +0.100] | -0.024 [-0.043, -0.006] |
| gb | B | -0.003 [-0.044, +0.035] | -0.029 [-0.069, +0.005] | -0.173 [-0.305, -0.063] |
| gb | C | -0.002 [-0.043, +0.037] | +0.002 [-0.034, +0.042] | -0.174 [-0.305, -0.065] |
| gb | D2 | +0.051 [-0.059, +0.143] | +0.249 [+0.114, +0.360] | +0.096 [+0.014, +0.168] |
| gb | D | +0.058 [-0.051, +0.155] | +0.258 [+0.130, +0.363] | +0.096 [+0.014, +0.168] |

Figure: `results/fig_4_skill.png`. Full tables with MAE, RMSE, R2 and per-woman mean +/- SD: `results/phase4_tables.md`.

**Findings.**
1. *Phase alone gives a small gain, only for DRSP.* Ridge on set A: DRSP psychological +0.051 [+0.005, +0.095], positive affect +0.021 [-0.009, +0.051], state anxiety -0.001 [-0.001, -0.000] (pooled).
2. *Hormones do not improve prediction beyond phase.* Set C versus set A: among the 27 paired RMSE comparisons (3 samples x 3 targets x 3 models) 0 showed a significant improvement and 15 a significant deterioration. Example, pooled DRSP, ridge: change in RMSE +0.061 [+0.008, +0.119] (positive = worse). Hormones alone (set B) never beat the baseline: significant improvements in 0 of 27.
3. *Trait scores carry the signal.* Set D, DRSP psychological, pooled: ridge +0.360 [+0.247, +0.457] (MAE 5.596 versus baseline MAE 7.334); anxiety +0.154 [+0.048, +0.248]; positive affect +0.093 [-0.005, +0.175]. This matches the large person-level ICC.
4. *PSST overlap is not the explanation.* Without PSST (set D2) pooled DRSP ridge gives +0.339 [+0.231, +0.438] against +0.360 [+0.247, +0.457] with PSST; D was significantly better than D2 in 0 comparisons (BDI and BAI drive the skill; they too overlap conceptually with DRSP and STAI).
5. *Model choice matters little:* for DRSP with set D (pooled) ridge, random forest and gradient boosting give +0.360, +0.377 and +0.377; trees on hormone-only sets are sometimes worse than the baseline (e.g. random forest, set B, anxiety, pooled: -0.044 [-0.084, -0.006]), i.e. overfitting noise.
6. *Personal history helps far more than hormones* (leave-one-session-out, longitudinal sample, 176 rows, 65 women): ridge on set C without history R2_oos +0.034 [-0.020, +0.084] for DRSP, with the woman's mean from her other sessions added +0.324 [+0.152, +0.459]; anxiety -0.007 [-0.015, -0.001] versus +0.415 [+0.221, +0.577].
7. *Leakage check:* with the target shuffled across rows (3 permutations x 3 targets, pooled 271 rows, 160 women, ridge on set D) the skill was -0.0057 to +0.0001 (expected: about zero or slightly negative).

**Overfitting check: train versus test R2** (feature set D, pooled, 271 rows, 160 women; leave-one-woman-out, nested CV, same folds and seeds as above; test predictions identical to the Phase 4 ones, largest difference 7e-15). Train R2 = R2 of each fold's fitted model on its own training sessions (mean over the 160 folds, SD over folds); test R2 = pooled out-of-fold R2 (1 - SSE/SST over all held-out sessions: a single held-out woman has too few sessions for a per-fold R2); gap = train minus test, with a 95% CI from resampling women (paired).

| Target | Model | Mean train R2 (SD over folds) | Test R2 [95% CI] | Gap [95% CI] |
|---|---|---|---|---|
| positive affect | ridge | 0.148 (0.008) | 0.082 [-0.020, 0.161] | +0.066 [-0.013, +0.169] |
| positive affect | rf | 0.218 (0.013) | 0.068 [-0.032, 0.153] | +0.150 [+0.065, +0.251] |
| positive affect | gb | 0.226 (0.038) | 0.077 [-0.023, 0.157] | +0.149 [+0.067, +0.251] |
| DRSP psychological symptoms | ridge | 0.407 (0.006) | 0.352 [0.244, 0.453] | +0.055 [-0.047, +0.163] |
| DRSP psychological symptoms | rf | 0.557 (0.004) | 0.370 [0.244, 0.468] | +0.188 [+0.089, +0.314] |
| DRSP psychological symptoms | gb | 0.523 (0.059) | 0.370 [0.266, 0.455] | +0.152 [+0.066, +0.260] |
| state anxiety | ridge | 0.203 (0.008) | 0.143 [0.028, 0.233] | +0.061 [-0.030, +0.177] |
| state anxiety | rf | 0.384 (0.068) | 0.119 [-0.027, 0.235] | +0.265 [+0.147, +0.413] |
| state anxiety | gb | 0.265 (0.046) | 0.082 [-0.029, 0.172] | +0.184 [+0.093, +0.296] |

The flexible models fit the training women much better than the held-out women: the random forest and gradient boosting gaps are positive with intervals excluding 0 for all three targets (largest: 0.265, rf on state anxiety), while ridge has a smaller gap whose interval includes 0 (range +0.055 to +0.066). Yet the test R2 of the three models is similar (for DRSP 0.352 / 0.370 / 0.370): the larger gap did not translate into a visibly lower test score here (one possible reason, not tested: nested CV limits the flexibility of the forests), but the large training R2 would be a misleading measure of what to expect for a new woman. The honest number is the leave-one-woman-out test R2.

### 4.3 Mixed-effects models and permutation tests (like-for-like with the paper: longitudinal sample)

Longitudinal sample: 176 rows, 65 women; pooled: 271 rows, 160 women. z-scored outcomes; b = standardised effect with 95% CI; menses = reference. Authors' values are **reported by the authors (reference only)**.

| Effect | Authors' reported (reference only) | Mine, longitudinal | Mine, pooled |
|---|---|---|---|
| Positive affect: peri-ovulatory - menses | b = 0.41 | +0.41 [+0.14, +0.67], p=0.003 | +0.42 [+0.18, +0.67], p<0.001 |
| Positive affect: luteal - menses | b = 0.47 | +0.47 [+0.21, +0.72], p<0.001 | +0.41 [+0.18, +0.63], p<0.001 |
| Negative affect: peri-ovulatory - menses | lower, not significant | -0.28 [-0.57, +0.01], p=0.057 | -0.24 [-0.50, +0.02], p=0.074 |
| DRSP psychological: peri-ovulatory - menses | b = -0.68 | -0.69 [-0.95, -0.44], p<0.001 | -0.68 [-0.92, -0.44], p<0.001 |
| DRSP psychological: luteal - menses | b = -0.52 | -0.51 [-0.75, -0.27], p<0.001 | -0.49 [-0.70, -0.27], p<0.001 |
| DRSP psychological ~ estradiol (z) | b = -0.30, p <0.001 | -0.28 [-0.44, -0.11], p=0.001 | -0.16 [-0.29, -0.03], p=0.014 |
| DRSP psychological ~ progesterone (z) | not related | -0.06 [-0.19, +0.07], p=0.355 | -0.07 [-0.18, +0.04], p=0.210 |
| State anxiety: peri-ovulatory - menses | b = -0.27, p = 0.07 | -0.27 [-0.51, -0.03], p=0.027 | -0.17 [-0.40, +0.06], p=0.148 |
| State anxiety ~ estradiol (z) | b = -0.15, p = 0.06 | -0.15 [-0.30, +0.01], p=0.063 | -0.08 [-0.21, +0.04], p=0.202 |
| Positive affect ~ estradiol (z) | all p > 0.16 | p=0.227 | p=0.114 |
| Negative affect ~ estradiol (z) | all p > 0.16 | p=0.168 | p=0.207 |
| BDI -> positive affect (phase x BDI model) | b = -0.45 | -0.45 [-0.69, -0.22], p<0.001 | -0.44 [-0.61, -0.27], p<0.001 |
| BDI -> negative affect | b = 0.56 | +0.56 [+0.34, +0.78], p<0.001 | +0.53 [+0.37, +0.69], p<0.001 |
| Phase x PSST on DRSP psychological (2-df Wald) | p = 0.001 | p=0.088 | p=0.030 |

Hormone means by phase, longitudinal sample (rows per phase 65/51/60): estradiol 0.65/1.10/1.02 pg/ml (authors' reported: 0.65/1.10/1.02), progesterone 34.74/43.75/155.56 (authors' reported: 34.7/43.8/155.6). The like-for-like longitudinal results reproduce the authors' pattern closely. The PSST interaction is weaker for my psychological score (see table); I cannot tell from the files how the authors' DRSP score and model differ.

**Estradiol and DRSP: within-woman versus between-woman.**

| Sample (n) | Outcome | within-woman b [95% CI], p | between-woman b [95% CI], p | permutation p (within / between) |
|---|---|---|---|---|
| long (176 rows, 65 women) | DRSP psychological | -0.48 [-0.68, -0.29], p<0.001 | -0.04 [-0.27, +0.19], p=0.746 | 0.0002 / 0.717 |
| long (176 rows, 65 women) | DRSP total (14 items) | -0.53 [-0.73, -0.33], p<0.001 | -0.02 [-0.25, +0.21], p=0.858 | 0.0002 / 0.826 |
| pooled (271 rows, 160 women) | DRSP psychological | -0.51 [-0.72, -0.31], p<0.001 | -0.01 [-0.16, +0.14], p=0.879 | 0.0002 / 0.898 |
| pooled (271 rows, 160 women) | DRSP total (14 items) | -0.56 [-0.76, -0.35], p<0.001 | -0.02 [-0.16, +0.13], p=0.813 | 0.0002 / 0.795 |

The estradiol-DRSP association lies entirely within women. Within a woman estradiol is confounded with phase: adding phase to the model shrinks the within-woman coefficient (DRSP psychological, longitudinal) from -0.48 [-0.68, -0.29] to -0.21 [-0.47, +0.04], so the data cannot separate an estradiol effect from a phase effect. In the pooled sample the within part is estimated only from the repeat-measured women.

**Permutation tests** (phase labels shuffled within woman; smallest attainable p is 1/(B+1)). Longitudinal phase omnibus: DRSP psychological p = 0.0002, positive affect p = 0.0006, state anxiety p = 0.0316, negative affect p = 0.0598; within-woman estradiol slope on DRSP psychological p = 0.0002 (n = 176 rows, 65 women).

**Why significant in-sample effects coexist with out-of-sample R2 near zero.** A p-value says an average effect is not zero; R2 says how much individual variation it explains. Fixed-effect (marginal) R2 of phase alone versus the fit including each woman's own intercept (conditional R2), and the out-of-sample skill of the same predictor (ridge, set A):

| Outcome | Sample (n) | Marginal R2 (phase) | Conditional R2 | Wald p (phase) | Out-of-sample R2_oos, ridge A [95% CI] |
|---|---|---|---|---|---|
| DRSP psychological symptoms | long (176 rows, 65 women) | 0.085 | 0.548 | <0.001 | +0.055 [+0.004, +0.110] |
| DRSP psychological symptoms | pooled (271 rows, 160 women) | 0.081 | 0.504 | <0.001 | +0.051 [+0.005, +0.095] |
| positive affect | long (176 rows, 65 women) | 0.045 | 0.501 | <0.001 | +0.027 [-0.009, +0.062] |
| positive affect | pooled (271 rows, 160 women) | 0.040 | 0.463 | <0.001 | +0.021 [-0.009, +0.051] |
| state anxiety | long (176 rows, 65 women) | 0.015 | 0.589 | 0.043 | -0.004 [-0.020, +0.009] |
| state anxiety | pooled (271 rows, 160 women) | 0.006 | 0.541 | 0.256 | -0.001 [-0.001, -0.000] |

Phase is a small share of total variance; the mixed model removes the woman-level baseline (which a new woman does not bring), estimation noise subtracts from out-of-sample R2, and the effect is relative to a woman's own level.

**Multiple comparisons.** The 27 paired C-vs-A comparisons (3 samples x 3 targets x 3 models) and, separately, the 27 B-vs-baseline comparisons were corrected with Holm and with Benjamini-Hochberg FDR (p-values from the same woman-bootstrap; the tests share data, so they are not independent):

| Family | raw p < 0.05 | Holm < 0.05 | BH-FDR < 0.05 | first set worse (dRMSE > 0) | smallest one-sided p for 'improves' (raw / Holm) |
|---|---|---|---|---|---|
| C vs A | 15 | 1 | 10 | 25 of 27 | 0.334 / 1.000 |
| B vs baseline | 5 | 1 | 1 | 24 of 27 | 0.170 / 1.000 |

No comparison shows that hormones improve prediction, with or without correction; the correction only weakens the 'hormones made it worse' statements. Holm is valid under any dependence; BH-FDR is approximate here.

### 4.4 Credibility checks

**Negative control (emotion recognition).** Per session accuracy (% of the 60 trials correct) and speed (median RT of correct trials) predicted with the same protocol (sets A-D, three models, three samples; pooled 271 rows, 160 women, longitudinal 176 rows, 65 women, cross-to-longitudinal test 176 rows, 65 women). Accuracy: R2_oos for all hormone-containing sets lies between -0.195 and +0.032; the highest value (random forest, set C, cross-to-longitudinal) is +0.032 [-0.023, +0.081]. Mixed models (longitudinal, 176 rows, 65 women): phase p = 0.571, estradiol p = 0.313, progesterone p = 0.701.

**One caveat: speed.** For speed the control is not perfectly clean. Highest R2_oos among hormone sets: +0.055 (model ridge, set D, longitudinal sample, 176 rows, 65 women), and estradiol relates to speed in the mixed model (longitudinal b = -0.18, nominal p = 0.019). Among the 18 B-vs-baseline negative-control tests, 1 survives Holm correction: speed, cross-to-longitudinal, ridge, change in RMSE -0.46 ms on a baseline RMSE of 199.7 ms. This is detectable but practically negligible, my speed measure differs from the authors' (no outlier cut, all emotions pooled), and the p-values are nominal; I do not claim an effect.

**Sanity check: cycle phase from hormones** (log1p estradiol and progesterone, three classes; chance for balanced accuracy = 0.333, also the score of the majority-class baseline: 0.333).

| Sample (test n) | Model | Balanced accuracy [95% CI] | Accuracy | Recall menses / peri-ov / luteal |
|---|---|---|---|---|
| pooled (271 rows, 160 women) | baseline | 0.333 [0.333, 0.333] | 0.380 | 1.00 / 0.00 / 0.00 |
| pooled (271 rows, 160 women) | logreg | 0.703 [0.650, 0.754] | 0.720 | 0.67 / 0.55 / 0.89 |
| pooled (271 rows, 160 women) | rf | 0.699 [0.645, 0.747] | 0.701 | 0.59 / 0.67 / 0.84 |
| pooled (271 rows, 160 women) | gb | 0.654 [0.595, 0.710] | 0.668 | 0.63 / 0.52 / 0.81 |
| longitudinal (176 rows, 65 women) | baseline | 0.333 [0.333, 0.333] | 0.369 | 1.00 / 0.00 / 0.00 |
| longitudinal (176 rows, 65 women) | logreg | 0.702 [0.644, 0.757] | 0.710 | 0.68 / 0.53 / 0.90 |
| longitudinal (176 rows, 65 women) | rf | 0.688 [0.622, 0.750] | 0.688 | 0.60 / 0.65 / 0.82 |
| longitudinal (176 rows, 65 women) | gb | 0.674 [0.605, 0.736] | 0.676 | 0.57 / 0.57 / 0.88 |
| cross-sectional to longitudinal (176 rows, 65 women) | baseline | 0.333 [0.333, 0.333] | 0.341 | 0.00 / 0.00 / 1.00 |
| cross-sectional to longitudinal (176 rows, 65 women) | logreg | 0.637 [0.575, 0.699] | 0.648 | 0.58 / 0.39 / 0.93 |
| cross-sectional to longitudinal (176 rows, 65 women) | rf | 0.670 [0.602, 0.741] | 0.670 | 0.49 / 0.57 / 0.95 |
| cross-sectional to longitudinal (176 rows, 65 women) | gb | 0.692 [0.626, 0.760] | 0.693 | 0.54 / 0.59 / 0.95 |

Confusion matrix, logistic regression, pooled (271 rows, 160 women; rows = true, columns = predicted menses / peri-ovulatory / luteal):

| true \ predicted | menses | peri-ovulatory | luteal |
|---|---|---|---|
| menses | 69 | 23 | 11 |
| peri-ov | 22 | 38 | 9 |
| luteal | 3 | 8 | 88 |

A label-permutation test (200 shuffles of phase within woman, logistic regression with C = 1) gave p = 0.005 (pooled) and 0.005 (longitudinal), the smallest value attainable. The luteal phase is recognised best (progesterone is high); menses and peri-ovulatory phases are mostly confused with each other. Figure `results/fig_6b_confusion.png`.

## 5. Discussion

The study separates two questions. At the **group level** the data reproduce the authors' findings: mood and symptoms differ by phase and within-woman estradiol tracks lower DRSP symptoms. At the **individual level** (an unseen woman) phase and hormones give at most a small gain for premenstrual symptoms and none for affect or anxiety, and hormones add nothing to phase. The reasons are structural: (i) between-woman differences dominate (ICC 0.39-0.57 in the longitudinal sample) and cannot be predicted from cycle information; (ii) the estradiol effect is a within-woman deviation from the woman's own level, which a model of a new woman cannot use; (iii) estradiol and phase are strongly entangled, so there is little information in estradiol beyond phase; (iv) the group-level phase effects are small relative to the noise (marginal R2 of phase 0.085 for DRSP psychological, 0.015 for anxiety). Knowing a woman's own earlier reports helps much more than any hormone (leave-one-session-out: +0.324 [+0.152, +0.459] versus +0.034 [-0.020, +0.084]), which suggests that personalised, longitudinal tracking, not hormone assays, is the promising direction. The credibility checks support the pipeline: it finds the phase from hormones (balanced accuracy 0.70 versus chance 0.33) and finds no skill for emotion-recognition accuracy; the speed result is the one nominal exception.

## 6. Limitations

- **Small samples.** Only 65 women contribute repeated measures and 95 contribute one session; bootstrap intervals are wide, and small true effects cannot be excluded. The pooled sample mixes the two kinds of women.
- **Noisy hormones.** Salivary immunoassays have limited validity (the authors say so too); measurement error attenuates any true hormone effect, and 9 readings of estradiol are exactly zero.
- **Concurrent, not forecasting.** Hormones and mood are measured in the same session; the models describe a state, they do not predict the future. Phase labels come from the study schedule.
- **Self-report overlap.** BDI, BAI, PSST, DRSP, PANAS and STAI are all self-reports and conceptually overlap, so the strong skill of set D is 'one questionnaire explains another', not early warning.
- **Differences from the paper.** My DRSP target (psychological sum) and models differ from the authors' (their DRSP mean score per symptom type, nlme models); p-values use Wald z instead of nlme t-tests; the PSST interaction is weaker for my score; the emotion-recognition outcomes were simplified.
- **No external validation.** Everything is cross-validated within one study; a different population or assay could behave differently.
- **Multiple testing.** Several targets, models, samples and sets were examined; corrections were applied to the paired hormone comparisons only.
- **Learning project, not a clinical tool.** Nothing here should be used to diagnose, treat or advise.

## 7. Conclusion

On 271 rows, 160 women from the open data of Pletzer & Noachtar (2023), salivary estradiol and progesterone did **not** improve the prediction of positive affect, DRSP psychological symptoms or state anxiety in unseen women beyond cycle phase (0 of 27 paired comparisons improved; 15 got worse), even though phase and within-woman estradiol effects are clearly present in-sample. Person-level questionnaire scores and, above all, a woman's own history are far more informative. The from-scratch ridge machinery matches scikit-learn, its optimisation behaviour is explained by the Hessian's condition number, and the pipeline passes a negative control (with one small speed caveat) and a positive sanity check.

<!-- EXTENSION SLOT: sections for Phase 7 extensions are written by src/ext_*_report.py (numbers loaded from results/ext*.csv) and by any docs/extensions/*.md -->

## 8. Extensions

*Phase 7 extensions, one note per method in `docs/math/`; same leave-one-participant-out protocol, same seed, same women; all numbers loaded from `results/ext*.csv`.*

### 8.1 E1: Principal component analysis from scratch (SVD)

Math note: `docs/math/E1_pca.md`. PCA = standardise, centre, SVD; fitted on **training rows only** at every level of the nested CV (test rows are only projected); results in `results/extE1_*.csv`.

**Collinearity of estradiol and progesterone.** Pearson correlation (pooled, 271 rows, 160 women): raw 0.19, log1p 0.32 (Spearman 0.34); longitudinal (176 rows, 65 women) log1p 0.34. Within phase (pooled, log1p): menses 0.33 (n = 103 rows), peri-ovulatory 0.32 (n = 69), luteal 0.13 (n = 99). For two variables the first principal component captures a share (1+|r|)/2 of their variance: 0.66 here. The two hormones are only moderately correlated, so they are not strongly collinear; the strong dependence of hormones on **phase** (not on each other) is what matters for prediction.

**Explained variance, rank and conditioning** (standardised features of each set, whole sample, descriptive):

| Sample (n) | Feature set | Components 1 / 2 / 3 / ... explained-variance ratio | Numerical rank (of p) | Condition number of the correlation matrix |
|---|---|---|---|---|
| pooled (271 rows, 160 women) | B | 0.66 / 0.34 | 2 of 2 | 1.9 |
| pooled (271 rows, 160 women) | C | 0.48 / 0.32 / 0.12 / 0.07 | 4 of 4 | 6.7 |
| pooled (271 rows, 160 women) | D2 | 0.32 / 0.27 / 0.21 / 0.08 / 0.07 / 0.05 | 6 of 6 | 6.8 |
| pooled (271 rows, 160 women) | D | 0.28 / 0.25 / 0.18 / 0.12 / 0.07 / 0.06 / 0.04 | 7 of 7 | 6.8 |
| long (176 rows, 65 women) | B | 0.67 / 0.33 | 2 of 2 | 2.0 |
| long (176 rows, 65 women) | C | 0.50 / 0.31 / 0.12 / 0.06 | 4 of 4 | 8.2 |
| long (176 rows, 65 women) | D2 | 0.34 / 0.28 / 0.20 / 0.08 / 0.05 / 0.04 | 6 of 6 | 8.4 |
| long (176 rows, 65 women) | D | 0.29 / 0.26 / 0.18 / 0.13 / 0.07 / 0.05 / 0.03 | 7 of 7 | 8.4 |

All sets have full numerical rank (no exactly redundant features); no component dominates (set D pooled: first component 0.28, first three together 0.71). Loadings of the first three components for set D (pooled, 271 rows, 160 women):

| Feature | PC1 | PC2 | PC3 |
|---|---|---|---|
| peri-ov | +0.39 | -0.04 | +0.62 |
| luteal | -0.65 | +0.00 | -0.07 |
| log E2 | -0.21 | -0.10 | +0.74 |
| log P | -0.61 | -0.09 | +0.21 |
| PSST | +0.04 | -0.41 | -0.01 |
| BDI | +0.02 | -0.64 | -0.05 |
| BAI | +0.04 | -0.64 | -0.13 |

Reading the loadings (signs are arbitrary): the first component is a **luteal-phase / progesterone axis** (luteal and log P carry the largest loadings), the second a **questionnaire axis** (BDI, BAI and, less, PSST), the third a **peri-ovulatory / estradiol axis** (log E2 and the peri-ovulatory dummy).

**Identity check.** Ridge on **all** principal components must equal ridge on the raw standardised features (rotation invariance). Over 36 configurations (3 samples x 3 targets x 4 sets) the largest difference between out-of-fold predictions was 5.3e-14, and the same lambda was selected in 1.000 of the folds.

**Predictive comparison, pooled sample** (271 rows, 160 women, leave-one-woman-out, nested CV; R2 versus the mean baseline [95% CI over women]). Raw = Phase 4 models on the standardised features; PCs = the same models on principal components fitted in the training fold:

| Model | Set | positive affect | DRSP psychological | state anxiety |
|---|---|---|---|---|
| ridge, raw | B | -0.005 [-0.012, -0.000] | -0.002 [-0.020, +0.014] | -0.001 [-0.002, -0.000] |
| ridge, raw | C | +0.001 [-0.028, +0.029] | +0.037 [-0.004, +0.076] | -0.002 [-0.002, -0.001] |
| ridge, raw | D2 | +0.102 [+0.004, +0.186] | +0.339 [+0.231, +0.438] | +0.149 [+0.048, +0.245] |
| ridge, raw | D | +0.093 [-0.005, +0.175] | +0.360 [+0.247, +0.457] | +0.154 [+0.048, +0.248] |
| ridge, all PCs | B | -0.005 [-0.012, -0.000] | -0.002 [-0.020, +0.014] | -0.001 [-0.002, -0.000] |
| ridge, all PCs | C | +0.001 [-0.028, +0.029] | +0.037 [-0.004, +0.076] | -0.002 [-0.002, -0.001] |
| ridge, all PCs | D2 | +0.102 [+0.004, +0.186] | +0.339 [+0.231, +0.438] | +0.149 [+0.048, +0.245] |
| ridge, all PCs | D | +0.093 [-0.005, +0.175] | +0.360 [+0.247, +0.457] | +0.154 [+0.048, +0.248] |
| PC regression (k by nested CV) | B | -0.006 [-0.022, +0.010] | +0.007 [-0.022, +0.032] | -0.008 [-0.013, -0.005] |
| PC regression (k by nested CV) | C | +0.002 [-0.033, +0.037] | +0.030 [-0.022, +0.080] | -0.005 [-0.007, -0.003] |
| PC regression (k by nested CV) | D2 | +0.070 [-0.046, +0.170] | +0.327 [+0.204, +0.437] | +0.143 [+0.017, +0.255] |
| PC regression (k by nested CV) | D | +0.056 [-0.065, +0.162] | +0.344 [+0.213, +0.454] | +0.148 [+0.018, +0.257] |
| random forest, raw | B | -0.007 [-0.046, +0.032] | -0.027 [-0.081, +0.024] | -0.044 [-0.084, -0.006] |
| random forest, raw | C | +0.001 [-0.039, +0.040] | +0.005 [-0.049, +0.056] | -0.045 [-0.084, -0.008] |
| random forest, raw | D2 | +0.068 [-0.029, +0.156] | +0.373 [+0.260, +0.473] | +0.108 [-0.024, +0.229] |
| random forest, raw | D | +0.078 [-0.015, +0.161] | +0.377 [+0.259, +0.472] | +0.131 [-0.002, +0.248] |
| random forest, PCs | B | -0.006 [-0.060, +0.043] | -0.006 [-0.058, +0.042] | -0.031 [-0.070, +0.009] |
| random forest, PCs | C | -0.005 [-0.052, +0.040] | +0.042 [-0.015, +0.100] | -0.047 [-0.079, -0.017] |
| random forest, PCs | D2 | +0.076 [-0.022, +0.163] | +0.316 [+0.199, +0.422] | +0.093 [-0.024, +0.201] |
| random forest, PCs | D | +0.054 [-0.048, +0.142] | +0.354 [+0.220, +0.464] | +0.129 [+0.002, +0.249] |

**Paired differences in RMSE** (PC-based minus raw model, same women resampled; negative = PCs better), across 36 comparisons each (3 samples x 3 targets x 4 feature sets): PC regression versus ridge on raw features: median change +0.0073 (range -0.048 to +0.132); significant improvement in 0 and significant deterioration in 6 of 36. Random forest on PCs versus raw: median change +0.0003; significant improvement in 1 and deterioration in 3 of 36. The number of components kept by PC regression (median over folds, pooled, DRSP, set D) was 6 of 7.

**Reading.** With all components kept PCA changes nothing (identity check). Dropping components (PC regression) or rotating the data for trees gives no systematic gain (no significant improvement for PC regression). A plausible reason, not tested here, is that PCA ignores the outcome, so the directions of high predictor variance need not be the ones that carry mood information; and the two hormones are only moderately correlated, leaving little to compress. In this data PCA is a tool for understanding the structure of the predictors, not for improving prediction. Full tables for all three samples: `results/extE1_metrics_all.csv`, `results/extE1_deltas.csv`.

### 8.2 E2: Bayesian ridge regression from scratch

Math note: `docs/math/E2_bayesian_ridge.md`. Gaussian prior on the weights, exact Gaussian posterior (mean m, covariance S), predictive mean and variance for a new session, noise precision beta and prior precision alpha chosen by **evidence maximisation inside each training fold** (no inner CV). Same leave-one-woman-out folds and same women as the Phase 4 ridge, which is the comparison (its lambda is chosen by nested CV).

**Check against scikit-learn.** `BayesianRidge` with all Gamma hyper-priors set to zero (pure evidence maximisation) on the full samples (30 configurations: pooled 271 rows, 160 women and longitudinal 176 rows, 65 women x 3 targets x 5 sets): largest difference in the posterior-mean weights 4.4e-11, in the noise precision 1.7e-11 (relative), in the prior precision 2.5e-10 (relative; 22 configurations with a finite alpha). In 8 configurations the evidence prefers no signal (alpha grows without bound, weights shrunk to zero, alpha not identifiable); the weights still agree. The evidence iterations converged in all 4350 folds (non-converged: 0).

**Out-of-fold results.** Columns: R2 versus the training-mean baseline [95% CI over women] for the Bayesian ridge and for the Phase 4 ridge; change in RMSE (Bayes minus ridge, negative = Bayes better); calibration of the predictive variance: mean squared standardised error z^2 (target 1; above 1 = over-confident) and the change in mean negative log predictive density (nats per session) relative to a Gaussian with the training mean and variance (negative = better).

**Pooled sample** (271 rows, 160 women).

| Target | Set | Bayesian ridge R2_oos | Ridge R2_oos | RMSE change | mean z^2 | NLPD change vs constant Gaussian |
|---|---|---|---|---|---|---|
| positive affect | A | +0.020 [-0.006, +0.046] | +0.021 [-0.009, +0.051] | +0.000 [-0.001, +0.002] | 1.02 [0.85, 1.18] | -0.010 [-0.023, +0.004] |
| positive affect | B | -0.004 [-0.009, -0.000] | -0.005 [-0.012, -0.000] | -0.001 [-0.002, +0.001] | 1.02 [0.86, 1.18] | +0.002 [-0.000, +0.005] |
| positive affect | C | +0.003 [-0.016, +0.023] | +0.001 [-0.028, +0.029] | -0.001 [-0.006, +0.004] | 1.03 [0.86, 1.18] | -0.001 [-0.012, +0.010] |
| positive affect | D2 | +0.112 [+0.010, +0.201] | +0.102 [+0.004, +0.186] | -0.004 [-0.008, -0.001] | 1.04 [0.85, 1.23] | -0.059 [-0.109, -0.005] |
| positive affect | D | +0.103 [+0.001, +0.191] | +0.093 [-0.005, +0.175] | -0.004 [-0.008, -0.000] | 1.04 [0.86, 1.23] | -0.053 [-0.102, -0.000] |
| DRSP psychological symptoms | A | +0.049 [+0.007, +0.092] | +0.051 [+0.005, +0.095] | +0.006 [-0.009, +0.022] | 1.02 [0.84, 1.22] | -0.025 [-0.048, -0.002] |
| DRSP psychological symptoms | B | -0.001 [-0.020, +0.014] | -0.002 [-0.020, +0.014] | -0.001 [-0.025, +0.025] | 1.03 [0.86, 1.22] | +0.001 [-0.007, +0.010] |
| DRSP psychological symptoms | C | +0.033 [-0.006, +0.070] | +0.037 [-0.004, +0.076] | +0.018 [-0.011, +0.045] | 1.02 [0.85, 1.23] | -0.016 [-0.036, +0.004] |
| DRSP psychological symptoms | D2 | +0.347 [+0.226, +0.451] | +0.339 [+0.231, +0.438] | -0.043 [-0.110, +0.017] | 1.04 [0.84, 1.25] | -0.213 [-0.303, -0.134] |
| DRSP psychological symptoms | D | +0.367 [+0.247, +0.469] | +0.360 [+0.247, +0.457] | -0.041 [-0.111, +0.026] | 1.05 [0.84, 1.29] | -0.228 [-0.322, -0.139] |
| state anxiety | A | -0.000 [-0.000, -0.000] | -0.001 [-0.001, -0.000] | -0.000 [-0.000, -0.000] | 1.02 [0.86, 1.20] | +0.000 [-0.000, +0.000] |
| state anxiety | B | -0.000 [-0.000, -0.000] | -0.001 [-0.002, -0.000] | -0.000 [-0.000, -0.000] | 1.02 [0.86, 1.20] | +0.000 [-0.000, +0.000] |
| state anxiety | C | -0.000 [-0.000, -0.000] | -0.002 [-0.002, -0.001] | -0.000 [-0.001, -0.000] | 1.02 [0.86, 1.20] | +0.000 [-0.000, +0.000] |
| state anxiety | D2 | +0.153 [+0.047, +0.256] | +0.149 [+0.048, +0.245] | -0.001 [-0.003, +0.001] | 1.03 [0.87, 1.22] | -0.083 [-0.148, -0.022] |
| state anxiety | D | +0.158 [+0.042, +0.262] | +0.154 [+0.048, +0.248] | -0.001 [-0.004, +0.003] | 1.04 [0.88, 1.23] | -0.086 [-0.154, -0.019] |

**Longitudinal sample** (176 rows, 65 women).

| Target | Set | Bayesian ridge R2_oos | Ridge R2_oos | RMSE change | mean z^2 | NLPD change vs constant Gaussian |
|---|---|---|---|---|---|---|
| positive affect | A | +0.024 [-0.006, +0.052] | +0.027 [-0.009, +0.062] | +0.001 [-0.002, +0.005] | 1.03 [0.83, 1.24] | -0.012 [-0.026, +0.004] |
| positive affect | B | -0.009 [-0.016, -0.003] | -0.005 [-0.013, +0.001] | +0.001 [+0.000, +0.003] | 1.04 [0.84, 1.24] | +0.005 [+0.001, +0.008] |
| positive affect | C | -0.004 [-0.022, +0.013] | -0.007 [-0.029, +0.016] | -0.001 [-0.006, +0.004] | 1.04 [0.84, 1.25] | +0.002 [-0.006, +0.011] |
| positive affect | D2 | +0.058 [-0.076, +0.167] | +0.061 [-0.040, +0.145] | +0.001 [-0.010, +0.014] | 1.07 [0.84, 1.31] | -0.031 [-0.089, +0.034] |
| positive affect | D | +0.045 [-0.091, +0.151] | +0.037 [-0.069, +0.117] | -0.003 [-0.017, +0.009] | 1.08 [0.84, 1.31] | -0.023 [-0.081, +0.041] |
| DRSP psychological symptoms | A | +0.051 [+0.005, +0.101] | +0.055 [+0.004, +0.110] | +0.020 [-0.005, +0.049] | 1.03 [0.80, 1.31] | -0.025 [-0.050, +0.000] |
| DRSP psychological symptoms | B | +0.002 [-0.037, +0.037] | -0.003 [-0.047, +0.030] | -0.027 [-0.088, +0.026] | 1.05 [0.82, 1.31] | -0.000 [-0.017, +0.019] |
| DRSP psychological symptoms | C | +0.032 [-0.014, +0.079] | +0.033 [-0.015, +0.084] | +0.004 [-0.047, +0.051] | 1.04 [0.81, 1.32] | -0.015 [-0.038, +0.012] |
| DRSP psychological symptoms | D2 | +0.271 [+0.109, +0.409] | +0.249 [+0.114, +0.365] | -0.121 [-0.316, +0.038] | 1.08 [0.84, 1.33] | -0.160 [-0.279, -0.052] |
| DRSP psychological symptoms | D | +0.279 [+0.112, +0.421] | +0.259 [+0.111, +0.382] | -0.114 [-0.305, +0.048] | 1.10 [0.84, 1.38] | -0.165 [-0.282, -0.045] |
| state anxiety | A | +0.000 [-0.000, +0.000] | -0.004 [-0.020, +0.009] | -0.001 [-0.004, +0.002] | 1.04 [0.83, 1.26] | +0.000 [-0.000, +0.001] |
| state anxiety | B | -0.000 [-0.000, +0.000] | -0.002 [-0.004, +0.000] | -0.000 [-0.001, +0.000] | 1.04 [0.83, 1.26] | +0.000 [-0.000, +0.001] |
| state anxiety | C | -0.000 [-0.000, +0.000] | -0.001 [-0.005, +0.002] | -0.000 [-0.001, +0.000] | 1.04 [0.83, 1.26] | +0.000 [-0.000, +0.001] |
| state anxiety | D2 | +0.120 [-0.030, +0.252] | +0.108 [-0.023, +0.219] | -0.003 [-0.009, +0.003] | 1.07 [0.85, 1.31] | -0.064 [-0.149, +0.015] |
| state anxiety | D | +0.131 [-0.050, +0.278] | +0.127 [-0.016, +0.244] | -0.001 [-0.010, +0.009] | 1.10 [0.87, 1.33] | -0.070 [-0.173, +0.033] |

**Cross-sectional to longitudinal sample** (training 95 rows, 95 women; test 176 rows, 65 women).

| Target | Set | Bayesian ridge R2_oos | Ridge R2_oos | RMSE change | mean z^2 | NLPD change vs constant Gaussian |
|---|---|---|---|---|---|---|
| positive affect | A | +0.005 [-0.004, +0.016] | +0.012 [-0.012, +0.037] | +0.003 [-0.003, +0.008] | 1.36 [1.10, 1.62] | -0.001 [-0.008, +0.005] |
| positive affect | B | -0.001 [-0.013, +0.011] | -0.010 [-0.044, +0.023] | -0.004 [-0.012, +0.005] | 1.37 [1.11, 1.63] | +0.003 [-0.005, +0.012] |
| positive affect | C | +0.003 [-0.008, +0.015] | +0.006 [-0.022, +0.035] | +0.001 [-0.006, +0.008] | 1.36 [1.10, 1.63] | +0.000 [-0.008, +0.008] |
| positive affect | D2 | +0.111 [+0.007, +0.205] | +0.111 [+0.005, +0.206] | +0.000 [-0.001, +0.001] | 1.44 [1.15, 1.75] | -0.048 [-0.120, +0.022] |
| positive affect | D | +0.108 [+0.011, +0.197] | +0.107 [+0.004, +0.201] | -0.000 [-0.003, +0.003] | 1.42 [1.13, 1.73] | -0.048 [-0.116, +0.019] |
| DRSP psychological symptoms | A | +0.040 [+0.017, +0.068] | +0.054 [+0.016, +0.099] | +0.069 [-0.004, +0.146] | 1.45 [1.13, 1.84] | -0.022 [-0.040, -0.005] |
| DRSP psychological symptoms | B | +0.000 [-0.000, +0.000] | +0.001 [-0.001, +0.002] | +0.003 [-0.003, +0.010] | 1.49 [1.17, 1.87] | +0.003 [+0.001, +0.005] |
| DRSP psychological symptoms | C | +0.000 [+0.000, +0.000] | +0.022 [+0.008, +0.040] | +0.104 [+0.036, +0.180] | 1.49 [1.17, 1.87] | +0.003 [+0.001, +0.005] |
| DRSP psychological symptoms | D2 | +0.244 [+0.099, +0.378] | +0.252 [+0.116, +0.378] | +0.045 [-0.015, +0.108] | 2.46 [1.96, 2.99] | +0.096 [-0.099, +0.291] |
| DRSP psychological symptoms | D | +0.288 [+0.146, +0.411] | +0.292 [+0.158, +0.409] | +0.025 [-0.028, +0.079] | 2.36 [1.88, 2.88] | +0.037 [-0.151, +0.238] |
| state anxiety | A | -0.000 [-0.000, +0.000] | -0.002 [-0.004, +0.000] | -0.000 [-0.001, +0.000] | 1.46 [1.19, 1.75] | +0.002 [+0.001, +0.004] |
| state anxiety | B | -0.000 [-0.000, +0.000] | -0.001 [-0.003, +0.001] | -0.000 [-0.001, +0.000] | 1.46 [1.19, 1.75] | +0.002 [+0.001, +0.004] |
| state anxiety | C | -0.000 [-0.000, +0.000] | -0.003 [-0.006, +0.000] | -0.001 [-0.001, +0.000] | 1.46 [1.19, 1.75] | +0.002 [+0.001, +0.004] |
| state anxiety | D2 | +0.114 [-0.010, +0.227] | +0.118 [-0.001, +0.226] | +0.001 [-0.001, +0.003] | 1.62 [1.33, 1.93] | -0.030 [-0.137, +0.070] |
| state anxiety | D | +0.121 [+0.001, +0.229] | +0.128 [+0.023, +0.226] | +0.002 [-0.002, +0.006] | 1.57 [1.29, 1.87] | -0.043 [-0.144, +0.050] |

**Accuracy: Bayesian ridge and ridge are practically the same.** Across the 45 comparisons the median RMSE change is -0.0003 and the largest absolute change 0.121; the interval excludes 0 in 7 (5 in favour of the Bayesian ridge, 2 in favour of ridge), differences that are tiny in size. This is expected: both give the same functional form, only the way the shrinkage is chosen differs (evidence versus cross-validation). As an example of how the two ways of choosing the shrinkage differ, in the pooled DRSP model with set D the evidence chose weaker shrinkage than cross-validation (equivalent lambda 0.053 versus 0.158, median over folds) and, when there is nothing to learn (e.g. anxiety from phase and hormones), shrinks the weights to zero: 13 of 45 configurations have a median effective number of parameters below 0.01. Choosing the shrinkage by the evidence needs no inner cross-validation loop.

**Calibration of the predictive variance.** Within the leave-one-woman-out samples (pooled and longitudinal; 30 configurations) the mean z^2 lies between 1.02 and 1.10, and the 95% interval contains 1 in 30 of them: the predictive variance is roughly calibrated on average. It is, however, almost constant across sessions (ratio of the largest to the smallest predictive sd within a configuration at most 1.14), because the noise term dominates and the weights are uncertain only in small ways, so it carries little session-specific information. The log predictive density improves over a constant Gaussian only for the questionnaire-based sets (pooled, DRSP, set D: -0.228 [-0.322, -0.139] nats per session); for phase and hormone sets it is -0.016 [-0.036, +0.004] (set C, DRSP), i.e. no gain. **When the test women come from a different study sample (training on the 95 cross-sectional women, test on the 65 longitudinal women) the model is over-confident**: mean z^2 between 1.36 and 2.46; the noise variance learned on one group underestimates the variability in the other: the standard deviation of DRSP psychological symptoms is 7.8 in the training sample but 9.4 in the test sample (n = 95 rows, 95 women and 176 rows, 65 women). This is a distribution-shift effect: the prior and the evidence cannot know about spread that the training sample does not contain. Coverage of the resulting intervals is examined in E4.

**Limits.** Rows are treated as independent although sessions of one woman are correlated, so the uncertainty about the *woman-level* level is folded into the noise; the intercept is fixed at the training mean; the predictive distribution is Gaussian by assumption.

### 8.3 E3: Gaussian Bayes classifier for the cycle phase

Math note: `docs/math/E3_gaussian_bayes_classifier.md`. Features: log1p estradiol and log1p progesterone. Class priors, class means and covariances are estimated in the **training fold only**; Bayes' theorem gives the posterior; the prediction is the posterior mode. Primary model: class-specific full covariances (QDA) with empirical priors. Variants: shared covariance (LDA), diagonal covariance (naive Bayes), and equal priors. Same folds as the Phase 6b logistic regression (the comparison), same women, same bootstrap seed. There are no hyper-parameters, so no inner CV.

**Check against scikit-learn.** On the full pooled sample (271 rows, 160 women) the posterior probabilities of my full-covariance classifier differ from `QuadraticDiscriminantAnalysis` by at most 1.0e-15.

**What the model learned** (pooled, 271 rows, 160 women; descriptive fit on all rows): class priors menses/peri-ovulatory/luteal 0.38/0.25/0.37.

| Class | mean log E2 | mean log P | SD log E2 | SD log P | correlation in class |
|---|---|---|---|---|---|
| menses | 0.45 | 3.05 | 0.28 | 1.03 | 0.33 |
| peri-ov | 0.71 | 3.34 | 0.27 | 0.97 | 0.32 |
| luteal | 0.64 | 4.78 | 0.25 | 0.59 | 0.13 |

Progesterone separates the luteal class (higher mean and smaller spread); estradiol differs mostly between menses and the other two classes.

**Out-of-fold performance** (balanced accuracy [95% CI over women]; chance 0.333; difference to logistic regression in balanced accuracy [95% CI], same women resampled; mean log loss and its change relative to a prior-only classifier that ignores the hormones, in nats):

**Pooled** (271 rows, 160 women):

| Model | Balanced accuracy | Accuracy | Difference to logistic regression | Log loss (change vs prior-only) |
|---|---|---|---|---|
| Gaussian Bayes, full cov., empirical priors (primary) | 0.650 [0.600, 0.700] | 0.683 | -0.053 [-0.090, -0.016] | 0.718 (-0.369 [-0.440, -0.290]) |
| Gaussian Bayes, full cov., equal priors | 0.688 [0.630, 0.741] | 0.701 | -0.015 [-0.047, +0.015] | 0.728 (-0.359 [-0.433, -0.276]) |
| Gaussian Bayes, shared cov. (LDA), empirical priors | 0.650 [0.600, 0.700] | 0.683 | -0.053 [-0.091, -0.015] | 0.698 (-0.389 [-0.445, -0.330]) |
| Gaussian Bayes, shared cov., equal priors | 0.684 [0.633, 0.734] | 0.697 | -0.020 [-0.045, +0.006] | 0.706 (-0.381 [-0.441, -0.317]) |
| Gaussian Bayes, diagonal cov., empirical priors | 0.655 [0.605, 0.704] | 0.694 | -0.048 [-0.088, -0.008] | 0.729 (-0.358 [-0.431, -0.277]) |
| Gaussian Bayes, diagonal cov., equal priors | 0.705 [0.652, 0.757] | 0.720 | +0.002 [-0.033, +0.036] | 0.736 (-0.351 [-0.426, -0.267]) |
| logistic regression (Phase 6b) | 0.703 [0.651, 0.753] | 0.720 | - | - |

**Longitudinal** (176 rows, 65 women):

| Model | Balanced accuracy | Accuracy | Difference to logistic regression | Log loss (change vs prior-only) |
|---|---|---|---|---|
| Gaussian Bayes, full cov., empirical priors (primary) | 0.675 [0.618, 0.730] | 0.693 | -0.027 [-0.058, +0.004] | 0.719 (-0.376 [-0.463, -0.276]) |
| Gaussian Bayes, full cov., equal priors | 0.683 [0.624, 0.738] | 0.693 | -0.020 [-0.043, +0.000] | 0.721 (-0.374 [-0.461, -0.271]) |
| Gaussian Bayes, shared cov. (LDA), empirical priors | 0.688 [0.629, 0.745] | 0.705 | -0.015 [-0.057, +0.025] | 0.692 (-0.403 [-0.475, -0.323]) |
| Gaussian Bayes, shared cov., equal priors | 0.692 [0.633, 0.750] | 0.699 | -0.010 [-0.045, +0.024] | 0.695 (-0.400 [-0.474, -0.318]) |
| Gaussian Bayes, diagonal cov., empirical priors | 0.669 [0.618, 0.722] | 0.688 | -0.033 [-0.077, +0.008] | 0.725 (-0.370 [-0.458, -0.268]) |
| Gaussian Bayes, diagonal cov., equal priors | 0.671 [0.611, 0.729] | 0.682 | -0.031 [-0.074, +0.008] | 0.726 (-0.369 [-0.458, -0.267]) |
| logistic regression (Phase 6b) | 0.702 [0.645, 0.757] | 0.710 | - | - |

**Cross-sectional to longitudinal** (training 95 rows, 95 women; test 176 rows, 65 women):

| Model | Balanced accuracy | Accuracy | Difference to logistic regression | Log loss (change vs prior-only) |
|---|---|---|---|---|
| Gaussian Bayes, full cov., empirical priors (primary) | 0.594 [0.539, 0.651] | 0.614 | -0.043 [-0.090, +0.003] | 0.879 (-0.244 [-0.366, -0.112]) |
| Gaussian Bayes, full cov., equal priors | 0.650 [0.587, 0.709] | 0.653 | +0.013 [-0.025, +0.051] | 0.813 (-0.311 [-0.428, -0.184]) |
| Gaussian Bayes, shared cov. (LDA), empirical priors | 0.575 [0.525, 0.625] | 0.597 | -0.062 [-0.116, -0.010] | 0.871 (-0.253 [-0.337, -0.161]) |
| Gaussian Bayes, shared cov., equal priors | 0.629 [0.570, 0.689] | 0.636 | -0.008 [-0.034, +0.021] | 0.795 (-0.329 [-0.410, -0.242]) |
| Gaussian Bayes, diagonal cov., empirical priors | 0.586 [0.531, 0.642] | 0.608 | -0.051 [-0.097, -0.007] | 0.843 (-0.281 [-0.373, -0.183]) |
| Gaussian Bayes, diagonal cov., equal priors | 0.660 [0.597, 0.721] | 0.665 | +0.023 [-0.015, +0.060] | 0.780 (-0.344 [-0.436, -0.247]) |
| logistic regression (Phase 6b) | 0.637 [0.576, 0.699] | 0.648 | - | - |

**Findings.** (1) The primary Gaussian Bayes classifier is clearly above chance (balanced accuracy 0.650 [0.600, 0.700] pooled, 0.675 [0.618, 0.730] longitudinal, versus 0.333), and its posterior probabilities are informative: the log loss is lower than that of the prior-only classifier (pooled change -0.369 [-0.440, -0.290] nats). (2) Compared with logistic regression the primary variant is lower in balanced accuracy (pooled -0.053 [-0.090, -0.016]), because empirical priors favour the common classes and the peri-ovulatory class, the smallest, is rarely predicted (recall 0.36, against 0.57 with equal priors). With equal priors, which matches the balanced-accuracy criterion and the class-balanced logistic regression, the difference to logistic regression is -0.015 [-0.047, +0.015] (pooled) and -0.020 [-0.043, +0.000] (longitudinal): no clear difference (the longitudinal interval only just reaches 0). (3) The covariance structure matters little (shared, full and diagonal give similar balanced accuracy within each prior setting). Across the 18 variant-sample combinations the interval of the difference to logistic regression excludes 0 in 5; none favours the Gaussian Bayes classifier (0 comparisons).

Confusion matrices, pooled (271 rows, 160 women; rows = true class, columns = predicted menses / peri-ovulatory / luteal):

| True class | Primary (empirical priors) | Equal priors |
|---|---|---|
| menses | 72 / 14 / 17 | 65 / 21 / 17 |
| peri-ov | 31 / 25 / 13 | 17 / 39 / 13 |
| luteal | 5 / 6 / 88 | 1 / 12 / 86 |

As for logistic regression (Phase 6b) the luteal phase is recognised best and menses and peri-ovulation are confused with each other. Leakage checks as before: folds are by woman (asserted), priors, means and covariances come from the training fold only, and the cross-sectional and longitudinal women are disjoint.

### 8.4 E4: Prediction intervals and empirical coverage

Math note: `docs/math/E4_prediction_intervals.md`. Three central intervals at 80%, 90% and 95%, evaluated on held-out women: (1) the **Bayesian predictive interval** of E2 (Gaussian); (2) a **jackknife-conformal interval** for ridge, built from the errors of leave-one-woman-out refits inside each training set; (3) a **constant Gaussian reference** (training mean +/- z x training SD, no predictors). Feature sets A (phase), C (phase + hormones), D (C + questionnaires); leave-one-woman-out for the pooled and longitudinal samples, and training on the cross-sectional women with testing on the longitudinal women. Coverage = share of held-out sessions inside the interval, with 95% CIs from resampling women; 'off' = the CI excludes the nominal level.

Sample sizes (test rows, women): pooled 271 rows, 160 women; longitudinal 176 rows, 65 women; cross-sectional to longitudinal: training 95 rows, 95 women, test 176 rows, 65 women.

**Pooled sample** (271 rows, 160 women). Empirical coverage at the nominal levels 0.80 / 0.90 / 0.95, and mean width at 90% relative to the constant reference (1.00):

| Target | Set | Bayesian predictive | Jackknife conformal | Constant Gaussian | Width at 90% (Bayes / conformal) |
|---|---|---|---|---|---|
| positive affect | A | 0.78 / 0.89 / 0.96 | 0.80 / 0.90 / 0.95 | 0.80 / 0.87 / 0.96 | 0.99 / 1.03 |
| positive affect | C | 0.79 / 0.88 / 0.95 | 0.80 / 0.90 / 0.95 | 0.80 / 0.87 / 0.96 | 0.99 / 1.03 |
| positive affect | D | 0.79 / 0.89 / 0.94 | 0.81 / 0.90 / 0.95 | 0.80 / 0.87 / 0.96 | 0.94 / 0.97 |
| DRSP psychological symptoms | A | 0.79 / 0.89 / 0.96 | 0.80 / 0.91 / 0.95 | 0.81 / 0.93 / 0.96 | 0.97 / 0.98 |
| DRSP psychological symptoms | C | 0.78 / 0.91 / 0.96 | 0.82 / 0.90 / 0.95 | 0.81 / 0.93 / 0.96 | 0.98 / 0.97 |
| DRSP psychological symptoms | D | 0.80 / 0.89 / 0.94 | 0.80 / 0.90 / 0.95 | 0.81 / 0.93 / 0.96 | 0.78 / 0.81 |
| state anxiety | A | 0.81 / 0.91 / 0.95 | 0.81 / 0.91 / 0.95 | 0.81 / 0.91 / 0.95 | 1.00 / 0.98 |
| state anxiety | C | 0.81 / 0.91 / 0.95 | 0.81 / 0.91 / 0.95 | 0.81 / 0.91 / 0.95 | 1.00 / 0.98 |
| state anxiety | D | 0.81 / 0.90 / 0.96 | 0.80 / 0.90 / 0.95 | 0.81 / 0.91 / 0.95 | 0.91 / 0.93 |

**Longitudinal sample** (176 rows, 65 women). Empirical coverage at the nominal levels 0.80 / 0.90 / 0.95, and mean width at 90% relative to the constant reference (1.00):

| Target | Set | Bayesian predictive | Jackknife conformal | Constant Gaussian | Width at 90% (Bayes / conformal) |
|---|---|---|---|---|---|
| positive affect | A | 0.77 / 0.89 / 0.99 | 0.80 / 0.90 / 0.95 | 0.78 / 0.89 / 0.97 | 0.99 / 1.03 |
| positive affect | C | 0.77 / 0.88 / 0.98 | 0.79 / 0.89 / 0.95 | 0.78 / 0.89 / 0.97 | 0.99 / 1.02 |
| positive affect | D | 0.77 / 0.89 / 0.97 | 0.80 / 0.90 / 0.95 | 0.78 / 0.89 / 0.97 | 0.95 / 0.98 |
| DRSP psychological symptoms | A | 0.79 / 0.91 / 0.95 | 0.81 / 0.90 / 0.95 | 0.81 / 0.93 / 0.96 | 0.97 / 0.94 |
| DRSP psychological symptoms | C | 0.79 / 0.92 / 0.94 | 0.80 / 0.90 / 0.95 | 0.81 / 0.93 / 0.96 | 0.98 / 0.93 |
| DRSP psychological symptoms | D | 0.82 / 0.88 / 0.94 | 0.80 / 0.90 / 0.95 | 0.81 / 0.93 / 0.96 | 0.82 / 0.85 |
| state anxiety | A | 0.82 / 0.91 / 0.96 | 0.82 / 0.90 / 0.95 | 0.82 / 0.91 / 0.96 | 1.00 / 0.98 |
| state anxiety | C | 0.82 / 0.91 / 0.96 | 0.81 / 0.90 / 0.95 | 0.82 / 0.91 / 0.96 | 1.00 / 0.97 |
| state anxiety | D | 0.77 / 0.90 / 0.94 | 0.79 / 0.91 / 0.95 | 0.82 / 0.91 / 0.96 | 0.90 / 0.89 |

**Cross-sectional to longitudinal sample** (training 95 rows, 95 women; test 176 rows, 65 women). Empirical coverage at the nominal levels 0.80 / 0.90 / 0.95, and mean width at 90% relative to the constant reference (1.00):

| Target | Set | Bayesian predictive | Jackknife conformal | Constant Gaussian | Width at 90% (Bayes / conformal) |
|---|---|---|---|---|---|
| positive affect | A | 0.70 / 0.80 / 0.90 | 0.73 / 0.85 / 0.94 | 0.69 / 0.80 / 0.91 | 0.99 / 1.03 |
| positive affect | C | 0.70 / 0.80 / 0.90 | 0.74 / 0.80 / 0.95 | 0.69 / 0.80 / 0.91 | 0.99 / 1.01 |
| positive affect | D | 0.70 / 0.81 / 0.89 | 0.70 / 0.81 / 0.98 | 0.69 / 0.80 / 0.91 | 0.92 / 0.93 |
| DRSP psychological symptoms | A | 0.69 / 0.87 / 0.92 | 0.74 / 0.89 / 0.93 | 0.72 / 0.86 / 0.93 | 0.99 / 1.07 |
| DRSP psychological symptoms | C | 0.72 / 0.86 / 0.92 | 0.74 / 0.89 / 0.92 | 0.72 / 0.86 / 0.93 | 0.99 / 1.03 |
| DRSP psychological symptoms | D | 0.59 / 0.71 / 0.81 | 0.61 / 0.71 / 0.84 | 0.72 / 0.86 / 0.93 | 0.66 / 0.67 |
| state anxiety | A | 0.62 / 0.85 / 0.91 | 0.62 / 0.87 / 0.98 | 0.66 / 0.85 / 0.91 | 0.99 / 1.03 |
| state anxiety | C | 0.62 / 0.85 / 0.91 | 0.62 / 0.85 / 0.98 | 0.66 / 0.85 / 0.91 | 0.99 / 1.02 |
| state anxiety | D | 0.68 / 0.79 / 0.90 | 0.68 / 0.82 / 0.93 | 0.66 / 0.85 / 0.91 | 0.90 / 0.94 |

**Where is coverage off?** A cell is one configuration (target x feature set) at one nominal level; 'ok' = the 95% CI of the coverage contains the nominal level, 'under' = the CI lies below it (intervals too narrow), 'over' = above it (too wide):

| Sample | Method | cells | ok | under | over |
|---|---|---|---|---|---|
| pooled | Bayesian predictive | 27 | 27 | 0 | 0 |
| pooled | jackknife conformal | 27 | 27 | 0 | 0 |
| pooled | constant Gaussian (no predictors) | 27 | 27 | 0 | 0 |
| longitudinal | Bayesian predictive | 27 | 25 | 0 | 2 |
| longitudinal | jackknife conformal | 27 | 27 | 0 | 0 |
| longitudinal | constant Gaussian (no predictors) | 27 | 27 | 0 | 0 |
| cross-sectional to longitudinal | Bayesian predictive | 27 | 8 | 19 | 0 |
| cross-sectional to longitudinal | jackknife conformal | 27 | 14 | 10 | 3 |
| cross-sectional to longitudinal | constant Gaussian (no predictors) | 27 | 12 | 15 | 0 |

**Findings.** (1) *Within the same population (leave-one-woman-out) coverage is close to nominal for both the Bayesian and the conformal interval*: median coverage at the 90% level 0.893 (Bayesian) and 0.900 (conformal) in the pooled sample, 0.898 and 0.903 in the longitudinal sample; the cells that are off within these two samples are listed in `results/extE4_coverage.csv` (`covers_nominal == False`). (2) *Across populations coverage is off.* Trained on the cross-sectional women and tested on the longitudinal women, the intervals are too narrow: median coverage at the nominal 80% level 0.693 (Bayesian) and 0.705 (conformal), at 90% 0.812 and 0.847. The same happens to the constant reference interval (0.852 at 90%), so the cause is the difference between the two samples (the longitudinal women vary more in DRSP, see E2), not the way the interval is built; the conformal guarantee needs exchangeable women. At the 95% level the conformal interval of some cross-sample configurations is conservative (over-coverage, see the table above). (3) *Intervals are only narrower when predictors carry information.* Averaged over targets, the 90% interval is 0.88 times the width of the no-predictor interval for set D (pooled), but 0.99 for phase alone and 0.99 for phase plus hormones: with the same coverage, phase and hormones make the interval practically no narrower, i.e. they do not make the predictions more precise, while the questionnaires do (narrower by 12%). The two methods give very similar widths.

**Honest summary.** Intervals for new women from the same study population are well calibrated, but they are wide: the predictions are useful as a rough range, not as a precise forecast, in line with the low R2 of the point predictions. They should not be transferred to women from a different sample without re-calibration.

### 8.5 E5: Ablation study and learning curves

Math note: `docs/math/E5_ablation_and_learning_curves.md`. **Ablation:** start from the full set D (phase, hormones, PSST, BDI, BAI), drop one feature group at a time, refit the whole nested-CV pipeline and compare. Change in RMSE = without group minus full set (positive = the group helped; negative = the model was better without it), with 95% CIs from resampling the same women. 'Alone' = the group by itself. Ridge in all three samples; random forest for the drop-one sets in the pooled and longitudinal samples. **Learning curve:** leave-one-woman-out where the model may only use k randomly chosen other women.

**Ablation, ridge, pooled sample** (271 rows, 160 women). R2 versus the mean baseline [95% CI]:

| Target | Group | R2 without the group | Change in RMSE when dropped [CI] | R2 of the group alone |
|---|---|---|---|---|
| positive affect | (full set D) | +0.093 [-0.005, +0.175] | - | - |
| | phase | +0.080 [-0.004, +0.153] | +0.005 [-0.003, +0.013] | +0.021 [-0.009, +0.051] |
| | hormones (E2, P) | +0.105 [+0.009, +0.185] | -0.005 [-0.010, -0.001] | -0.005 [-0.012, -0.000] |
| | PSST | +0.102 [+0.004, +0.186] | -0.004 [-0.007, -0.000] | -0.005 [-0.022, +0.011] |
| | BDI | +0.047 [-0.009, +0.100] | +0.018 [-0.016, +0.044] | +0.097 [-0.007, +0.181] |
| | BAI | +0.100 [-0.002, +0.184] | -0.003 [-0.009, +0.003] | +0.052 [+0.001, +0.102] |
| | PSST + BDI + BAI together | - | - | +0.082 [-0.011, +0.158] |
| DRSP psychological symptoms | (full set D) | +0.360 [+0.247, +0.457] | - | - |
| | phase | +0.331 [+0.213, +0.429] | +0.155 [+0.026, +0.282] | +0.051 [+0.005, +0.095] |
| | hormones (E2, P) | +0.368 [+0.257, +0.465] | -0.048 [-0.114, +0.018] | -0.002 [-0.020, +0.014] |
| | PSST | +0.339 [+0.231, +0.438] | +0.112 [-0.103, +0.326] | +0.080 [-0.007, +0.167] |
| | BDI | +0.275 [+0.175, +0.371] | +0.452 [-0.154, +0.975] | +0.282 [+0.165, +0.393] |
| | BAI | +0.350 [+0.238, +0.445] | +0.051 [-0.108, +0.215] | +0.207 [+0.106, +0.300] |
| | PSST + BDI + BAI together | - | - | +0.317 [+0.190, +0.421] |
| state anxiety | (full set D) | +0.154 [+0.048, +0.248] | - | - |
| | phase | +0.162 [+0.055, +0.256] | -0.002 [-0.003, -0.001] | -0.001 [-0.001, -0.000] |
| | hormones (E2, P) | +0.158 [+0.047, +0.255] | -0.001 [-0.003, +0.001] | -0.001 [-0.002, -0.000] |
| | PSST | +0.149 [+0.048, +0.245] | +0.001 [-0.007, +0.008] | +0.040 [-0.029, +0.100] |
| | BDI | +0.068 [+0.002, +0.128] | +0.019 [+0.000, +0.037] | +0.171 [+0.063, +0.272] |
| | BAI | +0.162 [+0.058, +0.255] | -0.002 [-0.005, +0.001] | +0.071 [+0.004, +0.134] |
| | PSST + BDI + BAI together | - | - | +0.169 [+0.055, +0.266] |

**Ablation, ridge, longitudinal sample** (176 rows, 65 women). R2 versus the mean baseline [95% CI]:

| Target | Group | R2 without the group | Change in RMSE when dropped [CI] | R2 of the group alone |
|---|---|---|---|---|
| positive affect | (full set D) | +0.037 [-0.069, +0.117] | - | - |
| | phase | +0.033 [-0.056, +0.101] | +0.001 [-0.007, +0.009] | +0.027 [-0.009, +0.062] |
| | hormones (E2, P) | +0.035 [-0.113, +0.134] | +0.001 [-0.012, +0.018] | -0.005 [-0.013, +0.001] |
| | PSST | +0.061 [-0.040, +0.145] | -0.010 [-0.016, -0.004] | -0.005 [-0.014, +0.003] |
| | BDI | +0.035 [-0.028, +0.092] | +0.001 [-0.034, +0.027] | +0.048 [-0.070, +0.138] |
| | BAI | +0.053 [-0.031, +0.128] | -0.007 [-0.022, +0.008] | +0.042 [-0.027, +0.106] |
| | PSST + BDI + BAI together | - | - | +0.017 [-0.114, +0.101] |
| DRSP psychological symptoms | (full set D) | +0.259 [+0.111, +0.382] | - | - |
| | phase | +0.228 [+0.079, +0.352] | +0.166 [+0.018, +0.303] | +0.055 [+0.004, +0.110] |
| | hormones (E2, P) | +0.249 [+0.087, +0.379] | +0.050 [-0.129, +0.227] | -0.003 [-0.047, +0.030] |
| | PSST | +0.249 [+0.114, +0.365] | +0.055 [-0.238, +0.333] | +0.040 [-0.055, +0.129] |
| | BDI | +0.195 [+0.083, +0.301] | +0.343 [-0.475, +1.032] | +0.205 [+0.053, +0.337] |
| | BAI | +0.277 [+0.136, +0.403] | -0.102 [-0.275, +0.074] | +0.135 [+0.031, +0.228] |
| | PSST + BDI + BAI together | - | - | +0.179 [+0.012, +0.307] |
| state anxiety | (full set D) | +0.127 [-0.016, +0.244] | - | - |
| | phase | +0.118 [-0.038, +0.242] | +0.002 [-0.002, +0.007] | -0.004 [-0.020, +0.009] |
| | hormones (E2, P) | +0.131 [-0.026, +0.257] | -0.001 [-0.006, +0.004] | -0.002 [-0.004, +0.000] |
| | PSST | +0.108 [-0.023, +0.219] | +0.004 [-0.008, +0.017] | +0.054 [-0.022, +0.130] |
| | BDI | +0.078 [-0.002, +0.159] | +0.011 [-0.017, +0.035] | +0.127 [-0.029, +0.252] |
| | BAI | +0.145 [+0.014, +0.258] | -0.004 [-0.009, -0.000] | +0.069 [-0.005, +0.139] |
| | PSST + BDI + BAI together | - | - | +0.124 [-0.034, +0.249] |

**Ablation, ridge, cross-sectional to longitudinal sample** (training 95 rows, 95 women; test 176 rows, 65 women). R2 versus the mean baseline [95% CI]:

| Target | Group | R2 without the group | Change in RMSE when dropped [CI] | R2 of the group alone |
|---|---|---|---|---|
| positive affect | (full set D) | +0.107 [+0.004, +0.201] | - | - |
| | phase | +0.085 [-0.036, +0.192] | +0.009 [+0.000, +0.019] | +0.012 [-0.012, +0.037] |
| | hormones (E2, P) | +0.105 [+0.001, +0.199] | +0.001 [-0.009, +0.010] | -0.010 [-0.044, +0.023] |
| | PSST | +0.111 [+0.005, +0.206] | -0.002 [-0.007, +0.003] | +0.002 [-0.002, +0.005] |
| | BDI | +0.044 [-0.009, +0.095] | +0.026 [-0.015, +0.065] | +0.087 [-0.059, +0.208] |
| | BAI | +0.108 [+0.002, +0.204] | -0.000 [-0.006, +0.005] | +0.042 [+0.005, +0.078] |
| | PSST + BDI + BAI together | - | - | +0.088 [-0.036, +0.194] |
| DRSP psychological symptoms | (full set D) | +0.292 [+0.158, +0.409] | - | - |
| | phase | +0.267 [+0.123, +0.396] | +0.141 [-0.009, +0.283] | +0.054 [+0.016, +0.099] |
| | hormones (E2, P) | +0.313 [+0.181, +0.431] | -0.119 [-0.212, -0.031] | +0.001 [-0.001, +0.002] |
| | PSST | +0.252 [+0.116, +0.378] | +0.223 [-0.052, +0.497] | +0.051 [-0.114, +0.202] |
| | BDI | +0.201 [+0.093, +0.299] | +0.492 [-0.107, +1.225] | +0.235 [+0.065, +0.397] |
| | BAI | +0.304 [+0.150, +0.436] | -0.064 [-0.329, +0.184] | +0.130 [+0.032, +0.222] |
| | PSST + BDI + BAI together | - | - | +0.271 [+0.116, +0.404] |
| state anxiety | (full set D) | +0.128 [+0.023, +0.226] | - | - |
| | phase | +0.158 [+0.059, +0.251] | -0.007 [-0.013, -0.001] | -0.002 [-0.004, +0.000] |
| | hormones (E2, P) | +0.129 [+0.010, +0.236] | -0.000 [-0.005, +0.004] | -0.001 [-0.003, +0.001] |
| | PSST | +0.118 [-0.001, +0.226] | +0.002 [-0.001, +0.006] | +0.045 [-0.006, +0.092] |
| | BDI | +0.039 [+0.006, +0.072] | +0.020 [-0.001, +0.041] | +0.184 [+0.058, +0.293] |
| | BAI | +0.117 [-0.014, +0.233] | +0.003 [-0.005, +0.011] | +0.063 [+0.011, +0.115] |
| | PSST + BDI + BAI together | - | - | +0.178 [+0.068, +0.278] |

**Ablation findings.** (1) *Dropping the hormones never costs anything.* In the ridge ablation (3 samples x 3 targets) dropping hormones left RMSE unchanged within the interval in all but 2 cases, and in those the model was *better* without them (change in RMSE negative); for DRSP psychological, pooled, R2 was +0.368 [+0.257, +0.465] without hormones versus +0.360 [+0.247, +0.457] with them. Hormones alone: -0.002 [-0.020, +0.014]. (2) *Phase matters for DRSP*: dropping phase raises RMSE by +0.155 [+0.026, +0.282] (pooled, interval excludes 0); for positive affect and anxiety the change is at most 0.005 RMSE units (the scales of these targets are about 1). (3) *BDI is the most valuable single group*: dropping it lowers pooled DRSP R2 to +0.275 [+0.175, +0.371]; BDI alone gives +0.282 [+0.165, +0.393], and the three questionnaires together +0.317 [+0.190, +0.421], essentially the whole of set D. PSST and BAI are partly interchangeable with BDI (BAI dropped: +0.350 [+0.238, +0.445]), the 'shared credit' effect described in the note. Many drop-one intervals include 0 because the targets are noisy and the groups overlap: the ablation shows the *order of magnitude* of each group's contribution, not exact values. (4) *Random forest* (drop-one, pooled and longitudinal, 30 comparisons): 0 significant improvements and 4 significant deteriorations; ridge (45 comparisons): 7 and 4. Figure: `results/fig_E5_ablation.png`.

**Learning curves** (ridge; leave-one-woman-out with the training set restricted to k other women). Test = R2 versus the mean baseline trained on the same k women [95% CI over the held-out women]; train = mean R2 on the training women. Set D:

Pooled (271 test rows, 160 women), columns = number of training women:

| Target | k = 10 | k = 20 | k = 40 | k = 80 | k = 159 |
|---|---|---|---|---|---|
| positive affect | -0.06 [-0.26, +0.08] (train 0.31) | +0.00 [-0.11, +0.09] (train 0.22) | +0.10 [+0.04, +0.16] (train 0.18) | +0.07 [+0.00, +0.14] (train 0.17) | +0.09 [-0.00, +0.17] (train 0.15) |
| DRSP psychological symptoms | +0.01 [-0.33, +0.22] (train 0.48) | +0.32 [+0.22, +0.41] (train 0.46) | +0.33 [+0.23, +0.42] (train 0.44) | +0.33 [+0.20, +0.45] (train 0.42) | +0.36 [+0.26, +0.46] (train 0.41) |
| state anxiety | -0.26 [-0.92, +0.10] (train 0.32) | +0.14 [+0.03, +0.23] (train 0.28) | +0.14 [+0.03, +0.23] (train 0.24) | +0.13 [+0.02, +0.23] (train 0.22) | +0.15 [+0.05, +0.25] (train 0.20) |

Longitudinal (176 test rows, 65 women), columns = number of training women:

| Target | k = 10 | k = 20 | k = 30 | k = 40 | k = 64 |
|---|---|---|---|---|---|
| positive affect | -0.09 [-0.24, +0.03] (train 0.23) | +0.03 [-0.08, +0.13] (train 0.17) | +0.01 [-0.11, +0.11] (train 0.15) | +0.01 [-0.08, +0.08] (train 0.13) | +0.04 [-0.06, +0.12] (train 0.11) |
| DRSP psychological symptoms | +0.07 [-0.07, +0.21] (train 0.41) | +0.29 [+0.15, +0.41] (train 0.38) | +0.22 [+0.09, +0.33] (train 0.37) | +0.26 [+0.13, +0.37] (train 0.36) | +0.26 [+0.11, +0.38] (train 0.35) |
| state anxiety | -0.10 [-0.38, +0.13] (train 0.32) | +0.11 [-0.01, +0.22] (train 0.24) | +0.09 [-0.07, +0.24] (train 0.25) | +0.04 [-0.11, +0.17] (train 0.23) | +0.13 [-0.01, +0.25] (train 0.21) |

The last column of each table (k = all other women) reproduces the Phase 4 result exactly (largest difference 1.1e-16). **Reading.** (1) With very few women (k = 10) the questionnaire model is not better than the mean (DRSP, pooled, R2 +0.01 [-0.33, +0.22]) and its training R2 (0.48) is far above its test R2: overfitting. (2) By k = 20 the test R2 has jumped to +0.32 [+0.22, +0.41] and then changes slowly up to all 160 women (+0.36 [+0.26, +0.46]), while the training R2 falls (0.41 at the end): the gap closes and the curve is flat, so for this ridge model on these predictors more women of the same kind would add little. (3) The other two targets show no systematic further rise beyond k = 40 either (see the tables), within wide intervals; for phase alone (set A) the gain with more women is small (DRSP, pooled: -0.012 at k = 10 to +0.051 with all women), and the phase-plus-hormones set C does not improve with more women either (DRSP, pooled, k = all: +0.037). Caveats: single random draw of the training women per point, intervals overlap between neighbouring k, and the curve cannot be extrapolated beyond 159 training women. Figure: `results/fig_E5_learning_curves.png`.

### 8.6 E7: Cross-sample validation (two independent groups of women from the same study)

Math note: `docs/math/E7_cross_sample.md`. **Terminology: this is cross-sample validation, not cross-dataset validation.** The 95 single-session women and the 65 longitudinal women are disjoint groups of women, but they come from the same study (presumably the same laboratory, assays and recruitment period; the files say nothing to the contrary). A second, truly independent dataset has not been used (see `docs/cross_dataset_candidates.md`). **Primary setting:** all standardisation and transformation statistics, and all hyper-parameters (inner cross-validation grouped by woman), come from the training group only; the baseline is the training-group mean. Models: baseline, ridge (scikit-learn), ridge from scratch, random forest, gradient boosting; feature sets A, B, C, D, D2; three targets.

| Direction | Training rows / women | Test rows / women | Women in both |
|---|---|---|---|
| Direction 1: train on the single-session women, test on the longitudinal women | 95 / 95 | 176 / 65 | 0 |
| Direction 2: train on the longitudinal women, test on the single-session women | 176 / 65 | 95 / 95 | 0 |

The assertion that no woman is on both sides is in the code and is also visible above. The scikit-learn ridge and my from-scratch ridge (same inner folds, alpha = n x lambda) give the same predictions: largest difference over 4065 out-of-sample predictions 2.1e-14.

**Direction 1: train on the single-session women, test on the longitudinal women** (test: 176 rows, 65 women). R2 versus the training-group mean [95% CI over test women]; first row: RMSE of the baseline.

| Model | Set | positive affect | DRSP psychological | state anxiety |
|---|---|---|---|---|
| baseline (training mean) RMSE | - | 0.797 | 9.398 | 0.435 |
| ridge | A | +0.012 [-0.012, +0.037] | +0.054 [+0.016, +0.099] | -0.002 [-0.004, +0.000] |
| ridge | B | -0.010 [-0.044, +0.023] | +0.001 [-0.001, +0.002] | -0.001 [-0.003, +0.001] |
| ridge | C | +0.006 [-0.022, +0.035] | +0.022 [+0.008, +0.040] | -0.003 [-0.006, +0.000] |
| ridge | D2 | +0.111 [+0.005, +0.206] | +0.252 [+0.116, +0.378] | +0.118 [-0.001, +0.226] |
| ridge | D | +0.107 [+0.004, +0.201] | +0.292 [+0.158, +0.409] | +0.128 [+0.023, +0.226] |
| ridge_scratch | A | +0.012 [-0.012, +0.037] | +0.054 [+0.016, +0.099] | -0.002 [-0.004, +0.000] |
| ridge_scratch | B | -0.010 [-0.044, +0.023] | +0.001 [-0.001, +0.002] | -0.001 [-0.003, +0.001] |
| ridge_scratch | C | +0.006 [-0.022, +0.035] | +0.022 [+0.008, +0.040] | -0.003 [-0.006, +0.000] |
| ridge_scratch | D2 | +0.111 [+0.005, +0.206] | +0.252 [+0.116, +0.378] | +0.118 [-0.001, +0.226] |
| ridge_scratch | D | +0.107 [+0.004, +0.201] | +0.292 [+0.158, +0.409] | +0.128 [+0.023, +0.226] |
| rf | A | +0.002 [-0.000, +0.003] | +0.056 [+0.013, +0.105] | -0.002 [-0.009, +0.004] |
| rf | B | -0.016 [-0.064, +0.040] | -0.040 [-0.092, +0.010] | +0.000 [-0.011, +0.012] |
| rf | C | -0.015 [-0.063, +0.040] | -0.015 [-0.077, +0.045] | -0.001 [-0.012, +0.010] |
| rf | D2 | +0.040 [-0.100, +0.158] | +0.241 [+0.088, +0.362] | +0.166 [+0.061, +0.259] |
| rf | D | +0.030 [-0.112, +0.149] | +0.248 [+0.101, +0.367] | +0.147 [+0.035, +0.245] |
| gb | A | +0.008 [-0.022, +0.042] | +0.055 [+0.016, +0.100] | -0.024 [-0.043, -0.006] |
| gb | B | -0.003 [-0.044, +0.035] | -0.029 [-0.069, +0.005] | -0.173 [-0.305, -0.063] |
| gb | C | -0.002 [-0.043, +0.037] | +0.002 [-0.034, +0.042] | -0.174 [-0.305, -0.065] |
| gb | D2 | +0.051 [-0.059, +0.143] | +0.249 [+0.114, +0.360] | +0.096 [+0.014, +0.168] |
| gb | D | +0.058 [-0.051, +0.155] | +0.258 [+0.130, +0.363] | +0.096 [+0.014, +0.168] |

**Direction 2: train on the longitudinal women, test on the single-session women** (test: 95 rows, 95 women). R2 versus the training-group mean [95% CI over test women]; first row: RMSE of the baseline.

| Model | Set | positive affect | DRSP psychological | state anxiety |
|---|---|---|---|---|
| baseline (training mean) RMSE | - | 0.683 | 7.717 | 0.362 |
| ridge | A | +0.003 [-0.091, +0.085] | +0.039 [-0.091, +0.152] | -0.016 [-0.035, +0.004] |
| ridge | B | -0.004 [-0.017, +0.008] | -0.010 [-0.064, +0.037] | -0.001 [-0.004, +0.002] |
| ridge | C | -0.003 [-0.090, +0.074] | +0.020 [-0.096, +0.124] | -0.003 [-0.008, +0.002] |
| ridge | D2 | +0.128 [+0.021, +0.226] | +0.456 [+0.277, +0.594] | +0.173 [+0.033, +0.289] |
| ridge | D | +0.115 [+0.025, +0.201] | +0.499 [+0.331, +0.634] | +0.153 [+0.023, +0.265] |
| ridge_scratch | A | +0.003 [-0.091, +0.085] | +0.039 [-0.091, +0.152] | -0.016 [-0.035, +0.004] |
| ridge_scratch | B | -0.004 [-0.017, +0.008] | -0.010 [-0.064, +0.037] | -0.001 [-0.004, +0.002] |
| ridge_scratch | C | -0.003 [-0.090, +0.074] | +0.020 [-0.096, +0.124] | -0.003 [-0.008, +0.002] |
| ridge_scratch | D2 | +0.128 [+0.021, +0.226] | +0.456 [+0.277, +0.594] | +0.173 [+0.033, +0.289] |
| ridge_scratch | D | +0.115 [+0.025, +0.201] | +0.499 [+0.331, +0.634] | +0.153 [+0.023, +0.265] |
| rf | A | -0.002 [-0.104, +0.087] | +0.037 [-0.096, +0.152] | -0.045 [-0.095, -0.000] |
| rf | B | +0.012 [-0.084, +0.090] | -0.061 [-0.183, +0.046] | -0.008 [-0.072, +0.044] |
| rf | C | +0.014 [-0.085, +0.096] | -0.035 [-0.154, +0.069] | -0.010 [-0.069, +0.040] |
| rf | D2 | +0.058 [-0.142, +0.215] | +0.440 [+0.260, +0.583] | +0.093 [-0.252, +0.329] |
| rf | D | +0.060 [-0.136, +0.216] | +0.458 [+0.281, +0.598] | +0.047 [-0.292, +0.280] |
| gb | A | +0.001 [-0.100, +0.089] | +0.041 [-0.085, +0.150] | -0.024 [-0.053, +0.004] |
| gb | B | +0.019 [-0.063, +0.092] | -0.022 [-0.113, +0.059] | -0.013 [-0.082, +0.050] |
| gb | C | +0.021 [-0.060, +0.093] | -0.000 [-0.084, +0.077] | -0.013 [-0.082, +0.050] |
| gb | D2 | +0.154 [-0.013, +0.295] | +0.430 [+0.240, +0.576] | +0.161 [-0.057, +0.323] |
| gb | D | +0.154 [-0.013, +0.295] | +0.461 [+0.284, +0.600] | +0.155 [-0.050, +0.305] |

Direction 1 repeats the cross-sectional-to-longitudinal experiment of Phase 4 with additional models and tests: for the from-scratch ridge, random forest and gradient boosting the 8448 predictions are identical to the Phase 4 ones (largest difference 0.0e+00).

**MAE and RMSE, ridge with the full feature set D versus the training-mean baseline** [95% CI over test women]:

| Direction (test n) | Target | Baseline MAE | Ridge D MAE | Baseline RMSE | Ridge D RMSE | R2 vs training mean | R2 vs test mean (diagnostic) |
|---|---|---|---|---|---|---|---|
| cross to long (176 rows, 65 women) | positive affect | 0.675 [0.599, 0.750] | 0.629 [0.559, 0.701] | 0.797 [0.720, 0.869] | 0.753 [0.677, 0.829] | +0.107 [+0.004, +0.201] | +0.107 [-0.015, +0.194] |
| cross to long (176 rows, 65 women) | DRSP psychological symptoms | 7.781 [6.928, 8.687] | 6.325 [5.649, 7.052] | 9.398 [8.351, 10.540] | 7.907 [7.050, 8.759] | +0.292 [+0.158, +0.409] | +0.291 [+0.144, +0.404] |
| cross to long (176 rows, 65 women) | state anxiety | 0.376 [0.339, 0.414] | 0.340 [0.307, 0.373] | 0.435 [0.396, 0.474] | 0.407 [0.372, 0.441] | +0.128 [+0.023, +0.226] | +0.124 [-0.010, +0.219] |
| long to cross (95 rows, 95 women) | positive affect | 0.562 [0.485, 0.641] | 0.528 [0.456, 0.602] | 0.683 [0.591, 0.773] | 0.643 [0.555, 0.728] | +0.115 [+0.025, +0.201] | +0.115 [+0.010, +0.196] |
| long to cross (95 rows, 95 women) | DRSP psychological symptoms | 6.378 [5.515, 7.216] | 4.352 [3.686, 5.014] | 7.717 [6.761, 8.635] | 5.460 [4.710, 6.181] | +0.499 [+0.331, +0.634] | +0.498 [+0.308, +0.631] |
| long to cross (95 rows, 95 women) | state anxiety | 0.271 [0.226, 0.319] | 0.250 [0.208, 0.295] | 0.362 [0.302, 0.422] | 0.333 [0.281, 0.387] | +0.153 [+0.023, +0.265] | +0.147 [+0.010, +0.249] |

**Do hormones help in each direction?** Set C (phase + hormones) versus set A (phase alone), change in RMSE on the test women (negative = hormones help), with the p-value, the Holm-adjusted p and the FDR-adjusted p over all 36 hormone comparisons (C versus A and hormones alone versus baseline, three models, three targets, two directions):

| Target | Model | Direction 1 (train single-session, test longitudinal) | Direction 2 (train longitudinal, test single-session) |
|---|---|---|---|
| positive affect | ridge | +0.002 [-0.005, +0.010] (p 0.558; Holm 1.000; FDR 0.866) | +0.002 [-0.001, +0.006] (p 0.250; Holm 1.000; FDR 0.818) |
| positive affect | rf | +0.007 [-0.015, +0.026] (p 0.552; Holm 1.000; FDR 0.866) | -0.005 [-0.029, +0.019] (p 0.706; Holm 1.000; FDR 0.866) |
| positive affect | gb | +0.004 [-0.014, +0.024] (p 0.690; Holm 1.000; FDR 0.866) | -0.007 [-0.036, +0.023] (p 0.662; Holm 1.000; FDR 0.866) |
| DRSP psychological symptoms | ridge | +0.154 [+0.026, +0.282] (p 0.014; Holm 0.462; FDR 0.115) | +0.077 [-0.098, +0.255] (p 0.395; Holm 1.000; FDR 0.866) |
| DRSP psychological symptoms | rf | +0.333 [+0.075, +0.604] (p 0.016; Holm 0.512; FDR 0.115) | +0.276 [-0.104, +0.652] (p 0.172; Holm 1.000; FDR 0.688) |
| DRSP psychological symptoms | gb | +0.254 [+0.076, +0.427] (p 0.007; Holm 0.245; FDR 0.108) | +0.160 [-0.230, +0.559] (p 0.447; Holm 1.000; FDR 0.866) |
| state anxiety | ridge | +0.000 [-0.000, +0.001] (p 0.361; Holm 1.000; FDR 0.866) | -0.002 [-0.005, +0.000] (p 0.087; Holm 1.000; FDR 0.522) |
| state anxiety | rf | -0.000 [-0.003, +0.002] (p 0.862; Holm 1.000; FDR 0.912) | -0.006 [-0.018, +0.006] (p 0.336; Holm 1.000; FDR 0.866) |
| state anxiety | gb | +0.031 [+0.008, +0.055] (p 0.009; Holm 0.306; FDR 0.108) | -0.002 [-0.015, +0.011] (p 0.767; Holm 1.000; FDR 0.866) |

Counts per direction and family (9 tests each; 'worse' / 'better' = the 95% CI of the RMSE change lies above / below 0):

| Direction | Family | raw p < 0.05 | Holm (all) < 0.05 | FDR (all) < 0.05 | hormones worse | hormones better | smallest one-sided p for an improvement (raw / Holm all) |
|---|---|---|---|---|---|---|---|
| cross to long | C vs A | 4 | 0 | 0 | 4 | 0 | 0.431 / 1.000 |
| cross to long | B vs baseline | 1 | 1 | 1 | 1 | 0 | 0.170 / 1.000 |
| long to cross | C vs A | 0 | 0 | 0 | 0 | 0 | 0.043 / 1.000 |
| long to cross | B vs baseline | 0 | 0 | 0 | 0 | 0 | 0.304 / 1.000 |

**Answer.** In neither direction do hormones improve prediction beyond phase once the multiple comparisons are accounted for: the smallest adjusted one-sided p-value for an improvement over all 36 comparisons is 1.000 (Holm) and 1.000 (FDR). The single uncorrected hint of an improvement (one-sided raw p = 0.043, long to cross, state anxiety, ridge) is a change in RMSE of -0.0023 on an outcome with a baseline RMSE of 0.362: practically nil. In direction 1 hormones make predictions *worse* in 4 of 9 comparisons by the uncorrected interval (none survives the correction), in direction 2 there is no difference in either direction. Details and the second family (hormones alone versus the baseline): `results/ext_e7_hormone_tests.csv`.

**Distribution shift between the two groups of women.** Standardised mean difference (SMD, longitudinal minus single-session; pooled SD; 95% CI resampling women within each group) for every feature and target; Kolmogorov-Smirnov statistic D on all sessions (descriptive, because sessions of one woman are not independent) and on the first session of each woman (independent units; Holm-adjusted p over the variables):

| Variable | SMD [95% CI] | KS D, all sessions | KS D, first sessions (n single / longitudinal) | KS p, Holm |
|---|---|---|---|---|
| peri-ovulatory phase (0/1) | +0.23 [+0.03, +0.44] | 0.10 | 0.07 (95 / 65) | 1.000 |
| luteal phase (0/1) | -0.14 [-0.35, +0.06] | 0.07 | 0.01 (95 / 65) | 1.000 |
| estradiol, raw pg/ml | +0.15 [-0.15, +0.43] | 0.17 | 0.20 (95 / 65) | 0.816 |
| progesterone, raw pg/ml | +0.22 [-0.01, +0.44] | 0.17 | 0.18 (95 / 65) | 1.000 |
| log1p estradiol | +0.17 [-0.11, +0.47] | 0.17 | 0.20 (95 / 65) | 0.816 |
| log1p progesterone | +0.41 [+0.15, +0.68] | 0.17 | 0.18 (95 / 65) | 1.000 |
| PSST | +0.24 [-0.07, +0.57] | 0.14 | 0.14 (95 / 65) | 1.000 |
| BDI | -0.04 [-0.40, +0.27] | 0.07 | 0.09 (95 / 65) | 1.000 |
| BAI | -0.25 [-0.58, +0.05] | 0.14 | 0.14 (95 / 65) | 1.000 |
| positive affect | -0.01 [-0.29, +0.27] | 0.09 | 0.09 (95 / 65) | 1.000 |
| DRSP psychological | +0.04 [-0.27, +0.30] | 0.11 | 0.10 (95 / 65) | 1.000 |
| state anxiety | -0.08 [-0.38, +0.20] | 0.14 | 0.14 (95 / 65) | 1.000 |

Figure: `results/fig_e7_shift.png`. Of the 12 variables, 2 have an SMD interval that excludes 0: peri-ovulatory phase (0/1) (+0.23); log1p progesterone (+0.41). The Kolmogorov-Smirnov tests on first sessions find no significant difference for any variable after the Holm correction (smallest adjusted p 0.816). All SMDs of the questionnaire scores and of the three targets are small and their intervals include 0.

**Hormones within each phase** (log1p scale; difference longitudinal minus single-session, 95% CI resampling women; n = sessions = women per phase and group):

| Phase | Hormone | n single-session | n longitudinal | Mean single-session | Mean longitudinal | Difference [95% CI] |
|---|---|---|---|---|---|---|
| menses | log E2 | 38 | 65 | 0.43 | 0.47 | +0.03 [-0.09, +0.14] |
| menses | log P | 38 | 65 | 2.63 | 3.29 | +0.66 [+0.24, +1.15] |
| peri-ovulatory | log E2 | 18 | 51 | 0.73 | 0.70 | -0.03 [-0.17, +0.11] |
| peri-ovulatory | log P | 18 | 51 | 2.69 | 3.57 | +0.89 [+0.34, +1.53] |
| luteal | log E2 | 39 | 60 | 0.60 | 0.67 | +0.08 [-0.03, +0.18] |
| luteal | log P | 39 | 60 | 4.60 | 4.90 | +0.31 [+0.07, +0.54] |

The composition of phases differs little between the two groups (chi-square test on sessions, descriptive, p = 0.18), but **progesterone is higher in the longitudinal women in each of the three phases** (all three intervals exclude 0), while estradiol is not. **Multivariate check:** a logistic classifier trained to tell the two groups apart from the features, cross-validated by woman, gives the following area under the ROC curve (0.5 = indistinguishable):

| Feature set | AUC [95% CI] | Rows / women |
|---|---|---|
| A | 0.50 [0.44, 0.56] | 271 / 160 |
| B | 0.57 [0.49, 0.64] | 271 / 160 |
| C | 0.65 [0.58, 0.72] | 271 / 160 |
| D | 0.64 [0.56, 0.72] | 271 / 160 |

Phase alone does not separate the two groups (the interval contains 0.5); phase plus hormones does, so the groups differ in the relation between hormone levels and phase, not in the phase mix. **Target spread and mean (per direction):**

| Direction | Target | Training mean | Test mean | Training SD | Test SD | SD ratio test / train | Baseline RMSE | Share of baseline MSE due to the mean shift |
|---|---|---|---|---|---|---|---|---|
| cross to long | positive affect | 3.17 | 3.17 | 0.69 | 0.80 | 1.16 | 0.797 | 0.000 |
| cross to long | DRSP psychological symptoms | 21.82 | 22.14 | 7.75 | 9.42 | 1.22 | 9.398 | 0.001 |
| cross to long | state anxiety | 1.91 | 1.88 | 0.36 | 0.44 | 1.20 | 0.435 | 0.005 |
| long to cross | positive affect | 3.17 | 3.17 | 0.80 | 0.69 | 0.86 | 0.683 | 0.000 |
| long to cross | DRSP psychological symptoms | 22.14 | 21.82 | 9.42 | 7.75 | 0.82 | 7.717 | 0.002 |
| long to cross | state anxiety | 1.88 | 1.91 | 0.44 | 0.36 | 0.83 | 0.362 | 0.007 |

**Relationship shift** (OLS on set D, features and outcome standardised with the pooled statistics; difference longitudinal minus single-session with 95% CI resampling women within each group): 2 of 21 coefficients differ between the groups: DRSP psychological symptoms, log E2: single-session +0.11, longitudinal -0.14, difference -0.25 [-0.45, -0.06]; state anxiety, peri-ovulatory phase: single-session +0.25, longitudinal -0.11, difference -0.36 [-0.60, -0.12]. With this many coefficients, a few intervals excluding 0 can arise by chance; the full table is `results/ext_e7_coef_shift.csv`. Note that for DRSP the estradiol coefficient has opposite sign in the two groups.

**Which shifts could explain the difference between the two directions?** (Suggestions, not proofs: two groups cannot separate the candidate causes.) (1) *Spread of the target.* The test group is more variable than the training group in direction 1 (SD ratio 1.16 to 1.22) and less variable in direction 2 (0.82 to 0.86); this goes with the lower skill in direction 1 and the higher skill in direction 2 for the questionnaire model (DRSP, set D: R2 +0.292 [+0.158, +0.409] versus +0.499 [+0.331, +0.634]). The two training samples also differ in size (176 sessions of 65 women versus 95 sessions of 95 women). (2) *Mean level:* negligible (at most 0.007 of the baseline error). (3) *Hormone-by-phase shift:* progesterone is higher in the longitudinal women in every phase and the classifier separates the groups only when hormones and phase are combined (AUC 0.65 [0.58, 0.72]); this is consistent with hormones adding nothing or even hurting when transferred (direction 1) and with different estradiol-DRSP coefficients in the two groups. (4) *Questionnaire scores* show no detectable shift (SMDs small, intervals include 0) and their predictive skill transfers in both directions, which is also what the pooled analyses found.

**Limits.** Both groups come from the same study: this validates across groups of women, not across laboratories, assays or populations, so it is *cross-sample*, not cross-dataset validation. The test groups are small (65 and 95 women), intervals are wide, and the designs differ (one session versus up to three sessions), which could itself make the groups differ. Distribution-shift tests have limited power at these sample sizes: 'no significant shift' means 'not detected'.

Figures: `results/fig_e7_cross_sample.png` (out-of-sample skill), `results/fig_e7_shift.png` (standardised mean differences).

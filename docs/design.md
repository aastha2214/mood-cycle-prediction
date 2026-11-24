> **Working notes: the checked numbers are in docs/report.md**

# Phase 2 - Design (no modelling done yet)

Data: Questionnaires file, `include == 1`, 272 sessions / 161 women. Longitudinal L = 176 rows / 65 women; cross-sectional C = 96 rows / 96 women (disjoint). Phase: 0 menses, 1 peri-ovulatory, 2 luteal. Numbers below are from `src/design_checks.py`.

## 1. Targets (one model per target)
| Target | Column | Scale | SD (all 272) | Within-woman SD vs total SD (L, n=176)* |
|---|---|---|---|---|
| Positive affect | `Affect_pos` | 1-5 | 0.77 | 0.48 vs 0.80 |
| DRSP psychological symptoms | `DRSP_psychisch` (NOT `DRSP_physical`, a duplicate) | 9-49 sum | 8.9 | 5.8 vs 9.4 |
| State anxiety | `STAI` | 1-4 item mean | 0.41 | 0.23 vs 0.44 |

*SD of deviations from each woman's own mean (divisor n, not df-corrected): roughly 40-50% of variance is between-women. Because we predict **unseen** women, this stable personal level cannot be learned, which is why the trait-score set D may help and why R2 will likely be low. Negative affect is a secondary target if you want it (the paper found no effect).
Units: DRSP/STAI/affect are comparable across women only as scale means or sums; report MAE in raw units next to R2.

## 2. Feature sets
| Set | Features | Columns |
|---|---|---|
| A | phase only | one-hot of `Phase` (2 dummies, menses = reference) |
| B | hormones only | E2, P (transformed, see 3) |
| C | A + B | 4 columns |
| D | C + traits | C + `PMS` (PSST), `BDI_RW`, `BAI_RW` |
Not used as features: cycle day (it nearly defines phase; mention as optional extension), `Session`/`Version` (design artefacts), other mood scores (leakage-by-construction), Age (not requested).
Note on A vs B: phase is itself a (noisy) function of hormones and cycle day, so the question "do hormones add beyond phase" is answered by C vs A, not B vs A.

## 3. Hormone transform: raw vs log1p (descriptive, all 272 rows)
| | skew | excess kurtosis | max z | zeros |
|---|---|---|---|---|
| E2 raw | 1.14 | 1.94 | 4.4 | 9 |
| E2 log1p | 0.22 | 0.02 | 3.1 | 9 |
| P raw | 2.80 | 13.33 | 7.4 | 7 |
| P log1p | -0.81 | 1.10 | 2.3 | 7 |
Pearson corr E2-P: 0.19 raw, 0.32 log1p (Spearman 0.34, rank-based so unaffected). Corr of P with luteal dummy: 0.66 either way.
Reading: raw P is dominated by 5 rows > 300 (max z = 7.4), which would pull ridge/GD coefficients and inflate the Hessian condition number; log1p handles the zeros (log(0) is undefined) and tames outliers, but over-corrects P slightly (skew becomes -0.81, a left tail from the low values). Fine for a learning project.
**Recommendation:** primary = log1p then standardise (mean/SD from training folds only). Report raw+standardise as a sensitivity analysis for the ridge models; trees are invariant to monotone transforms, so they are unaffected. No winsorising (do not silently alter data).
Standardising is required for ridge (penalty is scale dependent) and for the Phase 3c conditioning study.

**Within-person change from menses baseline** (hormone minus that woman's menses value): feasible for L (all 65 women have a menses row; 111/111 non-menses rows have one) but (i) impossible for C (one session each), (ii) for a *new* woman needs her own menses sample at prediction time, (iii) the menses rows become all-zero, losing information. Recommendation: **not in the main analysis**; run as a labelled secondary experiment on L only, if you approve.

## 4. Missing data
Among the 272 included rows: hormones, BDI, BAI, all three targets, Phase: **0 missing**; `PMS`: **1 missing** (so set D n = 271 unless imputed). Plan: no imputation for A-C. For D, either drop that 1 row (simple, but then D is evaluated on 1 fewer row than A-C, so all sets would be re-run on the same 271 rows for fair comparison) or impute the median inside the training fold. Recommendation: **restrict ALL sets to the same 271 complete rows** and state n everywhere; no missing-indicator needed with one value. Zero hormone values are real low readings, not missing, kept as-is.

## 5. Validation design (carried to Phase 4)
- Unit of grouping = `VPN`. Leave-one-participant-out (161 folds when pooling L+C; 65 when L only). Nested CV: inner GroupKFold on training women for lambda/depth.
- Primary evaluation sample: **pooled L+C** (n=161 women) for power, plus L-only results. Cross-sectional->longitudinal transfer as a separate experiment (disjoint, so no leakage).
- Baseline: global training mean. Skill = improvement in MAE/RMSE/R2 over it, with participant-level bootstrap CIs.
- Sessions of the same woman never split across train and test; all scaling/transforms fitted inside training folds.

## Decisions needed from you
1. Targets: the three above (negative affect optional)?
2. Primary sample: pooled L+C (161 women), or L only (65)?
3. Transform: log1p+standardise primary, raw as sensitivity?
4. Missing PMS: use the common 271 rows for all sets?
5. Within-person change feature: skip in main analysis, do as L-only secondary?

---
## Approved decisions (all five recommendations accepted) + additions
Targets as listed; primary sample pooled L+C with **longitudinal-only results reported alongside in every analysis**; log1p+standardise (raw = sensitivity); all feature sets on the same complete rows; within-person change = longitudinal-only secondary experiment.

**Analysis set after dropping the one row with missing PSST: 271 rows, 160 women** (longitudinal 176 rows / 65 women; cross-sectional 95 rows / 95 women). The woman with the missing PSST was cross-sectional, so every "161 folds" above is **160** leave-one-participant-out folds when pooled (65 longitudinal-only). Built by `src/data.py`.

## ICC of each target (`src/icc.py`, results/icc.csv)
ICC(1) = share of variance that is between women = sigma_u^2 / (sigma_u^2 + sigma_e^2). Two estimators (unbalanced one-way ANOVA with adjusted n0; REML random-intercept model) and a 95% percentile CI from 500 bootstraps that resample women (ANOVA estimator).
| Sample | Target | rows / women | ICC ANOVA | ICC REML | 95% CI |
|---|---|---|---|---|---|
| Pooled | Positive affect | 271 / 160 | 0.37 | 0.39 | 0.19-0.54 |
| Pooled | DRSP psychological | 271 / 160 | 0.31 | 0.35 | 0.13-0.45 |
| Pooled | State anxiety | 271 / 160 | 0.52 | 0.52 | 0.34-0.66 |
| Longitudinal | Positive affect | 176 / 65 | 0.43 | 0.43 | 0.23-0.57 |
| Longitudinal | DRSP psychological | 176 / 65 | 0.39 | 0.39 | 0.20-0.53 |
| Longitudinal | State anxiety | 176 / 65 | 0.57 | 0.57 | 0.42-0.71 |

Caveats: in the pooled sample the within-woman variance comes only from the 65 longitudinal women (95 cross-sectional women have a single row), so the pooled ICC mixes two populations and is less trustworthy than the longitudinal one. ICCs here include the phase effect in "within" variance (no covariates).
Consequence: 31-57% of the variance is a stable between-woman level. For unseen women a model can only reach that part via trait scores (set D) and not through phase/hormones, so we expect modest R2 for sets A-C; anxiety (highest ICC) should benefit most from traits. This is a prediction to test in Phase 4, not a result.

---
## Added feature set D2 (Phase 4 addition)
| Set | Features |
|---|---|
| D2 | phase (2 dummies) + log1p E2 + log1p P + BDI + BAI (**D without PSST**) |
Purpose: PSST (premenstrual symptom screening) and DRSP (daily premenstrual symptom rating) may measure overlapping constructs, so a high R2 for D on DRSP could partly reflect that overlap rather than genuine prediction. D and D2 are reported side by side in Phase 4; a big drop from D to D2 on DRSP would support the overlap explanation. Defined in `src/data.py` (`FEATURE_SETS["D2"]`). Phase 3d nested-CV numbers are demonstration only.

# Phase 5 comparison tables (written by src/phase5_tables.py)

**Authors' reported values are from the task description, reference only; they are NOT my results.** My estimates: random-intercept mixed model (REML, statsmodels), outcomes z-scored within sample (the authors' R script also uses scale()), menses = reference phase, Wald tests; Holm over the 3 phase contrasts (the authors used Tukey). The like-for-like sample is the longitudinal one.

| Effect | Authors' reported (reference only) | Mine: longitudinal (n=176 rows, 65 women) | Mine: pooled (n=271 rows, 160 women) |
|---|---|---|---|
| Positive affect: peri-ovulatory - menses | b = 0.41 | +0.41 [+0.14, +0.67], p=0.003 (Holm 0.005) | +0.42 [+0.18, +0.67], p=<0.001 (Holm 0.001) |
| Positive affect: luteal - menses | b = 0.47 | +0.47 [+0.21, +0.72], p=<0.001 (Holm <0.001) | +0.41 [+0.18, +0.63], p=<0.001 (Holm 0.001) |
| Negative affect: peri-ov - menses (authors: lower, n.s.) | n.s. | -0.28 [-0.57, +0.01], p=0.057 (Holm 0.171) | -0.24 [-0.50, +0.02], p=0.074 (Holm 0.169) |
| Negative affect: luteal - menses | n.s. | -0.26 [-0.54, +0.01], p=0.059 (Holm 0.171) | -0.23 [-0.47, +0.01], p=0.056 (Holm 0.169) |
| DRSP psychological: peri-ov - menses | b = -0.68 (DRSP premenstrual symptoms) | -0.69 [-0.95, -0.44], p=<0.001 (Holm <0.001) | -0.68 [-0.92, -0.44], p=<0.001 (Holm <0.001) |
| DRSP psychological: luteal - menses | b = -0.52 | -0.51 [-0.75, -0.27], p=<0.001 (Holm <0.001) | -0.49 [-0.70, -0.27], p=<0.001 (Holm <0.001) |
| DRSP total (14 items): peri-ov - menses | b = -0.68 | -0.84 [-1.09, -0.59], p=<0.001 (Holm <0.001) | -0.83 [-1.06, -0.60], p=<0.001 (Holm <0.001) |
| DRSP total: luteal - menses | b = -0.52 | -0.59 [-0.83, -0.36], p=<0.001 (Holm <0.001) | -0.61 [-0.82, -0.40], p=<0.001 (Holm <0.001) |
| State anxiety: peri-ov - menses | b = -0.27, p = 0.07 (trend) | -0.27 [-0.51, -0.03], p=0.027 (Holm 0.081) | -0.17 [-0.40, +0.06], p=0.148 (Holm 0.445) |
| DRSP psychological ~ estradiol (z) | b = -0.30, p < 0.001 | -0.28 [-0.44, -0.11], p=0.001 | -0.16 [-0.29, -0.03], p=0.014 |
| DRSP total ~ estradiol (z) | b = -0.30, p < 0.001 | -0.29 [-0.46, -0.12], p=<0.001 | -0.17 [-0.31, -0.04], p=0.009 |
| DRSP psychological ~ progesterone (z) | not related | -0.06 [-0.19, +0.07], p=0.355 | -0.07 [-0.18, +0.04], p=0.210 |
| State anxiety ~ estradiol (z) | b = -0.15, p = 0.06 (trend) | -0.15 [-0.30, +0.01], p=0.063 | -0.08 [-0.21, +0.04], p=0.202 |
| Positive affect ~ estradiol / progesterone | all p > 0.16 | E2 p=0.227, P p=0.273 | E2 p=0.114, P p=0.307 |
| Negative affect ~ estradiol / progesterone | all p > 0.16 | E2 p=0.168, P p=0.504 | E2 p=0.207, P p=0.383 |
| Phase x PSST on DRSP (joint 2-df Wald p), psychological / total | p = 0.001 | 0.0884 / 0.0072 | 0.0296 / 0.0010 |
| BDI -> positive affect (menses slope, phase x BDI model) | b = -0.45 | -0.45 [-0.69, -0.22], p=<0.001 | -0.44 [-0.61, -0.27], p=<0.001 |
| BDI -> negative affect | b = 0.56 | +0.56 [+0.34, +0.78], p=<0.001 | +0.53 [+0.37, +0.69], p=<0.001 |

## Hormone means by phase, longitudinal sample (raw pg/ml; rows = women per phase)

| Phase | n | E2 mean (authors ~) | E2 mine | P mean (authors ~) | P mine |
|---|---|---|---|---|---|
| menses | 65 | 0.65 | 0.65 | 34.7 | 34.74 |
| peri-ovulatory | 51 | 1.1 | 1.10 | 43.8 | 43.75 |
| luteal | 60 | 1.02 | 1.02 | 155.6 | 155.56 |

## Estradiol -> DRSP: within-woman vs between-woman (z-scored E2; Mundlak split: within = E2 - woman mean, between = woman mean)

| Sample | Outcome | Model | within-woman b [95% CI], p | between-woman b [95% CI], p | permutation p within / between |
|---|---|---|---|---|---|
| long (n=176 rows, 65 women) | drsp_psych | E2 within/between | -0.48 [-0.68, -0.29], p=<0.001 | -0.04 [-0.27, +0.19], p=0.746 | 0.0002 / 0.717 |
| long | drsp_psych | E2+P within/between | -0.46 [-0.67, -0.25], p=<0.001 | -0.04 [-0.27, +0.20], p=0.751 | - |
| long | drsp_psych | log1p E2 within/between | -0.49 [-0.68, -0.31], p=<0.001 | -0.02 [-0.26, +0.21], p=0.859 | - |
| long (n=176 rows, 65 women) | drsp_total | E2 within/between | -0.53 [-0.73, -0.33], p=<0.001 | -0.02 [-0.25, +0.21], p=0.858 | 0.0002 / 0.826 |
| long | drsp_total | E2+P within/between | -0.50 [-0.72, -0.29], p=<0.001 | -0.02 [-0.25, +0.21], p=0.848 | - |
| long | drsp_total | log1p E2 within/between | -0.53 [-0.72, -0.35], p=<0.001 | -0.00 [-0.24, +0.23], p=0.971 | - |
| pooled (n=271 rows, 160 women) | drsp_psych | E2 within/between | -0.51 [-0.72, -0.31], p=<0.001 | -0.01 [-0.16, +0.14], p=0.879 | 0.0002 / 0.898 |
| pooled | drsp_psych | E2+P within/between | -0.49 [-0.71, -0.27], p=<0.001 | -0.01 [-0.16, +0.14], p=0.929 | - |
| pooled | drsp_psych | log1p E2 within/between | -0.53 [-0.72, -0.33], p=<0.001 | -0.01 [-0.16, +0.14], p=0.888 | - |
| pooled (n=271 rows, 160 women) | drsp_total | E2 within/between | -0.56 [-0.76, -0.35], p=<0.001 | -0.02 [-0.16, +0.13], p=0.813 | 0.0002 / 0.795 |
| pooled | drsp_total | E2+P within/between | -0.53 [-0.75, -0.31], p=<0.001 | -0.01 [-0.16, +0.14], p=0.927 | - |
| pooled | drsp_total | log1p E2 within/between | -0.56 [-0.75, -0.37], p=<0.001 | -0.02 [-0.17, +0.13], p=0.765 | - |

SD of the z-scored E2 parts, long (n=176 rows): within-woman 0.54, between-woman (woman mean) 0.84.

SD of the z-scored E2 parts, pooled (n=271 rows): within-woman 0.43, between-woman (woman mean) 0.90.

## Permutation tests (B as stated; p = (1+#extreme)/(B+1))

| Sample | n rows / women | Outcome | Test | Statistic | permutation p | B |
|---|---|---|---|---|---|---|
| long | 176 / 65 | pos_affect | phase omnibus F (within-woman) | 8.084 | 0.0006 | 5000 |
| long | 176 / 65 | neg_affect | phase omnibus F (within-woman) | 2.831 | 0.0598 | 5000 |
| long | 176 / 65 | drsp_psych | phase omnibus F (within-woman) | 18.128 | 0.0002 | 5000 |
| long | 176 / 65 | drsp_total | phase omnibus F (within-woman) | 27.148 | 0.0002 | 5000 |
| long | 176 / 65 | stai | phase omnibus F (within-woman) | 3.434 | 0.0316 | 5000 |
| long | 176 / 65 | pos_affect | E2 within-woman slope (|slope|) | 0.279 | 0.0050 | 5000 |
| long | 176 / 65 | neg_affect | E2 within-woman slope (|slope|) | -0.208 | 0.1324 | 5000 |
| long | 176 / 65 | drsp_psych | E2 within-woman slope (|slope|) | -0.482 | 0.0002 | 5000 |
| long | 176 / 65 | drsp_total | E2 within-woman slope (|slope|) | -0.530 | 0.0002 | 5000 |
| long | 176 / 65 | stai | E2 within-woman slope (|slope|) | -0.256 | 0.0138 | 5000 |
| long | 176 / 65 | pos_affect | E2 between-woman slope on woman means (|slope|) | -0.084 | 0.5019 | 5000 |
| long | 176 / 65 | neg_affect | E2 between-woman slope on woman means (|slope|) | -0.039 | 0.7453 | 5000 |
| long | 176 / 65 | drsp_psych | E2 between-woman slope on woman means (|slope|) | -0.043 | 0.7167 | 5000 |
| long | 176 / 65 | drsp_total | E2 between-woman slope on woman means (|slope|) | -0.026 | 0.8260 | 5000 |
| long | 176 / 65 | stai | E2 between-woman slope on woman means (|slope|) | 0.010 | 0.9386 | 5000 |
| pooled | 271 / 160 | pos_affect | phase omnibus F (ordinary F, stratified perms) | 4.562 | 0.0022 | 5000 |
| pooled | 271 / 160 | neg_affect | phase omnibus F (ordinary F, stratified perms) | 1.264 | 0.1906 | 5000 |
| pooled | 271 / 160 | drsp_psych | phase omnibus F (ordinary F, stratified perms) | 8.573 | 0.0004 | 5000 |
| pooled | 271 / 160 | drsp_total | phase omnibus F (ordinary F, stratified perms) | 14.946 | 0.0002 | 5000 |
| pooled | 271 / 160 | stai | phase omnibus F (ordinary F, stratified perms) | 0.117 | 0.8318 | 5000 |
| pooled | 271 / 160 | pos_affect | phase Wald chi2 of the mixed model, stratified perms | 16.785 | 0.0010 | 1000 |
| pooled | 271 / 160 | neg_affect | phase Wald chi2 of the mixed model, stratified perms | 4.758 | 0.1079 | 1000 |
| pooled | 271 / 160 | drsp_psych | phase Wald chi2 of the mixed model, stratified perms | 36.295 | 0.0010 | 1000 |
| pooled | 271 / 160 | drsp_total | phase Wald chi2 of the mixed model, stratified perms | 58.753 | 0.0010 | 1000 |
| pooled | 271 / 160 | stai | phase Wald chi2 of the mixed model, stratified perms | 2.723 | 0.2717 | 1000 |
| pooled | 271 / 160 | pos_affect | E2 within-woman slope (|slope|) | 0.294 | 0.0054 | 5000 |
| pooled | 271 / 160 | neg_affect | E2 within-woman slope (|slope|) | -0.227 | 0.1408 | 5000 |
| pooled | 271 / 160 | drsp_psych | E2 within-woman slope (|slope|) | -0.514 | 0.0002 | 5000 |
| pooled | 271 / 160 | drsp_total | E2 within-woman slope (|slope|) | -0.557 | 0.0002 | 5000 |
| pooled | 271 / 160 | stai | E2 within-woman slope (|slope|) | -0.271 | 0.0140 | 5000 |
| pooled | 271 / 160 | pos_affect | E2 between-woman slope on woman means (|slope|) | 0.052 | 0.4901 | 5000 |
| pooled | 271 / 160 | neg_affect | E2 between-woman slope on woman means (|slope|) | -0.033 | 0.6417 | 5000 |
| pooled | 271 / 160 | drsp_psych | E2 between-woman slope on woman means (|slope|) | -0.009 | 0.8980 | 5000 |
| pooled | 271 / 160 | drsp_total | E2 between-woman slope on woman means (|slope|) | -0.020 | 0.7954 | 5000 |
| pooled | 271 / 160 | stai | E2 between-woman slope on woman means (|slope|) | 0.027 | 0.7281 | 5000 |
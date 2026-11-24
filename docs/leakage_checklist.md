> **Working notes: the checked numbers are in docs/report.md**

# Data-leakage checklist (Phase 4)
| # | Risk | How it is prevented / checked | Where |
|---|---|---|---|
| 1 | Same woman in train and test (sessions of one woman are correlated) | Leave-one-woman-out: every outer fold holds out all rows of one VPN; `assert set(g_train).isdisjoint(g_test)` in every fold; inner CV also split by woman with the same assertion | `src/cvlib.py` (`_one`, `predict_one`) |
| 2 | Scaling / centring fitted on all data | Standardisation and centring are fitted on training rows only, inside `fit_predict_path`; trees need no scaling | `src/mathcore/ridge.py` |
| 3 | Hyperparameters chosen with the test woman | Nested CV: lambda / tree settings picked by 5-fold CV over the *training women only*; test woman never seen | `src/cvlib.py` |
| 4 | Target (or a transform of it) among the features | Guard asserts that the target and the other mood scores (Affect, DRSP, STAI, PANAS) are not in the feature list | `lopo()` in `src/cvlib.py` |
| 5 | Duplicate rows / overlap of the two samples | Assert no duplicate (VPN, Session); assert cross-sectional and longitudinal VPN sets are disjoint (cross-sectional = women with one included session) | `src/phase4_run.py`, `lopo()` |
| 6 | Baseline uses test information | Global-mean baseline = mean of the training rows of each fold only | `predict_one("baseline")` |
| 7 | Pipeline bug that leaks labels (any of the above, undetected) | **Shuffled-target test**: target permuted across rows, same pipeline (ridge, set D, pooled, 3 permutations per target). A leak-free pipeline must give R2_oos <= ~0. Result: mean -0.0024 (DRSP), -0.0041 (positive affect), -0.0029 (anxiety); max over all 9 runs +0.0001 | `src/phase4_leak_loso.py`, `results/phase4_shuffle_test.csv` |
| 8 | Measuring the same construct as the target (not leakage in the code, but overlap in content) | Feature set D2 = D without PSST, reported side by side with D | see `docs/phase4_results.md` |
| 9 | Personal history in the leave-one-session-out experiment | Personal mean for each row excludes the row itself AND the held-out session (training rows of the held-out woman too) | `pm_feature()` in `src/phase4_leak_loso.py` |
| 10 | Hormones measured in the same saliva session as mood | Not leakage (both are same-day), but means the models "predict" a concurrent state, not a future one; stated as a limitation | report |
Not checked / residual: authors' `include` filter was applied on their criteria, which I cannot audit; one woman has 2 sessions only.

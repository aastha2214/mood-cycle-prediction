# results/ index

Key outputs. Everything is reproduced by `./run_all.sh`; the documents in `docs/` read these files.

## Figures

- `fig_3b_loss_curves.png`: GD/SGD/momentum/Adam loss curves
- `fig_3b_lr_study.png`: learning-rate study
- `fig_3b_momentum_tuned.png`: tuned heavy-ball vs GD
- `fig_3c_hessian.png`: Hessian spectrum and condition number
- `fig_3d_path_biasvar.png`: ridge path and bias-variance
- `fig_4_skill.png`: out-of-sample skill by model, feature set, target, sample
- `fig_6a_negctrl.png`: negative control (emotion recognition)
- `fig_6b_confusion.png`: phase from hormones: confusion matrices

## Key tables (CSV)

- `icc.csv`: ICC of the three targets
- `phase3c_condition.csv`: condition numbers
- `phase4_metrics.csv`: MAE/RMSE/R2/R2_oos with bootstrap CIs
- `phase4_deltas.csv`: paired differences between feature sets
- `phase4_loso.csv`: leave-one-session-out
- `phase5_mixed.csv`: mixed-model estimates
- `phase5_perm.csv`: permutation tests
- `phase5_r2_phase.csv`: marginal/conditional R2
- `phase5_multcomp.csv`: Holm/FDR on 27 comparisons
- `phase6a_metrics.csv`: negative control metrics
- `phase6b_metrics.csv`: phase classification metrics

## Readable tables (Markdown)

- `phase4_tables.md`: all Phase 4 tables
- `phase5_tables.md`: authors vs mine, within/between, permutations

## Logs

- `logs/timings.csv`: seconds per step of run_all.sh

## Extension outputs (Phase 7; CSV and figures)

- `extE1_deltas.csv`: 
- `extE1_hormone_corr.csv`: 
- `extE1_identity_check.csv`: 
- `extE1_loadings.csv`: 
- `extE1_metrics_all.csv`: 
- `extE1_oof.csv`: 
- `extE1_pca_summary.csv`: 
- `extE1_pcr_k.csv`: 
- `extE2_calibration.csv`: 
- `extE2_metrics_all.csv`: 
- `extE2_oof.csv`: 
- `extE2_sklearn_check.csv`: 
- `extE2_vs_ridge.csv`: 
- `extE3_confusion.csv`: 
- `extE3_metrics.csv`: 
- `extE3_oof.csv`: 
- `extE3_params.csv`: 
- `extE3_sklearn_check.csv`: 
- `extE4_coverage.csv`: 
- `extE4_intervals_oof.csv`: 
- `extE5_ablation_deltas.csv`: 
- `extE5_ablation_metrics.csv`: 
- `extE5_ablation_oof.csv`: 
- `extE5_lc_identity_check.csv`: 
- `extE5_learning_curves.csv`: 
- `extOF_train_test.csv`: 
- `ext_e7_coef_shift.csv`: 
- `ext_e7_deltas.csv`: 
- `ext_e7_hormone_tests.csv`: 
- `ext_e7_metrics.csv`: 
- `ext_e7_oof.csv`: 
- `ext_e7_phase4_check.csv`: 
- `ext_e7_phase_mix.csv`: 
- `ext_e7_ridge_parity.csv`: 
- `ext_e7_shift_auc.csv`: 
- `ext_e7_shift_by_phase.csv`: 
- `ext_e7_shift_summary.csv`: 
- `ext_e7_shift_variables.csv`: 
- `ext_e7_sizes.csv`: 

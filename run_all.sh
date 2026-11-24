#!/usr/bin/env bash
# Reproduce every result, figure, table and document. Fixed seeds (SEED=42 in src/data.py); run from anywhere.
# Usage: ./run_all.sh            (logs in results/logs/, per-step seconds in results/logs/timings.csv)
set -euo pipefail
cd "$(dirname "$0")"
PY=./.venv/bin/python
mkdir -p results/logs; echo "step,seconds" > results/logs/timings.csv
run() {  # run <script.py> : python -I isolates imports; stdout/stderr go to results/logs/<script>.log
  local s=$1 t0 t1; t0=$(date +%s)
  if ! "$PY" -I "src/$s" > "results/logs/${s%.py}.log" 2>&1; then echo "FAILED: $s (see results/logs/${s%.py}.log)"; tail -15 "results/logs/${s%.py}.log"; exit 1; fi
  t1=$(date +%s); echo "$s,$((t1-t0))" >> results/logs/timings.csv; printf '%-34s %5ss\n' "$s" "$((t1-t0))"
}
"$PY" -I src/download_data.py                                   # verify data (downloads only if missing)
# Phase 1 - inspection (console reports only)
for s in inspect_data.py inspect2.py inspect3.py inspect4.py inspect5.py design_checks.py; do run $s; done
# Design support + Phase 3 - maths core
run icc.py; run phase3a_verify.py; run phase3b_optim.py; run phase3b2_momentum_tuned.py; run phase3c_hessian.py; run phase3d_path_cv.py
# Phase 4 - ML with leave-one-woman-out + nested CV
run phase4_run.py; run phase4_report.py; run phase4_tables.py; run phase4_leak_loso.py
# Phase 5 - statistics
run phase5_stats.py; run phase5_tables.py; run phase5_e2_given_phase.py; run phase5_multcomp.py
# Phase 6 - credibility checks
run phase6_negctrl.py; run phase6_phase_clf.py
# Phase 7 - extensions (Group A: E2 Bayesian ridge, E3 Gaussian Bayes classifier, E1 PCA)
run ext_E2_bayes_ridge.py; run ext_E3_gaussian_bayes.py; run ext_E1_pca.py
# Phase 7 - extensions (Group B: E4 prediction intervals, E5 ablation + learning curves, train-vs-test overfitting check)
run ext_E4_intervals.py; run ext_E5_ablation_lc.py; run ext_OF_train_test_gap.py
# Phase 7 - E7 cross-sample validation (single-session women <-> longitudinal women) and distribution-shift analysis
run ext_e7_cross_sample.py; run ext_e7_shift.py
# Phase 8 - report, provenance, viva Q&A, results index (every number loaded from results/*.csv)
run build_report.py
run audit_numbers.py                                             # every numeric token in report/viva/README must be loaded or structural
echo "done. See docs/report.md, docs/numbers_provenance.md, docs/viva_qa.md, results/INDEX.md"

# E6: educational demo (Streamlit)

**Educational demo, not medical advice.** A learning project on one open research dataset; do not use it for health decisions.

## Start it
From the project root (the virtual environment already contains Streamlit):
```bash
streamlit run app/app.py
```
(with the project's environment: `./.venv/bin/streamlit run app/app.py`). To avoid Streamlit's first-run e-mail prompt and its usage statistics, add `--server.headless true --browser.gatherUsageStats false`. The app opens in the browser on this computer; nothing is sent anywhere.

## What it does
Sidebar inputs: cycle phase, estradiol, progesterone, PSST, BDI and BAI (ranges and starting values come from the data). For positive affect, DRSP psychological symptoms and state anxiety it shows the ridge prediction, the Bayesian ridge predictive mean with a 90% prediction interval (one new session), the training-mean baseline, and the leave-one-woman-out test R2 with its confidence interval for this model, plus a chart, a plain-language box and two expanders on the width and the failed cross-sample coverage of the intervals (E4, E7) and on the hormone inputs.

## Model
Full feature set D exactly as validated (phase, log1p estradiol, log1p progesterone, PSST, BDI, BAI), standardised with the training statistics; ridge with lambda chosen by the same inner grouped cross-validation as in the nested CV; Bayesian ridge from E2 (evidence maximisation); both trained on all women of the analysis set. `app/build_model.py` fits them from the data and writes `app/model_artifacts.json` (the only file that is derived from the raw data); it checks the ridge solution against the validation pipeline and the Bayesian ridge against scikit-learn.

## Where every number comes from
- Predictions: computed in the app from `app/model_artifacts.json` and the entered values.
- Test R2 and confidence intervals, hormone comparison counts, interval coverage and widths, cross-sample R2: loaded when the page is rendered from `results/phase4_metrics.csv`, `results/phase4_deltas.csv`, `results/extE4_coverage.csv`, `results/ext_e7_metrics.csv`, `results/ext_e7_hormone_tests.csv`, `results/ext_e7_sizes.csv`.
- Nothing is typed by hand in the displayed text. The app does not read the raw data.

## Files and rules
`app/app.py` (the app), `app/build_model.py` (fits the final model), `app/model_artifacts.json` (fitted model), `app/test_app.py` (headless tests: `./.venv/bin/python -I app/test_app.py`).
The demo adds files only inside `app/`; it modifies no existing file, `run_all.sh` or the project README, and it is not part of `run_all.sh`. User inputs are kept in memory for the browser session only: no file is written, nothing is logged or uploaded (the test checks that running the app creates or modifies no file).
If you change the data or the models, re-run `./.venv/bin/python -I app/build_model.py`.

## Limits (see the report)
Validated on one open study; hormone inputs add nothing beyond phase; prediction intervals are wide and did not hold across the two groups of women; predictions are for a single new session, not forecasts.

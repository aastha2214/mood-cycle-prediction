# Course-project demo (Streamlit)

Demo of a course project on one open research dataset; not a medical or diagnostic tool.

## Start it
From the project root, with the project's environment:
```bash
./.venv/bin/streamlit run app/app.py --server.headless true --browser.gatherUsageStats false
```
The app opens in the browser on this computer. Nothing you enter is stored, logged or sent anywhere.

## Tabs
- **Predict**: inputs (cycle phase, estradiol, progesterone, PSST, BDI, BAI), three presets, and for each target (positive affect, DRSP psychological symptoms, state anxiety) the prediction, the 90% prediction interval and the training-mean baseline.
- **What drives the prediction**: coefficient times standardized input for each input, in the target's units.
- **Hormone sweep**: estradiol and progesterone varied over the observed range, everything else fixed.
- **Compare scenarios**: two input sets side by side with the difference.
- **Study results**: out-of-sample R2 for feature sets A, C and D, the hormone-versus-phase comparison counts, cross-sample R2 and interval coverage, all loaded from `results/`.

Sidebar: choice of model (ridge or Bayesian ridge point prediction; the interval is always the Bayesian predictive interval of one new session). Only feature set D (phase, hormones, PSST, BDI, BAI) is stored in the model file; sets A and C appear only as validated results in the Study results tab.

## Where every number comes from
- Predictions, contributions, sweeps and scenarios: computed from `app/model_artifacts.json` and the entered values.
- Everything in the Study results tab, the hormone counts and the preset hormone values: read when the page is rendered from `results/phase4_metrics.csv`, `phase4_deltas.csv`, `extE4_coverage.csv`, `ext_e7_metrics.csv`, `ext_e7_sizes.csv` and `phase5_hormones_by_phase_long.csv`. The app does not read the raw data.

## Files
`app.py` (the page), `app_core.py` (the functions it uses), `build_model.py` (fits the final model and writes `model_artifacts.json`; re-run it with `./.venv/bin/python -I app/build_model.py` if the data or models change), `model_artifacts.json`, `requirements.txt` (packages the page imports, for hosted deployment with main file `app/app.py` and Python 3.12), `test_app.py` (`./.venv/bin/python -I app/test_app.py`: tests of the functions, of the result tables and of the page; it also checks that running the app creates or modifies no file).
The demo is not part of `run_all.sh`.

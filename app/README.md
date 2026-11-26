# Course-project demo (Streamlit)

Demo of a course project on one open research dataset; not a medical or diagnostic tool.

## Start it
From the project root, with the project's environment:
```bash
./.venv/bin/streamlit run app/app.py --server.headless true --browser.gatherUsageStats false
```

## Tabs
- **Predict**: key findings, inputs, presets and reset; prediction, 90% prediction interval and training-mean baseline for positive affect, DRSP psychological symptoms and state anxiety; CSV download.
- **What drives the prediction**: coefficient times standardized input for each input.
- **Hormone sweep**: estradiol and progesterone varied over the observed range.
- **Compare scenarios**: two input sets side by side, with CSV download.
- **Study results**: out-of-sample R² (sets A, C, D), hormone-versus-phase counts, cross-sample R², interval coverage and calibration, glossary.
- **Method lab**: illustrative NumPy examples (gradient descent versus momentum, ridge shrinkage), plus the study's condition numbers.

Sidebar: model (ridge or Bayesian ridge), presentation mode (larger text), link to the source code. The Predict-tab inputs and the model are kept in the page address, so a link restores them. The app does not save entered values. In the hosted demo they are processed on Streamlit's server, and the Predict-tab inputs appear in the web address when you share it. Run the app locally to keep everything on your machine.
Only feature set D (phase, hormones, PSST, BDI, BAI) is in the model file; sets A and C appear only as saved study results.

## Where the numbers come from
Predictions are computed from `model_artifacts.json` and the entered values. Study results, key findings and presets are read from the CSV files in `results/`. The Method lab examples are synthetic. The app does not read the raw data.

## Files
`app.py` (page), `app_core.py` (functions), `app_plots.py` (figures), `method_lab.py` (NumPy examples), `build_model.py` (writes `model_artifacts.json`; re-run if data or models change), `requirements.txt` (hosted deployment: main file `app/app.py`, Python 3.12), `test_app.py` (run with `./.venv/bin/python -I app/test_app.py`).
The demo is not part of `run_all.sh`.

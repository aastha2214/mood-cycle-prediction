# Mood Prediction from Menstrual Cycle Data: Ridge Regression, Optimization and Bayesian Methods from Scratch

**Question.** Mood is often said to follow the menstrual cycle. This project asks a practical version: if we know a woman's cycle phase and the estradiol and progesterone in her saliva, can we predict her mood better than by simply guessing the average, for a woman the model has never seen?
- *Hormones add nothing to cycle phase for unseen women.* Of 27 paired comparisons (phase + hormones versus phase alone) 0 showed an improvement and 15 were worse; the smallest uncorrected p-value for an improvement was 0.334.
- *Phase alone helps only a little, and only for premenstrual symptoms.* Share of the error removed compared with guessing the average (ridge, pooled): DRSP psychological symptoms +0.051 [+0.005, +0.095], positive affect +0.021 [-0.009, +0.051], state anxiety -0.001 [-0.001, -0.000].
- *Questionnaires, and above all the woman's own history, carry the predictable signal.* With BDI, BAI and PSST added the DRSP figure rises to +0.360 [+0.247, +0.457]; giving the model a woman's own mean from her other sessions lifted a phase-plus-hormone model from +0.034 [-0.020, +0.084] to +0.324 [+0.152, +0.459] (longitudinal women).

**Dataset.** Pletzer and Noachtar (2023), "Emotion recognition and mood along the menstrual cycle", *Hormones and Behavior* 154:105406; open data on OSF, project `dmhrw` (https://osf.io/dmhrw/). The raw files are not stored in this repository (see "Getting the data"). Analysis set: 271 rows, 160 women (176 rows, 65 women longitudinal; 95 rows, 95 women cross-sectional).

**Status.** Phases 1 to 6 and 8 are done, and Phase 7 extensions E1 (PCA), E2 (Bayesian ridge), E3 (Gaussian Bayes classifier), E4 (prediction intervals), E5 (ablation and learning curves), E7 (cross-sample validation) plus a train-versus-test overfitting check are done. E6 (Streamlit demo) is available in `app/`. *Learning project, not a medical tool.*

## Quick start
```bash
cd mood-cycle-prediction
python3 -m venv .venv                      # Python 3.12 was used
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -I src/download_data.py # download the OSF data (project dmhrw) into data/ and verify it against data/MANIFEST.md
bash run_all.sh                            # everything; between 30 and 35 minutes on the author's laptop
./.venv/bin/streamlit run app/app.py       # optional demo app (reads the results written by run_all.sh)
```
`run_all.sh` verifies the four data files against `data/MANIFEST.md` (SHA-256; it downloads them from OSF only if missing, via `src/download_data.py`), runs every step with `python -I`, writes one log per step to `results/logs/` and the seconds per step to `results/logs/timings.csv`, rebuilds the documents and finally audits them. It stops at the first failing step. All randomness uses seed **42** (`src/data.py`); joblib uses 10 parallel workers (the number of workers does not change any result, because every fold has its own fixed seed).

**Reproducibility.** `bash check_reproducible.sh` copies the generated results and documents, deletes them, re-runs `run_all.sh` from scratch and compares every file byte for byte (logs and timings excluded); it prints the number of identical and of different files.

## Getting the data (not stored in this repository)
The raw data are public (OSF project `dmhrw`, Pletzer & Noachtar 2023) and are kept out of the repository; put them in `data/`:
1. *Automatic:* `./.venv/bin/python -I src/download_data.py` downloads the four files (three `.sav` files and the authors' R script, which is read only to understand variable coding) from OSF into `data/` if they are missing and checks each one against `data/MANIFEST.md`. `bash run_all.sh` calls it as its first step.
2. *Manual:* open https://osf.io/dmhrw/ and download the same four files into `data/` under their original names.
3. *Verify:* `cd data && shasum -a 256 -c MANIFEST.md` (macOS) or `sha256sum -c MANIFEST.md` (Linux); every line must report OK. `data/MANIFEST.md` lists the SHA-256 checksum of each file.

## Run everything
`bash run_all.sh` (after the environment and the data are in place, see Quick start) reruns every script with the fixed seed and rebuilds `docs/report.md`, `docs/numbers_provenance.md`, `results/INDEX.md` and this README. `bash check_reproducible.sh` additionally checks that a rerun reproduces every file byte for byte.

## Demo app
After `bash run_all.sh` (the app reads only the saved files in `results/` and `app/model_artifacts.json`): `streamlit run app/app.py`. To skip Streamlit's first-run prompt and its usage statistics add `--server.headless true --browser.gatherUsageStats false`. `app/build_model.py` refits the final model to `app/model_artifacts.json`, `app/test_app.py` runs headless tests; see `app/README.md`. The values entered in the app are not stored or sent anywhere.

## Approximate runtimes
The classes below are derived from the reference timings in `docs/runtime_reference.csv`, measured once on the author's laptop while other programs were running, so a quiet machine may be faster; the seconds of your own run are written to `results/logs/timings.csv`.

| Step | Runtime |
|---|---|
| Phase 1/2 inspection printouts (`inspect*.py`, `design_checks.py`) | seconds |
| `icc.py`: ICCs with women-bootstraps | about a minute |
| Phase 3 maths core (`phase3*.py`) | seconds |
| `phase4_run.py`: **main experiment** (baseline/ridge/RF/GB, all sets, three samples, leave-one-woman-out with nested CV) | a few minutes |
| Phase 4 metrics, tables, leakage test and leave-one-session-out (`phase4_report.py`, `phase4_tables.py`, `phase4_leak_loso.py`) | seconds |
| Phase 5 statistics (`phase5_*.py`) | about a minute |
| `phase6_negctrl.py`: negative control | a few minutes |
| `phase6_phase_clf.py`: phase from hormones | a few minutes |
| E2 Bayesian ridge and E3 Gaussian Bayes classifier | seconds |
| E1 PCA | a few minutes |
| E4 prediction intervals | seconds |
| E5 ablation and learning curves | a few minutes |
| train-versus-test overfitting check | about a minute |
| E7 cross-sample validation and distribution-shift analysis | about a minute |

## What is where
| Path | Content |
|---|---|
| `docs/report.md` | **the report**: Summary of findings, Abstract, Introduction, Data, Methods, Results (including the train-versus-test table in 4.2), Discussion, Limitations, Conclusion, and the extensions E1 to E5 and E7 as sections 8.1 to 8.6. Every number is loaded from `results/*.csv` or the data when the file is built. |
| `docs/numbers_provenance.md` | maps every number in the report and this README to its source file, selection, column and script |
| `results/numbers_audit.md` | automatic audit: every numeric token in the report and in this README is either a logged value or a listed structural token |
| `docs/math/` | one note per maths-core item (3a to 3e) and per extension (`E1_pca.md`, `E2_bayesian_ridge.md`, `E3_gaussian_bayes_classifier.md`, `E4_prediction_intervals.md`, `E5_ablation_and_learning_curves.md`, `E7_cross_sample.md`) |
| `results/INDEX.md` | key figures and tables; `results/fig_*.png`, `results/*.csv`, `results/phase*_tables.md` |
| `src/` | code: `data.py`, `mathcore/` (from-scratch NumPy ridge and optimisers), `cvlib.py` / `cvclf.py` (leave-one-woman-out with nested CV), `extlib.py` / `evalutils.py` (bootstraps), `phase*.py`, `ext_*.py` (extensions), `report_lib.py` / `build_report.py` / `viva_content.py` / `readme_content.py` / `audit_numbers.py` (documents) |
| `data/` | the four OSF files and `MANIFEST.md` (SHA-256) |

## Checked results versus working notes
*Checked* (written by scripts from saved results, covered by the number audit): `docs/report.md`, `docs/numbers_provenance.md`, this README, `results/*.csv`, the figures, `results/phase4_tables.md`, `results/phase5_tables.md`, `results/INDEX.md`.
*Working notes* (written by hand from printed output; numbers not machine-checked, the report supersedes them): `docs/data_dictionary.md`, `docs/design.md`, `docs/leakage_checklist.md`, `docs/cross_dataset_candidates.md` (web research on possible second datasets, not machine-checked) and the maths-core notes `docs/math/3a_ridge_closed_form.md` to `docs/math/3e_map_gaussian_prior.md`. All of these carry the header line "Working notes: the checked numbers are in docs/report.md"; the extension notes `docs/math/E1_*.md` to `E5_*.md` carry it too but contain no results.

## Main results (details, intervals and sample sizes in `docs/report.md`)
- On 271 rows, 160 women, adding hormones to cycle phase improved out-of-sample prediction in 0 of 27 paired comparisons and made it worse in 15; questionnaires (BDI, BAI) and the woman's own history were far more informative.
- Group-level effects reproduce the authors' pattern (their numbers appear only as labelled reference values); the estradiol-DRSP association is within women and confounded with phase.
- Credibility checks: cycle phase is predictable from hormones (balanced accuracy 0.70 [0.65, 0.75] versus chance 0.333); emotion-recognition accuracy is not (one small, nominal speed caveat).

## Licence and data
The code is released under the MIT licence (`LICENSE`). The licence covers the code only; it does not cover the data (OSF project `dmhrw`).
`results/*_oof.csv` are derived from the OSF dataset (project dmhrw), which has no licence set; they are included for evaluation only and can be regenerated with `run_all.sh`.

## Notes
- The authors' R script in `data/` is used only to understand variable coding; it is never run.
- The data are public but kept local; nothing is sent anywhere.

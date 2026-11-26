# Executed inside src/build_report.py: writes README.md. Every number comes from the saved results, the code or the timing log (logged for provenance).
import sys as _sys
section("README")
_t = pd.read_csv(ROOT / "docs/runtime_reference.csv").set_index("step").seconds   # fixed reference timings of one full run (so that the README does not change from run to run)
def _cls(secs): return "seconds" if secs < 15 else ("about a minute" if secs < 120 else ("a few minutes" if secs < 420 else "many minutes"))
_grp = [("Phase 1/2 inspection printouts (`inspect*.py`, `design_checks.py`)", [x for x in _t.index if x.startswith("inspect") or x == "design_checks.py"]), ("`icc.py`: ICCs with women-bootstraps", ["icc.py"]), ("Phase 3 maths core (`phase3*.py`)", [x for x in _t.index if x.startswith("phase3")]),
        ("`phase4_run.py`: **main experiment** (baseline/ridge/RF/GB, all sets, three samples, leave-one-woman-out with nested CV)", ["phase4_run.py"]), ("Phase 4 metrics, tables, leakage test and leave-one-session-out (`phase4_report.py`, `phase4_tables.py`, `phase4_leak_loso.py`)", ["phase4_report.py", "phase4_tables.py", "phase4_leak_loso.py"]),
        ("Phase 5 statistics (`phase5_*.py`)", [x for x in _t.index if x.startswith("phase5")]), ("`phase6_negctrl.py`: negative control", ["phase6_negctrl.py"]), ("`phase6_phase_clf.py`: phase from hormones", ["phase6_phase_clf.py"]),
        ("E2 Bayesian ridge and E3 Gaussian Bayes classifier", ["ext_E2_bayes_ridge.py", "ext_E3_gaussian_bayes.py"]), ("E1 PCA", ["ext_E1_pca.py"]), ("E4 prediction intervals", ["ext_E4_intervals.py"]), ("E5 ablation and learning curves", ["ext_E5_ablation_lc.py"]), ("train-versus-test overfitting check", ["ext_OF_train_test_gap.py"]), ("E7 cross-sample validation and distribution-shift analysis", ["ext_e7_cross_sample.py", "ext_e7_shift.py"])]
_rt = "\n".join(f"| {nm} | {_cls(float(_t.reindex(ks).sum()))} |" for nm, ks in _grp if len(_t.reindex(ks).dropna()))
_tot = float(_t.sum()); _lo = int(_tot // 60 // 5 * 5); _hi = _lo + 5
_nw = const('src/cvlib.py', r'def lopo\([^)]*n_jobs=(\d+)', 'parallel workers (default of the leave-one-woman-out loop)'); _wk = 'worker' if _nw == '1' else 'workers'
_sm = SUMMARY.split("\n"); _q = next(l for l in _sm if l.startswith("**The question.**")).split(" Data:")[0].replace("**The question.**", "**Question.**"); _items = [l[3:] for l in _sm if l[:3] in ("1. ", "2. ", "3. ")]
_findings = "\n".join("- " + x for x in _items)
_live = "https://mood-cycle-prediction.streamlit.app"
R_ = f"""# Mood Prediction from Menstrual Cycle Data: Ridge Regression, Optimization and Bayesian Methods from Scratch

Live demo: {_live}

{_q}
{_findings}

**Dataset.** Pletzer and Noachtar (2023), "Emotion recognition and mood along the menstrual cycle", *Hormones and Behavior* 154:105406; open data on OSF, project `dmhrw` (https://osf.io/dmhrw/). The raw files are not stored in this repository (see "Getting the data"). Analysis set: {nn('pooled')} ({nn('long')} longitudinal; {nn('cross')} cross-sectional).

**Status.** Phases 1 to 6 and 8 are done, and Phase 7 extensions E1 (PCA), E2 (Bayesian ridge), E3 (Gaussian Bayes classifier), E4 (prediction intervals), E5 (ablation and learning curves), E7 (cross-sample validation) plus a train-versus-test overfitting check are done. E6 (Streamlit demo) is deployed at {_live} and lives in `app/`. *Learning project, not a medical tool.*

## Quick start
```bash
cd mood-cycle-prediction
python3 -m venv .venv                      # Python {calc('python major', _sys.version_info.major, '{:.0f}', 'python runtime', 'sys.version_info.major')}.{calc('python minor', _sys.version_info.minor, '{:.0f}', 'python runtime', 'sys.version_info.minor')} was used
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -I src/download_data.py # download the OSF data (project dmhrw) into data/ and verify it against data/MANIFEST.md
bash run_all.sh                            # everything; between {calc('runtime lower bound (min)', _lo, '{:.0f}', 'docs/runtime_reference.csv', 'floor(total seconds / 60 / 5) * 5')} and {calc('runtime upper bound (min)', _hi, '{:.0f}', 'docs/runtime_reference.csv', 'lower bound + 5')} minutes on the author's laptop
./.venv/bin/streamlit run app/app.py       # optional demo app (reads the results written by run_all.sh)
```
`run_all.sh` verifies the four data files against `data/MANIFEST.md` (SHA-256; it downloads them from OSF only if missing, via `src/download_data.py`), runs every step with `python -I`, writes one log per step to `results/logs/` and the seconds per step to `results/logs/timings.csv`, rebuilds the documents and finally audits them. It stops at the first failing step. All randomness uses seed **{calc('random seed', SEED, '{:.0f}', 'src/data.py', 'SEED')}** (`src/data.py`); joblib uses {_nw} parallel {_wk} (the number of {_wk} does not change any result, because every fold has its own fixed seed).

**Reproducibility.** `bash check_reproducible.sh` copies the generated results and documents, deletes them, re-runs `run_all.sh` from scratch and compares every file byte for byte (logs and timings excluded); it prints the number of identical and of different files.

## Getting the data (not stored in this repository)
The raw data are public (OSF project `dmhrw`, Pletzer & Noachtar 2023) and are kept out of the repository; put them in `data/`:
1. *Automatic:* `./.venv/bin/python -I src/download_data.py` downloads the four files (three `.sav` files and the authors' R script, which is read only to understand variable coding) from OSF into `data/` if they are missing and checks each one against `data/MANIFEST.md`. `bash run_all.sh` calls it as its first step.
2. *Manual:* open https://osf.io/dmhrw/ and download the same four files into `data/` under their original names.
3. *Verify:* `cd data && shasum -a 256 -c MANIFEST.md` (macOS) or `sha256sum -c MANIFEST.md` (Linux); every line must report OK. `data/MANIFEST.md` lists the SHA-256 checksum of each file.

## Run everything
`bash run_all.sh` (after the environment and the data are in place, see Quick start) reruns every script with the fixed seed and rebuilds `docs/report.md`, `docs/numbers_provenance.md`, `results/INDEX.md` and this README. `bash check_reproducible.sh` additionally checks that a rerun reproduces every file byte for byte.

## Demo app
Live demo: {_live}

- **Tabs:** Predict, What drives the prediction, Hormone sweep, Compare scenarios, Study results, Method lab.
- **Extras:** Presentation mode, CSV download, Reset inputs, and shareable links via the web address.

To run it locally after `bash run_all.sh` (the app reads only the saved files in `results/` and `app/model_artifacts.json`): `streamlit run app/app.py`. To skip Streamlit's first-run prompt and its usage statistics add `--server.headless true --browser.gatherUsageStats false`. `app/build_model.py` refits the final model to `app/model_artifacts.json`, `app/test_app.py` runs headless tests; see `app/README.md`. The app does not save entered values. In the hosted demo they are processed on Streamlit's server, and the Predict-tab inputs appear in the web address, so take care when sharing it. To keep everything on your machine, run the app locally with `--browser.gatherUsageStats false`.

## Approximate runtimes
The classes below are derived from the reference timings in `docs/runtime_reference.csv`, measured once on the author's laptop while other programs were running, so a quiet machine may be faster; the seconds of your own run are written to `results/logs/timings.csv`.

| Step | Runtime |
|---|---|
{_rt}

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
- On {nn('pooled')}, adding hormones to cycle phase improved out-of-sample prediction in {n_ca_better} of {n_ca_all} paired comparisons and made it worse in {n_ca_worse}; questionnaires (BDI, BAI) and the woman's own history were far more informative.
- Group-level effects reproduce the authors' pattern (their numbers appear only as labelled reference values); the estradiol-DRSP association is within women and confounded with phase.
- Credibility checks: cycle phase is predictable from hormones (balanced accuracy {ci(B6, dict(sample='pooled', model='logreg'), 'bal_acc', '{:.2f}', lo='bal_acc_lo', hi='bal_acc_hi', label='balacc pooled logreg')} versus chance {calc('chance', 1/3, '{:.3f}', 'definition', '1/3 for 3 classes')}); emotion-recognition accuracy is not (one small, nominal speed caveat).

## Licence and data
The code is released under the MIT licence (`LICENSE`). The licence covers the code only; it does not cover the data (OSF project `dmhrw`).
`results/*_oof.csv` are derived from the OSF dataset (project dmhrw), which has no licence set; they are included for evaluation only and can be regenerated with `run_all.sh`.

## Notes
- The authors' R script in `data/` is used only to understand variable coding; it is never run.
- The data are public but kept local; the analysis scripts send nothing anywhere. The hosted demo is different: values entered there are processed on Streamlit's server (see "Demo app").
"""
(ROOT / "README.md").write_text(R_)

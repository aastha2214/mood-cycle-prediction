"""Headless tests of the demo (no browser, no network): streamlit.testing AppTest + independent recomputation of the numbers. Run from the project root:  ./.venv/bin/python -I app/test_app.py"""
import sys, json, os, time; from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from sklearn.linear_model import BayesianRidge
from streamlit.testing.v1 import AppTest
from src.data import load_analysis, FEATURE_SETS, TARGETS, SEED
from src.cvlib import predict_one
def snapshot(): return {str(p): p.stat().st_mtime_ns for p in ROOT.rglob("*") if p.is_file() and ".venv" not in p.parts and "__pycache__" not in p.parts and ".git" not in p.parts}
before = snapshot(); t0 = time.time()
at = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=60).run(); assert not at.exception, [e.value for e in at.exception]
assert len(at.error) == 0, "the red banner should be gone"; caps = [c.value for c in at.caption]; assert "Learning project built on one open research dataset." in caps
assert len(at.info) == 1 and "educational demo" not in at.info[0].value.lower() and "so it says nothing about other populations." in at.info[0].value and at.info[0].value.rstrip().endswith("other populations.")
print("1. default run: no exception;", len(at.table), "table,", len(at.expander), "expanders, red banner gone:", len(at.error) == 0, "| grey caption present | blue box ends after the validation sentence")
# independent recomputation: train ridge and Bayesian ridge on all 160 women the way the validation does, predict for several input sets, compare with the app's table
d = load_analysis(); cols = FEATURE_SETS["D"]; X, g = d[cols].values, d.VPN.values; art = json.loads((ROOT / "app/model_artifacts.json").read_text())
def reference(t, ph, e2, p, pms, bdi, bai):
    tc = TARGETS[t]; y = d[tc].values; x = np.array([[ph == "peri-ovulatory", ph == "luteal", np.log1p(e2), np.log1p(p), pms, bdi, bai]], float)
    ridge = predict_one("ridge", X, y, g, x, SEED)[0][0]; mu, sd = X.mean(0), X.std(0); Z = (X - mu) / sd; sk = BayesianRidge(max_iter=100000, tol=1e-13, alpha_1=0, alpha_2=0, lambda_1=0, lambda_2=0).fit(Z, y); mean, std = sk.predict((x - mu) / sd, return_std=True); return ridge, mean[0], std[0]
cases = [("menses", None, None, None, None, None), ("luteal", 1.2, 150.0, 15, 25, 20), ("peri-ovulatory", 0.3, 20.0, 3, 2, 1)]
IN = art["inputs"]; worst = 0.0
for ph, e2, pg, pms, bdi, bai in cases:
    at.sidebar.selectbox[0].set_value(ph)
    if e2 is not None:
        at.sidebar.slider[0].set_value(e2); at.sidebar.slider[1].set_value(pg); at.sidebar.slider[2].set_value(pms); at.sidebar.slider[3].set_value(bdi); at.sidebar.slider[4].set_value(bai)
    at.run(); assert not at.exception, [e.value for e in at.exception]
    vals = [at.sidebar.slider[i].value for i in range(5)]; tab = at.table[0].value
    for t, name in {"pos_affect": "Positive affect", "drsp_psych": "DRSP psychological symptoms", "stai": "State anxiety"}.items():
        r, m, s = reference(t, ph, *vals); row = tab.loc[name]; lo, hi = [float(v) for v in row["90% prediction interval"].split(" to ")]; z = art["z_level"]
        for got, want in [(float(row["Ridge prediction"]), r), (float(row["Bayesian predictive mean"]), m), (lo, m - z * s), (hi, m + z * s)]: worst = max(worst, abs(got - want)); assert abs(got - want) < 6e-3, (t, ph, got, want)
print(f"2. predictions for {len(cases)} input sets x 3 targets agree with an independent recomputation (largest difference of the displayed 2-decimal values {worst:.4f})")
# extremes of every slider: no crash, interval ordering sane
for combo in [("menses", IN["estradiol"]["min"], IN["progesterone"]["min"], IN["PMS"]["min"], IN["BDI_RW"]["min"], IN["BAI_RW"]["min"]), ("luteal", IN["estradiol"]["max"], IN["progesterone"]["max"], IN["PMS"]["max"], IN["BDI_RW"]["max"], IN["BAI_RW"]["max"])]:
    at.sidebar.selectbox[0].set_value(combo[0]); [at.sidebar.slider[i].set_value(float(v) if i < 2 else int(v)) for i, v in enumerate(combo[1:])]; at.run(); assert not at.exception
    for _, row in at.table[0].value.iterrows(): lo, hi = [float(v) for v in row["90% prediction interval"].split(" to ")]; assert lo < float(row["Bayesian predictive mean"]) < hi
print("3. minimum and maximum of every input: no exception, intervals contain the predictive mean")
# numbers shown for validation equal the saved result files
m4 = pd.read_csv(ROOT / "results/phase4_metrics.csv"); tab = at.table[0].value; r = m4[(m4["sample"] == "pooled") & (m4.target == "drsp_psych") & (m4.model == "ridge") & (m4.fset == "D")].iloc[0]
assert f"{r.R2:+.3f} [{r.R2_lo:+.3f}, {r.R2_hi:+.3f}]" == tab.loc["DRSP psychological symptoms", "Test R2 [95% CI] (leave-one-woman-out)"]; print("4. displayed test R2 and CI equal results/phase4_metrics.csv:", tab.loc["DRSP psychological symptoms", "Test R2 [95% CI] (leave-one-woman-out)"])
# no user input is stored: nothing in the project was created or modified by running the app
after = snapshot(); changed = [p for p in after if p not in before or after[p] != before[p]]; assert not changed, changed; print("5. running the app (including all inputs above) created or modified no file in the project")
txt = " ".join(x.value for x in at.markdown) + " ".join(x.value for x in at.info) + " ".join(x.value for x in at.caption); print("6. text mentions cross-sample coverage failure:", "coverage was only" in txt, "| privacy statement:", "writes no files" in txt)
print(f"all tests passed in {time.time() - t0:.1f} s")

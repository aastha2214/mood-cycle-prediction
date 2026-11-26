"""Headless tests of the demo (no browser, no network): the functions of app_core against an independent recomputation, the result tables against the saved CSVs, and the Streamlit page
through streamlit.testing. Run from the project root:  ./.venv/bin/python -I app/test_app.py"""
import sys, time; from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "app"))
import numpy as np, pandas as pd
from sklearn.linear_model import BayesianRidge
from streamlit.testing.v1 import AppTest
from src.data import load_analysis, FEATURE_SETS, TARGETS, SEED
from src.cvlib import predict_one
import app_core as C
def snapshot(): return {str(p): p.stat().st_mtime_ns for p in ROOT.rglob("*") if p.is_file() and ".venv" not in p.parts and "__pycache__" not in p.parts and ".git" not in p.parts}
before = snapshot(); t0 = time.time(); R, missing = C.load_results(); assert not missing, missing; art = R["art"]; I = art["inputs"]
# 1. predict: ridge and Bayesian ridge against the validation pipeline and scikit-learn, trained on all women
d = load_analysis(); X, g = d[FEATURE_SETS["D"]].values, d.VPN.values
def reference(t, v):
    y = d[TARGETS[t]].values; x = np.array([[v["phase"] == "peri-ovulatory", v["phase"] == "luteal", np.log1p(v["e2"]), np.log1p(v["pg"]), v["pms"], v["bdi"], v["bai"]]], float)
    ridge = predict_one("ridge", X, y, g, x, SEED)[0][0]; mu, sd = X.mean(0), X.std(0); sk = BayesianRidge(max_iter=100000, tol=1e-13, alpha_1=0, alpha_2=0, lambda_1=0, lambda_2=0).fit((X - mu) / sd, y)
    m, s = sk.predict((x - mu) / sd, return_std=True); return ridge, m[0], s[0]
cases = [dict(phase="menses", e2=0.5, pg=30.0, pms=12, bdi=9, bai=8), dict(phase="luteal", e2=1.2, pg=150.0, pms=25, bdi=15, bai=20), dict(phase="peri-ovulatory", e2=0.3, pg=20.0, pms=2, bdi=1, bai=0)]; worst = 0.0
for v in cases:
    for t in art["targets"]:
        got = C.predict(art, t, v); want = reference(t, v); worst = max(worst, max(abs(a - b) for a, b in zip(got, want))); assert abs(got[0] - want[0]) < 6e-3 and np.allclose(got[1:], want[1:], atol=1e-5), (t, v, got, want)
        pt, lo, hi = C.point_and_interval(art, t, v, "Bayesian ridge"); assert abs(lo - (want[1] - art["z_level"] * want[2])) < 1e-5 and abs(hi - (want[1] + art["z_level"] * want[2])) < 1e-5 and lo < pt < hi
print(f"1. predict / interval: {len(cases)} input sets x 3 targets agree with the validation pipeline and scikit-learn (largest difference {worst:.1e})")
# 2. contributions add up to the prediction
for v in cases:
    for t in art["targets"]:
        for model in C.MODELS:
            df = C.contributions(art, t, v, model); assert list(df.feature) == C.FEATURE_LABELS and len(df) == 7; assert abs(art["targets"][t]["ybar"] + df.contribution.sum() - C.point_and_interval(art, t, v, model)[0]) < 1e-9
print("2. contributions: training mean + sum of contributions = point prediction for both models, all targets")
# 3. sweep: grid covers the observed range, endpoints equal direct predictions, intervals ordered; the summary is bounded by the interval width
v = cases[1]
for which, key in [("e2", "estradiol"), ("pg", "progesterone")]:
    for t in art["targets"]:
        s = C.sweep(art, t, v, which, "Ridge"); assert len(s) == 80 and s.x.iloc[0] == I[key]["min"] and abs(s.x.iloc[-1] - I[key]["max"]) < 1e-9 and (s.lo < s.pred).all() and (s.pred < s.hi).all()
        for i in [0, -1]: assert abs(s.pred.iloc[i] - C.point_and_interval(art, t, {**v, which: float(s.x.iloc[i])}, "Ridge")[0]) < 1e-12
sd_share, w_share = C.sweep_summary(art, v, "Ridge"); assert 0 <= w_share < sd_share < 1; print(f"3. sweep: 80-point grids over the observed ranges; prediction moves at most {100 * sd_share:.0f}% of the outcome SD, {100 * w_share:.0f}% of the interval width")
# 4. scenario comparison
same = C.compare(art, cases[0], cases[0], "Ridge"); assert (same["diff"].abs() < 1e-12).all(); cmp_ = C.compare(art, cases[0], cases[1], "Ridge")
assert np.allclose(cmp_["diff"], cmp_.b - cmp_.a) and np.allclose(cmp_.a, [C.point_and_interval(art, t, cases[0], "Ridge")[0] for t in art["targets"]]); print("4. scenario comparison: identical scenarios give 0, difference = B - A")
# 5. presets and result tables against the saved CSVs
P = C.presets(R); assert set(P) == {"typical menses", "typical luteal", "high questionnaire scores"} and P["high questionnaire scores"]["bdi"] == int(I["BDI_RW"]["max"])
for p in P.values():
    for t in art["targets"]: C.predict(art, t, p)
m4 = pd.read_csv(ROOT / "results/phase4_metrics.csv"); tab, nr, nw = C.r2_table(R, "pooled"); r = m4[(m4["sample"] == "pooled") & (m4.target == "drsp_psych") & (m4.model == "ridge") & (m4.fset == "D")].iloc[0]
assert tab.loc["Set D", "DRSP psychological symptoms"] == f"{r.R2_oos:+.3f} [{r.R2_oos_lo:+.3f}, {r.R2_oos_hi:+.3f}]" and (nr, nw) == (271, 160)
hc = C.hormone_counts(R); assert hc == dict(total=27, better=0, worse=15), hc; ct = C.cross_table(R); assert ct.shape == (6, 3); cv = C.coverage_table(R, art); assert cv.shape == (3, 2)
print(f"5. presets valid; R2 table equals results/phase4_metrics.csv ({nr} sessions, {nw} women); hormone comparisons {hc}; cross-sample table {ct.shape}; coverage table {cv.shape}")
# 6. the page: no exception, five tabs, presets fill the inputs, every control works, nothing written
at = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=90).run(); assert not at.exception, [e.value for e in at.exception]; assert [t.label for t in at.tabs] == ["Predict", "What drives the prediction", "Hormone sweep", "Compare scenarios", "Study results"]
assert len(at.metric) == 3 and any("not a medical or diagnostic tool" in i.value for i in at.info); assert at.session_state["phase"] == "menses"
btn = {b.label: b for b in at.button}; btn["Typical luteal"].click().run(); assert not at.exception and at.session_state["phase"] == "luteal" and abs(at.session_state["pg"] - P["typical luteal"]["pg"]) < 1e-9
btn = {b.label: b for b in at.button}; btn["High questionnaire scores"].click().run(); assert not at.exception and at.session_state["bdi"] == int(I["BDI_RW"]["max"])
at.sidebar.radio[0].set_value("Bayesian ridge").run(); assert not at.exception; at.selectbox[1].set_value("stai").run(); assert not at.exception
for combo in [dict(phase="menses", e2=I["estradiol"]["min"], pg=I["progesterone"]["min"], pms=int(I["PMS"]["min"]), bdi=int(I["BDI_RW"]["min"]), bai=int(I["BAI_RW"]["min"])), dict(phase="luteal", e2=I["estradiol"]["max"], pg=I["progesterone"]["max"], pms=int(I["PMS"]["max"]), bdi=int(I["BDI_RW"]["max"]), bai=int(I["BAI_RW"]["max"]))]:
    for k, val in combo.items(): at.session_state[k] = val
    at.run(); assert not at.exception, [e.value for e in at.exception]
at.session_state["a_phase"] = "luteal"; at.run(); assert not at.exception; [r for r in at.radio if r.label == "Sample"][0].set_value("Longitudinal women").run(); assert not at.exception
print("6. page: five tabs, honesty note, presets fill the inputs, model/target/sample switches and the minimum/maximum of every input run without exception")
after = snapshot(); changed = [p for p in after if p not in before or after[p] != before[p]]; assert not changed, changed; print("7. running the app created or modified no file in the project")
print(f"all tests passed in {time.time() - t0:.1f} s")

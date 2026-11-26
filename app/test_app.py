"""Headless tests of the demo (no browser, no network): the functions of the app against an independent recomputation, the result tables and charts against the saved CSVs,
and the Streamlit page through streamlit.testing. Run from the project root:  ./.venv/bin/python -I app/test_app.py"""
import ast, io, re, sys, time; from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; APP = ROOT / "app"; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(APP))
import numpy as np, pandas as pd
import streamlit as st
from matplotlib.container import BarContainer, ErrorbarContainer
from sklearn.linear_model import BayesianRidge
from streamlit.testing.v1 import AppTest
from src.data import load_analysis, FEATURE_SETS, TARGETS, SEED
from src.cvlib import predict_one
from src.mathcore import optim
import app_core as C
import app_plots as P
import method_lab as M

def snapshot(): return {str(p): p.stat().st_mtime_ns for p in ROOT.rglob("*") if p.is_file() and ".venv" not in p.parts and "__pycache__" not in p.parts and ".git" not in p.parts}
before = snapshot(); t0 = time.time(); R, missing = C.load_results(); assert not missing, missing; art = R["art"]; I = art["inputs"]; PRE = C.presets(R)
TABS = ["Predict", "What drives the prediction", "Hormone sweep", "Compare scenarios", "Study results", "Method lab"]
KEYS = ["phase", "e2", "pg", "pms", "bdi", "bai"]
csv = lambda name: pd.read_csv(ROOT / "results" / name)

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

# 2. contributions, sweep, scenario comparison, presets
for v in cases:
    for t in art["targets"]:
        for model in C.MODELS:
            df = C.contributions(art, t, v, model); assert list(df.feature) == C.FEATURE_LABELS and len(df) == 7; assert abs(art["targets"][t]["ybar"] + df.contribution.sum() - C.point_and_interval(art, t, v, model)[0]) < 1e-9
for which, key in [("e2", "estradiol"), ("pg", "progesterone")]:
    for t in art["targets"]:
        s = C.sweep(art, t, cases[1], which, "Ridge"); assert len(s) == 80 and s.x.iloc[0] == I[key]["min"] and abs(s.x.iloc[-1] - I[key]["max"]) < 1e-9 and (s.lo < s.pred).all() and (s.pred < s.hi).all()
sd_share, w_share = C.sweep_summary(art, cases[1], "Ridge"); assert 0 <= w_share < sd_share < 1
same = C.compare(art, cases[0], cases[0], "Ridge"); assert (same["diff"].abs() < 1e-12).all(); cmp_ = C.compare(art, cases[0], cases[1], "Ridge"); assert np.allclose(cmp_["diff"], cmp_.b - cmp_.a)
assert set(PRE) == {"typical menses", "typical luteal", "high questionnaire scores"} and PRE["high questionnaire scores"]["bdi"] == int(I["BDI_RW"]["max"])
print(f"2. contributions add up; sweeps cover the observed ranges (at most {100 * w_share:.0f}% of the interval width); scenario differences and presets are consistent")

# 3. result tables against the saved CSVs
m4 = csv("phase4_metrics.csv"); tab, nr, nw = C.r2_table(R, "pooled"); r = m4[(m4["sample"] == "pooled") & (m4.target == "drsp_psych") & (m4.model == "ridge") & (m4.fset == "D")].iloc[0]
assert tab.loc["Set D", "DRSP psychological symptoms"] == f"{r.R2_oos:+.3f} [{r.R2_oos_lo:+.3f}, {r.R2_oos_hi:+.3f}]" and (nr, nw) == (271, 160)
hc = C.hormone_counts(R); assert hc == dict(total=27, better=0, worse=15), hc; assert C.cross_table(R).shape == (6, 3) and C.coverage_table(R, art).shape == (3, 2)
print(f"3. result tables equal the saved files ({nr} sessions, {nw} women); hormone comparisons {hc}")

# 4. key findings: every number equals an independent reading of the CSVs, and follows the data when the data change
dl = csv("phase4_deltas.csv"); dl = dl[(dl.comparison == "C vs A") & (dl.metric == "RMSE")]; cov = csv("extE4_coverage.csv"); cov = cov[(cov.fset == "D") & (cov.method == "bayes_predictive") & np.isclose(cov.level, art["level"])]
within, across = cov[cov["sample"] == "pooled"], cov[cov["sample"] == "cross2long"]; findings = C.key_findings(R); assert len(findings) == 3
nums = [re.findall(r"[+-]?\d+(?:\.\d+)?", line) for line in findings]
assert nums[0] == [str(int(((dl.delta < 0) & (dl.hi < 0)).sum())), str(len(dl)), str(int(((dl.delta > 0) & (dl.lo > 0)).sum()))], nums[0]
assert nums[1] == [f"{r.R2_oos:+.3f}", f"{r.R2_oos_lo:+.3f}", f"{r.R2_oos_hi:+.3f}", str(int(r.n_women))], nums[1]
assert nums[2] == [f"{100 * art['level']:.0f}", f"{within.coverage.min():.3f}", f"{within.coverage.max():.3f}", str(int(within.n_women.iloc[0])), f"{across.coverage.min():.3f}", f"{across.coverage.max():.3f}", str(int(across.n_women.iloc[0]))], nums[2]
assert bool(within.covers_nominal.all()) and not across.covers_nominal.any() and (across.coverage < across.level).all() and "undercover" in findings[2]
changed = {**R, "m4": R["m4"].assign(R2_oos=0.111, R2_oos_lo=-0.222, R2_oos_hi=0.333, n_women=7), "d4": R["d4"][R["d4"]["sample"] != "cross2long"].assign(delta=-1.0, lo=-2.0, hi=-0.5), "cov": R["cov"].assign(coverage=0.95, covers_nominal=True, n_women=7)}
other = C.key_findings(changed); total = str(int(((changed["d4"].comparison == "C vs A") & (changed["d4"].metric == "RMSE")).sum())); assert total != nums[0][1]
assert [re.findall(r"[+-]?\d+(?:\.\d+)?", line) for line in other] == [[total, total, "0"], ["+0.111", "-0.222", "+0.333", "7"], ["90", "0.950", "0.950", "7", "0.950", "0.950", "7"]] and "undercover" not in other[2], other
print("4. key findings: all numbers equal the saved CSVs and change when the data change; the wording about coverage holds for these results")

# 5. downloads and settings in the address (functions)
for model in C.MODELS:
    table = C.download_table(art, {"A": cases[0], "B": cases[1]}, model); back = pd.read_csv(io.StringIO(table.to_csv(index=False)))
    assert list(back.scenario) == ["A", "B"] and (back.model == model).all() and list(back.cycle_phase) == ["menses", "luteal"] and list(back.BDI) == [9, 15] and np.allclose(back.progesterone_pg_ml, [30.0, 150.0])
    for t in art["targets"]:
        want = np.array([C.point_and_interval(art, t, v, model) for v in cases[:2]]); assert np.allclose(back[[f"{t}_prediction", f"{t}_interval90_low", f"{t}_interval90_high"]].values, want)
good = {"phase": "luteal", "estradiol": "1.25", "progesterone": "200", "psst": "20", "bdi": "30", "bai": "5", "model": "bayesian"}
assert C.parse_query(art, good) == (dict(phase="luteal", e2=1.25, pg=200.0, pms=20, bdi=30, bai=5), "Bayesian ridge")
edges = {name: str(int(I[src]["max"]) if kind is int else I[src]["max"]) for key, (src, kind) in C.QUERY_RANGES.items() for name in [C.QUERY_NAMES[key]]}; assert len(C.parse_query(art, edges)[0]) == 5
edges = {name: str(int(I[src]["min"]) if kind is int else I[src]["min"]) for key, (src, kind) in C.QUERY_RANGES.items() for name in [C.QUERY_NAMES[key]]}; assert len(C.parse_query(art, edges)[0]) == 5
bad_values = {"phase": ["spring", "Luteal", ""], "estradiol": ["abc", "nan", "inf", "-0.01", "3.36", "", "1,2"], "progesterone": ["-1", "652.04", "1e9", "-inf"], "psst": ["12.5", "-1", "23", "1e1", "x"], "bdi": ["40", "-3", ""], "bai": ["41", "4.0"], "model": ["Ridge", "lasso", ""]}
for name, values in bad_values.items():
    for value in values: assert C.parse_query(art, {name: value}) == ({}, None), (name, value)
assert C.parse_query(art, {"unknown": "1"}) == ({}, None) and C.parse_query(art, {}) == ({}, None)
for v in cases[:1] + list(PRE.values()):
    for model in C.MODELS: got, m = C.parse_query(art, C.query_from(v, model)); assert m == model and got["phase"] == v["phase"] and np.allclose([got[k] for k in KEYS[1:]], [v[k] for k in KEYS[1:]])
print(f"5. download table equals the predictions for both models; address settings: valid values and limits accepted, {sum(map(len, bad_values.values()))} invalid values ignored")

# 6. method lab (pure NumPy): optimisers against src/mathcore, theory, stability, shrinkage, condition numbers
lam, w0 = M.problem(100); n = len(lam); Xc = np.diag(np.sqrt(n * lam / 2)); yc = -Xc @ w0; eta_gd, eta_hb, beta = M.step_sizes(100); res = M.compare_methods(100)
assert abs(eta_hb - 4 / (1 + 0.1) ** 2) < 1e-12 and abs(beta - (9 / 11) ** 2) < 1e-12 and eta_gd == 1.0
for gap, losses in [(res["gd"], optim.gd(Xc, yc, 0.0, eta_gd, res["gd_iters"], w0=np.zeros(n))[1]), (res["hb"], optim.momentum(Xc, yc, 0.0, eta_hb, beta, res["hb_iters"])[1])]:
    assert np.allclose(gap, losses / losses[0], rtol=1e-9, atol=1e-15)
assert (np.diff(res["gd"]) <= 0).all() and res["gd"][-1] <= M.TOL < res["gd"][-2] and res["hb"][-1] <= M.TOL < res["hb"][-2]
counts = {k: M.compare_methods(k) for k in [10, 100, 1000]}
for k, c in counts.items(): assert c["gd_iters"] <= np.ceil(np.log(M.TOL) / (2 * np.log(1 - 1 / k))) and c["hb_iters"] < c["gd_iters"]
assert 7 < counts[1000]["gd_iters"] / counts[100]["gd_iters"] < 13 and 2.5 < counts[1000]["hb_iters"] / counts[100]["hb_iters"] < 4.5, {k: (c["gd_iters"], c["hb_iters"]) for k, c in counts.items()}
hard = M.compare_methods(100_000); assert hard["gd_iters"] is None and len(hard["gd"]) == M.CAP + 1 and hard["hb_iters"] is not None; easy = M.compare_methods(2); assert easy["gd_iters"] and easy["hb_iters"]
assert min(M.KAPPAS) == 2 and max(M.KAPPAS) == 100_000
stable, _ = M.descend(lam, w0, 1.99, cap=400); unstable, _ = M.descend(lam, w0, 2.01, cap=400); assert stable[-1] < 1 < unstable[-1]
s = [30.0, 10.0, 3.0]; lams = np.logspace(-4, 3, 50); F = M.shrinkage(s, lams); assert F.shape == (3, 50) and (np.diff(F, axis=1) < 0).all() and (F[0] > F[1]).all() and (F[1] > F[2]).all()
assert all(abs(M.shrinkage([si], [si ** 2 / M.N_ILLUSTRATIVE])[0, 0] - 0.5) < 1e-12 for si in s)
ct, rows = C.condition_table(R); k0 = csv("phase3c_condition.csv"); k0 = k0[(k0["sample"] == "pooled") & (k0.lam == 0)].set_index("variant").kappa
assert rows == 271 and [float(x.replace(",", "")) for x in ct["Condition number"]] == [round(k0["raw (unscaled, raw hormones)"]), round(k0["log1p hormones, unscaled"]), round(k0["log1p + standardised"], 1)] and C.condition_table({**R, "cond": None}) is None
print(f"6. method lab: both optimisers equal src/mathcore; iterations (kappa: GD, momentum) " + ", ".join(f"{k}: {c['gd_iters']}, {c['hb_iters']}" for k, c in counts.items()) + f"; GD not reached in {M.CAP} at kappa 100000; condition table equals the CSV")

# ------------------------------------------------------------------ the page
figures, downloads = {}, {}
_pyplot, _download = st.pyplot, st.download_button
def spy_pyplot(fig, *a, **k):
    info = {"font": max([t.get_fontsize() for ax in fig.axes for t in ax.get_xticklabels()] + [0]), "face": [tuple(fig.get_facecolor())] + [tuple(ax.get_facecolor()) for ax in fig.axes], "text": {t.get_color() for ax in fig.axes for t in ax.get_xticklabels()}}
    if fig.get_label() == "r2_bars":
        ax = fig.axes[0]; info["bars"] = {c.get_label(): [p.get_height() for p in c.patches] for c in ax.containers if isinstance(c, BarContainer)}; info["labels"] = [t.get_text() for t in ax.texts]
        info["ticks"] = [t.get_text().replace("\n", " ") for t in ax.get_xticklabels()]; info["title"] = ax.get_title(loc="left")
    if fig.get_label() == "calibration":
        info["points"] = {(i, c.get_label()): c.lines[0].get_ydata().tolist() for i, ax in enumerate(fig.axes) for c in ax.containers if isinstance(c, ErrorbarContainer)}
    figures[fig.get_label()] = info; return _pyplot(fig, *a, **k)
def spy_download(label, data=None, *a, **k):
    downloads[k.get("file_name")] = data; return _download(label, data, *a, **k)
st.pyplot, st.download_button = spy_pyplot, spy_download

def walk(node):
    yield node
    for child in getattr(node, "children", {}).values(): yield from walk(child)

BAD = re.compile(r"\b(NULL|null|None|nan|NaN)\b")
def assert_clean(at):
    """No exception, no JSON viewer and no rendered None / null anywhere on the page (all tabs are part of one run)."""
    assert not at.exception, [e.value for e in at.exception]; assert len(at.get("json")) == 0, [j.proto.body for j in at.get("json")]
    assert len(at.tabs) == len(TABS)
    for tab in list(at.tabs) + [at.sidebar]:
        for node in walk(tab):
            assert node.type != "json", node
            for attr in ("value", "label", "body", "delta"):
                value = getattr(node, attr, None) if node.type in ("markdown", "caption", "info", "warning", "subheader", "metric", "text", "table", "arrow_table") else None
                if isinstance(value, str): assert not BAD.search(value), (node.type, attr, value)
                if isinstance(value, pd.DataFrame): text = value.astype(str); assert not value.isna().any().any() and not text.apply(lambda col: col.str.fullmatch("None|null|NULL|nan")).any().any(), value
    return at

def page(**query):
    at = AppTest.from_file(str(APP / "app.py"), default_timeout=120)
    for k, v in query.items(): at.query_params[k] = v
    return assert_clean(at.run())
button = lambda at, label: [b for b in at.button if b.label == label][0]
one = lambda value: value[0] if isinstance(value, list) else value
state = lambda at, prefix="": {k: at.session_state[prefix + k] for k in KEYS}
texts = lambda at: " ".join([m.value for m in at.markdown] + [c.value for c in at.caption] + [i.value for i in at.info] + [w.value for w in at.warning])

# 7. no bare expressions that Streamlit would print, and a clean page: no JSON viewer, no None / null
for f in ["app.py", "app_core.py", "app_plots.py", "method_lab.py"]:
    tree = ast.parse((APP / f).read_text()); loose = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Expr) and not isinstance(n.value, (ast.Call, ast.Constant, ast.Yield))]; assert not loose, (f, loose)
st.cache_data.clear(); loads = []; _load = C.load_results
C.load_results = lambda: (loads.append(1), _load())[1]
at = page(); assert [t.label for t in at.tabs] == TABS
for t in at.tabs: assert len(list(walk(t))) > 2, t.label
print("7. page: six tabs with content, no JSON viewer, no None or null, no bare expressions in the source")

# 8. fixed texts and layout
assert any("not a medical or diagnostic tool" in i.value for i in at.info) and len(at.metric) == 3 and state(at) == PRE["typical menses"]
everything = texts(at)
assert "small compared with the interval width" in everything and "nearly flat" not in everything and f"in {hc['better']} of {hc['total']} comparisons and made it worse in {hc['worse']}" in everything and "11% of the interval width" in everything
assert "Bars show each input's effect relative to the average woman in the training data, so the reference phase can still show a non-zero bar." in everything
assert C.DATA_LINE in [m.value for m in at.markdown] and any(C.REPO_URL in m.value for m in at.sidebar.markdown)
for line in findings: assert line in everything
for term, sentence in C.GLOSSARY: assert f"**{term}.** {sentence}" in everything
assert {t for t, _ in C.GLOSSARY} == {"DRSP", "PSST", "BDI", "BAI", "STAI", "R² against the training mean", "Prediction interval", "Leave-one-woman-out validation", "Cross-sample validation", "Bayesian ridge", "Condition number"}
scen = [t.value for t in at.table if "Scenario A" in t.value.columns][0]; assert scen.index.name is None and list(scen.index) == list(C.TN.values())
pair = [col for col in walk(at.tabs[3]) if col.type == "column"]; assert [[s.value for s in col.subheader] for col in pair] == [["Scenario A"], ["Scenario B"]] and all(len(col.selectbox) == 2 and len(col.slider) == 5 for col in pair)   # each heading sits in its own input column
face = {c for f in figures.values() for c in f["face"]}; ink = {c for f in figures.values() for c in f["text"]}; assert face == {(1.0, 1.0, 1.0, 1.0)} and ink == {P.INK}, (face, ink)
assert set(figures) == {"intervals", "contributions", "sweep", "compare", "r2_bars", "calibration", "optimisers", "shrinkage"}, set(figures)
print(f"8. required texts, key findings, glossary, sidebar link and data line present; compare table has no stray header; all {len(figures)} figures use the same explicit colours")

# 9. bar chart equals the table above it, for both samples; calibration chart equals the CSV
for label, sample in [("Pooled", "pooled"), ("Longitudinal women", "long")]:
    [x for x in at.radio if x.label == "Sample"][0].set_value(label); assert_clean(at.run())
    shown = [t.value for t in at.table if list(t.value.index) == ["Set A", "Set C", "Set D"]][0]; fig = figures["r2_bars"]; assert fig["ticks"] == list(shown.columns) and set(fig["bars"]) == {"set A", "set C", "set D"}
    wanted_labels = []
    for fs in ["A", "C", "D"]:
        for height, target, cell in zip(fig["bars"][f"set {fs}"], C.TN, shown.loc[f"Set {fs}"]):
            assert f"{height:+.3f}" == cell.split()[0], (sample, fs, target, height, cell)
            row = m4[(m4["sample"] == sample) & (m4.target == target) & (m4.model == "ridge") & (m4.fset == fs)].iloc[0]; assert abs(height - row.R2_oos) < 1e-12; wanted_labels.append(cell.split()[0])
    assert fig["labels"] == wanted_labels and str(int(row.n_women)) in fig["title"], (fig["labels"], wanted_labels)
    print(f"9. {label}: 9 bar heights and bar labels equal the table and results/phase4_metrics.csv (DRSP, set D: {fig['bars']['set D'][1]:+.3f})")
cal = csv("extE4_coverage.csv"); cal = cal[cal.method == "bayes_predictive"]; checked = 0
for i, sample in enumerate(["pooled", "cross2long"]):
    for j, target in enumerate(C.TN):
        for fs in ["A", "C", "D"]:
            want = cal[(cal["sample"] == sample) & (cal.target == target) & (cal.fset == fs)].sort_values("level").coverage.tolist(); assert np.allclose(figures["calibration"]["points"][(i * 3 + j, f"set {fs}")], want); checked += len(want)
print(f"9. calibration chart: {checked} plotted coverages equal results/extE4_coverage.csv")

# 10. presets, controls, extremes of every input, method lab controls; the data are loaded only once
button(at, "Typical luteal").click(); assert_clean(at.run()); assert state(at) == PRE["typical luteal"]
button(at, "High questionnaire scores").click(); assert_clean(at.run()); assert at.session_state["bdi"] == int(I["BDI_RW"]["max"])
at.sidebar.radio[0].set_value("Bayesian ridge"); assert_clean(at.run()); [x for x in at.selectbox if x.label == "Target"][0].set_value("stai"); assert_clean(at.run())
for end in ["min", "max"]:
    for k, src, kind in [("e2", "estradiol", float), ("pg", "progesterone", float), ("pms", "PMS", int), ("bdi", "BDI_RW", int), ("bai", "BAI_RW", int)]: at.session_state[k] = kind(I[src][end])
    assert_clean(at.run())
[x for x in at.selectbox if x.key == "a_preset"][0].set_value("typical luteal"); assert_clean(at.run()); assert state(at, "a_") == PRE["typical luteal"]
for kappa in [2, 100_000]:
    at.select_slider[0].set_value(kappa); assert_clean(at.run()); table = [t.value for t in at.table if "Theory scale" in t.value.columns][0]
    assert table.loc["Gradient descent", "Theory scale"] == f"about κ = {kappa:,}" and f"{kappa ** 0.5:,.1f}" in table.loc["Heavy-ball momentum", "Theory scale"]
    assert (table.loc["Gradient descent", "Iterations to 1e-6"] == f"not reached in {M.CAP:,}") == (kappa == 100_000)
for k, value in [("s1", 100.0), ("s2", 0.5), ("s3", 55.5)]: [x for x in at.slider if x.key == k][0].set_value(value)
assert_clean(at.run()); assert any("Illustrative, not the study data" in w.value for w in at.warning) and "Gradient descent is stable only for step size below 2/Lmax." in texts(at) and "shrunk first" in texts(at)
assert [t.value for t in at.table if "Condition number" in t.value.columns][0].equals(ct) and len(loads) == 1, len(loads); C.load_results = _load
print("10. presets, model, target, extreme inputs, scenario preset and method lab controls work; the model file and CSVs were loaded once (cached)")

# 11. downloads from the page, reset button
at = page(); button(at, "Typical luteal").click(); assert_clean(at.run()); at.session_state["bdi"] = 21; assert_clean(at.run())
assert len(at.get("download_button")) == 2 and set(downloads) == {"predict_predictions.csv", "compare_predictions.csv"}
got = pd.read_csv(io.BytesIO(downloads["predict_predictions.csv"])); assert len(got) == 1 and got.BDI[0] == 21 and got.cycle_phase[0] == "luteal" and got.model[0] == "Ridge"
for metric, t in zip(at.metric, art["targets"]):
    assert metric.value == f"{got[f'{t}_prediction'][0]:.2f}" and any(c.value == f"90% prediction interval: {got[f'{t}_interval90_low'][0]:.2f} to {got[f'{t}_interval90_high'][0]:.2f}" for c in at.caption)
both = pd.read_csv(io.BytesIO(downloads["compare_predictions.csv"])); shown = [t.value for t in at.table if "Scenario A" in t.value.columns][0]
for t in art["targets"]:
    for i, col in enumerate(["Scenario A", "Scenario B"]): assert shown.loc[C.TN[t], col] == f"{both[f'{t}_prediction'][i]:.2f} ({both[f'{t}_interval90_low'][i]:.2f} to {both[f'{t}_interval90_high'][i]:.2f})"
at.sidebar.radio[0].set_value("Bayesian ridge"); assert_clean(at.run()); assert pd.read_csv(io.BytesIO(downloads["predict_predictions.csv"])).model[0] == "Bayesian ridge"
assert state(at) != PRE["typical menses"]; button(at, "Reset inputs").click(); assert_clean(at.run()); assert state(at) == PRE["typical menses"] and at.session_state["model"] == "Bayesian ridge"
assert one(at.query_params["phase"]) == "menses" and one(at.query_params["bdi"]) == str(PRE["typical menses"]["bdi"])
print("11. downloaded files equal the values on the page (both tabs, both models); reset returns the start inputs and keeps the model")

# 12. settings in the address, through the page
at = page(**good); assert state(at) == dict(phase="luteal", e2=1.25, pg=200.0, pms=20, bdi=30, bai=5) and at.session_state["model"] == "Bayesian ridge"
at.session_state["pms"] = 3; assert_clean(at.run()); assert one(at.query_params["psst"]) == "3" and one(at.query_params["phase"]) == "luteal"
bad = {"phase": "spring", "estradiol": "99", "progesterone": "abc", "psst": "-1", "bdi": "12.5", "bai": "nan", "model": "lasso"}; at = page(**bad)
assert state(at) == PRE["typical menses"] and at.session_state["model"] == "Ridge" and C.parse_query(art, {k: one(at.query_params[k]) for k in bad}) == (PRE["typical menses"], "Ridge")
at = page(phase="peri-ovulatory", estradiol="oops", bdi="7"); assert state(at) == {**PRE["typical menses"], "phase": "peri-ovulatory", "bdi": 7}
print("12. address settings: valid values restore the inputs and model, invalid ones fall back to the start values and are overwritten, mixed addresses keep the valid part")

# 13. presentation mode
at = page(); assert len(at.toggle) == 1 and at.toggle[0].label == "Presentation mode" and at.toggle[0].value is False
css = lambda a: " ".join(h.proto.body for h in a.get("html")); normal = [c.value for c in at.caption]; small = {k: f["font"] for k, f in figures.items()}; assert "presentation" not in css(at) and any("Typical values" in c for c in normal)
at.toggle[0].set_value(True); assert_clean(at.run()); large = [c.value for c in at.caption]
assert "presentation" in css(at) and "stMetricValue" in css(at) and "stCaptionContainer" in css(at) and len(large) < len(normal) and set(large) < set(normal)
for hidden in ["Typical values", "Slider ranges", "Each bar is", "Points: predictions", "Contribution =", "Quadratic loss", "Points on the dashed line"]: assert any(hidden in c for c in normal) and not any(hidden in c for c in large), hidden
for kept in ["prediction interval", "Bars show each input's effect", "Gradient descent is stable only", "shrunk first", "These are study results"]: assert any(kept in c for c in large), kept
assert all(figures[k]["font"] > small[k] for k in small), {k: (small[k], figures[k]["font"]) for k in small}; assert any("not a medical or diagnostic tool" in i.value for i in at.info)
at.toggle[0].set_value(False); assert_clean(at.run()); assert [c.value for c in at.caption] == normal and "presentation" not in css(at)
print(f"13. presentation mode: off by default; on: larger cards, captions and chart text, {len(normal) - len(large)} secondary captions hidden, essential ones kept; off again restores the page")

st.pyplot, st.download_button = _pyplot, _download
# 14. nothing written
after = snapshot(); modified = [p for p in after if p not in before or after[p] != before[p]]; assert not modified, modified; print("14. running the app created or modified no file in the project")
print(f"all tests passed in {time.time() - t0:.1f} s")

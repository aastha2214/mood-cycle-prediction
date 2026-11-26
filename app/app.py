"""Course-project demo (Streamlit). Start from the project root:  streamlit run app/app.py
Reads app/model_artifacts.json (fitted final model, built by app/build_model.py) and the saved result files in results/. Nothing the user enters is stored by the app;
the Predict-tab settings are written into the page address so that they can be shared."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd, streamlit as st
import app_core as C
import app_plots as P
import method_lab as M

st.set_page_config(page_title="Mood prediction demo", layout="wide")

@st.cache_data(show_spinner=False)
def load_results():
    """The model file and the result CSVs, read once and cached."""
    return C.load_results()

@st.cache_data(show_spinner=False, max_entries=20)
def run_optimisers(kappa):
    return M.compare_methods(kappa)

R, missing = load_results()
st.title("Mood prediction from cycle phase, hormones and questionnaires")
if missing:
    load_results.clear(); st.warning("Saved result files are missing: " + ", ".join(missing) + ". Run `bash run_all.sh` and `./.venv/bin/python -I app/build_model.py` first."); st.stop()
art, TN = R["art"], C.TN; I = art["inputs"]; lvl = int(round(100 * art["level"])); PRESETS = C.presets(R)

def apply_preset(prefix, name):
    if name:
        for k, v in PRESETS[name].items(): st.session_state[prefix + k] = v

# first run of a session: default inputs, then every valid value found in the page address
if "phase" not in st.session_state:
    apply_preset("", C.DEFAULT_PRESET); from_url, url_model = C.parse_query(art, st.query_params)
    for k, value in from_url.items(): st.session_state[k] = value
    if url_model: st.session_state["model"] = url_model
if "model" not in st.session_state: st.session_state["model"] = C.MODELS[0]
for prefix, name in [("a_", "typical menses"), ("b_", "typical luteal")]:
    if prefix + "phase" not in st.session_state: apply_preset(prefix, name)

def inputs(prefix, container):
    """The six input widgets; values live in st.session_state under prefix + key."""
    with container:
        phase = st.selectbox("Cycle phase", C.PHASES, key=prefix + "phase")
        e2 = st.slider("Estradiol (saliva, pg/ml)", I["estradiol"]["min"], I["estradiol"]["max"], step=I["estradiol"]["max"] / 200, key=prefix + "e2")
        pg = st.slider("Progesterone (saliva, pg/ml)", I["progesterone"]["min"], I["progesterone"]["max"], step=I["progesterone"]["max"] / 200, key=prefix + "pg")
        pms = st.slider("PSST score", int(I["PMS"]["min"]), int(I["PMS"]["max"]), key=prefix + "pms")
        bdi = st.slider("BDI score", int(I["BDI_RW"]["min"]), int(I["BDI_RW"]["max"]), key=prefix + "bdi")
        bai = st.slider("BAI score", int(I["BAI_RW"]["min"]), int(I["BAI_RW"]["max"]), key=prefix + "bai")
    return dict(phase=phase, e2=e2, pg=pg, pms=pms, bdi=bdi, bai=bai)

# ------------------------------------------------------------------ sidebar
st.sidebar.markdown("**Predicts mood from cycle phase, saliva hormones and questionnaires, and shows what the study found.**")
model = st.sidebar.radio("Model", C.MODELS, key="model", help="Point prediction and contributions use this model; the interval is the Bayesian predictive interval.")
big = st.sidebar.toggle("Presentation mode", key="presentation", help="Larger cards, captions and chart text for projecting; hides secondary captions.")
st.sidebar.markdown("Feature set: **D** (phase, estradiol, progesterone, PSST, BDI, BAI), fitted on " + f"{art['n_women']} women ({art['n_rows']} sessions). Sets A and C are not stored in the model file.")
if not big: st.sidebar.caption("Slider ranges are the minimum and maximum observed in the data; start values are medians or typical values per phase.")
st.sidebar.markdown(f"[Source code on GitHub]({C.REPO_URL})")

BASE_CSS = '[data-testid="stMetricLabel"] p, [data-testid="stMetricDelta"], [data-testid="stMetricDelta"] div, [data-testid="stMetricDelta"] p {white-space: normal !important; overflow: visible !important; text-overflow: clip !important;}'
BIG_CSS = ('/* presentation */ [data-testid="stMetricLabel"] p {font-size: 1.35rem !important;} [data-testid="stMetricValue"], [data-testid="stMetricValue"] div {font-size: 3.2rem !important; line-height: 1.2 !important;} '
           '[data-testid="stMetricDelta"], [data-testid="stMetricDelta"] div, [data-testid="stMetricDelta"] p {font-size: 1.15rem !important; line-height: 1.35 !important;} [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {font-size: 1.2rem !important;}')
st.html(f"<style>{BASE_CSS} {BIG_CSS if big else ''}</style>")

def note(text):
    """Secondary caption: hidden in presentation mode."""
    if not big: st.caption(text)

def show(build, *args):
    with P.style(big): st.pyplot(build(*args, presentation=big))

def download(label_key, scenarios):
    st.download_button("Download predictions (CSV)", C.download_table(art, scenarios, model).to_csv(index=False).encode("utf-8"), file_name=f"{label_key}_predictions.csv", mime="text/csv", key="download_" + label_key, on_click="ignore")

tab_pred, tab_drv, tab_sweep, tab_cmp, tab_study, tab_lab = st.tabs(["Predict", "What drives the prediction", "Hormone sweep", "Compare scenarios", "Study results", "Method lab"])

# ------------------------------------------------------------------ Predict
with tab_pred:
    st.info(C.HONESTY)
    with st.container(border=True):
        st.markdown("**Key findings of the study**\n\n" + "\n".join(f"- {line}" for line in C.key_findings(R)))
    cols = st.columns(4)
    for c, name in zip(cols, PRESETS): c.button(name.capitalize(), on_click=apply_preset, args=("", name), width="stretch")
    cols[3].button("Reset inputs", on_click=apply_preset, args=("", C.DEFAULT_PRESET), width="stretch", help="Back to the start values (typical menses). The model choice is kept.")
    note("Typical values: medians of the longitudinal women in that phase (hormones) and of all women (questionnaires). High scores: the largest PSST, BDI and BAI observed in the data. "
         "The inputs and the model are kept in the page address, so the address can be shared.")
    left, right = st.columns([1, 2]); v = inputs("", left)
    with right:
        cards = st.columns(3)
        for c, t in zip(cards, art["targets"]):
            M_ = art["targets"][t]; pt, lo, hi = C.point_and_interval(art, t, v, model)
            c.metric(TN[t], f"{pt:.2f}", f"{pt - M_['ybar']:+.2f} vs training mean {M_['ybar']:.2f}", delta_color="off"); c.caption(f"{lvl}% prediction interval: {lo:.2f} to {hi:.2f}")
        show(P.interval_figure, art, v, model)
        download("predict", {"predict": v})
wanted = C.query_from(v, model)
if any(st.query_params.get(k) != value for k, value in wanted.items()): st.query_params.update(wanted)

# ------------------------------------------------------------------ drivers
with tab_drv:
    t = st.selectbox("Target", list(TN), format_func=TN.get); M_ = art["targets"][t]; df = C.contributions(art, t, v, model); pt = C.point_and_interval(art, t, v, model)[0]
    show(P.contributions_figure, df)
    st.caption("Bars show each input's effect relative to the average woman in the training data, so the reference phase can still show a non-zero bar.")
    note(f"Contribution = coefficient x standardized input. Training mean {M_['ybar']:.2f} + sum of contributions {df.contribution.sum():+.2f} = prediction {pt:.2f} (inputs from the Predict tab, model: {model}).")

# ------------------------------------------------------------------ sweep
with tab_sweep:
    show(P.sweep_figure, art, v, model)
    sd_share, w_share = C.sweep_summary(art, v, model); hc = C.hormone_counts(R)
    st.markdown(f"Everything else is held at the Predict-tab inputs (orange line); band = {lvl}% interval; dashed = training mean. With phase in the model the effect of the hormones is small compared with the interval width: across the whole observed hormone range the prediction moves by at most {100 * sd_share:.0f}% of the outcome's standard deviation, "
                f"at most {100 * w_share:.0f}% of the interval width. In the study, adding the hormones to phase improved out-of-sample RMSE in {hc['better']} of {hc['total']} comparisons and made it worse in {hc['worse']}.")

# ------------------------------------------------------------------ compare scenarios
with tab_cmp:
    ca, cb = st.columns(2)
    for col, prefix, title in [(ca, "a_", "Scenario A"), (cb, "b_", "Scenario B")]:
        col.subheader(title, text_alignment="center")   # inside the column, so that it stays with its inputs when columns stack on narrow screens
        col.selectbox("Start from", list(PRESETS), key=prefix + "preset", index=None, placeholder="choose a preset", on_change=lambda p=prefix: apply_preset(p, st.session_state[p + "preset"]))
    va = inputs("a_", ca); vb = inputs("b_", cb); df = C.compare(art, va, vb, model)
    st.table(pd.DataFrame({"Scenario A": [f"{r.a:.2f} ({r.a_lo:.2f} to {r.a_hi:.2f})" for r in df.itertuples()], "Scenario B": [f"{r.b:.2f} ({r.b_lo:.2f} to {r.b_hi:.2f})" for r in df.itertuples()], "Difference B - A": [f"{r.diff:+.2f}" for r in df.itertuples()]}, index=list(df.name)))
    show(P.compare_figure, art, df)
    note(f"Points: predictions with {lvl}% intervals; dashed: training mean. Model: {model}.")
    download("compare", {"A": va, "B": vb})

# ------------------------------------------------------------------ study results
with tab_study:
    sample = {"Pooled": "pooled", "Longitudinal women": "long"}[st.radio("Sample", ["Pooled", "Longitudinal women"], horizontal=True)]
    tab, nr, nw = C.r2_table(R, sample); st.markdown(f"**Out-of-sample R² against the training mean** (ridge, leave-one-woman-out; {nr} sessions, {nw} women). A: phase; C: phase + hormones; D: C + PSST, BDI, BAI. [95% bootstrap CI over women]"); st.table(tab)
    show(P.r2_figure, R, sample)
    note("Each bar is the value in the table above for the selected sample; whiskers are its 95% interval.")
    hc = C.hormone_counts(R); st.markdown(f"**Do hormones add to phase?** Of {hc['total']} paired comparisons of phase + hormones against phase alone (3 samples x 3 targets x 3 models), {hc['better']} improved prediction and {hc['worse']} made it significantly worse.")
    st.markdown("**Cross-sample R²** (ridge, against the training-group mean; trained on one group of women, tested on the other):"); st.table(C.cross_table(R))
    st.markdown(f"**Coverage of the {lvl}% prediction interval** (set D, Bayesian predictive interval):"); st.table(C.coverage_table(R, art))
    st.markdown("**Calibration of the prediction intervals** (Bayesian predictive interval): empirical against nominal coverage.")
    show(P.calibration_figure, R)
    note("Points on the dashed line are exactly calibrated; points below it mean intervals that are too narrow. Whiskers: 95% interval over women. Top row: leave-one-woman-out on the pooled sample; bottom row: trained on the single-session women, tested on the longitudinal women.")
    st.subheader("Glossary")
    st.markdown("\n".join(f"- **{term}.** {text}" for term, text in C.GLOSSARY))

# ------------------------------------------------------------------ method lab
with tab_lab:
    st.warning("Illustrative, not the study data: the two examples below use made-up problems computed with NumPy.")
    st.subheader("Gradient descent and heavy-ball momentum")
    kappa = st.select_slider("Condition number κ (log scale)", options=M.KAPPAS, value=100, key="kappa")
    res = run_optimisers(kappa)
    show(P.optimiser_figure, res)
    reached = lambda n: f"{n:,}" if n is not None else f"not reached in {res['cap']:,}"
    st.table(pd.DataFrame({"Step size": [f"1/Lmax = {res['eta_gd']:.3g}", f"η = {res['eta_hb']:.3g}, β = {res['beta']:.4f}"], "Iterations to 1e-6": [reached(res["gd_iters"]), reached(res["hb_iters"])],
                           "Theory scale": [f"about κ = {kappa:,}", f"about √κ = {kappa ** 0.5:,.1f}"]}, index=["Gradient descent", "Heavy-ball momentum"]))
    st.caption("Gradient descent is stable only for step size below 2/Lmax.")
    note(f"Quadratic loss in {M.DIM} dimensions with Lmax = 1 and μ = 1/κ; momentum uses η = 4/(√Lmax + √μ)² and β = ((√κ − 1)/(√κ + 1))². "
         "The theory column says how the count grows with κ, not its exact value, which also depends on the target accuracy and the start. Momentum can raise the loss for a while before it converges.")
    st.subheader("Ridge shrinkage")
    s = [c.slider(f"Singular value {i}", 0.5, 100.0, start, step=0.5, key=f"s{i}") for i, (c, start) in enumerate(zip(st.columns(3), [30.0, 10.0, 3.0]), start=1)]
    show(P.shrinkage_figure, s)
    st.caption(f"Directions with small singular values are shrunk first: their factor reaches one half at a smaller λ (λ = s²/n, here with n = {M.N_ILLUSTRATIVE}).")
    study = C.condition_table(R)
    if study is not None:
        st.subheader("From the study data: Hessian condition numbers")
        st.table(study[0])
        st.caption(f"These are study results, not illustrations: pooled sample, {study[1]} sessions, penalty 0 (results/phase3c_condition.csv).")

st.divider()
st.markdown(C.DATA_LINE)

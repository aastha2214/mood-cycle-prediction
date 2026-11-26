"""Course-project demo (Streamlit). Start from the project root:  streamlit run app/app.py
Reads app/model_artifacts.json (fitted final model, built by app/build_model.py) and the saved result files in results/. Nothing the user enters is stored or sent anywhere."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, streamlit as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import app_core as C

st.set_page_config(page_title="Mood prediction demo", layout="wide")
R, missing = st.cache_data(C.load_results)()
st.title("Mood prediction from cycle phase, hormones and questionnaires")
if missing:
    st.warning("Saved result files are missing: " + ", ".join(missing) + ". Run `bash run_all.sh` and `./.venv/bin/python -I app/build_model.py` first."); st.stop()
art, TN = R["art"], C.TN; I = art["inputs"]; lvl = int(round(100 * art["level"])); P = C.presets(R)
BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#888888"

def apply_preset(prefix, name):
    for k, v in P[name].items(): st.session_state[prefix + k] = v
for prefix, name in [("", "typical menses"), ("a_", "typical menses"), ("b_", "typical luteal")]:
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
model = st.sidebar.radio("Model", C.MODELS, help="Point prediction and contributions use this model; the interval is the Bayesian predictive interval.")
st.sidebar.markdown("Feature set: **D** (phase, estradiol, progesterone, PSST, BDI, BAI), fitted on " + f"{art['n_women']} women ({art['n_rows']} sessions). Sets A and C are not stored in the model file.")
st.sidebar.caption("Slider ranges are the minimum and maximum observed in the data; start values are medians or typical values per phase.")

tab_pred, tab_drv, tab_sweep, tab_cmp, tab_study = st.tabs(["Predict", "What drives the prediction", "Hormone sweep", "Compare scenarios", "Study results"])

# ------------------------------------------------------------------ Predict
with tab_pred:
    st.info("Demo of a course project; not a medical or diagnostic tool. Intervals were roughly calibrated within the study but not across the two groups of women.")
    cols = st.columns(3)
    for c, name in zip(cols, P): c.button(name.capitalize(), on_click=apply_preset, args=("", name), use_container_width=True)
    st.caption("Typical values: medians of the longitudinal women in that phase (hormones) and of all women (questionnaires). High scores: the largest PSST, BDI and BAI observed in the data.")
    left, right = st.columns([1, 2]); v = inputs("", left)
    with right:
        cards = st.columns(3); fig, axes = plt.subplots(3, 1, figsize=(6, 3.6), sharex=False)
        for c, ax, t in zip(cards, axes, art["targets"]):
            M = art["targets"][t]; pt, lo, hi = C.point_and_interval(art, t, v, model)
            c.metric(TN[t], f"{pt:.2f}", f"{pt - M['ybar']:+.2f} vs training mean {M['ybar']:.2f}", delta_color="off"); c.caption(f"{lvl}% prediction interval: {lo:.2f} to {hi:.2f}")
            ax.axvspan(M["obs_min"], M["obs_max"], color="#EEEEEE", zorder=0); ax.axvline(M["ybar"], color=GREY, ls="--", lw=1.2, label="training mean")
            ax.errorbar([pt], [0], xerr=[[pt - lo], [hi - pt]], fmt="o", color=BLUE, capsize=4, lw=2, label=f"prediction and {lvl}% interval")
            ax.set_yticks([]); ax.set_title(TN[t], fontsize=8, loc="left"); ax.tick_params(labelsize=8); ax.grid(axis="x", alpha=.25); [ax.spines[s].set_visible(False) for s in ["top", "right", "left"]]
            ax.set_xlim(min(M["obs_min"], lo) - 0.05 * M["obs_sd"], max(M["obs_max"], hi) + 0.05 * M["obs_sd"])
        axes[0].legend(frameon=False, fontsize=7, loc="upper right"); fig.suptitle("Grey band: range observed in the data", fontsize=8); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

# ------------------------------------------------------------------ drivers
with tab_drv:
    t = st.selectbox("Target", list(TN), format_func=TN.get); M = art["targets"][t]; df = C.contributions(art, t, v, model); pt = C.point_and_interval(art, t, v, model)[0]
    fig, ax = plt.subplots(figsize=(6, 3)); ax.barh(df.feature[::-1], df.contribution[::-1], color=[BLUE if x >= 0 else ORANGE for x in df.contribution[::-1]]); ax.axvline(0, color="#333", lw=.8)
    ax.set_xlabel("contribution to this prediction, in the target's units", fontsize=8); ax.tick_params(labelsize=8); ax.grid(axis="x", alpha=.25); [ax.spines[s].set_visible(False) for s in ["top", "right"]]; fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.caption(f"Contribution = coefficient x standardized input. Training mean {M['ybar']:.2f} + sum of contributions {df.contribution.sum():+.2f} = prediction {pt:.2f} (inputs from the Predict tab, model: {model}).")

# ------------------------------------------------------------------ sweep
with tab_sweep:
    fig, axes = plt.subplots(2, 3, figsize=(9, 4.4))
    for j, t in enumerate(art["targets"]):
        for i, (which, nm) in enumerate([("e2", "Estradiol (pg/ml)"), ("pg", "Progesterone (pg/ml)")]):
            ax = axes[i, j]; s = C.sweep(art, t, v, which, model); ax.fill_between(s.x, s.lo, s.hi, color=BLUE, alpha=.15); ax.plot(s.x, s.pred, color=BLUE); ax.axhline(art["targets"][t]["ybar"], color=GREY, ls="--", lw=1)
            ax.axvline(v[which], color=ORANGE, lw=1); ax.set_xlabel(nm, fontsize=8); ax.tick_params(labelsize=8); ax.grid(alpha=.25); [ax.spines[k].set_visible(False) for k in ["top", "right"]]
            if i == 0: ax.set_title(TN[t], fontsize=8)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    sd_share, w_share = C.sweep_summary(art, v, model); hc = C.hormone_counts(R)
    st.markdown(f"Everything else is held at the Predict-tab inputs (orange line); band = {lvl}% interval; dashed = training mean. With phase in the model the hormone line is nearly flat: across the whole observed hormone range the prediction moves by at most {100 * sd_share:.0f}% of the outcome's standard deviation, "
                f"at most {100 * w_share:.0f}% of the interval width. In the study, adding the hormones to phase improved out-of-sample RMSE in {hc['better']} of {hc['total']} comparisons and made it worse in {hc['worse']}.")

# ------------------------------------------------------------------ compare scenarios
with tab_cmp:
    ca, cb = st.columns(2); va, vb = {}, {}
    for col, prefix, lab in [(ca, "a_", "Scenario A"), (cb, "b_", "Scenario B")]:
        col.subheader(lab); col.selectbox("Start from", list(P), key=prefix + "preset", index=None, placeholder="choose a preset", on_change=lambda p=prefix: apply_preset(p, st.session_state[p + "preset"]))
    va = inputs("a_", ca); vb = inputs("b_", cb); df = C.compare(art, va, vb, model)
    st.table(pd.DataFrame({"Scenario A": [f"{r.a:.2f} ({r.a_lo:.2f} to {r.a_hi:.2f})" for r in df.itertuples()], "Scenario B": [f"{r.b:.2f} ({r.b_lo:.2f} to {r.b_hi:.2f})" for r in df.itertuples()], "Difference B - A": [f"{r.diff:+.2f}" for r in df.itertuples()]}, index=df.name))
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.3))
    for ax, r in zip(axes, df.itertuples()):
        ax.errorbar([0, 1], [r.a, r.b], yerr=[[r.a - r.a_lo, r.b - r.b_lo], [r.a_hi - r.a, r.b_hi - r.b]], fmt="o", color=BLUE, capsize=4); ax.axhline(art["targets"][r.target]["ybar"], color=GREY, ls="--", lw=1)
        ax.set_xticks([0, 1], ["A", "B"]); ax.set_xlim(-.5, 1.5); ax.set_title(r.name, fontsize=8); ax.tick_params(labelsize=8); ax.grid(axis="y", alpha=.25); [ax.spines[k].set_visible(False) for k in ["top", "right"]]
    fig.tight_layout(); st.pyplot(fig); plt.close(fig); st.caption(f"Points: predictions with {lvl}% intervals; dashed: training mean. Model: {model}.")

# ------------------------------------------------------------------ study results
with tab_study:
    sample = {"Pooled": "pooled", "Longitudinal women": "long"}[st.radio("Sample", ["Pooled", "Longitudinal women"], horizontal=True)]
    tab, nr, nw = C.r2_table(R, sample); st.markdown(f"**Out-of-sample R2 against the training mean** (ridge, leave-one-woman-out; {nr} sessions, {nw} women). A: phase; C: phase + hormones; D: C + PSST, BDI, BAI. [95% bootstrap CI over women]"); st.table(tab)
    v_ = C.r2_values(R, sample); fig, ax = plt.subplots(figsize=(7, 2.6)); w = .25
    for k, (fs, col) in enumerate(zip(["A", "C", "D"], [BLUE, "#009E73", ORANGE])):
        q = v_[v_.fset == fs].set_index("target").loc[list(TN)]; ax.bar(np.arange(3) + (k - 1) * w, q.R2_oos, w, yerr=[q.R2_oos - q.R2_oos_lo, q.R2_oos_hi - q.R2_oos], color=col, capsize=3, label=f"set {fs}")
    ax.axhline(0, color="#333", lw=.8); ax.set_xticks(range(3), [TN[t] for t in TN], fontsize=8); ax.set_ylabel("R2 vs training mean", fontsize=8); ax.tick_params(labelsize=8); ax.legend(frameon=False, fontsize=8); ax.grid(axis="y", alpha=.25); [ax.spines[s].set_visible(False) for s in ["top", "right"]]; fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    hc = C.hormone_counts(R); st.markdown(f"**Do hormones add to phase?** Of {hc['total']} paired comparisons of phase + hormones against phase alone (3 samples x 3 targets x 3 models), {hc['better']} improved prediction and {hc['worse']} made it significantly worse.")
    st.markdown("**Cross-sample R2** (ridge, against the training-group mean; trained on one group of women, tested on the other):"); st.table(C.cross_table(R))
    st.markdown(f"**Coverage of the {lvl}% prediction interval** (set D, Bayesian predictive interval):"); st.table(C.coverage_table(R, art))

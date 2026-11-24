"""E6: educational Streamlit demo. Start from the project root:  streamlit run app/app.py
Educational demo, not medical advice. Nothing the user enters is stored, logged or sent anywhere; the app reads only app/model_artifacts.json (the fitted final model, built by app/build_model.py)
and the saved result files in results/ (every statistic shown is loaded from them)."""
import json
from pathlib import Path
import numpy as np, pandas as pd, streamlit as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
TN = {"pos_affect": "Positive affect", "drsp_psych": "DRSP psychological symptoms", "stai": "State anxiety"}; PHASES = ["menses", "peri-ovulatory", "luteal"]
st.set_page_config(page_title="Mood prediction demo (educational)", layout="wide")

@st.cache_data
def load_all():
    need = {"art": HERE / "model_artifacts.json", "m4": ROOT / "results/phase4_metrics.csv", "d4": ROOT / "results/phase4_deltas.csv", "cov": ROOT / "results/extE4_coverage.csv", "e7": ROOT / "results/ext_e7_metrics.csv", "e7h": ROOT / "results/ext_e7_hormone_tests.csv", "e7s": ROOT / "results/ext_e7_sizes.csv"}
    missing = [str(p.relative_to(ROOT)) for p in need.values() if not p.exists()]
    if missing: return None, missing
    out = {"art": json.loads(need["art"].read_text())}; out.update({k: pd.read_csv(p) for k, p in need.items() if k != "art"}); return out, []
R, missing = load_all()
st.title("Mood prediction from cycle phase, hormones and questionnaires")
st.caption("Learning project built on one open research dataset.")
if missing:
    st.warning("Saved result files are missing: " + ", ".join(missing) + ". Run `bash run_all.sh` and `./.venv/bin/python -I app/build_model.py` first."); st.stop()
A, m4, d4, cov, e7, e7h, e7s = R["art"], R["m4"], R["d4"], R["cov"], R["e7"], R["e7h"], R["e7s"].set_index("direction"); T = A["targets"]; IN = A["inputs"]
mu, sd, zm = (np.array(A[k]) for k in ["mu", "sd", "zm"])
def predict(t, phase, e2, p, pms, bdi, bai):
    """Ridge point prediction and Bayesian ridge predictive mean/sd, same transforms as in the validation (log1p, standardise with training statistics)."""
    x = np.array([phase == "peri-ovulatory", phase == "luteal", np.log1p(e2), np.log1p(p), pms, bdi, bai], float); z = (x - mu) / sd - zm; M = T[t]
    ridge = float(z @ np.array(M["w"]) + M["ybar"]); mean_b = float(z @ np.array(M["m"]) + M["ybar"]); sd_b = float(np.sqrt(1.0 / M["beta"] + z @ np.array(M["S"]) @ z)); return ridge, mean_b, sd_b
# ------------------------------------------------------------------ sidebar inputs (ranges from the data)
st.sidebar.header("Inputs (kept in memory only)")
phase = st.sidebar.selectbox("Cycle phase", PHASES, index=0)
e2 = st.sidebar.slider("Estradiol (saliva, pg/ml)", IN["estradiol"]["min"], IN["estradiol"]["max"], IN["estradiol"]["median"], step=IN["estradiol"]["max"] / 200)
pg = st.sidebar.slider("Progesterone (saliva, pg/ml)", IN["progesterone"]["min"], IN["progesterone"]["max"], IN["progesterone"]["median"], step=IN["progesterone"]["max"] / 200)
pms = st.sidebar.slider("PSST score (premenstrual symptoms screening)", int(IN["PMS"]["min"]), int(IN["PMS"]["max"]), int(round(IN["PMS"]["median"])))
bdi = st.sidebar.slider("BDI score (depressive symptoms)", int(IN["BDI_RW"]["min"]), int(IN["BDI_RW"]["max"]), int(round(IN["BDI_RW"]["median"])))
bai = st.sidebar.slider("BAI score (anxiety symptoms)", int(IN["BAI_RW"]["min"]), int(IN["BAI_RW"]["max"]), int(round(IN["BAI_RW"]["median"])))
st.sidebar.caption(f"Slider ranges are the minimum and maximum in the {A['n_rows']} sessions of {A['n_women']} women used to fit the model; the starting values are the medians. Entries outside such ranges would be extrapolation.")
# ------------------------------------------------------------------ predictions with saved validation numbers
lvl = int(round(100 * A["level"])); rows = []; plot = []
for t in T:
    ridge, mean_b, sd_b = predict(t, phase, e2, pg, pms, bdi, bai); lo, hi = mean_b - A["z_level"] * sd_b, mean_b + A["z_level"] * sd_b
    mm = m4[(m4["sample"] == "pooled") & (m4.target == t) & (m4.model == "ridge") & (m4.fset == "D")].iloc[0]; plot.append((t, ridge, mean_b, lo, hi, T[t]["ybar"]))
    rows.append({"Target": TN[t], "Ridge prediction": f"{ridge:.2f}", "Bayesian predictive mean": f"{mean_b:.2f}", f"{lvl}% prediction interval": f"{lo:.2f} to {hi:.2f}", "Training-mean baseline": f"{T[t]['ybar']:.2f}",
                 "Test R2 [95% CI] (leave-one-woman-out)": f"{mm.R2:+.3f} [{mm.R2_lo:+.3f}, {mm.R2_hi:+.3f}]", "Test R2 vs baseline [95% CI]": f"{mm.R2_oos:+.3f} [{mm.R2_oos_lo:+.3f}, {mm.R2_oos_hi:+.3f}]", "Observed range in the data": f"{T[t]['obs_min']:.2f} to {T[t]['obs_max']:.2f}"})
st.subheader("Predictions for the entered values")
st.table(pd.DataFrame(rows).set_index("Target"))   # HTML table: wraps text and fits the page width, so the last column needs no horizontal scrolling
nw, nr = int(mm.n_women), int(mm.n_rows)
st.caption(f"Model: ridge regression (lambda chosen by nested cross-validation) and Bayesian ridge (E2), full feature set D, fitted on all {A['n_women']} women ({A['n_rows']} sessions). Test R2 values come from leave-one-woman-out validation on {nw} women ({nr} sessions), loaded from results/phase4_metrics.csv. The two predictions are almost identical by construction; the interval belongs to the Bayesian one and describes a single new session.")
# ------------------------------------------------------------------ chart
fig, axes = plt.subplots(len(plot), 1, figsize=(8, 1.7 * len(plot) + 0.9))
for ax, (t, ridge, mean_b, lo, hi, base) in zip(np.atleast_1d(axes), plot):
    ax.axvspan(T[t]["obs_min"], T[t]["obs_max"], color="#EEEEEE", zorder=0); ax.axvline(base, color="#888888", ls="--", lw=1.2, label="training-mean baseline")
    ax.errorbar([mean_b], [0], xerr=[[mean_b - lo], [hi - mean_b]], fmt="o", color="#0072B2", capsize=5, lw=2, label=f"Bayesian mean and {lvl}% interval"); ax.plot([ridge], [0], "D", color="#D55E00", ms=6, label="ridge prediction")
    ax.set_yticks([]); ax.set_title(TN[t], fontsize=10, loc="left"); ax.set_xlim(min(T[t]["obs_min"], lo) - 0.05 * T[t]["obs_sd"], max(T[t]["obs_max"], hi) + 0.05 * T[t]["obs_sd"]); ax.grid(axis="x", alpha=.25)
    for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7, loc="upper right", ncol=1); fig.suptitle("Predictions with prediction intervals (grey band = range observed in the data)", fontsize=9); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
# ------------------------------------------------------------------ plain-language box
nimp = int(((d4.comparison == "C vs A") & (d4.metric == "RMSE") & (d4.delta < 0) & (d4.hi < 0)).sum()); ntot = int(((d4.comparison == "C vs A") & (d4.metric == "RMSE")).sum()); nwor = int(((d4.comparison == "C vs A") & (d4.metric == "RMSE") & (d4.delta > 0) & (d4.lo > 0)).sum())
bd = m4[(m4["sample"] == "pooled") & (m4.model == "ridge") & (m4.fset == "D") & (m4.target == "drsp_psych")].iloc[0]; ba = m4[(m4["sample"] == "pooled") & (m4.model == "ridge") & (m4.fset == "A") & (m4.target == "drsp_psych")].iloc[0]
st.info(f"**What this demo can and cannot tell you.** This model predicts well only from questionnaire scores (BDI, BAI, PSST): for DRSP symptoms the test R2 with them is {bd.R2:+.3f}, with cycle phase alone {ba.R2:+.3f}. Hormone levels added no predictive value beyond cycle phase in this study: of {ntot} paired comparisons, {nimp} showed an improvement and {nwor} were worse. "
        f"The model was validated on {nw} women from one open study (Pletzer and Noachtar, 2023), so it says nothing about other populations.")
with st.expander("How wide and how reliable are the intervals? (E4 and E7)"):
    c = cov[(cov.fset == "D") & (cov.method == "bayes_predictive") & (np.isclose(cov.level, A["level"]))]
    for t in T:
        a = c[(c["sample"] == "pooled") & (c.target == t)].iloc[0]; b = c[(c["sample"] == "cross2long") & (c.target == t)].iloc[0]
        st.markdown(f"**{TN[t]}**: the {lvl}% interval has a mean width of {a.mean_width:.2f}, while the outcome's standard deviation is {T[t]['obs_sd']:.2f}, so the interval is wide. In leave-one-woman-out validation ({int(a.n_rows)} sessions, {int(a.n_women)} women) it covered {a.coverage:.3f} of the new sessions [{a.cov_lo:.3f}, {a.cov_hi:.3f}]. "
                    f"When the model was trained on {int(e7s.loc['cross_to_long', 'train_women'])} women with one session and tested on a different group of women ({int(b.n_rows)} sessions, {int(b.n_women)} women), coverage was only {b.coverage:.3f} [{b.cov_lo:.3f}, {b.cov_hi:.3f}].")
    st.markdown("Intervals are for a **single new session** and assume the new woman resembles the women in the study. Their coverage failed across the two groups of women (E4, E7), so do not treat them as guaranteed ranges.")
    st.markdown("**Cross-sample transfer of the ridge model with questionnaires (E7, R2 against the training-group mean):**")
    for t in T:
        r1 = e7[(e7["sample"] == "cross_to_long") & (e7.target == t) & (e7.model == "ridge") & (e7.fset == "D")].iloc[0]; r2 = e7[(e7["sample"] == "long_to_cross") & (e7.target == t) & (e7.model == "ridge") & (e7.fset == "D")].iloc[0]
        st.markdown(f"- {TN[t]}: trained on single-session women, tested on {int(r1.n_women)} longitudinal women: {r1.R2_oos:+.3f} [{r1.R2_oos_lo:+.3f}, {r1.R2_oos_hi:+.3f}]; trained on longitudinal women, tested on {int(r2.n_women)} single-session women: {r2.R2_oos:+.3f} [{r2.R2_oos_lo:+.3f}, {r2.R2_oos_hi:+.3f}].")
with st.expander("What about the hormone inputs?"):
    h = e7h[e7h.family == "C vs A"]
    st.markdown(f"Hormones are inputs because the research question asked whether they help, and the answer in this study was no. Adding estradiol and progesterone to cycle phase improved out-of-sample prediction in {nimp} of {ntot} comparisons (leave-one-woman-out) and in {int(((h.p_improve_one_sided < 0.05) & (h.p_holm_all_improve < 0.05)).sum())} of {len(h)} comparisons across the two groups of women after Holm correction. Changing the hormone sliders therefore moves the prediction only by the small amount the fitted ridge assigns to them; see the report (section 4.2 and 8.6).")
st.caption("Privacy: the values you enter are used only inside this browser session to compute the numbers above; the app writes no files, keeps no log and sends nothing anywhere. Research data stay on this computer.")

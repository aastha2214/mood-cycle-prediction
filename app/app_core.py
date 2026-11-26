"""Pure functions of the demo (no Streamlit): prediction, contributions, hormone sweep, scenario comparison, downloads, URL settings, key findings and the tables of the study results.
Everything is computed from app/model_artifacts.json and the saved result files in results/ (read only)."""
import json, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
TN = {"pos_affect": "Positive affect", "drsp_psych": "DRSP psychological symptoms", "stai": "State anxiety"}
PHASES = ["menses", "peri-ovulatory", "luteal"]
FEATURE_LABELS = ["Peri-ovulatory phase", "Luteal phase", "Estradiol (log)", "Progesterone (log)", "PSST", "BDI", "BAI"]
MODELS = ["Ridge", "Bayesian ridge"]
INPUT_KEYS = ["phase", "e2", "pg", "pms", "bdi", "bai"]
FILES = {"art": HERE / "model_artifacts.json", "m4": ROOT / "results/phase4_metrics.csv", "d4": ROOT / "results/phase4_deltas.csv", "cov": ROOT / "results/extE4_coverage.csv",
         "e7": ROOT / "results/ext_e7_metrics.csv", "e7s": ROOT / "results/ext_e7_sizes.csv", "hp": ROOT / "results/phase5_hormones_by_phase_long.csv"}
OPTIONAL = {"cond": ROOT / "results/phase3c_condition.csv"}   # shown in the Method lab only if present
REPO_URL = "https://github.com/aastha2214/mood-cycle-prediction"
HONESTY = "Demo of a course project; not a medical or diagnostic tool. Intervals were roughly calibrated within the study but not across the two groups of women."
DATA_LINE = "Data: Pletzer and Noachtar (2023), Open Science Framework project dmhrw; secondary analysis of public, de-identified data."
DEFAULT_PRESET = "typical menses"

def load_results():
    """Return (dict of artifacts and tables, list of missing files)."""
    missing = [str(p.relative_to(ROOT)) for p in FILES.values() if not p.exists()]
    if missing: return None, missing
    out = {k: pd.read_csv(p) for k, p in FILES.items() if k != "art"}; out["art"] = json.loads(FILES["art"].read_text())
    for k, p in OPTIONAL.items(): out[k] = pd.read_csv(p) if p.exists() else None
    return out, []

def design_vector(art, v):
    """Standardised, centred feature vector for the inputs v = dict(phase, e2, pg, pms, bdi, bai); same transforms as in the validation (log1p, training statistics)."""
    x = np.array([v["phase"] == "peri-ovulatory", v["phase"] == "luteal", np.log1p(v["e2"]), np.log1p(v["pg"]), v["pms"], v["bdi"], v["bai"]], float)
    return (x - np.array(art["mu"])) / np.array(art["sd"]) - np.array(art["zm"])

def predict(art, t, v):
    """Ridge prediction, Bayesian ridge predictive mean and predictive sd for target t."""
    z = design_vector(art, v); M = art["targets"][t]
    return (float(z @ np.array(M["w"]) + M["ybar"]), float(z @ np.array(M["m"]) + M["ybar"]), float(np.sqrt(1.0 / M["beta"] + z @ np.array(M["S"]) @ z)))

def point_and_interval(art, t, v, model):
    """Chosen point prediction with the Bayesian predictive interval (one new session) of the saved level."""
    ridge, mean_b, sd_b = predict(art, t, v); pt = ridge if model == "Ridge" else mean_b; h = art["z_level"] * sd_b
    return pt, pt - h, pt + h

def contributions(art, t, v, model):
    """Coefficient times standardised input for each feature; the training mean of the target plus the sum of the contributions is the point prediction."""
    z = design_vector(art, v); M = art["targets"][t]; w = np.array(M["w"] if model == "Ridge" else M["m"])
    return pd.DataFrame({"feature": FEATURE_LABELS, "contribution": z * w})

def sweep(art, t, v, which, model, n=80):
    """Vary estradiol (which='e2') or progesterone ('pg') over the observed range, everything else fixed."""
    key = {"e2": "estradiol", "pg": "progesterone"}[which]; grid = np.linspace(art["inputs"][key]["min"], art["inputs"][key]["max"], n); rows = []
    for g in grid: pt, lo, hi = point_and_interval(art, t, {**v, which: float(g)}, model); rows.append((g, pt, lo, hi))
    return pd.DataFrame(rows, columns=["x", "pred", "lo", "hi"])

def sweep_summary(art, v, model):
    """Largest movement of the prediction across the observed hormone range, as a share of the outcome SD and of the interval width, over targets and both hormones."""
    best_sd = best_w = 0.0
    for t in art["targets"]:
        for which in ["e2", "pg"]:
            s = sweep(art, t, v, which, model); move = s.pred.max() - s.pred.min(); best_sd = max(best_sd, move / art["targets"][t]["obs_sd"]); best_w = max(best_w, move / float((s.hi - s.lo).mean()))
    return best_sd, best_w

def compare(art, va, vb, model):
    """Table of scenario A, scenario B and the difference B - A for the three targets, with intervals."""
    rows = []
    for t in art["targets"]:
        a, b = point_and_interval(art, t, va, model), point_and_interval(art, t, vb, model); rows.append(dict(target=t, name=TN[t], a=a[0], a_lo=a[1], a_hi=a[2], b=b[0], b_lo=b[1], b_hi=b[2], diff=b[0] - a[0]))
    return pd.DataFrame(rows)

def presets(R):
    """Input presets. 'Typical' hormone values are the medians of the longitudinal women in that phase (results/phase5_hormones_by_phase_long.csv); questionnaires: medians of all
    women (model_artifacts.json). 'High questionnaire scores': the largest values observed in the data."""
    I = R["art"]["inputs"]; hp = R["hp"].set_index("Phase"); q = {k: int(round(I[s]["median"])) for k, s in [("pms", "PMS"), ("bdi", "BDI_RW"), ("bai", "BAI_RW")]}
    mk = lambda ph, code: dict(phase=ph, e2=float(hp.loc[code, "E2_median"]), pg=float(hp.loc[code, "P_median"]), **q)
    return {"typical menses": mk("menses", 0.0), "typical luteal": mk("luteal", 2.0),
            "high questionnaire scores": dict(phase="menses", e2=float(hp.loc[0.0, "E2_median"]), pg=float(hp.loc[0.0, "P_median"]), pms=int(I["PMS"]["max"]), bdi=int(I["BDI_RW"]["max"]), bai=int(I["BAI_RW"]["max"]))}

# ------------------------------------------------------------------ study results (all loaded from results/)
def r2_table(R, sample="pooled", sets=("A", "C", "D")):
    m = R["m4"]; q = m[(m["sample"] == sample) & (m.model == "ridge") & m.fset.isin(sets)]; rows = {}
    for fs in sets:
        rows[f"Set {fs}"] = {TN[t]: (lambda r: f"{r.R2_oos:+.3f} [{r.R2_oos_lo:+.3f}, {r.R2_oos_hi:+.3f}]")(q[(q.fset == fs) & (q.target == t)].iloc[0]) for t in TN}
    r0 = q.iloc[0]; return pd.DataFrame(rows).T, int(r0.n_rows), int(r0.n_women)

def r2_values(R, sample="pooled", sets=("A", "C", "D")):
    m = R["m4"]; return m[(m["sample"] == sample) & (m.model == "ridge") & m.fset.isin(sets)][["target", "fset", "R2_oos", "R2_oos_lo", "R2_oos_hi", "n_rows", "n_women"]]

def hormone_counts(R):
    """Paired comparisons of set C (phase + hormones) against set A (phase only), RMSE: total, significantly better, significantly worse."""
    q = R["d4"]; q = q[(q.comparison == "C vs A") & (q.metric == "RMSE")]
    return dict(total=len(q), better=int(((q.delta < 0) & (q.hi < 0)).sum()), worse=int(((q.delta > 0) & (q.lo > 0)).sum()))

def cross_table(R, sets=("A", "C", "D")):
    """Cross-sample R2 (ridge, against the training-group mean): one row per direction and feature set, one column per target."""
    e, s = R["e7"], R["e7s"].set_index("direction"); name = {"cross_to_long": "single-session -> longitudinal", "long_to_cross": "longitudinal -> single-session"}; rows = {}
    for d, nm in name.items():
        for fs in sets:
            g = e[(e["sample"] == d) & (e.model == "ridge") & (e.fset == fs)].set_index("target"); rows[(f"{nm} (train {int(s.loc[d, 'train_women'])}, test {int(s.loc[d, 'test_women'])} women)", f"Set {fs}")] = {TN[t]: f"{g.loc[t, 'R2_oos']:+.3f} [{g.loc[t, 'R2_oos_lo']:+.3f}, {g.loc[t, 'R2_oos_hi']:+.3f}]" for t in TN}
    df = pd.DataFrame(rows).T; df.index.names = ["Direction", "Features"]; return df

def coverage_table(R, art):
    """Empirical coverage of the Bayesian predictive interval (set D) within the study (leave-one-woman-out, pooled) and across the two groups of women."""
    c = R["cov"]; c = c[(c.fset == "D") & (c.method == "bayes_predictive") & np.isclose(c.level, art["level"])]; rows = {}
    for t in TN:
        a = c[(c["sample"] == "pooled") & (c.target == t)].iloc[0]; b = c[(c["sample"] == "cross2long") & (c.target == t)].iloc[0]
        rows[TN[t]] = {f"Within the study ({int(a.n_women)} women)": f"{a.coverage:.3f} [{a.cov_lo:.3f}, {a.cov_hi:.3f}]", f"Across groups ({int(b.n_women)} test women)": f"{b.coverage:.3f} [{b.cov_lo:.3f}, {b.cov_hi:.3f}]"}
    return pd.DataFrame(rows).T

# ------------------------------------------------------------------ downloads
def download_table(art, scenarios, model):
    """One row per scenario (name -> inputs): model, inputs, and for each target the prediction with its prediction interval."""
    lvl = int(round(100 * art["level"])); rows = []
    for name, v in scenarios.items():
        row = {"scenario": name, "model": model, "cycle_phase": v["phase"], "estradiol_pg_ml": v["e2"], "progesterone_pg_ml": v["pg"], "PSST": v["pms"], "BDI": v["bdi"], "BAI": v["bai"]}
        for t in art["targets"]:
            pt, lo, hi = point_and_interval(art, t, v, model); row.update({f"{t}_prediction": pt, f"{t}_interval{lvl}_low": lo, f"{t}_interval{lvl}_high": hi})
        rows.append(row)
    return pd.DataFrame(rows)

# ------------------------------------------------------------------ settings in the URL
QUERY_NAMES = {"phase": "phase", "e2": "estradiol", "pg": "progesterone", "pms": "psst", "bdi": "bdi", "bai": "bai"}   # input key -> name in the URL
QUERY_RANGES = {"e2": ("estradiol", float), "pg": ("progesterone", float), "pms": ("PMS", int), "bdi": ("BDI_RW", int), "bai": ("BAI_RW", int)}
QUERY_MODELS = {"ridge": "Ridge", "bayesian": "Bayesian ridge"}

def parse_query(art, params):
    """(valid inputs, model or None) from URL parameters; anything unknown, unreadable or outside the slider range is left out."""
    out = {}
    for key, name in QUERY_NAMES.items():
        raw = params.get(name)
        if raw is None: continue
        if key == "phase":
            if raw in PHASES: out[key] = raw
            continue
        source, kind = QUERY_RANGES[key]; limit = art["inputs"][source]
        try: value = kind(raw)
        except (TypeError, ValueError, OverflowError): continue
        if math.isfinite(value) and limit["min"] <= value <= limit["max"]: out[key] = value
    return out, QUERY_MODELS.get(params.get("model"))

def query_from(v, model):
    """URL parameters (strings) for the inputs v and the model."""
    out = {name: (v[key] if key == "phase" else f"{v[key]:.10g}") for key, name in QUERY_NAMES.items()}
    out["model"] = {m: k for k, m in QUERY_MODELS.items()}[model]; return out

# ------------------------------------------------------------------ key findings, calibration, condition numbers (all read from results/)
def key_findings(R):
    """Three short statements whose numbers all come from the saved result files."""
    hc = hormone_counts(R); m = R["m4"]; r = m[(m["sample"] == "pooled") & (m.model == "ridge") & (m.fset == "D") & (m.target == "drsp_psych")].iloc[0]
    c = R["cov"]; c = c[(c.fset == "D") & (c.method == "bayes_predictive") & np.isclose(c.level, R["art"]["level"])]; w, x = c[c["sample"] == "pooled"], c[c["sample"] == "cross2long"]
    level = f"{100 * w.level.iloc[0]:.0f}%"; span = lambda q: f"{q.coverage.min():.3f} to {q.coverage.max():.3f}"; nw, nx = int(w.n_women.iloc[0]), int(x.n_women.iloc[0])
    as_stated = bool(w.covers_nominal.all() and (x.coverage < x.level).all() and not x.covers_nominal.any())
    third = (f"Set D {level} intervals are close to nominal within the study (coverage {span(w)}, {nw} women) but undercover across the two groups of women ({span(x)}, {nx} test women)." if as_stated
             else f"Coverage of the set D {level} intervals: {span(w)} within the study ({nw} women), {span(x)} across the two groups of women ({nx} test women).")
    return [f"Adding hormones to cycle phase improved prediction in {hc['better']} of {hc['total']} comparisons and made it significantly worse in {hc['worse']}.",
            f"Questionnaires carry the signal: pooled ridge, DRSP psychological symptoms, set D, out-of-sample R\u00b2 {r.R2_oos:+.3f} [{r.R2_oos_lo:+.3f}, {r.R2_oos_hi:+.3f}] ({int(r.n_women)} women).", third]

def calibration_data(R, sets=("A", "C", "D")):
    """Empirical coverage of the Bayesian predictive interval at every saved nominal level, within the study (pooled) and across the two groups of women."""
    c = R["cov"]; q = c[(c.method == "bayes_predictive") & c["sample"].isin(["pooled", "cross2long"]) & c.fset.isin(sets)]
    return q[["sample", "target", "fset", "level", "coverage", "cov_lo", "cov_hi", "n_rows", "n_women"]].sort_values(["sample", "target", "fset", "level"]).reset_index(drop=True)

CONDITION_LABELS = {"raw (unscaled, raw hormones)": "Raw hormones, unscaled", "log1p hormones, unscaled": "log(1+x) hormones, unscaled", "log1p + standardised": "log(1+x) hormones, standardized"}

def condition_table(R):
    """(table, sessions) of the saved Hessian condition numbers of the study data (pooled sample, penalty 0), or None if the file does not provide them."""
    q = R.get("cond")
    if q is None or not {"sample", "variant", "lam", "kappa", "n"} <= set(q.columns): return None
    q = q[(q["sample"] == "pooled") & np.isclose(q.lam, 0.0)].set_index("variant")
    if not set(CONDITION_LABELS) <= set(q.index): return None
    fmt = lambda k: f"{k:,.0f}" if k >= 100 else f"{k:.1f}"
    return pd.DataFrame({"Condition number": [fmt(q.loc[k, "kappa"]) for k in CONDITION_LABELS]}, index=list(CONDITION_LABELS.values())), int(q.n.iloc[0])

GLOSSARY = [
    ("DRSP", "The Daily Record of Severity of Problems is a self-rating of premenstrual symptoms; this app predicts a sum score of its psychological symptoms."),
    ("PSST", "The Premenstrual Symptoms Screening Tool is a questionnaire that screens for premenstrual symptoms."),
    ("BDI", "The Beck Depression Inventory is a self-report questionnaire about depressive symptoms."),
    ("BAI", "The Beck Anxiety Inventory is a self-report questionnaire about anxiety symptoms."),
    ("STAI", "The State-Trait Anxiety Inventory measures anxiety; the state anxiety target here is a mean over its items."),
    ("R\u00b2 against the training mean", "The share of squared prediction error removed compared with always predicting the training average: 0 means no better, below 0 means worse."),
    ("Prediction interval", "A range meant to contain the value of one new session with a stated probability, if the model's assumptions hold."),
    ("Leave-one-woman-out validation", "Each woman in turn is left out completely, the model is trained on the other women and then tested on her sessions."),
    ("Cross-sample validation", "The model is trained on one group of women of the study and tested on the other group; it is not a test on a second dataset."),
    ("Bayesian ridge", "Ridge regression written as a probability model, which gives an uncertainty for each prediction as well as a point value."),
    ("Condition number", "The ratio of the steepest to the flattest curvature of the loss; the larger it is, the slower plain gradient descent becomes.")]

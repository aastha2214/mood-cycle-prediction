"""Pure functions of the demo (no Streamlit): prediction, contributions, hormone sweep, scenario comparison and the tables of the study results.
Everything is computed from app/model_artifacts.json and the saved result files in results/ (read only)."""
import json
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

def load_results():
    """Return (dict of artifacts and tables, list of missing files)."""
    missing = [str(p.relative_to(ROOT)) for p in FILES.values() if not p.exists()]
    if missing: return None, missing
    out = {k: pd.read_csv(p) for k, p in FILES.items() if k != "art"}; out["art"] = json.loads(FILES["art"].read_text()); return out, []

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

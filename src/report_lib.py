"""Helpers that LOAD every reported number from results/*.csv (or the data) and log where it came from (-> docs/numbers_provenance.md)."""
import re
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
SCRIPT = {"results/icc.csv": "src/icc.py", "results/phase3a_sklearn_check.csv": "src/phase3a_verify.py", "results/phase3e_map_check.csv": "src/phase3a_verify.py",
          "results/phase3b_lr_study.csv": "src/phase3b_optim.py", "results/phase3b_momentum_tuned.csv": "src/phase3b2_momentum_tuned.py", "results/phase3c_condition.csv": "src/phase3c_hessian.py",
          "results/phase3d_bias_variance.csv": "src/phase3d_path_cv.py", "results/phase3d_nested_cv.csv": "src/phase3d_path_cv.py",
          "results/phase4_metrics.csv": "src/phase4_report.py (from results/phase4_oof.csv <- src/phase4_run.py)", "results/phase4_deltas.csv": "src/phase4_report.py",
          "results/phase4_loso.csv": "src/phase4_leak_loso.py", "results/phase4_shuffle_test.csv": "src/phase4_leak_loso.py",
          "results/phase5_mixed.csv": "src/phase5_stats.py", "results/phase5_perm.csv": "src/phase5_stats.py", "results/phase5_r2_phase.csv": "src/phase5_stats.py",
          "results/phase5_hormones_by_phase_long.csv": "src/phase5_stats.py", "results/phase5_e2_given_phase.csv": "src/phase5_e2_given_phase.py", "results/phase5_multcomp.csv": "src/phase5_multcomp.py",
          "results/phase6a_metrics.csv": "src/phase6_negctrl.py (src/evalutils.py)", "results/phase6a_deltas.csv": "src/phase6_negctrl.py (src/evalutils.py)", "results/phase6a_mixed.csv": "src/phase6_negctrl.py",
          "results/phase6b_metrics.csv": "src/phase6_phase_clf.py", "results/phase6b_confusion.csv": "src/phase6_phase_clf.py", "results/phase6b_perm.csv": "src/phase6_phase_clf.py"}
ENTRIES, TABLES = [], []; _state = {"section": ""}; _cache = {}
def section(name): _state["section"] = name
def df(file):
    if file not in _cache: _cache[file] = pd.read_csv(ROOT / file)
    return _cache[file]
def _mask(d, where):
    m = np.ones(len(d), bool)
    for c, v in where.items(): m &= np.isclose(d[c].astype(float), v) if isinstance(v, float) else (d[c] == v).values
    return m
def fmt(x, f):
    if isinstance(x, (str, np.str_)): return str(x)
    if x is None or (isinstance(x, float) and np.isnan(x)): return "not reached"
    if np.isinf(x): return "diverged"
    if f == "p": return "<0.001" if x < 0.001 else f"{x:.3f}"
    return f.format(x)
def log(kind, label, value, src, selection, col, script): ENTRIES.append(dict(section=_state["section"], kind=kind, label=label, value=value, source=src, selection=selection, column=col, script=script))
def val(file, where, col, f="{:.3f}", label=""):
    d = df(file); r = d[_mask(d, where)]; assert len(r) == 1, (file, where, len(r)); s = fmt(r[col].iloc[0], f)
    log("value", label, s, file, str(where), col, SCRIPT[file]); return s
def ci(file, where, col, f="{:+.3f}", lo=None, hi=None, label=""):
    d = df(file); r = d[_mask(d, where)]; assert len(r) == 1, (file, where, len(r)); lo = lo or col + "_lo"; hi = hi or col + "_hi"
    s = f"{fmt(r[col].iloc[0], f)} [{fmt(r[lo].iloc[0], f)}, {fmt(r[hi].iloc[0], f)}]"; log("value+CI", label, s, file, str(where), f"{col}, {lo}, {hi}", SCRIPT[file]); return s
def agg(file, query, col, how, f="{:.3f}", label=""):
    d = df(file); r = d.query(query) if query else d; assert len(r) > 0 or how == 'count', (file, query); x = {"count": len(r), "max": r[col].max() if col else None, "min": r[col].min() if col else None, "median": r[col].median() if col else None, "sum": r[col].sum() if col else None}[how]
    s = fmt(x, f); log(f"aggregate:{how}", label, s, file, query or "(all rows)", col or "-", SCRIPT[file]); return s
def const(file, pattern, label=""):
    m = re.search(pattern, (ROOT / file).read_text()); assert m, (file, pattern); x = m.group(1); log("constant in code", label, x, file, f"regex {pattern}", "-", file); return x
def calc(label, x, f, src, how):
    s = fmt(x, f); log("computed", label, s, src, how, "-", "src/build_report.py (computation from the source named)"); return s
def ref(key):
    from src.authors_reference import AUTH
    s = str(AUTH[key]); log("authors' reported", key, s, "task description / paper (reference only)", key, "-", "src/authors_reference.py"); return s
def table(name, desc, files, script): TABLES.append(dict(section=_state["section"], name=name, description=desc, files=files, script=script))
def provenance_md():
    out = ["# Numbers provenance", "", "Written by `src/build_report.py` at the same time as `docs/report.md` and `docs/viva_qa.md`: **every number is loaded through a helper that logs where it came from**, so this list cannot drift from the documents. Selections are `column == value` filters on the named file (exactly one row each unless the kind says aggregate). Re-running `./run_all.sh` regenerates the files, the documents and this list. Authors' numbers are constants in `src/authors_reference.py` (reference only, labelled in the report).", ""]
    e = pd.DataFrame(ENTRIES).drop_duplicates(["kind", "label", "value", "source", "selection", "column"]).map(lambda x: x.replace("|", "/") if isinstance(x, str) else x)
    out += [f"Number of logged individual values: {len(e)}; tables: {len(TABLES)}.", "", "## A. Individual numbers", ""]
    for sec, g in e.groupby("section", sort=False):
        out += [f"### {sec}", "", "| Quantity | Value shown | Source file | Selection | Column(s) | Produced by |", "|---|---|---|---|---|---|"]
        out += [f"| {r.label or '-'} | {r.value} | `{r.source}` | {r.selection} | {r.column} | `{r.script}` |" for r in g.itertuples()]; out.append("")
    out += ["## B. Tables (every cell is read from the listed files)", "", "| Section | Table | Content | Source file(s) | Produced by |", "|---|---|---|---|---|"]
    out += [f"| {t['section']} | {t['name']} | {t['description']} | {t['files']} | `{t['script']}` |" for t in TABLES]
    return "\n".join(out) + "\n"

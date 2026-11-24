"""Build results/phase5_tables.md: my estimates next to the AUTHORS' REPORTED values (reference only), Holm-adjusted phase contrasts, SDs of E2 within/between parts."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from statsmodels.stats.multitest import multipletests
from src.data import *
M = pd.read_csv("results/phase5_mixed.csv"); P = pd.read_csv("results/phase5_perm.csv"); H = pd.read_csv("results/phase5_hormones_by_phase_long.csv")
ph = M[(M.model == "phase") & M.term.isin(["ph1", "ph2", "ph2-ph1"])].copy()
ph["p_holm3"] = ph.groupby(["sample", "outcome"]).p.transform(lambda x: multipletests(x, method="holm")[1])
def g(s, o, mod, t, col="b_z"): return M[(M["sample"] == s) & (M.outcome == o) & (M.model == mod) & (M.term == t)].iloc[0]
pf = lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
def fmt(r, adj=None): return f"{r.b_z:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}], p={pf(r.p)}" + (f" (Holm {pf(adj)})" if adj is not None else "")
def phr(s, o, t): r = ph[(ph["sample"] == s) & (ph.outcome == o) & (ph.term == t)].iloc[0]; return fmt(r, r.p_holm3)
L = M[M["sample"] == "long"].iloc[0]; Po = M[M["sample"] == "pooled"].iloc[0]; hdr = f"longitudinal (n={L.n_rows} rows, {L.n_women} women)", f"pooled (n={Po.n_rows} rows, {Po.n_women} women)"
out = ["# Phase 5 comparison tables (written by src/phase5_tables.py)\n", "**Authors' reported values are from the task description, reference only; they are NOT my results.** My estimates: random-intercept mixed model (REML, statsmodels), outcomes z-scored within sample (the authors' R script also uses scale()), menses = reference phase, Wald tests; Holm over the 3 phase contrasts (the authors used Tukey). The like-for-like sample is the longitudinal one.\n",
       f"| Effect | Authors' reported (reference only) | Mine: {hdr[0]} | Mine: {hdr[1]} |", "|---|---|---|---|"]
rows = [("Positive affect: peri-ovulatory - menses", "b = 0.41", [phr(s, "pos_affect", "ph1") for s in ["long", "pooled"]]),
        ("Positive affect: luteal - menses", "b = 0.47", [phr(s, "pos_affect", "ph2") for s in ["long", "pooled"]]),
        ("Negative affect: peri-ov - menses (authors: lower, n.s.)", "n.s.", [phr(s, "neg_affect", "ph1") for s in ["long", "pooled"]]),
        ("Negative affect: luteal - menses", "n.s.", [phr(s, "neg_affect", "ph2") for s in ["long", "pooled"]]),
        ("DRSP psychological: peri-ov - menses", "b = -0.68 (DRSP premenstrual symptoms)", [phr(s, "drsp_psych", "ph1") for s in ["long", "pooled"]]),
        ("DRSP psychological: luteal - menses", "b = -0.52", [phr(s, "drsp_psych", "ph2") for s in ["long", "pooled"]]),
        ("DRSP total (14 items): peri-ov - menses", "b = -0.68", [phr(s, "drsp_total", "ph1") for s in ["long", "pooled"]]),
        ("DRSP total: luteal - menses", "b = -0.52", [phr(s, "drsp_total", "ph2") for s in ["long", "pooled"]]),
        ("State anxiety: peri-ov - menses", "b = -0.27, p = 0.07 (trend)", [phr(s, "stai", "ph1") for s in ["long", "pooled"]]),
        ("DRSP psychological ~ estradiol (z)", "b = -0.30, p < 0.001", [fmt(g(s, "drsp_psych", "E2*P", "z_estradiol")) for s in ["long", "pooled"]]),
        ("DRSP total ~ estradiol (z)", "b = -0.30, p < 0.001", [fmt(g(s, "drsp_total", "E2*P", "z_estradiol")) for s in ["long", "pooled"]]),
        ("DRSP psychological ~ progesterone (z)", "not related", [fmt(g(s, "drsp_psych", "E2*P", "z_progesterone")) for s in ["long", "pooled"]]),
        ("State anxiety ~ estradiol (z)", "b = -0.15, p = 0.06 (trend)", [fmt(g(s, "stai", "E2*P", "z_estradiol")) for s in ["long", "pooled"]]),
        ("Positive affect ~ estradiol / progesterone", "all p > 0.16", [f"E2 p={g(s,'pos_affect','E2*P','z_estradiol').p:.3f}, P p={g(s,'pos_affect','E2*P','z_progesterone').p:.3f}" for s in ["long", "pooled"]]),
        ("Negative affect ~ estradiol / progesterone", "all p > 0.16", [f"E2 p={g(s,'neg_affect','E2*P','z_estradiol').p:.3f}, P p={g(s,'neg_affect','E2*P','z_progesterone').p:.3f}" for s in ["long", "pooled"]]),
        ("Phase x PSST on DRSP (joint 2-df Wald p), psychological / total", "p = 0.001", [f"{g(s,'drsp_psych','phase x PSST','interaction joint (chi2 df=2)').p:.4f} / {g(s,'drsp_total','phase x PSST','interaction joint (chi2 df=2)').p:.4f}" for s in ["long", "pooled"]]),
        ("BDI -> positive affect (menses slope, phase x BDI model)", "b = -0.45", [fmt(g(s, "pos_affect", "phase x BDI", "z_BDI_RW")) for s in ["long", "pooled"]]),
        ("BDI -> negative affect", "b = 0.56", [fmt(g(s, "neg_affect", "phase x BDI", "z_BDI_RW")) for s in ["long", "pooled"]])]
for e, a, m in rows: out.append(f"| {e} | {a} | {m[0]} | {m[1]} |")
out += ["\n## Hormone means by phase, longitudinal sample (raw pg/ml; rows = women per phase)\n", "| Phase | n | E2 mean (authors ~) | E2 mine | P mean (authors ~) | P mine |", "|---|---|---|---|---|---|"]
for (i, r), ae, ap, nm in zip(H.iterrows(), [0.65, 1.10, 1.02], [34.7, 43.8, 155.6], ["menses", "peri-ovulatory", "luteal"]): out.append(f"| {nm} | {int(r.n)} | {ae} | {r.E2_mean:.2f} | {ap} | {r.P_mean:.2f} |")
out += ["\n## Estradiol -> DRSP: within-woman vs between-woman (z-scored E2; Mundlak split: within = E2 - woman mean, between = woman mean)\n", "| Sample | Outcome | Model | within-woman b [95% CI], p | between-woman b [95% CI], p | permutation p within / between |", "|---|---|---|---|---|---|"]
for s in ["long", "pooled"]:
    for o in ["drsp_psych", "drsp_total"]:
        for mod, tw, tb in [("E2 within/between", "zw_estradiol", "zb_estradiol"), ("E2+P within/between", "zw_estradiol", "zb_estradiol"), ("log1p E2 within/between", "zw_logE2", "zb_logE2")]:
            pw = P[(P["sample"] == s) & (P.outcome == o) & P.test.str.contains("E2 within")].perm_p.iloc[0]; pb = P[(P["sample"] == s) & (P.outcome == o) & P.test.str.contains("E2 between")].perm_p.iloc[0]
            r1, r2 = g(s, o, mod, tw), g(s, o, mod, tb); out.append(f"| {s} (n={int(r1.n_rows)} rows, {int(r1.n_women)} women) | {o} | {mod} | {fmt(r1)} | {fmt(r2)} | {pw:.4f} / {pb:.3f} |" if mod == "E2 within/between" else f"| {s} | {o} | {mod} | {fmt(r1)} | {fmt(r2)} | - |")
df = load_analysis()
for s in ["long", "pooled"]:
    d = subset(df, s); z = (d.estradiol - d.estradiol.mean()) / d.estradiol.std(ddof=1); b = z.groupby(d.VPN).transform("mean"); out.append(f"\nSD of the z-scored E2 parts, {s} (n={len(d)} rows): within-woman {np.std(z-b, ddof=1):.2f}, between-woman (woman mean) {b.std(ddof=1):.2f}.")
out += ["\n## Permutation tests (B as stated; p = (1+#extreme)/(B+1))\n", "| Sample | n rows / women | Outcome | Test | Statistic | permutation p | B |", "|---|---|---|---|---|---|---|"]
for r in P.itertuples(): out.append(f"| {r.sample} | {r.n_rows} / {r.n_women} | {r.outcome} | {r.test} | {r.stat:.3f} | {r.perm_p:.4f} | {r.B} |")
open("results/phase5_tables.md", "w").write("\n".join(out)); print("\n".join(out[:30])); print("\n".join(l for l in out if l.startswith("SD of") or l.startswith("\nSD of")))

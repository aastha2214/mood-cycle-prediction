"""Write results/phase4_tables.md from phase4_metrics.csv / phase4_deltas.csv (no hand-typed numbers)."""
import pandas as pd
M = pd.read_csv("results/phase4_metrics.csv"); D = pd.read_csv("results/phase4_deltas.csv"); out = ["# Phase 4 tables (written by src/phase4_tables.py)\n",
 "LOPO = leave-one-woman-out. CIs: 95% percentile bootstrap, resampling women (B=2000). `per-woman` columns: mean +/- SD over women of each woman's own MAE / RMSE (R2 has no per-woman value). R2 = 1-SSE/SST over pooled out-of-fold predictions; R2_oos = 1-SSE/SSE(global-mean baseline).\n"]
f = lambda r, k: f"{r[k]:.3f} [{r[k+'_lo']:.3f}, {r[k+'_hi']:.3f}]"
for (s, t), g in M.groupby(["sample", "target"]):
    r0 = g.iloc[0]; out.append(f"\n## {s} | {t} | n = {r0.n_rows} rows, {r0.n_women} women (test)\n")
    out.append("| model | set | MAE [CI] | RMSE [CI] | R2 [CI] | R2_oos [CI] | per-woman MAE | per-woman RMSE |\n|---|---|---|---|---|---|---|---|")
    for r in pd.concat([g[g.model == "baseline"]] + [g[(g.model == m) & (g.fset == fs)] for m in ["ridge", "rf", "gb"] for fs in ["A", "B", "C", "D2", "D"]]).itertuples():
        r = r._asdict(); out.append(f"| {r['model']} | {r['fset']} | {f(r,'MAE')} | {f(r,'RMSE')} | {f(r,'R2')} | {f(r,'R2_oos')} | {r['MAE_woman_mean']:.3f} +/- {r['MAE_woman_sd']:.3f} | {r['RMSE_woman_mean']:.3f} +/- {r['RMSE_woman_sd']:.3f} |")
out.append("\n## Paired RMSE differences (first minus second set; negative = first better), bootstrap over the same women\n\n| sample | target | model | comparison | n rows / women | delta RMSE [CI] | CI excludes 0 |\n|---|---|---|---|---|---|---|")
for r in D[D.metric == "RMSE"].itertuples(): out.append(f"| {r.sample} | {r.target} | {r.model} | {r.comparison} | {r.n_rows} / {r.n_women} | {r.delta:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] | {'yes' if r.excludes_0 else 'no'} |")
open("results/phase4_tables.md", "w").write("\n".join(out)); print("wrote results/phase4_tables.md", len(out), "lines")

"""All matplotlib figures of the demo, drawn with one explicit style (white figure and axes, dark text), so that they read the same on light and dark themes."""
import contextlib, textwrap, threading
import numpy as np
import matplotlib; matplotlib.use("Agg")
from matplotlib.figure import Figure
import app_core as C
import method_lab as M

BLUE, GREEN, ORANGE, GREY, INK, SURFACE, BAND = "#0072B2", "#009E73", "#D55E00", "#6E6E6E", "#1A1A1A", "#FFFFFF", "#EEEEEE"
SET_COLOURS = {"A": BLUE, "C": GREEN, "D": ORANGE}
_LOCK = threading.RLock()   # matplotlib settings are shared by all sessions of the server

def settings(presentation):
    size = 11.5 if presentation else 8.0
    return {"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "savefig.edgecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
            "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK, "xtick.labelcolor": INK, "ytick.labelcolor": INK, "legend.facecolor": SURFACE, "legend.edgecolor": SURFACE, "legend.labelcolor": INK,
            "grid.color": "#B0B0B0", "grid.alpha": 0.35, "axes.spines.top": False, "axes.spines.right": False, "font.size": size, "axes.titlesize": size, "axes.labelsize": size,
            "xtick.labelsize": size, "ytick.labelsize": size, "legend.fontsize": size, "figure.titlesize": size, "legend.frameon": False}

@contextlib.contextmanager
def style(presentation=False):
    """Build AND render a figure inside this block."""
    with _LOCK, matplotlib.rc_context(settings(presentation)): yield

def figure(width, height, name):
    fig = Figure(figsize=(width, height), layout="constrained"); fig.set_label(name); return fig

def wrapped(name, presentation, width=16):
    return textwrap.fill(name, width) if presentation else name

def interval_figure(art, v, model, presentation=False):
    """Prediction, interval and training mean for the three targets; the legend sits below the panels."""
    lvl = int(round(100 * art["level"])); fig = figure(6.2, 4.8 if presentation else 4.0, "intervals"); axes = fig.subplots(3, 1)
    for ax, t in zip(axes, art["targets"]):
        M_ = art["targets"][t]; pt, lo, hi = C.point_and_interval(art, t, v, model)
        ax.axvspan(M_["obs_min"], M_["obs_max"], color=BAND, zorder=0, label="range observed in the data"); ax.axvline(M_["ybar"], color=GREY, ls="--", lw=1.2, label="training mean")
        ax.errorbar([pt], [0], xerr=[[pt - lo], [hi - pt]], fmt="o", color=BLUE, capsize=4, lw=2, label=f"prediction and {lvl}% interval")
        ax.set_yticks([]); ax.set_title(C.TN[t], loc="left"); ax.grid(axis="x"); ax.spines["left"].set_visible(False)
        ax.set_xlim(min(M_["obs_min"], lo) - 0.05 * M_["obs_sd"], max(M_["obs_max"], hi) + 0.05 * M_["obs_sd"])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2 if presentation else 3); return fig

def contributions_figure(df, presentation=False):
    fig = figure(6.2, 3.3, "contributions"); ax = fig.subplots()
    ax.barh(df.feature[::-1], df.contribution[::-1], color=[BLUE if x >= 0 else ORANGE for x in df.contribution[::-1]]); ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("contribution to this prediction, in the target's units"); ax.grid(axis="x"); return fig

def sweep_figure(art, v, model, presentation=False):
    fig = figure(9, 5.2 if presentation else 4.4, "sweep"); axes = fig.subplots(2, 3)
    for j, t in enumerate(art["targets"]):
        for i, (which, name) in enumerate([("e2", "Estradiol (pg/ml)"), ("pg", "Progesterone (pg/ml)")]):
            ax = axes[i, j]; s = C.sweep(art, t, v, which, model); ax.fill_between(s.x, s.lo, s.hi, color=BLUE, alpha=0.15); ax.plot(s.x, s.pred, color=BLUE)
            ax.axhline(art["targets"][t]["ybar"], color=GREY, ls="--", lw=1); ax.axvline(v[which], color=ORANGE, lw=1); ax.set_xlabel(name); ax.grid(True)
            if i == 0: ax.set_title(wrapped(C.TN[t], presentation))
    return fig

def compare_figure(art, df, presentation=False):
    fig = figure(9, 3.0 if presentation else 2.4, "compare"); axes = fig.subplots(1, 3)
    for ax, r in zip(axes, df.itertuples()):
        ax.errorbar([0, 1], [r.a, r.b], yerr=[[r.a - r.a_lo, r.b - r.b_lo], [r.a_hi - r.a, r.b_hi - r.b]], fmt="o", color=BLUE, capsize=4); ax.axhline(art["targets"][r.target]["ybar"], color=GREY, ls="--", lw=1)
        ax.set_xticks([0, 1], ["A", "B"]); ax.set_xlim(-0.5, 1.5); ax.set_title(wrapped(r.name, presentation)); ax.grid(axis="y")
    return fig

def r2_figure(R, sample, presentation=False):
    """Bars = the point estimates of the table above it (same sample), with their bootstrap intervals and the value written on each bar."""
    values = C.r2_values(R, sample); fig = figure(7.5, 4.2 if presentation else 3.2, "r2_bars"); ax = fig.subplots(); width = 0.27
    for k, fs in enumerate(["A", "C", "D"]):
        q = values[values.fset == fs].set_index("target").loc[list(C.TN)]; x = np.arange(3) + (k - 1) * width
        ax.bar(x, q.R2_oos, width, yerr=[q.R2_oos - q.R2_oos_lo, q.R2_oos_hi - q.R2_oos], color=SET_COLOURS[fs], capsize=3, label=f"set {fs}", error_kw=dict(ecolor=INK, lw=1))
        for xi, value, high in zip(x, q.R2_oos, q.R2_oos_hi):
            ax.annotate(f"{value:+.3f}", (xi, max(high, 0.0)), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", rotation=90 if presentation else 0, fontsize=matplotlib.rcParams["font.size"] - 1)
    top = float(values.R2_oos_hi.max()); bottom = float(min(values.R2_oos_lo.min(), 0.0)); ax.set_ylim(bottom - 0.02, top * (1.45 if presentation else 1.2))
    first = values.iloc[0]; ax.set_title(f"{'Pooled sample' if sample == 'pooled' else 'Longitudinal women'}: {int(first.n_rows)} sessions, {int(first.n_women)} women", loc="left")
    ax.axhline(0, color=INK, lw=0.8); ax.set_xticks(range(3), [wrapped(C.TN[t], presentation) for t in C.TN]); ax.set_ylabel("R² vs training mean"); ax.grid(axis="y")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=3); return fig

def calibration_figure(R, presentation=False):
    """Empirical against nominal coverage; rows: within the study and across the two groups of women; columns: targets; colours: feature sets."""
    data = C.calibration_data(R); fig = figure(9, 6.4 if presentation else 5.4, "calibration"); axes = fig.subplots(2, 3, sharex=True, sharey=True)
    levels = sorted(data.level.unique()); lim = (min(levels[0], data.cov_lo.min()) - 0.02, 1.01)
    for i, (sample, name) in enumerate([("pooled", "Within the study"), ("cross2long", "Across groups")]):
        for j, t in enumerate(C.TN):
            ax = axes[i, j]; ax.plot([lim[0], lim[1]], [lim[0], lim[1]], color=GREY, ls="--", lw=1, label="perfect calibration")
            for k, fs in enumerate(["A", "C", "D"]):
                q = data[(data["sample"] == sample) & (data.target == t) & (data.fset == fs)]; x = q.level + (k - 1) * 0.006
                ax.errorbar(x, q.coverage, yerr=[q.coverage - q.cov_lo, q.cov_hi - q.coverage], color=SET_COLOURS[fs], marker="o", ms=4, lw=1.2, capsize=2, label=f"set {fs}")
            ax.set_xlim(levels[0] - 0.03, levels[-1] + 0.03); ax.set_ylim(*lim); ax.set_xticks(levels, [f"{100 * l:.0f}%" for l in levels]); ax.grid(True)
            if i == 0: ax.set_title(wrapped(C.TN[t], presentation))
            if i == 1: ax.set_xlabel("nominal coverage")
        women = int(data[data["sample"] == sample].n_women.iloc[0]); axes[i, 0].set_ylabel(f"{name}\n({women} {'women' if sample == 'pooled' else 'test women'})\nempirical coverage")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=4); return fig

def optimiser_figure(result, presentation=False):
    fig = figure(7.5, 3.8 if presentation else 3.2, "optimisers"); ax = fig.subplots()
    for key, name, colour, ls in [("gd", "gradient descent", BLUE, "-"), ("hb", "heavy-ball momentum", ORANGE, "-")]:
        gap = result[key]; step = max(1, len(gap) // 4000); idx = np.unique(np.r_[np.arange(0, len(gap), step), len(gap) - 1]); ax.semilogy(idx, gap[idx], color=colour, ls=ls, label=name)
    ax.axhline(result["tol"], color=GREY, ls="--", lw=1, label="target 1e-6"); ax.set_xlabel("iteration"); ax.set_ylabel("loss gap relative to the start"); ax.set_ylim(result["tol"] / 10, 3 * max(result["gd"].max(), result["hb"].max())); ax.grid(True)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=3); return fig

def shrinkage_figure(singular_values, presentation=False, n=M.N_ILLUSTRATIVE):
    lams = np.logspace(-4, 3, 300); factors = M.shrinkage(singular_values, lams, n); fig = figure(7.5, 3.8 if presentation else 3.0, "shrinkage"); ax = fig.subplots()
    for s, f, colour, ls in zip(singular_values, factors, [BLUE, GREEN, ORANGE], ["-", "--", "-."]):
        ax.semilogx(lams, f, color=colour, ls=ls, lw=2, label=f"s = {s:g}")
    ax.axhline(0.5, color=GREY, ls=":", lw=1); ax.set_xlabel("penalty λ"); ax.set_ylabel("factor s² / (s² + nλ)"); ax.set_ylim(-0.02, 1.02); ax.grid(True)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=3); return fig

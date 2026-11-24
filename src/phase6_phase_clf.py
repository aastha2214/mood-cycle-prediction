"""Phase 6b sanity check: predict cycle phase (0 menses, 1 peri-ovulatory, 2 luteal) from log1p estradiol + log1p progesterone (feature set B), leave-one-woman-out + nested CV."""
import sys, warnings; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from src.data import *; from src.cvclf import *; from src.plotstyle import *
warnings.filterwarnings("ignore"); df = load_analysis(); cols = FEATURE_SETS["B"]; NAMES = ["menses", "peri-ov", "luteal"]; rng = np.random.default_rng(SEED); B = 2000; MODELS = ["baseline", "logreg", "rf", "gb"]
def cm(y, p): return np.array([[((y == i) & (p == j)).sum() for j in range(3)] for i in range(3)])
def metrics(y, p): c = cm(y, p); rec = np.diag(c) / c.sum(1); return dict(bal_acc=rec.mean(), acc=np.trace(c) / c.sum(), **{f"recall_{n}": r for n, r in zip(NAMES, rec)})
res = []; P = {}
for which in ["pooled", "long", "cross2long"]:
    if which == "cross2long": tr, te = df[df["sample"] == "cross"], df[df["sample"] == "long"]
    else: tr = te = subset(df, which)
    y, g = te.Phase.values.astype(int), te.VPN.values
    for m in MODELS:
        if which == "cross2long": p, hp = predict_one_clf(m, tr[cols].values, tr.Phase.values.astype(int), tr.VPN.values, te[cols].values, SEED); hp = [hp] * len(y)
        else: p, hp = lopo_clf(te[cols].values, y, g, m)
        P[(which, m)] = (p, y, g, te.Session.values)
        women = np.unique(g); W = rng.multinomial(len(women), np.ones(len(women)) / len(women), size=B); idx = {w: np.where(g == w)[0] for w in women}
        wc = np.array([cm(y[idx[w]], p[idx[w]]) for w in women]); bs = []
        for b in range(B):
            c = np.tensordot(W[b], wc, axes=1); rec = np.diag(c) / np.maximum(c.sum(1), 1); bs.append(rec.mean())
        r = dict(sample=which, model=m, n_rows=len(y), n_women=len(women), n_train_rows=len(tr), **metrics(y, p)); r["bal_acc_lo"], r["bal_acc_hi"] = np.percentile(bs, [2.5, 97.5]); r["hp_mode"] = pd.Series(hp).mode().iloc[0]; res.append(r)
R = pd.DataFrame(res); R.to_csv("results/phase6b_metrics.csv", index=False)
cmrows = []
for (s, m), (p, y, g, ses) in P.items():
    c = cm(y, p); cmrows += [dict(sample=s, model=m, true=NAMES[i], pred=NAMES[j], count=int(c[i, j])) for i in range(3) for j in range(3)]
pd.DataFrame(cmrows).to_csv("results/phase6b_confusion.csv", index=False)
pd.DataFrame([dict(sample=s, model=m, VPN=g[i], Session=ses[i], y=y[i], pred=p[i]) for (s, m), (p, y, g, ses) in P.items() for i in range(len(y))]).to_csv("results/phase6b_oof.csv", index=False)
# label-permutation test for balanced accuracy (logistic regression, C fixed at 1 to keep it fast; phase labels shuffled within woman / among single-session women)
def perm_run(seed, d):
    r = np.random.default_rng(seed); y = d.Phase.values.astype(int).copy(); g = d.VPN.values
    for w in np.unique(g):
        ii = np.where(g == w)[0]; y[ii] = y[r.permutation(ii)]
    si = np.where(pd.Series(g).map(pd.Series(g).value_counts()).values == 1)[0]; y[si] = y[r.permutation(si)]
    return bacc(y, lopo_clf(d[cols].values, y, g, "logreg", n_jobs=1, fixed_hp=1)[0])
PT = []
for which in ["pooled", "long"]:
    d = subset(df, which); p0, _ = lopo_clf(d[cols].values, d.Phase.values.astype(int), d.VPN.values, "logreg", fixed_hp=1); obs = bacc(d.Phase.values.astype(int), p0)
    nul = np.array(Parallel(n_jobs=10)(delayed(perm_run)(SEED + k, d) for k in range(200))); PT.append(dict(sample=which, n_rows=len(d), n_women=d.VPN.nunique(), obs_bal_acc_C1=obs, null_mean=nul.mean(), null_max=nul.max(), perm_p=(1 + (nul >= obs).sum()) / 201, n_perm=200))
PT = pd.DataFrame(PT); PT.to_csv("results/phase6b_perm.csv", index=False)
pd.set_option("display.width", 200); print("PHASE FROM HORMONES (set B: log1p E2, log1p P). Chance: balanced accuracy = 1/3 = 0.333 for ANY classifier that ignores the features (majority-class baseline gives exactly 0.333)")
for s, g in R.groupby("sample"):
    cls = pd.Series(df[df["sample"].isin(["cross", "long"] if s == "pooled" else (["long"] if s != "cross2long" else ["long"]))].Phase.values).value_counts().sort_index()
    print(f"\n{s}: test n={g.n_rows.iloc[0]} rows, {g.n_women.iloc[0]} women (train rows {g.n_train_rows.iloc[0]}); class counts menses/peri-ov/luteal = {cls.values.tolist()}; majority-class accuracy = {cls.max()/cls.sum():.3f}")
    print(g[["model", "bal_acc", "bal_acc_lo", "bal_acc_hi", "acc", "recall_menses", "recall_peri-ov", "recall_luteal", "hp_mode"]].round(3).to_string(index=False))
print("\nPermutation test (logreg, C=1, 200 label shuffles): "); print(PT.round(4).to_string(index=False))
fig, ax = plt.subplots(1, 3, figsize=(12, 3.9))
for a, s in zip(ax, ["pooled", "long", "cross2long"]):
    p, y, g, _ = P[(s, "logreg")]; c = cm(y, p); rn = c / c.sum(1, keepdims=True); a.imshow(rn, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3): a.text(j, i, f"{c[i,j]}\n({rn[i,j]:.0%})", ha="center", va="center", color="white" if rn[i, j] > .6 else "#222", fontsize=9)
    a.set_xticks(range(3)); a.set_xticklabels(NAMES); a.set_yticks(range(3)); a.set_yticklabels(NAMES); a.set_xlabel("predicted"); a.set_ylabel("true"); a.grid(False)
    r = R[(R["sample"] == s) & (R.model == "logreg")].iloc[0]; a.set_title(f"{s}: n={r.n_rows} rows, {r.n_women} women\nbalanced acc {r.bal_acc:.2f} [{r.bal_acc_lo:.2f}, {r.bal_acc_hi:.2f}] (chance 0.33)", fontsize=9)
fig.savefig("results/fig_6b_confusion.png"); plt.close(fig)

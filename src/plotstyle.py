import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
C = {"GD": "#0072B2", "SGD": "#D55E00", "Momentum": "#009E73", "Adam": "#CC79A7", "ink": "#333333", "grey": "#888888"}
LS = {"GD": "-", "SGD": ":", "Momentum": "--", "Adam": "-."}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
                     "axes.titlesize": 11, "axes.labelsize": 10, "figure.dpi": 130, "savefig.bbox": "tight", "lines.linewidth": 1.8})

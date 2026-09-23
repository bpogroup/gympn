r"""Figure: the gain over PPO against K, with the headroom ceiling drawn.

Section 6's scaling claim is that K predicts WHETHER component-scoped credit
can help, not HOW MUCH, and that the non-monotonicity at eight copies is a
ceiling effect rather than weaker structure. That argument is currently a
five-row tabular plus a paragraph asking the reader to take the ceiling on
trust. Drawing the ceiling makes it checkable: the maximum gain any method
could post at each configuration is 1 - (PPO's own normalized score), and at
eight copies the measured gain sits essentially ON that bound.

Design choices, and why:
  * FORM. Five configurations, one measure with uncertainty, ordered by a
    continuous covariate -> points with CIs against K. Not a line: we do not
    fit a trend through five points, and the text says so.
  * The ceiling is a SECOND series (PPO's unclaimed headroom), coloured by the
    entity that defines it (PPO), not by rank.
  * One y-axis. Both series are in the same normalized-return units, which is
    the only reason they may share an axis at all.
  * Significance is shown as a mark and stated in the label, never by colour
    alone.
  * COLOR: Okabe-Ito, the assignment the other figures use. Validated
    (validate_palette.js, light): all six checks PASS.

Run: python gen_fig_k_axis.py
"""
import glob
import json
import os

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 120, "savefig.bbox": "tight",
    "font.family": "serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "legend.frameon": False, "legend.fontsize": 8,
})

C_METHOD = "#009E73"   # bluish green -- cgae-cf
C_PPO = "#D55E00"      # vermillion   -- PPO, and so the ceiling it leaves
INK = "#222222"
MUTED = "#666666"

OUT = "figures"
METRIC = "greedy_final"

SPECS = [
    ("single\ncomponent", 1.0, "suite_results_s1_ep40/cells",
     "s1_stoch_sequence__%s__s*.json", "cgae_cflow", "ppo_clip"),
    ("$N{=}2$", 2.0, "suite_results_n2_ep40/cells",
     "N2__%s__s*.json", "cgae_cflow", "ppo"),
    ("\n\nhard $N{=}4$", 3.1, "suite_results_ncopies_hard_n4_ep40/cells",
     "N4__%s__s*.json", "cgae_cflow", "ppo"),
    ("$N{=}4$", 3.3, "suite_results_n4_ep40/cells",
     "N4__%s__s*.json", "cgae_cflow", "ppo"),
    ("$N{=}8$", 6.3, "suite_results_n8_ep40/cells",
     "N8__%s__s*.json", "cgae_cflow", "ppo"),
    ("multi-site", 8.0, "suite_results_multisite_protocol/cells",
     "%s__s*.json", "cgae_cflow", "ppo"),
]


def norm(cell):
    b = cell["baselines"]
    return (cell[METRIC] - b["random_mean"]) / (b["heuristic_mean"] - b["random_mean"])


rows = []
for name, K, d, pat, m, p in SPECS:
    def get(meth):
        out = {}
        for f in glob.glob(os.path.join(d, pat % meth)):
            c = json.load(open(f))
            out[c["seed"]] = norm(c)
        return out
    A, B = get(m), get(p)
    ks = sorted(set(A) & set(B))
    diff = np.array([A[k] - B[k] for k in ks])
    ppo = float(np.mean([B[k] for k in ks]))
    tt = stats.ttest_rel([A[k] for k in ks], [B[k] for k in ks])
    ci = stats.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
    rows.append(dict(name=name, K=K, gain=float(diff.mean()), ci=float(ci),
                     ppo=ppo, ceiling=1.0 - ppo, p=float(tt.pvalue), n=len(ks)))
    print("%-14s K=%.1f n=%2d gain %+.3f +-%.3f  p=%.4f  ppo %.3f  ceiling %.3f"
          % (name, K, len(ks), diff.mean(), ci, tt.pvalue, ppo, 1 - ppo))

K = np.array([r["K"] for r in rows])
gain = np.array([r["gain"] for r in rows])
ci = np.array([r["ci"] for r in rows])
ceil = np.array([r["ceiling"] for r in rows])

fig, ax = plt.subplots(figsize=(6.6, 3.4))

o = np.argsort(K)
# The ceiling is a property of each configuration, not a curve in K: the two
# N=4 variants sit at nearly the same K with very different headroom, so it is
# drawn as a short bar per configuration with its own shaded headroom band.
HW = 0.16
for k, c in zip(K, ceil):
    ax.plot([k - HW, k + HW], [c, c], color=C_PPO, dashes=(5, 2), linewidth=1.4, zorder=2)
    ax.fill_between([k - HW, k + HW], [c, c], [1.05, 1.05], color=C_PPO, alpha=0.07, linewidth=0)
ax.axhline(0.0, color=MUTED, linewidth=0.8)

sig = np.array([r["p"] < 0.05 for r in rows])
ax.errorbar(K[sig], gain[sig], yerr=ci[sig], fmt="o", color=C_METHOD,
            ecolor=C_METHOD, elinewidth=1.4, capsize=3, markersize=7, zorder=5)
ax.errorbar(K[~sig], gain[~sig], yerr=ci[~sig], fmt="o", color=C_METHOD,
            ecolor=C_METHOD, elinewidth=1.4, capsize=3, markersize=7,
            markerfacecolor="white", markeredgewidth=1.6, zorder=5)

# Environment identity goes on the axis, not into the plot area, so nothing
# collides with the ceiling line.
ax.set_xticks(list(K[o]))
NL = chr(10)
# names keep their own line breaks; the two low-K ticks are close together, so
# "single component" wraps rather than colliding with the N=2 label
ax.set_xticklabels([rows[i]["name"] + NL + ("%.1f" % rows[i]["K"])
                    for i in o], fontsize=7.5)

ax.set_xlabel("$K$  (reward-bearing causal components, measured before training)")
ax.set_ylabel("normalized gain over PPO")
ax.set_xlim(0.4, 8.9)
ax.set_ylim(-0.14, 1.32)
ax.legend(handles=[
    plt.Line2D([0], [0], color=C_METHOD, marker="o", lw=1.4, markersize=6,
               label="cgae-cf $-$ PPO, paired, 95% CI (20 seeds); filled $p<0.05$"),
    plt.Line2D([0], [0], color=C_METHOD, marker="o", lw=0, markersize=6,
               markerfacecolor="white", markeredgewidth=1.6, label="not significant"),
    plt.Line2D([0], [0], color=C_PPO, lw=1.4, dashes=(5, 2),
               label="ceiling: headroom PPO leaves, $1-$PPO (per configuration)"),
], loc="upper left", handlelength=2.2)

os.makedirs(OUT, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, "fig_k_axis.%s" % ext))
print("wrote %s/fig_k_axis.pdf" % OUT)

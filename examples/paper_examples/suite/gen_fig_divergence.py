r"""Figure: the unnormalized coefficient DIVERGES, and it is not an exploration
failure.

Section 6 calls this the sharpest single result and reports it in prose: on the
single-component environment one `cgae_flow` seed climbs to 10.3, holds for
several evaluation points, then falls to 0.15 while entropy stays flat. A mean
with a confidence band cannot show that -- it is a per-seed failure mode, not a
shift in the average. This draws every seed.

Design choices, and why:
  * FORM. Change-over-time, per seed -> line chart. All 20 seeds drawn (a
    "spaghetti" plot) because the CLAIM is about individual runs; the method's
    band is drawn behind for reference.
  * NOT A DUAL AXIS. Return and entropy are different measures on different
    scales, so they get two stacked panels sharing one x-axis, never two y-axes
    on one plot. The shared x is what lets the reader read down from the
    collapse to the entropy at the same epoch.
  * COLOR by identity, Okabe-Ito, the same assignment gen_fig_curves.py uses.
    Validated (validate_palette.js, light): all checks pass; the contrast WARN
    on #56B4E9 is relieved by the legend and the direct annotation.
  * The collapsing seed is emphasised by WEIGHT, not by a new hue -- colour
    follows the entity (cgae-f), never the rank of one run within it.
  * PRINT/CVD. The method's band is also dashed at its mean, so the two series
    separate in greyscale.

Run: python gen_fig_divergence.py
"""
import glob
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 120, "savefig.bbox": "tight",
    "font.family": "serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "legend.frameon": False, "legend.fontsize": 8,
    "lines.linewidth": 1.6,
})

C_FLOW = "#56B4E9"   # sky blue     -- cgae_flow
C_CFLOW = "#009E73"  # bluish green -- cgae_cflow, the method
INK = "#222222"
MUTED = "#666666"

CELLS = "suite_results_s1_ep40/cells"
ENV = "s1_stoch_sequence"
OUT = "figures"
FOCUS = 1            # the diverging seed: best 10.30, final 0.15, entropy 0.184


def load(method):
    out = []
    for f in sorted(glob.glob(os.path.join(CELLS, "%s__%s__s*.json" % (ENV, method)))):
        d = json.load(open(f))
        out.append(d)
    return out


flow = load("cgae_flow")
cflow = load("cgae_cflow")
base = flow[0]["baselines"]
rand, heur = base["random_mean"], base["heuristic_mean"]

fig, (ax, axe) = plt.subplots(
    2, 1, figsize=(7.0, 5.0), sharex=True,
    gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.30})

# ---- panel (a): greedy return, every seed ------------------------------- #
ep = np.asarray(flow[0]["greedy_epochs"], dtype=float)
XMAX = ep[-1] + 6.0

cf = np.vstack([d["greedy_curve"] for d in cflow])
m, sd = cf.mean(0), cf.std(0)
ax.fill_between(ep, m - sd, m + sd, color=C_CFLOW, alpha=0.18, linewidth=0)
ax.plot(ep, m, color=C_CFLOW, dashes=(4, 2), linewidth=1.8, zorder=4)

for d in flow:
    if d["seed"] == FOCUS:
        continue
    ax.plot(ep, d["greedy_curve"], color=C_FLOW, alpha=0.35, linewidth=0.8)

focus = [d for d in flow if d["seed"] == FOCUS][0]
ax.plot(ep, focus["greedy_curve"], color=C_FLOW, linewidth=2.4, zorder=5)

for y, name in ((heur, "heuristic"), (rand, "random")):
    ax.axhline(y, color=MUTED, linewidth=0.8, dashes=(1, 2))
    ax.text(ep[-1] + 0.8, y, name, color=MUTED, fontsize=7.5, va="center")

ax.text(ep[-1] + 0.8, focus["greedy_curve"][-1],
        "seed %d\n" % FOCUS + r"$10.3\rightarrow$" + "%.2f" % focus["greedy_final"],
        color=C_FLOW, fontsize=7.5, va="center", fontweight="bold")

ax.set_ylabel("greedy return")
ax.set_xlim(ep[0], XMAX)
ax.set_title("(a) greedy return, every seed", fontsize=9, loc="left")
ax.legend(handles=[
    plt.Line2D([0], [0], color=C_CFLOW, lw=1.8, dashes=(4, 2),
               label=r"cgae-cf, mean $\pm$ 1 SD"),
    plt.Line2D([0], [0], color=C_FLOW, lw=0.8, alpha=0.6,
               label="cgae-f, each of 20 seeds"),
    plt.Line2D([0], [0], color=C_FLOW, lw=2.4, label="cgae-f, seed %d" % FOCUS),
], loc="lower left", fontsize=7.5, handlelength=2.0, borderpad=0.2)

# ---- panel (b): entropy, same seeds, same x ----------------------------- #
ent_ep = np.linspace(ep[0], ep[-1], len(flow[0]["entropy_curve"]))
for d in flow:
    if d["seed"] == FOCUS:
        continue
    axe.plot(ent_ep, d["entropy_curve"], color=C_FLOW, alpha=0.35, linewidth=0.8)
axe.plot(ent_ep, focus["entropy_curve"], color=C_FLOW, linewidth=2.4, zorder=5)

axe.text(ent_ep[-1] + 0.8, focus["entropy_curve"][-1],
         "flat at %.2f" % focus["entropy_final"],
         color=C_FLOW, fontsize=7.5, va="center", fontweight="bold")

axe.set_ylabel("policy entropy")
axe.set_xlabel("epoch")
axe.set_ylim(0, 1.05)
axe.set_xlim(ep[0], XMAX)
# tick only where there is data; the right margin exists for the end labels
axe.set_xticks(np.arange(5, ep[-1] + 1, 5))
axe.set_title("(b) policy entropy, the same runs", fontsize=9, loc="left")

os.makedirs(OUT, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, "fig_divergence.%s" % ext))

collapsed = sum(1 for d in flow
                if (d["greedy_final"] - rand) / (heur - rand) <= 0.25)
print("cgae_flow  seeds=%d  collapsed(<=0.25 norm)=%d" % (len(flow), collapsed))
print("focus seed %d: best %.2f  final %.2f  entropy_final %.3f"
      % (FOCUS, focus["greedy_best"], focus["greedy_final"], focus["entropy_final"]))
print("wrote %s/fig_divergence.pdf" % OUT)

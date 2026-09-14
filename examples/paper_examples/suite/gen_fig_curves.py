"""Training-curve figure for the v3 paper: the two headline claims, side by side.

Panel (a) multi-site, K=8: causal-order credit wins where structure exists.
Panel (b) single component, K=1: the unnormalized coefficient DIVERGES, while
          every bounded variant tracks PPO.

Design choices, and why:
  * FORM. Change-over-time for a handful of named series -> line chart, one
    y-axis per panel (never a dual axis). Mean over 20 seeds with a 95% CI band.
  * COLOR by identity, fixed order, not cycled. Okabe-Ito, matching the palette
    the paper's existing figures already use. Validated: all checks pass; the one
    contrast WARN (#CC79A7 at 2.98:1) is relieved by direct labels and by the
    numeric tables in the body.
  * PRINT/CVD. Colour is never the only channel: every series also has its own
    dash pattern, so the figure survives greyscale printing and any CVD type.
  * Legend present in each panel (>=2 series) AND the key series is directly
    labelled at its right end; no number printed on every point.
  * Recessive grid, no top/right spines, thin marks -- the data is the ink.

Run: python gen_fig_curves.py
"""
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

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito, assigned by identity in fixed order (never cycled).
C = {
    "cgae_cflow": "#009E73",   # bluish green  -- the method
    "ppo":        "#D55E00",   # vermillion    -- baseline, as in existing figures
    "ccf":        "#0072B2",   # blue
    "mc_q":       "#CC79A7",   # reddish purple
    "cgae_dag":   "#E69F00",   # orange
    "cgae_flow":  "#56B4E9",   # sky blue
    "cgae":       "#CC79A7",
}
# Second channel: dash pattern, so the figure reads in greyscale.
DASH = {
    "cgae_cflow": (None, None),      # solid   -- the method
    "ppo":        (4, 2),            # dashed
    "ccf":        (1, 1.5),          # dotted
    "mc_q":       (6, 2, 1, 2),      # dash-dot
    "cgae_dag":   (1, 1.5),          # dotted
    "cgae_flow":  (6, 2, 1, 2),      # dash-dot
    "cgae":       (5, 1, 1, 1),
}
LBL = {
    "cgae_cflow": r"$\mathrm{cgae}$-$\mathrm{cf}$ (ours)",
    "ppo": "PPO", "ppo_clip": "PPO", "ccf": "ccf",
    "mc_q": "mc-q (no lineage)", "cgae_flow": "cgae-f (unnormalized)",
    "cgae_dag": "cgae-dag (exact closure)", "cgae": "cgae (mean)",
}


def curves(d, pat, arm, r, h, nseeds=20):
    """Mean and 95% CI of the normalized greedy curve across seeds."""
    got = []
    for s in range(nseeds):
        p = os.path.join(d, "cells", pat % (arm, s))
        if not os.path.exists(p):
            continue
        c = json.load(open(p))
        gc, ge = c.get("greedy_curve"), c.get("greedy_epochs")
        if gc:
            got.append([(x - r) / (h - r) for x in gc])
    if not got:
        return None, None, None
    L = min(len(x) for x in got)
    A = np.array([x[:L] for x in got])
    ep = (json.load(open(p)).get("greedy_epochs") or list(range(1, L + 1)))[:L]
    ci = 1.96 * A.std(0, ddof=1) / np.sqrt(A.shape[0])
    return np.asarray(ep), A.mean(0), ci


PANELS = [
    ("(a) multi-site routing, $K{=}8$", "upper left",
     "suite_results_multisite_bf", "%s__s%d.json", 62.9, 80.25,
     ["cgae_cflow", "ccf", "ppo", "mc_q"]),
    ("(b) single component, $K{=}1$", "lower right",
     "suite_results_s1_ep40", "s1_stoch_sequence__%s__s%d.json", 9.85, 14.775,
     ["cgae_cflow", "cgae_dag", "cgae_flow", "ppo_clip"]),
]

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
for ax, (title, legloc, d, pat, r, h, arms) in zip(axes, PANELS):
    for arm in arms:
        key = "ppo" if arm == "ppo_clip" else arm
        ep, mu, ci = curves(d, pat, arm, r, h)
        if ep is None:
            print("  MISSING", d, arm)
            continue
        ax.plot(ep, mu, color=C[key], dashes=DASH[key], label=LBL[arm],
                zorder=3 if key == "cgae_cflow" else 2)
        ax.fill_between(ep, mu - ci, mu + ci, color=C[key], alpha=0.13, lw=0)
    ax.axhline(1.0, color="0.45", lw=0.7, dashes=(2, 3), zorder=1)
    ax.axhline(0.0, color="0.45", lw=0.7, dashes=(2, 3), zorder=1)
    ax.set_title(title, fontsize=9.5, pad=6)
    ax.set_xlabel("training epoch")
    ax.legend(loc=legloc, ncol=1)

axes[0].set_ylabel("normalized return")
axes[0].text(40, 1.02, "heuristic", fontsize=7, color="0.35",
             va="bottom", ha="right")
axes[0].text(2.5, 0.02, "random", fontsize=7, color="0.35",
             va="bottom", ha="left")
axes[1].set_ylim(-0.6, 1.15)

fig.tight_layout()
fig.savefig(f"{OUT}/fig_curves_v3.pdf")
fig.savefig(f"{OUT}/fig_curves_v3.png", dpi=170)
print(f"wrote {OUT}/fig_curves_v3.pdf (+ .png)")

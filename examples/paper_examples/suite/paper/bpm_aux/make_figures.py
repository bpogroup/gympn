"""Figures of the BPM paper, generated from the committed result cells.

Run from anywhere:  python make_figures.py
Writes vector PDFs to ../bpm/figures/ (the Overleaf-synced paper). Colours follow the entity in every figure:
PPO blue, NF-GAE orange, per-process learners aqua (validated categorical
slots 1-3); heuristics are neutral grey dashed reference lines. Series also
differ by line style and carry direct labels, so the figures survive
greyscale printing.
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "..", "results"))
OUT = os.path.join(HERE, "..", "bpm", "figures")   # the Overleaf-synced paper directory
os.makedirs(OUT, exist_ok=True)

PPO, NFGAE, UNIT = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TEXT_W = 4.8                        # LNCS text width, inches (12.2 cm)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "lines.linewidth": 1.6,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "mathtext.fontset": "cm",
})


# ------------------------------------------------------------------ data
def curves(d, method):
    """{seed: normalised greedy curve} for one results directory and method."""
    d = os.path.join(RESULTS, d)
    b = json.load(open(glob.glob(os.path.join(d, "baselines*.json"))[0]))
    r, h = b["random_mean"], b["heuristic_mean"]
    return {int(f.split("__s")[1][:-5]): np.array([(v - r) / (h - r) for v in json.load(open(f))["greedy_curve"]])
            for f in glob.glob(os.path.join(d, "cells", f"*__{method}__s*.json"))}


def raw_curves(d, method):
    d = os.path.join(RESULTS, d)
    return {int(f.split("__s")[1][:-5]): np.array(json.load(open(f))["greedy_curve"])
            for f in glob.glob(os.path.join(d, "cells", f"*__{method}__s*.json"))}


def anchors(d):
    b = json.load(open(glob.glob(os.path.join(RESULTS, d, "baselines*.json"))[0]))
    return b["random_mean"], b["heuristic_mean"]


def per_process(regions):
    """Organisation of separately trained per-process PPO learners: the expected
    total of independent processes is the sum of their returns (seeds paired as
    in the paper: one seed per process at 1 region, two at 2 regions)."""
    U = {p: raw_curves(f"bpm_insurer_{p}_n1_ep40_nopp_aepn_flat", "ppo")
         for p in ("claims", "underwriting", "complaints")}
    r, h = anchors(f"bpm_insurer_n{regions}_ep40_nopp_aepn_flat")
    if regions == 1:
        return {s: (sum(U[p][s] for p in U) - r) / (h - r) for s in range(10)}
    return {k: (sum(U[p][2 * k] + U[p][2 * k + 1] for p in U) - r) / (h - r) for k in range(5)}


def tcrit(n):
    """Two-sided 95% Student t critical value for n paired observations."""
    from scipy.stats import t
    return float(t.ppf(0.975, n - 1))


def mean_ci(C):
    A = np.array([C[s] for s in sorted(C)])
    m = A.mean(0)
    half = tcrit(len(A)) * A.std(0, ddof=1) / np.sqrt(len(A)) if len(A) > 1 else 0 * m
    return m, half


def ref_line(ax, y, label):
    ax.axhline(y, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.text(ax.get_xlim()[1], y, f" {label}", color=MUTED, fontsize=7.5, va="center", ha="left", clip_on=False)


def series(ax, x, C, color, label, ls="-", marker=None):
    m, half = mean_ci(C)
    ax.fill_between(x, m - half, m + half, color=color, alpha=0.15, lw=0, zorder=2)
    ax.plot(x, m, color=color, ls=ls, marker=marker, ms=3, label=label, zorder=3)
    return m


# ------------------------------------------------------- learning curves
AGENTS_F = "#0d6e4c"                  # component agents with a full batch: darker aqua


def probe_curves(pattern, w=5):
    """{seed: normalised greedy curve} for the allocation runs of alloc_long_probe,
    as a trailing moving average over w evaluations (single evaluations are noisy)."""
    out = {}
    for f in glob.glob(os.path.join(RESULTS, "alloc_long_probe", pattern, "cell.json")):
        c = json.load(open(f))
        b = c["baselines"]
        seed = int(os.path.basename(os.path.dirname(f)).split("_s")[1].split("_")[0])
        g = (np.array(c["greedy_curve"]) - b["random_mean"]) / (b["heuristic_mean"] - b["random_mean"])
        out[seed] = np.array([g[max(0, i - w + 1):i + 1].mean() for i in range(len(g))])
    return out


def panel(ax, arms, title, epochs):
    for C, color, label, ls in arms:
        x = 3 * (np.arange(len(next(iter(C.values())))) + 1)
        series(ax, x, C, color, label, ls=ls)
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(0, epochs)
    ax.set_title(title, color=INK, loc="left")


def fig_curves():
    """Insurer with 1 and 2 regions, start-of-shift allocation, slow mismatch."""
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_W, 4.3))
    for ax, regions in ((axes[0][0], 1), (axes[0][1], 2)):
        d = f"bpm_insurer_n{regions}_ep40_nopp_aepn_flat_fv"
        panel(ax, [(curves(d, "ppo"), PPO, "PPO", "-"),
                   (curves(d, "agents"), UNIT, "Comp. agents (S)", (0, (3, 2))),
                   (curves(d + "_abfull", "agents"), AGENTS_F, "Comp. agents (F)", (0, (1, 1.5))),
                   (curves(d, "nfgae"), NFGAE, "NF-GAE", "-")],
              f"Insurer, {regions} region{'s' if regions > 1 else ''} ($K={3 * regions}$)", 40)
        ax.set_ylim(0.2, 1.25)
    ax = axes[1][0]
    panel(ax, [(probe_curves("ppo_s*_ep150_beta0.05_fv"), PPO, "PPO", "-"),
               (probe_curves("agents_mc_s*_ep150_beta0.05_fv"), UNIT, "Comp. agents (S)", (0, (3, 2))),
               (probe_curves("agents_mc_s*_ep150_beta0.05_fv_abfull"), AGENTS_F, "Comp. agents (F)", (0, (1, 1.5))),
               (probe_curves("nfgae_mc_s*_ep150_beta0.05"), NFGAE, "NF-GAE, forward only", (0, (5, 1, 1, 1))),
               (probe_curves("nfgae_mc_s*_ep150_beta0.05_fv"), NFGAE, "NF-GAE", "-")],
          "Start-of-shift allocation", 150)
    ax.set_ylim(0.0, 1.25)
    ax.text(147, 0.62, "NF-GAE,\nforward only", color=NFGAE, fontsize=7, ha="right", va="top")
    ax = axes[1][1]
    d = lambda suf: f"bpm_insurer_slow_n1_ep40{suf}_aepn_flat_fv"
    panel(ax, [(curves(d(""), "ppo"), PPO, "PPO, global wait", "-"),
               (curves(d("_ppc"), "ppo"), PPO, "PPO, component wait", (0, (3, 2))),
               (curves(d("_ppc"), "nfgae"), NFGAE, "NF-GAE, component wait", "-")],
          "Slow mismatch, 1 region", 40)
    ax.set_ylim(0.0, 1.2)
    ax.legend(loc="lower right", frameon=False, fontsize=7, handlelength=2.2, labelspacing=0.25, borderaxespad=0.1)
    for a in axes[1]:
        a.set_xlabel("epoch")
    for a in axes[:, 0]:
        a.set_ylabel("normalized return")
    fig.tight_layout(w_pad=1.0, h_pad=1.2, rect=(0, 0.06, 1, 1))
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, handlelength=2.4, columnspacing=1.4,
               bbox_to_anchor=(0.5, -0.005))
    fig.savefig(os.path.join(OUT, "fig_curves.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_curves()
    print("figures written to", OUT)

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


# ------------------------------------------------------- Fig. 2: curves
def fig_curves():
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.25), sharey=True)
    for ax, regions, title in ((axes[0], 1, "1 region ($K=3$)"), (axes[1], 2, "2 regions ($K=6$)")):
        d = f"bpm_insurer_n{regions}_ep40_nopp_aepn_flat"
        P, N, U = curves(d, "ppo"), curves(d, "nfgae"), per_process(regions)
        x = 3 * (np.arange(len(next(iter(P.values())))) + 1)
        ax.set_xlim(0, 40)
        series(ax, x, P, PPO, "PPO")
        series(ax, x, U, UNIT, "Per-process", ls=(0, (3, 2)))
        series(ax, x, N, NFGAE, "NF-GAE")
        ax.set_ylim(0.2, 1.25)
        ref_line(ax, 1.0, "heuristic") if regions == 2 else ax.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.set_title(title, color=INK)
        ax.set_xlabel("epoch")
    axes[0].set_ylabel("normalised return")
    fig.tight_layout(w_pad=1.2, rect=(0, 0.1, 1, 1))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, handlelength=2.4, columnspacing=2.0,
               bbox_to_anchor=(0.5, -0.01))
    fig.savefig(os.path.join(OUT, "fig_curves.pdf"))
    plt.close(fig)


# ------------------------------------------------------- Fig. 3: gain vs K
def fig_gain_vs_k():
    cells = [(2, "bpm_insurer_shared_n1_ep40_nopp_aepn_flat"),
             (3, "bpm_insurer_n1_ep40_nopp_aepn_flat"),
             (6, "bpm_insurer_n2_ep40_nopp_aepn_flat")]
    rows = []
    for k, d in cells:
        P, N = curves(d, "ppo"), curves(d, "nfgae")
        s = sorted(set(P) & set(N))
        for name, f in (("final", lambda g: g[-1]), ("whole-run mean", lambda g: g.mean())):
            dd = np.array([f(N[i]) - f(P[i]) for i in s])
            rows.append((name, k, dd.mean(), tcrit(len(dd)) * dd.std(ddof=1) / np.sqrt(len(dd))))
    fig, ax = plt.subplots(figsize=(TEXT_W * 0.62, 2.2))
    ax.axhline(0, color=MUTED, lw=0.8)
    for name, color, ls, mk, dx in (("whole-run mean", NFGAE, "-", "o", -0.08), ("final", NFGAE, (0, (3, 2)), "s", 0.08)):
        R = [r for r in rows if r[0] == name]
        ks = np.array([r[1] for r in R]) + dx
        ax.errorbar(ks, [r[2] for r in R], yerr=[r[3] for r in R], color=color, ls=ls, marker=mk, ms=4,
                    capsize=2, lw=1.4, elinewidth=0.9, label=name)
        ax.text(ks[-1] + 0.25, R[-1][2], name, color=INK, fontsize=7.5, va="center")
    ax.plot([1], [0], marker="D", ms=4, color=MUTED, ls="none")
    ax.annotate("$K=1$: identical\nto PPO", (1, 0), xytext=(0.75, 0.24), fontsize=7.5, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=1, shrinkB=3))
    ax.set_xticks([1, 2, 3, 6])
    ax.set_xlim(0.6, 9.2)
    ax.set_xlabel("independent processes $K$")
    ax.set_ylabel("NF-GAE $-$ PPO")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_gain_vs_k.pdf"))
    plt.close(fig)


# ------------------------------------------------------- Fig. 4: postponement
def fig_postpone():
    arms = [("PPO", "none", "_nopp", "ppo", PPO, "o"),
            ("NF-GAE", "none", "_nopp", "nfgae", NFGAE, "o"),
            ("PPO", "global", "", "ppo", PPO, "^"),
            ("PPO", "component", "_ppc", "ppo", PPO, "s"),
            ("NF-GAE", "component", "_ppc", "nfgae", NFGAE, "s")]
    # work-conserving rate rule, normalised with the same anchors (1 = waiting heuristic)
    work_conserving = {1: 0.873, 2: 0.865}
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_W, 4.75), gridspec_kw={"width_ratios": [1, 1.25]})
    rng = np.random.default_rng(0)
    for row, regions in enumerate((1, 2)):
        ax1, ax2 = axes[row]
        d = lambda suf: f"bpm_insurer_slow_n{regions}_ep40{suf}_aepn_flat"
        for i, (meth, pp, suf, m, color, mk) in enumerate(arms):
            C = curves(d(suf), m)
            fin = np.array([C[s][-1] for s in sorted(C)])
            y = len(arms) - 1 - i
            ax1.scatter(fin, y + rng.uniform(-0.13, 0.13, len(fin)), s=16, color=color, alpha=0.6, lw=0,
                        marker=mk, zorder=3)
            ax1.plot([fin.mean()] * 2, [y - 0.3, y + 0.3], color=INK, lw=1.4, zorder=4)
        ax1.set_yticks(range(len(arms)))
        ax1.set_yticklabels([f"{a[0]}, {a[1]}" for a in arms][::-1])
        ax1.grid(axis="y", visible=False)
        ax1.set_xlim(0.42, 1.18)
        for xv, lab, ha, dx in ((1.0, "waiting\nheuristic", "left", 0.008),
                                (work_conserving[regions], "work-\nconserving", "right", -0.008)):
            ax1.axvline(xv, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
            ax1.text(xv + dx, len(arms) - 0.45, lab, color=MUTED, fontsize=7, ha=ha, va="bottom")
        ax1.set_ylim(-0.5, len(arms) + 0.15)
        ax1.set_xticks([0.5, 0.7, 0.9, 1.1])
        ax1.set_title(f"{regions} region{'s' if regions > 1 else ''} ($K={3 * regions}$): final return per seed",
                      color=INK, loc="left", x=-0.62)

        for meth, pp, suf, m, color, ls in (("NF-GAE", "component", "_ppc", "nfgae", NFGAE, "-"),
                                            ("NF-GAE", "none", "_nopp", "nfgae", NFGAE, (0, (3, 2))),
                                            ("PPO", "global", "", "ppo", PPO, "-"),
                                            ("PPO", "component", "_ppc", "ppo", PPO, (0, (3, 2)))):
            C = curves(d(suf), m)
            x = 3 * (np.arange(len(next(iter(C.values())))) + 1)
            series(ax2, x, C, color, f"{meth} ({pp})", ls=ls)
        ax2.set_xlim(0, 40)
        ax2.set_ylim(0.15, 1.12)
        ax2.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax2.set_ylabel("normalised return")
        ax2.set_title("training curves", color=INK, loc="left")
    axes[1][0].set_xlabel("final normalised return")
    axes[1][1].set_xlabel("epoch")
    fig.tight_layout(w_pad=1.0, h_pad=1.4, rect=(0, 0.075, 1, 1))
    h, l = axes[0][1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, handlelength=2.4, columnspacing=2.0,
               bbox_to_anchor=(0.5, -0.005))
    fig.savefig(os.path.join(OUT, "fig_postpone.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_curves()
    fig_gain_vs_k()
    fig_postpone()
    print("figures written to", OUT)

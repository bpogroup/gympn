"""Paper figure: convergence of ppo_clip / lrq / rudder on each of the 8 envs.

One 2x4 grid of small multiples. Each panel: mean deterministic (greedy-eval)
return over 10 seeds per method, +/- 1 std band, with the per-env random and
optimum baselines as reference lines. Raw returns (not normalized) so each
panel is self-anchored and the early LRQ postpone phase reads honestly instead
of exploding a shared normalized scale.

Usage: python plot_convergence_grid.py [results_dir] [out_png]
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("suite_results_paper")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("figures/convergence_grid.png")

ENVS = [
    ("a_sequence_joint", "a · sequence, joint"),
    ("b_sequence_disjoint", "b · sequence, disjoint"),
    ("c_parallel_joint", "c · parallel, joint"),
    ("d_parallel_disjoint", "d · parallel, disjoint"),
    ("e_loop_joint", "e · loop, joint"),
    ("f_loop_disjoint", "f · loop, disjoint"),
    ("g_exclusive_choice_joint", "g · exclusive choice, joint"),
    ("h_exclusive_choice_disjoint", "h · exclusive choice, disjoint"),
]
# Fixed categorical slot order (validated reference palette, light mode).
METHODS = [
    ("lrq", "LRQ", "#2a78d6"),       # slot 1 · blue
    ("ppo_clip", "PPO", "#1baf7a"),  # slot 2 · aqua
    ("rudder", "RUDDER", "#eda100"), # slot 3 · yellow (relief: direct labels)
    ("mc_q", "MC-Q (no lineage)", "#008300"),  # slot 4 · green
]
INK, INK_2, GRID = "#333333", "#666666", "#e4e4e0"


def load(env, method):
    curves, baselines = [], None
    for p in sorted((RESULTS / "cells").glob(f"{env}__{method}__*.json")):
        c = json.loads(p.read_text())
        curves.append(c["greedy_curve"])
        baselines = c["baselines"]
        epochs = c["greedy_epochs"]
    return np.array(curves, dtype=float), epochs, baselines


fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), dpi=200, sharex=True)
fig.patch.set_facecolor("white")

for ax, (env, title) in zip(axes.flat, ENVS):
    handles = []
    for method, label, color in METHODS:
        curves, epochs, bl = load(env, method)
        mean, std = curves.mean(axis=0), curves.std(axis=0)
        (h,) = ax.plot(epochs, mean, color=color, linewidth=2.25, label=label,
                       solid_capstyle="round", zorder=3)
        ax.fill_between(epochs, mean - std, mean + std, color=color,
                        alpha=0.13, linewidth=0, zorder=2)
        handles.append(h)

    ax.axhline(bl["heuristic_mean"], color=INK, linewidth=1.1,
               linestyle=(0, (5, 3)), zorder=1)
    ax.axhline(bl["random_mean"], color=INK_2, linewidth=1.0,
               linestyle=(0, (1, 2.5)), zorder=1)
    ax.text(30.4, bl["heuristic_mean"], "optimum", fontsize=8, color=INK,
            va="center", ha="left")
    ax.text(30.4, bl["random_mean"], "random", fontsize=8, color=INK_2,
            va="center", ha="left")

    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=8)
    ax.set_xlim(2, 30)
    ax.set_xticks([2, 10, 20, 30])
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)

for ax in axes[1]:
    ax.set_xlabel("epoch", fontsize=10, color=INK_2)
for ax in axes[:, 0]:
    ax.set_ylabel("greedy return", fontsize=10, color=INK_2)

# Direct labels in panel h, where the three curves separate widely
# (relief rule for the low-contrast yellow slot; identity never color-alone).
ax_h = axes[1, 3]
for label, color, (lx, ly) in (("LRQ", "#2a78d6", (12.5, 21.0)),
                               ("PPO", "#1baf7a", (3.0, 15.6)),
                               ("RUDDER", "#eda100", (17.5, 11.6)),
                               ("MC-Q", "#008300", (3.0, 21.0))):
    ax_h.annotate(label, xy=(lx, ly), fontsize=9, color=color, fontweight="bold")

fig.legend(handles=handles, labels=[m[1] for m in METHODS], loc="upper center",
           ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.005),
           labelcolor=INK)
fig.suptitle("Deterministic-evaluation convergence per environment "
             "(mean ± 1 std over 10 seeds)", fontsize=13, color=INK, y=1.045)

fig.tight_layout(rect=(0, 0, 1, 0.99))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT.resolve()}")

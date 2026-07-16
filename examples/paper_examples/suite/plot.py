"""Plot the causal-RL stability suite.

Produces, under <output_dir>/plots/:
  - per_env_curves.png : greedy-eval curve (mean ± std over seeds) per env, both
    methods, with Random and Heuristic(optimum) reference lines.
  - aggregate.png      : bar charts of normalized final score and greedy drift by
    method, aggregated across all env×seed.

Usage:
    python plot.py [output_dir]      # default: suite_results
"""
import os
import sys
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SuiteConfig  # noqa: E402

COLORS = {"ppo_clip": "#1f77b4", "causal_td0": "#ff7f0e"}
LABELS = {"ppo_clip": "PPO-Clip", "causal_td0": "PPO+Causal (TD0)"}


def _load(output_dir: Path):
    cells = [json.loads(p.read_text()) for p in sorted((output_dir / "cells").glob("*.json"))]
    return cells


def _norm(value, bl):
    rng = bl["heuristic_mean"] - bl["random_mean"]
    if value is None or abs(rng) < 1e-9:
        return None
    return (value - bl["random_mean"]) / rng


def _stack_curves(curves):
    """Stack ragged greedy curves into a (n, T) array padded with nan."""
    curves = [c for c in curves if c]
    if not curves:
        return None
    T = max(len(c) for c in curves)
    out = np.full((len(curves), T), np.nan)
    for i, c in enumerate(curves):
        out[i, :len(c)] = c
    return out


def plot_per_env(cells, envs, methods, out_dir: Path):
    n = len(envs)
    ncol = 2 if n > 1 else 1
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 3.2 * nrow), squeeze=False)

    by_em = defaultdict(list)
    bl_by_env = {}
    for c in cells:
        by_em[(c["env"], c["method"])].append(c)
        bl_by_env[c["env"]] = c["baselines"]

    for idx, env in enumerate(envs):
        ax = axes[idx // ncol][idx % ncol]
        bl = bl_by_env[env]
        for method in methods:
            cs = by_em.get((env, method), [])
            stack = _stack_curves([c["greedy_curve"] for c in cs])
            if stack is None:
                continue
            epochs = cs[0]["greedy_epochs"][:stack.shape[1]]
            mean = np.nanmean(stack, axis=0)
            std = np.nanstd(stack, axis=0)
            ax.plot(epochs, mean, color=COLORS.get(method), marker="o", ms=3,
                    label=f"{LABELS.get(method, method)} (n={len(cs)})")
            ax.fill_between(epochs, mean - std, mean + std, color=COLORS.get(method), alpha=0.15)
        ax.axhline(bl["heuristic_mean"], color="green", ls="--", lw=1, label="Heuristic (optimum)")
        ax.axhline(bl["random_mean"], color="grey", ls=":", lw=1, label="Random")
        ax.set_title(env, fontsize=10)
        ax.set_xlabel("epoch"); ax.set_ylabel("greedy return")
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7)

    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    fig.suptitle("Greedy-eval convergence per environment (mean ± std over seeds)", y=1.0)
    fig.tight_layout()
    p = out_dir / "per_env_curves.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_aggregate(cells, methods, out_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    nfinal = {m: [] for m in methods}
    drift = {m: [] for m in methods}
    for c in cells:
        nf = _norm(c["greedy_final"], c["baselines"])
        if nf is not None:
            nfinal[c["method"]].append(nf)
        if c["greedy_drift"] is not None:
            drift[c["method"]].append(c["greedy_drift"])

    xs = np.arange(len(methods))
    ax1.bar(xs, [np.mean(nfinal[m]) if nfinal[m] else 0 for m in methods],
            yerr=[np.std(nfinal[m]) if nfinal[m] else 0 for m in methods],
            color=[COLORS.get(m) for m in methods], capsize=4)
    ax1.axhline(1.0, color="green", ls="--", lw=1, label="optimum")
    ax1.axhline(0.0, color="grey", ls=":", lw=1, label="random")
    ax1.set_xticks(xs); ax1.set_xticklabels([LABELS.get(m, m) for m in methods])
    ax1.set_ylabel("normalized final score"); ax1.set_title("Performance (H2)\n0=random, 1=optimal")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3, axis="y")

    ax2.bar(xs, [np.mean(drift[m]) if drift[m] else 0 for m in methods],
            yerr=[np.std(drift[m]) if drift[m] else 0 for m in methods],
            color=[COLORS.get(m) for m in methods], capsize=4)
    ax2.set_xticks(xs); ax2.set_xticklabels([LABELS.get(m, m) for m in methods])
    ax2.set_ylabel("greedy drift (best − final)")
    ax2.set_title("Post-convergence drift (H1)\nlower = more stable")
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("Aggregate across all environments (mean ± std over env×seed)")
    fig.tight_layout()
    p = out_dir / "aggregate.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


def main(output_dir: Path):
    cells = _load(output_dir)
    if not cells:
        print(f"[plot] no cells in {output_dir/'cells'} — run run_suite.py first.")
        return
    envs = sorted({c["env"] for c in cells})
    methods = sorted({c["method"] for c in cells})
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    p1 = plot_per_env(cells, envs, methods, plots_dir)
    p2 = plot_aggregate(cells, methods, plots_dir)
    print(f"[plot] saved {p1}")
    print(f"[plot] saved {p2}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else SuiteConfig().output_dir
    main(out)
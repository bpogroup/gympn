"""Analyze the causal-RL stability suite.

Loads every cell JSON, normalizes against the per-env Random/Heuristic baselines
    normalized = (policy - random) / (heuristic - random)
(0 = random, 1 = optimal), and reports:

  - per-(env, method) aggregates: greedy final (raw + normalized), drift, entropy
  - per-method aggregates across all envs
  - paired tests (clip vs causal, paired by env+seed) for:
        H2 performance  -> normalized greedy_final   (higher = better)
        H1 stability    -> greedy_drift              (lower  = more stable)

Usage:
    python analyze.py [output_dir]      # default: suite_results
"""
import os
import sys
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np

# Windows consoles default to cp1252; force UTF-8 so ±, Δ, × print cleanly.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SuiteConfig  # noqa: E402

try:
    from scipy.stats import wilcoxon
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _load(output_dir: Path):
    cells = []
    for p in sorted((output_dir / "cells").glob("*.json")):
        cells.append(json.loads(p.read_text()))
    return cells


def _norm(value, baselines):
    rng = baselines["heuristic_mean"] - baselines["random_mean"]
    if value is None or abs(rng) < 1e-9:
        return None
    return (value - baselines["random_mean"]) / rng


def _fmt(x, nd=2):
    return "  n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return float(np.mean(xs)), float(np.std(xs))


CONV_THRESHOLD = 0.95   # fraction of (optimum - random) that counts as "converged"


def _conv_epoch(cell, thr=CONV_THRESHOLD):
    """First greedy-eval epoch at which the policy reaches >= thr of optimum.

    Returns None if the env is degenerate (no headroom) or the policy never
    reached the threshold within the run (censored). NOTE: resolution is
    test_freq epochs (greedy eval is not run every epoch)."""
    bl = cell["baselines"]
    rng = bl["heuristic_mean"] - bl["random_mean"]
    if abs(rng) < 1e-9:
        return None
    for g, e in zip(cell.get("greedy_curve", []), cell.get("greedy_epochs", [])):
        if (g - bl["random_mean"]) / rng >= thr:
            return e
    return None


def _auc(cell):
    """Mean normalized greedy score across the curve — a threshold-free
    sample-efficiency proxy (higher = converged earlier AND higher)."""
    bl = cell["baselines"]
    rng = bl["heuristic_mean"] - bl["random_mean"]
    curve = cell.get("greedy_curve", [])
    if abs(rng) < 1e-9 or not curve:
        return None
    return float(np.mean([(g - bl["random_mean"]) / rng for g in curve]))


def analyze(output_dir: Path):
    cells = _load(output_dir)
    if not cells:
        print(f"[analyze] no cells found in {output_dir/'cells'} — run run_suite.py first.")
        return

    methods = sorted({c["method"] for c in cells})
    envs = sorted({c["env"] for c in cells})

    # index: (env, method) -> list of cells ; (env, seed, method) -> cell
    by_em = defaultdict(list)
    by_esm = {}
    for c in cells:
        by_em[(c["env"], c["method"])].append(c)
        by_esm[(c["env"], c["seed"], c["method"])] = c

    # ---------------- per-env table ----------------
    print("\n=== Per-environment greedy results (mean over seeds) ===")
    header = f"{'env':<28} {'method':<12} {'final':>6} {'final*':>7} {'best*':>6} {'drift':>6} {'ent':>5} {'n':>3}"
    print(header)
    print("-" * len(header))
    for env in envs:
        for method in methods:
            cs = by_em.get((env, method), [])
            if not cs:
                continue
            bl = cs[0]["baselines"]
            fin_m, _ = _mean_std([c["greedy_final"] for c in cs])
            nfin_m, _ = _mean_std([_norm(c["greedy_final"], bl) for c in cs])
            nbest_m, _ = _mean_std([_norm(c["greedy_best"], bl) for c in cs])
            drift_m, _ = _mean_std([c["greedy_drift"] for c in cs])
            ent_m, _ = _mean_std([c["entropy_final"] for c in cs])
            print(f"{env:<28} {method:<12} {_fmt(fin_m):>6} {_fmt(nfin_m):>7} "
                  f"{_fmt(nbest_m):>6} {_fmt(drift_m):>6} {_fmt(ent_m):>5} {len(cs):>3}")
        print()

    # ---------------- per-method aggregate ----------------
    print("=== Aggregate across all environments (mean ± std over env×seed) ===")
    print(f"{'method':<12} {'norm_final':>16} {'norm_best':>14} {'drift':>14} {'ent_final':>14}")
    for method in methods:
        cs = [c for c in cells if c["method"] == method]
        nf = _mean_std([_norm(c["greedy_final"], c["baselines"]) for c in cs])
        nb = _mean_std([_norm(c["greedy_best"], c["baselines"]) for c in cs])
        dr = _mean_std([c["greedy_drift"] for c in cs])
        en = _mean_std([c["entropy_final"] for c in cs])
        print(f"{method:<12} {_fmt(nf[0]):>8} ± {_fmt(nf[1]):<5} {_fmt(nb[0]):>6} ± {_fmt(nb[1]):<5} "
              f"{_fmt(dr[0]):>6} ± {_fmt(dr[1]):<5} {_fmt(en[0]):>6} ± {_fmt(en[1]):<5}")

    # ---------------- convergence speed (sample efficiency) ----------------
    print(f"\n=== Convergence speed (first greedy epoch reaching ≥{int(CONV_THRESHOLD*100)}% of "
          f"optimum; resolution = test_freq epochs) ===")
    print(f"{'env':<28} {'method':<12} {'conv_ep':>8} {'reached':>9} {'auc':>6} {'n':>3}")
    print("-" * 70)
    for env in envs:
        for method in methods:
            cs = by_em.get((env, method), [])
            if not cs:
                continue
            convs = [_conv_epoch(c) for c in cs]
            reached = [c for c in convs if c is not None]
            mean_conv = float(np.mean(reached)) if reached else None
            auc_m, _ = _mean_std([_auc(c) for c in cs])
            degenerate = abs(cs[0]["baselines"]["heuristic_mean"]
                             - cs[0]["baselines"]["random_mean"]) < 1e-9
            reached_str = "n/a (no hr)" if degenerate else f"{len(reached)}/{len(cs)}"
            print(f"{env:<28} {method:<12} {_fmt(mean_conv, 1):>8} {reached_str:>9} "
                  f"{_fmt(auc_m):>6} {len(cs):>3}")
        print()

    # ------- paired tests: every method pair, paired by env+seed -------
    # Convergence speed first (the primary question for the LRQ/PPO/RUDDER
    # comparison), then performance and stability.
    from itertools import combinations

    def _report(name, pairs, la, lb, better="higher"):
        if not pairs:
            print(f"  {name}: no paired data")
            return
        a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
        diff = a - b
        # "win" = first method better. For higher-is-better that's diff>0;
        # for lower-is-better (drift, epochs) it's diff<0.
        if better == "higher":
            wins = int((diff > 0).sum()); losses = int((diff < 0).sum())
        else:
            wins = int((diff < 0).sum()); losses = int((diff > 0).sum())
        ties = int((diff == 0).sum())
        arrow = "higher=better" if better == "higher" else "lower=better"
        line = (f"  {name}: {la}={a.mean():.3f}  {lb}={b.mean():.3f}  "
                f"Δ({la}-{lb})={diff.mean():+.3f}  | {la} wins/losses/ties="
                f"{wins}/{losses}/{ties}  [{arrow}]")
        if _HAVE_SCIPY and len(diff) >= 6 and np.any(diff != 0):
            try:
                stat, p = wilcoxon(a, b)
                line += f"  Wilcoxon p={p:.4f}"
            except Exception:
                pass
        print(line)

    for ma, mb in combinations(methods, 2):
        print(f"\n=== Paired comparison: {ma} vs {mb} (paired by env+seed) ===")
        pairs_conv95, pairs_conv80, pairs_auc = [], [], []
        pairs_final, pairs_drift = [], []
        reach_a = reach_b = pair_n = 0
        for (env, seed, method), ca_cell in by_esm.items():
            if method != ma:
                continue
            cb_cell = by_esm.get((env, seed, mb))
            if cb_cell is None:
                continue
            pair_n += 1
            bl = ca_cell["baselines"]
            # convergence: pairs where BOTH reached; reach counted separately.
            for thr, coll in ((CONV_THRESHOLD, pairs_conv95), (0.8, pairs_conv80)):
                ea, eb = _conv_epoch(ca_cell, thr), _conv_epoch(cb_cell, thr)
                if ea is not None and eb is not None:
                    coll.append((ea, eb))
            if _conv_epoch(ca_cell) is not None:
                reach_a += 1
            if _conv_epoch(cb_cell) is not None:
                reach_b += 1
            xa, xb = _auc(ca_cell), _auc(cb_cell)
            if xa is not None and xb is not None:
                pairs_auc.append((xa, xb))
            fa, fb = _norm(ca_cell["greedy_final"], bl), _norm(cb_cell["greedy_final"], bl)
            if fa is not None and fb is not None:
                pairs_final.append((fa, fb))
            if ca_cell["greedy_drift"] is not None and cb_cell["greedy_drift"] is not None:
                pairs_drift.append((ca_cell["greedy_drift"], cb_cell["greedy_drift"]))

        print(f"  (n = {pair_n} env×seed pairs; reached {int(CONV_THRESHOLD*100)}%: "
              f"{ma} {reach_a}/{pair_n}, {mb} {reach_b}/{pair_n})")
        _report(f"CONVERGENCE epoch to {int(CONV_THRESHOLD*100)}% (both-reached n={len(pairs_conv95)})",
                pairs_conv95, ma, mb, better="lower")
        _report(f"CONVERGENCE epoch to 80% (both-reached n={len(pairs_conv80)})",
                pairs_conv80, ma, mb, better="lower")
        _report("sample-eff  (greedy AUC)", pairs_auc, ma, mb, better="higher")
        _report("performance (norm final)", pairs_final, ma, mb, better="higher")
        _report("stability   (greedy drift)", pairs_drift, ma, mb, better="lower")
    if not _HAVE_SCIPY:
        print("  (install scipy for Wilcoxon signed-rank p-values)")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else SuiteConfig().output_dir
    analyze(out)
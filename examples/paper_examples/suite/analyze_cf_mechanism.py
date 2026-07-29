"""X13 mechanism analysis: does the lineage restriction tighten the paired SE?

Reads a cf-family suite dir and reports, per method, both the outcome
(normalized final) and the FORK MECHANISM telemetry that the registered
prediction M1 is about: mean paired standard error, mean |gap|, the
gap/SE signal-to-noise ratio, gate-pass rate, and preferences harvested
per fork. Paired (by seed) Wilcoxon tests for the ablation contrasts.

Usage: python analyze_cf_mechanism.py [suite_dir]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Windows consoles default to cp1252; force UTF-8 so the delta prints.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    from scipy.stats import wilcoxon
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

ORDER = ["cfpk", "cfpn", "cfpl"]


def _mean_over_active(curve):
    """Mean over epochs where forks actually ran (ignore zero-padding)."""
    a = np.asarray(curve, dtype=float)
    a = a[np.isfinite(a)]
    a = a[a > 0]
    return float(a.mean()) if a.size else float("nan")


def main(d: Path):
    cells = [json.loads(p.read_text()) for p in sorted((d / "cells").glob("*.json"))]
    if not cells:
        print(f"no cells in {d/'cells'}")
        return

    by_m = defaultdict(dict)
    for c in cells:
        by_m[c["method"]][c["seed"]] = c

    def norm(c):
        bl = c["baselines"]
        rng = bl["heuristic_mean"] - bl["random_mean"]
        return (c["greedy_final"] - bl["random_mean"]) / rng

    methods = [m for m in ORDER if m in by_m] + [m for m in by_m if m not in ORDER]

    print(f"\n=== outcome + fork mechanism ({d.name}) ===")
    hdr = (f"{'method':<6}{'n':>3}{'norm_final':>12}{'best':>7}"
           f"{'|gap|':>8}{'SE':>8}{'gap/SE':>8}{'pass%':>7}"
           f"{'prefs/fork':>11}{'forks/ep':>9}")
    print(hdr)
    print("-" * len(hdr))
    stats = {}
    for m in methods:
        cs = [by_m[m][s] for s in sorted(by_m[m])]
        nf = [norm(c) for c in cs]
        nb = [(c["greedy_best"] - c["baselines"]["random_mean"]) /
              (c["baselines"]["heuristic_mean"] - c["baselines"]["random_mean"]) for c in cs]
        gap = [_mean_over_active(c.get("cf_gap_curve", [])) for c in cs]
        se = [_mean_over_active(c.get("cf_se_curve", [])) for c in cs]
        pr = [_mean_over_active(c.get("cf_pass_rate_curve", [])) for c in cs]
        forks = [_mean_over_active(c.get("cf_forks_curve", [])) for c in cs]
        prefs = [_mean_over_active(c.get("cf_prefs_curve", [])) for c in cs]
        ppf = [p / f if f and np.isfinite(f) else np.nan
               for p, f in zip(prefs, forks)]
        snr = [g / s if s and np.isfinite(s) else np.nan for g, s in zip(gap, se)]
        stats[m] = dict(nf=nf, gap=gap, se=se, pass_rate=pr, ppf=ppf, snr=snr)
        print(f"{m:<6}{len(cs):>3}{np.mean(nf):>9.3f}±{np.std(nf):<3.2f}"
              f"{np.mean(nb):>7.2f}{np.nanmean(gap):>8.3f}{np.nanmean(se):>8.4f}"
              f"{np.nanmean(snr):>8.2f}{100*np.nanmean(pr):>7.0f}"
              f"{np.nanmean(ppf):>11.2f}{np.nanmean(forks):>9.1f}")

    def paired(a, b, key, label, lower_better=False):
        sa, sb = stats[a], stats[b]
        xa, xb = np.asarray(sa[key], float), np.asarray(sb[key], float)
        ok = np.isfinite(xa) & np.isfinite(xb)
        xa, xb = xa[ok], xb[ok]
        if xa.size < 2:
            print(f"  {label}: insufficient paired data")
            return
        d = xa - xb
        wins = int((d < 0).sum() if lower_better else (d > 0).sum())
        losses = int((d > 0).sum() if lower_better else (d < 0).sum())
        line = (f"  {label}: {a}={xa.mean():.4f} {b}={xb.mean():.4f} "
                f"Δ={d.mean():+.4f} ({100*d.mean()/abs(xb.mean()):+.0f}%) "
                f"| {a} wins/losses={wins}/{losses}")
        if _HAVE_SCIPY and np.any(d != 0):
            try:
                line += f"  Wilcoxon p={wilcoxon(xa, xb)[1]:.4f}"
            except Exception:
                pass
        print(line)

    if "cfpl" in stats and "cfpn" in stats:
        print("\n=== M1 (PRIMARY): lineage effect, cfpl vs cfpn "
              "(clean ablation — both no-tail, raw vs lineage-restricted) ===")
        paired("cfpl", "cfpn", "se", "paired SE          (lower=better)", lower_better=True)
        paired("cfpl", "cfpn", "snr", "gap/SE ratio      (higher=better)")
        paired("cfpl", "cfpn", "pass_rate", "gate-pass rate    (higher=better)")
        paired("cfpl", "cfpn", "ppf", "prefs per fork    (higher=better)  [M2]")
        paired("cfpl", "cfpn", "nf", "norm final        (higher=better)  [P1]")

    if "cfpn" in stats and "cfpk" in stats:
        print("\n=== value-tail effect, cfpn vs cfpk (control contrast) ===")
        paired("cfpn", "cfpk", "se", "paired SE          (lower=better)", lower_better=True)
        paired("cfpn", "cfpk", "nf", "norm final        (higher=better)")

    if "cfpl" in stats and "cfpk" in stats:
        print("\n=== end-to-end, cfpl vs cfpk (both changes together) ===")
        paired("cfpl", "cfpk", "se", "paired SE          (lower=better)", lower_better=True)
        paired("cfpl", "cfpk", "nf", "norm final        (higher=better)")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "suite_results_cfpl"))
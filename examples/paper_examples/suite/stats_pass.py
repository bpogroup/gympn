r"""Statistics pass over the converged (40-epoch, 20-seed) result cells.

Reads the cells the paper's Tables 3/4/6 are built from and, for every
environment, reports each arm's paired difference against cgae_cflow with:

  * n paired seeds, both means, the mean difference and its 95% t-interval
  * paired t-test p, exact Wilcoxon signed-rank p, wins/losses/ties
  * Cohen's d_z (paired effect size)
  * Holm-adjusted t p within the environment's family of comparisons
  * a second family: cgae_cflow vs PPO across the five environments

and a convergence check per (environment, arm): the slope of the mean
normalized greedy curve over the last three evaluation points, the mean
change from the previous three points to the last three, and a paired t-test
of the last point against the third-to-last across seeds.

Nothing is trained. Outputs go to stats_pass_out/ as Markdown and as a
booktabs LaTeX table ready to \input.

Run: python stats_pass.py
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
OUT = HERE / "stats_pass_out"
METHOD = "cgae_cflow"

# environment -> (result dir, cell filename pattern, baselines file or None)
ENVS = {
    "multi-site":   ("suite_results_multisite_protocol", "{m}__s{s}.json", "baselines.json"),
    "N=8":          ("suite_results_n8_ep40", "N8__{m}__s{s}.json", "baselines_N8.json"),
    "N=4":          ("suite_results_n4_ep40", "N4__{m}__s{s}.json", "baselines_N4.json"),
    "N=2":          ("suite_results_n2_ep40", "N2__{m}__s{s}.json", "baselines_N2.json"),
    "hard N=4":     ("suite_results_ncopies_hard_n4_ep40", "N4__{m}__s{s}.json", "baselines_N4.json"),
    "single comp.": ("suite_results_s1_ep40", "s1_stoch_sequence__{m}__s{s}.json", None),
}
ARMS = ["ppo", "ppo_clip", "mc_q", "ccf", "cfgae", "cgae", "cgae_cap", "cgae_flow", "cgae_dag"]
PRETTY = {"ppo": "PPO", "ppo_clip": "PPO", "mc_q": r"\mcq", "ccf": r"\ccf", "cfgae": r"\cfgae",
          "cgae": r"\cgae", "cgae_cap": r"\ccap", "cgae_flow": r"\cflowraw", "cgae_dag": r"\cdag",
          "cgae_cflow": r"\cflow"}
PLAIN = {"ppo": "PPO", "ppo_clip": "PPO", "mc_q": "mc-q", "ccf": "ccf", "cfgae": "cfgae",
         "cgae": "cgae", "cgae_cap": "cgae-cap", "cgae_flow": "cgae-f", "cgae_dag": "cgae-dag",
         "cgae_cflow": "cgae-cf"}


def load_env(name):
    d, pat, bfile = ENVS[name]
    d = HERE / d
    base = json.loads((d / bfile).read_text()) if bfile else None
    cells = {}
    for m in ARMS + [METHOD]:
        for f in glob.glob(str(d / "cells" / pat.format(m=m, s="*"))):
            j = json.loads(open(f).read())
            if j.get("greedy_final") is None:
                continue
            b = base or j["baselines"]
            r, h = b["random_mean"], b["heuristic_mean"]
            norm = lambda v: (v - r) / (h - r)
            cells.setdefault(m, {})[int(j["seed"])] = {
                "final": norm(j["greedy_final"]),
                "curve": [norm(x) for x in j["greedy_curve"]],
                "epochs": j.get("greedy_epochs"),
            }
    return cells


def paired(a, b):
    """a, b: arrays aligned by seed. Returns stats of d = a - b."""
    d = a - b
    n = len(d)
    mean = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    t_p = stats.ttest_rel(a, b).pvalue
    nz = d[d != 0]
    if len(nz) == 0:
        w_p = 1.0
    else:
        try:
            w_p = stats.wilcoxon(d, zero_method="pratt", method="exact").pvalue
        except Exception:
            w_p = stats.wilcoxon(d, zero_method="pratt").pvalue
    return dict(n=n, mean=mean, lo=mean - tcrit * se, hi=mean + tcrit * se, t_p=t_p, w_p=w_p,
                wins=int((d > 0).sum()), losses=int((d < 0).sum()), ties=int((d == 0).sum()),
                dz=mean / d.std(ddof=1) if d.std(ddof=1) > 0 else float("inf"))


def holm(pvals):
    """Holm step-down adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        adj[idx] = running
    return adj


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def main():
    OUT.mkdir(exist_ok=True)
    md, tex = [], []
    md.append("# Statistics pass (converged 40-epoch cells)\n")
    md.append("Differences are `cgae-cf minus comparator` on normalized final greedy return, "
              "paired by seed. CI is the 95% t-interval; Wilcoxon is exact, zeros by Pratt; "
              "Holm is within the environment's family (all comparators of that row block).\n")
    hdr = ("| env | comparator | n | cgae-cf | comp. | diff | 95% CI | t p | Holm p | Wilcoxon p "
           "| W/L/T | d_z |")
    md.append(hdr)
    md.append("|" + "---|" * 12)

    tex.append(r"\begin{table}[tbp]")
    tex.append(r"\centering")
    tex.append(r"\caption{Paired comparisons against \cflow{} on normalized final return "
               r"($40$ epochs, seeds paired). Difference is \cflow{} minus comparator with its "
               r"$95\%$ $t$-interval; $p_t$ is the paired $t$-test, $p_H$ its Holm adjustment "
               r"within the environment's family of comparisons, $p_W$ the exact Wilcoxon "
               r"signed-rank test; W/L counts seeds on which \cflow{} scored higher/lower; "
               r"$d_z$ is the paired effect size.}")
    tex.append(r"\label{tab:paired_full}")
    tex.append(r"\small")
    tex.append(r"\begin{tabular}{llrrrrrrl}")
    tex.append(r"\toprule")
    tex.append(r"environment & comparator & $n$ & difference & $95\%$ CI & $p_t$ & $p_H$ & $p_W$ & W/L, $d_z$ \\")
    tex.append(r"\midrule")

    ppo_family = []  # (env, stats) for cflow vs PPO across envs
    conv_rows = []
    all_json = {}

    for env in ENVS:
        cells = load_env(env)
        if METHOD not in cells:
            print(f"[{env}] no {METHOD} cells", file=sys.stderr)
            continue
        ref = cells[METHOD]
        rows = []
        for m in ARMS:
            if m not in cells:
                continue
            seeds = sorted(set(ref) & set(cells[m]))
            if len(seeds) < 5:
                continue
            a = np.array([ref[s]["final"] for s in seeds])
            b = np.array([cells[m][s]["final"] for s in seeds])
            st = paired(a, b)
            st.update(ma=a.mean(), mb=b.mean(), m=m)
            rows.append(st)
            if m in ("ppo", "ppo_clip"):
                ppo_family.append((env, st))
        adj = holm([r["t_p"] for r in rows])
        for r, ph in zip(rows, adj):
            r["holm_p"] = ph
        rows.sort(key=lambda r: r["t_p"])
        all_json[env] = rows
        first = True
        for r in rows:
            ci = f"[{r['lo']:+.3f}, {r['hi']:+.3f}]"
            md.append(f"| {env} | {PLAIN[r['m']]} | {r['n']} | {r['ma']:.3f} | {r['mb']:.3f} | "
                      f"{r['mean']:+.3f} | {ci} | {r['t_p']:.4f}{stars(r['t_p'])} | "
                      f"{r['holm_p']:.4f}{stars(r['holm_p'])} | {r['w_p']:.4f}{stars(r['w_p'])} | "
                      f"{r['wins']}/{r['losses']}/{r['ties']} | {r['dz']:+.2f} |")
            fmt_p = lambda p: "$<0.0001$" if p < 1e-4 else f"${p:.4f}$"
            tex.append(f"{env if first else ''} & {PRETTY[r['m']]} & ${r['n']}$ & ${r['mean']:+.3f}$ & "
                       f"$[{r['lo']:+.3f},\\,{r['hi']:+.3f}]$ & {fmt_p(r['t_p'])} & {fmt_p(r['holm_p'])} & "
                       f"{fmt_p(r['w_p'])} & ${r['wins']}$/${r['losses']}$, ${r['dz']:+.2f}$ \\\\")
            first = False
        tex.append(r"\midrule")

        # ---- convergence: mean curve slope over the last three eval points ----
        for m in sorted(cells):
            seeds = sorted(cells[m])
            L = min(len(cells[m][s]["curve"]) for s in seeds)
            if L < 6:
                continue
            arr = np.array([cells[m][s]["curve"][:L] for s in seeds])
            ep = cells[m][seeds[0]]["epochs"]
            ep = np.array(ep[:L]) if ep and len(ep) >= L else np.arange(1, L + 1)
            mc = arr.mean(0)
            slope = np.polyfit(ep[-3:], mc[-3:], 1)[0]           # per epoch
            step = float(mc[-3:].mean() - mc[-6:-3].mean())      # last 3 vs previous 3
            drift_p = stats.ttest_rel(arr[:, -1], arr[:, -3]).pvalue
            conv_rows.append((env, PLAIN.get(m, m), len(seeds), mc[-1], slope, step, drift_p))

    tex[-1] = r"\bottomrule"
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")

    # ---- second family: cflow vs PPO across environments ----
    md.append("\n## Family 2: cgae-cf vs PPO across the five environments (Holm over 5 tests)\n")
    md.append("| env | n | diff | 95% CI | t p | Holm p | Wilcoxon p | W/L/T |")
    md.append("|---|---|---|---|---|---|---|---|")
    adj = holm([s["t_p"] for _, s in ppo_family])
    for (env, s), ph in zip(ppo_family, adj):
        md.append(f"| {env} | {s['n']} | {s['mean']:+.3f} | [{s['lo']:+.3f}, {s['hi']:+.3f}] | "
                  f"{s['t_p']:.4f}{stars(s['t_p'])} | {ph:.4f}{stars(ph)} | {s['w_p']:.4f}{stars(s['w_p'])} "
                  f"| {s['wins']}/{s['losses']}/{s['ties']} |")

    # ---- convergence table ----
    md.append("\n## Convergence at the 40-epoch budget\n")
    md.append("Slope is per training epoch of the seed-mean normalized greedy curve over its last "
              "three evaluation points; step is mean(last 3) minus mean(previous 3); drift p is a "
              "paired t-test of the last point against the third-to-last across seeds.\n")
    md.append("| env | arm | n | final (mean) | slope/epoch | step | drift p |")
    md.append("|---|---|---|---|---|---|---|")
    for env, m, n, fin, slope, step, p in conv_rows:
        md.append(f"| {env} | {m} | {n} | {fin:.3f} | {slope:+.4f} | {step:+.3f} | {p:.3f} |")

    (OUT / "paired_full.md").write_text("\n".join(md), encoding="utf-8")
    (OUT / "tab_paired_full.tex").write_text("\n".join(tex), encoding="utf-8")
    (OUT / "paired_full.json").write_text(json.dumps(all_json, indent=1, default=float))
    print("\n".join(md))
    print(f"\nwrote {OUT/'paired_full.md'}, {OUT/'tab_paired_full.tex'}, {OUT/'paired_full.json'}")


if __name__ == "__main__":
    main()

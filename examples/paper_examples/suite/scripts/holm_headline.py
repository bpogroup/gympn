"""Holm-corrected p-values of the headline comparisons of the paper.

The comparisons are grouped by the three things the evaluation shows (one Holm
family each): NF-DCL against DCL (Welch over training seeds of the final
network on the held-out days); NF-PPO against PPO and against component agents
(Welch over seeds); and postponement against none (Welch over seeds for the
networks, paired over the held-out days for the lookahead). Comparisons whose
runs are missing are skipped and listed, so the script can be rerun as cells
land.

usage: python scripts/holm_headline.py [which=final|selected]
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

SUITE = Path(__file__).resolve().parents[1]
R = SUITE / "results"
opts = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
WHICH = opts.get("which", "final")


def pi_runs(pattern, keep=lambda f: True):
    """Normalized held-out score of every seed of a run_pi arm (one summary per seed)."""
    out = []
    for d in sorted(glob.glob(str(R / pattern))):
        for f in sorted(Path(d).glob("summary_tau0.25*.json")):
            if keep(f):
                out.append(json.loads(f.read_text())[f"heldout_norm_{WHICH}"])
    return np.array(out)


def bpm_runs(pattern, method):
    """Normalized held-out finals of every seed of a run_bpm method."""
    out = []
    for d in sorted(glob.glob(str(R / pattern))):
        s = json.loads((Path(d) / "summary.json").read_text())
        row = s["rows"].get(method)
        if row and row.get(f"heldout_{'finals' if WHICH == 'final' else 'bests'}"):
            out.extend(row[f"heldout_{'finals' if WHICH == 'final' else 'bests'}"])
    return np.array(out)


def planner_days(name, rule="sh32"):
    f = R / f"planner_heldout_{name}.jsonl"
    if not f.exists():
        return None
    rs = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    d = {r["day"]: r["ret"] for r in rs if r["rule"] == rule}
    return np.array([d[i] for i in sorted(d)]) if len(d) == 100 else None


def rounds_welch(r):
    """NF-DCL against DCL after round r of the ten-seed array (results/snellius_bpic15_rounds.json)."""
    d = json.loads((R / "snellius_bpic15_rounds.json").read_text())["rounds"][str(r)]
    return welch(np.array(d["nfdcl_final"]), np.array(d["dcl_final"]))


def welch(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def paired(a, b):
    if a is None or b is None:
        return None
    return float(stats.ttest_rel(a, b).pvalue)


plain = lambda f: not f.name.startswith("summary_tau0.25_rn")  # noqa: E731
BP = "pi_bpic15_n1_R8_d40_mc_fair_mrg_clkn_sh32"
SLOW = "pi_insurer_slow_n1_R8_d40_mc{pp}_fair_mrg_clkn_sh32"
ALLOC = "pi_insurer_alloc_n{n}_R8_d40_mc_fair_mrg_clkn_sh32"
INS = "pi_insurer_n{n}_R8_d40_mc_fair_mrg_clkn_sh32"
MF = {"alloc1": "bpm_insurer_alloc_n1_ep150*beta0.05*clkn_mrg", "alloc2": "bpm_insurer_alloc_n2_ep150*beta0.05*clkn_mrg",
      "slow_ppc": "bpm_insurer_slow_n1_ep150*beta0.05_ppc_*clkn_mrg", "slow_none": "bpm_insurer_slow_n1_ep150*beta0.05*_nopp*clkn_mrg",
      "slow_global": "bpm_insurer_slow_n1_ep150*beta0.05_aepn*clkn_mrg",
      "bpic15": "bpm_bpic15_n1_ep40_beta0.05_nopp_aepn_eps32_bs128_flat_mc_fv_clkn_mrg_abfull",   # 10 seeds of all three methods
      "ins1": "bpm_insurer_n1_ep*beta0.05*clkn_mrg", "ins2": "bpm_insurer_n2_ep*beta0.05*clkn_mrg"}

SR = json.loads((R / "snellius_rounds.json").read_text())["benchmarks"]      # one seed per collection, paper anchors


def rounds_family():
    out = [(f"municipalities, round {r}", (lambda r=r: rounds_welch(r))) for r in (1, 2, 3)]
    for tag, name in [("ins1", "insurer, 1 region"), ("ins2", "insurer, 2 regions"), ("alloc1", "allocation, 1 region"),
                      ("alloc2", "allocation, 2 regions"), ("slow", "slow mismatch")]:
        for r, d in SR.get(tag, {}).items():
            out.append((f"{name}, round {r}", (lambda d=d: welch(np.array(d["nfdcl_final"]), np.array(d["dcl_final"])))))
    return out


FAMILIES = {
    "NF-DCL against DCL after every round (Table 5)": rounds_family(),
    "NF-DCL against DCL": [
        ("municipalities, round 1 (10 collections)", lambda: rounds_welch(1)),
        ("slow mismatch, component postponement", lambda: welch(pi_runs(SLOW.format(pp="_ppc") + "_cf*_dyn"), pi_runs(SLOW.format(pp="_ppc"), plain))),
        ("slow mismatch, no postponement", lambda: welch(pi_runs(SLOW.format(pp="") + "_cf*_dyn"), pi_runs(SLOW.format(pp=""), plain))),
        ("allocation, 1 region", lambda: welch(pi_runs(ALLOC.format(n=1) + "_cf*_dyn"), pi_runs(ALLOC.format(n=1), plain))),
        ("allocation, 2 regions", lambda: welch(pi_runs(ALLOC.format(n=2) + "_cf*_dyn"), pi_runs(ALLOC.format(n=2), plain))),
        ("insurer, 1 region", lambda: welch(pi_runs(INS.format(n=1) + "_cf*"), pi_runs(INS.format(n=1), plain))),
        ("insurer, 2 regions", lambda: welch(pi_runs(INS.format(n=2) + "_cf*"), pi_runs(INS.format(n=2), plain))),
        ("municipalities, round 3 (10 collections)", lambda: rounds_welch(3)),
    ],
    "NF-PPO against PPO and against component agents": [
        (f"{k}: NF-PPO vs PPO", (lambda k=k: welch(bpm_runs(MF[k], "nfgae"), bpm_runs(MF[k], "ppo")))) for k in MF if k != "slow_global"
    ] + [
        (f"{k}: NF-PPO vs agents", (lambda k=k: welch(bpm_runs(MF[k], "nfgae"), bpm_runs(MF[k] if MF[k].endswith("abfull") else MF[k] + "*abfull", "agents")))) for k in MF if k != "slow_global"
    ],
    "postponement against none (slow mismatch)": [
        ("lookahead (paired over days)", lambda: paired(planner_days("insurer_slow_n1_ppc"), planner_days("insurer_slow_n1"))),
        ("DCL", lambda: welch(pi_runs(SLOW.format(pp="_ppc"), plain), pi_runs(SLOW.format(pp=""), plain))),
        ("NF-DCL", lambda: welch(pi_runs(SLOW.format(pp="_ppc") + "_cf*_dyn"), pi_runs(SLOW.format(pp="") + "_cf*_dyn"))),
        ("PPO, global", lambda: welch(bpm_runs(MF["slow_global"], "ppo"), bpm_runs(MF["slow_none"], "ppo"))),
        ("PPO, component-scoped", lambda: welch(bpm_runs(MF["slow_ppc"], "ppo"), bpm_runs(MF["slow_none"], "ppo"))),
        ("NF-PPO, component-scoped", lambda: welch(bpm_runs(MF["slow_ppc"], "nfgae"), bpm_runs(MF["slow_none"], "nfgae"))),
    ],
}


def holm(ps):
    """Holm step-down adjusted p-values (monotone), None entries skipped."""
    idx = [i for i, p in enumerate(ps) if p is not None]
    m = len(idx)
    order = sorted(idx, key=lambda i: ps[i])
    adj, running = [None] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adj[i] = running
    return adj


for fam, comps in FAMILIES.items():
    print(f"\n{fam}")
    names, ps = [], []
    for name, fn in comps:
        try:
            p = fn()
        except Exception as e:  # missing or malformed runs
            p = None
        names.append(name); ps.append(p)
    adj = holm(ps)
    for name, p, a in zip(names, ps, adj):
        if p is None:
            print(f"  {name:45s}  (missing)")
        else:
            print(f"  {name:45s}  p = {p:.4f}   Holm = {a:.4f}{'  *' if a < 0.05 else ''}")

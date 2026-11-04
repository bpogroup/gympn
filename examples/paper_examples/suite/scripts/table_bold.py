"""Which methods of the main table are indistinguishable from the best of their row.
Per row, the held-out final score of every seed of every method (run_bpm summaries
for PPO / NF-PPO / agents, run_pi summaries for DCL / NF-DCL); the lookahead is one
policy, so its mean over the 100 held-out days is compared with a one-sample t-test
against the seeds of the best network. A method is bold if it has the best mean or
is not significantly below it (one-sided Welch's t-test over seeds, p >= 0.05).
usage: python scripts/table_bold.py
"""
import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

R = Path(__file__).resolve().parents[1] / "results"


def bpm(pattern, method):
    out = []
    for d in sorted(glob.glob(str(R / pattern))):
        rows = json.loads((Path(d) / "summary.json").read_text())["rows"]
        row = rows.get(method) if isinstance(rows, dict) else next((r for r in rows if r["method"] == method), None)
        if row:
            out.extend(row["heldout_finals"])
    return np.array(out)


def pi(pattern, exclude=("_rn_",)):
    out = []
    for d in sorted(glob.glob(str(R / pattern))):
        for f in sorted(Path(d).glob("summary_tau0.25*.json")):
            if not any(x in f.name for x in exclude):
                out.append(json.loads(f.read_text())["heldout_norm_final"])
    return np.array(out)


BP = "pi_bpic15_n1_R8_d40_mc_fair_mrg_clkn_sh32"
ROUNDS = json.loads((R / "snellius_bpic15_rounds.json").read_text())["rounds"]   # ten collections, paper anchors


INS_ANCHORS = {1: ((39.84, 54.35), (40.03, 54.35)), 2: ((81.45, 108.36), (80.67, 108.36))}   # (run, paper)


def renorm_ins(n, x):
    """Insurer DCL runs and lookahead: from the anchors of the run_pi collection to the
    paper's (those of the model-free runs, the observation with merged options)."""
    (r0, h0), (r1, h1) = INS_ANCHORS[n]
    return (np.asarray(x) * (h0 - r0) + (r0 - r1)) / (h1 - r1)


def renorm(x):
    """BPIC15 DCL runs: from the anchors of their collection (random 39.25, heuristic 55.3)
    to the paper's (38.73, 55.3), the observation with merged options."""
    return (np.asarray(x) * (55.3 - 39.25) + (39.25 - 38.73)) / (55.3 - 38.73)
ROWS = {
    "Insurer, 1 region": {
        "PPO": bpm("bpm_insurer_n1_ep150*clkn_mrg", "ppo"), "NF-PPO": bpm("bpm_insurer_n1_ep150*clkn_mrg", "nfgae"),
        "Lookahead": float(renorm_ins(1, 1.1289)),
        "DCL": renorm_ins(1, pi("pi_insurer_n1_R8_d40_mc_fair_mrg_clkn_sh32")),
        "NF-DCL": renorm_ins(1, pi("pi_insurer_n1_R8_d40_mc_fair_mrg_clkn_sh32_cf*"))},
    "Insurer, 2 regions": {
        "PPO": bpm("bpm_insurer_n2_ep150*clkn_mrg", "ppo"), "NF-PPO": bpm("bpm_insurer_n2_ep150*clkn_mrg", "nfgae"),
        "Lookahead": float(renorm_ins(2, 1.1215)),
        "DCL": renorm_ins(2, pi("pi_insurer_n2_R8_d40_mc_fair_mrg_clkn_sh32")),
        "NF-DCL": renorm_ins(2, pi("pi_insurer_n2_R8_d40_mc_fair_mrg_clkn_sh32_cf*"))},
    "Allocation, 1 region": {
        "PPO": bpm("bpm_insurer_alloc_n1_ep150*clkn_mrg", "ppo"), "NF-PPO": bpm("bpm_insurer_alloc_n1_ep150*clkn_mrg", "nfgae"),
        "Agents": bpm("bpm_insurer_alloc_n1_ep150*clkn_mrg_abfull", "agents"), "Lookahead": 1.098,
        "DCL": pi("pi_insurer_alloc_n1_R8_d40_mc_fair_mrg_clkn_sh32"), "NF-DCL": pi("pi_insurer_alloc_n1_R8_d40_mc_fair_mrg_clkn_sh32_cf*_dyn")},
    "Allocation, 2 regions": {
        "PPO": bpm("bpm_insurer_alloc_n2_ep150*clkn_mrg", "ppo"), "NF-PPO": bpm("bpm_insurer_alloc_n2_ep150*clkn_mrg", "nfgae"),
        "Agents": bpm("bpm_insurer_alloc_n2_ep150*clkn_mrg_abfull", "agents"), "Lookahead": 1.087,
        "DCL": pi("pi_insurer_alloc_n2_R8_d40_mc_fair_mrg_clkn_sh32"), "NF-DCL": pi("pi_insurer_alloc_n2_R8_d40_mc_fair_mrg_clkn_sh32_cf*_dyn")},
    "Slow mismatch, 1 region": {
        "PPO": bpm("bpm_insurer_slow_n1_ep150*beta0.05_ppc_*clkn_mrg", "ppo"), "NF-PPO": bpm("bpm_insurer_slow_n1_ep150*beta0.05_ppc_*clkn_mrg", "nfgae"),
        "Agents": bpm("bpm_insurer_slow_n1_ep150*beta0.05_ppc_*clkn_mrg_abfull", "agents"), "Lookahead": 1.069,
        "DCL": pi("pi_insurer_slow_n1_R8_d40_mc_ppc_fair_mrg_clkn_sh32"), "NF-DCL": pi("pi_insurer_slow_n1_R8_d40_mc_ppc_fair_mrg_clkn_sh32_cf*_dyn")},
    "Municipalities": {
        "PPO": bpm("bpm_bpic15_n1_ep40*clkn_mrg_abfull", "ppo"), "NF-PPO": bpm("bpm_bpic15_n1_ep40*clkn_mrg_abfull", "nfgae"),
        "Agents": bpm("bpm_bpic15_n1_ep40*clkn_mrg_abfull", "agents"), "Lookahead": 1.405,
        "DCL": np.array(ROUNDS["1"]["dcl_final"]), "NF-DCL": np.array(ROUNDS["1"]["nfdcl_final"])},
    "Municipalities, round 2": {"Lookahead": 1.405, "DCL": np.array(ROUNDS["2"]["dcl_final"]), "NF-DCL": np.array(ROUNDS["2"]["nfdcl_final"])},
    "Municipalities, round 3": {"Lookahead": 1.405, "DCL": np.array(ROUNDS["3"]["dcl_final"]), "NF-DCL": np.array(ROUNDS["3"]["nfdcl_final"])},
}

# Table 5 (rounds): per row, the lookahead and the final networks of DCL and NF-DCL after
# every round; the Snellius array (results/snellius_rounds.json, one seed per collection,
# already on the paper's anchors) and the ten-collection array on the municipalities.
SR = json.loads((R / "snellius_rounds.json").read_text())
for tag, name in [("ins1", "Insurer, 1 region"), ("ins2", "Insurer, 2 regions"), ("alloc1", "Allocation, 1 region"),
                  ("alloc2", "Allocation, 2 regions"), ("slow", "Slow mismatch, 1 region")]:
    if tag in SR["benchmarks"]:
        row = {"Lookahead": SR["lookahead_paper"][tag]}
        for r, d in SR["benchmarks"][tag].items():
            row[f"DCL, round {r}"] = np.array(d["dcl_final"]); row[f"NF-DCL, round {r}"] = np.array(d["nfdcl_final"])
        ROWS[f"Rounds: {name}"] = row
ROWS["Rounds: Municipalities"] = {"Lookahead": 1.405, **{f"{m}, round {r}": np.array(ROUNDS[r][k]) for r in "123" for m, k in [("DCL", "dcl_final"), ("NF-DCL", "nfdcl_final")]}}

for row, methods in ROWS.items():
    means = {m: (v if np.isscalar(v) else float(np.mean(v))) for m, v in methods.items()}
    best = max(means, key=means.get)
    print(f"\n{row}: best = {best} ({means[best]:.3f})")
    bold = []
    for m, v in methods.items():
        if m == best:
            bold.append(m); continue
        if np.isscalar(v):                      # the lookahead against the seeds of the best network
            p = stats.ttest_1samp(methods[best], v, alternative="greater").pvalue
        elif np.isscalar(methods[best]):        # a network against the lookahead's mean
            p = stats.ttest_1samp(v, methods[best], alternative="less").pvalue
        else:
            p = stats.ttest_ind(v, methods[best], equal_var=False, alternative="less").pvalue
        n = 1 if np.isscalar(v) else len(v)
        mark = "  BOLD" if p >= 0.05 else ""
        print(f"  {m:9s} {means[m]:.3f} (n={n})  p = {p:.3f}{mark}")
        if p >= 0.05:
            bold.append(m)
    print("  bold:", ", ".join(bold))

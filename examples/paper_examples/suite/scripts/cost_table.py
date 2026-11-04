"""Table 4 (cost) of the paper, from the cost probes on the idle laptop, one process of one
thread: per decision (results/time_decisions.jsonl, 20 held-out days, last entry per
method), model-free training in minutes per epoch (three-epoch probe, *_ep3_smoke3*),
the collection of DCL / NF-DCL in minutes per labeled day with one worker (four-day probe,
*_d4* / *_d2*; NF-DCL: per day of component-days, one per component) and their training
in minutes per epoch on the collections of Table 3 (summary_tau0.25_rn_timing.json, or
SNELLIUS_TRAIN for the collections that live on Snellius, from the epoch timestamps of
the round-1 logs).
usage: python scripts/cost_table.py
"""
import glob
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
ROWS = [  # name, env, N, probe days, static cf of the probe, cf of the full run, postponement
    ("Insurer, 1 region", "insurer", 1, 4, "4-4-4", "40-40-40", False),
    ("Insurer, 2 regions", "insurer", 2, 4, "4-4-4-4-4-4", "40-40-40-40-40-40", False),
    ("Allocation, 1 region", "insurer_alloc", 1, 4, "4", "40", False),
    ("Allocation, 2 regions", "insurer_alloc", 2, 4, "4-4", "40-40", False),
    ("Slow mismatch, 1 region", "insurer_slow", 1, 4, "4-4-4", "40-40-40", True),
    ("Municipalities", "bpic15", 1, 2, "2-2-2-2", "35-24-58-56", False),
]
SNELLIUS_TRAIN = {   # (env, N, method) -> minutes per epoch: epochs 1 to 15 of the round-1 logs of
    # results/snellius_rounds/ins{1,2}_c1 (32-core node, the other chain collecting alongside)
    ("insurer", 1, "DCL"): 188 / 14 / 60, ("insurer", 1, "NF-DCL"): 171 / 14 / 60,
    ("insurer", 2, "DCL"): 405 / 14 / 60, ("insurer", 2, "NF-DCL"): 385 / 14 / 60}
MF_METHODS = [("PPO", "ppo"), ("NF-PPO", "nfgae"), ("Agents", "agents")]


def timing():
    out = {}
    for line in (R / "time_decisions.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            if d["days"] == 20:
                out[(d["env"], d["N"], d["name"])] = d
    return out


def summary(pattern):
    fs = sorted(glob.glob(str(R / pattern)))
    return json.loads(Path(fs[-1]).read_text()) if fs else None


def fmt(x, nd=1):
    return "--" if x is None else f"{x:.{nd}f}"


T = timing()
lines = []
for name, env, n, days, cfp, cff, pp in ROWS:
    ms = [T[k]["ms_inference_per_decision"] for k in T if k[0] == env and k[1] == n]
    la = next((T[k].get("lookahead_s_per_decision") for k in T if k[0] == env and k[1] == n and T[k].get("lookahead_s_per_decision")), None)
    net = "--" if not ms else (f"{min(ms):.1f}--{max(ms):.1f}" if max(ms) - min(ms) >= 0.1 else f"{min(ms):.1f}")
    mf = []
    for lab, m in MF_METHODS:
        c = summary(f"bpm_{env}_n{n}_ep3_smoke3_*/cells/N{n}__{m}__s0.json")
        mf.append(fmt(c["minutes"] / 3 if c else None, 2))
    ppd = "_ppc" if pp else ""
    glob_dyn = "" if env == "bpic15" else "_dyn"
    col, tr = [], []
    for cf_probe, cf_full in [("", ""), (f"_cf{cfp}{glob_dyn}", f"_cf{cff}{glob_dyn}")]:
        s = summary(f"pi_{env}_n{n}_R8_d{days}_mc{ppd}_fair_mrg_clkn_sh32{cf_probe}/summary_tau0.25.json")
        col.append(fmt(s["collect_minutes"] / days if s else None, 1))
        s = summary(f"pi_{env}_n{n}_R8_d40_mc{ppd}_fair_mrg_clkn_sh32{cf_full}/summary_tau0.25_rn_timing.json")
        key = (env, n, "DCL" if not cf_full else "NF-DCL")
        tr.append(fmt(s["train_minutes"] / 15 if s else SNELLIUS_TRAIN.get(key), 2))
    lines.append(f"{name:24s} & {net:8s} & {fmt(la)} & {' & '.join(mf)} & {' & '.join(col)} & {' & '.join(tr)} \\\\")
print("\n".join(lines))

"""Time per decision of a trained network on the held-out days, next to the
lookahead's time per decision from planner_heldout, for the cost table.

For every day: the network's forward pass (inference, one torch thread) and the
simulation step that follows it (firing and the next observation) are timed
separately, so that the table can report what a process-aware information
system pays per decision. Component agents (a ComponentPolicy) are timed in
component turns.

usage: python scripts/time_decisions.py env=insurer_alloc N=1 policy=<policy.pth> [days=20]
                                         [postpone=component] [marking=1] [name=<label>]
Appends one JSON record to results/time_decisions.jsonl and prints it.
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

SUITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE / "envs"))
opts = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
ENV, N = opts.get("env", "insurer_alloc"), int(opts.get("N", 1))
DAYS = int(opts.get("days", 20))
POSTPONE = opts.get("postpone", "0") == "component"
MARKING = opts.get("marking", "1") == "1"
POLICY = opts["policy"]
NAME = opts.get("name", Path(POLICY).parent.name)

from heldout import make_env, machine_info, HELDOUT_SEED  # noqa: E402
if ENV.startswith("bpic15"):
    from bpic15_env import BPIC15_BUILDERS as B  # noqa: E402
else:
    from insurer_env import INSURER_BUILDERS as B  # noqa: E402

torch.set_num_threads(1)
env = make_env(B[ENV], N, 20, POSTPONE, 'component' if POSTPONE else 'global', True, 'edges', MARKING)
env.pn.clock_node, env.pn.merge_options = True, True
env.frozen_pn.clock_node, env.frozen_pn.merge_options = True, True
policy = torch.load(POLICY, weights_only=False).eval()
if type(policy).__name__ == "ComponentPolicy":
    env.pn.match_policy_view(policy)
    env.frozen_pn.match_policy_view(policy)

infer, step, decisions = [], [], 0
for i in range(DAYS):
    random.seed(HELDOUT_SEED + i)
    np.random.seed((HELDOUT_SEED + i) % (2 ** 32))
    t = time.perf_counter()
    obs, done = env.reset(), False
    step.append(time.perf_counter() - t)
    while not done:
        t = time.perf_counter()
        with torch.no_grad():
            a = int(torch.argmax(policy(obs).reshape(-1)).item())
        infer.append(time.perf_counter() - t)
        t = time.perf_counter()
        obs, _, done, _, _ = env.step(a)
        step.append(time.perf_counter() - t)
        decisions += 1

rec = {"env": ENV, "N": N, "name": NAME, "policy": POLICY, "days": DAYS, "decisions": decisions,
       "ms_inference_per_decision": 1000 * float(np.mean(infer)),
       "ms_simulation_per_decision": 1000 * float(np.sum(step) / max(1, decisions)),
       "machine": machine_info(workers=1, threads=1)}
f = SUITE / "results" / f"planner_heldout_{ENV}_n{N}{'_ppc' if POSTPONE else ''}.jsonl"
if f.exists():
    rs = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    sh = [r for r in rs if r["rule"] == "sh32"]
    if sh:
        rec["lookahead_s_per_decision"] = float(sum(r["sec"] for r in sh) / sum(r["decisions"] for r in sh))
        rec["lookahead_steps_per_decision"] = float(sum(r["steps"] for r in sh) / sum(r["decisions"] for r in sh))
with open(SUITE / "results" / "time_decisions.jsonl", "a") as out:
    out.write(json.dumps(rec) + "\n")
print(json.dumps(rec, indent=1))

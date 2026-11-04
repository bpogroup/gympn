"""Allocation probe for the start-of-shift allocation insurer with N regions, on
the policies of the paper protocol (DCL observation: options and clock node,
150 epochs, beta 0.05; results/bpm_insurer_alloc_n<N>_ep150_*_clkn_mrg).

Every final (or best) policy is played greedily on 20 scenarios (seeds
777000+i). Per policy we record, per region, the allocation of the pool it
chose, the policy's own return, the return the heuristic's operations get with
the SAME allocation on the same scenarios, and the heuristic with the best plan
in every region. The gap to the best plan then splits into what the policy
loses in the allocation (best plan - heuristic with its allocation) and what it
gains or loses in the operations (own return - heuristic with its allocation).

usage: python scripts/alloc_probe_n.py N=2 [which=final|best] [days=20] [workers=2] [run=agents]
run=agents probes the component agents (the *_abfull run) instead.
Writes results/alloc_probe_n<N>_<which>[_agents].json.
"""
import glob
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

SUITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE / "envs"))
opts = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
N, WHICH = int(opts.get("N", 2)), opts.get("which", "final")
DAYS, WORKERS, SEED0, LENGTH = int(opts.get("days", 20)), int(opts.get("workers", 2)), 777_000, 20
AGENTS = opts.get("run") == "agents"
RUN = SUITE / "results" / (f"bpm_insurer_alloc_n{N}_ep150_smoke150_beta0.05_nopp_aepn_flat_mc_fv_clkn_mrg"
                           + ("_abfull" if AGENTS else ""))


def _env(marking):
    from heldout import make_env
    from insurer_env import INSURER_BUILDERS
    env = make_env(INSURER_BUILDERS["insurer_alloc"], N, LENGTH, False, 'global', True, 'edges', marking)
    for pn in (env.pn, env.frozen_pn):          # the DCL observation, as run_bpm's _obs_options
        pn.clock_node, pn.obs_clock, pn.merge_options = True, False, True
    return env


def _region(b):
    tr = str(getattr(b[2], '_id', None) or getattr(b[2], 'name', ''))
    return tr.split('_')[2] if tr.startswith('al_') else None


def make_region_heuristic(plans):
    """make_alloc_heuristic with one plan per region tag."""
    from insurer_env import _alloc_binding, insurer_heuristic

    def heuristic(observable_net, tokens_comb, bindings=None):
        if not bindings:
            return None
        alloc, ops = [], []
        for b in bindings:
            ab = (_alloc_binding(b) if isinstance(b, tuple) and b and isinstance(b[0], list) else None)
            (alloc if ab is not None else ops).append((b, ab))
        if alloc:
            for b, (dest, skill) in alloc:
                plan = plans[_region(b)]
                k = len(plan) - len(b[0][0][0].marking)
                if 0 <= k < len(plan) and tuple(plan[k]) == (dest, skill):
                    return b
            return alloc[0][0]
        return insurer_heuristic(observable_net, tokens_comb, [b for b, _ in ops])
    return heuristic


def episode(env, seed, choose):
    """One greedy day: (allocation per region, first-decision allocation probabilities, return)."""
    from insurer_env import _alloc_binding
    random.seed(seed); np.random.seed(seed % (2 ** 32))
    obs, done, info = env.reset(), False, {'pn_reward': 0}
    alloc, first = {}, None
    while not done:
        B = env.pn.pn_actions
        a, probs = choose(env, obs)
        if first is None and probs is not None:
            first = Counter()
            for b, p in zip(B, probs):
                ab = _alloc_binding(b)
                first[f"{_region(b)}:{ab}" if ab is not None else "ops"] += float(p)
        ab = _alloc_binding(B[a])
        if ab is not None:
            alloc.setdefault(_region(B[a]), []).append(ab)
        obs, _, done, _, info = env.step(a)
    return {r: sorted(v) for r, v in alloc.items()}, first, float(info['pn_reward'])


def heuristic_chooser(h):
    def choose(env, obs):
        B = env.pn.pn_actions
        return B.index(h(None, None, B)), None
    return choose


def policy_chooser(path, env=None):
    import torch
    pol = torch.load(path, weights_only=False).eval()
    if env is not None and hasattr(pol, "pn_settings"):    # component agents: component turns
        env.pn.match_policy_view(pol)
        env.frozen_pn.match_policy_view(pol)

    def choose(env, obs):
        with torch.no_grad():
            p = pol(obs).reshape(-1)
        return int(torch.argmax(p).item()), p.tolist()
    return choose


def run_policy(args):
    import torch
    torch.set_num_threads(1)
    method, seed, path = args
    env = _env(method in ("nfgae", "agents"))
    ch = policy_chooser(path, env)
    eps = [episode(env, SEED0 + i, ch) for i in range(DAYS)]
    main = {}
    for r in sorted({k for e in eps for k in e[0]}):
        main[r] = Counter(tuple(e[0].get(r, [])) for e in eps).most_common(1)[0][0]
    first = Counter()
    for e in eps:
        for k, v in (e[1] or {}).items():
            first[k] += v / DAYS
    own = [e[2] for e in eps]
    henv = _env(False)
    hch = heuristic_chooser(make_region_heuristic({r: [tuple(x) for x in v] for r, v in main.items()}))
    same = [episode(henv, SEED0 + i, hch)[2] for i in range(DAYS)]
    return {"method": method, "seed": seed, "allocation": {r: [list(x) for x in v] for r, v in main.items()},
            "distinct": {r: len(Counter(tuple(e[0].get(r, [])) for e in eps)) for r in main},
            "first_decision_probs": {k: round(v, 3) for k, v in first.items()},
            "own": own, "heuristic_same_allocation": same}


def main():
    import multiprocessing as mp
    from insurer_env import ALLOC_PLAN
    jobs = []
    for path in sorted(glob.glob(str(RUN / "train" / "*" / f"{WHICH}_policy.pth"))):
        name = Path(path).parent.name
        method, seed = name.split("__s")
        jobs.append((method, int(seed), path))
    print(f"[probe N={N}] {len(jobs)} {WHICH} policies, {DAYS} scenarios each", flush=True)
    henv = _env(False)
    best = [episode(henv, SEED0 + i, heuristic_chooser(make_region_heuristic(
        {str(r): [tuple(x) for x in ALLOC_PLAN] for r in range(N)})))[2] for i in range(DAYS)]
    print(f"best plan in every region, heuristic operations: {np.mean(best):.2f}", flush=True)
    with mp.get_context("spawn").Pool(WORKERS) as pool:
        rows = pool.map(run_policy, jobs)
    bm = float(np.mean(best))
    print(f"\n{'policy':12s} {'own':>6s} {'h(same)':>8s} {'alloc loss':>10s} {'ops gain':>9s}  allocation")
    for r in sorted(rows, key=lambda x: (x["method"], x["seed"])):
        o, h = float(np.mean(r["own"])), float(np.mean(r["heuristic_same_allocation"]))
        r.update({"own_mean": o, "same_mean": h, "alloc_loss": bm - h, "ops_gain": o - h})
        al = "; ".join(f"{k}:" + ",".join(f"{d}{s}" for d, s in v) for k, v in r["allocation"].items())
        print(f"{r['method']+'_s'+str(r['seed']):12s} {o:6.2f} {h:8.2f} {bm-h:10.2f} {o-h:9.2f}  {al}", flush=True)
    for m in sorted({r["method"] for r in rows}):
        rs = [r for r in rows if r["method"] == m]
        print(f"{m}: mean own {np.mean([r['own_mean'] for r in rs]):.2f}  alloc loss {np.mean([r['alloc_loss'] for r in rs]):.2f}  "
              f"ops gain {np.mean([r['ops_gain'] for r in rs]):.2f}  seeds with best plan everywhere: "
              f"{sum(all(sorted(map(tuple, v)) == sorted(ALLOC_PLAN) for v in r['allocation'].values()) for r in rs)}/{len(rs)}")
    out = SUITE / "results" / f"alloc_probe_n{N}_{WHICH}{'_agents' if AGENTS else ''}.json"
    out.write_text(json.dumps({"N": N, "which": WHICH, "days": DAYS, "best_plan_value": bm, "best_plan_days": best,
                               "rows": rows}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()

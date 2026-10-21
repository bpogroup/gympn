"""Coordination of the allocation decisions (start-of-shift allocation, 1 region).

For each run of alloc_long_probe, loads the snapshot at the last epoch and
measures, at the decision where the second expert is allocated,
    P(second expert -> claims | first -> complaints)
  - P(second expert -> claims | first -> claims),
with the generalist sent to underwriting first. A policy that coordinates the
two experts scores close to 1; a policy whose allocation probabilities do not
depend on the first expert's destination scores 0.

Run: python scripts/alloc_coordination.py [epoch=150] [scenarios=5]
"""
import glob
import json
import os
import random
import re
import sys

SUITE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SUITE, "envs"))
sys.path.insert(0, os.path.join(SUITE, "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import alloc_behaviour  # noqa: E402
from alloc_behaviour import policy_chooser, PN_ATTRS, SEED0, LENGTH  # noqa: E402

ARGS = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
EPOCH, SCEN = int(ARGS.get("epoch", 150)), int(ARGS.get("scenarios", 5))
PROBE = os.path.join(SUITE, "results", "alloc_long_probe")
ARMS = {  # label: run-dir glob
    "PPO": "ppo_s*_ep150_beta0.05_fv",
    "agents static": "agents_s*_ep150_beta0.05_fv",
    "agents marking": "agents_mc_s*_ep150_beta0.05_fv",
    "agents marking, full batch": "agents_mc_s*_ep150_beta0.05_fv_abfull",
    "NF-GAE marking, forward only": "nfgae_mc_s*_ep150_beta0.05",
    "NF-GAE marking": "nfgae_mc_s*_ep150_beta0.05_fv",
}


def _step_until(env, obs, choose, want):
    """Greedy steps until an allocation binding is offered; then force `want`
    (if given) or return the probabilities of the offered bindings."""
    from insurer_env import _alloc_binding
    for _ in range(200):
        B = env.pn.pn_actions
        if any(_alloc_binding(b) is not None for b in B):
            if want is None:
                return obs, B
            i = next((i for i, b in enumerate(B) if _alloc_binding(b) == want), None)
            if i is not None:
                obs, *_ = env.step(i)
                return obs, None
        a, _ = choose(env, obs)
        obs, *_ = env.step(a)
    raise RuntimeError("no allocation binding offered")


def conditional(choose, first, seed):
    from gympn.environment import AEPN_Env
    from insurer_env import make_insurer_alloc, _alloc_binding
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    pn = make_insurer_alloc(1)
    pn.length = LENGTH
    for k, v in PN_ATTRS.items():
        setattr(pn, k, v)
    env = AEPN_Env(pn)
    obs = env.reset()
    obs = obs[0] if isinstance(obs, tuple) else obs
    obs, _ = _step_until(env, obs, choose, ("uw1", 0))
    obs, _ = _step_until(env, obs, choose, first)
    obs, B = _step_until(env, obs, choose, None)
    _, p = choose(env, obs)
    cl = sum(q for b, q in zip(B, p) if _alloc_binding(b) == ("cl", 1))
    co = sum(q for b, q in zip(B, p) if _alloc_binding(b) == ("co", 1))
    return cl / (cl + co) if cl + co > 0 else float("nan")


def main():
    torch.set_num_threads(1)
    out = {}
    for label, pat in ARMS.items():
        gaps = {}
        for d in sorted(glob.glob(os.path.join(PROBE, pat))):
            snaps = glob.glob(os.path.join(d, "train", "*", f"snap_ep{EPOCH:03d}.pth"))
            if not snaps:
                continue
            seed = int(re.search(r"_s(\d+)_", os.path.basename(d)).group(1))
            PN_ATTRS.clear()
            choose = policy_chooser(snaps[0])
            if "fv" not in os.path.basename(d):
                PN_ATTRS["global_view"] = False
            after_cl = np.nanmean([conditional(choose, ("cl", 1), SEED0 + i) for i in range(SCEN)])
            after_co = np.nanmean([conditional(choose, ("co", 1), SEED0 + i) for i in range(SCEN)])
            gaps[seed] = float(after_co - after_cl)
        out[label] = gaps
        v = list(gaps.values())
        print(f"{label:30s} n={len(v)}  coordination {np.mean(v):+.3f}  per seed "
              + " ".join(f"{g:+.2f}" for _, g in sorted(gaps.items())), flush=True)
    json.dump(out, open(os.path.join(SUITE, "results", "alloc_coordination.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

r"""How many lineage-certified NATURAL EXPERIMENTS does an episode contain?

cfpk's power is a PAIRED comparison under common random numbers: two branches,
same noise, subtract, and the environment's variance cancels. It pays for that
pairing by forking the simulator. But pairs also occur naturally inside a
single episode -- and lineage is exactly the tool that certifies one is not
confounded, because independence is what lineage can establish soundly (the
ccf argument), unlike causation.

A pair here is two decisions in the SAME episode that are:
  - in DIFFERENT realized causal components (union-find over reward lineages,
    ccf's own partition) -- so their outcomes are independent draws;
  - offered the SAME set of enabled action types -- so they faced the same
    choice;
  - and took DIFFERENT action types -- so the comparison is informative.
Their component-restricted returns then differ by an unbiased estimate of the
action gap, with episode-level noise shared rather than resampled.

This counts them. No training, no forking, no new mechanism -- if the count is
~0 on the envs of interest the idea dies here for the cost of one short run;
if it grows with K it has fuel, and the scaling story gains a second leg.

Reported per env: components per episode, decisions per episode, LOOSE pairs
(different component + different action type) and STRICT pairs (same enabled
type set as well).

Run: python _diag_pair_count.py
"""
import os
import sys
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn")

import numpy as np

import gympn.agents as agents_mod
import gympn.causal_traces as tm

_enabled = []           # per decision: frozenset of enabled action-type ids
_real_act = agents_mod.Agent.act


def _spy_act(self, state, *a, **kw):
    try:
        ad = state.get('actions_dict') if isinstance(state, dict) else None
        types = set()
        if ad:
            for e in ad:
                tid = getattr(e[2], '_id', None) if len(e) > 2 else None
                if tid:
                    types.add(str(tid).split('__')[0])
        _enabled.append(frozenset(types))
    except Exception:
        _enabled.append(frozenset())
    return _real_act(self, state, *a, **kw)


agents_mod.Agent.act = _spy_act

_R = {}
_real_redis = tm.CausalTraces.redistribute_rewards


def _components(ct):
    """ccf's realized partition: {decision_idx: component_root}."""
    acts = ct.transition_history.get_action_transitions()
    n = len(acts)
    if n == 0:
        return {}, acts
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    tok2act = {}
    for i, a in enumerate(acts):
        for t in a.get('output_tokens', ()):
            tok2act[t] = i

    def ancestors(ids):
        found, seen, stack = set(), set(), list(ids)
        while stack:
            tid = stack.pop()
            if tid in seen:
                continue
            seen.add(tid)
            if tid in tok2act:
                found.add(tok2act[tid])
            stack.extend(ct.token_history.get_parents(tid))
        return found

    touched = set()
    for tr in ct.transition_history.transitions:
        if tr.get('reward', 0.0) == 0.0:
            continue
        decs = sorted(ancestors(tr.get('input_tokens', [])))
        if not decs:
            continue
        touched.update(decs)
        for d in decs[1:]:
            union(decs[0], d)
    return {d: find(d) for d in touched}, acts


def probe(tag):
    _R[tag] = {"ep": 0, "K": [], "dec": [], "loose": [], "strict": []}

    def spy(self, scheme="lrq", **kw):
        out = _real_redis(self, scheme=scheme, **kw)
        if scheme == "lrq2":
            try:
                comp, acts = _components(self)
                n = len(acts)
                enb = _enabled[-n:] if len(_enabled) >= n else []
                _enabled.clear()
                if comp and enb:
                    types = [str(getattr(a.get('transition'), '_id', '')).split('__')[0]
                             for a in acts]
                    items = [(d, comp[d], types[d], enb[d])
                             for d in sorted(comp) if d < len(enb)]
                    loose = strict = 0
                    for (d1, c1, t1, e1), (d2, c2, t2, e2) in combinations(items, 2):
                        if c1 == c2 or t1 == t2:
                            continue
                        loose += 1
                        if e1 and e1 == e2:
                            strict += 1
                    r = _R[tag]
                    r["ep"] += 1
                    r["K"].append(len(set(comp.values())))
                    r["dec"].append(len(items))
                    r["loose"].append(loose)
                    r["strict"].append(strict)
            except Exception as e:
                print(f"[warn] {tag}: {e}")
            _enabled.clear()
        return out

    tm.CausalTraces.redistribute_rewards = spy


def run(tag, make_fn, length, episodes=4):
    probe(tag)
    args = {"algorithm": "ppo-clip", "episodes": episodes, "epochs": 1,
            "batch_size": 64, "max_episode_length": None, "policy_lr": 3e-4,
            "policy_updates": 1, "value_lr": 3e-4, "value_updates": 1,
            "gam": 0.99, "lam": 0.95, "eps": 0.2, "vf_coeff": 0.5,
            "ent_bonus": 0.01, "policy_kld_limit": 0.15, "causal_rl": True,
            "causal_scheme": "lrq2", "causal_beta": 0.5, "verbose": 0,
            "use_gpu": False, "agent_seed": 0, "use_wandb": False,
            "open_tensorboard": False, "test_in_train": False,
            "save_freq": 10**9, "name": "paircount", "datetag": False,
            "logdir": f"paircount_{tag}"}
    saved = sys.argv
    sys.argv = sys.argv[:1]
    try:
        make_fn().training_run(length=length, args_dict=args)
    except Exception as e:
        print(f"[skip] {tag}: {type(e).__name__}: {e}")
    finally:
        sys.argv = saved
        tm.CausalTraces.redistribute_rewards = _real_redis
        import shutil
        shutil.rmtree(f"paircount_{tag}", ignore_errors=True)


from envs import make_env  # noqa: E402
from ncopies_env import make_n_copies  # noqa: E402

run("s1  (K=1)", lambda: make_env("s1_stoch_sequence", causal_rl=True,
                                  allow_postpone=True), 20)
run("s2  (Kr=6)", lambda: make_env("s2_stoch_scaled", causal_rl=True,
                                   allow_postpone=True), 30)
for N in (2, 4, 8):
    run(f"ncopies{N} (K={N})",
        (lambda k: (lambda: make_n_copies(k, causal_rl=True,
                                          allow_postpone=False)))(N), 20)

print("\n" + "=" * 84)
print(f"{'env':<18}{'eps':>4}{'K':>7}{'decisions':>11}{'LOOSE pairs':>13}{'STRICT pairs':>14}{'strict/dec':>12}")
print("-" * 84)
for tag, r in _R.items():
    if not r["ep"]:
        print(f"{tag:<18} (no episodes)")
        continue
    K = np.mean(r["K"]); dc = np.mean(r["dec"])
    lo = np.mean(r["loose"]); st = np.mean(r["strict"])
    print(f"{tag:<18}{r['ep']:>4}{K:>7.1f}{dc:>11.1f}{lo:>13.1f}{st:>14.1f}"
          f"{st / max(dc, 1e-9):>12.2f}")
print()
print("STRICT pairs are the usable ones: same choice offered, different choice")
print("made, independent outcomes. If that column is ~0 the mechanism has no")
print("fuel; if it grows with K it scales the same way ccf's win does.")

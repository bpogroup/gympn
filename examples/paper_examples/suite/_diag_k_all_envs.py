r"""The K-diagnostic (EJOR_DISCUSSION.md 7.1) across every suite environment.

K is the causal-component count. The decision rule from 7.1:

    K > 1 reliably  -> cases partition into causally independent sub-streams;
                       s_ccf / cfpk apply, and the choice between them is a
                       cost/granularity trade-off.
    K ~ 1 always    -> every case's fate is entangled with every other's
                       (usually one shared, fully-utilized resource pool);
                       neither mechanism will beat a well-tuned PPO baseline.

Two counts, because 7.1 refers to both:

  K_static   -- s_ccf's partition, from `_static_component_reward_types`: two
                decision types are in one component when they can statically
                reach a common reward transition. Pure topology, no episodes.
                This is the one that governs s_ccf, whose Remark R1 says
                K=1 => s_ccf degenerates to PPO exactly, BY CONSTRUCTION.
  K_realized -- ccf's partition, the union-find over REALIZED reward lineages
                per episode. 7.1 asks for its distribution across episodes,
                since realized coupling varies with the stochastic case mix.
                Needs a few simulated episodes under any reasonable policy
                (K is a property of the net's coupling, not of policy quality).

Covers envs.ENV_BUILDERS (the a-h grid, i_mixed_credit, j_mixed_rework, and
the stochastic tier) plus the spectrum-tier envs that live in their own
modules (n-copies, foreclosure, join), which is where K>1 is expected.

Run: python _diag_k_all_envs.py [n_episodes]
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn")

import numpy as np

import gympn.causal_traces as traces_mod
from gympn.causal_traces import CausalTraces
from gympn.solvers import RandomSolver

N_EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 5

from envs import ENV_BUILDERS, make_env  # noqa: E402

# --- build the full env roster ------------------------------------------- #
CASES = []
for name in sorted(ENV_BUILDERS):
    CASES.append((name, (lambda n: (lambda: make_env(n, causal_rl=True,
                                                     allow_postpone=True)))(name), 20))
try:
    from ncopies_env import make_n_copies
    for N in (1, 2, 4, 8):
        CASES.append((f"ncopies{N}",
                      (lambda k: (lambda: make_n_copies(k, causal_rl=True,
                                                        allow_postpone=False)))(N), 20))
except Exception as e:
    print(f"[warn] ncopies_env unavailable: {e}")
try:
    from foreclosure_env import make_foreclosure
    CASES.append(("foreclosure",
                  lambda: make_foreclosure(causal_rl=True, allow_postpone=True), 8))
except Exception as e:
    print(f"[warn] foreclosure_env unavailable: {e}")
try:
    from join_env import make_join
    CASES.append(("join", lambda: make_join(causal_rl=True, allow_postpone=False), 20))
    CASES.append(("joinbal", lambda: make_join(causal_rl=True, allow_postpone=False,
                                               r_join=1.0, r1=2.0, r2=3.0), 20))
except Exception as e:
    print(f"[warn] join_env unavailable: {e}")


def k_static(env):
    """Number of distinct static causal components (s_ccf's partition)."""
    ct = CausalTraces()
    ct._pn = env
    comp = ct._static_component_reward_types()
    # Components are the distinct reward-type sets; action types sharing a set
    # are one component. Decisions reaching NO reward are their own trivial
    # component and are excluded -- they carry no credit either way.
    return len({frozenset(v) for v in comp.values() if v})


def _components_of(ct):
    """ccf's realized partition for ONE episode's trace: union the decisions in
    each reward's causal lineage, then count roots. Returns None when the
    episode produced no credited reward (nothing to partition)."""
    acts = ct.transition_history.get_action_transitions()
    n = len(acts)
    if n == 0:
        return None
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

    def ancestors(input_ids):
        found, seen, stack = set(), set(), list(input_ids)
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
    return len({find(d) for d in touched}) if touched else None


# The realized count needs a trace with token PARENTS populated, and only the
# TRAINING path does that -- `testing_run` + RandomSolver leaves parent tokens
# without ids ("One of the parent tokens does not have an _id attribute"), so
# the lineage walk finds nothing and every env reports (n/a). Hooking
# redistribute_rewards, which the training loop calls once per finished
# episode, gets the real thing.
_kr = []
_real_redis = traces_mod.CausalTraces.redistribute_rewards


def _spy_redis(self, *a, **kw):
    out = _real_redis(self, *a, **kw)
    try:
        k = _components_of(self)
        if k is not None:
            _kr.append(k)
    except Exception:
        pass
    return out


traces_mod.CausalTraces.redistribute_rewards = _spy_redis


def k_realized(env, length, n_ep):
    """Realized component count per episode, from a minimal real training run
    (1 epoch x n_ep episodes) -- enough for the distribution 7.1 asks for."""
    import gympn.train as train_mod  # noqa: F401
    _kr.clear()
    args = {
        "algorithm": "ppo-clip", "episodes": int(n_ep), "epochs": 1,
        "batch_size": 64, "max_episode_length": None,
        "policy_lr": 3e-4, "policy_updates": 1, "value_lr": 3e-4,
        "value_updates": 1, "gam": 0.99, "lam": 0.95, "eps": 0.2,
        "vf_coeff": 0.5, "ent_bonus": 0.01, "policy_kld_limit": 0.15,
        "causal_rl": True, "causal_scheme": "lrq2", "causal_beta": 0.5,
        "verbose": 0, "use_gpu": False, "agent_seed": 0,
        "use_wandb": False, "open_tensorboard": False, "test_in_train": False,
        "save_freq": 1_000_000, "name": "kdiag", "datetag": False,
        "logdir": "kdiag_train",
    }
    saved = sys.argv
    sys.argv = sys.argv[:1]
    try:
        env.training_run(length=length, args_dict=args)
    except Exception:
        return []
    finally:
        sys.argv = saved
        import shutil
        shutil.rmtree("kdiag_train", ignore_errors=True)
    return list(_kr)


print(f"K-DIAGNOSTIC  ({N_EPISODES} random episodes per env for K_realized)")
print("=" * 88)
print(f"{'env':<26}{'K_static':>9}  {'K_realized (per episode)':<30}{'verdict':<22}")
print("-" * 88)
rows = []
for name, mk, length in CASES:
    try:
        ks = k_static(mk())
    except Exception as e:
        print(f"{name:<26}{'ERR':>9}  {type(e).__name__}: {e}")
        continue
    try:
        kr = k_realized(mk(), length, N_EPISODES)
    except Exception:
        kr = []
    krs = (f"mean={np.mean(kr):.2f} " + str(dict(sorted(Counter(kr).items())))
           if kr else "(n/a)")
    applies = ks > 1 or (kr and np.mean(kr) > 1.5)
    verdict = "s_ccf/cfpk APPLY" if applies else "K~1 -> expect PPO parity"
    rows.append((name, ks, kr, applies))
    print(f"{name:<26}{ks:>9}  {krs:<30}{verdict:<22}")

print()
ok = [r[0] for r in rows if r[3]]
no = [r[0] for r in rows if not r[3]]
print(f"K>1 (mechanisms apply)      : {ok or 'NONE'}")
print(f"K~1 (expect PPO parity)     : {no}")

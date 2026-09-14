r"""K_pre-decision: the partition ccf wants, computed WITHOUT peeking at outcomes.

ccf's bias is that its components are REALIZED -- membership depends on what
happened after the action, so an action can enlarge its own component and thus
its own credit (join_env.py's AND-join: routing into the join pulls r2 into d1's
component, so ccf prefers the suboptimal join). s_ccf avoids that by
partitioning on TOPOLOGY, which is action-invariant -- but topology is
worst-case, so K_static=1 on both realistic envs and s_ccf is inert there.

The middle ground: partition the CURRENT MARKING at decision time. That is a
function of the state alone, so it is action-invariant -- unbiased by the same
argument that protects s_ccf -- yet strictly finer than topology, because it
knows which resources are actually free right now.

RULE (only claims independence when it can be shown):
  1. pools = re-supplied resource places.
  2. every token in a non-pool place is a work item; its "needs" are the pools
     forward-reachable from its place through the topology.
  3. for each pool p, let D_p be the work items needing it. If the pool
     currently holds >= |D_p| tokens it can serve them all now, so it induces
     NO coupling. Otherwise every item in D_p is unioned -- they contend.
  4. K_pre = number of components.
Abundant resources therefore decouple; saturated ones do not.

Reported against K_static (s_ccf's) and K_realized (ccf's) so the three sit on
the same scale. If K_pre lands near K_realized, the pre-decision partition buys
ccf's power with s_ccf's guarantee.

CAVEAT: pools are detected with the cyclic-place test, which cannot separate a
resource pool from a rework loop (established earlier today). s1, s2 and
multisite have no rework loop so it is safe HERE; it is not a general detector.

Run: python _diag_k_predecision.py
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn")

import numpy as np

from gympn.simulator import GymProblem
import gympn.causal_traces as tm

_K = defaultdict(list)
_real_obs = GymProblem.get_graph_observation


def _topology(pn):
    trans = list(pn.actions) + list(pn.events)
    consumers = defaultdict(list)
    for t in trans:
        for p in t.incoming:
            consumers[p._id].append(t)

    def cyclic(pid):
        seen_p, seen_t, stack = set(), set(), list(consumers.get(pid, []))
        while stack:
            t = stack.pop()
            if t._id in seen_t:
                continue
            seen_t.add(t._id)
            for p in t.outgoing:
                if p._id == pid:
                    return True
                if p._id in seen_p:
                    continue
                seen_p.add(p._id)
                stack.extend(consumers.get(p._id, []))
        return False

    pools = {pid for pid in consumers if cyclic(pid)}

    # pools forward-reachable from each place
    needs = {}
    for p in pn.places:
        seen_t, seen_p, stack, got = set(), set(), list(consumers.get(p._id, [])), set()
        while stack:
            t = stack.pop()
            if t._id in seen_t:
                continue
            seen_t.add(t._id)
            for q in t.incoming:
                if q._id in pools:
                    got.add(q._id)
            for q in t.outgoing:
                if q._id in seen_p:
                    continue
                seen_p.add(q._id)
                stack.extend(consumers.get(q._id, []))
        needs[p._id] = got
    return pools, needs


_cache = {}


def k_pre(pn):
    key = id(pn)
    if key not in _cache:
        _cache[key] = _topology(pn)
    pools, needs = _cache[key]
    items, want = [], []
    free = {}
    for p in pn.places:
        mk = list(getattr(p, 'marking', None) or ())
        if p._id in pools:
            free[p._id] = len(mk)
            continue
        for t in mk:
            items.append((p._id, id(t)))
            want.append(needs.get(p._id, set()))
    n = len(items)
    if n == 0:
        return None
    par = list(range(n))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    def uni(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            par[ra] = rb

    by_pool = defaultdict(list)
    for i, w in enumerate(want):
        for pid in w:
            by_pool[pid].append(i)
    for pid, D in by_pool.items():
        if free.get(pid, 0) >= len(D):
            continue            # pool can serve them all now -> no coupling
        for i in D[1:]:
            uni(D[0], i)
    return len({find(i) for i in range(n)})


def _spy_obs(self, *a, **kw):
    try:
        k = k_pre(self)
        if k:
            _K[_TAG[0]].append(k)
    except Exception:
        pass
    return _real_obs(self, *a, **kw)


GymProblem.get_graph_observation = _spy_obs
_TAG = ["?"]

_KR = defaultdict(list)
_real_redis = tm.CausalTraces.redistribute_rewards


def _comps(ct):
    acts = ct.transition_history.get_action_transitions()
    n = len(acts)
    if n == 0:
        return None
    par = list(range(n))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x

    def uni(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            par[ra] = rb
    t2a = {}
    for i, a in enumerate(acts):
        for t in a.get('output_tokens', ()):
            t2a[t] = i

    def anc(ids):
        f, seen, st = set(), set(), list(ids)
        while st:
            t = st.pop()
            if t in seen:
                continue
            seen.add(t)
            if t in t2a:
                f.add(t2a[t])
            st.extend(ct.token_history.get_parents(t))
        return f
    touched = set()
    for tr in ct.transition_history.transitions:
        if tr.get('reward', 0.0) == 0.0:
            continue
        d = sorted(anc(tr.get('input_tokens', [])))
        if not d:
            continue
        touched.update(d)
        for x in d[1:]:
            uni(d[0], x)
    return len({find(x) for x in touched}) if touched else None


def _spy_redis(self, *a, **kw):
    o = _real_redis(self, *a, **kw)
    sch = kw.get('scheme', a[0] if a else 'lrq')
    if sch == 'lrq2':
        k = _comps(self)
        if k:
            _KR[_TAG[0]].append(k)
    return o


tm.CausalTraces.redistribute_rewards = _spy_redis


def k_static(pn):
    ct = tm.CausalTraces(); ct._pn = pn
    comp = ct._static_component_reward_types()
    return len({frozenset(v) for v in comp.values() if v})


def run(tag, make_fn, length, eps=3):
    _TAG[0] = tag
    env = make_fn()
    ks = k_static(env)
    args = {"algorithm": "ppo-clip", "episodes": eps, "epochs": 1, "batch_size": 64,
            "max_episode_length": None, "policy_lr": 3e-4, "policy_updates": 1,
            "value_lr": 3e-4, "value_updates": 1, "gam": 0.99, "lam": 0.95,
            "eps": 0.2, "vf_coeff": 0.5, "ent_bonus": 0.01,
            "policy_kld_limit": 0.15, "causal_rl": True, "causal_scheme": "lrq2",
            "causal_beta": 0.5, "verbose": 0, "use_gpu": False, "agent_seed": 0,
            "use_wandb": False, "open_tensorboard": False, "test_in_train": False,
            "save_freq": 10**9, "name": "kpre", "datetag": False,
            "logdir": f"kpre_{tag}"}
    sv = sys.argv
    sys.argv = sys.argv[:1]
    try:
        env.training_run(length=length, args_dict=args)
    except Exception as e:
        print(f"[skip] {tag}: {type(e).__name__}: {e}")
    finally:
        sys.argv = sv
        import shutil
        shutil.rmtree(f"kpre_{tag}", ignore_errors=True)
    return ks


from envs import make_env  # noqa: E402
from multisite_env import make_multisite  # noqa: E402
from ncopies_env import make_n_copies  # noqa: E402

CASES = [
    ("multisite", lambda: make_multisite(causal_rl=True, allow_postpone=False), 30),
    ("s2", lambda: make_env("s2_stoch_scaled", causal_rl=True, allow_postpone=True), 30),
    ("s1", lambda: make_env("s1_stoch_sequence", causal_rl=True, allow_postpone=True), 20),
    ("ncopies8", lambda: make_n_copies(8, causal_rl=True, allow_postpone=False), 20),
]
stat = {}
for tag, fn, L in CASES:
    stat[tag] = run(tag, fn, L)

print("\n" + "=" * 74)
print(f"{'env':<12}{'K_static':>10}{'K_pre (mean)':>14}{'K_realized':>13}   verdict")
print("-" * 74)
for tag, _, _ in CASES:
    kp = np.mean(_K[tag]) if _K[tag] else float('nan')
    kr = np.mean(_KR[tag]) if _KR[tag] else float('nan')
    if np.isnan(kp):
        v = "no data"
    elif kp >= 0.6 * kr:
        v = "recovers ccf's power, unbiased"
    elif kp <= 1.5:
        v = "collapses like K_static"
    else:
        v = "partial"
    print(f"{tag:<12}{stat[tag]:>10}{kp:>14.2f}{kr:>13.2f}   {v}")
print()
print("K_pre near K_realized => the pre-decision partition gives ccf's power")
print("with s_ccf's action-invariance. Near K_static => no better than s_ccf.")

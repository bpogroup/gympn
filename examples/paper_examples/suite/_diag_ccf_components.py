r"""How many components does ccf's partition actually have on s1?

ccf unions every decision that co-causes a reward into one component, then
gives each decision its component's return-to-go. The prediction: on s1's
single shared 3-employee pool everything fuses into ONE component, in which
case ccf's reward set == mc_q's and ccf degenerates to mc_q exactly (which
scores 13.41 vs ppo's 13.59 on s1 -- i.e. nothing to gain).

Reported per episode:
  K            number of components over decisions (ccf's union-find)
  K_reward     components that actually own >=1 reward (the rest are
               singletons -- postpone, or decisions that caused nothing)
  largest      share of REWARD MASS held by the biggest component
  ccf/mc_q     mean over decisions of (ccf credit) / (mc_q credit): 1.0 means
               ccf and mc_q are the same estimator on this env

ncopies N=4 is run alongside as the contrast, where the partition should find
~4 components (one per copy) and the ratio should sit near 1/4.

Run: python _diag_ccf_components.py
"""
import os, sys, random, types, uuid
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gympn.environment import AEPN_Env
from envs import make_env
from ncopies_env import make_n_copies

EPISODES = 6
LENGTH = 20


def build(builder, causal_tokenflow):
    pn = builder()
    pn.length = LENGTH
    for p in pn.places:
        for t in p.marking:
            setattr(t, '_id', str(uuid.uuid4()))
    pn.causal_trace._pn = pn
    pn.causal_trace._static_comp_cache = None
    pn.causal_trace.postpone_tokenflow = causal_tokenflow
    pn.causal_trace.flush()
    sent = types.SimpleNamespace(_id="__initial__")
    for p in pn.places:
        for t in p.marking:
            pn.causal_trace.register_token(t, sent, parent_tokens=[], time=0)
    pn.causal_trace.register_transition(
        transition=sent, input_tokens=[],
        output_tokens=[t for p in pn.places for t in p.marking],
        is_action=False, reward=0.0, time=0)
    return AEPN_Env(pn)


def analyse(env):
    ct = env.pn.causal_trace
    acts = ct.transition_history.get_action_transitions()
    n = len(acts)
    if n == 0:
        return None

    token_to_action = {}
    for idx, a in enumerate(acts):
        for t in a.get('output_tokens', ()) or ():
            token_to_action[t] = idx

    def parents(tid):
        info = ct.token_history.get_token(tid)
        return info.get("parents", []) if info else []

    def lineage(in_ids, firing_idx):
        found = set()
        if firing_idx is not None:
            found.add(firing_idx)
        seen, stack = set(), list(in_ids)
        while stack:
            tid = stack.pop()
            if tid in seen:
                continue
            seen.add(tid)
            hit = token_to_action.get(tid)
            if hit is not None:
                found.add(hit)
            for p in parents(tid):
                if p not in seen:
                    stack.append(p)
        return found

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

    id_to_rec = {id(t): t for t in ct.transition_history.transitions}
    rewards = []
    for tr in ct.transition_history.transitions:
        rv = tr.get('reward', 0.0)
        if rv == 0.0:
            continue
        decs = [d for d in lineage(tr.get('input_tokens', ()), None) if 0 <= d < n]
        for i in range(1, len(decs)):
            union(decs[0], decs[i])
        rewards.append((rv, tr.get('time'), decs))

    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)

    mass = {}
    for rv, _t, decs in rewards:
        if decs:
            mass[find(decs[0])] = mass.get(find(decs[0]), 0.0) + rv
    total_mass = sum(mass.values()) or 1.0

    # ccf credit vs mc_q credit per decision (beta=0 -> undiscounted sums)
    ratios = []
    times = [a.get('time') for a in acts]
    for d in range(n):
        u = times[d]
        cd = find(d)
        ccf_q = sum(rv for rv, t_j, decs in rewards
                    if decs and find(decs[0]) == cd
                    and (t_j is None or u is None or u <= t_j))
        mcq_q = sum(rv for rv, t_j, _d in rewards
                    if t_j is None or u is None or u <= t_j)
        if mcq_q > 0:
            ratios.append(ccf_q / mcq_q)
    return dict(K=len(comps), K_rew=len(mass),
                largest=max(mass.values()) / total_mass if mass else float('nan'),
                ratio=float(np.mean(ratios)) if ratios else float('nan'),
                n=n)


for label, builder, tf in [
    ("s1        ", lambda: make_env("s1_stoch_sequence", causal_rl=True,
                                    allow_postpone=True, causal_postpone_tokenflow=True), True),
    ("ncopies N=4", lambda: make_n_copies(4, causal_rl=True, allow_postpone=True,
                                          causal_postpone_tokenflow=True), True),
]:
    rows = []
    for ep in range(EPISODES):
        random.seed(200 + ep)
        env = build(builder, tf)
        env.reset(); done = False
        while not done:
            _, _, done, _, _ = env.step(random.randrange(len(env.pn.pn_actions)))
        r = analyse(env)
        if r:
            rows.append(r)
    K = np.mean([r['K'] for r in rows]); Kr = np.mean([r['K_rew'] for r in rows])
    lg = np.mean([r['largest'] for r in rows]); rt = np.mean([r['ratio'] for r in rows])
    dec = np.mean([r['n'] for r in rows])
    print(f"{label}: decisions/ep {dec:5.1f} | K {K:5.1f} | K with rewards {Kr:4.1f} "
          f"| largest component holds {100*lg:5.1f}% of reward mass "
          f"| ccf/mc_q credit {rt:.3f}")

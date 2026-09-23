r"""WHY the Xi_d step holds, when the proof's stated reason does not.

_diag_xi_gap.py shows the counterexample configuration never fires (0 of
~177k pairs) even though multi-ancestry rewards are the norm, and that the
operative fact is

    own(j) is succ-reachable from every other member of A(j)    (13211/13211)

This script tests the structural condition that would EXPLAIN that, and hence
make it a lemma rather than a measurement:

  (S1) SINGLE ORIGIN. The nearest-producer decisions of the tokens consumed by
       a reward-emitting firing form a single decision a_j (or none). Then
       every other member of A(j) is reached by continuing the backward walk
       from a_j's own inputs, so A(j) is contained in anc*(a_j); and since
       reachability is time-forward, a_j is also the clock-latest member, i.e.
       a_j = own(j).

  (S2) UNIQUE MAXIMUM (weaker). A(j) has a unique maximum under
       succ-reachability. S1 implies S2; S2 is what the lemma actually needs.

If S1 holds everywhere the lemma is clean and checkable from the net's
structure. If S1 fails somewhere but S2 holds, the lemma must be stated on S2.

Run: python _diag_xi_origin.py [n_rollouts]
"""
import os, sys, random, types, uuid
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gympn.environment import AEPN_Env
from gympn.causal_traces import CausalTraces

EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 20
LENGTH = 20

CAP = {}
_orig = CausalTraces._cgae_structure


def _spy(self, action_transitions, token_to_action, record_to_action, get_parents):
    out = _orig(self, action_transitions, token_to_action, record_to_action,
                get_parents)
    n = len(action_transitions)
    times = [act.get('time') for act in action_transitions]

    out_tok = {}
    for idx, act in enumerate(action_transitions):
        for t in act.get('output_tokens', ()) or ():
            out_tok[t] = idx

    # the nearest-producer walk, verbatim from _cgae_structure._producers
    def producers(start_tid, self_idx):
        found, seen, stack = set(), set(), [start_tid]
        while stack:
            tid = stack.pop()
            if tid in seen:
                continue
            seen.add(tid)
            src = out_tok.get(tid)
            if src is not None and src != self_idx:
                found.add(src)
                continue
            for p in get_parents(tid):
                if p not in seen:
                    stack.append(p)
        return found

    def lineage(input_ids, firing_idx):
        found = set()
        if firing_idx is not None:
            found.add(firing_idx)
        seen, stack = set(), list(input_ids)
        while stack:
            tid = stack.pop()
            if tid in seen:
                continue
            seen.add(tid)
            hit = token_to_action.get(tid)
            if hit is not None:
                found.add(hit[0])
            for p in get_parents(tid):
                if p not in seen:
                    stack.append(p)
        return found

    rewards = []
    for tr in self.transition_history.transitions:
        rv = tr.get('reward', 0.0)
        if rv == 0.0:
            continue
        se = record_to_action.get(id(tr))
        self_idx = se[0] if se else None
        toks = list(tr.get('input_tokens', []) or [])
        decs = [d for d in lineage(toks, self_idx) if 0 <= d < n]
        if not decs:
            continue
        origins = set()
        for t in toks:
            origins |= producers(t, self_idx)
        if self_idx is not None:
            origins.add(self_idx)      # a rewarding firing that is itself a decision
        origins = {o for o in origins if 0 <= o < n}
        owner = max(decs, key=lambda d: (times[d] is not None, times[d] or 0.0))
        rewards.append((rv, set(decs), owner, origins, len(toks)))

    CAP['succ'], CAP['owned'], CAP['times'] = out[0], out[2], out[3]
    CAP['rewards'] = rewards
    return out


CausalTraces._cgae_structure = _spy


def build(maker):
    pn = maker()
    pn.length = LENGTH
    for p in pn.places:
        for t in p.marking:
            setattr(t, '_id', str(uuid.uuid4()))
    pn.causal_trace._pn = pn
    pn.causal_trace._static_comp_cache = None
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


def rollout(maker, seed):
    random.seed(seed)
    env = build(maker)
    env.reset()
    done, guard = False, 0
    while not done and guard < 10000:
        guard += 1
        k = len(env.pn.get_graph_observation().get('actions_dict') or [])
        if k == 0:
            break
        _, _, done, _, _ = env.step(random.randrange(k))
    return env


def closure(succ, order, n):
    pos = {d: i for i, d in enumerate(order)}
    D = np.zeros((n, n), dtype=bool)
    for d in range(n):
        r = D[d]
        r[d] = True
        for x in order[pos[d]:]:
            if not r[x]:
                continue
            for s in succ.get(x, ()):
                if s != x:
                    r[s] = True
    return D


def probe(label, maker):
    tot = 0
    s1_ok = s2_ok = owner_is_origin = 0
    ntok = []
    norig = []

    for ep in range(EPISODES):
        env = rollout(maker, 300 + ep)
        ct = env.pn.causal_trace
        n = len(ct.transition_history.get_action_transitions())
        if n == 0:
            continue
        ct.redistribute_rewards(scheme='cgae_dag', beta=0.0,
                                values=[0.0] * n, lam=1.0)
        succ, times, rewards = CAP['succ'], CAP['times'], CAP['rewards']
        tm = [0.0 if times[i] is None else float(times[i]) for i in range(n)]
        order = sorted(range(n), key=lambda i: (tm[i], i))
        D = closure(succ, order, n)

        for rv, anc, owner, origins, k in rewards:
            tot += 1
            ntok.append(k)
            norig.append(len(origins))
            if len(origins) == 1:
                s1_ok += 1
                if next(iter(origins)) == owner:
                    owner_is_origin += 1
            # S2: unique maximum under reachability == every member reaches
            # one common member, which must then be the owner.
            maxima = [a for a in anc if not any(a != b and D[a][b] for b in anc)]
            if len(maxima) == 1:
                s2_ok += 1

    f = lambda x: 100.0 * x / tot if tot else 0.0
    print("  %-14s rewards %5d | consumed tokens/firing %.2f | distinct origins "
          "%.3f | S1 single-origin %6.2f%% | origin==own(j) %6.2f%% | "
          "S2 unique-max %6.2f%%"
          % (label, tot, float(np.mean(ntok)), float(np.mean(norig)),
             f(s1_ok), f(owner_is_origin), f(s2_ok)))


ENVS = []
from multisite_env import make_multisite
ENVS.append(("multi-site", lambda: make_multisite(4, 1, 0, causal_rl=True,
                                                  allow_postpone=False)))
from ncopies_env import make_n_copies
for N in (4, 8):
    ENVS.append(("N=%d" % N,
                 (lambda N=N: make_n_copies(N, causal_rl=True, allow_postpone=True,
                                            causal_postpone_tokenflow=True))))
from envs import make_env
ENVS.append(("single comp.", lambda: make_env('s1_stoch_sequence', causal_rl=True,
                                              allow_postpone=True,
                                              causal_postpone_tokenflow=True)))

print("=" * 150)
print("WHY THE XI STEP HOLDS  (%d untrained rollouts per environment)" % EPISODES)
print("S1: the tokens a reward firing consumes have ONE nearest-producer "
      "decision. S1 => A(j) subset anc*(own(j)) => the Xi step.")
print("=" * 150)
for label, maker in ENVS:
    probe(label, maker)
print("=" * 150)

CausalTraces._cgae_structure = _orig

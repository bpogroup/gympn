r"""Does the Xi_d branch of the Proposition 1 proof have a live counterexample?

The proof argues that for a reward j with own(j) not in desc*(d),

    "since own(j) is the LATEST decision on j's lineage and the successor
     relation is time-forward, no decision of A(j) can lie in desc*(d) at all"

and concludes that no descendant of a_d lies on j's provenance, so r_j is
uninfluenced by a_d along the realized execution.

That step does not follow when a reward-emitting firing consumes tokens from
more than one ancestry. If r_j consumes token A produced (transitively) by
d' in desc*(d) and token B produced by an unrelated, later d'', then
own(j) = d'' is outside desc*(d) while d' is inside it: the reward IS a
provenance descendant of d, yet it lands in Xi_d, where the proof assumes it
away.

This script counts, per environment, how often that configuration occurs:

    BREAK(d, j)  <=>  A(j) ∩ desc*(d) != {}  AND  own(j) ∉ desc*(d)

reported as a share of (d, j) pairs and, more importantly, as a share of the
reward mass sitting in Xi_d. It also reports the in-degree distribution of
reward firings, since |A(j)| = 1 makes the configuration impossible.

Everything is read off the same objects `_cgae_structure` uses: the lineage
walk below is copied from it verbatim so the measurement is of the library's
own relation, not a reimplementation of it.

Run: python _diag_xi_gap.py [n_rollouts]
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

    # verbatim from _cgae_structure
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
        decs = lineage(tr.get('input_tokens', []), se[0] if se else None)
        decs = [d for d in decs if 0 <= d < n]
        if not decs:
            continue
        owner = max(decs, key=lambda d: (times[d] is not None, times[d] or 0.0))
        rewards.append((rv, set(decs), owner))

    CAP['succ'], CAP['w'], CAP['owned'], CAP['times'] = out
    CAP['rewards'] = rewards
    CAP['n'] = n
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
    pairs = brk = 0
    reach_pairs = [0, 0]
    reach_fail = []
    xi_mass = brk_mass = 0.0
    indeg = []
    multi_rewards = 0
    tot_rewards = 0

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

        for rv, anc, owner in rewards:
            tot_rewards += 1
            # HYPOTHESIS: own(j) is succ-reachable from every other member of
            # A(j). That, not the proof's time-order argument, is what would
            # make the Xi step sound.
            for a in anc:
                if a == owner:
                    continue
                reach_pairs[0] += 1
                if D[a][owner]:
                    reach_pairs[1] += 1
                else:
                    reach_fail.append((a, owner))
            indeg.append(len(anc))
            if len(anc) > 1:
                multi_rewards += 1
            for d in range(n):
                desc = D[d]
                if desc[owner]:
                    continue          # owner inside the closure: not in Xi_d
                # r_j is in Xi_d for this d (same component is checked by the
                # paper's own definition; the counterexample only needs the
                # ancestry intersection)
                pairs += 1
                xi_mass += rv
                if any(desc[a] for a in anc):
                    brk += 1
                    brk_mass += rv

    pct = 100.0 * brk / pairs if pairs else 0.0
    pctm = 100.0 * brk_mass / xi_mass if xi_mass else 0.0
    print("  %-14s rewards %5d (multi-ancestry %5.1f%%, mean |A(j)| %.2f) | "
          "(d,j) with own(j) outside closure: %7d | BREAK %7d = %5.2f%% of pairs, "
          "%5.2f%% of Xi mass"
          % (label, tot_rewards,
             100.0 * multi_rewards / tot_rewards if tot_rewards else 0.0,
             float(np.mean(indeg)) if indeg else 0.0,
             pairs, brk, pct, pctm))
    rp = 100.0 * reach_pairs[1] / reach_pairs[0] if reach_pairs[0] else 100.0
    print("  %-14s   -> own(j) succ-reachable from other members of A(j): "
          "%d/%d = %.3f%%" % (label, reach_pairs[1], reach_pairs[0], rp))
    return dict(env=label, pairs=pairs, brk=brk, reach=rp, pct=pct, pctm=pctm,
                multi=100.0 * multi_rewards / tot_rewards if tot_rewards else 0.0,
                meandeg=float(np.mean(indeg)) if indeg else 0.0)


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

print("=" * 140)
print("XI_d COUNTEREXAMPLE CENSUS  (%d untrained rollouts per environment)" % EPISODES)
print("BREAK(d,j): A(j) meets desc*(d) but own(j) does not -- the case the "
      "Proposition 1 proof rules out by a step that does not hold.")
print("=" * 140)
for label, maker in ENVS:
    probe(label, maker)
print("=" * 140)

CausalTraces._cgae_structure = _orig

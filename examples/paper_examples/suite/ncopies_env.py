"""N-copies scaling env: N fully INDEPENDENT single-stage match-assignment
tasks in one Petri net, one shared GNN policy, reward summed over copies.

Each copy i has its OWN arrival stream, queue, busy place, and dedicated
employee (no place is shared across copies), so the copies are causally
independent → the token-lineage union-find sees N separate components. This is
the regime where causal-component-factored credit (ccf) should beat PPO
DECISIVELY and by a margin that GROWS with N (LINEAGE_SPARSE_CORRECTION follow-up
/ the multi-agent credit-assignment story): PPO's advantage for a decision in
copy i is swamped by the independent reward noise of the other N-1 copies
(variance ~ N), while ccf factors it away (variance ~ 1). Predicted gap ~ sqrt(N).

Base copy: a type-0 employee; tasks of type 0 (match: fast service ~1) or type 1
(cross: slow ~3), arriving stochastically. The decision is which queued task to
serve; match-first maximizes throughput → reward. Stochastic arrivals + service
give each copy independent reward noise (the thing ccf filters and PPO cannot).
"""
import random
from simpn.simulator import SimToken
from gympn.simulator import GymProblem


def _start(c, r):
    base = 1 if c['task_type'] == r['code_employee'] else 3
    return [SimToken((c, r), delay=base + random.randint(0, 1))]


def _arrive(a):
    return [SimToken({'task_type': random.randint(0, 1)}, delay=1),
            SimToken({'task_type': random.randint(0, 1)})]


def _done(b):
    return [SimToken(b[-1])]


def make_n_copies(n=4, causal_rl=False, allow_postpone=True,
                  causal_postpone_tokenflow=False):
    ag = GymProblem(allow_postpone=allow_postpone, causal_rl=causal_rl,
                    causal_postpone_tokenflow=causal_postpone_tokenflow)
    for i in range(n):
        arrival = ag.add_var(f"arrival_{i}", var_attributes=['task_type'])
        waiting = ag.add_var(f"waiting_{i}", var_attributes=['task_type'])
        busy = ag.add_var(f"busy_{i}", var_attributes=['task_type', 'code_employee'])
        employee = ag.add_var(f"employee_{i}", var_attributes=['code_employee'])
        employee.put({'code_employee': 0})
        arrival.put({'task_type': 0})
        # warm start: one match + one cross task waiting → a real choice at t=0
        waiting.put({'task_type': 0})
        waiting.put({'task_type': 1})
        ag.add_event([arrival], [arrival, waiting], _arrive, name=f'arrive_{i}')
        ag.add_action([waiting, employee], [busy], behavior=_start, name=f"start_{i}")
        ag.add_event([busy], [employee], _done, name=f'done_{i}',
                     reward_function=lambda x: 1)
    return ag


# --------------------------------------------------------------------------
# HARD variant: 3 task types x 2 heterogeneous employees per copy. A genuine
# per-copy assignment (which task -> which specialist), not a binary match, so
# the policy is harder to learn within budget -> ccf lands BELOW the heuristic
# ceiling and the ccf-PPO gap has room to WIDEN with N instead of saturating.
# Employee 0 is fast for type 0, employee 1 for type 1; type 2 matches neither
# (always slow) so it must be deprioritized. Independence across copies is
# unchanged (own arrival/queue/busy/2-employee pool) -> still N components.
# --------------------------------------------------------------------------
def _start_hard(c, r):
    t, e = c['task_type'], r['code_employee']
    if t == e:      # matched specialist
        base = 1
    elif t == 2:    # no specialist -> generalist effort
        base = 3
    else:           # cross-assigned
        base = 4
    return [SimToken((c, r), delay=base + random.randint(0, 1))]


def _arrive_hard(a):
    return [SimToken({'task_type': random.randint(0, 2)}, delay=1),
            SimToken({'task_type': random.randint(0, 2)})]


def make_n_copies_hard(n=4, causal_rl=False, allow_postpone=False,
                       causal_postpone_tokenflow=False):
    ag = GymProblem(allow_postpone=allow_postpone, causal_rl=causal_rl,
                    causal_postpone_tokenflow=causal_postpone_tokenflow)
    for i in range(n):
        arrival = ag.add_var(f"arrival_{i}", var_attributes=['task_type'])
        waiting = ag.add_var(f"waiting_{i}", var_attributes=['task_type'])
        busy = ag.add_var(f"busy_{i}", var_attributes=['task_type', 'code_employee'])
        employee = ag.add_var(f"employee_{i}", var_attributes=['code_employee'])
        employee.put({'code_employee': 0})
        employee.put({'code_employee': 1})
        arrival.put({'task_type': 0})
        for t in (0, 1, 2):  # warm start: one of each type queued
            waiting.put({'task_type': t})
        ag.add_event([arrival], [arrival, waiting], _arrive_hard, name=f'arrive_{i}')
        ag.add_action([waiting, employee], [busy], behavior=_start_hard, name=f"start_{i}")
        ag.add_event([busy], [employee], _done, name=f'done_{i}',
                     reward_function=lambda x: 1)
    return ag


def ncopies_hard_heuristic(observable_net, tokens_comb, bindings=None):
    if not bindings:
        return None
    real = [b for b in bindings if isinstance(b, tuple) and b and isinstance(b[0], list)
            and b[0] != ['postpone']]
    if not real:
        return bindings[0]

    def matched(b):
        t = e = None
        for (place, tok) in b[0]:
            v = getattr(tok, 'value', tok)
            if isinstance(v, dict):
                if 'task_type' in v:
                    t = v['task_type']
                if 'code_employee' in v:
                    e = v['code_employee']
        return t is not None and t == e
    for b in real:
        if matched(b):
            return b
    return real[0]


# match-first heuristic (per copy): prefer a matching (type==0) binding.
def ncopies_heuristic(observable_net, tokens_comb, bindings=None):
    if not bindings:
        return None
    def is_match(b):
        try:
            for (place, tok) in b[0]:
                v = getattr(tok, 'value', tok)
                if isinstance(v, dict) and v.get('task_type') == 0:
                    return True
        except Exception:
            pass
        return False
    real = [b for b in bindings if isinstance(b, tuple) and b and isinstance(b[0], list)
            and b[0] != ['postpone']]
    if not real:
        return bindings[0]
    for b in real:
        if is_match(b):
            return b
    return real[0]


if __name__ == "__main__":
    # smoke: build a few sizes, check independence via the ccf estimator
    import sys, os, types, uuid
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    import numpy as np
    from gympn.environment import AEPN_Env

    def trace_check(n):
        random.seed(0); np.random.seed(0)
        pn = make_n_copies(n, causal_rl=True, allow_postpone=False)
        pn.length = 15
        for p in pn.places:
            for t in p.marking:
                setattr(t, '_id', str(uuid.uuid4()))
        pn.causal_trace.flush()
        sen = types.SimpleNamespace(_id="__initial__"); toks = [t for p in pn.places for t in p.marking]
        for t in toks:
            pn.causal_trace.register_token(t, sen, [], time=0)
        pn.causal_trace.register_transition(sen, [], toks, is_action=False, reward=0.0, time=0)
        env = AEPN_Env(pn); env.reset()
        for _ in range(40):
            k = len(env.pn.pn_actions)
            if k == 0:
                break
            _, _, d, _, _ = env.step(np.random.randint(k))
            if d:
                break
        ct = env.pn.causal_trace
        ccf = np.array(ct.redistribute_rewards(scheme='ccf', beta=0.5))
        mcq = np.array(ct.redistribute_rewards(scheme='mc_q', beta=0.5))
        sc = max(1e-9, float(np.mean(np.abs(mcq))))
        print(f"N={n:2d}: |ccf-mcq|/scale={np.mean(np.abs(ccf-mcq))/sc:.3f} "
              f"(0=one component/no factoring, higher=more factoring)  "
              f"sum ccf={ccf.sum():.1f} mcq={mcq.sum():.1f}")
    for n in (1, 2, 4, 8):
        trace_check(n)
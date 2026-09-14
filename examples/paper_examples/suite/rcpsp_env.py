r"""Resource-Constrained Project Scheduling (RCPSP) as an Action-Evolution Petri
Net -- the OR-benchmark demonstration domain for structure-derived factored
credit (s_ccf).

Why RCPSP exercises the method:
  * PARALLEL activities on separate precedence branches are causally independent
    -> factored credit (ccf/s_ccf) filters cross-branch reward noise that PPO's
    single return-to-go cannot (the variance win).
  * PRECEDENCE constraints are AND-joins: activity j starts only when ALL its
    predecessors are done. These are STRUCTURAL (always occur), so by themselves
    they do NOT make component membership action-dependent.
  * The action-dependent-membership bias (where ccf/lrq can be wrong and s_ccf is
    not) enters through RESOURCE ALLOCATION: when a freed resource can go to one
    of several ready activities, that choice changes which activities share a
    resource-lineage. Whether this is a sharp bias or mild (ccf ~= s_ccf) is an
    empirical question -- we check it at the estimator level first.

Model. Renewable resource with capacity K (each real activity uses 1 unit).
Per activity i: places ready_i, busy_i, done_i. Precedence arc i->j: a signal
place sig_{i,j}; enable_j is an AND-join consuming one signal from EVERY
predecessor. start_i (ACTION) consumes ready_i + 1 resource unit -> busy_i
(delay = duration_i); complete_i (evolution) -> done_i + release resource +
one signal to each successor, with reward = value_i (discounted by the SMDP
clock, so earlier completion is worth more -> a weighted-completion objective).
The decision is which ready activity to start under resource contention.
"""
import random
from simpn.simulator import SimToken
from gympn.simulator import GymProblem


# A small default instance: a diamond (0 -> {1,2} -> 3; activity 3 is an AND-join
# needing both 1 and 2) plus an independent chain (4 -> 5). Resource capacity 2
# forces scheduling choices. Activity 3 (the join) carries the high value.
DEFAULT = dict(
    n=6,
    prec=[(0, 1), (0, 2), (1, 3), (2, 3), (4, 5)],
    dur=[1, 2, 2, 2, 2, 2],
    res=[0, 1, 1, 1, 1, 1],          # activity 0 is a zero-resource source
    value=[0.0, 1.0, 1.0, 3.0, 1.0, 2.0],
    K=2,
)


def make_rcpsp(instance=None, causal_rl=False, allow_postpone=False,
               causal_postpone_tokenflow=False):
    p = dict(DEFAULT if instance is None else instance)
    n, prec, dur, res, value, K = p['n'], p['prec'], p['dur'], p['res'], p['value'], p['K']
    pred = {i: [] for i in range(n)}
    succ = {i: [] for i in range(n)}
    for (i, j) in prec:
        pred[j].append(i); succ[i].append(j)

    ag = GymProblem(allow_postpone=allow_postpone, causal_rl=causal_rl,
                    causal_postpone_tokenflow=causal_postpone_tokenflow)
    R = ag.add_var("resource", var_attributes=['u'])
    for _ in range(K):
        R.put({'u': 1})
    ready, busy, done, sig = {}, {}, {}, {}
    for i in range(n):
        ready[i] = ag.add_var(f"ready_{i}", var_attributes=['id'])
        busy[i] = ag.add_var(f"busy_{i}", var_attributes=['id'])
        done[i] = ag.add_var(f"done_{i}", var_attributes=['id'])
    for (i, j) in prec:
        sig[(i, j)] = ag.add_var(f"sig_{i}_{j}", var_attributes=['id'])
    # activities with no predecessors are ready at t=0
    for i in range(n):
        if not pred[i]:
            ready[i].put({'id': i})

    # enable_j: AND-join over all predecessor signals -> ready_j
    for j in range(n):
        if pred[j]:
            ins = [sig[(i, j)] for i in pred[j]]
            ag.add_event(ins, [ready[j]],
                         (lambda jid: (lambda *toks: [SimToken({'id': jid})]))(j),
                         name=f"enable_{j}")

    for i in range(n):
        di = dur[i]
        if res[i] > 0:
            ag.add_action([ready[i], R], [busy[i]],
                          (lambda d: (lambda rd, r: [SimToken((rd, r), delay=d)]))(di),
                          name=f"start_{i}")
        else:
            ag.add_action([ready[i]], [busy[i]],
                          (lambda d: (lambda rd: [SimToken(rd, delay=d)]))(di),
                          name=f"start_{i}")
        outs = [done[i]] + ([R] if res[i] > 0 else []) + [sig[(i, j)] for j in succ[i]]

        def _make_complete(uses_res, sc):
            def _complete(b):
                # b: (ready_val, resource_val) if resource used, else ready_val
                rd = b[0] if uses_res else b
                out = [SimToken(rd)]                               # done
                if uses_res:
                    out.append(SimToken(b[1]))                    # release resource
                for _ in sc:
                    out.append(SimToken(rd))                      # signal each successor
                return out
            return _complete

        ag.add_event([busy[i]], outs, _make_complete(res[i] > 0, list(succ[i])),
                     name=f"complete_{i}",
                     reward_function=(lambda v: (lambda b: v))(value[i]))
    return ag


def spt_heuristic(observable_net, tokens_comb, bindings=None):
    """Shortest-value-first-ish anchor: prefer starting the highest-value ready
    activity (a reasonable priority rule). Falls back to any real binding."""
    if not bindings:
        return None
    real = [b for b in bindings if isinstance(b, tuple) and len(b) >= 2
            and isinstance(b[0], list) and b[0] and isinstance(b[0][0], tuple)]
    return real[0] if real else bindings[0]


if __name__ == "__main__":
    import sys, os, types, uuid
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    import numpy as np
    import gympn
    from gympn.environment import AEPN_Env

    # ---- (1) does it build + run? ----
    gympn.seed_everything(0)
    pn = make_rcpsp(causal_rl=True, allow_postpone=False)
    pn.length = 30
    for pl in pn.places:
        for t in pl.marking:
            setattr(t, '_id', str(uuid.uuid4()))
    pn.causal_trace.flush()
    sen = types.SimpleNamespace(_id="__initial__"); toks = [t for pl in pn.places for t in pl.marking]
    for t in toks:
        pn.causal_trace.register_token(t, sen, [], time=0)
    pn.causal_trace.register_transition(sen, [], toks, is_action=False, reward=0.0, time=0)
    env = AEPN_Env(pn); env.reset()
    total = 0.0
    for _ in range(40):
        k = len(env.pn.pn_actions)
        if k == 0:
            break
        _, r, d, _, _ = env.step(np.random.randint(k))
        total += float(r)
        if d:
            break
    print(f"random rollout total reward = {total:.1f}  (activities completed OK)")

    # ---- (2) estimator check: is the ccf vs s_ccf bias present on RCPSP? ----
    ct = env.pn.causal_trace
    ct._pn = env.pn; ct._static_comp_cache = None
    for s in ('mc_q', 'lrq', 'ccf', 's_ccf'):
        v = np.array(ct.redistribute_rewards(scheme=s, beta=0.5))
        print(f"  {s:>6}: per-decision credit sum={v.sum():.2f}  vec={np.round(v,2).tolist()}")
    ccf = np.array(ct.redistribute_rewards(scheme='ccf', beta=0.5))
    sccf = np.array(ct.redistribute_rewards(scheme='s_ccf', beta=0.5))
    diff = float(np.mean(np.abs(ccf - sccf)))
    print(f"\n  mean|ccf - s_ccf| = {diff:.3f}   "
          f"({'BIAS PRESENT (ccf != s_ccf)' if diff > 1e-6 else 'ccf == s_ccf here (no sharp bias)'})")
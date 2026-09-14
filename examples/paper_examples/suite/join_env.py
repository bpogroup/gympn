r"""FAILED training capstone (documented negative). The rigorous, working
demonstration of ccf's join-bias and the s_ccf fix is `assembly_probe.py`
(estimator-level, 5 motifs, library-verified). Keep this only as a lab record of
why a *trained* demonstration was not obtained.

Goal it was chasing
-------------------
A trainable AND-join env where ccf's action-dependent component-membership bias
makes it LEARN the suboptimal policy, while s_ccf (static components) and PPO
learn the optimum. The mechanism (which IS the real finding):

  d1 on part1 : route into the join (route_asm) -> small r_join
                       OR standalone (route_sa) -> medium r1     (r1 > r_join)
  d2 on part2 : route2 -> jw2 (to join) + p2priv --[priv2]--> r2 (private)
  join(jw1,jw2, case_id match) -> r_join

Only r_join vs r1 depends on d1 (r2 fires either way), so the OPTIMUM is always
route_sa. But ccf credits d1 with r2 ONLY when the join fires (r2 enters d1's
*realized* component via the join), so ccf ranks route_asm (=r_join+r2) above
route_sa (=r1) and converges to the SUBOPTIMAL join. s_ccf's *static* component
always contains r2, so r2 cancels and it prefers route_sa; PPO is unbiased too.
NOTE: d2 MUST be a decision (an action, not an event) -- otherwise r2 has no
action-ancestor, is a policy-independent constant, and BOTH ccf and s_ccf drop
it (no coupling, no flip). This is `route2` below, added late.

Why it never produced a clean training result (4 configs, 4 different failures)
------------------------------------------------------------------------------
  1. `make_join`, r2=10 (streaming): variance WALL -- r2 dominates the return, so
     the decision (gap r1-r_join) is ~1% of it; no method converges.
  2. `make_join_pairs` (fixed independent pairs): SMDP all-zero-sojourn ERROR --
     all pairs available at t=0, so every decision has the same clock.
  3. `make_join` balanced (r_join=1,r1=2,r2=3) but WITHOUT d2: no flip (r2 is a
     constant) -- ccf == s_ccf == ppo == ~optimum.
  4. `make_join` balanced WITH d2 (route2): PPO itself falls below random; credit
     schemes do not separate (noise).
The constraints fight: learnable + d1&d2&join-coupling + staggered timing (SMDP)
+ balanced rewards + per-pair-independent components. Not resolved here.

`make_join`      : streaming, matched-case_id arrivals, route2=d2 action.
`make_join_pairs`: fixed independent pairs (hit the SMDP timing error).
Both are experimental negatives -- NOT part of the paper.
"""
import random
from simpn.simulator import SimToken
from gympn.simulator import GymProblem


def _arrive(a):
    cid = random.randint(0, 10 ** 9)
    return [SimToken(a, delay=1),               # recycle the arrival token
            SimToken({'case_id': cid}),          # part1
            SimToken({'case_id': cid})]          # part2 (matched)


def make_join(causal_rl=False, allow_postpone=False,
              causal_postpone_tokenflow=False, r_join=1.0, r1=5.0, r2=10.0,
              d2_decision=True):
    ag = GymProblem(allow_postpone=allow_postpone, causal_rl=causal_rl,
                    causal_postpone_tokenflow=causal_postpone_tokenflow)
    arrival = ag.add_var("arrival", var_attributes=['case_id'])
    part1 = ag.add_var("part1", var_attributes=['case_id'])
    part2 = ag.add_var("part2", var_attributes=['case_id'])
    jw1 = ag.add_var("jw1", var_attributes=['case_id'])
    jw2 = ag.add_var("jw2", var_attributes=['case_id'])
    p2priv = ag.add_var("p2priv", var_attributes=['case_id'])
    sa1 = ag.add_var("sa1", var_attributes=['case_id'])
    done = ag.add_var("done", var_attributes=['case_id'])
    arrival.put({'case_id': -1})
    # warm start: one pair ready at t=0 so a decision is available immediately
    part1.put({'case_id': 0}); part2.put({'case_id': 0})

    ag.add_event([arrival], [arrival, part1, part2], _arrive, name='arrive')
    # part2's routing: a DECISION d2 (single option) when d2_decision, else an
    # automatic event. d2 is needed for r2 to have an action-ancestor (else the
    # ccf flip vanishes) -- but toggling it isolates its effect on training.
    if d2_decision:
        ag.add_action([part2], [jw2, p2priv], behavior=lambda p: [SimToken(p), SimToken(p)], name='route2')
    else:
        ag.add_event([part2], [jw2, p2priv], lambda p: [SimToken(p), SimToken(p)], name='split2')
    ag.add_event([p2priv], [done], lambda p: [SimToken(p)],
                 name='priv2', reward_function=lambda b: r2)
    # the decision: route part1 to the join (A) or to standalone (B)
    ag.add_action([part1], [jw1], behavior=lambda c: [SimToken(c, delay=1)], name='route_asm')
    ag.add_action([part1], [sa1], behavior=lambda c: [SimToken(c, delay=1)], name='route_sa')
    ag.add_event([sa1], [done], lambda p: [SimToken(p)],
                 name='sa_done', reward_function=lambda b: r1)
    # join only fires for a MATCHED pair (same case_id)
    ag.add_event([jw1, jw2], [done], lambda a, b: [SimToken(a)], name='join',
                 guard=lambda a, b: a['case_id'] == b['case_id'],
                 reward_function=lambda a, b: r_join)
    return ag


def make_join_pairs(n_pairs=4, causal_rl=False, allow_postpone=False,
                    causal_postpone_tokenflow=False, r_join=1.0, r1=2.0, r2=3.0):
    """N INDEPENDENT join-pairs, fixed at t=0 (no streaming). Each pair is a clean
    route decision on part1 (join vs standalone); part2 always emits private r2.
    Rewards are SMALL and comparable so the decision (r1 vs r_join) is a large
    fraction of the return -> learnable, while still r2 > r1-r_join so ccf flips.
    Optimal = all standalone; ccf's bias should push it to the join trap."""
    ag = GymProblem(allow_postpone=allow_postpone, causal_rl=causal_rl,
                    causal_postpone_tokenflow=causal_postpone_tokenflow)
    for i in range(n_pairs):
        part1 = ag.add_var(f"part1_{i}", var_attributes=['case_id'])
        part2 = ag.add_var(f"part2_{i}", var_attributes=['case_id'])
        jw1 = ag.add_var(f"jw1_{i}", var_attributes=['case_id'])
        jw2 = ag.add_var(f"jw2_{i}", var_attributes=['case_id'])
        p2priv = ag.add_var(f"p2priv_{i}", var_attributes=['case_id'])
        sa1 = ag.add_var(f"sa1_{i}", var_attributes=['case_id'])
        done = ag.add_var(f"done_{i}", var_attributes=['case_id'])
        part1.put({'case_id': i}); part2.put({'case_id': i})
        ag.add_event([part2], [jw2, p2priv], lambda p: [SimToken(p), SimToken(p)], name=f'split2_{i}')
        ag.add_event([p2priv], [done], lambda p: [SimToken(p)], name=f'priv2_{i}',
                     reward_function=lambda b: r2)
        ag.add_action([part1], [jw1], behavior=lambda c: [SimToken(c, delay=1)], name=f'route_asm_{i}')
        ag.add_action([part1], [sa1], behavior=lambda c: [SimToken(c, delay=1)], name=f'route_sa_{i}')
        ag.add_event([sa1], [done], lambda p: [SimToken(p)], name=f'sa_done_{i}',
                     reward_function=lambda b: r1)
        ag.add_event([jw1, jw2], [done], lambda a, b: [SimToken(a)], name=f'join_{i}',
                     reward_function=lambda a, b: r_join)
    return ag


def _pick(bindings, want):
    """Return the binding whose action transition contains `want`, else None."""
    for b in bindings:
        if isinstance(b, tuple) and len(b) > 2 and b[2] is not None:
            if want in str(getattr(b[2], '_id', getattr(b[2], 'name', ''))):
                return b
    return None


def optimal_heuristic(observable_net, tokens_comb, bindings=None):
    """Always standalone (route_sa): r1 > r_join, and r2 fires regardless."""
    return _pick(bindings or [], 'route_sa') or (bindings[0] if bindings else None)


def trap_heuristic(observable_net, tokens_comb, bindings=None):
    """Always join (route_asm): the suboptimal policy ccf is biased toward."""
    return _pick(bindings or [], 'route_asm') or (bindings[0] if bindings else None)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    import numpy as np
    import gympn
    from gympn.solvers import RandomSolver, HeuristicSolver

    LEN = 20
    def avg(make_solver, n=15):
        rs = []
        for s in range(n):
            gympn.seed_everything(100 + s)
            rs.append(float(make_join(allow_postpone=False).testing_run(solver=make_solver(), length=LEN)))
        return float(np.mean(rs)), float(np.std(rs))
    r = avg(lambda: RandomSolver())
    opt = avg(lambda: HeuristicSolver(optimal_heuristic))
    trap = avg(lambda: HeuristicSolver(trap_heuristic))
    print(f"random          : {r[0]:6.1f} +/- {r[1]:.1f}")
    print(f"optimal (sa)    : {opt[0]:6.1f} +/- {opt[1]:.1f}   <- best")
    print(f"trap (join, asm): {trap[0]:6.1f} +/- {trap[1]:.1f}   <- ccf is biased toward this")
    print(f"\noptimal-trap gap = {opt[0]-trap[0]:+.1f}  (ccf that flips lands near the trap)")
"""Potential-based reward shaping from Petri-net topology (Ng, Harada & Russell 1999).

For any function Phi of state, adding

    F(s, a, s') = e^{-beta*tau} * Phi(s') - Phi(s)

to each step's reward provably does not change the optimal policy of any (S)MDP: the
shaping term telescopes to a constant (Phi(s_0) - discount_prod_T * Phi(s_T)) along any
trajectory, independent of the action sequence taken. This is the SMDP generalization
(per-sojourn discount e^{-beta*tau}, tau = elapsed simulator clock time between
decisions) of the classical fixed-gamma theorem -- the same discount convention already
used throughout gympn (see gympn.data.TrajectoryBuffer._smdp_discounts).

Phi here is built purely from Petri-net TOPOLOGY (place/transition/reward-function
structure) plus live token COUNTS -- never token attribute values -- so it is general
across resource-pool, loop/rework, join, and exclusive-choice AEPN shapes alike: the
same generality discipline gympn.conflict_graph already enforces, and the property that
"same-case vs cross-case" credit-restriction rules (see gympn.causal_traces's
_pure_contested_reward_types) turned out NOT to have in general.

SCOPE. Shaping mutates the raw per-step reward BEFORE any causal_scheme sees it
(injected in gympn.agents.Agent.run_episode, around its env.step() call). It fully
reaches the standard PPO path and every scheme whose advantage is built from
smdp_gae/discount_returns on raw rewards (lcv's adv_base, lva's adv_ep, the causal_mu
GAE hedge, cf's un-forked-step baseline). It is a DOCUMENTED NO-OP for schemes whose
credit is a lineage Q-sample that overrides the buffer's returns/advantages directly
(lrq, lrq2, lrq2c, ccf, s_ccf, ls_hca, lrq3, lqi) -- those read reward from
pn.causal_trace's independently-recorded transition history, never the locally-mutated
step reward. Safe either way (never corrupts a lineage credit); extending shaping into
those paths is future work, not this module's job.
"""


def _place_reward_hops(pn):
    """Multi-source BFS: place_id -> minimum transition-hop distance to the
    nearest reward-bearing transition.

    Walks BACKWARD from every reward transition via each place's PRODUCERS
    (transitions that emit it): a reward transition's own incoming places get
    hop=1, their producers' incoming places get hop=2, and so on. Places with
    no path to any reward transition are simply absent from the result (their
    weight is 0 -- see _place_weights). Pure PN topology: actions, events, and
    arcs only, never token values.
    """
    reward_types = set(pn.reward_functions.keys())
    trans = list(pn.actions) + list(pn.events)

    # producers[place_id] = transitions that OUTPUT this place (for the backward walk)
    producers = {}
    for t in trans:
        for p in t.outgoing:
            producers.setdefault(p._id, []).append(t)

    hops = {}
    frontier = []
    for t in trans:
        if t._id in reward_types:
            for p in t.incoming:
                if p._id not in hops:
                    hops[p._id] = 1
                    frontier.append(p._id)

    depth = 1
    while frontier:
        next_frontier = []
        for pid in frontier:
            for t in producers.get(pid, []):
                for p in t.incoming:
                    if p._id not in hops:
                        hops[p._id] = depth + 1
                        next_frontier.append(p._id)
        frontier = next_frontier
        depth += 1
    return hops


def _place_weights(pn, decay):
    """hop map -> {place_id: decay**hop}."""
    return {pid: decay ** h for pid, h in _place_reward_hops(pn).items()}


def get_place_weights(pn, decay=0.9):
    """Cached accessor: place_id -> decay**(hop to nearest reward transition).

    Cached on the `pn` instance itself (`pn._phi_place_weights`), not on
    `causal_trace` -- causal_trace only exists when causal_rl=True, but
    shaping must work under plain PPO too. This mirrors
    CausalTraces._static_comp_cache's per-pn-instance lifecycle: AEPN_Env's
    reset deepcopies pn fresh each episode, so this is effectively rebuilt
    once per episode at negligible one-time cost, same precedent as that
    cache. Keyed on `decay` too, so changing it invalidates the cache.
    """
    cache = getattr(pn, '_phi_place_weights', None)
    if cache is not None and cache[0] == decay:
        return cache[1]
    weights = _place_weights(pn, decay)
    pn._phi_place_weights = (decay, weights)
    return weights


def topology_potential(pn, decay=0.9, cap=None):
    """Phi(s) = sum_p weight(p) * min(len(p.marking), cap).

    weight(p) = decay**hop(p) if p can structurally reach a reward transition,
    else 0 (see get_place_weights). A token near a reward weighs close to 1; a
    token many stages upstream weighs decay**hop; a token that has already
    yielded its reward sits in a sink place and weighs 0. This moves visibly as
    work progresses through even a single-reward-type pipeline, unlike a flat
    reward-type-reachability count or an unweighted backlog count (both are
    near-constant on envs with only 1-2 reward types).

    `cap` (None by default = uncapped, the original behaviour) caps each
    PLACE's own contribution before summing. Motivation (see
    causal-stability-suite memory, 2026-08-01 s1 phi_coef dial): a
    queue-like place fed by EXOGENOUS arrivals (action-independent, e.g. s1's
    waiting1) can hold an unboundedly large and highly variable token count,
    so its uncapped contribution weight(p)*count(p) can swing far more from
    arrival-timing noise than from anything the agent's own actions did --
    still theorem-safe (policy-invariance holds for ANY Phi, capped or not),
    but a large, noisy per-step shaping term is a harder value-regression
    target at a finite training budget (confirmed empirically: phi_coef=1.0
    hurt badly, phi_coef=0.2 was roughly neutral, on the UNCAPPED Phi). A
    small integer cap (e.g. 1) turns Phi from "total weighted backlog
    VOLUME" into "which stages currently have ANY work" -- still purely
    topological/count-based (no token-value semantics), but insensitive to
    exactly how deep a queue got, which is the axis arrival timing (not the
    policy) controls.
    """
    weights = get_place_weights(pn, decay=decay)
    total = 0.0
    for p in pn.places:
        w = weights.get(p._id)
        if w:
            n = len(p.marking)
            if cap is not None:
                n = min(n, cap)
            total += w * n
    return total

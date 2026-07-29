"""Direction A — AlphaZero-over-AEPN: PUCT search as the learner
(AEPN_NATIVE_LEARNING.md §3).

The white-box net is a perfect, resettable simulator, so instead of a
model-free policy gradient we *plan exactly*: at every decision state run a
PUCT tree search over the enabled bindings, using the GNN policy as the prior
and the GNN critic as the leaf evaluator (no random rollouts — a uniform tail
was the X15 DCL failure), and distil the search-improved visit-count
distribution back into the policy (the AlphaZero improvement operator). This
generalizes what cfpk (a depth-1 fork) and the DCL planner (a depth-h *flat*
rollout) already do into a reusable tree with a learned prior+value.

Contract. ``mcts_target_pi`` returns ``(target_pi, stats, enabled)`` exactly
like ``dcl_planner.compute_target_pi``, so ``DCLAgent``'s distillation loop,
buffer, and training consume it unchanged (see ``agents_mcts.MCTSAgent``).

Reuse. Snapshot/restore + CRN are the env's own ``get_state``/``set_state``/
``set_seed`` (validated by cfpk and DCL). Binding signatures (transition id +
consumed token ids) key the tree so it survives stochastic divergence and
positional-index churn — the same device the DCL planner uses.

Direction B hook. ``conflict_adj`` (built from ``conflict_graph.analyze``)
lets the search treat a state where all enabled bindings structurally commute
(no shared input place, no shared token) as a *forced* node: it collapses to
the single highest-prior binding instead of branching, concentrating the
simulation budget on genuine contested decisions. This is the general,
formalism-level "where to spend search" object the Direction B probe
identified (§4.1). Toggle with ``cfg.conflict_gate`` to measure its value.

SMDP. Rewards are discounted e^{-beta*(t - t0)} on the sim clock, the
project-wide postpone fix; leaf values are discounted the same way.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from simpn.simulator import SimVarTime
except Exception:  # pragma: no cover - defensive
    SimVarTime = ()  # type: ignore

# Reuse the validated lineage descendant machinery from the fork stack.
from gympn.counterfactual import _lineage_return, descendants_of  # noqa: F401

_TRACE_ERR = (
    "[mcts] rollout/lineage backup reads rewards from the causal trace; build "
    "the ENV with causal_rl=True (trace recording is gated on it in "
    "simulator.fire, and causal_trace only exists when it is set).")


@dataclass
class MCTSConfig:
    n_simulations: int = 64
    """PUCT simulations per decision. AlphaZero works at 50-200 with a decent
    prior; keep small — each simulation is a suffix of env.step calls."""
    c_puct: float = 1.5
    """Exploration constant in PUCT. Q is min-max normalized inside the tree so
    this stays env-agnostic."""
    lookahead: float = 8.0
    """Search horizon in CLOCK units (like cfpk's lookahead), not decisions:
    beyond t0+lookahead the value net bootstraps. Clock-based so a state's
    truncation horizon is a function of the state alone — which is what makes
    the transposition table (coupling truncation) exact: a shared node's value
    cannot depend on which path/depth reached it. Essential on delayed-reward
    envs (the X15 lesson)."""
    max_depth: int = 64
    """Hard recursion-depth safety cap (in decisions). The real horizon is
    ``lookahead`` in clock units; this only guards against pathological nets
    that fire many zero-time decisions without advancing the clock."""
    coupling_truncate: bool = True
    """Coupling truncation (AEPN_NATIVE_LEARNING §3.1 / rethink §7.2): share
    tree nodes by state fingerprint (canonical marking + in-flight token times
    + clock). Two branches that reach the same state have coincided — their
    futures are identical under CRN — so they become ONE node, searched once.
    This is the faithful MCTS analog of cfpk's pairwise CRN coupling, and the
    place the concurrency the conflict graph flags actually pays off. Pure
    node-sharing loses no fidelity."""
    couple_min_visits: int = 0
    """If >0, additionally SHORT-CIRCUIT re-entry into an already-coupled node
    once it has this many visits: return its cached mean value instead of
    stepping deeper, saving the remaining descent's env.step calls. Exact under
    a deterministic continuation; on stochastic envs it reuses a mean over
    determinizations, so raise the threshold. 0 = node-sharing only (no
    value-reuse short-circuit, no fidelity loss)."""
    beta: float = 0.5
    """SMDP time-discount rate (project default 0.5). 0 = flat undiscounted."""
    temperature: float = 1.0
    """Visit-count temperature for the distilled target: pi ∝ N^(1/temp)."""
    value_bootstrap: bool = True
    """Evaluate leaves / depth-cap with the critic. Off => leaf value 0."""
    use_crn: bool = True
    """Reseed each simulation (common random numbers across the search)."""
    dirichlet_alpha: float = 0.0
    """Root Dirichlet exploration noise weight base (0 = off). AlphaZero adds
    noise to the root prior so search explores under-visited moves."""
    dirichlet_eps: float = 0.25
    conflict_gate: bool = True
    """Direction B: collapse structurally-forced (all-commuting) nodes to a
    single child so simulations concentrate on genuine contested decisions."""
    rollout_backup: bool = False
    """Use rollout-based per-decision backup (PUCT in-tree, policy rollout to
    the clock horizon, no value-net leaf) instead of the default AlphaZero
    value-net-leaf recursion. Each in-tree decision edge is scored by its
    return over the whole rollout suffix. Requires a causal-trace env
    (causal_rl=True) since rewards are read from the trace. This is the fair
    baseline for the lineage-backup A/B (whole return-to-go = mc_q inside the
    tree)."""
    lineage_backup: bool = False
    """THE LINEAGE CONTRIBUTION (AEPN_NATIVE_LEARNING §3.3). Implies
    rollout_backup. Credit each in-tree decision edge only by the rewards
    CAUSALLY DESCENDED from that decision's produced tokens (lineage-restricted
    return), instead of the whole rollout suffix. This is the tree-search lift
    of lrq-vs-mc_q (28W/0L): a strictly lower-variance, causally-targeted
    backup that standard MCTS cannot express. The opportunity-cost bias that
    sank cfpl is carried here by the SIBLING COMPARISON (Q of the alternative),
    not by each rollout's return — lineage supplies variance, the tree supplies
    bias. Needs causal_rl=True."""


# --------------------------------------------------------------------------- #
# Binding identity + structural conflict (shared idiom with the DCL planner)
# --------------------------------------------------------------------------- #
_POSTPONE = ("__postpone__",)


def _freeze(v):
    """Recursively convert a token value into a hashable key."""
    if isinstance(v, dict):
        return tuple(sorted((k, _freeze(x)) for k, x in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(_freeze(x) for x in v)
    return v


def _tok_key(var, tok):
    """Identity of a consumed token. Uses the unique ``_id`` when the env
    stamps one (causal_rl), else a value+place+time key. The value key makes
    interchangeable same-colour tokens share a signature — symmetry-aware and
    exactly what we want, since they lead to identical futures under CRN."""
    tid = getattr(tok, "_id", None)
    if tid is not None:
        return tid
    return (getattr(var, "_id", None), _freeze(getattr(tok, "value", tok)),
            getattr(tok, "time", None))


def _sig(binding):
    """Stable identity of a binding: (transition_id, sorted consumed-token keys).
    Postpone -> sentinel; anything unparseable -> None (skipped). Works with or
    without causal_rl token ids."""
    if binding is None:
        return None
    if isinstance(binding[0], list) and binding[0] == ["postpone"]:
        return _POSTPONE
    try:
        tr = getattr(binding[2], "_id", None)
        toks = tuple(sorted((_tok_key(var, tok) for var, tok in binding[0]),
                            key=repr))
        return (tr, toks)
    except Exception:
        return None


def _tr_tokens(sig):
    if sig is None or sig == _POSTPONE:
        return None, frozenset()
    return sig[0], frozenset(sig[1])


def build_conflict_adjacency(pn) -> Dict[str, set]:
    """{transition_id -> set of structurally-conflicting transition_ids} from
    the static net (share an input place). Built once per env."""
    from gympn.conflict_graph import analyze
    a = analyze(pn)
    adj: Dict[str, set] = {}
    for x, y in a.conflict_edges:
        adj.setdefault(x, set()).add(y)
        adj.setdefault(y, set()).add(x)
    return adj


def _conflicts(sig_a, sig_b, adj) -> bool:
    """Do two enabled bindings genuinely contend (so their order is a real
    decision)? Postpone always counts; same transition counts; a shared token
    or a static shared-input-place edge counts. Only when NO pair conflicts do
    all enabled bindings commute and the node is 'forced'."""
    tr_a, tk_a = _tr_tokens(sig_a)
    tr_b, tk_b = _tr_tokens(sig_b)
    if tr_a is None or tr_b is None:      # postpone / unknown -> real choice
        return True
    if tr_a == tr_b:                      # two bindings of one transition
        return True
    if tk_a & tk_b:                       # compete for a token in this marking
        return True
    if adj and tr_b in adj.get(tr_a, ()):  # static shared-input-place conflict
        return True
    return False


def _is_forced(sigs, adj) -> bool:
    """A node is forced (not a real decision) iff no two enabled bindings
    conflict — every ordering commutes to the same marking."""
    for i in range(len(sigs)):
        for j in range(i + 1, len(sigs)):
            if _conflicts(sigs[i], sigs[j], adj):
                return False
    return True


def state_fingerprint(pn):
    """Canonical identity of a PN state for coupling truncation: the multiset of
    (place_id, frozen token value, token time) over the marking, plus the clock.

    - Symmetry-canonical: interchangeable same-colour tokens collapse (the
      value key, not a unique id), so reorderings of independent firings that
      reach the same marking hash equal — that is the coupling.
    - Includes token TIMES (in-flight delays) and the CLOCK, so two states hash
      equal only when their futures AND their discount-to-t0 are identical.
    """
    items = []
    for p in getattr(pn, "places", []):
        if SimVarTime and isinstance(p, SimVarTime):
            continue  # the clock is captured separately below
        pid = getattr(p, "_id", None)
        for tok in getattr(p, "marking", []):
            items.append((pid, _freeze(getattr(tok, "value", tok)),
                          getattr(tok, "time", None)))
    items.sort(key=repr)
    return (tuple(items), float(getattr(pn, "clock", 0.0)))


def _whole_return_to_go(ct, idx, t0, beta):
    """Discounted sum of ALL rewards fired at or after trace position ``idx``
    (the whole rollout suffix from a decision), discounted to t0. This is the
    mc_q estimator for a decision — the fair no-lineage baseline for the
    lineage-restricted backup."""
    total = 0.0
    for tr in ct.transition_history.transitions[idx:]:
        rew = tr.get("reward", 0.0)
        if not rew:
            continue
        t = tr.get("time")
        dt = max(0.0, float(t) - t0) if t is not None else 0.0
        total += rew * math.exp(-beta * dt)
    return total


# --------------------------------------------------------------------------- #
# Tree node
# --------------------------------------------------------------------------- #
class _Node:
    __slots__ = ("expanded", "P", "N", "W", "n_total", "w_total")

    def __init__(self):
        self.expanded = False
        self.P: Dict[tuple, float] = {}   # prior per child signature
        self.N: Dict[tuple, int] = {}     # visit count per child edge
        self.W: Dict[tuple, float] = {}   # total backed-up value per edge
        self.n_total = 0                  # visits to THIS state (for coupling)
        self.w_total = 0.0                # summed return-from-here (disc to t0)

    def q(self, sig):
        n = self.N.get(sig, 0)
        return (self.W[sig] / n) if n > 0 else None

    def value(self):
        return (self.w_total / self.n_total) if self.n_total > 0 else 0.0


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
@torch.no_grad()
def mcts_target_pi(env, state_graph, actor, cfg: MCTSConfig,
                   conflict_adj: Optional[Dict[str, set]] = None,
                   critic=None):
    """Run PUCT from the current decision state; return the DCL contract
    ``(target_pi, stats, enabled)``. ``enabled`` are env positions
    (``range(len(pn.pn_actions))``); ``target_pi`` and ``stats['means']`` are
    aligned to it. ``actor``/``critic`` may be None (uniform prior / zero
    value)."""
    enabled: List[int] = env.enabled_actions(state_graph)
    A = len(enabled)
    if A == 0:
        return np.array([], dtype=np.float32), {"means": [], "stds": []}, enabled
    if A == 1:
        return (np.array([1.0], dtype=np.float32),
                {"means": [0.0], "stds": [0.0], "sims": 0}, enabled)

    root_snapshot = env.get_state()
    t0 = float(getattr(env.pn, "clock", 0.0))
    root = _Node()
    # Coupling truncation: when on, nodes are shared by state fingerprint so the
    # tree is a DAG — two branches that reach the same state become one node,
    # searched once. When off, a plain persistent tree keyed by (parent, action).
    trans_table: Dict[tuple, _Node] = {}
    child_table: Dict[int, Dict[tuple, _Node]] = {}
    counters = {"nodes": 1, "transpositions": 0, "truncations": 0, "steps": 0}

    vmin = [math.inf]
    vmax = [-math.inf]

    def _disc(clock):
        return math.exp(-cfg.beta * max(0.0, float(clock) - t0)) if cfg.beta > 0 else 1.0

    def _value(obs):
        if critic is None or not cfg.value_bootstrap:
            return 0.0
        try:
            return float(critic(obs).reshape(-1).mean().item())
        except Exception:
            return 0.0

    def _priors(obs, binds):
        if actor is None:
            return None
        try:
            p = actor(obs).reshape(-1)
            return p
        except Exception:
            return None

    def _expand(node, obs):
        node.expanded = True
        binds = list(getattr(env.pn, "pn_actions", []))
        probs = _priors(obs, binds)
        sig_pos, priors = [], {}
        for j, b in enumerate(binds):
            sg = _sig(b)
            if sg is None:
                continue
            sig_pos.append(sg)
            priors[sg] = float(probs[j]) if (probs is not None and j < probs.numel()) else 1.0
        if not sig_pos:
            return
        # Direction B gate: collapse a forced (all-commuting) node.
        if cfg.conflict_gate and len(sig_pos) > 1 and _is_forced(sig_pos, conflict_adj):
            keep = max(sig_pos, key=lambda s: priors[s])
            sig_pos = [keep]
        z = sum(priors[s] for s in sig_pos) or 1.0
        node.P = {s: priors[s] / z for s in sig_pos}
        node.N = {s: 0 for s in sig_pos}
        node.W = {s: 0.0 for s in sig_pos}

    def _current_positions():
        cur = {}
        for j, b in enumerate(getattr(env.pn, "pn_actions", [])):
            sg = _sig(b)
            if sg is not None:
                cur[sg] = j
        return cur

    def _select(node, cur, root_noise=None):
        total_n = sum(node.N.values())
        sqrt_n = math.sqrt(total_n + 1)
        rng = (vmax[0] - vmin[0]) if vmax[0] > vmin[0] else 0.0

        def qnorm(sig):
            q = node.q(sig)
            if q is None:
                return 0.0   # FPU: unvisited scored purely by the prior*U term
            if rng <= 0:
                # No value spread yet (all backed-up returns equal — common early
                # on delayed-reward envs where rollouts see no reward). Return the
                # SAME score as an unvisited node (0.0), so selection is driven by
                # the prior*U term and visits SPREAD. Returning 0.5 here instead
                # made a visited-but-zero action beat every unvisited one, so the
                # first-visited action captured all simulations and the target
                # collapsed to one action (measured: one-hot postpone on s1).
                return 0.0
            return (q - vmin[0]) / rng

        best, best_score = None, -math.inf
        for sig in node.P:
            if sig not in cur:
                continue
            p = node.P[sig]
            if root_noise is not None and sig in root_noise:
                p = (1 - cfg.dirichlet_eps) * p + cfg.dirichlet_eps * root_noise[sig]
            u = cfg.c_puct * p * sqrt_n / (1 + node.N.get(sig, 0))
            score = qnorm(sig) + u
            if score > best_score:
                best, best_score = sig, score
        return best

    def _child_return(parent, sig, obs2, done, depth):
        """Value of the state we just stepped into, discounted to t0. Handles
        terminal, the clock-based horizon bootstrap, and coupling truncation
        (share/short-circuit tree nodes by state fingerprint)."""
        if done:
            return 0.0
        clk = float(env.pn.clock)
        if clk - t0 > cfg.lookahead or depth >= cfg.max_depth:
            return _disc(clk) * _value(obs2)
        if cfg.coupling_truncate:
            # Node identity IS the state: coincident branches share one node
            # (correct under stochastic determinizations too — a different
            # resulting state simply has a different fingerprint).
            fp = state_fingerprint(env.pn)
            child = trans_table.get(fp)
            if child is None:
                child = _Node()
                trans_table[fp] = child
                counters["nodes"] += 1
            else:
                counters["transpositions"] += 1
                # value-reuse short-circuit: a coupled, already-resolved state
                # has an identical future — reuse its estimate, skip descent.
                if cfg.couple_min_visits > 0 and child.n_total >= cfg.couple_min_visits:
                    counters["truncations"] += 1
                    return child.value()
        else:
            kids = child_table.setdefault(id(parent), {})
            child = kids.get(sig)
            if child is None:
                child = _Node()
                counters["nodes"] += 1
                kids[sig] = child
        return _simulate(child, obs2, depth)

    def _simulate(node, obs, depth, root_noise=None):
        if not node.expanded:
            _expand(node, obs)
            v = _disc(env.pn.clock) * _value(obs)
            node.n_total += 1
            node.w_total += v
            return v
        cur = _current_positions()
        sig = _select(node, cur, root_noise=root_noise if depth == 0 else None)
        if sig is None:  # divergence: nothing cached is enabled now
            return _disc(env.pn.clock) * _value(obs)
        pos = cur[sig]
        obs2, r, done, _, _ = env.step(pos)
        counters["steps"] += 1
        contrib = float(r) * _disc(env.pn.clock)
        child_ret = _child_return(node, sig, obs2, done, depth + 1)
        total = contrib + child_ret
        node.N[sig] = node.N.get(sig, 0) + 1
        node.W[sig] = node.W.get(sig, 0.0) + total
        node.n_total += 1
        node.w_total += total
        if total < vmin[0]:
            vmin[0] = total
        if total > vmax[0]:
            vmax[0] = total
        return total

    # ---- rollout-based (whole / lineage) backup ------------------------- #
    rollout = cfg.rollout_backup or cfg.lineage_backup

    def _descend_child(parent, sig):
        """Node-sharing descent for the rollout path (no value-reuse
        short-circuit: the rollout must complete for the trace-based backup)."""
        if cfg.coupling_truncate:
            fp = state_fingerprint(env.pn)
            child = trans_table.get(fp)
            if child is None:
                child = _Node()
                trans_table[fp] = child
                counters["nodes"] += 1
            else:
                counters["transpositions"] += 1
            return child
        kids = child_table.setdefault(id(parent), {})
        child = kids.get(sig)
        if child is None:
            child = _Node()
            counters["nodes"] += 1
            kids[sig] = child
        return child

    def _uniform_action():
        # Uniform over REAL bindings, never postpone (unless postpone is the only
        # option). A rollout estimates the value of DOING the decision's work, so
        # a tail that postpones fails to complete cases and starves the lineage
        # (descendant) reward — measured: a plain-uniform tail raised the eval
        # postpone rate 2%->26%. Excluding postpone keeps the tail productive at
        # no extra cost (no graph, no network).
        binds = getattr(env.pn, "pn_actions", [])
        if not binds:
            return None
        real = [j for j, b in enumerate(binds) if _sig(b) != _POSTPONE]
        pool = real if real else list(range(len(binds)))
        return int(pool[np.random.randint(len(pool))])

    def _within(): return float(env.pn.clock) - t0 <= cfg.lookahead

    def _simulate_rollout(root_noise):
        ct = getattr(env.pn, "causal_trace", None)
        if ct is None:
            raise RuntimeError(_TRACE_ERR)
        ct.flush()   # branch-local trace: ancestry roots at the fork, not earlier
        node, depth, done = root, 0, False
        path = []  # (node, sig, roots, root_rec_id, trace_idx)
        # tree phase: PUCT through expanded nodes, recording decision edges.
        # Steps use build_obs=False: run_evolutions refreshes pn_actions cheaply
        # (compute_pn_actions) but skips the ~66%-of-step tensor graph. The graph
        # is built on demand ONLY to expand a new leaf (its network prior). The
        # rollout tail continues uniformly (no network ⇒ no graph). This is the
        # profiled speedup (get_graph_observation was ~80% of step cost).
        while node.expanded and not done and _within() and depth < cfg.max_depth:
            cur = _current_positions()
            sig = _select(node, cur, root_noise=root_noise if depth == 0 else None)
            if sig is None:
                break
            pos = cur[sig]
            idx = len(ct.transition_history.transitions)
            _, r, done, _, _ = env.step(pos, build_obs=False)
            counters["steps"] += 1
            trans = ct.transition_history.transitions
            if len(trans) > idx:
                roots = set(trans[idx].get("output_tokens", ()))
                rrid = id(trans[idx])
            else:
                roots, rrid = set(), None
            path.append((node, sig, roots, rrid, idx))
            if done:
                break
            node = _descend_child(node, sig)
            depth += 1
        if not node.expanded and not done and _within():
            _expand(node, env.pn.get_graph_observation())
        guard = 0
        while not done and _within() and guard < 500:
            a = _uniform_action()
            if a is None:
                break
            _, r, done, _, _ = env.step(a, build_obs=False)
            counters["steps"] += 1
            guard += 1
        # per-decision backup from the completed trace.
        #
        # POSTPONE is scored by its (near-zero, token-flow-OFF) lineage return in
        # BOTH backups — it is a pure timing action with no causal descendants.
        # This is essential for a controlled lineage-vs-whole comparison: whole
        # return-to-go credits postpone with ALL the reward it merely delayed
        # (postpone ≈ acting), which swamps the search into always-postponing
        # (measured: the whole arm collapsed to greedy 0.0 at every epoch on s1).
        # Suppressing postpone identically in both arms leaves the ONLY
        # difference as how REAL decisions are scored — whole (mc_q-in-tree) vs
        # lineage (lrq-in-tree) — which is exactly the effect the A/B isolates.
        for (node_i, sig_i, roots_i, rrid_i, idx_i) in path:
            if sig_i == _POSTPONE or cfg.lineage_backup:
                val = _lineage_return(ct, roots_i, rrid_i, t0, cfg.beta)
            else:
                val = _whole_return_to_go(ct, idx_i, t0, cfg.beta)
            node_i.N[sig_i] = node_i.N.get(sig_i, 0) + 1
            node_i.W[sig_i] = node_i.W.get(sig_i, 0.0) + val
            node_i.n_total += 1
            node_i.w_total += val
            if val < vmin[0]:
                vmin[0] = val
            if val > vmax[0]:
                vmax[0] = val

    # Pre-expand the root so every simulation descends through it.
    _expand(root, state_graph)
    root_noise = None
    if cfg.dirichlet_alpha > 0 and len(root.P) > 1:
        noise = np.random.dirichlet([cfg.dirichlet_alpha] * len(root.P))
        root_noise = {s: float(n) for s, n in zip(root.P.keys(), noise)}

    for i in range(cfg.n_simulations):
        env.set_state(root_snapshot)
        if cfg.use_crn:
            env.set_seed(np.random.randint(0, 2 ** 31 - 1))
        if rollout:
            _simulate_rollout(root_noise)
        else:
            _simulate(root, state_graph, 0, root_noise=root_noise)

    # Restore the real episode state (the X15 corruption guard).
    env.set_state(root_snapshot)

    # Build target_pi + means aligned to enabled env positions.
    cur = _current_positions()
    counts = np.zeros(A, dtype=np.float64)
    means = np.zeros(A, dtype=np.float32)
    for k, pos in enumerate(enabled):
        b = env.pn.pn_actions[pos] if pos < len(env.pn.pn_actions) else None
        sg = _sig(b)
        if sg in root.N:
            counts[k] = root.N[sg]
            q = root.q(sg)
            means[k] = float(q) if q is not None else 0.0

    if counts.sum() <= 0:
        # gate collapsed to a forced move (or no sims landed): one-hot it.
        target = np.zeros(A, dtype=np.float32)
        # the single kept child, else fall back to the prior argmax position 0
        kept = next(iter(root.P), None)
        idx = 0
        for k, pos in enumerate(enabled):
            if _sig(env.pn.pn_actions[pos]) == kept:
                idx = k
                break
        target[idx] = 1.0
    else:
        if cfg.temperature != 1.0:
            counts = counts ** (1.0 / max(cfg.temperature, 1e-8))
        target = (counts / counts.sum()).astype(np.float32)

    stds = np.zeros(A, dtype=np.float32)
    return target, {"means": means.tolist(), "stds": stds.tolist(),
                    "sims": int(cfg.n_simulations),
                    "visits": counts.tolist(),
                    "nodes": counters["nodes"],
                    "transpositions": counters["transpositions"],
                    "truncations": counters["truncations"],
                    "steps": counters["steps"]}, enabled
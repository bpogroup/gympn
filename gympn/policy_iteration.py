"""Rollout policy iteration: a rollout planner over a base policy, and its
distillation into a policy network.

The planner (RolloutPlanner) improves a base policy one decision at a time: at
a decision it lets the base policy pick an action, takes the net component
that action belongs to (GymProblem.net_partition, the static or the marking
certificate), and plays every distinct option of that component out to the end
of the episode with the base policy, all options under the same random seeds
(common random numbers). It takes the option with the best mean return of the
component, keeping the base policy's choice unless another is strictly better.
Under the certificate nothing outside the component depends on the choice, so
the component's own return is the right score and the other components'
rewards would only add noise. In expectation the planner is at least as good
as its base policy.

Distillation (collect_decisions, distill) records the planner's choices with
the observation the policy network sees, and trains the network to choose the
same option: the loss is the negative log of the probability the network puts
on the chosen option's class of equivalent bindings (same transition, same
token values), so it does not care which of several identical bindings it
picks. The network is the actor PPO uses (gympn.train.make_policy_network), so
a distilled policy can be loaded as a PPO starting point
(--load_policy_network) or serve as the next base policy (NetworkBase): that
loop is rollout policy iteration.

Base policies are callables env -> action index. HeuristicBase wraps a gympn
heuristic, NetworkBase a policy network.
"""
from __future__ import annotations

import hashlib
import math
import random
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch

__all__ = ["HeuristicBase", "NetworkBase", "RolloutPlanner", "option_key", "collect_decisions",
           "soft_targets", "distill", "class_nll", "soft_class_nll"]


# --------------------------------------------------------------------------- base policies
class HeuristicBase:
    """A gympn heuristic as a base policy. The heuristic receives the problem
    itself when nothing in it is unobservable (cheap), else its heuristic
    observation."""

    def __init__(self, heuristic):
        from gympn.solvers import HeuristicSolver
        self.solver = HeuristicSolver(heuristic)

    def __call__(self, env) -> int:
        pn = env.pn
        B = pn.pn_actions
        hidden = getattr(pn, 'unobservable_simvars', None) or getattr(pn, 'unobservable_token_attrs', None)
        sel = self.solver.solve(pn.get_heuristic_observation() if hidden else pn, B)
        return _index_of(B, sel)


class NetworkBase:
    """A policy network as a base policy (greedy, or sampling)."""

    needs_obs = True

    def __init__(self, policy, greedy: bool = True):
        self.policy, self.greedy = policy.eval(), greedy

    def __call__(self, env) -> int:
        with torch.no_grad():
            p = self.policy(env.pn.get_graph_observation()).reshape(-1)
        if self.greedy:
            return int(torch.argmax(p).item())
        return int(torch.multinomial(p, 1).item())


def _index_of(B, sel) -> int:
    if isinstance(sel, (int, np.integer)):
        return int(sel)
    for i, b in enumerate(B):
        if b is sel:
            return i
    if sel == 'postpone' or (isinstance(sel, tuple) and sel and sel[0] == ['postpone']):
        for i, b in enumerate(B):
            if isinstance(b[0], list) and b[0] == ['postpone']:
                return i
    for i, b in enumerate(B):
        if b == sel:
            return i
    raise ValueError("the base policy chose a binding that is not offered")


# --------------------------------------------------------------------------- planner
def _is_postpone(b) -> bool:
    return isinstance(b[0], list) and b[0] == ['postpone']


def option_key(b):
    """Bindings with equal keys are exactly equivalent: same transition and
    same token values (postpone: same component)."""
    if _is_postpone(b):
        return ('postpone', getattr(b[2], 'comp', None))
    return (b[2]._id, tuple(repr(tok.value) for _p, tok in b[0]))


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.blake2b(repr(parts).encode(), digest_size=4).digest(), 'little')


class RolloutPlanner:
    """One-step rollout improvement of `base` (see the module docstring).

    rollouts : episodes played out per option.
    component : compare only the options of the base choice's component and
        score them by that component's return (True), or compare all options
        by the whole net's return (False).
    beta : SMDP discount of the return-to-go, exp(-beta * (t - now)); 0 = the
        undiscounted return (what evaluation measures).
    """

    def __init__(self, base: Callable, rollouts: int = 8, component: bool = True, beta: float = 0.0,
                 seed: int = 0):
        self.base, self.rollouts, self.component, self.beta, self.seed = base, rollouts, component, beta, seed
        self.last = None            # details of the last decision (options, scores, chosen)

    def _comp(self, pn, b, part):
        return b[2].comp if _is_postpone(b) else part.get(b[2]._id)

    def _play_out(self, env, first, scored):
        """Fire `first`, then the base policy to the end; the (discounted)
        reward that `scored(info)` counts."""
        t0 = env.pn.clock
        total, a = 0.0, first
        build = getattr(self.base, 'needs_obs', False)
        while True:
            t = env.pn.clock
            _, r, done, _, info = env.step(a, build_obs=build)
            w = math.exp(-self.beta * max(0.0, t - t0)) if self.beta else 1.0
            total += w * scored(r, info)
            if done or not env.pn.pn_actions:
                return total
            a = self.base(env)

    def choose(self, env, tag=()) -> int:
        """The planner's action at env's current decision. `tag` (e.g. the
        episode and step) makes the rollout seeds distinct per decision."""
        pn = env.pn
        B = pn.pn_actions
        h = self.base(env)
        reps, seen = [], {}
        part = pn.net_partition() if self.component else None
        c = self._comp(pn, B[h], part) if self.component else None
        for i, b in enumerate(B):
            if self.component and self._comp(pn, b, part) != c:
                continue
            k = option_key(b)
            if k not in seen:
                seen[k] = i
                reps.append(i)
        h_rep = seen[option_key(B[h])]
        self.last = {"options": len(reps), "base": h, "scores": None, "reps": reps, "chosen": h}
        if len(reps) < 2:
            return h
        if self.component:
            if getattr(pn, 'marking_certificate', False):
                members = pn.net_component_members(c)
                scored = lambda r, info: sum(v for k, v in info.get('comp_reward', {}).items() if k in members)
            else:
                scored = lambda r, info: sum(v for k, v in info.get('comp_reward', {}).items() if k == c)
        else:
            scored = lambda r, info: r
        snap = env.get_state()
        scores = []
        try:
            for o in reps:
                s = 0.0
                for j in range(self.rollouts):
                    env.set_state(snap)
                    sd = _seed(self.seed, tag, j)
                    random.seed(sd); np.random.seed(sd)
                    s += self._play_out(env, o, scored)
                scores.append(s / self.rollouts)
        finally:
            env.set_state(snap)       # back to the decision, with the episode's own random state
        best = int(np.argmax(scores))
        chosen = reps[best] if scores[best] > scores[reps.index(h_rep)] + 1e-9 else h
        self.last = {"options": len(reps), "base": h, "scores": scores, "reps": reps, "chosen": chosen}
        return chosen


# --------------------------------------------------------------------------- distillation data
def _record(graph, B, chosen, last):
    """Attach the planner's decision to the observation graph:
    t_mask_a / t_mask_p: the action / postpone nodes of the chosen class of
        equivalent bindings;
    t_cls_a / t_cls_p: each node's option class among the classes the planner
        compared (its component), -1 for nodes outside them;
    t_scores: the classes' mean rollout returns (NaN when there was a single
        option, so no rollouts), t_chosen: the chosen class, t_ncls: their
        number. soft_targets() turns them into training targets."""
    keys = [option_key(b) for b in B]
    reps = last.get("reps") or [chosen]
    rep_keys = [keys[i] for i in reps]
    kc = keys[chosen]
    if kc not in rep_keys:                  # (the base's choice outside the compared classes)
        rep_keys, scores = [kc], None
    else:
        scores = last.get("scores")
    cls = [rep_keys.index(k) if k in rep_keys else -1 for k in keys]
    is_pp = [_is_postpone(b) for b in B]
    graph.t_mask_a = torch.tensor([k == kc for k, p in zip(keys, is_pp) if not p], dtype=torch.bool)
    graph.t_mask_p = torch.tensor([k == kc for k, p in zip(keys, is_pp) if p], dtype=torch.bool)
    graph.t_cls_a = torch.tensor([c for c, p in zip(cls, is_pp) if not p], dtype=torch.long)
    graph.t_cls_p = torch.tensor([c for c, p in zip(cls, is_pp) if p], dtype=torch.long)
    graph.t_scores = (torch.tensor(scores, dtype=torch.float32) if scores is not None
                      else torch.full((len(rep_keys),), float('nan')))
    graph.t_chosen = torch.tensor([rep_keys.index(kc)], dtype=torch.long)
    graph.t_ncls = torch.tensor([len(rep_keys)], dtype=torch.long)
    return graph


def soft_targets(graphs, tau: float = 0.0):
    """Set each graph's t_q, the target distribution over its option classes:
    softmax(scores / tau) when the planner compared several options and
    tau > 0, else all mass on the chosen class."""
    for g in graphs:
        n = int(g.t_ncls)
        q = torch.zeros(n)
        if tau > 0 and n > 1 and not torch.isnan(g.t_scores).any():
            q = torch.softmax(g.t_scores / tau, dim=0)
        else:
            q[int(g.t_chosen)] = 1.0
        g.t_q = q
    return graphs


def collect_decisions(make_env: Callable, planner_fn: Callable, seeds: Sequence[int], workers: int = 1,
                      log: Optional[Callable] = None) -> List:
    """Play the planner on one episode per seed (Python and NumPy random
    reseeded, then env.reset(), like the evaluation) and record every decision
    as the observation graph with t_mask_a / t_mask_p, the class of the
    planner's choice. make_env() and planner_fn(env) must be picklable
    (module-level functions) when workers > 1. Returns graphs, in seed order."""
    tasks = [(make_env, planner_fn, int(s)) for s in seeds]
    if workers <= 1:
        out = [_collect_one(t) for t in tasks]
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers) as pool:
            out = []
            for res in pool.imap(_collect_one, tasks):
                out.append(res)
                if log:
                    log(f"collected episode {len(out)}/{len(tasks)}: {len(res[0])} decisions, "
                        f"return {res[1]:.1f}, {res[2]} changed")
    return [g for graphs, _, _ in out for g in graphs]


def _collect_one(task):
    make_env, planner_fn, seed = task
    torch.set_num_threads(1)
    env = make_env()
    planner = planner_fn(env)
    random.seed(seed); np.random.seed(seed % 2 ** 32)
    obs = env.reset()
    graphs, total, changed, step = [], 0.0, 0, 0
    while env.pn.pn_actions:
        B = env.pn.pn_actions
        a = planner.choose(env, tag=(seed, step))
        changed += a != planner.last["base"]
        graphs.append(_record(obs['graph'], B, a, planner.last))
        obs, r, done, _, _ = env.step(a)
        total += r
        step += 1
        if done:
            break
    return graphs, total, changed


# --------------------------------------------------------------------------- distillation
def class_nll(policy, batch):
    """Mean over the batch's decisions of -log P(chosen class): the network's
    probability summed over the bindings equivalent to the chosen one."""
    probs = policy({'graph': batch}).reshape(-1)
    dev = probs.device
    nA, nP = int(batch.a_idx.numel()), int(batch.p_idx.numel())
    idx = torch.cat([batch.batch[batch.a_idx], batch.batch[batch.p_idx]]).to(dev)
    mask = torch.cat([batch.t_mask_a.reshape(-1), batch.t_mask_p.reshape(-1)]).to(dev)
    p_cls = torch.zeros(int(batch.num_graphs), device=dev).index_add(0, idx, probs * mask)
    return -(p_cls.clamp_min(1e-8)).log().mean()


def soft_class_nll(policy, batch):
    """Mean over the batch's decisions of the cross-entropy between the
    target t_q over option classes (soft_targets) and the network's class
    probabilities (summed over each class's equivalent bindings)."""
    probs = policy({'graph': batch}).reshape(-1)
    dev = probs.device
    G = int(batch.num_graphs)
    gidx = torch.cat([batch.batch[batch.a_idx], batch.batch[batch.p_idx]]).to(dev)
    cls = torch.cat([batch.t_cls_a.reshape(-1), batch.t_cls_p.reshape(-1)]).to(dev)
    ncls = batch.t_ncls.reshape(-1).to(dev)
    off = torch.cumsum(ncls, 0) - ncls
    keep = cls >= 0
    gcls = cls[keep] + off[gidx[keep]]
    P = torch.zeros(int(ncls.sum()), device=dev).index_add(0, gcls, probs[keep])
    q = batch.t_q.reshape(-1).to(dev)
    return -(q * P.clamp_min(1e-8).log()).sum() / G


def distill(policy, graphs: List, epochs: int = 15, lr: float = 1e-3, batch_size: int = 32,
            device: Optional[str] = None, seed: int = 0, log: Optional[Callable] = None,
            evaluate: Optional[Callable] = None, eval_every: int = 5, tau: float = 0.0,
            select: Optional[str] = None):
    """Train `policy` on the planner's choices with Adam: the cross-entropy
    with soft_targets(graphs, tau) (tau = 0: the chosen class only).
    evaluate(policy) -> dict is called every eval_every epochs and at the end
    (on the CPU, the policy in eval mode), and a copy of the weights is kept
    at each evaluation (history["checkpoints"]). With select = a key of
    evaluate's dict, the policy ends with the weights of the evaluation that
    maximised it (history["selected_epoch"]); else with the last ones.
    Returns the history."""
    import copy
    soft_targets(graphs, tau)
    from torch_geometric.data import Batch
    dev = torch.device(device) if device else torch.device('cpu')
    rng = random.Random(seed)
    with torch.no_grad():
        policy({'graph': Batch.from_data_list(graphs[:2])})     # materialise lazy layers
    policy.to(dev)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    hist = {"loss": [], "evals": [], "checkpoints": {}, "tau": tau}
    for ep in range(1, epochs + 1):
        policy.train()
        order = list(range(len(graphs)))
        rng.shuffle(order)
        losses = []
        for i in range(0, len(order), batch_size):
            b = Batch.from_data_list([graphs[j] for j in order[i:i + batch_size]]).to(dev)
            loss = soft_class_nll(policy, b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        hist["loss"].append(float(np.mean(losses)))
        msg = f"epoch {ep}: loss {hist['loss'][-1]:.3f}"
        if evaluate is not None and (ep % eval_every == 0 or ep == epochs):
            policy.to('cpu').eval()
            ev = evaluate(policy)
            hist["evals"].append({"epoch": ep, **ev})
            hist["checkpoints"][ep] = copy.deepcopy(policy.state_dict())
            msg += " | " + ", ".join(f"{k} {v:.3f}" for k, v in ev.items() if isinstance(v, float))
            policy.to(dev)
        if log:
            log(msg)
    policy.to('cpu').eval()
    if select is not None and hist["evals"]:
        best = max(hist["evals"], key=lambda e: e[select])
        hist["selected_epoch"] = best["epoch"]
        policy.load_state_dict(hist["checkpoints"][best["epoch"]])
    return hist

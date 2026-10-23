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
        self.deterministic = True

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
        self.deterministic = greedy

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
    lockstep : with a network base policy, play all rollouts of a decision
        side by side with batched forward passes (same returns as one at a
        time, much faster).
    restrict : with component=True, simulate only the component in the
        rollouts (GymProblem.sim_only = its member transitions): no other
        component can affect its future (the certificate), so this is exact
        for the score and removes the other components' cost and noise; the
        network base then reads the component's subgraph only (its logits do
        not depend on anything else: no edges leave a component, and the
        actor's attention is masked to it).
    keyed : common random numbers through keyed streams
        (GymProblem.stream_seed): in rollout j of every option, the k-th
        firing of each transition draws the same random numbers, however the
        options' trajectories differ. With keyed=False the options only share
        the start of one global stream, which shifts as soon as they fire
        different transitions.
    coalesce : with keyed streams, the base choice's rollouts are played
        first and record the state fingerprint (GymProblem.state_fingerprint)
        at each step; rollout j of another option that reaches a state rollout
        j of the base choice has visited stops there and takes the base
        rollout's remaining return, which it would have collected identically.
        Exact; saves the steps after the trajectories re-merge.
    """

    def __init__(self, base: Callable, rollouts: int = 8, component: bool = True, beta: float = 0.0,
                 seed: int = 0, lockstep: bool = True, restrict: bool = True, keyed: bool = True,
                 coalesce: bool = True):
        self.base, self.rollouts, self.component, self.beta, self.seed = base, rollouts, component, beta, seed
        self.lockstep, self.restrict, self.keyed = lockstep, restrict, keyed
        self.coalesce = coalesce and keyed
        self.steps = [0, 0]           # rollout steps simulated, and saved by coalescing
        self._only = None             # member transitions of the rollouts' component (restrict)
        self.last = None            # details of the last decision (options, scores, chosen)

    def _comp(self, pn, b, part):
        return b[2].comp if _is_postpone(b) else part.get(b[2]._id)

    def _play_out(self, env, first, scored, record=None, lookup=None):
        """Fire `first`, then the base policy to the end; the (discounted)
        reward that `scored(info)` counts. record: a dict filled with
        fingerprint -> reward collected so far, at every state visited;
        lookup: such a dict of a finished run turned into fingerprint ->
        remaining reward, to stop at a state it visited."""
        t0 = env.pn.clock
        total, a = 0.0, first
        self._k = 0
        build = getattr(self.base, 'needs_obs', False)
        while True:
            t = env.pn.clock
            _, r, done, _, info = env.step(a, build_obs=build)
            self.steps[0] += 1
            self._k += 1
            w = math.exp(-self.beta * max(0.0, t - t0)) if self.beta else 1.0
            total += w * scored(r, info)
            if done or not env.pn.pn_actions:
                return total
            if record is not None or lookup is not None:
                fp = env.pn.state_fingerprint()
                if record is not None:
                    record.setdefault(fp, (total, self._k))
                elif fp in lookup:
                    self.steps[1] += lookup[fp][1]
                    return total + lookup[fp][0]
            a = self.base(env)

    def _coalescing(self):
        """Coalescing is exact when the future is a function of the state:
        keyed streams and a deterministic base policy."""
        return self.coalesce and getattr(self.base, 'deterministic', False)

    @staticmethod
    def _remaining(record, total, steps):
        """record (fingerprint -> (reward so far, step)) -> (remaining reward,
        remaining steps) for a run that ended with `total` after `steps`."""
        return {fp: (total - c, steps - k) for fp, (c, k) in record.items()}

    def _first_actions(self, renv, B, reps):
        """Each option's index in renv's action list (renv at the decision,
        possibly restricted to the component, so its list is shorter):
        the first binding of the same equivalence class."""
        if self._only is None:
            return {o: o for o in reps}
        pos = {}
        for i, b in enumerate(renv.pn.pn_actions):
            pos.setdefault(option_key(b), i)
        return {o: pos[option_key(B[o])] for o in reps}

    def _scores_lockstep(self, env, snap, reps, scored, tag, ref=None):
        """The rollouts of all options side by side, for a network base
        policy: one batched forward pass per step for all of them. Each
        rollout runs on its own copy of env with its own random state (seeded
        as the sequential rollouts are, swapped in around each of its steps),
        so the returns are those of the sequential rollouts."""
        import copy
        from torch_geometric.data import Batch
        R = self.rollouts
        pool = self.__dict__.setdefault('_pool', [])
        while len(pool) < len(reps) * R:
            pool.append(copy.deepcopy(env))
        slots = []
        B = env.pn.pn_actions
        first = None
        coal = self._coalescing() and ref is not None
        ref_oi = reps.index(ref) if ref is not None else 0
        refs, ref_end, merged = {j: {} for j in range(R)}, {}, []
        for oi in [ref_oi] + [i for i in range(len(reps)) if i != ref_oi]:   # the base choice first
            for j in range(R):
                e = pool[len(slots)]
                e.pn.sim_only = self._only
                e.pn.stream_isolated = False
                e.set_state(snap)
                if first is None:
                    first = self._first_actions(e, B, reps)
                o = first[reps[oi]]
                sd = _seed(self.seed, tag, j)
                random.seed(sd); np.random.seed(sd)
                if self.keyed:
                    e.pn.stream_seed = sd
                # (keyed streams: no behaviour reads the global streams, so they need no swapping)
                slots.append({'env': e, 'oi': oi, 'j': j, 'k': 0, 'tot': 0.0, 't0': e.pn.clock, 'next': o,
                              'rng': None if self.keyed else (random.getstate(), np.random.get_state())})
        totals = np.zeros(len(reps))
        policy = self.base.policy
        active = slots
        while active:
            need = []
            for sl in active:
                e = sl['env']
                if sl['rng'] is not None:
                    random.setstate(sl['rng'][0]); np.random.set_state(sl['rng'][1])
                t = e.pn.clock
                obs, r, done, _, info = e.step(sl['next'])
                if sl['rng'] is not None:
                    sl['rng'] = (random.getstate(), np.random.get_state())
                self.steps[0] += 1
                sl['k'] += 1
                w = math.exp(-self.beta * max(0.0, t - sl['t0'])) if self.beta else 1.0
                sl['tot'] += w * scored(r, info)
                if done or not e.pn.pn_actions:
                    totals[sl['oi']] += sl['tot']
                    if coal and sl['oi'] == ref_oi:
                        ref_end[sl['j']] = (sl['tot'], sl['k'])
                    continue
                if coal:
                    fp = e.pn.state_fingerprint()
                    if sl['oi'] == ref_oi:
                        refs[sl['j']].setdefault(fp, (sl['tot'], sl['k']))
                    elif fp in refs[sl['j']]:
                        # re-merged with the base choice's rollout j: it continues identically
                        merged.append((sl, refs[sl['j']][fp]))
                        continue
                sl['graph'] = obs['graph']
                need.append(sl)
            if not need:
                break
            graphs = [sl['graph'] for sl in need]
            with torch.no_grad():
                pi = policy({'graph': Batch.from_data_list(graphs)}).reshape(-1)
            oa, op = 0, sum(int(g.a_idx.numel()) for g in graphs)
            for sl, g in zip(need, graphs):
                a_n, p_n = int(g.a_idx.numel()), int(g.p_idx.numel())
                p_s = torch.cat((pi[oa:oa + a_n], pi[op:op + p_n]))
                oa += a_n; op += p_n
                sl['next'] = (int(torch.argmax(p_s).item()) if self.base.greedy
                              else int(torch.multinomial(p_s, 1).item()))
            active = need
        for sl, (c_ref, k_ref) in merged:
            tot_ref, k_end = ref_end[sl['j']]
            totals[sl['oi']] += sl['tot'] + (tot_ref - c_ref)
            self.steps[1] += k_end - k_ref
        return list(totals / R)

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
        self._only = (frozenset(pn.net_component_members(c)) if self.component and self.restrict
                      else None)
        try:
            if self.lockstep and getattr(self.base, 'policy', None) is not None:
                scores = self._scores_lockstep(env, snap, reps, scored, tag, ref=h_rep)
            else:
                scores = []
                env.pn.sim_only = self._only
                env.pn.stream_isolated = False      # the snapshot restores the random state afterwards
                env.set_state(snap)
                first = self._first_actions(env, B, reps)
                order = [h_rep] + [o for o in reps if o != h_rep]
                refs, by_o = {}, {}
                for o in order:
                    s = 0.0
                    for j in range(self.rollouts):
                        env.set_state(snap)
                        sd = _seed(self.seed, tag, j)
                        random.seed(sd); np.random.seed(sd)
                        if self.keyed:
                            env.pn.stream_seed = sd
                        if self._coalescing() and o == h_rep:
                            rec = {}
                            g = self._play_out(env, first[o], scored, record=rec)
                            refs[j] = self._remaining(rec, g, self._k)
                        else:
                            g = self._play_out(env, first[o], scored,
                                               lookup=refs.get(j) if self._coalescing() else None)
                        s += g
                    by_o[o] = s / self.rollouts
                scores = [by_o[o] for o in reps]
        finally:
            env.pn.sim_only = None
            env.pn.stream_isolated = True
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

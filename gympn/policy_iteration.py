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

import copy
import hashlib
import math
import random
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch

__all__ = ["HeuristicBase", "NetworkBase", "RolloutPlanner", "option_key", "collect_decisions",
           "soft_targets", "distill", "class_nll", "soft_class_nll", "advantage_loss",
           "advantage_loss_from_probs", "unlabelled_record",
           "set_class_nll", "episode_components", "collect_component_decisions"]


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
    from gympn.simulator import binding_option_key
    return binding_option_key(b)


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
    resample : fn(problem) called at the start of every rollout (after the
        rollout's seeding) to redraw what the deciding policy cannot know,
        e.g. the already-drawn times of future arrivals, which the simulator
        state holds but the observation does not show. Without it the
        rollouts see that hidden future, and the planner's choices partly
        rest on information no policy has. The same draws for every option
        (common random numbers).
    margin : depart from the base's choice only when the best option's mean
        rollout return beats the base's by more than this.
    confirm : when > 0, a departure from the base is kept only if a second,
        independent scoring (`confirm` rollouts per option, other seeds)
        departs to the same option; else the base's choice stands. Removes
        the departures that are rollout noise, at a cost only where the
        planner departs.
    sh_budget : when > 0, choose() selects by Sequential Halving with common
        random numbers, as Deep Controlled Learning does (Temizoz et al.,
        EJOR 2025): sh_budget * (number of options) rollouts in total,
        spread evenly over ceil(log2 options) rounds; each round plays every
        surviving option with the round's rollouts (paired: the same seeds for
        all options, fresh seeds each round), returns accumulate over the
        rounds, and the better half (by mean accumulated return) survives.
        The last survivor is the choice; no margin, no test.
    race_eps : when set (permits), choose() selects by racing, an adaptive
        alternative to sh_budget: rounds of race_step paired rollouts of every
        surviving option (the same seeds for all of them, fresh seeds each
        round, so the survivors stay paired over all rounds). From race_min
        rollouts on, an option is eliminated when the leader's mean paired
        advantage over it exceeds race_z standard errors, and the race stops
        when no survivor's upper bound (mean paired difference to the leader
        + race_z standard errors) reaches race_eps, or at race_cap rollouts
        per option. The leader is the choice; the survivors are its label set
        (last["set"], _record's t_set). A clear winner stops early; a near tie
        stops once its gap is shown to be below race_eps. race_z is a
        per-look threshold (no correction for repeated looks), so the error
        rate is controlled in practice, not guaranteed.
    test, min_t : when test > 0, a departure is tested on `test` fresh
        rollouts (other seeds) of only the chosen option and the base's,
        paired by rollout: it is kept only if the mean paired advantage
        exceeds the margin and its t statistic (mean / standard error) is at
        least min_t. Unlike confirm, the test does not redo the choice among
        the options, so a near tie between two good departures does not
        veto both; and the fresh rollouts remove the selection's optimism.
    """

    def __init__(self, base: Callable, rollouts: int = 8, component: bool = True, beta: float = 0.0,
                 seed: int = 0, lockstep: bool = True, restrict: bool = True, keyed: bool = True,
                 coalesce: bool = True, resample: Optional[Callable] = None,
                 margin: float = 0.0, confirm: int = 0, test: int = 0, min_t: float = 2.0,
                 sh_budget: int = 0, race_eps: Optional[float] = None, race_z: float = 2.0,
                 race_step: int = 2, race_min: int = 4, race_cap: int = 64):
        self.base, self.rollouts, self.component, self.beta, self.seed = base, rollouts, component, beta, seed
        self.lockstep, self.restrict, self.keyed = lockstep, restrict, keyed
        self.coalesce = coalesce and keyed
        self.resample = resample
        self.margin = margin
        self.confirm = confirm
        self.test, self.min_t = test, min_t
        self.sh_budget = sh_budget
        self.race_eps, self.race_z = race_eps, race_z
        self.race_step, self.race_min, self.race_cap = race_step, race_min, race_cap
        self._confirming = False
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

    def _redraw(self, env, a):
        """Redraw the hidden state of env (resample) and return the index of
        action a's option in the refreshed action list. A redraw may change
        the hidden attributes of tokens that the bindings hold; option keys
        leave them out, so the option is the same one."""
        if self.resample is None:
            return a
        key = option_key(env.pn.pn_actions[a])
        self.resample(env.pn)
        env.pn.compute_pn_actions()
        return [option_key(b) for b in env.pn.pn_actions].index(key)

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
                o = self._redraw(e, o)
                if self.keyed:
                    e.pn.stream_seed = sd
                # (keyed streams: no behaviour reads the global streams, so they need no swapping)
                slots.append({'env': e, 'oi': oi, 'j': j, 'k': 0, 'tot': 0.0, 't0': e.pn.clock, 'next': o,
                              'rng': None if self.keyed else (random.getstate(), np.random.get_state())})
        totals = np.zeros((len(reps), R))
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
                    totals[sl['oi'], sl['j']] = sl['tot']
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
            totals[sl['oi'], sl['j']] = sl['tot'] + (tot_ref - c_ref)
            self.steps[1] += k_end - k_ref
        return totals

    def _returns(self, env, snap, B, reps, scored, tag, h_rep):
        """Rollout returns of the options reps (array options x rollouts),
        env restored to snap afterwards."""
        pn = env.pn
        prev_only = pn.sim_only             # an episode may itself be restricted (component-factored collection)
        try:
            if self.lockstep and getattr(self.base, 'policy', None) is not None:
                return self._scores_lockstep(env, snap, reps, scored, tag, ref=h_rep)
            env.pn.sim_only = self._only if self._only is not None else prev_only
            env.pn.stream_isolated = False      # the snapshot restores the random state afterwards
            env.set_state(snap)
            first = self._first_actions(env, B, reps)
            order = [h_rep] + [o for o in reps if o != h_rep]
            refs, by_o = {}, {}
            for o in order:
                g_o = []
                for j in range(self.rollouts):
                    env.set_state(snap)
                    sd = _seed(self.seed, tag, j)
                    random.seed(sd); np.random.seed(sd)
                    a = self._redraw(env, first[o])
                    if self.keyed:
                        env.pn.stream_seed = sd
                    if self._coalescing() and o == h_rep:
                        rec = {}
                        g = self._play_out(env, a, scored, record=rec)
                        refs[j] = self._remaining(rec, g, self._k)
                    else:
                        g = self._play_out(env, a, scored,
                                           lookup=refs.get(j) if self._coalescing() else None)
                    g_o.append(g)
                by_o[o] = g_o
            return np.array([by_o[o] for o in reps])
        finally:
            pn.sim_only = prev_only
            pn.stream_isolated = True
            env.set_state(snap)       # back to the decision, with the episode's own random state

    def option_values(self, env, tag=(), anchor=None):
        """Mean rollout return of every option (class of equivalent bindings)
        in the component of the binding `anchor` (default: the base policy's
        choice), each option played first and the base policy after, under
        common random numbers. Returns (reps, scores, h): one binding index per
        option, their scores (None when there is a single option, so no
        rollouts), and the base policy's choice. env is left as it was."""
        B, h, reps, h_rep, scored = self._prepare(env, anchor)
        if len(reps) < 2:
            return reps, None, h
        snap = env.get_state()
        returns = self._returns(env, snap, B, reps, scored, tag, h_rep)
        self._last_returns, self._last_ctx = returns, (snap, B, scored, h_rep)
        return reps, list(returns.mean(axis=1)), h

    def _prepare(self, env, anchor=None):
        """The decision's bindings B, the base policy's choice h, one binding
        per option of the component of `anchor` (default h), the anchor's
        option, and the component's reward function; sets the restricted
        simulation (self._only)."""
        pn = env.pn
        B = pn.pn_actions
        h = self.base(env)
        ref = h if anchor is None else anchor
        reps, seen = [], {}
        part = pn.net_partition() if self.component else None
        c = self._comp(pn, B[ref], part) if self.component else None
        for i, b in enumerate(B):
            if self.component and self._comp(pn, b, part) != c:
                continue
            k = option_key(b)
            if k not in seen:
                seen[k] = i
                reps.append(i)
        h_rep = seen[option_key(B[ref])]
        if self.component:
            if getattr(pn, 'marking_certificate', False):
                members = pn.net_component_members(c)
                scored = lambda r, info: sum(v for k, v in info.get('comp_reward', {}).items() if k in members)
            else:
                scored = lambda r, info: sum(v for k, v in info.get('comp_reward', {}).items() if k == c)
        else:
            scored = lambda r, info: r
        self._only = (frozenset(pn.net_component_members(c)) if self.component and self.restrict
                      else None)
        return B, h, reps, h_rep, scored

    def _choose_sh(self, env, tag=()) -> int:
        """Sequential Halving with common random numbers (sh_budget)."""
        B, h, reps, h_rep, scored = self._prepare(env)
        self.last = {"options": len(reps), "base": h, "scores": None, "reps": reps, "chosen": h}
        if len(reps) < 2:
            return h
        snap = env.get_state()
        n = len(reps)
        rounds = max(1, math.ceil(math.log2(n)))
        total = int(self.sh_budget) * n
        tot = {o: 0.0 for o in reps}
        cnt = {o: 0 for o in reps}
        alive = list(reps)
        saved = self.rollouts
        try:
            for r in range(rounds):
                self.rollouts = max(1, total // (len(alive) * rounds))
                ref = h_rep if h_rep in alive else alive[0]
                ret = self._returns(env, snap, B, alive, scored, (tag, 'sh', r), ref)
                for o, row in zip(alive, ret):
                    tot[o] += float(row.sum())
                    cnt[o] += len(row)
                if len(alive) > 1:
                    alive = sorted(alive, key=lambda o: -tot[o] / cnt[o])[:math.ceil(len(alive) / 2)]
        finally:
            self.rollouts = saved
        chosen = max(alive, key=lambda o: tot[o] / cnt[o])
        self.last = {"options": n, "base": h, "reps": reps, "chosen": chosen,
                     "scores": [tot[o] / cnt[o] for o in reps], "counts": [cnt[o] for o in reps]}
        return chosen

    def _choose_race(self, env, tag=()) -> int:
        """Racing with paired rollouts (race_eps)."""
        B, h, reps, h_rep, scored = self._prepare(env)
        self.last = {"options": len(reps), "base": h, "scores": None, "reps": reps, "chosen": h}
        if len(reps) < 2:
            return h
        snap = env.get_state()
        z, eps = float(self.race_z), float(self.race_eps)
        samples = {o: [] for o in reps}
        alive = list(reps)
        saved = self.rollouts
        rnd = 0
        try:
            while True:
                self.rollouts = int(self.race_step)
                ref = h_rep if h_rep in alive else alive[0]
                ret = self._returns(env, snap, B, alive, scored, (tag, 'race', rnd), ref)
                rnd += 1
                for o, row in zip(alive, ret):
                    samples[o].extend(float(v) for v in row)
                n = len(samples[alive[0]])
                X = np.array([samples[o] for o in alive])          # survivors: aligned, paired
                means = X.mean(axis=1)
                b = int(np.argmax(means))
                d = X[b] - X                                       # leader minus each, per rollout
                dm = d.mean(axis=1)
                se = d.std(axis=1, ddof=1) / math.sqrt(n) if n > 1 else np.full(len(alive), np.inf)
                if n >= self.race_min:
                    keep = [i for i in range(len(alive)) if i == b or not (dm[i] > z * se[i])]
                    lead = alive[b]
                    alive = [alive[i] for i in keep]
                    dm, se = dm[keep], se[keep]
                    b = alive.index(lead)
                    ucb = max((-dm[i] + z * se[i] for i in range(len(alive)) if i != b), default=-np.inf)
                    if ucb < eps:
                        break
                if n >= self.race_cap:
                    break
        finally:
            self.rollouts = saved
        means = {o: float(np.mean(samples[o])) for o in reps}
        chosen = max(alive, key=lambda o: means[o])
        self.last = {"options": len(reps), "base": h, "reps": reps, "chosen": chosen,
                     "scores": [means[o] for o in reps], "counts": [len(samples[o]) for o in reps],
                     "set": list(alive)}
        return chosen

    def choose(self, env, tag=()) -> int:
        """The planner's action at env's current decision. `tag` (e.g. the
        episode and step) makes the rollout seeds distinct per decision."""
        if self.race_eps is not None:
            return self._choose_race(env, tag)
        if self.sh_budget:
            return self._choose_sh(env, tag)
        reps, scores, h = self.option_values(env, tag)
        self.last = {"options": len(reps), "base": h, "scores": None, "reps": reps, "chosen": h}
        if scores is None:
            return h
        B = env.pn.pn_actions
        snap, _B, scored, h_rep = self._last_ctx
        best = int(np.argmax(scores))
        chosen = reps[best] if scores[best] > scores[reps.index(h_rep)] + max(self.margin, 1e-9) else h
        last = {"options": len(reps), "base": h, "scores": scores, "reps": reps, "chosen": chosen}
        if self.test and chosen != h:
            saved = (self.seed, self.rollouts)
            self.seed, self.rollouts = _seed(self.seed, 'test'), self.test
            try:
                ret = self._returns(env, snap, B, [h_rep, reps[best]], scored, tag, h_rep)
            finally:
                self.seed, self.rollouts = saved
            d = ret[1] - ret[0]
            se = d.std(ddof=1) / math.sqrt(len(d))
            adv = float(d.mean())
            t = adv / se if se > 0 else (math.inf if adv > 0 else 0.0)
            last["test_adv"], last["test_t"] = adv, float(t)
            if not (adv > max(self.margin, 1e-9) and t >= self.min_t):
                chosen = last["chosen"] = h
        if self.confirm and chosen != h and not self._confirming:
            saved = (self.seed, self.rollouts)
            self.seed, self.rollouts, self._confirming = _seed(self.seed, 'confirm'), self.confirm, True
            try:
                again = self.choose(env, tag)
            finally:
                (self.seed, self.rollouts), self._confirming = saved, False
            last["confirmed"] = option_key(B[again]) == option_key(B[chosen])
            if not last["confirmed"]:
                chosen = last["chosen"] = h
        self.last = last
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
        number, t_base: the base policy's class (-1 outside them), t_keys:
        the base's and the chosen option keys (strings, for inspection). soft_targets() turns them into training targets."""
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
    h = last.get("base", chosen)
    graph.t_base = torch.tensor([rep_keys.index(keys[h]) if keys[h] in rep_keys else -1], dtype=torch.long)
    graph.t_keys = f"{keys[h]!r} -> {kc!r}"
    members = {keys[i] for i in (last.get("set") or [chosen])} | {kc}
    graph.t_set = torch.tensor([k in members for k in rep_keys], dtype=torch.bool)
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
    make_env, planner_fn, seed = task[:3]
    members = task[3] if len(task) > 3 else None
    torch.set_num_threads(1)
    env = make_env()
    if members is not None:             # component-factored: this component alone, from the reset on
        # One unrestricted reset first fixes the flat observation's vocabulary
        # and width, so every observation of the restricted run (the first
        # one included) comes from the direct builder, which shows only the
        # simulated transitions and places.
        env.reset()
        for k in ('_flat_meta', '_flat_width', 'metadata'):
            setattr(env.frozen_pn, k, getattr(env.pn, k, None))
        env.pn.sim_only = env.frozen_pn.sim_only = members
    dynamic = len(task) > 4 and task[4]
    random.seed(seed); np.random.seed(seed % 2 ** 32)
    obs = env.reset()
    graphs, total, changed = [], 0.0, 0
    branches = [(env, obs, ())]
    while branches:                     # dynamic: one branch per component the marking certificate splits off
        env, obs, br = branches.pop(0)
        planner = planner_fn(env)
        step = 0
        while env.pn.pn_actions:
            subs = _marking_split(env) if dynamic else []
            if len(subs) > 1:
                # The couplings have dissolved: by the marking certificate each
                # sub-component's future depends on its own marking only, so it
                # is continued alone, exactly, from here (its own simulation,
                # observation and planner).
                for i, m in enumerate(subs):
                    e = copy.deepcopy(env)
                    e.pn.sim_only = m
                    # Its own run up to its first choice (a part may have
                    # nothing enabled at the split, or a single binding,
                    # which is fired without asking), as env.step would.
                    r0 = e.pn.reward
                    o, done, e.i = e.pn.run_evolutions(e.run, e.i, e.active_model)
                    total += e.pn.reward - r0
                    if not done:
                        branches.append((e, o, br + (i,)))
                break
            B = env.pn.pn_actions
            a = planner.choose(env, tag=(seed, step) if not br else (seed, br, step))
            changed += a != planner.last["base"]
            graphs.append(_record(obs['graph'], B, a, planner.last))
            obs, r, done, _, _ = env.step(a)
            total += r
            step += 1
            if done:
                break
    return graphs, total, changed


def _marking_split(env) -> List[frozenset]:
    """The components of the marking certificate among the simulated
    transitions (GymProblem.sim_only, all if None), as member sets in a fixed
    order; more than one once a coupling has dissolved."""
    pn = env.pn
    only = pn.sim_only
    comps = {}
    for t, k in pn.net_partition().items():
        if k is not None and (only is None or t in only):
            comps.setdefault(k, set()).add(t)
    return sorted((frozenset(m) for m in comps.values()), key=min)


def episode_components(make_env: Callable, seed: int = 0) -> List[frozenset]:
    """The net's components by the STATIC certificate (the net structure,
    before anything fires), each as the set of its member transitions, in a
    fixed order (by smallest member id). Static, not the marking certificate
    after a reset: the reset already fires start-of-episode transitions
    (e.g. a backlog that opens the day), which the marking certificate then
    drops as dead, and a component simulated alone without them would start
    from the wrong marking. Static components are independent too, so
    simulating one alone is exact. (seed is unused; kept for the signature.)"""
    import copy
    pn = copy.deepcopy(make_env().frozen_pn)
    pn.marking_certificate = False
    part = pn.net_partition()
    comps = {}
    for t in list(pn.actions) + list(pn.events):
        comps.setdefault(part.get(t._id), set()).add(t._id)
    comps.pop(None, None)
    return sorted((frozenset(m) for m in comps.values()), key=min)


def collect_component_decisions(make_env: Callable, planner_fn: Callable, tasks: Sequence, workers: int = 1,
                                log: Optional[Callable] = None, dynamic: bool = False) -> List:
    """Component-factored collection: like collect_decisions, but each task
    (seed, members) simulates ONE net component of the episode alone
    (GymProblem.sim_only = members, set before the reset, so the restricted
    run starts from the same initial marking), with the planner choosing at
    its decisions. By the certificate the component's future depends on its
    own marking only, so its days have the distribution they have inside the
    whole net, and its decisions get the labels they would get there; the
    actor's logits for a component read that component only, so they are
    the same on the component's subgraph. Each component can get its own number
    of days, and every rollout simulates one component only.
    dynamic: a component is split further once the MARKING certificate splits
    it (a coupling dissolves, e.g. a start-of-shift pool runs empty): each
    sub-component continues alone from the split. Returns graphs, in task order."""
    tasks = [(make_env, planner_fn, int(sd), frozenset(m), dynamic) for sd, m in tasks]
    if workers <= 1:
        out = [_collect_one(t) for t in tasks]
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers) as pool:
            out = []
            for res in pool.imap(_collect_one, tasks):
                out.append(res)
                if log:
                    log(f"collected component-day {len(out)}/{len(tasks)}: {len(res[0])} decisions, "
                        f"return {res[1]:.1f}, {res[2]} changed")
    return [g for graphs, _, _ in out for g in graphs]


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


def set_class_nll(policy, batch):
    """Mean over the batch's decisions of -log of the network's probability
    of the label SET t_set (the options the planner could not tell apart from
    its choice, e.g. a race's survivors): a partial-label loss, so a near tie
    does not force one arbitrary option."""
    probs = policy({'graph': batch}).reshape(-1)
    dev = probs.device
    G = int(batch.num_graphs)
    gidx = torch.cat([batch.batch[batch.a_idx], batch.batch[batch.p_idx]]).to(dev)
    cls = torch.cat([batch.t_cls_a.reshape(-1), batch.t_cls_p.reshape(-1)]).to(dev)
    ncls = batch.t_ncls.reshape(-1).to(dev)
    off = torch.cumsum(ncls, 0) - ncls
    keep = cls >= 0
    P = torch.zeros(int(ncls.sum()), device=dev).index_add(0, cls[keep] + off[gidx[keep]], probs[keep])
    owner = torch.repeat_interleave(torch.arange(G, device=dev), ncls)
    inset = batch.t_set.reshape(-1).to(dev)
    p_set = torch.zeros(G, device=dev).index_add(0, owner, P * inset)
    return -(p_set.clamp_min(1e-8)).log().mean()


def soft_class_nll(policy, batch, renorm: bool = False):
    """Mean over the batch's decisions of the cross-entropy between the
    target t_q over option classes (soft_targets) and the network's class
    probabilities (summed over each class's equivalent bindings).
    renorm: the class probabilities are renormalized over the compared
    classes (the deciding component's options), as if the decision were
    observed on the component's subgraph: the label then constrains only how
    the component's own options compare, not how much probability the other
    components' actions get."""
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
    if renorm:
        owner = torch.repeat_interleave(torch.arange(G, device=dev), ncls)
        tot = torch.zeros(G, device=dev).index_add(0, owner, P)
        P = P / tot[owner].clamp_min(1e-8)
    q = batch.t_q.reshape(-1).to(dev)
    w = getattr(batch, 't_w', None)
    if w is None:
        return -(q * P.clamp_min(1e-8).log()).sum() / G
    owner = torch.repeat_interleave(torch.arange(G, device=dev), ncls)
    ce = torch.zeros(G, device=dev).index_add(0, owner, -(q * P.clamp_min(1e-8).log()))
    w = w.reshape(-1).to(dev)
    return (w * ce).sum() / w.sum()


def advantage_loss_from_probs(probs, batch, per_labelled: bool = False):
    """advantage_loss from the network's node probabilities `probs` (action
    nodes then postpone nodes, as the policy returns them). per_labelled:
    average over the decisions that carry option scores (the others,
    labelled with a single class and NaN scores, contribute 0) instead of
    over all decisions in the batch."""
    probs = probs.reshape(-1)
    dev = probs.device
    G = int(batch.num_graphs)
    gidx = torch.cat([batch.batch[batch.a_idx], batch.batch[batch.p_idx]]).to(dev)
    cls = torch.cat([batch.t_cls_a.reshape(-1), batch.t_cls_p.reshape(-1)]).to(dev)
    ncls = batch.t_ncls.reshape(-1).to(dev)
    off = torch.cumsum(ncls, 0) - ncls
    keep = cls >= 0
    gcls = cls[keep] + off[gidx[keep]]
    P = torch.zeros(int(ncls.sum()), device=dev, dtype=probs.dtype).index_add(0, gcls, probs[keep])
    q = batch.t_scores.reshape(-1).to(dev)
    base = batch.t_base.reshape(-1).to(dev)
    owner = torch.repeat_interleave(torch.arange(G, device=dev), ncls)
    ok = (~torch.isnan(q)) & (base[owner] >= 0)
    qb = q[(off + base.clamp_min(0))[owner]]
    a = torch.where(ok, q - qb, torch.zeros_like(q))
    if per_labelled:
        lab = torch.zeros(G, dtype=torch.bool, device=dev).index_put_((owner[ok],), torch.ones_like(owner[ok], dtype=torch.bool))
        return -(P * a).sum() / lab.sum().clamp_min(1)
    return -(P * a).sum() / G


def unlabelled_record(graph):
    """Attach empty option labels (one class, no scores) to an observation
    graph, so it batches with graphs labelled by _record and contributes
    nothing to advantage_loss."""
    nA, nP = int(graph.a_idx.numel()), int(graph.p_idx.numel())
    graph.t_mask_a = torch.zeros(nA, dtype=torch.bool)
    graph.t_mask_p = torch.zeros(nP, dtype=torch.bool)
    graph.t_cls_a = torch.full((nA,), -1, dtype=torch.long)
    graph.t_cls_p = torch.full((nP,), -1, dtype=torch.long)
    graph.t_scores = torch.full((1,), float('nan'))
    graph.t_chosen = torch.zeros(1, dtype=torch.long)
    graph.t_ncls = torch.ones(1, dtype=torch.long)
    graph.t_base = torch.full((1,), -1, dtype=torch.long)
    graph.t_keys = ""
    graph.t_set = torch.ones(1, dtype=torch.bool)
    return graph


def advantage_loss(policy, batch):
    """Mean over the batch's decisions of -sum_c P(c) (Q(c) - Q(base)): minus
    the network's expected advantage over the base policy's class, with the
    planner's mean rollout returns t_scores as Q (unbiased for every class,
    unlike the chosen class's score). Decisions without scores (a single
    option) contribute 0. Value-aware: a near tie costs almost nothing
    whichever way it goes, a costly mistake costs its price."""
    return advantage_loss_from_probs(policy({'graph': batch}), batch)


def distill(policy, graphs: List, epochs: int = 15, lr: float = 1e-3, batch_size: int = 32,
            device: Optional[str] = None, seed: int = 0, log: Optional[Callable] = None,
            evaluate: Optional[Callable] = None, eval_every: int = 5, tau: float = 0.0,
            select: Optional[str] = None, ema: Optional[float] = None,
            override_weight: float = 1.0, adv_weight: float = 0.0, ce_weight: float = 1.0,
            set_labels: bool = False, renorm: bool = False):
    """Train `policy` on the planner's choices with Adam: the cross-entropy
    with soft_targets(graphs, tau) (tau = 0: the chosen class only).
    evaluate(policy) -> dict is called every eval_every epochs and at the end
    (on the CPU, the policy in eval mode), and a copy of the weights is kept
    at each evaluation (history["checkpoints"]). With select = a key of
    evaluate's dict, the policy ends with the weights of the evaluation that
    maximised it (history["selected_epoch"]); else with the last ones.
    With ema = d (e.g. 0.995), an exponential moving average of the
    weights (updated after every step: w_ema <- d w_ema + (1-d) w) is what
    is evaluated, kept as checkpoints and returned: averaging smooths the
    epoch-to-epoch swings of the greedy policy.
    override_weight: the loss weight of the decisions where the planner
    departed from its base policy (t_chosen != t_base), relative to the
    others: they are few, and copying the base already fits the rest.
    adv_weight, ce_weight: the loss is ce_weight * the cross-entropy +
    adv_weight * advantage_loss (the network's expected advantage over the
    base, by the planner's option scores). set_labels: the cross-entropy is
    set_class_nll instead (the label is the set t_set; graphs recorded
    without one use their chosen class). renorm: the cross-entropy uses class
    probabilities renormalized within the deciding component
    (soft_class_nll(renorm=True)).
    Returns the history."""
    import copy
    soft_targets(graphs, tau)
    if set_labels:
        for g in graphs:
            if not hasattr(g, 't_set'):
                g.t_set = torch.arange(int(g.t_ncls)) == int(g.t_chosen)
    if override_weight != 1.0:
        for g in graphs:
            g.t_w = torch.tensor([override_weight if int(g.t_chosen) != int(g.t_base) else 1.0])
    else:
        for g in graphs:
            if hasattr(g, 't_w'):
                del g.t_w
    from torch_geometric.data import Batch
    dev = torch.device(device) if device else torch.device('cpu')
    rng = random.Random(seed)
    with torch.no_grad():
        policy({'graph': Batch.from_data_list(graphs[:2])})     # materialise lazy layers
    policy.to(dev)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    hist = {"loss": [], "evals": [], "checkpoints": {}, "tau": tau, "ema": ema}
    avg = copy.deepcopy(policy) if ema else None

    def _averaged():
        """Swap the averaged weights in (returns the live ones to swap back)."""
        live = copy.deepcopy(policy.state_dict())
        policy.load_state_dict(avg.state_dict())
        return live
    for ep in range(1, epochs + 1):
        policy.train()
        order = list(range(len(graphs)))
        rng.shuffle(order)
        losses = []
        for i in range(0, len(order), batch_size):
            b = Batch.from_data_list([graphs[j] for j in order[i:i + batch_size]]).to(dev)
            if set_labels:
                ce_term = set_class_nll(policy, b)
            else:
                ce_term = soft_class_nll(policy, b, renorm=renorm)
            loss = ce_weight * ce_term if ce_weight else 0.0
            if adv_weight:
                loss = loss + adv_weight * advantage_loss(policy, b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if avg is not None:
                with torch.no_grad():
                    for pa, pw in zip(avg.parameters(), policy.parameters()):
                        pa.mul_(ema).add_(pw, alpha=1 - ema)
            losses.append(float(loss))
        hist["loss"].append(float(np.mean(losses)))
        msg = f"epoch {ep}: loss {hist['loss'][-1]:.3f}"
        if evaluate is not None and (ep % eval_every == 0 or ep == epochs):
            live = _averaged() if avg is not None else None
            policy.to('cpu').eval()
            ev = evaluate(policy)
            hist["evals"].append({"epoch": ep, **ev})
            hist["checkpoints"][ep] = copy.deepcopy(policy.state_dict())
            if live is not None:
                policy.load_state_dict(live)
            msg += " | " + ", ".join(f"{k} {v:.3f}" for k, v in ev.items() if isinstance(v, float))
            policy.to(dev)
        if log:
            log(msg)
    policy.to('cpu').eval()
    if avg is not None:
        policy.load_state_dict(avg.to('cpu').state_dict())
    if select is not None and hist["evals"]:
        best = max(hist["evals"], key=lambda e: e[select])
        hist["selected_epoch"] = best["epoch"]
        policy.load_state_dict(hist["checkpoints"][best["epoch"]])
    return hist

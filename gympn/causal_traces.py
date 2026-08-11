class TransitionHistory:
    """
    Tracks the history of transitions fired in the Petri net simulation, together with the transition type (action or evolution), the tokens used to fire, the produced tokens, and the generated reward.
    - Uses transition._id as unique transition id.
    """

    def __init__(self):
        self.transitions = []

    def add_transition(self, transition, input_tokens, output_tokens, is_action, created_by=None, reward=0.0, time=None):
        self.transitions.append({
                "transition": transition,
                "is_action": is_action,
                "input_tokens": [t._id for t in input_tokens],
                "output_tokens": [t._id for t in output_tokens],
                "created_by": created_by,
                "reward": reward,
                "time": time
            })

    def get_action_transitions_len(self):
        return len([t for t in self.transitions if t["is_action"]])

    def get_action_transitions(self):
        return [t for t in self.transitions if t["is_action"]]

    def flush(self):
        self.transitions = []

class TokenHistory:
    """
    Tracks token lineage and causal relationships for reward redistribution.
    - Uses Python's id(token) as unique token id.
    - Records: parents (list of token ids), creating transition id, created_by (action index or None), time.
    """

    def __init__(self):
        self.tokens = {}  # token_id -> dict
        self.transition_to_tokens = {}  # transition_id -> list of token_ids

    def add_token(self, token, transition, parent_tokens, created_by=None, time=None):
        tid = token._id
        parent_ids = []
        try:
            parent_ids = [p._id for p in parent_tokens]
        except AttributeError:
            print("Error: One of the parent tokens does not have an _id attribute.")
        self.tokens[tid] = {
            "token": token,
            "parents": parent_ids,
            "event": transition._id,
            #"created_by": created_by,
            "time": time
        }

        # Register token under its creating transition
        if transition._id not in self.transition_to_tokens:
            self.transition_to_tokens[transition._id] = []
        self.transition_to_tokens[transition._id].append(tid)

        return tid

    def get_token(self, token_id):
        return self.tokens.get(token_id)

    def get_parents(self, token_id):
        token_info = self.tokens.get(token_id)
        return token_info.get("parents", []) if token_info else []

    def get_tokens_by_transition(self, transition_id):
        return self.transition_to_tokens.get(transition_id, [])

    def get_causal_chain(self, token_id):
        """
        Returns a list of lists; each list contains token ids at one 'generation' step
        starting with the token itself, then its parents, then parents of those parents, etc.
        """
        chain = []
        visited = set()

        def layer(current_ids):
            next_ids = []
            group = []
            for tid in current_ids:
                if tid in visited:
                    continue
                visited.add(tid)
                group.append(tid)
                token_info = self.tokens.get(tid)
                if token_info:
                    parents = token_info.get("parents", [])
                    next_ids.extend(parents)
            if group:
                chain.append(group)
            if next_ids:
                layer(next_ids)

        layer([token_id])
        return chain




class CausalTraces:
    """
    Manages causal traces for tokens in a Petri net simulation.
    - Records token histories and supports reward redistribution based on causal chains.
    """

    def __init__(self):
        self.token_history = TokenHistory()
        self.transition_history = TransitionHistory()
        # When True, postpone sentinels are token-flow transitions whose output
        # tokens are real credit sinks (set by the simulator from its
        # causal_postpone_tokenflow flag). redistribute_rewards() defaults
        # include_postpone to this when the caller does not specify it.
        self.postpone_tokenflow = False
        # Back-reference to the GymProblem, set by the simulator once the net is
        # fully built. Used only by the s_ccf scheme, which needs the STATIC net
        # topology (actions/events/reward transitions and their arcs) to compute
        # action-invariant causal components. The static-component map is cached
        # since the topology does not change during training.
        self._pn = None
        self._static_comp_cache = None
        # ls_hca (FORKFREE_LINEAGE_RETHINK.md Idea 1): PURE/CONTESTED
        # reward-type classification per action-type, cached like
        # _static_comp_cache (same static topology, same invalidation site).
        self._ls_hca_classify_cache = None
        # Per-decision CONTESTED ingredients from the last _redistribute_ls_hca
        # call: list of (action_type, reward_type, in_lineage, contribution)
        # per action transition, UNCORRECTED (no hindsight factor applied --
        # that needs pi(a|s), which this trace object never sees; see
        # gympn.agents.Agent.run_episode's ls_hca combine step).
        self._ls_hca_pending = None

    def flush(self):
        # Preserve config flags (postpone_tokenflow) across episode resets;
        # only the recorded histories are cleared.
        self.token_history = TokenHistory()
        self.transition_history = TransitionHistory()

    def stats(self):
        """Return simple stats: (n_tokens, n_transitions)."""
        return len(self.token_history.tokens), len(self.transition_history.transitions)

    def register_token(self, token, transition, parent_tokens, created_by=None, time=None):
        """
        Adds a token to the history and returns its unique ID.
        """
        return self.token_history.add_token(token, transition, parent_tokens, created_by, time)

    def register_transition(self, transition, input_tokens, output_tokens, is_action, created_by=None, reward=0.0, time=None):
        """
        Adds a transition to the history and returns its unique ID.
        """
        return self.transition_history.add_transition(transition, input_tokens, output_tokens, is_action, created_by, reward, time)

    def redistribute_rewards(self, scheme="lrq", include_postpone: bool = None,
                             beta: float = 0.0, values=None, lam: float = 1.0):
        """
        Compute per-decision causal credit from the recorded traces.

        The only implemented scheme is "lrq" (Lineage-Restricted Q); every
        earlier scheme (flow_dag, shapley_dag, rec, flow and the delay/depth
        heuristics) has been removed — see CAUSAL_LRQ_PROPOSAL.md and
        CAUSAL_REC_CRITICAL_REVIEW.md for why they were superseded.

        "lrq" is NOT a redistribution — it is a per-decision hindsight
        *Q-sample*. Each decision receives the FULL discounted sum of every
        future reward in whose causal lineage it sits (no split), including
        postpone decisions (requires token-flow postpone,
        ``causal_postpone_tokenflow=True``, whenever postpone actions are
        present). The vector must be consumed as ``A_t = Q_t - V(s_t)`` with a
        value head regressed on the same quantity (data.py's causal branch),
        NEVER summed as a return: a reward with k lineage decisions is counted
        k times by design (per-step Q-sampling of the policy-gradient theorem).
        Exogenous rewards (empty lineage) enter no decision's Q — they are
        uncontrollable and correctly ignored. See ``_redistribute_lrq``.

        :param scheme: must be "lrq" (kept as a parameter so removed-scheme
                      callers fail with a clear error).
        :param include_postpone: if True, postpone sentinel output tokens are
                      lineage members (token-flow postpone, Design C). When left
                      as ``None`` (default) it resolves to
                      ``self.postpone_tokenflow`` (set by the simulator), so the
                      simulator's ``causal_postpone_tokenflow`` flag is the
                      single source of truth. LRQ refuses to run with postpone
                      actions present and this flag off.
        :param beta: SMDP discount rate for the reward-to-decision discount
                      ``e**(-beta*(t_j-u_t))``. MUST equal the ``causal_beta``
                      used by the downstream SMDP machinery so the Q-sample and
                      the sojourn discounts share one clock and one rate.
                      ``0.0`` => undiscounted (time carries no opportunity cost).
        :return: List of per-decision Q-samples aligned with action transitions.
        """

        # Resolve include_postpone default from the trace-level config flag set by
        # the simulator (causal_postpone_tokenflow), so callers need not thread it.
        if include_postpone is None:
            include_postpone = getattr(self, "postpone_tokenflow", False)

        # Keep full ordered list of action transitions (so redistribution indexes match
        # the simulator's action ordering), but do NOT assign credit to transitions
        # whose transition._id starts with 'postpone_'. We detect postpone sentinels
        # and skip mapping their output tokens to actions so they receive zero credit.
        action_transitions = self.transition_history.get_action_transitions()
        redistribution = [0.0] * len(action_transitions)

        # Map token -> (action_index, action_time)
        token_to_action = {}
        # Map action transition *record* (by identity) -> (action_index, is_postpone).
        # Lets LRQ recognise a rewarding transition that is itself an action.
        # Keyed by id() because the same SimAction object may fire many
        # times (sharing transition._id), so _id is not a unique record key.
        record_to_action = {}
        # Build a map token -> (action_index, action_time).
        # Optionally exclude postpone sentinel transitions from receiving credit
        # for backward compatibility.
        for idx, act in enumerate(action_transitions):
            tr_obj = act.get('transition')
            tr_id = None
            try:
                tr_id = getattr(tr_obj, '_id', None)
            except Exception:
                tr_id = None
            if tr_id is None:
                tr_id = act.get('transition')
            is_postpone = isinstance(tr_id, str) and tr_id.startswith('postpone_')
            record_to_action[id(act)] = (idx, is_postpone)
            if is_postpone and not include_postpone:
                continue
            for tid in act.get("output_tokens", []):
                token_to_action[tid] = (idx, act.get("time"))

        # Helper: safe accessors
        def get_parents(token_id):
            t = self.token_history.get_token(token_id)
            return t.get("parents", []) if t else []

        # LRQ: per-decision hindsight lineage return (a Q-sample, not a
        # redistribution). Consumed as A_t = Q_t - V(s_t); see _redistribute_lrq.
        if scheme == "lrq":
            return self._redistribute_lrq(
                action_transitions, token_to_action, record_to_action,
                redistribution, beta, get_parents, include_postpone
            )

        # LRQ-v2: same lineage Q-samples for PRODUCTION actions, but postpone
        # receives NO lineage credit (the walk still passes through its
        # re-emitted tokens to the upstream producers). Postpone's advantage
        # is instead the SMDP-TD form e^{-beta*tau}V(s')-V(s), applied by
        # data.py's finish() — the consistent-support fix for the
        # postpone-aggregation collapse (a token-flow postpone re-emits the
        # whole marking, so its v1 Q-sample equals the discounted BACKLOG MASS
        # and dominates any single production action's reward in loaded,
        # reward-dense systems).
        if scheme == "lrq2":
            return self._redistribute_lrq(
                action_transitions, token_to_action, record_to_action,
                redistribution, beta, get_parents, include_postpone,
                exclude_postpone_credit=True
            )

        # MC-Q: the lineage ABLATION of LRQ. Same Q-sample estimator and
        # consumption, but the lineage test is dropped — every decision's Q is
        # the FULL discounted return-to-go from its own clock. Isolates what
        # the lineage restriction itself contributes. See _redistribute_mcq.
        if scheme == "mc_q":
            return self._redistribute_mcq(action_transitions, redistribution, beta)

        # CGAE: GAE run along the CAUSAL successor instead of the time successor.
        # 'cgae_flow' is the variant Proposition 3(iii) actually covers: the
        # successor combination is a FLOW-WEIGHTED SUM rather than a mean, which
        # drops the single-successor assumption (A2) from the theorem. On a
        # chain the two coincide exactly, so they differ only under fan-out.
        if scheme in ("cgae", "cgae_flow"):
            return self._redistribute_cgae(
                action_transitions, token_to_action, record_to_action,
                redistribution, beta, get_parents, values, lam,
                flow=(scheme == "cgae_flow"))

        # ALIN: anti-lineage. mc_q MINUS the rewards a decision provably could
        # not have influenced. Subtraction, not restriction -- see
        # _redistribute_alin for why that distinction is the whole point.
        if scheme == "alin":
            return self._redistribute_alin(
                action_transitions, token_to_action, record_to_action,
                redistribution, beta, get_parents)

        # CCF: Causal-Component-Factored return-to-go (LINEAGE_SPARSE_CORRECTION
        # follow-up). Partition decisions into causally-connected COMPONENTS
        # (union-find over reward lineages — captures both token-flow AND
        # resource contention, since a freed resource re-emits into the DAG),
        # then give each decision the return-to-go RESTRICTED TO ITS COMPONENT.
        # UNBIASED within a component (full return-to-go, sees foreclosure) and
        # variance-filtered across components (cross-component reward is
        # independent of the action, a valid baseline). One component => full
        # return-to-go (= mc_q, matches PPO, no lineage bias); independent
        # components => per-case filter (= lrq's win). Strictly dominates lrq.
        if scheme == "ccf":
            return self._redistribute_ccf(action_transitions, token_to_action,
                                          record_to_action, redistribution, beta,
                                          get_parents)

        # S-CCF: STATIC-component-factored return-to-go. Same idea as ccf, but the
        # component partition is computed ONCE from the net TOPOLOGY (which
        # decisions could co-cause a reward under ANY action, grouping a
        # decision's competing actions), not the realized trajectory. Because
        # membership is action-invariant, the excluded (cross-component) rewards
        # are unreachable from the decision under any action -> action-independent
        # -> a valid baseline -> UNBIASED BY CONSTRUCTION (assumption A1 holds
        # automatically). Fixes ccf's action-dependent-membership bias at
        # AND-joins / shared-resource handoffs while keeping the variance
        # reduction on statically-independent structure. See assembly_probe.py.
        if scheme == "s_ccf":
            return self._redistribute_s_ccf(action_transitions, record_to_action,
                                            redistribution, beta)

        # LS-HCA: Lineage-Structured Hindsight Credit Assignment (FORKFREE_
        # LINEAGE_RETHINK.md Idea 1) -- the fork-free dual of the DAG-replay
        # counterfactual ('cf'). Returns the PURE-only credit here (exact,
        # static classification, no fit); the CONTESTED reward-types' hindsight
        # correction needs pi(a|s), which this trace never sees, so it is
        # applied by the caller (Agent.run_episode) from self._ls_hca_pending
        # -- see _redistribute_ls_hca's docstring.
        if scheme == "ls_hca":
            return self._redistribute_ls_hca(action_transitions, record_to_action,
                                             redistribution, beta, token_to_action,
                                             get_parents)

        raise ValueError(
            f"Unknown scheme: {scheme!r}. All redistribution schemes except 'lrq' "
            f"have been removed (flow_dag/shapley_dag/rec/flow/exponential/linear/"
            f"uniform/depth/hybrid). See CAUSAL_LRQ_PROPOSAL.md."
        )

    # ------------------------------------------------------------------ #
    # MC-Q: the no-lineage ablation of LRQ                               #
    # ------------------------------------------------------------------ #

    def _redistribute_alin(self, action_transitions, token_to_action,
                           record_to_action, redistribution, beta, get_parents):
        """ALIN — anti-lineage credit: subtract what a decision provably could
        NOT have influenced, instead of keeping only what it provably did.

            Q_d = SUM_{r : d can reach r}  r * e**(-beta * (t_r - u_d))

        equivalently mc_q minus the rewards with no causal path from d.

        WHY THIS SHAPE. `lrq` keeps the realized token lineage and discards
        everything else -- but "not in d's lineage" is NOT "independent of d".
        A decision that occupies a shared employee delays a case whose tokens
        it never touched (foreclosure); that reward is action-DEPENDENT and
        `lrq` drops it, which is a dropped gradient term, not a baseline.
        Measured cost of exactly that: lrq loses to its own lineage ablation
        mc_q by -1.80 on s1_stoch_sequence (p<.001) and -89.50 on
        s2_stoch_scaled (p<.001). This inverts the operation: keep the full
        return and subtract only terms with NO causal path, which is a valid
        baseline (independent of the action) and therefore variance reduction
        without bias.

        REACHABILITY. Two edge types over the space-time interaction graph:
          - lineage:    d -> r when d sits in r's realized token lineage.
          - contention: d -> d' when d' acquires a token from a RESOURCE POOL
                        that d is also drawing from, at a time at or after
                        d's own acquisition -- i.e. d' could have been made to
                        wait by d. Only the NEXT such acquirer is linked; the
                        transitive closure then covers all later ones, which
                        keeps the graph sparse without weakening the claim.
        A reward is subtracted from d only when it is unreachable under BOTH.
        Pools are detected structurally (`_is_cyclic_place`: a place whose own
        consumption chain leads back to itself), the same test `s_ccf` and
        LS-HCA's classifier use, so nothing here is task-assignment-specific.

        SCOPE. Soundness rests on contention propagating through acquisition
        ordering within a pool. Its teeth are where realized coupling is
        sparser than static coupling, i.e. K_realized > K_static (s2: 6 vs 1).

        WHAT IT DOES *NOT* DO, corrected 2026-08-11. This used to claim that
        where every decision draws on one saturated pool, every decision
        reaches every reward and alin degenerates to mc_q EXACTLY (s1, s3).
        That is false, and not for want of a better closure: contention edges
        run FORWARD in time, so a reward whose lineage completed BEFORE d is
        structurally unreachable from d -- and should be, since that case was
        already in service when d chose, so d neither started it nor could
        delay it. mc_q counts it only because it lands later on the wall clock.
        Measured on s1 (suite/_diag_alin_p3b.py, 6 episodes, 922 decision-
        reward pairs in the mc_q horizon): 748 reached; of the 174 missed, 133
        (76.4%) are exactly this already-in-flight case, correctly subtracted.

        The other 41 (23.6%) are a REAL hole: a lineage decision fired at or
        after d, yet d does not reach it. Ruled out as causes -- postpone
        sentinels sitting outside the pool chains (0 of the 41 are reachable
        only via a postpone node) and the time-tie ordering (fixed below; the
        closure now runs 0 fallbacks). Most likely the "next acquirer only"
        linking, which cannot express that d delayed a case whose starting
        decision draws on a pool d never touches. The gap is conservative --
        alin subtracts slightly MORE than it should, costing variance
        reduction rather than adding bias, which is why P1/P2 in
        suite/_test_alin.py stay clean -- but it means the headline claim over
        lrq (that alin recovers foreclosure) is only partially realized on
        shared-pool envs: s1 recovers 16.7% of the lrq2 -> mc_q gap, s2 21.9%.

        Needs the net topology for pool detection (``self._pn``); without it,
        falls back to lineage edges only.
        """
        import math

        def disc(t_reward, u_dec):
            if beta == 0.0 or t_reward is None or u_dec is None:
                return 1.0
            return math.exp(-beta * max(0.0, float(t_reward) - float(u_dec)))

        n = len(action_transitions)
        if n == 0:
            return redistribution

        # ---- rewards and their realized lineages ------------------------ #
        rewards = []
        for tr in self.transition_history.transitions:
            rv = tr.get('reward', 0.0)
            if rv == 0.0:
                continue
            firing_idx = None
            se = record_to_action.get(id(tr))
            if se is not None:
                firing_idx = se[0]
            rewards.append((rv, tr.get('time'), tr.get('input_tokens', []), firing_idx))

        def lineage_decisions(input_ids, firing_idx):
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

        direct = [set() for _ in range(n)]     # decision -> reward indices
        for j, (_rv, _t, in_ids, firing_idx) in enumerate(rewards):
            for idx in lineage_decisions(in_ids, firing_idx):
                if 0 <= idx < n:
                    direct[idx].add(j)

        # ---- contention successors: next acquirer of a shared pool ------- #
        succ = [set() for _ in range(n)]
        pn = self._pn
        if pn is not None:
            trans = list(pn.actions) + list(pn.events)
            consumers = {}
            for t in trans:
                for p in t.incoming:
                    consumers.setdefault(p._id, []).append(t)

            def _is_cyclic_place(pid, seeds):
                seen_p, seen_t, stack = set(), set(), list(seeds)
                while stack:
                    t = stack.pop()
                    if t._id in seen_t:
                        continue
                    seen_t.add(t._id)
                    for p in t.outgoing:
                        if p._id == pid:
                            return True
                        if p._id in seen_p:
                            continue
                        seen_p.add(p._id)
                        stack.extend(consumers.get(p._id, []))
                return False

            pool_ids = {pid for pid in consumers
                        if _is_cyclic_place(pid, consumers.get(pid, []))}
            # which pools each decision draws from, and when
            by_pool = {}
            for idx, act in enumerate(action_transitions):
                tobj = act.get('transition')
                if tobj is None:
                    continue
                u = act.get('time')
                for p in getattr(tobj, 'incoming', ()):
                    if p._id in pool_ids:
                        by_pool.setdefault(p._id, []).append(
                            (float(u) if u is not None else 0.0, idx))
            for pid, entries in by_pool.items():
                entries.sort()
                for k in range(len(entries) - 1):
                    succ[entries[k][1]].add(entries[k + 1][1])

        # ---- reachable reward sets, back-to-front over time -------------- #
        # The index is part of the sort key, and that is load-bearing. Every
        # contention edge runs from a smaller (time, index) to a larger one --
        # `entries` is sorted by exactly that key and only consecutive pairs
        # are linked -- so descending (time, index) is a true reverse
        # topological order and each successor's `reach` is always ready.
        #
        # Sorting on time ALONE (the previous version) left same-time
        # decisions in ascending index order, so a decision was processed
        # BEFORE its same-time successor and the guard below fell back to
        # `direct[s]`, silently truncating the transitive closure. That is not
        # a corner case where it matters most: on s1's 3-employee pool, 36.4%
        # of contention edges join same-time decisions, 63 fallbacks fired
        # over 6 episodes, and mean reachability was 0.677 of the mc_q horizon
        # with only 29.1% of decisions reaching everything -- against the
        # documented claim that a single saturated pool degenerates to mc_q
        # EXACTLY. See suite/_diag_alin_p3.py.
        order = sorted(range(n),
                       key=lambda i: (action_transitions[i].get('time') is None,
                                      action_transitions[i].get('time') or 0.0,
                                      i),
                       reverse=True)
        reach = [None] * n
        for idx in order:
            acc = set(direct[idx])
            for s_ in succ[idx]:
                if reach[s_] is not None:
                    acc |= reach[s_]
                else:                       # cycle guard: fall back to direct
                    acc |= direct[s_]
            reach[idx] = acc

        for idx, act in enumerate(action_transitions):
            u = act.get('time')
            tot = 0.0
            for j in reach[idx]:
                rv, t_j, _in, _fi = rewards[j]
                if t_j is None or u is None or u <= t_j:
                    tot += rv * disc(t_j, u)
            redistribution[idx] = tot
        return redistribution

    def component_step_rewards(self, include_postpone: bool = None):
        """Ingredients for component-filtered GAE (scheme ``cfgae``).

        Returns ``(comp_of_decision, step_rewards)`` where

          * ``comp_of_decision[d]``  -- component id of decision ``d`` (ccf's
            realized union-find partition: two decisions are merged when they
            both sit in some reward's causal lineage);
          * ``step_rewards[t]``      -- ``(reward_value, component_id)`` pairs
            credited to step ``t``, i.e. to the decision that was in effect
            when the reward fired (the latest decision with ``u_d <= t_j``).

        The caller masks the step-reward stream to one component at a time and
        runs the ORDINARY SMDP-GAE recursion over it. That is the whole point:
        with a single component the mask is identically one and the recursion
        is term-for-term PPO's, so the K=1 limit is PPO EXACTLY rather than
        mc_q (which is what every Monte-Carlo-return scheme here degenerates
        to, and why none of them can promise "no worse than PPO").

        Decisions touched by no reward form their own singleton components;
        rewards with no lineage decision are dropped (no action could have
        influenced them, so they are a constant and cannot affect the
        gradient).
        """
        action_transitions = self.transition_history.get_action_transitions()
        n = len(action_transitions)
        if n == 0:
            return [], []

        token_to_action = {}
        for idx, act in enumerate(action_transitions):
            for t in act.get("output_tokens", ()) or ():
                token_to_action[t] = idx

        def get_parents(tid):
            info = self.token_history.get_token(tid)
            return info.get("parents", []) if info else []

        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        def lineage(input_ids):
            found, seen, stack = set(), set(), list(input_ids)
            while stack:
                tid = stack.pop()
                if tid in seen:
                    continue
                seen.add(tid)
                hit = token_to_action.get(tid)
                if hit is not None:
                    found.add(hit)
                for p in get_parents(tid):
                    if p not in seen:
                        stack.append(p)
            return found

        rewards = []
        for tr in self.transition_history.transitions:
            rv = tr.get("reward", 0.0)
            if rv == 0.0:
                continue
            decs = sorted(lineage(tr.get("input_tokens", [])))
            if not decs:
                continue
            for d in decs[1:]:
                union(decs[0], d)
            rewards.append((float(rv), tr.get("time"), decs[0]))

        # Only decisions that actually sit in some reward's lineage define
        # components. A decision that influenced NO reward has no causal
        # component of its own -- making it a singleton would give it an
        # all-zero masked reward stream, which is emphatically not PPO, and on
        # s1 inflated K from 1 to 9-10 (caught by the exactness check). Such a
        # decision inherits the component of the nearest preceding touched
        # decision, so with a single real component every step carries the same
        # label, the mask is identically one, and the K=1 reduction is exact.
        touched = set()
        for _rv, _t, anchor in rewards:
            for d in range(n):
                if find(d) == find(anchor):
                    touched.add(d)
        roots = {}
        for d in sorted(touched):
            r = find(d)
            if r not in roots:
                roots[r] = len(roots)
        if not roots:
            roots[find(0)] = 0
        comp = []
        last = 0
        for d in range(n):
            r = find(d)
            if r in roots:
                last = roots[r]
            comp.append(last)

        times = [act.get("time") for act in action_transitions]
        step_rewards = [[] for _ in range(n)]
        for rv, t_j, anchor in rewards:
            cid = comp[find(anchor)] if find(anchor) < n else comp[anchor]
            cid = comp[anchor]
            if t_j is None:
                step_rewards[anchor].append((rv, cid))
                continue
            # the decision in effect when the reward fired
            step = None
            for d in range(n):
                if times[d] is not None and times[d] <= t_j:
                    step = d if step is None or (times[d] >= times[step]) else step
            step_rewards[step if step is not None else anchor].append((rv, cid))
        return comp, step_rewards

    def _redistribute_cgae(self, action_transitions, token_to_action,
                           record_to_action, redistribution, beta, get_parents,
                           values=None, lam=1.0, flow=False):
        """CGAE — GAE propagated along the CAUSAL successor, not the time one.

        Standard GAE accumulates TD errors along the trajectory index:
        ``A_t = delta_t + (gamma*lam) A_{t+1}``. In an interleaved queueing
        system consecutive decisions usually belong to DIFFERENT cases, so
        ``delta_{t+1}`` is mostly noise with respect to the action at ``t``:
        GAE is propagating credit along wall-clock order while causality flows
        along the provenance DAG. This runs the identical recursion along the
        DAG instead:

            A_d = delta_d + (e^{-beta*dt} * lam) * mean_{d' in succ(d)} A_{d'}
            delta_d = r_d + e^{-beta*dt} V(s_{d'}) - V(s_d)

        where ``succ(d)`` are the decisions whose inputs descend from ``d``'s
        outputs, and ``r_d`` is the reward OWNED by ``d`` -- each reward is
        assigned to the latest decision in its own lineage, so every reward is
        counted exactly once and then flows backward through the recursion
        rather than being duplicated across the whole lineage (which is what
        makes \\lrq{} a Q-sample rather than a return).

        lam=1 with no value model degenerates to a causal-chain Monte-Carlo
        return; lam<1 bootstraps through the critic at causal depth. Returns a
        Q-sample (``A_d + V(s_d)``) so it is consumed exactly like every other
        scheme here.

        General to any A-E PN: successors and ownership are read off the token
        DAG, no token-value semantics and no task-assignment assumptions.

        ``flow`` (scheme ``cgae_flow``) replaces the mean over successors with
        the FLOW-WEIGHTED SUM of Proposition 3(iii) in
        suite/EJOR_PROPOSITIONS.md. The weight of edge d->s is the share of
        s's consumed tokens that descend from d, so

            sum over d in pred(s) of w(d->s) = 1                          (*)

        exactly. (*) is what makes the sum safe: the credit flowing backward
        out of s totals A_s however many predecessors it has, so unrolling
        gives every owned reward total weight 1 across all paths and nothing is
        double counted -- which is precisely what the plain sum gets wrong on a
        DAG and what the mean "fixes" by shrinking instead.

        WHY THE MEAN IS NOT THE RIGHT DEFAULT. With one successor, mean = sum =
        the single term, so both agree and both reduce to ordinary SMDP-GAE.
        With fan-out k, a decision that genuinely enables k independent
        downstream continuations should receive all of them; the mean divides
        that by k. It is a shrinkage with no derivation, and Proposition 3
        does not cover it.

        WHY IT IS STILL THE DEFAULT HERE. The bootstrap term is the catch. V is
        a GLOBAL critic (it scores the whole marking), so summing V(s) over
        several successors multiply-counts everything outside the decision's
        component; the mean is a scale guard on that term. Proposition 3(ii)
        assumes the component-local V for exactly this reason (assumption A2').
        Under ``flow`` the discount is also folded per successor rather than
        averaged, which is the more faithful SMDP form regardless.

        Empirically the choice is nearly inert where the cgae result lives:
        fan-out is 1.07 on ncopies N=4 (93% of decisions have <=1 successor).
        It matters on s1 (1.34, 34.5% multi-successor), where cgae measures as
        null against PPO anyway.
        """
        import math

        n = len(action_transitions)
        if n == 0:
            return redistribution
        V = list(values) if values is not None else [0.0] * n
        if len(V) < n:
            V = V + [0.0] * (n - len(V))

        def disc(dt):
            if beta == 0.0 or dt is None:
                return 1.0
            return math.exp(-beta * max(0.0, float(dt)))

        # ---- causal successors: d -> decisions consuming d's descendants --- #
        out_tok = {}
        for idx, act in enumerate(action_transitions):
            for t in act.get('output_tokens', ()) or ():
                out_tok[t] = idx
        succ = [set() for _ in range(n)]
        w_edge = {}                            # (d, s) -> flow weight, `flow` only

        def _producers(start_tid, self_idx):
            """Nearest producing decisions reachable backward from ONE token."""
            found, seen, stack = set(), set(), [start_tid]
            while stack:
                tid = stack.pop()
                if tid in seen:
                    continue
                seen.add(tid)
                src = out_tok.get(tid)
                if src is not None and src != self_idx:
                    found.add(src)
                    continue                   # nearest producer only
                for p in get_parents(tid):
                    if p not in seen:
                        stack.append(p)
            return found

        for idx, act in enumerate(action_transitions):
            inputs = list(act.get('input_tokens', ()) or ())
            if not flow:
                seen, stack = set(), list(inputs)
                while stack:                   # walk back to the producing decision
                    tid = stack.pop()
                    if tid in seen:
                        continue
                    seen.add(tid)
                    src = out_tok.get(tid)
                    if src is not None and src != idx:
                        succ[src].add(idx)
                        continue               # nearest producer only
                    for p in get_parents(tid):
                        if p not in seen:
                            stack.append(p)
                continue

            # flow: attribute each consumed token to its producer(s) SEPARATELY,
            # so the shares are counts of tokens rather than a pooled set. A
            # token reaching several nearest producers splits its unit mass
            # evenly among them, which keeps (*) exact.
            contrib, n_attributed = {}, 0
            for tid in inputs:
                prods = _producers(tid, idx)
                if not prods:                  # initial/exogenous token: no producer
                    continue
                n_attributed += 1
                share = 1.0 / len(prods)
                for d_ in prods:
                    contrib[d_] = contrib.get(d_, 0.0) + share
            if n_attributed == 0:
                continue
            for d_, c in contrib.items():
                succ[d_].add(idx)
                w_edge[(d_, idx)] = c / n_attributed   # sums to 1 over d_

        # ---- reward ownership: latest decision in the reward's lineage ----- #
        times = [act.get('time') for act in action_transitions]

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

        owned = [0.0] * n
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
            owned[owner] += rv

        # ---- backward recursion over the DAG, latest decision first -------- #
        order = sorted(range(n), key=lambda i: (times[i] is not None,
                                                times[i] or 0.0), reverse=True)
        A = [0.0] * n
        for d in order:
            nxt = [s for s in succ[d] if s != d]
            if not nxt:
                A[d] = owned[d] - V[d]
                continue
            dts = {s: ((times[s] - times[d]) if (times[s] is not None
                       and times[d] is not None) else 0.0) for s in nxt}
            if flow:
                # Proposition 3(iii): weighted SUM, discount folded per
                # successor rather than averaged across them.
                v_next = sum(w_edge.get((d, s), 0.0) * disc(dts[s]) * V[s] for s in nxt)
                a_next = sum(w_edge.get((d, s), 0.0) * disc(dts[s]) * A[s] for s in nxt)
                A[d] = owned[d] + v_next - V[d] + lam * a_next
            else:
                g = sum(disc(dt) for dt in dts.values()) / len(nxt)
                v_next = sum(V[s] for s in nxt) / len(nxt)
                a_next = sum(A[s] for s in nxt) / len(nxt)
                A[d] = owned[d] + g * v_next - V[d] + g * lam * a_next

        for d in range(n):
            redistribution[d] = A[d] + V[d]       # consumed as Q, like every scheme
        return redistribution

    def _redistribute_mcq(self, action_transitions, redistribution, beta):
        """
        MC-Q — LRQ with the lineage restriction removed (the ablation that
        prices the lineage itself; see PAPER_PLAN_LRQ.md).

        For each decision t:

            Q_t = Σ_{j : t_j >= u_t}  r_j · e**(-beta * (t_j - u_t))

        i.e. the plain Monte-Carlo SMDP return-to-go from the decision's own
        clock — equivalently a PPO variant with lambda=1 (no bootstrap) and
        wall-clock e^{-beta*tau} discounting. Every decision (postpone
        included — there is no causal filtering, that is the point) sees every
        subsequent reward, whether or not it caused it. Consumed identically
        to LRQ (A_t = Q_t - V(s_t), value head regressed on Q_t). Needs no
        token DAG and no token-flow postpone. Missing timestamps are included
        undiscounted (conservative, same policy as LRQ's disc()).
        """
        import math

        decision_time = {idx: act.get("time")
                         for idx, act in enumerate(action_transitions)}

        def disc(t_reward, u_dec):
            if beta == 0.0 or t_reward is None or u_dec is None:
                return 1.0
            return math.exp(-beta * max(0.0, float(t_reward) - float(u_dec)))

        for tr in self.transition_history.transitions:
            reward = tr.get("reward", 0.0)
            if reward == 0.0:
                continue
            t_j = tr.get("time")
            for idx, u in decision_time.items():
                if t_j is None or u is None or u <= t_j:
                    redistribution[idx] += reward * disc(t_j, u)

        return redistribution

    # ------------------------------------------------------------------ #
    # CCF: Causal-Component-Factored return-to-go                        #
    # ------------------------------------------------------------------ #

    def _redistribute_ccf(self, action_transitions, token_to_action,
                          record_to_action, redistribution, beta, get_parents):
        """Component-factored return-to-go (see the dispatch comment).

        Step 1: for each reward, find the decisions in its causal lineage and
                UNION them into one component (a reward's causes interact).
        Step 2: each decision's credit = sum of its component's rewards at or
                after its decision clock, discounted.

        Cross-component rewards are causally independent of the action (they
        share no lineage), so omitting them is a valid, variance-reducing
        baseline — the estimator stays unbiased. Postpone is left as a singleton
        component (no reward-causing descendants under token-flow-off); finish()
        gives it the SMDP-TD advantage via the postpone mask, as for lrq2.
        """
        import math

        n = len(action_transitions)
        parent = list(range(n))

        def find(x):
            r = x
            while parent[r] != r:
                r = parent[r]
            while parent[x] != r:
                parent[x], x = r, parent[x]
            return r

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        decision_time = {idx: act.get("time")
                         for idx, act in enumerate(action_transitions)}

        def lineage_decisions(input_ids, firing_idx):
            found = set()
            if firing_idx is not None:
                found.add(firing_idx)
            seen = set()
            stack = list(input_ids)
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

        # Step 1: collect rewards with their lineage decisions; union components.
        rewards_list = []  # (reward, t_j, representative_decision_idx or None)
        for tr in self.transition_history.transitions:
            reward = tr.get("reward", 0.0)
            if reward == 0.0:
                continue
            t_j = tr.get("time")
            firing_idx = None
            se = record_to_action.get(id(tr))
            if se is not None:
                firing_idx = se[0]
            decs = list(lineage_decisions(tr.get("input_tokens", ()), firing_idx))
            for i in range(1, len(decs)):
                union(decs[0], decs[i])
            rewards_list.append((reward, t_j, decs[0] if decs else None))

        def disc(t_reward, u_dec):
            if beta == 0.0 or t_reward is None or u_dec is None:
                return 1.0
            return math.exp(-beta * max(0.0, float(t_reward) - float(u_dec)))

        # Step 2: component return-to-go per decision.
        for idx in range(n):
            ci = find(idx)
            u = decision_time.get(idx)
            for (reward, t_j, rep) in rewards_list:
                if rep is None or find(rep) != ci:
                    continue
                # return-to-go: reward at/after the decision (missing clocks
                # included, same conservative convention as mc_q/lrq).
                if u is not None and t_j is not None and t_j < u:
                    continue
                redistribution[idx] += reward * disc(t_j, u)

        return redistribution

    # ------------------------------------------------------------------ #
    # S-CCF: static (action-invariant) component-factored credit          #
    # ------------------------------------------------------------------ #

    def _static_component_reward_types(self):
        """Map each action-type to the set of reward-transition-types in its
        STATIC causal component. Two decisions are in one component when they can
        statically reach a common reward transition (topological reachability of
        the net, over ALL of a decision's competing actions -- grouped by shared
        input place -- so membership is action-invariant). Cached; the net
        topology is fixed during training."""
        if self._static_comp_cache is not None:
            return self._static_comp_cache
        pn = self._pn
        if pn is None:
            raise ValueError(
                "s_ccf requires the net topology: set causal_trace._pn = <GymProblem> "
                "(the simulator does this in training_run) before redistribute.")

        trans = list(pn.actions) + list(pn.events)
        consumers = {}
        for t in trans:
            for p in t.incoming:
                consumers.setdefault(p._id, []).append(t)
        reward_types = set(pn.reward_functions.keys())

        def reaches(t):
            reached, seen_t, seen_p, stack = set(), set(), set(), list(t.outgoing)
            if t._id in reward_types:
                reached.add(t._id)
            while stack:
                p = stack.pop()
                if p._id in seen_p:
                    continue
                seen_p.add(p._id)
                for ct in consumers.get(p._id, []):
                    if ct._id in reward_types:
                        reached.add(ct._id)
                    if ct._id not in seen_t:
                        seen_t.add(ct._id); stack.extend(ct.outgoing)
            return reached

        reach_by_action = {a._id: reaches(a) for a in pn.actions}

        # Group competing actions (shared input place) into one decision point.
        ap = {a._id: a._id for a in pn.actions}
        def af(x):
            while ap[x] != x:
                ap[x] = ap[ap[x]]; x = ap[x]
            return x
        place_to_actions = {}
        for a in pn.actions:
            for p in a.incoming:
                place_to_actions.setdefault(p._id, []).append(a._id)
        for aids in place_to_actions.values():
            for k in range(1, len(aids)):
                ap[af(aids[0])] = af(aids[k])
        dp_reach = {}
        for aid, rr in reach_by_action.items():
            dp_reach.setdefault(af(aid), set()).update(rr)

        # Union decision points that can reach a common reward -> components.
        dps = list(dp_reach)
        dp = {d: d for d in dps}
        def df(x):
            while dp[x] != x:
                dp[x] = dp[dp[x]]; x = dp[x]
            return x
        for i in range(len(dps)):
            for j in range(i + 1, len(dps)):
                if dp_reach[dps[i]] & dp_reach[dps[j]]:
                    dp[df(dps[i])] = df(dps[j])
        comp = {}
        for d in dps:
            comp.setdefault(df(d), set()).update(dp_reach[d])
        self._static_comp_cache = {aid: comp[df(af(aid))] for aid in reach_by_action}
        return self._static_comp_cache

    def _redistribute_s_ccf(self, action_transitions, record_to_action,
                            redistribution, beta):
        """Credit each decision with the realized rewards of its STATIC component,
        at/after its decision clock, discounted. Unbiased by construction; see the
        dispatch comment and assembly_probe.py."""
        import math
        comp_rw = self._static_component_reward_types()
        base = self._pn._get_string_before_last_dot
        rewards = []
        for tr in self.transition_history.transitions:
            rv = tr.get("reward", 0.0)
            if rv == 0.0:
                continue
            tobj = tr.get("transition")
            rtype = base(getattr(tobj, "_id", "")) if tobj is not None else None
            rewards.append((rv, tr.get("time"), rtype))

        for idx, act in enumerate(action_transitions):
            rec = record_to_action.get(id(act))
            if rec is not None and rec[1]:      # postpone sentinel -> no lineage credit
                continue
            tobj = act.get("transition")
            a_type = base(getattr(tobj, "_id", "")) if tobj is not None else None
            allowed = comp_rw.get(a_type, set())
            u = act.get("time")
            c = 0.0
            for (rv, t_j, rtype) in rewards:
                if rtype in allowed and t_j is not None and u is not None and t_j >= u:
                    c += rv * math.exp(-beta * max(0.0, float(t_j) - float(u)))
            redistribution[idx] = c
        return redistribution

    # ------------------------------------------------------------------ #
    # LS-HCA: Lineage-Structured Hindsight Credit Assignment              #
    # ------------------------------------------------------------------ #

    def _pure_contested_reward_types(self):
        """Per action-TYPE: ``(PURE(d), CONTESTED(d))`` reward-transition-
        types, where ``d`` is the decision point the action belongs to
        (competing actions grouped by a shared CHOICE place -- see
        ``_is_cyclic_place`` below for why this deliberately does NOT reuse
        ``_static_component_reward_types``'s "any shared input place" rule).
        ``PURE(d)`` = reward-types reachable ONLY from ``d``'s own decision
        point -- deterministic, ``d``-caused, credited exactly, no fit.
        ``CONTESTED(d)`` = reward-types ``d`` shares reachability with at
        least one OTHER decision point -- the only place LS-HCA estimates a
        hindsight correction (see ``_redistribute_ls_hca`` and
        ``examples/paper_examples/suite/FORKFREE_LINEAGE_RETHINK.md`` Idea 1).

        This is a SOUND OVER-APPROXIMATION: the reachability walk flows
        through any place a reward could structurally pass, including another
        decision's resource hand-back, so it can flag a reward-type CONTESTED
        that is actually pure to ``d`` (measured on the abundant-resource M4
        motif in ls_hca_probe.py). That costs a wasted hindsight fit -- the
        empirical part still discriminates correctly -- never a bias (the
        same "safe degradation" property as ``s_ccf``'s static components).
        Cached; the net topology does not change during training."""
        if self._ls_hca_classify_cache is not None:
            return self._ls_hca_classify_cache
        pn = self._pn
        if pn is None:
            raise ValueError(
                "ls_hca requires the net topology: set causal_trace._pn = "
                "<GymProblem> (the simulator does this in training_run) "
                "before redistribute.")

        trans = list(pn.actions) + list(pn.events)
        consumers = {}
        for t in trans:
            for p in t.incoming:
                consumers.setdefault(p._id, []).append(t)
        reward_types = set(pn.reward_functions.keys())

        def reaches(t):
            reached, seen_t, seen_p, stack = set(), set(), set(), list(t.outgoing)
            if t._id in reward_types:
                reached.add(t._id)
            while stack:
                p = stack.pop()
                if p._id in seen_p:
                    continue
                seen_p.add(p._id)
                for ct in consumers.get(p._id, []):
                    if ct._id in reward_types:
                        reached.add(ct._id)
                    if ct._id not in seen_t:
                        seen_t.add(ct._id); stack.extend(ct.outgoing)
            return reached

        reach_by_action = {a._id: reaches(a) for a in pn.actions}

        # Group competing actions into one decision point ONLY when they share
        # a genuine CHOICE place (one case token routed to exactly one of
        # them) -- NOT when they merely share a RECYCLABLE RESOURCE place (a
        # pool: consumed, then produced again downstream, so many decision
        # instances draw from it over time -- e.g. a shared employee pool, or
        # M4's abundant R). Conflating the two was a real bug this method
        # shipped with: two decisions that only compete for a resource pool
        # (instance-level self-contention, e.g. s1's start1/start2 sharing
        # its 3-employee pool) got merged into one decision point, leaving no
        # "other" decision to be CONTESTED against -- so every reward-type
        # came out PURE and LS-HCA had nothing left to correct (found by the
        # s1 training smoke; see FORKFREE_LINEAGE_RETHINK.md). A place is a
        # resource pool iff following its own consumption forward ever leads
        # back to itself (a produce-consume CYCLE) -- structural, no
        # token-value semantics, no multiplicity assumption (R in M2/M4 has
        # only 1 unit and is still correctly a pool by this test).
        def _is_cyclic_place(pid, seed_transitions):
            seen_p, seen_t, stack = set(), set(), list(seed_transitions)
            while stack:
                t = stack.pop()
                if t._id in seen_t:
                    continue
                seen_t.add(t._id)
                for p in t.outgoing:
                    if p._id == pid:
                        return True
                    if p._id in seen_p:
                        continue
                    seen_p.add(p._id)
                    stack.extend(consumers.get(p._id, []))
            return False

        ap = {a._id: a._id for a in pn.actions}
        def af(x):
            while ap[x] != x:
                ap[x] = ap[ap[x]]; x = ap[x]
            return x
        place_to_actions = {}
        for a in pn.actions:
            for p in a.incoming:
                place_to_actions.setdefault(p._id, []).append(a._id)
        for pid, aids in place_to_actions.items():
            if len(aids) < 2 or _is_cyclic_place(pid, consumers.get(pid, [])):
                continue    # <2 competitors, or a resource pool -> no merge
            for k in range(1, len(aids)):
                ap[af(aids[0])] = af(aids[k])
        dp_reach = {}
        for aid, rr in reach_by_action.items():
            dp_reach.setdefault(af(aid), set()).update(rr)

        # PURE(d) = d's reach minus every OTHER decision point's reach;
        # CONTESTED(d) = the intersection -- no merging into components (that
        # is s_ccf's conservative move; LS-HCA needs the split per decision).
        pure_by_dp, contested_by_dp = {}, {}
        for dp, rr in dp_reach.items():
            other = set()
            for dp2, rr2 in dp_reach.items():
                if dp2 != dp:
                    other |= rr2
            pure_by_dp[dp] = rr - other
            contested_by_dp[dp] = rr & other

        self._ls_hca_classify_cache = {
            aid: (pure_by_dp[af(aid)], contested_by_dp[af(aid)])
            for aid in reach_by_action
        }
        return self._ls_hca_classify_cache

    def _redistribute_ls_hca(self, action_transitions, record_to_action,
                             redistribution, beta, token_to_action, get_parents):
        """LS-HCA (FORKFREE_LINEAGE_RETHINK.md Idea 1): the fork-free dual of
        the DAG-replay counterfactual ('cf'). Per decision ``d``:

            Q_d = Sum_{r in PURE(d)}      r * disc            (exact, here)
                + Sum_{r in CONTESTED(d)} r * disc * (1 - pi(a|s)/hhat(a|s,r))

        PURE and EXOGENOUS are read off the static DAG (``_pure_contested_
        reward_types``, no fit). This method computes the PURE term exactly
        and returns it as ``redistribution``; the CONTESTED term needs
        ``pi(a|s)``, the acting policy's probability of the taken action,
        which this trace object never observes (it only sees the reward
        trace). So each CONTESTED reward instance's realized-lineage-
        membership ``z`` and raw (undiscounted-by-pi) contribution is stashed
        on ``self._ls_hca_pending`` -- a list aligned 1:1 with
        ``action_transitions``, each entry a list of
        ``(action_type, reward_type, z, reward*disc)`` tuples -- for the
        caller (``gympn.agents.Agent.run_episode``) to combine with the
        per-step ``pi(a|s)`` it already has and the currently-fit ``hhat``
        table (``Agent._ls_hca_hhat``, refit each epoch from the pooled
        records -- see ``Agent._fit_ls_hca_hhat``).

        Safe cold start: ``hhat`` starts empty, so every CONTESTED term's
        factor is 0 until it has data -- PURE-only credit, the same
        conservative floor as ``s_ccf``, never biased."""
        import math
        classify = self._pure_contested_reward_types()
        base = self._pn._get_string_before_last_dot

        def lineage_decisions(input_ids, firing_idx):
            """{decision_idx: hop depth} for one reward's lineage.

            Breadth-first (not the previous depth-first pop) so the first time
            a decision is reached is via a SHORTEST token path -- depth is only
            meaningful as a minimum. Membership is unchanged: the key set is
            exactly what the old version returned as a set.

            Depth is one of the three per-(decision, reward) statistics
            LS-HCA's conditioning variable can be built from. A binary
            membership indicator is constant across all k decisions in a
            reward's lineage -- constant on precisely the set the hindsight
            reweighting needs to rank -- so no estimator built on it can
            discriminate among them (measured: k averages 8.5 on s1, never 1).
            Depth varies within the lineage; so does delay; `share` (1/k) does
            not, and is kept only for comparison. See Agent.ls_hca_z_feature.
            """
            found = {}
            if firing_idx is not None:
                found[firing_idx] = 0
            seen = set()
            frontier = [(tid, 1) for tid in input_ids]
            while frontier:
                nxt = []
                for tid, d in frontier:
                    if tid in seen:
                        continue
                    seen.add(tid)
                    hit = token_to_action.get(tid)
                    if hit is not None and hit[0] not in found:
                        found[hit[0]] = d
                    for p in get_parents(tid):
                        if p not in seen:
                            nxt.append((p, d + 1))
                frontier = nxt
            return found

        rewards = []
        for tr in self.transition_history.transitions:
            rv = tr.get('reward', 0.0)
            if rv == 0.0:
                continue
            rtype = base(getattr(tr.get('transition'), '_id', ''))
            firing_idx = None
            se = record_to_action.get(id(tr))
            if se is not None:
                firing_idx = se[0]
            rewards.append((rv, tr.get('time'), rtype, tr.get('input_tokens', []), firing_idx))

        def disc(t_reward, u_dec):
            if beta == 0.0 or t_reward is None or u_dec is None:
                return 1.0
            return math.exp(-beta * max(0.0, float(t_reward) - float(u_dec)))

        # A reward's lineage set depends only on the REWARD, never on which
        # decision is being scored, so hoist it out of the inner loop: it was
        # being recomputed once per (decision, reward) pair, i.e. |decisions|
        # times more often than necessary. Exactly equivalent, just not
        # repeated.
        #
        # The sizes are also the quantity that says whether the hindsight
        # reweighting can matter at all. lrq2 hands a reward's full mass to
        # every one of the k decisions in its lineage, undiscriminated; the
        # reweighting's whole job is to discriminate among those k. At k == 1
        # there is nothing to discriminate and the correction is structurally
        # moot, whatever I(A;Z|X) says. Recorded per reward instance for
        # `_diag_ls_hca_sharing.py`.
        lineage_sets = [lineage_decisions(in_ids, firing_idx)
                        for (_rv, _t_j, _rt, in_ids, firing_idx) in rewards]
        self._ls_hca_lineage_sizes = [
            (rt, len(lineage_sets[j]))
            for j, (_rv, _t_j, rt, _in, _fi) in enumerate(rewards)
        ]

        pending = [[] for _ in action_transitions]
        for idx, act in enumerate(action_transitions):
            rec = record_to_action.get(id(act))
            if rec is not None and rec[1]:      # postpone sentinel -> no credit
                continue
            tobj = act.get('transition')
            a_type = base(getattr(tobj, '_id', '')) if tobj is not None else None
            pure, contested = classify.get(a_type, (set(), set()))
            u = act.get('time')
            c = 0.0
            for j, (rv, t_j, rtype, in_ids, firing_idx) in enumerate(rewards):
                if t_j is None or u is None or t_j < u:
                    continue
                if rtype in pure:
                    # PURE means no OTHER action-type can structurally reach
                    # this reward-type, not that every INSTANCE of this
                    # decision's type caused every instance of it -- an
                    # episode can fire this decision's type many times (many
                    # cases), so crediting on type-reachability alone would
                    # give every instance the full return-to-go (mc_q, not a
                    # lineage credit). Still gate on REALIZED lineage
                    # membership, exactly like lrq -- pure only tells us that
                    # membership, once realized, needs no hindsight
                    # correction (deterministic: nothing else could have
                    # caused it).
                    if idx in lineage_sets[j]:
                        c += rv * disc(t_j, u)
                elif rtype in contested:
                    lin = lineage_sets[j]
                    z = idx in lin
                    # All three candidate conditioning statistics, computed
                    # here and chosen by the agent (Agent.ls_hca_z_feature) --
                    # the trace needs no configuration of its own. Only
                    # defined when the decision IS in the lineage; the
                    # not-in-lineage case is its own conditioning level.
                    #   delay: reward time - decision time, how long after this
                    #          decision the reward landed. Varies within a
                    #          lineage.
                    #   depth: hop distance through the token DAG. Varies
                    #          within a lineage.
                    #   share: 1/k. Does NOT vary within a lineage (it is a
                    #          property of the reward, not of the decision), so
                    #          it cannot rank lineage members -- kept for
                    #          comparison, not as a serious candidate.
                    if z:
                        delay = (float(t_j) - float(u)) if (t_j is not None and u is not None) else None
                        depth = float(lin.get(idx, 0))
                        share = 1.0 / max(len(lin), 1)
                    else:
                        delay = depth = share = None
                    pending[idx].append((a_type, rtype, z, rv * disc(t_j, u),
                                         delay, depth, share))
            redistribution[idx] = c

        self._ls_hca_pending = pending
        return redistribution

    # ------------------------------------------------------------------ #
    # LRQ: Lineage-Restricted Q (per-decision hindsight Q-sample)        #
    # ------------------------------------------------------------------ #

    def _redistribute_lrq(self, action_transitions, token_to_action,
                          record_to_action, redistribution, beta, get_parents,
                          include_postpone, exclude_postpone_credit=False):
        """
        LRQ — the hindsight lineage return (see CAUSAL_LRQ_PROPOSAL.md).

        For each decision t:

            Q_t = Σ_{j : t ∈ L_j}  r_j · e**(-beta * (t_j - u_t))

        i.e. the FULL discounted value of every future reward in whose causal
        lineage decision t sits. Compared to the removed "rec" scheme
        (see CAUSAL_REDISTRIBUTION_SOUND_SCHEME.md for its design):

        - no 1/|P| split: a reward with k lineage decisions contributes its
          full (discounted) mass to all k of them. The output is a per-decision
          Monte-Carlo Q-sample (policy-gradient-theorem object), NOT a
          mass-conserving redistribution — it must never be summed as a return,
          and must be consumed as A_t = Q_t - V_L(s_t) with V_L regressed on
          the same quantity (data.py, causal_scheme == "lrq"). This is what
          fixes REC's return-to-go bias (CAUSAL_REC_CRITICAL_REVIEW.md §3): a
          late chain decision sees the chain's full reward, not a diluted
          share, so completing a k-stage case is never undervalued against a
          shorter exclusive alternative.
        - postpone decisions ARE lineage members (no null-player exclusion):
          under token-flow postpone they sit in the lineage of the case they
          delayed, so waiting sees the same rewards discounted from an earlier
          clock (timing penalty) and a *beneficial* wait sees the larger mass
          it enabled (direct benefit channel). Without token-flow postpone the
          sentinel has no output tokens, Q_postpone would be identically 0 and
          A = -V_L(s) would suppress waiting unconditionally — so LRQ refuses
          to run when postpone actions are present without token-flow.
        - no exogenous fallback: a reward with no decision in its lineage is
          uncontrollable and enters no Q_t (omitting an action-independent
          term from a PG estimator is exact; smearing it would only add
          variance).
        """
        import math

        postpone_idx = {idx for (idx, is_pp) in record_to_action.values() if is_pp}
        if postpone_idx and not include_postpone and not exclude_postpone_credit:
            # v1 only: v2 (exclude_postpone_credit) gives postpone its own
            # SMDP-TD advantage downstream, so a zero lineage credit is by
            # design, not a bug.
            raise ValueError(
                "[LRQ] postpone actions are present but token-flow postpone is "
                "disabled (causal_postpone_tokenflow=False). Under sink-less "
                "postpone Q_postpone == 0 identically and A = -V(s) would "
                "suppress waiting unconditionally. Enable "
                "causal_postpone_tokenflow=True on the simulator (or use "
                "scheme='lrq2')."
            )

        decision_time = {idx: act.get("time")
                         for idx, act in enumerate(action_transitions)}

        def lineage_decisions(input_ids, firing_idx):
            """Decisions in a reward's causal lineage. v1 includes postpone;
            v2 (exclude_postpone_credit) filters postpone out of the CREDITED
            set while still traversing through its re-emitted tokens so the
            upstream producers are found."""
            found = set()
            if firing_idx is not None and not (exclude_postpone_credit
                                               and firing_idx in postpone_idx):
                found.add(firing_idx)
            seen = set()
            stack = list(input_ids)
            while stack:
                tid = stack.pop()
                if tid in seen:
                    continue
                seen.add(tid)
                hit = token_to_action.get(tid)
                if hit is not None and not (exclude_postpone_credit
                                            and hit[0] in postpone_idx):
                    found.add(hit[0])
                for p in get_parents(tid):
                    if p not in seen:
                        stack.append(p)
            return found

        def disc(t_reward, u_dec):
            # e**(-beta (t_j - u_t)); clamp at 0 and treat missing timestamps
            # as no discount, same conservative choices as "rec".
            if beta == 0.0 or t_reward is None or u_dec is None:
                return 1.0
            return math.exp(-beta * max(0.0, float(t_reward) - float(u_dec)))

        for tr in self.transition_history.transitions:
            reward = tr.get("reward", 0.0)
            if reward == 0.0:
                continue
            t_j = tr.get("time")

            # A rewarding transition that is itself an action (postpone or not)
            # is trivially in its own lineage.
            firing_idx = None
            self_entry = record_to_action.get(id(tr))
            if self_entry is not None:
                firing_idx = self_entry[0]

            for idx in lineage_decisions(tr.get("input_tokens", []), firing_idx):
                redistribution[idx] += reward * disc(t_j, decision_time.get(idx))

        return redistribution


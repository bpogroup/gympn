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

    def redistribute_rewards(self, gamma=0.9, scheme="flow_dag", include_postpone: bool = None,
                             self_credit: float = 1.0, beta: float = 0.0):
        """
        Redistribute rewards to action transitions using causal traces.

        Implemented schemes:
          - "flow_dag" (default, recommended):
              conserved, single-pass reverse-topological flow on the token DAG.
              Action-output tokens are absorbing sinks that own a unit of
              responsibility for their producing action; every other token's
              responsibility is the gamma-decayed, parent-weighted combination of
              its parents'. Root/untracked tokens own nothing, so the
              *uncontrollable* fraction of a reward is left unassigned rather than
              force-distributed. Mass-conserving (no blow-up), exact in one
              O(V + E) pass, deterministic. See ``_redistribute_flow_dag``.
          - "shapley_dag" (Route B, marginal):
              per-reward Shapley value over the *decisions in the reward's
              lineage*, the fair non-linear counterpart of "flow_dag". The
              coalition value is the realized-trace characteristic function
              ``v(S) = reward * gamma**(total wait of postpones in S)`` when all
              non-postpone production decisions are in ``S`` (else 0). This makes
              every necessary production decision an equal "carrier" of the reward
              (series/AND => reward/k) while every **postpone** decision becomes a
              *timing modifier* whose marginal is NEGATIVE (including it delays the
              reward, shrinking ``gamma**delay``) — so a postpone that merely
              wastes time receives negative credit and a no-wait postpone receives
              0 (null player). Requires token-flow postpone
              (``causal_postpone_tokenflow=True``) for postpone to appear in the
              DAG at all. Exact for small lineages, Monte-Carlo for large ones.
              ``self_credit`` is unused (the split is fixed by the Shapley axioms).
              See ``_redistribute_shapley_dag`` and
              CAUSAL_REDISTRIBUTION_SMDP_THEORY.md (Route B).
          - "rec" (Return-Equivalent Causal, provably S0+S1+S2 sound):
              the only scheme that is mass-conserving *and* return-equivalent.
              Per reward it splits the FULL reward over the gating (production)
              decisions in its lineage as a partition of unity (existence-game
              Shapley => reward/|P| each; postpones are null players), and places
              each share with the intra-sojourn discount ``e**(-beta*(t_j-u_i))``
              (t_j = reward time, u_i = the decision's clock time). Composed with
              the wrapper's inter-decision discount (smdp_gae with the SAME beta)
              the discounted credit-return equals the true discounted SMDP return
              R_beta EXACTLY for every trajectory, so the credit-process shares the
              true optimal policy (RUDDER return-equivalence) and the value-
              baselined gradient is unbiased. The opportunity cost of time lives in
              the discount, not in dropped mass, so postpone is penalised WITHOUT
              negative credit and WITHOUT leaking mass. ``beta`` must equal the
              SMDP ``causal_beta`` used downstream; ``gamma``/``self_credit`` are
              unused. See ``_redistribute_rec`` and
              CAUSAL_REDISTRIBUTION_SOUND_SCHEME.md.
          - "flow" (legacy):
              reverse BFS per reward transition with per-action renormalization.
              Kept for A/B comparison only; see CAUSAL_RL_REDISTRIBUTION_PROBLEMS.md
              for why it is biased (renormalization discards causal strength,
              uniform fallback smears unattributable reward, etc.).
          - "exponential", "linear", "uniform", "depth", "hybrid" (legacy):
              backward-compatible heuristics based on (delay, depth).

        :param gamma: per-hop decay factor. ``gamma=1.0`` gives pure structural
                      attribution (no decay) under "flow_dag"/"flow".
        :param scheme: which redistribution algorithm to use.
        :param include_postpone: if False postpone sentinel transitions
                      are never treated as credit sinks, so they receive zero
                      redistributed reward while still occupying their step slot
                      (preserving 1:1 credit/step alignment). If True, postpone's
                      output tokens are treated as sinks (token-flow postpone,
                      Design C). When left as ``None`` (default) it resolves to
                      ``self.postpone_tokenflow`` (set by the simulator), so the
                      simulator's ``causal_postpone_tokenflow`` flag is the single
                      source of truth. Note: under sink-less postpone the sentinel
                      has no output tokens, so this flag is a no-op there.
        :param self_credit: ("flow_dag" only) fraction in [0, 1] of a reward
                      carried by an *action* transition that is credited directly
                      to that acting transition (its own proximal decision); the
                      remaining (1 - self_credit) flows to the enabling ancestors
                      as usual. ``1.0`` (default) = the action that fired the
                      rewarding transition owns its reward; ``0.0`` = legacy
                      ancestor-only behaviour. Rewards on *evolution* transitions
                      are unaffected (always flow fully to ancestors).
        :param beta: ("rec" only) SMDP discount rate used for the intra-sojourn
                      reward discount ``e**(-beta*(t_j-u_i))``. MUST match the
                      ``causal_beta`` used by the downstream SMDP advantage
                      (``smdp_gae``) for the return-equivalence proof to hold; the
                      two discounts compose into the single reward-to-start
                      discount. ``0.0`` => undiscounted return-equivalence (still
                      mass-conserving, but time carries no opportunity cost).
        :return: List of redistributed rewards aligned with action transitions.
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
        # Used by flow_dag's self_credit to attribute an action's own reward to
        # itself. Keyed by id() because the same SimAction object may fire many
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

        # New default: conserved single-pass reverse-topological flow.
        # Computes attribution once for the whole episode (O(V + E)) instead of
        # re-running a BFS per reward transition.
        if scheme == "flow_dag":
            return self._redistribute_flow_dag(
                token_to_action, record_to_action, redistribution, gamma,
                get_parents, self_credit
            )

        # Route B: per-reward Shapley over the lineage decisions (marginal, with
        # postpone as a negative timing player). See _redistribute_shapley_dag.
        if scheme == "shapley_dag":
            return self._redistribute_shapley_dag(
                token_to_action, record_to_action, redistribution, gamma,
                get_parents
            )

        # REC: return-equivalent, mass-conserving, timing-aware via the discount.
        # The single scheme that is provably S0+S1+S2 sound. See _redistribute_rec.
        if scheme == "rec":
            return self._redistribute_rec(
                action_transitions, token_to_action, record_to_action,
                redistribution, beta, get_parents
            )

        for tr in self.transition_history.transitions:
            reward = tr.get("reward", 0.0)
            if reward == 0.0:
                continue

            transition_time = tr.get("time")
            input_token_ids = tr.get("input_tokens", [])
            if not input_token_ids:
                continue

            if scheme == "flow":
                # Reverse-flow propagation (delta-driven to avoid infinite loops on cycles)
                from collections import deque, defaultdict

                # influence at tokens (accumulated) and pending delta to propagate
                token_influence = defaultdict(float)
                pending = defaultdict(float)
                q = deque()

                # initialize: each consumed input token has unit influence and pending delta 1
                for tid in input_token_ids:
                    token_influence[tid] += 1.0
                    pending[tid] += 1.0
                    q.append(tid)

                # Propagate deltas to parents while deltas are significant.
                # FIX: Removed max_depth limit that was artificially truncating
                # credit flow. The eps threshold and max_iters/max_processed
                # guards are sufficient to ensure termination.
                # FIX: Removed visited set — use delta-driven re-queuing so that
                # accumulated deltas from multiple paths are fully propagated.
                # A token is re-queued whenever it accumulates significant
                # pending delta, regardless of whether it was processed before.
                eps = 1e-6
                max_iters = 10000
                iters = 0
                max_processed = 50000
                processed = 0
                in_queue = set(input_token_ids)  # track what's currently in the queue

                while q and iters < max_iters and processed < max_processed:
                    iters += 1
                    cur = q.popleft()
                    in_queue.discard(cur)
                    delta = pending[cur]
                    pending[cur] = 0.0
                    if delta <= eps:
                        continue
                    parents = get_parents(cur)
                    if not parents:
                        continue
                    share = (delta * gamma) / len(parents)
                    if share <= eps:
                        continue
                    for p in parents:
                        token_influence[p] += share
                        pending[p] += share
                        # Re-queue if significant pending delta and not already queued
                        if pending[p] > eps and p not in in_queue:
                            q.append(p)
                            in_queue.add(p)
                    processed += 1

                if iters >= max_iters or processed >= max_processed:
                    import warnings
                    warnings.warn("Causal trace flow propagation stopped early (max iterations/processed). Results may be approximate.")

                # Collect action weights: sum influence on tokens that are outputs of actions.
                # FIX: Removed double discounting — the BFS already applies gamma
                # per hop, so applying time-based gamma again double-penalizes
                # temporally distant actions.
                action_weights = defaultdict(float)
                for tid, infl in token_influence.items():
                    if tid in token_to_action:
                        action_idx, action_time = token_to_action[tid]
                        action_weights[action_idx] += infl

                total_w = sum(action_weights.values())
                if total_w > 0:
                    for idx, w in action_weights.items():
                        redistribution[idx] += reward * (w / total_w)
                else:
                    # Fallback: no actions reached by the BFS.
                    # Distribute the reward uniformly across ALL action
                    # transitions so that no reward is silently lost.
                    n_actions = len(action_transitions)
                    if n_actions > 0:
                        uniform_share = reward / n_actions
                        for idx in range(n_actions):
                            redistribution[idx] += uniform_share

            else:
                # backward-compatible heuristics using (delay, depth)
                action_info_map = {}
                for token_id in input_token_ids:
                    # direct
                    if token_id in token_to_action:
                        action_idx, action_time = token_to_action[token_id]
                        if action_time is None or transition_time is None:
                            continue
                        delay = max(0, transition_time - action_time)
                        current_depth = 1
                        prev_delay, prev_depth = action_info_map.get(action_idx, (delay, current_depth))
                        action_info_map[action_idx] = (min(prev_delay, delay), min(prev_depth, current_depth))

                    # ancestors via causal chain
                    causal_chain = self.token_history.get_causal_chain(token_id)
                    if causal_chain:
                        for depth, token_ids in enumerate(causal_chain[1:], start=2):
                            for tid in token_ids:
                                if tid in token_to_action:
                                    action_idx, action_time = token_to_action[tid]
                                    if action_time is None or transition_time is None:
                                        continue
                                    delay = max(0, transition_time - action_time)
                                    prev_delay, prev_depth = action_info_map.get(action_idx, (delay, depth))
                                    action_info_map[action_idx] = (min(prev_delay, delay), min(prev_depth, depth))

                if action_info_map:
                    weights = []
                    key_list = []
                    for k, (delay, depth) in action_info_map.items():
                        key_list.append(k)
                        if scheme == "exponential":
                            w = gamma ** delay
                        elif scheme == "linear":
                            w = 1 / (1 + delay)
                        elif scheme == "uniform":
                            w = 1.0
                        elif scheme == "depth":
                            w = 1 / depth
                        elif scheme == "hybrid":
                            w = (gamma ** delay) / depth
                        else:
                            raise ValueError(f"Unknown scheme: {scheme}")
                        weights.append(w)

                    total = sum(weights)
                    if total > 0:
                        for idx, w in zip(key_list, weights):
                            redistribution[idx] += reward * (w / total)

        return redistribution

    def _redistribute_flow_dag(self, token_to_action, record_to_action, redistribution,
                               gamma, get_parents, self_credit=1.0):
        """
        Conserved, single-pass reverse-topological credit flow on the token DAG.

        For every token we compute an *attribution* vector ``attr[t]`` over action
        indices: of the unit of causal responsibility carried by token ``t``, what
        fraction is owed to each action?

            attr[t] = onehot(X)                          if t is output of action X
            attr[t] = {}                                 if t is a root/untracked token
            attr[t] = sum_p (gamma / |parents|) * attr[p]  otherwise

        Action-output tokens own responsibility for their producing action and,
        when ``self_credit < 1``, pass a ``(1 - self_credit)`` share back to the
        actions that enabled them (so credit flows through a chain of actions
        rather than being fully absorbed by the last one). Every reward is then
        distributed by averaging the attribution of the tokens it consumed:

            credit[a] += reward * mean_{t in inputs} attr[t][a]

        Properties (vs. the legacy "flow" scheme):
          - Mass-conserving: ||attr[t]||_1 <= 1 for every token, so influence
            cannot blow up and there is no per-action renormalization. The
            uncontrollable fraction ``reward * (1 - ||a||_1)`` is correctly left
            unassigned instead of being force-fed to weakly-linked actions.
          - Exact in a single O(V + E) pass (topological order), so there are no
            iteration/processed caps that silently make the result approximate.
          - Deterministic: independent of traversal/queue order.

        Note: gamma is applied per lineage hop (``gamma=1.0`` => pure structural
        attribution). Postpone sentinels are excluded from ``token_to_action`` by
        the caller, so their tokens act as pass-through nodes (credit flows
        *through* them to the real upstream actions) and they receive zero credit.

        ``self_credit`` (in [0, 1]) governs how much of a reward carried by a
        *non-postpone action* transition is credited directly to that acting
        decision (``record_to_action`` maps a reward-transition record to its
        action index); the remaining ``(1 - self_credit)`` flows to enabling
        ancestors. Rewards on evolution transitions are unaffected.
        """
        from collections import defaultdict

        tokens = self.token_history.tokens

        # --- 1. Topological order: every token appears after all its parents. ---
        # Parents are produced no later than their children, so the parent graph
        # is acyclic. We use an explicit-stack postorder to avoid Python recursion
        # depth limits on long lineages.
        order = []
        visited = set()
        for start in tokens:
            if start in visited:
                continue
            stack = [(start, False)]
            while stack:
                node, processed = stack.pop()
                if processed:
                    order.append(node)
                    continue
                if node in visited:
                    continue
                visited.add(node)
                stack.append((node, True))
                for p in get_parents(node):
                    if p not in visited:
                        stack.append((p, False))

        # --- 2. Propagate attribution from sinks down the DAG (parents first). ---
        #
        # An action-output token ("sink") owns responsibility for its producing
        # action. With self_credit == 1.0 it is fully absorbing (legacy
        # behaviour). With self_credit < 1.0 it ALSO lets a (1 - self_credit)
        # share flow back to the actions that ENABLED it — critical for
        # multi-stage / sequential processes where a reward is collected on the
        # last action's own output token (e.g. busy2), which would otherwise
        # give every earlier-stage action exactly zero credit.
        #
        # The proximal action keeps self_credit PLUS whatever its ancestor
        # actions do not account for, so an action output stays fully
        # controllable (||attr|| == 1): nothing leaks to roots at a sink, and
        # single-stage / parallel attribution is unchanged at any self_credit.
        attr = {}  # token_id -> {action_idx: responsibility}
        for tid in order:
            parents = get_parents(tid)
            share = gamma / len(parents) if parents else 0.0
            acc = defaultdict(float)
            for p in parents:
                p_attr = attr.get(p)
                if not p_attr:
                    continue
                for a_idx, val in p_attr.items():
                    acc[a_idx] += share * val

            sink = token_to_action.get(tid)
            if sink is not None:
                act_idx = sink[0]
                blended = {a: (1.0 - self_credit) * v for a, v in acc.items()}
                ancestor_mass = sum(blended.values())
                blended[act_idx] = blended.get(act_idx, 0.0) + (1.0 - ancestor_mass)
                attr[tid] = blended
            elif not parents:
                # Root / untracked: uncontrollable, owes credit to no action.
                attr[tid] = {}
            else:
                attr[tid] = dict(acc)

        # --- 3. Distribute each reward over the actions its inputs are owed to. ---
        for tr in self.transition_history.transitions:
            reward = tr.get("reward", 0.0)
            if reward == 0.0:
                continue

            # self_credit: if the reward is carried by a (non-postpone) action
            # transition, credit a fraction directly to that acting decision and
            # let only the remainder flow to enabling ancestors.
            remainder = reward
            if self_credit > 0.0:
                self_entry = record_to_action.get(id(tr))
                if self_entry is not None:
                    self_idx, self_is_postpone = self_entry
                    if not self_is_postpone:
                        redistribution[self_idx] += self_credit * reward
                        remainder = (1.0 - self_credit) * reward

            if remainder == 0.0:
                continue
            input_ids = tr.get("input_tokens", [])
            if not input_ids:
                continue
            inv = remainder / len(input_ids)  # average responsibility over consumed tokens
            for tid in input_ids:
                for a_idx, val in attr.get(tid, {}).items():
                    redistribution[a_idx] += inv * val

        return redistribution

    # ------------------------------------------------------------------ #
    # REC: Return-Equivalent Causal redistribution (S0 + S1 + S2 sound)  #
    # ------------------------------------------------------------------ #

    def _redistribute_rec(self, action_transitions, token_to_action,
                          record_to_action, redistribution, beta, get_parents):
        """
        REC — the provably sound (S0 + S1 + S2) redistribution.

        Idea (see CAUSAL_REDISTRIBUTION_SOUND_SCHEME.md): keep mass and timing in
        *separate* objects. Mass is conserved in the credit (a partition of unity
        per reward); the opportunity cost of time lives in the objective's
        discount. For each reward r_j firing at clock time ``t_j`` with gating
        (production) decisions ``P`` in its lineage:

            c_i += (1/|P|) * r_j * e**(-beta * (t_j - u_i))     for each i in P

        where ``u_i`` is decision i's clock time. ``1/|P|`` is the Shapley value of
        the *existence game* ``v(S) = r_j * [P subset of S]`` (necessary decisions
        split equally; postpones gate nothing => null players => weight 0). The
        weights sum to 1, so summed with the wrapper's inter-decision discount
        ``e**(-beta*(u_i-u_0))`` (smdp_gae, SAME beta) the discounted credit-return
        telescopes to the true discounted return ``R_beta = Σ_j e**(-beta*(t_j-u_0))
        r_j`` EXACTLY, per trajectory (the ``u_i`` cancels). Return-equivalence =>
        same optimal policy (RUDDER) and an unbiased value-baselined gradient.

        Postpone is penalised purely through the discount: waiting pushes ``t_j``
        later => smaller ``R_beta`` on that trajectory => "act now" is preferred,
        with no negative credit and no leaked mass.

        Exogenous rewards (no gating decision in the lineage) are spread uniformly
        over the causally-prior decisions so mass is still conserved; soundness is
        independent of this choice (any partition of unity preserves the optimum).
        """
        import math

        # Postpone action indices: null players of the existence game.
        postpone_idx = {idx for (idx, is_pp) in record_to_action.values() if is_pp}

        # Decision (action) clock times u_idx.
        decision_time = {idx: act.get("time")
                         for idx, act in enumerate(action_transitions)}

        def production_set(input_ids, firing_idx):
            """Gating (non-postpone) decisions in a reward's causal lineage."""
            prod = set()
            if firing_idx is not None:
                prod.add(firing_idx)
            seen = set()
            stack = list(input_ids)
            while stack:
                tid = stack.pop()
                if tid in seen:
                    continue
                seen.add(tid)
                hit = token_to_action.get(tid)
                if hit is not None and hit[0] not in postpone_idx:
                    prod.add(hit[0])
                # Traverse through every token (incl. action outputs) so earlier
                # enabling decisions are collected too.
                for p in get_parents(tid):
                    if p not in seen:
                        stack.append(p)
            return prod

        def disc(t_reward, u_dec):
            # e**(-beta (t_j - u_i)); clamp the delay at 0 so a (non-causal)
            # u_i > t_j cannot inflate a share above the reward, and treat a
            # missing timestamp as no discount (factor 1) to stay conservative.
            if beta == 0.0 or t_reward is None or u_dec is None:
                return 1.0
            return math.exp(-beta * max(0.0, float(t_reward) - float(u_dec)))

        for tr in self.transition_history.transitions:
            reward = tr.get("reward", 0.0)
            if reward == 0.0:
                continue
            t_j = tr.get("time")
            input_ids = tr.get("input_tokens", [])

            # If the rewarding transition is itself a non-postpone action, it is a
            # mandatory gating decision (the reward exists because it fired).
            firing_idx = None
            self_entry = record_to_action.get(id(tr))
            if self_entry is not None and not self_entry[1]:
                firing_idx = self_entry[0]

            P = production_set(input_ids, firing_idx)

            if not P:
                # Exogenous reward: spread over causally-prior decisions so the
                # mass is conserved (return-equivalence) instead of leaked.
                prior = [idx for idx, u in decision_time.items()
                         if u is None or t_j is None or u <= t_j]
                if not prior:
                    continue
                w = 1.0 / len(prior)
                for idx in prior:
                    redistribution[idx] += w * reward * disc(t_j, decision_time.get(idx))
                continue

            w = 1.0 / len(P)  # existence-game Shapley: equal split over necessary decisions
            for idx in P:
                redistribution[idx] += w * reward * disc(t_j, decision_time.get(idx))

        return redistribution

    # ------------------------------------------------------------------ #
    # Route B: Shapley return-decomposition on the token DAG             #
    # ------------------------------------------------------------------ #

    # Exact Shapley (enumerate 2^n coalitions) is used up to this lineage
    # size; beyond it we fall back to Monte-Carlo sampling. Per-reward
    # lineages are tiny in practice (a case touches ~2-3 decisions), so the
    # exact path is the common one and the cap only guards pathological cases.
    _SHAPLEY_EXACT_MAX = 10
    _SHAPLEY_MC_SAMPLES = 256

    @staticmethod
    def _shapley_values(players, v_func):
        """
        Exact (or Monte-Carlo) Shapley values of a coalition game.

        :param players: list of hashable player ids.
        :param v_func: callable ``frozenset(players_subset) -> float`` (the
                       characteristic function; ``v(empty)`` need not be 0).
        :return: dict ``player -> shapley_value``. The values sum to
                 ``v(all) - v(empty)`` (efficiency).
        """
        import math
        import random as _random

        players = list(players)
        n = len(players)
        if n == 0:
            return {}
        if n == 1:
            return {players[0]: v_func(frozenset(players)) - v_func(frozenset())}

        phi = {p: 0.0 for p in players}

        if n <= CausalTraces._SHAPLEY_EXACT_MAX:
            # Cache v over every coalition (bitmask), then accumulate the
            # standard weighted marginal contributions.
            vcache = [0.0] * (1 << n)
            for mask in range(1 << n):
                subset = frozenset(players[j] for j in range(n) if mask & (1 << j))
                vcache[mask] = v_func(subset)
            fact = [math.factorial(k) for k in range(n + 1)]
            inv_nfact = 1.0 / fact[n]
            for i in range(n):
                bit = 1 << i
                for mask in range(1 << n):
                    if mask & bit:
                        continue
                    s = bin(mask).count("1")
                    weight = fact[s] * fact[n - s - 1] * inv_nfact
                    phi[players[i]] += weight * (vcache[mask | bit] - vcache[mask])
            return phi

        # Monte-Carlo: average marginal contributions over random orderings.
        m = CausalTraces._SHAPLEY_MC_SAMPLES
        for _ in range(m):
            perm = players[:]
            _random.shuffle(perm)
            cur = set()
            prev_v = v_func(frozenset())
            for p in perm:
                cur.add(p)
                cur_v = v_func(frozenset(cur))
                phi[p] += (cur_v - prev_v)
                prev_v = cur_v
        for p in phi:
            phi[p] /= m
        return phi

    def _redistribute_shapley_dag(self, token_to_action, record_to_action,
                                  redistribution, gamma, get_parents):
        """
        Per-reward Shapley decomposition of the (timing-discounted) return over
        the decisions in each reward's causal lineage. See the ``"shapley_dag"``
        entry in ``redistribute_rewards`` for the model; in short:

          players(reward) = production decisions (non-postpone actions in the
                            lineage, plus the firing action if the rewarding
                            transition is itself a non-postpone action)
                          ∪ postpone decisions in the lineage
          v(S) = reward * gamma**(Σ delay of postpones in S)   if production ⊆ S
               = 0                                              otherwise

        Properties:
          - Efficiency: Σ_i c_i = reward * gamma**(total wait)  (the discounted
            realized value); the time cost ``reward*(1 - gamma**wait)`` is left
            uncredited (opportunity cost of waiting), like flow_dag leaves the
            uncontrollable fraction unassigned.
          - Production decisions: positive, equal share when there is no wait
            (unanimity/AND => reward/k); they carry the realized reward.
          - Postpone decisions: NEGATIVE marginal proportional to the delay they
            introduced (``gamma < 1``); a no-wait postpone gets 0 (null player).
          - Symmetry/linearity by construction (Shapley).

        ``gamma`` is reused as the *temporal* discount base (per unit clock time
        of wait). ``gamma = 1`` disables the postpone timing signal (postpone
        becomes a null player and production is split equally).
        """
        # Postpone action indices (their tokens were only added to
        # token_to_action when include_postpone/tokenflow is active).
        postpone_idx = {idx for (idx, is_pp) in record_to_action.values() if is_pp}

        # token_id -> creation time (postpone fire time for recreated tokens).
        token_time = {tid: info.get("time")
                      for tid, info in self.token_history.tokens.items()}

        # token_id -> earliest time it was consumed as an input by any transition
        # (used to measure how long a postponed token waited before the next
        # decision consumed it).
        consume_time = {}
        for tr in self.transition_history.transitions:
            t = tr.get("time")
            if t is None:
                continue
            for tid in tr.get("input_tokens", []):
                prev = consume_time.get(tid)
                if prev is None or t < prev:
                    consume_time[tid] = t

        def lineage(input_ids, firing_idx):
            """Collect (production set, {postpone_idx: delay}) for one reward."""
            production = set()
            postpone_delay = {}
            if firing_idx is not None:
                production.add(firing_idx)
            seen = set()
            stack = list(input_ids)
            while stack:
                tid = stack.pop()
                if tid in seen:
                    continue
                seen.add(tid)
                hit = token_to_action.get(tid)
                if hit is not None:
                    idx = hit[0]
                    if idx in postpone_idx:
                        t_tok = token_time.get(tid)
                        t_cons = consume_time.get(tid)
                        d = 0.0
                        if t_tok is not None and t_cons is not None:
                            d = max(0.0, t_cons - t_tok)
                        # Same postpone reached via several tokens: keep the
                        # longest wait on the path.
                        postpone_delay[idx] = max(postpone_delay.get(idx, 0.0), d)
                    else:
                        production.add(idx)
                # Always traverse through (even action-output) tokens so earlier
                # enabling decisions are collected too.
                for p in get_parents(tid):
                    if p not in seen:
                        stack.append(p)
            return production, postpone_delay

        for tr in self.transition_history.transitions:
            reward = tr.get("reward", 0.0)
            if reward == 0.0:
                continue
            input_ids = tr.get("input_tokens", [])

            # Firing action of the rewarding transition (if it is a non-postpone
            # action, it is a mandatory production player — the reward exists
            # because it fired).
            firing_idx = None
            self_entry = record_to_action.get(id(tr))
            if self_entry is not None and not self_entry[1]:
                firing_idx = self_entry[0]

            production, postpone_delay = lineage(input_ids, firing_idx)
            players = list(production | set(postpone_delay))
            if not players:
                # Uncontrollable reward (root/evolution only): leave unassigned.
                continue

            production_fs = frozenset(production)

            def v_func(S, _r=reward, _prod=production_fs, _pd=postpone_delay, _g=gamma):
                if not _prod.issubset(S):
                    return 0.0
                wait = 0.0
                for i, d in _pd.items():
                    if i in S:
                        wait += d
                if wait == 0.0:
                    return _r
                return _r * (_g ** wait)

            shap = self._shapley_values(players, v_func)
            for idx, val in shap.items():
                redistribution[idx] += val

        return redistribution

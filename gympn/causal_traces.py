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

    def redistribute_rewards(self, scheme="lrq", include_postpone: bool = None,
                             beta: float = 0.0):
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

        raise ValueError(
            f"Unknown scheme: {scheme!r}. All redistribution schemes except 'lrq' "
            f"have been removed (flow_dag/shapley_dag/rec/flow/exponential/linear/"
            f"uniform/depth/hybrid). See CAUSAL_LRQ_PROPOSAL.md."
        )

    # ------------------------------------------------------------------ #
    # LRQ: Lineage-Restricted Q (per-decision hindsight Q-sample)        #
    # ------------------------------------------------------------------ #

    def _redistribute_lrq(self, action_transitions, token_to_action,
                          record_to_action, redistribution, beta, get_parents,
                          include_postpone):
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
        if postpone_idx and not include_postpone:
            raise ValueError(
                "[LRQ] postpone actions are present but token-flow postpone is "
                "disabled (causal_postpone_tokenflow=False). Under sink-less "
                "postpone Q_postpone == 0 identically and A = -V(s) would "
                "suppress waiting unconditionally. Enable "
                "causal_postpone_tokenflow=True on the simulator."
            )

        decision_time = {idx: act.get("time")
                         for idx, act in enumerate(action_transitions)}

        def lineage_decisions(input_ids, firing_idx):
            """ALL decisions (incl. postpone) in a reward's causal lineage."""
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


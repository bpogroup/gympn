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

    def flush(self):
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

    def redistribute_rewards(self, gamma=0.9, scheme="flow", include_postpone: bool = True):
        """
        Redistribute rewards to action transitions based on causal chains.
        Supports multiple weighting schemes:
          - "exponential": weight = gamma ** delay
          - "linear": weight = 1 / (1 + delay)
          - "uniform": equal weights
          - "depth": weight = 1 / depth
          - "hybrid": weight = (gamma ** delay) / depth
        :param gamma: Discount factor for exponential/hybrid schemes.
        :param scheme: Weighting scheme ("exponential", "linear", "uniform", "hybrid").
        :return: List of redistributed rewards aligned with action transitions.
        """
        """
        Redistribute rewards to action transitions using causal traces.

        Implemented schemes:
          - "exponential", "linear", "uniform", "depth", "hybrid":
              backward-compatible behavior based on (delay, depth) heuristics.
          - "flow":
              graph-based reverse flow propagation. For each reward-generating
              transition the algorithm performs a reverse BFS from the consumed
              input tokens, propagating a unit influence back to ancestor tokens
              and accumulating influence at action-producing tokens. Influence
              decays by `gamma` at every hop (configurable) and is split
              equally among parents when there are multiple parents.

        The "flow" scheme is more general and better handles multiple causal
        paths, cycles, and branching: it treats credit assignment as a flow
        problem on the token-parent graph and normalizes contributions so the
        reward is fully distributed across actions.

        :param gamma: per-hop decay factor (used in "exponential"/"hybrid" as
                      before and as per-hop multiplier in "flow").
        :param scheme: which redistribution algorithm to use.
        :return: List of redistributed rewards aligned with action transitions.
        """

        # Keep full ordered list of action transitions (so redistribution indexes match
        # the simulator's action ordering), but do NOT assign credit to transitions
        # whose transition._id starts with 'postpone_'. We detect postpone sentinels
        # and skip mapping their output tokens to actions so they receive zero credit.
        action_transitions = self.transition_history.get_action_transitions()
        redistribution = [0.0] * len(action_transitions)

        # Map token -> (action_index, action_time)
        token_to_action = {}
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
            if is_postpone and not include_postpone:
                continue
            for tid in act.get("output_tokens", []):
                token_to_action[tid] = (idx, act.get("time"))

        # Helper: safe accessors
        def get_parents(token_id):
            t = self.token_history.get_token(token_id)
            return t.get("parents", []) if t else []

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

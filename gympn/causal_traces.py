from torch.ao.quantization.utils import activation_is_dynamically_quantized


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

    def redistribute_rewards(self, gamma=1, scheme="exponential"):
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
        action_transitions = self.transition_history.get_action_transitions()
        redistribution = [0.0] * len(action_transitions)

        # Map token -> (action_index, action_time)
        token_to_action = {}
        for idx, act in enumerate(action_transitions):
            for tid in act.get("output_tokens", []):
                token_to_action[tid] = (idx, act.get("time"))

        total_source_reward = 0.0
        total_distributed_reward = 0.0

        for tr in self.transition_history.transitions:
            reward = tr.get("reward", 0.0)
            if reward == 0.0:
                continue
            total_source_reward += reward

            transition_time = tr.get("time")
            if transition_time is None:
                continue

            output_token_ids = tr.get("output_tokens", [])
            action_info_map = {}

            for token_id in output_token_ids:
                # Direct credit
                if token_id in token_to_action:
                    action_idx, action_time = token_to_action[token_id]
                    if action_time is None:
                        continue
                    delay = max(0, transition_time - action_time)
                    # Depth = 1 for direct token
                    current_depth = 1
                    prev_delay, prev_depth = action_info_map.get(action_idx, (delay, current_depth))
                    # Keep min delay and min depth
                    action_info_map[action_idx] = (min(prev_delay, delay), min(prev_depth, current_depth))

                # Parents
                causal_chain = self.token_history.get_causal_chain(token_id)
                for depth, token_ids in enumerate(causal_chain, start=2):  # depth starts at 2 for parents (TODO: check, this does not seem right)
                    for tid in token_ids:
                        if tid in token_to_action:
                            action_idx, action_time = token_to_action[tid]
                            if action_time is None:
                                continue
                            delay = max(0, transition_time - action_time)
                            prev_delay, prev_depth = action_info_map.get(action_idx, (delay, depth))
                            action_info_map[action_idx] = (min(prev_delay, delay), min(prev_depth, depth))

            if action_info_map:
                # Compute weights
                weights = []
                for (delay, depth) in action_info_map.values():
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
                    for idx, w in zip(action_info_map.keys(), weights):
                        delta = reward * (w / total)
                        redistribution[idx] += delta
                        total_distributed_reward += delta
                else:
                    # Fallback: uniform split
                    print("Falling back to uniform split")
                    n = len(action_info_map)
                    for idx in action_info_map.keys():
                        delta = reward / n
                        redistribution[idx] += delta
                        total_distributed_reward += delta

            #print(f"Transition: {tr['transition']._id}, Reward: {reward}")
            #print(f"Action delay map: {action_info_map}")

        print("Conservation gap:", total_source_reward - total_distributed_reward)
        return redistribution

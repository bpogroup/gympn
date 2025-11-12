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
                "input_tokens": [id(t) for t in input_tokens],
                "output_tokens": [id(t) for t in output_tokens],
                "created_by": created_by,
                "reward": reward,
                "time": time
            })

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
        tid = id(token)
        parent_ids = [id(p) for p in parent_tokens]
        self.tokens[tid] = {
            "token": token,
            "parents": parent_ids,
            "event": transition._id,
            "created_by": created_by,
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
        Recursively backtrack from a token to all contributing (action, transition) pairs.
        """
        chain = []
        visited = set()

        def recurse(tid):
            if tid in visited:
                return
            visited.add(tid)
            token_info = self.tokens.get(tid)
            if token_info:
                #get the tokens that were used to fire the transition that created this token
                parents = token_info.get("parents", [])
                chain.append(parents)

                for parent in token_info.get("parents", []):
                    recurse(parent)

        recurse(token_id)
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

    def redistribute_rewards(self):
        """
        Redistributes rewards to action transitions based on causal chains of tokens used to fire them.
        Non-action transitions do not receive rewards, but their tokens contribute to the causal chains and their rewards are redistributed to the actions that led to them.
        Returns a list of reward assignments: (action_index, transition_id, reward).
        -------
        """

        #rewards will be a list of float values with length equal to the number of actions taken
        reward_assignments = []

        for transition_record in self.transition_history.transitions:
            transition = transition_record["transition"]
            is_action = transition_record["is_action"]
            reward = transition_record["reward"]
            output_token_ids = transition_record["output_tokens"]
            input_token_ids = transition_record["input_tokens"]

            if reward:
                # if the reward is different from 0, we need to propagate it back to the actions that created the tokens used to fire this transition
                for token_id in input_token_ids:
                    causal_chain = self.token_history.get_causal_chain(token_id)
                    # every token in the causal chain contributes to the reward of the action that created it. we go through the chain, checking for transitions (actions or evolutions indifferently) that used the token or an ancestor to fire and translate their reward to the action that created them
                    for action_index, transition_id in causal_chain:
                        if action_index is not None:
                            reward_assignments.append((action_index, transition_id, reward))
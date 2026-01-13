from typing import Optional


from torch_geometric.data import Data


from typing import Optional, List, Dict, Any, Sequence
import copy
import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

Tensor = torch.Tensor



class GraphDataLoader(torch.utils.data.Dataset):
    """
    A custom data loader for graph data in PyTorch Geometric format.
    This class is designed to handle both heterogeneous and homogeneous graph data.
    It takes in a batch size and lists of states, actions, logprobs, advantages,
    logpis, and values, and prepares them for training in a PyTorch DataLoader.
    Parameters
    ----------
    :param batch_size: int
        The size of the batches to be created.
    :param states: list
        A list of states, where each state is a dictionary containing graph data.
    :param actions: list
        A list of actions corresponding to each state.
    :param logprobs: list
        A list of log probabilities for the actions taken.
    :param advantages: list
        A list of advantages for each action taken.
    :param logpis: list
        A list of log probabilities for all actions in the state.
    :param values: list
        A list of value estimates for each state.
    :param data_type: str, optional
        (Unused) The type of graph data, either 'hetero' for heterogeneous or 'homogeneous' for homogeneous.
        Default is 'hetero'.
    force_batch_size : bool, optional
        If True, truncates the dataset to be a multiple of batch_size. Default is True.
    """

    def __init__(self, batch_size, states, actions, logprobs, advantages, logpis, values, data_type='hetero', force_batch_size=True):
        self.states = states
        self.actions = actions
        self.logprobs = logprobs
        self.advantages = advantages
        self.values = values
        self.logpis = logpis


        if data_type == 'hetero':
            self.data_list = []
            for index in range(len(states)):

                temp_h_data = states[index]['graph']
                temp_h_data.y = actions[index]
                temp_h_data.advantage = advantages[index]
                if logprobs.numel():
                    temp_h_data.logprobs = logprobs[index]
                else:
                    temp_h_data.logprobs = None
                temp_h_data.value = values[index]

                if logpis[index] is not None:
                    temp_h_data.logpis = logpis[index].squeeze(-1)
                else:
                    temp_h_data.logpis = None

                if 'q_first' in states[index]:
                    temp_h_data.q_first = states[index]['q_first']
                if 'target_pi' in states[index]:
                    temp_h_data.target_pi = states[index]['target_pi']

                self.data_list.append(temp_h_data)
        elif data_type == 'homogeneous':
            self.data_list = [Data(x=torch.from_numpy(states[index]['graph'].nodes), edge_index=torch.from_numpy(states[index]['graph'].edge_links),
                             y=actions[index], advantage=advantages[index], logprobs=logprobs[index], value=values[index])
                        for index in range(len(states))]
        else:
            raise ValueError("data_type must be either 'hetero' or 'homogeneous'")

        #Cut the data list to be a multiple of the batch size
        if force_batch_size:
            if len(self.data_list) % batch_size != 0:
                self.data_list = self.data_list[:len(self.data_list) - (len(self.data_list) % batch_size)]

        self.loader = DataLoader(self.data_list, batch_size=batch_size, shuffle=False) #shuffle to true would disrupt the normalized advantages

    def __getitem__(self, index):
        """
        Extract a data element and convert it to a PyTorch Geometric Data object.

        :param index: the index of the element to extract
        :return: the pytorch_geometric data object
        """
        data = Data(
            x=self.states[index][0],
            edge_index=self.states[index][1],
            y=self.actions[index],
            advantage=self.advantages[index],
            logprobs=self.logprobs[index],
            value=self.values[index],
            logpis=torch.tensor(self.logpis[index]),
            target_p=torch.tensor(self.states[index]['target_pi']) if 'target_pi' in self.states[index] else None,
            q_first=torch.tensor(self.states[index]['q_first']) if 'q_first' in self.states[index] else None
        )
        return data

    def __len__(self):
        """
            Returns the length of the dataset.

            Returns
            -------
            int
                Number of samples in the dataset.
        """
        return len(self.data_list)

def discount_rewards(rewards, gam):
    """Return discounted rewards-to-go computed from inputs.

    Uses vectorized cumsum for efficiency instead of Python loop.

    Parameters
    ----------
    :param rewards : array_like
        List or 1D array of rewards from a single complete trajectory.
    :param gam : float
        Discount rate.

    Returns
    -------
    rewards : ndarray
        1D array of discounted rewards-to-go.

    """
    # Vectorized version: ~5-10x faster than loop
    # Approach: flip rewards, cumsum with discount factors, flip back
    T = rewards.shape[0]
    if T == 0:
        return rewards

    # Compute discounted returns: G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
    # Reverse cumsum with geometric weights
    flipped = torch.flip(rewards, [0])
    discounts = torch.pow(gam, torch.arange(T, dtype=rewards.dtype, device=rewards.device))
    cumsum_flipped = torch.cumsum(flipped * discounts, dim=0)
    returns = torch.flip(cumsum_flipped, [0]) / discounts

    return returns

def apply_causal_credits(causal_trace):
    """
    Apply causal credits to redistribute rewards based on causal trace information.

    Parameters
    ----------
    :param causal_trace : CausalTrace object


    Returns
    -------
    action_rewards : list
        List of redistributed rewards per action.
    """
    return torch.tensor(causal_trace.redistribute_rewards(), dtype=torch.float32)


def compute_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lam: float,
    dones: Optional[torch.Tensor] = None,
    last_value: float = 0.0,
) -> torch.Tensor:
    """Compute generalized advantage estimation (GAE).

    Optimized to avoid tensor allocation in loop.

    Parameters
    ----------
    rewards : Tensor, shape (T,)
        Step rewards
    values : Tensor, shape (T,) or (T+1,)
        Value function estimates at each step
    gamma : float
        Discount factor
    lam : float
        GAE lambda parameter
    dones : Tensor, optional, shape (T,)
        Done flags (True = episode ended)
    last_value : float
        Bootstrap value for next state

    Returns
    -------
    advantages : Tensor, shape (T,)
        Computed advantages
    """
    device = rewards.device
    rewards = rewards.to(dtype=torch.float32, device=device)
    values = values.to(dtype=torch.float32, device=device)
    T = rewards.shape[0]

    if dones is None:
        masks = torch.ones(T, dtype=torch.float32, device=device)
    else:
        masks = (1.0 - dones.to(dtype=torch.float32, device=device))

    # Pre-compute next values to avoid tensor allocation in loop
    next_values = torch.cat([
        values[1:],
        torch.tensor([float(last_value)], dtype=torch.float32, device=device)
    ])

    # Compute deltas vectorized
    deltas = rewards + gamma * next_values[:T] * masks - values[:T]

    # Accumulate GAE backward (still needs sequential loop for dependencies)
    advantages = torch.zeros(T, dtype=torch.float32, device=device)
    gae = 0.0
    for t in range(T - 1, -1, -1):
        gae = float(deltas[t]) + gamma * lam * masks[t] * gae
        advantages[t] = gae

    return advantages


def _to_1d_tensor(x) -> Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().flatten().to(torch.float32)
    return torch.tensor([float(x)], dtype=torch.float32)


def discount_returns(rewards: Tensor, gamma: float) -> Tensor:
    """Compute discounted returns using vectorized operations (~5-10x faster).

    Parameters
    ----------
    rewards : Tensor, shape (T,)
        Step rewards from an episode
    gamma : float
        Discount factor

    Returns
    -------
    returns : Tensor, shape (T,)
        Discounted returns-to-go at each step
    """
    T = rewards.shape[0]
    if T == 0:
        return rewards

    # Vectorized cumsum: G_t = r_t + gamma*G_{t+1}
    # Process in reverse: flip, cumsum with decay, flip back
    flipped = torch.flip(rewards, [0])
    discounts = torch.pow(gamma, torch.arange(T, dtype=rewards.dtype, device=rewards.device))
    cumsum_flipped = torch.cumsum(flipped * discounts, dim=0)
    returns = torch.flip(cumsum_flipped, [0]) / discounts

    return returns


@torch.no_grad()
def compute_gae(rewards: Tensor, values: Tensor, gamma: float, lam: float,
                dones: Optional[Tensor] = None, last_value: float = 0.0) -> Tensor:
    """
    Wrapper that calls the vectorized `compute_advantages` implementation to
    ensure consistent behavior and device placement. Kept for backward
    compatibility with callers that expect `compute_gae`.
    """
    return compute_advantages(rewards, values, gamma, lam, dones=dones, last_value=last_value)


class TrajectoryBuffer:
    """
    DCL-ready episodic buffer.

    Stores per-step:
      - state (dict with 'graph' -> HeteroData)
      - action (int)
      - reward_raw (float)
      - logprob_sel (float)
      - value_pred (float)
      - logpis_nodes (Tensor): per-action-node log-probs from old policy
      - token_ids (list[int]) optional

      - target_pi (Tensor): per-action-node improved policy (zeros except enabled nodes)
      - q_first (Tensor): rollout diagnostic targets (vector or scalar)
    """

    def __init__(self, gam=1.0, lam=1.0, data_type='hetero', action_mode="node_selection"):
        self.gam = float(gam)
        self.lam = float(lam)
        self.data_type = data_type
        self.action_mode = action_mode

        # rolling storage
        self.states: List[Dict[str, Any]] = []
        self.actions: List[int] = []
        self.rewards_raw: Tensor = torch.empty(0, dtype=torch.float32)
        self.logprobs_sel: Tensor = torch.empty(0, dtype=torch.float32)
        self.values_pred: Tensor = torch.empty(0, dtype=torch.float32)
        self.logpis_nodes: List[Tensor] = []
        self.token_ids: List[List[int]] = []

        # DCL fields (lists; must be aligned with states)
        self.target_pi: List[Tensor] = []
        self.q_first: List[Tensor] = []

        # computed targets for training
        self.returns_: Tensor = torch.empty(0, dtype=torch.float32)
        self.advantages_: Tensor = torch.empty(0, dtype=torch.float32)

        # episode window
        self.start = 0
        self.end = 0

    def __len__(self) -> int:
        return len(self.states)

    @torch.no_grad()
    def store(self, state, action, reward, logprob, value, logpis,
              token_ids: Optional[List[int]] = None,
              target_pi: Optional[Tensor] = None,
              q_first: Optional[Tensor] = None):
        """Append one interaction; always append placeholders for DCL fields to keep alignment."""
        self.states.append(state)
        self.actions.append(int(action))
        self.rewards_raw = torch.cat([self.rewards_raw, _to_1d_tensor(reward)], dim=0)
        self.logprobs_sel = torch.cat([self.logprobs_sel, _to_1d_tensor(logprob)], dim=0)
        self.values_pred = torch.cat([self.values_pred, _to_1d_tensor(value)], dim=0)
        self.logpis_nodes.append(None if logpis is None else logpis.detach().flatten().to(torch.float32))
        self.token_ids.append([] if token_ids is None else list(token_ids))

        # --- DCL fields: build safe placeholders if missing ---
        if target_pi is None:
            n = 0
            try:
                g = state['graph']
                if hasattr(g, 'node_types') and 'a_transition' in g.node_types:
                    n = g['a_transition'].num_nodes
            except Exception:
                pass
            target_pi = (torch.full((n,), 1.0 / max(n, 1), dtype=torch.float32) if n > 0
                         else torch.tensor([], dtype=torch.float32))
        if q_first is None:
            q_first = torch.tensor([0.0], dtype=torch.float32)

        self.target_pi.append(target_pi.detach().cpu())
        self.q_first.append(q_first.detach().cpu())

        self.end += 1

    def apply_action_rewards(self, action_rewards: dict):
        """
        Replace rewards in the buffer using redistributed rewards per action.
        Each step may be associated with multiple token_ids; we use the first one to find the action.
        """
        new_rewards = []
        for i, token_ids in enumerate(getattr(self, 'token_ids', [])):
            # Use first token_id to trace back to action
            if token_ids:
                # Assume token_ids are linked to actions via causal trace
                # You may need to store action index per step if not already
                action_index = None
                for tid in token_ids:
                    if tid in action_rewards:
                        action_index = tid
                        break
                if action_index is not None:
                    new_rewards.append(action_rewards[action_index])
                else:
                    new_rewards.append(self.rewards_raw[i].item())  # fallback to original reward
            else:
                new_rewards.append(self.rewards_raw[i].item())  # no token info, keep original

        return torch.tensor(new_rewards, dtype=torch.float32, requires_grad=True)

    @torch.no_grad()
    def finish(self, credits: Optional[Any] = None, mode: str = "replace"):
        """
        Close current episode [start:end). Compute returns and advantages for that slice.
        credits:
            - None                          -> use raw rewards
            - Tensor/list length T          -> per-step redistributed rewards (aligned with this episode window)
            - object with .redistribute_rewards() -> will be called to get length-T vector
        mode:
            - "replace": use credits as the rewards
            - "add":     rewards_raw + credits
        """
        tau = slice(self.start, self.end)
        rewards_ep = self.rewards_raw[tau]
        values_ep = self.values_pred[tau]
        dones_ep = torch.zeros_like(rewards_ep, dtype=torch.bool)
        dones_ep[-1] = True

        # --- Check if we're in causal_rl mode (step rewards are zero) ---
        has_step_rewards = rewards_ep.sum().item() != 0.0

        # --- Handle credits if provided ---
        if credits is not None:
            if hasattr(credits, "redistribute_rewards") and callable(credits.redistribute_rewards):
                cr = credits.redistribute_rewards()
                credits_vec = torch.as_tensor(cr, dtype=torch.float32)
            else:
                credits_vec = torch.as_tensor(credits, dtype=torch.float32)

            # DEBUG: Check length matching
            import sys
            if credits_vec.numel() != rewards_ep.numel():
                print(f"[CAUSAL-WARNING] Credits length {credits_vec.numel()} != episode length {rewards_ep.numel()}", file=sys.stderr)
                print(f"  Credits: {credits_vec}", file=sys.stderr)
            else:
                if credits_vec.sum() > 0:
                    print(f"[CAUSAL-OK] Episode {rewards_ep.numel()} steps, credits sum={credits_vec.sum():.4f}", file=sys.stderr)

            if mode == "replace":
                # Use redistributed rewards directly as returns (causal mode)
                returns_ep = credits_vec
            else:
                # Add credits to original rewards, then discount
                modified_rewards = rewards_ep + credits_vec
                returns_ep = discount_returns(modified_rewards, self.gam)
        else:
            # No credits: discount original rewards
            returns_ep = discount_returns(rewards_ep, self.gam)
            credits_vec = None

        # --- Compute advantages ---
        if not has_step_rewards and credits_vec is not None:
            # Causal RL mode with value function integration:
            # In causal mode, step rewards are zero. The actual return signal comes from credits.
            # We compute cumulative credits (which are the TRUE returns), then use standard GAE
            # to compute advantages. This way:
            # - Value network learns to predict cumulative credits (the true returns)
            # - GAE provides variance reduction through temporal smoothing
            # - Advantage signal is clean and properly bootstrapped

            # Credits are per-step redistributed rewards from causal traces
            # We need to compute proper returns and advantages from them
            # Use standard discount_returns to get cumulative discounted credits
            returns_ep = discount_returns(credits_vec, self.gam)

            # Now use standard GAE with credits as the reward signal
            adv_ep = compute_advantages(
                credits_vec,
                values_ep,
                self.gam,
                self.lam,
                dones=dones_ep
            )
        else:
            # Normal mode: use step rewards for advantages
            adv_ep = compute_advantages(rewards_ep, values_ep, self.gam, self.lam, dones=dones_ep)

        if self.returns_.numel() == 0:
            self.returns_ = returns_ep.clone()
            self.advantages_ = adv_ep.clone()
        else:
            self.returns_ = torch.cat([self.returns_, returns_ep], dim=0)
            self.advantages_ = torch.cat([self.advantages_, adv_ep], dim=0)

        self.start = self.end

        # DEBUG: Log advantage statistics for causal RL
        import sys
        if not has_step_rewards and credits_vec is not None:
            mean_adv = adv_ep.mean().item()
            std_adv = adv_ep.std().item() if len(adv_ep) > 1 else 0.0
            print(f"[ADV-STATS] Ep len={len(adv_ep)}, credits_sum={credits_vec.sum():.2f}, "
                  f"mean={mean_adv:.6f}, std={std_adv:.6f}, min={adv_ep.min():.6f}, max={adv_ep.max():.6f}",
                  file=sys.stderr)


    def finish_wip(self, causal_trace: Optional[Sequence[float]] = None):
        """
        Close current episode [start:end). Compute returns and advantages for that slice.
        If 'credits' is a stepwise vector (same length as episode), we add it to rewards before discounting.
        """
        tau = slice(self.start, self.end)
        rewards_ep = self.rewards_raw[tau]
        values_ep = self.values_pred[tau]
        dones_ep = torch.zeros_like(rewards_ep, dtype=torch.bool)
        dones_ep[-1] = True

        # Create end_flag tensor: True for the last step of the trajectory
        end_flag = torch.zeros_like(self.rewards_raw[tau], dtype=torch.bool)
        end_flag[-1] = True  # Mark the last step as terminal

        if causal_trace is not None:
            rewards_ep = self.apply_causal_credits(causal_trace)

        # values = compute_advantages(rewards_ep, values_ep, self.gam, self.lam,
        #                            dones=end_flag)  # TODO: check if this is in the right place

        returns_ep = discount_rewards(rewards_ep, self.gam)
        # self.rewards[tau] = rewards
        # self.values[tau] = values

        adv_ep = compute_gae(rewards_ep, values_ep, self.gam, self.lam, dones=dones_ep)

        if self.returns_.numel() == 0:
            self.returns_ = returns_ep.clone()
            self.advantages_ = adv_ep.clone()
        else:
            self.returns_ = torch.cat([self.returns_, returns_ep], dim=0)
            self.advantages_ = torch.cat([self.advantages_, adv_ep], dim=0)

        self.start = self.end

    def clear(self):
        """Reset the buffer."""
        self.states.clear()
        self.actions.clear()
        self.token_ids.clear()
        self.logpis_nodes.clear()
        self.target_pi.clear()
        self.q_first.clear()

        self.rewards_raw = torch.empty(0, dtype=torch.float32)
        self.logprobs_sel = torch.empty(0, dtype=torch.float32)
        self.values_pred = torch.empty(0, dtype=torch.float32)
        self.returns_ = torch.empty(0, dtype=torch.float32)
        self.advantages_ = torch.empty(0, dtype=torch.float32)
        self.start = 0
        self.end = 0

    @torch.no_grad()
    def _normalize_advantages(self, adv: Tensor) -> Tensor:
        """Normalize advantages to zero mean and unit variance.

        Use population-standard-deviation (unbiased=False) for stability and add
        a small epsilon to avoid division by zero when the advantages are constant.
        """
        eps = 1e-8
        std = adv.std(unbiased=False)
        mean = adv.mean()
        return (adv - mean) / (std + eps)

    @torch.no_grad()
    def _normalize_returns(self, returns: Tensor) -> Tensor:
        """Normalize returns to zero mean and unit variance for value training.

        Use population-standard-deviation (unbiased=False) and an epsilon guard to
        avoid NaNs when the returns are constant.
        """
        eps = 1e-8
        std = returns.std(unbiased=False)
        mean = returns.mean()
        return (returns - mean) / (std + eps)

    @torch.no_grad()
    def get(self, batch_size=64, normalize_advantages=True, normalize_returns=False,
            sort=True, drop_remainder=False):
        """
        Build a PyG DataLoader. Each HeteroData sample contains:
          - y (action), advantage, value (discounted return), logprobs (scalar)
          - logpis split per node type (old policy over action nodes, incl. postpone)
          - target_pi split per node type (optional)
          - q_first (per-sample rollout scores)
        """
        N = len(self.states)
        if N == 0:
            raise ValueError("Buffer is empty; store()/finish() before get().")
        if self.returns_.numel() != N or self.advantages_.numel() != N:
            raise RuntimeError("Not all steps finalized; call finish() after each episode.")

        idx = np.arange(N)
        if sort:
            np.random.shuffle(idx)

        # Convert to torch tensor for consistent indexing behavior
        idx_tensor = torch.from_numpy(idx).long()

        # reorder everything consistently
        states = [self.states[i] for i in idx]
        actions = np.asarray([self.actions[i] for i in idx], dtype=np.int64)
        returns = self.returns_[idx_tensor]
        adv = self.advantages_[idx_tensor]
        logprob_s = self.logprobs_sel[idx_tensor]
        logpis = [self.logpis_nodes[i] for i in idx]  # per-step old-policy vector
        target_pi = [self.target_pi[i] for i in idx]
        q_first = [self.q_first[i] for i in idx]

        if normalize_advantages:
            adv = self._normalize_advantages(adv)

        if normalize_returns:
            returns = self._normalize_returns(returns)

        # build HeteroData list
        data_list: List[HeteroData] = []
        for i, s in enumerate(states):
            g: HeteroData = copy.deepcopy(s['graph'])
            g.y = torch.tensor(actions[i], dtype=torch.long)
            # Ensure scalar shapes for value and advantage
            g.advantage = adv[i].reshape(()).detach()
            g.value = returns[i].reshape(()).detach()
            g.logprobs = logprob_s[i].reshape(()).detach()

            # --- Standardize lp to 1-D
            lp = logpis[i] if isinstance(logpis[i], torch.Tensor) else torch.tensor([], dtype=torch.float32)
            if lp.dim() == 2 and lp.size(-1) == 1:
                lp = lp.squeeze(-1)
            g.logpis = lp  # optional: whole-step old policy for debugging

            # --- Split lp per node type by actual counts in THIS sample
            nA = g['a_transition'].x.size(0) if 'a_transition' in g.node_types else 0
            nP = g['postpone'].x.size(0) if 'postpone' in g.node_types else 0
            total_expected = nA + nP

            if total_expected > 0:
                if lp.numel() != total_expected:
                    # Try to recover: if we only have a_transition and lp is longer, truncate; if shorter, pad zeros.
                    # Warn once so you can inspect upstream ordering/length.
                    print(f"[WARN:get] old logpis length {lp.numel()} != nA+nP {total_expected} (sample {i}). "
                          f"{'Truncating' if lp.numel() > total_expected else 'Padding with zeros'}.")

                    if lp.numel() > total_expected:
                        lp = lp[:total_expected]
                    else:
                        lp = torch.cat([lp, torch.zeros(total_expected - lp.numel(), dtype=torch.float32)], dim=0)

            # Attach per-type old policy in the SAME order as actions_dict: [a_transition][postpone]
            if 'a_transition' in g.node_types:
                g['a_transition'].logpis = lp[:nA] if total_expected else torch.tensor([], dtype=torch.float32)
            if 'postpone' in g.node_types:
                g['postpone'].logpis = lp[nA:nA + nP] if total_expected else torch.tensor([], dtype=torch.float32)

            # --- DCL / targets (optional): split target_pi with same ordering
            tpi = target_pi[i] if isinstance(target_pi[i], torch.Tensor) else torch.tensor([], dtype=torch.float32)
            if tpi.dim() == 2 and tpi.size(-1) == 1:
                tpi = tpi.squeeze(-1)
            g.target_pi = tpi

            qf = q_first[i] if isinstance(q_first[i], torch.Tensor) else torch.tensor([0.0], dtype=torch.float32)
            g.q_first = qf

            if 'a_transition' in g.node_types:
                g['a_transition'].target_pi = (
                    tpi[:nA] if tpi.numel() >= nA else torch.tensor([], dtype=torch.float32)
                )
            if 'postpone' in g.node_types:
                g['postpone'].target_pi = (
                    tpi[nA:nA + nP] if tpi.numel() >= (nA + nP) else torch.tensor([], dtype=torch.float32)
                )

            data_list.append(g)

        if drop_remainder and (len(data_list) % batch_size != 0):

            keep = len(data_list) - (len(data_list) % batch_size)
            data_list = data_list[:keep]

        loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)

        return loader


def print_status_bar(i, epochs, history, verbose=1):
    """
    Print a formatted status bar showing training progress.

    Parameters
    ----------
    i : int
        Current epoch number.
    epochs : int
        Total number of epochs.
    history : dict
        Dictionary containing training metrics.
    verbose : int, optional
        Verbosity level. Default is 1.
        - 0: No output
        - 1 or higher: One line per epoch
    """
    metrics = "".join([" - {}: {:.4f}".format(m, history[m][i])
                       for m in ['mean_returns']])
    end = "\n" if verbose == 2 or i+1 == epochs else ""
    if verbose > 0:
        print("\rEpoch {}/{}".format(i+1, epochs) + metrics, end=end)
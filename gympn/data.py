from typing import Optional


from torch_geometric.data import Data


from typing import Optional, List, Dict, Any, Sequence
import copy
import os
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
    """Compute discounted returns-to-go: G_t = r_t + gamma * G_{t+1}.

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

    # Reverse accumulation: G_t = r_t + gamma * G_{t+1}
    returns = torch.zeros_like(rewards)
    G = 0.0
    for t in range(T - 1, -1, -1):
        G = float(rewards[t]) + gamma * G
        returns[t] = G

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


def smdp_gae(rewards: Tensor, values: Tensor, discounts: Tensor, lam: float,
             dones: Optional[Tensor] = None) -> Tensor:
    """GAE with a PER-STEP (SMDP) discount instead of a constant gamma.

        delta_t = r_t + d_t * V(s_{t+1}) - V(s_t)
        A_t     = delta_t + d_t * lam * A_{t+1}
    where d_t = discounts[t] * mask_t  (mask=0 at a terminal step, dropping the
    bootstrap). This is the time-aware advantage of SMDP actor-critic: the
    continuation V(s_{t+1}) is discounted by the actual sojourn, so a decision
    that only spends time (e.g. postpone) carries its opportunity cost.
    """
    T = rewards.shape[0]
    device = rewards.device
    if dones is None:
        masks = torch.ones(T, dtype=torch.float32, device=device)
    else:
        masks = 1.0 - dones.to(dtype=torch.float32, device=device)
    next_values = torch.cat([values[1:], torch.zeros(1, dtype=values.dtype, device=device)])
    adv = torch.zeros(T, dtype=torch.float32, device=device)
    gae = 0.0
    for t in range(T - 1, -1, -1):
        d = float(discounts[t]) * float(masks[t])
        delta = float(rewards[t]) + d * float(next_values[t]) - float(values[t])
        gae = delta + d * lam * gae
        adv[t] = gae
    return adv


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

    def __init__(self, gam=1.0, lam=1.0, data_type='hetero', action_mode="node_selection",
                 causal_scheme='lrq', causal_pg=False, causal_rl=False,
                 causal_beta=0.0, causal_mu=0.0, smdp_discount=False):
        self.gam = float(gam)
        self.lam = float(lam)
        self.data_type = data_type
        self.action_mode = action_mode
        # Only 'lrq' is implemented (all other redistribution schemes were
        # removed); kept as a parameter so stale configs fail loudly in
        # redistribute_rewards() rather than silently changing behaviour.
        self.causal_scheme = causal_scheme
        # SMDP discount rate for the causal advantage: the continuation after a
        # decision is discounted by e^{-causal_beta * tau}, tau = elapsed time to
        # the next decision (from per-step clocks). 0.0 = no time discounting
        # (legacy gamma=1 behaviour). >0 gives time an opportunity cost, so a
        # decision that only spends time (postpone) is correctly penalised.
        # Shared by the causal branch below AND the standard path when
        # smdp_discount=True.
        self.causal_beta = float(causal_beta)
        # LRQ hybrid coefficient: A = (1-mu)*A_LRQ + mu*A_GAE, where A_GAE
        # is the standard SMDP-GAE advantage on the RAW temporal rewards. mu=0
        # (default) = pure LRQ (no cross-case smearing, foreclosure-blind);
        # raising it restores a gradient path for resource-contention effects
        # LRQ cannot see, at the cost of re-admitting temporal smearing. See
        # CAUSAL_LRQ_PROPOSAL.md §3.
        self.causal_mu = float(causal_mu)
        self.causal_pg = causal_pg
        self.causal_rl = bool(causal_rl)
        # Standard (non-causal) path only: replace the constant-gamma GAE with
        # the per-sojourn SMDP form e^{-causal_beta*tau} (same clock/formula as
        # the causal branch), with no CV/lineage term -- this is LCV's exact
        # ĉ=0 limiting case ("lcv0"), used to factor the time-discount choice
        # out of the LCV-vs-PPO comparison (CAUSAL_LCV_CONTROL_VARIATE.md).
        # False (default) = legacy constant-gamma PPO, fully unaffected; `gam`
        # becomes unused for advantage computation when this is True (by
        # construction: see finish()).
        self.smdp_discount = bool(smdp_discount)

        # rolling storage
        self.states: List[Dict[str, Any]] = []
        self.actions: List[int] = []
        # Per-step scalars are accumulated into plain Python lists and only
        # materialized into 1-D tensors on demand (_materialize_steps). The old
        # per-step torch.cat made store() O(T^2) in the number of steps and
        # issued one torch.cat per step; the lists make store() O(1) and rebuild
        # the tensors with a single cat each, only when a consumer needs them.
        self._rewards_buf: List[Tensor] = []
        self._logprobs_buf: List[Tensor] = []
        self._values_buf: List[Tensor] = []
        self._steps_dirty: bool = False
        self.rewards_raw: Tensor = torch.empty(0, dtype=torch.float32)
        self.logprobs_sel: Tensor = torch.empty(0, dtype=torch.float32)
        self.values_pred: Tensor = torch.empty(0, dtype=torch.float32)
        self.logpis_nodes: List[Tensor] = []
        # Metadata about stored logpis: record node counts at storage time to debug ordering issues
        self.logpis_meta: List[dict] = []
        self.token_ids: List[List[int]] = []

        # Per-step decision time (wall-clock of the simulator at the decision).
        # Used to derive sojourn times tau_t for SMDP discounting.
        self.times: List[float] = []

        # LRQ-v3 / LQI / LCV: rollout-time auxiliary predictions (q_off(s,a)
        # for lrq3/lqi; v_off(s) for lcv) and the trace-computed off-lineage
        # targets (mc_q - lrq2 credit), aligned with steps.
        self.qoff_pred: List[float] = []
        self.qoff_targets_: Tensor = torch.empty(0, dtype=torch.float32)
        # LCV: per-step control-variate terms R_off - v_off(s); the adaptive
        # coefficient c_hat is estimated over the WHOLE epoch in get().
        self.cvterms_: Tensor = torch.empty(0, dtype=torch.float32)
        self.last_cv_coef: float = 0.0
        self.last_cv_var_reduction: float = 0.0
        # LVA: per-step lineage credits (lrq2 Q-samples) used as the critic's
        # AUXILIARY regression target — representation shaping only; the value
        # head keeps its own GAE-return target (see finish() 'lva' branch).
        self.qlin_targets_: Tensor = torch.empty(0, dtype=torch.float32)

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
              q_first: Optional[Tensor] = None,
              time: Optional[float] = None,
              qoff: Optional[float] = None):
        """Append one interaction; always append placeholders for DCL fields to keep alignment."""
        self.times.append(0.0 if time is None else float(time))
        # LRQ-v3: rollout-time off-lineage prediction q_off(s, a_taken)
        # (0.0 when the q_off head is absent — only the lrq3 path reads it).
        self.qoff_pred.append(0.0 if qoff is None else float(qoff))
        # Freeze the graph structure at storage time so later environment
        # mutations cannot change node counts and break alignment with the
        # stored logpis. The observation graph is rebuilt from scratch every step
        # (no shared/cached skeleton) and only its tensors are read back during
        # training-data construction, so a tensor-level HeteroData.clone() is
        # sufficient and ~10x cheaper than a full Python deepcopy. We keep a
        # shallow copy of the surrounding dict (its 'actions_dict' is never read
        # back from the buffer during training). See Tier 2.2 in
        # PERFORMANCE_OPTIMIZATION_PLAN.md.
        if isinstance(state, dict):
            frozen = dict(state)
            g = state.get('graph')
            frozen['graph'] = g.clone() if hasattr(g, 'clone') else copy.deepcopy(g)
            self.states.append(frozen)
        else:
            self.states.append(state.clone() if hasattr(state, 'clone') else copy.deepcopy(state))
        self.actions.append(int(action))
        self._rewards_buf.append(_to_1d_tensor(reward))
        self._logprobs_buf.append(_to_1d_tensor(logprob))
        self._values_buf.append(_to_1d_tensor(value))
        self._steps_dirty = True
        self.logpis_nodes.append(None if logpis is None else logpis.detach().flatten().to(torch.float32))
        # Record expected node counts at storage time (helps detect later mismatches)
        try:
            g = state['graph']
            nA = g['a_transition'].num_nodes if hasattr(g, 'node_types') and 'a_transition' in g.node_types else 0
            nP = g['postpone'].num_nodes if hasattr(g, 'node_types') and 'postpone' in g.node_types else 0
            self.logpis_meta.append({'nA': int(nA), 'nP': int(nP), 'total': int(nA + nP)})
        except Exception:
            self.logpis_meta.append({'nA': 0, 'nP': 0, 'total': 0})
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

    def _materialize_steps(self):
        """Rebuild the per-step tensors from the accumulation lists (one cat each).

        Idempotent and cheap: a no-op unless store() appended since the last
        materialization. Consumers that read rewards_raw/logprobs_sel/values_pred
        must call this first."""
        if not self._steps_dirty:
            return
        self.rewards_raw = (torch.cat(self._rewards_buf, dim=0) if self._rewards_buf
                            else torch.empty(0, dtype=torch.float32))
        self.logprobs_sel = (torch.cat(self._logprobs_buf, dim=0) if self._logprobs_buf
                             else torch.empty(0, dtype=torch.float32))
        self.values_pred = (torch.cat(self._values_buf, dim=0) if self._values_buf
                            else torch.empty(0, dtype=torch.float32))
        self._steps_dirty = False

    def apply_action_rewards(self, action_rewards: dict):
        """
        Replace rewards in the buffer using redistributed rewards per action.
        Each step may be associated with multiple token_ids; we use the first one to find the action.
        """
        self._materialize_steps()
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

    def _taus_from_times(self) -> Tensor:
        """Elapsed sojourn tau_t = time_{t+1} - time_t for the CURRENT episode
        slice [start:end); 0 at the last step (no continuation to discount).
        Shared clock for every SMDP-discounted path (causal branch and the
        standard path when smdp_discount=True)."""
        times_ep = torch.tensor(self.times[self.start:self.end], dtype=torch.float32)
        if times_ep.numel() >= 2:
            taus = times_ep[1:] - times_ep[:-1]
            taus = torch.cat([taus, torch.zeros(1, dtype=torch.float32)])
        else:
            taus = torch.zeros_like(times_ep)
        return torch.clamp(taus, min=0.0)

    def _smdp_discounts(self) -> Tensor:
        """Per-step continuation discount e^{-causal_beta*tau}."""
        return torch.exp(-self.causal_beta * self._taus_from_times())

    @torch.no_grad()
    def finish(self, credits: Optional[Any] = None, mode: str = "replace"):
        """
        Close current episode [start:end). Compute returns and advantages for that slice.

        credits:
            - None                          -> use raw rewards (standard RL)
            - Tensor/list length T          -> per-step causal attributions
            - object with .redistribute_rewards() -> will be called to get length-T vector

        ═══════════════════════════════════════════════════════════════════
        Causal RL Mode — LRQ (Lineage-Restricted Q)
        ═══════════════════════════════════════════════════════════════════

        When causal_rl=True, credits[t] = Q_t is the hindsight lineage return
        of decision t: the FULL discounted sum of every future reward in whose
        causal lineage the decision sits (per-decision Monte-Carlo Q-sample of
        the policy-gradient theorem, NOT a redistribution of the return — a
        reward with k lineage decisions is counted k times by design). The
        estimator is plain centering:

            Value target: G_t = Q_t                 (V_L regresses E[Q_t | s_t])
            Advantage:    A_t = Q_t − V_L(s_t)      (no GAE/TD chaining)

        Chaining return-to-go/GAE over hindsight credits is deliberately NOT
        done: backward-assigned mass vanishes from later decisions' return-
        to-go, systematically undervaluing chain completion (the bias that
        sank the removed credits-as-rewards schemes; see
        CAUSAL_REC_CRITICAL_REVIEW.md §3 and CAUSAL_LRQ_PROPOSAL.md).

        Postpone decisions are lineage members under token-flow postpone, so
        waiting sees the same rewards discounted from an earlier clock (timing
        penalty) and a beneficial wait sees the larger mass it enabled.
        self.causal_mu optionally mixes in the standard SMDP-GAE advantage on
        raw temporal rewards as a hedge for foreclosure effects (resource
        contention across cases) that lineage-restricted credit cannot see.
        """
        self._materialize_steps()
        tau = slice(self.start, self.end)
        rewards_ep = self.rewards_raw[tau]
        values_ep = self.values_pred[tau]
        dones_ep = torch.zeros_like(rewards_ep, dtype=torch.bool)
        dones_ep[-1] = True

        # --- Check if we're in causal_rl mode ---
        # Use the explicit flag instead of the fragile rewards_ep.sum() != 0 heuristic.
        # The old heuristic would incorrectly trigger causal mode when standard RL
        # step rewards happened to cancel to zero.
        is_causal = self.causal_rl

        # --- Handle credits if provided ---
        qoff_targets_ep = None
        if credits is not None:
            if hasattr(credits, "redistribute_rewards") and callable(credits.redistribute_rewards):
                if self.causal_scheme in ('lrq3', 'lqi', 'lcv', 'lva'):
                    # LRQ-v3 / LQI / LCV / LVA need BOTH decompositions of Q from
                    # the trace: the sharp lineage part (lrq2) and the full
                    # return-to-go (mc_q); their difference is the off-lineage
                    # regression target for the q_off head (lrq3/lqi/lcv), and
                    # the lrq2 part is LVA's auxiliary critic target.
                    cr_lin = credits.redistribute_rewards(scheme='lrq2',
                                                          beta=self.causal_beta)
                    cr_full = credits.redistribute_rewards(scheme='mc_q',
                                                           beta=self.causal_beta)
                    credits_vec = torch.as_tensor(cr_lin, dtype=torch.float32)
                    cr_full = torch.as_tensor(cr_full, dtype=torch.float32)
                    qoff_targets_ep = cr_full - credits_vec
                elif self.causal_scheme == 'lrq2c':
                    # lrq2 + sparse counterfactual correction: the cheap
                    # lineage-restricted Q (direct channel), plus the EXACT
                    # indirect (opportunity-cost) term measured by CRN forks at
                    # foreclosure-gated decisions (stored on the buffer by
                    # run_episode as `_lrq2c_indirect`, aligned 1:1 with steps).
                    # See LINEAGE_SPARSE_CORRECTION.md.
                    cr = credits.redistribute_rewards(scheme='lrq2',
                                                      beta=self.causal_beta)
                    credits_vec = torch.as_tensor(cr, dtype=torch.float32)
                    ind = getattr(self, '_lrq2c_indirect', None)
                    if ind is not None:
                        iv = torch.as_tensor(ind, dtype=torch.float32)
                        if iv.numel() == credits_vec.numel():
                            credits_vec = credits_vec + iv
                else:
                    cr = credits.redistribute_rewards(
                        scheme=self.causal_scheme,
                        beta=self.causal_beta,
                    )
                    credits_vec = torch.as_tensor(cr, dtype=torch.float32)
            else:
                credits_vec = torch.as_tensor(credits, dtype=torch.float32)

            # Length-alignment guard: credits MUST match episode length 1:1.
            # When causal_rl=True the simulator always consults the agent for
            # every action transition (including single-binding), so
            # len(action_transitions) == len(env.step() calls) == ep_len.
            # A mismatch indicates a real bug; do NOT silently pad/truncate.
            ep_len = rewards_ep.numel()
            cr_len = credits_vec.numel()
            if cr_len != ep_len:
                raise RuntimeError(
                    f"[CAUSAL-RL] Credits length ({cr_len}) != episode length ({ep_len}). "
                    f"This indicates a bug in the causal trace / simulator step alignment."
                )

            # LRQ-v2: postpone steps get the SMDP-TD advantage instead of a
            # lineage Q-sample (consistent supports; see causal_traces "lrq2").
            # Identify them from the trace's action records (1:1 with steps,
            # asserted above). Plain-vector credits (tests) => no mask.
            postpone_mask = None
            if self.causal_scheme in ('lrq2', 'lrq2c', 'ccf', 's_ccf') and hasattr(credits, 'transition_history'):
                flags = []
                for act in credits.transition_history.get_action_transitions():
                    tid = getattr(act.get('transition'), '_id', None)
                    flags.append(bool(isinstance(tid, str) and tid.startswith('postpone_')))
                postpone_mask = torch.tensor(flags, dtype=torch.bool)

            # LRQ: the intra-lineage discount uses TRACE decision times u_t,
            # while the SMDP sojourns below use BUFFER times (pn.clock read
            # agent-side). The estimator is only coherent if the two clocks
            # agree — same guard philosophy as the length check above.
            if hasattr(credits, 'transition_history'):
                trace_times = [a.get('time') for a in
                               credits.transition_history.get_action_transitions()]
                buf_times = self.times[self.start:self.end]
                for k, (tt, bt) in enumerate(zip(trace_times, buf_times)):
                    if tt is not None and abs(float(tt) - float(bt)) > 1e-6:
                        raise RuntimeError(
                            f"[LRQ] Decision-clock mismatch at step {k}: trace "
                            f"u_t={tt} vs buffer time={bt}. The Q-sample discount "
                            f"and the SMDP sojourns must share one clock."
                        )

            if mode == "replace_rewards":
                # RUDDER-style baseline: the credits are REDISTRIBUTED STEP
                # REWARDS (prefix-measurable, return-conserving), so they are
                # consumed through the ORDINARY GAE path exactly as if the
                # environment had emitted them — unlike LRQ Q-samples, which
                # must never be chained.
                rewards_ep = credits_vec
                credits_vec = None
                returns_ep = None  # set by the standard branch (adv + V)
            elif mode == "replace":
                returns_ep = credits_vec
            else:
                modified_rewards = rewards_ep + credits_vec
                returns_ep = discount_returns(modified_rewards, self.gam)
        else:
            # No credits: discount original rewards
            returns_ep = discount_returns(rewards_ep, self.gam)
            credits_vec = None

        # --- Compute advantages ---
        if credits_vec is not None and is_causal:
            # LRQ MODE — credits are per-decision hindsight lineage Q-samples
            # (full reward mass per lineage member, NOT a partition of the
            # return; see CAUSAL_LRQ_PROPOSAL.md). The policy-gradient-theorem
            # estimator is plain centering,
            #   A_t = Q_t - V_L(s_t),
            # with the value head V_L regressed on Q_t itself (returns_ep) and
            # NO return-to-go/GAE chaining over the credits — chaining hindsight
            # credits is exactly what produced the chain-dilution bias in the
            # removed credits-as-rewards schemes (CAUSAL_REC_CRITICAL_REVIEW.md §3).
            discounts = self._smdp_discounts()

            if self.causal_scheme == 'lcv':
                # LCV — adaptive measured control variate on the standard
                # SMDP-GAE estimator (see CAUSAL_LCV_CONTROL_VARIATE.md):
                #   A_cv = A_GAE - c_hat * (R_off - v_off(s))
                # R_off = measured off-lineage return (mc_q - lrq2, exact from
                # the trace); v_off = learned state-only centering; c_hat is
                # estimated over the whole epoch in get() (classical optimal
                # CV coefficient), so c_hat -> 0 recovers plain SMDP-GAE PPO
                # exactly — the floor is built into the estimator. Value
                # targets are the ordinary GAE returns (adv + V), NOT credit
                # targets: the policy/critic path is 100% standard PPO.
                adv_base = smdp_gae(rewards_ep, values_ep, discounts,
                                    self.lam, dones=dones_ep)
                returns_ep = adv_base + values_ep
                adv_ep = adv_base
                qoff_ep = torch.tensor(self.qoff_pred[self.start:self.end],
                                       dtype=torch.float32)
                cvterm_ep = qoff_targets_ep - qoff_ep
                if self.cvterms_.numel() == 0:
                    self.cvterms_ = cvterm_ep.clone()
                else:
                    self.cvterms_ = torch.cat([self.cvterms_, cvterm_ep], dim=0)
                if self.qoff_targets_.numel() == 0:
                    self.qoff_targets_ = qoff_targets_ep.clone()
                else:
                    self.qoff_targets_ = torch.cat(
                        [self.qoff_targets_, qoff_targets_ep], dim=0)
            elif self.causal_scheme == 'lva':
                # LVA — Lineage Value Auxiliary: the policy path is 100%
                # standard SMDP-GAE PPO on raw rewards (identical to lcv0 /
                # LCV's c_hat=0 floor — no CV term, no credit advantages), and
                # the lineage enters ONLY as the critic's auxiliary regression
                # target: the per-decision lrq2 Q-sample, predicted by a second
                # head on the critic's shared encoder (networks.HeteroCritic
                # aux_head). Aux gradients shape the representation; the value
                # head keeps its own unbiased GAE-return target, so the policy
                # gradient stays exactly PPO's (floor by construction, like
                # LCV — but consuming the full per-decision credit vector
                # instead of one epoch-level scalar).
                adv_ep = smdp_gae(rewards_ep, values_ep, discounts,
                                  self.lam, dones=dones_ep)
                returns_ep = adv_ep + values_ep
                if self.qlin_targets_.numel() == 0:
                    self.qlin_targets_ = credits_vec.clone()
                else:
                    self.qlin_targets_ = torch.cat(
                        [self.qlin_targets_, credits_vec], dim=0)
            elif self.causal_scheme in ('lrq3', 'lqi'):
                # LRQ-v3 / LQI — exact decomposition (for 'lqi' the buffer
                # advantage is a PLACEHOLDER: the AWR path recomputes per-state
                # advantages from the fresh Q heads at training time), no mixing knob:
                #   A_t = c_lineage(a_t) + q_off(s_t, a_t) - V(s_t)
                # where c_lineage is the MC lineage Q-sample (lrq2 credits,
                # postpone = 0), q_off is the rollout-time prediction of the
                # learned off-lineage head, and V regresses the FULL
                # return-to-go (mc_q sample = c_lineage + off target), giving
                # one coherent semantics: V(s) ~ E[Q_full(s, a~pi)]. Postpone
                # needs no special branch — its Q is entirely off-lineage and
                # the head carries it.
                qoff_ep = torch.tensor(self.qoff_pred[self.start:self.end],
                                       dtype=torch.float32)
                returns_ep = credits_vec + qoff_targets_ep  # = mc_q sample
                adv_ep = credits_vec + qoff_ep - values_ep
                if self.qoff_targets_.numel() == 0:
                    self.qoff_targets_ = qoff_targets_ep.clone()
                else:
                    self.qoff_targets_ = torch.cat(
                        [self.qoff_targets_, qoff_targets_ep], dim=0)
            else:
                returns_ep = credits_vec
                adv_ep = credits_vec - values_ep

            if postpone_mask is not None and postpone_mask.any():
                # LRQ-v2: postpone gates nothing, so its true Q IS the delayed
                # continuation — use the SMDP-TD form on the same critic:
                #   A(postpone) = e^{-beta*tau} V(s') - V(s)
                # and regress V toward that same (detached) target at postpone
                # steps, keeping one coherent value semantics: V(s) ~ E[Q̂|s].
                # This is the consistent-support fix for the v1 collapse where
                # postpone's lineage swallowed the discounted backlog mass.
                next_v = torch.cat([values_ep[1:],
                                    torch.zeros(1, dtype=values_ep.dtype)])
                pp_target = discounts * next_v
                adv_ep = torch.where(postpone_mask, pp_target - values_ep, adv_ep)
                returns_ep = torch.where(postpone_mask, pp_target, returns_ep)

            if self.causal_mu > 0.0:
                # Foreclosure hedge: mix in the standard SMDP-GAE advantage on
                # the RAW temporal rewards. Caveat: it bootstraps with the same
                # critic, which is trained on LRQ targets, so the hybrid term
                # is approximate — keep mu small.
                adv_gae = smdp_gae(rewards_ep, values_ep, discounts,
                                   self.lam, dones=dones_ep)
                adv_ep = (1.0 - self.causal_mu) * adv_ep + self.causal_mu * adv_gae
        else:
            # STANDARD RL MODE: temporal rewards → GAE advantages
            if self.smdp_discount:
                # lcv0 (LCV's ĉ=0 twin): same per-sojourn discount as the
                # causal branch, no CV/lineage term -- plain SMDP-GAE PPO.
                # `gam` is deliberately unused on this path (the discount
                # comes from elapsed time, not a per-decision constant).
                taus = self._taus_from_times()
                if (taus.numel() > 1 and self.causal_beta > 0.0
                        and float(taus.abs().sum()) < 1e-9):
                    # All-zero sojourns on a multi-step episode means decision
                    # times were never recorded upstream (buffer.store(...,
                    # time=...) missing) -- fail loudly rather than silently
                    # degenerating to an undiscounted (beta-inert) GAE.
                    raise RuntimeError(
                        "[smdp_discount] all sojourn times are 0 for a "
                        f"{taus.numel()}-step episode with causal_beta="
                        f"{self.causal_beta} > 0; decision times were not "
                        "recorded (buffer.store(..., time=...) missing)."
                    )
                discounts = torch.exp(-self.causal_beta * taus)
                adv_ep = smdp_gae(rewards_ep, values_ep, discounts, self.lam, dones=dones_ep)
            else:
                adv_ep = compute_advantages(rewards_ep, values_ep, self.gam, self.lam, dones=dones_ep)
            # Fix #3: value target = GAE / TD(λ) return (advantage + V), which is
            # lower-variance than the Monte-Carlo return-to-go set above. Applied
            # only to the genuine standard-PPO path (no credits); the causal /
            # credits paths keep their own return targets.
            if credits_vec is None:
                returns_ep = adv_ep + values_ep

        if self.returns_.numel() == 0:
            self.returns_ = returns_ep.clone()
            self.advantages_ = adv_ep.clone()
        else:
            self.returns_ = torch.cat([self.returns_, returns_ep], dim=0)
            self.advantages_ = torch.cat([self.advantages_, adv_ep], dim=0)

        self.start = self.end

        # DEBUG: Log advantage statistics for causal RL
        #import sys
        #if is_causal and credits_vec is not None:
        #    mean_adv = adv_ep.mean().item()
        #    std_adv = adv_ep.std().item() if len(adv_ep) > 1 else 0.0
        #    print(f"[ADV-STATS] Ep len={len(adv_ep)}, credits_sum={credits_vec.sum():.2f}, "
        #          f"mean={mean_adv:.6f}, std={std_adv:.6f}, min={adv_ep.min():.6f}, max={adv_ep.max():.6f}",
        #          file=sys.stderr)


    def finish_wip(self, causal_trace: Optional[Sequence[float]] = None):
        """
        Close current episode [start:end). Compute returns and advantages for that slice.
        If 'credits' is a stepwise vector (same length as episode), we add it to rewards before discounting.
        """
        self._materialize_steps()
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
        self.logpis_meta.clear()
        self.times.clear()
        self.target_pi.clear()
        self.q_first.clear()

        self._rewards_buf.clear()
        self._logprobs_buf.clear()
        self._values_buf.clear()
        self._steps_dirty = False
        self.rewards_raw = torch.empty(0, dtype=torch.float32)
        self.logprobs_sel = torch.empty(0, dtype=torch.float32)
        self.values_pred = torch.empty(0, dtype=torch.float32)
        self.returns_ = torch.empty(0, dtype=torch.float32)
        self.advantages_ = torch.empty(0, dtype=torch.float32)
        self.qoff_pred.clear()
        self.qoff_targets_ = torch.empty(0, dtype=torch.float32)
        self.cvterms_ = torch.empty(0, dtype=torch.float32)
        self.qlin_targets_ = torch.empty(0, dtype=torch.float32)
        self.start = 0
        self.end = 0

    @torch.no_grad()
    def _normalize_advantages(self, adv: Tensor) -> Tensor:
        """Standard per-batch advantage normalization: zero mean, unit variance.

        ``(adv - mean) / (std + eps)`` with population std (unbiased=False). Applied
        when the agent's ``normalize_advantages`` flag is set (default ON, standard
        PPO). It is needed to learn low-margin tasks (tiny raw advantages would give
        too weak a gradient otherwise). The downside — the step not decaying at
        convergence → post-peak drift — is handled by best-checkpoint restore, not
        by disabling normalization (which under-learns low-headroom envs). See
        INSTABILITY_ANALYSIS.md.
        """
        eps = 1e-8
        mean = adv.mean()
        std = adv.std(unbiased=False)
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
        self._materialize_steps()
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
        # LRQ-v3: off-lineage regression targets ride along when present.
        qoff_t = (self.qoff_targets_[idx_tensor]
                  if self.qoff_targets_.numel() == self.returns_.numel() else None)
        # LVA: lineage-credit auxiliary targets for the critic's aux head.
        qlin_t = (self.qlin_targets_[idx_tensor]
                  if self.qlin_targets_.numel() == self.returns_.numel() else None)
        logpis = [self.logpis_nodes[i] for i in idx]  # per-step old-policy vector
        target_pi = [self.target_pi[i] for i in idx]
        q_first = [self.q_first[i] for i in idx]

        # LCV: apply the adaptive measured control variate BEFORE any
        # normalization, with the classical optimal coefficient estimated over
        # the whole epoch: c_hat = Cov(A, cv) / Var(cv), clipped to [0, 2].
        # Var(cv) ~ 0 or negative correlation => c_hat = 0 => plain SMDP-GAE
        # PPO exactly (the floor property).
        if (self.causal_scheme == 'lcv'
                and self.cvterms_.numel() == self.advantages_.numel()
                and adv.numel() > 1):
            cv = self.cvterms_[idx_tensor]
            cv = cv - cv.mean()
            var_cv = float((cv * cv).mean())
            if var_cv > 1e-12:
                c_hat = float(((adv - adv.mean()) * cv).mean()) / var_cv
                c_hat = max(0.0, min(c_hat, 2.0))
                var_before = float(adv.var(unbiased=False))
                adv = adv - c_hat * cv
                var_after = float(adv.var(unbiased=False))
                self.last_cv_coef = c_hat
                # Fractional advantage-variance reduction achieved by the CV
                # this epoch — the quantity the CV theorem is about (and the
                # "mirage-critique" reporting standard for CV papers).
                self.last_cv_var_reduction = (
                    (var_before - var_after) / var_before if var_before > 1e-12 else 0.0)

        if normalize_advantages:
            adv = self._normalize_advantages(adv)

        if normalize_returns:
            returns = self._normalize_returns(returns)

        # build HeteroData list
        data_list: List[HeteroData] = []
        for i, s in enumerate(states):
            # Tier 2.2: per-sample, per-epoch copy used only to attach labels
            # (y/advantage/value/logprobs/logpis) without mutating the stored
            # graph. HeteroData.clone() copies the tensors at the torch level and
            # is far cheaper than copy.deepcopy walking the Python object graph;
            # the labels land on this clone, never on the buffered graph.
            g: HeteroData = s['graph'].clone()
            g.y = torch.tensor(actions[i], dtype=torch.long)
            # Ensure scalar shapes for value and advantage
            g.advantage = adv[i].reshape(()).detach()
            g.value = returns[i].reshape(()).detach()
            g.logprobs = logprob_s[i].reshape(()).detach()
            if qoff_t is not None:
                g.qoff_target = qoff_t[i].reshape(()).detach()
            if qlin_t is not None:
                g.qlin_target = qlin_t[i].reshape(()).detach()

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
                    # Detailed diagnostics to track ordering/length mismatches between
                    # stored per-step logpis and the node counts in the HeteroData sample.
                    debug_flag = os.environ.get('GP_DEBUG_NET_ORDER', '0') == '1'
                    info = {
                        'sample_index': i,
                        'lp_len': int(lp.numel()),
                        'expected_nA_nP': int(total_expected),
                        'nA': int(nA),
                        'nP': int(nP),
                        'g_node_types': list(g.node_types) if hasattr(g, 'node_types') else None,
                        'a_batch_len': int(getattr(g['a_transition'], 'batch', torch.tensor([], dtype=torch.int64)).numel()) if 'a_transition' in g.node_types else None,
                        'p_batch_len': int(getattr(g['postpone'], 'batch', torch.tensor([], dtype=torch.int64)).numel()) if 'postpone' in g.node_types else None,
                        'stored_meta': self.logpis_meta[i] if i < len(self.logpis_meta) else None,
                    }

                    msg = (f"[ERROR:get] old logpis length {info['lp_len']} != nA+nP {info['expected_nA_nP']} "
                           f"(sample {i}). nA={info['nA']} nP={info['nP']}. "
                           f"node_types={info['g_node_types']}. a_batch_len={info['a_batch_len']} p_batch_len={info['p_batch_len']}")

                    # If debugging is enabled, raise an error with diagnostics to force a fix upstream.
                    if debug_flag:
                        raise RuntimeError(msg + "\nSet GP_DEBUG_NET_ORDER=0 to fallback to legacy padding behaviour.")

                    # Otherwise, fall back to legacy behaviour but log the mismatch so it's visible.
                    print(f"[WARN:get] {msg} Falling back to padding/truncation for now.")
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

        # === CRITICAL FIX: Ensure at least one batch is created ===
        # If drop_remainder=True and data_list < batch_size, we'd get zero batches → no training
        if drop_remainder and len(data_list) < batch_size:
            import sys
            print(f"[WARN:get] Data size {len(data_list)} < batch_size {batch_size} with drop_remainder=True. "
                  f"Disabling drop_remainder to ensure ≥1 batch for training.", file=sys.stderr)
            drop_remainder = False

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
from typing import Optional, List, Dict, Any
import copy
import math
import os
import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from gympn.flat_graph import is_flat

Tensor = torch.Tensor


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


class _CachedBatches(list):
    """The batches of a shuffle=False DataLoader, collated once and replayed on
    every pass.

    Starting to iterate a torch DataLoader draws one number from the global
    torch RNG (its worker base seed), even with no workers. To keep training
    bit-identical to iterating the DataLoader once per pass, collating here
    leaves the RNG state untouched and every pass makes that same single draw.
    Without this, the cache shifts the RNG stream and later rollouts sample
    different actions (same computation, different random numbers)."""

    def __init__(self, loader):
        state = torch.get_rng_state()
        super().__init__(loader)
        torch.set_rng_state(state)

    def __iter__(self):
        torch.empty((), dtype=torch.int64).random_()
        return super().__iter__()


class TrajectoryBuffer:
    """Episodic rollout buffer.

    Stores per step the observation (dict with 'graph'), the action, the raw
    reward, the log-probability of the chosen action, the value prediction,
    the old policy's log-probabilities over every action node, and the
    simulator clock at the decision. ``finish()`` closes an episode and
    computes its advantages and value targets; ``get()`` hands back the
    training batches.

    Advantages:
      - default: GAE with a constant discount ``gam``;
      - ``smdp_discount=True``: SMDP-GAE, the continuation after a decision is
        discounted by e^{-beta * tau}, tau the time to the next decision;
      - ``nfgae=True``: net-factored GAE, one SMDP-GAE per net component on the
        component's own decision clock (suite/paper/NFGAE_THEORY.md). Needs
        ``_nf_comp``/``_nf_rew`` set by the agent before ``finish()``.
    """

    def __init__(self, gam=1.0, lam=1.0, beta=0.0, smdp_discount=False, nfgae=False):
        self.gam = float(gam)
        self.lam = float(lam)
        # SMDP discount rate: e^{-beta * tau} per sojourn tau. Used by the
        # smdp_discount and nfgae paths; `gam` is unused there.
        self.beta = float(beta)
        self.smdp_discount = bool(smdp_discount)
        self.nfgae = bool(nfgae)

        # rolling storage
        self.states: List[Dict[str, Any]] = []
        self.actions: List[int] = []
        # Per-step scalars are accumulated into plain Python lists and only
        # materialized into 1-D tensors on demand (_materialize_steps): store()
        # stays O(1) and each tensor is rebuilt with a single cat.
        self._rewards_buf: List[Tensor] = []
        self._logprobs_buf: List[Tensor] = []
        self._values_buf: List[Tensor] = []
        self._steps_dirty: bool = False
        self.rewards_raw: Tensor = torch.empty(0, dtype=torch.float32)
        self.logprobs_sel: Tensor = torch.empty(0, dtype=torch.float32)
        self.values_pred: Tensor = torch.empty(0, dtype=torch.float32)
        self.logpis_nodes: List[Tensor] = []

        # Per-step decision time (simulator clock at the decision), for the
        # sojourn times tau_t of SMDP discounting.
        self.times: List[float] = []

        # nfgae: deciding component and per-component reward of each step of
        # the current episode (set by the agent, consumed by finish()).
        self._nf_comp = None
        self._nf_rew = None

        # computed targets for training
        self.returns_: Tensor = torch.empty(0, dtype=torch.float32)
        self.advantages_: Tensor = torch.empty(0, dtype=torch.float32)

        # episode window
        self.start = 0
        self.end = 0

    def __len__(self) -> int:
        return len(self.states)

    @torch.no_grad()
    def store(self, state, action, reward, logprob, value, logpis, time: Optional[float] = None):
        """Append one interaction."""
        self.times.append(0.0 if time is None else float(time))
        # Freeze the graph at storage time so later environment mutations cannot
        # change node counts and break alignment with the stored logpis. Only
        # its tensors are read back, so a tensor-level clone is enough (and ~10x
        # cheaper than a deepcopy); the surrounding dict is copied shallowly.
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
        self.end += 1

    def _materialize_steps(self):
        """Rebuild the per-step tensors from the accumulation lists (one cat each).
        A no-op unless store() appended since the last materialization."""
        if not self._steps_dirty:
            return
        self.rewards_raw = (torch.cat(self._rewards_buf, dim=0) if self._rewards_buf
                            else torch.empty(0, dtype=torch.float32))
        self.logprobs_sel = (torch.cat(self._logprobs_buf, dim=0) if self._logprobs_buf
                             else torch.empty(0, dtype=torch.float32))
        self.values_pred = (torch.cat(self._values_buf, dim=0) if self._values_buf
                            else torch.empty(0, dtype=torch.float32))
        self._steps_dirty = False

    def _taus_from_times(self) -> Tensor:
        """Sojourn tau_t = time_{t+1} - time_t over the current episode slice
        [start:end); 0 at the last step (no continuation to discount)."""
        times_ep = torch.tensor(self.times[self.start:self.end], dtype=torch.float32)
        if times_ep.numel() >= 2:
            taus = times_ep[1:] - times_ep[:-1]
            taus = torch.cat([taus, torch.zeros(1, dtype=torch.float32)])
        else:
            taus = torch.zeros_like(times_ep)
        return torch.clamp(taus, min=0.0)

    def _nfgae(self, values_ep: Tensor) -> Tensor:
        """Net-factored GAE: per-component SMDP-GAE, each component on its OWN
        decision clock, with the component-indexed critic (values_ep[t] =
        V_{c(t)}(s_t), see HeteroCritic pool_mask). For component c with
        decision steps t_1 < t_2 < ...:
          R_i = c's rewards over steps [t_i, t_{i+1})   (undiscounted at t_i,
                the same convention smdp_gae uses for a single step)
          d_i = exp(-beta (time[t_{i+1}] - time[t_i])), 0 after c's last step
          A_i = R_i + d_i V_c(s_{t_{i+1}}) - V_c(s_{t_i}) + lam d_i A_{i+1}
        With one component this is term-for-term smdp_gae (NFGAE_THEORY.md,
        Theorem 2a)."""
        T = values_ep.shape[0]
        comp = list(self._nf_comp[:T])
        rw = list(self._nf_rew[:T])
        times = self.times[self.start:self.end]
        adv_ep = torch.zeros(T, dtype=torch.float32)
        for c in set(comp):
            idx = [t for t in range(T) if comp[t] == c]
            gae = 0.0
            for j in range(len(idx) - 1, -1, -1):
                t = idx[j]
                nxt = idx[j + 1] if j + 1 < len(idx) else None
                R = sum(rw[u].get(c, 0.0) for u in range(t, nxt if nxt is not None else T))
                if nxt is None:
                    d, v_next = 0.0, 0.0
                else:
                    d = math.exp(-self.beta * max(0.0, times[nxt] - times[t]))
                    v_next = float(values_ep[nxt])
                delta = R + d * v_next - float(values_ep[t])
                gae = delta + d * self.lam * gae
                adv_ep[t] = gae
        return adv_ep

    @torch.no_grad()
    def finish(self):
        """Close the current episode [start:end) and compute its advantages and
        value targets (the GAE / TD(lambda) return, advantage + V)."""
        self._materialize_steps()
        tau = slice(self.start, self.end)
        rewards_ep = self.rewards_raw[tau]
        values_ep = self.values_pred[tau]
        dones_ep = torch.zeros_like(rewards_ep, dtype=torch.bool)
        dones_ep[-1] = True

        if self.nfgae and self._nf_comp is not None:
            adv_ep = self._nfgae(values_ep)
            self._nf_comp = self._nf_rew = None
        elif self.smdp_discount:
            taus = self._taus_from_times()
            if taus.numel() > 1 and self.beta > 0.0 and float(taus.abs().sum()) < 1e-9:
                # All-zero sojourns on a multi-step episode means the decision
                # times were never recorded (store(..., time=...) missing):
                # fail loudly rather than run an undiscounted GAE.
                raise RuntimeError(
                    "[smdp_discount] all sojourn times are 0 for a "
                    f"{taus.numel()}-step episode with beta={self.beta} > 0; decision "
                    "times were not recorded (buffer.store(..., time=...) missing).")
            discounts = torch.exp(-self.beta * taus)
            adv_ep = smdp_gae(rewards_ep, values_ep, discounts, self.lam, dones=dones_ep)
        else:
            adv_ep = compute_advantages(rewards_ep, values_ep, self.gam, self.lam, dones=dones_ep)
        returns_ep = adv_ep + values_ep

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
        self.logpis_nodes.clear()
        self.times.clear()

        self._rewards_buf.clear()
        self._logprobs_buf.clear()
        self._values_buf.clear()
        self._steps_dirty = False
        self.rewards_raw = torch.empty(0, dtype=torch.float32)
        self.logprobs_sel = torch.empty(0, dtype=torch.float32)
        self.values_pred = torch.empty(0, dtype=torch.float32)
        self.returns_ = torch.empty(0, dtype=torch.float32)
        self.advantages_ = torch.empty(0, dtype=torch.float32)
        self._nf_comp = self._nf_rew = None
        self.start = 0
        self.end = 0

    @torch.no_grad()
    def _normalize_advantages(self, adv: Tensor) -> Tensor:
        """Per-batch advantage normalization: zero mean, unit (population) variance.

        Needed to learn low-margin tasks (tiny raw advantages would give too
        weak a gradient); the post-peak drift it can cause is handled by
        best-checkpoint restore."""
        eps = 1e-8
        mean = adv.mean()
        std = adv.std(unbiased=False)
        return (adv - mean) / (std + eps)

    @torch.no_grad()
    def _normalize_returns(self, returns: Tensor) -> Tensor:
        """Normalize returns to zero mean and unit (population) variance."""
        eps = 1e-8
        std = returns.std(unbiased=False)
        mean = returns.mean()
        return (returns - mean) / (std + eps)

    @torch.no_grad()
    def get(self, batch_size=64, normalize_advantages=True, normalize_returns=False,
            sort=True, drop_remainder=False):
        """
        Build the training batches. Each sample carries y (action), advantage,
        value (target), logprobs (chosen action) and the old policy's logpis
        split per action-node type (a_transition, postpone).
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
        idx_tensor = torch.from_numpy(idx).long()

        # reorder everything consistently
        states = [self.states[i] for i in idx]
        actions = np.asarray([self.actions[i] for i in idx], dtype=np.int64)
        returns = self.returns_[idx_tensor]
        adv = self.advantages_[idx_tensor]
        logprob_s = self.logprobs_sel[idx_tensor]
        logpis = [self.logpis_nodes[i] for i in idx]  # per-step old-policy vector

        if normalize_advantages:
            adv = self._normalize_advantages(adv)
        if normalize_returns:
            returns = self._normalize_returns(returns)

        data_list: List[HeteroData] = []
        for i, s in enumerate(states):
            # Labels go on a tensor-level clone, never on the buffered graph.
            g = s['graph'].clone()                     # HeteroData, or FlatGraph (flat_obs)
            g.y = torch.tensor(actions[i], dtype=torch.long)
            g.advantage = adv[i].reshape(()).detach()
            g.value = returns[i].reshape(()).detach()
            g.logprobs = logprob_s[i].reshape(()).detach()

            lp = logpis[i] if isinstance(logpis[i], torch.Tensor) else torch.tensor([], dtype=torch.float32)
            if lp.dim() == 2 and lp.size(-1) == 1:
                lp = lp.squeeze(-1)
            g.logpis = lp

            # Split lp per node type by the actual counts in THIS sample.
            flat = is_flat(g)
            if flat:
                nA, nP = int(g.a_idx.numel()), int(g.p_idx.numel())
            else:
                nA = g['a_transition'].x.size(0) if 'a_transition' in g.node_types else 0
                nP = g['postpone'].x.size(0) if 'postpone' in g.node_types else 0
            total_expected = nA + nP

            if total_expected > 0 and lp.numel() != total_expected:
                msg = (f"[ERROR:get] old logpis length {int(lp.numel())} != nA+nP {total_expected} "
                       f"(sample {i}). nA={nA} nP={nP}.")
                if os.environ.get('GP_DEBUG_NET_ORDER', '0') == '1':
                    raise RuntimeError(msg)
                print(f"[WARN:get] {msg} Falling back to padding/truncation.")
                if lp.numel() > total_expected:
                    lp = lp[:total_expected]
                else:
                    lp = torch.cat([lp, torch.zeros(total_expected - lp.numel(), dtype=torch.float32)], dim=0)

            # Per-type old policy, in the SAME order as actions_dict: [a_transition][postpone]
            empty = torch.tensor([], dtype=torch.float32)
            if flat:
                g.logpis_a = lp[:nA] if total_expected else empty
                g.logpis_p = lp[nA:nA + nP] if total_expected else empty
            else:
                if 'a_transition' in g.node_types:
                    g['a_transition'].logpis = lp[:nA] if total_expected else empty
                if 'postpone' in g.node_types:
                    g['postpone'].logpis = lp[nA:nA + nP] if total_expected else empty

            data_list.append(g)

        # Ensure at least one batch: with drop_remainder and fewer samples than
        # batch_size there would be no training at all.
        if drop_remainder and len(data_list) < batch_size:
            import sys
            print(f"[WARN:get] Data size {len(data_list)} < batch_size {batch_size} with drop_remainder=True. "
                  f"Disabling drop_remainder to ensure >=1 batch for training.", file=sys.stderr)
            drop_remainder = False

        if drop_remainder and (len(data_list) % batch_size != 0):
            keep = len(data_list) - (len(data_list) % batch_size)
            data_list = data_list[:keep]

        # Collate once and hand back the batches. Every consumer iterates them
        # once per update pass, and with shuffle=False each pass saw the same
        # batches anyway, so this only removes the repeated collation. No
        # consumer writes into a batch.
        return _CachedBatches(DataLoader(data_list, batch_size=batch_size, shuffle=False))


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
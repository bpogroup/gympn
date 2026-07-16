"""Resurrected pre-LRQ redistribution schemes — EXPERIMENT BASELINES ONLY.

The library removed every redistribution scheme except LRQ (commit 320f138);
the paper (PAPER_PLAN_LRQ.md, E1/E7) still needs flow_dag / rec / the legacy
heuristics as *baselines*. This module resurrects them from the frozen
pre-removal snapshot `_causal_traces_e790d95.py` (extracted verbatim from
commit e790d95) and grafts them onto the live classes at runtime:

    import legacy_schemes
    legacy_schemes.install(gamma=0.9, self_credit=0.5, legacy_lam=0.0)
    # ... then train with args_dict['causal_scheme'] in
    #     {'flow_dag', 'rec', 'shapley_dag', 'flow', 'uniform', ...}
    # (args_dict bypasses the CLI choices, which only allow 'lrq')

What install() patches:
  1. CausalTraces.redistribute_rewards — legacy scheme names are routed to the
     snapshot's implementation (executed against the live trace instance; the
     TokenHistory/TransitionHistory data layout is unchanged). 'lrq' still goes
     to the current implementation.
  2. TrajectoryBuffer.finish — legacy schemes are credits-as-rewards and need
     the removed SMDP consumption path (returns = smdp_discounted_returns,
     advantage = smdp_gae over credits at lambda = legacy_lam). LRQ episodes
     are untouched (delegated to the current finish).

Never import this from library code or the main suite run — only from the
baseline experiment scripts, so headline LRQ results can never silently run
through patched classes.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _causal_traces_e790d95 as _old  # frozen snapshot, do not edit

from gympn.causal_traces import CausalTraces
from gympn.data import TrajectoryBuffer, smdp_gae

LEGACY_SCHEMES = {"flow_dag", "shapley_dag", "rec", "flow",
                  "exponential", "linear", "uniform", "depth", "hybrid"}

# Legacy hyperparameters that the current buffer no longer threads
# (pre-removal suite defaults: causal_gamma=0.9, causal_self_credit=0.5,
# causal_lam=0.0 i.e. TD(0)).
_params = {"gamma": 0.9, "self_credit": 0.5, "legacy_lam": 0.0}

_installed = False


def smdp_discounted_returns(rewards, discounts):
    """Return-to-go with a per-step SMDP discount (removed from data.py;
    verbatim from commit e790d95): G_t = r_t + discounts[t] * G_{t+1}."""
    T = rewards.shape[0]
    out = torch.zeros_like(rewards)
    G = 0.0
    for t in range(T - 1, -1, -1):
        G = float(rewards[t]) + float(discounts[t]) * G
        out[t] = G
    return out


def install(gamma=0.9, self_credit=0.5, legacy_lam=0.0):
    """Graft the legacy schemes onto the live classes. Idempotent; later calls
    only update the legacy hyperparameters."""
    global _installed
    _params.update(gamma=float(gamma), self_credit=float(self_credit),
                   legacy_lam=float(legacy_lam))
    if _installed:
        return
    _installed = True

    # --- 1. redistribution: route legacy names to the snapshot ------------
    # The snapshot methods live on _old.CausalTraces; copy the private helpers
    # onto the live class so `self._redistribute_*` resolves on live instances.
    for name in ("_redistribute_flow_dag", "_redistribute_rec",
                 "_redistribute_shapley_dag", "_shapley_values",
                 "_SHAPLEY_EXACT_MAX", "_SHAPLEY_MC_SAMPLES"):
        setattr(CausalTraces, name, getattr(_old.CausalTraces, name))

    current_redistribute = CausalTraces.redistribute_rewards

    def redistribute(self, scheme="lrq", include_postpone=None, beta=0.0):
        if scheme not in LEGACY_SCHEMES:
            return current_redistribute(self, scheme=scheme,
                                        include_postpone=include_postpone,
                                        beta=beta)
        return _old.CausalTraces.redistribute_rewards(
            self, gamma=_params["gamma"], scheme=scheme,
            include_postpone=include_postpone,
            self_credit=_params["self_credit"], beta=beta)

    CausalTraces.redistribute_rewards = redistribute

    # --- 2. consumption: credits-as-rewards SMDP path for legacy schemes ---
    current_finish = TrajectoryBuffer.finish

    @torch.no_grad()
    def finish(self, credits=None, mode="replace"):
        if (credits is None or not self.causal_rl
                or self.causal_scheme not in LEGACY_SCHEMES):
            return current_finish(self, credits=credits, mode=mode)

        # Verbatim logic of the pre-removal causal branch (e790d95 data.py):
        # per-step discount d_t = e^{-beta*tau_t};
        # value target G_t = c_t + d_t*G_{t+1};
        # advantage A_t = c_t + d_t*V(s_{t+1}) - V(s_t) (+ legacy_lam mixing).
        self._materialize_steps()
        tau = slice(self.start, self.end)
        rewards_ep = self.rewards_raw[tau]
        values_ep = self.values_pred[tau]
        dones_ep = torch.zeros_like(rewards_ep, dtype=torch.bool)
        dones_ep[-1] = True

        if hasattr(credits, "redistribute_rewards") and callable(credits.redistribute_rewards):
            cr = credits.redistribute_rewards(scheme=self.causal_scheme,
                                              beta=self.causal_beta)
            credits_vec = torch.as_tensor(cr, dtype=torch.float32)
        else:
            credits_vec = torch.as_tensor(credits, dtype=torch.float32)
        if credits_vec.numel() != rewards_ep.numel():
            raise RuntimeError(
                f"[LEGACY] Credits length ({credits_vec.numel()}) != episode "
                f"length ({rewards_ep.numel()})."
            )

        times_ep = torch.tensor(self.times[self.start:self.end], dtype=torch.float32)
        if times_ep.numel() >= 2:
            taus = times_ep[1:] - times_ep[:-1]
            taus = torch.cat([taus, torch.zeros(1, dtype=torch.float32)])
        else:
            taus = torch.zeros_like(credits_vec)
        taus = torch.clamp(taus, min=0.0)
        discounts = torch.exp(-self.causal_beta * taus)
        returns_ep = smdp_discounted_returns(credits_vec, discounts)
        adv_ep = smdp_gae(credits_vec, values_ep, discounts,
                          _params["legacy_lam"], dones=dones_ep)

        if self.returns_.numel() == 0:
            self.returns_ = returns_ep.clone()
            self.advantages_ = adv_ep.clone()
        else:
            self.returns_ = torch.cat([self.returns_, returns_ep], dim=0)
            self.advantages_ = torch.cat([self.advantages_, adv_ep], dim=0)
        self.start = self.end

    TrajectoryBuffer.finish = finish

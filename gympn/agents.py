"""Policy gradient agents that support changing state spaces, specifically for graph environments.

Currently includes policy gradient agent (i.e., Monte Carlo policy
gradient or vanilla policy optimization
agent.
"""
import math
import numpy as np
import os
import random
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import multiprocessing as mp
from typing import Dict

from gympn.data import TrajectoryBuffer, print_status_bar
from gympn.logging_utils import Logger, TrainingMetrics, TestMetrics, get_logger
from gympn.potential import topology_potential


# torch.autograd.set_detect_anomaly(True)


def _normalized_entropy(probs, logpis, batch_index):
    """Compute mean normalized entropy across decisions with variable action spaces.

    For each decision (identified by ``batch_index``), the raw entropy
    ``H = -sum(p * log(p))`` is divided by ``log(num_actions)`` so that the
    result lies in [0, 1] regardless of how many actions are available.

    Parameters
    ----------
    probs : Tensor  – 1-D probability per action node (flat).
    logpis : Tensor – 1-D log-probability per action node.
    batch_index : Tensor – maps each node to its decision index.

    Returns
    -------
    Tensor (scalar) – mean normalized entropy in [0, 1].
    """
    ent_sum = torch.tensor(0.0, device=probs.device)
    count = 0
    for s in batch_index.unique():
        mask = (batch_index == s)
        p = probs[mask]
        lp = logpis[mask]
        n_actions = int(mask.sum().item())
        raw_ent = -(p * lp).sum()
        if n_actions > 1:
            ent_sum = ent_sum + raw_ent / torch.log(
                torch.tensor(float(n_actions), device=probs.device))
        count += 1
    if count == 0:
        return torch.tensor(0.0, device=probs.device)
    return ent_sum / count

# ============================================================================
# MULTIPROCESSING WORKER FUNCTION
# ============================================================================
def _run_episode_worker(args):
    """Worker function for parallel episode collection (picklable).

    Must be at module level to be serializable by multiprocessing.
    """
    agent, env_copy, max_len = args
    return agent.run_episode(env_copy, max_episode_length=max_len, buffer=None)


class Agent:
    """Base class for policy gradient agents.

    All functionality for policy gradient is implemented in this
    class. Derived classes must define the property `policy_loss`
    which is used to train the policy.

    Parameters
    ----------
    policy_network : network
        The network for the policy model.
    policy_lr : float, optional
        The learning rate for the policy model.
    policy_updates : int, optional
        The number of policy updates per epoch of training.
    value_network : network, None, or string, optional
        The network for the value model.
    value_lr : float, optional
        The learning rate for the value model.
    value_updates : int, optional
        The number of value updates per epoch of training.
    gam : float, optional
        The discount rate.
    lam : float, optional
        The parameter for generalized advantage estimation.
    normalize_advantages : bool, optional
        Whether to apply per-batch advantage normalization (zero-mean, unit-variance).
        Default is **True** (the standard PPO choice). Normalization is needed to
        learn LOW-headroom tasks: when the optimal-vs-suboptimal margin is small the
        raw advantages are tiny, and without rescaling the gradient is too weak to
        reach the optimum (empirically, env b peaks at 8/9 with it off vs 9/9 with
        it on). Its downside — the step size not decaying at convergence → post-peak
        drift — is *cosmetic* given best-checkpoint restore (the deployed policy is
        the peak, not the drifted tail). Turn it off only for ablations.
        See INSTABILITY_ANALYSIS.md.
    normalize_returns : bool, optional
        Whether to normalize returns (discounted cumulative rewards) for value training.
        Default is False (IMPORTANT for correctness).

        ⚠️  CRITICAL: If normalize_returns=True, the value network will be trained on
        normalized targets (mean=0, std=1). However, value predictions from rollout time
        will be in the normalized scale, while GAE calculations use raw rewards/credits.
        This creates a scale mismatch in delta = reward + gamma * v_{t+1} - v_t.

        RECOMMENDATION: Keep normalize_returns=False (default) to avoid this mismatch.
        If you need stable value training, prefer advantage normalization
        (``normalize_advantages=True``) — but note it can reintroduce post-peak drift.
    kld_limit : float or None, optional
        Early-stopping limit on the mean per-state KL(pi_old || pi_new),
        checked after each inner policy epoch. Default is None (disabled).

        The KL is computed EXACTLY per state, over that state's own (variable
        size) action set: KL_s = sum_a p_old(a|s) (log p_old − log p_new),
        then averaged over states — the standard PPO target_kl quantity, valid
        for variable |A(s)| because old and new always share a state's support.
        (Historical note: before 2026-07-09 this metric was the SIGNED
        chosen-action log-ratio — cancellation-prone, high-variance and
        |A(s)|-dependent — and early stopping was rightly discouraged. That no
        longer applies.) PPO's clip bounds each surrogate term but NOT the
        realized policy shift after multiple inner epochs; the KL brake is the
        guard against the rare catastrophic update that collapses a
        near-deterministic policy (observed as post-convergence drift).
    ent_bonus : float, optional
        Bonus factor for sampled policy entropy.

    """

    def __init__(self,
                 policy_network, value_network, policy_lr=1e-4, policy_updates=1,
                 value_lr=1e-3, value_updates=25,
                 gam=0.99, lam=0.97, normalize_advantages=True, eps=0.2,
                 kld_limit=None, ent_bonus=0.0, test_in_train=True, vf_coeff=0.05,
                 normalize_returns=False, lr_schedule=False,
                 causal_scheme='lrq', causal_pg=False,
                 causal_rl=False, causal_beta=0.0, causal_mu=0.0,
                 smdp_discount=False, causal_aux_coef=0.5,
                 qoff_network=None, qlin_network=None, cf_config=None,
                 phi_coef=0.0, phi_decay=0.9, phi_cap=None,
                 ls_hca_hhat_floor=0.05, ls_hca_factor_clip=3.0,
                 ls_hca_smoothing_alpha=1.0, ls_hca_state_min_n=30,
                 ls_hca_state_epochs=50, ls_hca_state_lr=0.05,
                 ls_hca_flat_fallback=False, ls_hca_residual=True,
                 ls_hca_state_l2=0.0, ls_hca_consistent=True,
                 ls_hca_ratio=True, ls_hca_weight_clip=10.0,
                 ls_hca_z_feature='delay', ls_hca_z_bins=4,
                 ls_hca_gate=False):
        self.policy_model = policy_network
        self.policy_loss = NotImplementedError
        self.policy_optimizer = torch.optim.Adam(params=list(policy_network.parameters()),
                                                 lr=policy_lr)
        self.policy_updates = policy_updates

        self.value_model = value_network
        self.value_loss = torch.nn.MSELoss()
        self.value_optimizer = torch.optim.Adam(params=list(value_network.parameters()), lr=value_lr)
        self.value_updates = value_updates

        # LRQ-v3 / LQI: learned per-action-node Q heads (see networks.
        # HeteroQOff). qoff = off-lineage component (lrq3, lqi); qlin =
        # lineage component (lqi only, where the FULL Q is learned and the
        # policy is improved MPO/AWR-style instead of via PPO advantages).
        self.causal_scheme = causal_scheme
        self.qoff_model = qoff_network
        self.qoff_optimizer = (torch.optim.Adam(qoff_network.parameters(), lr=value_lr)
                               if qoff_network is not None else None)
        self.qlin_model = qlin_network
        self.qlin_optimizer = (torch.optim.Adam(qlin_network.parameters(), lr=value_lr)
                               if qlin_network is not None else None)

        self.lam = lam
        self.gam = gam
        self.causal_pg = causal_pg
        self.causal_rl = bool(causal_rl)
        # LVA: weight of the critic's auxiliary lineage-credit regression in
        # the value loss (value_loss + coef * aux_loss). Only read when the
        # value network was built with aux_head=True (scheme 'lva').
        self.causal_aux_coef = float(causal_aux_coef)
        self.buffer = TrajectoryBuffer(gam=gam, lam=lam,
                                       causal_scheme=causal_scheme,
                                       causal_pg=causal_pg,
                                       causal_rl=causal_rl,
                                       causal_beta=causal_beta,
                                       causal_mu=causal_mu,
                                       smdp_discount=smdp_discount)
        self.normalize_advantages = normalize_advantages
        self.normalize_returns = normalize_returns  # New parameter
        self.lr_schedule = lr_schedule  # New parameter
        self.kld_limit = kld_limit
        self.ent_bonus = ent_bonus
        self._ent_bonus_initial = ent_bonus  # for entropy annealing
        self._total_epochs = None  # set at start of train()
        self._current_epoch = 0

        self.previous_policy_loss = 0
        self.best_test_metric = float('-inf')  # Initialize the best test metric

        self.test_during_train = test_in_train
        self.eps = eps
        self.vf_coeff = vf_coeff

        # Potential-based reward shaping (gympn/potential.py; Ng, Harada &
        # Russell 1999). 0.0 (default) = disabled, byte-identical no-op --
        # matches this codebase's "safe floor" convention (cf_config=None,
        # causal_mu=0.0). See run_episode for the injection point and
        # potential.py's module docstring for the theorem + scope notes.
        self.phi_coef = float(phi_coef)
        self.phi_decay = float(phi_decay)
        # None (default) = uncapped (the original Phi); a small int caps each
        # place's own contribution before summing -- see potential.py's
        # topology_potential docstring for why (exogenous-arrival queue-depth
        # variance).
        self.phi_cap = phi_cap

        # G1 forked counterfactual preferences (gympn/counterfactual.py,
        # CAUSAL_LINEAGE_RETHINK.md §7.2). None => disabled (exact PPO floor:
        # no forks, no aux loss, nothing else changes). Keys: fork_prob,
        # reps, gate, lookahead, max_forks, coef, updates, beta.
        self.cf_config = cf_config
        # causal_scheme='cf' (measured counterfactual advantage) needs a fork
        # config; default one so it works without CLI wiring. Forks EVERY
        # decision (fork_prob is unused on this path) up to max_forks.
        if getattr(self, 'causal_scheme', None) == 'cf' and self.cf_config is None:
            self.cf_config = {'fork_prob': 1.0, 'reps': 1, 'gate': 0.0,
                              'lookahead': 8.0, 'max_forks': 200, 'beta': 0.0}
        self._cf_prefs = []
        self._cf_diag = []   # per-fork {'gap','se','passed'} mechanism telemetry
        self._cf_records = []  # decomposed mode: unresolved fork records

        # ls_hca (FORKFREE_LINEAGE_RETHINK.md Idea 1): the fitted hindsight
        # model hhat(a | reward-type r realized in the decision's lineage),
        # keyed (action_type, reward_type, z) -> probability. Starts empty ->
        # every CONTESTED term's correction factor is 0 (safe floor: pure-only
        # credit) until the first epoch's records are fit. _ls_hca_records
        # pools this epoch's (action_type, reward_type, z) triples, consumed
        # and cleared by _fit_ls_hca_hhat() at epoch end (one-epoch lag: never
        # fit and applied on the same batch).
        self._ls_hca_hhat = {}
        self._ls_hca_records = []
        # Stability guards for the 1-pi/hhat correction (run_episode's ls_hca
        # combine step) -- added after a real 30-epoch/10-seed s1 run
        # collapsed 0W/10L against lrq2 (several seeds to greedy_final=0.0).
        # Root cause: hhat is refit fresh each epoch from the PREVIOUS
        # epoch's policy and deliberately lags (kept off the current batch
        # for unbiasedness -- see _fit_ls_hca_hhat), while pi(a|s) is from
        # the CURRENT, possibly much-sharpened policy (s1's entropy collapses
        # ~1.0->0.5 within ~8 epochs). When hhat is small relative to a
        # sharpened pi, 1-pi/hhat is unbounded -- e.g. pi=0.5, hhat=0.01 gives
        # factor=-49, an uncapped multiplier on the reward contribution.
        # ls_hca_hhat_floor: statistically-meaningful minimum for a FITTED
        # hhat value (replaces the old 1e-9, which only guarded against
        # literal division by zero, not against small-but-nonzero hhat).
        # ls_hca_factor_clip: hard cap on |factor|, a pure safety net
        # independent of the floor/smoothing below.
        # ls_hca_smoothing_alpha: Laplace pseudo-count added when FITTING
        # hhat (_fit_ls_hca_hhat), guarding sparser (reward_type, z) buckets
        # on other envs even where the floor/clip above are not the binding
        # constraint. None of these three touch the fit/apply batch
        # separation that makes ls_hca unbiased -- they only bound how large
        # the correction can get, or smooth a noisy small-count estimate.
        self.ls_hca_hhat_floor = float(ls_hca_hhat_floor)
        self.ls_hca_factor_clip = float(ls_hca_factor_clip)
        self.ls_hca_smoothing_alpha = float(ls_hca_smoothing_alpha)
        # Opt-in diagnostic hook, default off -- see the combine step below.
        self.ls_hca_debug = False
        self._ls_hca_debug_log = []

        # State-conditional hhat (the "logistic model" the original design
        # doc sketched and the first implementation simplified away for a
        # flat, type-only table -- see causal-stability-suite memory,
        # "LS-HCA REVISITED": even after fixing the pit/hhat scale mismatch
        # (type-aggregated pi), a real residual anti-correlation between pit
        # and factor remained (r=-0.44), consistent with hhat still being a
        # state-MARGINALIZED average rather than state-conditional). For
        # each (reward_type, z) group with enough pooled records
        # (>= ls_hca_state_min_n), fit a tiny per-group multinomial logistic
        # regression mapping the marking-vector state features (same
        # featurization as _rudder_features, general across any A-E PN) to a
        # distribution over that group's observed action types -- refit
        # fresh every epoch from scratch (no continuity assumed across
        # epochs, matching the flat table's own fit/apply separation for
        # unbiasedness).
        #
        # ls_hca_flat_fallback: what to do for a (reward_type, z) group with
        # no fitted state model (below ls_hca_state_min_n, or no state
        # features configured). True = fall back to the flat Laplace-smoothed
        # table self._ls_hca_hhat (the original behaviour). DEFAULT FALSE,
        # i.e. treat the group exactly like an unseen key and apply the
        # cold-start-safe factor=0.
        #
        # Why the default is off, measured not assumed (_diag_ls_hca_step0.py
        # on s1, 8272 pooled records over 10 epochs): the flat table's whole
        # content is the marginal association between action type and z, and
        # that association is ABSENT -- P(start1|z=False)=0.655 vs
        # P(start1|z=True)=0.640, G=1.92, p=0.166, Cramer's V=0.015, and not
        # one of the 10 individual epochs significant either. The SAME data
        # shows a real state-CONDITIONAL association (held-out CE 0.4182 with
        # the z split vs 0.4297+-0.0002 for 200 z-shuffled placebos, below
        # the placebo minimum, permutation p=0.005). So I(A;Z) ~ 0 while
        # I(A;Z|X) > 0: the state masks the signal, the state-conditional
        # model finds it, and the flat table can only ever fit noise --
        # every factor it produces is spurious, applied at full strength to
        # a real reward. Falling back to factor=0 is strictly safer: it
        # forfeits nothing real and it is the same conservative floor
        # already used for genuinely unseen keys.
        self._ls_hca_state_node_types = []  # set by make_agent from metadata
        self._ls_hca_hhat_model = {}        # {(reward_type,z): (nn.Module, [a_type,...])}
        self.ls_hca_state_min_n = int(ls_hca_state_min_n)
        self.ls_hca_state_epochs = int(ls_hca_state_epochs)
        self.ls_hca_state_lr = float(ls_hca_state_lr)
        self.ls_hca_flat_fallback = bool(ls_hca_flat_fallback)

        # ls_hca_residual: parameterize hhat as a perturbation OF the policy,
        #     h(a|x,z) = softmax_a( log pi_type(a|x) + g(a|x,z) )
        # over the types enabled at x, with g the per-(reward_type, z) linear
        # model above, zero-initialized. DEFAULT TRUE.
        #
        # The estimator's validity rests on a null: if the action did not
        # influence the reward then h == pi and factor = 1 - pi/h = 0. In the
        # non-residual form h and pi are estimated INDEPENDENTLY -- a per-group
        # linear model on marking counts versus a GNN softmax -- so that null
        # holds only if two unrelated estimators happen to agree, and every
        # systematic disagreement becomes bias multiplied straight into a real
        # reward. Here the null is exact by construction: g == 0 gives
        # h == pi identically, whatever pi's own error is, so only the
        # RESIDUAL log-odds is ever estimated. Zero-init means training starts
        # AT the null and departs only as far as the data pushes it.
        #
        # Measured motivation (_diag_ls_hca_step0.py, held-out): the true
        # correction is far smaller than the one being applied -- ideal
        # median |factor| 0.057 vs 0.194 applied on s1 (3.4x), and 0.0067 vs
        # 0.0832 on i_mixed_credit (12.4x). Roughly 85-90% of the correction
        # magnitude reaching the rewards was estimator mismatch, not signal.
        #
        # Requires the captured per-decision pi vector (run_episode's
        # ls_hca_pi_type_vecs). Where that is missing the code falls back to
        # the independent form, so nothing crashes on older pooled records.
        self.ls_hca_residual = bool(ls_hca_residual)

        # ls_hca_state_l2: decoupled (AdamW) weight decay on the residual model
        # g. UNDER ls_hca_residual THIS IS A PRIOR CENTERED EXACTLY ON THE
        # NULL -- g == 0 means h == pi means factor == 0 -- so shrinking g is
        # literally "apply no correction unless the data insists". (With
        # ls_hca_residual off, g == 0 instead means a UNIFORM h, so the same
        # penalty encodes a much weaker and less principled prior; the knob is
        # still honoured there, but its interpretation does not carry over.)
        #
        # Motivated by what is left after the reparameterization: it removes
        # the pi-vs-h structural mismatch by construction, but the applied
        # correction still ran ~2.5-3.3x the held-out ideal, and the residue is
        # g's OWN estimation error -- g is fit in-sample over ls_hca_state_
        # epochs passes with no shrinkage, while the true effect is a median
        # ~0.015-0.06 nudge (_diag_ls_hca_step0.py).
        #
        # DEFAULT 0.0 (off), because that hypothesis was MEASURED AND REFUTED.
        # `_diag_ls_hca_l2_sweep.py` swept lambda over 0..3 through this exact
        # fit path on cached records: held-out CE is best at lambda=0 on BOTH
        # envs (s1 0.41455, monotonically worse from there, +0.020 nats by
        # lambda=3; i_mixed_credit flat to 5 decimals out to lambda=0.03, then
        # worse). g is not overfitting, so the |factor| reduction shrinkage
        # does buy (s1 median 0.118 -> 0.101 at lambda=3) is shrinking SIGNAL,
        # not noise. The knob stays because it is cheap and the conclusion is
        # env-specific, but turning it on needs a sweep saying so first.
        #
        # The remaining over-correction was then traced elsewhere
        # (_diag_ls_hca_consistency.py): the residual form pins h to pi only at
        # g == 0, and fits each (rtype, z) group INDEPENDENTLY, so the law of
        # total probability sum_z P(z|x) h(a|x,z) == pi(a|x) is violated
        # (median 3.4% of pi, systematically signed). Enforcing it cuts the
        # applied |factor| 4.46x, onto the measured ideal. Coupling the
        # z-groups at fit time -- not shrinking g -- is the real fix.
        self.ls_hca_state_l2 = float(ls_hca_state_l2)

        # ls_hca_consistent: model the LINEAGE-MEMBERSHIP probability
        # P(z | x, a) -- one binary head per action type per reward-type --
        # instead of modelling hhat directly, and recover
        #     h(a|x,z) = pi(a|x) P(z|x,a) / P(z|x),   P(z|x) = sum_a pi(a|x) P(z|x,a)
        # DEFAULT TRUE. Supersedes ls_hca_residual, which stays as the
        # fallback when this cannot be fit (and is itself still exact at the
        # null); set both False for the original independent-fit behaviour.
        #
        # This is the fix for the LAST known error source. The residual form
        # made the null exact but fitted each (rtype, z) group INDEPENDENTLY,
        # so nothing enforced the law of total probability
        #     sum_z P(z|x) h(a|x,z) = pi(a|x)
        # that any real hindsight distribution obeys. Measured violation on
        # held-out records (_diag_ls_hca_consistency.py): median 3.4% of pi,
        # systematically signed (mean +0.043), and enforcing the identity cut
        # the applied |factor| 4.46x (median 0.0403 -> 0.0090) onto the
        # independently measured ideal. Writing h as a genuine posterior makes
        # the identity hold identically -- sum_z P(z|x,a) = 1 by definition --
        # rather than approximately.
        #
        # Two consequences worth knowing. (1) pi CANCELS: the applied factor
        # reduces to 1 - P(z|x)/P(z|x,a), so the correction no longer depends
        # on the policy network's calibration at all, only on whether the
        # action shifts lineage membership -- which is the causal question the
        # estimator was always trying to ask. (2) The null is exact for a
        # stronger reason than before: it needs P(z|x,a) == P(z|x), not
        # agreement between two estimators of pi.
        self.ls_hca_consistent = bool(ls_hca_consistent)
        self._ls_hca_zmodel = {}    # rtype -> (nn.Module, [a_type, ...])

        # ls_hca_ratio: use HCA's STATE-conditional estimator for the CONTESTED
        # term -- weight each lineage reward by w = h/pi -- instead of the
        # return-conditional one, which ADDS the mean-zero advantage
        # (1 - pi/h). DEFAULT TRUE.
        #
        #   Q_d = sum_{PURE,     d in lineage} r*disc          (exact, unchanged)
        #       + sum_{CONTESTED, d in lineage} r*disc * w,  w = P(z|x,a)/P(z|x)
        #
        # Why this and not the difference form. Harutyunyan et al. give two
        # estimators: the return-conditional one is an ADVANTAGE (mean zero)
        # and the state-conditional one is a Q (return-scaled). We were
        # computing the advantage and handing it to buffer.finish(mode=
        # "replace") AS THE REWARD. Where PURE is nonempty that is survivable
        # -- the exact term supplies the scale and the correction rides on top
        # -- but on a PURE=set() env the entire reward becomes a mean-zero
        # object, the value net learns ~0, and PPO's advantage normalization
        # rescales the residual noise to unit variance. MEASURED on s1:
        # 1.45+-2.33 vs lrq2's 12.02, paired -10.57, 0W/5L, p=.001, with 3/5
        # seeds ending at exactly 0.0 -- BELOW random (9.85), the signature of
        # collapsing onto postpone, which "the correction is merely small"
        # never explained.
        #
        # Under the ratio form the null is w = 1, so Q_d is EXACTLY lrq2's
        # lineage credit. LS-HCA stops being a replacement that can
        # catastrophically underperform the baseline and becomes lrq2
        # reweighted by how much the action actually moved lineage membership
        # -- which is precisely the discrimination lrq2 lacks (it hands a
        # reward's full mass to all k of its lineage decisions equally).
        # Cold start is also w = 1 rather than the old form's zero credit.
        #
        # Two deliberate consequences. (1) Only decisions IN a reward's
        # lineage are paid; the old form credited every decision x reward pair
        # regardless of z. That drops a large amount of mean-zero noise mass,
        # but also forfeits HCA's ability to PENALIZE a high-pi decision that
        # failed to land in the lineage. (2) Expect s1 to land NEAR lrq2, not
        # above it -- the measured signal there is a median |factor| of 0.072,
        # so w sits near 1. Neutral-not-win is the honest prediction.
        #
        # ls_hca_weight_clip bounds w to [1/c, c]. It is NOT interchangeable
        # with ls_hca_factor_clip: w = 1/(1-factor), so clipping the factor at
        # +3 would imply w = -0.5, flipping the reward's sign.
        self.ls_hca_ratio = bool(ls_hca_ratio)
        self.ls_hca_weight_clip = float(ls_hca_weight_clip)

        # ls_hca_z_feature: WHICH conditioning variable the hindsight model
        # conditions on. 'delay' (DEFAULT) | 'depth' | 'share' | 'membership'.
        #
        # Why this exists. Every version up to now conditioned on binary
        # lineage MEMBERSHIP, and that is provably incapable of the job. For a
        # reward with k decisions in its lineage, all k have z=True: the
        # conditioning variable is CONSTANT on exactly the set the reweighting
        # needs to rank, so h(a|x,z) can separate those k only through x and a,
        # never through their differing roles in the lineage. This is
        # structural, not a weak fit. And k is large in practice -- on s1 it
        # averages 8.48 (median 7, max 27) and is NEVER 1 -- so lrq2 is handing
        # a reward's full mass to ~8 undiscriminated decisions, which is both
        # the opportunity and, very likely, the reason lrq2 LOSES to plain PPO
        # there (11.21 vs 13.60, paired -2.385, 1W/9L, p=.003). Measured
        # consequence of the binary variable: 73% of applied weights land
        # within 1% of w=1, i.e. indistinguishable from lrq2.
        #
        #   'delay'      reward time - decision time. Varies WITHIN a lineage.
        #   'depth'      hop distance through the token DAG. Varies within.
        #   'share'      1/k. Does NOT vary within a lineage -- it is a
        #                property of the reward, not the decision -- so it
        #                cannot rank lineage members. Provided for comparison
        #                only; expect it to behave like 'membership'.
        #   'membership' the previous binary indicator, for A/B.
        #
        # The statistic is bucketed into ls_hca_z_bins levels (level 0 is
        # reserved for "not in this reward's lineage"), so P(z|x,a) stays a
        # finite categorical and the consistent parameterization carries over
        # unchanged -- 'membership' is exactly the ls_hca_z_bins=1 case, which
        # is why the binary path is recovered rather than special-cased. Bin
        # edges are quantiles of the pooled statistic, refit each epoch
        # alongside the model and reused at apply time next epoch, matching the
        # existing fit/apply lag discipline.
        if ls_hca_z_feature not in ('delay', 'depth', 'share', 'membership'):
            raise ValueError(
                f"ls_hca_z_feature must be one of 'delay', 'depth', 'share', "
                f"'membership'; got {ls_hca_z_feature!r}")
        self.ls_hca_z_feature = ls_hca_z_feature
        self.ls_hca_z_bins = max(1, int(ls_hca_z_bins))
        self._ls_hca_z_edges = None     # quantile cut points, set at fit time

        # ls_hca_gate: restrict CONTESTED credit to decisions actually in the
        # reward's lineage. DEFAULT FALSE (ungated), which changes what the
        # estimator falls back to when the correction says nothing:
        #
        #   gated   (True) : null = lrq2   -- only lineage members are paid
        #   ungated (False): null = mc_q   -- every reward is paid, weighted
        #
        # Why the default flipped. Harutyunyan's state-conditional estimator
        # sums over ALL rewards weighted by h/pi; the hard lineage gate was
        # this project's addition, not the theory's. Measured on the stochastic
        # tier, that addition does not pay for itself -- comparing lrq2 against
        # its own lineage ABLATION mc_q (`_redistribute_mcq`, identical
        # estimator minus the lineage test):
        #     s1  mc_q 13.41  vs lrq2 11.21   (-2.20, 2W/8L, p=.015)
        #     s2  mc_q 89.59  vs lrq2 87.11   (-2.48, 5W/4L, p=.21)
        #     s3  mc_q 110.85 vs lrq2 112.02  (+1.17, 4W/6L, p=.41)
        # The restriction is significantly HARMFUL on one env and
        # indistinguishable on the other two; it never significantly helps.
        # And against plain PPO, mc_q never significantly loses (s1 -0.18
        # p=.55, s2 +10.27 p=.069, s3 +0.90 p=.57) whereas lrq2 loses s1
        # outright (-2.38, 1W/9L, p=.003). So an ungated null is on par with
        # PPO by construction and wins where the correction has signal, which
        # a gated null cannot be -- it inherits lrq2's s1 deficit.
        #
        # Lineage does not disappear: it enters through the CONDITIONING
        # VARIABLE (ls_hca_z_feature buckets membership plus delay/depth),
        # which is what it is actually informative for, rather than as a hard
        # 0/1 mask on who may be paid at all.
        #
        # Scope note: only the CONTESTED term is ungated. The PURE term stays
        # exact and lineage-gated, since a PURE reward-type is by construction
        # unreachable from any other decision point. So the null is exactly
        # mc_q only where PURE is empty (s1 and 8 of the 11 suite envs); on
        # mixed envs (i_mixed_credit, j_mixed_rework) it is exact-PURE plus
        # ungated-CONTESTED, a hybrid.
        self.ls_hca_gate = bool(ls_hca_gate)

    def act(self, state, return_logprob=False, deterministic=False):
        """Return an action for the given state using the policy model.

        Parameters
        ----------
        state : np.array
            The state of the environment.
        return_logprob : bool, optional
            Whether to return the log probability of choosing the chosen action.
        deterministic : bool, optional
            Whether to use a deterministic policy.

        """
        self.policy_model.eval()  # set model to evaluation mode
        # Rollout/eval only: no autograd graph is needed here (the stored logpis
        # are detached old-policy constants; the policy fit recomputes a fresh
        # forward on batches). Wrapping in no_grad avoids building and discarding
        # an autograd graph on every single environment step.
        with torch.no_grad():
            pi = self.policy_model(state)
            logpi = pi.log()

            if deterministic:
                action = torch.argmax(pi).item()  # Choose the action with the highest probability
            else:
                action = torch.multinomial(torch.exp(logpi.squeeze(1)), 1)[0]

        if return_logprob:
            if os.environ.get('GP_DEBUG_PPO', '0') == '1':
                try:
                    print(f"[PPO DEBUG] act: logpi shape = {tuple(logpi.shape)}")
                except Exception:
                    pass
            return action.item(), logpi[action].item(), logpi
        else:
            return action

    def value(self, state):
        """Return the predicted value for the given state using the value model.

        Parameters
        ----------
        state : np.array
            The state of the environment.

        """
        self.value_model.eval()  # set model to evaluation mode
        with torch.no_grad():  # disable gradient calculation
            return self.value_model(state)

    def train(self, env, episodes=10, epochs=1, max_episode_length=None, verbose=0, save_freq=1,
              logdir=None, batch_size=64, sort_states=False, test_env=None, test_freq=5, test_episodes=10,
              wandb_logger=None, num_workers=1, eval_seed=None):
        """Train the agent on env with optional testing during training.

        Parameters
        ----------
        env : environment
            The environment to train on.
        test_env : environment, optional
            The test environment for evaluation during training.
        test_freq : int, optional
            Frequency (in epochs) to run testing during training.
        eval_seed : int, optional
            Base seed pinning the evaluation scenarios (common random numbers).
            See :meth:`test_in_train`. None keeps the historical behaviour of
            drawing fresh eval scenarios from the live RNG stream.
        wandb_logger : WandBLogger, optional
            Logger for Weights & Biases integration.

        Returns
        -------
        history : dict
            Dictionary with statistics from training and testing.
        """
        # Pin the initial policy to the run seed alone. Parameters are created
        # lazily at the first forward pass, so this has to happen here rather
        # than at construction -- see gympn.seeding.seed_network_init for the
        # measurements that motivated it (arms of the same experiment were
        # starting from different initial policies, silently unmatching the
        # comparison).
        _agent_seed = getattr(self, 'agent_seed', None)
        if _agent_seed is not None:
            from gympn.seeding import seed_network_init
            seed_network_init(_agent_seed)

        tb_writer = None if logdir is None else SummaryWriter(log_dir=logdir)

        # Initialize learning rate schedulers if enabled
        # Cosine annealing helps convergence by gradually reducing LR over training
        policy_scheduler = None
        value_scheduler = None
        if self.lr_schedule:
            policy_scheduler = CosineAnnealingLR(
                self.policy_optimizer,
                T_max=epochs,
                eta_min=1e-6
            )
            value_scheduler = CosineAnnealingLR(
                self.value_optimizer,
                T_max=epochs,
                eta_min=1e-6
            )

        history = {'mean_returns': np.zeros(epochs),
                   'min_returns': np.zeros(epochs),
                   'max_returns': np.zeros(epochs),
                   'std_returns': np.zeros(epochs),
                   'mean_ep_lens': np.zeros(epochs),
                   'min_ep_lens': np.zeros(epochs),
                   'max_ep_lens': np.zeros(epochs),
                   'std_ep_lens': np.zeros(epochs),
                   'policy_updates': np.zeros(epochs),
                   'delta_policy_loss': np.zeros(epochs),
                   'policy_ent': np.zeros(epochs),
                   'policy_kld': np.zeros(epochs)}

        if test_env is not None:
            history.update({'test_mean_returns': np.zeros(epochs // test_freq + 1),
                            'test_min_returns': np.zeros(epochs // test_freq + 1),
                            'test_max_returns': np.zeros(epochs // test_freq + 1),
                            'test_std_returns': np.zeros(epochs // test_freq + 1)})

        self._total_epochs = epochs

        for i in range(epochs):
            # === ENTROPY ANNEALING ===
            # Linearly decay entropy bonus from initial value to 0 over training.
            # Early epochs: high entropy encourages exploration.
            # Late epochs: zero entropy prevents the "success catastrophe" where
            # the entropy bonus destroys an optimal policy when advantages ≈ 0.
            self._current_epoch = i
            self.ent_bonus = self._ent_bonus_initial * max(0.0, 1.0 - i / max(1, epochs - 1))

            self.buffer.clear()
            self._cf_prefs = []
            self._cf_diag = []
            self._cf_records = []
            self._ls_hca_records = []  # this epoch's pool; _ls_hca_hhat itself persists
            # Use parallel episode collection with dill (4-8x speedup on collection, 2-4x overall)
            # Dill can serialize lambda functions and complex objects like SimVar
            return_history = self.run_episodes(env, episodes=episodes, max_episode_length=max_episode_length,
                                               store=True, num_workers=num_workers)

            # === RUDDER: fit the return-predicting LSTM on this epoch's
            # trajectories (added per-episode in run_episode) ===
            if getattr(self, 'rudder_agent', None) is not None and self.rudder_agent.should_train():
                rudder_loss = self.rudder_agent.train(num_epochs=5)
                if wandb_logger and i % 10 == 0:
                    wandb_logger.log({'rudder/loss': rudder_loss}, step=i)
                self.rudder_agent.step_epoch()
                get_logger().info(f"  [RUDDER] Training loss: {rudder_loss:.4f}")

            # Standard per-batch advantage normalization (default ON via
            # self.normalize_advantages). Needed to learn low-margin tasks; the
            # drift it can cause near convergence is handled by best-checkpoint
            # restore. Toggle off only for ablations.
            normalize_adv_for_batch = self.normalize_advantages

            dataloader = self.buffer.get(normalize_advantages=normalize_adv_for_batch,
                                         normalize_returns=self.normalize_returns,
                                         batch_size=batch_size,
                                         sort=sort_states, drop_remainder=True)

            # LCV mechanism telemetry: the epoch's adaptive CV coefficient and
            # the fractional advantage-variance reduction it achieved (both
            # computed in buffer.get(); zeros for every other scheme).
            history.setdefault('cv_coef', np.zeros(epochs))
            history.setdefault('cv_var_reduction', np.zeros(epochs))
            history['cv_coef'][i] = getattr(self.buffer, 'last_cv_coef', 0.0)
            history['cv_var_reduction'][i] = getattr(self.buffer, 'last_cv_var_reduction', 0.0)

            # Route to appropriate training method:
            # - LQI (Q-native): fitted lineage-decomposed Q + advantage-
            #   weighted policy iteration (no PPO clip, no advantages from
            #   the buffer).
            # - Causal RL/PG: per-decision Q-sample advantages (LRQ family).
            # - Standard PPO: uses GAE advantages and discounted returns.
            if getattr(self, 'causal_scheme', None) == 'lqi' and self.causal_rl:
                policy_history = self._fit_lqi_models(dataloader)
            elif getattr(self, 'causal_scheme', None) == 'lcv' and self.causal_rl:
                # LCV: 100% standard PPO on the CV-adjusted advantages (the
                # adjustment happened in buffer.get()); plus the v_off state
                # head regressed on the measured off-lineage returns.
                policy_history = self._fit_policy_and_value_models(
                    dataloader, epochs=self.policy_updates)
                if getattr(self, 'qoff_model', None) is not None:
                    self._fit_voff_model(dataloader, epochs=self.value_updates)
            elif getattr(self, 'causal_scheme', None) == 'lva' and self.causal_rl:
                # LVA: 100% standard PPO (plain SMDP-GAE advantages from the
                # buffer, no CV, no credit advantages). The lineage enters only
                # inside _fit_value_model_step, as the critic aux head's
                # regression target (batch.qlin_target) — representation
                # shaping, policy gradient untouched.
                policy_history = self._fit_policy_and_value_models(
                    dataloader, epochs=self.policy_updates)
            elif self.causal_rl or self.causal_pg:
                policy_history = self._fit_causal_policy_models(dataloader, epochs=self.policy_updates)
            else:
                policy_history = self._fit_policy_and_value_models(dataloader, epochs=self.policy_updates)

            # G1: SNR-gated counterfactual preference aux pass (after the
            # PPO epochs; coefficient annealed to 0 over training = floor).
            # Scheme 'cf' reuses cf_config for its ADVANTAGE forks, not the
            # preference path, so it must skip this hook.
            if (getattr(self, 'cf_config', None) is not None
                    and getattr(self, 'causal_scheme', None) != 'cf'):
                cf_stats = self._fit_cf_preferences()
                # Mechanism telemetry: the paired SE and the gate-pass rate are
                # where the lineage-restriction claim lives (a tighter SE at
                # equal gap => more forks clear the gate for the same compute).
                diag = getattr(self, '_cf_diag', [])
                n_forks = len(diag)
                mean_se = float(np.mean([d['se'] for d in diag])) if diag else 0.0
                mean_gap = float(np.mean([abs(d['gap']) for d in diag])) if diag else 0.0
                pass_rate = float(np.mean([d['passed'] for d in diag])) if diag else 0.0
                for key in ('cf_prefs', 'cf_loss', 'cf_forks', 'cf_se',
                            'cf_gap', 'cf_pass_rate'):
                    history.setdefault(key, np.zeros(epochs))
                history['cf_prefs'][i] = cf_stats['n']
                history['cf_loss'][i] = cf_stats['loss']
                history['cf_forks'][i] = n_forks
                history['cf_se'][i] = mean_se
                history['cf_gap'][i] = mean_gap
                history['cf_pass_rate'][i] = pass_rate
                rs = getattr(self, '_cf_resolve_stats', None)
                if rs is not None:
                    # decomposed mode: the honest numbers are the regression's
                    # held-out R^2 (does the lineage feature actually predict
                    # the opportunity-cost channel?) and the resolved SE
                    history.setdefault('cf_r2', np.zeros(epochs))
                    history['cf_r2'][i] = (rs['r2'] if np.isfinite(rs['r2'])
                                           else 0.0)
                    history['cf_se'][i] = rs['se']
                    history['cf_pass_rate'][i] = rs['pass_rate']
                    get_logger().info(
                        f"  [CF] forks={n_forks} prefs={cf_stats['n']} "
                        f"mode={rs['mode']} r2={rs['r2']:.3f} "
                        f"nfit={rs.get('n_fit', 0)} "
                        f"pass={rs['pass_rate']:.0%} se={rs['se']:.4f} "
                        f"coef={cf_stats['coef']:.3f} loss={cf_stats['loss']:.4f}")
                else:
                    get_logger().info(
                        f"  [CF] forks={n_forks} prefs={cf_stats['n']} "
                        f"pass={pass_rate:.0%} |gap|={mean_gap:.3f} se={mean_se:.3f} "
                        f"coef={cf_stats['coef']:.3f} loss={cf_stats['loss']:.4f}")

            # ls_hca: refit the hindsight model from THIS epoch's pooled
            # records, for use next epoch (see Agent._fit_ls_hca_hhat). Runs
            # after the policy fit so the credits just trained on used the
            # PREVIOUS epoch's hhat (no same-batch fit-and-apply).
            if getattr(self, 'causal_scheme', None) == 'ls_hca':
                hhat_stats = self._fit_ls_hca_hhat()
                history.setdefault('ls_hca_records', np.zeros(epochs))
                history.setdefault('ls_hca_hhat_size', np.zeros(epochs))
                history['ls_hca_records'][i] = hhat_stats['n']
                history['ls_hca_hhat_size'][i] = len(self._ls_hca_hhat)
                get_logger().info(
                    f"  [LS-HCA] pooled_records={hhat_stats['n']} "
                    f"hhat_entries={len(self._ls_hca_hhat)} "
                    f"state_models={hhat_stats.get('n_models', 0)} "
                    f"zmodels={hhat_stats.get('n_zmodels', 0)}")

            # Update training history
            history['mean_returns'][i] = np.mean(return_history['returns'])
            history['min_returns'][i] = np.min(return_history['returns'])
            history['max_returns'][i] = np.max(return_history['returns'])
            history['std_returns'][i] = np.std(return_history['returns'])
            history['mean_ep_lens'][i] = np.mean(return_history['lengths'])
            history['min_ep_lens'][i] = np.min(return_history['lengths'])
            history['max_ep_lens'][i] = np.max(return_history['lengths'])
            history['std_ep_lens'][i] = np.std(return_history['lengths'])
            history['policy_updates'][i] = len(policy_history['loss'])

            # === CRITICAL FIX: Check if policy_history has loss data before accessing ===
            if len(policy_history['loss']) > 0:
                history['delta_policy_loss'][i] = policy_history['loss'][-1] - self.previous_policy_loss
                self.previous_policy_loss = policy_history['loss'][-1]
                history['policy_ent'][i] = policy_history['ent'][-1]
                history['policy_kld'][i] = policy_history['kld'][-1]
            else:
                # No batches were processed - warn and skip metrics
                get_logger().warning(
                    f"Epoch {i + 1}: No complete batches to process. "
                    f"Buffer size ({len(self.buffer)}) < batch_size ({batch_size}). "
                    f"Consider reducing batch_size or increasing episodes per epoch."
                )
                history['delta_policy_loss'][i] = 0.0
                history['policy_ent'][i] = 0.0
                history['policy_kld'][i] = 0.0

            # Test the agent during training
            if test_env is not None and (i + 1) % test_freq == 0:
                test_metrics = self.test_in_train(test_env, episodes=test_episodes,
                                                  max_episode_length=max_episode_length, logdir=logdir,
                                                  eval_seed=eval_seed)
                test_index = (i + 1) // test_freq - 1
                history['test_mean_returns'][test_index] = test_metrics['mean_returns']
                history['test_min_returns'][test_index] = test_metrics['min_returns']
                history['test_max_returns'][test_index] = test_metrics['max_returns']
                history['test_std_returns'][test_index] = test_metrics['std_returns']

                if tb_writer is not None:
                    tb_writer.add_scalar('test_mean_returns', test_metrics['mean_returns'], global_step=i)
                    tb_writer.add_scalar('test_min_returns', test_metrics['min_returns'], global_step=i)
                    tb_writer.add_scalar('test_max_returns', test_metrics['max_returns'], global_step=i)
                    tb_writer.add_scalar('test_std_returns', test_metrics['std_returns'], global_step=i)

                # Log test metrics to W&B if logger provided
                if wandb_logger is not None:
                    wandb_logger.log_test(
                        epoch=i,
                        mean_return=test_metrics['mean_returns'],
                        std_return=test_metrics['std_returns'],
                        min_return=test_metrics['min_returns'],
                        max_return=test_metrics['max_returns'],
                    )

            if test_env is None and logdir is not None and (
                    i + 1) % save_freq == 0:  # only save all the policies when no test in train is performed
                self.save_policy_weights(logdir + "/policy-" + str(i + 1) + ".h5")
                self.save_value_weights(logdir + "/value-" + str(i + 1) + ".h5")
                self.save_policy_network(logdir + "/network-" + str(i + 1) + ".pth")

            # Log epoch metrics
            metrics = TrainingMetrics(
                epoch=i + 1,
                mean_return=float(history['mean_returns'][i]),
                std_return=float(history['std_returns'][i]),
                mean_length=float(history['mean_ep_lens'][i]),
                policy_loss=float(history['delta_policy_loss'][i]) if not np.isnan(
                    history['delta_policy_loss'][i]) else None,
                kld=float(history['policy_kld'][i]) if not np.isnan(history['policy_kld'][i]) else None,
                entropy=float(history['policy_ent'][i]) if not np.isnan(history['policy_ent'][i]) else None,
            )
            get_logger().epoch_metrics(metrics)

            if tb_writer is not None:
                tb_writer.add_scalar('mean_returns', history['mean_returns'][i], global_step=i)
                tb_writer.add_scalar('min_returns', history['min_returns'][i], global_step=i)
                tb_writer.add_scalar('max_returns', history['max_returns'][i], global_step=i)
                tb_writer.add_scalar('std_returns', history['std_returns'][i], global_step=i)
                tb_writer.add_scalar('mean_ep_lens', history['mean_ep_lens'][i], global_step=i)
                tb_writer.add_scalar('min_ep_lens', history['min_ep_lens'][i], global_step=i)
                tb_writer.add_scalar('max_ep_lens', history['max_ep_lens'][i], global_step=i)
                tb_writer.add_scalar('std_ep_lens', history['std_ep_lens'][i], global_step=i)
                tb_writer.add_scalar('policy_updates', history['policy_updates'][i], global_step=i)
                tb_writer.add_scalar('delta_policy_loss', history['delta_policy_loss'][i], global_step=i)
                tb_writer.add_scalar('policy_ent', history['policy_ent'][i], global_step=i)
                tb_writer.add_scalar('policy_kld', history['policy_kld'][i], global_step=i)
                tb_writer.flush()
            # Log to W&B if logger provided
            if wandb_logger is not None:
                wandb_logger.log_epoch(
                    epoch=i,
                    mean_return=float(history['mean_returns'][i]),
                    std_return=float(history['std_returns'][i]),
                    policy_loss=float(history['delta_policy_loss'][i]) if not np.isnan(
                        history['delta_policy_loss'][i]) else None,
                    kld=float(history['policy_kld'][i]) if not np.isnan(history['policy_kld'][i]) else None,
                    entropy=float(history['policy_ent'][i]) if not np.isnan(history['policy_ent'][i]) else None,
                )

            if verbose > 0:
                print_status_bar(i, epochs, history, verbose=verbose)

            # Step learning rate schedulers if enabled
            if self.lr_schedule:
                policy_scheduler.step()
                value_scheduler.step()

        # === Best-checkpoint restore (early stopping) ===
        # The live policy can drift off the optimum after convergence (greedy eval
        # touches the optimum, then degrades — see INSTABILITY_ANALYSIS.md). When
        # deterministic eval ran during training and saved a best policy, reload it
        # so the returned agent holds the best policy found, not the last (possibly
        # degraded) one. No-op when no eval/checkpoint was produced.
        if test_env is not None and logdir is not None and self.best_test_metric > float('-inf'):
            best_path = os.path.join(logdir, "best_policy.pth")
            if os.path.exists(best_path):
                try:
                    self.policy_model = torch.load(best_path, weights_only=False)
                    get_logger().info(
                        f"Restored best policy (eval metric = {self.best_test_metric:.4f}) "
                        f"from {best_path}")
                except Exception as e:
                    get_logger().warning(f"Could not restore best policy from {best_path}: {e}")

        return history

    def run_episode(self, env, max_episode_length=None, buffer=None):
        """Run an episode and return total reward and episode length.

        OPTIMIZATION: Uses batched value predictions for 5-20x speedup.
        Value network is called every N steps on a batch of states instead of
        calling it on every single step.

        Parameters
        ----------
        env : environment
            The environment to interact with.
        max_episode_length : int, optional
            The maximum number of interactions before the episode ends.
        buffer : TrajectoryBuffer object, optional
            If included, it will store the whole rollout in the given buffer.

        Returns
        -------
        (total_reward, episode_length) : (float, int)
            The total nondiscounted reward obtained in this episode and the
            episode length. In causal RL mode, returns the environment's actual
            reward (info['pn_reward']) instead of step rewards.

        """
        state = env.reset()
        # NOTE: Do NOT flush causal traces here. env.reset() already flushes
        # the trace and then get_to_first_action() populates it with initial
        # evolution data (e.g. 'arrive' tokens and their parent-child links).
        # A second flush would wipe that data and break the causal chain for
        # tokens created during the initial evolution phase.

        done = False
        episode_length = 0
        total_reward = 0
        info = {'pn_reward': 0}  # Initialize info

        # === OPTIMIZATION: Batch value predictions ===
        # Instead of computing value every step, collect states and compute in batches
        states_batch = []
        actions_batch = []
        logprobs_batch = []
        logpis_batch = []
        rewards_batch = []
        times_batch = []
        value_batch_size = 8  # Compute values for 8 states at a time

        # RUDDER baseline: per-step (feature, reward) sequence for the LSTM
        # return predictor; consumed at episode end.
        rudder_on = getattr(self, 'rudder_agent', None) is not None and buffer is not None
        rudder_feats, rudder_rewards = [], []

        qoff_batch = []
        qoff_on = getattr(self, 'qoff_model', None) is not None and buffer is not None

        # G1 counterfactual forking: training rollouts only (buffer present),
        # capped per episode. Forks snapshot/restore env.pn, so the main
        # trajectory is untouched.
        cf_on = (getattr(self, 'cf_config', None) is not None and buffer is not None
                 and getattr(self, 'causal_scheme', None) not in ('lrq2c', 'cf'))
        cf_forks_done = 0

        # cf: measured counterfactual (COMA) advantage per decision via CRN forks
        # with lineage-restricted returns. Stored on the buffer, aligned 1:1 with
        # steps, and consumed by finish() for scheme 'cf' (bypasses the trace path).
        cf_adv_on = (getattr(self, 'causal_scheme', None) == 'cf'
                     and buffer is not None and getattr(self, 'cf_config', None) is not None)
        cf_adv_list, cf_base_list = [], []
        cf_adv_forks = 0

        # ls_hca: fork-free hindsight credit (FORKFREE_LINEAGE_RETHINK.md
        # Idea 1). No forks -- only needs pi(a|s) for the taken action at each
        # step (the trace-side PURE/CONTESTED split + hindsight combine happen
        # after the episode ends; see the ls_hca block below and
        # causal_traces._redistribute_ls_hca).
        ls_hca_on = (getattr(self, 'causal_scheme', None) == 'ls_hca' and buffer is not None)
        ls_hca_pi_taken = []
        ls_hca_pi_type_taken = []
        ls_hca_state_feats = []
        ls_hca_pi_type_vecs = []
        ep_values = []          # per-decision V(s), for scheme 'cgae'

        # lrq2c: PPO + lrq2 lineage advantage + sparse EXACT indirect correction
        # (CRN forks at foreclosure-gated decisions only). See
        # LINEAGE_SPARSE_CORRECTION.md. Uses its own fork path (router-gated,
        # applied per-decision) rather than the cfpk/cfp preference path above.
        lrq2c_on = (getattr(self, 'causal_scheme', None) == 'lrq2c'
                    and buffer is not None and getattr(self, 'cf_config', None) is not None)
        cf_indirect = []
        lrq2c_forks = 0
        if lrq2c_on and getattr(self, '_lrq2c_router', None) is None:
            try:
                from gympn.conflict_graph import conflicted_transition_ids
                self._lrq2c_router = conflicted_transition_ids(env.pn)
                get_logger().info(f"[lrq2c] foreclosure router: {self._lrq2c_router}")
            except Exception:
                self._lrq2c_router = set()

        while not done:
            action, logprob, logpis = self.act(state, return_logprob=True)

            # Decision time u_i = simulator clock BEFORE stepping (used for the
            # SMDP sojourn tau_t = u_{i+1} - u_i in causal time-discounting).
            pn = getattr(env, 'pn', None) or getattr(env, 'problem', None)
            decision_time = float(getattr(pn, 'clock', 0.0)) if pn is not None else 0.0

            if cf_on and cf_forks_done < self.cf_config['max_forks']:
                from gympn.counterfactual import maybe_fork
                forked, pref, diag = maybe_fork(self, env, state, action,
                                                logpis, self.cf_config)
                if forked:
                    cf_forks_done += 1
                if diag is not None:
                    self._cf_diag.append(diag)
                if pref is not None:
                    # decomposed mode yields fork RECORDS (resolved into
                    # preferences at epoch end, once the indirect-channel
                    # regression can be pooled); other modes yield preferences
                    if self.cf_config.get('decompose', False):
                        self._cf_records.append(pref)
                    else:
                        self._cf_prefs.append(pref)

            # lrq2c: fork ONLY at foreclosure-gated (structurally-contested) real
            # decisions; add the exact indirect gap to this decision's advantage
            # if it clears its own SNR gate (else the fork self-cancels — no new
            # bias). Non-gated decisions get pure lrq2 (correction 0).
            corr = 0.0
            if lrq2c_on:
                tr_id = None
                binds = getattr(env.pn, 'pn_actions', [])
                if 0 <= action < len(binds):
                    b = binds[action]
                    if not (isinstance(b[0], list) and b[0] == ['postpone']):
                        try:
                            tr_id = getattr(b[2], '_id', None)
                        except Exception:
                            tr_id = None
                if tr_id in self._lrq2c_router and lrq2c_forks < self.cf_config['max_forks']:
                    from gympn.counterfactual import maybe_fork
                    forked, rec, diag = maybe_fork(
                        self, env, state, action, logpis,
                        dict(self.cf_config, decompose=True))
                    if forked:
                        lrq2c_forks += 1
                    if rec is not None and 'gap_ind' in rec:
                        gi = float(rec['gap_ind'])
                        se = float(rec.get('se_ind', float('inf')))
                        if abs(gi) > self.cf_config['gate'] * se:
                            corr = gi
                    if diag is not None:
                        self._cf_diag.append(diag)
            if lrq2c_on:
                cf_indirect.append(corr)

            if cf_adv_on:
                a_cf, b_cf = None, None
                if cf_adv_forks < self.cf_config['max_forks']:
                    from gympn.counterfactual import fork_cf_advantage
                    a_cf, b_cf = fork_cf_advantage(self, env, state, action,
                                                   logpis, self.cf_config)
                    if a_cf is not None:
                        cf_adv_forks += 1
                cf_adv_list.append(a_cf)
                cf_base_list.append(b_cf)

            if ls_hca_on:
                acts_d = state.get('actions_dict') if isinstance(state, dict) else None
                n_act = len(acts_d) if acts_d else logpis.detach().reshape(-1).numel()
                probs = torch.softmax(logpis.detach().reshape(-1)[:n_act], dim=0)
                pit = float(probs[action]) if 0 <= action < probs.numel() else 1.0 / max(n_act, 1)
                ls_hca_pi_taken.append(pit)
                # Type-aggregated probability: hhat is fit at the ACTION-TYPE
                # level (pooled over however many concrete bindings of that
                # type occurred historically), but `pit` above is the mass on
                # ONE specific binding among however many of that type are
                # enabled THIS step -- a state-dependent, size-varying
                # denominator hhat never shares. Confirmed empirically
                # (causal-stability-suite memory, "LS-HCA REVISITED"): raw
                # pit vs hhat gives corr(pit,factor)=-0.91, i.e. the
                # correction punishes confident decisions simply because more
                # bindings were competing for probability mass, not because
                # anything about the decision was actually uncertain. Summing
                # pi over all enabled bindings sharing the taken action's
                # TYPE gives a quantity on the same scale as hhat.
                pit_type = pit
                share = 1.0 / max(n_act, 1)  # fallback: uniform-policy share
                pi_type_vec = None
                if acts_d and 0 <= action < len(acts_d) and probs.numel() >= len(acts_d):
                    taken_entry = acts_d[action]
                    taken_type_obj = taken_entry[2] if len(taken_entry) > 2 else None
                    taken_type = getattr(taken_type_obj, '_id', None)
                    if taken_type is not None:
                        # Type-aggregated pi over EVERY enabled type, not just
                        # the taken one. The residual parameterization (see
                        # ls_hca_residual in __init__) needs the whole vector:
                        # it defines h as a perturbation OF pi, so pi is the
                        # softmax base at both fit and apply time, and the
                        # types not taken are exactly the competitors that
                        # base has to normalize against.
                        by_type = {}
                        for i, entry in enumerate(acts_d):
                            if len(entry) <= 2 or i >= probs.numel():
                                continue
                            tid = getattr(entry[2], '_id', None)
                            if tid is None:
                                continue
                            by_type[tid] = by_type.get(tid, 0.0) + float(probs[i])
                        pi_type_vec = by_type or None
                        same_type_idx = [
                            i for i, entry in enumerate(acts_d)
                            if len(entry) > 2 and getattr(entry[2], '_id', None) == taken_type
                        ]
                        pit_type = sum(float(probs[i]) for i in same_type_idx if i < probs.numel())
                        # n_type/N: the taken type's share of ALL enabled
                        # bindings this decision -- exactly what a UNIFORM
                        # policy would give pit_type, i.e. a direct
                        # calibration reference for the state-conditional
                        # model. Confirmed empirically (causal-stability-
                        # suite memory, "LS-HCA REVISITED"): a trained policy
                        # sees systematically SMALLER action sets than a
                        # random one (median 5 vs 7, p=6e-17 on s1) -- so
                        # n_type/N drifts over training for structural
                        # reasons the marking-vector features alone don't
                        # make explicit; feeding it directly lets the model
                        # calibrate against it instead of confusing "fewer
                        # competitors" with "genuinely more confident".
                        n_type = len(same_type_idx)
                        share = n_type / max(len(acts_d), 1)
                ls_hca_pi_type_taken.append(pit_type)
                ls_hca_pi_type_vecs.append(pi_type_vec)
                ls_hca_state_feats.append(
                    self._rudder_features(state, node_types=self._ls_hca_state_node_types)
                    + [share])

            # Collect for batch processing
            states_batch.append(state)
            actions_batch.append(action)
            logprobs_batch.append(logprob)
            logpis_batch.append(logpis)
            times_batch.append(decision_time)
            if rudder_on:
                rudder_feats.append(self._rudder_features(state))
            if qoff_on:
                # Rollout-time auxiliary prediction (old parameters, frozen at
                # collection): q_off(s, a_taken) for lrq3/lqi; the state-only
                # centering v_off(s) for lcv (scalar HeteroCritic head).
                try:
                    with torch.no_grad():
                        qv = self.qoff_model(state).reshape(-1)
                    if getattr(self, 'causal_scheme', None) == 'lcv':
                        qoff_batch.append(float(qv[0]) if qv.numel() else 0.0)
                    else:
                        qoff_batch.append(float(qv[action]) if action < qv.numel() else 0.0)
                except Exception:
                    qoff_batch.append(0.0)

            # Potential-based reward shaping (gympn/potential.py): read Phi(s)
            # from the MAIN trajectory's pn before stepping -- `pn` was
            # captured above before any fork block (cf_on/lrq2c_on/cf_adv_on)
            # could snapshot/restore env.pn, so this is never a forked branch.
            phi_s = (topology_potential(pn, decay=self.phi_decay, cap=self.phi_cap)
                    if self.phi_coef and pn is not None else 0.0)

            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward   # RAW reward -- eval/logging read info['pn_reward']
                                     # (independent accumulator), so this is unaffected by
                                     # shaping either way; kept raw here for clarity.

            if self.phi_coef and pn is not None:
                # Terminal-Phi convention: the finite-horizon telescoping
                # requires Phi(terminal) := 0, applied at the LAST step this
                # episode will ever store -- the same boolean condition the
                # break check below uses (episode_length here is PRE-
                # increment, hence `+ 1`), so this can never diverge from the
                # buffer's own tau[-1]:=0 / dones_ep[-1]=True convention.
                is_last_step = done or (max_episode_length is not None
                                        and episode_length + 1 > max_episode_length)
                tau = max(0.0, float(getattr(pn, 'clock', decision_time)) - decision_time)
                disc = math.exp(-self.buffer.causal_beta * tau)
                phi_next = 0.0 if is_last_step else topology_potential(pn, decay=self.phi_decay, cap=self.phi_cap)
                reward = reward + self.phi_coef * (disc * phi_next - phi_s)

            rewards_batch.append(reward)
            if rudder_on:
                rudder_rewards.append(float(reward))

            episode_length += 1

            # Compute values in batch every N steps or at episode end
            if len(states_batch) >= value_batch_size or done:
                values = self._compute_batch_values(states_batch, env)

                # Store all buffered transitions
                if buffer is not None:
                    for i, (s, a, lp, lpis, r, tm) in enumerate(zip(
                            states_batch, actions_batch, logprobs_batch,
                            logpis_batch, rewards_batch, times_batch)):
                        buffer.store(s, a, r, lp, values[i], lpis,
                                     token_ids=None, time=tm,
                                     qoff=(qoff_batch[i] if qoff_on else None))
                        ep_values.append(float(values[i]))

                # Clear batches for next iteration
                states_batch = []
                actions_batch = []
                logprobs_batch = []
                logpis_batch = []
                rewards_batch = []
                times_batch = []
                qoff_batch = []

            if max_episode_length is not None and episode_length > max_episode_length:
                break
            state = next_state

        if buffer is not None:
            if rudder_on and rudder_feats:
                # RUDDER baseline: add this episode to the LSTM's training set,
                # redistribute its rewards with the CURRENT predictor (early
                # epochs => near-uniform, standard RUDDER warm-up behaviour;
                # 'contribution' conserves the episode return exactly), and
                # feed the redistributed rewards through the ORDINARY GAE path.
                feats_np = np.asarray(rudder_feats, dtype=np.float32)
                rews_np = np.asarray(rudder_rewards, dtype=np.float32)
                self.rudder_agent.add_trajectory(
                    states=feats_np, actions=None, rewards=rews_np,
                    episode_return=float(rews_np.sum()))
                try:
                    red = self.rudder_agent.redistribute_rewards(feats_np, rews_np)
                    buffer.finish(credits=[float(x) for x in red],
                                  mode="replace_rewards")
                except Exception as e:
                    get_logger().warning(f"[RUDDER] redistribution failed ({e}); "
                                         f"falling back to raw rewards")
                    buffer.finish(credits=None)
            elif (self.causal_rl and 'eligibility_credits' in info
                  and info['eligibility_credits'] is not None):
                # NOTE the self.causal_rl guard: the ENV may record causal
                # traces while the AGENT stays on the standard SMDP-GAE path
                # (scheme 'cfpl' does exactly this — it needs the lineage DAG
                # for its forked counterfactual returns, but its policy
                # gradient must remain plain PPO). Behaviour-preserving for
                # every other method, where agent.causal_rl == env.causal_rl.
                # Diagnostic: if environment signals debugging, print causal trace stats
                try:
                    pn = getattr(env, 'pn', None) or getattr(env, 'problem', None)
                    if pn is not None and getattr(pn, '_debugging', False) and pn.causal_rl:
                        ct = pn.causal_trace
                        tok_count, trans_count = ct.stats()
                        # compute a quick credit sample to check sizes
                        try:
                            sample_cr = ct.redistribute_rewards(scheme='lrq')
                        except Exception as e:
                            sample_cr = None
                            import warnings
                            warnings.warn(
                                f"[CAUSAL-DIAG] tokens={tok_count}, transitions={trans_count}, ep_steps={episode_length}, redis_len={len(sample_cr) if sample_cr is not None else 'ERR'}, redis_sum={sum(sample_cr) if sample_cr is not None else 'ERR'}")
                except Exception:
                    pass
                if lrq2c_on:
                    # per-decision indirect corrections, aligned 1:1 with steps;
                    # finish() adds them to the lrq2 credits for scheme 'lrq2c'.
                    buffer._lrq2c_indirect = cf_indirect
                if cf_adv_on:
                    # per-decision measured counterfactual advantage + baseline,
                    # aligned 1:1 with steps; finish() uses them for scheme 'cf'.
                    buffer._cf_adv = cf_adv_list
                    buffer._cf_base = cf_base_list
                if ls_hca_on:
                    # Combine the trace's exact PURE credit with the CONTESTED
                    # hindsight correction here, where both ingredients this
                    # trace object cannot see on its own -- pi(a|s) for the
                    # taken action (ls_hca_pi_taken, this loop) and the fitted
                    # hhat table (self._ls_hca_hhat, previous epoch) -- are
                    # available. New (action_type, reward_type, z) records are
                    # pooled into self._ls_hca_records for the NEXT epoch's
                    # refit (_fit_ls_hca_hhat), never this one's.
                    ct = info['eligibility_credits']
                    pure = ct.redistribute_rewards(scheme='ls_hca',
                                                   beta=self.buffer.causal_beta)
                    pending = getattr(ct, '_ls_hca_pending', None) or [[] for _ in pure]
                    buffer._ls_hca_postpone = [
                        bool(isinstance(getattr(act.get('transition'), '_id', None), str)
                             and getattr(act.get('transition'), '_id').startswith('postpone_'))
                        for act in ct.transition_history.get_action_transitions()
                    ]
                    hhat = self._ls_hca_hhat
                    combined = list(pure)
                    for t, items in enumerate(pending):
                        if t >= len(ls_hca_pi_type_taken):
                            continue
                        # Type-aggregated pi (see the capture site above) --
                        # comparable in scale to hhat, which is fit at the
                        # action-TYPE level too.
                        pit = ls_hca_pi_type_taken[t]
                        state_feat = (ls_hca_state_feats[t] if t < len(ls_hca_state_feats)
                                     else None)
                        pi_vec = (ls_hca_pi_type_vecs[t] if t < len(ls_hca_pi_type_vecs)
                                  else None)
                        ratio_mode = getattr(self, 'ls_hca_ratio', True)
                        for entry in items:
                            a_type, rtype, z, contrib = entry[:4]
                            # Entries carry all three candidate conditioning
                            # statistics (delay, depth, share); pick the
                            # configured one and bucket it into a level. Older
                            # 4-tuples (no statistics) degrade to the binary
                            # membership variable.
                            delay, depth, share = (entry[4:7] if len(entry) >= 7
                                                   else (None, None, None))
                            zstat = self._ls_hca_zstat(z, delay, depth, share)
                            lvl = self._ls_hca_level(zstat, z=z)
                            key = (a_type, rtype, lvl)
                            h = self._ls_hca_predict_h(a_type, rtype, lvl, state_feat,
                                                       hhat, pi_type_vec=pi_vec)
                            if ratio_mode:
                                # HCA's OTHER estimator (see ls_hca_ratio in
                                # __init__): weight the lineage-gated reward by
                                # w = h/pi instead of adding the mean-zero
                                # advantage (1 - pi/h) to it. Only decisions
                                # actually IN this reward's lineage are paid;
                                # the z=False entries still fall through to the
                                # record pool below, because fitting P(z|x,a)
                                # needs both outcomes as labels.
                                if z or not getattr(self, 'ls_hca_gate', False):
                                    if h is not None and pit > 0:
                                        w = h / pit
                                    else:
                                        # Cold start (no model yet) is w = 1,
                                        # i.e. exactly lrq2's credit -- start AT
                                        # the working baseline and depart only
                                        # as the fit earns it. The old form's
                                        # cold start was factor = 0, i.e. NO
                                        # credit, which on a PURE=set() env is
                                        # no learning signal at all.
                                        w = 1.0
                                    # Clip w directly, NOT the factor: the two
                                    # are related by w = 1/(1-factor), so the
                                    # factor clip maps to nonsense here (a
                                    # factor of +3 would give w = -0.5, a
                                    # sign-flipped reward).
                                    wc = getattr(self, 'ls_hca_weight_clip', 10.0)
                                    combined[t] += contrib * max(1.0 / wc, min(wc, w))
                                self._ls_hca_records.append(
                                    (a_type, rtype, lvl, state_feat, pi_vec, zstat))
                                if getattr(self, 'ls_hca_debug', False):
                                    self._ls_hca_debug_log.append(
                                        (float(pit), h,
                                         float(h / pit) if (h is not None and pit > 0) else 1.0,
                                         key))
                                continue
                            if h is not None:
                                # The floor guards an h estimated INDEPENDENTLY
                                # of pi, where a small h and a sharpened pi can
                                # collide and blow the ratio up. Under the
                                # residual parameterization that failure mode
                                # cannot occur -- h is built FROM pi, so a
                                # small pi implies a correspondingly small h
                                # and pi/h stays O(1) -- and applying the floor
                                # anyway would DESTROY the exactness guarantee:
                                # with h == pi == 0.03 < floor, flooring gives
                                # factor = 1 - 0.03/0.05 = 0.4 instead of 0.
                                # So the floor is skipped exactly when it is
                                # both unnecessary and harmful.
                                if not ((getattr(self, 'ls_hca_residual', True)
                                         or getattr(self, 'ls_hca_consistent', True))
                                        and pi_vec):
                                    h = max(h, getattr(self, 'ls_hca_hhat_floor', 0.05))
                                raw_factor = 1.0 - pit / h
                            else:
                                raw_factor = 0.0  # unseen key: unchanged cold-start floor
                            clip = getattr(self, 'ls_hca_factor_clip', 3.0)
                            factor = max(-clip, min(clip, raw_factor))
                            combined[t] += contrib * factor
                            self._ls_hca_records.append(
                                (a_type, rtype, lvl, state_feat, pi_vec, zstat))
                            # Opt-in diagnostic hook (default off, zero cost
                            # otherwise): confirms/refutes whether `factor`
                            # is systematically most negative exactly when
                            # pit is high -- see causal-stability-suite
                            # memory, "LS-HCA REVISITED" hypothesis 1.
                            if getattr(self, 'ls_hca_debug', False):
                                self._ls_hca_debug_log.append(
                                    (float(pit), h, float(factor), key))
                    buffer.finish(credits=combined, mode="replace")
                elif getattr(self, 'causal_scheme', None) == 'cfgae':
                    # cfgae consumes the TRACE OBJECT (it needs the component
                    # partition and the per-step reward attribution, not a
                    # credit vector); data.py's finish does the filtered GAE.
                    buffer.finish(credits=info['eligibility_credits'],
                                  mode="replace")
                elif getattr(self, 'causal_scheme', None) in ('cgae', 'cgae_flow'):
                    # CGAE needs the critic's V(s) as well as the trace: the
                    # recursion bootstraps through it at causal depth, so the
                    # values collected during this episode are handed in here
                    # (the trace object never sees the value model).
                    ct = info['eligibility_credits']
                    q = ct.redistribute_rewards(
                        scheme=self.causal_scheme, beta=self.buffer.causal_beta,
                        values=ep_values, lam=getattr(self, 'lam', 0.95))
                    buffer.finish(credits=q, mode="replace")
                else:
                    buffer.finish(credits=info['eligibility_credits'], mode="replace")
            else:
                buffer.finish(credits=None)

        # Return actual environment reward (info['pn_reward']) which contains causal RL credits
        # In causal mode, step rewards are 0, so total_reward would be 0
        # info['pn_reward'] contains the true accumulated reward from causal redistribution
        actual_reward = info.get('pn_reward', total_reward)
        return actual_reward, episode_length

    def _compute_batch_values(self, states_list, env):
        """Compute values for a batch of states efficiently.

        OPTIMIZATION: Uses true PyTorch batching on heterogeneous graphs.
        Provides 2-5x speedup compared to individual forward passes.

        Parameters
        ----------
        states_list : list
            List of state observations from the environment
        env : environment
            The environment (for strategy-based value functions)

        Returns
        -------
        values : list
            List of scalar value predictions
        """
        if len(states_list) == 0:
            return []

        if self.value_model is None:
            return [0] * len(states_list)

        if isinstance(self.value_model, str):
            # Strategy-based value function - must call per step (no batching possible)
            return [env.value(strategy=self.value_model, gamma=self.gam)
                    for _ in states_list]

        # === OPTIMIZED: Use torch_geometric batching for heterogeneous graphs ===
        # This is much faster than looping through individual states
        self.value_model.eval()
        with torch.no_grad():
            try:
                # Try to use torch_geometric batching if states are graph objects
                from torch_geometric.data import HeteroData, Batch

                # Check if states are HeteroData objects
                if states_list and isinstance(states_list[0], dict) and 'graph' in states_list[0]:
                    # Extract graph objects and batch them
                    graphs = [s['graph'] for s in states_list]

                    if isinstance(graphs[0], HeteroData):
                        # Batch heterogeneous graphs
                        batched_graph = Batch.from_data_list(graphs)

                        # Single forward pass on batched graph
                        batch_values = self.value_model(batched_graph)

                        # Extract per-graph values
                        if isinstance(batch_values, torch.Tensor):
                            # Values should have shape [num_graphs] or [num_graphs, 1]
                            if batch_values.dim() > 1:
                                values = batch_values[:, 0].tolist() if batch_values.size(
                                    1) == 1 else batch_values.tolist()
                            else:
                                values = batch_values.tolist()
                            return values
            except Exception as e:
                # Fall back to individual computation if batching fails
                import warnings
                warnings.warn(f"Batching failed ({e}), falling back to sequential computation")

        # Fall back: compute individually (slower but always works)
        self.value_model.eval()
        with torch.no_grad():
            values = []
            for state in states_list:
                value = self.value_model(state)
                # Handle different value output shapes
                if isinstance(value, torch.Tensor):
                    value = value.squeeze().item() if value.numel() == 1 else value
                values.append(value)
            return values

    def run_episodes(self, env, episodes=100, tot_steps=None, max_episode_length=None, store=False, num_workers=None):
        """Run several episodes, store interaction in buffer, and return history.

        OPTIMIZATION: Supports parallel episode collection using multiprocessing.
        With num_workers > 1, episodes are collected in parallel across multiple CPU cores.
        This provides 4-8x speedup on episode collection (2-4x overall).

        Parameters
        ----------
        env : environment
            The environment to interact with.
        episodes : int, optional
            The number of episodes to perform.
        tot_steps : int, optional
            The total number of steps to perform across all episodes, if episodes is None.
        max_episode_length : int, optional
            The maximum number of steps before the episode is terminated.
        store : bool, optional
            Whether or not to store the rollout in self.buffer.
        num_workers : int, optional
            Number of parallel workers. If None, defaults to sequential.
            If > 1, uses multiprocessing.Pool for parallel collection.

        Returns
        -------
        history : dict
            Dictionary which contains information from the runs.

        """
        import copy
        import os

        history = {'returns': np.zeros(episodes),
                   'lengths': np.zeros(episodes)}

        # Determine number of workers
        if num_workers is None:
            num_workers = 1
        else:
            num_workers = min(num_workers, episodes, os.cpu_count() or 1)

        if num_workers <= 1 or episodes < 2:
            # Fall back to sequential for small episode counts or num_workers=1
            for i in range(episodes):
                R, L = self.run_episode(env, max_episode_length=max_episode_length,
                                        buffer=self.buffer if store else None)
                history['returns'][i] = R
                history['lengths'][i] = L
        else:
            # Parallel episode collection using dill for serialization
            # Dill can handle lambda functions and complex objects like SimVar
            try:
                import dill
                import multiprocessing

                # Create environment copies for each worker
                env_copies = [copy.deepcopy(env) for _ in range(num_workers)]

                # Prepare arguments for workers
                worker_args = [(self, env_copies[i % num_workers], max_episode_length)
                               for i in range(episodes)]

                # Use spawn context with dill for robust serialization
                ctx = multiprocessing.get_context('spawn')

                # Create a custom Pool that uses dill for pickling
                # When dill is imported, it automatically patches pickle to use dill's methods
                with ctx.Pool(processes=num_workers) as pool:
                    results = pool.map(_run_episode_worker, worker_args)

                # Aggregate results
                for i, (R, L) in enumerate(results):
                    history['returns'][i] = R
                    history['lengths'][i] = L

            except Exception as e:
                # Silently fall back to sequential if parallel fails
                for i in range(episodes):
                    R, L = self.run_episode(env, max_episode_length=max_episode_length,
                                            buffer=self.buffer if store else None)
                    history['returns'][i] = R
                    history['lengths'][i] = L

        return history

    def _fit_policy_model(self, dataloader, logpis, epochs=1):
        """Fit policy model using data from dataset.

        Parameters
        ----------
        dataloader : DataLoader
            The data loader for the dataset.
        logpis : list of Tensors
            The log probabilities of the actions taken in the dataset.
        epochs : int, optional
            The number of epochs to train the policy model.
        Returns

        -------
        dict
            Dictionary with loss, KLD, and entropy history for each epoch.

        """
        history = {'loss': [], 'kld': [], 'ent': []}

        for epoch in range(epochs):
            start = 0
            loss, kld, ent, batches = 0, 0, 0, 0

            for i, batch in enumerate(dataloader):
                lp = logpis[start:start + len(batch)]
                start += len(batch)
                batch_loss, batch_kld, batch_ent = self._fit_policy_model_step(batch, lp)
                loss += batch_loss
                kld += batch_kld
                ent += batch_ent
                batches += 1

            if batches == 0:
                get_logger().no_batches_warning()
                continue

            avg_loss = loss / batches
            avg_kld = kld / batches
            avg_ent = ent / batches
            history['loss'].append(avg_loss)
            history['kld'].append(avg_kld)
            history['ent'].append(avg_ent)

        return {k: np.array(v) for k, v in history.items()}

    def _fit_policy_model_step(self, batch, logpis):
        """Fit policy model on one batch of data.

        Parameters
        ----------
        batch : DataBatch
            The batch of data containing states, actions, advantages, etc.
        logpis : list of Tensors
            The log probabilities of the actions taken in the dataset.
        Returns
        -------
        loss : float
            The loss value for the policy model.
        kld : float
            The Kullback-Leibler divergence between the new and old policies.
        ent : float
            The entropy of the policy distribution.
        """
        self.policy_model.train()  # set model to training mode
        self.policy_optimizer.zero_grad()  # zero out gradients

        # Save the initial weights
        # initial_weights = {name: param.clone() for name, param in self.policy_model.named_parameters()}

        indexes = batch['a_transition'].batch.data
        states = batch
        actions = torch.tensor(batch.y)
        logprobs = batch.logprobs.clone()
        advantages = batch.advantage.clone()

        epsilon = 1e-7
        new_probs = self.policy_model(states)
        new_logpis = (new_probs + epsilon).log()

        # new_logprobs contains, for each unique index in indexes, the value in the slice of logpis corresponding
        # to the current index in indexes with index action[index]
        new_logprobs = torch.stack(
            [new_logpis[indexes == index][actions[index]] for index in indexes.unique()]).squeeze(1)

        # Calculate batch size
        batch_size = len(indexes.unique())

        # Compute normalized entropy
        ent = -torch.sum(new_probs * new_logpis) / batch_size

        # Compute normalized KLD
        logpis = torch.cat(logpis, dim=0)
        kld = torch.sum(new_probs * (new_logpis - logpis)) / batch_size
        loss = torch.mean(self.policy_loss(new_logprobs, logprobs, advantages)) - self.ent_bonus * ent

        try:
            # No second backward runs on this graph (each batch recomputes a
            # fresh forward), so retain_graph is unnecessary; dropping it frees
            # the activation graph immediately.
            loss.backward()  # compute gradients
        except Exception as e:
            print("Invalid loss", e)

        # Clip gradients for stability - critical for PPO
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        # Debug: print gradient norms per parameter if requested
        try:
            if os.environ.get('GP_DEBUG_PPO', '0') == '1':
                for name, param in self.policy_model.named_parameters():
                    if param.grad is not None:
                        try:
                            gnorm = float(torch.norm(param.grad).item())
                        except Exception:
                            gnorm = None
                        print(f"[PPO DEBUG] grad_norm policy {name}: {gnorm}")
        except Exception:
            pass

        self.policy_optimizer.step()

        try:
            if os.environ.get('GP_DEBUG_PPO', '0') == '1':
                print(f"KLD divergence: {kld.item():.6f} ent: {ent.item():.6f}")
        except Exception:
            pass
        return loss.item(), kld.item(), ent.item()

    # Assuming `model` is your PyTorch model
    def check_gradient_norms(self, model):
        """Check and print the gradient norms for each parameter in the model.

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch model to check gradients for.
        """
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                print(f"Gradient norm for {name}: {grad_norm}")
            else:
                print(f"No gradient for {name}")

    def load_policy_weights(self, filename):
        """Load weights from filename into the policy model.

        Parameters
        ----------
        filename : str
            The path to the file from which the model weights will be loaded.
        """
        self.policy_model.load_weights(filename)

    def save_policy_weights(self, filename):
        """Save the current weights in the policy model to filename.

        Parameters
        ----------
        filename : str
            The path to the file where the model weights will be saved.
        """
        self.policy_model.save_weights(filename)

    # ==================================================================
    # LQI: Lineage-Q Iteration (Q-native consumer of the trace credits)
    # ==================================================================
    # Motivation (see CAUSAL_LQI_QNATIVE.md): PPO's clip + multi-epoch reuse
    # amplifies small-but-CONSISTENT advantages to epsilon-sized policy moves
    # -> premature commitment (measured on s1); LRQ-v3's cold-start race
    # corrupted advantages while its head was untrained. LQI removes the
    # policy-gradient advantage entirely: fit the decomposed Q from trace
    # targets FIRST each epoch, then improve the policy by advantage-weighted
    # regression (AWR): maximize E[w * log pi(a|s)], w = exp(A_std / TAU)
    # clipped at W_MAX, A = q(s,a) - mean_{a' available} q(s,a'). The
    # weighted-BC form is KL-regularized policy iteration - proportionate
    # moves, no clip saturation, and an untrained Q merely yields ~uniform
    # weights (harmless warm-up) instead of corrupted gradients.
    _LQI_TAU = 1.0        # temperature on per-batch STANDARDIZED advantages
    _LQI_WMAX = 20.0      # AWR weight clip

    def _fit_lqi_models(self, dataloader):
        history = {'loss': [], 'kld': [], 'ent': [], 'policy_core_loss': [],
                   'value_loss': [], 'qlin_loss': [], 'qoff_loss': []}

        # 1. Q heads first (fresh Q before the policy is weighted by it).
        #    q_lin target = full mc_q sample - off target (= lrq2 credit).
        qlin_hist = self._fit_qhead(
            self.qlin_model, self.qlin_optimizer, dataloader,
            epochs=self.value_updates,
            get_target=lambda b: b.value - b.qoff_target)
        qoff_hist = self._fit_qhead(
            self.qoff_model, self.qoff_optimizer, dataloader,
            epochs=self.value_updates,
            get_target=lambda b: b.qoff_target)
        history['qlin_loss'] = qlin_hist['loss']
        history['qoff_loss'] = qoff_hist['loss']

        # 2. AWR policy epochs.
        for epoch in range(self.policy_updates):
            loss_acc = kld_acc = ent_acc = core_acc = 0.0
            batches = 0
            for batch in dataloader:
                if not hasattr(batch, 'qoff_target'):
                    continue
                b_loss, b_kld, b_ent, b_core = self._fit_lqi_policy_step(batch)
                loss_acc += b_loss
                kld_acc += b_kld
                ent_acc += b_ent
                core_acc += b_core
                batches += 1
            if batches == 0:
                get_logger().no_batches_warning()
                continue
            history['loss'].append(loss_acc / batches)
            history['kld'].append(kld_acc / batches)
            history['ent'].append(ent_acc / batches)
            history['policy_core_loss'].append(core_acc / batches)
            if self.kld_limit is not None and history['kld'][-1] > self.kld_limit:
                break

        return {k: np.array(v) for k, v in history.items()}

    def _fit_qhead(self, model, optimizer, dataloader, epochs, get_target):
        """Generic per-action-node regression: select the taken action's node
        output per sample and MSE against get_target(batch)."""
        history = {'loss': []}
        if model is None:
            return history
        for epoch in range(epochs):
            loss_acc, batches = 0.0, 0
            for batch in dataloader:
                if not hasattr(batch, 'qoff_target'):
                    continue
                model.train()
                out = model(batch)
                if out.dim() == 2 and out.size(-1) == 1:
                    out = out.squeeze(-1)
                sel, tgt = self._select_taken_nodes(batch, out, get_target(batch))
                if sel is None:
                    continue
                loss = torch.mean((sel - tgt) ** 2)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_acc += float(loss.item())
                batches += 1
            if batches:
                history['loss'].append(loss_acc / batches)
        return history

    @staticmethod
    def _per_sample_slices(batch, vec):
        """Yield (sample_id, concatenated per-node slice) in the policy's
        [a_transition; postpone] node order, per sample of the batch."""
        has_a = ('a_transition' in batch.x_dict)
        has_p = ('postpone' in batch.x_dict)
        nA = batch['a_transition'].x.size(0) if has_a else 0
        nP = batch['postpone'].x.size(0) if has_p else 0
        vec_a = vec[:nA] if nA else None
        vec_p = vec[nA:nA + nP] if nP else None
        idx_a = batch['a_transition'].batch.data if has_a else None
        idx_p = batch['postpone'].batch.data if has_p else None
        for s in (idx_a.unique() if has_a else idx_p.unique()):
            parts = []
            if has_a:
                v = vec_a[idx_a == s].reshape(-1)
                if v.numel():
                    parts.append(v)
            if has_p and vec_p is not None:
                v = vec_p[idx_p == s].reshape(-1)
                if v.numel():
                    parts.append(v)
            if parts:
                yield int(s), torch.cat(parts, dim=0)

    def _select_taken_nodes(self, batch, vec, targets):
        """(taken-node outputs, aligned targets) across the batch's samples."""
        actions = torch.as_tensor(batch.y)
        if targets.dim() == 0:
            targets = targets.unsqueeze(0)
        sel, tgt = [], []
        for s, cat in self._per_sample_slices(batch, vec):
            a_idx = int(actions[s])
            if 0 <= a_idx < cat.numel():
                sel.append(cat[a_idx].reshape(()))
                tgt.append(targets[s].reshape(()))
        if not sel:
            return None, None
        return torch.stack(sel), torch.stack(tgt)

    def _fit_lqi_policy_step(self, batch):
        """One AWR step: w = exp(A_std / TAU) with A = q(s,a_taken) minus the
        per-state mean of q over the AVAILABLE actions (availability-aware
        centering), loss = -mean(w * log pi(a_taken|s)) - ent_bonus * H."""
        self.policy_model.train()
        epsilon = 1e-7
        new_probs = self.policy_model(batch)
        new_logpis = (new_probs + epsilon).log()
        if new_probs.dim() == 2 and new_probs.size(-1) == 1:
            new_probs = new_probs.squeeze(-1)
        if new_logpis.dim() == 2 and new_logpis.size(-1) == 1:
            new_logpis = new_logpis.squeeze(-1)

        with torch.no_grad():
            q_all = self.qlin_model(batch) + self.qoff_model(batch)
            if q_all.dim() == 2 and q_all.size(-1) == 1:
                q_all = q_all.squeeze(-1)

        actions = torch.as_tensor(batch.y)
        old_logprob = batch.logprobs.clone()

        sel_logp, advs, kld_terms = [], [], []
        # Old per-node logpis for the true per-state KL monitor.
        has_a = ('a_transition' in batch.x_dict)
        has_p = ('postpone' in batch.x_dict)
        old_a = batch['a_transition'].logpis if has_a else None
        if old_a is not None and old_a.dim() == 2 and old_a.size(-1) == 1:
            old_a = old_a.squeeze(-1)
        old_p = None
        if has_p and hasattr(batch['postpone'], 'logpis'):
            old_p = batch['postpone'].logpis
            if old_p is not None and old_p.dim() == 2 and old_p.size(-1) == 1:
                old_p = old_p.squeeze(-1)
        old_vec = None
        if old_a is not None:
            old_vec = torch.cat([old_a, old_p], dim=0) if old_p is not None else old_a

        new_slices = dict(self._per_sample_slices(batch, new_logpis))
        q_slices = dict(self._per_sample_slices(batch, q_all))
        old_slices = (dict(self._per_sample_slices(batch, old_vec))
                      if old_vec is not None else {})

        for s, ns_cat in new_slices.items():
            q_cat = q_slices.get(s)
            if q_cat is None or q_cat.numel() != ns_cat.numel():
                continue
            a_idx = int(actions[s])
            if not (0 <= a_idx < ns_cat.numel()):
                continue
            sel_logp.append(ns_cat[a_idx].reshape(()))
            advs.append((q_cat[a_idx] - q_cat.mean()).reshape(()))
            os_cat = old_slices.get(s)
            if os_cat is not None and os_cat.numel() == ns_cat.numel():
                with torch.no_grad():
                    p_old = torch.exp(os_cat)
                    p_old = p_old / p_old.sum().clamp_min(1e-8)
                    kld_terms.append(float((p_old * (os_cat - ns_cat)).sum().item()))

        if not sel_logp:
            return 0.0, 0.0, 0.0, 0.0

        logp = torch.stack(sel_logp)
        A = torch.stack(advs)
        A = (A - A.mean()) / (A.std(unbiased=False) + 1e-8)
        w = torch.clamp(torch.exp(A / self._LQI_TAU), max=self._LQI_WMAX).detach()

        core = -torch.mean(w * logp)

        _ent_parts = []
        if has_a:
            _ent_parts.append(batch['a_transition'].batch.data)
        if has_p:
            _ent_parts.append(batch['postpone'].batch.data)
        ent = _normalized_entropy(new_probs, new_logpis, torch.cat(_ent_parts, dim=0))

        loss = core - self.ent_bonus * ent
        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.policy_optimizer.step()

        kld = sum(kld_terms) / len(kld_terms) if kld_terms else 0.0
        return float(loss.item()), kld, float(ent.item()), float(core.item())

    def _fit_qoff_model(self, dataloader, epochs=1):
        """LRQ-v3: fit the off-lineage per-action-node head on trace-computed
        targets (mc_q credit − lrq2 credit of the TAKEN action per step)."""
        history = {'loss': []}
        for epoch in range(epochs):
            loss_acc, batches = 0.0, 0
            for batch in dataloader:
                if not hasattr(batch, 'qoff_target'):
                    continue
                step_loss = self._fit_qoff_model_step(batch)
                loss_acc += step_loss
                batches += 1
            if batches:
                history['loss'].append(loss_acc / batches)
        return history

    def _fit_qoff_model_step(self, batch):
        """One regression step: select the taken action's node output per
        sample (same [a_transition; postpone] ordering as the policy) and MSE
        it against the off-lineage target."""
        self.qoff_model.train()
        out = self.qoff_model(batch)
        if out.dim() == 2 and out.size(-1) == 1:
            out = out.squeeze(-1)

        has_a = ('a_transition' in batch.x_dict)
        has_p = ('postpone' in batch.x_dict)
        nA = batch['a_transition'].x.size(0) if has_a else 0
        nP = batch['postpone'].x.size(0) if has_p else 0
        out_a = out[:nA] if nA else None
        out_p = out[nA:nA + nP] if nP else None
        idx_a = batch['a_transition'].batch.data if has_a else None
        idx_p = batch['postpone'].batch.data if has_p else None

        actions = torch.as_tensor(batch.y)
        targets = batch.qoff_target
        if targets.dim() == 0:
            targets = targets.unsqueeze(0)

        sel, tgt = [], []
        for s in (idx_a.unique() if has_a else idx_p.unique()):
            parts = []
            if has_a:
                v = out_a[idx_a == s].reshape(-1)
                if v.numel():
                    parts.append(v)
            if has_p and out_p is not None:
                v = out_p[idx_p == s].reshape(-1)
                if v.numel():
                    parts.append(v)
            if not parts:
                continue
            cat = torch.cat(parts, dim=0)
            a_idx = int(actions[s])
            if 0 <= a_idx < cat.numel():
                sel.append(cat[a_idx].reshape(()))
                tgt.append(targets[s].reshape(()))
        if not sel:
            return 0.0

        loss = torch.mean((torch.stack(sel) - torch.stack(tgt)) ** 2)
        self.qoff_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qoff_model.parameters(), 1.0)
        self.qoff_optimizer.step()
        return float(loss.item())

    def _fit_voff_model(self, dataloader, epochs=1):
        """LCV: regress the state-only v_off head on the measured off-lineage
        returns (batch.qoff_target). Better centering => more of the CV's
        variance is removable; a bad head only shrinks c_hat, never biases."""
        for epoch in range(epochs):
            for batch in dataloader:
                if not hasattr(batch, 'qoff_target'):
                    continue
                self.qoff_model.train()
                pred = self.qoff_model(batch).squeeze()
                tgt = batch.qoff_target
                if pred.dim() == 0:
                    pred = pred.unsqueeze(0)
                if tgt.dim() == 0:
                    tgt = tgt.unsqueeze(0)
                if pred.numel() != tgt.numel():
                    continue
                loss = torch.mean((pred - tgt) ** 2)
                self.qoff_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.qoff_model.parameters(), 1.0)
                self.qoff_optimizer.step()

    def _fit_value_model(self, dataloader, epochs=1):
        """Fit value model using data from dataset.

        Parameters
        ----------
        dataloader : DataLoader
            The data loader for the dataset.
        logpis : list of Tensors
            The log probabilities of the actions taken in the dataset.
        epochs : int, optional
            The number of epochs to train the policy model.

        Returns
        -------
        dict
            Dictionary containing training history with key 'loss'.

        """
        if self.value_model is None or isinstance(self.value_model, str):
            epochs = 0
        history = {'loss': []}
        for epoch in range(epochs):
            loss, batches = 0, 0
            for batch in dataloader:
                # batch = batch[0]
                batch_loss = self._fit_value_model_step(batch)
                loss += batch_loss
                batches += 1
            if batches == 0:
                print("No complete batches to process.")
                continue
            history['loss'].append(loss / batches)
        return {k: np.array(v) for k, v in history.items()}

    def _fit_value_model_step(self, batch):
        """Fit value model on one batch of data.

        LVA: when the critic has a lineage aux head AND the batch carries
        qlin_target (both only exist for causal_scheme='lva'), the loss is
            MSE(V, gae_returns) + causal_aux_coef * MSE(V_aux, qlin_target).
        The aux head predicts the per-decision lineage credit from the shared
        encoding — an auxiliary representation task. V's own target stays the
        unbiased GAE return, so any lineage bias stops at the encoder."""
        self.value_model.train()

        # indexes = batch['a_transition'].batch.data
        states = batch
        values = batch.value.clone()  # discounted returns

        aux_on = (getattr(self.value_model, 'lineage_aux_head', None) is not None
                  and hasattr(batch, 'qlin_target'))
        if aux_on:
            pred_values, pred_aux = self.value_model.forward_with_aux(states)
            pred_values = pred_values.squeeze()
            pred_aux = pred_aux.squeeze()
            aux_tgt = batch.qlin_target
            if pred_aux.dim() == 0:
                pred_aux = pred_aux.unsqueeze(0)
            if aux_tgt.dim() == 0:
                aux_tgt = aux_tgt.unsqueeze(0)
            loss = torch.mean(self.value_loss.forward(input=pred_values, target=values))
            if pred_aux.numel() == aux_tgt.numel():
                loss = loss + self.causal_aux_coef * torch.mean((pred_aux - aux_tgt) ** 2)
        else:
            pred_values = self.value_model(states).squeeze()
            loss = torch.mean(self.value_loss.forward(input=pred_values, target=values))

        self.value_optimizer.zero_grad()
        try:
            loss.backward()
        except Exception as e:
            print("Loss.backward produced an invalid output.")

        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), 1.0)
        self.value_optimizer.step()

        return loss.item()

    def _fit_ls_hca_hhat(self):
        """LS-HCA (Idea 1, FORKFREE_LINEAGE_RETHINK.md): refit the hindsight
        model hhat(a | reward-type r realized in the decision's lineage) from
        this epoch's pooled ``(action_type, reward_type, z)`` records -- a
        plain empirical table, Laplace-smoothed:
        P(a | r-type, z) = (count(a, r-type, z) + alpha) /
                           (count(r-type, z) + alpha * K),
        K = number of distinct action types observed for that (r-type, z)
        group. alpha=``self.ls_hca_smoothing_alpha`` (default 1.0); alpha=0
        recovers the original unsmoothed MLE. This is a batch statistic like
        LCV's c_hat, not a per-state model: coarser than the "logistic model"
        sketched in the doc, but a well-posed, tiny fit given the low-
        dimensional conditioning event (a lineage-membership indicator, not a
        return-space density). Used to correct NEXT epoch's contested credit
        (Agent.run_episode's ls_hca combine step, which also floors/clips the
        resulting correction factor -- smoothing here and the floor/clip
        there are complementary, not redundant: smoothing guards sparse
        buckets at FIT time, floor/clip guards any residual pi-vs-hhat
        staleness at APPLY time). The records pooled THIS epoch were
        themselves scored with the hhat fit at the END OF THE PREVIOUS epoch
        (or the empty table at epoch 0 -- the safe, pure-only floor), so
        fitting and applying never share a batch.

        ALSO fits a state-conditional model per (reward_type, z) group when
        there's enough data (>= ``self.ls_hca_state_min_n``) and state
        features were captured (``self._ls_hca_state_node_types`` non-empty)
        -- see ``_ls_hca_predict_h``. This is the "logistic model" referenced
        above, finally built: a tiny per-group ``nn.Linear`` (marking-vector
        state features -> logits over that group's observed action types),
        refit from scratch every epoch (no cross-epoch continuity, matching
        the flat table's own semantics). Groups below the data threshold, or
        when state features aren't available at all, fall back to the flat
        table -- this never changes behavior on envs/configs that predate
        this addition."""
        from collections import defaultdict
        recs = self._ls_hca_records
        if not recs:
            return {'n': 0}
        counts = defaultdict(int)
        totals = defaultdict(int)
        support = defaultdict(set)
        # Bin edges for the conditioning statistic, refit from THIS epoch's
        # pool and then used both for this fit's own labels and for next
        # epoch's apply step -- so labels and edges are always coherent, and
        # the model still lags the policy exactly as before. Quantiles rather
        # than fixed cut points because the statistic's scale is env-specific
        # (delays on one env are depths on another).
        n_bins = max(1, int(getattr(self, 'ls_hca_z_bins', 4)))
        stats_pool = [rec[5] for rec in recs if len(rec) > 5 and rec[5] is not None]
        if n_bins > 1 and len(stats_pool) >= n_bins:
            qs = np.quantile(stats_pool, [i / n_bins for i in range(1, n_bins)])
            # Collapse duplicate cut points: a statistic with few distinct
            # values (depth is often 1-3) would otherwise get empty levels.
            edges = sorted({float(q) for q in qs})
            self._ls_hca_z_edges = edges or None
        else:
            self._ls_hca_z_edges = None

        by_group = defaultdict(list)  # (rtype,lvl) -> [(a_type, state_feat, pi_vec), ...]
        by_rtype = defaultdict(list)  # rtype       -> [(a_type, state_feat, lvl), ...]
        for rec in recs:
            a_type, rtype = rec[0], rec[1]
            state_feat = rec[3] if len(rec) > 3 else None
            pi_vec = rec[4] if len(rec) > 4 else None
            zstat = rec[5] if len(rec) > 5 else None
            # Re-derive the level from the raw statistic under the edges just
            # computed; rec[2] carries the level assigned at APPLY time, under
            # the previous epoch's edges, which would be inconsistent here.
            lvl = self._ls_hca_level(zstat, z=bool(rec[2]))
            counts[(a_type, rtype, lvl)] += 1
            totals[(rtype, lvl)] += 1
            support[(rtype, lvl)].add(a_type)
            if state_feat is not None:
                by_group[(rtype, lvl)].append((a_type, state_feat, pi_vec))
                by_rtype[rtype].append((a_type, state_feat, lvl))
        alpha = getattr(self, 'ls_hca_smoothing_alpha', 1.0)
        self._ls_hca_hhat = {
            key: (counts[key] + alpha) /
                 (totals[(key[1], key[2])] + alpha * len(support[(key[1], key[2])]))
            for key in counts
        }

        self._ls_hca_hhat_model = {}
        min_n = getattr(self, 'ls_hca_state_min_n', 30)
        # state_dim derived from the actual captured feature vectors (marking
        # counts + the n_type/N share feature), not just len(node_types), so
        # this stays correct regardless of what _rudder_features returns or
        # how many extra scalars get appended alongside it in the future.
        has_state_config = len(getattr(self, '_ls_hca_state_node_types', []) or []) > 0
        if has_state_config:
            # --- consistent mode: fit P(z | x, a), one model per reward-type --
            # Modelling the LINEAGE-MEMBERSHIP probability instead of hhat
            # directly is what makes the law of total probability hold by
            # construction: h(a|x,z) = pi(a|x) P(z|x,a) / P(z|x) is a genuine
            # posterior, so sum_z P(z|x) h(a|x,z) = pi(a|x) sum_z P(z|x,a) =
            # pi(a|x) identically. Per-(rtype, z) fits could not do this --
            # nothing coupled the z=True and z=False models, and the resulting
            # violation was the whole residual over-correction (median 3.4% of
            # pi, 4.46x excess |factor|; _diag_ls_hca_consistency.py).
            # One binary head per action type, sharing the state features.
            # Zero init => P(z|x,a) = 0.5 for every a => P(z|x) = 0.5 =>
            # factor = 1 - P(z|x)/P(z|x,a) = 0, so this ALSO starts at the null.
            self._ls_hca_zmodel = {}
            if getattr(self, 'ls_hca_consistent', True):
                n_lv = self._ls_hca_n_levels()
                for rtype, items in by_rtype.items():
                    if len(items) < min_n:
                        continue
                    classes = sorted({a for a, _, _ in items})
                    state_dim = len(items[0][1])
                    if not classes or state_dim == 0:
                        continue
                    # More than one level must actually occur, else P(z|x,a) is
                    # degenerate and the correction is vacuous rather than
                    # merely small.
                    if len({lv for _, _, lv in items}) < 2:
                        continue
                    cls_idx = {c: i for i, c in enumerate(classes)}
                    X = torch.tensor([f for _, f, _ in items], dtype=torch.float32)
                    ai = torch.tensor([cls_idx[a] for a, _, _ in items], dtype=torch.long)
                    lv = torch.tensor([min(l, n_lv - 1) for _, _, l in items],
                                      dtype=torch.long)
                    # One categorical head per action type: outputs are
                    # (n_classes x n_levels), softmaxed over LEVELS. Zero init
                    # => uniform over levels for every action => P(z|x,a) is
                    # action-independent => w == 1, so training still starts at
                    # the null. n_levels==2 reproduces the binary case exactly.
                    zmodel = torch.nn.Linear(state_dim, len(classes) * n_lv)
                    torch.nn.init.zeros_(zmodel.weight)
                    torch.nn.init.zeros_(zmodel.bias)
                    zopt = torch.optim.AdamW(
                        zmodel.parameters(),
                        lr=getattr(self, 'ls_hca_state_lr', 0.05),
                        weight_decay=getattr(self, 'ls_hca_state_l2', 0.0))
                    rows = torch.arange(len(items))
                    for _ in range(getattr(self, 'ls_hca_state_epochs', 50)):
                        zopt.zero_grad()
                        # supervise only the head of the action actually taken
                        out = zmodel(X).view(-1, len(classes), n_lv)
                        torch.nn.functional.cross_entropy(
                            out[rows, ai], lv).backward()
                        zopt.step()
                    self._ls_hca_zmodel[rtype] = (zmodel, classes, n_lv)

            residual = getattr(self, 'ls_hca_residual', True)
            for group, items in by_group.items():
                if len(items) < min_n:
                    continue
                classes = sorted({a for a, _, _ in items})
                if not classes:
                    continue
                state_dim = len(items[0][1])
                if state_dim == 0:
                    continue
                cls_idx = {c: i for i, c in enumerate(classes)}
                X = torch.tensor([f for _, f, _ in items], dtype=torch.float32)
                y = torch.tensor([cls_idx[a] for a, _, _ in items], dtype=torch.long)
                model = torch.nn.Linear(state_dim, len(classes))

                base = None
                if residual:
                    # log pi over `classes`, per record: the softmax OFFSET the
                    # residual model perturbs. Types the policy could not take
                    # at that state get -inf (masked out of the softmax, which
                    # is what "not enabled" means). If a record has no captured
                    # pi vector at all, its base row is flat -- that record then
                    # contributes an ordinary unconditioned fit rather than
                    # silently corrupting the others.
                    rows = []
                    for _, _, pv in items:
                        if pv:
                            rows.append([float(pv.get(c, 0.0)) for c in classes])
                        else:
                            rows.append([1.0] * len(classes))
                    base = torch.tensor(rows, dtype=torch.float32)
                    base = torch.log(base.clamp_min(1e-12))
                    base[base <= math.log(1e-12)] = float('-inf')
                    # A record whose whole row is masked would produce a NaN
                    # loss; fall back to flat for those (cannot happen when the
                    # taken type is present, which it always is in practice).
                    allmask = torch.isinf(base).all(dim=1)
                    if bool(allmask.any()):
                        base[allmask] = 0.0
                    # Zero init => g == 0 => h == pi exactly on the first step,
                    # i.e. training STARTS at the null (factor == 0) and only
                    # moves away from it to the extent the data demands.
                    torch.nn.init.zeros_(model.weight)
                    torch.nn.init.zeros_(model.bias)

                # AdamW (decoupled) rather than Adam(weight_decay=): decoupling
                # makes the shrinkage strength mean the same thing regardless of
                # gradient scale, so ls_hca_state_l2 stays interpretable as
                # "how hard to pull g back to the null" across envs.
                l2 = getattr(self, 'ls_hca_state_l2', 0.0)
                opt = torch.optim.AdamW(model.parameters(),
                                        lr=getattr(self, 'ls_hca_state_lr', 0.05),
                                        weight_decay=l2)
                n_epochs = getattr(self, 'ls_hca_state_epochs', 50)
                for _ in range(n_epochs):
                    opt.zero_grad()
                    logits = model(X) if base is None else model(X) + base
                    loss = torch.nn.functional.cross_entropy(logits, y)
                    loss.backward()
                    opt.step()
                self._ls_hca_hhat_model[group] = (model, classes)

        # With the flat fallback off (the default), a group with no fitted
        # state model contributes factor=0. If NO group got a model, every
        # correction is 0 -- and on an env whose reward-types are all
        # CONTESTED (PURE is then empty, so the exact term is identically
        # zero too) that silently leaves the agent with no credit signal at
        # all. Loud, because the symptom otherwise looks like a training
        # failure rather than a configuration one.
        if not self._ls_hca_hhat_model and not getattr(self, 'ls_hca_flat_fallback', False):
            reason = ("no state features configured (metadata missing -> "
                      "_ls_hca_state_node_types is empty)" if not has_state_config
                      else f"every (reward_type, z) group had < ls_hca_state_min_n"
                           f"={min_n} records")
            get_logger().warning(
                f"  [LS-HCA] no state-conditional model could be fit ({reason}); "
                f"with ls_hca_flat_fallback=False every hindsight correction "
                f"will be 0 this epoch. Set ls_hca_flat_fallback=True to use "
                f"the flat table instead, or lower ls_hca_state_min_n.")

        return {'n': len(recs), 'n_models': len(self._ls_hca_hhat_model),
                'n_zmodels': len(getattr(self, '_ls_hca_zmodel', {}) or {})}

    def _ls_hca_zstat(self, z, delay, depth, share):
        """Pick the configured conditioning statistic; None when the decision
        is not in this reward's lineage (that is its own level, not a value)."""
        if not z:
            return None
        feat = getattr(self, 'ls_hca_z_feature', 'delay')
        if feat == 'delay':
            return delay
        if feat == 'depth':
            return depth
        if feat == 'share':
            return share
        return 1.0                      # 'membership': one in-lineage level

    def _ls_hca_level(self, zstat, z=None):
        """Conditioning LEVEL: 0 = not in this reward's lineage, 1..B = in it,
        bucketed by the configured statistic against the stored quantile edges.

        Falls back to a single in-lineage level when no edges exist yet (epoch
        0) or the statistic is missing, which reproduces the old binary
        membership variable exactly -- so a cold start is the previous
        behaviour, not an undefined one."""
        if zstat is None:
            return 1 if (z and getattr(self, 'ls_hca_z_feature', 'delay') == 'membership') else (1 if z else 0)
        edges = getattr(self, '_ls_hca_z_edges', None)
        if not edges:
            return 1
        lvl = 1
        for e in edges:
            if zstat > e:
                lvl += 1
            else:
                break
        return lvl

    def _ls_hca_n_levels(self):
        edges = getattr(self, '_ls_hca_z_edges', None)
        return 1 + (len(edges) + 1 if edges else 1)

    def _ls_hca_predict_h(self, a_type, rtype, z, state_feat, hhat, pi_type_vec=None):
        """hhat(a_type | state, rtype, z): the fitted state-conditional model
        (``_fit_ls_hca_hhat``'s ``self._ls_hca_hhat_model``) when available
        for this (reward_type, z) group and ``a_type`` is one of its fitted
        classes; else ``None``, so the caller applies the cold-start-safe
        factor=0 floor.

        The flat Laplace-smoothed table is consulted only when
        ``ls_hca_flat_fallback`` is on (default off -- it carries no
        measurable signal; see the knob's rationale in ``__init__``)."""
        # Consistent mode: h = pi * P(z|x,a) / P(z|x). Returned as an h so the
        # caller's `factor = 1 - pit/h` is unchanged, but note what that
        # becomes once pi cancels:
        #     factor = 1 - P(z|x) / P(z|x,a)
        # -- the policy drops out of the correction entirely. The null is then
        # exact for a strictly stronger reason than under the residual form: it
        # holds whenever the action does not shift the lineage-membership
        # probability, P(z|x,a) == P(z|x), with no reference to pi's own
        # accuracy at all.
        zentry = (self._ls_hca_zmodel.get(rtype)
                  if getattr(self, 'ls_hca_consistent', True)
                  and getattr(self, '_ls_hca_zmodel', None) else None)
        if zentry is not None and state_feat is not None and pi_type_vec:
            zmodel, zclasses, n_lv = zentry
            # `z` is the conditioning LEVEL here (0 = not in this reward's
            # lineage, 1..B = in it, bucketed by ls_hca_z_feature). Booleans
            # still work -- False/True are 0/1 -- so the binary callers and
            # tests need no change.
            lvl = int(z)
            if a_type in zclasses and 0 <= lvl < n_lv:
                enabled = [t for t, p in pi_type_vec.items()
                           if p > 0.0 and t in zclasses]
                if enabled:
                    with torch.no_grad():
                        x = torch.tensor([state_feat], dtype=torch.float32)
                        p = torch.softmax(
                            zmodel(x).view(len(zclasses), n_lv), dim=-1)
                    # P(z=lvl|x) marginalizes over the enabled+fitted types,
                    # renormalizing pi over them (an enabled type the fit never
                    # saw has no P(z|x,a) to contribute; on these envs every
                    # action type is fitted, so the renormalization is a no-op).
                    wsum = sum(pi_type_vec[t] for t in enabled)
                    p_marg = sum(pi_type_vec[t] * float(p[zclasses.index(t), lvl])
                                 for t in enabled) / max(wsum, 1e-12)
                    p_cond = float(p[zclasses.index(a_type), lvl])
                    eps = 1e-6
                    if p_cond > eps:
                        return float(pi_type_vec[a_type]) * p_cond / max(p_marg, eps)
                    return None

        group = (rtype, z)
        model_entry = self._ls_hca_hhat_model.get(group) if hasattr(self, '_ls_hca_hhat_model') else None
        if model_entry is not None and state_feat is not None:
            model, classes = model_entry
            residual = getattr(self, 'ls_hca_residual', True)
            if residual and pi_type_vec:
                # Residual form. The softmax runs over the types ACTUALLY
                # ENABLED at this state (pi_type_vec's support), not over the
                # fitted classes: a type the fit never saw simply gets g=0, and
                # a fitted class that is not enabled here is absent from the
                # normalization. That is what makes the null exact for any
                # enabled set -- with g == 0 the softmax returns pi itself
                # (pi_type_vec already sums to 1 over the enabled types), so
                # factor = 1 - pi/h = 0 identically.
                enabled = [t for t, p in pi_type_vec.items() if p > 0.0]
                if a_type not in enabled:
                    return None
                with torch.no_grad():
                    x = torch.tensor([state_feat], dtype=torch.float32)
                    g = model(x).reshape(-1)
                    logits = torch.tensor(
                        [math.log(max(pi_type_vec[t], 1e-12))
                         + (float(g[classes.index(t)]) if t in classes else 0.0)
                         for t in enabled], dtype=torch.float32)
                    probs = torch.softmax(logits, dim=-1)
                return float(probs[enabled.index(a_type)])
            if a_type in classes:
                with torch.no_grad():
                    x = torch.tensor([state_feat], dtype=torch.float32)
                    probs = torch.softmax(model(x), dim=-1).reshape(-1)
                return float(probs[classes.index(a_type)])
        if getattr(self, 'ls_hca_flat_fallback', False):
            return hhat.get((a_type, rtype, z))
        return None

    def _fit_cf_preferences(self):
        """G1 aux pass: pairwise logistic loss on the policy's log-probs for
        this epoch's SNR-gated counterfactual preferences.

        loss = -log sigmoid(logpi(winner) - logpi(loser)) per preference
        (the softmax normalization cancels in the difference, so this is the
        logit-difference DPO-style objective). Coefficient = cf coef *
        linear anneal to 0 over training: the floor is exact PPO by
        construction once the anneal completes. Runs AFTER the PPO policy
        epochs, so it is outside the KL early stop — kept safe by the small
        preference count, the gate, and the anneal.
        """
        cfg = self.cf_config
        # Decomposed mode: resolve this epoch's fork records into preferences
        # first — the indirect channel is a regression pooled over all of
        # them, so it cannot be decided fork-by-fork.
        self._cf_resolve_stats = None
        if cfg.get('decompose', False):
            from gympn.counterfactual import resolve_decomp_preferences
            recs = getattr(self, '_cf_records', [])
            # Rolling cross-epoch window: one epoch's forks cannot validate a
            # 5-parameter fit, and the occupancy->opportunity-cost relation
            # drifts slowly enough to pool.
            if not hasattr(self, '_cf_pool'):
                from collections import deque
                self._cf_pool = deque(maxlen=int(cfg.get('fit_window', 400)))
            self._cf_pool.extend(recs)
            resolved, rstats = resolve_decomp_preferences(
                recs, cfg['gate'], min_r2=cfg.get('min_r2', 0.05),
                fit_pool=self._cf_pool)
            self._cf_prefs = resolved
            self._cf_resolve_stats = rstats
        prefs = getattr(self, '_cf_prefs', [])
        if cfg.get('anneal', True):
            total = max(1, (self._total_epochs or 1) - 1)
            coef = cfg['coef'] * max(0.0, 1.0 - self._current_epoch / total)
        else:
            # X10 falsifier follow-up (mechanism probe): constant pressure,
            # floor knowingly sacrificed — tests the ceiling hypothesis
            # against the strongest version of G1.
            coef = cfg['coef']
        if not prefs or coef <= 0.0:
            return {'n': len(prefs), 'loss': 0.0, 'coef': coef}
        self.policy_model.train()
        last_loss = 0.0
        for _ in range(cfg['updates']):
            self.policy_optimizer.zero_grad()
            losses = []
            for p in prefs:
                pi = self.policy_model(p['state'])
                logp = pi.log().reshape(-1)
                if p['winner'] >= logp.numel() or p['loser'] >= logp.numel():
                    continue
                losses.append(-torch.nn.functional.logsigmoid(
                    logp[p['winner']] - logp[p['loser']]))
            if not losses:
                return {'n': len(prefs), 'loss': 0.0, 'coef': coef}
            loss = coef * torch.stack(losses).mean()
            loss.backward()
            self.policy_optimizer.step()
            last_loss = float(loss.item())
        return {'n': len(prefs), 'loss': last_loss, 'coef': coef}

    def test_in_train(self, env, episodes=100, max_episode_length=None, deterministic=True, logdir=None,
                      eval_seed=None):
        """Evaluate the agent on a test environment during training.

        Parameters
        ----------
        env : environment
            The test environment to evaluate on.
        episodes : int, optional
            The number of episodes to run for evaluation.
        max_episode_length : int, optional
            The maximum number of steps in an episode.
        deterministic : bool, optional
            Whether to use a deterministic policy during testing.
        logdir : str, optional
            Directory to save the best policy.
        eval_seed : int, optional
            Base seed for COMMON RANDOM NUMBERS across evaluations. When set,
            episode ``i`` runs under seed ``eval_seed + i``, so every eval point
            -- across epochs, across runs, and across methods -- scores the
            policy on the SAME fixed set of scenarios.

            Without it (the default, and the historical behaviour) each eval
            draws fresh scenarios from wherever training left the global RNG:
            unbiased, but two arms are compared on different sample paths.
            Measured on s1, one 20-episode eval point carries +-0.231 SD of
            pure scenario noise, so a paired single-point difference carries
            +-0.327 -- and ``greedy_drift``, a max over ~15 such points, is
            inflated by ~0.40 for a policy that is genuinely flat.

            The env's stochasticity comes from the global ``random`` / NumPy
            streams, so this reseeds those per episode and RESTORES the prior
            state afterwards. Training's stream therefore continues across the
            eval as if it had not run -- verified directly in
            suite/_test_eval_crn.py (T2).

            That does NOT make a CRN run step-identical to a non-CRN run of the
            same seed, and it cannot: without eval_seed the eval CONSUMES the
            training stream (20 episodes' worth of draws per eval point), so
            the two configurations' rollouts diverge from the first eval
            onward -- measured on a 4-epoch s1 cell, epoch 4's sampled return
            was 9.65 with CRN vs 9.40 without. Results produced with eval_seed
            set are a new baseline, not a re-scoring of existing cells.

            ``env.reset(seed=...)`` is deliberately NOT used: it routes to
            seed_everything, which would also reseed torch and re-apply the
            deterministic-kernel switches on every eval episode.

        Returns
        -------
        test_metrics : dict
            Dictionary containing evaluation metrics (mean, min, max, std returns and lengths).
        """
        history = {'returns': np.zeros(episodes), 'lengths': np.zeros(episodes)}
        crn = eval_seed is not None
        if crn:
            saved_random = random.getstate()
            saved_np = np.random.get_state()
            saved_torch = torch.get_rng_state()
        try:
            for i in range(episodes):
                if crn:
                    # Same scenario i for every arm and every epoch.
                    random.seed(eval_seed + i)
                    np.random.seed((eval_seed + i) % (2 ** 32))
                state = env.reset()
                done = False
                episode_length = 0
                total_reward = 0
                info = {'pn_reward': 0}
                while not done:
                    action = self.act(state, deterministic=deterministic)
                    next_state, reward, done, truncated, info = env.step(action)
                    episode_length += 1
                    state = next_state

                if episode_length == 0:
                    get_logger().warning("Episode length is zero - no valid action produced")
                history['returns'][i] = info['pn_reward']
                history['lengths'][i] = episode_length
        finally:
            if crn:
                random.setstate(saved_random)
                np.random.set_state(saved_np)
                torch.set_rng_state(saved_torch)

        test_metrics = {
            'mean_returns': np.mean(history['returns']),
            'min_returns': np.min(history['returns']),
            'max_returns': np.max(history['returns']),
            'std_returns': np.std(history['returns']),
            'mean_ep_lens': np.mean(history['lengths']),
            'min_ep_lens': np.min(history['lengths']),
            'max_ep_lens': np.max(history['lengths']),
            'std_ep_lens': np.std(history['lengths']),
        }

        # Save the best policy if the current mean_returns is better
        if test_metrics['mean_returns'] > self.best_test_metric:
            self.best_test_metric = test_metrics['mean_returns']
            if logdir is not None:
                self.save_policy_network(f"{logdir}/best_policy.pth")
                get_logger().best_policy_saved(logdir, self.best_test_metric)

        # After testing or inference
        self.policy_model.train()
        self.value_model.train()

        return test_metrics

    def load_value_weights(self, filename):
        """Load weights from filename into the value model."""
        if self.value_model is not None and self.value_model != 'env':
            self.value_model.load_weights(filename)

    def save_value_weights(self, filename):
        """Save the current weights in the value model to filename."""
        if self.value_model is not None and not isinstance(self.value_model, str):
            self.value_model.save_weights(filename)

    def save_policy_network(self, filename):
        """Save the current policy to file.

        Parameters
        ----------
        filename : str
            The path to the file where the model will be saved.
        """

        torch.save(self.policy_model, filename)

    def load_policy_network(self, filename):
        """Load the current policy from file.

        Parameters
        ----------
        filename : str
            The path to the file from which the model will be loaded.
        """
        self.policy_model = torch.load(torch.load(filename))

    def _rudder_features(self, state, node_types=None):
        """Featurize a graph observation: the marking vector (token count per
        node type, in a fixed metadata order). Cheap, Markov-ish and
        constant-dim regardless of which places are currently occupied.
        Originally built for the RUDDER LSTM (``node_types`` defaults to
        ``self._rudder_node_types``, from ``rudder_config``); reused as-is
        for LS-HCA's state-conditional hindsight model (``node_types=self.
        _ls_hca_state_node_types``) -- same general, task-assignment-agnostic
        featurization, no new machinery."""
        g = state['graph'] if isinstance(state, dict) and 'graph' in state else state
        if node_types is None:
            node_types = getattr(self, '_rudder_node_types', []) or []
        feats = []
        for nt in node_types:
            try:
                feats.append(float(g[nt].x.size(0)) if nt in g.node_types else 0.0)
            except Exception:
                feats.append(0.0)
        return feats

    def _fit_policy_and_value_models(self, dataloader, epochs=1):
        """Fit both policy and value models simultaneously using data from dataset.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            The data loader containing batches of training data.
        epochs : int, optional
            Number of epochs to train for.

        Returns
        -------
        dict
            Dictionary containing training history with keys 'loss', 'kld', and 'ent'.
        """
        history = {'loss': [], 'kld': [], 'ent': [], 'policy_core_loss': [], 'value_loss': []}

        # --- Policy: PPO clipped surrogate for `policy_updates` (=`epochs`) passes ---
        # The policy and value nets are decoupled (separate optimizers/backward), so
        # the value net is NOT trained here; it gets its own `value_updates` passes
        # below. This fixes the previous behaviour where the value net was trained
        # exactly `policy_updates` times and `value_updates` was silently ignored
        # (fix #1), and removes the inert `vf_coeff` from the policy step (fix #2).
        for epoch in range(epochs):
            loss, kld, ent, batches = 0, 0, 0, 0
            policy_core_acc = 0

            for batch_i, batch in enumerate(dataloader, start=1):
                batch_loss, batch_kld, batch_ent, batch_ploss = self._fit_policy_and_value_model_step(batch)
                loss += batch_loss
                kld += batch_kld
                ent += batch_ent
                policy_core_acc += batch_ploss
                batches += 1

                # Debugging hooks: print per-batch KLD and entropy if requested
                try:
                    if os.environ.get('GP_DEBUG_PPO', '0') == '1':
                        print(
                            f"[PPO DEBUG] epoch={epoch + 1} batch={batch_i} batch_kld={batch_kld:.6f} batch_ent={batch_ent:.6f}")
                except Exception:
                    pass

            if batches == 0:
                get_logger().no_batches_warning()
                continue

            history['loss'].append(loss / batches)
            history['kld'].append(kld / batches)
            history['ent'].append(ent / batches)
            history['policy_core_loss'].append(policy_core_acc / batches)

            # KL early stopping: abort remaining inner updates once the mean
            # per-state KL(old || new) exceeds the limit (true KL, >= 0).
            if self.kld_limit is not None and history['kld'][-1] > self.kld_limit:
                break

        # --- Value: MSE regression on the (GAE) return targets for `value_updates`
        # passes, with its own optimizer (fix #1). ---
        if self.value_model is not None and not isinstance(self.value_model, str):
            value_history = self._fit_value_model(dataloader, epochs=self.value_updates)
            history['value_loss'] = list(value_history.get('loss', []))

        return {k: np.array(v) for k, v in history.items()}

    # ==================================================================
    # Causal PG: policy-only training with causal credits as advantages
    # ==================================================================

    def _fit_causal_policy_models(self, dataloader, epochs=1):
        """Fit policy + value using causal credits with GAE(γ=1, λ).

        Mathematical basis — Credits-as-Rewards with GAE(γ=1, λ):
        ──────────────────────────────────────────────────────────
        In causal RL mode, the reward redistribution produces per-action credits
        c_t where Σ_t c_t = episode return.  These credits are treated as step
        rewards and processed with standard GAE but using γ=1 (no discounting
        on credits):

          • Value targets:  V_target(t) = Σ_{k≥t} c_k  (sum of future credits)
            With γ=1, V(s_t) = c_t + V(s_{t+1}) — the Bellman equation holds.

          • Advantages:  GAE(γ=1, λ)
            δ_t = c_t + V(s_{t+1}) − V(s_t), A_t = Σ_l λ^l δ_{t+l}
            Multi-step structure prevents oversmoothing where V(s_t) ≈ E[c_t|s_t]
            would kill the gradient signal prematurely.

          • Policy gradient:  ∇J ≈ Σ_t A_t ∇log π(a_t|s_t)

        Why GAE instead of pure credit-baseline (A_t = c_t − V(s_t)):
          With pure credit-baseline (equivalent to λ=0), V quickly learns to
          predict E[c_t|s_t], making advantages ≈ 0 and halting learning even
          for suboptimal policies.  With λ>0, GAE requires V to predict the
          entire future trajectory correctly before advantages vanish.
        """
        history = {'loss': [], 'kld': [], 'ent': [], 'policy_core_loss': [], 'value_loss': []}

        # --- Policy: PPO clipped surrogate on causal GAE advantages for
        # `policy_updates` (=`epochs`) passes. Value is trained separately below so
        # that `value_updates` is honoured (fix #1) and the inert `vf_coeff` is
        # dropped (fix #2) — mirrors the standard PPO path. ---
        for epoch in range(epochs):
            loss_acc, kld_acc, ent_acc, batches = 0.0, 0.0, 0.0, 0
            ploss_acc = 0.0

            for batch_i, batch in enumerate(dataloader, start=1):
                # 4-tuple: (policy_loss, kld, ent, policy_core_loss)
                batch_loss, batch_kld, batch_ent, batch_ploss = self._fit_causal_policy_step(batch)
                loss_acc += batch_loss
                kld_acc += batch_kld
                ent_acc += batch_ent
                ploss_acc += batch_ploss
                batches += 1

            if batches == 0:
                get_logger().no_batches_warning()
                continue

            history['loss'].append(loss_acc / batches)
            history['kld'].append(kld_acc / batches)
            history['ent'].append(ent_acc / batches)
            history['policy_core_loss'].append(ploss_acc / batches)

            # KL early stopping: abort remaining inner updates once the mean
            # per-state KL(old || new) exceeds the limit (true KL, >= 0).
            if self.kld_limit is not None and history['kld'][-1] > self.kld_limit:
                break

        # --- Value: MSE regression on the return-to-go-over-credits targets for
        # `value_updates` passes, with its own optimizer (fix #1). ---
        if self.value_model is not None and not isinstance(self.value_model, str):
            value_history = self._fit_value_model(dataloader, epochs=self.value_updates)
            history['value_loss'] = list(value_history.get('loss', []))

        # --- LRQ-v3: off-lineage head regression on (mc_q - lrq2) targets. ---
        if getattr(self, 'qoff_model', None) is not None:
            qoff_hist = self._fit_qoff_model(dataloader, epochs=self.value_updates)
            history['qoff_loss'] = list(qoff_hist.get('loss', []))

        return {k: np.array(v) for k, v in history.items()}

    def _fit_causal_policy_step(self, batch):
        """One gradient step for causal policy gradient with GAE(γ=1, λ).

        Credits-as-Rewards with GAE — no oversmoothing:
          • batch.advantage = GAE advantages (γ=1, λ)    [set by finish()]
          • batch.value     = Σ_{k≥t} c_k                [sum of future credits]
          • Policy loss: PPO clipped surrogate on GAE advantages
          • Value loss:  MSE(V(s_t), Σ_{k≥t} c_k) — learns expected remaining credit
          • Combined:    L = L_policy + vf_coeff * L_value − ent_bonus * H(π)

        Returns
        -------
        tuple of 5 floats: (loss_total, kld, ent, value_loss, policy_core_loss)
            Same signature as _fit_policy_and_value_model_step for consistency.
        """
        self.policy_model.train()
        if self.value_model is not None and not isinstance(self.value_model, str):
            self.value_model.train()

        epsilon = 1e-7
        new_probs = self.policy_model(batch)
        new_logpis = (new_probs + epsilon).log()
        # Squeeze both to 1-D to avoid broadcasting bugs in entropy
        if new_probs.dim() == 2 and new_probs.size(-1) == 1:
            new_probs = new_probs.squeeze(-1)
        if new_logpis.dim() == 2 and new_logpis.size(-1) == 1:
            new_logpis = new_logpis.squeeze(-1)

        actions = torch.as_tensor(batch.y)
        advantages = batch.advantage.clone()
        old_logprob = batch.logprobs.clone()

        has_a = ('a_transition' in batch.x_dict)
        has_p = ('postpone' in batch.x_dict)

        nA = batch['a_transition'].x.size(0) if has_a else 0
        nP = batch['postpone'].x.size(0) if has_p else 0

        new_logpis_a = new_logpis[:nA] if nA else None
        new_logpis_p = new_logpis[nA:nA + nP] if nP else None

        old_logpis_a = batch['a_transition'].logpis if has_a else None
        if old_logpis_a is not None and old_logpis_a.dim() == 2 and old_logpis_a.size(-1) == 1:
            old_logpis_a = old_logpis_a.squeeze(-1)

        old_logpis_p = None
        if has_p and hasattr(batch['postpone'], 'logpis'):
            old_logpis_p = batch['postpone'].logpis
            if old_logpis_p is not None and old_logpis_p.dim() == 2 and old_logpis_p.size(-1) == 1:
                old_logpis_p = old_logpis_p.squeeze(-1)

        idx_a = batch['a_transition'].batch.data if has_a else None
        idx_p = batch['postpone'].batch.data if has_p else None

        unique_samples = (idx_a.unique() if has_a else idx_p.unique())

        sel_new, sel_old, sel_adv = [], [], []
        kld_terms = []

        for s in unique_samples:
            parts_new = []
            parts_old = []

            if has_a:
                mask_a = (idx_a == s)
                ns_a = new_logpis_a[mask_a].reshape(-1)
                if ns_a.numel():
                    parts_new.append(ns_a)
                    if old_logpis_a is not None:
                        parts_old.append(old_logpis_a[mask_a].reshape(-1))
                    else:
                        parts_old.append(ns_a.detach())

            if has_p and new_logpis_p is not None:
                mask_p = (idx_p == s)
                ns_p = new_logpis_p[mask_p].reshape(-1)
                if ns_p.numel():
                    parts_new.append(ns_p)
                    if old_logpis_p is not None:
                        parts_old.append(old_logpis_p[mask_p].reshape(-1))
                    else:
                        # Missing old logpis contribute 0 to the KL (using the
                        # new values); zeros_like would mean p_old = 1.
                        parts_old.append(ns_p.detach())

            if len(parts_new) == 0:
                continue

            ns_cat = torch.cat(parts_new, dim=0)
            os_cat = torch.cat(parts_old, dim=0)
            a_idx = int(actions[s])

            sel_new.append(ns_cat[a_idx].reshape(()))
            sel_old.append(old_logprob[s].reshape(()))
            sel_adv.append(advantages[s].reshape(()))

            # TRUE per-state KL(old || new) over this state's OWN action set
            # (see _fit_policy_and_value_model_step for why the previous
            # chosen-action log-ratio was not a usable trust-region signal).
            with torch.no_grad():
                p_old = torch.exp(os_cat)
                p_old = p_old / p_old.sum().clamp_min(1e-8)
                kld_terms.append(float((p_old * (os_cat - ns_cat)).sum().item()))

        # ----- Policy-only update (fix #1/#2) -----
        # The value net is trained separately for `value_updates` passes in
        # _fit_causal_policy_models. Because policy and value are fully decoupled
        # (separate optimizers/backward), `vf_coeff` never affected the gradient and
        # is dropped from this path.
        self.policy_optimizer.zero_grad()

        if len(sel_new) == 0:
            # No policy samples this batch (value is trained separately below).
            return 0.0, 0.0, 0.0, 0.0

        new_sel = torch.stack(sel_new)
        old_sel = torch.stack(sel_old)
        adv_sel = torch.stack(sel_adv)

        # PPO clipped surrogate loss
        loss_policy_core = torch.mean(self.policy_loss(new_sel, old_sel, adv_sel))

        # Normalized entropy bonus (action-space invariant, in [0,1])
        _ent_parts = []
        if has_a and idx_a is not None:
            _ent_parts.append(idx_a)
        if has_p and idx_p is not None:
            _ent_parts.append(idx_p)
        _ent_idx = torch.cat(_ent_parts, dim=0) if _ent_parts else idx_a
        ent = _normalized_entropy(new_probs, new_logpis, _ent_idx)

        loss_policy = loss_policy_core - self.ent_bonus * ent

        # KLD for monitoring
        if len(kld_terms) > 0:
            kld = sum(kld_terms) / len(kld_terms)
        else:
            kld = 0.0

        # Policy backward + step
        loss_policy.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.policy_optimizer.step()

        # Return (policy_loss, kld, ent, policy_core_loss)
        return float(loss_policy.item()), kld, ent.item(), float(loss_policy_core.item())

    def _fit_policy_and_value_model_step(self, batch):
        self.policy_model.train()
        self.value_model.train()

        epsilon = 1e-7
        new_probs = self.policy_model(batch)
        new_logpis = (new_probs + epsilon).log()
        # Squeeze both to 1-D to avoid broadcasting bugs in entropy
        if new_probs.dim() == 2 and new_probs.size(-1) == 1:
            new_probs = new_probs.squeeze(-1)
        if new_logpis.dim() == 2 and new_logpis.size(-1) == 1:
            new_logpis = new_logpis.squeeze(-1)

        actions = torch.as_tensor(batch.y)
        advantages = batch.advantage.clone()
        old_logprob = batch.logprobs.clone()

        has_a = ('a_transition' in batch.x_dict)
        has_p = ('postpone' in batch.x_dict)

        nA = batch['a_transition'].x.size(0) if has_a else 0
        nP = batch['postpone'].x.size(0) if has_p else 0

        # Split new logits by node type in the same order as actions_dict
        new_logpis_a = new_logpis[:nA] if nA else None
        new_logpis_p = new_logpis[nA:nA + nP] if nP else None

        # Old logits (standardize them to 1-D up front)
        old_logpis_a = batch['a_transition'].logpis if has_a else None
        if old_logpis_a is not None and old_logpis_a.dim() == 2 and old_logpis_a.size(-1) == 1:
            old_logpis_a = old_logpis_a.squeeze(-1)

        old_logpis_p = None
        if has_p and hasattr(batch['postpone'], 'logpis'):
            old_logpis_p = batch['postpone'].logpis
            if old_logpis_p is not None and old_logpis_p.dim() == 2 and old_logpis_p.size(-1) == 1:
                old_logpis_p = old_logpis_p.squeeze(-1)

        idx_a = batch['a_transition'].batch.data if has_a else None
        idx_p = batch['postpone'].batch.data if has_p else None

        unique_samples = (idx_a.unique() if has_a else idx_p.unique())

        sel_new, sel_old, sel_adv = [], [], []
        kld_terms = []

        for s in unique_samples:
            parts_new = []
            parts_old = []

            if has_a:
                mask_a = (idx_a == s)
                ns_a = new_logpis_a[mask_a]  # 1-D slice
                os_a = old_logpis_a[mask_a]  # 1-D slice
                # Flatten defensively (handles accidental (k,1))
                ns_a = ns_a.reshape(-1)
                os_a = os_a.reshape(-1)
                if ns_a.numel():
                    parts_new.append(ns_a)
                    parts_old.append(os_a)

            if has_p and new_logpis_p is not None:
                mask_p = (idx_p == s)
                ns_p = new_logpis_p[mask_p].reshape(-1)  # 1-D slice
                if old_logpis_p is not None:
                    os_p = old_logpis_p[mask_p].reshape(-1)  # 1-D slice
                else:
                    # Old logpis not stored for postpone: use the new values so
                    # this part contributes 0 to the KL (zeros_like would mean
                    # log p_old = 0, i.e. p_old = 1 — corrupting the KL).
                    os_p = ns_p.detach()
                if ns_p.numel():
                    parts_new.append(ns_p)
                    parts_old.append(os_p)

            if len(parts_new) == 0:
                # No action nodes for this sample -> skip policy update; still train value below
                continue

            # Concatenate 1-D slices safely
            ns_cat = torch.cat(parts_new, dim=0)  # (A_s + P_s,)
            os_cat = torch.cat(parts_old, dim=0)  # same length

            a_idx = int(actions[s])
            # if a_idx < 0 or a_idx >= ns_cat.shape[0]:
            #    # out-of-range chosen index -> skip this sample
            #    continue

            sel_new.append(ns_cat[a_idx].reshape(()))
            sel_old.append(old_logprob[s].reshape(()))
            sel_adv.append(advantages[s].reshape(()))

            # TRUE per-state KL(old || new) over this state's OWN action set:
            #   KL_s = sum_a p_old(a|s) * (log p_old(a|s) - log p_new(a|s)).
            # Non-negative and well-defined for variable-size action sets (old
            # and new share the state's support), unlike the previous
            # chosen-action log-ratio, which was signed (batch mean cancels),
            # single-sample (huge variance) and |A(s)|-dependent.
            with torch.no_grad():
                p_old = torch.exp(os_cat)
                p_old = p_old / p_old.sum().clamp_min(1e-8)
                kld_terms.append(float((p_old * (os_cat - ns_cat)).sum().item()))

        # ----- Policy-only update (fix #1/#2) -----
        # The value net is trained separately for `value_updates` passes in
        # _fit_policy_and_value_models. Because policy and value are fully decoupled
        # (separate optimizers/backward), `vf_coeff` never affected the gradient and
        # is therefore dropped from this path.
        self.policy_optimizer.zero_grad()

        new_sel = torch.stack(sel_new)
        old_sel = torch.stack(sel_old)
        adv_sel = torch.stack(sel_adv)

        loss_policy_core = torch.mean(self.policy_loss(new_sel, old_sel, adv_sel))
        # Compute KLD as mean difference in log probabilities
        if len(kld_terms) > 0:
            kld = torch.tensor(kld_terms, device=new_probs.device).mean()
        else:
            kld = torch.tensor(0.0, device=new_probs.device)

        # Normalized entropy using combined batch index (same as causal path)
        _ent_parts = []
        if has_a and idx_a is not None:
            _ent_parts.append(idx_a)
        if has_p and idx_p is not None:
            _ent_parts.append(idx_p)
        _ent_idx = torch.cat(_ent_parts, dim=0) if _ent_parts else idx_a
        ent = _normalized_entropy(new_probs, new_logpis, _ent_idx)

        loss_policy = loss_policy_core - self.ent_bonus * ent

        # Policy backward + step
        loss_policy.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.policy_optimizer.step()

        # Return (policy_loss, kld, ent, policy_core_loss)
        return float(loss_policy.item()), kld.item(), ent.item(), float(loss_policy_core.item())


def pg_surrogate_loss(new_logps, old_logps, advantages):
    """Return loss with gradient for policy gradient.

    Parameters
    ----------
    new_logps : Tensor (batch_dim,)
        The output of the current model for the chosen action.
    old_logps : Tensor (batch_dim,)
        The previous logged probability of the chosen action.
    advantages : Tensor (batch_dim,)
        The computed advantages.

    Returns
    -------
    loss : Tensor (batch_dim,)
        The loss for each interaction.

    """
    return -new_logps * advantages


class PGAgent(Agent):
    """A policy gradient agent.

    Parameters
    ----------
    policy_network : network
        The network for the policy model.

    """

    def __init__(self, policy_network, **kwargs):
        super().__init__(policy_network, **kwargs)
        self.policy_loss = pg_surrogate_loss


# ============================================================================
# PPO LOSS CLASSES (FULLY PICKLABLE - no closures, just callable classes)
# ============================================================================

class PPOClipLoss:
    """Clipped PPO loss (picklable callable class).

    Parameters
    ----------
    eps : float
        The clip ratio.
    """

    def __init__(self, eps=0.2):
        self.eps = eps

    def __call__(self, new_logps, old_logps, advantages):
        """Compute clipped PPO loss.

        Parameters
        ----------
        new_logps : Tensor (batch_dim,)
            The output of the current model for the chosen action.
        old_logps : Tensor (batch_dim,)
            The previous logged probability for the chosen action.
        advantages : Tensor (batch_dim,)
            The computed advantages.

        Returns
        -------
        loss : Tensor (batch_dim,)
            The loss for each interaction.
        """
        ratio = torch.exp(new_logps - old_logps)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages
        try:
            ret_loss = -torch.min(surr1, surr2)
        except Exception as e:
            print("Invalid loss detected.")
            ret_loss = -torch.min(surr1, surr2)
        return ret_loss


class PPOPenaltyLoss:
    """Penalty PPO loss (picklable callable class).

    Parameters
    ----------
    c : float
        The fixed KLD weight.
    """

    def __init__(self, c=0.01):
        self.c = c

    def __call__(self, new_logps, old_logps, advantages):
        """Compute penalty PPO loss.

        Parameters
        ----------
        new_logps : Tensor (batch_dim,)
            The output of the current model for the chosen action.
        old_logps : Tensor (batch_dim,)
            The previous logged probability for the chosen action.
        advantages : Tensor (batch_dim,)
            The computed advantages.

        Returns
        -------
        loss : Tensor (batch_dim,)
            The loss for each interaction.
        """
        return -(torch.exp(new_logps - old_logps) * advantages - self.c * (old_logps - new_logps))


class PPOAgent(Agent):
    """Proximal Policy Optimization agent.

    Parameters
    ----------
    policy_network : network
        The network for the policy model.
    method : {'clip', 'penalty'}
        The loss type for PPO.
    eps : float
        The clip ratio if using 'clip'.
    c : float
        The fixed KLD weight if using 'penalty'.

    """

    def __init__(self, policy_network, method='clip', eps=0.2, c=0.01, rudder_config=None, **kwargs):
        super().__init__(policy_network, **kwargs)
        self.method = method
        self.eps = eps
        self.c = c
        self.rudder_config = rudder_config or {}
        self.rudder_agent = None

        # Initialize RUDDER if enabled
        if self.rudder_config.get('enabled', False):
            try:
                from gympn.rudder import RUDDERAgent
                self.rudder_agent = RUDDERAgent(
                    state_dim=self.rudder_config.get('state_dim', 128),
                    hidden_dim=self.rudder_config.get('hidden_dim', 256),
                    learning_rate=self.rudder_config.get('learning_rate', 1e-3),
                    device=self.rudder_config.get('device', 'cpu'),
                    training_frequency=self.rudder_config.get('training_frequency', 1),
                    redistribution_method=self.rudder_config.get('redistribution_method', 'contribution')
                )
                get_logger().info(
                    f"[RUDDER] agent initialized with state_dim={self.rudder_config.get('state_dim', 128)}")
            except ImportError:
                get_logger().warning("[RUDDER] module not available, skipping initialization")
                self.rudder_agent = None
        # Fixed node-type order for the marking-vector featurization
        # (must match state_dim; threaded by train.make_agent from metadata).
        self._rudder_node_types = self.rudder_config.get('node_types', [])

        # Instantiate picklable loss classes
        if method == 'clip':
            self.policy_loss = PPOClipLoss(eps=eps)
        elif method == 'penalty':
            self.policy_loss = PPOPenaltyLoss(c=c)
        else:
            raise ValueError(f"Unknown PPO method: {method}")


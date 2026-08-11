"""Configuration for the causal-RL stability suite.

The suite tests, across the 8 paper_examples environments (a 2x4 grid of
{sequence, parallel, loop, exclusive-choice} x {joint, disjoint}):

    H1 (primary): causal TD(0) is MORE STABLE at convergence than plain PPO
                  (holds the optimum, less post-convergence drift).
    H2: causal is >= plain PPO in final performance (smaller gap-to-optimal).

Each environment ships a `perfect_heuristic` (the intended optimum); together
with a Random baseline this gives a per-env normalization
    normalized = (policy - random) / (heuristic - random)
so 0 = random, 1 = optimal, and results aggregate across environments.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# All 8 environments (keys must match envs.ENV_BUILDERS).
ALL_ENVS: List[str] = [
    "a_sequence_joint",
    "b_sequence_disjoint",
    "c_parallel_joint",
    "d_parallel_disjoint",
    "e_loop_joint",
    "f_loop_disjoint",
    "g_exclusive_choice_joint",
    "h_exclusive_choice_disjoint",
]

# Methods under comparison (keys consumed by run_suite._make_args).
# "lrq"    = Lineage-Restricted Q (the trace-based causal credit scheme).
# "mc_q"   = the lineage ABLATION of lrq: same Q-sample estimator, same SMDP
#            discount, same consumption — but every decision sees the FULL
#            return-to-go instead of only its lineage. The lrq-vs-mc_q gap
#            prices the lineage itself; the mc_q-vs-ppo gap prices the
#            estimator form (MC + wall-clock discount vs bootstrapped GAE).
# "rudder" = RUDDER baseline: BiLSTM return predictor over marking vectors,
#            'contribution' redistribution (return-conserving), consumed
#            through the ordinary GAE path — the learned-decomposition
#            competitor to LRQ's model-given lineage.
# "lva"    = Lineage Value Auxiliary: identical policy path to lcv0 (plain
#            SMDP-GAE PPO, no CV, no credit advantages) but the critic gets a
#            second head on its shared encoder regressed on the per-decision
#            lrq2 lineage credit (weight causal_aux_coef). The lineage enters
#            as an auxiliary REPRESENTATION task only — bias cannot reach the
#            policy gradient (floor by construction), yet the critic consumes
#            the full per-decision credit vector instead of LCV's one
#            epoch-level scalar. [lva - lcv0] isolates the aux task's value.
# "lcv0"   = LCV's exact c_hat=0 limiting case: plain SMDP-GAE PPO (STANDARD,
#            non-causal_rl path with the per-sojourn discount e^{-beta*tau}
#            switched on via smdp_discount=True), no CV/lineage term at all.
#            Isolates the time-discount choice from the CV term: [lcv-lcv0] =
#            the CV alone, [lcv0-ppo_clip] = the discount alone. Not in
#            ALL_METHODS by default — add explicitly where needed (see
#            PAPER_PLAN_LCV.md §5 X0/X1/X6).
# "ccf"    = Causal-Component-Factored return-to-go: union-find the REALIZED
#            lineage of each reward into components, credit each decision its
#            component's full return-to-go. Unbiased ONLY when a component's
#            membership doesn't depend on the action taken (fails at AND-joins/
#            shared-resource handoffs — see assembly_probe.py M1/M2, proven in
#            EJOR_PROPOSITIONS.md) — kept for the ablation, not recommended.
# "s_ccf"  = the fix: STATIC (topology-only, action-invariant) component
#            partition instead of ccf's realized one — unbiased BY
#            CONSTRUCTION (assumption A1 holds automatically), proven +
#            variance-reduced by a provable factor on genuinely decomposable
#            envs (N-copies, multi-site: beats PPO decisively, p=.005), honest
#            NULL (reduces exactly to mc_q/PPO) on single-component envs like
#            s1/the grid. Conservative on some motifs (keeps action-independent
#            reward a finer estimator could factor out — see M4 in
#            assembly_probe.py) by design, not a bug — see CCF_EXPLAINED.md.
#            THE MAIN RESULT of this whole causal-RL arc; every other scheme
#            here is either an ablation of it or a documented attempt to do
#            better on its honest null that didn't generalize.
ALL_METHODS: List[str] = ["ppo_clip", "lrq", "rudder", "mc_q"]


@dataclass
class SuiteConfig:
    # --- experimental grid ---
    envs: List[str] = field(default_factory=lambda: list(ALL_ENVS))
    methods: List[str] = field(default_factory=lambda: list(ALL_METHODS))
    seeds: int = 10

    # --- per-run training budget ---
    epochs: int = 30
    episodes_per_epoch: int = 20
    batch_size: int = 32
    test_freq: int = 5          # deterministic (greedy) eval every N epochs
    test_episodes: int = 10     # episodes averaged per eval point (no-op on
                                # deterministic envs; matters for stochastic ones)
    # Common random numbers for the greedy eval: with a base seed set, eval
    # episode i always runs scenario eval_seed+i, so every method and every
    # epoch is scored on the SAME fixed scenario set and the arms differ only
    # by policy. None = historical behaviour (fresh draws off the live RNG),
    # which on s1 costs +-0.231 SD per 20-episode point, +-0.327 on a paired
    # difference, and inflates greedy_drift by ~0.40 even for a flat policy.
    # NOTE results produced with this set are NOT comparable point-for-point
    # with cells on disk that were produced without it.
    eval_seed: Optional[int] = None
    length: int = 10            # simulation horizon (timesteps)
    # Per-env horizon overrides (the stochastic tier runs longer horizons);
    # envs not listed fall back to `length`.
    env_length: Dict[str, int] = field(default_factory=dict)

    # --- env config (same for both methods so the comparison is fair) ---
    allow_postpone: bool = True

    # --- baselines (evaluated on the canonical no-postpone env, like the
    #     original paper examples, to anchor the normalization scale) ---
    baseline_episodes: int = 20

    # --- shared hyperparameters ---
    policy_lr: float = 3e-4
    value_lr: float = 3e-4
    policy_updates: int = 3
    value_updates: int = 4
    # ^ PPO update epochs per training epoch. These dominate cost: profiling a
    # heavy cell showed ~83% of wall time is the optimizer re-running fwd+bwd over
    # the collected graphs (value bwd alone ~49%). 10 value epochs was unusually
    # high; 4/3 keeps learning while cutting total time ~1.7-2x.
    ent_bonus: float = 0.01
    ppo_eps: float = 0.2
    # Early-stopping limit on the mean per-state KL(old||new) — the exact,
    # variable-action-set-valid metric (agents.py, fixed 2026-07-09; before
    # that the logged "KLD" was a signed chosen-action log-ratio and NO suite
    # run ever set the limit). 0.15 is a loose brake: healthy inner epochs
    # measure <~0.05, while the catastrophic updates behind post-convergence
    # collapse show KL >= 1 (a near-certain action dropping to p~0.01 alone
    # contributes ~4.6). None = disabled (pre-fix behaviour).
    policy_kld_limit: float = 0.15
    gam: float = 0.99
    lam: float = 0.95
    # SMDP time-discount rate for LRQ's Q-samples: postpone's Q sits e^{-beta*tau}
    # below acting on the same lineage. 0.2 left too thin a margin — 2/5 env-f
    # seeds slipped into the postpone attractor post-convergence; 0.5 rescued
    # both (see f_lrq_vs_ppo results, 2026-07-08).
    causal_beta: float = 0.5
    # LRQ foreclosure hedge: A = (1-mu)*A_LRQ + mu*A_GAE on raw rewards.
    causal_mu: float = 0.0
    # Potential-based reward shaping (gympn/potential.py; Ng, Harada & Russell
    # 1999): F=e^{-causal_beta*tau}*Phi(s')-Phi(s), Phi from pure PN topology.
    # 0.0 (default) = disabled. Provably cannot change the optimal policy for
    # ANY value; layers on top of any method (not gated on `method` in
    # run_suite._make_args).
    phi_coef: float = 0.0
    phi_decay: float = 0.9
    # None (default) = uncapped Phi; a small int caps each place's own
    # token-count contribution (min(count, phi_cap)) -- see potential.py's
    # topology_potential docstring (exogenous-arrival queue-depth variance).
    phi_cap: Optional[float] = None
    # Structural (conflict-graph-derived) INPUT features to the actor/critic
    # (gympn/simulator.py's GymProblem.use_structural_features): per action-
    # type, (in_conflict, conflict_degree) appended to its node feature.
    # False (default) = byte-identical observation shape to before. Unlike
    # phi_coef/causal_mu/etc this is plain network input, not a credit or
    # reward term -- no bias-variance or GAE-bootstrap concerns apply.
    use_structural_features: bool = False
    # Actor architecture fix (gympn/networks.py's HeteroActor global_context):
    # concatenates the SAME pooled action/postpone context HeteroCritic
    # already builds onto each action's own embedding before decoding its
    # logit. False (default) = byte-identical, matches the pre-existing
    # actor exactly. Motivation: HeteroActor decodes purely from a node's
    # own final embedding, and get_graph_observation's edges are directed
    # strictly along token flow (no reverse edges anywhere in real use), so
    # an action can be provably blind (proven via exact-invariance test, not
    # just empirically neutral) to state outside its forward-reachable
    # neighborhood within net_num_layers hops -- see
    # suite/_test_actor_global_context.py.
    actor_global_context: bool = False
    # LVA: weight of the critic's auxiliary lineage-credit regression
    # (value_loss + coef * MSE(V_aux, lrq2 credit)); only read by method "lva".
    causal_aux_coef: float = 0.5
    # RUDDER baseline knobs. hidden_dim 64 matches the HGT nets' scale (the
    # marking-vector inputs are ~11-13 dim); LSTM trains every epoch.
    rudder_hidden_dim: int = 64
    rudder_training_freq: int = 1
    rudder_redistribution_method: str = "contribution"

    # --- network size (HGT actor+critic) ---
    # None => use the historical defaults (hidden=256, layers=3, residual on).
    # These are the dominant wall-clock knobs: profiling showed ~all training time
    # is HGT fwd/bwd. On the small paper-suite graphs, hidden=128/L2/no-residual is
    # ~2x faster and hidden=64/L2/no-residual ~4x faster, with ample capacity for
    # the simple assignment optima. Applied identically to both methods, so the
    # PPO-vs-REC comparison stays fair. Leave as None to reproduce prior runs.
    net_hidden_size: int = None
    net_num_layers: int = None
    net_residual: bool = None

    # --- io ---
    # Fresh dir for the LRQ-era suite: results under the old "suite_results"
    # predate the scheme removal AND the smaller default nets (256 -> 64), so
    # the resume logic must never mix them in.
    output_dir: Path = Path("suite_results_lrq")

    def cell_id(self, env: str, method: str, seed: int) -> str:
        return f"{env}__{method}__s{seed}"


# A fast smoke configuration to validate the pipeline end-to-end before
# committing to the full ~50h run.
def smoke_config() -> SuiteConfig:
    return SuiteConfig(
        envs=["d_parallel_disjoint"],
        methods=["ppo_clip", "lrq", "rudder", "mc_q"],
        seeds=1,
        epochs=3,
        episodes_per_epoch=4,
        test_freq=1,
        baseline_episodes=5,
        output_dir=Path("suite_results_smoke"),
    )


def paper_config() -> SuiteConfig:
    """Paper-final protocol: 10 seeds, eval every 2 epochs (finer convergence-
    epoch resolution; eval is exact on the deterministic suite envs so extra
    points cost only sim time, not statistical validity). Fresh output dir —
    curves have a different eval grid than suite_results_lrq and must never be
    pooled with it."""
    return SuiteConfig(
        seeds=10,
        test_freq=2,
        output_dir=Path("suite_results_paper"),
    )


def stoch_config() -> SuiteConfig:
    """E5: the stochastic/scaled tier — random arrivals, task types and
    service times, more entities, longer horizons. Unsaturates the benchmark
    (the deterministic a-h grid can no longer discriminate above LRQ) and
    tests the variance-reduction claim where variance actually exists.
    Heuristic anchors are strong myopic rules, not optima: normalized > 1.0
    means the policy beat the anchor. Eval averages 20 stochastic episodes
    per point; baselines average 40.

    Methods include both LRQ variants: v1 ("lrq", postpone in the lineage —
    collapses on loaded reward-dense envs via postpone aggregation) and v2
    ("lrq2", consistent-support fix). Keeping v1 documents the failure mode;
    v2 is the method going forward."""
    return SuiteConfig(
        envs=["s1_stoch_sequence", "s2_stoch_scaled", "s3_stoch_mixed"],
        methods=["ppo_clip", "lrq", "lrq2", "rudder", "mc_q"],
        seeds=10,
        epochs=30,
        test_freq=2,
        test_episodes=20,
        baseline_episodes=40,
        env_length={"s1_stoch_sequence": 20,
                    "s2_stoch_scaled": 30,
                    "s3_stoch_mixed": 25},
        output_dir=Path("suite_results_stoch"),
    )


def seeds5_config() -> SuiteConfig:
    """Full 8-env grid at 5 seeds (~half the canonical budget). Resumable: a
    later 10-seed run in the same output_dir only trains the missing seeds."""
    return SuiteConfig(seeds=5)


def smoke8_config() -> SuiteConfig:
    """Tiny budget but ALL 8 envs — de-risks per-topology bugs before the full run."""
    return SuiteConfig(
        envs=list(ALL_ENVS),
        methods=["ppo_clip", "lrq", "rudder", "mc_q"],
        seeds=1,
        epochs=2,
        episodes_per_epoch=4,
        test_freq=1,
        baseline_episodes=5,
        output_dir=Path("suite_results_smoke8"),
    )
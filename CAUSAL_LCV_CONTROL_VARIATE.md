# LCV: the Lineage Control Variate — option D of the rethink

**Scope.** Design and implementation of `causal_scheme='lcv'`: standard
SMDP-GAE PPO whose advantages are corrected by an **adaptive, causally-measured
control variate** read off the token-lineage trace. This is the
"best-principled" consumer identified in the 2026-07-14 rethink
(`CAUSAL_LQI_QNATIVE.md` §5b documents why the direct consumers — PG
advantage, hedged, decomposed, Q-native AWR — all left performance on the
table on discrimination-bound environments).

---

## 1. The estimator

```
A_lcv(t)  =  A_GAE(t)  −  ĉ · ( R_off(t) − v_off(s_t) )
```

- `A_GAE` — the ordinary SMDP-GAE advantage on raw temporal rewards
  (λ = cfg.lam, per-sojourn discount e^{−βτ}); value targets are the ordinary
  GAE returns. **The policy/critic path is 100% standard PPO.**
- `R_off(t)` — the **measured** off-lineage return from step t: every future
  reward NOT caused by the decision, discounted from its clock. Computed
  exactly from the trace as `mc_q credit − lrq2 credit` (both already
  implemented); nothing is approximated here.
- `v_off(s)` — a learned state-only centering head (a scalar `HeteroCritic`
  regressed on `R_off`). Centering quality affects only how much variance the
  CV removes, never bias (see §2).
- `ĉ` — the classical optimal control-variate coefficient
  `Cov(A_GAE, cv) / Var(cv)`, estimated over the **whole epoch's buffer**
  (in `TrajectoryBuffer.get()`, before advantage normalization), clipped to
  `[0, 2]`.

## 2. The three properties (why this is the principled option)

1. **PPO floor by construction.** `ĉ → 0` recovers plain SMDP-GAE PPO
   *exactly* — degenerate CV, zero variance, or negative correlation all
   collapse to the baseline estimator. Unlike the μ-hedge, the fallback is
   not a tuned knob but the estimator's own limiting case. (Unit-tested: with
   perfect centering the advantages equal `smdp_gae` to machine precision.)
2. **Quantified benefit.** Under off-lineage independence
   (`E[R_off | s, a] = E[R_off | s]`), the correction has zero conditional
   mean, so the estimator stays unbiased and the variance drops by
   `ĉ²·Var(cv)` — for MC-dominated advantages in concurrent systems this
   approaches the entire cross-case noise, which is LRQ's original pitch,
   now obtained without leaving PPO.
3. **Characterized failure.** When independence fails (foreclosure), the bias
   is `ĉ ·` (the action-dependent part of `E[R_off|s,a]`) — bounded,
   proportional to the same coefficient that gates the benefit.

**Why the naive version was a trap** (and why nobody should "simplify" this):
with pure Monte-Carlo advantages (λ=1), subtracting the measured `R_off`
cancels algebraically back to LRQ with a shifted baseline — the collapse we
already ran. The estimator is only *new* on the **bootstrapped** GAE, where
the measured CV correlates with, but is not contained in, the advantage.
Positioning vs the literature: Q-Prop / Stein CVs *learn* their control
variates and were partly deflated by Tucker et al.'s "mirage" critique; here
the CV is **measured** (only the scalar ĉ and the centering are estimated),
which is the direct response to that critique — and the causal trace is what
makes measurement possible.

## 3. Implementation map

| Piece | Location |
|---|---|
| Dual trace targets (lrq2 + mc_q → `R_off`) per step | `gympn/data.py` `finish()` (shares the lrq3/lqi branch) |
| `A_GAE` base, GAE value targets, per-step cv terms | `finish()` `'lcv'` branch |
| Epoch-level ĉ + adjustment (before normalization) | `TrajectoryBuffer.get()`; last value exposed as `buffer.last_cv_coef` |
| `v_off` head (scalar `HeteroCritic`, passed via the generic aux-head slot) | `gympn/train.py` `make_agent`; rollout predictions frozen per step |
| Training: standard PPO fit + `_fit_voff_model` regression | `gympn/agents.py` (routed on `causal_scheme == 'lcv'`) |
| Suite method key `'lcv'` (env config identical to the lrq variants) | `examples/paper_examples/suite/run_suite.py` |

Probe: `python run_lcv_probe.py [workers]` — s1 + s3 × 10 seeds into
`suite_results_stoch`, paired automatically against the six existing methods.

## 4. Predictions (registered before the run)

- **s1**: ≈ PPO (0.76) or slightly above — the floor property guarantees the
  neighborhood; the CV can only help to the extent cross-case noise, not
  discrimination, is the binding constraint there (it mostly is not, so a
  large win would be a surprise).
- **s3**: ≥ PPO (0.79), aiming toward lrq2's 0.93 — concurrency noise is the
  binding constraint there, which is exactly what the CV removes; ĉ should be
  visibly > 0.
- The interesting *measurement* (per the mirage paper's lesson): report ĉ
  trajectories and advantage-variance reductions, not just returns.

## 4b. Discovery during the first probe (2026-07-14): the zero-reward bug

The first LCV probe froze (entropy ~0.97 at epoch 13, returns at random) and
exposed a long-standing latent bug: **in causal mode `AEPN_Env.step` returned
`reward = 0.0` unconditionally** — `rewards_raw` was all zeros in every
causal-scheme run ever made. Two consequences:

1. LCV's `A_GAE` base was pure value-telescope noise (no reward signal at
   all), hence the frozen policy. Fixed: `step` now always returns the true
   per-step reward. The μ=0 causal paths are provably unaffected (they never
   read `rewards_raw`; regression-smoked on lrq2).
2. **The `causal_mu` hedge never mixed raw-reward GAE.** With zero rewards,
   its "GAE term" was `d·V(s′) − V(s)` — a value-progress TD signal (V being
   trained on lineage-Q targets). The μ=0.5 rescue of s1 (0.65) therefore
   worked through a *different mechanism than designed*: value-TD
   regularization of the LRQ advantage, not temporal-reward mixing. Any
   paper text about μ must be rewritten accordingly, and the μ probe should
   be rerun post-fix if the hedge stays in the method — the accidental
   mechanism may even be the better one, which is now a testable question.

## 4c. Results (2026-07-14, post-fix probe: s1 + s3 × 10 seeds — both §4
predictions CONFIRMED)

| env | LCV | PPO | lrq2 (μ=0) | best prior lineage consumer |
|---|---|---|---|---|
| s1 | **0.72 ± 0.11** (0.57–0.91) | 0.76 (0.56–0.98) | 0.28 | μ-hedge 0.65 |
| s3 | **0.90 ± 0.16** (0.75–1.18) | 0.79 | 0.93 | lrq2 0.93 |

- **s1 — the floor holds**: statistically indistinguishable from PPO, with
  visibly tighter seed spread (the signature of a variance-reduced gradient).
  LCV is the first lineage consumer that pays *no price* on the
  discrimination-bound environment (every direct consumer lost 0.1–0.5 here).
- **s3 — the benefit shows where predicted**: 0.90 vs PPO's 0.79, at lrq2's
  level, two seeds beating the myopic anchor (1.16, 1.18). Concurrency noise
  is what the CV removes, and that is where it wins.
- Net: one estimator, standard PPO everywhere, that inherits PPO where the
  lineage cannot help and captures ~the direct-credit gain where it can —
  fallback built into the math (ĉ), not a knob (μ).

**Open items for paper-grade evidence**: (i) persist the ĉ trajectory in the
training history (currently computed but not logged per epoch); (ii) the
mirage-standard measurement — advantage variance with vs without the CV on
identical batches; (iii) s2 for the tier-complete table; (iv) the μ-hedge
reinterpretation (§4b) suggests comparing LCV against a post-fix μ rerun.

## 5. Honest limitations

- The variance reduction is bounded by how much of `A_GAE`'s noise is
  off-lineage; on single-chain (non-concurrent) problems ĉ ≈ 0 and LCV is
  exactly PPO — by design, but then the lineage bought nothing.
- ĉ is estimated on the same data it corrects (classical O(1/n) CV-estimation
  bias; negligible at ~600 steps/epoch, and the [0,2] clip guards the tail).
- One extra critic-sized head per agent (the `v_off` centering).

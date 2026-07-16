# LQI: Lineage-Q Iteration — the Q-native consumer of causal traces

**Scope.** Design, implementation and rationale of `causal_scheme='lqi'`, the
fourth and most structurally different consumer of the token-lineage credit
information: fitted, lineage-decomposed Q-functions improved by
advantage-weighted regression (AWR/MPO-family), replacing the PPO policy
gradient entirely. Companion to `CAUSAL_LRQ_PROPOSAL.md` (the LRQ estimator
family) and `PAPER_PLAN_LRQ.md` (experiment ledger).

---

## 1. Why leave the policy gradient at all

Three findings from the stochastic-tier campaign (2026-07-12/14) converge on
one diagnosis — the *information* in the lineage is sound, but delivering it
as a PPO advantage mis-couples it to the optimizer's dynamics:

1. **The credits already know the answer the training loop fails to install.**
   On s1, the credit-induced action ordering, executed as a scripted policy,
   scores 14.80 — full anchor quality, better than every trained method — yet
   trained LRQ variants reach only 0.28–0.34 normalized. The signal is right;
   the extraction is lossy.
2. **PPO's clip + multi-epoch reuse amplifies small-but-consistent
   advantages into ε-sized policy moves.** GAE advantages carry sign-flipping
   trajectory noise that partially cancels across inner epochs; LRQ advantages
   are nearly deterministic per action category, so every inner epoch pushes
   the ratio to the clip boundary regardless of the advantage's magnitude —
   a 0.06-gap discrimination moves as fast as a 0.5-gap one. Measured
   signature: LRQ policies commit at entropy ~0.14 while PPO sits at ~0.48,
   locking in half-learned rankings (s1) — and, historically, producing the
   post-convergence collapses the KL brake had to contain.
3. **LRQ-v3's cold-start race.** The exact decomposition
   `A = c_lineage + q_off_θ(s,a) − V` is the right *object*, but injecting an
   untrained `q_off` into PG advantages at full weight corrupts learning
   until the head converges: v3's seeds split bimodally (its best s1 seeds,
   0.87–0.92, beat everything ever run there; its worst sit below random, and
   it degrades s3, which pure lineage credit had already won).

LQI removes the shared root cause: **there is no policy-gradient advantage
anywhere.** The lineage information becomes what it naturally is — a
Q-function — and the policy is improved against it with a method whose step
size is proportionate and whose cold-start is benign.

## 2. The method

Per training epoch, on the freshly collected on-policy batch:

**(1) Fit the decomposed Q from trace targets** (both targets are computed
exactly by the existing machinery; nothing is learned that the trace can
provide):

```
q_lin_θ(s, a)  ←  regress on  c_lrq2(a_t)              (lineage Q-sample;
                                                        sharp, low-variance)
q_off_φ(s, a)  ←  regress on  c_mcq(a_t) − c_lrq2(a_t)  (off-lineage return;
                                                        diffuse, learned)
q(s, a) = q_lin_θ(s, a) + q_off_φ(s, a)
```

Both heads are `HeteroQOff` networks: one raw scalar per action node
(a_transition then postpone), in the policy's node order. The split matters:
the lineage part regresses on causally-filtered, nearly-deterministic
targets, while the off-lineage head absorbs the full cross-case noise but
only needs coarse accuracy (it carries foreclosure and availability effects,
not fine per-case values).

**(2) Improve the policy by advantage-weighted regression** (AWR; the
weighted-BC form of KL-regularized policy iteration):

```
A(s, a_t)  =  q(s, a_t) − mean_{a' ∈ available(s)} q(s, a')
A_std      =  per-batch standardized A
w          =  min( exp(A_std / TAU), W_MAX )          TAU = 1.0, W_MAX = 20
L_policy   =  − E[ w · log π(a_t | s) ]  −  ent_bonus · H(π)
```

Design choices and what they buy:

- **Per-state centering over AVAILABLE actions** (the COMA/dueling-style
  baseline): the heads score every enabled node, so the baseline is
  availability-aware — a scalar V(s) cannot distinguish "good state" from
  "good options", which is exactly what variable action sets need.
- **Standardize-then-exponentiate** makes TAU dimensionless (the AWAC
  convention), so no per-environment temperature tuning; W_MAX guards the
  exponential tail.
- **Proportionate moves**: exp-weighting moves the policy in proportion to
  the (standardized) advantage gap — the anti-thesis of clip saturation. Fine
  discriminations get gentle, persistent pressure instead of full-ε jumps.
- **Benign cold start**: an untrained q gives near-constant A → weights ≈ 1 →
  the update degenerates to behavior cloning of the current policy. Harmless
  warm-up, in contrast to v3's corrupted-gradient cold start. (Confirmed
  empirically: the s1 smoke reaches greedy ~13.3 in the FIRST epoch.)
- **Postpone needs no special case**: its q_lin target is 0 (lrq2 semantics)
  and its q_off target is the full delayed continuation — the decomposition
  routes it automatically to the learned head.
- The per-state exact KL(π_old‖π_new) monitor and `kld_limit` early stop are
  kept unchanged (they are estimator-agnostic).

**What is given up:** the PPO trust region (replaced by the implicit KL
regularization of weighted BC plus the retained KL early stop), and the pure
Monte-Carlo sharpness of LRQ's per-sample credits at decision time — LQI's
policy sees the lineage only through the fitted `q_lin`, so function
approximation error is the new residual. The bet, supported by the s1
scripted-ordering fact, is that a *regression* over thousands of sharp
targets extracts the ordering more reliably than per-sample policy-gradient
nudges racing an entropy schedule.

## 3. Relation to the literature

AWR (Peng et al. 2019) / AWAC (Nair et al. 2020) supply the weighted-BC
update; MPO (Abdolmaleki et al. 2018) is the same E/M structure with a dual
for the temperature (we keep TAU fixed on standardized advantages instead).
The per-state mean baseline is the dueling/COMA centering. The novelty is
none of these pieces — it is **where Q comes from**: an exact causal
decomposition read off the executable model's token lineage, with the
sharp/diffuse split deciding which component is sampled and which is learned.

## 4. Implementation map

| Piece | Location |
|---|---|
| Q-head architecture (`HeteroQOff`, shared by v3 and LQI) | `gympn/networks.py` |
| Scheme `'lqi'`: dual trace targets (lrq2 + mc_q) per step | `gympn/data.py` `finish()` (shares the lrq3 branch; buffer `advantage` is a placeholder — LQI recomputes advantages from fresh heads at training time) |
| Head construction (both heads for `'lqi'`) | `gympn/train.py` `make_agent` |
| Training path: `_fit_lqi_models` → `_fit_qhead` ×2 → AWR epochs (`_fit_lqi_policy_step`) | `gympn/agents.py` (routed before the causal-PG path when `causal_scheme == 'lqi'`) |
| Suite method key `'lqi'` (env config identical to the lrq variants: causal trace + token-flow postpone) | `examples/paper_examples/suite/run_suite.py` |
| Defaults | `TAU = 1.0` (standardized A), `W_MAX = 20`, head LR = `value_lr`, head epochs = `value_updates`, heads sized like the policy net |

Run on the stochastic tier: `python run_lqi_probe.py [workers]`
(s1 + s3 × 10 seeds into `suite_results_stoch`, so `analyze.py` pairs it
against ppo_clip / lrq / lrq2 / rudder / mc_q / lrq3 automatically).

## 5. Scoreboard it must beat (s1 + s3, normalized greedy final)

| method | s1 | s3 |
|---|---|---|
| lrq2 (μ=0) | 0.28 | **0.93**, peak 1.28 |
| lrq2 (μ=0.5) | 0.65 | — |
| lrq3 | 0.34 (bimodal, best seeds 0.87–0.92) | −0.95 (cold-start damage) |
| ppo_clip | 0.76 | 0.79 |
| scripted credit ordering (no training) | **1.00** (14.8 raw) | — |
| **LQI target** | ≥ 0.76, approach 1.00 | ≥ 0.9 |

Success = LQI at PPO level or better on s1 (approaching the scripted-ordering
ceiling) while holding s3 — that would answer "is the advantage route the
right way to use the lineage?" with a measured *no, Q-native is better*, and
make LQI the method the paper leads with. Failure modes to watch: head
approximation error on s3's chain values (q_lin must separate 8-value chains
from 2-value shorts), and AWR's slower late-stage sharpening (weights
saturate at W_MAX for clear decisions).

## 5b. Partial results (2026-07-14, probe stopped at 5/20 cells — s1 only)

Five s1 seeds: normalized finals ≈ **0.31 mean** (raw 10.75–11.7) — at
lrq2-μ=0 level, well below the μ-hedge (0.65) and PPO (0.76). The curve
shape, not the endpoint, is the diagnostic:

- **The warm-up works exactly as designed**: greedy ~12 (≈0.45) within 1–2
  epochs, vs ~10+ epochs for the PG variants; seed s0 peaks at 13.8 (≈0.80)
  mid-run. The benign cold start is real.
- **The sharpening never comes**: every seed then plateaus/wanders at
  ~11–12 while entropy still declines (0.1–0.3) — committing to a mediocre
  policy. Two suspected mechanisms, both fixable in principle:
  (a) per-batch standardization re-inflates whatever advantage variation
  remains once the policy is decent — which late in training is mostly
  q-head noise, so AWR keeps pushing on noise (a fixed-τ pathology; the MPO
  dual or a τ decay is the principled fix);
  (b) the uniform mean-over-available baseline is dominated by bad options,
  so nearly every taken action gets w > 1 — weight crowding weakens
  discrimination (π-weighted mean is the fix).

**Interim conclusion for the research line**: across four delivery
mechanisms on s1 — PG advantage (0.28), hedged PG (0.65), decomposed PG
(0.34, bimodal), Q-native AWR (0.31, plateaus) — none matches plain PPO
(0.76), while the lineage's own scripted ordering scores 1.00. The
extraction gap survives the change of optimizer family: fast-and-clean
early learning is easy to buy with the lineage; fine late-stage
discrimination under noise is what every consumer so far leaves on the
table. The paper's practical recommendation remains v2+μ; LQI's warm-up
speed and its two identified pathologies are the concrete starting point
for the follow-up work.

## 6. Honest limitations

- Two extra networks per agent (~2× the critic cost per epoch); rollout cost
  unchanged.
- TAU/W_MAX are fixed conventions, not tuned — a dual-solved temperature
  (full MPO) is the principled upgrade if sensitivity appears.
- The off-lineage head still carries the foreclosure signal only as well as
  it generalizes; the μ-hedge's GAE term is gone, so LQI's foreclosure
  handling rests entirely on `q_off`.
- On-policy AWR discards data each epoch like PPO does; the natural
  extension (replay + off-policy AWAC) is unexplored here.
